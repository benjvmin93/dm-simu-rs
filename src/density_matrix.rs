use core::fmt;
use std::collections::HashSet;

use num_complex::Complex;
use rayon::collections;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tensor::Tensor;

use crate::operators::{OneQubitOp, Operator, TwoQubitsOp};
use crate::tensor;
use crate::tools::{are_elements_unique, bitwise_bin_vec_to_int, bitwise_int_to_bin_vec, complex_approx_eq};

#[pyo3::pyclass]
#[derive(Debug, Copy, Clone)]
pub enum State {
    ZERO,
    ONE,
    PLUS,
    MINUS,
}

// 1D representation of a size * size density matrix.
pub struct DensityMatrix {
    pub data: Tensor<Complex<f64>>,
    pub size: usize, // 2 ** nqubits
    pub nqubits: usize,
}

impl fmt::Display for DensityMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print(f)
    }
}

impl DensityMatrix {
    // By default initialize in |0>.
    pub fn new(nqubits: usize, initial_state: State) -> Self {
        let size = 1 << nqubits;
        let shape = 2 * nqubits;
        match initial_state {
            State::PLUS => {
                // Set density matrix to |+><+| \otimes n
                let mut dm = Self {
                    data: Tensor::from_vec(&vec![Complex::ONE; size * size], vec![2; shape]),
                    size,
                    nqubits,
                };
                dm.data.data = dm
                    .data
                    .data
                    .iter()
                    .map(|n| *n / Complex::new(size as f64, 0.))
                    .collect();
                dm
            }
            State::ZERO => {
                // Set density matrix to |0><0| \otimes n
                let mut dm = Self {
                    data: Tensor::from_vec(&vec![Complex::ZERO; size * size], vec![2; shape]),
                    size,
                    nqubits,
                };
                let indices = bitwise_int_to_bin_vec(0, shape);
                dm.data.set(&indices, Complex::ONE);
                dm
            }
            State::ONE => {
                let mut dm = Self {
                    data: Tensor::from_vec(&vec![Complex::ZERO; size * size], vec![2; shape]),
                    size,
                    nqubits,
                };
                // Generate the binary vector for |1>
                let indices = bitwise_int_to_bin_vec((size * size) - 1, shape);
                dm.data.set(&indices, Complex::ONE);
                dm
            }
            State::MINUS => {
                let mut dm = Self {
                    data: Tensor::from_vec(&vec![Complex::ZERO; size * size], vec![2; shape]),
                    size,
                    nqubits,
                };

                // Generate the density matrix for |-\rangle^{\otimes n}
                let factor = Complex::new(1.0 / (size as f64), 0.0); // Normalize by size (2^n)
                for i in 0..size {
                    for j in 0..size {
                        let sign = if (i ^ j).count_ones() % 2 == 0 {
                            Complex::ONE
                        } else {
                            -Complex::ONE
                        };
                        let value = factor * sign;
                        let indices = bitwise_int_to_bin_vec(i * size + j, shape);
                        dm.data.set(&indices, value);
                    }
                }
                println!("dm = {}", dm);
                dm
            }
        }
    }

    pub fn from_statevec(statevec: &[Complex<f64>]) -> Result<Self, &'static str> {
        let len = statevec.len();
        if !len.is_power_of_two() {
            return Err("The size of the statevec is not a power of two");
        }
        let nqubits = len.ilog2() as usize;
        let size = len;
        let mut data = Vec::with_capacity(size * size);

        for i in 0..size {
            for j in 0..size {
                data.push(statevec[i] * statevec[j].conj());
            }
        }
        Ok(DensityMatrix {
            data: Tensor::from_vec(&data, vec![2; 2 * nqubits]),
            size,
            nqubits,
        })
    }

    pub fn from_tensor(tensor: Tensor<Complex<f64>>) -> Result<Self, &'static str> {
        let size: usize = tensor.shape.iter().product();
        if !size.is_power_of_two() {
            return Err("Tensor size is not a power of two.");
        }

        let n = (size as f32).log2();
        if n % 2.0 != 0. {
            return Err("Tensor size is not valid. It should be of size 2^(2n) with n the number of qubits.");
        }
        let nqubits = (n / 2.) as usize;

        Ok(DensityMatrix {
            data: tensor,
            size: 2_i32.pow(nqubits as u32) as usize,
            nqubits,
        })
    }

    pub fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            write!(f, "[")?;
            for j in 0..self.size {
                let indices = bitwise_int_to_bin_vec(i * self.size + j, 2 * self.nqubits);
                write!(f, "{}", self.data.get(&indices))?;
                if j != self.size - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i == self.size - 1 {
                write!(f, "]")?;
            }
            writeln!(f, "")?;
        }
        writeln!(f, "\n")
    }

    // Access element at row i and column j
    pub fn get(&self, i: u8, j: u8) -> Complex<f64> {
        let idx = i * self.size as u8 + j;
        let bin_idx = bitwise_int_to_bin_vec(idx.into(), 2 * self.nqubits);
        *self.data.get(&bin_idx)
    }

    // Set element at row i and column j
    pub fn set(&mut self, i: u8, j: u8, value: Complex<f64>) {
        let size = self.size as u8;
        let indices = bitwise_int_to_bin_vec((i * size + j) as usize, 2 * self.nqubits);
        self.data.set(&indices, value);
    }

    pub fn expectation_single(
        &self,
        op: &Operator,
        index: usize,
    ) -> Result<Complex<f64>, &str> {
        if index >= self.nqubits {
            return Err("Index out of range");
        }
    
        let index_mask = 1 << (self.nqubits - index - 1);
    
        // Parallel computation of expectation
        let expectation = self.data.data
            .par_iter()
            .enumerate()
            .map(|(idx, rho_elt)| {
                let i = idx / self.size;
                let j = idx % self.size;
    
                if (i & !index_mask) == (j & !index_mask) {
                    let op_i = (i & index_mask) >> (self.nqubits - index - 1);
                    let op_j = (j & index_mask) >> (self.nqubits - index - 1);
    
                    rho_elt * op.data.get(&[op_i as u8, op_j as u8])
                } else {
                    Complex::ZERO
                }
            })
            .reduce(|| Complex::ZERO, |a, b| a + b); // Combine partial results
    
        Ok(expectation)
    }

    pub fn trace(&self) -> Complex<f64> {
        // Compute the sum over the diagonal elements.
        (0..self.size)
            .map(|i| self.get(i as u8, i as u8))
            .fold(Complex::ZERO, |acc, x| acc + x) // Fold ensures a starting value
    }

    pub fn normalize(&mut self) {
        let trace = self.trace();
        self.data.data = self
            .data
            .data
            .par_iter()
            .map(|&c| c / trace)
            .collect::<Vec<_>>();
    }

    pub fn evolve_single(&mut self, op: &Operator, index: usize) -> Result<(), String> {
        if index >= self.nqubits {
            return Err(format!(
                "Target qubit {} is not in the range [0-{}].",
                index, self.nqubits
            ));
        }
        if op.nqubits != 1 {
            return Err(format!("Passed operator is not a one qubit operator."));
        }

        let dim = 1 << self.nqubits;
        let position_bitshift = self.nqubits - index - 1;

        let mut new_dm: Vec<Complex<f64>> = (0..dim * dim)
            .map(|idx| {
                let i = idx / dim;
                let j = idx % dim;
                
                // Extract bit at position index
                let b_i = (i >> position_bitshift) & 1;
                let b_j = (j >> position_bitshift) & 1;
                
                // Get state without the target qubit contribution
                let bitmask = 1 << position_bitshift;
                // Target qubit is setted to 0.
                // Other qubits remain unchanged. 
                let i_base = i & !bitmask;
                let j_base = j & !bitmask;

                let mut sum = Complex::<f64>::ZERO;
                (0..4).for_each(|op_idx| {
                    let p = op_idx / 2;
                    let q = op_idx % 2;
                    // Reconstruct indices with the target contribution
                    // and according to the operator indices
                    let i_prime = i_base | (p << position_bitshift);
                    let j_prime = j_base | (q << position_bitshift);
                    let data_idx = i_prime * dim + j_prime;
                    if self.data.data[data_idx] == Complex::ZERO {
                        return;
                    }
                    
                    // println!("idx: {idx}, i: {i}, j: {j}, i_prime: {i_prime}, j_prime: {j_prime}");
                    // println!("op[b_i, p]: {}, self[{}, {}]: {}, op[b_j, q].conj(): {}\n",
                    //     op.data.data[b_i * 2 + p],
                    //     i_prime,
                    //     j_prime,
                    //     self.data.data[i_prime * dim + j_prime],
                    //     op.data.data[b_j * 2 + q].conj()
                    // );
                    sum += op.data.data[b_i * 2 + p]
                        * self.data.data[i_prime * dim + j_prime]
                        * op.data.data[b_j * 2 + q].conj();
                });
                sum
            }).collect();

        std::mem::swap(&mut self.data.data, &mut new_dm);
        Ok(())
    }

    pub fn evolve(&mut self, op: &Operator, indices: &[usize]) -> Result<(), String> {
        if !are_elements_unique(indices) {
            return Err("Target qubits must be unique.".to_string());
        }
        for &i in indices.iter() {
            if i >= self.nqubits {
                return Err(format!(
                    "Target qubit {} is not in the range [0-{}].",
                    i, self.nqubits
                ));
            }
        }

        let nqb_op = op.nqubits;
        let first_axe = (0..indices.len())
            .map(|i| nqb_op + i)
            .collect::<Vec<usize>>();
        let second_axe = indices;
        self.data = op
            .data
            .tensordot(&self.data, (&first_axe, &second_axe))
            .unwrap();

        let op_transconj = op.transconj();
        let first_axe = indices
            .iter()
            .map(|i| i + self.nqubits)
            .collect::<Vec<usize>>();
        let second_axe = (0..indices.len()).collect::<Vec<usize>>();
        self.data = self
            .data
            .tensordot(&op_transconj.data, (&first_axe, &second_axe))
            .unwrap();

        let moveaxis_src_first = (0..indices.len() as i32).collect::<Vec<i32>>();
        let moveaxis_src_second = (1..(indices.len() + 1) as i32).map(|i| -i).collect();
        let src = [moveaxis_src_first, moveaxis_src_second].concat();

        let moveaxis_dest_first = indices.iter().map(|&i| i as i32).collect::<Vec<i32>>();
        let moveaxis_dest_second = indices
            .iter()
            .rev()
            .map(|&i| i as i32 + self.nqubits as i32)
            .collect();
        let dst = [moveaxis_dest_first, moveaxis_dest_second].concat();

        self.data = self.data.moveaxis(&src, &dst).unwrap();

        Ok(())
    }

    pub fn equals(&self, other: DensityMatrix, tol: f64) -> bool {
        if self.data.shape.iter().product::<usize>() == other.data.shape.iter().product::<usize>() {
            for i in 0..self.data.data.len() {
                if complex_approx_eq(self.data.data[i], other.data.data[i], tol) == false {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    /*
     ** Currently making a kronecker product by its own.
     ** Would like to generalize it to the tensor struct with the tensor.product method.
     */
    pub fn tensor(&mut self, other: &DensityMatrix) {
        // Update the number of qubits in `self`
        self.nqubits += other.nqubits;
    
        // Calculate the new size and shape for the resulting tensor
        let new_dim = self.size * other.size;
        let new_shape = vec![2; 2 * self.nqubits];
    
        // Create a new result tensor with the updated shape
        let mut result = Tensor::new(&new_shape);
    
        // Compute the tensor product in parallel
        result.data.iter_mut().enumerate().for_each(|(idx, res)| {
            // Decompose the linear index into 2D indices (i, j)
            let i = idx / new_dim;
            let j = idx % new_dim;
    
            // Decompose indices into `self` and `other` components
            let (self_row, other_row) = (i / other.size, i % other.size);
            let (self_col, other_col) = (j / other.size, j % other.size);
    
            // Retrieve elements from `self` and `other`
            let self_data = self.data.data[self_row * self.size + self_col];
            let other_data = other.data.data[other_row * other.size + other_col];
    
            // Compute and store the result
            *res = self_data * other_data;
        });
    
        // Update `self` with the resulting tensor
        self.data = result;
        self.size = new_dim; // Update the size for consistency
    }

    pub fn ptrace(&mut self, qargs: &[usize]) -> Result<(), &str> {
        let n = self.nqubits;
        if !qargs.iter().all(|&e| e < n) {
            return Err("Wrong qubit argument for partial trace");
        }
    
        let qargs_set: HashSet<usize> = qargs.iter().cloned().collect();
    
        let total_dim = 2_usize.pow(n as u32);
    
        // Indices for subsystems
        let remaining_qubits: Vec<usize> = (0..n).filter(|i| !qargs_set.contains(i)).collect();
        let remaining_dim = 2_usize.pow(remaining_qubits.len() as u32);
        
        // Use a parallel iterator for the outer loop
        let reduced_dm = (0..remaining_dim * remaining_dim)
            .into_iter()
            .map(|idx| {
                let reduced_i = idx / remaining_dim;
                let reduced_j = idx % remaining_dim;

                let mut contribution = Complex::ZERO;

                for i in 0..(1 << qargs.len()) {
                    // Map remaining and traced indices back to the full index space
                    let mut full_i = 0;
                    let mut full_j = 0;

                    for (k, &q) in remaining_qubits.iter().enumerate() {
                        let new_pos = n - q - 1;
                        full_i |= ((reduced_i >> remaining_qubits.len() - k - 1) & 1) << new_pos;
                        full_j |= ((reduced_j >> remaining_qubits.len() - k - 1) & 1) << new_pos;
                    }

                    for (k, &q) in qargs.iter().enumerate() {
                        let bit_value = (i >> qargs.len() - k - 1) & 1;
                        let new_pos = n - q - 1;
                        full_i |= bit_value << new_pos;
                        full_j |= bit_value << new_pos;
                    }
    
                    contribution += self.data.data[full_i * total_dim + full_j];
                }
                contribution

            }).collect();
    
        // Update density matrix with the reduced one
        self.data.data = reduced_dm;
        let mut shape: Vec<usize> = vec![1];
        if remaining_qubits.len() != 0 {
            shape = vec![2; remaining_dim];
        }
        self.data.shape = shape;
        self.nqubits -= qargs.len();
        self.size = 1 << self.nqubits;
        Ok(())
    }  

    pub fn entangle(&mut self, edge: &(usize, usize)) {
        self.evolve(&Operator::two_qubits(TwoQubitsOp::CZ), &[edge.0, edge.1]);
    }

    pub fn swap(&mut self, edge: &(usize, usize)) {
        self.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[edge.0, edge.1]);
    }

    pub fn cnot(&mut self, edge: &(usize, usize)) {
        self.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[edge.0, edge.1]);
    }
}
