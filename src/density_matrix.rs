use core::fmt;
use std::collections::HashSet;

use num_complex::Complex;
use num_traits::ConstZero;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::operators::{Operator, TwoQubitsOp};
use crate::tools::{are_elements_unique, complex_approx_eq};

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
    pub data: Vec<Complex<f64>>,
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
        match initial_state {
            State::PLUS => {
                // Set density matrix to |+><+| \otimes n
                let mut dm = Self {
                    data: vec![Complex::ONE; size * size],
                    size,
                    nqubits,
                };
                dm.data = dm
                    .data
                    .iter()
                    .map(|n| *n / Complex::new(size as f64, 0.))
                    .collect();
                dm
            }
            State::ZERO => {
                // Set density matrix to |0><0| \otimes n
                let mut dm = Self {
                    data: vec![Complex::ZERO; size * size],
                    size,
                    nqubits,
                };
                dm.data[0] = Complex::ONE;
                dm
            }
            State::ONE => {
                let mut dm = Self {
                    data: vec![Complex::ZERO; size * size],
                    size,
                    nqubits,
                };
                // Generate the binary vector for |1>
                dm.data[size * size - 1] =  Complex::ONE;
                dm
            }
            State::MINUS => {
                let mut dm = Self {
                    data: vec![Complex::ZERO; size * size],
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
                        dm.data[i * size + j] = value;
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
            data,
            size,
            nqubits,
        })
    }

    pub fn from_vec(vec: &[Complex<f64>]) -> Result<Self, &'static str> {
        let len = vec.len();
        if !len.is_power_of_two() {
            return Err("The size of the matrix is not a power of two");
        }
        let dim = len.ilog2() as usize;
        if dim % 2 != 0 {
            return Err("The dimensions of the matrix are not consistent");
        }

        Ok(DensityMatrix {
            data: vec.to_vec(),
            size: dim,
            nqubits: dim / 2
        })        
    }

    pub fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            write!(f, "[")?;
            for j in 0..self.size {
                write!(f, "{}", self.data[i * self.size + j])?;
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
    pub fn get(&self, i: usize, j: usize) -> Complex<f64> {
        self.data[i * self.size + j]
    }

    // Set element at row i and column j
    pub fn set(&mut self, i: usize, j: usize, value: Complex<f64>) {
        self.data[i * self.size + j] =  value;
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
        let expectation = self.data
            .par_iter()
            .enumerate()
            .map(|(idx, &rho_elt)| {
                let i = idx / self.size;
                let j = idx % self.size;
    
                if (i & !index_mask) == (j & !index_mask) {
                    let op_i = (i & index_mask) >> (self.nqubits - index - 1);
                    let op_j = (j & index_mask) >> (self.nqubits - index - 1);
    
                    rho_elt * op.data[op_i * 2 + op_j]
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
            .map(|i| self.get(i, i))
            .fold(Complex::ZERO, |acc, x| acc + x) // Fold ensures a starting value
    }

    pub fn normalize(&mut self) {
        let trace = self.trace();
        self.data = self
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
                    if self.data[data_idx] == Complex::ZERO {
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
                    sum += op.data[b_i * 2 + p]
                        * self.data[data_idx]
                        * op.data[b_j * 2 + q].conj();
                });
                sum
            }).collect();

        std::mem::swap(&mut self.data, &mut new_dm);
        Ok(())
    }

    pub fn evolve(&mut self, op: &Operator, indices: &[usize]) -> Result<(), String> {
        // Check for unique indices
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
    
        // Calculate dimensions
        let op_dim = 1 << op.nqubits;
        let dim = 1 << self.nqubits;
        println!("Operator dimension: {}, Density matrix dimension: {}", op_dim, dim);
    
        // Calculate target bitshifts
        let target_bitshifts: Vec<usize> = indices.iter().map(|i| self.nqubits - i - 1).collect();
        println!("Target bitshifts: {:?}", target_bitshifts);
    
        // Prepare a new density matrix
        let mut new_dm: Vec<Complex<f64>> = (0..dim * dim)
            .map(|idx| {
                let i = idx / dim;
                let j = idx % dim;
    
                // Debug: Current indices in binary representation
                println!(
                    "Processing density matrix element: (i: {}, j: {}) -> (i: {:b}, j: {:b})",
                    i, j, i, j
                );
    
                // Extract bits at target positions and compose bitmask
                let mut b_i = 0;
                let mut b_j = 0;
                /* `target_bitshifts` is ordered from the most to the least significant bits
                   to extract to `b_i` and `b_j`, therefore the iteration is reversed for
                   `index` to match the weight of the extracted bits. */
                let bitmask: usize = target_bitshifts.iter().rev().enumerate().map(|(index, t)| {
                    b_i |= ((i >> t) & 1) << index;
                    b_j |= ((j >> t) & 1) << index;
                    1 << t
                }).sum();
    
                println!(
                    "b_i: {:b}, b_j: {:b}, bitmask: {:b} for element (i: {:b}, j: {:b})",
                    b_i, b_j, bitmask, i, j
                );
    
                // Get bits with targets set to 0
                let i_base = i & !bitmask;
                let j_base = j & !bitmask;
    
                println!(
                    "i_base: {:b}, j_base: {:b} (i_base: {}, j_base: {})",
                    i_base, j_base, i_base, j_base
                );
    
                let mut sum = Complex::ZERO;
    
                // Iterate over operator indices
                (0..op_dim * op_dim).for_each(|op_idx| {
                    let p = op_idx / op_dim;
                    let q = op_idx % op_dim;
    
                    let mut p_prime = 0;
                    let mut q_prime = 0;
    
                    target_bitshifts.iter().enumerate().for_each(|(index, &bitshift)| {
                        let p_bit = (p >> (op.nqubits - index - 1)) & 1;
                        let q_bit = (q >> (op.nqubits - index - 1)) & 1;
                        p_prime |= p_bit << bitshift;
                        q_prime |= q_bit << bitshift;
                    });
    
                    let i_prime = i_base | p_prime;
                    let j_prime = j_base | q_prime;
    
                    println!(
                        "op_idx: {}, p: {:b}, q: {:b}, p_prime: {:b}, q_prime: {:b}, i_prime: {:b}, j_prime: {:b}",
                        op_idx, p, q, p_prime, q_prime, i_prime, j_prime
                    );
    
                    let data_idx = i_prime * dim + j_prime;
    
                    if self.data[data_idx] != Complex::ZERO {
                        let contrib = op.data[b_i * op_dim + p]
                            * self.data[data_idx]
                            * op.data[b_j * op_dim + q].conj();
                        println!(
                            "Contribution to sum: {} (from data_idx: {}, op indices: {}, {})",
                            contrib, data_idx, b_i * op_dim + p, b_j * op_dim + q
                        );
                        sum += contrib;
                    }
                });
    
                println!(
                    "New density matrix element (i: {:b}, j: {:b}): {}",
                    i, j, sum
                );
                sum
            }).collect();
    
        println!("Swapping old density matrix with the new one.");
        std::mem::swap(&mut self.data, &mut new_dm);
        println!("Density matrix evolution completed successfully.");
        Ok(())
    }
    

    pub fn equals(&self, other: &DensityMatrix, tol: f64) -> bool {
        if self.data.len() == other.data.len() {
            for i in 0..self.data.len() {
                if complex_approx_eq(self.data[i], other.data[i], tol) == false {
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
    
        // Create a new result tensor with the updated shape
        let mut result = vec![Complex::ZERO; new_dim * new_dim];
    
        // Compute the tensor product in parallel
        result.iter_mut().enumerate().for_each(|(idx, res)| {
            // Decompose the linear index into 2D indices (i, j)
            let i = idx / new_dim;
            let j = idx % new_dim;
    
            // Decompose indices into `self` and `other` components
            let (self_row, other_row) = (i / other.size, i % other.size);
            let (self_col, other_col) = (j / other.size, j % other.size);
    
            // Retrieve elements from `self` and `other`
            let self_data = self.data[self_row * self.size + self_col];
            let other_data = other.data[other_row * other.size + other_col];
    
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
    
                    contribution += self.data[full_i * total_dim + full_j];
                }
                contribution

            }).collect();
    
        // Update density matrix with the reduced one
        self.data = reduced_dm;
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
