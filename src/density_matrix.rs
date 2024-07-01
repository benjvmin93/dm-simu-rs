use core::fmt;

use num_complex::Complex;
use tensor::Tensor;

use crate::tensor;
use crate::tools::{bitwise_int_to_bin_vec, complex_approx_eq, are_elements_unique};
use crate::operators::{OneQubitOp, Operator, TwoQubitsOp};

#[pyo3::pyclass]
#[derive(Copy, Clone)]
pub enum State {
    ZERO,
    PLUS
}

// 1D representation of a size * size density matrix.
pub struct DensityMatrix {
    pub tensor: Tensor<Complex<f64>>,
    pub size: usize,    // 2 ** nqubits
    pub nqubits: usize
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
            State::PLUS => {  // Set density matrix to |+><+| \otimes n
                let mut dm =  Self {
                    tensor: Tensor::from_vec(vec![Complex::ONE; size * size], vec![2; shape]),
                    size,
                    nqubits
                };
                dm.tensor.data = dm.tensor.data.iter().map(|n| *n / Complex::new(size as f64, 0.)).collect();
                dm
            }
            State::ZERO => {  // Set density matrix to |0><0| \otimes n
                let mut dm = Self {
                    tensor: Tensor::from_vec(vec![Complex::ZERO; size * size], vec![2; shape]),
                    size,
                    nqubits
                };
                let indices = bitwise_int_to_bin_vec(0, shape);
                dm.tensor.set(&indices, Complex::ONE);
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
            tensor: Tensor::from_vec(data, vec![2; 2 * nqubits]),
            size,
            nqubits
        })
    }

    pub fn from_tensor(tensor: Tensor<Complex<f64>>) -> Result<Self, &'static str> {
        if tensor.shape.len() != 2 {
            return Err("Tensor has not the right shape.");
        } else {
            let nqubits = tensor.shape.len() / 2;
            Ok(DensityMatrix {
                tensor: tensor,
                size: 2_i32.pow(nqubits as u32) as usize,
                nqubits
            })
        }
    }
    
    pub fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            write!(f, "[")?;
            for j in 0..self.size {
                let indices = bitwise_int_to_bin_vec(i * self.size + j, 2 * self.nqubits);
                write!(f, "{}", self.tensor.get(&indices))?;
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
        self.tensor.get(&[i, j])
    }

    // Set element at row i and column j
    pub fn set(&mut self, i: u8, j: u8, value: Complex<f64>) {
        let size = self.size as u8;
        let indices = bitwise_int_to_bin_vec((i * size + j) as usize, 2 * self.nqubits);
        self.tensor.set(&indices, value);
    }

    pub fn expectation_single(&self, op: OneQubitOp, index: usize) -> Result<Complex<f64>, &str> {
        if index >= self.nqubits {
            return Err("Wrong target qubit.");
        }

        let op_tensor = Operator::one_qubit(op);
        let mut result_tensor = self.tensor.tensordot(&op_tensor.tensor, (&[1], &[index])).unwrap();
        result_tensor = result_tensor.moveaxis(&[0], &[index as i32]).unwrap();
        let dm_result = DensityMatrix::from_tensor(result_tensor).unwrap();
        Ok(dm_result.trace())
    }

    pub fn trace(&self) -> Complex<f64> {
        // Compute sum over each diagonal elements.
        let mut trace = Complex::ZERO;
        let mut step = 0;
        for i in 0..self.size {
            trace += self.tensor.get(&[i as u8, step]);
            step += 1;
        }

        trace
    }

    pub fn normalize(&mut self) {
        let trace = self.trace();
        self.tensor.data = self.tensor.data.iter()
            .map(|&c| c / trace)
            .collect::<Vec<_>>();
    }

    pub fn evolve_single(&mut self, op: &Operator, index: usize) -> Result<(), String> {
        if index >= self.nqubits {
            return Err(format!("Target qubit {} is not in the range [0-{}].", index, self.nqubits));
        }
        if op.nqubits != 1 {
            return Err(format!("Passed operator is not a one qubit operator."));
        }

        self.tensor = op.tensor.tensordot(&self.tensor, (&[1], &[index])).unwrap();
        self.tensor = self.tensor.tensordot(&Tensor::from_vec(op.transconj().tensor.data, vec![2, 2]), (&[index + self.nqubits], &[0])).unwrap();
        self.tensor = self.tensor.moveaxis(&[0, ((self.tensor.shape.len() - 1)).try_into().unwrap()], &[index.try_into().unwrap(), ((index + self.nqubits)).try_into().unwrap()]).unwrap();

        Ok(())
    }

    pub fn evolve(&mut self, op: &Operator, indices: &[usize]) -> Result<(), String> {
        if !are_elements_unique(indices) {
            return Err("Target qubits must be unique.".to_string());
        }
        for &i in indices.iter() {
            if i >= self.nqubits {
                return Err(format!("Target qubit {} is not in the range [0-{}].", i, self.nqubits));
            }
        }

        let nqb_op = op.nqubits;
        let first_axe = (0..indices.len()).map(|i| nqb_op + i).collect::<Vec<usize>>();
        let second_axe = indices;
        self.tensor = op.tensor.tensordot(
            &self.tensor, 
            (&first_axe, &second_axe)).unwrap();

        let op_transconj = op.transconj();
        let first_axe = indices.iter().map(|i| i + self.nqubits).collect::<Vec<usize>>();
        let second_axe = (0..indices.len()).collect::<Vec<usize>>();
        self.tensor = self.tensor.tensordot(
            &op_transconj.tensor,
            (&first_axe, &second_axe)).unwrap();

        let moveaxis_src_first = (0..indices.len() as i32).collect::<Vec<i32>>();
        let moveaxis_src_second = (1..(indices.len() + 1) as i32).map(|i| -i).collect();
        let src = [moveaxis_src_first, moveaxis_src_second].concat();
        
        let moveaxis_dest_first = indices.iter().map(|&i| i as i32).collect::<Vec<i32>>();
        let moveaxis_dest_second = indices.iter().rev().map(|&i| i as i32 + self.nqubits as i32).collect();
        let dst = [moveaxis_dest_first, moveaxis_dest_second].concat();

        self.tensor = self.tensor.moveaxis(&src, &dst).unwrap();

        Ok(())
    }

    pub fn equals(&self, other: DensityMatrix, tol: f64) -> bool {
        if self.tensor.shape.iter().product::<usize>() == other.tensor.shape.iter().product::<usize>() {
            for i in 0..self.tensor.data.len() {
                if complex_approx_eq(self.tensor.data[i], other.tensor.data[i], tol) == false {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    pub fn tensor(&mut self, other: &DensityMatrix) {
        self.tensor = self.tensor.tensor_product(&other.tensor);
        self.nqubits += other.nqubits;
    }

    pub fn ptrace(&mut self, qargs: &[usize]) -> Result<(), &str> {
        let n = self.nqubits;
        if !qargs.iter().all(|&e| e < n) {
            return Err("Wrong qubit argument for partial trace");
        }
        let nqubit_after = n - qargs.len();
        let second_trace_axe = qargs.iter().map(|e| e + n).collect::<Vec<_>>();
        let trace_axes = [qargs, &second_trace_axe].concat();

        // Build identity tensor
        let id_tensor_size = 2_i32.pow(qargs.len() as u32) as usize;
        let mut id_tensor = Tensor::new(&vec![2; qargs.len() * 2]);
        for i in 0..id_tensor_size * id_tensor_size {
            let index = bitwise_int_to_bin_vec(i * id_tensor_size + i, qargs.len() * 2);
            id_tensor.set(&index, Complex::ONE);
        }

        let tensordot_first_axe = (0..qargs.len() * 2).collect::<Vec<usize>>();
        let rho_res = id_tensor.tensordot(&self.tensor, (&tensordot_first_axe, &trace_axes)).unwrap();
        self.tensor = rho_res;
        self.nqubits = nqubit_after;
        Ok(())
    }

    pub fn entangle(&mut self, edge: &(usize, usize)) {
        self.evolve(
            &Operator::two_qubits(TwoQubitsOp::CZ),
            &[edge.0, edge.1]
        ).unwrap();
    }

    pub fn swap(&mut self, edge: &(usize, usize)) {
        self.evolve(
            &Operator::two_qubits(TwoQubitsOp::SWAP),
            &[edge.0, edge.1]
        ).unwrap();
    }

    pub fn cnot(&mut self, edge: &(usize, usize)) {
        self.evolve(
            &Operator::two_qubits(TwoQubitsOp::CX),
            &[edge.0, edge.1]
        ).unwrap();
    }
}