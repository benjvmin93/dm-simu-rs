use core::fmt;

use num_complex::Complex;
use tensor::Tensor;

use crate::tensor;
use crate::tools::{bitwise_int_to_bin_vec, complex_approx_eq};
use crate::operators::{OneQubitOp, Operator, TwoQubitsOp};

#[pyo3::pyclass]
#[derive(Copy, Clone)]
pub enum State {
    ZERO,
    PLUS
}

// 1D representation of a size * size density matrix.
pub struct DensityMatrix {
    pub data: Tensor<Complex<f64>>,
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
    pub fn new(nqubits: usize, initial_state: Option<State>) -> Self {
        let size = 1 << nqubits;
        match initial_state {
            Some(State::PLUS) => {  // Set density matrix to |+><+| \otimes n
                let mut dm =  Self {
                    data: Tensor::from_vec(&vec![Complex::new(1., 0.); size * size], &vec![2; 2 * nqubits]),
                    size,
                    nqubits
                };
                dm.data.data = dm.data.data.iter().map(|n| *n / Complex::new(size as f64, 0.)).collect();
                dm
            }
            Some(State::ZERO) => {  // Set density matrix to |0><0| \otimes n
                let mut dm = Self {
                    data: Tensor::from_vec(&vec![Complex::new(0., 0.); size * size], &vec![2; 2 * nqubits]),
                    size,
                    nqubits
                };
                let indices = bitwise_int_to_bin_vec(0, 2 * nqubits);
                dm.data.set(&indices, Complex::new(1., 0.));
                dm
            }
            None => Self {  // Set all matrix elements to 0.
                data: Tensor::new(&vec![2; 2 * nqubits]),
                size,
                nqubits
            },
        }
    }

    pub fn from_statevec(statevec: Vec<Complex<f64>>) -> Result<Self, &'static str> {
        let len = statevec.len();
        if !len.is_power_of_two() {
            return Err("The size of the statevec is not a power of two");
        }
        let nqubits = len.ilog2() as usize;
        let size = len;
        let mut data = vec![Complex::new(0., 0.); size * size];
        
        for i in 0..size {
            for j in 0..size {
                data[i * size + j] = statevec[i] * statevec[j].conj();
            }
        }
        Ok(DensityMatrix {
            data: Tensor::from_vec(&data, &vec![2; 2 * nqubits]),
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
                data: tensor,
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
        self.data.get(&[i, j])
    }

    // Set element at row i and column j
    pub fn set(&mut self, i: u8, j: u8, value: Complex<f64>) {
        let size = self.size as u8;
        let indices = bitwise_int_to_bin_vec((i * size + j) as usize, 2 * self.nqubits);
        self.data.set(&indices, value);
    }

    /* pub fn to_tensor(&self) -> Tensor {
        let shape = vec![2; 2 * self.nqubits];
        let mut tensor = Tensor::new(shape);
        for i in 0..(self.size * self.size) {
            let data = self.data[i];
            let tensor_index = bitwise_int_to_bin_vec(i, self.nqubits * 2);
            tensor.set(&tensor_index, data);
        }
        tensor
    } */

    pub fn expectation_single(&self, op: OneQubitOp, index: usize) -> Result<Complex<f64>, &str> {
        if index >= self.nqubits {
            return Err("Wrong target qubit.");
        }

        let op_tensor = Operator::one_qubit(op);
        let mut result_tensor = self.data.tensordot(&op_tensor.data, (&[1], &[index])).unwrap();
        result_tensor = result_tensor.moveaxis(&[0], &[index as i32]).unwrap();
        let dm_result = DensityMatrix::from_tensor(result_tensor).unwrap();
        Ok(dm_result.trace())
    }

    pub fn trace(&self) -> Complex<f64> {
        // Compute sum over each diagonal elements.
        let mut trace = Complex::new(0., 0.);
        let mut step = 0;
        for i in 0..self.size {
            trace += self.data.get(&[i as u8, step]);
            step += 1;
        }

        trace
    }

    pub fn normalize(&mut self) {
        let trace = self.trace();
        self.data.data = self.data.data.iter()
            .map(|&c| c / trace)
            .collect::<Vec<_>>();
    }

    pub fn evolve_single(&mut self, op: OneQubitOp, index: usize) {
        let op = Operator::one_qubit(op);
        // let op_tensor = Tensor::from_vec(&op.data, vec![2, 2]);
        // let mut rho_tensor: Tensor<Complex<f64>> = self.data;
        self.data = op.data.tensordot(&self.data, (&[1], &[index])).unwrap();
        self.data = self.data.tensordot(&Tensor::from_vec(&op.transconj().data.data, &[2, 2]), (&[index + self.nqubits], &[0])).unwrap();
        self.data = self.data.moveaxis(&[0, ((self.data.shape.len() - 1)).try_into().unwrap()], &[index.try_into().unwrap(), ((index + self.nqubits)).try_into().unwrap() ]).unwrap();
    }

    pub fn evolve(&mut self, op: TwoQubitsOp, indices: &[usize]) {
        let nqb_op = 2;
        let op = Operator::two_qubits(op);

        self.data = op.data.tensordot(&self.data, (
            &(0..indices.len())
                .map(|i| nqb_op + i)
                .collect::<Vec<usize>>(),
            &indices
        )).unwrap();

        let op_transconj = op.transconj();
        self.data = self.data.tensordot(&op_transconj.data,(
            (&indices.iter()
                .map(|i| i + nqb_op)
                .collect::<Vec<usize>>()),
            (&(0..indices.len()).collect::<Vec<usize>>())
        )).unwrap();

        let moveaxis_src_first = (0..indices.len() as i32).collect::<Vec<i32>>();
        let moveaxis_src_second = (1..(indices.len() + 1) as i32).map(|i| -i).collect();
        
        let moveaxis_dest_first = indices.iter().map(|&i| i as i32).collect::<Vec<i32>>();
        let moveaxis_dest_second = indices.iter().rev().map(|&i| i as i32 + nqb_op as i32).collect();
        self.data = self.data.moveaxis(
            &[moveaxis_src_first, moveaxis_src_second].concat(),
            &[moveaxis_dest_first, moveaxis_dest_second].concat()
        ).unwrap();
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

    pub fn tensor(&mut self, other: &DensityMatrix) {
        self.data = self.data.tensor_product(&other.data);
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
        let rho_res = id_tensor.tensordot(&self.data, (&tensordot_first_axe, &trace_axes)).unwrap();
        self.data = rho_res;
        self.nqubits = nqubit_after;
        Ok(())
    }

    pub fn entangle(&mut self, edge: &(usize, usize)) {
        self.evolve(TwoQubitsOp::CZ, &[edge.0, edge.1]);
    }

    pub fn swap(&mut self, edge: &(usize, usize)) {
        self.evolve(TwoQubitsOp::SWAP, &[edge.0, edge.1]);
    }

    pub fn cnot(&mut self, edge: &(usize, usize)) {
        self.evolve(TwoQubitsOp::CX, &[edge.0, edge.1]);
    }
}