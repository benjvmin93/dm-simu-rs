use core::fmt;

use num_complex::Complex;
use tensor::Tensor;

use crate::tensor;
use crate::tools::{tensor_to_dm, bitwise_int_to_bin_vec};
use crate::operators::{OneQubitOp, Operator, TwoQubitsOp};

pub enum State {
    ZERO,
    PLUS
}

// 1D representation of a size * size density matrix.
pub struct DensityMatrix {
    pub data: Vec<Complex<f64>>,
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
                    data: vec![Complex::new(1., 0.); size * size],
                    size,
                    nqubits
                };
                dm.data = dm.data.iter().map(|n| *n / Complex::new(size as f64, 0.)).collect();
                dm
            }
            Some(State::ZERO) => {  // Set density matrix to |0><0| \otimes n
                let mut dm = Self {
                    data: vec![Complex::new(0., 0.); size * size],
                    size,
                    nqubits
                };
                dm.data[0] = Complex::new(1., 0.);
                dm
            }
            None => Self {  // Set all matrix elements to 0.
                data: vec![Complex::new(0., 0.); size * size],
                size,
                nqubits
            },
        }

    }
    
    pub fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            write!(f, "[")?;
            for j in 0..self.size {
                write!(f, "{}", self.data[self.size * i + j])?;
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
        self.data[i * self.size + j] = value;
    }

    pub fn to_tensor(&self) -> Tensor {
        let shape = vec![2; 2 * self.nqubits];
        let mut tensor = Tensor::new(shape);
        for i in 0..(self.size * self.size) {
            let data = self.data[i];
            let tensor_index = bitwise_int_to_bin_vec(i, self.nqubits * 2);
            tensor.set(&tensor_index, data);
        }
        tensor
    }

    // Perform some operation using tensor representation
    pub fn multiply_using_tensor(&self, other: &DensityMatrix) -> DensityMatrix {
        let tensor_self = self.to_tensor();
        let tensor_other = other.to_tensor();
        let tensor_result = tensor_self.multiply(&tensor_other);
        tensor_to_dm(tensor_result)
    }

    /*
    pub fn evolve_single(&self, op: OneQubitOp, index: usize) {
        let op = Operator::one_qubit(op);
        let op_tensor = Tensor::from_vec(op.data, vec![2, 2]);
        let mut rho_tensor: Tensor = self.to_tensor();
        rho_tensor = op_tensor.tensordot(&rho_tensor, (&[1], &[index])).unwrap();   // U.rho
        rho_tensor = rho_tensor.tensordot(&rho_tensor, (&[index + self.nqubits], &[0])).unwrap();  // U.rho.U^T

    } */
}