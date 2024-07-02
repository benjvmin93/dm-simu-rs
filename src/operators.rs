use crate::tensor::Tensor;
use crate::tools::bitwise_int_to_bin_vec;
use num_complex::Complex;
use num_traits::pow;
use std::f64;
use std::{f64::consts::FRAC_1_SQRT_2, fmt};

pub enum OneQubitOp {
    I,
    H,
    X,
    Y,
    Z,
}

pub enum TwoQubitsOp {
    CX,
    CZ,
    SWAP,
}

#[derive(Clone)]
pub struct Operator {
    pub nqubits: usize,
    pub tensor: Tensor<Complex<f64>>,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tensor.print(f, &self.tensor.shape, &self.tensor.data)
    }
}

impl Operator {
    pub fn new(data: Vec<Complex<f64>>) -> Result<Self, String> {
        let size = (data.len() as f64).sqrt();
        if !(size as usize).is_power_of_two() {
            return Err(
                "Operator should be a squared matrix with size 2^nqubits * 2^nqubits".to_string(),
            );
        }
        let nqubits = size.log2() as usize;
        let shape = vec![2; 2 * nqubits];

        Ok(Operator {
            nqubits,
            tensor: Tensor::from_vec(data, shape),
        })
    }

    pub fn one_qubit(gate: OneQubitOp) -> Self {
        let data;
        let nqubits = 1;
        match gate {
            OneQubitOp::H => {
                data = vec![Complex::new(FRAC_1_SQRT_2, 0.); 4];
            }
            OneQubitOp::X => {
                data = vec![Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO];
            }
            OneQubitOp::Y => {
                data = vec![
                    Complex::ZERO,
                    Complex::new(0., -1.),
                    Complex::new(0., 1.),
                    Complex::ZERO,
                ];
            }
            OneQubitOp::Z => {
                data = vec![
                    Complex::ONE,
                    Complex::ZERO,
                    Complex::ZERO,
                    Complex::new(-1., 0.),
                ];
            }
            OneQubitOp::I => {
                data = vec![Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ONE];
            }
        }
        Self {
            nqubits,
            tensor: Tensor::from_vec(data, vec![2, 2]),
        }
    }

    pub fn two_qubits(gate: TwoQubitsOp) -> Self {
        let nqubits = 2;
        let mut data = vec![Complex::ZERO; 16];
        data[0 * 4 + 0] = Complex::ONE;
        match gate {
            TwoQubitsOp::CX => {
                data[2 * 4 + 3] = Complex::ONE;
                data[3 * 4 + 2] = Complex::ONE;
                data[1 * 4 + 1] = Complex::ONE;
            }
            TwoQubitsOp::CZ => {
                data[2 * 4 + 2] = Complex::ONE;
                data[3 * 4 + 3] = Complex::new(-1., 0.);
                data[1 * 4 + 1] = Complex::ONE;
            }
            TwoQubitsOp::SWAP => {
                data[2 * 4 + 1] = Complex::ONE;
                data[1 * 4 + 2] = Complex::ONE;
                data[3 * 4 + 3] = Complex::ONE;
            }
        }
        Self {
            nqubits,
            tensor: Tensor::from_vec(data, vec![2, 2, 2, 2]),
        }
    }

    pub fn conj(&self) -> Operator {
        let new_data = self
            .tensor
            .data
            .iter()
            .map(|e| e.conj())
            .collect::<Vec<Complex<f64>>>();
        Operator {
            nqubits: self.nqubits,
            tensor: Tensor::from_vec(new_data, self.tensor.shape.clone()),
        }
    }

    pub fn transpose(&self) -> Operator {
        let size = pow(2, self.nqubits);
        let mut result = vec![Complex::ZERO; size * size];
        for i in 0..size {
            for j in 0..size {
                let indices = bitwise_int_to_bin_vec(i * size + j, 2 * self.nqubits);
                result[j * size + i] = self.tensor.get(&indices);
            }
        }
        Operator {
            nqubits: self.nqubits,
            tensor: Tensor::from_vec(result, self.tensor.shape.clone()),
        }
    }

    pub fn transconj(&self) -> Operator {
        let size = pow(2, self.nqubits);
        let mut new_data = vec![Complex::ZERO; size * size];
        for i in 0..size {
            for j in 0..size {
                let indices = bitwise_int_to_bin_vec(i * size + j, 2 * self.nqubits);
                new_data[j * size + i] = self.tensor.get(&indices).conj();
            }
        }
        Operator {
            nqubits: self.nqubits,
            tensor: Tensor::from_vec(new_data, self.tensor.shape.clone()),
        }
    }
}
