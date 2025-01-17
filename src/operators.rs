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
    pub data: Vec<Complex<f64>>,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dim = 1 << self.nqubits;
        write!(f, "[")?;
        for i in 0..dim {
            write!(f, "[")?;
            for j in 0..dim {
                write!(f, "{}", self.data[i * dim + j])?;
                if j != dim - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i == dim - 1 {
                write!(f, "]")?;
            }
            writeln!(f, "")?;
        }
        writeln!(f, "\n")
    }
}

impl Operator {
    pub fn new(data: &[Complex<f64>]) -> Result<Self, String> {
        let size = data.len();
        if !size.is_power_of_two() {
            return Err(
                "Operator should be a squared matrix with size 2^nqubits * 2^nqubits".to_string(),
            );
        }
        let n = (size as f32).log2() as usize;

        if n % 2 != 0 {
            return Err("Operator is not of the size 2^2n".to_string());
        }

        Ok(Operator {
            nqubits: n / 2,
            data: data.to_vec(),
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
        Self { nqubits, data }
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
        Self { nqubits, data }
    }

    pub fn conj(&self) -> Operator {
        let new_data = self
            .data
            .iter()
            .map(|e| e.conj())
            .collect::<Vec<Complex<f64>>>();
        Operator {
            nqubits: self.nqubits,
            data: new_data,
        }
    }

    pub fn transpose(&self) -> Operator {
        let size = pow(2, self.nqubits);
        let mut result = vec![Complex::ZERO; size * size];
        for i in 0..size {
            for j in 0..size {
                result[j * size + i] = self.data[i * size + j];
            }
        }
        Operator {
            nqubits: self.nqubits,
            data: result,
        }
    }

    pub fn transconj(&self) -> Operator {
        let size = pow(2, self.nqubits);
        let mut new_data = vec![Complex::ZERO; size * size];
        for i in 0..size {
            for j in 0..size {
                new_data[j * size + i] = self.data[i * size + j].conj();
            }
        }
        Operator {
            nqubits: self.nqubits,
            data: new_data,
        }
    }
}
