use std::{f64::consts::FRAC_1_SQRT_2, fmt};

use num_complex::Complex;
use num_traits::pow;
use rand::Error;

pub enum OneQubitOp {
    I,
    H,
    X,
    Y,
    Z
}

pub enum TwoQubitsOp {
    CX,
    CZ,
    SWAP
}

pub struct Operator {
    pub is_one_qubit_op: bool,
    pub data: Vec<Complex<f64>>
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print(f)
    }
}

impl Clone for Operator {
    fn clone(&self) -> Operator {
        Operator { 
            is_one_qubit_op: self.is_one_qubit_op,
            data: self.data.clone()
        }
    }
}

impl Operator {
    pub fn new(vec: &Vec<Complex<f64>>) -> Result<Self, &str> {
        let mut is_one_qubit_op = false;
        if vec.len() == 4 {
            is_one_qubit_op = true;
        } else if vec.len() == 16 {

        } else {
            return Err("Operator for more than 2 qubits are not implemented.");
        }
        Ok(Operator { is_one_qubit_op, data: vec.to_vec() })
    }
    pub fn one_qubit(gate: OneQubitOp) -> Self {
        let data;
        match gate {
            OneQubitOp::H => {
                data = vec![Complex::new(FRAC_1_SQRT_2, 0.); 4];
            }
            OneQubitOp::X => {
                data = vec![Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(1., 0.), Complex::new(0., 0.)];
            },
            OneQubitOp::Y => {
                data = vec![Complex::new(0., 0.), Complex::new(0., -1.), Complex::new(0., 1.), Complex::new(0., 0.)];
            },
            OneQubitOp::Z => {
                data = vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(-1., 0.)];
            },
            OneQubitOp::I => {
                data = vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)];
            },
        }
        Self {
            is_one_qubit_op: true,
            data
        }
    }

    pub fn print(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size;
        if self.is_one_qubit_op {
            size = 2;
        } else {
            size = 4;
        }
        write!(f, "[")?;
        for i in 0..size {
            write!(f, "[")?;
            for j in 0..size {
                write!(f, "{}", self.data[size * i + j])?;
                if j != size - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i == size - 1 {
                write!(f, "]")?;
            }
            writeln!(f, "")?;
        }
        writeln!(f, "\n")
    }

    pub fn two_qubits(gate: TwoQubitsOp) -> Self {
        let mut data = vec![Complex::new(0., 0.); 16];
        data[0 * 4 + 0] = Complex::new(1., 0.);
        data[1 * 4 + 1] = Complex::new(1., 0.);
        match gate {
            TwoQubitsOp::CX => {
                data[2 * 4 + 3] = Complex::new(1., 0.);
                data[3 * 4 + 2] = Complex::new(1., 0.);
            },
            TwoQubitsOp::CZ => {
                data[2 * 4 + 2] = Complex::new(1., 0.);
                data[3 * 4 + 3] = Complex::new(-1., 0.);
            },
            TwoQubitsOp::SWAP => {
                data[2 * 4 + 1] = Complex::new(1., 0.);
                data[1 * 4 + 2] = Complex::new(1., 0.);
                data[3 * 4 + 3] = Complex::new(1., 0.);
            },
        }
        Self {
            is_one_qubit_op: false,
            data
        }
    }

    pub fn conj(&self) -> Operator {
        let new_data = self.data.iter().map(|e| e.conj()).collect();
        Operator { is_one_qubit_op: self.is_one_qubit_op, data: new_data }
    }
 
    pub fn transpose(&self) -> Operator {
        let mut size;
        if self.is_one_qubit_op {
            size = 1;
        } else {
            size = 2;
        }
        size = pow(2, size);
        let mut result = vec![Complex::new(0., 0.); size * size];
        for i in 0..size {
            for j in 0..size{
                result[j * size + i] = self.data[i * size + j];
            }
        }
        Operator { is_one_qubit_op: self.is_one_qubit_op, data: result }
    }

    pub fn transconj(&self) -> Operator {
        let mut size;
        if self.is_one_qubit_op {
            size = 1;
        } else {
            size = 2;
        }
        size = pow(2, size);
        let mut new_data = vec![Complex::new(0., 0.); size * size];
        for i in 0..size {
            for j in 0..size{
                new_data[j * size + i] = self.data[i * size + j].conj();
            }
        }
        Operator { is_one_qubit_op: self.is_one_qubit_op, data: new_data }        
    }
}