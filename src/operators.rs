use std::f64::consts::FRAC_1_SQRT_2;

use num_complex::Complex;
use num_traits::pow;

pub enum OneQubitOp {
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

impl Operator {
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
        }
        Self {
            is_one_qubit_op: true,
            data
        }
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

    pub fn conj(& mut self) {
        self.data = self.data.iter().map(|e| e.conj()).collect();
    }
 
    pub fn transpose(& mut self) {
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
        self.data = result;
    }
}