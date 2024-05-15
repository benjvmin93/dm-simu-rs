use num_complex::Complex;
use tensor::Tensor;

use crate::tensor;

pub enum State {
    ZERO,
    PLUS
}

pub fn tensor_to_dm(tensor: Tensor, size: usize) -> DensityMatrix {
    let mut dm = DensityMatrix::new(size, None);
    for i in 0..size {
        for j in 0..size {
            let value = tensor.get(&[2 * i, 2 * j]);
            dm.set(i, j, value);
        }
    }
    dm
}

fn int_to_slice_representation(num: usize, n: usize) -> Vec<usize> {
    let bin = format!("{:0>width$b}", num, width=n);
    let digits = bin.chars().map(|c| c.to_digit(10).unwrap() as usize).collect();
    digits
}

// 1D representation of a size * size density matrix.
pub struct DensityMatrix {
    pub data: Vec<Complex<f64>>,
    pub size: usize,
    pub nqubits: usize
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
            let tensor_index = int_to_slice_representation(i, self.size);
            tensor.set(&tensor_index, data);
        }
        tensor
    }

    // Perform some operation using tensor representation
    pub fn multiply_using_tensor(&self, other: &DensityMatrix) -> DensityMatrix {
        let tensor_self = self.to_tensor();
        let tensor_other = other.to_tensor();
        let tensor_result = tensor_self.multiply(&tensor_other);
        tensor_to_dm(tensor_result, self.size)
    }

    pub fn print(&self) -> () {
        for i in 0..self.size {
            for j in 0..self.size {
                print!("{:?}", self.data[self.size * i + j]);
            }
            println!("");
        }
        println!("\n");
    }
}