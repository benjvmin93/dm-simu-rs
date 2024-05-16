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
            let value = tensor.get(&[2 * i as u8, 2 * j as u8]);
            dm.set(i, j, value);
        }
    }
    dm
}

fn bitwise_int_to_bin_vec(mut num: usize, mut n: usize) -> Vec<u8> {
    let mut bin_vec: Vec<u8> = Vec::new();
    while n > 0 {
        bin_vec.push((num & 1) as u8);
        num >>= 1;
        n -= 1;
    }
    bin_vec.reverse();
    bin_vec
}

fn bitwise_bin_vec_to_int(bin_vec: &[u8]) -> usize {
    let mut weight = 0;
    bin_vec.iter().for_each(|b| {
        weight <<= 1;
        weight |= *b as usize;
    });
    weight
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
            let tensor_index = bitwise_int_to_bin_vec(i, self.nqubits * 2);
            println!("{:?}", tensor_index);
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