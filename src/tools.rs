use core::fmt;
use num_complex::Complex;
use crate::density_matrix::DensityMatrix;
use crate::tensor::Tensor;

pub struct DisplayComplex(pub Complex<f64>);

impl fmt::Display for DisplayComplex {    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = {
            if self.0.im < 0. {
                "+".to_string()
            } else {
                "-".to_string()
            }
        };
        write!(f, "{}{}{}i", self.0.re, sign, self.0.im)
    }
}

pub fn tensor_to_dm(tensor: Tensor) -> DensityMatrix {
    let nqubits = tensor.shape.len() / 2;
    let mut dm = DensityMatrix::new(nqubits, None);
    for i in 0..nqubits {
        for j in 0..nqubits {
            let value = tensor.get(&[2 * i as u8, 2 * j as u8]);
            dm.set(i, j, value);
        }
    }
    dm
}

pub fn bitwise_int_to_bin_vec(mut num: usize, mut n: usize) -> Vec<u8> {
    let mut bin_vec: Vec<u8> = Vec::new();
    while n > 0 {
        bin_vec.push((num & 1) as u8);
        num >>= 1;
        n -= 1;
    }
    bin_vec.reverse();
    bin_vec
}

pub fn bitwise_bin_vec_to_int(bin_vec: &[u8]) -> usize {
    let mut weight = 0;
    bin_vec.iter().for_each(|b| {
        weight <<= 1;
        weight |= *b as usize;
    });
    weight
}