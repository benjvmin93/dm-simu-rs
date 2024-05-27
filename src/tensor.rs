use core::fmt;
use std::{collections, result};
use num_complex::Complex;
use rand::Error;
use crate::tools::{bitwise_bin_vec_to_int, bitwise_int_to_bin_vec, DisplayComplex};

pub struct Tensor {
    pub data: Vec<Complex<f64>>,
    pub shape: Vec<usize>,
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array(")?;
        self.print(f, &self.shape, &self.data)?;
        write!(f, ")")
    }
}

impl Tensor {
    // Initialize a new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![Complex::new(0.0, 0.0); size],
            shape,
        }
    }

    // Initialize a new tensor from a given vector and a given shape.
    pub fn from_vec(vec: Vec<Complex<f64>>, shape: Vec<usize>) -> Self {
        assert_eq!(vec.len(),  shape.iter().product(), "Vector length {} does not match the given tensor shape {:?}", vec.len(), shape);
        Self {
            data: vec,
            shape
        }
    }

    pub fn print(&self, f: &mut fmt::Formatter<'_>, shape: &[usize], data: &[Complex<f64>]) -> fmt::Result {
        write!(f, "[")?;
        if shape.len() == 1 {
            for i in 0..self.shape[0] {
                write!(f, "{}", DisplayComplex(data[i]))?;
                if i < shape[0] - 1 {
                    write!(f, ", ")?;
                }
            }
        } else {
            let sub_tensor_size: usize = shape[1..].iter().product();
            for i in 0..shape[0] {
                self.print(f, &shape[1..], &data[i * sub_tensor_size..(i + 1) * sub_tensor_size])?;
                if i < shape[0] - 1 {
                    write!(f, ",\n\t")?;
                }
            }
        }
        write!(f, "]")?;
        Ok(())
    }

    // Get index with the given tensor indices
    pub fn get_index(&self, indices: &[u8]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        let mut index: usize = 0;
        let mut multiplier: usize = 1;
        for i in (0..indices.len()).rev() {
            index += indices[i] as usize * multiplier;
            multiplier *= self.shape[i];
        }
        index
    }

    // Access element at given indices
    pub fn get(&self, indices: &[u8]) -> Complex<f64> {
        let index = self.get_index(indices);
        self.data[index]
    }

    // Set element at given indices
    pub fn set(&mut self, indices: &[u8], value: Complex<f64>) {
        let index = self.get_index(indices);
        self.data[index] = value;
    }

    // Perform tensor addition
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(self.shape.clone());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    // Perform tensor multiplication (element-wise)
    pub fn multiply(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(self.shape.clone());
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    // Method to compute the tensor product of two tensors
    pub fn tensor_product(&self, other: &Tensor) -> Tensor {
        // Check if tensors are compatible for tensor product
        assert_eq!(self.data.len(), self.shape.iter().product());
        assert_eq!(other.data.len(), other.shape.iter().product());

        // Calculate the shape of the resulting tensor
        let mut new_shape = self.shape.clone();
        new_shape.extend(other.shape.iter().cloned());

        // Calculate the data of the resulting tensor
        let mut new_data = Vec::new();
        for i in 0..self.data.len() {
            for j in 0..other.data.len() {
                new_data.push(self.data[i] * other.data[j]);
            }
        }
        Tensor {
            data: new_data,
            shape: new_shape,
        }
    }

    pub fn tensordot(&self, other: &Tensor, axes: (&[usize], &[usize])) -> Result<Tensor, &str> {
        if axes.0.len() != axes.1.len() {
            return Err("Axes dimensions must match");
        }
        
        let mut new_shape_self = self.shape.clone();
        let mut new_shape_other = other.shape.clone();

        for &axis in axes.0.iter().rev() {
            new_shape_self.remove(axis);
        }

        for &axis in axes.1.iter().rev() {
            new_shape_other.remove(axis);
        }

        new_shape_self.extend(new_shape_other);
        
        let result_shape = new_shape_self;
        let result_data = vec![Complex::new(0., 0.); result_shape.iter().product()];
        let mut result = Tensor::from_vec(result_data, result_shape);

        for (i, &value_self) in self.data.iter().enumerate() {
            let indices_self = bitwise_int_to_bin_vec(i, self.shape.len());
            let indices_common: Vec<u8> = axes.0.iter().map(|&axis| indices_self[axis]).collect();
            let indices_self_reduced: Vec<u8> = indices_self.iter().enumerate()
                .filter(|&(idx, _)| !axes.0.contains(&idx))
                .map(|(_, &val)| val)
                .collect();

            for (j, &value_other) in other.data.iter().enumerate() {
                let indices_other = bitwise_int_to_bin_vec(j, other.shape.len());
                let indices_common_other: Vec<u8> = axes.1.iter().map(|&axis| indices_other[axis]).collect();

                if indices_common == indices_common_other {
                    let indices_other_reduces: Vec<u8> = indices_other.iter().enumerate()
                        .filter(|&(idx, _)| !axes.1.contains(&idx))
                        .map(|(_, &val)| val)
                        .collect();

                    let mut result_indices = indices_self_reduced.clone();
                    result_indices.extend(indices_other_reduces);

                    let result_index = bitwise_bin_vec_to_int(&result_indices);
                    result.data[result_index] += value_self * value_other;
                }
            }
        }
        Ok(result)
    }

}