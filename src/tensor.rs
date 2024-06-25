use core::fmt;
use num_traits::{Zero, One};
use std::ops::{Add, Mul, AddAssign};

use crate::tools::{bitwise_bin_vec_to_int, bitwise_int_to_bin_vec};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Zero + Clone + Mul<Output = T> + Add<Output = T> + AddAssign,
{
    // Initialize a new tensor with given shape
    pub fn new(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![T::zero(); size],
            shape: shape.to_vec(),
        }
    }

    // Initialize a new tensor from a given vector and a given shape.
    pub fn from_vec(vec: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(vec.len(),  shape.iter().product(), "Vector length {} does not match the given tensor shape {:?}", vec.len(), shape);
        Self {
            data: vec,
            shape
        }
    }

    pub fn print(&self, f: &mut fmt::Formatter<'_>, shape: &[usize], data: &[T]) -> fmt::Result
    where
        T: fmt::Debug,
    {
        if shape.len() == 1 {
            write!(f, "[")?;
            for (i, item) in data.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", item)?;
            }
            write!(f, "]")
        } else {
            let chunk_size: usize = shape[1..].iter().product();
            for (i, chunk) in data.chunks(chunk_size).enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                self.print(f, &shape[1..], chunk)?
            }
            write!(f, "]")
        }
    }
    

    // Get index with the given tensor indices
    pub fn get_index(&self, indices: &[u8]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        let mut index: usize = 0;
        let mut multiplier: usize = 1;
        for (i, &idx) in indices.iter().enumerate().rev() {
            index += idx as usize * multiplier;
            multiplier *= self.shape[i];
        }
        index
    }

    // Access element at given indices
    pub fn get(&self, indices: &[u8]) -> T {
        let index = self.get_index(indices);
        self.data[index].clone()
    }

    // Set element at given indices
    pub fn set(&mut self, indices: &[u8], value: T) {
        let index = self.get_index(indices);
        self.data[index] = value;
    }

    // Perform tensor addition
    pub fn add(&self, other: &Tensor<T>) -> Self {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(&self.shape);
        for (i, self_data) in self.data.iter().enumerate() {
            result.data[i] = self_data.clone() + other.data[i].clone();
        }
        result
    }

    // Perform tensor multiplication (element-wise)
    pub fn multiply(&self, other: &Tensor<T>) -> Self {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(&self.shape);
        for (i, self_data) in self.data.iter().enumerate() {
            result.data[i] = self_data.clone() * other.data[i].clone();
        }
        result
    }

    // Method to compute the tensor product of two tensors
    pub fn tensor_product(&self, other: &Tensor<T>) -> Tensor<T> {
        // Check if tensors are compatible for tensor product
        assert_eq!(self.data.len(), self.shape.iter().product());
        assert_eq!(other.data.len(), other.shape.iter().product());

        // Calculate the shape of the resulting tensor
        let mut new_shape = self.shape.clone();
        new_shape.extend(other.shape.iter().cloned());

        // Calculate the data of the resulting tensor
        let mut new_data = Vec::new();
        for self_data in self.data.iter() {
            for other_data in other.data.iter() {
                new_data.push(self_data.clone() * other_data.clone());
            }
        }
        Tensor {
            data: new_data,
            shape: new_shape,
        }
    }

    pub fn tensordot(&self, other: &Tensor<T>, axes: (&[usize], &[usize])) -> Result<Tensor<T>, &str> {
        if axes.0.len() != axes.1.len() {
            return Err("Axes dimensions must match");
        }
        
        let mut new_shape_self = self.shape.clone();
        let mut new_shape_other = other.shape.clone();

        let mut sorted_axes_self = axes.0.to_vec();
        sorted_axes_self.sort_unstable_by(|a: &usize, b: &usize| b.cmp(a));
        for &axis in sorted_axes_self.iter() {
            if axis >= new_shape_self.len() {
                return Err("Axis out of bounds for self");
            }
            new_shape_self.remove(axis);
        }

        let mut sorted_axes_other = axes.1.to_vec();
        sorted_axes_other.sort_unstable_by(|a, b| b.cmp(a));
        for &axis in sorted_axes_other.iter() {
            if axis >= new_shape_other.len() {
                return Err("Axis out of bounds for other");
            }
            new_shape_other.remove(axis);
        }

        new_shape_self.extend(new_shape_other);
        
        let result_shape = new_shape_self;
        let result_data = vec![T::zero(); result_shape.iter().product()];
        let mut result = Tensor::from_vec(result_data, result_shape);

        for (i, value_self) in self.data.iter().enumerate() {
            let indices_self = bitwise_int_to_bin_vec(i, self.shape.len());
            let indices_common: Vec<u8> = axes.0.iter().map(|&axis| indices_self[axis]).collect();
            let indices_self_reduced: Vec<u8> = indices_self.iter().enumerate()
                .filter(|&(idx, _)| !axes.0.contains(&idx))
                .map(|(_, &val)| val)
                .collect();

            for (j, value_other) in other.data.iter().enumerate() {
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
                    result.data[result_index] += value_self.clone() * value_other.clone();
                }
            }
        }
        Ok(result)
    }

    // Helper function to calculate strides for a given shape
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // Helper function to unravel a flat index to a multidimensional index
    fn unravel_index(index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
        let mut unraveled = vec![0; shape.len()];
        let mut remainder = index;
        for (i, stride) in strides.iter().enumerate() {
            unraveled[i] = remainder / stride;
            remainder %= stride;
        }
        unraveled
    }

    // Helper function to ravel a multidimensional index to a flat index
    fn ravel_index(index: &[usize], strides: &[usize]) -> usize {
        index.iter().zip(strides).map(|(i, s)| i * s).sum()
    }

    pub fn transpose(&self, axes: Vec<usize>) -> Self {
        let axes = if axes.is_empty() {
            (0..self.shape.len()).rev().collect()
        } else {
            axes
        };

        let new_shape = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let old_strides = Self::calculate_strides(&self.shape);
        let new_strides = Self::calculate_strides(&new_shape);

        let mut new_data = vec![T::zero(); self.data.len()];

        for (i, cell) in self.data.iter().enumerate() {
            let old_index = Self::unravel_index(i, &self.shape, &old_strides);
            let new_index = axes.iter().map(|&axis| old_index[axis]).collect::<Vec<_>>();
            let new_pos = Self::ravel_index(&new_index, &new_strides);
            new_data[new_pos] = cell.clone();
        }
        Tensor {
            data: new_data,
            shape: new_shape
        }
    }

    pub fn moveaxis(&self, source: &[i32], dest: &[i32]) -> Result<Tensor<T>, &str> {
        if source.len() != dest.len() {
            return Err("source and destination arguments must have the same number of elements");
        }

        let ndim = self.shape.len();

        let convert_index = |idx: isize| -> usize {
            if idx < 0 {
                (ndim as isize + idx) as usize
            } else {
                idx as usize
            }
        };
        let source: Vec<usize> = source.iter()
            .map(|&x| convert_index(x.try_into().unwrap()))
            .collect();
        let dest: Vec<usize> = dest.iter()
            .map(|&x| convert_index(x.try_into().unwrap()))
            .collect();
        
        let mut order: Vec<usize> = (0..ndim).filter(|&n| !source.contains(&(n as usize))).collect();

        let mut pairs: Vec<_> = dest.iter().cloned().zip(source.iter().cloned()).collect(); // Create pairs of (destination, source) elements
        pairs.sort_by_key(|&(dest, _)| dest);

        for (dest, src) in pairs {
            order.insert(dest as usize, src as usize);
        }

        let result = self.transpose(order);
        
        Ok(result)
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: fmt::Debug + Clone + Add<Output = T> + Mul<Output = T> + AddAssign + Zero
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array(")?;
        self.print(f, &self.shape, &self.data)?;
        write!(f, ")")
    }
}