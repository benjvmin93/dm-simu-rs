use core::fmt;
use num_traits::{One, Zero};
use std::ops::{Add, AddAssign, Mul};

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
    pub fn from_vec(vec: &[T], shape: Vec<usize>) -> Self {
        assert_eq!(
            vec.len(),
            shape.iter().product(),
            "Vector length {} does not match the given tensor shape {:?}",
            vec.len(),
            shape
        );
        Self { data: vec.into(), shape }
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
    pub fn get(&self, indices: &[u8]) -> &T {
        let index = self.get_index(indices);
        &self.data[index]
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
    pub fn product(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::marker::Copy,
    {
        // Check if tensors are compatible for tensor product
        assert_eq!(self.data.len(), self.shape.iter().product());
        assert_eq!(other.data.len(), other.shape.iter().product());

        // Calculate the shape of the resulting tensor
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(other.shape.iter())
            .map(|(self_dim, other_dim)| self_dim * other_dim)
            .collect();

        let result_size = new_shape.iter().copied().product::<usize>();
        let mut new_data = Vec::with_capacity(result_size);

        for i in 0..result_size {
            // Initialize arrays to store the indices in A and B
            let mut a_indices: Vec<u8> = vec![0; self.shape.len()];
            let mut b_indices: Vec<u8> = vec![0; other.shape.len()];

            let mut remaining_index = i;

            // For each dimension, calculate the corresponding indices in A and B
            for j in 0..self.shape.len() {
                a_indices[j] = remaining_index as u8 / other.shape[j] as u8; // Integer division
                b_indices[j] = remaining_index as u8 % other.shape[j] as u8; // Modulo operation
                remaining_index /= other.shape[j]; // Update remaining index for the next dimension
            }

            // Print the index mapping
            new_data.push(*self.get(&a_indices) * *other.get(&b_indices));
        }

        Tensor {
            data: new_data,
            shape: new_shape,
        }
    }

    pub fn tensordot(
        &self,
        other: &Tensor<T>,
        axes: (&[usize], &[usize]),
    ) -> Result<Tensor<T>, &str> {
        if axes.0.len() != axes.1.len() {
            return Err("Axes dimensions must match");
        }

        let mut new_shape_self = self.shape.clone();
        let mut new_shape_other = other.shape.clone();

        let mut sorted_axes_self = axes.0.to_vec();
        sorted_axes_self.sort_by(|a: &usize, b: &usize| b.cmp(a));
        for &axis in sorted_axes_self.iter() {
            if axis >= new_shape_self.len() {
                return Err("Axis out of bounds for self");
            }
            new_shape_self.remove(axis);
        }

        let mut sorted_axes_other = axes.1.to_vec();
        sorted_axes_other.sort_by(|a, b| b.cmp(a));
        for &axis in sorted_axes_other.iter() {
            if axis >= new_shape_other.len() {
                return Err("Axis out of bounds for other");
            }
            new_shape_other.remove(axis);
        }

        new_shape_self.extend(new_shape_other);

        let result_shape = new_shape_self;
        let result_data = vec![T::zero(); result_shape.iter().product()];
        let mut result = Tensor::from_vec(&result_data, result_shape.clone());

        for (i, value_self) in self.data.iter().enumerate() {
            let indices_self = Self::unravel_index(i, &self.shape);
            let indices_common: Vec<_> = axes.0.iter().map(|&axis| indices_self[axis]).collect();
            let indices_self_reduced: Vec<_> = indices_self
                .iter()
                .enumerate()
                .filter(|&(idx, _)| !axes.0.contains(&idx))
                .map(|(_, &val)| val)
                .collect();

            for (j, value_other) in other.data.iter().enumerate() {
                let indices_other = Self::unravel_index(j, &other.shape);
                let indices_common_other: Vec<_> =
                    axes.1.iter().map(|&axis| indices_other[axis]).collect();

                if indices_common == indices_common_other {
                    let indices_other_reduces: Vec<_> = indices_other
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _)| !axes.1.contains(&idx))
                        .map(|(_, &val)| val)
                        .collect();

                    let mut result_indices = indices_self_reduced.clone();
                    result_indices.extend(indices_other_reduces);

                    let result_index = Self::ravel_index(&result_indices, &result_shape);
                    result.data[result_index] += value_self.clone() * value_other.clone();
                }
            }
        }
        Ok(result)
    }
    // Helper function to unravel a flat index to a multidimensional index
    fn unravel_index(index: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = index;
        let mut indices = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            indices[i] = idx % shape[i];
            idx /= shape[i];
        }
        indices
    }

    // Helper function to ravel a multidimensional index to a flat index
    fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
        let mut index = 0;
        let mut factor = 1;
        for (i, &dim) in shape.iter().enumerate().rev() {
            index += indices[i] * factor;
            factor *= dim;
        }
        index
    }

    pub fn transpose(&self, axes: &[usize]) -> Result<Tensor<T>, &str> {
        let new_shape: Vec<usize>;
        let new_axes: Vec<usize>;

        if axes.is_empty() {
            // If axes are not provided, reverse the shape and create corresponding axes
            new_shape = self.shape.iter().rev().cloned().collect();
            new_axes = (0..self.shape.len()).rev().collect::<Vec<_>>();
        } else {
            if axes.len() != self.shape.len() {
                return Err("Axes dimensions must match tensor dimensions");
            }
            new_shape = axes.iter().map(|&axis| self.shape[axis]).collect();
            new_axes = axes.to_vec();
        }

        let mut new_data = vec![T::zero(); self.data.len()];

        let mut old_indices = vec![0; self.shape.len()];
        let mut new_indices = vec![0; self.shape.len()];
        let mut old_strides = vec![1; self.shape.len()];
        let mut new_strides = vec![1; self.shape.len()];

        for i in (1..self.shape.len()).rev() {
            old_strides[i - 1] = old_strides[i] * self.shape[i];
        }

        for i in (1..new_shape.len()).rev() {
            new_strides[i - 1] = new_strides[i] * new_shape[i];
        }

        for old_idx in 0..self.data.len() {
            let mut temp = old_idx;
            for i in 0..self.shape.len() {
                old_indices[i] = temp / old_strides[i];
                temp %= old_strides[i];
            }

            for i in 0..self.shape.len() {
                new_indices[i] = old_indices[new_axes[i]];
            }

            let mut new_idx = 0;
            for i in 0..self.shape.len() {
                new_idx += new_indices[i] * new_strides[i];
            }

            new_data[new_idx] = self.data[old_idx].clone();
        }

        Ok(Tensor {
            data: new_data,
            shape: new_shape,
        })
    }

    pub fn moveaxis(&self, source: &[i32], dest: &[i32]) -> Result<Tensor<T>, &str> {
        if source.len() != dest.len() {
            return Err("source and destination arguments must have the same number of elements");
        }

        let ndim = self.shape.len();

        let convert_index = |idx: i32| -> usize {
            if idx < 0 {
                (ndim as isize + idx as isize) as usize
            } else {
                idx as usize
            }
        };

        let source: Vec<usize> = source.iter().map(|&x| convert_index(x)).collect();
        let dest: Vec<usize> = dest.iter().map(|&x| convert_index(x)).collect();

        let mut order: Vec<usize> = (0..ndim).collect();

        // Remove the source indices from the order, starting from the highest index to avoid reindexing issues
        let mut temp_source = source.clone();
        temp_source.sort_by(|a, b| b.cmp(a));
        for &src in &temp_source {
            order.remove(src);
        }

        // Insert the source indices at the destination positions, starting from the lowest index
        let mut temp_dest = dest.clone();
        let mut temp_pairs: Vec<(usize, usize)> = temp_dest
            .iter()
            .cloned()
            .zip(source.iter().cloned())
            .collect();
        temp_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for &(dst, src) in &temp_pairs {
            order.insert(dst, src);
        }
        self.transpose(&order)
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: fmt::Debug + Clone + Add<Output = T> + Mul<Output = T> + AddAssign + Zero,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array(")?;
        self.print(f, &self.shape, &self.data)?;
        write!(f, ")")
    }
}
