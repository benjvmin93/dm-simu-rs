use num_traits::ToPrimitive;
use core::fmt;
use num_traits::Zero;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Zero + Clone + Mul<Output = T> + Add<Output = T> + AddAssign + Sized + std::fmt::Debug,
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
        assert_eq!(
            vec.len(),
            shape.iter().product(),
            "Vector length {} does not match the given tensor shape {:?}",
            vec.len(),
            shape
        );
        Self { data: vec, shape }
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
    pub fn add(&self, other: &Tensor<T>) -> Self
    where
        T: Send + Sync + Clone + Zero + Mul<Output = T> + std::ops::AddAssign,
    {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(&self.shape);
        result.data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        result
    }

    // Perform tensor multiplication (element-wise)
    pub fn multiply(&self, other: &Tensor<T>) -> Self
    where
        T: Send + Sync + Clone + Zero + Mul<Output = T> + std::ops::AddAssign,
    {
        assert_eq!(self.shape, other.shape);
        let mut result = Self::new(&self.shape);
        result.data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();
        result
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    
    // Method to compute the tensor product of two tensors
    pub fn product(&self, other: &Tensor<T>) -> Result<Tensor<T>, String>
    where
        T: Copy + Send + Sync + Sized + std::ops::Mul<Output = T>,
    {

        // Only works for 2^n sized squared matrix.
        let self_size: usize = self.size();
        let other_size: usize = other.size();

        if self_size & self_size - 1 != 0 {
            return Err("Self tensor is not a power of two matrix".to_string());
        } else if other_size & other_size - 1 != 0 {
            return Err("Other tensor is not a power of two matrix".to_string());
        }

        let self_cols = (self_size as f64).sqrt();
        let other_cols = (other_size as f64).sqrt();

        // println!("self_cols : {}", self_cols);
        // println!("other_cols : {}", other_cols);
        if self_cols.to_isize() == None {
            return Err("Self tensor is not a squared matrix".to_string());
        } else if other_cols.to_isize() == None {
            return Err("Other tensor is not a squared matrix".to_string());
        }


        // Calculate the shape of the resulting tensor
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .chain(other.shape.iter())
            .cloned()
            .collect();

        println!("tensor product: new_shape = {:?}", new_shape);

        let mut new_data = vec![T::zero(); self_size * other_size];
        let new_data_size = (new_data.len() as f64).sqrt();

        let self_cols = self_cols as usize;
        let other_cols = other_cols as usize;
        self.data.clone().into_iter().enumerate().for_each(|(i, x)| {
            // Compute (i, j) coef of self_data
            let self_indices = (i / self_cols, i % self_cols);
            // println!("self_mat_idx() for {} : {:?}\n==================", i, self_indices);
            for (j, &y) in other.data.iter().enumerate() {
                // Compute (i, j) coef of other_data
                let other_indices = (j / other_cols, j % other_cols);
                // println!("other_mat_idx for {} : {:?}", j, other_indices);
                
                let data_i = self_indices.0 * other_cols + other_indices.0;
                let data_j = self_indices.1 * other_cols + other_indices.1;
                let data_idx = data_i * new_data_size as usize + data_j;
                
                // println!("new_data idx : {}", data_idx);

                new_data[data_idx] = x * y;
            }
        });

        // Calculate the data of the resulting tensor
        // Trying to compute it with Rust methods but doesn't validate all test suite
        /*
        let new_data_size = self_size * other_size;
        let self_cols = self_cols as usize;
        let other_cols = other_cols as usize;

        println!("tensor product: new_data size = {}", new_data_size);
        let new_data = (0..new_data_size).into_par_iter().map(|i| {
            let data_i = i / new_data_size;
            let data_j = i % new_data_size;

            let self_indices = (data_i / other_cols, data_j / other_cols);
            let other_indices = (data_i % other_cols, data_j % other_cols);

            let self_idx = self_indices.0 * self_cols;
            let other_idx = other_indices.0 * other_cols;

            self.data[self_idx] * other.data[other_idx]
        }).collect();*/

        Ok(Tensor::from_vec(new_data, new_shape))
    }

    pub fn tensordot(
        &self,
        other: &Tensor<T>,
        axes: (&[usize], &[usize]),
    ) -> Result<Tensor<T>, &str>
    where
        T: Send + Sync + Clone + Zero + Mul<Output = T> + std::ops::AddAssign,
    {
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
        let result_size = result_shape.iter().product();
        let mut result_data = vec![T::zero(); result_size];

        let common_shape: Vec<_> = axes.0.iter().map(|&axis| self.shape[axis]).collect();

        result_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(result_i, result_value)| {
                let result_indices = Self::unravel_index(result_i, &result_shape);

                let (indices_self_reduced, indices_other_reduced): (Vec<usize>, Vec<usize>) = {
                    let mut self_indices = Vec::new();
                    let mut other_indices = Vec::new();
                    let mut axis_iter = result_indices.iter();

                    for i in 0..self.shape.len() {
                        if !axes.0.contains(&i) {
                            self_indices.push(*axis_iter.next().unwrap());
                        }
                    }

                    for i in 0..other.shape.len() {
                        if !axes.1.contains(&i) {
                            other_indices.push(*axis_iter.next().unwrap());
                        }
                    }

                    (self_indices, other_indices)
                };

                let mut temp_sum = T::zero();
                let mut common_indices = vec![0; common_shape.len()];
                loop {
                    let self_full_indices = {
                        let mut indices = vec![0; self.shape.len()];
                        let mut self_reduced_iter = indices_self_reduced.iter();
                        for (i, index) in indices.iter_mut().enumerate() {
                            if let Some(pos) = axes.0.iter().position(|&a| a == i) {
                                *index = common_indices[pos];
                            } else {
                                *index = *self_reduced_iter.next().unwrap();
                            }
                        }
                        indices
                    }; 

                    let other_full_indices: Vec<_> = {
                        let mut indices = vec![0; other.shape.len()];
                        let mut other_reduced_iter = indices_other_reduced.iter();
                        for (i, index) in indices.iter_mut().enumerate() {
                            if let Some(pos) = axes.1.iter().position(|&a| a == i) {
                                *index = common_indices[pos];
                            } else {
                                *index = *other_reduced_iter.next().unwrap();
                            }
                        }
                        indices
                    };

                    let self_index = Self::ravel_index(&self_full_indices, &self.shape);
                    let other_index = Self::ravel_index(&other_full_indices, &other.shape);

                    temp_sum += self.data[self_index].clone() * other.data[other_index].clone();

                    if !Self::increment_indices(&mut common_indices, &common_shape) {
                        break;
                    }
                }
                *result_value = temp_sum;
            });

        Ok(Tensor {
            data: result_data,
            shape: result_shape,
        })
    }

    fn increment_indices(indices: &mut [usize], shape: &[usize]) -> bool {
        for (i, dim) in shape.iter().enumerate().rev() {
            if indices[i] + 1 < *dim {
                indices[i] += 1;
                return true;
            } else {
                indices[i] = 0;
            }
        }
        false
    }

    // Helper function to unravel a flat index to a multidimensional index
    fn unravel_index(index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remainder = index;
        for i in (0..shape.len()).rev() {
            indices[i] = remainder % shape[i];
            remainder /= shape[i];
        }
        indices
    }

    // Helper function to ravel a multidimensional index to a flat index
    fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
        indices
            .iter()
            .zip(shape.iter())
            .fold(0, |acc, (index, dim)| acc * dim + index)
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
        if self.shape.len() == 1 {
            write!(f, "[")?;
            for (i, item) in self.data.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", item)?;
            }
            write!(f, "]\n")?;
        } else {
            let chunk_size: usize = self.shape[1..].iter().product();
            for (i, chunk) in self.data.chunks(chunk_size).enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                self.print(f, &self.shape[1..], chunk)?
            }
            write!(f, "]\n")?;
        }
        write!(f, ")")
    }
}
