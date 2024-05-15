use num_complex::Complex;

pub struct Tensor {
    pub data: Vec<Complex<f64>>,
    pub shape: Vec<usize>,
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

    pub fn print(&self) -> () {
        let size = self.shape.iter().product();
        for i in 0..size {
            println!("{:?}", self.data[i]);
        }
    }

    pub fn from_vec(vec: Vec<Complex<f64>>, shape: Vec<usize>) -> Self {
        Self {
            data: vec,
            shape
        }
    }

    // Get index with the given tensor indices
    pub fn get_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        let mut index = 0;
        let mut multiplier = 1;
        for i in (0..indices.len()).rev() {
            index += indices[i] * multiplier;
            multiplier *= self.shape[i];
        }
        index
    }

    // Access element at given indices
    pub fn get(&self, indices: &[usize]) -> Complex<f64> {
        let index = self.get_index(indices);
        self.data[index]
    }

    // Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: Complex<f64>) {
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

    // Method to perform tensor contraction
    pub fn contract(&self, other: &Tensor, axes: Option<(Vec<usize>, Vec<usize>)>, n: Option<usize>) -> Tensor {
        let (a_axes, b_axes) = if let Some((a_axes, b_axes)) = axes {
            (a_axes, b_axes)
        } else if let Some(n) = n {
            let a_ndim = self.shape.len();
            let b_ndim = other.shape.len();
            assert!(n <= a_ndim && n <= b_ndim, "N is larger than the number of dimensions in one of the tensors");
            ((a_ndim - n..a_ndim).collect(), (0..n).collect())
        } else {
            panic!("Either axes or n must be provided");
        };

        assert_eq!(a_axes.len(), b_axes.len(), "Axes lengths must match");

        // Calculate new shapes excluding the summed dimensions
        let mut new_shape_a: Vec<usize> = self.shape.clone();
        for &a_axis in &a_axes {
            new_shape_a[a_axis] = 1;
        }
        let new_shape_a: Vec<usize> = new_shape_a.into_iter().filter(|&dim| dim != 1).collect();

        let mut new_shape_b: Vec<usize> = other.shape.clone();
        for &b_axis in &b_axes {
            new_shape_b[b_axis] = 1;
        }
        let new_shape_b: Vec<usize> = new_shape_b.into_iter().filter(|&dim| dim != 1).collect();

        // Initialize result tensor shape
        let mut result_shape = new_shape_a.clone();
        result_shape.extend(new_shape_b.iter());

        // Perform the contraction
        let mut result_data = vec![Complex::new(0.0, 0.0); result_shape.iter().product()];

        let mut indices_a = vec![0; self.shape.len()];
        let mut indices_b = vec![0; other.shape.len()];

        for i in 0..self.data.len() {
            let mut temp_index = i;
            for j in (0..self.shape.len()).rev() {
                indices_a[j] = temp_index % self.shape[j];
                temp_index /= self.shape[j];
            }

            for j in 0..other.data.len() {
                let mut temp_index = j;
                for k in (0..other.shape.len()).rev() {
                    indices_b[k] = temp_index % other.shape[k];
                    temp_index /= other.shape[k];
                }

                let mut skip = false;
                for (&a_axis, &b_axis) in a_axes.iter().zip(b_axes.iter()) {
                    if indices_a[a_axis] != indices_b[b_axis] {
                        skip = true;
                        break;
                    }
                }

                if !skip {
                    let new_indices_a: Vec<usize> = indices_a.iter().enumerate()
                        .filter(|&(idx, _)| !a_axes.contains(&idx))
                        .map(|(_, &val)| val)
                        .collect();

                    let new_indices_b: Vec<usize> = indices_b.iter().enumerate()
                        .filter(|&(idx, _)| !b_axes.contains(&idx))
                        .map(|(_, &val)| val)
                        .collect();

                    let mut new_indices = new_indices_a.clone();
                    new_indices.extend(new_indices_b);

                    let mut result_index = 0;
                    let mut multiplier = 1;
                    for &idx in new_indices.iter().rev() {
                        result_index += idx * multiplier;
                        multiplier *= result_shape[new_indices.len() - 1];
                    }

                    result_data[result_index] += self.data[i] * other.data[j];
                }
            }
        }

        Tensor::from_vec(result_data, result_shape)
    }
}