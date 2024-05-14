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

    // Access element at given indices
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
}