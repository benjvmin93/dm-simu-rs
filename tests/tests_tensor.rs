#[cfg(test)]
mod tests_tensor {
    use num_complex::Complex;
    use mbqc::tensor::Tensor;

    #[test]
    fn test_tensor_creation() {
        let shape = vec![2, 3];
        let tensor = Tensor::new(shape.clone());
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.data.len(), 6); // 2 * 3 = 6 elements
        for &value in &tensor.data {
            assert_eq!(value, Complex::new(0.0, 0.0)); // All elements initialized to zero
        }
    }

    #[test]
    fn test_tensor_get_set() {
        let shape = vec![2, 2];
        let mut tensor = Tensor::new(shape.clone());

        // Set elements
        tensor.set(&[0, 0], Complex::new(1.0, 0.0));
        tensor.set(&[0, 1], Complex::new(2.0, 0.0));
        tensor.set(&[1, 0], Complex::new(3.0, 0.0));
        tensor.set(&[1, 1], Complex::new(4.0, 0.0));

        // Get elements
        assert_eq!(tensor.get(&[0, 0]), Complex::new(1.0, 0.0));
        assert_eq!(tensor.get(&[0, 1]), Complex::new(2.0, 0.0));
        assert_eq!(tensor.get(&[1, 0]), Complex::new(3.0, 0.));
        assert_eq!(tensor.get(&[1, 1]), Complex::new(4.0, 0.0));
    }

    #[test]
    fn test_tensor_addition() {
        let shape = vec![2, 2];
        let tensor1 = Tensor {
            data: vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.), Complex::new(3., 0.), Complex::new(4., 0.)],
            shape: shape.clone(),
        };
        let tensor2 = Tensor {
            data: vec![Complex::new(5.0, 0.), Complex::new(6., 0.), Complex::new(7., 0.), Complex::new(8., 0.)],
            shape: shape.clone(),
        };

        let result = tensor1.add(&tensor2);
        assert_eq!(result.data, vec![Complex::new(6., 0.), Complex::new(8., 0.), Complex::new(10., 0.), Complex::new(12., 0.)]);
    }

    #[test]
    fn test_tensor_multiplication() {
        let shape = vec![2, 2];
        let tensor1 = Tensor {
            data: vec![Complex::new(1., 0.), Complex::new(2., 0.), Complex::new(3., 0.), Complex::new(4., 0.)],
            shape: shape.clone(),
        };
        let tensor2 = Tensor {
            data: vec![Complex::new(5., 0.), Complex::new(6., 0.), Complex::new(7., 0.), Complex::new(8., 0.)],
            shape: shape.clone(),
        };

        let result = tensor1.multiply(&tensor2);
        assert_eq!(result.data, vec![Complex::new(5., 0.), Complex::new(12., 0.), Complex::new(21., 0.), Complex::new(32., 0.)]);
    }

    #[test]
    fn test_tensor_product() {
        // Create the first tensor: [1, 2, 3]
        let tensor1 = Tensor::from_vec(vec![Complex::new(1., 0.), Complex::new(2., 0.), Complex::new(3., 0.)], vec![3]);

        // Create the second tensor: [4, 5]
        let tensor2 = Tensor::from_vec(vec![Complex::new(4., 0.), Complex::new(5., 0.)], vec![2]);

        // Calculate the tensor product
        let result_tensor = tensor1.tensor_product(&tensor2);

        // Expected result:
        // Shape: [3, 2]
        // Data: [1*4, 1*5, 2*4, 2*5, 3*4, 3*5] => [4, 5, 8, 10, 12, 15]
        assert_eq!(result_tensor.shape, vec![3, 2]);
        assert_eq!(result_tensor.data, vec![Complex::new(4., 0.), Complex::new(5., 0.), Complex::new(8., 0.),  Complex::new(10., 0.), Complex::new(12., 0.), Complex::new(15., 0.)]);
    }
}