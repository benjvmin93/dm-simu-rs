#[cfg(test)]
mod tests_dm {
    use num_complex::Complex;
    use mbqc::density_matrix::DensityMatrix;

    #[test]
    fn test_to_tensor_1_qubit() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(1);
        density_matrix.set(0, 0, Complex::new(1., 0.));
        density_matrix.set(0, 1, Complex::new(2., 0.));
        density_matrix.set(1, 0, Complex::new(3., 0.));
        density_matrix.set(1, 1, Complex::new(4., 0.));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 1]), Complex::new(2., 0.));
        assert_eq!(tensor.get(&[1, 0]), Complex::new(3., 0.));
        assert_eq!(tensor.get(&[1, 1]), Complex::new(4., 0.));
    }

    #[test]
    fn test_to_tensor_2_qubits_1() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(2);
        density_matrix.set(0, 0, Complex::new(1., 0.));
        density_matrix.set(0, 1, Complex::new(2., 0.));
        density_matrix.set(0, 2, Complex::new(3., 0.));
        density_matrix.set(0, 3, Complex::new(4., 0.));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2, 2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0, 0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 0, 0, 1]), Complex::new(2., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 0]), Complex::new(3., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 1]), Complex::new(4., 0.));
    }
    #[test]
    fn test_to_tensor_2_qubits_1() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(2);
        density_matrix.set(0, 0, Complex::new(1., 0.));
        density_matrix.set(0, 1, Complex::new(2., 0.));
        density_matrix.set(0, 2, Complex::new(3., 0.));
        density_matrix.set(0, 3, Complex::new(4., 0.));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2, 2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0, 0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 0, 0, 1]), Complex::new(2., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 0]), Complex::new(3., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 1]), Complex::new(4., 0.));
    }

    #[test]
    fn test_to_tensor_2_qubits_2() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(2);
        density_matrix.set(0, 0, Complex::new(1., 0.));
        density_matrix.set(0, 1, Complex::new(2., 0.));
        density_matrix.set(0, 2, Complex::new(3., 0.));
        density_matrix.set(0, 3, Complex::new(4., 0.));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2, 2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0, 0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 0, 0, 1]), Complex::new(2., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 0]), Complex::new(3., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 1]), Complex::new(4., 0.));
    }
}