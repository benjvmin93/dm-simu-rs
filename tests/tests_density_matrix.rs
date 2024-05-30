#[cfg(test)]
mod tests_dm {
    use num_complex::Complex;
    use mbqc::density_matrix::{DensityMatrix, State};
    use mbqc::tools::tensor_to_dm;

    #[test]
    fn test_dm_to_tensor_1_qubit() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(1, None);
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
    fn test_dm_to_tensor_1_qubit_ket_0() {
        // Create a sample density matrix
        let density_matrix = DensityMatrix::new(1, Some(State::ZERO));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 1]), Complex::new(0., 0.));
    }

    #[test]
    fn test_dm_to_tensor_1_qubit_ket_plus() {
        // Create a sample density matrix
        let density_matrix = DensityMatrix::new(1, Some(State::PLUS));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0]), Complex::new(0.5, 0.));
        assert_eq!(tensor.get(&[0, 1]), Complex::new(0.5, 0.));
        assert_eq!(tensor.get(&[1, 0]), Complex::new(0.5, 0.));
        assert_eq!(tensor.get(&[1, 1]), Complex::new(0.5, 0.));
    }

    #[test]
    fn test_dm_to_tensor_2_qubits_ket_plus() {
        // Create a sample density matrix
        let density_matrix = DensityMatrix::new(2, Some(State::PLUS));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2, 2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0, 0, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 0, 0, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 1, 0, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 1, 0, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 1, 1, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[0, 1, 1, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 0, 0, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 0, 0, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 0, 1, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 0, 1, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 1, 0, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 1, 0, 1]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 1, 1, 0]), Complex::new(0.25, 0.));
        assert_eq!(tensor.get(&[1, 1, 1, 1]), Complex::new(0.25, 0.));
    }

    #[test]
    fn test_dm_to_tensor_2_qubits_ket_0() {
        // Create a sample density matrix
        let density_matrix = DensityMatrix::new(2, Some(State::ZERO));
        // Convert the density matrix to a tensor
        let tensor = density_matrix.to_tensor();

        // Verify the shape of the tensor
        assert_eq!(tensor.shape, vec![2, 2, 2, 2]); // Shape should be ((2,) * 2 * nqubits)

        // Verify the values in the tensor
        assert_eq!(tensor.get(&[0, 0, 0, 0]), Complex::new(1., 0.));
        assert_eq!(tensor.get(&[0, 0, 0, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 0, 1, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 1, 0, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 1, 0, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 1, 1, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[0, 1, 1, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 0, 0, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 0, 0, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 0, 1, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 0, 1, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 1, 0, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 1, 0, 1]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 1, 1, 0]), Complex::new(0., 0.));
        assert_eq!(tensor.get(&[1, 1, 1, 1]), Complex::new(0., 0.));
    }

    #[test]
    fn test_dm_to_tensor_2_qubits_1() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(2, None);
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
    fn test_dm_to_tensor_2_qubits_2() {
        // Create a sample density matrix
        let mut density_matrix = DensityMatrix::new(2, None);
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
    fn test_tensor_to_dm_1_qubit() {
        let dm_first = DensityMatrix::new(1, Some(State::ZERO));
        let dm_second = tensor_to_dm(dm_first.to_tensor());
        assert_eq!(dm_first.size, dm_second.size);
        assert_eq!(dm_first.nqubits, dm_second.nqubits);
        assert_eq!(dm_first.data, dm_second.data);
    }

    fn test_tensor_to_dm_2_qubits() {
        let dm_first = DensityMatrix::new(2, Some(State::ZERO));
        let dm_second = tensor_to_dm(dm_first.to_tensor());
        assert_eq!(dm_first.size, dm_second.size);
        assert_eq!(dm_first.nqubits, dm_second.nqubits);
        assert_eq!(dm_first.data, dm_second.data);
    }


}