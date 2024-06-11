#[cfg(test)]
mod tests_dm {
    use num_complex::Complex;
    use mbqc::density_matrix::{DensityMatrix, State};
    use mbqc::tools::tensor_to_dm;
    use mbqc::operators::{OneQubitOp, TwoQubitsOp};

    const TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_init_from_statevec_ket_0() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(1., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_1() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_00() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_01() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_10() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_11() {
        let rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
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
    fn test_tensor_to_dm_from_1_to_7_qubits_zero() {
        for i in 1..8 {
            let dm_first = DensityMatrix::new(i, Some(State::ZERO));
            let dm_second = tensor_to_dm(dm_first.to_tensor());
            assert_eq!(dm_first.size, dm_second.size);
            assert_eq!(dm_first.nqubits, dm_second.nqubits);
            assert_eq!(dm_first.data, dm_second.data);
        }
    }
    #[test]
    fn test_tensor_to_dm_from_1_to_7_qubits_plus() {
        for i in 1..8 {
            let dm_first = DensityMatrix::new(i, Some(State::PLUS));
            let dm_second = tensor_to_dm(dm_first.to_tensor());
            assert_eq!(dm_first.size, dm_second.size);
            assert_eq!(dm_first.nqubits, dm_second.nqubits);
            assert_eq!(dm_first.data, dm_second.data);
        }
    }
    #[test]
    fn test_one_qubit_evolve_single_i() {

        let mut rho = DensityMatrix::new(1, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::I, 0);

        let expected_data = vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)];
        assert_eq!(rho.data, expected_data);
    }
    #[test]
    fn test_one_qubit_evolve_single_h() {
        let mut rho = DensityMatrix::new(1, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::H, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![Complex::new(0.5, 0.), Complex::new(0.5, 0.), Complex::new(0.5, 0.), Complex::new(0.5, 0.)];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_x() {
        let mut rho = DensityMatrix::new(1, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::X, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_y() {
        let mut rho = DensityMatrix::new(1, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::Y, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_z() {
        let mut rho = DensityMatrix::new(1, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::Z, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_i() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::I, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_h() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::H, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![
            Complex::new(0.5, 0.), Complex::new(0., 0.), Complex::new(0.5, 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0.5, 0.), Complex::new(0., 0.), Complex::new(0.5, 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_x() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::X, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_y() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::Y, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_z() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve_single(OneQubitOp::Z, 0);
        println!("tolerance = {}", TOLERANCE);
        println!("rho after evolve single:\n{}", rho);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_evolve_cx_ket00_1() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve(TwoQubitsOp::CX, &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cx_ket00_2() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve(TwoQubitsOp::CX, &[1, 0]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cx_ket01() {
        let mut rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        rho.evolve(TwoQubitsOp::CX, &[1, 0]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cx_ket10() {
        let mut rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.)]).unwrap();
        rho.evolve(TwoQubitsOp::CX, &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cx_ket11() {
        let mut rho = DensityMatrix::from_statevec(vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        rho.evolve(TwoQubitsOp::CX, &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cz_ket00() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve(TwoQubitsOp::CZ, &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }
    
    #[test]
    fn test_evolve_swap_ket00() {
        let mut rho = DensityMatrix::new(2, Some(State::ZERO));
        rho.evolve(TwoQubitsOp::SWAP, &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data);
    }

}