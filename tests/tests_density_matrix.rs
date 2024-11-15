#[cfg(test)]
mod tests_dm { 
    use num_complex::Complex;
    use dm_simu_rs::density_matrix::{DensityMatrix, State};
    use dm_simu_rs::operators::{Operator, OneQubitOp, TwoQubitsOp};
    use dm_simu_rs::tensor::Tensor;
    use num_traits::pow;

    const TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_init_initial_state_zero() {
        let rho = DensityMatrix::new(1, State::ZERO);
        let expected_data = vec![
            Complex::ONE, Complex::ZERO,
            Complex::ZERO, Complex::ZERO
        ];

        assert_eq!(expected_data, rho.data.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_init_initial_state_plus() {
        let rho = DensityMatrix::new(1, State::PLUS);
        let expected_data = vec![
            Complex::new(0.5, 0.), Complex::new(0.5, 0.),
            Complex::new(0.5, 0.), Complex::new(0.5, 0.)
        ];

        assert_eq!(expected_data, rho.data.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_init_initial_state_one() {
        let rho = DensityMatrix::new(1, State::ONE);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ONE 
        ];

        assert_eq!(expected_data, rho.data.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_init_from_statevec_ket_0() {
        let rho = DensityMatrix::from_statevec(&[Complex::ONE, Complex::ZERO]).unwrap();
        let expected_data = &[
            Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_1() {
        let rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_00() {
        let rho = DensityMatrix::from_statevec(&[Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_01() {
        let rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_10() {
        let rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_11() {
        let rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(rho.data.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_one_qubit_evolve_single_i() {

        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::I), 0);

        let expected_data = &[Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)];
        assert_eq!(rho.data.data, expected_data);
    }   
    #[test]
    fn test_one_qubit_evolve_single_h() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = Tensor::from_vec(
            vec![Complex::new(0.5, 0.), Complex::new(0.5, 0.), Complex::new(0.5, 0.), Complex::new(0.5, 0.)],
            vec![2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_x() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = Tensor::from_vec(
            vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)],
            vec![2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_y() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = Tensor::from_vec(
            vec![Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)],
            vec![2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_z() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = Tensor::from_vec(
            vec![Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)],
            vec![2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_i() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::I), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
            ],
            vec![2, 2, 2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_h() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0.5, 0.), Complex::new(0., 0.), Complex::new(0.5, 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0.5, 0.), Complex::new(0., 0.), Complex::new(0.5, 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
            ],
            vec![2, 2, 2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_x() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
            ],
            vec![2, 2, 2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_y() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
            ],
            vec![2, 2, 2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_z() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
                Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
            ],
            vec![2, 2, 2, 2]
        );
        assert_eq!(rho.equals(DensityMatrix { data: expected_data, size: 2, nqubits: 1 }, TOLERANCE), true);
    }
    #[test]
    fn test_evolve_cx_ket00_1() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_cx_ket00_2() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_cx_ket01() {
        let mut rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.)]).unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_cx_ket10() {
        let mut rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.)]).unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_cx_ket11() {
        let mut rho = DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(1., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_cz_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CZ), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    
    #[test]
    fn test_evolve_swap_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.),
            Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.), Complex::new(0., 0.)
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    
    #[test]
    fn test_evolve_swap_ket001_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, 
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 1]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket001_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, 
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket100_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, 
            Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 0]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket100_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, 
            Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket111_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, 
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket111_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, 
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    fn test_evolve_swap_ket111_3() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, 
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE
        ]).unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
        let expected_data = vec![
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE,
        ];
        assert_eq!(expected_data, rho.data.data);
    }
    #[test]
    #[should_panic]
    fn test_evolve_single_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 5).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_single_wrong_operator() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single(&Operator::two_qubits(TwoQubitsOp::CZ), 0).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_similar_indices() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 0]).unwrap();
    }

    #[test]
    fn test_kron_1_qubit_zero_state() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        let other = DensityMatrix::new(1, State::ZERO);
        rho.tensor(&other);

        let expected_data: Vec<Complex<f64>> = vec![
            Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO
        ];

        assert_eq!(expected_data, rho.data.data);
    }

    #[test]
    fn test_kron_multiple_qubits_zero_state() {
        let mut rho = DensityMatrix::new(2, State::ZERO); // 2 qubits, |00⟩
        let other = DensityMatrix::new(2, State::ZERO); // 2 qubits, |00⟩
        rho.tensor(&other);

        let mut expected_data: Vec<Complex<f64>> = vec![Complex::ZERO; 16 * 16];
        expected_data[0] = Complex::ONE;

        assert_eq!(expected_data, rho.data.data);
    }

    #[test]
    fn test_kron_plus_state() {
        let mut rho = DensityMatrix::new(1, State::PLUS); // 1 qubit, |+⟩ state
        let other = DensityMatrix::new(1, State::PLUS); // 1 qubit, |+⟩ state
        rho.tensor(&other);
    
        let expected_data: Vec<Complex<f64>> = vec![Complex::new(0.25, 0.); 16];
    
        assert_eq!(expected_data, rho.data.data);
    }

    #[test]
    fn test_kron_zero_and_one_state() {
        let mut rho = DensityMatrix::new(1, State::ZERO); // 1 qubit, |0⟩ state
        let other = DensityMatrix::new(1, State::ONE); // 1 qubit, |1⟩ state
        rho.tensor(&other);
    
        let mut expected_data: Vec<Complex<f64>> = vec![Complex::ZERO; 16];
        expected_data[5] = Complex::ONE;
    
        assert_eq!(expected_data, rho.data.data);
    }

    #[test]
    fn test_kron_two_qubits_plus_state() {
        let mut rho = DensityMatrix::new(2, State::PLUS); // 2 qubits, |+⟩ state
        let other = DensityMatrix::new(2, State::PLUS); // 2 qubits, |+⟩ state
        rho.tensor(&other);
    
        let expected_data: Vec<Complex<f64>> = vec![Complex::new(0.0625, 0.); 16 * 16];
    
        assert_eq!(expected_data, rho.data.data);
    }
}