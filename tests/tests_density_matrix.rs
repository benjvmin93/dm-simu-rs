#[cfg(test)]
mod tests_dm {
    use std::f64::consts::FRAC_1_SQRT_2;

    use dm_simu_rs::density_matrix::{DensityMatrix, State};
    use dm_simu_rs::operators::{OneQubitOp, Operator, TwoQubitsOp};
    use dm_simu_rs::tensor::Tensor;
    use num_complex::Complex;
    use num_traits::{One, Zero};

    const TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_init_from_statevec_ket_0() {
        let rho = DensityMatrix::from_statevec(&[Complex::ONE, Complex::ZERO]).unwrap();
        let expected_data = &[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_1() {
        let rho =
            DensityMatrix::from_statevec(&[Complex::new(0., 0.), Complex::new(1., 0.)]).unwrap();
        let expected_data = &[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }
    #[test]
    fn test_init_from_statevec_ket_00() {
        let rho = DensityMatrix::from_statevec(&[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        let expected_data = &[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_01() {
        let rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        let expected_data = &[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_10() {
        let rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        let expected_data = &[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_init_from_statevec_ket_11() {
        let rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ])
        .unwrap();
        let expected_data = &[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_one_qubit_evolve_single_i() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::I), 0);

        let expected_data = &[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.tensor.data, expected_data);
    }
    #[test]
    fn test_one_qubit_evolve_single_h() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0.5, 0.),
                Complex::new(0.5, 0.),
                Complex::new(0.5, 0.),
                Complex::new(0.5, 0.),
            ],
            vec![2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_one_qubit_evolve_single_x() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(1., 0.),
            ],
            vec![2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_one_qubit_evolve_single_y() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(1., 0.),
            ],
            vec![2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_one_qubit_evolve_single_z() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_two_qubits_evolve_single_i() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::I), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_two_qubits_evolve_single_h() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0.5, 0.),
                Complex::new(0., 0.),
                Complex::new(0.5, 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0.5, 0.),
                Complex::new(0., 0.),
                Complex::new(0.5, 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_two_qubits_evolve_single_x() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_two_qubits_evolve_single_y() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_two_qubits_evolve_single_z() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = Tensor::from_vec(
            vec![
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            vec![2, 2, 2, 2],
        );
        assert_eq!(
            rho.equals(
                DensityMatrix {
                    tensor: expected_data,
                    size: 2,
                    nqubits: 1
                },
                TOLERANCE
            ),
            true
        );
    }
    #[test]
    fn test_evolve_cx_ket00_1() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_cx_ket00_2() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
        let expected_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_cx_ket01() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
        let expected_data = vec![
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_cx_ket10() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_cx_ket11() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ])
        .unwrap();
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
        let expected_data = vec![
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_cz_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CZ), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }

    #[test]
    fn test_evolve_swap_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
        let expected_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }

    #[test]
    fn test_evolve_swap_ket001_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 1]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket001_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket100_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 0]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket100_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket111_1() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket111_2() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    fn test_evolve_swap_ket111_3() {
        let mut rho = DensityMatrix::from_statevec(&[
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ])
        .unwrap();

        rho.evolve(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
        let expected_data = vec![
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ];
        assert_eq!(expected_data, rho.tensor.data);
    }
    #[test]
    #[should_panic]
    fn test_evolve_single_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single(&Operator::one_qubit(OneQubitOp::X), 5)
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_single_wrong_operator() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single(&Operator::two_qubits(TwoQubitsOp::CZ), 0)
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 3])
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evolve_similar_indices() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 0])
            .unwrap();
    }

    #[test]
    fn test_normalize_all_equal() {
        let mut dm = DensityMatrix::from_statevec(&[
            Complex::one(),
            Complex::one(),
            Complex::one(),
            Complex::one(),
        ])
        .unwrap();
        dm.normalize();
        assert_eq!(dm.tensor.data, vec![Complex::new(0.25, 0.); 16]);
    }

    #[test]
    fn test_normalize_with_zero() {
        let mut dm = DensityMatrix::from_statevec(&vec![Complex::zero(); 4]).unwrap();
        dm.normalize();
        assert_eq!(dm.tensor.data, vec![Complex::zero(); 16]);
    }

    #[test]
    fn test_normalize_with_negative_elements() {
        let mut dm =
            DensityMatrix::from_statevec(&vec![Complex::new(-1., 0.), Complex::new(-2., 0.)])
                .unwrap();
        dm.normalize();
        let mut expected = vec![
            Complex::new(0.2, 0.),
            Complex::new(0.4, 0.),
            Complex::new(0.4, 0.),
            Complex::new(0.8, 0.),
        ];
        assert_eq!(dm.tensor.data, expected);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut dm = DensityMatrix::from_statevec(&vec![
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();
        dm.normalize();
        let mut expected = vec![Complex::ZERO; 16];
        expected[0] = Complex::ONE;
        assert_eq!(dm.tensor.data, expected);
    }

    #[test]
    fn test_normalize_large_numbers() {
        let mut dm =
            DensityMatrix::from_statevec(&vec![Complex::new(1000., 0.), Complex::new(2000., 0.)])
                .unwrap();
        println!("{}", dm);
        dm.normalize();
        let expected_data = vec![
            Complex::new(0.2, 0.),
            Complex::new(0.4, 0.),
            Complex::new(0.4, 0.),
            Complex::new(0.8, 0.),
        ];
        let t = Tensor::from_vec(expected_data, vec![2, 2]);
        let expected = DensityMatrix::from_tensor(t).unwrap();
        println!("{}", dm);
        println!("{}", expected);
        assert!(dm.equals(expected, TOLERANCE));
    }

    #[test]
    #[should_panic]
    fn test_ptrace_fail_1() {
        let mut dm = DensityMatrix::new(0, State::ZERO);
        dm.ptrace(&[0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_ptrace_fail_2() {
        let mut dm = DensityMatrix::new(2, State::ZERO);
        dm.ptrace(&[2]).unwrap();
    }

    #[test]
    fn test_ptrace_1() {
        let mut dm = DensityMatrix::from_statevec(&[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();
        dm.ptrace(&[0]).unwrap();
        let expected = vec![
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ];
        let expected_dm =
            DensityMatrix::from_tensor(Tensor::from_vec(expected, vec![2; 2])).unwrap();
        assert_eq!(dm.nqubits, 1);
        assert!(dm.equals(expected_dm, TOLERANCE));
    }

    #[test]
    fn test_ptrace_2() {
        let mut dm = DensityMatrix::from_statevec(&[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();
        dm.ptrace(&[1]).unwrap();
        let expected = vec![Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO];
        let expected_dm =
            DensityMatrix::from_tensor(Tensor::from_vec(expected, vec![2; 2])).unwrap();
        assert_eq!(dm.nqubits, 1);
        assert!(dm.equals(expected_dm, TOLERANCE));
    }

    #[test]
    fn test_ptrace_3() {
        let mut dm = DensityMatrix::from_statevec(&[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();
        dm.ptrace(&[0, 1]).unwrap();
        let expected = vec![Complex::ONE];
        let expected_dm = DensityMatrix {
            tensor: Tensor {
                data: expected,
                shape: vec![],
            },
            size: 0,
            nqubits: 0,
        };
        assert_eq!(dm.nqubits, 0);
        assert!(dm.equals(expected_dm, TOLERANCE));
    }

    #[test]
    fn test_ptrace_4() {
        let mut dm = DensityMatrix::new(4, State::PLUS);
        dm.ptrace(&[0, 1, 2]).unwrap();
        let expected = vec![
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ];
        assert_eq!(dm.nqubits, 1);
        assert_eq!(dm.tensor.data, expected);
    }

    #[test]
    fn test_ptrace_5() {
        let mut dm = DensityMatrix::new(4, State::ZERO);
        dm.ptrace(&[0, 1, 2, 3]).unwrap();
        let expected = vec![Complex::ONE];
        assert_eq!(dm.nqubits, 0);
        assert_eq!(dm.tensor.data, expected);
    }

    #[test]
    fn test_ptrace_6() {
        let frac_1_sqrt_3 = 1. / f64::sqrt(3_f64);
        let sv = vec![
            Complex::ZERO,
            Complex::new(frac_1_sqrt_3, 0.),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::new(f64::sqrt(2_f64) * frac_1_sqrt_3, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ];
        let mut dm = DensityMatrix::from_statevec(&sv).unwrap();
        dm.ptrace(&[2]).unwrap();
        let expected = vec![
            Complex::new(1. / 3., 0.),
            Complex::ZERO,
            Complex::new(2_f64.sqrt() / 3., 0.),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::new(2_f64.sqrt() / 3., 0.),
            Complex::ZERO,
            Complex::new(2. / 3., 0.),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];

        assert!(dm.equals(
            DensityMatrix {
                tensor: Tensor {
                    data: expected,
                    shape: vec![2; 2 * 2]
                },
                size: 4,
                nqubits: 2
            },
            TOLERANCE
        ));
    }

    #[test]
    fn test_ptrace_7() {
        let mut dm = DensityMatrix::new(3, State::PLUS);
        dm.ptrace(&[1, 2]).unwrap();
        let expected = vec![
            Complex::new(0.5, 0.), Complex::new(0.5, 0.),
            Complex::new(0.5, 0.), Complex::new(0.5, 0.)
        ];
        assert_eq!(dm.tensor.data, expected);
    }

    #[test]
    fn test_kron_simple() {
        let mut dm_1 = DensityMatrix::new(1, State::ZERO);
        let dm_2 = DensityMatrix::new(1, State::ZERO);
        println!("dm1 = {}", dm_1);
        println!("dm2 = {}", dm_2);

        let res_dm = dm_1.tensor(&dm_2);
        println!("res = {}", res_dm);
        assert_eq!(res_dm.nqubits, 2);
        assert_eq!(res_dm.tensor.shape, vec![2, 2, 2, 2]);
        assert_eq!(res_dm.tensor.data, vec![
            Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
            Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO,
        ]);
    }

    #[test]
    fn test_kron_different_size() {
        let mut dm_1 = DensityMatrix::new(0, State::ZERO);
        let dm_2 = DensityMatrix::new(4, State::ZERO);

        let res_dm = dm_1.tensor(&dm_2);
        println!("{}", res_dm);
        assert_eq!(res_dm.nqubits, 4);
        assert_eq!(res_dm.tensor.shape, vec![2, 2, 2, 2, 2, 2, 2, 2]);
        let size = 2_u32.pow(res_dm.nqubits as u32);
        assert_eq!(res_dm.tensor.data[0], Complex::ONE);
        for i in 1..(size * size) {
            assert_eq!(res_dm.tensor.data[i as usize], Complex::ZERO);
        }
    }

    #[test]
    fn test_kron_different_states() {
        let dm_1 = DensityMatrix::new(1, State::ZERO);
        let dm_2 = DensityMatrix::new(2, State::PLUS);
        println!("dm1 = {}", dm_1);
        println!("dm2 = {}", dm_2);
        let res_dm = dm_1.tensor(&dm_2);
        println!("{}", res_dm);
        assert_eq!(res_dm.nqubits, 3);
        assert_eq!(res_dm.tensor.shape, vec![2, 2, 2, 2, 2, 2]);
        let size = 2_u32.pow(res_dm.nqubits as u32);
        for i in 0..4 {
            for j in 0..4 {
                println!("i = {}, j = {}", i, j);
                assert_eq!(res_dm.tensor.data[i as usize * size as usize + j as usize], Complex::new(0.25, 0.));
            }
        }
        for i in 4..size {
            for j in 4..size {
                println!("i = {}, j = {}", i, j);
                assert_eq!(res_dm.tensor.data[i as usize * size as usize + j as usize], Complex::ZERO);
            }
        }
    }

    #[test]
    fn test_kron_plus_plus() {
        let dm_1 = DensityMatrix::new(1, State::PLUS);
        let dm_2 = DensityMatrix::new(1, State::PLUS);
        println!("dm1 = {}", dm_1);
        println!("dm2 = {}", dm_2);
        let res_dm = dm_1.tensor(&dm_2);
        println!("res = {}", res_dm);
        assert_eq!(res_dm.nqubits, 2);
        assert_eq!(res_dm.tensor.shape, vec![2, 2, 2, 2]);
        for &data in res_dm.tensor.data.iter() {
            assert_eq!(data, Complex::new(0.25, 0.));
        }
    }
}
