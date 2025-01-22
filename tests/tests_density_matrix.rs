#[cfg(test)]
mod tests_dm {
    use dm_simu_rs::density_matrix::{DensityMatrix, State};
    use dm_simu_rs::operators::{OneQubitOp, Operator, TwoQubitsOp};
    use num_complex::Complex;
    use num_traits::PrimInt;
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

    const TOLERANCE: f64 = 1e-15;

    #[test]
    fn test_init_initial_state_zero() {
        let rho = DensityMatrix::new(1, State::ZERO);
        let expected_data = vec![Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO];

        assert_eq!(expected_data, rho.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_init_initial_state_plus() {
        let rho = DensityMatrix::new(1, State::PLUS);
        let expected_data = vec![
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ];

        assert_eq!(expected_data, rho.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_density_matrix_minus_state() {
        let nqubits = 3; // Test with 3 qubits (can try with other numbers too)
        let rho = DensityMatrix::new(nqubits, State::MINUS);

        println!("rho = {}", rho);
        let size = 1 << nqubits; // 2^nqubits
        let factor = 1.0 / (size as f64); // Normalization factor

        for i in 0..size {
            for j in 0..size {
                let expected_value = if (i ^ j).count_ones() % 2 == 0 {
                    Complex::new(factor, 0.0)
                } else {
                    Complex::new(-factor, 0.0)
                };

                let value = rho.data[i * size + j];
                assert_eq!(
                    expected_value, value,
                    "Mismatch at indices ({}, {}): expected {:?}, got {:?}",
                    i, j, expected_value, value
                );
            }
        }

        // Ensure trace is 1
        let trace = rho.trace();
        assert_eq!(
            trace,
            Complex::new(1.0, 0.0),
            "Trace of the density matrix should be 1, got {:?}",
            trace
        );
    }

    #[test]
    fn test_init_initial_state_one() {
        let rho = DensityMatrix::new(1, State::ONE);
        let expected_data = vec![Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE];

        assert_eq!(expected_data, rho.data);
        assert_eq!(rho.nqubits, 1);
        assert_eq!(rho.size, 2);
    }

    #[test]
    fn test_init_from_statevec_ket_0() {
        let rho = DensityMatrix::from_statevec(&[Complex::ONE, Complex::ZERO]).unwrap();
        let expected_data = &[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.data, expected_data);
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
        assert_eq!(rho.data, expected_data);
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
        assert_eq!(rho.data, expected_data);
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
        assert_eq!(rho.data, expected_data);
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
        assert_eq!(rho.data, expected_data);
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
        assert_eq!(rho.data, expected_data);
        assert_eq!(rho.nqubits, 2);
        assert_eq!(rho.size, 4);
    }
    #[test]
    fn test_one_qubit_evolve_single_i() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::I), 0);

        let expected_data = &[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];
        assert_eq!(rho.data, expected_data);
    }
    #[test]
    fn test_one_qubit_evolve_single_h() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = DensityMatrix::from_vec(&[
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_x() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = DensityMatrix::from_vec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_y() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = DensityMatrix::from_vec(&[
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_one_qubit_evolve_single_z() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = DensityMatrix::from_vec(&[
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }

    fn compare_matrices(m1: &[Complex<f64>], m2: &[Complex<f64>], tol: f64) -> bool {
        m1.iter().zip(m2.iter()).all(|(a, b)| (a - b).norm() < tol)
    }

    #[test]

    fn test_one_qubit_evolve_single_i_2() {
        let mut rho = DensityMatrix::new(1, State::PLUS);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::I), 0);
        println!("{rho:}");

        let expected_data = DensityMatrix::from_vec(&[
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ])
        .unwrap();

        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }

    #[test]
    fn test_two_qubits_evolve_single_i() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::I), 0);
        let expected_data = DensityMatrix::from_vec(&[
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
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_h() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::H), 0);
        let expected_data = DensityMatrix::from_vec(&[
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
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_x() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::X), 0);
        let expected_data = DensityMatrix::from_vec(&[
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
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_y() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::Y), 0);
        let expected_data = DensityMatrix::from_vec(&[
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
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_two_qubits_evolve_single_z() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::Z), 0);
        let expected_data = DensityMatrix::from_vec(&[
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
        ])
        .unwrap();
        assert_eq!(rho.equals(&expected_data, TOLERANCE), true);
    }
    #[test]
    fn test_evolve_cx_ket00_1() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
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
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    fn test_evolve_cx_ket00_2() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
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
        assert_eq!(expected_data, rho.data);
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
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[1, 0]);
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
        assert_eq!(expected_data, rho.data);
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
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
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
        assert_eq!(expected_data, rho.data);
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
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 1]);
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
        let expected_rho = DensityMatrix::from_vec(&expected_data).unwrap();
        assert!(
            rho.equals(&expected_rho, TOLERANCE),
            "Actual:\n{rho}\nExpected:\n{expected_rho}"
        );
    }
    #[test]
    fn test_evolve_cz_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CZ), &[0, 1]);
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
        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_evolve_swap_ket00() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 1]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[2, 0]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 2]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[0, 1]);
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
        assert_eq!(expected_data, rho.data);
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

        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::SWAP), &[1, 2]);
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
        assert_eq!(expected_data, rho.data);
    }
    #[test]
    #[should_panic]
    fn test_evolve_single_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single_new(&Operator::one_qubit(OneQubitOp::X), 5);
    }

    #[test]
    #[should_panic]
    fn test_evolve_single_wrong_operator() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_single_new(&Operator::two_qubits(TwoQubitsOp::CZ), 0);
    }

    #[test]
    #[should_panic]
    fn test_evolve_out_of_range_target() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 3]);
    }

    #[test]
    #[should_panic]
    fn test_evolve_similar_indices() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        rho.evolve_new(&Operator::two_qubits(TwoQubitsOp::CX), &[0, 0]);
    }

    #[test]
    fn test_kron_1_qubit_zero_state() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        let other = DensityMatrix::new(1, State::ZERO);
        rho.tensor(&other);

        let expected_data: Vec<Complex<f64>> = vec![
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
        ];

        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_kron_multiple_qubits_zero_state() {
        let mut rho = DensityMatrix::new(2, State::ZERO); // 2 qubits, |00⟩
        let other = DensityMatrix::new(2, State::ZERO); // 2 qubits, |00⟩
        rho.tensor(&other);

        let mut expected_data: Vec<Complex<f64>> = vec![Complex::ZERO; 16 * 16];
        expected_data[0] = Complex::ONE;

        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_kron_plus_state() {
        let mut rho = DensityMatrix::new(1, State::PLUS); // 1 qubit, |+⟩ state
        let other = DensityMatrix::new(1, State::PLUS); // 1 qubit, |+⟩ state
        rho.tensor(&other);

        let expected_data: Vec<Complex<f64>> = vec![Complex::new(0.25, 0.); 16];

        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_kron_zero_and_one_state() {
        let mut rho = DensityMatrix::new(1, State::ZERO); // 1 qubit, |0⟩ state
        let other = DensityMatrix::new(1, State::ONE); // 1 qubit, |1⟩ state
        rho.tensor(&other);

        let mut expected_data: Vec<Complex<f64>> = vec![Complex::ZERO; 16];
        expected_data[5] = Complex::ONE;

        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_kron_two_qubits_plus_state() {
        let mut rho = DensityMatrix::new(2, State::PLUS); // 2 qubits, |+⟩ state
        let other = DensityMatrix::new(2, State::PLUS); // 2 qubits, |+⟩ state
        rho.tensor(&other);

        let expected_data: Vec<Complex<f64>> = vec![Complex::new(0.0625, 0.); 16 * 16];

        assert_eq!(expected_data, rho.data);
    }

    #[test]
    fn test_kron_specific() {
        let dm_vec = vec![
            Complex::new(0.06159174, 0.),
            Complex::new(0., 0.24041256),
            Complex::new(0., -0.24041256),
            Complex::new(0.93840826, 0.),
        ];
        let mut rho_1 = DensityMatrix::from_vec(&dm_vec).unwrap();
        let rho_2 = DensityMatrix::new(1, State::PLUS);

        rho_1.tensor(&rho_2);

        let expected_data: Vec<Complex<f64>> = vec![
            Complex::new(0.03079587, 0.),
            Complex::new(0.03079587, 0.),
            Complex::new(0., 0.12020628),
            Complex::new(0., 0.12020628),
            Complex::new(0.03079587, 0.),
            Complex::new(0.03079587, 0.),
            Complex::new(0., 0.12020628),
            Complex::new(0., 0.12020628),
            Complex::new(0., -0.12020628),
            Complex::new(0., -0.12020628),
            Complex::new(0.46920413, 0.),
            Complex::new(0.46920413, 0.),
            Complex::new(0., -0.12020628),
            Complex::new(0., -0.12020628),
            Complex::new(0.46920413, 0.),
            Complex::new(0.46920413, 0.),
        ];

        let expected_dm = DensityMatrix::from_vec(&expected_data).unwrap();

        let tol = 1e-10;
        assert!(rho_1.equals(&expected_dm, tol));
    }

    #[test]
    fn test_trace_pure_state() {
        let pure_states = vec![State::ZERO, State::PLUS, State::ONE, State::MINUS];
        for state in pure_states {
            let _ = (1..16).into_iter().map(|n| {
                let rho = DensityMatrix::new(n, state);
                let expected: Complex<f64> = Complex::ONE;
                let res = rho.trace();
                assert_eq!(expected, res, "Failed for state {:?}, n = {}", state, n);
            });
        }
    }

    #[test]
    fn test_expectation_single_1() {
        let mut rho = DensityMatrix::new(1, State::ZERO);
        let expected = Complex::new(1., 0.);
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::Z), 0)
            .unwrap();

        assert_eq!(expected, res);
    }
    #[test]
    fn test_expectation_single_2() {
        let mut rho = DensityMatrix::new(2, State::ZERO);
        let expected = Complex::new(1., 0.);
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::Z), 1)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_3() {
        let mut rho = DensityMatrix::new(3, State::ZERO);
        let expected = Complex::new(1., 0.);
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::Z), 2)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_4() {
        let mut rho = DensityMatrix::new(1, State::PLUS);
        let expected = Complex::ONE;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::I), 0)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_5() {
        let mut rho = DensityMatrix::new(2, State::PLUS);
        let expected = Complex::ONE;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::I), 0)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_6() {
        let mut rho = DensityMatrix::new(4, State::PLUS);
        let expected = Complex::ONE;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::I), 0)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_7() {
        let mut rho = DensityMatrix::new(8, State::PLUS);
        let expected = Complex::ONE;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::I), 4)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_8() {
        let mut rho = DensityMatrix::new(2, State::PLUS);
        let expected = Complex::ONE;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::X), 0)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    fn test_expectation_single_9() {
        let mut rho = DensityMatrix::new(2, State::PLUS);
        let expected = Complex::new(SQRT_2, 0.);
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::H), 0)
            .unwrap();

        assert!(num_traits::abs(expected.re - res.re) < 1e-8);
    }

    #[test]
    fn test_expectation_single_10() {
        let mut rho = DensityMatrix::new(1, State::ONE);
        let expected = Complex::ZERO;
        let res = rho
            .expectation_single(&Operator::one_qubit(OneQubitOp::X), 0)
            .unwrap();

        assert_eq!(expected, res);
    }

    #[test]
    #[should_panic]
    fn test_ptrace_fail() {
        let mut rho = DensityMatrix::new(2, State::ONE);
        rho.ptrace(&[2]).unwrap();
    }

    #[test]
    fn test_ptrace_1() {
        // Init (|00> + |01>) / sqrt(2)
        let sv: &[Complex<f64>; 4] = &[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ];
        let mut dm = DensityMatrix::from_statevec(sv).unwrap();
        dm.ptrace(&[0]).unwrap();

        let expected_dm = DensityMatrix::from_vec(&[
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ])
        .unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }

    #[test]
    fn test_ptrace_2() {
        // Init (|00> + |01>) / sqrt(2)
        let sv: &[Complex<f64>; 4] = &[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ];
        let mut dm = DensityMatrix::from_statevec(sv).unwrap();
        dm.ptrace(&[1]).unwrap();
        let expected_dm =
            DensityMatrix::from_vec(&[Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO])
                .unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }

    #[test]
    fn test_ptrace_3() {
        // Init (|00> + |01>) / sqrt(2)
        let sv: &[Complex<f64>; 4] = &[
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::ZERO,
            Complex::ZERO,
        ];
        let mut dm = DensityMatrix::from_statevec(sv).unwrap();
        dm.ptrace(&[0, 1]).unwrap();
        let expected_dm = DensityMatrix::from_vec(&[Complex::ONE]).unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }

    #[test]
    fn test_ptrace_4() {
        let mut dm = DensityMatrix::new(4, State::PLUS);
        dm.ptrace(&[0, 1, 2]).unwrap();
        let expected_dm = DensityMatrix::from_vec(&[
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
            Complex::new(0.5, 0.),
        ])
        .unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }

    #[test]
    fn test_ptrace_5() {
        let mut dm = DensityMatrix::new(4, State::PLUS);
        dm.ptrace(&[0, 1, 2, 3]).unwrap();
        let expected_dm = DensityMatrix::from_vec(&[Complex::ONE]).unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }

    #[test]
    fn test_ptrace_6() {
        let sv = &[
            Complex::ZERO,
            Complex::new(1. / 3_f64.sqrt(), 0.),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::new(2_f64.sqrt() / 3_f64.sqrt(), 0.),
            Complex::ZERO,
            Complex::ZERO,
        ];
        let mut dm = DensityMatrix::from_statevec(sv).unwrap();
        dm.ptrace(&[2]).unwrap();
        let expected_dm = DensityMatrix::from_vec(&[
            (1. / 3.).into(),
            Complex::ZERO,
            (SQRT_2 / 3.).into(),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            (SQRT_2 / 3.).into(),
            Complex::ZERO,
            (2. / 3.).into(),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ])
        .unwrap();

        let tol = 1e-15;

        if !dm.equals(&expected_dm, 1e-15) {
            println!("First dm: {:}\n================", dm);
            println!("Second dm: \n{:}\n================", expected_dm);
        }

        assert!(dm.equals(&expected_dm, tol));
    }
}
