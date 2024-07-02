#[cfg(test)]
mod tests_operators {
    use std::f64::consts::FRAC_1_SQRT_2;

    use dm_simu_rs::operators::{OneQubitOp, Operator, TwoQubitsOp};
    use dm_simu_rs::tools::complex_approx_eq;
    use num_complex::Complex;

    #[test]
    fn test_operator_h() {
        let h_gate = Operator::one_qubit(OneQubitOp::H);
        let expected = vec![
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
        ];
        assert_eq!(h_gate.tensor.shape, vec![2, 2]);
        assert_eq!(h_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_x() {
        let x_gate = Operator::one_qubit(OneQubitOp::X);
        let expected = vec![Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO];
        assert_eq!(x_gate.tensor.shape, vec![2, 2]);
        assert_eq!(x_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_y() {
        let y_gate = Operator::one_qubit(OneQubitOp::Y);
        let expected = vec![
            Complex::ZERO,
            Complex::new(0., -1.),
            Complex::new(0., 1.),
            Complex::ZERO,
        ];
        assert_eq!(y_gate.tensor.shape, vec![2, 2]);
        assert_eq!(y_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_z() {
        let z_gate = Operator::one_qubit(OneQubitOp::Z);
        let expected = vec![Complex::ONE, Complex::ZERO, Complex::ZERO, -Complex::ONE];
        assert_eq!(z_gate.tensor.shape, vec![2, 2]);
        assert_eq!(z_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_cx() {
        let cx_gate = Operator::two_qubits(TwoQubitsOp::CX);
        let expected = vec![
            Complex::ONE,
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
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
        ];
        assert_eq!(cx_gate.tensor.shape, vec![2, 2, 2, 2]);
        assert_eq!(cx_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_cz() {
        let cz_gate = Operator::two_qubits(TwoQubitsOp::CZ);
        let expected = vec![
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            -Complex::ONE,
        ];
        assert_eq!(cz_gate.tensor.shape, vec![2, 2, 2, 2]);
        assert_eq!(cz_gate.tensor.data, expected);
    }
    #[test]
    fn test_operator_swap() {
        let swap_gate = Operator::two_qubits(TwoQubitsOp::SWAP);
        let expected = vec![
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ];
        assert_eq!(swap_gate.tensor.shape, vec![2, 2, 2, 2]);
        assert_eq!(swap_gate.tensor.data, expected);
    }
    #[test]
    fn test_transconjugate_x() {
        let mut x = Operator::one_qubit(OneQubitOp::X);
        x = x.transconj();
        let expected = vec![Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO];
        assert_eq!(x.tensor.shape, vec![2, 2]);
        assert_eq!(x.tensor.data, expected);
    }
    #[test]
    fn test_transconjugate_y() {
        let mut y = Operator::one_qubit(OneQubitOp::Y);
        y = y.transconj();
        let expected = vec![
            Complex::ZERO,
            Complex::new(0., -1.),
            Complex::new(0., 1.),
            Complex::ZERO,
        ];
        assert_eq!(y.tensor.shape, vec![2, 2]);
        assert_eq!(y.tensor.data, expected);
    }
    #[test]
    fn test_transconjugate_z() {
        let mut z = Operator::one_qubit(OneQubitOp::Z);
        z = z.transconj();
        let expected = vec![Complex::ONE, Complex::ZERO, Complex::ZERO, -Complex::ONE];
        assert_eq!(z.tensor.shape, vec![2, 2]);
        assert_eq!(z.tensor.data, expected);
    }
    #[test]
    fn test_transconjugate_h() {
        let mut h = Operator::one_qubit(OneQubitOp::H);
        h = h.transconj();
        let expected = vec![
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
            Complex::new(FRAC_1_SQRT_2, 0.),
        ];
        assert_eq!(h.tensor.shape, vec![2, 2]);
        assert_eq!(h.tensor.data, expected);
    }
    #[test]
    fn test_transconjugate_random_unitary() {
        let mut u = Operator::new(vec![
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, -0.5),
            Complex::new(0.5, 0.5),
        ])
        .unwrap();
        u = u.transconj();
        print!("{}", u);
        let expected = vec![
            Complex::new(0.5, -0.5),
            Complex::new(0.5, 0.5),
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
        ];
        assert_eq!(u.tensor.shape, vec![2, 2]);
        assert_eq!(u.tensor.data, expected);
    }
}
