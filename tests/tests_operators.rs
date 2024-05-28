#[cfg(test)]
mod tests_operators {
    use std::f64::consts::FRAC_1_SQRT_2;

    use mbqc::operators::{Operator, OneQubitOp, TwoQubitsOp};
    use num_complex::Complex;

    #[test]
    fn test_operator_h() {
        let h_gate = Operator::one_qubit(OneQubitOp::H);
        assert_eq!(h_gate.data[0 * 2 + 0], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[0 * 2 + 1], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[1 * 2 + 0], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[1 * 2 + 1], Complex::new(FRAC_1_SQRT_2, 0.));
    }
    #[test]
    fn test_operator_x() {
        let x_gate = Operator::one_qubit(OneQubitOp::X);
        assert_eq!(x_gate.data[0 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(x_gate.data[0 * 2 + 1], Complex::new(1., 0.));
        assert_eq!(x_gate.data[1 * 2 + 0], Complex::new(1., 0.));
        assert_eq!(x_gate.data[1 * 2 + 1], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_y() {
        let y_gate = Operator::one_qubit(OneQubitOp::Y);
        assert_eq!(y_gate.data[0 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(y_gate.data[0 * 2 + 1], Complex::new(0., -1.));
        assert_eq!(y_gate.data[1 * 2 + 0], Complex::new(0., 1.));
        assert_eq!(y_gate.data[1 * 2 + 1], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_z() {
        let z_gate = Operator::one_qubit(OneQubitOp::Z);
        assert_eq!(z_gate.data[0 * 2 + 0], Complex::new(1., 0.));
        assert_eq!(z_gate.data[0 * 2 + 1], Complex::new(0., 0.));
        assert_eq!(z_gate.data[1 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(z_gate.data[1 * 2 + 1], Complex::new(-1., 0.));
    }
    #[test]
    fn test_operator_cx() {
        let cx_gate = Operator::two_qubits(TwoQubitsOp::CX);
        assert_eq!(cx_gate.data[0 * 4 + 0], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[1 * 4 + 1], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[2 * 4 + 3], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[3 * 4 + 2], Complex::new(1., 0.));
    }
    #[test]
    fn test_operator_cz() {
        let cz_gate = Operator::two_qubits(TwoQubitsOp::CZ);
        assert_eq!(cz_gate.data[0 * 4 + 0], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[1 * 4 + 1], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[2 * 4 + 2], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[3 * 4 + 3], Complex::new(-1., 0.));
    }
    #[test]
    fn test_operator_swap() {
        let swap_gate = Operator::two_qubits(TwoQubitsOp::SWAP);
        assert_eq!(swap_gate.data[0 * 4 + 0], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[1 * 4 + 2], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[2 * 4 + 1], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[3 * 4 + 3], Complex::new(1., 0.));
    }
}