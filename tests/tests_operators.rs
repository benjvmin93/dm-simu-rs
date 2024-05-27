#[cfg(test)]
mod tests_operators {
    use std::f64::consts::FRAC_1_SQRT_2;

    use mbqc::operators::{Operator, OneQubitOp, TwoQubitsOp};
    use mbqc::density_matrix::{DensityMatrix, State};
    use num_complex::Complex;

    #[test]
    fn test_operator_H() {
        let h_gate = Operator::one_qubit(OneQubitOp::H);
        assert_eq!(h_gate.data[0 * 2 + 0], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[0 * 2 + 1], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[1 * 2 + 0], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[1 * 2 + 1], Complex::new(FRAC_1_SQRT_2, 0.));
    }
    #[test]
    fn test_operator_X() {
        let h_gate = Operator::one_qubit(OneQubitOp::X);
        assert_eq!(h_gate.data[0 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(h_gate.data[0 * 2 + 1], Complex::new(1., 0.));
        assert_eq!(h_gate.data[1 * 2 + 0], Complex::new(1., 0.));
        assert_eq!(h_gate.data[1 * 2 + 1], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_Y() {
        let h_gate = Operator::one_qubit(OneQubitOp::Y);
        assert_eq!(h_gate.data[0 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(h_gate.data[0 * 2 + 1], Complex::new(0., -1.));
        assert_eq!(h_gate.data[1 * 2 + 0], Complex::new(0., 1.));
        assert_eq!(h_gate.data[1 * 2 + 1], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_Z() {
        let h_gate = Operator::one_qubit(OneQubitOp::Z);
        assert_eq!(h_gate.data[0 * 2 + 0], Complex::new(1., 0.));
        assert_eq!(h_gate.data[0 * 2 + 1], Complex::new(0., 0.));
        assert_eq!(h_gate.data[1 * 2 + 0], Complex::new(0., 0.));
        assert_eq!(h_gate.data[1 * 2 + 1], Complex::new(-1., 0.));
    }
}