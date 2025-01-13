#[cfg(test)]
mod tests_operators {
    use std::f64::consts::FRAC_1_SQRT_2;

    use dm_simu_rs::operators::{OneQubitOp, Operator, TwoQubitsOp};
    use num_complex::Complex;

    #[test]
    fn test_operator_h() {
        let h_gate = Operator::one_qubit(OneQubitOp::H);
        assert_eq!(h_gate.data[0], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[1], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[2], Complex::new(FRAC_1_SQRT_2, 0.));
        assert_eq!(h_gate.data[3], Complex::new(FRAC_1_SQRT_2, 0.));
    }
    #[test]
    fn test_operator_x() {
        let x_gate = Operator::one_qubit(OneQubitOp::X);
        assert_eq!(x_gate.data[0], Complex::new(0., 0.));
        assert_eq!(x_gate.data[1], Complex::new(1., 0.));
        assert_eq!(x_gate.data[2], Complex::new(1., 0.));
        assert_eq!(x_gate.data[3], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_y() {
        let y_gate = Operator::one_qubit(OneQubitOp::Y);
        assert_eq!(y_gate.data[0], Complex::new(0., 0.));
        assert_eq!(y_gate.data[1], Complex::new(0., -1.));
        assert_eq!(y_gate.data[2], Complex::new(0., 1.));
        assert_eq!(y_gate.data[3], Complex::new(0., 0.));
    }
    #[test]
    fn test_operator_z() {
        let z_gate = Operator::one_qubit(OneQubitOp::Z);
        assert_eq!(z_gate.data[0], Complex::new(1., 0.));
        assert_eq!(z_gate.data[1], Complex::new(0., 0.));
        assert_eq!(z_gate.data[2], Complex::new(0., 0.));
        assert_eq!(z_gate.data[3], Complex::new(-1., 0.));
    }
    #[test]
    fn test_operator_cx() {
        let cx_gate = Operator::two_qubits(TwoQubitsOp::CX);
        assert_eq!(cx_gate.data[0], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[5], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[11], Complex::new(1., 0.));
        assert_eq!(cx_gate.data[14], Complex::new(1., 0.));
    }
    #[test]
    fn test_operator_cz() {
        let cz_gate = Operator::two_qubits(TwoQubitsOp::CZ);
        assert_eq!(cz_gate.data[0], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[5], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[10], Complex::new(1., 0.));
        assert_eq!(cz_gate.data[15], Complex::new(-1., 0.));
    }
    #[test]
    fn test_operator_swap() {
        let swap_gate = Operator::two_qubits(TwoQubitsOp::SWAP);
        assert_eq!(swap_gate.data[0], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[6], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[9], Complex::new(1., 0.));
        assert_eq!(swap_gate.data[15], Complex::new(1., 0.));
    }
    #[test]
    fn test_transconjugate_x() {
        let mut x = Operator::one_qubit(OneQubitOp::X);
        x = x.transconj();
        print!("{}", x);
        assert_eq!(x.data[0], Complex::new(0., 0.));
        assert_eq!(x.data[1], Complex::new(1., 0.));
        assert_eq!(x.data[2], Complex::new(1., 0.));
        assert_eq!(x.data[3], Complex::new(0., 0.));
    }
    #[test]
    fn test_transconjugate_y() {
        let mut y = Operator::one_qubit(OneQubitOp::Y);
        y = y.transconj();
        print!("{}", y);
        assert_eq!(y.data[0], Complex::new(0., 0.));
        assert_eq!(y.data[1], Complex::new(0., -1.));
        assert_eq!(y.data[2], Complex::new(0., 1.));
        assert_eq!(y.data[3], Complex::new(0., 0.));
    }
    #[test]
    fn test_transconjugate_z() {
        let mut z = Operator::one_qubit(OneQubitOp::Z);
        z = z.transconj();
        print!("{}", z);
        assert_eq!(z.data[0], Complex::new(1., 0.));
        assert_eq!(z.data[1], Complex::new(0., 0.));
        assert_eq!(z.data[2], Complex::new(0., 0.));
        assert_eq!(z.data[3], Complex::new(-1., 0.));
    }
    #[test]
    fn test_transconjugate_h() {
        let mut h = Operator::one_qubit(OneQubitOp::H);
        h = h.transconj();
        print!("{}", h);
        let sqrt_2_inv = FRAC_1_SQRT_2;
        assert_eq!(h.data[0], Complex::new(sqrt_2_inv, 0.));
        assert_eq!(h.data[1], Complex::new(sqrt_2_inv, 0.));
        assert_eq!(h.data[2], Complex::new(sqrt_2_inv, 0.));
        assert_eq!(h.data[3], Complex::new(sqrt_2_inv, 0.));
    }
    #[test]
    fn test_transconjugate_random_unitary() {
        let mut u = Operator::new(&[
            Complex::new(1., 0.5),
            Complex::new(2., -0.5),
            Complex::new(3., -0.5),
            Complex::new(4., 0.5),
        ])
        .unwrap();
        u = u.transconj();
        print!("{}", u);
        assert_eq!(u.data[0], Complex::new(1., -0.5));
        assert_eq!(u.data[1], Complex::new(3., 0.5));
        assert_eq!(u.data[2], Complex::new(2., 0.5));
        assert_eq!(u.data[3], Complex::new(4., -0.5));
    }
}
