use std::{ops::Mul};
use nalgebra::{Complex, ComplexField, DMatrix, DVector, Scalar};

const EPSILON: f64 = 1e-300;

fn is_close(a: Complex<f64>, b: Complex<f64>) -> bool {
    (a - b).abs() < EPSILON
}

fn tensor_dot<T>(
    a: &DMatrix<T>,
    b: &DMatrix<T>,
    a_axes: (usize, usize),
    b_axes: (usize, usize),
) -> Result<DMatrix<T>, &'static str>
where
    T: Scalar + Mul + Clone + std::ops::AddAssign<<T as std::ops::Mul>::Output>  + num_traits::Zero,
    <T as Mul>::Output: std::ops::Add
{
    if a.ncols() != b.nrows() {
        return Err("Incompatible shapes for tensor dot operation.");
    }

    let mut result = DMatrix::zeros(a.nrows(), b.ncols());
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                result[(i, j)] += a[(i, k)].clone() * b[(k, j)].clone();
            }
        }
    }
    Ok(result)
}

#[derive(Debug)]
pub struct QuantumState {
    state_vec: DVector::<Complex<f64>>
}

#[derive(Debug)]
pub struct DensityMatrix {
    d_matrix: DMatrix<Complex<f64>>
}

impl QuantumState {
    pub fn new(n_qubits: usize) -> Self {
        let mut qs = QuantumState { 
            state_vec: DVector::zeros(2usize.pow(n_qubits as u32))
        };
        qs.state_vec[0] = Complex::new(1., 0.);
        qs
    }
}

impl DensityMatrix {
    pub fn new(q_state: &QuantumState) -> Self {
        let dm = DensityMatrix {
            d_matrix: q_state.state_vec.clone() * q_state.state_vec.adjoint()
        };
        dm
    }

    pub fn evolve(&mut self, gate: &DMatrix::<Complex<f64>>) -> Result<(), &'static str> {
        let n_qubits = (gate.nrows() as f64).log2() as usize;
        if gate.ncols() != gate.nrows() || gate.nrows() != (1 << n_qubits) {
            return Err("Invalid gate dimensions.");
        }
        if self.d_matrix.ncols() != self.d_matrix.nrows() || self.d_matrix.ncols() != (1 << n_qubits) {
            return Err("Inconsistent dimensions between density matrix and gate.");
        }

        let result = tensor_dot(&gate, &self.d_matrix, (1, 0), (0, 1))?;
        self.d_matrix = tensor_dot(&result, &gate.adjoint(), (1, 0), (0, 1))?;
        self.normalize();

        Ok(())
    }

    fn normalize(&mut self) {
        self.d_matrix /= self.d_matrix.clone().norm().into();
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let q_state = QuantumState::new(2);
        assert_eq!(q_state.state_vec[0], Complex::new(1., 0.));
        assert_eq!(q_state.state_vec.norm(), 1.);
    }

    #[test]
    fn test_density_matrix_creation() {
        let q_state = QuantumState::new(2);

        let density_matrix = DensityMatrix::new(&q_state);

        // Check if the density matrix is Hermitian
        let hermitian_conjugate = density_matrix.d_matrix.adjoint();
        assert_eq!(density_matrix.d_matrix, hermitian_conjugate);

        // Check if the trace of the density matrix is 1
        let trace = density_matrix.d_matrix.trace();
        assert_eq!(trace, Complex::new(1., 0.));
    }

    #[test]
    fn test_evolve_single_qubit() {
        let q_state = QuantumState::new(1);
        let mut rho = DensityMatrix::new(&q_state);
        let h_gate = DMatrix::from_row_slice(2, 2, &[Complex::new(1.0 / f64::sqrt(2.0), 0.0), Complex::new(1.0 / f64::sqrt(2.0), 0.0), Complex::new(1.0 / f64::sqrt(2.0), 0.0), Complex::new(-1.0 / f64::sqrt(2.0), 0.0)]);
        let expected_result = DMatrix::from_row_slice(2, 2, &[Complex::new(0.5, 0.0), Complex::new(0.5, 0.0), Complex::new(0.5, 0.0), Complex::new(0.5, 0.0)]);

        rho.evolve(&h_gate).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    is_close(rho.d_matrix[(i, j)], expected_result[(i, j)]),
                    "Difference at position ({}, {}): expected {}, actual {}",
                    i, j, expected_result[(i, j)], rho.d_matrix[(i, j)]
                );
            }
        }
    }
}
