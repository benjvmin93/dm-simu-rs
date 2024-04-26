use nalgebra::{DMatrix, Complex, DVector};

#[derive(Debug)]
pub struct QuantumState {
    n_qubits: usize,
    state_vec: DVector::<Complex<f64>>
}

#[derive(Debug)]
pub struct DensityMatrix {
    n_qubits: usize,
    d_matrix: DMatrix::<Complex<f64>>
}

impl QuantumState {
    pub fn new(n_qubits: usize) -> Self {
        let mut qs = QuantumState { 
            n_qubits,
            state_vec: DVector::zeros(2usize.pow(n_qubits as u32))
        };
        qs.state_vec[0] = Complex::new(1., 0.);
        qs
    }
}

impl DensityMatrix {
    pub fn new(n_qubits: usize) -> Self {
        let mut dm = DensityMatrix {
            n_qubits,
            d_matrix: DMatrix::zeros(2usize.pow(n_qubits as u32) , 2usize.pow(n_qubits as u32))
        };
        dm.d_matrix[0] = Complex::new(1., 0.);
        dm
    }
}