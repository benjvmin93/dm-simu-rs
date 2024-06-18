pub mod tensor;
pub mod density_matrix;
pub mod operators;
pub mod tools;

use num_complex::Complex;
use pyo3::prelude::*;
use density_matrix::{DensityMatrix, State};

#[pyo3::pymodule]
fn dm_simu_rs<'py>(
    _py: pyo3::prelude::Python<'py>,
    m: &pyo3::prelude::Bound<'py, pyo3::types::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    m.add("Zero", State::ZERO);
    m.add("Plus", State::PLUS);

    type PyVec<'py> = Bound<'py, pyo3::types::PyCapsule>;

    fn make_pyvec<'py>(
        py: pyo3::prelude::Python<'py>,
        dm: DensityMatrix,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        let capsule_name = std::ffi::CString::new("dm").unwrap();
        pyo3::types::PyCapsule::new_bound(py, dm, Some(capsule_name))
    }

    fn get_dm_ref<'py>(dm: PyVec<'py>) -> &DensityMatrix {
        unsafe { dm.reference::<DensityMatrix>() }
    }

    fn get_dm_mut_ref<'py>(dm: PyVec<'py>) -> &mut DensityMatrix {
        unsafe { &mut *dm.pointer().cast() }
    }

    #[pyo3::pyfunction]
    fn new_dm<'py>(
        py: pyo3::prelude::Python<'py>,
        nqubits: usize,
        initial_state: State,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_pyvec(py, DensityMatrix::new(nqubits, Some(initial_state)))
    }
    m.add_function(pyo3::wrap_pyfunction!(new_dm, m)?)?;

    #[pyo3::pyfunction]
    fn from_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_pyvec(
            py,
            DensityMatrix::from_statevec(vec.as_slice()?.to_vec())
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(from_vec, m)?)?;

    #[pyo3::pyfunction]
    fn get_dm<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Complex<f64>>> {
        let dm = get_dm_ref(py_vec);
        numpy::IntoPyArray::into_pyarray_bound(dm.data.data.to_vec(), py)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_dm, m)?)?;

    #[pyo3::pyfunction]
    fn get_nqubits<'py>(dm: PyVec<'py>) -> pyo3::prelude::PyResult<usize> {
        let dm = get_dm_ref(dm);
        Ok(dm.nqubits)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_nqubits, m)?)?;

    #[pyo3::pyfunction]
    fn entangle<'py>(py_vec: PyVec<'py>, qubits: (usize, usize)) -> pyo3::prelude::PyResult<()> {
        let vec = get_dm_mut_ref(py_vec);
        Ok(vec.entangle(&qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(entangle, m)?)?;

    #[pyo3::pyfunction]
    fn swap<'py>(py_vec: PyVec<'py>, qubits: (usize, usize)) -> pyo3::prelude::PyResult<()> {
        let vec = get_dm_mut_ref(py_vec);
        Ok(vec.swap(&qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(swap, m)?)?;

    Ok(())
}