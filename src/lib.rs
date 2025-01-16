pub mod density_matrix;
pub mod operators;
pub mod tools;

use density_matrix::{DensityMatrix, State};
use num_complex::Complex;
use numpy::PyArrayMethods;
use operators::Operator;
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyComplex};

#[pyo3::pymodule]
fn dm_simu_rs<'py>(
    _py: pyo3::prelude::Python<'py>,
    m: &pyo3::prelude::Bound<'py, pyo3::types::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    m.add("Zero", State::ZERO);
    m.add("Plus", State::PLUS);
    m.add("One", State::ONE);
    m.add("Minus", State::MINUS);

    type PyVec<'py> = Bound<'py, pyo3::types::PyCapsule>;

    fn make_dm_pyvec<'py>(
        py: pyo3::prelude::Python<'py>,
        dm: DensityMatrix,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        let capsule_name = std::ffi::CString::new("dm").unwrap();
        pyo3::types::PyCapsule::new_bound(py, dm, Some(capsule_name))
    }

    fn make_op_pyvec<'py>(
        py: pyo3::prelude::Python<'py>,
        op: Operator,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        let capsule_name = std::ffi::CString::new("operator").unwrap();
        pyo3::types::PyCapsule::new_bound(py, op, Some(capsule_name))
    }

    fn get_op_ref<'py>(op: PyVec<'py>) -> &Operator {
        unsafe { op.reference::<Operator>() }
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
        make_dm_pyvec(py, DensityMatrix::new(nqubits, initial_state))
    }
    m.add_function(pyo3::wrap_pyfunction!(new_dm, m)?)?;

    #[pyo3::pyfunction]
    fn new_dm_from_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_dm_pyvec(
            py,
            DensityMatrix::from_statevec(&vec.as_slice()?)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(new_dm_from_vec, m)?)?;

    #[pyo3::pyfunction]
    fn get_dm<'py>(
        py: pyo3::prelude::Python<'py>,
        dm_py_vec: PyVec<'py>,
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Complex<f64>>> {
        let dm = get_dm_ref(dm_py_vec);
        numpy::IntoPyArray::into_pyarray_bound(dm.data.to_vec(), py)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_dm, m)?)?;

    #[pyo3::pyfunction]
    fn new_op<'py>(
        py: pyo3::prelude::Python<'py>,
        data: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_op_pyvec(
            py,
            Operator::new(data.as_slice()?).map_err(pyo3::exceptions::PyValueError::new_err)?,
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(new_op, m)?)?;

    #[pyo3::pyfunction]
    fn get_op<'py>(
        py: pyo3::prelude::Python<'py>,
        op_py_vec: PyVec<'py>,
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Complex<f64>>> {
        let op = get_op_ref(op_py_vec);
        numpy::IntoPyArray::into_pyarray_bound(op.data.to_vec(), py)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_op, m)?)?;

    #[pyo3::pyfunction]
    fn get_nqubits<'py>(dm: PyVec<'py>) -> pyo3::prelude::PyResult<usize> {
        let dm = get_dm_ref(dm);
        Ok(dm.nqubits)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_nqubits, m)?)?;

    #[pyo3::pyfunction]
    fn evolve_single<'py>(
        py: pyo3::prelude::Python<'py>,
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubit: usize,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_dm);
        let op = Operator::new(op.to_owned_array().as_slice().unwrap()).unwrap();
        Ok(dm.evolve_single(&op, qubit).unwrap())
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve_single, m)?)?;

    #[pyo3::pyfunction]
    fn evolve<'py>(
        py: pyo3::prelude::Python<'py>,
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubits: Vec<usize>,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_dm);
        let op = Operator::new(op.as_slice()?).unwrap();
        Ok(dm.evolve(&op, &qubits).unwrap())
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve, m)?)?;

    #[pyo3::pyfunction]
    fn entangle<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
        qubits: (usize, usize)
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Complex<f64>>> {
        let dm = get_dm_mut_ref(py_vec);
        let result = dm.cz(&qubits).unwrap();
        numpy::IntoPyArray::into_pyarray_bound(result, py)
    }
    m.add_function(pyo3::wrap_pyfunction!(entangle, m)?)?;

    #[pyo3::pyfunction]
    fn swap<'py>(py_vec: PyVec<'py>, qubits: (usize, usize)) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_vec);
        Ok(dm.swap(&qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(swap, m)?)?;

    #[pyo3::pyfunction]
    fn tensor_dm<'py>(dm: PyVec<'py>, other: PyVec<'py>) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(dm);
        let other_dm = get_dm_mut_ref(other);
        Ok(dm.tensor(other_dm))
    }
    m.add_function(pyo3::wrap_pyfunction!(tensor_dm, m)?)?;

    #[pyo3::pyfunction]
    fn expectation_single<'py>(
        py: pyo3::prelude::Python<'py>,
        dm: PyVec<'py>,
        op: PyVec<'py>,
        i: usize,
    ) -> pyo3::prelude::PyResult<pyo3::Bound<'py, PyComplex>> {
        let dm = get_dm_mut_ref(dm);
        let op = get_op_ref(op);

        let result = dm.expectation_single(op, i).unwrap();
        Ok(PyComplex::from_doubles_bound(py, result.re, result.im))
    }
    m.add_function(pyo3::wrap_pyfunction!(expectation_single, m)?)?;

    #[pyo3::pyfunction]
    fn ptrace<'py>(
        dm: PyVec<'py>,
        qargs: Vec<usize>
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(dm);
        Ok(dm.ptrace(&qargs.as_slice()).unwrap())
    }
    m.add_function(pyo3::wrap_pyfunction!(ptrace, m)?)?;

    #[pyo3::pyfunction]
    fn set<'py>(
        py: pyo3::prelude::Python<'py>,
        new_dm: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        let new_dm_vec = new_dm.as_slice().unwrap();

        if !new_dm_vec.len().is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "New dm size is not a power of two.".to_string(),
            ));
        }

        let mut new_n = (new_dm_vec.len() as f64).log2();
        if new_n % 2. != 0. {
            return Err(PyErr::new::<PyTypeError, _>(
                "New dm size is not 2 ** 2n.".to_string(),
            ));
        } else {
            new_n /= 2.;
        }

        let _dm = DensityMatrix::from_vec(new_dm_vec).unwrap();

        make_dm_pyvec(py, _dm)
    }
    m.add_function(pyo3::wrap_pyfunction!(set, m)?)?;

    Ok(())
}
