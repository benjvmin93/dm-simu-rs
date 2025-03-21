pub mod density_matrix;
pub mod operators;
pub mod tools;

use density_matrix::{DensityMatrix, State};
use exceptions::PyTypeError;
use num_complex::Complex;
use numpy::PyArrayMethods;
use operators::Operator;
use pyo3::types::{PyComplex, PyListMethods};
use pyo3::{types::{PyList, PyTuple}, *};
use pyo3::prelude::PyAnyMethods;
use types::PyCapsuleMethods;

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
    fn new_dm_from_statevec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_dm_pyvec(
            py,
            DensityMatrix::from_statevec(&vec.as_slice()?)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(new_dm_from_statevec, m)?)?;

    #[pyo3::pyfunction]
    fn new_empty_dm<'py>(py: pyo3::prelude::Python<'py>) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_dm_pyvec(
            py,
            DensityMatrix::empty()
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(new_empty_dm, m)?)?;

    #[pyo3::pyfunction]
    fn new_dm_from_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_dm_pyvec(
            py,
            DensityMatrix::from_vec(&vec.as_slice()?)
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
    fn evolve_single_new<'py>(
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubit: usize,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_dm);
        let op = Operator::new(op.to_owned_array().as_slice().unwrap()).unwrap();
        dm.evolve_single_new(&op, qubit);
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve_single_new, m)?)?;

    #[pyo3::pyfunction]
    fn evolve_single<'py>(
        py: pyo3::prelude::Python<'py>,
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubit: usize,
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Complex<f64>>> {
        let dm = get_dm_mut_ref(py_dm);
        let op = Operator::new(op.to_owned_array().as_slice().unwrap()).unwrap();
        let new_dm = dm.evolve_single(&op, qubit).unwrap();
        numpy::IntoPyArray::into_pyarray_bound(new_dm, py)
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve_single, m)?)?;

    #[pyo3::pyfunction]
    fn normalize<'py>(
        py_dm: PyVec<'py>,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_dm);
        dm.normalize();
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(normalize, m)?)?;

    #[pyo3::pyfunction]
    fn evolve_new<'py>(
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubits: Vec<usize>,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(py_dm);
        let op = Operator::new(op.as_slice().unwrap()).unwrap();
        dm.evolve_new(&op, &qubits);
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve_new, m)?)?;

    #[pyo3::pyfunction]
    fn evolve<'py>(
        py: pyo3::prelude::Python<'py>,
        py_dm: PyVec<'py>,
        op: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
        qubits: Vec<usize>,
    ) -> pyo3::prelude::PyResult<pyo3::Bound<'py, numpy::array::PyArray1<Complex<f64>>>> {
        let dm = get_dm_mut_ref(py_dm);
        let op = op.as_slice().unwrap();

        let op = match std::panic::catch_unwind(|| Operator::new(op)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
            Err(err) => {
                let panic_message = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic occurred"
                };
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Rust panic: {}", panic_message)));
            }
        };

        let new_dm = match std::panic::catch_unwind(|| dm.evolve(&op, &qubits)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
            Err(err) => {
                let panic_message = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic occurred"
                };
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Rust panic: {}", panic_message)));
            }
        };

        // Use the bound method and extract the PyArray
        Ok(numpy::array::PyArray1::from_vec_bound(py, new_dm))
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve, m)?)?;

    #[pyo3::pyfunction]
    fn entangle<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
        qubits: (usize, usize),
    ) -> pyo3::prelude::PyResult<pyo3::Bound<'py, numpy::array::PyArray1<Complex<f64>>>> {
        let dm = get_dm_ref(py_vec);

        let result = match std::panic::catch_unwind(|| dm.cz(&qubits)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
            Err(err) => {
                let panic_message = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic occurred"
                };
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Rust panic: {}", panic_message)));
            }
        };
        Ok(numpy::array::PyArray1::from_vec_bound(py, result))

    }
    m.add_function(pyo3::wrap_pyfunction!(entangle, m)?)?;

    #[pyo3::pyfunction]
    fn cnot<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
        qubits: (usize, usize),
    ) -> pyo3::prelude::PyResult<pyo3::Bound<'py, numpy::array::PyArray1<Complex<f64>>>> {
        let dm = get_dm_ref(py_vec);

        let result = match std::panic::catch_unwind(|| dm.cnot(&qubits)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
            Err(err) => {
                let panic_message = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic occurred"
                };
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Rust panic: {}", panic_message)));
            }
        };
        Ok(numpy::array::PyArray1::from_vec_bound(py, result))

    }
    m.add_function(pyo3::wrap_pyfunction!(cnot, m)?)?;

    #[pyo3::pyfunction]
    fn swap<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
        qubits: (usize, usize),
    ) -> pyo3::prelude::PyResult<pyo3::Bound<'py, numpy::array::PyArray1<Complex<f64>>>> {
        let dm = get_dm_mut_ref(py_vec);

        let result = match std::panic::catch_unwind(|| dm.swap(&qubits)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
            Err(err) => {
                let panic_message = if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic occurred"
                };
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Rust panic: {}", panic_message)));
            }
        };
        Ok(numpy::array::PyArray1::from_vec_bound(py, result))
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
    fn ptrace<'py>(dm: PyVec<'py>, qargs: Vec<usize>) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(dm);
        dm.ptrace(&qargs.as_slice()).unwrap();
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(ptrace, m)?)?;

    #[pyo3::pyfunction]
    fn set<'py>(
        py: pyo3::prelude::Python<'py>,
        dm_object: PyVec<'py>,
        new_dm: numpy::borrow::PyReadonlyArrayDyn<Complex<f64>>,
    ) -> pyo3::prelude::PyResult<()> {
        let dm = get_dm_mut_ref(dm_object);

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

        std::mem::swap(&mut dm.data, &mut new_dm_vec.to_vec());
        std::mem::swap(&mut dm.nqubits, &mut (new_n as usize));
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(set, m)?)?;


    /// Helper function to parse a Python `PyList` of `PyTuple` into `Vec<(Complex<f64>, Vec<Complex<f64>>)>`
    fn parse_krauss_channel<'py>(
        krauss_channel: &Bound<'_, PyList>,
    ) -> PyResult<Vec<(Complex<f64>, Vec<Complex<f64>>)>> {
        let mut channel_vec: Vec<(Complex<f64>, Vec<Complex<f64>>)> = Vec::new();

        while let Ok(item) = krauss_channel.as_ref().iter() {
            let item = item.as_ref();
            let tuple = item.downcast::<PyTuple>()?;
            if tuple.len()? != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Each Krauss channel element must be a tuple (complex, list of complex)",
                ));
            }

            // Extract the coefficient (a single complex number)
            let coef: Complex<f64> = extract_complex(tuple.get_item(0)?.extract()?)?;

            // Extract the operator (a list of complex numbers)
            let op_list: Vec<Complex<f64>> = tuple
                .get_item(1)?.downcast::<PyList>()?
                .iter()
                .map(|op| extract_complex(op.extract()?))
                .into_iter().collect::<PyResult<Vec<Complex<f64>>>>()?;

            // Push the tuple (coef, op_list) into the vector
            channel_vec.push((coef, op_list));
        }

        Ok(channel_vec)
    }

    /// Extracts a Python complex number into a Rust `Complex<f64>`
    fn extract_complex(obj: &PyAny) -> PyResult<Complex<f64>> {
        let real: f64 = obj.getattr("real")?.extract()?;
        let imag: f64 = obj.getattr("imag")?.extract()?;
        Ok(Complex { re: real, im: imag })
    }

    #[pyo3::pyfunction]
    fn apply_channel<'py>(
        py: Python<'py>,
        dm_object: PyVec<'py>, // Assume `get_dm_mut_ref` will handle this appropriately
        krauss_channel: &Bound<'_, PyList>,
        qargs: Vec<usize>,
    ) -> PyResult<()> {

        let new_krauss_channel: Vec<(Complex<f64>, Vec<Complex<f64>>)> = krauss_channel
            .iter()
            .map(|x| {
                let tuple = x.downcast::<PyTuple>()?;
                if tuple.len()? != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Each Krauss channel element must be a tuple (complex, list of complex)",
                    ));
                }
                let coef = extract_complex(tuple.get_item(0)?.extract()?)?;
                let py_op = tuple.get_item(1)?;
                let py_op = py_op.downcast::<PyList>()?;

                let op: Vec<Complex<f64>> = py_op
                .iter()
                .map(|op_data| {
                    extract_complex(op_data.extract()?)
                })
                .collect::<PyResult<Vec<Complex<f64>>>>()?;

                Ok((coef, op))
            }).collect::<PyResult<Vec<(Complex<f64>, Vec<Complex<f64>>)>>>()?;

        let dm = get_dm_mut_ref(dm_object);

        // Apply the Krauss channel to the density matrix
        let mut result = dm.apply_channel(&new_krauss_channel, &qargs).unwrap();

        // Swap the result back into the density matrix
        std::mem::swap(&mut dm.data, &mut result);

        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(apply_channel, m)?)?;

    Ok(())
}
