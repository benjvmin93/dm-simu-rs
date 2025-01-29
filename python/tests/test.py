import functools
import hypothesis as hyp
import numpy as np
import dm_simu_rs
from copy import deepcopy

import pytest

def dm_get_nqubits(array: np.ndarray) -> int:
    """
        Compute the number of qubit with a given density matrix.
    """
    size = np.sqrt(len(array))
    return int(np.log2(size))

def sv_get_nqubits(array: np.ndarray) -> int:
    """
        Compute the number of qubit with a given statevec matrix.
    """
    size = len(array)
    return int(np.log2(size))


def get_norm(array: np.ndarray, nqubits: int) -> float:
    size = 1 << nqubits
    return np.trace(np.reshape(array, (size, size)))


complex_st = hyp.strategies.complex_numbers(min_magnitude=1e-5, max_magnitude=1e5)


states = [
    np.array([[1, 0]], dtype=np.complex128),  # |0>
    np.array([[0, 1]], dtype=np.complex128),  # |1>
    np.array([[1, 1]] / np.sqrt(2), dtype=np.complex128), # |+>
    np.array([[1, -1]] / np.sqrt(2), dtype=np.complex128) # |->
]


op_single = [
    np.array([[1, 0],
              [0, 1]], dtype=np.complex128),    # I
    np.array([[0, 1],
              [1, 0]], dtype=np.complex128),    # X
    np.array([[1, 1],
              [1, -1]] / np.sqrt(2), dtype=np.complex128),  # H
    np.array([[0, -1j],
              [1j, 0]], dtype=np.complex128),   # Y
    np.array([[1, 0],
              [0, -1]], dtype=np.complex128),    # Z
]


op_double = [
    np.array([  [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=np.complex128),    # CNOT
    np.array([  [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]], dtype=np.complex128),    # CZ
    np.array([  [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]], dtype=np.complex128),    # SWAP
    np.array([  [1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0,],
                [0, 0, 0, 1]])                          # iSWAP
]


def evolve_single(rho, Nqubit, op, i):
    """Single-qubit operation.
    Parameters
    ----------
        op : np.ndarray
            2*2 matrix.
        i : int
            Index of qubit to apply operator.
    """
    assert i >= 0 and i < Nqubit
    if op.shape != (2, 2):
        raise ValueError("op must be 2*2 matrix.")
    rho_tensor = rho.reshape((2,) * Nqubit * 2)
    rho_tensor = np.tensordot(
        np.tensordot(op, rho_tensor, axes=[1, i]),
        op.conj().T,
        axes=[i + Nqubit, 0],
    )
    rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + Nqubit))
    rho = rho_tensor.reshape((2**Nqubit, 2**Nqubit))
    return rho


def evolve(rho, Nqubit, op, qargs):
    """Multi-qubit operation
    Args:
        op (np.array): 2^n*2^n matrix
        qargs (list of ints): target qubits' indexes
    """
    d = op.shape
    # check it is a matrix.
    if len(d) == 2:
        # check it is square
        if d[0] == d[1]:
            pass
        else:
            raise ValueError(f"The provided operator has shape {op.shape} and is not a square matrix.")
    else:
        raise ValueError(f"The provided data has incorrect shape {op.shape}.")
    nqb_op = np.log2(len(op))
    if not np.isclose(nqb_op, int(nqb_op)):
        raise ValueError("Incorrect operator dimension: not consistent with qubits.")
    nqb_op = int(nqb_op)
    if nqb_op != len(qargs):
        raise ValueError("The dimension of the operator doesn't match the number of targets.")
    if not all(0 <= i < Nqubit for i in qargs):
        raise ValueError("Incorrect target indices.")
    if len(set(qargs)) != nqb_op:
        raise ValueError("A repeated target qubit index is not possible.")
    op_tensor = op.reshape((2,) * 2 * nqb_op)
    rho_tensor = rho.reshape((2,) * Nqubit * 2)
    rho_tensor = np.tensordot(
        np.tensordot(
            op_tensor,
            rho_tensor,
            axes=[tuple(nqb_op + i for i in range(len(qargs))), tuple(qargs)],
        ),
        op.conj().T.reshape((2,) * 2 * nqb_op),
        axes=[
            tuple(i + Nqubit for i in qargs),
            tuple(i for i in range(len(qargs))),
        ],
    )
    rho_tensor = np.moveaxis(
        rho_tensor,
        [i for i in range(len(qargs))] + [-i for i in range(1, len(qargs) + 1)],
        [i for i in qargs] + [i + Nqubit for i in reversed(list(qargs))],
    )
    rho = rho_tensor.reshape((2**Nqubit, 2**Nqubit))
    return rho


def expectation_single(dm, op, i, Nqubit):
        """Expectation value of single-qubit operator.

        Args:
            op (np.array): 2*2 Hermite operator
            loc (int): Index of qubit on which to apply operator.
        Returns:
            complex: expectation value (real for hermitian ops!).
        """

        if not (0 <= i < Nqubit):
            raise ValueError(f"Wrong target qubit {i}. Must between 0 and {Nqubit-1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        dm1 = deepcopy(dm)
        dm1 /= np.trace(dm1)

        rho_tensor = dm1.reshape((2,) * Nqubit * 2)
        rho_tensor = np.tensordot(op, rho_tensor, axes=[1, i])
        rho_tensor = np.moveaxis(rho_tensor, 0, i)
        dm1 = rho_tensor.reshape((2**Nqubit, 2**Nqubit))

        return np.trace(dm1)
    
def ptrace(rho: np.ndarray, qargs):
    """partial trace
    Parameters
    ----------
        qargs : list of ints or int
            Indices of qubit to trace out.
    """
    n = int(np.log2(rho.shape[0]))
    if isinstance(qargs, int):
        qargs = [qargs]
    assert isinstance(qargs, (list, tuple))
    qargs_num = len(qargs)
    nqubit_after = n - qargs_num
    assert n > 0
    assert all([qarg >= 0 and qarg < n for qarg in qargs])
    rho_res = rho.reshape((2,) * n * 2)
    # ket, bra indices to trace out
    trace_axes = list(qargs) + [n + qarg for qarg in qargs]
    rho_res = np.tensordot(
        np.eye(2**qargs_num).reshape((2,) * qargs_num * 2),
        rho_res,
        axes=(list(range(2 * qargs_num)), trace_axes),
    )
    rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))
    Nqubit = nqubit_after
    
    return rho, Nqubit


def integer_st(min=1, max=10):
    return hyp.strategies.integers(min_value=min, max_value=max)


def state_st():
    return hyp.strategies.sampled_from(states)


def build_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    sv = functools.reduce(np.kron, (state for _ in range(nqubits)), np.array([1], dtype=np.complex128))
    return sv.flatten()


def build_rho_from_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    psi = functools.reduce(np.kron, (state for _ in range(nqubits)), np.array([1], dtype=np.complex128))
    return np.outer(psi, psi.conj())


def dm_st(min_qubits=1, max_qubits=10):  # Limit max_length to manage memory
    """
        Returns a random density matrix which size is between min_qubits and max_qubits.
    """
    return (
        integer_st(min_qubits, max_qubits)
        .flatmap(lambda nqubits: state_st().map(lambda state: build_rho_from_sv(state, nqubits)))
    )


def sv_st(min=1, max=10):
    """
        Returns a 
    """
    return (
        integer_st(min, max)
        .flatmap(lambda nqubits: state_st().map(lambda state: build_sv(state, nqubits)))
    )


@hyp.given(
    hyp.strategies.integers(min_value=1, max_value=8),
    hyp.strategies.sampled_from([dm_simu_rs.Zero, dm_simu_rs.Plus]),
)
def test_new_dm(nqubits, state):
    """
        Test for initializing new density matrix from number of qubit and sampled states |0> and |+>
    """
    dm = dm_simu_rs.new_dm(nqubits, state)
    assert dm_simu_rs.get_nqubits(dm) == nqubits
    array = dm_simu_rs.get_dm(dm)
    size = 1 << nqubits
    assert len(array) == size * size
    if state == dm_simu_rs.Zero:
        state_mat = np.array([1, 0])
    elif state == dm_simu_rs.Plus:
        state_mat = np.array([1, 1]) / np.sqrt(2)
    else:
        assert False
    rho = np.outer(state_mat, state_mat)
    ref = functools.reduce(np.kron, (rho for _ in range(nqubits)), np.array(1, dtype=np.complex128))
    np.testing.assert_allclose(array, ref.flatten())

def test_empty_dm():
    """
        Test for initializing an empty density matrix. Either with 0 qubit or with an empty vector passed in arguments.
        In these cases, the program should return a new density matrix with an empty vector data, 0 qubit.
    """
    dm = dm_simu_rs.new_empty_dm()
    dm_arr = dm_simu_rs.get_dm(dm)
    assert dm_simu_rs.get_nqubits(dm) == 0
    assert len(dm_arr) == 1
    assert dm_arr[0] == 1. + 0j


@hyp.given(sv_st(max=6))
def test_from_statevec(array):
    nqubits = sv_get_nqubits(array)

    # print(f'Testing with a statevec of size {len(array)} => nqubits = {nqubits}')
    norm = get_norm(np.outer(array, array.conj()), nqubits)
    try:
        # print(array)
        dm = dm_simu_rs.new_dm_from_statevec(array)
        assert norm != 0
        # print(f'Successfully created a dm of size {len(dm_simu_rs.get_dm(dm))}')
    except ValueError:
        assert norm == 0
        return
    dm_nqubits = dm_simu_rs.get_nqubits(dm)
    # print(f"DM SIMU NQUBITS = {dm_nqubits}")
    assert dm_nqubits == nqubits
    # print(f"nb qubits OK.")
    dm_array = dm_simu_rs.get_dm(dm)
    size = 1 << nqubits
    assert len(dm_array) == size * size
    # print(f"dm size OK.\n==========================")
    array = np.outer(array, array.conj())
    array /= norm
    np.testing.assert_allclose(array.flatten(), dm_array)

@hyp.given(sv_st(max=5), sv_st(max=5))
@hyp.settings(deadline=None)
def test_tensor_dm(array1, array2):
    dm_1 = dm_simu_rs.new_dm_from_statevec(array1)
    dm_2 = dm_simu_rs.new_dm_from_statevec(array2)
    nqubits_1 = sv_get_nqubits(array1)
    nqubits_2 = sv_get_nqubits(array2)
    ref_1 = np.outer(array1, array1.conj())
    ref_2 = np.outer(array2, array2.conj())
    norm_1 = get_norm(ref_1, nqubits_1)
    norm_2 = get_norm(ref_2, nqubits_2)
    ref_1 /= norm_1
    ref_2 /= norm_2

    dm_simu_rs.tensor_dm(dm_1, dm_2)
    
    ref = np.kron(ref_1, ref_2)
    array = dm_simu_rs.get_dm(dm_1)
    np.testing.assert_allclose(array, ref.flatten())
    np.testing.assert_equal(dm_simu_rs.get_nqubits(dm_1), nqubits_1 + nqubits_2)

@hyp.given(
    sv_st(),
    hyp.strategies.sampled_from(op_single),
)
@hyp.settings(deadline=None)
def test_expectation_single(sv, op):
    dm = dm_simu_rs.new_dm_from_statevec(sv)
    nqubits = sv_get_nqubits(sv)
    
    dm_arr = dm_simu_rs.get_dm(dm)
    dm_arr = np.reshape(dm_arr, (2 ** nqubits, 2 ** nqubits))
    index = np.random.randint(0, nqubits)
    
    # print(f"Expectation single test args:")
    # print(f"\tnqubits: {nqubits}\n\tdm{dm_arr}\n\top:{op}\n\ti:{index}")
    ref = expectation_single(dm_arr, op, index, nqubits)
    
    # print(f"\texpected: {ref}\n===============================")
    
    op_rs = dm_simu_rs.new_op(op.flatten())
    result = dm_simu_rs.expectation_single(dm, op_rs, index)
    
    #print(f"result rs: {result}\n===============================")
    
    tol = 1e-8
    assert(abs(result - ref) < tol)
    

@hyp.given(
        sv_st(max=10),
        hyp.strategies.sampled_from(op_single)
    )
def test_evolve_single(sv: np.ndarray, op: np.ndarray):
    """
            Test for evolve density matrix with single qubit operators.
    """
    dm = dm_simu_rs.new_dm_from_statevec(sv.flatten())
    dm_ref = np.outer(sv, sv)

    dm_arr = dm_simu_rs.get_dm(dm)
    np.testing.assert_allclose(dm_arr, dm_ref.flatten())

    Nqubits_dm = dm_get_nqubits(dm_arr)
    Nqubits_ref = dm_get_nqubits(dm_ref.flatten())
    assert Nqubits_dm == Nqubits_ref
    
    target = np.random.randint(0, Nqubits_dm)
    dm_rs = dm_simu_rs.evolve_single(dm, op.flatten(), target)
    dm_ref = evolve_single(dm_ref, Nqubits_ref, op, target)

    norm_ref = get_norm(dm_ref, Nqubits_ref)
    dm_ref /= norm_ref
    
    dm_arr = dm_simu_rs.get_dm(dm)
    np.testing.assert_allclose(dm_rs, dm_ref.flatten(), atol=1e-5)


@hyp.given(
        sv_st(min=2, max=10),
        hyp.strategies.sampled_from(op_double),   
    )
@hyp.settings(deadline=None)
def test_evolve(sv: np.ndarray, op: np.ndarray):
    """
            Test for evolve density matrix with any operator.
    """
    dm = dm_simu_rs.new_dm_from_statevec(sv)
    dm_ref = np.outer(sv, sv)

    dm_arr = dm_simu_rs.get_dm(dm)
    np.testing.assert_allclose(dm_arr, dm_ref.flatten())

    Nqubits_dm = dm_get_nqubits(dm_arr)
    Nqubits_ref = dm_get_nqubits(dm_ref.flatten())
    assert Nqubits_dm == Nqubits_ref
    
    # print(f'nqubits = {Nqubits_dm}')
    targets = tuple(np.random.choice(range(Nqubits_dm), size=2, replace=False))
    dm_after_rs = dm_simu_rs.evolve(dm, op.flatten(), targets)
    dm_ref = evolve(dm_ref, Nqubits_ref, op.reshape(4, 4), targets)
    #print(f'{dm_simu_rs.get_dm(dm)}')
    #print(f'{dm_ref.flatten()}')
    
    norm_ref = get_norm(dm_ref, Nqubits_ref)
    dm_ref /= norm_ref

    # if Nqubits_dm < 3:
    #     dm_after_rs = np.array(dm_after_rs).reshape((2 ** Nqubits_dm, 2 ** Nqubits_dm))
    #     print(f"TEST EVOLVE, nqubits: {Nqubits_dm}, targets = {targets}")
    #     print(f"initial ref: {sv}")
    #     print(f"ref output:\n{dm_ref}")
    #     print(f"rs output:\n{dm_after_rs}")
    
    np.testing.assert_allclose(dm_after_rs.flatten(), dm_ref.flatten(), atol=1e-5)

@hyp.given(
    sv_st(min=2, max=10),
)
@hyp.settings(deadline=None)
def test_ptrace(sv):
    nqubits = sv_get_nqubits(sv)

    rust_dm = dm_simu_rs.new_dm_from_statevec(sv)
    ref_dm = np.outer(sv, sv.conj())
    
    qargs = np.random.choice(range(nqubits), size=np.random.randint(1, nqubits), replace=False)
    qargs = list(qargs)

    dm_simu_rs.ptrace(rust_dm, qargs)

    dm_after_ref, nqubits_after_ref = ptrace(ref_dm, qargs)
    
    dm_after_rs = dm_simu_rs.get_dm(rust_dm)
    nqubits_after_rs = dm_simu_rs.get_nqubits(rust_dm)
    
    np.testing.assert_almost_equal(dm_after_rs, dm_after_ref.flatten(), decimal=5)
    np.testing.assert_equal(nqubits_after_rs, nqubits_after_ref)

@hyp.given(
    sv_st(min=2, max=10)
)
def test_entangle(sv):
    nqubits = sv_get_nqubits(sv)

    rust_dm = dm_simu_rs.new_dm_from_statevec(sv)
    ref_dm = np.outer(sv, sv.conj())

    qargs = tuple(np.random.choice(range(nqubits), size=2, replace=False))


    dm_after_rs = dm_simu_rs.entangle(rust_dm, qargs)

    CZ = op_double[1]

    dm_after_ref = evolve(ref_dm, nqubits, CZ.reshape(4, 4), qargs)

    # if nqubits < 3:
    #     dm_after_rs = np.array(dm_after_rs).reshape((2 ** nqubits, 2 ** nqubits))
    #     print(f"TEST ENTANGLE, nqubits: {nqubits}, targets = {qargs}")
    #     print(f"initial ref: {sv}")
    #     print(f"ref output:\n{dm_after_ref}")
    #     print(f"rs output:\n{dm_after_rs}")
    #     print("=======================================")

    np.testing.assert_almost_equal(dm_after_rs.flatten(), dm_after_ref.flatten(), decimal=5)

