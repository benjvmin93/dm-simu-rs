import functools
import hypothesis as hyp
import numpy as np
import dm_simu_rs

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


op = [
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


def integer_st(min=1, max=10):
    return hyp.strategies.integers(min_value=min, max_value=max)


def state_st():
    return hyp.strategies.sampled_from(states)


def build_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    return functools.reduce(np.kron, (state for _ in range(nqubits)), np.array([1], dtype=np.complex128))


def build_rho_from_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    psi = functools.reduce(np.kron, (state for _ in range(nqubits)), np.array([1], dtype=np.complex128))
    return np.outer(psi, psi)


def array_st(min_qubits=1, max_qubits=10):  # Limit max_length to manage memory
    """
        Returns a random statevector which size is between min_qubits and max_qubits.
    """
    return (
        integer_st()
        .flatmap(lambda nqubits: state_st().map(lambda state: build_rho_from_sv(state, nqubits)))
    )


def sv_st(min=1, max=10):
    return (
        integer_st(min, max)
        .flatmap(lambda nqubits: state_st().map(lambda state: build_sv(state, nqubits)))
    )


@hyp.given(
    hyp.strategies.integers(min_value=1, max_value=10),
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


@hyp.given(array_st(max_qubits=9))
def test_from_vec(array):
    nqubits = sv_get_nqubits(array)
    print(f'Testing with a statevec of size {len(array)} => nqubits = {nqubits}')
    norm = get_norm(np.outer(array, array.conj()), nqubits)
    try:
        dm = dm_simu_rs.new_dm_from_vec(array)
        assert norm != 0
        print(f'Successfully created a dm of size {len(dm_simu_rs.get_dm(dm))}')
    except ValueError:
        assert norm == 0
        return
    dm_nqubits = dm_simu_rs.get_nqubits(dm)
    print(f"DM SIMU NQUBITS = {dm_nqubits}")
    assert dm_nqubits == nqubits
    print(f"nb qubits OK.")
    dm_array = dm_simu_rs.get_dm(dm)
    size = 1 << nqubits
    assert len(dm_array) == size * size
    print(f"dm size OK.\n==========================")
    array = np.outer(array, array.conj())
    array /= norm
    np.testing.assert_allclose(array.flatten(), dm_array)

@hyp.given(array_st(max_qubits=4), array_st(max_qubits=5))
def test_tensor_dm(array1, array2):
    dm_1 = dm_simu_rs.new_dm_from_vec(array1)
    dm_2 = dm_simu_rs.new_dm_from_vec(array2)
    nqubits_1 = sv_get_nqubits(array1)
    nqubits_2 = sv_get_nqubits(array2)
    ref_1 = np.outer(array1, array1)
    ref_2 = np.outer(array2, array2)
    norm_1 = get_norm(ref_1, nqubits_1)
    norm_2 = get_norm(ref_2, nqubits_2)
    ref_1 /= norm_1
    ref_2 /= norm_2

    dm_simu_rs.tensor_dm(dm_1, dm_2)
    
    ref = np.kron(ref_1.flatten(), ref_2.flatten())
    array = dm_simu_rs.get_dm(dm_1)
    np.testing.assert_allclose(array, ref.flatten())


@hyp.given(
        sv_st(),
        hyp.strategies.sampled_from(op_single)
    )
def test_evolve_single(sv, op):
   """
        Test for evolve density matrix with single qubit operators.
   """
   op_simu = dm_simu_rs.new_op(op.flatten())
   dm = dm_simu_rs.new_dm_from_vec(sv)
   dm_ref = np.outer(sv, sv)

   dm_arr = dm_simu_rs.get_dm(dm)
   np.testing.assert_allclose(dm_arr, dm_ref.flatten())

   Nqubits_dm = dm_get_nqubits(dm_arr)
   Nqubits_ref = dm_get_nqubits(dm_ref.flatten())
   assert Nqubits_dm == Nqubits_ref
   
   target = np.random.randint(0, Nqubits_dm)
   dm_simu_rs.evolve_single(dm, op_simu, target)
   dm_ref = evolve_single(dm_ref, Nqubits_ref, op, target)

   norm_ref = get_norm(dm_ref, Nqubits_ref)
   dm_ref /= norm_ref
   
   dm_arr = dm_simu_rs.get_dm(dm)
   np.testing.assert_allclose(dm_simu_rs.get_dm(dm), dm_ref.flatten(), atol=1e-5)


@hyp.given(
        sv_st(min=2),
        hyp.strategies.sampled_from(op)
    )
def test_evolve(sv, op):
   """
        Test for evolve density matrix with any operator.
   """
   op_simu = dm_simu_rs.new_op(op.flatten())
   dm = dm_simu_rs.new_dm_from_vec(sv)
   dm_ref = np.outer(sv, sv)

   dm_arr = dm_simu_rs.get_dm(dm)
   np.testing.assert_allclose(dm_arr, dm_ref.flatten())

   Nqubits_dm = dm_get_nqubits(dm_arr)
   Nqubits_ref = dm_get_nqubits(dm_ref.flatten())
   assert Nqubits_dm == Nqubits_ref
   
   print(f'nqubits = {Nqubits_dm}')
   targets = np.random.choice(a=range(Nqubits_dm), size=2, replace=False)
   print(f'targets = {targets}')
   dm_simu_rs.evolve(dm, op_simu, targets)
   dm_ref = evolve(dm_ref, Nqubits_ref, op, targets)
   print(f'{dm_simu_rs.get_dm(dm)}')
   print(f'{dm_ref.flatten()}')
   norm_ref = get_norm(dm_ref, Nqubits_ref)
   dm_ref /= norm_ref
   
   dm_arr = dm_simu_rs.get_dm(dm)
   np.testing.assert_allclose(dm_simu_rs.get_dm(dm), dm_ref.flatten(), atol=1e-5)