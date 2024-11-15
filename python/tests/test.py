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
    np.array([1, 0], dtype=np.complex128),  # |0>
    np.array([0, 1], dtype=np.complex128),  # |1>
    np.array([1, 1] / np.sqrt(2), dtype=np.complex128), # |+>
    np.array([1, -1] / np.sqrt(2), dtype=np.complex128) # |->
]


def integer_st(min_value=1, max_value=8):
    return hyp.strategies.integers(min_value=min_value, max_value=max_value)


def state_st():
    return hyp.strategies.sampled_from(states)


def build_rho_from_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    state = functools.reduce(np.kron, (state for _ in range(nqubits)), np.array(1, dtype=np.complex128))
    return np.outer(state, state).flatten()


def array_st(min_qubits=1, max_qubits=10):  # Limit max_length to manage memory
    """
        Returns a random statevector which size is between min_qubits and max_qubits.
    """
    return (
        integer_st(min_value=min_qubits, max_value=max_qubits)
        .flatmap(lambda nqubits: hyp.strategies.lists(state_st(), min_size=nqubits, max_size=nqubits)
                 .map(lambda selected_states: functools.reduce(np.kron, selected_states)))
    )


@hyp.given(
    hyp.strategies.integers(min_value=1, max_value=5),
    hyp.strategies.sampled_from([dm_simu_rs.Zero, dm_simu_rs.Plus]),
)
def test_new_dm(nqubits, state):
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
    ref = np.kron(ref_1, ref_2)
    array = dm_simu_rs.get_dm(dm_1)
    print(f'initial arrays :\n-{ref_1}\n-{ref_2}')
    print(f'res = {np.reshape(array, (2 ** (nqubits_1 + nqubits_2), 2 ** (nqubits_1 + nqubits_2)))}')
    print(f'ref = {ref}\n=======================')
    np.testing.assert_equal(array, ref.flatten())
