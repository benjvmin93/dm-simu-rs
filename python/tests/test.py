import functools
import hypothesis as hyp
import numpy as np
import dm_simu_rs

def get_nqubits(array: np.ndarray) -> int:
    size = np.sqrt(len(array))
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


def integer_st(min=1, max=8):
    return hyp.strategies.integers(min_value=min, max_value=max)


def state_st():
    return hyp.strategies.sampled_from(states)


def build_rho_from_sv(state: np.ndarray, nqubits: int) -> np.ndarray:
    state = functools.reduce(np.kron, (state for _ in range(nqubits)), np.array(1, dtype=np.complex128))
    return np.outer(state, state).flatten()


def array_st(min_length=1, max_length=8):
    return (
        integer_st()
        .flatmap(lambda nqubits: state_st().map(lambda state: build_rho_from_sv(state, nqubits)))
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



@hyp.given(array_st())
def test_from_vec(array):

    print(f"array_size = {len(array)}")
    nqubits = get_nqubits(array)
    norm = get_norm(array, nqubits)
    try:
        dm = dm_simu_rs.new_dm_from_vec(array)
        assert norm != 0
    except ValueError:
        assert norm == 0
        return
    print(f"DM SIMU NQUBITS = {dm_simu_rs.get_nqubits(dm)}")
    print(f"nqubits = {nqubits}")
    assert dm_simu_rs.get_nqubits(dm) == nqubits
    array2 = dm_simu_rs.get_dm(dm)
    size = 1 << nqubits
    assert len(array2) == size * size
    array = np.outer(array, array)
    array /= norm
    np.testing.assert_allclose(array, array2)