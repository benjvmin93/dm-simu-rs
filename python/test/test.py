import functools
import hypothesis as hyp
import numpy as np
import dm_simu_rs

def get_nqubits(array: np.ndarray) -> int:
    return len(array).bit_length() - 1


def reshape_tensor(array: np.ndarray) -> np.ndarray:
    return array.reshape((2,) * 2 * get_nqubits(array))


def get_norm(array: np.ndarray) -> float:
    return np.sqrt(np.sum(array.flatten().conj() * array.flatten()))


def is_power_of_two(n: int) -> bool:
    return n & (n - 1) == 0


complex_st = hyp.strategies.complex_numbers(min_magnitude=1e-5, max_magnitude=1e5)


def array_st(min_length=0, max_length=8):
    return (
        hyp.strategies.integers(min_value=min_length, max_value=max_length)
        .flatmap(lambda nqubits: hyp.strategies.lists(complex_st, min_size=np.square(1 << nqubits), max_size=np.square(1 << nqubits)))
        .map(lambda l: np.array(l, dtype=np.complex128))
    )


@hyp.given(
    hyp.strategies.integers(min_value=0, max_value=16),
    hyp.strategies.sampled_from([dm_simu_rs.Zero, dm_simu_rs.Plus]),
)
def test_new_dm(nqubits, state):
    vec = dm_simu_rs.new_dm(nqubits, state)
    assert dm_simu_rs.get_nqubits(vec) == nqubits
    array = dm_simu_rs.get_dm(vec)
    size = 1 << nqubits
    assert len(array) == size * size
    if state == dm_simu_rs.Zero:
        state_mat = np.array([1, 0])
    elif state == dm_simu_rs.Plus:
        state_mat = np.array([1, 1]) / np.sqrt(2)
    else:
        assert False
    ref = functools.reduce(np.kron, (state_mat for _ in range(nqubits)), np.array(1, dtype=np.complex128))
    np.testing.assert_allclose(array, ref.flatten())


@hyp.given(array_st())
def test_from_vec(array):
    nqubits = get_nqubits(array)
    norm = get_norm(array)
    try:
        vec = mbqc_rs.from_vec(array)
        assert norm != 0
    except ValueError:
        assert norm == 0
        return
    assert mbqc_rs.get_nqubits(vec) == nqubits
    array2 = mbqc_rs.get_dm(vec)
    size = 1 << nqubits
    assert len(array2) == size * size
    array /= norm
    np.testing.assert_allclose(array, array2)



