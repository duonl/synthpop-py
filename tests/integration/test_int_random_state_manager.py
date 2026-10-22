"""
Integration tests of RandomStateManager.
The aim is to test if RandomStateManager behaves as intended.

Testing if the intended behaviour of RandomStateManager leads
to reproducible synthetic data is done in test_int_reproducibility.py
"""
import numpy as np
import pytest

from synthpop.reproducibility import RandomStateManager


@pytest.fixture(autouse=True)
def control_random_state_manager():
    """
    These test have the assumption that they start with
    an uninitialised RandomStateManager.
    When running the test, this is not guaranteed.
    This fixture is to give that guarantee. 
    """
    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None

    yield

    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None


def get_sample_from_rng(rng):
    """
    Shorthand to draw a sample from an RNG.
    NB: for a fixed RNG, the statement
    get_sample_from_rng(rng) == get_sample_from_rng(rng)
    will be false. So if the aim is to assert equivalence of RNGs,
    keep track of the calls to that RNG.
    """
    return rng.integers(low=0, high=1000, size=100).tolist()


def test_random_state_manager_same_instance_seed_same_output():
    RandomStateManager.set_root_seed(42)

    rng1 = RandomStateManager.create_rng(5)
    rng2 = RandomStateManager.create_rng(5)

    assert isinstance(rng1, np.random.Generator)

    assert get_sample_from_rng(rng1) == get_sample_from_rng(rng2)


def test_random_state_manager_same_root_seed_same_output():
    RandomStateManager.set_root_seed(10)
    rng_a = RandomStateManager.create_rng(5)
    RandomStateManager.set_root_seed(10)
    rng_b = RandomStateManager.create_rng(5)

    assert get_sample_from_rng(rng_a) == get_sample_from_rng(rng_b)


def test_random_state_manager_different_instance_seed_different_output():
    RandomStateManager.set_root_seed(10)
    rng1 = RandomStateManager.create_rng(1)
    rng2 = RandomStateManager.create_rng(2)

    assert get_sample_from_rng(rng1) != get_sample_from_rng(rng2)


def test_random_state_manager_different_root_seed_different_output():
    RandomStateManager.set_root_seed(1)
    a = RandomStateManager.create_rng(5).integers(0, 100, 100)

    RandomStateManager.set_root_seed(2)
    b = RandomStateManager.create_rng(5).integers(0, 100, 100)

    assert not np.array_equal(a, b)


def test_random_state_manager_setting_root_seed_reproduces():
    RandomStateManager.set_root_seed(10)

    a = RandomStateManager.create_instance_seed()
    b = RandomStateManager.create_instance_seed()

    RandomStateManager.set_root_seed(10)

    a2 = RandomStateManager.create_instance_seed()
    b2 = RandomStateManager.create_instance_seed()

    assert a == a2, "instance seeds are not reproduced"
    assert b == b2, "instance seeds are not reproduced for the second seed"

    assert a != b, "instance seeds are not independent"
    assert a2 != b2, "instance seeds are not independent"


def test_random_state_manager_context_block_exits_cleanly():
    RandomStateManager.set_root_seed(0)

    rng_before = RandomStateManager.create_rng(1)
    with RandomStateManager(444):
        rng_in_context_block = RandomStateManager.create_rng(1)
    rng_after = RandomStateManager.create_rng(1)

    rng_sample = get_sample_from_rng(rng_before)

    assert rng_sample == get_sample_from_rng(rng_after), (
        "RandomStateManager is not restored after a context block"
    )
    assert rng_sample != get_sample_from_rng(rng_in_context_block)


def test_random_state_manager_reproducible_when_not_initialised():
    # The control_random_state_manager (autouse) guarantees
    # that RandomStateManager is not initialised.
    rng1 = RandomStateManager.create_rng(1)
    rng2 = RandomStateManager.create_rng(1)

    assert get_sample_from_rng(rng1) == get_sample_from_rng(rng2)


def test_random_state_manager_creates_random_root_seed_when_not_initialised():
    """
    The aim of this test is to demonstrate that a seed is generated at
    random when no seed is provided.
    The control_random_state_manager fixture guarantees that this test
    starts with an uninitialised RandomStateManager.
    During this test, the RandomStateManager is forced to an uninitialised state.
    This way, we can assert that the outcomes are different.
    """
    rng1 = RandomStateManager.create_rng(1)

    # Force RandomStateManager to be uninitialised
    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None

    rng2 = RandomStateManager.create_rng(1)

    assert get_sample_from_rng(rng1) != get_sample_from_rng(rng2), (
        "RandomStateManager does not create a random seed "
        "when no seed is given"
    )


def test_random_state_preserves_exceptions():
    RandomStateManager.set_root_seed(100)

    with pytest.raises(ValueError, match="Some error"):
        with RandomStateManager(10):
            raise ValueError("Some error")

    assert RandomStateManager._root_seed == 100


def recurse_context(n):
    if n == 0:
        return

    before = get_sample_from_rng(RandomStateManager.create_rng(300))
    with RandomStateManager(1000 - n):
        in_context_block_before_recurse = get_sample_from_rng(
            RandomStateManager.create_rng(300),
        )
        recurse_context(n - 1)
        in_context_block_after_recurse = get_sample_from_rng(
            RandomStateManager.create_rng(300),
        )
    after = get_sample_from_rng(RandomStateManager.create_rng(300))

    assert before == after, f"Recursive context block did not recover for n = {n}"
    assert in_context_block_before_recurse == in_context_block_after_recurse, (
        f"recurse_context did not recover for n = {n}"
    )
    assert in_context_block_before_recurse != before, (
        f"entering context block did not change the state for n = {n}"
    )


def test_random_state_manager_nested_context():
    RandomStateManager.set_root_seed(123)
    recurse_context(4)
