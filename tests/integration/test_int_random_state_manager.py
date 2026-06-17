"""
Integration tests of RandomStateManager.
The aim is to test if RandomStateManager behaves as intended.

Testing if the intended behaviour of RandomStateManager leads to reproducible synthetic data 
is done in test_int_reproducibility.py
"""

import pytest

from synthpop.reproducibility import RandomStateManager


def get_sample(instance_seed=30):
    return RandomStateManager.create_rng(instance_seed).integers(low=0, high=1000, size=100).tolist()


def assert_rngs_reproducible():
    rng1 = RandomStateManager.create_rng(seed=1)
    rng2 = RandomStateManager.create_rng(seed=1)

    assert not (rng1 is rng2), "create_rng should return new RNG objects"

    sample1 = rng1.integers(low=0, high=1000, size=100).tolist()
    sample2 = rng2.integers(low=0, high=1000, size=100).tolist()

    assert sample1 == sample2, "create_rng does not produce reproducible RNGs"

    rng3 = RandomStateManager.create_rng(seed=2)
    sample3 = rng3.integers(low=0, high=1000, size=100).tolist()

    assert sample3 != sample1 and sample3 != sample2, \
        "create_rng does not produce independent RNGs"


def assert_create_new_seed_independence():
    first_seed = RandomStateManager.create_new_seed()
    second_seed = RandomStateManager.create_new_seed()

    assert first_seed != second_seed,\
          "create_new_seed does not produce independent seeds"


def test_random_state_manager_default_random_seed():
    """
    Test reproducibility of RNGs when no seed has been set
    """
    first_seed1 = RandomStateManager.create_new_seed()
    assert_create_new_seed_independence()
    assert_rngs_reproducible()

    # Assert that a random seed is used when no seed is given
    rng1 = RandomStateManager.create_rng(3)  # RNG created with random seed

    # Reset RandomStateManager
    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None

    first_seed2 = RandomStateManager.create_new_seed()
    assert_create_new_seed_independence()
    # RNG created with random seed (hopefully) another random seed
    rng2 = RandomStateManager.create_rng(3)

    sample1 = rng1.integers(low=0, high=1000, size=100).tolist()
    sample2 = rng2.integers(low=0, high=1000, size=100).tolist()
    assert sample1 != sample2, "no independent seed is used to create RNGs"

    assert first_seed1 != first_seed2, \
        "no independent seeds are produced by create_new_seed"


def test_random_state_manager_provided_seed():

    sample1 = get_sample()

    RandomStateManager.set_root_seed(42)
    sample2 = get_sample()
    first_seed1 = RandomStateManager.create_new_seed()

    assert sample1 != sample2

    assert_rngs_reproducible()
    assert_create_new_seed_independence()

    # Assert that a change of root seeds causes a change in RNG samples.
    RandomStateManager.set_root_seed(40)
    first_seed2 = RandomStateManager.create_new_seed()
    sample3 = get_sample()
    assert sample3 != sample2
    assert first_seed1 != first_seed2

    # Assert that returning to a previous seed reproduces samples from RNGs
    RandomStateManager.set_root_seed(42)
    sample4 = get_sample()
    first_seed3 = RandomStateManager.create_new_seed()
    assert_rngs_reproducible()
    assert_create_new_seed_independence()
    assert sample4 == sample2, \
        "samples of RNGs are not reproduced when using the same seed"
    assert first_seed1 == first_seed3,\
          "seeds produced by create_new_seed are not reproduced when using the same seed."


def test_random_state_manager_initialised_context_manager():

    # initialise RandomStateManager
    RandomStateManager.set_root_seed(42)
    sample1 = get_sample()

    with RandomStateManager(200):
        sample2 = get_sample()

    sample3 = get_sample()

    # Reset RandomStateManager
    RandomStateManager.set_root_seed(42)
    sample4 = get_sample()
    sample5 = get_sample()

    assert sample1 == sample4,\
          "RandomStateManager did not reset when setting the seed"
    assert sample3 == sample5,\
          "Exiting the context manager did not restore RandomStateManager"
    assert sample2 != sample5, \
        "Entering the context block did not change the seed"


def test_random_state_manager_uninitialised_context_manager():

    assert RandomStateManager._root_seed is None, "test is invalid"
    with RandomStateManager(200):
        sample1 = get_sample()

    assert RandomStateManager._root_seed is None,\
          "exiting context block does not return RandomStateManager._root_seed to uninitialised"
    assert RandomStateManager._seed_sequence is None,\
          "exiting context block does not return RandomStateManager._seed_sequence to uninitialised"

    RandomStateManager.set_root_seed(200)

    sample3 = get_sample()

    assert sample1 == sample3,\
          "using the context block does not have a similar effect as setting the root seed."


def test_random_state_preserves_exceptions():

    with pytest.raises(ValueError, match="Some error"):
        with RandomStateManager(10):
            raise ValueError("Some error")
