import pytest
import numpy

from synthpop.reproducibility import RandomStateManager


def assert_rngs_reproducible():
    rng1 = RandomStateManager.create_rng(seed=1)
    rng2 = RandomStateManager.create_rng(seed=1)

    assert not (rng1 is rng2), "create_rng should return new RNG objects"

    sample1 = rng1.integers(low=0,high=1000,size=100).tolist()
    sample2 = rng2.integers(low=0,high=1000,size=100).tolist()

    assert sample1 == sample2, "create_rng does not produce reproducible RNGs"

    rng3 = RandomStateManager.create_rng(seed=2)
    sample3 = rng3.integers(low=0,high=1000,size=100).tolist()

    assert sample3 != sample1 and sample3 != sample2, "create_rng does not produce independent RNGs"

def assert_create_new_seed_independence():
    first_seed = RandomStateManager.create_new_seed()
    second_seed = RandomStateManager.create_new_seed()

    assert first_seed != second_seed, "create_new_seed does not produce independent seeds"

def test_random_state_manager_default_random_seed():
    """
    Test reproducibility of RNGs when no seed has been set
    """
    first_seed1 = RandomStateManager.create_new_seed()
    assert_create_new_seed_independence()
    assert_rngs_reproducible()

    # Assert that a random seed is used when no seed is given
    rng1 = RandomStateManager.create_rng(3) #RNG created with random seed

    #Reset RandomStateManager
    RandomStateManager._root_seed = None
    RandomStateManager._seed_sequence = None

    first_seed2 = RandomStateManager.create_new_seed()
    assert_create_new_seed_independence()
    rng2 = RandomStateManager.create_rng(3) #RNG created with random seed (hopefully) another random seed
    

    sample1 = rng1.integers(low=0,high=1000,size=100).tolist()
    sample2 = rng2.integers(low=0,high=1000,size=100).tolist()
    assert sample1 != sample2, "no independent seed is used to create RNGs"

    assert first_seed1 != first_seed2, "no independent seeds are produced by create_new_seed"

def test_random_state_manager_provided_seed():

    sample1 = RandomStateManager.create_rng(30).integers(low=0,high=1000,size=100).tolist()

    RandomStateManager.set_root_seed(42)
    sample2 = RandomStateManager.create_rng(30).integers(low=0,high=1000,size=100).tolist()
    first_seed1 = RandomStateManager.create_new_seed()

    assert sample1 != sample2

    assert_rngs_reproducible()
    assert_create_new_seed_independence()

    #Assert that a change of root seeds causes a change in RNG samples.
    RandomStateManager.set_root_seed(40)
    first_seed2 = RandomStateManager.create_new_seed()
    sample3 = RandomStateManager.create_rng(30).integers(low=0,high=1000,size=100).tolist()
    assert sample3 != sample2
    assert first_seed1 != first_seed2

    #Assert that returning to a previous seed reproduces samples from RNGs
    RandomStateManager.set_root_seed(42)
    sample4 = RandomStateManager.create_rng(30).integers(low=0,high=1000,size=100).tolist()
    first_seed3 = RandomStateManager.create_new_seed()
    assert_rngs_reproducible()
    assert_create_new_seed_independence()
    assert sample4 == sample2, "samples of RNGs are not reproduced when using the same seed"
    assert first_seed1 == first_seed3, "seeds produced by create_new_seed are not reproduced when using the same seed."






