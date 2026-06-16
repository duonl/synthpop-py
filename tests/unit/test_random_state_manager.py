import pytest
import numpy as np
from synthpop.reproducibility import RandomStateManager

import secrets

"""
The random state manager itself has 3 main states:
1. uninitialized
2. initialized
3. overwritten. (temporary seed using __enter__ and __exit__)

The behaviour of the methods can be dependent on this state.
The methods can cause transitions in this state. 

"""

def set_random_state_manager_state(root_seed,seed_sequence,monkeypatch):
    """
    Set the class variables of RandomStateManager,
    """
    monkeypatch.setattr(RandomStateManager,"_root_seed",root_seed)
    monkeypatch.setattr(RandomStateManager,"_seed_sequence",seed_sequence)

@pytest.fixture
def uninitialized_random_state_manager():
    set_random_state_manager_state(None,None)

@pytest.fixture
def patch_default_rng(mocker):
    expected_rng = np.random.default_rng(4)
    mock_default_rng = mocker.patch("numpy.random.default_rng",return_value=expected_rng)
    return {"mock":mock_default_rng, "expected_rng":expected_rng}

@pytest.fixture
def initialized_random_state_manager(request):
    #TODO: initialize the random state manager with a set seed. 
    #Return seed and random state
    pass

def overwritten_from_initialized_random_state_manager():
    # TODO: overwrite the random state manager from an initialized state (there is a previous seed)
    pass

def overwritten_from_uninitialized_random_state_manager():
    # TODO: overwrite the random state manager from an initialized state (there is no previous seed)
    pass
# ------------------ Assertion helpers -------------------------
def assert_random_state_manager_is_uninitialized():
    pass

def assert_random_state_manager_is_initialized():
    pass

def assert_random_state_manager_is_overwritten():
    pass

def assert_seed_state(expected_seed):
    assert RandomStateManager._root_seed == expected_seed
    assert isinstance(RandomStateManager._seed_sequence,np.random.SeedSequence)
    assert RandomStateManager._seed_sequence.entropy == expected_seed


#------------------ test cases ---------------------------------

def test_random_state_manager_set_root_seed_initializes(monkeypatch):
    """
    Given: The random state manager is not initialized.
    When: a root seed is provided (set_root_seed), seed is not None, 
    Then: The root seed is set with the provided value, and a seedsequence is stored from that seed.
    """

    #Given: The random state manager is not initialized.
    set_random_state_manager_state(None,None,monkeypatch)

    #When: a root seed is provided (set_root_seed), seed is not None
    RandomStateManager.set_root_seed(42)

    #Then: The root seed is set with the provided value, and a seedsequence is stored from that seed.
    assert_seed_state(42)
    


def test_random_state_manager_when_no_seed_provided_set_root_seed_initializes_using_secure_random(monkeypatch):
    """
    Given: The random state manager is not initialized.
    When: no root seed is provided (set_root_seed), seed is None, 
    Then: The root seed is set with the value from secrets.randbits.
    """

    #Given: The random state manager is not initialized.
    set_random_state_manager_state(None,None,monkeypatch)
    def mock_random_bits(n):
        assert n == 128, "generated seeds should be 128 bits"
        return 123
    monkeypatch.setattr(secrets,"randbits",mock_random_bits)

    #When: no root seed is provided (set_root_seed), seed is None, 
    RandomStateManager.set_root_seed(None)

    #Then: The root seed is set with the value from secrets.randbits, and a seedsequence is stored from that seed.
    assert_seed_state(123)


def test_random_state_manager_given_uninitialized_when_creating_rng(monkeypatch,mocker,patch_default_rng):
    """
    Given: The random state manager is not initialized.
    When: an RNG is created (create_rng)
    Then: 
        - The random state manager gets initialized with a random seed.
        - This seed is used to create an RNG.
    """

    #Given: The random state manager is not initialized.
    set_random_state_manager_state(None,None,monkeypatch)
    def mock_random_bits(n):
        assert n == 128, "generated seeds should be 128 bits"
        return 1234
    monkeypatch.setattr(secrets,"randbits",mock_random_bits)

    # When: an RNG is created (create_rng)
    result = RandomStateManager.create_rng(seed=3)

    #Then:
    assert_seed_state(1234)# The random state manager gets initialized with a random seed.
    assert result is patch_default_rng["expected_rng"], "the returned RNG should be from np.random.default_rng"
    patch_default_rng["mock"].assert_called_with([1234,3])



def test_random_state_manager_given_initialized_when_creating_rng(monkeypatch,patch_default_rng):
    """
    Given: The random state manager is initialized.
    When: an RNG is created (create_rng)
    Then: 
        - The root seed is used to create an RNG
    """

    # Given: The random state manager is initialized.
    set_random_state_manager_state(3,np.random.SeedSequence(3),monkeypatch)

    #When: an RNG is created (create_rng)
    result = RandomStateManager.create_rng(seed=10)

    #The root seed is used to create an RNG
    patch_default_rng["mock"].assert_called_with([3,10])
    assert result is patch_default_rng["expected_rng"], "the returned RNG should be from np.random.default_rng"
    assert_seed_state(3)


def test_random_state_manager_given_uninitialized_when_creating_new_seed(monkeypatch,mocker):
    """
    Given: The random state manager is not initialized.
    When: an seed is created (create_new_seed)
    Then: 
        - The random state manager gets initialized with a random seed.
        - A seed is created using the seed sequence
    """

    #Given: The random state manager is not initialized.
    set_random_state_manager_state(None,None,monkeypatch)

    expected_seed = 33
    expected_seed_sequence = np.random.SeedSequence(expected_seed)
    def mock_set_seed(seed):
        assert seed is None
        monkeypatch.setattr(RandomStateManager,"_root_seed",expected_seed)
        monkeypatch.setattr(RandomStateManager,"_seed_sequence",expected_seed_sequence)

    monkeypatch.setattr(RandomStateManager,"set_root_seed",mock_set_seed)

    expected_returned_seed = 44
    mock_create_seed_from_sequence = mocker.patch("synthpop.reproducibility._create_seed_from_sequence",return_value=expected_returned_seed)

    #When: an seed is created (create_new_seed)
    result = RandomStateManager.create_new_seed()

    #Then: The random state manager gets initialized with a random seed.
    assert_seed_state(expected_seed)#This asserts that set_root_seed has been called with seed=None.

    #Then: A seed is created using the seed sequence
    mock_create_seed_from_sequence.assert_called_with(expected_seed_sequence)
    assert result == expected_returned_seed

def test_random_state_manager_given_initialized_when_creating_new_seed(monkeypatch,mocker):
    """
    Given: The random state manager is initialized.
    When: an seed is created (create_new_seed)
    Then: 
        - A seed is created using the seed sequence
    """

    #Given: The random state manager is initialized.
    root_seed = 852
    expected_seed_sequence = np.random.SeedSequence(root_seed)
    set_random_state_manager_state(root_seed,expected_seed_sequence,monkeypatch)

    expected_returned_seed = 44
    mock_create_seed_from_sequence = mocker.patch("synthpop.reproducibility._create_seed_from_sequence",return_value=expected_returned_seed)

    #When: an seed is created (create_new_seed)
    result = RandomStateManager.create_new_seed()

    #Then: A seed is created using the seed sequence
    mock_create_seed_from_sequence.assert_called_with(expected_seed_sequence)
    assert result == expected_returned_seed

    assert_seed_state(root_seed)
    
