"""
The random state manager itself has 3 main states:
1. uninitialised
2. initialised
3. overwritten. (temporary seed using __enter__ and __exit__)

The behaviour of the methods can be dependent on this state.
The methods can cause transitions in this state. 

"""

import re
import secrets

import numpy as np
import pytest

from synthpop.reproducibility import RandomStateManager


def set_random_state_manager_state(root_seed, seed_sequence, monkeypatch):
    """
    Set the class variables of RandomStateManager,
    """
    monkeypatch.setattr(RandomStateManager, "_root_seed", root_seed)
    monkeypatch.setattr(RandomStateManager, "_seed_sequence", seed_sequence)


@pytest.fixture
def patch_default_rng(mocker):
    expected_rng = np.random.default_rng(4)
    mock_default_rng = mocker.patch(
        "numpy.random.default_rng", return_value=expected_rng)
    return {"mock": mock_default_rng, "expected_rng": expected_rng}

# ------------------ Assertion helpers -------------------------


def assert_seed_state(expected_seed):
    assert RandomStateManager._root_seed == expected_seed
    assert isinstance(RandomStateManager._seed_sequence,
                      np.random.SeedSequence)
    assert RandomStateManager._seed_sequence.entropy == expected_seed


# ------------------ test cases ---------------------------------

def test_random_state_manager_set_root_seed_initialises(monkeypatch):
    """
    Given: The random state manager is not initialised.
    When: a root seed is provided (set_root_seed), seed is not None, 
    Then: The root seed is set with the provided value, and a seedsequence is stored from that seed.
    """

    # Given: The random state manager is not initialised.
    set_random_state_manager_state(None, None, monkeypatch)

    # When: a root seed is provided (set_root_seed), seed is not None
    RandomStateManager.set_root_seed(42)

    # Then: The root seed is set with the provided value, and a seedsequence is stored from that seed.
    assert_seed_state(42)


def test_random_state_manager_when_no_seed_provided_set_root_seed_initialises_using_secure_random(monkeypatch):
    """
    Given: The random state manager is not initialised.
    When: no root seed is provided (set_root_seed), seed is None, 
    Then: The root seed is set with the value from secrets.randbits.
    """

    # Given: The random state manager is not initialised.
    set_random_state_manager_state(None, None, monkeypatch)

    # We patch secrets.randbits so that we can control the return value
    # and assert that it has been called correctly.
    def mock_random_bits(n):
        assert n == 128, "generated seeds should be 128 bits"
        return 123
    monkeypatch.setattr(secrets, "randbits", mock_random_bits)

    # When: no root seed is provided (set_root_seed), seed is None,
    RandomStateManager.set_root_seed(None)

    # Then: The root seed is set with the value from secrets.randbits, and a seedsequence is stored from that seed.
    assert_seed_state(123)


def test_random_state_manager_given_uninitialised_when_creating_rng(monkeypatch, patch_default_rng):
    """
    Given: The random state manager is not initialised.
    When: an RNG is created (create_rng)
    Then: 
        - The random state manager gets initialised with a random seed.
        - This seed is used to create an RNG.
    """

    # Given: The random state manager is not initialised.
    set_random_state_manager_state(None, None, monkeypatch)

    # We patch secrets.randbits so that we can control the returned value and assert that the correct seed is used to create the RNG.
    def mock_random_bits(n):
        assert n == 128, "generated seeds should be 128 bits"
        return 1234
    monkeypatch.setattr(secrets, "randbits", mock_random_bits)

    # When: an RNG is created (create_rng)
    result = RandomStateManager.create_rng(seed=3)

    # Then:
    # The random state manager gets initialised with a random seed.
    assert_seed_state(1234)
    assert result is patch_default_rng["expected_rng"], "the returned RNG should be from np.random.default_rng"
    patch_default_rng["mock"].assert_called_with([1234, 3])


def test_random_state_manager_given_initialised_when_creating_rng(monkeypatch, patch_default_rng):
    """
    Given: The random state manager is initialised.
    When: an RNG is created (create_rng)
    Then: 
        - The root seed is used to create an RNG
    """

    # Given: The random state manager is initialised.
    set_random_state_manager_state(3, np.random.SeedSequence(3), monkeypatch)

    # When: an RNG is created (create_rng)
    result = RandomStateManager.create_rng(seed=10)

    # The root seed is used to create an RNG
    patch_default_rng["mock"].assert_called_with([3, 10])
    assert result is patch_default_rng["expected_rng"], "the returned RNG should be from np.random.default_rng"
    assert_seed_state(3)


def test_random_state_manager_given_uninitialised_when_creating_new_seed(monkeypatch, mocker):
    """
    Given: The random state manager is not initialised.
    When: an seed is created (create_new_seed)
    Then: 
        - The random state manager gets initialised with a random seed.
        - A seed is created using the seed sequence
    """

    # Given: The random state manager is not initialised.
    set_random_state_manager_state(None, None, monkeypatch)

    # Instead of patching secrets.randbits, we patch RandomStateManager.set_root_seed.
    # In the patched version, we simulate the important side effect: setting _root_seed and _seed_sequence.
    # We also assert that it has been called correctly.
    # Mocker does not seem to allow such control over side effects.
    expected_seed = 33
    expected_seed_sequence = np.random.SeedSequence(expected_seed)

    def mock_set_seed(seed):
        assert seed is None
        monkeypatch.setattr(RandomStateManager, "_root_seed", expected_seed)
        monkeypatch.setattr(RandomStateManager,
                            "_seed_sequence", expected_seed_sequence)

    monkeypatch.setattr(RandomStateManager, "set_root_seed", mock_set_seed)

    expected_returned_seed = 44
    mock_create_seed_from_sequence = mocker.patch(
        "synthpop.reproducibility._create_seed_from_sequence", return_value=expected_returned_seed)

    # When: an seed is created (create_new_seed)
    result = RandomStateManager.create_new_seed()

    # Then: The random state manager gets initialised with a random seed.
    # This asserts that set_root_seed has been called with seed=None.
    assert_seed_state(expected_seed)

    # Then: A seed is created using the seed sequence
    mock_create_seed_from_sequence.assert_called_with(expected_seed_sequence)
    assert result == expected_returned_seed


def test_random_state_manager_given_initialised_when_creating_new_seed(monkeypatch, mocker):
    """
    Given: The random state manager is initialised.
    When: an seed is created (create_new_seed)
    Then: 
        - A seed is created using the seed sequence
    """

    # Given: The random state manager is initialised.
    root_seed = 852
    expected_seed_sequence = np.random.SeedSequence(root_seed)
    set_random_state_manager_state(
        root_seed, expected_seed_sequence, monkeypatch)

    expected_returned_seed = 44
    mock_create_seed_from_sequence = mocker.patch(
        "synthpop.reproducibility._create_seed_from_sequence", return_value=expected_returned_seed)

    # When: an seed is created (create_new_seed)
    result = RandomStateManager.create_new_seed()

    # Then: A seed is created using the seed sequence
    mock_create_seed_from_sequence.assert_called_with(expected_seed_sequence)
    assert result == expected_returned_seed

    assert_seed_state(root_seed)


def test_random_state_manager_uninitialised_enter(monkeypatch):
    """
    Given: The random state manager is not initialised.
    When: an instance of RandomStateManager is created and __enter__ is called
    Then:
        - The instance properties old_seed and old_seed_sequence are None
        - The root seed is set to the seed provided when creating an instance of RandomStateManager
    """

    # Given: The random state manager is not initialised.
    set_random_state_manager_state(None, None, monkeypatch)

    # When:
    # an instance of RandomStateManager is created
    instance = RandomStateManager(seed=5)
    instance.__enter__()  # and __enter__ is called

    # Then:
    # - The instance properties old_seed and old_seed_sequence are None
    assert instance.old_seed is None
    assert instance.old_seed_sequence is None

    # - The root seed is set to the value used when creating an instance of RandomStateManager
    assert_seed_state(5)


def test_random_state_manager_initialised_enter(monkeypatch):
    """
    Given: The random state manager is initialised.
    When: an instance of RandomStateManager is created and __enter__ is called
    Then:
        - The instance properties old_seed and old_seed_sequence are set to the initial root_seed and seed_sequence.
        - The root seed is set to the seed provided when creating an instance of RandomStateManager
    """

    # Given: The random state manager is not initialised.
    initial_seed = 300
    initial_seed_sequence = np.random.SeedSequence(initial_seed)
    set_random_state_manager_state(
        initial_seed, initial_seed_sequence, monkeypatch)

    # When:
    # an instance of RandomStateManager is created
    instance = RandomStateManager(seed=6)
    instance.__enter__()  # and __enter__ is called

    # Then:
    # - The instance properties old_seed and old_seed_sequence are set to the initial root_seed and seed_sequence.
    assert instance.old_seed == initial_seed
    assert instance.old_seed_sequence is initial_seed_sequence

    # - The root seed is set to the seed provided when creating an instance of RandomStateManager
    assert_seed_state(6)


def test_random_state_manager_initialised_exit(monkeypatch):
    """
    Given: 
        - The random state manager is initialised.
        - a context block with an RandomStateManager instance has been entered (__enter__ has been called)
    When: the context block is exited (__exit__ has been called)
    Then:
        - The root seed is set to the value of the old_seed property of the instance
        - The seed sequence is set to the same object as the old_seed_sequence property of the instance
    """

    # Given:
    # - The random state manager is initialised.
    # - a context block with an RandomStateManager instance has been entered (__enter__ has been called)

    # The random state manager has been initialised, so there is an initial seed and seed sequence.

    # The context block has been entered.
    # That means that the old_seed is set to the initial seed, and old_seed_sequence is set to the initial seed sequence.
    # It also means that the root seed and seed sequence have been changed.
    overwritten_seed = 111
    overwritten_seed_sequence = np.random.SeedSequence(overwritten_seed)
    set_random_state_manager_state(
        overwritten_seed, overwritten_seed_sequence, monkeypatch)

    initial_seed = 222
    initial_seed_sequence = np.random.SeedSequence(initial_seed)

    random_state_manager = RandomStateManager(seed=overwritten_seed)
    random_state_manager.old_seed = initial_seed
    random_state_manager.old_seed_sequence = initial_seed_sequence

    # When:
    # an instance of RandomStateManager is created
    random_state_manager.__exit__()  # and __enter__ is called

    # Then:
    # - The root seed is set to the seed provided when creating an instance of RandomStateManager
    assert RandomStateManager._root_seed == initial_seed, "initial seed is not restored"

    # - The seed sequence is set to the same object as the old_seed_sequence property of the instance
    assert RandomStateManager._seed_sequence is initial_seed_sequence, "initial seed is not restored"


def test_random_state_manager_create_new_seed_returns_int():

    result = RandomStateManager.create_new_seed()
    assert isinstance(result, int)


def test_random_state_manager_raises_on_invalid_type():

    with pytest.raises(TypeError,match = ".* expects int or sequence of ints .*"):
        RandomStateManager.set_root_seed("not an integer")

    with pytest.raises(TypeError,match = ".* expects int or sequence of ints .*"):
        RandomStateManager.set_root_seed(1.2)

    with pytest.raises(ValueError,match = re.escape("expected non-negative integer")):
        RandomStateManager.set_root_seed(-1)

    with pytest.raises(TypeError,match = ".* expects int or sequence of ints .*"):
        RandomStateManager.set_root_seed(np.nan)

def test_random_state_manager_warns_on_empty_list():

    with pytest.warns(UserWarning) as record:
        RandomStateManager.set_root_seed([])

    assert len(record) == 1
    assert str(record[0].message) =="empty list as no entropy for seed. Use None for system entropy."