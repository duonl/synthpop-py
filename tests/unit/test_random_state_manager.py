import pytest
import numpy as np

"""
The random state manager itself has 3 main states:
1. uninitialized
2. initialized
3. overwritten. (temporary seed using __enter__ and __exit__)

The behaviour of the methods can be dependent on this state.
The methods can cause transitions in this state. 

"""

def uninitialized_random_state_manager():
    #TODO: make sure the random state manager is uninitialized
    pass

def initialized_random_state_manager():
    #TODO: initialize the random state manager with a set seed. 
    #Return seed and random state
    pass

def overwritten_from_initialized_random_state_manager():
    # TODO: overwrite the random state manager from an initialized state (there is a previous seed)
    pass

def overwritten_from_uninitialized_random_state_manager():
    # TODO: overwrite the random state manager from an initialized state (there is no previous seed)
    pass


def test_random_state_manager_given_no_rootseed_when_creating_rng():
    """
    Given: the user has not specified a root seed.
    When: an RNG is created (create_rng)
    Then: 
        - A randomly generated seed is used to create an RNG,
    """

def test_random_state_manager_given_no_rootseed_when_creating_new_seed():
    """
    Given: the user has not specified a root seed.
    When: an RNG is created (create_rng)
    Then: 
        - A randomly generated seed is used to create an RNG,
    """