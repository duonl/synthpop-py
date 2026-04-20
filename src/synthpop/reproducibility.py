import os.path
import secrets
import numpy as np
from numpy.random import SeedSequence

_root_seed = None
_root_rng = np.random.default_rng(_root_seed)
_seed_sequence = SeedSequence(_root_seed)



# now that we have a reproducible RNG, the rest of the package needs to use it.
# There are multiple possibilities here. 

# One possibility is to let the entire package use the same RNG instance. This would imply that multiple calls to fit or transform will give different results.
# Besides, it can introduce tricky to spot bugs when the RNG is altered (reset, for example). 
# It can also introduce problems with concurrency. 



# Another possibility is to use number drawn from the root RNG to seed RNGs in classes that need random numbers.
# This solves many of the problems with one "global" RNG. However, it places the burden of instanciating RNGs on classes that need random numbers.
# An possible advantage might be that it is probabily compatible with most other packages.
def get_seed():
    return _seed_sequence.spawn(1)[0]

# Another possibility is to provide RNGS to the other classes directly. This keeps the responsibility of creating RNGs in this module.
# It means we can alter the strategy for random numbers without altering the classes using them. 
# A disadvantage is that all classes are forced to accept the type of RNG that this module provides.

# The rng returned by this function should depend on the _root_seed and the seed parameter.
# same _root_seed + same seed => same RNG, for reproducibility. 
def get_rng(seed):
    return np.random.default_rng(_root_seed + seed)#_root_rng.spawn(1)[0]


class Reseed():

    _root_seed = None
    _seed_sequence = None

    @classmethod
    def set_root_seed(cls,seed,seed_sequence = None):
        cls._root_seed = seed
        if seed_sequence is None:
            cls._seed_sequence = SeedSequence(seed)
        else:
            cls._seed_sequence = seed_sequence

    @classmethod
    def get_seed(cls):
        return cls._seed_sequence.spawn(1)[0]


    @classmethod
    def get_rng(cls,seed):
        return np.random.default_rng(SeedSequence(entropy=(cls._root_seed + seed.entropy)))


    def __init__(self,seed):
        self.new_seed = seed
        self.old_seed = None

    def __enter__(self):
        self.old_seed = Reseed._root_seed
        self.old_seed_sequence = Reseed._seed_sequence
        Reseed.set_root_seed(self.new_seed)

    def __exit__(self, type, value, traceback):
        Reseed.set_root_seed(self.new_seed,self.old_seed_sequence)