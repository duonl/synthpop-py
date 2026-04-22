import numpy as np
from numpy.random import SeedSequence



class Reseed():
    """
    Manages random numbers and reproducibility in this package.

    Instances of this class can be used as a context manager to temporary switch seed:
    
    Example:
        >>> from reproducibility import Reseed
        >>> Reseed.set_root_seed(42)
        >>> Reseed.get_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])
        >>> Reseed.get_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])
        >>> with Reseed(6):     
        ...   Reseed.get_rng(seed=7).integers(0, 100, 3)
        ...   Reseed.get_rng(seed=7).integers(0, 100, 3)
        ...
        array([79, 17,  7])
        array([79, 17,  7])
        >>> Reseed.get_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])

    """

    _root_seed = None
    """
    The root seed is the basis for all random behaviour in this package.
    The root seed is used to create a `numpy.random.SeedSequence`.   
    """
    _seed_sequence = None
    """
    The seed sequence is used to provide proper initialisation for all RNGs used in this package, even if the user provided seed is suboptimal.
    """

    @classmethod
    def set_root_seed(cls,seed:int,seed_sequence = None):
        """
        Set the root seed.
        """
        cls._root_seed = seed
        if seed_sequence is None:
            cls._seed_sequence = SeedSequence(seed)
        else:
            cls._seed_sequence = seed_sequence


    @classmethod
    def get_new_seed(cls) -> int:
        """
        Returns a seed that can be used to make a RNG.
        The seed is based on the root seed.
        """
        return cls._seed_sequence.spawn(1)[0].generate_state(1)


    @classmethod
    def get_rng(cls,seed) ->np.random.Generator:
        """
        Creates an RNG.
        Same root seed + same seed => same RNG.
        This means that executing `Reseed.get_rng(seed=3).integers(0, 100, size=10)` in a loop would produce the same sequence of "random" numbers each time.
        """

        # default_rng uses seed sequences internally, so the easiest and safest way to combine is to pass a list of seeds.
        return np.random.default_rng([cls._root_seed,seed])


    def __init__(self,seed):
        self.new_seed = seed
        self.old_seed = None

    def __enter__(self):
        self.old_seed = Reseed._root_seed
        self.old_seed_sequence = Reseed._seed_sequence
        Reseed.set_root_seed(self.new_seed)

    def __exit__(self, type, value, traceback):
        Reseed.set_root_seed(self.old_seed ,self.old_seed_sequence)