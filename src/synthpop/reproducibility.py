import secrets

import numpy as np
from numpy.random import SeedSequence


def _create_seed_from_sequence(seed_sequence: SeedSequence) -> int:
    return int(seed_sequence.spawn(1)[0].generate_state(1)[0])


class RandomStateManager:
    """
    Manages random numbers and reproducibility in this package.

    Instances of this class can be used as a context manager to temporary switch seed:

    Examples
    --------
        >>> from reproducibility import RandomStateManager
        >>> RandomStateManager.set_root_seed(42)
        >>> RandomStateManager.create_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])
        >>> RandomStateManager.create_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])
        >>> with RandomStateManager(6):     
        ...   RandomStateManager.create_rng(seed=7).integers(0, 100, 3)
        ...   RandomStateManager.create_rng(seed=7).integers(0, 100, 3)
        ...
        array([79, 17,  7])
        array([79, 17,  7])
        >>> RandomStateManager.create_rng(seed=7).integers(0, 100, 3)
        array([48, 51, 33])

    """

    _root_seed = None
    """
    The root seed is the basis for all random behaviour in this package.
    The root seed is used to create a `numpy.random.SeedSequence`.   
    """
    _seed_sequence = None
    """
    The seed sequence is used to provide proper initialisation for all RNGs
    used in this package, even if the user provided seed is suboptimal.
    """

    @classmethod
    def set_root_seed(cls, seed: int | None):
        """
        Set the root seed.
        The intended usage is within the Synthesiser class.
        """

        if seed is None:
            cls._root_seed = secrets.randbits(128)
        else:
            cls._root_seed = seed

        cls._seed_sequence = SeedSequence(cls._root_seed)

    @classmethod
    def create_new_seed(cls) -> int:
        """
        Returns a seed that can be used to make a RNG.
        The seed is based on the root seed.
        It is used to make instance seeds and seeds for
        external dependencies (legacy).
        The instance seeds are integers is to facilitate
        combining the root seed and instance seed.

        :returns: an integer that can be used as a seed.

    Examples
    --------
        >>> from reproducibility import RandomStateManager
        >>> class UsesRandom:
        ...     def fit(self, X, y):
        ...             self.random_state_ = RandomStateManager.create_new_seed()
        """
        if cls._seed_sequence is None:
            cls.set_root_seed(None)
        return _create_seed_from_sequence(cls._seed_sequence)

    @classmethod
    def create_rng(cls, seed: int) -> np.random.Generator:
        """
        Creates a new instance of an RNG with a fixed initial state.
        Same root seed + same seed => same RNG.
        This means that executing `RandomStateManager.create_rng(seed=3).integers(0, 100, size=10)` in a loop would produce the same sequence of "random" numbers each time.
        However, `RandomStateManager.create_rng(seed=3) is RandomStateManager.create_rng(seed=3) ` would evaluate to `False`

        The reason that the instance seeds are integers is to facilitate combining the root seed and instance seed.
        """

        if cls._root_seed is None:
            cls.set_root_seed(seed=None)

        # default_rng uses seed sequences internally
        # so the easiest and safest way to combine is to pass a list of seeds.
        return np.random.default_rng([cls._root_seed, seed])

    def __init__(self, seed):
        self.new_seed = seed

    def __enter__(self):
        self.old_seed = RandomStateManager._root_seed
        self.old_seed_sequence = RandomStateManager._seed_sequence
        RandomStateManager.set_root_seed(self.new_seed)

    def __exit__(self, type=None, value=None, traceback=None):
        RandomStateManager._root_seed = self.old_seed
        RandomStateManager._seed_sequence = self.old_seed_sequence
