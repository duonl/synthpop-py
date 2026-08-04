import secrets
from typing import List, Sequence
import warnings

import numpy as np
from numpy.random import SeedSequence
import numpy.typing as npt


def _create_seed_from_sequence(seed_sequence: SeedSequence) -> int:

    # SeedSequence.generate_state returns the same value if you call it multiple times.
    # To guarantee independent RNGs, we need independent seeds.
    # That is why we need to generate a new seed sequence for each seed.
    new_seed_sequence = seed_sequence.spawn(1)[0]
    new_seed = new_seed_sequence.generate_state(1)[0]  # generate_state generates a list of seeds, we need only one
    return int(new_seed)


class RandomStateManager:
    """
    Manages random numbers and reproducibility in this package.

    Instances of this class can be used as a context manager to temporary switch seed:

    See `the developer guide on randomness <./developer/randomness.html>`_ and the `functional description <./develop/developer/functional_descriptions/reproducibility.html>`_ about this.

    Examples
    --------
        >>> from synthpop.reproducibility import RandomStateManager
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
    The seed sequence is used to derive independent integer seeds for
    components that require their own RNG, even if the user provided
    seed has poor statistical properties.
    """

    @classmethod
    def set_root_seed(cls, seed: int | Sequence[int] | npt.NDArray[np.integer] | None):
        """
        Set the root seed.
        The intended usage is within the Synthesiser class.
        """

        # The error messages of SeedSequence when seed is invalid are clear enough.

        if isinstance(seed, (list, tuple, np.ndarray)) and len(seed) == 0:
            warnings.warn("empty list as no entropy for seed. Use None for system entropy.", UserWarning)

        if seed is None:
            cls._root_seed = secrets.randbits(128)
        else:
            cls._root_seed = seed

        cls._seed_sequence = SeedSequence(cls._root_seed)

    @classmethod
    def create_instance_seed(cls) -> int:
        """
        Returns an independent integer seed derived from the root seed.
        It can also be used to initialise external dependencies that require integer seeds.
        external dependencies (legacy).
        The instance seeds are integers to facilitate
        combining the root seed and instance seed.

        :returns: an integer that can be used as a seed.

    Examples
    --------
        >>> from reproducibility import RandomStateManager
        >>> class UsesRandom:
        ...     def fit(self, X, y):
        ...             self.random_state_ = RandomStateManager.create_instance_seed()
        """
        if cls._seed_sequence is None:
            cls.set_root_seed(None)
        return _create_seed_from_sequence(cls._seed_sequence)

    @classmethod
    def create_rng(cls, seed: int) -> np.random.Generator:
        """
        Creates a new instance of an RNG with a fixed initial state.
        Same root seed + same seed => RNGs with identical random streams.
        This means that executing `RandomStateManager.create_rng(seed=3).integers(0, 100, size=10)` in a loop would produce the same sequence of "random" numbers each time.
        However, `RandomStateManager.create_rng(seed=3) is
        RandomStateManager.create_rng(seed=3) ` would evaluate to `False`.

        In other words, this method creates replay RNGs.

        The reason that the instance seeds are integers is to facilitate combining the root seed and instance seed.
        """

        if cls._root_seed is None:
            cls.set_root_seed(seed=None)

        # default_rng uses seed sequences internally
        # so the easiest and safest way to combine is to pass a list of seeds.
        root_seed = cls._root_seed
        if isinstance(root_seed, (list, tuple, np.ndarray)):
            entropy = [*root_seed, seed]
        else:
            entropy = [root_seed, seed]
        
        return np.random.default_rng(entropy)

    def __init__(self, seed: int) -> None:
        self.new_seed = seed

    def __enter__(self):
        self.old_seed = RandomStateManager._root_seed
        self.old_seed_sequence = RandomStateManager._seed_sequence
        RandomStateManager.set_root_seed(self.new_seed)

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None):
        RandomStateManager._root_seed = self.old_seed
        RandomStateManager._seed_sequence = self.old_seed_sequence
