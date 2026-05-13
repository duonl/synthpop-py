import numpy as np
import pandas as pd
from typing import Self
import numpy.typing as npt
import warnings
from sklearn.exceptions import NotFittedError

def sample_array(rng: np.random.Generator, counts: np.ndarray, values: np.ndarray, n_samples: int):
    """
    Helper function that draws samples with replacement from the empirical distribution of an array.

    :param rng: A random number generator.
    :param counts: Array of the counts for each values.
    :param values: Array of the distinct values corresponding to `counts`.
    :n_samples: Number of samples to be drawn.
    :return: Sampled values.
    """
    cum_counts = np.cumsum(counts)
    r = rng.integers(0, counts.sum(), size=n_samples)
    idx = np.searchsorted(cum_counts, r, side="right")
    sampled = values[idx]
    return sampled

class LeafNodeSampler():
    """
    Leaf-based synthetic target sampler driven by explicit leaf IDs.

    This class constructs an empirical conditional distribution of target values
    within each leaf node of a fitted decision tree, and uses this distribution
    to generate synthetic target values for new inputs.

    The procedure consists of two phases:
    
    1. Fitting phase (`fit_sampler`):
        - Provide `leaf_ids` corresponding to each sample in `X`
        - Construct empirical distributions of `y` per leaf
        - The resulting mapping is stored as:
            leaf_id -> {target_value -> count}
    
    2. Sampling phase (`sample_from_leaves`):
        - Provide `leaf_ids` for new samples (`X_syn`)
        - A target value is sampled from the empirical distribution associated
            with that leaf, with probabilities proportional to observed counts.

    Usage:
    class TreeMethod(BaseDecisionTree):
        def __init(tree_sampler: LeafNodeSampler | None = None):
            self.tree_sampler = tree_sampler
            
        def fit(self, X, y):
            super().fit(X, y)
            leaf_ids = self.apply(X)
            if self.tree_sampler is None:
                self.tree_sampler_= LeafNodeSampler()
            else:
                self.tree_sampler_ = clone(self.tree_sampler)
            self.tree_sampler_.fit_sampler(leaf_ids, y)
            return self
        
        def transform(self, X_syn):
            leaf_ids = self.apply(X_syn)
            return self.tree_sampler_.sample_from_leaves(leaf_ids)
    """
    def __init__(self, random_state: int | np.random.Generator | None = None):
        """
        Initialise the sampler.
        :param random_state: Controls the random number generation used for sampling.
            - If `int`, it is used as a seed to initialise a new `numpy.random.Generator`
                via `np.random.default_rng`. This generator is reset with each call so output
                is consistent between calls.
            - If `numpy.random.Generator`, it is used directly. This generator is not reset with each
                call, so the state advances with each call.
            - If `None`, a default seed (42) is used to ensure reproducibility.
        """
        self.random_state = random_state
        pass

    def fit_sampler(self, leaf_ids: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Fit the sampler by constructing leaf-wise target histograms.

        This function passes any missing or non-missing values of `y` to `tree`. However, at this point
        in the CartMethod synthesis, missing values are not expected. So when missing values are seen in 
        `y`, a warning will be raised.

        :param leaf_ids:Leaf identifiers assigned to each training sample `X`. Can be any
            Array-like of shape (n_samples,).
        :param y: Target values corresponding to the rows of `X`. Can be any 
            Array-like shape (n_samples,).
        :return: The fitted sampler instance.        
        """
        
        # input validation
        leaf_ids = np.asarray(leaf_ids)
        y = np.asarray(y)

        if leaf_ids.ndim != 1:
            raise ValueError(f"leaf_ids must be 1-dimensional with shape (n_samples,), got shape {leaf_ids.shape} instead.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional with shape (n_samples,), got shape {y.shape} instead.")
        if len(leaf_ids) == 0 or len(y) == 0:
            raise ValueError("leaf_ids and y must be non-empty.")
        if leaf_ids.shape[0] != y.shape[0]:
            raise ValueError(
                f"leaf_ids and y must have the same number of samples. Got {leaf_ids.shape[0]} and {y.shape[0]} instead.")

        self._leaf_map = {}

        for leaf_id, target in zip(leaf_ids, y):
            if leaf_id not in self._leaf_map:
                self._leaf_map[leaf_id] = {}

            leaf_hist = self._leaf_map[leaf_id]
            leaf_hist[target] = leaf_hist.get(target, 0) + 1

            if pd.isna(target):
                warnings.warn(f"LeafNodeSampler sees missing values ({target}) in target `y` during fitting. "
                              "NaN values will be included in the leaf distributions and may be "
                              "sampled. Review your input data if this is unintended.")

        if isinstance(self.random_state, np.random.Generator):
        # Extract seed is not possible → treat as non-resettable (repeated calls do not give the same output)
            self._seed = None
            self.random_state_ = self.random_state
        else:
            self._seed = 42 if self.random_state is None else self.random_state
            self.random_state_ = np.random.default_rng(self._seed)
        
        self._y_dtype = np.asarray(y).dtype

        return self

    def sample_from_leaves(self, leaf_ids: npt.ArrayLike) -> np.ndarray:
        """
        Generate synthetic target values for new samples.

        For each input sample (`X_syn`), the corresponding `leaf_ids` are given. 
        A target value (`y_syn`) is then sampled from the empirical distribution 
        associated with that leaf.

        Sampling is performed proportionally to observed frequencies:
            P(y = v | leaf) = count(v) / sum(counts in leaf)

        :param leaf_ids: Leaf IDs of synthetic samples for which target values
            should be generated. Array-like of shape (n_samples,)
        :return: Synthetic target values sampled from the leaf-wise empirical
            distributions. A np.ndarray of shape (n_samples,). The dtype is
            the same as the input dtype.
        """

        if not hasattr(self, "_leaf_map") or not hasattr(self, "random_state_") or not hasattr(self, "_y_dtype"):
            raise NotFittedError("LeafNodeSampler is not fitted. Call `fit_sampler` first.")
        
        seed = getattr(self, "_seed", None)

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.random_state_ # fallback, not-resettable
        
        leaf_ids = np.asarray(leaf_ids)
        n_samples = len(leaf_ids)

        if leaf_ids.ndim != 1:
            raise ValueError(f"leaf_ids must be 1-dimensional with shape (n_samples,), got shape {leaf_ids.shape} instead.")
        if n_samples == 0:
            raise ValueError(f"leaf_ids must be non-empty.")

        y_syn = np.empty(n_samples, dtype=self._y_dtype)

        unique_leaves, inverse_indices = np.unique(leaf_ids, return_inverse=True)
        order = np.argsort(inverse_indices)
        sorted_inv = inverse_indices[order]
        split_points = np.flatnonzero(np.diff(sorted_inv)) + 1
        groups = np.split(order, split_points)

        for leaf_idx, leaf in enumerate(unique_leaves):
            if leaf not in self._leaf_map:
                raise ValueError(f"Leaf id {leaf} not seen during fitting.")

            indices = groups[leaf_idx]

            leaf_hist = self._leaf_map[leaf]
            values = list(leaf_hist.keys())
            counts = np.array(list(leaf_hist.values()))

            total = counts.sum()
            if total == 0:
                raise ValueError(f"Leaf {leaf} has an empty leaf map. This indicates a corrupted or inconsistent LeafNodeSampler state.")

            sampled = sample_array(
                rng=rng,
                counts=counts,
                values=np.asarray(values, dtype=self._y_dtype),
                n_samples=len(indices),
            )

            y_syn[indices] = sampled

        return y_syn

    def clone(self):
        """
        Create a new instance of the sampler with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state (i.e., `_leaf_map` or `random_state_`).
        Similar to sklearn's `clone()`.

        :return: A new, unfitted instance of `LeafNodeSampler()` with the 
            same `random_state` setting.
        
        Examples
        --------
        >>> sampler = LeafNodeSampler().clone()
        """
        return self.__class__(random_state=self.random_state)
    

def build_feature_matrix(X: dict[str, np.ndarray],feature_order:list[str]) -> np.ndarray:

    if set(X.keys())>set(feature_order):
        raise ValueError("cannot build feature matrix: received more columns than expected")
    if set(X.keys())<set(feature_order):
        raise ValueError("cannot build feature matrix: received less columns than expected")
    if len(X.keys()) == 0:
        return np.empty(shape=(0,0))
    
    return np.hstack([X[k].reshape(-1,1) if X[k].ndim==1 else X[k] for k in feature_order],dtype=np.float32)
