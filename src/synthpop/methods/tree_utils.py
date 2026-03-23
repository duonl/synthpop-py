import numpy as np
import pandas as pd
from typing import Self, TypeVar
import numpy.typing as npt
from sklearn.tree import BaseDecisionTree
from sklearn.utils.validation import check_is_fitted
import warnings

T = TypeVar("T")

class LeafNodeSampler():
    """
    Leaf-based synthetic target sampler for a fitted decision tree.

    This class constructs an empirical conditional distribution of target values
    within each leaf node of a fitted decision tree, and uses this distribution
    to generate synthetic target values for new inputs.

    The procedure consists of two phases:
    
    1. Fitting phase (`fit_sampler`):
        - Each sample in the input dataset is assigned to a leaf node using
            `tree.apply(X)`.
        - For each leaf node, a histogram (frequency table) of observed target
            values is constructed.
        - The resulting mapping is stored as:
            leaf_id -> {target_value -> count}

        This defines an empirical discrete distribution per leaf.
    
    2. Sampling phase (`sample_from_leaves`):
        - For each new input sample, its corresponding leaf nodes is determined.
        - A target value is sampled from the empirical distribution associated
            with that leaf, with probabilities proportional to observed counts.

    Usage:
    class TreeMethod(BaseDecisionTree):
        def __init(tree_sampler: LeafNodeSampler | None = None):
            self.tree_sampler = tree_sampler
            
        def fit(self, X, y):
            super().fit(X, y)
            if self.tree_sampler is None:
                self.tree_sampler_= LeafNodeSampler()
            else:
                self.tree_sampler_ = clone(self.tree_sampler)
            self.tree_sampler_.fit_sampler(self, X, y)
            return self
        
        def transform(self, X_syn):
            return self.tree_sampler_.sample_from_leaves(X_syn)
    """
    def __init__(self, random_state: int | np.random.Generator | None = None):
        """
        Initialise the sampler.
        :param random_state: Controls the random number generation used for sampling.
            - If `int`, it is used as a seed to initialise a new `numpy.random.Generator`
                via `np.random.default_rng`.
            - If `numpy.random.Generator`, it is used directly.
            - If `None`, a default seed (42) is used to ensure reproducibility.
        """
        self.random_state = random_state
        pass

    def fit_sampler(self, tree: BaseDecisionTree, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Fit the sampler by constructing leaf-wise target histograms.

        Each sample in `X` is assigned to a leaf node of the provided decision tree, 
        and the corresponding target values in `y` are aggregated into frequency
        histograms per leaf.

        The resulting empirical distributions are stored internally and used
        for sampling.

        This function passes any missing or non-missing values of `y` to `tree`. However, at this point
        in the CartMethod synthesis, missing values are not expected. So when missing values are seen in 
        `y`, a warning will be raised.

        :param tree: A fitted decision tree. It must implement `apply(X)` to return leaf node IDs and must
            have compatibility with sklearn's `check_is_fitted`.
        :param X: Feature matrix used to determine leaf membership. Can be any ArrayLike shape.
            No assumptions about `X` are made in this function.
        :param y: Target values corresponding to the rows of `X`. Can be any ArrayLike shape.
        :return: The fitted sampler instance.        
        """
        check_is_fitted(tree)
        
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got {X.shape[0]} and {y.shape[0]}"
            )

        leaf_ids = tree.apply(X)
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

        self.tree_ = tree

        if isinstance(self.random_state, np.random.Generator):
            self.random_state_ = self.random_state
        else:
            seed = 42 if self.random_state is None else self.random_state
            self.random_state_ = np.random.default_rng(seed)

        return self

    def sample_from_leaves(self, X_syn: npt.ArrayLike) -> np.ndarray:
        """
        Generate synthetic target values for new samples.

        For each input sample (`X_syn`), the corresponding leaf node is determined
        using the fitted decision tree. A target value is then sampled from the 
        empirical distribution associated with that leaf.

        Sampling is performed proportionally to observed frequencies:
            P(y = v | leaf) = count(v) / sum(counts in leaf)

        :param X_syn: Feature matrix of synthetic samples for which target values
            should be generated. Array-like of shape (n_samples, n_features)
        :return: Synthetic target values sampled from the leaf-wise empirical
            distributions. A np.ndarray of shape (n_samples,). The dtype is
            `object`.
        """
        
        required_attrs = ["tree_", "_leaf_map", "random_state_"]
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise AttributeError(
            f"LeafNodeSampler is not fitted. Missing attributes: {missing}. "
            "Call `fit_sampler` first.")
        
        X_syn = np.asarray(X_syn)

        leaf_ids = self.tree_.apply(X_syn)
        n_samples = len(leaf_ids)

        y_syn = np.empty(n_samples, dtype=object)

        for i, leaf_id in enumerate(leaf_ids):
            if leaf_id not in self._leaf_map:
                raise ValueError(
                    f"Leaf id {leaf_id} not seen during fitting."
                )
            
            leaf_hist = self._leaf_map[leaf_id]
            values = np.array(list(leaf_hist.keys()), dtype=object)
            counts = np.array(list(leaf_hist.values()), dtype=np.int64)

            total = counts.sum()
            if total == 0:
                y_syn[i] = np.nan
                continue
            
            cum_counts = np.cumsum(counts)
            r = self.random_state_.integers(0, total)
            idx = np.searchsorted(cum_counts, r, side="right")
            y_syn[i] = values[idx]

        return y_syn

    def clone(self):
        """
        Create a new instance of the sampler with the same configuration.

        The method only copies initialisation parameters and does not copy
        any fitted state (i.e., `_leaf_map`, `tree_` or `random_state_`).
        Similar to sklearn's `clone()`.

        :return: A new, unfitted instance of `LeafNodeSampler()` with the 
            same `random_state` setting.
        
        Examples:
        -----
        >>> sampler = LeafNodeSampler().clone()
        """
        return self.__class__(random_state=self.random_state)