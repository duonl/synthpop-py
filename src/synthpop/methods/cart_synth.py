"""
This module contains the CART method for synthesising data.
"""
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from synthpop import utils
from synthpop.data_processing.encoders import MeanEncoder, PCAEncoder
from synthpop.data_processing.missing_value_handling import (
    BaseMissingValueHandler,
    MissingValuePredictor,
    ReplaceMissingWithValue
)
from synthpop.methods import base_synth
import synthpop.methods.tree_utils as tree_utils
from synthpop.methods.tree_utils import LeafNodeSampler
from synthpop.reproducibility import RandomStateManager


def _to_fixed_length_string_array(a: npt.NDArray) -> npt.NDArray:
    """
    Converts an array of StringDType to an array of fixed length string dtype.
    Missing values are not supported.
    """
    max_length = max(len(v) for v in a)
    return a.astype("U" + str(max_length))


class _AbstractTreeMethod(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    :param tree: a Decision Tree to construct the conditional probability distributions.
    :param encoder: a transformer object.
    :param missing_handler: handler for missing values in the target variable.
    :param tree_sampler: a  :class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.
    :param rare_categories_threshold: Threshold for when a categorical value is considered rare.
        If a categorical predictor contains values occurring fewer than this threshold, 
        an exception is raised to prevent potential :ref:`unintended attribute disclosure <612-attribute-disclosure>`.
        If set to an integer, categories occurring fewer than this threshold raise an exception.
        If set to ``0``, the check is disabled.
        .. note:: Contrary to :func:`tune_cart`, ``rare_categories_threshold`` should not be set to ``None``.

    """

    def __init__(
        self,
        *,
        rare_categories_threshold: int = 5,
        tree: BaseDecisionTree | None = None,
        encoder: TransformerMixin | None = None,
        missing_handler: BaseMissingValueHandler | None = None,
        tree_sampler: LeafNodeSampler | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.missing_handler = missing_handler
        self.tree_sampler = tree_sampler
        self.tree = tree
        self.rare_categories_threshold = rare_categories_threshold

    def _new_encoder(self):
        return clone(self.encoder) if self.encoder is not None else self._get_encoder()

    def _new_missing_handling(self):
        return self.missing_handler.clone() if self.missing_handler is not None else self._get_missing_handling()

    def _new_tree_sampler(self):
        return self.tree_sampler.clone() if self.tree_sampler is not None else LeafNodeSampler()

    def _new_tree(self):
        return clone(self.tree) if self.tree is not None else self._get_tree()

    def _convert_y(self, y: npt.NDArray) -> npt.NDArray:
        # overwritten in TreeClassifierMethod and TreeRegressorMethod
        return y

    def fit(self, X: Dict[str, npt.NDArray], y: npt.NDArray) -> Self:
        """
        Fit to predict `y` using `X`

        If the target consists entirely of missing values, 
        the estimator records this state and returns without fitting a tree. 
        Subsequent calls to transform() produce an all-missing output

        :param X: features to predict `y`.
        :param y: target to synthesise.

        """

        if self.rare_categories_threshold > 0:
            for key, values in X.items():
                is_numeric = pd.api.types.is_numeric_dtype(values.dtype)
                is_boolean = pd.api.types.is_bool_dtype(values.dtype)
                if (not is_numeric) or is_boolean:
                    utils._raise_on_rare_category(
                        x=values,
                        rare_threshold=self.rare_categories_threshold,
                        name=key,
                    )
        X_val, n_samples = utils._validate_2d_dict(X)

        y = utils._validate_1d_target(y, n_samples)
        self._all_missing = False

        self.n_features_in_ = len(X.keys())
        self.feature_order_ = list(X.keys())

        if pd.isna(y).all():
            self._all_missing = True
            return self

        self.encoders_ = {name: self._new_encoder().fit(value, y) for (
            name, value) in X_val.items() if not pd.api.types.is_numeric_dtype(value.dtype)}
        self.missing_handler_ = self._new_missing_handling()

        prepared_for_fit_X, prepared_y = self.missing_handler_.prepare_data_for_fit(
            X_val, y)

        all_features_dict = {
            k: self.encoders_[k].transform(v)
            if k in self.encoders_
            else v
            for (k, v) in prepared_for_fit_X.items()
        }
        all_features = tree_utils._build_feature_matrix(
            all_features_dict, self.feature_order_)

        self.tree_ = tree_utils._fit_decision_tree_with_reachable_leaves(
            decision_tree=self._new_tree(),
            X=all_features,
            y=self._convert_y(prepared_y)
        )

        leaf_ids = self.tree_.apply(all_features)

        self.tree_sampler_ = self._new_tree_sampler().fit_sampler(leaf_ids, prepared_y)

        return self

    def transform(self, X: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Synthesise new column.

        :param X: features used to predict the target variable.

        :return: synthesised column.

        """

        if not hasattr(self, '_all_missing'):
            raise NotFittedError(
                f"{self.__class__.__name__} instance is not fitted yet.")
        elif self._all_missing:
            n = len(next(iter(X.values()))) if X else 0
            # method in child class will do correct dtype conversion
            return np.full(n, np.nan)

        # Apply encoding, sample, apply (inverse) handling of missing values.
        check_is_fitted(
            self,
            [
                "tree_",
                "encoders_",
                "missing_handler_",
                "tree_sampler_",
                "feature_order_"
            ],
        )

        X_val, _ = utils._validate_2d_dict(X)

        n_features_given = len(X.keys())
        if n_features_given != self.n_features_in_:
            raise ValueError(
                f"X has {n_features_given} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input")

        all_features_dict = {k: self.encoders_[k].transform(
            v) if k in self.encoders_ else v for (k, v) in X_val.items()}

        all_features = tree_utils._build_feature_matrix(
            all_features_dict, self.feature_order_)
        leaf_ids = self.tree_.apply(all_features)

        sample = self.tree_sampler_.sample_from_leaves(leaf_ids)
        result = self.missing_handler_.post_synth_transform(X_val, sample)

        return result

    @abstractmethod
    def _get_encoder(self):
        raise NotImplementedError

    @abstractmethod
    def _get_missing_handling(self):
        raise NotImplementedError

    @abstractmethod
    def _get_tree(self):
        raise NotImplementedError

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.input_tags.two_d_array = False
        tags.input_tags.categorical = False
        tags.input_tags.string = False
        tags.input_tags.dict = True
        tags.input_tags.allow_nan = True
        return tags


class TreeClassifierMethod(_AbstractTreeMethod):
    """
    :param tree: a Decision Tree to construct the conditional probability distributions. Default is a :class:`sklearn.tree.DecisionTreeClassifier`
    :param encoder: a transformer object to transform non-numeric data to numeric data. Default is :class:`~synthpop.data_processing.encoders.PCAEncoder`
    :param missing_handler: handler for missing values in the target variable. Default is :class:`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`
    :param tree_sampler: a  :class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.
    :param rare_categories_threshold: Threshold for when a categorical value is considered rare.
        If a categorical predictor contains values occurring fewer than this threshold, 
        an exception is raised to prevent potential :ref:`unintended attribute disclosure <612-attribute-disclosure>`.
        If set to an integer, categories occurring fewer than this threshold raise an exception.
        If set to ``0``, the check is disabled.
        Default value is 5.

    .. note:: Contrary to :func:`tune_cart`, ``rare_categories_threshold`` should not be set to ``None``.

    The output will always be a numpy array. The output will always have ``np.dtypes.StringDType(na_object=np.nan)`` as dtype.
    Missing values will always be represented with ``np.nan``.


    Examples
    --------
    >>> from synthpop.methods import TreeClassifierMethod
    >>> import numpy as np
    >>> from synthpop.utils import str_dtype
    >>>
    >>> X = {
    ...         "column1":np.array([1.1, 2.2]),
    ...         "column2":np.array([1.4, 1.2]),
    ...         "column3":np.array(["a", "b"], dtype=str_dtype)
    ...         }
    >>> y = np.array(["x", "y"], dtype=str_dtype)
    >>>
    >>> tree_method = TreeClassifierMethod()
    >>> tree_method.fit(X, y)
    TreeClassifierMethod()
    >>>
    >>> tree_method.transform(X)
    array(['x', 'y'], dtype=StringDType(na_object=nan))

    """

    def __init__(
            self,
            *,
            rare_categories_threshold: int = 5,
            tree=None,
            encoder=None,
            missing_handler=None,
            tree_sampler=None
    ) -> None:
        super().__init__(rare_categories_threshold=rare_categories_threshold, encoder=encoder, missing_handler=missing_handler,
                         tree_sampler=tree_sampler, tree=tree)

    def _get_encoder(self):
        return PCAEncoder()

    def _get_missing_handling(self):
        return ReplaceMissingWithValue()

    def _get_tree(self):
        return DecisionTreeClassifier(min_samples_leaf=5,   # equivalent to minbucket in synthpop-r
                                      min_impurity_decrease=1e-08,  # equivalent to cp in synthpop-r
                                      random_state=RandomStateManager.create_instance_seed(),
                                      )

    def _convert_y(self, y: npt.NDArray) -> npt.NDArray:
        return _to_fixed_length_string_array(y)

    def transform(self, X: Dict[str, npt.NDArray]) -> npt.NDArray:
        return super().transform(X).astype(utils.str_dtype, copy=False)


class TreeRegressorMethod(_AbstractTreeMethod):
    """
    :param tree: a Decision Tree to construct the conditional probability distributions. Default is a :class:`sklearn.tree.DecisionTreeRegressor`
    :param encoder: a transformer object to transform non-numeric data to numeric data. Default is :class:`~synthpop.data_processing.encoders.MeanEncoder`
    :param missing_handler: handler for missing values in the target variable. Default is :class:`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`
    :param tree_sampler: a  :class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.
    :param rare_categories_threshold:  Threshold for when a categorical value is considered rare.
        If a categorical predictor contains values occurring fewer than this threshold, 
        an exception is raised to prevent potential :ref:`unintended attribute disclosure <612-attribute-disclosure>`.
        If set to an integer, categories occurring fewer than this threshold raise an exception.
        If set to ``0``, the check is disabled.
        Default value is 5.

    .. note:: Contrary to :func:`tune_cart`, ``rare_categories_threshold`` should not be set to ``None``.

    The output will always be a numpy array. The output will always have ``np.float32`` as dtype.
    Missing values will always be represented with ``np.nan``.


    Examples
    --------
    >>> from synthpop.methods import TreeRegressorMethod
    >>> import numpy as np
    >>> from synthpop.utils import str_dtype
    >>>
    >>> X = {
    ...         "column1":np.array([1.1, 2.2]),
    ...         "column2":np.array([1.4, 1.2]),
    ...         "column3":np.array(["a", "b"], dtype=str_dtype)
    ...         }
    >>> y = np.array([1, 2], dtype=np.float32)
    >>>
    >>> tree_method = TreeRegressorMethod()
    >>> tree_method.fit(X, y)
    TreeRegressorMethod()
    >>>
    >>> tree_method.transform(X)
    array([1., 2.], dtype=float32)

    """

    def __init__(
            self,
            *,
            rare_categories_threshold: int = 5,
            tree=None,
            encoder=None,
            missing_handler=None,
            tree_sampler=None
    ) -> None:
        super().__init__(rare_categories_threshold=rare_categories_threshold, encoder=encoder, missing_handler=missing_handler,
                         tree_sampler=tree_sampler, tree=tree)

    def _get_encoder(self):
        return MeanEncoder()

    def _get_missing_handling(self):
        return MissingValuePredictor()

    def _get_tree(self):
        return DecisionTreeRegressor(min_samples_leaf=5,    # equivalent to minbucket in synthpop-r
                                     min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
                                     random_state=RandomStateManager.create_instance_seed()
                                     )

    def _convert_y(self, y: npt.NDArray) -> npt.NDArray:
        return y.astype(np.float32, copy=False)

    def transform(self, X: Dict[str, npt.NDArray]) -> npt.NDArray:
        return super().transform(X).astype(np.float32, copy=False)


class CartMethod(base_synth.BaseSynthMethod):
    """
    CART synthesiser wrapper that automatically selects either a
    TreeClassifierMethod or TreeRegressorMethod depending on the dtype of `y`.

    When called without existing predictors (`X` is empty), CART automatically samples to create a synthetic `y`.
    When `X` has columns during `transform` that were not present during `fit`, those columns are ignored.

    Input/output API uses pandas objects exclusively:
    - X must be a pandas DataFrame
    - y must be a pandas Series
    - transform returns a pandas Series

    Internal tree methods operate on:
    - dict[str, np.ndarray] for X
    - np.ndarray for y

    The returned pandas Series preserves the dtype of the input target variable.

    :class:`CartMethod` is the default method in :class:`Synthesiser`. As required by its parent class :class:`BaseSynthMethod`, fit and transform methods are implemented.

    :param regressor: a TreeRegressorMethod object. It is the selected algorithm if the target variable is numeric. 
    :param classifier: a TreeClassifierMethod object. It is the selected algorithm if the target variable is non-numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> from synthpop.methods import CartMethod
    >>>
    >>> X = pd.DataFrame({'age': [20, 40, 60], 'profession': ['butler', 'cook', 'cook']})
    >>> y_num = pd.Series([50, 60, 70], name='length', dtype='float32')
    >>> y_cat = pd.Series(['A', 'B', 'AB'], name='blood type', dtype='string')
    >>> method = CartMethod()
    >>> method.fit(X, y_num)                                                                                                                                                                                                    
    CartMethod()                                                                                                                                                                                                                
    >>> method.transform(X)                                                                                                                                                                                                     
    0    50.0                                                                                                                                                                                                                   
    1    70.0                                                                                                                                                                                                                   
    2    60.0                                                                                                                                                                                                                   
    Name: length, dtype: float32
    >>>
    >>> method.fit(X, y_cat)                                                                                                                                                                                                    
    CartMethod()  
    >>> method.transform(X)                                                                                                                                                                                                     
    0     A                                                                                                                                                                                                                     
    1    AB                                                                                                                                                                                                                     
    2     B                                                                                                                                                                                                                     
    Name: blood type, dtype: string         
    """

    def __init__(self,
                 regressor: TreeRegressorMethod | None = None,
                 classifier: TreeClassifierMethod | None = None) -> None:
        super().__init__()
        self.regressor = regressor
        self.classifier = classifier

    def _new_regressor(self) -> TreeRegressorMethod:
        return (
            clone(self.regressor) if self.regressor is not None else TreeRegressorMethod()
        )

    def _new_classifier(self) -> TreeClassifierMethod:
        return (
            clone(
                self.classifier) if self.classifier is not None else TreeClassifierMethod()
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Fits the CART synthesiser by assessing the data type of the target variable and
        calls the :py:meth:`fit` of the correct regressor or classifier.

        :param X: Feature dataset.
        :param y: Target variable. Length must be equal to number of rows in `X`.
        :return: Fitted estimator.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X)} instead.")
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"y must be a pandas Series, got {type(y)} instead.")
        if len(X) != len(y):
            raise ValueError(f"X and y must contain the same number of samples: "
                             f"{len(X)} != {len(y)}.")

        self.feature_names_in_ = list(X.columns)
        self.target_name_ = y.name
        self.target_dtype_ = y.dtype

        X_dict = utils._to_standardised_array_dict(X)
        y_array = utils._standardise_array_dtypes(y)

        if pd.api.types.is_numeric_dtype(y_array.dtype):
            self.method_ = self._new_regressor()
        else:
            self.method_ = self._new_classifier()

        self.method_.fit(X_dict, y_array)

        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Synthesise the target column using the fitted model.

        :param X: Feature dataset.
        :return: Synthesised target variable.
        """

        check_is_fitted(
            self,
            [
                "method_",
                "feature_names_in_",
                "target_name_",
                "target_dtype_"
            ]
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X)} instead.")

        missing_cols = [
            col for col in self.feature_names_in_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"X is missing required columns: {missing_cols}.")

        # preserve original feature ordering used during fit
        X_dict = utils._to_standardised_array_dict(X[self.feature_names_in_])

        result = self.method_.transform(X_dict)

        return pd.Series(
            result,
            index=X.index,
            name=self.target_name_,
            dtype=self.target_dtype_
        )

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        check_is_fitted(self, ["method_"])
        return self.method_.get_feature_names_out(input_features)


def tune_cart(n_leaves: int = 5, n_components: int | float | None = None, rare_categories_threshold: int | None = None) -> CartMethod:
    """
    Shortcut to set parameters of the CartMethod.

    :param n_leaves: minimum number of samples in the leaf nodes.\
        This parameter is applied to the decision trees used for classification, regression, and predicting missing values. \
        See `sklearn.tree.DecisionTreeClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_ for more information.
    :param n_components: sets the number of principal components used in encoding in the classifier. \
        For float values between 0 and 1, it is the percentage of variance that should be explained by the principal components.\
        For integers => 1, it is the number of principal components.\
        See `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ for more information.
    :param rare_categories_threshold: Threshold for when a categorical value is considered rare.
        If a categorical predictor contains values occurring fewer than this threshold, 
        an exception is raised to prevent potential :ref:`unintended attribute disclosure <612-attribute-disclosure>`.

        If set to an integer, categories occurring fewer than this threshold raise an exception.
        If set to ``0``, the check is disabled.
        If set to ``None``, the value of ``n_leaves`` is used.
        The default value is ``None`` for :func:`tune_cart`, which means the threshold
        defaults to ``n_leaves``. Since ``n_leaves`` defaults to 5, the effective
        default threshold is also 5.
        See :ref:`the user guide <612-attribute-disclosure>` and :doc:`the examples <../../examples/examples_index>` for more information. TODO: add link to example about attribute disclosure.

    :return: a CartMethod object with the parameters consistently applied.

    Examples
    --------
    >>> from synthpop.methods import CartMethod, tune_cart
    >>> from synthpop import Synthesiser
    >>> import pandas as pd
    >>> data = pd.DataFrame({"a": [1], "b": [2]})
    >>> synth = Synthesiser(random_seed=10,
    ... default_syn_method=tune_cart(n_leaves=10), 
    ... special_syn_method={"b": tune_cart(n_leaves=20)})

    """

    if rare_categories_threshold is None:
        effective_categories_threshold = n_leaves
    else:
        effective_categories_threshold = rare_categories_threshold

    return CartMethod(
        regressor=TreeRegressorMethod(
            rare_categories_threshold=effective_categories_threshold,
            tree=DecisionTreeRegressor(
                min_samples_leaf=n_leaves,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
            ),
            missing_handler=MissingValuePredictor(
                tree=DecisionTreeClassifier(min_samples_leaf=n_leaves)
            )
        ),
        classifier=TreeClassifierMethod(
            rare_categories_threshold=effective_categories_threshold,
            tree=DecisionTreeClassifier(
                min_samples_leaf=n_leaves,    # equivalent to minbucket in synthpop-r
                min_impurity_decrease=1e-08,   # equivalent to cp in synthpop-r
            ),
            encoder=PCAEncoder(
                pca_transform=PCA(n_components=n_components)
            )
        )
    )
