"""
This module contains the CART method for synthesising data. 
"""
from abc import abstractmethod, ABCMeta
from typing import Self
import pandas as pd
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, BaseDecisionTree
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
import numpy.typing as npt
import numpy as np
from synthpop.data_processing.encoders import PCAEncoder, MeanEncoder
from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler, \
    MissingValuePredictor, ReplaceNoneWithValue
from synthpop.methods import base_synth
from synthpop.methods.tree_utils import LeafNodeSampler
import synthpop.methods.tree_utils as tree_utils
from synthpop.utils import validate_y, validate_dict_x,str_dtype #old line from different branch that is not yet merged
from synthpop import utils


class _AbstractTreeMethod(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    :param tree: a Decision Tree to construct the conditional probability distributions.
    :param encoder: a transformer object.
    :param missing_handler: handler for missing values in the target variable.
    :param tree_sampler: a  :class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.
    
    """

    def __init__(self, *,tree: BaseDecisionTree | None = None,
                  encoder: TransformerMixin | None = None,
                 missing_handler: BaseMissingValueHandler | None = None,
                 tree_sampler: LeafNodeSampler | None = None,
                 ):
        super().__init__()
        self.encoder = encoder
        self.missing_handler = missing_handler
        self.tree_sampler = tree_sampler
        self.tree = tree

    def _new_encoder(self):
        return clone(self.encoder) if self.encoder is not None else self._get_encoder()

    def _new_missing_handling(self):
        return clone(self.missing_handler) if self.missing_handler is not None else self._get_missing_handling()

    def _new_tree_sampler(self):
        return clone(self.tree_sampler) if self.tree_sampler is not None else LeafNodeSampler()

    def _new_tree(self):
        return clone(self.tree) if self.tree is not None else self._get_tree()


    def fit(self, X: dict[str, npt.ArrayLike], y: npt.ArrayLike) -> Self:
        """
        Fit to predict `y` using `X`

        :param X: features to predict `y`.
        :param y: target to synthesise.

        """

        self.target_name_ = getattr(y, "name", None)
        X_val, n_samples = validate_dict_x(X)
        y = validate_y(y, n_samples)

        self.n_features_in_ = len(X.keys())
        self.feature_order_ = list(X.keys())

        self.encoders_ = {name: self._new_encoder().fit(value, y) for (
            name, value) in X_val.items() if not pd.api.types.is_numeric_dtype(value.dtype)}
        self.missing_handler_ = self._new_missing_handling()

        prepared_for_fit_X, prepared_y = self.missing_handler_.prepare_data_for_fit(X_val, y)
        
        all_features_dict = {k: self.encoders_[k].transform(v) if k in self.encoders_ else v for (k,v) in prepared_for_fit_X.items()}
        all_features = tree_utils.build_feature_matrix(all_features_dict,self.feature_order_)

        self.tree_ = self._new_tree().fit(all_features, prepared_y)

        leaf_ids = self.tree_.apply(all_features)

        self.tree_sampler_ = self._new_tree_sampler().fit_sampler(leaf_ids, prepared_y)

        return self

    def transform(self, X: dict[str, npt.ArrayLike]) -> np.ndarray:
        """
        Synthesise new column

        :param X: features used to predict the target variable.

        :return: synthesised column.

        """

        # Apply encoding, sample, apply (inverse) handling of missing values.
        check_is_fitted(self
                        ,["tree_", "tree_sampler_", "missing_handler_", "feature_order_"]
                        )
        
        X_val, _ = validate_dict_x(X)

        n_features_given = len(X.keys())
        if n_features_given != self.n_features_in_:
            raise ValueError(
                f"X has {n_features_given} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input")

        
        all_features_dict = {k: self.encoders_[k].transform(v) if k in self.encoders_ else v for (k,v) in X_val.items()}

        all_features = tree_utils.build_feature_matrix(all_features_dict,self.feature_order_)
        leaf_ids = self.tree_.apply(all_features)

        sample = self.tree_sampler_.sample_from_leaves(leaf_ids)
        result = self.missing_handler_.post_synth_transform(X_val, sample)
        return result

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = getattr(self, "feature_order_", [])

        if self.target_name_ is None:
            return [input_features]

        return [self.target_name_]

        # if not (self.target_name_ is None):
        #     return [self.target_name_]
        
        # if input_features is None:
        #     input_features = getattr(self, "feature_order_", [])
        #     return [input_features[0]]

        # return input_features

    @abstractmethod
    def _get_encoder(self):
        pass

    @abstractmethod
    def _get_missing_handling(self):
        pass

    @abstractmethod
    def _get_tree(self):
        pass

    def __sklearn_tags__(self):
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
    :param missing_handler: handler for missing values in the target variable. Default is :class:`~synthpop.data_processing.missing_value_handling.ReplaceNoneWithValue`
    :param tree_sampler: a  :py:class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.

    Examples
    --------
        >>> from synthpop.methods.cart_synth import TreeClassifierMethod
        >>> import numpy as np
        >>> tree_method = TreeClassifierMethod()
        >>> X = {
        ...         "column1":np.array([1.1,2.2]),
        ...         "column2":np.array([1.4,1.2]),
        ...         "column3":np.array(["a","b"])
        ...         }
        >>> y = np.array(["x","y"])
        >>> tree_method.fit(X,y)
        TreeClassifierMethod()
        >>> tree_method.transform(X)
        array(['x', 'y'], dtype='<U1')

    """

    def __init__(self, *, tree=None,encoder=None, missing_handler=None, tree_sampler=None):
        super().__init__(encoder=encoder, missing_handler=missing_handler,
                         tree_sampler=tree_sampler, tree=tree)

    def _get_encoder(self):
        return PCAEncoder()

    def _get_missing_handling(self):
        return ReplaceNoneWithValue()

    def _get_tree(self):
        return DecisionTreeClassifier(min_samples_leaf=5, #equivalent to minbucket in synthpop-r
                                      min_impurity_decrease= 1e-08# equivalent to cp in synthpop-r
                                      ,)


class TreeRegressorMethod(_AbstractTreeMethod):
    """
    :param tree: a Decision Tree to construct the conditional probability distributions. Default is a :class:`sklearn.tree.DecisionTreeRegressor`
    :param encoder: a transformer object to transform non-numeric data to numeric data. Default is :class:`~synthpop.data_processing.encoders.MeanEncoder`
    :param missing_handler: handler for missing values in the target variable. Default is :class:`~synthpop.data_processing.missing_value_handling.MissingValuePredictor`
    :param tree_sampler: a  :py:class:`~synthpop.methods.tree_utils.LeafNodeSampler` object to sample from the leaves of the decision tree.


    Examples
    --------
        >>> from synthpop.methods.cart_synth import TreeRegressorMethod
        >>> import numpy as np
        >>> tree_method = TreeRegressorMethod()
        >>> X = {
        ...         "column1":np.array([1.1,2.2]),
        ...         "column2":np.array([1.4,1.2]),
        ...         "column3":np.array(["a","b"])
        ...         }
        >>> y = np.array([1,2])
        >>> tree_method.fit(X,y)
        TreeRegressorMethod()
        >>> tree_method.transform(X)
        array([1, 2])

    """

    def __init__(self, *, tree=None,encoder=None, missing_handler=None, tree_sampler=None):
        super().__init__(encoder=encoder, missing_handler=missing_handler,
                         tree_sampler=tree_sampler, tree=tree)

    def _get_encoder(self):
        return MeanEncoder()

    def _get_missing_handling(self):
        return MissingValuePredictor()

    def _get_tree(self):
        return DecisionTreeRegressor(min_samples_leaf=5, #equivalent to minbucket in synthpop-r
                                      min_impurity_decrease= 1e-08# equivalent to cp in synthpop-r
                                      ,)


class CartMethod(base_synth.BaseSynthMethod):
    """
    CART synthesiser wrapper that automatically selects either a
    TreeClassifierMethod or TreeRegressorMethod depending on the dtype of `y`.

    Input/output API uses pandas objects exclusively:
    - X must be a pandas DataFrame
    - y must be a pandas Series
    - transform returns a pandas Series

    Internal tree methods operate on:
    - dict[str, np.ndarray] for X
    - np.ndarray for y

    Arrays are standardised to:
    - np.float32 for numeric data
    - StringDType(na_object=np.nan) for non-numeric data

    :class:`CartMethod` is the default method in :class:`Synthesiser`. Following requirements of its parent class :class:`BaseSynthMethod`, a fit and a transform methods are implemented.

    :param regressor: a TreeRegressorMethod object. It is the selected algorithm if the target variable is numeric. 
    :param classifier: a TreeClassifierMethod object. It is the selected algorithm if the target variable is non-numeric.

    """

    def __init__(self, regressor: TreeRegressorMethod | None = None, classifier: TreeClassifierMethod | None = None) -> None:
        super().__init__()
        # see https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.regressor = regressor
        self.classifier = classifier

        # parameters of TreeRegressorMethod and TreeClassifierMethod 
        # should not be set in this __init__ to avoid contradictory values.

    def _new_regressor(self) -> TreeRegressorMethod:
        return(
            clone(self.regressor) if self.regressor is not None else TreeRegressorMethod()
        )
    
    def _new_classifier(self) -> TreeClassifierMethod:
        return(
            clone(self.classifier) if self.classifier is not None else TreeClassifierMethod()
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Fits the CART synthesiser by assessing the data type of the target variable and
        calls the :py:meth:`fit` of the correct regressor or classifier.

        :param X: Features dataset.
        :param y: Target variable. Length must be equal to number of rows in `X`.
        :return: Fitted estimator.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)} instead.")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)} instead.")
        if len(X) != len(y):
            raise ValueError(f"X and y must contain the same number of samples: {len(X)} != {len(y)}.")
        
        self.feature_names_in_ = list(X.columns)
        self.target_name_ = y.name

        X_dict = utils.to_standardised_array_dict(X)
        y_array = utils.standardise_array_dtypes(y)

        if pd.api.types.is_numeric_dtype(y_array.dtype):
            self.method_ = self._new_regressor()
        else:
            self.method_ = self._new_classifier()

        self.method_.fit(X_dict, y_array)

        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Synthesise the target column using the fitted model.

        :param X: Features dataset.
        :return: Synthesised target variable.
        """

        check_is_fitted(self, ["method_", "feature_names_in_", "target_name_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)} instead.")
        
        missing_cols = [col for col in self.feature_names_in_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"X is missing required columns: {missing_cols}.")
        
        # preserve original feature ordering used during fit
        X_dict = utils.to_standardised_array_dict(X[self.feature_names_in_])

        result = self.method_.transform(X_dict)

        return pd.Series(
            result,
            index=X.index,
            name=self.target_name_
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["method_"])
        return self.method_.get_feature_names_out(input_features)
        
        
