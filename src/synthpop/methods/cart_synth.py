"""
This module contains the CART method for synthesising data. 
"""
from typing import Literal, Mapping, Self, Sequence
from numpy.random import RandomState
import pandas as pd
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, BaseDecisionTree
from sklearn.base import TransformerMixin
from synthpop.data_processing.encoders import PCAEncoder, MeanEncoder
from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler, MissingValuePredictor, ReplaceNoneWithValue
from synthpop.methods import base_synth, tree_utils
from synthpop.methods.tree_utils import LeafNodeSampler
import numpy.typing as npt
from abc import abstractmethod, ABCMeta


class _AbstractTreeMethod(BaseDecisionTree, metaclass=ABCMeta):

    def __init__(self, *, encoder: TransformerMixin | None = None,
                 missing_handling: BaseMissingValueHandler | None = None,
                 tree_sampler: LeafNodeSampler | None = None,
                 criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, random_state, min_impurity_decrease, class_weight=None, ccp_alpha=0):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)
        self.encoder = encoder
        self.missing_handler = missing_handling
        self.tree_sampler = tree_sampler

    def fit(self, X: dict[str, npt.ArrayLike], y: npt.ArrayLike) -> Self:
        # Apply encoding en handling of missing values, pass on to super().fit
        self.encoders_ = {name: clone(self.encoder).fit(value,y) for (name,value) in X.items() if not pd.api.types.is_numeric_dtype(value.dtype)}
        self.missing_handler_ = clone(self.missing_handler)

        prepared_for_fit_X,prepared_y = self.missing_handler_.prepare_data_for_fit(X,y)

        encoded_features = {name:self.encoders_[name].transform(prepared_for_fit_X[name]) for name in self.encoders_.keys()}

        all_features = encoded_features | {name: value for (name,value) in prepared_for_fit_X.items() if pd.api.types.is_numeric_dtype(value.dtype)}

        super().fit(all_features,prepared_y)

        leaf_ids = super().apply(all_features)

        self.tree_sampler_ = clone(self.tree_sampler).fit_sampler(leaf_ids,prepared_y)


    def transform(self, X: dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        # Apply encoding, sample, apply (inverse) handling of missing values.
        pass

    def get_feature_names_out(self):
        pass

    @abstractmethod
    def _get_encoder(self):
        pass

    @abstractmethod
    def _get_missing_handling(self):
        pass

    # The leafnode sampler does not vary between regression and classification.
    # @abstractmethod
    # def _get_leafnode_sampler(self):
    #     pass


class TreeClassifierMethod(_AbstractTreeMethod):
    """
    A decision tree classifier algorithm, augmented with PCA encoding and NA predictor.

    :param encoder: an transformer object. Default is PCA encoder.
    :param rest: Parameters inherent to DecisionTreeClassifier
    """

    def __init__(self, *, encoder=None, missing_handling=None, tree_sampler=None,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,):
        super().__init__(encoder=encoder, missing_handling=missing_handling, tree_sampler=tree_sampler, criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)

    def _get_encoder(self):
        return PCAEncoder()

    def _get_missing_handling(self):
        return ReplaceNoneWithValue()
    
    # The distinction between regression and classification is made in BaseDecisionTree based on tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        return tags


class TreeRegressorMethod(_AbstractTreeMethod):
    """
    A decision tree regressor algorithm, augmented with PCA encoding and NA predictor.

    :param encoder: an transformer object. Default is PCA encoder.
    :param rest: Parameters inherent to DecisionTreeRegressor
    """

    def __init__(self, *, encoder=None, missing_handling=None, tree_sampler=None, criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None):
        super().__init__(encoder=encoder, missing_handling=missing_handling, tree_sampler=tree_sampler, criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state, min_impurity_decrease=min_impurity_decrease, class_weight=None, ccp_alpha=ccp_alpha)
    
    def _get_encoder(self):
        return MeanEncoder()

    def _get_missing_handling(self):
        return MissingValuePredictor()
    

    # The distinction between regression and classification is made in BaseDecisionTree based on tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags.required = True
        return tags


class CartMethod(base_synth.BaseSynthMethod):
    """
    Assigns the right decision tree model based on the target variable data type: if numeric, we use a regressor, if categorical, we use a classifier.

    :class:`CartMethod` is the default method in :class:`Synthesiser`. Following requirements of its parent class :class:`BaseSynthMethod`, a fit and a transform methods are implemented.

    :param regressor: a TreeRegressorMethod object. It is the selected algorithm if the target variable is numeric. 
    :param classifier: a TreeClassifierMethod object. It is the selected algorithm if the target variable is categorical.

    """

    def __init__(self, regressor: TreeRegressorMethod = TreeRegressorMethod(), classifier: TreeClassifierMethod = TreeClassifierMethod()) -> None:
        super().__init__()
        # see https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.reg = regressor
        self.classi = classifier

        # parameters of TreeRegressorMethod and TreeClassifierMethod should not be set in this __init__, for consistency:
        # The user could specify contradicting values.

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Assess data type of target variable and calls the :py:meth:`fit` of the correct regressor or classifier.

        :param X: Features dataset.
        :param y: Target variable. Length must be equal to number of rows in X.
        :return: Fitted estimator.
        """
        # Using sklearn.utils.validation.validate_data, set the attribute feature_names_in_ to X and y.
        # That method sets the attribute.
        # For example:
        # from sklearn.utils.validation import validate_data
        # ....
        # X, y = validate_data(self, X, y)

        # The return values of (TreeRegressorMethod/TreeClassifierMethod).fit() should be stored in an attribute that ends in an underscore.
        # In this way, the check_is_fitted method still works. See https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a new column to ``X`` using the fitted model.

        :param X: Input dataset
        :return: Input dataset with predicted column.
        """
        # should call sklearn.utils.validation.check_is_fitted(self),
        return pd.DataFrame()

    def get_feature_names_out(self, input_features=None):
        # delegates to TreeRegressorMethod/TreeClassifierMethod
        pass
