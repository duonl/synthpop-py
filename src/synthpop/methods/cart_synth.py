"""
This module contains the CART method for synthesising data. 
"""
from typing import Literal, Mapping, Self, Sequence
from numpy.random import RandomState
import pandas as pd
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, BaseDecisionTree
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from synthpop.data_processing.encoders import PCAEncoder, MeanEncoder
from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler, MissingValuePredictor, ReplaceNoneWithValue
from synthpop.methods import base_synth, tree_utils
from synthpop.methods.tree_utils import LeafNodeSampler
import numpy.typing as npt
from abc import abstractmethod, ABCMeta
import numpy as np
from sklearn.utils.validation import validate_data


class _AbstractTreeMethod(TransformerMixin,BaseEstimator,metaclass=ABCMeta):

    def __init__(self, *, encoder: TransformerMixin | None = None,
                 missing_handler: BaseMissingValueHandler | None = None,
                 tree_sampler: LeafNodeSampler | None = None,
                 tree: BaseDecisionTree | None = None):
        super().__init__()
        self.encoder = encoder
        self.missing_handler = missing_handler
        self.tree_sampler = tree_sampler
        self.tree = tree


    def _new_encoder(self):
        return clone(self.encoder) if not(self.encoder is None) else self._get_encoder()
    
    def _new_missing_handling(self):
        return clone(self.missing_handler) if not(self.missing_handler is None) else self._get_missing_handling()
    
    def _new_tree_sampler(self):
        return clone(self.tree_sampler) if not(self.tree_sampler is None) else LeafNodeSampler()
    
    def _new_tree(self):
        return clone(self.tree) if not (self.tree is None) else self._get_tree()
    

    def _validate_X(self,X):

        if isinstance(X,np.ndarray):
            X_d = {i:X[:,i] for i in range(X.shape[1])}
        else:
            X_d = X

        n_features_given = len(X_d.keys())
        if not hasattr(self,"n_features_in_"):
            self.n_features_in_ = n_features_given
        else:
            
            if n_features_given != self.n_features_in_:
                raise ValueError(f"X has {n_features_given} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input")
        return X_d
        
    def _build_X_matrix(self,encoded_features,X_prep):
        all_features_dict = encoded_features | {name: X_prep[name] for name in self.feature_order_ if pd.api.types.is_numeric_dtype(X_prep[name].dtype)}
        all_features = np.column_stack(list(all_features_dict.values()))
        return all_features


    def fit(self, X: dict[str, npt.ArrayLike], y: npt.ArrayLike) -> Self:
        # Apply encoding en handling of missing values, pass on to super().fit
        X_val = self._validate_X(X)

        self.encoders_ = {name: self._new_encoder().fit(value,y) for (name,value) in X_val.items() if not pd.api.types.is_numeric_dtype(value.dtype)}
        self.missing_handler_ = self._new_missing_handling()

        self.feature_order_ = list(X_val.keys())
        

        prepared_for_fit_X,prepared_y = self.missing_handler_.prepare_data_for_fit(X_val,y)

        encoded_features = {name:self.encoders_[name].transform(prepared_for_fit_X[name]) for name in self.encoders_.keys()}

        all_features = self._build_X_matrix(encoded_features,prepared_for_fit_X)

        self.tree_ = self._new_tree().fit(all_features,prepared_y)

        #self._fit(all_features,prepared_y)

        leaf_ids = self.tree_.apply(all_features)

        self.tree_sampler_ = self._new_tree_sampler().fit_sampler(leaf_ids,prepared_y)

        return self


    def transform(self, X: dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        # Apply encoding, sample, apply (inverse) handling of missing values.
        check_is_fitted(self)
        X_val = self._validate_X(X)
        encoded_features = {name:self.encoders_[name].transform(X_val[name]) for name in self.encoders_.keys()}

        all_features = self._build_X_matrix(encoded_features,X_val)
        leaf_ids = self.tree_.apply(all_features)

        sample = self.tree_sampler_.sample_from_leaves(leaf_ids)
        result = self.missing_handler_.post_synth_transform(X_val,sample)
        return result

    def get_feature_names_out(self):
        pass

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
        tags.target_tags.required=True
        tags.target_tags.two_d_labels=True
        tags.input_tags.two_d_array=True
        tags.input_tags.categorical=True
        tags.input_tags.string=True
        tags.input_tags.dict=True
        tags.input_tags.allow_nan =True
        return tags
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

    def __init__(self, *, encoder=None, missing_handler=None, tree_sampler=None,tree=None):
        super().__init__(encoder=encoder, missing_handler=missing_handler, tree_sampler=tree_sampler,tree=tree)

    def _get_encoder(self):
        return PCAEncoder()

    def _get_missing_handling(self):
        return ReplaceNoneWithValue()
    
    def _get_tree(self):
        return DecisionTreeClassifier()#TODO: set default params
    


class TreeRegressorMethod(_AbstractTreeMethod):
    """
    A decision tree regressor algorithm, augmented with PCA encoding and NA predictor.

    :param encoder: an transformer object. Default is PCA encoder.
    :param rest: Parameters inherent to DecisionTreeRegressor
    """

    def __init__(self, *, encoder=None, missing_handler=None, tree_sampler=None,tree = None):
        super().__init__(encoder=encoder, missing_handler=missing_handler, tree_sampler=tree_sampler, tree=tree)
    
    def _get_encoder(self):
        return MeanEncoder()

    def _get_missing_handling(self):
        return MissingValuePredictor()
    
    def _get_tree(self):
        return DecisionTreeRegressor()#TODO: set default params
    

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
