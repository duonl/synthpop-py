from numpy.random import RandomState
from synthpop.data_processing.Encoders import PCAEncoder, MeanEncoder
from synthpop.data_processing.missing_value_handling import BaseMissingValueHandler, MissingValuePredictor, ReplaceNoneWithValue
from synthpop.methods import base_synth
import pandas as pd
from typing import Literal, Mapping, Self, Sequence
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin

class TreeClassifierMethod(DecisionTreeClassifier):
    '''
    A decision tree classifier algorithm, augmented with PCA encoding and NA predictor.

    :param encoder: an transformer object. Default is PCA encoder.
    :param rest: Parameters inherent to DecisionTreeClassifier
    '''

    def __init__(self, *, 
                 encoder: TransformerMixin = PCAEncoder(),
                 NaNHandling: BaseMissingValueHandler = ReplaceNoneWithValue(),
                 criterion: Literal['gini'] | Literal['entropy'] | Literal['log_loss'] = "gini", 
                 splitter: Literal['best'] | Literal['random'] = "best", 
                 max_depth: None | int  = None, 
                 min_samples_split: float = 2, 
                 min_samples_leaf: float = 1, 
                 min_weight_fraction_leaf: float = 0, 
                 max_features: float | None | Literal['auto'] | Literal['sqrt'] | Literal['log2'] = None, 
                 random_state: RandomState | None | int = None, #mandated by scikit-learn developer guide
                 max_leaf_nodes: None | int  = None, 
                 min_impurity_decrease: float = 0, 
                 class_weight: None | Mapping | str | Sequence[Mapping] = None, 
                 ccp_alpha: float = 0,
                ) -> None:
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha
                        )
        self.random_state = random_state #mandated by scikit-learn developer guide
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit PCA encoder on X, build a decision tree classifier on (encoded_X, y), and build a decision tree classifier to forecast missing values in y. 
        
        :param X: Features dataset.
        :param y: Target variable. Length must be equal to number of rows in X.
        :return: Fitted estimator.
        """
        # sklearn.utils.validation.validate_data is called in super().fit(), therefore we don't need to do it here.
        self.random_state_ = check_random_state(self.random_state)#mandated by scikit-learn developer guide since we need the rng after fitting.
        self.encoder.fit(X)
        data_encoded = self.encoder.transform(X)
        super().fit(data_encoded, y)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Go through decision tree classifier using the encoded X and sample from the corresponding leaf node. Use the NullPredictor to determine which output values should be missing.
        
        :param X: Input dataset
        :return: Input dataset with predicted column.
        """
        # should call sklearn.utils.validation.check_is_fitted(self),
        return pd.DataFrame()
    
    def get_feature_names_out(self):
        pass


class TreeRegressorMethod(DecisionTreeRegressor):
    '''
    A decision tree regressor algorithm, augmented with PCA encoding and NA predictor.

    :param encoder: an transformer object. Default is PCA encoder.
    :param rest: Parameters inherent to DecisionTreeRegressor
    '''

    #De vscode autocomplete was erg specifiek met typehints. Het was op het niveau van hoeveel bits sommige numeric velden mogen zijn. Die heb ik weg gehaald, omdat we niet zo gedetailleerd werken.
    def __init__(self, *,
                encoder: TransformerMixin = MeanEncoder(),
                NaNHandling: BaseMissingValueHandler = MissingValuePredictor(),
                criterion: Literal['squared_error'] | Literal['friedman_mse'] | Literal['absolute_error'] | Literal['poisson'] = "squared_error", 
                splitter: Literal['best'] | Literal['random'] = "best", 
                max_depth: None | int  = None, 
                min_samples_split: float = 2, 
                min_samples_leaf: float = 1, 
                min_weight_fraction_leaf: float = 0, 
                max_features: float | None | Literal['auto'] | Literal['sqrt'] | Literal['log2'] = None, 
                random_state: RandomState | None | int = None, 
                max_leaf_nodes: None | int = None, 
                min_impurity_decrease: float = 0, 
                ccp_alpha: float = 0
                ) -> None:

        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha)
        self.random_state = random_state #mandated by scikit-learn developer guide

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit mean encoder on X, build a decision tree regressor on (encoded_X, y), and build a decision tree classifier to forecast missing values in y. 
        
        :param X: Features dataset.
        :param y: Target variable. Length must be equal to number of rows in X.
        :return: Fitted estimator.
        """
        # sklearn.utils.validation.validate_data is called in super().fit(), therefore we don't need to do it here.
        self.random_state_ = check_random_state(self.random_state)#mandated by scikit-learn developer guide since we need the rng after fitting.
        self.encoder.fit(X)
        data_encoded = self.encoder.transform(X)
        super().fit(data_encoded, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Go through decision tree classifier using the encoded X and sample from the corresponding leaf node. Use the NullPredictor to determine which output values should be missing.
        
        :param X: Input dataset
        :return: Input dataset with predicted column.
        """
        # should call sklearn.utils.validation.check_is_fitted(self), 
        return pd.DataFrame()
    
    def get_feature_names_out(self):
        pass

    

class CartMethod(base_synth.BaseSynthMethod):
    """
    Assigns the right decision tree model based on the target variable data type: if numeric, we use a regressor, if categorical, we use a classifier.
    
    :class:`CartMethod` is the default method in :class:`Synthesiser`. Following requirements of its parent class :class:`BaseSynthMethod`, a fit and a transform methods are implemented.

    :param regressor: a TreeRegressorMethod object. It is the selected algorithm if the target variable is numeric. 
    :param classifier: a TreeClassifierMethod object. It is the selected algorithm if the target variable is categorical.

    """
    # De clone method kijkt naar de signature van de __init__ om te bepalen wat de parameters zijn van een estimator
    # Dat daar andere estimators zijn kan in de weg zitten. Echter, meta estimators doen dit ook. De code van TransformedTargetRegressor().get_params() laat zien dat:
    # 1. BaseEstimators implementeerd get_params(), 2. De parameters van subestimators worden recursief meegenomen.
    def __init__(self, regressor: TreeRegressorMethod = TreeRegressorMethod(), classifier: TreeClassifierMethod = TreeClassifierMethod()) -> None:
        super().__init__()
        # Sklearn schrijft voor dat het overnemen van parameter het enige is dat in de __init__ mag gebeuren.
        # zie https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.reg = regressor
        self.classi = classifier

        # stel de gebruiker doet het volgende:
        # CartSynth(minleaves = 6, regressor = CartRegressorSynth(minleaves=7), classifier=CartClassifierSynth(minleaves=8))
        # Wat is dan de waarde van minleaves die sklearn gebruikt?
        # Dit is de reden waarom we de parameters zoals minleaves niet in deze __init__ doen.

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

        # maakt eerst een clone van de regressor of classifier
        # The return values of (TreeRegressorMethod/TreeClassifierMethod).fit() should be stored in an attribute that ends in an underscore.
        # In this way, the check_is_fitted method still works. See https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a new column to ``X`` using the fitted model.
        
        :param X: Input dataset
        :return: Input dataset with predicted column.
        """
        #should call sklearn.utils.validation.check_is_fitted(self), 
        return pd.DataFrame()
    

    def get_feature_names_out(self):
        #delegates to TreeRegressorMethod/TreeClassifierMethod
        pass
