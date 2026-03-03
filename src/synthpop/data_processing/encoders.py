"""
This module contains classes to encode categorical data to numeric data. 

"""
from typing import Self
from sklearn import clone
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import numpy.typing as npt



class PCAEncoder(TransformerMixin, BaseEstimator):
    """
    Transforms categorical data to one or more numeric columns.
    The user can adjust the amount of principle components by passing an instance of sklearn.decomposition.PCA to ``_pca_transform``

    :param _pca_transform: The pca transform used. See `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html/>`_ for the possible parameters.

    Examples
    ========

        >>> from synthpop.data_processing.encoders import PCAEncoder
        >>> import numpy as np
        >>> X = np.array(["a", "a","b","b","c"])
        >>> y = np.array(["x", "x","y","z","w"])
        >>> pca_encoder = PCAEncoder() 
        >>> pca_encoder.fit(X=X,y=y)
        PCAEncoder()
        >>> pca_encoder.transform(X)
        array([[ 1.4437655e+00, -1.6325523e-01,  2.1175776e-17],
            [ 1.4437655e+00, -1.6325523e-01,  2.1175776e-17],
            [-9.3231344e-01, -7.5844318e-01,  2.1175776e-17],
            [-9.3231344e-01, -7.5844318e-01,  2.1175776e-17],
            [-5.1145202e-01,  9.2169839e-01,  2.1175776e-17]], dtype=float32)

        >>> from synthpop.data_processing.encoders import PCAEncoder
        >>> import pandas as pd
        >>> X = pd.Series(["a", "a","b","b","c"],name="input_feature")
        >>> y = pd.Series(["x", "x","y","z","w"])
        >>> encoder = PCAEncoder().set_output(transform="pandas")
        >>> encoder.fit_transform(X=X,y=y)
        input_feature_pca0  input_feature_pca1  input_feature_pca2
        0            1.443766           -0.163255        2.117578e-17
        1            1.443766           -0.163255        2.117578e-17
        2           -0.932313           -0.758443        2.117578e-17
        3           -0.932313           -0.758443        2.117578e-17
        4           -0.511452            0.921698        2.117578e-17
    
    """

    def __init__(self, _pca_transform:PCA = PCA()):
        self._pca_transform = _pca_transform

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required=False
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output= True

        tags.input_tags.categorical = True
        #tags.input_tags.two_d_array = False
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan= True
        tags.input_tags.string = True

        tags.estimator_type = "transformer"
        #tags.array_api_support = True
        return tags

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Calculate the encoding.

        :param X: 1D array of categorical data. This is contains the data to be encoded.
        :param y: 1D array of categorical data. The encoding is based on this data.
        """

        self.n_features_in_ = 1

        if X.shape[0] == 0 and y.shape[0]==0:#validate_data does not work well when there is no data
            self.mapping_ = {}
            if isinstance(X,pd.Series):
                self.feature_names_in_ = [X.name]

            self.n_features_out_ = 0

            return self

        X_val,y_val = validate_data(self,X=X,y=y, validate_separately = (
            dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan")
            ,dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan")
            ))

        if isinstance(X,pd.Series):#validate data does not seem to get the name of the feature when it is a pd.Series instead of a pd.Dataframe.
            self.feature_names_in_ = [X.name]
        if X_val.ndim != 1:
            raise ValueError("X should by 1D")
        if y_val.ndim != 1:
            raise ValueError("Y should by 1D")
   
        #The alternative to using pandas here is either use scipy or DIY.
        contingency_table = pd.crosstab(X_val,y_val,) #the result of pd.crosstab is a pandas dataframe

        self._pca_transform_ = clone(self._pca_transform)# for compatibility with sklearn we need to do this.

        pca_result = self._pca_transform_ .fit_transform(
            X=contingency_table.to_numpy()
            ,y=None)

        #sklearn.decomposition.PCA implements the set_output api
        # that means that the user might configure globally that all transformers return pandas dataframes.
        if isinstance(pca_result,pd.DataFrame):
            pca_result = pca_result.to_numpy()

        self.n_features_out_ = pca_result.shape[1] #needed for get_feature_names_out

        value_mapping = {contingency_table.index[i]: pca_result[i] for i in range(pca_result.shape[0])}

        #The alternative to using pandas here is either use scipy or DIY.
        missing_contingency_table = pd.crosstab(X_val,[v is None or v is pd.NA or v is np.nan for v in y_val])
        x_such_that_y_is_always_missing = missing_contingency_table[missing_contingency_table[False]==0].index
        mapping_for_missing = {k:[None]*self.n_features_out_ for k in x_such_that_y_is_always_missing}#The values of X s.t. y is always missing.

        self.mapping_ = value_mapping | mapping_for_missing

        return self

    def transform(self,X:npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        replaces each level of ``X`` with the numerical values determined in :py:meth:`fit`

        :param X: the feature to be encoded.
        """

        mapping_including_missing = self.mapping_| {None:[None]*self.n_features_out_}

        # if X contains Nones, then the dtype of X is object.
        # In that case, X cannot be sorted.
        # many routines of numpy for finding differences between lists depend on the items being sortable.
        unique_values_in_x = np.unique([str(v) for v in X if v is not None])

        if len(np.setdiff1d(unique_values_in_x,list(mapping_including_missing.keys()),assume_unique=True)) !=0:
            raise ValueError("new values not seen during fitting when encoding.")

        return np.array(
            [# if the categorical data is represented as integers (as floats) (as in the standard sklearn tests), floating point errors can emerge when indexing the dictionary.
                mapping_including_missing[str(v) if v is not None else None] for v in X
            ],dtype=np.float32
        )

    def get_feature_names_out(self,input_features=None):

        if input_features is None:
            return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
        if input_features != self.feature_names_in_:
            raise ValueError(f"input_features is not feature_names_in_. Expected: {self.feature_names_in_}, actual: {input_features}")
        return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
    
class MeanEncoder(OneToOneFeatureMixin,TransformerMixin, BaseEstimator): 
    def __init__(self):
        pass

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Calculate average y value for each X category.
        
        :param X: Feature column.
        :param y: Target column.

        Examples
        X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
        y = pd.Series([1, 0, 2, 0, 3], name='score')

        encoder = MeanEncoder()
        encoder.fit(X, y)
        """
        # Required for get_feature_names_out
        self.feature_names_in_ = np.array([X.name], dtype=object)
        self.n_features_in_ = 1

        # Raises exception if y is not numeric
        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError(f"Column '{y.name}' must be numeric, got {y.dtype}")
        
        # Calculates encoding map
        data = pd.concat([X, y], axis=1)
        self.mapping_ = data.groupby(X.name)[y.name].mean().to_dict()

        return self

    def transform(self,X:npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        Apply mapping from fitting function to ``X`` and returns the encoded version ``X_transformed``
        
        :param X: Original column to be encoded
        :return: Encoded column

        Examples
        X = pd.Series(["red", "blue", "red", "blue", "red"], name='color')
        y = pd.Series([1, 0, 2, 0, 3], name='score')

        encoder = MeanEncoder()
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        """
        check_is_fitted(self, 'mapping_')

        unseen_X_categories = set(X.unique()) - set(self.mapping_.keys())

        if unseen_X_categories:
            # Returns only NaNs if new values are all "missing"
            if all(pd.isna(val) for val in unseen_X_categories):
                return pd.DataFrame(np.nan, index=X.index, columns=[X.name])
            # Raises error otherwise
            else:
                raise ValueError(f"Column to be encoded has unseen values: {unseen_X_categories}")
        
        # Apply encoding map to X
        X_transformed = X.map(self.mapping_)

        return X_transformed.to_frame()