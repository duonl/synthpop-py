"""
This module contains classes to encode categorical data to numeric data. 

"""
from typing import Self
from sklearn import clone
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import numpy.typing as npt

from synthpop.utils import str_dtype, to_missing_str_array


class PCAEncoder(TransformerMixin, BaseEstimator):
    """
    Transforms categorical data to one or more numeric columns.
    The user can adjust the amount of principle components by passing an instance of sklearn.decomposition.PCA to `pca_transform`

    :param pca_transform: The pca transform used. The default value is :py:class:`sklearn.decomposition.PCA`. 
         See `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ for the possible parameters. With the default parameters, all principle components are computed and used.

    Examples
    --------

        >>> from synthpop.data_processing.encoders import PCAEncoder
        >>> import numpy as np
        >>> X = np.array(["a", "a","b","b","c"])
        >>> y = np.array(["x", "x","y","z","w"])
        >>> pca_encoder = PCAEncoder() 
        >>> pca_encoder.fit(X=X,y=y)
        PCAEncoder()
        >>> pca_encoder.transform(X)
        array([[-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
        [-1.1180340e+00, -1.5000000e+00, -1.2019867e-16],
        [ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
        [ 2.2360680e+00, -1.2953263e-15, -1.2019867e-16],
        [-1.1180340e+00,  1.5000000e+00, -1.2019867e-16]], dtype=float32)

        >>> from synthpop.data_processing.encoders import PCAEncoder
        >>> import pandas as pd
        >>> X = pd.Series(["a", "a","b","b","c"],name="input_feature")
        >>> y = pd.Series(["x", "x","y","z","w"])
        >>> encoder = PCAEncoder().set_output(transform="pandas")
        >>> encoder.fit_transform(X=X,y=y)
        input_feature_pca0  input_feature_pca1  input_feature_pca2
        0           -1.118034       -1.500000e+00       -1.201987e-16
        1           -1.118034       -1.500000e+00       -1.201987e-16
        2            2.236068       -1.295326e-15       -1.201987e-16
        3            2.236068       -1.295326e-15       -1.201987e-16
        4           -1.118034        1.500000e+00       -1.201987e-16
    
        with a different number of principle components (only the first):
        
        >>> import numpy as np
        >>> from synthpop.data_processing.encoders import PCAEncoder
        >>> from sklearn.decomposition import PCA
        >>> pca_encoder = PCAEncoder(pca_transform = PCA(n_components=1))
        >>> X = np.array(["a", "a","b","b","c"])
        >>> y = np.array(["x", "x","y","z","w"])
        >>> pca_encoder.fit_transform(X,y)
        array([[-1.118034],
        [-1.118034],
        [ 2.236068],
        [ 2.236068],
        [-1.118034]], dtype=float32)
        
        preserving 75% of variance:

        >>> pca_encoder2 = PCAEncoder(pca_transform = PCA(n_components=0.75)) 
        >>> pca_encoder2.fit_transform(X,y) 
        array([[-1.1180340e+00, -1.5000000e+00],
        [-1.1180340e+00, -1.5000000e+00],
        [ 2.2360680e+00, -1.2953263e-15],
        [ 2.2360680e+00, -1.2953263e-15],
        [-1.1180340e+00,  1.5000000e+00]], dtype=float32)

    """

    def __init__(self, pca_transform: PCA | None = None):
        self.pca_transform = pca_transform

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required=True
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output= True

        tags.input_tags.categorical = True
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan= True
        tags.input_tags.string = True

        tags.estimator_type = "transformer"
        return tags

    def fit(self,X:npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Calculate the encoding.
        

        :param X: 1D array of categorical data. This is contains the data to be encoded.
        :param y: 1D array of categorical data. The encoding is based on this data.
        """

        self.n_features_in_ = 1

        X_val,y_val = validate_data(self,X=X,y=y, validate_separately = (
            dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan",ensure_min_samples=0)
            ,dict(ensure_2d=False,dtype=["str","object"],ensure_all_finite="allow-nan",ensure_min_samples=0)
            ))

        if isinstance(X,pd.Series):
            self.feature_names_in_ = [X.name]
        if X_val.ndim != 1:
            raise ValueError("X should be 1D")
        if y_val.ndim != 1:
            raise ValueError("Y should be 1D")
        
        if X_val.shape[0] == 0 and y_val.shape[0]==0:
            self.mapping_ = {}
            self.n_features_out_ = 0

            return self

        # the core of this implementation


        #The alternative to using pandas here is either use scipy or DIY.
        missing_contingency_table = pd.crosstab(X_val,pd.isna(y_val))
        if not False in missing_contingency_table:
            self.mapping_ = {k: [np.nan] for k in missing_contingency_table.index}
            self.n_features_out_ = 1
            return self  
        x_such_that_y_is_not_always_missing = missing_contingency_table[missing_contingency_table[False]!=0].index
        
        contingency_table = pd.crosstab(X_val,y_val,dropna=False).loc[x_such_that_y_is_not_always_missing]#the result of pd.crosstab is a pandas dataframe

        x_such_that_y_is_always_missing = missing_contingency_table[missing_contingency_table[False]==0].index
       
        if contingency_table.shape[0] == 1:
            self.mapping_ = {contingency_table.index[0]: np.zeros(1)}|{k:[np.nan] for k in x_such_that_y_is_always_missing}
            self.n_features_out_ = 1
            return self
            
        pca_input = scale(contingency_table,axis=0)

        if self.pca_transform is None:
            self.pca_transform_ = PCA()
        else:
            self.pca_transform_ = clone(self.pca_transform)# for compatibility with sklearn we need to do this.

        pca_result = self.pca_transform_.fit_transform(
            X=pca_input
            ,y=None)

        #sklearn.decomposition.PCA implements the set_output api
        # that means that the user might configure globally that all transformers return pandas dataframes.
        if isinstance(pca_result,pd.DataFrame):
            pca_result = pca_result.to_numpy()

        self.n_features_out_ = pca_result.shape[1] #needed for get_feature_names_out
    
        mapping_for_missing = {k:[np.nan]*self.n_features_out_ for k in x_such_that_y_is_always_missing}#The values of X s.t. y is always missing

        value_mapping = {contingency_table.index[i]: pca_result[i] for i in range(contingency_table.shape[0])}

        self.mapping_ = value_mapping | mapping_for_missing

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        replaces each level of `X` with the numerical values determined in :py:meth:`fit`

        :param X: the feature to be encoded.
        """

        check_is_fitted(self)
        # if pd.isna(X).all():
        #     return np.zeros(X.shape[0])
        missing_mapping =  {None:[np.nan]*self.n_features_out_, pd.NA:[np.nan]*self.n_features_out_, np.nan:[np.nan]*self.n_features_out_}
        mapping_including_missing = self.mapping_|missing_mapping if hasattr(self,"mapping_") else missing_mapping

        # if X contains Nones, then the dtype of X is object.
        # In that case, X cannot be sorted.
        # many routines of numpy for finding differences between lists depend on the items being sortable.
        
        if hasattr(self,"mapping_"):
            unique_values_in_x =set(X[~pd.isna(X)])#np.unique([v for v in X if not pd.isna(v)])
            if not unique_values_in_x.issubset(self.mapping_.keys()):
                raise ValueError("new values not seen during fitting when encoding.")

        x_na = pd.isna(X)

        keys = np.where(x_na,None,X)
        mapping_to_np_array = {k: np.array(v,dtype=np.float32) for (k,v) in mapping_including_missing.items()}
        f = np.frompyfunc(mapping_to_np_array.get,nin=1,nout=1)
        return np.array(np.asanyarray(f(keys)).tolist())


    def get_feature_names_out(self,input_features=None):
        if not hasattr(self,"feature_names_in_"):
            if input_features is  None:
                return [f"x{i}" for i in range(self.n_features_out_)]
            else:
                return [f"{input_features[0]}_pca{i}" for i in range(self.n_features_out_)]

        if input_features is None:
            return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
        if input_features != self.feature_names_in_:
            raise ValueError(f"input_features is not feature_names_in_. Expected: {self.feature_names_in_}, actual: {input_features}")
        return [self.feature_names_in_[0]+f"_pca{i}" for i in range(self.n_features_out_)]
    
   
class MeanEncoder(OneToOneFeatureMixin,TransformerMixin, BaseEstimator): 
    """
    Transforms categorical data to numeric using mean encoding. The feature column `X` is encoded based on a numeric target column `y`.

    Examples
    --------
        >>> X = np.array(["a", "a", "b", "b", "c"])
        >>> y = np.array([1, 0, 2, 0, 3])
        >>>
        >>> encoder = MeanEncoder()
        >>> encoder.fit(X, y)
        >>> X_transformed = encoder.transform(X)
        >>> X_transformed
        array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
    """
    def __init__(self):
        pass

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = True
        tags.target_tags.single_output = True
        
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan = True
        
        tags.estimator_type = "transformer"
        return tags

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """
        Calculate average y value for each X category.
        
        :param X: Feature column.
        :param y: Target column.

        Examples
        --------
            >>> X = np.array(["a", "a", "b", "b", "c"])
            >>> y = np.array([1, 0, 2, 0, 3])
            >>>
            >>> encoder = MeanEncoder()
            >>> encoder.fit(X, y)
        """

        if not np.issubdtype(y.dtype,np.number):
            raise ValueError("target must be numeric dtype for mean encoding")

        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("mean encoding not possible for empty arrays")
        X_val =  to_missing_str_array(X)
        y_val = y

        self.n_features_in_ = 1


        # Identify missing
        X_missing = np.isnan(X_val)
        y_missing = np.isnan(y_val)
        
        # Fit encoder
        self.mapping_ = {}
        unique_categories = np.unique(X_val[~X_missing])
        for cat in unique_categories:
            mask = (~X_missing) & (X_val == cat)
            valid_targets = y_val[mask & ~y_missing]
            if valid_targets.size == 0:
                mean_val = np.nan
            else:
                mean_val = valid_targets.mean()
            
            self.mapping_[cat] = np.float32(mean_val)

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float32]:#float32 is optimal for decision trees.
        """
        Apply mapping from fitting function to ``X`` and returns the encoded version ``X_transformed``
        
        :param X: Original column to be encoded
        :return: Encoded column

        Examples
            >>> X = np.array(["a", "a", "b", "b", "c"])
            >>> y = np.array([1, 0, 2, 0, 3])
            >>>
            >>> encoder = MeanEncoder()
            >>> encoder.fit(X, y)
            >>> X_transformed = encoder.transform(X)
            >>> X_transformed
            array([0.5, 0.5, 1.,  1.,  3. ], dtype=float32)
        """

        check_is_fitted(self, 'mapping_')

        # Input validation
        #X = np.asarray(X)

        if X.ndim != 1:
            raise ValueError(f"X must be a 1D array, got shape {X.shape}.")
        if len(self.mapping_) == 0:
            return np.full(len(X), np.nan, dtype=np.float32).reshape(-1, 1) #2D output with only nans
        
        # Start transform
        result = np.full(len(X), np.nan, dtype=np.float32)

        X_missing = np.isnan(X)
        
        # Detect unseen categories
        unseen_categories = set(X[~X_missing]) - set(self.mapping_.keys())
        if unseen_categories:
            raise ValueError(f"Column to be encoded X has unseen categories: {unseen_categories}")
        
        # Apply mapping
        for i, val in enumerate(X):
            if not X_missing[i]:
                result[i] = self.mapping_[val]

        return result
