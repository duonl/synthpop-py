"""
This module contains metrics to evaluate the utility of synthetic data.
"""
import pandas as pd
from itertools import combinations_with_replacement
import numpy as np
import warnings

def preprocessing(column: pd.Series, bins=None, na_label= 'N.a.N.'):
    """
    Preprocessing of the dataframes s.t. S_pMSE statistic can be calculated
    Checks if the na_label is already in use, and if not, fills nan values accordingly
    Bins numerical values in bins
    :param column: the specific column. 
    :type: pd.DataFrame Datatypes can be numeric, categorical, or string
    :param bins: array of bin edges
    :type: None (for non numerical columns) or a sequence (array/list)
    :param na_label: Label for NaN values
    :type: str
    """

    if (column == na_label).any():
        raise ValueError(f"column {column.dtype} contains N.a.N. This value should be reserved for handling missing values implemented by Synthpop.")
        
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.cat.add_categories([na_label])
        
    if pd.api.types.is_numeric_dtype(column):
        if column.notna().any():
            binned_column = pd.cut(column,bins) 
            column = binned_column
            column = column.cat.add_categories([na_label]).fillna(na_label)
        else: #Special case for when entire array is nan
            column = pd.Series(na_label, index=column.index, dtype=pd.CategoricalDtype)

    else:
        mask = pd.isna(column)
        column.loc[mask]= na_label
    
    return column

def joint_frequencies(df: pd.DataFrame, col1: str, col2: str):
    """
    Calculate joint frequency tables for variable pairs.
    :param df: Dataset
    :type: DataFrame
    :param col1: column name 1
    :type: str
    :param col1: column name 2
    :type: str
    """

    if col1 == col2:
        all_idx = df[col1].unique()
        jf = (df.groupby(col1, dropna=False).size().reindex(all_idx, fill_value=0))

    else:
        all_idx = pd.MultiIndex.from_product(
            [df[col1].unique(), df[col2].unique()],
            names=[col1, col2])
        jf = (df.groupby([col1, col2], dropna=False).size().reindex(all_idx, fill_value=0))
    
    return jf

def Calc_S_pSME(jf_or: pd.Series, jf_syn: pd.Series, n_o: int, n_s: int):
    """
    Calculates the S_pSME for a combination of two columns from the joint frequency tables
    :param jf_or: Original dataset joint frequency table
    :type: Series
    :param jf_syn: Synthetic dataset joint frequency table
    :type: Series
    :param n_o: number of rows in original dataset
    :type: int
    :param n_s: number of rows in synthetic dataset
    :type: int
    """

    rescaled_differences = jf_syn - (n_s / n_o) * jf_or
    expected_frequency = (jf_or + jf_syn) * n_s/ (n_o + n_s)

    k = (expected_frequency>0)
    if k.sum()>1:
        S_pMSE = 1/(k.sum()-1) * np.nansum(rescaled_differences[k]**2) / np.nansum(expected_frequency[k])

    else: #k=1 
        S_pMSE = 0.
        warnings.warn(f'both variables are constant and equal, so the statistic is undefined. Return 0 for variable pair: {jf_syn.index.name}')

    return S_pMSE

def pairwise_spmse(orig_df: pd.DataFrame, syn_df: pd.DataFrame, max_bins: int = 25, na_label: str = "N.a.N.") -> pd.DataFrame:
    """
    Compute the Standardized propensity Mean Squared Error (S_pMSE) as defined in [1] for all pairs of variables 
    between two similarly-structured dataframes: one original dataset and one synthetic version of the original dataset,
    assuming the same variables and the same number of rows.
    
    :param orig_df: Original dataset.
    :type orig_df: pd.DataFrame
    :param syn_df: Synthetic dataset. Should have the same columns as the original dataset, in the same order and the same number of rows.
    :type syn_df: pd.DataFrame
    :param max_bins: Maximum number categories in which numeric variables can be discretized. Missing values are discretized in bin number=max_bin+1. Default value is 25.
    :type max_bins: int
    :param na_label: String value to be given to missing values. Default is N.a.N.
    :type na_label: str
    :return: Dataset of variable pairs along with their corresponding S_pMSE value.
    :rtype: DataFrame
    """
    
    if not isinstance(orig_df, pd.DataFrame) or not isinstance(syn_df, pd.DataFrame):
        raise ValueError("Original and synthetic dataframes should both be a pandas dataframe")
    
    n_o = len(orig_df.index)
    n_s = len(syn_df.index)

    if n_o == 0 or n_s ==0:
        raise ValueError('Both the original and synthetic dataframe should consist out of non-zero rows')
    
    if len(orig_df.columns) != len(syn_df.columns) or not all(orig_df.columns==syn_df.columns):
        raise ValueError("Original and synthetic dataframes must have the same shape and column names.")
    
    if max_bins < 1 or not isinstance(max_bins, int):
        raise ValueError("The number of bins should be an integer with value larger than 1.")

    #Start calculations for preprocessing here
    for column_name in orig_df:

        if pd.api.types.is_numeric_dtype(orig_df[column_name]):
            combined = pd.concat([orig_df[column_name], syn_df[column_name]])
            _, bins = pd.cut(combined, bins=max_bins, retbins=True, duplicates='drop')

        else:
            bins=None

        orig_df[column_name] = preprocessing(orig_df[column_name],bins=bins, na_label=na_label)
        syn_df[column_name] = preprocessing(syn_df[column_name],bins=bins, na_label=na_label)

    #Calculate joint frequency tables and S_pMSE calculations here
    rows= []

    for col1, col2 in combinations_with_replacement(orig_df.columns, 2):

        jf_orig = joint_frequencies(orig_df,col1,col2)
        jf_syn = joint_frequencies(syn_df,col1,col2)

        rows.append([col1, col2, Calc_S_pSME(jf_orig, jf_syn, n_o, n_s)])

    S_pMSEdf = pd.DataFrame(rows, columns=["column1", "column2", "S_pMSE"])
    return S_pMSEdf
