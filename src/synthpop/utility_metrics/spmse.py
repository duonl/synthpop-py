"""
This module contains metrics to evaluate the utility of synthetic data.
"""
import pandas as pd
from itertools import combinations_with_replacement
import numpy as np
import warnings

__all__ = ["pairwise_spmse"]

def _preprocessing(column: pd.Series, bins: Sequence[float] | None =None):
    """
    Preprocessing of the dataframes s.t. S_pMSE statistic can be calculated
    Bins numerical values in bins
    :param column: the specific column. 
    :type: pd.DataFrame Datatypes can be numeric, categorical, or string
    :param bins: array of bin edges
    :type: None (for non numerical columns) or a sequence (array/list)
    """

    if pd.api.types.is_numeric_dtype(column):

        if column.notna().any():
            binned_column = pd.cut(column,bins)
            column = binned_column

    else:
        mask = pd.isna(column)
        column.loc[mask]= np.nan
    
    return column

def _make_joint_frequencies_indices(all_levels, col1: str, col2: str):
    """
    Makes all the required indices/levels of possible combinations (x,y), for x as element of col1 and y as element of col2
    Takes both the synthetic as the original dataset into account, as specific combinations might be present in one or the other
    :param all_levels: Dictionary of the union of all combinations of (x,y)
    :type: dict
    :param col1: column name 1
    :type: str
    :param col2: column name 2
    :type: str
    """

    if col1 == col2:
        all_idx = all_levels[col1]
    
    else:
        idx1 = all_levels[col1]
        idx2 = all_levels[col2]

        all_idx = pd.MultiIndex.from_product([idx1, idx2],names=[col1, col2])  

    return all_idx

def _joint_frequencies(df: pd.DataFrame, col1: str, col2: str, full_idx: pd.MultiIndex | list):
    """
    Calculate joint frequency tables for variable pairs.
    :param df: Dataset
    :type: DataFrame
    :param col1: column name 1
    :type: str
    :param col2: column name 2
    :type: str
    :param full_idx: list of indices for both synthetic and original frame
    :type: Either pd.MultiIndex or list of names
    """

    if col1 == col2:
        jf = (df.groupby(col1, dropna=False).size().reindex(full_idx, fill_value=0).rename(col1))
        #jf = df[col1].value_counts(dropna=False)#.reindex(full_idx, fill_value=0).rename(col1)
    else:
        jf = (df.groupby([col1, col2], dropna=False).size().reindex(full_idx, fill_value=0).rename(col1+','+col2))
        #jf = df[[col1,col2]].value_counts(dropna=False)#.reindex(full_idx, fill_value=0).rename(col1+','+col2)
    return jf

def _calc_spmse(jf_or: pd.Series, jf_syn: pd.Series, n_o: int, n_s: int):
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

    #rescaled_differences = jf_syn - (n_s / n_o) * jf_or
    #expected_frequency = (jf_or + jf_syn) * n_s/ (n_o + n_s)

    rescaled_differences = jf_syn.sub((n_s / n_o) * jf_or, fill_value=0)
    expected_frequency = jf_syn.add(jf_or, fill_value=0) * n_s/ (n_o + n_s)

    k = (expected_frequency>0)
    if k.sum()>1:
        S_pMSE = 1/(k.sum()-1) * np.nansum(rescaled_differences[k]**2 / expected_frequency[k])

    else: #k=1 
        S_pMSE = 0.
        warnings.warn(f'both variables are constant and equal, so the statistic is undefined. Return 0 for variable pair: {jf_syn.name}')

    return S_pMSE

def pairwise_spmse(orig_df: pd.DataFrame, syn_df: pd.DataFrame, max_bins: int = 25) -> pd.DataFrame:
    """
    Compute the Standardised propensity Mean Squared Error (S_pMSE) as defined in [1] for all pairs of variables 
    between two similarly-structured dataframes: one original dataset and one synthetic version of the original dataset,
    assuming the same variables and the same number of rows.
    
    :param orig_df: Original dataset.
    :type orig_df: pd.DataFrame
    :param syn_df: Synthetic dataset. Should have the same columns as the original dataset, in the same order and the same number of rows.
    :type syn_df: pd.DataFrame
    :param max_bins: Maximum number categories in which numeric variables can be discretized. Missing values are discretized in bin number=max_bin+1. Default value is 25.
    :type max_bins: int
    :return: Dataset of variable pairs along with their corresponding S_pMSE value.
    :rtype: DataFrame
    """
    
    if not isinstance(orig_df, pd.DataFrame) or not isinstance(syn_df, pd.DataFrame):
        raise ValueError("Original and synthetic dataframes should both be a pandas dataframe")
    
    n_o = len(orig_df.index)
    n_s = len(syn_df.index)

    if n_o == 0 or n_s ==0:
        raise ValueError('Both the original and synthetic dataframe should consist out of non-zero rows')
    
    if len(orig_df.columns) != len(syn_df.columns) or set(orig_df.columns) != set(syn_df.columns):
        raise ValueError("Original and synthetic dataframes must have the same shape and column names.")
    
    if max_bins < 1 or not isinstance(max_bins, int):
        raise ValueError("The number of bins should be an integer with value of at least 1.")

    o_df = orig_df.copy(deep=True) #Make sure the original DataFrames are not modified 
    s_df = syn_df.copy(deep=True)

    #Start calculations for preprocessing here
    all_levels = {}
    for column_name in orig_df:

        if pd.api.types.is_numeric_dtype(o_df[column_name]):
            combined = pd.concat([o_df[column_name], s_df[column_name]])
            _, bins = pd.cut(combined, bins=max_bins, retbins=True, duplicates='drop')

        else:
            bins=None

        o_df[column_name] = _preprocessing(o_df[column_name],bins=bins)
        s_df[column_name] = _preprocessing(s_df[column_name],bins=bins)

        #Pre compute the union of column level names for joint frequency tables
        all_levels[column_name] = (
            pd.Index(o_df[column_name].unique())
            .union(pd.Index(s_df[column_name].unique()))
        )


    #Calculate joint frequency tables and s_pmse calculations below
    rows= []
    
    for col1, col2 in combinations_with_replacement(o_df.columns, 2):
        
        full_idx = _make_joint_frequencies_indices(all_levels, col1, col2)

        jf_orig = _joint_frequencies(o_df,col1,col2, full_idx)
        jf_syn = _joint_frequencies(s_df,col1,col2, full_idx)
        rows.append([col1, col2, _calc_spmse(jf_orig, jf_syn, n_o, n_s)])

    S_pMSEdf = pd.DataFrame(rows, columns=["column1", "column2", "S_pMSE"])
    return S_pMSEdf
