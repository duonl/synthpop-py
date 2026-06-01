"""
This module contains metrics to evaluate the utility of synthetic data.
"""
import pandas as pd
from itertools import combinations_with_replacement
import numpy as np
import warnings

def preprocessing(df, max_bins=25, na_label= 'N.a.N.'):
    for colname in df:
        if (df[colname] == na_label).any():
            raise ValueError(f"column {colname} contains N.a.N. This value should be reserved for handling missing values implemented by Synthpop")
        
        if isinstance(df[colname].dtype, pd.CategoricalDtype):
            df[colname] = df[colname].cat.add_categories([na_label])
        
        if pd.api.types.is_numeric_dtype(df[colname]):
            if df[colname].notna().any():
                binned_column = pd.cut(df[colname],max_bins, labels=range(max_bins)) 
                df[colname] =binned_column
                df[colname].cat.add_categories([na_label]).fillna(na_label)
            else: #Special case if entire array is NAN
                df[colname] = pd.Series(na_label, index=df[colname].index, dtype=pd.CategoricalDtype)
        else:
            mask = pd.isna(df[colname])
            df.loc[mask, colname] = na_label
    return df

def joint_frequencies(df):
    """
    Create pairwise joint frequency tables for all variable pairs.
    """
    results = {}
    for col1, col2 in combinations_with_replacement(df.columns, 2):
        if col1 == col2:
            all_idx = df[col1].unique() #Makes sure you also return combinations with count=0
            group = (df.groupby(col1, dropna=False).size().reindex(all_idx, fill_value=0))
        else:
            all_idx = pd.MultiIndex.from_product(
            [df[col1].unique(), df[col2].unique()],
            names=[col1, col2]
            )
            group = (df.groupby([col1, col2], dropna=False).size().reindex(all_idx, fill_value=0))
        results[(col1, col2)] = group
    return results

def Calc_S_pSME(jf_or: pd.DataFrame, jf_syn: pd.DataFrame, n_o: int, n_s: int):
    rescaled_differences = jf_syn-n_s/n_o*jf_or
    expected_frequency = (jf_or+jf_syn)*(n_s/(n_o+n_s))
    k = (expected_frequency>0)
    if k.sum()>1:
        S_pMSE = 1/(k.sum()-1)*np.nansum(rescaled_differences[k]**2)/np.nansum(expected_frequency[k])
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
        raise ValueError("The number of bins should be a positive integer.")

    """Start calculations here"""
    binned_orig = preprocessing(orig_df.copy(),max_bins=max_bins, na_label=na_label)
    binned_synth = preprocessing(syn_df.copy(),max_bins=max_bins, na_label=na_label)

    joint_frequencies_orig = joint_frequencies(binned_orig)
    joint_frequencies_synth = joint_frequencies(binned_synth)
    S_pMSEdict = {}
    for key in joint_frequencies_orig:
        S_pMSEdict[key] = Calc_S_pSME(joint_frequencies_orig[key], joint_frequencies_synth[key], n_o, n_s)
    
    S_pMSEdf = pd.DataFrame(S_pMSEdict, index=[0]) #make it a dataframe
    correct_form = S_pMSEdf.T.rename_axis(["column1", "column2"]).reset_index().rename(columns={0: 'S_pMSE'}) #make it a dataframe with 3 colums: col1, col2, S_pMSE
    return correct_form
