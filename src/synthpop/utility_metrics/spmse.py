"""
This module contains metrics to evaluate the utility of synthetic data.
"""
import pandas as pd
from itertools import combinations_with_replacement
import numpy as np

def preprocessing(df, max_bins=25):
    for colname in df:
        if pd.api.types.is_numeric_dtype(df[colname]):
            binned_column = pd.cut(df[colname],max_bins, labels=range(max_bins)) 
            df[colname] =binned_column
    return df

def joint_frequencies(df, name):
    """
    Create pairwise joint frequency tables for all variable pairs.
    """
    results = {}
    for col1, col2 in combinations_with_replacement(df.columns, 2):
        if col1 == col2:
            group = (df.groupby(col1, dropna=False).size().reset_index(name=name))
        else:
            group = (df.groupby([col1, col2], dropna=False).size().reset_index(name=name))

        results[(col1, col2)] = group
    return results

def Calc_S_pSME(jf_or: pd.DataFrame, jf_syn: pd.DataFrame, n_o: int, n_s: int, f_or: str,  f_syn: str):

    rescaled_differences = jf_syn[f_syn]-n_s/n_o*jf_or[f_or]
    expected_frequency = (jf_or[f_or]+jf_syn[f_syn])*(n_s/(n_o+n_s))
    k = (expected_frequency>0)
    if k.sum()>1:
        S_pMSE = 1/(k.sum()-1)*np.nansum(rescaled_differences[k]**2)/np.nansum(expected_frequency[k])
    else:
        raise ValueError("There is only one unique category pair in the data")
    return S_pMSE

def pairwise_spmse(orig_df: pd.DataFrame, syn_df: pd.DataFrame, max_bins: int = 25, na_label: str = "__NA__") -> pd.DataFrame:
    """
    Compute the Standardized propensity Mean Squared Error (S_pMSE) as defined in [1] for all pairs of variables 
    between two similarly-structured dataframes: one original dataset and one synthetic version of the original dataset,
    assuming the same variables and the same number of rows.
    
    :param orig_df: Original dataset.
    :type orig_df: pd.DataFrame
    :param syn_df: Synthetic dataset. Should have the same columns as the original dataset, in the same order and the same number of rows.
    :type syn_df: pd.DataFrame
    :param max_bins: Maximum number categories in which numeric variables can be discretized. Default value is 25.
    :type max_bins: int
    :param na_label: String value to be given to missing values. Default is __NA__
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
    if max_bins <= 1 or not isinstance(max_bins, int):
        raise ValueError("The number of bins should be a positive integer.")

    """Functional parameters"""
    f_origxy= 'f_or'
    f_synthxy= 'f_syn'
    #Wellicht kan dit ook als onderdeel van een class. Maar idk of het 'omgooien' nodig is. In dat geval kan je ook de tabellen e.d. opslaan als self. ...

    """Start calculations here"""
    binned_orig, binned_synth = preprocessing(orig_df.copy(),max_bins=max_bins), preprocessing(syn_df.copy(),max_bins=max_bins)
    #Hier ontbreekt nog een NaN handling

    joint_frequencies_orig = joint_frequencies(binned_orig, name=f_origxy)
    joint_frequencies_synth = joint_frequencies(binned_synth, name=f_synthxy)
    S_pMSEdict = {}
    for key in joint_frequencies_orig:
        S_pMSEdict[key] = Calc_S_pSME(joint_frequencies_orig[key], joint_frequencies_synth[key], n_o, n_s,  f_origxy, f_synthxy)

    S_pMSEdf = pd.DataFrame(S_pMSEdict, index=[0]) #make it a dataframe
    correct_form = S_pMSEdf.T.rename_axis(["column1", "column2"]).reset_index().rename(columns={0: 'S_pMSE'}) #make it a dataframe with 3 colums: col1, col2, S_pMSE

    return correct_form
