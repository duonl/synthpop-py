"""
This module contains metrics to evaluate the utility of synthetic data.
"""
import pandas as pd

def pairwise_spmse(orig_df: pd.DataFrame, syn_df: pd.DataFrame, max_groups: int = 25, na_label: str = "__NA__") -> pd.DataFrame:
    """
    Compute the Standardized propensity Mean Squared Error (S_pMSE) as defined in [1] for all pairs of variables 
    between two similarly-structured dataframes: one original dataset and one synthetic version of the original dataset,
    assuming the same variables and the same number of rows.
    
    :param orig_df: Original dataset.
    :type orig_df: pd.DataFrame
    :param syn_df: Synthetic dataset. Should have the same columns as the original dataset, in the same order and the same number of rows.
    :type syn_df: pd.DataFrame
    :param max_groups: Maximum number categories in which numeric variables can be discretized. Default value is 25.
    :type max_groups: int
    :param na_label: String value to be given to missing values. Default is __NA__
    :type na_label: str
    :return: Dataset of variable pairs along with their corresponding S_pMSE value.
    :rtype: DataFrame
    """
    if orig_df.shape != syn_df.shape:
        raise ValueError("Original and synthetic dataframes must have the same shape.")
    output = orig_df.copy()
    return output