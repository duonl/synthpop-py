"""
This module contains metrics to evaluate the utility of synthetic data.
"""

from itertools import combinations_with_replacement
from typing import Sequence
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from synthpop.utils import str_dtype

__all__ = ["pairwise_spmse"]

def standardise_array_dtypes(X: pd.DataFrame)-> npt.NDArray:
    """
    Helper to standardise a 1D-array:
    - float64 for numeric data
    - `StringDType(na_object = np.nan)` for non-numeric data

    Missing values are normalised to `np.nan`.
    """

    is_numeric = pd.api.types.is_numeric_dtype(np.asanyarray(X))
    arr = np.asanyarray(X, dtype=object) #to avoid casting van np.nan to 'nan'

    if arr.ndim not in (1, 2):
        raise TypeError(f"Input must be a 1D or 2D array-like object, received {arr.ndim} instead.")

    original_shape = arr.shape
    flat = arr.reshape(-1)

    if is_numeric:
        result = np.array(
            [v if not pd.isna(v) else np.nan for v in flat],
            dtype=np.float64,
        )

    else:
        result = np.array(
            [v if not pd.isna(v) else np.nan for v in flat],
            dtype=str_dtype,
        )

    return result.reshape(original_shape)

def _preprocessing_numeric(
    column: pd.Series, bins: Sequence[float] | None = None
) -> pd.Series:
    """
    Preprocess a numeric column for S_pMSE computation by discretising it into left-closed bins.

    Numeric values are assigned to bin intervals using the provided bin edges.
    Missing values are preserved and are not modified.

    :param column: Numeric column to be discretised.
    :param bins: Bin edges used for discretisation. If `None`, no binning is applied.
    """

    if column.notna().any():
        binned_column = pd.cut(column, bins, right=False)
        column = binned_column
    return column

def _preprocess_columns(o_df, s_df, max_bins):
    """
    Preprocess the original and synthetic dataframes prior to S_pMSE calculation.

    Missing values are standardised to ``np.nan`` across all columns. 
    Numeric columns are discretised into bins

    :param o_df: Original dataframe
    :param s_df: Synthetic dataframe
    :param max_bins: Maximum number of bins to use when discretising numeric columns
    """

    for col in o_df.columns:
        o_col = pd.Series(standardise_array_dtypes(o_df[col]), index=o_df.index)
        s_col = pd.Series(standardise_array_dtypes(s_df[col]), index=s_df.index)

        o_is_numeric = pd.api.types.is_numeric_dtype(o_col)
        s_is_numeric = pd.api.types.is_numeric_dtype(s_col)

        if o_is_numeric != s_is_numeric:
            raise ValueError(
                f"Both synthetic and original column {col} must be "
                "either numeric or non-numeric"
            )

        if o_is_numeric:
            combined = pd.concat([o_col, s_col])
            print(combined.dtype)
            _, bins = pd.qcut(
                combined,
                max_bins,
                duplicates="drop",
                retbins=True,
            )

            bins[-1] = np.nextafter(bins[-1], np.inf) #Similar to synthpop R we define a left closed interval. This makes sure the 
            # highest value will also be in the final bin. This is directly implemented in R, but not in python.
            
            if len(bins) <4: # Equal to forming less than 3 groups in R: 
                __, bins = pd.cut(
                combined,
                max_bins,
                duplicates="drop",
                retbins=True,
                right=False,
            ) #Linear equal binning

            o_df[col] = _preprocessing_numeric(o_col, bins=bins)
            s_df[col] = _preprocessing_numeric(s_col, bins=bins)
    
    return o_df, s_df


def _joint_frequencies(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """
    Calculate the joint frequency tables for variable pairs.

    :param df: Dataset
    :param col1: column name 1
    :param col2: column name 2
    """

    if col1 == col2:
        jf = df.groupby(col1, dropna=False, observed=True).size().rename(col1)
    else:
        jf = (
            df.groupby([col1, col2], dropna=False, observed=True)
            .size()
            .rename(col1 + "," + col2)
        )

    return jf

def _calc_spmse(jf_or: pd.Series, jf_syn: pd.Series, n_o: int, n_s: int) -> np.float32:
    """
    Calculates the S_pSME for a combination of two columns from the joint frequency tables

    :param jf_or: Original dataset joint frequency table
    :param jf_syn: Synthetic dataset joint frequency table
    :param n_o: number of rows in original dataset
    :param n_s: number of rows in synthetic dataset
    """

    rescaled_differences = jf_syn.sub((n_s / n_o) * jf_or, fill_value=0)

    expected_frequency = jf_syn.add(jf_or, fill_value=0) * n_s / (n_o + n_s)

    print('rescaled', rescaled_differences)
    print('expected', expected_frequency)

    num_independent_combinations = len(expected_frequency.index)
    print(num_independent_combinations-1)
    if num_independent_combinations > 1:

        spmse = (
            1
            / (num_independent_combinations - 1)
            * np.sum(np.power(rescaled_differences, 2) / expected_frequency)
        ).astype(np.float32)
        print(np.sum(np.power(rescaled_differences, 2) / expected_frequency))
    else:  # number_independent_combinations=1

        spmse = np.float32(0.0)
        warnings.warn(
            f"Both variables are constant and equal; "
            f"the statistic is undefined. Return 0 for variable pair: {jf_syn.name}",
            UserWarning,
        )

    return spmse


def pairwise_spmse(
    orig_df: pd.DataFrame, syn_df: pd.DataFrame, max_bins: int = 25
) -> pd.DataFrame:
    """
    Compute the pairwise Standardised propensity Mean Squared Error (S_pMSE) between an original and a synthetic dataset.

    The metric compares the preservation of pairwise joint distributions between corresponding variables in both datasets.
    Numeric variables are discretised into at most `max_bins` bins using bin edges derived jointly from the original and synthetic dataset.
    Categorical and string variables are used directly.
    Missing values are standardised to np.nan and included in the frequency calculations as a separate level.

    :param orig_df: Original dataset.
    :param syn_df: Synthetic dataset. Must have the same column names as the original dataset, but the order does not matter.
    :param max_bins: Maximum number of categories in which numeric variables can be discretised. Missing values are discretised in bin number=max_bin+1. Default value is 25.
    :return: Dataset of variable pairs along with their corresponding S_pMSE value.

    Example:
    --------
    >>> import pandas as pd
    >>> from synthpop.utility_metrics.spmse import pairwise_spmse
    >>>
    >>> orig_data = pd.DataFrame({
    ...     "sex": ["M", "M", "F"],
    ...     "income": [50000, 50000, 60000],
    ... })
    >>>
    >>> syn_data = pd.DataFrame({
    ...     "sex": ["M", "F", "F"],
    ...     "income": [60000, 50000, 60000],
    ... })
    >>>
    >>> result = pairwise_spmse(orig_data, syn_data)
    >>> print(result)
    column1 column2    S_pMSE
    0     sex     sex  1.333333
    1     sex  income  2.666667
    2  income  income  1.333333
    """

    if not isinstance(orig_df, pd.DataFrame) or not isinstance(syn_df, pd.DataFrame):
        raise ValueError(
            "Original and synthetic dataframes should both be a pandas DataFrame."
        )

    n_o = len(orig_df.index)
    n_s = len(syn_df.index)

    if n_o == 0 or n_s == 0:
        raise ValueError("Both the original and synthetic dataframe must be non-empty.")

    if not orig_df.columns.is_unique or not syn_df.columns.is_unique:
        raise ValueError(
            "Original and synthetic dataframes must have unique column names."
        )

    if set(orig_df.columns) != set(syn_df.columns):
        raise ValueError(
            "Original and synthetic dataframes must have the same shape and column names."
        )

    if max_bins < 1 or not isinstance(max_bins, int):
        raise ValueError(
            "The number of bins should be an integer with value of at least 1."
        )

    # Make sure the original DataFrames are not modified
    o_df = orig_df.copy(deep=False)
    s_df = syn_df.copy(deep=False)

    # Start calculations for preprocessing here
    o_df, s_df = _preprocess_columns(o_df, s_df, max_bins)

    # Calculate joint frequency tables and s_pmse calculations below
    rows = []

    for col1, col2 in combinations_with_replacement(o_df.columns, 2):

        jf_orig = _joint_frequencies(o_df, col1, col2)
        jf_syn = _joint_frequencies(s_df, col1, col2)
        print('syn', jf_syn, col1, col2)
        print('orig', jf_orig, col1, col2)
        full_index = jf_orig.index.union(jf_syn.index, sort=False)
        jf_orig = jf_orig.reindex(full_index, fill_value=0)
        jf_syn = jf_syn.reindex(full_index, fill_value=0)

        rows.append([col1, col2, _calc_spmse(jf_orig, jf_syn, n_o, n_s)])

    spmse_df = pd.DataFrame(rows, columns=["column1", "column2", "S_pMSE"])
    return spmse_df
