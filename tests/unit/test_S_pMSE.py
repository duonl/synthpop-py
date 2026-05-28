from typing import Any, Literal

import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse, preprocessing, joint_frequencies, Calc_S_pSME

@pytest.mark.parametrize(
    "orig_df, syn_df, max_bins",
    [
        (pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[1,2,3],[4,5,6]]), 25), #Check for unequal number of columns
        (pd.DataFrame({"A": [10], "B": [20]}), pd.DataFrame({"A": [10], "C": [20]}), 25), #Check column names not equal
        ([],[], 12), #Check for non pandas dataframes
        (pd.DataFrame(), pd.DataFrame(), 25.), #Check if max_bins is not an integer
        (pd.DataFrame(), pd.DataFrame(), -12), #Check for negative bins
        (pd.DataFrame(), pd.DataFrame(), 35), #Check empty DataFrames
        (pd.DataFrame(['a', 'N.a.N.'], dtype=str), (pd.DataFrame(['a', 'N.a.N.'], dtype=str)), 25) #Check if N.a.N. is already in use
    ] 
)
def test_pairwise_spmse_inputtests(orig_df, syn_df, max_bins):
    with pytest.raises(ValueError):
        pairwise_spmse(orig_df, syn_df, max_bins)

@pytest.mark.parametrize(
    "orig_df, syn_df, expected, na_label",
    [
        (pd.DataFrame([[1,2],[3,4]], columns=['c1', 'c2']), 
        pd.DataFrame([[1,2],[3,4]], columns=['c1', 'c2']), 
        pd.DataFrame([['c1', 'c1', 0.],['c1', 'c2', 0.],['c2','c2',0.]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'),#Desired output format.
        #Results should all be zero as original dataset=synthetic

        (pd.DataFrame([['a', 0], ['a', 0], ['b',1]], columns=['c1', 'c2']), 
        pd.DataFrame([['a', 1], ['b', 0], ['b',1]], columns=['c1', 'c2']), 
        pd.DataFrame([['c1', 'c1', 2/3],['c1', 'c2', 2/3],['c2','c2',2/3]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'),
        #A non-zero answer, calculated by hand

        (pd.DataFrame([[0], [0], [0],[1]], columns=['c1']), 
        pd.DataFrame([[0], [1]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'),
        #A one-dimensional input with different number of rows

        (pd.DataFrame([['nan'], ['nan'], ['nan'],[np.nan]], columns=['c1']), 
        pd.DataFrame([['nan'], [np.nan]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'), #Check missing value handling

        (pd.DataFrame([[0], [0], [0],[np.nan]], columns=['c1']), 
        pd.DataFrame([[0], [np.nan]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'), #Check missing value handling

        (pd.DataFrame([['a'], ['a'], ['a'],[np.nan]], columns=['c1']), 
        pd.DataFrame([['a'], [np.nan]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'), #Check missing value handling

        (pd.DataFrame([['a'], ['a'], ['a'],[pd.NA]], columns=['c1']), 
        pd.DataFrame([['a'], [pd.NA]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.'), #Check missing value handling

        (pd.DataFrame([['a'], ['a'], ['a'],['Different N.a.N.']], columns=['c1']), 
        pd.DataFrame([['a'], ['Different N.a.N.']], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 1/4]], columns=['column1', 'column2', 'S_pMSE']), 'N.a.N.') #Check missing value handling
        #This should also work because Different N.a.N. is not exactly equal to N.a.N.
    ] 
)
def test_pairwise_spmse_output(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame, na_label : str):

    output = pairwise_spmse(orig_df, syn_df, na_label=na_label)

    assert output.equals(expected)


@pytest.mark.parametrize(
    "orig_df, syn_df, expected",
    [
        (pd.DataFrame([[0], [0]], columns=['c1']), 
        pd.DataFrame([[0], [0], [0],[0]], columns=['c1']), 
        pd.DataFrame([['c1', 'c1', 0.]], columns=['column1', 'column2', 'S_pMSE']))
        #If both variables are constant and equal, k=1 so the statistic is undefined due to division by zero. 
        #The function sends a warning and returns 0 for the variable pair.
    ]
)
def test_warning(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame):

    with pytest.warns(UserWarning) as record:
        output = pairwise_spmse(orig_df, syn_df)

    assert output.equals(expected)
    assert "c1" in str(record[0].message)
