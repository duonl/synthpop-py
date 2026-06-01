from typing import Any, Literal

import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse

@pytest.mark.parametrize(
    "orig_df, syn_df, max_bins, error",
    [
        (pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[1,2,3],[4,5,6]]), 25, "must have the same shape and column names."), #Check for unequal number of columns
        (pd.DataFrame({"A": [10], "B": [20]}), pd.DataFrame({"A": [10], "C": [20]}), 25 , "must have the same shape and column names."), #Check column names not equal
        ([],[], 12, "both be a pandas dataframe"), #Check for non pandas dataframes
        (pd.DataFrame([0]), pd.DataFrame([0]), 25., "The number of bins should be a positive integer."), #Check if max_bins is not an integer
        (pd.DataFrame([0]), pd.DataFrame([0]), -12, "The number of bins should be a positive integer."), #Check for negative bins
        (pd.DataFrame(), pd.DataFrame(), 35, "dataframe should consist out of non-zero rows"), #Check empty DataFrames
        (pd.DataFrame(['a', 'N.a.N.'], dtype=str), (pd.DataFrame(['a', 'N.a.N.'], dtype=str)), 25, "This value should be reserved for handling missing values") #Check if N.a.N. is already in use
    ] 
)
def test_pairwise_spmse_inputtests(orig_df, syn_df, max_bins, error):
    with pytest.raises(ValueError, match=error):
        pairwise_spmse(orig_df, syn_df, max_bins)

@pytest.mark.parametrize(
    "orig_df, syn_df, expected, na_label, max_bins",
    [
        (pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"column1": ["c1", "c1", "c2"], "column2": ["c1", "c2", "c2"], "S_pMSE": [0.0, 0.0, 0.0]}), 'N.a.N.', 25),#Desired output format.
        #Results should all be zero as original dataset=synthetic
        
        (pd.DataFrame({"c1": ["a", "a", "b"],"c2": [0, 0, 1]}),
        pd.DataFrame({"c1": ["a", "b", "b"],"c2": [1, 0, 1]}),
        pd.DataFrame({"column1": ["c1", "c1", "c2"],"column2": ["c1", "c2", "c2"],"S_pMSE": [2/3, 2/3, 2/3]}),"N.a.N.", 1000), #high number of bins
        #A non-zero answer, calculated by hand
        
        (pd.DataFrame({"c1": ["a", "a", "b"],"c2": [0, 0, 1]}),
        pd.DataFrame({"c1": ["a", "b", "b"],"c2": [1, 0, 1]}),
        pd.DataFrame({"column1": ["c1", "c1", "c2"],"column2": ["c1", "c2", "c2"],"S_pMSE": [2/3, 2/3, 2/3]}),"N.a.N.", 2), 
        #Check for two bins

        #Also make a check for a single bin
        #test with categorical dtype
        (pd.DataFrame({"c1": [0, 0, 0, 1]}), 
        pd.DataFrame({"c1": [0, 1]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25),
        #A one-dimensional input with different number of rows

        (pd.DataFrame({"c1": [0, 0, 0, 1]}, dtype='category'), 
        pd.DataFrame({"c1": [0, 1]}, dtype='category'),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25),
        #A one-dimensional input with different number of rows using datatype category

        (pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"column1": ["c1", "c1", "c2"], "column2": ["c1", "c2", "c2"], "S_pMSE": [0.0, 0.0, 0.0]}), 'N.a.N.', 25),#Desired output format and should be zero as both are the same.

        (pd.DataFrame({"c1": ['nan', 'nan', 'nan', np.nan]}), 
        pd.DataFrame({"c1": ['nan', np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25), #Check missing value handling, np.nan+strings that spell nan (str DataFrame)

        (pd.DataFrame({"c1": [np.nan, np.nan, np.nan, 'nan']}), 
        pd.DataFrame({"c1": [np.nan, 'nan']}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25), #Check missing value handling, Multiple occurences

        (pd.DataFrame({"c1": [0, 0, 0, np.nan]}), 
        pd.DataFrame({"c1": [0, np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25), #Check missing value handling, np.nan+integers (float DataFrame)

        (pd.DataFrame({"c1": ['a', 'a', 'a', pd.NA]}), 
        pd.DataFrame({"c1": ['a', pd.NA]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25), #Check missing value handling, (str + pd.NA)

        (pd.DataFrame({"c1": ['a', 'a', 'a', 'Different N.a.N.']}), 
        pd.DataFrame({"c1": ['a', 'Different N.a.N.']}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [1/4]}), "N.a.N.", 25), #Check missing value handling, and whether it takes N.a.N. as values
        #This should also not raise an error because 'Different N.a.N.' is not exactly equal to N.a.N.

        (pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}), 
        pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [0.]}), "N.a.N.", 25),
        #Full nan bin

        (pd.DataFrame({"c1": [0.,0.,0.,0.]}), 
        pd.DataFrame({"c1": [0.,0.,0.,0.]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [0.]}), "N.a.N.", 25)
    ] #Unbinnable data
)
def test_pairwise_spmse_output(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame, na_label : str, max_bins : int):

    output = pairwise_spmse(orig_df, syn_df, na_label=na_label, max_bins=max_bins)

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
