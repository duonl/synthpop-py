from typing import Any, Literal

import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse

@pytest.mark.parametrize(
    "orig_df, syn_df, max_bins, error",
    [
        
        (pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[1,2,3],[4,5,6]]), 25, "must have the same shape and column names."), 
        #Check for unequal number of columns

        (pd.DataFrame({"A": [10], "B": [20]}), pd.DataFrame({"A": [10], "C": [20]}), 25 , "must have the same shape and column names."), 
        #Check column names not equal

        (pd.DataFrame({"B": [10], "A": [20]}), pd.DataFrame({"A": [10], "A": [20]}), 35, "must have the same shape and column names."), 
        #Check for multiple columns having the same name

        ([],[], 12, "both be a pandas dataframe"), 
        #Check for non pandas dataframes

        (pd.DataFrame([0]), pd.DataFrame([0]), 25., "with value of at least 1."), 
        #Check if max_bins is not an integer

        (pd.DataFrame([0]), pd.DataFrame([0]), -12, "with value of at least 1."), 
        #Check for negative bins

        (pd.DataFrame(), pd.DataFrame(), 35, "dataframe should consist out of non-zero rows"), 
        #Check empty DataFrames

    ] 
)
def test_pairwise_spmse_inputtests(orig_df, syn_df, max_bins, error):

    with pytest.raises(ValueError, match=error):
        pairwise_spmse(orig_df, syn_df, max_bins)

@pytest.mark.parametrize(
    "orig_df, syn_df, expected, max_bins",
    [

        (pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"c1": [1, 3],"c2": [2, 4]}), 
        pd.DataFrame({"column1": ["c1", "c1", "c2"], "column2": ["c1", "c2", "c2"], "S_pMSE": [0.0, 0.0, 0.0]}), 25),
        #S_pMSE should all be zero as the original_dataset=the synthetic_dataset
        
        (pd.DataFrame({"c1": ["a", "a", "b"],"c2": [0, 0, 1]}),
        pd.DataFrame({"c1": ["a", "b", "b"],"c2": [1, 0, 1]}),
        pd.DataFrame({"column1": ["c1", "c1", "c2"],"column2": ["c1", "c2", "c2"],"S_pMSE": [4/3, 8/3, 4/3]}), 1000), 
        #A non-zero answer to the S_pMSE, calculated by hand, with a high value of bins
        
        (pd.DataFrame({"c1": ["a", "a", "b"],"c2": [0, 0, 1]}),
        pd.DataFrame({"c2": [1,0,1],"c1": ["a", "b", "b"]}),
        pd.DataFrame({"column1": ["c1", "c1", "c2"],"column2": ["c1", "c2", "c2"],"S_pMSE": [4/3, 8/3, 4/3]}), 25), 
        #A non-zero answer to the S_pMSE, calculated by hand, with different column order

        (pd.DataFrame({"c1": [0,1,2]}),
        pd.DataFrame({"c1": [1,2]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [450/1944]}), 2), 
        #Check for two bins

        (pd.DataFrame({"c1": [0,1,2]}),
        pd.DataFrame({"c1": [1,2]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [50/72]}), 3), 
        #Check for three bins, same input as above. but number of bins will produce different output
        #This test will produce a floating point error if pd.DataFrame.equals() is used

        (pd.DataFrame({"c1": ['a', 'b', 'c']}),
        pd.DataFrame({"c1": ['b','c']}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [50/72]}), 25), 
        #Check statistics if not every value of the original dataset is represented in the synthetic dataset

        (pd.DataFrame({"c1": [0, 0, 0, 1]}), 
        pd.DataFrame({"c1": [0, 1]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25),
        #A one-dimensional input with different number of rows

        (pd.DataFrame({"c1": ['a', 'a', 'a', 'b']}, dtype='category'), 
        pd.DataFrame({"c1": ['a', 'b']}, dtype='category'),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25),
        #A one-dimensional input with different number of rows using datatype category

        (pd.DataFrame({"c1": ['nan', 'nan', 'nan', np.nan]}), 
        pd.DataFrame({"c1": ['nan', np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25), 
        #Check missing value handling, np.nan+strings that spell nan (str DataFrame)

        (pd.DataFrame({"c1": [np.nan, np.nan, np.nan, 'nan']}), 
        pd.DataFrame({"c1": [np.nan, 'nan']}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25), 
        #Check missing value handling, Multiple occurrences of nan

        (pd.DataFrame({"c1": [0, 0, 0, np.nan]}), 
        pd.DataFrame({"c1": [0, np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25), 
        #Check missing value handling, np.nan+integers (float DataFrame)

        (pd.DataFrame({"c1": ['a', 'a', 'a', pd.NA]}), 
        pd.DataFrame({"c1": ['a', pd.NA]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25), 
        #Check missing value handling, (str + pd.NA)

        (pd.DataFrame({"c1": ['a', 'a', 'a', 'Different N.a.N.']}), 
        pd.DataFrame({"c1": ['a', 'Different N.a.N.']}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [9/16]}), 25), 
        #Check missing value handling, and whether it takes N.a.N. as values
        #This should also not raise an error because 'Different N.a.N.' is not exactly equal to N.a.N.

        (pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}), 
        pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [0.]}), 25),
        #Full nan bin

        (pd.DataFrame({"c1": [pd.NA, pd.NA, pd.NA, pd.NA]}), 
        pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan, np.nan]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [0.]}), 25),
        #Full nan bin

        (pd.DataFrame({"c1": [0.,0.,0.,0.]}), 
        pd.DataFrame({"c1": [0.,0.,0.,0.]}),
        pd.DataFrame({"column1": ["c1"], "column2": ["c1"], "S_pMSE": [0.]}), 25),
        #Data where every value will fall into the same bin

        (pd.DataFrame({"c1": [1, 0, np.nan], "c2": ['a', pd.NA, 'c'], "c3": [6, 7, 3]}), 
        pd.DataFrame({ "c2": [pd.NA, pd.NA, pd.NA], "c3": [6, 3, 6], "c1": [np.nan, np.nan, 0]}),
        pd.DataFrame({"column1": ["c1", "c1", "c1",
                                "c2", "c2", "c3"], 

                        "column2": ["c1", "c2", "c3",
                                "c2", "c3", "c3"], 

                        "S_pMSE": [4/3, 8/3, 4/3, 
                                3., 20/9, 0.]}), 3)
        #A very extensive test, including multiple instances of nan, a full nan array, non-zero outputs
        #but also a zero output due to binning, and different order of columns

    ]
)
def test_pairwise_spmse_output(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame, max_bins : int):

    output = pairwise_spmse(orig_df, syn_df, max_bins=max_bins)
    pd.testing.assert_frame_equal(output, expected, check_exact=False, rtol=1e-9)
    #Test with index 3 will produce a floating point error, and hence assert, if pd.DataFrame.equals() is used.


def test_warning():

    orig_df = pd.DataFrame({'c1': [0, 0]})
    syn_df = pd.DataFrame({'c1': [0, 0, 0, 0]})
    expected = pd.DataFrame({"column1": ['c1'], "column2": ['c1'], "S_pMSE": [0.0]})

    with pytest.warns(UserWarning) as record:
        output = pairwise_spmse(orig_df, syn_df)

    assert output.equals(expected)
    assert "c1" in str(record[0].message)

def test_pairwise_spmse_does_not_mutate_inputs():
    orig = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "categorical": ["a", "b", "a", None],
        }
    )

    syn = pd.DataFrame(
        {
            "numeric": [1, 2, 4, 5],
            "categorical": ["a", "b", None, "c"],
        }
    )

    orig_before = orig.copy(deep=True)
    syn_before = syn.copy(deep=True)

    pairwise_spmse(orig, syn)

    pd.testing.assert_frame_equal(orig, orig_before)

    pd.testing.assert_frame_equal(syn, syn_before)