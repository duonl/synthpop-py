import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse


@pytest.mark.parametrize(
    "orig_df, syn_df, max_bins, error",
    [

        (
            pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]]), 25, "must have the same shape and column names."
        ),
        # Check for unequal number of columns

        (
            pd.DataFrame({"A": [10], "B": [20]}), pd.DataFrame(
                {"A": [10], "C": [20]}), 25, "must have the same shape and column names."
        ),
        # Check column names not equal

        (
            pd.DataFrame([[10, 20]], columns=["A", "A"]), pd.DataFrame(
                [[10, 20]], columns=["A", "A"]), 35, "must have unique column names."
        ),
        # Check for multiple columns having the same name

        (
            [], [], 12, "both be a pandas DataFrame"
        ),
        # Check for non pandas dataframes

        (
            pd.DataFrame([0]), pd.DataFrame([0]),
            25., "with value of at least 1."
        ),
        # Check if max_bins is not an integer

        (
            pd.DataFrame([0]), pd.DataFrame([0]), -
            12, "with value of at least 1."
        ),
        # Check for negative bins

        (
            pd.DataFrame(), pd.DataFrame(), 35,
            "dataframe must be non-empty"
        ),
        # Check empty DataFrames

    ]
)
def test_pairwise_spmse_raises_wrong_inputs(orig_df, syn_df, max_bins, error):

    with pytest.raises(ValueError, match=error):
        pairwise_spmse(orig_df, syn_df, max_bins)


@pytest.mark.parametrize(
    "orig_df, syn_df, expected",
    [

        (
            pd.DataFrame({"c1": [1, 3], "c2": [2, 4]}),
            pd.DataFrame({"c1": [1, 3], "c2": [2, 4]}),
            pd.DataFrame(
                {
                    "column1": ["c1", "c1", "c2"],
                    "column2": ["c1", "c2", "c2"],
                    "S_pMSE": [0.0, 0.0, 0.0]
                }
            ),
        ),
        # S_pMSE should all be zero as the original_dataset = the synthetic_dataset

        (
            pd.DataFrame(
                {"sex": ["M", "M", "F"], "income": [50000, 50000, 60000]}),
            pd.DataFrame(
                {"sex": ["M", "F", "F"], "income": [60000, 50000, 60000]}),
            pd.DataFrame(
                {
                    "column1": ["sex", "sex", "income"],
                    "column2": ["sex", "income", "income"],
                    "S_pMSE": [4/3, 8/3, 4/3]
                }
            ),
        ),
        # A non-zero answer to the S_pMSE, calculated by hand. Also this is the example in the docstrings

        (
            pd.DataFrame({"c1": ["a", "a", "b"], "c2": [0, 0, 1]}),
            pd.DataFrame({"c2": [1, 0, 1], "c1": ["a", "b", "b"]}),
            pd.DataFrame(
                {
                    "column1": ["c1", "c1", "c2"],
                    "column2": ["c1", "c2", "c2"],
                    "S_pMSE": [4/3, 8/3, 4/3]
                }
            ),
        ),
        # A non-zero answer to the S_pMSE, calculated by hand, with different column order

        (
            pd.DataFrame({"c1": ['a', 'b', 'c']}),
            pd.DataFrame({"c1": ['b', 'c']}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [50/72]
                }
            ),
        ),
        # Check spmse if not every value of the original dataset is represented in the synthetic dataset

        (
            pd.DataFrame({"c1": [0, 0, 0, 1]}),
            pd.DataFrame({"c1": [0, 1]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # A one-dimensional input with different number of rows

        (
            pd.DataFrame({"c1": ['a', 'a', 'a', 'b']}, dtype='category'),
            pd.DataFrame({"c1": ['a', 'b']}, dtype='category'),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # A one-dimensional input with different number of rows using datatype category

        (
            pd.DataFrame({"c1": [0., 0., 0., 0.]}),
            pd.DataFrame({"c1": [0., 0., 0., 0.]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [0.]
                }
            ),
        ),
        # Data where every value will fall into the same bin
    ]
)
def test_pairwise_spmse_input_shapes_and_types(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame):

    output = pairwise_spmse(orig_df, syn_df)
    expected['S_pMSE'] = expected['S_pMSE'].astype(
        np.float32)  # Output should be float32

    pd.testing.assert_frame_equal(
        output, expected, check_exact=False, rtol=1e-9)
    # Test with index 3 will produce a floating point error, and hence assert, if pd.DataFrame.equals() is used.


@pytest.mark.parametrize(
    "orig_df, syn_df, expected, max_bins",
    [

        (
            pd.DataFrame({"c1": [0, 1, 2]}),
            pd.DataFrame({"c1": [1, 2]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [450/1944]
                }
            ),
            2,
        ),
        # Check for two bins

        (
            pd.DataFrame({"c1": [0, 1, 2]}),
            pd.DataFrame({"c1": [1, 2]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [50/72]
                }
            ),
            3,
        ),
        # Check for three bins, same input as above. but number of bins will produce different output
        # This test will produce a floating point error if pd.DataFrame.equals() is used

        (
            pd.DataFrame({"c1": [0, 1, 2]}),
            pd.DataFrame({"c1": [1, 2]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [50/72]
                }
            ),
            1000,
        ),
        # Check for high number of bins
    ]
)
def test_pairwise_spmse_binsizes(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame, max_bins: int):

    output = pairwise_spmse(orig_df, syn_df, max_bins=max_bins)
    expected['S_pMSE'] = expected['S_pMSE'].astype(
        np.float32)  # Output should be float32

    pd.testing.assert_frame_equal(
        output, expected, check_exact=False, rtol=1e-9)


@pytest.mark.parametrize(
    "orig_df, syn_df, expected",
    [

        (
            pd.DataFrame({"c1": ['nan', 'nan', 'nan', np.nan]}),
            pd.DataFrame({"c1": ['nan', np.nan]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # Check missing value handling, np.nan+strings that spell nan (str DataFrame)

        (
            pd.DataFrame({"c1": [np.nan, np.nan, np.nan, 'nan']}),
            pd.DataFrame({"c1": [np.nan, 'nan']}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # Check missing value handling, Multiple occurrences of nan

        (
            pd.DataFrame({"c1": [0, 0, 0, np.nan]}),
            pd.DataFrame({"c1": [0, np.nan]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # Check missing value handling, np.nan+integers (float DataFrame)

        (
            pd.DataFrame({"c1": ['a', 'a', 'a', pd.NA]}),
            pd.DataFrame({"c1": ['a', pd.NA]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [9/16]
                }
            ),
        ),
        # Check missing value handling, (str + pd.NA)

        (
            pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}),
            pd.DataFrame({"c1": [np.nan, np.nan, np.nan, np.nan]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [0.]
                }
            ),
        ),
        # Full nan bin

        (
            pd.DataFrame({"c1": [pd.NA, pd.NA, pd.NA, pd.NA]}),
            pd.DataFrame({"c1": [None, None, None, None, None]}),
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [0.]
                }
            ),
        ),
    ]  # Full nan bin, checks pd.NA and None
)
def test_pairwise_spmse_missing_value_handling(orig_df: pd.DataFrame, syn_df: pd.DataFrame, expected: pd.DataFrame):

    output = pairwise_spmse(orig_df, syn_df, max_bins=25)
    expected['S_pMSE'] = expected['S_pMSE'].astype(
        np.float32)  # Output should be float32
    pd.testing.assert_frame_equal(
        output, expected, check_exact=False, rtol=1e-9)


def test_pairwise_spmse_expected_frequency_warning():

    orig_df = pd.DataFrame({'c1': [0, 0]})
    syn_df = pd.DataFrame({'c1': [0, 0, 0, 0]})
    expected = pd.DataFrame(
        {"column1": ['c1'], "column2": ['c1'], "S_pMSE": [0.0]}).astype({"S_pMSE": np.float32})

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


def test_pairwise_spmse_extensive_output():
    orig_df = pd.DataFrame(
        {
            "c1": [1, 0, np.nan],
            "c2": ['a', pd.NA, 'c'],
            "c3": [6, 7, 3]
        }
    )

    syn_df = pd.DataFrame(
        {
            "c2": [pd.NA, pd.NA, pd.NA],
            "c3": [6, 3, 6],
            "c1": [np.nan, np.nan, 0]
        }
    )

    expected = pd.DataFrame(
        {
            "column1": ["c1", "c1", "c1", "c2", "c2", "c3"],
            "column2": ["c1", "c2", "c3", "c2", "c3", "c3"],
            "S_pMSE": [4/3, 8/3, 4/3, 3., 20/9, 0.]
        }
    ).astype({"S_pMSE": np.float32})

    output = pairwise_spmse(orig_df, syn_df, max_bins=3)

    pd.testing.assert_frame_equal(
        output, expected, check_exact=False, rtol=1e-9)


def test_pairwise_spmse_symmetry():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 4], "b": ["x", "x", "z"]})

    r1 = pairwise_spmse(df1, df2)
    r2 = pairwise_spmse(df2, df1)

    pd.testing.assert_frame_equal(
        r1.sort_values(["column1", "column2"]).reset_index(drop=True),
        r2.sort_values(["column1", "column2"]).reset_index(drop=True),
    )
    # Tests symmetry s.t. spmse(X,Y) == spmse(Y,X)


def test_pairwise_spmse_scaling_invariance_on_identical_distributions():
    orig_df1 = pd.DataFrame({"c1": [1, 3], "c2": ['a', 'b']})
    syn_df1 = pd.DataFrame({"c1": [1, 3], "c2": ['a', 'b']})

    expected = pd.DataFrame(
        {
            "column1": ["c1", "c1", "c2"],
            "column2": ["c1", "c2", "c2"],
            "S_pMSE": [0.0, 0.0, 0.0]
        }
    ).astype({"S_pMSE": np.float32})

    orig_df2 = orig_df1.loc[orig_df1.index.repeat(100)].reset_index(drop=True)
    syn_df2 = syn_df1.copy()

    output1 = pairwise_spmse(orig_df1, syn_df1, max_bins=3)
    output2 = pairwise_spmse(orig_df2, syn_df2, max_bins=3)

    pd.testing.assert_frame_equal(
        output1, output2, check_exact=False, rtol=1e-9)

    assert output1.equals(expected)


def test_pairwise_spmse_no_division_by_zero():
    orig_df = pd.DataFrame({"c1": ["a"]*1001+['b']})
    syn_df = pd.DataFrame({"c1": ["a"]+['b']*1001})

    output = pairwise_spmse(orig_df, syn_df, max_bins=3)

    assert np.isfinite(output['S_pMSE']).all()
