import pandas as pd
import numpy as np
import pytest

from synthpop.utility_metrics.spmse import pairwise_spmse, preprocessing, joint_frequencies, Calc_S_pSME

@pytest.mark.parametrize(
    "orig_df, syn_df, expected",
    [
        (pd.DataFrame([[1,2],[3,4]], columns=['c1', 'c2']), 
        pd.DataFrame([[1,2],[3,4]], columns=['c1', 'c2']), 
        pd.DataFrame([['c1', 'c1', 0.],['c1', 'c2', 0.],['c2','c2',0.]], columns=['column1', 'column2', 'S_pMSE'])) #Desired output format.
    ] #The S_pSME should be 0 for all three as the synthetic and original are the exact same!
)
def test_pairwise_spmse_output(orig_df, syn_df, expected):

    output = pairwise_spmse(orig_df, syn_df)

    assert output.equals(expected)

@pytest.mark.parametrize(
    "orig_df, syn_df, max_bins",
    [
        (pd.DataFrame([[1,2],[3,4]]), pd.DataFrame([[1,2,3],[4,5,6]]), 25), #Check for unequal number of columns
        (pd.DataFrame({"A": [10], "B": [20]}), pd.DataFrame({"A": [10], "C": [20]}), 25), #Check column names not equal
        ([],[], 12), #Check for non pandas dataframes
        (pd.DataFrame(), pd.DataFrame(), 25.), #Check if max_bins is not an integer
        (pd.DataFrame(), pd.DataFrame(), -12), #Check for negative bins
        (pd.DataFrame(), pd.DataFrame(), 35) #Check empty DataFrames
    ] 
)
def test_pairwise_spmse_inputtests(orig_df, syn_df, max_bins):
    with pytest.raises(ValueError):
        pairwise_spmse(orig_df, syn_df, max_bins)

def test_preprocessing():
    df = pd.DataFrame({
        "a": [0, 25, 50, 75, 100],
        "b": ["this", "is", "a", "test", "case"]
    })
    num_bins=10
    
    result = preprocessing(df.copy(), max_bins=num_bins)
    expected_a = pd.cut(df["a"], num_bins, labels=range(num_bins)) #run the pd cut on numeric column
    
    assert result["a"].tolist() == expected_a.tolist()
    assert result["b"].tolist() == df["b"].tolist() #check if string column remains unaffected

@pytest.mark.parametrize(
    "df_input, output",
    [
    (pd.DataFrame({"c1" : [0,0]}), {('c1', 'c1'): pd.DataFrame({"c1" : [0], 'x': [2]})}),
    (pd.DataFrame({"c1" : [0,0], "c2" : [1,1]}), {('c1', 'c1'): pd.DataFrame({"c1" : [0], 'x': [2]}),
                                                    ('c1', 'c2'): pd.DataFrame({"c1" : [0], "c2" : [1], 'x': [2]}),
                                                    ('c2', 'c2'): pd.DataFrame({"c2" : [1], 'x': [2]})})
    ] 
)
def test_joint_frequencies(df_input,output):
    res = joint_frequencies(df_input, 'x')
    for k in output:
        assert k in res
        assert res[k].equals(output[k])

@pytest.mark.parametrize(
    "orig_df, syn_df, should_raise",
    [
        ({('c1', 'c1'): pd.DataFrame({"c1" : [0], 'f_or': [2]})}, {('c1', 'c1'): pd.DataFrame({"c1" : [0], 'f_syn': [2]})}, True),     #THIS SHOULD RAISE AN ERROR as k=1
        
        ({('c1', 'c2'): pd.DataFrame({"c1" : [0,1],  "c2" : [2,3],'f_or': [0,0]})},
        {('c1', 'c2'): pd.DataFrame({"c1" : [0,1], "c2" : [2,3], 'f_syn': [0,0]})}, 
        True), #THIS SHOULD RAISE AN ERROR as E will give negative numbers

        ({('c1', 'c1'): pd.DataFrame({"c1" : [0,1], 'f_or': [2,2]}), 
        ('c1', 'c2'): pd.DataFrame({"c1" : [0,1], "c2" : [1,2], 'f_or': [2,2]}),
        ('c2', 'c2'): pd.DataFrame({"c2" : [1,2], 'f_or': [2,2]})},
        
        {('c1', 'c1'): pd.DataFrame({"c1" : [0,1], 'f_syn': [2,2]}), 
        ('c1', 'c2'): pd.DataFrame({"c1" : [0,1], "c2" : [1,2], 'f_syn': [2,2]}),
        ('c2', 'c2'): pd.DataFrame({"c2" : [1,2], 'f_syn': [2,2]})},
        False) #In this case f_orig=f_synth, so outcome statistic should be 0!

    ] 
)
def test_calc_S_pMSE(orig_df, syn_df, should_raise):
    n_s, n_o = 2, 2
    f_origxy= 'f_or'
    f_synthxy= 'f_syn'
    if should_raise:
        with pytest.raises(ValueError):
            for key in orig_df:
                Calc_S_pSME(orig_df[key], syn_df[key], n_o, n_s,  f_origxy, f_synthxy)
    else:
        res = {}
        for key in orig_df:
            res[key] = Calc_S_pSME(orig_df[key], syn_df[key], n_o, n_s,  f_origxy, f_synthxy)
        assert all(v == 0. for v in res.values())
    return


