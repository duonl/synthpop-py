import string

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from synthpop.synthesiser import Synthesiser
from synthpop.reproducibility import RandomStateManager

from sklearn.datasets import make_classification, make_regression
from sklearn import tree
from matplotlib import pyplot as plt

from synthpop.utils import str_dtype

def simulate_realistic_dataset_correlations(n_samples=100):
    rng = np.random.default_rng(seed=852456)

    # first column is uniform random between 0 and 1.
    first_column = rng.random((n_samples,))
    # Second column is linearly related to the first
    second_column = first_column*3 + 5.5 + rng.random((n_samples,))*0.1
    # third column is independent categorical
    third_column = rng.choice(["a", "b", "c"], size=n_samples, replace=True)

    # fourth column is correlated with both numeric and categoric variables.
    fourth_column = [first_column[i] if third_column[i] in [
        "a", "b"] else second_column[i] for i in range(n_samples)] + rng.random((n_samples,))*0.1

    # fifth column is categorial with many levels and correlated with both numeric and categorical columns
    # This is done by calculating a numeric value roughly between 0 and 26 and map that value to the alphabet.

    # The thrid column decides if the fifth is near the begin or the end of the alphabet
    distribution_general_means = [9 if third_column[i] in [
        "b", "c"] else 18 for i in range(n_samples)]

    # The first column causes variance in the fifth column
    distribution_means = distribution_general_means + (first_column - 0.5)*6

    alphabet_index = [int(rng.normal(distribution_means[i], 6)) %
                      26 for i in range(n_samples)]

    fifth_column = [string.ascii_lowercase[alphabet_index[i]]
                    for i in range(n_samples)]

    dataset = pd.DataFrame({
        "first": first_column,
        "second": second_column,
        "third": third_column,
        "fourth": fourth_column,
        "fifth": fifth_column
    })

    return (dataset, ["first", "second", "fourth"], ["third", "fifth"])

def make_data_missing(X):

    #We need a pattern of missingness that is different for each column
    # The missingness pattern should not be too predictable.

    for ik, k in enumerate(X.keys()):

        # The missingness is periodic. ever p-th element is missing.
        # The value of p decreases for each column.
        p = (len(X.keys())-ik) 

        values = [v if i % p !=1 else np.nan for i,v in enumerate(X[k])]
        if pd.api.types.is_numeric_dtype(X[k].dtype):
            X[k]=np.array(values)
        else:
            X[k] = np.array(values,dtype=str_dtype)

    return X
def get_test_data_regressor(seed = 10,with_cats=False,with_missing_features=False,with_missing_target=False):
    X,y = make_regression(random_state=seed)

    n_cols = X.shape[1]
    X = {i:X[:,i] for i in range(X.shape[1])}

    idx_cats = [ i for i in range(n_cols) if i%2 == 0]
    if with_cats:
        for idx in idx_cats:
            x = (X[idx]*10).astype(int)
            x_i = [f %26 for f in x]
            X[idx] = np.array([string.ascii_lowercase[i] for i in x_i],dtype = str_dtype)

    if with_missing_features:
        X = make_data_missing(X)

    if with_missing_target:
        y = np.array([v if i%5 !=0 else np.nan for i,v in enumerate(y)])

    return (X,y)

nodes_diag_data = pd.DataFrame()
#[i for i in range(200)]
[7,9,14,17,28,32,35]
@pytest.mark.parametrize("seed",[7,9,14,17,28,32,35])
def test_error_unseen_node(seed):
    X,y = get_test_data_regressor(seed=seed,with_cats=True,with_missing_features= True,with_missing_target=True)

    RandomStateManager.set_root_seed([seed])
    obs = pd.DataFrame(X)
    obs["target"] = y

    # 7 -> 4
    # 9 -> 100
    #14 -> 100
    #17 -> 6
    #28 -> 4

    d_bad_cols = {
        7:4,
        9:"target",
        14:"target",
        17:6,
        28:4
    }

    #bad_col = d_bad_cols[seed]
    

    part_obs = obs#[[1:bad_col]]
    synth = Synthesiser(random_seed=0)
    synth.fit(part_obs)
    # try:
    synth.generate(100)
    # except:
    #     pass
        # tree_method = synth.models_[d_bad_cols[seed]].method_
        # df = pd.DataFrame(tree_method.all_features)
        # df["target"] = tree_method.target_data
        # df.to_csv(f"problematicData/data_seed{seed}_col{d_bad_cols[seed]}.csv")
        # df.to_excel(f"problematicData/data_seed{seed}_col{d_bad_cols[seed]}.xlsx")

    # cols_data = []
    # for col in synth.models_:
    #     tree = synth.models_[col].method_.tree_
    #     X_test = synth.models_[col].method_.all_features
    #     y_test = synth.models_[col].method_.target_data
    #     n_nodes = tree.tree_.node_count
    #     sample_ids = [i for i in range(X_test.shape[0])]

    #     node_indicator = tree.decision_path(X_test)

    #     common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) !=0
    #     common_node_id = np.arange(n_nodes)[common_nodes]
    #     n_nans_in_threshold = pd.isna(tree.tree_.threshold).sum()

    #     diag_data = {
    #         "seed":seed,
    #         "col":col,
    #         "n_nodes":n_nodes,
    #         "n_nodes_used": len(common_node_id),
    #         "is_regressor": isinstance(tree,DecisionTreeRegressor),
    #         "unused_nodes":n_nodes-len(common_node_id),
    #         "percentage": 100*((n_nodes-len(common_node_id))/n_nodes),
    #         "accuracy": tree.score(X_test,y_test),
    #         "n_nan_in_threshold":n_nans_in_threshold
    #     }
    #     cols_data.append(diag_data)

    #     #assert len(common_node_id)==n_nodes, f"trainings data does not hit all nodes for seed { seed} column {col}"

    # nodes_diag_data = pd.DataFrame.from_records(cols_data)
    # nodes_diag_data.reset_index()
    # nodes_diag_data.to_excel(f"problematicData/diag_data{seed}.xlsx")




    

    # try:
        
    # except:
    #     tree.plot_tree(synth.models_[bad_col].method_.tree_)#TODO pinpoint the columns that go wrong
    #     plt.savefig(f"tree_plots/tree_seed_{seed}_col{bad_col}.svg")

@pytest.mark.parametrize("seed",[i for i in range(50)])
def test_error_unseen_node_nan_treshold(seed):
    X,y = get_test_data_regressor(seed=seed,with_cats=True,with_missing_features= True,with_missing_target=True)
    obs = pd.DataFrame(X)
    obs["target"] = y

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    synth.generate(100)

    for col in list(synth.models_.keys()):
        assert not pd.isna(synth.models_[col].method_.tree_.tree_.threshold).any(), f"tree for column {col} with seed {seed} has a nan treshold"
    #synth.generate(100)

    # try:
        
    # except:
    #     tree.plot_tree(synth.models_[bad_col].method_.tree_)#TODO pinpoint the columns that go wrong
    #     plt.savefig(f"tree_plots/tree_seed_{seed}_col{bad_col}.svg")


def test_fit_quality_bad_seeds():
    import warnings
    warnings.simplefilter("error")
    seed=7
    X,y = get_test_data_regressor(seed=seed,with_cats=True,with_missing_features= True,with_missing_target=True)

    RandomStateManager.set_root_seed([seed])
    obs = pd.DataFrame(X)
    obs["target"] = y

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)


    assert tree


def test_reproducibilty_synthesis():

    RandomStateManager.set_root_seed([1])
    obs = simulate_realistic_dataset_correlations(n_samples=1000)[0][["first","second","third"]]

    synth = Synthesiser(random_seed=0)
    synth.fit(obs)

    syn1 = synth.generate(2000)
    syn2 = synth.generate(2000)

    assert syn1.equals(syn2)

    RandomStateManager.set_root_seed([1])
    synth2 = Synthesiser(random_seed=0)
    synth2.fit(obs)

    syn3 = synth2.generate(2000)

    for col in syn3.columns:
        assert (syn3[col] == syn2[col]).all(), f"column {col} not reproduced"

@pytest.mark.parametrize("seed",[i for i in range(50)])
def test_decision_tree_regressor(seed):

    X,y = make_regression(random_state=seed)
    clf = DecisionTreeRegressor(min_samples_leaf=5,   # equivalent to minbucket in synthpop-r
                                      min_impurity_decrease=1e-08,  # equivalent to cp in synthpop-r
                                      random_state=seed)
    

    for ik in range(X.shape[1]):

        # The missingness is periodic. ever p-th element is missing.
        # The value of p decreases for each column.
        p = (X.shape[1]-ik) 

        values = [v if i % p !=1 else np.nan for i,v in enumerate(X[ik])]
        X[:,ik]=np.array(values)

    
    clf.fit(X,y)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    sample_ids = [i for i in range(100)]

    node_indicator = clf.decision_path(X)

    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) !=0
    common_node_id = np.arange(n_nodes)[common_nodes]

    assert len(common_node_id)==n_nodes
