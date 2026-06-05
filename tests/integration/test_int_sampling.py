import pytest
import numpy as np
from synthpop.methods.tree_utils import LeafNodeSampler


def test_leafnode_sampler_no_bias():
    sampler = LeafNodeSampler()
    n_samples = 300
    nodes = np.array([0]*n_samples)
    y = np.array([0]*int(n_samples*0.5)+[1]*int(n_samples*0.5))

    sampler.fit_sampler(nodes,y)

    result = sampler.sample_from_leaves(nodes)
    assert np.unique(result,return_counts=True)[1][0]!=int(n_samples*0.5)
    assert np.abs((np.unique(result,return_counts=True)[1][0])/n_samples -0.5)<0.05

def test_leafnode_sampler_no_bias_boolean():
    sampler = LeafNodeSampler()
    n_samples = 300
    nodes = np.array([0]*n_samples)

    p_first = 0.6
    p_second = 1-p_first
    y = np.array([True]*int(n_samples*p_first)+[False]*int(n_samples*p_second))

    sampler.fit_sampler(nodes,y)

    result = sampler.sample_from_leaves(nodes)
    assert np.unique(result,return_counts=True)[1][1]!=int(n_samples*p_first)
    assert np.abs((np.unique(result,return_counts=True)[1][1]/n_samples) -p_first)<0.05


