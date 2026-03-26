import pandas as pd
import numpy as np
import pytest

from synthpop.methods.cart_synth import _AbstractTreeMethod

class TestTreeMethod(_AbstractTreeMethod):
    def __init__(self, *,
                 encoder = None,
                missing_handling = None,
                tree_sampler = None,
                criterion,
                splitter,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_features,
                max_leaf_nodes,
                random_state,
                min_impurity_decrease,
                class_weight=None,
                ccp_alpha=0):
        super().__init__(encoder=encoder, missing_handling=missing_handling, tree_sampler=tree_sampler, criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)


#next steps: make TesTreeMethod instanciable, test fixtures, first test.