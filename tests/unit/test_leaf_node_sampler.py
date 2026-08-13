import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from synthpop.methods.tree_utils import LeafNodeSampler


# ----- stubs -----


class StubRNG:
    def __init__(self, values):
        self.values = iter(values)

    def integers(self, low, high, size=None):
        if size is None:
            v = next(self.values)
            return low + (v % (high - low))

        out = []
        for _ in range(size):
            v = next(self.values)
            out.append(low + (v % (high - low)))
        return np.array(out)


# ----- fit sampler test cases -----


@pytest.mark.parametrize(
    "leaf_ids, y, expected_map",
    [
        # --- numeric numpy arrays ---
        (
            np.array([10, 10, 20, 20]),
            np.array([0, 0, 1, 1]),
            {10: {0: 2}, 20: {1: 2}},
        ),
        # --- lists ---
        (
            [10, 10, 20],
            [0, 1, 1],
            {10: {0: 1, 1: 1}, 20: {1: 1}},
        ),
        # --- pandas Series ---
        (
            pd.Series([10, 10, 20]),
            pd.Series([0, 1, 1]),
            {10: {0: 1, 1: 1}, 20: {1: 1}},
        ),
        # --- strings ---
        (
            np.array([1, 1, 2]),
            np.array(["x", "z", "y"]),
            {1: {"x": 1, "z": 1}, 2: {"y": 1}},
        ),
        (
            np.array([1, 1, 2]),
            np.array(["a", "a", "b"]),
            {1: {"a": 2}, 2: {"b": 1}},
        ),
        # --- mixed: None, np.nan, pd.NA ---
        (
            np.array([10, 10, 20, 20]),
            np.array([None, np.nan, pd.NA, 1], dtype=object),
            {10: {None: 1, np.nan: 1}, 20: {pd.NA: 1, 1: 1}},
        ),
        # --- mixed string and integer ---
        (
            [1, 2, 1],
            np.array([1, "1", 1.0], dtype=object),
            {1: {1: 2}, 2: {"1": 1}},
        ),
        # --- single input ---
        (
            np.array([10]),
            np.array([5]),
            {10: {5: 1}},
        ),
    ],
)
def test_fit_sampler_parametrized_inputs(leaf_ids, y, expected_map):
    """
    From various inputs the correct count mapping should be created.
    """
    sampler = LeafNodeSampler()
    sampler.fit_sampler(leaf_ids, y)

    assert hasattr(sampler, "_leaf_map")
    assert hasattr(sampler, "random_state_")
    assert sampler._leaf_map.keys() == expected_map.keys()

    for leaf_id in expected_map:
        for key, count in expected_map[leaf_id].items():
            assert sampler._leaf_map[leaf_id][key] == count


def test_fit_sampler_raises_shape_mismatch():
    y = np.array([0, 1])
    leaf_ids = np.array([10, 10, 20])
    sampler = LeafNodeSampler()

    with pytest.raises(ValueError, match="must have the same number of samples"):
        sampler.fit_sampler(leaf_ids, y)


def test_fit_sampler_raises_dimension_mismatch():
    sampler = LeafNodeSampler()
    leaf_ids = [[10], [10], [20]]
    y = [0, 1, 2]
    leaf_ids2 = [10, 10, 20]
    y2 = np.array([[0], [1], [2]])

    with pytest.raises(ValueError, match="leaf_ids must be 1-dimensional"):
        sampler.fit_sampler(leaf_ids, y)
    with pytest.raises(ValueError, match="y must be 1-dimensional"):
        sampler.fit_sampler(leaf_ids2, y2)


def test_fit_sampler_raises_non_empty():
    sampler = LeafNodeSampler()
    leaf_ids = []
    y = [0, 1, 2]
    leaf_ids2 = [1, 2, 3]
    y2 = []

    with pytest.raises(ValueError, match="must be non-empty"):
        sampler.fit_sampler(leaf_ids, y)
    with pytest.raises(ValueError, match="must be non-empty"):
        sampler.fit_sampler(leaf_ids2, y2)


@pytest.mark.parametrize(
    "y_input, expected_dtype_check",
    [
        (np.array([1, 2, 3], dtype=int), np.integer),
        (np.array([1.0, 2.0, 3.0], dtype=float), float),
        (np.array([True, False, True]), np.bool_),
        (np.array(["a", "b", "c"]), np.str_),
        (np.array([1, "a", 3], dtype=object), object),
    ],
)
def test_fit_sampler_sets_y_dtype_correctly(y_input, expected_dtype_check):
    sampler = LeafNodeSampler()
    leaf_ids = np.array([10] * len(y_input))
    sampler.fit_sampler(leaf_ids, y_input)

    assert np.issubdtype(sampler._y_dtype, expected_dtype_check)


def test_fit_sampler_creates_seed_from_random_state_manager(mocker):
    X = np.array([10, 10, 20, 20])
    y = np.array([0, 0, 1, 1])
    mock_create_instance_seed = mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_instance_seed",
        return_value=333
    )

    sampler = LeafNodeSampler()
    sampler.fit_sampler(X, y)

    assert sampler.random_state_ == 333
    mock_create_instance_seed.assert_called_once()


def test_fit_sampler_does_not_create_seed_when_seed_is_given(mocker):
    X = np.array([10, 10, 20, 20])
    y = np.array([0, 0, 1, 1])
    mock_create_instance_seed = mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_instance_seed",
        return_value=333
    )

    sampler = LeafNodeSampler(random_state=123456)
    sampler.fit_sampler(X, y)

    assert sampler.random_state_ == 123456
    mock_create_instance_seed.assert_not_called()


# ----- sample from leaves test cases -----


def helper_make_sampler(leaf_map, leaf_ids, random_state=42, y_dtype=np.float32):
    """
    Helper to construct a minimally fitted sampler
    """
    sampler = LeafNodeSampler(random_state=random_state)
    sampler._leaf_map = leaf_map
    sampler.random_state_ = random_state
    sampler._y_dtype = np.dtype(y_dtype)
    return sampler


@pytest.mark.parametrize(
    "leaf_ids",
    [
        np.array([10, 10, 10]),     # numpy numeric
        [10, 10, 10],               # python list numeric
        pd.Series([10, 10, 10]),    # pandas Series numeric
    ],
)
def test_sample_from_leaves_various_input_types(leaf_ids):
    """
    Ensure sample_from_leaves works with multiple input container types and dtypes.
    """

    leaf_map = {10: {0: 3, 1: 1}}
    sampler = helper_make_sampler(leaf_map, leaf_ids, random_state=42)

    y_syn = sampler.sample_from_leaves(leaf_ids)

    expected_values = set(leaf_map[10].keys())

    for val in y_syn:
        assert val in expected_values

    assert len(y_syn) == len(leaf_ids)
    assert y_syn.dtype == sampler._y_dtype


def test_sampling_deterministic_with_stub_rng(mocker):
    rng = StubRNG([0, 3, 1, 2])  # controlled indices

    mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_rng",
        return_value=rng
    )

    sampler = LeafNodeSampler()
    sampler._leaf_map = {10: {0: 3, 1: 1}}
    sampler.random_state_ = rng
    sampler._y_dtype = np.int64

    leaf_ids = [10, 10, 10, 10]

    y = sampler.sample_from_leaves(leaf_ids)

    assert np.array_equal(y, np.array([0, 1, 0, 0]))


@pytest.mark.parametrize("missing_value", [np.nan, pd.NA, None])
def test_sample_from_leaves_each_missing_type(missing_value):
    leaf_ids = [10] * 100
    y_dtype = float if (
        missing_value is not None
        and missing_value is not pd.NA
        and np.isnan(missing_value)
    ) else object

    sampler = helper_make_sampler(
        leaf_map={10: {missing_value: 10, 1: 1}},
        leaf_ids=leaf_ids,
        y_dtype=y_dtype
    )

    y_syn = sampler.sample_from_leaves(leaf_ids)

    if missing_value is None:
        assert any(v is None for v in y_syn)
    elif missing_value is pd.NA:
        assert any(v is pd.NA for v in y_syn)
    else:  # np.nan
        assert any(isinstance(v, float) and np.isnan(v) for v in y_syn)


def test_sample_raises_empty_histogram():
    sampler = helper_make_sampler(
        leaf_map={10: {0: 0, 1: 0}},
        leaf_ids=[10]
    )

    with pytest.raises(ValueError, match="has an empty leaf map"):
        sampler.sample_from_leaves([10])


def test_sample_from_leaves_raises_unseen():
    """
    Unseen leaf id should raise ValueError.
    """
    sampler = helper_make_sampler(
        leaf_map={10: {0: 1}},
        leaf_ids=[999]  # not in leaf_map
    )

    with pytest.raises(ValueError, match="Leaf id .* not seen during fitting"):
        sampler.sample_from_leaves(np.array([0]))


def test_sample_from_leaves_raises_unfitted():
    """
    Missing required attributes should raise AttributeError.
    """
    sampler = LeafNodeSampler()
    with pytest.raises(NotFittedError):
        sampler.sample_from_leaves(np.array([0]))
    sampler._leaf_map = {}
    with pytest.raises(NotFittedError):
        sampler.sample_from_leaves(np.array([0]))


def test_sample_from_leaves_raises_input_val():
    sampler = LeafNodeSampler()
    sampler._leaf_map = {1: {0: 1}}
    sampler.random_state_ = 42
    sampler._y_dtype = np.int64

    with pytest.raises(ValueError, match="leaf_ids must be 1-dimensional"):
        sampler.sample_from_leaves([[1]])
    with pytest.raises(ValueError, match="leaf_ids must be non-empty"):
        sampler.sample_from_leaves([])


def test_sample_from_leaves_creates_rng(mocker):
    X = np.array([10, 10, 20, 20])
    y = np.array([0, 0])

    leaf_ids = [10] * 2
    leaf_map = {10: {2: 10, 1: 1}}

    expected_rng = np.random.default_rng()
    mock_create_rng = mocker.patch(
        "synthpop.reproducibility.RandomStateManager.create_rng",
        return_value=expected_rng
    )
    mock_sample_array = mocker.patch(
        "synthpop.methods.tree_utils._sample_array",
        return_value=y
    )

    sampler = helper_make_sampler(
        leaf_map=leaf_map, leaf_ids=leaf_ids, random_state=987)
    sampler.sample_from_leaves(leaf_ids)

    mock_create_rng.assert_called_once_with(987)
    mock_sample_array.call_args_list[0][0] is expected_rng, (
        "rng is not used to sample"
    )


# ----- clonability tests -----


def test_clone_works_and_fitted_sampler_does_not_preserve_state():
    sampler = LeafNodeSampler(random_state=42)
    sampler.fit_sampler([10, 20, 30], np.array([0, 1, 1]))

    cloned = sampler.clone()

    # Fitted attributes should NOT be copied, original remains intact
    for attr in ["_leaf_map", "random_state_"]:
        assert not hasattr(cloned, attr)
        assert hasattr(sampler, attr)
    assert hasattr(cloned, "random_state")
    assert hasattr(sampler, "random_state")
