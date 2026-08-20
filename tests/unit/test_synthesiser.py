import copy
import re

import numpy as np
import pandas as pd
import pytest
from sklearn import clone
from sklearn.exceptions import NotFittedError

from synthpop.methods.base_synth import BaseSynthMethod
from synthpop.synthesiser import Synthesiser


# ----- stubs -----


class StubSynthMethod(BaseSynthMethod):
    def __init__(self, transform_result=None, name=None):
        super().__init__()
        self.transform_result = transform_result
        self.fit_X = []
        self.fit_y = []
        self.transform_X = []
        self.name = name

    def fit(self, X, y):
        self.fit_X = self.fit_X + [X]
        self.fit_y = self.fit_y + [y]

        return self

    def transform(self, X):
        self.transform_X = self.transform_X + [copy.copy(X)]
        return self.transform_result

    def get_feature_names_out(self, input_features=None):
        # get_feature_names_out is required when inheriting from BaseSynthMethod
        # However, the Synthesiser class does not need it and should not call it.
        # That is why it raises an exception when this method is called, so that the test fails.
        raise Exception(
            "get_feature_names_out should not be called in these tests.")


# ----- helper asserts -----


def assert_fit_call(model, expected_X, expected_y, expected_model):
    assert isinstance(model, expected_model)
    assert isinstance(model.fit_X[0], pd.DataFrame)
    assert model.fit_X[0].equals(expected_X)
    assert model.fit_y[0].equals(expected_y)
    assert len(model.fit_X) == 1, (
        "fitting should happen 1 time per column"
    )


def assert_distinct_instances(objects, origin):
    for a in objects:
        assert objects[a] is not origin, "instance should not be original"
        for b in objects:

            if a == b:
                continue

            assert objects[a] is not objects[b], "instances are not distinct"


# ----- fit tests -----


def test_synthesiser_fit_special_syn_method():
    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=synth_method,
        special_syn_method={
            "a": StubSynthMethod(name="method for a"),
            "c": StubSynthMethod(name="method for c"),
        },
    )

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["a", "b", "c"]

    expected_initial_data = pd.DataFrame({"init": [0, 0]})

    assert_fit_call(synth.models_["a"], expected_X=expected_initial_data,
                    expected_y=test_data["a"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["b"], expected_X=test_data[[
                    "a"]], expected_y=test_data["b"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["c"], expected_X=test_data[[
                    "a", "b"]], expected_y=test_data["c"], expected_model=StubSynthMethod)

    assert synth.models_["a"].name == "method for a"
    assert synth.models_["c"].name == "method for c"


def test_synthesiser_fit_callable_method():
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    counter = 0

    def def_method_factory():
        nonlocal counter
        counter += 1
        return StubSynthMethod(name=f"method{counter}")

    def special_method_factory():
        nonlocal counter
        counter += 1
        return StubSynthMethod(name=f"special_method{counter}")

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=def_method_factory,
        special_syn_method={
            "a": special_method_factory,
            "c": special_method_factory,
        },
    )

    synth.fit(test_data)

    assert synth.models_["a"].name == "special_method1"
    assert synth.models_["b"].name == "method2"
    assert synth.models_["c"].name == "special_method3"


def test_synthesiser_fit_raises_on_wrong_return_of_callable_default_method():
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    def method():
        return np.array([])  # returns something other than a synthesis method

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=method,
    )

    with pytest.raises(TypeError, match=".*default_syn_method.*BaseSynthMethod"):
        synth.fit(test_data)


def test_synthesiser_fit_raises_on_factory_returns_callable():

    def wrong_factory():
        return lambda: 0

    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=wrong_factory,
    )

    with pytest.raises(TypeError, match=".*default_syn_method.*another callable.*BaseSynthMethod"):
        synth.fit(test_data)


def test_synthesiser_fit_raises_on_wrong_return_of_callable_special_method():
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    def method():
        return np.array([])  # returns something other than a synthesis method

    synth = Synthesiser(
        random_seed=2,
        special_syn_method={"b": method},
    )

    with pytest.raises(TypeError, match=r".*special_syn_method.* 'b'.*BaseSynthMethod"):
        synth.fit(test_data)


def test_synthesiser_fit_default_synthesis():
    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=synth_method,
    )

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["a", "b", "c"]

    expected_initial_data = pd.DataFrame({"init": [0, 0]})

    assert_fit_call(synth.models_["a"], expected_X=expected_initial_data,
                    expected_y=test_data["a"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["b"], expected_X=test_data[[
                    "a"]], expected_y=test_data["b"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["c"], expected_X=test_data[[
                    "a", "b"]], expected_y=test_data["c"], expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_, origin=synth_method)


def test_synthesiser_fit_custom_order_by_column_name():
    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=synth_method,
        column_order=["b", "a", "c"],
    )

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["b", "a", "c"]

    expected_initial_data = pd.DataFrame({"init": [0, 0]})

    assert_fit_call(synth.models_["b"], expected_X=expected_initial_data,
                    expected_y=test_data["b"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["a"], expected_X=test_data[[
                    "b"]], expected_y=test_data["a"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["c"], expected_X=test_data[[
                    "b", "a"]], expected_y=test_data["c"], expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_, origin=synth_method)


def test_synthesiser_fit_custom_order_by_column_index():
    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        random_seed=2,
        default_syn_method=synth_method,
        column_order=[2, 1, 0],
    )

    synth.fit(test_data)

    assert synth.n_samples_ == 2
    assert synth.column_order_ == ["c", "b", "a"]

    expected_initial_data = pd.DataFrame({"init": [0, 0]})

    assert_fit_call(synth.models_["c"], expected_X=expected_initial_data,
                    expected_y=test_data["c"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["b"], expected_X=test_data[[
                    "c"]], expected_y=test_data["b"], expected_model=StubSynthMethod)
    assert_fit_call(synth.models_["a"], expected_X=test_data[[
                    "c", "b"]], expected_y=test_data["a"], expected_model=StubSynthMethod)

    assert_distinct_instances(synth.models_, origin=synth_method)


@pytest.fixture
def mock_random_state_manager_and_method(mocker):
    # There is no elegant way to assert that method.fit
    # is called within that contextblock (and not before/after).

    # Testing if create_instance_seed is called within the context block
    # is not so difficult either, but that would make this not a unit test:
    # the test could fail for any problem in the synthesis process.

    # The test about the interaction with RandomStateManager works by
    # replacing RandomStateManager with MockRandomStateManager,
    # which only keeps track of the contextblock.
    # We then modify StubSynthMethod to assert that it has been called
    # in the contextblock (using a class variable of MockRandomStateManager)

    class MockRandomStateManager:
        is_in_contextblock = False

        def __enter__(self):
            MockRandomStateManager.is_in_contextblock = True

        def __exit__(self, exc_type, exc, tb):
            MockRandomStateManager.is_in_contextblock = False

        def __init__(self):
            pass

    class MockSynthMethod(StubSynthMethod):
        def fit(self, X, y):
            assert MockRandomStateManager.is_in_contextblock, "fit called outside context block"
            return super().fit(X, y)

        def transform(self, X):
            assert MockRandomStateManager.is_in_contextblock, "transform called outside context block"
            return super().transform(X)

    return mocker.patch(
        "synthpop.reproducibility.RandomStateManager", return_value=MockRandomStateManager()), MockSynthMethod()


def test_synthesiser_fit_sets_seed_given_in_init(mock_random_state_manager_and_method):
    # The aim of this test is to assert that fit is called within the context block with the correct seed.

    expected_seed = 1234

    mocked_rsm = mock_random_state_manager_and_method[0]
    synth = Synthesiser(
        random_seed=expected_seed,
        default_syn_method=mock_random_state_manager_and_method[1],
    )

    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth.fit(test_data)

    # Entering the context block resets the seed sequence.
    # This would cause all the instance seeds to be not independent.
    # That is why we need to assert that exactly one context block has been created in the call to synthesiser.fit.

    mocked_rsm.assert_called_once_with(seed=expected_seed)


def test_synthesiser_fit_passes_no_seed(mock_random_state_manager_and_method):
    mocked_rsm = mock_random_state_manager_and_method[0]

    synth = Synthesiser(
        default_syn_method=mock_random_state_manager_and_method[1],
    )
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })
    synth.fit(test_data)
    mocked_rsm.assert_called_once_with(seed=None)


def test_synthesiser_fit_throws_on_non_dataframe():
    not_a_df = {}
    synth = Synthesiser(random_seed=3)

    with pytest.raises(
        ValueError,
        match="X must be a pandas DataFrame, got <class 'dict'> instead.",
    ):
        synth.fit(not_a_df)


def test_synthesiser_fit_throws_on_empty_dataframe():
    df = pd.DataFrame()
    synth = Synthesiser(random_seed=3)
    with pytest.raises(ValueError, match="X cannot be empty."):
        synth.fit(X=df)


@pytest.mark.parametrize(
    "column_order,expected_message",
    [
        (["a", "a", "b", "b", "c"],
         "The following columns occur multiple times in Synthesiser.column_order: ['a' 'b']"),
        ([2, 2, 1, 1, 0], "The following columns occur multiple times in Synthesiser.column_order: [1 2]"),
        (["a", "d", "c", "x"],
         "The following columns of Synthesiser.column_order are not in the dataframe: ['d', 'x']"),
        ([0, 3, 2, 4], "The following indices of Synthesiser.column_order are out of bounds: [3 4]"),
        ([0, "b", 1],  "Synthesiser.column_order expects input to be a list of column names (str) or column indices (int), "
         "got datatypes {'str', 'int'} instead."),
        ([0, -2, 1, -1], "The following indices of Synthesiser.column_order are negative: [-2 -1]"),
        ([True, False, True, False], "Synthesiser.column_order expects input to be a list of column names (str) or column indices (int), "
         "got datatypes {'bool'} instead."),
        (pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]}),
         "Synthesiser.column_order expects input to be a list of column names (str) or column indices (int), got DataFrame instead."),
        (pd.Index(['a', 'b', 'c']),
         "Synthesiser.column_order expects input to be a list of column names (str) or column indices (int), got Index instead."),
        (np.array(['a', 'b', 'c']),
         "Synthesiser.column_order expects input to be a list of column names (str) or column indices (int), got ndarray instead."),
    ]
)
def test_synthesiser_fit_throws_on_invalid_column_order(column_order, expected_message):
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })

    synth = Synthesiser(
        column_order=column_order,
        random_seed=2,
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        synth.fit(test_data)


# ----- generate tests -----


def test_generate_with_default_configuration():
    synth = Synthesiser(random_seed=2)

    expected_result = pd.DataFrame({
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
        "c": ["q", "w", "e"],
    })

    expected_initial_data = pd.DataFrame({"init": [0, 0, 0]})

    synth.column_order_ = ["a", "b", "c"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(transform_result=expected_result["a"])
    synth.models_["b"] = StubSynthMethod(transform_result=expected_result["b"])
    synth.models_["c"] = StubSynthMethod(transform_result=expected_result["c"])

    result = synth.generate()
    assert isinstance(result, pd.DataFrame)
    assert expected_result.equals(result)

    assert synth.models_["a"].transform_X[0].equals(expected_initial_data)
    assert synth.models_["b"].transform_X[0].equals(expected_result[["a"]])
    assert synth.models_["c"].transform_X[0].equals(
        expected_result[["a", "b"]])


def test_generate_different_rowcount():
    synth = Synthesiser(random_seed=2)

    expected_row_count = 5

    expected_initial_data = pd.DataFrame(
        {"init": [0] * expected_row_count}
    )

    synth.column_order_ = ["a"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(
        transform_result=pd.Series([], name="a"))

    synth.generate(n=expected_row_count)

    assert synth.models_["a"].transform_X[0].equals(expected_initial_data)


def test_generate_zero_rows():
    synth = Synthesiser(random_seed=2)

    expected_initial_data = pd.DataFrame({"init": np.array([], dtype=int)})

    synth.column_order_ = ["a"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(
        transform_result=pd.Series([], name="a"))

    synth.generate(n=0)

    assert synth.models_["a"].transform_X[0].equals(expected_initial_data)


def test_generate_custom_order():
    synth = Synthesiser(random_seed=2)

    expected_result = pd.DataFrame({
        "c": ["q", "w", "e"],
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
    })

    expected_initial_data = pd.DataFrame({"init": [0, 0, 0]})

    synth.column_order_ = ["c", "a", "b"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(transform_result=expected_result["a"])
    synth.models_["b"] = StubSynthMethod(transform_result=expected_result["b"])
    synth.models_["c"] = StubSynthMethod(transform_result=expected_result["c"])

    result = synth.generate()
    assert isinstance(result, pd.DataFrame)
    assert expected_result.equals(result)

    assert synth.models_["c"].transform_X[0].equals(expected_initial_data)
    assert synth.models_["a"].transform_X[0].equals(expected_result[["c"]])
    assert synth.models_["b"].transform_X[0].equals(
        expected_result[["c", "a"]])


def test_synthesiser_generate_passes_seed_given_in_init(mock_random_state_manager_and_method):
    expected_seed = 753
    mocked_rsm = mock_random_state_manager_and_method[0]
    synth_method = mock_random_state_manager_and_method[1]
    expected_result = pd.DataFrame({
        "c": ["q", "w", "e"],
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
    })

    synth = Synthesiser(
        random_seed=expected_seed,
        default_syn_method=synth_method,
    )

    synth.column_order_ = ["a", "b", "c"]
    synth.n_samples_ = 3
    synth.models_ = {}

    for col in expected_result.columns:
        synth.models_[col] = clone(synth_method)
        synth.models_[col].transform_result = expected_result[col]

    synth.generate(100)
    mocked_rsm.assert_called_once_with(seed=expected_seed)


def test_synthesiser_generate_passes_seed_given_in_argument(mock_random_state_manager_and_method):
    expected_seed = 753123
    mocked_rsm = mock_random_state_manager_and_method[0]
    synth_method = mock_random_state_manager_and_method[1]
    expected_result = pd.DataFrame({
        "c": ["q", "w", "e"],
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
    })

    synth = Synthesiser(
        random_seed=1456,
        default_syn_method=synth_method,
    )

    synth.column_order_ = ["a", "b", "c"]
    synth.n_samples_ = 3
    synth.models_ = {}

    for col in expected_result.columns:
        synth.models_[col] = clone(synth_method)
        synth.models_[col].transform_result = expected_result[col]

    synth.generate(100, random_seed=expected_seed)
    mocked_rsm.assert_called_once_with(seed=expected_seed)


def test_generate_raises_when_not_fitted():
    synth = Synthesiser(random_seed=0)
    with pytest.raises(NotFittedError):
        synth.generate()


def test_generate_raises_on_invalid_n():
    synth = Synthesiser(random_seed=2)
    synth.column_order_ = ["c", "a", "b"]
    synth.n_samples_ = 3
    synth.models_ = {}

    with pytest.raises(ValueError, match=re.escape("number of rows of the synthetic data must be positive, got -3")):
        synth.generate(-3)
