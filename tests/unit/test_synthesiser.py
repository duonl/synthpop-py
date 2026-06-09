import pytest
from synthpop.synthesiser import Synthesiser
from synthpop.methods.base_synth import BaseSynthMethod
import pandas as pd
import copy
from sklearn.exceptions import NotFittedError
import re


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


def assert_fit_call(model, expected_X, expected_y, expected_model):
    assert isinstance(model, expected_model)
    assert isinstance(model.fit_X[0], pd.DataFrame)
    assert model.fit_X[0].equals(expected_X)
    assert model.fit_y[0].equals(expected_y)
    assert len(model.fit_X) == 1, "fitting should happen 1 time per column"


def assert_distinct_instances(objects, origin):
    for a in objects:
        assert not (objects[a] is origin), "instance should not be original"
        for b in objects:

            if a == b:
                continue

            assert not (objects[a] is objects[b]), "instances are not distinct"


def test_synthesiser_fit_special_syn_method():
    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    synth = Synthesiser(random_seed=2, default_syn_method=synth_method, special_syn_method={
        "a": StubSynthMethod(name="method for a"),
        "c": StubSynthMethod(name="method for c"),
    })

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


def test_synthesiser_fit_default_synthesis():

    synth_method = StubSynthMethod()
    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    synth = Synthesiser(random_seed=2, default_syn_method=synth_method)

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
        "c": [5, 6]
    })

    synth = Synthesiser(
        random_seed=2, default_syn_method=synth_method, column_order=["b", "a", "c"])

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
        "c": [5, 6]
    })

    synth = Synthesiser(
        random_seed=2, default_syn_method=synth_method, column_order=[2, 1, 0])

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


def test_synthesiser_fit_throws_on_non_dataframe():
    not_a_df = {}
    synth = Synthesiser(random_seed=3)

    with pytest.raises(ValueError, match="X must be a pandas DataFrame, got <class 'dict'> instead."):
        synth.fit(not_a_df)


def test_synthesiser_fit_throws_on_empty_dataframe():
    df = pd.DataFrame()
    synth = Synthesiser(random_seed=3)
    with pytest.raises(ValueError, match="X cannot be empty."):
        synth.fit(X=df)


@pytest.mark.parametrize("column_order,expected_message",
                         [
                             (["a", "a", "b", "b", "c"],
                              "The following columns occur multiple times in Synthesiser.column_order: ['a' 'b']"),
                             ([2, 2, 1, 1, 0], "The following columns occur multiple times in Synthesiser.column_order: [1 2]"),
                             (["a", "d", "c", "x"],
                              "The following columns of Synthesiser.column_order are not in the dataframe: ['d', 'x']"),
                             ([0, 3, 2, 4], "The following indices of Synthesiser.column_order are out of bounds: [3 4]"),
                             ([0, "b", 1], "invalid column order: [0, 'b', 1]"),
                             ([0,1, -1], "negative indices not allowed."),
                         ])
def test_synthesiser_fit_throws_on_invalid_column_order(column_order, expected_message):

    test_data = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    synth = Synthesiser(column_order=column_order, random_seed=2)

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        synth.fit(test_data)


def test_generate_default():

    synth = Synthesiser(random_seed=2)

    expected_result = pd.DataFrame({
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
        "c": ["q", "w", "e"]
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

    expected_initial_data = pd.DataFrame({"init": [0]*expected_row_count})

    synth.column_order_ = ["a"]
    synth.n_samples_ = 3
    synth.models_ = {}
    synth.models_["a"] = StubSynthMethod(transform_result=pd.Series([],name="a"))

    synth.generate(n=expected_row_count)


    assert synth.models_["a"].transform_X[0].equals(expected_initial_data)


def test_generate_custom_order():
    synth = Synthesiser(random_seed=2)

    expected_result = pd.DataFrame({
        "c": ["q", "w", "e"],
        "a": ["x", "y", "z"],
        "b": [1, 2, 3]
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


def test_generate_raises_when_not_fitted():
    X = pd.DataFrame({"a": [0]})
    synth = Synthesiser(random_seed=0)
    with pytest.raises(NotFittedError):
        synth.generate()
