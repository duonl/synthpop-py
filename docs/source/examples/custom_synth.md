# Create a custom synthesis method
Synthesis methods are the centre of synthpop-py's synthesis workflow. A synthesis method learns how to generate a target column, optionally using predictor columns that have already been synthesised. synthpop-py provides several built-in synthesis methods, but you may want to implement your own method for a particular use case. See [User Guide 3: Synthesis methods](../user_guides/3_synthesis_methods.md) for an overview of the synthesis methods provided by synthpop-py.

This example shows how to implement a custom synthesis method that follows the [`scikit-learn` conventions](https://scikit-learn.org/stable/developers/develop.html) used throughout synthpop-py. The example method is deliberately simple: for a numeric target, it generates the mean of the observed values; for a categorical target, it generates the most frequent value.

## BaseSynthMethod
All synthesis methods in synthpop-py inherit from {class}`~synthpop.methods.base_synth.BaseSynthMethod`.
This class defines the interface that a synthesis method must provide to work with the {class}`~synthpop.synthesiser.Synthesiser`.

A synthesis method should implement three main methods:

- `fit`: learn the parameters required to synthesise the target from the original target and, where applicable, the predictor variables.
- `transform`: generate synthetic target values using the parameters learned by `fit`.
- `get_feature_names_out`: report the name of the output produces by the method.


A synthesis method must also support cases where there are no predictors (i.e. `X` is `None` or an empty dictionary) available yet. synthpop-py synthesises datasets sequentially. As a result, the first column will never have predictors available.

Altogether, a minimal custom synthesis method should be structured like this:

```python
from typing import Self
import pandas as pd
from synthpop.methods import BaseSynthMethod

class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        return self

    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        return pd.Series()

    def get_feature_names_out(self, input_features=None):
        return []
```

However, as you can see, this class does not yet learn or generate anything. We will implement the behaviour step by step.

### Fit a custom synthesis method
The `fit` method is responsible for learning everything required to generate the synthetic target variable. It receives:
- `X`: the predictor variables that are available when the target is synthesised. These may be `None` if the target has no predictors.
- `y`: the original target variable that the method should learn to synthesise.

As four our example, we will create a method that learns the mean of a numeric target and the mode of a categorical target:

```python
from typing import Self
import pandas as pd
from synthpop.methods import BaseSynthMethod

class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:

        self.target_name_ = y.name
        self.n_samples_ = len(y)
        self.target_dtype_ = y.dtype

        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()

        else:
            self.value_ = y.value_counts().idxmax()

        return self
        
    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        return pd.Series()

    def get_feature_names_out(self, input_features=None):
        return []

```

As you can, the first few lines store metadata about the input and the target. This is useful for two reasons:
1. **Compatibility with `scikit-learn`:** Similarly to creating [custom encoders](./custom_encoder.md), we aim to be `scikit-learn` compatible. `scikit-learn` estimators conventionally store information learned from the training data as attributes ending in `_`. For example, `self.feature_names_in_` and `self.n_samples_` should be defined. See the [`scikit-learn` documentation on custom estimators](https://scikit-learn.org/stable/developers/develop.html) for more information.
2. **Preserving the target:** The target's name and dtype can be lost or changed during processing. Storing `target_name_` and `target_dtype_` allows the generated output to retain the same name and data type as the original target. This is important for compatibility with other parts of synthpop-py.

The learned value is stored as `value_`. The trailing underscore indicates that it is a parameter learned during `fit`, rather than a parameter supplied when the estimator was constructed.

We can now fit the method to a categorical target:

```python
y = pd.Series(['20','30','30','50'], name='age', dtype='str')

synth = CustomSynth()
synth.fit(None, y)
```

The learned value is stored is the most frequent value, `"30"`:
```python
print(synth.value_)
# 30
```

For a numeric target, the method instead learns the mean:

```python
y = pd.Series([20, 30, 30, 50], name='age', dtype='float')

synth = CustomSynth()
synth.fit(None, y)
print(synth.value_)
# 32.5
```

### Transforming the data
The `transform` method uses the parameters learned during `fit` to generate synthetic target values.

For our example, we generate the learned value once for every row in the input data:

```python
from sklearn.utils.validation import check_is_fitted

class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:

        self.target_name_ = y.name
        self.n_samples_ = len(y)
        self.target_dtype_ = y.dtype

        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()

        else:
            self.value_ = y.value_counts().idxmax()

        return self

    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        check_is_fitted(self, ["value_", "target_name_", "target_dtype_"])

        n_rows = 1 if X is None else len(X)

        return pd.Series(
            [self.value_] * n_rows,
            name=self.target_name_,
            dtype=self.target_dtype_,
        )
    
    def get_feature_names_out(self, input_features=None):
        return []
```
Notice that `transform` uses the metadata and learned value from `fit` to construct the output. It also uses `check_is_fitted` to ensure that the method cannot be used before it has been fitted.

We can now generate synthetic values:

```python
X = pd.DataFrame(index=range(5))
y = pd.Series([20, 30, 30, 50], name='age', dtype='float')

syn = CustomSynth()
syn.fit(X, y)

values = syn.transform(X)

print(values)
# 0    32.5
# 1    32.5
# 2    32.5
# 3    32.5
# 4    32.5
# Name: age, dtype: float64
```

As you can see, `fit` learns the value from the original target (and predictors, if supplied), while `transform` uses that learned value to generate a synthetic version of the target.

### Providing get_feature_names_out

{class}`~synthpop.methods.base_synth.BaseSynthMethod` also requires `get_feature_names_out`. This method follows the `scikit-learn` estimator interface and reports the name of the feature produced by the estimator. See the [get_feature_names_out documentation](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out) for more information.

For all current synthesis methods in synthpop-py, the output has the same name as the target that was passed to fit. Therefore, `get_feature_names_out` has a relatively limited role here. It becomes more useful for components that change the number or names of features, such as dimensionality-reduction methods.

Nevertheless, a custom synthesis method should implement the method to follow the expected interface. Following the `scikit-learn conventions`, we can define it as follows:
```python
def get_feature_names_out(self, input_features=None):
    check_is_fitted(self, ["target_name_"])

    if input_features is None:
        if hasattr(self, "feature_names_in_"):
            input_features = list(self.feature_names_in_)
        elif hasattr(self, "n_features_in_"):
            input_features = [
                f"x{i}" for i in range(self.n_features_in_)
            ]
        else:
            input_features = []

    if self.target_name_ is None:
        return input_features

    return [self.target_name_]
```

The complete synthesis method is then:

```python
from typing import Self

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from synthpop import BaseSynthMethod

class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:

        self.target_name_ = y.name
        self.n_samples_ = len(y)
        self.target_dtype_ = y.dtype

        if X is not None:
            self.feature_names_in_ = getattr(X, "columns", None)

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()

        else:
            self.value_ = y.value_counts().idxmax()

        return self

    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        check_is_fitted(self, ["value_", "target_name_"])

        n_rows = 1 if X is None else len(X)

        return pd.Series(
            [self.value_] * n_rows,
            name=self.target_name_,
            dtype=self.target_dtype_,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["target_name_"])

        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = list(self.feature_names_in_)
            elif hasattr(self, "n_features_in_"):
                input_features = [
                    f"x{i}" for i in range(self.n_features_in_)
                ]
            else:
                input_features = []

        if self.target_name_ is None:
            return input_features

        return [self.target_name_]
```

## Use the custom synthesis method in synthpop-py

Once the custom synthesis method has been implemented, it can be passed to {class}`~synthpop.synthesiser.Synthesiser` using either `default_syn_method` or `special_syn_method`.

For example, to use `CustomSynth` specifically for column `B`:
```python
from synthpop import Synthesiser

synth = Synthesiser(
    special_syn_method={
        "A": CustomSynth(),
    },
)

syn_data = synth.fit(data).generate()
```
Alternatively, if the custom method should be used as the default synthesis method, pass it through `default_syn_method`:
```python
synth = Synthesiser(
    default_syn_method=CustomSynth(),
)

syn_data = synth.fit(data).generate()
```

See [Change the default synthesis method](./changing_the_default_method.md) and [Use different methods for different columns](./using_different_methods_for_different_columns.md) for more examples on how to change the synthesis method.

## Things to keep in mind
When implementing a custom synthesis method, consider the following:
1. **Inherit from `BaseSynthMethod`.** This provides the interface expected by synthpop-py.

2. **Implement `fit`, `transform`, and `get_feature_names_out`**. `fit` should learn all parameters required for synthesis, `transform` should use those parameters to generate synthetic values, and `get_feature_names_out` should report the generated output name.

3. **Support both predictors and no predictors.** `X` may contain predictor columns that have already been synthesised, but it may also be `None` when the target has no predictors.

4. **Do not modify input data in place.** `fit` and `transform` should operate on the provided data without modifying it. When working with pandas you could use `copy()` at the beginning of your code to prevent modification.

5. **Keep learned parameters separate from constructor parameters.** Parameters learned during `fit` should use a trailing underscore, such as `value_` or `target_name_`. Parameters supplied by the user should be defined in `__init__`.

6. **Check that the model has been fitted.** Use `check_is_fitted` in `transform` and other methods that depend on parameters learned during `fit`.

7. **Handle missing values explicitly.** A synthesis method should define how missing values in both predictors and the target are handled. synthpop-py provides {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` and {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`. However, you are free to implement your own strategy.

8. **Support the data types required by your method.** If your algorithm requires numeric inputs, categorical encoding may be necessary. synthpop-py provides {class}`~synthpop.data_processing.encoders.MeanEncoder` and {class}`~synthpop.data_processing.encoders.PCAEncoder` fro this purpose. Again, you are free to implement your own custom encoder; see [Example: Custom Encoder](./custom_encoder.md).

9. **Consider cloning behaviour.** If your synthesis method contains another estimator as a constructor parameter, ensure that it follows `scikit-learn`'s cloning conventions. IN particular, estimator parameters should be stored unchanged in `__init__` so that {class}`sklearn.base.clone` can recreate the estimator correctly.

10. **Test your estimator.** Since synthesis methods are built around `scikit-learn`'s conventions,testing fitted and unfitted states and other expected estimator behaviour can help identify compatibility issues early.

## Summary
Custom synthesis methods in synthpop-py can be implemented by inheriting from {class}`~synthpop.methods.base_synth.BaseSynthMethod` and implementing the required `fit`, `transform`, and `get_feature_names_out` methods.

The fit method learns the parameters required to synthesise a target variable, while transform uses those parameters to generate synthetic values. A synthesis method must support both cases where predictor variables are available and cases where there are no predictors.

Once implemented, the method can be supplied to {class}`synthpop.synthesiser.Synthesiser` through `default_syn_method` or `special_syn_method`.

The example in this guide deliberately uses a simple mean/mode strategy. In practice `CustomSynth` can be replaced by any synthesis algorithm that implements the {class}`~synthpop.methods.base_synth.BaseSynthMethod` interface.

## Next steps
With a custom synthesis method implemented, the next step is to adapt the example to your synthesis algorithm and implement the appropriate handling of predictors, categorical and numeric variables, and missing values.

For more information about the `scikit-learn` conventions used by custom estimators, see the [`scikit-learn` developer guide](https://scikit-learn.org/stable/developers/develop.html).

## Contributing to synthpop-py
If you want to contribute your custom synthesis method to synthpop-py itself, rather than using it only in your own project, see the [developer documentation](../developer/developer_index.md). It describes the development workflow, coding conventions, testing requirements, and guidelines for contributing new functionality to the package.

The developer documentation is also the place to start if you would like to improve or extend synthpop-py in other ways.