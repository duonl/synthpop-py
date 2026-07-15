# 8. Custom synthesis methods

synthpop-py is designed to be extensible. In addition to the built-in synthesis methods, users can define custom synthesis methods for specialised synthesis requirements.

A synthesis method defines how a single target column is generated based on previously synthesised columns. During synthesis, the {class}`~synthpop.synthesiser.Synthesiser` fits one synthesis method per column and subsequently uses those fitted methods to generate synthetic values sequentially.

Custom synthesis methods are useful when:
- a specialised statistical model is required;
- external machine learning models should be integrated;
- domain-specific synthesis rules need to be implemented;
- existing synthesis methods need to be extended with custom preprocessing.

The recommended approach is to inherit from {class}`~synthpop.methods.base_synth.BaseSynthMethod` and implement the required interface.

## 8.1. Synthesis method workflow

The {class}`~synthpop.synthesiser.Synthesiser` generates data sequentially. Each synthesis method is responsible for generating one target column and received previously synthesised columns as predictors.

During fitting, synthesis methods learn from the original dataset:
```{mermaid}
flowchart TD

    A[Original data]

    subgraph FIT["Synthesiser.fit()"]
        direction TB
        B[Initial predictors]
        C["fit(X, y) for column 1"]
        D["fit(previous columns, y) for column 2"]
        E["fit(previous columns, y) for remaining columns"]

        B --> C --> D --> E
    end

    F[Fitted synthesis methods]

    A --> FIT
    FIT --> F
```

During generation, the fitted methods generate synthetic columns sequentially:
```{mermaid}
flowchart TD

    A[Fitted synthesis methods]

    subgraph GEN["Synthesiser.generate()"]
        direction TB
        B["transform(initial predictors)<br>Generate column 1"]
        C["transform(synthetic column 1)<br>Generate column 2"]
        D["transform(previous columns)<br>Generate remaining columns"]

        B --> C --> D
    end

    H[Synthetic dataset]

    A --> GEN --> H
```

A custom synthesis method therefore follows the same pattern as the built-in methods:
- `fit()` learns all parameters required for synthesis
- `transform()` generates one synthetic column using the fitted model.

## 8.2. Required interface
A custom synthesis method should inherit from {class}`~synthpop.methods.base_synth.BaseSynthMethod`.

The base class follows the [`scikit-learn` estimator API](https://scikit-learn.org/stable/developers/develop.html) by inheriting from:
- {class}`BaseEstimator <sklearn:sklearn.base.BaseEstimator>`
- {class}`TransformerMixin <sklearn:sklearn.base.TransformerMixin>`

This ensures compatibility with tools such as cloning and pipelines.

A synthesis method must implement:
- `fit()`
- `transform()`
- `get_feature_names_out()`

### 8.2.1. Constructor
The constructor should only contain configuration parameters.

For example:
```python
class CustomSynth(BaseSynthMethod):

    def __init__(self, some_param):
        super().__init__()
        self.some_param = some_param
```
Following the `scikit-learn` estimator convention, parameters should be stored as attributes and should not be modified during initialisation.

Learned parameters should be created during `fit()` instead.

### 8.2.2. The fit method
The `fit()` method learns the parameters required to generate the target variable.

The signature is:
```python
fit(
    X: pd.DataFrame | None,
    y: pd.Series
) -> self
```
Where:
- `X` contains original predictor columns that have already been synthesised during the sequential synthesis process (earlier columns in the column order);
- `y` is the original target column;

For the first column in the synthesis order, no predictors are available. The {class}`~synthpop.synthesiser.Synthesiser` provides an initial placeholder DataFrame instead:
```python
pd.DataFrame({"init": np.zeros(n_rows, dtype=int)})
```
Custom methods should therefore not depend on the predictor columns having a particular meaning for the first variable.

A fit method may look like:
```python
def fit(self, X, y):
    self.some_parameter_ = ...
    return self
```
The method should:
- not modify `X` or `y`;
- store fitted parameters as attributes ending in `_`;
- return the fitted object.

### 8.2.3. The transform method
The `transform()` method generates a synthetic version of the target column.

The signature is:
```python
transform(
    X: pd.DataFrame | None
) -> pd.Series
```
During generation, `X` contains previously generated synthetic column. The returned object should be a single named [`pandas.Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html). The series name is used as the column name in the generated dataset.

A transform method may look like:
```python
def transform(self, X):
    synthetic_values = ...

    return pd.Series(
        synthetic_values,
        name="target"
    )
```
The method should:
- use only parameters learned during `fit()`;
- not modify `X`;
- return one synthetic column.

Calling `transform()` before `fit()` should raise an appropriate error.

## 8.3. Implementing a custom synthesis method
A minimal custom synthesis method can be defined as:
```python
from typing import Self
import pandas as pd

from synthpop.methods.base_synth import BaseSynthMethod


class CustomSynth(BaseSynthMethod):

    def __init__(self, some_param) -> None:
        super().__init__()
        self.some_param = some_param

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Self:

        # Learn parameters from X and y
        self.some_parameter_ = self.some_param

        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.Series:

        # Generate synthetic values
        synthetic_values = ...

        return pd.Series(
            synthetic_values,
            name="target"
        )

    def get_feature_names_out(self, input_features=None):
        return ["target"]
```
This method can then be supplied to the {class}`~synthpop.synthesiser.Synthesiser`:
```python
>>> synth = Synthesiser(
...     default_syn_method=CustomSynth(some_param=42)
... )

>>> synthetic_data = (
...     synth
...     .fit(data)
...     .generate()
... )
```
Custom synthesis methods can be assigned to selected variables using `special_syn_method`.

For example:
```python
>>> Synthesiser(
...     default_syn_method=CartMethod(),
...     special_syn_method={
...         "income": CustomSynth(some_param=42),
...         "age": SampleMethod()
...     }
... )
```
In this example:
- `income` is generated using `CustomSynth`;
- `age` is generated using `SampleMethod`;
- all other variables use `CartMethod`.

Each synthesis method is cloned before fitting. Therefore, custom synthesis methods should follow the [`scikit-learn` estimator conventions](https://scikit-learn.org/stable/developers/develop.html).

A full example can be found on the [Examples page](../examples/custom_synth.md).

## 8.4. Adding custom processing components
A new synthesis method is not always required when only the preprocessing needs to change. Existing synthesis methods can be combined with custom preprocessing components. For example, the {class}`~synthpop.methods.cart_synth.CartMethod` is built from separate tree-based synthesis components:
- {class}`~synthpop.methods.cart_synth.TreeRegressorMethod` for numeric targets;
- {class}`~synthpop.methods.cart_synth.TreeClassifierMethod` for categorical target

These components allow customisation of parts of the synthesis process, including:
- the underlying decision tree model;
- the encoder used for categorical predictors (see {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>`);
- the missing value handling strategy (see {ref}`Guide 4.2: Handling missing values <42-handling-missing-values>`);
- the method used for sampling values within terminal leaf nodes.

This allows users to modify individual parts of the CART synthesis pipeline while retaining the general CART workflow.

For example, custom encoders can be implemented using the `scikit-learn` transformer interface:
```python
from sklearn.base import TransformerMixin, BaseEstimator


class CustomEncoder(
    TransformerMixin, # Or a different Mixin like OneToOneFeatureMixin
    BaseEstimator
):

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return transformed_X
```
The default methods can then be customised by replacing the encoder or other components:
```python
CartMethod(
    regressor=TreeRegressorMethod(
        tree=CustomTree(),
        encoder=CustomEncoder(),
        missing_handler=CustomMissingValuePredictor(),
        tree_sampler=CustomSampler()
    ),
    classifier=TreeClassifierMethod(
        tree=CustomTree(),
        encoder=CustomEncoder(),
        missing_handler=CustomMissingValuePredictor(),
        tree_sampler=CustomSampler()
    )
)
```