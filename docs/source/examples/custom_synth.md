# Create custom synthesiser method
Synthesis methods are the central part of synthpop-py's synthesis workflow.
A synthesis method learns how to generate a target column based on zero or more already synthesised predictor columns. synthpop-py provides several built-in synthesis methods, but you may want to implement your own method for a specific use case.

This section explains how to create a custom synthesis method that follows the scikit-learn conventions used throughout synthpop-py. 
Specifically, we will create a simple synthesis method that generates a target variable using its mean if it is numeric, or its most common value if it is categorical.

## BaseSynthMethod

All synthesis methods in synthpop-py inherit from {class}`~synthpop.methods.copy_synth.BaseSynthMethod`.
This class provides the interface required for a synthesis method to work with the {class}`~synthpop.synthesiser.Synthesiser`.

A synthesis method should implement three main methods:

- `fit`: learn the parameters required to synthesise the target variable based on input data X.
- `transform`: generate a synthetic target variable using the parameters learned during fit.
- `get_feature_names_out`: provide the name of the generated output.
A synthesis method should also work when there are no predictor variables, meaning that X can be `None`.

A custom synthesiser method can therefore start like:


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

However, this class does not yet learn anything from the input data. We therefore need to implement the fit method.

## Fitting a custom synthesis method

The `fit` method is responsible for learning everything required to generate the synthetic target variable.

As a simple example, we can create a method that learns the mean of a numeric target and the mode of a categorical target:

```python
from typing import Self

import pandas as pd
from sklearn.utils.validation import validate_data

from synthpop.methods import BaseSynthMethod

class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        if X is not None:
            X, y = validate_data(self, X, y, reset=True)
        else:
            y = validate_data(
                self,
                X="no_validation",
                y=y,
                reset=True,
            )

        self.target_name_ = y.name

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()
        else:
            self.value_ = y.mode().iloc[0]

        return self
```

We can now fit our synthesis method to a target variable:
```python
data = pd.DataFrame({
    "age": [20, 30, 40, 50]
})

synth = CustomSynth()
synth.fit(None, data["age"])
```

CHECK FOR NAMING CONVENTIONS!!
After fitting, the learned value is stored in `value_`:
```python
print(synth.value_)
35.0
```
The trailing underscore is intentional. As with other scikit-learn estimators, parameters learned from the training data should be stored as attributes ending in _.

## Tranforming the data

The `transform` method uses the parameters learned during `fit` to generate the synthetic target variable.

For our simple example, we generate the learned value once for every row in the input:

```python
from sklearn.utils.validation import check_is_fitted


class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        if X is not None:
            X, y = validate_data(self, X, y, reset=True)
        else:
            y = validate_data(
                self,
                X="no_validation",
                y=y,
                reset=True,
            )

        self.target_name_ = y.name

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()
        else:
            self.value_ = y.mode().iloc[0]

        return self

    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        check_is_fitted(self, ["value_", "target_name_"])

        n_rows = 1 if X is None else len(X)

        return pd.Series(
            [self.value_] * n_rows,
            name=self.target_name_,
        )
```
We can now generate synthetic values:

```python
X = pd.DataFrame(index=range(5))

syn = CustomSynth()
syn.fit(None, data["age"])

values = syn.transform(X)

print(values)
0    35.0
1    35.0
2    35.0
3    35.0
4    35.0
Name: age, dtype: float64
```

As you can see, fit learns the value from the original target, while transform uses that learned value to generate a synthetic version of the target.

We also added `check_is_fitted` to transform. This ensures that the synthesis method cannot be used before it has been fitted and raises a `NotFittedError` if transform is called too early.

## Feature Names

{class}`~synthpop.methods.copy_synth.BaseSynthMethod` also requires `get_feature_names_out`. This method tells scikit-learn which column is produced by the synthesis method.

For our example, we can return the name of the target column learned during fit: `THIS IS NOT CORRECT YET`

```python
def get_feature_names_out(self, input_features=None):
    check_is_fitted(self, ["target_name_"])
    return [self.target_name_]
```

The complete synthesis method is:

```python
from typing import Self

import pandas as pd
from sklearn.utils.validation import check_is_fitted, validate_data

from synthpop import BaseSynthMethod


class CustomSynth(BaseSynthMethod):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame | None, y: pd.Series) -> Self:
        if X is not None:
            X, y = validate_data(self, X, y, reset=True)
        else:
            y = validate_data(
                self,
                X="no_validation",
                y=y,
                reset=True,
            )

        self.target_name_ = y.name

        if pd.api.types.is_numeric_dtype(y):
            self.value_ = y.mean()
        else:
            self.value_ = y.mode().iloc[0]

        return self

    def transform(self, X: pd.DataFrame | None) -> pd.Series:
        check_is_fitted(self, ["value_", "target_name_"])

        n_rows = 1 if X is None else len(X)

        return pd.Series(
            [self.value_] * n_rows,
            name=self.target_name_,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["target_name_"])
        return [self.target_name_]
```

## Using the custom synthesis method in synthpop-py

Once the custom synthesis method has been implemented, it can be passed to {class}`~synthpop.synthesiser.Synthesiser` using either `default_syn_method` or `special_syn_method`.

For example, if we want column 'B' to be synthesised using our custom method:
```python
synth = Synthesiser(
    special_syn_method={
        "B": CustomSynth()
    }
)

data = pd.DataFrame()
syn_data = synth.fit(data).generate()
```

See examples [Change the default synthesis method](./changing_the_default_method.md) and [Use different methods for different columns](./using_different_methods_for_different_columns.md) for more in depth guidelines on how to change the synthesis method.

## Keep in mind
1. Inherit from BaseSynthMethod. Your synthesis method should inherit from BaseSynthMethod so that it implements the interface expected by synthpop-py.

2. Implement fit and transform. fit should learn all parameters required for synthesis, while transform should use those parameters to generate the synthetic target.

3. Think about handling missing values.

4. Support predictors and no predictors. X may contain already synthesised predictor columns, but your method should also work when X is None.

5. Do not modify the input data. fit and transform should operate on the provided data without changing it in-place.

6. Keep learned parameters separate from constructor parameters. Parameters learned during fit should use a trailing underscore, such as value_ or target_name_.

7. Check that the model is fitted. Use check_is_fitted in transform to ensure that the method cannot be used before fit.

8. Handle missing values. A production synthesis method should explicitly decide how missing values in predictors and the target are handled and apply this behaviour consistently.

9. Support both numeric and categorical targets. BaseSynthMethod is designed to support both types of variables. The exact implementation will depend on the synthesis algorithm.

10. Consider cloning behaviour. If your synthesis method contains another estimator as an initialisation parameter, consider using sklearn.base.clone so that the estimator follows scikit-learn's cloning conventions.

11. Test your estimator. Because synthesis methods are built around scikit-learn conventions, testing the estimator's behaviour and fitted/unfitted states can help identify compatibility issues early.

## Summary
Custom synthesis methods in synthpop-py can be implemented by inheriting from {class}`~synthpop.methods.copy_synth.BaseSynthMethod` and implementing the required `fit`, `transform`, and `get_feature_names_out` methods.

The fit method learns the parameters needed to synthesise a target variable, while transform uses those parameters to generate synthetic values. By passing an instance of the custom method through `special_syn_method` or `default_syn_method`, the method can be applied to your data.

The example above deliberately uses a simple mean/mode strategy. In practice, CustomSynth can be replaced with any synthesis algorithm that follows the {class}`~synthpop.methods.copy_synth.BaseSynthMethod` interface.

## Next Steps
With a custom synthesis method implemented, it can be integrated into the synthesis workflow using special_syn_method.

For a more detailed explanation of the scikit-learn conventions used by custom estimators, see developing scikit-learn estimators.

The next step is to adapt the example above to the synthesis algorithm you want to use and implement the appropriate handling of predictors, categorical variables, numeric variables, and missing values.