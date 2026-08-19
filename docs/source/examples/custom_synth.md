# Create custom synthesiser method
Synthesis methods are the central part of synthpop-py's synthesis workflow.
A synthesis method learns how to generate a target column based on zero or more already synthesised predictor columns. synthpop-py provides several built-in synthesis methods, but you may want to implement your own method for a specific use case. See [User Guide 3: Synthesis methods](../user_guides/3_synthesis_methods.md) for synthesis methods currently implemented in synthpop-py.

This section explains how to create a custom synthesis method that follows the sklearn conventions used throughout synthpop-py. 
Specifically, we will create a simple synthesis method that generates a target variable using its mean if it is numeric, or its most common value if it is a categorical value.

## BaseSynthMethod

All synthesis methods in synthpop-py inherit from {class}`~synthpop.methods.copy_synth.BaseSynthMethod`.
This class provides the interface required for a synthesis method to work with the {class}`~synthpop.synthesiser.Synthesiser`.

A synthesis method should implement three main methods:

- `fit`: learn the parameters required to synthesise the target variable based on input data X.
- `transform`: generate a synthetic target variable using the parameters learned during fit.
- `get_feature_names_out`: provide the name of the generated output.


A synthesis method should also work when there are no predictor variables, meaning that X can be `None`. A custom synthesiser method could therefore start like:


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

However, as you can see, this class does not yet do anything usefull. We start by implementing the fit method.

## Fitting a custom synthesis method

The `fit` method is responsible for learning everything required to generate the synthetic target variable.

As a simple example, we can create a method that learns the mean of a numeric target and the mode of a categorical target:

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

As you can see we store many different type of parameters in the first few lines of the fitting function. We do this, and you should too, for two main reasons:
1. `sklearn` compatibility: Similarly to creating [Custom Encoders](./custom_encoder.md), we aim to be sklearn compatible. As such, `self.feature_names_in_` and `self.n_samples_` should be defined. See the [sklearn docs on custom estimators](https://scikit-learn.org/stable/developers/develop.html) and references therein.

2. Continuity: During the fitting or transform process it is possible datatypes are cast to other datatypes (such as integers to floats). Storing the datatype allows us to always cast the final output to the original input datatype. But also, storing the `target_name_` allows us to give the output data the same name, which is for instance required for the {class}`~synthpop.utility_metrics.spmse.S_pMSE`. You are, of course, free to define whatever you feel like, but be aware that other/missing naming conventions might break upstream compatibility.

The trailing underscore is intentional. As with other sklearn estimators, parameters learned from the training data should be stored as attributes ending in _.

We should now be able to run:

```python
y = pd.Series(['20','30','30','50'], name='age', dtype='str')

synth = CustomSynth()
synth.fit(None, y)
```

The learned value is stored in `value_`, and returns `'30'` as it is the most frequent value in our categorical target data y:
```python
print(synth.value_)
30
```

However, if y is of numeric datatype the function will return the mean value, 32.5:

```python
y = pd.Series([20, 30, 30, 50], name='age', dtype='float')

synth = CustomSynth()
synth.fit(None, y)
print(synth.value_)
32.5
```

## Transforming the data

The `transform` method uses the parameters learned during `fit` to generate the synthetic target variable.

For our simple example, we generate the learned value once for every row in the input:

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
Here you can see that we use the learned parameters in the fit function to cast our output data to our desired format.
Moreover, similarly to creating [Custom Encoders](./custom_encoder.md) we check whether the parameters are learned using `check_is_fitted`.
We can now generate synthetic values:

```python
X = pd.DataFrame(index=range(5))
y = pd.Series([20, 30, 30, 50], name='age', dtype='float')

syn = CustomSynth()
syn.fit(X, y)

values = syn.transform(X)

print(values)
0    32.5
1    32.5
2    32.5
3    32.5
4    32.5
Name: age, dtype: float64
```

As you can see, fit learns the value from the original target (and predictors, if necessary), while transform uses that learned value to generate a synthetic version of the target.

## Get Feature Names

{class}`~synthpop.methods.copy_synth.BaseSynthMethod` also requires `get_feature_names_out`. This method tells `sklearn` which column is produced by the synthesis method. Please see the [get_feature_names_out docs](https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out) for more information. Throughout the package synthpop-py generally does not require `get_feature_names_out`. This is because we define that synthetic data columns have the same name `target_name_` as the original data. `get_feature_names_out` has more practical usecases in dimensionality reductions (such as PCA's) or when naming between input and output are different. However, one should define the function nonetheless, as we aim to be `sklearn` compatible.  

Following the sklearn docs, one could define `get_feature_names_out` as:

```python
def get_feature_names_out(self, input_features=None):
    if not hasattr(self, "target_name_"):
        raise NotFittedError("CustomSynthMethod is not fitted. Call `fit` first.")

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
        if not hasattr(self, "target_name_"):
            raise NotFittedError("CustomSynthMethod is not fitted. Call `fit` first.")

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

## Using the custom synthesis method in synthpop-py

Once the custom synthesis method has been implemented, it can be passed to {class}`~synthpop.synthesiser.Synthesiser` using either `default_syn_method` or `special_syn_method`.

For example, if we want column 'B' to be synthesised using our custom method:
```python
from synthpop import Synthesiser

synth = Synthesiser(
    special_syn_method={
        "B": CustomSynth()
    }
)

syn_data = synth.fit(data).generate()
```

See examples [Change the default synthesis method](./changing_the_default_method.md) and [Use different methods for different columns](./using_different_methods_for_different_columns.md) for more in depth guidelines on how to change the synthesis method.

## Keep in mind
1. Inherit from `BaseSynthMethod`. Your synthesis method should inherit from `BaseSynthMethod` so that it implements the interface expected by synthpop-py.

2. Implement fit and transform. fit should learn all parameters required for synthesis, while transform should use those parameters to generate the synthetic target.

3. Support predictors and no predictors. X may contain already synthesised predictor columns, but your method should also work when X is None.

4. Do not modify the input data. fit and transform should operate on the provided data without changing it in-place. When working with pandas you could start with `copy()` at the beginning of your code.

5. Keep learned parameters separate from constructor parameters. Parameters learned during fit should use a trailing underscore, such as `value_` or `target_name_`.

6. Check that the model is fitted. Use `check_is_fitted` in transform to ensure that the method cannot be used before fit.

7. Handle missing values. A production synthesis method should explicitly decide how missing values in predictors and the target are handled and apply this behaviour consistently. synthpop-py implements two methods for handling missing values: {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` and {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`. But of course, feel free to implement your own methodology.

8. If necessary, support both numeric and categorical targets and input features. The exact implementation will depend on the synthesis algorithm. synthpop-py implements two encoders, {class}`~synthpop.data_processing.encoders.MeanEncoder` and {class}`~synthpop.data_processing.encoders.PCAEncoder`, to make categorical input features numeric. But of course, feel free to implement your own methodology. See [Example: Custom Encoder](./custom_encoder.md) to build your own synthpop-py compatible encoder.

9. Consider cloning behaviour. If your synthesis method contains another estimator as an initialisation parameter, consider using sklearn.base.clone so that the estimator follows sklearn's cloning conventions.

10. Test your estimator. Because synthesis methods are built around sklearn conventions, testing the estimator's behaviour and fitted/unfitted states can help identify compatibility issues early.

## Summary
Custom synthesis methods in synthpop-py can be implemented by inheriting from {class}`~synthpop.methods.copy_synth.BaseSynthMethod` and implementing the required `fit`, `transform`, and `get_feature_names_out` methods.

The fit method learns the parameters needed to synthesise a target variable, while transform uses those parameters to generate synthetic values. By passing an instance of the custom method through `special_syn_method` or `default_syn_method`, the method can be applied to your data.

The example above deliberately uses a simple mean/mode strategy. In practice, CustomSynth can be replaced with any synthesis algorithm that follows the {class}`~synthpop.methods.copy_synth.BaseSynthMethod` interface.

## Next Steps
With a custom synthesis method implemented, it can be integrated into the synthesis workflow.

For a more detailed explanation of the `sklearn` conventions used by custom estimators, see [developing sklearn estimators](https://scikit-learn.org/stable/developers/develop.html) and references therein.

The next step is to adapt the example above to the synthesis algorithm you want to use and implement the appropriate handling of predictors, categorical variables, numeric variables, and missing values.