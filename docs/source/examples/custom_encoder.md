# Create a custom encoder

Encoding of categorical input features is an important part of synthpop-py's internal workflow. Encoding categorical features vastly improves the computation speed, as leaf nodes can be fitted in numerical intervals instead of single value categories (which means $2^k-1$ are required, with $k$ the number of categories). synthpop-py implements two encoder methods: {class}`~synthpop.data_processing.encoders.MeanEncoder` is used if the target column is numeric, and {class}`~synthpop.data_processing.encoders.PCAEncoder` if the target column is categorical. See {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>`, for more theoretical background on encoding.

However, you may want to use a different encoder for a specific use case. In this example we explain how to create a custom encoder. Specifically one that maps categorical data to random numerical values, while following sklearn conventions. If you would rather use an existing alternative encoder, see [configure cart directly](configure_cart_directly.md).

## Encoder requirements for synthpop
For an encoder to be compatible with synthpop it should consider to have the following aspects:
1. Return one dimensional arrays with the same shape, especially if the encoder is required to work with {class}`~synthpop.methods.cart_synth.CartMethod`. However, if you also [build your own synthesis method](./custom_synth.md), there is more flexibility.
2. [Cloneable estimator object](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html), which allows synthpop-py to use the same encoder for the whole dataset.
3. Missing value handling: Many types of data have missing values. As such, if the encoding method itself does not accept missing values a preprocessing step is required. synthpop-py has two methodology's for handling missing values: the {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` and {class}`~synthpop.data_processing.missing_value_handling.ReplaceMissingWithValue`.
4. Reproducibility: Dependent on whether your encoder implements a random state. In this case, synthpop-py has a {class}`~synthpop.reproducibility.RandomStateManager` implementation.

**For developers, these requirements are a must-have**, as they allow for a new encoder to seemingly fit into the synthpop framework. However, if you build an encoder for your own dataset you are free to leave out whatever is required for your use case. In this example we will include all four requirements.

## sklearn conventions
In order to be compatible with `sklearn`, and `synthpop`, a new estimator/encoder should also inherit from base `sklearn` objects explained below. This provides the standard interface and functionality required for your encoder to integrate seamlessly with the rest of the package (such as cloning).

## BaseEstimator
An estimator is an object that fits a model based on some training data. Thereafter one can use that model to infer properties or make predictions on new data. It can be either a classifier or regressor. The base class for all estimators is [BaseEstimator](https://scikit-learn.org/dev/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator). As such, one can start by defining their own estimator as:

 ```python
from sklearn.base import BaseEstimator

class CustomEncoder(BaseEstimator): 

    def __init__(self) -> None:
        super().__init__()
 ```
 
 However, as of yet, this only learns all init parameters from `BaseEstimator`. Ideally, we want to create a function that learns a mapping from the observed data. This can be done by defining a fit function. The core concept of our fit function is to map categorical data to a random numerical value. As such, the following *should* suffice:

  ```python
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator

class CustomEncoder(BaseEstimator):

    def __init__() -> None:
        super().__init__()

    def fit(self, X: npt.NDArray, y=None) -> Self:

        categories = np.unique(X)
        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: np.random.rand()
            for category in self.categories_
        }

        return self
 ```

### Reproducibility

Even though this fitting function will define a mapping for all values in X, it is not a reproducible encoder. This is because {class}`np.random.rand`, will create a new number everytime fit is called, or the file is ran. Luckily, synthpop has built tools surrounding this problem of Reproducibility, using the {class}`~synthpop.reproducibility.RandomStateManager`. Using that class, one could now define:

```python
from synthpop.reproducibility import RandomStateManager

class CustomEncoder(BaseEstimator):
    def __init__(self, random_state: int | None = None) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: npt.NDArray, y=None) -> Self:

        RandomStateManager.set_root_seed(self.random_state)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: RandomStateManager.create_instance_seed()
            for category in self.categories_
        }

        return self
```

One could now call the ``CustomEncoder`` using a random_state argument to enforce reproducibility:
```python
encoder = CustomEncoder(random_state=12) # Arbitrary number
```
For more in depth information on the {class}`~synthpop.reproducibility.RandomStateManager`, please see the API reference or [Example: Make your synthesis reproducible](./reproducible_synthesis.md)

### Handling missing values
Handling missing values is a delicate task. Generally packages have different methodologies and definitions for missing values, for instance, `pandas` has `pd.NA`, whereas `numpy` uses `np.nan`. Moreover, defining a `numpy` array as follows:
```python
np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"])
```
removes the missing values, as it is cast to:
```python
["cat", "dog", "nan", "cat", "nan", "bird"]
```

Inside synthpop-py we internally define a datatype that is able to combine categorical data (such as `str`) with missing values. This is defined as a {class}`np.dtypes.StringDType(na_object=np.nan)`, and stored in the `utils` module.
We can now run:
```python
from synthpop.utils import str_dtype
X = np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"], dtype=str_dtype)

encoder = CustomEncoder(random_state=12)
encoder.fit(X)
```

Now the learned parameter `mapping_` is a dictionairy that stores which type of animal (`X`) should be mapped to which integer value:
```python
print(encoder.mapping_)
{'bird': 570356716, 'cat': 741055479, 'dog': 2285044420, nan: 577497900}
```

We see here that nan is also included with a mapping value. However, we might want to handle nan as a different/specific case we can easily access. As such, we handle it manually:

```python
    def fit(self, X: npt.NDArray, y=None) -> Self:

        RandomStateManager.set_root_seed(self.random_state)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: RandomStateManager.create_instance_seed()
            for category in self.categories_
        }

        self.mapping_[np.nan] = 0

        return self
```
Now we have:
```python
print(encoder.mapping_)
{'bird': 570356716, 'cat': 741055479, 'dog': 2285044420, nan: 0}
```

## TransformerMixin

In order for our encoder to actually map input variables to output variables we need a method that is able to transform the data, given a learned model. Luckily `sklearn` also provides a mixin class for that: [TransformerMixin](https://sklearn.org/dev/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin). We can, and should, also inherit from this class. With that we can define a `transform` function:

```python
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from synthpop.reproducibility import RandomStateManager
from synthpop.utils import str_dtype

class CustomEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, random_state: int | None = None) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: npt.NDArray, y=None) -> Self:

        RandomStateManager.set_root_seed(self.random_state)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: RandomStateManager.create_instance_seed()
            for category in self.categories_
        }

        self.mapping_[np.nan] = 0

        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, ["mapping_"])

        output= [
            self.mapping_[value]
            for value in X.flatten()
        ]
        return np.array(output, dtype=np.float32)
 ```

So we now have a function that is able to transform input data X to an output, using the `mapping_` as is learned by `fit`. If we now run the following code:
```python
X = np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"], dtype=str_dtype)

encoder = CustomEncoder(random_state=12)
encoder.fit(X)
values = encoder.transform(X)
print(values)
[7.4105549e+08 2.2850445e+09 0.0000000e+00 7.4105549e+08 0.0000000e+00
 5.7035674e+08]
```
****We successfully managed to transform our animals into numbers!****![Tada](../images/tada_emoji.gif){width=25px}

As you saw we also added `check_is_fitted` in our `transform` function. Even though this is not a requisite, we heavily recommend the use of this `sklearn` utility function as it checks whether required learned parameters are actually learned by `fit`. It will produce an `NotFittedError` in cases where one calls `transform` before `fit`.

For a more in depth explanation (such as implementing other mixins), please take a careful look at [developing sklearn estimators](https://scikit-learn.org/dev/developers/develop.html#)

## sklearn tags
When creating a custom estimator or transformer that follows the sklearn conventions, it is useful to tell sklearn what kind of estimator it is and what type of data it expects. This is done using [estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags).

Tags provide metadata about an estimator. sklearn can use this information when running estimator checks, validating inputs, and integrating custom estimators with the wider sklearn package.

In our case we find that we explicitly expect the data to be a one-dimensional categorical (list of strings). As such, we can inherent from the base class and mixin, and define specific tags from there:

```python
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from synthpop.reproducibility import RandomStateManager
from synthpop.utils import str_dtype

class CustomEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, random_state: int | None = None) -> None:
        super().__init__()
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()

        tags.input_tags.categorical = True
        tags.input_tags.one_d_array = True
        tags.input_tags.allow_nan = True

        return tags

    def fit(self, X: npt.NDArray, y=None) -> Self:

        RandomStateManager.set_root_seed(self.random_state)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: RandomStateManager.create_instance_seed()
            for category in self.categories_
        }

        self.mapping_[np.nan] = 0

        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, ["mapping_"])

        output= [
            self.mapping_[value]
            for value in X.flatten()
        ]
        return np.array(output, dtype=np.float32)
```
Now the general framework is ready to be used inside synthpop-py:

```python
from synthpop import Synthesiser
from synthpop.methods import CartMethod, TreeClassifierMethod, TreeRegressorMethod

Cart_Custom_Encoder = CartMethod(
    regressor=TreeRegressorMethod(
        encoder=CustomEncoder(),
    ),
    classifier=TreeClassifierMethod(
        encoder=CustomEncoder(),
    )
)

X = np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"]*10, dtype=str_dtype)
Y = np.array([0,1,2,3,3,5]*10) # times 10 make a larger set

synth = Synthesiser(12, default_syn_method=Cart_Custom_Encoder)
synth.fit(pd.DataFrame({'a' : X, 'b': Y}))
syndata = synth.generate()
print(syndata.head(5))
      a  b
0   cat  0
1   NaN  2
2   NaN  2
3   cat  3
4  bird  5
```

For more information on implementing alternative methodologies in CART, see [configure cart directly](configure_cart_directly.md).

## Keep in mind:
1. Test if your encoder works as expected. You can, for instance, check for sklearn compatibility by running [pytest](https://docs.pytest.org/en/stable/):
```python
from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([CustomEncoder()])
def test(estimator, check):
    check(estimator)
```
2. synthpop-py's `CartMethod` expects encoders to be one dimensional, write your functions accordingly and validate the input where appropriate. Make sure the encoder receives the expected one-dimensional categorical input and provide a useful error message when it does not.

3. Handle unseen categories. The data passed to transform may contain categories that were not present when fit was called. Your encoder should therefore define how unseen categories are handled, for example by assigning them a default value or raising a clear error.

4. Handle missing values consistently. Decide how values such as None or np.nan should be treated. The behaviour should be consistent between fit and transform and documented as part of your encoder. Change your sklearn tags accordingly.

5. Keep learned parameters separate from constructor parameters. Parameters learned from the training data should be stored with a trailing underscore, such as categories_ or mapping_. Parameters provided by the user should be defined in `__init__` and stored as attributes with the same name.

6. Do not modify the input data in-place. fit and transform should operate on the provided data without changing the original input.

7. Keep the encoder stateless before fitting. Do not learn categories or mappings in `__init__`. All information derived from the input data should be learned in fit.

## Summary
Custom encoders in synthpop-py can be implemented following sklearn conventions by inheriting from `BaseEstimator` and `TransformerMixin`. 
The encoder should learn its mapping during the fit, and transform categorical values into numerical representations.
Defining appropriate sklearn `__tags__` also helps ensure compatibility with sklearn validation and synthpop-py.

When implementing a custom encoder, it is important to validate one-dimensional categorical input, handle unseen and missing values consistently, keep learned parameters separate from constructor parameters, avoid modifying input data, and remain stateless until fit is called. Running sklearn's estimator checks can help verify that the implementation follows the expected conventions.

## Next Steps
With the custom encoder implemented and validated, the next step is to integrate it into synthpop-py. 
See [configure cart directly](configure_cart_directly.md) for an example of how to connect a custom encoder to the encoding workflow and use it during synthesis.

In the [following example](./custom_synth.md) we will explain how to build your own synthesis model.