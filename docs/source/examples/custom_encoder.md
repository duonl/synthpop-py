# Create a custom encoder

Encoding of categorical input features is an important part of synthpop-py's internal {class}`~synthpop.methods.cart_synth.CartMethod` workflow. Converting categorical features to numeric representations can substantially reduce the computational cost of fitting a decision tree, as the tree can split on numeric intervals rather than considering individual category values. Without encoding, a categorical feature with $k$ categories may require up to $2^k-1$ binary partitions, which can become computationally expensive as the number of categories increases.

synthpop-py provides two built-in encoder methods: {class}`~synthpop.data_processing.encoders.MeanEncoder` is used when the target column is numeric, while {class}`~synthpop.data_processing.encoders.PCAEncoder` is used when target column is categorical. See {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>` for more theoretical background on encoding.

In some cases, you may want to use a different encoding strategy. If you would rather use an existing alternative encoder, see [Example: Configure the CART components directly](./configure_cart_directly).

## Encoder requirements for synthpop-py
To integrate an encoder with synthpop-py, there are several requirements to consider. The requirements are particularly important when the encoder is used together with {class}`~synthpop.methods.cart_synth.CartMethod`, which expects encoders to follow a specific interface.
1. **Output shape:** The encoder should return a two-dimensional array with shape `(N, m)`, where `N` is the number of observations in the input and `m` is the number of features produced by the encoder. If you [build your own synthesis method](./custom_synth), you may have more flexibility in how the encoder represents its output.
2. **Cloneability:** The encoder should be a [cloneable estimator object](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html). This allows synthpop-py to create independent copies of an encoder when it is used across the dataset.
3. **Missing values:** Many types of datasets contain missing values. If the encoder does not support missing values itself, they must be handled before encoding.
4. **Reproducibility:** If the encoder uses randomness, its random behaviour should be controlled through a `random_state`. Synthpop-py provides {class}`~synthpop.reproducibility.RandomStateManager` to manage random states consistently throughout the synthesis process. See [the developer guide on reproducibility](../developer/way_of_working/randomness) for more information on this topic.

For **developers implementing an encoder for current synthesis methods in synthpop-py**, these requirements must be met as they allow the new encoder to integrate with the existing synthesis framework. However, if you are developing an encoder for a specific use case or your own synthesis method, you may find that not all features are required to be implemented. Here, we will implement all four requirements.

### `scikit-learn` conventions
In order for an encoder to be compatible with synthpop-py, it should also follow the `scikit-learn` estimator interface. Following these conventions provides a standard interface, allowing synthpop-py to adhere to the requirements, such as cloning. 

Generally, this implies two things:
1. A synthpop-py encoder should inherit from the base class {class}`~sklearn.base.BaseEstimator` and the mixin class for transformers {class}`~sklearn.base.TransformerMixin`
2. Define [estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags) to declare the encoder’s capabilities and input requirements, allowing `scikit-learn` and synthpop-py to validate and test the estimator appropriately.

For more information on developing `scikit-learn` estimators and using other mixins, see the [scikit-learn developer guide](https://scikit-learn.org/dev/developers/develop.html#). In this example, we explain how to create a custom encoder that maps categorical data to random numeric values while following the synthpop-py requirements.

## BaseEstimator: Cloning and `fit` method
An estimator is an object that learns parameters from training data and can subsequently use those learned parameters to transform data or make predictions. For a custom encoder, the `fit` method is responsible for learning the mapping from categorical values to their numeric representations. 

To follow `scikit-learn` convention, a custom encoder should inherit from {class}`~sklearn.base.BaseEstimator`. A minimal encoder can therefore start as follows:

 ```python
from sklearn.base import BaseEstimator
from typing import Self

class CustomEncoder(BaseEstimator): 

    def __init__(self) -> None:
        super().__init__()

    def fit(self) -> Self:
        # Your code
        return self
 ```

Another benefit of inheriting from {class}`~sklearn.base.BaseEstimator` is support for `scikit-learn`'s cloning conventions. Cloning allows synthpop-py to create independent copies of an estimator while preserving its configuration.

For example, {class}`~synthpop.methods.cart_synth.CartMethod` uses cloning when creating and configuring its internal components:

```python
from sklearn import clone
def _new_encoder(self):
    clone(self.encoder) if self.encoder is not None else self._get_encoder()
```
Here, `_get_encoder()` selects the default encoder used by {class}`~synthpop.methods.cart_synth.CartMethod`:
{class}`~synthpop.data_processing.encoders.MeanEncoder` when the target column is numeric, and {class}`~synthpop.data_processing.encoders.PCAEncoder` when the target column is categorical.

### The fit method

Although we defined the basic structure of our encoder in the previous step, it does not yet learn anything from the input data. Inheriting from `BaseEstimator` provides the standard estimator interface and support for functionality such as cloning, but the encoder still needs a `fit` method that learns parameters from the training data.

In this example, we want `fit` to learn a mapping that assigns each observed categorical value a random numeric value. For now, the following implementation is sufficient:

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

## Reproducibility

One limitation of the method above is that it is not reproducible. Although the encoder learns a mapping for each category, every call to {class}`np.random.rand` generates new random values. As a result, fitting the encoder multiple times can produce different mappings.

To make the encoder reproducible, its randomness should instead be controlled through a `random_state`. synthpop-py provides {class}`~synthpop.reproducibility.RandomStateManager` to manage random states throughout the synthesis process. We can therefore modify the encoder to accept a `random_state` and use {class}`~synthpop.reproducibility.RandomStateManager` when generating the numeric mapping:

```python
from synthpop.reproducibility import RandomStateManager

class CustomEncoder(BaseEstimator):
    def __init__(self, random_state: int | None = None) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: npt.NDArray, y=None) -> Self:

        self.random_state_ = RandomStateManager.create_instance_seed() if self.random_state is None else self.random_state
        rng = RandomStateManager.create_rng(self.random_state_)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: rng.random()
            for category in self.categories_
        }

        return self
```
The encoder can now be instantiated with a fixed `random_state` to make its random behaviour reproducible.
```python
RandomStateManager.set_root_seed(0)
encoder = CustomEncoder(random_state=12) # Arbitrary number
```
For more information about the {class}`~synthpop.reproducibility.RandomStateManager`, please see the API reference, [Example: Make your synthesis reproducible](./reproducible_synthesis.md) or [Developer Guide: Using randomness in this package](../developer/way_of_working/randomness.md)

## Handling missing values
Handling missing values is a delicate task. Different Python libraries represent missing values differently. For example, `pandas` uses `pd.NA`, while `numpy` uses `np.nan`. In addition, `numpy` will convert an array containing strings and `np.nan` to a string array, causing the missing values to be represented as the string `"nan"` rather than as actual missing values.

For example:
```python
np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"])
```
will be converted to:
```python
["cat", "dog", "nan", "cat", "nan", "bird"]
```

synthpop-py provides a data type that is able to combine categorical data (such as `str`) with missing values. This is defined as a {class}`np.dtypes.StringDType(na_object=np.nan)`, and is stored in the `utils` module. We can now run:
```python
from synthpop.utils import str_dtype

X = np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"], dtype=str_dtype)

RandomStateManager.set_root_seed(0)
encoder = CustomEncoder(random_state=12)
encoder.fit(X)
```

The resulting `mapping_` is a dictionary that stores the floating-point value assigned to each animal type in `X`:
```python
print(encoder.mapping_)
{'bird': 0.42366158937171916, 'cat': 0.6306062562352069, 'dog': 0.28937982790273686, nan: 0.21039328747277808}
```

We may, however, want to treat missing values separately from the observed categories. We can therefore assign a specific encoded value to `np.nan`:

```python
    def fit(self, X: npt.NDArray, y=None) -> Self:

        self.random_state_ = RandomStateManager.create_instance_seed() if self.random_state is None else self.random_state
        rng = RandomStateManager.create_rng(self.random_state_)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: rng.random()
            for category in self.categories_
        }

        self.mapping_[np.nan] = 0

        return self
```
Now we have:
```python
print(encoder.mapping_)
{'bird': 0.42366158937171916, 'cat': 0.6306062562352069, 'dog': 0.28937982790273686, nan: 0}
```

## TransformerMixin: define a transform method

The `fit` method learns the mapping from categorical values to numeric values, but the encoder also needs a way to apply that learned mapping to incoming data. 

With a `transform` method our encoder can therefore be extended as follows:

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

        self.random_state_ = RandomStateManager.create_instance_seed() if self.random_state is None else self.random_state
        rng = RandomStateManager.create_rng(self.random_state_)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: rng.random()
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
As you may have noticed, we also added `check_is_fitted` in our `transform` method. Even though this is not strictly required, we recommend the use of it because it verifies that the attributes learned during `fit` are available before `transform` is called. If `transform` is called before `fit`, it raises a `NotFittedError` rather than failing later with a less informative error.

## Output shape

Even though our encoder is now fully functional, it still needs to satisfy the last synthpop-py requirement: returning the expected output shape.

 An encoder's `transform` method should return a two-dimensional array with shape `(N, m)`, where `N` is the number of input observations and `m` is the number of features produced by the encoder.
 
 Our encoder currently produces one numeric value for each input observation, resulting in a one-dimensional array with shape `(N,)`. In this case, there is only one output feature (`m = 1`), so we need to reshape the array from `(N,)` to `(N, 1)`.
 
 We can do this by adding a new axis to the output:

```python
return np.array(output, dtype=np.float32).reshape(-1, 1)
```

The encoder is now able to transform input data `X` to an output with shape `(N, 1)`, using the `mapping_` learned during `fit`. Altogether the entire framework will look like:
```python
X = np.array(["cat", "dog", np.nan, "cat", np.nan, "bird"], dtype=str_dtype)

RandomStateManager.set_root_seed(0)
encoder = CustomEncoder(random_state=12)
encoder.fit(X)
values = encoder.transform(X)
print(values)
# [[0.63060623]
#  [0.28937984]
#  [0.        ]
#  [0.63060623]
#  [0.        ]
#  [0.4236616 ]]
```
****Finally, we successfully managed to transform our animals into numbers!****![Tada](../images/tada_emoji.gif){width=25px}


## Estimator tags
Lastly, even though it is not required for an encoder to work, when creating a custom estimator or transformer that follows the `scikit-learn` conventions, it is useful to tell `scikit-learn` what kind of estimator it is and what type of input it expects. This information is provided through [estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags).

Tags provide metadata about an estimator that `scikit-learn` can use for tasks such as validating inputs, running estimator checks, and determining how the estimator can be used within the broader `scikit-learn` ecosystem.

Our encoder expects one-dimensional, categorical input and allows missing values. We can communicate these requirements by overriding the initial `__sklearn_tags__`:

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

        self.random_state_ = RandomStateManager.create_instance_seed() if self.random_state is None else self.random_state
        rng = RandomStateManager.create_rng(self.random_state_)
        categories = np.unique(X)

        self.categories_ = categories.tolist()
        self.mapping_ = {
            category: rng.random()
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
        return np.array(output, dtype=np.float32).reshape(-1, 1)
```
With these tags in place, the encoder provides `scikit-learn` with the information it needs to understand the type of input it accepts.

## Use the custom encoder inside synthpop-py
The custom encoder is now ready to be used with synthpop-py. We can supply it to the CART components in the same way as the built-in encoders. Because both the classifier and regressor can use categorical predictors, we configure each component with an instance of our `CustomEncoder`:

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
#       a  b
# 0   cat  0
# 1   NaN  2
# 2   NaN  3
# 3   cat  0
# 4  bird  5
```

For more information on configuring alternative components in CART, see [Example: Configure the CART components directly](configure_cart_directly.md).

## Things to keep in mind
When implementing a custom encoder, consider the following:
1. **Test the encoder.** You can use `scikit-learn`'s estimator checks to test whether your encoder follows the expected estimator behaviour by running the following with [pytest](https://docs.pytest.org/en/stable/):
```python
from sklearn.utils.estimator_checks import parametrize_with_checks
import pytest

@parametrize_with_checks([CustomEncoder()])
def test(estimator, check):
    check(estimator)
```

2. **Handle unseen categories.** Data passed to `transform` may contain categories that were not present when `fit` was called. Therefore, your encoder should define how unseen categories are handled, for example, by assigning them a default value or raising a clear error.

3. **Handle missing values consistently.** Decide how values such as `None` or `np.nan` should be treated. The behaviour should be consistent between fit and transform and documented as part of your encoder. Change your `scikit-learn` tags accordingly.

4. **Separate learned and constructor parameters.** Parameters learned from the training data should use a trailing underscore, such as `categories_` or `mapping_`. Parameters provided by the user should be defined in `__init__` and stored as attributes with the same name.

5. **Do not modify input data in place.** Both `fit` and `transform` should operate on the provided data without modifying the original input.

6. **Keep the encoder stateless before fitting.** Do not learn categories or mappings in `__init__`. All information derived from the input data should be learned during `fit`.

## Summary
Custom encoders can be integrated with synthpop-py by following the standard [`scikit-learn` conventions](https://scikit-learn.org/stable/developers/develop.html). The encoder should inherit from `BaseEstimator` and `TransformerMixin`, learn its mapping during `fit`, and apply that mapping during `transform`. 

Defining appropriate estimator tags helps communicate the encoder's input requirements to `scikit-learn` and improves compatibility with its validation tools. When implementing a custom encoder for use with `CartMethod`, it is also important to support one-dimensional, categorical input and handle missing and unseen values consistently. Additionally,
you need to keep learned parameters separate from constructor parameters, avoid modifying input data, and remain stateless until `fit` is called. Finally, `scikit-learn`'s estimator checks can help identify compatibility issues before integrating the encoder into the synthesis workflow.

## Next steps
With the custom encoder implemented and validated, it can be integrated into CART as a custom component. See [Example: Configure the CART components directly](configure_cart_directly.md) for an example of connecting a custom encoder to the CART encoding workflow.

If you want to go further and implement an entirely new synthesis method, rather than customising an existing CART component, see the [following example](./custom_synth.md). It provides a step-by-step guide to creating your own synthesis method for synthpop-py.