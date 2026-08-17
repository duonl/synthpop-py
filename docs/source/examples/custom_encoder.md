# Create a custom encoder

Encoding of categorical input features is a huge part of synthpop-py's internal workflow. Encoding categorical features vastly improves the computation speed and reduces the size of decision trees, as leaf nodes can be fitted in numerical intervals instead of single value categories. synthpop-py inherently implements two encoder methods. {class}`~synthpop.data_processing.encoders.MeanEncoder` is used if the target column is numeric, and {class}`~synthpop.data_processing.encoders.PCAEncoder` if the target column is categorical. See {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>`, for more theoretical background on encoding.

However, you may want to use a different encoder for a specific use case. This section explains how to create a custom encoder. Specifically one that maps categorical data to numerical values while following scikit-learn conventions. If you would rather use an existing alternative encoder, see [alternative encoding using CART](alternative_encoder.md).

## sklearn conventions
synthpop-py is build around base `sklearn` objects. In order to be compatible with `sklearn`, and `synthpop`, a new estimator/encoder should also inherit from these base `sklearn` objects. This provides the standard interface and functionality required for your encoder to integrate seamlessly with the rest of the software.

### BaseEstimator
 An estimator is an object that fits a model based on some training data and can use that model to infer properties or make predictions on new data. It can be either a classifier or regressor. The base class for all estimators is [BaseEstimator](https://scikit-learn.org/dev/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator). As such, one can start by defining their own estimator as:

 ```python
from sklearn.base import BaseEstimator

class CustomEncoder(BaseEstimator): 

    def __init__(self) -> None:
        super().__init__()
 ```
 
 However, as of yet, this only learns all init parameters from BaseEstimator. Ideally we want to create a function that learns from the initial data. This can be done by defining a `fit` function:

  ```python
from typing import Self
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

class CustomEncoder(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: npt.NDArray, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X).tolist()
        self.mapping_ = {
            category: i
            for i, category in enumerate(self.categories_)
        }
        return self
 ```

We can now run:
```python
X = ["cat", "dog", "cat", "bird"]

encoder = CustomEncoder()
encoder.fit(X)
```

Now the learned parameter `mapping_` is a dictionairy that stores which type of animal (`X`) should be mapped to which integer value:
```python
print(encoder.mapping_)
{"bird": 0, "cat": 1, "dog": 2}
```

However this does not yet do anything with input data, it just learns a model.

### TransformerMixin

In order for our encoder to actually map input variables to output variables we need a method that is able to transform the data, given a learned model. Luckily `sklearn` also provides a mixin class for that: [TransformerMixin](https://scikit-learn.org/dev/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin). We can, and should, also inherit from this class. With that we can define a `transform` function:

```python
from typing import Self
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CustomEncoder(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: npt.NDArray, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X).tolist()
        self.mapping_ = {
            category: i
            for i, category in enumerate(self.categories_)
        }
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, ['mapping_'])
        # Convert categories to integers
        return np.array([
            self.mapping_[value]
            for value in X
        ])
 ```

So we now have a function that is able to transform input data X to an output. If we now run the following code:
```python
X = ["cat", "dog", "cat", "bird"]

encoder = CustomEncoder()
encoder.fit(X)
values = encoder.transform(X)

print(values)
[1, 2, 1, 0]
```
We see that we successfully managed to encode our animals into numbers. Ain't that awesome? 

As you saw we also added `check_is_fitted` in our `transform` function. Even though this is not a requisite, we heavily recommend the use of this `sklearn` utility function as it checks whether required learned parameters are actually learned by `fit`. It will produce an `NotFittedError` in cases where one calls `transform` before `fit`.

For a more in depth explanation (such as implementing other mixins), please take a careful look at [developing scikit-learn estimators](https://scikit-learn.org/dev/developers/develop.html#)

### sklearn tags
When creating a custom estimator or transformer that follows the scikit-learn conventions, it is useful to tell scikit-learn what kind of estimator it is and what type of data it expects. This is done using [estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags).

Tags provide metadata about an estimator. Scikit-learn can use this information when running estimator checks, validating inputs, and integrating custom estimators with the wider scikit-learn package.

In our case we find that we explicitly expect the data to be a one-dimensional categorical (list of strings). As such, we can inherent from the base class and mixin, and define specific tags from there:

```python
class CustomEncoder(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()

        tags.input_tags.categorical = True
        tags.input_tags.one_d_array = True

        return tags

    def fit(self, X: npt.NDArray, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X).tolist()
        self.mapping_ = {
            category: i
            for i, category in enumerate(self.categories_)
        }
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, ['mapping_'])
        # Convert categories to integers
        return np.array([
            self.mapping_[value]
            for value in X
        ])
 ```

Now the general framework is ready to be used inside synthpop-py! See [alternative encoding using CART](alternative_encoder.md) on how to implement your encoder in synthpop.

## Keep in mind:
1. Test if your encoder works as expected. You can, for instance, check for sklearn compatibility by running [pytest](https://docs.pytest.org/en/stable/):
```python
from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([CustomEncoder()])
def test(estimator, check):
    check(estimator)
```
2. synthpop-py expects encoders to be one dimensional, write your functions accordingly and validate the input where appropriate. Make sure the encoder receives the expected one-dimensional categorical input and provide a useful error message when it does not.

3. Handle unseen categories. The data passed to transform may contain categories that were not present when fit was called. Your encoder should therefore define how unseen categories are handled, for example by assigning them a default value or raising a clear error.

4. Handle missing values consistently. Decide how values such as None or np.nan should be treated. The behaviour should be consistent between fit and transform and documented as part of your encoder. Change your sklearn tags accordingly.

5. Keep learned parameters separate from constructor parameters. Parameters learned from the training data should be stored with a trailing underscore, such as categories_ or mapping_. Parameters provided by the user should be defined in __init__ and stored as attributes with the same name.

6. Do not modify the input data in-place. fit and transform should operate on the provided data without changing the original input.

7. Keep the encoder stateless before fitting. Do not learn categories or mappings in __init__. All information derived from the input data should be learned in fit.
