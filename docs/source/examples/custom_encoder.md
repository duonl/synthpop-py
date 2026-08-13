# Create a custom encoder

Encoding of categorical features is a huge part of synthpop-py's internal workflow. Encoding categorical features vastly improves the computation speed and size of decision trees, as leaf nodes can be fitted in numerical intervals instead of single value categories. synthpop-py inherently implements two encoder methods. {class}`~synthpop.data_processing.encoders.MeanEncoder` is used if the target column is numeric, and {class}`~synthpop.data_processing.encoders.PCAEncoder` if the target column is categorical. See {ref}`Guide 4.1: Encoding categorical predictors <41-encoding-categorical-predictors>`, for more theoretical background on encoding.

However, using other encoders for your specific use cases might be desired. In this section we will explain how one can build their own encoder that maps categorical data to a numerical value, using scikit-learn conventions. If you prefer to use a different encoder that is already built, see [alternative encoding using CART](alternative_encoder.md).

## sklearn conventions
synthpop-py is build around base `sklearn` objects. In order to be compatible with `sklearn`, and `synthpop`, a new estimator/encoder should also inherit from these base `sklearn` objects. This will make sure your encoder has all the required tools to seamlessly fit in with the rest of the software. 

### BaseEstimator
 An estimator is an object that fits a model based on some training data and can use that model to infer properties or make predictions on new data. It can be either a classifier or regressor. The base class for all estimators is [BaseEstimator](https://scikit-learn.org/dev/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator). As such, one can start by defining their own estimator as:

 ```python
from sklearn.base import BaseEstimator

class CustomEncoder(BaseEstimator): 
    def __init__(self):
        super().__init__()
 ```
 
 However, as of yet, this only learns all init parameters from BaseEstimator. Ideally we want to create a function that learns from the initial data. This can be done by defining a `fit` function:

  ```python
class CustomEncoder(BaseEstimator): 
    def __init__(self):
        super().__init__()

     def fit(self, X, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X)
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

However this does not yet do anything with new data, it just learns our model.

### TransformerMixin

In order for our encoder to actually map input variables to output variables we need a method that is able to transform the data, given a learned model. Luckily `sklearn` also provides a mixin class for that: [TransformerMixin](https://scikit-learn.org/dev/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin). We can, and should, also inherit from this class. With that we can define a `transform` function:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CustomEncoder(TransformerMixin, BaseEstimator): 
    def __init__(self):
        super().__init__()

     def fit(self, X, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X)
        self.mapping_ = {
            category: i
            for i, category in enumerate(self.categories_)
        }
        return self

    def transform(self, X) -> Self:
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
np.array([1, 2, 1, 0])
```
We see that we successfully managed to encode our animals into numbers. Ain't that great? 

As you saw we also added `check_is_fitted` in our `transform` function. Even though this is not requisite, we heavily recommend the use of this `sklearn` function as it checks whether required learned parameters are actually learned by `fit`.


For a more in depth explanation (such as implementing other mixins), please take a careful look at [developing scikit-learn estimators](https://scikit-learn.org/dev/developers/develop.html#)

```python

class CustomEncoder(TransformerMixin, BaseEstimator): 
# This class can also inherit from OneToOneFeatureMixin if input- and output features are the same
    def __init__(self):

    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.one_d_array = True
        tags.target_tags.required = False
    
        return tags
    
    def fit(self, X, y=None) -> Self:
        # Learn the unique categories
        self.categories_ = np.unique(X)
        self.mapping_ = {
            category: i
            for i, category in enumerate(self.categories_)
        }
        return self

    def transform(self, X) -> Self:
        check_is_fitted(self, 'mapping_')
        # Convert categories to integers
        return np.array([
            self.mapping_[value]
            for value in X
        ])
```

