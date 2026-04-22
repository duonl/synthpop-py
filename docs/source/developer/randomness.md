# Using randomness in this package

As written in the [functional descriptions](../functional%20descriptions/reproducibility.md), there are requirements for randomness.
Furthermore, [sklearn](https://scikit-learn.org/stable/developers/develop.html#random-numbers) specifies standards for random number too.
On top of that, [numpy](https://numpy.org/doc/stable/reference/random/generator.html) has [their](https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy) own additional recommendations.

It can be seen that sklearn assumes `numpy.random.RandomState` to generate random numbers, while [numpy disrecommends](https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState) `numpy.random.RandomState` for new code. 
Instead, numpy recommends [`numpy.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator) for new code that needs random numbers. Numpy [explicitly](https://numpy.org/neps/nep-0019-rng-policy.html#nep19) does not give a [compatibility guarantee](https://numpy.org/doc/stable/reference/random/compatibility.html). 

## Unifying Numpy, Sklearn, and this package

Most of the recommendations of Numpy and the requirements from the functional description are implemented in  `reproducibility.Reseed`.
If you are developing an estimator in this package, the guidelines by sklearn apply mostly. The points where we deviate are:

- use **`numpy.random.Generator`** instead of `numpy.random.RandomState`.
- if no seed is provided in the initializer, use `Reseed.get_new_seed()` to obtain a random state. 
- use **`Reseed.get_rng(self.random_state_ )`** instead of `check_random_state` to obtain an RNG.
- do **not** store the RNG in the object. 
Using one RNG for the entire object breaks the requirement that methods in this package should yield the same output when called multiple times.

The example from sklearn modified for this package is this:
```python
from reproducibility import Reseed

class GaussianNoise(BaseEstimator, TransformerMixin):
    """This estimator ignores its input and returns random Gaussian noise.

    It also does not adhere to all scikit-learn conventions,
    but showcases how to handle randomness.
    """

    def __init__(self, n_components=100, random_state=None):
        self.random_state = random_state
        self.n_components = n_components

    # the arguments are ignored anyway, so we make them optional
    def fit(self, X=None, y=None):
        self.random_state_ = Reseed.get_new_seed() if self.random_state is None else self.random_state
        rng = Reseed.get_rng(self._seed)
        # use the RNG when fitting if needed.

    def transform(self, X):
        n_samples = X.shape[0]
        rng = Reseed.get_rng(self._seed)
        return rng.integers(0, 100, size=n_samples)
```

Some estimators of sklearn (or other packages for that matter) need a "random state" or a "seed". 
In such cases, use `Reseed.get_seed()` to provide a random state that stills follows the requirements of this package.
Note that the requirement that methods in this package should yield the same output when called multiple times could break in this scenario, since we cannot control the RNG in other packages.

