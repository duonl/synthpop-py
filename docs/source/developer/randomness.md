# Using randomness in this package

This package enforces strict requirements on randomness to ensure both reproducibility (deterministic outputs given a seed) and statistical validaty (high-quality, reproducible random streams). These requirements are defined in the [functional descriptions](../functional%20descriptions/reproducibility.md) and are consistent with guidance from the
[scikit-learn's](https://scikit-learn.org/stable/developers/develop.html#random-numbers) standards for random numbers and aligned with [NumPy's](https://numpy.org/doc/stable/reference/random/generator.html) Generator API and [seeding recommendations](https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy), with a focus on deterministic reproducibility.

## Background: NumPy vs. scikitlearn
There is a mismatch between ecosystem conventions:
- Scikit-learn APIs historically rely on `numpy.random.RandomState`.
- NumPy [discourages](https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState) `RandomState` for new code and recommends [`numpy.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator).
- NumPy [explictly stated](https://numpy.org/neps/nep-0019-rng-policy.html#nep19) that it does not [guarantee bitwise reproducibility across versions](https://numpy.org/doc/stable/reference/random/compatibility.html).

The policy in this package is to always use `numpy.random.Generator` and never introduce new uses of `RandomState`.

## Unifying NumPy, Scikit-learn and this package

Most of the recommendations of NumPy and the requirements from the functional description are implemented in `reproducibility.Reseed`.
If you are developing an estimator in this package, the guidelines by scikit-learn apply mostly. The points where we deviate are:

- use **`numpy.random.Generator`** instead of `numpy.random.RandomState`.
- if no seed is provided in the initialiser, use `Reseed.get_new_seed()` to obtain a deterministic instance seed.
- use **`Reseed.get_rng(self.random_state_ )`** instead of `check_random_state` to obtain an RNG.
- do **not** store the RNG in the object. 
Using one RNG for the entire object breaks the requirement that methods in this package should yield the same output when called multiple times.

This results in these three core principles. Randomness must be:
1. Expliclty derived from a root seed
2. Locally instantiated per method call
3. Never stored as mutable state
All randomness ust flow through the package's seed management system (`Reseed`)

## Developer Guidelines
When implementing estimators or components:

**1. Accept a user-facing seed**
```python
def __init(self, random_state: int | None = None):
    self.random_state = random_state
```
**2. Create an instance seed from the root seed if no seed is given**:
```python
from reproducibility import Reseed
def fit(self, X, y):
    self.random_state_ = Reseed.get_new_seed() if self.random_state is None else self.random_state
```
This root seed is set at the beginning of the synthesis pipeline when creating a `Synthesiser` object.

**3. Create RNGs on demand and do not store them**:
```python
def transform(self, X):
    rng = Reseed.get_rng(self.random_state_)
```

**4. Never share RNGs across methods or components**:
Each  method call must construct its own RNG. This guarantees that:
```python
estimator.transform(X) == estimator.transform(X)
```

## Complete example
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

## Interoperability with External Libraries
Some libraries (including scikit-learn) expect a "random state" argument. In those cases, use `Reseed.get_seed()` to provide a random state that stills follows the requirements of this package. Note that passing randomness into external libraries may break this package's guarantee that repeated method calls produce identical outputs. This is unavoidable when delegating randomness outside the package. However, try to find an alternative that supports our package's guarantee.

## Anti-patterns (do not do this)
**Storing RNGS**
```python
self.rng = Reseed.get_rng(self.seed)
```
This breaks reproducibility across repeated calls.

**Using global NumPy randomness**
```python
np.random.rand(...)
```
This bypasses the controlled randomness system.

**Mixing `RandomState` and `Generator`**
```python
np.random.RandomState(...)
```
This leads to inconsistent and legacy behaviour.

