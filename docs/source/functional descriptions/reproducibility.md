# Reproducibility and Randomness

For the trustworthiness of synthetic data, it is important to be able to reproduce a synthetic dataset. 
It might even be a formal requirement for some users. 
Furthermore, a predictable package helps the user to build a reliable process to make a synthetic dataset.
Random behaviour can frustrate debugging efforts. 

The methods to make a synthetic datasets fundamentally need randomness. 
If the randomness is not random enough it might reduce the quality of the generated synthetic datasets by introducing patterns that are not in the observed data.
It is not excluded that a lack of randomness can make it possible to reconstruct the observed data. 

We expect that the user is running a Python script to produce a synthetic dataset.

## Reproducibility

- consecutive runs of the Python script should yield the same synthetic dataset.
- Calling methods and functions of this package multiple times should yield the same result. 
- The user should be able to generate multiple synthetic datasets from the same observed dataset in the same script.
- The seed for the randomness must be supplied by the user.


## Randomness

- all random numbers used in this package should be independent.
- Care should be taken to properly seed the randomness. See [here](https://numpy.org/doc/stable/reference/random/parallel.html) for some indication of what to consider. Not every user-provided seed is a good seed.


## other constraints

- There should be no default seed. If the user does not provide a seed, an error should be raised. 
- Some dependencies of this package might require randomness as well. 

