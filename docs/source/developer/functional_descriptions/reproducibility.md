# Reproducibility and Randomness

Synthetic data generation requires both determinism and high-quality randomness. These goals are not contradictory but must be explicitly engineered. Reproducibility ensures that results are deterministic and auditable. Randomness quality ensures that generated data does not exhibit artificial structure or leak information about the source of the data.

Poor handling of either undermines trust. Non-determinism breaks pipelines and debugging, while also makes it impossible to recreate the synthetic data. Weak randomness can introduce bias, or in some cases, enable reconstruction of the original data.

We assume users generate datasets via Python scripts.

## Reproducibility
This package enforces strong reproducibility guarantees:
- **Deterministic execution**: running the same script with the same inputs and seed must yield identical synthetic datasets.
- **Function-level determinism**: repeated calls to the same method with identical arguments and seeds must produce identical outputs.
- **Multiple datasets per run**: Users must be able to generate multiple independent datasets from the same observed data within a single script in a controlled and reproducible way.
- **Implicit seeding**: If the user does not provide a seed, a secure random seed will be generated. The above mentioned "Deterministic execution" does not apply in this case.

Determinism assumes identical call order and identical seeds passed to all random number generator (RNG) creating functions.

## Randomness
While deterministic, the system must still produce statistically sound randomness:
- **Deterministic variability**: different instance seeds produce different random streams under te same root seed.
- **Root-seed isolation**: changing the root seed results in entirely different random streams across the system.
- **Robust seeding**: root seeds are processed through `SeedSequence` to avoid poor statistical properties from low-entropy or structured seeds. See [here](https://numpy.org/doc/stable/reference/random/parallel.html) for considerations. Note that the default RNG of numpy uses a `SeedSequence` internally. 
- **No hidden correlations**: derived RNGs must not introduce unintended correlations between components.

## Constraints
- **No default seed**: The root seed is either provided by the user or created non-deterministically. No seed (root or instance) should be hard-coded in this package.
- **Centralised randomness**: all randomness must be created via the package's RNG utilities.
- **Dependency handling**: any dependency that requires randomness must accept externally provided seeds or RNGs derived from this system.

## Design
The package uses a deterministic seed composition strategy:
- A **root seed** is supplied by the user when creating a `Synthesiser`. This seed is stored globally.
- Each component that requires randomness derives its own **instance seed**.
- RNGs are constructed from the combination of a rood seed and an instance seed.
This ensures that identical seeds give identical RNG streams. Different instance seeds give independent streams and different root seeds give entirely different experiments.

Using these seeds an RNG is created. We leverage NumPy's internal `SeedSequence` logic to safely combine entropy sources. Seeding management is handled in a class that provides a global root seed registry, a context manager for temporary reseeding and utilities for generating derived seeds.

