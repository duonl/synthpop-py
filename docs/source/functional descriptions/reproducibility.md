# Reproducibility and Randomness

Synthetic data generation requires both determinism and high-quality randomness. These goals are not contradictory but must be explicitly engineered. Reproducibility ensures that results are deterministic and auditable. Randomness quality ensures that generated data does not exhibit artificial structure or leak information about the source of the data.

Poor handling of either undermines trust. Non-determinism breaks pipelines and debugging, while also makes it impossible to recreate the synthetic data. Weak randomness can introduce bias, or in extreme cases, enable reconstruction of the original data.

We assume users generate datasets via Python scripts.

## Reproducibility
This package enforces strong reproducibility guarantees:
- **Deterministic execution**: running the same script with the same inputs and seed must yield identical synthetic datasets.
- **Function-level determinism**: repeated calls to the same method with identical arguments and seeds must produce identical outputs.
- **Multiple datasets per run**: Users must be able to generate multiple independent datasets from the same observed data within a single script in a controlled and reproducible way.
- **Explicit seeding**: a root seed must be provided by the user. No implicit or hidden seeding is allowed.

## Randomness
While deterministic, the system must still produce statistically sound randomness:
- **Independence of random streams**: all random number generators (RNGs) used across components must be independent unless explicitly coupled.
- **Robust seeding**: user-provided seeds are normaised via a `SeedSequence` to avoid poor statistical properties. See [here](https://numpy.org/doc/stable/reference/random/parallel.html) for considerations.
- **No hidden correlations**: derived RNGs must not introduce unintended correlations between components.

## Constraints
- **No default seed**: if no seed is provided, the package must raise an explicit error.
- **Dependency handling**: any dependency that requires randomness must be driven by the same controlled seeding mechanism.

## Design
The package uses a hierarchical seeding strategy:
- A **root seed** is supplied by the user when creating a `Synthesiser`.
- Each component that requires randomness derives its own **instance seed**.
- RNGs are constructed from the combination of a rood seed and an instance seed.
This ensures that identical seeds give identical RNG streams. Different instance seeds give independent streams and different root seeds give entirely different experiments.

Using these seeds an RNG is created. We leverage NumPy's internal `SeedSequence` logic to safely combine entropy sources. Seeding management is handled in a class that provides a global root seed registry, a context manager for temporary reseeding and utilities for generating derived seeds.

