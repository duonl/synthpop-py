# Use different methods for different columns
In the previous example, we changed the `default_syn_method`, which applies the same synthesis method to **every** column in the dataset.

In practice, however, different variables often require different treatment. For example:
- most variables may benefit from the default CART synthesis;
- a variable with little relationship to the others may only need to be sampled;
- a structural variable may need to remain unchanged.

The `special_syn_method` parameter makes this possible. It allows you to override the default synthesis method for selected columns while leaving all other columns unchanged.

In this example, we will use the Titanic dataset again to demonstrate several common configurations.

## Use a different method for one column
Suppose we want to synthesise every variable using CART, except for `embarked`, which we simply want to sample from its observed distribution. We pass a dictionary with the column name and the desired synthesis method to the `special_syn_method`:
```python
from synthpop import Synthesiser
from synthpop.methods import SampleMethod

synthesiser = Synthesiser(
    random_seed=1,
    special_syn_method={
        "embarked": SampleMethod(),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
Only the `embarked` column is synthesised using {class}`~synthpop.methods.sample_synth.SampleMethod`. Every other variable continues to use {class}`~synthpop.methods.cart_synth.CartMethod`.

This approach is often useful when a particular variable has weak relationships with the rest of the dataset, making a simpler synthesis method sufficient.

## Copy a column
Sometimes a variable should remain exactly the same as in the original dataset. This can be achieved using {class}`~synthpop.methods.copy_synth.CopyMethod`.
```python
from synthpop.methods import CopyMethod

synthesiser = Synthesiser(
    random_seed=1,
    special_syn_method={
        "survived": CopyMethod(),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
The values of `survived`, which is the first column, are copied directly from the original dataset. This may be appropriate for structural variables that should remain unchanged. We can now check if the first column of the synthetic dataset is identical to the first column of the original dataset:
```python
data.iloc[:, 0].equals(synthetic_data.iloc[:, 0])
```
```text
True
```
The second columns should not be identical:
```python
data.iloc[:, 1].equals(synthetic_data.iloc[:, 1])
```
```text
False
```

```{warning}
`CopyMethod` reproduces the original values exactly and therefore does **not** provide privacy protection for that variable. It should only be used for variables where copying the original values is acceptable.

Additionally, `CopyMethod` requires the synthetic dataset to contain the same number of rows as the original dataset. It cannot be used together with `generate(n=...)` where `n` differs from the size of the original dataset.
```

## Combine all synthesis methods
The real strength of `special_syn_method` is that every column can use a different synthesis method.

For example:
```python
from synthpop.methods import CartMethod

synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=CartMethod(),
    special_syn_method={
        "embarked": SampleMethod(),
        "pclass": CopyMethod(),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
This configuration results in:
**Column**              | **Method**
------------------------|-----------
`embarked`              | Sample
`pclass`                | Copy
all remaining columns   | CART

## Combine with another default synthesis method'
The `default_syn_method` and `special_syn_method` parameters always work together.

For example, suppose we want every variable to use `SampleMethod`, except for `fare`, which should still be synthesised using CART.
```python
synthesiser = Synthesiser(
    random_seed=1,
    default_syn_method=SampleMethod(),
    special_syn_method={
        "fare": CartMethod(),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate()
```
Now the configuration becomes:
**Column**              | **Method**
------------------------|-----------
`fare`                  | CART
all remaining columns   | Sample

This illustrates that `special_syn_method` always overrides the default method for the specified variables.

## Combine with a custom synthesis order
`special_syn_method` can also be combined with a custom `column_order`. The two parameters control different aspects of the synthesis process:
- `column_order` determines **when** a variable is synthesised and what predictors are available.
- `special_syn_method` determines **how** a variable is synthesised.

For example:
```python
synthesiser = Synthesiser(
    random_seed=1,
    column_order=[
        "survived",
        "pclass",
        "sex",
        "age",
        "fare",
        "sibsp",
        "parch",
        "embarked",
    ],
    special_syn_method={
        "embarked": SampleMethod(),
    },
)

synthesiser.fit(data)

synthetic_data = synthesiser.generate(n=5000)
```
Here:
- the variables are synthesised in the specified order;
- `embarked` is still generated using `SampleMethod`;
- all remaining variables continue to use `CartMethod`.

The synthesis order and synthesis method are completely independent settings and can be configured together.

## Summary
The `special_syn_method` parameter allows individual variables to use a different synthesis method than the default. This makes it possible to build hybrid synthesis workflows where each variable is synthesised using the method most appropriate for its role in the dataset.

More information about the parameter can be found in [User Guide 2: Synthetic data generation](../user_guides/2_synthetic_data_generation.md). More information about the available synthesis methods can be found in [User Guide 3: Synthesis methods](../user_guides/3_synthesis_methods.md).

## Next steps
So far we have configured the `Synthesiser` using the available synthesis methods and their default settings.

In the next example, we will learn how to customise these methods further. In particular, we will see how to tune the CART synthesis method by changing the underlying decision tree parameters and preprocessing components to better match the characteristics of a dataset.