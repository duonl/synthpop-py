# Frequently Asked Questions and Troubleshooting

This page collects common questions and issues that may arise when using synthpop-py. It will be expanded as new questions, issues and workarounds are identified.

## CART synthesis
<details>
<summary> Why am I seeing a rare-category warning?</summary>
<br>
You may see a warning such as:

> Categorical predictor contains categories occurring fewer than 5 times for more than 25% of the rows.

This warning means that a categorical predictor used by {class}`~synthpop.methods.cart_synth.CartMethod` contains a substantial number of observations belonging to **rare categories**. By default, a category is considered rare when it occurs fewer than `rare_categories_threshold` times. Rare categories can increase the risk of {ref}`unintended attribute disclosure <6122-rare-categories>` because CART may create small groups in which observed target values are more easily reproduced in the synthetic data. The warning is therefore a privacy safeguard. It does not stop synthesis; synthesis continues after the warning.

<p class="fake-h3"><strong>What can I do?</strong></p>

Consider whether the rare categories are appropriate for use as a predictor and whether the associated privacy risk is acceptable. You can adjust the threshold using {func}`~synthpop.methods.cart_synth.tune_cart`:
```python
tune_cart(rare_categories_threshold=10)
```
A higher threshold classifies more categories as rare and makes the check more conservative. A lower threshold classifier fewer categories as rare. By default, `rare_categories_threshold` follows `min_samples_leaf`. More details can be found in [this example](../examples/tune_cart_function.md. You can also disable the check by setting the threshold to `0`:
```python
tune_cart(rare_categories_threshold=0)
```
Disabling the check ony suppresses the warning; it does not remove the underlying privacy risk. See the {ref}`User Guide <6122-rare-categories>` and [Examples](../examples/rare_categories.md) for more information about this risk. Disabling the check can also be done when {ref}`configuring CART directly <314-configuring-cart>`. See [this example](../examples/configure_cart_directly.md) to see how.



</details>

## Performance
<details>
<summary> Why is my synthesis taking so long? </summary>
<br>
Synthesis time depends on both the size and structure of your dataset. In general, synthesis takes longer as the number of rows and variables increases. Some types of variables can also make synthesis considerably more computationally intensive.

One important example is a categorical variable with many possible values (high cardinality). synthpop-py synthesises variables sequentially, using previously synthesised variables as predictors. This means that a variable with many categories can affect the computational cost of not only its own synthesis, but also the synthesis of variables that use it as a predictor.

The default {class}`~synthpop.methods.cart_synth.CartMethod` performs preprocessing of categorical data before fitting the decision tree. In particular, principal component analysis (PCA) can be used to represent categorical data with a smaller number of components. When a categorical predictor has many possible values, this preprocessing and the subsequent tree fitting can become computationally expensive. If that variable is then used as a predictor for several later variables, this computational cost can occur repeatedly during the sequential synthesis process.

<p class="fake-h3"><strong>What can I do?</strong></p>

**Consider whether the high-cardinality variable needs to be used as a categorical predictor.**<br>
First, consider whether the predictor-target relationship makes sense for the variables involved. If a categorical variable has a very large number of possible values, ask whether it is appropriate to use it as a predictor for the target variable. Depending on the meaning of the variables, another representation or synthesis strategy may be more appropriate.

**Reduce the number of principal components.**<br>
CART uses principal components when processing categorical data. Reducing the number of components can reduce the computational cost of this preprocessing and the subsequent modelling. However, using fewer components can also remove information from the representation of the categorical data, so this is a trade-off between computational cost and the information available to the model.

**Consider changing the synthesis order.**<br>
synthpop-py uses previously synthesised variables as predictors. Therefore, the synthesis order determines which variables are available as predictors for each subsequent variable. If a high-cardinality variable is currently synthesised early in the column order, it may be used as a predictor for many later variables. Moving it later in the synthesis order means that it will not be available as a predictor for those earlier variables, which can reduce the amount of computation required for subsequent models. However, moving the variable to the end does not necessarily make the synthesis of that variable itself faster. The variable still has to be synthesised using the predictors that precede it. Changing the order is therefore most useful when the high-cardinality variable is expensive primarily because it is being used as a predictor for many other variables.

Changing the synthesis order can also affect the statistical utility of the synthetic data. The order should therefore be chosen based on both computational considerations and the relationships that you want to preserve. See {ref}`User Guide 2.2.4: Changing the column order <224-column-order>` for more information.

**Consider synthesising the data in strata.**<br>
For a large dataset, you can consider dividing the data into meaningful strata, synthesising each subset separately, and recombining the resulting synthetic data afterwards. Working with smaller subsets can reduce the computational and memory requirements of each individual synthesis run.

This approach is most appropriate when the strata have a meaningful interpretation and it is reasonable to model them separately. For example, you might synthesise separate groups defined by a variable such as region or another structural characteristic, provided that the relationships between the strata do not need to be modelled jointly.

When using this approach, consider how the strata are defined and how many synthetic observations should be generated for each stratum. Synthesising strata independently means that relationships between variables across strata are not modelled by the same synthesis process.

**Consider the size of the dataset.**<br>
Larger datasets generally require more time and computational resources to synthesise. The practical impact depends on the number of rows, number and types of variables, number of categories, synthesis methods, and available computational resources.

If synthesis is unexpectedly slow, consider first which variables are high-cardinality categorical variables and whether they are being used as predictors for many subsequent variables.

</details>

## Getting help and reporting an issue
If you cannot find an answer in this FAQ or the rest of the documentation, search the
existing [GitHub issues](https://github.com/duonl/synthpop-py/issues) first. Your question or problem may already have been discussed.

If you still need help, you can [open a new GitHub issue](https://github.com/duonl/synthpop-py/issues/new). Please provide as much relevant context as possible, such as:
- what you are trying to achieve;
- the code or configuration you are using;
- what you expected to happen and what happened instead;
- your synthpop-py and Python versions; and
- any relevant error messages or traceback.

For problems involving large datasets, also include the approximate dataset size and why you need to synthesise the dataset at that scale. This helps us understand the use case and improve support for large datasets.

See the [Contributing guide](../developer/contributing_to_package.md) for more detailed guidance on how to submit a GitHub issue.
