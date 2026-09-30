# Frequently Asked Questions and Troubleshooting

This page collects common questions and issues that may arise when using synthpop-py. It will be expanded as new questions, issues and workarounds are identified.

## CART synthesis
<details>
<summary> Why am I seeing a rare-category warning?</summary>

### Why am I seeing a rare-category warning?
You may see a warning such as:

> Categorical predictor contains categories occurring fewer than 5 times for more than 25% of the rows.

This warning means that a categorical predictor used by {class}`~synthpop.methods.cart_synth.CartMethod` contains at least 25% of observations belonging to **rare categories**. By default, a category is considered rare when it occurs fewer than `rare_categories_threshold` times. The default value of `rare_categories_threshold` is 5.

Rare categories can increase the risk of {ref}`unintended attribute disclosure <6122-rare-categories>`. When a category contains only a few observations, CART may create a decision-tree group containing only those observations. If the target values in that group are also unique or uncommon, the synthetic data may reproduce information about individuals in that group more closely than intended. 

The warning is therefore a privacy safeguard. It does not stop synthesis; synthesis continues after the warning. However, you should carefully consider the potential privacy risk and evaluate the privacy protection of your synthetic dataset.

<p class="fake-h3"><strong>What can I do?</strong></p>

First, consider whether the rare categories are appropriate for use as a predictor and whether the associated privacy risk is acceptable. You can adjust the `rare_categories_threshold` using {func}`~synthpop.methods.cart_synth.tune_cart`. For example:
```python
tuned_cart = tune_cart(rare_categories_threshold=10)

synthesiser = Synthesiser(default_syn_method=tuned_cart)
```
This sets the threshold to `10`, so categories occurring fewer than 10 times are considered rare. A higher threshold classifies more categories as rare, while a lower threshold classifies fewer categories as rare.

For a complete example showing how to configure the rare-category check, see [Example: Use the `tune_cart` function](../examples/tune_cart_function.md).

If you want to disable the check entirely, set `rare_categories_threshold=0`:
```python
tuned_cart = tune_cart(rare_categories_threshold=0)

synthesiser = Synthesiser(default_syn_method=tuned_cart)
```
Disabling the check only suppresses the warning; it does not remove the underlying privacy risk. See the {ref}`User Guide: Rare categories and overfitting <6122-rare-categories>` and [Example: Risk of privacy loss due to rare categories](../examples/rare_categories.md) for more information. 

The rare-category check can also be configured when {ref}`configuring CART directly <314-configuring-cart>`. See [Example: Configure CART directly](../examples/configure_cart_directly.md) to see how.

</details>

## Performance
<details>
<summary> Why is my synthesis taking so long? </summary>

### Why is my synthesis taking so long?

Synthesis time depends on both the size and structure of your dataset. In general, synthetic data generation takes longer as the number of rows and variables increases. Some specific types of variables can also make a synthesis considerably more computationally intensive.
<br>

One such example is a categorical variable with many possible values (high cardinality). This can make decision-tree fitting more computationally intensive because the tree needs to evaluate many possible splits, such as dividing observations according to different categories or groups of categories. synthpop-py synthesises variables sequentially, using previously synthesised variables as predictors. This means that a variable with many categories can affect the computational cost of not only its own synthesis, but also the synthesis of variables that use it as a predictor.

The default {class}`~synthpop.methods.cart_synth.CartMethod` performs preprocessing of categorical data before fitting the decision tree. In particular, principal component analysis (PCA) can be used to represent categorical data with a smaller number of components. When a categorical predictor has many possible values, this preprocessing and the subsequent tree fitting can become computationally expensive. If that variable is then used as a predictor for several later variables, this computational cost can occur repeatedly during the sequential synthesis process.

<p class="fake-h3"><strong>What can I do?</strong></p>

**Consider whether the high-cardinality variable needs to be used as a categorical predictor.**<br>
First, consider whether the predictor-target relationship makes sense for the variables involved. If a categorical variable has a very large number of possible values, ask whether it is appropriate to use it as a predictor for the target variable. Depending on the meaning of the variables, another representation or synthesis strategy may be more appropriate. For instance, by changing the synthesis order or synthesising in strata. More on this below.

**Consider creating a helper variable.**<br>
If a high-cardinality variable is useful for capturing a relationship but is too expensive to use directly as a predictor, consider whether a lower-cardinality or numerical summary can capture some of the same information. Whether this is appropriate depends on the relationships you want to preserve and the meaning of the variables. A derived variable may capture some information from the original variable while avoiding the computational cost of using the high-cardinality categorical variable directly.

**Reduce the number of principal components.**<br>
CART implements a principal component method to process categorical data. By default, synthpop-py retains all principal components, but users can reduce the dimensionality. Reducing the number of components can reduce the computational cost of this preprocessing and the subsequent modelling. However, using fewer components can also remove information from the representation of the categorical data, so this is a trade-off between computational cost and the information available to the model. See [Example: Use the `tune_cart` function](../examples/tune_cart_function.md) to learn how to choose a different number of principal components.

**Consider changing the synthesis order.**<br>
synthpop-py uses previously synthesised variables as predictors for the target column. Therefore, the synthesis order determines which variables are available as predictors for each subsequent variable. If a high-cardinality variable is placed early in the synthesis column order, it will be used as a predictor for many later variables. Moving it later in the synthesis order means that it will not be available as a predictor for those earlier variables, which can reduce the amount of computation required for subsequent models. However, moving the variable to the end does not necessarily make the synthesis of that variable itself faster. 

Changing the synthesis order can also affect the statistical utility of the synthetic data. The order should therefore be chosen based on both computational considerations and the relationships that you want to preserve. See {ref}`User Guide 2.2.4: Changing the column order <224-column-order>` and [Example: Change the synthesis order](../examples/changing_the_synthesis_order.md) for more information.

**Consider synthesising the data in strata.**<br>
For a large dataset, you can consider dividing the data into meaningful strata, synthesising each subset separately, and recombining the resulting synthetic data afterwards. Working with smaller subsets can reduce the computational and memory requirements of each individual synthesis run.

This approach is most appropriate when the strata have a meaningful interpretation and it is reasonable to model them separately. For example, you might synthesise separate groups defined by a variable such as region or another structural characteristic, provided that the relationships between the strata do not need to be modelled jointly.

When using this approach, consider how the strata are defined and how many synthetic observations should be generated for each stratum. Synthesising strata independently means that relationships between variables across strata are not modelled by the same synthesis process.

</details>

## Getting help and reporting an issue
If you cannot find an answer in this FAQ or the rest of the documentation, search the
existing [GitHub issues](https://github.com/duonl/synthpop-py/issues) first. Your question or problem may already have been discussed.

If you still need help, you can [open a new GitHub issue](https://github.com/duonl/synthpop-py/issues/new). Please provide as much relevant context as possible, such as:
- what you are trying to achieve;
- the characteristics of your dataset, including the number of rows and variables, data types and cardinality;
- the code or configuration you are using;
- what you expected to happen and what happened instead;
- your synthpop-py and Python versions; and
- any relevant error messages or traceback.

See the [Contributing guide](../developer/contributing_to_package.md) for more detailed guidance on how to submit a GitHub issue.
