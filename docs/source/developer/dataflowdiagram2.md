# more dataflow diagrams

## fit flow for numeric target
```{mermaid}
flowchart TD

subgraph input
obs_features[("observed data features: **DataFrame**")]
obs_target[("observed data target: **Series(numeric)**")]
end
obs_features-->homogenise["split in categorical and numeric features"]

homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]

cat_f-->mean_encoding["mean encoding"]
obs_target-->to_np["convert to numpy array"] -->np_target[("converted target: **np.array(float)**")]
np_target -->mean_encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]

combined_features-->remove_missing["remove rows where target is missing"]-->ft_no_nan[("features with rows removed: **np.array(float)**")]-->fit_tree_reg["fit decision tree regressor"]
np_target-->remove_missing-->tg_no_nan[("target with missing values removed: **np.array(float)**")]-->fit_tree_reg


np_target-->binarize["binarize for missing or not"]

binarize-->missing_target[("**np.Array(bool)** for missing values")]
cat_f-->mean_encoding_missing["mean encode for missing values"]-->encoded_for_missing[("encoded with missing target: **np.array(numeric)**")]-->combine_feat_missing["combine features for missing target"]
num_f-->combine_feat_missing
missing_target-->mean_encoding_missing
combine_feat_missing-->ft_missing[("features for missing: **np.Array(float)**")]
missing_target-->fit_missing["fit tree for missing values"]
ft_missing-->fit_missing
```

## fit flow for categorical targets
```{mermaid}
flowchart TD

subgraph input
obs_features[("observed data features: **DataFrame**")]
obs_target[("observed data target: **Series(categorical)**")]
end
obs_features-->homogenise["split in categorical and numeric features"]

homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]

cat_f-->pca_encoding["pca encoding"]
obs_target-->to_np["convert to numpy array"] -->np_target[("converted target: **np.array(string)**")]-->pca_encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]

combined_features-->fit_tree_cls["fit decision tree classifier"]
np_target-->fill_na["replace missing values with 'N.a.N.'"]-->no_na_target[("target without missing values: **np.Array(string)**")]-->fit_tree_cls
```

## generating categorical column
```{mermaid}
flowchart TD

prev_syn[("previously synthesised data: **Dataframe**")]
prev_syn-->homogenise["split in categorical and numeric features"]
homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]
cat_f-->pca_encoding["pca encoding"]
pca_encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]
combined_features-->sample_from_tree["sample from tree"]-->sample_res[("sampled data: **np.Array(string)**")]
sample_res-->replace_missing["replace 'N.a.N.' with None"]-->new_column[("new synthetic column: **np.Array(string)**")]
new_column --> to_df["convert to series/dataframe"] -->prev_syn
```
## generate numeric column

```{mermaid}
flowchart TD

prev_syn[("previously synthesised data: **Dataframe**")]

prev_syn-->homogenise["split in categorical and numeric features"]
homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]
cat_f-->mean_encoding["mean encoding"]
mean_encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]
combined_features-->sample_from_tree["sample from tree"]-->sample_res[("sampled data: **np.Array(float)**")]

cat_f-->mean_enc_missing["mean encoding for missing values"] --> encoded_missing[("encoded features for missing values: **np.Array(float)**")]
encoded_missing --> combining_features_for_missing["combining features for missing"] --> combined_features_for_missing[("combined features for missing: **np.Array(float)**")]
num_f-->combining_features_for_missing
combined_features_for_missing-->predict_missing["generate missing"]-->missing_mask[("indicator for missing: **np.Array(bool)**")]

sample_res-->replace_missing["remove values where it should be missing"]-->new_column[("new synthetic column: **np.Array(string)**")]
missing_mask-->replace_missing
new_column --> to_df["convert to series/dataframe"] -->prev_syn
```

## abstract diagram:
```{mermaid}
flowchart LR

subgraph input
obs_features[("observed data features: **DataFrame**")]
obs_target[("observed data target: **Series**")]
end
obs_features-->homogenise["split in categorical and numeric features"]

homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]

cat_f-->encoding["encoding"]
obs_target-->to_np["convert to numpy array"] -->np_target[("converted target: **np.array**")]
np_target -->encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]

combined_features-->fit_tree["fit decision tree"]
np_target-->fit_tree

```