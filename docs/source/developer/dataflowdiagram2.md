```{mermaid}
flowchart TD
obs_features[("observed data features: **DataFrame**")] -->homogenise["split in categorical and numeric features"]
obs_target[("observed data target: **Series**")] --> RorC["differentiate in regression or classification"]
homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]
RorC-->num_target[("numeric target: **np.Array(float)**")]
RorC-->cat_target[("categorical target: **np.Array(string)**")]

cat_f-->mean_encoding["mean encoding"]
num_target -->mean_encoding --> encoded_features[("encoded features: **np.Array(float)**")]
cat_f-->pca_encoding["pca encoding"]
cat_target-->pca_encoding --> encoded_features --> combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]

combined_features-->fit_tree_cls["fit decision tree classifier"]
cat_target-->fit_tree_cls

combined_features-->fit_tree_reg["fit decision tree regressor"]
num_target-->fit_tree_reg

```
## flow for numeric target
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

## flow for categorical targets
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

## unified diagram:
```{mermaid}
flowchart TD

subgraph numeric_target
subgraph input
obs_features[("observed data features: **DataFrame**")]
obs_target[("observed data target: **Series(numeric)**")]
end
obs_features-->homogenise["split in categorical and numeric features"]

homogenise-->cat_f[("Categorical features: **np.Array(string)**")]
homogenise-->num_f[("Numeric features: **np.Array(float)**")]

cat_f-->mean_encoding["mean encoding"]
obs_target-->to_np["convert to numpy array"] -->np_target[("converted target: **np.array(string)**")]
np_target -->mean_encoding --> encoded_features[("encoded features: **np.Array(float)**")]-->combining_features["combining features"]
num_f -->combining_features-->combined_features[("combined features: **np.Array(float)**")]

combined_features-->fit_tree_reg["fit decision tree regressor"]
np_target-->fit_tree_reg
end

subgraph categorical_target
subgraph input_cat_target
cat_obs_features[("observed data features: **DataFrame**")]
cat_obs_target[("observed data target: **Series(numeric)**")]
end
cat_obs_features-->cat_homogenise["split in categorical and numeric features"]

cat_homogenise-->cat_cat_f[("Categorical features: **np.Array(string)**")]
cat_homogenise-->cat_num_f[("Numeric features: **np.Array(float)**")]

cat_cat_f-->pca_encoding["pca encoding"]
cat_obs_target-->cat_to_np["convert to numpy array"] -->cat_np_target[("converted target: **np.array(string)**")]-->pca_encoding --> cat_encoded_features[("encoded features: **np.Array(float)**")]-->cat_combining_features["combining features"]
cat_num_f -->cat_combining_features-->cat_combined_features[("combined features: **np.Array(float)**")]

cat_combined_features-->fit_tree_cls["fit decision tree classifier"]
cat_np_target-->fit_tree_cls
end
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