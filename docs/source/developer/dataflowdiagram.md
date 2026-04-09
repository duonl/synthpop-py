
# Dataflow diagrams
This page provides a visual overview of how data moves through the Synthpop system. it is intended to help developers understand the internal architecture by showing how inputs are transformed into outputs across different stages of the pipeline.

The diagrams illustrate key processes such as fitting a Synthesiser and generating synthetic data, including both numeric and categorical workflows.

## Overall process data flow
```{mermaid}
flowchart TD
      U(["User"])-->|Step 1| S["Synthesiser()"]
      U-->|Step 2
      x: pd.DataFrame
      y=None|SF["Synthesiser.fit(x)"]-->
      i1{{Loops through x to pick column y each time}}-->CMF["CartMethod.fit(X: pd.DataFrame, y: pd.Series)"]-->icm{{Transform pd.DataFrame and pd.Series to np.Array}}-->
      TCM["TreeClassifierMethod.fit(X: dict, y: np.Array(string))"]-->i2{{"Loops through arrays X to find where X is np.Array(string). Sends this to the encoder."}}-->
      PCAF["PCAEncoder.fit(X: np.Array(string), y: np.Array(string))"]--> PCAT["PCAEncoder.transform(X: np.Array(float))"]-->|"X: np.Array(float) (encoded)"|TCM---->DTC["DecisionTreeClassifier.fit(X (encoded): np.Array(float), y (missing-handled): np.Array(string))
      sklearn"]-->i4
      TCM --> RMV["ReplaceNoneWithValue.prepare_data_for_fit(X: dict, y: np.Array(string))"]-->|"y: np.Array(string) (missing-handled)"|TCM

      icm-->TRM["TreeRegressorMethod.fit(X: dict, y: np.array(float))"]-->i3{{"Loops through arrays X to find where X is np.Array(string). Sends this to the encoder."}}-->MEF["MeanEncoder.fit(X: np.Array(string), y: np.Array(float))"]-->i11{{Save the fitted mean encoder in the TreeRegressorMethod object.}}-->
      MVP["MissingValuePredictor.prepare_data_for_fit(X: np.Array(float), y: np.Array(float))"]
      MET["MeanEncoder.transform(X: np.Array(float))"]-->|"X: np.Array(float) (encoded)"|TRM
      TRM---->DTR["DecisionTreeRegressor.fit(X (encoded): np.Array(float), y (missing-handled): np.Array(float)) 
      sklearn"]-->i4{{"At the end of the fitting phase, two items are saved in the Synthetiser class.
      1. probability distribution of the first column of X
      2. fitted models (decision trees)"}}

      MVP-->i10{{remove rows from X and y where y has a missing value}} -->|"X (with less rows)"|MET
      i10-->|"y (without missing values)"|TRM
      MVP--->i8{{"transforms the target to y(binary), a boolean: missing or not missing"}} -->|"X: np.Array(float), y: np.Array(bool)"| ME3["MeanEncoder.fit_transform(X: np.Array(float), : np.Array(bool))"] -->|"X (encoded): np.Array(float), y (binary): np.Array(float)"| DTC2["DecisionTreeClassifier.fit(X: np.Array(float), y: np.Array(float))"]-->i9{{Save the fitted Decision Tree Classifier in the MissingValuePredictor object.}}


      U----->|Step 3
      n: int|SG["Synthesiser.generate(n: int)"]-->i7{{"Sample with size n from the distribution stored in the Synthesiser object. This becomes X."}}-->CMT["CartMethod.transform(X: dict)"]-->icmt{{Transform pd.DataFrame to np.Arrays}}-->TCMT["TreeClassifierMethod.transform(X: dict)"]-->i5{{"Loops through arrays X to find where X is np.Array(string). Sends this to the encoder"}}-->PCA2["PCAEncoder.transform(X: np.Array(string))"]-->|"X: np.Array(float) (encoded)"|TCMT-->DTCT["DecisionTreeClassifier.predict_proba(X (encoded): np.Array(float))
      sklearn"]----->|"proba: ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1 (returns the predicted class probabilities of the input samples X)"|sampleClass["Sample from that probability distribution"] -->|"X: np.Array(string) (sampled)"|RMV2["ReplaceNoneWithValue.post_synth_transform(X: np.Array(string))"] -->|Newly synthesised column: pd.Series|CMT
      icmt-->TRMT["TreeRegressorMethod.transform(X: dict)"]-->i6{{"Loops through arrays X to find where X is np.Array(string). Sends this to the encoder"}}-->ME2["MeanEncoder.transform(X: np.Array(string))"]-->|"X: np.Array(float) (encoded)"|TRMT-->BDTA["BaseDecisionTree.apply(X (encoded): np.Array(float))
      sklearn"]-------->|"X_leaves: array-like of shape n_samples (returns the index of the leaf that each sample is predicted as)"|sampleReg["sample from the leaf nodes"]-->MVP2["MissingValuePredictor.post_synth_transform(X: np.Array(float))"]-->|Newly synthesised column: pd.Series|CMT -->syndf["Output (synthetic) dataframe: pd.DataFrame"]
```


## Fit flow for numeric target
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

## Fit flow for categorical targets
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

## Generating a numeric column

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

## Generating a categorical column
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

## Abstract diagram
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
