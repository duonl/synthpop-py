
# Dataflow diagrams
This page provides a visual overview of how data moves through the Synthpop system. it is intended to help developers understand the internal architecture by showing how inputs are transformed into outputs across different stages of the pipeline.

The diagrams illustrate key processes such as fitting a Synthesiser and generating synthetic data, including both numeric and categorical workflows.

## Overall process data flow
### `Synthesiser.fit()` data flow
```{mermaid}
flowchart TD
      U(["User"]) -->|X: pd.DataFrame| S["Synthesiser.fit(X: pd.DataFrame)"]

      S --> O["Determine column_order_"]
      O --> L{{"Loop through columns y in column_order_"}}

      L --> P["Create predictors<br/>First column:<br/>pd.DataFrame(init: np.ndarray[int])<br/><br/>Other columns:<br/>pd.DataFrame(previous columns)"]

      L --> M["Select synthesis model<br/>_get_model(y)<br/><br/>CartMethod() or user supplied BaseSynthMethod"]

      P --> CMF["CartMethod.fit(<br/>X: pd.DataFrame,<br/>y: pd.Series<br/>)"]
      M --> CMF

      CMF --> CONV["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]<br/><br/>y:<br/>pd.Series → np.ndarray"]

      CONV --> TYPE{{"Target dtype"}}

      TYPE -->|numeric| TR["TreeRegressorMethod.fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]
      TYPE -->|categorical| TC["TreeClassifierMethod.fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)"]

      TR --> LOOP_R{{"Loop through columns in X<br/>If column is non-numeric,<br/>fit and apply MeanEncoder"}}
      TC --> LOOP_C{{"Loop through columns in X<br/>If column is non-numeric,<br/>fit and apply PCAEncoder"}}

      LOOP_R --> ME["MeanEncoder.fit(<br/>X: np.ndarray[str],<br/>y: np.ndarray[float32]<br/>)"]
      LOOP_C --> PCA["PCAEncoder.fit(<br/>X: np.ndarray[str],<br/>y: np.ndarray[str]<br/>)"]

      ME --> MET["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]"]
      PCA --> PCAT["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]"]

      LOOP_R --> NUM_R["Numeric columns<br/>pass through unchanged"]
      LOOP_C --> NUM_C["Numeric columns<br/>pass through unchanged"]

      MET --> MERGE_R["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]
      PCAT --> MERGE_C["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM_R --> MERGE_R
      NUM_C --> MERGE_C

      MERGE_R --> RMV["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[float32] (without rows with missing values)<br/><br/>Remove missing target rows<br/>Fit missingness model"]
      MERGE_C --> RV["ReplaceMissingWithValue.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[str] (missing handled)<br/><br/>Missing categories replaced by marker"]

      RMV --> FM_R["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]
      RV --> FM_C["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]

      FM_R --> DTR["DecisionTreeRegressor.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[float32]<br/>)"]
      FM_C --> DTC["DecisionTreeClassifier.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[str]<br/>)"]

      DTR --> DTRA["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]
      DTC --> DTCA["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>) <br/><br/></br>Output:<br/>leaf_ids: np.ndarray[int64]"]

      DTRA --> LS1["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[float32]<br/>)"]

      DTCA --> LS2["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[str]<br/>)"]

      LS1 --> FIT_R["TreeRegressorMethod fully fitted"]
      LS2 --> FIT_C["TreeClassifierMethod fully fitted"]
    
      FIT_R --> CART_FIT["CartMethod fully fitted"]
      FIT_C --> CART_FIT

      CART_FIT --> STORE["Store fitted CartMethod in Synthesiser.models_"]

      STORE --> END["Fitted Synthesiser"]
```

### `Synthesiser.generate()` data flow
```{mermaid}
flowchart TD
      U(["User"]) -->|"n: int | None"| S["Synthesiser.generate(n)"]

      S --> CHECK["Check fitted Synthesiser<br/><br/>Load models_ and column_order_"]

      CHECK --> SIZE["Determine number of synthetic rows<br/><br/>n is None:<br/>use n_samples_<br/><br/>otherwise:<br/>use requested n"]

      SIZE --> L{{"Loop through columns y in column_order_"}}

      L --> P["Create predictors<br/>First column:<br/>pd.DataFrame(init: np.ndarray[int])<br/><br/>Other columns:<br/>Synthetic pd.DataFrame generated so far"]

      L --> M["Retrieve fitted CartMethod<br/>from Synthesiser.models_[y]"]

      P --> CMT["CartMethod.transform(<br/>X: pd.DataFrame<br/>)"]
      M --> CMT

      CMT --> CONV["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]"]

      CONV --> TYPE{{"Stored TreeMethod type"}}

      TYPE -->|numeric| TR["TreeRegressorMethod.transform(<br/>X: Dict[str, np.ndarray]<br/>)"]
      TYPE -->|categorical| TC["TreeClassifierMethod.transform(<br/>X: Dict[str, np.ndarray]<br/>)"]

      TR --> LOOP_R{{"Loop through columns in X<br/>If column is non-numeric,<br/>apply MeanEncoder"}}
      TC --> LOOP_C{{"Loop through columns in X<br/>If column is non-numeric,<br/>apply PCAEncoder"}}

      LOOP_R --> ME["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]"]
      LOOP_C --> PCA["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]"]

      LOOP_R --> NUM_R["Numeric columns<br/>pass through unchanged"]
      LOOP_C --> NUM_C["Numeric columns<br/>pass through unchanged"]

      ME --> MERGE_R["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]
      PCA --> MERGE_C["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM_R --> MERGE_R
      NUM_C --> MERGE_C

      MERGE_R --> FM_R["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]
      MERGE_C --> FM_C["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]

      FM_R --> DTR["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]
      FM_C --> DTC["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]

      DTR --> LS_R["LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>y: np.ndarray[float32]"]

      DTC --> LS_C["LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>y: np.ndarray[str]"]

      LS_R --> MVP["MissingValuePredictor.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>y: np.ndarray[float32]<br/><br/>Missing values restored"]

      LS_C --> RMV["ReplaceMissingWithValue.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)<br/><br/>Output:<br/>y: np.ndarray[str]<br/><br/>Missing categories restored"]

      MVP --> SR["Create pd.Series<br/>dtype: float32<br/>name: y"]
      RMV --> SC["Create pd.Series<br/>dtype: str<br/>name: y"]

      SR --> ADD["Add generated column to result pd.DataFrame"]
      SC --> ADD

      ADD --> |Move to next column|L

      L --->|After looping<br/>through all columns| END["Synthetic dataframe<br/><br/>pd.DataFrame"]
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


np_target-->binarise["binarise for missing or not"]

binarise-->missing_target[("**np.Array(bool)** for missing values")]
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
