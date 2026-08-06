
# Dataflow diagrams
This page provides a visual overview of how data moves through the Synthpop system. it is intended to help developers understand the internal architecture by showing how inputs are transformed into outputs across different stages of the pipeline.

The diagrams illustrate key processes such as fitting a Synthesiser and generating synthetic data, including both numeric and categorical workflows.

## Overall process data flow
The diagrams below show the data flow of using a Synthesiser with the default `CartMethod`.
### `Synthesiser.fit()` data flow
```{mermaid}
---
zoom:
---
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

      ME --> MET["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]
      PCA --> PCAT["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]

      LOOP_R --> NUM_R["Numeric columns<br/>pass through unchanged"]
      LOOP_C --> NUM_C["Numeric columns<br/>pass through unchanged"]

      MET --> MERGE_R["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]
      PCAT --> MERGE_C["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM_R --> MERGE_R
      NUM_C --> MERGE_C

      MERGE_R --> RMV["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[float32] (without rows with missing values)<br/><br/>Remove missing target rows<br/>Fit missingness model"]
      MERGE_C --> RV["ReplaceMissingWithValue.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[str] (missing made a category)<br/><br/>Missing categories replaced by marker"]

      RMV --> FM_R["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]
      RV --> FM_C["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]

      FM_R --> DTR["DecisionTreeRegressor.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[float32]<br/>)"]
      FM_C --> DTC["DecisionTreeClassifier.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[str]<br/>)"]

      DTR --> DTRA["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]
      DTC --> DTCA["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>) <br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]

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
---
zoom:
---

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

      LOOP_R --> ME["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]
      LOOP_C --> PCA["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]

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

      LS_R --> MVP["MissingValuePredictor.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>y: np.ndarray[float32]<br/><br/>Predict which values should be missing"]

      LS_C --> RMV["ReplaceMissingWithValue.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)<br/><br/>Output:<br/>y: np.ndarray[str]<br/><br/>Missing categories restored"]

      MVP --> SR["Create pd.Series<br/>dtype: float32<br/>name: y"]
      RMV --> SC["Create pd.Series<br/>dtype: str<br/>name: y"]

      SR --> ADD["Add generated column to result pd.DataFrame"]
      SC --> ADD

      ADD --> |Move to next column|L

      L --->|After looping<br/>through all columns| END["Synthetic dataframe<br/><br/>pd.DataFrame"]
```
## Zoomed in: Missing value prediction
Data flows for the {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` class.

### Prepare data for fit flow
```{mermaid}
---
zoom:
---

flowchart LR

      subgraph input
            FEATURES[("Feature dictionary<br/><br/>X: Dict[str, np.ndarray}<br/><br/>Predictor columns")]
            TARGET[("Target array<br/><br/>y: np.ndarray[float32]<br/><br/>Contains np.nan values")]
      end


      FEATURES --> MVP["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]

      TARGET --> MVP


      MVP --> Z["Create missingness indicator<br/><br/>z = pd.isna(y)<br/><br/>Output:<br/>z: np.ndarray[bool]<br/><br/>True = missing target"]


      MVP --> LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit MeanEncoder using z<br/><br/>If column is numeric:<br/>pass through unchanged"}}


      LOOP --> CAT["Categorical feature column<br/><br/>X: np.ndarray[str]"]

      LOOP --> NUM["Numeric feature column<br/><br/>X: np.ndarray[float32]"]


      CAT --> ENC["MeanEncoder.fit_transform(<br/>X: np.ndarray[str],<br/>y: np.ndarray[bool]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]"]


      NUM --> MERGE["Combine encoded and numeric columns<br/><br/>Output:<br/>X_encoded: Dict[str, np.ndarray]"]

      ENC --> MERGE


      MERGE --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      Z --> TREE["DecisionTreeClassifier.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[bool]<br/>)"]

      MATRIX --> TREE


      TREE --> APPLY["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[bool]<br/>)"]


      SAMPLE --> FITTED["Fitted MissingValuePredictor<br/><br/>Stores:<br/>- encoders_<br/>- tree_<br/>- tree_sampler_<br/>- feature_order_"]


      FITTED --> FILTER["Remove rows where y is missing<br/><br/>mask = ~pd.isna(y)"]


      FILTER --> OUTPUT["Return cleaned data<br/><br/>X_filtered:<br/>Dict[str, np.ndarray]<br/><br/>y_filtered:<br/>np.ndarray[float32]<br/><br/>Only observed target values"]
```

### Post synthesis transform flow
```{mermaid}
---
zoom:
---

flowchart LR

      subgraph input
            FEATURES[("Synthetic predictor features<br/><br/>X: Dict[str, np.ndarray}<br/><br/>Previously synthesised columns")]
            TARGET[("Sampled synthetic target<br/><br/>y: np.ndarray[float32]<br/><br/>Output from LeafNodeSampler")]
      end


      FEATURES --> MVP["MissingValuePredictor.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]

      TARGET --> MVP


      MVP --> LOOP{{"Loop through columns in X<br/><br/>If column has stored encoder:<br/>apply encoder.transform()<br/><br/>If numeric:<br/>pass through unchanged"}}


      LOOP --> CAT["Categorical feature column<br/><br/>X: np.ndarray[str]"]

      LOOP --> NUM["Numeric feature column<br/><br/>X: np.ndarray[float32]"]


      CAT --> ENC["Stored MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]


      NUM --> MERGE["Combine encoded and numeric columns<br/><br/>Output:<br/>X_encoded: Dict[str, np.ndarray]"]

      ENC --> MERGE


      MERGE --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      MATRIX --> APPLY["Stored DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["Stored LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>missing_mask: np.ndarray[bool]<br/><br/>True = set target to missing"]


      TARGET --> MASK["Copy sampled target values<br/><br/>y_out = y.copy()"]

      SAMPLE --> MASK


      MASK --> RESTORE["Apply missingness mask<br/><br/>y_out[missing_mask] = np.nan"]


      RESTORE --> OUTPUT["Synthetic target with missing values restored<br/><br/>Output:<br/>np.ndarray[float32]<br/><br/>Observed values preserved<br/>Missing values reintroduced"]
```

## Zoomed in: CART flows
### Fit flow for a numeric target
```{mermaid}
---
zoom:
---

flowchart LR
      subgraph input
            FEATURES[("Observed data features<br/><br/>X: pd.DataFrame<br/><br/>Predictor columns")]
      
            TARGET[("Observed target column<br/><br/>y: pd.Series<br/><br/>Numeric target to synthesise")]
      end


      FEATURES --> CART["CartMethod.fit(<br/>X: pd.DataFrame,<br/>y: pd.Series<br/>)"]
      TARGET --> CART

      CART --> CONVERT["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]<br/><br/>y:<br/>pd.Series → np.ndarray"]

      CONVERT --> TR["TreeRegressorMethod.fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]


      TR --> LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit and apply MeanEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}

      LOOP --> CAT["Categorical feature column<br/><br/>Input:<br/>np.ndarray[str]"]

      LOOP --> NUM["Numeric feature column<br/><br/>Input:<br/>np.ndarray[float32]"]


      CAT --> ME["MeanEncoder.fit(<br/>X: np.ndarray[str],<br/>y: np.ndarray[float32]<br/>)"]

      ME --> MET["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]

      MET --> MERGE["Recombine transformed columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM --> MERGE


      MERGE --> MV["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[float32] (without rows with missing values)<br/><br/>Operations:<br/>- identify missing target values<br/>- fit missingness decision tree<br/>- remove missing target rows"]


      MV --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      MATRIX --> TREE["DecisionTreeRegressor.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[float32]<br/>)"]


      TREE --> APPLY["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[float32]<br/>)"]


      SAMPLE --> DONE["Fully fitted numeric CART model<br/><br/>TreeRegressorMethod → CartMethod"]
```

### Fit flow for a categorical target
```{mermaid}
---
zoom:
---

flowchart LR
      subgraph input
            FEATURES[("Observed data features<br/><br/>X: pd.DataFrame<br/><br/>Predictor columns")]

            TARGET[("Observed target column<br/><br/>y: pd.Series<br/><br/>Categorical target to synthesise")]
      end


      FEATURES --> CART["CartMethod.fit(<br/>X: pd.DataFrame,<br/>y: pd.Series<br/>)"]

      TARGET --> CART


      CART --> CONVERT["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]<br/><br/>y:<br/>pd.Series → np.ndarray"]


      CONVERT --> TREE_METHOD["TreeClassifierMethod.fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)"]


      TREE_METHOD --> LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit and apply PCAEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}


      LOOP --> CAT["Categorical feature<br/><br/>Input:<br/>np.ndarray[str]"]

      LOOP --> NUM["Numeric feature<br/><br/>Input:<br/>np.ndarray[float32]"]


      CAT --> ENCODER["PCAEncoder.fit(<br/>X: np.ndarray[str],<br/>y: np.ndarray[str]<br/>)"]

      ENCODER --> ENCODED["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]


      ENCODED --> MERGE["Recombine feature columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM --> MERGE


      MERGE --> MISSING["ReplaceMissingWithValue.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[str] (missing made as category)<br/><br/>Operations:<br/>- replace missing target values<br/>- use missing marker: 'N.a.N.'"]


      MISSING --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      MATRIX --> DT["DecisionTreeClassifier.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[str]<br/>)"]


      DT --> LEAVES["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      LEAVES --> SAMPLER["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[str]<br/>)"]


      SAMPLER --> DONE["Fully fitted categorical CART model<br/><br/>TreeClassifierMethod → CartMethod"]
```

### Generate flow for a numeric column

```{mermaid}
---
zoom:
---

flowchart LR
      subgraph input
            FEATURES[("Previously synthesised data<br/><br/>X: pd.DataFrame<br/><br/>Previously generated columns")] 
      end

      FEATURES --> CART["CartMethod.transform(<br/>X: pd.DataFrame<br/>)"]


      CART --> CONVERT["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]"]


      CONVERT --> TREE_METHOD["TreeRegressorMethod.transform(<br/>X: Dict[str, np.ndarray]<br/>)"]


      TREE_METHOD --> LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit and apply MeanEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}


      LOOP --> CAT["Categorical feature<br/><br/>Input:<br/>np.ndarray[str]"]

      LOOP --> NUM["Numeric feature<br/><br/>Input:<br/>np.ndarray[float32]"]


      CAT --> ENCODER["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]


      ENCODER --> MERGE["Recombine transformed columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM --> MERGE


      MERGE --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      MATRIX --> APPLY["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/><br/>Sample target values from leaf distributions"]


      SAMPLE --> MISSING["MissingValuePredictor.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/><br/>Restore missing numeric values"]


      MISSING --> SERIES["Create synthetic column<br/><br/>pd.Series<br/>dtype: float32<br/>name: target column"]


      SERIES --> DONE["TreeRegressorMethod.transform() completed<br/>CartMethod.transform() completed<br/><br/>Add column to synthetic dataset"]
```

### Generate flow for categorical column
```{mermaid}
---
zoom:
---

flowchart LR
      subgraph input
            FEATURES[("Previously synthesised data<br/><br/>X: pd.DataFrame<br/><br/>Previously generated columns")] 
      end

      FEATURES --> CART["CartMethod.transform(<br/>X: pd.DataFrame<br/>)"]


      CART --> CONVERT["Convert data representation<br/><br/>X:<br/>pd.DataFrame → Dict[str, np.ndarray]"]


      CONVERT --> TREE_METHOD["TreeClassifierMethod.transform(<br/>X: Dict[str, np.ndarray]<br/>)"]


      TREE_METHOD --> LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>apply stored PCAEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}


      LOOP --> CAT["Categorical feature<br/><br/>Input:<br/>np.ndarray[str]"]

      LOOP --> NUM["Numeric feature<br/><br/>Input:<br/>np.ndarray[float32]"]


      CAT --> ENCODER["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]


      ENCODER --> MERGE["Recombine transformed columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      NUM --> MERGE


      MERGE --> MATRIX["Convert feature dictionary to tree input matrix<br/><br/>Input:<br/>Dict[str, np.ndarray]<br/><br/>Output:<br/>np.ndarray[float32]<br/>shape: (n_samples, n_features)"]


      MATRIX --> APPLY["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>np.ndarray[str]<br/><br/>Sample target categories from leaf distributions"]


      SAMPLE --> MISSING["ReplaceMissingWithValue.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[str]<br/><br/>Restore missing categorical values"]


      MISSING --> SERIES["Create synthetic column<br/><br/>pd.Series<br/>dtype: str<br/>name: target column"]


      SERIES --> DONE["TreeClassifierMethod.transform() completed<br/>CartMethod.transform() completed<br/><br/>Return generated column"]
```

## Abstract diagram
```{mermaid}
---
zoom:
---
flowchart LR

subgraph input
FEATURES[("Observed data features:<br/>X: pd.DataFrame<br/>Predictor columns")]
TARGET[("Observed data target:<br/>y: pd.Series<br/>Column to synthesise")]
end
FEATURES-->homogenise["Split in categorical and numeric features"]

homogenise-->cat_f[("Categorical features in X:<br/>np.Array(string)")]
homogenise-->num_f[("Numeric features in X:<br/>np.Array(float)")]

cat_f-->encoding["Encoding"]
TARGET-->to_np["Convert to numpy array"] --> np_target[("Converted target y:<br/>np.array")]
np_target --> encoding --> encoded_features[("Encoded features in X:<br/>np.Array(float)")]--> combining_features["Combining features"]
num_f -->combining_features-->combined_features[("Combined features X:<br/>np.Array(float)")]

combined_features-->fit_tree["Fit decision tree"]
np_target-->fit_tree

```
