
# Dataflow diagrams
This page provides a visual overview of how data moves through the Synthpop system. it is intended to help developers understand the internal architecture by showing how inputs are transformed into outputs across different stages of the pipeline.

The diagrams illustrate key processes such as fitting a Synthesiser and generating synthetic data, including both numeric and categorical workflows.

## Overall process data flow
The diagrams below show the data flow of using a Synthesiser with the default `CartMethod`. The function {func}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor.fit` is abstracted to one step. For a detailed data flow of the `MissingValuePredictor`, see {ref}`its focus diagram <mvp-diagram>`.
### `Synthesiser.fit()` data flow

<details open>
<summary>Show/hide diagram</summary>

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


      TR --> ENC_R{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit MeanEncoder<br/><br/>If column is numeric:<br/>no encoder"}}
      TC --> ENC_C{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit PCAEncoder<br/><br/>If column is numeric:<br/>no encoder"}}


      ENC_R -->|Categorical X| ME["MeanEncoder.fit(<br/>X: np.ndarray[str],<br/>y: np.ndarray[float32]<br/>)"]
      ENC_C -->|Categorical X| PCA["PCAEncoder.fit(<br/>X: np.ndarray[str]<br/>y: np.ndarray[str]<br/>)"]

      TR --> RMV["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray] <br/>y: np.ndarray[float32]<br/>X and y without rows where y is missing<br/><br/>Fit missingness model"]
      TC --> RV["ReplaceMissingWithValue.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[str]<br/>) <br/><br/>Output:<br/>X: Dict[str, np.ndarray]<br/>y: np.ndarray[str] (missing made a category)<br/><br/>Missing categories in y replaced by marker"]

      RMV -->|Continue with this output| LOOP_R{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>transform with MeanEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}
      RV -->|Continue with this output| LOOP_C{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>transform with PCAEncoder<br/><br/>If column is numeric:<br/>pass through unchanged"}}
      
      
      LOOP_R -->|Categorical X| APPLY_R["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]
      LOOP_C -->|Categorical X| APPLY_C["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]

      ME -->|Fitted encoder stored| APPLY_R
      PCA -->|Fitted encoder stored| APPLY_C

      APPLY_R --> MERGE_R["Recombine transformed columns<br/><br/>X: Dict[str, np.ndarray]"]
      APPLY_C --> MERGE_C["Recombine transformed columns<br/><br/>X: Dict[str, np.ndarray]"]

      LOOP_R -->|Numeric X| MERGE_R
      LOOP_C -->|Numeric X| MERGE_C


      MERGE_R --> FM_R["Convert feature dictionary to tree input matrix<br/><br/>Dict[str, np.ndarray]<br/>→ np.ndarray[float32]<br/><br/>shape: (n_samples, n_features)"]

      MERGE_C --> FM_C["Convert feature dictionary to tree input matrix<br/><br/>Dict[str, np.ndarray]<br/>→ np.ndarray[float32]<br/><br/>shape: (n_samples, n_features)"]


      FM_R --> DTR["DecisionTreeRegressor.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[float32]<br/>)"]

      FM_C --> DTC["DecisionTreeClassifier.fit(<br/>X: np.ndarray[float32],<br/>y: np.ndarray[str]<br/>)"]


      DTR --> DTRA["DecisionTreeRegressor.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>leaf_ids: np.ndarray[int64]"]

      DTC --> DTCA["DecisionTreeClassifier.apply(<br/>X: np.ndarray[float32]<br/>)<br/><br/>leaf_ids: np.ndarray[int64]"]


      DTRA --> LS1["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[float32]<br/>)"]

      DTCA --> LS2["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[str]<br/>)"]


      LS1 --> FIT_R["TreeRegressorMethod fully fitted"]
      LS2 --> FIT_C["TreeClassifierMethod fully fitted"]

      FIT_R --> CART_FIT["CartMethod fully fitted"]
      FIT_C --> CART_FIT

      CART_FIT --> STORE["Store fitted CartMethod in Synthesiser.models_"]

      STORE --> END["Fitted Synthesiser"]
```
</details>

### `Synthesiser.generate()` data flow

<details open>
<summary>Show/hide diagram</summary>

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

      LOOP_R -->|Categorical X| ME["MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]
      LOOP_C -->|Categorical X| PCA["PCAEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]

      ME --> MERGE_R["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]
      PCA --> MERGE_C["Recombine columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray]"]

      LOOP_R -->|Numeric X| MERGE_R
      LOOP_C -->|Numeric X| MERGE_C

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

      ADD -->|Move to next column|L

      L --->|After looping<br/>through all columns| END["Synthetic dataframe<br/><br/>pd.DataFrame"]
```
</details>

(mvp-diagram)=
## Zoomed in: Missing value prediction
Data flows for the {class}`~synthpop.data_processing.missing_value_handling.MissingValuePredictor` class.

### MissingValuePredictor.prepare_data_for_fit() flow

<details>
<summary>Show/hide diagram</summary>

```{mermaid}
---
zoom:
---

flowchart TD

      subgraph input
            FEATURES[("Feature dictionary<br/>X: Dict[str, np.ndarray]<br/>Predictor columns")]
            TARGET[("Target array<br/>y: np.ndarray[float32]<br/>May contain missing values")]
      end


      FEATURES --> MVP["MissingValuePredictor.prepare_data_for_fit(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]

      TARGET --> MVP


      MVP --> Z["Create missingness indicator<br/><br/>z = pd.isna(y)<br/><br/>Output:<br/>z: np.ndarray[bool]<br/><br/>True = missing target"]

      Z --> STATUS{Are there both missing and observed y values?}

      STATUS -->|No| SKIP["Skip tree fitting<br/>"]
      STATUS -->|Yes| LOOP{{"Loop through columns in X<br/><br/>If column is non-numeric:<br/>fit MeanEncoder using z<br/><br/>If column is numeric:<br/>pass through unchanged"}}

      LOOP -->|Categorical X<br/>y = z| ENC["MeanEncoder.fit_transform(<br/>X: np.ndarray[str],<br/>y: np.ndarray[bool]<br/>)<br/><br/>Output:<br/>X: np.ndarray[float32]<br/>(encoded)"]


      LOOP -->|Numeric X| MERGE["Combine encoded and numeric columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray[float32]]"]

      ENC --> MERGE


      MERGE --> MATRIX["`Convert feature dictionary to tree input matrix<br/>*_build_feature_matrix(<br/>X: Dict[str, np.ndarray],<br/>feature_order<br/>)*<br/><br/>Output:<br/>X_matrix: np.ndarray[float32]<br/>shape: (n_samples, n_features)`"]


      MATRIX --> TREE["DecisionTreeClassifier.fit(<br/>X_matrix: np.ndarray[float32],<br/>y: np.ndarray[bool]<br/>)"]

      TREE --> APPLY["DecisionTreeClassifier.apply(<br/>X_matrix: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["LeafNodeSampler.fit_sampler(<br/>leaf_ids: np.ndarray[int64],<br/>y: np.ndarray[bool]<br/>)"]

      SAMPLE -->|Fitted missingness model| FITTED["Fitted MissingValuePredictor<br/><br/>Stores:<br/>- encoders_<br/>- tree_ (if fitted)<br/>- tree_sampler_ (if fitted)<br/>- feature_order_"]
      SKIP --> FITTED

      FITTED --> FILTER["Remove rows where y is missing<br/><br/>mask = ~pd.isna(y)"]


      FILTER --> OUTPUT["Return cleaned data<br/><br/>X_filtered:<br/>Dict[str, np.ndarray]<br/><br/>y_filtered:<br/>np.ndarray[float32]<br/><br/>X and y without rows where y is missing"]
```
</details>

### MissingValuePredictor.post_synth_transform() flow

<details>
<summary>Show/hide diagram</summary>

```{mermaid}
---
zoom:
---

flowchart TD

      subgraph input
            FEATURES[("Synthetic predictor feature dictionary<br/>X: Dict[str, np.ndarray]<br/>Previously synthesised predictor columns")]
            TARGET[("Sampled synthetic target array<br/>y: np.ndarray[float32]<br/>Output from LeafNodeSampler")]
      end


      FEATURES --> MVP["MissingValuePredictor.post_synth_transform(<br/>X: Dict[str, np.ndarray],<br/>y: np.ndarray[float32]<br/>)"]
      TARGET --> MVP

      MVP --> STATUS{"Missingness state<br/>from fitting?"}

      STATUS -->|Mixed: some missing, some observed y values| LOOP{{"Loop through stored feature_order_<br/><br/>If column has stored encoder:<br/>apply MeanEncoder<br/><br/>If numeric:<br/>pass through unchanged"}}

      LOOP -->|Categorical X| ENC["Stored MeanEncoder.transform(<br/>X: np.ndarray[str]<br/>)<br/><br/>Output:<br/>np.ndarray[float32]<br/>(encoded)"]


      LOOP -->|Numeric X| MERGE["Combine encoded and numeric columns<br/><br/>Output:<br/>X: Dict[str, np.ndarray[float32]]"]

      ENC --> MERGE


      MERGE --> MATRIX["`Convert feature dictionary to tree input matrix<br/>*_build_feature_matrix(<br/>X: Dict[str, np.ndarray[float32]]<br/>feature_order_<br/>)*<br/><br/>Output:<br/>X_matrix: np.ndarray[float32]<br/>shape: (n_samples, n_features)`"]


      MATRIX --> APPLY["Stored DecisionTreeClassifier.apply(<br/>X_matrix: np.ndarray[float32]<br/>)<br/><br/>Output:<br/>leaf_ids: np.ndarray[int64]"]


      APPLY --> SAMPLE["Stored LeafNodeSampler.sample_from_leaves(<br/>leaf_ids: np.ndarray[int64]<br/>)<br/><br/>Output:<br/>missing_mask: np.ndarray[bool]<br/><br/>True = set target to missing"]


      SAMPLE --> MASK["Copy sampled target values<br/><br/>y_out = y.copy()"]
      TARGET --> MASK


      MASK --> RESTORE["Apply missingness mask<br/><br/>y_out[missing_mask] = np.nan"]

      STATUS -->|All y was missing| Y_MISSING["Return np.full(<br/>len(y), np.nan<br/>)"] --> OUTPUT
      STATUS -->|No y was missing| Y_OBSERVED("Return y unchanged") --> OUTPUT

      RESTORE --> OUTPUT["Synthetic target with missing values restored<br/><br/>Output:<br/>y: np.ndarray<br/><br/>Observed values preserved<br/>Missing values reintroduced"]
```
</details>

## Zoomed in: CART flows
### Fit flow for a numeric target

<details>
<summary>Show/hide diagram</summary>

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
</details>

### Fit flow for a categorical target

<details>
<summary>Show/hide diagram</summary>

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
</details>

### Generate flow for a numeric column

<details>
<summary>Show/hide diagram</summary>

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
</details>

### Generate flow for categorical column

<details>
<summary>Show/hide diagram</summary>

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
</details>

## Abstract diagram

<details>
<summary>Show/hide diagram</summary>

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
</details>