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
