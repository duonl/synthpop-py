'''mermaid
---
config:
      theme: redux
---
flowchart TD
        U(["User"])-->|Step 1| S["Synthesiser()"]
        U-->|Step 2
        x: pd.DataFrame
        y=None|SF["Synthesiser.fit(x)"]-->
        i1{{Loops through x to pick column y each time}}-->CMF["CartMethod.fit(X: pd.DataFrame, y: pd.Series)"]-->
        TCM["TreeClassifierMethod.fit(X: pd.DataFrame, y: pd.Series)"]-->i2{{Loops through pd.DataFrame X to find categorical X. Sends this as pd.Series to the encoder.}}-->
        PCAF["PCAEncoder.fit(X: pd.Series, y: pd.Series)"]--> PCAT["PCAEncoder.transform(X: pd.Series)"]-->|"X: pd.DataFrame (encoded)"|TCM---->DTC["DecisionTreeClassifier.fit(X (encoded), y(missing handled))
        sklearn"]-->i4
        TCM --> RMV["ReplaceNoneWithValue.prepare_data_for_fit(X,y)"]-->|"y(MissingHandled): pd.Series"|TCM

        CMF-->TRM["TreeRegressorMethod.fit(X: pd.DataFrame, y: pd.Series)"]-->i3{{Loops through pd.DataFrame X to find categorical X. Sends this as pd.Series to the encoder.}}-->MEF["MeanEncoder.fit(X: pd.Series, y: pd.Series)"]-->i11{{Save the fitted mean encoder in the TreeRegressorMethod object.}}-->
        MVP["MissingValuePredictor.prepare_data_for_fit(X,y)"]
        MET["MeanEncoder.transform(X: pd.Series)"]-->|"X: pd.DataFrame (encoded)"|TRM
        TRM---->DTR["DecisionTreeRegressor.fit(X (encoded), y(MissingValueHandled)) 
        sklearn"]-->i4{{"At the end of the fitting phase, two items are saved in the Synthetiser class.
        1. probability distribution of the first column of X
        2. fitted models (decision trees)"}}

        MVP-->i10{{remove rows from X and y where y has a missing value}} -->|"X(with less rows)"|MET
        i10-->|"y(without missing values)"|TRM
        MVP-->i8{{"transforms the target to y(binary), a boolean: missing or not missing"}} -->|"X,y(binary)"| ME3["MeanEncoder.fit_transform(X,y(binary))"] -->|"X(encoded),y(binary)"| DTC2["DecisionTreeClassifier.fit(X,y)"]-->i9{{Save the fitted Decision Tree Classifier in the MissnigValuePredictor object.}}
        
        
        U----->|Step 3
        n: int|SG["Synthesiser.generate(n: int)"]-->i7{{"Sample with size n from the distribution stored in the Synthesiser object. This becomes X."}}-->CMT["CartMethod.transform(X: pd.DataFrame)"]-->TCMT["TreeClassifierMethod.transform(X: pd.DataFrame)"]-->i5{{"Loops through pd.DataFrame X to find categorical X. Sends this as pd.Series to the encoder"}}-->PCA2["PCAEncoder.transform(X: pd.Series)"]-->|"X: pd.DataFrame (encoded)"|TCMT-->DTCT["DecisionTreeClassifier.predict_proba(X (encoded))
        sklearn"]----->|"proba: ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1 (returns the predicted class probabilities of the input samples X)"|sampleClass["Sample from that probability distribution"] --> RMV2["ReplaceNoneWithValue.post_synth_transform"] -->|Newly synthesised column: pd.Series|CMT
        CMT-->TRMT["TreeRegressorMethod.transform(X: pd.DataFrame)"]-->i6{{"Loops through pd.DataFrame X to find categorical X. Sends this as pd.Series to the encoder"}}-->ME2["MeanEncoder.transform(X: pd.Series)"]-->|"X: pd.DataFrame (encoded)"|TRMT-->BDTA["BaseDecisionTree.apply(X (encoded))
        sklearn"]-------->|"X_leaves: array-like of shape n_samples (returns the index of the leaf that each sample is predicted as)"|sampleReg["sample from the leaf nodes"]-->MVP2["MissingValuePredictor.post_synth_transform"]-->|Newly synthesised column: pd.Series|CMT -->syndf["Output (synthetic) dataframe: pd.DataFrame"]

       

'''

bij missing value predictor output pijlen (binary)