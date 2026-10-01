# S_pMSE Heatmap Visualisation
## 1. Introduction
This method provides a visual representation of the pairwise relationships between variables based on their S_pMSE values. Its purpose is to enable quick identification of variable combinations whose relationship has been poorly synthesised.

## 2. Input and output
The method requires a dataset in which each record describes a pair of variables together with an associated S_pMSE value. Each record represents the symmetric relationship between two variables. The output is a square heatmap in which all unique variables are displayed on both the horizontal and vertical axes. Each cell in the matrix contains the S_pMSE value for the corresponding variable pair, encoded using a colour scale. The result is a compact and interpretable overview of all pairwise S_pMSE relationships.

## 3. Detailed process
### 3.1 Construction of the symmetric S_pMSE matrix
Based on the input data, a square matrix is constructed in which each cell corresponds to a specific pair of variables. Since the relationship between Variable A and B is identical to that between B and A, the matrix is populated symmetrically. For variable pairs for which no S_pMSE value is available, the corresponding matrix cells remain empty. No imputation or estimation is applied, ensuring that only original relationships are represented.

### 3.2 Categorisation of S_pMSE values
Continuous S_pMSE values are mapped to discrete intervals that represent meaningful divergence levels. Each interval is associated with a fixed colour, enabling immediate visual differentiation between low, moderate and high values. The S_pMSE values are visualised in five bins: ${\text(0,3]}$, ${\text(3,10]}$, ${\text(10,30]}$, ${\text(30,100]}$, ${\text(100,\infty)}$. Cases in which the S_pMSE equals 0 are assigned distinguishable as a constant. This occurs when either a column contains only constant values, or when the synthetic data exactly matches the original data. If a combination of columns is not present in the dataset, the corresponding S_pMSE value is marked as undefined.

The colour scheme shall be sequential, colour-blind friendly, print-friendly, and retain sufficient contrast when reproduced in greyscale. Suitable palettes include the discretised versions of the  `plotly` colour scales *YlOrBr* and *Reds*, or the *iridescent* and *YlOrBr* palettes from the `tol_colors` python package.

### 3.3 Visual encoding and rendering
The matrix is converted into a heatmap where the cells show the S_pMSE values and are coloured according to our predefined groups. Both axes are labelled with variable names and a legend (colour bar) provides a clear mapping between colours and S_pMSE ranges. To improve readability with a large number of variables, tooltips should be added to see which cell represents which variable pair relationship. Optionally, the heatmap can be saved as a static image file. If interactive rendering is enabled, the heatmap is displayed to the active graphical output device. Rendering is optional and context-dependent, and does not affect the saved image output.

## 4. Mathematical properties and constraints
The resulting matrix is always square and symmetric. The visual scale is similar in all plots for easy comparison.

## 5. Edge cases and special situations
### 5.1 Missing values
Missing S_pMSE values for variable pairs are considered undefined. The corresponding heatmap cells shall not display a numeric value and shall be assigned to a separate visual category that is be clearly distinguishable from all bins in the consequential colour scheme defined in Section 3.2, including when reproduced in greyscale.

### 5.2 Execution in headless or non-interactive environments
When the method is executed in a headless environment, interactive rendering is not available or desirable. In such cases the visualisation should be saved to file only, rendering to a display should be disabled or skipped, and the saved image becomes the primary output artefact. File path and folder creation are handled automatically. This ensures that the method remains robust and usable in automated workflows and production environments.

## 6. Limitations and considerations
The method does not provide statistical inference. It is only intended for visual diagnostics and assumes that S_pMSE values are comparable across all variable pairs. For a large number of variables, the heatmap may become less readable.
