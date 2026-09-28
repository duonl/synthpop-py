# Univariate Distribution Comparison
## 1. Introduction
This method visualises the univariate distributions of variables from an original dataset and a corresponding synthetic dataset. Its purpose is to allow a direct comparison between the two datasets for each variable, highlighting differences in distribution and the presence of missing values.

## 2. Input and output
This method requires an original and synthetic dataset with identical column names. The result is a single interactive HTML file in which the distribution of each variable is visualised in a separate plot. All plots are written sequentially under each other in the same document, allowing scrolling and text-based search.

## 3. Detailed process
### 3.1 Iteration over variables
Each variable in the original dataset is processed individually, ensuring that numeric and categorical variables are handled appropriately. For each variable is the number of missing values computed, separately for the original and synthetic datasets. This information is displayed on the plot for diagnostic purposes.

### 3.2 Distribution plotting
For numeric variables, density histograms are generated. In one plot are both the original and synthetic data separately added. This overlap comparison is possible by using transparency. For integer-type variables, histograms are treated as discrete counts. The bins should be the same for the original and synthetic datasets.

For non-numeric variables (such as categorical and character), bar plots are generated. Level frequencies are computed for both datasets and normalised to relative densities. This allows for comparison when the synthetic dataset has a different number of observations than the original data. For each level, a bar plot is made with both datasets shown side-by-side. Levels with missing representation in either dataset are filled with zeroes to maintain comparability.

### 3.3 Visual encoding and rendering
The axes are labelled with variable names and "Density". Titles indicate the variable being compared and the legends differentiate between original and synthetic distributions. The number of missing values per dataset are displayed in a text annotation below the plot. Tooltips will show exact counts of the levels. All plots are rendered into a single HTML document, where each variable corresponds to one vertically stacked plot. The document supports scrolling and browser-based search functionality. If interactive mode is enabled, the HTML file is automatically opened in the default web browser or viewer.

## 4. Mathematical properties and constraints.
Plots are variable-specific and independent such that each variable is evaluated separately. For numeric variables, density normalisation ensures comparability regardless of scale. For non-numeric variables, relate frequencies sum to 1 for each dataset.

## 5. Edge cases and special situations
### 5.1 Missing values
Variables with missing data are still plotted, with the missing value counts annotated. Cells with only missing values result in empty plots but remain valid.

### 5.2 Zero-variance variables
Variables with constant values produce uniform histograms or single-category bars. Differences are still visible if the comparing data varies.

### 5.3 Execution in headless or non-interactive environments
When the method is executed in a headless environment, interactive rendering is not available or desirable. In such cases the visualisation is written to the HTML file only. No graphical device is opened and no plots are rendered to screen. The generated HTML document becomes the primary and sufficient output artefact for downstream inspection. This ensures that the method remains robust and usable in automated workflows and production environments.

## 6. Limitations and considerations
This method does not provide statistical testing or formal similarity metrics. Interpretation relies on visual inspection and relative differences in distributions. Large numbers of variables may result in many plots, requiring appropriate organisation. Comparisons assume consistent data types and column alignment between original and synthetic datasets.

Because all plots are stored in a single HTML document, very large numbers of variables may result in large file sizes and increased browser memory usage.