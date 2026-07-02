# User Guides

This is where we could put our user Guides.

Proposed structure:
## 1. Introduction
- What is synthetic data?
- When should you use this package?
- Main design philosophy
- Basic terminology
    - Original data & synthetic data
    - Synthesiser
    - Synthesis method
    - Utility

## 2. Synthetic Data Generation
This will be the core section and probably the largest.
### 2.1 Overview of the synthesis workflow
A conceptual page showing Original data -> preprocessing -> Synthesiser.fit() -> Synthesiser.generate() (LeafNodeSampler) -> Synthetic data
### 2.2 The Synthesiser Class
Central guide. Topics:
- Creating a synthesiser
- Fitting
- Generating data
- Choosing synthesis methods
- Important parameters
- How preprocessing fits in
- Relationship with LeafNodeSampler
### 2.3 Advanced generation topics
Here could things be like:
- Reproducibility
- Custom synthesis methods
- Performance

## 3 Synthesis methods
One page per method (or a comparison page)
- CART synthesis
- Copy method
- Sample method

Where each shows:
- Intuition
- Algorithm
- Advantages
- Disadvantages
- Recommend use cases
- Example (or link to)

At the end, you'd want to include a comparison table with probably something like Method, Preserves relationships, Speed, Privacy, Typical use.

### 4 Preprocessing
Start with an explanation why preprocessing exists.

Subsections:
- Encoding
    - PCA
    - Mean
- Missing value handling

Each page should answer:
- What problem does this solve?
- When should I use it?
- Example



## 5. Evaluation
### 5.1 Evaluating synthetic data
Explain the difference between utility, privacy and fidelity. Then explain what the current package provides.

### 5.2 Utility metrics
#### 5.2.1 S_pMSE
Topics:
- Intuition
- Mathematical definition
- Interpretation
- Example
- Limitations

### 5.3 Privacy metrics
Empty for now

## 6. Visualisation
Explain what visual inspection is useful for.
### 6.1 Distribution plots
- Comparing original vs. synthetic
- Numerical variables
- Categorical variables
- Examples
### 6.2 S_pMSE heatmap
- Interpretation
- Detecting problematic variables
- Examples


```{toctree}
:maxdepth: 1

1_introduction.md
2_synthetic_data_generation.md
user_guide1.md
user_guide2.md
user_guide3.md
```

