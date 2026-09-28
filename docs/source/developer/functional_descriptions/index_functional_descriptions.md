# Functional descriptions

This section describes the functional behaviour of the main components of synthpop-py. It explains what each component does, how it fits into the synthesis process, and the assumptions and constraints that govern its behaviour.

The descriptions are organised according to their role in the synthesis process.

```{mermaid}
---
zoom:
---
flowchart LR

S["synthpop-py"<br/>functional<br/>descriptions]--> SYN[Synthesis]
S--->EVAL[Evaluation]
S--->SUP[Supporting functionality]

SYN--->SYNP["Workflow:<br/><a href='SynthpopSynthesis.html'>synthpop synthesis</a>"]
SYN-->PREP[Data preparation]
SYN-->METH[Synthesis methods]

PREP-->ENC["encoding:<br/><a href='Mean-encoding.html'>Mean encoding</a><br/><a href='PCA-encoding.html'>PCA encoding</a>"]
PREP-->MISS["Missing value handling:<br/><a href='MissingValuePredictor.html'>Missing value predictor</a>"]

METH-->CART["<a href='CART.html'>CART method</a>"]
METH-->COPY["<a href='Copy-method.html'>Copy method</a>"]
METH-->SAMPLE["<a href='Sample-method.html'>Sample method</a>"]

EVAL-->UTIL["Utility metrics:<br/><a href='S_pMSE.html'>Pairwise S_pMSE</a>"]
EVAL-->PLOT["Plotting:<br/><a href='univariate_distributions.html'>Univariate distributions</a><br/><a href='S_pMSE heatmap.html'>Pairwise S_pMSE heatmap</a>"]

SUP-->REPR["<a href='reproducibility.html'>Reproducibility and randomness</a>"]
```

## Synthesis
Components responsible for generating synthetic data.

### Synthesis workflow
- {doc}`SynthpopSynthesis`

### Data preparation
Components that prepare, represent, or transform variables during synthesis.
#### Encoding
- {doc}`Mean-encoding`
- {doc}`PCA-encoding`

#### Missing value handling
- {doc}`MissingValuePredictor`

### Synthesis methods
- {doc}`CART`
- {doc}`Copy-method`
- {doc}`Sample-method`

## Evaluation
Methods used to assess the quality, utility or privacy of generated synthetic data.

### Utility metrics
- {doc}`S_pMSE`

### Plotting
- {doc}`univariate_distributions`
- {doc}`S_pMSE_heatmap`

## Supporting functionality
Functionality that supports synthesis, analysis, or reproducibility.
- {doc}`reproducibility`