# 6. Evaluating privacy
Synthetic data is designed to preserve useful statistical properties of the original data while reducing the risk of disclosing information about individuals. Privacy evaluation assesses whether the synthetic dataset could reveal sensitive information about individuals represented in the original dataset.

Privacy and utility are inherently connected. Increasing the similarity between synthetic and original data can improve analytical usefulness, but may also increase privacy risks. Conversely, stronger privacy protection may require reducing the amount of information retained from the original data.

There is therefore no universally optimal balance between utility and privacy. The appropriate trade-off depends on the intended use of the synthetic data, the sensitivity of the information and the acceptable level of disclosure risk.

Privacy evaluation should be considered alongside utility evaluation. A synthetic dataset that has high utility but insufficient privacy protection may not be suitable for release, while a highly private dataset with very low utility may not be useful for analysis.

synthpop-py currently provides the framework for synthetic data generation but does not yet include automated privacy risk metrics. This guide describes common privacy risks that should be considered when evaluating synthetic data.

---
## 6.1. Privacy risks
Privacy risks in synthetic data are commonly divided into three main categories[^1]:

[^1]: Yu, Z.S. (2025). *Synthetic Data Disclosure Risk Assessment: A Literature Review*. In Proceedings of Statistics Canada Symposium 2024: The Future of Official Statistics, Statistics Canada International Symposium Series: Proceedings, 2025001. https://www150.statcan.gc.ca/n1/pub/11-522-x/2025001/article/00016-eng.pdf

| Risk type | Main question | Information revealed
| --- | --- | --- |
| Identity disclosure | "Can I identify this person in the dataset?"| The person's record or attributes can be linked to their identity |
| Attribute disclosure | "What sensitive information can I infer about this person?" | Sensitive characteristics or values |
| Membership disclosure | "Was this person included in the dataset?" | The person's participation in the original dataset |

These risks describe different ways in which information about individuals in the original dataset may be inferred from the synthetic data.

### 6.1.1. Identity disclosure
Identity disclosure occurs when an individual in the original dataset can be linked to a record in the synthetic dataset.

This risk is related to the ability to identify a person based on combinations of characteristics. For example, a synthetic record containing a rare combination of age, occupation, location and other attributes could allow a third-party to infer which original individual it represents.

Identity disclosure is more likely when:
- The original dataset contains unique or rare combinations of variables.
- The synthesis method overfits the original data so that the synthetic dataset reproduces rare or individual-level patterns exactly.
- Many quasi-identifying variables are released.
- The synthetic generation process memorises individual records.

A common approach for evaluating identity disclosure is to measure how closely synthetic records match original records based on identifying attributes.

(612-attribute-disclosure)=
### 6.1.2. Attribute disclosure
Attribute disclosure occurs when a third-party learns sensitive information about an individual, even without identifying the exact record.

A sensitive attribute is an attribute whose value is considered confidential and could cause harm or reveal private information if inferred about an individual. Examples include medical diagnoses, income, political affiliation or other personal characteristics. In the context of attribute disclosure, the concern is that a third-party who already knows an individual's quasi-identifying characteristics (such as age, sex or location) may use the synthetic data to infer the value of one or more sensitive attributes with high confidence.

For example, if an individual is known to belong to a particular group and the synthetic data reveal that all individuals in that group share a sensitive attribute, a third-party may infer information about the individual.

Attribute disclosure can occur when:
-  Relationships between quasi-identifiers and sensitive attributes are preserved too strongly.
- Rare groups have little variation in sensitive attributes.
- The synthetic data reveal deterministic relationships from the original data.

Preventing attribute disclosure requires considering not only whether individual records can be identified, but also whether sensitive information can be inferred from preserved statistical relationships.

#### 6.1.2.1 Rare categories and overfitting

Categories with very few records in the data, so-called rare categories, can create additional privacy risks when generating synthetic data. When a categorical variable contains rare or unique values, synthesis models may reproduce relationships involving those values too accurately. This is particularly relevant when a rare category is associated with a sensitive attributbe, as preserving that relationship may enable attribute disclosure.

This can occur when a synthesis model overfits the original data. For example, suppose a categorical predictor contains a unique value for every observation. A classification model may be able to split the data into groups that are highly homogeneous with respect to the target, including groups containing relatively few observations. When synthetic values are subsequently sampled from these groups, there may be little or no randomness in the generated values. The relationship between the predictor and target may therefore be reproduced very closely, causing the synthetic target to closely reflect the original target.

For example, consider a dataset with 100 observations where a predictor has a unique value for every observation and the target has 10 possible values. CART could create 10 leaves containing approximately 10 observations each, even with a minimum leaf size of 5. If each leaf corresponds predominantly to one target value, the predictor can still provide substantial information about the target despite the relatively large leaf size. With 200 observations, the same structure could result in leaves containing approximately 20 observations each. Increasing the minimum leaf size may therefore have little effect when the underlying predictor-target relationship is already highly informative.

This illustrates an important distinction between **model overfitting** and **relationships that are inherently highly predictive in the original data**. Increasing the minimum leaf size can reduce some forms of overfitting by preventing the model from creating very small groups. However, it cannot necessarily remove privacy risk when the source data itself contains a strong or deterministic relationship between a predictor and a sensitive target. In such cases, preserving the relationship accurately may inherently limit the extent to which privacy can be protected while retaining the utility of the data.

To reduce risks arising from overfitting, we recommend considering CART with a minimum number of observations per leaf node. Increasing the minimum leaf size can prevent the model from creating very small groups based on rare or unique values, thereby forcing observations that would otherwise be separated into small groups to be modelled together. This can introduce greater variability when synthetic values are generated. The effectiveness from this approach depends on the structure of the data, however, and it does not by itself guarantee protection against attribute disclosure. See {ref}`Guide 3.1.4: Configuring CART <314-configuring-cart>`  and [Example: The tune_cart function](../examples/tune_cart_function.md) for information on how to adjust the minimum leaf size parameter.

The effect of increasing the minimum leaf size is generally greater in datasets where categorical predictors contain repeated values and relationships between predictors and sensitive attributes are not deterministic. In these circumstances, a larger leaf size can combine observations that would otherwise form small, highly homogeneous groups, reducing the extent to which the synthesis model can reproduce relationships for rare categories. The appropriate value should therefore be selected based on the structure of the data, balancing privacy protection against the ability of the model to preserve meaningful relationships.

Rare categories are particularly important when they identify small groups of individuals. For example, suppose a categorical variable identifies a small group and another variable describes a sensitive characteristic of individuals in that group. If the synthesis model learns a highly deterministic relationship between the two variables, the synthetic data may reproduce that relationship even though the individual records themselves are not directly copied. A person with external knowledge about the group could then use the synthetic data to infer the sensitive characteristic of members of that group. This is a form of attribute disclosure.

More generally, the risk of attribute disclosure can be increased by:

* categorical variables with many rare or unique categories;
* small subgroups with little variation in sensitive attributes;
* strong or deterministic relationships between quasi-identifiers and sensitive variables;
* synthesis models that overfit small groups; and
* synthesis methods that reproduce original relationships with little or no added uncertainty.

The presence and characteristics of rare categories should therefore be carefully assessed. In particular, users should examine whether small or rare groups are represented in the synthetic data and whether sensitive attributes associated with those groups are preserved in a way that could increase disclosure risk. Where a predictor is highly informative about a sensitive attribute, users should consider whether preserving that relationship is compatible with the intended privacy protection.

This risk is not specific to any particular synthesis method. It depends on the interaction between the characteristics of the original data, the synthesis model, and the specific synthesis configuration. A model that captures common categories well may still overfit rare categories or small subgroups. Conversely, increasing model constraints may reduce overfitting without eliminating disclosure risks arising from relationships that are already strongly present in the source data.

A synthetic dataset should therefore not be assumed to provide strong privacy protection simply because individual records are not explicitly copied. Privacy evaluation should also consider whether the synthesis process has preserved relationships within rare groups in a way that could enable inference of sensitive information.

For a worked example demonstrating how rare or unique categories can cause a synthesis model to overfit and reproduce sensitive attributes, see [Example: Risk of privacy loss due to rare categories](../examples/rare_categories.md).

### 6.1.3. Membership disclosure
Membership disclosure occurs when a third-party can determine whether a particular individual was included in the original dataset.

For example, if a person suspects that their medical records were included in a dataset, synthetic data should not allow a third party to confidently confirm or deny their presence.

Unlike identity disclosure, membership disclosure does not require identifying a record of learning any attributes about the individual. The third party only learns that the person's data contributed to the original dataset.

Factors that can increase membership disclosure include
- Small original datasets.
- Rare individuals or rare combinations of characteristics.
- Overfitting during synthesis.
- Synthetic records that are too close to the original records.

Membership disclosure is particularly relevant when synthetic data are released publicly, because third parties may combine the synthetic dataset with external information.
