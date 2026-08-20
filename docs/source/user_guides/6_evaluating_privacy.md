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

Categories with very few records in the data, so-called rare categories, can create additional privacy risks when generating synthetic data. When a categorical variable contains rare or unique values, synthesis models may be able to reproduce relationships for those rare values too accurately.

This can occur when a synthesis model overfits the original data. For example, suppose a categorical predictor contains a unique value for every observation. A classification model may be able to split the data into groups that contain only a single target value, including very small groups. When synthetic values are subsequently sampled from these groups, there may be little or no randomness in the generated values. This means that the relationship between the predictor and target column will be almost exactly reproduced. As a result, the synthetic target column may closely match the original target column. 

To avoid this problem, we recommend synthesising data with CART, which offers the option to specify a minimum number of observations per leaf node. Increasing this minimum leaf size prevents the model from creating very small groups based on rare or unique values. The appropriate minimum leaf size should be chosen by balancing privacy protection against the ability of the model to preserve meaningful relationships in the data. See {ref}`Guide 3.1.4: Configuring CART <314-configuring-cart>`  and `Example: configure tune_cart <../../examples/tune_cart_function.html>`__ on how to adjust this parameter.

This is particularly relevant when rare categories are associated with sensitive attributes. If a rare category corresponds to a small group of individuals and the relationship between that category and a sensitive variable is preserved too accurately during the synthesis process, the synthetic data may allow a third-party to infer sensitive information about members of that group. This is a form of attribute disclosure.

For example, consider a dataset containing a categorical variable that identifies a small group and a variable describing a sensitive characteristic of those individuals.
If the synthesis model learns a deterministic relationship between the two variables, the synthetic dataset may reproduce that relationship even though the individual records themselves are not directly copied.
A person with external knowledge about the group could then use the synthetic data to infer the sensitive characteristic.

While preserving relationships between variables is generally an objective of the synthesis method it can also increase the risk of attribute disclosure, particularly for small groups or individual records.

Generally, the risk of attribute disclosure can be increased by:

* categorical variables with many rare or unique categories;
* small subgroups with little variation in sensitive attributes;
* strong or deterministic relationships between quasi-identifiers and sensitive variables;
* synthesis models that overfit small groups; and
* synthesis methods that reproduce original relationships with little or no added uncertainty.

The presence and characteristics of rare categories should therefore be carefully assessed. In particular, users should examine whether small or rare groups are represented in the synthetic data and whether sensitive attributes associated with these groups are preserved in a way that increases disclosure risk.

This risk is not specific to any particular synthesis method. It depends on the interaction between the characteristics of the original data, the synthesis model, and the specific synthesis configuration. A model that captures common categories well may still overfit rare categories or small subgroups.

A synthetic dataset should therefore not be assumed to be provide strong privacy protection simply because individual records are not explicitly copied. Privacy evaluation should also consider whether the synthesis process has preserved relationships within rare groups in a way that could enable inference of sensitive information.

For a worked example demonstrating how rare or unique categories can cause a synthesis model to overfit and reproduce sensitive attributes, see the [Risk of privacy loss due to rare categories example](../examples/rare_categories.md).

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