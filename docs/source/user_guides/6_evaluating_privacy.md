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

A sensitive attribute is an attribute whose value is considered confidential and could cause harm or reveal private information if inferred about an individual. Examples include medical diagnoses, income, political affiliation or other personal characteristics. In the context of attribute disclosure, the concern is that a third-party who already knows an individual's quasi-identifying characteristics, such as age, sex or location, may use the synthetic data to infer the value of one or more sensitive attributes with high confidence.

For example, if an individual is known to belong to a particular group and the synthetic data reveal that all individuals in that group share a sensitive attribute, a third-party may infer information about the individual.

Attribute disclosure can occur when:
* strong or deterministic relationships between quasi-identifiers and sensitive variables;
* categorical variables with many rare or unique categories;
* small subgroups with little variation in sensitive attributes;
* synthesis models that overfit small groups; and
* synthesis methods that reproduce original relationships with little or no added uncertainty.

Preventing attribute disclosure therefore requires considering not only whether individual records can be identified, but also whether sensitive information can be inferred from relationships preserved in the synthetic data.

#### 6.1.2.1 Highly predictive relationships
A strong relationship between a quasi-identifier and a sensitive attribute can create attribute disclosure risk when the relationship is preserved too accurately in the synthetic data. This risk can arise even when the predictor is not rare and the synthesis model has not overfitted the data.

For example, suppose that individuals with a particular combination of age, occupation and location almost always have the same value for a sensitive attribute. If the synthesis model preserves this relationship closely, a third party who knows those characteristics about an individual may be able to infer the individual's sensitive attribute from the synthetic data.

The risk is particularly high when the relationship is deterministic or nearly deterministic. In such cases, preserving the relationship with high fidelity may leave little uncertainty about the sensitive attribute. Adding more randomness to the synthesis process may reduce the disclosure risk, but may also reduce the statistical utility of the synthetic data.

This illustrates an important privacy-utility trade-off. Relationships between variables are generally important for producing useful synthetic data, but preserving a relationship involving a sensitive attribute can also make that attribute easier to infer. The objective is therefore not necessarily to remove all relationships, but to assess whether the relationships that are preserved provide sufficient information to enable sensitive attributes to be inferred.

The risk should be assessed in the context of the information reasonably available to a potential data recipient. A relationship may present little disclosure risk if the relevant quasi-identifying information is not available outside the dataset. Conversely, a relationship may present substantial risk where the quasi-identifiers can readily be obtained from external sources.

Highly predictive relationships can therefore create attribute disclosure risk independently of model overfitting. Even a well-fitting synthesis model may preserve a strong relationship from the original data if that relationship is genuinely present in the source population. In such cases, changing model parameters intended to reduce overfitting may provide limited additional protection.

#### 6.1.2.2. Rare categories and overfitting
Categories with very few records in the data, so-called rare categories, can create additional privacy risks when generating synthetic data. When a categorical variable contains rare or unique values, a synthesis model may create very small and highly homogeneous groups, increasing the risk of overfitting the original data.

For example, suppose a categorical predictor contains a unique value for every observation. A classification model may be able to split the data into groups that are highly homogeneous with respect to the target, including groups containing relatively few observations. If synthetic target values are subsequently sampled from these groups, there may be little uncertainty in the generated values. The relationship between the predictor and target may therefore be reproduced very closely.

To reduce this risk, we recommend considering CART with a minimum number of observations per leaf node. Increasing the minimum leaf size can prevent the model from creating very small groups based on rare or unique values. This can force observations that would otherwise be separated into small groups to be modelled together, introducing more variability when synthetic values are generated. However, the effectiveness of this approach depends on the structure of the data and it does not by itself guarantee protection against disclosure. See {ref}`Guide 3.1.4: Configuring CART <314-configuring-cart>`  and [Example: The tune_cart function](../examples/tune_cart_function.md) for information on how to adjust the minimum leaf size parameter.

For example, consider a dataset with 100 observations where a predictor has a unique value for every observation and the target has 10 possible values. CART could create 10 leaves containing approximately 10 observations each, even with a minimum leaf size of 5. If each leaf corresponds predominantly to one target value, the predictor can still provide substantial information about the target despite the relatively large leaf size. With 200 observations, the same structure could result in leaves containing approximately 20 observations each. In such cases, increasing the minimum leaf size may have little effect because the underlying structure of the data already allows the target to be inferred from the predictor.

This illustrates an important distinction between model overfitting and relationships that are inherently highly predictive in the original data. Increasing the minimum leaf size can reduce some forms of overfitting by preventing the model from creating very small, highly homogeneous groups. It cannot necessarily remove the risk when the source data itself contains a strong or deterministic relationship between a predictor and a target. In such cases, accurately preserving the relationship may inherently limit the extent to which privacy can be protected while retaining the utility of the data.

The appropriate minimum leaf size should therefore be selected based on the structure of the data, balancing the reduction of overfitting against the ability of the model to preserve meaningful relationships. In datasets with repeated and non-unique predictor values, increasing the minimum leaf size may be more effective because it can combine observations that would otherwise form small groups. In contrast, where a predictor is highly informative about the target regardless of the model configuration, adjusting the leaf size may provide limited additional protection.

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
