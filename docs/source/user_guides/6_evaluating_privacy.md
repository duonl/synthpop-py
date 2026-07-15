# 6. Evaluating privacy
Synthetic data is designed to preserve useful statistical properties of the original data while reducing the risk of disclosing information about individuals. Privacy evaluation assesses whether the synthetic dataset could reveal sensitive information about individuals represented in the original dataset.

Privacy and utility are inherently connected. Increasing the similarity between synthetic and original data can improve analytical usefulness, but may also increase privacy risks. Conversely, stronger privacy protection may require reducing the amount of information retained from the original data.

There is therefore no universally optimal balance between utility and privacy. The appropriate trade-off depends on the intended use of the synthetic data, the sensitivity of the information and the acceptable level of disclosure risk.

Privacy evaluation should be considered alongside utility evaluation. A synthetic dataset that has high utility but insufficient privacy protection may not be suitable for release, while a highly private dataset with very low utility may not be useful for analysis.

synthpop-py currently provides the framework for synthetic data generation but does not yet include automated privacy risk metrics. This guide describes common privacy risks that should be considered when evaluating synthetic data.

---
## 6.1. Privacy risks
Privacy risks in synthetic data are commonly divided into three main categories[^1]:

[^1]: Zhe Si Yu (2025), *Synthetic Data Disclosure Risk Assessment: A Literature Review*, in *Proceedings of Statistics Canada Symposium 2024: The Future of Official Statistics*, Statistics Canada International Symposium Series: Proceedings, Issue 2025001.

| Risk type | Main question | Information revealed
| --- | --- | --- |
| Identity disclosure | "Can I identify this person in the dataset?"| The person's record or attributes can be linked to their identity |
| Attribute disclosure | "What sensitive information can I infer about this person?" | Sensitive characteristics or values |
| Membership disclosure | "Was this person included in the dataset?" | The person's participation in the original dataset |

These risks describe different ways in which information about individuals in the original dataset may be inferred from the synthetic data.

### 6.1.1. Identity disclosure
Identity disclosure occurs when an individual in the original dataset can be linked to a record in the synthetic dataset.

This risk is related to the ability to identity a person based on combinations of characteristics. For example, a synthetic record containing a rare combination of age, occupation, location and other attributes could allow an attacker to infer which original individual it represents.

Identity disclosure is more likely when:
- The original dataset contains unique or rare combinations of variables.
- The synthesis method overfits the original data so that the synthetic dataset reproduces rare or individual-level patterns exactly.
- Many quasi-identifying variables are released.
- The synthetic generation process memorises individual records.

A common approach for evaluating identity disclosure is to measure how closely synthetic records match original records based on identifying attributes.

### 6.1.2. Attribute disclosure
Attribute disclosure occurs when an attacker learns sensitive information about an individual, even without identifying the exact record.

For example, if an individual is known to belong to a particular group and the synthetic data reveal that all individuals in that group share a sensitive attribute, an attacker may infer information about the individual.

Attribute disclosure can occur when:
-  Relationships between quasi-identifiers and sensitive attributes are preserved too strongly.
- Rare groups have little variation in sensitive attributes.
- The synthetic data reveal deterministic relationships from the original data.

Preventing attribute disclosure requires considering not only whether individual records can be identified, but also whether sensitive information can be inferred from preserved statistical relationships.

### 6.1.3. Membership disclosure
Membership disclosure occurs when an attacker can determine whether a particular individual was included in the original dataset.

For example, if a person suspects that their medical records were included in a dataset, synthetic data should not allow an attacker to confidently confirm or deny their presence.

Unlike identity disclosure, membership disclosure does not require identifying a record of learning any attributes about the individual. The attacker only learns that the person's data contributed to the original dataset.

Factors that can increase membership disclosure include
- Small original datasets.
- Rare individuals or rare combinations of characteristics.
- Overfitting during synthesis.
- Synthetic records that are too close to the original records.

Membership disclosure is particularly relevant when synthetic data are released publicly, because attackers may combine the synthetic dataset with external information.