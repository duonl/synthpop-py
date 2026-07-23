# Make your synthesis reproducible
In the previous example, we created our first synthetic dataset. You may have noticed that we created the Synthesiser using the `random_seed` parameter:
```python
synthesiser = Synthesiser(random_seed=1)
```
Without specifying a `random_seed`, if you run the same code again, you may notice that the generated synthetic dataset is different each time. This is expected. Synthetic data generation is a stochastic process: many synthesis methods involve randomness in, for example, sampling.

Sometimes this is exactly what you want. However, when developing analyses, writing reports or collaborating with others, it is often useful to generate exactly the same synthetic dataset every time. This is called **reproducibility**.

## Generate two synthetic datasets
Let's start by fitting a Synthesiser without specifying a random seed.
```python
from sklearn.datasets import load_diabetes
from synthpop.synthesiser import Synthesiser

data = load_diabetes(as_frame=True).frame

synthesiser = Synthesiser()
synthesiser.fit(data)

synthetic_data_1 = synthesiser.generate()
synthetic_data_2 = synthesiser.generate()
```
Both datasets were generated from the same fitted Synthesiser, but they are not identical.

**Note**: the results below may vary from yours as they are not reproducible.
```python
synthetic_data_1.head(3)
```
|    |        age |        sex |        bmi |          bp |          s1 |          s2 |         s3 |          s4 |         s5 |         s6 |   target |
|---:|-----------:|-----------:|-----------:|------------:|------------:|------------:|-----------:|------------:|-----------:|-----------:|---------:|
|  0 | -0.107226  | -0.0446416 | -0.011595  | -0.0400989  | -0.0153285  | -0.00131388 | -0.0323559 |  0.0158583  | -0.027129  |  0.0196328 |       55 |
|  1 |  0.0489735 | -0.0446416 | -0.0256066 | -0.00567042 |  0.0176944  |  0.0434664  |  0.0707299 | -0.0247329  | -0.0159989 |  0.0610539 |      113 |
|  2 | -0.0563701 | -0.0446416 |  0.0444512 | -0.0194418  | -0.00707277 | -0.0238606  | -0.0139477 | -0.00259226 |  0.0757406 | -0.013504  |      121 |

```python
synthetic_data_2.head(3)
```
|    |        age |       sex |         bmi |         bp |         s1 |         s2 |          s3 |          s4 |          s5 |         s6 |   target |
|---:|-----------:|----------:|------------:|-----------:|-----------:|-----------:|------------:|------------:|------------:|-----------:|---------:|
|  0 |  0.0671362 | 0.0506801 | -0.0148285  |  0.0586076 |  0.0176944 | 0.0494162  | -0.00658447 | -0.00259226 | -0.0332456  |  0.0569118 |      111 |
|  1 | -0.0273098 | 0.0506801 | -0.00728377 | -0.0297704 |  0.0465894 | 0.0732155  | -0.0286743  |  0.0343089  | -0.00149595 | -0.0466409 |      200 |
|  2 |  0.0417084 | 0.0506801 | -0.0223731  | -0.0400989 | -0.030464  | 0.00119131 | -0.021311   |  0.0158583  | -0.0109033  |  0.0444855 |       83 |  

Although both datasets preserve similar statistical properties, the individual synthetic records differ because new random samples are drawn each time.
```python
synthetic_data_1.describe()
```
|       |           age |          sex |           bmi |           bp |           s1 |           s2 |           s3 |           s4 |           s5 |           s6 |   target |
|:------|--------------:|-------------:|--------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------:|
| count | 442           | 442          | 442           | 442          | 442          | 442          | 442          | 442          | 442          | 442          | 442      |
| mean  |   0.000345173 |   0.00150962 |   0.000777878 |  -0.00262506 |  -0.00255579 |  -0.00409148 |  -0.00128691 |  -0.00179496 |   0.0023855  |  -0.0012089  | 151.928  |
| std   |   0.0484934   |   0.0476909  |   0.0485026   |   0.0476364  |   0.0488929  |   0.049221   |   0.0486984  |   0.0477245  |   0.0468386  |   0.0464237  |  76.3776 |
| min   |  -0.107226    |  -0.0446416  |  -0.0891975   |  -0.112399   |  -0.126781   |  -0.115613   |  -0.102307   |  -0.0763945  |  -0.104366   |  -0.129483   |  31      |
| 25%   |  -0.0409318   |  -0.0446416  |  -0.0331513   |  -0.0366561  |  -0.0373437  |  -0.0356819  |  -0.0360376  |  -0.0394934  |  -0.0307479  |  -0.033179   |  89      |
| 50%   |   0.00538306  |  -0.0446416  |  -0.00620595  |  -0.0108347  |  -0.00707277 |  -0.010082   |  -0.00658447 |  -0.00259226 |   0.00200444 |   0.00306441 | 137.5    |
| 75%   |   0.0380759   |   0.0506801  |   0.0315175   |   0.0287581  |   0.0245741  |   0.0262432  |   0.0302319  |   0.0343089  |   0.0354587  |   0.0279171  | 202      |
| max   |   0.110727    |   0.0506801  |   0.170555    |   0.125158   |   0.153914   |   0.198788   |   0.173816   |   0.185234   |   0.133597   |   0.135612   | 341      |   

```python
synthetic_data_2.describe()
```
|       |           age |          sex |          bmi |           bp |            s1 |           s2 |           s3 |           s4 |           s5 |            s6 |   target |
|:------|--------------:|-------------:|-------------:|-------------:|--------------:|-------------:|-------------:|-------------:|-------------:|--------------:|---------:|
| count | 442           | 442          | 442          | 442          | 442           | 442          | 442          | 442          | 442          | 442           |  442     |
| mean  |   0.000575288 |   0.00021566 |   0.00363823 |   0.001088   |   1.55649e-05 |  -0.00141909 |  -0.00410228 |   0.00398565 |   0.00285974 |   0.00231471  |  155.394 |
| std   |   0.0492173   |   0.0476323  |   0.047053   |   0.0482004  |   0.0471131   |   0.0470191  |   0.0447474  |   0.0500362  |   0.0491795  |   0.0484639   |   77.315 |
| min   |  -0.107226    |  -0.0446416  |  -0.0902753  |  -0.108956   |  -0.108893    |  -0.112795   |  -0.102307   |  -0.0763945  |  -0.126097   |  -0.137767    |   31     |
| 25%   |  -0.0418399   |  -0.0446416  |  -0.0320734  |  -0.0366561  |  -0.0345918   |  -0.0338813  |  -0.0397192  |  -0.0394934  |  -0.0316777  |  -0.0300724   |   88.25  |
| 50%   |   0.0090156   |  -0.0446416  |  -0.0013558  |  -0.00567042 |  -0.00363289  |  -0.00976889 |  -0.0139477  |  -0.00259226 |   0.00286131 |   0.000993356 |  151     |
| 75%   |   0.0380759   |   0.0506801  |   0.0358287  |   0.0390866  |   0.0256061   |   0.0293747  |   0.0265503  |   0.0343089  |   0.037667   |   0.0279171   |  216.75  |
| max   |   0.0961965   |   0.0506801  |   0.137143   |   0.125158   |   0.153914    |   0.155887   |   0.159089   |   0.185234   |   0.133597   |   0.135612    |  341     |   

## Setting a random seed
To obtain reproducible results, specify the `random_seed` parameter when creating the Synthesiser.
```python
synthesiser_1 = Synthesiser(random_seed=1)

synthesiser_1.fit(data)

synthetic_data_1 = synthesiser_1.generate()
```
Now create a **new** Synthesiser with the same random seed.
```python
synthesiser_2 = Synthesiser(random_seed=1)

synthesiser_2.fit(data)

synthetic_data_2 = synthesiser_2.generate()
```
Both datasets are now identical.
```python
synthetic_data_1.equals(synthetic_data_2)
```
```text
True
```
Because every random decision is made from the same initial random state, the complete synthesis process can be reproduced exactly.

## When should you use a random seed?
Using a random seed is recommended when you want reproducible results, for example when:
- writing documentation or tutorials;
- comparing different synthesis methods;
- debugging your synthesis pipeline;
- sharing analyses with colleagues;
- running automated tests.

## Reproducibility and multiple synthetic datasets
Sometimes you may want several different synthetic datasets from the same fitted Synthesiser, for example to quantify the variability introduced by the synthesis process.

However, if a random seed is provided, you cannot simply call `generate()` multiple times as we saw in the first code block of this page.
```python
synthesiser = Synthesiser(random_seed=1)
synthesiser.fit(data)

synthetic_data_1 = synthesiser.generate()
synthetic_data_2 = synthesiser.generate()
```
```python
synthetic_data_1.equals(synthetic_data_2)
```
```text
True
```
To create multiple reproducible synthetic dataset from the same fitted synthesiser, you can use the `random_seed` parameter in the `generate()` function.
```python
synthetic_data_3 = synthesiser.generate(random_seed=2)
```
Each generated dataset with a different random seed will be different, because the synthesiser continues drawing new random samples.
```python
synthetic_data_3.equals(synthetic_data_1)
```
```text
False
```
Omitting the `random_seed` parameter will again provide you with the first synthetic dataset.
```python
synthetic_data_4 = synthesiser.generate()

synthetic_data_4.equals(synthetic_data_1)
```
```text
True
```
## Next steps
Now that your synthesis is reproducible, the next examples in this module show you how to generate larger datasets and how to choose a synthesis order.