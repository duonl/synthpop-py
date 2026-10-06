# import seaborn as sns
import numpy as np

# data = sns.load_dataset("titanic").astype(np.float32)

# data.head(3)

from synthpop import Synthesiser

# synthesiser = Synthesiser(random_seed=1)

# synthesiser.fit(data)

# synthetic_data = synthesiser.generate(n=5000)

# from synthpop.plotting import plot_univariate_distributions

# plot_univariate_distributions(
#     orig_df=data,
#     syn_df=synthetic_data,
#     interactive=True
# )

from synthpop.utility_metrics import pairwise_spmse
from synthpop.plotting import plot_spmse

# spmse = pairwise_spmse(
#     orig_df=data,
#     syn_df=synthetic_data,
# )

# plot = plot_spmse(spmse, show_plot=True)

import seaborn as sns
 
data = sns.load_dataset("titanic")
data['fare'] = data['fare'].astype(np.float32)
synthesiser = Synthesiser(random_seed=1)
 
synthesiser.fit(data)
synthetic_data = synthesiser.generate(n=5000)
spmse = pairwise_spmse(
    orig_df=data,
    syn_df=synthetic_data,
)
 
plot = plot_spmse(spmse,show_plot=True)