import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Reading input file
data_file = sys.argv[1]
data = pd.read_csv(data_file)

# ANOVA test, to see if any of the groups differ
anova = stats.f_oneway(data['qs1'], data['qs2'], data['qs3'], data['qs4'], data['qs5'], data['merge1'], data['partition_sort'])

# Using melt to un-pivot the DataFrame
data_melt = pd.melt(data)

# Post Hoc Analysis
posthoc = pairwise_tukeyhsd(data_melt['value'], data_melt['variable'], alpha=0.05)

print(data.mean())

# Print results
print(posthoc)
fig = posthoc.plot_simultaneous()
plt.show()