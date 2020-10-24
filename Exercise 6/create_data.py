import time
from implementations import all_implementations
import numpy as np
import pandas as pd

# Creating a DataFrame using np.arange, taking n=200
data = pd.DataFrame(columns = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'], index=np.arange(200))

for i in range(200):
    random_array = np.random.randint(-500,500,1000)
    for sort in all_implementations:
        st = time.time()
        res = sort(random_array)
        en = time.time()
        total = en - st
        data.iloc[i][sort.__name__] = total

# Saving the DataFrame
data.to_csv('data.csv', index=False)