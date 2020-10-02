import sys
import matplotlib.pyplot as plt
import pandas as pd

filename1 = sys.argv[1]
filename2 = sys.argv[2]

# Plot 1: Distribution of Views
plot1_data = pd.read_csv(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
plot1_data_sort = plot1_data.sort_values(by='views', ascending=False)


# Plot 2: Hourly Views
plot2_data = pd.read_csv(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
plot1_data['views_sec'] = plot2_data['views']


# Plotting Figure 1
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(plot1_data_sort['views'].values)
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')


# Plotting Figure 2
plt.subplot(1,2,2)
plt.xscale('log')
plt.yscale('log')
plt.scatter(plot1_data['views'], plot1_data['views_sec'])
plt.title('Hourly Correlation')
plt.xlabel('Hour 1 Views')
plt.ylabel('Hour 2 Views')


plt.savefig('wikipedia.png')
