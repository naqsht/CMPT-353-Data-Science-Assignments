import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

# Lowest Total Precipitation
total_city_prec = totals.sum(axis=1)
lowest_prec = total_city_prec.idxmin(axis=1)

# Average Precipitation for each Month
total_mon_prec = totals.sum(axis=0)
num_obs_mon = counts.sum(axis=0)
avg_mon_prec = total_mon_prec/num_obs_mon

# Average Precipitation in each City
# We already have Total Precipitation for a city in total_city_prec  
num_obs_city = counts.sum(axis=1)
avg_city_prec = total_city_prec/num_obs_city

# Printing Results
print ("City with lowest total precipitation:\n",lowest_prec)
print("Average precipitation in each month:\n",avg_mon_prec)
print("Average precipitation in each city:\n",avg_city_prec)