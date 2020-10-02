import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# To calculate number of cities
num_cities = totals.shape[0]

# Lowest Total Precipitation
total_city_prec = np.sum(totals, axis=1)
lowest_prec = np.argmin(total_city_prec)

# Average Precipitation for each Month
total_mon_prec = np.sum(totals, axis=0)
num_obs_mon = np.sum(counts, axis=0)
avg_mon_prec = total_mon_prec/num_obs_mon

# Average Precipitation in each City
# We already have Total Precipitation for a city in total_city_prec  
num_obs_city = np.sum(counts, axis=1)
avg_city_prec = total_city_prec/num_obs_city

# Quarterly Precipitation in each City
quar_prec = totals.reshape((4*num_cities,3))
total_quar_prec = np.sum(quar_prec, axis=1)
total_quar_prec_re = total_quar_prec.reshape(num_cities,4)


# Printing Results
print ("Row with lowest total precipitation:\n",lowest_prec)
print("Average precipitation in each month:\n",avg_mon_prec)
print("Average precipitation in each city:\n",avg_city_prec)
print("Quarterly precipitation totals:\n",total_quar_prec_re)