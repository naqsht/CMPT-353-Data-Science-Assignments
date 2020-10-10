import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip


# Command line arguments
stations_file = sys.argv[1]
cities_file = sys.argv[2]
output_file = sys.argv[3]

# Opening and reading gzip file
stations_gzip = gzip.open(stations_file, 'rt', encoding = 'utf-8')

# Creating DataFrame for stations and cities
stations_df = pd.read_json(stations_gzip, lines=True)
cities_df = pd.read_csv(cities_file)

# Divide 'avg_tmax' column by 10, since the data is in C*10
stations_df['avg_tmax'] = stations_df['avg_tmax'] / 10

# Omitting the data that can't be used by using isfinite()
cities_df = cities_df[np.isfinite(cities_df.population)]

# Resetting indexes
cities_df = cities_df[np.isfinite(cities_df.area)].reset_index(drop=True)

# To convert metre square to km square
cities_df['area'] = cities_df['area']/1000000

# To exclude cities with area greater than 10000 km square
# Then resetting index
cities_df = cities_df[cities_df.area <= 10000]
cities_df.reset_index(drop=True)

# To calculate population density
cities_df['density'] = cities_df['population'] / cities_df['area']


# Function for distance
def distance(city, stations):
    # Obtaining latitude and longitude values for city and stations, and converting to radians
    lat_city = np.deg2rad(city['latitude'])
    lon_city = np.deg2rad(city['longitude'])

    lat_stat = np.deg2rad(stations['latitude'])
    lon_stat = np.deg2rad(stations['longitude'])

    # Computing difference in latitude and longitude 
    lat_diff = lat_stat - lat_city
    lon_diff = lon_stat - lon_city

    # Computing the distance
    x = 0.5 - np.cos(lat_diff)/2 + np.cos(lat_city)*np.cos(lat_stat)*(1-np.cos(lon_diff))/2
    y = 12742*np.arcsin(np.sqrt(x))

    return y


# Function for best value found
def best_tmax(city,stations):
    # Adding a distance column
    stations['distance'] = distance(city, stations)
    station = stations_df[stations_df['distance'] == stations_df['distance'].min()]
    station = station.reset_index(drop= True)

    result = station.loc[0, 'avg_tmax']

    return result
 

# Applying the function across all cities
cities_df['avg_tmax'] = cities_df.apply(best_tmax, axis=1, stations= stations_df)


# Scatter Plot of Average Maximum Temperature
plt.scatter(cities_df['avg_tmax'], cities_df['population'])
plt.title('Temperature vs Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)')
plt.ylabel('Population Density (people/km\u00b2)')

# Saving plot figure
plt.savefig(output_file)

