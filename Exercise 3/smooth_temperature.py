import sys
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from datetime import datetime
from pykalman import KalmanFilter

# Function to convert to timestamp
def to_timestamp(d):
    return d.timestamp()

file_load = sys.argv[1]
cpu_data = pd.read_csv(file_load)

# Have to convert timestamp to Float value using to_datetime() and to_timestamp()
cpu_data['timestamp_datetime'] = pd.to_datetime(cpu_data['timestamp'])
cpu_data['timestamp_float'] = cpu_data['timestamp_datetime'].apply(to_timestamp)

# LOESS Smoothing
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp_float'], frac = 0.02)
plt.plot(cpu_data['timestamp_float'], loess_smoothed[:,1], 'c-')


# Kalman Smoothing
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]
initial_state = kalman_data.iloc[0]

# Setting up and predicting observations
observation_covariance = np.diag([0.7,0.7,0.7])**2
transition_covariance = np.diag([0.2,0.2,0.2])**2
transition = [[1,-1,0.7], [0,0.6,0.03], [0,1.3,0.8]]

# Kalman Filtering
kf = KalmanFilter(initial_state_mean = initial_state, initial_state_covariance = observation_covariance, observation_covariance = observation_covariance, transition_covariance = transition_covariance, transition_matrices = transition)
kalman_smoothed, _ = kf.smooth(kalman_data)


# Plotting Graphs 
plt.figure(figsize=(12,4))
plt.plot(cpu_data['timestamp_float'], cpu_data['temperature'], 'c.', alpha=0.5)
plt.plot(cpu_data['timestamp_float'], loess_smoothed[:,1], 'r-', alpha=0.5)
plt.plot(cpu_data['timestamp_float'], kalman_smoothed[:, 0], 'g-', alpha=0.5)

plt.legend(['CPU Data', 'LOESS Smoothing', 'Kalman Smoothing'])

# plt.show()

# Saving file
plt.savefig('cpu.svg')
