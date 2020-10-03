import sys
import pandas as pd
import numpy as np
from xml.dom.minidom import parse, parseString
from pykalman import KalmanFilter
import statistics as std
from math import radians, cos, sin, asin, sqrt, pi


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

# Function to create Data Frame
def get_data(filename):

    # Parsing the file
    file_parse = parse(filename)

    # Get elements by 'trkpt'
    file_elem = file_parse.getElementsByTagName('trkpt')

    # Create a dataframe to store latitude and longitude values
    df = pd.DataFrame(columns=['latitude','longitude'])

    # Iterate through file to read 'lat' and 'lon' elements
    for trkpt in file_elem:
        df.latitude[trkpt] = trkpt.getAttribute('lat')
        df.longitude[trkpt] = trkpt.getAttribute('lon')

    # Create and fill the main dataframe for GPS values
    gps_df = pd.DataFrame()
    gps_df['lat'] = df['latitude'].values.astype(float)
    gps_df['lon'] = df['longitude'].values.astype(float)

    return gps_df

# Function to calculate distance
def distance(data):

    # Using shift for easier distance calculation
    data['lat_s'] = data['lat'].shift(-1,fill_value=0)
    data['lon_s'] = data['lon'].shift(-1,fill_value=0)

    # Radius of earth in km
    radius_earth = 6371

    # Storing values of various columns of 'data' in to variables
    lat = data['lat']
    lon = data['lon']
    lat_s = data['lat_s']
    lon_s = data['lon_s']

    # Computations for calculating distance
    lat_rad_diff = np.deg2rad(lat_s-lat)
    lon_rad_diff = np.deg2rad(lon_s-lon)

    x = np.sin(lat_rad_diff/2) * np.sin(lat_rad_diff/2) +np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lat_s)) * np.sin(lon_rad_diff/2) * np.sin(lon_rad_diff/2)
    y = 2 * np.arctan2(np.sqrt(x), np.sqrt(1-x))

    # Resulting array
    arr_result = radius_earth*y

    # Omit last element
    arr_result = arr_result[:-1]
    result = np.sum(arr_result)

    # Delete unwanted columns
    del(data['lat_s'])
    del(data['lon_s'])

    # Return result in metres
    return result*1000


# Function for Kalman Filtering
def smooth(points):
    # Defining initial state, covariances and transition
    initial_state = points.iloc[0]
    observation_covariance = np.diag([2/100000,2/100000]) ** 2
    transition_covariance = np.diag([1/100000,1/100000]) ** 2
    transition = [[1,0],[0,1]]

    kf = KalmanFilter(initial_state_mean=initial_state, initial_state_covariance=observation_covariance, observation_covariance=observation_covariance, transition_covariance=transition_covariance, transition_matrices=transition)

    x,y = kf.smooth(points)
    arr = np.array(x)
    data_frame = pd.DataFrame(arr, columns=['lat','lon'])

    return data_frame



def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()