"""
Name: Spatial_functions.py
Desc: Helper functions for cmip6 and era5 data
"""

import urllib
import pandas as pd
import numpy as np

import multiprocessing 
import time 

#from tqdm import tqdm
from itertools import product
from requests import Session

#-----------------------------------------------------------
# Functions
#-----------------------------------------------------------
# Make a grid from dictionary d inputs
def expand_grid(d):
   return pd.DataFrame([row for row in product(*d.values())], 
                       columns=d.keys())


# Define elevation function for API 
def elevation_function(lonlat_list):
    """
    Query service using lon, lat & output the elevation values as a new list.
    """
    url = "https://epqs.nationalmap.gov/v1/json?"
    elevations = []
    lon = []
    lat = []
    out = {}
    with Session() as s:
        for l in lonlat_list:                    
            # define rest query params
            params = {
                'output': 'json',
                'x': l[0],
                'y': l[1],
                'units': 'Meters'
            }
            
            result = s.get(url + urllib.parse.urlencode(params))
            if result.status_code == 200:
                lon.append(l[0])
                lat.append(l[1])
                try:
                    elevations.append(result.json()['value'])
                except:
                    elevations.append(-999)
            else:
                # Handle too many requests
                n = 0
                while result.status_code == 429 and n < 10:
                    time.sleep(5) # Back off 10 seconds.
                    result = s.get(url, params=params)
                    n=n+1
                
                if result.status_code == 200:
                    elevations.append(result.json())
                    lon.append(l[0])
                    lat.append(l[1])
                else:
                    print("Requests session expired...returning elevations")
                    out = {'lon':lon, 'lat':lat, 'elev':elevations}
                    return out
    # Return dictionary of results
    out = {'lon':lon, 'lat':lat, 'elev':elevations}
    return out
