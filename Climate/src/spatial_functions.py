"""
Name: Spatial_functions.py
Desc: Helper functions for cmip6 and era5 data
Refs:
    https://github.com/Jorl17/open-elevation/blob/master/docs/api.md
"""

import urllib
import pandas as pd
import numpy as np

import multiprocessing 
import time 
import json

from tqdm import tqdm
from itertools import product
from requests import Session

#-----------------------------------------------------------
# Functions
#-----------------------------------------------------------
# Make a grid from dictionary d inputs
def expand_grid(d):
   return pd.DataFrame([row for row in product(*d.values())], 
                       columns=d.keys())


# Define elevation function for national map api (this api is bad outside of north america) 
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
        for l in tqdm(lonlat_list,leave=False):                    
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


# Define elevation function with API from open elevation resource
def open_elevation(lonlat_list):
    """
    Query service using lon, lat & output the elevation values as a new list.
    """
    # Set long and lat argumaents as dictionary and then json
    args=[{}]*len(lonlat_list)
    for i in range(len(lonlat_list)):
        args[i]={"latitude":lonlat_list[i][1],"longitude":lonlat_list[i][0]}

    location={"locations":args}
    json_data=json.dumps(location,skipkeys=int).encode('utf8')

    # Send the request 
    url="https://api.open-elevation.com/api/v1/lookup"
    response = urllib.request.Request(url,json_data,headers={'Content-Type': 'application/json'})
    rq=urllib.request.urlopen(response)

    res_byte=rq.read()
    res_str=res_byte.decode("utf8")
    js_str=json.loads(res_str)
    rq.close()

    # Get elevation 
    response_len=len(js_str['results'])
    elevations=[]
    lon = []
    lat = []
    for j in range(response_len):
        elevations.append(js_str['results'][j]['elevation'])
        lon.append(lonlat_list[j][0])
        lat.append(lonlat_list[j][1])

    out = {'lon':lon, 'lat':lat, 'elev':elevations}
    return out


# Conver longitude to merc projection between -180 and 180
def to_mercator(lon):
    # Convert to array and shift to mercator
    lon_array = np.array(lon)
    lon_array = np.where(lon_array > 180, lon_array - 360, lon_array) 
    return lon_array