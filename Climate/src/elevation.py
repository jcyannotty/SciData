"""
Name: elevation.py
Desc: Extract elevation from API
Refernces:
https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
"""

import requests
import urllib
import pandas as pd
import numpy as np

import multiprocessing 
import time 

import cftime
import xarray as xr 
import netCDF4 as nc

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


# Define elevation function for API 
def elevation_function(lonlat_list):
    """
    Query service using lon, lat & output the elevation values as a new list.
    """
    elevations = []
    lon = []
    lat = []
    out = {}
    with Session() as s:
        for l in tqdm(lonlat_list):                    
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


#-----------------------------------------------------------
# Execution
#-----------------------------------------------------------
# Small SW Grid
lon_list = np.linspace(235, 259.75, 100).tolist()
lat_list = np.linspace(30, 59.75, 120).tolist()
lon_lat = [[x,y] for x in lon_list for y in lat_list]
len(lon_lat)
#lldf = pd.DataFrame(lon_lat)
#lldf.columsn = ["lon","lat"]

# St url for the API
url = "https://epqs.nationalmap.gov/v1/json?"
era5elevpath = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"
#era5elevpath = "/home/yannotty.1/"

# Batch read
nbatches = 30
tc = 4
nlat = len(lat_list)
nlon = len(lon_list)
n = len(lon_lat) 
idx = np.array_split(range(n),4*nbatches)
out_elev = []
out_lat = []
out_lon = []
if __name__ == '__main__': 
    pool = multiprocessing.Pool(processes=4) 
    for b in range(nbatches):
        if b+1 == nbatches:
            end = n
        else:
            end = idx[tc*(b+1)][0]

        inputs = []
        for i in range(tc-1):
            inputs.append(lon_lat[idx[b*tc + i][0]:idx[b*tc + i+1][0]])    
        # For the last core
        inputs.append(lon_lat[idx[b*tc + tc-1][0]:end])

        #inputs = [lon_lat[idx[b*4][0]:idx[b*4 + 1][0]], lon_lat[idx[b*4+1][0]:idx[b*4+2][0]],
        #        lon_lat[idx[b*4+2][0]:idx[b*4+3][0]], lon_lat[idx[b*4+3][0]:end]]
        
        outputs = pool.map(elevation_function, inputs) 
        
        for out in outputs:
            out_elev = out_elev + out["elev"]
            out_lat = out_lat + out["lat"]
            out_lon = out_lon + out["lon"]

        # Store results in df
        df = pd.DataFrame({"lon":out_lon,"lat":out_lat,"elev":out_elev})
        df.to_csv(era5elevpath + "SWUSA_" + str(b) + ".csv", index = False)

        # Clear results
        out_elev = []
        out_lat = []
        out_lon = []






# Process results
# for b in range(nbatches):
#     if b == 0:
#         elev_df = pd.read_csv(era5elevpath + "SWUSA_" + str(b))
#     else:
#         temp_df = pd.read_csv(era5elevpath + "SWUSA_" + str(b))
#         pd.concat(elev_df,temp_df)

# df.loc[df["elev"]==-999]
# elev = np.array(df["elev"].to_list()).reshape(nlon,nlat)

# # Write file
# era5elevpath = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"

# ncfile = nc.Dataset(era5elevpath + "SW_USA9000.nc",mode='w',format='NETCDF4_CLASSIC')
# lat_dim = ncfile.createDimension('lat', len(lat_list))     # latitude axis
# lon_dim = ncfile.createDimension('lon', len(lon_list))    # longitude axis

# ncfile.title="ERA5-Elevations"

# lat = ncfile.createVariable('lat', np.float32, ('lat',))
# lat.long_name = 'latitude'
# lat = lat_list

# lon = ncfile.createVariable('lon', np.float32, ('lon',))
# lon.long_name = 'longitude'
# lon = lon_list

# elv = ncfile.createVariable('elev',np.float64,('lat','lon'))
# elv.units = 'm'
# elv.standard_name = 'Elevation'
# elv[:,:] = elev

# ncfile.close()


# #ev = elevation_function(lon_lat[0:12])

# # Small SW Grid
# lon_list = np.linspace(235, 259.5, 50).tolist()
# lat_list = np.linspace(30.25, 59.75, 60).tolist()
# lon_lat = [[x,y] for x in lon_list for y in lat_list]
# len(lon_lat)

# swsubdf = pd.DataFrame(lon_lat)
# swsubdf.columns = ["lon","lat"]

# # Get entire SW Grid
# swlon_list = np.linspace(235, 259.75, 100).tolist()
# swlat_list = np.linspace(30, 59.75, 120).tolist()
# swlon_lat = [[x,y] for x in swlon_list for y in swlat_list]
# len(swlon_lat)

# swdf = pd.DataFrame(swlon_lat)
# swdf.columns = ["lon","lat"]

# # Remove the values you've already pulled
# swdf = swdf.loc[~(swdf["lon"].isin(lon_list) & swdf["lat"].isin(lat_list))]

