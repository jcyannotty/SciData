"""
Name: elevation.py
Desc: Load elevation data and construct netcdf file
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

# Write as a netcdf file
era5elevpath = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"
nlon = 100
nlat = 120
lon_list = np.linspace(235, 259.75, nlon).tolist()
lat_list = np.linspace(30, 59.75, nlat).tolist()

elev_df = pd.read_csv(era5elevpath + "SWUSA.txt")

ncfile = nc.Dataset(era5elevpath + "SWUSA.nc",mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', len(lat_list))     # latitude axis
lon_dim = ncfile.createDimension('lon', len(lon_list))    # longitude axis

ncfile.title="ERA5-Elevations"

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.long_name = 'latitude'
lat = lat_list

lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.long_name = 'longitude'
lon = lon_list

elv = ncfile.createVariable('elev',np.float64,('lon','lat'))
elv.units = 'm'
elv.standard_name = 'Elevation'
elev = np.array(elev_df["elev"].to_list()).reshape(nlon,nlat)
elv[:,:] = elev

ncfile.close()
