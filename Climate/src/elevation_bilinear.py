"""
Name: elevation_bilinear.py
Desc: Bilinear interpolation for the elevation data from conepercius
Refernces:
    https://confluence.ecmwf.int/display/CKB/Near+surface+meteorological+variables+from+1979+to+2019+derived+from+bias-corrected+reanalysis+%28WFDE5%29%3A+Product+User+Guide
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-near-surface-meteorological-variables?tab=overview
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import cftime
import xarray as xr 
import netCDF4 as nc

from tqdm import tqdm
from scipy.interpolate import interp2d
from sklearn.neighbors import KNeighborsRegressor

from tqdm import tqdm

era5evdir = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"
era5elevname = "copernicus_altitude.nc"

# Read in the original elevation data
evdata = nc.Dataset(era5evdir + era5elevname, "r")

evlon = np.array(evdata.variables['lon'])
evlat = np.array(evdata.variables['lat'])
ev = np.array(evdata.variables['ASurf'])

lon_min = -180; lon_max = 180; lon_num = (lon_max - lon_min)*4
lon_list = np.linspace(lon_min,lon_max, lon_num+1)

lat_min = -90; lat_max = 90
lat_num = (lat_max - lat_min)*4
lat_list = np.linspace(lat_min,lat_max, lat_num+1)

nlon = len(lon_list)
nlat = len(lat_list)
lon_list[0:4]

# Get bilinear interpolation
ev = np.where(ev > 10**10,0,ev) # Deal with the missing values of 10^20
bilinear_interp = interp2d(evlon,evlat,ev, kind = "linear")

# Create new netcdf with the bilinear interpolations
ncfile = nc.Dataset(era5evdir + "bilinear_" + era5elevname,mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
lon_dim = ncfile.createDimension('lon', nlon) # longitude axis

ncfile.title='Elevation Interpolations'

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.long_name = 'latitude'
lat[:] = lat_list

lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.long_name = 'longitude'
lon[:] = lon_list

elev = ncfile.createVariable('elev',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
elev.units = 'm'
elev.standard_name = 'elevation'
elev[:,:] = bilinear_interp(np.array(lon_list), np.array(lat_list))
ncfile.close()
