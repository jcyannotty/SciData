import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import cftime
import xarray as xr 
import netCDF4 as nc

from tqdm import tqdm
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp2d
from sklearn.neighbors import KNeighborsRegressor
from Climate.src.interpolators import bilinear

#----------------------------------------------------------
# Functions
#----------------------------------------------------------
def get_timeperiod(f):
    rng = f.split('_')[-1][:-3]
    return [int(s[:-2]) for s in rng.split('-')]

#----------------------------------------------------------
# TAS from list
#----------------------------------------------------------
cmip6path = "/home/johnyannotty/NOAA_DATA/CMIP6/historical/"
file_list = sorted(os.listdir(cmip6path))
file_list = [f for f in file_list if "tas_" in f]
tas_data = []
lat_data = []
lon_data = []
fillval_data = []
i = 0
for f in file_list:
    ncdata = nc.Dataset(cmip6path + f, "r")
    print(i)

    # Get index and appropriate time period
    tp = get_timeperiod(f)
    n_begin = (1950 - tp[0]) * 12
    n_end = (2015 - tp[0]) * 12

    # Pull the temperature data using the indexes & convert to celsius 
    xi = ncdata.variables['tas'][n_begin:n_end]
    xi = np.array(xi)
    xi = np.array(xi) - 273.15
    
    # Rotate and flip to get the eastern and western hemisphere oriented correctly
    #xi = np.rot90(xi, k = 2, axes = (1, 2))
    #xi = np.flip(xi, axis = 2)

    # Get the fill values
    fv = ncdata.variables['tas']._FillValue

    # Get longitude and latitude
    lat = np.array(ncdata.variables["lat"])
    lon = np.array(ncdata.variables["lon"])

    # Append to lists
    tas_data.append(xi)
    lat_data.append(lat)
    lon_data.append(lon)
    fillval_data.append(fv)
    i+=1

len(lat_data)


#----------------------------------------------------------
# ERA5 Pull
#----------------------------------------------------------
era5path = "/home/johnyannotty/NOAA_DATA/ERA5/"
ncdata = nc.Dataset(era5path + 'era5_avg_mon_tas/data_1990-2023.nc', "r")
#ncdata = nc.Dataset(era5path + 'era5_land_avg_mon_tas_1990-2023/data.nc', "r")
#y = ncdata.variables['t2m']
era5_lon = np.array(ncdata.variables['longitude'])
era5_lat = np.array(ncdata.variables['latitude'])

ncdata.close()
del ncdata


#----------------------------------------------------------
# CMIP6 interpolations
#----------------------------------------------------------
cmip6intpath = '/home/johnyannotty/NOAA_DATA/CMIP6_Interpolations/'
num_tpds = 12 
yr_list = [2014]

for yr in yr_list:
    yr_offset = 2014-yr+1
    for c,f in tqdm(enumerate(file_list),desc = "Simulator" ,leave = False):
        f = f.split("185")[0] + str(yr) + ".nc"
        #cmpi6_interp_files.append(cmip6intpath + "bilinear_" + f)
        try:
            ncfile = nc.Dataset(cmip6intpath + "bilinear_" + f,mode='w',format='NETCDF4_CLASSIC')
            lat_dim = ncfile.createDimension('lat', era5_lat.shape[0])     # latitude axis
            lon_dim = ncfile.createDimension('lon', era5_lon.shape[0])    # longitude axis
            time_dim = ncfile.createDimension('time', num_tpds)#tas_data[0].shape[0]) # unlimited axis (can be appended to).

            ncfile.title=f
            ncfile.subtitle='TAS Interpolations'

            lat = ncfile.createVariable('lat', np.float32, ('lat',))
            lat.long_name = 'latitude'
            lat = era5_lat

            lon = ncfile.createVariable('lon', np.float32, ('lon',))
            lon.long_name = 'longitude'
            lon = era5_lon

            time = ncfile.createVariable('time', np.float64, ('time',))
            time.units = 'months from 2000 to 2014'
            time.long_name = 'time'
            time = np.linspace(1,num_tpds,num_tpds)

            temp = ncfile.createVariable('tas',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
            temp.units = 'C'
            temp.standard_name = 't2m'


            # Batch Interpolations....(batch over time, long and lat)        
            for i in tqdm(range(num_tpds),desc = "Time Period",leave=False):
                tm = tas_data[c].shape[0] - num_tpds*yr_offset + i
                bilinear_interp = interp2d(np.array(lon_data[c]),np.array(lat_data[c]), tas_data[c][tm], kind = "linear")
                temp[i,:,:] = np.flip(bilinear_interp(np.array(era5_lon), np.array(era5_lat)),axis = 0)
            ncfile.close()
        except: 
            print("Error for file: " + f)
            ncfile.close()

len(tas_data[c].shape[0])
