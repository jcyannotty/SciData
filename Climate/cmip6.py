"""
https://claut.gitlab.io/man_ccia/lab2.html

https://nordicesmhub.github.io/NEGI-Abisko-2019/training/CMIP6_example.html

Motivated by:
"""

import pandas as pd
import numpy as np
import json
import requests
import os

#import intake
import xarray as xr 
import proplot as plot 
import matplotlib.pyplot as plt

#import cartopy.crs as ccrs
import cftime

from tqdm import tqdm
from pyesgf.search import SearchConnection

import netCDF4 as nc



#----------------------------------------------------------
# Functions
#----------------------------------------------------------
def get_timeperiod(f):
    rng = f.split('_')[-1][:-3]
    return [int(s[:-2]) for s in rng.split('-')]


#----------------------------------------------------------
# TAS from list
#----------------------------------------------------------
cmip6path = "/home/johnyannotty/NOAA_DATA/CMIP6/"
#filename = "tas_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
#filename =  "tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc"
#dset = xr.open_dataset(cmip6path+filename, decode_times=True, use_cftime=True)

file_list = sorted(os.listdir(cmip6path))
file_list = [f for f in file_list if "tas_" in f]
tas_data = []
lat_data = []
lon_data = []

for f in file_list:
    ncdata = nc.Dataset(cmip6path + f, "r")

    # Get index and appropriate time period
    tp = get_timeperiod(f)
    n_begin = (1950 - tp[0]) * 12
    n_end = (2015 - tp[0]) * 12

    # Pull the temperature data using the indexes & convert to celsius 
    xi = ncdata.variables['tas'][n_begin:n_end]
    xi = np.array(xi)
    xi = np.array(xi) - 272.15
    
    # Rotate and flip to get the eastern and western hemisphere oriented correctly
    #xi = np.rot90(xi, k = 2, axes = (1, 2))
    #xi = np.flip(xi, axis = 2)

    # Get longitude and latitude
    lat = np.array(ncdata.variables["lat"])
    lon = np.array(ncdata.variables["lon"])

    # Append to lists
    tas_data.append(xi)
    lat_data.append(lat)
    lon_data.append(lon)


#----------------------------------------------------------
# ERA5 Pull
#----------------------------------------------------------
era5path = "/home/johnyannotty/NOAA_DATA/ERA5/"
#ncdata = nc.Dataset(era5path + 'era5_avg_mon_tas/data_1990-2023.nc', "r")
#ncdata = nc.Dataset(era5path + 'era5_land_avg_mon_tas_1990-2023/data.nc', "r")
y = ncdata.variables['t2m']
era5_lon = np.array(ncdata.variables['longitude'])
era5_lat = np.array(ncdata.variables['latitude'])


# Need to do this in batches.....
np.array(y[0][0][:5,:5]) - 272.15


# print(dset)
# dset["tas"][0][0].values

# dset.attrs.keys()
# dset.coords["time"].as_numpy().shape
# dset.coords["lat"].as_numpy().shape
# dset.coords["lon"].as_numpy().shape
# xx = dset["tas"].as_numpy()
# xx.shape



# lat = dset['tas']["lat"].as_numpy()
# lon = dset['tas']["lon"].as_numpy()
# tm = dset['tas']["time"].as_numpy()


# tas = dset['tas'].where(dset.time.isin(cftime.DatetimeProlepticGregorian(1950, 1, 16, 12, 0, 0, 0, 2, 15)), drop=True)
# tas.plot()
# plt.show()

# #tas = dset['tas'].sel(time=cftime.DatetimeNoLeap(1850, 1, 16, 12, 0, 0, 0, 2, 15))
# tas_np = tas.to_numpy()
# tas_np.size
# tas_np[10].size

# del tas
# del dset




# #----------------------------------------------------------
# # TA from BCC
# #----------------------------------------------------------
# cmip6path = "/home/johnyannotty/NOAA_DATA/CMIP6/"
# #filename = "tas_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
# filename =  "ta_Amon_BCC-CSM2-MR_hist-nat_r1i1p1f1_gn_197001-200912.nc"
# dset = xr.open_dataset(cmip6path+filename, decode_times=True, use_cftime=True)
# print(dset)
# dset["ta"][0][0].plot(cmap = 'coolwarm')
# dset["ta"][0][0].values



# dset.attrs.keys()
# dset.attrs["tracking_id"]
# dset.coords["time"].as_numpy().shape
# dset.coords["plev"].as_numpy().shape
# dset.coords["lat"].as_numpy().shape
# dset.coords["lon"].as_numpy().shape
# xx = dset["ta"].as_numpy()




# #dset['tas'].sel(time=cftime.DatetimeNoLeap(1850, 1, 15, 12, 0, 0, 0, 2, 15)).plot(cmap = 'coolwarm')
# dset['ta'].sel(time=cftime.DatetimeNoLeap(1970, 1, 16, 12, 0, 0, 0)).plot()
# plt.show()


# lat = dset['ta']["lat"].as_numpy()
# lon = dset['ta']["lon"].as_numpy()
# tm = dset['ta']["time"].as_numpy()

# #tas = dset["tas"].to_numpy()
# #tas[0].size

# tas = dset['tas'].sel(time=cftime.DatetimeNoLeap(1850, 1, 15, 12, 0, 0, 0, 2, 15))
# tas_np = tas.to_numpy()
# tas_np.size
# tas_np[10].size

# del tas
# del dset

# #----------------------------------------------------------
# # Old Code

# # necessary url
# #url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"

# #resp = requests.get(url)
# #data = json.loads(resp.text)
# #print(data)
# #pd.DataFrame(data["attributes"])


# os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "on"
# conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)

# query = conn.new_context(
#     latest = True,
#     project='CMIP6',
#     source_id='CanESM5',
#     experiment_id='historical,ssp119,ssp126,ssp245,ssp370,ssp585',
#     variable_id='tas,tasmax,tasmin',
#     table_id="Amon",
#     member_id='r1i1p1f1')

# results = query.search()
# len(results)
# query.hit_count

# hit = results[0].file_context().search()
# files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)

# files = list(files)

# stop = 5
# #stop = len(results)
# files_list = []
# for i in tqdm(range(stop)):
#     hit = results[i].file_context().search()
#     files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)
#     files_list.append(list(files))


# pd.DataFrame(files_list).loc[2,0]
# pd.DataFrame(files_list).loc[2,1]


# query = conn.new_context(
#     latest = True,
#     project='PMIP3',
#     source_id='CanESM5',
#     experiment_id='historical,ssp119,ssp126,ssp245,ssp370,ssp585',
#     variable_id='tas,tasmax,tasmin',
#     table_id="Amon",
#     member_id='r1i1p1f1')

# results = query.search()
# len(results)
# query.hit_count

# hit = results[0].file_context().search()
# files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)

# files = list(files)

# stop = 5
# #stop = len(results)
# files_list = []
# for i in tqdm(range(stop)):
#     hit = results[i].file_context().search()
#     files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)
#     files_list.append(list(files))

# files_list[0]


