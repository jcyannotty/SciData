"""
https://claut.gitlab.io/man_ccia/lab2.html

https://nordicesmhub.github.io/NEGI-Abisko-2019/training/CMIP6_example.html

https://netcdf-scm.readthedocs.io/en/latest/usage/ocean-data.html

"""

import pandas as pd
import numpy as np
import os

import xarray as xr 
import proplot as plot 
import matplotlib.pyplot as plt

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
    xi = np.array(xi) - 272.15
    
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

# Check for missing data
np.where(xi[100] == fv + 272.15)

#----------------------------------------------------------
# ERA5 Pull
#----------------------------------------------------------
era5path = "/home/johnyannotty/NOAA_DATA/ERA5/"
ncdata = nc.Dataset(era5path + 'era5_avg_mon_tas/data_1990-2023.nc', "r")
#ncdata = nc.Dataset(era5path + 'era5_land_avg_mon_tas_1990-2023/data.nc', "r")
y = ncdata.variables['t2m']
era5_lon = np.array(ncdata.variables['longitude'])
era5_lat = np.array(ncdata.variables['latitude'])

y.shape

# Need to do this in batches.....
np.array(y[0][0][:5,:]) - 272.15

era5 = xr.open_dataset(era5path + 'era5_avg_mon_tas/data_1990-2023.nc', decode_times=True, use_cftime=True)

fig, ax = plt.subplots(1,1,figsize = (15,10)) 
xx = np.where(era5["t2m"][:,0,:,:].time.isin(cftime.DatetimeGregorian(1990, 1, 1, 0, 0, 0, 0, 2, 15)))[0][0]
era5["t2m"][xx,0,:,:].plot(cmap = "coolwarm", ax = ax, vmin = 200, vmax = 325)
#era5["t2m"][xx,0,:,:].plot(cmap = "coolwarm", ax = ax, vmin = -75, vmax = 50)
plt.show()

era5["t2m"][xx,0,:,:] = era5["t2m"][xx,0,:,:] - 272.15



# import geopandas as gpd
# from shapely import Point

# temp_data = pd.DataFrame(tas_data[1][200].reshape(tas_data[1][200].size))
# temp_data.columns = ["tas"]

# lat_vec = np.repeat(lat_data[1],lon_data[1].shape[0])
# lon_vec = np.resize(lon_data[1],lat_data[1].shape[0]*lon_data[1].shape[0])
# lon_lat_df = pd.DataFrame(np.concatenate([lat_vec,lon_vec]).reshape(2,int(len(lon_vec))).transpose())

# temp_yr_geo_df = pd.concat([lon_lat_df,temp_data], axis = 1)
# temp_yr_geo_df.rename(columns={0:"Lat",1:"Long"}, inplace = True)

# geom = [Point(xy) for xy in zip(temp_yr_geo_df['Long'],temp_yr_geo_df['Lat'])]
# gdf = gpd.GeoDataFrame(temp_data, geometry=geom)
# world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# gdf.plot(column = "tas", marker='o', markersize=15,
#          cmap='viridis', legend=True, ax = world.plot(figsize=(15, 15), color = "lightgrey"),vmin = 200, vmax = 300,
#          legend_kwds={"label": "Average Temperature in January 2005", "orientation": "horizontal","shrink":0.6}
# )
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Average Monthly Temperatue in January ", size = 24)
# plt.show()



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


