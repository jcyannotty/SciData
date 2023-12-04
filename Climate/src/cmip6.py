"""
https://claut.gitlab.io/man_ccia/lab2.html

https://nordicesmhub.github.io/NEGI-Abisko-2019/training/CMIP6_example.html

https://netcdf-scm.readthedocs.io/en/latest/usage/ocean-data.html

"""

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
# np.where(xi[100] == fv + 272.15)

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

# Need to do this in batches.....
# np.array(y[0][0][:5,:]) - 272.15

era5 = xr.open_dataset(era5path + 'era5_avg_mon_tas/data_1990-2023.nc', decode_times=True, use_cftime=True)

fig, ax = plt.subplots(1,1,figsize = (15,10)) 
xx = np.where(era5["t2m"][:,0,:,:].time.isin(cftime.DatetimeGregorian(2014, 12, 1, 0, 0, 0, 0, 2, 15)))[0][0]
era5["t2m"][xx,0,:,:].plot(cmap = "coolwarm", ax = ax, vmin = 200, vmax = 325)
#era5["t2m"][xx,0,:,:].plot(cmap = "coolwarm", ax = ax, vmin = -75, vmax = 50)
plt.show()

#era5["t2m"][xx,0,:,:] = era5["t2m"][xx,0,:,:] - 273.15

#----------------------------------------------------------
# CMIP6 bilinear interpolations 
#----------------------------------------------------------
cmip6intpath = '/home/johnyannotty/NOAA_DATA/CMIP6_Interpolations/'
f = file_list[0]
#f = f.split("185")[0] + "2005-2015.nc"
f = f.split("185")[0] + "2015.nc"
ncfile = nc.Dataset(cmip6intpath + f,mode='w',format='NETCDF4_CLASSIC')

lat_dim = ncfile.createDimension('lat', era5_lat.shape[0])     # latitude axis
lon_dim = ncfile.createDimension('lon', era5_lon.shape[0])    # longitude axis
time_dim = ncfile.createDimension('time', 12)#tas_data[0].shape[0]) # unlimited axis (can be appended to).

ncfile.title=f
ncfile.subtitle='TAS Interpolations'

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.long_name = 'latitude'

lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.long_name = 'longitude'

time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'months since 1950-01-01'
time.long_name = 'time'

temp = ncfile.createVariable('tas',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
temp.units = 'C'
temp.standard_name = 't2m'


#num_tpds = tas_data[0].shape[0] # number of time periods
num_tpds = 12 # Last year worth of data
for i in tqdm(range(num_tpds),desc = "Time Period",leave=False):
    tm = tas_data[0].shape[0] - num_tpds + i
    bilinear_interp = interp2d(np.array(lon_data[0]),np.array(lat_data[0]), tas_data[0][tm], kind = "linear")
    temp[i,:,:] = np.flip(bilinear_interp(np.array(era5_lon), np.array(era5_lat)),axis = 0)

ncfile.close()

#yy = bilinear_interp(era5_lldf["lon"], era5_lldf["lat"])
bilinear_interp(np.array(era5_lon[0:3]), np.array(era5_lat[10:13]))
yy = bilinear_interp(np.array(era5_lon), np.array(era5_lat))
bilinear_interp([0,150,300], -85)
np.flip(bilinear_interp([0,150,300], [-85,85,0,40,90]),axis=0)


# NN interpolator and IDW (not saved as netcdf)
# Get cmpi6 mesh grid
cm6_llmesh = np.meshgrid(np.array(lon_data[0]),np.array(lat_data[0]))
cm6_lldf = pd.DataFrame()
cm6_lldf["lon"] = cm6_llmesh[0].ravel()
cm6_lldf["lat"] = cm6_llmesh[1].ravel()
cm6_lldf["t2m"] = tas_data[0][tm].ravel()
del cm6_llmesh

# Get era5 mesh grid
era5_llmesh = np.meshgrid(era5_lon,era5_lat)
era5_lldf = pd.DataFrame()
era5_lldf["lon"] = era5_llmesh[0].ravel()
era5_lldf["lat"] = era5_llmesh[1].ravel()
del era5_llmesh

# Set interpolators
nn_interp = KNeighborsRegressor(algorithm='kd_tree', n_neighbors=1, weights='uniform')
idw_interp = KNeighborsRegressor(algorithm='kd_tree', n_neighbors=10, weights='distance')

# Set training data (reference points) and interpolate
nn_interp.fit(cm6_lldf[["lon","lat"]],cm6_lldf["t2m"])
nn_pred = nn_interp.predict(era5_lldf)

idw_interp.fit(cm6_lldf[["lon","lat"]],cm6_lldf["t2m"])
iwd_pred = idw_interp.predict(era5_lldf)


# Plot the results
ncdata2 = nc.Dataset(cmip6intpath + f, "r")
ncdata2["tas"][0].shape

xx = np.where(era5["t2m"][:,0,:,:].time.isin(cftime.DatetimeGregorian(2014, 12, 1, 0, 0, 0, 0, 2, 15)))[0][0]
resid = era5["t2m"][xx,0,:,:] - 272.15 - ncdata2["tas"][11]

# Read data back in....
#ncdata2 = nc.Dataset(cmip6intpath + f, "r")

fig, ax = plt.subplots(1,3,figsize = (16,8)) 
fig, ax = plt.subplots(1,3,figsize = (16,8)) 
pcm0 = ax[0].pcolormesh(np.rot90(np.flip(era5["t2m"][xx,0,:,:].transpose() - 272.15),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
#ax[0][1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=2))
pcm1 = ax[1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
pcm2 = ax[2].pcolormesh(np.rot90(np.flip(resid.transpose()),k=3),cmap = "PRGn")
fig.colorbar(pcm0, ax = ax[0], location = "bottom")
fig.colorbar(pcm1, ax = ax[1], location = "bottom")
fig.colorbar(pcm2, ax = ax[2], location = "bottom")
ax[0].set_title("ERA 5 Predictions", size = 18)
ax[1].set_title("Access-CM2 Interpolations", size = 18)
ax[2].set_title("Residuals", size = 18)
plt.show()

pcm0 = ax[0].pcolormesh(np.rot90(np.flip(era5["t2m"][xx,0,:,:].transpose() - 272.15),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
#ax[0][1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=2))
pcm1 = ax[1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
pcm2 = ax[2].pcolormesh(np.rot90(np.flip(resid.transpose()),k=3),cmap = "PRGn")
fig.colorbar(pcm0, ax = ax[0], location = "bottom")
fig.colorbar(pcm1, ax = ax[1], location = "bottom")
fig.colorbar(pcm2, ax = ax[2], location = "bottom")
ax[0].set_title("ERA 5 Predictions", size = 18)
ax[1].set_title("Access-CM2 Interpolations", size = 18)
ax[2].set_title("Residuals", size = 18)
plt.show()

ncdata2.close()


cmip6intpath = '/home/johnyannotty/NOAA_DATA/CMIP6_Interpolations/'
cmpi6_interp_files = sorted(os.listdir(cmip6intpath))
ncdata1 = nc.Dataset(cmip6intpath + cmpi6_interp_files[0], "r")

#----------------------------------------------------------
# CMIP6 interpolations using custom functions
#----------------------------------------------------------
cmip6intpath = '/home/johnyannotty/NOAA_DATA/CMIP6_Interpolations'
f = file_list[0]
#f = f.split("185")[0] + "2005-2015.nc"
f = f.split("185")[0] + "2015.nc"
ncfile = nc.Dataset(cmip6intpath + f,mode='w',format='NETCDF4_CLASSIC')

lat_dim = ncfile.createDimension('lat', era5_lat.shape[0])     # latitude axis
lon_dim = ncfile.createDimension('lon', era5_lon.shape[0])    # longitude axis
time_dim = ncfile.createDimension('time', 12)#tas_data[0].shape[0]) # unlimited axis (can be appended to).

ncfile.title=f
ncfile.subtitle='TAS Interpolations'

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.long_name = 'latitude'

lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.long_name = 'longitude'

time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'months since 1950-01-01'
time.long_name = 'time'

temp = ncfile.createVariable('tas',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
temp.units = 'C'
temp.standard_name = 't2m'


# Batch Interpolations....(batch over time, long and lat)
batch_sz = 100 # number of columns per batch with 1440 rows 
ncol = era5_lon.shape[0]
#num_tpds = tas_data[0].shape[0] # number of time periods
num_tpds = 1 # last 10 years of data
era5_lat_flip = np.flip(era5_lat)

#temp[0,:,:] = map_coordinates(np.array(lon_data[0]),np.array(lat_data[0]))
i = 0
tm = tas_data[0].shape[0] - num_tpds + i
#bilin = interp2d(np.array(lon_data[0]),np.array(lat_data[0]), tas_data[0][tm], kind = "linear")
#xx = bilin(era5_lon, era5_lat)
#temp[i,:,:] = bilin(era5_lon, era5_lat)

for i in tqdm(range(num_tpds),desc = "Time Period",leave=False):
    tm = tas_data[0].shape[0] - num_tpds + i
    for b in tqdm(range(int(np.ceil(era5_lat.shape[0]/batch_sz))),desc = "Batch",leave=False):
        nrow = min(batch_sz, era5_lat.shape[0]-b*batch_sz)
        temp0 = np.array(-999.0).repeat(nrow*ncol).reshape(nrow,ncol)
        for j in range(ncol):
            x0 = era5_lon[j]
            for k in range(nrow):
                #y0 = era5_lat_flip[b*batch_sz+k]
                y0 = era5_lat[b*batch_sz+k]            
                temp0[k,j] = bilinear(x0, y0, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][tm])
        temp[i,range(100*b,nrow+100*b),:] = temp0

ncfile.close()

100*era5_lon.shape[0]

temp[i,:,range(100*b,ncol+100*b)].shape
temp0.shape



#ncfile.close()

ncdata2 = nc.Dataset(cmip6intpath + f, "r")
ncdata2["tas"][0].shape

xx = np.where(era5["t2m"][:,0,:,:].time.isin(cftime.DatetimeGregorian(2014, 12, 1, 0, 0, 0, 0, 2, 15)))[0][0]
era5["t2m"][xx,0,:,:].transpose() - 272.15 - ncdata2["tas"][0]
resid = era5["t2m"][xx,0,:,:] - 272.15 - ncdata2["tas"][0]

# Read data back in....
#ncdata2 = nc.Dataset(cmip6intpath + f, "r")

fig, ax = plt.subplots(1,3,figsize = (16,8)) 
pcm0 = ax[0].pcolormesh(np.rot90(np.flip(era5["t2m"][xx,0,:,:].transpose() - 272.15),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
#ax[0][1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=2))
pcm1 = ax[1].pcolormesh(np.rot90(np.flip(ncdata2['tas'][0].transpose()),k=3),cmap = "coolwarm",vmin = -50,vmax = 40)
pcm2 = ax[2].pcolormesh(np.rot90(np.flip(resid.transpose()),k=3),cmap = "PRGn")
fig.colorbar(pcm0, ax = ax[0], location = "bottom")
fig.colorbar(pcm1, ax = ax[1], location = "bottom")
fig.colorbar(pcm2, ax = ax[2], location = "bottom")
ax[0].set_title("ERA 5 Predictions", size = 18)
ax[1].set_title("Access-CM2 Interpolations", size = 18)
ax[2].set_title("Residuals", size = 18)
plt.show()




#----------------------------------------------------------
# Scratch
#----------------------------------------------------------
# nlon = lon_data[0].shape[0]
# nlat = lat_data[0].shape[0]
# lon_grid = lon_data[0].repeat(nlat).reshape(nlon,nlat)
# lat_grid = lat_data[0].repeat(nlon).reshape(nlat,nlon).transpose()
# plt.scatter(lon_grid,lat_grid)
# plt.show()


cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(1,2, figsize = (12,5))

pcm1 = ax[0].pcolormesh(tas_data[0][100],cmap = cmap)
ax[0].set_title("Original", size = 16)
ax[0].set(xlabel = "$x_1$", ylabel = "$x_2$")
#ax[0].xaxis.set_major_locator(ticker.FixedLocator(np.round(np.linspace(0, n_test, 6),3)))
#ax[0].xaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.linspace(-np.pi, np.pi, 6),3)))
#ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.round(np.linspace(0, n_test, 6),3)))
#ax[0].yaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.linspace(-np.pi, np.pi, 6),3)))
fig.colorbar(pcm1,ax = ax[0])

bilinear(271.3, 31.4, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100])




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




import geopandas as gpd
from shapely import Point

temp_data = pd.DataFrame(tas_data[1][200].reshape(tas_data[1][200].size))
temp_data.columns = ["tas"]

lat_vec = np.repeat(lat_data[1],lon_data[1].shape[0])
lon_vec = np.resize(lon_data[1],lat_data[1].shape[0]*lon_data[1].shape[0])
lon_lat_df = pd.DataFrame(np.concatenate([lat_vec,lon_vec]).reshape(2,int(len(lon_vec))).transpose())

temp_yr_geo_df = pd.concat([lon_lat_df,temp_data], axis = 1)
temp_yr_geo_df.rename(columns={0:"Lat",1:"Long"}, inplace = True)

geom = [Point(xy) for xy in zip(temp_yr_geo_df['Long'],temp_yr_geo_df['Lat'])]
gdf = gpd.GeoDataFrame(temp_data, geometry=geom)
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
gdf.plot(column = "tas", marker='o', markersize=15,
          cmap='viridis', legend=True, ax = world.plot(figsize=(15, 15), color = "lightgrey"),vmin = 200, vmax = 300,
          legend_kwds={"label": "Average Temperature in January 2005", "orientation": "horizontal","shrink":0.6}
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Average Monthly Temperatue in January ", size = 24)
plt.show()





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


