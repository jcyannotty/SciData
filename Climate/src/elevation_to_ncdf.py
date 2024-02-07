"""
Name: elevation.py
Desc: Load elevation data and construct netcdf file
Refernces:
https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
"""

import pandas as pd
import numpy as np

import cftime
import xarray as xr 
import netCDF4 as nc

from Climate.src.spatial_functions import elevation_function

# Read in csv files
era5elevpath = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"
era5elevfiles = ["era5_elvations_west_hemisphere.csv","era5_elvations_east_hemisphere.csv"]

for i,ev in enumerate(era5elevfiles):
    if i == 0:
        dfc = pd.read_csv(era5elevpath + ev)
    else:
        df1 = pd.read_csv(era5elevpath + ev)
        dfc = pd.concat([dfc,df1])


# Sort dfc by lat and lon
dfc.sort_values(["lon","lat"], inplace=True)

# Create dataframe with all lon and lat - fill in 0 for sea level (only antartica is missing right now)
lon_min = -180; lon_max = 180; lon_num = (lon_max - lon_min)*4
lat_min = -90; lat_max = 90; lat_num = (lat_max - lat_min)*4

lon_list = np.linspace(lon_min,lon_max, lon_num+1).tolist()
lat_list = np.linspace(lat_min,lat_max, lat_num+1).tolist()
lon_lat = [[x,y] for x in lon_list for y in lat_list]
nlon = len(lon_list)
nlat = len(lat_list)

dfworld = pd.DataFrame(lon_lat, columns = ["lon","lat"])
dfworld = pd.merge(dfworld, dfc,on = ["lon","lat"], how = "left")
dfworld["elev"].fillna(0.0, inplace = True)


# Create updated netcdf
era5elevnc = "era5_elevations_12_31_23.nc"
ncfile = nc.Dataset(era5elevpath + era5elevnc,mode='w',format='NETCDF4_CLASSIC')

lat_dim = ncfile.createDimension('lat', len(lat_list))     # latitude axis
lon_dim = ncfile.createDimension('lon', len(lon_list))    # longitude axis

ncfile.title="ERA5-Elevations"

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.long_name = 'latitude'
lat[:] = lat_list

lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.long_name = 'longitude'
lon[:] = lon_list

elv = ncfile.createVariable('elev',np.float64,('lon','lat'))
elv.units = 'm'
elv.standard_name = 'Elevation'
elev = np.array(dfworld["elev"].to_list()).reshape(nlon,nlat)
elv[:,:] = elev

ncfile.close()


# Read in two different data sources
# from_df = True
# from_two = False
# if from_two:
#     if from_df:
#         # Concat two dfs
#         df1 = pd.read_csv(era5elevpath + "SWUSA.txt")
#         df2 = pd.read_csv(era5elevpath + "NA_notSW.txt")
#         dfc = pd.concat([df1,df2])
#     else:
#         # Merging two ncdfs
#         pass
# else:
#     dfc = pd.read_csv(era5elevpath + era5elevfile + ".csv")

# # Sort dfc by lat and lon
# dfc.sort_values(["lon","lat"], inplace=True)

# # Get lat and lon lists
# lat_list = dfc["lat"].sort_values().unique().tolist()
# lon_list = dfc["lon"].sort_values().unique().tolist()
# lon_lat = [[x,y] for x in lon_list for y in lat_list]
# nlon = len(lon_list)
# nlat = len(lat_list)


# # Get missing long and lats
# get_elev = False
# if dfc.shape[0] < nlon*nlat:
#     dfll = pd.DataFrame(lon_lat)
#     dfll.columns = ["lon","lat"]
#     dfmiss = pd.merge(dfll,dfc, on = ["lon","lat"], how = "left")
#     dfmiss = dfmiss[dfmiss["elev"].isna()].reset_index(drop = True)

#     # Get the elevation for the missing values
#     if get_elev:
#         evmiss = elevation_function(dfmiss[["lon","lat"]].values)
#         dfe = pd.DataFrame(evmiss)
#         dfc = pd.concat([dfc,dfe]).reset_index(drop = True)

#         # Sort and update the remaining info
#         dfc.sort_values(["lon","lat"], inplace=True)

#         # Get lat and lon lists
#         lat_list = dfc["lat"].sort_values().unique().tolist()
#         lon_list = dfc["lon"].sort_values().unique().tolist()
#     else:
#         dfmiss[["lon","lat"]].to_csv(era5elevpath + era5elevfile + "_missing.csv")



