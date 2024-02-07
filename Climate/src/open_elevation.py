"""
Name: elevation.py
Desc: Extract elevation from API
Refernces:
    https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
    https://github.com/Jorl17/open-elevation/blob/master/docs/api.md
    https://csidotinfo.wordpress.com/data/srtm-90m-digital-elevation-database-v4-1/
    https://www.pgc.umn.edu/data/rema/

    https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-near-surface-meteorological-variables?tab=form

"""

import urllib
import pandas as pd
import numpy as np
import json

import multiprocessing 
import time 


#from tqdm import tqdm
from itertools import product
from requests import Session
from tqdm import tqdm

#import importlib
#import Climate.src.world_regions
#importlib.reload(Climate.src.world_regions)

import Climate.src.world_regions as wr
from Climate.src.spatial_functions import open_elevation

#-----------------------------------------------------------
# Configure Water region
#-----------------------------------------------------------
lon_list = []
lat_list = []
water_lon_lat = []
for dim in wr.east_water:
    lon_min = dim["lon"][0] 
    lon_max = dim["lon"][1]
    lon_num = (lon_max - lon_min)*4
    lon_list = lon_list + np.linspace(lon_min,lon_max, lon_num+1).tolist()

    lat_min = dim["lat"][0] 
    lat_max = dim["lat"][1]
    lat_num = (lat_max - lat_min)*4
    lat_list = lat_list + np.linspace(lat_min,lat_max, lat_num+1).tolist()

    water_lon_lat = water_lon_lat + [[x,y] for x in lon_list for y in lat_list]

wdf = pd.DataFrame(water_lon_lat)
wdf.columns = ["lon","lat"]
wdf.drop_duplicates(inplace=True)
wdf.reset_index(drop = True, inplace=True)

water_lon_lat = wdf.values.tolist()

    
#-----------------------------------------------------------
# Configure Data region
#-----------------------------------------------------------
lon_list = []
lat_list = []
lon_lat = []
tag = "east_hemisphere"
for dim in wr.regions[tag]:
    lon_min = dim["lon"][0] 
    lon_max = dim["lon"][1]
    lon_num = (lon_max - lon_min)*4
    lon_list = lon_list + np.linspace(lon_min,lon_max, lon_num+1).tolist()

    lat_min = dim["lat"][0] 
    lat_max = dim["lat"][1]
    lat_num = (lat_max - lat_min)*4
    lat_list = lat_list + np.linspace(lat_min,lat_max, lat_num+1).tolist()

    lon_lat = lon_lat + [[x,y] for x in lon_list for y in lat_list]

len(lon_lat)
lldf = pd.DataFrame(lon_lat)
lldf.columns = ["lon","lat"]
lldf.drop_duplicates(inplace=True)
lldf.reset_index(drop = True, inplace=True)

coords = lldf.values.tolist()

#-----------------------------------------------------------
# Remove water regions
#-----------------------------------------------------------
#lldf = lldf.loc[~(lldf["lon"].isin(wdf.values.tolist()) & lldf["lat"].isin(wdf.values.tolist()))]
lldf_rm = lldf.merge(wdf,on=["lon","lat"])

lldf = lldf.drop(lldf_rm.index).reset_index(drop = True)
coords = lldf.values.tolist()
len(coords)

#-----------------------------------------------------------
# API Pull
#-----------------------------------------------------------
# Batch read
n = len(coords)
batchsz = 300
idx = np.array_split(range(n),batchsz)
out_elev = []
out_lat = []
out_lon = []
len(idx[0])
#idx = np.array_split(range(148708,n),batchsz)
for i in tqdm(idx):
    sz = i.size
    res = open_elevation(coords[i[0]:(i[sz-1]+1)])
    out_lon = out_lon + res["lon"]
    out_lat = out_lat + res["lat"]
    out_elev = out_elev + res["elev"]


#olon = []
#olat = []
#oelev = []
#for i in range(len(out_lat)):
#    olon = olon + out_lon[i]
#    olat = olat + out_lat[i]
#    oelev = oelev + out_elev[i]

elev_df = pd.DataFrame({"lon":out_lon,"lat":out_lat,"elev":out_elev})
#elev_df = pd.DataFrame({"lon":olon,"lat":olat,"elev":oelev})

# Write data
era5elevpath = "/home/johnyannotty/NOAA_DATA/ERA5_Elevations/"
elev_df.to_csv(era5elevpath + "era5_elvations_" +tag+".csv", index = False)


#-----------------------------------------------------------
# API Pull for any lon,lat pairs that are still missing
#-----------------------------------------------------------
# Get the missing values
len(out_elev)
missing_df = lldf.merge(elev_df, how = "left")
missing_df = missing_df[missing_df["elev"].isna()]

missing_coords = missing_df[["lon","lat"]].values.tolist()
nm = len(missing_coords)
batchsz = 10
idx = np.array_split(range(nm),batchsz)
miss_elev = []
miss_lat = []
miss_lon = []
for i in tqdm(idx):
    sz = i.size
    res = open_elevation(missing_coords[i[0]:(i[sz-1]+1)])
    miss_lon = miss_lon + res["lon"]
    miss_lat = miss_lat + res["lat"]
    miss_elev = miss_elev + res["elev"]

melev_df = pd.DataFrame({"lon":miss_lon,"lat":miss_lat,"elev":miss_elev})


total_elev_df = pd.concat([elev_df,melev_df]).reset_index(drop = True)
total_elev_df.to_csv(era5elevpath + "era5_elvations_" +tag+".csv", index = False)


#-----------------------------------------------------------
# Read in data
#-----------------------------------------------------------
wh_elev = pd.read_csv(era5elevpath + "era5_elvations_west_hemisphere.csv")
wh_elev[(wh_elev["lat"]>58) & (wh_elev["lat"]<59) & (wh_elev["lon"]>-135) & (wh_elev["lon"]<-133)]

eh_elev = pd.read_csv(era5elevpath + "era5_elvations_east_hemisphere.csv")
eh_elev[(eh_elev["lat"]>0) & (eh_elev["lat"]<45) & (eh_elev["lon"]>0) & (eh_elev["lon"]<40)]

lldf[(lldf["lat"]>0) & (lldf["lat"]<45) & (lldf["lon"]>0) & (lldf["lon"]<40)]