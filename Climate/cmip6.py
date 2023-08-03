"""
https://claut.gitlab.io/man_ccia/lab2.html
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



cmip6path = "/home/johnyannotty/NOAA_DATA/CMIP6/"
#filename = "tas_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
filename =  "ta_Amon_BCC-CSM2-MR_hist-nat_r1i1p1f1_gn_197001-200912.nc"
dset = xr.open_dataset(cmip6path+filename, decode_times=True, use_cftime=True)
print(dset)
dset["ta"][0][0].plot(cmap = 'coolwarm')
dset["ta"][0][0].values



dset.attrs.keys()
dset.attrs["tracking_id"]
dset.coords["time"].as_numpy().shape
dset.coords["plev"].as_numpy().shape
dset.coords["lat"].as_numpy().shape
dset.coords["lon"].as_numpy().shape
xx = dset["ta"].as_numpy()




#dset['tas'].sel(time=cftime.DatetimeNoLeap(1850, 1, 15, 12, 0, 0, 0, 2, 15)).plot(cmap = 'coolwarm')
dset['ta'].sel(time=cftime.DatetimeNoLeap(1970, 1, 16, 12, 0, 0, 0)).plot()
plt.show()


lat = dset['ta']["lat"].as_numpy()
lon = dset['ta']["lon"].as_numpy()
tm = dset['ta']["time"].as_numpy()

#tas = dset["tas"].to_numpy()
#tas[0].size

tas = dset['tas'].sel(time=cftime.DatetimeNoLeap(1850, 1, 15, 12, 0, 0, 0, 2, 15))
tas_np = tas.to_numpy()
tas_np.size
tas_np[10].size

del tas
del dset


# necessary url
#url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"

#resp = requests.get(url)
#data = json.loads(resp.text)
#print(data)
#pd.DataFrame(data["attributes"])


os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "on"
conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)

query = conn.new_context(
    latest = True,
    project='CMIP6',
    source_id='CanESM5',
    experiment_id='historical,ssp119,ssp126,ssp245,ssp370,ssp585',
    variable_id='tas,tasmax,tasmin',
    table_id="Amon",
    member_id='r1i1p1f1')

results = query.search()
len(results)
query.hit_count

hit = results[0].file_context().search()
files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)

files = list(files)

stop = 5
#stop = len(results)
files_list = []
for i in tqdm(range(stop)):
    hit = results[i].file_context().search()
    files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)
    files_list.append(list(files))


pd.DataFrame(files_list).loc[2,0]
pd.DataFrame(files_list).loc[2,1]




query = conn.new_context(
    latest = True,
    project='PMIP3',
    source_id='CanESM5',
    experiment_id='historical,ssp119,ssp126,ssp245,ssp370,ssp585',
    variable_id='tas,tasmax,tasmin',
    table_id="Amon",
    member_id='r1i1p1f1')

results = query.search()
len(results)
query.hit_count

hit = results[0].file_context().search()
files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)

files = list(files)

stop = 5
#stop = len(results)
files_list = []
for i in tqdm(range(stop)):
    hit = results[i].file_context().search()
    files = map(lambda f : {'filename': f.filename, 'url': f.download_url}, hit)
    files_list.append(list(files))

files_list[0]


