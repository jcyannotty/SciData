import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm 
from shapely import Point

datapath = "/home/johnyannotty/NOAA_DATA/ghcnm_v4_01_20230730"
dname = "/ghcnm.tavg.v4.0.1.20230730.qcu.dat"
mdname = "/ghcnm.tavg.v4.0.1.20230730.qcu.inv"

# Read in temperature data
tmp_list = []
with open(datapath+dname,"r") as f:
    for row in tqdm(f):
        tmp = []
        tmp.append(row[0:11]) # Site ID 
        tmp.append(int(row[11:15])) # Year
        for j in range(12):
            tmp.append(int(row[(19+8*j):(24+8*j)]))
        tmp_list.append(tmp)
f.close()

temp_df = pd.DataFrame(tmp_list, columns = ["ID","Year"]+["Month"+str(i) for i in range(1,13)])
#temp_df["Year"].value_counts()


# Read in meta data
md_list = []
with open(datapath+mdname,"r") as fm:
    for row in tqdm(fm):
        md = row.split(" ")
        md = [txt for txt in md if not txt in ["","\n","*\n"]]
        if len(md) == 6:
            print(md)
        md_list.append(md)
fm.close()

meta_df = pd.DataFrame(md_list, columns = ["ID","Lat","Long","Station_Elevation","Station_Name"])

#TODO: Plots, Rescale Temp, bind with location, filter by date, output final df as pickle or csv
yr = 2020
temp_df_sub = temp_df[temp_df["Year"] == yr] 
temp_df_sub.iloc[:,2:] = temp_df_sub.iloc[:,2:]/100 # conver to celcius

temp_yr_geo_df = pd.merge(meta_df,temp_df_sub, on = "ID", how = "right")
#temp_yr_geo_df["ID"].isna().sum()
#temp_yr_geo_df["ID"].value_counts()

geom = [Point(xy) for xy in zip(temp_yr_geo_df['Long'],temp_yr_geo_df['Lat'])]
gdf = gpd.GeoDataFrame(temp_yr_geo_df, geometry=geom)
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
gdf.plot(column = "Month1", marker='o', markersize=15,
         cmap='viridis', legend=True, ax = world.plot(figsize=(15, 15), color = "lightgrey"),
         legend_kwds={"label": "Average Temperature in January 2005", "orientation": "horizontal","shrink":0.5}
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Average Monthly Temperatue in January "+ str(yr), size = 24)
plt.show()



# Plot temperature over time at a given loaction
temp_df.groupby("ID")
temp_df["ID"].nunique()

# Temps from 1970<=
temp_current_df = temp_df[temp_df["Year"] >= 1970]
temp_current_df
columbus = temp_current_df[temp_current_df["ID"] == "USW00093834"]
columbus_data = columbus[["Month"+str(i+1) for i in range(12)]].values.reshape(12*columbus.shape[0])
columbus_data = columbus_data[columbus_data != -9999]
columbus_data = columbus_data/100
#temp_current_df.isna().sum()

plt.scatter(range(columbus_data.size),columbus_data)
plt.xlabel("Month Number")
plt.ylabel("Average Monthly Temperature")
plt.title("Average Monthly Temperature: Columbus 1970-2013")
plt.show()

meta_df[(meta_df["Lat"].astype(float)>39)&(meta_df["Lat"].astype(float)<41)&\
         (meta_df["Long"].astype(float)>-83)&(meta_df["Long"].astype(float)<-82)]



from sklearn.linear_model import LinearRegression as LR
model = LR()
model.fit(np.array(list(range(columbus_data.size))).reshape(-1,1), columbus_data.reshape(-1,1))
model.coef_





gdf.plot(ax=world.plot(figsize=(15, 15), color = "lightgrey"), marker='o', c=temp_yr_geo_df["Month1"].astype(float), markersize=15, legend = True)
plt.title("Title", size = 28)
plt.colorbar()
plt.show()



norm = colors.Normalize(vmin=temp_yr_geo_df["Month1"].astype(float).min(), vmax=temp_yr_geo_df["Month1"].astype(float).max())
cbar = plt.cm.ScalarMappable(norm,cmap='viridis')
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax = plt.subplots(figsize = (15,7))
#ax=world.plot(figsize=(15, 15), color = "lightgrey")
gdf.plot(ax = world.plot(figsize=(15, 15), color = "lightgrey"), marker='o', c=temp_yr_geo_df["Month1"].astype(float), markersize=15,
         cmap='viridis', legend=True)
ax.set_title("Plot title")
fig.colorbar(cbar,ax=ax)
#fig.title("Title", size = 28)
plt.show()



import matplotlib.colors as colors

# generate data
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# # create the colorbar
norm = colors.Normalize(vmin=gdf.pop_est.min(), vmax=gdf.pop_est.max())
cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu')

# plot
fig, ax = plt.subplots(figsize=(15, 7))
# with no normalization
gdf.plot(column='pop_est', cmap='RdBu', legend=False, ax=ax)

# add colorbar
ax_cbar = fig.colorbar(cbar, ax=ax)
# add label for the colorbar
ax_cbar.set_label('I am a label')
plt.show()

plt.scatter(temp_yr_geo_df["Lat"],temp_yr_geo_df["Lat"],c=temp_yr_geo_df["Month1"])

f, ax = plt.subplots()

points = ax.scatter(temp_yr_geo_df["Lat"],temp_yr_geo_df["Long"], c=temp_yr_geo_df["Month1"].astype(float),cmap="plasma")
f.colorbar(points)
plt.show()

#row[20:24]
#row[28:32]
#row[28:33]
#row[0:11]
#row[19:24]
