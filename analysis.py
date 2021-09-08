### This script holds basic functions for the analysis and evaluation of the MCS case in July 2008

import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import cartopy 
import cartopy.crs as ccrs
from matplotlib import ticker, cm



def basin_mask(file, variable, coords = True):
    import geopandas
    import rioxarray
    import xarray
    from shapely.geometry import mapping
    
    basin = geopandas.read_file('yangtze/yangtze.shp', crs="epsg:4326")
    
    data = xr.open_dataset(file)

    if coords is False:
        data_vars= dict(pr=(["lat", "lon"], data.pr.values[0]))
        coords = dict( lat= data.lat.values[:,0], lon = data.lon.values[0])
        data= xr.Dataset(data_vars= data_vars, coords = coords)
        precip = data[variable]
        precip.rio.set_spatial_dims(x_dim=precip.dims[1], y_dim= precip.dims[0], inplace=True)
    else:
        precip = data[variable]
        precip.rio.set_spatial_dims(x_dim=precip.dims[1], y_dim= precip.dims[2], inplace=True)
    precip.rio.write_crs("epsg:4326", inplace=True)
    return precip.rio.clip(basin.geometry.apply(mapping), basin.crs, drop=False)










def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r



def get_precip(tpv_lon,tpv_lat,files):
    tpv_precip= np.zeros(73) 
    for time_idx in np.arange(73):
        # tpv coords 
        tpvlon = tpv_lon[time_idx]
        tpvlat = tpv_lat[time_idx]
        # model data 
        data = xr.open_dataset(files[time_idx])
        precip = data['pr'][0] * 3600 
        pr_lon = precip.lon
        pr_lat = precip.lat
        # get closest center 
        array = np.asarray(pr_lat.values.flatten())
        idx = (np.abs(array - tpvlat)).argmin()
        tlat = array[idx]
        array = np.asarray(pr_lon.values.flatten())
        idx = (np.abs(array - tpvlon)).argmin()
        tlon = array[idx] 
        distances = haversine(tlon, tlat, pr_lon, pr_lat)
        mask = (distances <= 300.0) 
        
        if time_idx == 0:
            tpv_precip[time_idx] = np.nanmean(precip.where(mask).values)
        else:
            tpv_precip[time_idx] = np.nanmean(precip.where(mask).values) + tpv_precip[time_idx - 1 ]
    return tpv_precip




def make_plot(x, y, data, title):

    cmap=plt.cm.magma
    r = np.arange(0,260,20)
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    fs = 25 

    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([70,115,25,38])
    xlon = [70,80,90,100,110]
    ylat = [25,30,35]
    m=ax.pcolormesh(x,y, data, norm = norm, cmap = cmap)
    ax.contour(lo,la, elevations.data.T, [3000], cmap = 'Greys')
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title(title, fontsize= fs)
    return m


def get_precip_gpm(tpv_lon,tpv_lat,files):
    tpv_precip= np.zeros(73) 
    for time_idx in np.arange(73):
        # tpv coords 
        tpvlon = tpv_lon[time_idx]
        tpvlat = tpv_lat[time_idx]
        # model data 
        data = xr.open_dataset(files[time_idx])
        precip = data['precipitationCal'][0] * 0.5
        pr_lon = precip.lon
        pr_lat = precip.lat
        # get closest center 
        array = np.asarray(pr_lat.values.flatten())
        idx = (np.abs(array - tpvlat)).argmin()
        tlat = array[idx]
        array = np.asarray(pr_lon.values.flatten())
        idx = (np.abs(array - tpvlon)).argmin()
        tlon = array[idx] 
        distances = haversine(tlon, tlat, pr_lon, pr_lat)
        mask = (distances <= 300.0) 
        
        if time_idx == 0:
            tpv_precip[time_idx] = np.nanmean(precip.where(mask).values)
        else:
            tpv_precip[time_idx] = np.nanmean(precip.where(mask).values) + tpv_precip[time_idx - 1 ]
    return tpv_precip



### get list with experiment names ###
def get_experiments(file_list, substring1, substring2):
    experiments = list()
    for i in file_list:
        s= str(i.stem)
        start = s.find(substring1) + len(substring1)
        end = s.find(substring2)
        substring = s[start:end]
        experiments.append(substring)
    return experiments
