"""
cptp.plotting
========================

This script contains some functions used for plotting and data visualization of WRF output data.

"""
from pathlib import Path 
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import cartopy.crs as ccrs
import quiver 
import pandas as pd 
import shapely.geometry as sgeom
import seaborn as sns
from scipy import interpolate
import cartopy.feature as cfeature


# elevation data for plotting 
dem = '~/data/elevation_600x350.nc'
elevations = xr.open_dataarray(dem)
longitude = elevations.lon.values
latitude= elevations.lat.values
lo, la = np.meshgrid(longitude, latitude)


# ERA5 TPV track for plotting 
track_era5 = pd.read_table('/glade/scratch/kukulies/data/ERA5/ERA5_TPV_track_2008-07-18:21.txt', sep = '\s',names = ['time', 'lon', 'lat', 'geopotential'])
# interpolate time steps from 6-hourly to hourly 
f = interpolate.interp1d(track_era5.lon.values, track_era5.lat.values)
lonnew = np.linspace(track_era5.lon.values.min(), track_era5.lon.values.max(), 73)
latnew = f(lonnew)  
coords= {'case':[lonnew,latnew]}
lon_t= lonnew
lat_t=latnew

def plot_precip(extent, precip_data, acc_precip, era_precip_ds, acc_precip_era, out, xlon= None , ylat = None):
    """
    Make subplots of accumulated precipitation of WRF simulations in comparison with ERA5. 
    
    Args:
    extent: extent of map 
    precip_data(dict): dict with WRF experiment names and xarray data sets 
    acc_precip(dict): dict with WRF experiment names and computed accumulated precip in mm
    era_precip_ds(xarray.Dataset): ERA5 precipitation dataset 
    acc_precip_era(array): 2d field of computed accumulated precip in mm
    out(str): name of output file 
    xlon/ylat: array-like, used for xtick and ytick labels 
    
    """
    fig =plt.figure(figsize=(20,10))

    # customizing of colorbar 
    cmap=plt.cm.magma
    cmap.set_over(color='lightyellow')
    #cmap.set_under(color='white')
    r = np.array([0,5,10,15,20,30,40,60,80,100,150,200])
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    levels = [0,10,20,30,50,100,150,200,250,300,500] 
    fs= 25
    if xlon is None:
        xlon = [90,95,100,105,110,115]
        ylat = [25,30,35]
        
    ax = plt.subplot(3, 2, 1 , projection=ccrs.PlateCarree())
    if extent is not None:
        ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('ERA5', fontsize= fs)

    for idx in np.arange(5):
        # read in data 
        key =list(acc_precip.keys())[idx]
        data= acc_precip[key]
        lon = precip_data[key].lon
        lat = precip_data[key].lat
        # make subplot 
        ax = plt.subplot(3, 2, idx +2 , projection=ccrs.PlateCarree())
        if extent is not None: 
            ax.set_extent(extent)
        m=ax.pcolormesh(lon, lat, data,  cmap = cmap, norm = norm , vmin = 0)
        ax.coastlines(color = 'black')
        ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
        ax.set_title(key, fontsize= fs)
        ax.set_xticks(xlon)
        ax.set_xticklabels(xlon, fontsize= 14)
        ax.set_yticks(ylat)
        ax.set_yticklabels(ylat, fontsize= 14)
        data.close()

    cb_ax2 = fig.add_axes([0.92, 0.14,0.02, 0.75])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max', ticks = r, drawedges=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)

    cbar.set_label(r'accumulated precip [mm]', size=35)
    plt.rcParams.update({'font.size': 32})

    plt.savefig(Path('plots/') / out , transparent = None)
    plt.show()


    
    
def plot_ivt(era_wvflx, ivt_era, qu_era, qv_era, wrf_4km, ivt_wrf4km, qu_wrf4km, qv_wrf4km, wrf_4km_l, 
             ivt_wrf4km_l, qu_wrf4km_l, qv_wrf4km_l, wrf_12km, ivt_wrf12km, 
             qu_wrf12km, qv_wrf12km, extent = None, out = None):
    
    # elevation data for plotting 
    dem = '~/data/elevation_600x350.nc'
    elevations = xr.open_dataarray(dem)
    longitude = elevations.lon.values
    latitude= elevations.lat.values
    lo, la = np.meshgrid(longitude, latitude)
    
    fig =plt.figure(figsize=(22,14))
    r= np.array([10,20,30,40,50,100,150,200,250])

    # labels, fontsize, color 
    fs= 25
    c= 'white'
    # customizing of colorbar 
    cmap=plt.cm.viridis
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    if extent is None: 
        extent = [70,115,20,40]
        
    xlabels= list(np.arange(extent[0], extent[1]+5, 5))
    ylabels= list(np.arange(extent[2], extent[3]+5, 5))

    # Plot ERA5 moisture transport in map 
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('ERA5', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = np.meshgrid(era_wvflx.longitude.values, era_wvflx.latitude.values)
    skip  =(slice(None,None,3),slice(None,None,3))
    # Plot wind vectors 
    m= ax.pcolormesh(era_wvflx.longitude, era_wvflx.latitude , ivt_era, norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_era[skip], qv_era[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)


    # Plot WRF moisture transport in map 
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF4km', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_4km.XLONG.values, wrf_4km.XLAT.values
    skip  =(slice(None,None,20),slice(None,None,20))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf4km ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf4km[skip]*1.2, qv_wrf4km[skip]*1.2, color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs)

    # Plot WRF 4km large domain moisture transport in map 
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF4km large', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_4km_l.XLONG.values, wrf_4km_l.XLAT.values
    skip  =(slice(None,None,20),slice(None,None,20))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf4km_l ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf4km_l[skip]*1.2, qv_wrf4km_l[skip]*1.2, color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs)


    # Plot WRF moisture transport in map 
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF4km_5deg', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_12km.XLONG.values, wrf_12km.XLAT.values
    skip  =(slice(None,None,7),slice(None,None,7))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf12km ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf12km[skip], qv_wrf12km[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)

    # colorbar 
    cb_ax2 = fig.add_axes([0.92, 0.14,0.02, 0.72])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max', ticks = r, drawedges=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(r'IVT [kg m${-1}$ s${-1}$]', size=35)
    plt.rcParams.update({'font.size': fs})

    #plt.suptitle(str(qu_era.time.values)[0:-10], fontsize= 28)

    if out is None:
        out= 'plots/mcs-case_IVT.png'

    plt.savefig(out, transparent = False, bbox_inches = 'tight', face_color = 'white')
    plt.show()

    


    
def subplot_precip(extent, precip_data, acc_precip, era_precip_ds, acc_precip_era, out, xlon= None , ylat = None, r = np.array([0,5,10,15,20,30,40,60,80,100,150,200]), cmap = None ):
    """
    Make subplots of accumulated precipitation of WRF simulations in comparison with ERA5. 
    
    Args:
    extent: extent of map 
    precip_data(dict): dict with WRF experiment names and xarray data sets 
    acc_precip(dict): dict with WRF experiment names and computed accumulated precip in mm
    era_precip_ds(xarray.Dataset): ERA5 precipitation dataset 
    acc_precip_era(array): 2d field of computed accumulated precip in mm
    out(str): name of output file 
    xlon/ylat: array-like, used for xtick and ytick labels 
    
    """
    fig =plt.figure(figsize=(20,15))


    # customizing of colorbar 
    if cmap is None:
        cmap=plt.cm.magma
    cmap.set_over(color='lightyellow')
    #cmap.set_under(color='white')
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    levels = [0,10,20,30,50,100,150,200,250,300,500] 
    fs= 25
    if xlon is None:
        xlon = [90,95,100,105,110,115]
        ylat = [25,30,35]
        
    ax = plt.subplot(3, 2, 1 , projection=ccrs.PlateCarree())
    if extent is not None:
        ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('ERA5', fontsize= fs)

    for idx in np.arange(5):
        # read in data 
        key =list(acc_precip.keys())[idx]
        data= acc_precip[key]
        lon = precip_data[key].lon
        lat = precip_data[key].lat
        # make subplot 
        ax = plt.subplot(3, 2, idx + 2 , projection=ccrs.PlateCarree())
        if extent is not None: 
            ax.set_extent(extent)
        m=ax.pcolormesh(lon, lat, data,  cmap = cmap, norm = norm , vmin = 0)
        ax.coastlines(color = 'black')
        ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
        ax.set_title(key, fontsize= fs)
        ax.set_xticks(xlon)
        ax.set_xticklabels(xlon, fontsize= 14)
        ax.set_yticks(ylat)
        ax.set_yticklabels(ylat, fontsize= 14)
        data.close()

    cb_ax2 = fig.add_axes([0.92, 0.14,0.02, 0.75])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max', ticks = r, drawedges=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)

    cbar.set_label(r'accumulated precip [mm]', size=35)
    plt.rcParams.update({'font.size': 32})

    plt.savefig(Path('plots/') / out , transparent = None)
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_synoptic(time_idx, u4km, v4km, z4km, u4km_l, v4km_l, z4km_l, u12km, v12km, z12km, 
                 era_u, era_v, era_z, ds4km, ds4km_l, ds12km, era_ds, titles,
                  fs = None, extent = None, show = False, out = None ):
    # some options
    if fs is None:
        fs = 17
    if extent is None:
        extent = [70,115,25,37]
    #get 2d field for time step
    if time_idx is not None:
        era_z = era_z[time_idx] 
        era_u = era_u[time_idx]
        era_v = era_v[time_idx]

        z4km = z4km[:,:, time_idx] 
        u4km = u4km[:, :,time_idx]
        v4km = v4km[:, :,time_idx]

        z4km_l = z4km_l[:,:,time_idx] 
        u4km_l= u4km_l[:,:,time_idx]
        v4km_l = v4km_l[:,:,time_idx] 

        z12km = z12km[:,:,time_idx] 
        u12km = u12km[:,:,time_idx]
        v12km = v12km[:,:,time_idx]
    #############plot##########################
    plt.clf()
    
    fig =plt.figure(figsize=(20,8))
    
    # customizing of colorbar 
    cmap= plt.cm.magma
    r = np.arange(5760,5860,5)
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    r2 = np.arange(-50,55,5)
    norm2 = colors.BoundaryNorm(boundaries= r2,  ncolors= 256)

    # spacing of wind vectors
    x,y = np.meshgrid(era_u.longitude.values, era_u.latitude.values)
    skip  =(slice(None,None,3),slice(None,None,3))

    # Plot ERA5
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    timestr = era_ds.time.values[time_idx].astype(str)
    if time_idx is None:
        timestr= 'Mean'
    ax1.set_title('ERA5 ', fontsize= 28)
    ax1.set_extent(extent)
    ax1.coastlines(linewidth = 1.5)
    ax1.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')

    # Plot wind vectors 
    m= ax1.pcolormesh(x, y, era_z, norm=norm, cmap = cmap  )    
    ax1.quiver(x[skip], y[skip], era_u[skip], era_v[skip],color = 'grey', transform= ccrs.PlateCarree()) 
    lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)

    # TPV track 
    if time_idx is not None:
        for key in coords.keys():
            track_lons = coords[key][0]
            track_lats =  coords[key][1]
            track = sgeom.LineString(zip(track_lons, track_lats))
            ax1.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth= 5.0)

        # plot vorticity center of timestep 
        ax1.plot(lon_t[0], lat_t[0],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')
        ax1.plot(lon_t[time_idx], lat_t[time_idx],transform=ccrs.PlateCarree(), color='lightgrey', markersize=25,marker= 'o')
        ax1.plot(lon_t[-1], lat_t[-1],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')

    # axis labels 
    ax1.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax1.set_xlabel('Lon $^\circ$E',  fontsize=fs)
    
    # Plot WRF
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax.set_title(titles[0], fontsize= 28)
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')

    # Plot wind vectors 
    x,y = ds12km.lon.values, ds12km.lat.values
    skip  =(slice(None,None,20),slice(None,None,20))
    m2= ax.pcolormesh(x, y, z12km, norm=norm, cmap = cmap  )    
    ax.quiver(x[skip], y[skip], u12km[skip], v12km[skip],color = 'grey', transform= ccrs.PlateCarree()) 
    lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)

    # TPV track 
    if time_idx is not None:
        for key in coords.keys():
            track_lons = coords[key][0]
            track_lats =  coords[key][1]
            track = sgeom.LineString(zip(track_lons, track_lats))
            ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth= 5.0)

        # plot vorticity center of timestep 
        ax.plot(lon_t[0], lat_t[0],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')
        ax.plot(lon_t[time_idx], lat_t[time_idx],transform=ccrs.PlateCarree(), color='lightgrey', markersize=25,marker= 'o')
        ax.plot(lon_t[-1], lat_t[-1],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')

    # axis labels 
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs) 
    
    # Plot WRF
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax.set_title(titles[1], fontsize= 28)
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')

    # Plot wind vectors 
    x,y = ds4km_l.lon.values, ds4km_l.lat.values
    skip  =(slice(None,None,20),slice(None,None,20))
    m2= ax.pcolormesh(x,y, z4km_l, norm=norm, cmap = cmap  )    
    ax.quiver(x[skip], y[skip], u4km_l[skip], v4km_l[skip],color = 'grey', transform= ccrs.PlateCarree()) 
    lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)

    # TPV track 
    if time_idx is not None:
        for key in coords.keys():
            track_lons = coords[key][0]
            track_lats =  coords[key][1]
            track = sgeom.LineString(zip(track_lons, track_lats))
            ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth= 5.0)

        # plot vorticity center of timestep 
        ax.plot(lon_t[0], lat_t[0],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')
        ax.plot(lon_t[time_idx], lat_t[time_idx],transform=ccrs.PlateCarree(), color='lightgrey', markersize=25,marker= 'o')
        ax.plot(lon_t[-1], lat_t[-1],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')

    # axis labels 
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs) 
    
    
    # Plot WRF
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax.set_title(titles[2], fontsize= 28)
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')

    # Plot wind vectors 
    x,y = ds4km.lon.values, ds4km.lat.values
    skip  =(slice(None,None,20),slice(None,None,20))
    m2= ax.pcolormesh(x,y, z4km, norm=norm, cmap = cmap  )    
    ax.quiver(x[skip], y[skip], u4km[skip], v4km[skip],color = 'grey', transform= ccrs.PlateCarree()) 
    lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)

    # TPV track 
    if time_idx is not None:
        for key in coords.keys():
            track_lons = coords[key][0]
            track_lats =  coords[key][1]
            track = sgeom.LineString(zip(track_lons, track_lats))
            ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth= 5.0)

        # plot vorticity center of timestep 
        ax.plot(lon_t[0], lat_t[0],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')
        ax.plot(lon_t[time_idx], lat_t[time_idx],transform=ccrs.PlateCarree(), color='lightgrey', markersize=25,marker= 'o')
        ax.plot(lon_t[-1], lat_t[-1],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')

    # axis labels 
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs) 
    
    # colorbar
    cb_ax1 = fig.add_axes([0.93, 0.14,0.03, 0.74])
    cbar = fig.colorbar(m, cax=cb_ax1, extend = 'both', label ='500 hPa geopotential [m]')
    plt.rcParams.update({'font.size': 22})  
    

    plt.suptitle(timestr[:-10], fontsize= 28)
 
    if out is None:
        out = 'plots/'

    plt.savefig(out + 'synoptic_500hpa_'+ str(timestr)+'comparison_timelag.png', face_color ='w', transparent = False, bbox_inches = 'tight')
    if show is True:
        plt.show()
    


def plot_ivt(era_wvflx, ivt_era, qu_era, qv_era, wrf_4km, ivt_wrf4km, qu_wrf4km, qv_wrf4km, wrf_4km_l, 
             ivt_wrf4km_l, qu_wrf4km_l, qv_wrf4km_l, wrf_12km, ivt_wrf12km, 
             qu_wrf12km, qv_wrf12km, extent = None, out = None):
    
    # elevation data for plotting 
    dem = '~/data/elevation_600x350.nc'
    elevations = xr.open_dataarray(dem)
    longitude = elevations.lon.values
    latitude= elevations.lat.values
    lo, la = np.meshgrid(longitude, latitude)
    
    fig =plt.figure(figsize=(22,12))

    # labels, fontsize, color 
    fs= 25
    c= 'white'
    # customizing of colorbar 
    cmap=plt.cm.viridis
    r = np.array([0,10,20,30,40,50,100,150,200,250,300,350,400,600])
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    if extent is None: 
        extent = [70,115,20,40]
        
    xlabels= list(np.arange(extent[0], extent[1]+5, 5))
    ylabels= list(np.arange(extent[2], extent[3]+5, 5))

    # Plot ERA5 moisture transport in map 
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('ERA5', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = np.meshgrid(era_wvflx.longitude.values, era_wvflx.latitude.values)
    skip  =(slice(None,None,4),slice(None,None,4))
    # Plot wind vectors 
    m= ax.pcolormesh(era_wvflx.longitude, era_wvflx.latitude , ivt_era, norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_era[skip], qv_era[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)


    # Plot WRF moisture transport in map 
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF4km', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_4km.XLONG.values, wrf_4km.XLAT.values
    skip  =(slice(None,None,25),slice(None,None,25))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf4km ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf4km[skip], qv_wrf4km[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_ylabel('Lon $^\circ$E',  fontsize=fs)

    # Plot WRF 4km large domain moisture transport in map 
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF4km large', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_4km_l.XLONG.values, wrf_4km_l.XLAT.values
    skip  =(slice(None,None,25),slice(None,None,25))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf4km_l ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf4km_l[skip], qv_wrf4km_l[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_ylabel('Lon $^\circ$E',  fontsize=fs)


    # Plot WRF moisture transport in map 
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    plt.title('WRF12km', fontsize = fs)
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys')
    # spacing of wind vectors, convert lats and lons into 2d array 
    x,y  = wrf_12km.XLONG.values, wrf_12km.XLAT.values
    skip  =(slice(None,None,10),slice(None,None,10))
    # Plot wind vectors 
    m= ax.pcolormesh(x,y , ivt_wrf12km ,norm = norm, cmap = cmap  )    
    ax.quiver(x[skip],y[skip],qu_wrf12km[skip], qv_wrf12km[skip], color =c, transform= ccrs.PlateCarree()) 
    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)

    # colorbar 
    cb_ax2 = fig.add_axes([0.92, 0.14,0.02, 0.72])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max', ticks = r, drawedges=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(r'IVT [kg m${-1}$ s${-1}$]', size=35)
    plt.rcParams.update({'font.size': fs})

    plt.suptitle(str(qu_era.time.values)[0:-10], fontsize= 30)

    if out is None:
        out= 'plots/mcs-case_IVT.png'

    plt.savefig(out, bbox_inches = 'tight',transparent = False,   face_color = 'white')
    plt.show()

    
    
def plot_precip_ensemble(fnames, era_precip_ds, acc_precip_era, acc_precip_era_lsp, acc_precip_era_cp, out):
    """
    Make subplots of accumulated precipitation of WRF simulations in comparison with ERA5. 
    
    Args:
    extent: extent of map 
    fnames(list of str or path objects): list with filenames of WRF experiment 
    era_precip_ds(xarray.Dataset): ERA5 precipitation dataset 
    acc_precip_era(array): 2d field of computed accumulated precip in mm
    out(str): name of output file 
    sp(int): subplot number 
    
    """
    fig  = plt.figure(figsize=(20,35)) 
    
    # customizing of colorbar 
    cmap=plt.cm.plasma
    cmap.set_over(color='lightyellow')
    #cmap.set_under(color='white')
    r = np.array([0,5,10,15,20,30,40,60,80,100,150,200])
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    levels = [0,10,20,30,50,100,150,200,250,300,500] 
    fs= 20
    xlon = [90,100,110]
    ylat = [25,30,35]
    extent = [90,115,25,38]
    
    ################################# plot observation data in first row ############################
        
    subplots = int(len(fnames) / 4 ) + 5
    ax = plt.subplot(subplots, 4, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('ERA5', fontsize= fs)
    
    ax = plt.subplot(subplots, 4, 2, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era_lsp,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('ERA5 lsp', fontsize= fs)

    
    ax = plt.subplot(subplots, 4, 3, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era_cp,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('ERA5 cp', fontsize= fs)

    ax = plt.subplot(subplots, 4, 4, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    m=ax.pcolormesh(era_precip_ds.longitude, era_precip_ds.latitude, acc_precip_era,  cmap = cmap, norm = norm , vmin = 0)
    ax.coastlines(color = 'black')
    ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
    ax.set_xticks(xlon)
    ax.set_xticklabels(xlon, fontsize= 14)
    ax.set_yticks(ylat)
    ax.set_yticklabels(ylat, fontsize= 14)
    ax.set_title('GPM', fontsize= fs)

    ############################################# WRF experiments #######################################################
    for idx in np.arange(len(fnames)):
        # read in data 
        data = xr.open_dataset(fnames[idx]).pr[96:96*2] * 3600
        try:
            accumulated = data.sum('Time')
        except:
            accumulated= data.sum('time')
        ensemble_name = analysis.get_experiments([fnames[idx]], 'evaluation_', '_v1_hour_')[0]
        lon = data.lon.values
        lat = data.lat.values
        # make subplot 
        ax = plt.subplot(subplots, 4, idx + 5 , projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        m=ax.pcolormesh(lon, lat, accumulated,  cmap = cmap, norm = norm , vmin = 0)
        ax.coastlines(color = 'black')
        ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys', linewidths = [3.0])
        ax.set_title(ensemble_name, fontsize= fs)
        ax.set_xticks(xlon)
        ax.set_xticklabels(xlon, fontsize= 14)
        ax.set_yticks(ylat)
        ax.set_yticklabels(ylat, fontsize= 14)
        data.close()

    cb_ax2 = fig.add_axes([0.92, 0.35,0.02, 0.51])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max', ticks = r, drawedges=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)
    
    cbar.set_label(r'accumulated precip [mm]', size=35)
    plt.rcParams.update({'font.size': 32})
    plt.tight_layout()
    
    plt.savefig(Path('plots/') / out , transparent = None, bbox_inches = 'tight')
    plt.show()
    
    
    
    
def plot_synoptic_obs(time_idx, gpm , tb, era_u, era_v, era_z, era_ds,
                  fs = None, extent = None, show = False, out = None ):
    # some options
    if fs is None:
        fs = 22
    if extent is None:
        extent = [70,115,25,37] 
        
    xlabels=[80,90,100,110,115]
    ylabels= [25,30,35]
    
    #get 2d field for time step    
    era_z = era_z[time_idx] 
    era_u = era_u[time_idx]
    era_v = era_v[time_idx]
    
    precip = xr.open_dataset(gpm[time_idx])
    tbb= xr.open_dataset(tb[time_idx])
    #############plot##########################
    plt.clf()
    
    fig =plt.figure(figsize=(20,13))
    
    # customizing of colorbar 
    cmap1= plt.cm.Blues_r
    r = np.arange(0.5, 5.5, 0.5)
    norm1 = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    r = np.arange(230, 280, 5)
    norm2 = colors.BoundaryNorm(boundaries= r,  ncolors= 256)
    cmap2= plt.cm.Greys_r

    
    # customizing of colorbar 
    cmap3= plt.cm.magma
    r = np.arange(5760,5860,5)
    norm3 = colors.BoundaryNorm(boundaries= r,  ncolors= 256)


    # spacing of wind vectors
    x,y = np.meshgrid(era_u.longitude.values, era_u.latitude.values)
    skip  =(slice(None,None,3),slice(None,None,3))

    # Plot ERA5
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    timestr = era_ds.time.values[time_idx].astype(str)
    ax1.set_title('ERA5 ' + timestr, fontsize= 28)
    ax1.set_extent(extent)
    ax1.coastlines(linewidth = 1.5)
    ax1.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')

    # Plot wind vectors 
    s= ax1.pcolormesh(x, y, era_z, norm=norm3, cmap = cmap3  )    
    ax1.quiver(x[skip], y[skip], era_u[skip], era_v[skip],color = 'grey', transform= ccrs.PlateCarree()) 
    lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)

    # axis labels 
    ax1.set_xticks(xlabels, xlabels)
    ax1.set_yticks(ylabels,ylabels)
    ax1.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    
    ## observational plot 
    ax = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(linewidth = 1.5)
    ax.set_title('GPM & NCEP/CPC ', fontsize= 28)
    p = ax.pcolormesh(tbb.lon.values, tbb.lat.values, tbb.Tb.values ,norm = norm2, cmap = cmap2) 
    precipitation = precip.precipitationCal[0] * 0.5 
    gpm_mask= np.ma.masked_where( precipitation < 0.5,  precipitation )
    m = ax.pcolormesh(precip.lon.values, precip.lat.values,gpm_mask.T ,norm = norm1, cmap = cmap1)  
    ax.contour(elevations.lon.values,elevations.lat.values, elevations.data.T, [3000], cmap = 'Greys_r')
    
    # TPV track 
    for key in coords.keys():
        track_lons = coords[key][0]
        track_lats =  coords[key][1]
        track = sgeom.LineString(zip(track_lons, track_lats))
        ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='black', linewidth= 5.0)

    # plot vorticity center of timestep 
    ax.plot(lon_t[0], lat_t[0],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')
    ax.plot(lon_t[time_idx], lat_t[time_idx],transform=ccrs.PlateCarree(), color='crimson', markersize=25,marker= 'o')
    ax.plot(lon_t[-1], lat_t[-1],transform=ccrs.PlateCarree(), color='black', markersize=20,marker= 'o')

    # axis labels 
    ax.set_xticks(xlabels, xlabels)
    ax.set_yticks(ylabels,ylabels)
    ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
    ax.set_xlabel('Lon $^\circ$E',  fontsize=fs)
    
    # colorbars
    cb_ax = fig.add_axes([0.88, 0.53,0.02, 0.35])
    cbar = fig.colorbar(s, cax=cb_ax, extend = 'both', label ='500 hPa geopotential [m]')
    plt.rcParams.update({'font.size': 22})  
    
    
    # colorbars
    cb_ax1 = fig.add_axes([0.88, 0.11,0.02, 0.35])
    cbar1 = fig.colorbar(p, cax=cb_ax1, extend = 'both', label = 'Brightness temperature [K]')
    cb_ax2 = fig.add_axes([0.98, 0.11,0.02, 0.35])
    cbar2 = fig.colorbar(m, cax=cb_ax2, extend = 'both', label = 'Rain rate [mm h$^{-1}$]')

    cbar.outline.set_edgecolor('black')
    cbar1.outline.set_edgecolor('black')
    cbar2.outline.set_edgecolor('black')
    
    plt.rcParams.update({'font.size': 20}) 
    if out is None:
        out = 'plots/'

    plt.savefig(out + 'synoptic_500hpa_'+ str(timestr)+'comparison_satellite.png', face_color ='w', transparent = False, bbox_inches = 'tight')
    if show is True:
        plt.show()

    
# make map for each day to get an overview 
def plot_station_obs(station_data, days, extent, out ):
    """
    Creates subplots with all stations with precip record for specific days. 
    
    Args: 
        station_data(pd.DataFrame): pandas dataframe containing all data 
        days(list): list with days to plot
        extent(list): extent of map (minlon, maxlon, minlat, maxlat)
        xticks(list): xticks 
        yticks(list): yticks 
        elevations(xr.DataArray): elevation data
        out(str/path): output location and filename
        
    """
    
    import matplotlib.pyplot as plt 
    import matplotlib.colors as colors
    import cartopy.crs as ccrs
    
    fs = 22
    # precip colorbar norm 
    r = np.array([1, 2, 3, 4,  5, 10,15,20,25, 30,35, 40,45,50, 60, 80, 100, 150])
    norm = colors.BoundaryNorm(boundaries= r,  ncolors= 256)

    fig = plt.figure(figsize= (15,13))
    
    for dayidx in np.arange(len(days)): 
        # select day 
        day = days[dayidx]
        selected= station_data[station_data.DD == day]
        # only wet days  
        selected = selected[selected.Pinmm >= 1]
        # get number of rows (dependent on number of days)
        rows = int(len(days) / 2) 
        # change the 2 if you want more than 2 columns 
        ax = plt.subplot(rows, 2, dayidx +1 , projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.coastlines(color = 'black')
        ax.add_feature(cfeature.RIVERS)
        # plot elevation data 
        lo,la = np.meshgrid(elevations.lon.values,elevations.lat.values)
        ax.contour(lo, la, elevations.data.T, [3000], cmap = 'Greys_r', linewidths = [3.0])
        ax.pcolormesh(lo, la, elevations.data.T, cmap = 'Greys')
        ax.set_title(str(selected.YYYY.values[0])  + '-0' + str(selected.MM.values[0]) + '-' +str(selected.DD.values[0]), fontsize = fs)
        m= ax.scatter(station_data.lon.values, station_data.lat.values, s= 20, color = 'black', marker = 'v')
        m= ax.scatter(selected.lon.values, selected.lat.values, s= 37, c= selected.Pinmm.values, cmap= 'plasma', norm = colors.LogNorm( vmin=1, vmax=station_data.Pinmm.values.max() ))
        # set axis ticks and labels 
        xticks = list(np.arange(extent[0], extent[1]+5, 5)) 
        yticks = list(np.arange(extent[2], extent[3]+5, 5))
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize= 14)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize= 14) 
        ax.set_ylabel('Lat $^\circ$N',  fontsize=fs)
        if dayidx > 3: 
            ax.set_xlabel('Lon $^\circ$E',  fontsize=fs)
        
    cb_ax2 = fig.add_axes([0.92, 0.14,0.03, 0.72])
    cbar = fig.colorbar(m, cax=cb_ax2, extend = 'max')
    #cbar.outline.set_edgecolor('black')
    #cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(r'accumulated daily precip [mm d$^{-1}$]', size=fs*1.5)
    plt.savefig(out, bbox_inches = 'tight', transparent = False, facecolor = 'white')
    plt.show()
