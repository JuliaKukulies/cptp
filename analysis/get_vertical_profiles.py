### This python script creates vertical profiles based on WRF output ### 

import numpy as np 
from pathlib import Path 
import xarray as xr 
import wrf 
from netCDF4 import Dataset
from metpy.units import units 
import metpy.calc as mcalc

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

exp_names = ['WRF8km', 'WRF8km_vertical', 'WRF4km_L', 'WRF4km_spectral']

for day in [19, 20,21]:
    # WRF output files for day XX
    files = list()
    files.append(Path('/glade/scratch/kukulies/CPTP/8km_ref/wrfout/wrfout_d01_2008-07-' +str(day)+ '_01:00:00' ))
    files.append(Path('/glade/scratch/kukulies/CPTP/8km_vertical/wrfout/wrfout_d01_2008-07-' +str(day)+ '_01:00:00' ))
    files.append(Path('/glade/campaign/mmm/c3we/prein/CPTP/data/4km_MCS_L/wrfout/wrfout_d01_2008-07-'+str(day)+'_01:00:00'))
    files.append(Path('/glade/scratch/kukulies/CPTP/4km_spectral/wrfout/wrfout_d01_2008-07-'+str(day)+'_01:00:00'))
    # loop through different experiments
    for idx in np.arange(len(exp_names)):
        print('....start processing day', str(day), 'for experiment',exp_names[idx]) 
        data = files[idx]
        ds = Dataset(data)
        xrds = xr.open_dataset(data)
        # get pressure for each eta level 
        wrf_pressure = (xrds['PB'] + xrds['P'] ) * 0.01 
        wrf_pressure = wrf_pressure.mean('Time')
        # coordinates, define subregions (over which lats to average)
        wrf_lats = xrds.XLAT
        wrf_lons = xrds.XLONG
        minlat,maxlat = 32, 36
        minlon,maxlon = 75, 110
        minlat = find_nearest_idx(wrf_lats[0, :,0].values, minlat)
        maxlat = find_nearest_idx(wrf_lats[0, :,0].values, maxlat)
        minlon = find_nearest_idx(wrf_lons[0, 0,:].values, minlon)
        maxlon = find_nearest_idx(wrf_lons[0, 0,:].values, maxlon)

        # get diagnostica variables from WRF 
        uwind= wrf.getvar(ds, 'ua', timeidx = wrf.ALL_TIMES)
        vwind = wrf.getvar(ds, 'va', timeidx = wrf.ALL_TIMES)
        wwind = wrf.getvar(ds, 'wa', timeidx = wrf.ALL_TIMES) 
        theta = wrf.getvar(ds, 'theta', timeidx = wrf.ALL_TIMES )
        geop= wrf.getvar(ds, 'geopotential', timeidx = wrf.ALL_TIMES) 

        # calculate squared brunt vais frequency 
        brunt_vais= mcalc.brunt_vaisala_frequency_squared(geop * units.meters, theta * units.K) 

        # get meridional cross section of pressure and variables 
        meridional = dict()
        meridional['pressure']= wrf_pressure.where((wrf_pressure.south_north> minlat) & (wrf_pressure.south_north <maxlat)& (wrf_pressure.west_east > minlon) & (wrf_pressure.west_east <maxlon), drop = True).mean('south_north') 
        meridional['uwind']= uwind.where((uwind.south_north > minlat) & (uwind.south_north <maxlat)& (uwind.west_east > minlon) & (uwind.west_east <maxlon), drop = True).mean('south_north')
        meridional['vwind'] = vwind.where((vwind.south_north > minlat) & (vwind.south_north <maxlat)& (vwind.west_east > minlon) & (vwind.west_east <maxlon), drop = True).mean('south_north')
        meridional['wwind']= wwind.where((wwind.south_north > minlat) & (wwind.south_north <maxlat)& (wwind.west_east > minlon) & (wwind.west_east <maxlon), drop = True).mean('south_north')
        #meridional['n_squared']= brunt_vais.where((wwind.south_north > minlat) & (wwind.south_north <maxlat)& (wwind.west_east > minlon) & (wwind.west_east <maxlon), drop = True).mean('south_north')

        # save xarrays to netCDF file 
        for key in meridional.keys():
            meridional[key].to_netcdf('/glade/scratch/kukulies/wrf_processed/vertical-cross-section-'+ key+'_' + exp_names[idx] +'_' +str(day) + '_hourly.nc')
        print('all data saved for ', exp_names[idx])

        ds.close()
    print('everything processed for day ', str(day))







