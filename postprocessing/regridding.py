#### This python script performs a regridding of WRF data onto the GPM grid using the ESMF library via cf-python. 

#julia.kukulies@gu.se
### 

import numpy as np 
import xarray as xr 
from pathlib import Path
import cf

year = 2020
for mon in [1]:
    month = str(mon).zfill(2)
    gpm = '/glade/work/kukulies/obs/gpm/2019-2020/' +str(year)+ month+ '/'
    wrf = '/glade/scratch/kukulies/WY2020/nudging/'
    #wrf_fname= 'olr_CPTP-WY2020-4_ECMWF-ERA5_evaluation_r1i1p1_NCAR-WRF421P_v1_hour_' + str(year)+'-'+month +'.nc'
    #wrf_precip = xr.open_dataset(wrf + wrf_fname).olr
    gpm_file = gpm  +  'gpm_'+ str(year)+ month + '_cat.nc_small.nc'
    wrf_file = wrf  +  'olr_CPTP-WY2020-4_ECMWF-ERA5_evaluation_r2i1p1_BNU-WRF421P_v1_hour_' +str(year)+ '-'+month +'.nc'

    gpm_grid = xr.open_dataset(gpm_file)
    times = xr.open_dataset(wrf_file).time.values
    gpm_cf = cf.read(gpm_file)
    wrf_cf = cf.read(wrf_file)
    gpm_in = gpm_cf[0]
    wrf_in = wrf_cf[0]
    #wrf_in.data = wrf_precip

    # apply regridding 
    regridded = wrf_in.regrids(gpm_in, 'bilinear', src_axes={'X': 1, 'Y': 0})
    wrf_regridded = np.array(regridded) 
    print('regridded WRF data for ' + str(year) + month + '....new shape:', wrf_regridded.shape)
    # save regridded data as netCDF file 
    # set coordinates and dimensions
                                                                                                                     
    data_vars = dict(olr=(["time", "lat", "lon"], wrf_regridded),)
    coords = dict(time=times, lat=gpm_grid.lat.values, lon=gpm_grid.lon.values,)
    data = xr.Dataset(data_vars=data_vars, coords=coords)

    # set attributes 
    data['olr'].attrs = {'long name' : 'Instantaneous Upwellng Longwave Flux at Top of Atmosphere','standard name' : 'upwelling_longwave_flux_TOA', 'units' : 'W m-2', 'cell method' : 'time: point'}
    data['lat'].attrs = {'long name': 'latitude', 'unit': 'degrees north'}
    data['lon'].attrs = {'long name': 'longitude', 'unit': 'degrees east'}
    data.attrs = {'simulation': 'WRF 4km', 'forcing': 'ECMWF-ERA5', 'institute': 'National Center for Atmospheric Research, Boulder', 'grid': 'interpolated on to GPM IMERG grid using bilinear interpolation with the ESMF library'}

    # fix filename 
    out = 'olr_CPTP-WY2020-4_ECMWF-ERA5_evaluation_r2i1p1_BNU-WRF421P_v1_hour_'+ str(year)+'-' +month + '_regridded.nc'
    # save to netcdf 
    data.to_netcdf("/glade/scratch/kukulies/WY2020/nudging/" + out )
    print('saved as ' , out )
