'''

This python script shows an example of how to use cf-python (a python wrapper of the ESMF library) for conservative regridding.

In this example, postprocessed precipitation data from a 4km WRF simulation is used to be regridded onto the 0.1 x 0.1 deg 
grid of GPM IMERG. The regridded data is saved as a netCDF file which contains the same metadata as the input file plus information
on the regridding method.  

Email contact: julia.kukulies@gu.se

'''

import numpy as np 
import cf
import xarray as xr 

#########################################################
# user specific 

# WRF output to be regridded (here: one month of postprocessed hourly rain rates)
wrf_data = '/glade/scratch/kukulies/WY2020/pr/pr_CPTP-WY2020-4_ECMWF-ERA5_evaluation_r1i1p1_NCAR-WRF421P_v1_hour_2020-08.nc'
# GPM data subset to WRF domain, does not matter which time 
gpm_data = '/glade/u/home/kukulies/data/obs/gpm/2019-2020/202002/gpm_202002_d02.nc'
# filename of output file                                                                                                                                     
out = 'pr_CPTP-WY2020-4_ECMWF-ERA5_evaluation_r1i1p1_NCAR-WRF421P_v1_hour_2020-08_regridded.nc'
# specify output location 
dest = '/glade/scratch/kukulies/WY2020/'

#########################################################

# read in data as cf.Field objects 
gpm_cf = cf.read(gpm_data)[0][0] # only one time step is needed for GPM 
wrf_cf = cf.read(wrf_data)[0]

### set cell boundaries for GPM (needed only for conservative regridding, not for bilinear) ### 
gpm_lats = np.array(gpm_cf.coord('latitude').data)
gpm_lons = np.array(gpm_cf.coord('longitude').data)
lat_bounds = (gpm_lats[:-1] + gpm_lats[1:]) / 2
lat_bnds = np.vstack(( np.append(gpm_lats[0], lat_bounds), np.append(lat_bounds, gpm_lats[-1]) ))
lon_bounds = (gpm_lons[:-1] + gpm_lons[1:]) / 2 
lon_bnds = np.vstack(( np.append(gpm_lons[0], lon_bounds), np.append(lon_bounds, gpm_lons[-1]) ))

bounds = cf.Bounds()
bounds.set_properties({})
bounds.set_data(lat_bnds.T)
bounds.nc_set_variable('lat_bnds')
gpm_cf.coord('latitude').set_bounds(bounds)

bounds = cf.Bounds()
bounds.set_properties({})
bounds.set_data(lon_bnds.T)
bounds.nc_set_variable('lon_bnds')
gpm_cf.coord('longitude').set_bounds(bounds)

# fix coordinates in  WRF output to make them compatible with cf.Field format  
lats = np.array(wrf_cf.aux('ncvar%lat').data)[:,0]
lons = np.array(wrf_cf.aux('ncvar%lon').data)[0]
wrf_cf.del_construct('auxiliarycoordinate0')
wrf_cf.del_construct('auxiliarycoordinate1')

# dimension coordinate lat 
lat = cf.DimensionCoordinate()
lat.set_properties({'units': 'degrees_north', 'standard_name': 'latitude'})
lat.set_data(lats)
lat.nc_set_variable('lat')
# dimension coordinate lon 
lon = cf.DimensionCoordinate()
lon.set_properties({'units': 'degrees_east', 'standard_name': 'longitude'})
lon.set_data(lons)
lon.nc_set_variable('lon')
wrf_cf.set_construct(lat, axes=('domainaxis1',), key='dimensioncoordinate1', copy=False)
wrf_cf.set_construct(lon, axes=('domainaxis2',), key='dimensioncoordinate2', copy=False)


### set cell boundaries for  coordinates in  WRF (needed only for conservative regriddring, not for bilinear) ### 
lat_bounds = (lats[:-1] + lats[1:]) / 2
lat_bnds = np.vstack(( np.append(lats[0], lat_bounds), np.append(lat_bounds, lats[-1]) ))
lon_bounds = (lons[:-1] + lons[1:]) / 2 
lon_bnds = np.vstack(( np.append(lons[0], lon_bounds), np.append(lon_bounds, lons[-1]) ))

bounds = cf.Bounds()
bounds.set_properties({})
bounds.set_data(lat_bnds.T)
bounds.nc_set_variable('lat_bnds')
wrf_cf.coord('latitude').set_bounds(bounds)

bounds = cf.Bounds()
bounds.set_properties({})
bounds.set_data(lon_bnds.T)
bounds.nc_set_variable('lon_bnds')
wrf_cf.coord('longitude').set_bounds(bounds)


################################  perform 1st order conservative regridding ###########################################
regridded = wrf_cf.regrids(gpm_cf, method='conservative') 
wrf_regridded = np.array(regridded.data)

# bilinear version with same method:
#regridded= wrf_cf.regridc(gpm_cf, axes=('X','Y'), method='bilinear')


####################################### create a netCDF file with regridded field ######################################

# set coordinates and dimensions
data_vars = dict(pr=(["time", "lat", "lon"], wrf_regridded),)
times = xr.open_dataset(wrf_data).time.values
coords = dict(time=times, lat=gpm_lats, lon=gpm_lons,)
data = xr.Dataset(data_vars=data_vars, coords=coords)

# set attributes                                                                                                                                           
data['pr'].attrs = {'long name' : 'Precipitation','standard name' : 'precipitation_flux', 'units' : 'kg m-2 s-1', 'cell method' : 'time: mean'}
data['lat'].attrs = {'long name': 'latitude', 'unit': 'degrees north'}
data['lon'].attrs = {'long name': 'longitude', 'unit': 'degrees east'}
data.attrs = {'simulation': 'WRF 4km', 'forcing': 'ECMWF-ERA5', 'institute': 'National Center for Atmospheric Research, Boulder',
              'grid': 'interpolated on to GPM IMERG grid using conservative regridding with the ESMF library'}

# save to netcdf                                                                                                                                           
data.to_netcdf(dest + out )
 
