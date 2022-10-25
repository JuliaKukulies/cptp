import xarray as xr 
import numpy as np 
import pickle
from pathlib import Path 
import matplotlib.pyplot as plt 
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
import seaborn as sns 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# dictionarties with tracked cells during summer 
path = Path('/glade/work/kukulies/tracks/')
precip_path = Path('/glade/scratch/kukulies/WY2020/nudging/')
precip_path2 = Path('/glade/work/kukulies/obs/gpm/2019-2020/')
simulated_mcs= list(path.rglob('*0[6-8]*WRF*mcs.pickle'))
observed_mcs= list(path.rglob('*0[6-8]*CPC*mcs.pickle')) 
observed_mcs.sort()
simulated_mcs.sort()

### calculate contribution to total summer precipitation ### 
simulated_objects= list(path.glob('*0[6-8]*nudging*.nc'))
observed_objects= list(path.glob('*0[6-8]*CPC.nc')) 

simulated_prec= list(precip_path.glob('*0[6-8]_regridded.nc'))
observed_prec= list(precip_path2.glob('??????/*0[6-8]*small.nc')) 
simulated_prec.sort()
observed_prec.sort()
simulated_objects.sort()
observed_objects.sort()
month = 0 

for idx in range(len(simulated_objects)):
    print(simulated_objects[idx], simulated_prec[idx])
    print(simulated_prec[idx], observed_prec[idx])
    # open MCS dictionaries for respective month 
    fname = simulated_mcs[idx]
    with open(fname, 'rb') as f:
        wrf = pickle.load(f)
    fname = observed_mcs[idx]
    with open(fname, 'rb') as f:
        gpm = pickle.load(f)
    # and corresponding object nc files 
    wrf_objects = xr.open_dataset(simulated_objects[idx], decode_times = False).objects
    gpm_objects = xr.open_dataset(observed_objects[idx], decode_times = False).objects
    if '08' in str(observed_objects[idx]):
        gpm_objects = xr.open_dataset(observed_objects[idx], decode_times = False).objects[:-1]
    wrf_objects = wrf_objects.rename({'xc': 'lat', 'yc':'lon'})
    gpm_objects = gpm_objects.rename({'xc': 'lat', 'yc':'lon'})

   
    # binary mask with objects marked 1 and no objects marked 0 
    wrf_data = wrf_objects.where(wrf_objects.values > 0, 0 )
    wrf_data.data[wrf_data.data > 0] = 1 
    gpm_data = gpm_objects.where(gpm_objects.values > 0, 0 )
    gpm_data.data[gpm_data.data > 0] = 1 


    # loop through MCS features and add in each iteration data that is not 0 (all other features are set to 0)
    for object_id in np.unique(np.array(list(wrf.keys())).astype(int)):
        ID = int(object_id) + 1
        if ID == np.unique(np.array(list(wrf.keys())).astype(int))[0] + 1 and month ==0:
            contribution_wrf = wrf_data.where( wrf_objects.values == ID, 0 ).sum('time').values
        else:
            contribution_wrf += wrf_data.where( wrf_objects.values == ID, 0 ).sum('time').values

   
    for object_id in np.unique(np.array(list(gpm.keys())).astype(int)):
        ID = int(object_id) + 1
        if ID == np.unique(np.array(list(gpm.keys())).astype(int))[0] + 1 and month ==0:
            contribution_gpm = gpm_data.where( gpm_objects.values == ID, 0 ).sum('time').values
        else:
            contribution_gpm += gpm_data.where( gpm_objects.values == ID, 0 ).sum('time').values

    month = 1 
   
out = '/glade/scratch/kukulies/tracks/water-year/tbb/'
x_gpm = xr.DataArray(contribution_gpm)
x_gpm.to_netcdf(out + 'mcs_density_gpm2.nc')
x_wrf = xr.DataArray(contribution_wrf)
x_wrf.to_netcdf(out + 'mcs_density_wrf_nudging.nc') 

 
