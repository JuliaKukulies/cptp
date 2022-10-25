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
path = Path('/glade/scratch/kukulies/tracks/water-year/precip/')

precip_path = Path('/glade/scratch/kukulies/WY2020/pr/')
precip_path2 = Path('/glade/u/home/kukulies/data/obs/gpm/2019-2020/')

simulated_mcs= list(path.rglob('*0[5-7]*WRF*mcs.pickle'))
observed_mcs= list(path.rglob('*0[5-7]*GPM*mcs.pickle')) 
observed_mcs.sort()
simulated_mcs.sort()


### calculate contribution to total summer precipitation ### 
simulated_objects= list(path.glob('*0[5-7]*WRF.nc'))
observed_objects= list(path.glob('*0[5-7]*GPM.nc')) 

simulated_prec= list(precip_path.glob('*0[5-7]_regridded.nc'))
observed_prec= list(precip_path2.glob('??????/*0[5-7]*small.nc')) 
simulated_prec.sort()
observed_prec.sort()
simulated_objects.sort()
observed_objects.sort()



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
    wrf_objects = wrf_objects.rename({'yc': 'lat', 'xc':'lon'})
    gpm_objects = gpm_objects.rename({'yc': 'lat', 'xc':'lon'})
    wrf_data = xr.open_dataset(simulated_prec[idx]).pr
    gpm_data = xr.open_dataset(observed_prec[idx]).precipitationCal.resample(time ='1H').mean('time').transpose('time', 'lon', 'lat')
    
    # loop through MCS features and add in each iteration data that is not 0 (all other features are set to 0)
    for object_id in np.unique(np.array(list(wrf.keys())).astype(int)):
        ID = int(object_id) + 1
        if ID == np.unique(np.array(list(wrf.keys())).astype(int))[0] + 1 :
            contribution_wrf = wrf_data.where( wrf_objects.values == ID, 0 ).sum('time').values
        contribution_wrf += wrf_data.where( wrf_objects.values == ID, 0 ).sum('time').values

    for object_id in np.unique(np.array(list(gpm.keys())).astype(int)):
        ID = int(object_id) + 1
        if ID == np.unique(np.array(list(gpm.keys())).astype(int))[0] + 1 :
            contribution_gpm = gpm_data.where( gpm_objects.values == ID, 0 ).sum('time').values
        contribution_gpm += gpm_data.where( gpm_objects.values == ID, 0 ).sum('time').values


# save computed contribution to precip 
out = '/glade/scratch/kukulies/tracks/water-year/precip/'
x_gpm = xr.DataArray(contribution_gpm)
x_gpm.to_netcdf(out + 'mcs_contribution_gpm.nc')
x_wrf = xr.DataArray(contribution_wrf)
x_wrf.to_netcdf(out + 'mcs_contribution_wrf.nc') 

 
