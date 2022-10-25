import numpy as np 
import pandas as pd 
import pickle as pickle 
import xarray as xr 
from pathlib import Path  
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage


minlat, maxlat = 28, 33
minlon, maxlon = 102, 105

### get statistics for filtered MCSs ###
# initialize counters 
mcs_seasonal = np.zeros((12,))
mcs_sichuan = np.zeros((12,))
lifetimes_mcs = np.array(())
hours_mcs = np.array(())
hours =  4
mon = 0 
path = Path('/glade/scratch/kukulies/tracks/water-year/tbb/')
mcs_files = list(path.glob('*nudging*precip-colocs'))
mcs_files.sort()

print(mcs_files)

for fname in mcs_files:
    sichuan = dict()
    newdict = dict()
    with open(fname, 'rb') as f:
        monthly = pickle.load(f)
    for key in monthly.keys():
        sichuan_flag = 0 
        # check condition on cloud and axis features by replacing entries with 1 and 0 
        pf_axis = (np.array(monthly[key]['major_axis_length']) > 100 ) * 1
        cloud_shield = (np.array(monthly[key]['rgrSize']) > 1e4 ) * 1   
        cold_core = (np.array(monthly[key]['rgrPR_Percentiles'][:, 0]) < 225 ) * 1 

        # test for continuity 
        regions = ndimage.find_objects(ndimage.label(pf_axis)[0])
        pf_regions = [np.sum(pf_axis[r]) for r in regions]
        regions = ndimage.find_objects(ndimage.label(cloud_shield)[1])
        cl_regions = [np.sum(cloud_shield[r]) for r in regions]
        regions = ndimage.find_objects(ndimage.label(cloud_shield)[1])
        cc = [np.sum(cold_core[r]) for r in regions]
            
        if np.array(cl_regions).sum() > 0 and np.array(pf_axis).sum() > 0:
            # check if more than 4 pf100km and 240K cloud shields exist:
            if np.array(cl_regions).max() > hours and np.array(cc).max() > hours and np.array(pf_regions).max() > hours:
            #np.array(monthly[key]['cloud_min']).min() < 225:
                mcs_seasonal[mon] += 1 
                newdict[key] = monthly[key].copy()
                # diurnal
                cell_hours= pd.DatetimeIndex(monthly[key]['TimeAct']).hour
                hours_mcs= np.append(hours_mcs, cell_hours) 
                # lifetime 
                lt = monthly[key]['TimeAct'].shape[0]
                lifetimes_mcs = np.append(lifetimes_mcs, lt )
                # filter out sichuan MCSs                                                              
                lats = monthly[key]['rgrMassCentLatLon'][:,1]                                                                                                
                lons = monthly[key]['rgrMassCentLatLon'][:,0]                                                                                                
                for idx, lat in enumerate(lats):
                    lon = lons[idx]
                    if lat > minlat and lat < maxlat and lon > minlon and lon < maxlon : 
                        sichuan[key]  = monthly[key].copy()
                        sichuan_flag  = 1 
                        break
                    else:
                        sichuan_flag  = 0 

                if sichuan_flag == 1:
                    mcs_sichuan[mon] +=1 
    # save MCS dict 
    with open(str(fname) + '-mcs.pickle', 'wb') as handle:
        pickle.dump(newdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(str(fname) + '_sichuan.pickle', 'wb') as h:
        pickle.dump(sichuan, h, protocol=pickle.HIGHEST_PROTOCOL)
    h.close()
    f.close()
    mon += 1 
