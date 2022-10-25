### This script filters out MCS based on tracked precipitation features. 
### The criteria proposed by Feng et al (2020) are applied, which means that co-locations between precip data and clouds are 
### first found, in order to check if the precipitation features also occur in conjunction with a cloud that has an area of at least 10^4km 2 < 240K 
### as well as a cold core with < 225 K for at least 4 hours.


# julia.kukulies@gu.se

import xarray as xr 
import numpy as np 
import pickle
from pathlib import Path 
import matplotlib.pyplot as plt 
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
 
import pandas as pd
import sys
sys.path.insert(1, '../../analysis/')
from cptp.analysis import get_tb
import warnings
warnings.filterwarnings('ignore')

### GPM and NCEP/CPC track files ### 
# dictionaries
path = Path('/glade/scratch/kukulies/tracks/water-year/tbb/')
features = list(path.glob('Cloud_tracking_water-year??????_spectral-nudging_WRF'))
features.sort()
# nc files with objects
precip_cells = list(path.glob('Cloud_tracking_water-year_??????_spectral-nudging_WRF.nc'))
precip_cells.sort()
# precip co-locations 
scratch = Path('/glade/scratch/kukulies/WY2020/nudging/')
clouds = list(scratch.glob('pr*regridded*nc'))
clouds.sort()

minlon, maxlon = 65, 120
minlat, maxlat = 20, 45
dx = 15.725

####### modify monthly characteristics of tracked convective features #####
for ii in np.arange(12):
    if Path(str(features[ii]) +  '_precip-colocs').exists() is False:
    # read in nc and pickle file for specific month 
        precip = xr.open_dataset(precip_cells[ii], decode_times = False)
        print('reading in .....', precip_cells[ii])
        with open(features[ii], 'rb') as f:
            cell_stats = pickle.load(f)
            f.close()
        # also get right tbb file for month 
        data= xr.open_dataset(clouds[ii ])
        print('corresponding brightness temperatures:', clouds[ii ])
        prec =data.pr.where( (data.lat > minlat ) & (data.lat < maxlat )&(data.lon > minlon ) & (data.lon < maxlon ), drop = True)    
        prec = np.array(prec.values) * 3600 
        # loop through all time steps 
        for tt in np.arange(precip.objects.shape[0]):
            # labeled image precip cells 
            label_img = precip.objects.values.astype(int)[tt]
            ################# collocate with precip ######################################
            pr = prec[tt] 
            assert label_img.shape == pr.shape 

            # identify largest connected  PF in specific timestep
            pr[pr <= 3] = 0 
            # get individual IDs for PF > 3mm/hr 
            prec_features, num_features = ndimage.label(pr)

            ############## identify axis length of precip object and convert to km #############
            props = regionprops_table(prec_features, properties=['label', 'major_axis_length'])
            properties = pd.DataFrame(props)

            # loop through individual cloud features (start from 1 because 0 is background)                                                                  
            for ob in np.unique(label_img)[1:]:
                total_precip = 0
                major_axis = 0 
                # get corresponding precip feature IDs                                                                                                        
                prec_ids  = np.unique(prec_features[ (label_img == ob) & (prec_features != 0) ])
                if prec_ids.sum() > 0:
                    # calculate area with nr. of grid cells connected with cloud feature and grid spacing                                                     
                    for ID in prec_ids:
                        total_precip += prec_features[prec_features == ID].flatten().sum()
                        length = properties[properties.label == ID ].major_axis_length.values[0] * dx
                        if length > major_axis:
                            major_axis= length
                object_id = str(ob - 1 )
                if object_id in cell_stats.keys():
                    if 'major_axis_length' not in cell_stats[object_id].keys():
                        cell_stats[object_id]['major_axis_length'] = []
                    cell_stats[object_id]['major_axis_length'].append(major_axis)
                    if 'total_precip' not in cell_stats[object_id].keys():
                        cell_stats[object_id]['total_precip'] = []
                    cell_stats[object_id]['total_precip'].append(total_precip)
        # write modified dict to file 
        out = str(features[ii]) +  '_precip-colocs'
        with open(out, 'wb') as handle:
            pickle.dump(cell_stats , handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('modified dictionary saved.')
    
