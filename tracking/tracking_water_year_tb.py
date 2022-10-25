### This script tracks precipitation cells > 3mm /hr in the TP region (as suggested in Feng et al., 2020). From this dataset, MCSs can be filtered out following additional criteria from Feng et al. (2020): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JD034202



# julia.kukulies@gu.se


from pathlib import Path
import numpy as np
from scipy import stats
import glob
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from scipy import ndimage
import pickle
import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#### speed up interpolation
import numpy as np
import xarray as xr
from netCDF4 import Dataset

#### Functions from "Tracking_Function.py" file
from Tracking_Functions import ObjectCharacteristics
from Tracking_Functions import haversine
from Tracking_Functions import BreakupObjects


DataName = 'CPC'
OutputFolder='/glade/scratch/kukulies/tracks/'  # you will have to modify this location
sPlotDir = OutputFolder

StartDay = datetime.datetime(2019, 10, 1,0)  #datetime.datetime(YYYY, MM, DD,HH)
StopDay = datetime.datetime(2020, 9, 30,23) 

dT = 1
TimeHH=pd.date_range(StartDay, end=StopDay, freq='h')
TimeMM=pd.date_range(StartDay, end=StopDay + datetime.timedelta(days=1), freq='m')
TimeBT = pd.date_range(StartDay, end=StopDay, freq='3h')
Time = pd.date_range(StartDay, end=StopDay, freq=str(dT)+'h')
Years = np.unique(TimeMM.year)
iHHall = np.array(range(len(TimeHH)))

rgiObj_Struct=np.ones((3,3,3))
threshold = 240 
MinTime = 4 # minimum time  of persistence in hours


from scipy.ndimage import generate_binary_structure
# 2D choose if features should only be connected in 2D and not over time
#structure = generate_binary_structure(2,2) 

# define structure for connecting features (3D)
structure = generate_binary_structure(3, 3 ) 

from pathlib import PurePath
path = Path('/glade/scratch/kukulies/tbb/')
sub_directories = list(path.glob('mergir*'))
sub_directories.sort()

minlon, maxlon = 65, 120 
minlat, maxlat = 20, 45 

REGION = [maxlat, minlon, minlat, maxlon] #  N, E, S, W
year = 2019
# read in monthly data 

for mon in np.arange(10,13):
    month = str(mon).zfill(2)
    print('retrieving data for... ' + month )
    data = xr.open_dataset(str(path) + '/mergir_'+ str(year)+ '-'+  month +'.nc4')
    # decrease domain size to Sichuan basin 
    DataAll =  data.Tb.where( (data.lat > minlat ) & (data.lat < maxlat )&(data.lon > minlon ) & (data.lon < maxlon ), drop = True)
    Lat = DataAll.lat.values
    Lon = DataAll.lon.values
    Lat_mean=np.mean(Lat)
    Lon_mean=np.mean(Lon)
    dLat_mean = np.abs(np.mean(Lat[1:]-Lat[:-1]))
    dLon_mean = np.mean(Lon[1:]-Lon[:-1])
    Gridspacing=haversine(Lon_mean, dLat_mean, Lon_mean+dLon_mean, dLat_mean+dLat_mean)*1000. #  horizontal grid spacing in m
    Time = data.time.values 
    Lon, Lat = np.meshgrid(Lon, Lat)
    print(DataAll.shape, Lon.shape, Lat.shape) 

    # perform feature tracking on monthly file 
    # convert xarray to numpy 
    DataAll = np.array(DataAll.values)
    DataAll[DataAll > threshold] = 0 
    DataAll[np.isnan(DataAll)] = 0

    rgiObjectsUD, nr_objectsUD = ndimage.label(DataAll, structure = structure)
    print('            '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsUD)
    rgiVolObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    TT_ZG = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(nr_objectsUD)]) * dT

    # create final object array (apply time threshold)
    ZG_objectsTMP=np.copy(rgiObjectsUD); ZG_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiVolObj)):
        if TT_ZG[ob] >= MinTime:
            ZG_objectsTMP[rgiObjectsUD == (ob+1)] = ii
            ii = ii + 1

    # lable the objects from 1 to N (rather than starting from 0; 0 is background)
    ZG_objects=np.copy(rgiObjectsUD); ZG_objects[:]=0
    Unique = np.unique(ZG_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        ZG_objects[ZG_objectsTMP == Unique[ob]] = ii
        ii = ii + 1

    ## break up long living objects by extracting largest object from each time step 
    objects_fin =  BreakupObjects(ZG_objects, MinTime, dT)
    #objects_fin = ZG_objects

    ###################### get and save  object characteristics ############################################
    grZGclonesPT = ObjectCharacteristics(objects_fin, # feature object file
                                     DataAll,         # original file used for feature detection
                                     OutputFolder+'Cloud_tracking_wateryear'+ str(year)+ month+'_CPC',
                                     Time,             # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing=Gridspacing,
                                    MinTime=MinTime, Boundary = 1) 

    ########################## save netCDF file with identified objects ####################################

    Time = pd.date_range(Time[0], Time[-1], freq= 'h')
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(OutputFolder+'Cloud_tracking_water-year_'+ str(year)+ month+'_CPC.nc','w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    ZG_real = dataset.createVariable('data', np.float32,('time','yc','xc'))
    ZG_obj = dataset.createVariable('objects', np.float32,('time','yc','xc'))

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    ZG_real.coordinates = "lon lat"
    ZG_obj.coordinates = "lon lat"

    lat[:] = Lat
    lon[:] = Lon
    ZG_real[:] = DataAll
    ZG_obj[:] =objects_fin
    times[:] = iTime

    dataset.close()

    print('all tracks saved.')




# if idx != 0 : 
#     DataAll = xr.concat( (DataAll,Data ),dim = 'time' )
# else: 
#     DataAll =  data.where( (data.lon > minlon ) & (data.lon < maxlon )&(data.lon > minlon ) & (data.lon < maxlon ), drop = True)
