#!/usr/bin/env python

''' 
   Tracking_Functions.py

   This file contains the tracking fuctions for the object
   identification and tracking of precipitation areas, cyclones,
   clouds, and moisture streams

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
import matplotlib.path as mplPath
import sys
from calendar import monthrange

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
#import h5py
import xarray as xr
import netCDF4


# ==============================================================
# ==============================================================

def ObjectCharacteristics(PR_objectsFull, # feature object file
                         PR_orig,         # original file used for feature detection
                         SaveFile,        # output file name and locaiton
                         TIME,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,     # average grid spacing
                         MinTime=1,       # minimum lifetime of an object
                         Boundary = 1):   # 1 --> remove object when it hits the boundary of the domain

    import scipy
    import pickle
    
    nr_objectsUD=PR_objectsFull.max()
    rgiObjectsUDFull = PR_objectsFull
    if nr_objectsUD >= 1:
        grObject={}
        print('    Loop over '+str(PR_objectsFull.max())+' objects')
        for ob in range(int(PR_objectsFull.max())):
#             print('        process object '+str(ob+1)+' out of '+str(PR_objectsFull.max()))
            TT=(np.sum((PR_objectsFull == (ob+1)), axis=(1,2)) > 0)
            if sum(TT) >= MinTime:
                PR_objects=PR_objectsFull[TT,:,:]
                rgrObAct=np.array(np.copy(PR_orig[TT,:,:])).astype('float')
                rgrObAct[PR_objects != (ob+1)]=0
                rgiObjectsUD=rgiObjectsUDFull[TT,:,:]
                TimeAct=TIME[TT]

                # Does the object hit the boundary?
                rgiObjActSel=np.array(PR_objects == (ob+1)).astype('float')
                if Boundary == 1:
                    rgiBoundary=(np.sum(rgiObjActSel[:,0,:], axis=1)+np.sum(rgiObjActSel[:,-1,:], axis=1)+np.sum(rgiObjActSel[:,:,0], axis=1)+np.sum(rgiObjActSel[:,:,-1], axis=1) != 0)
                    rgrObAct[rgiBoundary,:,:]=np.nan
                    rgiObjActSel[rgiBoundary,:,:]=np.nan
                rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])
                rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(PR_objects.shape[0]-1)])*(Gridspacing/1000.)

                # plt.plot(rgrObjSpeed); plt.plot(SpeedRMSE); plt.plot(SpeedCorr); plt.plot(SpeedAverage, c='k', lw=3); plt.show()
                SpeedAverage=np.copy(rgrObjSpeed) #np.nanmean([rgrObjSpeed,SpeedRMSE,SpeedCorr], axis=0)
                rgrPR_Vol=(np.array([np.sum(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])/(12.*60.*5.))*Gridspacing**2
                rgrPR_Max=np.array([np.max(rgrObAct[tt,:,:]) for tt in range(PR_objects.shape[0])])
                for tt in range(rgiObjectsUD.shape[0]):
                    if np.sum(rgiObjectsUD[tt,:,:] == (ob+1)) >0:
                        PR_perc=np.percentile(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)], range(101))
                        PR_mean=np.mean(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)])
                    else:
                        PR_mean=np.nan
                        PR_perc=np.array([np.nan]*101)
                    if tt == 0:
                        rgrPR_Percentiles=PR_perc[None,:]
                        rgrPR_Mean=PR_mean
                    else:
                        rgrPR_Percentiles=np.append(rgrPR_Percentiles,PR_perc[None,:], axis=0)
                        rgrPR_Mean=np.append(rgrPR_Mean,PR_mean)

                rgrSize=np.array([np.sum(rgiObjActSel[tt,:,:] == 1) for tt in range(rgiObjectsUD.shape[0])])*(Gridspacing/1000.)**2
                rgrSize[(rgrSize == 0)]=np.nan
#                 rgrAccumulation=np.sum(rgrObAct, axis=0)
                
                # Track lat/lon
                TrackAll = np.zeros((len(rgrMassCent),2)); TrackAll[:] = np.nan
                try:
                    FIN = ~np.isnan(rgrMassCent[:,0])
                    for ii in range(len(rgrMassCent)):
                        if ~np.isnan(rgrMassCent[ii,0]) == True:
                            TrackAll[ii,1] = Lat[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                            TrackAll[ii,0] = Lon[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                except:
                    stop()
    

                grAct={'rgrMassCent':rgrMassCent, 
                       'rgrObjSpeed':SpeedAverage,
                       'rgrPR_Vol':rgrPR_Vol,
                       'rgrPR_Percentiles':rgrPR_Percentiles,
                       'rgrPR_Max':rgrPR_Max,
                       'rgrPR_Mean':rgrPR_Mean,
                       'rgrSize':rgrSize,
#                        'rgrAccumulation':rgrAccumulation,
                       'TimeAct':TimeAct,
                       'rgrMassCentLatLon':TrackAll}
                try:
                    grObject[str(ob)]=grAct
                except:
                    stop()
                    continue
        if SaveFile != None:
            pickle.dump(grObject, open(SaveFile, "wb" ) )
        return grObject
    
    
# ==============================================================
# ==============================================================

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import xarray as xr

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# ==============================================================
# ==============================================================

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


# ==============================================================
# ==============================================================
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)   


# ==============================================================
# ==============================================================
def Feature_Calculation(DATA_all,    # np array that contains [time,lat,lon,Variables] with vars
                        Variables,   # Variables beeing ['V', 'U', 'T', 'Q', 'SLP']
                        dLon,        # distance between longitude cells
                        dLat,        # distance between latitude cells
                        Lat,         # Latitude coordinates
                        dT,          # time step in hours
                        Gridspacing):# grid spacing in m
    from scipy import ndimage
    
    
    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U')]*DATA_all[:,:,:,Variables.index('Q')])**2 + (DATA_all[:,:,:,Variables.index('V')]*DATA_all[:,:,:,Variables.index('Q')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U')]
    VV = DATA_all[:,:,:,Variables.index('V')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45#*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold

    SLP = DATA_all[:,:,:,Variables.index('SLP')]/100.
    # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
    SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
    # smoothign over 3000 x 3000 km and 78 hours
    SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    SLP_Anomaly = np.array(SLP_smooth-SLPsmoothAn)
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < -12 # 12 hPa depression
    HighPressure_annomaly = SLP_Anomaly > 12

    return Pressure_anomaly, Frontal_Diagnostic, VapTrans, SLP_Anomaly, vgrad, HighPressure_annomaly



# ==============================================================
# ==============================================================
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km



def ReadERA5(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly ERA5 data for one variable from NCAR's RDA archive in a region of interest.
    # ----------

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    Plevels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)
    
    # check if variable is defined
    if var == 'V':
        ERAvarfile = 'v.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'V'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'U':
        ERAvarfile = 'u.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'U'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'T':
        ERAvarfile = 't.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'T'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'ZG':
        ERAvarfile = 'z.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Z'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'VO':
        ERAvarfile = 'vo.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'VO'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'Q':
        ERAvarfile = 'q.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Q'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'SLP':
        ERAvarfile = 'msl.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.sfc/'
        NCvarname = 'MSL'
        PL = -1
    if var == 'IVTE':
        ERAvarfile = 'viwve.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVE'
        PL = -1
    if var == 'IVTN':
        ERAvarfile = 'viwvn.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVN'
        PL = -1

    print(ERAvarfile)
    # read in the coordinates
    ncid=Dataset("/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1])); DataAll[:]=np.nan
    tt=0
    
    for mm in range(len(TimeDD)):
        YYYYMM = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)
        YYYYMMDD = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)+str(TimeDD[mm].day).zfill(2)
        DirAct = Dir + YYYYMM + '/'
        if (var == 'SLP') | (var == 'IVTE') | (var == 'IVTN'):
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMM+'*.nc')
        else:
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMMDD+'*.nc')
        FILES = np.sort(FILES)
        
        TIMEACT = TIME[(TimeDD[mm].year == TIME.year) &  (TimeDD[mm].month == TIME.month) & (TimeDD[mm].day == TIME.day)]
        
        for fi in range(len(FILES)): #[7:9]:
            print(FILES[fi])
            ncid = Dataset(FILES[fi], mode='r')
            time_var = ncid.variables['time']
            dtime = netCDF4.num2date(time_var[:],time_var.units)
            TimeNC = pd.to_datetime([pd.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dtime])
            TT = np.isin(TimeNC, TIMEACT)
            if iRoll != 0:
                if PL !=-1:
                    try:
                        DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,:])
                    except:
                        stop()
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,:])
                ncid.close()
            else:
                if PL !=-1:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,iWest:iEeast])
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,iWest:iEeast])
                ncid.close()
            # cut out region
            if len(DATAact.shape) == 2:
                DATAact=DATAact[None,:,:]
            DATAact=np.roll(DATAact,iRoll, axis=2)
            if iRoll != 0:
                DATAact = DATAact[:,:,iWest:iEeast]
            else:
                DATAact = DATAact[:,:,:]
            try:
                DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
            except:
                continue
            tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon


def ConnectLon(Objects):
    for tt in range(Objects.shape[0]):
        EDGE = np.append(Objects[tt,:,-1][:,None],Objects[tt,:,0][:,None], axis=1)
        iEDGE = (np.sum(EDGE>0, axis=1) == 2)
        OBJ_Left = EDGE[iEDGE,0]
        OBJ_Right = EDGE[iEDGE,1]
        OBJ_joint = np.array([OBJ_Left[ii].astype(str)+'_'+OBJ_Right[ii].astype(str) for ii in range(len(OBJ_Left))])
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for ob in range(len(OBJ_unique)):
            ObE = int(OBJ_unique[ob].split('_')[1])
            ObW = int(OBJ_unique[ob].split('_')[0])
            Objects[Objects == ObE] = ObW
    return Objects


def ConnectLonOld(rgiObjectsAR):
    # connect objects allong date line
    for tt in range(rgiObjectsAR.shape[0]):
        for y in range(rgiObjectsAR.shape[1]):
            if rgiObjectsAR[tt, y, 0] > 0 and rgiObjectsAR[tt, y, -1] > 0:
#                 rgiObjectsAR[rgiObjectsAR == rgiObjectsAR[tt, y, -1]] = rgiObjectsAR[tt, y, 0]
                COPY_Obj_tt = np.copy(rgiObjectsAR[tt,:,:])
                COPY_Obj_tt[COPY_Obj_tt == rgiObjectsAR[tt, y, -1]] = rgiObjectsAR[tt, y, 0]
                rgiObjectsAR[tt,:,:] = COPY_Obj_tt
    return(rgiObjectsAR)


# In[228]:


### Break up long living cyclones by extracting the biggest cyclone at each time
def BreakupObjects(DATA,     # 3D matrix [time,lat,lon] containing the objects
                  MinTime,    # minimum volume of each object
                  dT):       # time step in hours

    Objects = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(24/dT) # min livetime of object to be split
    AVmax = 1.5

    rgiObj_Struct2D = np.zeros((3,3,3)); rgiObj_Struct2D[1,:,:]=1
    rgiObjects2D, nr_objects2D = ndimage.label(DATA, structure=rgiObj_Struct2D)

    rgiObjNrs = np.unique(DATA)[1:]
    TT = np.array([Objects[ob][0].stop - Objects[ob][0].start for ob in range(MaxOb)])
    # Sel_Obj = rgiObjNrs[TT > MinLif]


    # Average 2D objects in 3D objects?
    Av_2Dob = np.zeros((len(rgiObjNrs))); Av_2Dob[:] = np.nan
    ii = 1
    for ob in range(len(rgiObjNrs)):
#         if TT[ob] <= MinLif:
#             # ignore short lived objects
#             continue
        SelOb = rgiObjNrs[ob]-1
        DATA_ACT = np.copy(DATA[Objects[SelOb]])
        iOb = rgiObjNrs[ob]
        rgiObjects2D_ACT = np.copy(rgiObjects2D[Objects[SelOb]])
        rgiObjects2D_ACT[DATA_ACT != iOb] = 0

        Av_2Dob[ob] = np.mean(np.array([len(np.unique(rgiObjects2D_ACT[tt,:,:]))-1 for tt in range(DATA_ACT.shape[0])]))
        if Av_2Dob[ob] > AVmax:
            ObjectArray_ACT = np.copy(DATA_ACT); ObjectArray_ACT[:] = 0
            rgiObAct = np.unique(rgiObjects2D_ACT[0,:,:])[1:]
            for tt in range(1,rgiObjects2D_ACT[:,:,:].shape[0]):
                rgiObActCP = list(np.copy(rgiObAct))
                for ob1 in rgiObAct:
                    tt1_obj = list(np.unique(rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt-1,:] == ob1])[1:])
                    if len(tt1_obj) == 0:
                        # this object ends here
                        rgiObActCP.remove(ob1)
                        continue
                    elif len(tt1_obj) == 1:
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == tt1_obj[0]] = ob1
                    else:
                        VOL = [np.sum(rgiObjects2D_ACT[tt,:] == tt1_obj[jj]) for jj in range(len(tt1_obj))]
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == tt1_obj[np.argmax(VOL)]] = ob1
                        tt1_obj.remove(tt1_obj[np.argmax(VOL)])
                        rgiObActCP = rgiObActCP + list(tt1_obj)

                # make sure that mergers are assigned the largest object
                for ob2 in rgiObActCP:
                    ttm1_obj = list(np.unique(rgiObjects2D_ACT[tt-1,rgiObjects2D_ACT[tt,:] == ob2])[1:])
                    if len(ttm1_obj) > 1:
                        VOL = [np.sum(rgiObjects2D_ACT[tt-1,:] == ttm1_obj[jj]) for jj in range(len(ttm1_obj))]
                        rgiObjects2D_ACT[tt,rgiObjects2D_ACT[tt,:] == ob2] = ttm1_obj[np.argmax(VOL)]


                # are there new object?
                NewObj = np.unique(rgiObjects2D_ACT[tt,:,:])[1:]
                NewObj = list(np.setdiff1d(NewObj,rgiObAct))
                if len(NewObj) != 0:
                    rgiObActCP = rgiObActCP + NewObj
                rgiObActCP = np.unique(rgiObActCP)
                rgiObAct = np.copy(rgiObActCP)

            rgiObjects2D_ACT[rgiObjects2D_ACT !=0] = np.copy(rgiObjects2D_ACT[rgiObjects2D_ACT !=0]+MaxOb)
            MaxOb = np.max(DATA)

            # save the new objects to the original object array
            TMP = np.copy(DATA[Objects[SelOb]])
            TMP[rgiObjects2D_ACT != 0] = rgiObjects2D_ACT[rgiObjects2D_ACT != 0]
            DATA[Objects[SelOb]] = np.copy(TMP)

    # clean up object matrix
    Unique = np.unique(DATA)[1:]
    Objects=ndimage.find_objects(DATA)
    rgiVolObj=np.array([np.sum(DATA[Objects[Unique[ob]-1]] == Unique[ob]) for ob in range(len(Unique))])
    TT = np.array([Objects[Unique[ob]-1][0].stop - Objects[Unique[ob]-1][0].start for ob in range(len(Unique))])

    # create final object array
    CY_objectsTMP=np.copy(DATA); CY_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiVolObj)):
        if TT[ob] >= MinTime/dT:
            CY_objectsTMP[DATA == Unique[ob]] = ii
            ii = ii + 1
            
    # lable the objects from 1 to N
    DATA_fin=np.copy(CY_objectsTMP); DATA_fin[:]=0
    Unique = np.unique(CY_objectsTMP)[1:]
    ii = 1
    for ob in range(len(Unique)):
        DATA_fin[CY_objectsTMP == Unique[ob]] = ii
        ii = ii + 1
        
    return DATA_fin
