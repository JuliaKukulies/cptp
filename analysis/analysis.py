"""
cptp.analysis
========================

This script contains some basic functions for WRF output data processing, inspection and analysis.

"""
import numpy as np 
import xarray as xr 
from scipy import interpolate 
import netCDF4 as nc 
import scipy
import wrf


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_uvz(data, lev, timedim = 'time'):
    '''
    Getting the u and v wind components as well as geopotential height from WRF output on specific pressure level. 
    
    Args:
    data(xarray.Dataset): xarray dataset that contains data at pressure levels
    lev: pressure level in hpa 
    
    Returns:
    u, v, z: 2D fields of variables as xarray.DataArray 
    
    '''
    press= lev*100
    # get index of pressure level
    if timedim == 'time':
        idx = np.where(data.P_PL.sel(time = data.time.values[0]).where(data.P_PL == press) > 0)[0][0]
    if timedim == 'Time':
        idx = np.where(data.P_PL.sel(Time = data.Time.values[0]).where(data.P_PL == press) > 0)[0][0]
    if timedim is None:
        idx = np.where(data.P_PL.where(data.P_PL == press) > 0)[0][0]

    # reading in data for specific pressure level
    data = data.sel(num_press_levels_stag = idx)
    # get vars 
    u = data.U_PL
    v = data.V_PL
    z = data.GHT_PL
    
    return u, v, z 


def get_var(wrfin, var,lev,  timedim = 'time'):
    """
    Extracts diagnostic variable at specific pressure level from wrf output at model levels. 

    """
    from netCDF4 import Dataset
    wrfnc = Dataset(wrfin)
    data =wrf.getvar(wrfnc, var, timeidx = None)
    if data.ndim == 3:
        base_pressure = np.array(wrfnc['PB']).squeeze()
        perturbation_pressure = np.array(wrfnc['P']).squeeze()
        var= wrf.interplevel(data, (base_pressure + perturbation_pressure) * 0.01, lev)
    else:
        assert data.dims[1] == 'bottom_top'
        var = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
        for tt in np.arange(data.shape[0]):
            base_pressure = np.array(wrfnc['PB']).squeeze()[tt]
            perturbation_pressure = np.array(wrfnc['P']).squeeze()[tt]
            var[tt]= wrf.interplevel(data[tt], (base_pressure + perturbation_pressure) * 0.01, lev)
    return var



def get_tb(olr): 
    """
    This function converts outgoing longwave radiation to brightness temperatures. 
     using the Planck constant. 
     
    Args:
        olr(xr.DataArray or numpy array): 2D field of model output with OLR
        
    Returns:
        tb(xr.DataArray or numpy array): 2D field with estimated brightness temperatures
    """
    # constants 
    aa = 1.228 
    bb = -1.106 * 10**(-3) # K−1 
    # Planck constant
    sig = 5.670374419 * 10**(-8) # W⋅m−2⋅K−4 
    # flux equivalent brightness temperature 
    Tf = (abs(olr)/sig) ** (1./4) 
    tb = (((aa ** 2 + 4 * bb *Tf ) ** (1./2)) - aa)/(2*bb) 
    return tb




def wrf_vort( U, V, dx ):
    """
    Calculate the relative vorticity given the U and V vector components in m/s
    and the grid spacing dx in meters.
    
    U and V must be the same shape.
    ---------------------
    U (numpy.ndarray): ndarray of U vector values in m/s
    V (numpy.ndarray): ndarray of V vector values in m/s
    dx (float or int): float or integer of U and V grispacing in meters
    ---------------------
    returns:
        numpy.ndarray of vorticity values s^-1 same shape as U and V
    """
    assert U.shape == V.shape, 'Arrays are different shapes. They must be the same shape.'
    dy = dx
    du = np.gradient( U )
    dv = np.gradient( V )
    return ( dv[-1]/dx - du[-2]/dy )



def subset_domain(ds, minlon, maxlon, minlat, maxlat, lonname, latname):
    """
    Subsets the domain of an xarray to given extent. 
    
    Args:
    ds(xarray.Dataset): xarray dataset which to crop 
    lonname(str): name of longitude dimension in dataset
    latname(str): name of latitude dimension in dataset 
    
    """
    mask_lon = (ds[lonname] >= minlon) & (ds[lonname] <= maxlon)
    mask_lat = (ds[latname] >= minlat) & (ds[latname] <= maxlat)
    
    cropped_ds = ds.where(mask_lon & mask_lat, drop=True)
    
    return cropped_ds


def select_time(data, start, end, timedim, times):
    """                                                                                                                                                        
    Extracting specific time steps from input xarray data.                                                                                                    

    Note that time dimension needs to be redefined for this, since the WRF output dimension Time 
    only contains the indices and not the actual timestamps. 
                                                                                                                                                             
    Args:                                                                                                                                                      
    start(str): start time yyyy-mm-dd-hh                                                                                                                      
    end(str):end time yyyy-mm-dd-hh                                                                                                                         

    timedim(str): name of time dimension in dataset
    times(array-like): array or list with timestamps 
    """
    # use time variable as dimension                                                                                                                           
    data = data.swap_dims({timedim : "time"})
    data = data.assign_coords({'time': times})
    # subset dataset                                                                                                                                           
    data = data.sel(time = slice(start,end ))
    return data



def geopotential_to_height(z):
    """ This function converts geopotential heights to geometric heights. This approximation takes into account the varying gravitational force with heights, but neglects latitudinal vairations.
    Parameters:
    ------------
    z(float) : (1D or multi-dimenstional) array with geopotential heights
    Returns:
    ----------
    geometric_heights : array of same shape containing altitudes in metres
    """
    g = 9.80665 # standard gravity 
    Re = 6.371 * 10**6  # earth radius
    geometric_heights   = (z*Re) / (g * Re - z)
    return geometric_heights 



def get_surface_humidity(temperature, spressure):
    '''
    This function calculates near-surface humidity for ERA5 
    based on the 2m dew point temperature and surafce pressure. 
    Args: 
    temperature: 2D array with 2m dew point temperature 
    spressure: 2D array with surface pressure values in hpa
    Returns:
    q_sat: near surface humidity in kg/kg 
    '''
    
    #### define constants #### 

    # gas constants for dry air and water vapour in J K-1 kg-1
    Rdry= 287
    Rvap= 461
    # constants for Tetens formula (for saturation over water)
    c1= 611.21
    c2= 17.502
    c3= 32.19
    # freezing point
    T0 = 273.16 
    
    spressure = spressure*100
    e_sat = c1* np.exp( c2 * ((temperature - T0)/ (temperature - c3)))
    
    q_sat = ((Rdry / Rvap) * e_sat ) / (spressure - (1- Rdry/Rvap) * e_sat )
    return q_sat



def colint_pressure(values,pressure_levels):
    """ This function calculates the column-integrated water vapor
    in kg/m2 from specific humidity (kg/kg) at different hpa levels.
    """

    g = 9.18 # gravitational acceleration
    return np.trapz(values, pressure_levels, axis = 0)* g


def calculate_ivt(data):
    """
    Calculate vertically integrated water vapor transport from WRF output. 
    
    Args:
    data(xarray.Dataset): WRF output file as xarray 
    
    Returns:
    qu_int(xarray.Dataset): vertically integrated u component of water vapor flux 
    qv_int(xarray.Dataset): vertically integrated v component of water vapor flux
    IVT(np.array): 2D field of total amount of vertically integrated water vapor
    """
    
    # gravitational accelration 
    g = 9.8
    
    # get necessary variables
    uwind = data.U
    vwind = data.V
    # full pressure in hpa (base state pressure + perturbation pressure)
    pressure = (data.P +data.PB ) / 100 
    qvapor = data.QVAPOR # + data.QCLOUD to include cloud water 

    # interpolate u and v vectors to mass grid 
    vwind_mass = np.zeros(qvapor.shape)
    uwind_mass = np.zeros(qvapor.shape)

    for lev in uwind.bottom_top.values:
        vwind_mass[lev] = Vstagger_to_mass(vwind[lev].values)
        uwind_mass[lev] = Ustagger_to_mass(uwind[lev].values)
   
    # calculate fluxes 
    QU = uwind_mass * qvapor 
    QV = vwind_mass * qvapor 
    
    # column-integration over pressure (q needs to be in kg/kg)
    # if q is given in mass per volume -> use geometric heights 
    # set x pressure negative because values are decreasing along array
    qu_int = colint_pressure(QV , -pressure.values) 
    qv_int = colint_pressure(QU , -pressure.values)
 
    # compute total amount 
    IVT = np.sqrt(qu_int**2 + qv_int**2)
    
    return qu_int, qv_int, IVT 

    
# from https://github.com/blaylockbk/Ute_WRF/blob/master/functions/stagger_to_mass.py
def Vstagger_to_mass(V):
    """
    V are the data on the top and bottom of a grid box    
    A simple conversion of the V stagger grid to the mass points.
    Calculates the average of the top and bottom value of a grid box. Looping
    over all rows reduces the staggered grid to the same dimensions as the 
    mass point.
    Useful for converting V, XLAT_V, and XLONG_V to masspoints
    Differnce between XLAT_V and XLAT is usually small, on order of 10e-5
    
    (row_j1+row_j2)/2 = masspoint_inrow
    
    Input:
        Vgrid with size (##+1, ##)
    Output:
        V on mass points with size (##,##)
        
    """
    
    # create the first column manually to initialize the array with correct dimensions
    V_masspoint = (V[0,:]+V[1,:])/2. # average of first and second column
    V_num_rows = int(V.shape[0])-1 # we want one less row than we have
    
    # Loop through the rest of the rows
    # We want the same number of rows as we have columns.
    # Take the first and second row, average them, and store in first row in V_masspoint
    for row in range(1,V_num_rows):
        row_avg = (V[row,:]+V[row+1,:])/2.
        # Stack those onto the previous for the final array        
        V_masspoint = np.row_stack((V_masspoint,row_avg))
    
    return V_masspoint
    
def Ustagger_to_mass(U):
    """
    U are the data on the left and right of a grid box    
    A simple conversion of the U stagger grid to the mass points.
    Calculates the average of the left and right value of a grid box. Looping
    over all columns it reduces the staggered grid to the same dimensions as the 
    mass point.
    Useful for converting U, XLAT_U, and XLONG_U to masspoints
    Differnce between XLAT_U and XLAT is usually small, on order of 10e-5
    
    (column_j1+column_j2)/2 = masspoint_incolumn
    
    Input:
        Ugrid with size (##, ##+1)
    Output:
        U on mass points with size (##,##)
        
    """
    
    # create the first column manually to initialize the array with correct dimensions
    U_masspoint = (U[:,0]+U[:,1])/2. # average of first and second row
    U_num_cols = int(U.shape[1])-1 # we want one less column than we have
    # Loop through the rest of the columns
    # We want the same number of columns as we have rows.
    # Take the first and second column, average them, and store in first column in U_masspoint
    for col in range(1,U_num_cols):
        col_avg = (U[:,col]+U[:,col+1])/2.
        # Stack those onto the previous for the final array        
        U_masspoint = np.column_stack((U_masspoint,col_avg))
    
    return U_masspoint




def get_uv_massgrid_xray(wrfout):
    """
    Returns xarray with u and v wind components on a mass grid. 
    
    Args:
        wrfout(xarray.Dataset): WRF input data as xarray 
    
    """
    # interpolate u and v vectors to mass grid 
    vwind_mass = np.zeros(wrfout.QVAPOR.shape)
    uwind_mass = np.zeros(wrfout.QVAPOR.shape)

    for lev in wrfout.bottom_top.values:
        vwind_mass[lev] = Vstagger_to_mass(wrfout.V[lev])
        uwind_mass[lev] = Ustagger_to_mass(wrfout.U[lev])

    # xarray metadata 
    data_vars = dict(
        u=(["south_north", "west_east"], uwind_mass),
        v=(["south_north", "west_east"], vwind_mass),
    )
    coords = dict(
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    
    return dataset 



def calculate_vorticity(fname, lev, timeidx, squeeze = False):
    """
    Computes the absolute and relative vorticity for specific level and timestep. 
    
    Args:
        fname(str or path): filename of WRF input dataset
        lev(int): vertical pressure level in hpa
        
    Returns:
        avo_lev(xarray.DataArray): 2d field of absolute vorticity
        rv_lev(xarray.DataArray): 2d field of relative vorticity 
    
    """
    wrf_nc = nc.Dataset(fname)
    # compute absolute vorticity with wrf python module 
    avo = wrf.g_vorticity.get_avo(wrf_nc, timeidx )
    # get coriolis parameter 
    OMEGA = 7.292e-5
    if squeeze is True:
        lats = np.array(wrf_nc['XLAT'][0]).squeeze() 
        base_pressure = np.array(wrf_nc['PB'])[timeidx].squeeze()
        perturbation_pressure = np.array(wrf_nc['P'])[timeidx].squeeze()
   
    else:
        lats = np.array(wrf_nc['XLAT'][0]).squeeze()
        base_pressure = np.array(wrf_nc['PB']).squeeze()
        perturbation_pressure = np.array(wrf_nc['P']).squeeze()
   
    cor = 2 * OMEGA * np.sin(np.deg2rad(lats))
    # get relative vorticity
    assert cor.ndim == 2 
    rv = avo - cor 
    # WRF interpolation to specific pressure level 
    rv_lev= wrf.interplevel(avo, (base_pressure + perturbation_pressure) * 0.01, lev)
    avo_lev= wrf.interplevel(rv, (base_pressure + perturbation_pressure)  * 0.01, lev)
    
    return avo_lev, rv_lev 

                                                                                                  
def get_experiments(file_list, substring1, substring2):
    """
    extract experiment names from list with complete filenames
    """
    experiments = list()
    for i in file_list:
        s= str(i.stem)
        start = s.find(substring1) + len(substring1)
        end = s.find(substring2)
        substring = s[start:end]
        experiments.append(substring)
    return experiments


def get_stats(obs, models):
    """
    Calculates basic statistics between models and reference data based on common grid. 
    
    Args:
        obs(np.array): 2D field of observation/ reference dataset 
        models: list or array with 2D fields for each model 
    """
    # initiate arrays
    crmsd = np.array((0.0))
    ccoef = np.array((1.0))
    mean = np.array((np.nanmean(obs)))
    sdev = np.array((np.nanstd(obs)))
    
    for model in models: 
        # correlation coefficient
        ccoef= np.append(ccoef, np.ma.corrcoef(np.ma.masked_invalid(obs.flatten()),np.ma.masked_invalid(model.flatten())).data[0][1])
        # RMSE 
        crmsd = np.append(crmsd, np.sqrt(np.nanmean( (obs - model)**2) ) )
        # mean 
        mean = np.append(mean, np.nanmean(model))
        # std
        sdev = np.append(sdev, np.nanstd(model))
        # bias 
        bias = np.append(mean - mean, np.nanmean(model) - mean)
        
    return crmsd, ccoef, mean, sdev, bias 



def haversine(lon1, lat1, lon2, lat2):
    """
    calculate distance between two points in lon-lat grid. 
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


def common_grid(era, wrf_in, timedim = None, method = None, wrf_lons = None, wrf_lats = None):
    """
    Regrid WRF output to ERA5 grid, in order to calculate spatial correlations.

    Args:
        gpm(xarray.Dataset): 2D ERA5 input field 
        wrf(xarray.Daratset): 2D WRF input field 
        xlat(np.array):flattened array with WRF latitudes
        xlon(np.array): flattened array with WRF longitudes
    """
    # time mean
    if timedim == 'time':
        era_mean = era.mean('time')
        wrf_mean = wrf_in.mean('time')
    if timedim == 'Time':
        era_mean = era.mean('time')
        wrf_mean = wrf_in.mean('Time')
    else:
        era_mean= era
        wrf_mean = wrf_in

    # subset domain ERA5
    era_lon = era.coords["longitude"]
    era_lat = era.coords["latitude"]
    era_mean = era_mean.loc[
            dict(
                longitude=era_lon[(era_lon > 70) & (era_lon < 115)],
                latitude=era_lat[(era_lat > 25) & (era_lat < 40)],
            )
        ]

    newlon, newlat = np.meshgrid(era_mean.longitude.values, era_mean.latitude.values)
    if wrf_lons is None:
        wrf_lons = wrf_in.XLONG.values.flatten()
        wrf_lats = wrf_in.XLAT.values.flatten()
        if 'time' in wrf_in.XLAT.dims:
                wrf_lons = wrf_in.XLONG.values[:,:,0].flatten()
                wrf_lats = wrf_in.XLAT.values[:,:,0].flatten()
                
    if method is None: 
        method = 'linear'

    wrf_interp = scipy.interpolate.griddata(
            (wrf_lons, wrf_lats),
            wrf_mean.values.flatten(),
            (newlon, newlat),
            method=method)
    
    return era_mean.values, wrf_interp


def common_grid_gpm(gpm, wrf):
    """
    Regrid WRF output to GPM grid, in order to calculate spatial correlations.

    Args:
    gpm(xarray.Dataset): GPM data
    wrf(xarray.Daratset): WRF surface precipitation

    """
    # subset domain
    gpm_lon = gpm.coords["lon"]
    gpm_lat = gpm.coords["lat"]
    # coordinates
    gpm = gpm.loc[
        dict(
            lon=gpm_lon[(gpm_lon > 70) & (gpm_lon < 115)],
            lat=gpm_lat[(gpm_lat > 25) & (gpm_lat < 40)],
        )
    ]
    gpmlon = gpm.lon.values
    gpmlat = gpm.lat.values
    newlon, newlat = np.meshgrid(gpmlon, gpmlat)

    # subset domain
    wrf_sel = wrf.where(
        (wrf.lat > 25) & (wrf.lat < 40) & (wrf.lon > 70) & (wrf.lon < 115)
    )
    if wrf_sel.lat.ndim == 1:
        lons, lats = np.meshgrid(wrf_sel.lon.values, wrf_sel.lat.values)
    else:
        lats = wrf_sel.lat.values
        lons = wrf_sel.lon.values
    # bring to same grid (obs)
    wrf_interp = scipy.interpolate.griddata(
        (lons.flatten(), lats.flatten()),
        wrf_sel.values.flatten(),
        (newlon, newlat),
        method="linear",
    )
    return gpm, wrf_interp



def create_bandpass_filter(dx, Ni, Nj, lambda_min, lambda_max):
    """
    This function creates a 2D transfer function that can be used as a bandpass filter to remove 
    certain wavelengths of an atmospheric field (e.g. vorticity). 
    
    Args:
        dx(float): grid spacing in km 
        Ni(int): number of grid cells in y- direction 
        Nj(int): number of grid cells in x-direction
        lambda_min(float): minimum acceptable wavelength in km 
        lambda_max(float): maximum acceptable wavelength in km 
        
    Returns:
        tf(np.array): 2D array of shape NixNj with normalized coefficients. This matrix can be multiplied 
                     with the spectral coefficients (after FFT or DCT).
    """
    from scipy import signal
    m, n  = np.meshgrid(np.arange(Nj), np.arange(Ni))
    alpha = np.sqrt(m**2/Nj**2  +  n**2/Ni**2)
    # compute wavelengths in km 
    lambda_rect= 2*dx/ alpha

    ############### 2D bandpass filter(butterworth) #######################
    b, a = signal.iirfilter(2, [1/lambda_max,1/lambda_min], btype='band', ftype='butter', fs= 1/dx, output ='ba')
    w, h = signal.freqz(b, a, 1/lambda_rect.flatten(),fs = 1/dx)
    tf = np.reshape(abs(h), lambda_rect.shape)
    
    return tf 











