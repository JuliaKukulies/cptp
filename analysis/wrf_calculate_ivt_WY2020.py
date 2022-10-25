##### WRF postprocessing  ######

# This python script calculates the vertically integrated moisture transport [kg/m/s] from original WRF output files.
# Contact: julia.kukulies@gu.se



# Imports
import numpy as np
import xarray as xr
from pathlib import Path 

###################################### user input section #############################################################

# directory containing the WRF files  
wrf_data = Path('/glade/campaign/mmm/c3we/prein/CPTP/data/WY2020_large_domain/')
# directory to save postprocessed files in  
out = Path('/glade/scratch/kukulies/wrf_processed/')
# target months (output is saved per monthly files)
months = np.arange(7,9)
# hours in each output file (e.g. 24 for daily)
hours = 24 

##################################### define functions #################################################################

def colint_pressure(values,pressure_levels):
    """ This function calculates the column-integrated water vapor                                                                                             
    in kg/m2 from specific humidity (kg/kg) at different hPa levels. 

    Parameters: 
       values(array-like): 3D field of specific humidity)
       pressure_levels(array-like) : 1D field with corresponding pressure levels in hPa                       

    Returns: 
      2D field with column-integrated values 
    """

    g = 9.81 # gravitational acceleration (needed because levels are given in pressure rather than height)                                                                                                                   
    return np.trapz(values, pressure_levels, axis = 0)* g


def Vstagger_to_mass(V):
    """
    Function to convert staggered velocities to mass points (v-component)
    """
    # create the first column manually to initialize the array with correct dimensions                                                                 
    V_masspoint = (V[0,:]+V[1,:])/2. # average of first and second column                                                                                      
    V_num_rows = int(V.shape[0])-1 # we want one less row than we have                                                                                         
    for row in range(1,V_num_rows):
        row_avg = (V[row,:]+V[row+1,:])/2.
        # Stack those onto the previous for the final array                                                                                                    
        V_masspoint = np.row_stack((V_masspoint,row_avg))

    return V_masspoint


def Ustagger_to_mass(U):
    """
    Function to convert staggered velocitiesto mass points (u-component)
    """
    U_masspoint = (U[:,0]+U[:,1])/2. # average of first and second row                                                                                        
    U_num_cols = int(U.shape[1])-1 # we want one less column than we have                                                                                  
    for col in range(1,U_num_cols):
        col_avg = (U[:,col]+U[:,col+1])/2.
        U_masspoint = np.column_stack((U_masspoint,col_avg))
    return U_masspoint


def calculate_ivt(data):
    """
    Calculates vertically integrated water vapor fluxes and total transport
    from WRF output data.

    Parameters:
       data(xr.Dataset): original WRF output file as xarray 

    Returns: 
       qu_int: vertically integrated northward water vapor flux 
       qv_int: vertically integrated eastward water vapor flux 
       IVT: vertically integrated total moisture transport (sqrt(qu^2 + qv^2))

    """
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


###########################################   calculate monthly IVT #######################################################

for mon in months: 
    # get list with all files for month
    mcs4km_l = list(wrf_data.glob(  ("wrfout/wrfout_*2020-"+ str(mon).zfill(2) + "*00")) )
    mcs4km_l.sort()

    # 4km large - daily files
    tsteps =  len(mcs4km_l) *hours

    ivt4km_l = np.zeros(
        (
            tsteps,
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[1],
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[2],
        )
    )
    qu4km_l = np.zeros(
        (
            tsteps,
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[1],
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[2],
        )
    )
    qv4km_l = np.zeros(
        (
            tsteps,
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[1],
            xr.open_dataset(mcs4km_l[0]).XLAT.shape[2],
        )
    )
    times = np.array(())
    idx = 0
    # calculate IVT for each time step 
    for fname in mcs4km_l:
        print('processing...', fname)
        for t in np.arange(xr.open_dataset(fname).Time.values.shape[0]):
            wrfout = xr.open_dataset(fname).sel(Time=t)
            qu, qv, wrf_ivt = calculate_ivt(wrfout)
            ivt4km_l[idx] = wrf_ivt
            qu4km_l[idx] = qu
            qv4km_l[idx] = qv
            times = np.append(times, wrfout.Times.values)
            idx += 1

    # create netcdf for entire month 
    data_vars = dict(
        IVT=(["time", "south_north", "west_east"], ivt4km_l),
        qu=(["time", "south_north", "west_east"], qu4km_l),
        qv=(["time", "south_north", "west_east"], qv4km_l),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf( out / ( "WY2020_"+str(mon).zfill(2)+"_IVT.nc") ) 
    print("IVT calculated for month ", str(mon).zfill(2) ) 








