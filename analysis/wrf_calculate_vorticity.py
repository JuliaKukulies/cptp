##### WRF postprocessing to get vorticity fields all in one array.  ######

# This python script calculates the absolute and relative vorticity [10-5 s-1] for WRF output files. 
# It also calculates a filtered version of the relative vorticity field by applring a bandpass filter
# that removes wavelengths < 400km and > 1000km, in order to focus on meso-scale disturbances. 


# julia.kukulies@gu.se

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import wrf
from scipy import fft

# import python module with analysis functions
sys.path.insert(1, '../analysis/')
from cptp import analysis
from cptp.analysis import create_bandpass_filter

import warnings
warnings.filterwarnings("ignore")

# campaign storage
cptp = Path("/glade/campaign/mmm/c3we/prein/CPTP/data/")
# WRF files for all vars at model levels
mcs4km = list(cptp.glob("4km_MCS/wrfout/wrfout_d01_2008-07*"))
mcs4km.sort()
#assert len(mcs4km) ==  744
#print(len(mcs4km), ' files for 4km')


mcs4km_l = list(cptp.glob("4km_MCS_L/wrfout/wrfout_d01_2008-07*"))
mcs4km_l.sort()
mcs4km_l.pop(1)
assert len(mcs4km_l) < 20
assert len(mcs4km_l) > 10
print(len(mcs4km_l), ' files for 4km large domain')


mcs12km = list(cptp.glob("2008_MCS/12km_data/wrfout/wrfout_d01_2008-07*"))
mcs12km.sort()
mcs12km = mcs12km[13:24]
assert len(mcs12km) < 20
assert len(mcs12km) > 10
print(len(mcs12km), ' files for 12km')



### additional experiments ### 
exps = Path('/glade/scratch/kukulies/CPTP/')
mcs4km = list(exps.glob("4km_5deg/wrfout/wrfout_d01_2008-07*00"))
mcs4km.sort()
#mcs4km.pop(1)
assert len(mcs4km) > 0 
print(len(mcs4km), ' files for experiment')


#### calculate absolute and relative vorticity #### 

if Path("/glade/scratch/kukulies/wrf_processed/WRF4km_d03_VORT.nc").is_file() is False:
    # 4km large - daily files
    timesteps = len(mcs4km) * 24
    timesteps = 265 
    Ni = xr.open_dataset(mcs4km[0]).XLAT.shape[1]
    Nj = xr.open_dataset(mcs4km[0]).XLAT.shape[2]
    avo4km = np.zeros((timesteps,Ni, Nj))
    rv4km = np.zeros((timesteps,Ni, Nj))
    rv4km_filtered = np.zeros((timesteps,Ni, Nj))
    times = np.array(())
    wrfout = xr.open_dataset(mcs4km[0])

    
    # bandpass filter                                                                                                    
    tf = create_bandpass_filter(4, Ni, Nj, 400 , 1000)
    idx= 0
    for fname in mcs4km:
        for t in np.arange(xr.open_dataset(fname).Time.values.shape[0]):
            print(fname, idx)
            avo500hpa, rv500hpa  = analysis.calculate_vorticity(fname,500,t, squeeze= True)
            # get rid of nan values (not accepted for DCT)                                                               
            rv500_wrf = rv500hpa.interpolate_na(dim="west_east", method="linear", fill_value="extrapolate")
            rv500_wrf = np.array(rv500_wrf)
            avo4km[idx] = avo500hpa.data
            rv4km[idx] = rv500_wrf
            # get filtered field                                                                                          
            spectral = fft.dctn(rv500_wrf)
            filtered = fft.idctn(tf* spectral)
            rv4km_filtered[idx] = filtered
            times = np.append(times, wrfout.Times.values[0])
            idx +=1

    # create netcdf
    data_vars = dict(
        absolute_vorticity=(["time", "south_north", "west_east"], avo4km),
    relative_vorticity=(["time", "south_north", "west_east"], rv4km), 
        relative_vorticity_filtered=(["time", "south_north", "west_east"], rv4km_filtered),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values[0]),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values[0]),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf("/glade/scratch/kukulies/wrf_processed/WRF4km_d03_VORT.nc")
    print("VORT calculated for daily files", mcs4km[0])



### additional experiments ### 
exps = Path('/glade/scratch/kukulies/CPTP/')
mcs4km = list(exps.glob("4km_spectral/wrfout/wrfout_d01_2008-07*00"))
mcs4km.sort()
#mcs4km.pop(1)
assert len(mcs4km) > 0 
print(len(mcs4km), ' files for experiment')


#### calculate absolute and relative vorticity #### 

if Path("/glade/scratch/kukulies/wrf_processed/WRF4km_nudging_VORT.nc").is_file() is False:
    # 4km large - daily files
    timesteps = len(mcs4km) * 24
    timesteps = 265 
    Ni = xr.open_dataset(mcs4km[0]).XLAT.shape[1]
    Nj = xr.open_dataset(mcs4km[0]).XLAT.shape[2]
    avo4km = np.zeros((timesteps,Ni, Nj))
    rv4km = np.zeros((timesteps,Ni, Nj))
    rv4km_filtered = np.zeros((timesteps,Ni, Nj))
    times = np.array(())
    wrfout = xr.open_dataset(mcs4km[0])

    
    # bandpass filter                                                                                                    
    tf = create_bandpass_filter(4, Ni, Nj, 400 , 1000)
    idx= 0
    for fname in mcs4km:
        for t in np.arange(xr.open_dataset(fname).Time.values.shape[0]):
            print(fname, idx)
            avo500hpa, rv500hpa  = analysis.calculate_vorticity(fname,500,t, squeeze= True)
            # get rid of nan values (not accepted for DCT)                                                               
            rv500_wrf = rv500hpa.interpolate_na(dim="west_east", method="linear", fill_value="extrapolate")
            rv500_wrf = np.array(rv500_wrf)
            avo4km[idx] = avo500hpa.data
            rv4km[idx] = rv500_wrf
            # get filtered field                                                                                          
            spectral = fft.dctn(rv500_wrf)
            filtered = fft.idctn(tf* spectral)
            rv4km_filtered[idx] = filtered
            times = np.append(times, wrfout.Times.values[0])
            idx +=1

    # create netcdf
    data_vars = dict(
        absolute_vorticity=(["time", "south_north", "west_east"], avo4km),
    relative_vorticity=(["time", "south_north", "west_east"], rv4km), 
        relative_vorticity_filtered=(["time", "south_north", "west_east"], rv4km_filtered),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values[0]),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values[0]),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf("/glade/scratch/kukulies/wrf_processed/WRF4km_nudging_VORT.nc")
    print("VORT calculated for daily files", mcs4km[0])



if Path("/glade/scratch/kukulies/wrf_processed/WRF8km_ref_VORT.nc").is_file() is False:
    # 4km large - daily files
    timesteps = len(mcs4km) * 24 
    Ni = xr.open_dataset(mcs4km[0]).XLAT.shape[1]
    Nj = xr.open_dataset(mcs4km[0]).XLAT.shape[2]
    avo4km = np.zeros((timesteps,Ni, Nj))
    rv4km = np.zeros((timesteps,Ni, Nj))
    rv4km_filtered = np.zeros((timesteps,Ni, Nj))
    times = np.array(())
    wrfout = xr.open_dataset(mcs4km[0])

    
    # bandpass filter                                                                                                    
    tf = create_bandpass_filter(8, Ni, Nj, 400 , 1000)
    idx= 0
    for fname in mcs4km:
        for t in np.arange(xr.open_dataset(fname).Time.values.shape[0]):
            print(fname, idx)
            avo500hpa, rv500hpa  = analysis.calculate_vorticity(fname,500,t, squeeze= True)
            # get rid of nan values (not accepted for DCT)                                                               
            rv500_wrf = rv500hpa.interpolate_na(dim="west_east", method="linear", fill_value="extrapolate")
            rv500_wrf = np.array(rv500_wrf)
            avo4km[idx] = avo500hpa.data
            rv4km[idx] = rv500_wrf
            # get filtered field                                                                                          
            spectral = fft.dctn(rv500_wrf)
            filtered = fft.idctn(tf* spectral)
            rv4km_filtered[idx] = filtered
            times = np.append(times, wrfout.Times.values[0])
            idx +=1

    # create netcdf
    data_vars = dict(
        absolute_vorticity=(["time", "south_north", "west_east"], avo4km),
    relative_vorticity=(["time", "south_north", "west_east"], rv4km), 
        relative_vorticity_filtered=(["time", "south_north", "west_east"], rv4km_filtered),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values[0]),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values[0]),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf("/glade/scratch/kukulies/wrf_processed/WRF8km_ref_VORT.nc")
    print("VORT calculated for daily files", mcs4km[0])






if Path("/glade/scratch/kukulies/wrf_processed/WRF4km_L_VORT.nc").is_file() is False:
    # 4km large - daily files
    timesteps = len(mcs4km_l) * 24 
    Ni = xr.open_dataset(mcs4km_l[0]).XLAT.shape[1]
    Nj = xr.open_dataset(mcs4km_l[0]).XLAT.shape[2]
    avo4km_l = np.zeros((timesteps,Ni, Nj))
    rv4km_l = np.zeros((timesteps,Ni, Nj))
    rv4km_l_filtered = np.zeros((timesteps,Ni, Nj))
    times = np.array(())
    wrfout = xr.open_dataset(mcs4km_l[0])

    # bandpass filter                                                                                                    
    tf = create_bandpass_filter(4, Ni, Nj, 400 , 1000)
    idx= 0
    for fname in mcs4km_l:
        for t in np.arange(xr.open_dataset(fname).Time.values.shape[0]):
            print(fname, idx)
            avo500hpa, rv500hpa  = analysis.calculate_vorticity(fname,500,t, squeeze= True)
            # get rid of nan values (not accepted for DCT)                                                               
            rv500_wrf = rv500hpa.interpolate_na(dim="west_east", method="linear", fill_value="extrapolate")
            rv500_wrf = np.array(rv500_wrf)
            avo4km_l[idx] = avo500hpa.data
            rv4km_l[idx] = rv500_wrf
            # get filtered field                                                                                          
            spectral = fft.dctn(rv500_wrf)
            filtered = fft.idctn(tf* spectral)
            rv4km_l_filtered[idx] = filtered
            times = np.append(times, wrfout.Times.values[0])
            idx +=1


    # create netcdf
    data_vars = dict(
        absolute_vorticity=(["time", "south_north", "west_east"], avo4km_l),
    relative_vorticity=(["time", "south_north", "west_east"], rv4km_l), 
        relative_vorticity_filtered=(["time", "south_north", "west_east"], rv4km_l_filtered),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values[0]),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values[0]),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )
    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf("/glade/scratch/kukulies/wrf_processed/WRF4km_L_VORT.nc")
    print("VORT calculated for daily files", mcs4km_l[0])



# 12km - daily files
if Path("/glade/scratch/kukulies/wrf_processed/WRF12km_VORT.nc").is_file() is False:

    timesteps = len(mcs12km) * 24
    Ni = xr.open_dataset(mcs12km[0]).XLAT.shape[1]
    Nj = xr.open_dataset(mcs12km[0]).XLAT.shape[2]
    avo12km = np.zeros((timesteps,Ni, Nj ))
    rv12km = np.zeros((timesteps, Ni, Nj ))
    rv12km_filtered = np.zeros((timesteps, Ni, Nj ))
    times = np.array(()) 

    wrfout = xr.open_dataset(mcs12km[0])


    # bandpass filter 
    tf = create_bandpass_filter(12, Ni, Nj, 400 , 1000)

    idx = 0 
    for fname in mcs12km:
        for t in np.arange(24):
            avo500hpa, rv500hpa  = analysis.calculate_vorticity(fname,500, t , squeeze= True)
           # get rid of nan values (not accepted for DCT)
            rv500_wrf = rv500hpa.interpolate_na(dim="west_east", method="linear", fill_value="extrapolate")
            rv500_wrf = np.array(rv500_wrf)
            avo12km[idx] = avo500hpa.data
            rv12km[idx] = rv500_wrf
            # get filtered field 
            spectral = fft.dctn(rv500_wrf)
            filtered = fft.idctn(tf* spectral)     
            rv12km_filtered[idx] = filtered
            times = np.append(times, wrfout.Times.values[0])
            idx+=1 

    # create netcdf
    data_vars = dict(
        absolute_vorticity=(["time", "south_north", "west_east"], avo12km),
        relative_vorticity=(["time", "south_north", "west_east"], rv12km),
        relative_vorticity_filtered=(["time", "south_north", "west_east"], rv12km_filtered),
        XLAT=(["south_north", "west_east"], wrfout.XLAT.squeeze().values[0]),
        XLONG=(["south_north", "west_east"], wrfout.XLONG.squeeze().values[0]),
    )
    coords = dict(
        time=times,
        south_north=wrfout.south_north.values,
        west_east=wrfout.west_east.values,
    )

    data = xr.Dataset(data_vars=data_vars, coords=coords)
    data.to_netcdf("/glade/scratch/kukulies/wrf_processed/WRF12km_VORT.nc")
    print("VORT calculated for daily files", mcs12km[0])














