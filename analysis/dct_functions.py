"""
Functions for data analysis in spectral space using a discrete cosine transformation on 2D atmospheric fields.

julia.kukulies@gu.se

"""

import numpy as np
import netCDF4 as nc
#import wrf


def lambda_to_k(wavelengths):
    """
    This function convert wavelengths lambda in km to wavenumbers k 
    in rad/m. 
    """
    return 2* np.pi/ (wavelengths * 1000)



def k_to_lambda(Ni, Nj, dx):
    """
    This function converts the non-dimensional wavenumbers
    of a 2D k-space to wavelengths that represent the real spatial scale.

    Args:
        Ni(int): number of grid cells in y-direction
        Nj(int): number of grid cells in x-direction
        dx(float): grid spacing in m or km

    Returns:
        lambda_mn(np.array): 2D array with wavelengths for
                            each m,n -pair. The output unit is
                            same unit given for dx.
    """
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing = 'ij')
    k = np.sqrt(m ** 2 + n ** 2)

    ########   squared domain case   ##########
    if Ni == Nj:
        lambda_mn = (2 * Ni * (dx)) / k

    else:
        ######## rectangular domain case ##########

        ## normalization of k
        alpha = np.sqrt(m ** 2 / Ni ** 2 + n ** 2 / Nj ** 2)
        # compute wavelength
        lambda_mn = 2 * dx / alpha

    return lambda_mn


def get_variances(spectral):
    """
    This function computes sigma squared
    (variances of the spectral coefficients).

    Args:
        spectral(np.array): 2D array with spectral coefficients
    Returns:
        variance(np.array): 2D array with variances

    """
    Ni = spectral.shape[0]
    Nj = spectral.shape[1]
    return (spectral ** 2) / (Ni * Nj)



def get_power_spectrum(variance, dx):
    """
    This function creates a power spectrum for a given field
    of spectral variances.

    Args:
        variance(np.array): 2D field of sigma squared (variances of spectral coefficients)
        dx(float): grid spacing in m or km
    Returns:
        wavelengths(np.array): 1D array with wavelengths (same unit as dx)
        histogram(np.array): binned variances corresponding to wavelengths

    """
    Ni = variance.shape[0]
    Nj = variance.shape[1]
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing = 'ij')

    # to ensure that the number of wavenumber bands is not larger than any axis of the domain
    mindim = min(Ni, Nj)
    # array with wavenumber bands
    k = np.arange(1, mindim)
    # normalized k for each element in k space, alpha max should be square root of 2! 
    alpha = np.sqrt(m ** 2 / Ni ** 2 + n ** 2 / Nj ** 2)
    # limits of contributing bands
    lowerbound = k / mindim
    upperbound = (k+1)/ mindim
    
    # binning 
    histogram, bins = np.histogram(alpha.flatten(), bins=upperbound, weights=variance.flatten())
    alpha_mean = np.nanmean([lowerbound, upperbound], axis = 0 )
    wavelengths = 2 * dx / alpha_mean

    return wavelengths, histogram


def calculate_vorticity(fname, lev, timeidx):
    """
    Computes the absolute and relative vorticity for a specific level and timestep
    based on WRF output data.

    Args:
        fname(str or path): filename of WRF input dataset
        lev(int): vertical pressure level in hpa

    Returns:
        avo_lev(xarray.DataArray): 2d field of absolute vorticity
        rv_lev(xarray.DataArray): 2d field of relative vorticity

    """
  
    wrf_nc = nc.Dataset(fname)
    # compute absolute vorticity with wrf python module
    avo = wrf.g_vorticity.get_avo(wrf_nc, timeidx)
    # coriolis parameter
    OMEGA = 7.292e-5
    lats = np.array(wrf_nc["XLAT"]).squeeze()
    cor = 2 * OMEGA * np.sin(lats)
    # get relative vorticity
    rv = avo - cor
    # WRF interpolation to specific pressure level
    base_pressure = np.array(wrf_nc["PB"]).squeeze()
    perturbation_pressure = np.array(wrf_nc["P"]).squeeze()
    rv_lev = wrf.interplevel(avo, (base_pressure + perturbation_pressure) * 0.01, lev)
    avo_lev = wrf.interplevel(rv, (base_pressure + perturbation_pressure) * 0.01, lev)

    return avo_lev, rv_lev


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




























