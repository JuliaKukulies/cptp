# ======================================
# This scripts post-processes WRF output of CPTP case study experiments. It operates as follows:
#
# 1) It extracts variable from wrfout file into single NetCDF files
# 2) The new NetCDFs that cover the case study period are combined to a single NetCDF file
# 3) The attributes of this file are changed to correspond approximately to ESGF standards
#
# Note, this program does not produce ESGF conform NetCDF output
#
# Dependencies:
# NCO
#
# Andreas Prein (prein@ucar.edu)
#
# ======================================
# ======================================

# modified by Julia Kukulies 04 March, 2022 (julia.kukulies@gu.se) - produce monthly aggregated output files for multi-year simulation 

# USER Input
CASE='WY2020' # ['4km_MCS', '4km_Monsoon', '4km_Snow']
VARS='pr' # ['pr','tas','swe','olr']
DataDir='/glade/scratch/prein/CPTP/WY2020_large_domain/wrfout/'
Institude='NCAR'
RCM='WRF'
Version='421'
Forcing='ECMWF-ERA5'
Experiment='evaluation'
Run='r1i1p1' # rN, N is the number of ensemble members; iN, N is the number of different initialization states; pN, N is the number of used physical parameterizations
dX='4' # approximate grid spacing in degrees (only needed for name )

# ======================================
# crete variables to start processing
WRFOUT=$DataDir
# creat output directory
OutDIR='/glade/scratch/kukulies/WY2020/pr/'
if [ ! -d $OutDIR ]; then
  mkdir -p $OutDIR
  echo $OutDIR
fi

if [ $CASE == '12km_casestudy-simulations' ]; then
    CASEname='MCS'
    DayStart='20080714'
    DayStop='20080724'
elif [ $CASE == '4km_Monsoon' ]; then
    CASEname='Monsoon'
    DayStart='20140727'
    DayStop='20140831'
elif [ $CASE == '4km_Snow' ]; then
    CASEname='Snow'
    DayStart='20081001'
    DayStop='20181010'
# for monthly output of multi-year simulation: indicate start and end year! 
elif [ $CASE == 'WY2020' ]; then
    CASEname='WY2020'
    StartYear=2019
    StopYear=2020
fi

# ======================================
# set variable attributes
if [ $VARS == "pr" ]; then
  wrfname='PREC_ACC_NC'
  varname='pr'
  units="kg m-2 s-1"
  standard_name="precipitation_flux"
  long_name="Precipitation"
  cell_methods="time: mean"
fi
if [ $VARS == "tas" ]; then
  wrfname='T2'
  varname='tas'
  units="K"
  standard_name="air_temperature"
  long_name="Near-Surface Air Temperature"
  cell_methods="time: mean"
fi
if [ $VARS == "swe" ]; then
  wrfname='SNOW'
  varname='swe'
  units="kg m-2"
  standard_name="snow_water_equivalent"
  long_name="Snow Water Equicalent"
  cell_methods="time: sum"
fi
if [ $VARS == "olr" ]; then
  wrfname='LWUPT'
  varname='olr'
  units="W m-2"
  standard_name="upwelling_longwave_flux_TOA"
  long_name="Instantaneous Upwellng Longwave Flux at Top of Atmosphere"
  cell_methods="time: point"
fi 


# loop through each year and month (subset months if needed) 
for YEAR in $(seq $StartYear $StopYear)
do 
    for MON in {08..08}
    do 
	cd $WRFOUT
	if compgen -G "wrfout_*$YEAR-$MON*00"; then # check if any files for this year and month exist
	    for files in "wrfout_*$YEAR-$MON*00"
	    do
		for fi in $files
		do
		echo $fi
		ncks -O -v $wrfname,XLAT,XLONG,XTIME $WRFOUT$fi $OutDIR$fi
		done
	    done
	     
	    # combine NetCDF files 
	    cd $OutDIR
	    File=$VARS'_CPTP-'$CASEname'-'$dX'_'$Forcing'_'$Experiment'_'$Run'_'$Institude'-'$RCM$Version'P_v1_hour_'$YEAR'-'$MON'.nc'    
	    FinFileName=$OutDIR$File
	    for ii in $OutDIR"wrfout_*"
	    do
		# echo $ii
		ncrcat $ii $FinFileName
	    done

	    # make XLAT and XLONG 2D
	    cd $OutDIR
	    ncwa -v XLAT,XLONG -a Time $File $File'.tmp'
	    ncks -C -x -v XLAT,XLONG $File $File'.tmp2'
	    ncks -A -v XLAT,XLONG $File'.tmp' $File'.tmp2'
	    rm $File.tmp $File
	    mv $File'.tmp2' $File

	    # fix variable names and attributes
	    ncrename -v $wrfname,$varname -v XLAT,lat -v XLONG,lon -v XTIME,time $FinFileName
	    ncatted -O -a long_name,$varname,o,c,"$long_name""" $FinFileName
	    ncatted -O -a units,$varname,o,c,"$units""" $FinFileName
	    ncatted -O -a standard_name,$varname,o,c,"$standard_name""" $FinFileName
	    ncatted -O -a cell_methods,$varname,o,c,"$cell_methods""" $FinFileName
	    ncatted -O -a coordinates,$varname,o,c,"lon lat" $FinFileName
	    ncatted -O -a coordinates,lat,o,c,"lon lat" $FinFileName
	    ncatted -O -a coordinates,lon,o,c,"lon lat" $FinFileName
	    ncatted -O -a description,,d,, $FinFileName

	    # fix the variable units if nescessary
	    if [ $VARS == "pr" ]; then
		ncap2 -s "$varname=$varname/3600" $FinFileName $FinFileName'TMP'
		mv $FinFileName'TMP' $FinFileName
	    fi

	    # fix the time variable
	    ncatted -O -a calendar,time,o,c,"standard" $FinFileName
	    ncatted -O -a standard_name,time,o,c,"time" $FinFileName

	    # clean up                                                                                                                                        
	    cd $OutDIR
	    rm wrfout_*
	fi 
    done 
done 


