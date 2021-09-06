#!/bin/bash

## This bash script collects basic preprocessing operations for aggregated files with hourly timesteps of the MCS case, July 2008.
## Contact: julia.kukulies@gu.se


# accumulated precipitation 
for file in pr*nc; do cdo timsum ${file} /data/Home/Julia/cptp/${file}_accumulated.nc; done

# accumulated precipitation 20-21 July
for file in pr*nc; do cdo timsum -selday,20,21 ${file} /data/Home/Julia/cptp/${file}_accumulated_20-21.nc; done


# accumulated precipitation 18-23 July
for file in pr*nc; do cdo timsum -selday,18,19,20,21,22,23 ${file} /data/Home/Julia/cptp/${file}_accumulated_18-23.nc; done
