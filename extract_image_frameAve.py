"""
Input:
argv[1] -- input file with pixel coordinates (output from "find_pixels.py")
argv[2] -- the output file name to write time stamp and flux matrix (num_time x (num_pixel+1)) to
argv[3] -- directory to your fits files, e.g. /Users/shryguo/Documents/exoplanet/atmosphere/HD97658b/spitzer/ch1/r49696512/ch1/bcd/
"""
import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

pix_in_file = str(sys.argv[1])
flux_out_file = str(sys.argv[2])
dir = str(sys.argv[3])     # directory to your fits files, e.g. /Users/shryguo/Documents/exoplanet/atmosphere/HD97658b/spitzer/ch1/r49696512/ch1/bcd/

## inport the pixels to extract flux from #######################
index_list = []
for row in open(pix_in_file, "r"):
    ps = row.split(" ")
    index_list.append((int(ps[0]), int(ps[1])))
print(index_list)
############################################

f_out = open(flux_out_file, 'w')

fits_file_list = glob.glob(dir+"*_bcd.fits")

for fname in fits_file_list:
#    label = "%04d" % (j,)
    print(fname)
    image_fits = fits.open(fname)
    time = image_fits[0].header['BMJD_OBS']  #observation start time
    flux = image_fits[0].data
    frame_number = len(flux)  #if only one frame, set frame_number=1
    time += image_fits[0].header['FRAMTIME']*frame_number/2./24./60/60  #observation mid time
    f_out.write("%s " % (time))
    x_dim, y_dim = np.shape(flux[0])   #if only one frame, "flux"
    flux_tot = np.zeros((x_dim, y_dim))

    for i in range(frame_number):
        flux_tot += flux[i, :, :]
    
    flux_tot = flux_tot/frame_number
    for num in range(len(index_list)):
        pix_coord = index_list[num]
        f_out.write("%s " % (flux_tot[pix_coord]))
    f_out.write("\n")

f_out.close()
