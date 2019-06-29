"""
Input:
argv[1] -- the output file name to write selected pixel coordinates to
argv[2] -- directory to your fits files, e.g. /Users/shryguo/Documents/exoplanet/atmosphere/HD97658b/spitzer/ch1/r49696512/ch1/bcd/

setup:
find the max flux pixel, and circle in all pixels around it which contains more than 1% of the total flux (sum from all selected pixels).
"""
import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
import glob

pix_out_file = str(sys.argv[1])  # write selected pixel coordinates in this file
dir = str(sys.argv[2])     # directory to your fits files, e.g. /Users/shryguo/Documents/exoplanet/atmosphere/HD97658b/spitzer/ch1/r49696512/ch1/bcd/

fits_file_list = glob.glob(dir+"*_bcd.fits")

fname = fits_file_list[0]
image_fits = fits.open(fname)
flux_pack = image_fits[0].data
len_x, len_y = np.shape(flux_pack[0])   #"flux_pack[0]" if multiple frames, "flux_pack" if single frame.
matrix_index = np.zeros((len_x, len_y))


f_out = open(pix_out_file, 'w')
frame_number_sum = 0
for fname in fits_file_list:
#    label = "%04d" % (j,)
    image_fits = fits.open(fname)
    flux_pack = image_fits[0].data
    frame_number_tot = len(flux_pack)  # if only single frame, do frame_number_tot = 1
    frame_number_sum += frame_number_tot
    print(fname)
    
    for frame_number in range(frame_number_tot):
        flux = flux_pack[frame_number]
        x_center = 0
        y_center = 0

        flux_max = 0.0
        for i in range(1, len_x-1):
            index_max_y = np.argmax(flux[i, :])
            if index_max_y == 0 or index_max_y == len_y-1:
                continue
            flux_tmp = flux[i, index_max_y]
            if flux_tmp > flux_max:
                x_center = i
                y_center = index_max_y
                flux_max = flux_tmp

        flux_max = flux[x_center, y_center]

        matrix_index[x_center, y_center] += 1.0
        print(x_center, y_center)

        flux_sum = flux_max
        side_length = 3
        start_index_x = x_center-int((side_length-1)/2)
        start_index_y = y_center-int((side_length-1)/2)
        for i in range(side_length-1):
            flux_sum += flux[start_index_x, start_index_y+i]
            matrix_index[start_index_x, start_index_y+i] += 1.0
            #f_out.write("%s %s\n" % (start_index_x, start_index_y+i))
        for i in range(side_length-1):
            flux_sum += flux[start_index_x+i, start_index_y+side_length-1]
            matrix_index[start_index_x+i, start_index_y+side_length-1] += 1.0
        for i in range(side_length-1):
            flux_sum += flux[start_index_x+side_length-1, start_index_y+side_length-1-i]
            matrix_index[start_index_x+side_length-1, start_index_y+side_length-1-i] += 1.0
        for i in range(side_length-1):
            flux_sum += flux[start_index_x+side_length-1-i, start_index_y]
            matrix_index[start_index_x+side_length-1-i, start_index_y] += 1.0

        stop = False
        while stop == False:
            stop = True
            flux_limit = 0.01*flux_sum
            #print(flux_limit)
            side_length = side_length +2
            start_index_x = x_center-int((side_length-1)/2)
            start_index_y = y_center-int((side_length-1)/2)
            for i in range(side_length-1):
                flux_tmp = flux[start_index_x, start_index_y+i]
                if flux_tmp > flux_limit:
                    flux_sum += flux_tmp
                    matrix_index[start_index_x, start_index_y+i] += 1.0
                    stop = False
            for i in range(side_length-1):
                flux_tmp = flux[start_index_x+i, start_index_y+side_length-1]
                if flux_tmp > flux_limit:
                    flux_sum += flux_tmp
                    matrix_index[start_index_x+i, start_index_y+side_length-1] += 1.0
                    stop = False
            for i in range(side_length-1):
                flux_tmp = flux[start_index_x+side_length-1, start_index_y+side_length-1-i]
                if flux_tmp > flux_limit:
                    flux_sum += flux_tmp
                    matrix_index[start_index_x+side_length-1, start_index_y+side_length-1-i] += 1.0
                    stop = False
            for i in range(side_length-1):
                flux_tmp = flux[start_index_x+side_length-1-i, start_index_y]
                if flux_tmp > flux_limit:
                    flux_sum += flux_tmp
                    matrix_index[start_index_x+side_length-1-i, start_index_y] += 1.0
                    stop = False
        print(flux_limit)

for i in range(len_x):
    for j in range(len_y):
        if matrix_index[i, j] > 0.01*(frame_number_sum-1):
            print(i, j, matrix_index[i, j])
            f_out.write("%s %s\n" % (i, j))
f_out.close()

fig = plt.figure()
plt.imshow(matrix_index, interpolation='nearest', cmap=cm.gray, vmin = -0.5, vmax = frame_number_sum-1)
plt.savefig("show_index.pdf")
plt.show()
plt.close(fig)

