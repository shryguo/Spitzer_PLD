# Spitzer_PLD

The three scripts should be used in the following order:
1. find_pixels.py : pixel selection, output a file with pixel coordinates.
2. extract_image_frameAve.py: extract time and flux from the fits files according to the pixel selection.
3. fit_spitzer_batman_gp.py: preprocessing and PLD, the main class "MeanModel" contains the PLD algorithm.
