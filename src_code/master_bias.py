#!/usr/bin/env python3

from astropy.io import fits
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up: 
project_root = os.path.dirname(script_dir)
bias_dir = os.path.join(project_root, 'intermediate/bias')
output_dir = os.path.join(project_root, 'output')

bias_cube = fits.getdata(f'{bias_dir}/Bias_50KHz_5frs.fits')


master_bias = np.median(bias_cube, axis=0)

masterbias_file = f'{output_dir}/masterbias_file.fits'
hdu = fits.PrimaryHDU(master_bias)
hdu.writeto(masterbias_file, overwrite=True)

print(f"Master bias saved to: {masterbias_file}")
