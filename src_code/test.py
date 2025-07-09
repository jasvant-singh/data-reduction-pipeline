import os
from astropy.io import fits
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up: 
project_root = os.path.dirname(script_dir)
output_path_dir = f'{project_root}/intermediate/science/scattered_light_substraction'

data1 = fits.getdata(f'{output_path_dir}/science_bias_scattered_light_subtracted_1.fits')
data2 = fits.getdata(f'{output_path_dir}/science_bias_scattered_light_subtracted_4.fits')

same = np.array_equal(data1, data2)
print("Image data is the same." if same else "Image data is different.")

