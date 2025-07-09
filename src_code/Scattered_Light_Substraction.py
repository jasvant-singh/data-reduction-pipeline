from scipy.optimize import curve_fit
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import random
import statistics
#from scipy.interpolate import interp2d
#from shapely.geometry import LineString
#from sklearn.preprocessing import normalize
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from pylab import *
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.signal import argrelmax
from scipy.signal import find_peaks
from scipy.signal import medfilt2d
from scipy.stats import norm
#from itertools import product
#from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import scipy.signal as signal
import scipy.stats as stats
import bisect
from astropy.modeling import models
from astropy import units as unit
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
import warnings
import os
import re
warnings.filterwarnings('ignore')


def load_parameters(param_file):
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue  # skip comments
            if '=' in line:
                key, val = line.split('=')
                key = key.strip()
                val = val.strip()
                # Convert value to int or float
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
                params[key] = val
                
    return params['starting_order'], params['number_of_peaks'], params['plot_flag'], params['sigma_FWHM'], params['detector_pixels'], params['centre_column_median']



def extract_hwp_serial(file_path):
    """
    Extracts the HWPPosition from filename and returns a corresponding serial number.
    HWP 0 → 1, 22.5 → 2, 45 → 3, 67.5 → 4
    """
    match = re.search(r'HWPPosition-([0-9]?)', file_path)
    if not match:
        return 1  # default to 0° → serial 1

    hwp_str = match.group(1)
    hwp_map = {
        "0": 1,
        "2": 2,
        "4": 3,
        "6": 4
    }
    return hwp_map.get(hwp_str, 1)  # default to 1 if unrecognized




def scattered_light_substraction (science_frame_Fits, echelle_trace_y_interp, detector_pixels):
    
    science_frame_41 = science_frame_Fits

    science_frame_41 = np.rot90(science_frame_41)

    # ~ hdu = fits.PrimaryHDU(science_frame_41)
    # ~ hdu.writeto(f'{output_path_dir}science_frame_Rotated.fits', overwrite=True)
    scattered_light_substract = []
    
    scattered_light_substract.append(np.zeros((len(echelle_trace_y_interp[0]))))    #  For column 0

    for i in range (1, len(echelle_trace_y_interp[0])-1):    #  For columns 1 to 1022
        mid_point_pixel = []
        mid_point_flux = []
        for j in range (len(echelle_trace_y_interp)-1):
            peak_mid_point = int((echelle_trace_y_interp[j][i] + echelle_trace_y_interp[j+1][i])/2)
            median_box = []
            for k in range (-1, 2):
                for l in range (-1, 2):
                    median_box.append(science_frame_41[peak_mid_point+k][i+l])
            median_box = np.array(median_box)
            peak_mid_point_median = np.median(median_box)
            mid_point_flux.append(peak_mid_point_median)
            mid_point_pixel.append(peak_mid_point)
        
        scattered_light_column_interp = interp1d(mid_point_pixel, mid_point_flux, kind = 'linear', bounds_error = False, fill_value="extrapolate")
        
        y_new = np.linspace(0, detector_pixels-1, detector_pixels)
        scattered_light_column = scattered_light_column_interp(y_new)
        bad_pixels = np.where(scattered_light_column < 0)
        scattered_light_column[bad_pixels] = 0
        scattered_light_substract.append(scattered_light_column)
            
        #x_new = np.zeros((len(mid_point_pixel)))
        #x_new.fill(i)
        #plt.scatter(x_new, mid_point_pixel, s=1, color='blue')
    scattered_light_substract.append(np.zeros((len(echelle_trace_y_interp[0]))))    #  For column 1023

    scattered_light_substract_reversed = []
    for i in range (len(scattered_light_substract)):
        scattered_light_substract_reversed.append(np.flip(np.array(scattered_light_substract[i])))

    scattered_light_substract_reversed = np.array(scattered_light_substract_reversed)
    #scattered_light_substract_reversed = medfilt2d(scattered_light_substract_reversed)
    # ~ scattered_light_substract_reversed = np.rot90(scattered_light_substract_reversed)###########################################
    
    plt.imshow(scattered_light_substract_reversed, cmap="gray", origin='lower')
    plt.title("Estimated Scattered Light")
    plt.colorbar(label='Flux')
    plt.savefig(f"{output_path_dir}scattered_light_estimate{hwp_serial}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    return scattered_light_substract_reversed

def load_traced_orders_from_txt(directory):
    """
    Load traced order files from a directory.

    Returns:
    - orders: list of order numbers (int)
    - peaks: list of central peak positions (int)
    - xcor: list of X coordinate arrays (np.array)
    - ycor: list of Y coordinate arrays (np.array)
    """
    orders = []
    peaks = []
    xcor = []
    ycor = []

    for file in sorted(os.listdir(directory)):
        if not file.startswith("order_") or not file.endswith(".txt"):
            continue

        filepath = os.path.join(directory, file)
        match = re.match(r"order_(\d+)_([oe])\.txt", file)
        if not match:
            continue

        order = int(match.group(1))
        with open(filepath, "r") as f:
            lines = f.readlines()
            peak_line = lines[0].strip()
            peak = int(peak_line.split(":")[1])
            xs, ys = [], []
            for line in lines[2:]:
                if line.strip() == "":
                    continue
                x, y = map(int, line.strip().split())
                xs.append(x)
                ys.append(y)

        orders.append(order)
        peaks.append(peak)
        xcor.append(np.array(xs))
        ycor.append(np.array(ys))

    return orders, peaks, xcor, ycor



script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)


output_path_dir = f'{project_root}/intermediate/science/scattered_light_substraction/'
os.makedirs(output_path_dir, exist_ok=True)

param_file = f'{project_root}/parameters.txt' 

starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median = load_parameters(param_file)

# ~ print(starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median)

# ~ base_name = 'bias_sub_HD61421_RedCD-FilterIn_20s_HWPPosition-0_EncVal-0pt63_1' #############################filename


# ~ science_frame_path = f"{project_root}/intermediate/science/bias_subtracted/{base_name}.fits"

# ~ hwp_serial = extract_hwp_serial(science_frame_path) ########## serial number

# ~ science_data_cube_41 = fits.getdata(science_frame_path)
# ~ science_frame_41 = science_data_cube_41[0][:][:]


with open(f'{project_root}/intermediate/science/bias_subtracted/bias_subtracted_paths.txt', "r") as f:
    science_frame_paths = [line.strip() for line in f if line.strip()]
    science_frame_paths = [p for p in science_frame_paths
                       if "bias_sub_HD61421" in p and "Sky" not in p and "Dark" not in p and "Calib" not in p]

for file_path in science_frame_paths:
    print(f"Processing file: {file_path}")
    hwp_serial = extract_hwp_serial(file_path)

    # Load FITS data
    science_data_cube_41 = fits.getdata(file_path)
    
    science_frame_41 = science_data_cube_41[0][:][:]
    base_name = file_path.split('/')[-1].replace('.fits', '')

    # Directory containing your traced order files (only 'e' type)
    traced_dir = f'{project_root}/intermediate/science/traced_orders/{hwp_serial}'  # <-- Change this path

    orders, peaks, echelle_trace_x_interp, echelle_trace_y_interp = load_traced_orders_from_txt(traced_dir)

    # Step 1: Subtract scattered light
    scattered_light = scattered_light_substraction(
        science_frame_41,
        echelle_trace_y_interp,
        detector_pixels
    )

    # Step 2: Subtract scattered light from science frame
    science_bias_scattered_light_subtracted = science_frame_41 - scattered_light

    # Step 3: Optional — remove negative values
    science_bias_scattered_light_subtracted[science_bias_scattered_light_subtracted < 0] = 0

    # Step 4: Save to FITS
    output_fits_path = os.path.join(output_path_dir, f"science_bias_scattered_light_subtracted_{hwp_serial}.fits")
    fits.writeto(output_fits_path, science_bias_scattered_light_subtracted, overwrite=True)

    print(f"Saved: {output_fits_path}")
