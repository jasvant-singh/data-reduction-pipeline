from scipy.optimize import curve_fit
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import random
import statistics
from statistics import mode
import os
import astroscrappy
import itertools
#from scipy.interpolate import interp2d
#from shapely.geometry import LineString
#from sklearn.preprocessing import normalize
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from pylab import *
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import medfilt2d
from scipy.optimize import differential_evolution
from scipy.signal import correlate
#from itertools import product
#from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from astropy import units as unit
import scipy.signal as signal
from scipy.signal import savgol_filter
import scipy.stats as stats
import bisect
import warnings
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
import re
warnings.filterwarnings('ignore')



detector_pixels = 1024
CCD_gain = 5.5

star_name = 'YGem'
set_number = 'Set_1'
observation_session = '2024-2025'

grating_choice = 1  #  Choose 1 for Red Grating; Choose 2 for Blue Grating

manual_oe_spectral_shift_condition = False
manual_oe_spectral_shift = -1

manual_0_22pt5_spectral_shift_condition = False
manual_0_22pt5_spectral_shift = 1                   # Use +ve value of spectral shift if HWP-0 wavelength value is less than HWP-22pt5 wavelength values and vice-versa 
manual_45_67pt5_spectral_shift_condition = False
manual_45_67pt5_spectral_shift = 1
manual_22pt5_67pt5_spectral_shift_condition = False
manual_22pt5_67pt5_spectral_shift = 2



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





def find_pixel_shift(spec1, spec2):
    # Normalize spectra
    spec1_normalized = (spec1 - np.mean(spec1)) / np.std(spec1)
    spec2_normalized = (spec2 - np.mean(spec2)) / np.std(spec2)
    #spec1_normalized = spec1
    #spec2_normalized = spec2
    # Perform cross-correlation
    correlation = correlate(spec1_normalized, spec2_normalized, mode='full')
    # Find shift that maximizes correlation
    shift = np.argmax(correlation) - len(spec1) + 1
    
    return shift




def OrderExtraction (science_frame_fits, sky_frame_fits, x_trace, y_trace, sigma_FWHM, sky_scale_factor):
    
    order_flux_star = []
    order_flux_sky = []
    
    science_frame = science_frame_fits
    sky = sky_frame_fits
    
    #sigma_FWHM_multiple = int(7/sigma_FWHM)
    sigma_FWHM_multiple = 4
    for i in range (len(y_trace)):
        flux_col = []
        flux_col_sky = []
        for j in range (len(y_trace[i])):
            sum_flux = 0
            sum_flux_sky = 0   
            for k in range (-int(sigma_FWHM_multiple*sigma_FWHM), int(sigma_FWHM_multiple*sigma_FWHM)+1):
                if y_trace[i][j]+k >= 0 and y_trace[i][j]+k < detector_pixels:
                    sum_flux = sum_flux + science_frame[y_trace[i][j]+k][x_trace[i][j]]
                    sum_flux_sky = sum_flux_sky + sky[y_trace[i][j]+k][x_trace[i][j]]
                """
                if x_trace[i][j] > 0 and x_trace[i][j] < (len(x_trace[i]) - 1):
                    fragmant = []
                    for m in range (-1, 2):
                        frag = []
                        for n in range (-1, 2):
                            if y_trace[i][j]+m >= 0 and y_trace[i][j]+m < detector_pixels and x_trace[i][j]+n >= 0 and x_trace[i][j]+n < detector_pixels:
                                frag.append(sky_0[y_trace[i][j]+m][x_trace[i][j]+n])
                        fragmant.append(frag)
                    fragmant = np.array(fragmant)
                    sky_0[y_trace[i][j]][x_trace[i][j]] = np.median(fragmant)
                if y_trace[i][j]+k >= 0 and y_trace[i][j]+k < detector_pixels:                                          
                    sum_flux_sky = sum_flux_sky + sky_0[y_trace[i][j]+k][x_trace[i][j]]
                """
            flux_col.append(sum_flux)
            flux_col_sky.append(sum_flux_sky)
        flux_col = np.array(flux_col)
        flux_col_sky = np.array(flux_col_sky)
        order_flux_star.append(flux_col)  
        order_flux_sky.append(flux_col_sky*sky_scale_factor)       
            
    return (order_flux_star, order_flux_sky)
            




###########  Input all sciencee (bias, scattered light, etc. substracted) and sky (bias substracted) fits files  ###############

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)

param_file = f'{project_root}/parameters.txt' 

starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median = load_parameters(param_file)

science_frame_starting_order = starting_order
science_frame_NumberOfPeaks = NumberOfPeaks
sigma_FWHM = CD_sigma_FWHM


science_frame_path = f'{project_root}/intermediate/science/HD61421_RedCD-FilterIn_20s_HWPPosition-0_EncVal-0pt63_1.fits'
sky_frame_path = f'{project_root}/intermediate/sky/Sky_HD61421_RedCD-FilterIn_300s_HWPPosition-0_EncVal-0pt45_1.fits'

# --- Step 1: Load Science Frames ---

science_frame_41_path = f'{project_root}/intermediate/science/scattered_light_substraction/science_bias_scattered_light_subtracted_1.fits'
science_frame_42_path = f'{project_root}/intermediate/science/scattered_light_substraction/science_bias_scattered_light_subtracted_2.fits'
science_frame_43_path = f'{project_root}/intermediate/science/scattered_light_substraction/science_bias_scattered_light_subtracted_3.fits'
science_frame_44_path = f'{project_root}/intermediate/science/scattered_light_substraction/science_bias_scattered_light_subtracted_4.fits'

def load_valid_image(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data
        raise ValueError(f"No valid 2D image data found in {path}")

science_frame_41 = load_valid_image(science_frame_41_path)
science_frame_42 = load_valid_image(science_frame_42_path)
science_frame_43 = load_valid_image(science_frame_43_path)
science_frame_44 = load_valid_image(science_frame_44_path)

# --- Step 2: Load Sky Frames from File List ---
with open(f'{project_root}/intermediate/sky/bias_subtracted/bias_subtracted_paths.txt', "r") as f:
    sky_frame_paths = [line.strip() for line in f if line.strip()]

# Dictionary to store data cubes by HWP serial
sky_data_cubes = {}

for file_path in sky_frame_paths:
    # ~ print(f"Processing file: {file_path}")
    hwp_serial = extract_hwp_serial(file_path)

    # Load FITS data
    data_cube = fits.getdata(file_path)
    sky_data_cubes[hwp_serial] = data_cube

sky_0_cube = sky_data_cubes[1]
sky_0 = sky_0_cube[0][:][:]

sky_22pt5_cube = sky_data_cubes[2]
sky_22pt5 = sky_22pt5_cube[0][:][:]

sky_45_cube = sky_data_cubes[3]
sky_45 = sky_45_cube[0][:][:]

sky_67pt5_cube = sky_data_cubes[4]
sky_67pt5 = sky_67pt5_cube[0][:][:]

# Ensure all frames are valid before proceeding
for hwp, frame in zip([0, 22.5, 45, 67.5], [sky_0, sky_22pt5, sky_45, sky_67pt5]):
    if frame is None:
        raise RuntimeError(f"Sky frame for HWP {hwp} not loaded properly.")
        
        
        
        
        
        

with fits.open(science_frame_path) as hdul:
    # Get the header of the primary HDU (Header/Data Unit)
    header = hdul[0].header
    science_frame_exposure = header['EXPOSURE']
    observation_date = header['FRAME']
    observation_date = observation_date[0:10]
    print("Observation Date: " + str(observation_date))
    star_name = star_name + "_" + str(observation_date)

with fits.open(sky_frame_path) as hdul:
    # Get the header of the primary HDU (Header/Data Unit)
    header = hdul[0].header
    sky_frame_exposure =  header['EXPOSURE']
    
science_sky_scale_factor = science_frame_exposure/sky_frame_exposure









traced_dir = f'{project_root}/intermediate/science/traced_orders'  # <-- Change this path

orders_41, peaks_41, xcor_41, ycor_41 = load_traced_orders_from_txt(f'{traced_dir}/1')   #  Load from text file
orders_42, peaks_42, xcor_42, ycor_42 = load_traced_orders_from_txt(f'{traced_dir}/2')     
orders_43, peaks_43, xcor_43, ycor_43 = load_traced_orders_from_txt(f'{traced_dir}/3')
orders_44, peaks_44, xcor_44, ycor_44 = load_traced_orders_from_txt(f'{traced_dir}/4')


len_order = [len(orders_41), len(orders_42), len(orders_43), len(orders_44)]
min_len_order = min(len_order)


while len(orders_41) > min_len_order:
    orders_41.pop(0)
    peaks_41.pop(0)
    xcor_41.pop(0)
    ycor_41.pop(0)

while len(orders_42) > min_len_order:        
    orders_42.pop(0)
    peaks_42.pop(0)
    xcor_42.pop(0)
    ycor_42.pop(0)
        
while len(orders_43) > min_len_order:
    orders_43.pop(0)
    peaks_43.pop(0)
    xcor_43.pop(0)
    ycor_43.pop(0)
        
while len(orders_44) > min_len_order:
    orders_44.pop(0)
    peaks_44.pop(0)
    xcor_44.pop(0)
    ycor_44.pop(0)
        

if grating_choice == 1:
    wave_calib_first_order_minus_one = 27
    wave_calib_last_order_plus_one = 49
elif grating_choice == 2:
    wave_calib_first_order_minus_one = 43
    wave_calib_last_order_plus_one = 66    

if orders_41[len(orders_41)-1] == wave_calib_first_order_minus_one:
    orders_41.pop(len(orders_41)-1)
    orders_41.pop(len(orders_41)-1)
    peaks_41.pop(len(peaks_41)-1)
    peaks_41.pop(len(peaks_41)-1)
    xcor_41.pop(len(xcor_41)-1)
    xcor_41.pop(len(xcor_41)-1)
    ycor_41.pop(len(ycor_41)-1)
    ycor_41.pop(len(ycor_41)-1)
    
    orders_42.pop(len(orders_42)-1)
    orders_42.pop(len(orders_42)-1)
    peaks_42.pop(len(peaks_42)-1)
    peaks_42.pop(len(peaks_42)-1)
    xcor_42.pop(len(xcor_42)-1)
    xcor_42.pop(len(xcor_42)-1)
    ycor_42.pop(len(ycor_42)-1)
    ycor_42.pop(len(ycor_42)-1)
    
    orders_43.pop(len(orders_43)-1)
    orders_43.pop(len(orders_43)-1)
    peaks_43.pop(len(peaks_43)-1)
    peaks_43.pop(len(peaks_43)-1)
    xcor_43.pop(len(xcor_43)-1)
    xcor_43.pop(len(xcor_43)-1)
    ycor_43.pop(len(ycor_43)-1)
    ycor_43.pop(len(ycor_43)-1)
    
    orders_44.pop(len(orders_44)-1)
    orders_44.pop(len(orders_44)-1)
    peaks_44.pop(len(peaks_44)-1)
    peaks_44.pop(len(peaks_44)-1)
    xcor_44.pop(len(xcor_44)-1)
    xcor_44.pop(len(xcor_44)-1)
    ycor_44.pop(len(ycor_44)-1)
    ycor_44.pop(len(ycor_44)-1)
    
if orders_41[0] == wave_calib_last_order_plus_one:
    orders_41.pop(0)
    orders_41.pop(0)
    peaks_41.pop(0)
    peaks_41.pop(0)
    xcor_41.pop(0)
    xcor_41.pop(0)
    ycor_41.pop(0)
    ycor_41.pop(0)
    
    orders_42.pop(0)
    orders_42.pop(0)
    peaks_42.pop(0)
    peaks_42.pop(0)
    xcor_42.pop(0)
    xcor_42.pop(0)
    ycor_42.pop(0)
    ycor_42.pop(0)
    
    orders_43.pop(0)
    orders_43.pop(0)
    peaks_43.pop(0)
    peaks_43.pop(0)
    xcor_43.pop(0)
    xcor_43.pop(0)
    ycor_43.pop(0)
    ycor_43.pop(0)
    
    orders_44.pop(0)
    orders_44.pop(0)
    peaks_44.pop(0)
    peaks_44.pop(0)
    xcor_44.pop(0)
    xcor_44.pop(0)
    ycor_44.pop(0)
    ycor_44.pop(0)



orders_41 = np.array(orders_41)
orders_42 = np.array(orders_42)
orders_43 = np.array(orders_43)
orders_44 = np.array(orders_44)

peaks_41 = np.array(peaks_41)
peaks_42 = np.array(peaks_42)
peaks_43 = np.array(peaks_43)
peaks_44 = np.array(peaks_44)

xcor_41 = np.array(xcor_41)
xcor_42 = np.array(xcor_42)
xcor_43 = np.array(xcor_43)
xcor_44 = np.array(xcor_44)

ycor_41 = np.array(ycor_41)
ycor_42 = np.array(ycor_42)
ycor_43 = np.array(ycor_43)
ycor_44 = np.array(ycor_44)

orders_41 = orders_41.astype('int32')
orders_42 = orders_42.astype('int32')
orders_43 = orders_43.astype('int32')
orders_44 = orders_44.astype('int32')

print(orders_41)


science_frame_41, science_frame_42, science_frame_43, science_frame_44 = np.rot90(science_frame_41), np.rot90(science_frame_42), np.rot90(science_frame_43), np.rot90(science_frame_44)
sky_0, sky_22pt5, sky_45, sky_67pt5 = np.rot90(sky_0), np.rot90(sky_22pt5), np.rot90(sky_45), np.rot90(sky_67pt5)



# Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
bad_pixels = np.where(science_frame_41 < 0)
science_frame_41[bad_pixels] = 0
bad_pixels = np.where(science_frame_42 < 0)
science_frame_42[bad_pixels] =0
bad_pixels = np.where(science_frame_43 < 0)
science_frame_43[bad_pixels] = 0
bad_pixels = np.where(science_frame_44 < 0)
science_frame_44[bad_pixels] = 0

bad_pixels = np.where(sky_0 < 0)
sky_0[bad_pixels] = 0
bad_pixels = np.where(sky_22pt5 < 0)
sky_22pt5[bad_pixels] = 0
bad_pixels = np.where(sky_45 < 0)
sky_45[bad_pixels] = 0
bad_pixels = np.where(sky_67pt5 < 0)
sky_67pt5[bad_pixels] = 0



science_frame_1_flux_ADU, sky_frame_1_flux_ADU = OrderExtraction(science_frame_41, sky_0, xcor_41, ycor_41, sigma_FWHM, science_sky_scale_factor)
science_frame_2_flux_ADU, sky_frame_2_flux_ADU = OrderExtraction(science_frame_42, sky_22pt5, xcor_42, ycor_42, sigma_FWHM, science_sky_scale_factor)
science_frame_3_flux_ADU, sky_frame_3_flux_ADU = OrderExtraction(science_frame_43, sky_45, xcor_43, ycor_43, sigma_FWHM, science_sky_scale_factor)
science_frame_4_flux_ADU, sky_frame_4_flux_ADU = OrderExtraction(science_frame_44, sky_67pt5, xcor_44, ycor_44, sigma_FWHM, science_sky_scale_factor)


science_frame_1_flux = []
science_frame_2_flux = []
science_frame_3_flux = []
science_frame_4_flux = []

science_frame_1_flux_err = []
science_frame_2_flux_err = []
science_frame_3_flux_err = []
science_frame_4_flux_err = []
    
for i in range (len(orders_41)):
    science_frame_1_flux_order = []
    science_frame_2_flux_order = []
    science_frame_3_flux_order = []
    science_frame_4_flux_order = []

    science_frame_1_flux_err_order = []
    science_frame_2_flux_err_order = []
    science_frame_3_flux_err_order = []
    science_frame_4_flux_err_order = []
    
    for j in range (len(science_frame_1_flux_ADU[0])):
        science_frame_1_flux_order.append(CCD_gain * (science_frame_1_flux_ADU[i][j] - sky_frame_1_flux_ADU[i][j]))
        science_frame_2_flux_order.append(CCD_gain * (science_frame_2_flux_ADU[i][j] - sky_frame_2_flux_ADU[i][j]))
        science_frame_3_flux_order.append(CCD_gain * (science_frame_3_flux_ADU[i][j] - sky_frame_3_flux_ADU[i][j]))
        science_frame_4_flux_order.append(CCD_gain * (science_frame_4_flux_ADU[i][j] - sky_frame_4_flux_ADU[i][j]))

        science_frame_1_flux_err_order.append(np.sqrt(CCD_gain * (science_frame_1_flux_ADU[i][j] + sky_frame_1_flux_ADU[i][j])))
        science_frame_2_flux_err_order.append(np.sqrt(CCD_gain * (science_frame_2_flux_ADU[i][j] + sky_frame_2_flux_ADU[i][j])))
        science_frame_3_flux_err_order.append(np.sqrt(CCD_gain * (science_frame_3_flux_ADU[i][j] + sky_frame_3_flux_ADU[i][j])))
        science_frame_4_flux_err_order.append(np.sqrt(CCD_gain * (science_frame_4_flux_ADU[i][j] + sky_frame_4_flux_ADU[i][j])))
    
    science_frame_1_flux.append(science_frame_1_flux_order)
    science_frame_2_flux.append(science_frame_2_flux_order)
    science_frame_3_flux.append(science_frame_3_flux_order)
    science_frame_4_flux.append(science_frame_4_flux_order)
    
    science_frame_1_flux_err.append(science_frame_1_flux_err_order)
    science_frame_2_flux_err.append(science_frame_2_flux_err_order)
    science_frame_3_flux_err.append(science_frame_3_flux_err_order)
    science_frame_4_flux_err.append(science_frame_4_flux_err_order)


for i in range (len(orders_41)):
    for j in range (1, len(science_frame_1_flux_ADU[0])-1):
        science_frame_1_flux[i][j] = np.median(np.array([science_frame_1_flux[i][j-1], science_frame_1_flux[i][j], science_frame_1_flux[i][j+1]]))
        science_frame_2_flux[i][j] = np.median(np.array([science_frame_2_flux[i][j-1], science_frame_2_flux[i][j], science_frame_2_flux[i][j+1]]))
        science_frame_3_flux[i][j] = np.median(np.array([science_frame_3_flux[i][j-1], science_frame_3_flux[i][j], science_frame_3_flux[i][j+1]]))
        science_frame_4_flux[i][j] = np.median(np.array([science_frame_4_flux[i][j-1], science_frame_4_flux[i][j], science_frame_4_flux[i][j+1]]))
    

I_0_o_beforeWPshiftCorrection = []
I_0_e_beforeWPshiftCorrection = []
I_22pt5_o_beforeWPshiftCorrection = []
I_22pt5_e_beforeWPshiftCorrection = []
I_45_o_beforeWPshiftCorrection = []
I_45_e_beforeWPshiftCorrection = []
I_67pt5_o_beforeWPshiftCorrection = []
I_67pt5_e_beforeWPshiftCorrection = []

I_0_o_err_beforeWPshiftCorrection = []
I_0_e_err_beforeWPshiftCorrection = []
I_22pt5_o_err_beforeWPshiftCorrection = []
I_22pt5_e_err_beforeWPshiftCorrection = []
I_45_o_err_beforeWPshiftCorrection = []
I_45_e_err_beforeWPshiftCorrection = []
I_67pt5_o_err_beforeWPshiftCorrection = []
I_67pt5_e_err_beforeWPshiftCorrection = []

for i in range (len(orders_41)):
    if i%2 != 0:
        I_0_o_beforeWPshiftCorrection.append(science_frame_1_flux[i])
        I_22pt5_o_beforeWPshiftCorrection.append(science_frame_2_flux[i])
        I_45_o_beforeWPshiftCorrection.append(science_frame_3_flux[i])
        I_67pt5_o_beforeWPshiftCorrection.append(science_frame_4_flux[i])
        
        I_0_o_err_beforeWPshiftCorrection.append(science_frame_1_flux_err[i])
        I_22pt5_o_err_beforeWPshiftCorrection.append(science_frame_2_flux_err[i])
        I_45_o_err_beforeWPshiftCorrection.append(science_frame_3_flux_err[i])
        I_67pt5_o_err_beforeWPshiftCorrection.append(science_frame_4_flux_err[i])
    else:
        I_0_e_beforeWPshiftCorrection.append(science_frame_1_flux[i])
        I_22pt5_e_beforeWPshiftCorrection.append(science_frame_2_flux[i])
        I_45_e_beforeWPshiftCorrection.append(science_frame_3_flux[i])
        I_67pt5_e_beforeWPshiftCorrection.append(science_frame_4_flux[i])
        
        I_0_e_err_beforeWPshiftCorrection.append(science_frame_1_flux_err[i])
        I_22pt5_e_err_beforeWPshiftCorrection.append(science_frame_2_flux_err[i])
        I_45_e_err_beforeWPshiftCorrection.append(science_frame_3_flux_err[i])
        I_67pt5_e_err_beforeWPshiftCorrection.append(science_frame_4_flux_err[i])
        


orders_41_wc = list(set(orders_41))
orders_41_wc = np.array(orders_41_wc)
orders_41_wc.sort()
orders_41_wc = list(orders_41_wc)
orders_41_wc.reverse()


pixels = []

I_0_o1 = []
I_0_e1 = []
I_22pt5_o1 = []
I_22pt5_e1 = []
I_45_o1 = []
I_45_e1 = []
I_67pt5_o1 = []
I_67pt5_e1 = []

if grating_choice == 1:
    pixel_shift_comparison_order = 35
elif grating_choice == 2:
    pixel_shift_comparison_order = 51

x_new = np.linspace(0, detector_pixels-1, detector_pixels)
x_new = x_new.astype(int)

pixels_beforeWPshiftCorrection = []
for i in range (len(orders_41)):
    pixels_beforeWPshiftCorrection.append(x_new)

pixel_shift_order = []
for i in range (len(I_67pt5_o_beforeWPshiftCorrection)):
    if orders_41_wc[i] == pixel_shift_comparison_order: 
        pixel_shift_order.append(find_pixel_shift(I_67pt5_o_beforeWPshiftCorrection[i], I_67pt5_e_beforeWPshiftCorrection[i]))
pixel_shift = mode(pixel_shift_order)
print("o and e ray pixel shift: " + str(pixel_shift))

if manual_oe_spectral_shift_condition == True:
    pixel_shift = manual_oe_spectral_shift

for i in range (int(len(orders_41)/2)):
    
    pixels_order = []
    I_0_o_order = []
    I_0_e_order = []
    I_22pt5_o_order = []
    I_22pt5_e_order = []
    I_45_o_order = []
    I_45_e_order = []
    I_67pt5_o_order = []
    I_67pt5_e_order = []
    
    if pixel_shift < 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_beforeWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o_beforeWPshiftCorrection[i][j])
            I_22pt5_o_order.append(I_22pt5_o_beforeWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o_beforeWPshiftCorrection[i][j])
            I_67pt5_o_order.append(I_67pt5_o_beforeWPshiftCorrection[i][j])
        
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_beforeWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o_beforeWPshiftCorrection[i][j])
            I_0_e_order.append(I_0_e_beforeWPshiftCorrection[i][j])
            I_22pt5_o_order.append(I_22pt5_o_beforeWPshiftCorrection[i][j])
            I_22pt5_e_order.append(I_22pt5_e_beforeWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o_beforeWPshiftCorrection[i][j])
            I_45_e_order.append(I_45_e_beforeWPshiftCorrection[i][j])
            I_67pt5_o_order.append(I_67pt5_o_beforeWPshiftCorrection[i][j])
            I_67pt5_e_order.append(I_67pt5_e_beforeWPshiftCorrection[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_0_e_order.append(I_0_e_beforeWPshiftCorrection[i][j])
            I_22pt5_e_order.append(I_22pt5_e_beforeWPshiftCorrection[i][j])
            I_45_e_order.append(I_45_e_beforeWPshiftCorrection[i][j])
            I_67pt5_e_order.append(I_67pt5_e_beforeWPshiftCorrection[i][j])
    
    elif pixel_shift >= 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_beforeWPshiftCorrection[i][j])
            I_0_e_order.append(I_0_e_beforeWPshiftCorrection[i][j])
            I_22pt5_e_order.append(I_22pt5_e_beforeWPshiftCorrection[i][j])
            I_45_e_order.append(I_45_e_beforeWPshiftCorrection[i][j])
            I_67pt5_e_order.append(I_67pt5_e_beforeWPshiftCorrection[i][j])
        
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_beforeWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o_beforeWPshiftCorrection[i][j])
            I_0_e_order.append(I_0_e_beforeWPshiftCorrection[i][j])
            I_22pt5_o_order.append(I_22pt5_o_beforeWPshiftCorrection[i][j])
            I_22pt5_e_order.append(I_22pt5_e_beforeWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o_beforeWPshiftCorrection[i][j])
            I_45_e_order.append(I_45_e_beforeWPshiftCorrection[i][j])
            I_67pt5_o_order.append(I_67pt5_o_beforeWPshiftCorrection[i][j])
            I_67pt5_e_order.append(I_67pt5_e_beforeWPshiftCorrection[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):          
            I_0_o_order.append(I_0_o_beforeWPshiftCorrection[i][j])
            I_22pt5_o_order.append(I_22pt5_o_beforeWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o_beforeWPshiftCorrection[i][j])
            I_67pt5_o_order.append(I_67pt5_o_beforeWPshiftCorrection[i][j])
    
    
    pixels.append(pixels_order)
    I_0_o1.append(I_0_o_order)
    I_0_e1.append(I_0_e_order)
    I_22pt5_o1.append(I_22pt5_o_order)
    I_22pt5_e1.append(I_22pt5_e_order)
    I_45_o1.append(I_45_o_order)
    I_45_e1.append(I_45_e_order)
    I_67pt5_o1.append(I_67pt5_o_order)
    I_67pt5_e1.append(I_67pt5_e_order)



pixels = []
I_0_o = []
I_0_e = []
I_22pt5_o = []
I_22pt5_e = []
I_45_o = []
I_45_e = []
I_67pt5_o = []
I_67pt5_e = []



pixels_afterWPshiftCorrection = []
for i in range (len(orders_41)):
    pixels_afterWPshiftCorrection.append(x_new)

pixel_shift_order = []
for i in range (len(I_0_o1)):
    if orders_41_wc[i] == pixel_shift_comparison_order: 
        pixel_shift_order.append(find_pixel_shift(I_0_o1[i], I_22pt5_o1[i]))
pixel_shift = mode(pixel_shift_order)
print("HWP-0 and HWP-22.5 frames pixel shift: " + str(pixel_shift))

if manual_0_22pt5_spectral_shift_condition == True:
    pixel_shift = manual_0_22pt5_spectral_shift

for i in range (int(len(orders_41)/2)):
    
    pixels_order = []
    I_0_o_order = []
    I_0_e_order = []
    I_22pt5_o_order = []
    I_22pt5_e_order = []
    
    if pixel_shift < 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o1[i][j])
            I_0_e_order.append(I_0_e1[i][j])
        
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o1[i][j])
            I_0_e_order.append(I_0_e1[i][j])
            I_22pt5_o_order.append(I_22pt5_o1[i][j])
            I_22pt5_e_order.append(I_22pt5_e1[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_22pt5_o_order.append(I_22pt5_o1[i][j])
            I_22pt5_e_order.append(I_22pt5_e1[i][j])
    
    elif pixel_shift >= 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_22pt5_o_order.append(I_22pt5_o1[i][j])
            I_22pt5_e_order.append(I_22pt5_e1[i][j])
            
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o1[i][j])
            I_0_e_order.append(I_0_e1[i][j])
            I_22pt5_o_order.append(I_22pt5_o1[i][j])
            I_22pt5_e_order.append(I_22pt5_e1[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_0_o_order.append(I_0_o1[i][j])
            I_0_e_order.append(I_0_e1[i][j])
            
    pixels.append(pixels_order)
    I_0_o.append(I_0_o_order)
    I_0_e.append(I_0_e_order)
    I_22pt5_o.append(I_22pt5_o_order)
    I_22pt5_e.append(I_22pt5_e_order)


#pixels = []
#I_0_o = []
#_0_e = []

pixel_shift_order = []
for i in range (len(I_0_o1)):
    if orders_41_wc[i] == pixel_shift_comparison_order:
        pixel_shift_order.append(find_pixel_shift(I_45_o1[i], I_67pt5_o1[i]))
pixel_shift = mode(pixel_shift_order)
print("HWP-45 and HWP-67.5 frames pixel shift: " + str(pixel_shift))

if manual_45_67pt5_spectral_shift_condition == True:
    pixel_shift = manual_45_67pt5_spectral_shift

for i in range (int(len(orders_41)/2)):
    
    pixels_order = []   
    I_45_o_order = []
    I_45_e_order = []
    I_67pt5_o_order = []
    I_67pt5_e_order = []
    
    if pixel_shift < 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o1[i][j])
            I_45_e_order.append(I_45_e1[i][j])
        
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o1[i][j])
            I_45_e_order.append(I_45_e1[i][j])
            I_67pt5_o_order.append(I_67pt5_o1[i][j])
            I_67pt5_e_order.append(I_67pt5_e1[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_67pt5_o_order.append(I_67pt5_o1[i][j])
            I_67pt5_e_order.append(I_67pt5_e1[i][j])
    
    elif pixel_shift >= 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_67pt5_o_order.append(I_67pt5_o1[i][j])
            I_67pt5_e_order.append(I_67pt5_e1[i][j])
            
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o1[i][j])
            I_45_e_order.append(I_45_e1[i][j])
            I_67pt5_o_order.append(I_67pt5_o1[i][j])
            I_67pt5_e_order.append(I_67pt5_e1[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_45_o_order.append(I_45_o1[i][j])
            I_45_e_order.append(I_45_e1[i][j])
            
    pixels.append(pixels_order)
    I_45_o.append(I_45_o_order)
    I_45_e.append(I_45_e_order)
    I_67pt5_o.append(I_67pt5_o_order)
    I_67pt5_e.append(I_67pt5_e_order)
    
  
    
#pixels = []
#I_0_o = []
#I_0_e = [] 

pixels_final = []
I_0_o_final = []
I_0_e_final = []
I_22pt5_o_final = []
I_22pt5_e_final = []
I_45_o_final = []
I_45_e_final = []
I_67pt5_o_final = []
I_67pt5_e_final = []

pixel_shift_order = []
for i in range (len(I_0_o1)):
    if orders_41_wc[i] == pixel_shift_comparison_order:
        pixel_shift_order.append(find_pixel_shift(I_22pt5_o[i], I_67pt5_o[i]))
pixel_shift = mode(pixel_shift_order)
print("HWP-22.5 and HWP-67.5 frames pixel shift: " + str(pixel_shift))

if manual_22pt5_67pt5_spectral_shift_condition == True:
    pixel_shift = manual_22pt5_67pt5_spectral_shift


for i in range (int(len(orders_41)/2)):
    
    pixels_order = []
    I_0_o_order = []
    I_0_e_order = []
    I_22pt5_o_order = []
    I_22pt5_e_order = []
    I_45_o_order = []
    I_45_e_order = []
    I_67pt5_o_order = []
    I_67pt5_e_order = []
    
    if pixel_shift < 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o[i][j])
            I_0_e_order.append(I_0_e[i][j])
            I_22pt5_o_order.append(I_22pt5_o[i][j])
            I_22pt5_e_order.append(I_22pt5_e[i][j])
        
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o[i][j])
            I_0_e_order.append(I_0_e[i][j])
            I_22pt5_o_order.append(I_22pt5_o[i][j])
            I_22pt5_e_order.append(I_22pt5_e[i][j])
            I_45_o_order.append(I_45_o[i][j])
            I_45_e_order.append(I_45_e[i][j])
            I_67pt5_o_order.append(I_67pt5_o[i][j])
            I_67pt5_e_order.append(I_67pt5_e[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_45_o_order.append(I_45_o[i][j])
            I_45_e_order.append(I_45_e[i][j])
            I_67pt5_o_order.append(I_67pt5_o[i][j])
            I_67pt5_e_order.append(I_67pt5_e[i][j])
    
    elif pixel_shift >= 0:
        for j in range (0, abs(pixel_shift)):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_45_o_order.append(I_45_o[i][j])
            I_45_e_order.append(I_45_e[i][j])
            I_67pt5_o_order.append(I_67pt5_o[i][j])
            I_67pt5_e_order.append(I_67pt5_e[i][j])
            
        for j in range (abs(pixel_shift), detector_pixels):
            pixels_order.append(pixels_afterWPshiftCorrection[i][j])
            I_0_o_order.append(I_0_o[i][j])
            I_0_e_order.append(I_0_e[i][j])
            I_22pt5_o_order.append(I_22pt5_o[i][j])
            I_22pt5_e_order.append(I_22pt5_e[i][j])
            I_45_o_order.append(I_45_o[i][j])
            I_45_e_order.append(I_45_e[i][j])
            I_67pt5_o_order.append(I_67pt5_o[i][j])
            I_67pt5_e_order.append(I_67pt5_e[i][j])
        
        for j in range (detector_pixels-abs(pixel_shift), detector_pixels):
            I_0_o_order.append(I_0_o[i][j])
            I_0_e_order.append(I_0_e[i][j])
            I_22pt5_o_order.append(I_22pt5_o[i][j])
            I_22pt5_e_order.append(I_22pt5_e[i][j])
            
    pixels_final.append(pixels_order)
    I_0_o_final.append(I_0_o_order)
    I_0_e_final.append(I_0_e_order)
    I_22pt5_o_final.append(I_22pt5_o_order)
    I_22pt5_e_final.append(I_22pt5_e_order)
    I_45_o_final.append(I_45_o_order)
    I_45_e_final.append(I_45_e_order)
    I_67pt5_o_final.append(I_67pt5_o_order)
    I_67pt5_e_final.append(I_67pt5_e_order)
  

I_0_o = I_0_o_final
I_0_e = I_0_e_final
I_22pt5_o = I_22pt5_o_final
I_22pt5_e = I_22pt5_e_final
I_45_o = I_45_o_final
I_45_e = I_45_e_final
I_67pt5_o = I_67pt5_o_final
I_67pt5_e = I_67pt5_e_final
  

for i in range (len(I_0_o)):
    for j in range (len(I_0_o[i])):
        if I_0_o[i][j] <= 0:
            I_0_o[i][j] = 1
        if I_0_e[i][j] <= 0:
            I_0_e[i][j] = 1
        if I_22pt5_o[i][j] <= 0:
            I_22pt5_o[i][j] = 1
        if I_22pt5_e[i][j] <= 0:
            I_22pt5_e[i][j] = 1
        if I_45_o[i][j] <= 0:
            I_45_o[i][j] = 1
        if I_45_e[i][j] <= 0:
            I_45_e[i][j] = 1
        if I_67pt5_o[i][j] <= 0:
            I_67pt5_o[i][j] = 1
        if I_67pt5_e[i][j] <= 0:
            I_67pt5_e[i][j] = 1




"""
for i in range (int(len(orders_41)/2)):
    for j in range (1, len(pixels[i])-1):
        I_0_o[i][j] = np.median(np.array([I_0_o[i][j-1], I_0_o[i][j], I_0_o[i][j+1]]))
        I_0_e[i][j] = np.median(np.array([I_0_e[i][j-1], I_0_e[i][j], I_0_e[i][j+1]]))
        I_22pt5_o[i][j] = np.median(np.array([I_22pt5_o[i][j-1], I_22pt5_o[i][j], I_22pt5_o[i][j+1]]))
        I_22pt5_e[i][j] = np.median(np.array([I_22pt5_e[i][j-1], I_22pt5_e[i][j], I_22pt5_e[i][j+1]]))
        I_45_o[i][j] = np.median(np.array([I_45_o[i][j-1], I_45_o[i][j], I_45_o[i][j+1]]))
        I_45_e[i][j] = np.median(np.array([I_45_e[i][j-1], I_45_e[i][j], I_45_e[i][j+1]]))
        I_67pt5_o[i][j] = np.median(np.array([I_67pt5_o[i][j-1], I_67pt5_o[i][j], I_67pt5_o[i][j+1]]))
        I_67pt5_e[i][j] = np.median(np.array([I_67pt5_e[i][j-1], I_67pt5_e[i][j], I_67pt5_e[i][j+1]]))

"""

I_0_o_err = []
I_0_e_err = []
I_22pt5_o_err = []
I_22pt5_e_err = []
I_45_o_err = []
I_45_e_err = []
I_67pt5_o_err = []
I_67pt5_e_err = []

for i in range (int(len(orders_41)/2)):
    I_0_o_err_order = []
    I_0_e_err_order = []
    I_22pt5_o_err_order = []
    I_22pt5_e_err_order = []
    I_45_o_err_order = []
    I_45_e_err_order = []
    I_67pt5_o_err_order = []
    I_67pt5_e_err_order = []
    
    for j in range (detector_pixels):
        I_0_o_err_order.append(np.sqrt(I_0_o[i][j]))
        I_0_e_err_order.append(np.sqrt(I_0_e[i][j]))
        I_22pt5_o_err_order.append(np.sqrt(I_22pt5_o[i][j]))
        I_22pt5_e_err_order.append(np.sqrt(I_22pt5_e[i][j]))
        I_45_o_err_order.append(np.sqrt(I_45_o[i][j]))
        I_45_e_err_order.append(np.sqrt(I_45_e[i][j]))
        I_67pt5_o_err_order.append(np.sqrt(I_67pt5_o[i][j]))
        I_67pt5_e_err_order.append(np.sqrt(I_67pt5_e[i][j]))
    
    I_0_o_err.append(I_0_o_err_order)
    I_0_e_err.append(I_0_e_err_order)
    I_22pt5_o_err.append(I_22pt5_o_err_order)
    I_22pt5_e_err.append(I_22pt5_e_err_order)
    I_45_o_err.append(I_45_o_err_order)
    I_45_e_err.append(I_45_e_err_order)
    I_67pt5_o_err.append(I_67pt5_o_err_order)
    I_67pt5_e_err.append(I_67pt5_e_err_order)
    
    

path = f"{project_root}/output/{observation_session}/{star_name}"
os.makedirs(path, exist_ok=True)

if grating_choice == 1:
    path = path + '/RedCD/'
elif grating_choice == 2:
    path = path + '/BlueCD/'

if not os.path.exists(path):
    os.mkdir(path)
    
path = path + set_number

if not os.path.exists(path):
    os.mkdir(path)

for i in range (int(len(orders_41)/2)):
    
    order_number = orders_41[(2*i)]
        
        
    dt = np.dtype([('pixel', 'd'), ('I_0_o', 'd'), ('I_0_e', 'd'), ('I_22pt5_o', 'd'), ('I_22pt5_e', 'd'), ('I_45_o', 'd'), ('I_45_e', 'd'), ('I_67pt5_o', 'd'), ('I_67pt5_e', 'd'), ('I_0_o_err', 'd'), ('I_0_e_err', 'd'), ('I_22pt5_o_err', 'd'), ('I_22pt5_e_err', 'd'), ('I_45_o_err', 'd'), ('I_45_e_err', 'd'), ('I_67pt5_o_err', 'd'), ('I_67pt5_e_err', 'd')])  
    a = np.zeros(detector_pixels, dt)                        # Saving wavelength and the corresponding
    a['pixel'] = pixels[i]        
    a['I_0_o'] = I_0_o[i]
    a['I_0_e'] = I_0_e[i]    
    a['I_22pt5_o'] = I_22pt5_o[i]
    a['I_22pt5_e'] = I_22pt5_e[i]
    a['I_45_o'] = I_45_o[i]
    a['I_45_e'] = I_45_e[i]    
    a['I_67pt5_o'] = I_67pt5_o[i]
    a['I_67pt5_e'] = I_67pt5_e[i]
    a['I_0_o_err'] = I_0_o_err[i]
    a['I_0_e_err'] = I_0_e_err[i]    
    a['I_22pt5_o_err'] = I_22pt5_o_err[i]
    a['I_22pt5_e_err'] = I_22pt5_e_err[i]
    a['I_45_o_err'] = I_45_o_err[i]
    a['I_45_e_err'] = I_45_e_err[i]    
    a['I_67pt5_o_err'] = I_67pt5_o_err[i]
    a['I_67pt5_e_err'] = I_67pt5_e_err[i]
    #np.savetxt('C:\\Users\\Mudit Shrivastav\\.ipython\\Science_spectra\\BetUMa\\BetUMa_9_IntensityTestBeforeEffCorr.txt', a, '%.5f', delimiter = ',')
    filename = f"{star_name}_{set_number}_Order-{order_number}_IntensityBeforeEffCorr.txt"
    output_path = os.path.join(path, filename)
    np.savetxt(output_path, a, fmt='%.3f', delimiter='    ')
