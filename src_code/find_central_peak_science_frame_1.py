#!/usr/bin/env python3

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
import re
import os
warnings.filterwarnings('ignore')

CCD_gain = 5.5
detector_pixels = 1024

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def load_parameters(param_file):
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=')
                params[key.strip()] = float(val.strip())
    return params['starting_order'], params['number_of_peaks'], params['plot_flag']


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


#plt.imshow(science_frame_41, cmap = "gray")
#plt.ylim(1, 1024)
#plt.show()
def OrderPeak (science_frame_Fits, start_order, NumberOfPeaks, plot_flag):
    
    #CCD_gain = 5.5
    
    starting_order = start_order

    CD_peaks = NumberOfPeaks  #  redCD_peaks = 46 for observations taken with 1.2m telescope; redCD_peaks = 42 for observations taken with 2.5m telescope   


    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240430\\HR5191\\RedCD\\HR5191_RedCD-FilterIn_60s_HWPPosition-22pt5_EncVal-1574pt64_2.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240429\\HD147084\\HD147084_RedCD-FilterIn_180s_HWPPosition-22pt5_EncVal-1574pt64_2.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240508\\AG_Dra\\RedCD\\Set_1\\AGDra_RedCD-FilterIn_600s_HWPPosition-0_EncVal-0pt54_1.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240428\\HD47105\\RedCD\\HD47105_RedCD-FilterIn_300s_HWPPosition-0_EncVal-0pt81_1.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240429\\HD61421\\RedCD\\HD61421_RedCD-FilterIn_15s_HWPPosition-0_EncVal-0pt54_1.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240429\\AG_Peg\\Set_1\\AGPeg_RedCD-FilterIn_600s_HWPPosition-0_EncVal-0pt72_1.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240310\\YGem\\Set2\\Calib-Hal-25s_YGem_RedCD-FilterIn_600s_HWPPosition-0_EncVal-0pt45_5.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240107\\HD47105\\RedCDFilterIn\\HR47105_V2mag_RedCD-FilterIn_180s_HWPPosition-0_EncVal-0pt54_1.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240112\\BetUMa\\RedCD\\Set_2\\BetUMa_RedCD-FilterIN_180s_HWPPosition-0_EncVal-0pt72_5.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240214\\RW_Hya\\Set_2\\RWHya_RedCD-FilterIn_600s_HWPPosition-22pt5_EncVal-1574pt46_6.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240123\\R_Leo\\RedCD\\Set_3\\RLeo_RedCD-FilterIn_300s_HWPPosition-0_EncVal-0pt45_9.fits')
    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240112\\Jupiter\\Calib-Hal-120s_Jupiter_RedCD-FilterIN_180s_HWPPosition-67pt5_EncVal-4725pt54_4.fits')

    #science_data_cube_41 = fits.getdata('G:\\Arijit_PhD\\ProtoPol_Observations\\20240429\\HD61421\\RedCD\\HD61421_RedCD-FilterIn_15s_HWPPosition-45_EncVal-3150pt27_3.fits')
    #science_data_cube_41 = fits.getdata('/home/jasvant/data_reduction/star_name/data_frames/HD61421_RedCD-FilterIn_20s_HWPPosition-0_EncVal-0pt63_1.fits')



    science_frame_41 = science_frame_Fits

    science_frame_41 = np.rot90(science_frame_41)

    hdu = fits.PrimaryHDU(science_frame_41)
    hdu.writeto(f'{output_path}/science_frame_Rotated.fits', overwrite=True)

    #plt.imshow(science_frame_41, cmap = "gray")
    #plt.ylim(1, 1024)
    #plt.show()

    x_new = np.linspace(0, 1023, 1024)




    ###################################################  Peak Detection  #################################################################



    N = 3    # No. of centre columns to median combine for peak detectection
    master_peak = []
    for i in range(len(science_frame_41)):
        column_trace1 = np.zeros((N))
        for j in range (int(len(science_frame_41)/2 - (N/2)), int(len(science_frame_41)/2 + (N/2))):
            column_trace1[j - int(len(science_frame_41)/2 - (N/2))] = science_frame_41[i][j]  
        master_peak.append(np.median(column_trace1))

    master_peak = np.array(master_peak)    
    

    sigma_FWHM = 1.1
    kernel = Gaussian1DKernel(stddev = sigma_FWHM)
    master_peak = convolve(master_peak, kernel)


    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i+1]
            wid = params[i+2]
            y = y + amp * np.exp( -((x - ctr)/wid)**2)
        return y


    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = func(x_new, *parameterTuple)
        return np.sum((master_peak - val) ** 2.0)

    def generate_Initial_Parameters(x, y):
        # min and max used for bounds
        maxX = max(x)
        maxY = max(y)
        maxXY = max(maxX, maxY)

        parameterBounds = []
        parameterBounds.append([-maxXY, maxXY]) # seach bounds for a
        parameterBounds.append([-maxXY, maxXY]) # seach bounds for b
        parameterBounds.append([-maxXY, maxXY]) # seach bounds for c

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    # generate initial parameter values
    geneticParameters = generate_Initial_Parameters(x_new, master_peak)

    # curve fit the test data
    popt, pcov = curve_fit(func, x_new, master_peak, geneticParameters)
    popt = np.abs(popt)
    #area_gauss = popt[1] * (popt[2]/(1/np.sqrt(2 * np.pi)))
    fit = func(x_new, *popt)
    #fit = np.abs(fit)
    #print(fit)
    #print(popt[0])
    #print("Integrated flux along CD: " + str(area_gauss))
    #plt.plot(x_new, master_peak)
    #plt.plot(x_new, fit , 'r-')
    #plt.show()

    """
    cutoff_peak = 12
    pixel, value = x_new, master_peak

    ax = plt.axes()
    ax.plot(pixel, value, '-')
    ax.axis([0, np.max(pixel), np.min(value), np.max(value) + 1])

    maxTemp = argrelmax(value, order=5)
    maxes = []
    for maxi in maxTemp[0]:
        if value[maxi] > cutoff_peak:
            maxes.append(maxi)

    plt.plot(maxes, value[maxes], 'ro')
    print(maxes)
    #plt.yticks(np.arange(rounddown(value.min()), value.max(), 10))
    #plt.savefig("spectrum1.pdf")
    plt.show()
    """
    """
    peaks, _ = find_peaks(master_peak, height=0)
    #print(peaks)
    peaks_cut = []
    for i in range (len(peaks)):
        if peaks[i]> 400 and peaks[i] < 600:
            peaks_cut.append(peaks[i])
    peaks_cut = np.array(peaks_cut)
    #plt.plot(master_peak)
    #plt.plot(peaks_cut, master_peak[peaks_cut], "x")
    #plt.plot(fit, "--", color="red")
    #plt.show()

    valley_cut = []
    for i in range (len(peaks_cut) - 1):
        valley_cut.append(int((peaks_cut[i] + peaks_cut[i+1])/2))
    valley_cut = np.array(valley_cut)

    continuum_interp = interp1d(valley_cut, master_peak[valley_cut], kind = 'linear', bounds_error = False, fill_value=(valley_cut[0], valley_cut[len(valley_cut)-1]))
    continuum_valley = np.abs(continuum_interp(x_new))
    valley_median = np.median(continuum_valley)
    print(valley_median)

    continuum_interp = interp1d(valley_cut, master_peak[valley_cut], kind = 'linear', bounds_error = False, fill_value=valley_median)
    plt.plot(x_new, continuum_valley)
    plt.show()
    #fit_offset = 0
    #for i in range (len(fit)):
    #    fit[i] = fit[i] + fit_offset
    valley_median = valley_median + 3
    #peaks, _ = find_peaks(master_peak, height=fit)
    """


    spectrum = Spectrum1D(flux=master_peak*unit.Jy, spectral_axis=x_new*unit.pix)

    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = fit_generic_continuum(spectrum)
        
    y_continuum_fitted = g1_fit(x_new*unit.pix)
    y_continuum_fitted = np.array(y_continuum_fitted)
    #f, ax = plt.subplots()  
    #ax.plot(x_new, master_peak)  
    #ax.plot(x_new, y_continuum_fitted)  
    #ax.set_title("Continuum Fitting")  
    #ax.grid(True)





    peaks, _ = find_peaks(master_peak, height=y_continuum_fitted)
    #peak_flux = master_peak[peaks]
    #print(peak_flux)
    #min_peak_flux = min(peak_flux)
    #min_peak_flux_pixel = np.where(peak_flux == min_peak_flux)

    #print(peaks)

    continuum_points = []
    continuum_flux_points = []
    for i in range (1, len(peaks)):
        #if peaks[i] - peaks[i-1]!= 13 or peaks[i] - peaks[i-1]!= 14 or peaks[i] - peaks[i-1]!= 15:
        if peaks[i] - peaks[i-1] <= 13:
            continue
        else:
            continuum_points.append(peaks[i-1] + int((peaks[i] - peaks[i-1])/2))
            continuum_flux_points.append(master_peak[peaks[i-1] + int((peaks[i] - peaks[i-1])/2)])

    continuum_points = np.array(continuum_points)
    continuum_flux_points = np.array(continuum_flux_points)

    continuum_fit = interp1d(continuum_points, continuum_flux_points, kind = 'quadratic', bounds_error = False, fill_value="extrapolate")
    continuum = continuum_fit(x_new)

    #continumm_flux_at_min_peak_flux = continuum[min_peak_flux_pixel]
    #continuum = continuum + int((min_peak_flux - continumm_flux_at_min_peak_flux)/2)
        
    #print(continuum_points)
    #print(continuum_flux_points)

    """
    spectrum = Spectrum1D(flux=continuum*u.Jy, spectral_axis=x_new*u.pix)

    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = fit_generic_continuum(spectrum)
        
    y_continuum_fitted = g1_fit(x_new*u.pix)
    y_continuum_fitted = np.array(y_continuum_fitted)
    """
    #plt.plot(x_new, continuum)
    #plt.plot(x_new, y_continuum_fitted)
    #plt.scatter(continuum_points, continuum_flux_points, s=15, color = 'black')
    #plt.show()

    peaks, _ = find_peaks(master_peak, height=continuum)


    fit_offset = 0
    #redCD_peaks = 43  #  redCD_peaks = 46 for observations taken with 1.2m telescope; redCD_peaks = 42 for observations taken with 2.5m telescope
    while len(peaks) >= CD_peaks:
        fit_offset = fit_offset + 0.1
        for i in range (len(continuum)):
            continuum[i] = continuum[i] + fit_offset
        peaks, _ = find_peaks(master_peak, height=continuum)


    #peaks2 = []
    #for i in range (len(peaks)):
    #    if peaks[i] >= 15 and peaks[i] <= 1009:
    #       peaks2.append(peaks[i])

    #peaks = np.array(peaks2)
    #print(fit_offset)
    #print(peaks)
    #print(len(peaks))

    peaks = list(peaks)
    
    if peaks[len(peaks)-1] >= (detector_pixels - 10):
        peaks.pop(len(peaks)-1)

    if peaks[0] <= 10:
        peaks.pop(0)
    
    if (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 13 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 14 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 15 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 16:
        peaks.pop(len(peaks)-1)
        
    if len(peaks)%2 == 1:
        peaks.pop(0)
        
    #peaks = np.array(peaks)
    peaks = np.array(peaks, dtype=int)

    
    if plot_flag == 1:
        plt.plot(x_new, continuum)
        plt.scatter(continuum_points, continuum_flux_points, s=25, color = 'black')
        plt.plot(master_peak, color = 'tab:blue')
        plt.plot(peaks, master_peak[peaks], "x", color = 'orange')
        plt.plot(x_new, continuum , "--", color="red")
        plt.savefig(f"{output_path}/Peak_Detection_HWP_{hwp_serial}.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    #starting_order = 27
    orders = np.zeros((len(peaks)))
    k = 0
    for i in range (len(peaks)-1, -1, -2):
        orders[i] = starting_order + k
        orders[i-1] = starting_order + k
        k = k + 1

    orders = orders.astype('int32')
    
    print(orders)
    print(peaks)
    
    
    return (orders, peaks)
    
    



param_file = f'{project_root}/parameters.txt' 
input_path = f'{project_root}/intermediate/science'
output_path = os.path.join(input_path, 'central_peak_detection/')
os.makedirs(output_path, exist_ok=True)

starting_order, NumberOfPeaks, plot_flag = load_parameters(param_file)

def read_file_paths(txt_file):
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


# --- Load file paths from your .txt file ---


with open(f'{input_path}/bias_subtracted/bias_subtracted_paths.txt', "r") as f:
    science_frame_paths = [line.strip() for line in f if line.strip()]
    science_frame_paths = [p for p in science_frame_paths
                       if "bias_sub_HD61421" in p and "Sky" not in p and "Dark" not in p and "Calib" not in p]

for file_path in science_frame_paths:
    print(f"Processing file: {file_path}")
    hwp_serial = extract_hwp_serial(file_path)

    # Load FITS data
    science_data_cube_41 = fits.getdata(file_path)
    
    science_frame_41 = science_data_cube_41[0][:][:] 

    # Clean negative pixels
    bad_pixels = np.where(science_frame_41 < 0)
    science_frame_41[bad_pixels] = 0

    # Detect peaks
    order_number, peak_position = OrderPeak(science_frame_41, starting_order, NumberOfPeaks, plot_flag)

    # Ask user to verify the result
    user_input = input("\nAre you satisfied with the peak detection? (y/n): ").strip().lower()

    if user_input == 'y':
        
        print(f"→ HWP Serial: {hwp_serial}")


        dt = np.dtype([('order', 'd'), ('y_peak', 'd')])  
        a = np.zeros(len(order_number), dt)  
        a['order'] = order_number 
        a['y_peak'] = peak_position
        
        
        base_name = file_path.split('/')[-1].replace('.fits', '')
        filename = f"Detected_peak_coordinates_{base_name}_HWP_{hwp_serial}.txt"
        np.savetxt(output_path + filename, a, '%.5f', delimiter='    ')
        print(f"\nPeaks saved to: {output_path}{filename}")

    elif user_input == 'n':
        print("\nPlease edit the parameter file to reset values and re-run the script.")
        print(f"File: {param_file}")
    else:
        print("\nInvalid input. Skipping saving for this file.")
