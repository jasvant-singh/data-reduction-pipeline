from scipy.optimize import curve_fit
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import random
import statistics
from statistics import mode
import os
import lacosmic
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


science_frame_starting_order = 28
science_frame_NumberOfPeaks = 42

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



def Star_FWHM_Determination (science_frame_Fits, start_order, NumberOfPeaks):
    
    #CCD_gain = 5.5
    
    starting_order = start_order

    CD_peaks = NumberOfPeaks 
    
    science_frame_41 = science_frame_Fits

    science_frame_41 = np.rot90(science_frame_41)

    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    x_new = x_new.astype(int)

    ###################################################  Peak Detection  #################################################################

    N = 10    # No. of centre columns to median combine for peak detectection
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
    #fit = func(x_new, *popt)
    #fit = np.abs(fit)
    #print(fit)
    #print(popt[0])
    #print("Integrated flux along CD: " + str(area_gauss))
    #plt.plot(x_new, master_peak)
    #plt.plot(x_new, fit , 'r-')
    #plt.show()



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
    #redCD_peaks = 42  #  redCD_peaks = 46 for observations taken with 1.2m telescope; redCD_peaks = 42 for observations taken with 2.5m telescope
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
    #plt.plot(master_peak)
    #plt.plot(peaks, master_peak[peaks], "x")
    #plt.scatter(continuum_points, continuum[continuum_points], "o")
    #plt.plot(x_new, continuum , "--", color="red")
    #plt.savefig("Peak_Detection_Halogen_20240310.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

    peaks = list(peaks)
    
    if peaks[len(peaks)-1] >= (detector_pixels - 60):
        peaks.pop(len(peaks)-1)

    if peaks[0] <= 10:
        peaks.pop(0)
    
    if (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 13 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 14 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 15 and (peaks[len(peaks)-1] - peaks[len(peaks)-2]) != 16:
        peaks.pop(len(peaks)-1)
        
    if len(peaks)%2 == 1:
        peaks.pop(0)
        
    peaks = np.array(peaks)

    starting_order = 28
    orders = np.zeros((len(peaks)))
    k = 0
    for i in range (len(peaks)-1, -1, -2):
        orders[i] = starting_order + k
        orders[i-1] = starting_order + k
        k = k + 1


    #################################################    Order Trace   ############################################################################


    def guess_gaussian_params(x, y):
        
        #Guess the initial parameters for a Gaussian fit to the data (x, y).
        
        mean = np.mean(x)
        stddev = np.std(x)
        amplitude = np.max(y)
        return [amplitude, mean, stddev]

    def gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))




    for i in range (int(len(peaks)/2), int(len(peaks)/2)+1):    #  Determinaion of FWHM in Cross-Dispersion Direction
        peaks1 = peaks[i]

        for j in range (int(len(science_frame_41)/2), int(len(science_frame_41)/2)-1, -1):
            try:
                cd_pixel = []
                cd_flux = []
                for k in range (-int(3*sigma_FWHM), int(3*sigma_FWHM)+1):
                    cd_flux.append(science_frame_41[peaks1+k][j])
                    cd_pixel.append((peaks1+k))
            #y.append(sum2)
                cd_pixel = np.array(cd_pixel)
                cd_flux = np.array(cd_flux)
            #d = np.linspace(peaks1, peaks1 + 1, 1)
            #d[0] = int(d[0])
            #geneticParameters = generate_Initial_Parameters(cd, d)
            # curve fit the test data
            #popt, pcov = curve_fit(func, cd, d, geneticParameters)
            #popt = np.abs(popt)
            #order_peaks.append(int(popt[0]))
            #peaks1 = int(popt[0])
                init_param = guess_gaussian_params(cd_pixel, cd_flux)
                popt, pcov = curve_fit(gaussian, cd_pixel, cd_flux, init_param)
                
            except:
                break
            
            #print(popt[1])
            sigma_FWHM = popt[2]
            
    return sigma_FWHM




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









science_frame_path = science_frame_path_2   # Add correct path
sky_frame_path = sky_22pt5_path

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





sigma_FWHM = Star_FWHM_Determination(science_frame_42, science_frame_starting_order, science_frame_NumberOfPeaks)



orders_41, peaks_41, xcor_41, ycor_41 = OrderTrace(science_frame_41, science_frame_starting_order, science_frame_NumberOfPeaks)   #  Load from text file
orders_42, peaks_42, xcor_42, ycor_42 = OrderTrace(science_frame_42, science_frame_starting_order, science_frame_NumberOfPeaks)     
orders_43, peaks_43, xcor_43, ycor_43 = OrderTrace(science_frame_43, science_frame_starting_order, science_frame_NumberOfPeaks)
orders_44, peaks_44, xcor_44, ycor_44 = OrderTrace(science_frame_44, science_frame_starting_order, science_frame_NumberOfPeaks)



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
    
    

path = 'C:\\Users\\Mudit Shrivastav\\.ipython\\Science_spectra\\' + observation_session + "\\" + star_name

if not os.path.exists(path):
    os.mkdir(path)

if grating_choice == 1:
    path = path + '\\RedCD\\'
elif grating_choice == 2:
    path = path + '\\BlueCD\\'

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
    np.savetxt(path + '\\' + star_name + '_' + set_number + '_Order-' + str(order_number) +'_IntensityBeforeEffCorr.txt', a, '%.3f', delimiter = '    ')


starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median = load_parameters(param_file)
