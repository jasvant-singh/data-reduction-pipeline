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
# ~ import lacosmic
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



script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)


master_bias = fits.getdata(f'{project_root}/output/masterbias_file.fits')


def load_parameters(param_file):
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and full-line comments
            if not line or line.startswith('#'):
                continue
            
            # Split only once on '=', rest of the line could be a comment
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()

                # Remove inline comments (e.g., 1  # comment)
                if '#' in val:
                    val = val.split('#')[0].strip()

                # Convert value to bool, int, float or str
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        # Remove quotes if present
                        val = val.strip('"').strip("'")

                params[key] = val

    return params


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





def wavelength_calibration1 (calib_UAr_data_cube, orders, I_BeforeEffCorr_list, xcor, ycor, sig_FWHM, grating_choice): 
    
    xcor_44 = xcor
    ycor_44 = ycor
    sigma_FWHM = sig_FWHM
    orders_41 = orders
    
    I_0_o = []
    I_0_e = []
    I_22pt5_o = []
    I_22pt5_e = []
    I_45_o = []
    I_45_e = []
    I_67pt5_o = []
    I_67pt5_e = []
    
    I_0_o_err = []
    I_0_e_err = []
    I_22pt5_o_err = []
    I_22pt5_e_err = []
    I_45_o_err = []
    I_45_e_err = []
    I_67pt5_o_err = []
    I_67pt5_e_err = []

    
    for i in range (len(I_BeforeEffCorr_list)):
        I_0_o.append(I_BeforeEffCorr_list[i][0])
        I_0_e.append(I_BeforeEffCorr_list[i][1])
        I_22pt5_o.append(I_BeforeEffCorr_list[i][2])
        I_22pt5_e.append(I_BeforeEffCorr_list[i][3])
        I_45_o.append(I_BeforeEffCorr_list[i][4])
        I_45_e.append(I_BeforeEffCorr_list[i][5])
        I_67pt5_o.append(I_BeforeEffCorr_list[i][6])
        I_67pt5_e.append(I_BeforeEffCorr_list[i][7])
        
        I_0_o_err.append(I_BeforeEffCorr_list[i][8])
        I_0_e_err.append(I_BeforeEffCorr_list[i][9])
        I_22pt5_o_err.append(I_BeforeEffCorr_list[i][10])
        I_22pt5_e_err.append(I_BeforeEffCorr_list[i][11])
        I_45_o_err.append(I_BeforeEffCorr_list[i][12])
        I_45_e_err.append(I_BeforeEffCorr_list[i][13])
        I_67pt5_o_err.append(I_BeforeEffCorr_list[i][14])
        I_67pt5_e_err.append(I_BeforeEffCorr_list[i][15])
        
        
    
    
    
    ###############################################    Wavelength Calibration ######################################################
    
    calib_UAr_data_cube_1 = calib_UAr_data_cube

    calib_UAr_frame_1 = calib_UAr_data_cube_1[0][:][:]

    calib_UAr_frame_1 = calib_UAr_frame_1 - master_bias

    calib_UAr_frame_1 = np.rot90(calib_UAr_frame_1)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_1 < 0)
    calib_UAr_frame_1[bad_pixels] = 0

    calib_UAr_frame_1_flux_ADU, sky_frame_1_flux_ADU = OrderExtraction(calib_UAr_frame_1, sky_67pt5, xcor_44, ycor_44, sigma_FWHM, 1)


    calib_UAr_frame_1_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_1_flux_order = []
        
        for j in range (len(calib_UAr_frame_1_flux_ADU[0])):
            calib_UAr_frame_1_flux_order.append(CCD_gain * (calib_UAr_frame_1_flux_ADU[i][j]))
        
        calib_UAr_frame_1_flux.append((calib_UAr_frame_1_flux_order/max(calib_UAr_frame_1_flux_order)))



    calib_UAr_I_67pt5_o_beforeWPshiftCorrection = []
    calib_UAr_I_67pt5_e_beforeWPshiftCorrection = []



    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_67pt5_o_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])
        else:
            calib_UAr_I_67pt5_e_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])


    #path_calib = path + "\\UAr_Calib_Flux\\"

    #if not os.path.exists(path_calib):
    #    os.mkdir(path_calib)
    

    #for i in range (int(len(orders_41)/2)):
        
    #    order_number = orders_41[(2*i)]
        
    #    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    #    x_new = x_new.astype('int32')        
            
    #    dt = np.dtype([('pixel', 'd'), ('I_o', 'd'), ('I_e', 'd')])  
    #    a = np.zeros(detector_pixels, dt)                        # Saving wavelength and the corresponding
    #    a['pixel'] = x_new
    #    a['I_o'] = calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i]
    #    a['I_e'] = calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i]    

        #np.savetxt('C:\\Users\\Mudit Shrivastav\\.ipython\\Science_spectra\\BetUMa\\BetUMa_9_IntensityTestBeforeEffCorr.txt', a, '%.5f', delimiter = ',')
    #    np.savetxt(path_calib + 'Order-' + str(order_number) +'_Intensity.txt', a, '%.3f', delimiter = '    ')



    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_67pt5_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-67.5 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
    calib_pixels = []
    calib_I_67pt5_o = []
    calib_I_67pt5_e = []

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_67pt5_o_order = []
        calib_I_67pt5_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
                
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
        
        calib_pixels.append(calib_pixels_order)
        calib_I_67pt5_o.append(calib_I_67pt5_o_order)
        calib_I_67pt5_e.append(calib_I_67pt5_e_order)


    # Get the list of all files and directories
    if grating_choice == 1:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/RedCD/UAr/"######################################################################################
    elif grating_choice == 2:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/BlueCD/UAr/"#####################################################################################
      
    dir_list = os.listdir(path1)
    #print("Files and directories in '", path1, "' :")
    # prints all files
    #print(dir_list)
    index = []
    for i in range (len(dir_list)):
        if dir_list[i].find('o') == -1:
            index.append(i)

    dir_list1 = []
    for i in range (len(index)):
        dir_list1.append(dir_list[index[i]])
      
    if grating_choice == 1:
        orders_ref = np.linspace(28, 48, 21)
    elif grating_choice == 2:
        orders_ref = np.linspace(44, 65, 22)
        
    orders_ref = orders_ref.astype('int32')

    calib_UAr_I_ref = []
    for i in range (len(dir_list1)):
        pix, calib_f = np.loadtxt(path1+dir_list1[i], usecols=(0, 1), delimiter = ',', unpack = True)
        calib_UAr_I_ref.append((calib_f/max(calib_f)))


    # Get the list of all files and directories
    if grating_choice == 1:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/RedCD/"
    elif grating_choice == 2:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/BlueCD/"
    
    dir_list2 = os.listdir(path2)
    dir_list2.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))########################## sorting required
    # ~ print(dir_list2)


    calib_UAr_reference_pix = []
    calib_UAr_reference_wave = []
    for i in range (len(dir_list2)):
        pix, wave = np.loadtxt(path2+dir_list2[i], usecols=(0, 1), unpack = True)
        calib_UAr_reference_pix.append(pix)
        calib_UAr_reference_wave.append(wave)
    # ~ print("calibrefe",(calib_UAr_reference_wave[0]))

    orders_41_wc = list(set(orders_41))
    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc.sort()
    orders_41_wc = np.flip(orders_41_wc)

            
    
    pixel_shift_wc_list = []

    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_I_67pt5_o[i],calib_UAr_I_ref[ref_order_match])
            
            print(pixel_shift_wc)
            pixel_shift_wc_list.append(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            #plt.plot(x_new, calib_I_67pt5_o[i])
            #plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
    
            
    def guess_gaussian_params(x, y):
        
        #Guess the initial parameters for a Gaussian fit to the data (x, y).
        
        mean = np.mean(x)
        stddev = np.std(x)
        amplitude = np.max(y)
        return [amplitude, mean, stddev]

    def gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    
    UAr_lamp_sigma_dispersion = 1
    
    residual_list = []
    
    if grating_choice == 1:
        gaussian_wavelength_for_shifting = [5527.980, 5650.704, 5772.118, 5888.580, 6059.373, 6155.237, 6307.657, 6416.307, 6578.794, 6766.612, 6937.664, 7147.041, 7372.118, 7590.524, 7814.326, 8046.115, 8273.505, 8521.441, 8799.086, 9093.654, 9354.219]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 28:
                diff_low_order = np.min(orders_41_wc) - 28
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 48:
                diff_low_order = 48 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting)  
          
    elif grating_choice == 2:
        gaussian_wavelength_for_shifting = [4052.921, 4131.735, 4200.674, 4277.551, 4348.081, 4379.689, 4481.835, 4579.367, 4589.920, 4726.899, 4806.042, 4879.873, 4965.099, 5063.756, 5187.746, 5315.282, 5410.472, 5451.655, 5572.543, 5739.520, 5882.624, 6032.128]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 44:
                diff_low_order = np.min(orders_41_wc) - 44
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 65:                
                diff_low_order = 65 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting) 
    
    print("orders_41_wc: ",orders_41_wc)
    print("gaussian_wavelength_for_shifting",gaussian_wavelength_for_shifting)
    print(len(orders_41_wc))
    print(len(gaussian_wavelength_for_shifting))
    
    gaussian_residual_add = []
    calib_UAr_reference_pix_gaussian_dispersion_corrected = []
    calib_UAr_reference_wave_gaussian_dispersion_corrected = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_I_67pt5_o[i],calib_UAr_I_ref[ref_order_match])
            
            if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
                pixel_shift_wc = mode(pixel_shift_wc_list)
                
            #pixel_shift_wc = -22
            #print(pixel_shift_wc)
            #pixel_shift_wc = dispersion_pixel_manual_shift

            #plt.plot(x_new, calib_I_67pt5_o[i])
            #plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #plt.title(str(orders_41_wc[i]))
            #plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + pixel_shift_wc
                
            
            calib_UAr_reference_pix_gaussian_dispersion_corrected.append(calib_UAr_reference_pix[ref_order_match])
            calib_UAr_reference_wave_gaussian_dispersion_corrected.append(calib_UAr_reference_wave[ref_order_match])
            
           
            residual = []   
            for j in range (len(calib_I_67pt5_o[i])):
                 
                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            
                            diff = (j+(4*UAr_lamp_sigma_dispersion)) - (j-(4*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j-(2*UAr_lamp_sigma_dispersion)),(j+(6*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_I_67pt5_o[i][(j-(4*UAr_lamp_sigma_dispersion)):(j+(4*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            residual.append(gaussian_peak - calib_UAr_reference_pix[ref_order_match][k])
                            

                    except:
                        continue
            
            residual_list.append(residual)
            
               
            #residual = np.array(residual)
            #print(residual)
            #median_residual = np.median(residual)
            median_residual = min(residual)
            print('median_residual: ',median_residual)
            pix = []
            wave = []    
            for j in range (len(calib_I_67pt5_o[i])):

                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            j_res = int(round(j + median_residual)) - 1
                            
                            diff = (j_res+(2*UAr_lamp_sigma_dispersion)) - (j_res-(2*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j_res-(2*UAr_lamp_sigma_dispersion)),(j_res+(2*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_I_67pt5_o[i][(j_res-(2*UAr_lamp_sigma_dispersion)):(j_res+(2*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            fit_x = np.linspace(min(gaussian_extract_pixels), max(gaussian_extract_pixels), 100)
                            fit = gaussian(fit_x, popt[0], popt[1], popt[2])
                            
                            #plt.plot(gaussian_extract_pixels, gaussian_extract_flux, color = 'b')
                            #plt.plot(fit_x, fit, color = 'r')
                            #plt.title(str(orders_41_wc[i]) + "  " + str(calib_UAr_reference_pix[ref_order_match][k]) + "  " + str(calib_UAr_reference_wave[ref_order_match][k]))
                            #lt.show()
                                
                            # ~ print("loop working this len",len(gaussian_wavelength_for_shifting), len(orders_41_wc))
                            if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                                # ~ print(calib_UAr_reference_wave[ref_order_match][k], gaussian_wavelength_for_shifting[i], 'ref order', ref_order_match)
                                if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                    gaussian_residual_add.append((gaussian_peak - calib_UAr_reference_pix[ref_order_match][k]))
                                    
                                    
                            #else:
                            #    gaussian_residual_add.append(0)
                            
                            pix.append(gaussian_peak)
                            wave.append(calib_UAr_reference_wave[ref_order_match][k])
                            
                    except:
                        pix.append(calib_UAr_reference_pix[ref_order_match][k] + int(round(median_residual)))
                        wave.append(calib_UAr_reference_wave[ref_order_match][k])
                        if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                            if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                if not gaussian_residual_add:
                                    gaussian_residual_add.append(0)
                                else:
                                    gaussian_residual_add.append(statistics.median(gaussian_residual_add))
                                
                        #gaussian_residual_add.append(0)
            
            pix = np.array(pix)
            wave = np.array(wave)
            #calib_UAr_reference_pix_negative_pixels_removed.append(pix)
            #calib_UAr_reference_wave_negative_pixels_removed.append(wave)
            
    for i in range (len(gaussian_residual_add)):
        # ~ print('loop worked')
        
        # ~ print(gaussian_residual_add[i])
        if gaussian_residual_add[i] > 5 or gaussian_residual_add[i] < -5:
            gaussian_residual_add[i] = statistics.median(gaussian_residual_add)
            # ~ print('gaussian' ,gaussian_residual_add[i])

    print("\n Gaussian Residual Add: \n")
    # ~ print(len(gaussian_residual_add))
    
    calib_UAr_reference_pix_gaussian_dispersion_corrected = []
    calib_UAr_reference_wave_gaussian_dispersion_corrected = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            #pixel_shift_wc = find_pixel_shift(calib_I_67pt5_o[i],calib_UAr_I_ref[ref_order_match])
            
            #if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
            #    pixel_shift_wc = mode(pixel_shift_wc_list)
            
            #print(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            #    plt.plot(x_new, calib_I_67pt5_o[i])
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #    plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + gaussian_residual_add[i]
            
            calib_UAr_reference_pix_gaussian_dispersion_corrected.append(calib_UAr_reference_pix[ref_order_match])
            calib_UAr_reference_wave_gaussian_dispersion_corrected.append(calib_UAr_reference_wave[ref_order_match])
    
    
    
    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected[i])):
            #if calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] >= 0 and calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] <= (detector_pixels-1):
            pix.append(calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j])
            wave.append(calib_UAr_reference_wave_gaussian_dispersion_corrected[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave) 
                       
    

    #for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
    #    print(list(calib_UAr_reference_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_wave_negative_pixels_removed[i]))
    #    print("\n")

    #orders_41_rev = np.flip(orders_41)

    """

    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix[i])):
            if calib_UAr_reference_pix[i][j] >= 0 and calib_UAr_reference_pix[i][j] <= (detector_pixels-1):
                pix.append(calib_UAr_reference_pix[i][j])
                wave.append(calib_UAr_reference_wave[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave)
    
    """


    #for i in range (int(len(orders_41)/2)):
    #    order_number = orders_41_rev[(2*i)]
    #    print(order_number)
    #    print("e-ray")
    #    print(list(calib_UAr_reference_e_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_e_wave_negative_pixels_removed[i]))
        #print("o-ray")
        #print(list(calib_UAr_reference_o_pix_negative_pixels_removed[i]))
        #print(list(calib_UAr_reference_o_wave_negative_pixels_removed[i]))
        
    def arraySortedOrNot(arr, n):
    # Array has one or no element
        if (n == 0 or n == 1):
            return True
        for i in range(1, n):
        # Unsorted pair found
            if (arr[i-1] < arr[i]):
                return False
    # No unsorted pair found
        return True
       
    wavelength = []
    for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
        pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'cubic', bounds_error = False, fill_value="extrapolate")
       
        pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
        pixel_row = pixel_row.astype('int32')
        wave = pix_wave_e(pixel_row)
        if arraySortedOrNot(wave, len(wave)) == True:
            print(str(orders_41_wc[i]) + " Cubic")
            
        if arraySortedOrNot(wave, len(wave)) == False:
            pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'quadratic', bounds_error = False, fill_value="extrapolate")            
            pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
            pixel_row = pixel_row.astype('int32')
            wave = pix_wave_e(pixel_row)
            if arraySortedOrNot(wave, len(wave)) == True:
                print(str(orders_41_wc[i]) + " Quadratic")
                
            if arraySortedOrNot(wave, len(wave)) == False:
                pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'linear', bounds_error = False, fill_value="extrapolate")
                print(str(orders_41_wc[i]) + " Linear")
                pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
                pixel_row = pixel_row.astype('int32')
                wave = pix_wave_e(pixel_row)
                
        wavelength.append(wave)
        
    wavelength = wavelength[::-1]
    
    #for i in range (len(wavelength_e)):
    #    print(str(orders_41_wc[i]) + "  " + str(min(wavelength_e[i])) + "  " + str(max(wavelength_e[i])))


    #wavelength = []
    #for i in range (len(wavelength_e)):
    #    wavelength.append(wavelength_e[i])
    #    wavelength.append(wavelength_o[i])

    #path_wavecal = path + "\\Wavelength_Calibrated\\"

    #if not os.path.exists(path_wavecal):
    #    os.mkdir(path_wavecal)

    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc = np.flip(orders_41_wc)

    wavelength_calibrated_intensities = []
    
    for i in range (len(orders_41_wc)):
        
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_intensities_order = []
            #wavelength_calibrated_intensities_order.append(wavelength[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(wavelength[i]) 
            wavelength_calibrated_intensities_order.append(I_0_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_intensities.append(wavelength_calibrated_intensities_order)
        
            




    I0_o_I0_e = []
    I45_o_I45_e = []
    I22pt5_o_I22pt5_e = []
    I67pt5_o_I67pt5_e = []

    Aq = []
    Au = []

    #Aq_interpolated = []
    #Au_interpolated = []

    q = []
    u = []
    p = []
    theta = []

    q_err = []
    u_err = []
    p_err = []
    theta_err = []



    #Aq_raw = []
    #Au_raw = []

    #q_raw = []
    #u_raw = []
    #p_raw = []
    #theta_raw = []

    #q_raw_err = []
    #u_raw_err = []
    #p_raw_err = []
    #theta_raw_err = []



    #def fit_func (x, a, b, c, d):
    #    return a*(x**3) + b*(x**2) +c*x + d

    for i in range (int(len(orders_41)/2)):
        
        I0_o_I0_e_order = []
        I45_o_I45_e_order = []
        I22pt5_o_I22pt5_e_order = []
        I67pt5_o_I67pt5_e_order = []
        
        for j in range (detector_pixels):
            I0_o_I0_e_order.append(I_0_o[i][j]/I_0_e[i][j])
            I45_o_I45_e_order.append(I_45_o[i][j]/I_45_e[i][j])
            I22pt5_o_I22pt5_e_order.append(I_22pt5_o[i][j]/I_22pt5_e[i][j])
            I67pt5_o_I67pt5_e_order.append(I_67pt5_o[i][j]/I_67pt5_e[i][j])
        
        
        temp_I0_o_I0_e_order = I0_o_I0_e_order
        temp_I45_o_I45_e_order = I45_o_I45_e_order
        temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
            I0_o_I0_e_median = []
            I45_o_I45_e_median = []
            I22pt5_o_I22pt5_e_median = []
            I67pt5_o_I67pt5_e_median = []
            
            for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)+1):
                I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
                I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
                I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
                I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
            I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
            I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
            I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
            I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        I0_o_I0_e.append(I0_o_I0_e_order)
        I45_o_I45_e.append(I45_o_I45_e_order)
        I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        Aq_order = []
        Au_order = []
        for j in range (detector_pixels):
            Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
            Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        Aq_order = np.array(Aq_order)
        Au_order = np.array(Au_order)
        
        Aq.append(Aq_order)
        Au.append(Au_order)

        q_order = []
        u_order = []
        q_err_order = []
        u_err_order = []

        for j in range (detector_pixels):
            q_order.append((Aq_order[j] - 1)/(Aq_order[j] + 1))
            u_order.append((Au_order[j] - 1)/(Au_order[j] + 1))
            q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
            u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        q_order = np.array(q_order)
        u_order = np.array(u_order)
        q_err_order = np.array(q_err_order)
        u_err_order = np.array(u_err_order)


        p_order = []
        theta_order = []
        for j in range (detector_pixels):
            p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
            theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        p_order = np.array(p_order)
        theta_order = np.array(theta_order)

        p_err_order = []
        theta_err_order = []
        for j in range (detector_pixels):
            p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
            theta_err_order.append(p_err_order[j]/(2*p_order[j]) * (180/np.pi))
            
        p_err_order = np.array(p_err_order)
        theta_err_order = np.array(theta_err_order)
        
        
        q.append(q_order)
        u.append(u_order)
        p.append(p_order)
        theta.append(theta_order)
        
        q_err.append(q_err_order)
        u_err.append(u_err_order)
        p_err.append(p_err_order)
        theta_err.append(theta_err_order)
        
        

        
        
        #temp_I0_o_I0_e_order = I0_o_I0_e_order
        #temp_I45_o_I45_e_order = I45_o_I45_e_order
        #temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        #temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        #for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
        #    I0_o_I0_e_median = []
        #    I45_o_I45_e_median = []
        #    I22pt5_o_I22pt5_e_median = []
        #    I67pt5_o_I67pt5_e_median = []
            
        #    for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)):
        #        I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
        #        I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
        #        I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
        #        I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
        #    I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
        #    I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
        #    I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
        #    I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        #I0_o_I0_e.append(I0_o_I0_e_order)
        #I45_o_I45_e.append(I45_o_I45_e_order)
        #I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        #I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        #Aq_order = []
        #Au_order = []
        #for j in range (detector_pixels):
        #    Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
        #    Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        #Aq_order = np.array(Aq_order)
        #Au_order = np.array(Au_order)
        
        #Aq.append(Aq_order)
        #Au.append(Au_order)
        
        
        #pixels_interp_points = []
        #Aq_order_interp_points = []
        #Au_order_interp_points = []
            
        #for j in range (0, detector_pixels, 1):
        #    pixels_interp_points.append(pixels[i][j])
        #    Aq_order_interp_points.append(Aq_order[j])
        #    Au_order_interp_points.append(Au_order[j])
        
        
        #Aq_polynomial = interp1d(pixels_interp_points, Aq_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #Au_polynomial = interp1d(pixels_interp_points, Au_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #Aq_polynomial_coeff = np.polyfit(pixels_interp_points, Aq_order_interp_points, 15) 
        #Au_polynomial_coeff = np.polyfit(pixels_interp_points, Au_order_interp_points, 15)
        #Aq_polynomial = np.poly1d(Aq_polynomial_coeff)
        #Au_polynomial = np.poly1d(Au_polynomial_coeff)
        
        #Aq_interp = Aq_polynomial(pixels[i])
        #Au_interp = Au_polynomial(pixels[i])
        
        #I0_o_I0_e_polynomial = interp1d(pixels[i], I0_o_I0_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I45_o_I45_e_polynomial = interp1d(pixels[i], I45_o_I45_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I22pt5_o_I22pt5_e_polynomial = interp1d(pixels[i], I22pt5_o_I22pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I67pt5_o_I67pt5_e_polynomial = interp1d(pixels[i], I67pt5_o_I67pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #popt_q, pcov_q = curve_fit(fit_func, pixels_interp_points, Aq_order_interp_points)
        #popt_u, pcov_u = curve_fit(fit_func, pixels_interp_points, Au_order_interp_points)
        #Aq_interp = []
        #Au_interp = []
        #aq, bq, cq, dq = popt_q
        #au, bu, cu, du = popt_u
        
        #for j in range (len(pixels[i])):        
        #    Aq_interp.append(fit_func(pixels[i][j], aq, bq, cq, dq))
        #    Au_interp.append(fit_func(pixels[i][j], au, bu, cu, du))
        
        #Aq_interp = np.array(Aq_interp)
        #Au_interp = np.array(Au_interp)
        
        #Aq_interpolated.append(Aq_interp)
        #Au_interpolated.append(Au_interp)


        #q_order = []
        #u_order = []
        #q_err_order = []
        #u_err_order = []

        #for j in range (detector_pixels):
        #    q_order.append((Aq_interp[j] - 1)/(Aq_interp[j] + 1))
        #    u_order.append((Au_interp[j] - 1)/(Au_interp[j] + 1))
        #    q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
        #    u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        #q_order = np.array(q_order)
        #u_order = np.array(u_order)
        #q_err_order = np.array(q_err_order)
        #u_err_order = np.array(u_err_order)


        #p_order = []
        #theta_order = []
        #for j in range (detector_pixels):
        #    p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
        #    theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        #p_order = np.array(p_order)
        #theta_order = np.array(theta_order)

        #p_err_order = []
        #theta_err_order = []
        #for j in range (detector_pixels):
        #    p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
        #    theta_err_order.append(p_err_order[j]/(2*p_order[j]))
            
        #p_err_order = np.array(p_err_order)
        #theta_err_order = np.array(theta_err_order)
        
        
        #q.append(q_order)
        #u.append(u_order)
        #p.append(p_order)
        #theta.append(theta_order)
        
        #q_err.append(q_err_order)
        #u_err.append(u_err_order)
        #p_err.append(p_err_order)
        #theta_err.append(theta_err_order)
        
        
    wavelength_calibrated_Stokes_parameters = []

    for i in range (len(orders_41_wc)):
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_Stokes_parameters_order = []
            
            wavelength_calibrated_Stokes_parameters_order.append(wavelength[i])
            wavelength_calibrated_Stokes_parameters_order.append(I0_o_I0_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I45_o_I45_e[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(I22pt5_o_I22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I67pt5_o_I67pt5_e[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters_order.append(Aq[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(Au[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_interpolated[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_interpolated[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u_err[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta_err[len(I_0_e)-1-i])
            
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw_err[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters.append(wavelength_calibrated_Stokes_parameters_order)
        
        
    print("calibration 1 finished")
    return (wavelength_calibrated_intensities, wavelength_calibrated_Stokes_parameters)



def wavelength_calibration2 (calib_UAr_data_cube1, calib_UAr_data_cube2, orders, I_BeforeEffCorr_list, xcor1, ycor1, xcor2, ycor2, sig_FWHM, grating_choice): 
    
    xcor_42 = xcor1
    ycor_42 = ycor1
    xcor_44 = xcor2
    ycor_44 = ycor2
    sigma_FWHM = sig_FWHM
    orders_41 = orders
    
    I_0_o = []
    I_0_e = []
    I_22pt5_o = []
    I_22pt5_e = []
    I_45_o = []
    I_45_e = []
    I_67pt5_o = []
    I_67pt5_e = []
    
    I_0_o_err = []
    I_0_e_err = []
    I_22pt5_o_err = []
    I_22pt5_e_err = []
    I_45_o_err = []
    I_45_e_err = []
    I_67pt5_o_err = []
    I_67pt5_e_err = []

    
    for i in range (len(I_BeforeEffCorr_list)):
        I_0_o.append(I_BeforeEffCorr_list[i][0])
        I_0_e.append(I_BeforeEffCorr_list[i][1])
        I_22pt5_o.append(I_BeforeEffCorr_list[i][2])
        I_22pt5_e.append(I_BeforeEffCorr_list[i][3])
        I_45_o.append(I_BeforeEffCorr_list[i][4])
        I_45_e.append(I_BeforeEffCorr_list[i][5])
        I_67pt5_o.append(I_BeforeEffCorr_list[i][6])
        I_67pt5_e.append(I_BeforeEffCorr_list[i][7])
        
        I_0_o_err.append(I_BeforeEffCorr_list[i][8])
        I_0_e_err.append(I_BeforeEffCorr_list[i][9])
        I_22pt5_o_err.append(I_BeforeEffCorr_list[i][10])
        I_22pt5_e_err.append(I_BeforeEffCorr_list[i][11])
        I_45_o_err.append(I_BeforeEffCorr_list[i][12])
        I_45_e_err.append(I_BeforeEffCorr_list[i][13])
        I_67pt5_o_err.append(I_BeforeEffCorr_list[i][14])
        I_67pt5_e_err.append(I_BeforeEffCorr_list[i][15])
    
    
    calib_UAr_data_cube_1 = calib_UAr_data_cube1
    calib_UAr_frame_1 = calib_UAr_data_cube_1[0][:][:]
    calib_UAr_frame_1 = calib_UAr_frame_1 - master_bias
    calib_UAr_frame_1 = np.rot90(calib_UAr_frame_1)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_1 < 0)
    calib_UAr_frame_1[bad_pixels] = 0

    calib_UAr_frame_1_flux_ADU, sky_frame_1_flux_ADU = OrderExtraction(calib_UAr_frame_1, sky_22pt5, xcor_42, ycor_42, sigma_FWHM, 1)


    calib_UAr_frame_1_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_1_flux_order = []
        
        for j in range (len(calib_UAr_frame_1_flux_ADU[0])):
            calib_UAr_frame_1_flux_order.append(CCD_gain * (calib_UAr_frame_1_flux_ADU[i][j]))
        
        calib_UAr_frame_1_flux.append((calib_UAr_frame_1_flux_order/max(calib_UAr_frame_1_flux_order)))

    
    calib_UAr_I_22pt5_o_beforeWPshiftCorrection = []
    calib_UAr_I_22pt5_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_22pt5_o_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])
        else:
            calib_UAr_I_22pt5_e_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_22pt5_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i], calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-22.5 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
    calib_pixels = []
    calib_I_22pt5_o = []
    calib_I_22pt5_e = []
    

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_22pt5_o_order = []
        calib_I_22pt5_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
             
        
        calib_pixels.append(calib_pixels_order)
        calib_I_22pt5_o.append(calib_I_22pt5_o_order)
        calib_I_22pt5_e.append(calib_I_22pt5_e_order)



    calib_UAr_data_cube_2 = calib_UAr_data_cube2
    calib_UAr_frame_2 = calib_UAr_data_cube_2[0][:][:]
    calib_UAr_frame_2 = calib_UAr_frame_2 - master_bias
    calib_UAr_frame_2 = np.rot90(calib_UAr_frame_2)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_2 < 0)
    calib_UAr_frame_2[bad_pixels] = 0

    calib_UAr_frame_2_flux_ADU, sky_frame_2_flux_ADU = OrderExtraction(calib_UAr_frame_2, sky_67pt5, xcor_44, ycor_44, sigma_FWHM, 1)


    calib_UAr_frame_2_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_2_flux_order = []
        
        for j in range (len(calib_UAr_frame_2_flux_ADU[0])):
            calib_UAr_frame_2_flux_order.append(CCD_gain * (calib_UAr_frame_2_flux_ADU[i][j]))
        
        calib_UAr_frame_2_flux.append((calib_UAr_frame_2_flux_order/max(calib_UAr_frame_2_flux_order)))

    
    calib_UAr_I_67pt5_o_beforeWPshiftCorrection = []
    calib_UAr_I_67pt5_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_67pt5_o_beforeWPshiftCorrection.append(calib_UAr_frame_2_flux[i])
        else:
            calib_UAr_I_67pt5_e_beforeWPshiftCorrection.append(calib_UAr_frame_2_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_67pt5_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-67.5 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift

    calib_pixels = []
    calib_I_67pt5_o = []
    calib_I_67pt5_e = []

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_67pt5_o_order = []
        calib_I_67pt5_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
                
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
        
        calib_pixels.append(calib_pixels_order)
        calib_I_67pt5_o.append(calib_I_67pt5_o_order)
        calib_I_67pt5_e.append(calib_I_67pt5_e_order)



    calib_pixels_beforeFrameShiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeFrameShiftCorrection .append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (int(len(orders_41)/2)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_I_22pt5_o[i], calib_I_67pt5_o[i]))
    calib_pixel_shift = mode(calib_pixel_shift_order)
    #calib_pixel_shift = 0
    print(calib_pixel_shift)
    
    print("Calib frame - HWP-22.5 and HWP-67.5 frame pixel shift: " + str(calib_pixel_shift))
    
    if manual_22pt5_67pt5_spectral_shift_condition == True:
        calib_pixel_shift = manual_22pt5_67pt5_spectral_shift
    
    calib_pixels = []
    calib_FrameShift_I_o_22pt5 = []
    calib_FrameShift_I_e_22pt5 = []
    calib_FrameShift_I_o_67pt5 = []
    calib_FrameShift_I_e_67pt5 = []
    
    
    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_FrameShift_I_o_22pt5_order = []
        calib_FrameShift_I_e_22pt5_order = []
        calib_FrameShift_I_o_67pt5_order = []
        calib_FrameShift_I_e_67pt5_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])          
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j]) 
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])
            
               
        calib_pixels.append(calib_pixels_order)
        calib_FrameShift_I_o_22pt5.append(calib_FrameShift_I_o_22pt5_order)
        calib_FrameShift_I_e_22pt5.append(calib_FrameShift_I_e_22pt5_order)
        calib_FrameShift_I_o_67pt5.append(calib_FrameShift_I_o_67pt5_order)
        calib_FrameShift_I_e_67pt5.append(calib_FrameShift_I_e_67pt5_order)
    


    # Get the list of all files and directories
    if grating_choice == 1:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/RedCD/UAr/"######################################################################################
    elif grating_choice == 2:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/BlueCD/UAr/"#####################################################################################
      
    dir_list = os.listdir(path1)
    #print("Files and directories in '", path1, "' :")
    # prints all files
    #print(dir_list)
    index = []
    for i in range (len(dir_list)):
        if dir_list[i].find('o') == -1:
            index.append(i)

    dir_list1 = []
    for i in range (len(index)):
        dir_list1.append(dir_list[index[i]])
      
    if grating_choice == 1:
        orders_ref = np.linspace(28, 48, 21)
    elif grating_choice == 2:
        orders_ref = np.linspace(44, 65, 22)
        
    orders_ref = orders_ref.astype('int32')

    calib_UAr_I_ref = []
    for i in range (len(dir_list1)):
        pix, calib_f = np.loadtxt(path1+dir_list1[i], usecols=(0, 1), delimiter = ',', unpack = True)
        calib_UAr_I_ref.append((calib_f/max(calib_f)))


    # Get the list of all files and directories
    if grating_choice == 1:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/RedCD/"
    elif grating_choice == 2:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/BlueCD/"
    
    dir_list2 = os.listdir(path2)
    # ~ dir_list2.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))


    calib_UAr_reference_pix = []
    calib_UAr_reference_wave = []
    for i in range (len(dir_list2)):
        pix, wave = np.loadtxt(path2+dir_list2[i], usecols=(0, 1), unpack = True)
        calib_UAr_reference_pix.append(pix)
        calib_UAr_reference_wave.append(wave)




    orders_41_wc = list(set(orders_41))
    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc.sort()
    orders_41_wc = np.flip(orders_41_wc)


    pixel_shift_wc_list = []

    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_FrameShift_I_e_67pt5[i],calib_UAr_I_ref[ref_order_match])
            
            print(pixel_shift_wc)
            pixel_shift_wc_list.append(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            plt.plot(x_new, calib_FrameShift_I_e_67pt5[i])
            plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
    
            
    def guess_gaussian_params(x, y):
        
        #Guess the initial parameters for a Gaussian fit to the data (x, y).
        
        mean = np.mean(x)
        stddev = np.std(x)
        amplitude = np.max(y)
        return [amplitude, mean, stddev]

    def gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    
    UAr_lamp_sigma_dispersion = 1
    
    residual_list = []
    
    if grating_choice == 1:
        gaussian_wavelength_for_shifting = [5527.980, 5650.704, 5772.118, 5888.580, 6059.373, 6155.237, 6307.657, 6416.307, 6578.794, 6766.612, 6937.664, 7147.041, 7372.118, 7590.524, 7814.326, 8046.115, 8273.505, 8521.441, 8799.086, 9093.654, 9354.219]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 28:
                diff_low_order = np.min(orders_41_wc) - 28
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 48:
                diff_low_order = 48 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting)  
          
    elif grating_choice == 2:
        gaussian_wavelength_for_shifting = [4052.921, 4131.735, 4200.674, 4277.551, 4348.081, 4379.689, 4481.835, 4579.367, 4589.920, 4726.899, 4806.042, 4879.873, 4965.099, 5063.756, 5187.746, 5315.282, 5410.472, 5451.655, 5572.543, 5739.520, 5882.624, 6032.128]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 44:
                diff_low_order = np.min(orders_41_wc) - 44
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 65:                
                diff_low_order = 65 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting) 
    
    print(orders_41_wc)
    print(gaussian_wavelength_for_shifting)
    print(len(orders_41_wc))
    print(len(gaussian_wavelength_for_shifting))
    
    gaussian_residual_add = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_FrameShift_I_e_67pt5[i],calib_UAr_I_ref[ref_order_match])
            
            if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
                pixel_shift_wc = mode(pixel_shift_wc_list)
            
            #print(pixel_shift_wc)
            #pixel_shift_wc = dispersion_pixel_manual_shift

            #plt.plot(x_new, calib_I_67pt5_o[i])
            #lt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #plt.title(str(orders_41_wc[i]))
            #plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + pixel_shift_wc
                
            
            
            residual = []   
            for j in range (len(calib_FrameShift_I_e_67pt5[i])):
                 
                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            
                            diff = (j+(4*UAr_lamp_sigma_dispersion)) - (j-(4*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j-(2*UAr_lamp_sigma_dispersion)),(j+(6*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_FrameShift_I_e_67pt5[i][(j-(4*UAr_lamp_sigma_dispersion)):(j+(4*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            residual.append(gaussian_peak - calib_UAr_reference_pix[ref_order_match][k])
                            

                    except:
                        continue
            
            residual_list.append(residual)
            
               
            #residual = np.array(residual)
            #print(residual)
            #median_residual = np.median(residual)
            median_residual = min(residual)
            print(median_residual)
            pix = []
            wave = []    
            for j in range (len(calib_FrameShift_I_e_67pt5[i])):

                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            j_res = int(round(j + median_residual)) - 1
                            
                            diff = (j_res+(2*UAr_lamp_sigma_dispersion)) - (j_res-(2*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j_res-(2*UAr_lamp_sigma_dispersion)),(j_res+(2*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_FrameShift_I_e_67pt5[i][(j_res-(2*UAr_lamp_sigma_dispersion)):(j_res+(2*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            fit_x = np.linspace(min(gaussian_extract_pixels), max(gaussian_extract_pixels), 100)
                            fit = gaussian(fit_x, popt[0], popt[1], popt[2])
                            #plt.plot(gaussian_extract_pixels, gaussian_extract_flux, color = 'b')
                            #plt.plot(fit_x, fit, color = 'r')
                            #plt.title(str(orders_41_wc[i]) + "  " + str(calib_UAr_reference_pix[ref_order_match][k]) + "  " + str(calib_UAr_reference_wave[ref_order_match][k]))
                            #lt.show()
                            
                            if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                                if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                    gaussian_residual_add.append((gaussian_peak - calib_UAr_reference_pix[ref_order_match][k]))
                                    
                            #else:
                            #    gaussian_residual_add.append(0)
                            
                            pix.append(gaussian_peak)
                            wave.append(calib_UAr_reference_wave[ref_order_match][k])
                            
                    except:
                        pix.append(calib_UAr_reference_pix[ref_order_match][k] + int(round(median_residual)))
                        wave.append(calib_UAr_reference_wave[ref_order_match][k])
                        if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                            if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                if not gaussian_residual_add:
                                    gaussian_residual_add.append(0)
                                else:
                                    gaussian_residual_add.append(statistics.median(gaussian_residual_add))
                                
                        #gaussian_residual_add.append(0)
            
            pix = np.array(pix)
            wave = np.array(wave)
            #calib_UAr_reference_pix_negative_pixels_removed.append(pix)
            #calib_UAr_reference_wave_negative_pixels_removed.append(wave)
            
    for i in range (len(gaussian_residual_add)):
        if gaussian_residual_add[i] > 5 or gaussian_residual_add[i] < -5:
            gaussian_residual_add[i] = statistics.median(gaussian_residual_add)

    print("\n Gaussian Residual Add: \n")
    print(gaussian_residual_add)
    
    calib_UAr_reference_pix_gaussian_dispersion_corrected = []
    calib_UAr_reference_wave_gaussian_dispersion_corrected = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            #pixel_shift_wc = find_pixel_shift(calib_I_67pt5_o[i],calib_UAr_I_ref[ref_order_match])
            
            #if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
            #    pixel_shift_wc = mode(pixel_shift_wc_list)
            
            #print(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            #    plt.plot(x_new, calib_I_67pt5_o[i])
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #    plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + gaussian_residual_add[i]
            
            calib_UAr_reference_pix_gaussian_dispersion_corrected.append(calib_UAr_reference_pix[ref_order_match])
            calib_UAr_reference_wave_gaussian_dispersion_corrected.append(calib_UAr_reference_wave[ref_order_match])
    
    
    
    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected[i])):
            #if calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] >= 0 and calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] <= (detector_pixels-1):
            pix.append(calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j])
            wave.append(calib_UAr_reference_wave_gaussian_dispersion_corrected[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave) 
                       
    

    #for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
    #    print(list(calib_UAr_reference_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_wave_negative_pixels_removed[i]))
    #    print("\n")

    #orders_41_rev = np.flip(orders_41)

    """

    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix[i])):
            if calib_UAr_reference_pix[i][j] >= 0 and calib_UAr_reference_pix[i][j] <= (detector_pixels-1):
                pix.append(calib_UAr_reference_pix[i][j])
                wave.append(calib_UAr_reference_wave[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave)
    
    """

        
        
        


    #for i in range (int(len(orders_41)/2)):
    #    order_number = orders_41_rev[(2*i)]
    #    print(order_number)
    #    print("e-ray")
    #    print(list(calib_UAr_reference_e_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_e_wave_negative_pixels_removed[i]))
        #print("o-ray")
        #print(list(calib_UAr_reference_o_pix_negative_pixels_removed[i]))
        #print(list(calib_UAr_reference_o_wave_negative_pixels_removed[i]))
       
    def arraySortedOrNot(arr, n):
    # Array has one or no element
        if (n == 0 or n == 1):
            return True
        for i in range(1, n):
        # Unsorted pair found
            if (arr[i-1] < arr[i]):
                return False
    # No unsorted pair found
        return True
       
    wavelength = []
    for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
        pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'cubic', bounds_error = False, fill_value="extrapolate")
       
        pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
        pixel_row = pixel_row.astype('int32')
        wave = pix_wave_e(pixel_row)
        if arraySortedOrNot(wave, len(wave)) == True:
            print(str(orders_41_wc[i]) + " Cubic")
            
        if arraySortedOrNot(wave, len(wave)) == False:
            pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'quadratic', bounds_error = False, fill_value="extrapolate")            
            pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
            pixel_row = pixel_row.astype('int32')
            wave = pix_wave_e(pixel_row)
            if arraySortedOrNot(wave, len(wave)) == True:
                print(str(orders_41_wc[i]) + " Quadratic")
                
            if arraySortedOrNot(wave, len(wave)) == False:
                pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'linear', bounds_error = False, fill_value="extrapolate")
                print(str(orders_41_wc[i]) + " Linear")
                pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
                pixel_row = pixel_row.astype('int32')
                wave = pix_wave_e(pixel_row)
                
        wavelength.append(wave)
        
    wavelength = wavelength[::-1]
    
    #for i in range (len(wavelength_e)):
    #    print(str(orders_41_wc[i]) + "  " + str(min(wavelength_e[i])) + "  " + str(max(wavelength_e[i])))


    #wavelength = []
    #for i in range (len(wavelength_e)):
    #    wavelength.append(wavelength_e[i])
    #    wavelength.append(wavelength_o[i])

    #path_wavecal = path + "\\Wavelength_Calibrated\\"

    #if not os.path.exists(path_wavecal):
    #    os.mkdir(path_wavecal)

    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc = np.flip(orders_41_wc)

    wavelength_calibrated_intensities = []
    
    for i in range (len(orders_41_wc)):
        
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_intensities_order = []
            #wavelength_calibrated_intensities_order.append(wavelength[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(wavelength[i]) 
            wavelength_calibrated_intensities_order.append(I_0_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_intensities.append(wavelength_calibrated_intensities_order)
        
            




    I0_o_I0_e = []
    I45_o_I45_e = []
    I22pt5_o_I22pt5_e = []
    I67pt5_o_I67pt5_e = []

    Aq = []
    Au = []

    #Aq_interpolated = []
    #Au_interpolated = []

    q = []
    u = []
    p = []
    theta = []

    q_err = []
    u_err = []
    p_err = []
    theta_err = []



    #Aq_raw = []
    #Au_raw = []

    #q_raw = []
    #u_raw = []
    #p_raw = []
    #theta_raw = []

    #q_raw_err = []
    #u_raw_err = []
    #p_raw_err = []
    #theta_raw_err = []



    #def fit_func (x, a, b, c, d):
    #    return a*(x**3) + b*(x**2) +c*x + d

    for i in range (int(len(orders_41)/2)):
        
        I0_o_I0_e_order = []
        I45_o_I45_e_order = []
        I22pt5_o_I22pt5_e_order = []
        I67pt5_o_I67pt5_e_order = []
        
        for j in range (detector_pixels):
            I0_o_I0_e_order.append(I_0_o[i][j]/I_0_e[i][j])
            I45_o_I45_e_order.append(I_45_o[i][j]/I_45_e[i][j])
            I22pt5_o_I22pt5_e_order.append(I_22pt5_o[i][j]/I_22pt5_e[i][j])
            I67pt5_o_I67pt5_e_order.append(I_67pt5_o[i][j]/I_67pt5_e[i][j])
        
        
        temp_I0_o_I0_e_order = I0_o_I0_e_order
        temp_I45_o_I45_e_order = I45_o_I45_e_order
        temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
            I0_o_I0_e_median = []
            I45_o_I45_e_median = []
            I22pt5_o_I22pt5_e_median = []
            I67pt5_o_I67pt5_e_median = []
            
            for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)+1):
                I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
                I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
                I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
                I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
            I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
            I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
            I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
            I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        I0_o_I0_e.append(I0_o_I0_e_order)
        I45_o_I45_e.append(I45_o_I45_e_order)
        I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        Aq_order = []
        Au_order = []
        for j in range (detector_pixels):
            Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
            Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        Aq_order = np.array(Aq_order)
        Au_order = np.array(Au_order)
        
        Aq.append(Aq_order)
        Au.append(Au_order)

        q_order = []
        u_order = []
        q_err_order = []
        u_err_order = []

        for j in range (detector_pixels):
            q_order.append((Aq_order[j] - 1)/(Aq_order[j] + 1))
            u_order.append((Au_order[j] - 1)/(Au_order[j] + 1))
            q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
            u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        q_order = np.array(q_order)
        u_order = np.array(u_order)
        q_err_order = np.array(q_err_order)
        u_err_order = np.array(u_err_order)


        p_order = []
        theta_order = []
        for j in range (detector_pixels):
            p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
            theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        p_order = np.array(p_order)
        theta_order = np.array(theta_order)

        p_err_order = []
        theta_err_order = []
        for j in range (detector_pixels):
            p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
            theta_err_order.append(p_err_order[j]/(2*p_order[j]) * (180/np.pi))
            
        p_err_order = np.array(p_err_order)
        theta_err_order = np.array(theta_err_order)
        
        
        q.append(q_order)
        u.append(u_order)
        p.append(p_order)
        theta.append(theta_order)
        
        q_err.append(q_err_order)
        u_err.append(u_err_order)
        p_err.append(p_err_order)
        theta_err.append(theta_err_order)
        
        

        
        
        #temp_I0_o_I0_e_order = I0_o_I0_e_order
        #temp_I45_o_I45_e_order = I45_o_I45_e_order
        #temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        #temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        #for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
        #    I0_o_I0_e_median = []
        #    I45_o_I45_e_median = []
        #    I22pt5_o_I22pt5_e_median = []
        #    I67pt5_o_I67pt5_e_median = []
            
        #    for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)):
        #        I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
        #        I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
        #        I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
        #        I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
        #    I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
        #    I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
        #    I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
        #    I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        #I0_o_I0_e.append(I0_o_I0_e_order)
        #I45_o_I45_e.append(I45_o_I45_e_order)
        #I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        #I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        #Aq_order = []
        #Au_order = []
        #for j in range (detector_pixels):
        #    Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
        #    Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        #Aq_order = np.array(Aq_order)
        #Au_order = np.array(Au_order)
        
        #Aq.append(Aq_order)
        #Au.append(Au_order)
        
        
        #pixels_interp_points = []
        #Aq_order_interp_points = []
        #Au_order_interp_points = []
            
        #for j in range (0, detector_pixels, 1):
        #    pixels_interp_points.append(pixels[i][j])
        #    Aq_order_interp_points.append(Aq_order[j])
        #    Au_order_interp_points.append(Au_order[j])
        
        
        #Aq_polynomial = interp1d(pixels_interp_points, Aq_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #Au_polynomial = interp1d(pixels_interp_points, Au_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #Aq_polynomial_coeff = np.polyfit(pixels_interp_points, Aq_order_interp_points, 15) 
        #Au_polynomial_coeff = np.polyfit(pixels_interp_points, Au_order_interp_points, 15)
        #Aq_polynomial = np.poly1d(Aq_polynomial_coeff)
        #Au_polynomial = np.poly1d(Au_polynomial_coeff)
        
        #Aq_interp = Aq_polynomial(pixels[i])
        #Au_interp = Au_polynomial(pixels[i])
        
        #I0_o_I0_e_polynomial = interp1d(pixels[i], I0_o_I0_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I45_o_I45_e_polynomial = interp1d(pixels[i], I45_o_I45_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I22pt5_o_I22pt5_e_polynomial = interp1d(pixels[i], I22pt5_o_I22pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I67pt5_o_I67pt5_e_polynomial = interp1d(pixels[i], I67pt5_o_I67pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #popt_q, pcov_q = curve_fit(fit_func, pixels_interp_points, Aq_order_interp_points)
        #popt_u, pcov_u = curve_fit(fit_func, pixels_interp_points, Au_order_interp_points)
        #Aq_interp = []
        #Au_interp = []
        #aq, bq, cq, dq = popt_q
        #au, bu, cu, du = popt_u
        
        #for j in range (len(pixels[i])):        
        #    Aq_interp.append(fit_func(pixels[i][j], aq, bq, cq, dq))
        #    Au_interp.append(fit_func(pixels[i][j], au, bu, cu, du))
        
        #Aq_interp = np.array(Aq_interp)
        #Au_interp = np.array(Au_interp)
        
        #Aq_interpolated.append(Aq_interp)
        #Au_interpolated.append(Au_interp)


        #q_order = []
        #u_order = []
        #q_err_order = []
        #u_err_order = []

        #for j in range (detector_pixels):
        #    q_order.append((Aq_interp[j] - 1)/(Aq_interp[j] + 1))
        #    u_order.append((Au_interp[j] - 1)/(Au_interp[j] + 1))
        #    q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
        #    u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        #q_order = np.array(q_order)
        #u_order = np.array(u_order)
        #q_err_order = np.array(q_err_order)
        #u_err_order = np.array(u_err_order)


        #p_order = []
        #theta_order = []
        #for j in range (detector_pixels):
        #    p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
        #    theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        #p_order = np.array(p_order)
        #theta_order = np.array(theta_order)

        #p_err_order = []
        #theta_err_order = []
        #for j in range (detector_pixels):
        #    p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
        #    theta_err_order.append(p_err_order[j]/(2*p_order[j]))
            
        #p_err_order = np.array(p_err_order)
        #theta_err_order = np.array(theta_err_order)
        
        
        #q.append(q_order)
        #u.append(u_order)
        #p.append(p_order)
        #theta.append(theta_order)
        
        #q_err.append(q_err_order)
        #u_err.append(u_err_order)
        #p_err.append(p_err_order)
        #theta_err.append(theta_err_order)
        
        
    wavelength_calibrated_Stokes_parameters = []

    for i in range (len(orders_41_wc)):
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_Stokes_parameters_order = []
            
            wavelength_calibrated_Stokes_parameters_order.append(wavelength[i])
            wavelength_calibrated_Stokes_parameters_order.append(I0_o_I0_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I45_o_I45_e[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(I22pt5_o_I22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I67pt5_o_I67pt5_e[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters_order.append(Aq[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(Au[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_interpolated[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_interpolated[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u_err[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta_err[len(I_0_e)-1-i])
            
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw_err[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters.append(wavelength_calibrated_Stokes_parameters_order)
        
    print("calibration 2 finished")
    return (wavelength_calibrated_intensities, wavelength_calibrated_Stokes_parameters)



def wavelength_calibration4 (calib_UAr_data_cube1, calib_UAr_data_cube2, calib_UAr_data_cube3, calib_UAr_data_cube4, orders, I_BeforeEffCorr_list, xcor1, ycor1, xcor2, ycor2, xcor3, ycor3, xcor4, ycor4, sig_FWHM, grating_choice): 
    
    xcor_41 = xcor1
    ycor_41 = ycor1
    xcor_42 = xcor2
    ycor_42 = ycor2
    xcor_43 = xcor3
    ycor_43 = ycor3
    xcor_44 = xcor4
    ycor_44 = ycor4
    sigma_FWHM = sig_FWHM
    orders_41 = orders 
    
    I_0_o = []
    I_0_e = []
    I_22pt5_o = []
    I_22pt5_e = []
    I_45_o = []
    I_45_e = []
    I_67pt5_o = []
    I_67pt5_e = []
    
    I_0_o_err = []
    I_0_e_err = []
    I_22pt5_o_err = []
    I_22pt5_e_err = []
    I_45_o_err = []
    I_45_e_err = []
    I_67pt5_o_err = []
    I_67pt5_e_err = []

    
    for i in range (len(I_BeforeEffCorr_list)):
        I_0_o.append(I_BeforeEffCorr_list[i][0])
        I_0_e.append(I_BeforeEffCorr_list[i][1])
        I_22pt5_o.append(I_BeforeEffCorr_list[i][2])
        I_22pt5_e.append(I_BeforeEffCorr_list[i][3])
        I_45_o.append(I_BeforeEffCorr_list[i][4])
        I_45_e.append(I_BeforeEffCorr_list[i][5])
        I_67pt5_o.append(I_BeforeEffCorr_list[i][6])
        I_67pt5_e.append(I_BeforeEffCorr_list[i][7])
        
        I_0_o_err.append(I_BeforeEffCorr_list[i][8])
        I_0_e_err.append(I_BeforeEffCorr_list[i][9])
        I_22pt5_o_err.append(I_BeforeEffCorr_list[i][10])
        I_22pt5_e_err.append(I_BeforeEffCorr_list[i][11])
        I_45_o_err.append(I_BeforeEffCorr_list[i][12])
        I_45_e_err.append(I_BeforeEffCorr_list[i][13])
        I_67pt5_o_err.append(I_BeforeEffCorr_list[i][14])
        I_67pt5_e_err.append(I_BeforeEffCorr_list[i][15])
    
    
    calib_UAr_data_cube_1 = calib_UAr_data_cube1
    calib_UAr_frame_1 = calib_UAr_data_cube_1[0][:][:]
    calib_UAr_frame_1 = calib_UAr_frame_1 - master_bias
    calib_UAr_frame_1 = np.rot90(calib_UAr_frame_1)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_1 < 0)
    calib_UAr_frame_1[bad_pixels] = 0

    calib_UAr_frame_1_flux_ADU, sky_frame_1_flux_ADU = OrderExtraction(calib_UAr_frame_1, sky_0, xcor_41, ycor_41, sigma_FWHM, 1)


    calib_UAr_frame_1_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_1_flux_order = []
        
        for j in range (len(calib_UAr_frame_1_flux_ADU[0])):
            calib_UAr_frame_1_flux_order.append(CCD_gain * (calib_UAr_frame_1_flux_ADU[i][j]))
        
        calib_UAr_frame_1_flux.append((calib_UAr_frame_1_flux_order/max(calib_UAr_frame_1_flux_order)))

    
    calib_UAr_I_0_o_beforeWPshiftCorrection = []
    calib_UAr_I_0_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_0_o_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])
        else:
            calib_UAr_I_0_e_beforeWPshiftCorrection.append(calib_UAr_frame_1_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_0_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_0_o_beforeWPshiftCorrection[i], calib_UAr_I_0_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-0 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
        
    calib_pixels = []
    calib_I_0_o = []
    calib_I_0_e = []

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_0_o_order = []
        calib_I_0_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_0_o_order.append(calib_UAr_I_0_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_0_o_order.append(calib_UAr_I_0_o_beforeWPshiftCorrection[i][j])
                calib_I_0_e_order.append(calib_UAr_I_0_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_0_e_order.append(calib_UAr_I_0_e_beforeWPshiftCorrection[i][j])
                
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_0_e_order.append(calib_UAr_I_0_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_0_o_order.append(calib_UAr_I_0_o_beforeWPshiftCorrection[i][j])
                calib_I_0_e_order.append(calib_UAr_I_0_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_0_o_order.append(calib_UAr_I_0_o_beforeWPshiftCorrection[i][j])
        
        
        
        calib_pixels.append(calib_pixels_order)
        calib_I_0_o.append(calib_I_0_o_order)
        calib_I_0_e.append(calib_I_0_e_order)
    
    
    
    calib_UAr_data_cube_2 = calib_UAr_data_cube2
    calib_UAr_frame_2 = calib_UAr_data_cube_2[0][:][:]
    calib_UAr_frame_2 = calib_UAr_frame_2 - master_bias
    calib_UAr_frame_2 = np.rot90(calib_UAr_frame_2)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_2 < 0)
    calib_UAr_frame_2[bad_pixels] = 0

    calib_UAr_frame_2_flux_ADU, sky_frame_2_flux_ADU = OrderExtraction(calib_UAr_frame_2, sky_22pt5, xcor_42, ycor_42, sigma_FWHM, 1)


    calib_UAr_frame_2_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_2_flux_order = []
        
        for j in range (len(calib_UAr_frame_2_flux_ADU[0])):
            calib_UAr_frame_2_flux_order.append(CCD_gain * (calib_UAr_frame_2_flux_ADU[i][j]))
        
        calib_UAr_frame_2_flux.append((calib_UAr_frame_2_flux_order/max(calib_UAr_frame_2_flux_order)))

    
    calib_UAr_I_22pt5_o_beforeWPshiftCorrection = []
    calib_UAr_I_22pt5_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_22pt5_o_beforeWPshiftCorrection.append(calib_UAr_frame_2_flux[i])
        else:
            calib_UAr_I_22pt5_e_beforeWPshiftCorrection.append(calib_UAr_frame_2_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_22pt5_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i], calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-22.5 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
    calib_pixels = []
    calib_I_22pt5_o = []
    calib_I_22pt5_e = []
    
    

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_22pt5_o_order = []
        calib_I_22pt5_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_22pt5_e_order.append(calib_UAr_I_22pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_22pt5_o_order.append(calib_UAr_I_22pt5_o_beforeWPshiftCorrection[i][j])
             
        
        calib_pixels.append(calib_pixels_order)
        calib_I_22pt5_o.append(calib_I_22pt5_o_order)
        calib_I_22pt5_e.append(calib_I_22pt5_e_order)

    
    
    calib_UAr_data_cube_3 = calib_UAr_data_cube3
    calib_UAr_frame_3 = calib_UAr_data_cube_3[0][:][:]
    calib_UAr_frame_3 = calib_UAr_frame_3 - master_bias
    calib_UAr_frame_3 = np.rot90(calib_UAr_frame_3)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_3 < 0)
    calib_UAr_frame_3[bad_pixels] = 0

    calib_UAr_frame_3_flux_ADU, sky_frame_3_flux_ADU = OrderExtraction(calib_UAr_frame_3, sky_45, xcor_43, ycor_43, sigma_FWHM, 1)


    calib_UAr_frame_3_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_3_flux_order = []
        
        for j in range (len(calib_UAr_frame_3_flux_ADU[0])):
            calib_UAr_frame_3_flux_order.append(CCD_gain * (calib_UAr_frame_3_flux_ADU[i][j]))
        
        calib_UAr_frame_3_flux.append((calib_UAr_frame_3_flux_order/max(calib_UAr_frame_3_flux_order)))

    
    calib_UAr_I_45_o_beforeWPshiftCorrection = []
    calib_UAr_I_45_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_45_o_beforeWPshiftCorrection.append(calib_UAr_frame_3_flux[i])
        else:
            calib_UAr_I_45_e_beforeWPshiftCorrection.append(calib_UAr_frame_3_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_45_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_45_o_beforeWPshiftCorrection[i], calib_UAr_I_45_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-45 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
    calib_pixels = []
    calib_I_45_o = []
    calib_I_45_e = []

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_45_o_order = []
        calib_I_45_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, calib_pixel_shift):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_45_o_order.append(calib_UAr_I_45_o_beforeWPshiftCorrection[i][j])
            
            for j in range (calib_pixel_shift, detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_45_o_order.append(calib_UAr_I_45_o_beforeWPshiftCorrection[i][j])
                calib_I_45_e_order.append(calib_UAr_I_45_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-calib_pixel_shift, detector_pixels):
                calib_I_45_e_order.append(calib_UAr_I_45_e_beforeWPshiftCorrection[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, calib_pixel_shift):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_45_e_order.append(calib_UAr_I_45_e_beforeWPshiftCorrection[i][j])
            
            for j in range (calib_pixel_shift, detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_45_o_order.append(calib_UAr_I_45_o_beforeWPshiftCorrection[i][j])
                calib_I_45_e_order.append(calib_UAr_I_45_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-calib_pixel_shift, detector_pixels):
                calib_I_45_o_order.append(calib_UAr_I_45_o_beforeWPshiftCorrection[i][j])
        
        
        calib_pixels.append(calib_pixels_order)
        calib_I_45_o.append(calib_I_45_o_order)
        calib_I_45_e.append(calib_I_45_e_order)
    
    

    calib_UAr_data_cube_4 = calib_UAr_data_cube4
    calib_UAr_frame_4 = calib_UAr_data_cube_4[0][:][:]
    calib_UAr_frame_4 = calib_UAr_frame_4 - master_bias
    calib_UAr_frame_4 = np.rot90(calib_UAr_frame_4)

    # Eliminating bad pixel, ie. pixels with negative counts after bias substraction -- repacing the negative counts with value 0
    bad_pixels = np.where(calib_UAr_frame_4 < 0)
    calib_UAr_frame_4[bad_pixels] = 0

    calib_UAr_frame_4_flux_ADU, sky_frame_4_flux_ADU = OrderExtraction(calib_UAr_frame_4, sky_67pt5, xcor_44, ycor_44, sigma_FWHM, 1)


    calib_UAr_frame_4_flux = []
        
    for i in range (len(orders_41)):
        calib_UAr_frame_4_flux_order = []
        
        for j in range (len(calib_UAr_frame_4_flux_ADU[0])):
            calib_UAr_frame_4_flux_order.append(CCD_gain * (calib_UAr_frame_4_flux_ADU[i][j]))
        
        calib_UAr_frame_4_flux.append((calib_UAr_frame_4_flux_order/max(calib_UAr_frame_4_flux_order)))

    
    calib_UAr_I_67pt5_o_beforeWPshiftCorrection = []
    calib_UAr_I_67pt5_e_beforeWPshiftCorrection = []

    for i in range (len(orders_41)):
        if i%2 != 0:
            calib_UAr_I_67pt5_o_beforeWPshiftCorrection.append(calib_UAr_frame_4_flux[i])
        else:
            calib_UAr_I_67pt5_e_beforeWPshiftCorrection.append(calib_UAr_frame_4_flux[i])


    calib_pixels_beforeWPshiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeWPshiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (len(calib_UAr_I_67pt5_o_beforeWPshiftCorrection)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i]))
    calib_pixel_shift = abs(mode(calib_pixel_shift_order))
    #calib_pixel_shift = 0
    print("Calib frame - HWP-67.5 - o and e ray pixel shift: " + str(calib_pixel_shift))
    
    if manual_oe_spectral_shift_condition == True:
        calib_pixel_shift = manual_oe_spectral_shift
        
    calib_pixels = []
    calib_I_67pt5_o = []
    calib_I_67pt5_e = []

    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_I_67pt5_o_order = []
        calib_I_67pt5_e_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
                
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
                calib_I_67pt5_e_order.append(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_I_67pt5_o_order.append(calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i][j])
        
        calib_pixels.append(calib_pixels_order)
        calib_I_67pt5_o.append(calib_I_67pt5_o_order)
        calib_I_67pt5_e.append(calib_I_67pt5_e_order)



    calib_pixels_beforeFrameShiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeFrameShiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (int(len(orders_41)/2)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_I_0_o[i], calib_I_22pt5_o[i]))
    calib_pixel_shift = mode(calib_pixel_shift_order)
    #calib_pixel_shift = 0
    print("Calib frame - HWP-0 and HWP-22.5 frame pixel shift: " + str(calib_pixel_shift))
    
    if manual_0_22pt5_spectral_shift_condition == True:
        calib_pixel_shift = manual_0_22pt5_spectral_shift
    
    calib_pixels = []
    calib_FrameShift_I_o_0 = []
    calib_FrameShift_I_e_0 = []
    calib_FrameShift_I_o_22pt5 = []
    calib_FrameShift_I_e_22pt5 = []
    
    
    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_FrameShift_I_o_0_order = []
        calib_FrameShift_I_e_0_order = []
        calib_FrameShift_I_o_22pt5_order = []
        calib_FrameShift_I_e_22pt5_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_0_order.append(calib_I_0_o[i][j])
                calib_FrameShift_I_e_0_order.append(calib_I_0_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_0_order.append(calib_I_0_o[i][j])
                calib_FrameShift_I_e_0_order.append(calib_I_0_e[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])          
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])    
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_0_order.append(calib_I_0_o[i][j])
                calib_FrameShift_I_e_0_order.append(calib_I_0_e[i][j])
                calib_FrameShift_I_o_22pt5_order.append(calib_I_22pt5_o[i][j])
                calib_FrameShift_I_e_22pt5_order.append(calib_I_22pt5_e[i][j]) 
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_0_order.append(calib_I_0_o[i][j])
                calib_FrameShift_I_e_0_order.append(calib_I_0_e[i][j])
            
        
        calib_pixels.append(calib_pixels_order)
        calib_FrameShift_I_o_0.append(calib_FrameShift_I_o_0_order)
        calib_FrameShift_I_e_0.append(calib_FrameShift_I_e_0_order)
        calib_FrameShift_I_o_22pt5.append(calib_FrameShift_I_o_22pt5_order)
        calib_FrameShift_I_e_22pt5.append(calib_FrameShift_I_e_22pt5_order)
            
            
            
    calib_pixels_beforeFrameShiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeFrameShiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (int(len(orders_41)/2)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_I_45_o[i], calib_I_67pt5_o[i]))
    calib_pixel_shift = mode(calib_pixel_shift_order)
    #calib_pixel_shift = 0
    print("Calib frame - HWP-45 and HWP-67.5 frame pixel shift: " + str(calib_pixel_shift))
    
    if manual_45_67pt5_spectral_shift_condition == True:
        calib_pixel_shift = manual_45_67pt5_spectral_shift
        
    calib_pixels = []
    calib_FrameShift_I_o_45 = []
    calib_FrameShift_I_e_45 = []
    calib_FrameShift_I_o_67pt5 = []
    calib_FrameShift_I_e_67pt5 = []
    
    
    for i in range (int(len(orders_41)/2)):
        
        calib_pixels_order = []
        calib_FrameShift_I_o_45_order = []
        calib_FrameShift_I_e_45_order = []
        calib_FrameShift_I_o_67pt5_order = []
        calib_FrameShift_I_e_67pt5_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_45_order.append(calib_I_45_o[i][j])
                calib_FrameShift_I_e_45_order.append(calib_I_45_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_45_order.append(calib_I_45_o[i][j])
                calib_FrameShift_I_e_45_order.append(calib_I_45_e[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])          
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShift_I_o_45_order.append(calib_I_45_o[i][j])
                calib_FrameShift_I_e_45_order.append(calib_I_45_e[i][j])
                calib_FrameShift_I_o_67pt5_order.append(calib_I_67pt5_o[i][j])
                calib_FrameShift_I_e_67pt5_order.append(calib_I_67pt5_e[i][j]) 
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShift_I_o_45_order.append(calib_I_45_o[i][j])
                calib_FrameShift_I_e_45_order.append(calib_I_45_e[i][j])
            
               
        calib_pixels.append(calib_pixels_order)
        calib_FrameShift_I_o_45.append(calib_FrameShift_I_o_45_order)
        calib_FrameShift_I_e_45.append(calib_FrameShift_I_e_45_order)
        calib_FrameShift_I_o_67pt5.append(calib_FrameShift_I_o_67pt5_order)
        calib_FrameShift_I_e_67pt5.append(calib_FrameShift_I_e_67pt5_order)
        
        
 
    
    calib_pixels_beforeFrameShiftCorrection = []
    x_new = np.linspace(0, detector_pixels-1, detector_pixels)
    for i in range (len(orders_41)):
        calib_pixels_beforeFrameShiftCorrection.append(x_new)
        
    calib_pixel_shift_order = []
    for i in range (int(len(orders_41)/2)):
        calib_pixel_shift_order.append(find_pixel_shift(calib_FrameShift_I_o_22pt5[i], calib_FrameShift_I_o_67pt5[i]))
    calib_pixel_shift = mode(calib_pixel_shift_order)
    #calib_pixel_shift = 0
    print("Calib frame - HWP-22.5 and HWP-67.5 frame pixel shift: " + str(calib_pixel_shift))
    
    if manual_22pt5_67pt5_spectral_shift_condition == True:
        calib_pixel_shift = manual_22pt5_67pt5_spectral_shift
    
    calib_pixels = []
    calib_FrameShiftFinal_I_o_0 = []
    calib_FrameShiftFinal_I_e_0 = []
    calib_FrameShiftFinal_I_o_22pt5 = []
    calib_FrameShiftFinal_I_e_22pt5 = []
    calib_FrameShiftFinal_I_o_45 = []
    calib_FrameShiftFinal_I_e_45 = []
    calib_FrameShiftFinal_I_o_67pt5 = []
    calib_FrameShiftFinal_I_e_67pt5 = []
    
    for i in range (int(len(orders_41)/2)): 
        calib_pixels_order = []
        calib_FrameShiftFinal_I_o_0_order = []
        calib_FrameShiftFinal_I_e_0_order = []
        calib_FrameShiftFinal_I_o_22pt5_order = []
        calib_FrameShiftFinal_I_e_22pt5_order = []
        calib_FrameShiftFinal_I_o_45_order = []
        calib_FrameShiftFinal_I_e_45_order = []
        calib_FrameShiftFinal_I_o_67pt5_order = []
        calib_FrameShiftFinal_I_e_67pt5_order = []
        
        if calib_pixel_shift < 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShiftFinal_I_o_0_order.append(calib_FrameShift_I_o_0[i][j])
                calib_FrameShiftFinal_I_e_0_order.append(calib_FrameShift_I_e_0[i][j])
                calib_FrameShiftFinal_I_o_22pt5_order.append(calib_FrameShift_I_o_22pt5[i][j])
                calib_FrameShiftFinal_I_e_22pt5_order.append(calib_FrameShift_I_e_22pt5[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShiftFinal_I_o_0_order.append(calib_FrameShift_I_o_0[i][j])
                calib_FrameShiftFinal_I_e_0_order.append(calib_FrameShift_I_e_0[i][j])
                calib_FrameShiftFinal_I_o_22pt5_order.append(calib_FrameShift_I_o_22pt5[i][j])
                calib_FrameShiftFinal_I_e_22pt5_order.append(calib_FrameShift_I_e_22pt5[i][j])
                calib_FrameShiftFinal_I_o_45_order.append(calib_FrameShift_I_o_45[i][j])
                calib_FrameShiftFinal_I_e_45_order.append(calib_FrameShift_I_e_45[i][j])
                calib_FrameShiftFinal_I_o_67pt5_order.append(calib_FrameShift_I_o_67pt5[i][j])
                calib_FrameShiftFinal_I_e_67pt5_order.append(calib_FrameShift_I_e_67pt5[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShiftFinal_I_o_45_order.append(calib_FrameShift_I_o_45[i][j])
                calib_FrameShiftFinal_I_e_45_order.append(calib_FrameShift_I_e_45[i][j])
                calib_FrameShiftFinal_I_o_67pt5_order.append(calib_FrameShift_I_o_67pt5[i][j])
                calib_FrameShiftFinal_I_e_67pt5_order.append(calib_FrameShift_I_e_67pt5[i][j])
        
        elif calib_pixel_shift >= 0:
            
            for j in range (0, abs(calib_pixel_shift)):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShiftFinal_I_o_45_order.append(calib_FrameShift_I_o_45[i][j])
                calib_FrameShiftFinal_I_e_45_order.append(calib_FrameShift_I_e_45[i][j])
                calib_FrameShiftFinal_I_o_67pt5_order.append(calib_FrameShift_I_o_67pt5[i][j])
                calib_FrameShiftFinal_I_e_67pt5_order.append(calib_FrameShift_I_e_67pt5[i][j])
                
            for j in range (abs(calib_pixel_shift), detector_pixels):
                calib_pixels_order.append(calib_pixels_beforeFrameShiftCorrection[i][j])
                calib_FrameShiftFinal_I_o_0_order.append(calib_FrameShift_I_o_0[i][j])
                calib_FrameShiftFinal_I_e_0_order.append(calib_FrameShift_I_e_0[i][j])
                calib_FrameShiftFinal_I_o_22pt5_order.append(calib_FrameShift_I_o_22pt5[i][j])
                calib_FrameShiftFinal_I_e_22pt5_order.append(calib_FrameShift_I_e_22pt5[i][j])
                calib_FrameShiftFinal_I_o_45_order.append(calib_FrameShift_I_o_45[i][j])
                calib_FrameShiftFinal_I_e_45_order.append(calib_FrameShift_I_e_45[i][j])
                calib_FrameShiftFinal_I_o_67pt5_order.append(calib_FrameShift_I_o_67pt5[i][j])
                calib_FrameShiftFinal_I_e_67pt5_order.append(calib_FrameShift_I_e_67pt5[i][j])
            
            for j in range (detector_pixels-abs(calib_pixel_shift), detector_pixels):
                calib_FrameShiftFinal_I_o_0_order.append(calib_FrameShift_I_o_0[i][j])
                calib_FrameShiftFinal_I_e_0_order.append(calib_FrameShift_I_e_0[i][j])
                calib_FrameShiftFinal_I_o_22pt5_order.append(calib_FrameShift_I_o_22pt5[i][j])
                calib_FrameShiftFinal_I_e_22pt5_order.append(calib_FrameShift_I_e_22pt5[i][j])
       
        
        calib_pixels.append(calib_pixels_order)
        calib_FrameShiftFinal_I_o_0.append(calib_FrameShiftFinal_I_o_0_order)
        calib_FrameShiftFinal_I_e_0.append(calib_FrameShiftFinal_I_e_0_order)
        calib_FrameShiftFinal_I_o_22pt5.append(calib_FrameShiftFinal_I_o_22pt5_order)
        calib_FrameShiftFinal_I_e_22pt5.append(calib_FrameShiftFinal_I_e_22pt5_order)
        calib_FrameShiftFinal_I_o_45.append(calib_FrameShiftFinal_I_o_45_order)
        calib_FrameShiftFinal_I_e_45.append(calib_FrameShiftFinal_I_e_45_order)
        calib_FrameShiftFinal_I_o_67pt5.append(calib_FrameShiftFinal_I_o_67pt5_order)
        calib_FrameShiftFinal_I_e_67pt5.append(calib_FrameShiftFinal_I_e_67pt5_order)
        
    

    #plt.plot(calib_FrameShiftFinal_I_o_22pt5[15])
    #plt.plot(calib_FrameShiftFinal_I_e_22pt5[15])
    #plt.plot(calib_FrameShiftFinal_I_o_67pt5[15])
    #plt.plot(calib_FrameShiftFinal_I_e_67pt5[15])
    #plt.show()



    # Get the list of all files and directories
    if grating_choice == 1:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/RedCD/UAr/"######################################################################################
    elif grating_choice == 2:
        path1 = f"{project_root}/UAr_Wavelength-Calibrarion/20231130/BlueCD/UAr/"#####################################################################################
      
    # ~ if grating_choice == 1:
        # ~ path1 = "C:\\Users\\Mudit Shrivastav\\.ipython\\UAr_order_flux\\20231130\\RedCD\\UAr\\"
    # ~ elif grating_choice == 2:
        # ~ path1 = "C:\\Users\\Mudit Shrivastav\\.ipython\\UAr_order_flux\\20231130\\BlueCD\\UAr\\"
        
    dir_list = os.listdir(path1)
    #print("Files and directories in '", path1, "' :")
    # prints all files
    #print(dir_list)
    index = []
    for i in range (len(dir_list)):
        if dir_list[i].find('o') == -1:
            index.append(i)

    dir_list1 = []
    for i in range (len(index)):
        dir_list1.append(dir_list[index[i]])
      
    if grating_choice == 1:
        orders_ref = np.linspace(28, 48, 21)
    elif grating_choice == 2:
        orders_ref = np.linspace(44, 65, 22)
        
    orders_ref = orders_ref.astype('int32')

    calib_UAr_I_ref = []
    for i in range (len(dir_list1)):
        pix, calib_f = np.loadtxt(path1+dir_list1[i], usecols=(0, 1), delimiter = ',', unpack = True)
        calib_UAr_I_ref.append((calib_f/max(calib_f)))


    # Get the list of all files and directories
    if grating_choice == 1:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/RedCD/"
    elif grating_choice == 2:
        path2 = "/home/jasvant/data-reduction/UAr_Wavelength-Calibrarion/Wavelength_Calibration_Reference/BlueCD/"
    
        # ~ if grating_choice == 1:
        # ~ path2 = "C:\\Users\\Mudit Shrivastav\\.ipython\\Wavelength_Calibration_Reference\\RedCD\\"
    # ~ elif grating_choice == 2:
        # ~ path2 = "C:\\Users\\Mudit Shrivastav\\.ipython\\Wavelength_Calibration_Reference\\BlueCD\\"
        
    dir_list2 = os.listdir(path2)
    # ~ dir_list2.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))


    calib_UAr_reference_pix = []
    calib_UAr_reference_wave = []
    for i in range (len(dir_list2)):
        pix, wave = np.loadtxt(path2+dir_list2[i], usecols=(0, 1), unpack = True)
        calib_UAr_reference_pix.append(pix)
        calib_UAr_reference_wave.append(wave)




    orders_41_wc = list(set(orders_41))
    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc.sort()
    orders_41_wc = np.flip(orders_41_wc)


    pixel_shift_wc_list = []

    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_FrameShiftFinal_I_e_67pt5[i],calib_UAr_I_ref[ref_order_match])
            
            print(pixel_shift_wc)
            pixel_shift_wc_list.append(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            #plt.plot(x_new, calib_I_67pt5_o[i])
            #plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
    
            
    def guess_gaussian_params(x, y):
        
        #Guess the initial parameters for a Gaussian fit to the data (x, y).
        
        mean = np.mean(x)
        stddev = np.std(x)
        amplitude = np.max(y)
        return [amplitude, mean, stddev]

    def gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    
    UAr_lamp_sigma_dispersion = 1
    
    residual_list = []
    
    if grating_choice == 1:
        gaussian_wavelength_for_shifting = [5527.980, 5650.704, 5772.118, 5888.580, 6059.373, 6155.237, 6307.657, 6416.307, 6578.794, 6766.612, 6937.664, 7147.041, 7372.118, 7590.524, 7814.326, 8046.115, 8273.505, 8521.441, 8799.086, 9093.654, 9354.219]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 28:
                diff_low_order = np.min(orders_41_wc) - 28
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 48:
                diff_low_order = 48 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting)  
          
    elif grating_choice == 2:
        gaussian_wavelength_for_shifting = [4052.921, 4131.735, 4200.674, 4277.551, 4348.081, 4379.689, 4481.835, 4579.367, 4589.920, 4726.899, 4806.042, 4879.873, 4965.099, 5063.756, 5187.746, 5315.282, 5410.472, 5451.655, 5572.543, 5739.520, 5882.624, 6032.128]
        
        if len(orders_41_wc) < len(gaussian_wavelength_for_shifting):
            if np.min(orders_41_wc) > 44:
                diff_low_order = np.min(orders_41_wc) - 44
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(-1)
                    k = k + 1
            if np.max(orders_41_wc) < 65:                
                diff_low_order = 65 - np.max(orders_41_wc)
                k = 0
                while k < diff_low_order:
                    gaussian_wavelength_for_shifting.pop(0)
                    k = k + 1  
        
        gaussian_wavelength_for_shifting = np.array(gaussian_wavelength_for_shifting) 
    
    print(orders_41_wc)
    print(gaussian_wavelength_for_shifting)
    print(len(orders_41_wc))
    print(len(gaussian_wavelength_for_shifting))
    
    gaussian_residual_add = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            pixel_shift_wc = find_pixel_shift(calib_FrameShiftFinal_I_e_67pt5[i],calib_UAr_I_ref[ref_order_match])
            
            if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
                pixel_shift_wc = mode(pixel_shift_wc_list)
            
            #print(pixel_shift_wc)
            #pixel_shift_wc = dispersion_pixel_manual_shift

            #plt.plot(x_new, calib_I_67pt5_o[i])
            #plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #plt.title(str(orders_41_wc[i]))
            #plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + pixel_shift_wc
                
            
            
            residual = []   
            for j in range (len(calib_FrameShiftFinal_I_e_67pt5[i])):
                 
                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            
                            diff = (j+(4*UAr_lamp_sigma_dispersion)) - (j-(4*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j-(2*UAr_lamp_sigma_dispersion)),(j+(6*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_FrameShiftFinal_I_e_67pt5[i][(j-(4*UAr_lamp_sigma_dispersion)):(j+(4*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            residual.append(gaussian_peak - calib_UAr_reference_pix[ref_order_match][k])
                            

                    except:
                        continue
            
            residual_list.append(residual)
            
               
            #residual = np.array(residual)
            #print(residual)
            #median_residual = np.median(residual)
            median_residual = min(residual)
            print(median_residual)
            pix = []
            wave = []    
            for j in range (len(calib_FrameShiftFinal_I_e_67pt5[i])):

                for k in range (len(calib_UAr_reference_pix[ref_order_match])):
                    try:
                        if j == calib_UAr_reference_pix[ref_order_match][k]:
                            j_res = int(round(j + median_residual)) - 1
                            
                            diff = (j_res+(2*UAr_lamp_sigma_dispersion)) - (j_res-(2*UAr_lamp_sigma_dispersion)) + 1
                            gaussian_extract_pixels = np.linspace((j_res-(2*UAr_lamp_sigma_dispersion)),(j_res+(2*UAr_lamp_sigma_dispersion)), diff)
                            gaussian_extract_flux = calib_FrameShiftFinal_I_e_67pt5[i][(j_res-(2*UAr_lamp_sigma_dispersion)):(j_res+(2*UAr_lamp_sigma_dispersion)+1)]
                            
                            init_param = guess_gaussian_params(gaussian_extract_pixels, gaussian_extract_flux)
                            popt, pcov = curve_fit(gaussian, gaussian_extract_pixels, gaussian_extract_flux, init_param)
                            
                            gaussian_peak = popt[1]
                            
                            fit_x = np.linspace(min(gaussian_extract_pixels), max(gaussian_extract_pixels), 100)
                            fit = gaussian(fit_x, popt[0], popt[1], popt[2])
                            #plt.plot(gaussian_extract_pixels, gaussian_extract_flux, color = 'b')
                            #plt.plot(fit_x, fit, color = 'r')
                            #plt.title(str(orders_41_wc[i]) + "  " + str(calib_UAr_reference_pix[ref_order_match][k]) + "  " + str(calib_UAr_reference_wave[ref_order_match][k]))
                            #plt.show()
                            
                            if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                                if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                    gaussian_residual_add.append((gaussian_peak - calib_UAr_reference_pix[ref_order_match][k]))
                                    
                            #else:
                            #    gaussian_residual_add.append(0)
                            
                            pix.append(gaussian_peak)
                            wave.append(calib_UAr_reference_wave[ref_order_match][k])
                            
                    except:
                        pix.append(calib_UAr_reference_pix[ref_order_match][k] + int(round(median_residual)))
                        wave.append(calib_UAr_reference_wave[ref_order_match][k])
                        if len(orders_41_wc) == len(gaussian_wavelength_for_shifting): 
                            if calib_UAr_reference_wave[ref_order_match][k] == gaussian_wavelength_for_shifting[i]:
                                if not gaussian_residual_add:
                                    gaussian_residual_add.append(0)
                                else:
                                    gaussian_residual_add.append(statistics.median(gaussian_residual_add))
                                
                        #gaussian_residual_add.append(0)
            
            pix = np.array(pix)
            wave = np.array(wave)
            #calib_UAr_reference_pix_negative_pixels_removed.append(pix)
            #calib_UAr_reference_wave_negative_pixels_removed.append(wave)
            
    for i in range (len(gaussian_residual_add)):
        if gaussian_residual_add[i] > 5 or gaussian_residual_add[i] < -5:
            gaussian_residual_add[i] = statistics.median(gaussian_residual_add)

    print("\n Gaussian Residual Add: \n")
    print(gaussian_residual_add)
    
    calib_UAr_reference_pix_gaussian_dispersion_corrected = []
    calib_UAr_reference_wave_gaussian_dispersion_corrected = []
    
    for i in range (len(calib_UAr_I_67pt5_e_beforeWPshiftCorrection)):
        
        if len(np.where(orders_ref == orders_41_wc[i])[0]) == 0:
            continue
        else:
            ref_order_match = np.where(orders_ref == orders_41_wc[i])[0][0]
            #if i == 17:
            #    plt.plot(x_new, calib_UAr_I_67pt5_o_beforeWPshiftCorrection[i], label="o")
            #    plt.plot(x_new, calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i], label="e")
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match], label="ref")
            #    plt.legend()
            #    plt.show()
                
            #pixel_shift_wc_e = find_pixel_shift(calib_UAr_I_67pt5_e_beforeWPshiftCorrection[i],calib_UAr_I_ref[ref_order_match])
            #pixel_shift_wc = find_pixel_shift(calib_I_67pt5_o[i],calib_UAr_I_ref[ref_order_match])
            
            #if abs(pixel_shift_wc - mode(pixel_shift_wc_list)) > 3:
            #    pixel_shift_wc = mode(pixel_shift_wc_list)
            
            #print(pixel_shift_wc)
            #if orders_41_wc[i] == 28:
            #    plt.plot(x_new, calib_I_67pt5_o[i])
            #    plt.plot(x_new, calib_UAr_I_ref[ref_order_match])
            #    plt.show()
            #print(str(orders_41_wc[i]) + "  " + str(pixel_shift_wc_e))
            
            for j in range (len(calib_UAr_reference_pix[ref_order_match])):
                calib_UAr_reference_pix[ref_order_match][j] = calib_UAr_reference_pix[ref_order_match][j] + gaussian_residual_add[i]
            
            calib_UAr_reference_pix_gaussian_dispersion_corrected.append(calib_UAr_reference_pix[ref_order_match])
            calib_UAr_reference_wave_gaussian_dispersion_corrected.append(calib_UAr_reference_wave[ref_order_match])
    
    
    
    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix_gaussian_dispersion_corrected[i])):
            #if calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] >= 0 and calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j] <= (detector_pixels-1):
            pix.append(calib_UAr_reference_pix_gaussian_dispersion_corrected[i][j])
            wave.append(calib_UAr_reference_wave_gaussian_dispersion_corrected[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave) 
                       
    

    #for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
    #    print(list(calib_UAr_reference_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_wave_negative_pixels_removed[i]))
    #    print("\n")

    #orders_41_rev = np.flip(orders_41)

    """

    calib_UAr_reference_pix_negative_pixels_removed = []
    calib_UAr_reference_wave_negative_pixels_removed = []

    for i in range (len(calib_UAr_reference_pix)):
        pix = []
        wave = []
        for j in range (len(calib_UAr_reference_pix[i])):
            if calib_UAr_reference_pix[i][j] >= 0 and calib_UAr_reference_pix[i][j] <= (detector_pixels-1):
                pix.append(calib_UAr_reference_pix[i][j])
                wave.append(calib_UAr_reference_wave[i][j])
        pix = np.array(pix)
        wave = np.array(wave)
        calib_UAr_reference_pix_negative_pixels_removed.append(pix)
        calib_UAr_reference_wave_negative_pixels_removed.append(wave)
    
    """

        
        
        


    #for i in range (int(len(orders_41)/2)):
    #    order_number = orders_41_rev[(2*i)]
    #    print(order_number)
    #    print("e-ray")
    #    print(list(calib_UAr_reference_e_pix_negative_pixels_removed[i]))
    #    print(list(calib_UAr_reference_e_wave_negative_pixels_removed[i]))
        #print("o-ray")
        #print(list(calib_UAr_reference_o_pix_negative_pixels_removed[i]))
        #print(list(calib_UAr_reference_o_wave_negative_pixels_removed[i]))
       
    def arraySortedOrNot(arr, n):
    # Array has one or no element
        if (n == 0 or n == 1):
            return True
        for i in range(1, n):
        # Unsorted pair found
            if (arr[i-1] < arr[i]):
                return False
    # No unsorted pair found
        return True
       
    wavelength = []
    for i in range (len(calib_UAr_reference_pix_negative_pixels_removed)):
        pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'cubic', bounds_error = False, fill_value="extrapolate")
       
        pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
        pixel_row = pixel_row.astype('int32')
        wave = pix_wave_e(pixel_row)
        if arraySortedOrNot(wave, len(wave)) == True:
            print(str(orders_41_wc[i]) + " Cubic")
            
        if arraySortedOrNot(wave, len(wave)) == False:
            pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'quadratic', bounds_error = False, fill_value="extrapolate")            
            pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
            pixel_row = pixel_row.astype('int32')
            wave = pix_wave_e(pixel_row)
            if arraySortedOrNot(wave, len(wave)) == True:
                print(str(orders_41_wc[i]) + " Quadratic")
                
            if arraySortedOrNot(wave, len(wave)) == False:
                pix_wave_e = interp1d(calib_UAr_reference_pix_negative_pixels_removed[i], calib_UAr_reference_wave_negative_pixels_removed[i], kind = 'linear', bounds_error = False, fill_value="extrapolate")
                print(str(orders_41_wc[i]) + " Linear")
                pixel_row = np.linspace(0, detector_pixels-1, detector_pixels)
                pixel_row = pixel_row.astype('int32')
                wave = pix_wave_e(pixel_row)
                
        wavelength.append(wave)
        
    wavelength = wavelength[::-1]
    
    #for i in range (len(wavelength_e)):
    #    print(str(orders_41_wc[i]) + "  " + str(min(wavelength_e[i])) + "  " + str(max(wavelength_e[i])))


    #wavelength = []
    #for i in range (len(wavelength_e)):
    #    wavelength.append(wavelength_e[i])
    #    wavelength.append(wavelength_o[i])

    #path_wavecal = path + "\\Wavelength_Calibrated\\"

    #if not os.path.exists(path_wavecal):
    #    os.mkdir(path_wavecal)

    orders_41_wc = np.array(orders_41_wc)
    orders_41_wc = np.flip(orders_41_wc)

    wavelength_calibrated_intensities = []
    
    for i in range (len(orders_41_wc)):
        
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_intensities_order = []
            #wavelength_calibrated_intensities_order.append(wavelength[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(wavelength[i]) 
            wavelength_calibrated_intensities_order.append(I_0_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_0_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_22pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_22pt5_e_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_45_e_err[len(I_0_e)-1-i])    
            wavelength_calibrated_intensities_order.append(I_67pt5_o_err[len(I_0_e)-1-i])
            wavelength_calibrated_intensities_order.append(I_67pt5_e_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_intensities.append(wavelength_calibrated_intensities_order)
        
            




    I0_o_I0_e = []
    I45_o_I45_e = []
    I22pt5_o_I22pt5_e = []
    I67pt5_o_I67pt5_e = []

    Aq = []
    Au = []

    #Aq_interpolated = []
    #Au_interpolated = []

    q = []
    u = []
    p = []
    theta = []

    q_err = []
    u_err = []
    p_err = []
    theta_err = []



    #Aq_raw = []
    #Au_raw = []

    #q_raw = []
    #u_raw = []
    #p_raw = []
    #theta_raw = []

    #q_raw_err = []
    #u_raw_err = []
    #p_raw_err = []
    #theta_raw_err = []



    #def fit_func (x, a, b, c, d):
    #    return a*(x**3) + b*(x**2) +c*x + d

    for i in range (int(len(orders_41)/2)):
        
        I0_o_I0_e_order = []
        I45_o_I45_e_order = []
        I22pt5_o_I22pt5_e_order = []
        I67pt5_o_I67pt5_e_order = []
        
        for j in range (detector_pixels):
            I0_o_I0_e_order.append(I_0_o[i][j]/I_0_e[i][j])
            I45_o_I45_e_order.append(I_45_o[i][j]/I_45_e[i][j])
            I22pt5_o_I22pt5_e_order.append(I_22pt5_o[i][j]/I_22pt5_e[i][j])
            I67pt5_o_I67pt5_e_order.append(I_67pt5_o[i][j]/I_67pt5_e[i][j])
        
        
        temp_I0_o_I0_e_order = I0_o_I0_e_order
        temp_I45_o_I45_e_order = I45_o_I45_e_order
        temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
            I0_o_I0_e_median = []
            I45_o_I45_e_median = []
            I22pt5_o_I22pt5_e_median = []
            I67pt5_o_I67pt5_e_median = []
            
            for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)+1):
                I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
                I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
                I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
                I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
            I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
            I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
            I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
            I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        I0_o_I0_e.append(I0_o_I0_e_order)
        I45_o_I45_e.append(I45_o_I45_e_order)
        I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        Aq_order = []
        Au_order = []
        for j in range (detector_pixels):
            Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
            Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        Aq_order = np.array(Aq_order)
        Au_order = np.array(Au_order)
        
        Aq.append(Aq_order)
        Au.append(Au_order)

        q_order = []
        u_order = []
        q_err_order = []
        u_err_order = []

        for j in range (detector_pixels):
            q_order.append((Aq_order[j] - 1)/(Aq_order[j] + 1))
            u_order.append((Au_order[j] - 1)/(Au_order[j] + 1))
            q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
            u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        q_order = np.array(q_order)
        u_order = np.array(u_order)
        q_err_order = np.array(q_err_order)
        u_err_order = np.array(u_err_order)


        p_order = []
        theta_order = []
        for j in range (detector_pixels):
            p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
            theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        p_order = np.array(p_order)
        theta_order = np.array(theta_order)

        p_err_order = []
        theta_err_order = []
        for j in range (detector_pixels):
            p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
            theta_err_order.append(p_err_order[j]/(2*p_order[j]) * (180/np.pi))
            
        p_err_order = np.array(p_err_order)
        theta_err_order = np.array(theta_err_order)
        
        
        q.append(q_order)
        u.append(u_order)
        p.append(p_order)
        theta.append(theta_order)
        
        q_err.append(q_err_order)
        u_err.append(u_err_order)
        p_err.append(p_err_order)
        theta_err.append(theta_err_order)
        
        

        
        
        #temp_I0_o_I0_e_order = I0_o_I0_e_order
        #temp_I45_o_I45_e_order = I45_o_I45_e_order
        #temp_I22pt5_o_I22pt5_e_order = I22pt5_o_I22pt5_e_order
        #temp_I67pt5_o_I67pt5_e_order = I67pt5_o_I67pt5_e_order
        
        #for j in range (int(median_smoothening_pixels/2), detector_pixels-int(median_smoothening_pixels/2)-1):
        #    I0_o_I0_e_median = []
        #    I45_o_I45_e_median = []
        #    I22pt5_o_I22pt5_e_median = []
        #    I67pt5_o_I67pt5_e_median = []
            
        #    for k in range (-int(median_smoothening_pixels/2), int(median_smoothening_pixels/2)):
        #        I0_o_I0_e_median.append(temp_I0_o_I0_e_order[j+k])
        #        I45_o_I45_e_median.append(temp_I45_o_I45_e_order[j+k])
        #        I22pt5_o_I22pt5_e_median.append(temp_I22pt5_o_I22pt5_e_order[j+k])
        #        I67pt5_o_I67pt5_e_median.append(temp_I67pt5_o_I67pt5_e_order[j+k])
            
        #    I0_o_I0_e_order[j] = statistics.median(I0_o_I0_e_median)
        #    I45_o_I45_e_order[j] = statistics.median(I45_o_I45_e_median)
        #    I22pt5_o_I22pt5_e_order[j] = statistics.median(I22pt5_o_I22pt5_e_median)
        #    I67pt5_o_I67pt5_e_order[j] = statistics.median(I67pt5_o_I67pt5_e_median)
        
        #I0_o_I0_e.append(I0_o_I0_e_order)
        #I45_o_I45_e.append(I45_o_I45_e_order)
        #I22pt5_o_I22pt5_e.append(I22pt5_o_I22pt5_e_order)
        #I67pt5_o_I67pt5_e.append(I67pt5_o_I67pt5_e_order)
        
        
        #Aq_order = []
        #Au_order = []
        #for j in range (detector_pixels):
        #    Aq_order.append(np.sqrt((I0_o_I0_e_order[j])/(I45_o_I45_e_order[j])))
        #    Au_order.append(np.sqrt((I22pt5_o_I22pt5_e_order[j])/(I67pt5_o_I67pt5_e_order[j])))
            
        #Aq_order = np.array(Aq_order)
        #Au_order = np.array(Au_order)
        
        #Aq.append(Aq_order)
        #Au.append(Au_order)
        
        
        #pixels_interp_points = []
        #Aq_order_interp_points = []
        #Au_order_interp_points = []
            
        #for j in range (0, detector_pixels, 1):
        #    pixels_interp_points.append(pixels[i][j])
        #    Aq_order_interp_points.append(Aq_order[j])
        #    Au_order_interp_points.append(Au_order[j])
        
        
        #Aq_polynomial = interp1d(pixels_interp_points, Aq_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #Au_polynomial = interp1d(pixels_interp_points, Au_order_interp_points, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #Aq_polynomial_coeff = np.polyfit(pixels_interp_points, Aq_order_interp_points, 15) 
        #Au_polynomial_coeff = np.polyfit(pixels_interp_points, Au_order_interp_points, 15)
        #Aq_polynomial = np.poly1d(Aq_polynomial_coeff)
        #Au_polynomial = np.poly1d(Au_polynomial_coeff)
        
        #Aq_interp = Aq_polynomial(pixels[i])
        #Au_interp = Au_polynomial(pixels[i])
        
        #I0_o_I0_e_polynomial = interp1d(pixels[i], I0_o_I0_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I45_o_I45_e_polynomial = interp1d(pixels[i], I45_o_I45_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I22pt5_o_I22pt5_e_polynomial = interp1d(pixels[i], I22pt5_o_I22pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        #I67pt5_o_I67pt5_e_polynomial = interp1d(pixels[i], I67pt5_o_I67pt5_e_order, kind = 'cubic', bounds_error = False, fill_value="extrapolate")
        
        #popt_q, pcov_q = curve_fit(fit_func, pixels_interp_points, Aq_order_interp_points)
        #popt_u, pcov_u = curve_fit(fit_func, pixels_interp_points, Au_order_interp_points)
        #Aq_interp = []
        #Au_interp = []
        #aq, bq, cq, dq = popt_q
        #au, bu, cu, du = popt_u
        
        #for j in range (len(pixels[i])):        
        #    Aq_interp.append(fit_func(pixels[i][j], aq, bq, cq, dq))
        #    Au_interp.append(fit_func(pixels[i][j], au, bu, cu, du))
        
        #Aq_interp = np.array(Aq_interp)
        #Au_interp = np.array(Au_interp)
        
        #Aq_interpolated.append(Aq_interp)
        #Au_interpolated.append(Au_interp)


        #q_order = []
        #u_order = []
        #q_err_order = []
        #u_err_order = []

        #for j in range (detector_pixels):
        #    q_order.append((Aq_interp[j] - 1)/(Aq_interp[j] + 1))
        #    u_order.append((Au_interp[j] - 1)/(Au_interp[j] + 1))
        #    q_err_order.append(np.sqrt((I_0_o_err[i][j]/I_0_o[i][j])**2 + (I_0_e_err[i][j]/I_0_e[i][j])**2 + (I_45_o_err[i][j]/I_45_o[i][j])**2 + (I_45_e_err[i][j]/I_45_e[i][j])**2) * (Aq_order[j]/(Aq_order[j]+1)**2))
        #    u_err_order.append(np.sqrt((I_22pt5_o_err[i][j]/I_22pt5_o[i][j])**2 + (I_22pt5_e_err[i][j]/I_22pt5_e[i][j])**2 + (I_67pt5_o_err[i][j]/I_67pt5_o[i][j])**2 + (I_67pt5_e_err[i][j]/I_67pt5_e[i][j])**2) * (Au_order[j]/(Au_order[j]+1)**2))

        #q_order = np.array(q_order)
        #u_order = np.array(u_order)
        #q_err_order = np.array(q_err_order)
        #u_err_order = np.array(u_err_order)


        #p_order = []
        #theta_order = []
        #for j in range (detector_pixels):
        #    p_order.append(np.sqrt((q_order[j])**2 + (u_order[j])**2))
        #    theta_order.append((1/2) * np.arctan(q_order[j]/u_order[j]) * (180/np.pi))
            
        #p_order = np.array(p_order)
        #theta_order = np.array(theta_order)

        #p_err_order = []
        #theta_err_order = []
        #for j in range (detector_pixels):
        #    p_err_order.append(np.sqrt(abs(q_order[j]*(q_err_order[j]**2) + u_order[j]*(u_err_order[j]**2))/p_order[j]))
        #    theta_err_order.append(p_err_order[j]/(2*p_order[j]))
            
        #p_err_order = np.array(p_err_order)
        #theta_err_order = np.array(theta_err_order)
        
        
        #q.append(q_order)
        #u.append(u_order)
        #p.append(p_order)
        #theta.append(theta_order)
        
        #q_err.append(q_err_order)
        #u_err.append(u_err_order)
        #p_err.append(p_err_order)
        #theta_err.append(theta_err_order)
        
        
    wavelength_calibrated_Stokes_parameters = []

    for i in range (len(orders_41_wc)):
        
        
        order_diff = int(min(orders_ref) - min(orders_41))
        #order_number = orders_41_rev[(2*i)]
        order_number = orders_41_wc[i]
        if grating_choice == 1:
            start_order = 28
        elif grating_choice == 2:
            start_order = 44
        
        if order_number >= start_order:
            wavelength_calibrated_Stokes_parameters_order = []
            
            wavelength_calibrated_Stokes_parameters_order.append(wavelength[i])
            wavelength_calibrated_Stokes_parameters_order.append(I0_o_I0_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I45_o_I45_e[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(I22pt5_o_I22pt5_e[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(I67pt5_o_I67pt5_e[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters_order.append(Aq[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(Au[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_interpolated[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_interpolated[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(q_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(u_err[len(I_0_e)-1-i])          
            wavelength_calibrated_Stokes_parameters_order.append(p_err[len(I_0_e)-1-i])
            wavelength_calibrated_Stokes_parameters_order.append(theta_err[len(I_0_e)-1-i])
            
            #wavelength_calibrated_Stokes_parameters_order.append(Aq_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(Au_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(q_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(u_raw_err[len(I_0_e)-1-i])          
            #wavelength_calibrated_Stokes_parameters_order.append(p_raw_err[len(I_0_e)-1-i])
            #wavelength_calibrated_Stokes_parameters_order.append(theta_raw_err[len(I_0_e)-1-i])
            
            wavelength_calibrated_Stokes_parameters.append(wavelength_calibrated_Stokes_parameters_order)
        
        
    print("calibration 4 finished")
    return (wavelength_calibrated_intensities, wavelength_calibrated_Stokes_parameters)


############ Load Parameters ####################
param_file = f'{project_root}/parameters.txt' 
parameters = load_parameters(param_file) 
# Assign variables
CCD_gain = parameters['CCD_gain']
detector_pixels = parameters['detector_pixels']

star_name = parameters['star_name']
set_number = parameters['set_number']
standard_star_name = parameters['standard_star_name']
observation_session = parameters['observation_session']

grating_choice = parameters['grating_choice']

manual_oe_spectral_shift_condition = parameters['manual_oe_spectral_shift_condition']
manual_oe_spectral_shift = parameters['manual_oe_spectral_shift']

manual_0_22pt5_spectral_shift_condition = parameters['manual_0_22pt5_spectral_shift_condition']
manual_0_22pt5_spectral_shift = parameters['manual_0_22pt5_spectral_shift']

manual_45_67pt5_spectral_shift_condition = parameters['manual_45_67pt5_spectral_shift_condition']
manual_45_67pt5_spectral_shift = parameters['manual_45_67pt5_spectral_shift']

manual_22pt5_67pt5_spectral_shift_condition = parameters['manual_22pt5_67pt5_spectral_shift_condition']
manual_22pt5_67pt5_spectral_shift = parameters['manual_22pt5_67pt5_spectral_shift']

median_smoothening_pixels = parameters['median_smoothening_pixels']

science_frame_starting_order = parameters['starting_order']
science_frame_NumberOfPeaks = parameters['number_of_peaks']

IsTraceFrameAvailable = parameters['IsTraceFrameAvailable']
NumberOfCalibFrames = parameters['NumberOfCalibFrames']

###########  Input all science bias substracted sky fits files  ###############


sky_frame_path = f'{project_root}/intermediate/sky/'
sky_0_cube = fits.getdata(f'{sky_frame_path}/Sky_HD61421_RedCD-FilterIn_300s_HWPPosition-0_EncVal-0pt45_1.fits')
sky_0 = sky_0_cube[0][:][:]
sky_22pt5_cube = fits.getdata(f'{sky_frame_path}/Sky_HD61421_RedCD-FilterIn_300s_HWPPosition-22pt5_EncVal-1574pt64_2.fits')
sky_22pt5 = sky_22pt5_cube[0][:][:]

sky_45_cube = fits.getdata(f'{sky_frame_path}/Sky_HD61421_RedCD-FilterIn_300s_HWPPosition-45_EncVal-3150pt0_3.fits')
sky_45 = sky_45_cube[0][:][:]

sky_67pt5_cube = fits.getdata(f'{sky_frame_path}/Sky_HD61421_RedCD-FilterIn_300s_HWPPosition-67pt5_EncVal-4725pt45_4.fits')
sky_67pt5 = sky_67pt5_cube[0][:][:]


###############   Input all extracted intensities and errors here   #############################
science_frame_path = f'{project_root}/intermediate/science/HD61421_RedCD-FilterIn_20s_HWPPosition-0_EncVal-0pt63_1.fits' ####### normal science file for Observational date extraction
science_frame_42_path = f'{project_root}/intermediate/science/scattered_light_substraction/science_bias_scattered_light_subtracted_2.fits'
science_frame_42 = fits.getdata(science_frame_42_path)
sigma_FWHM = Star_FWHM_Determination(science_frame_42, science_frame_starting_order, science_frame_NumberOfPeaks)
# ~ print(sigma_FWHM)

with fits.open(science_frame_path) as hdul:
    # Get the header of the primary HDU (Header/Data Unit)
    header = hdul[0].header
    # ~ science_frame_exposure = header['EXPOSURE']
    observation_date = header['FRAME']
    observation_date = observation_date[0:10]
    print("Observation Date: " + str(observation_date))
    star_name = star_name + "_" + str(observation_date)

traced_dir = f'{project_root}/intermediate/science/traced_orders'  # <-- Change this path

if IsTraceFrameAvailable == 0:
    orders_41, peaks_41, xcor_41, ycor_41 = load_traced_orders_from_txt(f'{traced_dir}/1')   #  Load from text file
    orders_42, peaks_42, xcor_42, ycor_42 = load_traced_orders_from_txt(f'{traced_dir}/2')     
    orders_43, peaks_43, xcor_43, ycor_43 = load_traced_orders_from_txt(f'{traced_dir}/3')
    orders_44, peaks_44, xcor_44, ycor_44 = load_traced_orders_from_txt(f'{traced_dir}/4')



orders_num = list(range(28, 46))  # 27 to 45 inclusive

# Initialize empty lists for each intensity type across all orders
I_0_o_all = []
I_0_e_all = []
I_22pt5_o_all = []
I_22pt5_e_all = []
I_45_o_all = []
I_45_e_all = []
I_67pt5_o_all = []
I_67pt5_e_all = []

I_0_o_err_all = []
I_0_e_err_all = []
I_22pt5_o_err_all = []
I_22pt5_e_err_all = []
I_45_o_err_all = []
I_45_e_err_all = []
I_67pt5_o_err_all = []
I_67pt5_e_err_all = []

# Loop over each order file
for order_num in orders_num:
    # Temporary containers for current file
    I_0_o = []
    I_0_e = []
    I_22pt5_o = []
    I_22pt5_e = []
    I_45_o = []
    I_45_e = []
    I_67pt5_o = []
    I_67pt5_e = []

    I_0_o_err = []
    I_0_e_err = []
    I_22pt5_o_err = []
    I_22pt5_e_err = []
    I_45_o_err = []
    I_45_e_err = []
    I_67pt5_o_err = []
    I_67pt5_e_err = []

    # File path pattern
    

    Intensity_file_path = f"{project_root}/output/{observation_session}/{star_name}"


    if grating_choice == 1:
        Intensity_file_path = Intensity_file_path + '/RedCD/'
    elif grating_choice == 2:
        Intensity_file_path = Intensity_file_path + '/BlueCD/'


    
    Intensity_file_path = Intensity_file_path + set_number 

    filename = f"{star_name}_{set_number}_Order-{order_num}_IntensityBeforeEffCorr.txt"
    file_path = f"{Intensity_file_path}/{filename}"
    # ~ print(file_path)
    # ~ file_path = f"{project_root}/output/2024-2025/YGem_2025-01-24/RedCD/Set_1/YGem_2025-01-24_Set_1_Order-{order_num}_IntensityBeforeEffCorr.txt"
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.strip():  # skip empty lines
                    parts = line.strip().split()
                    I_0_o.append(float(parts[1]))
                    I_0_e.append(float(parts[2]))
                    I_22pt5_o.append(float(parts[3]))
                    I_22pt5_e.append(float(parts[4]))
                    I_45_o.append(float(parts[5]))
                    I_45_e.append(float(parts[6]))
                    I_67pt5_o.append(float(parts[7]))
                    I_67pt5_e.append(float(parts[8]))
                    I_0_o_err.append(float(parts[9]))
                    I_0_e_err.append(float(parts[10]))
                    I_22pt5_o_err.append(float(parts[11]))
                    I_22pt5_e_err.append(float(parts[12]))
                    I_45_o_err.append(float(parts[13]))
                    I_45_e_err.append(float(parts[14]))
                    I_67pt5_o_err.append(float(parts[15]))
                    I_67pt5_e_err.append(float(parts[16]))
    except FileNotFoundError:
        print(f"File not found for order {order_num}")
        continue

    # Append data from this order to global lists
    I_0_o_all.append(I_0_o)
    I_0_e_all.append(I_0_e)
    I_22pt5_o_all.append(I_22pt5_o)
    I_22pt5_e_all.append(I_22pt5_e)
    I_45_o_all.append(I_45_o)
    I_45_e_all.append(I_45_e)
    I_67pt5_o_all.append(I_67pt5_o)
    I_67pt5_e_all.append(I_67pt5_e)

    I_0_o_err_all.append(I_0_o_err)
    I_0_e_err_all.append(I_0_e_err)
    I_22pt5_o_err_all.append(I_22pt5_o_err)
    I_22pt5_e_err_all.append(I_22pt5_e_err)
    I_45_o_err_all.append(I_45_o_err)
    I_45_e_err_all.append(I_45_e_err)
    I_67pt5_o_err_all.append(I_67pt5_o_err)
    I_67pt5_e_err_all.append(I_67pt5_e_err)



I_BeforeEffCorr = []
for i in range (int(len(orders_41)/2)):
    
    I_BeforeEffCorr_order = []
    I_BeforeEffCorr_order.append(I_0_o_all[i])
    I_BeforeEffCorr_order.append(I_0_e_all[i])
    I_BeforeEffCorr_order.append(I_22pt5_o_all[i])
    I_BeforeEffCorr_order.append(I_22pt5_e_all[i])
    I_BeforeEffCorr_order.append(I_45_o_all[i])
    I_BeforeEffCorr_order.append(I_45_e_all[i])
    I_BeforeEffCorr_order.append(I_67pt5_o_all[i])
    I_BeforeEffCorr_order.append(I_67pt5_e_all[i])
    
    I_BeforeEffCorr_order.append(I_0_o_err_all[i])
    I_BeforeEffCorr_order.append(I_0_e_err_all[i])
    I_BeforeEffCorr_order.append(I_22pt5_o_err_all[i])
    I_BeforeEffCorr_order.append(I_22pt5_e_err_all[i])
    I_BeforeEffCorr_order.append(I_45_o_err_all[i])
    I_BeforeEffCorr_order.append(I_45_e_err_all[i])
    I_BeforeEffCorr_order.append(I_67pt5_o_err_all[i])
    I_BeforeEffCorr_order.append(I_67pt5_e_err_all[i])
    
    I_BeforeEffCorr.append(I_BeforeEffCorr_order)





##########   Input path to UAr Calib frames   ##########################


calib_UAr_data_cube_path_1 = f'{project_root}/intermediate/calib/Calib-UAr-20s_HD61421_RedCD-FilterIn_20s_HWPPosition-67pt5_EncVal-4725pt18_4.fits'   # Not Bias substracted frame


if NumberOfCalibFrames == 1:
    calib_UAr_data_cube_1 = fits.getdata(calib_UAr_data_cube_path_1)
elif NumberOfCalibFrames == 2:
    calib_UAr_data_cube_1 = fits.getdata(calib_UAr_data_cube_path_1)
    calib_UAr_data_cube_2 = fits.getdata(calib_UAr_data_cube_path_2)
elif NumberOfCalibFrames == 4:
    calib_UAr_data_cube_1 = fits.getdata(calib_UAr_data_cube_path_1)
    calib_UAr_data_cube_2 = fits.getdata(calib_UAr_data_cube_path_2)
    calib_UAr_data_cube_3 = fits.getdata(calib_UAr_data_cube_path_3)
    calib_UAr_data_cube_4 = fits.getdata(calib_UAr_data_cube_path_4)
else:
    print("Wrong choice for number of UAr Calibration Frames")
    
    

##########   Input path to UAr Calib frames   ##########################


    
if NumberOfCalibFrames == 1:
    wave_cal_intensities, wave_cal_Stokes_parameters = wavelength_calibration1(calib_UAr_data_cube_1, orders_41, I_BeforeEffCorr, xcor_44, ycor_44, sigma_FWHM, grating_choice)
elif NumberOfCalibFrames == 2:
    wave_cal_intensities, wave_cal_Stokes_parameters = wavelength_calibration2(calib_UAr_data_cube_1, calib_UAr_data_cube_2, orders_41, I_BeforeEffCorr, xcor_42, ycor_42, xcor_44, ycor_44, sigma_FWHM, grating_choice)
elif NumberOfCalibFrames == 4:
    wave_cal_intensities, wave_cal_Stokes_parameters = wavelength_calibration4(calib_UAr_data_cube_1, calib_UAr_data_cube_2, calib_UAr_data_cube_3, calib_UAr_data_cube_4, orders_41, I_BeforeEffCorr, xcor_41, ycor_41, xcor_42, ycor_42, xcor_43, ycor_43, xcor_44, ycor_44, sigma_FWHM, grating_choice)



orders_41_wc = list(set(orders_41))
orders_41_wc = np.array(orders_41_wc)
orders_41_wc.sort()

#orders_41_wc = np.flip(orders_41_wc)
if grating_choice == 1:
    orders_ref = np.linspace(28, 48, 21)
elif grating_choice == 2:
    orders_ref = np.linspace(44, 65, 22)

orders_ref = orders_ref.astype('int32')


###########  path is the output path of the intensity files   ################

path_wavecal = f"{project_root}/output/2024-2025/YGem_2025-01-24/RedCD/Set_1/Wavelength_Calibrated/"

if not os.path.exists(path_wavecal):
    os.mkdir(path_wavecal)
path_wavecal_stokes = f"{path_wavecal}/stocks_parameters/"
if not os.path.exists(path_wavecal_stokes):
    os.mkdir(path_wavecal_stokes)

wavelength = []
I0o = []
I0e = []
I22pt5o = []
I22pt5e = []
I45o = []
I45e = []
I67pt5o = []
I67pt5e = []



for i in range (len(wave_cal_intensities)):
    
    order_diff = int(abs(min(orders_ref) - min(orders_41)))
    #order_number = orders_41_rev[(2*i)]
    order_number = orders_41_wc[i]
    if grating_choice == 1:
        start_order = 28
    elif grating_choice == 2:
        start_order = 44
    
    if order_number >= start_order:    
        
        dt = np.dtype([('wavelength', 'd'), ('I_0_o', 'd'), ('I_0_e', 'd'), ('I_22pt5_o', 'd'), ('I_22pt5_e', 'd'), ('I_45_o', 'd'), ('I_45_e', 'd'), ('I_67pt5_o', 'd'), ('I_67pt5_e', 'd'), ('I_0_o_err', 'd'), ('I_0_e_err', 'd'), ('I_22pt5_o_err', 'd'), ('I_22pt5_e_err', 'd'), ('I_45_o_err', 'd'), ('I_45_e_err', 'd'), ('I_67pt5_o_err', 'd'), ('I_67pt5_e_err', 'd')])  
        a = np.zeros(detector_pixels, dt)                        # Saving wavelength and the corresponding
        a['wavelength'] = wave_cal_intensities[i][0]           
        a['I_0_o'] = wave_cal_intensities[i][1]
        a['I_0_e'] = wave_cal_intensities[i][2]    
        a['I_22pt5_o'] = wave_cal_intensities[i][3]
        a['I_22pt5_e'] = wave_cal_intensities[i][4]
        a['I_45_o'] = wave_cal_intensities[i][5]
        a['I_45_e'] = wave_cal_intensities[i][6]    
        a['I_67pt5_o'] = wave_cal_intensities[i][7]
        a['I_67pt5_e'] = wave_cal_intensities[i][8]
        a['I_0_o_err'] = wave_cal_intensities[i][9]
        a['I_0_e_err'] = wave_cal_intensities[i][10]
        a['I_22pt5_o_err'] = wave_cal_intensities[i][11]
        a['I_22pt5_e_err'] = wave_cal_intensities[i][12]
        a['I_45_o_err'] = wave_cal_intensities[i][13]
        a['I_45_e_err'] = wave_cal_intensities[i][14]
        a['I_67pt5_o_err'] = wave_cal_intensities[i][15]
        a['I_67pt5_e_err'] = wave_cal_intensities[i][16]
        #np.savetxt('C:\\Users\\Mudit Shrivastav\\.ipython\\Science_spectra\\BetUMa\\BetUMa_9_IntensityTestBeforeEffCorr.txt', a, '%.5f', delimiter = ',')
        np.savetxt(path_wavecal + star_name + '_' + set_number + '_Order-' + str(order_number) + '_WavelengthCalibrated_Intensity.txt', a, '%.10f', delimiter = '    ')

    wavelength.append(wave_cal_intensities[i][0])
    I0o.append(wave_cal_intensities[i][1])
    I0e.append(wave_cal_intensities[i][2])
    I22pt5o.append(wave_cal_intensities[i][3])
    I22pt5e.append(wave_cal_intensities[i][4])
    I45o.append(wave_cal_intensities[i][5])
    I45e.append(wave_cal_intensities[i][6])
    I67pt5o.append(wave_cal_intensities[i][7])
    I67pt5e.append(wave_cal_intensities[i][8])

wavelength = np.array(wavelength)
I0o = np.array(I0o)
I0e = np.array(I0e)
I22pt5o = np.array(I22pt5o)
I22pt5e = np.array(I22pt5e)
I45o = np.array(I45o)
I45e = np.array(I45e)
I67pt5o = np.array(I67pt5o)
I67pt5e = np.array(I67pt5e)

hdu = fits.PrimaryHDU(wavelength)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_wavelength.fits', overwrite=True)
hdu = fits.PrimaryHDU(I0o)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I0o.fits', overwrite=True)
hdu = fits.PrimaryHDU(I0e)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I0e.fits', overwrite=True)
hdu = fits.PrimaryHDU(I22pt5o)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I22pt5o.fits', overwrite=True)
hdu = fits.PrimaryHDU(I22pt5e)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I22pt5e.fits', overwrite=True)
hdu = fits.PrimaryHDU(I45o)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I45o.fits', overwrite=True)
hdu = fits.PrimaryHDU(I45e)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I45e.fits', overwrite=True)
hdu = fits.PrimaryHDU(I67pt5o)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I67pt5o.fits', overwrite=True)
hdu = fits.PrimaryHDU(I67pt5e)
hdu.writeto(path_wavecal + star_name + '_' + set_number + '_I67pt5e.fits', overwrite=True)    
    

for i in range (len(wave_cal_Stokes_parameters)):
    
    order_diff = int(abs(min(orders_ref) - min(orders_41)))
    #order_number = orders_41_rev[(2*i)]
    order_number = orders_41_wc[i]
    if grating_choice == 1:
        start_order = 28
    elif grating_choice == 2:
        start_order = 44
    
    if order_number >= start_order:
        
        dt = np.dtype([('wavelength', 'd'), ('I0_o_I0_e', 'd'), ('I45_o_I45_e', 'd'), ('I22pt5_o_I22pt5_e', 'd'), ('I67pt5_o_I67pt5_e', 'd'), ('Aq', 'd'), ('Au', 'd'), ('q', 'd'), ('u', 'd'), ('p', 'd'), ('theta', 'd'), ('q_err', 'd'), ('u_err', 'd'), ('p_err', 'd'), ('theta_err', 'd')])  
        a = np.zeros(detector_pixels, dt)                       
        a['wavelength'] = wave_cal_Stokes_parameters[i][0]
        a['I0_o_I0_e'] = wave_cal_Stokes_parameters[i][1]
        a['I45_o_I45_e'] = wave_cal_Stokes_parameters[i][2]
        a['I22pt5_o_I22pt5_e'] = wave_cal_Stokes_parameters[i][3]
        a['I67pt5_o_I67pt5_e'] = wave_cal_Stokes_parameters[i][4]
        a['Aq'] = wave_cal_Stokes_parameters[i][5]
        a['Au'] = wave_cal_Stokes_parameters[i][6]
        #a['Aq_interp'] = wave_cal_Stokes_parameters[i][7]
        #a['Au_interp'] = wave_cal_Stokes_parameters[i][8]
        a['q'] = wave_cal_Stokes_parameters[i][7]
        a['u'] = wave_cal_Stokes_parameters[i][8]
        a['p'] = wave_cal_Stokes_parameters[i][9]
        a['theta'] = wave_cal_Stokes_parameters[i][10]
        a['q_err'] = wave_cal_Stokes_parameters[i][11]
        a['u_err'] = wave_cal_Stokes_parameters[i][12]
        a['p_err'] = wave_cal_Stokes_parameters[i][13]
        a['theta_err'] = wave_cal_Stokes_parameters[i][14]
        
        #a['Aq_raw'] = wave_cal_Stokes_parameters[i][17]
        #a['Au_raw'] = wave_cal_Stokes_parameters[i][18]
        #a['q_raw'] = wave_cal_Stokes_parameters[i][19]
        #a['u_raw'] = wave_cal_Stokes_parameters[i][20]
        #a['p_raw'] = wave_cal_Stokes_parameters[i][21]
        #a['theta_raw'] = wave_cal_Stokes_parameters[i][22]
        #a['q_raw_err'] = wave_cal_Stokes_parameters[i][23]
        #a['u_raw_err'] = wave_cal_Stokes_parameters[i][24]
        #a['p_raw_err'] = wave_cal_Stokes_parameters[i][25]
        #a['theta_raw_err'] = wave_cal_Stokes_parameters[i][26]
        
        
        np.savetxt(path_wavecal_stokes + star_name + '_' + set_number + '_Order-' + str(order_number) + '_WavelengthCalibrated_StokesParameters.txt', a, '%.10f', delimiter = '    ') 
print("saved")
