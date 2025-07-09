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






def OrderTrace(science_frame_Fits, peaks_file_path, sigma_FWHM, plot_flag, detector_pixels, centre_column_median):
    
    #CCD_gain = 5.5
    
    # ~ starting_order = start_order

    # ~ CD_peaks = NumberOfPeaks  #  redCD_peaks = 46 for observations taken with 1.2m telescope; redCD_peaks = 42 for observations taken with 2.5m telescope   

    # Load peaks and orders from file
    data = np.loadtxt(peaks_file_path)
    orders = data[:, 0].astype(int).tolist()
    peaks = data[:, 1].astype(int).tolist()




    science_frame_41 = science_frame_Fits

    science_frame_41 = np.rot90(science_frame_41)

    hdu = fits.PrimaryHDU(science_frame_41)
    hdu.writeto(f'{output_path_dir}science_frame_Rotated.fits', overwrite=True)

    #plt.imshow(science_frame_41, cmap = "gray")
    #plt.ylim(1, 1024)
    #plt.show()

    x_new = np.linspace(0, 1023, 1024)




    ###################################################  Peak Detection  #################################################################




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
        order_peaks = []
        x = []
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


    echelle_trace_x = []
    echelle_trace_y = []

    echelle_trace_x_left = []
    echelle_trace_y_left = []
    for i in range (len(peaks)):
        peaks1 = peaks[i]
        order_peaks = []
        x = []
        median_order_drift = 0
        for j in range (int(len(science_frame_41)/2), -1, -1):
            try:
                cd_pixel = []
                cd_flux = []
                for k in range (-int(4*sigma_FWHM), int(4*sigma_FWHM)+1):
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
                
                #if i == 30 and j == int(len(science_frame_41)/2):
                #    fit_x = np.linspace(min(cd_pixel), max(cd_pixel), 100)
                #    fit = gaussian(fit_x, popt[0], popt[1], popt[2])
                #    plt.scatter(cd_pixel, cd_flux)
                #    plt.plot(fit_x, fit)
                #    plt.show()
                
            except:
                peaks1 = -1.0
                order_peaks.append(peaks1)
                x.append(j)
                continue
            
            #print(popt[1])
            peaks1 = popt[1]
            if len(order_peaks) != 0:
                if peaks1 - order_peaks[-1] > 1 or peaks1 - order_peaks[-1] < -1:
                    peaks1 = order_peaks[-1]
            
            
            #if int(len(science_frame_41)/2) - j >= 2: 
            #    if peaks1 - order_peaks[len(order_peaks)-1] > 15*median_order_drift:                
            #        peaks1 = order_peaks[len(order_peaks)-1] + median_order_drift
            #        print(str(orders[i]) + " condition triggered " + str(median_order_drift) + " " + str(peaks1))
            
            
            #print(str(j) + "  " + str(peaks1))
            #mu, sigma = norm.fit(cd)
            order_peaks.append(peaks1)
            #peaks1 = int(mu)
            x.append(j)
            
            #if int(len(science_frame_41)/2) - j >= 1:
            #    median_order_drift = (peaks1 - order_peaks[0])/(int(len(science_frame_41)/2) - j)
            #peaks1 = round(peaks1)
            peaks1 = int(peaks1)
        x = np.array(x)
        order_peaks = np.array(order_peaks)

        #order_peaks = order_peaks.astype('int32') 
        
        echelle_trace_x_left.append(x)
        echelle_trace_y_left.append(order_peaks)


    peak_coordinates_1_new_con_left = np.concatenate(echelle_trace_x_left, axis = 0)
    peak_coordinates_2_new_con_left = np.concatenate(echelle_trace_y_left, axis = 0)
    echelle_trace_x.append(peak_coordinates_1_new_con_left)
    echelle_trace_y.append(peak_coordinates_2_new_con_left)


    echelle_trace_x_right = []
    echelle_trace_y_right = []
    for i in range (len(peaks)):
        peaks1 = peaks[i]
        order_peaks = []
        x = []
        median_order_drift = 0
        for j in range (int(len(science_frame_41)/2)+1, len(science_frame_41)):
            try:
                cd_pixel = []
                cd_flux = []
                for k in range (-int(4*sigma_FWHM), int(4*sigma_FWHM)+1):
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
                peaks1 = -1.0
                order_peaks.append(peaks1)
                x.append(j)
                continue
            
            peaks1 = popt[1]
            if len(order_peaks) != 0:
                if peaks1 - order_peaks[-1] > 1 or peaks1 - order_peaks[-1] < -1:
                    peaks1 = order_peaks[-1]
            
            #if j - (int(len(science_frame_41)/2)+1) >= 2: 
            #    if order_peaks[len(order_peaks)-1] - peaks1 > 15*median_order_drift:
            #        peaks1 = order_peaks[len(order_peaks)-1] - median_order_drift
                    
            #mu, sigma = norm.fit(cd)
            order_peaks.append(peaks1)
            #mu, sigma = norm.fit(cd)
            #order_peaks.append(int(mu))
            #peaks1 = int(mu)
            x.append(j)
            
            #if j - (int(len(science_frame_41)/2)+1) >= 1:
            #    median_order_drift = (order_peaks[0] - peaks1)/(j - (int(len(science_frame_41)/2)+1))
            #peaks1 = round(peaks1)
            peaks1 = int(peaks1)
        x = np.array(x)
        order_peaks = np.array(order_peaks)

        #order_peaks = order_peaks.astype('int32')

        echelle_trace_x_right.append(x)
        echelle_trace_y_right.append(order_peaks)




    peak_coordinates_1_new_con_right = np.concatenate(echelle_trace_x_right, axis = 0)
    peak_coordinates_2_new_con_right = np.concatenate(echelle_trace_y_right, axis = 0)
    echelle_trace_x.append(peak_coordinates_1_new_con_right)
    echelle_trace_y.append(peak_coordinates_2_new_con_right)




    peak_coordinates_x_concatination = np.concatenate(echelle_trace_x, axis = 0)
    peak_coordinates_y_concatination = np.concatenate(echelle_trace_y, axis = 0)



    #plt.scatter(peak_coordinates_x_concatination, peak_coordinates_y_concatination, s=0.1)
    #plt.scatter(peak_coordinates_2_new_con_before_interpolation, peak_coordinates_1_new_con_before_interpolation, s=1)
    #plt.xlim(0,1024)
    #plt.ylim(0,1024)
    #plt.show()


    echelle_trace_x_interp1 = []
    echelle_trace_y_interp1 = []
    k = 0
    while k < len(echelle_trace_x_left):
        echelle_trace_x_interp1.append(echelle_trace_x_left[k][::-1])
        echelle_trace_x_interp1.append(echelle_trace_x_right[k])
        echelle_trace_y_interp1.append(echelle_trace_y_left[k][::-1])
        echelle_trace_y_interp1.append(echelle_trace_y_right[k])
        k = k + 1

    #print(len(echelle_trace_x_interp1))
    #print(len(echelle_trace_y_interp1))

    echelle_trace_x = []
    echelle_trace_y = []

    for i in range (0, len(echelle_trace_x_interp1), 2):
        x = []
        y = []
        
        x.append(echelle_trace_x_interp1[i])
        x.append(echelle_trace_x_interp1[i+1])
        x1 = np.concatenate(x, axis = 0)
        echelle_trace_x.append(x1)
        
        y.append(echelle_trace_y_interp1[i])
        y.append(echelle_trace_y_interp1[i+1])
        y1 = np.concatenate(y, axis = 0)
        echelle_trace_y.append(y1)  





    echelle_trace_x_interp = []         ############  Tracing the order by fitting a polynomial only to parts of the order where peaks were not detected.                                    
    echelle_trace_y_interp = []         ############  Polynomial function generated from the detected peaks. If peak was detected, then that peak value was used for trace
    #print(echelle_trace_x[0][0])
    for i in range (len(echelle_trace_x)):
        detected_peak_indices = np.where(echelle_trace_y[i] != -1)
        x1 = echelle_trace_x[i][detected_peak_indices]
        y1 = echelle_trace_y[i][detected_peak_indices]
        #l = len(x1) - 1
        order_fit_function_coefficients = np.polyfit(x1, y1, 2)
        order_fit_function = np.poly1d(order_fit_function_coefficients)
        
        x2 = []
        y2 = []
        for j in range (len(science_frame_41)):
            x2.append(j)
            is_in_array = np.isin(j, detected_peak_indices)
            if is_in_array == True:
                y2.append(echelle_trace_y[i][j])
            else:
                y2.append(order_fit_function(j))
        x2 = np.array(x2)
        y2 = np.array(y2)
        #y2 = np.round(y2)
        y2 = y2.astype('int32')
        
        echelle_trace_x_interp.append(x2)
        echelle_trace_y_interp.append(y2)


    for i in range (len(echelle_trace_y_interp)):
        for j in range (len(echelle_trace_y_interp[0])):
            if echelle_trace_y_interp[i][j]+1 < len(echelle_trace_y_interp[0]):
                if science_frame_41[echelle_trace_y_interp[i][j]][j] < science_frame_41[echelle_trace_y_interp[i][j]+1][j]:
                    echelle_trace_y_interp[i][j] = echelle_trace_y_interp[i][j] + 1


    flag = np.zeros(len(echelle_trace_y_interp))
    for i in range (len(echelle_trace_y_interp)):
        for j in range (len(echelle_trace_y_interp[i])):
            if echelle_trace_y_interp[i][j] <= 0:
                flag[i] = -1
                break
            elif echelle_trace_y_interp[i][j] >= 1023:
                flag[i] = -2
                break

    peaks = list(peaks)
    orders = list(orders)

    echelle_trace_x_interp1 = []
    echelle_trace_y_interp1 = []
    orders1 = []
    peaks1 = []

    for i in range (len(echelle_trace_y_interp)):
        if flag[i] != -1 and flag[i] != -2:
            echelle_trace_x_interp1.append(echelle_trace_x_interp[i])
            echelle_trace_y_interp1.append(echelle_trace_y_interp[i])
            peaks1.append(peaks[i])
            orders1.append(orders[i])


    echelle_trace_x_interp = echelle_trace_x_interp1
    echelle_trace_y_interp = echelle_trace_y_interp1
    peaks = peaks1
    orders = orders1


    min_order = min(orders)
    max_order = max(orders)
    extracted_orders = np.linspace(min_order, max_order, (max_order-min_order+1))
    extracted_orders = extracted_orders.astype('int32')
    order_counts = np.zeros((len(extracted_orders)))
    order_counts = order_counts.astype('int32')

    for i in range (len(order_counts)):
        order_counts[i] = orders.count(extracted_orders[i])

    orders_to_be_removed = []
    for i in range (len(order_counts)):
        if order_counts[i] != 2:
            orders_to_be_removed.append(extracted_orders[i])
        
        
    echelle_trace_x_interp1 = []
    echelle_trace_y_interp1 = []
    orders1 = []
    peaks1 = []

    for i in range (len(echelle_trace_y_interp)):
        if orders[i] not in orders_to_be_removed:
            echelle_trace_x_interp1.append(echelle_trace_x_interp[i])
            echelle_trace_y_interp1.append(echelle_trace_y_interp[i])
            peaks1.append(peaks[i])
            orders1.append(orders[i])        

    echelle_trace_x_interp = echelle_trace_x_interp1
    echelle_trace_y_interp = echelle_trace_y_interp1
    peaks = peaks1
    orders = orders1


    peaks = np.array(peaks)
    orders = np.array(orders)
    # ~ print (peaks)
    # ~ print(orders)

    peak_coordinates_x_concatination_interp = np.concatenate(echelle_trace_x_interp, axis = 0)
    peak_coordinates_y_concatination_interp = np.concatenate(echelle_trace_y_interp, axis = 0)    



    if plot_flag == 1: 
        
        plt.imshow(science_frame_41, cmap = "gray", norm = "log")

        plt.scatter(peak_coordinates_x_concatination_interp, peak_coordinates_y_concatination_interp, s=1, color = "red")
        plt.xlim(0,1024)
        plt.ylim(0,1024)
        plt.savefig(f"{output_path_dir}/Traced_spectra.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    
        
    orders = list(orders)
    peaks = list(peaks)
    echelle_trace_x_interp = list(echelle_trace_x_interp)
    echelle_trace_y_interp = list(echelle_trace_y_interp)

    return (orders, peaks, echelle_trace_x_interp, echelle_trace_y_interp)




def save_orders_to_txt(per_order_output_dir, orders, peaks, xcor, ycor):
    """
    Save each traced order to a separate text file.
    
    Parameters:
    - per_order_output_dir: directory where files will be saved
    - orders, peaks: list of order numbers and peak positions (length N)
    - xcor, ycor: list of arrays of shape (1024,) for each order (length N)
    """

    if not os.path.exists(per_order_output_dir):
        os.makedirs(per_order_output_dir)

    order_counter = {}

    for i in range(len(orders)):
        order = int(orders[i])
        peak = int(peaks[i])
        x = xcor[i]
        y = ycor[i]

        # Track occurrence (first = 'o', second = 'e')
        if order not in order_counter:
            order_counter[order] = 1
            suffix = 'o'
        else:
            order_counter[order] += 1
            suffix = 'e'

        # Filename: order_XX_o.txt or order_XX_e.txt
        filename = f"order_{order}_{suffix}.txt"
        filepath = os.path.join(per_order_output_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"# Peak: {peak}\n")
            f.write("# X\tY\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi}\t{yi}\n")

        print(f"Saved: {filepath}")





# ~ NumberOfPeaks = 41 #  redCD_peaks = 46 for observations taken with 1.2m telescope; redCD_peaks = 42 for observations taken with 2.5m telescope

# ~ starting_order = 28

# ~ CD_sigma_FWHM = 1.1

# ~ plot_flag = 1

# ~ detector_pixels = 1024

# ~ centre_column_median = 3


 
script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)




param_file = f'{project_root}/parameters.txt' 

starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median = load_parameters(param_file)

# ~ print(starting_order, NumberOfPeaks, plot_flag, CD_sigma_FWHM, detector_pixels, centre_column_median)

# ~ base_name = 'bias_sub_HD61421_RedCD-FilterIn_20s_HWPPosition-0_EncVal-0pt63_1' #############################filename


# ~ science_frame_path = f"{project_root}/intermediate/science/bias_subtracted/{base_name}.fits"

# ~ hwp_serial = extract_hwp_serial(science_frame_path) ########## serial number

# ~ science_data_cube_41 = fits.getdata(science_frame_path)
# ~ science_frame_41 = science_data_cube_41[0][:][:]

# --- Load file paths from your .txt file ---


with open(f'{project_root}/intermediate/science/bias_subtracted/bias_subtracted_paths.txt', "r") as f:
    science_frame_paths = [line.strip() for line in f if line.strip()]
    science_frame_paths = [p for p in science_frame_paths
                       if "bias_sub_HD61421" in p and "Sky" not in p and "Dark" not in p and "Calib" not in p]

for file_path in science_frame_paths:
    print(f"Processing file: {file_path}")
    hwp_serial = extract_hwp_serial(file_path)

    output_path_dir = f'{project_root}/intermediate/science/traced_orders/{hwp_serial}/'
    os.makedirs(output_path_dir, exist_ok=True)
    

    # Load FITS data
    science_data_cube_41 = fits.getdata(file_path)
    
    science_frame_41 = science_data_cube_41[0][:][:]
    base_name = file_path.split('/')[-1].replace('.fits', '')
    
    central_peak_filename = f"Detected_peak_coordinates_{base_name}_HWP_{hwp_serial}.txt"
    peaks_file_path = f"{project_root}/intermediate/science/central_peak_detection/{central_peak_filename}"

    print("processing....... it will take a few seconds.....")

    orders_41, peaks_41, xcor_41, ycor_41 = OrderTrace(
        science_frame_41,
        peaks_file_path,
        CD_sigma_FWHM,
        plot_flag,
        detector_pixels,
        centre_column_median
    )

    # Save the above lists into a number of text files labelled order wise

    
    
    save_orders_to_txt(
        per_order_output_dir=f"{output_path_dir}",
        orders=orders_41,
        peaks=peaks_41,
        xcor=xcor_41,
        ycor=ycor_41
    )


