#!/usr/bin/env python3

from astropy.io import fits
import numpy as np
import os

# --- Determine paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
masterbias_file = f'{project_root}/output/masterbias_file.fits'
files_type = ['calib', 'dark', 'science', 'sky']

# --- Load master bias ---
masterbias_data = fits.getdata(masterbias_file)

# --- Process each type of file ---
for file_type in files_type:
    input_dir = os.path.join(project_root, 'intermediate', file_type)
    output_dir = os.path.join(input_dir, 'bias_subtracted')
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(input_dir, f))]


    # Prepare output path list file
    output_list_file = os.path.join(output_dir, 'bias_subtracted_paths.txt')
    saved_paths = []

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        
        try:
            # Load the FITS data
            input_data = fits.getdata(input_path)

            # Subtract master bias
            corrected_data = input_data - masterbias_data

            # Prepare output file name
            output_filename = f'bias_sub_{input_file}'
            output_path = os.path.join(output_dir, output_filename)

            # Save the corrected FITS file
            hdu = fits.PrimaryHDU(corrected_data)
            hdu.writeto(output_path, overwrite=True)

            saved_paths.append(output_path)
            print(f"Saved: {output_path}")
        
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    # --- Save all output paths for this file type ---
    with open(output_list_file, 'w') as f:
        for path in saved_paths:
            f.write(path + '\n')

    #print(f"\nAll paths for '{file_type}' saved to: {output_list_file}")
