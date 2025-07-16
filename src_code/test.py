import os

def load_parameters(param_file):
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # skip comments and blank lines
            if '=' in line:
                key, val = line.split('=')
                key = key.strip()
                val = val.strip()

                # Type inference
                if val.lower() in ['true', 'false']:
                    val = val.lower() == 'true'
                else:
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass  # keep as string if not a number

                params[key] = val

    return params

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)


param_file = f'{project_root}/parameters.txt' 

params = load_parameters(param_file)
starting_order = params['starting_order']
NumberOfPeaks = params['number_of_peaks']
plot_flag = params['plot_flag']
CD_sigma_FWHM = params['sigma_FWHM']
detector_pixels = params['detector_pixels']
centre_column_median = params['centre_column_median']
