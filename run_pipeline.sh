#!/bin/bash

# Step 0: Create intermediate and output directories if they don't exist
echo "Creating intermediate and output directories..."
mkdir -p intermediate
mkdir -p output

echo "Step 1: Sorting files..."
python3 src_code/sort.py

echo "Step 2: Creating master bias..."
python3 src_code/master_bias.py

echo "Step 3: Subtracting master bias from science frames..."
python3 src_code/subtraction_science_masterbias.py

echo "Step 4: Finding central peak from science frames..."
python3 src_code/find_central_peak_science_frame_1.py

echo "Step 5: Tracing orders from science frame..."
python3 src_code/Order_Trace.py

echo "Data reduction pipeline completed successfully!"

