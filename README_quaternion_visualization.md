# SPPARKS Quaternion Visualization Scripts

This directory contains Python scripts to visualize quaternion data from SPPARKS additive manufacturing simulations.

## Scripts

### 1. `visualize_quaternions.py` (Full Featured)
Advanced script with comprehensive orix integration, including:
- Full quaternion analysis with orix library
- Attempts CrystalMap creation with IPF coloring for Inconel 625
- Euler angle analysis
- May have compatibility issues with newer orix/NumPy versions

### 2. `visualize_quaternions_simple.py` (Recommended)
Simplified, reliable script that works across different orix versions:
- Basic quaternion analysis using orix
- Simplified IPF-like coloring based on quaternion components
- Spatial distribution maps
- More robust and less dependent on specific orix API versions

## Usage

### Basic Usage
```bash
# Activate the correct Python environment
conda activate WORK

# Run with default settings
python visualize_quaternions_simple.py examples/ReducedTempAM/QuatTest/DumpFiles/dump.additive8.33

# With sampling for large datasets (recommended)
python visualize_quaternions_simple.py examples/ReducedTempAM/QuatTest/DumpFiles/dump.additive8.33 --sample 5000

# Specify z-slice and output directory
python visualize_quaternions_simple.py examples/ReducedTempAM/QuatTest/DumpFiles/dump.additive8.33 --sample 2000 --z-slice 10 --output-dir my_plots

# Run in quiet mode (no screen output)
python visualize_quaternions_simple.py examples/ReducedTempAM/QuatTest/DumpFiles/dump.additive8.33 --sample 5000 --quiet
```

### Command Line Options
- `--sample N`: Sample N random points from the dataset (recommended for large files)
- `--z-slice Z`: Z-coordinate for 2D slice visualization (default: 5.0)
- `--output-dir DIR`: Directory to save plots (default: plots_simple)
- `--quiet`: Suppress all output to screen (silent mode)

## Generated Plots

The scripts generate several types of visualizations:

1. **Quaternion Distributions** (`quaternion_distributions_simple.png`)
   - Histograms of each quaternion component (d1, d2, d3, d4)

2. **Spatial Quaternion Maps** (`spatial_quaternion_maps_simple.png`)
   - 2D maps showing quaternion magnitude and components across the XY plane

3. **Simplified IPF Map** (`simplified_ipf_map.png`)
   - 2D slice with simplified IPF-like coloring based on quaternion orientation
   - Uses quaternion components to create RGB colors representing grain orientations

4. **IPF Component Analysis** (`simplified_ipf_components.png`)
   - Histograms of RGB components used in the simplified IPF visualization

5. **Inverse Pole Figure Density** (`inverse_pole_figure_density.png`)
   - Pole density plots or scatter plots showing orientation distributions
   - Displays IPF projections for X, Y, and Z sample directions
   - Uses stereographic projection with density visualization when available

## Data Format

The scripts expect SPPARKS dump files with the following format:
- Header with timestep, number of atoms, box bounds
- Data columns including: `id i1 i2 d1 d2 d3 d4 d6 d8 x y z`
- Quaternion data in columns d1, d2, d3, d4 (w, x, y, z components)
- Supports both plain text and gzipped (.gz) dump files

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- orix

Install via conda:
```bash
conda install pandas numpy matplotlib
conda install -c conda-forge orix
```

## Notes

- The "simplified IPF" coloring is not true crystallographic IPF coloring but provides a visualization of grain orientation variations
- For true IPF coloring with proper crystal symmetry, the full script attempts to use orix CrystalMap functionality
- Sampling is recommended for datasets with >50,000 points for better performance
- All quaternions in the SPPARKS data appear to be normalized (magnitude = 1.0)

## Troubleshooting

If you encounter orix compatibility issues:
1. Use the simplified script (`visualize_quaternions_simple.py`)
2. Try sampling a smaller dataset first
3. Check that the WORK conda environment has the correct package versions

## Example Output

Successfully tested with:
- Dataset: 619,755 atoms from `dump.additive8.33`
- Sampled to 5,000 points for visualization
- Generated 4 different plot types showing quaternion distributions and spatial orientation maps