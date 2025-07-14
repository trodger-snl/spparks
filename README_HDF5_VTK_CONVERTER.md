# HDF5 to VTK Converter for Thermal Data

This tool converts the reduced thermal output HDF5 format from FastAM to VTK format for visualization in ParaView and other VTK-compatible viewers.

## Files

- `hdf5_to_vtk_converter.py` - Basic converter for single time points
- `hdf5_to_vtk_advanced.py` - Advanced converter with time series support

## Requirements

- Python 3.6+
- h5py
- numpy

Activate the WORK conda environment: `conda activate WORK`

## Basic Usage

### Simple Conversion
Convert HDF5 data to VTK for a single time point:

```bash
python hdf5_to_vtk_converter.py <hdf5_file> [output_dir] [time_step]
```

Example:
```bash
python hdf5_to_vtk_converter.py reduced_thermal_output.hdf5 vtk_output 50.0
```

### Advanced Conversion

The advanced converter offers more options:

```bash
python hdf5_to_vtk_advanced.py <hdf5_file> [options]
```

#### Options:
- `-o, --output DIR` - Output directory (default: vtk_output)
- `-l, --layers N1 N2 ...` - Layer indices to convert (default: 0,1,2)
- `-t, --time TIME` - Single time point to extract
- `--timeseries` - Create time series animation
- `--time-start TIME` - Start time for series (default: 0.0)
- `--time-end TIME` - End time for series (default: 100.0)
- `--time-step STEP` - Time step for series (default: 5.0)
- `--normalize` - Normalize temperatures globally

#### Examples:

Single time point:
```bash
python hdf5_to_vtk_advanced.py reduced_thermal_output.hdf5 -o results -t 75.0 -l 0 1 2
```

Time series animation:
```bash
python hdf5_to_vtk_advanced.py reduced_thermal_output.hdf5 -o animation \
    --timeseries --time-start 0 --time-end 100 --time-step 5 -l 0 1
```

## Data Structure

The HDF5 file contains:
- **Groups**: One per layer (0, 1, 2, ...)
- **layerTimes**: Global dataset with time stamps for each layer
- **Per-layer datasets**:
  - `nodeCoords`: 3D coordinates (N×3)
  - `elementToNode`: Tetrahedral connectivity (M×4)
  - `temperatures`: Temperature time series per node
  - `times`: Time stamps per node
  - `dataCounts`: Number of valid time points per node

## Output

### VTK Files
- Unstructured grid format with tetrahedral cells
- Point data includes:
  - `Temperature`: Interpolated temperature values
  - `Temperature_Normalized`: Normalized temperatures (if --normalize used)
  - `DataCount`: Number of time points available per node

### Time Series
When using `--timeseries`, the converter creates:
- Individual VTK files for each time step
- A PVD collection file (`thermal_animation.pvd`) for ParaView animation

## Visualization

### ParaView
1. Open ParaView
2. Load VTK files or PVD collection file
3. Apply "Temperature" coloring
4. For time series: use animation controls

### Recommended Workflow
1. Start with a few layers: `-l 0 1 2`
2. Use coarse time stepping for initial exploration: `--time-step 10`
3. Focus on specific time ranges of interest
4. Use `--normalize` for consistent color mapping across time steps

## Performance Notes

- Processing is memory-intensive for large datasets
- Each layer can have 40k-50k nodes and 150k+ elements
- Time series generation creates many files (plan disk space accordingly)
- Temperature interpolation is done node-by-node for accuracy

## Troubleshooting

1. **ImportError**: Ensure h5py is installed (`conda install h5py`)
2. **Out of memory**: Process fewer layers at once
3. **Missing data**: Some nodes may have no temperature data (shown as 0.0)
4. **File size**: VTK ASCII files are large; consider binary format for production use

## Example Data

The provided `reduced_thermal_output.hdf5` contains:
- 34 layers (0-33)
- Time range: 0 to ~165 seconds
- Tetrahedral meshes with temperature history
- Spatially chunked data for efficient access