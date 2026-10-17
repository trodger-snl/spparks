# HDF5 Bounds Checking Feature

This document describes the new bounds checking feature added to the `app_additive_ext_temp_texture` application.

## Overview

The bounds checking feature validates that the simulation domain size is compatible with the HDF5 temperature data dimensions. This prevents runtime errors and ensures data consistency. The feature uses the `data_counts` dataset for dimensions and the optional `x0` dataset for origin information.

## Usage

Add the `bounds_check_mode` command to your input script:

```
bounds_check_mode <mode>
```

Where `<mode>` is:
- `0` = **Exact match** (default): Simulation domain must exactly match HDF5 data dimensions
- `1` = **Subvolume mode**: Simulation domain must fit within HDF5 data bounds

## Examples

### Exact Match Mode (Default)
```
bounds_check_mode 0
region box block 0 200 0 200 0 198  # Must match HDF5 dimensions exactly
```

### Subvolume Mode
```
bounds_check_mode 1
region box block 50 150 50 150 0 100  # Can be smaller than HDF5 data
```

## Validation Details

### Data Sources
- **Dimensions**: Read from the `data_counts` dataset (3D array)
- **Origin**: Read from the `x0` dataset if present (defaults to [0,0,0])

### Exact Match Mode (mode 0)
- Simulation domain size must exactly equal HDF5 data dimensions
- Error if: `sim_nx != hdf5_nx || sim_ny != hdf5_ny || sim_nz != hdf5_nz`
- Warning if simulation doesn't start at (0,0,0)

### Subvolume Mode (mode 1)
- Simulation domain must fit within HDF5 data bounds
- Accounts for HDF5 origin offset from `x0` dataset
- HDF5 effective bounds: `[x0, x0+nx] × [y0, y0+ny] × [z0, z0+nz]`
- Error if simulation domain extends outside HDF5 effective bounds

## Error Messages

The feature provides clear error messages when validation fails:

**Exact Match Error:**
```
Simulation domain size (150x150x100) does not exactly match HDF5 data size (200x200x198). 
Use bounds_check_mode 1 for subvolume mode or adjust domain size.
```

**Subvolume Error:**
```
Simulation domain [0,250] x [0,250] x [0,200] is not contained within HDF5 data bounds 
[0,200] x [0,200] x [0,198]. Adjust simulation domain to fit within HDF5 data.
```

### Example with Origin Offset

If HDF5 has `x0 = [100, 100, 0]` and dimensions `[200, 200, 198]`:
- HDF5 effective bounds: `[100,300] × [100,300] × [0,198]`
- Valid subvolume: `region box block 150 250 150 250 50 150`

## Implementation Details

- Validation occurs during HDF5 file initialization
- HDF5 dimensions are read from the `data_counts` dataset (3D)
- HDF5 origin is read from the `x0` dataset if present
- Global domain bounds are compared (not processor-specific subdomains)
- Validation runs on all processors but only rank 0 outputs messages
- Console output includes detailed bounds information for debugging

## Testing

Use the provided test input file:
```bash
cd examples/ReducedTempAM
mpirun -np 1 ../../src/spk_mac_arm < in.additive_bounds_test
```

This will validate bounds checking with your HDF5 data file.