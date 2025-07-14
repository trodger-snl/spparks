# 3D Polycrystalline Microstructure Visualization for Inconel 625
# Using orix for IPF coloring and PyVista for volume rendering
# Configured for gamma phase (FCC, Fm-3m space group)

import numpy as np
import pandas as pd
import pyvista as pv
from orix import plot
from orix.quaternion import Orientation
from orix.crystal_map import Phase
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# Set PyVista to use notebook plotting
pv.set_jupyter_backend('trame')

def read_lammps_dump(filename):
    """
    Read LAMMPS dump file and extract relevant data
    """
    print(f"Reading {filename}...")
    
    # Read the file and parse header information
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the data section
    atom_line_idx = None
    for i, line in enumerate(lines):
        if 'ITEM: ATOMS' in line:
            atom_line_idx = i
            break
    
    if atom_line_idx is None:
        raise ValueError("Could not find ATOMS section in file")
    
    # Extract column names from ATOMS line
    column_line = lines[atom_line_idx].strip()
    columns = column_line.split()[2:]  # Skip "ITEM:" and "ATOMS"
    
    # Read the data starting from the line after ATOMS
    data = []
    for line in lines[atom_line_idx + 1:]:
        if line.strip():  # Skip empty lines
            values = line.strip().split()
            data.append([float(v) if v.replace('.', '').replace('-', '').isdigit() else v for v in values])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert appropriate columns to numeric
    numeric_columns = ['id', 'i1', 'i2', 'd1', 'd2', 'd3', 'd4', 'd6', 'd8', 'x', 'y', 'z']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    
    print(f"Loaded {len(df)} atoms")
    print(f"Unique grains: {df['i1'].nunique()}")
    print(f"Data range: x=[{df['x'].min():.1f}, {df['x'].max():.1f}], "
          f"y=[{df['y'].min():.1f}, {df['y'].max():.1f}], "
          f"z=[{df['z'].min():.1f}, {df['z'].max():.1f}]")
    
    return df

def create_voxel_grid(df, voxel_size=2.0):
    """
    Convert point data to voxel grid for volume rendering
    """
    print(f"Creating voxel grid with size {voxel_size}...")
    
    # Get bounds
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()
    
    # Create grid dimensions
    nx = int(np.ceil((x_max - x_min) / voxel_size))
    ny = int(np.ceil((y_max - y_min) / voxel_size))
    nz = int(np.ceil((z_max - z_min) / voxel_size))
    
    print(f"Grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels")
    
    # Initialize arrays
    grain_ids = np.zeros((nx, ny, nz), dtype=int)
    quaternions = np.zeros((nx, ny, nz, 4))
    
    # Convert coordinates to grid indices
    x_indices = ((df['x'] - x_min) / voxel_size).astype(int)
    y_indices = ((df['y'] - y_min) / voxel_size).astype(int)
    z_indices = ((df['z'] - z_min) / voxel_size).astype(int)
    
    # Clip indices to valid range
    x_indices = np.clip(x_indices, 0, nx-1)
    y_indices = np.clip(y_indices, 0, ny-1)
    z_indices = np.clip(z_indices, 0, nz-1)
    
    # Fill voxels (taking the last atom that falls in each voxel)
    for i in range(len(df)):
        xi, yi, zi = x_indices[i], y_indices[i], z_indices[i]
        grain_ids[xi, yi, zi] = df.iloc[i]['i1']
        quaternions[xi, yi, zi] = [df.iloc[i]['d1'], df.iloc[i]['d2'], 
                                   df.iloc[i]['d3'], df.iloc[i]['d4']]
    
    # Create coordinate arrays
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)
    
    return {
        'grain_ids': grain_ids,
        'quaternions': quaternions,
        'coordinates': (x_coords, y_coords, z_coords),
        'dimensions': (nx, ny, nz),
        'voxel_size': voxel_size
    }

def compute_ipf_colors(quaternions, grain_ids, crystal_symmetry='m-3m'):
    """
    Compute IPF colors using orix for Inconel 625 gamma phase (FCC structure)
    """
    print("Computing IPF colors for Inconel 625 gamma phase (FCC, Fm-3m)...")
    
    # Flatten arrays for processing
    flat_quats = quaternions.reshape(-1, 4)
    flat_grains = grain_ids.flatten()
    
    # Find non-zero grain voxels
    valid_mask = flat_grains > 0
    valid_quats = flat_quats[valid_mask]
    
    if len(valid_quats) == 0:
        print("Warning: No valid quaternions found!")
        return np.zeros_like(quaternions[..., :3])
    
    print(f"Processing {len(valid_quats)} valid voxels...")
    
    # Normalize quaternions
    norms = np.linalg.norm(valid_quats, axis=1)
    valid_quats = valid_quats / norms[:, np.newaxis]
    
    # Create orix Orientation objects
    try:
        orientations = Orientation.from_quaternion(valid_quats)
        
        # Set crystal symmetry for Inconel 625 gamma phase (FCC)
        # Fm-3m space group corresponds to m-3m point group (cubic Oh symmetry)
        phase = Phase(point_group=crystal_symmetry, name='gamma-Inconel625')
        orientations.symmetry = phase.point_group
        
        # Create IPF color key (Z-direction for build/loading direction)
        # This is common for additive manufacturing or mechanical testing
        ipf_key = plot.IPFColorKeyTSL(phase.point_group, direction=[0, 0, 1])
        
        # Compute colors
        rgb_colors = ipf_key.orientation2color(orientations)
        
        print(f"Generated IPF colors for {len(rgb_colors)} orientations")
        print(f"Crystal symmetry: {crystal_symmetry} (FCC gamma phase)")
        
    except Exception as e:
        print(f"Error in orix processing: {e}")
        # Fallback: use random colors
        print("Using fallback random colors...")
        rgb_colors = np.random.rand(len(valid_quats), 3)
    
    # Create full color array
    full_colors = np.zeros((len(flat_grains), 3))
    full_colors[valid_mask] = rgb_colors
    
    # Reshape back to grid
    grid_colors = full_colors.reshape(quaternions.shape[:-1] + (3,))
    
    return grid_colors

def create_pyvista_volume(voxel_data, colors):
    """
    Create PyVista volume for rendering
    """
    print("Creating PyVista volume...")
    
    grain_ids = voxel_data['grain_ids']
    nx, ny, nz = voxel_data['dimensions']
    
    # Create PyVista ImageData
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    
    # Set spacing based on voxel size
    grid.spacing = (voxel_data['voxel_size'], voxel_data['voxel_size'], voxel_data['voxel_size'])
    
    # Set origin
    x_coords, y_coords, z_coords = voxel_data['coordinates']
    grid.origin = (x_coords[0], y_coords[0], z_coords[0])
    
    # Add data arrays
    # Grain IDs as scalars
    grid['GrainID'] = grain_ids.flatten(order='F')  # PyVista uses Fortran ordering
    
    # RGB colors (convert to uint8 for better performance)
    rgb_uint8 = (colors * 255).astype(np.uint8)
    grid['IPF_Colors'] = rgb_uint8.reshape(-1, 3, order='F')
    
    # Create a scalar field for better volume rendering
    # Use grain boundary distance or grain ID mapping
    scalar_field = np.where(grain_ids > 0, grain_ids, 0).astype(float)
    grid['ScalarField'] = scalar_field.flatten(order='F')
    
    print(f"Grid info: {grid}")
    return grid

def visualize_microstructure(grid, method='volume'):
    """
    Visualize the microstructure using different methods
    """
    print(f"Creating visualization using {method} method...")
    
    plotter = pv.Plotter(notebook=True)
    
    if method == 'volume':
        # Volume rendering
        # Create a subset for better performance if the dataset is too large
        if grid.n_points > 1000000:  # If more than 1M points, subsample
            print("Large dataset detected, creating subset for visualization...")
            subset = grid.extract_subset([0, grid.dimensions[0]//2, 
                                        0, grid.dimensions[1]//2, 
                                        0, grid.dimensions[2]//2])
            grid_to_plot = subset
        else:
            grid_to_plot = grid
        
        # Use scalar field for volume rendering with custom opacity
        opacity = [0, 0.1, 0.3, 0.6, 0.8]
        plotter.add_volume(grid_to_plot, 
                          scalars='ScalarField',
                          opacity=opacity,
                          cmap='viridis',
                          show_scalar_bar=True)
        
    elif method == 'surface':
        # Extract surfaces of grains
        # Threshold to get non-zero grains
        thresholded = grid.threshold(0.5, scalars='GrainID')
        
        if thresholded.n_points > 0:
            plotter.add_mesh(thresholded, 
                           scalars='IPF_Colors',
                           rgb=True,
                           show_edges=False,
                           opacity=0.8)
        
    elif method == 'slices':
        # Create orthogonal slices
        slices = grid.slice_orthogonal()
        plotter.add_mesh(slices, 
                        scalars='IPF_Colors',
                        rgb=True,
                        show_edges=False)
    
    # Set camera and show
    plotter.camera_position = 'iso'
    plotter.show_grid()
    plotter.add_axes()
    
    return plotter

# Main execution
def main():
    """
    Main function to process data and create visualization
    """
    # Read data
    filename = '/Users/Tron/spparks/examples/ReducedTempAM/MisOrientTest/DumpFiles/dump.additive8.101'  # Update this path as needed
    df = read_lammps_dump(filename)
    
    # Create voxel grid (adjust voxel_size for performance vs resolution)
    voxel_size = 5.0  # Increase for better performance, decrease for higher resolution
    voxel_data = create_voxel_grid(df, voxel_size=voxel_size)
    
    # Compute IPF colors for Inconel 625 gamma phase
    colors = compute_ipf_colors(voxel_data['quaternions'], 
                               voxel_data['grain_ids'],
                               crystal_symmetry='m-3m')  # FCC gamma phase (Fm-3m space group)
    
    # Create PyVista volume
    grid = create_pyvista_volume(voxel_data, colors)
    
    # Create visualizations
    print("\n" + "="*50)
    print("Creating Volume Rendering...")
    plotter_volume = visualize_microstructure(grid, method='volume')
    
    print("\n" + "="*50)
    print("Creating Surface Rendering...")
    plotter_surface = visualize_microstructure(grid, method='surface')
    
    print("\n" + "="*50)
    print("Creating Slice View...")
    plotter_slices = visualize_microstructure(grid, method='slices')
    
    return df, voxel_data, colors, grid

# Run the analysis
if __name__ == "__main__":
    df, voxel_data, colors, grid = main()
    
    # Print summary statistics
    print("="*50)
    print("INCONEL 625 GAMMA PHASE MICROSTRUCTURE ANALYSIS")
    print("="*50)
    print(f"Crystal Structure: Face-Centered Cubic (FCC)")
    print(f"Space Group: Fm-3m")
    print(f"Point Group: m-3m (Oh symmetry)")
    print(f"Original atoms: {len(df):,}")
    print(f"Unique grains: {df['i1'].nunique()}")
    print(f"Voxel grid: {voxel_data['dimensions']}")
    print(f"Non-empty voxels: {np.sum(voxel_data['grain_ids'] > 0):,}")
    print(f"Voxel size: {voxel_data['voxel_size']}")
    print(f"IPF Direction: [001] (Z-axis)")
    
    # Show data sample
    print(f"\nSample data:")
    print(df[['id', 'i1', 'd1', 'd2', 'd3', 'd4', 'x', 'y', 'z']].head())
