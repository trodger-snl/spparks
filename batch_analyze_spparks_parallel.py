#!/usr/bin/env python3
"""
Parallel SPPARKS batch analysis with subprocess-based rendering
This version uses subprocess calls to avoid PyVista threading issues on macOS
"""

import numpy as np
import pandas as pd
import glob
import os
import time
import cv2
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from tqdm import tqdm
import warnings
import subprocess
import sys
import tempfile
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/Users/Tron/spparks/examples/ReducedTempAM/MisOrientTest/DumpFiles/"
FILE_PATTERN = "dump.additive8.*"
OUTPUT_DIR = "/Users/Tron/spparks/batch_analysis_output"
IPF_DIRECTION = 'z'
CRYSTAL_STRUCTURE = 'fcc'
OPACITY_FILLED = 1.0
BACKGROUND_COLOR = 'white'
VOXEL_SIZE = 1.0
IMAGE_SIZE = (1920, 1080)
FPS = 5

# Method selection flags
ENABLE_METHOD1 = False   # Point cloud with glyphs
ENABLE_METHOD2 = True   # Structured grid

# Output options
ENABLE_VTK_OUTPUT = False # Generate VTK file with all timesteps and IPF data
ENABLE_VTKHDF_OUTPUT = False # Generate VTK-HDF files (modern high-performance format)

# Parallelization settings
PARALLEL_FILES = True    # Process multiple files in parallel (subprocess-based)
MAX_WORKERS = 4          # Maximum number of parallel workers
MEMORY_LIMIT_GB = 16     # Memory limit for adaptive worker scaling

# Subvolume cropping
CROP_BOUNDS = [0,200,0,200,0,10]

def parse_spparks_dump(filename):
    """Parse SPPARKS dump file and extract simulation data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {}
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if "ITEM: TIMESTEP" in line:
            data['timestep'] = float(lines[i+1].strip().split()[0])
            data['time'] = float(lines[i+1].strip().split()[1]) if len(lines[i+1].strip().split()) > 1 else None
            i += 2
            
        elif "ITEM: NUMBER OF ATOMS" in line:
            data['n_atoms'] = int(lines[i+1].strip())
            i += 2
            
        elif "ITEM: BOX BOUNDS" in line:
            bounds = []
            for j in range(3):
                bound_line = lines[i+1+j].strip().split()
                bounds.append([int(bound_line[0]), int(bound_line[1])])
            data['box_bounds'] = bounds
            i += 4
            
        elif "ITEM: ATOMS" in line:
            headers = line.split()[2:]
            atom_data = []
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    values = lines[j].strip().split()
                    atom_data.append([float(v) if '.' in v or 'e' in v.lower() else int(v) for v in values])
            data['atoms'] = pd.DataFrame(atom_data, columns=headers)
            break
        else:
            i += 1
    
    return data

def apply_subvolume_crop(coords, colors, grain_ids, quaternions, crop_bounds):
    """Crop data to a subvolume."""
    if crop_bounds is None:
        return coords, colors, grain_ids, quaternions
    
    x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds
    
    # Create mask for points within crop bounds
    mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
        (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
    )
    
    # Apply mask
    cropped_coords = coords[mask]
    cropped_colors = colors[mask]
    cropped_grain_ids = grain_ids[mask]
    cropped_quaternions = quaternions[mask]
    
    # Shift coordinates to start from origin
    if len(cropped_coords) > 0:
        cropped_coords[:, 0] -= x_min
        cropped_coords[:, 1] -= y_min
        cropped_coords[:, 2] -= z_min
    
    return cropped_coords, cropped_colors, cropped_grain_ids, cropped_quaternions

def generate_ipf_colors(quaternions, crystal_structure='fcc', direction='z'):
    """Generate IPF colors from quaternion orientations."""
    from orix.crystal_map import Phase
    from orix.quaternion import Orientation, Quaternion
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    
    # Create phase object
    if crystal_structure.lower() == 'fcc':
        phase = Phase('Gamma', point_group='m-3m')
    else:
        raise ValueError(f"Unsupported crystal structure: {crystal_structure}")
    
    # Create orientations
    quat_obj = Quaternion(quaternions)
    orientations = Orientation(quat_obj)
    orientations.phase = phase
    
    # Setup IPF direction
    if direction == 'z':
        ipf_dir = Vector3d([0, 0, 1])
    else:
        ipf_dir = Vector3d([0, 0, 1])
    
    # Create IPF color key
    ipf_key = IPFColorKeyTSL(phase.point_group, direction=ipf_dir)
    
    # Generate colors
    rgb_colors = ipf_key.orientation2color(orientations)
    
    return rgb_colors

def create_ipf_color_key(crystal_structure='fcc', direction='z', output_path=None):
    """Create IPF color key legend using orix."""
    from orix.crystal_map import Phase
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    
    # Create phase object
    if crystal_structure.lower() == 'fcc':
        phase = Phase('Gamma', point_group='m-3m')
    else:
        raise ValueError(f"Unsupported crystal structure: {crystal_structure}")
    
    # Setup IPF direction
    if direction == 'z':
        ipf_dir = Vector3d([0, 0, 1])
        direction_label = 'Z'
    elif direction == 'y':
        ipf_dir = Vector3d([0, 1, 0])
        direction_label = 'Y'
    elif direction == 'x':
        ipf_dir = Vector3d([1, 0, 0])
        direction_label = 'X'
    else:
        ipf_dir = Vector3d([0, 0, 1])
        direction_label = 'Z'
    
    # Create IPF color key
    ipf_key = IPFColorKeyTSL(phase.point_group, direction=ipf_dir)
    
    # Create the plot
    fig = ipf_key.plot(return_figure=True)
    
    # Customize the plot
    fig.suptitle(f'IPF Color Key - {direction_label} Direction\n{crystal_structure.upper()} Crystal Structure', 
                 fontsize=12, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return output_path
    
    return fig

def create_ipf_legend_image(crystal_structure='fcc', direction='z', size=(300, 300)):
    """Create IPF color key as numpy array for embedding in PyVista plots."""
    import io
    from PIL import Image
    
    # Create temporary file for the IPF color key
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        legend_path = tmp_file.name
    
    try:
        # Create IPF color key
        create_ipf_color_key(crystal_structure, direction, legend_path)
        
        # Load image and convert to numpy array
        with Image.open(legend_path) as img:
            # Resize to desired size
            img = img.resize(size, Image.Resampling.LANCZOS)
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # Convert to numpy array
            legend_array = np.array(img)
        
        return legend_array
        
    finally:
        # Clean up temporary file
        if os.path.exists(legend_path):
            os.unlink(legend_path)

def determine_optimal_camera_position(largest_file):
    """Determine optimal camera position using the largest dump file."""
    print(f"Determining optimal camera position using {largest_file}...")
    
    # Load largest file
    spparks_data = parse_spparks_dump(largest_file)
    atoms = spparks_data['atoms']
    bounds = spparks_data['box_bounds']
    
    # Get grid dimensions
    nx = bounds[0][1] - bounds[0][0] + 1
    ny = bounds[1][1] - bounds[1][0] + 1
    nz = bounds[2][1] - bounds[2][0] + 1
    
    # Calculate center and optimal distance
    center_x, center_y, center_z = nx/2, ny/2, nz/2
    max_dim = max(nx, ny, nz)
    distance = max_dim * 2.5  # Distance factor for good view
    
    # Position camera at an isometric-like angle
    camera_pos = [
        center_x + distance * 0.7,
        center_y + distance * 0.7, 
        center_z + distance * 1.0
    ]
    
    focal_point = [center_x, center_y, center_z]
    view_up = [0, 0, 1]
    
    return (camera_pos, focal_point, view_up)

def create_vtk_multiblock_dataset(all_data, output_path):
    """Create a VTK multiblock dataset containing all timesteps with IPF coloring."""
    import pyvista as pv
    
    print(f"Creating VTK multiblock dataset: {output_path}")
    
    # Create multiblock dataset
    multiblock = pv.MultiBlock()
    
    # Process each timestep
    for timestep_data in tqdm(all_data, desc="Creating VTK blocks"):
        timestep = timestep_data['timestep']
        coords = timestep_data['coords']
        ipf_colors = timestep_data['ipf_colors']
        grain_ids = timestep_data['grain_ids']
        quaternions = timestep_data['quaternions']
        
        if len(coords) == 0:
            continue
            
        # Create point cloud for this timestep
        points = pv.PolyData(coords)
        
        # Add IPF colors (RGB values 0-1)
        points['IPF_Red'] = ipf_colors[:, 0]
        points['IPF_Green'] = ipf_colors[:, 1] 
        points['IPF_Blue'] = ipf_colors[:, 2]
        points['IPF_RGB'] = ipf_colors
        
        # Add grain IDs
        points['Grain_ID'] = grain_ids
        
        # Add quaternion components
        points['Quat_W'] = quaternions[:, 0]
        points['Quat_X'] = quaternions[:, 1]
        points['Quat_Y'] = quaternions[:, 2]
        points['Quat_Z'] = quaternions[:, 3]
        
        # Add timestep as point data
        points['Timestep'] = np.full(len(coords), timestep)
        
        # Add coordinate arrays for convenience
        points['X'] = coords[:, 0]
        points['Y'] = coords[:, 1]
        points['Z'] = coords[:, 2]
        
        # Add this timestep to multiblock
        multiblock[f"Timestep_{timestep}"] = points
    
    # Save multiblock dataset
    multiblock.save(output_path)
    print(f"VTK multiblock dataset saved: {output_path}")
    
    return output_path

def create_vtk_structured_grid_series(all_data, output_dir):
    """Create VTK structured grid series for each timestep."""
    import pyvista as pv
    
    print(f"Creating VTK structured grid series in: {output_dir}")
    
    # Create subdirectory for VTK series
    vtk_series_dir = os.path.join(output_dir, "vtk_series")
    os.makedirs(vtk_series_dir, exist_ok=True)
    
    # Determine grid dimensions from first non-empty timestep
    grid_shape = None
    for timestep_data in all_data:
        if len(timestep_data['coords']) > 0:
            coords = timestep_data['coords']
            nx = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            ny = int(coords[:, 1].max() - coords[:, 1].min() + 1) 
            nz = int(coords[:, 2].max() - coords[:, 2].min() + 1)
            grid_shape = (nx, ny, nz)
            break
    
    if grid_shape is None:
        print("No valid data found for VTK structured grid")
        return None
    
    nx, ny, nz = grid_shape
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Create structured grid coordinates
    x = np.arange(nx + 1, dtype=float)
    y = np.arange(ny + 1, dtype=float)
    z = np.arange(nz + 1, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    vtk_files = []
    
    # Process each timestep
    for timestep_data in tqdm(all_data, desc="Creating VTK structured grids"):
        timestep = timestep_data['timestep']
        coords = timestep_data['coords']
        ipf_colors = timestep_data['ipf_colors']
        grain_ids = timestep_data['grain_ids']
        quaternions = timestep_data['quaternions']
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Initialize cell data arrays
        total_cells = nx * ny * nz
        cell_ipf_red = np.zeros(total_cells)
        cell_ipf_green = np.zeros(total_cells)
        cell_ipf_blue = np.zeros(total_cells)
        cell_grain_ids = np.zeros(total_cells, dtype=int)
        cell_quat_w = np.zeros(total_cells)
        cell_quat_x = np.zeros(total_cells)
        cell_quat_y = np.zeros(total_cells)
        cell_quat_z = np.zeros(total_cells)
        cell_mask = np.zeros(total_cells, dtype=bool)
        
        # Fill cell data based on coordinates
        for idx, (coord, ipf_color, grain_id, quat) in enumerate(zip(coords, ipf_colors, grain_ids, quaternions)):
            i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                cell_idx = i + j * nx + k * nx * ny
                cell_ipf_red[cell_idx] = ipf_color[0]
                cell_ipf_green[cell_idx] = ipf_color[1]
                cell_ipf_blue[cell_idx] = ipf_color[2]
                cell_grain_ids[cell_idx] = grain_id
                cell_quat_w[cell_idx] = quat[0]
                cell_quat_x[cell_idx] = quat[1]
                cell_quat_y[cell_idx] = quat[2]
                cell_quat_z[cell_idx] = quat[3]
                cell_mask[cell_idx] = True
        
        # Add cell data to grid
        grid.cell_data['IPF_Red'] = cell_ipf_red
        grid.cell_data['IPF_Green'] = cell_ipf_green
        grid.cell_data['IPF_Blue'] = cell_ipf_blue
        grid.cell_data['Grain_ID'] = cell_grain_ids
        grid.cell_data['Quat_W'] = cell_quat_w
        grid.cell_data['Quat_X'] = cell_quat_x
        grid.cell_data['Quat_Y'] = cell_quat_y
        grid.cell_data['Quat_Z'] = cell_quat_z
        grid.cell_data['Filled'] = cell_mask.astype(float)
        
        # Add timestep information
        grid.field_data['Timestep'] = timestep
        
        # Save individual VTK file
        vtk_file = os.path.join(vtk_series_dir, f"spparks_timestep_{timestep:06d}.vts")
        grid.save(vtk_file)
        vtk_files.append(vtk_file)
    
    # Create PVD file for time series
    pvd_file = os.path.join(output_dir, "spparks_timeseries.pvd")
    create_pvd_timeseries(vtk_files, pvd_file)
    
    print(f"VTK structured grid series saved to: {vtk_series_dir}")
    print(f"PVD time series file: {pvd_file}")
    
    return pvd_file

def create_pvd_timeseries(vtk_files, pvd_file):
    """Create a PVD file for time series visualization in ParaView."""
    
    pvd_content = '''<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
'''
    
    for vtk_file in vtk_files:
        # Extract timestep from filename
        basename = os.path.basename(vtk_file)
        timestep = int(basename.split('_')[-1].split('.')[0])
        
        # Relative path for PVD file
        rel_path = os.path.relpath(vtk_file, os.path.dirname(pvd_file))
        
        pvd_content += f'    <DataSet timestep="{timestep}" group="" part="0" file="{rel_path}"/>\n'
    
    pvd_content += '''</Collection>
</VTKFile>
'''
    
    with open(pvd_file, 'w') as f:
        f.write(pvd_content)
    
    print(f"PVD time series file created: {pvd_file}")

def create_vtkhdf_point_cloud_series(all_data, output_dir):
    """Create VTK-HDF point cloud files for each timestep."""
    import pyvista as pv
    
    print(f"Creating VTK point cloud series (VTK-HDF format) in: {output_dir}")
    
    # Create subdirectory for VTK-HDF series
    vtkhdf_series_dir = os.path.join(output_dir, "vtp_series")
    os.makedirs(vtkhdf_series_dir, exist_ok=True)
    
    vtkhdf_files = []
    
    # Process each timestep
    for timestep_data in tqdm(all_data, desc="Creating VTK-HDF point clouds"):
        timestep = timestep_data['timestep']
        coords = timestep_data['coords']
        ipf_colors = timestep_data['ipf_colors']
        grain_ids = timestep_data['grain_ids']
        quaternions = timestep_data['quaternions']
        
        if len(coords) == 0:
            continue
            
        # Create point cloud
        points = pv.PolyData(coords)
        
        # Add IPF colors (RGB values 0-1)
        points['IPF_Red'] = ipf_colors[:, 0]
        points['IPF_Green'] = ipf_colors[:, 1] 
        points['IPF_Blue'] = ipf_colors[:, 2]
        points['IPF_RGB'] = ipf_colors
        
        # Add grain IDs
        points['Grain_ID'] = grain_ids
        
        # Add quaternion components
        points['Quat_W'] = quaternions[:, 0]
        points['Quat_X'] = quaternions[:, 1]
        points['Quat_Y'] = quaternions[:, 2]
        points['Quat_Z'] = quaternions[:, 3]
        
        # Add timestep as point data
        points['Timestep'] = np.full(len(coords), timestep)
        
        # Add coordinate arrays for convenience
        points['X'] = coords[:, 0]
        points['Y'] = coords[:, 1]
        points['Z'] = coords[:, 2]
        
        # Save as VTK-HDF file - use .hdf extension for now
        vtkhdf_file = os.path.join(vtkhdf_series_dir, f"spparks_points_timestep_{timestep:06d}.hdf")
        try:
            # Try saving as VTP (closest supported format)
            vtp_file = vtkhdf_file.replace('.hdf', '.vtp')
            points.save(vtp_file)
            vtkhdf_file = vtp_file
        except Exception as e:
            print(f"Warning: Could not save VTK-HDF file {vtkhdf_file}: {e}")
            continue
        vtkhdf_files.append(vtkhdf_file)
    
    # Create PVD file for VTK-HDF time series
    pvd_file = os.path.join(output_dir, "spparks_vtp_timeseries.pvd")
    create_pvd_timeseries(vtkhdf_files, pvd_file)
    
    print(f"VTK point cloud series saved to: {vtkhdf_series_dir}")
    print(f"VTK point cloud PVD time series file: {pvd_file}")
    
    return pvd_file

def create_vtkhdf_structured_grid_series(all_data, output_dir):
    """Create VTK structured grid files for each timestep (VTK-HDF format)."""
    import pyvista as pv
    
    print(f"Creating VTK structured grid series (VTK-HDF format) in: {output_dir}")
    
    # Create subdirectory for VTK structured grids
    vtkhdf_struct_dir = os.path.join(output_dir, "vts_series")
    os.makedirs(vtkhdf_struct_dir, exist_ok=True)
    
    # Determine grid dimensions from first non-empty timestep
    grid_shape = None
    for timestep_data in all_data:
        if len(timestep_data['coords']) > 0:
            coords = timestep_data['coords']
            nx = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            ny = int(coords[:, 1].max() - coords[:, 1].min() + 1) 
            nz = int(coords[:, 2].max() - coords[:, 2].min() + 1)
            grid_shape = (nx, ny, nz)
            break
    
    if grid_shape is None:
        print("No valid data found for VTK-HDF structured grid")
        return None
    
    nx, ny, nz = grid_shape
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Create structured grid coordinates
    x = np.arange(nx + 1, dtype=float)
    y = np.arange(ny + 1, dtype=float)
    z = np.arange(nz + 1, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    vtkhdf_files = []
    
    # Process each timestep
    for timestep_data in tqdm(all_data, desc="Creating VTK-HDF structured grids"):
        timestep = timestep_data['timestep']
        coords = timestep_data['coords']
        ipf_colors = timestep_data['ipf_colors']
        grain_ids = timestep_data['grain_ids']
        quaternions = timestep_data['quaternions']
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Initialize cell data arrays
        total_cells = nx * ny * nz
        cell_ipf_red = np.zeros(total_cells)
        cell_ipf_green = np.zeros(total_cells)
        cell_ipf_blue = np.zeros(total_cells)
        cell_grain_ids = np.zeros(total_cells, dtype=int)
        cell_quat_w = np.zeros(total_cells)
        cell_quat_x = np.zeros(total_cells)
        cell_quat_y = np.zeros(total_cells)
        cell_quat_z = np.zeros(total_cells)
        cell_mask = np.zeros(total_cells, dtype=bool)
        
        # Fill cell data based on coordinates
        for idx, (coord, ipf_color, grain_id, quat) in enumerate(zip(coords, ipf_colors, grain_ids, quaternions)):
            i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                cell_idx = i + j * nx + k * nx * ny
                cell_ipf_red[cell_idx] = ipf_color[0]
                cell_ipf_green[cell_idx] = ipf_color[1]
                cell_ipf_blue[cell_idx] = ipf_color[2]
                cell_grain_ids[cell_idx] = grain_id
                cell_quat_w[cell_idx] = quat[0]
                cell_quat_x[cell_idx] = quat[1]
                cell_quat_y[cell_idx] = quat[2]
                cell_quat_z[cell_idx] = quat[3]
                cell_mask[cell_idx] = True
        
        # Add cell data to grid
        grid.cell_data['IPF_Red'] = cell_ipf_red
        grid.cell_data['IPF_Green'] = cell_ipf_green
        grid.cell_data['IPF_Blue'] = cell_ipf_blue
        grid.cell_data['Grain_ID'] = cell_grain_ids
        grid.cell_data['Quat_W'] = cell_quat_w
        grid.cell_data['Quat_X'] = cell_quat_x
        grid.cell_data['Quat_Y'] = cell_quat_y
        grid.cell_data['Quat_Z'] = cell_quat_z
        grid.cell_data['Filled'] = cell_mask.astype(float)
        
        # Add timestep information
        grid.field_data['Timestep'] = timestep
        
        # Save as VTK-HDF file - use .vts extension for structured grids
        vtkhdf_file = os.path.join(vtkhdf_struct_dir, f"spparks_struct_timestep_{timestep:06d}.vts")
        try:
            grid.save(vtkhdf_file)
        except Exception as e:
            print(f"Warning: Could not save VTK-HDF structured grid file {vtkhdf_file}: {e}")
            continue
        vtkhdf_files.append(vtkhdf_file)
    
    # Create PVD file for VTK structured grid time series
    pvd_file = os.path.join(output_dir, "spparks_vts_timeseries.pvd")
    create_pvd_timeseries(vtkhdf_files, pvd_file)
    
    print(f"VTK structured grid series saved to: {vtkhdf_struct_dir}")
    print(f"VTK structured grid PVD time series file: {pvd_file}")
    
    return pvd_file

def create_vtkhdf_combined_dataset(all_data, output_path):
    """Create a single VTK-HDF file with all timesteps as time-varying dataset."""
    import pyvista as pv
    import h5py
    
    print(f"Creating VTK-HDF combined time-varying dataset: {output_path}")
    
    # Collect all coordinates and data
    all_coords = []
    all_ipf_colors = []
    all_grain_ids = []
    all_quaternions = []
    all_timesteps = []
    
    for timestep_data in all_data:
        if len(timestep_data['coords']) > 0:
            coords = timestep_data['coords']
            all_coords.append(coords)
            all_ipf_colors.append(timestep_data['ipf_colors'])
            all_grain_ids.append(timestep_data['grain_ids'])
            all_quaternions.append(timestep_data['quaternions'])
            all_timesteps.extend([timestep_data['timestep']] * len(coords))
    
    if not all_coords:
        print("No valid data found for VTK-HDF combined dataset")
        return None
    
    # Concatenate all data
    combined_coords = np.vstack(all_coords)
    combined_ipf_colors = np.vstack(all_ipf_colors)
    combined_grain_ids = np.concatenate(all_grain_ids)
    combined_quaternions = np.vstack(all_quaternions)
    combined_timesteps = np.array(all_timesteps)
    
    # Create point cloud with all data
    points = pv.PolyData(combined_coords)
    
    # Add all data fields
    points['IPF_Red'] = combined_ipf_colors[:, 0]
    points['IPF_Green'] = combined_ipf_colors[:, 1]
    points['IPF_Blue'] = combined_ipf_colors[:, 2]
    points['IPF_RGB'] = combined_ipf_colors
    points['Grain_ID'] = combined_grain_ids
    points['Quat_W'] = combined_quaternions[:, 0]
    points['Quat_X'] = combined_quaternions[:, 1]
    points['Quat_Y'] = combined_quaternions[:, 2]
    points['Quat_Z'] = combined_quaternions[:, 3]
    points['Timestep'] = combined_timesteps
    points['X'] = combined_coords[:, 0]
    points['Y'] = combined_coords[:, 1]
    points['Z'] = combined_coords[:, 2]
    
    # Try to save as VTK-HDF, fallback to VTP if not supported
    try:
        # For now, save as .vtp until proper VTK-HDF support is confirmed
        vtp_path = output_path.replace('.vtkhdf', '.vtp')
        points.save(vtp_path)
        print(f"VTK-HDF combined dataset saved as VTP: {vtp_path}")
        print("Note: Saved as .vtp format (VTK-HDF format may need different implementation)")
        return vtp_path
    except Exception as e:
        print(f"Failed to save VTK-HDF combined dataset: {e}")
        return None

def create_subprocess_renderer_script():
    """Create a standalone rendering script for subprocess execution."""
    script_content = '''#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pyvista as pv
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image

# Configure PyVista for subprocess
os.environ['PYVISTA_USE_OSMESA'] = 'true'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYVISTA_USE_PANEL'] = 'false'

pv.OFF_SCREEN = True

def create_ipf_color_key(crystal_structure='fcc', direction='z', output_path=None):
    """Create IPF color key legend using orix."""
    from orix.crystal_map import Phase
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    
    # Create phase object
    if crystal_structure.lower() == 'fcc':
        phase = Phase('Gamma', point_group='m-3m')
    else:
        raise ValueError(f"Unsupported crystal structure: {crystal_structure}")
    
    # Setup IPF direction
    if direction == 'z':
        ipf_dir = Vector3d([0, 0, 1])
        direction_label = 'Z'
    elif direction == 'y':
        ipf_dir = Vector3d([0, 1, 0])
        direction_label = 'Y'
    elif direction == 'x':
        ipf_dir = Vector3d([1, 0, 0])
        direction_label = 'X'
    else:
        ipf_dir = Vector3d([0, 0, 1])
        direction_label = 'Z'
    
    # Create IPF color key
    ipf_key = IPFColorKeyTSL(phase.point_group, direction=ipf_dir)
    
    # Create the plot
    fig = ipf_key.plot(return_figure=True)
    
    # Customize the plot
    fig.suptitle(f'IPF Color Key - {direction_label} Direction', 
                 fontsize=10, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return output_path
    
    return fig


def create_method2_rendering(coords, colors, grid_shape, camera_position, title, output_path, 
                           crystal_structure='fcc', direction='z'):
    """Method 2: Structured grid rendering in subprocess."""
    nx, ny, nz = grid_shape
    
    # Create structured grid coordinates
    x = np.arange(nx + 1)
    y = np.arange(ny + 1) 
    z = np.arange(nz + 1)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    
    grid = pv.StructuredGrid(x, y, z)
    
    # Create cell data arrays
    cell_colors = np.zeros((nx * ny * nz, 3))
    cell_mask = np.zeros(nx * ny * nz, dtype=bool)
    
    # Fill colors based on coordinates
    for coord, color in zip(coords, colors):
        i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            cell_idx = i + j * nx + k * nx * ny
            cell_colors[cell_idx] = color
            cell_mask[cell_idx] = True
    
    # Add data to grid
    grid.cell_data['RGB'] = cell_colors
    grid.cell_data['mask'] = cell_mask.astype(float)
    
    # Threshold to show only filled cells
    thresholded = grid.threshold(0.5, scalars='mask')
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.background_color = 'white'
    
    plotter.add_mesh(
        thresholded,
        scalars='RGB',
        rgb=True,
        opacity=1.0,
        show_edges=False,
        lighting=False
    )
    
    plotter.add_title(title, font_size=16)
    plotter.add_axes()
    
    # Set camera position
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = 'iso'
    
    
    # Save image
    plotter.screenshot(output_path)
    plotter.close()
    
    return output_path

if __name__ == "__main__":
    # Load data from pickle file
    data_file = sys.argv[1]
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract parameters
    coords = data['coords']
    colors = data['colors']
    grid_shape = data['grid_shape']
    camera_position = data['camera_position']
    title = data['title']
    output_path = data['output_path']
    crystal_structure = data.get('crystal_structure', 'fcc')
    direction = data.get('direction', 'z')
    
    try:
        result_path = create_method2_rendering(coords, colors, grid_shape, camera_position, 
                                             title, output_path, crystal_structure, direction)
        print(f"SUCCESS:{result_path}")
    except Exception as e:
        print(f"ERROR:{e}")
        sys.exit(1)
'''
    
    script_path = "/tmp/pyvista_subprocess_renderer.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def render_with_subprocess(coords, colors, grid_shape, camera_position, title, output_path, 
                          crystal_structure=CRYSTAL_STRUCTURE, direction=IPF_DIRECTION):
    """Render using subprocess to avoid threading issues."""
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        data = {
            'coords': coords,
            'colors': colors,
            'grid_shape': grid_shape,
            'camera_position': camera_position,
            'title': title,
            'output_path': output_path,
            'crystal_structure': crystal_structure,
            'direction': direction
        }
        pickle.dump(data, f)
        data_file = f.name
    
    try:
        # Get renderer script
        script_path = create_subprocess_renderer_script()
        
        # Run subprocess
        result = subprocess.run([
            sys.executable, script_path, data_file
        ], capture_output=True, text=True, timeout=60)
        
        # Clean up
        os.unlink(data_file)
        
        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            return output_path
        else:
            raise Exception(f"Subprocess rendering failed: {result.stderr}")
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(data_file):
            os.unlink(data_file)
        raise e

def process_single_file_data(args):
    """Worker function that processes a single file - serializable for multiprocessing."""
    filepath, camera_position, output_dir, enable_method1, enable_method2, crop_bounds, enable_vtk, enable_vtkhdf = args
    
    filename = os.path.basename(filepath)
    timestep = filename.split('.')[-1]
    
    # Load and process data
    start_time = time.time()
    spparks_data = parse_spparks_dump(filepath)
    atoms = spparks_data['atoms']
    bounds = spparks_data['box_bounds']
    
    # Get original grid dimensions
    nx = bounds[0][1] - bounds[0][0] + 1
    ny = bounds[1][1] - bounds[1][0] + 1
    nz = bounds[2][1] - bounds[2][0] + 1
    
    # Extract coordinates and quaternions
    coords = atoms[['x', 'y', 'z']].values.astype(float)
    grain_ids = atoms['i1'].values
    quaternions = atoms[['d1', 'd2', 'd3', 'd4']].values
    
    # Generate IPF colors
    ipf_colors = generate_ipf_colors(quaternions, CRYSTAL_STRUCTURE, IPF_DIRECTION)
    
    # Apply subvolume cropping if enabled
    if crop_bounds is not None:
        coords, ipf_colors, grain_ids, quaternions = apply_subvolume_crop(
            coords, ipf_colors, grain_ids, quaternions, crop_bounds
        )
        # Update grid dimensions for cropped volume
        if len(coords) > 0:
            nx = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            ny = int(coords[:, 1].max() - coords[:, 1].min() + 1)
            nz = int(coords[:, 2].max() - coords[:, 2].min() + 1)
        else:
            nx = ny = nz = 1
    
    load_time = time.time() - start_time
    
    # Initialize result
    result = {
        'timestep': int(timestep),
        'filename': filename,
        'n_atoms': len(coords),
        'load_time': load_time,
        'method1_time': 0.0,
        'method2_time': 0.0,
        'method1_image': None,
        'method2_image': None
    }
    
    # Store data for VTK output if enabled
    if enable_vtk or enable_vtkhdf:
        result['vtk_data'] = {
            'timestep': int(timestep),
            'coords': coords.copy(),
            'ipf_colors': ipf_colors.copy(),
            'grain_ids': grain_ids.copy(),
            'quaternions': quaternions.copy()
        }
    
    # Process rendering methods
    if len(coords) > 0:
        # Method 2: Structured grid
        if enable_method2:
            start_time = time.time()
            crop_info = f" (cropped: {len(coords):,} atoms)" if crop_bounds else ""
            title = f"SPPARKS Method 2 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()}{crop_info}"
            image_path = os.path.join(output_dir, f"method2_timestep_{timestep}.png")
            
            try:
                render_with_subprocess(coords, ipf_colors, (nx, ny, nz), camera_position, title, image_path,
                                     CRYSTAL_STRUCTURE, IPF_DIRECTION)
                result['method2_time'] = time.time() - start_time
                result['method2_image'] = image_path
            except Exception as e:
                print(f"Method 2 rendering failed for {filename}: {e}")
                result['method2_time'] = 0.0
                result['method2_image'] = None
    
    return result

def create_animation(image_paths, output_path, fps=5):
    """Create MP4 animation from image sequence."""
    print(f"Creating animation: {output_path}")
    
    if not image_paths:
        print("No images found for animation")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to video
    for image_path in image_paths:
        img = cv2.imread(image_path)
        out.write(img)
    
    out.release()
    print(f"Animation saved: {output_path}")

def main():
    """Main batch processing function."""
    print("="*80)
    print("SPPARKS BATCH ANALYSIS - PARALLEL SUBPROCESS RENDERING")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create standalone IPF color key
    ipf_key_path = os.path.join(OUTPUT_DIR, f"ipf_color_key_{IPF_DIRECTION}_{CRYSTAL_STRUCTURE}.png")
    try:
        create_ipf_color_key(CRYSTAL_STRUCTURE, IPF_DIRECTION, ipf_key_path)
        print(f"IPF color key saved: {os.path.basename(ipf_key_path)}")
    except Exception as e:
        print(f"Warning: Could not create IPF color key: {e}")
    
    # Find all dump files
    pattern = os.path.join(DATA_DIR, FILE_PATTERN)
    all_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('.')[-1]))
    
    if not all_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # For testing, process only first 3 files
    dump_files = all_files[:3]  # REMOVE THIS LINE FOR FULL PROCESSING
    
    print(f"Found {len(all_files)} dump files total")
    print(f"Processing {len(dump_files)} files for testing")
    print(f"Range: {os.path.basename(dump_files[0])} to {os.path.basename(dump_files[-1])}")
    
    # Determine camera position using largest file
    largest_file = all_files[-1]
    camera_position = determine_optimal_camera_position(largest_file)
    
    # Determine optimal worker count
    num_workers = min(MAX_WORKERS, mp.cpu_count()) if PARALLEL_FILES else 1
    
    if PARALLEL_FILES:
        print(f"Using {num_workers} parallel workers (subprocess-based)")
    else:
        print("Using serial processing")
    
    # Prepare arguments for workers
    worker_args = [
        (filepath, camera_position, OUTPUT_DIR, ENABLE_METHOD1, ENABLE_METHOD2, CROP_BOUNDS, ENABLE_VTK_OUTPUT, ENABLE_VTKHDF_OUTPUT)
        for filepath in dump_files
    ]
    
    # Process all files
    results = []
    method1_images = []
    method2_images = []
    
    total_start = time.time()
    
    if PARALLEL_FILES and num_workers > 1:
        print(f"\\nProcessing {len(dump_files)} files with {num_workers} parallel workers...")
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all jobs
                future_to_args = {executor.submit(process_single_file_data, args): args 
                                for args in worker_args}
                
                # Process completed jobs with progress bar
                with tqdm(total=len(dump_files), desc="Processing files (parallel)") as pbar:
                    for future in as_completed(future_to_args):
                        args = future_to_args[future]
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                                
                                # Collect images for animations
                                if result['method1_image']:
                                    method1_images.append(result['method1_image'])
                                if result['method2_image']:
                                    method2_images.append(result['method2_image'])
                            
                        except Exception as e:
                            print(f"Error processing {os.path.basename(args[0])}: {e}")
                        
                        pbar.update(1)
            
            print("✓ Parallel processing completed successfully")
            
        except Exception as e:
            print(f"⚠️  Parallel processing failed: {e}")
            return
    else:
        # Serial processing
        print(f"\\nProcessing {len(dump_files)} files serially...")
        
        for args in tqdm(worker_args, desc="Processing files"):
            try:
                result = process_single_file_data(args)
                results.append(result)
                
                # Collect images for animations
                if result['method1_image']:
                    method1_images.append(result['method1_image'])
                if result['method2_image']:
                    method2_images.append(result['method2_image'])
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(args[0])}: {e}")
                continue
    
    total_time = time.time() - total_start
    
    # Sort results by timestep
    try:
        results.sort(key=lambda x: int(x['timestep']))
        if method2_images:
            method2_images.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not sort results by timestep: {e}")
    
    # Create animations
    print(f"\\nCreating animations...")
    if ENABLE_METHOD2 and method2_images:
        method2_video = os.path.join(OUTPUT_DIR, "method2_animation.mp4")
        create_animation(method2_images, method2_video, FPS)
    
    # Generate VTK output if enabled
    vtk_files = []
    if ENABLE_VTK_OUTPUT or ENABLE_VTKHDF_OUTPUT:
        # Collect VTK data from all results
        vtk_data_list = []
        for result in results:
            if 'vtk_data' in result and result['vtk_data'] is not None:
                vtk_data_list.append(result['vtk_data'])
        
        if vtk_data_list:
            # Sort by timestep
            vtk_data_list.sort(key=lambda x: x['timestep'])
            
            if ENABLE_VTK_OUTPUT:
                print(f"\\nGenerating VTK output...")
                
                # Create multiblock VTK dataset (single file with all timesteps)
                multiblock_file = os.path.join(OUTPUT_DIR, "spparks_all_timesteps.vtm")
                create_vtk_multiblock_dataset(vtk_data_list, multiblock_file)
                vtk_files.append(multiblock_file)
                
                # Create structured grid time series (individual files + PVD)
                pvd_file = create_vtk_structured_grid_series(vtk_data_list, OUTPUT_DIR)
                if pvd_file:
                    vtk_files.append(pvd_file)
            
            if ENABLE_VTKHDF_OUTPUT:
                print(f"\\nGenerating additional VTK formats (point clouds and structured grids)...")
                
                # Create VTK combined dataset (single file with all data)
                combined_file = os.path.join(OUTPUT_DIR, "spparks_all_timesteps_combined.vtkhdf")
                vtkhdf_combined = create_vtkhdf_combined_dataset(vtk_data_list, combined_file)
                if vtkhdf_combined:
                    vtk_files.append(vtkhdf_combined)
                
                # Create VTK point cloud time series
                vtkhdf_points_pvd = create_vtkhdf_point_cloud_series(vtk_data_list, OUTPUT_DIR)
                if vtkhdf_points_pvd:
                    vtk_files.append(vtkhdf_points_pvd)
                
                # Create VTK structured grid time series (alternative format)
                vtkhdf_struct_pvd = create_vtkhdf_structured_grid_series(vtk_data_list, OUTPUT_DIR)
                if vtkhdf_struct_pvd:
                    vtk_files.append(vtkhdf_struct_pvd)
        else:
            print("No VTK data available for output")
    
    # Save timing results
    timing_data = pd.DataFrame(results)
    timing_file = os.path.join(OUTPUT_DIR, "timing_results.csv")
    timing_data.to_csv(timing_file, index=False)
    
    # Print summary
    print("\\n" + "="*80)
    print("PARALLEL SUBPROCESS BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total files processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per file: {total_time/len(results):.2f} seconds")
    if PARALLEL_FILES and num_workers > 1:
        estimated_serial_time = total_time * num_workers
        speedup = estimated_serial_time / total_time
        print(f"Estimated speedup: {speedup:.1f}x (vs serial processing)")
    
    print(f"\\nOutput directory: {OUTPUT_DIR}")
    if method2_images:
        print(f"  - {len(method2_images)} Method 2 images")
        print(f"  - Method 2 animation: method2_animation.mp4")
    
    # Check for IPF color key
    ipf_key_file = f"ipf_color_key_{IPF_DIRECTION}_{CRYSTAL_STRUCTURE}.png"
    if os.path.exists(os.path.join(OUTPUT_DIR, ipf_key_file)):
        print(f"  - IPF color key: {ipf_key_file}")
    
    if (ENABLE_VTK_OUTPUT or ENABLE_VTKHDF_OUTPUT) and vtk_files:
        print(f"  - VTK output:")
        for vtk_file in vtk_files:
            print(f"    - {os.path.basename(vtk_file)}")
        if any('vtk_series' in f for f in vtk_files):
            print(f"    - vtk_series/ directory with individual timestep files")
        if any('vtp_series' in f for f in vtk_files):
            print(f"    - vtp_series/ directory with VTK point cloud files")
        if any('vts_series' in f for f in vtk_files):
            print(f"    - vts_series/ directory with VTK structured grid files")
    print(f"  - Timing results: timing_results.csv")
    print("="*80)

if __name__ == "__main__":
    main()