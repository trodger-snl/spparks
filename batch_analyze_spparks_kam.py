#!/usr/bin/env python3
"""
SPPARKS batch analysis with IPF coloring and KAM (Kernel Average Misorientation) calculations
This version includes both IPF and KAM rendering capabilities using orix
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
ENABLE_KAM = True       # Enable KAM calculations and rendering

# KAM Configuration
KAM_THRESHOLD_DEG = 2.5  # Maximum misorientation angle to consider (degrees)
KAM_ORDER = 1           # Order of neighbors (1 = immediate neighbors)
KAM_MAX_KAM = False     # If True, use max instead of mean

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
    
    # Create mask for points within bounds
    mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
        (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
    )
    
    # Apply mask and adjust coordinates to start from 0
    cropped_coords = coords[mask].copy()
    cropped_coords[:, 0] -= x_min
    cropped_coords[:, 1] -= y_min
    cropped_coords[:, 2] -= z_min
    
    cropped_colors = colors[mask]
    cropped_grain_ids = grain_ids[mask]
    cropped_quaternions = quaternions[mask]
    
    return cropped_coords, cropped_colors, cropped_grain_ids, cropped_quaternions

def generate_ipf_colors(quaternions, crystal_structure, direction):
    """Generate IPF colors using orix."""
    from orix.quaternion import Orientation, Quaternion
    from orix.crystal_map import Phase
    from orix.vector import Vector3d
    from orix.plot import IPFColorKeyTSL
    
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
    elif direction == 'y':
        ipf_dir = Vector3d([0, 1, 0])
    elif direction == 'x':
        ipf_dir = Vector3d([1, 0, 0])
    else:
        ipf_dir = Vector3d([0, 0, 1])
    
    # Create IPF color key
    ipf_key = IPFColorKeyTSL(phase.point_group, direction=ipf_dir)
    
    # Generate colors
    rgb_colors = ipf_key.orientation2color(orientations)
    
    return rgb_colors

def create_crystal_map_from_spparks(coords, quaternions, grain_ids, crystal_structure='fcc'):
    """Create an orix CrystalMap from SPPARKS data."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Orientation, Quaternion
    from orix.crystal_map import Phase
    
    # Create phase
    if crystal_structure.lower() == 'fcc':
        phase = Phase('Austenite', point_group='m-3m')
    else:
        raise ValueError(f"Unsupported crystal structure: {crystal_structure}")
    
    phase_list = PhaseList([phase])
    
    # Create orientations from quaternions
    quat_obj = Quaternion(quaternions)
    orientations = Orientation(quat_obj)
    orientations.phase = phase
    
    # Determine grid dimensions
    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    z_min, z_max = int(coords[:, 2].min()), int(coords[:, 2].max())
    
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    nz = z_max - z_min + 1
    
    # Create 3D arrays for orientations and properties  
    identity_quat = Quaternion.identity()
    identity_orientation = Orientation(identity_quat)
    identity_orientation.phase = phase
    orientation_array = np.full((nz, ny, nx), identity_orientation, dtype=object)
    phase_array = np.zeros((nz, ny, nx), dtype=int)
    is_indexed = np.zeros((nz, ny, nx), dtype=bool)
    
    # Fill arrays based on coordinates
    for i, (coord, orientation, grain_id) in enumerate(zip(coords, orientations, grain_ids)):
        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
        # Convert to array indices
        xi, yi, zi = x - x_min, y - y_min, z - z_min
        if 0 <= xi < nx and 0 <= yi < ny and 0 <= zi < nz:
            orientation_array[zi, yi, xi] = orientation
            phase_array[zi, yi, xi] = 0  # Single phase
            is_indexed[zi, yi, xi] = True
    
    # Create step sizes (assuming unit spacing)
    step_sizes = (1.0, 1.0, 1.0)  # (z, y, x) step sizes
    
    # Create CrystalMap using simplified approach
    # Flatten orientations for CrystalMap
    flat_orientations = []
    flat_phase_ids = []
    flat_is_indexed = []
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                flat_orientations.append(orientation_array[z, y, x])
                flat_phase_ids.append(phase_array[z, y, x])
                flat_is_indexed.append(is_indexed[z, y, x])
    
    crystal_map = CrystalMap(
        rotations=flat_orientations,
        phase_id=flat_phase_ids,
        phase_list=phase_list,
        is_indexed=flat_is_indexed,
        scan_unit='um',
        step_sizes=step_sizes,
        shape=(nz, ny, nx)
    )
    
    return crystal_map

def calculate_kam_3d(crystal_map, threshold_deg=None, order=1, max_kam=False):
    """
    Calculate 3D Kernel Average Misorientation (KAM) for an orix CrystalMap.
    Based on the implementation from orix_kam_3d_implementation.py
    """
    from orix.quaternion import Misorientation
    
    # Check if data is 3D
    if len(crystal_map.shape) != 3:
        raise ValueError("This function requires 3D EBSD data.")
    
    # Get the orientation data and shape
    orientations = crystal_map.orientations
    shape = crystal_map.shape  # (nz, ny, nx)
    
    # Create output array
    kam_map = np.full(shape, np.nan)
    
    # Define 3D neighbor offsets
    def get_3d_neighbor_offsets(order):
        """Generate 3D neighbor offsets up to specified order"""
        offsets = []
        for k in range(-order, order + 1):
            for j in range(-order, order + 1):
                for i in range(-order, order + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue  # Skip center voxel
                    # Check if within specified order (using Chebyshev distance)
                    if max(abs(i), abs(j), abs(k)) <= order:
                        offsets.append((k, j, i))  # (z, y, x)
        return offsets
    
    neighbor_offsets = get_3d_neighbor_offsets(order)
    
    # Calculate KAM for each voxel
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                # Check if current voxel is indexed
                if not crystal_map.is_indexed[z, y, x]:
                    continue
                
                flat_idx = z * shape[1] * shape[2] + y * shape[2] + x
                center_orientation = orientations[flat_idx]
                
                misorientations = []
                
                # Check all neighbors
                for dz, dy, dx in neighbor_offsets:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    
                    # Check bounds
                    if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                        # Check if neighbor is indexed
                        if crystal_map.is_indexed[nz, ny, nx]:
                            neighbor_flat_idx = nz * shape[1] * shape[2] + ny * shape[2] + nx
                            neighbor_orientation = orientations[neighbor_flat_idx]
                            
                            # Calculate misorientation angle
                            misori = Misorientation(neighbor_orientation * ~center_orientation)
                            angle_rad = misori.angle.data[0]
                            
                            # Apply threshold
                            if threshold_deg is None or np.degrees(angle_rad) <= threshold_deg:
                                misorientations.append(angle_rad)
                
                # Calculate KAM value
                if len(misorientations) > 0:
                    if max_kam:
                        kam_map[z, y, x] = np.max(misorientations)
                    else:
                        kam_map[z, y, x] = np.mean(misorientations)
    
    return kam_map

def kam_to_colors(kam_map, vmin=0, vmax=None, colormap='viridis'):
    """Convert KAM values to RGB colors."""
    import matplotlib.cm as cm
    
    # Convert to degrees
    kam_degrees = np.degrees(kam_map)
    
    # Set default vmax if not provided
    if vmax is None:
        vmax = np.nanpercentile(kam_degrees, 95)  # Use 95th percentile
    
    # Normalize values
    kam_normalized = np.clip((kam_degrees - vmin) / (vmax - vmin), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors = cmap(kam_normalized)[:, :, :, :3]  # Remove alpha channel
    
    # Handle NaN values (set to white or background color)
    nan_mask = np.isnan(kam_degrees)
    colors[nan_mask] = [1.0, 1.0, 1.0]  # White for NaN values
    
    return colors

def create_ipf_color_key(crystal_structure='fcc', direction='z', output_path=None):
    """Create IPF color key legend using orix."""
    from orix.crystal_map import Phase
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    import matplotlib.pyplot as plt
    
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

def create_kam_color_key(kam_values, colormap='viridis', output_path=None):
    """Create KAM colorbar legend."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Convert to degrees and get range
    kam_degrees = np.degrees(kam_values[~np.isnan(kam_values)])
    vmin = 0
    vmax = np.percentile(kam_degrees, 95)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 1.5))
    
    # Create colorbar
    cmap = cm.get_cmap(colormap)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    sm.set_clim(vmin, vmax)
    
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')
    cbar.set_label('KAM (degrees)', fontsize=12, fontweight='bold')
    
    # Hide axes
    ax.set_visible(False)
    
    # Set title
    fig.suptitle('Kernel Average Misorientation Color Scale', fontsize=14, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return output_path
    
    return fig

def create_subprocess_renderer_script():
    """Create the subprocess script for PyVista rendering."""
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
    
    script_path = "/tmp/pyvista_subprocess_renderer_kam.py"
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

def process_single_file_data(args):
    """Worker function that processes a single file - serializable for multiprocessing."""
    filepath, camera_position, output_dir, enable_method1, enable_method2, enable_kam, crop_bounds, enable_vtk, enable_vtkhdf = args
    
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
    
    # Calculate KAM if enabled
    kam_colors = None
    if enable_kam and len(coords) > 0:
        try:
            # Create crystal map from SPPARKS data
            crystal_map = create_crystal_map_from_spparks(coords, quaternions, grain_ids, CRYSTAL_STRUCTURE)
            
            # Calculate KAM
            kam_map = calculate_kam_3d(crystal_map, 
                                     threshold_deg=KAM_THRESHOLD_DEG, 
                                     order=KAM_ORDER, 
                                     max_kam=KAM_MAX_KAM)
            
            # Convert KAM to colors
            kam_colors = kam_to_colors(kam_map)
            # Flatten KAM colors to match coordinate structure
            kam_colors_flat = []
            for i, coord in enumerate(coords):
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                    kam_colors_flat.append(kam_colors[z, y, x])
                else:
                    kam_colors_flat.append([1.0, 1.0, 1.0])  # White for out of bounds
            kam_colors = np.array(kam_colors_flat)
            
        except Exception as e:
            print(f"Warning: KAM calculation failed for {filename}: {e}")
            kam_colors = None
    
    load_time = time.time() - start_time
    
    # Initialize result
    result = {
        'timestep': int(timestep),
        'filename': filename,
        'n_atoms': len(coords),
        'load_time': load_time,
        'method1_time': 0.0,
        'method2_time': 0.0,
        'kam_time': 0.0,
        'method1_image': '',
        'method2_image': '',
        'ipf_image': '',
        'kam_image': ''
    }
    
    # Method 2: Structured grid rendering (IPF)
    if enable_method2 and len(coords) > 0:
        method2_start = time.time()
        try:
            output_path = os.path.join(output_dir, f"method2_timestep_{timestep}.png")
            title = f"SPPARKS Method 2 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()} (cropped: {len(coords)} atoms)"
            
            render_with_subprocess(coords, ipf_colors, (nx, ny, nz), camera_position, title, output_path)
            result['method2_image'] = output_path
            result['ipf_image'] = output_path
            
        except Exception as e:
            print(f"Method 2 rendering failed for {filename}: {e}")
        
        result['method2_time'] = time.time() - method2_start
    
    # KAM rendering
    if enable_kam and kam_colors is not None and len(coords) > 0:
        kam_start = time.time()
        try:
            output_path = os.path.join(output_dir, f"kam_timestep_{timestep}.png")
            title = f"SPPARKS KAM - Timestep {timestep} - Threshold {KAM_THRESHOLD_DEG}° (cropped: {len(coords)} atoms)"
            
            render_with_subprocess(coords, kam_colors, (nx, ny, nz), camera_position, title, output_path)
            result['kam_image'] = output_path
            
        except Exception as e:
            print(f"KAM rendering failed for {filename}: {e}")
        
        result['kam_time'] = time.time() - kam_start
    
    return result

def main():
    """Main batch processing function."""
    print("=" * 80)
    print("SPPARKS BATCH ANALYSIS - IPF AND KAM RENDERING")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create IPF color key
    ipf_key_file = f"ipf_color_key_{IPF_DIRECTION}_{CRYSTAL_STRUCTURE}.png"
    ipf_key_path = os.path.join(OUTPUT_DIR, ipf_key_file)
    create_ipf_color_key(CRYSTAL_STRUCTURE, IPF_DIRECTION, ipf_key_path)
    print(f"IPF color key saved: {ipf_key_file}")
    
    # Get dump files
    dump_files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
    print(f"Found {len(dump_files)} dump files total")
    
    # For testing, limit to first few files
    test_files = dump_files[:3]
    print(f"Processing {len(test_files)} files for testing")
    print(f"Range: {os.path.basename(test_files[0])} to {os.path.basename(test_files[-1])}")
    
    # Determine optimal camera position using largest file
    largest_file = max(dump_files, key=lambda x: int(x.split('.')[-1]))
    camera_position = determine_optimal_camera_position(largest_file)
    
    # Process files
    if PARALLEL_FILES and len(test_files) > 1:
        # Determine number of workers
        num_workers = min(MAX_WORKERS, len(test_files), psutil.cpu_count())
        print(f"Using {num_workers} parallel workers (subprocess-based)")
        
        # Prepare arguments for parallel processing
        args_list = [
            (filepath, camera_position, OUTPUT_DIR, ENABLE_METHOD1, ENABLE_METHOD2, ENABLE_KAM, 
             CROP_BOUNDS, ENABLE_VTK_OUTPUT, ENABLE_VTKHDF_OUTPUT) 
            for filepath in test_files
        ]
        
        print(f"\\nProcessing {len(test_files)} files with {num_workers} parallel workers...")
        
        # Process files in parallel
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(args_list), desc="Processing files (parallel)") as pbar:
                futures = {executor.submit(process_single_file_data, args): args for args in args_list}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing file: {e}")
                        pbar.update(1)
        
        total_time = time.time() - start_time
        print(f"✓ Parallel processing completed successfully")
        
    else:
        # Sequential processing
        print("Processing files sequentially...")
        start_time = time.time()
        results = []
        
        for filepath in tqdm(test_files, desc="Processing files (sequential)"):
            args = (filepath, camera_position, OUTPUT_DIR, ENABLE_METHOD1, ENABLE_METHOD2, ENABLE_KAM,
                   CROP_BOUNDS, ENABLE_VTK_OUTPUT, ENABLE_VTKHDF_OUTPUT)
            try:
                result = process_single_file_data(args)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        total_time = time.time() - start_time
    
    # Save timing results
    results_df = pd.DataFrame(results)
    timing_file = os.path.join(OUTPUT_DIR, "timing_results.csv")
    results_df.to_csv(timing_file, index=False)
    
    # Create animations
    print(f"\\nCreating animations...")
    
    # IPF animation
    method2_images = [r['method2_image'] for r in results if r['method2_image']]
    if method2_images:
        animation_path = os.path.join(OUTPUT_DIR, "ipf_animation.mp4")
        print(f"Creating IPF animation: {animation_path}")
        create_animation(method2_images, animation_path, FPS)
        print(f"IPF animation saved: {animation_path}")
    
    # KAM animation
    kam_images = [r['kam_image'] for r in results if r['kam_image']]
    if kam_images:
        animation_path = os.path.join(OUTPUT_DIR, "kam_animation.mp4")
        print(f"Creating KAM animation: {animation_path}")
        create_animation(kam_images, animation_path, FPS)
        print(f"KAM animation saved: {animation_path}")
        
        # Create KAM color key
        if ENABLE_KAM:
            # Load a KAM image to extract values for color key
            kam_key_path = os.path.join(OUTPUT_DIR, f"kam_color_key.png")
            # For demo, create with dummy values - in real implementation,
            # you'd extract actual KAM values from calculations
            dummy_kam = np.linspace(0, np.radians(5), 1000)  # 0-5 degrees
            create_kam_color_key(dummy_kam, colormap='viridis', output_path=kam_key_path)
            print(f"KAM color key saved: kam_color_key.png")
    
    # Print summary
    print(f"\\n" + "=" * 80)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total files processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    if len(results) > 0:
        avg_time = total_time / len(results)
        print(f"Average time per file: {avg_time:.2f} seconds")
        
        if PARALLEL_FILES and len(test_files) > 1:
            speedup = len(results) / (total_time / avg_time)
            print(f"Estimated speedup: {speedup:.1f}x (vs serial processing)")
    
    print(f"\\nOutput directory: {OUTPUT_DIR}")
    if method2_images:
        print(f"  - {len(method2_images)} IPF images")
        print(f"  - IPF animation: ipf_animation.mp4")
    if kam_images:
        print(f"  - {len(kam_images)} KAM images")
        print(f"  - KAM animation: kam_animation.mp4")
        print(f"  - KAM color key: kam_color_key.png")
    
    # Check for IPF color key
    print(f"  - IPF color key: {ipf_key_file}")
    print(f"  - Timing results: timing_results.csv")
    print("=" * 80)

def create_animation(image_files, output_path, fps=5):
    """Create MP4 animation from image files."""
    if not image_files:
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    height, width, _ = first_img.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add images to video
    for img_path in image_files:
        img = cv2.imread(img_path)
        out.write(img)
    
    # Release everything
    out.release()

if __name__ == "__main__":
    main()