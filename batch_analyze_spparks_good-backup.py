#!/usr/bin/env python3
"""
Batch analysis script for SPPARKS IPF volume rendering with PyVista
Processes all dump files matching pattern and creates images + animations

Configuration Options:
- ENABLE_METHOD1/ENABLE_METHOD2: Turn rendering methods on/off individually
- CROP_BOUNDS: Crop data to subvolume [x_min, x_max, y_min, y_max, z_min, z_max]

Examples:
  # Run only Method 1 (faster):
  ENABLE_METHOD1 = True
  ENABLE_METHOD2 = False
  
  # Crop to central 100x100x50 region:
  CROP_BOUNDS = [50, 150, 50, 150, 25, 75]
"""

import numpy as np
import pandas as pd
import pyvista as pv
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Quaternion
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import glob
import os
import time
import cv2
from pathlib import Path
import warnings
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

# Subvolume cropping (set to None to disable cropping)
# Format: [x_min, x_max, y_min, y_max, z_min, z_max] or None
#CROP_BOUNDS = None
CROP_BOUNDS = [0,200,0,200,0,10]
# Example: CROP_BOUNDS = [50, 150, 50, 150, 0, 100]  # Crop to 100x100x100 subvolume

# Configure PyVista for off-screen rendering
pv.OFF_SCREEN = True

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
    """
    Crop data to a subvolume.
    
    Args:
        coords: (N, 3) coordinates
        colors: (N, 3) IPF colors
        grain_ids: (N,) grain IDs
        quaternions: (N, 4) quaternions
        crop_bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
    
    Returns:
        Cropped versions of all inputs
    """
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

def create_method1_rendering(coords, colors, grid_shape, camera_position=None, title="Method 1"):
    """Method 1: Point cloud with glyphs"""
    nx, ny, nz = grid_shape
    
    # Create points
    points = pv.PolyData(coords)
    points['RGB'] = colors
    
    # Create small cubes at each point
    glyphs = points.glyph(
        geom=pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0),
        scale=False,
        orient=False
    )
    
    # Transfer colors to glyphs
    glyphs['RGB'] = np.repeat(colors, 8, axis=0)  # 8 vertices per cube
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=IMAGE_SIZE)
    plotter.background_color = BACKGROUND_COLOR
    
    plotter.add_mesh(
        glyphs,
        scalars='RGB',
        rgb=True,
        opacity=OPACITY_FILLED,
        show_edges=False,
        lighting=False
    )
    
    plotter.add_title(title, font_size=16)
    plotter.add_axes()
    
    # Set camera position if provided
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = 'iso'
    
    return plotter

def create_method2_rendering(coords, colors, grid_shape, camera_position=None, title="Method 2"):
    """Method 2: Structured grid"""
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
    plotter = pv.Plotter(off_screen=True, window_size=IMAGE_SIZE)
    plotter.background_color = BACKGROUND_COLOR
    
    plotter.add_mesh(
        thresholded,
        scalars='RGB',
        rgb=True,
        opacity=OPACITY_FILLED,
        show_edges=False,
        lighting=False
    )
    
    plotter.add_title(title, font_size=16)
    plotter.add_axes()
    
    # Set camera position if provided
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = 'iso'
    
    return plotter

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

def process_single_file(filepath, camera_position, output_dir):
    """Process a single dump file and create images with enabled methods."""
    filename = os.path.basename(filepath)
    timestep = filename.split('.')[-1]
    
    print(f"Processing {filename}...")
    
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
    if CROP_BOUNDS is not None:
        coords, ipf_colors, grain_ids, quaternions = apply_subvolume_crop(
            coords, ipf_colors, grain_ids, quaternions, CROP_BOUNDS
        )
        # Update grid dimensions for cropped volume
        if len(coords) > 0:
            nx = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            ny = int(coords[:, 1].max() - coords[:, 1].min() + 1)
            nz = int(coords[:, 2].max() - coords[:, 2].min() + 1)
            crop_info = f" (cropped: {len(coords):,} atoms)"
        else:
            nx = ny = nz = 1
            crop_info = " (cropped: empty)"
    else:
        crop_info = ""
    
    load_time = time.time() - start_time
    
    # Initialize result dictionary
    result = {
        'timestep': timestep,
        'filename': filename,
        'n_atoms': len(coords),
        'load_time': load_time,
        'method1_time': 0.0,
        'method2_time': 0.0,
        'method1_image': None,
        'method2_image': None
    }
    
    # Method 1 - Point cloud with glyphs
    if ENABLE_METHOD1 and len(coords) > 0:
        start_time = time.time()
        title1 = f"SPPARKS Method 1 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()}{crop_info}"
        plotter1 = create_method1_rendering(coords, ipf_colors, (nx, ny, nz), camera_position, title1)
        image1_path = os.path.join(output_dir, f"method1_timestep_{timestep}.png")
        plotter1.screenshot(image1_path)
        plotter1.close()
        result['method1_time'] = time.time() - start_time
        result['method1_image'] = image1_path
    
    # Method 2 - Structured grid
    if ENABLE_METHOD2 and len(coords) > 0:
        start_time = time.time()
        title2 = f"SPPARKS Method 2 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()}{crop_info}"
        plotter2 = create_method2_rendering(coords, ipf_colors, (nx, ny, nz), camera_position, title2)
        image2_path = os.path.join(output_dir, f"method2_timestep_{timestep}.png")
        plotter2.screenshot(image2_path)
        plotter2.close()
        result['method2_time'] = time.time() - start_time
        result['method2_image'] = image2_path
    
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
    print("SPPARKS BATCH ANALYSIS - IPF VOLUME RENDERING")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all dump files
    pattern = os.path.join(DATA_DIR, FILE_PATTERN)
    dump_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('.')[-1]))
    
    if not dump_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(dump_files)} dump files")
    print(f"Range: {os.path.basename(dump_files[0])} to {os.path.basename(dump_files[-1])}")
    
    # Determine camera position using largest file (last one)
    largest_file = dump_files[-1]
    camera_position = determine_optimal_camera_position(largest_file)
    
    # Process all files
    results = []
    method1_images = []
    method2_images = []
    
    total_start = time.time()
    
    for i, filepath in enumerate(dump_files):
        print(f"\nProgress: {i+1}/{len(dump_files)}")
        
        try:
            result = process_single_file(filepath, camera_position, OUTPUT_DIR)
            results.append(result)
            
            # Collect images for animations (only if they were created)
            if result['method1_image']:
                method1_images.append(result['method1_image'])
            if result['method2_image']:
                method2_images.append(result['method2_image'])
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # Create animations for enabled methods
    print(f"\nCreating animations...")
    if ENABLE_METHOD1 and method1_images:
        method1_video = os.path.join(OUTPUT_DIR, "method1_animation.mp4")
        create_animation(method1_images, method1_video, FPS)
    
    if ENABLE_METHOD2 and method2_images:
        method2_video = os.path.join(OUTPUT_DIR, "method2_animation.mp4")
        create_animation(method2_images, method2_video, FPS)
    
    # Save timing results
    timing_data = pd.DataFrame(results)
    timing_file = os.path.join(OUTPUT_DIR, "timing_results.csv")
    timing_data.to_csv(timing_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Method 1 (Glyphs): {'✓ Enabled' if ENABLE_METHOD1 else '✗ Disabled'}")
    print(f"  Method 2 (Struct): {'✓ Enabled' if ENABLE_METHOD2 else '✗ Disabled'}")
    if CROP_BOUNDS:
        print(f"  Subvolume crop: {CROP_BOUNDS}")
    else:
        print(f"  Subvolume crop: ✗ Disabled")
    
    print(f"\nProcessing Summary:")
    print(f"  Total files processed: {len(results)}")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average time per file: {total_time/len(results):.2f} seconds")
    
    print(f"\nTiming Summary:")
    print(f"  Average load time: {timing_data['load_time'].mean():.3f} ± {timing_data['load_time'].std():.3f} s")
    if ENABLE_METHOD1:
        print(f"  Average Method 1 time: {timing_data['method1_time'].mean():.3f} ± {timing_data['method1_time'].std():.3f} s")
    if ENABLE_METHOD2:
        print(f"  Average Method 2 time: {timing_data['method2_time'].mean():.3f} ± {timing_data['method2_time'].std():.3f} s")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    if ENABLE_METHOD1:
        print(f"  - {len(method1_images)} Method 1 images")
        if method1_images:
            print(f"  - Method 1 animation: method1_animation.mp4")
    if ENABLE_METHOD2:
        print(f"  - {len(method2_images)} Method 2 images")
        if method2_images:
            print(f"  - Method 2 animation: method2_animation.mp4")
    print(f"  - Timing results: {os.path.basename(timing_file)}")
    print("="*80)

if __name__ == "__main__":
    main()