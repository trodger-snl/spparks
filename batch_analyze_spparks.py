#!/usr/bin/env python3
"""
Batch analysis script for SPPARKS IPF volume rendering with PyVista
Processes all dump files matching pattern and creates images + animations

Configuration Options:
- ENABLE_METHOD1/ENABLE_METHOD2: Turn rendering methods on/off individually
- PARALLEL_FILES: Process multiple files in parallel using threading (I/O bound optimization)
- PARALLEL_METHODS: Run Method 1 and Method 2 in parallel per file using threading
- MAX_WORKERS: Maximum number of parallel workers (None = auto-detect)
- MEMORY_LIMIT_GB: Memory limit for adaptive worker scaling
- CROP_BOUNDS: Crop data to subvolume [x_min, x_max, y_min, y_max, z_min, z_max]

Examples:
  # Fast parallel processing:
  PARALLEL_FILES = True
  MAX_WORKERS = 4
  
  # Run only Method 1 with parallelization:
  ENABLE_METHOD1 = True
  ENABLE_METHOD2 = False
  PARALLEL_FILES = True
  
  # Crop to central region with parallel methods:
  CROP_BOUNDS = [50, 150, 50, 150, 25, 75]
  PARALLEL_METHODS = True
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from tqdm import tqdm
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

# Parallelization settings
PARALLEL_FILES = False   # Process multiple files in parallel (DISABLED - use batch_analyze_spparks_parallel.py for parallel processing)
MAX_WORKERS = 4          # Maximum number of parallel workers (None = auto-detect)
PARALLEL_METHODS = False # Run Method 1 and Method 2 in parallel per file
MEMORY_LIMIT_GB = 16     # Memory limit for adaptive worker scaling (None = no limit)

# Subvolume cropping (set to None to disable cropping)
# Format: [x_min, x_max, y_min, y_max, z_min, z_max] or None
#CROP_BOUNDS = None
CROP_BOUNDS = [0,200,0,200,0,10]
# Example: CROP_BOUNDS = [50, 150, 50, 150, 0, 100]  # Crop to 100x100x100 subvolume

# Configure PyVista for thread-safe off-screen rendering with OSMesa
import os

# Set OSMesa backend environment variables BEFORE importing pyvista
os.environ['PYVISTA_USE_OSMESA'] = 'true'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYVISTA_USE_PANEL'] = 'false'

# Configure PyVista for off-screen rendering
pv.OFF_SCREEN = True

# Set PyVista to use a safe backend
try:
    pv.set_plot_theme('document')
    print("✓ PyVista configured with OSMesa backend for thread-safe parallel processing")
except Exception as e:
    print(f"⚠️  PyVista configuration warning: {e}")
    print("Continuing with default configuration...")

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

def get_optimal_worker_count():
    """Determine optimal number of workers based on system resources."""
    # Default to CPU count
    cpu_count = mp.cpu_count()
    
    if MAX_WORKERS is not None:
        optimal_workers = min(MAX_WORKERS, cpu_count)
    else:
        optimal_workers = cpu_count
    
    # Check memory constraints
    if MEMORY_LIMIT_GB is not None:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # Estimate ~400MB per worker for large files
        estimated_memory_per_worker = 0.4
        max_workers_by_memory = int(available_memory_gb / estimated_memory_per_worker)
        optimal_workers = min(optimal_workers, max_workers_by_memory)
    
    return max(1, optimal_workers)


def process_methods_parallel(coords, ipf_colors, grid_shape, camera_position, timestep, output_dir):
    """Process both methods in parallel for a single file."""
    results = {'method1_time': 0.0, 'method2_time': 0.0, 
               'method1_image': None, 'method2_image': None}
    
    def run_method1():
        if ENABLE_METHOD1 and len(coords) > 0:
            start_time = time.time()
            crop_info = f" (cropped: {len(coords):,} atoms)" if CROP_BOUNDS else ""
            title = f"SPPARKS Method 1 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()}{crop_info}"
            plotter = create_method1_rendering(coords, ipf_colors, grid_shape, camera_position, title)
            image_path = os.path.join(output_dir, f"method1_timestep_{timestep}.png")
            plotter.screenshot(image_path)
            plotter.close()
            return time.time() - start_time, image_path
        return 0.0, None
    
    def run_method2():
        if ENABLE_METHOD2 and len(coords) > 0:
            start_time = time.time()
            crop_info = f" (cropped: {len(coords):,} atoms)" if CROP_BOUNDS else ""
            title = f"SPPARKS Method 2 - Timestep {timestep} - IPF {IPF_DIRECTION.upper()}{crop_info}"
            plotter = create_method2_rendering(coords, ipf_colors, grid_shape, camera_position, title)
            image_path = os.path.join(output_dir, f"method2_timestep_{timestep}.png")
            plotter.screenshot(image_path)
            plotter.close()
            return time.time() - start_time, image_path
        return 0.0, None
    
    if PARALLEL_METHODS and ENABLE_METHOD1 and ENABLE_METHOD2:
        # Run both methods in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(run_method1)
            future2 = executor.submit(run_method2)
            
            results['method1_time'], results['method1_image'] = future1.result()
            results['method2_time'], results['method2_image'] = future2.result()
    else:
        # Run methods sequentially
        results['method1_time'], results['method1_image'] = run_method1()
        results['method2_time'], results['method2_image'] = run_method2()
    
    return results

def create_method1_rendering(coords, colors, grid_shape, camera_position=None, title="Method 1"):
    """Method 1: Point cloud with glyphs"""
    # Ensure PyVista is configured for this rendering
    configure_pyvista_for_worker()
    
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
    # Ensure PyVista is configured for this rendering
    configure_pyvista_for_worker()
    
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

def configure_pyvista_for_worker():
    """Configure PyVista environment for worker threads."""
    import os
    import pyvista as pv
    
    # Set OSMesa environment for this worker
    os.environ['PYVISTA_USE_OSMESA'] = 'true'
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['PYVISTA_USE_PANEL'] = 'false'
    
    # Ensure PyVista is in off-screen mode
    pv.OFF_SCREEN = True
    
    # Set theme
    try:
        pv.set_plot_theme('document')
    except:
        pass

def process_single_file(filepath, camera_position, output_dir):
    """Process a single dump file and create images with enabled methods."""
    # Configure PyVista for this worker
    configure_pyvista_for_worker()
    
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
    
    # Process rendering methods (potentially in parallel)
    if len(coords) > 0:
        method_results = process_methods_parallel(
            coords, ipf_colors, (nx, ny, nz), camera_position, timestep, output_dir
        )
        result.update(method_results)
    
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
    all_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('.')[-1]))
    
    if not all_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # For testing, process only first 3 files (remove this line for full processing)
    dump_files = all_files[:3]  # REMOVE THIS LINE FOR FULL PROCESSING
    
    print(f"Found {len(all_files)} dump files total")
    print(f"Processing {len(dump_files)} files for testing")
    print(f"Range: {os.path.basename(dump_files[0])} to {os.path.basename(dump_files[-1])}")
    
    # Determine camera position using largest file (last one from all files)
    largest_file = all_files[-1]
    camera_position = determine_optimal_camera_position(largest_file)
    
    # Determine optimal worker count
    if PARALLEL_FILES:
        num_workers = get_optimal_worker_count()
        print(f"Using {num_workers} parallel workers for file processing")
        if MEMORY_LIMIT_GB:
            available_gb = psutil.virtual_memory().available / (1024**3)
            print(f"Available memory: {available_gb:.1f} GB (limit: {MEMORY_LIMIT_GB} GB)")
    else:
        num_workers = 1
        print("Using serial processing (PARALLEL_FILES = False)")
    
    # Process all files
    results = []
    method1_images = []
    method2_images = []
    
    total_start = time.time()
    
    if PARALLEL_FILES and num_workers > 1:
        # Try parallel processing, fall back to serial if it fails
        print(f"\nAttempting parallel processing with {num_workers} workers...")
        
        try:
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid serialization issues
            # This still provides parallelization for I/O bound operations
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all jobs
                future_to_file = {executor.submit(process_single_file, filepath, camera_position, OUTPUT_DIR): filepath 
                                for filepath in dump_files}
                
                # Process completed jobs with progress bar
                with tqdm(total=len(dump_files), desc="Processing files (parallel)") as pbar:
                    for future in as_completed(future_to_file):
                        filepath = future_to_file[future]
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
                            print(f"Error processing {os.path.basename(filepath)}: {e}")
                        
                        pbar.update(1)
            
            print("✓ Parallel processing completed successfully")
            
        except Exception as e:
            print(f"⚠️  Parallel processing failed: {e}")
            print("Falling back to serial processing...")
            
            # Clear any partial results and process serially
            results.clear()
            method1_images.clear()
            method2_images.clear()
            
            for filepath in tqdm(dump_files, desc="Processing files (serial fallback)"):
                try:
                    result = process_single_file(filepath, camera_position, OUTPUT_DIR)
                    results.append(result)
                    
                    # Collect images for animations
                    if result['method1_image']:
                        method1_images.append(result['method1_image'])
                    if result['method2_image']:
                        method2_images.append(result['method2_image'])
                        
                except Exception as e:
                    print(f"Error processing {os.path.basename(filepath)}: {e}")
                    continue
    else:
        # Serial processing with progress bar
        print(f"\nProcessing {len(dump_files)} files serially...")
        
        for filepath in tqdm(dump_files, desc="Processing files"):
            try:
                result = process_single_file(filepath, camera_position, OUTPUT_DIR)
                results.append(result)
                
                # Collect images for animations
                if result['method1_image']:
                    method1_images.append(result['method1_image'])
                if result['method2_image']:
                    method2_images.append(result['method2_image'])
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(filepath)}: {e}")
                continue
    
    total_time = time.time() - total_start
    
    # Sort results by timestep to ensure correct animation order
    try:
        results.sort(key=lambda x: int(x['timestep']))
        if method1_images:
            method1_images.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        if method2_images:
            method2_images.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not sort results by timestep: {e}")
        print("Animations may not be in correct chronological order")
    
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
    print(f"  Parallel files: {'✓ Enabled' if PARALLEL_FILES else '✗ Disabled'}")
    if PARALLEL_FILES:
        print(f"    Workers used: {num_workers}")
    print(f"  Parallel methods: {'✓ Enabled' if PARALLEL_METHODS else '✗ Disabled'}")
    if CROP_BOUNDS:
        print(f"  Subvolume crop: {CROP_BOUNDS}")
    else:
        print(f"  Subvolume crop: ✗ Disabled")
    
    print(f"\nProcessing Summary:")
    print(f"  Total files processed: {len(results)}")
    print(f"  Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Average time per file: {total_time/len(results):.2f} seconds")
    if PARALLEL_FILES and num_workers > 1:
        estimated_serial_time = total_time * num_workers
        speedup = estimated_serial_time / total_time
        print(f"  Estimated speedup: {speedup:.1f}x (vs serial processing)")
        print(f"  Estimated serial time: {estimated_serial_time/60:.1f} minutes")
    
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