#!/usr/bin/env python3
"""
Simplified SPPARKS batch analysis with IPF coloring and basic KAM-like calculations
This version includes a simplified KAM approach that doesn't require complex orix CrystalMap setup
"""

import numpy as np
import pandas as pd
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Import our new modules
import config
from io_utils import parse_spparks_dump, create_animation
from ipf_analysis import generate_ipf_colors, create_ipf_color_key
from kam_analysis import calculate_simplified_kam, kam_to_colors, create_kam_color_key
from gnd_analysis import calculate_gnd_density, gnd_to_colors, create_gnd_color_key
from common_utils import apply_subvolume_crop, apply_expanded_crop_for_kam, detect_grain_boundaries
from visualization import determine_optimal_camera_position, render_with_subprocess, render_triple_ipf_with_subprocess
from flythrough import create_flythrough_animation


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

def apply_expanded_crop_for_kam(coords, quaternions, crop_bounds, neighbor_radius):
    """
    Create an expanded crop region for KAM calculations to ensure proper neighborhoods.
    Returns expanded data and a mask indicating which points are in the original crop region.
    """
    if crop_bounds is None:
        # No cropping - return all data with all points marked as "in region"
        mask_in_original = np.ones(len(coords), dtype=bool)
        return coords, quaternions, mask_in_original, (0, 0, 0)
    
    x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds
    
    # Expand bounds by neighbor_radius to include buffer zone
    expanded_x_min = x_min - neighbor_radius
    expanded_x_max = x_max + neighbor_radius
    expanded_y_min = y_min - neighbor_radius
    expanded_y_max = y_max + neighbor_radius
    expanded_z_min = z_min - neighbor_radius
    expanded_z_max = z_max + neighbor_radius
    
    # Create mask for expanded region
    expanded_mask = (
        (coords[:, 0] >= expanded_x_min) & (coords[:, 0] <= expanded_x_max) &
        (coords[:, 1] >= expanded_y_min) & (coords[:, 1] <= expanded_y_max) &
        (coords[:, 2] >= expanded_z_min) & (coords[:, 2] <= expanded_z_max)
    )
    
    # Create mask for original crop region within the expanded data
    original_mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
        (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
    )
    
    # Extract expanded data
    expanded_coords = coords[expanded_mask].copy()
    expanded_quaternions = quaternions[expanded_mask]
    
    # Adjust coordinates to start from 0
    offset = (expanded_x_min, expanded_y_min, expanded_z_min)
    expanded_coords[:, 0] -= expanded_x_min
    expanded_coords[:, 1] -= expanded_y_min
    expanded_coords[:, 2] -= expanded_z_min
    
    # Create mask indicating which points in expanded data are in original crop region
    # We need to recompute this for the expanded coordinate system
    mask_in_original = (
        (expanded_coords[:, 0] >= neighbor_radius) & 
        (expanded_coords[:, 0] <= x_max - x_min + neighbor_radius) &
        (expanded_coords[:, 1] >= neighbor_radius) & 
        (expanded_coords[:, 1] <= y_max - y_min + neighbor_radius) &
        (expanded_coords[:, 2] >= neighbor_radius) & 
        (expanded_coords[:, 2] <= z_max - z_min + neighbor_radius)
    )
    
    return expanded_coords, expanded_quaternions, mask_in_original, offset

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

def calculate_misorientation_angle(q1, q2):
    """Calculate misorientation angle between two quaternions in radians."""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot_product = np.abs(np.dot(q1, q2))
    
    # Ensure dot product is within valid range
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Calculate misorientation angle
    angle = 2 * np.arccos(dot_product)
    
    return angle

def calculate_simplified_kam(coords, quaternions, neighbor_radius=1, threshold_deg=None, mask_in_region=None):
    """
    Calculate simplified KAM using vectorized operations for improved performance.
    This avoids the complex orix CrystalMap setup while providing similar results.
    
    Parameters:
    -----------
    coords : array
        Coordinates of all points (including buffer zone if expanded)
    quaternions : array
        Quaternions for all points
    neighbor_radius : int
        Radius for neighbor search
    threshold_deg : float
        Threshold angle in degrees
    mask_in_region : array, optional
        Boolean mask indicating which points are in the original region of interest.
        If provided, KAM will only be calculated for these points.
    """
    if len(coords) == 0:
        return np.array([])
    
    total_points = len(coords)
    region_points = np.sum(mask_in_region) if mask_in_region is not None else total_points
    
    
    # Convert coordinates to integers for indexing
    coords_int = coords.astype(int)
    
    # Create a 3D grid representation with efficient lookups
    x_min, x_max = coords_int[:, 0].min(), coords_int[:, 0].max()
    y_min, y_max = coords_int[:, 1].min(), coords_int[:, 1].max()
    z_min, z_max = coords_int[:, 2].min(), coords_int[:, 2].max()
    
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    nz = z_max - z_min + 1
    
    # Create a 3D grid with indices (-1 means no data)
    grid = np.full((nx, ny, nz), -1, dtype=int)
    
    # Fill grid with point indices (vectorized)
    grid_coords = coords_int - np.array([x_min, y_min, z_min])
    grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = np.arange(len(coords_int))
    
    # Generate neighbor offsets once (vectorized)
    r = neighbor_radius
    dx, dy, dz = np.meshgrid(range(-r, r+1), range(-r, r+1), range(-r, r+1), indexing='ij')
    offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    center_mask = ~((offsets == 0).all(axis=1))
    offsets = offsets[center_mask]
    
    # Normalize quaternions once for all calculations
    quaternions_norm = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    # Calculate KAM only for points in the region of interest
    kam_values = np.full(len(coords), np.nan)
    
    # Determine which points to process
    if mask_in_region is not None:
        points_to_process = np.where(mask_in_region)[0]
    else:
        points_to_process = np.arange(len(coords))
    
    # Process points in batches for memory efficiency
    batch_size = min(1000, len(points_to_process))
    
    for start_idx in range(0, len(points_to_process), batch_size):
        end_idx = min(start_idx + batch_size, len(points_to_process))
        batch_indices = points_to_process[start_idx:end_idx]
        
        for point_idx in batch_indices:
            x, y, z = coords_int[point_idx]
            grid_x, grid_y, grid_z = x - x_min, y - y_min, z - z_min
            center_quat = quaternions_norm[point_idx]
            
            # Calculate neighbor positions
            neighbor_positions = np.array([grid_x, grid_y, grid_z]) + offsets
            
            # Filter positions within grid bounds
            valid_mask = (
                (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < nx) &
                (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < ny) &
                (neighbor_positions[:, 2] >= 0) & (neighbor_positions[:, 2] < nz)
            )
            
            if not np.any(valid_mask):
                continue
                
            valid_positions = neighbor_positions[valid_mask]
            
            # Get neighbor indices from grid (vectorized lookup)
            neighbor_indices = grid[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
            
            # Filter out positions without data
            data_mask = neighbor_indices >= 0
            if not np.any(data_mask):
                continue
                
            neighbor_indices = neighbor_indices[data_mask]
            neighbor_quats = quaternions_norm[neighbor_indices]
            
            # Vectorized misorientation calculation
            # Calculate dot products between center quaternion and all neighbors
            dot_products = np.abs(np.dot(neighbor_quats, center_quat))
            
            # Clip to valid range and calculate angles
            dot_products = np.clip(dot_products, 0.0, 1.0)
            angles_rad = 2 * np.arccos(dot_products)
            
            # Apply threshold if specified
            if threshold_deg is not None:
                angles_deg = np.degrees(angles_rad)
                valid_angles = angles_rad[angles_deg <= threshold_deg]
            else:
                valid_angles = angles_rad
            
            # Calculate KAM (average misorientation)
            if len(valid_angles) > 0:
                kam_values[point_idx] = np.mean(valid_angles)
    
    valid_count = np.sum(~np.isnan(kam_values))
    
    return kam_values

def kam_to_colors(kam_values, vmin=0, vmax=None, colormap='viridis'):
    """Convert KAM values to RGB colors."""
    import matplotlib.cm as cm
    
    # Convert to degrees
    kam_degrees = np.degrees(kam_values)
    
    # Set default vmax if not provided
    if vmax is None:
        vmax = np.nanpercentile(kam_degrees, 95) if not np.all(np.isnan(kam_degrees)) else 5.0
    
    # Normalize values
    kam_normalized = np.clip((kam_degrees - vmin) / (vmax - vmin), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors = cmap(kam_normalized)[:, :3]  # Remove alpha channel
    
    # Handle NaN values (set to white)
    nan_mask = np.isnan(kam_degrees)
    colors[nan_mask] = [1.0, 1.0, 1.0]
    
    return colors

def calculate_gnd_density(kam_values, burgers_vector=BURGERS_VECTOR_MAGNITUDE, step_size=VOXEL_STEP_SIZE):
    """
    Calculate Geometrically Necessary Dislocation (GND) density from KAM values.
    
    Parameters:
    -----------
    kam_values : array
        KAM values in radians
    burgers_vector : float
        Burgers vector magnitude in meters (default: 0.255e-9 m for FCC)
    step_size : float
        Physical step size between measurement points in meters (default: 1.0e-6 m)
        
    Returns:
    --------
    gnd_density : array
        GND density in m^-2
    """
    if len(kam_values) == 0:
        return np.array([])
    
    # Convert KAM (radians) to GND density using: ρ_GND = 2θ / (b × d)
    # where θ = KAM angle, b = Burgers vector, d = step size
    gnd_density = 2 * kam_values / (burgers_vector * step_size)
    
    # Handle NaN values - keep them as NaN
    # Physical range: typical GND densities are 10^12 to 10^16 m^-2
    
    return gnd_density

def gnd_to_colors(gnd_values, vmin=1e12, vmax=1e15, colormap='plasma'):
    """Convert GND density values to RGB colors."""
    import matplotlib.cm as cm
    
    # Handle empty array
    if len(gnd_values) == 0:
        return np.array([])
    
    # Set default vmax if not provided
    if vmax is None:
        valid_values = gnd_values[~np.isnan(gnd_values)]
        if len(valid_values) > 0:
            vmax = np.percentile(valid_values, 95)
        else:
            vmax = 1e15
    
    # Use log scale for GND density visualization
    log_gnd = np.log10(np.clip(gnd_values, vmin, vmax))
    log_vmin = np.log10(vmin)
    log_vmax = np.log10(vmax)
    
    # Normalize values
    gnd_normalized = np.clip((log_gnd - log_vmin) / (log_vmax - log_vmin), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors = cmap(gnd_normalized)[:, :3]  # Remove alpha channel
    
    # Handle NaN values (set to white)
    nan_mask = np.isnan(gnd_values)
    colors[nan_mask] = [1.0, 1.0, 1.0]
    
    return colors

def detect_grain_boundaries(coords, quaternions, threshold_deg=5.0, neighbor_radius=1):
    """
    Detect grain boundaries based on misorientation threshold.
    
    Parameters:
    -----------
    coords : array
        Coordinates of all points
    quaternions : array  
        Quaternions for all points
    threshold_deg : float
        Misorientation threshold for grain boundaries (degrees)
    neighbor_radius : int
        Radius for neighbor search
        
    Returns:
    --------
    boundary_edges : list of tuples
        List of (point1_idx, point2_idx) pairs representing boundary edges
    boundary_coords : array
        Coordinates of boundary line segments for rendering
    """
    if len(coords) == 0:
        return [], np.array([])
    
    
    # Convert coordinates to integers for indexing
    coords_int = coords.astype(int)
    
    # Create a 3D grid representation
    x_min, x_max = coords_int[:, 0].min(), coords_int[:, 0].max()
    y_min, y_max = coords_int[:, 1].min(), coords_int[:, 1].max()
    z_min, z_max = coords_int[:, 2].min(), coords_int[:, 2].max()
    
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    nz = z_max - z_min + 1
    
    # Create grid with point indices
    grid = np.full((nx, ny, nz), -1, dtype=int)
    
    # Fill grid with point indices (vectorized)
    grid_coords = coords_int - np.array([x_min, y_min, z_min])
    grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = np.arange(len(coords_int))
    
    # Normalize quaternions
    quaternions_norm = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    # Generate neighbor offsets (face neighbors only for cleaner boundaries)
    face_offsets = np.array([
        [-1, 0, 0], [1, 0, 0],   # x-direction
        [0, -1, 0], [0, 1, 0],   # y-direction  
        [0, 0, -1], [0, 0, 1]    # z-direction
    ])
    
    boundary_edges = []
    boundary_coords = []
    
    # Check each point for boundary neighbors
    for i, (x, y, z) in enumerate(coords_int):
        grid_x, grid_y, grid_z = x - x_min, y - y_min, z - z_min
        center_quat = quaternions_norm[i]
        
        # Check face neighbors
        for dx, dy, dz in face_offsets:
            nx_pos = grid_x + dx
            ny_pos = grid_y + dy
            nz_pos = grid_z + dz
            
            # Check bounds
            if (0 <= nx_pos < nx and 0 <= ny_pos < ny and 0 <= nz_pos < nz):
                neighbor_idx = grid[nx_pos, ny_pos, nz_pos]
                
                if neighbor_idx >= 0 and neighbor_idx > i:  # Avoid duplicate edges
                    neighbor_quat = quaternions_norm[neighbor_idx]
                    
                    # Calculate misorientation angle
                    dot_product = np.abs(np.dot(center_quat, neighbor_quat))
                    dot_product = np.clip(dot_product, 0.0, 1.0)
                    angle_rad = 2 * np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Check if this is a grain boundary
                    if angle_deg >= threshold_deg:
                        boundary_edges.append((i, neighbor_idx))
                        
                        # Create line segment coordinates for visualization
                        coord1 = coords[i]
                        coord2 = coords[neighbor_idx]
                        boundary_coords.extend([coord1, coord2])
    
    boundary_coords = np.array(boundary_coords) if boundary_coords else np.array([])
    
    
    return boundary_edges, boundary_coords

def create_slice_planes(coords, slice_vector, num_slices):
    """
    Create slice planes through the domain along a specified vector.
    For negative vectors, slice positions are reversed to ensure the cut surface faces the camera.
    
    Parameters:
    -----------
    coords : array
        Coordinates of all points
    slice_vector : list
        Direction vector for slicing [x, y, z]
    num_slices : int
        Number of slice planes to create
        
    Returns:
    --------
    slice_positions : array
        Array of slice plane positions along the vector
    slice_normals : array
        Normal vectors for each slice plane
    """
    # Normalize the slice vector
    slice_vector = np.array(slice_vector, dtype=float)
    slice_vector = slice_vector / np.linalg.norm(slice_vector)
    
    # Project all coordinates onto the slice direction
    projections = np.dot(coords, slice_vector)
    
    # Get min and max projections to define slice range
    min_proj = projections.min()
    max_proj = projections.max()
    
    # Check if we have a negative vector (any component is negative)
    # For negative vectors, we reverse the slice order to make the cut surface face the camera
    has_negative_component = np.any(np.array(slice_vector) < 0)
    
    if has_negative_component:
        # For negative vectors, start from max and go to min
        slice_positions = np.linspace(max_proj, min_proj, num_slices)
    else:
        # For positive vectors, start from min and go to max (original behavior)
        slice_positions = np.linspace(min_proj, max_proj, num_slices)
    
    # All slices have the same normal (the slice vector)
    slice_normals = np.tile(slice_vector, (num_slices, 1))
    
    return slice_positions, slice_normals

def create_flythrough_slice(coords, colors, slice_vector, slice_position, slice_thickness=2.0):
    """
    Create a slice of the data at a specific position along the slice vector.
    
    Parameters:
    -----------
    coords : array
        Coordinates of all points
    colors : array
        Colors for all points
    slice_vector : array
        Direction vector for slicing
    slice_position : float
        Position along the slice vector
    slice_thickness : float
        Thickness of the slice
        
    Returns:
    --------
    slice_coords : array
        Coordinates of points in the slice
    slice_colors : array
        Colors of points in the slice
    """
    # Normalize slice vector
    slice_vector = np.array(slice_vector, dtype=float)
    slice_vector = slice_vector / np.linalg.norm(slice_vector)
    
    # Project coordinates onto slice direction
    projections = np.dot(coords, slice_vector)
    
    # Find points within the slice thickness
    half_thickness = slice_thickness / 2.0
    mask = np.abs(projections - slice_position) <= half_thickness
    
    slice_coords = coords[mask]
    slice_colors = colors[mask]
    
    return slice_coords, slice_colors

def create_flythrough_progressive(coords, colors, slice_vector, slice_position):
    """
    Create a progressive fly-through showing all data up to the current slice position.
    This creates a "sectioning" effect where the volume is progressively revealed.
    For negative vectors, the logic is reversed to ensure proper progressive reveal.
    
    Parameters:
    -----------
    coords : array
        Coordinates of all points
    colors : array
        Colors for all points
    slice_vector : array
        Direction vector for slicing
    slice_position : float
        Position along the slice vector (all data up to this position is shown)
        
    Returns:
    --------
    visible_coords : array
        Coordinates of points visible up to the slice position
    visible_colors : array
        Colors of points visible up to the slice position
    """
    # Normalize slice vector
    slice_vector = np.array(slice_vector, dtype=float)
    slice_vector = slice_vector / np.linalg.norm(slice_vector)
    
    # Project coordinates onto slice direction
    projections = np.dot(coords, slice_vector)
    
    # Check if we have a negative vector component
    has_negative_component = np.any(np.array(slice_vector) < 0)
    
    if has_negative_component:
        # For negative vectors, show all points from the slice position and above
        # This creates the progressive reveal effect in the correct direction
        mask = projections >= slice_position
    else:
        # For positive vectors, show all points up to the slice position (original behavior)
        mask = projections <= slice_position
    
    visible_coords = coords[mask]
    visible_colors = colors[mask]
    
    return visible_coords, visible_colors

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
    fig.suptitle(f'IPF Color Key - {direction_label} Direction\\n{crystal_structure.upper()} Crystal Structure', 
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
    if len(kam_values) > 0 and not np.all(np.isnan(kam_values)):
        kam_degrees = np.degrees(kam_values[~np.isnan(kam_values)])
        vmin = 0
        vmax = np.percentile(kam_degrees, 95)
    else:
        vmin, vmax = 0, 5  # Default range
    
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

def create_gnd_color_key(gnd_values, colormap='plasma', output_path=None, vmin=1e12, vmax=1e15):
    """Create GND density colorbar legend with log scale."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Get range from data if values provided
    if len(gnd_values) > 0 and not np.all(np.isnan(gnd_values)):
        valid_values = gnd_values[~np.isnan(gnd_values)]
        if len(valid_values) > 0:
            data_min = np.min(valid_values)
            data_max = np.percentile(valid_values, 95)
            # Use data range but ensure it's reasonable
            vmin = max(data_min, 1e10)
            vmax = min(data_max, 1e17)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 1.5))
    
    # Create log-scale colorbar
    cmap = cm.get_cmap(colormap)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    
    # Set log scale limits
    sm.set_clim(np.log10(vmin), np.log10(vmax))
    
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')
    
    # Set log scale ticks
    log_ticks = np.arange(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax)) + 1)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f'10$^{{{int(t)}}}$' for t in log_ticks])
    
    cbar.set_label('GND Density (m⁻²)', fontsize=12, fontweight='bold')
    
    # Hide axes
    ax.set_visible(False)
    
    # Set title
    fig.suptitle('Geometrically Necessary Dislocation Density Color Scale', fontsize=14, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return output_path
    
    return fig

# Import remaining functions from the original script (subprocess rendering, etc.)
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
import warnings
warnings.filterwarnings('ignore')

# Configure PyVista for subprocess
os.environ['PYVISTA_USE_OSMESA'] = 'true'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYVISTA_USE_PANEL'] = 'false'

pv.OFF_SCREEN = True

def create_method2_rendering(coords, colors, grid_shape, camera_position, title, output_path, 
                           crystal_structure='fcc', direction='z', boundary_coords=None):
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
    
    # Add grain boundary lines if provided
    if boundary_coords is not None and len(boundary_coords) > 0:
        
        try:
            # Create points and lines arrays for PyVista
            n_lines = len(boundary_coords) // 2
            points = boundary_coords.astype(np.float32)
            
            # Create lines connectivity array
            lines_array = []
            for i in range(n_lines):
                lines_array.extend([2, i*2, i*2+1])  # [n_points_in_line, point1_idx, point2_idx]
            
            # Create PolyData
            boundary_mesh = pv.PolyData(points, lines=lines_array)
            
            # Add grain boundary lines to plotter
            plotter.add_mesh(
                boundary_mesh,
                color='black',
                line_width=3.0,
                render_lines_as_tubes=False
            )
            
        except Exception as e:
            pass
    
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

def create_triple_ipf_rendering(coords, colors_x, colors_y, colors_z, grid_shape, camera_position, 
                               title_base, output_path, boundary_coords=None):
    """Create triple IPF rendering with x, y, z directions side by side."""
    nx, ny, nz = grid_shape
    
    directions = ['X', 'Y', 'Z']
    colors_list = [colors_x, colors_y, colors_z]
    
    # Create individual images for each direction
    temp_images = []
    
    for idx, (direction, colors) in enumerate(zip(directions, colors_list)):
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
        
        # Create individual plotter for this direction
        plotter = pv.Plotter(off_screen=True, window_size=(640, 640))
        plotter.background_color = 'white'
        
        plotter.add_mesh(
            thresholded,
            scalars='RGB',
            rgb=True,
            opacity=1.0,
            show_edges=False,
            lighting=False
        )
        
        # Add grain boundary lines if provided
        if boundary_coords is not None and len(boundary_coords) > 0:
            try:
                n_lines = len(boundary_coords) // 2
                points = boundary_coords.astype(np.float32)
                
                lines_array = []
                for i in range(n_lines):
                    lines_array.extend([2, i*2, i*2+1])
                
                boundary_mesh = pv.PolyData(points, lines=lines_array)
                plotter.add_mesh(boundary_mesh, color='black', line_width=2.0, render_lines_as_tubes=False)
            except:
                pass
        
        plotter.add_title(f'IPF {direction}', font_size=14)
        
        # Set camera position
        if camera_position:
            plotter.camera_position = camera_position
        else:
            plotter.camera_position = 'iso'
        
        # Save individual image
        temp_image_path = f"/tmp/triple_ipf_{direction}_{os.getpid()}.png"
        plotter.screenshot(temp_image_path)
        plotter.close()
        temp_images.append(temp_image_path)
    
    # Combine the three images side by side using PIL
    if len(temp_images) == 3:
        from PIL import Image
        
        images = [Image.open(img_path) for img_path in temp_images]
        
        # Calculate combined image size
        width = sum(img.size[0] for img in images)
        height = max(img.size[1] for img in images)
        
        # Create combined image
        combined = Image.new('RGB', (width, height), color='white')
        
        # Paste images side by side
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        # Save combined image
        combined.save(output_path)
        
        # Clean up temporary images
        for temp_path in temp_images:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return output_path
    
    return None

if __name__ == "__main__":
    # Load data from pickle file
    data_file = sys.argv[1]
    triple_ipf_mode = len(sys.argv) > 2 and sys.argv[2] == '--triple-ipf'
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    try:
        if triple_ipf_mode:
            # Extract triple IPF parameters
            coords = data['coords']
            colors_x = data['colors_x']
            colors_y = data['colors_y']
            colors_z = data['colors_z']
            grid_shape = data['grid_shape']
            camera_position = data['camera_position']
            title_base = data['title_base']
            output_path = data['output_path']
            boundary_coords = data.get('boundary_coords', None)
            
            result_path = create_triple_ipf_rendering(coords, colors_x, colors_y, colors_z, 
                                                    grid_shape, camera_position, title_base, 
                                                    output_path, boundary_coords)
        else:
            # Standard single IPF rendering
            coords = data['coords']
            colors = data['colors']
            grid_shape = data['grid_shape']
            camera_position = data['camera_position']
            title = data['title']
            output_path = data['output_path']
            crystal_structure = data.get('crystal_structure', 'fcc')
            direction = data.get('direction', 'z')
            boundary_coords = data.get('boundary_coords', None)
            
            result_path = create_method2_rendering(coords, colors, grid_shape, camera_position, 
                                                 title, output_path, crystal_structure, direction, boundary_coords)
        # Print success message for main script to detect
        print(f"SUCCESS: {result_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
'''
    
    script_path = "/tmp/pyvista_subprocess_renderer_kam_simple.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def render_triple_ipf_with_subprocess(coords, quaternions, grid_shape, camera_position, 
                                     title_base, output_path, crystal_structure=CRYSTAL_STRUCTURE,
                                     boundary_coords=None):
    """Render triple IPF (x, y, z directions) in a single image using subprocess."""
    
    # Generate colors for all three directions
    ipf_colors_x = generate_ipf_colors(quaternions, crystal_structure, 'x')
    ipf_colors_y = generate_ipf_colors(quaternions, crystal_structure, 'y') 
    ipf_colors_z = generate_ipf_colors(quaternions, crystal_structure, 'z')
    
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        data = {
            'coords': coords,
            'colors_x': ipf_colors_x,
            'colors_y': ipf_colors_y,
            'colors_z': ipf_colors_z,
            'grid_shape': grid_shape,
            'camera_position': camera_position,
            'title_base': title_base,
            'output_path': output_path,
            'crystal_structure': crystal_structure,
            'boundary_coords': boundary_coords,
            'background_color': BACKGROUND_COLOR,
            'opacity': OPACITY_FILLED,
            'voxel_size': VOXEL_SIZE,
            'image_size': IMAGE_SIZE,
            'enable_grain_boundaries': ENABLE_GRAIN_BOUNDARIES,
            'grain_boundary_line_width': GRAIN_BOUNDARY_LINE_WIDTH
        }
        pickle.dump(data, f)
        temp_file = f.name
    
    # Create and get the subprocess script
    script_path = create_subprocess_renderer_script()
    
    try:
        # Run subprocess with triple IPF mode
        result = subprocess.run([
            sys.executable, script_path, temp_file, '--triple-ipf'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            return output_path
        else:
            raise RuntimeError(f"Subprocess failed: {result.stderr}")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_file)
        except:
            pass

def render_with_subprocess(coords, colors, grid_shape, camera_position, title, output_path, 
                          crystal_structure=CRYSTAL_STRUCTURE, direction=IPF_DIRECTION, 
                          boundary_coords=None):
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
            'direction': direction,
            'boundary_coords': boundary_coords
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
    """Worker function that processes a single file."""
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
    
    # Calculate simplified KAM if enabled (using expanded region for proper neighborhoods)
    kam_colors = None
    kam_values = None
    if enable_kam and len(coords) > 0:
        try:
            # Use expanded cropping for KAM to ensure proper neighborhoods
            expanded_coords, expanded_quaternions, mask_in_original, offset = apply_expanded_crop_for_kam(
                atoms[['x', 'y', 'z']].values.astype(float),  # Use original coordinates
                atoms[['d1', 'd2', 'd3', 'd4']].values,       # Use original quaternions
                crop_bounds, 
                KAM_NEIGHBOR_RADIUS
            )
            
            
            # Calculate KAM on expanded region, but only for points in original crop region
            expanded_kam_values = calculate_simplified_kam(
                expanded_coords, expanded_quaternions,
                neighbor_radius=KAM_NEIGHBOR_RADIUS, 
                threshold_deg=KAM_THRESHOLD_DEG,
                mask_in_region=mask_in_original
            )
            
            # Extract KAM values for only the original crop region
            kam_values = expanded_kam_values[mask_in_original]
            
            # Convert KAM to colors
            kam_colors = kam_to_colors(kam_values)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            kam_colors = None
            kam_values = None
    
    # Calculate GND density if KAM is available and GND is enabled
    gnd_colors = None
    gnd_values = None
    if ENABLE_GND and kam_values is not None and len(kam_values) > 0:
        try:
            # Calculate GND density from KAM values
            gnd_values = calculate_gnd_density(kam_values, BURGERS_VECTOR_MAGNITUDE, VOXEL_STEP_SIZE)
            
            # Convert GND to colors
            gnd_colors = gnd_to_colors(gnd_values)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            gnd_colors = None
            gnd_values = None
    
    # Detect grain boundaries if enabled
    boundary_coords = None
    if ENABLE_GRAIN_BOUNDARIES and len(coords) > 0:
        try:
            boundary_edges, boundary_coords = detect_grain_boundaries(
                coords, quaternions, 
                threshold_deg=GRAIN_BOUNDARY_THRESHOLD_DEG
            )
        except Exception as e:
            boundary_coords = None
    
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
        'gnd_time': 0.0,
        'method1_image': '',
        'method2_image': '',
        'ipf_image': '',
        'kam_image': '',
        'gnd_image': ''
    }
    
    # Method 2: Structured grid rendering (IPF)
    if enable_method2 and len(coords) > 0:
        method2_start = time.time()
        try:
            if ENABLE_TRIPLE_IPF:
                # Triple IPF rendering (x, y, z directions)
                output_path = os.path.join(output_dir, f"ipf_triple_timestep_{timestep}.png")
                title_base = f"SPPARKS - Timestep {timestep} (atoms: {len(coords)})"
                
                returned_path = render_triple_ipf_with_subprocess(coords, quaternions, (nx, ny, nz), camera_position, 
                                                title_base, output_path, boundary_coords=boundary_coords)
                if returned_path and os.path.exists(returned_path):
                    result['method2_image'] = returned_path
                    result['ipf_image'] = returned_path
                    result['triple_ipf_image'] = returned_path
            else:
                # Standard single IPF rendering
                output_path = os.path.join(output_dir, f"ipf_timestep_{timestep}.png")
                title = f"SPPARKS IPF - Timestep {timestep} - {IPF_DIRECTION.upper()} Direction (atoms: {len(coords)})"
                
                returned_path = render_with_subprocess(coords, ipf_colors, (nx, ny, nz), camera_position, title, output_path, 
                                      boundary_coords=boundary_coords)
                if returned_path and os.path.exists(returned_path):
                    result['method2_image'] = returned_path
                    result['ipf_image'] = returned_path
            
        except Exception as e:
            pass
        
        result['method2_time'] = time.time() - method2_start
    
    # KAM rendering
    if enable_kam and kam_colors is not None and len(coords) > 0:
        kam_start = time.time()
        try:
            output_path = os.path.join(output_dir, f"kam_timestep_{timestep}.png")
            title = f"SPPARKS KAM - Timestep {timestep} - Threshold {KAM_THRESHOLD_DEG}° (atoms: {len(coords)})"
            
            returned_path = render_with_subprocess(coords, kam_colors, (nx, ny, nz), camera_position, title, output_path,
                                  boundary_coords=boundary_coords)
            if returned_path and os.path.exists(returned_path):
                result['kam_image'] = returned_path
            
        except Exception as e:
            pass
        
        result['kam_time'] = time.time() - kam_start
    
    # GND rendering
    if ENABLE_GND and gnd_colors is not None and len(coords) > 0:
        gnd_start = time.time()
        try:
            output_path = os.path.join(output_dir, f"gnd_timestep_{timestep}.png")
            title = f"SPPARKS GND - Timestep {timestep} - Density (m⁻³) (atoms: {len(coords)})"
            
            returned_path = render_with_subprocess(coords, gnd_colors, (nx, ny, nz), camera_position, title, output_path,
                                  boundary_coords=boundary_coords)
            if returned_path and os.path.exists(returned_path):
                result['gnd_image'] = returned_path
            
        except Exception as e:
            pass
        
        result['gnd_time'] = time.time() - gnd_start
    
    return result

def create_flythrough_animation(results_sorted, output_dir, camera_position):
    """Create fly-through animation for the final timestep."""
    if not results_sorted:
        return
    
    # Get the final timestep result
    final_result = results_sorted[-1]
    final_timestep = final_result['timestep']
    
    
    # We need to reload the final timestep data to get coordinates and colors
    final_file_pattern = f"dump.additive8.{final_timestep}"
    final_filepath = None
    
    # Find the final file
    import glob
    dump_files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    for filepath in dump_files:
        if final_file_pattern in os.path.basename(filepath):
            final_filepath = filepath
            break
    
    if not final_filepath:
        return
    
    
    # Load and process the final timestep data
    spparks_data = parse_spparks_dump(final_filepath)
    atoms = spparks_data['atoms']
    bounds = spparks_data['box_bounds']
    
    # Get grid dimensions
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
    
    if len(coords) == 0:
        return
    
    
    # Calculate KAM colors if enabled
    kam_colors = None
    if ENABLE_KAM:
        try:
            # Use expanded cropping for KAM if crop bounds are defined
            if CROP_BOUNDS is not None:
                # Reload original data for expanded KAM calculation
                expanded_coords, expanded_quaternions, mask_in_original, offset = apply_expanded_crop_for_kam(
                    atoms[['x', 'y', 'z']].values.astype(float),
                    atoms[['d1', 'd2', 'd3', 'd4']].values,
                    CROP_BOUNDS, 
                    KAM_NEIGHBOR_RADIUS
                )
                
                expanded_kam_values = calculate_simplified_kam(
                    expanded_coords, expanded_quaternions,
                    neighbor_radius=KAM_NEIGHBOR_RADIUS, 
                    threshold_deg=KAM_THRESHOLD_DEG,
                    mask_in_region=mask_in_original
                )
                
                kam_values = expanded_kam_values[mask_in_original]
            else:
                kam_values = calculate_simplified_kam(
                    coords, quaternions,
                    neighbor_radius=KAM_NEIGHBOR_RADIUS, 
                    threshold_deg=KAM_THRESHOLD_DEG
                )
            
            kam_colors = kam_to_colors(kam_values)
            
        except Exception as e:
            kam_colors = None
    
    # Calculate GND colors if enabled and KAM is available
    gnd_colors = None
    if ENABLE_GND and kam_values is not None and len(kam_values) > 0:
        try:
            # Calculate GND density from KAM values
            gnd_values = calculate_gnd_density(kam_values, BURGERS_VECTOR_MAGNITUDE, VOXEL_STEP_SIZE)
            gnd_colors = gnd_to_colors(gnd_values)
        except Exception as e:
            gnd_colors = None
    
    # Detect grain boundaries if enabled
    boundary_coords = None
    if ENABLE_GRAIN_BOUNDARIES:
        try:
            boundary_edges, boundary_coords = detect_grain_boundaries(
                coords, quaternions, 
                threshold_deg=GRAIN_BOUNDARY_THRESHOLD_DEG
            )
        except Exception as e:
            boundary_coords = None
    
    # Create slice planes
    slice_positions, slice_normals = create_slice_planes(coords, FLYTHROUGH_SLICE_VECTOR, FLYTHROUGH_NUM_SLICES)
    
    
    # Create IPF fly-through frames
    if ENABLE_METHOD2:
        print(f"Creating IPF fly-through frames...")
        ipf_flythrough_frames = []
        for i, slice_pos in tqdm(enumerate(slice_positions), total=len(slice_positions), 
                                 desc="IPF fly-through frames", unit="frame"):
            if ENABLE_TRIPLE_IPF:
                # Triple IPF fly-through
                visible_coords, _ = create_flythrough_progressive(coords, ipf_colors, FLYTHROUGH_SLICE_VECTOR, slice_pos)
                
                if len(visible_coords) > 0:
                    # Get visible quaternions for triple IPF
                    visible_indices = []
                    for vc in visible_coords:
                        for idx, coord in enumerate(coords):
                            if np.allclose(vc, coord):
                                visible_indices.append(idx)
                                break
                    
                    visible_quaternions = quaternions[visible_indices] if len(visible_indices) > 0 else quaternions[:len(visible_coords)]
                    
                    frame_path = os.path.join(output_dir, f"flythrough_ipf_triple_frame_{i:03d}.png")
                    title_base = f"SPPARKS Fly-through - Timestep {final_timestep} - Section {i+1}/{FLYTHROUGH_NUM_SLICES}"
                    
                    try:
                        render_triple_ipf_with_subprocess(visible_coords, visible_quaternions, (nx, ny, nz), camera_position, 
                                                        title_base, frame_path, boundary_coords=boundary_coords)
                        ipf_flythrough_frames.append(frame_path)
                    except Exception as e:
                        pass
            else:
                # Standard single IPF fly-through
                visible_coords, visible_colors = create_flythrough_progressive(coords, ipf_colors, FLYTHROUGH_SLICE_VECTOR, slice_pos)
                
                if len(visible_coords) > 0:
                    frame_path = os.path.join(output_dir, f"flythrough_ipf_frame_{i:03d}.png")
                    title = f"SPPARKS IPF Fly-through - Timestep {final_timestep} - Section {i+1}/{FLYTHROUGH_NUM_SLICES}"
                    
                    try:
                        render_with_subprocess(visible_coords, visible_colors, (nx, ny, nz), camera_position, 
                                             title, frame_path, boundary_coords=boundary_coords)
                        ipf_flythrough_frames.append(frame_path)
                    except Exception as e:
                        pass
        
        # Create IPF fly-through animation
        if ipf_flythrough_frames:
            if ENABLE_TRIPLE_IPF:
                flythrough_path = os.path.join(output_dir, "flythrough_ipf_triple_animation.mp4")
            else:
                flythrough_path = os.path.join(output_dir, "flythrough_ipf_animation.mp4")
            create_animation(ipf_flythrough_frames, flythrough_path, FLYTHROUGH_FPS)
    
    # Create KAM fly-through frames
    if ENABLE_KAM and kam_colors is not None:
        print(f"Creating KAM fly-through frames...")
        kam_flythrough_frames = []
        for i, slice_pos in tqdm(enumerate(slice_positions), total=len(slice_positions), 
                                 desc="KAM fly-through frames", unit="frame"):
            # Use progressive fly-through to show all data up to current slice
            visible_coords, visible_colors = create_flythrough_progressive(coords, kam_colors, FLYTHROUGH_SLICE_VECTOR, slice_pos)
            
            if len(visible_coords) > 0:
                frame_path = os.path.join(output_dir, f"flythrough_kam_frame_{i:03d}.png")
                title = f"SPPARKS KAM Fly-through - Timestep {final_timestep} - Section {i+1}/{FLYTHROUGH_NUM_SLICES}"
                
                try:
                    render_with_subprocess(visible_coords, visible_colors, (nx, ny, nz), camera_position, 
                                         title, frame_path, boundary_coords=boundary_coords)
                    kam_flythrough_frames.append(frame_path)
                except Exception as e:
                    pass
        
        # Create KAM fly-through animation
        if kam_flythrough_frames:
            flythrough_path = os.path.join(output_dir, "flythrough_kam_animation.mp4")
            create_animation(kam_flythrough_frames, flythrough_path, FLYTHROUGH_FPS)
    
    # Create GND fly-through frames
    if ENABLE_GND and gnd_colors is not None:
        print(f"Creating GND fly-through frames...")
        gnd_flythrough_frames = []
        for i, slice_pos in tqdm(enumerate(slice_positions), total=len(slice_positions), 
                                 desc="GND fly-through frames", unit="frame"):
            # Use progressive fly-through to show all data up to current slice
            visible_coords, visible_colors = create_flythrough_progressive(coords, gnd_colors, FLYTHROUGH_SLICE_VECTOR, slice_pos)
            
            if len(visible_coords) > 0:
                frame_path = os.path.join(output_dir, f"flythrough_gnd_frame_{i:03d}.png")
                title = f"SPPARKS GND Fly-through - Timestep {final_timestep} - Section {i+1}/{FLYTHROUGH_NUM_SLICES} - Density (m⁻³)"
                
                try:
                    render_with_subprocess(visible_coords, visible_colors, (nx, ny, nz), camera_position, 
                                         title, frame_path, boundary_coords=boundary_coords)
                    gnd_flythrough_frames.append(frame_path)
                except Exception as e:
                    pass
        
        # Create GND fly-through animation
        if gnd_flythrough_frames:
            flythrough_path = os.path.join(output_dir, "flythrough_gnd_animation.mp4")
            create_animation(gnd_flythrough_frames, flythrough_path, FLYTHROUGH_FPS)

def create_animation(image_files, output_path, fps=5):
    """Create MP4 animation from image files using ffmpeg for multi-core encoding."""
    if not image_files:
        return
    
    try:
        # Use ffmpeg for multi-threaded encoding
        # Create a temporary list file for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for img_path in image_files:
                # Write relative path for ffmpeg concat demuxer
                f.write(f"file '{os.path.abspath(img_path)}'\n")
                f.write(f"duration {1.0/fps}\n")
            # Add last frame duration
            f.write(f"file '{os.path.abspath(image_files[-1])}'\n")
            list_file = f.name
        
        # Run ffmpeg with multi-threading
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            '-preset', 'medium',  # Balance between speed and compression
            '-threads', '0',  # Use all available CPU cores
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temporary file
        os.unlink(list_file)
        
        if result.returncode != 0:
            print(f"Warning: ffmpeg failed, falling back to OpenCV. Error: {result.stderr}")
            # Fallback to OpenCV method
            create_animation_opencv(image_files, output_path, fps)
            
    except (FileNotFoundError, subprocess.SubprocessError):
        print("Warning: ffmpeg not found, using OpenCV (single-threaded)")
        # Fallback to OpenCV method
        create_animation_opencv(image_files, output_path, fps)

def create_animation_opencv(image_files, output_path, fps=5):
    """Fallback OpenCV animation creation (single-threaded)."""
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

def main():
    """Main batch processing function."""
    print("=" * 80)
    print("SPPARKS BATCH ANALYSIS - IPF AND SIMPLIFIED KAM RENDERING")
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
    test_files = dump_files[:]  # Test with 2 files for GND implementation
    print(f"Processing {len(test_files)} files for testing")
    print(f"Range: {os.path.basename(test_files[0])} to {os.path.basename(test_files[-1])}")
    
    # Determine optimal camera position using largest file
    largest_file = max(dump_files, key=lambda x: int(x.split('.')[-1]))
    camera_position = determine_optimal_camera_position(largest_file)
    
    # Check if we should only create fly-through
    if FLYTHROUGH_ONLY and ENABLE_FLYTHROUGH:
        print("=" * 60)
        print("FLY-THROUGH ONLY MODE")
        print("=" * 60)
        print("Skipping timestep processing, creating fly-through animation only...")
        
        # Create mock results with just the final timestep
        final_timestep = int(largest_file.split('.')[-1])
        mock_results = [{'timestep': final_timestep, 'filename': os.path.basename(largest_file)}]
        
        # Create fly-through animation directly
        create_flythrough_animation(mock_results, OUTPUT_DIR, camera_position)
        
        print("=" * 60)
        print("FLY-THROUGH ONLY MODE COMPLETE")
        print("=" * 60)
        print(f"Output directory: {OUTPUT_DIR}")
        if ENABLE_METHOD2:
            print(f"  - IPF fly-through animation: flythrough_ipf_animation.mp4")
        if ENABLE_KAM:
            print(f"  - KAM fly-through animation: flythrough_kam_animation.mp4")
        print("=" * 60)
        return
    
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
        
        print(f"Processing {len(test_files)} files with {num_workers} parallel workers...")
        
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
                                pbar.update(1)
        
        total_time = time.time() - start_time
        
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
                pass
        
        total_time = time.time() - start_time
    
    # Save timing results
    results_df = pd.DataFrame(results)
    timing_file = os.path.join(OUTPUT_DIR, "timing_results.csv")
    results_df.to_csv(timing_file, index=False)
    
    # Create animations
    print(f"Creating animations...")
    
    # Sort results by timestep to ensure proper frame ordering
    results_sorted = sorted(results, key=lambda x: x['timestep'])
    
    # IPF animation (only create if multiple frames available)
    ipf_images = [r['ipf_image'] for r in results_sorted if r['ipf_image']]
    if len(ipf_images) > 1:
        if ENABLE_TRIPLE_IPF:
            animation_path = os.path.join(OUTPUT_DIR, "ipf_triple_animation.mp4")
        else:
            animation_path = os.path.join(OUTPUT_DIR, "ipf_animation.mp4")
        create_animation(ipf_images, animation_path, FPS)
    
    # KAM animation (only create if multiple frames available)
    kam_images = [r['kam_image'] for r in results_sorted if r['kam_image']]
    if len(kam_images) > 1:
        animation_path = os.path.join(OUTPUT_DIR, "kam_animation.mp4")
        create_animation(kam_images, animation_path, FPS)
        
        # Create KAM color key with dummy data
        if ENABLE_KAM:
            kam_key_path = os.path.join(OUTPUT_DIR, f"kam_color_key.png")
            dummy_kam = np.linspace(0, np.radians(5), 1000)  # 0-5 degrees
            create_kam_color_key(dummy_kam, colormap='viridis', output_path=kam_key_path)
    
    # GND animation (only create if multiple frames available)
    gnd_images = [r['gnd_image'] for r in results_sorted if r['gnd_image']]
    if len(gnd_images) > 1:
        animation_path = os.path.join(OUTPUT_DIR, "gnd_animation.mp4")
        create_animation(gnd_images, animation_path, FPS)
        
        # Create GND color key with dummy data
        if ENABLE_GND:
            gnd_key_path = os.path.join(OUTPUT_DIR, f"gnd_color_key.png")
            dummy_gnd = np.logspace(12, 15, 1000)  # 10^12 to 10^15 m^-2
            create_gnd_color_key(dummy_gnd, colormap='plasma', output_path=gnd_key_path)
    
    # Create fly-through animation for final timestep
    if ENABLE_FLYTHROUGH and len(results) > 0:
        create_flythrough_animation(results_sorted, OUTPUT_DIR, camera_position)
    
    # Print summary
    print("=" * 80)
    print("SIMPLIFIED BATCH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total files processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    if len(results) > 0:
        avg_time = total_time / len(results)
        print(f"Average time per file: {avg_time:.2f} seconds")
        
        if PARALLEL_FILES and len(test_files) > 1:
            speedup = len(results) / (total_time / avg_time)
            print(f"Estimated speedup: {speedup:.1f}x (vs serial processing)")
    
    print(f"Output directory: {OUTPUT_DIR}")
    if ipf_images:
        print(f"  - {len(ipf_images)} IPF images")
        if len(ipf_images) > 1:
            if ENABLE_TRIPLE_IPF:
                print(f"  - IPF animation: ipf_triple_animation.mp4")
            else:
                print(f"  - IPF animation: ipf_animation.mp4")
    if kam_images:
        print(f"  - {len(kam_images)} KAM images")
        if len(kam_images) > 1:
            print(f"  - KAM animation: kam_animation.mp4")
        print(f"  - KAM color key: kam_color_key.png")
    if gnd_images:
        print(f"  - {len(gnd_images)} GND images")
        if len(gnd_images) > 1:
            print(f"  - GND animation: gnd_animation.mp4")
        print(f"  - GND color key: gnd_color_key.png")
    
    if ENABLE_FLYTHROUGH:
        print(f"  - IPF fly-through animation: flythrough_ipf_animation.mp4")
        if ENABLE_KAM:
            print(f"  - KAM fly-through animation: flythrough_kam_animation.mp4")
        if ENABLE_GND:
            print(f"  - GND fly-through animation: flythrough_gnd_animation.mp4")
    
    print(f"  - IPF color key: {ipf_key_file}")
    print(f"  - Timing results: timing_results.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()