#!/usr/bin/env python3
"""
SPPARKS Quaternion Data Visualization Script

This script reads SPPARKS dump files containing quaternion data (d1-d4 columns)
and provides visualization and analysis using the orix library.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Quaternion, Orientation
from orix.vector import Vector3d
from orix.crystal_map import CrystalMap, Phase
# IPF color key import - will be handled in try/except blocks
import os
import argparse


def read_spparks_dump(filename):
    """
    Read SPPARKS dump file and extract quaternion data.
    
    Parameters:
    filename (str): Path to the SPPARKS dump file
    
    Returns:
    tuple: (pandas.DataFrame, dict) containing the data and metadata
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header information
    metadata = {}
    data_start = 0
    
    for i, line in enumerate(lines):
        if line.startswith('ITEM: TIMESTEP'):
            metadata['timestep'] = float(lines[i+1].split()[0])
            metadata['time'] = float(lines[i+1].split()[1]) if len(lines[i+1].split()) > 1 else None
        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            metadata['n_atoms'] = int(lines[i+1].strip())
        elif line.startswith('ITEM: BOX BOUNDS'):
            bounds = []
            for j in range(3):  # x, y, z bounds
                bound_line = lines[i+1+j].strip().split()
                bounds.append([float(bound_line[0]), float(bound_line[1])])
            metadata['box_bounds'] = bounds
        elif line.startswith('ITEM: ATOMS'):
            # Extract column names
            columns = line.strip().split()[2:]  # Skip 'ITEM:' and 'ATOMS'
            metadata['columns'] = columns
            data_start = i + 1
            break
    
    # Read the data portion
    data_lines = lines[data_start:]
    data = []
    for line in data_lines:
        if line.strip():  # Skip empty lines
            data.append(line.strip().split())
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=metadata['columns'])
    
    # Convert numeric columns
    numeric_columns = ['id', 'i1', 'i2', 'd1', 'd2', 'd3', 'd4', 'd6', 'd8', 'x', 'y', 'z']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, metadata


def process_quaternions(df):
    """
    Process quaternion data using orix library.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing quaternion data in d1-d4 columns
    
    Returns:
    orix.quaternion.Quaternion: Quaternion object
    """
    # Extract quaternion components (assuming d1, d2, d3, d4 are w, x, y, z respectively)
    quat_data = df[['d1', 'd2', 'd3', 'd4']].values
    
    # Create orix Quaternion object
    # Note: orix expects quaternions in [w, x, y, z] format
    quaternions = Quaternion(quat_data)
    
    return quaternions


def create_crystal_map(quaternions, df, z_slice=5):
    """
    Create a CrystalMap from quaternion data for a specific z-slice.
    
    Parameters:
    quaternions (orix.quaternion.Quaternion): Quaternion data
    df (pandas.DataFrame): Original dataframe with spatial information
    z_slice (float): Z-coordinate for the 2D slice
    
    Returns:
    orix.crystal_map.CrystalMap: CrystalMap object
    """
    # Filter data for the specified z-slice (with tolerance)
    z_tolerance = 0.5
    mask = np.abs(df['z'] - z_slice) < z_tolerance
    
    if np.sum(mask) == 0:
        print(f"Warning: No data found at z={z_slice}. Available z-range: {df['z'].min():.1f} to {df['z'].max():.1f}")
        # Use the closest available z-slice
        z_slice = df['z'].unique()[np.argmin(np.abs(df['z'].unique() - z_slice))]
        mask = np.abs(df['z'] - z_slice) < z_tolerance
        print(f"Using z={z_slice} instead")
    
    # Extract slice data
    slice_df = df[mask].copy()
    slice_quaternions = quaternions[mask]
    
    # Create Inconel 625 phase (gamma phase has FCC structure)
    inconel_phase = Phase(
        name="gamma-Inconel625",
        space_group=225  # Fm-3m (face-centered cubic)
    )
    
    # Create orientations from quaternions
    orientations = Orientation(slice_quaternions)
    
    # Get coordinates
    x = slice_df['x'].values
    y = slice_df['y'].values
    
    # Create phase ID array
    n_points = len(slice_df)
    phase_ids = np.ones(n_points, dtype=int)
    
    # Create phase list properly
    phase_list = [inconel_phase]
    
    # Create the crystal map with simplified approach
    try:
        from orix.crystal_map import PhaseList
        phase_list_obj = PhaseList(phases=phase_list, ids=[1])
        
        xmap = CrystalMap(
            rotations=orientations,
            phase_id=phase_ids,
            x=x,
            y=y,
            phase_list=phase_list_obj,
            scan_unit="um"
        )
    except Exception as e:
        print(f"CrystalMap creation with PhaseList failed: {e}")
        # Try alternative approach
        try:
            xmap = CrystalMap(
                rotations=orientations,
                phase_id=phase_ids,
                x=x,
                y=y
            )
            # Manually add the phase
            xmap.phases.add(inconel_phase)
        except Exception as e2:
            print(f"Alternative CrystalMap creation failed: {e2}")
            raise e2
    
    return xmap


def visualize_quaternions(quaternions, df, output_dir='plots', z_slice=5.0):
    """
    Create visualizations of quaternion data.
    
    Parameters:
    quaternions (orix.quaternion.Quaternion): Quaternion data
    df (pandas.DataFrame): Original dataframe with spatial information
    output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Quaternion magnitude distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnitude histogram
    magnitudes = quaternions.norm.data
    axes[0, 0].hist(magnitudes, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Quaternion Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Quaternion Magnitudes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Individual component distributions
    components = ['d1', 'd2', 'd3', 'd4']
    for i, comp in enumerate(components):
        if i < 3:
            row, col = (0, 1) if i == 0 else (1, i-1)
            axes[row, col].hist(df[comp], bins=50, alpha=0.7, label=comp, edgecolor='black')
            axes[row, col].set_xlabel(f'Quaternion Component {comp}')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'Distribution of {comp}')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quaternion_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Spatial distribution of quaternion properties
    if all(col in df.columns for col in ['x', 'y', 'z']):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Color by quaternion magnitude
        scatter = axes[0, 0].scatter(df['x'], df['y'], c=magnitudes, 
                                   cmap='viridis', alpha=0.6, s=1)
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].set_title('Quaternion Magnitude (XY Plane)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Magnitude')
        
        # Color by individual quaternion components
        for i, comp in enumerate(['d1', 'd2', 'd3']):
            row, col = (0, 1) if i == 0 else (1, i-1)
            scatter = axes[row, col].scatter(df['x'], df['y'], c=df[comp], 
                                          cmap='RdBu', alpha=0.6, s=1)
            axes[row, col].set_xlabel('X Position')
            axes[row, col].set_ylabel('Y Position')
            axes[row, col].set_title(f'{comp} Component (XY Plane)')
            plt.colorbar(scatter, ax=axes[row, col], label=comp)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/spatial_quaternion_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Orientation analysis using orix
    try:
        # Convert to orientations for Euler angle calculation
        orientations = Orientation(quaternions)
        
        # Try different methods for Euler angle conversion
        try:
            euler_angles = orientations.to_euler()
        except:
            # Alternative method
            euler_angles = orientations.to_euler_angles()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        euler_labels = ['φ₁', 'Φ', 'φ₂']
        
        # Handle different data formats
        if hasattr(euler_angles, 'data'):
            euler_data = euler_angles.data
        else:
            euler_data = euler_angles
            
        for i in range(3):
            axes[i].hist(euler_data[:, i], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(f'{euler_labels[i]} (radians)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {euler_labels[i]} Euler Angles')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/euler_angle_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Warning: Could not compute Euler angles: {e}")
        print("Skipping Euler angle analysis - orix version compatibility issue")
    
    # 4. CrystalMap with IPF coloring for z=5 slice
    try:
        print("Creating CrystalMap with IPF coloring...")
        xmap = create_crystal_map(quaternions, df, z_slice=z_slice)
        
        # Generate IPF colors for the gamma-Inconel625 phase
        ipf_direction = Vector3d.zvector()  # IPF color along Z direction
        
        # Get the orientations and generate IPF colors
        orientations = xmap.orientations
        rgb_inconel = orientations.get_ipf_color(ipf_direction)
        
        # Create RGB array for all points
        rgb_all = np.zeros((xmap.size, 3))
        rgb_all[xmap.phase_id == 1] = rgb_inconel
        
        # Plot the crystal map with IPF coloring
        fig = xmap.plot(rgb_all, return_figure=True)
        
        # Add IPF color key
        rc = {"font.size": 8}
        with plt.rc_context(rc):  # Temporarily reduce font size
            try:
                # Try to get the phase by name first
                phase_name = list(xmap.phases.names)[0] if xmap.phases.names else "gamma-Inconel625"
                symmetry = xmap.phases[phase_name].point_group if phase_name in xmap.phases.names else None
                
                if symmetry is None:
                    # Use m-3m symmetry for FCC Inconel 625
                    from orix.crystal_map import Phase
                    temp_phase = Phase(space_group=225)  # FCC
                    symmetry = temp_phase.point_group
                
                ax_ipfkey = fig.add_axes(
                    [0.72, 0.87, 0.2, 0.1],
                    projection="ipf",
                    symmetry=symmetry,
                )
                ax_ipfkey.plot_ipf_color_key()
                ax_ipfkey.set_title("IPF-Z", fontsize=8)
            except Exception as key_error:
                print(f"Could not add IPF color key: {key_error}")
                # Add a simple text label instead
                fig.text(0.82, 0.92, "IPF-Z", fontsize=10, ha='center')
        
        # Set main plot title
        fig.suptitle(f'Inconel 625 IPF Coloring (Z-slice at z={z_slice})', fontsize=12)
        
        plt.savefig(f'{output_dir}/inconel625_ipf_map.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Crystal map created with {xmap.size} points at z≈{z_slice}")
        
    except Exception as e:
        print(f"Warning: Could not create crystal map: {e}")
        print("Trying alternative IPF visualization approach...")
        
        # Alternative IPF visualization without CrystalMap
        try:
            # Filter data for the specified z-slice
            z_tolerance = 0.5
            mask = np.abs(df['z'] - z_slice) < z_tolerance
            
            if np.sum(mask) == 0:
                print(f"Warning: No data found at z={z_slice}. Available z-range: {df['z'].min():.1f} to {df['z'].max():.1f}")
                z_slice = df['z'].unique()[np.argmin(np.abs(df['z'].unique() - z_slice))]
                mask = np.abs(df['z'] - z_slice) < z_tolerance
                print(f"Using z={z_slice} instead")
            
            # Extract slice data
            slice_df = df[mask].copy()
            slice_quaternions = quaternions[mask]
            
            # Create orientations for IPF coloring
            orientations = Orientation(slice_quaternions)
            
            # Generate IPF colors along Z direction
            ipf_direction = Vector3d.zvector()
            ipf_colors = orientations.get_ipf_color(ipf_direction)
            
            # Create scatter plot with IPF coloring
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            scatter = ax.scatter(slice_df['x'], slice_df['y'], 
                               c=ipf_colors, s=10, alpha=0.8)
            
            ax.set_xlabel('X Position (μm)')
            ax.set_ylabel('Y Position (μm)')
            ax.set_title(f'Inconel 625 IPF-Z Coloring (Z-slice at z={z_slice})')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add text annotation
            textstr = f'IPF-Z coloring\nFCC structure\n{len(slice_df)} points'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/inconel625_ipf_map.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Alternative IPF visualization created with {len(slice_df)} points at z≈{z_slice}")
            
        except Exception as alt_error:
            print(f"Alternative IPF visualization also failed: {alt_error}")


def main():
    """Main function to run the quaternion visualization script."""
    parser = argparse.ArgumentParser(description='Visualize SPPARKS quaternion data')
    parser.add_argument('filename', help='Path to SPPARKS dump file')
    parser.add_argument('--output-dir', default='plots', 
                       help='Directory to save plots (default: plots)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N random points from dataset for faster processing')
    parser.add_argument('--z-slice', type=float, default=5.0,
                       help='Z-coordinate for 2D crystal map slice (default: 5.0)')
    
    args = parser.parse_args()
    
    print(f"Reading SPPARKS dump file: {args.filename}")
    
    # Read the data
    df, metadata = read_spparks_dump(args.filename)
    
    print(f"Loaded {len(df)} atoms at timestep {metadata['timestep']}")
    print(f"Columns: {metadata['columns']}")
    
    # Sample data if requested
    if args.sample and len(df) > args.sample:
        print(f"Sampling {args.sample} points from {len(df)} total points...")
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Using {len(df)} sampled points")
    
    # Check if quaternion columns exist
    quat_cols = ['d1', 'd2', 'd3', 'd4']
    if not all(col in df.columns for col in quat_cols):
        print(f"Error: Quaternion columns {quat_cols} not found in data")
        return
    
    # Process quaternions
    print("Processing quaternions with orix...")
    quaternions = process_quaternions(df)
    
    print(f"Quaternion statistics:")
    norms = quaternions.norm.data
    print(f"  Mean magnitude: {np.mean(norms):.4f}")
    print(f"  Std magnitude: {np.std(norms):.4f}")
    print(f"  Min magnitude: {np.min(norms):.4f}")
    print(f"  Max magnitude: {np.max(norms):.4f}")
    
    # Create visualizations
    print(f"Creating visualizations in {args.output_dir}/...")
    visualize_quaternions(quaternions, df, args.output_dir, args.z_slice)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()