#!/usr/bin/env python3
"""
SPPARKS Quaternion Data Visualization Script - Simplified Version

This script reads SPPARKS dump files containing quaternion data (d1-d4 columns)
and provides basic visualization that works reliably across orix versions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from orix.quaternion import Quaternion, Orientation
from orix.vector import Vector3d
from orix.crystal_map import Phase
import os
import argparse
import gzip


def read_spparks_dump(filename):
    """Read SPPARKS dump file and extract quaternion data."""
    # Handle gzipped files
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            lines = f.readlines()
    else:
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
    """Process quaternion data using orix library."""
    # Extract quaternion components (assuming d1, d2, d3, d4 are w, x, y, z respectively)
    quat_data = df[['d1', 'd2', 'd3', 'd4']].values
    
    # Create orix Quaternion object
    quaternions = Quaternion(quat_data)
    
    return quaternions


def create_basic_ipf_colors(quaternions):
    """
    Create basic IPF-like colors from quaternions.
    This is a simplified approach when full orix IPF functionality is not available.
    """
    # Use the quaternion components to create RGB colors
    # This is a simplified mapping, not true crystallographic IPF
    
    # Normalize quaternion components for color mapping
    q_data = quaternions.data
    
    # Use the imaginary parts (x, y, z components) for RGB
    # Take absolute values and normalize to [0,1]
    rgb_colors = np.abs(q_data[:, 1:4])  # Skip w component, use x,y,z
    
    # Normalize each color channel
    for i in range(3):
        channel = rgb_colors[:, i]
        if np.max(channel) > np.min(channel):
            rgb_colors[:, i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    
    return rgb_colors


def visualize_quaternions_simple(quaternions, df, output_dir='plots', z_slice=5.0, quiet=False):
    """Create simplified visualizations of quaternion data."""
    os.makedirs(output_dir, exist_ok=True)
    
    def print_if_not_quiet(*args_print, **kwargs):
        if not quiet:
            print(*args_print, **kwargs)
    
    # 1. Quaternion component distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Individual component distributions
    components = ['d1', 'd2', 'd3', 'd4']
    for i, comp in enumerate(components):
        row, col = i // 2, i % 2
        axes[row, col].hist(df[comp], bins=50, alpha=0.7, label=comp, edgecolor='black')
        axes[row, col].set_xlabel(f'Quaternion Component {comp}')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Distribution of {comp}')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quaternion_distributions_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial distribution of quaternion properties
    if all(col in df.columns for col in ['x', 'y', 'z']):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Color by quaternion magnitude
        magnitudes = quaternions.norm.data
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
        plt.savefig(f'{output_dir}/spatial_quaternion_maps_simple.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Simplified IPF-like coloring for z-slice
    try:
        print_if_not_quiet(f"Creating simplified IPF-like visualization for z={z_slice}...")
        
        # Filter data for the specified z-slice
        z_tolerance = 0.5
        mask = np.abs(df['z'] - z_slice) < z_tolerance
        
        if np.sum(mask) == 0:
            print_if_not_quiet(f"Warning: No data found at z={z_slice}. Available z-range: {df['z'].min():.1f} to {df['z'].max():.1f}")
            z_slice = df['z'].unique()[np.argmin(np.abs(df['z'].unique() - z_slice))]
            mask = np.abs(df['z'] - z_slice) < z_tolerance
            print_if_not_quiet(f"Using z={z_slice} instead")
        
        # Extract slice data
        slice_df = df[mask].copy()
        slice_quaternions = quaternions[mask]
        
        # Create simplified IPF-like colors
        ipf_colors = create_basic_ipf_colors(slice_quaternions)
        
        # Create scatter plot with simplified IPF coloring
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        scatter = ax.scatter(slice_df['x'], slice_df['y'], 
                           c=ipf_colors, s=10, alpha=0.8)
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_title(f'Simplified IPF-like Coloring (Z-slice at z={z_slice})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        textstr = f'Simplified IPF-like coloring\nBased on quaternion components\n{len(slice_df)} points'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/simplified_ipf_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print_if_not_quiet(f"Simplified IPF visualization created with {len(slice_df)} points at z≈{z_slice}")
        
        # RGB component analysis
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, color_name in enumerate(['Red', 'Green', 'Blue']):
            axes[i].hist(ipf_colors[:, i], bins=50, alpha=0.7, color=color_name.lower(), edgecolor='black')
            axes[i].set_xlabel(f'{color_name} Component')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Simplified IPF {color_name} Component')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/simplified_ipf_components.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print_if_not_quiet(f"Warning: Could not create simplified IPF visualization: {e}")
    
    # 4. Inverse Pole Figure with pole density
    try:
        print_if_not_quiet("Creating inverse pole figure with pole density...")
        
        # Create Inconel 625 phase (FCC structure)
        inconel_phase = Phase(
            name="gamma-Inconel625",
            space_group=225  # Fm-3m (face-centered cubic)
        )
        
        # Create orientations from quaternions
        orientations = Orientation(quaternions)
        
        # Create figure for IPF plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Try to create pole density plots for different sample directions
        directions = [Vector3d.zvector(), Vector3d.xvector(), Vector3d.yvector()]
        direction_names = ['Z', 'X', 'Y']
        
        for i, (direction, name) in enumerate(zip(directions, direction_names)):
            try:
                # Get IPF colors using the color key
                from orix.plot import IPFColorKeyTSL
                ipf_key = IPFColorKeyTSL(
                    symmetry=inconel_phase.point_group,
                    direction=direction
                )
                
                # Get crystal directions
                v_crystal = (~orientations) * direction
                
                # Get IPF colors
                ipf_colors = ipf_key.orientation2color(orientations)
                
                # Reduce to fundamental zone for plotting
                v_fundamental = v_crystal.in_fundamental_sector(inconel_phase.point_group)
                
                # Create stereographic projection coordinates
                theta = np.arccos(np.clip(v_fundamental.z.data, -1, 1))
                phi = np.arctan2(v_fundamental.y.data, v_fundamental.x.data)
                
                # Stereographic projection (from north pole)
                X = np.sin(theta) * np.cos(phi) / (1 + np.cos(theta) + 1e-10)
                Y = np.sin(theta) * np.sin(phi) / (1 + np.cos(theta) + 1e-10)
                
                # Create hexbin for density
                hb = axes[i].hexbin(X, Y, C=ipf_colors.mean(axis=1), 
                                  gridsize=30, cmap='viridis', 
                                  reduce_C_function=np.mean,
                                  extent=[-1.1, 1.1, -1.1, 1.1])
                
                # Add colorbar
                cb = plt.colorbar(hb, ax=axes[i])
                cb.set_label('Mean orientation', fontsize=8)
                
                # Draw fundamental zone boundary
                axes[i].set_xlim(-1.1, 1.1)
                axes[i].set_ylim(-1.1, 1.1)
                axes[i].set_aspect('equal')
                axes[i].set_title(f'IPF Density - {name} Direction')
                
                # Draw unit circle
                circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
                axes[i].add_patch(circle)
                
            except Exception as ipf_error:
                print_if_not_quiet(f"Could not create IPF for {name} direction: {ipf_error}")
                # Simple fallback - just plot the quaternion components
                try:
                    axes[i].scatter(quaternions.a.data, quaternions.b.data, 
                                  c=quaternions.c.data, s=5, alpha=0.3, cmap='viridis')
                    axes[i].set_xlabel('Q_a')
                    axes[i].set_ylabel('Q_b') 
                    axes[i].set_title(f'Quaternion Components - {name} Direction')
                    axes[i].grid(True, alpha=0.3)
                except Exception as e2:
                    print_if_not_quiet(f"Fallback also failed: {e2}")
                    axes[i].text(0.5, 0.5, f'IPF {name}\nNot Available', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'IPF - {name} Direction')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/inverse_pole_figure_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print_if_not_quiet("Inverse pole figure plots created")
        
    except Exception as e:
        print_if_not_quiet(f"Warning: Could not create inverse pole figure plots: {e}")


def main():
    """Main function to run the simplified quaternion visualization script."""
    parser = argparse.ArgumentParser(description='Visualize SPPARKS quaternion data (simplified version)')
    parser.add_argument('filename', help='Path to SPPARKS dump file')
    parser.add_argument('--output-dir', default='plots_simple', 
                       help='Directory to save plots (default: plots_simple)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N random points from dataset for faster processing')
    parser.add_argument('--z-slice', type=float, default=5.0,
                       help='Z-coordinate for 2D slice visualization (default: 5.0)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output to screen')
    
    args = parser.parse_args()
    
    def print_if_not_quiet(*args_print, **kwargs):
        if not args.quiet:
            print(*args_print, **kwargs)
    
    print_if_not_quiet(f"Reading SPPARKS dump file: {args.filename}")
    
    # Read the data
    df, metadata = read_spparks_dump(args.filename)
    
    print_if_not_quiet(f"Loaded {len(df)} atoms at timestep {metadata['timestep']}")
    print_if_not_quiet(f"Columns: {metadata['columns']}")
    
    # Sample data if requested
    if args.sample and len(df) > args.sample:
        print_if_not_quiet(f"Sampling {args.sample} points from {len(df)} total points...")
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print_if_not_quiet(f"Using {len(df)} sampled points")
    
    # Check if quaternion columns exist
    quat_cols = ['d1', 'd2', 'd3', 'd4']
    if not all(col in df.columns for col in quat_cols):
        print_if_not_quiet(f"Error: Quaternion columns {quat_cols} not found in data")
        return
    
    # Process quaternions
    print_if_not_quiet("Processing quaternions with orix...")
    quaternions = process_quaternions(df)
    
    print_if_not_quiet(f"Quaternion statistics:")
    norms = quaternions.norm.data
    print_if_not_quiet(f"  Mean magnitude: {np.mean(norms):.4f}")
    print_if_not_quiet(f"  Std magnitude: {np.std(norms):.4f}")
    print_if_not_quiet(f"  Min magnitude: {np.min(norms):.4f}")
    print_if_not_quiet(f"  Max magnitude: {np.max(norms):.4f}")
    
    # Create visualizations
    print_if_not_quiet(f"Creating simplified visualizations in {args.output_dir}/...")
    visualize_quaternions_simple(quaternions, df, args.output_dir, args.z_slice, args.quiet)
    
    print_if_not_quiet("Simple visualization complete!")
    print_if_not_quiet(f"Generated plots in {args.output_dir}/:")
    if not args.quiet:
        for file in os.listdir(args.output_dir):
            if file.endswith('.png'):
                print_if_not_quiet(f"  - {file}")


if __name__ == "__main__":
    main()