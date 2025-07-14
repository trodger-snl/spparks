#!/usr/bin/env python3
"""
Test script for SPPARKS IPF volume rendering with PyVista
Fixes the black surfaces issue on +x, +y, +z faces
"""

import numpy as np
import pandas as pd
import pyvista as pv
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Quaternion
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "/Users/Tron/spparks/examples/ReducedTempAM/MisOrientTest/DumpFiles/dump.additive8.13"
IPF_DIRECTION = 'z'
CRYSTAL_STRUCTURE = 'fcc'
OPACITY_FILLED = 1.0
BACKGROUND_COLOR = 'white'
VOXEL_SIZE = 1.0

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

def create_working_volume_rendering(coords, colors, grid_shape, method='points'):
    """
    Create volume rendering that properly preserves colors.
    
    Methods:
    - 'points': Use PolyData with points (most reliable)
    - 'structured': Use StructuredGrid
    """
    nx, ny, nz = grid_shape
    print(f"\nUsing {method} method for rendering...")
    
    plotter = pv.Plotter()
    plotter.background_color = BACKGROUND_COLOR
    
    if method == 'points':
        # Method 1: Create point cloud with glyphs
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
        
        plotter.add_mesh(
            glyphs,
            scalars='RGB',
            rgb=True,
            opacity=OPACITY_FILLED,
            show_edges=False,
            lighting=False  # Disable lighting to show pure colors
        )
        
    elif method == 'structured':
        # Method 2: Create StructuredGrid with proper color mapping
        # Create full grid
        x = np.arange(nx + 1)
        y = np.arange(ny + 1) 
        z = np.arange(nz + 1)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        
        grid = pv.StructuredGrid(x, y, z)
        
        # Create cell data array
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
        
        plotter.add_mesh(
            thresholded,
            scalars='RGB',
            rgb=True,
            opacity=OPACITY_FILLED,
            show_edges=False,
            lighting=False
        )
        
    
    # Add title and axes
    plotter.add_title(
        f'SPPARKS Microstructure - IPF {IPF_DIRECTION.upper()}-direction ({method})',
        font_size=16
    )
    plotter.add_axes()
    plotter.show_grid()
    
    return plotter

def main():
    """Main function to test different rendering approaches."""
    print("Loading SPPARKS data...")
    spparks_data = parse_spparks_dump(DATA_FILE)
    
    # Extract data
    atoms = spparks_data['atoms']
    bounds = spparks_data['box_bounds']
    
    # Get grid dimensions
    nx = bounds[0][1] - bounds[0][0] + 1
    ny = bounds[1][1] - bounds[1][0] + 1
    nz = bounds[2][1] - bounds[2][0] + 1
    
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Number of atoms: {len(atoms)}")
    
    # Extract coordinates and quaternions
    coords = atoms[['x', 'y', 'z']].values.astype(float)
    quaternions = atoms[['d1', 'd2', 'd3', 'd4']].values
    
    # Generate IPF colors
    print("\nGenerating IPF colors...")
    ipf_colors = generate_ipf_colors(quaternions, CRYSTAL_STRUCTURE, IPF_DIRECTION)
    print(f"Generated {len(ipf_colors)} IPF colors")
    print(f"Color range: [{ipf_colors.min():.3f}, {ipf_colors.max():.3f}]")
    
    # Test different rendering methods
    print("\n" + "="*60)
    print("Testing different rendering methods...")
    print("="*60)
    
    # Method 1: Point cloud with glyphs (most reliable)
    print("\n1. Point cloud with glyphs method:")
    plotter1 = create_working_volume_rendering(coords, ipf_colors, (nx, ny, nz), method='points')
    plotter1.show()
    
    # Method 2: Structured grid
    print("\n2. Structured grid method:")
    plotter2 = create_working_volume_rendering(coords, ipf_colors, (nx, ny, nz), method='structured')
    plotter2.show()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("Both methods should show proper colors on all faces.")
    print("="*60)

if __name__ == "__main__":
    main()