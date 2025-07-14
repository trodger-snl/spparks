#!/usr/bin/env python3
"""
HDF5 to VTK Converter for Thermal Data

Converts the reduced thermal output HDF5 format to VTK files for visualization.
Each layer is saved as a separate VTK file with temperature data.
"""

import h5py
import numpy as np
import sys
import os
from pathlib import Path

def write_vtk_unstructured_grid(filename, points, cells, point_data=None, cell_data=None):
    """
    Write an unstructured grid to a VTK file in ASCII format.
    
    Args:
        filename: Output VTK filename
        points: Array of 3D coordinates (N x 3)
        cells: Array of tetrahedral connectivity (M x 4)
        point_data: Dictionary of point data arrays
        cell_data: Dictionary of cell data arrays
    """
    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Thermal data from HDF5\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        n_points = len(points)
        f.write(f"POINTS {n_points} double\n")
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        # Cells (tetrahedra)
        n_cells = len(cells)
        f.write(f"CELLS {n_cells} {n_cells * 5}\n")  # 5 = 1 + 4 nodes per tet
        for cell in cells:
            f.write(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}\n")
        
        # Cell types (10 = VTK_TETRA)
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("10\n")
        
        # Point data
        if point_data:
            f.write(f"POINT_DATA {n_points}\n")
            for name, data in point_data.items():
                if data.ndim == 1:
                    f.write(f"SCALARS {name} double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.6f}\n")
                elif data.ndim == 2 and data.shape[1] == 3:
                    f.write(f"VECTORS {name} double\n")
                    for vec in data:
                        f.write(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n")

def extract_layer_data(hdf5_file, layer_idx):
    """
    Extract mesh and temperature data for a specific layer.
    
    Args:
        hdf5_file: Open HDF5 file object
        layer_idx: Layer index to extract
        
    Returns:
        tuple: (points, cells, temperatures, times, data_counts)
    """
    layer_group = hdf5_file[str(layer_idx)]
    
    # Read mesh data
    node_coords = layer_group['nodeCoords'][:]
    elem_to_node = layer_group['elementToNode'][:]
    
    # Read temperature data
    temperatures = layer_group['temperatures'][:]
    times = layer_group['times'][:]
    data_counts = layer_group['dataCounts'][:]
    
    return node_coords, elem_to_node, temperatures, times, data_counts

def interpolate_temperature_at_time(temperatures, times, data_counts, target_time):
    """
    Interpolate temperature at a specific time for all nodes.
    
    Args:
        temperatures: Temperature data array (nodes x max_time_steps)
        times: Time data array (nodes x max_time_steps)  
        data_counts: Number of valid time steps per node
        target_time: Time to interpolate to
        
    Returns:
        Array of interpolated temperatures for each node
    """
    n_nodes = len(data_counts)
    result = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        n_times = data_counts[i]
        if n_times == 0:
            result[i] = 0.0
            continue
            
        node_times = times[i, :n_times]
        node_temps = temperatures[i, :n_times]
        
        # Handle edge cases
        if target_time <= node_times[0]:
            result[i] = node_temps[0]
        elif target_time >= node_times[-1]:
            result[i] = node_temps[-1]
        else:
            # Linear interpolation
            idx = np.searchsorted(node_times, target_time)
            if idx == 0:
                idx = 1
            
            t0, t1 = node_times[idx-1], node_times[idx]
            T0, T1 = node_temps[idx-1], node_temps[idx]
            
            result[i] = T0 + (T1 - T0) * (target_time - t0) / (t1 - t0)
    
    return result

def convert_hdf5_to_vtk(hdf5_filename, output_dir, time_step=None, layers=None):
    """
    Convert HDF5 thermal data to VTK format.
    
    Args:
        hdf5_filename: Input HDF5 file path
        output_dir: Output directory for VTK files
        time_step: Specific time to extract (if None, uses first time step)
        layers: List of layer indices to convert (if None, converts all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_filename, 'r') as f:
        # Get layer times
        layer_times = f['layerTimes'][:]
        print(f"Found {len(layer_times)} layers with times: {layer_times}")
        
        # Determine which layers to process
        if layers is None:
            layers = list(range(len(layer_times)))
        
        # Determine time to extract
        if time_step is None:
            time_step = layer_times[0] + 1.0  # Use a time slightly after first layer
            
        print(f"Extracting temperature data at time {time_step}")
        
        for layer_idx in layers:
            try:
                print(f"Processing layer {layer_idx}...")
                
                # Extract layer data
                points, cells, temperatures, times, data_counts = extract_layer_data(f, layer_idx)
                
                # Interpolate temperatures at target time
                temp_at_time = interpolate_temperature_at_time(temperatures, times, data_counts, time_step)
                
                # Create point data dictionary
                point_data = {
                    'Temperature': temp_at_time,
                    'DataCount': data_counts.astype(float)
                }
                
                # Write VTK file
                vtk_filename = os.path.join(output_dir, f"layer_{layer_idx:03d}_t_{time_step:.3f}.vtk")
                write_vtk_unstructured_grid(vtk_filename, points, cells, point_data)
                
                print(f"  Wrote {len(points)} points, {len(cells)} cells to {vtk_filename}")
                
            except Exception as e:
                print(f"  Error processing layer {layer_idx}: {e}")
                continue

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python hdf5_to_vtk_converter.py <hdf5_file> [output_dir] [time_step]")
        print("Example: python hdf5_to_vtk_converter.py thermal_data.hdf5 vtk_output 50.0")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "vtk_output"
    time_step = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file '{hdf5_file}' not found")
        sys.exit(1)
    
    print(f"Converting {hdf5_file} to VTK format...")
    print(f"Output directory: {output_dir}")
    
    # Convert first few layers as examples
    convert_hdf5_to_vtk(hdf5_file, output_dir, time_step, layers=[0, 1, 2])
    
    print("Conversion complete!")
    print(f"VTK files saved to: {output_dir}")
    print("You can now visualize these files in ParaView or similar VTK viewers.")

if __name__ == "__main__":
    main()