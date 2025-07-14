#!/usr/bin/env python3
"""
Advanced HDF5 to VTK Converter for Thermal Data

Enhanced version that can create time series animations and handle multiple output formats.
Supports both single timestep and time series conversion.
"""

import h5py
import numpy as np
import sys
import os
import argparse
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

def write_pvd_collection(pvd_filename, vtk_files_info):
    """
    Write a ParaView Data (PVD) collection file for time series animation.
    
    Args:
        pvd_filename: Output PVD filename
        vtk_files_info: List of (time, filename) tuples
    """
    with open(pvd_filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        
        for time_val, filename in vtk_files_info:
            rel_path = os.path.basename(filename)
            f.write(f'    <DataSet timestep="{time_val}" file="{rel_path}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

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

def get_temperature_range(hdf5_file, layer_indices=None):
    """
    Calculate the global temperature range across all layers for consistent coloring.
    
    Args:
        hdf5_file: Open HDF5 file object
        layer_indices: List of layer indices to analyze (None for all)
        
    Returns:
        tuple: (min_temp, max_temp)
    """
    if layer_indices is None:
        layer_times = hdf5_file['layerTimes'][:]
        layer_indices = list(range(len(layer_times)))
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for layer_idx in layer_indices[:5]:  # Sample first few layers for speed
        try:
            layer_group = hdf5_file[str(layer_idx)]
            temperatures = layer_group['temperatures'][:]
            data_counts = layer_group['dataCounts'][:]
            
            for i, count in enumerate(data_counts):
                if count > 0:
                    temps = temperatures[i, :count]
                    global_min = min(global_min, np.min(temps))
                    global_max = max(global_max, np.max(temps))
        except:
            continue
    
    return global_min, global_max

def convert_layer_timeseries(hdf5_file, layer_idx, output_dir, time_points, 
                           temp_min=None, temp_max=None):
    """
    Convert a single layer to multiple VTK files for time series animation.
    
    Args:
        hdf5_file: Open HDF5 file object
        layer_idx: Layer index to convert
        output_dir: Output directory
        time_points: List of time points to extract
        temp_min, temp_max: Temperature range for normalization
        
    Returns:
        List of (time, filename) tuples for PVD file
    """
    print(f"Processing layer {layer_idx} time series...")
    
    # Extract layer data once
    points, cells, temperatures, times, data_counts = extract_layer_data(hdf5_file, layer_idx)
    
    vtk_files = []
    
    for time_val in time_points:
        try:
            # Interpolate temperatures at this time
            temp_at_time = interpolate_temperature_at_time(temperatures, times, data_counts, time_val)
            
            # Normalize temperatures if range provided
            if temp_min is not None and temp_max is not None:
                temp_normalized = (temp_at_time - temp_min) / (temp_max - temp_min)
            else:
                temp_normalized = temp_at_time
            
            # Create point data
            point_data = {
                'Temperature': temp_at_time,
                'Temperature_Normalized': temp_normalized,
                'DataCount': data_counts.astype(float)
            }
            
            # Write VTK file
            vtk_filename = os.path.join(output_dir, f"layer_{layer_idx:03d}_t_{time_val:06.2f}.vtk")
            write_vtk_unstructured_grid(vtk_filename, points, cells, point_data)
            
            vtk_files.append((time_val, vtk_filename))
            
        except Exception as e:
            print(f"  Warning: Failed to process time {time_val}: {e}")
            continue
    
    return vtk_files

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Convert HDF5 thermal data to VTK format')
    parser.add_argument('hdf5_file', help='Input HDF5 file')
    parser.add_argument('-o', '--output', default='vtk_output', help='Output directory')
    parser.add_argument('-l', '--layers', nargs='+', type=int, help='Layer indices to convert (default: 0,1,2)')
    parser.add_argument('-t', '--time', type=float, help='Single time point to extract')
    parser.add_argument('--timeseries', action='store_true', help='Create time series animation')
    parser.add_argument('--time-start', type=float, default=0.0, help='Start time for series')
    parser.add_argument('--time-end', type=float, default=100.0, help='End time for series')
    parser.add_argument('--time-step', type=float, default=5.0, help='Time step for series')
    parser.add_argument('--normalize', action='store_true', help='Normalize temperatures globally')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file '{args.hdf5_file}' not found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Converting {args.hdf5_file} to VTK format...")
    print(f"Output directory: {args.output}")
    
    with h5py.File(args.hdf5_file, 'r') as f:
        # Get layer times
        layer_times = f['layerTimes'][:]
        print(f"Found {len(layer_times)} layers")
        
        # Determine layers to process
        if args.layers is None:
            layers = [0, 1, 2]  # Default to first 3 layers
        else:
            layers = args.layers
        
        # Get temperature range for normalization
        temp_min, temp_max = None, None
        if args.normalize:
            print("Calculating global temperature range...")
            temp_min, temp_max = get_temperature_range(f, layers)
            print(f"Temperature range: {temp_min:.2f} to {temp_max:.2f}")
        
        if args.timeseries:
            # Create time series
            time_points = np.arange(args.time_start, args.time_end + args.time_step, args.time_step)
            print(f"Creating time series: {len(time_points)} time points")
            
            all_vtk_files = []
            
            for layer_idx in layers:
                vtk_files = convert_layer_timeseries(f, layer_idx, args.output, time_points, 
                                                   temp_min, temp_max)
                all_vtk_files.extend(vtk_files)
            
            # Create PVD collection file for animation
            pvd_filename = os.path.join(args.output, "thermal_animation.pvd")
            write_pvd_collection(pvd_filename, sorted(all_vtk_files))
            print(f"Created animation collection: {pvd_filename}")
            
        else:
            # Single time point
            time_point = args.time if args.time is not None else (layer_times[0] + 10.0)
            print(f"Extracting single time point: {time_point}")
            
            for layer_idx in layers:
                try:
                    points, cells, temperatures, times, data_counts = extract_layer_data(f, layer_idx)
                    temp_at_time = interpolate_temperature_at_time(temperatures, times, data_counts, time_point)
                    
                    point_data = {'Temperature': temp_at_time}
                    if args.normalize and temp_min is not None and temp_max is not None:
                        temp_normalized = (temp_at_time - temp_min) / (temp_max - temp_min)
                        point_data['Temperature_Normalized'] = temp_normalized
                    
                    vtk_filename = os.path.join(args.output, f"layer_{layer_idx:03d}_t_{time_point:.3f}.vtk")
                    write_vtk_unstructured_grid(vtk_filename, points, cells, point_data)
                    
                    print(f"  Layer {layer_idx}: {len(points)} points, {len(cells)} cells -> {vtk_filename}")
                    
                except Exception as e:
                    print(f"  Error processing layer {layer_idx}: {e}")
    
    print("Conversion complete!")
    print("You can now visualize these files in ParaView or similar VTK viewers.")
    if args.timeseries:
        print("Load the .pvd file in ParaView for time series animation.")

if __name__ == "__main__":
    main()