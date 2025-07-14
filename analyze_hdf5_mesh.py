#!/usr/bin/env python3
"""
Analyze HDF5 mesh file to determine element sizes and mesh resolution
"""
import h5py
import numpy as np
import sys

def print_hdf5_structure(name, obj):
    """Recursively print HDF5 file structure"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
        # Print attributes
        for attr_name, attr_value in obj.attrs.items():
            print(f"{indent}  @{attr_name}: {attr_value}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name}: Group")
        # Print attributes
        for attr_name, attr_value in obj.attrs.items():
            print(f"{indent}  @{attr_name}: {attr_value}")

def analyze_mesh_elements(filename):
    """Analyze mesh elements to calculate typical element sizes"""
    print(f"Analyzing HDF5 file: {filename}")
    print("=" * 60)
    
    try:
        with h5py.File(filename, 'r') as f:
            print("FILE STRUCTURE:")
            print("-" * 30)
            f.visititems(print_hdf5_structure)
            
            print("\nFILE ATTRIBUTES:")
            print("-" * 30)
            for attr_name, attr_value in f.attrs.items():
                print(f"@{attr_name}: {attr_value}")
            
            # Look for coordinate and connectivity data
            coords = None
            connectivity = None
            
            # Common dataset names for coordinates
            coord_names = ['coordinates', 'coords', 'nodes', 'vertices', 'points']
            conn_names = ['connectivity', 'elements', 'cells', 'topology']
            
            print("\nSEARCHING FOR MESH DATA:")
            print("-" * 30)
            
            # Search through all datasets
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                    
                    # Try to identify coordinates
                    if any(coord_name in name.lower() for coord_name in coord_names):
                        print(f"  -> Potential coordinates dataset")
                        return name, 'coords'
                    
                    # Try to identify connectivity
                    if any(conn_name in name.lower() for conn_name in conn_names):
                        print(f"  -> Potential connectivity dataset")
                        return name, 'conn'
                    
                    # Check shape to guess data type
                    if len(obj.shape) == 2:
                        if obj.shape[1] in [2, 3]:  # Likely coordinates (2D or 3D)
                            print(f"  -> Likely coordinates (shape suggests {obj.shape[1]}D points)")
                            return name, 'coords'
                        elif obj.shape[1] in [3, 4, 8]:  # Likely connectivity (triangles, tetrahedra, hexes)
                            print(f"  -> Likely connectivity (shape suggests {obj.shape[1]} nodes per element)")
                            return name, 'conn'
                
                return None, None
            
            # Find datasets
            dataset_info = []
            def collect_datasets(name, obj):
                result = find_datasets(name, obj)
                if result[0] is not None:
                    dataset_info.append(result)
            
            f.visititems(collect_datasets)
            
            # Try to load coordinates and connectivity
            coords_data = None
            conn_data = None
            
            for dataset_name, dataset_type in dataset_info:
                if dataset_type == 'coords' and coords_data is None:
                    coords_data = f[dataset_name][:]
                    print(f"\nLoaded coordinates from: {dataset_name}")
                    print(f"Coordinates shape: {coords_data.shape}")
                    print(f"Coordinate range:")
                    for i in range(coords_data.shape[1]):
                        axis_name = ['X', 'Y', 'Z'][i]
                        print(f"  {axis_name}: [{coords_data[:, i].min():.6f}, {coords_data[:, i].max():.6f}]")
                
                elif dataset_type == 'conn' and conn_data is None:
                    conn_data = f[dataset_name][:]
                    print(f"\nLoaded connectivity from: {dataset_name}")
                    print(f"Connectivity shape: {conn_data.shape}")
                    print(f"Connectivity range: [{conn_data.min()}, {conn_data.max()}]")
            
            # Calculate element sizes if we have both coordinates and connectivity
            if coords_data is not None and conn_data is not None:
                print("\nCALCULATING ELEMENT SIZES:")
                print("-" * 30)
                
                # Calculate element sizes (using first few elements as sample)
                element_sizes = []
                num_samples = min(1000, conn_data.shape[0])  # Sample up to 1000 elements
                
                for i in range(num_samples):
                    element_nodes = conn_data[i]
                    # Get coordinates of element nodes
                    element_coords = coords_data[element_nodes]
                    
                    # Calculate distances between all pairs of nodes in element
                    distances = []
                    for j in range(len(element_nodes)):
                        for k in range(j+1, len(element_nodes)):
                            dist = np.linalg.norm(element_coords[j] - element_coords[k])
                            distances.append(dist)
                    
                    # Use maximum distance as element size
                    element_sizes.append(max(distances))
                
                element_sizes = np.array(element_sizes)
                
                print(f"Analyzed {len(element_sizes)} elements")
                print(f"Element size statistics:")
                print(f"  Min size: {element_sizes.min():.6f}")
                print(f"  Max size: {element_sizes.max():.6f}")
                print(f"  Mean size: {element_sizes.mean():.6f}")
                print(f"  Median size: {np.median(element_sizes):.6f}")
                print(f"  Std dev: {element_sizes.std():.6f}")
                
                # Calculate recommended ghost zone width
                max_element_size = element_sizes.max()
                recommended_ghost_width = max_element_size * 2.0  # Conservative factor
                print(f"\nRECOMMENDED GHOST ZONE WIDTH:")
                print(f"  Based on max element size: {recommended_ghost_width:.6f}")
                print(f"  (2x maximum element size for safety)")
                
            else:
                print("\nWARNING: Could not find both coordinates and connectivity data")
                if coords_data is None:
                    print("  - No coordinate data found")
                if conn_data is None:
                    print("  - No connectivity data found")
            
            # Look for any existing mesh resolution metadata
            print("\nLOOKING FOR MESH RESOLUTION METADATA:")
            print("-" * 30)
            
            def search_attributes(name, obj):
                for attr_name, attr_value in obj.attrs.items():
                    attr_lower = attr_name.lower()
                    if any(keyword in attr_lower for keyword in ['size', 'resolution', 'spacing', 'width', 'length']):
                        print(f"Found relevant attribute: {name}@{attr_name} = {attr_value}")
            
            f.visititems(search_attributes)
            
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return

if __name__ == "__main__":
    filename = "/Users/Tron/spparks/examples/ReducedTempAM/unstructured/reduced_thermal_output.hdf5"
    analyze_mesh_elements(filename)