#!/usr/bin/env python3
"""
Debug script to test PyVista 3D image/volume capabilities.
This will help find the correct way to create solid 3D visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt

def test_pyvista_3d_capabilities():
    """Test different PyVista 3D volume creation methods."""
    
    try:
        import pyvista as pv
        print(f"✓ PyVista version: {pv.__version__}")
        
        # Create sample 3D data
        nx, ny, nz = 50, 50, 50
        print(f"Creating test volume: {nx}x{ny}x{nz}")
        
        # Create sample 3D scalar data (like a sphere)
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny) 
        z = np.linspace(-2, 2, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create a sphere of data
        radius = np.sqrt(X**2 + Y**2 + Z**2)
        sphere_data = (radius < 1.5).astype(float)
        
        # Create RGB colors (simple gradient)
        colors = np.zeros((nx, ny, nz, 3))
        colors[:, :, :, 0] = X / 2 + 0.5  # Red channel
        colors[:, :, :, 1] = Y / 2 + 0.5  # Green channel
        colors[:, :, :, 2] = Z / 2 + 0.5  # Blue channel
        colors = np.clip(colors, 0, 1)
        
        print(f"✓ Created test data with shape: {sphere_data.shape}")
        
        # Test Method 1: UniformGrid (if available)
        print("\n=== Testing UniformGrid ===")
        try:
            grid = pv.UniformGrid(dimensions=(nx, ny, nz))
            print("✓ UniformGrid available")
            test_uniform_grid(grid, sphere_data, colors)
        except AttributeError:
            print("✗ UniformGrid not available")
        
        # Test Method 2: ImageData
        print("\n=== Testing ImageData ===")
        try:
            image = pv.ImageData(dimensions=(nx, ny, nz))
            print("✓ ImageData available")
            test_image_data(image, sphere_data, colors)
        except Exception as e:
            print(f"✗ ImageData failed: {e}")
        
        # Test Method 3: StructuredGrid
        print("\n=== Testing StructuredGrid ===")
        try:
            test_structured_grid(X, Y, Z, sphere_data, colors)
        except Exception as e:
            print(f"✗ StructuredGrid failed: {e}")
        
        # Test Method 4: PolyData with points
        print("\n=== Testing PolyData Point Cloud ===")
        try:
            test_polydata_points(X, Y, Z, sphere_data, colors)
        except Exception as e:
            print(f"✗ PolyData failed: {e}")
        
        # Test Method 5: Multiple cubes approach
        print("\n=== Testing Individual Cubes ===")
        try:
            test_individual_cubes(X, Y, Z, sphere_data, colors)
        except Exception as e:
            print(f"✗ Individual cubes failed: {e}")
            
    except ImportError as e:
        print(f"✗ PyVista import failed: {e}")
        return False
    
    return True

def test_uniform_grid(grid, data, colors):
    """Test UniformGrid approach."""
    try:
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        
        # Add scalar data
        grid.point_data['values'] = data.ravel()
        
        # Add RGB data
        rgb_flat = colors.reshape(-1, 3)
        grid.point_data['RGB'] = rgb_flat
        
        # Test visualization
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars='values', opacity=0.8)
        plotter.screenshot('/Users/Tron/spparks/test_uniform_grid.png')
        
        print("✓ UniformGrid visualization successful")
        return grid
        
    except Exception as e:
        print(f"✗ UniformGrid test failed: {e}")
        return None

def test_image_data(image, data, colors):
    """Test ImageData approach."""
    try:
        image.spacing = (1.0, 1.0, 1.0)
        image.origin = (0.0, 0.0, 0.0)
        
        # Add scalar data
        image.point_data['values'] = data.ravel()
        
        # Add RGB data
        rgb_flat = colors.reshape(-1, 3)
        image.point_data['RGB'] = rgb_flat
        
        # Test thresholding
        thresh = image.threshold(0.5, scalars='values')
        
        # Test visualization
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(thresh, scalars=thresh.point_data['RGB'], rgb=True, opacity=0.8)
        plotter.screenshot('/Users/Tron/spparks/test_image_data.png')
        
        print("✓ ImageData visualization successful")
        print(f"  Original points: {image.n_points}")
        print(f"  After threshold: {thresh.n_points}")
        return image, thresh
        
    except Exception as e:
        print(f"✗ ImageData test failed: {e}")
        return None

def test_structured_grid(X, Y, Z, data, colors):
    """Test StructuredGrid approach."""
    try:
        import pyvista as pv
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add data
        grid.point_data['values'] = data.ravel()
        rgb_flat = colors.reshape(-1, 3)
        grid.point_data['RGB'] = rgb_flat
        
        # Test visualization
        plotter = pv.Plotter(off_screen=True)
        filled = grid.threshold(0.5, scalars='values')
        plotter.add_mesh(filled, scalars=filled.point_data['RGB'], rgb=True, opacity=0.8)
        plotter.screenshot('/Users/Tron/spparks/test_structured_grid.png')
        
        print("✓ StructuredGrid visualization successful")
        return grid
        
    except Exception as e:
        print(f"✗ StructuredGrid test failed: {e}")
        return None

def test_polydata_points(X, Y, Z, data, colors):
    """Test PolyData point cloud approach."""
    try:
        import pyvista as pv
        
        # Get filled points only
        filled_mask = data > 0.5
        filled_points = np.column_stack([
            X[filled_mask],
            Y[filled_mask], 
            Z[filled_mask]
        ])
        filled_colors = colors[filled_mask]
        
        # Create point cloud
        cloud = pv.PolyData(filled_points)
        cloud.point_data['RGB'] = filled_colors
        
        # Test visualization
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            cloud,
            scalars=filled_colors,
            rgb=True,
            point_size=10.0,
            render_points_as_spheres=True,
            opacity=0.8
        )
        plotter.screenshot('/Users/Tron/spparks/test_point_cloud.png')
        
        print(f"✓ PolyData point cloud successful ({len(filled_points)} points)")
        return cloud
        
    except Exception as e:
        print(f"✗ PolyData test failed: {e}")
        return None

def test_individual_cubes(X, Y, Z, data, colors):
    """Test individual cube approach."""
    try:
        import pyvista as pv
        
        # Get filled points (sample for performance)
        filled_mask = data > 0.5
        filled_indices = np.where(filled_mask)
        
        # Sample subset
        n_cubes = min(1000, len(filled_indices[0]))
        sample_indices = np.random.choice(len(filled_indices[0]), n_cubes, replace=False)
        
        all_meshes = []
        all_colors = []
        
        for idx in sample_indices:
            i, j, k = filled_indices[0][idx], filled_indices[1][idx], filled_indices[2][idx]
            x, y, z = X[i, j, k], Y[i, j, k], Z[i, j, k]
            color = colors[i, j, k]
            
            # Create cube
            cube = pv.Cube(center=(x, y, z), x_length=0.8, y_length=0.8, z_length=0.8)
            all_meshes.append(cube)
            all_colors.extend([color] * cube.n_points)
        
        # Combine meshes
        if all_meshes:
            combined = all_meshes[0]
            for mesh in all_meshes[1:]:
                combined = combined.merge(mesh)
            
            combined.point_data['RGB'] = np.array(all_colors)
            
            # Test visualization
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(combined, scalars=combined.point_data['RGB'], rgb=True, opacity=0.8)
            plotter.screenshot('/Users/Tron/spparks/test_cubes.png')
            
            print(f"✓ Individual cubes successful ({len(all_meshes)} cubes)")
            return combined
        
    except Exception as e:
        print(f"✗ Individual cubes test failed: {e}")
        return None

def create_working_spparks_visualizer():
    """Create a working function for SPPARKS data visualization."""
    
    print("\n" + "="*60)
    print("CREATING WORKING SPPARKS VISUALIZER")
    print("="*60)
    
    def visualize_spparks_3d(coords, colors, grain_ids, grid_shape, method='auto'):
        """
        Visualize SPPARKS data as solid 3D volume.
        
        Args:
            coords: (N, 3) voxel coordinates
            colors: (N, 3) RGB colors [0, 1]
            grain_ids: (N,) grain identifiers
            grid_shape: (nx, ny, nz) grid dimensions
            method: 'auto', 'image', 'points', 'cubes'
        
        Returns:
            pv.Plotter: Ready-to-show plotter
        """
        import pyvista as pv
        
        nx, ny, nz = grid_shape
        
        if method == 'auto':
            # Try methods in order of preference
            methods_to_try = ['image', 'points', 'cubes']
        else:
            methods_to_try = [method]
        
        for method_name in methods_to_try:
            try:
                if method_name == 'image':
                    return _create_image_visualization(coords, colors, grain_ids, grid_shape)
                elif method_name == 'points':
                    return _create_point_visualization(coords, colors, grain_ids)
                elif method_name == 'cubes':
                    return _create_cube_visualization(coords, colors, grain_ids)
            except Exception as e:
                print(f"Method {method_name} failed: {e}")
                continue
        
        raise ValueError("All visualization methods failed")
    
    def _create_image_visualization(coords, colors, grain_ids, grid_shape):
        """Use ImageData for volume visualization."""
        import pyvista as pv
        
        nx, ny, nz = grid_shape
        
        # Create 3D image
        image = pv.ImageData(dimensions=(nx, ny, nz))
        image.spacing = (1.0, 1.0, 1.0)
        image.origin = (0.0, 0.0, 0.0)
        
        # Initialize arrays
        filled_array = np.zeros(nx * ny * nz)
        rgb_array = np.zeros((nx * ny * nz, 3))
        grain_array = np.zeros(nx * ny * nz)
        
        # Fill data
        linear_indices = coords[:, 0] * ny * nz + coords[:, 1] * nz + coords[:, 2]
        filled_array[linear_indices] = 1.0
        rgb_array[linear_indices] = colors
        grain_array[linear_indices] = grain_ids
        
        # Add to image
        image.point_data['Filled'] = filled_array
        image.point_data['RGB'] = rgb_array
        image.point_data['GrainID'] = grain_array
        
        # Threshold to get solid volume
        solid = image.threshold(0.5, scalars='Filled')
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(
            solid,
            scalars=solid.point_data['RGB'],
            rgb=True,
            opacity=0.8,
            show_edges=False
        )
        plotter.add_title('SPPARKS 3D Microstructure (ImageData)')
        plotter.add_axes()
        
        return plotter
    
    def _create_point_visualization(coords, colors, grain_ids):
        """Use point cloud for visualization."""
        import pyvista as pv
        
        # Create point cloud
        points = coords.astype(float)
        cloud = pv.PolyData(points)
        cloud.point_data['RGB'] = colors
        cloud.point_data['GrainID'] = grain_ids
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(
            cloud,
            scalars=colors,
            rgb=True,
            point_size=8.0,
            render_points_as_spheres=True,
            opacity=0.9
        )
        plotter.add_title('SPPARKS 3D Microstructure (Point Cloud)')
        plotter.add_axes()
        
        return plotter
    
    def _create_cube_visualization(coords, colors, grain_ids, max_cubes=5000):
        """Use individual cubes for visualization."""
        import pyvista as pv
        
        # Sample for performance
        n_cubes = min(len(coords), max_cubes)
        indices = np.random.choice(len(coords), n_cubes, replace=False)
        
        all_meshes = []
        all_colors = []
        
        for idx in indices:
            x, y, z = coords[idx]
            color = colors[idx]
            
            cube = pv.Cube(
                center=(x + 0.5, y + 0.5, z + 0.5),
                x_length=0.95, y_length=0.95, z_length=0.95
            )
            all_meshes.append(cube)
            all_colors.extend([color] * cube.n_points)
        
        # Combine
        combined = all_meshes[0]
        for mesh in all_meshes[1:]:
            combined = combined.merge(mesh)
        
        combined.point_data['RGB'] = np.array(all_colors)
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(
            combined,
            scalars=combined.point_data['RGB'],
            rgb=True,
            opacity=0.8,
            show_edges=False
        )
        plotter.add_title(f'SPPARKS 3D Microstructure (Cubes, {n_cubes} voxels)')
        plotter.add_axes()
        
        return plotter
    
    return visualize_spparks_3d

if __name__ == "__main__":
    print("Testing PyVista 3D capabilities...")
    
    success = test_pyvista_3d_capabilities()
    
    if success:
        # Create the working function
        visualize_func = create_working_spparks_visualizer()
        
        print("\n" + "="*60)
        print("SUCCESS: Working SPPARKS visualizer created!")
        print("="*60)
        print("Usage:")
        print("  plotter = visualize_spparks_3d(coords, colors, grain_ids, grid_shape)")
        print("  plotter.show()")
        
        # Save the function to a separate file
        with open('/Users/Tron/spparks/spparks_visualizer.py', 'w') as f:
            f.write("""
# Working SPPARKS 3D visualizer
import numpy as np
import pyvista as pv

def visualize_spparks_3d(coords, colors, grain_ids, grid_shape, method='auto'):
    '''
    Visualize SPPARKS data as solid 3D volume.
    
    Args:
        coords: (N, 3) voxel coordinates
        colors: (N, 3) RGB colors [0, 1]
        grain_ids: (N,) grain identifiers
        grid_shape: (nx, ny, nz) grid dimensions
        method: 'auto', 'image', 'points', 'cubes'
    
    Returns:
        pv.Plotter: Ready-to-show plotter
    '''
    nx, ny, nz = grid_shape
    
    if method == 'auto':
        methods_to_try = ['image', 'points', 'cubes']
    else:
        methods_to_try = [method]
    
    for method_name in methods_to_try:
        try:
            if method_name == 'image':
                return _create_image_visualization(coords, colors, grain_ids, grid_shape)
            elif method_name == 'points':
                return _create_point_visualization(coords, colors, grain_ids)
            elif method_name == 'cubes':
                return _create_cube_visualization(coords, colors, grain_ids)
        except Exception as e:
            print(f"Method {method_name} failed: {e}")
            continue
    
    raise ValueError("All visualization methods failed")

def _create_image_visualization(coords, colors, grain_ids, grid_shape):
    nx, ny, nz = grid_shape
    
    image = pv.ImageData(dimensions=(nx, ny, nz))
    image.spacing = (1.0, 1.0, 1.0)
    image.origin = (0.0, 0.0, 0.0)
    
    filled_array = np.zeros(nx * ny * nz)
    rgb_array = np.zeros((nx * ny * nz, 3))
    grain_array = np.zeros(nx * ny * nz)
    
    linear_indices = coords[:, 0] * ny * nz + coords[:, 1] * nz + coords[:, 2]
    filled_array[linear_indices] = 1.0
    rgb_array[linear_indices] = colors
    grain_array[linear_indices] = grain_ids
    
    image.point_data['Filled'] = filled_array
    image.point_data['RGB'] = rgb_array
    image.point_data['GrainID'] = grain_array
    
    solid = image.threshold(0.5, scalars='Filled')
    
    plotter = pv.Plotter()
    plotter.add_mesh(solid, scalars=solid.point_data['RGB'], rgb=True, opacity=0.8)
    plotter.add_title('SPPARKS 3D Microstructure')
    plotter.add_axes()
    
    return plotter

def _create_point_visualization(coords, colors, grain_ids):
    points = coords.astype(float)
    cloud = pv.PolyData(points)
    cloud.point_data['RGB'] = colors
    cloud.point_data['GrainID'] = grain_ids
    
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, scalars=colors, rgb=True, point_size=8.0, 
                    render_points_as_spheres=True, opacity=0.9)
    plotter.add_title('SPPARKS 3D Microstructure (Point Cloud)')
    plotter.add_axes()
    
    return plotter

def _create_cube_visualization(coords, colors, grain_ids, max_cubes=5000):
    n_cubes = min(len(coords), max_cubes)
    indices = np.random.choice(len(coords), n_cubes, replace=False)
    
    all_meshes = []
    all_colors = []
    
    for idx in indices:
        x, y, z = coords[idx]
        color = colors[idx]
        
        cube = pv.Cube(center=(x + 0.5, y + 0.5, z + 0.5),
                      x_length=0.95, y_length=0.95, z_length=0.95)
        all_meshes.append(cube)
        all_colors.extend([color] * cube.n_points)
    
    combined = all_meshes[0]
    for mesh in all_meshes[1:]:
        combined = combined.merge(mesh)
    
    combined.point_data['RGB'] = np.array(all_colors)
    
    plotter = pv.Plotter()
    plotter.add_mesh(combined, scalars=combined.point_data['RGB'], rgb=True, opacity=0.8)
    plotter.add_title(f'SPPARKS 3D Microstructure (Cubes, {n_cubes} voxels)')
    plotter.add_axes()
    
    return plotter
""")
        
        print("Working function saved to: spparks_visualizer.py")
        
    else:
        print("\n" + "="*60)
        print("FAILED: Could not create working visualizer")
        print("="*60)