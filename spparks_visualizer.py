
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
