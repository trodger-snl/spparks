
import numpy as np
import pyvista as pv

def create_marching_cubes_slab(coords, colors, grain_ids, grid_shape):
    """
    Create smooth solid surface using marching cubes algorithm.
    This should create a single continuous slab.
    """
    nx, ny, nz = grid_shape
    
    # Create scalar field with padding
    scalar_field = np.zeros((nx+2, ny+2, nz+2), dtype=float)
    color_field = np.zeros((nx+2, ny+2, nz+2, 3), dtype=float)
    
    # Fill the field
    for coord, color in zip(coords, colors):
        x, y, z = coord
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            scalar_field[x+1, y+1, z+1] = 1.0
            color_field[x+1, y+1, z+1] = color
    
    # Create image data
    image = pv.ImageData(dimensions=scalar_field.shape)
    image.spacing = (1.0, 1.0, 1.0)
    image.origin = (-1.0, -1.0, -1.0)
    
    image.point_data['values'] = scalar_field.ravel()
    image.point_data['RGB'] = color_field.reshape(-1, 3)
    
    # Create smooth surface with marching cubes
    surface = image.contour(isosurfaces=[0.5], scalars='values')
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    
    plotter.add_mesh(
        surface,
        scalars=surface.point_data['RGB'],
        rgb=True,
        opacity=1.0,
        show_edges=False,
        style='surface',
        smooth_shading=True
    )
    
    plotter.add_title('SPPARKS Solid Slab - Marching Cubes')
    plotter.add_axes()
    
    return plotter
