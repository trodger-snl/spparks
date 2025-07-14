
import numpy as np
import pyvista as pv

def create_structured_solid_visualization(coords, colors, grain_ids, grid_shape):
    """
    Create solid visualization using StructuredGrid with cell extraction.
    This avoids the thresholding issues with ImageData.
    """
    nx, ny, nz = grid_shape
    
    # Create structured grid points
    x_coords = np.arange(nx + 1, dtype=float)
    y_coords = np.arange(ny + 1, dtype=float)
    z_coords = np.arange(nz + 1, dtype=float)
    
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid = pv.StructuredGrid(X, Y, Z)
    
    # Create cell data
    n_cells = nx * ny * nz
    cell_filled = np.zeros(n_cells, dtype=bool)
    cell_colors = np.zeros((n_cells, 3), dtype=float)
    cell_grains = np.zeros(n_cells, dtype=int)
    
    # Map coordinates to cell indices
    cell_indices = coords[:, 0] * ny * nz + coords[:, 1] * nz + coords[:, 2]
    
    # Fill cell data
    cell_filled[cell_indices] = True
    cell_colors[cell_indices] = colors
    cell_grains[cell_indices] = grain_ids
    
    # Add to grid
    grid.cell_data['Filled'] = cell_filled.astype(float)
    grid.cell_data['RGB'] = cell_colors
    grid.cell_data['GrainID'] = cell_grains
    
    # Extract filled cells only
    filled_grid = grid.extract_cells(cell_filled)
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    plotter.add_mesh(
        filled_grid,
        scalars=filled_grid.cell_data['RGB'],
        rgb=True,
        opacity=1.0,
        show_edges=False
    )
    plotter.add_title('SPPARKS Solid Microstructure')
    plotter.add_axes()
    
    return plotter

def create_large_sphere_visualization(coords, colors, grain_ids):
    """
    Create visualization using large spheres for each voxel.
    Simple and effective for solid appearance.
    """
    # Center points at voxel centers
    points = coords.astype(float) + 0.5
    
    cloud = pv.PolyData(points)
    cloud.point_data['RGB'] = colors
    cloud.point_data['GrainID'] = grain_ids
    
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    plotter.add_mesh(
        cloud,
        scalars=cloud.point_data['RGB'],
        rgb=True,
        point_size=12.0,
        render_points_as_spheres=True,
        opacity=1.0
    )
    plotter.add_title('SPPARKS Microstructure - Sphere Voxels')
    plotter.add_axes()
    
    return plotter
