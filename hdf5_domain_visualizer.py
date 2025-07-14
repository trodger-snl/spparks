#!/usr/bin/env python3
"""
HDF5 Unstructured Domain Visualization Tool for SPPARKS

This tool provides comprehensive visualization and analysis of unstructured HDF5 
thermal domains to help optimize SPPARKS subdomain decomposition for parallel 
additive manufacturing simulations.

Author: Claude Code Assistant
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class HDF5Reader:
    """Efficient reader for unstructured HDF5 thermal domain files"""
    
    def __init__(self, filename: str):
        """
        Initialize HDF5 reader
        
        Args:
            filename: Path to HDF5 file containing unstructured thermal data
        """
        self.filename = filename
        self.file = None
        self.layer_times = None
        self.num_layers = None
        self._domain_bounds = None
        self._layer_cache = {}
        
    def open(self):
        """Open HDF5 file and read basic structure"""
        try:
            self.file = h5py.File(self.filename, 'r')
            self.layer_times = self.file['layerTimes'][:]
            self.num_layers = len(self.layer_times)
            print(f"Opened HDF5 file: {self.filename}")
            print(f"Found {self.num_layers} layers with times: {self.layer_times[0]:.3f} to {self.layer_times[-1]:.3f}")
            return True
        except Exception as e:
            print(f"Error opening HDF5 file: {e}")
            return False
    
    def close(self):
        """Close HDF5 file"""
        if self.file:
            self.file.close()
            self.file = None
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def get_layer_info(self, layer_idx: int = 0) -> Dict[str, Any]:
        """
        Get basic information about a specific layer
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            
        Returns:
            Dictionary with layer information
        """
        if layer_idx not in range(self.num_layers):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers-1}]")
        
        layer_group = self.file[str(layer_idx)]
        
        info = {
            'layer_idx': layer_idx,
            'layer_time': self.layer_times[layer_idx],
            'num_chunks': layer_group['boundingBoxes'].shape[0],
            'num_nodes': layer_group['nodeCoords'].shape[0],
            'num_elements': layer_group['elementToNode'].shape[0],
            'max_time_points': layer_group['temperatures'].shape[1]
        }
        
        return info
    
    def get_chunk_bboxes(self, layer_idx: int = 0) -> np.ndarray:
        """
        Get chunk bounding boxes for a layer
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Array of shape (N_chunks, 6) with [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        layer_group = self.file[str(layer_idx)]
        return layer_group['boundingBoxes'][:]
    
    def get_node_coordinates(self, layer_idx: int = 0, chunk_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get node coordinates for a layer
        
        Args:
            layer_idx: Layer index
            chunk_indices: Optional list of chunk indices to load (loads all if None)
            
        Returns:
            Array of shape (N_nodes, 3) with node coordinates
        """
        layer_group = self.file[str(layer_idx)]
        
        if chunk_indices is None:
            return layer_group['nodeCoords'][:]
        
        # Load only nodes from specified chunks
        elem_ptrs = layer_group['elemPtrs'][:]
        node_ptrs = layer_group['nodePtrs'][:]
        
        node_indices = []
        for chunk_idx in chunk_indices:
            start_node = node_ptrs[chunk_idx]
            end_node = node_ptrs[chunk_idx + 1]
            node_indices.extend(range(start_node, end_node))
        
        return layer_group['nodeCoords'][node_indices]
    
    def get_domain_bounds(self, layer_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get overall domain bounds for a layer
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (min_bounds, max_bounds) each with shape (3,)
        """
        if self._domain_bounds is None:
            chunk_bboxes = self.get_chunk_bboxes(layer_idx)
            min_bounds = np.min(chunk_bboxes[:, :3], axis=0)  # [xmin, ymin, zmin]
            max_bounds = np.max(chunk_bboxes[:, 3:], axis=0)  # [xmax, ymax, zmax]
            self._domain_bounds = (min_bounds, max_bounds)
        
        return self._domain_bounds
    
    def get_chunks_overlapping_region(self, region_bounds: np.ndarray, layer_idx: int = 0) -> List[int]:
        """
        Find chunks that overlap with a specified rectangular region
        
        Args:
            region_bounds: Array [xmin, ymin, zmin, xmax, ymax, zmax]
            layer_idx: Layer index
            
        Returns:
            List of chunk indices that overlap with the region
        """
        chunk_bboxes = self.get_chunk_bboxes(layer_idx)
        
        overlapping_chunks = []
        for i, bbox in enumerate(chunk_bboxes):
            # Check if bounding boxes overlap
            if (bbox[0] < region_bounds[3] and region_bounds[0] < bbox[3] and
                bbox[1] < region_bounds[4] and region_bounds[1] < bbox[4] and
                bbox[2] < region_bounds[5] and region_bounds[2] < bbox[5]):
                overlapping_chunks.append(i)
        
        return overlapping_chunks
    
    def estimate_memory_usage(self, layer_idx: int = 0) -> Dict[str, float]:
        """
        Estimate memory usage for loading a complete layer
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dictionary with memory estimates in MB
        """
        info = self.get_layer_info(layer_idx)
        
        # Estimate sizes (in bytes)
        node_coords_size = info['num_nodes'] * 3 * 8  # 3 doubles per node
        element_connectivity_size = info['num_elements'] * 4 * 4  # 4 ints per element
        temperatures_size = info['num_nodes'] * info['max_time_points'] * 8  # doubles
        times_size = temperatures_size  # same size as temperatures
        
        total_size = node_coords_size + element_connectivity_size + temperatures_size + times_size
        
        return {
            'node_coordinates_mb': node_coords_size / 1024**2,
            'element_connectivity_mb': element_connectivity_size / 1024**2,
            'temperatures_mb': temperatures_size / 1024**2,
            'times_mb': times_size / 1024**2,
            'total_mb': total_size / 1024**2
        }


class DomainVisualizer:
    """3D visualization of thermal domain and chunks"""
    
    def __init__(self, reader: HDF5Reader):
        """
        Initialize domain visualizer
        
        Args:
            reader: HDF5Reader instance
        """
        self.reader = reader
        
    def plot_domain_overview(self, layer_idx: int = 0, max_chunks: int = 500, 
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot 3D overview of the thermal domain showing chunk distribution
        
        Args:
            layer_idx: Layer index to visualize
            max_chunks: Maximum number of chunks to plot (for performance)
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
        min_bounds, max_bounds = self.reader.get_domain_bounds(layer_idx)
        
        # Subsample chunks if too many
        if len(chunk_bboxes) > max_chunks:
            indices = np.random.choice(len(chunk_bboxes), max_chunks, replace=False)
            chunk_bboxes = chunk_bboxes[indices]
            print(f"Subsampled {max_chunks} chunks from {len(chunk_bboxes)} total")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot chunk centers colored by volume
        centers = (chunk_bboxes[:, :3] + chunk_bboxes[:, 3:]) / 2
        volumes = np.prod(chunk_bboxes[:, 3:] - chunk_bboxes[:, :3], axis=1)
        
        scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                           c=volumes, cmap='viridis', alpha=0.6, s=20)
        
        # Plot domain bounds
        ax.plot([min_bounds[0], max_bounds[0]], [min_bounds[1], min_bounds[1]], 
                [min_bounds[2], min_bounds[2]], 'r-', linewidth=2, label='Domain bounds')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Thermal Domain Overview - Layer {layer_idx}\n'
                    f'Time: {self.reader.layer_times[layer_idx]:.3f}s, '
                    f'Chunks: {len(chunk_bboxes)}')
        
        plt.colorbar(scatter, ax=ax, label='Chunk Volume (m³)', shrink=0.8)
        ax.legend()
        
        return fig
    
    def plot_chunk_distribution(self, layer_idx: int = 0, 
                              figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot chunk size and spatial distribution analysis
        
        Args:
            layer_idx: Layer index to visualize
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
        
        # Calculate chunk properties
        sizes = chunk_bboxes[:, 3:] - chunk_bboxes[:, :3]  # [dx, dy, dz]
        volumes = np.prod(sizes, axis=1)
        centers = (chunk_bboxes[:, :3] + chunk_bboxes[:, 3:]) / 2
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Volume distribution
        axes[0].hist(volumes, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Chunk Volume (m³)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Chunk Volume Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Size distribution (box plot)
        axes[1].boxplot([sizes[:, 0], sizes[:, 1], sizes[:, 2]], 
                       labels=['X', 'Y', 'Z'])
        axes[1].set_ylabel('Size (m)')
        axes[1].set_title('Chunk Size Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Spatial distribution (2D projection)
        scatter = axes[2].scatter(centers[:, 0], centers[:, 1], 
                                c=volumes, cmap='viridis', alpha=0.6, s=10)
        axes[2].set_xlabel('X (m)')
        axes[2].set_ylabel('Y (m)')
        axes[2].set_title('Chunk Distribution (X-Y Projection)')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2], label='Volume (m³)')
        
        plt.tight_layout()
        return fig
    
    def plot_spparks_overlay(self, spparks_bounds: np.ndarray, 
                           processor_counts: Tuple[int, int, int] = (2, 2, 2),
                           layer_idx: int = 0, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot SPPARKS grid overlay on thermal domain
        
        Args:
            spparks_bounds: Array [xmin, ymin, zmin, xmax, ymax, zmax] for SPPARKS domain
            processor_counts: Tuple (nx, ny, nz) for processor decomposition
            layer_idx: Layer index to visualize
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot thermal domain chunks
        centers = (chunk_bboxes[:, :3] + chunk_bboxes[:, 3:]) / 2
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                  alpha=0.3, s=5, c='blue', label='Thermal chunks')
        
        # Plot SPPARKS domain outline
        xmin, ymin, zmin, xmax, ymax, zmax = spparks_bounds
        
        # Plot domain edges
        edges = [
            [[xmin, xmax], [ymin, ymin], [zmin, zmin]],
            [[xmin, xmax], [ymax, ymax], [zmin, zmin]],
            [[xmin, xmax], [ymin, ymin], [zmax, zmax]],
            [[xmin, xmax], [ymax, ymax], [zmax, zmax]],
            [[xmin, xmin], [ymin, ymax], [zmin, zmin]],
            [[xmax, xmax], [ymin, ymax], [zmin, zmin]],
            [[xmin, xmin], [ymin, ymax], [zmax, zmax]],
            [[xmax, xmax], [ymin, ymax], [zmax, zmax]],
            [[xmin, xmin], [ymin, ymin], [zmin, zmax]],
            [[xmax, xmax], [ymin, ymin], [zmin, zmax]],
            [[xmin, xmin], [ymax, ymax], [zmin, zmax]],
            [[xmax, xmax], [ymax, ymax], [zmin, zmax]]
        ]
        
        for edge in edges:
            ax.plot(edge[0], edge[1], edge[2], 'r-', linewidth=2, alpha=0.8)
        
        # Plot processor subdomain boundaries
        nx, ny, nz = processor_counts
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        dz = (zmax - zmin) / nz
        
        # X boundaries
        for i in range(1, nx):
            x = xmin + i * dx
            ax.plot([x, x], [ymin, ymax], [zmin, zmin], 'g--', alpha=0.6)
            ax.plot([x, x], [ymin, ymax], [zmax, zmax], 'g--', alpha=0.6)
            ax.plot([x, x], [ymin, ymin], [zmin, zmax], 'g--', alpha=0.6)
            ax.plot([x, x], [ymax, ymax], [zmin, zmax], 'g--', alpha=0.6)
        
        # Y boundaries
        for j in range(1, ny):
            y = ymin + j * dy
            ax.plot([xmin, xmax], [y, y], [zmin, zmin], 'g--', alpha=0.6)
            ax.plot([xmin, xmax], [y, y], [zmax, zmax], 'g--', alpha=0.6)
            ax.plot([xmin, xmin], [y, y], [zmin, zmax], 'g--', alpha=0.6)
            ax.plot([xmax, xmax], [y, y], [zmin, zmax], 'g--', alpha=0.6)
        
        # Z boundaries
        for k in range(1, nz):
            z = zmin + k * dz
            ax.plot([xmin, xmax], [ymin, ymin], [z, z], 'g--', alpha=0.6)
            ax.plot([xmin, xmax], [ymax, ymax], [z, z], 'g--', alpha=0.6)
            ax.plot([xmin, xmin], [ymin, ymax], [z, z], 'g--', alpha=0.6)
            ax.plot([xmax, xmax], [ymin, ymax], [z, z], 'g--', alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'SPPARKS Grid Overlay - Layer {layer_idx}\n'
                    f'Processors: {nx}×{ny}×{nz} = {nx*ny*nz} total')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=5, alpha=0.6, label='Thermal chunks'),
            Line2D([0], [0], color='red', linewidth=2, label='SPPARKS domain'),
            Line2D([0], [0], color='green', linestyle='--', label='Processor boundaries')
        ]
        ax.legend(handles=legend_elements)
        
        return fig


class SubdomainAnalyzer:
    """Analysis tools for SPPARKS subdomain optimization"""
    
    def __init__(self, reader: HDF5Reader):
        """
        Initialize subdomain analyzer
        
        Args:
            reader: HDF5Reader instance
        """
        self.reader = reader
    
    def analyze_load_balancing(self, spparks_bounds: np.ndarray, 
                              processor_counts: Tuple[int, int, int],
                              layer_idx: int = 0) -> pd.DataFrame:
        """
        Analyze computational load balancing for SPPARKS decomposition
        
        Args:
            spparks_bounds: Array [xmin, ymin, zmin, xmax, ymax, zmax]
            processor_counts: Tuple (nx, ny, nz) for processor decomposition
            layer_idx: Layer index to analyze
            
        Returns:
            DataFrame with per-processor load analysis
        """
        nx, ny, nz = processor_counts
        xmin, ymin, zmin, xmax, ymax, zmax = spparks_bounds
        
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        dz = (zmax - zmin) / nz
        
        results = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Define processor subdomain
                    proc_bounds = np.array([
                        xmin + i * dx, ymin + j * dy, zmin + k * dz,
                        xmin + (i+1) * dx, ymin + (j+1) * dy, zmin + (k+1) * dz
                    ])
                    
                    # Find overlapping chunks
                    overlapping_chunks = self.reader.get_chunks_overlapping_region(
                        proc_bounds, layer_idx)
                    
                    # Calculate load metrics
                    chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
                    total_volume = 0
                    if overlapping_chunks:
                        overlap_bboxes = chunk_bboxes[overlapping_chunks]
                        chunk_volumes = np.prod(overlap_bboxes[:, 3:] - overlap_bboxes[:, :3], axis=1)
                        total_volume = np.sum(chunk_volumes)
                    
                    results.append({
                        'processor_id': i * ny * nz + j * nz + k,
                        'i': i, 'j': j, 'k': k,
                        'num_chunks': len(overlapping_chunks),
                        'total_volume': total_volume,
                        'subdomain_volume': dx * dy * dz,
                        'coverage_ratio': total_volume / (dx * dy * dz) if dx * dy * dz > 0 else 0
                    })
        
        df = pd.DataFrame(results)
        
        # Add load balance statistics
        if len(df) > 0:
            df['chunks_normalized'] = df['num_chunks'] / df['num_chunks'].mean()
            df['volume_normalized'] = df['total_volume'] / df['total_volume'].mean()
        
        return df
    
    def recommend_decomposition(self, spparks_bounds: np.ndarray, 
                               target_processors: int = 8,
                               layer_idx: int = 0) -> Dict[str, Any]:
        """
        Recommend optimal processor decomposition
        
        Args:
            spparks_bounds: Array [xmin, ymin, zmin, xmax, ymax, zmax]
            target_processors: Target number of processors
            layer_idx: Layer index to analyze
            
        Returns:
            Dictionary with decomposition recommendations
        """
        # Get domain dimensions
        domain_size = spparks_bounds[3:] - spparks_bounds[:3]
        
        # Find factors of target_processors
        factors = []
        for i in range(1, target_processors + 1):
            if target_processors % i == 0:
                for j in range(1, i + 1):
                    if i % j == 0:
                        k = target_processors // (i * j)
                        if i * j * k == target_processors:
                            factors.append((i, j, k))
        
        # Remove duplicates and sort by aspect ratio similarity
        unique_factors = list(set(factors))
        
        best_decomposition = None
        best_score = float('inf')
        
        for nx, ny, nz in unique_factors:
            # Calculate aspect ratios
            proc_aspect = np.array([domain_size[0]/nx, domain_size[1]/ny, domain_size[2]/nz])
            
            # Score based on load balance and aspect ratio uniformity
            load_df = self.analyze_load_balancing(spparks_bounds, (nx, ny, nz), layer_idx)
            
            if len(load_df) > 0:
                load_imbalance = load_df['chunks_normalized'].std()
                aspect_variance = np.var(proc_aspect)
                score = load_imbalance + 0.1 * aspect_variance  # Weight factors
                
                if score < best_score:
                    best_score = score
                    best_decomposition = {
                        'processor_counts': (nx, ny, nz),
                        'total_processors': nx * ny * nz,
                        'processor_size': proc_aspect,
                        'load_imbalance': load_imbalance,
                        'aspect_variance': aspect_variance,
                        'score': score
                    }
        
        return best_decomposition
    
    def generate_spparks_config(self, spparks_bounds: np.ndarray,
                               processor_counts: Tuple[int, int, int],
                               lattice_spacing: float = 1e-6) -> str:
        """
        Generate SPPARKS input file region and processor configuration
        
        Args:
            spparks_bounds: Array [xmin, ymin, zmin, xmax, ymax, zmax]
            processor_counts: Tuple (nx, ny, nz)
            lattice_spacing: SPPARKS lattice spacing in meters
            
        Returns:
            String with SPPARKS configuration commands
        """
        xmin, ymin, zmin, xmax, ymax, zmax = spparks_bounds
        nx, ny, nz = processor_counts
        
        # Convert to lattice units (assuming meters input)
        xlattice = int((xmax - xmin) / lattice_spacing)
        ylattice = int((ymax - ymin) / lattice_spacing)
        zlattice = int((zmax - zmin) / lattice_spacing)
        
        config = f"""# SPPARKS configuration for unstructured HDF5 thermal data
# Generated automatically by HDF5 domain visualizer

# Domain definition
region box block 0 {xlattice} 0 {ylattice} 0 {zlattice}
create_box box
create_sites box

# Processor decomposition: {nx}×{ny}×{nz} = {nx*ny*nz} processors
# Run with: mpirun -np {nx*ny*nz} spk_executable < input_file

# Physical domain bounds (meters):
# X: {xmin:.6f} to {xmax:.6f}
# Y: {ymin:.6f} to {ymax:.6f}  
# Z: {zmin:.6f} to {zmax:.6f}

# Lattice spacing: {lattice_spacing:.2e} m
# Grid dimensions: {xlattice} × {ylattice} × {zlattice}

# Temperature source configuration
temperature hdf5_unstructured thermal_data.hdf5 {lattice_spacing:.2e}
"""
        
        return config


def main():
    """Example usage of the HDF5 domain visualization tool"""
    
    # Example file path - replace with your actual file
    hdf5_file = "/Users/Tron/Downloads/Huddle-resourcesfornewformat/reduced_thermal_output.hdf5"
    
    print("=== HDF5 Unstructured Domain Visualization Tool ===")
    
    try:
        with HDF5Reader(hdf5_file) as reader:
            # Get basic domain information
            print("\n--- Domain Information ---")
            layer_info = reader.get_layer_info(0)
            for key, value in layer_info.items():
                print(f"{key}: {value}")
            
            min_bounds, max_bounds = reader.get_domain_bounds(0)
            print(f"\nDomain bounds:")
            print(f"  X: {min_bounds[0]:.6f} to {max_bounds[0]:.6f} m")
            print(f"  Y: {min_bounds[1]:.6f} to {max_bounds[1]:.6f} m")
            print(f"  Z: {min_bounds[2]:.6f} to {max_bounds[2]:.6f} m")
            
            # Memory usage estimate
            memory_usage = reader.estimate_memory_usage(0)
            print(f"\nMemory usage estimate: {memory_usage['total_mb']:.1f} MB")
            
            # Create visualizer
            visualizer = DomainVisualizer(reader)
            
            # Plot domain overview
            print("\n--- Creating Visualizations ---")
            fig1 = visualizer.plot_domain_overview(layer_idx=0)
            plt.savefig('domain_overview.png', dpi=150, bbox_inches='tight')
            print("Saved: domain_overview.png")
            
            # Plot chunk distribution
            fig2 = visualizer.plot_chunk_distribution(layer_idx=0)
            plt.savefig('chunk_distribution.png', dpi=150, bbox_inches='tight')
            print("Saved: chunk_distribution.png")
            
            # Example SPPARKS subdomain analysis
            print("\n--- SPPARKS Subdomain Analysis ---")
            
            # Define a SPPARKS domain that covers the thermal domain
            margin = 0.001  # 1mm margin
            spparks_bounds = np.array([
                min_bounds[0] - margin, min_bounds[1] - margin, min_bounds[2] - margin,
                max_bounds[0] + margin, max_bounds[1] + margin, max_bounds[2] + margin
            ])
            
            # Analyze different processor decompositions
            analyzer = SubdomainAnalyzer(reader)
            
            for n_proc in [4, 8, 16]:
                recommendation = analyzer.recommend_decomposition(
                    spparks_bounds, target_processors=n_proc, layer_idx=0)
                
                if recommendation:
                    nx, ny, nz = recommendation['processor_counts']
                    print(f"\nRecommended {n_proc}-processor decomposition: {nx}×{ny}×{nz}")
                    print(f"  Load imbalance: {recommendation['load_imbalance']:.3f}")
                    print(f"  Processor sizes: {recommendation['processor_size']}")
            
            # Generate SPPARKS configuration
            config = analyzer.generate_spparks_config(
                spparks_bounds, (2, 2, 2), lattice_spacing=1e-6)
            
            with open('spparks_config.txt', 'w') as f:
                f.write(config)
            print("\nSaved: spparks_config.txt")
            
            # Create SPPARKS overlay visualization
            fig3 = visualizer.plot_spparks_overlay(
                spparks_bounds, processor_counts=(2, 2, 2), layer_idx=0)
            plt.savefig('spparks_overlay.png', dpi=150, bbox_inches='tight')
            print("Saved: spparks_overlay.png")
            
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required dependencies:")
        print("  pip install h5py numpy matplotlib pandas")


if __name__ == "__main__":
    main()