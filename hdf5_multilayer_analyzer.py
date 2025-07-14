#!/usr/bin/env python3
"""
Multi-Layer HDF5 Domain Analysis for SPPARKS

Enhanced analysis toolkit that considers the complete 4D (3D space + time) thermal domain
evolution across all layers to provide optimal SPPARKS subdomain decomposition for the
entire simulation timeline.

Key Features:
- Temporal domain bounds analysis
- Consolidated chunk coverage across all layers
- Domain evolution visualization
- Optimal SPPARKS sizing for complete time history
- Load balancing analysis considering temporal changes

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import base classes
from hdf5_domain_visualizer import HDF5Reader, DomainVisualizer, SubdomainAnalyzer

class MultiLayerAnalyzer:
    """Multi-layer analysis for complete 4D thermal domain"""
    
    def __init__(self, reader: HDF5Reader):
        """
        Initialize multi-layer analyzer
        
        Args:
            reader: HDF5Reader instance
        """
        self.reader = reader
        self._global_bounds_cache = None
        self._layer_stats_cache = None
        self._chunk_evolution_cache = None
        
    def get_global_domain_bounds(self, layer_subset: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get overall domain bounds across all layers (or subset)
        
        Args:
            layer_subset: Optional list of layer indices to analyze (None = all layers)
            
        Returns:
            Tuple of (global_min_bounds, global_max_bounds) each with shape (3,)
        """
        if self._global_bounds_cache is None or layer_subset is not None:
            
            if layer_subset is not None:
                layers_to_analyze = layer_subset
            else:
                # Check which layers actually exist
                layers_to_analyze = []
                for i in range(self.reader.num_layers):
                    try:
                        # Test if layer exists by trying to get its info
                        self.reader.get_layer_info(i)
                        layers_to_analyze.append(i)
                    except:
                        continue
            
            global_min = np.array([np.inf, np.inf, np.inf])
            global_max = np.array([-np.inf, -np.inf, -np.inf])
            
            print(f"Analyzing domain bounds across {len(layers_to_analyze)} layers...")
            
            for i, layer_idx in enumerate(layers_to_analyze):
                if i % 5 == 0:  # Progress indicator
                    print(f"  Processing layer {layer_idx} ({i+1}/{len(layers_to_analyze)})")
                
                try:
                    chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
                    if len(chunk_bboxes) > 0:
                        layer_min = np.min(chunk_bboxes[:, :3], axis=0)
                        layer_max = np.max(chunk_bboxes[:, 3:], axis=0)
                        
                        global_min = np.minimum(global_min, layer_min)
                        global_max = np.maximum(global_max, layer_max)
                except Exception as e:
                    print(f"    Warning: Skipping layer {layer_idx} due to error: {e}")
                    continue
            
            if layer_subset is None:
                self._global_bounds_cache = (global_min, global_max)
            
            return global_min, global_max
        
        return self._global_bounds_cache
    
    def analyze_layer_statistics(self, layer_subset: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Analyze statistics for each layer
        
        Args:
            layer_subset: Optional list of layer indices to analyze
            
        Returns:
            DataFrame with per-layer statistics
        """
        if self._layer_stats_cache is None or layer_subset is not None:
            
            layers_to_analyze = layer_subset if layer_subset is not None else range(self.reader.num_layers)
            
            stats_list = []
            
            print(f"Analyzing statistics for {len(layers_to_analyze)} layers...")
            
            for i, layer_idx in enumerate(layers_to_analyze):
                if i % 5 == 0:
                    print(f"  Processing layer {layer_idx} ({i+1}/{len(layers_to_analyze)})")
                
                try:
                    # Get layer info
                    layer_info = self.reader.get_layer_info(layer_idx)
                except Exception as e:
                    print(f"    Warning: Skipping layer {layer_idx} due to error: {e}")
                    continue
                
                # Get chunk statistics
                chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
                if len(chunk_bboxes) > 0:
                    # Calculate chunk properties
                    sizes = chunk_bboxes[:, 3:] - chunk_bboxes[:, :3]
                    volumes = np.prod(sizes, axis=1)
                    centers = (chunk_bboxes[:, :3] + chunk_bboxes[:, 3:]) / 2
                    
                    # Domain bounds for this layer
                    layer_min = np.min(chunk_bboxes[:, :3], axis=0)
                    layer_max = np.max(chunk_bboxes[:, 3:], axis=0)
                    domain_volume = np.prod(layer_max - layer_min)
                    
                    stats = {
                        'layer_idx': layer_idx,
                        'layer_time': layer_info['layer_time'],
                        'num_chunks': layer_info['num_chunks'],
                        'num_nodes': layer_info['num_nodes'],
                        'num_elements': layer_info['num_elements'],
                        'max_time_points': layer_info['max_time_points'],
                        'domain_volume': domain_volume,
                        'total_chunk_volume': np.sum(volumes),
                        'mean_chunk_volume': np.mean(volumes),
                        'std_chunk_volume': np.std(volumes),
                        'min_chunk_volume': np.min(volumes),
                        'max_chunk_volume': np.max(volumes),
                        'domain_x_size': layer_max[0] - layer_min[0],
                        'domain_y_size': layer_max[1] - layer_min[1],
                        'domain_z_size': layer_max[2] - layer_min[2],
                        'domain_x_min': layer_min[0],
                        'domain_y_min': layer_min[1],
                        'domain_z_min': layer_min[2],
                        'domain_x_max': layer_max[0],
                        'domain_y_max': layer_max[1],
                        'domain_z_max': layer_max[2],
                    }
                else:
                    # Empty layer
                    stats = {
                        'layer_idx': layer_idx,
                        'layer_time': layer_info['layer_time'],
                        'num_chunks': 0,
                        'num_nodes': 0,
                        'num_elements': 0,
                        'max_time_points': 0,
                        'domain_volume': 0,
                        'total_chunk_volume': 0,
                        'mean_chunk_volume': 0,
                        'std_chunk_volume': 0,
                        'min_chunk_volume': 0,
                        'max_chunk_volume': 0,
                        'domain_x_size': 0,
                        'domain_y_size': 0,
                        'domain_z_size': 0,
                        'domain_x_min': 0,
                        'domain_y_min': 0,
                        'domain_z_min': 0,
                        'domain_x_max': 0,
                        'domain_y_max': 0,
                        'domain_z_max': 0,
                    }
                
                stats_list.append(stats)
            
            df = pd.DataFrame(stats_list)
            
            if layer_subset is None:
                self._layer_stats_cache = df
            
            return df
        
        return self._layer_stats_cache
    
    def analyze_chunk_evolution(self, sample_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze how chunk distribution evolves over time
        
        Args:
            sample_layers: Optional list of layer indices to sample (for performance)
            
        Returns:
            Dictionary with chunk evolution analysis
        """
        if sample_layers is None:
            # Sample every 3rd layer for performance
            sample_layers = list(range(0, self.reader.num_layers, 3))
        
        print(f"Analyzing chunk evolution across {len(sample_layers)} sample layers...")
        
        evolution_data = {
            'layer_indices': sample_layers,
            'layer_times': [],
            'chunk_counts': [],
            'domain_volumes': [],
            'chunk_density': [],  # chunks per unit volume
            'spatial_extent': [],  # max distance between chunks
        }
        
        for i, layer_idx in enumerate(sample_layers):
            print(f"  Processing layer {layer_idx} ({i+1}/{len(sample_layers)})")
            
            try:
                layer_time = self.reader.layer_times[layer_idx]
                chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
            except Exception as e:
                print(f"    Warning: Skipping layer {layer_idx} due to error: {e}")
                continue
            
            evolution_data['layer_times'].append(layer_time)
            evolution_data['chunk_counts'].append(len(chunk_bboxes))
            
            if len(chunk_bboxes) > 0:
                # Domain volume
                layer_min = np.min(chunk_bboxes[:, :3], axis=0)
                layer_max = np.max(chunk_bboxes[:, 3:], axis=0)
                domain_volume = np.prod(layer_max - layer_min)
                evolution_data['domain_volumes'].append(domain_volume)
                
                # Chunk density
                evolution_data['chunk_density'].append(len(chunk_bboxes) / domain_volume if domain_volume > 0 else 0)
                
                # Spatial extent (maximum distance between chunk centers)
                centers = (chunk_bboxes[:, :3] + chunk_bboxes[:, 3:]) / 2
                if len(centers) > 1:
                    distances = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
                    max_distance = np.max(distances)
                else:
                    max_distance = 0
                evolution_data['spatial_extent'].append(max_distance)
            else:
                evolution_data['domain_volumes'].append(0)
                evolution_data['chunk_density'].append(0)
                evolution_data['spatial_extent'].append(0)
        
        # Convert to numpy arrays
        for key in ['layer_times', 'chunk_counts', 'domain_volumes', 'chunk_density', 'spatial_extent']:
            evolution_data[key] = np.array(evolution_data[key])
        
        self._chunk_evolution_cache = evolution_data
        return evolution_data
    
    def get_optimal_spparks_domain(self, margin_factor: float = 0.05, 
                                  layer_subset: Optional[List[int]] = None) -> np.ndarray:
        """
        Get optimal SPPARKS domain bounds that encompass all layers
        
        Args:
            margin_factor: Fractional margin to add around global bounds
            layer_subset: Optional list of layers to consider
            
        Returns:
            Array [xmin, ymin, zmin, xmax, ymax, zmax] for SPPARKS domain
        """
        global_min, global_max = self.get_global_domain_bounds(layer_subset)
        
        # Add margins
        domain_size = global_max - global_min
        margin = domain_size * margin_factor
        
        spparks_bounds = np.array([
            global_min[0] - margin[0], global_min[1] - margin[1], global_min[2] - margin[2],
            global_max[0] + margin[0], global_max[1] + margin[1], global_max[2] + margin[2]
        ])
        
        return spparks_bounds
    
    def analyze_temporal_load_balancing(self, spparks_bounds: np.ndarray,
                                       processor_counts: Tuple[int, int, int],
                                       sample_layers: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Analyze load balancing across multiple time layers
        
        Args:
            spparks_bounds: SPPARKS domain bounds
            processor_counts: Tuple (nx, ny, nz)
            sample_layers: Optional list of layers to analyze
            
        Returns:
            DataFrame with temporal load balancing analysis
        """
        if sample_layers is None:
            # Sample every 5th layer for performance
            sample_layers = list(range(0, self.reader.num_layers, 5))
        
        print(f"Analyzing temporal load balancing across {len(sample_layers)} layers...")
        
        analyzer = SubdomainAnalyzer(self.reader)
        
        all_results = []
        
        for i, layer_idx in enumerate(sample_layers):
            print(f"  Processing layer {layer_idx} ({i+1}/{len(sample_layers)})")
            
            # Get load balance for this layer
            load_df = analyzer.analyze_load_balancing(spparks_bounds, processor_counts, layer_idx)
            
            # Add layer information
            load_df['layer_idx'] = layer_idx
            load_df['layer_time'] = self.reader.layer_times[layer_idx]
            
            all_results.append(load_df)
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        return combined_df
    
    def plot_domain_evolution(self, sample_layers: Optional[List[int]] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot domain evolution over time
        
        Args:
            sample_layers: Optional list of layers to sample
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        # Get evolution data
        evolution_data = self.analyze_chunk_evolution(sample_layers)
        layer_stats = self.analyze_layer_statistics(sample_layers if sample_layers else None)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Thermal Domain Evolution Over Time', fontsize=16)
        
        times = evolution_data['layer_times']
        
        # Chunk count evolution
        axes[0, 0].plot(times, evolution_data['chunk_counts'], 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Number of Chunks')
        axes[0, 0].set_title('Chunk Count Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Domain volume evolution
        axes[0, 1].plot(times, evolution_data['domain_volumes'], 'r-o', markersize=4)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Domain Volume (m³)')
        axes[0, 1].set_title('Domain Volume Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Chunk density evolution
        axes[0, 2].plot(times, evolution_data['chunk_density'], 'g-o', markersize=4)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Chunks per m³')
        axes[0, 2].set_title('Chunk Density Evolution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Domain size evolution (X, Y, Z)
        if not layer_stats.empty:
            axes[1, 0].plot(layer_stats['layer_time'], layer_stats['domain_x_size'], 'b-', label='X', linewidth=2)
            axes[1, 0].plot(layer_stats['layer_time'], layer_stats['domain_y_size'], 'r-', label='Y', linewidth=2)
            axes[1, 0].plot(layer_stats['layer_time'], layer_stats['domain_z_size'], 'g-', label='Z', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Domain Size (m)')
            axes[1, 0].set_title('Domain Size Evolution by Axis')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Node count evolution
        if not layer_stats.empty:
            axes[1, 1].plot(layer_stats['layer_time'], layer_stats['num_nodes'], 'm-o', markersize=4)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Number of Nodes')
            axes[1, 1].set_title('Node Count Evolution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Spatial extent evolution
        axes[1, 2].plot(times, evolution_data['spatial_extent'], 'c-o', markersize=4)
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Max Distance (m)')
        axes[1, 2].set_title('Spatial Extent Evolution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_load_balance(self, temporal_load_df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plot temporal load balancing analysis
        
        Args:
            temporal_load_df: DataFrame from analyze_temporal_load_balancing
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Temporal Load Balancing Analysis', fontsize=16)
        
        # Group by layer for analysis
        layer_groups = temporal_load_df.groupby('layer_idx')
        
        # Extract time series data
        layer_times = []
        load_imbalances = []
        max_chunks = []
        min_chunks = []
        mean_chunks = []
        
        for layer_idx, group in layer_groups:
            layer_times.append(group['layer_time'].iloc[0])
            
            # Calculate load imbalance (std of normalized chunks)
            if len(group) > 0 and group['num_chunks'].mean() > 0:
                normalized_chunks = group['num_chunks'] / group['num_chunks'].mean()
                load_imbalances.append(normalized_chunks.std())
                max_chunks.append(group['num_chunks'].max())
                min_chunks.append(group['num_chunks'].min())
                mean_chunks.append(group['num_chunks'].mean())
            else:
                load_imbalances.append(0)
                max_chunks.append(0)
                min_chunks.append(0)
                mean_chunks.append(0)
        
        layer_times = np.array(layer_times)
        load_imbalances = np.array(load_imbalances)
        max_chunks = np.array(max_chunks)
        min_chunks = np.array(min_chunks)
        mean_chunks = np.array(mean_chunks)
        
        # Load imbalance over time
        axes[0, 0].plot(layer_times, load_imbalances, 'r-o', markersize=4)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Load Imbalance Factor')
        axes[0, 0].set_title('Load Imbalance Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Chunk distribution over time
        axes[0, 1].plot(layer_times, max_chunks, 'r-', label='Max', linewidth=2)
        axes[0, 1].plot(layer_times, mean_chunks, 'b-', label='Mean', linewidth=2)
        axes[0, 1].plot(layer_times, min_chunks, 'g-', label='Min', linewidth=2)
        axes[0, 1].fill_between(layer_times, min_chunks, max_chunks, alpha=0.3, color='gray')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Chunks per Processor')
        axes[0, 1].set_title('Chunk Distribution Range')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Load imbalance distribution
        axes[1, 0].hist(load_imbalances, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Load Imbalance Factor')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Load Imbalance Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Processor load variation over time
        if len(temporal_load_df) > 0:
            # Show load for first few processors as examples
            unique_procs = sorted(temporal_load_df['processor_id'].unique())[:8]  # First 8 processors
            for proc_id in unique_procs:
                proc_data = temporal_load_df[temporal_load_df['processor_id'] == proc_id]
                axes[1, 1].plot(proc_data['layer_time'], proc_data['num_chunks'], 
                               'o-', markersize=3, label=f'Proc {proc_id}', alpha=0.7)
            
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Chunks per Processor')
            axes[1, 1].set_title('Per-Processor Load Evolution (Sample)')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_multilayer_report(self, output_file: str = 'multilayer_analysis_report.txt'):
        """
        Generate comprehensive multi-layer analysis report
        
        Args:
            output_file: Output file path
        """
        print("Generating comprehensive multi-layer analysis report...")
        
        # Perform all analyses
        global_min, global_max = self.get_global_domain_bounds()
        layer_stats = self.analyze_layer_statistics()
        evolution_data = self.analyze_chunk_evolution()
        optimal_bounds = self.get_optimal_spparks_domain()
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-LAYER HDF5 THERMAL DOMAIN ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Global domain information
            f.write("GLOBAL DOMAIN BOUNDS (All Layers)\n")
            f.write("-" * 40 + "\n")
            f.write(f"X: {global_min[0]:.6f} to {global_max[0]:.6f} m (size: {global_max[0]-global_min[0]:.6f} m)\n")
            f.write(f"Y: {global_min[1]:.6f} to {global_max[1]:.6f} m (size: {global_max[1]-global_min[1]:.6f} m)\n")
            f.write(f"Z: {global_min[2]:.6f} to {global_max[2]:.6f} m (size: {global_max[2]-global_min[2]:.6f} m)\n")
            f.write(f"Total domain volume: {np.prod(global_max - global_min):.2e} m³\n\n")
            
            # Temporal evolution summary
            f.write("TEMPORAL EVOLUTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Time range: {self.reader.layer_times[0]:.3f} to {self.reader.layer_times[-1]:.3f} s\n")
            f.write(f"Number of layers: {self.reader.num_layers}\n")
            f.write(f"Time step: {np.mean(np.diff(self.reader.layer_times)):.3f} ± {np.std(np.diff(self.reader.layer_times)):.3f} s\n\n")
            
            # Layer statistics summary
            if not layer_stats.empty:
                f.write("LAYER STATISTICS SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Chunks per layer: {layer_stats['num_chunks'].mean():.1f} ± {layer_stats['num_chunks'].std():.1f}\n")
                f.write(f"  Range: {layer_stats['num_chunks'].min()} to {layer_stats['num_chunks'].max()}\n")
                f.write(f"Nodes per layer: {layer_stats['num_nodes'].mean():.0f} ± {layer_stats['num_nodes'].std():.0f}\n")
                f.write(f"  Range: {layer_stats['num_nodes'].min()} to {layer_stats['num_nodes'].max()}\n")
                f.write(f"Domain volume per layer: {layer_stats['domain_volume'].mean():.2e} ± {layer_stats['domain_volume'].std():.2e} m³\n")
                f.write(f"Memory estimate per layer: {layer_stats['num_nodes'].mean() * 8 * 3 / 1024**2:.1f} MB (coordinates only)\n\n")
            
            # Evolution analysis
            f.write("DOMAIN EVOLUTION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Chunk count variation: {np.std(evolution_data['chunk_counts']):.1f} (coefficient of variation: {np.std(evolution_data['chunk_counts'])/np.mean(evolution_data['chunk_counts']):.3f})\n")
            f.write(f"Domain volume variation: {np.std(evolution_data['domain_volumes']):.2e} m³\n")
            f.write(f"Spatial extent range: {np.min(evolution_data['spatial_extent']):.6f} to {np.max(evolution_data['spatial_extent']):.6f} m\n\n")
            
            # Optimal SPPARKS configuration
            f.write("OPTIMAL SPPARKS DOMAIN CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Recommended domain bounds:\n")
            f.write(f"  X: {optimal_bounds[0]:.6f} to {optimal_bounds[3]:.6f} m\n")
            f.write(f"  Y: {optimal_bounds[1]:.6f} to {optimal_bounds[4]:.6f} m\n")
            f.write(f"  Z: {optimal_bounds[2]:.6f} to {optimal_bounds[5]:.6f} m\n")
            
            # Calculate grid dimensions for different spacings
            spacings = [1e-6, 5e-7, 2e-7, 1e-7]
            f.write(f"\nGrid dimensions for different lattice spacings:\n")
            for spacing in spacings:
                nx = int((optimal_bounds[3] - optimal_bounds[0]) / spacing)
                ny = int((optimal_bounds[4] - optimal_bounds[1]) / spacing)
                nz = int((optimal_bounds[5] - optimal_bounds[2]) / spacing)
                total_sites = nx * ny * nz
                f.write(f"  {spacing:.0e} m: {nx} × {ny} × {nz} = {total_sites:.1e} sites\n")
            
            f.write(f"\nRecommendations:\n")
            f.write(f"- Use global domain bounds to capture all thermal data\n")
            f.write(f"- Monitor load balancing across different time layers\n")
            f.write(f"- Consider temporal load redistribution for long simulations\n")
            f.write(f"- Account for domain evolution in processor decomposition\n")
        
        print(f"Multi-layer analysis report saved to: {output_file}")


def main():
    """Example usage of multi-layer analysis"""
    
    hdf5_file = "/Users/Tron/Downloads/Huddle-resourcesfornewformat/reduced_thermal_output.hdf5"
    
    print("=== Multi-Layer HDF5 Domain Analysis ===")
    
    try:
        with HDF5Reader(hdf5_file) as reader:
            # Create multi-layer analyzer
            ml_analyzer = MultiLayerAnalyzer(reader)
            
            # Analyze global domain bounds
            print("\n1. Global Domain Analysis")
            global_min, global_max = ml_analyzer.get_global_domain_bounds()
            print(f"Global domain bounds:")
            print(f"  X: {global_min[0]:.6f} to {global_max[0]:.6f} m")
            print(f"  Y: {global_min[1]:.6f} to {global_max[1]:.6f} m")
            print(f"  Z: {global_min[2]:.6f} to {global_max[2]:.6f} m")
            
            # Analyze domain evolution
            print("\n2. Domain Evolution Analysis")
            evolution_data = ml_analyzer.analyze_chunk_evolution()
            print(f"Chunk count range: {np.min(evolution_data['chunk_counts'])} to {np.max(evolution_data['chunk_counts'])}")
            print(f"Domain volume range: {np.min(evolution_data['domain_volumes']):.2e} to {np.max(evolution_data['domain_volumes']):.2e} m³")
            
            # Get optimal SPPARKS domain
            print("\n3. Optimal SPPARKS Domain")
            optimal_bounds = ml_analyzer.get_optimal_spparks_domain()
            print(f"Recommended SPPARKS bounds:")
            print(f"  X: {optimal_bounds[0]:.6f} to {optimal_bounds[3]:.6f} m")
            print(f"  Y: {optimal_bounds[1]:.6f} to {optimal_bounds[4]:.6f} m")
            print(f"  Z: {optimal_bounds[2]:.6f} to {optimal_bounds[5]:.6f} m")
            
            # Temporal load balancing analysis
            print("\n4. Temporal Load Balancing Analysis")
            temporal_load_df = ml_analyzer.analyze_temporal_load_balancing(
                optimal_bounds, (2, 2, 2), sample_layers=list(range(0, reader.num_layers, 10)))
            
            # Calculate overall load balance statistics
            layer_groups = temporal_load_df.groupby('layer_idx')
            overall_imbalances = []
            for layer_idx, group in layer_groups:
                if len(group) > 0 and group['num_chunks'].mean() > 0:
                    normalized = group['num_chunks'] / group['num_chunks'].mean()
                    overall_imbalances.append(normalized.std())
            
            if overall_imbalances:
                print(f"Average load imbalance across time: {np.mean(overall_imbalances):.3f}")
                print(f"Load imbalance variation: {np.std(overall_imbalances):.3f}")
            
            # Create visualizations
            print("\n5. Creating Visualizations")
            
            # Domain evolution plot
            fig1 = ml_analyzer.plot_domain_evolution()
            plt.savefig('domain_evolution.png', dpi=150, bbox_inches='tight')
            print("Saved: domain_evolution.png")
            
            # Temporal load balance plot
            fig2 = ml_analyzer.plot_temporal_load_balance(temporal_load_df)
            plt.savefig('temporal_load_balance.png', dpi=150, bbox_inches='tight')
            print("Saved: temporal_load_balance.png")
            
            # Generate comprehensive report
            print("\n6. Generating Comprehensive Report")
            ml_analyzer.generate_multilayer_report()
            
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()