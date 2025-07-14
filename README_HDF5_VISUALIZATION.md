# HDF5 Unstructured Domain Visualization Tools for SPPARKS

This directory contains Python tools for visualizing and analyzing unstructured HDF5 thermal domains to optimize SPPARKS subdomain decomposition for parallel additive manufacturing simulations.

## Overview

The unstructured HDF5 format stores thermal data on finite element meshes with spatially-varying resolution. Unlike regular grids, this format requires spatial interpolation and careful analysis to understand domain geometry and optimize parallel processing.

## Tools Provided

### 1. `hdf5_domain_visualizer.py` - Core Analysis Tool

**Features:**
- **HDF5Reader**: Efficient reading of unstructured thermal domain files
- **DomainVisualizer**: 3D visualization of thermal domains and chunk distributions
- **SubdomainAnalyzer**: SPPARKS grid analysis and optimization
- **Automated reporting**: Statistics and SPPARKS configuration generation

**Usage:**
```bash
python hdf5_domain_visualizer.py
```

**Outputs:**
- `domain_overview.png` - 3D visualization of chunk distribution
- `chunk_distribution.png` - Statistical analysis of chunk sizes
- `spparks_overlay.png` - SPPARKS grid overlay on thermal domain
- `spparks_config.txt` - Generated SPPARKS input configuration

### 2. `hdf5_interactive_explorer.py` - Interactive GUI Tool

**Features:**
- Interactive GUI for parameter adjustment
- Real-time load balancing analysis
- 3D visualization with PyVista (optional)
- Dynamic processor decomposition testing
- Export capabilities for different configurations

**Usage:**
```bash
# Launch GUI
python hdf5_interactive_explorer.py

# Auto-load specific file
python hdf5_interactive_explorer.py path/to/thermal_data.hdf5
```

## Installation Requirements

### Basic Requirements
```bash
pip install h5py numpy matplotlib pandas
```

### Advanced 3D Visualization (Optional)
```bash
pip install pyvista
```

## Understanding the Analysis Results

### Domain Information

The example thermal domain has these characteristics:
- **Spatial Extent**: 1.92mm × 1.92mm × 7.79mm (thin, elongated geometry)
- **Chunk Organization**: 151 chunks with varying sizes and node densities
- **Memory Usage**: ~96 MB for complete layer data
- **Time Evolution**: 34 layers covering 0-165 seconds

### Load Balancing Analysis

From the example analysis:

**4-Processor Decomposition (2×2×1):**
- Load imbalance: 0.012 (excellent)
- Processor sizes: 1.96mm × 1.96mm × 7.79mm
- **Recommended**: Best balance for this geometry

**8-Processor Decomposition (2×2×2):**
- Load imbalance: 0.898 (poor)
- Processor sizes: 1.96mm × 1.96mm × 3.90mm
- **Issue**: Z-direction split creates uneven chunk distribution

**16-Processor Decomposition (8×2×1):**
- Load imbalance: 1.064 (poor)
- Processor sizes: 0.49mm × 1.96mm × 7.79mm
- **Issue**: Too fine X-direction splitting

### Key Insights

1. **Geometry Matters**: The elongated Z-direction makes splitting along Z problematic
2. **Chunk Distribution**: Non-uniform chunk sizes cause load imbalance
3. **Optimal Strategy**: Keep processor aspect ratios similar to domain aspect ratios

## Generated SPPARKS Configuration

The tools generate complete SPPARKS input configurations:

```
# Domain definition
region box block 0 3920 0 3920 0 7790
create_box box
create_sites box

# Processor decomposition: 2×2×2 = 8 processors
# Run with: mpirun -np 8 spk_executable < input_file

# Physical domain bounds (meters):
# X: -0.001960 to 0.001960
# Y: -0.001960 to 0.001960  
# Z: -0.006760 to 0.001030

# Lattice spacing: 1.00e-06 m
# Grid dimensions: 3920 × 3920 × 7790

# Temperature source configuration
temperature hdf5_unstructured thermal_data.hdf5 1.00e-06
```

## Workflow Recommendations

### 1. Initial Domain Exploration
```bash
python hdf5_domain_visualizer.py
```
- Understand overall domain geometry
- Check memory requirements
- Get baseline decomposition recommendations

### 2. Interactive Optimization
```bash
python hdf5_interactive_explorer.py your_file.hdf5
```
- Test different processor counts
- Analyze load balancing in real-time
- Export optimized configurations

### 3. Production Setup
- Use generated SPPARKS configuration
- Test with small processor counts first
- Monitor memory usage and performance
- Scale up based on load balance analysis

## Understanding Chunk Organization

The HDF5 file structure contains:
```
/layerTimes              # Time values for each layer
/0/                      # Layer 0 data
  boundingBoxes          # Chunk spatial bounds [N_chunks × 6]
  nodeCoords            # Node coordinates [N_nodes × 3]
  elementToNode         # Element connectivity [N_elements × 4]
  elemPtrs/nodePtrs     # Chunk indexing arrays
  temperatures/times    # Reduced thermal history [N_nodes × max_points]
  dataCounts           # Number of time points per node
/1/                      # Layer 1 data
...
```

Each chunk contains:
- **Spatial region**: Defined by bounding box
- **Mesh data**: Tetrahedral elements and nodes
- **Thermal history**: Time-temperature data per node
- **Reduced representation**: Adaptive time sampling

## Performance Considerations

### Memory Usage
- Complete layer: ~96 MB
- Per-processor: Depends on chunk overlap
- Cache effects: Pre-computed interpolation weights

### Parallel Efficiency
- **Good**: Load balanced chunk distribution
- **Poor**: Few chunks per processor
- **Critical**: Boundary handling between processors

### Scaling Guidelines
- **4-8 processors**: Good for development/testing
- **16-32 processors**: Production simulations
- **>64 processors**: Requires careful load balancing

## Troubleshooting

### Common Issues

**"Cannot open HDF5 file"**
- Check file path and permissions
- Ensure h5py is installed correctly

**"Poor load balancing"**
- Try different processor decompositions
- Consider domain geometry in splitting strategy
- Use interactive tool to test alternatives

**"Memory issues"**
- Reduce number of layers loaded simultaneously
- Consider chunked processing for large domains
- Monitor per-processor memory usage

**"Visualization problems"**
- Install PyVista for advanced 3D features
- Use matplotlib-only version for basic analysis
- Check display settings for remote connections

### Getting Help

The tools provide detailed error messages and logging. For additional support:

1. Check console output for specific error details
2. Verify HDF5 file format matches expected structure
3. Test with smaller processor counts first
4. Review load balancing statistics for optimization hints

## Integration with SPPARKS

### Input File Setup
1. Use generated region commands
2. Specify correct temperature source
3. Set appropriate lattice spacing
4. Configure processor decomposition

### Runtime Considerations
- MPI processor count must match decomposition
- HDF5 file must be accessible to all processors
- Monitor memory usage during simulation
- Check interpolation accuracy at processor boundaries

This comprehensive analysis framework enables optimal SPPARKS parallel simulation setup for complex unstructured thermal domains.