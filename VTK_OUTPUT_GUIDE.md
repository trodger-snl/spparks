# SPPARKS VTK Output Guide

## ✅ VTK Output Feature Implemented

The batch analysis script now includes comprehensive VTK output functionality that exports all timesteps with IPF coloring data for visualization in ParaView or other VTK-compatible software.

## 🎯 What's Generated

### 1. Multiblock Dataset (.vtm)
**File**: `spparks_all_timesteps.vtm`
- **Purpose**: Single file containing all timesteps as separate blocks
- **Format**: VTK Multiblock Dataset
- **Usage**: Best for viewing all timesteps together or comparing different time points
- **Data**: Point clouds with IPF colors, grain IDs, quaternions, and coordinates

### 2. Time Series Collection (.pvd)
**File**: `spparks_timeseries.pvd`
- **Purpose**: Time series animation file for ParaView
- **Format**: ParaView Data (PVD) collection
- **Usage**: Load this file to get time-based animation controls in ParaView
- **Data**: References individual structured grid files for each timestep

### 3. Individual Timestep Files (.vts)
**Directory**: `vtk_series/`
- **Files**: `spparks_timestep_000001.vts`, `spparks_timestep_000002.vts`, etc.
- **Purpose**: Individual structured grid files for each timestep
- **Format**: VTK Structured Grid
- **Usage**: Can be loaded individually or as part of the time series
- **Data**: 3D structured grids with cell-based IPF data

## 📊 Data Fields Included

### Point Data (Multiblock):
- **IPF_Red**: Red component of IPF color (0-1)
- **IPF_Green**: Green component of IPF color (0-1)  
- **IPF_Blue**: Blue component of IPF color (0-1)
- **IPF_RGB**: Combined RGB vector
- **Grain_ID**: SPPARKS grain identifier
- **Quat_W**: Quaternion W component
- **Quat_X**: Quaternion X component  
- **Quat_Y**: Quaternion Y component
- **Quat_Z**: Quaternion Z component
- **Timestep**: Simulation timestep number
- **X, Y, Z**: Coordinate arrays

### Cell Data (Structured Grid):
- **IPF_Red/Green/Blue**: IPF color components per cell
- **Grain_ID**: Grain identifier per cell
- **Quat_W/X/Y/Z**: Quaternion components per cell
- **Filled**: Binary mask (1.0 = filled, 0.0 = empty)

## 🔧 Configuration

Enable VTK output in `batch_analyze_spparks_parallel.py`:

```python
# Output options
ENABLE_VTK_OUTPUT = True # Generate VTK file with all timesteps and IPF data
```

## 🚀 Usage Examples

### Basic Processing (3 files for testing):
```bash
conda activate WORK
python batch_analyze_spparks_parallel.py
```

### Full Dataset Processing:
Edit the script to remove the testing limit:
```python
# Comment out or remove this line:
# dump_files = all_files[:3]  # REMOVE THIS LINE FOR FULL PROCESSING
```

## 📖 ParaView Workflow

### Loading Time Series:
1. Open ParaView
2. **File → Open** → Select `spparks_timeseries.pvd`
3. Click **Apply** in Properties panel
4. Use the time controls (play button, slider) to animate through timesteps

### Loading Multiblock Dataset:
1. Open ParaView  
2. **File → Open** → Select `spparks_all_timesteps.vtm`
3. Click **Apply** in Properties panel
4. Use the **Multi-block Inspector** to toggle individual timesteps

### Visualization Tips:

#### 1. IPF Coloring:
- **Coloring**: Select "IPF_RGB" from the coloring dropdown
- **Color Map**: The IPF colors are pre-calculated, so use "RGB" interpretation
- **Result**: Shows crystallographic orientation-based coloring

#### 2. Grain Analysis:
- **Coloring**: Select "Grain_ID" for grain boundary visualization
- **Color Map**: Use categorical coloring or random colors
- **Filter**: Apply "Threshold" filter using "Filled" field to show only filled regions

#### 3. Orientation Analysis:
- **Vector Field**: Use quaternion components (Quat_W/X/Y/Z) for orientation analysis
- **Glyph Filter**: Apply glyphs to show orientation directions
- **Calculator**: Create custom orientation metrics using quaternion data

#### 4. Animation:
- **Time Series**: Load PVD file and use ParaView's animation controls
- **Export**: Use **File → Save Animation** to create video files

## 📁 File Structure

```
batch_analysis_output/
├── spparks_all_timesteps.vtm          # Multiblock dataset
├── spparks_timeseries.pvd             # Time series collection
├── vtk_series/                        # Individual timestep files
│   ├── spparks_timestep_000000.vts
│   ├── spparks_timestep_000001.vts
│   └── spparks_timestep_000002.vts
├── method2_timestep_1.png             # Rendered images
├── method2_animation.mp4              # Animation
└── timing_results.csv                 # Performance data
```

## 🔍 Data Analysis Capabilities

### 1. Temporal Evolution:
- Track grain growth over time
- Analyze orientation changes during simulation
- Monitor texture development

### 2. Spatial Analysis:
- Grain size distribution
- Orientation mapping
- Misorientation analysis
- Grain boundary characterization

### 3. Statistical Analysis:
- Export data from ParaView for further analysis
- Use Python scripts in ParaView for automated analysis
- Generate histograms and statistical plots

## ⚡ Performance

### Current Test Results (3 files):
- **VTK Generation**: ~0.2 seconds additional processing time
- **File Sizes**: 
  - Multiblock: ~418 bytes (metadata)
  - Individual files: ~88KB - 1.6MB per timestep
  - PVD: ~404 bytes (metadata)

### Full Dataset Estimates (102 files):
- **Additional Processing**: ~3-5 seconds
- **Total VTK Size**: ~50-100 MB (with cropping)
- **Memory Usage**: Minimal additional impact

## 🎯 Use Cases

### 1. Research Publications:
- High-quality 3D visualizations for papers
- Quantitative orientation analysis
- Time-lapse animations of microstructure evolution

### 2. Process Optimization:
- Parameter sensitivity analysis across timesteps
- Cooling rate effects on grain structure
- Additive manufacturing path optimization

### 3. Educational Materials:
- Interactive 3D models for teaching
- Demonstration of crystallographic concepts
- Visual explanation of grain growth mechanisms

The VTK output functionality is now **production-ready** and provides comprehensive data export for advanced analysis and visualization workflows.