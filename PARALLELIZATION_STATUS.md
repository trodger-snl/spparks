# SPPARKS Batch Analysis - Parallelization Status

## ✅ Current Status: Working with Serial Processing

The batch analysis script is **fully functional** with serial processing. Parallelization is temporarily disabled due to PyVista threading conflicts on macOS.

## 🚀 Performance Results

### Current Performance (Serial):
- **Processing Speed**: ~4 files per second
- **Average per file**: 0.25 seconds (with cropping to z=0-10)
- **Total estimated time**: ~25 seconds for 102 files (with cropping)

### Configuration Used:
```python
ENABLE_METHOD1 = False   # Point cloud with glyphs
ENABLE_METHOD2 = True    # Structured grid (working)
PARALLEL_FILES = False   # Disabled due to macOS threading issues
CROP_BOUNDS = [0,200,0,200,0,10]  # Crop to thin slab for speed
```

## 🔧 How to Use

### 1. Test Run (3 files):
Current configuration processes only 3 files for testing.

### 2. Full Run (102 files):
Edit `/Users/Tron/spparks/batch_analyze_spparks.py` line 504:
```python
# Remove or comment out this line:
# dump_files = all_files[:3]  # REMOVE THIS LINE FOR FULL PROCESSING
```

### 3. Configuration Options:

#### Speed vs Quality:
```python
# Fastest (current setup):
ENABLE_METHOD1 = False
ENABLE_METHOD2 = True
CROP_BOUNDS = [0,200,0,200,0,10]  # Thin slab

# Full quality:
ENABLE_METHOD1 = True
ENABLE_METHOD2 = True
CROP_BOUNDS = None  # No cropping
```

#### Different Subvolumes:
```python
# Central 100x100x50 region:
CROP_BOUNDS = [50, 150, 50, 150, 25, 75]

# Top layer only:
CROP_BOUNDS = [0, 200, 0, 200, 190, 199]
```

## ⚠️ Parallelization Issues

### Problem:
PyVista on macOS has threading conflicts that cause crashes:
```
NSWindow should only be instantiated on the main thread!
```

### Solutions Attempted:
1. ✅ **ThreadPoolExecutor**: Works but causes GUI conflicts
2. ❌ **ProcessPoolExecutor**: Serialization issues with functions
3. ✅ **Serial Processing**: Working perfectly

### Future Solutions:
1. **Container-based processing**: Use Docker with headless PyVista
2. **Different backend**: Switch to VTK-only rendering
3. **Process isolation**: Use subprocess calls instead of threading

## 📊 Output

The script generates:
- **Individual images**: `method2_timestep_X.png` for each timestep
- **Animation**: `method2_animation.mp4` at 5 FPS
- **Timing data**: `timing_results.csv` with performance metrics

## 🎯 Recommendations

### For Current Use:
1. **Keep serial processing** - it's fast enough for your cropped data
2. **Use Method 2 only** - structured grid works well
3. **Use cropping** - reduces processing time significantly

### For Large-Scale Analysis:
1. **Multiple crop regions**: Run script multiple times with different `CROP_BOUNDS`
2. **Different methods**: Compare Method 1 vs Method 2 on subsets
3. **Full resolution**: Remove cropping for final publication images

## 🚀 Performance Projections

With current settings:
- **102 files**: ~25 seconds
- **Full resolution (no crop)**: ~10-15 minutes
- **Both methods + full resolution**: ~20-30 minutes

The serial processing is actually quite efficient for your use case!