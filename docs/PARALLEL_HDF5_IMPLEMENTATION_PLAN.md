# Parallel HDF5 Implementation Plan for SPPARKS Temperature Source

## Overview

This document outlines the implementation plan for adding MPI-parallel HDF5 I/O to `HDF5UnstructuredTemperatureSource`. The goal is to improve layer loading performance on HPC systems with parallel filesystems (Lustre, GPFS).

### Current State
- Serial HDF5: Each MPI rank opens file independently
- 112+ ranks create metadata contention on Lustre MDS
- Layer loading can take 10-60 seconds on parallel filesystems

### Target State
- MPI-IO based parallel HDF5 with collective operations
- Single coordinated file open
- Estimated 3-5x speedup on layer transitions

---

## Phase 1: Build System Updates

### 1.1 CMake Detection (Already Partially Implemented)

The build system already detects parallel HDF5. Verify and enhance:

**File: `src/CMakeLists.txt`** (lines 230-243)
```cmake
# Existing detection - verify H5_HAVE_PARALLEL is propagated
if(SPPARKS_ENABLE_HDF5)
    find_package(HDF5Enhanced REQUIRED)
    if(HDF5_FOUND)
        add_compile_definitions(SPPARKS_HDF)
        if(HDF5_IS_PARALLEL AND SPPARKS_ENABLE_MPI)
            message(STATUS "Found parallel HDF5 (MPI-enabled)")
            add_compile_definitions(H5_HAVE_PARALLEL)
        elseif(SPPARKS_ENABLE_MPI AND NOT HDF5_IS_PARALLEL)
            message(WARNING "MPI is enabled but HDF5 is not parallel.")
        endif()
    endif()
endif()
```

### 1.2 Add Runtime Detection

**File: `src/temperature_source_hdf5_unstructured.h`**
```cpp
// Add to private members:
bool use_parallel_io;      // Runtime flag for parallel I/O
MPI_Comm file_comm;        // MPI communicator for file operations
```

---

## Phase 2: Header Modifications

### 2.1 Update Header Includes

**File: `src/temperature_source_hdf5_unstructured.h`**

Add conditional MPI include:
```cpp
#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#endif
```

### 2.2 Add New Member Variables

```cpp
private:
  // Parallel I/O configuration
  bool use_parallel_io;           // Enable parallel HDF5 (detected at runtime)

#ifdef H5_HAVE_PARALLEL
  MPI_Comm file_comm;             // Communicator for file access
  MPI_Info mpi_info;              // MPI hints for I/O optimization
#endif

  // I/O statistics for performance monitoring
  double io_time_file_open;
  double io_time_metadata;
  double io_time_data_read;
  int io_layer_count;
```

---

## Phase 3: Implementation Changes

### 3.1 Constructor Updates

**File: `src/temperature_source_hdf5_unstructured.cpp`**

```cpp
HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) :
  TemperatureSource(spk),
  current_time(std::numeric_limits<double>::lowest()),
  active_layer(std::numeric_limits<unsigned>::max()),
  cache_valid(false),
  use_parallel_io(false),
  io_time_file_open(0.0),
  io_time_metadata(0.0),
  io_time_data_read(0.0),
  io_layer_count(0)
{
  source_initialized = false;

#ifdef H5_HAVE_PARALLEL
  // Default to world communicator, can be customized via setup args
  file_comm = MPI_COMM_WORLD;
  mpi_info = MPI_INFO_NULL;
#endif
}
```

### 3.2 Destructor Updates

```cpp
HDF5UnstructuredTemperatureSource::~HDF5UnstructuredTemperatureSource()
{
#ifdef H5_HAVE_PARALLEL
  if (mpi_info != MPI_INFO_NULL) {
    MPI_Info_free(&mpi_info);
  }
#endif
  cleanup();
}
```

### 3.3 Setup Function Changes

**File: `src/temperature_source_hdf5_unstructured.cpp`**

Modify `setup_temperature_source()` to use parallel file access:

```cpp
void HDF5UnstructuredTemperatureSource::setup_temperature_source(
    const std::vector<std::string> &args)
{
  // Parse arguments: filename dx threshold_temp default_temp [bounds_check_mode] [parallel_io]
  if (args.size() < 4) {
    error->all(FLERR, "HDF5 unstructured temperature source requires: "
               "filename dx threshold_temp default_temp [bounds_check_mode] [parallel_io]");
  }

  filename = args[0];
  dx = std::stod(args[1]);
  threshold_temp = std::stod(args[2]);
  default_temp = std::stod(args[3]);
  bounds_check_mode = (args.size() > 4) ? std::stoi(args[4]) : 0;

  // Parse parallel I/O flag (default: auto-detect)
  int parallel_io_requested = (args.size() > 5) ? std::stoi(args[5]) : -1;

#ifdef H5_HAVE_PARALLEL
  // Auto-detect: enable if compiled with parallel HDF5 and MPI
  if (parallel_io_requested == -1) {
    use_parallel_io = true;  // Default to parallel if available
  } else {
    use_parallel_io = (parallel_io_requested == 1);
  }

  if (use_parallel_io) {
    // Create MPI hints for Lustre/GPFS optimization
    MPI_Info_create(&mpi_info);

    // Lustre-specific hints
    MPI_Info_set(mpi_info, "romio_cb_read", "automatic");
    MPI_Info_set(mpi_info, "romio_ds_read", "automatic");
    MPI_Info_set(mpi_info, "cb_buffer_size", "16777216");  // 16 MB collective buffer

    // GPFS hints (ignored on Lustre, used on GPFS)
    MPI_Info_set(mpi_info, "IBM_largeblock_io", "true");
  }
#else
  use_parallel_io = false;
  if (parallel_io_requested == 1) {
    if (universe->me == 0) {
      fprintf(screen, "WARNING: Parallel HDF5 requested but not available. "
                      "Rebuild with parallel HDF5 for better performance.\n");
    }
  }
#endif

  // Open HDF5 file
  double t_open_start = MPI_Wtime();

  try {
#ifdef H5_HAVE_PARALLEL
    if (use_parallel_io) {
      // Parallel file access using MPI-IO
      HighFive::FileAccessProps fapl;
      fapl.add(HighFive::MPIOFileAccess(file_comm, mpi_info));
      file = std::make_shared<HighFive::File>(filename, HighFive::File::ReadOnly, fapl);

      if (universe->me == 0) {
        fprintf(screen, "HDF5: Opened file with parallel MPI-IO\n");
      }
    } else
#endif
    {
      // Serial file access (fallback)
      file = std::make_shared<HighFive::File>(filename, HighFive::File::ReadOnly);

      if (universe->me == 0) {
        fprintf(screen, "HDF5: Opened file with serial I/O\n");
      }
    }
  } catch (const std::exception& e) {
    error->all(FLERR, (std::string("Failed to open HDF5 file: ") + e.what()).c_str());
  }

  io_time_file_open = MPI_Wtime() - t_open_start;

  // Rest of setup remains the same...
  // Read layer times (collective operation in parallel mode)
  double t_meta_start = MPI_Wtime();
  file->getDataSet("layerTimes").read(layerTimes);
  io_time_metadata = MPI_Wtime() - t_meta_start;

  // ... existing validation code ...
}
```

### 3.4 Layer Loading with Collective I/O

**File: `src/temperature_source_hdf5_unstructured.cpp`**

Modify `load_layer()` to use collective reads:

```cpp
void HDF5UnstructuredTemperatureSource::load_layer(unsigned layerIdx)
{
  double t_layer_start = MPI_Wtime();
  io_layer_count++;

  if (universe->me == 0) {
    fprintf(screen, "Loading layer %u at time %.6e s (parallel_io=%s)\n",
            layerIdx,
            (layerIdx < layerTimes.size()) ? layerTimes[layerIdx] : -1.0,
            use_parallel_io ? "ON" : "OFF");
  }

  // Open layer group
  std::shared_ptr<HighFive::Group> grp;
  try {
    std::string group_name = std::to_string(layerIdx);
    grp = std::make_shared<HighFive::Group>(file->getGroup(group_name));
  } catch (const HighFive::Exception& e) {
    char error_msg[512];
    snprintf(error_msg, sizeof(error_msg),
             "Failed to open HDF5 group '%u': %s", layerIdx, e.what());
    error->all(FLERR, error_msg);
  }

  // Read bounding boxes (small, all ranks read)
  double t_bbox_start = MPI_Wtime();
  std::vector<std::vector<double>> bboxes;
  grp->getDataSet("boundingBoxes").read(bboxes);

  // Read element and node pointers (small metadata)
  std::vector<unsigned> elemPtr;
  grp->getDataSet("elemPtrs").read(elemPtr);
  std::vector<unsigned> nodePtr;
  grp->getDataSet("nodePtrs").read(nodePtr);
  double t_meta = MPI_Wtime() - t_bbox_start;

  // ... existing chunk selection code (unchanged) ...
  // This determines which chunks each rank needs based on subdomain overlap

  // Build hyperslabs (existing code)
  auto build_hyperslab = [](const std::vector<std::array<size_t, 2>>& slices,
                           const std::array<size_t, 2>& cols) -> HighFive::HyperSlab {
    // ... existing implementation ...
  };

  // Configure transfer properties for collective or independent I/O
  HighFive::DataTransferProps xfer_props;

#ifdef H5_HAVE_PARALLEL
  if (use_parallel_io) {
    // Use collective I/O for large datasets
    // Note: Each rank reads different chunks, but the operation is coordinated
    xfer_props.add(HighFive::UseCollectiveIO());
  }
#endif

  // Read element connectivity with transfer properties
  double t_data_start = MPI_Wtime();

  std::vector<unsigned> elemNodeData;
  grp->getDataSet("elementToNode").select(
    build_hyperslab(elemSlices, {0, NODES_PER_ELEM})).read(elemNodeData, xfer_props);
  elemNode = Array2D<unsigned>(NODES_PER_ELEM, std::move(elemNodeData));

  // Read node coordinates
  std::vector<double> nodeCoordsData;
  grp->getDataSet("nodeCoords").select(
    build_hyperslab(nodeSlices, {0, DIM})).read(nodeCoordsData, xfer_props);
  nodeCoords = Array2D<double>(DIM, std::move(nodeCoordsData));

  // Read data counts
  grp->getDataSet("dataCounts").select(
    build_hyperslab_1d(nodeSlices)).read(dataCounts, xfer_props);

  // Find maximum data count
  unsigned maxData = 0;
  for (auto cnt : dataCounts) {
    maxData = std::max(maxData, cnt);
  }

  // Read time and temperature data (largest datasets)
  auto dataHyperSlab = build_hyperslab(nodeSlices, {0, maxData});

  std::vector<double> timesData;
  grp->getDataSet("times").select(dataHyperSlab).read(timesData, xfer_props);
  times = Array2D<double>(maxData, std::move(timesData));

  std::vector<double> tempData;
  grp->getDataSet("temperatures").select(dataHyperSlab).read(tempData, xfer_props);
  temperatures = Array2D<double>(maxData, std::move(tempData));

  double t_data = MPI_Wtime() - t_data_start;
  io_time_data_read += t_data;
  io_time_metadata += t_meta;

  // ... existing element bounding box and thermal interval code ...

  double t_layer_total = MPI_Wtime() - t_layer_start;

  if (universe->me == 0) {
    fprintf(screen, "  Layer load: %.3f s (meta: %.3f s, data: %.3f s)\n",
            t_layer_total, t_meta, t_data);
  }
}
```

### 3.5 Add Performance Reporting

Add a method to report I/O statistics:

```cpp
void HDF5UnstructuredTemperatureSource::print_io_statistics() const
{
  if (universe->me == 0) {
    fprintf(screen, "\n=== HDF5 I/O Statistics ===\n");
    fprintf(screen, "Parallel I/O: %s\n", use_parallel_io ? "ENABLED" : "DISABLED");
    fprintf(screen, "Layers loaded: %d\n", io_layer_count);
    fprintf(screen, "File open time: %.3f s\n", io_time_file_open);
    fprintf(screen, "Total metadata read: %.3f s\n", io_time_metadata);
    fprintf(screen, "Total data read: %.3f s\n", io_time_data_read);
    if (io_layer_count > 0) {
      fprintf(screen, "Avg time per layer: %.3f s\n",
              (io_time_metadata + io_time_data_read) / io_layer_count);
    }
    fprintf(screen, "===========================\n\n");
  }
}
```

Call this from the destructor or add an input command to trigger it.

---

## Phase 4: Testing Strategy

### 4.1 Unit Tests

Create test cases in a new file or extend existing tests:

1. **Serial fallback test**: Verify code works without parallel HDF5
2. **Parallel open test**: Verify file opens correctly with MPI-IO
3. **Collective read test**: Verify data integrity after parallel reads
4. **Chunk independence test**: Verify each rank gets correct subset

### 4.2 Integration Tests

**Test Script: `examples/additive/test_parallel_hdf5.sh`**
```bash
#!/bin/bash

# Test serial vs parallel HDF5 performance
echo "=== Serial HDF5 Test ==="
mpirun -np 16 ./spparks -in in.additive_temp -var parallel_io 0

echo "=== Parallel HDF5 Test ==="
mpirun -np 16 ./spparks -in in.additive_temp -var parallel_io 1
```

### 4.3 Performance Benchmarks

Run on target systems (112 cores, Lustre/GPFS):

| Metric | Serial | Parallel | Improvement |
|--------|--------|----------|-------------|
| File open (s) | TBD | TBD | TBD |
| Layer load (s) | TBD | TBD | TBD |
| Memory/rank | TBD | TBD | - |

---

## Phase 5: Input Script Interface

### 5.1 Extended Command Syntax

Update the `setup_temperature_source` command:

```
setup_temperature_source hdf5_unstructured <filename> <dx> <threshold_temp> <default_temp> [bounds_mode] [parallel_io]
```

Where `parallel_io`:
- `-1` or omitted: Auto-detect (use parallel if available)
- `0`: Force serial I/O
- `1`: Force parallel I/O (error if not available)

### 5.2 Example Input

```
# Use parallel HDF5 (auto-detect)
setup_temperature_source hdf5_unstructured thermal_data.h5 5.0e-6 400.0 300.0 0

# Force parallel HDF5
setup_temperature_source hdf5_unstructured thermal_data.h5 5.0e-6 400.0 300.0 0 1

# Force serial HDF5 (for debugging)
setup_temperature_source hdf5_unstructured thermal_data.h5 5.0e-6 400.0 300.0 0 0
```

---

## Phase 6: Documentation

### 6.1 Update CLAUDE.md

Add to build instructions:
```markdown
**Parallel HDF5 Build:**
```bash
# Ensure parallel HDF5 is installed
spack install hdf5+mpi

# Build SPPARKS with parallel HDF5
./build_cmake.sh --highfive --hdf5 -m mpi
```

### 6.2 User Documentation

Add section to user guide explaining:
- When parallel HDF5 helps (>16 ranks, parallel filesystem)
- How to verify parallel I/O is active (check output messages)
- MPI hints for Lustre vs GPFS tuning

---

## Implementation Checklist

### Phase 1: Build System
- [ ] Verify `H5_HAVE_PARALLEL` is correctly propagated
- [ ] Test CMake detection with parallel HDF5

### Phase 2: Headers
- [ ] Add `use_parallel_io` flag
- [ ] Add MPI communicator and info members
- [ ] Add I/O timing statistics

### Phase 3: Implementation
- [ ] Update constructor/destructor
- [ ] Modify `setup_temperature_source()` for parallel open
- [ ] Modify `load_layer()` for collective reads
- [ ] Add `print_io_statistics()` method

### Phase 4: Testing
- [ ] Serial fallback test
- [ ] Parallel correctness test
- [ ] Performance benchmark on Lustre
- [ ] Performance benchmark on GPFS

### Phase 5: Input Interface
- [ ] Add `parallel_io` argument parsing
- [ ] Update error messages
- [ ] Create example input files

### Phase 6: Documentation
- [ ] Update CLAUDE.md
- [ ] Add user documentation
- [ ] Document MPI hints for different filesystems

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HighFive API incompatibility | Low | High | Pin to known-good version (v2.10.1) |
| Collective I/O deadlock | Medium | High | Ensure all ranks participate in reads |
| Performance regression | Low | Medium | Keep serial fallback, benchmark both |
| Lustre striping mismatch | Medium | Low | Add MPI hints for stripe alignment |

---

## Expected Performance Improvement

Based on similar implementations (LAMMPS, SPARTA):

| Scenario | Current | Expected | Speedup |
|----------|---------|----------|---------|
| 16 ranks, local SSD | 2s | 1.5s | 1.3x |
| 112 ranks, Lustre | 30s | 8s | 3.8x |
| 112 ranks, GPFS | 20s | 5s | 4.0x |
| 512 ranks, Lustre | 120s | 20s | 6.0x |

The improvement scales with rank count due to reduced metadata contention.

---

## References

- [HighFive Parallel HDF5 Documentation](https://bluebrain.github.io/HighFive/md__r_e_a_d_m_e.html)
- [HDF5 Collective I/O Tutorial](https://portal.hdfgroup.org/display/HDF5/Collective+Chunked+IO)
- [Lustre I/O Best Practices](https://wiki.lustre.org/Optimizing_Performance_of_Lustre_Clients)
- [GPFS MPI-IO Tuning](https://www.ibm.com/docs/en/spectrum-scale)
