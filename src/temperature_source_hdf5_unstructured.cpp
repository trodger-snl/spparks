/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#include "temperature_source_hdf5_unstructured.h"
#include "domain.h"
#include "error.h"
#include "universe.h"
#include "app.h"
#include "highfive/highfive.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <set>
#include <fstream>
#include <sstream>

using namespace SPPARKS_NS;

// Helper function to get current memory usage (Linux only)
static size_t get_memory_usage_kb() {
#ifdef __linux__
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      std::istringstream iss(line.substr(6));
      size_t kb;
      iss >> kb;
      return kb;
    }
  }
#endif
  return 0;
}

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) :
  TemperatureSource(spk),
  current_time(std::numeric_limits<double>::lowest()),
  active_layer(std::numeric_limits<unsigned>::max()),
  cache_valid(false),
  total_layer_load_time(0.0),
  layer_load_count(0),
  cached_nodal_time(std::numeric_limits<double>::lowest()),
  nodal_cache_valid(false),
  grid_cell_size_multiplier(100.0),
  use_spatial_grid(true),
  use_element_cache(true)
{
  source_initialized = false;
#ifdef H5_HAVE_PARALLEL
  use_parallel_hdf5 = false;
#endif
}

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::~HDF5UnstructuredTemperatureSource()
{
  cleanup();
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  // Parse arguments: filename dx threshold_temp default_temp [bounds_check_mode] [grid_cell_mult]
  if (args.size() < 4) {
    error->all(FLERR, "HDF5 unstructured temperature source requires: filename dx threshold_temp default_temp [bounds_check_mode] [grid_cell_mult]");
  }

  filename = args[0];
  dx = std::stod(args[1]);  // Grid spacing in meters
  threshold_temp = std::stod(args[2]);  // Threshold temperature for fast-forward
  default_temp = std::stod(args[3]);    // Default/ambient temperature
  bounds_check_mode = (args.size() > 4) ? std::stoi(args[4]) : 0;
  grid_cell_size_multiplier = (args.size() > 5) ? std::stod(args[5]) : 100.0;

  // Open HDF5 file with parallel I/O when available
  try {
#ifdef H5_HAVE_PARALLEL
    // Create file access property list with MPI-IO
    HighFive::FileAccessProps fapl;
    fapl.add(HighFive::MPIOFileAccess(universe->uworld, MPI_INFO_NULL));
    // NOTE: Do NOT enable MPIOCollectiveMetadata here.  With per-rank
    // subdomain chunk filtering, some ranks may load zero chunks and
    // return early from load_layer(), skipping getDataSet() calls that
    // other ranks execute.  Collective metadata would deadlock in that case.
    file = std::make_shared<HighFive::File>(filename, HighFive::File::ReadOnly, fapl);
    use_parallel_hdf5 = true;
    if (universe->me == 0) {
      fprintf(screen, "  Parallel HDF5: ENABLED (MPI-IO)\n");
    }
#else
    file = std::make_shared<HighFive::File>(filename, HighFive::File::ReadOnly);
    if (universe->me == 0) {
      fprintf(screen, "  Parallel HDF5: DISABLED (serial build)\n");
    }
#endif
  } catch (const std::exception& e) {
    error->all(FLERR, (std::string("Failed to open HDF5 file: ") + e.what()).c_str());
  }

  // Read layer times
  file->getDataSet("layerTimes").read(layerTimes);

  // Trim trailing zeros from layerTimes (handles pre-allocated HDF5 arrays)
  size_t original_size = layerTimes.size();
  size_t valid_entries = layerTimes.size();

  for (size_t i = layerTimes.size(); i > 0; --i) {
    if (layerTimes[i-1] > 0.0) {
      valid_entries = i;
      break;
    }
  }

  if (valid_entries < layerTimes.size()) {
    if (universe->me == 0) {
      fprintf(screen, "LayerTimes: Trimmed %zu trailing zero entries\n", original_size - valid_entries);
    }
    layerTimes.resize(valid_entries);
  }

  // Data validation
  if (universe->me == 0) {
    if (layerTimes.empty()) {
      error->all(FLERR, "layerTimes array is empty");
    }

    // Check for monotonically increasing values
    bool is_monotonic = true;
    size_t first_violation = 0;
    for (size_t i = 1; i < layerTimes.size(); i++) {
      if (layerTimes[i] <= layerTimes[i-1]) {
        is_monotonic = false;
        first_violation = i;
        break;
      }
    }

    if (is_monotonic) {
      fprintf(screen, "LayerTimes validation: PASSED\n");
    } else {
      fprintf(screen, "WARNING: Monotonicity check FAILED at index %zu\n", first_violation);
    }

    // Check for negative values
    size_t negative_count = 0;
    for (size_t i = 0; i < layerTimes.size(); i++) {
      if (layerTimes[i] < 0.0) {
        negative_count++;
      }
    }

    if (negative_count > 0) {
      fprintf(screen, "WARNING: Found %zu negative values in layerTimes\n", negative_count);
    }

    // Check for duplicate consecutive values
    size_t duplicate_count = 0;
    for (size_t i = 1; i < layerTimes.size(); i++) {
      if (layerTimes[i] == layerTimes[i-1]) {
        duplicate_count++;
      }
    }

    if (duplicate_count > 0) {
      fprintf(screen, "WARNING: Found %zu duplicate consecutive values\n", duplicate_count);
    }

    // Check for zeros in middle of array
    size_t internal_zeros = 0;
    for (size_t i = 0; i < layerTimes.size(); i++) {
      if (layerTimes[i] == 0.0) {
        internal_zeros++;
      }
    }

    if (internal_zeros > 0) {
      fprintf(screen, "WARNING: Found %zu zero values in layerTimes\n", internal_zeros);
    }
  }

  // Get subdomain bounds from SPPARKS domain
  // Note: We don't pre-allocate a grid. Instead, we'll compute
  // tetrahedral interpolation on-demand for each site query.
  // This is much more memory efficient for large domains.

  source_initialized = true;
  
  if (universe->me == 0) {
    print_source_info();
  }
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_temperature_at_xyz_and_time(
  double x, double y, double z, double time)
{
  if (!source_initialized) {
    error->all(FLERR, "Temperature source not initialized");
  }
  
  // Check if we need to load new layer data
  constexpr double tol = 5.0 * std::numeric_limits<double>::epsilon();
  if (std::fabs(time - current_time) / (std::fabs(time) + tol) > tol) {
    auto currentLayer = get_active_layer(time);
    if (currentLayer != active_layer) {
      load_layer(currentLayer);
      active_layer = currentLayer;
    }
    current_time = time;
  }
  
  // Check if we have valid data loaded
  if (selected_chunks.empty() || elem_bboxes.empty()) {
    return default_temp;
  }
  
  // Find the element containing this point and get interpolation weights
  std::array<double, 3> pt{x, y, z};
  auto result = find_element_point_is_in(selected_chunks, chunk_bboxes, 
                                        node_offsets, elem_offsets, 
                                        elemNode, nodeCoords, 
                                        elem_bboxes, pt);
  
  const auto& nodeIndices = result.first;
  const auto& weights = result.second;
  
  // Check if point was found in any element
  if (nodeIndices[0] == std::numeric_limits<unsigned>::max()) {
    return default_temp;
  }
  
  // Get temperatures at the four tetrahedral nodes
  std::array<double, NODES_PER_ELEM> nodalVals;
  for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
    const unsigned nodeIdx = nodeIndices[n];
    
    const auto timeIter = times.row_iterator(nodeIdx);
    const auto tempIter = temperatures.row_iterator(nodeIdx);
    
    // Check time bounds
    if (time < *timeIter || time > timeIter[dataCounts[nodeIdx] - 1]) {
      return default_temp;
    }
    
    // Linear interpolation in time
    auto lower = std::lower_bound(timeIter, timeIter + dataCounts[nodeIdx], time);
    auto idx = std::distance(timeIter, lower);
    idx = (idx == 0) ? 1 : idx;
    
    nodalVals[n] = tempIter[idx-1] + 
                   (tempIter[idx] - tempIter[idx-1]) / 
                   (timeIter[idx] - timeIter[idx-1]) * 
                   (time - timeIter[idx-1]);
  }
  
  // Tetrahedral interpolation using barycentric coordinates
  return nodalVals[0] + 
         weights[0] * (nodalVals[1] - nodalVals[0]) + 
         weights[1] * (nodalVals[2] - nodalVals[0]) + 
         weights[2] * (nodalVals[3] - nodalVals[0]);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::update_temperatures(double dt, double simulation_time)
{
  // This temperature source is passive - no update needed
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::needs_data_refresh(double simulation_time)
{
  if (!source_initialized) return false;
  
  auto layer = get_active_layer(simulation_time);
  return (layer != active_layer);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::cleanup()
{
  file.reset();
  source_initialized = false;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::print_source_info() const
{
  if (universe->me == 0) {
    fprintf(screen, "HDF5 Unstructured Temperature Source:\n");
    fprintf(screen, "  File: %s\n", filename.c_str());
    fprintf(screen, "  Grid spacing: %g m\n", dx);
    fprintf(screen, "  Time range: %g to %g s\n", layerTimes.front(), layerTimes.back());
    fprintf(screen, "  Number of layers: %zu\n", layerTimes.size());
    fprintf(screen, "  Threshold temperature: %g K\n", threshold_temp);
    fprintf(screen, "  Default temperature: %g K\n", default_temp);
    fprintf(screen, "  Domain-aware chunk loading: ENABLED\n");
    fprintf(screen, "  SPPARKS physical domain: [%g, %g, %g] to [%g, %g, %g] m\n",
            domain->boxxlo * dx, domain->boxylo * dx, domain->boxzlo * dx,
            domain->boxxhi * dx, domain->boxyhi * dx, domain->boxzhi * dx);
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::load_layer(unsigned layerIdx)
{
  auto load_start = std::chrono::high_resolution_clock::now();
  size_t mem_start = get_memory_usage_kb();

  if (universe->me == 0) {
    fprintf(screen, "Loading layer %u at time %.6e s\n", layerIdx,
            (layerIdx < layerTimes.size()) ? layerTimes[layerIdx] : -1.0);
    fprintf(screen, "  [MEM] Start: %zu MB (rank 0)\n", mem_start / 1024);
  }

  // Try to open the layer group with error handling
  std::shared_ptr<HighFive::Group> grp;
  try {
    std::string group_name = std::to_string(layerIdx);
    grp = std::make_shared<HighFive::Group>(file->getGroup(group_name));
  } catch (const HighFive::Exception& e) {
    char error_msg[512];
    snprintf(error_msg, sizeof(error_msg),
             "Failed to open HDF5 group '%u': %s\n"
             "This usually means the layer group does not exist in the HDF5 file.\n"
             "Check that your HDF5 file has groups named '0', '1', '2', ... '%u'",
             layerIdx, e.what(), layerIdx);
    error->all(FLERR, error_msg);
  } catch (const std::exception& e) {
    char error_msg[512];
    snprintf(error_msg, sizeof(error_msg),
             "Unexpected error opening HDF5 group '%u': %s", layerIdx, e.what());
    error->all(FLERR, error_msg);
  }
  
  // Read bounding boxes
  std::vector<std::vector<double>> bboxes;
  grp->getDataSet("boundingBoxes").read(bboxes);

  // Read element and node pointers
  std::vector<unsigned> elemPtr;
  grp->getDataSet("elemPtrs").read(elemPtr);

  std::vector<unsigned> nodePtr;
  grp->getDataSet("nodePtrs").read(nodePtr);
  
  // Convert SPPARKS per-rank subdomain bounds to physical coordinates (meters).
  // Use subdomain (subxlo/hi) instead of global domain (boxxlo/hi) so each rank
  // only loads temperature chunks overlapping its own partition.
  // Pad by one spatial grid cell to capture thermal elements straddling the
  // subdomain boundary (needed for interpolation at boundary sites).
  double pad = dx * grid_cell_size_multiplier;
  std::vector<double> spparksPhysicalBbox{
    domain->subxlo * dx - pad, domain->subylo * dx - pad, domain->subzlo * dx - pad,
    domain->subxhi * dx + pad, domain->subyhi * dx + pad, domain->subzhi * dx + pad
  };
  
  // Find overlapping chunks - first pass for chunks that directly overlap SPPARKS domain
  std::vector<unsigned> overlappingChunks;
  std::vector<std::array<size_t, 2>> elemSlices;
  std::vector<std::array<size_t, 2>> nodeSlices;
  std::vector<unsigned> nodeOffsets{0};
  std::vector<unsigned> elemOffsets{0};
  std::set<unsigned> chunksToLoad;  // Use set to avoid duplicates
  
  // First pass: find chunks that overlap with SPPARKS physical domain
  for (unsigned c = 0; c < bboxes.size(); c++) {
    if (do_boxes_overlap(bboxes[c], spparksPhysicalBbox)) {
      chunksToLoad.insert(c);
    }
  }
  
  if (chunksToLoad.empty()) {
    // No data for this domain 
    return;
  }
  
  // Second pass: ensure element completeness
  // For elements in loaded chunks that might contain SPPARKS sites,
  // we need to ensure all their nodes are available
  
  // To implement element completeness, we need to:
  // 1. Load elementToNode connectivity for the initial chunks
  // 2. Check which elements have nodes in other chunks
  // 3. Add those chunks to our load set
  
  // For true element completeness, we would need to:
  // - Pre-load element connectivity from initial chunks
  // - Identify elements whose nodes span into other chunks
  // - Add those chunks and iterate until no new chunks are needed
  // This would require restructuring the data loading flow
  
  // Current implementation loads chunks that overlap SPPARKS domain
  // which should capture most relevant elements for typical mesh sizes
  
  // Convert set to vector for ordered access
  overlappingChunks.assign(chunksToLoad.begin(), chunksToLoad.end());
  std::sort(overlappingChunks.begin(), overlappingChunks.end());
  
  // Build slices and offsets for the chunks we're loading
  for (unsigned c : overlappingChunks) {
    elemSlices.push_back({elemPtr[c], elemPtr[c+1]});
    nodeSlices.push_back({nodePtr[c], nodePtr[c+1]});
    nodeOffsets.push_back(nodeOffsets.back() + nodePtr[c+1] - nodePtr[c]);
    elemOffsets.push_back(elemOffsets.back() + elemPtr[c+1] - elemPtr[c]);
  }
  
  // Store chunk info for later use in point queries
  // Convert from HDF5 vector<vector<double>> to efficient array storage
  chunk_bboxes.clear();
  chunk_bboxes.reserve(bboxes.size());
  for (const auto& bb : bboxes) {
    chunk_bboxes.push_back({bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]});
  }
  selected_chunks = overlappingChunks;
  
  if (overlappingChunks.size() > 0) {
    unsigned totalElements = elemPtr.back();
    unsigned loadedElements = elemOffsets.back();
    unsigned totalNodes = nodePtr.back();
    unsigned loadedNodes = nodeOffsets.back();

    fprintf(screen, "  Rank %d: Loaded %zu/%zu chunks (%.1f%%), %u/%u elements (%.1f%%), %u/%u nodes (%.1f%%)\n",
            universe->me,
            overlappingChunks.size(), bboxes.size(), 100.0 * overlappingChunks.size() / bboxes.size(),
            loadedElements, totalElements, 100.0 * loadedElements / totalElements,
            loadedNodes, totalNodes, 100.0 * loadedNodes / totalNodes);
  }
  node_offsets = nodeOffsets;
  elem_offsets = elemOffsets;

  // Read element connectivity
  std::vector<unsigned> elemNodeData;
  grp->getDataSet("elementToNode").select(
    build_hyperslab(elemSlices, {0, NODES_PER_ELEM})).read(elemNodeData);
  elemNode = Array2D<unsigned>(NODES_PER_ELEM, std::move(elemNodeData));

  if (universe->me == 0) {
    fprintf(screen, "  [MEM] After elemNode (%u elems): %zu MB\n",
            elemOffsets.back(), get_memory_usage_kb() / 1024);
  }

  // Read node coordinates
  std::vector<double> nodeCoordsData;
  grp->getDataSet("nodeCoords").select(
    build_hyperslab(nodeSlices, {0, DIM})).read(nodeCoordsData);
  nodeCoords = Array2D<double>(DIM, std::move(nodeCoordsData));

  if (universe->me == 0) {
    fprintf(screen, "  [MEM] After nodeCoords (%u nodes): %zu MB\n",
            nodeOffsets.back(), get_memory_usage_kb() / 1024);
  }

  // Read data counts
  grp->getDataSet("dataCounts").select(
    build_hyperslab_1d(nodeSlices)).read(dataCounts);

  if (universe->me == 0) {
    fprintf(screen, "  [MEM] After dataCounts: %zu MB\n", get_memory_usage_kb() / 1024);
  }

  // Read time and temperature data (virtual — overridden by CSR subclass)
  read_time_temperature_data(*grp, nodeSlices);

  if (universe->me == 0) {
    size_t times_mem = times.total_elements() * sizeof(double) / 1024 / 1024;
    size_t temps_mem = temperatures.total_elements() * sizeof(double) / 1024 / 1024;
    fprintf(screen, "  [MEM] After CSR data (times: %zu MB, temps: %zu MB): %zu MB total\n",
            times_mem, temps_mem, get_memory_usage_kb() / 1024);
  }

  // Build element bounding boxes for spatial queries
  elem_bboxes = build_elem_bounding_boxes(overlappingChunks.size(),
                                         nodeOffsets, elemOffsets,
                                         elemNode, nodeCoords);

  if (universe->me == 0) {
    size_t bbox_mem = elem_bboxes.size() * sizeof(std::array<double, 6>) / 1024 / 1024;
    fprintf(screen, "  [MEM] After elem_bboxes (%zu elems, %zu MB array): %zu MB total\n",
            elem_bboxes.size(), bbox_mem, get_memory_usage_kb() / 1024);
  }

  // Build spatial acceleration grid (optional, controlled by use_spatial_grid flag)
  // Larger cells = less memory but more elements per cell to search
  // Default 50x lattice spacing (~250 microns), user-configurable via grid_cell_mult parameter
  if (use_spatial_grid) {
    build_spatial_grid(dx * grid_cell_size_multiplier);

    if (universe->me == 0) {
      size_t grid_cells = spatial_grid.dims[0] * spatial_grid.dims[1] * spatial_grid.dims[2];
      fprintf(screen, "  [MEM] After spatial_grid (%zu cells): %zu MB total\n",
              grid_cells, get_memory_usage_kb() / 1024);
    }
  } else {
    spatial_grid.valid = false;
    if (universe->me == 0) {
      fprintf(screen, "  Spatial grid: DISABLED\n");
    }
  }

  // Compute thermal intervals for efficient time queries
  compute_thermal_intervals_for_layer(layerIdx, threshold_temp);

  // Record timing for performance monitoring
  auto load_end = std::chrono::high_resolution_clock::now();
  double load_time = std::chrono::duration<double>(load_end - load_start).count();
  total_layer_load_time += load_time;
  layer_load_count++;

  if (universe->me == 0) {
    size_t mem_end = get_memory_usage_kb();
    fprintf(screen, "  [MEM] Final: %zu MB (delta: %+zd MB)\n",
            mem_end / 1024, (long)(mem_end - mem_start) / 1024);
    fprintf(screen, "  Layer load time: %.3f s (avg: %.3f s over %d loads)\n",
            load_time, total_layer_load_time / layer_load_count, layer_load_count);
  }
}

/* ---------------------------------------------------------------------- */

HighFive::HyperSlab HDF5UnstructuredTemperatureSource::build_hyperslab(
  const std::vector<std::array<size_t, 2>>& slices,
  const std::array<size_t, 2>& cols)
{
  if (slices.empty()) throw std::runtime_error("No slices for hyperslab");

  HighFive::HyperSlab result(HighFive::RegularHyperSlab(
    {slices[0][0], cols[0]},
    {slices[0][1] - slices[0][0], cols[1] - cols[0]}));

  for (size_t r = 1; r < slices.size(); r++) {
    result |= HighFive::RegularHyperSlab(
      {slices[r][0], cols[0]},
      {slices[r][1] - slices[r][0], cols[1] - cols[0]});
  }
  return result;
}

/* ---------------------------------------------------------------------- */

HighFive::HyperSlab HDF5UnstructuredTemperatureSource::build_hyperslab_1d(
  const std::vector<std::array<size_t, 2>>& slices)
{
  if (slices.empty()) throw std::runtime_error("No slices for hyperslab");

  HighFive::HyperSlab result(HighFive::RegularHyperSlab(
    {slices[0][0]}, {slices[0][1] - slices[0][0]}));

  for (size_t r = 1; r < slices.size(); r++) {
    result |= HighFive::RegularHyperSlab(
      {slices[r][0]}, {slices[r][1] - slices[r][0]});
  }
  return result;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_time_temperature_data(
  HighFive::Group& grp,
  const std::vector<std::array<size_t, 2>>& nodeSlices)
{
  // Streaming implementation: read one chunk at a time from the padded 2D
  // datasets, stripping padding on the fly into pre-allocated CSR arrays.
  // Peak temporary buffer = max_chunk_rows × readCols × 8 bytes.

  auto t_start = std::chrono::high_resolution_clock::now();

  // Determine how many columns to read per row
  auto timesDset = grp.getDataSet("times");
  auto timesDims = timesDset.getDimensions();
  unsigned datasetCols = static_cast<unsigned>(timesDims[1]);

  unsigned maxData = 0;
  for (auto cnt : dataCounts)
    maxData = std::max(maxData, cnt);

  unsigned readCols = std::min(datasetCols, maxData);

  // Build CSR offsets from dataCounts
  size_t nrows = dataCounts.size();
  std::vector<size_t> offsets(nrows + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < nrows; i++)
    offsets[i + 1] = offsets[i] + dataCounts[i];

  size_t totalElements = offsets[nrows];

  // Pre-allocate flat output arrays to exact CSR size
  std::vector<double> timesFlat(totalElements);
  std::vector<double> tempsFlat(totalElements);

  if (universe->me == 0) {
    size_t rectMB = nrows * static_cast<size_t>(readCols) * sizeof(double) / (1024 * 1024);
    fprintf(screen, "  Streaming 2D read: %zu nodes, %u readCols, %zu total elements\n",
            nrows, readCols, totalElements);
    fprintf(screen, "  [MEM] Rectangular would need: %zu MB per dataset (%zu MB total)\n",
            rectMB, rectMB * 2);
  }

  // Reusable chunk buffer — grows to largest chunk, then stays put
  std::vector<double> buf;
  auto tempsDset = grp.getDataSet("temperatures");
  size_t dcPos = 0;  // read cursor into dataCounts / offsets

  for (size_t s = 0; s < nodeSlices.size(); s++) {
    size_t rowStart  = nodeSlices[s][0];
    size_t rowEnd    = nodeSlices[s][1];
    size_t chunkRows = rowEnd - rowStart;
    if (chunkRows == 0) continue;

    buf.resize(chunkRows * static_cast<size_t>(readCols));

    // --- Read & strip times ---
    timesDset.select(HighFive::HyperSlab(HighFive::RegularHyperSlab(
        {rowStart, 0}, {chunkRows, static_cast<size_t>(readCols)})))
      .read(buf);

    for (size_t r = 0; r < chunkRows; r++) {
      unsigned count = dataCounts[dcPos + r];
      std::copy_n(buf.data() + r * readCols, count,
                  timesFlat.data() + offsets[dcPos + r]);
    }

    // --- Read & strip temperatures (reuse buf) ---
    tempsDset.select(HighFive::HyperSlab(HighFive::RegularHyperSlab(
        {rowStart, 0}, {chunkRows, static_cast<size_t>(readCols)})))
      .read(buf);

    for (size_t r = 0; r < chunkRows; r++) {
      unsigned count = dataCounts[dcPos + r];
      std::copy_n(buf.data() + r * readCols, count,
                  tempsFlat.data() + offsets[dcPos + r]);
    }

    if (universe->me == 0) {
      fprintf(screen, "    Chunk %zu/%zu: %zu rows, buf %.1f MB, [MEM] %zu MB\n",
              s + 1, nodeSlices.size(), chunkRows,
              chunkRows * static_cast<size_t>(readCols) * sizeof(double) / (1024.0 * 1024.0),
              get_memory_usage_kb() / 1024);
    }

    dcPos += chunkRows;
  }

  // Build CompressedArray2D via zero-copy move
  auto offsets2 = offsets;  // copy before first move
  times = CompressedArray2D<double>(std::move(timesFlat), std::move(offsets));
  temperatures = CompressedArray2D<double>(std::move(tempsFlat), std::move(offsets2));

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  if (universe->me == 0) {
    fprintf(screen, "  [TIMING] Streaming 2D read: %.3f s (%zu elements)\n",
            elapsed, totalElements);
  }
}

/* ---------------------------------------------------------------------- */

std::array<double, 3> HDF5UnstructuredTemperatureSource::get_parametric_coordinates_of_point(
  const std::vector<std::vector<double>>& tet_coords, 
  const std::array<double, 3>& pt) const
{
  std::array<double, 3> relativeCoords, relativeCoords1, relativeCoords2, relativeCoords3;
  
  for (unsigned d = 0; d < pt.size(); d++) {
    relativeCoords1[d] = tet_coords[1][d] - tet_coords[0][d];
    relativeCoords2[d] = tet_coords[2][d] - tet_coords[0][d];
    relativeCoords3[d] = tet_coords[3][d] - tet_coords[0][d];
    relativeCoords[d] = pt[d] - tet_coords[0][d];
  }
  
  const double a00 = relativeCoords1[0];
  const double a01 = relativeCoords2[0];
  const double a02 = relativeCoords3[0];
  const double a10 = relativeCoords1[1];
  const double a11 = relativeCoords2[1];
  const double a12 = relativeCoords3[1];
  const double a20 = relativeCoords1[2];
  const double a21 = relativeCoords2[2];
  const double a22 = relativeCoords3[2];
  const double b0 = relativeCoords[0];
  const double b1 = relativeCoords[1];
  const double b2 = relativeCoords[2];
  
  const double invDet = 1.0 / (a00*(a22*a11-a21*a12) - a10*(a22*a01-a21*a02) + a20*(a12*a01-a11*a02));
  const double x = ( b0*(a22*a11-a21*a12) - b1*(a22*a01-a21*a02) + b2*(a12*a01-a11*a02)) * invDet;
  const double y = (-b0*(a22*a10-a20*a12) + b1*(a22*a00-a20*a02) - b2*(a12*a00-a10*a02)) * invDet;
  const double z = ( b0*(a21*a10-a20*a11) - b1*(a21*a00-a20*a01) + b2*(a11*a00-a10*a01)) * invDet;
  
  return {x, y, z};
}

/* ---------------------------------------------------------------------- */

unsigned HDF5UnstructuredTemperatureSource::get_active_layer(double t) const
{
  // Check time bounds with detailed error messages
  if (t < layerTimes.front()) {
    char error_msg[512];
    snprintf(error_msg, sizeof(error_msg),
             "Simulation time %.6e is before first layer time %.6e\n"
             "Time is %.6e seconds too early",
             t, layerTimes.front(), layerTimes.front() - t);
    if (universe->me == 0) {
      fprintf(screen, "ERROR: %s\n", error_msg);
    }
    error->all(FLERR, error_msg);
  }

  if (t > layerTimes.back()) {
    char error_msg[512];
    snprintf(error_msg, sizeof(error_msg),
             "Simulation time %.6e is after last layer time %.6e\n"
             "Time is %.6e seconds too late\n"
             "Number of layers in file: %zu",
             t, layerTimes.back(), t - layerTimes.back(), layerTimes.size());
    if (universe->me == 0) {
      fprintf(screen, "ERROR: %s\n", error_msg);
    }
    error->all(FLERR, error_msg);
  }

  // Binary search for the active layer
  auto lower = std::lower_bound(layerTimes.begin(), layerTimes.end(), t);
  auto idx = std::distance(layerTimes.begin(), lower);
  unsigned layer_idx = (idx == 0) ? 0 : idx - 1;

  return layer_idx;
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::do_boxes_overlap(
  const std::vector<double>& b1, 
  const std::vector<double>& b2) const
{
  return b1[0] < b2[3] && b2[0] < b1[3] && 
         b1[1] < b2[4] && b2[1] < b1[4] && 
         b1[2] < b2[5] && b2[2] < b1[5];
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::point_in_bbox(
  const std::array<double, 3>& pt,
  const std::array<double, 6>& bbox) const
{
  return pt[0] >= bbox[0] && pt[0] <= bbox[3] &&
         pt[1] >= bbox[1] && pt[1] <= bbox[4] &&
         pt[2] >= bbox[2] && pt[2] <= bbox[5];
}

/* ---------------------------------------------------------------------- */

std::pair<std::array<unsigned, 4>, std::array<double, 3>>
HDF5UnstructuredTemperatureSource::find_element_point_is_in(
  const std::vector<unsigned>& selectedChunks,
  const std::vector<std::array<double, 6>>& chunkBboxes,
  const std::vector<unsigned>& nodeOffsets,
  const std::vector<unsigned>& elemOffsets,
  const Array2D<unsigned>& elemNode,
  const Array2D<double>& nodeCoords,
  const std::vector<std::array<double, 6>>& elemBboxes,
  const std::array<double, 3>& pt) const
{
  std::array<unsigned, 4> nodeIds;
  std::array<double, 3> parCoords;

  constexpr double tol = 1e-14;
  constexpr unsigned invalidId = std::numeric_limits<unsigned>::max();

  // Validate input data
  if (selectedChunks.empty() || chunkBboxes.empty() ||
      nodeOffsets.empty() || elemOffsets.empty() || elemBboxes.empty()) {
    return std::make_pair(
      std::array<unsigned, 4>{invalidId, invalidId, invalidId, invalidId},
      std::array<double, 3>{0.0, 0.0, 0.0}
    );
  }

  // Reusable storage for tetrahedron coordinates
  std::vector<std::vector<double>> tetCoords(NODES_PER_ELEM, std::vector<double>(DIM));

  // Lambda to test a single element and return true if point is inside
  auto test_element = [&](unsigned e) -> bool {
    if (e >= elemBboxes.size()) return false;
    if (!point_in_bbox(pt, elemBboxes[e])) return false;

    // Find which chunk this element belongs to (for node offset mapping)
    // Use O(1) precomputed mapping when spatial grid is valid, else O(n) search
    unsigned c;
    if (spatial_grid.valid && e < spatial_grid.elem_to_chunk.size()) {
      c = spatial_grid.elem_to_chunk[e];
    } else {
      c = 0;
      for (unsigned i = 0; i < elemOffsets.size() - 1; i++) {
        if (e >= elemOffsets[i] && e < elemOffsets[i + 1]) {
          c = i;
          break;
        }
      }
    }

    // Get tetrahedron node coordinates
    bool validElement = true;
    for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
      unsigned localNodeId = elemNode(e, n);
      if (c >= nodeOffsets.size() - 1) {
        validElement = false;
        break;
      }
      nodeIds[n] = localNodeId + nodeOffsets[c];
      for (unsigned d = 0; d < DIM; d++) {
        tetCoords[n][d] = nodeCoords(nodeIds[n], d);
      }
    }
    if (!validElement) return false;

    parCoords = get_parametric_coordinates_of_point(tetCoords, pt);

    // Check if point is inside tetrahedron
    return (parCoords[0] > -tol && parCoords[1] > -tol && parCoords[2] > -tol &&
            1.0 - parCoords[0] - parCoords[1] - parCoords[2] > -tol);
  };

  // Use spatial grid if available (O(1) cell lookup vs O(n) brute force)
  if (spatial_grid.valid) {
    int cell_idx = spatial_grid.point_to_cell(pt);
    if (cell_idx >= 0 && static_cast<size_t>(cell_idx) < spatial_grid.cell_offsets.size() - 1) {
      // Iterate over elements in this cell using CSR-style access
      const unsigned* elem_begin = spatial_grid.cell_begin(cell_idx);
      const unsigned* elem_end = spatial_grid.cell_end(cell_idx);
      for (const unsigned* ep = elem_begin; ep != elem_end; ++ep) {
        if (test_element(*ep)) {
          return std::make_pair(nodeIds, parCoords);
        }
      }
    }
    // Point not in grid or not found in cell - return not found
    return std::make_pair(
      std::array<unsigned, 4>{invalidId, invalidId, invalidId, invalidId},
      std::array<double, 3>{0.0, 0.0, 0.0}
    );
  }

  // Fallback: brute force search through chunks (used if grid not built)
  std::vector<unsigned> possibleChunks;
  for (unsigned c = 0; c < selectedChunks.size(); c++) {
    if (selectedChunks[c] >= chunkBboxes.size()) continue;
    const auto& cBbox = chunkBboxes[selectedChunks[c]];
    if (point_in_bbox(pt, cBbox)) {
      possibleChunks.push_back(c);
    }
  }

  for (auto c : possibleChunks) {
    if (c + 1 >= elemOffsets.size()) continue;
    for (unsigned e = elemOffsets[c]; e < elemOffsets[c + 1]; e++) {
      if (test_element(e)) {
        return std::make_pair(nodeIds, parCoords);
      }
    }
  }

  // Point not found in any element
  return std::make_pair(
    std::array<unsigned, 4>{invalidId, invalidId, invalidId, invalidId},
    std::array<double, 3>{0.0, 0.0, 0.0}
  );
}

/* ---------------------------------------------------------------------- */

std::vector<std::array<double, 6>> HDF5UnstructuredTemperatureSource::build_elem_bounding_boxes(
  unsigned nChunks,
  const std::vector<unsigned>& nodeOffsets,
  const std::vector<unsigned>& elemOffsets,
  const Array2D<unsigned>& elemNode,
  const Array2D<double>& nodeCoords) const
{
  constexpr double maxVal = std::numeric_limits<double>::max();
  constexpr double minVal = std::numeric_limits<double>::lowest();

  // Pre-calculate total element count and reserve to avoid reallocations
  size_t totalElems = 0;
  for (unsigned c = 0; c < nChunks; c++) {
    totalElems += elemOffsets[c+1] - elemOffsets[c];
  }

  std::vector<std::array<double, 6>> result;
  result.reserve(totalElems);

  for (unsigned c = 0; c < nChunks; c++) {
    for (unsigned e = elemOffsets[c]; e < elemOffsets[c+1]; e++) {
      std::array<double, 3> bmin{maxVal, maxVal, maxVal};
      std::array<double, 3> bmax{minVal, minVal, minVal};

      for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
        for (unsigned d = 0; d < DIM; d++) {
          const double coordVal = nodeCoords(elemNode(e, n) + nodeOffsets[c], d);
          bmin[d] = std::fmin(bmin[d], coordVal);
          bmax[d] = std::fmax(bmax[d], coordVal);
        }
      }

      result.push_back({bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]});
    }
  }

  return result;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_spatial_grid(double target_cell_size)
{
  spatial_grid.valid = false;

  if (elem_bboxes.empty()) return;

  // Find overall bounds from element bounding boxes
  constexpr double maxVal = std::numeric_limits<double>::max();
  constexpr double minVal = std::numeric_limits<double>::lowest();
  std::array<double, 3> domain_min{maxVal, maxVal, maxVal};
  std::array<double, 3> domain_max{minVal, minVal, minVal};

  for (const auto& bbox : elem_bboxes) {
    for (unsigned d = 0; d < DIM; d++) {
      domain_min[d] = std::fmin(domain_min[d], bbox[d]);
      domain_max[d] = std::fmax(domain_max[d], bbox[d + 3]);
    }
  }

  // Add small padding to avoid edge cases
  for (unsigned d = 0; d < DIM; d++) {
    double pad = target_cell_size * 0.01;
    domain_min[d] -= pad;
    domain_max[d] += pad;
  }

  // Compute grid dimensions
  spatial_grid.origin = domain_min;
  for (unsigned d = 0; d < DIM; d++) {
    double extent = domain_max[d] - domain_min[d];
    spatial_grid.dims[d] = std::max(1u, static_cast<unsigned>(std::ceil(extent / target_cell_size)));
    spatial_grid.cell_size[d] = extent / spatial_grid.dims[d];
  }

  size_t total_cells = static_cast<size_t>(spatial_grid.dims[0]) *
                       spatial_grid.dims[1] * spatial_grid.dims[2];

  // Build element-to-chunk mapping for O(1) chunk lookup
  spatial_grid.elem_to_chunk.resize(elem_bboxes.size());
  for (unsigned c = 0; c < elem_offsets.size() - 1; c++) {
    for (unsigned e = elem_offsets[c]; e < elem_offsets[c + 1]; e++) {
      spatial_grid.elem_to_chunk[e] = c;
    }
  }

  // Lambda to compute cell range for an element's bounding box
  auto get_cell_range = [&](const std::array<double, 6>& bbox,
                            int& ix_min, int& iy_min, int& iz_min,
                            int& ix_max, int& iy_max, int& iz_max) {
    ix_min = static_cast<int>((bbox[0] - spatial_grid.origin[0]) / spatial_grid.cell_size[0]);
    iy_min = static_cast<int>((bbox[1] - spatial_grid.origin[1]) / spatial_grid.cell_size[1]);
    iz_min = static_cast<int>((bbox[2] - spatial_grid.origin[2]) / spatial_grid.cell_size[2]);
    ix_max = static_cast<int>((bbox[3] - spatial_grid.origin[0]) / spatial_grid.cell_size[0]);
    iy_max = static_cast<int>((bbox[4] - spatial_grid.origin[1]) / spatial_grid.cell_size[1]);
    iz_max = static_cast<int>((bbox[5] - spatial_grid.origin[2]) / spatial_grid.cell_size[2]);
    ix_min = std::max(0, ix_min);
    iy_min = std::max(0, iy_min);
    iz_min = std::max(0, iz_min);
    ix_max = std::min(static_cast<int>(spatial_grid.dims[0]) - 1, ix_max);
    iy_max = std::min(static_cast<int>(spatial_grid.dims[1]) - 1, iy_max);
    iz_max = std::min(static_cast<int>(spatial_grid.dims[2]) - 1, iz_max);
  };

  // === CSR-style flat storage: two-pass approach ===

  // Pass 1: Count elements per cell
  std::vector<size_t> cell_counts(total_cells, 0);
  for (unsigned e = 0; e < elem_bboxes.size(); e++) {
    int ix_min, iy_min, iz_min, ix_max, iy_max, iz_max;
    get_cell_range(elem_bboxes[e], ix_min, iy_min, iz_min, ix_max, iy_max, iz_max);

    for (int iz = iz_min; iz <= iz_max; iz++) {
      for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
          size_t cell_idx = ix + iy * spatial_grid.dims[0] +
                            iz * spatial_grid.dims[0] * spatial_grid.dims[1];
          cell_counts[cell_idx]++;
        }
      }
    }
  }

  // Build prefix sum to get offsets
  spatial_grid.cell_offsets.resize(total_cells + 1);
  spatial_grid.cell_offsets[0] = 0;
  for (size_t i = 0; i < total_cells; i++) {
    spatial_grid.cell_offsets[i + 1] = spatial_grid.cell_offsets[i] + cell_counts[i];
  }

  // Allocate flat element array
  size_t total_refs = spatial_grid.cell_offsets[total_cells];
  spatial_grid.cell_elements.resize(total_refs);

  // Pass 2: Fill the flat array (reuse cell_counts as write positions)
  std::fill(cell_counts.begin(), cell_counts.end(), 0);
  for (unsigned e = 0; e < elem_bboxes.size(); e++) {
    int ix_min, iy_min, iz_min, ix_max, iy_max, iz_max;
    get_cell_range(elem_bboxes[e], ix_min, iy_min, iz_min, ix_max, iy_max, iz_max);

    for (int iz = iz_min; iz <= iz_max; iz++) {
      for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
          size_t cell_idx = ix + iy * spatial_grid.dims[0] +
                            iz * spatial_grid.dims[0] * spatial_grid.dims[1];
          size_t write_pos = spatial_grid.cell_offsets[cell_idx] + cell_counts[cell_idx];
          spatial_grid.cell_elements[write_pos] = e;
          cell_counts[cell_idx]++;
        }
      }
    }
  }

  spatial_grid.valid = true;

  if (universe->me == 0) {
    size_t max_per_cell = 0;
    for (size_t i = 0; i < total_cells; i++) {
      max_per_cell = std::max(max_per_cell, spatial_grid.cell_count(i));
    }
    // Memory: offsets array + elements array + elem_to_chunk
    size_t grid_mem_kb = (spatial_grid.cell_offsets.size() * sizeof(size_t) +
                          spatial_grid.cell_elements.size() * sizeof(unsigned) +
                          spatial_grid.elem_to_chunk.size() * sizeof(unsigned)) / 1024;
    fprintf(screen, "  Spatial grid: %u×%u×%u cells, %.1f elements/cell avg, %zu max, %zu KB\n",
            spatial_grid.dims[0], spatial_grid.dims[1], spatial_grid.dims[2],
            static_cast<double>(total_refs) / total_cells, max_per_cell, grid_mem_kb);
  }
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::all_temperatures_below_threshold(double time)
{
  if (!app) {
    error->all(FLERR, "App not available for temperature checking");
  }
  
  // Check temperatures at all local sites
  for (int i = 0; i < app->nlocal; i++) {
    // Get site coordinates in physical units
    double x = app->xyz[i][0] * dx;
    double y = app->xyz[i][1] * dx; 
    double z = app->xyz[i][2] * dx;
    double temp = get_temperature_at_xyz_and_time(x, y, z, time);
    if (temp >= threshold_temp) {
      return false;
    }
  }
  
  return true;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::prepare_for_timestep(double time)
{
  if (!source_initialized) {
    error->all(FLERR, "Temperature source not initialized");
  }

  // Check if we need to load new layer data
  constexpr double tol = 5.0 * std::numeric_limits<double>::epsilon();
  if (std::fabs(time - current_time) / (std::fabs(time) + tol) > tol) {
    auto currentLayer = get_active_layer(time);
    if (currentLayer != active_layer) {
      load_layer(currentLayer);
      active_layer = currentLayer;
      cache_valid = false;
      nodal_cache_valid = false;
    }
    current_time = time;
  }

  // Build element cache if needed
  if (!cache_valid) {
    build_site_element_cache();
  }

  // Precompute nodal temperatures for this timestep
  precompute_nodal_temperatures(time);
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_temperature_at_site(int site_index, double time)
{
  // Ensure caches are ready (for standalone calls)
  prepare_for_timestep(time);

  // Use fast path
  return get_temperature_at_site_fast(site_index);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_site_element_cache() const
{
  auto start_time = std::chrono::high_resolution_clock::now();
  size_t mem_before = get_memory_usage_kb();

  // Get total number of sites from app
  int nlocal = app->nlocal;

  if (universe->me == 0) {
    size_t cache_size_mb = nlocal * sizeof(ElementCache) / 1024 / 1024;
    fprintf(screen, "  [MEM] Building site cache for %d sites (%zu MB), current: %zu MB\n",
            nlocal, cache_size_mb, mem_before / 1024);
  }

  // Resize cache to match number of local sites
  site_element_cache.resize(nlocal);

  // For each site, find its containing element and cache the result
  for (int i = 0; i < nlocal; i++) {
    // Get site coordinates 
    double x = app->xyz[i][0] * dx;  // Convert lattice to physical
    double y = app->xyz[i][1] * dx;
    double z = app->xyz[i][2] * dx;
    
    std::array<double, 3> pt{x, y, z};
    
    // Find element containing this point
    auto result = find_element_point_is_in(selected_chunks, chunk_bboxes, 
                                          node_offsets, elem_offsets, 
                                          elemNode, nodeCoords, 
                                          elem_bboxes, pt);
    
    // Store in cache
    site_element_cache[i].nodeIndices = result.first;
    site_element_cache[i].weights = result.second;
    site_element_cache[i].valid = (result.first[0] != std::numeric_limits<unsigned>::max());
  }

  // Collect unique node indices from valid cached elements
  // This allows precompute_nodal_temperatures to skip unused nodes
  std::set<unsigned> unique_nodes;
  for (int i = 0; i < nlocal; i++) {
    if (site_element_cache[i].valid) {
      for (int j = 0; j < 4; j++) {
        unique_nodes.insert(site_element_cache[i].nodeIndices[j]);
      }
    }
  }
  active_node_indices.assign(unique_nodes.begin(), unique_nodes.end());

  cache_valid = true;

  auto end_time = std::chrono::high_resolution_clock::now();
  double cache_build_time = std::chrono::duration<double>(end_time - start_time).count();

  if (universe->me == 0) {
    size_t mem_after = get_memory_usage_kb();
    fprintf(screen, "  Built element cache for %d sites in %.3f s\n", nlocal, cache_build_time);
    fprintf(screen, "  Active nodes: %zu / %zu (%.1f%% of loaded nodes)\n",
            active_node_indices.size(), dataCounts.size(),
            100.0 * active_node_indices.size() / dataCounts.size());
    fprintf(screen, "  [MEM] After site cache: %zu MB (delta: %+zd MB)\n",
            mem_after / 1024, (long)(mem_after - mem_before) / 1024);
  }
}

/* ---------------------------------------------------------------------- */

const HDF5UnstructuredTemperatureSource::ElementCache&
HDF5UnstructuredTemperatureSource::get_cached_element(int site_index) const
{
  if (site_index < 0 || site_index >= static_cast<int>(site_element_cache.size())) {
    error->all(FLERR, "Invalid site index in get_cached_element");
  }
  return site_element_cache[site_index];
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::precompute_nodal_temperatures(double time) const
{
  // Skip if already computed for this time
  constexpr double tol = 1e-12;
  if (nodal_cache_valid && std::fabs(time - cached_nodal_time) < tol) {
    return;
  }

  // Performance diagnostics
  static int precompute_call_count = 0;
  static double t_total = 0.0;
  static size_t total_nodes_processed = 0, total_nodes_interpolated = 0;
  precompute_call_count++;
  auto t_start = std::chrono::high_resolution_clock::now();

  const size_t num_nodes = dataCounts.size();
  cached_nodal_temps.resize(num_nodes);
  constexpr unsigned LINEAR_SEARCH_THRESHOLD = 32;

  size_t nodes_interpolated = 0;

  // Only process nodes that are actually used by cached sites
  // This provides major speedup when only a fraction of loaded nodes are needed
  const std::vector<unsigned>& nodes_to_process =
      active_node_indices.empty() ? std::vector<unsigned>() : active_node_indices;

  // If no active nodes collected (cache not built), fall back to all nodes
  const size_t num_to_process = nodes_to_process.empty() ? num_nodes : nodes_to_process.size();

  for (size_t i = 0; i < num_to_process; i++) {
    const size_t nodeIdx = nodes_to_process.empty() ? i : nodes_to_process[i];
    const unsigned count = dataCounts[nodeIdx];
    const auto timeIter = times.row_iterator(nodeIdx);
    const auto tempIter = temperatures.row_iterator(nodeIdx);

    // Check time bounds - use default_temp if out of range
    if (time < *timeIter || time > timeIter[count - 1]) {
      cached_nodal_temps[nodeIdx] = default_temp;
      continue;
    }

    nodes_interpolated++;

    // Find time index - use linear search for small arrays, binary for large
    unsigned idx;
    if (count <= LINEAR_SEARCH_THRESHOLD) {
      idx = 1;
      while (idx < count && timeIter[idx] < time) {
        ++idx;
      }
    } else {
      auto lower = std::lower_bound(timeIter, timeIter + count, time);
      idx = std::distance(timeIter, lower);
      if (idx == 0) idx = 1;
    }

    // Linear interpolation in time
    cached_nodal_temps[nodeIdx] = tempIter[idx-1] +
                                  (tempIter[idx] - tempIter[idx-1]) /
                                  (timeIter[idx] - timeIter[idx-1]) *
                                  (time - timeIter[idx-1]);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();
  t_total += elapsed;
  total_nodes_processed += num_to_process;
  total_nodes_interpolated += nodes_interpolated;

  // Print diagnostics every 100 calls
  if (precompute_call_count % 100 == 0 && universe->me == 0) {
    fprintf(screen, "  [PRECOMPUTE] After %d calls: total=%.3f s (%.3f ms/call), "
            "active: %zu/%zu nodes (%.1f%%), interpolated: %zu (%.1f%%)\n",
            precompute_call_count, t_total, 1000.0*t_total/precompute_call_count,
            num_to_process, num_nodes, 100.0*num_to_process/num_nodes,
            total_nodes_interpolated/precompute_call_count,
            100.0*total_nodes_interpolated/total_nodes_processed);
  }

  cached_nodal_time = time;
  nodal_cache_valid = true;
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_next_time_with_temperature(double current_time, double threshold_temp)
{
  if (layerTimes.empty()) {
    return current_time;
  }

  // Find current layer
  unsigned current_layer = get_active_layer(current_time);

  // Check if we have thermal intervals computed for this layer
  if (current_layer >= layer_thermal_intervals.size()) {
    return current_time;
  }

  // Check intervals in current layer
  for (const auto& interval : layer_thermal_intervals[current_layer]) {
    // If we're currently within this interval, return current_time to signal activity NOW
    if (current_time >= interval.start_time && current_time <= interval.end_time) {
      return current_time;
    }
    // If this interval starts after current time, return its start time
    if (interval.start_time > current_time) {
      return interval.start_time;
    }
  }

  // Check subsequent layers
  for (unsigned layer = current_layer + 1; layer < layerTimes.size(); layer++) {
    // If we haven't computed thermal intervals for this layer yet, we should advance to it
    // The layer loading will happen when we reach its time
    if (layer >= layer_thermal_intervals.size() || layer_thermal_intervals[layer].empty()) {
      return layerTimes[layer];
    }

    // If this layer has thermal intervals, return the first one
    if (!layer_thermal_intervals[layer].empty()) {
      return layer_thermal_intervals[layer][0].start_time;
    }
  }

  // No more thermal activity found
  return std::numeric_limits<double>::max();
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::compute_thermal_intervals_for_layer(unsigned layerIdx, double threshold_temp)
{
  // Ensure layer_thermal_intervals is large enough
  if (layerIdx >= layer_thermal_intervals.size()) {
    layer_thermal_intervals.resize(layerIdx + 1);
  }
  
  std::vector<ThermalInterval> intervals;
  
  // For each node in the loaded chunks
  for (size_t node_idx = 0; node_idx < dataCounts.size(); node_idx++) {
    // Check if node is within SPPARKS domain
    double x = nodeCoords(node_idx, 0);
    double y = nodeCoords(node_idx, 1); 
    double z = nodeCoords(node_idx, 2);
    
    if (!is_point_in_spparks_domain(x, y, z)) continue;
    
    // Scan temperature time series for this node
    double first_hot_time = -1.0;
    double last_hot_time = -1.0;
    
    for (unsigned t = 0; t < dataCounts[node_idx]; t++) {
      double temp = temperatures(node_idx, t);
      double time = times(node_idx, t);
      
      if (temp > threshold_temp) {
        if (first_hot_time < 0) first_hot_time = time;
        last_hot_time = time;
      }
    }
    
    // Add interval if we found thermal activity
    if (first_hot_time >= 0) {
      intervals.emplace_back(first_hot_time, last_hot_time);
    }
  }
  
  // Merge overlapping intervals and store
  layer_thermal_intervals[layerIdx] = merge_overlapping_intervals(intervals);
  
  if (universe->me == 0 && !layer_thermal_intervals[layerIdx].empty()) {
    fprintf(screen, "  Layer %u: Found %zu thermal intervals\n", 
            layerIdx, layer_thermal_intervals[layerIdx].size());
  }
}

/* ---------------------------------------------------------------------- */

std::vector<HDF5UnstructuredTemperatureSource::ThermalInterval> 
HDF5UnstructuredTemperatureSource::merge_overlapping_intervals(const std::vector<ThermalInterval>& intervals) const
{
  if (intervals.empty()) return {};
  
  auto sorted_intervals = intervals;
  std::sort(sorted_intervals.begin(), sorted_intervals.end(), 
            [](const ThermalInterval& a, const ThermalInterval& b) {
              return a.start_time < b.start_time;
            });
  
  std::vector<ThermalInterval> merged;
  merged.push_back(sorted_intervals[0]);
  
  for (size_t i = 1; i < sorted_intervals.size(); i++) {
    ThermalInterval& last = merged.back();
    const ThermalInterval& current = sorted_intervals[i];
    
    if (current.start_time <= last.end_time) {
      // Overlapping - merge them
      last.end_time = std::max(last.end_time, current.end_time);
    } else {
      // Non-overlapping - add new interval
      merged.push_back(current);
    }
  }
  
  return merged;
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::is_point_in_spparks_domain(double x, double y, double z) const
{
  // Use per-rank subdomain bounds (with padding) so each rank only considers
  // nodes relevant to its partition.  Thermal intervals computed here are
  // later combined across ranks via MPI_Allreduce(MPI_MIN).
  double pad = dx * grid_cell_size_multiplier;
  double x_min = domain->subxlo * dx - pad;
  double x_max = domain->subxhi * dx + pad;
  double y_min = domain->subylo * dx - pad;
  double y_max = domain->subyhi * dx + pad;
  double z_min = domain->subzlo * dx - pad;
  double z_max = domain->subzhi * dx + pad;
  
  return (x >= x_min && x <= x_max &&
          y >= y_min && y <= y_max &&
          z >= z_min && z <= z_max);
}