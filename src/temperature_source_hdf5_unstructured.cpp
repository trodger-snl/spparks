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

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) :
  TemperatureSource(spk),
  current_time(std::numeric_limits<double>::lowest()),
  active_layer(std::numeric_limits<unsigned>::max()),
  cache_valid(false),
  total_layer_load_time(0.0),
  layer_load_count(0)
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
  // Parse arguments: filename dx threshold_temp default_temp [bounds_check_mode]
  if (args.size() < 4) {
    error->all(FLERR, "HDF5 unstructured temperature source requires: filename dx threshold_temp default_temp [bounds_check_mode]");
  }
  
  filename = args[0];
  dx = std::stod(args[1]);  // Grid spacing in meters
  threshold_temp = std::stod(args[2]);  // Threshold temperature for fast-forward
  default_temp = std::stod(args[3]);    // Default/ambient temperature
  bounds_check_mode = (args.size() > 4) ? std::stoi(args[4]) : 0;

  // Open HDF5 file with parallel I/O when available
  try {
#ifdef H5_HAVE_PARALLEL
    // Create file access property list with MPI-IO
    HighFive::FileAccessProps fapl;
    fapl.add(HighFive::MPIOFileAccess(universe->uworld, MPI_INFO_NULL));
    fapl.add(HighFive::MPIOCollectiveMetadata(true));
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

  if (universe->me == 0) {
    fprintf(screen, "Loading layer %u at time %.6e s\n", layerIdx,
            (layerIdx < layerTimes.size()) ? layerTimes[layerIdx] : -1.0);
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
  
  // Convert SPPARKS domain bounds from lattice units to physical coordinates (meters)
  std::vector<double> spparksPhysicalBbox{
    domain->boxxlo * dx, domain->boxylo * dx, domain->boxzlo * dx,
    domain->boxxhi * dx, domain->boxyhi * dx, domain->boxzhi * dx
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
  chunk_bboxes = bboxes;
  selected_chunks = overlappingChunks;
  
  if (universe->me == 0 && overlappingChunks.size() > 0) {
    unsigned totalElements = elemPtr.back();
    unsigned loadedElements = elemOffsets.back();
    unsigned totalNodes = nodePtr.back();
    unsigned loadedNodes = nodeOffsets.back();

    fprintf(screen, "  Loaded %zu/%zu chunks (%.1f%%), %u/%u elements (%.1f%%), %u/%u nodes (%.1f%%)\n",
            overlappingChunks.size(), bboxes.size(), 100.0 * overlappingChunks.size() / bboxes.size(),
            loadedElements, totalElements, 100.0 * loadedElements / totalElements,
            loadedNodes, totalNodes, 100.0 * loadedNodes / totalNodes);
  }
  node_offsets = nodeOffsets;
  elem_offsets = elemOffsets;

  // Build hyperslabs for selective reading
  auto build_hyperslab = [](const std::vector<std::array<size_t, 2>>& slices, 
                           const std::array<size_t, 2>& cols) -> HighFive::HyperSlab {
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
  };
  
  auto build_hyperslab_1d = [](const std::vector<std::array<size_t, 2>>& slices) -> HighFive::HyperSlab {
    if (slices.empty()) throw std::runtime_error("No slices for hyperslab");
    
    HighFive::HyperSlab result(HighFive::RegularHyperSlab(
      {slices[0][0]}, {slices[0][1] - slices[0][0]}));
    
    for (size_t r = 1; r < slices.size(); r++) {
      result |= HighFive::RegularHyperSlab(
        {slices[r][0]}, {slices[r][1] - slices[r][0]});
    }
    return result;
  };
  
  // Read element connectivity
  std::vector<unsigned> elemNodeData;
  grp->getDataSet("elementToNode").select(
    build_hyperslab(elemSlices, {0, NODES_PER_ELEM})).read(elemNodeData);
  elemNode = Array2D<unsigned>(NODES_PER_ELEM, std::move(elemNodeData));

  // Read node coordinates
  std::vector<double> nodeCoordsData;
  grp->getDataSet("nodeCoords").select(
    build_hyperslab(nodeSlices, {0, DIM})).read(nodeCoordsData);
  nodeCoords = Array2D<double>(DIM, std::move(nodeCoordsData));

  // Read data counts
  grp->getDataSet("dataCounts").select(
    build_hyperslab_1d(nodeSlices)).read(dataCounts);

  // Find maximum data count
  unsigned maxData = 0;
  for (auto cnt : dataCounts) {
    maxData = std::max(maxData, cnt);
  }

  // Read time and temperature data
  auto dataHyperSlab = build_hyperslab(nodeSlices, {0, maxData});

  std::vector<double> timesData;
  grp->getDataSet("times").select(dataHyperSlab).read(timesData);
  times = Array2D<double>(maxData, std::move(timesData));

  std::vector<double> tempData;
  grp->getDataSet("temperatures").select(dataHyperSlab).read(tempData);
  temperatures = Array2D<double>(maxData, std::move(tempData));
  
  // Build element bounding boxes for spatial queries
  elem_bboxes = build_elem_bounding_boxes(overlappingChunks.size(), 
                                         nodeOffsets, elemOffsets, 
                                         elemNode, nodeCoords);
  
  // Compute thermal intervals for efficient time queries
  compute_thermal_intervals_for_layer(layerIdx, threshold_temp);

  // Record timing for performance monitoring
  auto load_end = std::chrono::high_resolution_clock::now();
  double load_time = std::chrono::duration<double>(load_end - load_start).count();
  total_layer_load_time += load_time;
  layer_load_count++;

  if (universe->me == 0) {
    fprintf(screen, "  Layer load time: %.3f s (avg: %.3f s over %d loads)\n",
            load_time, total_layer_load_time / layer_load_count, layer_load_count);
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
  const std::vector<double>& bbox) const
{
  return pt[0] >= bbox[0] && pt[0] <= bbox[3] && 
         pt[1] >= bbox[1] && pt[1] <= bbox[4] && 
         pt[2] >= bbox[2] && pt[2] <= bbox[5];
}

/* ---------------------------------------------------------------------- */

std::pair<std::array<unsigned, 4>, std::array<double, 3>> 
HDF5UnstructuredTemperatureSource::find_element_point_is_in(
  const std::vector<unsigned>& selectedChunks,
  const std::vector<std::vector<double>>& chunkBboxes,
  const std::vector<unsigned>& nodeOffsets,
  const std::vector<unsigned>& elemOffsets,
  const Array2D<unsigned>& elemNode,
  const Array2D<double>& nodeCoords,
  const std::vector<std::vector<double>>& elemBboxes,
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
  
  // Find possible chunks
  std::vector<unsigned> possibleChunks;
  for (unsigned c = 0; c < selectedChunks.size(); c++) {
    if (selectedChunks[c] >= chunkBboxes.size()) continue;
    const auto& cBbox = chunkBboxes[selectedChunks[c]];
    if (point_in_bbox(pt, cBbox)) {
      possibleChunks.push_back(c);
    }
  }
  
  // Search elements in possible chunks
  for (auto c : possibleChunks) {
    if (c + 1 >= elemOffsets.size()) continue;
    for (unsigned e = elemOffsets[c]; e < elemOffsets[c+1]; e++) {
      if (e >= elemBboxes.size()) continue;
      if (point_in_bbox(pt, elemBboxes[e])) {
        std::vector<std::vector<double>> tetCoords(4, std::vector<double>(DIM));
        
        // Validate array bounds before accessing
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
        
        if (!validElement) continue;
        
        parCoords = get_parametric_coordinates_of_point(tetCoords, pt);
        
        // Check if point is inside tetrahedron
        if (parCoords[0] > -tol && parCoords[1] > -tol && parCoords[2] > -tol && 
            1.0 - parCoords[0] - parCoords[1] - parCoords[2] > -tol) {
          return std::make_pair(nodeIds, parCoords);
        }
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

std::vector<std::vector<double>> HDF5UnstructuredTemperatureSource::build_elem_bounding_boxes(
  unsigned nChunks,
  const std::vector<unsigned>& nodeOffsets,
  const std::vector<unsigned>& elemOffsets,
  const Array2D<unsigned>& elemNode,
  const Array2D<double>& nodeCoords) const
{
  constexpr double maxVal = std::numeric_limits<double>::max();
  constexpr double minVal = std::numeric_limits<double>::lowest();
  
  std::vector<std::vector<double>> result;
  
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

double HDF5UnstructuredTemperatureSource::get_temperature_at_site(int site_index, double time)
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
      cache_valid = false;  // Invalidate cache when layer changes
    }
    current_time = time;
  }
  
  // Build cache if needed
  if (!cache_valid) {
    build_site_element_cache();
  }
  
  // Get cached element for this site
  const ElementCache& cache = get_cached_element(site_index);
  
  if (!cache.valid) {
    return default_temp;  // Site not in any element
  }
  
  // Get temperatures at the four tetrahedral nodes
  std::array<double, NODES_PER_ELEM> nodalVals;
  for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
    const unsigned nodeIdx = cache.nodeIndices[n];
    
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
  
  // Tetrahedral interpolation using cached barycentric coordinates
  return nodalVals[0] + 
         cache.weights[0] * (nodalVals[1] - nodalVals[0]) + 
         cache.weights[1] * (nodalVals[2] - nodalVals[0]) + 
         cache.weights[2] * (nodalVals[3] - nodalVals[0]);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_site_element_cache() const
{
  // Get total number of sites from app
  int nlocal = app->nlocal;
  
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
  
  cache_valid = true;
  
  if (universe->me == 0) {
    fprintf(screen, "  Built element cache for %d sites\n", nlocal);
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

  // Check remaining intervals in current layer
  for (const auto& interval : layer_thermal_intervals[current_layer]) {
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
  // Convert SPPARKS domain bounds from lattice units to physical coordinates (meters)
  double x_min = domain->boxxlo * dx;
  double x_max = domain->boxxhi * dx;
  double y_min = domain->boxylo * dx;
  double y_max = domain->boxyhi * dx;
  double z_min = domain->boxzlo * dx;
  double z_max = domain->boxzhi * dx;
  
  return (x >= x_min && x <= x_max &&
          y >= y_min && y <= y_max &&
          z >= z_min && z <= z_max);
}