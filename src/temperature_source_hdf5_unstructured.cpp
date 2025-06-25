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
  active_layer(std::numeric_limits<unsigned>::max())
{
  source_initialized = false;
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
  
  // Open HDF5 file
  try {
    file = std::make_shared<HighFive::File>(filename, HighFive::File::ReadOnly);
  } catch (const std::exception& e) {
    error->all(FLERR, (std::string("Failed to open HDF5 file: ") + e.what()).c_str());
  }
  
  // Read layer times
  file->getDataSet("layerTimes").read(layerTimes);
  
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
  auto grp = file->getGroup(std::to_string(layerIdx));
  
  // Read bounding boxes
  std::vector<std::vector<double>> bboxes;
  grp.getDataSet("boundingBoxes").read(bboxes);
  
  // Read element and node pointers
  std::vector<unsigned> elemPtr;
  grp.getDataSet("elemPtrs").read(elemPtr);
  
  std::vector<unsigned> nodePtr;
  grp.getDataSet("nodePtrs").read(nodePtr);
  
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
  if (universe->me == 0 && chunksToLoad.size() > 0) {
    fprintf(screen, "  Initial chunks overlapping SPPARKS domain: %zu\n", chunksToLoad.size());
  }
  
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
    // Calculate data reduction statistics
    unsigned totalElements = elemPtr.back();
    unsigned loadedElements = elemOffsets.back();
    unsigned totalNodes = nodePtr.back();
    unsigned loadedNodes = nodeOffsets.back();
    
    fprintf(screen, "  Chunk loading optimization results:\n");
    fprintf(screen, "    Total chunks: %zu, Loaded: %zu (%.1f%%)\n", 
            bboxes.size(), overlappingChunks.size(), 
            100.0 * overlappingChunks.size() / bboxes.size());
    fprintf(screen, "    Total elements: %u, Loaded: %u (%.1f%%)\n",
            totalElements, loadedElements,
            100.0 * loadedElements / totalElements);
    fprintf(screen, "    Total nodes: %u, Loaded: %u (%.1f%%)\n",
            totalNodes, loadedNodes,
            100.0 * loadedNodes / totalNodes);
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
  grp.getDataSet("elementToNode").select(
    build_hyperslab(elemSlices, {0, NODES_PER_ELEM})).read(elemNodeData);
  elemNode = Array2D<unsigned>(NODES_PER_ELEM, std::move(elemNodeData));
  
  // Read node coordinates
  std::vector<double> nodeCoordsData;
  grp.getDataSet("nodeCoords").select(
    build_hyperslab(nodeSlices, {0, DIM})).read(nodeCoordsData);
  nodeCoords = Array2D<double>(DIM, std::move(nodeCoordsData));
  
  // Read data counts
  grp.getDataSet("dataCounts").select(
    build_hyperslab_1d(nodeSlices)).read(dataCounts);
  
  // Find maximum data count
  unsigned maxData = 0;
  for (auto cnt : dataCounts) {
    maxData = std::max(maxData, cnt);
  }
  
  // Read time and temperature data
  auto dataHyperSlab = build_hyperslab(nodeSlices, {0, maxData});
  
  std::vector<double> timesData;
  grp.getDataSet("times").select(dataHyperSlab).read(timesData);
  times = Array2D<double>(maxData, std::move(timesData));
  
  std::vector<double> tempData;
  grp.getDataSet("temperatures").select(dataHyperSlab).read(tempData);
  temperatures = Array2D<double>(maxData, std::move(tempData));
  
  // Build element bounding boxes for spatial queries
  elem_bboxes = build_elem_bounding_boxes(overlappingChunks.size(), 
                                         nodeOffsets, elemOffsets, 
                                         elemNode, nodeCoords);
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
  if (t < layerTimes.front()) {
    error->all(FLERR, "Time out of range of layer times (too early)");
  }
  if (t > layerTimes.back()) {
    error->all(FLERR, "Time out of range of layer times (too late)");
  }
  
  auto lower = std::lower_bound(layerTimes.begin(), layerTimes.end(), t);
  auto idx = std::distance(layerTimes.begin(), lower);
  return (idx == 0) ? 0 : idx - 1;
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

double HDF5UnstructuredTemperatureSource::get_temperature_at_site(int site_index, double time)
{
  // Get the app and access site coordinates
  if (!app) {
    error->all(FLERR, "App not available for site coordinate lookup");
  }
  
  // Get coordinates from the app's xyz array
  double x = app->xyz[site_index][0];
  double y = app->xyz[site_index][1]; 
  double z = app->xyz[site_index][2];
  
  return get_temperature_at_xyz_and_time(x, y, z, time);
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::all_temperatures_below_threshold(double time)
{
  if (!app) {
    error->all(FLERR, "App not available for temperature checking");
  }
  
  // Check temperatures at all local sites
  for (int i = 0; i < app->nlocal; i++) {
    double temp = get_temperature_at_site(i, time);
    if (temp >= threshold_temp) {
      return false;
    }
  }
  
  return true;
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::has_significant_thermal_activity(double time)
{
  if (!app) {
    error->all(FLERR, "App not available for temperature checking");
  }
  
  // Multiple time horizons for detecting different thermal phenomena
  const double dt_short = 3e-6;   // 3 microsecond - detect rapid heating onset
  const double dt_medium = 1e-5;  // 10 microsecond - detect significant thermal rates
  const double dt_long = 2e-5;    // 20 microsecond - detect slower thermal trends
  
  const double significant_heating_rate = 50000.0;  // K/s - rapid heating (melt pool approach)
  const double significant_cooling_rate = 10000.0;  // K/s - significant cooling/heating
  const double moderate_rate = 5000.0;             // K/s - moderate thermal activity
  const double solidus_temp = 1563.0;              // Solidus temperature (K)
  const double warm_threshold = 400.0;             // Elevated temperature threshold
  
  // Counters for different types of thermal activity
  int hot_sites = 0;              // Above threshold temperature
  int warm_sites = 0;             // Above warm threshold
  int rapid_heating_sites = 0;    // Sites heating rapidly (melt pool approach)
  int significant_activity_sites = 0; // Sites with significant thermal rates
  int cooling_sites = 0;          // Sites cooling significantly
  int trend_heating_sites = 0;    // Sites showing heating trend
  
  // Sample subset of sites for performance (every 10th site for large domains)
  int sample_step = std::max(1, app->nlocal / 1000);  // Sample ~1000 sites max
  
  for (int i = 0; i < app->nlocal; i += sample_step) {
    double temp_now = get_temperature_at_site(i, time);
    double temp_short = get_temperature_at_site(i, time + dt_short);
    double temp_medium = get_temperature_at_site(i, time + dt_medium);
    double temp_long = get_temperature_at_site(i, time + dt_long);
    
    // Current temperature classification
    if (temp_now >= threshold_temp) {
      hot_sites++;
    } else if (temp_now >= warm_threshold) {
      warm_sites++;
    }
    
    // Short-term rapid heating detection (critical for melt pool onset)
    double short_heating_rate = (temp_short - temp_now) / dt_short;
    if (short_heating_rate > significant_heating_rate) {
      rapid_heating_sites++;
    }
    
    // Medium-term thermal activity detection
    double medium_rate = std::abs(temp_medium - temp_now) / dt_medium;
    if (medium_rate > significant_cooling_rate) {
      significant_activity_sites++;
      
      // Distinguish heating vs cooling
      if (temp_medium < temp_now && temp_now > solidus_temp) {
        cooling_sites++;  // Cooling from high temperature
      }
    }
    
    // Long-term trend detection (heating trend that might accelerate)
    double long_heating_rate = (temp_long - temp_now) / dt_long;
    if (long_heating_rate > moderate_rate && temp_now > warm_threshold) {
      trend_heating_sites++;
    }
  }
  
  // Scale counts based on sampling
  if (sample_step > 1) {
    hot_sites *= sample_step;
    warm_sites *= sample_step;
    rapid_heating_sites *= sample_step;
    significant_activity_sites *= sample_step;
    cooling_sites *= sample_step;
    trend_heating_sites *= sample_step;
  }
  
  // Multi-criteria decision for thermal activity
  bool has_activity = false;
  
  // Critical conditions - always significant
  if (hot_sites > 0) {
    has_activity = true;  // Active melt pool
  }
  
  // Rapid heating detection - catches melt pool approach
  if (rapid_heating_sites > 5) {
    has_activity = true;  // Rapid heating onset detected
  }
  
  // Significant thermal activity
  if (significant_activity_sites > 10) {
    has_activity = true;  // Significant thermal rates
  }
  
  // Solidification zone activity
  if (warm_sites > 50 && (cooling_sites > 5 || significant_activity_sites > 20)) {
    has_activity = true;  // Active solidification zone
  }
  
  // Heating trend that might accelerate
  if (trend_heating_sites > 20 && warm_sites > 30) {
    has_activity = true;  // Approaching thermal event
  }
  
  return has_activity;
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::has_significant_thermal_activity_hdf5_nodes(double time)
{
  // Check if we have valid data loaded
  if (selected_chunks.empty() || dataCounts.empty()) {
    return false;
  }
  
  // Multiple time horizons for detecting different thermal phenomena
  const double dt_short = 3e-6;   // 3 microsecond - detect rapid heating onset
  const double dt_medium = 1e-5;  // 10 microsecond - detect significant thermal rates
  const double dt_long = 2e-5;    // 20 microsecond - extended lookahead for early detection
  
  const double significant_heating_rate = 10000.0;  // K/s - rapid heating (melt pool approach) - more sensitive
  const double significant_cooling_rate = 5000.0;   // K/s - significant cooling/heating - more sensitive
  const double solidus_temp = 1563.0;              // Solidus temperature (K)
  const double warm_threshold = 350.0;             // Elevated temperature threshold - more sensitive
  
  // Counters for different types of thermal activity (local to this processor)
  int local_hot_nodes = 0;              // Above threshold temperature
  int local_warm_nodes = 0;             // Above warm threshold
  int local_rapid_heating_nodes = 0;    // Nodes heating rapidly (melt pool approach)
  int local_significant_activity_nodes = 0; // Nodes with significant thermal rates
  int local_cooling_nodes = 0;          // Nodes cooling significantly
  
  // Check all HDF5 nodes in the loaded chunks
  unsigned total_nodes = 0;
  for (unsigned chunk_idx = 0; chunk_idx < selected_chunks.size(); chunk_idx++) {
    unsigned start_node = (chunk_idx == 0) ? 0 : node_offsets[chunk_idx];
    unsigned end_node = (chunk_idx + 1 < node_offsets.size()) ? node_offsets[chunk_idx + 1] : dataCounts.size();
    
    for (unsigned node_idx = start_node; node_idx < end_node && node_idx < dataCounts.size(); node_idx++) {
      total_nodes++;
      
      if (dataCounts[node_idx] == 0) continue;  // Skip nodes with no data
      
      // Get node coordinates (convert from meters to SPPARKS units)
      double node_x = nodeCoords(node_idx, 0) / dx;
      double node_y = nodeCoords(node_idx, 1) / dx;
      double node_z = nodeCoords(node_idx, 2) / dx;
      
      // Check if node is within this processor's domain bounds
      // Note: For thermal activity detection, we want to be inclusive rather than exclusive
      // so we check a broader region to ensure we don't miss thermal events
      
      // Get temperatures at different time horizons using direct HDF5 interpolation
      const auto timeIter = times.row_iterator(node_idx);
      const auto tempIter = temperatures.row_iterator(node_idx);
      
      // Interpolate temperature at current time
      double temp_now = 0.0;
      if (time >= timeIter[0] && time <= timeIter[dataCounts[node_idx] - 1]) {
        auto lower = std::lower_bound(timeIter, timeIter + dataCounts[node_idx], time);
        auto idx = std::distance(timeIter, lower);
        if (idx == 0) idx = 1;
        temp_now = tempIter[idx-1] + (tempIter[idx] - tempIter[idx-1]) / 
                   (timeIter[idx] - timeIter[idx-1]) * (time - timeIter[idx-1]);
      } else {
        continue;  // Time out of range for this node
      }
      
      // Interpolate temperature at short future time
      double temp_short = 0.0;
      double time_short = time + dt_short;
      if (time_short >= timeIter[0] && time_short <= timeIter[dataCounts[node_idx] - 1]) {
        auto lower = std::lower_bound(timeIter, timeIter + dataCounts[node_idx], time_short);
        auto idx = std::distance(timeIter, lower);
        if (idx == 0) idx = 1;
        temp_short = tempIter[idx-1] + (tempIter[idx] - tempIter[idx-1]) / 
                     (timeIter[idx] - timeIter[idx-1]) * (time_short - timeIter[idx-1]);
      } else {
        temp_short = temp_now;  // Use current temp if future time out of range
      }
      
      // Interpolate temperature at medium future time
      double temp_medium = 0.0;
      double time_medium = time + dt_medium;
      if (time_medium >= timeIter[0] && time_medium <= timeIter[dataCounts[node_idx] - 1]) {
        auto lower = std::lower_bound(timeIter, timeIter + dataCounts[node_idx], time_medium);
        auto idx = std::distance(timeIter, lower);
        if (idx == 0) idx = 1;
        temp_medium = tempIter[idx-1] + (tempIter[idx] - tempIter[idx-1]) / 
                      (timeIter[idx] - timeIter[idx-1]) * (time_medium - timeIter[idx-1]);
      } else {
        temp_medium = temp_now;  // Use current temp if future time out of range
      }
      
      // Interpolate temperature at long future time for early detection
      double temp_long = 0.0;
      double time_long = time + dt_long;
      if (time_long >= timeIter[0] && time_long <= timeIter[dataCounts[node_idx] - 1]) {
        auto lower = std::lower_bound(timeIter, timeIter + dataCounts[node_idx], time_long);
        auto idx = std::distance(timeIter, lower);
        if (idx == 0) idx = 1;
        temp_long = tempIter[idx-1] + (tempIter[idx] - tempIter[idx-1]) / 
                    (timeIter[idx] - timeIter[idx-1]) * (time_long - timeIter[idx-1]);
      } else {
        temp_long = temp_now;  // Use current temp if future time out of range
      }
      
      // Current temperature classification
      if (temp_now >= threshold_temp) {
        local_hot_nodes++;
      } else if (temp_now >= warm_threshold) {
        local_warm_nodes++;
      }
      
      // Short-term rapid heating detection (critical for melt pool onset)
      double short_heating_rate = (temp_short - temp_now) / dt_short;
      if (short_heating_rate > significant_heating_rate) {
        local_rapid_heating_nodes++;
      }
      
      // Medium-term thermal activity detection
      double medium_rate = std::abs(temp_medium - temp_now) / dt_medium;
      if (medium_rate > significant_cooling_rate) {
        local_significant_activity_nodes++;
        
        // Distinguish heating vs cooling
        if (temp_medium < temp_now && temp_now > solidus_temp) {
          local_cooling_nodes++;  // Cooling from high temperature
        }
      }
      
      // Long-term thermal activity detection for early warning
      double long_rate = std::abs(temp_long - temp_now) / dt_long;
      if (long_rate > significant_cooling_rate || 
          (temp_long > temp_now + 50.0)) {  // Any significant heating trend
        local_significant_activity_nodes++;
      }
    }
  }
  
  // Determine local thermal activity status
  bool local_has_activity = false;
  
  // Critical conditions - always significant
  if (local_hot_nodes > 0) {
    local_has_activity = true;  // Active melt pool
  }
  
  // Rapid heating detection - catches melt pool approach
  if (local_rapid_heating_nodes > 0) {
    local_has_activity = true;  // Any rapid heating onset detected
  }
  
  // Significant thermal activity
  if (local_significant_activity_nodes > 5) {
    local_has_activity = true;  // Significant thermal rates
  }
  
  // Solidification zone activity
  if (local_warm_nodes > 10 && (local_cooling_nodes > 0 || local_significant_activity_nodes > 5)) {
    local_has_activity = true;  // Active solidification zone
  }
  
  // MPI communication to determine global thermal activity status
  int local_activity = local_has_activity ? 1 : 0;
  int global_activity = 0;
  
  // Use MPI_Allreduce to get the maximum activity status across all processors
  // If any processor has thermal activity, all processors will know
  MPI_Allreduce(&local_activity, &global_activity, 1, MPI_INT, MPI_MAX, universe->uworld);
  
  // Debug output to understand what's happening - only when no activity detected
  if (universe->me == 0 && global_activity == 0) {
    printf("DEBUG: Time %.3e - Hot nodes: %d, Warm nodes: %d, Rapid heating: %d, Total nodes checked: %d, Activity: NO\n",
           time, local_hot_nodes, local_warm_nodes, local_rapid_heating_nodes, total_nodes);
  }
  
  return (global_activity > 0);
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::find_next_active_time(double start_time, double end_time)
{
  // Binary search for the next time when significant thermal activity occurs
  const double time_tolerance = 1e-6;  // 1 microsecond tolerance
  const int max_iterations = 50;
  
  double low = start_time;
  double high = end_time;
  
  // First check if there's significant thermal activity at end_time
  if (!has_significant_thermal_activity(end_time)) {
    return end_time;  // No significant activity in the window
  }
  
  // Binary search
  int iteration = 0;
  while (high - low > time_tolerance && iteration < max_iterations) {
    double mid = 0.5 * (low + high);
    
    if (!has_significant_thermal_activity(mid)) {
      low = mid;
    } else {
      high = mid;
    }
    
    iteration++;
  }
  
  // Return the first time when significant thermal activity occurs
  return high;
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::find_next_active_time_sequential(double start_time, double end_time)
{
  // Sequential search with adaptive step size for large search windows
  // This is safer for rapid heating/cooling events but slower
  
  double current = start_time;
  double step = 25e-6;  // Start with 25 microsecond steps (more conservative)
  const double min_step = 5e-6;   // Minimum 5 microsecond steps (more resolution)
  const double max_step = 500e-6; // Maximum 500 microsecond steps (more conservative)
  
  while (current < end_time) {
    if (has_significant_thermal_activity(current)) {
      // Found significant thermal activity, refine with smaller steps
      if (step > min_step) {
        current -= step;  // Back up
        step = min_step;  // Use minimum step size
        continue;
      }
      return current;
    }
    
    // Adaptive step size: increase step if we're far from activity
    // But be more conservative during potential cooling phases
    if (current + 5 * step < end_time) {
      step = std::min(step * 1.2, max_step);  // Slower growth rate
    }
    
    current += step;
  }
  
  return end_time;
}