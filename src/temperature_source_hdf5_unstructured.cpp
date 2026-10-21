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
#ifdef SPPARKS_HIGHFIVE
#include "highfive/highfive.hpp"
#ifdef SPPARKS_ZLIB
#include <zlib.h>
#endif
#endif
#include <algorithm>
#include <functional>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <set>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) :
  TemperatureSource(spk),
  current_time(std::numeric_limits<double>::lowest()),
  active_layer(std::numeric_limits<unsigned>::max()),
  cache_valid(false),
  cache_generation_(0),
  total_layer_load_time(0.0),
  layer_load_count(0),
  cached_nodal_time(std::numeric_limits<double>::lowest()),
  nodal_cache_valid(false),
  site_intervals_for_layer(std::numeric_limits<unsigned>::max()),
  grid_cell_size_multiplier(100.0),
  nodal_smooth_enabled(false),
  nodal_sigma_xy(4.0),
  nodal_sigma_z(0.3),
  nodal_passes(1),
  nodal_alpha(0.5),
  nodal_adjacency_valid(false)
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
#ifndef SPPARKS_HIGHFIVE
  (void)args;
  error->all(FLERR, "HDF5 temperature sources need a build with HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
  // Parse arguments: filename dx threshold_temp default_temp [ignored] [grid_cell_mult]
  // (args[4] was a bounds_check_mode that nothing ever read; the slot is kept
  // so existing decks that pass grid_cell_mult in position 5 keep working.)
  if (args.size() < 4) {
    error->all(FLERR, "HDF5 unstructured temperature source requires: filename dx threshold_temp default_temp [ignored] [grid_cell_mult]");
  }

  filename = args[0];
  dx = std::stod(args[1]);  // Grid spacing in meters
  threshold_temp = std::stod(args[2]);  // Threshold temperature for fast-forward
  default_temp = std::stod(args[3]);    // Default/ambient temperature
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
      // error->one, NOT error->all: this is inside `if (universe->me == 0)`, and
      // Error::all opens with MPI_Barrier(world), so firing it on rank 0 alone
      // blocks rank 0 in the barrier while every other rank runs on -- a hang
      // instead of a diagnostic.
      error->one(FLERR, "layerTimes array is empty");
    }

    // Layer times must be strictly increasing (this also rejects negative,
    // duplicate and interior-zero entries).
    if (layerTimes.front() < 0.0 ||
        std::adjacent_find(layerTimes.begin(), layerTimes.end(),
                           std::greater_equal<double>()) != layerTimes.end()) {
      error->one(FLERR, "layerTimes must be non-negative and strictly increasing");
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
#endif
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_temperature_at_xyz_and_time(
  double, double, double, double)
{
  // Every live path goes through the per-site element cache
  // (prepare_for_timestep + get_temperature_at_site_fast).
  error->all(FLERR, "HDF5 temperature source: point queries are not supported; "
                    "use prepare_for_timestep + get_temperature_at_site_fast");
  return default_temp;
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
#ifndef SPPARKS_HIGHFIVE
  (void)layerIdx;
  error->all(FLERR, "HDF5 temperature sources need a build with HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
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
  
  // Convert SPPARKS per-rank subdomain bounds to physical coordinates (meters).
  // Use subdomain (subxlo/hi) instead of global domain (boxxlo/hi) so each rank
  // only loads temperature chunks overlapping its own partition.
  // Pad by grid_cell_mult*dx to capture thermal elements straddling the
  // subdomain boundary (needed for interpolation at boundary sites).
  double pad = dx * grid_cell_size_multiplier;
  std::vector<double> spparksPhysicalBbox{
    domain->subxlo * dx - pad, domain->subylo * dx - pad, domain->subzlo * dx - pad,
    domain->subxhi * dx + pad, domain->subyhi * dx + pad, domain->subzhi * dx + pad
  };

  // Diagnostic: on any axis that is actually decomposed, a pad wider than the
  // subdomain itself erases that axis from chunk selection -- every rank's
  // selection box then spans the whole domain on that axis and they all load
  // the same chunks, paying N-fold I/O and memory for no locality.
  //
  // Measured on the ct15 box deck (311x317x51 sites, dx=5e-6, so a 0.255 mm
  // z-extent against a 0.5 mm pad): forcing `processors 1 1 4` made all four
  // ranks load a byte-identical 93/291 chunks, versus 66/69/62/62 for the
  // default 2x2x1. Same temperatures, 39% more wall time, 25% more memory.
  //
  // This is NOT a reason to shrink the pad -- it is load-bearing for boundary
  // element interpolation, for ghost sites (the cache covers nall), and for the
  // nodal_smooth halo, which already warns that it only has ~one tet hop. It is
  // a reason to not decompose that axis. Warned once per run, from rank 0.
  if (!pad_vs_subdomain_warned && universe->me == 0 && screen) {
    const char *axis_name[3] = {"x","y","z"};
    const double sublo[3] = {domain->subxlo, domain->subylo, domain->subzlo};
    const double subhi[3] = {domain->subxhi, domain->subyhi, domain->subzhi};
    for (int ax = 0; ax < 3; ax++) {
      if (domain->procgrid[ax] <= 1) continue;      // axis not decomposed
      const double extent = (subhi[ax] - sublo[ax]) * dx;
      if (pad >= extent) {
        fprintf(screen,
          "WARNING: chunk-load pad (%.3g m = %g*dx) >= this rank's %s-extent "
          "(%.3g m) while %s is split over %d procs. Chunk selection ignores "
          "the %s decomposition, so every rank loads nearly the same chunks: "
          "expect ~%dx redundant thermal I/O and memory on this axis with no "
          "locality gained. Decompose %s less (or not at all), or raise the "
          "site count per rank along %s.\n",
          pad, grid_cell_size_multiplier, axis_name[ax], extent,
          axis_name[ax], domain->procgrid[ax], axis_name[ax],
          domain->procgrid[ax], axis_name[ax], axis_name[ax]);
        pad_vs_subdomain_warned = true;
      }
    }
  }
  
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
  
  // Drop the previous layer before reading the next one, so peak memory is one
  // layer, not two (the members below were otherwise replaced only after the
  // new arrays were already allocated).
  clear_layer_data();

  if (chunksToLoad.empty()) {
    // This rank's padded subdomain overlaps no chunk of this layer, so there is
    // genuinely no thermal data here -- normal when thin boxes are tiled across
    // a part and stitched back together, since a box can sit far from the layer
    // currently being deposited.
    //
    // The clear above matters most here: returning with the PREVIOUS layer's
    // mesh and time series still resident made every site keep
    // interpolating the old layer's field: silently harmless while each layer's
    // series happens to end where the next begins (the reader then falls back to
    // default_temp on the time-bounds check), but silently WRONG the moment a
    // layer carries a cooldown tail into the next layer's window, which nothing
    // in the format forbids.
    //
    // With the state cleared, build_site_element_cache finds no mesh, every
    // cache entry comes back invalid, and every site reads default_temp -- the
    // correct answer for "no data" -- while the empty interval list lets
    // get_next_time_with_temperature fast-forward to the next layer.
    if (layerIdx >= layer_thermal_intervals.size())
      layer_thermal_intervals.resize(layerIdx + 1);
    layer_thermal_intervals[layerIdx].clear();
    if (screen)
      fprintf(screen, "  Rank %d: layer %u has no chunks overlapping this "
              "subdomain; no thermal data here, sites read default_temp "
              "(%.1f K) until a later layer covers them\n",
              universe->me, layerIdx, default_temp);
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

    // Deliberately per-rank (not gated on rank 0) -- the point is the per-rank
    // chunk split. But `screen` is NULL on every non-root rank whenever ANY
    // -screen flag is given (spparks.cpp:136-140 sets uscreen = stdout only when
    // screenflag == 0), and NULL on rank 0 too under `-screen none`. Unguarded,
    // this fprintf segfaults there.
    if (screen)
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

  // Read node coordinates
  std::vector<double> nodeCoordsData;
  grp->getDataSet("nodeCoords").select(
    build_hyperslab(nodeSlices, {0, DIM})).read(nodeCoordsData);
  nodeCoords = Array2D<double>(DIM, std::move(nodeCoordsData));

  // Read data counts
  grp->getDataSet("dataCounts").select(
    build_hyperslab_1d(nodeSlices)).read(dataCounts);

  // Build element bounding boxes for spatial queries
  elem_bboxes = build_elem_bounding_boxes();

  // Build the spatial acceleration grid used by every element lookup.
  build_spatial_grid();

  // Build node-graph adjacency for mesh-Laplacian smoothing (optional).
  // Must run AFTER elemNode + nodeCoords + dataCounts are populated for the
  // new layer and BEFORE any precompute_nodal_temperatures call uses the
  // result. Rebuilt on every layer load when enabled.
  if (nodal_smooth_enabled) {
    build_nodal_adjacency_weighted();
  } else {
    nodal_adjacency_valid = false;
  }

  // The element cache needs only the mesh, and it decides which nodes' series
  // are ever read, so build it before reading the series.
  build_site_element_cache();

  // Keep the series only for nodes something reads: the tet nodes of this
  // rank's sites (or their smoothing halo, which precompute interpolates) and
  // nodes inside the global box (node-level thermal intervals). The rest --
  // most of the chunk-load pad -- get dataCount 0, so the readers below store
  // nothing for them and every later reader sees an empty row.
  {
    const std::vector<unsigned> &used =
      (nodal_smooth_enabled && !smooth_interp_indices.empty())
        ? smooth_interp_indices : active_node_indices;
    std::vector<unsigned char> keep(dataCounts.size(), 0);
    for (unsigned nd : used) keep[nd] = 1;
    for (size_t nd = 0; nd < dataCounts.size(); nd++)
      if (!keep[nd] && !is_point_in_spparks_domain(nodeCoords(nd, 0), nodeCoords(nd, 1),
                                                   nodeCoords(nd, 2)))
        dataCounts[nd] = 0;
  }

  // Read time and temperature data (virtual — overridden by CSR subclass)
  read_time_temperature_data(*grp, nodeSlices);

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
#endif
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::clear_layer_data()
{
  // Mesh + connectivity
  elemNode = Array2D<unsigned>();
  nodeCoords = Array2D<double>();
  chunk_bboxes.clear();
  selected_chunks.clear();
  node_offsets.clear();
  elem_offsets.clear();
  elem_bboxes.clear();
  spatial_grid = SpatialGrid();          // resets valid = false and frees the CSR

  // Per-node time series. dataCounts must go too: num_loaded_nodes() is derived
  // from it and sizes the device nodal-temperature slice, so leaving it behind
  // would advertise nodes whose series no longer exists.
  dataCounts.clear();
  cache_valid = false;   // the site cache indexes the mesh being dropped
  times = CompressedArray2D<double>();
  temperatures = CompressedArray2D<double>();

  // Nodal-smoothing adjacency and its derived sets
  nbr_offsets.clear();
  nbr_indices.clear();
  nbr_weights.clear();
  nodal_smooth_buffer.clear();
  nodal_adjacency_valid = false;

  // Derived per-step caches
  cached_nodal_temps.clear();
  nodal_cache_valid = false;
  active_node_indices.clear();
  smooth_interp_indices.clear();
  smooth_update_indices.clear();
}

/* ---------------------------------------------------------------------- */

#ifdef SPPARKS_HIGHFIVE
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
#endif // SPPARKS_HIGHFIVE

#if defined(SPPARKS_HIGHFIVE) && defined(SPPARKS_ZLIB)
/* ----------------------------------------------------------------------
   Read the kept rows of a series dataset by inflating its storage chunks on
   all threads. HDF5 inflates one chunk at a time under its global lock (~4.5 s
   for a 99M-site box); here the compressed chunks are fetched serially with
   H5Dread_chunk and inflated in parallel. rows = {file row, local row}, in
   ascending file-row order. Handles only the layout the thermal exporter
   writes (2D, chunked as full rows, deflate as the sole filter, little-endian
   f64/f32); returns false otherwise, or on any read/inflate error, and the
   caller falls back to H5Dread, which overwrites every kept row.
------------------------------------------------------------------------- */

static bool read_series_chunks_parallel(hid_t ds, const std::vector<std::array<size_t, 2>> &rows,
                                        const std::vector<unsigned> &counts,
                                        const std::vector<size_t> &offsets, double *out)
{
  hsize_t dims[2] = {0, 0}, cdims[2] = {0, 0};
  hid_t space = H5Dget_space(ds), dcpl = H5Dget_create_plist(ds), type = H5Dget_type(ds);
  const bool f64 = H5Tequal(type, H5T_IEEE_F64LE) > 0, f32 = H5Tequal(type, H5T_IEEE_F32LE) > 0;
  unsigned flags = 0;
  size_t nelmts = 0;
  bool ok = H5Sget_simple_extent_ndims(space) == 2 && H5Sget_simple_extent_dims(space, dims, nullptr) == 2 &&
            H5Pget_layout(dcpl) == H5D_CHUNKED && H5Pget_chunk(dcpl, 2, cdims) == 2 &&
            cdims[1] == dims[1] && H5Pget_nfilters(dcpl) == 1 &&
            H5Pget_filter2(dcpl, 0, &flags, &nelmts, nullptr, 0, nullptr, nullptr) == H5Z_FILTER_DEFLATE &&
            (f64 || f32) && H5Tequal(H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE) > 0;
  H5Tclose(type); H5Pclose(dcpl); H5Sclose(space);
  if (!ok) return false;

  const size_t R = cdims[0], W = cdims[1], es = f64 ? 8 : 4, chunk_bytes = R * W * es;

  // first row (index into rows) of each storage chunk touched, plus an end marker
  std::vector<size_t> starts;
  for (size_t k = 0; k < rows.size(); k++)
    if (k == 0 || rows[k][0] / R != rows[k - 1][0] / R) starts.push_back(k);
  starts.push_back(rows.size());

  const size_t nchunk = starts.size() - 1, batch = 64;   // ~64 MB of compressed data in flight
  std::vector<std::vector<unsigned char>> comp(batch);
  std::vector<uint32_t> mask(batch);
  for (size_t b0 = 0; b0 < nchunk; b0 += batch) {
    const size_t nb = std::min(batch, nchunk - b0);
    for (size_t b = 0; b < nb; b++) {
      hsize_t off[2] = {rows[starts[b0 + b]][0] / R * R, 0}, nbytes = 0;
      if (H5Dget_chunk_storage_size(ds, off, &nbytes) < 0) return false;
      comp[b].resize(nbytes);
      if (H5Dread_chunk(ds, H5P_DEFAULT, off, &mask[b], comp[b].data()) < 0) return false;
    }
    int bad = 0;
    #pragma omp parallel reduction(|:bad)
    {
      std::vector<unsigned char> raw(chunk_bytes);
      #pragma omp for schedule(dynamic)
      for (size_t b = 0; b < nb; b++) {
        const unsigned char *src = comp[b].data();
        if (!(mask[b] & 1)) {                         // filter applied: inflate
          uLongf len = chunk_bytes;
          if (uncompress(raw.data(), &len, src, comp[b].size()) != Z_OK || len != chunk_bytes) { bad = 1; continue; }
          src = raw.data();
        } else if (comp[b].size() != chunk_bytes) { bad = 1; continue; }
        const size_t row0 = rows[starts[b0 + b]][0] / R * R;
        for (size_t k = starts[b0 + b]; k < starts[b0 + b + 1]; k++) {
          const unsigned char *p = src + (rows[k][0] - row0) * W * es;
          double *dst = out + offsets[rows[k][1]];
          const unsigned n = counts[rows[k][1]];
          if (f64) memcpy(dst, p, n * sizeof(double));
          else for (unsigned j = 0; j < n; j++) { float v; memcpy(&v, p + 4 * j, 4); dst[j] = v; }
        }
      }
    }
    if (bad) return false;
  }
  return true;
}
#endif

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_time_temperature_data(
  HighFive::Group& grp,
  const std::vector<std::array<size_t, 2>>& nodeSlices)
{
#ifndef SPPARKS_HIGHFIVE
  (void)grp;
  (void)nodeSlices;
  error->all(FLERR, "HDF5 temperature sources need a build with HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
  // Streaming implementation: read one chunk at a time from the padded 2D
  // datasets, stripping padding on the fly into pre-allocated CSR arrays.
  // Peak temporary buffer = max_chunk_rows × readCols × 8 bytes.

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
  std::vector<size_t> offsets(nrows + 1, 0);
  std::inclusive_scan(dataCounts.begin(), dataCounts.end(), offsets.begin() + 1,
                      std::plus<size_t>(), size_t{0});

  size_t totalElements = offsets[nrows];

  // Pre-allocate flat output arrays to exact CSR size
  SeriesVector<double> timesFlat(totalElements);
  SeriesVector<double> tempsFlat(totalElements);

  // Reusable chunk buffer — grows to largest chunk, then stays put.
  // Read through select(offset, count), whose memory space has the 2D shape
  // of the selection: a HyperSlab select gets a 1D memory space, and the shape
  // mismatch sends HDF5 down a per-element chunk-mapping path that took ~4x
  // the whole CSR read on ct15 (H5D__chunk_mem_cb dominated the profile).
  std::vector<double> buf;
  auto tempsDset = grp.getDataSet("temperatures");
  size_t dcPos = 0;  // read cursor into dataCounts / offsets

  bool parallel_done = false;
#ifdef SPPARKS_ZLIB
  {
    std::vector<std::array<size_t, 2>> rows;   // {file row, local row} of kept nodes
    size_t dc = 0;
    for (const auto &sl : nodeSlices)
      for (size_t r = sl[0]; r < sl[1]; r++, dc++)
        if (dataCounts[dc] > 0) rows.push_back({r, dc});
    std::sort(rows.begin(), rows.end());
    parallel_done =
      read_series_chunks_parallel(timesDset.getId(), rows, dataCounts, offsets, timesFlat.data()) &&
      read_series_chunks_parallel(tempsDset.getId(), rows, dataCounts, offsets, tempsFlat.data());
  }
#endif

  for (size_t s = 0; !parallel_done && s < nodeSlices.size(); s++) {
    size_t rowStart  = nodeSlices[s][0];
    size_t rowEnd    = nodeSlices[s][1];
    size_t chunkRows = rowEnd - rowStart;
    if (chunkRows == 0) continue;
    if (offsets[dcPos + chunkRows] == offsets[dcPos]) {  // no row kept here
      dcPos += chunkRows;
      continue;
    }

    buf.resize(chunkRows * static_cast<size_t>(readCols));

    // --- Read & strip times ---
    timesDset.select({rowStart, 0}, {chunkRows, static_cast<size_t>(readCols)})
      .read(buf.data());

    for (size_t r = 0; r < chunkRows; r++) {
      unsigned count = dataCounts[dcPos + r];
      std::copy_n(buf.data() + r * readCols, count,
                  timesFlat.data() + offsets[dcPos + r]);
    }

    // --- Read & strip temperatures (reuse buf) ---
    tempsDset.select({rowStart, 0}, {chunkRows, static_cast<size_t>(readCols)})
      .read(buf.data());

    for (size_t r = 0; r < chunkRows; r++) {
      unsigned count = dataCounts[dcPos + r];
      std::copy_n(buf.data() + r * readCols, count,
                  tempsFlat.data() + offsets[dcPos + r]);
    }

    dcPos += chunkRows;
  }

  // Build CompressedArray2D via zero-copy move
  auto offsets2 = offsets;  // copy before first move
  times = CompressedArray2D<double>(std::move(timesFlat), std::move(offsets));
  temperatures = CompressedArray2D<double>(std::move(tempsFlat), std::move(offsets2));
#endif
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
  const std::array<double, 3>& pt) const
{
  const auto& selectedChunks = selected_chunks;
  const auto& chunkBboxes = chunk_bboxes;
  const auto& nodeOffsets = node_offsets;
  const auto& elemOffsets = elem_offsets;
  const auto& elemBboxes = elem_bboxes;
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

    // Which chunk this element belongs to (for node offset mapping)
    if (e >= spatial_grid.elem_to_chunk.size()) return false;
    const unsigned c = spatial_grid.elem_to_chunk[e];

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

  // O(1) spatial-grid cell lookup (the grid is built on every layer load)
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
  }

  // Point not found in any element
  return std::make_pair(
    std::array<unsigned, 4>{invalidId, invalidId, invalidId, invalidId},
    std::array<double, 3>{0.0, 0.0, 0.0}
  );
}

/* ---------------------------------------------------------------------- */

std::vector<std::array<double, 6>> HDF5UnstructuredTemperatureSource::build_elem_bounding_boxes() const
{
  const unsigned nChunks = static_cast<unsigned>(selected_chunks.size());
  const auto& nodeOffsets = node_offsets;
  const auto& elemOffsets = elem_offsets;
  constexpr double maxVal = std::numeric_limits<double>::max();
  constexpr double minVal = std::numeric_limits<double>::lowest();

  // Pre-calculate total element count and reserve to avoid reallocations
  size_t totalElems = 0;
  for (unsigned c = 0; c < nChunks; c++) {
    totalElems += elemOffsets[c+1] - elemOffsets[c];
  }

  std::vector<std::array<double, 6>> result(totalElems);

  // Each element writes only its own slot, so threads give the serial result.
  #pragma omp parallel for schedule(dynamic)
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

      result[e] = {bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]};
    }
  }

  return result;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_spatial_grid()
{
  spatial_grid.valid = false;

  // The only lookups are this rank's owned + ghost sites
  // (build_site_element_cache), so the grid spans their bounding box -- not
  // the loaded chunks, which reach far past the subdomain and made a dense
  // grid over them grow with the cube of 1/cell size (31 GB/rank on ct15 at
  // 5 dx). Cells are sized for ~one element per cell among the elements
  // overlapping that box, independent of the chunk-load pad.
  const int nall = app->nlocal + app->nghost;
  if (elem_bboxes.empty() || nall == 0) return;

  constexpr double maxVal = std::numeric_limits<double>::max();
  constexpr double minVal = std::numeric_limits<double>::lowest();
  std::array<double, 3> domain_min{maxVal, maxVal, maxVal};
  std::array<double, 3> domain_max{minVal, minVal, minVal};
  for (int i = 0; i < nall; i++) {
    for (unsigned d = 0; d < DIM; d++) {
      domain_min[d] = std::fmin(domain_min[d], app->xyz[i][d] * dx);
      domain_max[d] = std::fmax(domain_max[d], app->xyz[i][d] * dx);
    }
  }
  // Pad by a fraction of dx so sites on the box faces land inside it
  for (unsigned d = 0; d < DIM; d++) {
    domain_min[d] -= 0.01 * dx;
    domain_max[d] += 0.01 * dx;
  }

  auto overlaps = [&](const std::array<double, 6>& b) {
    for (unsigned d = 0; d < DIM; d++)
      if (b[d] > domain_max[d] || b[d + 3] < domain_min[d]) return false;
    return true;
  };
  size_t n_overlap = 0;
  for (const auto& b : elem_bboxes) n_overlap += overlaps(b);
  if (n_overlap == 0) return;

  double volume = 1.0;
  for (unsigned d = 0; d < DIM; d++) volume *= domain_max[d] - domain_min[d];
  const double target_cell_size = std::cbrt(volume / n_overlap);

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
    if (!overlaps(elem_bboxes[e])) continue;
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
  spatial_grid.cell_offsets.assign(total_cells + 1, 0);
  std::partial_sum(cell_counts.begin(), cell_counts.end(),
                   spatial_grid.cell_offsets.begin() + 1);

  // Allocate flat element array
  size_t total_refs = spatial_grid.cell_offsets[total_cells];
  spatial_grid.cell_elements.resize(total_refs);

  // Pass 2: Fill the flat array (reuse cell_counts as write positions)
  std::fill(cell_counts.begin(), cell_counts.end(), 0);
  for (unsigned e = 0; e < elem_bboxes.size(); e++) {
    if (!overlaps(elem_bboxes[e])) continue;
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
      load_layer(currentLayer);   // also builds the site element cache
      active_layer = currentLayer;
      nodal_cache_valid = false;
      site_intervals_for_layer = std::numeric_limits<unsigned>::max();
    }
    current_time = time;
  }

  // Build element cache if needed
  if (!cache_valid) {
    build_site_element_cache();
  }

  // Once the element cache exists, compute per-site thermal intervals for
  // the current layer.  These supersede the per-node layer_thermal_intervals
  // in get_next_time_with_temperature and yield tight, site-accurate skip
  // windows instead of node-bound unions.
  if (site_intervals_for_layer != active_layer) {
    compute_site_thermal_intervals(threshold_temp);
  }

  // Precompute nodal temperatures for this timestep -- UNLESS the device does the
  // time interpolation itself (device_time_interp_), in which case the per-node
  // host time-interp is redundant and skipped (its dominant per-step host cost
  // moves onto the device fill).
  if (!device_time_interp_)
    precompute_nodal_temperatures(time);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_site_element_cache() const
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // Build the cache over OWNED + GHOST sites (nall), not just owned. The chunk
  // load already pads the subdomain by grid_cell_size_multiplier cells (>> the
  // one-cell ghost layer), so every ghost site's coordinate lies inside this
  // rank's loaded mesh and finds its containing element. Giving ghosts valid
  // cache entries lets the device fill evaluate ghost temperature POSITIONALLY
  // (like the Rosenthal/Moser device paths) -- ghost T is then correct on-device
  // with no host round-trip and no T ghost comm, because every rank evaluates
  // the same global thermal field at the same physical point. Host consumers
  // (thermal intervals, threshold scan) still loop nlocal, so the extra ghost
  // entries are used only by the device fill.
  int nlocal = app->nlocal;
  int nall = nlocal + app->nghost;

  // Resize cache to match owned+ghost site count
  site_element_cache.resize(nall);

  // For each site (owned + ghost), find its containing element and cache it.
  // The point-location is independent per site and read-only against the shared
  // mesh (find_element_point_is_in is const with only function-local scratch),
  // and each iteration writes its own distinct cache slot -- so this is
  // embarrassingly parallel. dynamic schedule because sites inside the loaded
  // mesh find their element in a few tet tests while sites outside it search the
  // whole grid cell and fail, so the per-site cost is uneven. (Pragma is a no-op
  // when built without OpenMP.)
  #pragma omp parallel for schedule(dynamic, 256)
  for (int i = 0; i < nall; i++) {
    // Get site coordinates
    double x = app->xyz[i][0] * dx;  // Convert lattice to physical
    double y = app->xyz[i][1] * dx;
    double z = app->xyz[i][2] * dx;

    std::array<double, 3> pt{x, y, z};

    // Find element containing this point
    auto result = find_element_point_is_in(pt);

    // Store in cache
    site_element_cache[i].nodeIndices = result.first;
    site_element_cache[i].weights = result.second;
    site_element_cache[i].valid = (result.first[0] != std::numeric_limits<unsigned>::max());
  }

  // Collect unique node indices from valid cached elements (owned + ghost) so
  // precompute_nodal_temperatures fills every node the device fill will read --
  // including ghost sites' tet nodes.
  // A mark-and-compact over a byte array per loaded node, NOT a std::set: the
  // set took 4*nall (~20M here) red-black-tree inserts to end up with at most
  // n_nodes (~233k) distinct entries. Marking is O(1) per touch and the
  // compaction walks node indices in ascending order, which is exactly the
  // order std::set iteration produced -- so active_node_indices is identical.
  const size_t n_nodes = dataCounts.size();
  std::vector<unsigned char> node_used(n_nodes, 0);
  for (int i = 0; i < nall; i++) {
    if (!site_element_cache[i].valid) continue;
    for (int j = 0; j < 4; j++) {
      const unsigned nd = site_element_cache[i].nodeIndices[j];
      if (nd < n_nodes) node_used[nd] = 1;
    }
  }
  active_node_indices.clear();
  for (size_t nd = 0; nd < n_nodes; nd++)
    if (node_used[nd]) active_node_indices.push_back(static_cast<unsigned>(nd));

  // Build BFS-expanded node sets used to keep the per-step interpolation
  // and smoothing tight. Requires adjacency; safe to skip otherwise.
  if (nodal_smooth_enabled && nodal_adjacency_valid) {
    build_smooth_node_sets();
  } else {
    smooth_interp_indices.clear();
    smooth_update_indices.clear();
  }

  cache_valid = true;
  // Bump the generation so a device caller re-uploads its static cache Views.
  ++cache_generation_;

  auto end_time = std::chrono::high_resolution_clock::now();
  double cache_build_time = std::chrono::duration<double>(end_time - start_time).count();

  if (universe->me == 0) {
    fprintf(screen, "  Built element cache for %d sites (incl ghosts) in %.3f s\n", nall, cache_build_time);
    fprintf(screen, "  Active nodes: %zu / %zu (%.1f%% of loaded nodes)\n",
            active_node_indices.size(), dataCounts.size(),
            100.0 * active_node_indices.size() / dataCounts.size());
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::precompute_nodal_temperatures(double time) const
{
  // Skip if already computed for this time
  constexpr double tol = 1e-12;
  if (nodal_cache_valid && std::fabs(time - cached_nodal_time) < tol) {
    return;
  }

  const size_t num_nodes = dataCounts.size();
  cached_nodal_temps.resize(num_nodes);

  // Pick the tightest interpolation set that still satisfies downstream reads:
  //   • smoothing on + BFS sets built → (K+1)-hop expansion of active set
  //   • smoothing off                → active set (tet nodes of cached sites)
  //   • otherwise (cache not yet built) → all loaded nodes
  const std::vector<unsigned>* subset = nullptr;
  if (nodal_smooth_enabled && !smooth_interp_indices.empty()) {
    subset = &smooth_interp_indices;
  } else if (!nodal_smooth_enabled && !active_node_indices.empty()) {
    subset = &active_node_indices;
  }
  const size_t num_to_process = subset ? subset->size() : num_nodes;

  // Each node's time-interpolation is independent: it reads the shared
  // (immutable-per-layer) times/temps CSR and writes its own distinct
  // cached_nodal_temps[nodeIdx] slot, so this loop parallelizes with no
  // reduction -> the result is bit-identical to serial. This runs every step (~33k nodes here) and was
  // the dominant host cost in the per-step "Temperature update" timer. Same
  // proven pattern as build_site_element_cache / compute_site_thermal_intervals.
  // (Pragma is a no-op when built without OpenMP.)
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < num_to_process; i++) {
    const size_t nodeIdx = subset ? (*subset)[i] : i;
    const unsigned count = dataCounts[nodeIdx];
    const auto timeIter = times.row_iterator(nodeIdx);
    const auto tempIter = temperatures.row_iterator(nodeIdx);

    // Empty row (series not kept) or out of the time range -> default_temp
    if (count == 0 || time < *timeIter || time > timeIter[count - 1]) {
      cached_nodal_temps[nodeIdx] = default_temp;
      continue;
    }

    // First sample at or after `time` (bracket [idx-1, idx]).
    unsigned idx = std::distance(timeIter, std::lower_bound(timeIter, timeIter + count, time));
    if (idx == 0) idx = 1;

    // Linear interpolation in time
    cached_nodal_temps[nodeIdx] = tempIter[idx-1] +
                                  (tempIter[idx] - tempIter[idx-1]) /
                                  (timeIter[idx] - timeIter[idx-1]) *
                                  (time - timeIter[idx-1]);
  }

  // Apply mesh-Laplacian smoothing on the FEA node graph before any
  // barycentric query sees the result. Adjacency was built in load_layer.
  if (nodal_smooth_enabled) smooth_nodal_temps();

  cached_nodal_time = time;
  nodal_cache_valid = true;
}

/* ----------------------------------------------------------------------
   Enable anisotropic mesh-Laplacian smoothing. Stores params and clears
   the adjacency-valid flag so the CSR is rebuilt on the next load_layer.
   If a layer is already loaded, build the adjacency immediately so the
   very next precompute_nodal_temperatures call finds it ready.
------------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::enable_nodal_smoothing(
    double sigma_xy, double sigma_z, int passes, double alpha)
{
  nodal_sigma_xy = sigma_xy;
  nodal_sigma_z  = sigma_z;
  nodal_passes   = passes;
  nodal_alpha    = alpha;
  nodal_smooth_enabled = true;
  nodal_adjacency_valid = false;
  smooth_interp_indices.clear();
  smooth_update_indices.clear();

  // Nodal smoothing is rank-local and relies on the chunk-load pad for
  // interior-accurate values near subdomain boundaries. Each pass widens
  // the effective support by one tet-edge; the pad gives ~one tet hop of
  // halo, so at passes>=4 boundary nodes near the subdomain seam start
  // seeing staler values. Warn the user to bump grid_cell_size_multiplier
  // or implement node-halo exchange in that regime.
  if (passes >= 4 && universe->me == 0) {
    fprintf(screen,
      "WARNING: nodal_smooth passes=%d exceeds the one-hop halo implied by "
      "the default chunk-load pad; near-subdomain-boundary nodes may see "
      "stale neighbors. Bump grid_cell_size_multiplier or keep passes <= 3 "
      "for accurate boundary behavior.\n", passes);
  }

  // If elemNode / nodeCoords are already populated from a prior load_layer,
  // build adjacency now so the caller doesn't need to reload the layer.
  // Reload it rather than patching: load_layer kept the series only for the
  // nodes the unsmoothed path reads, and smoothing reads a halo beyond them.
  if (!dataCounts.empty()) {
    load_layer(active_layer);
    nodal_cache_valid = false;
  }
}

/* ----------------------------------------------------------------------
   Build symmetric edge-weighted CSR node adjacency from the loaded tet
   connectivity (elemNode) and node coordinates. Weight per edge:

     w(i,j) = exp( -( (Δx² + Δy²)/(2 σ_xy²)  +  Δz²/(2 σ_z²) ) )

   where Δ is the edge vector expressed in lattice units (physical / dx).
   Duplicate edges arising from tets sharing a face have their weights
   summed, matching build_node_adjacency_aniso in the Python prototype.
------------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_nodal_adjacency_weighted()
{
  auto t_start = std::chrono::high_resolution_clock::now();

  const unsigned n_nodes = static_cast<unsigned>(dataCounts.size());
  const unsigned n_elems = elem_offsets.empty() ? 0u : elem_offsets.back();

  nbr_offsets.assign(n_nodes + 1, 0u);
  nbr_indices.clear();
  nbr_weights.clear();
  nodal_adjacency_valid = false;

  if (n_nodes == 0 || n_elems == 0) {
    nodal_adjacency_valid = true;
    return;
  }

  const double two_sxy2 = 2.0 * nodal_sigma_xy * nodal_sigma_xy;
  const double two_sz2  = 2.0 * nodal_sigma_z  * nodal_sigma_z;
  const double inv_dx   = 1.0 / dx;

  // Collect undirected edges (a < b) with weights. Each tet contributes
  // 6 edges; an interior face-shared edge will appear once per incident
  // tet and the weights will be summed via sort+merge below.
  std::vector<std::tuple<unsigned, unsigned, double>> edges;
  edges.reserve(static_cast<size_t>(n_elems) * 6);

  static constexpr int PAIRS[6][2] = {
      {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

  for (unsigned e = 0; e < n_elems; e++) {
    unsigned n[4];
    for (int k = 0; k < 4; k++) n[k] = elemNode(e, k);

    for (int p = 0; p < 6; p++) {
      unsigned a = n[PAIRS[p][0]];
      unsigned b = n[PAIRS[p][1]];
      if (a == b) continue;
      unsigned i = std::min(a, b);
      unsigned j = std::max(a, b);

      const double dxe = (nodeCoords(j, 0) - nodeCoords(i, 0)) * inv_dx;
      const double dye = (nodeCoords(j, 1) - nodeCoords(i, 1)) * inv_dx;
      const double dze = (nodeCoords(j, 2) - nodeCoords(i, 2)) * inv_dx;
      const double expo = (dxe*dxe + dye*dye) / two_sxy2 + dze*dze / two_sz2;
      edges.emplace_back(i, j, std::exp(-expo));
    }
  }

  std::sort(edges.begin(), edges.end(),
    [](const std::tuple<unsigned,unsigned,double>& a,
       const std::tuple<unsigned,unsigned,double>& b) {
      if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
      return std::get<1>(a) < std::get<1>(b);
    });

  // Merge duplicate edges by summing weights.
  size_t merged = 0;
  for (size_t k = 0; k < edges.size(); ) {
    size_t kk = k + 1;
    double wsum = std::get<2>(edges[k]);
    while (kk < edges.size() &&
           std::get<0>(edges[kk]) == std::get<0>(edges[k]) &&
           std::get<1>(edges[kk]) == std::get<1>(edges[k])) {
      wsum += std::get<2>(edges[kk]);
      ++kk;
    }
    edges[merged] = std::make_tuple(std::get<0>(edges[k]),
                                    std::get<1>(edges[k]),
                                    wsum);
    ++merged;
    k = kk;
  }
  edges.resize(merged);

  // Count degree per node (undirected → add to both endpoints).
  for (const auto& t : edges) {
    ++nbr_offsets[std::get<0>(t) + 1];
    ++nbr_offsets[std::get<1>(t) + 1];
  }
  for (unsigned i = 0; i < n_nodes; i++) nbr_offsets[i + 1] += nbr_offsets[i];

  const size_t total_dir = nbr_offsets[n_nodes];
  nbr_indices.resize(total_dir);
  nbr_weights.resize(total_dir);

  std::vector<unsigned> cursor(n_nodes, 0u);
  for (const auto& t : edges) {
    unsigned i = std::get<0>(t);
    unsigned j = std::get<1>(t);
    double   w = std::get<2>(t);
    size_t pi = nbr_offsets[i] + cursor[i]++;
    size_t pj = nbr_offsets[j] + cursor[j]++;
    nbr_indices[pi] = j; nbr_weights[pi] = w;
    nbr_indices[pj] = i; nbr_weights[pj] = w;
  }

  nodal_smooth_buffer.assign(n_nodes, 0.0);
  nodal_adjacency_valid = true;

  auto t_end = std::chrono::high_resolution_clock::now();
  if (universe->me == 0) {
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    size_t edge_bytes = nbr_indices.size() * (sizeof(unsigned) + sizeof(double))
                      + nbr_offsets.size() * sizeof(unsigned);
    fprintf(screen,
      "  Nodal adjacency: %u nodes, %zu directed edges, avg deg %.1f "
      "(%.1f MB, built in %.3f s)\n",
      n_nodes, total_dir,
      n_nodes ? static_cast<double>(total_dir) / n_nodes : 0.0,
      edge_bytes / (1024.0 * 1024.0), elapsed);
  }
}

/* ----------------------------------------------------------------------
   Build (K+1)-hop and K-hop BFS expansions of active_node_indices over
   the node adjacency. The K+1 set is the read domain during smoothing;
   the K set is the write domain. Reads at any update node go one hop
   outside the update set, which always lands inside the interp set so
   every read is a freshly-interpolated value.
------------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::build_smooth_node_sets() const
{
  smooth_interp_indices.clear();
  smooth_update_indices.clear();

  if (!nodal_smooth_enabled || !nodal_adjacency_valid) return;
  if (active_node_indices.empty()) return;
  if (nodal_passes <= 0) return;

  auto t_start = std::chrono::high_resolution_clock::now();

  const unsigned n_nodes = static_cast<unsigned>(dataCounts.size());
  constexpr uint8_t UNSEEN = 255;
  std::vector<uint8_t> depth(n_nodes, UNSEEN);

  std::vector<unsigned> frontier;
  frontier.reserve(active_node_indices.size());
  for (unsigned v : active_node_indices) {
    if (v < n_nodes && depth[v] == UNSEEN) {
      depth[v] = 0;
      frontier.push_back(v);
    }
  }

  const int max_depth = nodal_passes + 1;   // read ring
  std::vector<unsigned> next;
  next.reserve(frontier.size());
  for (int d = 1; d <= max_depth; d++) {
    next.clear();
    const uint8_t dd = static_cast<uint8_t>(d);
    for (unsigned u : frontier) {
      const unsigned beg = nbr_offsets[u];
      const unsigned end = nbr_offsets[u + 1];
      for (unsigned k = beg; k < end; k++) {
        const unsigned v = nbr_indices[k];
        if (depth[v] == UNSEEN) {
          depth[v] = dd;
          next.push_back(v);
        }
      }
    }
    frontier.swap(next);
    if (frontier.empty()) break;
  }

  const uint8_t update_max = static_cast<uint8_t>(nodal_passes);
  for (unsigned i = 0; i < n_nodes; i++) {
    if (depth[i] == UNSEEN) continue;
    smooth_interp_indices.push_back(i);
    if (depth[i] <= update_max) smooth_update_indices.push_back(i);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  if (universe->me == 0) {
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(screen,
      "  Nodal BFS expansion: active=%zu, update(K=%d)=%zu, "
      "interp(K+1=%d)=%zu / %u loaded (%.1f%%); built in %.3f s\n",
      active_node_indices.size(),
      nodal_passes, smooth_update_indices.size(),
      max_depth, smooth_interp_indices.size(),
      n_nodes,
      n_nodes ? 100.0 * smooth_interp_indices.size() / n_nodes : 0.0,
      elapsed);
  }
}

/* ----------------------------------------------------------------------
   K-pass anisotropic Laplacian smoothing on cached_nodal_temps.

     T_new[i] = (1 - alpha) T_old[i]
              + alpha * (T_old[i] + Σ_j w_ij T_old[j]) / (1 + Σ_j w_ij)

   Double-buffered: writes into nodal_smooth_buffer, then swaps. Rank-local;
   relies on the grid_cell_size_multiplier pad in load_layer for interior-
   accurate values at subdomain boundaries (adequate for passes <= 3).

   When smooth_update_indices is populated, the pass only writes entries in
   the K-hop set and starts by copying cached_nodal_temps so that non-updated
   entries carry forward unchanged across the swap. Reads at update nodes
   reach the (K+1)-hop interp set, which was interpolated this step.
------------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::smooth_nodal_temps() const
{
  if (!nodal_adjacency_valid) return;
  if (nodal_passes <= 0) return;

  const size_t n_nodes = cached_nodal_temps.size();
  if (n_nodes == 0) return;
  // Nothing to smooth if no node has been claimed as "active" on this rank
  // (e.g. a rank whose local sites all fell outside the thermal mesh). The
  // smoothed values would never be read back through the site cache anyway.
  if (smooth_update_indices.empty()) return;
  if (nodal_smooth_buffer.size() != n_nodes) nodal_smooth_buffer.assign(n_nodes, 0.0);

  const double alpha  = nodal_alpha;
  const double one_ma = 1.0 - alpha;

  for (int pass = 0; pass < nodal_passes; pass++) {
    // Propagate non-updated entries forward unchanged through the swap.
    nodal_smooth_buffer = cached_nodal_temps;
    for (unsigned i : smooth_update_indices) {
      const unsigned beg = nbr_offsets[i];
      const unsigned end = nbr_offsets[i + 1];
      double wsum = 1.0;
      double tsum = cached_nodal_temps[i];
      for (unsigned k = beg; k < end; k++) {
        const double w = nbr_weights[k];
        tsum += w * cached_nodal_temps[nbr_indices[k]];
        wsum += w;
      }
      nodal_smooth_buffer[i] = one_ma * cached_nodal_temps[i]
                             + alpha  * (tsum / wsum);
    }
    cached_nodal_temps.swap(nodal_smooth_buffer);
  }
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_next_time_with_temperature(double current_time, double threshold_temp)
{
  if (layerTimes.empty()) {
    return current_time;
  }

  // Find current layer
  unsigned current_layer = get_active_layer(current_time);

  // Prefer per-site intervals for the currently loaded layer when available.
  // They are tight (computed from the true barycentric T_site) rather than
  // the looser union bounds produced by the per-node scan.
  if (site_intervals_for_layer == current_layer
      && !current_layer_site_intervals.empty()) {
    for (const auto& interval : current_layer_site_intervals) {
      if (current_time >= interval.start_time && current_time <= interval.end_time) {
        return current_time;
      }
      if (interval.start_time > current_time) {
        return interval.start_time;
      }
    }
  } else {
    // Fall back to the node-level intervals (e.g., before the element
    // cache has been built for this layer).
    if (current_layer >= layer_thermal_intervals.size()) {
      return current_time;
    }
    for (const auto& interval : layer_thermal_intervals[current_layer]) {
      if (current_time >= interval.start_time && current_time <= interval.end_time) {
        return current_time;
      }
      if (interval.start_time > current_time) {
        return interval.start_time;
      }
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

  // One pass over every loaded sample, so it is parallel over nodes. Each thread
  // collects privately; merge_overlapping_intervals sorts, so the result does
  // not depend on the order the buffers are joined.
  #pragma omp parallel
  {
  std::vector<ThermalInterval> local;
  #pragma omp for schedule(dynamic, 1024) nowait
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
      local.emplace_back(first_hot_time, last_hot_time);
    }
  }
  #pragma omp critical
  intervals.insert(intervals.end(), local.begin(), local.end());
  }  // omp parallel

  // Merge overlapping intervals and store
  layer_thermal_intervals[layerIdx] = merge_overlapping_intervals(intervals);

  if (universe->me == 0 && !layer_thermal_intervals[layerIdx].empty()) {
    fprintf(screen, "  Layer %u: Found %zu thermal intervals\n",
            layerIdx, layer_thermal_intervals[layerIdx].size());
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::compute_site_thermal_intervals(double threshold)
{
  current_layer_site_intervals.clear();
  site_intervals_for_layer = std::numeric_limits<unsigned>::max();

  if (!cache_valid || !app) return;

  auto t_start = std::chrono::high_resolution_clock::now();

  std::vector<ThermalInterval> raw;
  size_t nraw = 0;

  const int nlocal = app->nlocal;

  // Per-site interval extraction is independent across sites and reads only
  // shared (immutable here) mesh/cache data. merge_overlapping_intervals()
  // below sorts, so the accumulation order does not affect the result -- each
  // thread fills a private buffer and we concatenate. (When built without
  // OpenMP the pragmas are no-ops, raw_local is a single buffer, and this runs
  // exactly as the previous serial loop did.)
  #pragma omp parallel
  {
  std::vector<ThermalInterval> raw_local;
  raw_local.reserve(256);
  #pragma omp for schedule(dynamic, 512) nowait
  for (int i = 0; i < nlocal; i++) {
    const ElementCache& cache = site_element_cache[i];
    if (!cache.valid) continue;

    const unsigned n0 = cache.nodeIndices[0];
    const unsigned n1 = cache.nodeIndices[1];
    const unsigned n2 = cache.nodeIndices[2];
    const unsigned n3 = cache.nodeIndices[3];
    const unsigned c0 = dataCounts[n0];
    const unsigned c1 = dataCounts[n1];
    const unsigned c2 = dataCounts[n2];
    const unsigned c3 = dataCounts[n3];
    if (c0 == 0 || c1 == 0 || c2 == 0 || c3 == 0) continue;

    const auto t0 = times.row_iterator(n0);
    const auto t1 = times.row_iterator(n1);
    const auto t2 = times.row_iterator(n2);
    const auto t3 = times.row_iterator(n3);
    const auto T0 = temperatures.row_iterator(n0);
    const auto T1 = temperatures.row_iterator(n1);
    const auto T2 = temperatures.row_iterator(n2);
    const auto T3 = temperatures.row_iterator(n3);

    const double ww1 = cache.weights[0];
    const double ww2 = cache.weights[1];
    const double ww3 = cache.weights[2];
    const double ww0 = 1.0 - ww1 - ww2 - ww3;

    // The cached barycentric formula used by get_temperature_at_site_fast()
    // is T_site = T0 + w1*(T1-T0) + w2*(T2-T0) + w3*(T3-T0).  Expanded:
    //   T_site = (1 - w1 - w2 - w3)*T0 + w1*T1 + w2*T2 + w3*T3
    //          = ww0*T0 + ww1*T1 + ww2*T2 + ww3*T3
    // so the weights are a partition of unity and T_site is piecewise
    // linear between the union of the four nodes' sample times.

    // Per-node cursors: cursor points at the NEXT sample whose time is
    // strictly greater than the current critical time.  Between calls we
    // maintain T_j(t_crit) via linear interpolation at the last critical
    // time.
    unsigned i0 = 1, i1 = 1, i2 = 1, i3 = 1;

    // Start at the max of the four nodes' first sample times so every
    // node has data at t_crit; stop at the min of the four nodes' last
    // sample times for the same reason.
    double t_crit = std::max({t0[0], t1[0], t2[0], t3[0]});
    const double t_end = std::min({t0[c0 - 1], t1[c1 - 1],
                                   t2[c2 - 1], t3[c3 - 1]});
    if (t_end <= t_crit) continue;

    auto advance_cursor = [](unsigned& idx, unsigned count, auto ti, double t) {
      while (idx < count - 1 && ti[idx] <= t) ++idx;
    };
    advance_cursor(i0, c0, t0, t_crit);
    advance_cursor(i1, c1, t1, t_crit);
    advance_cursor(i2, c2, t2, t_crit);
    advance_cursor(i3, c3, t3, t_crit);

    auto interp = [](auto ti, auto Ti, unsigned idx, double t) -> double {
      // idx is such that ti[idx-1] <= t <= ti[idx]
      const double t_lo = ti[idx - 1];
      const double t_hi = ti[idx];
      const double frac = (t - t_lo) / (t_hi - t_lo);
      return Ti[idx - 1] + frac * (Ti[idx] - Ti[idx - 1]);
    };

    double T_prev = ww0 * interp(t0, T0, i0, t_crit)
                  + ww1 * interp(t1, T1, i1, t_crit)
                  + ww2 * interp(t2, T2, i2, t_crit)
                  + ww3 * interp(t3, T3, i3, t_crit);
    double t_prev = t_crit;
    bool in_hot = (T_prev > threshold);
    double hot_start = in_hot ? t_prev : 0.0;

    // Cold skip (exact). With non-negative weights T_site is a convex
    // combination of the four node temperatures, so while every node stays at
    // or below thr_cold (a margin far above rounding) the site cannot reach
    // the threshold and no crossing can occur. Jump to L, the last critical
    // time before the first node warms, and rebuild the state the full walk
    // has there: T at L from the lower-bound cursors, then cursors past L.
    // The crossings found afterwards are therefore bit-identical.
    const bool convex = ww0 >= 0.0 && ww1 >= 0.0 && ww2 >= 0.0 && ww3 >= 0.0;
    const double thr_cold = threshold - 1e-6 * (std::fabs(threshold) + 1.0);
    unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;   // first sample > thr_cold at/after cursor-1
    auto cold_until = [thr_cold](unsigned& s, unsigned idx, unsigned count, auto ti, auto Ti) {
      if (s < idx - 1) s = idx - 1;
      while (s < count && Ti[s] <= thr_cold) ++s;
      return s < idx ? -std::numeric_limits<double>::infinity() : ti[s - 1];
    };
    auto lower = [](unsigned idx, unsigned count, auto ti, double t) {
      while (idx < count - 1 && ti[idx] < t) ++idx;
      return idx;
    };

    while (t_crit < t_end) {
      if (convex && !in_hot) {
        const double L = std::min({t_end, cold_until(s0, i0, c0, t0, T0), cold_until(s1, i1, c1, t1, T1),
                                   cold_until(s2, i2, c2, t2, T2), cold_until(s3, i3, c3, t3, T3)});
        if (L > t_crit) {
          i0 = lower(i0, c0, t0, L); i1 = lower(i1, c1, t1, L);
          i2 = lower(i2, c2, t2, L); i3 = lower(i3, c3, t3, L);
          T_prev = ww0 * interp(t0, T0, i0, L)
                 + ww1 * interp(t1, T1, i1, L)
                 + ww2 * interp(t2, T2, i2, L)
                 + ww3 * interp(t3, T3, i3, L);
          if (i0 < c0 && t0[i0] == L) ++i0;
          if (i1 < c1 && t1[i1] == L) ++i1;
          if (i2 < c2 && t2[i2] == L) ++i2;
          if (i3 < c3 && t3[i3] == L) ++i3;
          t_prev = t_crit = L;
          continue;
        }
      }

      // Next critical time = smallest of the four nodes' next sample times.
      double t_next = t_end;
      if (i0 < c0 && t0[i0] < t_next) t_next = t0[i0];
      if (i1 < c1 && t1[i1] < t_next) t_next = t1[i1];
      if (i2 < c2 && t2[i2] < t_next) t_next = t2[i2];
      if (i3 < c3 && t3[i3] < t_next) t_next = t3[i3];

      const double T_cur = ww0 * interp(t0, T0, i0, t_next)
                         + ww1 * interp(t1, T1, i1, t_next)
                         + ww2 * interp(t2, T2, i2, t_next)
                         + ww3 * interp(t3, T3, i3, t_next);

      const bool cur_hot = (T_cur > threshold);
      if (!in_hot && cur_hot) {
        // Threshold crossing in [t_prev, t_next]; solve for exact crossing.
        const double denom = T_cur - T_prev;
        const double alpha = (threshold - T_prev) / denom;
        hot_start = t_prev + alpha * (t_next - t_prev);
        in_hot = true;
      } else if (in_hot && !cur_hot) {
        const double denom = T_cur - T_prev;
        const double alpha = (threshold - T_prev) / denom;
        const double hot_end = t_prev + alpha * (t_next - t_prev);
        raw_local.emplace_back(hot_start, hot_end);
        in_hot = false;
      }

      // Advance cursors whose sample time equals t_next.
      if (i0 < c0 && t0[i0] == t_next) ++i0;
      if (i1 < c1 && t1[i1] == t_next) ++i1;
      if (i2 < c2 && t2[i2] == t_next) ++i2;
      if (i3 < c3 && t3[i3] == t_next) ++i3;
      t_prev = t_next;
      T_prev = T_cur;
      t_crit = t_next;
    }

    if (in_hot) {
      raw_local.emplace_back(hot_start, t_end);
    }
  }
  // Merge per thread first: the union is associative, so this is the same
  // result without one serial sort of every raw interval (~66M at 99M sites).
  const size_t nraw_local = raw_local.size();
  raw_local = merge_overlapping_intervals(raw_local);
  #pragma omp critical
  {
  nraw += nraw_local;
  raw.insert(raw.end(), raw_local.begin(), raw_local.end());
  }
  }  // omp parallel

  current_layer_site_intervals = merge_overlapping_intervals(raw);
  site_intervals_for_layer = active_layer;

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // Aggregate per-rank counts so the diagnostic reflects the whole
  // communicator, not just rank 0's subdomain.  Build time is reduced with
  // MAX so the reported elapsed is the slowest rank (the actual bottleneck).
  bigint local_site_count = current_layer_site_intervals.size();
  bigint local_raw_count  = nraw;
  bigint local_node_count = (active_layer < layer_thermal_intervals.size())
    ? layer_thermal_intervals[active_layer].size() : 0;
  bigint total_site_count = 0, total_raw_count = 0, total_node_count = 0;
  double max_elapsed = 0.0;
  MPI_Reduce(&local_site_count, &total_site_count, 1, MPI_SPK_BIGINT, MPI_SUM, 0, world);
  MPI_Reduce(&local_raw_count,  &total_raw_count,  1, MPI_SPK_BIGINT, MPI_SUM, 0, world);
  MPI_Reduce(&local_node_count, &total_node_count, 1, MPI_SPK_BIGINT, MPI_SUM, 0, world);
  MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, world);

  if (universe->me == 0) {
    fprintf(screen, "  Layer %u: Found " BIGINT_FORMAT " site-level thermal intervals "
            "(vs " BIGINT_FORMAT " node-level, " BIGINT_FORMAT " raw sites) across %d ranks in %.3f s\n",
            active_layer,
            total_site_count,
            total_node_count,
            total_raw_count,
            universe->nprocs,
            max_elapsed);
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
  // Use GLOBAL domain bounds — not subdomain — so thermal intervals are not
  // contaminated by hot nodes outside the SPPARKS simulation box that sit
  // inside the per-rank chunk-loading padding halo.  Such padding-only nodes
  // never affect any SPPARKS site but, if included here, would make
  // fast-forward conservatively refuse to skip quiet periods.
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