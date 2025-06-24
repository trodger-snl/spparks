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
#include "lattice.h"
#include "error.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cmath>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) : 
  TemperatureSource(spk)
{
  file_id = H5I_INVALID_HID;
  active_layer = std::numeric_limits<unsigned>::max();
  current_time = std::numeric_limits<double>::lowest();
  dx = 0.0;
  x0.fill(0.0);
  size.fill(0);
}

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::~HDF5UnstructuredTemperatureSource()
{
  cleanup();
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  parse_arguments(args);
  open_hdf5_file();
  read_layer_times();
  setup_grid_bbox();
  
  // Pre-allocate interpolation cache
  unsigned total_sites = size[0] * size[1] * size[2];
  interpolation_cache.resize(total_sites);
  
  source_initialized = true;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::parse_arguments(const std::vector<std::string> &args)
{
  if (args.size() < 2) {
    error->all(FLERR,"HDF5 unstructured temperature source requires: filename dx");
  }
  
  filename = args[0];
  dx = std::stod(args[1]);
  
  // Extract domain bounds from existing domain configuration
  if (domain->lattice == NULL) {
    error->all(FLERR,"HDF5 unstructured temperature source requires lattice to be defined");
  }
  
  // Use subdomain bounds for this processor instead of global bounds
  x0[0] = domain->subxlo;
  x0[1] = domain->subylo;
  x0[2] = domain->subzlo;
  
  // Calculate grid dimensions from processor subdomain bounds
  size[0] = static_cast<unsigned>(std::round((domain->subxhi - domain->subxlo) / dx)) + 1;
  size[1] = static_cast<unsigned>(std::round((domain->subyhi - domain->subylo) / dx)) + 1;
  size[2] = static_cast<unsigned>(std::round((domain->subzhi - domain->subzlo) / dx)) + 1;
  
  if (dx <= 0.0) {
    error->all(FLERR,"Thermal data grid spacing dx must be positive");
  }
  
  if (size[0] == 0 || size[1] == 0 || size[2] == 0) {
    error->all(FLERR,"Calculated grid dimensions must be positive");
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::open_hdf5_file()
{
  // Create parallel file access property list
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
  
#ifdef SPPARKS_MPI
  // Set up parallel HDF5 access if MPI is available
  H5Pset_fapl_mpio(fapl_id, world, MPI_INFO_NULL);
#endif
  
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl_id);
  H5Pclose(fapl_id);
  
  if (file_id < 0) {
    std::string msg = "Cannot open HDF5 file: " + filename;
    error->all(FLERR, msg.c_str());
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_layer_times()
{
  hid_t dataset_id = H5Dopen2(file_id, "layerTimes", H5P_DEFAULT);
  if (dataset_id < 0) {
    error->all(FLERR, "Cannot open layerTimes dataset in HDF5 file");
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  layer_times.resize(dims[0]);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                         plist_id, layer_times.data());
  
  H5Pclose(plist_id);
  H5Sclose(space_id);
  H5Dclose(dataset_id);
  
  if (status < 0) {
    error->all(FLERR, "Failed to read layerTimes from HDF5 file");
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::setup_grid_bbox()
{
  // Grid bbox represents this processor's subdomain bounds
  grid_bbox = {
    x0[0], x0[1], x0[2],
    x0[0] + dx * (size[0] - 1),
    x0[1] + dx * (size[1] - 1), 
    x0[2] + dx * (size[2] - 1)
  };
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_temperature_at_xyz_and_time(double x, double y, double z, double time)
{
  check_initialization();
  
  constexpr double tol = 5.0 * std::numeric_limits<double>::epsilon();
  if (std::abs(time - current_time) / (std::abs(time) + tol) > tol) {
    unsigned new_layer = get_active_layer(time);
    if (new_layer != active_layer) {
      load_layer(new_layer);
      active_layer = new_layer;
    }
    current_time = time;
  }
  
  // Convert physical coordinates to local grid indices within this processor's subdomain
  int i = static_cast<int>(std::round((x - x0[0]) / dx));
  int j = static_cast<int>(std::round((y - x0[1]) / dx));
  int k = static_cast<int>(std::round((z - x0[2]) / dx));
  
  // Check bounds against local subdomain size
  if (i < 0 || i >= static_cast<int>(size[0]) ||
      j < 0 || j >= static_cast<int>(size[1]) ||
      k < 0 || k >= static_cast<int>(size[2])) {
    return ambient_temperature;
  }
  
  unsigned flat_idx = i * size[1] * size[2] + j * size[2] + k;
  
  // Check if we have valid cached interpolation data for this site
  const auto &cached_data = interpolation_cache[flat_idx];
  if (!cached_data.valid) {
    return ambient_temperature;
  }
  
  // Use cached interpolation data for fast temperature lookup
  double temperature = 0.0;
  for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
    unsigned node_idx = cached_data.node_ids[n];
    double nodal_temp = interpolate_nodal_temperature(node_idx, time);
    temperature += cached_data.weights[n] * nodal_temp;
  }
  
  return temperature;
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::interpolate_nodal_temperature(unsigned node_idx, double time) const
{
  const auto time_iter = times.row_iterator(node_idx);
  const auto temp_iter = temperatures.row_iterator(node_idx);
  unsigned count = data_counts[node_idx];
  
  if (time < *time_iter) {
    std::ostringstream msg;
    msg << "Time " << time << " is before first nodal time " << *time_iter;
    error->one(FLERR, msg.str().c_str());
  }
  
  if (time > time_iter[count - 1]) {
    std::ostringstream msg;
    msg << "Time " << time << " is after last nodal time " << time_iter[count - 1];
    error->one(FLERR, msg.str().c_str());
  }
  
  // Find the time interval
  auto lower = std::lower_bound(time_iter, time_iter + count, time);
  auto idx = std::distance(time_iter, lower);
  
  if (idx == 0) {
    return temp_iter[0];
  }
  
  // Linear interpolation between time points
  double t0 = time_iter[idx - 1];
  double t1 = time_iter[idx];
  double T0 = temp_iter[idx - 1];
  double T1 = temp_iter[idx];
  
  return T0 + (T1 - T0) / (t1 - t0) * (time - t0);
}

/* ---------------------------------------------------------------------- */

unsigned HDF5UnstructuredTemperatureSource::get_active_layer(double time) const
{
  if (time < layer_times.front()) {
    std::ostringstream msg;
    msg << "Time " << time << " is before first layer time " << layer_times.front();
    error->one(FLERR, msg.str().c_str());
  }
  
  if (time > layer_times.back()) {
    std::ostringstream msg;
    msg << "Time " << time << " is after last layer time " << layer_times.back();
    error->one(FLERR, msg.str().c_str());
  }
  
  auto lower = std::lower_bound(layer_times.begin(), layer_times.end(), time);
  auto idx = std::distance(layer_times.begin(), lower);
  return idx == 0 ? 0 : idx - 1;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::load_layer(unsigned layer_idx)
{
  std::string group_name = std::to_string(layer_idx);
  hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
  if (group_id < 0) {
    std::string msg = "Cannot open layer group: " + group_name;
    error->all(FLERR, msg.c_str());
  }
  
  try {
    // Clear previous data
    data_counts.clear();
    times.clear();
    temperatures.clear();
    elem_node.clear();
    node_coords.clear();
    chunk_bboxes.clear();
    node_offsets.clear();
    elem_offsets.clear();
    elem_bboxes.clear();
    
    // Identify overlapping chunks
    std::vector<unsigned> overlapping_chunks;
    read_chunk_info(group_id, overlapping_chunks);
    
    // Read mesh data for overlapping chunks
    read_mesh_data(group_id, overlapping_chunks);
    
    // Read thermal data
    unsigned max_data_count = 0;
    for (auto count : data_counts) {
      max_data_count = std::max(max_data_count, count);
    }
    read_thermal_data(group_id, max_data_count);
    
    // Compute interpolation weights for all grid points
    compute_interpolation_weights();
    
  } catch (...) {
    H5Gclose(group_id);
    throw;
  }
  
  H5Gclose(group_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_chunk_info(hid_t group_id, std::vector<unsigned> &overlapping_chunks)
{
  // Read bounding boxes
  std::vector<std::vector<double>> bboxes;
  hid_t bbox_dataset = H5Dopen2(group_id, "boundingBoxes", H5P_DEFAULT);
  if (bbox_dataset < 0) {
    error->all(FLERR, "Cannot open boundingBoxes dataset");
  }
  
  hid_t bbox_space = H5Dget_space(bbox_dataset);
  hsize_t bbox_dims[2];
  H5Sget_simple_extent_dims(bbox_space, bbox_dims, NULL);
  
  std::vector<double> bbox_data(bbox_dims[0] * bbox_dims[1]);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  herr_t status = H5Dread(bbox_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id, bbox_data.data());
  H5Pclose(plist_id);
  H5Sclose(bbox_space);
  H5Dclose(bbox_dataset);
  
  if (status < 0) {
    error->all(FLERR, "Failed to read boundingBoxes");
  }
  
  // Convert to vector of vectors
  bboxes.resize(bbox_dims[0]);
  for (size_t i = 0; i < bbox_dims[0]; i++) {
    bboxes[i].resize(bbox_dims[1]);
    for (size_t j = 0; j < bbox_dims[1]; j++) {
      bboxes[i][j] = bbox_data[i * bbox_dims[1] + j];
    }
  }
  
  // Find overlapping chunks using expanded bounding box for better boundary handling
  overlapping_chunks.clear();
  std::vector<double> expanded_bbox = get_expanded_bbox(1.1); // 10% expansion
  for (size_t c = 0; c < bboxes.size(); c++) {
    if (do_boxes_overlap(bboxes[c], expanded_bbox)) {
      overlapping_chunks.push_back(c);
    }
  }
  
  chunk_bboxes = std::move(bboxes);
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::do_boxes_overlap(const std::vector<double> &b1, const std::vector<double> &b2) const
{
  return b1[0] < b2[3] && b2[0] < b1[3] && 
         b1[1] < b2[4] && b2[1] < b1[4] && 
         b1[2] < b2[5] && b2[2] < b1[5];
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::update_temperatures(double dt, double simulation_time)
{
  // This implementation doesn't need to update temperatures actively
  // Temperature data is loaded on-demand when layer changes
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::needs_data_refresh(double simulation_time)
{
  if (!source_initialized) return false;
  
  try {
    unsigned new_layer = get_active_layer(simulation_time);
    return new_layer != active_layer;
  } catch (...) {
    return false;
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::cleanup()
{
  if (file_id >= 0) {
    H5Fclose(file_id);
    file_id = H5I_INVALID_HID;
  }
  
  // Clear all data
  layer_times.clear();
  data_counts.clear();
  times.clear();
  temperatures.clear();
  interpolation_cache.clear();
  elem_node.clear();
  node_coords.clear();
  chunk_bboxes.clear();
  node_offsets.clear();
  elem_offsets.clear();
  elem_bboxes.clear();
  
  active_layer = std::numeric_limits<unsigned>::max();
  current_time = std::numeric_limits<double>::lowest();
  source_initialized = false;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_mesh_data(hid_t group_id, const std::vector<unsigned> &overlapping_chunks)
{
  // Read element and node pointers
  std::vector<unsigned> elem_ptr, node_ptr;
  read_hdf5_dataset_1d(group_id, "elemPtrs", elem_ptr);
  read_hdf5_dataset_1d(group_id, "nodePtrs", node_ptr);
  
  // Build slices for overlapping chunks
  std::vector<std::array<size_t, 2>> elem_slices, node_slices;
  node_offsets.clear();
  elem_offsets.clear();
  node_offsets.push_back(0);
  elem_offsets.push_back(0);
  
  for (unsigned c : overlapping_chunks) {
    elem_slices.push_back({elem_ptr[c], elem_ptr[c + 1]});
    node_slices.push_back({node_ptr[c], node_ptr[c + 1]});
    node_offsets.push_back(node_offsets.back() + node_ptr[c + 1] - node_ptr[c]);
    elem_offsets.push_back(elem_offsets.back() + elem_ptr[c + 1] - elem_ptr[c]);
  }
  
  // Read element-to-node connectivity
  std::vector<unsigned> elem_node_data;
  read_hdf5_dataset_2d(group_id, "elementToNode", elem_slices, 
                       {0, NODES_PER_ELEM}, elem_node_data);
  elem_node = array2D<unsigned>(NODES_PER_ELEM, std::move(elem_node_data));
  
  // Read node coordinates
  std::vector<double> node_coords_data;
  read_hdf5_dataset_2d(group_id, "nodeCoords", node_slices, 
                       {0, DIM}, node_coords_data);
  node_coords = array2D<double>(DIM, std::move(node_coords_data));
  
  // Read data counts for thermal data
  read_hdf5_hyperslab_1d(group_id, "dataCounts", node_slices, data_counts);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_thermal_data(hid_t group_id, unsigned max_data_count)
{
  // Build node slices for thermal data reading
  std::vector<std::array<size_t, 2>> node_slices;
  for (size_t i = 0; i < node_offsets.size() - 1; i++) {
    node_slices.push_back({node_offsets[i], node_offsets[i + 1]});
  }
  
  // Read times data
  std::vector<double> times_data;
  read_hdf5_dataset_2d(group_id, "times", node_slices, 
                       {0, max_data_count}, times_data);
  times = array2D<double>(max_data_count, std::move(times_data));
  
  // Read temperatures data
  std::vector<double> temp_data;
  read_hdf5_dataset_2d(group_id, "temperatures", node_slices, 
                       {0, max_data_count}, temp_data);
  temperatures = array2D<double>(max_data_count, std::move(temp_data));
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::compute_interpolation_weights()
{
  // Build element bounding boxes
  std::vector<unsigned> overlapping_chunks;
  for (size_t i = 0; i < chunk_bboxes.size(); i++) {
    if (do_boxes_overlap(chunk_bboxes[i], grid_bbox)) {
      overlapping_chunks.push_back(i);
    }
  }
  
  elem_bboxes = build_elem_bounding_boxes(overlapping_chunks);
  
  // Pre-compute and cache interpolation data for each grid point
  for (unsigned i = 0; i < size[0]; i++) {
    for (unsigned j = 0; j < size[1]; j++) {
      for (unsigned k = 0; k < size[2]; k++) {
        unsigned flat_idx = i * size[1] * size[2] + j * size[2] + k;
        std::array<double, 3> pt{x0[0] + i * dx, x0[1] + j * dx, x0[2] + k * dx};
        
        // Find element and compute barycentric weights
        interpolation_cache[flat_idx] = find_element_and_compute_weights(overlapping_chunks, pt);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

std::vector<std::vector<double>> HDF5UnstructuredTemperatureSource::build_elem_bounding_boxes(
  const std::vector<unsigned> &overlapping_chunks) const
{
  constexpr double max_val = std::numeric_limits<double>::max();
  constexpr double min_val = std::numeric_limits<double>::lowest();
  std::vector<std::vector<double>> result;
  
  for (size_t c_idx = 0; c_idx < overlapping_chunks.size(); c_idx++) {
    unsigned c = overlapping_chunks[c_idx];
    for (unsigned e = elem_offsets[c_idx]; e < elem_offsets[c_idx + 1]; e++) {
      std::array<double, 3> bmin{max_val, max_val, max_val};
      std::array<double, 3> bmax{min_val, min_val, min_val};
      
      for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
        unsigned node_idx = elem_node(e, n) + node_offsets[c_idx];
        for (unsigned d = 0; d < DIM; d++) {
          double coord_val = node_coords(node_idx, d);
          bmin[d] = std::fmin(bmin[d], coord_val);
          bmax[d] = std::fmax(bmax[d], coord_val);
        }
      }
      result.push_back({bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]});
    }
  }
  return result;
}

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::InterpolationData 
HDF5UnstructuredTemperatureSource::find_element_and_compute_weights(
  const std::vector<unsigned> &overlapping_chunks,
  const std::array<double, 3> &pt) const
{
  InterpolationData result;
  
  constexpr double tol = 1e-14;
  
  // Find possible chunks that contain the point
  std::vector<unsigned> possible_chunks;
  for (size_t c_idx = 0; c_idx < overlapping_chunks.size(); c_idx++) {
    unsigned c = overlapping_chunks[c_idx];
    if (point_in_bbox(pt, chunk_bboxes[c])) {
      possible_chunks.push_back(c_idx);
    }
  }
  
  // Search through elements in possible chunks
  for (unsigned c_idx : possible_chunks) {
    for (unsigned e = elem_offsets[c_idx]; e < elem_offsets[c_idx + 1]; e++) {
      if (point_in_bbox(pt, elem_bboxes[e])) {
        // Get element node coordinates
        std::vector<std::vector<double>> tet_coords(4, std::vector<double>(DIM));
        for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
          result.node_ids[n] = elem_node(e, n) + node_offsets[c_idx];
          for (unsigned d = 0; d < DIM; d++) {
            tet_coords[n][d] = node_coords(result.node_ids[n], d);
          }
        }
        
        // Compute parametric coordinates
        std::array<double, 3> par_coords = get_parametric_coordinates_of_point(tet_coords, pt);
        
        // Check if point is inside tetrahedron
        if (par_coords[0] > -tol && par_coords[1] > -tol && par_coords[2] > -tol &&
            1.0 - par_coords[0] - par_coords[1] - par_coords[2] > -tol) {
          
          // Convert parametric coordinates to barycentric weights
          // For tetrahedron: w0 = 1 - w1 - w2 - w3, where w1,w2,w3 are parametric coords
          result.weights[0] = 1.0 - par_coords[0] - par_coords[1] - par_coords[2]; // w0
          result.weights[1] = par_coords[0]; // w1
          result.weights[2] = par_coords[1]; // w2
          result.weights[3] = par_coords[2]; // w3
          result.valid = true;
          
          return result;
        }
      }
    }
  }
  
  // Point not found in any element - result already initialized as invalid
  return result;
}

/* ---------------------------------------------------------------------- */

std::array<double, 3> HDF5UnstructuredTemperatureSource::get_parametric_coordinates_of_point(
  const std::vector<std::vector<double>> &tet_coords, 
  const std::array<double, 3> &pt) const
{
  std::array<double, 3> relative_coords, relative_coords1, relative_coords2, relative_coords3;
  
  for (unsigned d = 0; d < DIM; d++) {
    relative_coords1[d] = tet_coords[1][d] - tet_coords[0][d];
    relative_coords2[d] = tet_coords[2][d] - tet_coords[0][d];
    relative_coords3[d] = tet_coords[3][d] - tet_coords[0][d];
    relative_coords[d] = pt[d] - tet_coords[0][d];
  }
  
  // Solve 3x3 system using Cramer's rule
  const double a00 = relative_coords1[0];
  const double a01 = relative_coords2[0];
  const double a02 = relative_coords3[0];
  const double a10 = relative_coords1[1];
  const double a11 = relative_coords2[1];
  const double a12 = relative_coords3[1];
  const double a20 = relative_coords1[2];
  const double a21 = relative_coords2[2];
  const double a22 = relative_coords3[2];
  const double b0 = relative_coords[0];
  const double b1 = relative_coords[1];
  const double b2 = relative_coords[2];
  
  const double inv_det = 1.0 / (a00 * (a22 * a11 - a21 * a12) - 
                                a10 * (a22 * a01 - a21 * a02) + 
                                a20 * (a12 * a01 - a11 * a02));
  
  const double x = (b0 * (a22 * a11 - a21 * a12) - 
                    b1 * (a22 * a01 - a21 * a02) + 
                    b2 * (a12 * a01 - a11 * a02)) * inv_det;
  const double y = (-b0 * (a22 * a10 - a20 * a12) + 
                    b1 * (a22 * a00 - a20 * a02) - 
                    b2 * (a12 * a00 - a10 * a02)) * inv_det;
  const double z = (b0 * (a21 * a10 - a20 * a11) - 
                    b1 * (a21 * a00 - a20 * a01) + 
                    b2 * (a11 * a00 - a10 * a01)) * inv_det;
  
  return std::array<double, 3>{x, y, z};
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::point_in_bbox(
  const std::array<double, 3> &pt, 
  const std::vector<double> &bbox) const
{
  return pt[0] >= bbox[0] && pt[0] <= bbox[3] && 
         pt[1] >= bbox[1] && pt[1] <= bbox[4] && 
         pt[2] >= bbox[2] && pt[2] <= bbox[5];
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_dataset_1d(
  hid_t group_id, const char* dataset_name, std::vector<double> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  data.resize(dims[0]);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                         plist_id, data.data());
  
  H5Pclose(plist_id);
  H5Sclose(space_id);
  H5Dclose(dataset_id);
  
  if (status < 0) {
    std::string msg = std::string("Failed to read dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_dataset_1d(
  hid_t group_id, const char* dataset_name, std::vector<unsigned> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  data.resize(dims[0]);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, 
                         plist_id, data.data());
  
  H5Pclose(plist_id);
  H5Sclose(space_id);
  H5Dclose(dataset_id);
  
  if (status < 0) {
    std::string msg = std::string("Failed to read dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_dataset_2d(
  hid_t group_id, const char* dataset_name, 
  const std::vector<std::array<size_t, 2>> &row_slices,
  const std::array<size_t, 2> &col_slice,
  std::vector<double> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  // Create hyperslab selection
  hid_t file_space = H5Dget_space(dataset_id);
  
  // Calculate total size needed
  size_t total_rows = 0;
  for (const auto &slice : row_slices) {
    total_rows += slice[1] - slice[0];
  }
  size_t total_cols = col_slice[1] - col_slice[0];
  data.resize(total_rows * total_cols);
  
  // Create memory space
  hsize_t mem_dims[2] = {total_rows, total_cols};
  hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  // Read data using hyperslab selections
  size_t offset = 0;
  for (const auto &slice : row_slices) {
    hsize_t start[2] = {slice[0], col_slice[0]};
    hsize_t count[2] = {slice[1] - slice[0], total_cols};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    
    hsize_t mem_start[2] = {offset, 0};
    H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_start, NULL, count, NULL);
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, 
                           plist_id, data.data());
    if (status < 0) {
      H5Sclose(mem_space);
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      std::string msg = std::string("Failed to read dataset: ") + dataset_name;
      error->all(FLERR, msg.c_str());
    }
    
    offset += count[0];
  }
  
  H5Pclose(plist_id);
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_dataset_2d(
  hid_t group_id, const char* dataset_name, 
  const std::vector<std::array<size_t, 2>> &row_slices,
  const std::array<size_t, 2> &col_slice,
  std::vector<unsigned> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  // Create hyperslab selection
  hid_t file_space = H5Dget_space(dataset_id);
  
  // Calculate total size needed
  size_t total_rows = 0;
  for (const auto &slice : row_slices) {
    total_rows += slice[1] - slice[0];
  }
  size_t total_cols = col_slice[1] - col_slice[0];
  data.resize(total_rows * total_cols);
  
  // Create memory space
  hsize_t mem_dims[2] = {total_rows, total_cols};
  hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  // Read data using hyperslab selections
  size_t offset = 0;
  for (const auto &slice : row_slices) {
    hsize_t start[2] = {slice[0], col_slice[0]};
    hsize_t count[2] = {slice[1] - slice[0], total_cols};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    
    hsize_t mem_start[2] = {offset, 0};
    H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_start, NULL, count, NULL);
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_UINT, mem_space, file_space, 
                           plist_id, data.data());
    if (status < 0) {
      H5Sclose(mem_space);
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      std::string msg = std::string("Failed to read dataset: ") + dataset_name;
      error->all(FLERR, msg.c_str());
    }
    
    offset += count[0];
  }
  
  H5Pclose(plist_id);
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_hyperslab_1d(
  hid_t group_id, const char* dataset_name,
  const std::vector<std::array<size_t, 2>> &row_slices,
  std::vector<double> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t file_space = H5Dget_space(dataset_id);
  
  // Calculate total size needed
  size_t total_size = 0;
  for (const auto &slice : row_slices) {
    total_size += slice[1] - slice[0];
  }
  data.resize(total_size);
  
  // Create memory space
  hsize_t mem_dims[1] = {total_size};
  hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  // Read data using hyperslab selections
  size_t offset = 0;
  for (const auto &slice : row_slices) {
    hsize_t start[1] = {slice[0]};
    hsize_t count[1] = {slice[1] - slice[0]};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    
    hsize_t mem_start[1] = {offset};
    H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_start, NULL, count, NULL);
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, 
                           plist_id, data.data());
    if (status < 0) {
      H5Sclose(mem_space);
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      std::string msg = std::string("Failed to read dataset: ") + dataset_name;
      error->all(FLERR, msg.c_str());
    }
    
    offset += count[0];
  }
  
  H5Pclose(plist_id);
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_hdf5_hyperslab_1d(
  hid_t group_id, const char* dataset_name,
  const std::vector<std::array<size_t, 2>> &row_slices,
  std::vector<unsigned> &data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = std::string("Cannot open dataset: ") + dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t file_space = H5Dget_space(dataset_id);
  
  // Calculate total size needed
  size_t total_size = 0;
  for (const auto &slice : row_slices) {
    total_size += slice[1] - slice[0];
  }
  data.resize(total_size);
  
  // Create memory space
  hsize_t mem_dims[1] = {total_size};
  hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);
  
  // Create parallel data transfer property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef SPPARKS_MPI  
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
#endif
  
  // Read data using hyperslab selections
  size_t offset = 0;
  for (const auto &slice : row_slices) {
    hsize_t start[1] = {slice[0]};
    hsize_t count[1] = {slice[1] - slice[0]};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    
    hsize_t mem_start[1] = {offset};
    H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_start, NULL, count, NULL);
    
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_UINT, mem_space, file_space, 
                           plist_id, data.data());
    if (status < 0) {
      H5Sclose(mem_space);
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      std::string msg = std::string("Failed to read dataset: ") + dataset_name;
      error->all(FLERR, msg.c_str());
    }
    
    offset += count[0];
  }
  
  H5Pclose(plist_id);
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(dataset_id);
}
/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::is_point_near_boundary(
  const std::array<double, 3> &pt) const
{
  // Check if point is within one grid spacing of the subdomain boundary
  double tolerance = dx;
  
  return (pt[0] - x0[0] < tolerance) || (x0[0] + dx * (size[0] - 1) - pt[0] < tolerance) ||
         (pt[1] - x0[1] < tolerance) || (x0[1] + dx * (size[1] - 1) - pt[1] < tolerance) ||
         (pt[2] - x0[2] < tolerance) || (x0[2] + dx * (size[2] - 1) - pt[2] < tolerance);
}

/* ---------------------------------------------------------------------- */

std::vector<double> HDF5UnstructuredTemperatureSource::get_expanded_bbox(double expansion_factor) const
{
  std::vector<double> expanded_bbox = grid_bbox;
  
  // Expand bounding box by the given factor
  double dx_expand = (expanded_bbox[3] - expanded_bbox[0]) * (expansion_factor - 1.0) * 0.5;
  double dy_expand = (expanded_bbox[4] - expanded_bbox[1]) * (expansion_factor - 1.0) * 0.5;
  double dz_expand = (expanded_bbox[5] - expanded_bbox[2]) * (expansion_factor - 1.0) * 0.5;
  
  expanded_bbox[0] -= dx_expand;  // xmin
  expanded_bbox[1] -= dy_expand;  // ymin
  expanded_bbox[2] -= dz_expand;  // zmin
  expanded_bbox[3] += dx_expand;  // xmax
  expanded_bbox[4] += dy_expand;  // ymax
  expanded_bbox[5] += dz_expand;  // zmax
  
  return expanded_bbox;
}
