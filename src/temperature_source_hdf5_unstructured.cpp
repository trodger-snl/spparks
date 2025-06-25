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
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <set>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::HDF5UnstructuredTemperatureSource(SPPARKS *spk) : TemperatureSource(spk)
{
  // Initialize defaults
  filename_ = "";
  x0_ = {0.0, 0.0, 0.0};
  size_ = {0, 0, 0};
  dx_ = 1e-6;  // 1 micron default
  
  // HDF5 file handle
  file_id_ = -1;
  file_open_ = false;
  
  // Layer management
  active_layer_ = std::numeric_limits<unsigned>::max();
  current_time_ = std::numeric_limits<double>::lowest();
  
  ambient_temperature = 300.0;  // Room temperature default
  fast_forward_threshold_ = 400.0;  // Default fast-forward threshold (K)
  enable_thermal_window_ = true;  // Enable thermal window pre-calculation by default
  source_initialized = false;
}

/* ---------------------------------------------------------------------- */

HDF5UnstructuredTemperatureSource::~HDF5UnstructuredTemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup HDF5 unstructured temperature source from command line arguments
   Expected format: filename dx ambient_temp [x0_x x0_y x0_z]
------------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  parse_setup_arguments(args);
  open_hdf5_file();
  read_layer_times();
  
  if (domain->me == 0) {
    std::cout << "HDF5 Unstructured temperature source initialized:" << std::endl;
    std::cout << "  File: " << filename_ << std::endl;
    std::cout << "  Lattice spacing: " << dx_ << " m" << std::endl;
    std::cout << "  Fast-forward threshold: " << fast_forward_threshold_ << " K" << std::endl;
    std::cout << "  Ambient temperature: " << ambient_temperature << " K" << std::endl;
    std::cout << "  Origin: (" << x0_[0] << ", " << x0_[1] << ", " << x0_[2] << ")" << std::endl;
    std::cout << "  Thermal window analysis: " << (enable_thermal_window_ ? "enabled" : "disabled") << std::endl;
    std::cout << "  Layers available: " << layer_times_.size() << std::endl;
  }
  
  source_initialized = true;
  
  // Conditionally perform thermal window detection
  if (enable_thermal_window_) {
    if (domain->me == 0) {
      std::cout << "Performing thermal window analysis..." << std::endl;
    }
    auto thermal_window = find_thermal_window();
    double start_time = thermal_window.first;
    double end_time = thermal_window.second;
  
  if (start_time > 0.0 && end_time > start_time) {
    if (domain->me == 0) {
      std::cout << "🔥 THERMAL WINDOW DETECTED:" << std::endl;
      std::cout << "   Start time: " << start_time << " s" << std::endl;
      std::cout << "   End time: " << end_time << " s" << std::endl;
      std::cout << "   Duration: " << (end_time - start_time) << " s" << std::endl;
      std::cout << std::endl;
      std::cout << "📋 RECOMMENDED SPPARKS PARAMETERS:" << std::endl;
      std::cout << "   reset_time " << start_time << std::endl;
      std::cout << "   run " << end_time << " upto" << std::endl;
      std::cout << std::endl;
    }
    
    // Automatically set the simulation times in SPPARKS if not already set
    // Note: This would require SPPARKS integration to set the simulation times
    // For now, just report the findings
    } else {
      if (domain->me == 0) {
        std::cout << "⚠️  No thermal activity found in SPPARKS domain above " << fast_forward_threshold_ << "K" << std::endl;
        std::cout << "   Consider adjusting the domain or threshold" << std::endl;
      }
    }
  } else {
    if (domain->me == 0) {
      std::cout << "Thermal window analysis disabled." << std::endl;
    }
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::parse_setup_arguments(const std::vector<std::string> &args)
{
  if (args.size() < 3) {
    error->all(FLERR,"Insufficient arguments for HDF5 unstructured temperature source: need filename dx fast_forward_threshold [ambient_temp] [enable_thermal_window] OR [ambient_temp] [x0_x x0_y x0_z] [enable_thermal_window]");
  }
  
  filename_ = args[0];
  
  try {
    dx_ = std::stod(args[1]);
    fast_forward_threshold_ = std::stod(args[2]);
    
    if (args.size() >= 4) {
      ambient_temperature = std::stod(args[3]);
    }
    
    // Parse coordinates and thermal window flag
    bool coordinates_provided = false;
    if (args.size() >= 7) {
      x0_[0] = std::stod(args[4]);
      x0_[1] = std::stod(args[5]);
      x0_[2] = std::stod(args[6]);
      coordinates_provided = true;
      
      // Thermal window flag after coordinates
      if (args.size() >= 8) {
        int thermal_window_flag = std::stoi(args[7]);
        enable_thermal_window_ = (thermal_window_flag != 0);
      }
    } else {
      // Check for thermal window flag at position 5 (after ambient_temp)
      if (args.size() >= 5) {
        int thermal_window_flag = std::stoi(args[4]);
        enable_thermal_window_ = (thermal_window_flag != 0);
      }
      
      // Auto-detect origin from domain bounds
      if (domain && domain->box_exist) {
        // Get domain bounds in lattice units
        double xlo = domain->boxxlo;
        double ylo = domain->boxylo;
        double zlo = domain->boxzlo;
        double xhi = domain->boxxhi;
        double yhi = domain->boxyhi;
        double zhi = domain->boxzhi;
        
        // Convert to physical coordinates
        x0_[0] = xlo * dx_;
        x0_[1] = ylo * dx_;
        x0_[2] = zlo * dx_;
        
        // Calculate domain size in grid points
        size_[0] = static_cast<unsigned>(xhi - xlo + 1);
        size_[1] = static_cast<unsigned>(yhi - ylo + 1);
        size_[2] = static_cast<unsigned>(zhi - zlo + 1);
        
        if (domain->me == 0) {
          std::cout << "Auto-detected SPPARKS domain origin: (" 
                    << x0_[0] << ", " << x0_[1] << ", " << x0_[2] << ") m" << std::endl;
          std::cout << "Auto-detected SPPARKS domain size: [" 
                    << size_[0] << ", " << size_[1] << ", " << size_[2] << "] grid points" << std::endl;
          std::cout << "Domain bounds: x=[" << xlo << "," << xhi << "], y=[" << ylo << "," << yhi << "], z=[" << zlo << "," << zhi << "]" << std::endl;
        }
      }
    }
    
  } catch (const std::exception &e) {
    error->all(FLERR,"Invalid numeric arguments in HDF5 unstructured temperature source setup");
  }
  
  if (dx_ <= 0.0) {
    error->all(FLERR,"dx must be positive in HDF5 unstructured temperature source");
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::open_hdf5_file()
{
  file_id_ = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id_ < 0) {
    std::string msg = "Cannot open HDF5 file: " + filename_;
    error->all(FLERR, msg.c_str());
  }
  file_open_ = true;
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::close_hdf5_file()
{
  if (file_open_ && file_id_ >= 0) {
    H5Fclose(file_id_);
    file_id_ = -1;
    file_open_ = false;
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::cleanup()
{
  close_hdf5_file();
  layer_times_.clear();
  data_counts_.clear();
  node_ids_.clear();
  weights_.clear();
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_layer_times()
{
  hid_t dataset_id = H5Dopen2(file_id_, "layerTimes", H5P_DEFAULT);
  if (dataset_id < 0) {
    error->all(FLERR, "Cannot read layerTimes dataset from HDF5 file");
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  layer_times_.resize(dims[0]);
  H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, layer_times_.data());
  
  H5Sclose(space_id);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

unsigned HDF5UnstructuredTemperatureSource::get_active_layer(double time) const
{
  if (time < layer_times_.front()) {
    std::string msg = "Time " + std::to_string(time) + " is before first layer time " + std::to_string(layer_times_.front());
    error->all(FLERR, msg.c_str());
  }
  if (time > layer_times_.back()) {
    std::string msg = "Time " + std::to_string(time) + " is after last layer time " + std::to_string(layer_times_.back());
    error->all(FLERR, msg.c_str());
  }
  
  auto lower = std::lower_bound(layer_times_.begin(), layer_times_.end(), time);
  auto idx = std::distance(layer_times_.begin(), lower);
  return idx == 0 ? 0 : idx - 1;
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::get_temperature_at_xyz_and_time(double x, double y, double z, double time)
{
  check_initialization();
  
  constexpr double tol = 5.0 * std::numeric_limits<double>::epsilon();
  if (std::abs(time - current_time_) / (std::abs(time) + tol) > tol) {
    auto current_layer = get_active_layer(time);
    if (current_layer != active_layer_) {
      load_layer(current_layer);
      active_layer_ = current_layer;
    }
    current_time_ = time;
  }
  
  // Convert world coordinates to lattice indices for size calculation
  unsigned i = static_cast<unsigned>((x - x0_[0]) / dx_ + 0.5);
  unsigned j = static_cast<unsigned>((y - x0_[1]) / dx_ + 0.5);
  unsigned k = static_cast<unsigned>((z - x0_[2]) / dx_ + 0.5);
  
  // Check if we have valid size information
  if (size_[0] == 0 || size_[1] == 0 || size_[2] == 0) {
    // Set size based on domain if not explicitly set
    size_[0] = std::max(size_[0], i + 1);
    size_[1] = std::max(size_[1], j + 1);
    size_[2] = std::max(size_[2], k + 1);
  }
  
  // Check bounds
  if (i >= size_[0] || j >= size_[1] || k >= size_[2]) {
    return ambient_temperature;
  }
  
  unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
  
  // Check if we have interpolation data for this point
  if (flat_idx >= node_ids_.size()) {
    return ambient_temperature;
  }
  
  const auto& node_list = node_ids_[flat_idx];
  const auto& wts = weights_[flat_idx];
  
  // Check if point is outside the mesh
  if (node_list[0] == std::numeric_limits<unsigned>::max()) {
    return ambient_temperature;
  }
  
  // Interpolate temperature at nodes
  std::array<double, NODES_PER_ELEM> nodal_vals;
  for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
    const unsigned node_idx = node_list[n];
    if (node_idx >= data_counts_.size()) {
      return ambient_temperature;
    }
    
    const auto time_iter = times_.row_iterator(node_idx);
    const auto temp_iter = temperatures_.row_iterator(node_idx);
    const unsigned count = data_counts_[node_idx];
    
    if (count == 0) {
      nodal_vals[n] = ambient_temperature;
      continue;
    }
    
    // Return ambient temperature if time is outside node's data range
    if (time < *time_iter || time > time_iter[count - 1]) {
      nodal_vals[n] = ambient_temperature;
      continue;
    }
    
    auto lower = std::lower_bound(time_iter, time_iter + count, time);
    auto idx = std::distance(time_iter, lower);
    idx = idx == 0 ? 1 : idx;
    
    // Linear interpolation in time
    double t0 = time_iter[idx - 1];
    double t1 = time_iter[idx];
    double temp0 = temp_iter[idx - 1];
    double temp1 = temp_iter[idx];
    
    nodal_vals[n] = temp0 + (temp1 - temp0) / (t1 - t0) * (time - t0);
  }
  
  // Interpolate in space using barycentric coordinates
  return nodal_vals[0] + wts[0] * (nodal_vals[1] - nodal_vals[0]) + 
         wts[1] * (nodal_vals[2] - nodal_vals[0]) + wts[2] * (nodal_vals[3] - nodal_vals[0]);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::update_temperatures(double dt, double simulation_time)
{
  // Temperature updates are handled on-demand in get_temperature_at_xyz_and_time
  // This method can be used for any periodic maintenance if needed
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::needs_data_refresh(double simulation_time)
{
  if (!source_initialized) return false;
  
  try {
    unsigned current_layer = get_active_layer(simulation_time);
    return current_layer != active_layer_;
  } catch (...) {
    return false;  // Time out of range
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::print_source_info() const
{
  if (domain->me == 0) {
    std::cout << "HDF5 Unstructured Temperature Source:" << std::endl;
    std::cout << "  File: " << filename_ << std::endl;
    std::cout << "  Layers: " << layer_times_.size() << std::endl;
    std::cout << "  Current layer: " << active_layer_ << std::endl;
    std::cout << "  Ambient temperature: " << ambient_temperature << " K" << std::endl;
  }
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::load_layer(unsigned layer_idx)
{
  if (layer_idx >= layer_times_.size()) {
    error->all(FLERR, "Invalid layer index in load_layer");
  }
  
  // Open layer group
  std::string group_name = std::to_string(layer_idx);
  hid_t group_id = H5Gopen2(file_id_, group_name.c_str(), H5P_DEFAULT);
  if (group_id < 0) {
    std::string msg = "Cannot open layer group: " + group_name;
    error->all(FLERR, msg.c_str());
  }
  
  try {
    // Read bounding boxes
    std::vector<std::vector<double>> bboxes;
    read_dataset_2d(group_id, "boundingBoxes", bboxes);
    
    // Read element and node pointers
    std::vector<unsigned> elem_ptr, node_ptr;
    read_dataset_1d(group_id, "elemPtrs", elem_ptr);
    read_dataset_1d(group_id, "nodePtrs", node_ptr);
    
    // Define grid bounding box
    std::vector<double> grid_bbox = {
      x0_[0], x0_[1], x0_[2],
      x0_[0] + dx_ * (size_[0] - 1), 
      x0_[1] + dx_ * (size_[1] - 1), 
      x0_[2] + dx_ * (size_[2] - 1)
    };
    
    // Find overlapping chunks
    std::vector<unsigned> overlapping_chunks;
    std::vector<std::array<size_t, 2>> elem_slices, node_slices;
    std::vector<unsigned> node_offsets{0}, elem_offsets{0};
    
    for (unsigned c = 0; c < bboxes.size(); c++) {
      if (do_boxes_overlap(bboxes[c], grid_bbox)) {
        overlapping_chunks.push_back(c);
        elem_slices.push_back({elem_ptr[c], elem_ptr[c + 1]});
        node_slices.push_back({node_ptr[c], node_ptr[c + 1]});
        node_offsets.push_back(node_offsets.back() + node_ptr[c + 1] - node_ptr[c]);
        elem_offsets.push_back(elem_offsets.back() + elem_ptr[c + 1] - elem_ptr[c]);
      }
    }
    
    if (overlapping_chunks.empty()) {
      // No overlapping chunks - initialize with empty data
      if (domain->me == 0) {
        std::cout << "    DEBUG: No overlapping chunks found for layer " << layer_idx << std::endl;
        std::cout << "    SPPARKS grid bbox: [" << grid_bbox[0] << ", " << grid_bbox[1] << ", " << grid_bbox[2] 
                  << "] to [" << grid_bbox[3] << ", " << grid_bbox[4] << ", " << grid_bbox[5] << "]" << std::endl;
        std::cout << "    Total HDF5 chunks in layer: " << bboxes.size() << std::endl;
      }
      node_ids_.assign(size_[0] * size_[1] * size_[2], 
                       {std::numeric_limits<unsigned>::max(), 
                        std::numeric_limits<unsigned>::max(),
                        std::numeric_limits<unsigned>::max(), 
                        std::numeric_limits<unsigned>::max()});
      weights_.assign(size_[0] * size_[1] * size_[2], {0.0, 0.0, 0.0});
      H5Gclose(group_id);
      return;
    }
    
    // Read element-to-node connectivity
    std::vector<double> elem_node_data;
    read_partial_dataset_2d(group_id, "elementToNode", elem_slices, {0, NODES_PER_ELEM}, elem_node_data);
    Array2D<unsigned> elem_node(NODES_PER_ELEM, std::vector<unsigned>(elem_node_data.begin(), elem_node_data.end()));
    
    // Read node coordinates
    std::vector<double> node_coords_data;
    read_partial_dataset_2d(group_id, "nodeCoords", node_slices, {0, DIM}, node_coords_data);
    Array2D<double> node_coords(DIM, std::move(node_coords_data));
    
    // Read data counts
    data_counts_.clear();
    read_dataset_1d(group_id, "dataCounts", data_counts_);
    
    // Find maximum data count
    unsigned max_data = 0;
    for (auto count : data_counts_) {
      max_data = std::max(max_data, count);
    }
    
    if (domain->me == 0) {
      std::cout << "    DEBUG: data_counts size=" << data_counts_.size() << ", max_data=" << max_data << std::endl;
      if (!data_counts_.empty()) {
        unsigned non_zero_counts = 0;
        for (auto count : data_counts_) {
          if (count > 0) non_zero_counts++;
        }
        std::cout << "    Non-zero data counts: " << non_zero_counts << "/" << data_counts_.size() << std::endl;
      }
    }
    
    // Read times and temperatures
    std::vector<double> times_data, temp_data;
    
    if (domain->me == 0) {
      std::cout << "    DEBUG: About to read temperature data - max_data=" << max_data << ", node_slices.size()=" << node_slices.size() << std::endl;
      if (!node_slices.empty()) {
        std::cout << "    First node slice: [" << node_slices[0][0] << ", " << node_slices[0][1] << "]" << std::endl;
      }
    }
    
    read_partial_dataset_2d(group_id, "times", node_slices, {0, max_data}, times_data);
    read_partial_dataset_2d(group_id, "temperatures", node_slices, {0, max_data}, temp_data);
    
    // Create mapping from global node ID to local Array2D row index
    global_to_local_node_map_.clear();
    unsigned local_row_idx = 0;
    for (const auto& slice : node_slices) {
      for (unsigned global_node_id = slice[0]; global_node_id < slice[1]; global_node_id++) {
        global_to_local_node_map_[global_node_id] = local_row_idx++;
      }
    }
    
    times_ = Array2D<double>(max_data, std::move(times_data));
    temperatures_ = Array2D<double>(max_data, std::move(temp_data));
    
    if (domain->me == 0) {
      std::cout << "    DEBUG: Successfully read temperature data for layer " << layer_idx << std::endl;
      std::cout << "    Overlapping chunks: " << overlapping_chunks.size() << std::endl;
      std::cout << "    Max data count: " << max_data << std::endl;
      std::cout << "    Global->local node mapping created: " << global_to_local_node_map_.size() << " entries" << std::endl;
      std::cout << "    Array2D dimensions: " << local_row_idx << " rows x " << max_data << " cols" << std::endl;
      
      // Sample temperature values to verify
      if (temperatures_.rows() > 0 && temperatures_.cols() > 0) {
        double sample_temp = temperatures_(0, 0);
        std::cout << "    Sample temperature at (0,0): " << sample_temp << "K" << std::endl;
      }
    }
    
    // Build element bounding boxes
    auto elem_bboxes = build_elem_bounding_boxes(overlapping_chunks.size(), node_offsets, 
                                                elem_offsets, elem_node, node_coords);
    
    // Initialize interpolation data
    node_ids_.resize(size_[0] * size_[1] * size_[2]);
    weights_.resize(size_[0] * size_[1] * size_[2]);
    
    // Find interpolation weights for each grid point
    unsigned total_points = size_[0] * size_[1] * size_[2];
    unsigned points_mapped = 0;
    unsigned points_outside = 0;
    
    for (unsigned i = 0; i < size_[0]; i++) {
      for (unsigned j = 0; j < size_[1]; j++) {
        for (unsigned k = 0; k < size_[2]; k++) {
          unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
          std::array<double, 3> pt = {x0_[0] + i * dx_, x0_[1] + j * dx_, x0_[2] + k * dx_};
          
          auto result = find_element_point_is_in(overlapping_chunks, bboxes, node_offsets,
                                                elem_offsets, elem_node, node_coords, elem_bboxes, pt);
          node_ids_[flat_idx] = result.first;
          weights_[flat_idx] = result.second;
          
          if (result.first[0] != std::numeric_limits<unsigned>::max()) {
            points_mapped++;
          } else {
            points_outside++;
          }
        }
      }
    }
    
    if (domain->me == 0) {
      std::cout << "    Total SPPARKS grid points: " << total_points << std::endl;
      std::cout << "    Spatial interpolation: " << points_mapped << " mapped, " 
                << points_outside << " outside mesh" << std::endl;
      std::cout << "    Coverage: " << (100.0 * points_mapped / total_points) << "%" << std::endl;
      
      // Sanity check
      if (points_mapped + points_outside != total_points) {
        std::cout << "    WARNING: Point count mismatch! " 
                  << (points_mapped + points_outside) << " != " << total_points << std::endl;
      }
      
      // Debug interpolation quality in a small region
      std::cout << "    Checking interpolation consistency in center region..." << std::endl;
      unsigned center_i = size_[0] / 2;
      unsigned center_j = size_[1] / 2; 
      unsigned center_k = size_[2] / 2;
      
      unsigned discontinuities = 0;
      for (unsigned i = center_i; i < std::min(center_i + 10, size_[0]); i++) {
        for (unsigned j = center_j; j < std::min(center_j + 10, size_[1]); j++) {
          for (unsigned k = center_k; k < std::min(center_k + 10, size_[2]); k++) {
            unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
            auto& node_ids = node_ids_[flat_idx];
            
            // Check neighboring points for consistency
            if (i + 1 < size_[0]) {
              unsigned neighbor_idx = (i + 1) * size_[1] * size_[2] + j * size_[2] + k;
              auto& neighbor_ids = node_ids_[neighbor_idx];
              
              // Compare node assignments
              bool same_element = true;
              for (int n = 0; n < 4; n++) {
                if (node_ids[n] != neighbor_ids[n]) {
                  same_element = false;
                  break;
                }
              }
              
              if (!same_element && 
                  node_ids[0] != std::numeric_limits<unsigned>::max() &&
                  neighbor_ids[0] != std::numeric_limits<unsigned>::max()) {
                discontinuities++;
              }
            }
          }
        }
      }
      std::cout << "    Found " << discontinuities << " element discontinuities in 10×10×10 test region" << std::endl;
      
      // Debug specific discontinuous points
      if (discontinuities > 0) {
        std::cout << "    Analyzing first few discontinuous points:" << std::endl;
        unsigned debug_count = 0;
        for (unsigned i = center_i; i < std::min(center_i + 10, size_[0]) && debug_count < 3; i++) {
          for (unsigned j = center_j; j < std::min(center_j + 10, size_[1]) && debug_count < 3; j++) {
            for (unsigned k = center_k; k < std::min(center_k + 10, size_[2]) && debug_count < 3; k++) {
              if (i + 1 >= size_[0]) continue;
              
              unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
              unsigned neighbor_idx = (i + 1) * size_[1] * size_[2] + j * size_[2] + k;
              
              auto& node_ids = node_ids_[flat_idx];
              auto& neighbor_ids = node_ids_[neighbor_idx];
              
              bool same_element = true;
              for (int n = 0; n < 4; n++) {
                if (node_ids[n] != neighbor_ids[n]) {
                  same_element = false;
                  break;
                }
              }
              
              if (!same_element && 
                  node_ids[0] != std::numeric_limits<unsigned>::max() &&
                  neighbor_ids[0] != std::numeric_limits<unsigned>::max()) {
                
                std::array<double, 3> pt1 = {x0_[0] + i * dx_, x0_[1] + j * dx_, x0_[2] + k * dx_};
                std::array<double, 3> pt2 = {x0_[0] + (i+1) * dx_, x0_[1] + j * dx_, x0_[2] + k * dx_};
                
                std::cout << "      Point (" << i << "," << j << "," << k << ") at (" 
                          << pt1[0] << "," << pt1[1] << "," << pt1[2] << ") -> elements ["
                          << node_ids[0] << "," << node_ids[1] << "," << node_ids[2] << "," << node_ids[3] << "]" << std::endl;
                std::cout << "      Point (" << (i+1) << "," << j << "," << k << ") at (" 
                          << pt2[0] << "," << pt2[1] << "," << pt2[2] << ") -> elements ["
                          << neighbor_ids[0] << "," << neighbor_ids[1] << "," << neighbor_ids[2] << "," << neighbor_ids[3] << "]" << std::endl;
                
                debug_count++;
              }
            }
          }
        }
      }
    }
    
  } catch (const std::exception& e) {
    H5Gclose(group_id);
    std::string msg = "Error loading layer " + std::to_string(layer_idx) + ": " + e.what();
    error->all(FLERR, msg.c_str());
  }
  
  H5Gclose(group_id);
}

/* ---------------------------------------------------------------------- */
// Geometric calculation methods (adapted from original HighFive code)

std::array<double, 3> HDF5UnstructuredTemperatureSource::get_parametric_coordinates_of_point(
  const std::vector<std::vector<double>>& tet_coords, 
  const std::array<double, 3>& pt) const
{
  std::array<double, 3> relative_coords, relative_coords1, relative_coords2, relative_coords3;
  
  for (unsigned d = 0; d < pt.size(); d++) {
    relative_coords1[d] = tet_coords[1][d] - tet_coords[0][d];
    relative_coords2[d] = tet_coords[2][d] - tet_coords[0][d];
    relative_coords3[d] = tet_coords[3][d] - tet_coords[0][d];
    relative_coords[d] = pt[d] - tet_coords[0][d];
  }
  
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
  
  const double inv_det = 1.0 / (a00 * (a22 * a11 - a21 * a12) - a10 * (a22 * a01 - a21 * a02) + a20 * (a12 * a01 - a11 * a02));
  const double x = (b0 * (a22 * a11 - a21 * a12) - b1 * (a22 * a01 - a21 * a02) + b2 * (a12 * a01 - a11 * a02)) * inv_det;
  const double y = (-b0 * (a22 * a10 - a20 * a12) + b1 * (a22 * a00 - a20 * a02) - b2 * (a12 * a00 - a10 * a02)) * inv_det;
  const double z = (b0 * (a21 * a10 - a20 * a11) - b1 * (a21 * a00 - a20 * a01) + b2 * (a11 * a00 - a10 * a01)) * inv_det;
  
  return std::array<double, 3>{x, y, z};
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::do_boxes_overlap(const std::vector<double>& b1, const std::vector<double>& b2) const
{
  return b1[0] < b2[3] && b2[0] < b1[3] && b1[1] < b2[4] && b2[1] < b1[4] && b1[2] < b2[5] && b2[2] < b1[5];
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::point_in_bbox(const std::array<double, 3>& pt, const std::vector<double>& bbox) const
{
  return pt[0] >= bbox[0] && pt[0] <= bbox[3] && pt[1] >= bbox[1] && pt[1] <= bbox[4] && pt[2] >= bbox[2] && pt[2] <= bbox[5];
}

/* ---------------------------------------------------------------------- */

std::pair<std::array<unsigned, 4>, std::array<double, 3>> 
HDF5UnstructuredTemperatureSource::find_element_point_is_in(
  const std::vector<unsigned>& selected_chunks,
  const std::vector<std::vector<double>>& chunk_bboxes,
  const std::vector<unsigned>& node_offsets,
  const std::vector<unsigned>& elem_offsets,
  const Array2D<unsigned>& elem_node,
  const Array2D<double>& node_coords,
  const std::vector<std::vector<double>>& elem_bboxes,
  const std::array<double, 3>& pt) const
{
  std::array<unsigned, 4> node_ids;
  std::array<double, 3> par_coords;
  
  constexpr double tol = 1e-8;   // Tolerance appropriate for mesh coordinates in mm range
  constexpr unsigned invalid_id = std::numeric_limits<unsigned>::max();
  
  std::vector<unsigned> possible_chunks;
  for (unsigned c = 0; c < selected_chunks.size(); c++) {
    const auto& c_bbox = chunk_bboxes[selected_chunks[c]];
    if (point_in_bbox(pt, c_bbox)) {
      possible_chunks.push_back(c);
    }
  }
  
  for (auto c : possible_chunks) {
    for (unsigned e = elem_offsets[c]; e < elem_offsets[c + 1]; e++) {
      if (point_in_bbox(pt, elem_bboxes[e])) {
        std::vector<std::vector<double>> tet_coords(4, std::vector<double>(DIM));
        for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
          node_ids[n] = elem_node(e, n) + node_offsets[c];
          for (unsigned d = 0; d < DIM; d++) {
            tet_coords[n][d] = node_coords(node_ids[n], d);
          }
        }
        
        par_coords = get_parametric_coordinates_of_point(tet_coords, pt);
        if (par_coords[0] > -tol && par_coords[1] > -tol && par_coords[2] > -tol &&
            1.0 - par_coords[0] - par_coords[1] - par_coords[2] > -tol) {
          return std::make_pair(node_ids, par_coords);
        }
      }
    }
  }
  
  return std::make_pair(std::array<unsigned, 4>{invalid_id, invalid_id, invalid_id, invalid_id}, 
                       std::array<double, 3>());
}

/* ---------------------------------------------------------------------- */

std::vector<std::vector<double>> HDF5UnstructuredTemperatureSource::build_elem_bounding_boxes(
  unsigned n_chunks,
  const std::vector<unsigned>& node_offsets,
  const std::vector<unsigned>& elem_offsets,
  const Array2D<unsigned>& elem_node,
  const Array2D<double>& node_coords) const
{
  constexpr double max_val = std::numeric_limits<double>::max();
  constexpr double min_val = std::numeric_limits<double>::lowest();
  
  std::vector<std::vector<double>> result;
  for (unsigned c = 0; c < n_chunks; c++) {
    for (unsigned e = elem_offsets[c]; e < elem_offsets[c + 1]; e++) {
      std::array<double, 3> bmin{max_val, max_val, max_val};
      std::array<double, 3> bmax{min_val, min_val, min_val};
      
      for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
        for (unsigned d = 0; d < DIM; d++) {
          const double coord_val = node_coords(elem_node(e, n) + node_offsets[c], d);
          bmin[d] = std::min(bmin[d], coord_val);
          bmax[d] = std::max(bmax[d], coord_val);
        }
      }
      result.push_back({bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]});
    }
  }
  return result;
}

/* ---------------------------------------------------------------------- */
// HDF5 helper functions

void HDF5UnstructuredTemperatureSource::read_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<double>& data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = "Cannot read dataset: ";
    msg += dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  data.resize(dims[0]);
  H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  
  H5Sclose(space_id);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<unsigned>& data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = "Cannot read dataset: ";
    msg += dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  data.resize(dims[0]);
  H5Dread(dataset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  
  H5Sclose(space_id);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_dataset_2d(hid_t group_id, const char* dataset_name, std::vector<std::vector<double>>& data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = "Cannot read dataset: ";
    msg += dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[2];
  H5Sget_simple_extent_dims(space_id, dims, NULL);
  
  data.resize(dims[0]);
  std::vector<double> flat_data(dims[0] * dims[1]);
  H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_data.data());
  
  for (hsize_t i = 0; i < dims[0]; i++) {
    data[i].resize(dims[1]);
    for (hsize_t j = 0; j < dims[1]; j++) {
      data[i][j] = flat_data[i * dims[1] + j];
    }
  }
  
  H5Sclose(space_id);
  H5Dclose(dataset_id);
}

/* ---------------------------------------------------------------------- */

void HDF5UnstructuredTemperatureSource::read_partial_dataset_2d(hid_t group_id, const char* dataset_name, 
                                                               const std::vector<std::array<size_t, 2>>& row_slices,
                                                               const std::array<size_t, 2>& col_slice,
                                                               std::vector<double>& data)
{
  hid_t dataset_id = H5Dopen2(group_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::string msg = "Cannot read dataset: ";
    msg += dataset_name;
    error->all(FLERR, msg.c_str());
  }
  
  // Get dataset dimensions for validation
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[2];
  int ndims = H5Sget_simple_extent_dims(space_id, dims, NULL);
  if (ndims != 2) {
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    error->all(FLERR, "Dataset must be 2D");
  }
  H5Sclose(space_id);
  
  // Reduced debug output for dataset dimensions
  if (domain->me == 0 && std::string(dataset_name).find("elementToNode") != std::string::npos) {
    std::cout << "      Reading " << dataset_name << ": " << dims[0] << "×" << dims[1] << std::endl;
  }
  
  // Calculate total size needed
  size_t total_rows = 0;
  for (const auto& slice : row_slices) {
    total_rows += slice[1] - slice[0];
  }
  size_t total_cols = col_slice[1] - col_slice[0];
  
  // Minimal debug output for temperature data reading
  if (domain->me == 0) {
    std::cout << "      Reading temperature data: " << total_rows << "×" << total_cols << std::endl;
  }
  
  data.clear();
  data.reserve(total_rows * total_cols);
  
  // Read each row slice
  size_t slice_index = 0;
  for (const auto& slice : row_slices) {
    size_t slice_rows = slice[1] - slice[0];
    if (slice_rows == 0) {
      slice_index++;
      continue;
    }
    
    // Validate slice bounds
    if (slice[0] >= dims[0] || slice[1] > dims[0]) {
      if (domain->me == 0) {
        std::cout << "      WARNING: Row slice [" << slice[0] << ", " << slice[1] 
                  << "] exceeds dataset bounds [0, " << dims[0] << "]" << std::endl;
      }
      slice_index++;
      continue;
    }
    
    if (col_slice[0] >= dims[1] || col_slice[1] > dims[1]) {
      if (domain->me == 0) {
        std::cout << "      WARNING: Col slice [" << col_slice[0] << ", " << col_slice[1] 
                  << "] exceeds dataset bounds [0, " << dims[1] << "]" << std::endl;
      }
      slice_index++;
      continue;
    }
    
    hsize_t start[2] = {slice[0], col_slice[0]};
    hsize_t count[2] = {slice_rows, total_cols};
    
    // Reduced debug output - only show for first few slices
    if (domain->me == 0 && slice_index < 3) {
      std::cout << "      DEBUG: Reading slice start=[" << start[0] << "," << start[1] 
                << "] count=[" << count[0] << "," << count[1] << "]" << std::endl;
    }
    
    hid_t file_space = H5Dget_space(dataset_id);
    herr_t select_status = H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    if (select_status < 0) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      error->all(FLERR, "Failed to select hyperslab for dataset read");
    }
    
    hsize_t mem_dims[2] = {slice_rows, total_cols};
    hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);
    
    std::vector<double> slice_data(slice_rows * total_cols);
    herr_t read_status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, 
                                slice_data.data());
    
    if (read_status < 0) {
      H5Sclose(mem_space);
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      error->all(FLERR, "Failed to read dataset slice");
    }
    
    // Only show slice data range for first few slices, but track overall min/max
    if (domain->me == 0) {
      if (!slice_data.empty()) {
        double slice_min = *std::min_element(slice_data.begin(), slice_data.end());
        double slice_max = *std::max_element(slice_data.begin(), slice_data.end());
        if (slice_index < 3) {
          std::cout << "      DEBUG: Read " << slice_data.size() << " elements, range: " << slice_min << " to " << slice_max << std::endl;
        }
      }
    }
    
    // Append slice data to the main data vector
    data.insert(data.end(), slice_data.begin(), slice_data.end());
    
    H5Sclose(mem_space);
    H5Sclose(file_space);
    
    slice_index++;
  }
  
  // Remove redundant total data read message
  
  H5Dclose(dataset_id);
}

/* ----------------------------------------------------------------------
   Fast-forward optimization methods
------------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::all_temperatures_below_threshold(double time)
{
  check_initialization();
  
  // For unstructured data, we'll implement a simple check
  // This is a simplified implementation - in practice you might want to 
  // cache temperature statistics for better performance
  
  try {
    auto current_layer = get_active_layer(time);
    if (current_layer != active_layer_) {
      load_layer(current_layer);
      active_layer_ = current_layer;
    }
    current_time_ = time;
    
    // Check all nodal temperatures in the current layer's data
    // This ensures we don't miss any hot spots in the unstructured mesh
    if (temperatures_.rows() == 0 || temperatures_.cols() == 0) {
      return true;  // No data loaded, safe to fast-forward
    }
    
    if (data_counts_.empty()) {
      return true;  // No count data, safe to fast-forward
    }
    
    // Check every node's temperature at the requested time
    for (unsigned node_idx = 0; node_idx < temperatures_.rows(); node_idx++) {
      const unsigned count = data_counts_[node_idx];
      if (count == 0) continue;  // No data for this node
      
      const auto time_iter = times_.row_iterator(node_idx);
      const auto temp_iter = temperatures_.row_iterator(node_idx);
      
      // Skip if time is outside this node's range
      if (time < *time_iter || time > time_iter[count - 1]) {
        continue;  // Use ambient temperature (below threshold)
      }
      
      // Interpolate temperature at this node
      auto lower = std::lower_bound(time_iter, time_iter + count, time);
      auto idx = std::distance(time_iter, lower);
      idx = idx == 0 ? 1 : idx;
      
      double t0 = time_iter[idx - 1];
      double t1 = time_iter[idx];
      double temp0 = temp_iter[idx - 1];
      double temp1 = temp_iter[idx];
      
      double node_temp = temp0 + (temp1 - temp0) / (t1 - t0) * (time - t0);
      
      if (node_temp >= fast_forward_threshold_) {
        return false;  // Found temperature above threshold
      }
    }
    return true;
  } catch (...) {
    return false;  // If we can't check, don't fast-forward
  }
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::find_next_active_time(double start_time, double end_time)
{
  check_initialization();
  
  // Binary search approach to find next time when temperature exceeds threshold
  double left = start_time;
  double right = end_time;
  constexpr double time_tolerance = 1e-6;
  
  // If end time is already active, return it
  if (!all_temperatures_below_threshold(right)) {
    return right;
  }
  
  // Binary search for the transition point
  while (right - left > time_tolerance) {
    double mid = 0.5 * (left + right);
    if (all_temperatures_below_threshold(mid)) {
      left = mid;
    } else {
      right = mid;
    }
  }
  
  return right;
}

/* ---------------------------------------------------------------------- */

double HDF5UnstructuredTemperatureSource::find_next_active_time_sequential(double start_time, double end_time)
{
  check_initialization();
  
  // Sequential sampling approach - safer for complex thermal histories
  constexpr double time_step = 1e-4;  // 0.1ms sampling
  
  for (double t = start_time; t <= end_time; t += time_step) {
    if (!all_temperatures_below_threshold(t)) {
      return t;
    }
  }
  
  return end_time;  // No active time found in range
}

/* ----------------------------------------------------------------------
   Thermal window detection methods
------------------------------------------------------------------------- */

std::pair<double, double> HDF5UnstructuredTemperatureSource::find_thermal_window()
{
  check_initialization();
  
  if (domain->me == 0) {
    std::cout << "Searching for thermal window above " << fast_forward_threshold_ << "K..." << std::endl;
  }
  
  double first_active_time = std::numeric_limits<double>::max();
  double last_active_time = std::numeric_limits<double>::lowest();
  
  // Search through all layers to find temperature data
  for (unsigned layer_idx = 0; layer_idx < layer_times_.size(); layer_idx++) {
    // Check if layer exists in HDF5 file
    std::string layer_name = std::to_string(layer_idx);
    hid_t group_test = H5Gopen2(file_id_, layer_name.c_str(), H5P_DEFAULT);
    if (group_test < 0) {
      if (domain->me == 0) {
        std::cout << "  Skipping layer " << layer_idx << " (not found in HDF5 file)" << std::endl;
      }
      continue;
    }
    H5Gclose(group_test);
    
    // Load this layer to check temperatures
    load_layer(layer_idx);
    active_layer_ = layer_idx;
    
    if (domain->me == 0) {
      std::cout << "  Checking layer " << layer_idx << " (t=" << layer_times_[layer_idx] << "s)..." << std::flush;
      std::cout << " Domain size: [" << size_[0] << ", " << size_[1] << ", " << size_[2] << "]" << std::flush;
    }
    
    bool layer_has_hot_temps = false;
    unsigned hot_points = 0;
    double layer_max_temp = 0.0;
    
    // Track statistics for debugging
    unsigned points_checked = 0;
    unsigned points_outside_mesh = 0;
    unsigned points_no_mapping = 0;
    unsigned points_with_data = 0;
    
    // Sample grid points in SPPARKS domain  
    // Only show detailed sampling info for first layer
    if (domain->me == 0 && layer_idx == 0) {
      std::cout << " Sampling domain: [" << size_[0] << "×" << size_[1] << "×" << size_[2] << "]" << std::endl;
    }
    
    for (unsigned i = 0; i < size_[0]; i += std::max(1u, size_[0]/20)) {
      for (unsigned j = 0; j < size_[1]; j += std::max(1u, size_[1]/20)) {
        for (unsigned k = 0; k < size_[2]; k += std::max(1u, size_[2]/20)) {
          double x = x0_[0] + i * dx_;
          double y = x0_[1] + j * dx_;
          double z = x0_[2] + k * dx_;
          
          points_checked++;
          
          unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
          if (flat_idx >= node_ids_.size()) {
            points_outside_mesh++;
            continue;
          }
          
          const auto& node_list = node_ids_[flat_idx];
          const auto& wts = weights_[flat_idx];
          
          // Check if point is outside the mesh
          if (node_list[0] == std::numeric_limits<unsigned>::max()) {
            points_outside_mesh++;
            continue;
          }
          
          points_with_data++;
          
          // Sample temperatures at different times for this spatial location
          for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
            const unsigned global_node_id = node_list[n];
            if (global_node_id >= data_counts_.size()) continue;
            
            // Map global node ID to local Array2D row index
            auto it = global_to_local_node_map_.find(global_node_id);
            if (it == global_to_local_node_map_.end()) continue;  // Node not in current layer data
            
            const unsigned local_row_idx = it->second;
            const auto time_iter = times_.row_iterator(local_row_idx);
            const auto temp_iter = temperatures_.row_iterator(local_row_idx);
            const unsigned count = data_counts_[global_node_id];
            
            if (count == 0) continue;
            
            // Check all time points for this node
            if (domain->me == 0 && hot_points < 3 && layer_max_temp < 400) {
              std::cout << "      DEBUG Layer " << layer_idx << ": Node " << global_node_id 
                        << " (local row " << local_row_idx << ") has " << count << " time points" << std::endl;
            }
            
            for (unsigned t = 0; t < count; t++) {
              double temp = temp_iter[t];
              double time = time_iter[t];
              layer_max_temp = std::max(layer_max_temp, temp);
              
              if (domain->me == 0 && hot_points < 3 && t < 3 && layer_max_temp < 400) {
                std::cout << "        Time point " << t << ": temp=" << temp << "K, time=" << time << "s" << std::endl;
              }
              
              if (temp > fast_forward_threshold_) {
                layer_has_hot_temps = true;
                hot_points++;
                first_active_time = std::min(first_active_time, time);
                last_active_time = std::max(last_active_time, time);
                
                if (domain->me == 0 && hot_points <= 3) {
                  std::cout << "        🔥 HOT TEMP FOUND in Layer " << layer_idx 
                            << ": " << temp << "K at time " << time << "s (node " 
                            << global_node_id << ")" << std::endl;
                }
              }
            }
          }
        }
      }
    }
    
    if (domain->me == 0) {
      std::cout << " max_temp=" << layer_max_temp << "K, hot_points=" << hot_points << std::endl;
      std::cout << "      Stats: checked=" << points_checked << ", outside_mesh=" << points_outside_mesh 
                << ", with_data=" << points_with_data << std::endl;
      
      // If we found no temperatures above 0, there's likely an issue with data access
      if (layer_max_temp == 0.0 && points_with_data > 0) {
        std::cout << "      ⚠️  WARNING: Found " << points_with_data << " points with data but all temperatures are 0K!" << std::endl;
        std::cout << "      This suggests an issue with Array2D data access or node mapping." << std::endl;
      }
    }
  }
  
  if (first_active_time == std::numeric_limits<double>::max()) {
    if (domain->me == 0) {
      std::cout << "❌ No thermal activity found above " << fast_forward_threshold_ << "K in SPPARKS domain" << std::endl;
    }
    return std::make_pair(0.0, 0.0);
  }
  
  if (domain->me == 0) {
    std::cout << "✅ Thermal window found: " << first_active_time << "s to " << last_active_time << "s" << std::endl;
    std::cout << "   Duration: " << (last_active_time - first_active_time) << "s" << std::endl;
  }
  
  return std::make_pair(first_active_time, last_active_time);
}

/* ---------------------------------------------------------------------- */

bool HDF5UnstructuredTemperatureSource::has_temperatures_above_threshold_at_time(double time)
{
  check_initialization();
  
  try {
    auto current_layer = get_active_layer(time);
    if (current_layer != active_layer_) {
      load_layer(current_layer);
      active_layer_ = current_layer;
    }
    current_time_ = time;
    
    // Sample a subset of grid points
    for (unsigned i = 0; i < size_[0]; i += std::max(1u, size_[0]/10)) {
      for (unsigned j = 0; j < size_[1]; j += std::max(1u, size_[1]/10)) {
        for (unsigned k = 0; k < size_[2]; k += std::max(1u, size_[2]/10)) {
          double x = x0_[0] + i * dx_;
          double y = x0_[1] + j * dx_;
          double z = x0_[2] + k * dx_;
          
          double temp = get_temperature_at_xyz_and_time(x, y, z, time);
          if (temp > fast_forward_threshold_) {
            return true;
          }
        }
      }
    }
    return false;
  } catch (...) {
    return false;
  }
}

/* ---------------------------------------------------------------------- */

std::vector<double> HDF5UnstructuredTemperatureSource::get_active_time_points()
{
  check_initialization();
  
  std::vector<double> active_times;
  
  // Collect all unique time points where temperatures exceed threshold
  for (unsigned layer_idx = 0; layer_idx < layer_times_.size(); layer_idx++) {
    load_layer(layer_idx);
    active_layer_ = layer_idx;
    
    std::set<double> layer_active_times;
    
    // Check temperature data across the domain
    for (unsigned i = 0; i < std::min(size_[0], 50u); i += std::max(1u, size_[0]/10)) {
      for (unsigned j = 0; j < std::min(size_[1], 50u); j += std::max(1u, size_[1]/10)) {
        for (unsigned k = 0; k < std::min(size_[2], 50u); k += std::max(1u, size_[2]/10)) {
          unsigned flat_idx = i * size_[1] * size_[2] + j * size_[2] + k;
          if (flat_idx >= node_ids_.size()) continue;
          
          const auto& node_list = node_ids_[flat_idx];
          if (node_list[0] == std::numeric_limits<unsigned>::max()) continue;
          
          for (unsigned n = 0; n < NODES_PER_ELEM; n++) {
            const unsigned node_idx = node_list[n];
            if (node_idx >= data_counts_.size()) continue;
            
            const auto time_iter = times_.row_iterator(node_idx);
            const auto temp_iter = temperatures_.row_iterator(node_idx);
            const unsigned count = data_counts_[node_idx];
            
            for (unsigned t = 0; t < count; t++) {
              if (temp_iter[t] > fast_forward_threshold_) {
                layer_active_times.insert(time_iter[t]);
              }
            }
          }
        }
      }
    }
    
    // Add layer active times to global list
    for (double t : layer_active_times) {
      active_times.push_back(t);
    }
  }
  
  // Sort and remove duplicates
  std::sort(active_times.begin(), active_times.end());
  active_times.erase(std::unique(active_times.begin(), active_times.end()), active_times.end());
  
  return active_times;
}