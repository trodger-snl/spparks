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

#ifndef SPPARKS_TEMPERATURE_SOURCE_HDF5_UNSTRUCTURED_H
#define SPPARKS_TEMPERATURE_SOURCE_HDF5_UNSTRUCTURED_H

#include "temperature_source.h"
#include "hdf5.h"
#include <vector>
#include <array>
#include <limits>
#include <memory>

namespace SPPARKS_NS {

template <typename T>
class array2D
{
public:
  array2D() : ncols(0), nrows(0) {}
  array2D(unsigned ncols_, std::vector<T> && data_) : 
    ncols(ncols_), data(std::move(data_)) 
  {
    if(data.size() % ncols != 0) {
      throw std::runtime_error("Data size must be multiple of number of columns");
    }
    nrows = data.size() / ncols;
  }

  const T & operator()(unsigned i, unsigned j) const { return data[i * ncols + j]; }
  T & operator()(unsigned i, unsigned j) { return data[i * ncols + j]; }

  typename std::vector<T>::iterator row_iterator(unsigned i) { 
    return data.begin() + i * ncols; 
  }
  
  typename std::vector<T>::const_iterator row_iterator(unsigned i) const { 
    return data.begin() + i * ncols; 
  }

  unsigned get_nrows() const { return nrows; }
  unsigned get_ncols() const { return ncols; }
  bool empty() const { return data.empty(); }
  void clear() { data.clear(); ncols = 0; nrows = 0; }

private:
  unsigned ncols;
  unsigned nrows;
  std::vector<T> data;
};

class HDF5UnstructuredTemperatureSource : public TemperatureSource
{
public:
  static constexpr unsigned NODES_PER_ELEM = 4;
  static constexpr unsigned DIM = 3;
  
  HDF5UnstructuredTemperatureSource(SPPARKS *spk);
  ~HDF5UnstructuredTemperatureSource();

  void setup_temperature_source(const std::vector<std::string> &args) override;
  double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  void update_temperatures(double dt, double simulation_time) override;
  bool needs_data_refresh(double simulation_time) override;
  void cleanup() override;
  std::string get_source_type() const override { return "hdf5_unstructured"; }

private:
  // HDF5 file management
  hid_t file_id;
  std::string filename;
  
  // Spatial domain information
  std::array<double, 3> x0;        // Origin of SPPARKS grid
  std::array<unsigned, 3> size;    // Grid dimensions
  double dx;                       // Grid spacing
  
  // Layer and time management
  std::vector<double> layer_times;
  unsigned active_layer;
  double current_time;
  
  // Interpolation cache structure
  struct InterpolationData {
    std::array<unsigned, 4> node_ids;    // Node IDs for tetrahedral interpolation
    std::array<double, 4> weights;       // Barycentric weights [w0, w1, w2, w3] where w0 = 1-w1-w2-w3
    bool valid;                          // True if site has valid interpolation data
    
    InterpolationData() : valid(false) {
      node_ids.fill(std::numeric_limits<unsigned>::max());
      weights.fill(0.0);
    }
  };
  
  // Current layer data
  std::vector<unsigned> data_counts;           // Number of time points per node
  array2D<double> times;                       // Time data for each node
  array2D<double> temperatures;                // Temperature data for each node
  std::vector<InterpolationData> interpolation_cache;  // Pre-computed interpolation data per SPPARKS site
  
  // Mesh data for current layer
  array2D<unsigned> elem_node;                 // Element-to-node connectivity
  array2D<double> node_coords;                 // Node coordinates
  std::vector<std::vector<double>> chunk_bboxes;    // Chunk bounding boxes
  std::vector<unsigned> node_offsets;          // Node offsets per chunk
  std::vector<unsigned> elem_offsets;          // Element offsets per chunk
  
  // Cache for element bounding boxes
  std::vector<std::vector<double>> elem_bboxes;
  
  // Grid bounding box
  std::vector<double> grid_bbox;
  
  // Helper methods
  void parse_arguments(const std::vector<std::string> &args);
  void open_hdf5_file();
  void read_layer_times();
  void setup_grid_bbox();
  void extract_domain_parameters();
  
  // Layer loading
  unsigned get_active_layer(double time) const;
  void load_layer(unsigned layer_idx);
  void read_chunk_info(hid_t group_id, std::vector<unsigned> &overlapping_chunks);
  void read_mesh_data(hid_t group_id, const std::vector<unsigned> &overlapping_chunks);
  void read_thermal_data(hid_t group_id, unsigned max_data_count);
  void compute_interpolation_weights();
  
  // Spatial interpolation
  std::vector<std::vector<double>> build_elem_bounding_boxes(
    const std::vector<unsigned> &overlapping_chunks) const;
  InterpolationData find_element_and_compute_weights(
    const std::vector<unsigned> &overlapping_chunks,
    const std::array<double, 3> &pt) const;
  std::array<double, 3> get_parametric_coordinates_of_point(
    const std::vector<std::vector<double>> &tet_coords, 
    const std::array<double, 3> &pt) const;
  
  // Utility methods
  bool do_boxes_overlap(const std::vector<double> &b1, const std::vector<double> &b2) const;
  bool point_in_bbox(const std::array<double, 3> &pt, const std::vector<double> &bbox) const;
  double interpolate_nodal_temperature(unsigned node_idx, double time) const;
  bool is_point_near_boundary(const std::array<double, 3> &pt) const;
  std::vector<double> get_expanded_bbox(double expansion_factor = 1.1) const;
  
  // HDF5 reading utilities
  void read_hdf5_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<double> &data);
  void read_hdf5_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<unsigned> &data);
  void read_hdf5_dataset_2d(hid_t group_id, const char* dataset_name, 
                           const std::vector<std::array<size_t, 2>> &row_slices,
                           const std::array<size_t, 2> &col_slice,
                           std::vector<double> &data);
  void read_hdf5_dataset_2d(hid_t group_id, const char* dataset_name, 
                           const std::vector<std::array<size_t, 2>> &row_slices,
                           const std::array<size_t, 2> &col_slice,
                           std::vector<unsigned> &data);
  void read_hdf5_hyperslab_1d(hid_t group_id, const char* dataset_name,
                             const std::vector<std::array<size_t, 2>> &row_slices,
                             std::vector<double> &data);
  void read_hdf5_hyperslab_1d(hid_t group_id, const char* dataset_name,
                             const std::vector<std::array<size_t, 2>> &row_slices,
                             std::vector<unsigned> &data);
};

}

#endif