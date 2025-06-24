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
#include <unordered_map>

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
  
  // Fast-forward capability
  double find_next_active_time(double current_time, double max_search_time = -1.0);
  bool all_temperatures_below_threshold(double time);
  void set_fast_forward_threshold(double threshold) { fast_forward_threshold = threshold; }
  double get_fast_forward_threshold() const { return fast_forward_threshold; }
  
  // Accessor for lattice spacing
  double get_dx() const { return dx; }

private:
  // HDF5 file management
  hid_t file_id;
  std::string filename;
  
  // Spatial domain information
  std::vector<double> spparks_domain_bbox;  // SPPARKS domain bounding box [xmin,ymin,zmin,xmax,ymax,zmax]
  double dx;                                // Reference grid spacing (for boundary expansion)
  
  // Layer and time management
  std::vector<double> layer_times;
  unsigned active_layer;
  double current_time;
  
  // Fast-forward capability
  double fast_forward_threshold;                    // Temperature threshold for fast-forward (default 800K)
  double fast_forward_check_interval;              // How often to check temperatures for fast-forward
  
  // Spatial caching for fast element lookup
  struct SpatialElement {
    unsigned element_id;                     // Local element ID in our filtered mesh
    std::vector<double> bbox;                // Element bounding box [xmin,ymin,zmin,xmax,ymax,zmax]
    std::array<unsigned, 4> node_ids;        // Global node IDs for this element
  };
  
  // Variable-length thermal data storage
  struct NodalThermalData {
    std::vector<double> times;        // Time points for this node
    std::vector<double> temperatures; // Temperature values for this node
    
    NodalThermalData() = default;
    NodalThermalData(std::vector<double>&& t, std::vector<double>&& temp) 
      : times(std::move(t)), temperatures(std::move(temp)) {}
  };
  
  // Current layer data (only for nodes in overlapping elements)
  std::unordered_map<unsigned, unsigned> global_to_local_node_map;  // Maps global node IDs to local indices
  std::vector<NodalThermalData> nodal_data;                        // Thermal data for filtered nodes only
  std::vector<SpatialElement> spatial_elements;                    // Filtered mesh elements overlapping SPPARKS domain
  
  // Filtered mesh data for current layer (only overlapping SPPARKS domain)
  std::vector<std::array<double, 3>> filtered_node_coords;  // Coordinates of nodes in overlapping elements
  std::vector<std::vector<double>> chunk_bboxes;            // Original chunk bounding boxes from HDF5
  
  // Helper methods
  void parse_arguments(const std::vector<std::string> &args);
  void open_hdf5_file();
  void read_layer_times();
  void setup_spparks_domain_bbox();
  void extract_domain_parameters();
  
  // Layer loading
  unsigned get_active_layer(double time) const;
  void load_layer(unsigned layer_idx);
  void read_chunk_info(hid_t group_id, std::vector<unsigned> &overlapping_chunks);
  void read_and_filter_mesh_data(hid_t group_id, const std::vector<unsigned> &overlapping_chunks);
  void read_filtered_thermal_data(hid_t group_id);
  void build_spatial_elements();
  
  // On-demand spatial interpolation
  bool find_element_and_interpolate(
    const std::array<double, 3> &pt, double time, double &temperature) const;
  std::array<double, 3> get_parametric_coordinates_of_point(
    const std::vector<std::vector<double>> &tet_coords, 
    const std::array<double, 3> &pt) const;
  double interpolate_in_element(
    const SpatialElement &element, const std::array<double, 3> &pt, double time) const;
  
  // Utility methods
  bool do_boxes_overlap(const std::vector<double> &b1, const std::vector<double> &b2) const;
  bool point_in_bbox(const std::array<double, 3> &pt, const std::vector<double> &bbox) const;
  double interpolate_nodal_temperature(unsigned local_node_idx, double time) const;
  std::vector<double> get_expanded_spparks_bbox(double expansion_factor = 1.1) const;
  bool element_overlaps_spparks_domain(const std::vector<double> &elem_bbox) const;
  
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
  void read_variable_length_thermal_data(hid_t group_id, const char* dataset_name,
                                        const std::vector<unsigned> &filtered_global_node_ids,
                                        std::vector<NodalThermalData> &nodal_data);
};

}

#endif