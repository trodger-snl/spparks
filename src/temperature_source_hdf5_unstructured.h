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

#ifndef SPK_TEMPERATURE_SOURCE_HDF5_UNSTRUCTURED_H
#define SPK_TEMPERATURE_SOURCE_HDF5_UNSTRUCTURED_H

#include "temperature_source.h"
#include "hdf5.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   HDF5 Unstructured temperature source
   
   Reads unstructured mesh temperature data from HDF5 files with support for:
   - Multi-layer additive manufacturing simulation data
   - Tetrahedral element interpolation
   - Temporal interpolation between time points
   - On-demand layer loading for memory efficiency
   
   This class adapts the functionality from resourcesfornewformat/fAM_Read_Reduced_Output.*
   to work with SPPARKS' modular temperature source architecture using standard HDF5 C API.
------------------------------------------------------------------------- */

class HDF5UnstructuredTemperatureSource : public TemperatureSource {
 public:
  static constexpr unsigned NODES_PER_ELEM = 4;
  static constexpr unsigned DIM = 3;
  
  HDF5UnstructuredTemperatureSource(class SPPARKS *);
  virtual ~HDF5UnstructuredTemperatureSource();

  // Required virtual methods from base class
  virtual void setup_temperature_source(const std::vector<std::string> &args) override;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  virtual void update_temperatures(double dt, double simulation_time) override;
  virtual bool needs_data_refresh(double simulation_time) override;
  virtual void cleanup() override;
  virtual std::string get_source_type() const override { return "hdf5_unstructured"; }
  virtual void print_source_info() const override;

  // Additional methods for fast-forward optimization
  bool all_temperatures_below_threshold(double time);
  double find_next_active_time(double start_time, double end_time);
  double find_next_active_time_sequential(double start_time, double end_time);
  double get_fast_forward_threshold() const { return fast_forward_threshold_; }
  double get_dx() const { return dx_; }

  // Thermal window detection methods
  std::pair<double, double> find_thermal_window();
  bool has_temperatures_above_threshold_at_time(double time);
  std::vector<double> get_active_time_points();

 private:
  // 2D array template for efficient data storage
  template <typename T>
  class Array2D {
   public:
    Array2D() : ncols_(0), nrows_(0) {}
    Array2D(unsigned ncols, std::vector<T>&& data) : ncols_(ncols), data_(std::move(data)) {
      if (data_.size() % ncols != 0) {
        throw std::runtime_error("Data size must be multiple of number of columns");
      }
      nrows_ = data_.size() / ncols;
    }
    
    const T& operator()(unsigned i, unsigned j) const { return data_[i * ncols_ + j]; }
    T& operator()(unsigned i, unsigned j) { return data_[i * ncols_ + j]; }
    
    unsigned rows() const { return nrows_; }
    unsigned cols() const { return ncols_; }
    
    // Get iterator to start of row i
    auto row_iterator(unsigned i) { return data_.begin() + i * ncols_; }
    
   private:
    unsigned ncols_, nrows_;
    std::vector<T> data_;
  };

  // HDF5 file parameters
  std::string filename_;
  std::array<double, 3> x0_;  // SPPARKS domain origin
  std::array<unsigned, 3> size_;  // SPPARKS domain size in voxels
  double dx_;  // SPPARKS lattice spacing
  double fast_forward_threshold_;  // Temperature threshold for fast-forward optimization
  bool enable_thermal_window_;  // Whether to run thermal window pre-calculation
  
  // HDF5 file handle
  hid_t file_id_;
  bool file_open_;
  
  // Layer management
  std::vector<double> layer_times_;
  std::vector<unsigned> data_counts_;
  unsigned active_layer_;
  double current_time_;
  
  // Current layer data
  Array2D<double> times_;
  Array2D<double> temperatures_;
  std::vector<std::array<unsigned, 4>> node_ids_;
  std::vector<std::array<double, 3>> weights_;
  std::unordered_map<unsigned, unsigned> global_to_local_node_map_;  // Maps global node ID to local Array2D row index
  
  // Private helper methods
  void parse_setup_arguments(const std::vector<std::string> &args);
  void open_hdf5_file();
  void close_hdf5_file();
  void read_layer_times();
  void load_layer(unsigned layer_idx);
  unsigned get_active_layer(double time) const;
  
  // Geometric calculations
  std::array<double, 3> get_parametric_coordinates_of_point(
    const std::vector<std::vector<double>>& tet_coords, 
    const std::array<double, 3>& pt
  ) const;
  
  bool do_boxes_overlap(const std::vector<double>& b1, const std::vector<double>& b2) const;
  bool point_in_bbox(const std::array<double, 3>& pt, const std::vector<double>& bbox) const;
  
  std::pair<std::array<unsigned, 4>, std::array<double, 3>> find_element_point_is_in(
    const std::vector<unsigned>& selected_chunks,
    const std::vector<std::vector<double>>& chunk_bboxes,
    const std::vector<unsigned>& node_offsets,
    const std::vector<unsigned>& elem_offsets,
    const Array2D<unsigned>& elem_node,
    const Array2D<double>& node_coords,
    const std::vector<std::vector<double>>& elem_bboxes,
    const std::array<double, 3>& pt
  ) const;
  
  std::vector<std::vector<double>> build_elem_bounding_boxes(
    unsigned n_chunks,
    const std::vector<unsigned>& node_offsets,
    const std::vector<unsigned>& elem_offsets,
    const Array2D<unsigned>& elem_node,
    const Array2D<double>& node_coords
  ) const;
  
  // HDF5 helper functions
  void read_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<double>& data);
  void read_dataset_1d(hid_t group_id, const char* dataset_name, std::vector<unsigned>& data);
  void read_dataset_2d(hid_t group_id, const char* dataset_name, std::vector<std::vector<double>>& data);
  void read_partial_dataset_2d(hid_t group_id, const char* dataset_name, 
                               const std::vector<std::array<size_t, 2>>& row_slices,
                               const std::array<size_t, 2>& col_slice,
                               std::vector<double>& data);
};

}

#endif