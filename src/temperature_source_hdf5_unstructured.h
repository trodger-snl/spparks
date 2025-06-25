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
#include <memory>
#include <array>
#include <vector>
#include <string>

namespace HighFive {
  class File;
}

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   HDF5-based unstructured mesh temperature field source
   
   Reads pre-computed temperature field data from HDF5 files containing
   unstructured tetrahedral mesh data with:
   - Layer-based temporal organization
   - Chunk-based spatial organization for efficient parallel access
   - Node-based temperature time series
   - Tetrahedral element connectivity
   - MPI-parallel data access with subdomain filtering
   
   Data format matches reduced thermal output from external AM solvers.
------------------------------------------------------------------------- */

// Helper class for 2D array storage (from reference implementation)
template <typename T>
class Array2D {
public:
  Array2D() {}
  Array2D(unsigned ncols_, std::vector<T>&& data_) : 
    ncols(ncols_), data(std::move(data_)) {
    if (data.size() % ncols != 0) 
      throw std::runtime_error("Data size must be multiple of number of columns");
    nrows = data.size() / ncols;
  }

  const T& operator()(unsigned i, unsigned j) const { return data[i*ncols + j]; }
  T& operator()(unsigned i, unsigned j) { return data[i*ncols + j]; }
  
  auto row_iterator(unsigned i) { return data.begin() + i*ncols; }
  const auto row_iterator(unsigned i) const { return data.begin() + i*ncols; }

private:
  unsigned ncols;
  unsigned nrows;
  std::vector<T> data;
};

class HDF5UnstructuredTemperatureSource : public TemperatureSource {
public:
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

  // Constants from reference implementation
  static constexpr unsigned NODES_PER_ELEM = 4;  // Tetrahedral elements
  static constexpr unsigned DIM = 3;             // 3D space

  // Fast-forward capability methods
  bool all_temperatures_below_threshold(double time);
  bool has_significant_thermal_activity(double time);
  bool has_significant_thermal_activity_hdf5_nodes(double time);
  double find_next_active_time(double start_time, double end_time);
  double find_next_active_time_sequential(double start_time, double end_time);
  double get_fast_forward_threshold() const { return threshold_temp; }
  double get_dx() const { return dx; }

  // Site-based temperature access optimized for lattice
  virtual double get_temperature_at_site(int site_index, double time) override;
  
  // Element cache for performance optimization
  struct ElementCache {
    std::array<unsigned, 4> nodeIndices;
    std::array<double, 3> weights;
    bool valid;
    
    ElementCache() : nodeIndices{0,0,0,0}, weights{0.0,0.0,0.0}, valid(false) {}
  };

private:
  // File and coordinate parameters
  std::string filename;
  double dx;  // Grid spacing in meters
  double threshold_temp;  // Threshold temperature for fast-forward (K)
  double default_temp;    // Default/ambient temperature (K)
  int bounds_check_mode;  // 0 = exact, 1 = subvolume

  
  // HDF5 file handle
  std::shared_ptr<HighFive::File> file;
  
  // Layer management
  std::vector<double> layerTimes;
  double current_time;
  unsigned active_layer;
  
  // Data storage for current layer
  std::vector<unsigned> dataCounts;
  Array2D<double> times;
  Array2D<double> temperatures;
  Array2D<unsigned> elemNode;
  Array2D<double> nodeCoords;
  
  // Chunk data for spatial queries
  std::vector<std::vector<double>> chunk_bboxes;
  std::vector<unsigned> selected_chunks;
  std::vector<unsigned> node_offsets;
  std::vector<unsigned> elem_offsets;
  std::vector<std::vector<double>> elem_bboxes;
  
  // Helper methods from reference implementation
  void load_layer(unsigned layerIdx);
  
  // Anonymous namespace functions (now as private methods)
  std::array<double, 3> get_parametric_coordinates_of_point(
    const std::vector<std::vector<double>>& tet_coords, 
    const std::array<double, 3>& pt) const;
  
  unsigned get_active_layer(double t) const;
  
  bool do_boxes_overlap(const std::vector<double>& b1, 
                       const std::vector<double>& b2) const;
  
  bool point_in_bbox(const std::array<double, 3>& pt, 
                    const std::vector<double>& bbox) const;
  
  std::pair<std::array<unsigned, 4>, std::array<double, 3>> 
  find_element_point_is_in(const std::vector<unsigned>& selectedChunks,
                          const std::vector<std::vector<double>>& chunkBboxes,
                          const std::vector<unsigned>& nodeOffsets,
                          const std::vector<unsigned>& elemOffsets,
                          const Array2D<unsigned>& elemNode,
                          const Array2D<double>& nodeCoords,
                          const std::vector<std::vector<double>>& elemBboxes,
                          const std::array<double, 3>& pt) const;
  
  std::vector<std::vector<double>> build_elem_bounding_boxes(
    unsigned nChunks,
    const std::vector<unsigned>& nodeOffsets,
    const std::vector<unsigned>& elemOffsets,
    const Array2D<unsigned>& elemNode,
    const Array2D<double>& nodeCoords) const;
  
  // Check if all nodes of an element are below threshold temperature
  bool all_nodes_below_threshold(
    const std::array<unsigned, 4>& nodeIndices, 
    double time) const;
  
  // Element cache management
  mutable std::vector<ElementCache> site_element_cache;
  mutable bool cache_valid;
  
  // Build element cache for all sites
  void build_site_element_cache() const;
  
  // Get cached element for a site
  const ElementCache& get_cached_element(int site_index) const;
};

}

#endif