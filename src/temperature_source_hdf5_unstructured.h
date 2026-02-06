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
#include <stdexcept>
#include <chrono>

#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#endif

namespace HighFive {
  class File;
  class Group;
  class HyperSlab;
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
  Array2D() : ncols(0), nrows(0) {}
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

// CSR-style compressed 2D array — stores ragged rows without padding.
// Each row i has (offsets[i+1] - offsets[i]) elements.
template <typename T>
class CompressedArray2D {
public:
  CompressedArray2D() {}

  // Construct from rectangular Array2D data by compacting rows in-place.
  // ncols_rect: number of columns in the source rectangular layout
  // rect_data:  row-major rectangular data (nrows * ncols_rect elements)
  // row_lengths: actual element count per row (dataCounts)
  CompressedArray2D(unsigned ncols_rect,
                    std::vector<T>&& rect_data,
                    const std::vector<unsigned>& row_lengths)
  {
    unsigned nrows = row_lengths.size();
    offsets.resize(nrows + 1);
    offsets[0] = 0;
    for (unsigned i = 0; i < nrows; i++)
      offsets[i + 1] = offsets[i] + row_lengths[i];

    size_t total = offsets[nrows];
    data.resize(total);
    for (unsigned i = 0; i < nrows; i++) {
      std::copy(rect_data.begin() + i * ncols_rect,
                rect_data.begin() + i * ncols_rect + row_lengths[i],
                data.begin() + offsets[i]);
    }
  }

  // Construct directly from flat CSR data (zero-copy move).
  CompressedArray2D(std::vector<T>&& flat_data,
                    std::vector<size_t>&& row_offsets)
    : data(std::move(flat_data)), offsets(std::move(row_offsets)) {}

  const T& operator()(unsigned i, unsigned j) const { return data[offsets[i] + j]; }
  T& operator()(unsigned i, unsigned j) { return data[offsets[i] + j]; }

  auto row_iterator(unsigned i) { return data.begin() + offsets[i]; }
  const auto row_iterator(unsigned i) const { return data.begin() + offsets[i]; }

  size_t total_elements() const { return data.size(); }

private:
  std::vector<T> data;
  std::vector<size_t> offsets;
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
  double get_fast_forward_threshold() const { return threshold_temp; }
  double get_dx() const { return dx; }
  
  // Time query interface implementation
  virtual bool supports_time_queries() const override { return true; }
  virtual double get_next_time_with_temperature(double current_time, double threshold_temp) override;
  
  // Thermal interval data structure
  struct ThermalInterval {
    double start_time;
    double end_time;
    
    ThermalInterval(double start, double end) : start_time(start), end_time(end) {}
  };

  // Site-based temperature access optimized for lattice
  virtual double get_temperature_at_site(int site_index, double time) override;

  // High-performance batch temperature access:
  // 1. Call prepare_for_timestep() once at start of timestep
  // 2. Call get_temperature_at_site_fast() for each site (no redundant checks)
  void prepare_for_timestep(double time);
  inline double get_temperature_at_site_fast(int site_index) const {
    const ElementCache& cache = site_element_cache[site_index];
    if (!cache.valid) return default_temp;
    return cached_nodal_temps[cache.nodeIndices[0]] +
           cache.weights[0] * (cached_nodal_temps[cache.nodeIndices[1]] - cached_nodal_temps[cache.nodeIndices[0]]) +
           cache.weights[1] * (cached_nodal_temps[cache.nodeIndices[2]] - cached_nodal_temps[cache.nodeIndices[0]]) +
           cache.weights[2] * (cached_nodal_temps[cache.nodeIndices[3]] - cached_nodal_temps[cache.nodeIndices[0]]);
  }

  // Element cache for performance optimization
  struct ElementCache {
    std::array<unsigned, 4> nodeIndices;
    std::array<double, 3> weights;
    bool valid;
    
    ElementCache() : nodeIndices{0,0,0,0}, weights{0.0,0.0,0.0}, valid(false) {}
  };

protected:
  // Data storage accessible to subclasses for CSR override
  std::vector<unsigned> dataCounts;
  CompressedArray2D<double> times;
  CompressedArray2D<double> temperatures;

  // HDF5 file handle
  std::shared_ptr<HighFive::File> file;

  // Parallel HDF5 state
#ifdef H5_HAVE_PARALLEL
  bool use_parallel_hdf5;
#endif

  // Virtual hook: read time/temperature arrays for the loaded node slices.
  // Base implementation reads rectangular 2D datasets and compacts via CompressedArray2D.
  // CSR subclass overrides to read flat 1D data + nodeOffsets directly.
  virtual void read_time_temperature_data(
      HighFive::Group& grp,
      const std::vector<std::array<size_t, 2>>& nodeSlices);

  // Hyperslab construction helpers (promoted from lambdas for subclass access)
  static HighFive::HyperSlab build_hyperslab(
      const std::vector<std::array<size_t, 2>>& slices,
      const std::array<size_t, 2>& cols);

  static HighFive::HyperSlab build_hyperslab_1d(
      const std::vector<std::array<size_t, 2>>& slices);

private:
  // File and coordinate parameters
  std::string filename;
  double dx;  // Grid spacing in meters
  double threshold_temp;  // Threshold temperature for fast-forward (K)
  double default_temp;    // Default/ambient temperature (K)
  int bounds_check_mode;  // 0 = exact, 1 = subvolume
  double grid_cell_size_multiplier;  // Spatial grid cell size = dx * multiplier (default 50)

  // Layer management
  std::vector<double> layerTimes;
  double current_time;
  unsigned active_layer;

  // Data storage for current layer (non-CSR members stay private)
  Array2D<unsigned> elemNode;
  Array2D<double> nodeCoords;

  // Chunk data for spatial queries
  std::vector<std::array<double, 6>> chunk_bboxes;  // [xmin,ymin,zmin,xmax,ymax,zmax]
  std::vector<unsigned> selected_chunks;
  std::vector<unsigned> node_offsets;
  std::vector<unsigned> elem_offsets;
  std::vector<std::array<double, 6>> elem_bboxes;  // [xmin,ymin,zmin,xmax,ymax,zmax]

  // Spatial acceleration grid for fast element lookup
  // Uses CSR-style flat storage to minimize memory overhead
  struct SpatialGrid {
    std::vector<unsigned> cell_elements;       // Flat array of element indices (CSR data)
    std::vector<size_t> cell_offsets;          // Offset into cell_elements for each cell (CSR offsets)
    std::vector<unsigned> elem_to_chunk;       // Maps element index to chunk index
    std::array<double, 3> origin;              // Grid origin (min corner)
    std::array<double, 3> cell_size;           // Size of each cell
    std::array<unsigned, 3> dims;              // Number of cells in each dimension
    bool valid;

    SpatialGrid() : valid(false) {}

    // Convert point to cell index, returns -1 if outside grid
    int point_to_cell(const std::array<double, 3>& pt) const {
      if (!valid) return -1;
      int ix = static_cast<int>((pt[0] - origin[0]) / cell_size[0]);
      int iy = static_cast<int>((pt[1] - origin[1]) / cell_size[1]);
      int iz = static_cast<int>((pt[2] - origin[2]) / cell_size[2]);
      if (ix < 0 || ix >= static_cast<int>(dims[0]) ||
          iy < 0 || iy >= static_cast<int>(dims[1]) ||
          iz < 0 || iz >= static_cast<int>(dims[2])) {
        return -1;
      }
      return ix + iy * dims[0] + iz * dims[0] * dims[1];
    }

    // Get elements in a cell (CSR-style access)
    const unsigned* cell_begin(size_t cell_idx) const {
      return cell_elements.data() + cell_offsets[cell_idx];
    }
    const unsigned* cell_end(size_t cell_idx) const {
      return cell_elements.data() + cell_offsets[cell_idx + 1];
    }
    size_t cell_count(size_t cell_idx) const {
      return cell_offsets[cell_idx + 1] - cell_offsets[cell_idx];
    }
  };
  SpatialGrid spatial_grid;

  // Timing statistics for performance monitoring
  double total_layer_load_time;
  int layer_load_count;

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
                    const std::array<double, 6>& bbox) const;

  std::pair<std::array<unsigned, 4>, std::array<double, 3>>
  find_element_point_is_in(const std::vector<unsigned>& selectedChunks,
                          const std::vector<std::array<double, 6>>& chunkBboxes,
                          const std::vector<unsigned>& nodeOffsets,
                          const std::vector<unsigned>& elemOffsets,
                          const Array2D<unsigned>& elemNode,
                          const Array2D<double>& nodeCoords,
                          const std::vector<std::array<double, 6>>& elemBboxes,
                          const std::array<double, 3>& pt) const;

  std::vector<std::array<double, 6>> build_elem_bounding_boxes(
    unsigned nChunks,
    const std::vector<unsigned>& nodeOffsets,
    const std::vector<unsigned>& elemOffsets,
    const Array2D<unsigned>& elemNode,
    const Array2D<double>& nodeCoords) const;

  void build_spatial_grid(double target_cell_size);

  // Thermal interval management for efficient time queries
  std::vector<std::vector<ThermalInterval>> layer_thermal_intervals;

  void compute_thermal_intervals_for_layer(unsigned layerIdx, double threshold_temp);
  std::vector<ThermalInterval> merge_overlapping_intervals(const std::vector<ThermalInterval>& intervals) const;
  bool is_point_in_spparks_domain(double x, double y, double z) const;

  // Element cache management
  mutable std::vector<ElementCache> site_element_cache;
  mutable std::vector<unsigned> active_node_indices;  // Nodes actually used by cached sites
  mutable bool cache_valid;

  // Build element cache for all sites
  void build_site_element_cache() const;

  // Get cached element for a site
  const ElementCache& get_cached_element(int site_index) const;

  // Nodal temperature cache - precomputed temps at all loaded nodes for current time
  // Eliminates redundant time interpolation (36x reduction in time searches)
  mutable std::vector<double> cached_nodal_temps;
  mutable double cached_nodal_time;
  mutable bool nodal_cache_valid;

  // Precompute interpolated temperatures at all loaded nodes for given time
  void precompute_nodal_temperatures(double time) const;
};

}

#endif