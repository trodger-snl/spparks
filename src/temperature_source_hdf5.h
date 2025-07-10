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

/* ----------------------------------------------------------------------
   WARNING: This temperature source is UNTESTED with the current modular
   temperature source system and is unlikely to work in its current state.
   
   Only temperature_source_hdf5_unstructured.cpp has been tested and 
   validated with the modular system. Use of this temperature source
   may result in compilation errors or runtime failures.
   
   For working temperature sources, use:
   - hdf5_unstructured (tested and validated)
------------------------------------------------------------------------- */

#ifndef SPK_TEMPERATURE_SOURCE_HDF5_H
#define SPK_TEMPERATURE_SOURCE_HDF5_H

#include "temperature_source.h"
#include "temperatureQueues.h"
#include "hdf5.h"
#include <vector>
#include <string>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   HDF5-based external temperature field source
   
   Reads pre-computed temperature field data from HDF5 files with support for:
   - Chunked temporal data loading for memory efficiency
   - Spatial interpolation across lattice sites
   - Temporal interpolation between data points
   - MPI-parallel data access
   - Bounds validation and error checking
   
   This class extracts and modularizes the HDF5 temperature handling
   from app_additive_ext_temp_texture.
------------------------------------------------------------------------- */

class HDF5TemperatureSource : public TemperatureSource {
 public:
  HDF5TemperatureSource(class SPPARKS *);
  virtual ~HDF5TemperatureSource();

  // Required virtual methods from base class
  virtual void setup_temperature_source(const std::vector<std::string> &args) override;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  virtual void update_temperatures(double dt, double simulation_time) override;
  virtual bool needs_data_refresh(double simulation_time) override;
  virtual void cleanup() override;
  virtual std::string get_source_type() const override { return "hdf5"; }
  virtual void print_source_info() const override;

  // HDF5-specific methods
  void load_next_chunk();
  void close_hdf5_file();
  bool needs_new_chunk(double simulation_time);
  void initialize_time_cache();
  void validate_simulation_bounds();
  
  // Site-based temperature access optimized for lattice
  virtual double get_temperature_at_site(int site_index, double time) override;

 private:
  // HDF5 file parameters
  std::string temp_file_string;
  double dx;  // Source lattice spacing (m)
  double dt;  // Source timestep (s)
  
  // HDF5 file handles
  hid_t hdf5_file_id;
  hid_t hdf5_temp_dataset;
  hid_t hdf5_time_dataset;
  hid_t hdf5_count_dataset;
  bool hdf5_file_open;
  
  // Chunked reading parameters
  double chunk_time_window;
  double current_chunk_start_time;
  double current_chunk_end_time;
  
  // Temperature data storage (from temperatureQueues.h)
  DoubleQueueContainer temp_in;
  DoubleQueueContainer time_in;
  double prior_time;
  
  // Data structure information
  std::vector<std::vector<std::vector<int>>> data_counts_array;
  
  // Time cache for performance
  std::vector<std::vector<std::vector<std::vector<double>>>> cached_time_values;
  std::vector<std::vector<std::vector<bool>>> time_cache_loaded;
  
  // Bounds checking
  int bounds_check_mode; // 0 = exact match, 1 = subvolume
  hsize_t hdf5_dims[3];  // HDF5 data dimensions
  double hdf5_origin[3]; // HDF5 data origin
  bool bounds_validated;
  
  // App interface (for accessing site coordinates)
  class AppAdditiveExtTempTexture *app_additive;
  
  // Private helper methods
  void parse_setup_arguments(const std::vector<std::string> &args);
  void reduced_temperature_hdf_chunked();
  void load_data_counts_array();
  void get_hdf5_dimensions();
  void fill_missing_temperature_data();
  
  double temperature_time_interpolate(int site_index, double simulation_time);
  int xyz_to_local(double x, double y, double z);
  
  // Convert between coordinate systems
  std::vector<std::vector<std::vector<int>>> convert_to_3d_array_with_range(
    std::vector<int>& inputVector, 
    int xStart, int xEnd, 
    int yStart, int yEnd, 
    int zStart, int zEnd
  );
};

}

#endif