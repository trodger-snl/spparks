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

#include "temperature_source_hdf5.h"
#include "app_additive_ext_temp_texture.h"
#include "domain.h"
#include "error.h"
#include "timer.h"
#include "comm_lattice.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5TemperatureSource::HDF5TemperatureSource(SPPARKS *spk) : TemperatureSource(spk)
{
  // Initialize defaults
  temp_file_string = "";
  dx = 1e-6;  // 1 micron default
  dt = 1e-6;  // 1 microsecond default
  
  // HDF5 handles
  hdf5_file_id = -1;
  hdf5_temp_dataset = -1;
  hdf5_time_dataset = -1;
  hdf5_count_dataset = -1;
  hdf5_file_open = false;
  
  // Chunked reading defaults
  chunk_time_window = 0.1; // 100ms time window per chunk
  current_chunk_start_time = 0.0;
  current_chunk_end_time = 0.0;
  prior_time = 0.0;
  
  // Initialize bounds checking
  bounds_check_mode = 0; // Exact match by default
  bounds_validated = false;
  hdf5_dims[0] = hdf5_dims[1] = hdf5_dims[2] = 0;
  hdf5_origin[0] = hdf5_origin[1] = hdf5_origin[2] = 0.0;
  
  app_additive = nullptr;
}

/* ---------------------------------------------------------------------- */

HDF5TemperatureSource::~HDF5TemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup HDF5 temperature source from command line arguments
   Expected format: filename dx dt [chunk_time_window]
------------------------------------------------------------------------- */

void HDF5TemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  if (args.size() < 3) {
    error->all(FLERR,"Insufficient arguments for HDF5 temperature source setup");
  }
  
  temp_file_string = args[0];
  
  try {
    dx = std::stod(args[1]);
    dt = std::stod(args[2]);
    
    if (args.size() >= 4) {
      chunk_time_window = std::stod(args[3]);
    }
  } catch (const std::exception &e) {
    error->all(FLERR,"Invalid numeric arguments in HDF5 temperature source setup");
  }
  
  if (dx <= 0.0 || dt <= 0.0) {
    error->all(FLERR,"dx and dt must be positive in HDF5 temperature source");
  }
  
  // Try to cast app to AppAdditiveExtTempTexture for coordinate access
  app_additive = dynamic_cast<AppAdditiveExtTempTexture*>(app);
  if (!app_additive) {
    error->all(FLERR,"HDF5 temperature source requires AppAdditiveExtTempTexture application");
  }
  
  // Initialize HDF5 data structures
  reduced_temperature_hdf_chunked();
  
  source_initialized = true;
  
  if (domain->me == 0) {
    print_source_info();
  }
}

/* ----------------------------------------------------------------------
   Get temperature at specified coordinates and time
------------------------------------------------------------------------- */

double HDF5TemperatureSource::get_temperature_at_xyz_and_time(double x, double y, double z, double time)
{
  check_initialization();
  
  // Convert coordinates to local indices
  int local_index = xyz_to_local(x, y, z);
  if (local_index < 0) {
    return ambient_temperature; // Outside domain
  }
  
  return temperature_time_interpolate(local_index, time);
}

/* ----------------------------------------------------------------------
   Optimized site-based temperature access for lattice applications
------------------------------------------------------------------------- */

double HDF5TemperatureSource::get_temperature_at_site(int site_index, double time)
{
  check_initialization();
  
  if (site_index < 0 || site_index >= app->nlocal + app->nghost) {
    error->one(FLERR,"Invalid site index in HDF5 temperature source");
  }
  
  if (!app_additive) {
    error->one(FLERR,"HDF5 temperature source not properly linked to additive app");
  }
  
  // Get site coordinates from the additive app
  double x = app_additive->xyz[site_index][0];
  double y = app_additive->xyz[site_index][1]; 
  double z = app_additive->xyz[site_index][2];
  
  return get_temperature_at_xyz_and_time(x, y, z, time);
}

/* ----------------------------------------------------------------------
   Update temperatures - check if new chunk needs to be loaded
------------------------------------------------------------------------- */

void HDF5TemperatureSource::update_temperatures(double dt, double simulation_time)
{
  if (needs_new_chunk(simulation_time)) {
    load_next_chunk();
  }
}

/* ----------------------------------------------------------------------
   Check if data refresh is needed based on current time
------------------------------------------------------------------------- */

bool HDF5TemperatureSource::needs_data_refresh(double simulation_time)
{
  return needs_new_chunk(simulation_time);
}

/* ----------------------------------------------------------------------
   Check if new chunk needs to be loaded
------------------------------------------------------------------------- */

bool HDF5TemperatureSource::needs_new_chunk(double simulation_time)
{
  return simulation_time > current_chunk_end_time;
}

/* ----------------------------------------------------------------------
   Initialize chunked HDF5 reading (extracted from app_additive_ext_temp_texture)
------------------------------------------------------------------------- */

void HDF5TemperatureSource::reduced_temperature_hdf_chunked()
{
  if (domain->me == 0) {
    std::cout << "Starting chunked HDF5 reading from: " << temp_file_string << std::endl;
    std::cout << "Opening HDF5 file..." << std::endl;
  }
  
  // Initialize chunked reading - open HDF5 file and datasets for reuse
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
  hdf5_file_id = H5Fopen(temp_file_string.c_str(), H5F_ACC_RDONLY, fapl_id);
  H5Pclose(fapl_id);
  
  if (hdf5_file_id < 0) {
    error->all(FLERR,"Cannot open HDF5 file for chunked reading");
  }
  
  // Open datasets for reuse
  hdf5_count_dataset = H5Dopen(hdf5_file_id, "data_counts", H5P_DEFAULT);
  hdf5_temp_dataset = H5Dopen(hdf5_file_id, "temperature", H5P_DEFAULT);
  hdf5_time_dataset = H5Dopen(hdf5_file_id, "time", H5P_DEFAULT);
  hdf5_file_open = true;
  
  // Validate simulation bounds against HDF5 data BEFORE loading any data
  if (!bounds_validated) {
    if (domain->me == 0) std::cout << "Validating simulation bounds..." << std::endl;
    validate_simulation_bounds();
  }
  
  if (domain->me == 0) std::cout << "Loading data counts array..." << std::endl;
  // Read data_counts array once (small and needed for all chunks)
  load_data_counts_array();
  if (domain->me == 0) std::cout << "Data counts loaded, initializing time cache..." << std::endl;
  
  // Initialize time cache for better performance
  initialize_time_cache();
  if (domain->me == 0) std::cout << "Time cache initialized, loading first chunk..." << std::endl;
  
  // Load the first chunk
  current_chunk_start_time = 0.0;
  current_chunk_end_time = chunk_time_window;
  load_next_chunk();
  if (domain->me == 0) std::cout << "First chunk loaded successfully" << std::endl;
  
  if (domain->me == 0) {
    std::cout << "Initial chunk loaded: [" << current_chunk_start_time 
              << ", " << current_chunk_end_time << ")" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Load data counts array (extracted from app_additive_ext_temp_texture)
------------------------------------------------------------------------- */

void HDF5TemperatureSource::load_data_counts_array()
{
  // Calculate the size using correct integer bounds  
  int xlo = (int)domain->subxlo;
  int xhi = (int)ceil(domain->subxhi);
  int ylo = (int)domain->subylo;
  int yhi = (int)ceil(domain->subyhi);
  int zlo = (int)domain->subzlo;
  int zhi = (int)ceil(domain->subzhi);
  
  int subdomain_x_size = xhi - xlo;
  int subdomain_y_size = yhi - ylo;
  int subdomain_z_size = zhi - zlo;
  int hyperslab_size = subdomain_x_size * subdomain_y_size * subdomain_z_size;
  
  std::vector<int> data_counts(hyperslab_size);
  
  // Setup parallel reading
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
  
  // Define hyperslab for data_counts using correct integer bounds
  hsize_t start_counts[3] = {(hsize_t)xlo, (hsize_t)ylo, (hsize_t)zlo};
  hsize_t count_counts[3] = {(hsize_t)subdomain_x_size, (hsize_t)subdomain_y_size, (hsize_t)subdomain_z_size};
  
  hid_t memspace_count_id = H5Screate_simple(3, count_counts, NULL);
  hid_t filespace_id = H5Dget_space(hdf5_count_dataset);
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start_counts, NULL, count_counts, NULL);
  
  hid_t data_type = H5Dget_type(hdf5_count_dataset);
  H5Dread(hdf5_count_dataset, data_type, memspace_count_id, filespace_id, plist_id, data_counts.data());
  
  // Clean up
  H5Sclose(filespace_id);
  H5Sclose(memspace_count_id);
  H5Tclose(data_type);
  H5Pclose(plist_id);
  
  // Convert to 3D array using correct integer bounds
  data_counts_array = convert_to_3d_array_with_range(data_counts, xlo, xhi, ylo, yhi, zlo, zhi);
  
  if (domain->me == 0) std::cout << "Data counts array loaded" << std::endl;
}

/* ----------------------------------------------------------------------
   Load next time chunk (extracted and simplified from app_additive_ext_temp_texture)
------------------------------------------------------------------------- */

void HDF5TemperatureSource::load_next_chunk()
{
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Clear existing data in queues
  for (int x = 0; x < (int)data_counts_array.size(); x++) {
    for (int y = 0; y < (int)data_counts_array[x].size(); y++) {
      for (int z = 0; z < (int)data_counts_array[x][y].size(); z++) {
        // Clear the queues completely
        while (!temp_in(x,y,z).empty()) temp_in(x,y,z).pop();
        while (!time_in(x,y,z).empty()) time_in(x,y,z).pop();
      }
    }
  }
  
  // Setup parallel reading
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
  hid_t data_type_temp = H5Dget_type(hdf5_temp_dataset);
  
  // Load data for each local site using cached time data
  for (int local_index = 0; local_index < app->nlocal; local_index++) {
    
    // Get site coordinates (this may need adjustment based on actual coordinate access)
    double *xyz = app_additive->xyz[local_index]; // Access coordinates from app
    
    // Use consistent coordinate calculation matching the array sizing
    int x_loc = (int)xyz[0] - (int)domain->subxlo;
    int y_loc = (int)xyz[1] - (int)domain->subylo;
    int z_loc = (int)xyz[2] - (int)domain->subzlo;
    int x = (int)xyz[0];
    int y = (int)xyz[1];
    int z = (int)xyz[2];
    
    // Enhanced bounds checking for chunk loading
    try {
      if (x_loc < 0 || x_loc >= (int)data_counts_array.size() ||
          y_loc < 0 || y_loc >= (int)data_counts_array[x_loc].size() ||
          z_loc < 0 || z_loc >= (int)data_counts_array[x_loc][y_loc].size()) {
        continue;
      }
    } catch (const std::exception& e) {
      continue;
    }
    
    int valid_count;
    try {
      valid_count = data_counts_array[x_loc][y_loc][z_loc];
    } catch (const std::exception& e) {
      continue;
    }
    
    // Load temperature data for this location if valid data exists
    if (valid_count > 0) {
      // HDF5 reading logic would go here
      // This is a simplified version - full implementation would include
      // the complex HDF5 hyperslab reading from the original code
    }
  }
  
  // Clean up
  H5Tclose(data_type_temp);
  H5Pclose(plist_id);
  
  // Update chunk timing
  current_chunk_start_time = current_chunk_end_time;
  current_chunk_end_time += chunk_time_window;
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  if (domain->me == 0) {
    std::cout << "Chunk loaded in " << duration.count() << " ms" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Temperature interpolation (extracted from app_additive_ext_temp_texture)
------------------------------------------------------------------------- */

double HDF5TemperatureSource::temperature_time_interpolate(int site_index, double simulation_time)
{
  // This is a simplified version of the original temperature_time_interpolate
  // The full implementation would include queue-based temporal interpolation
  
  // For now, return ambient temperature as placeholder
  // Full implementation would access temp_in and time_in queues for interpolation
  return ambient_temperature;
}

/* ----------------------------------------------------------------------
   Convert xyz coordinates to local index
------------------------------------------------------------------------- */

int HDF5TemperatureSource::xyz_to_local(double x, double y, double z)
{
  // Simplified coordinate conversion
  // Full implementation would match the original xyz_to_local method
  return -1; // Placeholder - indicates outside domain
}

/* ----------------------------------------------------------------------
   Initialize time cache (placeholder)
------------------------------------------------------------------------- */

void HDF5TemperatureSource::initialize_time_cache()
{
  // Initialize time cache data structures
  // Full implementation would set up cached_time_values and time_cache_loaded arrays
}

/* ----------------------------------------------------------------------
   Validate simulation bounds (placeholder)
------------------------------------------------------------------------- */

void HDF5TemperatureSource::validate_simulation_bounds()
{
  // Validate that simulation domain fits within HDF5 data bounds
  bounds_validated = true;
}

/* ----------------------------------------------------------------------
   Convert 1D vector to 3D array (from app_additive_ext_temp_texture)
------------------------------------------------------------------------- */

std::vector<std::vector<std::vector<int>>> HDF5TemperatureSource::convert_to_3d_array_with_range(
    std::vector<int>& inputVector, 
    int xStart, int xEnd, 
    int yStart, int yEnd, 
    int zStart, int zEnd)
{
  // Calculate the sizes of each dimension
  int xSize = xEnd - xStart;
  int ySize = yEnd - yStart;
  int zSize = zEnd - zStart;

  // Ensure the input vector has the correct size
  if ((int)inputVector.size() != xSize * ySize * zSize) {
    error->all(FLERR,"Input vector size does not match the specified dimensions in HDF5 source");
  }

  // Create the 3D array
  std::vector<std::vector<std::vector<int>>> outputArray(
      xSize, 
      std::vector<std::vector<int>>(ySize, std::vector<int>(zSize))
  );

  // Fill the 3D array using the input vector
  for (int x = xStart; x < xEnd; x++) {
    for (int y = yStart; y < yEnd; y++) {
      for (int z = zStart; z < zEnd; z++) {
        // Compute the index in the 1D vector
        int index = (x - xStart) * (ySize * zSize) + (y - yStart) * zSize + (z - zStart);
        outputArray[x - xStart][y - yStart][z - zStart] = inputVector[index];
      }
    }
  }
  return outputArray;
}

/* ----------------------------------------------------------------------
   Print source information
------------------------------------------------------------------------- */

void HDF5TemperatureSource::print_source_info() const
{
  std::cout << "HDF5 Temperature Source Configuration:" << std::endl;
  std::cout << "  File: " << temp_file_string << std::endl;
  std::cout << "  Lattice Spacing (dx): " << dx << " m" << std::endl;
  std::cout << "  Time Step (dt): " << dt << " s" << std::endl;
  std::cout << "  Chunk Time Window: " << chunk_time_window << " s" << std::endl;
  std::cout << "  Ambient Temperature: " << ambient_temperature << " K" << std::endl;
}

/* ----------------------------------------------------------------------
   Close HDF5 file and cleanup
------------------------------------------------------------------------- */

void HDF5TemperatureSource::close_hdf5_file()
{
  if (hdf5_file_open) {
    if (hdf5_count_dataset >= 0) H5Dclose(hdf5_count_dataset);
    if (hdf5_temp_dataset >= 0) H5Dclose(hdf5_temp_dataset);
    if (hdf5_time_dataset >= 0) H5Dclose(hdf5_time_dataset);
    if (hdf5_file_id >= 0) H5Fclose(hdf5_file_id);
    hdf5_file_open = false;
  }
}

/* ----------------------------------------------------------------------
   Cleanup resources
------------------------------------------------------------------------- */

void HDF5TemperatureSource::cleanup()
{
  close_hdf5_file();
  data_counts_array.clear();
  cached_time_values.clear();
  time_cache_loaded.clear();
  source_initialized = false;
}