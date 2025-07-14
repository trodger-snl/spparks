/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
   
   Contributing author: Theron Rodgers

   Additive manufacturing application with external temperature field and crystallographic texture modeling.
   Uses quaternion-based representation for crystal orientations and implements experimentally-derived
   parameters for nucleation and grain growth during solidification processes.
   
   Key features:
   - HDF5-based external temperature field reading with chunked data loading
   - Monte Carlo nucleation model with temperature-dependent kinetics
   - Quaternion-based crystallographic orientation tracking
   - Arrhenius mobility model for grain boundary kinetics
   - Epitaxial growth and competitive nucleation mechanisms
   - Support for both 2D and 3D additive manufacturing simulations
   
 ------------------------------------------------------------------------- */
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <limits>
#include "math.h"
#include "math_const.h"
#include "app_additive_ext_temp_texture.h"
#include "comm_lattice.h"
#include "lattice.h"
#include "timer.h"
#include "random_park.h"
#include "error.h"
#include "domain.h"
#include "hdf5.h"
#include "potts_quaternion/cubic_symmetries.h"
#include "potts_quaternion/disorientation.h"
#include "potts_quaternion/hcp_symmetries.h"
#include "potts_quaternion/quaternion.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_hdf5_unstructured.h"


using namespace SPPARKS_NS;
using namespace MathConst;


/* ---------------------------------------------------------------------- */

static char** construct_parent_args(char **arg) {
  static char* parent_args[3];
  static char app_name[] = "additive_temperature_texture";
  static char crystal_type[] = "cubic";
  
  parent_args[0] = app_name;
  parent_args[1] = arg[1]; // nspins
  parent_args[2] = crystal_type; // crystal structure
  
  return parent_args;
}

/* ---------------------------------------------------------------------- */

AppAdditiveExtTempTexture::AppAdditiveExtTempTexture(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternion(spk,3,construct_parent_args(arg))
{

    if (narg != 6  )
    error->all(FLERR,"Illegal app_style command");

    nspins = atoi(arg[1]);
    temp_file_string = arg[2]; //The name of the temperature file
    dx = atof(arg[3]); //The source lattice spacing ( in m)
    dt = atof(arg[4]); //The source timestep (in seconds)
    nrefine = atoi(arg[5]); //How many refinement MC steps to perform after a site solidifies
    
    //I think we need all of these variables still!
    ndouble = 9;
    allow_app_update = 1;
    app_update_only = 1; //Skip solid-state growth for now.
    ninteger = 2;
    total_time = 0;
    sites = unique = NULL;
    unique_dot = NULL;
    neigh_dist = NULL;
    nucleation_flags = NULL;
    nucleation_temps = NULL;
    nucleation_sizes = NULL;

    //Set default values    
    tl = 1723;
    ts = 1673;
    t_cool_max = tl - ts;
    no = 1e15;
    tc = 5;
    tsig = 3;
    size_norm = pow(dx,3) * 2;
    size_sig = pow(dx,3);
    solid_front_length = 4;
    // Set default values for coefficient arrays
    solid_front_coeffs = new double[4];    
    solid_front_coeffs[0] = 1.091e-5;
    solid_front_coeffs[1] = -2.034e-4;
    solid_front_coeffs[2] = 2.74e-3;
    solid_front_coeffs[3] = 1.151e-4;
    //Default texture parameters
    c1 = 0.5;
    c2 = 0.5;
    c3 = 2.5;
    time_step = dt;
    t_room = 300;
    max_misorient = 0;
    mis_thresh = 10.0 * MY_PI / 180.0; // Default 10 degrees converted to radians
    misorient_alpha = 5.0; // Default exponential steepness parameter
    
    // Debug defaults
    normal_finder_debug = 0; // Off by default
    
    // Initialize bounds checking variables
    bounds_check_mode = 0; // Default to exact match
    bounds_validated = false;
    hdf5_dims[0] = hdf5_dims[1] = hdf5_dims[2] = 0;
    hdf5_origin[0] = hdf5_origin[1] = hdf5_origin[2] = 0.0;

    // Read temperature values (full dataset loading for compatibility)
    //Initialize chunked reading variables
    chunk_time_window = 0.1; // Default 100ms time window per chunk
    current_chunk_start_time = 0.0;
    current_chunk_end_time = 0.0;
    hdf5_file_open = false;
    
    // Initialize modular temperature source system
    temperature_source = nullptr;
    use_temperature_source = false;  // Default to legacy HDF5 system for backward compatibility
    fast_forward_search_window = 0.1;  // Default 100ms search window
    
    //add the double array
    recreate_arrays();  
}

/* ---------------------------------------------------------------------- */

AppAdditiveExtTempTexture::~AppAdditiveExtTempTexture()
{
  // Only delete arrays that are specific to this class
  // Parent class destructors will handle sites, unique, unique_neigh
  if (unique_dot) delete[] unique_dot;
  if (neigh_dist) delete[] neigh_dist;
  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  if (solid_front_coeffs) delete[] solid_front_coeffs;
  
  if (hdf5_file_open) {
    close_hdf5_file();
  }
}

/* ----------------------------------------------------------------------
   input script commands unique to this app
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"liquidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal liquidus command");
    tl = atof(arg[0]);
    if (tl <= 0) 
      error->all(FLERR,"Illegal liquidus temperature");
  }
  else if (strcmp(command,"solidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal solidus command");
    ts = atof(arg[0]);
    if (ts <= 0) 
      error->all(FLERR,"Illegal solidus temperature");
  }
  else if (strcmp(command,"nucleation_density") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nucleation density command");
    no = atof(arg[0]);
    if (no < 0) 
      error->all(FLERR,"Illegal nucleation density");
  }
  else if (strcmp(command,"critical_undercooling") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal critical_undercooling command");
    tc = atof(arg[0]);
    if (tc < 0) 
      error->all(FLERR,"Illegal critical undercooling");
  }
  else if (strcmp(command,"undercooling_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal undercooling_deviation command");
    tsig = atof(arg[0]);
    if (tsig < 0) 
      error->all(FLERR,"Illegal undercooling standard deviation");
  }
  else if (strcmp(command,"mean_nuclei_volume") == 0) {
    if (narg !=1) error->all(FLERR,"Illegal mean_nuclei_volume command");
    size_norm = atof(arg[0]);
    if (size_norm < 0) 
      error->all(FLERR,"Illegal mean nuclei volume");
  }
  else if (strcmp(command,"nuclei_volume_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nuclei_volume_deviation command");
    size_sig = atof(arg[0]);
    if (size_sig < 0) 
      error->all(FLERR,"Illegal nuclei volume standard deviation");
  }
  else if (strcmp(command,"solid_front_vel") == 0) {
    delete [] solid_front_coeffs;
    if (narg < 1) error->all(FLERR,"Illegal solid_front_vel command");
    int nVel;
    nVel = atoi(arg[0]);
    if (nVel <= 0) {
      error->all(FLERR,"Illegal solidification front velocity specification");
    }
    solid_front_coeffs = new double [nVel];
    int j = 0;
    for(int i = 1; i < nVel + 1; i++) {
        solid_front_coeffs[j] = atof(arg[i]);
        j++;
    }
  }
  else if (strcmp(command,"texture_parameters") == 0) {
     if (narg != 3) error->all(FLERR,"Illegal texture_parameters command");
     c1 = atof(arg[0]);
     c2 = atof(arg[1]);
     c3 = atof(arg[2]);
  }
  else if (strcmp(command,"bounds_check_mode") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal bounds_check_mode command");
     bounds_check_mode = atoi(arg[0]);
     if (bounds_check_mode < 0 || bounds_check_mode > 1) 
       error->all(FLERR,"Illegal bounds_check_mode value: must be 0 (exact) or 1 (subvolume)");
  }
  else if (strcmp(command,"normal_finder_debug") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal normal_finder_debug command");
     normal_finder_debug = atoi(arg[0]);
     if (normal_finder_debug < 0 || normal_finder_debug > 1) 
       error->all(FLERR,"Illegal normal_finder_debug value: must be 0 (off) or 1 (on)");
  }
  else if (strcmp(command,"misorientation_function") == 0) {
     if (narg != 2 && narg != 3) error->all(FLERR,"Illegal misorientation_function command");
     max_misorient = atof(arg[0]);
     mis_thresh = atof(arg[1]) * MY_PI / 180.0; // Convert from degrees to radians
     if (narg == 3) {
       misorient_alpha = atof(arg[2]);
       if (misorient_alpha <= 0)
         error->all(FLERR,"Illegal misorientation_function alpha value: must be greater than 0");
     }
     if (max_misorient < 0) 
       error->all(FLERR,"Illegal misorientation_function maximum value: must be greater than 0");
     if (mis_thresh < 0)
       error->all(FLERR,"Illegal misorientation_function threshold value: must be greater than 0");
  }
  else if (strcmp(command,"chunk_time_window") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal chunk_time_window command");
     chunk_time_window = atof(arg[0]);
     if (chunk_time_window < 0) 
       error->all(FLERR,"Illegal chunk_time_window value: must be >= 0");
  }
  else if (strcmp(command,"fast_forward_search_window") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal fast_forward_search_window command");
     fast_forward_search_window = atof(arg[0]);
     if (fast_forward_search_window <= 0.0)
       error->all(FLERR,"Illegal fast_forward_search_window value: must be > 0");
  }
  
  // Modular temperature source commands
  else if (strcmp(command,"setup_temperature_source") == 0) {
    setup_temperature_source_cmd(narg, arg);
  }
  else if (strcmp(command,"scan_path") == 0) {
    setup_scan_path_cmd(narg, arg);
  }
  
  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
  When app_update is called, read in an input file
 ------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::app_update(double dt)
{

  //Reset t_active to assume we don't have a melt pool right now
  t_active = 0;
  
  // Update temperature first to get current temperatures
  if (use_temperature_source) {
    // Use new modular temperature source system
    update_temperature_from_source(time);
    
    // Check for active sites using temperature source's threshold
    double fast_forward_threshold = t_room; // Default to t_room
    
    // If using HDF5 temperature source, use its threshold
    HDF5UnstructuredTemperatureSource* hdf5_source = 
      dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
    if (hdf5_source) {
      fast_forward_threshold = hdf5_source->get_fast_forward_threshold();
    }
    
    for (int i = 0; i < nlocal; i++) {
      if (T[i] > fast_forward_threshold) t_active = 1;
    }
    
    // Use MPI_Allreduce to check if t_active is 0 on all processors
    int global_t_active;
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    
    // If no active sites and temperature source supports time queries, fast forward
    if (global_t_active == 0 && time > 1e-6 && temperature_source->supports_time_queries()) {
      double next_thermal_time = temperature_source->get_next_time_with_temperature(time, fast_forward_threshold);
      
      if (next_thermal_time > time && next_thermal_time < std::numeric_limits<double>::max()) {
        double time_skip = next_thermal_time - time;
        if (domain->me == 0 && time_skip > 1e-6) {  // Only report significant time skips
          std::cout << "Fast-forward: Skipping " << time_skip << " seconds from time " 
                    << time << " to " << next_thermal_time << " (all temperatures < " << fast_forward_threshold << "K)" << std::endl;
        }
        
        // Update simulation time
        time = next_thermal_time;
        
        // Update temperatures at new time
        update_temperature_from_source(time);
      }
    }
  }
  
  // Legacy HDF5 system (when not using modular temperature source)
  if (!use_temperature_source) {
    // Use legacy HDF5 system
    //Check if we need to load next chunk
    if (hdf5_file_open && needs_new_chunk(time)) {
      current_chunk_start_time = current_chunk_end_time;
      current_chunk_end_time += chunk_time_window;
      load_next_chunk();
    }
    //iterate through all the sets
    for (int i=0; i<nlocal; i++) {    
            
        //Update the temperature at all the sites
        temperature_time_interpolate(i,T[i]);
        if(T[i] > t_room) t_active = 1;
    
      
    }
  } // End legacy HDF5 system block
  
  // Common processing for both temperature systems
  
  //iterate through all the sites for phase transitions (applies to both systems)
  for (int i=0; i<nlocal; i++) {    
    
    //Turn the sites on/off depending on the phase data and whether or not the
    //site's temperature has gone above tl. Only do this when melting the first time 
    //to avoid repeated initialization
    if( (T[i] >= tl) && active_flag[i] != 2) {
        active_flag[i] = 2;
        spin[i] = (int) (nspins * ranapp->uniform());
        // Create random orientation as each site
        vector<double> uq = quaternion::generate_random_unit_quaternions(1);
        q0[i] = uq[0];
        qx[i] = uq[1];
        qy[i] = uq[2];
        qz[i] = uq[3];
        solid_d[i] = 0;
        //melt_misorientation_out[i] = 0;
        G[i] = 0;
        V[i] = 0;
    }
    //If we're molten, call the mushy_phase function to figure out any phase change
    else if (active_flag[i] == 2 && T[i] <= tl) {
        mushy_phase(i, ranapp);
    } 
    else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3)    {
            mobility_out[i] = 1;
            site_event_rejection(i, ranapp);
            solid_d[i]--;
    }
  }
    
  //Communicate changes
  timer->stamp();
  comm->all();
  timer->stamp(TIME_COMM);
  
    // Use MPI_Allreduce to check if t_active is 0 on all processors
    int global_t_active;
    timer->stamp();
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    timer->stamp(TIME_COMM);

    // Check if all processors have t_active = 0. If so, fast forward our simulation time.
    if (global_t_active == 0) {
        double min_time = time_in.findAndSyncSmallestFrontValue(MPI_COMM_WORLD);
        
        //If min time is valid (not DBL_MAX), fast forward. Otherwise continue normally.
        if(min_time < std::numeric_limits<double>::max() && min_time > time) {
            if (domain->me == 0) std::cout << "Fast fowarding to " << min_time << std::endl;
            //If min time is past stop time, update to it. Otherwise, fast forward to min_time.
            //We might want to fast forward to min_time - dt instead...
            if(min_time > stoptime) time = stoptime;
            else time = min_time - dt;
        }
        else if(min_time >= std::numeric_limits<double>::max()) {
            // All queues are empty - try loading next chunk
            if (hdf5_file_open) {
                current_chunk_start_time = current_chunk_end_time;
                current_chunk_end_time += chunk_time_window;
                if (domain->me == 0) std::cout << "Loading next chunk [" << current_chunk_start_time << ", " << current_chunk_end_time << ") due to empty queues" << std::endl;
                load_next_chunk();
            }
        }
    }
    
    timer->stamp(TIME_UPDATE);
}

/* ----------------------------------------------------------------------
	Read in and map reduced temperature data from an hdf5 file
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::reduced_temperature_hdf(){

    // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  //Setup file name
  // std::stringstream os;
  // os << temp_file_string;
  // const std::string tmp = os.str();
  // const char* cstr = tmp.c_str();

  if (domain->me == 0) std::cout << "File path: " << temp_file_string << std::endl;

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
  
  // Create a vector to hold the data counts with the correct hyperslab size
  std::vector<int> data_counts(hyperslab_size);
  

  // Open the HDF5 file in parallel mode
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
  hid_t file_id = H5Fopen(temp_file_string.c_str(), H5F_ACC_RDONLY, fapl_id);

  // Open the data_counts dataset
  hid_t dataset_counts_id = H5Dopen(file_id, "data_counts",H5P_DEFAULT);

  hid_t data_type = H5Dget_type(dataset_counts_id);

  //Figure out which 


  // Define the hyperslab for data_counts using correct integer bounds
  hsize_t start_counts[3] = { (hsize_t)xlo, (hsize_t)ylo, (hsize_t)zlo};
  hsize_t count_counts[3] = { (hsize_t)subdomain_x_size, (hsize_t)subdomain_y_size, (hsize_t)subdomain_z_size };
  



  //Create memory space
  hid_t memspace_count_id = H5Screate_simple(3, count_counts, NULL);

  // Select the hyperslab in the data_counts dataset
  hid_t filespace_id = H5Dget_space(dataset_counts_id);
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start_counts, NULL, count_counts, NULL);

  // Read the relevant portion of data_counts into the vector
  H5Dread(dataset_counts_id, data_type, memspace_count_id, filespace_id, plist_id, data_counts.data());

  // Close the data_counts dataset
  MPI_Barrier(world); // Synchronize all processes
  H5Sclose(filespace_id);
  H5Dclose(dataset_counts_id);

  //convert 1D data to 3d array using correct integer bounds
  int bounds_xlo = (int)domain->subxlo;
  int bounds_xhi = (int)ceil(domain->subxhi);
  int bounds_ylo = (int)domain->subylo;
  int bounds_yhi = (int)ceil(domain->subyhi);
  int bounds_zlo = (int)domain->subzlo;
  int bounds_zhi = (int)ceil(domain->subzhi);
  auto data_counts_array = convert_to_3d_array_with_range(data_counts, bounds_xlo, bounds_xhi, bounds_ylo, bounds_yhi, bounds_zlo, bounds_zhi);
  if (domain->me == 0) std::cout << "Read in the data_count values" << std::endl;

  // Open the temperature and time datasets
  hid_t dataset_temperature_id = H5Dopen(file_id, "temperature",H5P_DEFAULT);
  hid_t dataset_time_id = H5Dopen(file_id, "time",H5P_DEFAULT);

  hid_t data_type_temp = H5Dget_type(dataset_temperature_id);
  hid_t data_type_time = H5Dget_type(dataset_time_id);

  if (domain->me == 0) std::cout << "Began reading temperatures " << std::endl;


// Process the valid entries for the sub-domain using a single loop
for (int local_index = 0; local_index < nlocal; local_index++) {

  hid_t temperature_filespace_id, time_filespace_id;

  // Calculate the (x, y, z) coordinates from the local index relative to the subdomain and global
  int x_loc = (int)xyz[local_index][0] - (int)domain->subxlo;
  int y_loc = (int)xyz[local_index][1] - (int)domain->subylo;
  int z_loc = (int)xyz[local_index][2] - (int)domain->subzlo;
  int x = (int)xyz[local_index][0];
  int y = (int)xyz[local_index][1];
  int z = (int)xyz[local_index][2];

  int valid_count = data_counts_array[x_loc][y_loc][z_loc]; // Get the number of valid entries
  
  // Define the hyperslab to read only the valid entries
  if (valid_count > 0) {
      // Create a vector to hold the valid entries
      std::vector<double> temp_values(valid_count); // Temporary array to hold the valid entries
      std::vector<double> time_values(valid_count); // Temporary array to hold the valid entries

      // Define the starting point in the file
      hsize_t start[4] = { (hsize_t)x, (hsize_t)y, (hsize_t)z, 0};
      hsize_t count[4] = { 1, 1, 1, (hsize_t)valid_count};

      // Select the hyperslab in the temperature dataset
      temperature_filespace_id = H5Dget_space(dataset_temperature_id);
      H5Sselect_hyperslab(temperature_filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);

      // Select the hyperslab in the temperature dataset
      time_filespace_id = H5Dget_space(dataset_time_id);
      H5Sselect_hyperslab(time_filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);

      //Create memory space
      hid_t memspace_id = H5Screate_simple(4, count, NULL);

      // Read the valid entries into the temporary array
      H5Dread(dataset_temperature_id, data_type_temp, memspace_id, temperature_filespace_id, plist_id, temp_values.data());
      H5Dread(dataset_time_id, data_type_time, memspace_id, time_filespace_id, plist_id, time_values.data());

      // Store the valid entries in the corresponding queue
      for (int k = 0; k < valid_count; ++k) {
          temp_in(x_loc,y_loc,z_loc).push(temp_values[k]);
          time_in(x_loc,y_loc,z_loc).push(time_values[k]);
      }

      // Free the hyperslab space
      H5Sclose(temperature_filespace_id);
      H5Sclose(time_filespace_id);
  }
}

  // Close the dataset and file
  MPI_Barrier(world); // Synchronize all processes
  H5Dclose(dataset_temperature_id);
  H5Dclose(dataset_time_id);
  H5Fclose(file_id);

  if (domain->me == 0) std::cout << "Finished reading temperatures " << std::endl;

    // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Print the duration
  if (domain->me == 0) std::cout << "Reading temperatures took: " << duration.count() << " ms" << std::endl;


}

/* ----------------------------------------------------------------------
  Does linear interpolation between two known temperatures and times to calculate current value at timestep.
  Also has checks for empty temperature vectors and timesteps before time
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::temperature_time_interpolate(int site, double priorTemp) {

  int x_loc = xyz[site][0] - (int)domain->subxlo;
  int y_loc = xyz[site][1] - (int)domain->subylo;
  int z_loc = xyz[site][2] - (int)domain->subzlo;


  //If we're out of entires, set to room temperature.
  if (time_in(x_loc,y_loc,z_loc).empty()){ 
    T[site] = t_room;
    return;
  }
  //If we haven't encountered our first time value, set to default
  else if(prior_time == 0 && time < time_in(x_loc,y_loc,z_loc).front()) {
    T[site] = t_room;
    return;
  }

  //If we've stepped past the current time, update the stored values
  if(time >= time_in(x_loc,y_loc,z_loc).front()) {
    prior_time = time_in(x_loc,y_loc,z_loc).front();
    priorTemp = temp_in(x_loc,y_loc,z_loc).front();

    //Pop off old values
    time_in(x_loc,y_loc,z_loc).pop();
    temp_in(x_loc,y_loc,z_loc).pop();
    
    //If we're out of entires, also set to room temperature.
    if (time_in(x_loc,y_loc,z_loc).empty()){ 
      T[site] = t_room;
      return;
    }
  }

  //If we're inbetween melt cycles, set temp to room temp
  if((priorTemp < ts && temp_in(x_loc,y_loc,z_loc).front() < ts)) {
    T[site] = t_room;
  }

  //Do linear interpolation
  else {
    T[site] = priorTemp + (temp_in(x_loc,y_loc,z_loc).front() - priorTemp)/(time_in(x_loc,y_loc,z_loc).front() - prior_time) * (time - prior_time);
  }
}

/* ----------------------------------------------------------------------
   Nucleation site initializer. Find the volume of a voxel from dx^3 and Multiply by no.
   This will be the average number of nucleation sites in the voxel. This is also the
   fraction of spins that we want to be able to nucleate new grains. If the value is greater
   than 1 (which we should avoid), allow all spins to nucleate. If not, call a random number
   between zero and one. If the number is less than the fraction, make true. If not, make false.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_spins(RandomPark *random) {
    timer->stamp();
    
    nucleation_flags = new int[nspins];
    double nucleationFraction = dx * dx * dx * no;
    
    // Initialize all spins as nucleation sites (fallback behavior)
    if(nucleationFraction >= 1.0) {
        fprintf(screen,"Nucleation fraction is greater than 1. Decrease no or increase mesh resolution. no = %f\n", nucleationFraction);
        for (int i = 0; i < nspins; i++) {
            nucleation_flags[i] = 1;
        }
    }
    //Do a random number test and allow the spin to nucleate if less than
    else {
        for (int i = 0; i < nspins; i++) {
            if(random->uniform() <= nucleationFraction) {
                nucleation_flags[i] = 1;
            }
            else {
                nucleation_flags[i] = 0;
            }
        }
    }   
    
    timer->stamp(TIME_APP);
}


/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::grow_app()
{
  // Call parent to set up quaternion arrays (q0, qx, qy, qz)
  AppPottsQuaternion::grow_app();
  
  // Set up additional arrays for this class
  active_flag = iarray[1];
  mobility_out = darray[4];
  T = darray[5];
  solid_d = darray[6];
  //melt_misorientation_out = darray[7];
  G = darray[7];
  V = darray[8];

  if (nlocal_app < nlocal) {
    nlocal_app = nlocal;
            
  }
  //Determine whether the domain is fully periodic, if not. Specify the periodicity of each boundary
  if(domain->nonperiodic == 0) {
  		fully_periodic = 1;
  }
  else {
  	x_period = 1- domain->xperiodic;
  	y_period = 1- domain->yperiodic;
  	z_period = 1- domain->zperiodic;
  }
}


/* ----------------------------------------------------------------------
   initialize before each run
   check validity of site values
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::init_app()
{
  // Call parent init_app to handle sites, unique, and unique_neigh
  AppPottsQuaternion::init_app();
  
  // Clean up our own arrays
  if (unique_dot) delete [] unique_dot;
  if (neigh_dist) delete [] neigh_dist;
  
  // Allocate our own arrays
  unique_dot = new double[1 + maxneigh];
  
  double sqrt2 = 1.4142135624;
  double sqrt3 = 1.7320508076;
  RandomPark random(3000);

  //Allocate our temperature and time data structures
  // Use ceiling to match the sizing of data_counts_array
  int x_size = (int)ceil(domain->subxhi) - (int)domain->subxlo;
  int y_size = (int)ceil(domain->subyhi) - (int)domain->subylo;
  int z_size = (int)ceil(domain->subzhi) - (int)domain->subzlo;
  temp_in.initialize(x_size, y_size, z_size);
  time_in.initialize(x_size, y_size, z_size);

  dt_sweep = dt;
  time_index = 0;
  prior_time = 0;

  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  nucleation_flags = new int[nspins];
  nucleation_temps = new double[nspins];
	nucleation_sizes = new double[nspins];

  int flag = 0;
	int flagall;
    for (int i = 0; i < nlocal; i++) {
			if (spin[i] < 1 || spin[i] > nspins) {
				flag = 1;
			}
      // Create random orientation as each site
      vector<double> uq = quaternion::generate_random_unit_quaternions(1);
      q0[i] = uq[0];
      qx[i] = uq[1];
      qy[i] = uq[2];
      qz[i] = uq[3];
    }

  timer->stamp();
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  timer->stamp(TIME_COMM);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
  
	//Initialize the nucleation_flags vector
	if (domain->me==0) {
		nucleation_spins(ranapp);    
	}

	timer->stamp();
	MPI_Bcast(nucleation_flags,nspins, MPI_INT,0,world);
	timer->stamp(TIME_COMM);

	//Initialize the nucleation_temps and nucleation_sizes vectors
	if (domain->me==0) {
			nucleation_init();    
	}

	timer->stamp();
	MPI_Bcast(nucleation_temps,nspins, MPI_DOUBLE,0,world);
	MPI_Bcast(nucleation_sizes,nspins, MPI_DOUBLE,0,world);
	timer->stamp(TIME_COMM);
	
	//Initialize the neigh_dist array need to fill with good values
	neigh_dist = new double[26];
	neigh_dist[0] = sqrt3 * dx;
	neigh_dist[1] = sqrt2 * dx;
	neigh_dist[2] = sqrt3 * dx;
	neigh_dist[3] = sqrt2 * dx;
	neigh_dist[4] = dx;
	neigh_dist[5] = sqrt2 * dx;
	neigh_dist[6] = sqrt3 * dx;
	neigh_dist[7] = sqrt2 * dx;
	neigh_dist[8] = sqrt3 * dx;
	neigh_dist[9] =  sqrt2 * dx;
	neigh_dist[10] = dx;
	neigh_dist[11] =  sqrt2 * dx;
	neigh_dist[12] = sqrt3 * dx;
	neigh_dist[13] = sqrt3 * dx;
	neigh_dist[14] =  sqrt2 * dx;
	neigh_dist[15] = dx;
	neigh_dist[16] =  sqrt2 * dx;
	neigh_dist[17] = sqrt3 * dx;
	neigh_dist[18] = sqrt2 * dx;
	neigh_dist[19] = sqrt3 * dx;
	neigh_dist[20] =  sqrt2 * dx;
	neigh_dist[21] = dx;
	neigh_dist[22] =  sqrt2 * dx;
	neigh_dist[23] = sqrt3 * dx;
	neigh_dist[24] = sqrt2 * dx;
	neigh_dist[25] = sqrt3 * dx;

  t_cool_max = tl - ts;

  //Check that our timestep is small enough	
  double max_front_vel = 0;
	int power = solid_front_length -1;
	for(int k = 0; k < solid_front_length; k++) {
		max_front_vel = max_front_vel + solid_front_coeffs[k] * pow(tl - ts, power);
		power--;
	}

  
  if(dt > dx/max_front_vel) {
      fprintf(screen,"Temperature timestep too large, reduce to %f\n", dx/max_front_vel);
          error->all(FLERR,"Temperature timestep too large");
  }
  else{
    if(domain->me == 0) {
      fprintf(screen,"Maximum allowable timestep is %e\n", dx/max_front_vel);
    }
  }
  
  // Only initialize legacy HDF5 system if not using modular temperature source
  if (!use_temperature_source) {
    if (domain->me == 0) {
      std::cout << "Starting chunked HDF5 reading" << std::endl;
      std::cout << "Using liquidus temperature tl = " << tl << " K" << std::endl;
      std::cout << "Using solidus temperature ts = " << ts << " K" << std::endl;
    }
    reduced_temperature_hdf_chunked();
    if (domain->me == 0) std::cout << "Chunked HDF5 reading completed" << std::endl;
  } else {
    if (domain->me == 0) {
      std::cout << "Using modular temperature source - skipping legacy HDF5 initialization" << std::endl;
      std::cout << "Using liquidus temperature tl = " << tl << " K" << std::endl;
      std::cout << "Using solidus temperature ts = " << ts << " K" << std::endl;
    }
  }

	this->app_update(0.0);
}


/* ----------------------------------------------------------------------
   Override parent site_energy to only calculate energy for solidified sites
   and exclude non-solidified neighbors from energy calculation
------------------------------------------------------------------------- */

double AppAdditiveExtTempTexture::site_energy(int i) {
  timer->stamp();
  
  // Condition 1: If active_flag != 3 (not solidified), return zero energy
  if (active_flag[i] != 3) {
    timer->stamp(TIME_SOLVE);
    return 0.0;
  }
  
  // Same algorithm as parent class but with active_flag filtering
  double energy = 0.0;
  vector<double> qi{q0[i], qx[i], qy[i], qz[i]};
  
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    
    // Condition 2: Exclude neighbors with active_flag != 3 or 1
    if (active_flag[nj] != 3 || active_flag[nj] != 1) {
      continue;
    }
    
    vector<double> qj{q0[nj], qx[nj], qy[nj], qz[nj]};
    double di = disorientation::compute_disorientation(symmetries, qi, qj);
    double ratio = di / theta_cut;

    // Read-Shockley equation logic (same as parent)
    if (ratio <= 0)
      continue;
    else if (ratio < 1.0)
      energy += ratio * (1 - log(ratio));
    else
      energy += 1.0;
  }
  
  // Each site carries half the grain boundary energy
  // Neighbor sites carry the other half
  timer->stamp(TIME_SOLVE);
  return 0.5 * energy;
}

/* ----------------------------------------------------------------------
   rKMC method
   perform a site event with no null bin rejection
   flip to random neighbor spin without null bin
   technically this is an incorrect rejection-KMC algorithm
   I think we can eventually get rid of mobilityout
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::site_event_rejection(int i, RandomPark *random)
{
  timer->stamp();
  
  int oldstate = spin[i];
  SiteState s_old(spin[i], {q0[i], qx[i], qy[i], qz[i]});
  double einitial = site_energy(i);
  double efinal = 0;
  double Mobloc = 0;
  double dotValue = 0;
  // std::cout << "Running site_event_rejection when I shouldn't!" << std::endl;

    //Assign the local mobility
    Mobloc = mobility_out[i];
    
    int j,m,value;
    int nevent = 0;
    
    if((Mobloc < 0.0) || (Mobloc > 1.0001)) {
        mobility_out[i] = 0;
        return;
    }
    
    if(solid_d[i] < 0 && solid_d[i] > -nrefine -1) {
        //Go through neighbor list and add them to possible switches
        for (int j = 0; j < numneigh[i]; j++) {
            if(active_flag[neighbor[i][j]] == 3 || active_flag[neighbor[i][j]] == 1) {
                // Calculate temperature gradient/grain misorientation and store in array
                // Use cumulative probability for random sampling
                //Exclude gas or molten sites from the Potts neighbor tally
                double melt_misori_val = melt_misorientation(neighbor[i][j],c1,c2,c3);
                //melt_misorientation_out[i] = melt_misori_val;
                dotValue += melt_misori_val;
                unique_dot[nevent] = dotValue;
                value = spin[neighbor[i][j]];
                unique[nevent] = value;
                unique_neigh[nevent] = neighbor[i][j];
                nevent++;										
            }
        }
        //If no neighbor is eligible, return before changing anything. Will try next sweep.
        if (nevent == 0) return;
        // Use nevent-1 to account for extra event at end
        double dran = (unique_dot[nevent - 1]*random->uniform());
        //if (iran >= nevent) iran = nevent-1;
        //Go through possible events and pick one
        for( int j = 0; j < nevent -1; j++) {
            if(dran <= unique_dot[j]) {
              int neighran = unique_neigh[j];
              // melt_misorientation_out[i] = melt_misorientation(neighran,c1,c2,c3);
              SiteState s_new(unique[j], {q0[neighran], qx[neighran], qy[neighran], qz[neighran]});
              flip_site(i, s_new);
              efinal = site_energy(i);
            }
        }
    }

  else {
      for (j = 0; j < numneigh[i]; j++) {
        value = spin[neighbor[i][j]];
        //Exclude gas, powder or molten sites from the Potts neighbor tally
        if (value == spin[i] || value == nspins || active_flag[neighbor[i][j]] != 3) continue;
        for (m = 0; m < nevent; m++) 
          if (value == unique[m]) break;
        if (m < nevent) continue;
        unique[nevent] = value;
        unique_neigh[nevent] = neighbor[i][j];
        nevent += 1;
      }

      if (nevent == 0) return;
      int iran = (int) (nevent*random->uniform());
      if (iran >= nevent) iran = nevent-1;
      int neighran = unique_neigh[iran]; // Get neighbor index
      SiteState s_new(unique[iran], {q0[neighran], qx[neighran], qy[neighran], qz[neighran]});
      flip_site(i, s_new);
      efinal = site_energy(i);
  }
  // accept or reject via Boltzmann criterion

  if (efinal <= einitial) {
     if (random->uniform() > Mobloc){
       flip_site(i, s_old);
      //  melt_misorientation_out[i] = 0;
     }
  }
  else if (temperature == 0.0) {
    flip_site(i, s_old);
    // melt_misorientation_out[i] = 0;

  } 
  else if (random->uniform() > Mobloc * exp((einitial-efinal)*t_inverse)) {
    flip_site(i, s_old);
    // melt_misorientation_out[i] = 0;

  }



  if (spin[i] != oldstate) {
    naccept++;
  }

  // set mask if site could not have changed
  // if site changed, unset mask of sites with affected propensity
  // OK to change mask of ghost sites since never used
  if (Lmask) {
    if (einitial < 0.5*numneigh[i]) mask[i] = 1;
    if (spin[i] != oldstate)
      for (int j = 0; j < numneigh[i]; j++)
	      mask[neighbor[i][j]] = 0;
  }

  timer->stamp(TIME_SOLVE);
}

/* ----------------------------------------------------------------------
   Perform evolution for sites in the mushy zone. There are several things that need to happen.
   1. Determine if the site is solid or liquid (from active_flag)
   2. Determine if the site should nucleate a new grain (from nucleation_flags)
   3. If our current site is liquid & can't nucleate, have it try to switch to a solid neighbor
      (with 4 calculated from the undercooling in someway, not temperature)
   4. If our current site is liquid & can nucleate, check if local undercooling is equal to its 
      critical temp. If so, change the active_flag value to solid. If not, see if there are any solid sites
      that should capture it.
   5. If our current site is solid, see if it should flip to a neighboring solid value (with
      mobility calculated from undercooling.)
THIS NEEDS UPDATED TO INCLUDE TEXTURE
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::mushy_phase(int i, RandomPark *random){
    double Tcool = tl - T[i];

    
    //Our site should always be molten and below tl
    //Check if it's eligible to nucleate
    if(nucleation_flags[spin[i]]) {
        //Can and will nucleate
        if(Tcool >= nucleation_temps[spin[i]]){
            active_flag[i] = 3;
            //Don't let nucleated site disappear during smoothing
            //Neighboring sites will be flipped during the next iterate_rejection call
            solid_d[i] = -nrefine-2;

            std::vector<double> solid_G(3);
            solid_G = normal_finder(i); //Update gradient
            G[i] = sqrt(pow(solid_G[0],2) + pow(solid_G[1],2) + pow(solid_G[2],2));

            int power = solid_front_length - 1;
            for(int k = 0; k < solid_front_length; k++) {
                V[i] = V[i] + solid_front_coeffs[k] * pow(Tcool, power); //Update solidification rate
                power--;
            }

            //Call nucleation particle flipper
            naccept++;
            nucleation_particle_flipper(i, round(nucleation_sizes[spin[i]]/pow(dx,3)), ranapp);
            return;
        }
        //Can nucleate, but won't yet. Allow to solidify if the solidification front captures it first.
        else {
            // fprintf(screen,"Epitaxialy growing instead of nucleating\n");
            site_event_solidification(i, Tcool, random);
        }
    }
    else {
        site_event_solidification(i, Tcool, random);
    }
}

/* ----------------------------------------------------------------------
   Perform solidification event for a site in the mushy zone.
   This handles the common logic for both nucleation and non-nucleation paths.
   Updates solid front distance and attempts epitaxial growth from neighboring sites.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::site_event_solidification(int i, double Tcool, RandomPark *random) {
    int nevent = 0;
    int value;
    double dotValue = 0;
    int solidNeigh = 0;

    //Add the distance of the front travel. Need to multiply by timestep to get distance
    //Only start growing if we're on the boundary or have a solid neighbor
    if (numneigh[i] < 26) solidNeigh = 1;
    else {
      for (int j = 0; j < numneigh[i]; j++) {
        if(active_flag[neighbor[i][j]] == 3 || active_flag[neighbor[i][j]] == 1) {
          solidNeigh = 1;
          break;
        }
      }
    }

    if(solidNeigh == 1) {
      int power = solid_front_length - 1;
      for(int k = 0; k < solid_front_length; k++) {
          solid_d[i] = solid_d[i] + solid_front_coeffs[k] * pow(Tcool, power) * time_step;
          power--;
      }
    
      //Go through neighbor list and add them to possible switches
      for (int j = 0; j < numneigh[i]; j++) {
          if(neigh_dist[2] <= solid_d[i] && (active_flag[neighbor[i][j]] == 1 || active_flag[neighbor[i][j]] == 3)) {
              // Calculate temperature gradient/grain misorientation and store in array
              // Use cumulative probability for random sampling
              //Exclude gas or molten sites from the Potts neighbor tally
              double melt_misori_val = melt_misorientation(neighbor[i][j],c1,c2,c3);
              // melt_misorientation_out[i] = melt_misori_val;
              dotValue += melt_misori_val;
              unique_dot[nevent] = dotValue;
              value = spin[neighbor[i][j]];
              unique[nevent] = value;
              unique_neigh[nevent] = neighbor[i][j];
              nevent++;
          }
      }
    }
    //If no neighbor is eligible, return before changing anything. Will try next sweep.
    if (nevent == 0) return;
    
    // Use nevent-1 to account for extra event at end
    double dran = (unique_dot[nevent - 1]*random->uniform());
    //if (iran >= nevent) iran = nevent-1;
    //Go through possible events and pick one
    for( int j = 0; j < nevent - 1; j++) {
        if(dran <= unique_dot[j]) {
            int neighran = unique_neigh[j];
            SiteState s1(unique[j],{q0[neighran],qx[neighran],qy[neighran],qz[neighran]});
            // melt_misorientation_out[i] = melt_misorientation(neighran,c1,c2,c3);
            flip_site(i,s1);
            active_flag[i] = 3;
            solid_d[i] = -1;
            
            std::vector<double> solid_G(3);
            solid_G = normal_finder(i); //Update gradient
            G[i] = sqrt(pow(solid_G[0],2) + pow(solid_G[1],2) + pow(solid_G[2],2));

            int power = solid_front_length - 1;
            for(int k = 0; k < solid_front_length; k++) {
               V[i] = V[i] + solid_front_coeffs[k] * pow(Tcool, power); //Update solidification rate
                power--;
            }
            naccept++;

            // Apply misorientation based on temperature gradient
            apply_misorientation(i, Tcool, ranapp);
            return;
        }
    }
}

/* ----------------------------------------------------------------------
    Apply misorientation based on temperature gradient and cooling rate
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::apply_misorientation(int i, double Tcool, RandomPark *ranapp) {
    if(max_misorient <= 0) return;
    
    // Exponential function: grows slowly initially, then rapidly in last ~10%
    double normalized_temp = Tcool / t_cool_max;
    double mis_angle = max_misorient * (exp(misorient_alpha * normalized_temp) - 1.0) / (exp(misorient_alpha) - 1.0) * MY_PI/180;

    //Calculate the unit-vector surface norm (use voxel counting)
    vector<double> grad_out = normal_finder(i);

    // Create quaternion vector from site's orientation
    vector<double> q_site = {q0[i], qx[i], qy[i], qz[i]};
    
    // Normalize quaternion to ensure it's a unit quaternion
    double q_mag = sqrt(q_site[0]*q_site[0] + q_site[1]*q_site[1] + q_site[2]*q_site[2] + q_site[3]*q_site[3]);
    if (q_mag > 1e-15) {
        q_site[0] /= q_mag;
        q_site[1] /= q_mag;
        q_site[2] /= q_mag;
        q_site[3] /= q_mag;
    }

    vector<double> q_new = quaternion::rotate_q_towards_u(q_site,grad_out, mis_angle);

    q0[i] = q_new[0];
    qx[i] = q_new[1];
    qy[i] = q_new[2];
    qz[i] = q_new[3];

    // Compute direct quaternion rotation angle (without symmetry considerations)
    double dot_product = q_site[0]*q_new[0] + q_site[1]*q_new[1] + q_site[2]*q_new[2] + q_site[3]*q_new[3];
    // Handle both positive and negative dot products (quaternion double cover)
    dot_product = std::abs(dot_product);
    // Clamp to avoid numerical issues
    dot_product = std::min(1.0, dot_product);
    double direct_angle = 2.0 * acos(dot_product);
    
    if(direct_angle >= mis_thresh) {
        spin[i] = (int) (nspins * ranapp->uniform()); //If greater than misorientation threshold, make new grain
    }
}

/* ----------------------------------------------------------------------
    Only nucleating one site at a time introduce lattice size dependency. Here we will
    use a user-defined nucleation particle size and flip neighboring sites until that size is met
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_particle_flipper(int i, int partRad, RandomPark *random) {
    
    //If one site is big enough to satisfy, skip evertyhing
    if(partRad <= 0) return;
    
    int nSites = partRad;
    int nSitesIn = nSites;
    int nearest_neigh [ ] = {4, 10, 12, 15, 21, 13};
    int second_nearest [ ] = {9,16,3,22,14,11,20,5,1,24,7,18};
    int third_nearest [ ] = {0,25,23,2,6,19,17,8};
    int nneigh = 0;
    int possible_neigh[26];
    SiteState s_in(spin[i],{q0[i], qx[i], qy[i], qz[i]});

    

    //Its hard to go through shells iteravely if the number of neighbors isn't the full 26.
    //Check if this is the case and just loop through the neihgbor list if so
    if(numneigh[i] != 26) {

        
        for(int j = 0; j < numneigh[i]; j++) {
            if(active_flag[neighbor[i][j]] == 2) {
                int i_chosen = neighbor[i][j];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                //comm->all();
                return;
            }
        }
    }
    else {
        //Go through neighbors and nucleate liquid ones
        //Create shuffled indices for each shell
        std::vector<int> shell1_indices(6);
        std::vector<int> shell2_indices(12);
        std::vector<int> shell3_indices(8);
        
        // Initialize index arrays
        for(int j = 0; j < 6; j++) shell1_indices[j] = j;
        for(int j = 0; j < 12; j++) shell2_indices[j] = j;
        for(int j = 0; j < 8; j++) shell3_indices[j] = j;
        
        // Shuffle each shell's indices using Fisher-Yates algorithm
        // Shell 1 (nearest neighbors)
        for(int j = 5; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell1_indices[j], shell1_indices[k]);
        }
        
        // Shell 2 (second nearest)
        for(int j = 11; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell2_indices[j], shell2_indices[k]);
        }
        
        // Shell 3 (third nearest)
        for(int j = 7; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell3_indices[j], shell3_indices[k]);
        }
        
        //Do 1st shell in randomized order
        for(int j = 0; j < 6; j++) {
            int idx = shell1_indices[j];
            if(active_flag[neighbor[i][nearest_neigh[idx]]] == 2) {
                int i_chosen = neighbor[i][nearest_neigh[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
        //Do 2nd shell in randomized order
        for(int j = 0; j < 12; j++) {
            int idx = shell2_indices[j];
            if(active_flag[neighbor[i][second_nearest[idx]]] == 2) {
                int i_chosen = neighbor[i][second_nearest[idx]];
                flip_site(i_chosen, s_in);            
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }      
          }
        //Do 3rd shell in randomized order   
        for(int j = 0; j < 8; j++) {
            int idx = shell3_indices[j];
            if(active_flag[neighbor[i][third_nearest[idx]]] ==2) {
                int i_chosen = neighbor[i][third_nearest[idx]];
                //spin[i_chosen] = spin[i];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
    }
    //If we didn't fill any sites this time, just quit
    if (nSites == nSitesIn) {
        return;
    }
    
    //If we still haven't satisfied the particle size, pick a neighbor at random and solidify from there.
    //Build a list of same-particle neighbors and pick one randomly
    for(int j =0; j < numneigh[i]; j++) {
        if(spin[neighbor[i][j]] == spin[i] && active_flag[neighbor[i][j]] == 3) {
            possible_neigh[nneigh] = j;
            nneigh++;
        }
    }
    //If no possible nieghbors, quit
    if(nneigh == 0) {
        return;
    }
    //Otherwise, randomly pick a possilbe neighbor
    int neighran =  round(((nneigh -1) * random->uniform()));
    
    nucleation_particle_flipper(neighbor[i][possible_neigh[neighran]],nSites, random);
    return;
    
}


//Texture specific functions start here

// Calculate the local direction of the largest temperature gradient
// using weighted least squares with multiple neighbors for improved accuracy
std::vector<double> AppAdditiveExtTempTexture::normal_finder(int site)
{
	// Debug statistics for comparing old vs new methods (only if debugging enabled)
	static int debug_counter = 0;
	static double bulk_angle_sum = 0.0;
	static int bulk_count = 0;
	static double boundary_angle_sum = 0.0;
	static int boundary_count = 0;
	static double melt_surface_angle_sum = 0.0;
	static int melt_surface_count = 0;
	
	bool should_debug = (normal_finder_debug == 1);
	bool should_output = false;
	
	if (should_debug) {
		debug_counter++;
		should_output = (debug_counter % 60000 == 0);
	}
	
	// Initialize gradient components
	double grad_x = 0, grad_y = 0, grad_z = 0;
	
	// Set up neighbor offset mapping for 3D cubic lattice (SC_26N)
	// Order: i,j,k from -1 to 1 (excluding center at 0,0,0)
	int offset_map[26][3] = {
		{-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},  // i=-1, j=-1
		{-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},  // i=-1, j=0 
		{-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},  // i=-1, j=1
		{ 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},  // i=0, j=-1
		{ 0, 0,-1},             { 0, 0, 1},  // i=0, j=0 (skip center)
		{ 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},  // i=0, j=1
		{ 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},  // i=1, j=-1
		{ 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},  // i=1, j=0
		{ 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}   // i=1, j=1
	};
	
	// Calculate old 6-neighbor result for comparison (only if debugging enabled)
	double old_result[3] = {0, 0, 0};
	if (should_debug) {
		double grad_x_old = (T[neighbor[site][4]] - T[neighbor[site][21]]) / (2.0 * dx);
		double grad_y_old = (T[neighbor[site][10]] - T[neighbor[site][15]]) / (2.0 * dx);
		double grad_z_old = (T[neighbor[site][12]] - T[neighbor[site][13]]) / (2.0 * dx);
		
		double norm_old = sqrt(grad_x_old*grad_x_old + grad_y_old*grad_y_old + grad_z_old*grad_z_old);
		if (norm_old > 1e-12) {
			old_result[0] = fabs(grad_x_old)/norm_old;
			old_result[1] = fabs(grad_y_old)/norm_old;
			old_result[2] = fabs(grad_z_old)/norm_old;
		}
	}
	
	// Special case: top of melt pool where some neighbors are inactive
	// Use only neighbors at or below current site's z-coordinate
	bool is_melt_surface = (active_flag[neighbor[site][13]] <= 1);
	
	if (is_melt_surface) {
		// Store old melt surface method result for comparison (only if debugging enabled)
		double old_melt_result[3] = {0, 0, 0};
		double norm_old_melt = 0.0;
		if (should_debug) {
			double xDel_old = fabs(T[neighbor[site][4]] - T[neighbor[site][21]]);
			double yDel_old = fabs(T[neighbor[site][10]] - T[neighbor[site][15]]);
			int lower_site = neighbor[site][12];
			double zDel_old = fabs(T[neighbor[lower_site][12]] - T[neighbor[lower_site][13]]);
			norm_old_melt = sqrt(xDel_old*xDel_old + yDel_old*yDel_old + zDel_old*zDel_old);
			if (norm_old_melt > 1e-12) {
				old_melt_result[0] = xDel_old/norm_old_melt;
				old_melt_result[1] = yDel_old/norm_old_melt;
				old_melt_result[2] = zDel_old/norm_old_melt;
			}
		}
		// Filter neighbors to only include those at or below current z-level
		double site_z = xyz[site][2];
		
		// Weighted least squares with z-filtered neighbors
		double AtA[9] = {0}; // 3x3 matrix A^T * W * A
		double Atb[3] = {0}; // 3x1 vector A^T * W * b
		double T_center = T[site];
		int valid_neighbors = 0;
		
		for (int n = 0; n < numneigh[site] && n < 26; n++) {
			int neighbor_site = neighbor[site][n];
			
			// Skip invalid neighbors
			if (neighbor_site < 0 || neighbor_site >= app->nlocal + app->nghost) continue;
			
			// Only use neighbors at or below current site's z-coordinate
			if (xyz[neighbor_site][2] > site_z) continue;
			
			// Get neighbor offset coordinates
			double dx_n = offset_map[n][0] * dx;
			double dy_n = offset_map[n][1] * dx;
			double dz_n = offset_map[n][2] * dx;
			
			// Temperature difference
			double dT = T[neighbor_site] - T_center;
			
			// Distance and weight
			double dist = sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
			double weight = 1.0 / (dist*dist + 1e-12);
			
			// Build weighted normal equations
			AtA[0] += weight * dx_n * dx_n;  // AtA[0,0]
			AtA[1] += weight * dx_n * dy_n;  // AtA[0,1]
			AtA[2] += weight * dx_n * dz_n;  // AtA[0,2]
			AtA[3] += weight * dy_n * dx_n;  // AtA[1,0] 
			AtA[4] += weight * dy_n * dy_n;  // AtA[1,1]
			AtA[5] += weight * dy_n * dz_n;  // AtA[1,2]
			AtA[6] += weight * dz_n * dx_n;  // AtA[2,0]
			AtA[7] += weight * dz_n * dy_n;  // AtA[2,1]
			AtA[8] += weight * dz_n * dz_n;  // AtA[2,2]
			
			Atb[0] += weight * dx_n * dT;
			Atb[1] += weight * dy_n * dT;
			Atb[2] += weight * dz_n * dT;
			
			valid_neighbors++;
		}
		
		// Solve if we have enough neighbors
		if (valid_neighbors >= 3) {
			double det = AtA[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) - 
			             AtA[1]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) + 
			             AtA[2]*(AtA[3]*AtA[7] - AtA[4]*AtA[6]);
			
			if (fabs(det) > 1e-12) {
				// Compute gradient using Cramer's rule
				grad_x = (Atb[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) - 
				          AtA[1]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) + 
				          AtA[2]*(Atb[1]*AtA[7] - AtA[4]*Atb[2])) / det;
				          
				grad_y = (AtA[0]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) - 
				          Atb[0]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) + 
				          AtA[2]*(AtA[3]*Atb[2] - Atb[1]*AtA[6])) / det;
				          
				grad_z = (AtA[0]*(AtA[4]*Atb[2] - Atb[1]*AtA[7]) - 
				          AtA[1]*(AtA[3]*Atb[2] - Atb[1]*AtA[6]) + 
				          Atb[0]*(AtA[3]*AtA[7] - AtA[4]*AtA[6])) / det;
			} else {
				// Matrix is singular, fall back to simple differences
				grad_x = (T[neighbor[site][4]] - T[neighbor[site][21]]) / (2.0 * dx);
				grad_y = (T[neighbor[site][10]] - T[neighbor[site][15]]) / (2.0 * dx);
				grad_z = (T[neighbor[site][12]] - T_center) / dx; // Only downward gradient
			}
		} else {
			// Too few neighbors, use simple finite differences
			grad_x = (T[neighbor[site][4]] - T[neighbor[site][21]]) / (2.0 * dx);
			grad_y = (T[neighbor[site][10]] - T[neighbor[site][15]]) / (2.0 * dx);
			grad_z = (T[neighbor[site][12]] - T_center) / dx; // Only downward gradient
		}
		
		// Normalize and return
		double norm = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
		std::vector<double> result(3);
		if (norm > 1e-12) {
			result[0] = fabs(grad_x)/norm;
			result[1] = fabs(grad_y)/norm;
			result[2] = fabs(grad_z)/norm;
		} else {
			result[0] = result[1] = result[2] = 0.0;
		}
		
		// Compare old vs new melt surface methods for debugging
		if (should_debug && norm > 1e-12 && norm_old_melt > 1e-12) {
			double dot_product = old_melt_result[0]*result[0] + old_melt_result[1]*result[1] + old_melt_result[2]*result[2];
			double old_norm = sqrt(old_melt_result[0]*old_melt_result[0] + old_melt_result[1]*old_melt_result[1] + old_melt_result[2]*old_melt_result[2]);
			double new_norm = sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
			
			if (old_norm > 1e-12 && new_norm > 1e-12) {
				dot_product = dot_product / (old_norm * new_norm);
				if (dot_product > 1.0) dot_product = 1.0;
				if (dot_product < -1.0) dot_product = -1.0;
				
				double angle_rad = acos(dot_product);
				double angle_deg = angle_rad * 180.0 / 3.14159265359;
				
				melt_surface_angle_sum += angle_deg;
				melt_surface_count++;
			}
		}
		return result;
	}
	
	// Use weighted least squares with all available neighbors
	
	// Weighted least squares matrices: A*grad = b
	// A is 26x3 matrix of [dx dy dz] for each neighbor
	// b is 26x1 vector of temperature differences
	// weights based on inverse distance
	double AtA[9] = {0}; // 3x3 matrix A^T * W * A
	double Atb[3] = {0}; // 3x1 vector A^T * W * b
	
	double T_center = T[site];
	
	for (int n = 0; n < numneigh[site] && n < 26; n++) {
		int neighbor_site = neighbor[site][n];
		
		// Skip invalid neighbors
		if (neighbor_site < 0 || neighbor_site >= app->nlocal + app->nghost) continue;
		
		// Get neighbor offset coordinates
		double dx_n = offset_map[n][0] * dx;  // x offset in physical units
		double dy_n = offset_map[n][1] * dx;  // y offset 
		double dz_n = offset_map[n][2] * dx;  // z offset
		
		// Temperature difference
		double dT = T[neighbor_site] - T_center;
		
		// Distance and weight (inverse distance squared with small regularization)
		double dist = sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
		double weight = 1.0 / (dist*dist + 1e-12);
		
		// Build weighted normal equations: AtA * grad = Atb
		// A matrix row: [dx_n, dy_n, dz_n]
		AtA[0] += weight * dx_n * dx_n;  // AtA[0,0]
		AtA[1] += weight * dx_n * dy_n;  // AtA[0,1]
		AtA[2] += weight * dx_n * dz_n;  // AtA[0,2]
		AtA[3] += weight * dy_n * dx_n;  // AtA[1,0] 
		AtA[4] += weight * dy_n * dy_n;  // AtA[1,1]
		AtA[5] += weight * dy_n * dz_n;  // AtA[1,2]
		AtA[6] += weight * dz_n * dx_n;  // AtA[2,0]
		AtA[7] += weight * dz_n * dy_n;  // AtA[2,1]
		AtA[8] += weight * dz_n * dz_n;  // AtA[2,2]
		
		Atb[0] += weight * dx_n * dT;    // Atb[0]
		Atb[1] += weight * dy_n * dT;    // Atb[1] 
		Atb[2] += weight * dz_n * dT;    // Atb[2]
	}
	
	// Solve 3x3 system AtA * grad = Atb using Cramer's rule
	double det = AtA[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) - 
	             AtA[1]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) + 
	             AtA[2]*(AtA[3]*AtA[7] - AtA[4]*AtA[6]);
	
	if (fabs(det) > 1e-12) {
		// Compute gradient using Cramer's rule
		grad_x = (Atb[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) - 
		          AtA[1]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) + 
		          AtA[2]*(Atb[1]*AtA[7] - AtA[4]*Atb[2])) / det;
		          
		grad_y = (AtA[0]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) - 
		          Atb[0]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) + 
		          AtA[2]*(AtA[3]*Atb[2] - Atb[1]*AtA[6])) / det;
		          
		grad_z = (AtA[0]*(AtA[4]*Atb[2] - Atb[1]*AtA[7]) - 
		          AtA[1]*(AtA[3]*Atb[2] - Atb[1]*AtA[6]) + 
		          Atb[0]*(AtA[3]*AtA[7] - AtA[4]*AtA[6])) / det;
	} else {
		// Fallback to simple finite differences if matrix is singular
		grad_x = (T[neighbor[site][4]] - T[neighbor[site][21]]) / (2.0 * dx);
		grad_y = (T[neighbor[site][10]] - T[neighbor[site][15]]) / (2.0 * dx);
		grad_z = (T[neighbor[site][12]] - T[neighbor[site][13]]) / (2.0 * dx);
	}
	
	// Normalize and return gradient direction
	double norm = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
	std::vector<double> result(3);
	if (norm > 1e-12) {
		result[0] = fabs(grad_x)/norm;
		result[1] = fabs(grad_y)/norm;
		result[2] = fabs(grad_z)/norm;
	} else {
		result[0] = result[1] = result[2] = 0.0;
	}
	
	// Accumulate statistics for comparing old vs new methods
	if (should_debug && norm > 1e-12) {
		// Calculate angle between old and new vectors
		double dot_product = old_result[0]*result[0] + old_result[1]*result[1] + old_result[2]*result[2];
		double old_norm = sqrt(old_result[0]*old_result[0] + old_result[1]*old_result[1] + old_result[2]*old_result[2]);
		double new_norm = sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
		
		if (old_norm > 1e-12 && new_norm > 1e-12) {
			// Clamp dot product to [-1,1] to avoid numerical issues with acos
			dot_product = dot_product / (old_norm * new_norm);
			if (dot_product > 1.0) dot_product = 1.0;
			if (dot_product < -1.0) dot_product = -1.0;
			
			double angle_rad = acos(dot_product);
			double angle_deg = angle_rad * 180.0 / 3.14159265359;
			
			// Determine if this is a boundary site (has fewer than 26 neighbors)
			bool is_boundary = (numneigh[site] < 26);
			
			if (is_boundary) {
				boundary_angle_sum += angle_deg;
				boundary_count++;
			} else {
				bulk_angle_sum += angle_deg;
				bulk_count++;
			}
		}
	}
	
	// Output statistics every 60000 calls
	if (should_output) {
		std::cout << "NORMAL_FINDER STATS [Calls " << (debug_counter-59999) << "-" << debug_counter << "]:\n";
		if (bulk_count > 0) {
			double avg_bulk = bulk_angle_sum / bulk_count;
			std::cout << "  Bulk sites: " << bulk_count << " samples, average angle: " << avg_bulk << " degrees\n";
		} else {
			std::cout << "  Bulk sites: 0 samples\n";
		}
		if (boundary_count > 0) {
			double avg_boundary = boundary_angle_sum / boundary_count;
			std::cout << "  Boundary sites: " << boundary_count << " samples, average angle: " << avg_boundary << " degrees\n";
		} else {
			std::cout << "  Boundary sites: 0 samples\n";
		}
		if (melt_surface_count > 0) {
			double avg_melt_surface = melt_surface_angle_sum / melt_surface_count;
			std::cout << "  Melt surface sites: " << melt_surface_count << " samples, average angle: " << avg_melt_surface << " degrees\n";
			std::cout << "    (Old: crude finite differences vs New: z-filtered weighted least squares)\n";
		} else {
			std::cout << "  Melt surface sites: 0 samples\n";
		}
		std::cout << std::endl;
		
		// Reset counters for next interval
		bulk_angle_sum = 0.0;
		bulk_count = 0;
		boundary_angle_sum = 0.0;
		boundary_count = 0;
		melt_surface_angle_sum = 0.0;
		melt_surface_count = 0;
	}
	
	return result;
}

/* ----------------------------------------------------------------------
   Figure out the dot product between the temperature gradient and the x'tal orientation.
   We'll use this to change the Q-value at the lattice site...

   Inputs: lattice site :: site
   Outputs: double of max dot product
------------------------------------------------------------------------- */
double AppAdditiveExtTempTexture::melt_misorientation(int site, double c1, double c2, double c3)
{
	double dotMax;
	double theta;
	double mobOut;

	//Calculate the unit-vector surface norm (use voxel counting)
	vector<double> grad_vector = normal_finder(site);
	
	// Create quaternion vector from site's orientation
	vector<double> q_site = {q0[site], qx[site], qy[site], qz[site]};
	
	// Use quaternion function to get cosine of minimum angle between
	// crystal orientation and temperature gradient
	dotMax = quaternion::get_cosine_of_minumum_angle_between_q_and_u(q_site, grad_vector);
	
	// Calculate angle from cosine
	theta = acos(dotMax);
	
	// Apply texture mobility model
	mobOut = c1 + c2 * cos(c3 * theta);
	
	return mobOut;
}

//Let's build a global array of euler angles (much like the quaternions), so we dont have to
//Do trig calculations over and over again.
//Call at the beginning of the program
//This function won't do any loops, just the pure math
//Will need verification that I didn't get indices screwed up...
void AppAdditiveExtTempTexture::quat2euler(int site_index, double *eulers) {
  // Convert quaternion to Euler angles
  // Input: site_index - index of the site whose quaternion orientation to convert
  // Output: eulers - array of 3 Euler angles [phi1, Phi, phi2]
  
  // Get quaternion components for this site
  vector<double> q = {q0[site_index], qx[site_index], qy[site_index], qz[site_index]};
  
  // Convert quaternion to rotation matrix using quaternion utility
  vector<double> rotation_matrix = quaternion::to_rotation_matrix(q);
  
  // Extract rotation matrix elements (row-major format)
  // Matrix format: [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
  double r11 = rotation_matrix[0]; double r12 = rotation_matrix[1]; double r13 = rotation_matrix[2];
  double r21 = rotation_matrix[3]; double r22 = rotation_matrix[4]; double r23 = rotation_matrix[5];
  double r31 = rotation_matrix[6]; double r32 = rotation_matrix[7]; double r33 = rotation_matrix[8];
  
  // Convert rotation matrix to Euler angles using ZXZ convention
  // This matches the original algorithm's approach
  
  // Handle special case where r33 is close to ±1
  if (abs(r33) < 1.000001 && abs(r33) > 0.999999) {
    eulers[0] = atan2(r12, r11);
    eulers[1] = MY_PI/2.0 * (1 - r33);
    eulers[2] = 0;
    return;
  }
  // General case
  else {
    double ksi = 1.0 / sqrt(1.0 - r33 * r33);
    
    eulers[0] = atan2(r31 * ksi, -r32 * ksi);
    eulers[1] = acos(r33);
    eulers[2] = atan2(r13 * ksi, r23 * ksi);
    return;
  }
}


/* ----------------------------------------------------------------------
    The first version of this just initialized critical nucleation temperatures.
    This version will also initialize nucleii size (starting with a normal dist)
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_init() {
    std::normal_distribution<> dist_T{tc,tsig};
    std::normal_distribution<> dist_S{size_norm,size_sig};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    // Randomly assign critical temperature to every spin
    for(int i = 0; i < nspins; i++) {
        nucleation_temps[i] = dist_T(gen);
        nucleation_sizes[i] = dist_S(gen);
    }
}

/**
 * @brief Converts a 1D vector (in row-major order) into a 3D array with custom index ranges.
 *
 * This function takes a 1D vector of values and maps it to a 3D array with specified ranges
 * for the x, y, and z indices. The ranges allow the 3D array indices to start at arbitrary
 * values instead of zero. The input vector must have a size equal to the total number of
 * elements in the 3D array, calculated as:
 *     (xEnd - xStart + 1) * (yEnd - yStart + 1) * (zEnd - zStart + 1).
 *
 * @param inputVector The 1D vector containing the values in row-major order.
 * @param xStart The starting index for the x dimension.
 * @param xEnd The ending index for the x dimension.
 * @param yStart The starting index for the y dimension.
 * @param yEnd The ending index for the y dimension.
 * @param zStart The starting index for the z dimension.
 * @param zEnd The ending index for the z dimension.
 * @return A 3D array (vector of vectors of vectors) containing the mapped values.
 * @throws std::invalid_argument If the size of the input vector does not match the expected
 *         number of elements in the 3D array based on the specified ranges.
 */
std::vector<std::vector<std::vector<int>>> AppAdditiveExtTempTexture::convert_to_3d_array_with_range(std::vector<int>& inputVector, 
    int xStart, int xEnd, 
    int yStart, int yEnd, 
    int zStart, int zEnd) {
    // Calculate the sizes of each dimension
    int xSize = xEnd - xStart;
    int ySize = yEnd - yStart;
    int zSize = zEnd - zStart;

    // Ensure the input vector has the correct size

    if (inputVector.size() != xSize * ySize * zSize) {
        throw std::invalid_argument("Input vector size does not match the specified dimensions.");
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

void AppAdditiveExtTempTexture::reduced_temperature_hdf_chunked(){
  timer->stamp();
  
  if (domain->me == 0) std::cout << "Starting chunked HDF5 reading from: " << temp_file_string << std::endl;
  if (domain->me == 0) std::cout << "Opening HDF5 file..." << std::endl;
  
  //Initialize chunked reading - open HDF5 file and datasets for reuse
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
  hdf5_file_id = H5Fopen(temp_file_string.c_str(), H5F_ACC_RDONLY, fapl_id);
  H5Pclose(fapl_id);
  
  if (hdf5_file_id < 0) {
    error->all(FLERR,"Cannot open HDF5 file for chunked reading");
  }
  
  //Open datasets for reuse
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
  //Read data_counts array once (small and needed for all chunks)
  load_data_counts_array();
  if (domain->me == 0) std::cout << "Data counts loaded, initializing time cache..." << std::endl;
  
  //Initialize time cache for better performance
  initialize_time_cache();
  if (domain->me == 0) std::cout << "Time cache initialized, loading first chunk..." << std::endl;
  
  //Load the first chunk
  current_chunk_start_time = 0.0;
  current_chunk_end_time = chunk_time_window;
  load_next_chunk();
  if (domain->me == 0) std::cout << "First chunk loaded successfully" << std::endl;
  
  if (domain->me == 0) std::cout << "Initial chunk loaded: [" << current_chunk_start_time << ", " << current_chunk_end_time << ")" << std::endl;
}


void AppAdditiveExtTempTexture::load_data_counts_array(){
  
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
  hsize_t start_counts[3] = { (hsize_t)xlo, (hsize_t)ylo, (hsize_t)zlo};
  hsize_t count_counts[3] = { (hsize_t)subdomain_x_size, (hsize_t)subdomain_y_size, (hsize_t)subdomain_z_size };
  
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
  // Use ceiling for upper bounds to include all sites in this processor's domain
  int xlo2 = (int)domain->subxlo;
  int xhi2 = (int)ceil(domain->subxhi);
  int ylo2 = (int)domain->subylo;
  int yhi2 = (int)ceil(domain->subyhi);
  int zlo2 = (int)domain->subzlo;
  int zhi2 = (int)ceil(domain->subzhi);
  
  
  data_counts_array = convert_to_3d_array_with_range(data_counts, xlo2, xhi2, ylo2, yhi2, zlo2, zhi2);
  
  if (domain->me == 0) std::cout << "Data counts array loaded" << std::endl;
  
  timer->stamp(TIME_APP);
}

void AppAdditiveExtTempTexture::load_next_chunk(){
  timer->stamp();
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Clear existing data in queues
  for (int x = 0; x < data_counts_array.size(); x++) {
    for (int y = 0; y < data_counts_array[x].size(); y++) {
      for (int z = 0; z < data_counts_array[x][y].size(); z++) {
        // Clear the queues completely
        while (!temp_in(x,y,z).empty()) temp_in(x,y,z).pop();
        while (!time_in(x,y,z).empty()) time_in(x,y,z).pop();
      }
    }
  }
  
  // Setup parallel reading (only for temperature data now)
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
  hid_t data_type_temp = H5Dget_type(hdf5_temp_dataset);
  
  // Load data for each local site using cached time data
  for (int local_index = 0; local_index < nlocal; local_index++) {
    
    // Use consistent coordinate calculation matching the array sizing
    int x_loc = (int)xyz[local_index][0] - (int)domain->subxlo;
    int y_loc = (int)xyz[local_index][1] - (int)domain->subylo;
    int z_loc = (int)xyz[local_index][2] - (int)domain->subzlo;
    int x = (int)xyz[local_index][0];
    int y = (int)xyz[local_index][1];
    int z = (int)xyz[local_index][2];
    
    // Enhanced bounds checking for chunk loading
    try {
      if (x_loc < 0 || x_loc >= (int)data_counts_array.size() ||
          y_loc < 0 || y_loc >= (int)data_counts_array[x_loc].size() ||
          z_loc < 0 || z_loc >= (int)data_counts_array[x_loc][y_loc].size()) {
        if (domain->me == 0) {
          std::cout << "Chunk loading: index out of bounds for site " << local_index 
                    << ": (" << x_loc << "," << y_loc << "," << z_loc << ") array_size(" 
                    << data_counts_array.size() << "," 
                    << (x_loc >= 0 && x_loc < (int)data_counts_array.size() ? data_counts_array[x_loc].size() : -1) << ","
                    << (x_loc >= 0 && x_loc < (int)data_counts_array.size() && y_loc >= 0 && y_loc < (int)data_counts_array[x_loc].size() ? data_counts_array[x_loc][y_loc].size() : -1) << ")" << std::endl;
        }
        continue;
      }
    } catch (const std::exception& e) {
      std::cout << "Exception in chunk loading bounds check: " << e.what() << " for site " << local_index 
                << " coords(" << x_loc << "," << y_loc << "," << z_loc << ") on processor " << domain->me << std::endl;
      continue;
    }
    
    int valid_count;
    try {
      valid_count = data_counts_array[x_loc][y_loc][z_loc];
    } catch (const std::exception& e) {
      std::cout << "Exception accessing data_counts_array in load_next_chunk[" << x_loc << "][" << y_loc << "][" << z_loc 
                << "]: " << e.what() << " on processor " << domain->me << std::endl;
      continue;
    }
    
    try {
      if (valid_count > 0 && time_cache_loaded[x_loc][y_loc][z_loc]) {
        
        // Use cached time values to find the range we need
        const std::vector<double>& time_values = cached_time_values[x_loc][y_loc][z_loc];
      
      // Find the index range that falls within our time window with buffer
      // Include extra data before and after for interpolation
      double buffer_time = chunk_time_window * 0.2; // 20% buffer on each side
      double extended_start = current_chunk_start_time - buffer_time;
      double extended_end = current_chunk_end_time + buffer_time;
      
      int start_idx = -1, end_idx = -1;
      for (int k = 0; k < valid_count; ++k) {
        if (time_values[k] >= extended_start && time_values[k] <= extended_end) {
          if (start_idx == -1) start_idx = k;
          end_idx = k;
        }
      }
      
      // Only read the needed temperature data if we found valid time indices
      if (start_idx != -1 && end_idx != -1) {
        int needed_count = end_idx - start_idx + 1;
        std::vector<double> temp_values(needed_count);
        
        // Read only the needed slice of temperature data
        hsize_t temp_start[4] = { (hsize_t)x, (hsize_t)y, (hsize_t)z, (hsize_t)start_idx};
        hsize_t temp_count[4] = { 1, 1, 1, (hsize_t)needed_count};
        
        hid_t temperature_filespace_id = H5Dget_space(hdf5_temp_dataset);
        H5Sselect_hyperslab(temperature_filespace_id, H5S_SELECT_SET, temp_start, NULL, temp_count, NULL);
        hid_t temp_memspace_id = H5Screate_simple(4, temp_count, NULL);
        H5Dread(hdf5_temp_dataset, data_type_temp, temp_memspace_id, temperature_filespace_id, plist_id, temp_values.data());
        H5Sclose(temperature_filespace_id);
        H5Sclose(temp_memspace_id);
        
        // Store the filtered values directly
        for (int k = 0; k < needed_count; ++k) {
          temp_in(x_loc,y_loc,z_loc).push(temp_values[k]);
          time_in(x_loc,y_loc,z_loc).push(time_values[start_idx + k]);
        }
      }
      } // closing brace for if (valid_count > 0 && time_cache_loaded[x_loc][y_loc][z_loc])
    } catch (const std::exception& e) {
      std::cout << "Exception in load_next_chunk cache access: " << e.what() 
                << " for site " << local_index << " coords(" << x_loc << "," << y_loc << "," << z_loc 
                << ") on processor " << domain->me << std::endl;
      continue;
    }
  }
  
  // Clean up
  H5Tclose(data_type_temp);
  H5Pclose(plist_id);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  // Count how many sites got data loaded in this chunk
  int sites_with_data = 0;
  for (int x = 0; x < data_counts_array.size(); x++) {
    for (int y = 0; y < data_counts_array[x].size(); y++) {
      for (int z = 0; z < data_counts_array[x][y].size(); z++) {
        if (!temp_in(x,y,z).empty()) sites_with_data++;
      }
    }
  }
  
  if (domain->me == 0) {
    std::cout << "Loaded chunk [" << current_chunk_start_time << ", " << current_chunk_end_time 
              << ") in " << duration.count() << " ms, " << sites_with_data << " sites got data" << std::endl;
  }
  
  timer->stamp(TIME_APP);
}

bool AppAdditiveExtTempTexture::needs_new_chunk(double simulation_time) {
  
  // Check if we are approaching the end of current chunk
  double buffer_time = chunk_time_window * 0.1; // 10% buffer
  return (simulation_time >= (current_chunk_end_time - buffer_time));
}

void AppAdditiveExtTempTexture::initialize_time_cache() {
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Initialize cache arrays with exact same dimensions as data_counts_array
  // This ensures perfect index compatibility
  int x_size = data_counts_array.size();
  int y_size = (x_size > 0) ? data_counts_array[0].size() : 0;
  int z_size = (y_size > 0) ? data_counts_array[0][0].size() : 0;
  
  if (domain->me == 0) {
    std::cout << "Initializing cache arrays with dimensions: " << x_size << " x " << y_size << " x " << z_size << std::endl;
    std::cout << "Subdomain: [" << domain->subxlo << "-" << domain->subxhi << ") x [" 
              << domain->subylo << "-" << domain->subyhi << ") x [" 
              << domain->subzlo << "-" << domain->subzhi << ")" << std::endl;
  }
  
  cached_time_values.resize(x_size);
  time_cache_loaded.resize(x_size);
  
  for (int x = 0; x < x_size; x++) {
    cached_time_values[x].resize(y_size);
    time_cache_loaded[x].resize(y_size);
    for (int y = 0; y < y_size; y++) {
      cached_time_values[x][y].resize(z_size);
      time_cache_loaded[x][y].resize(z_size, false);
    }
  }
  
  // Setup parallel reading for time cache
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
  hid_t data_type_time = H5Dget_type(hdf5_time_dataset);
  
  // Track statistics for debugging
  int sites_processed = 0;
  int sites_with_data = 0;
  int sites_skipped_bounds = 0;
  int sites_no_data = 0;
  
  // Read time data for all local sites and cache it
  for (int local_index = 0; local_index < nlocal; local_index++) {
    
    int x_loc = (int)xyz[local_index][0] - (int)domain->subxlo;
    int y_loc = (int)xyz[local_index][1] - (int)domain->subylo;
    int z_loc = (int)xyz[local_index][2] - (int)domain->subzlo;
    int x = (int)xyz[local_index][0];
    int y = (int)xyz[local_index][1];
    int z = (int)xyz[local_index][2];
    
    sites_processed++;
    
    // Bounds checking to catch indexing issues
    if (x_loc < 0 || x_loc >= (int)data_counts_array.size()) {
      if (domain->me == 0) {
        std::cout << "X index out of bounds for site " << local_index 
                  << ": x_loc=" << x_loc << ", array x_size=" << data_counts_array.size() << std::endl;
      }
      sites_skipped_bounds++;
      continue;
    }
    if (y_loc < 0 || y_loc >= (int)data_counts_array[x_loc].size()) {
      if (domain->me == 0) {
        std::cout << "Y index out of bounds for site " << local_index 
                  << ": y_loc=" << y_loc << ", array y_size=" << data_counts_array[x_loc].size() << std::endl;
      }
      sites_skipped_bounds++;
      continue;
    }
    if (z_loc < 0 || z_loc >= (int)data_counts_array[x_loc][y_loc].size()) {
      if (domain->me == 0) {
        std::cout << "Z index out of bounds for site " << local_index 
                  << ": z_loc=" << z_loc << ", array z_size=" << data_counts_array[x_loc][y_loc].size() << std::endl;
      }
      sites_skipped_bounds++;
      continue;
    }
    
    int valid_count = data_counts_array[x_loc][y_loc][z_loc];
    
    if (valid_count > 0) {
      sites_with_data++;
      
      // Cache all time values for this location
      cached_time_values[x_loc][y_loc][z_loc].resize(valid_count);
      
      hsize_t start[4] = { (hsize_t)x, (hsize_t)y, (hsize_t)z, 0};
      hsize_t count[4] = { 1, 1, 1, (hsize_t)valid_count};
      
      hid_t time_filespace_id = H5Dget_space(hdf5_time_dataset);
      H5Sselect_hyperslab(time_filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);
      hid_t memspace_id = H5Screate_simple(4, count, NULL);
      H5Dread(hdf5_time_dataset, data_type_time, memspace_id, time_filespace_id, plist_id, cached_time_values[x_loc][y_loc][z_loc].data());
      H5Sclose(time_filespace_id);
      H5Sclose(memspace_id);
      
      time_cache_loaded[x_loc][y_loc][z_loc] = true;
    }
    else {
      sites_no_data++;
      // Debug: Print locations of first few sites with no data
      if (sites_no_data <= 10 && domain->me == 0) {
        std::cout << "Site with no data: local_index=" << local_index 
                  << " global_coords=(" << x << "," << y << "," << z << ")"
                  << " local_coords=(" << x_loc << "," << y_loc << "," << z_loc << ")"
                  << " valid_count=" << valid_count << std::endl;
      }
    }
  }
  
  // Clean up
  H5Tclose(data_type_time);
  H5Pclose(plist_id);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  // Fill missing data by interpolating from neighbors
  if (sites_no_data > 0) {
    fill_missing_temperature_data();
    if (domain->me == 0) {
      std::cout << "Filled " << sites_no_data << " sites with missing data using spatial interpolation" << std::endl;
    }
  }
  
  if (domain->me == 0) {
    std::cout << "Time cache initialization completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Sites processed: " << sites_processed << ", with data: " << sites_with_data 
              << ", no data: " << sites_no_data << ", bounds errors: " << sites_skipped_bounds << std::endl;
  }
}

void AppAdditiveExtTempTexture::fill_missing_temperature_data() {
  
  auto start_time = std::chrono::high_resolution_clock::now();
  int filled_sites = 0;
  
  // Setup HDF5 reading for temperature data
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
  hid_t data_type_temp = H5Dget_type(hdf5_temp_dataset);
  
  // Find all sites with missing data and try to interpolate from neighbors
  for (int local_index = 0; local_index < nlocal; local_index++) {
    
    int x_loc = (int)xyz[local_index][0] - (int)domain->subxlo;
    int y_loc = (int)xyz[local_index][1] - (int)domain->subylo;
    int z_loc = (int)xyz[local_index][2] - (int)domain->subzlo;
    int x = (int)xyz[local_index][0];
    int y = (int)xyz[local_index][1];
    int z = (int)xyz[local_index][2];
    
    // Enhanced bounds checking with debug output
    try {
      if (x_loc < 0 || x_loc >= (int)data_counts_array.size() ||
          y_loc < 0 || y_loc >= (int)data_counts_array[x_loc].size() ||
          z_loc < 0 || z_loc >= (int)data_counts_array[x_loc][y_loc].size()) {
        if (domain->me == 0 && local_index < 10) {
          std::cout << "Bounds check failed: site " << local_index << " global(" << x << "," << y << "," << z 
                    << ") local(" << x_loc << "," << y_loc << "," << z_loc << ") array_size(" 
                    << data_counts_array.size() << "," 
                    << (x_loc >= 0 && x_loc < (int)data_counts_array.size() ? data_counts_array[x_loc].size() : -1) << ","
                    << (x_loc >= 0 && x_loc < (int)data_counts_array.size() && y_loc >= 0 && y_loc < (int)data_counts_array[x_loc].size() ? data_counts_array[x_loc][y_loc].size() : -1) << ")" << std::endl;
        }
        continue;
      }
    } catch (const std::exception& e) {
      std::cout << "Exception in bounds checking: " << e.what() << " for site " << local_index 
                << " coords(" << x_loc << "," << y_loc << "," << z_loc << ")" << std::endl;
      continue;
    }
    
    int valid_count;
    try {
      valid_count = data_counts_array[x_loc][y_loc][z_loc];
    } catch (const std::exception& e) {
      std::cout << "Exception accessing data_counts_array[" << x_loc << "][" << y_loc << "][" << z_loc 
                << "]: " << e.what() << " on processor " << domain->me << std::endl;
      continue;
    }
    
    // Only process sites that have no data
    if (valid_count == 0) {
      
      // Check 6 nearest neighbors (±1 in x, y, z directions)
      int dx[] = {-1, 1, 0, 0, 0, 0};
      int dy[] = {0, 0, -1, 1, 0, 0};
      int dz[] = {0, 0, 0, 0, -1, 1};
      
      // Find first neighbor with data to copy from
      for (int n = 0; n < 6; n++) {
        int nx_loc = x_loc + dx[n];
        int ny_loc = y_loc + dy[n];
        int nz_loc = z_loc + dz[n];
        int nx = x + dx[n];
        int ny = y + dy[n];
        int nz = z + dz[n];
        
        // Check bounds for neighbor
        if (nx_loc >= 0 && nx_loc < (int)data_counts_array.size() &&
            ny_loc >= 0 && ny_loc < (int)data_counts_array[nx_loc].size() &&
            nz_loc >= 0 && nz_loc < (int)data_counts_array[nx_loc][ny_loc].size()) {
          
          // Check if neighbor has data
          try {
            if (time_cache_loaded[nx_loc][ny_loc][nz_loc] && 
                !cached_time_values[nx_loc][ny_loc][nz_loc].empty()) {
              
              // Copy neighbor's time data
              cached_time_values[x_loc][y_loc][z_loc] = cached_time_values[nx_loc][ny_loc][nz_loc];
              time_cache_loaded[x_loc][y_loc][z_loc] = true;
            
              // Update data_counts_array to reflect that this site now has data
              data_counts_array[x_loc][y_loc][z_loc] = cached_time_values[x_loc][y_loc][z_loc].size();
              
              filled_sites++;
              
              if (filled_sites <= 5 && domain->me == 0) {
                std::cout << "Filled site (" << x_loc << "," << y_loc << "," << z_loc 
                          << ") from neighbor (" << nx_loc << "," << ny_loc << "," << nz_loc 
                          << ") with " << cached_time_values[x_loc][y_loc][z_loc].size() << " data points" << std::endl;
              }
              
              break; // Found a neighbor, move to next missing site
            }
          } catch (const std::exception& e) {
            std::cout << "Exception in spatial interpolation: " << e.what() 
                      << " accessing cache[" << nx_loc << "][" << ny_loc << "][" << nz_loc << "] or [" 
                      << x_loc << "][" << y_loc << "][" << z_loc << "] on processor " << domain->me << std::endl;
            continue;
          }
        }
      }
    }
  }
  
  // Clean up
  H5Tclose(data_type_temp);
  H5Pclose(plist_id);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  if (domain->me == 0) {
    std::cout << "Spatial interpolation filled " << filled_sites 
              << " sites in " << duration.count() << " ms" << std::endl;
  }
}

void AppAdditiveExtTempTexture::close_hdf5_file() {
  
  if (hdf5_file_open) {
    H5Dclose(hdf5_count_dataset);
    H5Dclose(hdf5_temp_dataset);
    H5Dclose(hdf5_time_dataset);
    H5Fclose(hdf5_file_id);
    hdf5_file_open = false;
    
    if (domain->me == 0) std::cout << "HDF5 file closed" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Get HDF5 data dimensions from data_counts and origin from x0
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::get_hdf5_dimensions() {
  
  if (!hdf5_file_open) {
    error->all(FLERR,"HDF5 file must be open to get dimensions");
  }
  
  // Get dataspace from data_counts dataset (3D: x,y,z)
  hid_t dataspace = H5Dget_space(hdf5_count_dataset);
  
  // Get number of dimensions
  int ndims = H5Sget_simple_extent_ndims(dataspace);
  if (ndims != 3) {
    H5Sclose(dataspace);
    error->all(FLERR,"HDF5 data_counts dataset must have 3 dimensions (x,y,z)");
  }
  
  // Get dimension sizes from data_counts
  H5Sget_simple_extent_dims(dataspace, hdf5_dims, NULL);
  H5Sclose(dataspace);
  
  // Read x0 origin data if it exists
  // Check if x0 dataset exists
  htri_t exists = H5Lexists(hdf5_file_id, "x0", H5P_DEFAULT);
  if (exists > 0) {
    // Open x0 dataset
    hid_t x0_dataset = H5Dopen(hdf5_file_id, "x0", H5P_DEFAULT);
    
    // Get dataspace
    hid_t x0_space = H5Dget_space(x0_dataset);
    
    // Get dimensions
    hsize_t x0_dims[1];
    int x0_ndims = H5Sget_simple_extent_dims(x0_space, x0_dims, NULL);
    
    if (x0_ndims == 1 && x0_dims[0] >= 3) {
      // Read the origin values
      double x0_data[3];
      hid_t x0_type = H5Dget_type(x0_dataset);
      H5Dread(x0_dataset, x0_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, x0_data);
      
      // Store origin
      hdf5_origin[0] = x0_data[0];
      hdf5_origin[1] = x0_data[1];
      hdf5_origin[2] = x0_data[2];
      
      H5Tclose(x0_type);
    }
    else {
      if (domain->me == 0) {
        std::cout << "Warning: x0 dataset exists but has unexpected dimensions. Using origin (0,0,0)." << std::endl;
      }
    }
    
    H5Sclose(x0_space);
    H5Dclose(x0_dataset);
  }
  else {
    if (domain->me == 0) {
      std::cout << "Note: No x0 dataset found. Using origin (0,0,0)." << std::endl;
    }
  }
  
  if (domain->me == 0) {
    std::cout << "HDF5 data dimensions (from data_counts): " << hdf5_dims[0] << " x " 
              << hdf5_dims[1] << " x " << hdf5_dims[2] << std::endl;
    std::cout << "HDF5 data origin (x0): (" << hdf5_origin[0] << ", " 
              << hdf5_origin[1] << ", " << hdf5_origin[2] << ")" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Validate simulation bounds against HDF5 data dimensions
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::validate_simulation_bounds() {
  
  // Get HDF5 dimensions first
  get_hdf5_dimensions();
  
  // Get global simulation domain bounds
  int sim_nx = (int)(domain->boxxhi - domain->boxxlo);
  int sim_ny = (int)(domain->boxyhi - domain->boxylo);
  int sim_nz = (int)(domain->boxzhi - domain->boxzlo);
  
  if (domain->me == 0) {
    std::cout << "Simulation domain: size(" << sim_nx << " x " 
              << sim_ny << " x " << sim_nz << ") at [" 
              << domain->boxxlo << "," << domain->boxxhi << "] x ["
              << domain->boxylo << "," << domain->boxyhi << "] x ["
              << domain->boxzlo << "," << domain->boxzhi << "]" << std::endl;
    std::cout << "HDF5 data: size(" << hdf5_dims[0] << " x " 
              << hdf5_dims[1] << " x " << hdf5_dims[2] << ") with origin ("
              << hdf5_origin[0] << "," << hdf5_origin[1] << "," << hdf5_origin[2] << ")" << std::endl;
    std::cout << "Bounds check mode: " << (bounds_check_mode == 0 ? "exact match" : "subvolume") << std::endl;
  }
  
  if (bounds_check_mode == 0) {
    // Exact match mode - simulation and HDF5 dimensions must be identical
    // For exact match, we expect the simulation to start at (0,0,0) matching HDF5 origin convention
    if (sim_nx != (int)hdf5_dims[0] || sim_ny != (int)hdf5_dims[1] || sim_nz != (int)hdf5_dims[2]) {
      std::ostringstream error_msg;
      error_msg << "Simulation domain size (" << sim_nx << "x" << sim_ny << "x" << sim_nz 
                << ") does not exactly match HDF5 data size (" 
                << hdf5_dims[0] << "x" << hdf5_dims[1] << "x" << hdf5_dims[2] << "). "
                << "Use bounds_check_mode 1 for subvolume mode or adjust domain size.";
      error->all(FLERR,error_msg.str().c_str());
    }
    
    // Warn if simulation doesn't start at origin
    if ((domain->boxxlo != 0.0 || domain->boxylo != 0.0 || domain->boxzlo != 0.0) && domain->me == 0) {
      std::cout << "Warning: In exact match mode, simulation domain should typically start at (0,0,0)" << std::endl;
    }
    
    if (domain->me == 0) std::cout << "✓ Exact match validation passed" << std::endl;
  }
  else if (bounds_check_mode == 1) {
    // Subvolume mode - simulation domain must fit within HDF5 data bounds
    // Account for HDF5 origin offset
    
    // Calculate effective HDF5 bounds in simulation coordinate system
    double hdf5_xlo = hdf5_origin[0];
    double hdf5_xhi = hdf5_origin[0] + hdf5_dims[0];
    double hdf5_ylo = hdf5_origin[1];
    double hdf5_yhi = hdf5_origin[1] + hdf5_dims[1];
    double hdf5_zlo = hdf5_origin[2];
    double hdf5_zhi = hdf5_origin[2] + hdf5_dims[2];
    
    if (domain->me == 0) {
      std::cout << "HDF5 effective bounds: [" << hdf5_xlo << "," << hdf5_xhi << "] x ["
                << hdf5_ylo << "," << hdf5_yhi << "] x ["
                << hdf5_zlo << "," << hdf5_zhi << "]" << std::endl;
    }
    
    // Check if simulation domain is within HDF5 bounds
    if (domain->boxxlo < hdf5_xlo || domain->boxxhi > hdf5_xhi ||
        domain->boxylo < hdf5_ylo || domain->boxyhi > hdf5_yhi ||
        domain->boxzlo < hdf5_zlo || domain->boxzhi > hdf5_zhi) {
      std::ostringstream error_msg;
      error_msg << "Simulation domain [" << domain->boxxlo << "," << domain->boxxhi << "] x ["
                << domain->boxylo << "," << domain->boxyhi << "] x ["
                << domain->boxzlo << "," << domain->boxzhi << "] "
                << "is not contained within HDF5 data bounds ["
                << hdf5_xlo << "," << hdf5_xhi << "] x ["
                << hdf5_ylo << "," << hdf5_yhi << "] x ["
                << hdf5_zlo << "," << hdf5_zhi << "]. "
                << "Adjust simulation domain to fit within HDF5 data.";
      error->all(FLERR,error_msg.str().c_str());
    }
    
    if (domain->me == 0) std::cout << "✓ Subvolume validation passed" << std::endl;
  }
  
  bounds_validated = true;
  if (domain->me == 0) std::cout << "Bounds validation completed successfully" << std::endl;
}

/* ----------------------------------------------------------------------
   Setup modular temperature source command
   Format: setup_temperature_source type [arguments...]
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::setup_temperature_source_cmd(int narg, char **arg)
{
  if (narg < 1) {
    error->all(FLERR,"Illegal setup_temperature_source command: must specify type");
  }
  
  std::string source_type = arg[0];
  std::vector<std::string> source_args;
  
  // Convert remaining arguments to string vector
  for (int i = 1; i < narg; i++) {
    source_args.push_back(std::string(arg[i]));
  }
  
  // Create temperature source using factory
  temperature_source = SPPARKS_NS::create_temperature_source(source_type, spk);
  if (!temperature_source) {
    error->all(FLERR,"Failed to create temperature source");
  }
  
  // Setup the temperature source with provided arguments
  temperature_source->setup_temperature_source(source_args);
  
  // Enable the new temperature source system
  use_temperature_source = true;
  
  if (domain->me == 0) {
    std::cout << "Modular temperature source (" << source_type << ") enabled" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Setup scan path command for temperature sources that support it
   Format: scan_path type [parameters...]
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::setup_scan_path_cmd(int narg, char **arg)
{
  if (!use_temperature_source || !temperature_source) {
    error->all(FLERR,"scan_path command requires setup_temperature_source to be called first");
  }
  
  if (narg < 1) {
    error->all(FLERR,"Illegal scan_path command: must specify type");
  }
  
  // Try to cast to RosenthalTemperatureSource (only source that currently supports scan paths)
  auto rosenthal_source = dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());
  if (!rosenthal_source) {
    error->all(FLERR,"scan_path command only supported for Rosenthal temperature source");
  }
  
  std::string path_type_str = arg[0];
  std::vector<double> path_params;
  
  // Convert remaining arguments to double vector
  for (int i = 1; i < narg; i++) {
    try {
      path_params.push_back(std::stod(arg[i]));
    } catch (const std::exception &e) {
      error->all(FLERR,"Invalid numeric argument in scan_path command");
    }
  }
  
  // Map string to enum
  ScanPathType path_type;
  if (path_type_str == "linear") {
    path_type = LINEAR;
  } else if (path_type_str == "serpentine") {
    path_type = SERPENTINE;
  } else if (path_type_str == "spiral") {
    path_type = SPIRAL;
  } else if (path_type_str == "custom") {
    path_type = CUSTOM;
  } else {
    error->all(FLERR,"Unknown scan path type in scan_path command");
  }
  
  // Setup scan path
  rosenthal_source->setup_scan_path(path_type, path_params);
  
  if (domain->me == 0) {
    std::cout << "Scan path (" << path_type_str << ") configured" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Update temperature from modular source
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::update_temperature_from_source(double simulation_time)
{
  timer->stamp();
  
  if (!use_temperature_source || !temperature_source) {
    timer->stamp(TIME_APP);
    return; // Fall back to legacy HDF5 system
  }
  
  // Update temperature source (may load new data)
  temperature_source->update_temperatures(dt, simulation_time);
  
  // Update temperature array for all local sites
  for (int i = 0; i < nlocal; i++) {
    // Get site coordinates in lattice units
    double x_lattice = xyz[i][0];
    double y_lattice = xyz[i][1];
    double z_lattice = xyz[i][2];
    
    // Convert to physical coordinates (meters) using dx from temperature source
    // For HDF5 unstructured source, dx is the lattice spacing in meters
    HDF5UnstructuredTemperatureSource* hdf5_source = 
      dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
    
    double x_physical, y_physical, z_physical;
    if (hdf5_source) {
      // HDF5 source expects physical coordinates in meters
      double dx = hdf5_source->get_dx();
      x_physical = x_lattice * dx;
      y_physical = y_lattice * dx;
      z_physical = z_lattice * dx;
    } else {
      // Other temperature sources might expect lattice coordinates
      x_physical = x_lattice;
      y_physical = y_lattice;
      z_physical = z_lattice;
    }
    
    // Use site-based temperature access for HDF5 unstructured source (uses caching)
    if (hdf5_source) {
      T[i] = temperature_source->get_temperature_at_site(i, simulation_time);
    } else {
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
  }
  
  timer->stamp(TIME_APP);
}

