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

   This is an extension of the external_temperature branch. This branch will have the same basic functions a external_temperature,
   but will introduce experimentally-derived parameters. These will be applied in separate steps of nucleation and grain growth.
   
   Modifications needed for the code:
   General
   1. Introduce additional input variables:
    a) dx - lattice spacing (in m)
    b) dt - timestep (assume constant in seconds)
    c) no - Max nucleation site density (in m^-3)
    d) k1 - kinetic parameter (in m^2/s)
    e) Q - activation energy (in J/mol)
    f) tc - critical undercooling (K)
    g) tsig - standard deviation of undercooling Gaussian
   2. Introduce a check to see if we're in the mushy zone.
    a) If above T_l, make random & molten (as we do now)
    b) If between T_l and T_s, use a "nucleation" type rule
    c) 
   Nucleation
   1. Create a new array to hold critical undercooling for each spin (size of nspins)
   2. Write a function to sample from undercooling distribution and give -1 if spin not allowed to nucleate
   3. Change the site_event_rejection function
    a) If a molten site cannot nucleate (undercooling temp less than 0)
     i) Check if the site's temperature is below the liquidus and above liquidus
     ii) If so, force the site to change to the grain ID of a solid neighbor site (maybe)
         It might be better to check if a solid site is within a distance v * dt to get captured by the solidification front
    b) If the molten site needs to nucleate
     i) Check if the site's T[i] is l.t.e. the spin's Tn
     ii) If so, retain the spin and switch its phase to solid
    c) If a solid's site's temperature is above the solidus
     i) Do not allow the solid grain to switch to the liquid state.
     ii) Allow switching between solid states, but use mobility expression based on undercooling (or some type of dendrite envelope expression)...
   Grain Growth
   1. Change the boundary mobility equation to incorporate k1, dt, and Q.
   2. Don't let "solid" material switch to molten spins
   
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
    ndouble = 7;
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
    no = 1e15;
    tc = 5;
    tsig = 3;
    size_norm = pow(dx,3) * 2;
    size_sig = pow(dx,3);
    solid_front_length = 4;
    //Let's put in default values for arrays
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
    
    // Initialize bounds checking variables
    bounds_check_mode = 0; // Default to exact match
    bounds_validated = false;
    hdf5_dims[0] = hdf5_dims[1] = hdf5_dims[2] = 0;
    hdf5_origin[0] = hdf5_origin[1] = hdf5_origin[2] = 0.0;
    
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
  
  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
  When app_update is called, read in an input file
 ------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::app_update(double dt)
{

  //Reset t_active to assume we don't have a melt pool right now
  t_active = 0;
  
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
    
        //Turn the sites on/off depending on the phase data and whether or not the
        //site's temperature has gone above tl. Only do this when melting the first time, so we don't have to 
        //do it over and over
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
        }
        //If we're molten, call the mushy_phase function to figure out any phase change
        else if (active_flag[i] == 2 && T[i] <= tl) {
            mushy_phase(i, ranapp);
//             fprintf(screen,"Ran mushy_phase\n");
      } 
      else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3)    {
              mobility_out[i] = 1;
              site_event_rejection(i, ranapp);
              solid_d[i]--;
      }
      
    }
    
    //Communicate changes
    comm->all();
    // Use MPI_Allreduce to check if t_active is 0 on all processors
    int global_t_active;
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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
    nucleation_flags = new int[nspins];
    double nucleationFraction = dx * dx * dx * no;
    
    //Make all spins nucleation sites. Should avoid this. Maybe return an error instead?
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

  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
  
	//Initialize the nucleation_flags vector
	if (domain->me==0) {
		nucleation_spins(ranapp);    
	}

	MPI_Bcast(nucleation_flags,nspins, MPI_INT,0,world);

	//Initialize the nucleation_temps and nucleation_sizes vectors
	if (domain->me==0) {
			nucleation_init();    
	}

	MPI_Bcast(nucleation_temps,nspins, MPI_DOUBLE,0,world);
	MPI_Bcast(nucleation_sizes,nspins, MPI_DOUBLE,0,world);
	
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


  
  //Read in our temperature values -- for now we're doing this all at once but might need to do in chunks.
  //Initialize chunked reading variables
  chunk_time_window = 0.1; // Default 100ms time window per chunk
  current_chunk_start_time = 0.0;
  current_chunk_end_time = 0.0;
  hdf5_file_open = false;
  
  if (domain->me == 0) {
    std::cout << "Starting chunked HDF5 reading" << std::endl;
    std::cout << "Using liquidus temperature tl = " << tl << " K" << std::endl;
    std::cout << "Using solidus temperature ts = " << ts << " K" << std::endl;
  }
  reduced_temperature_hdf_chunked();
  if (domain->me == 0) std::cout << "Chunked HDF5 reading completed" << std::endl;

	this->app_update(0.0);
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
  int oldstate = spin[i];
  SiteState s_old(spin[i], {q0[i], qx[i], qy[i], qz[i]});
  double einitial = site_energy(i);
  double efinal = 0;
  double Mobloc = 0;
  double dotValue = 0;
    
/* ----------------------------------------------------------------------
   Define variables to identify melt travel distance, specific melt location,
   and the height of each layer as a function of time after additive 
   manufacturing process starts.
   
   The melt travel pattern depends on the value of direction_switch: 0 = scan
   in x and y directions, 1 = scan in x direction only, 2 = scan in y direction only.
   For x & y scans, odd layers are x-oriented and even layers are y-oriented.
   For all patterns, odd passes will travel in the postive direction of the specified
   variable and even passes will travel in the negative direction.
------------------------------------------------------------------------- */

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
            if(active_flag[neighbor[i][j]] == 3) {
                //Calculate temp gradient/grain misorientation and store in "unique" array
                //We should make this cumulative, so we can use a random number to sample it
                //Exclude gas or molten sites from the Potts neighbor tally
                dotValue += melt_misorientation(neighbor[i][j],c1,c2,c3);
                unique_dot[nevent] = dotValue;
                value = spin[neighbor[i][j]];
                unique[nevent] = value;
                unique_neigh[nevent] = neighbor[i][j];
                nevent++;										
            }
        }
        //If no neighbor is eligible, return before changing anything. Will try next sweep.
        if (nevent == 0) return;
        //I think we should use nevent -1 (there will be an extra event at the end)
        double dran = (unique_dot[nevent - 1]*random->uniform());
        //if (iran >= nevent) iran = nevent-1;
        //Go through possible events and pick one
        for( int j = 0; j < nevent -1; j++) {
            if(dran <= unique_dot[j]) {
              int neighran = unique_neigh[j];
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
     }
  }
  else if (temperature == 0.0) {
    flip_site(i, s_old);
  } 
  else if (random->uniform() > Mobloc * exp((einitial-efinal)*t_inverse)) {
    flip_site(i, s_old);
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
            //Call nucleation particle flipper
            naccept++;
            nucleation_particle_flipper(i, round(nucleation_sizes[spin[i]]/pow(dx,3)), ranapp);
            return;
        }
        //Can nucleate, but won't yet. Allow to if the solidification front gets captured.
        else {
            fprintf(screen,"Epitaxialy growing instead of nucleating\n");
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
    
    //Add the distance of the front travel. This is for 304L. Need to multiply by timestep to get distance
    //Try doing this with an arbitrary array
    int power = solid_front_length - 1;
    for(int k = 0; k < solid_front_length; k++) {
        solid_d[i] = solid_d[i] + solid_front_coeffs[k] * pow(Tcool, power) * time_step;
        power--;
    }
    
    //Go through neighbor list and add them to possible switches
    for (int j = 0; j < numneigh[i]; j++) {
        if(neigh_dist[j] <= solid_d[i] && (active_flag[neighbor[i][j]] == 1 || active_flag[neighbor[i][j]] == 3)) {
            //Calculate temp gradient/grain misorientation and store in "unique" array
            //We should make this cumulative, so we can use a random number to sample it
            //Exclude gas or molten sites from the Potts neighbor tally
            dotValue += melt_misorientation(neighbor[i][j],c1,c2,c3);
            unique_dot[nevent] = dotValue;
            value = spin[neighbor[i][j]];
            unique[nevent] = value;
            unique_neigh[nevent] = neighbor[i][j];
            nevent++;
        }
    }
    
    //If no neighbor is eligible, return before changing anything. Will try next sweep.
    if (nevent == 0) return;
    
    //I think we should use nevent -1 (there will be an extra event at the end)
    double dran = (unique_dot[nevent - 1]*random->uniform());
    //if (iran >= nevent) iran = nevent-1;
    //Go through possible events and pick one
    for( int j = 0; j < nevent - 1; j++) {
        if(dran <= unique_dot[j]) {
            int neighran = unique_neigh[j];
            SiteState s1(unique[j],{q0[neighran],qx[neighran],qy[neighran],qz[neighran]});
            flip_site(i,s1);
            active_flag[i] = 3;
            solid_d[i] = -1;
            naccept++;
            return;
        }
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
        //Do 1st ones first (this isn't random yet)
        for(int j = 0; j < 6; j++) {

            if(active_flag[neighbor[i][nearest_neigh[j]]] == 2) {
                int i_chosen = neighbor[i][nearest_neigh[j]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
        //Do 2nd shell
        for(int j = 0; j < 12; j++) {
            if(active_flag[neighbor[i][second_nearest[j]]] == 2) {
                int i_chosen = neighbor[i][second_nearest[j]];
                flip_site(i_chosen, s_in);            
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }      
          }
        //Do 3rd shell    
        for(int j = 0; j < 8; j++) {
            if(active_flag[neighbor[i][third_nearest[j]]] ==2) {
                int i_chosen = neighbor[i][third_nearest[j]];
                spin[i_chosen] = spin[i];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                solid_d[i_chosen] = -nrefine -3;
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

//Calculate the local direction of the largest temperature gradient. Will need to update
//for boundary conditions
void AppAdditiveExtTempTexture::normal_finder(int site, double *outV)
{

	double xDel = 0;
	double yDel = 0;
	double zDel = 0;
	int lower_site = 0;
	
	
	//Do a special thing for the top of the melt pool
	//This is just cheating and copying the value from the next lower layer, but it should be good enough.
	if(active_flag[neighbor[site][13]] < 1) {
		xDel = fabs(T[neighbor[site][4]] - T[neighbor[site][21]]);
		yDel = fabs(T[neighbor[site][10]] - T[neighbor[site][15]]);
		
		lower_site = neighbor[site][12];
		zDel = fabs(T[neighbor[lower_site][12]] - T[neighbor[lower_site][13]]);
	}
	else {
		//Look at the 2 neighbors in + and - x directions
		//Need to look up what neighbors are right (assume unit-lattice)
		xDel = fabs(T[neighbor[site][4]] - T[neighbor[site][21]]);
		yDel = fabs(T[neighbor[site][10]] - T[neighbor[site][15]]);
		zDel = fabs(T[neighbor[site][12]] - T[neighbor[site][13]]);
	}

		
	//Now Normalize and return them
	double norm = sqrt(xDel*xDel + yDel*yDel + zDel*zDel);
	outV[0] = xDel/norm;
	outV[1] = yDel/norm;
	outV[2] = zDel/norm;
}

/* ----------------------------------------------------------------------
   Figure out the dot product between the temperature gradient and the x'tal orientation.
   We'll use this to change the Q-value at the lattice site...

   Inputs: lattice site :: site
   Outputs: double of max dot product
------------------------------------------------------------------------- */
double AppAdditiveExtTempTexture::melt_misorientation(int site, double c1, double c2, double c3)
{
	double gradNorm[3] = {0,0,0};
	double dotMax;
	double theta;
	double mobOut;

	
	//Calculate the unit-vector surface norm (use voxel counting)
	normal_finder(site, gradNorm);
	
	// Create quaternion vector from site's orientation
	vector<double> q_site = {q0[site], qx[site], qy[site], qz[site]};
	
	// Create gradient normal vector
	vector<double> grad_vector = {gradNorm[0], gradNorm[1], gradNorm[2]};
	
	// Use quaternion function to get cosine of minimum angle between
	// crystal orientation and temperature gradient
	theta = quaternion::get_cosine_of_minumum_angle_between_q_and_u(q_site, grad_vector);
	
	// Calculate angle from cosine
	//theta = acos(dotMax);
	
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
    
    //Randomly assign a temperature to every spin
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
}

void AppAdditiveExtTempTexture::load_next_chunk(){
  
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

