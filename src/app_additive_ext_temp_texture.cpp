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
    c) No - Max nucleation site density (in m^-3)
    d) k1 - kinetic parameter (in m^2/s)
    e) Q - activation energy (in J/mol)
    f) Tc - critical undercooling (K)
    g) Tsig - standard deviation of undercooling Gaussian
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


using namespace SPPARKS_NS;
using namespace MathConst;


/* ---------------------------------------------------------------------- */

AppAdditiveExtTempTexture::AppAdditiveExtTempTexture(SPPARKS *spk, int narg, char **arg) :
  AppPotts(spk,narg,arg)
{

    if (narg != 8  )
    error->all(FLERR,"Illegal app_style command");

    nspins = atoi(arg[1]);
    temp_file_str = arg[2]; //The name of the temperature file
    Tl = atof(arg[3]); //The material liquidus point
    Ts = atof(arg[4]); //The materials solidus point
    dx = atof(arg[5]); //The source lattice spacing ( in m)
    dt = atof(arg[6]); //The source timestep (in seconds)
    nrefine = atoi(arg[7]); //How many refinement MC steps to perform after a site solidifies
    
    //I think we need all of these variables still!
    ndouble = 9;
    allow_app_update = 1;
    ninteger = 2;
    totalTime = 0;
    sites = unique = NULL;

    //Set default values    
    Tl = 1723;
    Ts = 1673;
    No = 1e15;
    Tc = 5;
    Tsig = 3;
    sizeNorm = pow(dx,3) * 2;
    sizeSig = pow(dx,3);
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
    T_room = 300;
    
    //add the double array
    recreate_arrays();  
}

/* ----------------------------------------------------------------------
   input script commands unique to this app
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"liquidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal liquidus command");
    Tl = atof(arg[0]);
    if (Tl <= 0) 
      error->all(FLERR,"Illegal liquidus temperature");
  }
  else if (strcmp(command,"solidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal solidus command");
    Ts = atof(arg[0]);
    if (Ts <= 0) 
      error->all(FLERR,"Illegal solidus temperature");
  }
  else if (strcmp(command,"nucleation_density") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nucleation density command");
    No = atof(arg[0]);
    if (No < 0) 
      error->all(FLERR,"Illegal nucleation density");
  }
  else if (strcmp(command,"critical_undercooling") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal critical_undercooling command");
    Tc = atof(arg[0]);
    if (Tc < 0) 
      error->all(FLERR,"Illegal critical undercooling");
  }
  else if (strcmp(command,"undercooling_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal undercooling_deviation command");
    Tsig = atof(arg[0]);
    if (Tsig < 0) 
      error->all(FLERR,"Illegal undercooling standard deviation");
  }
  else if (strcmp(command,"mean_nuclei_volume") == 0) {
    if (narg !=1) error->all(FLERR,"Illegal mean_nuclei_volume command");
    sizeNorm = atof(arg[0]);
    if (sizeNorm < 0) 
      error->all(FLERR,"Illegal mean nuclei volume");
  }
  else if (strcmp(command,"nuclei_volume_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nuclei_volume_deviation command");
    sizeSig = atof(arg[0]);
    if (sizeSig < 0) 
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
  
  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
  When app_update is called, read in an input file
 ------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::app_update(double dt)
{

  //Reset t_active to assume we don't have a melt pool right now
  t_active = 0;
    //iterate through all the sets
    for (int i=0; i<nlocal; i++) {    
      
      
      //Update the temperature at all the sites
      temperature_time_interpolate(i,T[i]);
  
      //Turn the sites on/off depending on the phase data and whether or not the
      //site's temperature has gone above Tl
      //We will also have an activeFlag value of 3 for something that's re-solidified
      if( T[i] >= Tl) {
          activeFlag[i] = 2;
          spin[i] = (int) (nspins * ranapp->uniform());
          x1[i] = orientation_vectors[spin[i]*9];
          x2[i] = orientation_vectors[spin[i]*9+1];
          x3[i] = orientation_vectors[spin[i]*9+2];
          y1[i] = orientation_vectors[spin[i]*9+3];
          y2[i] = orientation_vectors[spin[i]*9+4];
          y3[i] = orientation_vectors[spin[i]*9+5];
          SolidD[i] = 0;
          t_active = 1;
      }
      //If we're molten, call the mushy_phase function to figure out any phase change
      else if (activeFlag[i] == 2 && T[i] <= Tl) {
          mushy_phase(i, ranapp);
          t_active = 1;
//             fprintf(screen,"Ran mushy_phase\n");
      } 
      else if(SolidD[i] < 0 && SolidD[i] > -nrefine -1 && activeFlag[i] == 3)    {
              MobilityOut[i] = 1;
              site_event_rejection(i, ranapp);
              SolidD[i]--;
              t_active = 1;
      }
    }

    // Use MPI_Allreduce to check if t_active is 0 on all processors
    int global_t_active;
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Check if all processors have t_active = 0. If so, fast forward our simulation time.
    if (global_t_active == 0) {
        double min_time = temp_in.findAndSyncSmallestFrontValue(MPI_COMM_WORLD);

        //If min time is past stop time, update to it. Otherwise, fast forward to min_time.
        //We might want to fast forward to min_time - dt instead...
        if(min_time > stoptime) time = stoptime;
        else time = min_time - dt;
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
  // os << temp_file_str;
  // const std::string tmp = os.str();
  // const char* cstr = tmp.c_str();

  std::cout << "File path: " << temp_file_str << std::endl;

  // Create a vector to hold the data counts
  //We should be able to break this up the same way for MPI as the temperature data.
  std::vector<int> data_counts(nlocal);

  // Open the HDF5 file in parallel mode
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
  hid_t file_id = H5Fopen(temp_file_str.c_str(), H5F_ACC_RDONLY, fapl_id);

  // Open the data_counts dataset
  hid_t dataset_counts_id = H5Dopen(file_id, "data_counts",H5P_DEFAULT);

  hid_t data_type = H5Dget_type(dataset_counts_id);

  //Figure out which 


  // Define the hyperslab for data_counts
  hsize_t start_counts[3] = { (hsize_t)domain->subxlo, (hsize_t)domain->subylo, (hsize_t)domain->subzlo};
  hsize_t count_counts[3] = { (hsize_t)(domain->subxhi - domain->subxlo), (hsize_t)(domain->subyhi - domain->subylo), (hsize_t)(domain->subzhi - domain->subzlo) };

  //std::cout << "Subdomain dimensions x " << domain->subxhi << " y " << domain->subyhi << " z " << domain->subzlo << std::endl;

  std::cout << "Hyperslab start: " << start_counts[0] << ", " << start_counts[1] << ", " << start_counts[2] << std::endl;
  std::cout << "Hyperslab count: " << count_counts[0] << ", " << count_counts[1] << ", " << count_counts[2]  << std::endl;

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

  //convert 1D data to 3d array
  auto data_counts_array = convertTo3DArrayWithRange(data_counts,(int)domain->subxlo,(int)domain->subxhi,(int)domain->subylo,(int)domain->subyhi,(int)domain->subzlo,(int)domain->subzhi);
  std::cout << "Read in the data_count values" << std::endl;

  // Open the temperature and time datasets
  hid_t dataset_temperature_id = H5Dopen(file_id, "temperature",H5P_DEFAULT);
  hid_t dataset_time_id = H5Dopen(file_id, "time",H5P_DEFAULT);

  hid_t data_type_temp = H5Dget_type(dataset_temperature_id);
  hid_t data_type_time = H5Dget_type(dataset_time_id);

    std::cout << "Began reading temperatures " << std::endl;


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

  std::cout << "Finished reading temperatures " << std::endl;

    // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Print the duration
  std::cout << "Function execution time: " << duration.count() << " ms" << std::endl;


}

/* ----------------------------------------------------------------------
	Read in full-field temperature data from an hdf5 file
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::temperature_hdf(int timestep)
{

    hid_t	file, tempdata;
    hid_t	datatype, dataspace, memspace;
    herr_t status;
    hsize_t offset[2];
    hsize_t offset2[1];
    hsize_t count[2];
    hsize_t count2[1];
    int nx = domain->boxxhi;
    int ny = domain->boxyhi;
    int nz = domain->boxzhi;
    line_count = nx * ny * nz;


    temp_in_array = new double[line_count];
    std::stringstream os;
    os << temp_file_str << ".hdf5";
    const std::string tmp = os.str();
    const char* cstr = tmp.c_str();
//     std::string in_file = temp_file_template + timestep + ".hdf5";

    file = H5Fopen(cstr, H5F_ACC_RDONLY, H5P_DEFAULT);

    tempdata = H5Dopen2( file, "/timeDependentValues/temperature", H5P_DEFAULT);

    datatype = H5Dget_type(tempdata);
    dataspace = H5Dget_space(tempdata);


    //Need to define the incides of the current "hyperslab"
    //Always select an entire column of data
    offset[0] = 0;
    offset[1] = timestep;
    count[0] = line_count;
    count[1] = 1;

    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,offset, NULL, count, NULL);

    //We also need to define the dimensions of the "hyperslab" in memory, which should match
    //the on-file dimension.
    offset2[0] = 0;
    count2[0] = line_count;
    memspace = H5Screate_simple(1, count, NULL);
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset2, NULL, count2,NULL);

    //Now that we have everything configured, we can actually read the data from the file
    status = H5Dread(tempdata, datatype, memspace, dataspace, H5P_DEFAULT, temp_in_array);

    //Close everything back up
    H5Tclose(datatype);
    H5Dclose(tempdata);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
}

/* ----------------------------------------------------------------------
  Does linear interpolation between two known temperatures and times to calculate current value at timestep.
  Also has checks for empty temperature vectors and timesteps before time
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::temperature_time_interpolate(int site, double priorTemp) {

  int x_loc = xyz[site][0] - (int)domain->subxlo;
  int y_loc = xyz[site][1] - (int)domain->subylo;
  int z_loc = xyz[site][2] - (int)domain->subzlo;

  //std::cout << " x_loc " << x_loc << " y_loc " <<y_loc << " z_loc " << z_loc <<  std::endl;

  //If we're out of entires, also set to room temperature.
  if (time_in(x_loc,y_loc,z_loc).empty()){ 
    T[site] = T_room;
    return;
  }
  //If we haven't encountered our first time value, set to default
  else if(priorTime == 0 && time < time_in(x_loc,y_loc,z_loc).front()) {
    T[site] = T_room;
    return;
  }

//  std::cout << " x_loc " << x_loc << " y_loc " <<y_loc << " z_loc " << z_loc <<  " time front " << time_in(x_loc,y_loc,z_loc).front() << std::endl;
  //If we've stepped past the current time, update the stored values
  if(time >= time_in(x_loc,y_loc,z_loc).front()) {
    priorTime = time_in(x_loc,y_loc,z_loc).front();
    priorTemp = temp_in(x_loc,y_loc,z_loc).front();

    //Pop off old values
    time_in(x_loc,y_loc,z_loc).pop();
    temp_in(x_loc,y_loc,z_loc).pop();
  }

  //If we're inbetween melt cycles, set temp to room temp
  if(priorTemp < Ts && temp_in(x_loc,y_loc,z_loc).front() < Ts) {
    T[site] = T_room;
  }

  //Do linear interpolation
  else {
    T[site] = priorTemp + (temp_in(x_loc,y_loc,z_loc).front() - priorTemp)/(time_in(x_loc,y_loc,z_loc).front() - priorTime) * (time - priorTime);
  }
}

/* ----------------------------------------------------------------------
   Nucleation site initializer. Find the volume of a voxel from dx^3 and Multiply by No.
   This will be the average number of nucleation sites in the voxel. This is also the
   fraction of spins that we want to be able to nucleate new grains. If the value is greater
   than 1 (which we should avoid), allow all spins to nucleate. If not, call a random number
   between zero and one. If the number is less than the fraction, make true. If not, make false.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_spins(RandomPark *random) {
    nucleationFlags = new int[nspins];
    double nucleationFraction = dx * dx * dx * No;
    
    //Make all spins nucleation sites. Should avoid this. Maybe return an error instead?
    if(nucleationFraction >= 1.0) {
        fprintf(screen,"Nucleation fraction is greater than 1. Decrease No or increase mesh resolution. No = %f\n", nucleationFraction);
        for (int i = 0; i < nspins; i++) {
            nucleationFlags[i] = 1;
        }
    }
    //Do a random number test and allow the spin to nucleate if less than
    else {
        for (int i = 0; i < nspins; i++) {
            if(random->uniform() <= nucleationFraction) {
                nucleationFlags[i] = 1;
            }
            else {
                nucleationFlags[i] = 0;
            }
        }
    }   
}


/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::grow_app()
{
  spin = iarray[0];
  activeFlag = iarray[1];
  MobilityOut = darray[0];
  T = darray[1];
  SolidD = darray[2];
  //phi1 = darray[3];
  //Phi = darray[4];
  //phi2 = darray[5];
  x1 = darray[3];
  x2 = darray[4];
  x3 = darray[5];
  y1 = darray[6];
  y2 = darray[7];
  y3 = darray[8];

      if (nlocal_app < nlocal) {
        nlocal_app = nlocal;
                
      }
  //Determine whether the domain is fully periodic, if not. Specify the periodicity of each boundary
  if(domain->nonperiodic == 0) {
  		fully_periodic = 1;
  }
  else {
  	xPeriod = 1- domain->xperiodic;
  	yPeriod = 1- domain->yperiodic;
  	zPeriod = 1- domain->zperiodic;
  }
}


/* ----------------------------------------------------------------------
   initialize before each run
   check validity of site values
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::init_app()
{
  delete [] sites;
  delete [] unique;
  double sqrt2 = 1.4142135624;
  double sqrt3 = 1.7320508076;
  sites = new int[1 + maxneigh];
  unique = new int[1 + maxneigh];
  uniqueDot = new double[1 + maxneigh];
  RandomPark random(3000);

  //Allocate our temperature and time data structures
  temp_in.initialize((int)(domain->subxhi - domain->subxlo),(int)(domain->subyhi - domain->subylo),(int)(domain->subzhi - domain->subzlo));
  time_in.initialize((int)(domain->subxhi - domain->subxlo),(int)(domain->subyhi - domain->subylo),(int)(domain->subzhi - domain->subzlo));

  dt_sweep = dt;
  time_index = 0;
  priorTime = 0;

  orientation_vectors = new double[nspins * 9];
  spin_euler = new double[nspins * 3];
  nucleationFlags = new int[nspins];
  nucleationTemps = new double[nspins];
	nucleationSizes = new double[nspins];


  //Initialize orientations based on spins (which have hopefully been determined)...
  if (domain->me==0) {
  	orientation_init(ranapp);    
  }


  MPI_Bcast(orientation_vectors,nspins*9, MPI_DOUBLE,0,world);
  
  if(domain->me==0){
    euler_init();
  }
  
//   MPI_Bcast(spin_euler, nspins*3, MPI_DOUBLE,0,world);

  int flag = 0;
	int flagall;
    for (int i = 0; i < nlocal; i++) {
			if (spin[i] < 1 || spin[i] > nspins) {
				flag = 1;
			}
			//Initialize spin-based orientation
			x1[i] = orientation_vectors[spin[i]*9];
			x2[i] = orientation_vectors[spin[i]*9+1];
			x3[i] = orientation_vectors[spin[i]*9+2];
			y1[i] = orientation_vectors[spin[i]*9+3];
			y2[i] = orientation_vectors[spin[i]*9+4];
			y3[i] = orientation_vectors[spin[i]*9+5];
    }

  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
  
	//Initialize the nucleationFlags vector
	if (domain->me==0) {
		nucleation_spins(ranapp);    
	}

	MPI_Bcast(nucleationFlags,nspins, MPI_INT,0,world);

	//Initialize the nucleationTemps and nucleationSizes vectors
	if (domain->me==0) {
			nucleation_init();    
	}

	MPI_Bcast(nucleationTemps,nspins, MPI_DOUBLE,0,world);
	MPI_Bcast(nucleationSizes,nspins, MPI_DOUBLE,0,world);
	
	//Initialize the neighDist array need to fill with good values
	neighDist = new double[26];
	neighDist[0] = sqrt3 * dx;
	neighDist[1] = sqrt2 * dx;
	neighDist[2] = sqrt3 * dx;
	neighDist[3] = sqrt2 * dx;
	neighDist[4] = dx;
	neighDist[5] = sqrt2 * dx;
	neighDist[6] = sqrt3 * dx;
	neighDist[7] = sqrt2 * dx;
	neighDist[8] = sqrt3 * dx;
	neighDist[9] =  sqrt2 * dx;
	neighDist[10] = dx;
	neighDist[11] =  sqrt2 * dx;
	neighDist[12] = sqrt3 * dx;
	neighDist[13] = sqrt3 * dx;
	neighDist[14] =  sqrt2 * dx;
	neighDist[15] = dx;
	neighDist[16] =  sqrt2 * dx;
	neighDist[17] = sqrt3 * dx;
	neighDist[18] = sqrt2 * dx;
	neighDist[19] = sqrt3 * dx;
	neighDist[20] =  sqrt2 * dx;
	neighDist[21] = dx;
	neighDist[22] =  sqrt2 * dx;
	neighDist[23] = sqrt3 * dx;
	neighDist[24] = sqrt2 * dx;
	neighDist[25] = sqrt3 * dx;

  //Check that our timestep is small enough	
  double max_front_vel = 0;
	int power = solid_front_length -1;
	for(int k = 0; k < solid_front_length; k++) {
		max_front_vel = max_front_vel + solid_front_coeffs[k] * pow(Tl - Ts, power);
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
  reduced_temperature_hdf();

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
    Mobloc = MobilityOut[i];
    
    int j,m,value;
    int nevent = 0;
    
    if((Mobloc < 0.0) || (Mobloc > 1.0001)) {
        MobilityOut[i] = 0;
        return;
    }
    
    if(SolidD[i] < 0 && SolidD[i] > -nrefine -1) {
        //Go through neighbor list and add them to possible switches
        for (int j = 0; j < numneigh[i]; j++) {
            if(activeFlag[neighbor[i][j]] == 3) {
                //Calculate temp gradient/grain misorientation and store in "unique" array
                //We should make this cumulative, so we can use a random number to sample it
                //Exclude gas or molten sites from the Potts neighbor tally
                dotValue += melt_misorientation(neighbor[i][j],c1,c2,c3);
                uniqueDot[nevent] = dotValue;
                value = spin[neighbor[i][j]];
                unique[nevent] = value;
                nevent++;										
            }
        }
        //If no neighbor is eligible, return before changing anything. Will try next sweep.
        if (nevent == 0) return;
        //I think we should use nevent -1 (there will be an extra event at the end)
        double dran = (uniqueDot[nevent - 1]*random->uniform());
        //if (iran >= nevent) iran = nevent-1;
        //Go through possible events and pick one
        for( int j = 0; j < nevent -1; j++) {
            if(dran <= uniqueDot[j]) {
                spin[i] = unique[j];
                efinal = site_energy(i);
            }
        }
    }

  else {
      for (j = 0; j < numneigh[i]; j++) {
        value = spin[neighbor[i][j]];
        //Exclude gas, powder or molten sites from the Potts neighbor tally
        if (value == spin[i] || value == nspins || activeFlag[neighbor[i][j]] != 3) continue;
        for (m = 0; m < nevent; m++)
          if (value == unique[m]) break;
        if (m < nevent) continue;
        unique[nevent++] = value;
      }

      if (nevent == 0) return;
      int iran = (int) (nevent*random->uniform());
      if (iran >= nevent) iran = nevent-1;
      spin[i] = unique[iran];
      efinal = site_energy(i);
  }
  // accept or reject via Boltzmann criterion

  if (efinal <= einitial) {
     if (random->uniform() > Mobloc){
       spin[i] = oldstate;
     }
  }
  else if (temperature == 0.0) {
    spin[i] = oldstate;
  } 
  else if (random->uniform() > Mobloc * exp((einitial-efinal)*t_inverse)) {
    spin[i] = oldstate;
  }



  if (spin[i] != oldstate) {
 //    phi1[i] = spin_euler[spin[i] * 3 -2];
//     Phi[i] = spin_euler[spin[i] * 3 -1];
//     phi2[i] = spin_euler[spin[i] *3];
    x1[i] = orientation_vectors[spin[i]*9];
    x2[i] = orientation_vectors[spin[i]*9+1];
    x3[i] = orientation_vectors[spin[i]*9+2];
    y1[i] = orientation_vectors[spin[i]*9+3];
    y2[i] = orientation_vectors[spin[i]*9+4];
    y3[i] = orientation_vectors[spin[i]*9+5];
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
   1. Determine if the site is solid or liquid (from activeFlag)
   2. Determine if the site should nucleate a new grain (from nucleationFlags)
   3. If our current site is liquid & can't nucleate, have it try to switch to a solid neighbor
      (with 4 calculated from the undercooling in someway, not temperature)
   4. If our current site is liquid & can nucleate, check if local undercooling is equal to its 
      critical temp. If so, change the activeFlag value to solid. If not, see if there are any solid sites
      that should capture it.
   5. If our current site is solid, see if it should flip to a neighboring solid value (with
      mobility calculated from undercooling.)
THIS NEEDS UPDATED TO INCLUDE TEXTURE
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::mushy_phase(int i, RandomPark *random){
  	int nevent = 0;
  	int m,value;
  	double dotValue = 0;
    double Tcool = Tl - T[i];
    
//     fprintf(screen,"Made it to mushy_phase %d\n", i);
    //Our site should always be molten and below Tl
    //Check if it's eligible to nucleate
    if(nucleationFlags[spin[i]]) {
//       fprintf(screen,"Trying to nucleate %d\n",i);
        //Can and will nucleate
        if(Tcool >= nucleationTemps[spin[i]]){
            activeFlag[i] = 3;
            //Don't let nucleated site disappear during smoothing
            //Neighboring sites will be flipped during the next iterate_rejection call
            SolidD[i] = -nrefine-2;
            //Call nucleation particle flipper
//             fprintf(screen,"Nucleating\n");
            naccept++;
            nucleation_particle_flipper(i, round(nucleationSizes[spin[i]]/pow(dx,3)), ranapp);
            return;
        }
        //Can nucleate, but won't yet. Allow to if the solidification front gets captured.
        else {
            //Add the distance of the front travel. This is for 304L. Need to multiply by timestep to get distance
            //Try doing this with an arbitrary array
            int power = solid_front_length -1;
            for(int k = 0; k < solid_front_length; k++) {
                SolidD[i] = SolidD[i] + solid_front_coeffs[k] * pow(Tcool, power) * time_step;
                power--;
            }
            //Go through neighbor list and add them to possible switches
            for (int j = 0; j < numneigh[i]; j++) {
              if(neighDist[j] <= SolidD[i] && (activeFlag[neighbor[i][j]] == 1 || activeFlag[neighbor[i][j]] == 3)) {
                    //Calculate temp gradient/grain misorientation and store in "unique" array
                    //We should make this cumulative, so we can use a random number to sample it
                    //Exclude gas or molten sites from the Potts neighbor tally
//                     if(neighDist[j] > SolidD[i]) continue;
                    dotValue += melt_misorientation(neighbor[i][j],c1,c2,c3);
                    uniqueDot[nevent] =  dotValue;
                    value = spin[neighbor[i][j]];
                    unique[nevent] = value;
                    nevent++;										
                }
            }
            //If no neighbor is eligible, return before changing anything. Will try next sweep.
            if (nevent == 0) return;
            fprintf(screen,"Epitaxialy growing instead of nucleating\n");
            //I think we should use nevent -1 (there will be an extra event at the end)
            double dran = (uniqueDot[nevent - 1]*random->uniform());
            //if (iran >= nevent) iran = nevent-1;
            //Go through possible events and pick one
            for( int j = 0; j < nevent -1; j++) {
                if(dran <= uniqueDot[j]) {
                    spin[i] = unique[j];
 //                    phi1[i] = spin_euler[spin[i] * 3 -2];
//                     Phi[i] = spin_euler[spin[i] * 3 -1];
//                     phi2[i] = spin_euler[spin[i] *3];
                    x1[i] = orientation_vectors[spin[i]*9];
                    x2[i] = orientation_vectors[spin[i]*9+1];
                    x3[i] = orientation_vectors[spin[i]*9+2];
                    y1[i] = orientation_vectors[spin[i]*9+3];
                    y2[i] = orientation_vectors[spin[i]*9+4];
                    y3[i] = orientation_vectors[spin[i]*9+5];
                    activeFlag[i] = 3;
                    SolidD[i] = -1;
                    naccept++;
                    return;
                }
            }
        }
    }
    else {
//       fprintf(screen,"Trying to directionally solidify %d\n",i);
        //Add the distance of the front travel. This is for 304L I think...
        int power = solid_front_length -1;
        for(int k = 0; k < solid_front_length; k++) {
            SolidD[i] = SolidD[i] + solid_front_coeffs[k] * pow(Tcool, power) * time_step;
            power--;
        }
        //Go through neighbor list and add them to possible switches
        for (int j = 0; j < numneigh[i]; j++) {
//             fprintf(screen,"calculating neighbor distance, nieghDist %f, solidD %e\n",neighDist[j], SolidD[i]);
            if(neighDist[j] <= SolidD[i] && (activeFlag[neighbor[i][j]] == 1 || activeFlag[neighbor[i][j]] == 3)) {
                //Calculate temp gradient/grain misorientation and store in "unique" array
                //We should make this cumulative, so we can use a random number to sample it
                //Exclude gas or molten sites from the Potts neighbor tally
//                 if(neighDist[j] > SolidD[i]) continue;
                dotValue += melt_misorientation(neighbor[i][j],c1,c2,c3);
                uniqueDot[nevent] =  dotValue;
                value = spin[neighbor[i][j]];
                unique[nevent] = value;
                nevent++;										
//                 fprintf(screen,"Adding events to list, %d\n", nevent);
            }
        }
        //If no neighbor is eligible, return before changing anything. Will try next sweep.
        if (nevent == 0) return;
        
//         fprintf(screen,"Epitaxialy growing\n");
        //I think we should use nevent -1 (there will be an extra event at the end)
        double dran = (uniqueDot[nevent - 1]*random->uniform());
        //if (iran >= nevent) iran = nevent-1;
        //Go through possible events and pick one
        for( int j = 0; j < nevent -1; j++) {
            if(dran <= uniqueDot[j]) {
                spin[i] = unique[j];
 //                phi1[i] = spin_euler[spin[i] * 3 -2];
//                 Phi[i] = spin_euler[spin[i] * 3 -1];
//                 phi2[i] = spin_euler[spin[i] *3];
                x1[i] = orientation_vectors[spin[i]*9];
                x2[i] = orientation_vectors[spin[i]*9+1];
                x3[i] = orientation_vectors[spin[i]*9+2];
                y1[i] = orientation_vectors[spin[i]*9+3];
                y2[i] = orientation_vectors[spin[i]*9+4];
                y3[i] = orientation_vectors[spin[i]*9+5];
                activeFlag[i] = 3;
                SolidD[i] = -1;
                naccept++;
                return;
            }
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
    

    

    //Its hard to go through shells iteravely if the number of neighbors isn't the full 26.
    //Check if this is the case and just loop through the neihgbor list if so
    if(numneigh[i] != 26) {

        
        for(int j = 0; j < numneigh[i]; j++) {
            if(activeFlag[neighbor[i][j]] == 2) {
                int i_chosen = neighbor[i][j];
                spin[i_chosen] = spin[i];
//                 phi1[i_chosen] = spin_euler[spin[i] * 3 -2];
//                 Phi[i_chosen] = spin_euler[spin[i] * 3 -1];
//                 phi2[i_chosen] = spin_euler[spin[i] *3];
                x1[i_chosen] = orientation_vectors[spin[i]*9];
                x2[i_chosen] = orientation_vectors[spin[i]*9+1];
                x3[i_chosen] = orientation_vectors[spin[i]*9+2];
                y1[i_chosen] = orientation_vectors[spin[i]*9+3];
                y2[i_chosen] = orientation_vectors[spin[i]*9+4];
                y3[i_chosen] = orientation_vectors[spin[i]*9+5];
                activeFlag[i_chosen] = 3;
                SolidD[i_chosen] = -nrefine -3;
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

            if(activeFlag[neighbor[i][nearest_neigh[j]]] == 2) {
                int i_chosen = neighbor[i][nearest_neigh[j]];
                spin[i_chosen] = spin[i];
//                 phi1[i_chosen] = spin_euler[spin[i] * 3 -2];
//                 Phi[i_chosen] = spin_euler[spin[i] * 3 -1];
//                 phi2[i_chosen] = spin_euler[spin[i] *3];     
                x1[i_chosen] = orientation_vectors[spin[i]*9];
                x2[i_chosen] = orientation_vectors[spin[i]*9+1];
                x3[i_chosen] = orientation_vectors[spin[i]*9+2];
                y1[i_chosen] = orientation_vectors[spin[i]*9+3];
                y2[i_chosen] = orientation_vectors[spin[i]*9+4];
                y3[i_chosen] = orientation_vectors[spin[i]*9+5];
                activeFlag[i_chosen] = 3;
                SolidD[i_chosen] = -nrefine -3;
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
        //Do 2nd shell
        for(int j = 0; j < 12; j++) {
            if(activeFlag[neighbor[i][second_nearest[j]]] == 2) {
                int i_chosen = neighbor[i][second_nearest[j]];
                spin[i_chosen] = spin[i];
//                 phi1[i_chosen] = spin_euler[spin[i] * 3 -2];
//                 Phi[i_chosen] = spin_euler[spin[i] * 3 -1];
//                 phi2[i_chosen] = spin_euler[spin[i] *3];   
                x1[i_chosen] = orientation_vectors[spin[i]*9];
                x2[i_chosen] = orientation_vectors[spin[i]*9+1];
                x3[i_chosen] = orientation_vectors[spin[i]*9+2];
                y1[i_chosen] = orientation_vectors[spin[i]*9+3];
                y2[i_chosen] = orientation_vectors[spin[i]*9+4];
                y3[i_chosen] = orientation_vectors[spin[i]*9+5];             
                activeFlag[i_chosen] = 3;
                SolidD[i_chosen] = -nrefine -3;
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }      
          }
        //Do 3rd shell    
        for(int j = 0; j < 8; j++) {
            if(activeFlag[neighbor[i][third_nearest[j]]] ==2) {
                int i_chosen = neighbor[i][third_nearest[j]];
                spin[i_chosen] = spin[i];
//                 phi1[i_chosen] = spin_euler[spin[i] * 3 -2];
//                 Phi[i_chosen] = spin_euler[spin[i] * 3 -1];
//                 phi2[i_chosen] = spin_euler[spin[i] *3];   
                x1[i_chosen] = orientation_vectors[spin[i]*9];
                x2[i_chosen] = orientation_vectors[spin[i]*9+1];
                x3[i_chosen] = orientation_vectors[spin[i]*9+2];
                y1[i_chosen] = orientation_vectors[spin[i]*9+3];
                y2[i_chosen] = orientation_vectors[spin[i]*9+4];
                y3[i_chosen] = orientation_vectors[spin[i]*9+5];
                activeFlag[i_chosen] = 3;
                SolidD[i_chosen] = -nrefine -3;
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
        if(spin[neighbor[i][j]] == spin[i] && activeFlag[neighbor[i][j]] == 3) {
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
	if(activeFlag[neighbor[site][13]] < 1) {
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
	
	//Define a rotation matrix to fill in
	double gradNorm[3] = {0,0,0};
	double dot1;
	double dot2;
	double dot3;
	double dotMax;
	double siteVec1[3] = {0,0,0};
	double siteVec2[3] = {0,0,0};
	double siteVec3[3] = {0,0,0};
	double theta;
	double mobOut;
	
	//Get the 1st x'tal vector from memory
	siteVec1[0] = orientation_vectors[spin[site]*9];
	siteVec1[1] = orientation_vectors[spin[site]*9 + 1];
	siteVec1[2] = orientation_vectors[spin[site]*9 + 2];
	
	siteVec2[0] = orientation_vectors[spin[site]*9 + 3];
	siteVec2[1] = orientation_vectors[spin[site]*9 + 4];
	siteVec2[2] = orientation_vectors[spin[site]*9 + 5];
	
	siteVec3[0] = orientation_vectors[spin[site]*9 + 6];
	siteVec3[1] = orientation_vectors[spin[site]*9 + 7];
	siteVec3[2] = orientation_vectors[spin[site]*9 + 8];
	

	//Calculate the unit-vector surface norm (use voxel counting)
	normal_finder(site, gradNorm);

	//Now lets take the dot product of the vectors with the surface normal
	//I want to try out my voxel counting scheme, as I think it'll work!
	dot1 = fabs(siteVec1[0] * gradNorm[0] + siteVec1[1] * gradNorm[1] + siteVec1[2] * gradNorm[2]);
	dot2 = fabs(siteVec2[0] * gradNorm[0] + siteVec2[1] * gradNorm[1] + siteVec2[2] * gradNorm[2]);
	dot3 = fabs(siteVec3[0] * gradNorm[0] + siteVec3[1] * gradNorm[1] + siteVec3[2] * gradNorm[2]);
	
	//Figure out which dot is the largest magnitude and return it.
	if(dot1 >= dot2) {
		if(dot1 >= dot3) dotMax = dot1;
		else dotMax = dot3;
	}
	else {
		if(dot2 >= dot3) dotMax = dot2;
		else dotMax = dot3;
	}
	theta = acos(dotMax);
	
	mobOut = c1 + c2 * cos(c3 * theta);
	
	return mobOut;
}

/* ----------------------------------------------------------------------
  Create the lookup table to link xtal orientation and spin. For starters, we'll align
  all the spins in the same direction and that will be aligned with the xyz axes of the domain.
  Each spin will have 9 double-values associated with it in the table...
  
  I think we should only do this once. Currently, we do this on all processors and each one
  has its own random number scheme. This results in inconsistent results across proc boundaries
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::orientation_init(RandomPark *random)
{

	double x1o;
	double x2o;
	double x_unnorm;
	double y_unnorm;
	double z_unnorm;
	double norm1;
	double dot_val;
	double dot_val2;
	double dot_val3;
	int flag;
	
	//We can store these values as a n*9 1D array and use fancy indexing to access it.
	//Initialize all vectors to x,y,z orthogonal directions.
	//This is a stupid way to do it, but it should work.
	//This might be off by 1 iteration...
	for (int i = 0; i <= nspins * 9 + 1; i = i + 9) {
	
	//Let's write the algorithm to do it for random orientations while I'm at it
	//I didn't do it correctly the first time...
	//This new method by Marsaglia will return already normalized vectors!
		flag = 0;
		while (flag == 0) {
			x1o = random->uniform()*2.0 - 1.0;
			x2o = random->uniform() * 2.0 - 1.0;
			
			if (x1o*x1o + x2o*x2o < 1.0) {
				flag = 1;
			}
		}
	
		x_unnorm = 2 * x1o * sqrt(1 - x1o*x1o - x2o*x2o);
		y_unnorm = 2 * x2o * sqrt(1 - x1o*x1o - x2o*x2o);
		z_unnorm = 1 - 2 * (x1o*x1o + x2o*x2o);
		
		//Now let's normalize the first one and put it in memory
		orientation_vectors[i] = x_unnorm;
		orientation_vectors[i + 1] = y_unnorm;
		orientation_vectors[i + 2] = z_unnorm;
	
		//Ok, now let's randomly find an orthogonal vector and it's compliment
		//Generate a new random vector
		flag = 0;
		while (flag == 0) {
			x1o = random->uniform()*2.0 - 1.0;
			x2o = random->uniform() * 2.0 - 1.0;
			
			if (x1o*x1o + x2o*x2o < 1.0) {
				flag = 1;
			}
		}
	
		x_unnorm = 2 * x1o * sqrt(1 - x1o*x1o - x2o*x2o);
		y_unnorm = 2 * x2o * sqrt(1 - x1o*x1o - x2o*x2o);
		z_unnorm = 1 - 2 * (x1o*x1o + x2o*x2o);
	
		dot_val = x_unnorm * orientation_vectors[i] + y_unnorm * orientation_vectors[i + 1] + z_unnorm * orientation_vectors[i + 2]; 
	
		//Now let's make it orthogonal to our first one
		x_unnorm = x_unnorm - (dot_val*orientation_vectors[i]);
		y_unnorm = y_unnorm - (dot_val*orientation_vectors[i+1]);
		z_unnorm = z_unnorm - (dot_val*orientation_vectors[i + 2]);
	
		//Normalize the new vector and save it
		norm1 = sqrt(x_unnorm * x_unnorm + y_unnorm * y_unnorm + z_unnorm * z_unnorm);
		orientation_vectors[i + 3] = x_unnorm/norm1;
		orientation_vectors[i + 4] = y_unnorm/norm1;
		orientation_vectors[i + 5] = z_unnorm/norm1;
		
		dot_val = orientation_vectors[i+3] * orientation_vectors[i] + orientation_vectors[i+4] * orientation_vectors[i + 1] + orientation_vectors[i+5] * orientation_vectors[i + 2]; 
	
		//Now lets make the third one, have to do a cross-product
		//If we did the first two right, this should be normalized
		//They're closed to normalized but not quite... We should go ahead and redo it
		x_unnorm = orientation_vectors[i + 1] * orientation_vectors[i + 5] - orientation_vectors[i+2] * orientation_vectors[i + 4];
		y_unnorm = orientation_vectors[i + 2] * orientation_vectors[i + 3] - orientation_vectors[i] * orientation_vectors[i + 5];
		z_unnorm = orientation_vectors[i] * orientation_vectors[i + 4] - orientation_vectors[i+1] * orientation_vectors[i + 3];	
		norm1 = sqrt(x_unnorm * x_unnorm + y_unnorm * y_unnorm + z_unnorm * z_unnorm);
		
		orientation_vectors[i + 6] = x_unnorm/norm1;
		orientation_vectors[i + 7] = y_unnorm/norm1;
		orientation_vectors[i + 8] = z_unnorm/norm1;
		
		dot_val2 = orientation_vectors[i+3] * orientation_vectors[i+6] + orientation_vectors[i+4] * orientation_vectors[i + 7] + orientation_vectors[i+5] * orientation_vectors[i + 8]; 
		dot_val3 = orientation_vectors[i+6] * orientation_vectors[i] + orientation_vectors[i+7] * orientation_vectors[i + 1] + orientation_vectors[i+8] * orientation_vectors[i + 2]; 
		
	}
}

//Let's build a global array of euler angles (much like the vectors), so we dont have to
//Do trig calculations over and over again.
//Call at the beginning of the program
//This function won't do any loops, just the pure math
//Will need verification that I didn't get indices screwed up...
void AppAdditiveExtTempTexture::vec2euler(int spin_loc, double *eulers) {

  //Find the rotation matrix, I think it should be of the form:
  //{{x1,x2o,x3o},{y1,y2o,y3o},{z1o,z2,z3}} where each vector is normalized
  //We should be able to build this directly from the orientation_vectors... vector

    double x1o = orientation_vectors[spin_loc*9];
    double y1o = orientation_vectors[spin_loc*9 + 1];
    double z1 = orientation_vectors[spin_loc*9 + 2];

    double x2o = orientation_vectors[spin_loc*9 + 3];
    double y2o = orientation_vectors[spin_loc*9 + 4];
    double z2 = orientation_vectors[spin_loc*9 + 5];

    double x3o = orientation_vectors[spin_loc*9 + 6];
    double y3o = orientation_vectors[spin_loc*9 + 7];
    double z3 = orientation_vectors[spin_loc*9 + 8];
    

  
  //Make sure z3 isn't one
  if (abs(z3) < 1.000001 && abs(z3) > 0.999999) {
    eulers[0] = atan2(x2o,x1o);
    eulers[1] = MY_PI/2.0 * (1 - z3);
    eulers[2] = 0;
    return;
  }
  //This will be more typical.
  else {
    double ksi = 1/sqrt(1 - pow(z3,2));
    
    eulers[0] = atan2(z1 * ksi, -z2 * ksi);
    eulers[1] = acos(z3);
    eulers[2] = atan2(x3o * ksi, y3o * ksi);
    return;
  }
}

//Sets up the euler angle per spin mapping
void AppAdditiveExtTempTexture::euler_init() {

    double eulerLoc[3] = {0, 0 ,0};
    int j = 1;
    //Go through all spins and determine euler angles
    for (int i = 0; i < nspins * 3; i = i + 3) {
        vec2euler(j, eulerLoc);
        spin_euler[i] = eulerLoc[0];
        spin_euler[i +1]  = eulerLoc[1];
        spin_euler[i + 2] = eulerLoc[2];
        j++;
    }
}

/* ----------------------------------------------------------------------
    The first version of this just initialized critical nucleation temperatures.
    This version will also initialize nucleii size (starting with a normal dist)
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_init() {
    std::normal_distribution<> dist_T{Tc,Tsig};
    std::normal_distribution<> dist_S{sizeNorm,sizeSig};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    //Randomly assign a temperature to every spin
    for(int i = 0; i < nspins; i++) {
        nucleationTemps[i] = dist_T(gen);
        nucleationSizes[i] = dist_S(gen);
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
std::vector<std::vector<std::vector<int>>> AppAdditiveExtTempTexture::convertTo3DArrayWithRange(std::vector<int>& inputVector, 
    int xStart, int xEnd, 
    int yStart, int yEnd, 
    int zStart, int zEnd) {
    // Calculate the sizes of each dimension
    int xSize = xEnd - xStart;
    int ySize = yEnd - yStart;
    int zSize = zEnd - zStart;

    // Ensure the input vector has the correct size

    std::cout << "sizes " << inputVector.size() << " and " << xSize * ySize * zSize << std::endl;
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