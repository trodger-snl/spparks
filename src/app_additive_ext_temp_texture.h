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

#ifdef APP_CLASS
AppStyle(additive_temperature_texture,AppAdditiveExtTempTexture)

#else

#ifndef SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_H
#define SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_H

#include "app_potts_quaternion.h"
#include "temperatureQueues.h"
#include "hdf5.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>
#include <cmath>

namespace SPPARKS_NS {

class AppAdditiveExtTempTexture : public AppPottsQuaternion {
 public:
  AppAdditiveExtTempTexture(class SPPARKS *, int, char **);
  virtual void grow_app();
  virtual void init_app();
  virtual void site_event_rejection(int, class RandomPark *);
  virtual void app_update(double);
  virtual void nucleation_spins(class RandomPark *);
  virtual void  input_app(char *, int , char **);
  virtual void nucleation_init();
  virtual void nucleation_particle_flipper(int, int,class RandomPark *);
  virtual void mushy_phase(int, class RandomPark *);
  void normal_finder(int, double *);
	double melt_misorientation(int, double, double, double);
  void quat2euler(int, double *);
  // void euler_init(); // NO LONGER NEEDED WITH QUATERNIONS
  void flip_site(int site, const SiteState &s);
  virtual void reduced_temperature_hdf();
  virtual void reduced_temperature_hdf_chunked();
  void load_data_counts_array();
  int xyz_to_local( double x, double y, double z );
  virtual void temperature_time_interpolate(int, double);
	std::vector<std::vector<std::vector<int>>> convert_to_3d_array_with_range(std::vector<int>&,int,int,int,int,int,int);

 protected:

	double *mobility_out;
	double *solid_d;
		double *neigh_dist;
    int time_index;
    double dtFD;
 	//To help improve ease of visualization, lets introduce another integer array. It will be 0 every to begin with
 	//When its layer becomes "active" we'll switch it to 1. This will help visualization and image dumping.
 	int *active_flag;
 	double read_interval;
 	const double R = 8.314459848; //Define a constant gas constant
	/// parameters for the thermal diffusion eq
	double *T;
	double time_step;
	int total_time; //Keep track of time for app_update
   int line_count;
  	double *temp_in_array;
  	double *phase_in_array;
	int x_period, y_period, z_period;
	int nrefine;
	//New inputs
	double dt;
	double no;
	double tc;
	double tsig;
	double dx;
    double mo; //Arrhenius pre-factor
 	double q; //Arrhenius exponential factor
 	double tl; //Liquidus point
 	double ts; //Solidus point
  double t_room;
  double *unique_dot;
  double *q0, *qx, *qy, *qz;
  int *unique_neigh;
    
    
  //New stuff
  double size_norm;
	double size_sig;
	double *solid_front_coeffs;
	int    solid_front_length;
	//Texture parameters
	double c1, c2, c3;
	int nlocal_app; //size of cnew, not sure if needed
 	int fully_periodic;
 	
 	//New MC model parameters
	int *nucleation_flags;
	double *nucleation_temps;
	double *nucleation_sizes;
  std::string temp_file_string;

  //Variables for reduced temperature files
  DoubleQueueContainer temp_in;
  DoubleQueueContainer time_in;
  double prior_time;
  int t_active;

  //Variables for chunked HDF5 reading
  hid_t hdf5_file_id;
  hid_t hdf5_temp_dataset;
  hid_t hdf5_time_dataset;
  hid_t hdf5_count_dataset;
  double chunk_time_window;
  double current_chunk_start_time;
  double current_chunk_end_time;
  bool hdf5_file_open;
  std::vector<std::vector<std::vector<int>>> data_counts_array;
  
  // Cache time ranges to avoid re-reading time data
  std::vector<std::vector<std::vector<std::vector<double>>>> cached_time_values;
  std::vector<std::vector<std::vector<bool>>> time_cache_loaded;
  
  // HDF5 bounds checking variables
  int bounds_check_mode; // 0 = exact match, 1 = subvolume
  hsize_t hdf5_dims[3];  // HDF5 data dimensions from data_counts
  double hdf5_origin[3]; // HDF5 data origin from x0 field
  bool bounds_validated;
  
  void load_next_chunk();
  void close_hdf5_file();
  bool needs_new_chunk(double simulation_time);
  void initialize_time_cache();
  void fill_missing_temperature_data();
  void validate_simulation_bounds();
  void get_hdf5_dimensions();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPPARKS to see the offending
line.

E: One or more sites have invalid values

The application only allows sites to be initialized with specific
values.

*/
