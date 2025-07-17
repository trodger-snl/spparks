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
#include "temperature_source.h"
#include "hdf5.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <memory>

namespace SPPARKS_NS {

class AppAdditiveExtTempTexture : public AppPottsQuaternion {
 public:
  AppAdditiveExtTempTexture(class SPPARKS *, int, char **);
  virtual ~AppAdditiveExtTempTexture();
  virtual void grow_app();
  virtual void init_app();
  virtual void site_event_rejection(int, class RandomPark *);
  virtual void app_update(double);
  virtual void nucleation_spins(class RandomPark *);
  virtual void  input_app(char *, int , char **);
  virtual void nucleation_init();
  virtual void nucleation_particle_flipper(int, int,class RandomPark *);
  virtual void mushy_phase(int, class RandomPark *);
  virtual double site_energy(int);
  void site_event_solidification(int, double, class RandomPark *);
  std::vector<double> normal_finder(int);
	double melt_misorientation(int, double, double, double);
  void apply_misorientation(int, double, class RandomPark *);
  std::vector<double> get_average_neighbor_quaternion(int site, int target_spin);
  void smooth_site(int site);
  double site_energy_smooth(int site);
  
  // Modular temperature source methods
  void setup_temperature_source_cmd(int narg, char **arg);
  void setup_scan_path_cmd(int narg, char **arg);
  void update_temperature_from_source(double simulation_time);
  
  // Legacy HDF5 temperature methods
  virtual void reduced_temperature_hdf();
  virtual void reduced_temperature_hdf_chunked();
  void load_data_counts_array();
  int xyz_to_local( double x, double y, double z );
  virtual void temperature_time_interpolate(int, double);
	std::vector<std::vector<std::vector<int>>> convert_to_3d_array_with_range(std::vector<int>&,int,int,int,int,int,int);

 protected:

	double *mobility_out;
	double *solid_d;
	double *melt_misorientation_out;
		double *neigh_dist;
    int time_index;
    double dtFD;
 	// Active flag for site visualization and state tracking
 	// 0: inactive, 1: active layer, 2: molten, 3: solidified
 	int *active_flag;
 	double read_interval;
 	const double R = 8.314459848; // Gas constant (J/mol/K)
	// Temperature array for thermal field
	double *T;
  double *G; //Temperature gradient at solidification
  double *V; //Solidification rate at solidification.
	double time_step;
	int total_time; // Time tracking for app_update
   int line_count;
  	double *temp_in_array;
  	double *phase_in_array;
	int x_period, y_period, z_period;
	int nrefine;
	// Physical and simulation parameters
	double dt;
	double no;
	double tc;
	double tsig;
	double dx;
    double mo; // Arrhenius pre-factor
 	double q; // Arrhenius exponential factor
 	double tl; // Liquidus temperature
 	double ts; // Solidus temperature
  double t_room;
  double *unique_dot;
    
    
  // Nucleation and growth parameters
  double size_norm;
	double size_sig;
	double *solid_front_coeffs;
	int    solid_front_length;
	// Texture model parameters
	double c1, c2, c3;
	int nlocal_app; // Local application size
 	int fully_periodic;
  double max_misorient;
  double t_cool_max;
  double mis_thresh;
  double misorient_alpha;
 	
 	// Debug parameters
 	int normal_finder_debug; // Flag to enable/disable normal_finder debugging
 	
 	// Monte Carlo nucleation model parameters
	int *nucleation_flags;
	double *nucleation_temps;
	double *nucleation_sizes;
  std::string temp_file_string;

  // Modular temperature source interface
  std::unique_ptr<TemperatureSource> temperature_source;
  bool use_temperature_source;  // Flag to enable new modular system
  double fast_forward_search_window;  // Search window for fast-forward (default 0.1s)
  
  // Temperature file handling variables (legacy HDF5 system)
  DoubleQueueContainer temp_in;
  DoubleQueueContainer time_in;
  double prior_time;
  int t_active;

  // HDF5 chunked reading variables
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
