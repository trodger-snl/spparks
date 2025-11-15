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

  // Void generation methods
  void generate_voids(class RandomPark *);

 protected:
	// Void structure for tracking spherical voids
	struct Void {
		double x, y, z;  // Center coordinates
		double radius;   // Radius in meters

		Void(double x_, double y_, double z_, double r_)
			: x(x_), y(y_), z(z_), radius(r_) {}
	};

	double *mobility_out;
	double *solid_d;
		double *neigh_dist;
 	// Active flag for site visualization and state tracking
 	// 0: inactive, 1: active layer, 2: molten, 3: solidified, 5: void
 	int *active_flag;
 	const double R = 8.314459848; // Gas constant (J/mol/K)
	// Temperature array for thermal field
	double *T;
  double *G; //Temperature gradient at solidification
  double *V; //Solidification rate at solidification.
	double time_step;
	int nrefine;
	// Physical and simulation parameters
	double dt;
	double no;
	double tc;
	double tsig;
	double dx;
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
  double max_misorient;
  double t_cool_max;
  double mis_thresh;
  double misorient_alpha;
 	
 	// Debug parameters
 	int normal_finder_debug; // Flag to enable/disable normal_finder debugging

 	// Diagnostic parameters for temporary analysis (will be removed after refactoring)
 	int normal_finder_diagnostics; // Flag: 0=off, 1=on
 	FILE *normal_diagnostics_fp;   // Log file pointer

 	// Branch execution counters
 	struct DiagnosticCounters {
 		// Main path counters
 		int melt_surface_path;
 		int bulk_boundary_path;

 		// Melt surface sub-branches
 		int melt_sufficient_neighbors;
 		int melt_insufficient_neighbors;
 		int melt_nonsingular_matrix;
 		int melt_singular_matrix;
 		int melt_normalized;
 		int melt_zero_norm;

 		// Bulk/boundary sub-branches
 		int bulk_nonsingular_matrix;
 		int bulk_singular_matrix;
 		int bulk_normalized;
 		int bulk_zero_norm;

 		// Calling context
 		int call_from_nucleation;
 		int call_from_solidification;
 		int call_from_misorientation;
 		int call_from_other;

 		// Site property accumulators
 		double melt_neighbors_sum;
 		double melt_z_sum;
 		double bulk_neighbors_sum;
 		double bulk_temp_sum;
 		double boundary_neighbors_sum;
 		double boundary_temp_sum;

 		int total_calls;

 		DiagnosticCounters() : melt_surface_path(0), bulk_boundary_path(0),
 		                       melt_sufficient_neighbors(0), melt_insufficient_neighbors(0),
 		                       melt_nonsingular_matrix(0), melt_singular_matrix(0),
 		                       melt_normalized(0), melt_zero_norm(0),
 		                       bulk_nonsingular_matrix(0), bulk_singular_matrix(0),
 		                       bulk_normalized(0), bulk_zero_norm(0),
 		                       call_from_nucleation(0), call_from_solidification(0),
 		                       call_from_misorientation(0), call_from_other(0),
 		                       melt_neighbors_sum(0.0), melt_z_sum(0.0),
 		                       bulk_neighbors_sum(0.0), bulk_temp_sum(0.0),
 		                       boundary_neighbors_sum(0.0), boundary_temp_sum(0.0),
 		                       total_calls(0) {}
 	} diag_counters;
 	
 	// Monte Carlo nucleation model parameters
	int *nucleation_flags;
	double *nucleation_temps;
	double *nucleation_sizes;

  // Void generation parameters
  double void_density;           // Voids per cubic meter
  double void_pore_fraction;     // Volumetric pore fraction (0.0 to 1.0)
  double void_radius_mean;       // Mean radius in micrometers
  double void_radius_std;        // Standard deviation in micrometers
  double void_radius_min;        // Minimum radius in micrometers
  double void_radius_max;        // Maximum radius in micrometers
  int enable_voids;              // Flag to enable void generation (0=off, 1=on)
  std::vector<Void> voids;       // List of generated voids for collision detection

  // Modular temperature source interface
  std::unique_ptr<TemperatureSource> temperature_source;
  bool use_temperature_source;  // Flag to enable new modular system
  double fast_forward_search_window;  // Search window for fast-forward (default 0.1s)
  
  int t_active;

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
