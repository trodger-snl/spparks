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
#include "am_raster.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <memory>

namespace SPPARKS_NS {

class AppAdditiveExtTempTexture : public AppPottsQuaternion {
 public:
  enum MisorientationTargetMode {
    MISORI_TARGET_GRADIENT = 0,
    MISORI_TARGET_RANDOM = 1
  };

  enum NucleationMode {
    NUCLEATION_THRESHOLD = 0,   // Existing binary threshold
    NUCLEATION_CONTINUOUS = 1   // Continuous Gaussian probability
  };

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
  void execute_nucleation_event(int, double);
  std::vector<double> normal_finder(int);
	double melt_misorientation(int, double, double, double);
  void apply_misorientation(int, double, class RandomPark *);
  std::vector<double> get_average_neighbor_quaternion(int site, int target_spin);
  void smooth_site(int site);
  double site_energy_smooth(int site);

  // Post-smoothing single-voxel grain cleanup. Fires at most once per
  // voxel, when it exits the nrefine smoothing window and its full
  // 26-neighbor first shell is solidified. Returns true if the voxel
  // was resolved (either flipped or confirmed not a 1-voxel grain) and
  // false if the check was deferred because the neighborhood was still
  // maturing.
  bool flip_single_voxel_grain(int site);

  // Gaussian smoothing of T[] inside a user-specified temperature window
  // (intended for the solidification band). Gated by active_flag so only
  // molten/solidifying sites participate. Requires ghost-comm of T; a
  // comm->all_selective is issued per pass.
  void apply_temperature_smoothing();

  // In-situ diagnostic: isotherm compactness of the T >= liquidus region.
  // Counts pool area A (sites with T >= tl) and boundary-site count P
  // (those with any neighbor whose T < tl), then reports P/sqrt(A).
  // Ideal circular pool ~= 2*sqrt(pi) ~= 3.545; wiggly tails push it higher.
  // Requires T ghosts to be current before the call.
  void compute_smoothing_diagnostics();

  // Powder phase activation methods
  bool is_powder_eligible_site(int i);
  void activate_powder_sites();

  // Modular temperature source methods
  void setup_temperature_source_cmd(int narg, char **arg);
  void laser_path_cmd(int narg, char **arg);
  void laser_fluctuations_cmd(int narg, char **arg);
  void update_temperature_from_source(double simulation_time);

  // Fast-forward predictor for the Rosenthal source. Walks scan_layer
  // (on a copy) to find the earliest time at which a Rosenthal-driven
  // temperature on the local domain rises above threshold_temp, given
  // current sim time. Returns +inf if the laser never reaches threshold
  // before all repeats are exhausted.
  double rosenthal_next_active_time(double current_time, double threshold_temp);

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
  int misorientation_target_mode;

 	// Monte Carlo nucleation model parameters
	int *nucleation_flags;
	double *nucleation_temps;
	double *nucleation_sizes;
	int nucleation_mode;    // NUCLEATION_THRESHOLD or NUCLEATION_CONTINUOUS
	std::unordered_map<int, double> T_prev_map;  // Previous temperature for molten sites (continuous nucleation)

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

  // Laser scan path state for the analytical Rosenthal source.
  // Coordinates are SI meters; speeds are m/s. Not used by HDF5 sources.
  RASTER::Layer scan_layer;
  double scan_layer_z;       // physical Z of the scan plane (meters)
  double scan_layer_time;    // sim time at which scan_layer's pose is current
  bool   scan_layer_active;  // true while the path has remaining motion
  bool   laser_path_set;     // true once laser_path has been parsed

  // Temperature optimization flags (for performance testing)
  bool opt_use_spatial_grid;    // Use spatial grid for element lookup (default: true)
  bool opt_use_element_cache;   // Cache element per site (default: true)
  bool opt_use_nodal_precompute; // Precompute nodal temps each step (default: true, requires element_cache)

  int t_active;

  // Index-based selective ghost communication.
  // Only the specified iarray/darray indices are communicated to ghost sites;
  // remaining arrays are local-only (used for I/O, physics, not by KMC neighbors).
  const int *ghost_iindices;   // iarray indices to ghost-comm (set once in constructor)
  int nghost_iarray;           // count
  const int *ghost_dindices;   // darray indices to ghost-comm (set once in constructor)
  int nghost_darray;           // count

  // Powder activation tracking
  double last_powder_activation_time;  // Track when we last activated powder sites

  // Solidification-band temperature smoothing (opt-in via `temperature_smooth`).
  // Applied once per temperature update, after T[] has been filled from the
  // modular source. Only sites whose T falls inside the window participate;
  // a guard band tapers the blend factor to zero at the outer edges so that
  // sites crossing in/out of the window don't see a discontinuity.
  bool   temperature_smooth_enabled;
  double smooth_tmin;        // lower edge of full-strength window (K)
  double smooth_tmax;        // upper edge of full-strength window (K)
  double smooth_guard;       // guard-band half-width (K); blend ramps to 0 here
  double smooth_sigma;       // Gaussian width (lattice sites)
  double smooth_alpha;       // blend factor (0=off, 1=replace with neighbor avg)
  int    smooth_passes;      // number of Gaussian passes per step
  int    smooth_diag_interval;  // report compactness every N steps; 0 disables
  std::vector<double> smooth_buffer;  // scratch for double-buffered pass

  // Post-solidification single-voxel grain cleanup (opt-in via
  // `single_voxel_cleanup`). When a voxel finishes the nrefine smoothing
  // window and has no same-spin 26-connected neighbor, flip it to the
  // energy-minimizing neighbor grain. solid_d == -nrefine-3 marks the
  // cleanup as already resolved for a voxel so it is not reconsidered.
  bool      single_voxel_cleanup_enabled;
  long long n_single_voxel_flips;  // counter reset each app_update() call

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
