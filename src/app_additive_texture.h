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
AppStyle(additive/texture,AppAdditiveTexture)
// Deprecated alias kept for backward compatibility with existing input
// scripts. Resolves to a thin subclass that emits a one-shot warning at
// parse time, then forwards to AppAdditiveTexture verbatim.
AppStyle(additive_temperature_texture,AppAdditiveTextureDeprecated)

#else

#ifndef SPK_APP_ADDITIVE_TEXTURE_H
#define SPK_APP_ADDITIVE_TEXTURE_H

#include "app_potts_quaternion.h"
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <memory>

namespace SPPARKS_NS {

class ThermalManager;

class AppAdditiveTexture : public AppPottsQuaternion {
  // ThermalManager owns the temperature-I/O orchestration and reaches the
  // app's lattice/state arrays (T, xyz, neighbor lists, ghost-comm config,
  // alloy thresholds) through a typed back-pointer.
  friend class ThermalManager;
 public:
  enum MisorientationTargetMode {
    MISORI_TARGET_GRADIENT = 0,
    MISORI_TARGET_RANDOM = 1,
    MISORI_TARGET_RANDOM_ROTATION = 2
  };

  enum NucleationMode {
    NUCLEATION_THRESHOLD = 0,   // Existing binary threshold
    NUCLEATION_CONTINUOUS = 1   // Continuous Gaussian probability
  };

  AppAdditiveTexture(class SPPARKS *, int, char **);
  virtual ~AppAdditiveTexture();
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

  // Powder phase activation methods
  void activate_powder_sites();

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

  // Temperature-I/O orchestration: source setup/update, laser scan path,
  // multi-pass pause/reset scheduling, fast-forward predictors, and
  // solidification-band temperature smoothing + diagnostics all live here.
  std::unique_ptr<ThermalManager> thermal;

  // Index-based selective ghost communication.
  // Only the specified iarray/darray indices are communicated to ghost sites;
  // remaining arrays are local-only (used for I/O, physics, not by KMC neighbors).
  const int *ghost_iindices;   // iarray indices to ghost-comm (set once in constructor)
  int nghost_iarray;           // count
  const int *ghost_dindices;   // darray indices to ghost-comm (set once in constructor)
  int nghost_darray;           // count

  // Powder activation tracking
  double last_powder_activation_time;  // Track when we last activated powder sites

  // Post-solidification single-voxel grain cleanup (opt-in via
  // `single_voxel_cleanup`). When a voxel finishes the nrefine smoothing
  // window and has no same-spin 26-connected neighbor, flip it to the
  // energy-minimizing neighbor grain. solid_d == -nrefine-3 marks the
  // cleanup as already resolved for a voxel so it is not reconsidered.
  bool      single_voxel_cleanup_enabled;
  long long n_single_voxel_flips;  // counter reset each app_update() call

  // When true, smooth_site() evaluates every distinct neighbor spin per
  // pass and adopts the energy-minimizing one (steepest descent), rather
  // than the default behavior of testing a single random proposal. Tied
  // to the input command `smooth_greedy_multiproposal on|off`. Has no
  // effect on the Boltzmann acceptance gate, so a non-zero `temperature`
  // setting is still respected for the uphill branch.
  bool      smooth_greedy_multiproposal_enabled;

  // When true, nucleation events initialize solid_d[i] = -1 for both the
  // seed site (execute_nucleation_event) and the captured shell sites
  // (nucleation_particle_flipper) so they enter the same smoothing window
  // as epitaxially solidified sites. When false (default) the historical
  // sentinel values (-nrefine-2 for the seed, -nrefine-3 for the shell)
  // are used, which excludes nuclei from smooth_site() and the greedy
  // variant. Tied to the input command `nuclei_smooth on|off`.
  bool      nuclei_smooth_enabled;

};

// Deprecated alias for app_style 'additive_temperature_texture'.
// Forwards construction to AppAdditiveTexture and emits a one-shot
// deprecation warning so existing input scripts keep working while
// users migrate to the canonical 'additive/texture' style.
class AppAdditiveTextureDeprecated : public AppAdditiveTexture {
 public:
  AppAdditiveTextureDeprecated(class SPPARKS *, int, char **);
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
