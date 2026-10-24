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
#include "thermal_host.h"
#include "additive_params.h"
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <memory>

namespace SPPARKS_NS {

class ThermalManager;
namespace quaternion { class SiteOrientationRNG; }

class AppAdditiveTexture : public AppPottsQuaternion, public AdditiveParams {
 public:
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
  virtual void nucleation_spins();
  virtual void  input_app(char *, int , char **);
  virtual void nucleation_init();
  // Grow a nucleus from site i to partRad sites. seed/Tcool: the seed voxel
  // every claimed site inherits from and the undercooling of the nucleation
  // event (recursion re-roots i, never seed).
  virtual void nucleation_particle_flipper(int i, int partRad, quaternion::SiteOrientationRNG &,
                                           int seed, double Tcool);
  virtual void mushy_phase(int, quaternion::SiteOrientationRNG &);
  virtual double site_energy(int);
  void site_event_solidification(int, double, quaternion::SiteOrientationRNG &);
  void execute_nucleation_event(int, double, quaternion::SiteOrientationRNG &);
  std::vector<double> normal_finder(int);
	double melt_misorientation(int, double, double, double);
  double mobility_of_orientation(const std::vector<double> &q, const std::vector<double> &grad_full);
  void apply_misorientation(int, int, double, quaternion::SiteOrientationRNG &);
  // Per-hop rotation angle (rad) of the misorientation random walk:
  // sqrt(D_eff * hop_m), D_eff = mis_diffusion * g(Tcool)^2 with the
  // exponential undercooling shape g when misorient_alpha > 0, else
  // mis_diffusion. Scaling as sqrt(hop) makes the accumulated misorientation
  // (RMS^2 = D * distance) independent of the lattice spacing.
  double diffusion_angle(double Tcool, double hop_m) const;
  // Rotate site i by mis_angle per the target mode, then apply the
  // mis_thresh new-grain check. Shared by the solidification path
  // (apply_misorientation) and the nucleation flipper.
  void rotate_toward_target(int, double, int, quaternion::SiteOrientationRNG &);
  std::vector<double> get_average_neighbor_quaternion(int site, int target_spin,
                                                      quaternion::SiteOrientationRNG &);
  void smooth_site(int site, quaternion::SiteOrientationRNG &);
  double site_energy_smooth(int site);

  // Post-smoothing single-voxel grain cleanup. Fires at most once per
  // voxel, when it exits the nrefine smoothing window and its full
  // 26-neighbor first shell is solidified. Returns true if the voxel
  // was resolved (either flipped or confirmed not a 1-voxel grain) and
  // false if the check was deferred because the neighborhood was still
  // maturing.
  bool flip_single_voxel_grain(int site, quaternion::SiteOrientationRNG &);

  // Every random draw of site i's events in one step comes from step_rng(i),
  // keyed by (deck seed, step, the site's GLOBAL id); the per-spin nucleation
  // properties are keyed by (deck seed, spin). A site's draws depend on
  // nothing else, so a run repeats at a fixed decomposition, and voids (or any
  // site drawing more or fewer numbers) change the microstructure only through
  // the physics around them, not by shifting a shared stream.
  quaternion::SiteOrientationRNG step_rng(int i) const;
  uint64_t rng_seed = 0;   // the deck seed
  uint64_t rng_step = 0;   // app_update calls so far

  // Powder phase activation methods
  virtual void activate_powder_sites();

  // Void generation methods
  void generate_voids(class RandomPark *);

 protected:
	// Void structure for tracking spherical voids
	struct Void {
		double x, y, z;  // Center coordinates
		double radius;   // Radius in lattice units (xyz units)

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
  double *spath; // d10: inheritance path length from the grain origin, lattice units (grain walk 'path' mode)
	int nrefine;
	// Physical and simulation parameters
	double dt;
	double dx;
  double t_room;
    
    
  // Nucleation and growth parameters
  double t_cool_max;

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

  // Lattice-state snapshot ThermalManager reads/writes. Populated by
  // refresh_thermal_host() at the top of app_update() and before dispatching
  // any thermal command in input_app(). See thermal_host.h.
  ThermalHost thermal_host_;
  virtual void refresh_thermal_host();

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
