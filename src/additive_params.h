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

#ifndef SPK_ADDITIVE_PARAMS_H
#define SPK_ADDITIVE_PARAMS_H

#include <cstdio>
#include <set>
#include <string>

namespace SPPARKS_NS {

class Error;
class ThermalManager;

// Microstructure parameters and their input commands, shared by the CPU
// (additive/texture) and Kokkos (additive/texture/kk) apps, with the defaults
// both use. The nuclei volumes scale with the lattice spacing, so each app's
// constructor sets those from dx.
struct AdditiveParams {
  // Rotation-axis choice for the per-hop misorientation step. random_rotation
  // (default) rotates about a uniformly random axis -- the isotropic random
  // walk the diffusion coefficient is calibrated for. random rotates the
  // nearest <100> toward a random direction. The former gradient target was
  // removed: 3D EBSD (Lamb et al. 2026) shows cells keep their lattice and
  // switch <100> growth variants at fusion lines rather than rotating toward
  // the heat flux.
  enum MisorientationTargetMode {
    MISORI_TARGET_RANDOM = 1,
    MISORI_TARGET_RANDOM_ROTATION = 2
  };

  double tl = 1723;    // liquidus temperature (K)
  double ts = 1673;    // solidus temperature (K)
  double no = 1e15;    // nucleation density (1/m^3)
  double tc = 5;       // critical undercooling (K)
  double tsig = 3;     // undercooling standard deviation (K)
  double size_norm;    // mean nuclei volume (m^3), default 2 dx^3
  double size_sig;     // nuclei volume standard deviation (m^3), default dx^3
  // Front velocity V(Tcool) in m/s, Tcool = tl - T in K: the polynomial
  // solid_front_coeffs (highest power first), or, when solid_front_length
  // is 0, the power law solid_front_a * Tcool^solid_front_n.
  double *solid_front_coeffs = new double[4]{1.091e-5, -2.034e-4, 2.74e-3, 1.151e-4};
  int solid_front_length = 4;
  double solid_front_a = 0.0, solid_front_n = 0.0;
  double front_velocity(double Tcool) const;

  // Texture model: mobility = c1 + c2*cos(c3*theta), c1 fixed at 1.0.
  double c1 = 0.5, c2 = 0.5, c3 = 2.5;

  // Misorientation random walk (`misorientation_diffusion <D> [thresh] [alpha]`).
  // mis_diffusion: D in rad^2/m (input in deg^2/m); <= 0 disables rotation.
  // misorient_alpha > 0 makes D_eff = D * g(Tcool/(tl-ts))^2 with
  // g(x) = (exp(alpha x)-1)/(exp(alpha)-1); 0 = constant D.
  // mis_thresh (rad): a single step at or above it starts a new grain (default 2 deg).
  double mis_diffusion = 0.0;
  double mis_thresh = 0.034906585039886591;   // 2 deg
  double misorient_alpha = 0.0;
  // misorientation_grain_walk <D_deg2_per_m> [rho] [path|height]: coherent
  // per-grain rotation walk (grain_walk.h); rad^2/m, 0 = off.
  double grain_walk_D = 0.0;
  double grain_walk_rho = 0.0;  // axis persistence per level (0 memoryless, ->1 fixed bending axis)
  int grain_walk_path = 1;      // 1: level index = inheritance path length (per hop); 0: lattice z level
  int misorientation_target_mode = MISORI_TARGET_RANDOM_ROTATION;
  // Nucleated shell sites take one walk step of sqrt(D_eff * distance_from_seed)
  // relative to the seed. `nucleation_misorientation_target off|none` disables.
  bool nucleation_misorient_enabled = true;

  // Boundary smoothing: greedy steepest descent over the distinct neighbour
  // spins instead of one random proposal; `copy` takes one neighbour's
  // quaternion instead of the sign-aligned average; nuclei_smooth puts
  // nucleated sites inside the smoothing window.
  bool smooth_greedy_multiproposal_enabled = false;
  bool smooth_orientation_copy = false;
  bool nuclei_smooth_enabled = false;

  // Material commands given by the deck or a material card, for the
  // completeness check when a card is loaded.
  std::set<std::string> material_given;

  ~AdditiveParams() { delete[] solid_front_coeffs; }

  // Handle a shared microstructure or thermal command. Returns false if
  // `command` is not one of them. `me` is the rank, for screen output.
  bool parse_command(const char *command, int narg, char **arg,
                     Error *error, FILE *screen, int me, ThermalManager &thermal);

  // Warn (rank 0) when even the shortest hop's rotation, sqrt(D*dx) at the
  // full D, reaches mis_thresh: nearly every solidifying site would then
  // start a new grain. dx is the lattice spacing in m.
  void check_misorientation_step(double dx, Error *error, int me) const;
  // With a material card loaded, every material value must come from the card
  // or the deck: none falls back to a built-in default.
  void check_material_complete(Error *error, const ThermalManager &thermal) const;
  // The loop step is sector nstop x the thermal timestep, while the physics
  // always advances one thermal timestep: anything but nstop 1 desyncs them.
  void check_sector_stop(Error *error, double nstop, double tstop) const;
};

}

#endif
