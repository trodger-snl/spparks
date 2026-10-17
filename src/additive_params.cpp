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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "additive_params.h"
#include "error.h"
#include "thermal_manager.h"
#include "math_const.h"

using namespace SPPARKS_NS;
using namespace MathConst;

static bool on_off(const char *cmd, const char *v, Error *error)
{
  if (strcmp(v, "on") == 0) return true;
  if (strcmp(v, "off") == 0) return false;
  char msg[128];
  snprintf(msg, sizeof(msg), "%s: argument must be on or off", cmd);
  error->all(FLERR, msg);
  return false;
}

/* ---------------------------------------------------------------------- */

bool AdditiveParams::parse_command(const char *command, int narg, char **arg,
                                   Error *error, FILE *screen, int me,
                                   ThermalManager &thermal)
{
  if (!screen || me != 0) screen = nullptr;   // screen output on rank 0 only

  // One positive/non-negative scalar each.
  struct Scalar { const char *name; double *v; bool zero_ok; };
  const Scalar scalars[] = {
    {"liquidus", &tl, false}, {"solidus", &ts, false},
    {"nucleation_density", &no, true}, {"critical_undercooling", &tc, true},
    {"undercooling_deviation", &tsig, true}, {"mean_nuclei_volume", &size_norm, true},
    {"nuclei_volume_deviation", &size_sig, true}};
  for (const Scalar &s : scalars) {
    if (strcmp(command, s.name) != 0) continue;
    char msg[128];
    snprintf(msg, sizeof(msg), "Illegal %s command", s.name);
    if (narg != 1) error->all(FLERR, msg);
    *s.v = atof(arg[0]);
    if (s.zero_ok ? *s.v < 0 : *s.v <= 0) error->all(FLERR, msg);
    return true;
  }

  // Temperature I/O lives in the ThermalManager.
  if (strcmp(command, "material") == 0) {
    // The card's app-level values; deck commands after this one override them.
    thermal.material_cmd(narg, arg);
    const MaterialCard &c = *thermal.card();
    auto take = [](double &dst, MaterialCard::opt v) { if (v) dst = *v; };
    take(tl, c.liquidus); take(ts, c.solidus);
    take(no, c.nucleation_density); take(tc, c.critical_undercooling);
    take(tsig, c.undercooling_deviation); take(size_norm, c.mean_nuclei_volume);
    take(size_sig, c.nuclei_volume_deviation);
    if (!c.front.empty()) {
      delete[] solid_front_coeffs;
      solid_front_length = (int) c.front.size();
      solid_front_coeffs = new double[solid_front_length];
      std::copy(c.front.begin(), c.front.end(), solid_front_coeffs);
    }
  }
  else if (strcmp(command, "setup_temperature_source") == 0) thermal.setup_cmd(narg, arg);
  else if (strcmp(command, "laser_path") == 0) thermal.laser_path_cmd(narg, arg);
  else if (strcmp(command, "laser_fluctuations") == 0) thermal.laser_fluctuations_cmd(narg, arg);
  else if (strcmp(command, "temperature_smooth") == 0) thermal.smooth_cmd(narg, arg);
  else if (strcmp(command, "nodal_smooth") == 0) thermal.nodal_smooth_cmd(narg, arg);
  else if (strcmp(command, "fast_forward_search_window") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal fast_forward_search_window command");
    thermal.set_fast_forward_search_window(atof(arg[0]));

  } else if (strcmp(command, "solid_front_vel") == 0) {
    if (narg < 1) error->all(FLERR, "Illegal solid_front_vel command");
    int n = atoi(arg[0]);
    if (n <= 0) error->all(FLERR, "Illegal solidification front velocity specification");
    if (narg != n + 1)
      error->all(FLERR, "Illegal solid_front_vel command: expected <n> followed "
                        "by n coefficients");
    delete[] solid_front_coeffs;
    solid_front_length = n;
    solid_front_coeffs = new double[n];
    for (int i = 0; i < n; i++) solid_front_coeffs[i] = atof(arg[i + 1]);

  } else if (strcmp(command, "texture_parameters") == 0) {
    if (narg == 3)
      error->all(FLERR,
        "texture_parameters now takes 2 arguments (c2 c3), not 3. "
        "The baseline coefficient c1 has been fixed at 1.0 because "
        "only mobility ratios affect selection. To migrate an old "
        "input file: texture_parameters c2/c1 c3 (divide old c2 by "
        "old c1, drop old c1)");
    if (narg != 2) error->all(FLERR, "Illegal texture_parameters command");
    // c1 is fixed at 1.0: roulette-wheel selection only sees mobility ratios.
    c1 = 1.0;
    c2 = atof(arg[0]);
    c3 = atof(arg[1]);

    // mob(theta) = 1 + c2*cos(c3*theta) over theta in [0, theta_max], the
    // misorientation of the <100> family from the gradient folded into the
    // cubic fundamental zone.
    const double theta_max = std::acos(1.0 / std::sqrt(3.0));
    if (c2 < 0.0)
      error->all(FLERR,
        "texture_parameters: c2 must be >= 0 (negative amplitude "
        "would mean misaligned grains grow faster than aligned ones, "
        "which is unphysical for cubic solidification)");
    if (c3 < 0.0) error->all(FLERR, "texture_parameters: c3 must be >= 0");
    // Monotonic on [0, theta_max] requires c3*theta_max <= pi.
    if (c3 * theta_max > MY_PI + 1.0e-12) {
      char msg[256];
      snprintf(msg, sizeof(msg),
        "texture_parameters: c3 = %g exceeds monotonicity limit "
        "pi/theta_max ~ %g; mobility would be non-monotonic in "
        "misorientation angle", c3, MY_PI / theta_max);
      error->all(FLERR, msg);
    }
    // A negative minimum mobility is allowed: both apps clamp it to a
    // zero-weight candidate (strong selection).
    double mob_min = 1.0 + c2 * std::cos(c3 * theta_max);
    if (mob_min < 0.0 && screen)
      fprintf(screen, "texture_parameters: minimum mobility %g < 0 at theta_max = %g rad; "
              "such candidates get zero weight (strong selection)\n", mob_min, theta_max);
    if (c2 == 0.0 && screen)
      fprintf(screen, "WARNING: texture_parameters c2 = 0; mobility has no "
              "orientation dependence\n");

  } else if (strcmp(command, "misorientation_diffusion") == 0) {
    // misorientation_diffusion <D_deg2_per_m> [thresh_deg] [alpha]
    // thresh_deg left out keeps the app's default (2 deg)
    if (narg < 1 || narg > 3) error->all(FLERR, "Illegal misorientation_diffusion command");
    double D_deg2 = atof(arg[0]);
    mis_diffusion = D_deg2 * (MY_PI / 180.0) * (MY_PI / 180.0);
    if (narg >= 2) mis_thresh = atof(arg[1]) * MY_PI / 180.0;
    if (mis_thresh < 0)
      error->all(FLERR, "Illegal misorientation_diffusion threshold: must be >= 0");
    misorient_alpha = (narg == 3) ? atof(arg[2]) : 0.0;
    if (misorient_alpha < 0)
      error->all(FLERR, "Illegal misorientation_diffusion alpha: must be >= 0 (0 = constant D)");
    if (screen)
      fprintf(screen, "misorientation_diffusion D = %g deg^2/m, thresh = %g deg, alpha = %g\n",
              D_deg2, mis_thresh * 180.0 / MY_PI, misorient_alpha);

  } else if (strcmp(command, "misorientation_grain_walk") == 0) {
    // misorientation_grain_walk <D_deg2_per_m> [rho] [path|height]: rho in [0,1)
    // correlates the rotation between levels (persistent bending,
    // L_c ~ dx/(1-rho)); the level index is the inheritance path length in
    // lattice units (path, default) or the lattice z level (height).
    if (narg < 1 || narg > 3) error->all(FLERR, "Illegal misorientation_grain_walk command");
    if (narg == 3) {
      if (strcmp(arg[2], "path") == 0) grain_walk_path = 1;
      else if (strcmp(arg[2], "height") == 0) grain_walk_path = 0;
      else error->all(FLERR, "misorientation_grain_walk: mode must be path or height");
    }
    grain_walk_D = atof(arg[0]) * (MY_PI / 180.0) * (MY_PI / 180.0);
    if (grain_walk_D < 0) error->all(FLERR, "Illegal misorientation_grain_walk D: must be >= 0");
    grain_walk_rho = (narg >= 2) ? atof(arg[1]) : 0.0;
    if (grain_walk_rho < 0.0 || grain_walk_rho >= 1.0)
      error->all(FLERR, "Illegal misorientation_grain_walk rho: must be in [0,1)");
    if (screen)
      fprintf(screen, "misorientation_grain_walk D = %g deg^2/m, rho = %g, %s\n", atof(arg[0]),
              grain_walk_rho, grain_walk_path ? "path" : "height");

  } else if (strcmp(command, "misorientation_function") == 0 ||
             strcmp(command, "misorientation_cap") == 0) {
    error->all(FLERR, "misorientation_function/misorientation_cap were replaced by "
               "'misorientation_diffusion <D_deg2_per_m> [thresh_deg] [alpha]' "
               "(docs/MISORIENTATION_DIFFUSION_PLAN.md)");

  } else if (strcmp(command, "misorientation_target") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal misorientation_target command");
    if (strcmp(arg[0], "random") == 0)
      misorientation_target_mode = MISORI_TARGET_RANDOM;
    else if (strcmp(arg[0], "random_rotation") == 0)
      misorientation_target_mode = MISORI_TARGET_RANDOM_ROTATION;
    else if (strcmp(arg[0], "gradient") == 0)
      error->all(FLERR, "misorientation_target gradient was removed (cells switch <100> "
                 "variants at fusion lines rather than rotating toward the heat flux); "
                 "use random_rotation");
    else
      error->all(FLERR, "Illegal misorientation_target value: use random_rotation or random");
    if (screen) fprintf(screen, "misorientation_target = %s\n", arg[0]);

  } else if (strcmp(command, "nucleation_misorientation_target") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal nucleation_misorientation_target command");
    if (strcmp(arg[0], "on") == 0) nucleation_misorient_enabled = true;
    else if (strcmp(arg[0], "off") == 0 || strcmp(arg[0], "none") == 0)
      nucleation_misorient_enabled = false;
    else
      error->all(FLERR, "Illegal nucleation_misorientation_target value: use on, off, or none "
                 "(shells use the misorientation_target axis rule)");
    if (screen) fprintf(screen, "nucleation_misorientation_target = %s\n", arg[0]);

  } else if (strcmp(command, "smooth_greedy_multiproposal") == 0) {
    // on: each nrefine smoothing pass picks the energy-minimizing distinct
    // neighbour spin (steepest descent) instead of one random proposal.
    if (narg != 1) error->all(FLERR, "Illegal smooth_greedy_multiproposal command");
    smooth_greedy_multiproposal_enabled = on_off(command, arg[0], error);
    if (screen)
      fprintf(screen, "smooth_greedy_multiproposal %s\n",
              smooth_greedy_multiproposal_enabled ? "enabled" : "disabled");

  } else if (strcmp(command, "smooth_orientation") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal smooth_orientation command");
    if (strcmp(arg[0], "copy") == 0) smooth_orientation_copy = true;
    else if (strcmp(arg[0], "average") == 0) smooth_orientation_copy = false;
    else error->all(FLERR, "smooth_orientation: argument must be average or copy");
    if (screen)
      fprintf(screen, "smooth_orientation %s\n", smooth_orientation_copy ? "copy" : "average");

  } else if (strcmp(command, "nuclei_smooth") == 0) {
    // on: nucleated sites (seed and captured shell) start inside the
    // smoothing window instead of on the lock-out sentinels. Default off.
    if (narg != 1) error->all(FLERR, "Illegal nuclei_smooth command");
    nuclei_smooth_enabled = on_off(command, arg[0], error);
    if (screen)
      fprintf(screen, "nuclei_smooth %s\n", nuclei_smooth_enabled ? "enabled" : "disabled");

  } else return false;
  return true;
}

/* ---------------------------------------------------------------------- */

void AdditiveParams::check_misorientation_step(double dx, Error *error, int me) const
{
  if (me != 0 || mis_diffusion <= 0.0) return;
  const double step = sqrt(mis_diffusion * dx);
  if (step < mis_thresh) return;
  char msg[256];
  snprintf(msg, sizeof msg,
           "misorientation_diffusion step sqrt(D*dx) = %g deg is at or above the "
           "%g deg new-grain threshold: nearly every solidifying site will start "
           "a new grain", step * 180.0 / MY_PI, mis_thresh * 180.0 / MY_PI);
  error->warning(FLERR, msg);
}
