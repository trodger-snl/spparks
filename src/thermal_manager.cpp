/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.

   ThermalManager: temperature-I/O orchestration extracted from
   AppAdditiveTexture. Owns the modular temperature source, laser scan-path
   state, multi-pass pause/reset scheduling, the fast-forward predictors, and
   the optional solidification-band temperature smoothing + diagnostics. The
   microstructure-evolution physics stays in AppAdditiveTexture.
------------------------------------------------------------------------- */

#include "thermal_manager.h"
#include "thermal_host.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_moser.h"
#include "temperature_source_hdf5_unstructured.h"
#include "comm_lattice.h"
#include "domain.h"
#include "timer.h"
#include "error.h"
#include "math_const.h"
#include "spktype.h"

#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace SPPARKS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

ThermalManager::ThermalManager(SPPARKS *spk, ThermalHost *host) :
  Pointers(spk), host_(host)
{
  // Initialize modular temperature source system
  temperature_source = nullptr;
  fast_forward_search_window = 0.1;  // Default 100ms search window

  // Rosenthal scan-path state (only used when temperature_source is Rosenthal)
  scan_layer = RASTER::Layer();
  scan_layer_z = 0.0;
  scan_layer_time = 0.0;
  scan_layer_active = false;
  laser_path_set = false;

  // Multi-pass pause control (disabled until laser_path specifies pause/pause_below)
  laser_pause_constant     = 0.0;
  laser_pause_below        = 0.0;
  laser_reset_temperature  = 0.0;
  pending_moser_passes     = 0;

  // Temperature optimization flags (all enabled by default)
  opt_use_spatial_grid = true;
  opt_use_element_cache = true;
  opt_use_nodal_precompute = true;

  // Solidification-band smoothing defaults (off until `temperature_smooth`)
  temperature_smooth_enabled = false;
  smooth_tmin  = 0.0;
  smooth_tmax  = 0.0;
  smooth_guard = 20.0;
  smooth_sigma_xy = 1.2;
  smooth_sigma_z  = 1.2;
  smooth_alpha = 0.4;
  smooth_passes = 1;
  smooth_diag_interval = 0;

  // Temperature-update timing accumulators
  g_t_temp_prepare = 0.0;
  g_t_temp_site_loop = 0.0;
  g_t_cache_build = 0.0;
}

/* ----------------------------------------------------------------------
   setup_temperature_source command: instantiate a temperature source via
   the factory and configure it with the remaining arguments.
------------------------------------------------------------------------- */

void ThermalManager::setup_cmd(int narg, char **arg)
{
  if (narg < 1) {
    error->all(FLERR,"Illegal setup_temperature_source command: must specify type");
  }

  std::string source_type = arg[0];
  std::vector<std::string> source_args;

  // Convert remaining arguments to string vector
  for (int i = 1; i < narg; i++) {
    source_args.push_back(std::string(arg[i]));
  }

  // Create temperature source using factory
  temperature_source = SPPARKS_NS::create_temperature_source(source_type, spk);
  if (!temperature_source) {
    error->all(FLERR,"Failed to create temperature source");
  }

  // Pass optimization flags to temperature source before setup
  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();
  if (hdf5_source) {
    hdf5_source->set_use_spatial_grid(opt_use_spatial_grid);
    hdf5_source->set_use_element_cache(opt_use_element_cache);
  }

  // Setup the temperature source with provided arguments
  temperature_source->setup_temperature_source(source_args);

  if (domain->me == 0) {
    std::cout << "Modular temperature source (" << source_type << ") enabled" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_path command: define a single linear scan, optionally repeated N
   times. Coordinates and velocity are SI (meters, m/s).
------------------------------------------------------------------------- */

void ThermalManager::laser_path_cmd(int narg, char **arg)
{
  // setup_temperature_source must precede laser_path: the Moser source
  // consumes the path geometry inside build_scan, and the Rosenthal
  // source needs r_min sized below.
  if (!temperature_source)
    error->all(FLERR,
      "laser_path: setup_temperature_source must be called first "
      "(no active temperature source). Add a 'setup_temperature_source "
      "<type> ...' line before 'laser_path'.");

  // Expected token layout (0-based, command name already stripped):
  //   start X0 Y0 Z0 end X1 Y1 speed V [repeats N]
  //     [pause <T> | pause_below <Tk>]
  if (narg < 9)
    error->all(FLERR,"Illegal laser_path command: expected start X0 Y0 Z0 end X1 Y1 speed V [repeats N] [pause T | pause_below Tk]");
  if (strcmp(arg[0],"start") != 0)
    error->all(FLERR,"laser_path: expected keyword 'start'");
  const double x0 = atof(arg[1]);
  const double y0 = atof(arg[2]);
  const double z0 = atof(arg[3]);
  if (strcmp(arg[4],"end") != 0)
    error->all(FLERR,"laser_path: expected keyword 'end' after start coordinates");
  const double x1 = atof(arg[5]);
  const double y1 = atof(arg[6]);
  if (strcmp(arg[7],"speed") != 0)
    error->all(FLERR,"laser_path: expected keyword 'speed' after end coordinates");
  const double v = atof(arg[8]);
  if (v <= 0.0)
    error->all(FLERR,"laser_path: speed must be > 0");

  int repeats = 1;
  double pause_constant   = 0.0;
  double pause_below_T    = 0.0;
  double reset_temperature_T = 0.0;
  int i = 9;
  while (i < narg) {
    if (strcmp(arg[i],"repeats") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'repeats'");
      repeats = atoi(arg[i+1]);
      if (repeats < 1)
        error->all(FLERR,"laser_path: repeats must be >= 1");
      i += 2;
    }
    else if (strcmp(arg[i],"pause") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'pause'");
      pause_constant = atof(arg[i+1]);
      if (pause_constant <= 0.0)
        error->all(FLERR,"laser_path: pause must be > 0 seconds");
      i += 2;
    }
    else if (strcmp(arg[i],"pause_below") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'pause_below'");
      pause_below_T = atof(arg[i+1]);
      if (pause_below_T <= 0.0)
        error->all(FLERR,"laser_path: pause_below threshold must be > 0 K");
      i += 2;
    }
    else if (strcmp(arg[i],"reset_temperature") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'reset_temperature'");
      reset_temperature_T = atof(arg[i+1]);
      if (reset_temperature_T <= 0.0)
        error->all(FLERR,"laser_path: reset_temperature must be > 0 K");
      i += 2;
    }
    else {
      error->all(FLERR,"laser_path: unknown keyword (expected 'repeats', 'pause', 'pause_below', or 'reset_temperature')");
    }
  }

  if (pause_constant > 0.0 && pause_below_T > 0.0)
    error->all(FLERR,
      "laser_path: 'pause' and 'pause_below' are mutually exclusive. "
      "Use 'pause <T>' for a fixed inter-pass delay, or "
      "'pause_below <Tk>' for a temperature-threshold-gated delay.");
  if (reset_temperature_T > 0.0 && pause_constant <= 0.0 && pause_below_T <= 0.0)
    error->all(FLERR,
      "laser_path: 'reset_temperature' requires either 'pause <T>' or "
      "'pause_below <Tk>' to define when the reset is applied.");

  // Both pause modes require a time-resolved source. Rosenthal is the
  // steady-state moving-point solution, so each pass is already independent
  // of every preceding pass; reject pause at parse time.
  const bool src_is_rosenthal =
    (rosenthal_source() != nullptr);
  if ((pause_constant > 0.0 || pause_below_T > 0.0) && src_is_rosenthal) {
    error->all(FLERR,
      "laser_path pause/pause_below: not supported for the Rosenthal "
      "source. Rosenthal is a steady-state moving-point solution; "
      "multi-pass repeats are already physically independent (no "
      "thermal history carries between passes). Use a time-resolved "
      "source (moser, finitediff [planned]) if you need cooling "
      "between passes.");
  }
  if ((pause_constant > 0.0 || pause_below_T > 0.0) && repeats < 2) {
    if (domain->me == 0) {
      std::cout << "laser_path: warning: pause keyword specified but "
                   "repeats=1; no inter-pass interval will occur."
                << std::endl;
    }
  }

  laser_pause_constant    = pause_constant;
  laser_pause_below       = pause_below_T;
  laser_reset_temperature = reset_temperature_T;

  // Build N identical paths back-to-back in the Rosenthal scan_layer
  // (Moser does not consume scan_layer).
  RASTER::Point a(x0,y0,0.0), b(x1,y1,0.0);
  std::vector<RASTER::Path> paths;
  paths.reserve(repeats);
  for (int r = 0; r < repeats; ++r) paths.emplace_back(a, b, v);

  scan_layer = RASTER::Layer(paths, /*thickness*/0);
  scan_layer_z = z0;
  scan_layer_time = host_->time;  // anchor the layer pose to the current sim time
  scan_layer_active = true;
  laser_path_set = true;

  // Size the singularity cutoff to the lattice so the 1/R prefactor at the
  // laser site stays finite. Only meaningful for a Rosenthal source.
  if (auto* ros = rosenthal_source()) {
    ros->set_r_min(0.5 * host_->dx);
  }

  // Hand the path geometry to the Moser source so it can build its own
  // GREENAM LaserScan/ScanIntegration. Dynamic scheduling (only pass 1 built
  // upfront, remaining appended one at a time) is used when the app must
  // interpose between passes: pause_below alone, or pause + reset_temperature.
  const bool dynamic_pass_scheduling =
    (pause_below_T > 0.0) ||
    (pause_constant > 0.0 && reset_temperature_T > 0.0);
  if (auto* moser = moser_source()) {
    if (dynamic_pass_scheduling) {
      moser->build_scan(host_->time, x0, y0, x1, y1, z0, v, /*repeats*/1, /*pause*/0.0);
      pending_moser_passes = repeats - 1;
    } else {
      moser->build_scan(host_->time, x0, y0, x1, y1, z0, v, repeats, pause_constant);
      pending_moser_passes = 0;
    }
  }

  if (domain->me == 0) {
    std::cout << "laser_path: (" << x0 << "," << y0 << "," << z0
              << ") -> (" << x1 << "," << y1 << "," << z0 << ")"
              << " speed=" << v << " m/s, repeats=" << repeats;
    if (pause_constant > 0.0)
      std::cout << ", pause=" << pause_constant << " s";
    if (pause_below_T > 0.0)
      std::cout << ", pause_below=" << pause_below_T << " K";
    if (reset_temperature_T > 0.0)
      std::cout << ", reset_temperature=" << reset_temperature_T << " K";
    std::cout << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_fluctuations command: in-source PSD form (Moser source only).
------------------------------------------------------------------------- */

void ThermalManager::laser_fluctuations_cmd(int narg, char **arg)
{
  if (narg < 1)
    error->all(FLERR,"Illegal laser_fluctuations command: expected psd <shape> ...");

  if (!temperature_source) {
    error->all(FLERR,"laser_fluctuations: setup_temperature_source must be called first");
  }
  MoserTemperatureSource *moser =
    moser_source();
  if (!moser) {
    error->all(FLERR,"laser_fluctuations: only the Moser unsteady Green's-function source supports fluctuations");
  }

  // The fluctuation table is materialized inside build_scan (called from
  // laser_path_cmd). Reject the reversed order explicitly.
  if (moser->has_scan()) {
    error->all(FLERR,
      "laser_fluctuations must be issued BEFORE laser_path "
      "(the fluctuation table is materialized inside laser_path once the "
      "scan duration is known). Correct sequence: "
      "setup_temperature_source moser ... ; laser_fluctuations psd ... ; "
      "laser_path start ... . If you intend to change fluctuations after "
      "the scan was built, re-issue laser_path so build_scan reruns.");
  }

  if (strcmp(arg[0],"psd") != 0)
    error->all(FLERR,"laser_fluctuations: expected 'psd' subcommand");
  if (narg < 2)
    error->all(FLERR,"laser_fluctuations psd: missing <shape>");

  // Optional top|bot|both selector between `psd` and the shape token.
  enum class Selector { NONE, TOP, BOT, BOTH };
  Selector which = Selector::NONE;
  int shape_idx = 1;
  if      (strcmp(arg[1],"top")  == 0) { which = Selector::TOP;  shape_idx = 2; }
  else if (strcmp(arg[1],"bot")  == 0) { which = Selector::BOT;  shape_idx = 2; }
  else if (strcmp(arg[1],"both") == 0) { which = Selector::BOTH; shape_idx = 2; }

  const bool is_keyhole = (moser->get_mode() == MoserTemperatureSource::MoserMode::KEYHOLE);
  if (is_keyhole && which == Selector::NONE)
    error->all(FLERR,
      "laser_fluctuations psd: keyhole mode requires top|bot|both selector");
  if (!is_keyhole && which != Selector::NONE)
    error->all(FLERR,
      "laser_fluctuations psd: top|bot|both selector is only valid in keyhole mode");

  if (shape_idx >= narg)
    error->all(FLERR,"laser_fluctuations psd: missing <shape>");

  MoserTemperatureSource::PsdSpec spec;
  spec.sigma_W = 0.0;
  spec.sigma_D = 0.0;
  spec.sigma_P = 0.0;
  spec.rho     = 0.0;
  spec.seed    = 12345UL;
  spec.dt_psd  = 5.0e-6;
  spec.tau     = 0.0;
  spec.f0      = 0.0;
  spec.df      = 0.0;

  if      (strcmp(arg[shape_idx],"white")       == 0) spec.shape = MoserTemperatureSource::PsdShape::WHITE;
  else if (strcmp(arg[shape_idx],"lorentzian")  == 0) spec.shape = MoserTemperatureSource::PsdShape::LORENTZIAN;
  else if (strcmp(arg[shape_idx],"pink")        == 0) spec.shape = MoserTemperatureSource::PsdShape::PINK;
  else if (strcmp(arg[shape_idx],"narrow_band") == 0) spec.shape = MoserTemperatureSource::PsdShape::NARROW_BAND;
  else error->all(FLERR,"laser_fluctuations psd: unknown shape (use white|lorentzian|pink|narrow_band)");

  int i = shape_idx + 1;
  while (i < narg) {
    const char *k = arg[i];
    if (i + 1 >= narg)
      error->all(FLERR,"laser_fluctuations psd: missing value for keyword");
    const double val = atof(arg[i+1]);
    if      (strcmp(k,"sigma_W") == 0) spec.sigma_W = val;
    else if (strcmp(k,"sigma_D") == 0) spec.sigma_D = val;
    else if (strcmp(k,"sigma_P") == 0) spec.sigma_P = val;
    else if (strcmp(k,"rho")     == 0) spec.rho     = val;
    else if (strcmp(k,"seed")    == 0) spec.seed    = static_cast<unsigned long>(atol(arg[i+1]));
    else if (strcmp(k,"dt")      == 0) spec.dt_psd  = val;
    else if (strcmp(k,"tau")     == 0) spec.tau     = val;
    else if (strcmp(k,"f0")      == 0) spec.f0      = val;
    else if (strcmp(k,"df")      == 0) spec.df      = val;
    else error->all(FLERR,"laser_fluctuations psd: unknown keyword");
    i += 2;
  }

  // Dispatch to the appropriate lobe(s).
  if (which == Selector::NONE || which == Selector::TOP) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_TOP, spec);
  }
  if (which == Selector::BOT) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_BOT, spec);
  }
  if (which == Selector::BOTH) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_TOP, spec);
    // Derive bot's seed from top's so the two streams are independent yet
    // identically reproducible across MPI ranks.
    MoserTemperatureSource::PsdSpec bot_spec = spec;
    bot_spec.seed = spec.seed ^ 0x9E3779B97F4A7C15ULL;
    moser->set_psd_spec(MoserTemperatureSource::LOBE_BOT, bot_spec);
  }

  if (domain->me == 0) {
    std::cout << "laser_fluctuations psd:\n";
    if (which != Selector::BOT) moser->print_lobe_info(MoserTemperatureSource::LOBE_TOP);
    if (which == Selector::BOT || which == Selector::BOTH)
      moser->print_lobe_info(MoserTemperatureSource::LOBE_BOT);
    std::cout.flush();
  }
}

/* ----------------------------------------------------------------------
   temperature_optimization command: toggle HDF5 source performance paths.
   Usage: temperature_optimization <option> <on|off>
------------------------------------------------------------------------- */

void ThermalManager::optimization_cmd(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal temperature_optimization command: requires 2 args (option on/off)");
  bool enable = (strcmp(arg[1],"on") == 0 || strcmp(arg[1],"1") == 0 || strcmp(arg[1],"true") == 0);

  // Get HDF5 source if available to pass flags
  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();

  if (strcmp(arg[0],"spatial_grid") == 0) {
    opt_use_spatial_grid = enable;
    if (hdf5_source) hdf5_source->set_use_spatial_grid(enable);
    if (domain->me == 0) fprintf(screen,"Temperature optimization: spatial_grid = %s\n", enable ? "ON" : "OFF");
  } else if (strcmp(arg[0],"element_cache") == 0) {
    opt_use_element_cache = enable;
    if (!enable) opt_use_nodal_precompute = false;  // nodal_precompute requires element_cache
    if (hdf5_source) hdf5_source->set_use_element_cache(enable);
    if (domain->me == 0) fprintf(screen,"Temperature optimization: element_cache = %s\n", enable ? "ON" : "OFF");
  } else if (strcmp(arg[0],"nodal_precompute") == 0) {
    if (enable && !opt_use_element_cache)
      error->all(FLERR,"nodal_precompute requires element_cache to be enabled");
    opt_use_nodal_precompute = enable;
    if (domain->me == 0) fprintf(screen,"Temperature optimization: nodal_precompute = %s\n", enable ? "ON" : "OFF");
  } else {
    error->all(FLERR,"Unknown temperature_optimization option (use: spatial_grid, element_cache, nodal_precompute)");
  }
}

/* ----------------------------------------------------------------------
   fast_forward_search_window command setter.
------------------------------------------------------------------------- */

void ThermalManager::set_fast_forward_search_window(double w)
{
  fast_forward_search_window = w;
  if (fast_forward_search_window <= 0.0)
    error->all(FLERR,"Illegal fast_forward_search_window value: must be > 0");
}

/* ----------------------------------------------------------------------
   temperature_smooth command: configure solidification-band smoothing.
------------------------------------------------------------------------- */

void ThermalManager::smooth_cmd(int narg, char **arg)
{
  // Usage: temperature_smooth <tmin> <tmax> <sigma_xy> <sigma_z>
  //                           [alpha] [guard] [passes] [diag_interval]
  if (narg < 4 || narg > 8)
    error->all(FLERR,"Illegal temperature_smooth command");
  smooth_tmin     = atof(arg[0]);
  smooth_tmax     = atof(arg[1]);
  smooth_sigma_xy = atof(arg[2]);
  smooth_sigma_z  = atof(arg[3]);
  if (narg >= 5) smooth_alpha         = atof(arg[4]);
  if (narg >= 6) smooth_guard         = atof(arg[5]);
  if (narg >= 7) smooth_passes        = atoi(arg[6]);
  if (narg >= 8) smooth_diag_interval = atoi(arg[7]);
  if (smooth_tmax <= smooth_tmin)
    error->all(FLERR,"temperature_smooth: tmax must be > tmin");
  if (smooth_sigma_xy <= 0.0)
    error->all(FLERR,"temperature_smooth: sigma_xy must be > 0");
  if (smooth_sigma_z <= 0.0)
    error->all(FLERR,"temperature_smooth: sigma_z must be > 0");
  if (smooth_alpha < 0.0 || smooth_alpha > 1.0)
    error->all(FLERR,"temperature_smooth: alpha must be in [0,1]");
  if (smooth_guard < 0.0)
    error->all(FLERR,"temperature_smooth: guard must be >= 0");
  if (smooth_passes < 1)
    error->all(FLERR,"temperature_smooth: passes must be >= 1");
  if (smooth_diag_interval < 0)
    error->all(FLERR,"temperature_smooth: diag_interval must be >= 0");
  temperature_smooth_enabled = true;
  if (domain->me == 0)
    fprintf(screen,
      "Temperature smoothing enabled: window=[%.1f,%.1f] K, "
      "guard=%.1f K, sigma_xy=%.2f sigma_z=%.2f sites, "
      "alpha=%.2f, passes=%d, diag_interval=%d\n",
      smooth_tmin, smooth_tmax, smooth_guard,
      smooth_sigma_xy, smooth_sigma_z,
      smooth_alpha, smooth_passes, smooth_diag_interval);
}

/* ----------------------------------------------------------------------
   nodal_smooth command: forward mesh-Laplacian smoothing to the source.
------------------------------------------------------------------------- */

void ThermalManager::nodal_smooth_cmd(int narg, char **arg)
{
  // Usage: nodal_smooth <sigma_xy> <sigma_z> <passes> <alpha>
  if (narg != 4)
    error->all(FLERR,"Illegal nodal_smooth command");
  if (!temperature_source)
    error->all(FLERR,"nodal_smooth requires setup_temperature_source first");
  const double sxy = atof(arg[0]);
  const double sz  = atof(arg[1]);
  const int    K   = atoi(arg[2]);
  const double al  = atof(arg[3]);
  if (sxy <= 0.0) error->all(FLERR,"nodal_smooth: sigma_xy must be > 0");
  if (sz  <= 0.0) error->all(FLERR,"nodal_smooth: sigma_z must be > 0");
  if (K    < 1)   error->all(FLERR,"nodal_smooth: passes must be >= 1");
  if (al < 0.0 || al > 1.0)
    error->all(FLERR,"nodal_smooth: alpha must be in [0,1]");
  temperature_source->enable_nodal_smoothing(sxy, sz, K, al);
  if (domain->me == 0)
    fprintf(screen,
      "Nodal mesh-Laplacian smoothing enabled: "
      "sigma_xy=%.3f sigma_z=%.3f passes=%d alpha=%.3f\n",
      sxy, sz, K, al);
}

/* ----------------------------------------------------------------------
   Fill app->T from the modular temperature source at simulation_time.
------------------------------------------------------------------------- */

void ThermalManager::update_temperatures(double simulation_time)
{
  timer->stamp();

  if (!temperature_source) {
    timer->stamp(TIME_APP);
    return; // No temperature source configured
  }

  double *T = host_->T;
  double **xyz = host_->xyz;
  const int nlocal = host_->nlocal;
  const double dx = host_->dx;

  // Cache the dynamic_cast results outside the loop.
  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();
  RosenthalTemperatureSource* ros_source = hdf5_source ? nullptr : rosenthal_source();

  // -------------------------------------------------------------------
  // HDF5 path: three optimization tiers.
  // -------------------------------------------------------------------
  if (hdf5_source && opt_use_element_cache && opt_use_nodal_precompute) {
    // Path 3: Fast path with nodal precomputation
    double t0 = MPI_Wtime();
    hdf5_source->prepare_for_timestep(simulation_time);
    double t1 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      T[i] = hdf5_source->get_temperature_at_site_fast(i);
    }
    double t2 = MPI_Wtime();

    double prepare_time = t1 - t0;
    if (prepare_time > 1.0) {
      g_t_cache_build += prepare_time;
    } else {
      g_t_temp_prepare += prepare_time;
    }
    g_t_temp_site_loop += (t2 - t1);

  } else if (hdf5_source && opt_use_element_cache) {
    // Path 2: Lazy per-site lookup with element cache (no nodal precomputation)
    double t0 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      T[i] = hdf5_source->get_temperature_at_site(i, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  } else if (hdf5_source) {
    // Path 1: XYZ-based lookup (no element cache)
    double t0 = MPI_Wtime();
    const double src_dx = hdf5_source->get_dx();
    for (int i = 0; i < nlocal; i++) {
      double x_physical = xyz[i][0] * src_dx;
      double y_physical = xyz[i][1] * src_dx;
      double z_physical = xyz[i][2] * src_dx;
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  // -------------------------------------------------------------------
  // Rosenthal path: analytical evaluation in pool-local frame.
  // -------------------------------------------------------------------
  } else if (ros_source) {
    if (!laser_path_set) {
      error->all(FLERR,"Rosenthal source selected but no laser_path defined");
    }

    double t0 = MPI_Wtime();

    // Sync the laser pose to simulation_time, advancing in small dt-sized
    // chunks (move(dt) does not carry remainder across path boundaries).
    if (scan_layer_active) {
      double advance = simulation_time - scan_layer_time;
      const double step = (host_->dt > 0.0) ? host_->dt : 1.0e-6;
      while (advance > step) {
        if (!scan_layer.move(step)) { scan_layer_active = false; break; }
        advance -= step;
      }
      if (scan_layer_active && advance > 0.0) {
        if (!scan_layer.move(advance)) scan_layer_active = false;
      }
    }
    scan_layer_time = simulation_time;

    const double v_laser = scan_layer.get_speed();
    // Use the source's own preheat/ambient (Rosenthal T0), not the app's
    // t_room -- they may differ.
    const double T0 = ros_source->get_ambient_temperature();
    for (int i = 0; i < nlocal; i++) {
      double phys[3] = { xyz[i][0]*dx, xyz[i][1]*dx, xyz[i][2]*dx };
      RASTER::Point loc =
        scan_layer.compute_position_relative_to_pool(phys, scan_layer_z);
      T[i] = scan_layer_active
               ? ros_source->rosenthal_pointwise(loc[0], loc[1], loc[2], v_laser, T0)
               : T0;
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  // -------------------------------------------------------------------
  // Generic xyz fallback for any other future source type.
  // -------------------------------------------------------------------
  } else {
    double t0 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      double x_physical = xyz[i][0] * dx;
      double y_physical = xyz[i][1] * dx;
      double z_physical = xyz[i][2] * dx;
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);
  }

  // Optional Gaussian smoothing of the solidification band.
  if (temperature_smooth_enabled) apply_smoothing();

  timer->stamp(TIME_APP);
}

/* ----------------------------------------------------------------------
   One step of thermal orchestration: update T, run Moser pause/reset and
   pause_below dynamic-fire logic, then the fast-forward predictor. Returns
   the decision the app applies (advance time / activate powder / end run).
------------------------------------------------------------------------- */

ThermalManager::ThermalStep ThermalManager::advance(double current_time, double stoptime)
{
  ThermalStep step{current_time, false, false};

  // Update temperature first to get current temperatures
  update_temperatures(current_time);

  // Check for active sites using temperature source's threshold. HDF5
  // sources expose their own threshold; Rosenthal/Moser use liquidus tl
  // (or pause_below override).
  double fast_forward_threshold = host_->t_room; // Default to t_room

  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();
  RosenthalTemperatureSource* ros_source = hdf5_source ? nullptr : rosenthal_source();
  MoserTemperatureSource* moser_source =
    (hdf5_source || ros_source) ? nullptr : this->moser_source();

  // Moser pause + reset_temperature: time-gated dynamic fire (unconditional
  // on time, not gated on cool-state, so it lives outside the fast-forward
  // block).
  if (moser_source &&
      laser_pause_constant > 0.0 &&
      laser_reset_temperature > 0.0 &&
      pending_moser_passes > 0) {
    const double next_fire =
      moser_source->get_last_pass_end_time() + laser_pause_constant;
    if (current_time >= next_fire) {
      moser_source->reset_history_and_start(current_time, laser_reset_temperature);
      --pending_moser_passes;
      if (domain->me == 0) {
        std::cout << "pause+reset: gap of " << laser_pause_constant
                  << " s elapsed at t=" << current_time
                  << " s; firing next Moser pass (reset to "
                  << laser_reset_temperature << " K, "
                  << pending_moser_passes << " more queued)" << std::endl;
      }
      update_temperatures(current_time);
    }
  }

  if (hdf5_source) {
    fast_forward_threshold = hdf5_source->get_fast_forward_threshold();
  } else if (ros_source || moser_source) {
    // `laser_path ... pause_below <Tk>` overrides the default liquidus-based
    // threshold.
    fast_forward_threshold = (laser_pause_below > 0.0) ? laser_pause_below : host_->tl;
  }

  int t_active = 0;
  for (int i = 0; i < host_->nlocal; i++) {
    if (host_->T[i] > fast_forward_threshold) t_active = 1;
  }

  // Use MPI_Allreduce to check if t_active is 0 on all processors
  int global_t_active;
  MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, world);

  // If no active sites, fast forward.
  if (global_t_active == 0 && current_time > 1e-6) {
    // Moser pause_below: fire a pending pass once the global peak temperature
    // first drops below threshold AFTER the current pass has finished.
    bool dynamic_fire = false;
    if (moser_source && laser_pause_below > 0.0 && pending_moser_passes > 0) {
      if (current_time >= moser_source->get_last_pass_end_time()) {
        if (laser_reset_temperature > 0.0) {
          moser_source->reset_history_and_start(current_time, laser_reset_temperature);
        } else {
          moser_source->append_pass(current_time);
        }
        --pending_moser_passes;
        dynamic_fire = true;
        if (domain->me == 0) {
          std::cout << "pause_below: max(T) < " << fast_forward_threshold
                    << " K reached at t=" << current_time
                    << " s; firing next Moser pass (";
          if (laser_reset_temperature > 0.0)
            std::cout << "reset to " << laser_reset_temperature << " K, ";
          std::cout << pending_moser_passes << " more queued)" << std::endl;
        }
        // Refresh temperatures to reflect the just-fired pass.
        update_temperatures(current_time);
      }
    }
    // pause_below: once the FINAL pass has finished emitting and the global
    // peak temperature has dropped below threshold, end the run early by
    // advancing time to stoptime.
    if (!dynamic_fire && moser_source && laser_pause_below > 0.0 &&
        pending_moser_passes == 0 &&
        current_time >= moser_source->get_last_pass_end_time() &&
        current_time < stoptime) {
      if (domain->me == 0) {
        std::cout << "pause_below: final pass complete and max(T) < "
                  << fast_forward_threshold << " K at t=" << current_time
                  << " s; advancing to stoptime=" << stoptime
                  << " s for clean exit." << std::endl;
      }
      step.new_time = stoptime;
      step.end_run = true;
      return step;
    }
    if (!dynamic_fire) {
      double local_next_time;
      if (ros_source) {
        local_next_time = rosenthal_next_active_time(current_time, fast_forward_threshold);
      } else if (moser_source) {
        local_next_time = moser_next_active_time(current_time, fast_forward_threshold);
      } else {
        local_next_time = temperature_source->get_next_time_with_temperature(current_time, fast_forward_threshold);
      }

      // Earliest wake-up across the entire MPI communicator is the single
      // time we may safely advance to.
      double next_thermal_time;
      MPI_Allreduce(&local_next_time, &next_thermal_time, 1, MPI_DOUBLE,
                    MPI_MIN, world);

      if (next_thermal_time > current_time && next_thermal_time < std::numeric_limits<double>::max()) {
        // Fast-forward to whichever comes first: next thermal activity or run end.
        double target_time = std::min(next_thermal_time, stoptime);
        double time_skip = target_time - current_time;

        if (time_skip > 1e-6) {  // Only skip if meaningful time difference
          if (domain->me == 0) {
            if (target_time < next_thermal_time) {
              std::cout << "Fast-forward: Skipping " << time_skip << " seconds from time "
                        << current_time << " to run end " << target_time
                        << " (next thermal activity at " << next_thermal_time << ")" << std::endl;
            } else {
              std::cout << "Fast-forward: Skipping " << time_skip << " seconds from time "
                        << current_time << " to " << target_time
                        << " (all temperatures < " << fast_forward_threshold << "K)" << std::endl;
            }
          }

          step.new_time = target_time;

          // Update temperatures at new time
          update_temperatures(target_time);

          // Activate powder after significant time skip (new layer deposition)
          if (time_skip > 1.0) {
            step.activate_powder = true;
          }
        }
      }
    }
  }

  return step;
}

/* ----------------------------------------------------------------------
   Build a RosenthalDevicePlan for the current step: advance the laser scan
   path to simulation_time on the host and snapshot the laser pose + material
   parameters so a device kernel can evaluate the temperature per site. See
   rosenthal_device.h. The scan-path advance mirrors update_temperatures()'s
   Rosenthal branch exactly, so the device fill matches the host fill.
------------------------------------------------------------------------- */

RosenthalDevicePlan ThermalManager::rosenthal_device_plan(double simulation_time)
{
  RosenthalDevicePlan plan;

  RosenthalTemperatureSource* ros =
    rosenthal_source();
  if (!ros || !laser_path_set) return plan;  // plan.valid stays false

  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path.
  if (temperature_smooth_enabled) return plan;

  // Material parameters; false (e.g. KEYHOLE) -> host path.
  if (!ros->fill_device_plan(plan)) { plan.valid = false; return plan; }

  // Sync the laser pose to simulation_time in dt-sized chunks (move(dt) does
  // not carry remainder across path boundaries) -- identical to the host loop.
  if (scan_layer_active) {
    double advance = simulation_time - scan_layer_time;
    const double step = (host_->dt > 0.0) ? host_->dt : 1.0e-6;
    while (advance > step) {
      if (!scan_layer.move(step)) { scan_layer_active = false; break; }
      advance -= step;
    }
    if (scan_layer_active && advance > 0.0) {
      if (!scan_layer.move(advance)) scan_layer_active = false;
    }
  }
  scan_layer_time = simulation_time;

  const RASTER::Point pos = scan_layer.get_position();
  const RASTER::Point dir = scan_layer.get_unit_dir();
  plan.px = pos[0];
  plan.py = pos[1];
  plan.layer_z = scan_layer_z;
  plan.ux = dir[0];
  plan.uy = dir[1];
  plan.v = scan_layer.get_speed();
  plan.scan_active = scan_layer_active;
  plan.T0 = ros->get_ambient_temperature();
  plan.valid = true;
  return plan;
}

/* ----------------------------------------------------------------------
   Fast-forward decision for the Rosenthal device path. Same logic as the
   Rosenthal branch of advance(), but the active-site test uses the device-
   computed global max temperature instead of a host T scan, and temperatures
   are NOT refilled here (the caller refills its device view at new_time).
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::rosenthal_fast_forward(double current_time, double stoptime,
                                       double global_max_T)
{
  ThermalStep step;
  step.new_time = current_time;
  step.activate_powder = false;
  step.end_run = false;

  RosenthalTemperatureSource* ros =
    rosenthal_source();
  if (!ros) return step;

  // `laser_path ... pause_below <Tk>` overrides the default liquidus threshold.
  const double fast_forward_threshold =
    (laser_pause_below > 0.0) ? laser_pause_below : host_->tl;

  // global_max_T is already reduced across ranks, so this test is consistent
  // on every rank -> all ranks take the same branch (keeps the MIN reduce
  // below collective-safe).
  const bool any_active = (global_max_T > fast_forward_threshold);

  if (!any_active && current_time > 1e-6) {
    const double local_next =
      rosenthal_next_active_time(current_time, fast_forward_threshold);
    double next_thermal_time;
    MPI_Allreduce(&local_next, &next_thermal_time, 1, MPI_DOUBLE, MPI_MIN, world);

    if (next_thermal_time > current_time &&
        next_thermal_time < std::numeric_limits<double>::max()) {
      const double target_time = std::min(next_thermal_time, stoptime);
      const double time_skip = target_time - current_time;
      if (time_skip > 1e-6) {
        if (domain->me == 0) {
          std::cout << "Fast-forward (device): skipping " << time_skip
                    << " s from t=" << current_time << " to " << target_time
                    << " (all T < " << fast_forward_threshold << " K)"
                    << std::endl;
        }
        step.new_time = target_time;
        if (time_skip > 1.0) step.activate_powder = true;
      }
    }
  }
  return step;
}

/* ----------------------------------------------------------------------
   Device (Kokkos) Moser path: build a MoserDevicePlan for the current step.
   Unlike Rosenthal there is no scan-path pose to advance (the Moser source
   integrates over its own absolute-time path history); we just let the source
   do any per-step bookkeeping and snapshot its scalar parameters. Returns an
   invalid plan (host path) for non-Moser sources, KEYHOLE / fluctuations, or
   when a dynamic inter-pass pause/reset is configured.
------------------------------------------------------------------------- */

MoserDevicePlan ThermalManager::moser_device_plan(double simulation_time)
{
  MoserDevicePlan plan;

  MoserTemperatureSource* moser =
    moser_source();
  if (!moser) return plan;  // plan.valid stays false

  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path. (Dynamic
  // inter-pass pause/reset is handled on the device via moser_advance_device.)
  if (temperature_smooth_enabled) return plan;

  if (!moser->fill_device_plan(plan)) { plan.valid = false; return plan; }
  plan.time = simulation_time;
  return plan;
}

/* ----------------------------------------------------------------------
   Device Moser driver. Mirrors the Moser branch of advance() -- time-gated
   pause+reset firing, pause_below dynamic firing / clean-exit, and the
   fast-forward predictor -- but replaces the host per-site fill +
   host-temperature scan with the registered device filler (which fills k_T and
   returns the global max). The pause/reset firings (reset_history_and_start /
   append_pass) rebuild the scan from temperature feedback on the host; each is
   followed by a device refill. The static (no-pause) case falls straight
   through to the predictor, so this also subsumes the plain fast-forward path.
   The global max is already MPI-reduced, so the active-site test is consistent
   on every rank and all ranks make the same collective calls.
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::moser_advance_device(double current_time, double stoptime)
{
  ThermalStep step;
  step.new_time = current_time;
  step.activate_powder = false;
  step.end_run = false;

  MoserTemperatureSource* moser =
    moser_source();
  if (!moser || !moser_device_filler_) return step;

  // Fill the device field at current_time and get the global peak temperature.
  double gmax = moser_device_filler_(current_time);

  // Moser pause + reset_temperature: time-gated dynamic fire (fires on elapsed
  // time since the last pass end, independent of temperature).
  if (laser_pause_constant > 0.0 && laser_reset_temperature > 0.0 &&
      pending_moser_passes > 0) {
    const double next_fire =
      moser->get_last_pass_end_time() + laser_pause_constant;
    if (current_time >= next_fire) {
      moser->reset_history_and_start(current_time, laser_reset_temperature);
      --pending_moser_passes;
      if (domain->me == 0) {
        std::cout << "pause+reset: gap of " << laser_pause_constant
                  << " s elapsed at t=" << current_time
                  << " s; firing next Moser pass (reset to "
                  << laser_reset_temperature << " K, "
                  << pending_moser_passes << " more queued)" << std::endl;
      }
      gmax = moser_device_filler_(current_time);   // scan changed -> refill
    }
  }

  const double fast_forward_threshold =
    (laser_pause_below > 0.0) ? laser_pause_below : host_->tl;

  if (gmax <= fast_forward_threshold && current_time > 1e-6) {
    bool dynamic_fire = false;

    // pause_below: fire a pending pass once the peak temperature drops below
    // threshold AFTER the current pass has finished emitting.
    if (laser_pause_below > 0.0 && pending_moser_passes > 0) {
      if (current_time >= moser->get_last_pass_end_time()) {
        if (laser_reset_temperature > 0.0)
          moser->reset_history_and_start(current_time, laser_reset_temperature);
        else
          moser->append_pass(current_time);
        --pending_moser_passes;
        dynamic_fire = true;
        if (domain->me == 0) {
          std::cout << "pause_below: max(T) < " << fast_forward_threshold
                    << " K reached at t=" << current_time
                    << " s; firing next Moser pass (";
          if (laser_reset_temperature > 0.0)
            std::cout << "reset to " << laser_reset_temperature << " K, ";
          std::cout << pending_moser_passes << " more queued)" << std::endl;
        }
        gmax = moser_device_filler_(current_time);   // scan changed -> refill
      }
    }

    // pause_below: once the FINAL pass has finished and the peak has dropped
    // below threshold, advance to stoptime for a clean early exit.
    if (!dynamic_fire && laser_pause_below > 0.0 &&
        pending_moser_passes == 0 &&
        current_time >= moser->get_last_pass_end_time() &&
        current_time < stoptime) {
      if (domain->me == 0) {
        std::cout << "pause_below: final pass complete and max(T) < "
                  << fast_forward_threshold << " K at t=" << current_time
                  << " s; advancing to stoptime=" << stoptime
                  << " s for clean exit." << std::endl;
      }
      step.new_time = stoptime;
      step.end_run = true;
      return step;
    }

    // Fast-forward predict-skip (also the sole behavior in the static case).
    if (!dynamic_fire) {
      const double local_next =
        moser_next_active_time(current_time, fast_forward_threshold);
      double next_thermal_time;
      MPI_Allreduce(&local_next, &next_thermal_time, 1, MPI_DOUBLE, MPI_MIN, world);
      if (next_thermal_time > current_time &&
          next_thermal_time < std::numeric_limits<double>::max()) {
        const double target_time = std::min(next_thermal_time, stoptime);
        const double time_skip = target_time - current_time;
        if (time_skip > 1e-6) {
          if (domain->me == 0) {
            std::cout << "Fast-forward (device Moser): skipping " << time_skip
                      << " s from t=" << current_time << " to " << target_time
                      << " (all T < " << fast_forward_threshold << " K)"
                      << std::endl;
          }
          step.new_time = target_time;
          moser_device_filler_(target_time);   // refill at the advanced time
          if (time_skip > 1.0) step.activate_powder = true;
        }
      }
    }
  }
  return step;
}

/* ----------------------------------------------------------------------
   Device (Kokkos) HDF5 unstructured-mesh path. The expensive host work --
   HDF5 reads, layer load, per-site element search -- stays on the host; only
   the per-site barycentric interpolation moves to the device. See hdf5_device.h.
------------------------------------------------------------------------- */

bool ThermalManager::hdf5_device_eligible() const
{
  if (!temperature_source) return false;
  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path.
  if (temperature_smooth_enabled) return false;
  return dynamic_cast<HDF5UnstructuredTemperatureSource*>(
             temperature_source.get()) != nullptr;
}

HDF5DevicePlan ThermalManager::hdf5_device_plan(double simulation_time)
{
  HDF5DevicePlan plan;

  HDF5UnstructuredTemperatureSource* hdf5 =
    hdf5_source();
  if (!hdf5 || temperature_smooth_enabled) return plan;  // plan.valid stays false

  // Per-step host prep: advance source state, load the layer if needed, rebuild
  // the per-site element cache at a layer boundary (bumps cache_generation), and
  // precompute the nodal-temperature slice (time interpolation + optional nodal
  // mesh smoothing). This is exactly the host fast path (update_temperatures
  // Path 3) minus the per-site T loop -- that loop is what moves to the device.
  hdf5->prepare_for_timestep(simulation_time);

  plan.version = hdf5->cache_generation();
  plan.nsites = hdf5->num_cache_sites();   // nall (owned + ghost); fills positionally
  plan.n_nodes = hdf5->num_loaded_nodes();
  plan.default_temp = hdf5->get_default_temp();
  plan.sim_time = simulation_time;          // device-side time interpolation
  plan.valid = true;
  return plan;
}

RosenthalTemperatureSource *ThermalManager::rosenthal_source() const
{
  return dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());
}

MoserTemperatureSource *ThermalManager::moser_source() const
{
  return dynamic_cast<MoserTemperatureSource*>(temperature_source.get());
}

HDF5UnstructuredTemperatureSource *ThermalManager::hdf5_source() const
{
  return dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
}

/* ----------------------------------------------------------------------
   Device HDF5 driver. Mirrors the HDF5 branch of advance() -- the global
   active-site reduction (here via the device-computed global max) and the
   generic fast-forward predictor (get_next_time_with_temperature) -- but
   replaces the host per-site fill + host-temperature scan with the registered
   device filler. HDF5 has no pause/reset, so this is fill + predict-skip. The
   global max is already MPI-reduced, so the active-site test is consistent on
   every rank and all ranks make the same collective calls.
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::hdf5_advance_device(double current_time, double stoptime)
{
  ThermalStep step;
  step.new_time = current_time;
  step.activate_powder = false;
  step.end_run = false;

  HDF5UnstructuredTemperatureSource* hdf5 =
    hdf5_source();
  if (!hdf5 || !hdf5_device_filler_) return step;

  // Fill the device field at current_time and get the global peak temperature.
  double gmax = hdf5_device_filler_(current_time);

  const double fast_forward_threshold = hdf5->get_fast_forward_threshold();

  if (gmax <= fast_forward_threshold && current_time > 1e-6) {
    const double local_next =
      hdf5->get_next_time_with_temperature(current_time, fast_forward_threshold);
    double next_thermal_time;
    MPI_Allreduce(&local_next, &next_thermal_time, 1, MPI_DOUBLE, MPI_MIN, world);
    if (next_thermal_time > current_time &&
        next_thermal_time < std::numeric_limits<double>::max()) {
      const double target_time = std::min(next_thermal_time, stoptime);
      const double time_skip = target_time - current_time;
      if (time_skip > 1e-6) {
        if (domain->me == 0) {
          std::cout << "Fast-forward (device HDF5): skipping " << time_skip
                    << " s from t=" << current_time << " to " << target_time
                    << " (all T < " << fast_forward_threshold << " K)"
                    << std::endl;
        }
        step.new_time = target_time;
        hdf5_device_filler_(target_time);   // refill at the advanced time
        if (time_skip > 1.0) step.activate_powder = true;
      }
    }
  }
  return step;
}

/* ----------------------------------------------------------------------
   Run the isotherm-compactness diagnostic on the configured cadence.
------------------------------------------------------------------------- */

void ThermalManager::maybe_run_diagnostics(int step_count)
{
  if (temperature_smooth_enabled && smooth_diag_interval > 0 &&
      step_count % smooth_diag_interval == 0) {
    compute_diagnostics();
  }
}

/* ----------------------------------------------------------------------
   Fast-forward predictor for the Rosenthal source. Walks a copy of
   scan_layer forward and returns the earliest time the conservative upper
   bound on the Rosenthal temperature over the local domain reaches
   threshold_temp.
------------------------------------------------------------------------- */

double ThermalManager::rosenthal_next_active_time(double current_time, double threshold_temp)
{
  if (!laser_path_set || !scan_layer_active)
    return std::numeric_limits<double>::max();

  auto* ros = rosenthal_source();
  if (!ros) return std::numeric_limits<double>::max();

  double lo[3], hi[3];
  if (!local_bbox(lo, hi)) return std::numeric_limits<double>::max();

  const double T0 = ros->get_ambient_temperature();
  // Cheap early-out: if even the unmodulated peak at r_min can't reach the
  // threshold, no laser position ever will.
  {
    const double T_max_possible = ros->rosenthal_peak_at_distance(0.0, T0);
    if (T_max_possible < threshold_temp) {
      return std::numeric_limits<double>::max();
    }
  }

  // Walk a copy of the layer forward in time using the search window.
  RASTER::Layer probe = scan_layer;
  const double ddt = (fast_forward_search_window > 0.0)
                       ? fast_forward_search_window : 0.01;
  // Cap the look-ahead so we don't spin forever on a never-active path.
  const double max_lookahead = 3600.0; // 1 hour of sim time
  double t = current_time;
  const double t_stop = current_time + max_lookahead;

  while (t < t_stop) {
    // Closest point on the local box to the current laser position.
    const RASTER::Point laser = probe.get_position();
    const double dxr = std::clamp(laser[0], lo[0], hi[0]) - laser[0];
    const double dyr = std::clamp(laser[1], lo[1], hi[1]) - laser[1];
    const double dzr = std::clamp(scan_layer_z, lo[2], hi[2]) - scan_layer_z;
    const double R_min_box = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);

    const double T_upper = ros->rosenthal_peak_at_distance(R_min_box, T0);
    if (T_upper >= threshold_temp) return t;

    // Advance the probe by ddt; stop if path is exhausted.
    if (!probe.move(ddt)) return std::numeric_limits<double>::max();
    t += ddt;
  }
  return std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   Fast-forward predictor for the Moser source. Walks each remaining active
   scan interval forward, evaluating the Moser integrator at the closest
   bounding-box point, and returns the first time it reaches threshold_temp.
------------------------------------------------------------------------- */

double ThermalManager::moser_next_active_time(double current_time, double threshold_temp)
{
  if (!laser_path_set) return std::numeric_limits<double>::max();

  auto* moser = moser_source();
  if (!moser || !moser->has_scan()) return std::numeric_limits<double>::max();

  double lo[3], hi[3];
  if (!local_bbox(lo, hi)) return std::numeric_limits<double>::max();

  const double t_origin    = moser->get_scan_t_origin();
  const double pass_dur    = moser->get_scan_pass_duration();
  const double pause_gap   = moser->get_scan_pause();
  const int    n_reps      = moser->get_scan_repeats();
  const double x0          = moser->get_scan_x0();
  const double y0          = moser->get_scan_y0();
  const double x1          = moser->get_scan_x1();
  const double y1          = moser->get_scan_y1();
  const double laser_z     = moser->get_scan_laser_plane_z();

  const double ddt = (fast_forward_search_window > 0.0)
                       ? fast_forward_search_window : 0.01;
  for (int r = 0; r < n_reps; ++r) {
    const double t_r_start = t_origin + r * pass_dur + r * pause_gap;
    const double t_r_end   = t_r_start + pass_dur;
    if (t_r_end <= current_time) continue;  // pass already completed

    double t = std::max(current_time, t_r_start);
    while (t <= t_r_end) {
      // Lerp laser (x,y) along this pass.
      const double s = (t - t_r_start) / pass_dur;
      // Closest point on local box to laser.
      const double cx = std::clamp(x0 + s * (x1 - x0), lo[0], hi[0]);
      const double cy = std::clamp(y0 + s * (y1 - y0), lo[1], hi[1]);
      const double cz = std::clamp(laser_z, lo[2], hi[2]);
      const double T_probe = moser->get_temperature_at_xyz_and_time(cx, cy, cz, t);
      if (T_probe >= threshold_temp) return t;
      t += ddt;
    }
  }
  return std::numeric_limits<double>::max();
}

bool ThermalManager::local_bbox(double lo[3], double hi[3]) const
{
  const int nlocal = host_->nlocal;
  if (nlocal == 0) return false;
  double **xyz = host_->xyz;
  const double dx = host_->dx;
  for (int d = 0; d < 3; ++d) lo[d] = hi[d] = xyz[0][d] * dx;
  for (int i = 1; i < nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      const double v = xyz[i][d] * dx;
      lo[d] = std::min(lo[d], v);
      hi[d] = std::max(hi[d], v);
    }
  return true;
}

/* ----------------------------------------------------------------------
   Gaussian smoothing of T[] inside the user-specified solidification
   window. Called from update_temperatures() when enabled.
------------------------------------------------------------------------- */

void ThermalManager::apply_smoothing()
{
  double *T = host_->T;
  double **xyz = host_->xyz;
  int *numneigh = host_->numneigh;
  int **neighbor = host_->neighbor;
  const int nlocal = host_->nlocal;

  const double two_sxy2 = 2.0 * smooth_sigma_xy * smooth_sigma_xy;
  const double two_sz2  = 2.0 * smooth_sigma_z  * smooth_sigma_z;
  const double lo_outer = smooth_tmin - smooth_guard;
  const double hi_outer = smooth_tmax + smooth_guard;
  const double ramp_lo = (smooth_guard > 0.0) ? smooth_guard : 1.0;
  const double ramp_hi = (smooth_guard > 0.0) ? smooth_guard : 1.0;

  if ((int)smooth_buffer.size() < nlocal) smooth_buffer.resize(nlocal);

  // Precompute Gaussian weights per 26-neighbor slot (uniform SC_26N lattice).
  constexpr int MAX_NBR = 26;
  double w_slot[MAX_NBR];
  bool have_weight_lut = false;
  for (int ref = 0; ref < nlocal && !have_weight_lut; ref++) {
    if (numneigh[ref] < MAX_NBR) continue;
    bool all_valid = true;
    for (int j = 0; j < MAX_NBR; j++) {
      if (neighbor[ref][j] < 0) { all_valid = false; break; }
    }
    if (!all_valid) continue;
    const double xr = xyz[ref][0];
    const double yr = xyz[ref][1];
    const double zr = xyz[ref][2];
    for (int j = 0; j < MAX_NBR; j++) {
      const int nj = neighbor[ref][j];
      const double dxn = xyz[nj][0] - xr;
      const double dyn = xyz[nj][1] - yr;
      const double dzn = xyz[nj][2] - zr;
      const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
      w_slot[j] = exp(-expo);
    }
    have_weight_lut = true;
  }

  for (int pass = 0; pass < smooth_passes; pass++) {
    // Ghosts need current T before each pass.
    host_->comm->all_selective(host_->ghost_iindices, host_->nghost_iarray,
                              host_->ghost_dindices, host_->nghost_darray);

    for (int i = 0; i < nlocal; i++) {
      const double Ti = T[i];
      if (Ti < lo_outer || Ti > hi_outer) {
        smooth_buffer[i] = Ti;
        continue;
      }

      double wsum = 1.0;   // self
      double Tsum = Ti;    // self-contribution

      const int nn = numneigh[i];
      if (have_weight_lut) {
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          // Skip neighbors whose T is outside the smoothing window.
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double w = w_slot[j];
          wsum += w;
          Tsum += w * Tj;
        }
      } else {
        const double xi = xyz[i][0];
        const double yi = xyz[i][1];
        const double zi = xyz[i][2];
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double dxn = xyz[nj][0] - xi;
          const double dyn = xyz[nj][1] - yi;
          const double dzn = xyz[nj][2] - zi;
          const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
          const double w = exp(-expo);
          wsum += w;
          Tsum += w * Tj;
        }
      }
      const double Tavg = Tsum / wsum;

      // Guard-band taper so the blend factor ramps to zero at the outer edges.
      double aeff = smooth_alpha;
      if (Ti < smooth_tmin) {
        aeff *= (Ti - lo_outer) / ramp_lo;
      } else if (Ti > smooth_tmax) {
        aeff *= (hi_outer - Ti) / ramp_hi;
      }
      if (aeff < 0.0) aeff = 0.0;

      smooth_buffer[i] = (1.0 - aeff) * Ti + aeff * Tavg;
    }

    // Commit this pass before the next one.
    for (int i = 0; i < nlocal; i++) T[i] = smooth_buffer[i];
  }

  // Leave ghosts consistent with the final committed T.
  host_->comm->all_selective(host_->ghost_iindices, host_->nghost_iarray,
                            host_->ghost_dindices, host_->nghost_darray);
}

/* ----------------------------------------------------------------------
   In-situ smoothing diagnostic: isotherm compactness of the T >= liquidus
   region. Assumes T ghosts are current (caller must ensure this).
------------------------------------------------------------------------- */

void ThermalManager::compute_diagnostics()
{
  double *T = host_->T;
  int *numneigh = host_->numneigh;
  int **neighbor = host_->neighbor;
  const int nlocal = host_->nlocal;
  const double tl = host_->tl;

  bigint V_local = 0, S_local = 0;
  double sumT_local = 0.0;
  double maxT_local = -1.0e30;

  for (int i = 0; i < nlocal; i++) {
    const double Ti = T[i];
    if (Ti < tl) continue;

    V_local++;
    sumT_local += Ti;
    if (Ti > maxT_local) maxT_local = Ti;

    const int nn = numneigh[i];
    for (int j = 0; j < nn; j++) {
      const int nj = neighbor[i][j];
      if (nj < 0) continue;
      if (T[nj] < tl) { S_local++; break; }
    }
  }

  bigint V_global = 0, S_global = 0;
  double sumT_global = 0.0, maxT_global = 0.0;
  MPI_Allreduce(&V_local,    &V_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&S_local,    &S_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&sumT_local, &sumT_global, 1, MPI_DOUBLE,     MPI_SUM, world);
  MPI_Allreduce(&maxT_local, &maxT_global, 1, MPI_DOUBLE,     MPI_MAX, world);

  if (domain->me == 0) {
    const double ideal = cbrt(36.0 * MY_PI);  // ~= 4.8360
    fprintf(screen, "\n=== Smoothing diagnostic (T >= liquidus=%.1f K) ===\n", tl);
    if (V_global > 0) {
      const double V = static_cast<double>(V_global);
      const double ratio = static_cast<double>(S_global) / cbrt(V * V);  // V^(2/3)
      const double meanT = sumT_global / V;
      fprintf(screen,
        "  Pool volume:       " BIGINT_FORMAT " sites\n"
        "  Pool surface:      " BIGINT_FORMAT " sites\n"
        "  S/V^(2/3):         %.3f  (ideal 3D sphere = %.3f)\n"
        "  Pool mean T:       %.2f K   max T: %.2f K\n",
        V_global, S_global, ratio, ideal, meanT, maxT_global);
    } else {
      fprintf(screen, "  No sites above liquidus this step.\n");
    }
  }
}
