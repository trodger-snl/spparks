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

#ifndef SPK_THERMAL_MANAGER_H
#define SPK_THERMAL_MANAGER_H

#include "pointers.h"
#include "temperature_source.h"
#include "am_raster.h"
#include <memory>
#include <vector>

namespace SPPARKS_NS {

class AppAdditiveTexture;

/* ----------------------------------------------------------------------
   ThermalManager owns the temperature-I/O orchestration that was formerly
   embedded in AppAdditiveTexture: temperature-source command parsing, the
   per-step temperature update dispatch, the Rosenthal/Moser fast-forward
   predictors, multi-pass laser pause/reset scheduling, and the optional
   solidification-band temperature smoothing + diagnostics.

   It is composed into AppAdditiveTexture (the app holds a
   std::unique_ptr<ThermalManager>) and reaches the app's lattice/state
   arrays (T, xyz, neighbor lists, ghost-comm config, alloy thresholds)
   through a typed back-pointer. AppAdditiveTexture declares this class a
   friend so the manager can read/write those members directly; the
   manager owns the temperature_source itself.
------------------------------------------------------------------------- */

class ThermalManager : protected Pointers {
 public:
  ThermalManager(class SPPARKS *, AppAdditiveTexture *app);
  ~ThermalManager() = default;

  // Decision returned by advance() that the app must apply at the top of
  // app_update(). The manager itself never mutates the app's simulation
  // time or activates powder; it reports what should happen so those
  // app-level effects stay in the app.
  struct ThermalStep {
    double new_time;        // time to advance to (== current if no skip)
    bool   activate_powder; // app should activate powder after the skip
    bool   end_run;         // app should set time=new_time and return
  };

  // True once the modular temperature system is enabled (always) and a
  // temperature source has been configured via setup_temperature_source.
  bool enabled() const {
    return use_temperature_source && (bool)temperature_source;
  }
  // Mirrors the historical use_temperature_source flag (always true);
  // used by init-time validation in the app.
  bool uses_temperature_source() const { return use_temperature_source; }

  // ---- input-script command handlers (called from app input_app) ----
  void setup_cmd(int narg, char **arg);
  void laser_path_cmd(int narg, char **arg);
  void laser_fluctuations_cmd(int narg, char **arg);
  void optimization_cmd(int narg, char **arg);
  void smooth_cmd(int narg, char **arg);
  void nodal_smooth_cmd(int narg, char **arg);
  void set_fast_forward_search_window(double w);

  // ---- per-step driver ----
  // Fill app->T from the source at simulation_time (and apply optional
  // solidification-band smoothing when enabled).
  void update_temperatures(double simulation_time);

  // Run one step of thermal orchestration: refresh app->T at current_time,
  // execute the Moser pause+reset / pause_below dynamic-fire logic, then the
  // global active-site reduction and Rosenthal/Moser/generic fast-forward
  // predictor. When a skip occurs, app->T is refreshed at the advanced time
  // before returning. The returned ThermalStep tells the app whether to
  // advance time, activate powder, or end the run.
  ThermalStep advance(double current_time, double stoptime);

  // ---- diagnostics ----
  // Run the diagnostic when step_count lands on the configured cadence.
  // Assumes T ghosts are current (caller syncs before this).
  void maybe_run_diagnostics(int step_count);

  // ---- timing getters for the app's step-100 timing summary ----
  double cache_build_time() const { return g_t_cache_build; }
  double prepare_time()     const { return g_t_temp_prepare; }
  double site_loop_time()   const { return g_t_temp_site_loop; }

 private:
  AppAdditiveTexture *app_;   // owning app (typed back-reference)

  // Modular temperature source interface
  std::unique_ptr<TemperatureSource> temperature_source;
  bool use_temperature_source;        // enable flag (always true)
  double fast_forward_search_window;  // look-ahead for fast-forward (s)

  // Laser scan path state for the analytical Rosenthal source. SI meters,
  // m/s. Not used by HDF5 sources.
  RASTER::Layer scan_layer;
  double scan_layer_z;       // physical Z of the scan plane (meters)
  double scan_layer_time;    // sim time at which scan_layer's pose is current
  bool   scan_layer_active;  // true while the path has remaining motion
  bool   laser_path_set;     // true once laser_path has been parsed

  // Multi-pass inter-pass pause control (laser_path keywords pause /
  // pause_below; mutually exclusive). Time-resolved sources only (Moser).
  double laser_pause_constant;     // [s], 0 = disabled
  double laser_pause_below;        // [K], 0 = disabled
  double laser_reset_temperature;  // [K], 0 = disabled
  int    pending_moser_passes;     // remaining queued Moser passes

  // Temperature optimization flags (HDF5 source performance paths)
  bool opt_use_spatial_grid;
  bool opt_use_element_cache;
  bool opt_use_nodal_precompute;

  // Solidification-band temperature smoothing (opt-in via temperature_smooth)
  bool   temperature_smooth_enabled;
  double smooth_tmin;       // lower edge of full-strength window (K)
  double smooth_tmax;       // upper edge of full-strength window (K)
  double smooth_guard;      // guard-band half-width (K)
  double smooth_sigma_xy;   // Gaussian width in x,y (lattice sites)
  double smooth_sigma_z;    // Gaussian width in z (lattice sites)
  double smooth_alpha;      // blend factor (0=off, 1=replace)
  int    smooth_passes;     // Gaussian passes per step
  int    smooth_diag_interval;  // diagnostic cadence; 0 disables
  std::vector<double> smooth_buffer;  // scratch for double-buffered pass

  // Temperature-update timing breakdown (were file-scope statics in
  // app_additive_texture.cpp; exposed to the app via getters above).
  double g_t_temp_prepare;    // time in prepare_for_timestep()
  double g_t_temp_site_loop;  // time in the per-site temperature loop
  double g_t_cache_build;     // one-time cache build time (per layer)
  bool   g_cache_was_built;   // cache build detected this call

  // Fast-forward predictors (closest-point bound on local bounding box).
  double rosenthal_next_active_time(double current_time, double threshold_temp);
  double moser_next_active_time(double current_time, double threshold_temp);

  // Gaussian smoothing of T[] inside the configured window, and the
  // isotherm-compactness diagnostic.
  void apply_smoothing();
  void compute_diagnostics();
};

}

#endif
