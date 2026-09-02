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
#include "thermal_host.h"
#include "rosenthal_device.h"
#include "moser_device.h"
#include "hdf5_device.h"
#include "am_raster.h"
#include <functional>
#include <memory>
#include <vector>

namespace SPPARKS_NS {

class RosenthalTemperatureSource;
class MoserTemperatureSource;
class HDF5UnstructuredTemperatureSource;

/* ----------------------------------------------------------------------
   ThermalManager owns the temperature-I/O orchestration that was formerly
   embedded in AppAdditiveTexture: temperature-source command parsing, the
   per-step temperature update dispatch, the Rosenthal/Moser fast-forward
   predictors, multi-pass laser pause/reset scheduling, and the optional
   solidification-band temperature smoothing + diagnostics.

   It is composed into an additive/texture app (the app holds a
   std::unique_ptr<ThermalManager>) and reaches the app's lattice/state
   arrays (T, xyz, neighbor lists, ghost-comm config, alloy thresholds)
   through a ThermalHost snapshot the app populates each step. Using the
   ThermalHost struct rather than a typed back-pointer lets both the CPU app
   (AppAdditiveTexture) and the Kokkos app (AppAdditiveTextureKokkos) -- which
   do not share an additive-app base class -- reuse this manager. The manager
   owns the temperature_source itself.
------------------------------------------------------------------------- */

class ThermalManager : protected Pointers {
 public:
  ThermalManager(class SPPARKS *, ThermalHost *host);
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

  // True once a temperature source has been configured via
  // setup_temperature_source.
  bool enabled() const { return (bool)temperature_source; }

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

  // ---- device (Kokkos) Rosenthal path ----
  // Advance the laser scan path to simulation_time on the host (cheap) and
  // return a RosenthalDevicePlan the caller can use to evaluate the Rosenthal
  // temperature on a device, filling its own T view directly. plan.valid is
  // false when the active source is not a device-eligible Rosenthal (i.e. not
  // Rosenthal, or KEYHOLE mode) -- the caller must then use the host
  // update_temperatures() + sync path. This advances the same scan_layer state
  // update_temperatures() would, so for a Rosenthal device step the caller must
  // NOT also call update_temperatures() (that would double-advance the path).
  RosenthalDevicePlan rosenthal_device_plan(double simulation_time);

  // Fast-forward decision for the Rosenthal device path. Mirrors advance()'s
  // Rosenthal branch but takes the already-computed global max temperature
  // (from a device reduction) instead of scanning host T, and does NOT refill
  // temperatures (the caller refills its device view at step.new_time when a
  // skip is returned). Returns {new_time, activate_powder, end_run}.
  ThermalStep rosenthal_fast_forward(double current_time, double stoptime,
                                     double global_max_T);

  // ---- device (Kokkos) Moser path ----
  // Mirror of the Rosenthal device hooks for the time-domain Green's-function
  // (Moser) source. moser_device_plan() returns a MoserDevicePlan POD; the
  // variable-length scan segment arrays are fetched separately via
  // moser_copy_device_data() (only re-copied by the caller when plan.version
  // changes). plan.valid is false only for a non-Moser source, configured
  // temperature smoothing, or an unbuilt/>2-lobe scan -- KEYHOLE, fluctuations,
  // and dynamic inter-pass pause are all handled on the device.
  MoserDevicePlan moser_device_plan(double simulation_time);
  // Register the device temperature filler used by moser_advance_device(): given
  // a simulation time, it fills the device temperature field (k_T) at that time
  // and returns the global (MPI-reduced) max temperature. Set once by the app.
  void set_moser_device_filler(std::function<double(double)> f) {
    moser_device_filler_ = std::move(f);
  }

  // Device Moser driver. Runs the full per-pass inter-pass pause/reset firing
  // and fast-forward control flow on the host (those rebuild the scan from
  // temperature feedback), but delegates every per-site temperature fill to the
  // registered device filler and uses the device-computed global max for the
  // active-site test -- so the expensive Green's-function integral always runs
  // on the device. Subsumes the static fast-forward path. The filler leaves the
  // device field filled at step.new_time on return. Requires the filler to be
  // set.
  ThermalStep moser_advance_device(double current_time, double stoptime);

  // ---- device (Kokkos) HDF5 unstructured-mesh path ----
  // Mirror of the Moser device hooks for the HDF5 source. The expensive work
  // (HDF5 reads, layer load, per-site element search) stays on the host; only
  // the per-site barycentric interpolation moves to the device, so the device
  // field matches the host fast path bit-for-bit.
  //   hdf5_device_eligible() - cheap host gate: source is HDF5 (incl. the CSR
  //     subclass) and solidification-band temperature_smooth is off (that
  //     smoothing is applied in update_temperatures(), which the device fill
  //     bypasses, so smoothed runs stay on the host path).
  //   hdf5_device_plan()     - runs the per-step host prep (prepare_for_timestep:
  //     layer load + cache rebuild at a layer boundary + nodal-temp precompute,
  //     incl. optional nodal mesh smoothing) and returns the scalar POD. This is
  //     exactly the host fast path minus the per-site T loop.
  // The device caller reads the element cache, nodal slice and time-series
  // CSR straight from hdf5_source().
  bool hdf5_device_eligible() const;
  HDF5DevicePlan hdf5_device_plan(double simulation_time);

  // Typed view of the active source; nullptr when it is another type.
  RosenthalTemperatureSource *rosenthal_source() const;
  MoserTemperatureSource *moser_source() const;
  HDF5UnstructuredTemperatureSource *hdf5_source() const;

  // Register the device temperature filler used by hdf5_advance_device(): given a
  // simulation time it fills k_T[0,nlocal) on the device and returns the global
  // (MPI-reduced) max temperature. Set once by the app.
  void set_hdf5_device_filler(std::function<double(double)> f) {
    hdf5_device_filler_ = std::move(f);
  }

  // Device HDF5 driver. Fills the device field via the registered filler (which
  // returns the global max), then runs the generic fast-forward predictor using
  // the source's threshold + time-query interface. HDF5 has no pause/reset, so
  // this is just fill + predict-skip. The filler leaves the device field filled
  // at step.new_time on return. Requires the filler to be set.
  ThermalStep hdf5_advance_device(double current_time, double stoptime);

  // ---- diagnostics ----
  // Run the diagnostic when step_count lands on the configured cadence.
  // Assumes T ghosts are current (caller syncs before this).
  void maybe_run_diagnostics(int step_count);

  // ---- timing getters for the app's step-100 timing summary ----
  double cache_build_time() const { return g_t_cache_build; }
  double prepare_time()     const { return g_t_temp_prepare; }
  double site_loop_time()   const { return g_t_temp_site_loop; }

 private:
  ThermalHost *host_;   // lattice-state snapshot owned by the app

  // Device temperature filler for moser_advance_device(): fills k_T at a given
  // time and returns the global max temperature (set by the Kokkos app).
  std::function<double(double)> moser_device_filler_;

  // Device temperature filler for hdf5_advance_device(): fills k_T[0,nlocal) at
  // a given time and returns the global max temperature (set by the Kokkos app).
  std::function<double(double)> hdf5_device_filler_;

  // Modular temperature source interface
  std::unique_ptr<TemperatureSource> temperature_source;
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

  // Fast-forward predictors (closest-point bound on local bounding box).
  double rosenthal_next_active_time(double current_time, double threshold_temp);
  double moser_next_active_time(double current_time, double threshold_temp);
  // Physical (meters) bounding box of this rank's owned sites; false if none.
  bool local_bbox(double lo[3], double hi[3]) const;

  // Gaussian smoothing of T[] inside the configured window, and the
  // isotherm-compactness diagnostic.
  void apply_smoothing();
  void compute_diagnostics();
};

}

#endif
