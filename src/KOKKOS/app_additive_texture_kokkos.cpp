/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator

   Website
   https://spparks.github.io/

   See authors
   https://spparks.github.io/authors.html

   Copyright(C) 1999-2025 National Technology & Engineering Solutions
   of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
   NTESS, the U.S. Government retains certain rights in this software.

   This software is distributed under the GNU General Public License.  See
   LICENSE in top-level SPPARKS directory.

   Contributing author: AI-assisted implementation (Phase 2: Temperature Integration)

   Kokkos GPU-accelerated version of additive manufacturing application.
   Extends AppPottsQuaternionKokkos with temperature field integration
   for AM simulations with thermal gradient-driven grain growth.

------------------------------------------------------------------------- */

#include "app_additive_texture_kokkos.h"
#include "app_kokkos.h"
#include "comm_lattice.h"
#include "domain.h"
#include "error.h"
#include "random_park.h"
#include "timer.h"
#include "math_const.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_hdf5_unstructured.h"
#include <cstring>
#include <limits>
#include <random>

using namespace SPPARKS_NS;
using namespace MathConst;

/* ----------------------------------------------------------------------
   Helper to construct parent class args from user input
------------------------------------------------------------------------- */

static char** construct_parent_args(char **arg) {
  static char* parent_args[3];
  static char app_name[] = "potts/quaternion/kk";  // Use Kokkos quaternion parent
  static char crystal_type[] = "cubic";

  parent_args[0] = app_name;
  parent_args[1] = arg[1]; // nspins
  parent_args[2] = crystal_type;

  return parent_args;
}

/* ----------------------------------------------------------------------
   Constructor
------------------------------------------------------------------------- */

AppAdditiveTextureKokkos::AppAdditiveTextureKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternionKokkos(spk, 3, construct_parent_args(arg))
{
  // Validate input arguments
  if (narg == 6)
    error->all(FLERR,"app_style command format changed: remove temperature file path argument and use 'setup_temperature_source' command instead");

  if (narg != 5)
    error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  dx = atof(arg[2]);     // Lattice spacing (meters)
  dt = atof(arg[3]);     // Timestep (seconds)
  nrefine = atoi(arg[4]); // Refinement MC steps after solidification

  // Configure array sizes (inherit from AppPottsQuaternion)
  // Parent AppPottsQuaternionKokkos called recreate_arrays() with ndouble=4
  // We need MORE arrays, so call recreate_arrays() again with larger ndouble
  ndouble = 9;  // q0, qx, qy, qz, mobility_out, T, solid_d, G, V
  ninteger = 2; // spin, active_flag
  recreate_arrays();  // Reallocate iarray/darray pointers for ndouble=9

  allow_app_update = 1;
  app_update_only = 1;  // Skip solid-state growth for now

  // Initialize pointers
  active_flag = nullptr;
  mobility_out = nullptr;
  T = nullptr;
  solid_d = nullptr;
  G = nullptr;
  V = nullptr;
  neigh_dist = nullptr;
  unique_dot = nullptr;
  nucleation_flags = nullptr;
  nucleation_temps = nullptr;
  nucleation_sizes = nullptr;

  // Set default physical parameters (304L stainless steel)
  tl = 1723.0;  // Liquidus temperature (K)
  ts = 1673.0;  // Solidus temperature (K)
  t_room = 300.0;  // Room temperature (K)

  // Nucleation parameters
  no = 1.0e11;   // Nucleation density (#/m^3)
  tc = 5.0;      // Critical undercooling (K)
  tsig = 3.0;    // Undercooling std dev (K)
  size_norm = 8.4375e-16;  // Mean nuclei volume (m^3)
  size_sig = 1.5625e-17;   // Nuclei volume std dev (m^3)

  // Texture parameters (default values)
  c1 = 0.1;
  c2 = 0.9;
  c3 = 2.5;
  max_misorient = 10.0;  // degrees
  t_cool_max = 100.0;    // K
  mis_thresh = 5.0;      // degrees
  misorient_alpha = 5.0;

  // Solidification velocity polynomial (default: V = -1.0302e-7*dT^3 + 1.0533e-4*dT^2 + 2.2196e-3*dT)
  solid_front_length = 4;
  solid_front_coeffs = new double[4];
  solid_front_coeffs[0] = -1.0302e-7;
  solid_front_coeffs[1] = 1.0533e-4;
  solid_front_coeffs[2] = 2.2196e-3;
  solid_front_coeffs[3] = 0.0;

  // Void parameters (disabled by default)
  enable_voids = 0;
  void_density = 0.0;
  void_pore_fraction = 0.0;
  void_radius_mean = 1.0;
  void_radius_std = 0.1;
  void_radius_min = 0.5;
  void_radius_max = 2.0;

  // Temperature source (initialized via input command)
  use_temperature_source = false;
  fast_forward_search_window = 0.1;  // seconds

  // Temperature-I/O orchestration (source setup/update, laser scan path,
  // multi-pass pause/reset scheduling, fast-forward predictors, smoothing)
  // lives in the shared ThermalManager, exactly as the CPU AppAdditiveTexture.
  // This Kokkos app populates a ThermalHost snapshot (refresh_thermal_host)
  // the manager reads/writes; the manager fills the host T[] which we then
  // sync to the device k_T DualView.
  thermal = std::unique_ptr<ThermalManager>(new ThermalManager(spk, &thermal_host_));

  t_active = 0;
  time_step = dt;

  // Initialize powder activation tracking
  last_powder_activation_time = -1.0;  // Force activation at init

  // Initialize debug counters
  debug_nucleation_count = 0;
  debug_solidification_count = 0;
  debug_zero_G_count = 0;
  debug_nonzero_G_count = 0;
  debug_G_sum = 0.0;
  debug_V_sum = 0.0;

  // RNG pool is inherited from AppLatticeKokkos - initialized in parent
}

/* ----------------------------------------------------------------------
   Destructor
------------------------------------------------------------------------- */

AppAdditiveTextureKokkos::~AppAdditiveTextureKokkos()
{
  delete[] solid_front_coeffs;
  delete[] nucleation_flags;
  delete[] nucleation_temps;
  delete[] nucleation_sizes;
  delete[] neigh_dist;
  delete[] unique_dot;
}

/* ----------------------------------------------------------------------
   Allocate memory for site arrays
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::grow_app()
{
  // Call parent chain - this sets up spin, quaternions, and neighbor arrays
  // AppPottsQuaternionKokkos::grow_app() -> AppLatticeKokkos::grow_app() -> allocate_app_views()
  AppPottsQuaternionKokkos::grow_app();

  // Set up additional AM pointers from iarray/darray
  active_flag = iarray[1];
  mobility_out = darray[4];
  T = darray[5];
  solid_d = darray[6];
  G = darray[7];
  V = darray[8];

  // Allocate neighbor distance and unique arrays on host (used in site_event methods)
  if (neigh_dist) delete[] neigh_dist;
  neigh_dist = new double[maxneigh];

  if (unique_dot) delete[] unique_dot;
  unique_dot = new double[maxneigh];
}

/* ----------------------------------------------------------------------
   Allocate app-specific Kokkos views
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::allocate_app_views()
{
  // Call parent (allocates d_spin, d_q0, d_qx, d_qy, d_qz)
  AppPottsQuaternionKokkos::allocate_app_views();

  int nall = nlocal + nghost;

  // AM-specific device views
  d_active_flag = Kokkos::View<int*>("active_flag", nall);
  d_mobility_out = Kokkos::View<double*>("mobility_out", nall);
  d_solid_d = Kokkos::View<double*>("solid_d", nall);

  // Temperature field (CPU updates from HDF5, device reads)
  k_T = Kokkos::DualView<double*>("temperature", nall);

  // Gradient cache (device-only, computed once per timestep on device)
  d_G = Kokkos::View<double*>("gradient_magnitude", nall);
  d_grad_x = Kokkos::View<double*>("gradient_x", nall);
  d_grad_y = Kokkos::View<double*>("gradient_y", nall);
  d_grad_z = Kokkos::View<double*>("gradient_z", nall);
  d_V = Kokkos::View<double*>("solidification_velocity", nall);

  // Per-spin nucleation parameter arrays (device-accessible)
  d_nucleation_flags = Kokkos::View<int*>("nucleation_flags", nspins);
  d_nucleation_temps = Kokkos::View<double*>("nucleation_temps", nspins);
  d_nucleation_sizes = Kokkos::View<double*>("nucleation_sizes", nspins);

  // Global parameter arrays (device-accessible constants)
  d_solid_front_coeffs = Kokkos::View<double*>("solid_front_coeffs", solid_front_length);
  d_texture_params = Kokkos::View<double[3]>("texture_params");
}

/* ----------------------------------------------------------------------
   Initialize application
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::init_app()
{
  // Call parent initialization - sets up color lists, syncs neighbors, etc.
  // Parent also initializes rand_pool with seed from MPI rank
  AppPottsQuaternionKokkos::init_app();

  // Initialize temperature field to room temperature
  for (int i = 0; i < nlocal + nghost; i++) {
    T[i] = t_room;
    G[i] = 0.0;
    V[i] = 0.0;
    active_flag[i] = 0;  // Inactive
    mobility_out[i] = 0.0;
    solid_d[i] = 0.0;
  }

  // Set timestep for framework (matches CPU version behavior)
  dt_sweep = dt;

  // Initialize nucleation parameters (per-spin arrays and global params)
  initialize_nucleation_parameters();

  // Sync initial data to device (including new arrays)
  sync_app_data_to_device();

  // Initialize temperature field at t=0 via the ThermalManager (fills host T[]).
  if (thermal->enabled()) {
    refresh_thermal_host();
    thermal->update_temperatures(0.0);
  }

  // Note: Powder activation happens after meltpool appears, not during init
}

/* ----------------------------------------------------------------------
   Sync app data from host to device (implements AppLatticeKokkos pure virtual)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_app_data_to_device()
{
  // Call parent (syncs spin, quaternions)
  AppPottsQuaternionKokkos::sync_app_data_to_device();

  int nall = nlocal + nghost;

  // Sync AM-specific arrays to device
  auto h_active_flag = Kokkos::create_mirror_view(d_active_flag);
  auto h_mobility_out = Kokkos::create_mirror_view(d_mobility_out);
  auto h_solid_d = Kokkos::create_mirror_view(d_solid_d);

  for (int i = 0; i < nall; i++) {
    h_active_flag(i) = active_flag[i];
    h_mobility_out(i) = mobility_out[i];
    h_solid_d(i) = solid_d[i];
  }

  Kokkos::deep_copy(d_active_flag, h_active_flag);
  Kokkos::deep_copy(d_mobility_out, h_mobility_out);
  Kokkos::deep_copy(d_solid_d, h_solid_d);

  // Note: Temperature k_T is synced separately in sync_temperature_to_device()
  // Gradients d_G and d_V are device-computed, synced to host only when needed for output
}

/* ----------------------------------------------------------------------
   Sync app data from device to host (implements AppLatticeKokkos pure virtual)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_app_data_to_host()
{
  // Call parent (syncs spin, quaternions with normalization)
  AppPottsQuaternionKokkos::sync_app_data_to_host();

  int nall = nlocal + nghost;

  // Sync AM-specific arrays from device
  auto h_active_flag = Kokkos::create_mirror_view(d_active_flag);
  auto h_mobility_out = Kokkos::create_mirror_view(d_mobility_out);
  auto h_solid_d = Kokkos::create_mirror_view(d_solid_d);

  Kokkos::deep_copy(h_active_flag, d_active_flag);
  Kokkos::deep_copy(h_mobility_out, d_mobility_out);
  Kokkos::deep_copy(h_solid_d, d_solid_d);

  for (int i = 0; i < nlocal; i++) {
    active_flag[i] = h_active_flag(i);
    mobility_out[i] = h_mobility_out(i);
    solid_d[i] = h_solid_d(i);
  }

  // Sync gradient and velocity arrays for output/visualization
  // These are device-only views that must be copied to host for STITCH dumps
  auto h_G = Kokkos::create_mirror_view(d_G);
  auto h_V = Kokkos::create_mirror_view(d_V);
  Kokkos::deep_copy(h_G, d_G);
  Kokkos::deep_copy(h_V, d_V);
  for (int i = 0; i < nlocal; i++) {
    G[i] = h_G(i);
    V[i] = h_V(i);
  }
}

/* ----------------------------------------------------------------------
   sweep_color_kokkos - Override for additive app
   For the additive app, the main MC work is done in app_update() via
   phase transition kernels. This sweep method uses the parent's
   implementation for any optional Potts sweeping on solidified regions.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sweep_color_kokkos(int color, int start, int count)
{
  // For additive manufacturing, the phase transition kernels in app_update()
  // handle the main simulation work. The sweep is primarily used for
  // optional grain coarsening in solidified regions.
  // Delegate to parent class implementation.
  AppPottsQuaternionKokkos::sweep_color_kokkos(color, start, count);
}

/* ----------------------------------------------------------------------
   Update temperature from external source (CPU operation)
------------------------------------------------------------------------- */

// Populate the ThermalHost snapshot ThermalManager reads/writes. Cheap pointer
// and scalar copies; called before any thermal-> use so the manager sees the
// current lattice state without a typed back-pointer into this app. Mirrors
// AppAdditiveTexture::refresh_thermal_host().
void AppAdditiveTextureKokkos::refresh_thermal_host()
{
  thermal_host_.T = T;
  thermal_host_.xyz = xyz;
  thermal_host_.numneigh = numneigh;
  thermal_host_.neighbor = neighbor;
  thermal_host_.nlocal = nlocal;
  thermal_host_.dx = dx;
  thermal_host_.tl = tl;
  thermal_host_.ts = ts;
  thermal_host_.t_room = t_room;
  thermal_host_.time = time;
  thermal_host_.dt = dt;
  thermal_host_.comm = comm;
  thermal_host_.ghost_iindices = ghost_iindices;
  thermal_host_.nghost_iarray = nghost_iarray;
  thermal_host_.ghost_dindices = ghost_dindices;
  thermal_host_.nghost_darray = nghost_darray;
}

void AppAdditiveTextureKokkos::update_temperature_from_source(double simulation_time)
{
  // Delegate to the ThermalManager, which fills host T[] (== this->T) from the
  // configured source at simulation_time (and applies optional smoothing).
  if (!thermal->enabled()) return;
  refresh_thermal_host();
  thermal->update_temperatures(simulation_time);
}

/* ----------------------------------------------------------------------
   Sync temperature from host to device (after CPU update from HDF5)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_temperature_to_device()
{
  int nall = nlocal + nghost;

  // Copy host temperature array to DualView host mirror
  auto h_T = k_T.view_host();
  for (int i = 0; i < nall; i++) {
    h_T(i) = T[i];
  }

  // Mark host as modified and sync to device
  k_T.modify_host();
  k_T.sync_device();
}

/* ----------------------------------------------------------------------
   Sync temperature from device to host (after a device-side fill). Copies
   k_T back into the host T array (darray[5]) so host consumers -- the d6
   diagnostic, activate_powder_sites(), optional smoothing diagnostics -- see
   the current field. Cheap O(nall) copy; the expensive per-site evaluation
   stays on the device.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_temperature_to_host()
{
  int nall = nlocal + nghost;
  k_T.sync_host();
  auto h_T = k_T.view_host();
  for (int i = 0; i < nall; i++) {
    T[i] = h_T(i);
  }
  host_T_stale = false;
}

/* ----------------------------------------------------------------------
   Selective ghost sync (Phase 2). Move ONLY the neighbor-read Monte Carlo
   state -- spin (iarray[0]), active_flag (iarray[1]), quaternions q0-3
   (darray[0-3]) -- between device and the raw host arrays, wrapping each raw
   pointer as an unmanaged host View so deep_copy writes straight into the
   comm-visible iarray/darray memory (one contiguous DMA per array, no serial
   loop, no per-call mirror allocation). T is device-resident (no ghost comm)
   and mobility/solid_d/G/V are output-only, so none of those are touched here.

   _to_host copies the OWNED slice (so comm->all_selective sends current values);
   _to_device copies the received GHOST slice back. Owned device state is left
   untouched (the kernels are authoritative for owned). Quaternions are moved
   raw (no renormalization); they stay near-unit through the physics, and the
   output-step full sync renormalizes for dumps.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_ghost_to_host()
{
  if (nlocal == 0) return;
  using HUi = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  using HUd = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  const int nall = nlocal + nghost;
  const auto owned = Kokkos::make_pair(0, nlocal);
  Kokkos::deep_copy(Kokkos::subview(HUi(spin, nall), owned),
                    Kokkos::subview(d_spin, owned));
  Kokkos::deep_copy(Kokkos::subview(HUi(active_flag, nall), owned),
                    Kokkos::subview(d_active_flag, owned));
  Kokkos::deep_copy(Kokkos::subview(HUd(q0, nall), owned), Kokkos::subview(d_q0, owned));
  Kokkos::deep_copy(Kokkos::subview(HUd(qx, nall), owned), Kokkos::subview(d_qx, owned));
  Kokkos::deep_copy(Kokkos::subview(HUd(qy, nall), owned), Kokkos::subview(d_qy, owned));
  Kokkos::deep_copy(Kokkos::subview(HUd(qz, nall), owned), Kokkos::subview(d_qz, owned));
}

void AppAdditiveTextureKokkos::sync_ghost_to_device()
{
  if (nghost == 0) return;
  using HUi = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  using HUd = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  const int nall = nlocal + nghost;
  const auto ghost = Kokkos::make_pair(nlocal, nall);
  Kokkos::deep_copy(Kokkos::subview(d_spin, ghost),
                    Kokkos::subview(HUi(spin, nall), ghost));
  Kokkos::deep_copy(Kokkos::subview(d_active_flag, ghost),
                    Kokkos::subview(HUi(active_flag, nall), ghost));
  Kokkos::deep_copy(Kokkos::subview(d_q0, ghost), Kokkos::subview(HUd(q0, nall), ghost));
  Kokkos::deep_copy(Kokkos::subview(d_qx, ghost), Kokkos::subview(HUd(qx, nall), ghost));
  Kokkos::deep_copy(Kokkos::subview(d_qy, ghost), Kokkos::subview(HUd(qy, nall), ghost));
  Kokkos::deep_copy(Kokkos::subview(d_qz, ghost), Kokkos::subview(HUd(qz, nall), ghost));
}

/* ----------------------------------------------------------------------
   Checksum of the device GHOST-slice MC state (spin/active_flag/quaternions),
   used by the SPK_VERIFY_GHOST correctness check to compare the selective and
   full ghost-exchange paths. Distinct per-array scales keep different fields
   from cancelling. Reduced over this rank's ghost slice only.
------------------------------------------------------------------------- */

double AppAdditiveTextureKokkos::ghost_mc_checksum()
{
  if (nghost == 0) return 0.0;
  auto d_s = d_spin; auto d_a = d_active_flag;
  auto q0v = d_q0; auto qxv = d_qx; auto qyv = d_qy; auto qzv = d_qz;
  double sum = 0.0;
  Kokkos::parallel_reduce("ghost_mc_checksum",
    Kokkos::RangePolicy<>(nlocal, nlocal + nghost),
    KOKKOS_LAMBDA(const int i, double &s) {
      s += 1.0 * d_s(i) + 1000.0 * d_a(i)
         + q0v(i) + 2.0 * qxv(i) + 3.0 * qyv(i) + 4.0 * qzv(i);
    }, sum);
  return sum;
}

/* ----------------------------------------------------------------------
   Global peak temperature over owned sites, reduced on the device (no host T
   needed) and then across ranks. Used by the device fast-forward decision.
------------------------------------------------------------------------- */

double AppAdditiveTextureKokkos::device_global_max_temperature()
{
  auto d_T = k_T.view_device();
  const int n = nlocal;
  double local_max = 0.0;
  Kokkos::parallel_reduce("Tmax", n,
    KOKKOS_LAMBDA(const int i, double &m) { if (d_T(i) > m) m = d_T(i); },
    Kokkos::Max<double>(local_max));
  double global_max = local_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, world);
  return global_max;
}

/* ----------------------------------------------------------------------
   Evaluate the analytical Rosenthal temperature on the device, writing k_T
   for all nlocal+nghost sites. The laser pose + material parameters come from
   the host-advanced RosenthalDevicePlan; the pool-local transform
   (am_raster.h compute_position_relative_to_pool) and the Rosenthal kernel
   (temperature_source_rosenthal.cpp rosenthal_pointwise/rosenthal_kernel) are
   reproduced here device-side -- keep them in sync.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::ensure_device_xyz()
{
  const int nall = nlocal + nghost;
  if (d_xyz_n == nall) return;

  // Build a host mirror by explicit element assignment from the double** so the
  // copy is correct regardless of the device View layout (host xyz is row-major
  // [nall][3]; a LayoutLeft device view would otherwise misread it as
  // column-major -- the source of an earlier melt-surface gradient bug).
  d_xyz = Kokkos::View<double*[3]>("d_xyz", nall);
  auto h_xyz = Kokkos::create_mirror_view(d_xyz);
  for (int i = 0; i < nall; i++) {
    h_xyz(i,0) = xyz[i][0];
    h_xyz(i,1) = xyz[i][1];
    h_xyz(i,2) = xyz[i][2];
  }
  Kokkos::deep_copy(d_xyz, h_xyz);
  d_xyz_n = nall;
}

// Device reproduction of RosenthalTemperatureSource::rosenthal_kernel:
// Q_eff / (2 pi k R) * exp(-v (xi + R) / (2 alpha)), with the 1/R singularity
// clamped at r_min. (The image-source factor of 2 is applied by the caller.)
KOKKOS_INLINE_FUNCTION
static double rosenthal_device_kernel(double Qeff, double v, double xi, double R,
                                      double r_min, double k_cond, double alpha,
                                      double two_pi)
{
  double Rk = (R < r_min) ? r_min : R;
  if (Rk <= 0.0) return 0.0;
  return (Qeff / (two_pi * k_cond * Rk)) *
         Kokkos::exp(-v * (xi + Rk) / (2.0 * alpha));
}

void AppAdditiveTextureKokkos::fill_temperature_rosenthal_device(
    const RosenthalDevicePlan &plan)
{
  const int nall = nlocal + nghost;

  ensure_device_xyz();

  // KEYHOLE: upload the (fixed) line-integral quadrature once.
  if (plan.mode == 2 && d_ros_nquad != plan.n_quad) {
    std::vector<double> nodes, weights;
    thermal->rosenthal_copy_keyhole_quadrature(nodes, weights);
    auto up = [](Kokkos::View<double*> &dv, const std::vector<double> &h,
                 const char *name) {
      const int n = (int)h.size();
      dv = Kokkos::View<double*>(name, n);
      if (n == 0) return;
      auto hm = Kokkos::create_mirror_view(dv);
      for (int i = 0; i < n; i++) hm(i) = h[i];
      Kokkos::deep_copy(dv, hm);
    };
    up(d_ros_qnodes, nodes, "ros_qnodes");
    up(d_ros_qweights, weights, "ros_qweights");
    d_ros_nquad = plan.n_quad;
  }

  auto d_T   = k_T.view_device();
  auto d_xyz_local = d_xyz;
  auto qnodes = d_ros_qnodes;
  auto qweights = d_ros_qweights;

  // Capture plan scalars by value for device lambda.
  const double dx_l = this->dx;
  const double px = plan.px, py = plan.py, layer_z = plan.layer_z;
  const double ux = plan.ux, uy = plan.uy;
  const double v = plan.v, T0 = plan.T0;
  const double lambdaQ = plan.lambdaQ, k_cond = plan.k_cond, alpha = plan.alpha;
  const double r_min = plan.r_min, eta_y = plan.eta_y, eta_z = plan.eta_z;
  const double lambda_pQ = plan.lambda_pQ, lambda_lQ = plan.lambda_lQ;
  const int n_quad = plan.n_quad;
  const int mode = plan.mode;
  const bool scan_active = plan.scan_active;
  const double two_pi = 2.0 * 3.14159265358979323846;

  Kokkos::parallel_for("RosenthalFill", nall, KOKKOS_LAMBDA(const int i) {
    // Site physical coordinates (meters).
    const double x = d_xyz_local(i,0) * dx_l;
    const double y = d_xyz_local(i,1) * dx_l;
    const double z = d_xyz_local(i,2) * dx_l;

    // Pool-local frame: project (site - laser) onto scan / transverse axes.
    const double rsx0 = x - px;
    const double rsx1 = y - py;
    const double z_rel = z - layer_z;
    const double xi = ux*rsx0 + uy*rsx1;            // along scan direction
    const double tsx0 = rsx0 - xi*ux;
    const double tsx1 = rsx1 - xi*uy;
    const double y_rel = -uy*tsx0 + ux*tsx1;        // transverse

    double Tval = T0;
    // Free-surface cutoff: vacuum above the laser plane. Inactive path -> T0.
    if (scan_active && z_rel <= 0.0) {
      if (mode == 2) {
        // KEYHOLE: image-doubled surface point source + buried line source
        // integrated over the Gauss-Legendre nodes (image-doubled per node).
        const double R = Kokkos::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
        double T_rise = 2.0 * rosenthal_device_kernel(lambda_pQ, v, xi, R,
                                                       r_min, k_cond, alpha, two_pi);
        for (int q = 0; q < n_quad; q++) {
          const double zarg = qnodes(q) + z_rel;
          const double Rp = Kokkos::sqrt(xi*xi + y_rel*y_rel + zarg*zarg);
          T_rise += 2.0 * qweights(q) *
                    rosenthal_device_kernel(lambda_lQ, v, xi, Rp,
                                            r_min, k_cond, alpha, two_pi);
        }
        Tval = T0 + T_rise;
      } else {
        double R;
        if (mode == 1) {                            // ANISOTROPIC
          const double yy = eta_y * y_rel;
          const double zz = eta_z * z_rel;
          R = Kokkos::sqrt(xi*xi + yy*yy + zz*zz);
        } else {                                    // STANDARD
          R = Kokkos::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
        }
        Tval = T0 + 2.0 * rosenthal_device_kernel(lambdaQ, v, xi, R,
                                                   r_min, k_cond, alpha, two_pi);
      }
    }
    d_T(i) = Tval;
  });
  Kokkos::fence();
  k_T.modify_device();
}

/* ----------------------------------------------------------------------
   Build and upload the Gauss-Legendre roots/weights table for orders
   2..MOSER_GL_MAX_ORDER. Order N occupies N entries starting at offset(N).
   Roots/weights are generated by the same Newton solve on the Legendre roots
   used by GREENAM GaussLegendreQuadrature, so the device quadrature matches the
   host integrator. Order 1 is the midpoint rule (handled in the kernel).
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::build_gl_table()
{
  if (d_gl_built) return;
  const int maxN = MOSER_GL_MAX_ORDER;

  std::vector<int> offset(maxN + 1, 0);
  std::vector<double> roots, weights;
  for (int N = 2; N <= maxN; N++) {
    offset[N] = (int)roots.size();
    for (int i = 1; i <= N; i++) {
      // Initial guess for the i-th root, then Newton-refine on P_N.
      double x = std::cos(M_PI * (i - 0.25) / (N + 0.5));
      double d = 0.0;
      for (int it = 0; it < 100; it++) {
        // Evaluate P_N(x) value (v) and derivative (d) by recurrence.
        double vsub1 = x, vsub2 = 1.0, v = 1.0;
        const double f = 1.0 / (x * x - 1.0);
        for (int kk = 2; kk <= N; kk++) {
          v = ((2 * kk - 1) * x * vsub1 - (kk - 1) * vsub2) / kk;
          d = kk * f * (x * v - vsub1);
          vsub2 = vsub1;
          vsub1 = v;
        }
        const double dr = v / d;
        x -= dr;
        if (std::fabs(dr) <= 2e-16) break;
      }
      roots.push_back(x);
      weights.push_back(2.0 / ((1.0 - x * x) * d * d));
    }
  }

  d_gl_offset = Kokkos::View<int*>("gl_offset", maxN + 1);
  d_gl_roots  = Kokkos::View<double*>("gl_roots", roots.size());
  d_gl_weights = Kokkos::View<double*>("gl_weights", weights.size());
  auto h_off = Kokkos::create_mirror_view(d_gl_offset);
  auto h_r   = Kokkos::create_mirror_view(d_gl_roots);
  auto h_w   = Kokkos::create_mirror_view(d_gl_weights);
  for (int N = 0; N <= maxN; N++) h_off(N) = offset[N];
  for (size_t i = 0; i < roots.size(); i++) { h_r(i) = roots[i]; h_w(i) = weights[i]; }
  Kokkos::deep_copy(d_gl_offset, h_off);
  Kokkos::deep_copy(d_gl_roots, h_r);
  Kokkos::deep_copy(d_gl_weights, h_w);
  d_gl_built = true;
}

/* ----------------------------------------------------------------------
   Device-side Moser temperature. Each site integrates the Green's-function
   over the scan-path history, reproducing GREENAM
   ScanIntegration::integrate_point_adaptive() + GaussGreenEllips (STANDARD
   single lobe, no fluctuations). The host advances pass scheduling and hands us
   scalar params (plan) + the scan segment arrays (uploaded on version change).
------------------------------------------------------------------------- */

// Device reproduction of FluctuationTable::get_at (binary search + clamped
// linear interpolation over the [foff, foff+fn) slice). fn <= 0 -> factors = 1.
KOKKOS_INLINE_FUNCTION
static void moser_fluct_get_at(
    const Kokkos::View<double*> &ft, const Kokkos::View<double*> &fw,
    const Kokkos::View<double*> &fd, const Kokkos::View<double*> &fp,
    int foff, int fn, double s, double &fW, double &fD, double &fP)
{
  if (fn <= 0)            { fW = fD = fP = 1.0; return; }
  if (s <= ft(foff))      { fW = fw(foff); fD = fd(foff); fP = fp(foff); return; }
  const int last = foff + fn - 1;
  if (s >= ft(last))      { fW = fw(last); fD = fd(last); fP = fp(last); return; }
  int lo = 0, hi = fn - 1;
  while (lo + 1 < hi) { int mid = (lo + hi) / 2;
    if (ft(foff + mid) <= s) lo = mid; else hi = mid; }
  const double t0 = ft(foff + lo), t1 = ft(foff + lo + 1);
  const double denom = t1 - t0;
  const double frac = (denom > 0.0) ? (s - t0) / denom : 0.0;
  const double om = 1.0 - frac;
  fW = om * fw(foff + lo) + frac * fw(foff + lo + 1);
  fD = om * fd(foff + lo) + frac * fd(foff + lo + 1);
  fP = om * fp(foff + lo) + frac * fp(foff + lo + 1);
}

void AppAdditiveTextureKokkos::fill_temperature_moser_device(
    const MoserDevicePlan &plan)
{
  const int nall = nlocal + nghost;
  ensure_device_xyz();
  build_gl_table();

  // (Re)upload the shared geometry, per-lobe powers, and per-lobe fluctuation
  // tables when the scan generation changes.
  if (d_seg_version != plan.version) {
    std::vector<double> b, xl, yl, nx, ny, sp, pw, ft, fw, fd, fp;
    thermal->moser_copy_device_data(b, xl, yl, nx, ny, sp, pw, ft, fw, fd, fp);
    auto up = [](Kokkos::View<double*> &dv, const std::vector<double> &h,
                 const char *name) {
      const int n = (int)h.size();
      dv = Kokkos::View<double*>(name, n);   // n == 0 is valid (never indexed)
      if (n == 0) return;
      auto hm = Kokkos::create_mirror_view(dv);
      for (int i = 0; i < n; i++) hm(i) = h[i];
      Kokkos::deep_copy(dv, hm);
    };
    up(d_seg_bounds, b, "seg_bounds"); up(d_seg_xl, xl, "seg_xl");
    up(d_seg_yl, yl, "seg_yl");        up(d_seg_nx, nx, "seg_nx");
    up(d_seg_ny, ny, "seg_ny");        up(d_seg_speeds, sp, "seg_speeds");
    up(d_seg_powers, pw, "seg_powers");
    up(d_fluct_t, ft, "fluct_t");      up(d_fluct_w, fw, "fluct_w");
    up(d_fluct_d, fd, "fluct_d");      up(d_fluct_p, fp, "fluct_p");
    d_seg_version = plan.version;
  }

  auto d_T = k_T.view_device();
  auto d_xyz_local = d_xyz;
  auto bounds = d_seg_bounds; auto xlv = d_seg_xl; auto ylv = d_seg_yl;
  auto powers = d_seg_powers;  auto nxv = d_seg_nx;  auto nyv = d_seg_ny;
  auto speeds = d_seg_speeds;
  auto ftv = d_fluct_t; auto fwv = d_fluct_w; auto fdv = d_fluct_d; auto fpv = d_fluct_p;
  auto gl_roots = d_gl_roots; auto gl_weights = d_gl_weights; auto gl_off = d_gl_offset;

  const double dx_l = this->dx;
  const double tt = plan.time, T0 = plan.T0;
  const double alpha = plan.alpha, c = plan.c;
  const double cl = plan.char_length;
  const int nseg = plan.nseg;
  const int nlobes = plan.n_lobes;
  // Per-lobe scalars (captured individually; rebuilt into local arrays in-kernel).
  const double sx0 = plan.sx[0], sx1 = plan.sx[1];
  const double sy0 = plan.sy[0], sy1 = plan.sy[1];
  const double sz0 = plan.sz[0], sz1 = plan.sz[1];
  const double zl0 = plan.zl[0], zl1 = plan.zl[1];
  const int fn0 = plan.fluct_n[0], fn1 = plan.fluct_n[1];
  const double surface_z = plan.zl[0];   // top lobe zl = real free surface (z_offset 0)
  const int maxN = MOSER_GL_MAX_ORDER;
  const double eps = 2.220446049250313e-16;
  const double invPi3_2 = 1.0 / (M_PI * 1.7724538509055160273); // 1/(pi*sqrt(pi))

  Kokkos::parallel_for("MoserFill", nall, KOKKOS_LAMBDA(const int site) {
    const double x = d_xyz_local(site,0) * dx_l;
    const double y = d_xyz_local(site,1) * dx_l;
    const double z = d_xyz_local(site,2) * dx_l;

    // Vacuum above the real free surface; integral is empty before t=0.
    if (z > surface_z || tt <= 0.0 || nseg <= 0) { d_T(site) = T0; return; }

    // binary_search(bounds, tt, nseg): largest i with bounds(i) <= tt.
    int end_idx = -1;
    { int L = 0, R = nseg - 1;
      while (L <= R) { int m = (L + R) / 2;
        if (bounds(m) <= tt) {
          if (m + 1 == nseg || bounds(m+1) >= tt) { end_idx = m; break; }
          L = m + 1;
        } else R = m - 1; } }
    if (end_idx < 0) { d_T(site) = T0; return; }
    if (Kokkos::fabs(tt - bounds(end_idx)) <= 4.0 * eps) end_idx--;

    const double sxa[2] = {sx0, sx1}, sya[2] = {sy0, sy1};
    const double sza[2] = {sz0, sz1}, zla[2] = {zl0, zl1};
    const int    fna[2] = {fn0, fn1};

    // KEYHOLE sums the contributions of both lobes; STANDARD has one.
    double res = 0.0;
    for (int lobe = 0; lobe < nlobes; lobe++) {
      const double sx = sxa[lobe], sy = sya[lobe], sz = sza[lobe], zl = zla[lobe];
      const int fn = fna[lobe];
      const int foff = (lobe == 0) ? 0 : fn0;     // fluct slice for this lobe
      const int poff = lobe * nseg;               // power slice for this lobe
      const double avg_sigma = (sx + sy + sz) / 3.0;
      const double avg_sigma_sq = avg_sigma * avg_sigma;

      for (int i = 0; i <= end_idx; i++) {
        const double p_i = powers(poff + i);
        if (Kokkos::fabs(p_i) <= 4.0 * eps) continue;
        const double xl_i = xlv(i), yl_i = ylv(i);
        const double nx_i = nxv(i), ny_i = nyv(i);
        const double sp_i = speeds(i), t0_i = bounds(i);

        const double tstart = bounds(i);
        const double tend = (i + 1 >= nseg) ? tt
            : (tt <= bounds(i+1) + 4.0*eps ? tt : bounds(i+1));

        const double sigma_eff = 4.0*alpha*(tt - tend) + avg_sigma_sq;
        double quad_size = cl*cl*sigma_eff/4.0/alpha;
        if (sp_i > 4.0*eps)
          quad_size = Kokkos::fmin(cl*Kokkos::sqrt(sigma_eff)/sp_i, quad_size);

        const double n_quad_pts = (tend - tstart)/quad_size;
        int order = (int)Kokkos::ceil(n_quad_pts); if (order < 1) order = 1;
        if (order > maxN) order = maxN;
        const double step_size = quad_size*order;
        double n_int = Kokkos::ceil((tend - tstart)/step_size);
        if (n_int < 1.0) n_int = 1.0;
        const int nint = (int)n_int;
        const double dt = (tend - tstart)/n_int;

        for (int j = 0; j < nint; j++) {
          const double a = tstart + j*dt;
          const double bb = tstart + (j+1)*dt;
          const double pmid = 0.5*(bb - a), qmid = 0.5*(bb + a);
          // node count: 1 -> midpoint rule, else `order` Gauss-Legendre nodes.
          const int nnodes = (order == 1) ? 1 : order;
          const int goff = (order == 1) ? 0 : gl_off(order);
          double sum = 0.0;
          for (int nn = 0; nn < nnodes; nn++) {
            const double s = (order == 1) ? qmid : (pmid*gl_roots(goff+nn) + qmid);
            const double w = (order == 1) ? (bb - a) : gl_weights(goff+nn);
            // Fluctuation modulation of (sx,sy), sz, power at this emission time.
            double fW = 1.0, fD = 1.0, fP = 1.0;
            if (fn > 0) moser_fluct_get_at(ftv, fwv, fdv, fpv, foff, fn, s, fW, fD, fP);
            const double sxu = sx*fW, syu = sy*fW, szu = sz*fD, pu = p_i*fP;
            const double st = 4.0*alpha*(tt - s);
            const double cx = x - xl_i, cy = y - yl_i;
            const double xt = cx*nx_i + cy*ny_i;
            const double yt = -cx*ny_i + cy*nx_i;
            const double zt = z - zl;
            const double vx = 1.0/(sxu*sxu+st), vy = 1.0/(syu*syu+st), vz = 1.0/(szu*szu+st);
            const double xeff = xt - sp_i*(s - t0_i);
            const double e = Kokkos::exp(-xeff*xeff*vx - yt*yt*vy - zt*zt*vz);
            sum += (2.0*pu/c*invPi3_2 * e * Kokkos::sqrt(vx*vy*vz)) * w;
          }
          // GL needs the *pmid scale; the midpoint rule already used (bb-a) as w.
          res += (order == 1) ? sum : (pmid * sum);
        }
      }
    }
    d_T(site) = T0 + res;
  });
  Kokkos::fence();
  k_T.modify_device();
}

/* ----------------------------------------------------------------------
   Device-side HDF5 unstructured-mesh temperature. The host has already done the
   expensive work in thermal->hdf5_device_plan(): loaded the layer, (re)built the
   per-site element cache at a layer boundary, and precomputed the per-node
   temperature slice for this time (cached_nodal_temps; time interpolation +
   optional nodal mesh smoothing). Here we just upload that data and do the
   per-site barycentric interpolation on the GPU, reproducing
   HDF5UnstructuredTemperatureSource::get_temperature_at_site_fast() in the same
   operation order so the device field matches the host fast path bit-for-bit.

   Fills ALL nall sites (owned + ghost) POSITIONALLY, like the Rosenthal/Moser
   device fills. The element cache is built over nall (the chunk-load pad covers
   ghosts), so every ghost has a valid entry and computes its own T from this
   rank's haloed mesh. A ghost equals the owning rank's value by construction
   (same global field, same physical point), so multi-rank needs no T ghost comm
   and no host round-trip -- the field is fully device-resident. See hdf5_device.h.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::fill_temperature_hdf5_device(
    const HDF5DevicePlan &plan)
{
  if (!plan.valid) return;
  const int n = plan.nsites;   // nall (owned + ghost); fills positionally
  const int nnodes = plan.n_nodes;

  // (Re)upload the STATIC per-site element cache only when the host rebuilds it
  // (layer boundary -> cache generation bumps) or the site count changes.
  if (d_hdf5_cache_version != plan.version || d_hdf5_nsites != n) {
    std::vector<int> node_idx, valid;
    std::vector<double> weights;
    thermal->hdf5_copy_cache(node_idx, weights, valid);

    d_hdf5_node_idx = Kokkos::View<int*[4]>("hdf5_node_idx", n);
    d_hdf5_weights  = Kokkos::View<double*[3]>("hdf5_weights", n);
    d_hdf5_valid    = Kokkos::View<int*>("hdf5_valid", n);
    auto h_idx = Kokkos::create_mirror_view(d_hdf5_node_idx);
    auto h_w   = Kokkos::create_mirror_view(d_hdf5_weights);
    auto h_v   = Kokkos::create_mirror_view(d_hdf5_valid);
    for (int i = 0; i < n; i++) {
      h_v(i) = valid[i];
      const size_t i4 = (size_t)i * 4, i3 = (size_t)i * 3;  // avoid int overflow
      for (int j = 0; j < 4; j++) h_idx(i, j) = node_idx[i4 + j];
      for (int j = 0; j < 3; j++) h_w(i, j) = weights[i3 + j];
    }
    Kokkos::deep_copy(d_hdf5_node_idx, h_idx);
    Kokkos::deep_copy(d_hdf5_weights, h_w);
    Kokkos::deep_copy(d_hdf5_valid, h_v);
    d_hdf5_cache_version = plan.version;
    d_hdf5_nsites = n;
  }

  // Upload the per-step nodal temperature slice (resize on node-count change).
  if ((int) d_hdf5_nodal_temps.extent(0) != nnodes)
    d_hdf5_nodal_temps = Kokkos::View<double*>("hdf5_nodal_temps", nnodes);
  if (nnodes > 0) {
    const std::vector<double> &nt = thermal->hdf5_nodal_temps();
    auto h_nt = Kokkos::create_mirror_view(d_hdf5_nodal_temps);
    for (int k = 0; k < nnodes; k++) h_nt(k) = nt[k];
    Kokkos::deep_copy(d_hdf5_nodal_temps, h_nt);
  }

  auto d_T = k_T.view_device();
  auto nidx = d_hdf5_node_idx;
  auto wts  = d_hdf5_weights;
  auto vld  = d_hdf5_valid;
  auto nt   = d_hdf5_nodal_temps;
  const double def = plan.default_temp;

  // Full fusion: evaluate the active-front predicate here (T is in a register;
  // the fill already visits every site) and ATOMIC-APPEND active owned sites
  // straight into d_active_sites, so there is no separate compaction scan at all
  // (the parallel_scan is a multi-pass prefix sum -- the GPU's #2 cost). Owned
  // sites only (i < nlocal); ghosts just get T. active_flag/solid_d are stable
  // between the fill and the microstructure kernels (which run after), so this
  // list matches the full predicate scan as a set (the appended order differs,
  // which is fine -- kernels process sites independently; temperature untouched).
  if ((int) d_active_sites.extent(0) != nlocal)
    d_active_sites = Kokkos::View<int*>("active_sites", nlocal);
  if (d_active_count.data() == nullptr)
    d_active_count = Kokkos::View<int>("active_count");
  Kokkos::deep_copy(d_active_count, 0);
  auto d_asites = d_active_sites;
  auto d_acount = d_active_count;
  auto d_af = d_active_flag;
  auto d_sd = d_solid_d;
  const double tl_f = this->tl;
  const int nrefine_f = this->nrefine;
  const int nlocal_f = this->nlocal;

  Kokkos::parallel_for("HDF5Fill", n, KOKKOS_LAMBDA(const int i) {
    double Ti;
    if (!vld(i)) {
      Ti = def;
    } else {
      const int i0 = nidx(i, 0), i1 = nidx(i, 1), i2 = nidx(i, 2), i3 = nidx(i, 3);
      const double t0 = nt(i0);
      Ti = t0 + wts(i, 0) * (nt(i1) - t0)
              + wts(i, 1) * (nt(i2) - t0)
              + wts(i, 2) * (nt(i3) - t0);
    }
    d_T(i) = Ti;
    if (i < nlocal_f) {
      const int af = d_af(i);
      const double sd = d_sd(i);
      if (af == 2 || Ti >= tl_f ||
          (sd > -nrefine_f - 1 && sd < 0.0) ||
          (af == 3 && sd == -nrefine_f - 2)) {
        const int slot = Kokkos::atomic_fetch_add(&d_acount(), 1);
        d_asites(slot) = i;
      }
    }
  });
  Kokkos::fence();
  k_T.modify_device();
  device_active_list_fresh = true;
}

/* ----------------------------------------------------------------------
   Compute thermal gradients on device (cache for later use)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::compute_gradients_kernel()
{
  // Get device views
  auto d_T = k_T.view_device();  // Read-only temperature on device
  auto k_numneigh_d = k_numneigh.view_device();
  auto k_neighbor_d = k_neighbor.view_device();

  // Cached, layout-correct device copy of site coordinates.
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;

  // Get active_flag for melt surface detection (simple View)
  auto d_active_flag_local = this->d_active_flag;

  // Capture member variables for lambda
  double dx_lattice = this->dx;
  auto d_G_local = this->d_G;
  auto d_grad_x_local = this->d_grad_x;
  auto d_grad_y_local = this->d_grad_y;
  auto d_grad_z_local = this->d_grad_z;
  int nlocal_local = this->nlocal;

  // Compute gradients in parallel
  Kokkos::parallel_for("ComputeGradients", nlocal,
    KOKKOS_LAMBDA(int i) {
      double gradient[3], magnitude;

      ThermalOps::compute_temperature_gradient(
        i, d_T, d_active_flag_local, k_numneigh_d, k_neighbor_d, k_xyz_d,
        dx_lattice, gradient, &magnitude);

      // Cache results for later use
      d_G_local(i) = magnitude;
      d_grad_x_local(i) = gradient[0];
      d_grad_y_local(i) = gradient[1];
      d_grad_z_local(i) = gradient[2];
    });

  // Fence removed - single fence at end of app_update() handles all kernels
}

/* ----------------------------------------------------------------------
   Initialize nucleation parameters on device
   Phase 3.1: Port per-spin nucleation arrays from CPU to device
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::initialize_nucleation_parameters()
{
  // Allocate host temp arrays for nucleation parameters
  delete[] nucleation_flags;
  delete[] nucleation_temps;
  delete[] nucleation_sizes;
  nucleation_flags = new int[nspins];
  nucleation_temps = new double[nspins];
  nucleation_sizes = new double[nspins];

  // Initialize nucleation_flags (which spins can nucleate)
  // Nucleation fraction = volume of voxel * nucleation density (grains/m^3)
  double nucleationFraction = dx * dx * dx * no;

  if (nucleationFraction >= 1.0 && domain->me == 0) {
    fprintf(screen, "Nucleation fraction is %g (>= 1). All spins can nucleate.\n", nucleationFraction);
  }

  // Assign nucleation flags randomly based on nucleation density
  for (int i = 0; i < nspins; i++) {
    if (nucleationFraction >= 1.0) {
      nucleation_flags[i] = 1;  // All spins can nucleate
    } else {
      nucleation_flags[i] = (ranapp->uniform() <= nucleationFraction) ? 1 : 0;
    }
  }

  // Initialize nucleation_temps and nucleation_sizes using Gaussian distributions
  // Use C++ standard library random (consistent with CPU version)
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> dist_T{tc, tsig};       // Critical undercooling (K)
  std::normal_distribution<> dist_S{size_norm, size_sig};  // Nucleus volume (m^3)

  for (int i = 0; i < nspins; i++) {
    nucleation_temps[i] = dist_T(gen);
    nucleation_sizes[i] = dist_S(gen);
  }

  // Copy to device views
  auto h_nucleation_flags = Kokkos::create_mirror_view(d_nucleation_flags);
  auto h_nucleation_temps = Kokkos::create_mirror_view(d_nucleation_temps);
  auto h_nucleation_sizes = Kokkos::create_mirror_view(d_nucleation_sizes);

  for (int i = 0; i < nspins; i++) {
    h_nucleation_flags(i) = nucleation_flags[i];
    h_nucleation_temps(i) = nucleation_temps[i];
    h_nucleation_sizes(i) = nucleation_sizes[i];
  }

  Kokkos::deep_copy(d_nucleation_flags, h_nucleation_flags);
  Kokkos::deep_copy(d_nucleation_temps, h_nucleation_temps);
  Kokkos::deep_copy(d_nucleation_sizes, h_nucleation_sizes);

  // Copy global parameters to device
  auto h_solid_front_coeffs = Kokkos::create_mirror_view(d_solid_front_coeffs);
  for (int i = 0; i < solid_front_length; i++) {
    h_solid_front_coeffs(i) = solid_front_coeffs[i];
  }
  Kokkos::deep_copy(d_solid_front_coeffs, h_solid_front_coeffs);

  auto h_texture_params = Kokkos::create_mirror_view(d_texture_params);
  h_texture_params(0) = c1;
  h_texture_params(1) = c2;
  h_texture_params(2) = c3;
  Kokkos::deep_copy(d_texture_params, h_texture_params);
}

/* ----------------------------------------------------------------------
   Melting kernel: Transition sites to liquid state when T >= liquidus
   Phase 3.1: Simplest phase transition kernel (no neighbor interactions)
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::melting_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (no double buffering needed for self-only writes)
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = k_T.view_device();
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;
  auto d_G_local = d_G;
  auto d_V_local = d_V;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nspins_local = this->nspins;
  auto rand_pool_local = *rand_pool;

  // Melting kernel: parallel over the active-front list with event tracking
  // Race-free: each site only reads/writes itself
  int naccept_local = 0;
  Kokkos::parallel_reduce("MeltingKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count) {
      const int i = d_active_sites_local(idx);
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);

      // Check if site should melt: T >= liquidus && not already liquid
      if (T_i >= tl_local && active_flag_i != 2) {
        // Get thread-local RNG state
        auto generator = rand_pool_local.get_state();

        // Transition to liquid state
        d_active_flag_local(i) = 2;  // LIQUID

        // Randomize grain ID (destroy grain structure)
        d_spin_local(i) = static_cast<int>(nspins_local * generator.drand());

        // Randomize orientation (destroy crystallographic texture)
        // Generate random unit quaternion using Marsaglia method
        double x0, x1, x2, x3, r1, r2;
        do {
          x0 = 2.0 * generator.drand() - 1.0;
          x1 = 2.0 * generator.drand() - 1.0;
          r1 = x0*x0 + x1*x1;
        } while (r1 >= 1.0);

        do {
          x2 = 2.0 * generator.drand() - 1.0;
          x3 = 2.0 * generator.drand() - 1.0;
          r2 = x2*x2 + x3*x3;
        } while (r2 >= 1.0);

        double s = Kokkos::sqrt((1.0 - r1) / r2);
        d_q0_local(i) = x0;
        d_qx_local(i) = x1;
        d_qy_local(i) = x2 * s;
        d_qz_local(i) = x3 * s;

        // Reset solidification tracking
        d_solid_d_local(i) = 0.0;
        d_G_local(i) = 0.0;
        d_V_local(i) = 0.0;

        // Return RNG state to pool
        rand_pool_local.free_state(generator);

        // Track melting event
        count++;
      }
      // No else branch needed - site keeps its current values if not melting
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Nucleation kernel: Stochastic nucleation in undercooled liquid
   Phase 3.2: Sites transition to solid when undercooling exceeds critical value
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::nucleation_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (no double buffering needed for self-only writes)
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = k_T.view_device();
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nspins_local = this->nspins;
  int nrefine_local = this->nrefine;
  auto rand_pool_local = *rand_pool;
  auto d_nucleation_flags_local = this->d_nucleation_flags;
  auto d_nucleation_temps_local = this->d_nucleation_temps;
  auto d_nucleation_sizes_local = this->d_nucleation_sizes;
  auto d_solid_front_coeffs_local = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;
  double dx_local = this->dx;

  // Debug counters for this kernel
  int debug_zero_G_local = 0;
  int debug_nonzero_G_local = 0;
  double debug_G_sum_local = 0.0;
  double debug_V_sum_local = 0.0;

  // Nucleation kernel: parallel over all local sites with event tracking
  // Race-free: each site only reads/writes itself
  int naccept_local = 0;
  Kokkos::parallel_reduce("NucleationKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count, int& zero_G_count, int& nonzero_G_count, double& G_sum, double& V_sum) {
      const int i = d_active_sites_local(idx);
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);
      int spin_i = d_spin_local(i);

      // Check if site is liquid and below liquidus (undercooled)
      if (active_flag_i == 2 && T_i < tl_local) {
        // Check if this spin can nucleate
        if (d_nucleation_flags_local(spin_i) == 1) {
          // Calculate undercooling
          double Tcool = tl_local - T_i;

          // Check if undercooling exceeds critical value for this spin
          if (Tcool >= d_nucleation_temps_local(spin_i)) {
            // Nucleate: transition to solid
            d_active_flag_local(i) = 3;  // SOLID
            count++;  // Track nucleation event

            // Set solid_d flag to prevent disappearance during smoothing
            // Negative value triggers smoothing countdown later
            d_solid_d_local(i) = -nrefine_local - 2;

            // Keep the assigned spin (already set during initialization)
            // d_spin_local(i) = spin_i;  // No-op, already has this value

            // Generate random unit quaternion for new grain orientation
            // Using Marsaglia method (same as melting kernel)
            auto generator = rand_pool_local.get_state();
            double x0, x1, x2, x3, r1, r2;
            do {
              x0 = 2.0 * generator.drand() - 1.0;
              x1 = 2.0 * generator.drand() - 1.0;
              r1 = x0*x0 + x1*x1;
            } while (r1 >= 1.0);

            do {
              x2 = 2.0 * generator.drand() - 1.0;
              x3 = 2.0 * generator.drand() - 1.0;
              r2 = x2*x2 + x3*x3;
            } while (r2 >= 1.0);

            double s = Kokkos::sqrt((1.0 - r1) / r2);
            d_q0_local(i) = x0;
            d_qx_local(i) = x1;
            d_qy_local(i) = x2 * s;
            d_qz_local(i) = x3 * s;
            rand_pool_local.free_state(generator);

            // Compute solidification velocity using polynomial: V = Σ coeffs[k] * Tcool^k
            double V_i = 0.0;
            double Tcool_power = 1.0;
            for (int k = solid_front_length_local - 1; k >= 0; k--) {
              V_i += d_solid_front_coeffs_local(k) * Tcool_power;
              Tcool_power *= Tcool;
            }
            d_V_local(i) = V_i;

            // Note: Gradient G is already computed by compute_gradients_kernel()

            // DEBUG: Check G and V values after nucleation
            double G_val = d_G_local(i);
            double V_val = d_V_local(i);
            if (G_val == 0.0) {
              zero_G_count++;
            } else {
              nonzero_G_count++;
              G_sum += G_val;
              V_sum += V_val;
            }
          }
          // No else branch - site keeps current values if not nucleating
        }
        // No else branch - site keeps current values if can't nucleate
      }
      // No else branch - site keeps current values if not liquid/undercooled
    }, naccept_local, debug_zero_G_local, debug_nonzero_G_local, debug_G_sum_local, debug_V_sum_local);

  // DEBUG: Accumulate kernel results into class members
  debug_nucleation_count += naccept_local;
  debug_zero_G_count += debug_zero_G_local;
  debug_nonzero_G_count += debug_nonzero_G_local;
  debug_G_sum += debug_G_sum_local;
  debug_V_sum += debug_V_sum_local;

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Phase 3.5: Particle flipper) kernel - grow nuclei to specified volume

   Grows newly nucleated grains by flipping neighboring liquid sites to
   match the nucleus orientation and grain ID. This avoids lattice-size
   dependency where nuclei start as single sites.

   Algorithm:
   1. Identify newly nucleated sites (solid_d == -nrefine-2)
   2. Calculate target number of sites from nucleation_sizes
   3. Flip liquid neighbors in randomized order until target reached
   4. Copy nucleus spin, quaternion, G, and V to flipped sites
   5. Set flipped sites to solid_d = -nrefine-3 (ready for smoothing)

   Note: Simplified parallel version - flips immediate neighbors only.
   CPU version uses recursive shell-based expansion which is complex
   to parallelize. This version achieves similar physics with simpler logic.
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::particle_flipper_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (atomic CAS makes neighbor writes race-safe)
  auto d_active_sites_local = d_active_sites;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_neighbor_d = k_neighbor.view_device();
  auto k_numneigh_d = k_numneigh.view_device();
  auto d_G_local = d_G;
  auto d_grad_x_local = d_grad_x;
  auto d_grad_y_local = d_grad_y;
  auto d_grad_z_local = d_grad_z;
  auto d_V_local = d_V;
  auto d_nucleation_sizes_local = d_nucleation_sizes;
  auto rand_pool_local = *rand_pool;

  int nrefine_local = nrefine;
  double dx_local = dx;

  // Parallel over the active-front list - each nucleus flips its neighbors with
  // event tracking. Race-safe: uses atomic CAS to claim neighbors.
  int naccept_local = 0;
  Kokkos::parallel_reduce("ParticleFlipperKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count) {
      const int i = d_active_sites_local(idx);
      // Only process newly nucleated sites (marked with solid_d = -nrefine-2)
      double solid_d_i = d_solid_d_local(i);
      int active_flag_i = d_active_flag_local(i);

      // Check if this is a newly nucleated site
      if (active_flag_i != 3 || solid_d_i != -nrefine_local - 2) {
        // Not a nucleus - site keeps its current values
        return;
      }

      // This is a newly nucleated site - attempt to grow it
      int spin_i = d_spin_local(i);
      double q_i[4] = {d_q0_local(i), d_qx_local(i), d_qy_local(i), d_qz_local(i)};
      double G_i = d_G_local(i);
      double V_i = d_V_local(i);

      // Calculate target number of sites from nucleus volume
      // nucleation_sizes is in m^3, dx^3 is voxel volume
      double nucleus_volume = d_nucleation_sizes_local(spin_i);
      double voxel_volume = dx_local * dx_local * dx_local;
      int target_sites = static_cast<int>(Kokkos::round(nucleus_volume / voxel_volume));

      // Nucleus itself counts as 1 site
      if (target_sites <= 1) {
        // No neighbors to flip - just update solid_d to ready state
        d_solid_d_local(i) = -nrefine_local - 3;  // Ready for smoothing
        count++;  // Track nucleation event
        return;
      }

      // Build list of liquid neighbors that can be flipped
      int nn = k_numneigh_d(i);
      constexpr int MAX_NEIGHBORS = 26;  // sc/26n lattice assumption
      int liquid_neighbors[MAX_NEIGHBORS];
      int num_liquid = 0;

      for (int j = 0; j < nn && j < MAX_NEIGHBORS; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_local(nbr);

        // Only flip liquid neighbors (active_flag == 2)
        if (nbr_flag == 2) {
          liquid_neighbors[num_liquid++] = nbr;
        }
      }

      // Shuffle the liquid neighbor list using Fisher-Yates
      auto generator = rand_pool_local.get_state();
      for (int j = num_liquid - 1; j > 0; j--) {
        int k = static_cast<int>(generator.drand() * (j + 1));
        if (k > j) k = j;  // Clamp to valid range [0, j] to prevent out-of-bounds
        // Swap
        int temp = liquid_neighbors[j];
        liquid_neighbors[j] = liquid_neighbors[k];
        liquid_neighbors[k] = temp;
      }

      // Flip up to (target_sites - 1) neighbors
      // Note: In parallel execution, some neighbors may be claimed by other nuclei
      // This is acceptable - results in statistical variation in nucleus sizes
      int sites_to_flip = Kokkos::min(target_sites - 1, num_liquid);

      for (int j = 0; j < sites_to_flip; j++) {
        int nbr = liquid_neighbors[j];

        // Attempt to flip this neighbor using atomic compare-and-swap
        // to prevent race conditions when multiple nuclei compete for same neighbor
        int expected = 2;  // LIQUID
        int desired = 3;   // SOLID (claimed)

        // Atomic CAS: only succeeds if neighbor is still liquid
        int old_flag = Kokkos::atomic_compare_exchange(&d_active_flag_local(nbr), expected, desired);

        if (old_flag == 2) {
          // We successfully claimed this neighbor - flip it to match nucleus
          d_spin_local(nbr) = spin_i;
          d_q0_local(nbr) = q_i[0];
          d_qx_local(nbr) = q_i[1];
          d_qy_local(nbr) = q_i[2];
          d_qz_local(nbr) = q_i[3];
          // active_flag already set to SOLID by CAS above
          d_solid_d_local(nbr) = -nrefine_local - 3;  // Ready for smoothing

          // Copy thermal properties from nucleus
          d_G_local(nbr) = G_i;
          d_V_local(nbr) = V_i;
          d_grad_x_local(nbr) = d_grad_x_local(i);
          d_grad_y_local(nbr) = d_grad_y_local(i);
          d_grad_z_local(nbr) = d_grad_z_local(i);
        }
        // If old_flag != 2, another nucleus already claimed it - skip
      }

      rand_pool_local.free_state(generator);

      // Update nucleus solid_d to ready state
      d_solid_d_local(i) = -nrefine_local - 3;  // Ready for smoothing

      // Track nucleus growth event
      count++;
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Solidification kernel: Epitaxial growth with texture weighting
   Phase 3.3: Liquid sites captured by solid neighbors based on texture
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::solidification_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (no double buffering needed for self-only writes)
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = k_T.view_device();
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_numneigh_d = k_numneigh.view_device();
  auto k_neighbor_d = k_neighbor.view_device();

  // Cached, layout-correct device copy of site coordinates (used for on-demand
  // gradient computation in the kernel).
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;

  // Capture parameters
  double tl_local = this->tl;
  double ts_local = this->ts;
  double dt_local = this->dt;
  int nrefine_local = this->nrefine;
  double dx_local = this->dx;
  auto d_solid_front_coeffs_local = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  auto d_texture_params_local = this->d_texture_params;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;
  auto d_grad_x_local = this->d_grad_x;
  auto d_grad_y_local = this->d_grad_y;
  auto d_grad_z_local = this->d_grad_z;
  auto rand_pool_local = *rand_pool;

  // Misorientation parameters (Phase 3.6)
  double max_misorient_local = this->max_misorient;
  double t_cool_max_local = this->t_cool_max;
  double mis_thresh_local = this->mis_thresh;
  double misorient_alpha_local = this->misorient_alpha;
  const int misori_mode_local = this->misorientation_target_mode;
  int nspins_local = this->nspins;

  // Solid-front reach threshold: a neighbor becomes eligible once the front
  // distance solid_d has advanced past the farthest SC_26N neighbor (the cube
  // corner, sqrt(3) lattice units). This matches the CPU reference, which tests
  // the constant neigh_dist[2] == sqrt3*dx for every neighbor
  // (app_additive_texture.cpp:1143) -- NOT a per-neighbor-slot distance. Using a
  // slot-indexed table here was wrong twice over: it deviated from the CPU even
  // for interior sites, and for low-coordination boundary sites (compacted
  // neighbor list) the slot did not correspond to the neighbor's real offset.
  constexpr double solid_reach = 1.7320508075688772;  // sqrt(3), lattice units

  // Debug counters for this kernel
  int debug_zero_G_local = 0;
  int debug_nonzero_G_local = 0;
  double debug_G_sum_local = 0.0;
  double debug_V_sum_local = 0.0;

  // Solidification kernel: parallel over all local sites with event tracking
  // Race-free: each site only reads/writes itself
  int naccept_local = 0;
  Kokkos::parallel_reduce("SolidificationKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count, int& zero_G_count, int& nonzero_G_count, double& G_sum, double& V_sum) {
      const int i = d_active_sites_local(idx);
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);

      // Check if site is liquid and below liquidus
      if (active_flag_i == 2 && T_i < tl_local) {
        double Tcool = tl_local - T_i;
        double solid_d_i = d_solid_d_local(i);

        // Advance solid front distance: solid_d += V(Tcool) * dt
        double V_advance = 0.0;
        double Tcool_power = 1.0;
        for (int k = solid_front_length_local - 1; k >= 0; k--) {
          V_advance += d_solid_front_coeffs_local(k) * Tcool_power;
          Tcool_power *= Tcool;
        }
        solid_d_i += V_advance * dt_local;

        // Find solid neighbors within reach and calculate texture weights
        int nn = k_numneigh_d(i);
        int num_solid_neighbors = 0;
        double cumulative_weight[26];  // Max 26 neighbors
        int solid_neighbor_idx[26];

        // On-demand gradient computation: only compute if we might have solid neighbors
        // This is the key optimization - gradients computed only when needed
        double grad_dir[3] = {0.0, 0.0, 0.0};
        double grad_magnitude = 0.0;
        bool gradient_computed = false;

        double cumulative_sum = 0.0;
        for (int j = 0; j < nn && j < 26; j++) {
          int nbr = k_neighbor_d(i, j);
          int nbr_flag = d_active_flag_local(nbr);

          // Check if neighbor is solid (active_flag == 1 or 3) and within reach
          if ((nbr_flag == 1 || nbr_flag == 3) && solid_reach * dx_local <= solid_d_i) {
            // Compute gradient on-demand (only once, on first solid neighbor)
            if (!gradient_computed) {
              // Use ThermalOps to compute temperature gradient at site i
              ThermalOps::compute_temperature_gradient(
                i, d_T_local, d_active_flag_local, k_numneigh_d, k_neighbor_d, k_xyz_d, dx_local,
                grad_dir, &grad_magnitude);

              // Store gradient magnitude and components for later use
              d_G_local(i) = grad_magnitude;
              d_grad_x_local(i) = grad_dir[0];
              d_grad_y_local(i) = grad_dir[1];
              d_grad_z_local(i) = grad_dir[2];
              gradient_computed = true;
            }

            // Get neighbor quaternion
            double q_nbr[4] = {
              d_q0_local(nbr),
              d_qx_local(nbr),
              d_qy_local(nbr),
              d_qz_local(nbr)
            };

            // Compute texture weight: mobility = c1 + c2 * cos(c3 * theta)
            // where theta is angle between crystal <100> and gradient
            double c1 = d_texture_params_local(0);
            double c2 = d_texture_params_local(1);
            double c3 = d_texture_params_local(2);

            // Use QuaternionOps from parent class to get cosine of min angle
            // This finds the minimum angle between any <100> direction and gradient
            double cos_min_angle = QuaternionOps::get_cosine_angle_q_u(q_nbr, grad_dir);
            double theta = Kokkos::acos(cos_min_angle);
            double mobility = c1 + c2 * Kokkos::cos(c3 * theta);

            // Accumulate weight (must be positive)
            if (mobility > 0.0) {
              cumulative_sum += mobility;
              cumulative_weight[num_solid_neighbors] = cumulative_sum;
              solid_neighbor_idx[num_solid_neighbors] = nbr;
              num_solid_neighbors++;
            }
          }
        }

        // If we have solid neighbors, perform stochastic selection
        if (num_solid_neighbors > 0) {
          // Get RNG state
          auto generator = rand_pool_local.get_state();
          double rand_val = generator.drand() * cumulative_sum;

          // Find selected neighbor using binary-like search
          int selected_idx = 0;
          for (int k = 0; k < num_solid_neighbors; k++) {
            if (rand_val <= cumulative_weight[k]) {
              selected_idx = k;
              break;
            }
          }

          int selected_nbr = solid_neighbor_idx[selected_idx];

          // Solidify: copy grain ID and quaternion from selected neighbor
          int new_spin = d_spin_local(selected_nbr);
          double q_nbr_inherited[4] = {
            d_q0_local(selected_nbr),
            d_qx_local(selected_nbr),
            d_qy_local(selected_nbr),
            d_qz_local(selected_nbr)
          };

          // Phase 3.6: Apply misorientation rotation toward gradient
          // Rotate orientation based on cooling rate and gradient direction
          if (max_misorient_local > 0.0) {
            // Calculate misorientation angle from cooling rate
            // Exponential function: grows slowly initially, then rapidly in last ~10%
            // Clamp normalized_temp to [0, 1] to prevent exponential overflow
            double normalized_temp = Kokkos::fmin(Tcool / t_cool_max_local, 1.0);
            double mis_angle = max_misorient_local *
                               (Kokkos::exp(misorient_alpha_local * normalized_temp) - 1.0) /
                               (Kokkos::exp(misorient_alpha_local) - 1.0) * MY_PI / 180.0;

            // Use gradient direction from on-demand computation above
            // (grad_dir already computed when first solid neighbor was found)

            // Normalize the inherited quaternion (should already be normalized, but ensure it)
            double q_mag = Kokkos::sqrt(q_nbr_inherited[0] * q_nbr_inherited[0] +
                                        q_nbr_inherited[1] * q_nbr_inherited[1] +
                                        q_nbr_inherited[2] * q_nbr_inherited[2] +
                                        q_nbr_inherited[3] * q_nbr_inherited[3]);
            if (q_mag > 1.0e-15) {
              q_nbr_inherited[0] /= q_mag;
              q_nbr_inherited[1] /= q_mag;
              q_nbr_inherited[2] /= q_mag;
              q_nbr_inherited[3] /= q_mag;
            }

            // Perturb the inherited orientation per the misorientation_target
            // mode. random_rotation rotates by mis_angle about a uniformly
            // random axis (composition, no toward-target step); gradient and
            // random rotate the orientation TOWARD a target direction -- the
            // thermal-gradient direction, or a uniform random unit vector (the
            // kk default). Random unit vector uses the same z/phi sampling as
            // AppAdditiveTexture::sample_random_unit_vector.
            double q_rotated[4];
            if (misori_mode_local == MISORI_TARGET_RANDOM_ROTATION) {
              double z = 2.0 * generator.drand() - 1.0;
              double phi = 2.0 * MY_PI * generator.drand();
              double radial = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - z * z));
              double half = 0.5 * mis_angle;
              double s = Kokkos::sin(half);
              double dq[4] = {Kokkos::cos(half),
                              s * radial * Kokkos::cos(phi),
                              s * radial * Kokkos::sin(phi),
                              s * z};
              QuaternionOps::composition(dq, q_nbr_inherited, q_rotated);
              QuaternionOps::normalize(q_rotated);
            } else {
              double target_dir[3];
              if (misori_mode_local == MISORI_TARGET_GRADIENT) {
                target_dir[0] = grad_dir[0];
                target_dir[1] = grad_dir[1];
                target_dir[2] = grad_dir[2];
              } else { // MISORI_TARGET_RANDOM (default)
                double z = 2.0 * generator.drand() - 1.0;
                double phi = 2.0 * MY_PI * generator.drand();
                double radial = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - z * z));
                target_dir[0] = radial * Kokkos::cos(phi);
                target_dir[1] = radial * Kokkos::sin(phi);
                target_dir[2] = z;
              }
              QuaternionOps::rotate_q_towards_u(q_nbr_inherited, target_dir, mis_angle, q_rotated);
            }

            // Compute rotation angle between original and rotated quaternion
            // This determines if we create a new grain
            double dot_product = q_nbr_inherited[0] * q_rotated[0] +
                                 q_nbr_inherited[1] * q_rotated[1] +
                                 q_nbr_inherited[2] * q_rotated[2] +
                                 q_nbr_inherited[3] * q_rotated[3];
            dot_product = Kokkos::fabs(dot_product);  // Handle quaternion double cover
            dot_product = Kokkos::fmin(1.0, dot_product);  // Clamp to avoid numerical issues
            double direct_angle = 2.0 * Kokkos::acos(dot_product);

            // If rotation exceeds threshold, create new grain with random spin
            if (direct_angle >= mis_thresh_local) {
              new_spin = static_cast<int>(nspins_local * generator.drand());
            }

            // Use the rotated quaternion
            d_q0_local(i) = q_rotated[0];
            d_qx_local(i) = q_rotated[1];
            d_qy_local(i) = q_rotated[2];
            d_qz_local(i) = q_rotated[3];
          } else {
            // No misorientation - use neighbor's orientation as-is
            d_q0_local(i) = q_nbr_inherited[0];
            d_qx_local(i) = q_nbr_inherited[1];
            d_qy_local(i) = q_nbr_inherited[2];
            d_qz_local(i) = q_nbr_inherited[3];
          }

          // Assign grain ID (may be new grain if rotation exceeded threshold)
          d_spin_local(i) = new_spin;

          // Set to solid state
          d_active_flag_local(i) = 3;  // SOLID
          count++;  // Track solidification event

          // Set solid_d flag to trigger smoothing countdown
          d_solid_d_local(i) = -1.0;

          // Record solidification parameters
          d_V_local(i) = V_advance;

          // DEBUG: Check G and V values after solidification
          double G_val = d_G_local(i);
          double V_val = d_V_local(i);
          if (G_val == 0.0) {
            zero_G_count++;
          } else {
            nonzero_G_count++;
            G_sum += G_val;
            V_sum += V_val;
          }

          // Return RNG state
          rand_pool_local.free_state(generator);
        } else {
          // No solid neighbors within reach - update advanced distance only
          d_solid_d_local(i) = solid_d_i;
        }
      }
      // No else branch - site keeps current values if not liquid/undercooled
    }, naccept_local, debug_zero_G_local, debug_nonzero_G_local, debug_G_sum_local, debug_V_sum_local);

  // DEBUG: Accumulate kernel results into class members
  debug_solidification_count += naccept_local;
  debug_zero_G_count += debug_zero_G_local;
  debug_nonzero_G_count += debug_nonzero_G_local;
  debug_G_sum += debug_G_sum_local;
  debug_V_sum += debug_V_sum_local;

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Phase 3.4: Smoothing kernel) - Potts MC for grain boundary refinement

   Performs Monte Carlo sweeps on newly solidified sites (solid_d < 0) to
   smooth grain boundaries. Uses texture-weighted neighbor selection and
   Metropolis acceptance based on quaternion misorientation energy.

   Algorithm:
   1. For sites with solid_d ∈ (-nrefine-1, 0):
      - Build list of solid/active neighbors
      - Calculate texture weight for each neighbor using gradient alignment
      - Stochastic selection weighted by texture
      - Metropolis acceptance based on energy change
      - If accepted: copy neighbor's spin and quaternion
   2. Increment solid_d counter after sweep
   3. Mark as ACTIVE (flag=1) when solid_d reaches 0
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::smoothing_kernel()
{
  // Get device views - simple Views (color-based sweeping makes this race-safe)
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_neighbor_d = k_neighbor.view_device();
  auto k_numneigh_d = k_numneigh.view_device();
  auto d_grad_x_local = d_grad_x;
  auto d_grad_y_local = d_grad_y;
  auto d_grad_z_local = d_grad_z;
  auto d_symmetries_local = d_symmetries;
  auto rand_pool_local = *rand_pool;

  int nlocal_local = nlocal;
  int nrefine_local = nrefine;
  double theta_cut_local = theta_cut;
  double temperature_local = temperature;
  double t_inverse_local = t_inverse;
  int num_symmetries_local = num_symmetries;

  // Texture parameters
  double c1_local = c1;
  double c2_local = c2;
  double c3_local = c3;

  // Parallel sweep over all local sites with acceptance tracking
  // Race-free: color-based sweeping ensures no conflicts
  int naccept_local = 0;
  Kokkos::parallel_reduce("SmoothingKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      // Get current state
      int active_flag_i = d_active_flag_local(i);
      double solid_d_i = d_solid_d_local(i);

      // Only smooth sites in the refinement range: (-nrefine-1, 0)
      if (solid_d_i >= 0.0 || solid_d_i <= -nrefine_local - 1) {
        // Not eligible for smoothing - increment solid_d if in range
        if (solid_d_i < 0.0 && solid_d_i > -nrefine_local - 1) {
          d_solid_d_local(i) = solid_d_i + 1.0;
          // Reset solid_d to 0 when smoothing completes
          if (d_solid_d_local(i) >= 0.0) {
            d_solid_d_local(i) = 0.0;
          }
        }
        return;
      }

      // Site is eligible for smoothing
      int spin_i = d_spin_local(i);
      double q_i[4] = {d_q0_local(i), d_qx_local(i), d_qy_local(i), d_qz_local(i)};

      // Get thermal gradient direction at this site
      double grad_norm = Kokkos::sqrt(
        d_grad_x_local(i) * d_grad_x_local(i) +
        d_grad_y_local(i) * d_grad_y_local(i) +
        d_grad_z_local(i) * d_grad_z_local(i)
      );

      double grad_dir[3];
      if (grad_norm > 1e-12) {
        grad_dir[0] = d_grad_x_local(i) / grad_norm;
        grad_dir[1] = d_grad_y_local(i) / grad_norm;
        grad_dir[2] = d_grad_z_local(i) / grad_norm;
      } else {
        grad_dir[0] = 0.0;
        grad_dir[1] = 0.0;
        grad_dir[2] = 0.0;
      }

      // Build list of eligible neighbors with texture weights
      int nn = k_numneigh_d(i);
      int eligible_neighbors[26];  // Max 26 neighbors for sc lattice
      double cumulative_weight[26];
      double cumulative_sum = 0.0;
      int num_eligible = 0;

      for (int j = 0; j < nn && j < 26; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_local(nbr);

        // Only consider solid or active neighbors (not liquid, molten, or void)
        if (nbr_flag != 1 && nbr_flag != 3) continue;

        // Skip neighbors with same spin (no energy change possible)
        int nbr_spin = d_spin_local(nbr);
        if (nbr_spin == spin_i) continue;

        // Calculate texture weight: mobility = c1 + c2 * cos(c3 * theta)
        // where theta is angle between neighbor's orientation and gradient
        double q_nbr[4] = {
          d_q0_local(nbr), d_qx_local(nbr),
          d_qy_local(nbr), d_qz_local(nbr)
        };

        double cos_min_angle = QuaternionOps::get_cosine_angle_q_u(q_nbr, grad_dir);
        double theta = Kokkos::acos(Kokkos::fmax(-1.0, Kokkos::fmin(1.0, cos_min_angle)));
        double mobility = c1_local + c2_local * Kokkos::cos(c3_local * theta);

        // Only include neighbors with positive mobility
        if (mobility > 0.0) {
          cumulative_sum += mobility;
          cumulative_weight[num_eligible] = cumulative_sum;
          eligible_neighbors[num_eligible] = nbr;
          num_eligible++;
        }
      }

      // If no eligible neighbors, keep current state
      if (num_eligible == 0) {
        d_spin_local(i) = spin_i;
        d_q0_local(i) = q_i[0];
        d_qx_local(i) = q_i[1];
        d_qy_local(i) = q_i[2];
        d_qz_local(i) = q_i[3];
        d_active_flag_local(i) = active_flag_i;
        d_solid_d_local(i) = solid_d_i + 1.0;
        // CRITICAL FIX: Reset solid_d to 0 when smoothing completes
        // Keep active_flag unchanged (should remain SOLID=3)
        // CPU version does NOT change active_flag after smoothing
        if (d_solid_d_local(i) >= 0.0) {
          d_solid_d_local(i) = 0.0;
        }
        return;
      }

      // Stochastic selection of neighbor weighted by texture
      auto generator = rand_pool_local.get_state();
      double rand_val = generator.drand() * cumulative_sum;

      int selected_idx = 0;
      for (int k = 0; k < num_eligible; k++) {
        if (rand_val <= cumulative_weight[k]) {
          selected_idx = k;
          break;
        }
      }

      int selected_nbr = eligible_neighbors[selected_idx];
      int new_spin = d_spin_local(selected_nbr);
      double q_new[4] = {
        d_q0_local(selected_nbr),
        d_qx_local(selected_nbr),
        d_qy_local(selected_nbr),
        d_qz_local(selected_nbr)
      };

      // Compute energy change using quaternion disorientation
      // Initial energy: sum over neighbors
      double energy_initial = 0.0;
      for (int j = 0; j < nn && j < 26; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_local(nbr);

        // Skip void and non-solid neighbors
        if (nbr_flag == 5 || (nbr_flag != 3 && nbr_flag != 1)) continue;

        double q_nbr[4] = {
          d_q0_local(nbr), d_qx_local(nbr),
          d_qy_local(nbr), d_qz_local(nbr)
        };

        double di = QuaternionOps::compute_disorientation(
          d_symmetries_local.data(), num_symmetries_local, q_i, q_nbr
        );

        double ratio = di / theta_cut_local;
        if (ratio <= 0.0) {
          continue;
        } else if (ratio < 1.0) {
          energy_initial += ratio * (1.0 - Kokkos::log(ratio));
        } else {
          energy_initial += 1.0;
        }
      }
      energy_initial *= 0.5;  // Each site carries half the boundary energy

      // Final energy: sum with new orientation
      double energy_final = 0.0;
      for (int j = 0; j < nn && j < 26; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_local(nbr);

        if (nbr_flag == 5 || (nbr_flag != 3 && nbr_flag != 1)) continue;

        double q_nbr[4] = {
          d_q0_local(nbr), d_qx_local(nbr),
          d_qy_local(nbr), d_qz_local(nbr)
        };

        double di = QuaternionOps::compute_disorientation(
          d_symmetries_local.data(), num_symmetries_local, q_new, q_nbr
        );

        double ratio = di / theta_cut_local;
        if (ratio <= 0.0) {
          continue;
        } else if (ratio < 1.0) {
          energy_final += ratio * (1.0 - Kokkos::log(ratio));
        } else {
          energy_final += 1.0;
        }
      }
      energy_final *= 0.5;

      // Metropolis acceptance criterion
      bool accept = false;
      if (energy_final <= energy_initial) {
        accept = true;  // Always accept energy-lowering moves
      } else if (temperature_local > 0.0) {
        double prob = Kokkos::exp((energy_initial - energy_final) * t_inverse_local);
        double rand_accept = generator.drand();
        if (rand_accept < prob) {
          accept = true;
        }
      }
      // If temperature == 0, only accept downhill moves (already handled)

      // Apply move if accepted
      if (accept) {
        d_spin_local(i) = new_spin;
        d_q0_local(i) = q_new[0];
        d_qx_local(i) = q_new[1];
        d_qy_local(i) = q_new[2];
        d_qz_local(i) = q_new[3];
      } else {
        // Keep current state
        d_spin_local(i) = spin_i;
        d_q0_local(i) = q_i[0];
        d_qx_local(i) = q_i[1];
        d_qy_local(i) = q_i[2];
        d_qz_local(i) = q_i[3];
      }

      // Increment solid_d counter (moves toward 0)
      d_solid_d_local(i) = solid_d_i + 1.0;

      // CRITICAL FIX: Keep active_flag = 3 (SOLID) when smoothing completes
      // CPU version does NOT change active_flag after smoothing
      // Sites remain solid (active_flag=3) permanently until melted
      if (d_solid_d_local(i) >= 0.0) {
        d_solid_d_local(i) = 0.0;  // Reset counter
      }
      d_active_flag_local(i) = active_flag_i;  // Keep current state (should be 3=SOLID)

      // Track acceptance for stats output
      if (accept) count++;

      rand_pool_local.free_state(generator);
    }, naccept_local);

  // Fence removed - single fence at end of app_update() handles all kernels

  // Return count of accepted moves
  return naccept_local;
}

/* ----------------------------------------------------------------------
   PHASE 1 OPTIMIZATION: Work-efficient smoothing kernel

   Processes ONLY sites in the compact smoothing list (not all 2.99M sites).
   This is a direct replacement for smoothing_kernel() but operates on
   a filtered list of ~100-1000 sites instead of checking all sites.

   Expected performance: 0.4 ms vs 1.2 ms (3× faster)
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::smoothing_kernel_efficient(
    Kokkos::View<int*> d_smoothing_sites, int n_sites)
{
  // Get device views - simple Views (color-based sweeping makes this race-safe)
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_neighbor_d = k_neighbor.view_device();
  auto k_numneigh_d = k_numneigh.view_device();
  auto rand_pool_local = *rand_pool;

  double temperature_local = temperature;
  double t_inverse_local = t_inverse;
  const bool greedy_local = smooth_greedy_multiproposal_enabled;

  // Parallel sweep over ONLY smoothing sites (not all sites!)
  int naccept_local = 0;
  Kokkos::parallel_reduce("SmoothingKernelEfficient", n_sites,
    KOKKOS_LAMBDA(int idx, int& count) {
      // Get actual site index from compact list
      int i = d_smoothing_sites(idx);

      // All sites in this list need smoothing - no branching needed!
      const int active_flag_i = d_active_flag_local(i);
      const int spin_i = d_spin_local(i);
      const double solid_d_i = d_solid_d_local(i);

      // CPU-aligned Potts boundary smoothing (AppAdditiveTexture::smooth_site).
      // Energy = number of SOLID (active_flag==3) neighbors with a different spin
      // (site_energy_smooth). Candidates = distinct neighbor spins != self.
      // Proposal: greedy steepest-descent (evaluate every candidate, keep the
      // bond-energy minimizer; first-encounter tie-break -> deterministic) or a
      // random single proposal. Boltzmann accept. On accept the orientation
      // becomes the sign-aligned average of the selected grain's neighbor
      // quaternions (get_average_neighbor_quaternion). NB: smoothing sites are
      // swept in parallel here (vs the CPU's sequential per-site sweep), so the
      // microstructure is statistically -- not bit -- equal to the CPU even at
      // temperature 0.
      const int nn = k_numneigh_d(i);

      // Distinct neighbor spins != self.
      int unique_spins[27];
      int nevent = 0;
      for (int n = 0; n < nn; n++) {
        const int v = d_spin_local(k_neighbor_d(i, n));
        if (v == spin_i) continue;
        bool found = false;
        for (int k = 0; k < nevent; k++)
          if (unique_spins[k] == v) { found = true; break; }
        if (!found && nevent < 27) unique_spins[nevent++] = v;
      }

      if (nevent > 0) {
        // einitial: bond energy of the current spin.
        int einitial = 0;
        for (int n = 0; n < nn; n++) {
          const int nbr = k_neighbor_d(i, n);
          if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != spin_i) einitial++;
        }

        auto generator = rand_pool_local.get_state();

        int new_spin, efinal;
        if (greedy_local) {
          new_spin = unique_spins[0];
          efinal = 0;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d(i, n);
            if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != new_spin) efinal++;
          }
          for (int c = 1; c < nevent; c++) {
            const int cand = unique_spins[c];
            int e = 0;
            for (int n = 0; n < nn; n++) {
              const int nbr = k_neighbor_d(i, n);
              if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != cand) e++;
            }
            if (e < efinal) { efinal = e; new_spin = cand; }  // strict -> first-encounter
          }
        } else {
          int iran = (int)(nevent * generator.drand());
          if (iran >= nevent) iran = nevent - 1;
          new_spin = unique_spins[iran];
          efinal = 0;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d(i, n);
            if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != new_spin) efinal++;
          }
        }

        // Boltzmann accept (matches smooth_site).
        bool accept;
        if (efinal <= einitial) accept = true;
        else if (temperature_local == 0.0) accept = false;
        else accept = (generator.drand() <=
                       Kokkos::exp((double)(einitial - efinal) * t_inverse_local));

        rand_pool_local.free_state(generator);

        if (accept) {
          // Orientation = sign-aligned average of neighbors with new_spin.
          double s0 = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
          double r0 = 0.0, rx = 0.0, ry = 0.0, rz = 0.0;
          int cq = 0; bool first = false;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d(i, n);
            if (d_spin_local(nbr) != new_spin) continue;
            double n0 = d_q0_local(nbr), nx = d_qx_local(nbr),
                   ny = d_qy_local(nbr), nz = d_qz_local(nbr);
            if (!first) { r0 = n0; rx = nx; ry = ny; rz = nz; first = true; }
            else if (r0*n0 + rx*nx + ry*ny + rz*nz < 0.0) {
              n0 = -n0; nx = -nx; ny = -ny; nz = -nz;  // q and -q are the same orientation
            }
            s0 += n0; sx += nx; sy += ny; sz += nz; cq++;
          }
          if (cq > 0) {
            double a0 = s0/cq, ax = sx/cq, ay = sy/cq, az = sz/cq;
            const double nrm = Kokkos::sqrt(a0*a0 + ax*ax + ay*ay + az*az);
            if (nrm > 0.0) { a0/=nrm; ax/=nrm; ay/=nrm; az/=nrm; }
            d_spin_local(i) = new_spin;
            d_q0_local(i) = a0; d_qx_local(i) = ax;
            d_qy_local(i) = ay; d_qz_local(i) = az;
            count++;
          }
        }
        // Reject / no-grain: spin and quaternion already hold the original
        // values (this kernel has not written site i up to here).
      }

      // Advance the smoothing window (solid_d) and keep the solid state;
      // active_flag is left unchanged, matching the CPU.
      d_solid_d_local(i) = solid_d_i + 1.0;
      if (d_solid_d_local(i) >= 0.0) d_solid_d_local(i) = 0.0;
      d_active_flag_local(i) = active_flag_i;
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   App update - main time integration loop
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::app_update(double dt_sweep)
{
  // PHASE 3.1-3.2: Melting and nucleation kernels
  // PHASE 3.3-3.4: Will add solidification and smoothing

  // Timing instrumentation
  static int step_count = 0;
  static double t_temp_update = 0.0, t_temp_sync = 0.0;
  static double t_melting = 0.0, t_nucleation = 0.0, t_particle = 0.0;
  static double t_solidification = 0.0, t_smoothing = 0.0, t_sync_host = 0.0;
  double t_start, t_end;
  step_count++;

  // 1. Update temperature + fast-forward.
  //
  // Rosenthal (STANDARD/ANISOTROPIC): evaluate the per-site temperature on the
  // DEVICE -- the host only advances the laser pose (cheap) and we fill k_T
  // directly on the GPU, then copy back to host T for the diagnostic and
  // host-side consumers. This replaces the dominant host exp/sqrt per-site
  // loop. Fast-forward uses a global max temperature + ThermalManager's
  // path-geometry predictor.
  //
  // Other sources (HDF5 / Moser / Rosenthal KEYHOLE): host path via
  // ThermalManager::advance(), which fills host T[]; we then sync to device.
  refresh_thermal_host();
  t_start = MPI_Wtime();
  bool device_temp = false;
  device_active_list_fresh = false;  // set true only by a fill that built d_active_sites (HDF5)
  if (thermal->enabled()) {
    RosenthalDevicePlan rplan = thermal->rosenthal_device_plan(time);
    MoserDevicePlan mplan;
    if (!rplan.valid) mplan = thermal->moser_device_plan(time);

    // The device->host copy-back is deferred (host_T_stale) and done only before
    // a host consumer reads T -- powder activation below, or the d6 diagnostic
    // at an output step (end of app_update). The global peak temperature for the
    // fast-forward decision is reduced on the device, so no host T is needed.
    if (rplan.valid) {
      // ---- DEVICE Rosenthal path ----
      device_temp = true;
      fill_temperature_rosenthal_device(rplan);
      host_T_stale = true;
      double global_max = device_global_max_temperature();
      ThermalManager::ThermalStep s =
        thermal->rosenthal_fast_forward(time, stoptime, global_max);
      if (s.end_run) { time = s.new_time; sync_temperature_to_host(); return; }
      if (s.new_time > time) {
        time = s.new_time;
        fill_temperature_rosenthal_device(thermal->rosenthal_device_plan(time));
        host_T_stale = true;
        if (s.activate_powder) {
          activate_powder_sites();
          last_powder_activation_time = time;
        }
      }
    } else if (mplan.valid) {
      // ---- DEVICE Moser path ----
      // ThermalManager runs the per-pass pause/reset + fast-forward decisions on
      // the host (they rebuild the scan from temperature feedback) and calls back
      // here to do every per-site Green's-function fill on the GPU. The filler
      // fills k_T at a given time and returns the device-reduced global max; it
      // also re-uploads segment/fluctuation Views when a fired pass bumps the
      // scan version. On return the device field is filled at s.new_time.
      device_temp = true;
      thermal->set_moser_device_filler([this](double t) -> double {
        fill_temperature_moser_device(thermal->moser_device_plan(t));
        host_T_stale = true;
        return device_global_max_temperature();
      });
      ThermalManager::ThermalStep s = thermal->moser_advance_device(time, stoptime);
      if (s.end_run) { time = s.new_time; sync_temperature_to_host(); return; }
      if (s.new_time > time) {
        time = s.new_time;
        if (s.activate_powder) {
          activate_powder_sites();
          last_powder_activation_time = time;
        }
      }
    } else if (thermal->hdf5_device_eligible()) {
      // ---- DEVICE HDF5 (unstructured-mesh) path ----
      // The host still does the expensive HDF5 work (reads, layer load, element
      // search/cache, nodal precompute) inside hdf5_device_plan(); the GPU does
      // the per-site barycentric interpolation into k_T for all nall sites
      // (owned + ghost, positional). The filler fills the device field at a
      // given time and returns the device-reduced global max for fast-forward.
      device_temp = true;
      thermal->set_hdf5_device_filler([this](double t) -> double {
        fill_temperature_hdf5_device(thermal->hdf5_device_plan(t));
        host_T_stale = true;
        return device_global_max_temperature();
      });
      ThermalManager::ThermalStep s = thermal->hdf5_advance_device(time, stoptime);
      if (s.end_run) { time = s.new_time; sync_temperature_to_host(); return; }
      if (s.new_time > time) {
        time = s.new_time;
        if (s.activate_powder) {
          activate_powder_sites();
          last_powder_activation_time = time;
        }
      }
      // Ghost temperatures need no special handling: the element cache covers
      // nall (owned + ghost via the chunk-load pad), so the device fill already
      // wrote correct ghost k_T positionally. A ghost equals the owning rank's
      // value by construction (same global field, same point), so the field is
      // fully device-resident -- no host T round-trip, no T ghost comm.
    } else {
      // ---- HOST path (Rosenthal-keyhole / temperature_smooth / other) ----
      ThermalManager::ThermalStep s = thermal->advance(time, stoptime);
      if (s.end_run) { time = s.new_time; return; }
      if (s.new_time > time) {
        time = s.new_time;
        if (s.activate_powder) {
          activate_powder_sites();
          last_powder_activation_time = time;
        }
      }
    }
  }
  t_end = MPI_Wtime();
  t_temp_update += (t_end - t_start);

  // 2. Sync temperature to device (host path only; the device path already
  //    wrote k_T on the GPU).
  t_start = MPI_Wtime();
  if (!device_temp) sync_temperature_to_device();
  t_end = MPI_Wtime();
  t_temp_sync += (t_end - t_start);

  // 3. Gradient computation moved to on-demand inside solidification kernel
  // (Removed global compute_gradients_kernel() - gradients now computed only when needed)

  // Build the compact "active-front" list once per step: the union superset of
  // every owned site any microstructure kernel could touch this step --
  //   active_flag==2 (liquid: can nucleate/solidify),
  //   T>=tl (hot: can melt),
  //   solid_d in (-nrefine-1, 0) (smoothing band),
  //   active_flag==3 && solid_d==-nrefine-2 (leftover nucleus for particle flip).
  // The phase kernels below run over this list (size n_active, the thin melt/
  // solidification shell) instead of nlocal. The list View is cached and
  // reallocated only when nlocal changes. Capture LOCAL View copies so the
  // lambda does not capture `this` (device page-fault hazard; see
  // BuildSmoothingList below). n_active==0 => no liquid, nothing hot, no band
  // site, no leftover nucleus => nothing for any microstructure kernel to do.
  int n_active = 0;
  if (device_active_list_fresh) {
    // Full fusion: the temperature fill already atomic-appended the active owned
    // sites into d_active_sites and counted them -- just read the count back.
    auto h_count = Kokkos::create_mirror_view(d_active_count);
    Kokkos::deep_copy(h_count, d_active_count);
    n_active = h_count();
  } else {
    // Full predicate scan (Rosenthal/Moser/host fills that didn't build the list).
    if ((int)d_active_sites.extent(0) != nlocal)
      d_active_sites = Kokkos::View<int*>("active_sites", nlocal);
    auto d_active_sites_local = d_active_sites;
    const int nlocal_act = this->nlocal;
    auto d_T_act = k_T.view_device();
    auto d_active_flag_act = d_active_flag;
    auto d_solid_d_act = d_solid_d;
    const double tl_act = this->tl;
    const int nrefine_act = this->nrefine;
    Kokkos::parallel_scan("BuildActiveList", nlocal_act,
      KOKKOS_LAMBDA(int i, int& offset, bool final) {
        const int af = d_active_flag_act(i);
        const double sd = d_solid_d_act(i);
        const bool active =
            (af == 2) ||
            (d_T_act(i) >= tl_act) ||
            (sd > -nrefine_act - 1 && sd < 0.0) ||
            (af == 3 && sd == -nrefine_act - 2);
        if (active) {
          if (final) d_active_sites_local(offset) = i;
          offset++;
        }
      }, n_active);
  }

  // 4. Phase transition kernels with event tracking
  // Phase 3.1: Melting (with Phase 1 optimization: early temperature scan)
  t_start = MPI_Wtime();

  // Melt over the active-front list. Melt candidates (T>=tl && active_flag!=2)
  // are a subset of the active list (hot clause), so the old CheckMeltingCandidates
  // pre-scan over nlocal is gone -- the kernel's own naccept is the candidate
  // count (melting is deterministic given T>=tl: it always melts a candidate,
  // RNG only randomizes the destroyed grain/orientation). So naccept>0 iff there
  // were candidates, preserving the powder-activation trigger (melting_occurred).
  int nmelt = melting_kernel(d_active_sites, n_active);
  naccept += nmelt;
  bool melting_occurred = (nmelt > 0);

  t_end = MPI_Wtime();
  t_melting += (t_end - t_start);

  // Phase 3.2: Nucleation (stochastic nucleation in undercooled liquid)
  t_start = MPI_Wtime();

  // Nucleation ranges over the active-front list, which contains EVERY liquid
  // (active_flag==2) site, so nucleation still sees all liquid every step -- the
  // "always launch / never gate on a candidate sub-count" recovery invariant
  // (see the removed block below) is preserved. n_active==0 means no liquid
  // anywhere, where nucleation is impossible by its own guard, so ranging over 0
  // is correct.
  naccept += nucleation_kernel(d_active_sites, n_active);

  t_end = MPI_Wtime();
  t_nucleation += (t_end - t_start);

  // Phase 3.5: Particle flipper (grow nuclei to specified size)
  // Only launch if there are newly nucleated sites (solid_d == -nrefine-2)
  t_start = MPI_Wtime();

  // Grow nuclei over the active-front list. Nucleus sites (active_flag==3 &&
  // solid_d==-nrefine-2) are a subset of the active list (leftover-nucleus
  // clause), so the separate CheckNucleiSites pre-scan over nlocal is gone -- the
  // kernel's own guard rejects non-nuclei, and ranging over n_active==0 is a
  // correct no-op (no nuclei).
  naccept += particle_flipper_kernel(d_active_sites, n_active);

  t_end = MPI_Wtime();
  t_particle += (t_end - t_start);

  // Phase 3.3: Solidification (epitaxial growth with texture weighting)
  t_start = MPI_Wtime();
  naccept += solidification_kernel(d_active_sites, n_active);
  t_end = MPI_Wtime();
  t_solidification += (t_end - t_start);

  // Phase 3.4: Smoothing kernel (Potts MC for grain boundary refinement)
  // Call once per timestep - sites smooth over multiple timesteps as solid_d increments
  // OPTIMIZATION 3: Work-efficient smoothing - only process sites in smoothing range
  t_start = MPI_Wtime();

  // Build compact list of smoothing sites using parallel_scan (Phase 1
  // optimization). The list View is cached (member) and reallocated only when
  // nlocal changes, rather than allocated on the GPU every timestep.
  int n_smoothing_sites = 0;
  if ((int)d_smoothing_sites.extent(0) != nlocal)
    d_smoothing_sites = Kokkos::View<int*>("smoothing_sites", nlocal);
  {
    auto d_solid_d_check = d_solid_d;
    int nrefine_check = nrefine;
    // Capture a LOCAL copy of the View handle. Using the member
    // d_smoothing_sites directly inside KOKKOS_LAMBDA ([=]) would capture
    // `this` and dereference the host `this` pointer on the device to reach
    // the View object -> GPU "page not present" fault (and the
    // -Wdeprecated-this-capture warning). The fault only fired once a site
    // first entered the smoothing band (final && needs_smoothing), which made
    // it look like a melt-triggered, run-dependent crash.
    auto d_smoothing_sites_local = d_smoothing_sites;
    auto d_active_sites_local = d_active_sites;

    // Scan the active-front list (size n_active) rather than nlocal: every
    // smoothing-band site is in d_active_sites (band clause), and the active
    // list is in ascending site-index order, so this builds the IDENTICAL
    // compact smoothing list at O(n_active) instead of O(nlocal).
    Kokkos::parallel_scan("BuildSmoothingList", n_active,
      KOKKOS_LAMBDA(int idx, int& offset, bool final) {
        const int i = d_active_sites_local(idx);
        double sd = d_solid_d_check(i);
        bool needs_smoothing = (sd > -nrefine_check - 1 && sd < 0.0);

        if (needs_smoothing) {
          if (final) {
            d_smoothing_sites_local(offset) = i;  // Store site index
          }
          offset++;
        }
      }, n_smoothing_sites);
  }

  // Only launch smoothing kernel if there are eligible sites
  // Process ONLY the sites in the compact list (not all 2.99M sites!)
  if (n_smoothing_sites > 0) {
    naccept += smoothing_kernel_efficient(d_smoothing_sites, n_smoothing_sites);
  }

  t_end = MPI_Wtime();
  t_smoothing += (t_end - t_start);

  // 5. Single fence after all kernels complete (replaces individual kernel fences)
  // This ensures all device writes are visible before proceeding
  t_start = MPI_Wtime();
  Kokkos::fence();
  t_end = MPI_Wtime();
  t_sync_host += (t_end - t_start);

  // Activate powder once melting first occurs, BEFORE syncing to host so the
  // device activation (which writes d_active_flag) is reflected in the host
  // copy used for output. Runs on the device.
  // CRITICAL: MPI_Allreduce the melting flag so ALL ranks call
  // activate_powder_sites() together (it contains collective calls); otherwise
  // ranks with no local melting would skip the collective and deadlock.
  if (last_powder_activation_time < 0.0) {
    int local_melting = melting_occurred ? 1 : 0;
    int global_melting = 0;
    MPI_Allreduce(&local_melting, &global_melting, 1, MPI_INT, MPI_SUM, world);

    if (global_melting > 0) {
      activate_powder_sites();
      last_powder_activation_time = time;
    }
  }

  // Device->host / ghost sync. The device is the source of truth: the physics
  // kernels read and write device Views every step. Host iarray/darray are only
  // consumed by stats, dumps, and diagnostics, so a full device->host copy is
  // only needed on output steps -- doing it every step (the old behavior) was
  // the dominant per-step cost (a ~10-array, O(nall) malloc+copy over millions
  // of sites). "Output fires this step" is time + dt_sweep >= nextoutput (the
  // iterate loop advances time AFTER app_update; nextoutput spans stats/dumps/
  // diagnostics), the same gate the lazy temperature copy-back below uses.
  const bool host_output_imminent =
      (time + dt_sweep >= nextoutput) || (time + dt_sweep >= stoptime);

  if (nprocs > 1) {
    // Multi-rank ghost exchange (Phase 2: selective). The kernels read as
    // NEIGHBORS the Monte Carlo state -- spin (iarray[0]), active_flag
    // (iarray[1]), quaternions q0-3 (darray[0-3]) -- and, in the solidification
    // gradient, neighbor temperature T (darray[5]). mobility/solid_d/G/V are
    // never read as neighbors, so they are dropped (full-synced only at output).
    //
    // T is included ONLY on the host-temperature path (device_temp == false:
    // Rosenthal keyhole / generic xyz / temperature_smooth). There the host fill
    // covers nlocal only, so ghost T must be exchanged (host owned T is current
    // from the fill; comm writes host ghost T, picked up by next step's
    // sync_temperature_to_device). On a device-temp path the fill already wrote
    // k_T over all nall positionally, so ghost T needs no comm.
    //
    // sync_ghost_to_host/device move the OWNED out / ghost in for spin/
    // active_flag/q0-3 only (T lives in host T[]/k_T, handled above). all_selective
    // uses the same swap as all(), so the same ghost sites are exchanged.
    static const int sel_i[2] = {0, 1};            // spin, active_flag
    static const int sel_d_dev[4]  = {0, 1, 2, 3};       // q0-3 (T device-resident)
    static const int sel_d_host[5] = {0, 1, 2, 3, 5};    // q0-3 + T (host fill)
    sync_ghost_to_host();                       // device owned -> host raw
    timer->stamp();
    if (device_temp) comm->all_selective(sel_i, 2, sel_d_dev, 4);
    else             comm->all_selective(sel_i, 2, sel_d_host, 5);
    timer->stamp(TIME_COMM);
    sync_ghost_to_device();                     // host ghost -> device

    // Optional deterministic correctness check (SPK_VERIFY_GHOST): redo the old
    // FULL exchange and confirm the device ghost MC state is bit-identical to
    // the selective path. Same owned state + same swap => must match; a mismatch
    // means a bug in the selective subviews/indices. Debug-only (doubles sync).
    if (getenv("SPK_VERIFY_GHOST")) {
      double sel = ghost_mc_checksum();
      sync_app_data_to_host(); comm->all(); sync_app_data_to_device();
      double full = ghost_mc_checksum();
      if (Kokkos::fabs(sel - full) > 1.0e-6 && domain->me == 0)
        fprintf(stderr, "GHOST MISMATCH step=%d selective=%.17g full=%.17g\n",
                step_count, sel, full);
    }
  }
  // Full device->host of ALL arrays only on output steps (stats/dumps/diags read
  // the host iarray/darray). Applies to both single and multi rank.
  if (host_output_imminent) sync_app_data_to_host();

  // Optional isotherm-compactness diagnostic on its configured cadence, matching
  // the CPU app. A no-op unless temperature_smooth + a diag interval are set
  // (which also force the host temperature path, so host T is current here).
  if (host_T_stale) sync_temperature_to_host();
  thermal->maybe_run_diagnostics(step_count);

  // DEBUG: Print gradient/velocity statistics every 10 steps
  if (step_count % 10 == 0 && domain->me == 0) {
    int total_events = debug_nucleation_count + debug_solidification_count;
    double zero_fraction = (total_events > 0) ? (double)debug_zero_G_count / total_events : 0.0;
    double avg_G = (debug_nonzero_G_count > 0) ? debug_G_sum / debug_nonzero_G_count : 0.0;
    double avg_V = (debug_nonzero_G_count > 0) ? debug_V_sum / debug_nonzero_G_count : 0.0;

    fprintf(screen, "DEBUG [t=%.6f, step=%d]: Nuc=%d, Solid=%d, ZeroG=%d (%.1f%%), NonZeroG=%d, AvgG=%.2e, AvgV=%.2e\n",
            time, step_count, debug_nucleation_count, debug_solidification_count,
            debug_zero_G_count, zero_fraction*100, debug_nonzero_G_count,
            avg_G, avg_V);

    // Reset counters for next interval
    debug_nucleation_count = 0;
    debug_solidification_count = 0;
    debug_zero_G_count = 0;
    debug_nonzero_G_count = 0;
    debug_G_sum = 0.0;
    debug_V_sum = 0.0;
  }

  // Print timing summary every 100 steps
  if (step_count % 100 == 0 && domain->me == 0) {
    double total_time = t_temp_update + t_temp_sync + t_melting +
                        t_nucleation + t_particle + t_solidification + t_smoothing + t_sync_host;
    fprintf(screen, "\n=== Kokkos Timing Summary (after %d steps, %.4f total sec) ===\n", step_count, total_time);
    fprintf(screen, "  Temperature update:   %8.3f s (%5.1f%%)\n", t_temp_update, 100.0*t_temp_update/total_time);
    fprintf(screen, "  Temp sync to device:  %8.3f s (%5.1f%%)\n", t_temp_sync, 100.0*t_temp_sync/total_time);
    fprintf(screen, "  Melting kernel:       %8.3f s (%5.1f%%) [selective launch]\n", t_melting, 100.0*t_melting/total_time);
    fprintf(screen, "  Nucleation kernel:    %8.3f s (%5.1f%%) [selective launch]\n", t_nucleation, 100.0*t_nucleation/total_time);
    fprintf(screen, "  Particle kernel:      %8.3f s (%5.1f%%) [selective launch]\n", t_particle, 100.0*t_particle/total_time);
    fprintf(screen, "  Solidification kernel:%8.3f s (%5.1f%%) [on-demand gradients]\n", t_solidification, 100.0*t_solidification/total_time);
    fprintf(screen, "  Smoothing kernel:     %8.3f s (%5.1f%%) [selective launch, 1x/timestep]\n", t_smoothing, 100.0*t_smoothing/total_time);
    fprintf(screen, "  Sync to host:         %8.3f s (%5.1f%%)\n", t_sync_host, 100.0*t_sync_host/total_time);
    fprintf(screen, "  TOTAL:                %8.3f s\n", total_time);
    fprintf(screen, "  Avg per step:         %8.6f s\n", total_time/step_count);
    fprintf(screen, "  Optimizations: on-demand gradients, selective kernel launching\n\n");
  }

  // Lazy device->host temperature copy-back: if a device fill left host T (d6)
  // stale and an output step is imminent, refresh host T now so the diagnostic
  // and dumps see the current field. This app sets app_update_only=1, so it runs
  // via AppLattice::iterate_app_update_only, which advances time once per step
  // (time += dt, dt == dt_sweep here) AFTER this returns -- so output fires next
  // when time + dt_sweep >= nextoutput (or the run ends).
  if (host_T_stale &&
      (time + dt_sweep >= nextoutput || time + dt_sweep >= stoptime)) {
    sync_temperature_to_host();
  }

  // NOTE: time is advanced by the iterate loop (iterate_app_update_only does
  // time += dt), exactly as in the CPU AppAdditiveTexture. Do NOT increment time
  // here -- an earlier version added a second `time += dt_sweep`, double-counting
  // time (advancing 2x per step), which halved the MC sweep resolution and
  // shifted the temperature sampling phase vs the CPU app.
}

/* ----------------------------------------------------------------------
   Activate powder sites that are at or below the meltpool surface

   Strategy: Find the maximum z-coordinate of all molten sites (meltpool top),
   then set all inactive sites below that level to powder (active_flag = 1).

   This function should be called:
   - After fast forwards > 1 second (new layer deposition)
   - During app_update when meltpool is present

   Runs entirely on the device, operating directly on d_active_flag (the
   authoritative phase-state array): a parallel_reduce for the meltpool-top z
   over molten sites, then a parallel_reduce that activates inactive sites at or
   below it. Keeping it on the device (rather than mutating host active_flag) is
   what makes it consistent -- a host write here would be overwritten by the
   device->host sync, and for nprocs==1 never reaches the device at all.
------------------------------------------------------------------------- */
void AppAdditiveTextureKokkos::activate_powder_sites() {
  ensure_device_xyz();
  auto d_af = d_active_flag;
  auto d_xyz_l = d_xyz;
  const int n = nlocal;

  // Local meltpool top z over molten sites (active_flag == 2), then the global
  // max across ranks. z is in lattice units, matching the host comparison.
  // local_max_molten_z is the reduction OUTPUT; Kokkos::Max seeds the reduction
  // with its own identity (-DBL_MAX), so with no molten sites the result stays
  // very negative and the -1e99 guard below detects "no meltpool".
  double local_max_molten_z;
  Kokkos::parallel_reduce("MeltpoolTopZ", n,
    KOKKOS_LAMBDA(const int i, double &m) {
      if (d_af(i) == 2) { const double z = d_xyz_l(i,2); if (z > m) m = z; }
    }, Kokkos::Max<double>(local_max_molten_z));

  double global_max_molten_z;
  MPI_Allreduce(&local_max_molten_z, &global_max_molten_z, 1, MPI_DOUBLE, MPI_MAX, world);
  if (global_max_molten_z <= -1e99) return;   // no meltpool anywhere

  // Activate all inactive sites (active_flag == 0) at or below the meltpool top;
  // the reduction returns the count for reporting.
  int powder_activated = 0;
  Kokkos::parallel_reduce("PowderActivate", n,
    KOKKOS_LAMBDA(const int i, int &cnt) {
      if (d_af(i) == 0 && d_xyz_l(i,2) <= global_max_molten_z) {
        d_af(i) = 1;  // POWDER state
        cnt++;
      }
    }, powder_activated);
  Kokkos::fence();

  int global_powder_activated = 0;
  MPI_Allreduce(&powder_activated, &global_powder_activated, 1, MPI_INT, MPI_SUM, world);
  if (domain->me == 0 && global_powder_activated > 0) {
    fprintf(screen, "Powder activation at t=%.6e s: activated %d sites (meltpool top at z=%.3f)\n",
            time, global_powder_activated, global_max_molten_z);
  }
}

/* ----------------------------------------------------------------------
   Override stats to sync device data before computing statistics
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::stats(char *strtmp)
{
  // Sync device data to host so stats can access current values. sync_app_data_
  // to_host() does not touch the temperature field, so refresh host T (d6) here
  // too when a device fill left it stale (the d6 diagnostic reads host T).
  sync_app_data_to_host();
  if (host_T_stale) sync_temperature_to_host();

  // Call parent class stats implementation
  AppPottsQuaternionKokkos::stats(strtmp);
}

void AppAdditiveTextureKokkos::stats_header(char *strtmp)
{
  // Just forward to parent - no sync needed for header
  AppPottsQuaternionKokkos::stats_header(strtmp);
}

/* ----------------------------------------------------------------------
   Process input commands
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::input_app(char *command, int narg, char **arg)
{
  // Keep the thermal snapshot current before any thermal command handler
  // (e.g. laser_path reads host->time/dx when building the scan path).
  refresh_thermal_host();

  // Temperature-I/O commands are delegated to the shared ThermalManager,
  // exactly as the CPU AppAdditiveTexture routes them.
  if (strcmp(command, "setup_temperature_source") == 0) {
    thermal->setup_cmd(narg, arg);

  } else if (strcmp(command, "laser_path") == 0) {
    thermal->laser_path_cmd(narg, arg);

  } else if (strcmp(command, "laser_fluctuations") == 0) {
    thermal->laser_fluctuations_cmd(narg, arg);

  } else if (strcmp(command, "temperature_smooth") == 0) {
    thermal->smooth_cmd(narg, arg);

  } else if (strcmp(command, "nodal_smooth") == 0) {
    thermal->nodal_smooth_cmd(narg, arg);

  } else if (strcmp(command, "temperature_optimization") == 0) {
    thermal->optimization_cmd(narg, arg);

  } else if (strcmp(command, "liquidus") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal liquidus command");
    tl = atof(arg[0]);

  } else if (strcmp(command, "solidus") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal solidus command");
    ts = atof(arg[0]);

  } else if (strcmp(command, "texture_parameters") == 0) {
    // Reparametrized to match the CPU AppAdditiveTexture: 2 args (c2 c3) with
    // c1 fixed at 1.0 (only mobility ratios affect roulette-wheel selection).
    // The device kernels use mobility = c1 + c2*cos(c3*theta) with c1 == 1.0.
    if (narg == 3)
      error->all(FLERR,
        "texture_parameters now takes 2 arguments (c2 c3), not 3. "
        "The baseline coefficient c1 has been fixed at 1.0 because "
        "only mobility ratios affect selection. To migrate an old "
        "input file: texture_parameters c2/c1 c3 (divide old c2 by "
        "old c1, drop old c1)");
    if (narg != 2) error->all(FLERR, "Illegal texture_parameters command");
    c1 = 1.0;
    c2 = atof(arg[0]);
    c3 = atof(arg[1]);

    // Validation (mirrors AppAdditiveTexture): mob(theta)=1+c2*cos(c3*theta)
    // over theta in [0, theta_max], theta_max = acos(1/sqrt(3)).
    const double theta_max = std::acos(1.0 / std::sqrt(3.0));
    if (c2 < 0.0)
      error->all(FLERR,
        "texture_parameters: c2 must be >= 0 (negative amplitude "
        "would mean misaligned grains grow faster than aligned ones, "
        "which is unphysical for cubic solidification)");
    if (c3 < 0.0)
      error->all(FLERR, "texture_parameters: c3 must be >= 0");
    if (c3 * theta_max > MY_PI + 1.0e-12) {
      char msg[256];
      snprintf(msg, sizeof(msg),
        "texture_parameters: c3 = %g exceeds monotonicity limit "
        "pi/theta_max ~ %g; mobility would be non-monotonic in "
        "misorientation angle", c3, MY_PI / theta_max);
      error->all(FLERR, msg);
    }
    double mob_min = 1.0 + c2 * std::cos(c3 * theta_max);
    if (mob_min < -1.0e-12) {
      char msg[256];
      snprintf(msg, sizeof(msg),
        "texture_parameters: minimum mobility = %g < 0 at theta = "
        "theta_max (= %g rad); roulette-wheel selection requires "
        "non-negative mobility", mob_min, theta_max);
      error->all(FLERR, msg);
    }
    if (c2 == 0.0 && screen)
      fprintf(screen,
        "WARNING: texture_parameters c2 = 0; mobility has no "
        "orientation dependence\n");

  } else if (strcmp(command, "nucleation_density") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal nucleation_density command");
    no = atof(arg[0]);

  } else if (strcmp(command, "critical_undercooling") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal critical_undercooling command");
    tc = atof(arg[0]);

  } else if (strcmp(command, "undercooling_deviation") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal undercooling_deviation command");
    tsig = atof(arg[0]);

  } else if (strcmp(command, "fast_forward_search_window") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal fast_forward_search_window command");
    thermal->set_fast_forward_search_window(atof(arg[0]));

  } else if (strcmp(command, "mean_nuclei_volume") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal mean_nuclei_volume command");
    size_norm = atof(arg[0]);

  } else if (strcmp(command, "nuclei_volume_deviation") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal nuclei_volume_deviation command");
    size_sig = atof(arg[0]);

  } else if (strcmp(command, "solid_front_vel") == 0) {
    delete[] solid_front_coeffs;
    if (narg < 1) error->all(FLERR, "Illegal solid_front_vel command");
    solid_front_length = atoi(arg[0]);
    if (solid_front_length <= 0) {
      error->all(FLERR, "Illegal solidification front velocity specification");
    }
    if (narg != solid_front_length + 1)
      error->all(FLERR, "Illegal solid_front_vel command: expected <n> followed "
                        "by n coefficients");
    solid_front_coeffs = new double[solid_front_length];
    for (int i = 1; i < solid_front_length + 1; i++) {
      solid_front_coeffs[i - 1] = atof(arg[i]);
    }

  } else if (strcmp(command, "smooth_greedy_multiproposal") == 0) {
    // Usage: smooth_greedy_multiproposal on|off  (mirrors the CPU app)
    if (narg != 1) error->all(FLERR, "Illegal smooth_greedy_multiproposal command");
    if (strcmp(arg[0], "on") == 0) smooth_greedy_multiproposal_enabled = true;
    else if (strcmp(arg[0], "off") == 0) smooth_greedy_multiproposal_enabled = false;
    else error->all(FLERR, "smooth_greedy_multiproposal: argument must be on or off");
    if (domain->me == 0 && screen)
      fprintf(screen, "smooth_greedy_multiproposal %s\n",
              smooth_greedy_multiproposal_enabled ? "enabled" : "disabled");

  } else if (strcmp(command, "misorientation_function") == 0) {
    if (narg != 2 && narg != 3) error->all(FLERR, "Illegal misorientation_function command");
    max_misorient = atof(arg[0]);
    mis_thresh = atof(arg[1]) * MY_PI / 180.0; // Convert from degrees to radians
    if (narg == 3) {
      misorient_alpha = atof(arg[2]);
      if (misorient_alpha <= 0)
        error->all(FLERR, "Illegal misorientation_function alpha value: must be greater than 0");
    }

  } else if (strcmp(command, "misorientation_target") == 0) {
    // Select the target the per-site misorientation rotates toward during
    // solidification (matches AppAdditiveTexture::input_app). gradient =
    // thermal-gradient direction; random = uniform random unit vector (the kk
    // default); random_rotation = rotation by mis_angle about a random axis.
    if (narg != 1) error->all(FLERR, "Illegal misorientation_target command");
    if (strcmp(arg[0], "gradient") == 0)
      misorientation_target_mode = MISORI_TARGET_GRADIENT;
    else if (strcmp(arg[0], "random") == 0)
      misorientation_target_mode = MISORI_TARGET_RANDOM;
    else if (strcmp(arg[0], "random_rotation") == 0)
      misorientation_target_mode = MISORI_TARGET_RANDOM_ROTATION;
    else
      error->all(FLERR, "Illegal misorientation_target value: use gradient, random, or random_rotation");
    if (domain->me == 0 && screen)
      fprintf(screen, "misorientation_target = %s\n", arg[0]);

  } else {
    // Pass to parent class
    AppPottsQuaternionKokkos::input_app(command, narg, arg);
  }
}
