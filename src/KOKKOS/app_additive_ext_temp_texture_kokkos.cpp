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

#include "app_additive_ext_temp_texture_kokkos.h"
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

AppAdditiveExtTempTextureKokkos::AppAdditiveExtTempTextureKokkos(SPPARKS *spk, int narg, char **arg) :
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

  t_active = 0;
  time_step = dt;

  // Initialize RNG pool (seed will be set in init_app() using ranapp->state())
  rand_pool = Kokkos::Random_XorShift64_Pool<>();
}

/* ----------------------------------------------------------------------
   Destructor
------------------------------------------------------------------------- */

AppAdditiveExtTempTextureKokkos::~AppAdditiveExtTempTextureKokkos()
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

void AppAdditiveExtTempTextureKokkos::grow_app()
{
  // Follow same pattern as AppPottsQuaternionKokkos:
  // Call AppPotts::grow_app() then manually set all pointers
  // Don't call AppPottsQuaternion::grow_app() - it breaks the chain
  AppPotts::grow_app();

  // Manually set quaternion pointers (from AppPottsQuaternion)
  q0 = darray[0];
  qx = darray[1];
  qy = darray[2];
  qz = darray[3];

  // Set up additional AM pointers
  active_flag = iarray[1];
  mobility_out = darray[4];
  T = darray[5];
  solid_d = darray[6];
  G = darray[7];
  V = darray[8];

  int nall = nlocal + nghost;

  // Allocate Kokkos views for device computation
  // Allocate neighbor arrays (same as AppPottsQuaternionKokkos)
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }

  // Allocate double buffers for spin and quaternions (from parent)
  buffer_spin.allocate("spin", nall);
  buffer_q0.allocate("q0", nall);
  buffer_qx.allocate("qx", nall);
  buffer_qy.allocate("qy", nall);
  buffer_qz.allocate("qz", nall);

  // Allocate DualView for temperature (CPU updates from HDF5, device reads)
  k_T = Kokkos::DualView<double*>("temperature", nall);

  // Allocate device-only Views for gradient cache (computed once per timestep on device)
  d_G = Kokkos::View<double*>("gradient_magnitude", nall);
  d_grad_x = Kokkos::View<double*>("gradient_x", nall);
  d_grad_y = Kokkos::View<double*>("gradient_y", nall);
  d_grad_z = Kokkos::View<double*>("gradient_z", nall);
  d_V = Kokkos::View<double*>("solidification_velocity", nall);

  // Allocate double buffers for arrays modified during sweeps
  buffer_active_flag.allocate("active_flag", nall);
  buffer_mobility_out.allocate("mobility_out", nall);
  buffer_solid_d.allocate("solid_d", nall);

  // Allocate per-spin nucleation parameter arrays (device-accessible)
  d_nucleation_flags = Kokkos::View<int*>("nucleation_flags", nspins);
  d_nucleation_temps = Kokkos::View<double*>("nucleation_temps", nspins);
  d_nucleation_sizes = Kokkos::View<double*>("nucleation_sizes", nspins);

  // Allocate global parameter arrays (device-accessible constants)
  d_solid_front_coeffs = Kokkos::View<double*>("solid_front_coeffs", solid_front_length);
  d_texture_params = Kokkos::View<double[3]>("texture_params");

  // Allocate neighbor distance and unique arrays on host (used in site_event methods)
  if (neigh_dist) delete[] neigh_dist;
  neigh_dist = new double[maxneigh];

  if (unique_dot) delete[] unique_dot;
  unique_dot = new double[maxneigh];
}

/* ----------------------------------------------------------------------
   Initialize application
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::init_app()
{
  // Call parent initialization
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

  // Initialize RNG pool with seed from ranapp
  // Use ranapp->seed to get reproducible seed from MPI rank
  uint64_t seed = static_cast<uint64_t>(ranapp->seed);
  rand_pool = Kokkos::Random_XorShift64_Pool<>(seed);

  // Set timestep for framework (matches CPU version behavior)
  dt_sweep = dt;

  // Initialize nucleation parameters (per-spin arrays and global params)
  initialize_nucleation_parameters();

  // Sync initial data to device (including new arrays)
  sync_data_to_device();

  // Initialize temperature source if configured
  if (use_temperature_source && temperature_source) {
    temperature_source->update_temperatures(0.0, 0.0);
  }
}

/* ----------------------------------------------------------------------
   Sync data from host to device (called after MPI ghost exchange)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::sync_data_to_device()
{
  int nall = nlocal + nghost;

  // Sync parent class arrays (spin and quaternions) - duplicate parent logic
  // since sync_data_to_device() is private

  // Sync spin directly
  buffer_spin.sync_to_device(spin);

  // Sync quaternions with normalization (inline normalization logic)
  double* q0_norm = new double[nall];
  double* qx_norm = new double[nall];
  double* qy_norm = new double[nall];
  double* qz_norm = new double[nall];

  // Normalize quaternion arrays
  for (int i = 0; i < nall; i++) {
    double q[4] = {q0[i], qx[i], qy[i], qz[i]};
    double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      q0_norm[i] = q[0] * inv_norm;
      qx_norm[i] = q[1] * inv_norm;
      qy_norm[i] = q[2] * inv_norm;
      qz_norm[i] = q[3] * inv_norm;
    } else {
      q0_norm[i] = q[0];
      qx_norm[i] = q[1];
      qy_norm[i] = q[2];
      qz_norm[i] = q[3];
    }
  }

  buffer_q0.sync_to_device(q0_norm);
  buffer_qx.sync_to_device(qx_norm);
  buffer_qy.sync_to_device(qy_norm);
  buffer_qz.sync_to_device(qz_norm);

  delete[] q0_norm;
  delete[] qx_norm;
  delete[] qy_norm;
  delete[] qz_norm;

  // Sync our new double-buffered arrays
  buffer_active_flag.sync_to_device(active_flag);
  buffer_mobility_out.sync_to_device(mobility_out);
  buffer_solid_d.sync_to_device(solid_d);

  // Note: Temperature k_T is synced separately in sync_temperature_to_device()
  // Gradients d_G and d_V are device-computed, synced to host only when needed for output
}

/* ----------------------------------------------------------------------
   Sync data from device to host (called before MPI ghost exchange)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::sync_data_to_host()
{
  int nall = nlocal + nghost;

  // Sync parent class arrays (spin and quaternions) - duplicate parent logic
  // since sync_data_to_host() is private

  // Sync spin directly
  buffer_spin.sync_to_host(spin);

  // Sync quaternions with normalization
  double* q0_temp = new double[nall];
  double* qx_temp = new double[nall];
  double* qy_temp = new double[nall];
  double* qz_temp = new double[nall];

  buffer_q0.sync_to_host(q0_temp);
  buffer_qx.sync_to_host(qx_temp);
  buffer_qy.sync_to_host(qy_temp);
  buffer_qz.sync_to_host(qz_temp);

  // Normalize and copy to final host arrays (inline normalization logic)
  for (int i = 0; i < nall; i++) {
    double q[4] = {q0_temp[i], qx_temp[i], qy_temp[i], qz_temp[i]};
    double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      q0[i] = q[0] * inv_norm;
      qx[i] = q[1] * inv_norm;
      qy[i] = q[2] * inv_norm;
      qz[i] = q[3] * inv_norm;
    } else {
      q0[i] = q[0];
      qx[i] = q[1];
      qy[i] = q[2];
      qz[i] = q[3];
    }
  }

  delete[] q0_temp;
  delete[] qx_temp;
  delete[] qy_temp;
  delete[] qz_temp;

  // Sync our double-buffered arrays back to host
  buffer_active_flag.sync_to_host(active_flag);
  buffer_mobility_out.sync_to_host(mobility_out);
  buffer_solid_d.sync_to_host(solid_d);

  // CRITICAL FIX: Sync gradient and velocity arrays for output/visualization
  // These are device-only views that must be copied to host for STITCH dumps
  auto h_G = Kokkos::create_mirror_view(d_G);
  auto h_V = Kokkos::create_mirror_view(d_V);
  Kokkos::deep_copy(h_G, d_G);
  Kokkos::deep_copy(h_V, d_V);
  for (int i = 0; i < nall; i++) {
    G[i] = h_G(i);
    V[i] = h_V(i);
  }
}

/* ----------------------------------------------------------------------
   Swap double buffers (called after each sweep)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::swap_buffers()
{
  // Note: Can't call parent swap_buffers() directly as it's private
  // Instead, swap all buffers (parent + ours) here
  AppPottsQuaternionKokkos::buffer_spin.swap();
  AppPottsQuaternionKokkos::buffer_q0.swap();
  AppPottsQuaternionKokkos::buffer_qx.swap();
  AppPottsQuaternionKokkos::buffer_qy.swap();
  AppPottsQuaternionKokkos::buffer_qz.swap();

  // Swap our new buffers
  buffer_active_flag.swap();
  buffer_mobility_out.swap();
  buffer_solid_d.swap();

  // Note: k_T does NOT need swap() - it's a DualView, not a double buffer
  // Gradients are device-only Views, no swap needed
}

/* ----------------------------------------------------------------------
   Update temperature from external source (CPU operation)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::update_temperature_from_source(double simulation_time)
{
  if (!use_temperature_source || !temperature_source) return;

  // Update temperature source (may load new HDF5 layer)
  temperature_source->update_temperatures(dt, simulation_time);

  // Update host temperature array for all local sites
  for (int i = 0; i < nlocal; i++) {
    // Use site-based temperature access (faster with element caching)
    T[i] = temperature_source->get_temperature_at_site(i, simulation_time);
  }

  // Ghost sites will get temperatures via MPI exchange
}

/* ----------------------------------------------------------------------
   Sync temperature from host to device (after CPU update from HDF5)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::sync_temperature_to_device()
{
  int nall = nlocal + nghost;

  // Copy host temperature array to DualView host mirror
  auto h_T = k_T.h_view;
  for (int i = 0; i < nall; i++) {
    h_T(i) = T[i];
  }

  // Mark host as modified and sync to device
  k_T.modify_host();
  k_T.sync_device();
}

/* ----------------------------------------------------------------------
   Compute thermal gradients on device (cache for later use)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::compute_gradients_kernel()
{
  // Get device views
  auto d_T = k_T.d_view;  // Read-only temperature on device
  auto k_numneigh_d = k_numneigh.d_view;
  auto k_neighbor_d = k_neighbor.d_view;

  // Create temporary device view for xyz (xyz comes from CPU parent, not Kokkos)
  // Use unmanaged view wrapped around host xyz array
  int nall = nlocal + nghost;
  using HostSpace = Kokkos::HostSpace;
  using LayoutRight = Kokkos::LayoutRight;
  Kokkos::View<double*[3], LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_xyz(xyz[0], nall);
  auto k_xyz_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xyz);

  // Get active_flag for melt surface detection
  auto d_active_flag = buffer_active_flag.read_buffer;

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
        i, d_T, d_active_flag, k_numneigh_d, k_neighbor_d, k_xyz_d,
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

void AppAdditiveExtTempTextureKokkos::initialize_nucleation_parameters()
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

int AppAdditiveExtTempTextureKokkos::melting_kernel()
{
  // Get device views (read-only and read-write)
  auto d_T_local = k_T.d_view;
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  auto d_qx_current = buffer_qx.read_buffer;
  auto d_qx_next = buffer_qx.write_buffer;
  auto d_qy_current = buffer_qy.read_buffer;
  auto d_qy_next = buffer_qy.write_buffer;
  auto d_qz_current = buffer_qz.read_buffer;
  auto d_qz_next = buffer_qz.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nspins_local = this->nspins;
  int nlocal_local = this->nlocal;
  auto rand_pool_local = this->rand_pool;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;

  // Melting kernel: parallel over all local sites with event tracking
  int naccept_local = 0;
  Kokkos::parallel_reduce("MeltingKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_current(i);

      // Check if site should melt: T >= liquidus && not already liquid
      if (T_i >= tl_local && active_flag_i != 2) {
        // Get thread-local RNG state
        auto generator = rand_pool_local.get_state();

        // Transition to liquid state
        d_active_flag_next(i) = 2;  // LIQUID

        // Randomize grain ID (destroy grain structure)
        d_spin_next(i) = static_cast<int>(nspins_local * generator.drand());

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
        d_q0_next(i) = x0;
        d_qx_next(i) = x1;
        d_qy_next(i) = x2 * s;
        d_qz_next(i) = x3 * s;

        // Reset solidification tracking
        d_solid_d_next(i) = 0.0;
        d_G_local(i) = 0.0;
        d_V_local(i) = 0.0;

        // Return RNG state to pool
        rand_pool_local.free_state(generator);

        // Track melting event
        count++;
      } else {
        // No change - copy current state to next
        d_active_flag_next(i) = active_flag_i;
        d_spin_next(i) = d_spin_current(i);
        d_q0_next(i) = d_q0_current(i);
        d_qx_next(i) = d_qx_current(i);
        d_qy_next(i) = d_qy_current(i);
        d_qz_next(i) = d_qz_current(i);
        d_solid_d_next(i) = d_solid_d_current(i);
      }
    }, naccept_local);

  // Fence removed - single fence at end of app_update() handles all kernels

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Nucleation kernel: Stochastic nucleation in undercooled liquid
   Phase 3.2: Sites transition to solid when undercooling exceeds critical value
------------------------------------------------------------------------- */

int AppAdditiveExtTempTextureKokkos::nucleation_kernel()
{
  // Get device views
  auto d_T_local = k_T.d_view;
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nlocal_local = this->nlocal;
  int nrefine_local = this->nrefine;
  auto d_nucleation_flags = this->d_nucleation_flags;
  auto d_nucleation_temps = this->d_nucleation_temps;
  auto d_nucleation_sizes = this->d_nucleation_sizes;
  auto d_solid_front_coeffs = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;
  double dx_local = this->dx;

  // Nucleation kernel: parallel over all local sites with event tracking
  int naccept_local = 0;
  Kokkos::parallel_reduce("NucleationKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_current(i);
      int spin_i = d_spin_current(i);

      // Check if site is liquid and below liquidus (undercooled)
      if (active_flag_i == 2 && T_i < tl_local) {
        // Check if this spin can nucleate
        if (d_nucleation_flags(spin_i) == 1) {
          // Calculate undercooling
          double Tcool = tl_local - T_i;

          // Check if undercooling exceeds critical value for this spin
          if (Tcool >= d_nucleation_temps(spin_i)) {
            // Nucleate: transition to solid
            d_active_flag_next(i) = 3;  // SOLID
            count++;  // Track nucleation event

            // Set solid_d flag to prevent disappearance during smoothing
            // Negative value triggers smoothing countdown later
            d_solid_d_next(i) = -nrefine_local - 2;

            // Compute solidification velocity using polynomial: V = Σ coeffs[k] * Tcool^k
            double V_i = 0.0;
            double Tcool_power = 1.0;
            for (int k = solid_front_length_local - 1; k >= 0; k--) {
              V_i += d_solid_front_coeffs(k) * Tcool_power;
              Tcool_power *= Tcool;
            }
            d_V_local(i) = V_i;

            // Note: Gradient G is already computed by compute_gradients_kernel()
            // No need to recompute here

            // TODO Phase 3.2: Particle flipper to grow nucleus
            // For now, nucleus is single site
            // Will implement particle growth in separate kernel
          } else {
            // Can nucleate but hasn't reached critical undercooling yet
            d_active_flag_next(i) = active_flag_i;  // Remain liquid
            d_solid_d_next(i) = d_solid_d_current(i);
          }
        } else {
          // This spin cannot nucleate
          d_active_flag_next(i) = active_flag_i;  // Remain liquid
          d_solid_d_next(i) = d_solid_d_current(i);
        }
      } else {
        // Not liquid or not undercooled - no change
        d_active_flag_next(i) = active_flag_i;
        d_solid_d_next(i) = d_solid_d_current(i);
      }
    }, naccept_local);

  // Fence removed - single fence at end of app_update() handles all kernels

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

int AppAdditiveExtTempTextureKokkos::particle_flipper_kernel()
{
  // Capture all necessary data for device lambda
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  auto d_qx_current = buffer_qx.read_buffer;
  auto d_qx_next = buffer_qx.write_buffer;
  auto d_qy_current = buffer_qy.read_buffer;
  auto d_qy_next = buffer_qy.write_buffer;
  auto d_qz_current = buffer_qz.read_buffer;
  auto d_qz_next = buffer_qz.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  auto k_neighbor_d = k_neighbor.d_view;
  auto k_numneigh_d = k_numneigh.d_view;
  auto d_G_local = d_G;
  auto d_grad_x_local = d_grad_x;
  auto d_grad_y_local = d_grad_y;
  auto d_grad_z_local = d_grad_z;
  auto d_V_local = d_V;
  auto d_nucleation_sizes_local = d_nucleation_sizes;
  auto rand_pool_local = rand_pool;

  int nlocal_local = nlocal;
  int nrefine_local = nrefine;
  double dx_local = dx;

  // Parallel over all local sites - each nucleus flips its neighbors with event tracking
  int naccept_local = 0;
  Kokkos::parallel_reduce("ParticleFlipperKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      // Only process newly nucleated sites (marked with solid_d = -nrefine-2)
      double solid_d_i = d_solid_d_current(i);
      int active_flag_i = d_active_flag_current(i);

      // Check if this is a newly nucleated site
      if (active_flag_i != 3 || solid_d_i != -nrefine_local - 2) {
        // Not a nucleus - copy state unchanged
        d_spin_next(i) = d_spin_current(i);
        d_q0_next(i) = d_q0_current(i);
        d_qx_next(i) = d_qx_current(i);
        d_qy_next(i) = d_qy_current(i);
        d_qz_next(i) = d_qz_current(i);
        d_active_flag_next(i) = active_flag_i;
        d_solid_d_next(i) = solid_d_i;
        return;
      }

      // This is a newly nucleated site - attempt to grow it
      int spin_i = d_spin_current(i);
      double q_i[4] = {d_q0_current(i), d_qx_current(i), d_qy_current(i), d_qz_current(i)};
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
        d_spin_next(i) = spin_i;
        d_q0_next(i) = q_i[0];
        d_qx_next(i) = q_i[1];
        d_qy_next(i) = q_i[2];
        d_qz_next(i) = q_i[3];
        d_active_flag_next(i) = 3;  // SOLID
            count++;  // Track nucleation event
        d_solid_d_next(i) = -nrefine_local - 3;  // Ready for smoothing
        return;
      }

      // Build list of liquid neighbors that can be flipped
      int nn = k_numneigh_d(i);
      constexpr int MAX_NEIGHBORS = 26;  // sc/26n lattice assumption
      int liquid_neighbors[MAX_NEIGHBORS];
      int num_liquid = 0;

      for (int j = 0; j < nn && j < MAX_NEIGHBORS; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_current(nbr);

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
        int old_flag = Kokkos::atomic_compare_exchange(&d_active_flag_next(nbr), expected, desired);

        if (old_flag == 2) {
          // We successfully claimed this neighbor - flip it to match nucleus
          d_spin_next(nbr) = spin_i;
          d_q0_next(nbr) = q_i[0];
          d_qx_next(nbr) = q_i[1];
          d_qy_next(nbr) = q_i[2];
          d_qz_next(nbr) = q_i[3];
          // active_flag already set to SOLID by CAS above
          d_solid_d_next(nbr) = -nrefine_local - 3;  // Ready for smoothing

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
      d_spin_next(i) = spin_i;
      d_q0_next(i) = q_i[0];
      d_qx_next(i) = q_i[1];
      d_qy_next(i) = q_i[2];
      d_qz_next(i) = q_i[3];
      d_active_flag_next(i) = 3;  // SOLID
      d_solid_d_next(i) = -nrefine_local - 3;  // Ready for smoothing

      // Track nucleus growth event
      count++;
    }, naccept_local);

  // Fence removed - single fence at end of app_update() handles all kernels

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Solidification kernel: Epitaxial growth with texture weighting
   Phase 3.3: Liquid sites captured by solid neighbors based on texture
------------------------------------------------------------------------- */

int AppAdditiveExtTempTextureKokkos::solidification_kernel()
{
  // Get device views
  auto d_T_local = k_T.d_view;
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  auto d_qx_current = buffer_qx.read_buffer;
  auto d_qx_next = buffer_qx.write_buffer;
  auto d_qy_current = buffer_qy.read_buffer;
  auto d_qy_next = buffer_qy.write_buffer;
  auto d_qz_current = buffer_qz.read_buffer;
  auto d_qz_next = buffer_qz.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  auto k_numneigh_d = k_numneigh.d_view;
  auto k_neighbor_d = k_neighbor.d_view;

  // Create device view of xyz coordinates (needed for on-demand gradient computation)
  int nall = nlocal + nghost;
  using HostSpace = Kokkos::HostSpace;
  using LayoutRight = Kokkos::LayoutRight;
  Kokkos::View<double*[3], LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_xyz(xyz[0], nall);
  auto k_xyz_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xyz);

  // Capture parameters
  double tl_local = this->tl;
  double ts_local = this->ts;
  double dt_local = this->dt;
  int nlocal_local = this->nlocal;
  int nrefine_local = this->nrefine;
  double dx_local = this->dx;
  auto d_solid_front_coeffs = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  auto d_texture_params = this->d_texture_params;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;
  auto d_grad_x_local = this->d_grad_x;
  auto d_grad_y_local = this->d_grad_y;
  auto d_grad_z_local = this->d_grad_z;
  auto rand_pool_local = this->rand_pool;

  // Misorientation parameters (Phase 3.6)
  double max_misorient_local = this->max_misorient;
  double t_cool_max_local = this->t_cool_max;
  double mis_thresh_local = this->mis_thresh;
  double misorient_alpha_local = this->misorient_alpha;
  int nspins_local = this->nspins;

  // Neighbor distance lookup (26-neighbor sc lattice)
  // Distance to each neighbor type in lattice units
  constexpr double neigh_dist[26] = {
    1.732050808, 1.414213562, 1.732050808,  // Layer z=-1: corners, edges, corners
    1.414213562, 1.000000000, 1.414213562,
    1.732050808, 1.414213562, 1.732050808,
    1.414213562, 1.000000000, 1.414213562,  // Layer z=0: edges, faces, edges
                 1.000000000,                // (skip center)
    1.414213562, 1.000000000, 1.414213562,
    1.732050808, 1.414213562, 1.732050808,  // Layer z=1: corners, edges, corners
    1.414213562, 1.000000000, 1.414213562,
    1.732050808, 1.414213562, 1.732050808
  };

  // Solidification kernel: parallel over all local sites with event tracking
  int naccept_local = 0;
  Kokkos::parallel_reduce("SolidificationKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_current(i);

      // Check if site is liquid and below liquidus
      if (active_flag_i == 2 && T_i < tl_local) {
        double Tcool = tl_local - T_i;
        double solid_d_i = d_solid_d_current(i);

        // Advance solid front distance: solid_d += V(Tcool) * dt
        double V_advance = 0.0;
        double Tcool_power = 1.0;
        for (int k = solid_front_length_local - 1; k >= 0; k--) {
          V_advance += d_solid_front_coeffs(k) * Tcool_power;
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
          int nbr_flag = d_active_flag_current(nbr);

          // Check if neighbor is solid (active_flag == 1 or 3) and within reach
          if ((nbr_flag == 1 || nbr_flag == 3) && neigh_dist[j] * dx_local <= solid_d_i) {
            // Compute gradient on-demand (only once, on first solid neighbor)
            if (!gradient_computed) {
              // Use ThermalOps to compute temperature gradient at site i
              ThermalOps::compute_temperature_gradient(
                i, d_T_local, d_active_flag_current, k_numneigh_d, k_neighbor_d, k_xyz_d, dx_local,
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
              d_q0_current(nbr),
              d_qx_current(nbr),
              d_qy_current(nbr),
              d_qz_current(nbr)
            };

            // Compute texture weight: mobility = c1 + c2 * cos(c3 * theta)
            // where theta is angle between crystal <100> and gradient
            double c1 = d_texture_params(0);
            double c2 = d_texture_params(1);
            double c3 = d_texture_params(2);

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
          int new_spin = d_spin_current(selected_nbr);
          double q_nbr_inherited[4] = {
            d_q0_current(selected_nbr),
            d_qx_current(selected_nbr),
            d_qy_current(selected_nbr),
            d_qz_current(selected_nbr)
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

            // Rotate quaternion toward gradient direction
            double q_rotated[4];
            QuaternionOps::rotate_q_towards_u(q_nbr_inherited, grad_dir, mis_angle, q_rotated);

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
            d_q0_next(i) = q_rotated[0];
            d_qx_next(i) = q_rotated[1];
            d_qy_next(i) = q_rotated[2];
            d_qz_next(i) = q_rotated[3];
          } else {
            // No misorientation - use neighbor's orientation as-is
            d_q0_next(i) = q_nbr_inherited[0];
            d_qx_next(i) = q_nbr_inherited[1];
            d_qy_next(i) = q_nbr_inherited[2];
            d_qz_next(i) = q_nbr_inherited[3];
          }

          // Assign grain ID (may be new grain if rotation exceeded threshold)
          d_spin_next(i) = new_spin;

          // Set to solid state
          d_active_flag_next(i) = 3;  // SOLID
            count++;  // Track nucleation event

          // Set solid_d flag to trigger smoothing countdown
          d_solid_d_next(i) = -1.0;

          // Record solidification parameters
          d_V_local(i) = V_advance;
          // Gradient already computed by gradient kernel

          // Return RNG state
          rand_pool_local.free_state(generator);
        } else {
          // No solid neighbors within reach - remain liquid
          d_spin_next(i) = d_spin_current(i);
          d_q0_next(i) = d_q0_current(i);
          d_qx_next(i) = d_qx_current(i);
          d_qy_next(i) = d_qy_current(i);
          d_qz_next(i) = d_qz_current(i);
          d_active_flag_next(i) = active_flag_i;  // Remain liquid
          d_solid_d_next(i) = solid_d_i;  // Update advanced distance
        }
      } else {
        // Not liquid or not undercooled - no change
        d_spin_next(i) = d_spin_current(i);
        d_q0_next(i) = d_q0_current(i);
        d_qx_next(i) = d_qx_current(i);
        d_qy_next(i) = d_qy_current(i);
        d_qz_next(i) = d_qz_current(i);
        d_active_flag_next(i) = active_flag_i;
        d_solid_d_next(i) = d_solid_d_current(i);
      }
    }, naccept_local);

  // Fence removed - single fence at end of app_update() handles all kernels

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

int AppAdditiveExtTempTextureKokkos::smoothing_kernel()
{
  // Capture all necessary data for device lambda
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  auto d_qx_current = buffer_qx.read_buffer;
  auto d_qx_next = buffer_qx.write_buffer;
  auto d_qy_current = buffer_qy.read_buffer;
  auto d_qy_next = buffer_qy.write_buffer;
  auto d_qz_current = buffer_qz.read_buffer;
  auto d_qz_next = buffer_qz.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  auto k_neighbor_d = k_neighbor.d_view;
  auto k_numneigh_d = k_numneigh.d_view;
  auto d_grad_x_local = d_grad_x;
  auto d_grad_y_local = d_grad_y;
  auto d_grad_z_local = d_grad_z;
  auto d_symmetries_local = d_symmetries;
  auto rand_pool_local = rand_pool;

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
  int naccept_local = 0;
  Kokkos::parallel_reduce("SmoothingKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      // Get current state
      int active_flag_i = d_active_flag_current(i);
      double solid_d_i = d_solid_d_current(i);

      // Only smooth sites in the refinement range: (-nrefine-1, 0)
      if (solid_d_i >= 0.0 || solid_d_i <= -nrefine_local - 1) {
        // Not eligible for smoothing - copy state unchanged
        d_spin_next(i) = d_spin_current(i);
        d_q0_next(i) = d_q0_current(i);
        d_qx_next(i) = d_qx_current(i);
        d_qy_next(i) = d_qy_current(i);
        d_qz_next(i) = d_qz_current(i);
        d_active_flag_next(i) = active_flag_i;

        // Increment solid_d if in smoothing range
        if (solid_d_i < 0.0 && solid_d_i > -nrefine_local - 1) {
          d_solid_d_next(i) = solid_d_i + 1.0;
          // If just reached 0, mark as ACTIVE
          if (d_solid_d_next(i) >= 0.0) {
            d_active_flag_next(i) = 1;  // ACTIVE
            d_solid_d_next(i) = 0.0;
          }
        } else {
          d_solid_d_next(i) = solid_d_i;
        }
        return;
      }

      // Site is eligible for smoothing
      int spin_i = d_spin_current(i);
      double q_i[4] = {d_q0_current(i), d_qx_current(i), d_qy_current(i), d_qz_current(i)};

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
        int nbr_flag = d_active_flag_current(nbr);

        // Only consider solid or active neighbors (not liquid, molten, or void)
        if (nbr_flag != 1 && nbr_flag != 3) continue;

        // Skip neighbors with same spin (no energy change possible)
        int nbr_spin = d_spin_current(nbr);
        if (nbr_spin == spin_i) continue;

        // Calculate texture weight: mobility = c1 + c2 * cos(c3 * theta)
        // where theta is angle between neighbor's orientation and gradient
        double q_nbr[4] = {
          d_q0_current(nbr), d_qx_current(nbr),
          d_qy_current(nbr), d_qz_current(nbr)
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
        d_spin_next(i) = spin_i;
        d_q0_next(i) = q_i[0];
        d_qx_next(i) = q_i[1];
        d_qy_next(i) = q_i[2];
        d_qz_next(i) = q_i[3];
        d_active_flag_next(i) = active_flag_i;
        d_solid_d_next(i) = solid_d_i + 1.0;
        if (d_solid_d_next(i) >= 0.0) {
          d_active_flag_next(i) = 1;  // ACTIVE
          d_solid_d_next(i) = 0.0;
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
      int new_spin = d_spin_current(selected_nbr);
      double q_new[4] = {
        d_q0_current(selected_nbr),
        d_qx_current(selected_nbr),
        d_qy_current(selected_nbr),
        d_qz_current(selected_nbr)
      };

      // Compute energy change using quaternion disorientation
      // Initial energy: sum over neighbors
      double energy_initial = 0.0;
      for (int j = 0; j < nn && j < 26; j++) {
        int nbr = k_neighbor_d(i, j);
        int nbr_flag = d_active_flag_current(nbr);

        // Skip void and non-solid neighbors
        if (nbr_flag == 5 || (nbr_flag != 3 && nbr_flag != 1)) continue;

        double q_nbr[4] = {
          d_q0_current(nbr), d_qx_current(nbr),
          d_qy_current(nbr), d_qz_current(nbr)
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
        int nbr_flag = d_active_flag_current(nbr);

        if (nbr_flag == 5 || (nbr_flag != 3 && nbr_flag != 1)) continue;

        double q_nbr[4] = {
          d_q0_current(nbr), d_qx_current(nbr),
          d_qy_current(nbr), d_qz_current(nbr)
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
        d_spin_next(i) = new_spin;
        d_q0_next(i) = q_new[0];
        d_qx_next(i) = q_new[1];
        d_qy_next(i) = q_new[2];
        d_qz_next(i) = q_new[3];
      } else {
        // Keep current state
        d_spin_next(i) = spin_i;
        d_q0_next(i) = q_i[0];
        d_qx_next(i) = q_i[1];
        d_qy_next(i) = q_i[2];
        d_qz_next(i) = q_i[3];
      }

      // Increment solid_d counter (moves toward 0)
      d_solid_d_next(i) = solid_d_i + 1.0;

      // If smoothing complete, mark as ACTIVE
      if (d_solid_d_next(i) >= 0.0) {
        d_active_flag_next(i) = 1;  // ACTIVE
        d_solid_d_next(i) = 0.0;
      } else {
        d_active_flag_next(i) = active_flag_i;
      }

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

int AppAdditiveExtTempTextureKokkos::smoothing_kernel_efficient(
    Kokkos::View<int*> d_smoothing_sites, int n_sites)
{
  // Capture all necessary data for device lambda
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  auto d_qx_current = buffer_qx.read_buffer;
  auto d_qx_next = buffer_qx.write_buffer;
  auto d_qy_current = buffer_qy.read_buffer;
  auto d_qy_next = buffer_qy.write_buffer;
  auto d_qz_current = buffer_qz.read_buffer;
  auto d_qz_next = buffer_qz.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  auto d_solid_d_next = buffer_solid_d.write_buffer;

  auto k_neighbor_d = k_neighbor.d_view;
  auto k_numneigh_d = k_numneigh.d_view;
  auto d_grad_x_local = d_grad_x;
  auto d_grad_y_local = d_grad_y;
  auto d_grad_z_local = d_grad_z;
  auto d_symmetries_local = d_symmetries;
  auto rand_pool_local = rand_pool;

  int nrefine_local = nrefine;
  double theta_cut_local = theta_cut;
  double temperature_local = temperature;
  double t_inverse_local = t_inverse;
  int num_symmetries_local = num_symmetries;

  // Texture parameters
  double c1_local = c1;
  double c2_local = c2;
  double c3_local = c3;

  // Parallel sweep over ONLY smoothing sites (not all sites!)
  int naccept_local = 0;
  Kokkos::parallel_reduce("SmoothingKernelEfficient", n_sites,
    KOKKOS_LAMBDA(int idx, int& count) {
      // Get actual site index from compact list
      int i = d_smoothing_sites(idx);

      // All sites in this list need smoothing - no branching needed!
      int active_flag_i = d_active_flag_current(i);
      int spin_i = d_spin_current(i);
      double solid_d_i = d_solid_d_current(i);
      double q_i[4] = {d_q0_current(i), d_qx_current(i), d_qy_current(i), d_qz_current(i)};

      // Get thermal gradient direction at this site
      double grad_norm = Kokkos::sqrt(
        d_grad_x_local(i) * d_grad_x_local(i) +
        d_grad_y_local(i) * d_grad_y_local(i) +
        d_grad_z_local(i) * d_grad_z_local(i)
      );

      double grad_dir[3] = {0.0, 0.0, 1.0};  // Default: +z direction
      if (grad_norm > 1.0e-10) {
        grad_dir[0] = d_grad_x_local(i) / grad_norm;
        grad_dir[1] = d_grad_y_local(i) / grad_norm;
        grad_dir[2] = d_grad_z_local(i) / grad_norm;
      }

      // Build list of unique solid/active neighbors
      int nn = k_numneigh_d(i);
      int unique_spins[27];
      double texture_weights[27];
      int unique_count = 0;

      for (int n = 0; n < nn; n++) {
        int nbr = k_neighbor_d(i, n);
        int nbr_spin = d_spin_current(nbr);
        int nbr_flag = d_active_flag_current(nbr);

        // Only consider solid/active neighbors
        if (nbr_spin != -1 && (nbr_flag == 1 || nbr_flag == 3)) {
          // Check if already in unique list
          bool found = false;
          for (int k = 0; k < unique_count; k++) {
            if (unique_spins[k] == nbr_spin) {
              found = true;
              break;
            }
          }

          if (!found && unique_count < 27) {
            unique_spins[unique_count] = nbr_spin;

            // Compute texture weight for this neighbor
            double q_nbr[4] = {
              d_q0_current(nbr), d_qx_current(nbr),
              d_qy_current(nbr), d_qz_current(nbr)
            };

            // Get cosine of angle between grain <100> and gradient
            double cos_min_angle = QuaternionOps::get_cosine_angle_q_u(q_nbr, grad_dir);
            double theta = Kokkos::acos(cos_min_angle);

            // Texture weight: c1 + c2*cos(c3*theta)
            double weight = c1_local + c2_local * Kokkos::cos(c3_local * theta);
            texture_weights[unique_count] = (weight > 0.0) ? weight : 0.0;

            unique_count++;
          }
        }
      }

      if (unique_count == 0) {
        // No neighbors - keep current state
        d_spin_next(i) = spin_i;
        d_q0_next(i) = q_i[0];
        d_qx_next(i) = q_i[1];
        d_qy_next(i) = q_i[2];
        d_qz_next(i) = q_i[3];
        d_active_flag_next(i) = active_flag_i;
        d_solid_d_next(i) = solid_d_i + 1.0;
        if (d_solid_d_next(i) >= 0.0) {
          d_active_flag_next(i) = 1;
          d_solid_d_next(i) = 0.0;
        }
        return;
      }

      // Select neighbor using texture-weighted probabilities
      auto generator = rand_pool_local.get_state();

      // Compute cumulative sum of weights
      double total_weight = 0.0;
      for (int k = 0; k < unique_count; k++) {
        total_weight += texture_weights[k];
      }

      // Weighted random selection
      double rand_val = generator.drand() * total_weight;
      double cumsum = 0.0;
      int selected_spin = unique_spins[0];
      int selected_nbr = -1;

      for (int k = 0; k < unique_count; k++) {
        cumsum += texture_weights[k];
        if (rand_val <= cumsum) {
          selected_spin = unique_spins[k];
          // Find actual neighbor index with this spin
          for (int n = 0; n < nn; n++) {
            int nbr = k_neighbor_d(i, n);
            if (d_spin_current(nbr) == selected_spin) {
              selected_nbr = nbr;
              break;
            }
          }
          break;
        }
      }

      if (selected_nbr < 0) selected_nbr = k_neighbor_d(i, 0);  // Fallback

      // Get proposed quaternion from selected neighbor
      double q_new[4] = {
        d_q0_current(selected_nbr), d_qx_current(selected_nbr),
        d_qy_current(selected_nbr), d_qz_current(selected_nbr)
      };
      int new_spin = d_spin_current(selected_nbr);

      // Compute energy change (Metropolis criterion)
      // Energy = sum of disorientations with neighbors
      double energy_initial = 0.0;
      double energy_final = 0.0;

      for (int n = 0; n < nn; n++) {
        int nbr = k_neighbor_d(i, n);
        double q_nbr[4] = {
          d_q0_current(nbr), d_qx_current(nbr),
          d_qy_current(nbr), d_qz_current(nbr)
        };

        // Initial energy
        double di_init = QuaternionOps::compute_disorientation(
          d_symmetries_local.data(), num_symmetries_local, q_i, q_nbr
        );
        double ratio_init = di_init / theta_cut_local;
        if (ratio_init > 0.0) {
          if (ratio_init < 1.0) {
            energy_initial += ratio_init * (1.0 - Kokkos::log(ratio_init));
          } else {
            energy_initial += 1.0;
          }
        }

        // Final energy
        double di_final = QuaternionOps::compute_disorientation(
          d_symmetries_local.data(), num_symmetries_local, q_new, q_nbr
        );
        double ratio_final = di_final / theta_cut_local;
        if (ratio_final > 0.0) {
          if (ratio_final < 1.0) {
            energy_final += ratio_final * (1.0 - Kokkos::log(ratio_final));
          } else {
            energy_final += 1.0;
          }
        }
      }
      energy_initial *= 0.5;
      energy_final *= 0.5;

      // Metropolis acceptance
      bool accept = false;
      if (energy_final <= energy_initial) {
        accept = true;
      } else if (temperature_local > 0.0) {
        double prob = Kokkos::exp((energy_initial - energy_final) * t_inverse_local);
        if (generator.drand() < prob) {
          accept = true;
        }
      }

      // Apply move
      if (accept) {
        d_spin_next(i) = new_spin;
        d_q0_next(i) = q_new[0];
        d_qx_next(i) = q_new[1];
        d_qy_next(i) = q_new[2];
        d_qz_next(i) = q_new[3];
        count++;
      } else {
        d_spin_next(i) = spin_i;
        d_q0_next(i) = q_i[0];
        d_qx_next(i) = q_i[1];
        d_qy_next(i) = q_i[2];
        d_qz_next(i) = q_i[3];
      }

      // Increment solid_d counter
      d_solid_d_next(i) = solid_d_i + 1.0;
      if (d_solid_d_next(i) >= 0.0) {
        d_active_flag_next(i) = 1;  // ACTIVE
        d_solid_d_next(i) = 0.0;
      } else {
        d_active_flag_next(i) = active_flag_i;
      }

      rand_pool_local.free_state(generator);
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   App update - main time integration loop
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::app_update(double dt_sweep)
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

  // 1. Update temperature from HDF5 source on CPU
  t_start = MPI_Wtime();
  update_temperature_from_source(time);
  t_end = MPI_Wtime();
  t_temp_update += (t_end - t_start);

  // Phase 3.2: Fast-forward logic - skip inactive thermal periods
  if (use_temperature_source && temperature_source) {
    // Determine fast-forward threshold
    double fast_forward_threshold = t_room;  // Default to room temperature

    // If using HDF5 source, use its specific threshold
    HDF5UnstructuredTemperatureSource* hdf5_source =
      dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
    if (hdf5_source) {
      fast_forward_threshold = hdf5_source->get_fast_forward_threshold();
    }

    // Check if any local sites have temperature above threshold
    t_active = 0;
    for (int i = 0; i < nlocal; i++) {
      if (T[i] > fast_forward_threshold) {
        t_active = 1;
        break;  // Found an active site
      }
    }

    // Global check across all MPI ranks
    int global_t_active = 0;
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, world);

    // If no thermal activity anywhere and source supports time queries
    if (global_t_active == 0 && time > 1e-6 && temperature_source->supports_time_queries()) {
      double next_thermal_time = temperature_source->get_next_time_with_temperature(time, fast_forward_threshold);

      if (next_thermal_time > time && next_thermal_time < std::numeric_limits<double>::max()) {
        double time_skip = next_thermal_time - time;
        if (domain->me == 0 && time_skip > 1e-6) {  // Report significant skips
          fprintf(screen, "Fast-forward: Skipping %.6f s from t=%.6f to t=%.6f (all T < %.1f K)\n",
                  time_skip, time, next_thermal_time, fast_forward_threshold);
        }

        // Jump to next thermal event
        time = next_thermal_time;

        // Update temperature at new time
        update_temperature_from_source(time);
      }
    }
  }

  // 2. Sync temperature to device
  t_start = MPI_Wtime();
  sync_temperature_to_device();
  t_end = MPI_Wtime();
  t_temp_sync += (t_end - t_start);

  // 3. Gradient computation moved to on-demand inside solidification kernel
  // (Removed global compute_gradients_kernel() - gradients now computed only when needed)

  // 4. Phase transition kernels with event tracking
  // Phase 3.1: Melting (with Phase 1 optimization: early temperature scan)
  t_start = MPI_Wtime();

  // OPTIMIZATION 1A: Quick pre-scan to check if ANY sites might melt
  // This avoids launching the expensive kernel when no melting occurs (~80% of timesteps)
  int n_melt_candidates = 0;
  {
    auto d_T_check = k_T.d_view;
    auto d_active_flag_check = buffer_active_flag.read_buffer;
    double tl_check = this->tl;
    int nlocal_check = this->nlocal;

    Kokkos::parallel_reduce("CheckMeltingCandidates", nlocal_check,
      KOKKOS_LAMBDA(int i, int& count) {
        if (d_T_check(i) >= tl_check && d_active_flag_check(i) != 2) {
          count++;
        }
      }, n_melt_candidates);
  }

  // Only launch melting kernel if there are candidates
  if (n_melt_candidates > 0) {
    naccept += melting_kernel();
    swap_buffers();  // Swap double buffers after melting
  } else {
    // No melting - buffers unchanged, no swap needed
    // (Sites remain in current state)
  }

  t_end = MPI_Wtime();
  t_melting += (t_end - t_start);

  // Phase 3.2: Nucleation (stochastic nucleation in undercooled liquid)
  // OPTIMIZATION 2A: Check if any sites are in undercooled liquid state before launching
  t_start = MPI_Wtime();

  int n_nucleation_candidates = 0;
  {
    auto d_T_check = k_T.d_view;
    auto d_active_flag_check = buffer_active_flag.read_buffer;
    double ts_check = this->ts;
    double tl_check = this->tl;
    int nlocal_check = this->nlocal;

    Kokkos::parallel_reduce("CheckNucleationCandidates", nlocal_check,
      KOKKOS_LAMBDA(int i, int& count) {
        // Nucleation occurs in undercooled liquid: ts <= T < tl && active_flag == 2
        if (d_active_flag_check(i) == 2 && d_T_check(i) >= ts_check && d_T_check(i) < tl_check) {
          count++;
        }
      }, n_nucleation_candidates);
  }

  // Only launch nucleation kernel if there are undercooled liquid sites
  if (n_nucleation_candidates > 0) {
    naccept += nucleation_kernel();
    swap_buffers();
  }

  t_end = MPI_Wtime();
  t_nucleation += (t_end - t_start);

  // Phase 3.5: Particle flipper (grow nuclei to specified size)
  // Only launch if there are newly nucleated sites (solid_d == -nrefine-2)
  t_start = MPI_Wtime();

  // Check if any nuclei need growing (selective kernel launch optimization)
  int n_nuclei_sites = 0;
  {
    auto d_solid_d_check = buffer_solid_d.read_buffer;
    auto d_active_flag_check = buffer_active_flag.read_buffer;
    int nrefine_check = nrefine;
    Kokkos::parallel_reduce("CheckNucleiSites", nlocal,
      KOKKOS_LAMBDA(int i, int& count) {
        if (d_active_flag_check(i) == 3 && d_solid_d_check(i) == -nrefine_check - 2) {
          count++;
        }
      }, n_nuclei_sites);
  }

  // Only launch particle kernel if there are nuclei to grow
  if (n_nuclei_sites > 0) {
    naccept += particle_flipper_kernel();
    swap_buffers();
  }

  t_end = MPI_Wtime();
  t_particle += (t_end - t_start);

  // Phase 3.3: Solidification (epitaxial growth with texture weighting)
  t_start = MPI_Wtime();
  naccept += solidification_kernel();
  swap_buffers();
  t_end = MPI_Wtime();
  t_solidification += (t_end - t_start);

  // Phase 3.4: Smoothing kernel (Potts MC for grain boundary refinement)
  // Call once per timestep - sites smooth over multiple timesteps as solid_d increments
  // OPTIMIZATION 3: Work-efficient smoothing - only process sites in smoothing range
  t_start = MPI_Wtime();

  // Build compact list of smoothing sites using parallel_scan (Phase 1 optimization)
  int n_smoothing_sites = 0;
  Kokkos::View<int*> d_smoothing_sites("smoothing_sites", nlocal);
  {
    auto d_solid_d_check = buffer_solid_d.read_buffer;
    int nrefine_check = nrefine;
    int nlocal_check = nlocal;

    // Parallel scan to build compact list of sites needing smoothing
    Kokkos::parallel_scan("BuildSmoothingList", nlocal_check,
      KOKKOS_LAMBDA(int i, int& offset, bool final) {
        double sd = d_solid_d_check(i);
        bool needs_smoothing = (sd > -nrefine_check - 1 && sd < 0.0);

        if (needs_smoothing) {
          if (final) {
            d_smoothing_sites(offset) = i;  // Store site index
          }
          offset++;
        }
      }, n_smoothing_sites);
  }

  // Only launch smoothing kernel if there are eligible sites
  // Process ONLY the sites in the compact list (not all 2.99M sites!)
  if (n_smoothing_sites > 0) {
    naccept += smoothing_kernel_efficient(d_smoothing_sites, n_smoothing_sites);
    swap_buffers();
  }

  t_end = MPI_Wtime();
  t_smoothing += (t_end - t_start);

  // 5. Single fence after all kernels complete (replaces individual kernel fences)
  // This ensures all device writes are visible before proceeding
  t_start = MPI_Wtime();
  Kokkos::fence();
  t_end = MPI_Wtime();
  t_sync_host += (t_end - t_start);

  // Sync device data to host for output/dumps
  // Ensures all iarray/darray fields are up-to-date for stats, dumps, and diagnostics
  // Overhead: ~1-3% of total runtime, but necessary for correct visualization/output
  sync_data_to_host();

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

  // Increment time
  time += dt_sweep;
}

/* ----------------------------------------------------------------------
   Override stats to sync device data before computing statistics
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::stats(char *strtmp)
{
  // Sync device data to host so stats can access current values
  sync_data_to_host();

  // Call parent class stats implementation
  AppPottsQuaternionKokkos::stats(strtmp);
}

void AppAdditiveExtTempTextureKokkos::stats_header(char *strtmp)
{
  // Just forward to parent - no sync needed for header
  AppPottsQuaternionKokkos::stats_header(strtmp);
}

/* ----------------------------------------------------------------------
   Process input commands
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::input_app(char *command, int narg, char **arg)
{
  // Handle temperature source setup (same as CPU version)
  if (strcmp(command, "setup_temperature_source") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal setup_temperature_source command");

    std::string source_type(arg[0]);

    if (source_type == "rosenthal") {
      temperature_source = std::make_unique<RosenthalTemperatureSource>(spk);
    } else if (source_type == "hdf5_unstructured") {
      temperature_source = std::make_unique<HDF5UnstructuredTemperatureSource>(spk);
    } else {
      error->all(FLERR, "Unknown temperature source type");
    }

    // Pass remaining args to temperature source for setup (skip arg[0] which is source type)
    std::vector<std::string> setup_args;
    for (int i = 1; i < narg; i++) {
      setup_args.push_back(std::string(arg[i]));
    }
    temperature_source->setup_temperature_source(setup_args);
    use_temperature_source = true;

  } else if (strcmp(command, "liquidus") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal liquidus command");
    tl = atof(arg[0]);

  } else if (strcmp(command, "solidus") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal solidus command");
    ts = atof(arg[0]);

  } else if (strcmp(command, "texture_parameters") == 0) {
    if (narg != 3) error->all(FLERR, "Illegal texture_parameters command");
    c1 = atof(arg[0]);
    c2 = atof(arg[1]);
    c3 = atof(arg[2]);

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
    fast_forward_search_window = atof(arg[0]);

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
    solid_front_coeffs = new double[solid_front_length];
    for (int i = 1; i < solid_front_length + 1; i++) {
      solid_front_coeffs[i - 1] = atof(arg[i]);
    }

  } else if (strcmp(command, "misorientation_function") == 0) {
    if (narg != 2 && narg != 3) error->all(FLERR, "Illegal misorientation_function command");
    max_misorient = atof(arg[0]);
    mis_thresh = atof(arg[1]) * MY_PI / 180.0; // Convert from degrees to radians
    if (narg == 3) {
      misorient_alpha = atof(arg[2]);
      if (misorient_alpha <= 0)
        error->all(FLERR, "Illegal misorientation_function alpha value: must be greater than 0");
    }

  } else if (strcmp(command, "sync_profiling") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal sync_profiling command");
    if (strcmp(arg[0], "yes") == 0) {
      enable_sync_profiling(true);
    } else if (strcmp(arg[0], "no") == 0) {
      enable_sync_profiling(false);
    } else {
      error->all(FLERR, "Illegal sync_profiling command: use 'yes' or 'no'");
    }

  } else {
    // Pass to parent class
    AppPottsQuaternionKokkos::input_app(command, narg, arg);
  }
}
