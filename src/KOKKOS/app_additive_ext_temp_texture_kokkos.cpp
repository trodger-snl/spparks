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
#include "error.h"
#include "random_park.h"
#include "timer.h"
#include "math_const.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_hdf5_unstructured.h"
#include <cstring>

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
  // Gradients are device-only, computed on device, no host sync needed
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

  // Optionally sync gradients to host if needed for CPU operations
  // For now, G and V are computed on device and only needed for output
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
        i, d_T, k_numneigh_d, k_neighbor_d, k_xyz_d,
        dx_lattice, gradient, &magnitude);

      // Cache results for later use
      d_G_local(i) = magnitude;
      d_grad_x_local(i) = gradient[0];
      d_grad_y_local(i) = gradient[1];
      d_grad_z_local(i) = gradient[2];
    });

  Kokkos::fence();  // Ensure gradients are computed before next kernel
}

/* ----------------------------------------------------------------------
   App update - main time integration loop (placeholder for Phase 3)
------------------------------------------------------------------------- */

void AppAdditiveExtTempTextureKokkos::app_update(double dt_sweep)
{
  // PHASE 2: Temperature update and gradient computation for validation
  // PHASE 3 will add phase transition kernels (melting, nucleation, solidification)

  // 1. Update temperature from HDF5 source on CPU
  update_temperature_from_source(time);

  // 2. Sync temperature to device
  sync_temperature_to_device();

  // 3. Compute thermal gradients on device (cache for later use)
  compute_gradients_kernel();

  // 4. Sync gradients back to host for validation/output
  // This allows us to verify gradient computation against CPU version
  int nall = nlocal + nghost;
  auto h_G = Kokkos::create_mirror_view(d_G);
  auto h_V = Kokkos::create_mirror_view(d_V);
  Kokkos::deep_copy(h_G, d_G);
  Kokkos::deep_copy(h_V, d_V);
  for (int i = 0; i < nall; i++) {
    G[i] = h_G(i);
    V[i] = h_V(i);
  }

  // TODO Phase 3: Add phase transition kernels here
  // - Melting kernel (T >= T_liquidus)
  // - Nucleation kernel (molten + undercooled)
  // - Solidification kernel (epitaxial growth with texture weighting)

  // For now, just call parent class grain growth (if not app_update_only)
  if (!app_update_only) {
    // Parent class handles grain growth
    // AppPottsQuaternionKokkos::iterate_rejection(dt_sweep);
  }

  // Increment time
  time += dt_sweep;
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
