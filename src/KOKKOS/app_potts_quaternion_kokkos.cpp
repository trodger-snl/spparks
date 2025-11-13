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
------------------------------------------------------------------------- */

#include "math.h"
#include "app_potts_quaternion_kokkos.h"
#include "random_park.h"
#include "error.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"
#include "crystalline_orientations/cubic_symmetries.h"
#include "crystalline_orientations/hcp_symmetries.h"

using namespace SPPARKS_NS;

enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};

/* ---------------------------------------------------------------------- */

AppPottsQuaternionKokkos::AppPottsQuaternionKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternion(spk, narg, arg), rand_pool(nullptr), num_symmetries(0)
{
  // Parent constructor returns early for child classes (style != "potts/quaternion")
  // So we need to parse arguments ourselves

  if (narg != 3 && narg != 4)
    error->all(FLERR, "Illegal 'potts/quaternion/kk' command");

  // Parse nspins
  nspins = atoi(arg[1]);
  if (nspins <= 0)
    error->all(FLERR, "Illegal app_style command");

  // Parse crystal structure
  if (strcmp(arg[2], "cubic") == 0) {
    symmetries = CUBIC::get_symmetries();
  } else if (strcmp(arg[2], "hcp") == 0) {
    symmetries = HCP::get_symmetries();
  } else {
    error->all(FLERR, "Illegal 'potts/quaternion/kk' command; expected "
                      "'cubic' or 'hcp'");
  }

  // Parse optional theta_cut
  if (4 == narg) {
    theta_cut = atof(arg[3]);
    if (theta_cut <= 0.0)
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; theta_cut must be positive");
    if (strcmp(arg[2], "cubic") == 0 && theta_cut > 62.7) {
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; "
                        "theta_cut for 'cubic' must "
                        "be between 0.0 and 62.7 degrees");
    } else if (strcmp(arg[2], "hcp") == 0 && theta_cut > 93.8) {
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; "
                        "theta_cut for 'hcp' must "
                        "be between 0.0 and 93.8 degrees");
    }
  }
  // theta_cut already has default value of 15.0 from parent class

  // Set up quaternion arrays (parent constructor skipped this)
  ndouble = 4;
  recreate_arrays();

  // Initialize RNG pool with a seed using world rank
  int me;
  MPI_Comm_rank(world, &me);
  rand_pool = new pool_type(12345 + me);

  // Copy symmetries to device
  num_symmetries = symmetries.size() / 4;
  d_symmetries = Kokkos::View<double*>("symmetries", symmetries.size());
  auto h_symmetries = Kokkos::create_mirror_view(d_symmetries);
  for (size_t i = 0; i < symmetries.size(); i++) {
    h_symmetries(i) = symmetries[i];
  }
  Kokkos::deep_copy(d_symmetries, h_symmetries);
}

/* ---------------------------------------------------------------------- */

AppPottsQuaternionKokkos::~AppPottsQuaternionKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::grow_app()
{
  // Don't call AppPottsQuaternion::grow_app() - it doesn't call parent chain
  // Instead, manually set pointers after parent classes allocate arrays
  // Call AppPotts::grow_app which will set spin pointer
  AppPotts::grow_app();

  // Set quaternion pointers to darray (parent constructor allocated these)
  q0 = darray[0];
  qx = darray[1];
  qy = darray[2];
  qz = darray[3];

  // Allocate/resize Kokkos views
  int nall = nlocal + nghost;

  // Allocate neighbor arrays
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }

  // Allocate double buffers using generalized framework
  buffer_spin.allocate("spin", nall);
  buffer_q0.allocate("q0", nall);
  buffer_qx.allocate("qx", nall);
  buffer_qy.allocate("qy", nall);
  buffer_qz.allocate("qz", nall);
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::init_app()
{
  // Call parent initialization (sets up sites array, random orientations, etc.)
  AppPottsQuaternion::init_app();

  // Sync neighbor arrays to device
  int nall = nlocal + nghost;
  auto h_numneigh = k_numneigh.h_view;
  for (int i = 0; i < nall; i++) {
    h_numneigh(i) = numneigh[i];
  }
  k_numneigh.modify_host();

  if (maxneigh > 0) {
    auto h_neighbor = k_neighbor.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < numneigh[i]; j++) {
        h_neighbor(i, j) = neighbor[i][j];
      }
    }
    k_neighbor.modify_host();
  }

  // Sync site data to device - use full version to initialize both buffers
  sync_data_to_device_full();
}

/* ----------------------------------------------------------------------
   sync all site data to device
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   sync all site data to device - FULL VERSION (both buffers)
   Used only during initialization
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_data_to_device_full()
{
  int nall = nlocal + nghost;

  // Sync spin directly (no transformation needed)
  buffer_spin.sync_to_device_full(spin);

  // Sync quaternions with normalization to prevent drift
  // Allocate temporary normalized arrays
  double* q0_norm = new double[nall];
  double* qx_norm = new double[nall];
  double* qy_norm = new double[nall];
  double* qz_norm = new double[nall];

  // Use helper to normalize quaternion arrays
  normalize_quaternion_arrays(q0_norm, qx_norm, qy_norm, qz_norm,
                               q0, qx, qy, qz, nall);

  // Use framework to sync normalized data
  buffer_q0.sync_to_device_full(q0_norm);
  buffer_qx.sync_to_device_full(qx_norm);
  buffer_qy.sync_to_device_full(qy_norm);
  buffer_qz.sync_to_device_full(qz_norm);

  delete[] q0_norm;
  delete[] qx_norm;
  delete[] qy_norm;
  delete[] qz_norm;
}

/* ----------------------------------------------------------------------
   sync all site data to device
   Used after MPI communication to update ALL sites from host
   Must update all sites because after MPI comm, host has the authoritative data
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_data_to_device()
{
  int nall = nlocal + nghost;

  // OPTIMIZATION: Only sync read_buffer after MPI communication
  // write_buffer will be overwritten in next sweep anyway
  // This reduces device memory copies by 33% (5 arrays × 2 buffers → 5 arrays × 1 buffer)

  // Sync spin directly (no transformation needed)
  buffer_spin.sync_to_device_selective(spin, true, false);

  // Sync quaternions with normalization to prevent drift
  // Allocate temporary normalized arrays
  double* q0_norm = new double[nall];
  double* qx_norm = new double[nall];
  double* qy_norm = new double[nall];
  double* qz_norm = new double[nall];

  // Use helper to normalize quaternion arrays
  normalize_quaternion_arrays(q0_norm, qx_norm, qy_norm, qz_norm,
                               q0, qx, qy, qz, nall);

  // Use framework to sync normalized data (only read_buffer for optimization)
  buffer_q0.sync_to_device_selective(q0_norm, true, false);
  buffer_qx.sync_to_device_selective(qx_norm, true, false);
  buffer_qy.sync_to_device_selective(qy_norm, true, false);
  buffer_qz.sync_to_device_selective(qz_norm, true, false);

  delete[] q0_norm;
  delete[] qx_norm;
  delete[] qy_norm;
  delete[] qz_norm;
}

/* ----------------------------------------------------------------------
   sync all site data to host
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_data_to_host()
{
  int nall = nlocal + nghost;

  // Sync spin directly (no transformation needed)
  buffer_spin.sync_to_host(spin);

  // Sync quaternions with normalization to prevent accumulation of errors
  // Allocate temporary arrays
  double* q0_temp = new double[nall];
  double* qx_temp = new double[nall];
  double* qy_temp = new double[nall];
  double* qz_temp = new double[nall];

  // Get data from device
  buffer_q0.sync_to_host(q0_temp);
  buffer_qx.sync_to_host(qx_temp);
  buffer_qy.sync_to_host(qy_temp);
  buffer_qz.sync_to_host(qz_temp);

  // Use helper to normalize and copy to final host arrays
  normalize_quaternion_arrays(q0, qx, qy, qz,
                               q0_temp, qx_temp, qy_temp, qz_temp,
                               nall);

  delete[] q0_temp;
  delete[] qx_temp;
  delete[] qy_temp;
  delete[] qz_temp;
}

/* ----------------------------------------------------------------------
   swap read/write buffers after each sweep
   prevents race conditions from concurrent read/write to same data
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::swap_buffers()
{
  // Use framework to swap all buffers
  buffer_spin.swap();
  buffer_q0.swap();
  buffer_qx.swap();
  buffer_qy.swap();
  buffer_qz.swap();
}

/* ----------------------------------------------------------------------
   normalize quaternion arrays on host
   prevents accumulation of numerical errors
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::normalize_quaternion_arrays(
    double* q0_out, double* qx_out, double* qy_out, double* qz_out,
    const double* q0_in, const double* qx_in,
    const double* qy_in, const double* qz_in,
    int n)
{
  for (int i = 0; i < n; i++) {
    double q[4] = {q0_in[i], qx_in[i], qy_in[i], qz_in[i]};
    double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      q0_out[i] = q[0] * inv_norm;
      qx_out[i] = q[1] * inv_norm;
      qy_out[i] = q[2] * inv_norm;
      qz_out[i] = q[3] * inv_norm;
    } else {
      q0_out[i] = q[0];
      qx_out[i] = q[1];
      qy_out[i] = q[2];
      qz_out[i] = q[3];
    }
  }
}

/* ----------------------------------------------------------------------
   compute energy of site - calls base class for now
------------------------------------------------------------------------- */

double AppPottsQuaternionKokkos::site_energy(int i)
{
  return AppPottsQuaternion::site_energy(i);
}

/* ----------------------------------------------------------------------
   rKMC method - calls base class for now
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::site_event_rejection(int i, RandomPark *random)
{
  AppPottsQuaternion::site_event_rejection(i, random);
}

/* ----------------------------------------------------------------------
   Energy calculation functor for Kokkos parallel_for
------------------------------------------------------------------------- */

struct AppPottsQuaternionKokkos::ComputeEnergyFunctor {
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_int_type;
  typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> view_double_type;

  view_int_type spin;
  view_int_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  view_double_type q0, qx, qy, qz;
  view_double_type symmetries;
  int num_symmetries;
  double theta_cut;
  view_double_type energy;

  ComputeEnergyFunctor(view_int_type spin_, view_int_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       view_double_type q0_, view_double_type qx_,
                       view_double_type qy_, view_double_type qz_,
                       view_double_type symmetries_, int num_sym_,
                       double theta_cut_, view_double_type energy_) :
    spin(spin_), numneigh(numneigh_), neighbor(neighbor_),
    q0(q0_), qx(qx_), qy(qy_), qz(qz_),
    symmetries(symmetries_), num_symmetries(num_sym_),
    theta_cut(theta_cut_), energy(energy_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    double eng = 0.0;
    double qi[4] = {q0(i), qx(i), qy(i), qz(i)};

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

      // Compute disorientation
      double di = QuaternionOps::compute_disorientation(
        symmetries.data(), num_symmetries, qi, qj);

      double ratio = di / theta_cut;

      // Read-Shockley equation logic
      if (ratio <= 0.0) {
        continue;
      } else if (ratio < 1.0) {
        eng += ratio * (1.0 - Kokkos::log(ratio));
      } else {
        eng += 1.0;
      }
    }

    // Each site carries half the grain boundary energy
    energy(i) = 0.5 * eng;
  }
};

/* ----------------------------------------------------------------------
   Rejection KMC sweep functor
------------------------------------------------------------------------- */

struct AppPottsQuaternionKokkos::RejectionSweepFunctor {
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_int_type;
  typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> view_double_type;
  typedef int value_type;  // For parallel_reduce
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  // Double-buffer approach: read from one, write to other
  view_int_type spin_read, spin_write;
  view_int_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  view_double_type q0_read, q0_write;
  view_double_type qx_read, qx_write;
  view_double_type qy_read, qy_write;
  view_double_type qz_read, qz_write;
  view_double_type symmetries;
  int num_symmetries;
  double theta_cut;
  pool_type rand_pool;
  double temperature;
  double t_inverse;
  int maxneigh;

  RejectionSweepFunctor(view_int_type spin_read_, view_int_type spin_write_,
                        view_int_type numneigh_,
                        Kokkos::View<int**> neighbor_,
                        view_double_type q0_read_, view_double_type q0_write_,
                        view_double_type qx_read_, view_double_type qx_write_,
                        view_double_type qy_read_, view_double_type qy_write_,
                        view_double_type qz_read_, view_double_type qz_write_,
                        view_double_type symmetries_, int num_sym_,
                        double theta_cut_,
                        pool_type& rand_pool_,
                        double temp_, double t_inv_, int maxneigh_) :
    spin_read(spin_read_), spin_write(spin_write_),
    numneigh(numneigh_), neighbor(neighbor_),
    q0_read(q0_read_), q0_write(q0_write_),
    qx_read(qx_read_), qx_write(qx_write_),
    qy_read(qy_read_), qy_write(qy_write_),
    qz_read(qz_read_), qz_write(qz_write_),
    symmetries(symmetries_), num_symmetries(num_sym_),
    theta_cut(theta_cut_), rand_pool(rand_pool_),
    temperature(temp_), t_inverse(t_inv_), maxneigh(maxneigh_) {}

  KOKKOS_INLINE_FUNCTION
  double compute_site_energy(int i, const double* qi) const {
    double eng = 0.0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      // Read neighbor quaternion from read buffer
      double qj[4] = {q0_read(nj), qx_read(nj), qy_read(nj), qz_read(nj)};

      double di = QuaternionOps::compute_disorientation(
        symmetries.data(), num_symmetries, qi, qj);

      double ratio = di / theta_cut;

      if (ratio <= 0.0) {
        continue;
      } else if (ratio < 1.0) {
        eng += ratio * (1.0 - Kokkos::log(ratio));
      } else {
        eng += 1.0;
      }
    }

    return 0.5 * eng;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& accepted) const {
    // Each thread gets its own RNG state from the pool
    auto rand_gen = rand_pool.get_state();

    // Get current state from READ buffer
    int oldstate = spin_read(i);
    double q_old[4] = {q0_read(i), qx_read(i), qy_read(i), qz_read(i)};

    // Calculate current energy
    double einitial = compute_site_energy(i, q_old);

    // Build list of unique neighboring spins
    int unique[100];  // Stack array, should be large enough
    int unique_neigh[100];
    int nevent = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      // Read neighbor spin from READ buffer
      int value = spin_read(nj);
      if (value == oldstate) continue;

      // Check if this spin value is already in unique list
      bool found = false;
      for (int m = 0; m < nevent; m++) {
        if (value == unique[m]) {
          found = true;
          break;
        }
      }

      if (!found && nevent < 100) {
        unique[nevent] = value;
        unique_neigh[nevent] = nj;
        nevent++;
      }
    }

    // If no unique neighbors, no event possible
    if (nevent == 0) {
      rand_pool.free_state(rand_gen);
      return;
    }

    // Randomly select one unique neighbor
    double rand_val = rand_gen.drand();
    int iran = static_cast<int>(nevent * rand_val);
    if (iran >= nevent) iran = nevent - 1;

    int neighran = unique_neigh[iran];

    // Read neighbor's state from READ buffer
    int newstate = unique[iran];
    double q_new[4] = {q0_read(neighran), qx_read(neighran), qy_read(neighran), qz_read(neighran)};

    // Normalize to ensure unit quaternion
    QuaternionOps::normalize(q_new);

    // Calculate new energy (using temp quaternion, not modifying site yet)
    // We need to compute energy as if site i had q_new
    // This is tricky - we need to swap temporarily for energy calculation
    double efinal = 0.0;
    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      // Read neighbor quaternion from READ buffer
      double qj[4] = {q0_read(nj), qx_read(nj), qy_read(nj), qz_read(nj)};

      double di = QuaternionOps::compute_disorientation(
        symmetries.data(), num_symmetries, q_new, qj);

      double ratio = di / theta_cut;

      if (ratio <= 0.0) {
        continue;
      } else if (ratio < 1.0) {
        efinal += ratio * (1.0 - Kokkos::log(ratio));
      } else {
        efinal += 1.0;
      }
    }
    efinal *= 0.5;

    // Metropolis acceptance
    int move_accepted = 0;
    if (efinal <= einitial) {
      move_accepted = 1;
    } else if (temperature > 0.0) {
      double prob = Kokkos::exp((einitial - efinal) * t_inverse);
      if (rand_gen.drand() < prob) {
        move_accepted = 1;
      }
    }

    // Write to WRITE buffer (either new state if accepted, or old state if rejected)
    if (move_accepted) {
      spin_write(i) = newstate;
      q0_write(i) = q_new[0];
      qx_write(i) = q_new[1];
      qy_write(i) = q_new[2];
      qz_write(i) = q_new[3];
    } else {
      // Copy old state to write buffer
      spin_write(i) = oldstate;
      q0_write(i) = q_old[0];
      qx_write(i) = q_old[1];
      qy_write(i) = q_old[2];
      qz_write(i) = q_old[3];
    }

    // Return RNG state to pool
    rand_pool.free_state(rand_gen);

    accepted += move_accepted;
  }
};

/* ----------------------------------------------------------------------
   Override main iteration loop to use device kernels
   Phase 2: Full sector and MPI support
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::iterate_rejection(double stoptime)
{
  int i, icolor, nselect, nrange, jset;
  int *site2i;

  // Set loop is over:
  // sectors if there are sectors and no colors
  // colors if there are colors and no sectors
  // first nsector sets if there are both sectors and colors

  int nset_loop = nset;
  if (bothflag) nset_loop = nsector;

  int done = 0;
  while (!done) {
    for (int iset = 0; iset < nset_loop; iset++) {
      // Sync device to host before MPI communication
      // MPI operates on host data, so we need current device state
      if (nprocs > 1 && iset > 0) {
        sync_data_to_host();
      }

      // MPI communication for this sector
      if (nprocs > 1) {
        timer->stamp();
        if (sectorflag) comm->sector(iset);
        else comm->all();
        timer->stamp(TIME_COMM);
      }

      if (Lmask) boundary_clear_mask(iset);

      timer->stamp();

      // Sync data to device after MPI communication
      // Need to get updated ghost sites from MPI
      sync_data_to_device();
      if (iset == 0) {
        if (!k_numneigh.need_sync_device()) k_numneigh.sync_device();
        if (!k_neighbor.need_sync_device()) k_neighbor.sync_device();
      }

      // Sectors but no colors (could also be no sectors)
      // Random selection of sites in iset
      if (sweepflag == RANDOM) {
        site2i = set[iset].site2i;
        nrange = set[iset].nlocal;
        nselect = set[iset].nselect;
        for (i = 0; i < nselect; i++)
          sitelist[i] = site2i[ranapp->irandom(nrange) - 1];
        sweep_rejection_kokkos(nselect, sitelist);
        nattempt += nselect;

      // Sectors but no colors, or colors but no sectors
      // Ordered sweep over all sites in iset
      } else if (bothflag == 0) {
        for (i = 0; i < set[iset].nloop; i++)
          sweep_rejection_kokkos(set[iset].nlocal, set[iset].site2i);
        nattempt += set[iset].nselect;

      // Sectors and colors
      // icolor loop is over all colors in a sector
      // jset = set that contains sites of one color in one sector
      // Ordered sweep over all sites in jset
      } else {
        for (icolor = 0; icolor < ncolors; icolor++) {
          jset = nsector + iset*ncolors + icolor;
          for (i = 0; i < set[jset].nloop; i++)
            sweep_rejection_kokkos(set[jset].nlocal, set[jset].site2i);
          nattempt += set[jset].nselect;
        }
      }

      timer->stamp(TIME_SOLVE);

      // Sync device to host before reverse MPI communication
      sync_data_to_host();

      if (nprocs > 1) {
        if (sectorflag) comm->reverse_sector(iset);
        else comm->all_reverse();
        timer->stamp(TIME_COMM);
      }
    }

    // Time advancement
    if (allow_app_update) app_update(dt_rkmc);

    nsweeps++;
    time += dt_rkmc;

    if (time >= stoptime) done = 1;
    if (done || time >= nextoutput) nextoutput = output->compute(time,done);
    timer->stamp(TIME_OUTPUT);
  }
}

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device - all local sites
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sweep_rejection_kokkos()
{
  // Note: sync is handled in iterate_rejection after MPI communication
  // Don't re-sync here as it would overwrite device data with stale host data

  // Launch kernel with reduction to count accepted moves
  RejectionSweepFunctor functor(
    buffer_spin.read_buffer,      // Read from read buffer
    buffer_spin.write_buffer,     // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    buffer_q0.read_buffer, buffer_q0.write_buffer,
    buffer_qx.read_buffer, buffer_qx.write_buffer,
    buffer_qy.read_buffer, buffer_qy.write_buffer,
    buffer_qz.read_buffer, buffer_qz.write_buffer,
    d_symmetries,
    num_symmetries,
    theta_cut,
    *rand_pool,
    temperature,
    t_inverse,
    maxneigh
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPottsQuaternion::RejectionSweep", nlocal, functor, naccept_local);
  Kokkos::fence();

  // Swap read/write buffers so next iteration reads the just-written data
  swap_buffers();

  // Update global acceptance counter
  naccept += naccept_local;
}

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device - specific site list (for sectors)
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sweep_rejection_kokkos(int nsite, int *site_list)
{
  // Note: sync is handled in iterate_rejection after MPI communication
  // Don't re-sync here as it would overwrite device data with stale host data

  // Copy site list to device
  Kokkos::View<int*> d_site_list("sector_sites", nsite);
  auto h_site_list = Kokkos::create_mirror_view(d_site_list);
  for (int i = 0; i < nsite; i++) {
    h_site_list(i) = site_list[i];
  }
  Kokkos::deep_copy(d_site_list, h_site_list);

  // Create sector sweep functor
  struct SectorSweepFunctor {
    typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_int_type;
    typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> view_double_type;
    typedef int value_type;
    typedef Kokkos::Random_XorShift64_Pool<> pool_type;

    view_int_type spin_read, spin_write;
    view_int_type numneigh;
    Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
    view_double_type q0_read, q0_write, qx_read, qx_write, qy_read, qy_write, qz_read, qz_write;
    view_double_type symmetries;
    int num_symmetries;
    double theta_cut;
    view_int_type site_indices;
    pool_type rand_pool;
    double temperature;
    double t_inverse;
    int maxneigh;

    SectorSweepFunctor(view_int_type spin_read_, view_int_type spin_write_,
                       view_int_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       view_double_type q0_read_, view_double_type q0_write_,
                       view_double_type qx_read_, view_double_type qx_write_,
                       view_double_type qy_read_, view_double_type qy_write_,
                       view_double_type qz_read_, view_double_type qz_write_,
                       view_double_type symmetries_, int num_sym_,
                       double theta_cut_,
                       view_int_type site_indices_,
                       pool_type& rand_pool_,
                       double temp_, double t_inv_, int maxneigh_) :
      spin_read(spin_read_), spin_write(spin_write_),
      numneigh(numneigh_), neighbor(neighbor_),
      q0_read(q0_read_), q0_write(q0_write_),
      qx_read(qx_read_), qx_write(qx_write_),
      qy_read(qy_read_), qy_write(qy_write_),
      qz_read(qz_read_), qz_write(qz_write_),
      symmetries(symmetries_), num_symmetries(num_sym_),
      theta_cut(theta_cut_), site_indices(site_indices_),
      rand_pool(rand_pool_), temperature(temp_), t_inverse(t_inv_),
      maxneigh(maxneigh_) {}

    KOKKOS_INLINE_FUNCTION
    double compute_site_energy(int i, const double* qi) const {
      double eng = 0.0;

      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        double qj[4] = {q0_read(nj), qx_read(nj), qy_read(nj), qz_read(nj)};

        double di = QuaternionOps::compute_disorientation(
          symmetries.data(), num_symmetries, qi, qj);

        double ratio = di / theta_cut;

        if (ratio <= 0.0) {
          continue;
        } else if (ratio < 1.0) {
          eng += ratio * (1.0 - Kokkos::log(ratio));
        } else {
          eng += 1.0;
        }
      }

      return 0.5 * eng;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx, int& accepted) const {
      // idx is index into site_indices array
      // i is the actual site index
      int i = site_indices(idx);

      // Each thread gets its own RNG state from the pool
      auto rand_gen = rand_pool.get_state();

      // Get current state from READ buffer
      int oldstate = spin_read(i);
      double q_old[4] = {q0_read(i), qx_read(i), qy_read(i), qz_read(i)};

      // Calculate current energy
      double einitial = compute_site_energy(i, q_old);

      // Build list of unique neighboring spins
      int unique[100];
      int unique_neigh[100];
      int nevent = 0;

      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        int value = spin_read(nj);
        if (value == oldstate) continue;

        bool found = false;
        for (int m = 0; m < nevent; m++) {
          if (value == unique[m]) {
            found = true;
            break;
          }
        }

        if (!found && nevent < 100) {
          unique[nevent] = value;
          unique_neigh[nevent] = nj;
          nevent++;
        }
      }

      if (nevent == 0) {
        rand_pool.free_state(rand_gen);
        return;
      }

      // Randomly select one unique neighbor
      double rand_val = rand_gen.drand();
      int iran = static_cast<int>(nevent * rand_val);
      if (iran >= nevent) iran = nevent - 1;

      int neighran = unique_neigh[iran];
      int newstate = unique[iran];
      double q_new[4] = {q0_read(neighran), qx_read(neighran), qy_read(neighran), qz_read(neighran)};

      // Normalize to ensure unit quaternion
      QuaternionOps::normalize(q_new);

      // Calculate new energy
      double efinal = 0.0;
      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        double qj[4] = {q0_read(nj), qx_read(nj), qy_read(nj), qz_read(nj)};

        double di = QuaternionOps::compute_disorientation(
          symmetries.data(), num_symmetries, q_new, qj);

        double ratio = di / theta_cut;

        if (ratio <= 0.0) {
          continue;
        } else if (ratio < 1.0) {
          efinal += ratio * (1.0 - Kokkos::log(ratio));
        } else {
          efinal += 1.0;
        }
      }
      efinal *= 0.5;

      // Metropolis acceptance
      int move_accepted = 0;
      if (efinal <= einitial) {
        move_accepted = 1;
      } else if (temperature > 0.0) {
        double prob = Kokkos::exp((einitial - efinal) * t_inverse);
        if (rand_gen.drand() < prob) {
          move_accepted = 1;
        }
      }

      // Write to WRITE buffer (either new state if accepted, or old state if rejected)
      if (move_accepted) {
        spin_write(i) = newstate;
        q0_write(i) = q_new[0];
        qx_write(i) = q_new[1];
        qy_write(i) = q_new[2];
        qz_write(i) = q_new[3];
      } else {
        // Copy old state to write buffer
        spin_write(i) = oldstate;
        q0_write(i) = q_old[0];
        qx_write(i) = q_old[1];
        qy_write(i) = q_old[2];
        qz_write(i) = q_old[3];
      }

      rand_pool.free_state(rand_gen);

      accepted += move_accepted;
    }

    // Need access to QuaternionOps
    typedef AppPottsQuaternionKokkos::QuaternionOps QuaternionOps;
  };

  // Launch kernel over site list
  SectorSweepFunctor functor(
    buffer_spin.read_buffer,      // Read from read buffer
    buffer_spin.write_buffer,     // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    buffer_q0.read_buffer, buffer_q0.write_buffer,
    buffer_qx.read_buffer, buffer_qx.write_buffer,
    buffer_qy.read_buffer, buffer_qy.write_buffer,
    buffer_qz.read_buffer, buffer_qz.write_buffer,
    d_symmetries,
    num_symmetries,
    theta_cut,
    d_site_list,
    *rand_pool,
    temperature,
    t_inverse,
    maxneigh
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPottsQuaternion::SectorSweep", nsite, functor, naccept_local);
  Kokkos::fence();

  // Swap read/write buffers so next iteration reads the just-written data
  swap_buffers();

  // Update global acceptance counter
  naccept += naccept_local;
}
