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
  k_spin = Kokkos::DualView<int*>("spin", nall);

  // Quaternion arrays
  k_q0 = Kokkos::DualView<double*>("q0", nall);
  k_qx = Kokkos::DualView<double*>("qx", nall);
  k_qy = Kokkos::DualView<double*>("qy", nall);
  k_qz = Kokkos::DualView<double*>("qz", nall);

  // Allocate neighbor arrays
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }
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

  // Sync site data to device
  sync_data_to_device();
}

/* ----------------------------------------------------------------------
   sync all site data to device
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_data_to_device()
{
  int nall = nlocal + nghost;

  // Sync spin
  auto h_spin = k_spin.h_view;
  for (int i = 0; i < nall; i++) {
    h_spin(i) = spin[i];
  }
  k_spin.modify_host();
  k_spin.sync_device();

  // Sync quaternions
  auto h_q0 = k_q0.h_view;
  auto h_qx = k_qx.h_view;
  auto h_qy = k_qy.h_view;
  auto h_qz = k_qz.h_view;
  for (int i = 0; i < nall; i++) {
    h_q0(i) = q0[i];
    h_qx(i) = qx[i];
    h_qy(i) = qy[i];
    h_qz(i) = qz[i];
  }
  k_q0.modify_host();
  k_qx.modify_host();
  k_qy.modify_host();
  k_qz.modify_host();

  k_q0.sync_device();
  k_qx.sync_device();
  k_qy.sync_device();
  k_qz.sync_device();
}

/* ----------------------------------------------------------------------
   sync all site data to host
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_data_to_host()
{
  int nall = nlocal + nghost;

  // Sync from device to host views
  k_spin.sync_host();
  k_q0.sync_host();
  k_qx.sync_host();
  k_qy.sync_host();
  k_qz.sync_host();

  // Copy to CPU arrays
  auto h_spin = k_spin.h_view;
  auto h_q0 = k_q0.h_view;
  auto h_qx = k_qx.h_view;
  auto h_qy = k_qy.h_view;
  auto h_qz = k_qz.h_view;

  for (int i = 0; i < nall; i++) {
    spin[i] = h_spin(i);
    q0[i] = h_q0(i);
    qx[i] = h_qx(i);
    qy[i] = h_qy(i);
    qz[i] = h_qz(i);
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

  view_int_type spin;
  view_int_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  view_double_type q0, qx, qy, qz;
  view_double_type symmetries;
  int num_symmetries;
  double theta_cut;
  pool_type rand_pool;
  double temperature;
  double t_inverse;
  int maxneigh;

  RejectionSweepFunctor(view_int_type spin_, view_int_type numneigh_,
                        Kokkos::View<int**> neighbor_,
                        view_double_type q0_, view_double_type qx_,
                        view_double_type qy_, view_double_type qz_,
                        view_double_type symmetries_, int num_sym_,
                        double theta_cut_,
                        pool_type& rand_pool_,
                        double temp_, double t_inv_, int maxneigh_) :
    spin(spin_), numneigh(numneigh_), neighbor(neighbor_),
    q0(q0_), qx(qx_), qy(qy_), qz(qz_),
    symmetries(symmetries_), num_symmetries(num_sym_),
    theta_cut(theta_cut_), rand_pool(rand_pool_),
    temperature(temp_), t_inverse(t_inv_), maxneigh(maxneigh_) {}

  KOKKOS_INLINE_FUNCTION
  double compute_site_energy(int i, const double* qi) const {
    double eng = 0.0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

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

    // Get current state
    int oldstate = spin(i);
    double q_old[4] = {q0(i), qx(i), qy(i), qz(i)};

    // Calculate current energy
    double einitial = compute_site_energy(i, q_old);

    // Build list of unique neighboring spins
    int unique[100];  // Stack array, should be large enough
    int unique_neigh[100];
    int nevent = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      int value = spin(nj);
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

    // Temporarily update site with new state
    int newstate = unique[iran];
    double q_new[4] = {q0(neighran), qx(neighran), qy(neighran), qz(neighran)};

    // Calculate new energy (using temp quaternion, not modifying site yet)
    // We need to compute energy as if site i had q_new
    // This is tricky - we need to swap temporarily for energy calculation
    double efinal = 0.0;
    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor(i, j);
      double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

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

    // Apply move if accepted
    if (move_accepted) {
      spin(i) = newstate;
      q0(i) = q_new[0];
      qx(i) = q_new[1];
      qy(i) = q_new[2];
      qz(i) = q_new[3];
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
  // Sync to device (caller already does this, but check for safety)
  if (k_spin.need_sync_device()) sync_data_to_device();
  if (k_numneigh.need_sync_device()) k_numneigh.sync_device();
  if (k_neighbor.need_sync_device()) k_neighbor.sync_device();

  // Launch kernel with reduction to count accepted moves
  RejectionSweepFunctor functor(
    k_spin.d_view,
    k_numneigh.d_view,
    k_neighbor.d_view,
    k_q0.d_view,
    k_qx.d_view,
    k_qy.d_view,
    k_qz.d_view,
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

  // Update global acceptance counter
  naccept += naccept_local;

  // Mark data as modified on device
  k_spin.modify_device();
  k_q0.modify_device();
  k_qx.modify_device();
  k_qy.modify_device();
  k_qz.modify_device();
}

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device - specific site list (for sectors)
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sweep_rejection_kokkos(int nsite, int *site_list)
{
  // Sync to device
  if (k_spin.need_sync_device()) sync_data_to_device();
  if (k_numneigh.need_sync_device()) k_numneigh.sync_device();
  if (k_neighbor.need_sync_device()) k_neighbor.sync_device();

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

    view_int_type spin;
    view_int_type numneigh;
    Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
    view_double_type q0, qx, qy, qz;
    view_double_type symmetries;
    int num_symmetries;
    double theta_cut;
    view_int_type site_indices;
    pool_type rand_pool;
    double temperature;
    double t_inverse;
    int maxneigh;

    SectorSweepFunctor(view_int_type spin_, view_int_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       view_double_type q0_, view_double_type qx_,
                       view_double_type qy_, view_double_type qz_,
                       view_double_type symmetries_, int num_sym_,
                       double theta_cut_,
                       view_int_type site_indices_,
                       pool_type& rand_pool_,
                       double temp_, double t_inv_, int maxneigh_) :
      spin(spin_), numneigh(numneigh_), neighbor(neighbor_),
      q0(q0_), qx(qx_), qy(qy_), qz(qz_),
      symmetries(symmetries_), num_symmetries(num_sym_),
      theta_cut(theta_cut_), site_indices(site_indices_),
      rand_pool(rand_pool_), temperature(temp_), t_inverse(t_inv_),
      maxneigh(maxneigh_) {}

    KOKKOS_INLINE_FUNCTION
    double compute_site_energy(int i, const double* qi) const {
      double eng = 0.0;

      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

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

      // Get current state
      int oldstate = spin(i);
      double q_old[4] = {q0(i), qx(i), qy(i), qz(i)};

      // Calculate current energy
      double einitial = compute_site_energy(i, q_old);

      // Build list of unique neighboring spins
      int unique[100];
      int unique_neigh[100];
      int nevent = 0;

      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        int value = spin(nj);
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
      double q_new[4] = {q0(neighran), qx(neighran), qy(neighran), qz(neighran)};

      // Calculate new energy
      double efinal = 0.0;
      for (int j = 0; j < numneigh(i); j++) {
        int nj = neighbor(i, j);
        double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

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

      // Apply move if accepted
      if (move_accepted) {
        spin(i) = newstate;
        q0(i) = q_new[0];
        qx(i) = q_new[1];
        qy(i) = q_new[2];
        qz(i) = q_new[3];
      }

      rand_pool.free_state(rand_gen);

      accepted += move_accepted;
    }

    // Need access to QuaternionOps
    typedef AppPottsQuaternionKokkos::QuaternionOps QuaternionOps;
  };

  // Launch kernel over site list
  SectorSweepFunctor functor(
    k_spin.d_view,
    k_numneigh.d_view,
    k_neighbor.d_view,
    k_q0.d_view,
    k_qx.d_view,
    k_qy.d_view,
    k_qz.d_view,
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

  // Update global acceptance counter
  naccept += naccept_local;

  // Mark data as modified on device
  k_spin.modify_device();
  k_q0.modify_device();
  k_qx.modify_device();
  k_qy.modify_device();
  k_qz.modify_device();
}
