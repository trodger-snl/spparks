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
#include "app_potts_neigh_kokkos.h"
#include "random_park.h"
#include "error.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"

using namespace SPPARKS_NS;

enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};

/* ---------------------------------------------------------------------- */

AppPottsNeighKokkos::AppPottsNeighKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPottsNeigh(spk, narg, arg), rand_pool(nullptr)
{
  // Parse arguments for potts/neigh/kk
  if (narg != 2) error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");

  // Initialize RNG pool with a seed using world rank
  int me;
  MPI_Comm_rank(world, &me);
  rand_pool = new pool_type(12345 + me);
}

/* ---------------------------------------------------------------------- */

AppPottsNeighKokkos::~AppPottsNeighKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::grow_app()
{
  // Call parent
  AppPottsNeigh::grow_app();

  // Allocate/resize Kokkos views
  int nall = nlocal + nghost;

  // Allocate double buffer for spin using generalized framework
  buffer_spin.allocate("spin", nall);

  // Allocate neighbor arrays
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }
  k_propensity = Kokkos::DualView<double*>("propensity", nall);
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::init_app()
{
  // Call parent initialization
  AppPottsNeigh::init_app();

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

  // Sync spin data to device - use full version to initialize both buffers
  sync_data_to_device_full();
}

/* ----------------------------------------------------------------------
   sync all site data to device - FULL VERSION (both buffers)
   Used only during initialization
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::sync_data_to_device_full()
{
  buffer_spin.sync_to_device_full(spin);
}

/* ----------------------------------------------------------------------
   sync all site data to device
   Used after MPI communication to update ALL sites from host
   Must update all sites because after MPI comm, host has the authoritative data
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::sync_data_to_device()
{
  buffer_spin.sync_to_device(spin);
}

/* ----------------------------------------------------------------------
   sync all site data to host
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::sync_data_to_host()
{
  buffer_spin.sync_to_host(spin);
}

/* ----------------------------------------------------------------------
   swap read/write buffers after each sweep
   prevents race conditions from concurrent read/write to same data
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::swap_buffers()
{
  buffer_spin.swap();
}

/* ----------------------------------------------------------------------
   Rejection KMC sweep functor for neighbor-based rejection
   Key difference from AppPottsKokkos:
   - Builds unique list of neighboring spins (not random 1-nspins)
   - Uses null bin extending to maxneigh (not nspins)
------------------------------------------------------------------------- */

struct AppPottsNeighKokkos::RejectionSweepFunctor {
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_type;
  typedef int value_type;  // For parallel_reduce
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  view_type spin_read;   // Read from this buffer
  view_type spin_write;  // Write to this buffer
  view_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  pool_type rand_pool;
  int maxneigh;
  double temperature;
  double t_inverse;

  RejectionSweepFunctor(view_type spin_read_, view_type spin_write_,
                        view_type numneigh_,
                        Kokkos::View<int**> neighbor_,
                        pool_type& rand_pool_,
                        int maxneigh_, double temp_, double t_inv_) :
    spin_read(spin_read_), spin_write(spin_write_),
    numneigh(numneigh_), neighbor(neighbor_),
    rand_pool(rand_pool_), maxneigh(maxneigh_),
    temperature(temp_), t_inverse(t_inv_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& accepted) const {
    // Each thread gets its own RNG state from the pool
    auto rand_gen = rand_pool.get_state();

    // Get current spin from READ buffer
    int oldstate = spin_read(i);

    // Calculate current energy (reading from READ buffer)
    int eng_before = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (oldstate != spin_read(neighbor(i, j))) eng_before++;
    }

    // Build unique list of neighboring spin values
    // Use local array (stack allocation for thread)
    int unique[32];  // Assume maxneigh <= 32 for typical lattices
    int nevent = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int value = spin_read(neighbor(i, j));
      if (value == oldstate) continue;

      // Check if already in unique list
      bool found = false;
      for (int m = 0; m < nevent; m++) {
        if (value == unique[m]) {
          found = true;
          break;
        }
      }
      if (!found) unique[nevent++] = value;
    }

    // Pick random event from 0 to maxneigh-1 (null bin rejection)
    int iran = static_cast<int>(maxneigh * rand_gen.drand());

    // Null bin rejection - if iran >= nevent, no event occurs
    if (iran >= nevent) {
      // Copy old state to write buffer
      spin_write(i) = oldstate;
      rand_pool.free_state(rand_gen);
      return;
    }

    // Propose spin flip to selected neighbor value
    int newstate = unique[iran];

    // Calculate new energy (reading from READ buffer)
    int eng_after = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (newstate != spin_read(neighbor(i, j))) eng_after++;
    }

    // Metropolis acceptance
    double delta = eng_after - eng_before;
    int move_accepted = 0;
    if (delta <= 0.0) {
      spin_write(i) = newstate;
      move_accepted = 1;
    } else if (temperature > 0.0) {
      double prob = exp(-delta * t_inverse);
      if (rand_gen.drand() < prob) {
        spin_write(i) = newstate;
        move_accepted = 1;
      } else {
        // Rejected - copy old state to write buffer
        spin_write(i) = oldstate;
      }
    } else {
      // T=0 rejection - copy old state to write buffer
      spin_write(i) = oldstate;
    }

    // Return RNG state to pool
    rand_pool.free_state(rand_gen);

    accepted += move_accepted;
  }
};

/* ----------------------------------------------------------------------
   Override main iteration loop to use device kernels
   Same structure as AppPottsKokkos but with neighbor-based rejection
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::iterate_rejection(double stoptime)
{
  int i, icolor, nselect, nrange, jset;
  int *site2i;

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

void AppPottsNeighKokkos::sweep_rejection_kokkos()
{
  // Note: sync is handled in iterate_rejection after MPI communication
  // Don't re-sync here as it would overwrite device data with stale host data

  // Launch kernel with double-buffer: read from read_buffer, write to write_buffer
  RejectionSweepFunctor functor(
    buffer_spin.read_buffer,   // Read from immutable read buffer
    buffer_spin.write_buffer,  // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    *rand_pool,
    maxneigh,
    temperature,
    t_inverse
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPottsNeigh::RejectionSweep", nlocal, functor, naccept_local);
  Kokkos::fence();

  // Update global acceptance counter
  naccept += naccept_local;

  // Swap buffers: write_buffer becomes new read_buffer for next sweep
  swap_buffers();
}

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device - specific site list (for sectors)
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::sweep_rejection_kokkos(int nsite, int *site_list)
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

  // Create functor that operates on site list with double-buffer
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_type;

  struct SectorSweepFunctor {
    typedef int value_type;
    typedef Kokkos::Random_XorShift64_Pool<> pool_type;

    view_type spin_read;   // Read from this buffer
    view_type spin_write;  // Write to this buffer
    view_type numneigh;
    Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
    view_type site_indices;
    pool_type rand_pool;
    int maxneigh;
    double temperature;
    double t_inverse;

    SectorSweepFunctor(view_type spin_read_, view_type spin_write_,
                       view_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       view_type site_indices_,
                       pool_type& rand_pool_,
                       int maxneigh_, double temp_, double t_inv_) :
      spin_read(spin_read_), spin_write(spin_write_),
      numneigh(numneigh_), neighbor(neighbor_),
      site_indices(site_indices_),
      rand_pool(rand_pool_), maxneigh(maxneigh_),
      temperature(temp_), t_inverse(t_inv_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx, int& accepted) const {
      int i = site_indices(idx);

      auto rand_gen = rand_pool.get_state();

      // Get current spin from READ buffer
      int oldstate = spin_read(i);

      // Calculate current energy (reading from READ buffer)
      int eng_before = 0;
      for (int j = 0; j < numneigh(i); j++) {
        if (oldstate != spin_read(neighbor(i, j))) eng_before++;
      }

      // Build unique list of neighboring spin values
      int unique[32];
      int nevent = 0;

      for (int j = 0; j < numneigh(i); j++) {
        int value = spin_read(neighbor(i, j));
        if (value == oldstate) continue;

        bool found = false;
        for (int m = 0; m < nevent; m++) {
          if (value == unique[m]) {
            found = true;
            break;
          }
        }
        if (!found) unique[nevent++] = value;
      }

      // Pick random event with null bin
      int iran = static_cast<int>(maxneigh * rand_gen.drand());

      if (iran >= nevent) {
        // Copy old state to write buffer
        spin_write(i) = oldstate;
        rand_pool.free_state(rand_gen);
        return;
      }

      int newstate = unique[iran];

      // Calculate new energy (reading from READ buffer)
      int eng_after = 0;
      for (int j = 0; j < numneigh(i); j++) {
        if (newstate != spin_read(neighbor(i, j))) eng_after++;
      }

      // Metropolis acceptance
      double delta = eng_after - eng_before;
      int move_accepted = 0;
      if (delta <= 0.0) {
        spin_write(i) = newstate;
        move_accepted = 1;
      } else if (temperature > 0.0) {
        double prob = exp(-delta * t_inverse);
        if (rand_gen.drand() < prob) {
          spin_write(i) = newstate;
          move_accepted = 1;
        } else {
          // Rejected - copy old state to write buffer
          spin_write(i) = oldstate;
        }
      } else {
        // T=0 rejection - copy old state to write buffer
        spin_write(i) = oldstate;
      }

      rand_pool.free_state(rand_gen);

      accepted += move_accepted;
    }
  };

  // Launch kernel over site list with double-buffer
  SectorSweepFunctor functor(
    buffer_spin.read_buffer,   // Read from immutable read buffer
    buffer_spin.write_buffer,  // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    d_site_list,
    *rand_pool,
    maxneigh,
    temperature,
    t_inverse
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPottsNeigh::SectorSweep", nsite, functor, naccept_local);
  Kokkos::fence();

  naccept += naccept_local;

  // Swap buffers: write_buffer becomes new read_buffer for next sweep
  swap_buffers();
}
