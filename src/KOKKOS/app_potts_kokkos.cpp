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
#include "app_potts_kokkos.h"
#include "random_park.h"
#include "error.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"

using namespace SPPARKS_NS;

enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};

/* ---------------------------------------------------------------------- */

AppPottsKokkos::AppPottsKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPotts(spk, narg, arg), rand_pool(nullptr)
{
  // Parse arguments for potts/kk - parent skips this for child classes
  if (narg != 2) error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");
  dt_sweep = 1.0/nspins;

  // Initialize RNG pool with a seed using world rank
  int me;
  MPI_Comm_rank(world, &me);
  rand_pool = new pool_type(12345 + me);
}

/* ---------------------------------------------------------------------- */

AppPottsKokkos::~AppPottsKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppPottsKokkos::grow_app()
{
  // Call parent
  AppPotts::grow_app();

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

void AppPottsKokkos::init_app()
{
  // Call parent initialization
  AppPotts::init_app();

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

  // Sync spin data to device (full sync for initialization)
  sync_data_to_device_full();
}

/* ----------------------------------------------------------------------
   sync spin data to device - FULL VERSION (both buffers)
   Used only during initialization
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_data_to_device_full()
{
  buffer_spin.sync_to_device_full(spin);
}

/* ----------------------------------------------------------------------
   sync spin data to device - MPI VERSION (both buffers)
   Used after MPI communication to update ghost sites
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_data_to_device()
{
  buffer_spin.sync_to_device(spin);
}

/* ----------------------------------------------------------------------
   sync spin data to host
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_data_to_host()
{
  buffer_spin.sync_to_host(spin);
}

/* ----------------------------------------------------------------------
   swap read/write buffers after each sweep
------------------------------------------------------------------------- */

void AppPottsKokkos::swap_buffers()
{
  buffer_spin.swap();
}

/* ----------------------------------------------------------------------
   compute energy of site - calls base class for now
------------------------------------------------------------------------- */

double AppPottsKokkos::site_energy(int i)
{
  return AppPotts::site_energy(i);
}

/* ----------------------------------------------------------------------
   Kokkos version of site energy
------------------------------------------------------------------------- */

double AppPottsKokkos::site_energy_kokkos(int i)
{
  return site_energy(i);
}

/* ----------------------------------------------------------------------
   rKMC method - calls base class for now
------------------------------------------------------------------------- */

void AppPottsKokkos::site_event_rejection(int i, RandomPark *random)
{
  AppPotts::site_event_rejection(i, random);
}

/* ----------------------------------------------------------------------
   Kokkos version of rejection event
------------------------------------------------------------------------- */

void AppPottsKokkos::site_event_rejection_kokkos(int i, RandomPark *random)
{
  site_event_rejection(i, random);
}

/* ----------------------------------------------------------------------
   KMC method - calls base class for now
------------------------------------------------------------------------- */

double AppPottsKokkos::site_propensity(int i)
{
  return AppPotts::site_propensity(i);
}

/* ----------------------------------------------------------------------
   Kokkos version of propensity
------------------------------------------------------------------------- */

double AppPottsKokkos::site_propensity_kokkos(int i)
{
  return site_propensity(i);
}

/* ----------------------------------------------------------------------
   KMC method - calls base class for now
------------------------------------------------------------------------- */

void AppPottsKokkos::site_event(int i, RandomPark *random)
{
  AppPotts::site_event(i, random);
}

/* ----------------------------------------------------------------------
   Energy calculation functor for Kokkos parallel_for
------------------------------------------------------------------------- */

struct AppPottsKokkos::ComputeEnergyFunctor {
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_type;

  view_type spin;
  view_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> energy;

  ComputeEnergyFunctor(view_type spin_, view_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       Kokkos::View<double*> energy_) :
    spin(spin_), numneigh(numneigh_), neighbor(neighbor_), energy(energy_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    int isite = spin(i);
    int eng = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int jsite = neighbor(i, j);
      if (isite != spin(jsite)) eng++;
    }

    energy(i) = static_cast<double>(eng);
  }
};

/* ----------------------------------------------------------------------
   Compute all energies in parallel on device
------------------------------------------------------------------------- */

void AppPottsKokkos::compute_all_energies_kokkos()
{
  // Sync data to device
  k_numneigh.sync_device();
  k_neighbor.sync_device();

  // Allocate energy array on device
  Kokkos::View<double*> d_energy("energy", nlocal);

  // Launch kernel (use read_buffer for reading spin data)
  ComputeEnergyFunctor functor(
    buffer_spin.read_buffer,
    k_numneigh.d_view,
    k_neighbor.d_view,
    d_energy
  );

  Kokkos::parallel_for("AppPotts::ComputeEnergies", nlocal, functor);
  Kokkos::fence();

  // Results are in d_energy - could copy back if needed
}

/* ----------------------------------------------------------------------
   Rejection KMC sweep functor
------------------------------------------------------------------------- */

struct AppPottsKokkos::RejectionSweepFunctor {
  typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> view_type;
  typedef int value_type;  // For parallel_reduce
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  view_type spin_read;   // Read from this buffer (immutable)
  view_type spin_write;  // Write to this buffer
  view_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  pool_type rand_pool;
  int nspins;
  double temperature;
  double t_inverse;

  RejectionSweepFunctor(view_type spin_read_, view_type spin_write_,
                        view_type numneigh_,
                        Kokkos::View<int**> neighbor_,
                        pool_type& rand_pool_,
                        int nspins_, double temp_, double t_inv_) :
    spin_read(spin_read_), spin_write(spin_write_), numneigh(numneigh_), neighbor(neighbor_),
    rand_pool(rand_pool_), nspins(nspins_), temperature(temp_), t_inverse(t_inv_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& accepted) const {
    // Each thread gets its own RNG state from the pool
    auto rand_gen = rand_pool.get_state();

    // Get current spin from read buffer
    int oldstate = spin_read(i);

    // Calculate current energy (read neighbors from read buffer)
    int eng_before = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (oldstate != spin_read(neighbor(i, j))) eng_before++;
    }

    // Propose random new spin
    double rand_val = rand_gen.drand();
    int iran = static_cast<int>(nspins * rand_val) + 1;
    if (iran > nspins) iran = nspins;

    // Calculate new energy (read neighbors from read buffer)
    int eng_after = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (iran != spin_read(neighbor(i, j))) eng_after++;
    }

    // Metropolis acceptance
    double delta = eng_after - eng_before;
    int move_accepted = 0;
    int newstate = oldstate;  // Default: keep old state

    if (delta <= 0.0) {
      newstate = iran;
      move_accepted = 1;
    } else if (temperature > 0.0) {
      double prob = exp(-delta * t_inverse);
      if (rand_gen.drand() < prob) {
        newstate = iran;
        move_accepted = 1;
      }
    }

    // ALWAYS write to write buffer (accepted OR rejected state)
    spin_write(i) = newstate;

    // Return RNG state to pool
    rand_pool.free_state(rand_gen);

    accepted += move_accepted;
  }
};

/* ----------------------------------------------------------------------
   Override main iteration loop to use device kernels
   Phase 2: Full sector and MPI support
------------------------------------------------------------------------- */

void AppPottsKokkos::iterate_rejection(double stoptime)
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
      // Need to get updated ghost sites from MPI (both buffers)
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

void AppPottsKokkos::sweep_rejection_kokkos()
{
  // Sync to device (caller already does this, but check for safety)
  if (k_numneigh.need_sync_device()) k_numneigh.sync_device();
  if (k_neighbor.need_sync_device()) k_neighbor.sync_device();

  // Launch kernel with double-buffer: read from read_buffer, write to write_buffer
  RejectionSweepFunctor functor(
    buffer_spin.read_buffer,   // Read from immutable read buffer
    buffer_spin.write_buffer,  // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    *rand_pool,
    nspins,
    temperature,
    t_inverse
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPotts::RejectionSweep", nlocal, functor, naccept_local);
  Kokkos::fence();

  // Update global acceptance counter
  naccept += naccept_local;

  // Swap buffers: write_buffer becomes new read_buffer for next sweep
  swap_buffers();
}

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device - specific site list (for sectors)
------------------------------------------------------------------------- */

void AppPottsKokkos::sweep_rejection_kokkos(int nsite, int *site_list)
{
  // Sync to device (caller already does this once per timestep)
  if (k_numneigh.need_sync_device()) k_numneigh.sync_device();
  if (k_neighbor.need_sync_device()) k_neighbor.sync_device();

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
    int nspins;
    double temperature;
    double t_inverse;

    SectorSweepFunctor(view_type spin_read_, view_type spin_write_,
                       view_type numneigh_,
                       Kokkos::View<int**> neighbor_,
                       view_type site_indices_,
                       pool_type& rand_pool_,
                       int nspins_, double temp_, double t_inv_) :
      spin_read(spin_read_), spin_write(spin_write_), numneigh(numneigh_), neighbor(neighbor_),
      site_indices(site_indices_),
      rand_pool(rand_pool_), nspins(nspins_),
      temperature(temp_), t_inverse(t_inv_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx, int& accepted) const {
      int i = site_indices(idx);

      auto rand_gen = rand_pool.get_state();

      // Get current spin from read buffer
      int oldstate = spin_read(i);

      // Calculate current energy (read neighbors from read buffer)
      int eng_before = 0;
      for (int j = 0; j < numneigh(i); j++) {
        if (oldstate != spin_read(neighbor(i, j))) eng_before++;
      }

      // Propose random new spin
      double rand_val = rand_gen.drand();
      int iran = static_cast<int>(nspins * rand_val) + 1;
      if (iran > nspins) iran = nspins;

      // Calculate new energy (read neighbors from read buffer)
      int eng_after = 0;
      for (int j = 0; j < numneigh(i); j++) {
        if (iran != spin_read(neighbor(i, j))) eng_after++;
      }

      // Metropolis acceptance
      double delta = eng_after - eng_before;
      int move_accepted = 0;
      int newstate = oldstate;  // Default: keep old state

      if (delta <= 0.0) {
        newstate = iran;
        move_accepted = 1;
      } else if (temperature > 0.0) {
        double prob = exp(-delta * t_inverse);
        if (rand_gen.drand() < prob) {
          newstate = iran;
          move_accepted = 1;
        }
      }

      // ALWAYS write to write buffer (accepted OR rejected state)
      spin_write(i) = newstate;

      rand_pool.free_state(rand_gen);

      accepted += move_accepted;
    }
  };

  // Launch kernel over site list
  SectorSweepFunctor functor(
    buffer_spin.read_buffer,   // Read from immutable read buffer
    buffer_spin.write_buffer,  // Write to write buffer
    k_numneigh.d_view,
    k_neighbor.d_view,
    d_site_list,
    *rand_pool,
    nspins,
    temperature,
    t_inverse
  );

  int naccept_local = 0;
  Kokkos::parallel_reduce("AppPotts::SectorSweep", nsite, functor, naccept_local);
  Kokkos::fence();

  naccept += naccept_local;

  // Swap buffers: write_buffer becomes new read_buffer for next sweep
  swap_buffers();
}
