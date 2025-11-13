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

using namespace SPPARKS_NS;

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
  // Don't copy data here - it will be uninitialized
  // Data copying happens in init_app() after sites are set
  int nall = nlocal + nghost;
  k_spin = Kokkos::DualView<int*>("spin", nall);

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

  // Sync spin data to device
  sync_spin_to_device();
}

/* ----------------------------------------------------------------------
   sync spin data to device
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_spin_to_device()
{
  int nall = nlocal + nghost;
  auto h_spin = k_spin.h_view;
  for (int i = 0; i < nall; i++) {
    h_spin(i) = spin[i];
  }
  k_spin.modify_host();
  k_spin.sync_device();
}

/* ----------------------------------------------------------------------
   sync spin data to host
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_spin_to_host()
{
  int nall = nlocal + nghost;
  k_spin.sync_host();
  auto h_spin = k_spin.h_view;
  for (int i = 0; i < nall; i++) {
    spin[i] = h_spin(i);
  }
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
  k_spin.sync_device();
  k_numneigh.sync_device();
  k_neighbor.sync_device();

  // Allocate energy array on device
  Kokkos::View<double*> d_energy("energy", nlocal);

  // Launch kernel
  ComputeEnergyFunctor functor(
    k_spin.d_view,
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

  view_type spin;
  view_type numneigh;
  Kokkos::View<int**, Kokkos::DefaultExecutionSpace::memory_space> neighbor;
  typename Kokkos::Random_XorShift64_Pool<>::generator_type rand_gen;
  int nspins;
  double temperature;
  double t_inverse;

  RejectionSweepFunctor(view_type spin_, view_type numneigh_,
                        Kokkos::View<int**> neighbor_,
                        typename Kokkos::Random_XorShift64_Pool<>::generator_type rand_gen_,
                        int nspins_, double temp_, double t_inv_) :
    spin(spin_), numneigh(numneigh_), neighbor(neighbor_),
    rand_gen(rand_gen_), nspins(nspins_), temperature(temp_), t_inverse(t_inv_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // Get current spin
    int oldstate = spin(i);

    // Calculate current energy
    int eng_before = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (oldstate != spin(neighbor(i, j))) eng_before++;
    }

    // Propose random new spin - make copy of RNG to avoid const issues
    auto local_rand = rand_gen;
    double rand_val = local_rand.drand();
    int iran = static_cast<int>(nspins * rand_val) + 1;
    if (iran > nspins) iran = nspins;

    // Calculate new energy
    int eng_after = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (iran != spin(neighbor(i, j))) eng_after++;
    }

    // Metropolis acceptance
    double delta = eng_after - eng_before;
    if (delta <= 0.0) {
      spin(i) = iran;
    } else if (temperature > 0.0) {
      double prob = exp(-delta * t_inverse);
      if (local_rand.drand() < prob) {
        spin(i) = iran;
      }
    }
  }
};

/* ----------------------------------------------------------------------
   Perform rejection KMC sweep on device
------------------------------------------------------------------------- */

void AppPottsKokkos::sweep_rejection_kokkos()
{
  // Sync to device
  k_spin.sync_device();
  k_numneigh.sync_device();
  k_neighbor.sync_device();

  // Get RNG state
  auto rand_gen = rand_pool->get_state();

  // Launch kernel
  RejectionSweepFunctor functor(
    k_spin.d_view,
    k_numneigh.d_view,
    k_neighbor.d_view,
    rand_gen,
    nspins,
    temperature,
    t_inverse
  );

  Kokkos::parallel_for("AppPotts::RejectionSweep", nlocal, functor);
  Kokkos::fence();

  // Return RNG state
  rand_pool->free_state(rand_gen);

  // Mark data as modified on device
  k_spin.modify_device();

  // Sync back to host if needed for I/O
  // sync_spin_to_host();
}
