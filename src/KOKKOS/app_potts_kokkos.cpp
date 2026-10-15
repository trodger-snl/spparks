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
#include "solve.h"
#include "string.h"
#include "stdlib.h"
#include "app_potts_kokkos.h"
#include "random_park.h"
#include "random_park_kokkos.h"
#include "error.h"
#include "memory.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsKokkos::AppPottsKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLatticeKokkos(spk, narg, arg)
{
  // Potts-specific setup (from AppPotts constructor)
  ninteger = 1;
  ndouble = 0;
  delpropensity = 1;
  delevent = 0;
  allow_kmc = 1;
  allow_rejection = 1;
  allow_masking = 1;
  numrandom = 1;

  create_arrays();

  // Parse arguments
  if (narg != 2) error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");
  dt_sweep = 1.0/nspins;

  // Initialize pointers
  spin = nullptr;
  sites = nullptr;
  unique = nullptr;
}

/* ---------------------------------------------------------------------- */

AppPottsKokkos::~AppPottsKokkos()
{
  // Memory cleanup for temp arrays
  memory->destroy(sites);
  memory->destroy(unique);
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppPottsKokkos::grow_app()
{
  // Set spin pointer to iarray[0]
  spin = iarray[0];

  // Call parent (allocates neighbors and calls allocate_app_views)
  AppLatticeKokkos::grow_app();
}

/* ----------------------------------------------------------------------
   allocate app-specific Kokkos views
------------------------------------------------------------------------- */

void AppPottsKokkos::allocate_app_views()
{
  int nall = nlocal + nghost;
  d_spin = Kokkos::View<int*>("spin", nall);
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppPottsKokkos::init_app()
{
  // Validate spin values
  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (spin[i] < 1 || spin[i] > nspins) flag = 1;
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");

  // Allocate temp arrays for KMC if needed
  if (allow_kmc) {
    memory->destroy(sites);
    memory->destroy(unique);
    memory->create(sites, 1 + maxneigh, "app:sites");
    memory->create(unique, 1 + maxneigh, "app:unique");
  }

  // Call parent (validates coloring, sets up color lists, syncs neighbors)
  AppLatticeKokkos::init_app();

  // Initial sync of spin data to device
  sync_app_data_to_device();
}

/* ----------------------------------------------------------------------
   sync spin data to device
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_app_data_to_device()
{
  int nall = nlocal + nghost;
  auto h_spin = Kokkos::create_mirror_view(d_spin);
  for (int i = 0; i < nall; i++) {
    h_spin(i) = spin[i];
  }
  Kokkos::deep_copy(d_spin, h_spin);
}

/* ----------------------------------------------------------------------
   sync spin data to host
------------------------------------------------------------------------- */

void AppPottsKokkos::sync_app_data_to_host()
{
  auto h_spin = Kokkos::create_mirror_view(d_spin);
  Kokkos::deep_copy(h_spin, d_spin);
  for (int i = 0; i < nlocal; i++) {
    spin[i] = h_spin(i);
  }
}

/* ----------------------------------------------------------------------
   compute energy of site i
------------------------------------------------------------------------- */

double AppPottsKokkos::site_energy(int i)
{
  int isite = spin[i];
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (isite != spin[neighbor[i][j]]) eng++;
  return (double) eng;
}

/* ----------------------------------------------------------------------
   rKMC method - pick random spin
------------------------------------------------------------------------- */

void AppPottsKokkos::site_event_rejection(int i, RandomPark *random)
{
  int oldstate = spin[i];
  double einitial = site_energy(i);

  // Pick random spin
  int iran = (int) (nspins*random->uniform()) + 1;
  if (iran > nspins) iran = nspins;
  spin[i] = iran;
  double efinal = site_energy(i);

  // Metropolis acceptance
  if (efinal <= einitial) {
    // accept
  } else if (temperature == 0.0) {
    spin[i] = oldstate;
  } else if (random->uniform() > exp((einitial-efinal)*t_inverse)) {
    spin[i] = oldstate;
  }

  if (spin[i] != oldstate) naccept++;
}

/* ----------------------------------------------------------------------
   KMC propensity, same scheme as AppPotts::site_propensity
   events = flips to each distinct unlike neighbour spin
   each weighs 1 if downhill or level, exp(-dE/kT) if uphill
------------------------------------------------------------------------- */

double AppPottsKokkos::site_propensity(int i)
{
  int oldstate = spin[i];
  int nevent = 0;

  for (int j = 0; j < numneigh[i]; j++) {
    int value = spin[neighbor[i][j]];
    if (value == oldstate) continue;
    int m;
    for (m = 0; m < nevent; m++)
      if (value == unique[m]) break;
    if (m < nevent) continue;
    unique[nevent++] = value;
  }

  double einitial = site_energy(i);
  double prob = 0.0;

  for (int m = 0; m < nevent; m++) {
    spin[i] = unique[m];
    double efinal = site_energy(i);
    if (efinal <= einitial) prob += 1.0;
    else if (temperature > 0.0) prob += exp((einitial-efinal)*t_inverse);
  }

  spin[i] = oldstate;
  return prob;
}

/* ----------------------------------------------------------------------
   KMC event, same scheme as AppPotts::site_event
   pick a flip in proportion to its weight, then update the propensities
   of the site and its neighbours
------------------------------------------------------------------------- */

void AppPottsKokkos::site_event(int i, RandomPark *random)
{
  double threshhold = random->uniform() * propensity[i2site[i]];

  int oldstate = spin[i];
  double einitial = site_energy(i);
  double prob = 0.0;
  int nevent = 0;

  for (int j = 0; j < numneigh[i]; j++) {
    int value = spin[neighbor[i][j]];
    if (value == oldstate) continue;
    int m;
    for (m = 0; m < nevent; m++)
      if (value == unique[m]) break;
    if (m < nevent) continue;
    unique[nevent++] = value;

    spin[i] = value;
    double efinal = site_energy(i);
    if (efinal <= einitial) prob += 1.0;
    else if (temperature > 0.0) prob += exp((einitial-efinal)*t_inverse);
    if (prob >= threshhold) break;
  }

  // ignore neighbour sites with isite < 0 (ghosts)

  int nsites = 0;
  int isite = i2site[i];
  sites[nsites++] = isite;
  propensity[isite] = site_propensity(i);

  for (int j = 0; j < numneigh[i]; j++) {
    int m = neighbor[i][j];
    isite = i2site[m];
    if (isite < 0) continue;
    sites[nsites++] = isite;
    propensity[isite] = site_propensity(m);
  }

  solve->update(nsites,sites,propensity);
}

/* ----------------------------------------------------------------------
   Kokkos sweep functor for one color
------------------------------------------------------------------------- */

struct AppPottsKokkos::ColorSweepFunctor {
  Kokkos::View<int*> spin;
  Kokkos::View<int*> numneigh;
  Kokkos::View<int**> neighbor;
  Kokkos::View<int*> color_sites;
  int color_start;
  pool_type rand_pool;
  double temperature;
  double t_inverse;
  int nspins;
  Kokkos::View<int*> siteseeds;   // per-site streams; empty unless strict
  int strict;                     // 1 iff sweep color/strict

  // The site update, written once and templated over which RNG supplies the
  // draws (Kokkos pool generator or RandomParkDevice -- both expose drand()).
  // The draw SEQUENCE mirrors AppPotts::site_event_rejection exactly: one draw
  // for iran, and a second only when efinal > einitial && temperature > 0.
  // That is what lets the strict path reproduce the CPU sweep bit-for-bit.
  template <class Gen>
  KOKKOS_INLINE_FUNCTION
  void update_site(int i, Gen& gen, int& accepted) const {
    int oldstate = spin(i);

    // Compute initial energy (count unlike neighbors)
    int einitial = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i,j)) != oldstate) einitial++;
    }

    // Pick random new spin
    int iran = static_cast<int>(gen.drand() * nspins) + 1;
    if (iran > nspins) iran = nspins;

    // If same as old, nothing to do (CPU: efinal == einitial, so it likewise
    // accepts a no-op without taking a second draw)
    if (iran == oldstate) return;

    // Compute final energy
    int efinal = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i,j)) != iran) efinal++;
    }

    // Metropolis acceptance
    bool accept = false;
    if (efinal <= einitial) {
      accept = true;
    } else if (temperature > 0.0) {
      double prob = Kokkos::exp((einitial - efinal) * t_inverse);
      if (gen.drand() < prob) accept = true;
    }

    if (accept) {
      spin(i) = iran;  // Safe - no same-color neighbors!
      accepted++;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx, int& accepted) const {
    int i = color_sites(color_start + idx);

    if (strict) {
      // Load this site's own stream, advance it, store it back -- the device
      // equivalent of AppLattice::sweep_nomask_strict's siteseeds[] handling.
      // No pool, so no lock race and no thread-dependent stream assignment.
      RandomParkDevice rp(siteseeds(i));
      update_site(i, rp, accepted);
      siteseeds(i) = rp.seed;
    } else {
      auto rand_gen = rand_pool.get_state();
      update_site(i, rand_gen, accepted);
      rand_pool.free_state(rand_gen);
    }
  }
};

/* ----------------------------------------------------------------------
   sweep all sites of one color on device
------------------------------------------------------------------------- */

void AppPottsKokkos::sweep_color_kokkos(int color, int start, int count)
{
  ColorSweepFunctor functor{
    d_spin,
    k_numneigh.view_device(),
    k_neighbor.view_device(),
    d_color_sites,
    start,
    *rand_pool,
    temperature,
    t_inverse,
    nspins,
    d_siteseeds,
    strict_rng
  };

  int naccept_color = 0;
  Kokkos::parallel_reduce("PottsSweep", count, functor, naccept_color);
  naccept += naccept_color;
}
