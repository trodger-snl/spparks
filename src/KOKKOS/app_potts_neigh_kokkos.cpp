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

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsNeighKokkos::AppPottsNeighKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPottsKokkos(spk, narg, arg) {}

/* ---------------------------------------------------------------------- */

void AppPottsNeighKokkos::init_app()
{
  dt_sweep = 1.0/maxneigh;   // matches AppPottsNeigh
  AppPottsKokkos::init_app();
}

/* ----------------------------------------------------------------------
   rKMC method - potts/neigh picks from unique neighbors with null bin
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::site_event_rejection(int i, RandomPark *random)
{
  int oldstate = spin[i];

  // Build list of unique neighbor spins
  int nevent = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    int value = spin[neighbor[i][j]];
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
  int iran = (int)(maxneigh * random->uniform());
  if (iran >= nevent) return;  // Null bin rejection

  double einitial = site_energy(i);
  spin[i] = unique[iran];
  double efinal = site_energy(i);

  // Metropolis acceptance
  if (efinal <= einitial) {
    // accept
  } else if (temperature == 0.0) {
    spin[i] = oldstate;
  } else if (random->uniform() > exp((einitial - efinal) * t_inverse)) {
    spin[i] = oldstate;
  }

  if (spin[i] != oldstate) naccept++;
}

/* ----------------------------------------------------------------------
   Kokkos sweep functor for one color - picks from unique neighbors
------------------------------------------------------------------------- */

struct AppPottsNeighKokkos::ColorSweepFunctor {
  Kokkos::View<int*> spin;
  Kokkos::View<int*> numneigh;
  Kokkos::View<int**> neighbor;
  Kokkos::View<int*> color_sites;
  int color_start;
  int maxneigh;
  pool_type rand_pool;
  double temperature;
  double t_inverse;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx, int& accepted) const {
    int i = color_sites(color_start + idx);
    auto rand_gen = rand_pool.get_state();

    int oldstate = spin(i);

    // Build list of unique neighbor spins
    int unique[32];  // Assume maxneigh <= 32 for typical lattices
    int nevent = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int value = spin(neighbor(i, j));
      if (value == oldstate) continue;

      bool found = false;
      for (int m = 0; m < nevent; m++) {
        if (value == unique[m]) {
          found = true;
          break;
        }
      }
      if (!found && nevent < 32) unique[nevent++] = value;
    }

    // Pick random event with null bin
    int iran = static_cast<int>(maxneigh * rand_gen.drand());

    // Null bin rejection
    if (iran >= nevent) {
      rand_pool.free_state(rand_gen);
      return;
    }

    int newstate = unique[iran];

    // Compute initial energy
    int einitial = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i, j)) != oldstate) einitial++;
    }

    // Compute final energy
    int efinal = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i, j)) != newstate) efinal++;
    }

    // Metropolis acceptance
    bool accept = false;
    if (efinal <= einitial) {
      accept = true;
    } else if (temperature > 0.0) {
      double prob = Kokkos::exp((einitial - efinal) * t_inverse);
      if (rand_gen.drand() < prob) accept = true;
    }

    if (accept) {
      spin(i) = newstate;  // Safe - no same-color neighbors!
      accepted++;
    }

    rand_pool.free_state(rand_gen);
  }
};

/* ----------------------------------------------------------------------
   sweep all sites of one color on device
------------------------------------------------------------------------- */

void AppPottsNeighKokkos::sweep_color_kokkos(int color, int start, int count)
{
  ColorSweepFunctor functor{
    d_spin,
    d_numneigh,
    d_neighbor,
    d_color_sites,
    start,
    maxneigh,
    *rand_pool,
    temperature,
    t_inverse
  };

  int naccept_color = 0;
  Kokkos::parallel_reduce("PottsNeighSweep", count, functor, naccept_color);
  naccept += naccept_color;
}
