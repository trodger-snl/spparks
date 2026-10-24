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
#include "app_potts_quaternion_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park.h"
#include "random_mars.h"
#include "random_park_kokkos.h"
#include "error.h"
#include "memory.h"
#include "crystalline_orientations/crystal_symmetries.h"
#include "crystalline_orientations/quaternion.h"
#include "crystalline_orientations/disorientation.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsQuaternionKokkos::AppPottsQuaternionKokkos(SPPARKS *spk, int narg, char **arg) :
  QuaternionKokkos<AppPottsQuaternion>(spk, narg, arg)
{
  numrandom = 2;   // AppPotts sets the rest of the Potts flags
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::init_app()
{
  // host KMC scratch (AppPotts and AppPottsQuaternion free it)
  if (allow_kmc) {
    delete[] sites;
    delete[] unique_neigh;
    sites = new int[1 + maxneigh];
    unique_neigh = new int[1 + maxneigh];
  }
  QuaternionKokkos<AppPottsQuaternion>::init_app();
}

/* ----------------------------------------------------------------------
   compute energy of site using Read-Shockley equation
------------------------------------------------------------------------- */

double AppPottsQuaternionKokkos::site_energy(int i)
{
  KK_ACC_FLOAT eng = 0.0;
  const double qi[4] = {q0[i], qx[i], qy[i], qz[i]};

  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    // identical orientation -> di = 0 (identity symmetry), contributes 0
    if (q0[nj] == qi[0] && qx[nj] == qi[1] && qy[nj] == qi[2] &&
        qz[nj] == qi[3]) continue;
    const double qj[4] = {q0[nj], qx[nj], qy[nj], qz[nj]};

    // device formula on host too: no per-call vector allocations (KMC calls
    // this nevent+1 times per site), and precision follows KK_FLOAT
    double di = QuaternionOps::compute_disorientation(
      symmetries.data(), num_symmetries, qi, qj);

    eng += QuaternionOps::read_shockley(di, theta_cut);
  }

  return 0.5 * eng;
}

/* ----------------------------------------------------------------------
   rKMC method - pick random neighbor orientation
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::site_event_rejection(int i, RandomPark *random)
{
  int oldstate = spin[i];
  double q_old[4] = {q0[i], qx[i], qy[i], qz[i]};
  double einitial = site_energy(i);

  // Build list of unique neighbor spins
  int nevent = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    int value = spin[nj];
    if (value == oldstate) continue;

    // Check if already in unique list
    bool found = false;
    for (int m = 0; m < nevent; m++) {
      if (value == sites[m]) {
        found = true;
        break;
      }
    }
    if (!found) {
      sites[nevent] = value;
      unique_neigh[nevent] = nj;
      nevent++;
    }
  }

  if (nevent == 0) return;

  // Pick random unique neighbor
  int iran = (int)(nevent * random->uniform());
  if (iran >= nevent) iran = nevent - 1;

  int neighran = unique_neigh[iran];
  int newstate = sites[iran];
  double q_new[4] = {q0[neighran], qx[neighran], qy[neighran], qz[neighran]};

  // Apply new state temporarily
  spin[i] = newstate;
  q0[i] = q_new[0];
  qx[i] = q_new[1];
  qy[i] = q_new[2];
  qz[i] = q_new[3];

  double efinal = site_energy(i);

  // Metropolis acceptance
  if (efinal <= einitial) {
    // accept
  } else if (temperature == 0.0) {
    spin[i] = oldstate;
    q0[i] = q_old[0];
    qx[i] = q_old[1];
    qy[i] = q_old[2];
    qz[i] = q_old[3];
  } else if (random->uniform() > exp((einitial - efinal) * t_inverse)) {
    spin[i] = oldstate;
    q0[i] = q_old[0];
    qx[i] = q_old[1];
    qy[i] = q_old[2];
    qz[i] = q_old[3];
  }

  if (spin[i] != oldstate) naccept++;
}

/* ----------------------------------------------------------------------
   KMC propensity, same scheme as AppPotts::site_propensity
   events = taking the spin and orientation of each distinct unlike
   neighbour spin; each weighs 1 if downhill or level, exp(-dE/kT) if uphill
   unique_neigh holds one neighbour index per distinct spin
------------------------------------------------------------------------- */

int AppPottsQuaternionKokkos::unique_neighbors(int i)
{
  int nevent = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    if (spin[nj] == spin[i]) continue;
    int m;
    for (m = 0; m < nevent; m++)
      if (spin[nj] == spin[unique_neigh[m]]) break;
    if (m < nevent) continue;
    unique_neigh[nevent++] = nj;
  }
  return nevent;
}

void AppPottsQuaternionKokkos::copy_state(int i, int nj)
{
  spin[i] = spin[nj];
  q0[i] = q0[nj];
  qx[i] = qx[nj];
  qy[i] = qy[nj];
  qz[i] = qz[nj];
}

double AppPottsQuaternionKokkos::site_propensity(int i)
{
  int nevent = unique_neighbors(i);
  if (nevent == 0) return 0.0;
  int oldstate = spin[i];
  double q_old[4] = {q0[i], qx[i], qy[i], qz[i]};
  double einitial = site_energy(i);
  double prob = 0.0;

  for (int m = 0; m < nevent; m++) {
    copy_state(i, unique_neigh[m]);
    double efinal = site_energy(i);
    if (efinal <= einitial) prob += 1.0;
    else if (temperature > 0.0) prob += exp((einitial-efinal)*t_inverse);
  }

  spin[i] = oldstate;
  q0[i] = q_old[0];
  qx[i] = q_old[1];
  qy[i] = q_old[2];
  qz[i] = q_old[3];
  return prob;
}

/* ----------------------------------------------------------------------
   KMC event, same scheme as AppPotts::site_event
   pick a flip in proportion to its weight, then update the propensities
   of the site and its neighbours
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::site_event(int i, RandomPark *random)
{
  double threshhold = random->uniform() * propensity[i2site[i]];

  // list is built before any flip, so spin[i] is still the old state
  int nevent = unique_neighbors(i);
  double einitial = site_energy(i);
  double prob = 0.0;

  for (int m = 0; m < nevent; m++) {
    copy_state(i, unique_neigh[m]);
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

struct AppPottsQuaternionKokkos::ColorSweepFunctor {
  Kokkos::View<int*> spin;
  Kokkos::View<double*> q0, qx, qy, qz;
  Kokkos::View<int*> numneigh;
  Kokkos::View<int**> neighbor;
  Kokkos::View<int*> color_sites;
  Kokkos::View<double*> symmetries;
  int color_start;
  int num_symmetries;
  double theta_cut;
  pool_type rand_pool;
  double temperature;
  double t_inverse;
  // `sweep color/strict`: per-site Park/Miller streams instead of the thread-
  // keyed pool, keyed like `spin`. Empty and unused when strict == 0.
  Kokkos::View<int*> siteseeds;
  int strict;

  KOKKOS_INLINE_FUNCTION
  double compute_energy(int i, const double* qi) const {
    KK_ACC_FLOAT eng = 0.0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor((size_t) i, j);
      double qj[4] = {q0(nj), qx(nj), qy(nj), qz(nj)};

      double di = QuaternionOps::compute_disorientation(
        symmetries.data(), num_symmetries, qi, qj);

      eng += QuaternionOps::read_shockley(di, theta_cut);
    }

    return 0.5 * eng;
  }

  // The site update, written once and templated over which RNG supplies the
  // draws (Kokkos pool generator or RandomParkDevice -- both expose drand()).
  // The draw SEQUENCE mirrors AppPottsQuaternion::site_event_rejection exactly:
  // one draw for iran (only once nevent > 0), and a second only when
  // efinal > einitial && temperature > 0.
  template <class Gen>
  KOKKOS_INLINE_FUNCTION
  void update_site(int i, Gen& rand_gen, int& accepted) const {
    int oldstate = spin(i);
    double q_old[4] = {q0(i), qx(i), qy(i), qz(i)};

    double einitial = compute_energy(i, q_old);

    // Build list of unique neighbor spins (stack allocated)
    int unique[64];
    int unique_neigh[64];
    int nevent = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int nj = neighbor((size_t) i, j);
      int value = spin(nj);
      if (value == oldstate) continue;

      bool found = false;
      for (int m = 0; m < nevent; m++) {
        if (value == unique[m]) {
          found = true;
          break;
        }
      }
      if (!found && nevent < 64) {
        unique[nevent] = value;
        unique_neigh[nevent] = nj;
        nevent++;
      }
    }

    // No unique neighbors - nothing to do
    // No unique neighbors - nothing to do. Returns before drawing, as the CPU
    // does, so such a site consumes no numbers from its stream.
    if (nevent == 0) return;

    // Pick random unique neighbor
    int iran = static_cast<int>(nevent * rand_gen.drand());
    if (iran >= nevent) iran = nevent - 1;

    int neighran = unique_neigh[iran];
    int newstate = unique[iran];
    double q_new[4] = {q0(neighran), qx(neighran), qy(neighran), qz(neighran)};
    quaternion::normalize(q_new);

    // Compute new energy
    double efinal = compute_energy(i, q_new);

    // Metropolis acceptance
    bool accept = false;
    if (efinal <= einitial) {
      accept = true;
    } else if (temperature > 0.0) {
      double prob = Kokkos::exp((einitial - efinal) * t_inverse);
      if (rand_gen.drand() < prob) accept = true;
    }

    if (accept) {
      spin(i) = newstate;
      q0(i) = q_new[0];
      qx(i) = q_new[1];
      qy(i) = q_new[2];
      qz(i) = q_new[3];
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

void AppPottsQuaternionKokkos::sweep_color_kokkos(int color, int start, int count)
{
  ColorSweepFunctor functor{
    d_spin,
    d_q0, d_qx, d_qy, d_qz,
    d_numneigh,
    d_neighbor,
    d_color_sites,
    d_symmetries,
    start,
    num_symmetries,
    theta_cut,
    *rand_pool,
    temperature,
    t_inverse,
    d_siteseeds,
    strict_rng
  };

  int naccept_color = 0;
  Kokkos::parallel_reduce("QuaternionSweep", count, functor, naccept_color);
  naccept += naccept_color;
}

namespace SPPARKS_NS {
template class LatticeKokkos<AppPottsQuaternion>;
}
