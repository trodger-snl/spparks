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
#include "random_park.h"
#include "random_mars.h"
#include "random_park_kokkos.h"
#include "error.h"
#include "memory.h"
#include "crystalline_orientations/cubic_symmetries.h"
#include "crystalline_orientations/hcp_symmetries.h"
#include "crystalline_orientations/quaternion.h"
#include "crystalline_orientations/disorientation.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsQuaternionKokkos::AppPottsQuaternionKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLatticeKokkos(spk, narg, arg), num_symmetries(0), theta_cut(15.0)
{
  // Quaternion Potts setup
  ninteger = 1;
  ndouble = 4;
  delpropensity = 1;
  delevent = 0;
  allow_kmc = 1;
  allow_rejection = 1;
  allow_masking = 1;
  numrandom = 2;

  create_arrays();

  // Parse arguments
  if (narg != 3 && narg != 4)
    error->all(FLERR, "Illegal 'potts/quaternion/kk' command");

  nspins = atoi(arg[1]);
  if (nspins <= 0)
    error->all(FLERR, "Illegal app_style command");

  // Parse crystal structure
  if (strcmp(arg[2], "cubic") == 0) {
    symmetries = CUBIC::get_symmetries();
  } else if (strcmp(arg[2], "hcp") == 0) {
    symmetries = HCP::get_symmetries();
  } else if (strcmp(arg[2], "trigonal") == 0) {
    symmetries = TRIGONAL::get_symmetries();
  } else {
    error->all(FLERR, "Illegal 'potts/quaternion/kk' command; expected 'cubic', 'hcp' or 'trigonal'");
  }

  // Parse optional theta_cut
  if (narg == 4) {
    theta_cut = atof(arg[3]);
    if (theta_cut <= 0.0)
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; theta_cut must be positive");
    if (strcmp(arg[2], "cubic") == 0 && theta_cut > 62.7) {
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; "
                        "theta_cut for 'cubic' must be between 0.0 and 62.7 degrees");
    } else if (strcmp(arg[2], "hcp") == 0 && theta_cut > 93.8) {
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; "
                        "theta_cut for 'hcp' must be between 0.0 and 93.8 degrees");
    } else if (strcmp(arg[2], "trigonal") == 0 && theta_cut > 104.4) {
      error->all(FLERR, "Illegal 'potts/quaternion/kk' command; "
                        "theta_cut for 'trigonal' must be between 0.0 and 104.4 degrees");
    }
  }

  // dt_sweep set in init_app() when maxneigh is known

  // Initialize pointers
  spin = nullptr;
  q0 = qx = qy = qz = nullptr;
  sites = nullptr;
  unique_neigh = nullptr;

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
  memory->destroy(sites);
  memory->destroy(unique_neigh);
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::grow_app()
{
  // Set spin pointer to iarray[0]
  spin = iarray[0];

  // Set quaternion pointers to darray
  q0 = darray[0];
  qx = darray[1];
  qy = darray[2];
  qz = darray[3];

  // Call parent (allocates neighbors and calls allocate_app_views)
  AppLatticeKokkos::grow_app();
}

/* ----------------------------------------------------------------------
   allocate app-specific Kokkos views
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::allocate_app_views()
{
  int nall = nlocal + nghost;
  d_spin = site_view("spin", spin, nall);
  d_q0 = site_view("q0", q0, nall);
  d_qx = site_view("qx", qx, nall);
  d_qy = site_view("qy", qy, nall);
  d_qz = site_view("qz", qz, nall);
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::init_app()
{
  // Set dt_sweep now that maxneigh is known (matches CPU version)
  dt_sweep = 1.0/maxneigh;

  // Validate spin values
  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (spin[i] < spin_lo || spin[i] > nspins) flag = 1;
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");

  // Initialize random quaternions for sites that don't have one, keyed to each
  // site's GLOBAL id. Same idiom as the color/strict per-site seeds in
  // AppLattice::init(): the stream depends only on the deck's `seed` and the
  // site id, so the field is repeatable run to run AND identical at any rank
  // count -- which is what makes an np=1 vs np=N regression test possible.
  // Derived from ranmaster's stored seed rather than ranmaster->uniform() so
  // that no other stream shifts.
  const uint64_t qseed = (uint64_t) ranmaster->initial_seed();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nlocal; i++) {
    double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
    if (norm < 1.0e-6) {
      quaternion::SiteOrientationRNG rng(qseed, (uint64_t) id[i]);
      std::vector<double> quat = quaternion::generate_unit_quaternion(rng);
      q0[i] = quat[0];
      qx[i] = quat[1];
      qy[i] = quat[2];
      qz[i] = quat[3];
    }
  }

  // Allocate temp arrays for KMC if needed
  if (allow_kmc) {
    memory->destroy(sites);
    memory->destroy(unique_neigh);
    memory->create(sites, 1 + maxneigh, "app:sites");
    memory->create(unique_neigh, 1 + maxneigh, "app:unique_neigh");
  }

  // Call parent (validates coloring, sets up color lists, syncs neighbors)
  AppLatticeKokkos::init_app();

  // Normalise once, in place on the host, then upload an exact copy: the
  // host and device hold the same bits from here on.
  normalize_host(nlocal + nghost, false);
  sync_app_data_to_device();
}

/* ----------------------------------------------------------------------
   normalise host quaternions in place. off_unit_only: leave any quaternion
   within 1e-10 of unit length untouched.

   Renormalising is not idempotent in the last bit -- a value can flip between
   two neighbours on alternate passes -- so values that came from the device
   must never be renormalised, or the trajectory depends on how often the data
   happened to cross the bus (rank count, number of runs). init_app does the
   one full pass; later uploads only fix clearly non-unit input (a `set`
   between runs). Per-element independent, so threading is bit-identical.
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::normalize_host(int n, bool off_unit_only)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
    if (norm > 1.0e-15 && (!off_unit_only || fabs(norm - 1.0) > 1.0e-10)) {
      double inv_norm = 1.0 / norm;
      q0[i] *= inv_norm;
      qx[i] *= inv_norm;
      qy[i] *= inv_norm;
      qz[i] *= inv_norm;
    }
  }
}

/* ----------------------------------------------------------------------
   sync spin and quaternion data to device: an exact copy, apart from
   normalising clearly non-unit host input (see normalize_host)
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_app_data_to_device()
{
  const int nall = nlocal + nghost;
  normalize_host(nall, true);
  upload_range(d_spin, spin, 0, nall);
  upload_range(d_q0, q0, 0, nall);
  upload_range(d_qx, qx, 0, nall);
  upload_range(d_qy, qy, 0, nall);
  upload_range(d_qz, qz, 0, nall);
}

/* ----------------------------------------------------------------------
   sync owned spin and quaternion data to host: an exact copy. Host ghost
   entries come from the ghost exchange, not from here.
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal);
  download_range(q0, d_q0, nlocal);
  download_range(qx, d_qx, nlocal);
  download_range(qy, d_qy, nlocal);
  download_range(qz, d_qz, nlocal);
}

/* ----------------------------------------------------------------------
   upload the ghost range after a forward exchange. A pure copy: ghosts are
   another rank's owned values, which are already unit (the run-start upload
   normalises, and sweeps only copy whole quaternions between sites).
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_ghosts_to_device()
{
  const int first = nlocal, last = nlocal + nghost;
  upload_range(d_spin, spin, first, last);
  upload_range(d_q0, q0, first, last);
  upload_range(d_qx, qx, first, last);
  upload_range(d_qy, qy, first, last);
  upload_range(d_qz, qz, first, last);
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
    QuaternionOps::normalize(q_new);

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
