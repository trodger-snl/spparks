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
#include "string.h"
#include "stdlib.h"
#include "app_potts_quaternion_kokkos.h"
#include "random_park.h"
#include "random_mars.h"
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
  } else {
    error->all(FLERR, "Illegal 'potts/quaternion/kk' command; expected 'cubic' or 'hcp'");
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
  d_spin = Kokkos::View<int*>("spin", nall);
  d_q0 = Kokkos::View<double*>("q0", nall);
  d_qx = Kokkos::View<double*>("qx", nall);
  d_qy = Kokkos::View<double*>("qy", nall);
  d_qz = Kokkos::View<double*>("qz", nall);
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
    if (spin[i] < 1 || spin[i] > nspins) flag = 1;
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
  RandomPark ranquat(ranmaster->initial_seed());
  const double qseed = ranquat.uniform();

  for (int i = 0; i < nlocal; i++) {
    double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
    if (norm < 1.0e-6) {
      ranquat.tagreset(qseed, id[i], 100);
      std::vector<double> quat = quaternion::generate_unit_quaternion(ranquat);
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

  // Initial sync of data to device
  sync_app_data_to_device();
}

/* ----------------------------------------------------------------------
   sync spin and quaternion data to device
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   (Re)allocate the persistent host mirrors used by both sync paths.
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::ensure_sync_mirrors()
{
  const int nall = nlocal + nghost;
  if (sync_mirror_n == nall) return;
  h_q0_m = Kokkos::create_mirror_view(d_q0);
  h_qx_m = Kokkos::create_mirror_view(d_qx);
  h_qy_m = Kokkos::create_mirror_view(d_qy);
  h_qz_m = Kokkos::create_mirror_view(d_qz);
  sync_mirror_n = nall;
}

void AppPottsQuaternionKokkos::sync_app_data_to_device()
{
  int nall = nlocal + nghost;
  ensure_sync_mirrors();

  // Sync spin: a straight copy with no arithmetic, so move it as one contiguous
  // block out of the raw host array rather than element by element.
  Kokkos::deep_copy(
      d_spin,
      Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(spin, nall));

  // Sync quaternions with normalization.
  //
  // Threaded, NOT vectorised into a bulk copy: the normalisation is real
  // arithmetic, and its low bits are load-bearing. Renormalising an already-unit
  // quaternion still perturbs the last ulp, and a past bug had two sync paths
  // disagreeing there, which made the trajectory depend on rank count. Each
  // iteration reads only element i and writes only element i, with no reduction
  // and no reassociation, so threading is bit-identical to the serial loop --
  // every site's four values are combined in exactly the same order as before.
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nall; i++) {
    double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      h_q0_m(i) = q0[i] * inv_norm;
      h_qx_m(i) = qx[i] * inv_norm;
      h_qy_m(i) = qy[i] * inv_norm;
      h_qz_m(i) = qz[i] * inv_norm;
    } else {
      h_q0_m(i) = q0[i];
      h_qx_m(i) = qx[i];
      h_qy_m(i) = qy[i];
      h_qz_m(i) = qz[i];
    }
  }

  Kokkos::deep_copy(d_q0, h_q0_m);
  Kokkos::deep_copy(d_qx, h_qx_m);
  Kokkos::deep_copy(d_qy, h_qy_m);
  Kokkos::deep_copy(d_qz, h_qz_m);
}

/* ----------------------------------------------------------------------
   sync data from device to host
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::sync_app_data_to_host()
{
  const int nall = nlocal + nghost;
  ensure_sync_mirrors();

  // Sync spin: straight copy of the OWNED slice, in one block. The range
  // matches the previous element loop exactly (nlocal, not nall -- host ghost
  // entries come from the ghost exchange, not from here).
  const auto owned = Kokkos::make_pair(0, nlocal);
  Kokkos::deep_copy(
      Kokkos::subview(
          Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(spin, nall), owned),
      Kokkos::subview(d_spin, owned));

  // Sync quaternions with normalization
  Kokkos::deep_copy(h_q0_m, d_q0);
  Kokkos::deep_copy(h_qx_m, d_qx);
  Kokkos::deep_copy(h_qy_m, d_qy);
  Kokkos::deep_copy(h_qz_m, d_qz);

  // Threaded, but otherwise byte-for-byte the previous loop -- see the note in
  // sync_app_data_to_device on why the normalisation arithmetic is left exactly
  // as it was. Per-element independent, no reduction, so threading cannot
  // change a single result bit.
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nlocal; i++) {
    double norm = sqrt(h_q0_m(i)*h_q0_m(i) + h_qx_m(i)*h_qx_m(i) +
                       h_qy_m(i)*h_qy_m(i) + h_qz_m(i)*h_qz_m(i));
    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      q0[i] = h_q0_m(i) * inv_norm;
      qx[i] = h_qx_m(i) * inv_norm;
      qy[i] = h_qy_m(i) * inv_norm;
      qz[i] = h_qz_m(i) * inv_norm;
    } else {
      q0[i] = h_q0_m(i);
      qx[i] = h_qx_m(i);
      qy[i] = h_qy_m(i);
      qz[i] = h_qz_m(i);
    }
  }
}

/* ----------------------------------------------------------------------
   normalize quaternion arrays on host
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::normalize_quaternion_arrays(
    double* q0_out, double* qx_out, double* qy_out, double* qz_out,
    const double* q0_in, const double* qx_in,
    const double* qy_in, const double* qz_in, int n)
{
  for (int i = 0; i < n; i++) {
    double norm = sqrt(q0_in[i]*q0_in[i] + qx_in[i]*qx_in[i] +
                       qy_in[i]*qy_in[i] + qz_in[i]*qz_in[i]);

    if (norm > 1.0e-15) {
      double inv_norm = 1.0 / norm;
      q0_out[i] = q0_in[i] * inv_norm;
      qx_out[i] = qx_in[i] * inv_norm;
      qy_out[i] = qy_in[i] * inv_norm;
      qz_out[i] = qz_in[i] * inv_norm;
    } else {
      q0_out[i] = q0_in[i];
      qx_out[i] = qx_in[i];
      qy_out[i] = qy_in[i];
      qz_out[i] = qz_in[i];
    }
  }
}

/* ----------------------------------------------------------------------
   compute energy of site using Read-Shockley equation
------------------------------------------------------------------------- */

double AppPottsQuaternionKokkos::site_energy(int i)
{
  double eng = 0.0;
  std::vector<double> qi = {q0[i], qx[i], qy[i], qz[i]};

  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    std::vector<double> qj = {q0[nj], qx[nj], qy[nj], qz[nj]};

    // Compute disorientation using host-side quaternion library
    double di = disorientation::compute_disorientation(symmetries, qi, qj);

    double ratio = di / theta_cut;

    if (ratio <= 0.0) {
      continue;
    } else if (ratio < 1.0) {
      eng += ratio * (1.0 - log(ratio));
    } else {
      eng += 1.0;
    }
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
   KMC propensity
------------------------------------------------------------------------- */

double AppPottsQuaternionKokkos::site_propensity(int i)
{
  // Count unique neighbor spins different from mine
  int isite = spin[i];
  int nunique = 0;

  for (int j = 0; j < numneigh[i]; j++) {
    int value = spin[neighbor[i][j]];
    if (value == isite) continue;

    bool found = false;
    for (int m = 0; m < nunique; m++) {
      if (value == sites[m]) {
        found = true;
        break;
      }
    }
    if (!found) sites[nunique++] = value;
  }

  return (double) nunique;
}

/* ----------------------------------------------------------------------
   KMC event - pick random neighbor orientation
------------------------------------------------------------------------- */

void AppPottsQuaternionKokkos::site_event(int i, RandomPark *random)
{
  int oldstate = spin[i];

  // Build list of unique neighbor spins
  int nevent = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    int value = spin[nj];
    if (value == oldstate) continue;

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
  int m = (int)(nevent * random->uniform());
  if (m >= nevent) m = nevent - 1;

  int neighran = unique_neigh[m];
  spin[i] = sites[m];
  q0[i] = q0[neighran];
  qx[i] = qx[neighran];
  qy[i] = qy[neighran];
  qz[i] = qz[neighran];
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

  KOKKOS_INLINE_FUNCTION
  double compute_energy(int i, const double* qi) const {
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
    int i = color_sites(color_start + idx);
    auto rand_gen = rand_pool.get_state();

    int oldstate = spin(i);
    double q_old[4] = {q0(i), qx(i), qy(i), qz(i)};

    double einitial = compute_energy(i, q_old);

    // Build list of unique neighbor spins (stack allocated)
    int unique[64];
    int unique_neigh[64];
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
      if (!found && nevent < 64) {
        unique[nevent] = value;
        unique_neigh[nevent] = nj;
        nevent++;
      }
    }

    // No unique neighbors - nothing to do
    if (nevent == 0) {
      rand_pool.free_state(rand_gen);
      return;
    }

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

    rand_pool.free_state(rand_gen);
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
    k_numneigh.view_device(),
    k_neighbor.view_device(),
    d_color_sites,
    d_symmetries,
    start,
    num_symmetries,
    theta_cut,
    *rand_pool,
    temperature,
    t_inverse
  };

  int naccept_color = 0;
  Kokkos::parallel_reduce("QuaternionSweep", count, functor, naccept_color);
  naccept += naccept_color;
}
