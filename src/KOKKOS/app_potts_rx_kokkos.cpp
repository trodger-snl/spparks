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

#include "app_potts_rx_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"
#include "comm_lattice.h"

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPottsRX>; }

namespace {
// a Kokkos pool generator behind the uniform() rx::event draws with
template <class G> struct PoolUniform {
  G &g;
  KOKKOS_INLINE_FUNCTION double uniform() { return g.drand(); }
};
}

/* ---------------------------------------------------------------------- */

AppPottsRXKokkos::AppPottsRXKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPottsRX>(spk, narg, arg)
{
  device_app_update = true;   // app_update nucleates on the device views
}

/* ---------------------------------------------------------------------- */

void AppPottsRXKokkos::grow_app()
{
  AppPottsRX::grow_app();
  LatticeKokkos<AppPottsRX>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init only checks the sites
------------------------------------------------------------------------- */

void AppPottsRXKokkos::init_app()
{
  AppPottsRX::init_app();
  LatticeKokkos<AppPottsRX>::init_app();
  upload_range(d_id, id, 0, nlocal + nghost);
}

/* ---------------------------------------------------------------------- */

void AppPottsRXKokkos::allocate_app_views()
{
  const int nall = nlocal + nghost;
  d_spin = Kokkos::View<int*>("spin", nall);
  d_h = Kokkos::View<double*>("stored_energy", nall);
  d_w = Kokkos::View<double*>("boltz_stored", nall);
  d_id = Kokkos::View<tagint*>("id", nall);
  d_flag = Kokkos::View<char*>("nucleus_seed", nall);
}

void AppPottsRXKokkos::sync_app_data_to_device()
{
  upload_range(d_spin, spin, 0, nlocal + nghost);
  upload_range(d_h, h, 0, nlocal + nghost);
  upload_range(d_w, w, 0, nlocal + nghost);
}

void AppPottsRXKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal);
  download_range(h, d_h, nlocal);
  download_range(w, d_w, nlocal);
}

void AppPottsRXKokkos::sync_ghosts_to_device()
{
  upload_range(d_spin, spin, nlocal, nlocal + nghost);
  upload_range(d_h, h, nlocal, nlocal + nghost);
  upload_range(d_w, w, nlocal, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour, from each site's own Park/Miller stream under
   sweep color/strict (as the CPU sweep, so the two match) or the pool
------------------------------------------------------------------------- */

void AppPottsRXKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto hh = d_h;
  auto ww = d_w;
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  const rx::Params p = par;
  const int strict = strict_rng;

  int accepted = 0;
  Kokkos::parallel_reduce("PottsRXSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += rx::event(i, rp, nn, nbr, sp, hh, ww, p);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        PoolUniform<decltype(g)> u{g};
        acc += rx::event(i, u, nn, nbr, sp, hh, ww, p);
        pool.free_state(g);
      }
    }, accepted);
  naccept += accepted;
}

/* ----------------------------------------------------------------------
   AppPottsRX::app_update on the device: the ghosts made current on the
   host, then the seeds of every site, then the nuclei of the owned ones
------------------------------------------------------------------------- */

void AppPottsRXKokkos::app_update(double dt)
{
  if (nrate == 0.0) return;
  if (nprocs > 1) {
    ensure_host_data_current();
    comm->all();
    sync_ghosts_to_device();
  }

  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto hh = d_h;
  auto ww = d_w;
  auto ids = d_id;
  auto flag = d_flag;
  const int ns = nspins;
  const double hm = hmin, prob = nrate * dt;
  const unsigned long long s = salt();

  Kokkos::parallel_for("PottsRXSeeds", nlocal + nghost,
    KOKKOS_LAMBDA(const int c) {
      flag(c) = rx::seeds(ids(c), sp(c), hh(c), ns, hm, prob, s);
    });
  Kokkos::parallel_for("PottsRXNuclei", nlocal, KOKKOS_LAMBDA(const int i) {
    const tagint g = rx::nucleus(i, nn, nbr, sp, flag, ids, ns);
    if (g) {
      sp(i) = ns + (int) g;
      hh(i) = 0.0;
      ww(i) = 1.0;
    }
  });
}
