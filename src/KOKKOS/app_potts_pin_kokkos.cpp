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
#include "app_potts_pin_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPottsPin>; }

namespace {
// AppPottsPin::site_event_rejection: no event on a pinned site; a random
// spin 1..nspins, a second draw only if uphill at T > 0
template <class G, class NN, class N, class S, class B>
KOKKOS_INLINE_FUNCTION int pin_event(int i, G &g, const NN &nn, const N &nbr,
                                     const S &spin, int nspins, double temp,
                                     const B &boltz)
{
  const int old = spin(i);
  if (old > nspins) return 0;
  int ei = 0;
  for (int j = 0; j < nn(i); j++)
    if (spin(nbr((size_t) i, j)) != old) ei++;
  int iran = (int) (nspins * g.drand()) + 1;
  if (iran > nspins) iran = nspins;
  int ef = 0;
  for (int j = 0; j < nn(i); j++)
    if (spin(nbr((size_t) i, j)) != iran) ef++;
  if (ef > ei) {
    if (temp == 0.0) return 0;
    if (g.drand() > boltz.b[ef - ei]) return 0;
  }
  spin(i) = iran;
  return iran != old;
}
}

/* ---------------------------------------------------------------------- */

AppPottsPinKokkos::AppPottsPinKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPottsPin>(spk, narg, arg)
{
  allow_masking = 0;   // the device sweep has no mask
}

/* ---------------------------------------------------------------------- */

void AppPottsPinKokkos::grow_app()
{
  AppPottsPin::grow_app();
  LatticeKokkos<AppPottsPin>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init only checks the spins
------------------------------------------------------------------------- */

void AppPottsPinKokkos::init_app()
{
  if (maxneigh > 32) error->all(FLERR,"potts/pin/kk supports at most 32 neighbors");
  AppPottsPin::init_app();
  LatticeKokkos<AppPottsPin>::init_app();
}

/* ----------------------------------------------------------------------
   Boltzmann factors for this run's temperature, made on the host with the
   CPU app's expression, so the device draws compare against the same values
------------------------------------------------------------------------- */

void AppPottsPinKokkos::setup_end_app()
{
  for (int d = 0; d <= 32; d++)
    boltz.b[d] = temperature > 0.0 ? exp(-(double) d * t_inverse) : 0.0;
}

/* ---------------------------------------------------------------------- */

void AppPottsPinKokkos::allocate_app_views()
{
  d_spin = Kokkos::View<int*>("spin", nlocal + nghost);
}

void AppPottsPinKokkos::sync_app_data_to_device()
{
  upload_range(d_spin, spin, 0, nlocal + nghost);
}

void AppPottsPinKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal);
}

void AppPottsPinKokkos::sync_ghosts_to_device()
{
  upload_range(d_spin, spin, nlocal, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour, from each site's own Park/Miller stream under
   sweep color/strict (as the CPU sweep, so the two match) or the pool
------------------------------------------------------------------------- */

void AppPottsPinKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  const Boltz b = boltz;
  const int ns = nspins, strict = strict_rng;
  const double temp = temperature;

  int accepted = 0;
  Kokkos::parallel_reduce("PottsPinSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += pin_event(i, rp, nn, nbr, sp, ns, temp, b);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        acc += pin_event(i, g, nn, nbr, sp, ns, temp, b);
        pool.free_state(g);
      }
    }, accepted);
  naccept += accepted;
}
