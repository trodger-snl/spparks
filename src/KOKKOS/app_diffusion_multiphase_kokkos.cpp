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

#include "app_diffusion_multiphase_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"
#include <vector>

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppDiffusionMultiphase>; }

namespace {
template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

struct MPTables {
  Kokkos::View<int*> lat, pidx, pinned;
  Kokkos::View<double*> w;
  Kokkos::View<int*> nn;
  Kokkos::View<int**> nbr;
  int np;

  // AppDiffusionMultiphase::site_energy, summed in the same order
  KOKKOS_INLINE_FUNCTION double energy(int k) const {
    const int lk = lat(k), pk = pidx(lk) * np;
    double e = 0.0;
    for (int j = 0; j < nn(k); j++) {
      const int lj = lat(nbr((size_t) k, j));
      if (lj != lk) e += w(pk + pidx(lj));
    }
    return 0.5 * e;
  }

  // AppDiffusionMultiphase::site_event_rejection: swap with a random
  // neighbour j (out of maxneigh) unless either is pinned or they match,
  // a second draw only if uphill at T > 0. Returns 1 if swapped.
  template <class G>
  KOKKOS_INLINE_FUNCTION int event(int i, G &g, int mn, double temp,
                                   double tinv) const {
    const int li = lat(i);
    if (pinned(pidx(li))) return 0;
    int iran = (int) (mn * g.drand());
    if (iran >= mn) iran = mn - 1;
    const int j = nbr((size_t) i, iran);
    const int lj = lat(j);
    if (pinned(pidx(lj)) || lj == li) return 0;

    const double ei = energy(i) + energy(j);
    lat(i) = lj;
    lat(j) = li;
    const double ed = energy(i) + energy(j) - ei;
    if (ed <= 0.0) return 1;
    if (temp > 0.0 && g.drand() < Kokkos::exp(-1.0 * ed * tinv)) return 1;
    lat(i) = li;
    lat(j) = lj;
    return 0;
  }
};
}

/* ---------------------------------------------------------------------- */

AppDiffusionMultiphaseKokkos::
AppDiffusionMultiphaseKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppDiffusionMultiphase>(spk, narg, arg), nphase(0) {}

/* ---------------------------------------------------------------------- */

void AppDiffusionMultiphaseKokkos::grow_app()
{
  AppDiffusionMultiphase::grow_app();
  LatticeKokkos<AppDiffusionMultiphase>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init checks the sites and fills the default weights; then the
   phase tables go to the device
------------------------------------------------------------------------- */

void AppDiffusionMultiphaseKokkos::init_app()
{
  AppDiffusionMultiphase::init_app();
  LatticeKokkos<AppDiffusionMultiphase>::init_app();

  std::vector<int> labels(phase_labels.begin(), phase_labels.end());
  nphase = labels.size();
  const int maxlabel = nphase ? labels.back() : 0;
  std::vector<int> pidx(maxlabel + 1, 0), pin(nphase);
  std::vector<double> w((size_t) nphase * nphase, 0.0);
  for (int p = 0; p < nphase; p++) {
    pidx[labels[p]] = p;
    pin[p] = is_pinned[labels[p]];
    for (int q = 0; q < nphase; q++)
      if (p != q) w[(size_t) p * nphase + q] = weights[{labels[p], labels[q]}];
  }
  d_pidx = Kokkos::View<int*>("mp_pidx", pidx.size());
  d_pinned = Kokkos::View<int*>("mp_pinned", nphase);
  d_w = Kokkos::View<double*>("mp_w", w.size());
  Kokkos::deep_copy(d_pidx, HostView<int>(pidx.data(), pidx.size()));
  Kokkos::deep_copy(d_pinned, HostView<int>(pin.data(), nphase));
  Kokkos::deep_copy(d_w, HostView<double>(w.data(), w.size()));
}

/* ----------------------------------------------------------------------
   the ghosts go to the host too: a swap changes a neighbour, and the
   reverse comm sends ghost changes to their owners
------------------------------------------------------------------------- */

void AppDiffusionMultiphaseKokkos::allocate_app_views()
{
  d_lattice = Kokkos::View<int*>("lattice", nlocal + nghost);
}

void AppDiffusionMultiphaseKokkos::sync_app_data_to_device()
{
  upload_range(d_lattice, lattice, 0, nlocal + nghost);
}

void AppDiffusionMultiphaseKokkos::sync_app_data_to_host()
{
  download_range(lattice, d_lattice, nlocal + nghost);
}

void AppDiffusionMultiphaseKokkos::sync_ghosts_to_device()
{
  upload_range(d_lattice, lattice, nlocal, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour, from each site's own Park/Miller stream under
   sweep color/strict (as the CPU sweep) or the pool
------------------------------------------------------------------------- */

void AppDiffusionMultiphaseKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  const MPTables t{d_lattice, d_pidx, d_pinned, d_w, d_numneigh, d_neighbor, nphase};
  const int mn = maxneigh, strict = strict_rng;
  const double temp = temperature, tinv = t_inverse;

  int accepted = 0;
  Kokkos::parallel_reduce("MultiphaseSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += t.event(i, rp, mn, temp, tinv);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        acc += t.event(i, g, mn, temp, tinv);
        pool.free_state(g);
      }
    }, accepted);
  naccept += accepted;
}
