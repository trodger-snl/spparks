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

#include "app_sinter_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"
#include "comm_lattice.h"
#include "error.h"
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <climits>

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppSinter>; }

namespace {
// a Kokkos pool generator behind the uniform() sinter::event draws with
template <class G> struct PoolUniform {
  G &g;
  KOKKOS_INLINE_FUNCTION double uniform() { return g.drand(); }
};

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
}

/* ---------------------------------------------------------------------- */

AppSinterKokkos::AppSinterKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppSinter>(spk, narg, arg)
{
  device_app_update = true;   // app_update's hooks use the device views
}

/* ---------------------------------------------------------------------- */

void AppSinterKokkos::grow_app()
{
  AppSinter::grow_app();
  LatticeKokkos<AppSinter>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init (checks, cell map) only reads the lattice: safe to run
------------------------------------------------------------------------- */

void AppSinterKokkos::init_app()
{
  AppSinter::init_app();
  LatticeKokkos<AppSinter>::init_app();
  upload_range(d_id, id, 0, nlocal + nghost);
}

/* ---------------------------------------------------------------------- */

void AppSinterKokkos::allocate_app_views()
{
  const int nall = nlocal + nghost;
  d_spin = Kokkos::View<int*>("spin", nall);
  d_id = Kokkos::View<tagint*>("id", nall);
  d_nmade = Kokkos::View<int>("nmade");
}

/* ----------------------------------------------------------------------
   room for a vacancy per site and pass of a sweep (setup sets nloop)
------------------------------------------------------------------------- */

void AppSinterKokkos::setup_end_app()
{
  const int nloop = set[bothflag ? nsector : 0].nloop;
  const size_t n = (size_t) nlocal * std::max(1, nloop) + 1;
  if (d_made.extent(0) < n) d_made = Kokkos::View<int*>("made", n);
}

/* ----------------------------------------------------------------------
   the ghosts go to the host too: a sweep changes them and the reverse
   comm sends them to their owners
------------------------------------------------------------------------- */

void AppSinterKokkos::sync_app_data_to_device()
{
  upload_range(d_spin, spin, 0, nlocal + nghost);
}

void AppSinterKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal + nghost);
}

void AppSinterKokkos::sync_ghosts_to_device()
{
  upload_range(d_spin, spin, nlocal, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour: sinter::event per site, drawing from the
   site's own Park/Miller stream under sweep color/strict (as the CPU
   sweep does, so the two match) or from the Kokkos pool
------------------------------------------------------------------------- */

void AppSinterKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  auto made = d_made;
  auto nm = d_nmade;
  const int cap = made.extent(0), strict = strict_rng;
  const sinter::Params p = par;

  int accepted = 0;
  Kokkos::parallel_reduce("SinterSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      int vac;
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += sinter::event(i, rp, nn, nbr, sp, p, vac);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        PoolUniform<decltype(g)> u{g};
        acc += sinter::event(i, u, nn, nbr, sp, p, vac);
        pool.free_state(g);
      }
      if (vac >= 0) {
        const int k = Kokkos::atomic_fetch_add(&nm(), 1);
        if (k < cap) made(k) = vac;
      }
    }, accepted);
  naccept += accepted;
}

/* ---------------------------------------------------------------------- */

int AppSinterKokkos::nmade()
{
  int n;
  Kokkos::deep_copy(n, d_nmade);
  if (n > (int) d_made.extent(0))
    error->one(FLERR, "sinter/kk: vacancy list overflow");
  return n;
}

/* ----------------------------------------------------------------------
   ghosts current on the device and the host copy current (gb_average,
   diags read it)
------------------------------------------------------------------------- */

void AppSinterKokkos::update_ghosts()
{
  ensure_host_data_current();
  if (nprocs > 1) {
    comm->all();
    sync_ghosts_to_device();
  }
}

/* ---------------------------------------------------------------------- */

void AppSinterKokkos::candidates(std::vector<Vac> &v)
{
  const int n = nmade();
  Kokkos::View<tagint*> gid("vac_gid", n);
  Kokkos::View<int*> grain("vac_grain", n);
  auto made = d_made;
  auto ids = d_id;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  const unsigned long long s = salt();
  Kokkos::parallel_for("SinterVacancies", n, KOKKOS_LAMBDA(const int k) {
    const int j = made(k);
    gid(k) = ids(j);
    grain(k) = sinter::adjacent_grain(j, sinter::mix(s + ids(j)) % nn(j),
                                      nn, nbr, sp);
  });
  auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gid);
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grain);
  for (int k = 0; k < n; k++)
    if (hs(k) > 0) v.push_back({hg(k), hs(k)});
  Kokkos::deep_copy(d_nmade, 0);
}

/* ---------------------------------------------------------------------- */

void AppSinterKokkos::grain_sums(const std::vector<int> &grains,
                                 std::vector<long long> &sums)
{
  const int ng = grains.size();
  Kokkos::View<int*> g("grains", ng);
  Kokkos::deep_copy(g, HostView<const int>(grains.data(), ng));
  Kokkos::View<long long*> sum("grain_sums", 4 * ng);
  auto sp = d_spin;
  auto ids = d_id;
  const int nb = nbasis, mx = nx, my = ny;
  Kokkos::parallel_for("SinterGrainSums", nlocal, KOKKOS_LAMBDA(const int i) {
    const int s = sp(i);
    if (s <= sinter::VACANT) return;
    int lo = 0, hi = ng;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (g(mid) < s) lo = mid + 1;
      else hi = mid;
    }
    if (lo == ng || g(lo) != s) return;
    int c[3];
    sinter::cell(ids(i), nb, mx, my, c);
    Kokkos::atomic_add(&sum(4 * lo), 1LL);
    for (int d = 0; d < 3; d++)
      Kokkos::atomic_add(&sum(4 * lo + 1 + d), (long long) c[d]);
  });
  Kokkos::deep_copy(HostView<long long>(sums.data(), sums.size()), sum);
}

/* ---------------------------------------------------------------------- */

void AppSinterKokkos::get_sites(const std::vector<int> &idx,
                                std::vector<int> &val)
{
  const int n = idx.size();
  Kokkos::View<int*> di("site_idx", n), dv("site_val", n);
  Kokkos::deep_copy(di, HostView<const int>(idx.data(), n));
  auto sp = d_spin;
  Kokkos::parallel_for("SinterGetSites", n, KOKKOS_LAMBDA(const int k) {
    dv(k) = di(k) >= 0 ? sp(di(k)) : INT_MIN;
  });
  Kokkos::deep_copy(HostView<int>(val.data(), n), dv);
}

void AppSinterKokkos::set_sites(const std::vector<int> &idx,
                                const std::vector<int> &val)
{
  const int n = idx.size();
  Kokkos::View<int*> di("site_idx", n), dv("site_val", n);
  Kokkos::deep_copy(di, HostView<const int>(idx.data(), n));
  Kokkos::deep_copy(dv, HostView<const int>(val.data(), n));
  auto sp = d_spin;
  Kokkos::parallel_for("SinterSetSites", n, KOKKOS_LAMBDA(const int k) {
    if (di(k) >= 0) sp(di(k)) = dv(k);
  });
  host_data_stale = true;
}

/* ----------------------------------------------------------------------
   AppSinter::gb_faces on the device: each grain site's faces go to its
   slot of one array, which is sorted and made unique there
------------------------------------------------------------------------- */

void AppSinterKokkos::gb_faces(long long &fsites,
                               std::vector<unsigned long long> &faces)
{
  using ull = unsigned long long;
  const int nloc = nlocal;
  auto sp = d_spin;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;

  Kokkos::View<long long*> off("face_offset", nloc + 1);
  long long total = 0;
  Kokkos::parallel_scan("SinterFaceCount", nloc,
    KOKKOS_LAMBDA(const int i, long long &upd, const bool final) {
      int u[sinter::MAXNEIGH];
      const int nu = sp(i) > sinter::VACANT ?
        sinter::other_grains(i, nn, nbr, sp, u) : 0;
      if (final) off(i) = upd;
      upd += nu;
    }, total);
  fsites += total;

  Kokkos::View<ull*> key("faces", total);
  Kokkos::parallel_for("SinterFaces", nloc, KOKKOS_LAMBDA(const int i) {
    const int s = sp(i);
    if (s <= sinter::VACANT) return;
    int u[sinter::MAXNEIGH];
    const int nu = sinter::other_grains(i, nn, nbr, sp, u);
    for (int m = 0; m < nu; m++)
      key(off(i) + m) = (ull) s << 32 | (unsigned) u[m];
  });

  long long nuniq = 0;
  Kokkos::View<ull*> uniq("faces_unique", total);
  if (total) {
    Kokkos::sort(key);
    Kokkos::parallel_scan("SinterFacesUnique", total,
      KOKKOS_LAMBDA(const long long k, long long &upd, const bool final) {
        const bool first = k == 0 || key(k) != key(k - 1);
        if (final && first) uniq(upd) = key(k);
        upd += first;
      }, nuniq);
  }
  faces.resize(nuniq);
  Kokkos::deep_copy(HostView<ull>(faces.data(), nuniq),
                    Kokkos::subview(uniq, Kokkos::make_pair(0LL, nuniq)));
}
