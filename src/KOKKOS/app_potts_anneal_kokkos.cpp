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

#include "app_potts_anneal_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"
#include "comm_lattice.h"
#include <Kokkos_Sort.hpp>
#include <cstdint>

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPottsAnneal>; }

namespace {
// a Kokkos pool generator behind the uniform() anneal::event draws with
template <class G> struct PoolUniform {
  G &g;
  KOKKOS_INLINE_FUNCTION double uniform() { return g.drand(); }
};

template <class T>
using HostView = Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
}

/* ---------------------------------------------------------------------- */

AppPottsAnnealKokkos::AppPottsAnnealKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPottsAnneal>(spk, narg, arg)
{
  device_app_update = true;   // app_update runs on the device views
}

/* ---------------------------------------------------------------------- */

void AppPottsAnnealKokkos::grow_app()
{
  AppPottsAnneal::grow_app();
  LatticeKokkos<AppPottsAnneal>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init checks and orients the sites; the tables go to the device
------------------------------------------------------------------------- */

void AppPottsAnnealKokkos::init_app()
{
  AppPottsAnneal::init_app();
  LatticeKokkos<AppPottsAnneal>::init_app();
  upload_range(d_id, id, 0, nlocal + nghost);
  d_sym = DView("symmetries", symmetries.size());
  Kokkos::deep_copy(d_sym, HostView<double>(symmetries.data(), symmetries.size()));
  std::vector<KK_FLOAT> symk(symmetries.begin(), symmetries.end());
  d_symk = Kokkos::View<KK_FLOAT*>("symmetries_kk", symk.size());
  Kokkos::deep_copy(d_symk, HostView<KK_FLOAT>(symk.data(), symk.size()));
  d_orient = DView("nucleus_orientations", orient.size());
  Kokkos::deep_copy(d_orient, HostView<double>(orient.data(), orient.size()));
}

/* ---------------------------------------------------------------------- */

void AppPottsAnnealKokkos::allocate_app_views()
{
  const int nall = nlocal + nghost;
  d_spin = IView("spin", nall);
  d_phase = IView("phase", nall);
  d_seed = IView("nucleus_seed", nall);
  d_next = IView("nucleus_next", nall);
  d_q0 = DView("q0", nall);
  d_qx = DView("qx", nall);
  d_qy = DView("qy", nall);
  d_qz = DView("qz", nall);
  d_h = DView("stored_energy", nall);
  d_id = Kokkos::View<tagint*>("id", nall);
}

void AppPottsAnnealKokkos::sync_app_data_to_device()
{
  const int nall = nlocal + nghost;
  upload_range(d_spin, spin, 0, nall);
  upload_range(d_phase, phase, 0, nall);
  upload_range(d_q0, q0, 0, nall);
  upload_range(d_qx, qx, 0, nall);
  upload_range(d_qy, qy, 0, nall);
  upload_range(d_qz, qz, 0, nall);
  upload_range(d_h, h, 0, nall);
}

void AppPottsAnnealKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal);
  download_range(q0, d_q0, nlocal);
  download_range(qx, d_qx, nlocal);
  download_range(qy, d_qy, nlocal);
  download_range(qz, d_qz, nlocal);
  download_range(h, d_h, nlocal);
}

void AppPottsAnnealKokkos::sync_ghosts_to_device()
{
  const int first = nlocal, last = nlocal + nghost;
  upload_range(d_spin, spin, first, last);
  upload_range(d_phase, phase, first, last);
  upload_range(d_q0, q0, first, last);
  upload_range(d_qx, qx, first, last);
  upload_range(d_qy, qy, first, last);
  upload_range(d_qz, qz, first, last);
  upload_range(d_h, h, first, last);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour, from each site's own Park/Miller stream under
   sweep color/strict (as the CPU sweep, so the two match) or the pool
------------------------------------------------------------------------- */

void AppPottsAnnealKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  const auto s = device_sites();
  const anneal::ParamsT<KK_FLOAT, KK_ACC_FLOAT> p{
    d_symk.data(), par.nsym, (KK_FLOAT) par.theta_cut, par.temp, par.tinv,
    (KK_FLOAT) par.theta_m, par.disorient};
  const int strict = strict_rng;

  int accepted = 0;
  Kokkos::parallel_reduce("PottsAnnealSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += anneal::event(i, rp, nn, nbr, s, p);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        PoolUniform<decltype(g)> u{g};
        acc += anneal::event(i, u, nn, nbr, s, p);
        pool.free_state(g);
      }
    }, accepted);
  naccept += accepted;
}

/* ----------------------------------------------------------------------
   AppPottsAnneal::app_update on the device: recovery; then the ghosts
   made current on the host, the seeds, the dilation passes (the seeds
   sent to the ghosts between them through the host), the nuclei
------------------------------------------------------------------------- */

void AppPottsAnnealKokkos::app_update(double dt)
{
  if (merge_theta > 0.0 && merge_every > 0 && nsweeps % merge_every == 0)
    merge_labels_device();

  auto hh = d_h;
  if (rrecover > 0.0) {
    const double f = exp(-rrecover * dt);
    Kokkos::parallel_for("PottsAnnealRecover", nlocal, KOKKOS_LAMBDA(const int i) {
      anneal::recover(i, hh, f);
    });
    host_data_stale = true;
  }

  if (nrate == 0.0) return;
  if (nprocs > 1) {
    ensure_host_data_current();
    comm->all();
    sync_ghosts_to_device();
  }

  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto ph = d_phase;
  auto ids = d_id;
  auto flag = d_seed;
  auto q0v = d_q0, qxv = d_qx, qyv = d_qy, qzv = d_qz;
  auto ori = d_orient;
  const unsigned long long s = salt(), k = key;
  const int ns = nspins;
  const unsigned long long nori = NORIENT;
  rx::Nucleation q = nuc;
  q.prob = nrate * dt;

  Kokkos::parallel_for("PottsAnnealSeeds", nuc.boundary ? nlocal : nlocal + nghost,
    KOKKOS_LAMBDA(const int c) {
      flag(c) = ph(c) == anneal::VOID ? 0 : rx::seed(c, ids(c), nn, nbr, sp, hh, q, s);
    });
  int nlocal_seeds = 0;
  Kokkos::parallel_reduce("PottsAnnealNSeeds", nlocal,
    KOKKOS_LAMBDA(const int i, int &n) { n += flag(i) != 0; }, nlocal_seeds);
  bigint n = nlocal_seeds, nall;
  MPI_Allreduce(&n, &nall, 1, MPI_SPK_BIGINT, MPI_SUM, world);
  if (nall == 0) return;

  auto next = d_next;
  for (int pass = 0; pass < nradius; pass++) {
    if ((pass > 0 || nuc.boundary) && nprocs > 1) {
      download_range(seedspin, flag, nlocal);
      comm->all();
      upload_range(flag, seedspin, nlocal, nlocal + nghost);
    }
    Kokkos::parallel_for("PottsAnnealGrow", nlocal, KOKKOS_LAMBDA(const int i) {
      next(i) = anneal::grow_nucleus(i, nn, nbr, sp, ph, flag, ns);
    });
    Kokkos::parallel_for("PottsAnnealCopy", nlocal, KOKKOS_LAMBDA(const int i) {
      flag(i) = next(i);
    });
  }

  Kokkos::parallel_for("PottsAnnealNuclei", nlocal, KOKKOS_LAMBDA(const int i) {
    const int snew = flag(i);
    if (!snew) return;
    const size_t o = 4 * (gw_mix(k ^ (unsigned long long) (snew - ns)) % nori);
    sp(i) = snew;
    q0v(i) = ori(o); qxv(i) = ori(o + 1); qyv(i) = ori(o + 2); qzv(i) = ori(o + 3);
    hh(i) = 0.0;
  });
}

/* ----------------------------------------------------------------------
   AppPottsAnneal::merge_labels with the bond counts and the relabel on
   the device: one key per counted bond, a << 33 | b << 1 | below theta,
   sorted, so a pair's bonds are one run; the union-find stays on the host
   (merge_resolve). Disorientations in double, as the CPU.
------------------------------------------------------------------------- */

void AppPottsAnnealKokkos::merge_labels_device()
{
  if (nprocs > 1) {
    ensure_host_data_current();
    comm->all();
    sync_ghosts_to_device();
  }
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin;
  auto ph = d_phase;
  const auto s = device_sites();
  const double *sym = d_sym.data();
  const int nsym = par.nsym, ns = nspins;
  const double theta = merge_theta;

  Kokkos::View<size_t*> off("merge_offsets", nlocal);
  size_t nkey = 0;
  Kokkos::parallel_scan("PottsAnnealMergeCount", nlocal,
    KOKKOS_LAMBDA(const int i, size_t &sum, const bool final) {
      if (final) off(i) = sum;
      for (int m = 0; m < nn(i); m++)
        if (anneal::merge_counted(i, nbr((size_t) i, m), sp, ph, ns)) sum++;
    }, nkey);

  Kokkos::View<uint64_t*> key("merge_keys", nkey);
  Kokkos::parallel_for("PottsAnnealMergeKeys", nlocal, KOKKOS_LAMBDA(const int i) {
    size_t o = off(i);
    double qi[4];
    anneal::get_q(s, i, qi);
    for (int m = 0; m < nn(i); m++) {
      const int k = nbr((size_t) i, m);
      if (!anneal::merge_counted(i, k, sp, ph, ns)) continue;
      double qk[4];
      anneal::get_q(s, k, qk);
      const uint64_t a = sp(i) < sp(k) ? sp(i) : sp(k), b = sp(i) < sp(k) ? sp(k) : sp(i);
      const uint64_t low = quaternion::disorientation_deg(sym, nsym, qi, qk) < theta;
      key(o++) = (a << 33) | (b << 1) | low;
    }
  });
  if (nkey > 0) Kokkos::sort(key);

  // one quad per run of a pair: (a, b, low, all)
  Kokkos::View<int*> slot("merge_slot", nkey);
  int npair = 0;
  Kokkos::parallel_scan("PottsAnnealMergeRuns", nkey,
    KOKKOS_LAMBDA(const size_t j, int &n, const bool final) {
      const bool start = j == 0 || (key(j) >> 1) != (key(j - 1) >> 1);
      if (start) n++;
      if (final) slot(j) = n - 1;
    }, npair);
  Kokkos::View<int*> quad("merge_quads", 4 * npair);
  Kokkos::parallel_for("PottsAnnealMergeQuads", nkey, KOKKOS_LAMBDA(const size_t j) {
    const int q = slot(j);
    if (j == 0 || (key(j) >> 1) != (key(j - 1) >> 1)) {
      quad(4 * q) = (int) (key(j) >> 33);
      quad(4 * q + 1) = (int) ((key(j) >> 1) & 0xffffffffULL);
    }
    if (key(j) & 1) Kokkos::atomic_inc(&quad(4 * q + 2));
    Kokkos::atomic_inc(&quad(4 * q + 3));
  });
  std::vector<int> buf(4 * npair);
  if (npair > 0)
    Kokkos::deep_copy(Kokkos::View<int*, Kokkos::HostSpace,
                      Kokkos::MemoryUnmanaged>(buf.data(), buf.size()), quad);

  const std::vector<int> map = merge_resolve(buf, 0);
  const int nmap = map.size() / 2;
  if (nmap == 0) return;
  Kokkos::View<int*> dmap("merge_map", 2 * nmap);
  Kokkos::deep_copy(dmap, HostView<int>(map.data(), map.size()));
  Kokkos::parallel_for("PottsAnnealRelabel", nlocal + nghost, KOKKOS_LAMBDA(const int i) {
    int lo = 0, hi = nmap;           // map is sorted by label
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (dmap(2 * mid) < sp(i)) lo = mid + 1;
      else hi = mid;
    }
    if (lo < nmap && dmap(2 * lo) == sp(i)) sp(i) = dmap(2 * lo + 1);
  });
  host_data_stale = true;
}
