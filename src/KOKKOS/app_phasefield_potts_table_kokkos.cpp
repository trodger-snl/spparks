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

#include "app_phasefield_potts_table_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "random_park_kokkos.h"
#include "comm_lattice.h"
#include "timer.h"
#include "error.h"

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPhaseFieldPottsTable>; }

namespace {
// a Kokkos pool generator behind the uniform() the pftable:: event draws with
template <class G> struct PoolUniform {
  G &g;
  KOKKOS_INLINE_FUNCTION double uniform() { return g.drand(); }
};

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
}

/* ---------------------------------------------------------------------- */

AppPhaseFieldPottsTableKokkos::
AppPhaseFieldPottsTableKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPhaseFieldPottsTable>(spk, narg, arg)
{
  device_app_update = true;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::grow_app()
{
  AppPhaseFieldPottsTable::grow_app();   // spin, phase, c, T
  LatticeKokkos<AppPhaseFieldPottsTable>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init (checks, FD stencil, tables, phases) is safe to run here,
   unlike the quaternion apps' (see LatticeKokkos::init_app)
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::init_app()
{
  AppPhaseFieldPottsTable::init_app();
  LatticeKokkos<AppPhaseFieldPottsTable>::init_app();

  pf.upload_tables(ptab, tab);
  upload_reset_list();

  sync_app_data_to_device();
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::upload_reset_list()
{
  if (reset_uploaded == pf_resetversion) return;
  const int n = pf_resetlist.size();
  d_reset = Kokkos::View<int*>("pf_reset", n);
  d_resetval = Kokkos::View<double*>("pf_resetval", n);
  Kokkos::deep_copy(d_reset, HostView<int>(pf_resetlist.data(), n));
  Kokkos::deep_copy(d_resetval, HostView<double>(pf_resetlistvals.data(), n));
  reset_uploaded = pf_resetversion;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::allocate_app_views()
{
  pf.allocate(nlocal + nghost, 2, 1);
  d_spin = Kokkos::View<int*>("spin", nlocal + nghost);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::sync_app_data_to_device()
{
  const int nall = nlocal + nghost;
  upload_range(d_spin, spin, 0, nall);
  upload_range(pf.phase, phase, 0, nall);
  upload_range(pf.f[0], c, 0, nall);
  upload_range(pf.f[1], T, 0, nall);
}

void AppPhaseFieldPottsTableKokkos::sync_app_data_to_host()
{
  download_range(spin, d_spin, nlocal);
  download_range(phase, pf.phase, nlocal);
  download_range(c, pf.f[0], nlocal);
  download_range(T, pf.f[1], nlocal);
}

void AppPhaseFieldPottsTableKokkos::sync_ghosts_to_device()
{
  const int nall = nlocal + nghost;
  upload_range(d_spin, spin, nlocal, nall);
  upload_range(pf.phase, phase, nlocal, nall);
  upload_range(pf.f[0], c, nlocal, nall);
  upload_range(pf.f[1], T, nlocal, nall);
}

/* ----------------------------------------------------------------------
   rKMC sweep of one colour: pftable::rejection_event per site, drawing from
   the site's own Park/Miller stream under sweep color/strict (as the CPU
   sweep does, so the two match) or from the Kokkos pool
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::sweep_color_kokkos(int, int start, int count)
{
  auto sites = d_color_sites;
  auto nn = d_numneigh;
  auto nbr = d_neighbor;
  auto sp = d_spin, ph = pf.phase;
  auto cc = pf.f[0], TT = pf.f[1];
  auto seeds = d_siteseeds;
  auto pool = *rand_pool;
  const PFPhaseTable *pt = pf.ptab.data();
  const double *tb = pf.tab.data();
  const PFParams q = chh.q;
  const int np = ptab.size(), strict = strict_rng;
  const double temp = temperature, tinv = t_inverse;

  int accepted = 0;
  Kokkos::parallel_reduce("PFPottsSweep", count,
    KOKKOS_LAMBDA(const int idx, int &acc) {
      const int i = sites(start + idx);
      if (strict) {
        RandomParkDevice rp(seeds(i));
        acc += pftable::rejection_event(i, rp, nn, nbr, sp, ph, cc, TT, pt, np,
                                        tb, q, temp, tinv);
        seeds(i) = rp.seed;
      } else {
        auto g = pool.get_state();
        PoolUniform<decltype(g)> u{g};
        acc += pftable::rejection_event(i, u, nn, nbr, sp, ph, cc, TT, pt, np,
                                        tb, q, temp, tinv);
        pool.free_state(g);
      }
    }, accepted);
  naccept += accepted;
}

/* ----------------------------------------------------------------------
   phase-field steps over one Potts step, as AppPhaseFieldPottsTable::
   app_update: mu, then the coupled c/T step, then the reset list; ghosts
   exchanged through the host after each step (np > 1)
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTableKokkos::app_update(double stoptime)
{
  // ponytail: the update-only loop (AppLattice::iterate_app_update_only)
  // never syncs, so do the full round trip here; it is the validation path
  if (app_update_only) sync_app_data_to_device();
  upload_reset_list();

  // the sweeps left the ghosts' spins, phases stale: exchange everything
  if (nprocs > 1) {
    timer->stamp();
    ensure_host_data_current();
    comm->all();
    timer->stamp(TIME_COMM);
    sync_ghosts_to_device();
  }

  const int nall = nlocal + nghost;
  auto rs = d_reset;
  auto rv = d_resetval;
  const int nreset = rs.extent(0);
  const int dsel[2] = {0, 1};

  double localtime = 0.0;
  int done = 0;
  while (!done) {
    timer->stamp();
    pf.step(chh, stencil, d_neighbor, d_numneigh, nlocal, nall, maxneigh, 2);
    auto cc = pf.f[0];
    if (nreset)
      Kokkos::parallel_for("PFReset", nreset, KOKKOS_LAMBDA(const int m) {
        cc(rs(m)) = rv(m);
      });

    // ponytail: ghost c,T through the host every PF step; pack on the device
    // if np > 1 runs are comm-bound
    if (nprocs > 1) {
      download_range(c, pf.f[0], nlocal);
      download_range(T, pf.f[1], nlocal);
      timer->stamp(TIME_SOLVE);
      comm->all_selective(nullptr, 0, dsel, 2);
      timer->stamp(TIME_COMM);
      upload_range(pf.f[0], c, nlocal, nall);
      upload_range(pf.f[1], T, nlocal, nall);
    }
    timer->stamp(TIME_SOLVE);

    localtime += dt;
    if (localtime >= (stoptime - 1e-6)) done = 1;
    if (localtime > (stoptime + 1e-6)) {
      char errorstr[128];
      snprintf(errorstr, sizeof(errorstr),
               "PF step time (%f) in excess of Potts step time (%f)",
               localtime, stoptime);
      error->all(FLERR, errorstr);
    }
  }

  // the kernels finish here (the copy fences): charge them to Solve
  const int f = pf.take_flags();
  timer->stamp(TIME_SOLVE);
  check_flags(f);

  if (app_update_only) sync_app_data_to_host();
  else host_data_stale = true;
}
