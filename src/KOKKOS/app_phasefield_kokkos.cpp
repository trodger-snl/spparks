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

#include "app_phasefield_kokkos.h"
#include "app_lattice_kokkos_impl.h"
#include "comm_lattice.h"
#include "timer.h"
#include "error.h"
#include <utility>

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPhaseField>; }

namespace {
template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
}

/* ---------------------------------------------------------------------- */

AppPhaseFieldKokkos::AppPhaseFieldKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPhaseField>(spk, narg, arg)
{
  d_flags = Kokkos::View<int>("pf_flags");
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::grow_app()
{
  AppPhaseField::grow_app();   // phase
  LatticeKokkos<AppPhaseField>::grow_app();
}

/* ----------------------------------------------------------------------
   the CPU init (checks, stencil, tables, dt estimate) is safe to run here
------------------------------------------------------------------------- */

void AppPhaseFieldKokkos::init_app()
{
  AppPhaseField::init_app();
  LatticeKokkos<AppPhaseField>::init_app();

  d_tab = Kokkos::View<double*>("pf_tab", tab.size());
  Kokkos::deep_copy(d_tab, HostView<double>(tab.data(), tab.size()));
  d_ptab = Kokkos::View<PFPhaseTable*>("pf_ptab", ptab.size());
  Kokkos::deep_copy(d_ptab, HostView<PFPhaseTable>(ptab.data(), ptab.size()));
  need_upload = true;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::allocate_app_views()
{
  const int nall = nlocal + nghost;
  for (int k = 0; k < nfield; k++) {
    d_f[k] = Kokkos::View<double*>("pf_f", nall);
    d_fnew[k] = Kokkos::View<double*>("pf_fnew", nall);   // swapped with d_f
  }
  for (int k = 0; k < naux; k++) d_aux[k] = Kokkos::View<double*>("pf_aux", nall);
  d_phase = Kokkos::View<int*>("pf_phase", nall);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::sync_app_data_to_device()
{
  const int nall = nlocal + nghost;
  for (int k = 0; k < nfield; k++) upload_range(d_f[k], darray[k], 0, nall);
  upload_range(d_phase, phase, 0, nall);
}

void AppPhaseFieldKokkos::sync_app_data_to_host()
{
  for (int k = 0; k < nfield; k++) download_range(darray[k], d_f[k], nlocal);
}

void AppPhaseFieldKokkos::sync_ghosts_to_device()
{
  for (int k = 0; k < nfield; k++)
    upload_range(d_f[k], darray[k], nlocal, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   one step of the model, as AppPhaseField::app_update
------------------------------------------------------------------------- */

template <class M>
void AppPhaseFieldKokkos::step_kernels(const M &m)
{
  DevCtx x;
  x.st = stencil;
  x.nbr = d_neighbor;
  for (int k = 0; k < PF_MAXF; k++) x.f[k] = d_f[k];
  for (int k = 0; k < PF_MAXA; k++) x.a[k] = d_aux[k];
  x.phase = d_phase;
  x.pt = d_ptab.data();
  x.tab = d_tab.data();

  Kokkos::Array<Kokkos::View<double*>, PF_MAXF> fn;
  for (int k = 0; k < PF_MAXF; k++) fn[k] = d_fnew[k];
  auto nn = d_numneigh;
  auto flags = d_flags;
  const int nloc = nlocal, nall = nlocal + nghost, mn = maxneigh, nf = nfield;
  const int na = M::naux;

  if (na)
    Kokkos::parallel_for("PFAux", nall, KOKKOS_LAMBDA(const int i) {
      if (i < nloc || nn(i) == mn) {
        double a[PF_MAXA];
        m.aux(x, i, a);
        for (int k = 0; k < na; k++) x.a[k](i) = a[k];
      }
    });
  Kokkos::parallel_for("PFUpdate", nloc, KOKKOS_LAMBDA(const int i) {
    double out[PF_MAXF];
    const int f = m.update(x, i, out);
    for (int k = 0; k < nf; k++) fn[k](i) = out[k];
    if (f) Kokkos::atomic_or(&flags(), f);
  });
  // the new fields become current; their ghosts are refreshed by the next
  // step's exchange (np > 1; np = 1 has no ghosts)
  for (int k = 0; k < nf; k++) std::swap(d_f[k], d_fnew[k]);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::app_update(double step)
{
  // the update-only loop never syncs: upload at the first step of a run,
  // download when this step ends at an output (or the end of the run)
  if (need_upload) {
    sync_app_data_to_device();
    need_upload = false;
  }

  // ponytail: ghost fields through the host every step; pack on the device
  // if np > 1 runs are comm-bound
  if (nprocs > 1) {
    timer->stamp();
    sync_app_data_to_host();
    int didx[PF_MAXF];
    for (int k = 0; k < nfield; k++) didx[k] = k;
    comm->all_selective(nullptr, 0, didx, nfield);
    timer->stamp(TIME_COMM);
    sync_ghosts_to_device();
  }

  set_dt(step);
  with_model([&](auto &m) { step_kernels(m); });

  // flags accumulate on the device and are checked with the output copy:
  // a readback every step costs a fence (launch-bound on small lattices)
  if (time + step >= nextoutput || time + step >= stoptime) {
    int f;
    Kokkos::deep_copy(f, d_flags);
    Kokkos::deep_copy(d_flags, 0);
    check_flags(f);
    sync_app_data_to_host();
  }
}
