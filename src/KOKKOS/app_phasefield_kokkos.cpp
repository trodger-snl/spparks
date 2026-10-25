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

using namespace SPPARKS_NS;

namespace SPPARKS_NS { template class LatticeKokkos<AppPhaseField>; }

/* ---------------------------------------------------------------------- */

AppPhaseFieldKokkos::AppPhaseFieldKokkos(SPPARKS *spk, int narg, char **arg) :
  LatticeKokkos<AppPhaseField>(spk, narg, arg) {}

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

  pf.upload_tables(ptab, tab);
  need_upload = true;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::allocate_app_views()
{
  pf.allocate(nlocal + nghost, nfield, naux);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldKokkos::sync_app_data_to_device()
{
  const int nall = nlocal + nghost;
  for (int k = 0; k < nfield; k++) upload_range(pf.f[k], darray[k], 0, nall);
  upload_range(pf.phase, phase, 0, nall);
}

void AppPhaseFieldKokkos::sync_app_data_to_host()
{
  for (int k = 0; k < nfield; k++) download_range(darray[k], pf.f[k], nlocal);
}

void AppPhaseFieldKokkos::sync_ghosts_to_device()
{
  for (int k = 0; k < nfield; k++)
    upload_range(pf.f[k], darray[k], nlocal, nlocal + nghost);
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
  with_model([&](auto &m) {
    pf.step(m, stencil, d_neighbor, d_numneigh, nlocal, nlocal + nghost, maxneigh,
            nfield);
  });

  // flags accumulate on the device and are checked with the output copy:
  // a readback every step costs a fence (launch-bound on small lattices)
  if (time + step >= nextoutput || time + step >= stoptime) {
    check_flags(pf.take_flags());
    sync_app_data_to_host();
  }
}
