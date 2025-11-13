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

#include "app_kokkos.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppKokkos::AppKokkos(SPPARKS *spk, int narg, char **arg) :
  App(spk, narg, arg)
{
  rand_pool = nullptr;
}

/* ---------------------------------------------------------------------- */

AppKokkos::~AppKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   allocate Kokkos arrays for site data
------------------------------------------------------------------------- */

void AppKokkos::allocate_kokkos_arrays()
{
  int nall = nlocal + nghost;

  // Allocate site IDs
  k_id = Kokkos::DualView<tagint*>("id", nall);

  // Allocate coordinates
  k_xyz = Kokkos::DualView<double*[3]>("xyz", nall);

  // Allocate integer arrays if needed
  if (ninteger > 0) {
    k_iarray = Kokkos::DualView<int**>("iarray", nall, ninteger);
  }

  // Allocate double arrays if needed
  if (ndouble > 0) {
    k_darray = Kokkos::DualView<double**>("darray", nall, ndouble);
  }
}

/* ----------------------------------------------------------------------
   sync all data to device
------------------------------------------------------------------------- */

void AppKokkos::sync_to_device()
{
  k_id.sync_device();
  k_xyz.sync_device();
  if (ninteger > 0) k_iarray.sync_device();
  if (ndouble > 0) k_darray.sync_device();
}

/* ----------------------------------------------------------------------
   sync all data to host
------------------------------------------------------------------------- */

void AppKokkos::sync_to_host()
{
  k_id.sync_host();
  k_xyz.sync_host();
  if (ninteger > 0) k_iarray.sync_host();
  if (ndouble > 0) k_darray.sync_host();
}

/* ----------------------------------------------------------------------
   mark all data as modified on device
------------------------------------------------------------------------- */

void AppKokkos::modify_device()
{
  k_id.modify_device();
  k_xyz.modify_device();
  if (ninteger > 0) k_iarray.modify_device();
  if (ndouble > 0) k_darray.modify_device();
}

/* ----------------------------------------------------------------------
   mark all data as modified on host
------------------------------------------------------------------------- */

void AppKokkos::modify_host()
{
  k_id.modify_host();
  k_xyz.modify_host();
  if (ninteger > 0) k_iarray.modify_host();
  if (ndouble > 0) k_darray.modify_host();
}

/* ----------------------------------------------------------------------
   sync arrays from host to device
------------------------------------------------------------------------- */

void AppKokkos::sync_arrays_host_to_device()
{
  int nall = nlocal + nghost;

  // Copy site IDs
  auto h_id = k_id.h_view;
  for (int i = 0; i < nall; i++) {
    h_id(i) = id[i];
  }
  k_id.modify_host();

  // Copy coordinates
  auto h_xyz = k_xyz.h_view;
  for (int i = 0; i < nall; i++) {
    h_xyz(i, 0) = xyz[i][0];
    h_xyz(i, 1) = xyz[i][1];
    h_xyz(i, 2) = xyz[i][2];
  }
  k_xyz.modify_host();

  // Copy integer arrays
  if (ninteger > 0) {
    auto h_iarray = k_iarray.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < ninteger; j++) {
        h_iarray(i, j) = iarray[i][j];
      }
    }
    k_iarray.modify_host();
  }

  // Copy double arrays
  if (ndouble > 0) {
    auto h_darray = k_darray.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < ndouble; j++) {
        h_darray(i, j) = darray[i][j];
      }
    }
    k_darray.modify_host();
  }

  // Sync to device
  sync_to_device();
}

/* ----------------------------------------------------------------------
   sync arrays from device to host
------------------------------------------------------------------------- */

void AppKokkos::sync_arrays_device_to_host()
{
  int nall = nlocal + nghost;

  // Sync from device first
  sync_to_host();

  // Copy site IDs
  auto h_id = k_id.h_view;
  for (int i = 0; i < nall; i++) {
    id[i] = h_id(i);
  }

  // Copy coordinates
  auto h_xyz = k_xyz.h_view;
  for (int i = 0; i < nall; i++) {
    xyz[i][0] = h_xyz(i, 0);
    xyz[i][1] = h_xyz(i, 1);
    xyz[i][2] = h_xyz(i, 2);
  }

  // Copy integer arrays
  if (ninteger > 0) {
    auto h_iarray = k_iarray.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < ninteger; j++) {
        iarray[i][j] = h_iarray(i, j);
      }
    }
  }

  // Copy double arrays
  if (ndouble > 0) {
    auto h_darray = k_darray.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < ndouble; j++) {
        darray[i][j] = h_darray(i, j);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   query if execution is on device
------------------------------------------------------------------------- */

bool AppKokkos::is_device_execution()
{
  return !std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::value;
}
