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

#include "app_lattice_kokkos.h"
#include "random_park.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppLatticeKokkos::AppLatticeKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLattice(spk, narg, arg)
{
  rand_pool = nullptr;
}

/* ---------------------------------------------------------------------- */

AppLatticeKokkos::~AppLatticeKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   allocate Kokkos arrays for neighbor data
------------------------------------------------------------------------- */

void AppLatticeKokkos::allocate_kokkos_neighbor_arrays()
{
  int nall = nlocal + nghost;

  // Allocate neighbor count array
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);

  // Allocate neighbor array (2D ragged array approximated as rectangular)
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }

  // Allocate propensity array
  k_propensity = Kokkos::DualView<double*>("propensity", nall);
}

/* ----------------------------------------------------------------------
   sync neighbor data to device
------------------------------------------------------------------------- */

void AppLatticeKokkos::sync_neighbor_data_to_device()
{
  k_numneigh.sync_device();
  k_neighbor.sync_device();
  k_propensity.sync_device();
}

/* ----------------------------------------------------------------------
   sync neighbor data to host
------------------------------------------------------------------------- */

void AppLatticeKokkos::sync_neighbor_data_to_host()
{
  k_numneigh.sync_host();
  k_neighbor.sync_host();
  k_propensity.sync_host();
}

/* ----------------------------------------------------------------------
   sync neighbor arrays from host to device
------------------------------------------------------------------------- */

void AppLatticeKokkos::sync_neighbor_arrays_host_to_device()
{
  int nall = nlocal + nghost;

  // Copy neighbor counts
  auto h_numneigh = k_numneigh.h_view;
  for (int i = 0; i < nall; i++) {
    h_numneigh(i) = numneigh[i];
  }
  k_numneigh.modify_host();

  // Copy neighbor lists
  if (maxneigh > 0) {
    auto h_neighbor = k_neighbor.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < numneigh[i]; j++) {
        h_neighbor(i, j) = neighbor[i][j];
      }
    }
    k_neighbor.modify_host();
  }

  // Copy propensities
  auto h_propensity = k_propensity.h_view;
  for (int i = 0; i < nall; i++) {
    h_propensity(i) = propensity[i];
  }
  k_propensity.modify_host();

  // Sync to device
  sync_neighbor_data_to_device();
}

/* ----------------------------------------------------------------------
   copy neighbor data from device back to host arrays
------------------------------------------------------------------------- */

void AppLatticeKokkos::copy_neighbor_data_to_host()
{
  int nall = nlocal + nghost;

  // Sync from device first
  sync_neighbor_data_to_host();

  // Copy neighbor counts
  auto h_numneigh = k_numneigh.h_view;
  for (int i = 0; i < nall; i++) {
    numneigh[i] = h_numneigh(i);
  }

  // Copy neighbor lists
  if (maxneigh > 0) {
    auto h_neighbor = k_neighbor.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < numneigh[i]; j++) {
        neighbor[i][j] = h_neighbor(i, j);
      }
    }
  }

  // Copy propensities
  auto h_propensity = k_propensity.h_view;
  for (int i = 0; i < nall; i++) {
    propensity[i] = h_propensity(i);
  }
}
