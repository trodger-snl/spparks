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
#include "error.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"
#include "domain.h"

using namespace SPPARKS_NS;

enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};

/* ---------------------------------------------------------------------- */

AppLatticeKokkos::AppLatticeKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLattice(spk, narg, arg), rand_pool(nullptr),
  kokkos_ncolors(0), color_lists_initialized(false)
{
  // Initialize RNG pool with a seed using world rank
  int me;
  MPI_Comm_rank(world, &me);
  rand_pool = new pool_type(12345 + me);
}

/* ---------------------------------------------------------------------- */

AppLatticeKokkos::~AppLatticeKokkos()
{
  if (rand_pool) delete rand_pool;
}

/* ----------------------------------------------------------------------
   grow arrays - called when lattice grows
   Note: AppLattice::grow_app() is pure virtual so we cannot call it
------------------------------------------------------------------------- */

void AppLatticeKokkos::grow_app()
{
  int nall = nlocal + nghost;

  // Allocate neighbor arrays
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }

  // Derived class allocates app-specific views
  allocate_app_views();
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

void AppLatticeKokkos::init_app()
{
  // Call parent initialization (sets up sweepflag, ncolors, sets, etc.)
  AppLattice::init_app();

  // Kokkos apps require coloring for correct parallel GPU execution
  if (sweepflag != NOSWEEP && sweepflag != COLOR && sweepflag != COLOR_STRICT)
    error->all(FLERR,"Kokkos app requires 'sweep color' or 'sweep color/strict' "
               "for correct GPU execution");

  // Sync neighbor arrays to device
  sync_neighbors_to_device();

  // Setup color lists for GPU
  setup_color_lists_kokkos();
}

/* ----------------------------------------------------------------------
   sync neighbor arrays from host to device
------------------------------------------------------------------------- */

void AppLatticeKokkos::sync_neighbors_to_device()
{
  int nall = nlocal + nghost;

  auto h_numneigh = k_numneigh.h_view;
  for (int i = 0; i < nall; i++) {
    h_numneigh(i) = numneigh[i];
  }
  k_numneigh.modify_host();
  k_numneigh.sync_device();

  if (maxneigh > 0) {
    auto h_neighbor = k_neighbor.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < numneigh[i]; j++) {
        h_neighbor(i, j) = neighbor[i][j];
      }
    }
    k_neighbor.modify_host();
    k_neighbor.sync_device();
  }
}

/* ----------------------------------------------------------------------
   setup color lists for GPU sweeps
   uses existing coloring from AppLattice (computed in AppLattice::init_app)
------------------------------------------------------------------------- */

void AppLatticeKokkos::setup_color_lists_kokkos()
{
  // Use existing coloring from AppLattice (computed in AppLattice::init_app)

  if (ncolors <= 1) {
    // No coloring - single color with all local sites
    kokkos_ncolors = 1;

    d_color_offsets = Kokkos::View<int*>("color_offsets", 2);
    auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);
    h_offsets(0) = 0;
    h_offsets(1) = nlocal;
    Kokkos::deep_copy(d_color_offsets, h_offsets);

    d_color_sites = Kokkos::View<int*>("color_sites", nlocal);
    auto h_sites = Kokkos::create_mirror_view(d_color_sites);
    for (int i = 0; i < nlocal; i++) {
      h_sites(i) = i;
    }
    Kokkos::deep_copy(d_color_sites, h_sites);

    color_lists_initialized = true;
    return;
  }

  kokkos_ncolors = ncolors;

  // Build CSR offsets from set[] data
  d_color_offsets = Kokkos::View<int*>("color_offsets", ncolors + 1);
  auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);

  h_offsets(0) = 0;
  for (int c = 0; c < ncolors; c++) {
    // When bothflag=0 and ncolors>1, set[c] contains sites of color c
    // When bothflag=1, set[nsector + isector*ncolors + icolor] has color sites per sector
    // For now, handle simple case (no sectors or single sector)
    int set_idx = c;
    h_offsets(c + 1) = h_offsets(c) + set[set_idx].nlocal;
  }
  Kokkos::deep_copy(d_color_offsets, h_offsets);

  // Build flattened site list
  int total_sites = h_offsets(ncolors);
  d_color_sites = Kokkos::View<int*>("color_sites", total_sites);
  auto h_sites = Kokkos::create_mirror_view(d_color_sites);

  int offset = 0;
  for (int c = 0; c < ncolors; c++) {
    int set_idx = c;
    for (int j = 0; j < set[set_idx].nlocal; j++) {
      h_sites(offset++) = set[set_idx].site2i[j];
    }
  }
  Kokkos::deep_copy(d_color_sites, h_sites);

  color_lists_initialized = true;
}

/* ----------------------------------------------------------------------
   main iteration loop - Kokkos version with color-based sweeps
------------------------------------------------------------------------- */

void AppLatticeKokkos::iterate_rejection(double stoptime)
{
  // Ensure color lists are ready
  if (!color_lists_initialized) {
    setup_color_lists_kokkos();
  }

  // Cache color offsets on host for the loop
  auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);
  Kokkos::deep_copy(h_offsets, d_color_offsets);

  int done = 0;
  while (!done) {
    // For each sector (or single pass if no sectors)
    int nset_loop = sectorflag ? nsector : 1;

    for (int iset = 0; iset < nset_loop; iset++) {

      // MPI communication
      if (nprocs > 1) {
        timer->stamp();
        sync_app_data_to_host();  // Virtual - derived class implements
        timer->stamp(TIME_SOLVE);

        timer->stamp();
        if (sectorflag) comm->sector(iset);
        else comm->all();
        timer->stamp(TIME_COMM);
      }

      if (Lmask) boundary_clear_mask(iset);

      // Sync to device after MPI
      timer->stamp();
      sync_app_data_to_device();  // Virtual - derived class implements
      timer->stamp(TIME_SOLVE);

      // === CORE: Process each color sequentially ===
      for (int icolor = 0; icolor < kokkos_ncolors; icolor++) {
        int color_start = h_offsets(icolor);
        int color_count = h_offsets(icolor + 1) - color_start;

        if (color_count > 0) {
          // Derived class implements the actual sweep kernel
          sweep_color_kokkos(icolor, color_start, color_count);

          // CRITICAL: Fence before next color to ensure writes are visible
          Kokkos::fence();
        }
      }
      nattempt += nlocal;

      // Sync back for reverse communication
      timer->stamp();
      sync_app_data_to_host();
      timer->stamp(TIME_SOLVE);

      if (nprocs > 1) {
        timer->stamp();
        if (sectorflag) comm->reverse_sector(iset);
        else comm->all_reverse();
        timer->stamp(TIME_COMM);
      }
    }

    // Time advancement
    if (allow_app_update) app_update(dt_rkmc);
    nsweeps++;
    time += dt_rkmc;

    if (time >= stoptime) done = 1;
    if (done || time >= nextoutput) nextoutput = output->compute(time, done);
    timer->stamp(TIME_OUTPUT);
  }
}
