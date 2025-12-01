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
  kokkos_ncolors(0), kokkos_nsectors(0), color_lists_initialized(false)
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

   Set structure from AppLattice::init_app():
   - When bothflag=0 and ncolors>1: set[c] for c=0..ncolors-1 contains sites of color c
   - When bothflag=1 (sectors+colors):
     - set[0..nsector-1] = sector-only sets (not used here)
     - set[nsector + isector*ncolors + icolor] = sites of color icolor in sector isector
------------------------------------------------------------------------- */

void AppLatticeKokkos::setup_color_lists_kokkos()
{
  // Use existing coloring from AppLattice (computed in AppLattice::init_app)

  // Clear any existing color lists
  d_color_offsets_vec.clear();
  d_color_sites_vec.clear();

  if (ncolors <= 1) {
    // No coloring - single color with all local sites, single "sector"
    kokkos_ncolors = 1;
    kokkos_nsectors = 1;

    d_color_offsets_vec.resize(1);
    d_color_sites_vec.resize(1);

    d_color_offsets_vec[0] = Kokkos::View<int*>("color_offsets_0", 2);
    auto h_offsets = Kokkos::create_mirror_view(d_color_offsets_vec[0]);
    h_offsets(0) = 0;
    h_offsets(1) = nlocal;
    Kokkos::deep_copy(d_color_offsets_vec[0], h_offsets);

    d_color_sites_vec[0] = Kokkos::View<int*>("color_sites_0", nlocal);
    auto h_sites = Kokkos::create_mirror_view(d_color_sites_vec[0]);
    for (int i = 0; i < nlocal; i++) {
      h_sites(i) = i;
    }
    Kokkos::deep_copy(d_color_sites_vec[0], h_sites);

    color_lists_initialized = true;
    return;
  }

  kokkos_ncolors = ncolors;

  // Determine number of sectors to track
  // When bothflag=1, we need per-sector color lists
  // When bothflag=0, we use a single set of color lists
  if (bothflag) {
    kokkos_nsectors = nsector;
  } else {
    kokkos_nsectors = 1;
  }

  // Resize vectors to hold per-sector views
  d_color_offsets_vec.resize(kokkos_nsectors);
  d_color_sites_vec.resize(kokkos_nsectors);

  // Build color lists for each sector
  for (int isector = 0; isector < kokkos_nsectors; isector++) {
    char name[64];
    snprintf(name, sizeof(name), "color_offsets_%d", isector);
    d_color_offsets_vec[isector] = Kokkos::View<int*>(name, ncolors + 1);
    auto h_offsets = Kokkos::create_mirror_view(d_color_offsets_vec[isector]);

    // Build CSR offsets for this sector
    h_offsets(0) = 0;
    for (int c = 0; c < ncolors; c++) {
      int set_idx;
      if (bothflag) {
        // set[nsector + isector*ncolors + icolor] has color sites for this sector
        set_idx = nsector + isector * ncolors + c;
      } else {
        // set[c] has all sites of color c
        set_idx = c;
      }
      h_offsets(c + 1) = h_offsets(c) + set[set_idx].nlocal;
    }
    Kokkos::deep_copy(d_color_offsets_vec[isector], h_offsets);

    // Build flattened site list for this sector
    int total_sites = h_offsets(ncolors);
    snprintf(name, sizeof(name), "color_sites_%d", isector);
    d_color_sites_vec[isector] = Kokkos::View<int*>(name, total_sites > 0 ? total_sites : 1);
    auto h_sites = Kokkos::create_mirror_view(d_color_sites_vec[isector]);

    int offset = 0;
    for (int c = 0; c < ncolors; c++) {
      int set_idx;
      if (bothflag) {
        set_idx = nsector + isector * ncolors + c;
      } else {
        set_idx = c;
      }
      for (int j = 0; j < set[set_idx].nlocal; j++) {
        h_sites(offset++) = set[set_idx].site2i[j];
      }
    }
    Kokkos::deep_copy(d_color_sites_vec[isector], h_sites);
  }

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

  // Cache color offsets on host for each sector
  // h_offsets_vec[isector] holds the mirror view for sector isector
  std::vector<Kokkos::View<int*>::HostMirror> h_offsets_vec(kokkos_nsectors);
  for (int s = 0; s < kokkos_nsectors; s++) {
    h_offsets_vec[s] = Kokkos::create_mirror_view(d_color_offsets_vec[s]);
    Kokkos::deep_copy(h_offsets_vec[s], d_color_offsets_vec[s]);
  }

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

      // Determine which sector's color lists to use
      // When bothflag=1, use iset (the current sector)
      // When bothflag=0, always use sector 0's lists
      int sector_idx = bothflag ? iset : 0;
      auto& h_offsets = h_offsets_vec[sector_idx];
      // Set d_color_sites to the current sector's view - derived classes use this
      d_color_sites = d_color_sites_vec[sector_idx];

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
