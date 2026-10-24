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

// LatticeKokkos<Base> member definitions. Include in the one .cpp that
// instantiates the layer for a Base (template class LatticeKokkos<Base>;).
#ifndef SPK_APP_LATTICE_KOKKOS_IMPL_H
#define SPK_APP_LATTICE_KOKKOS_IMPL_H

#include "app_lattice_kokkos.h"
#include <algorithm>
#include <cstring>
#include "error.h"
#include "memory.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"
#include "domain.h"
#include "random_mars.h"

namespace SPPARKS_NS {

// AppLattice's sweepflag values (app_lattice.cpp)
namespace LatticeKokkosSweep { enum {NOSWEEP, RANDOM, RASTER, COLOR, COLOR_STRICT}; }

/* ---------------------------------------------------------------------- */

template <class Base>
LatticeKokkos<Base>::LatticeKokkos(SPPARKS *spk, int narg, char **arg) :
  Base(spk, narg, arg), rand_pool(nullptr),
  kokkos_ncolors(0), kokkos_nsectors(0),
  strict_rng(0)
{
#ifdef KOKKOS_ENABLE_HIP
  int dev = 0, integrated = 0;
  if (hipGetDevice(&dev) == hipSuccess &&
      hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, dev) == hipSuccess)
    integrated_gpu = integrated;
#endif
}

/* ---------------------------------------------------------------------- */

template <class Base>
LatticeKokkos<Base>::~LatticeKokkos()
{
  if (rand_pool) delete rand_pool;
#ifdef KOKKOS_ENABLE_HIP
  if (pinned_sites) {
    hipDeviceSynchronize();
    for (void *p : pinned) hipHostFree(p);
    // ~AppLattice and ~App free these blocks: leave them none to free
    numneigh = nullptr;
    if (neighbor) neighbor[0] = nullptr;
    xyz[0] = nullptr;
    for (int i = 0; i < ninteger; i++) iarray[i] = nullptr;
    for (int i = 0; i < ndouble; i++) darray[i] = nullptr;
  }
#endif
}

/* ----------------------------------------------------------------------
   grow arrays - called when lattice grows
   Note: AppLattice::grow_app() is pure virtual so we cannot call it

   Device views are NOT allocated here. create_sites grows the lattice
   DELTA sites at a time, so allocating here reallocated and zero-filled every
   view ~nsites/DELTA times (quadratic; seconds at 256^3, worse for the
   additive app's dozen views). allocate_device_views() sizes them once, in
   init_app(); nothing touches them before that.
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::grow_app() {}

/* ----------------------------------------------------------------------
   integrated GPU: move the per-site host arrays into pinned host memory the
   GPU reads in place (hipHostMalloc, non-coherent = coarse-grained, as the
   HDF5 SeriesMemory hook), so site_view() can alias them instead of holding
   a device copy. Runs once, when the sites exist: AppLattice::grow() never
   runs again (create_sites/read_sites refuse once sites exist), so nothing
   reallocs these blocks. A gather from this memory measured ~10% slower than
   hipMalloc on gfx1151.
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::pin_site_arrays()
{
#ifdef KOKKOS_ENABLE_HIP
  if (nmax == 0) return;
  auto move = [&](void *old, size_t bytes) {
    void *p = nullptr;
    if (hipHostMalloc(&p, bytes, hipHostMallocNonCoherent) != hipSuccess)
      error->one(FLERR, "Could not allocate pinned memory for the site arrays");
    // threaded copy: one memcpy stream is far below the memory bandwidth
    const size_t chunk = (size_t) 1 << 24;
    #pragma omp parallel for schedule(static)
    for (size_t off = 0; off < bytes; off += chunk)
      memcpy((char *) p + off, (char *) old + off, std::min(chunk, bytes - off));
    memory->sfree(old);
    pinned.push_back(p);
    return p;
  };
  const size_t n = nmax;
  numneigh = (int *) move(numneigh, n * sizeof(int));
  if (neighbor) {
    int *data = (int *) move(neighbor[0], n * maxneigh * sizeof(int));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) neighbor[i] = &data[i * maxneigh];
  }
  double *x = (double *) move(xyz[0], n * 3 * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) xyz[i] = &x[3 * i];
  for (int k = 0; k < ninteger; k++) iarray[k] = (int *) move(iarray[k], n * sizeof(int));
  for (int k = 0; k < ndouble; k++) darray[k] = (double *) move(darray[k], n * sizeof(double));
  pinned_sites = true;
  grow_app();  // re-point the apps' named aliases (spin, q0, T, ...)
  if (domain->me == 0 && screen)
    fprintf(screen, "Site arrays shared with the integrated GPU (pinned host memory)\n");
#endif
}

/* ----------------------------------------------------------------------
   (re)allocate device views when nlocal+nghost changed since the last run.
   Same size -> keep them: the additive app carries device-only state
   (solid_d, spath, ...) from one run to the next.
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::allocate_device_views()
{
  int nall = nlocal + nghost;
  if (d_numneigh.extent(0) == (size_t) nall && views_maxneigh == maxneigh) return;
  views_maxneigh = maxneigh;

  // Allocate neighbor arrays
  d_numneigh = site_view("numneigh", numneigh, nall);
  // No zero fill: sync_neighbors_to_device() writes every j < numneigh[i]
  // and every kernel stops at numneigh, so the padding is never read.
  if (maxneigh > 0 && pinned_sites)
    d_neighbor_rm = Kokkos::View<int**, Kokkos::LayoutRight>(neighbor[0], nall, maxneigh);
  else if (maxneigh > 0 && host_layout_sites)
    d_neighbor_rm = Kokkos::View<int**, Kokkos::LayoutRight>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighbor"), nall, maxneigh);
  else if (maxneigh > 0)
    d_neighbor = Kokkos::View<int**>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighbor"), nall, maxneigh);

  // Derived class allocates app-specific views
  allocate_app_views();
}

/* ----------------------------------------------------------------------
   initialize before each run
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::init_app()
{
  // Not Base::init_app(): that is the CPU app's own run setup (AppLattice's
  // is empty), which the kk app replaces -- the quaternion apps' would
  // re-randomise every orientation, the additive app's would step the physics.

  // Device RNG pool, from the deck seed (unknown at construction) and the
  // rank. Made once, so a second run continues the stream, not repeats it.
  if (!rand_pool) {
    int me;
    MPI_Comm_rank(world, &me);
    rand_pool = new pool_type((uint64_t) ranmaster->initial_seed() * 65536 + me);
  }

  if (host_layout_sites && integrated_gpu && !pinned_sites) pin_site_arrays();
  allocate_device_views();

  // Kokkos apps require coloring for correct parallel GPU execution
  using namespace LatticeKokkosSweep;
  if (sweepflag != NOSWEEP && sweepflag != COLOR && sweepflag != COLOR_STRICT)
    error->all(FLERR,"Kokkos app requires 'sweep color' or 'sweep color/strict' "
               "for correct GPU execution");

  // Sync neighbor arrays to device
  sync_neighbors_to_device();

  // Setup color lists for GPU
  setup_color_lists_kokkos();

  // Per-site RNG streams, after the color lists.
  setup_siteseeds_kokkos();
}

/* ----------------------------------------------------------------------
   Upload AppLattice's per-site Park/Miller seeds for `sweep color/strict`.

   AppLattice::init() has already built siteseeds[nlocal], each seeded from the
   site's GLOBAL id (tagreset(seed,id[i],100) -- the 100-draw warmup matters,
   Park/Miller streams from adjacent seeds are correlated at the start). It runs
   before init_app(), so the array is populated by the time we get here. All we
   do is move it to the device so the kernel can carry each site's stream the
   way sweep_nomask_strict does on the host.
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::setup_siteseeds_kokkos()
{
  strict_rng = (sweepflag == LatticeKokkosSweep::COLOR_STRICT) ? 1 : 0;
  if (!strict_rng) { d_siteseeds = Kokkos::View<int*>(); return; }

  // A rank may legitimately own ZERO sites (a subdomain that misses the
  // geometry), and Memory::smalloc returns NULL for a zero-length array
  // (memory.cpp:38) -- so NULL is only an error when this rank actually has
  // sites. Report with error->one(), NOT error->all(): the condition is
  // rank-local, and error->all() BARRIERS, so a zero-site rank would sit in
  // that barrier while the others ran on into AppLattice::setup's Allreduce.
  if (nlocal > 0 && siteseeds == NULL)
    error->one(FLERR,"sweep color/strict requested but per-site seeds were "
                     "not initialized");

  // Only owned sites are swept, so only they carry a stream; ghosts never draw.
  d_siteseeds = Kokkos::View<int*>("siteseeds", nlocal > 0 ? nlocal : 1);
  auto h_seeds = Kokkos::create_mirror_view(d_siteseeds);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nlocal; i++) h_seeds(i) = siteseeds[i];
  Kokkos::deep_copy(d_siteseeds, h_seeds);
}

/* ----------------------------------------------------------------------
   sync neighbor arrays from host to device
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::sync_neighbors_to_device()
{
  const int nall = nlocal + nghost;
  using HostUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

  Kokkos::deep_copy(d_numneigh,
      Kokkos::View<const int*, Kokkos::HostSpace, HostUnmanaged>(numneigh, nall));

  if (maxneigh > 0 && nall > 0 && host_layout_sites) {
    // neighbor[][] is one row-major nall x maxneigh block (Memory::create),
    // the same layout as d_neighbor_rm: one straight copy.
    Kokkos::deep_copy(d_neighbor_rm,
        Kokkos::View<const int**, Kokkos::LayoutRight, Kokkos::HostSpace, HostUnmanaged>(
            neighbor[0], nall, maxneigh));
  } else if (maxneigh > 0 && nall > 0) {
    // neighbor[][] is one row-major nall x maxneigh block (Memory::create);
    // d_neighbor is the device layout (LayoutLeft on GPUs). Move it in row
    // chunks through a small row-major device buffer and transpose on the
    // device. No full-size host or device staging copy, and only plain
    // host->device copies, so this works on discrete GPUs, APUs and OpenMP.
    using RowMajor = Kokkos::View<int**, Kokkos::LayoutRight>;
    const int chunk = std::min(nall, std::max(1, (1 << 26) / maxneigh));  // ~256 MB
    RowMajor stage(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighbor_stage"),
                   chunk, maxneigh);
    const Kokkos::View<const int**, Kokkos::LayoutRight, Kokkos::HostSpace, HostUnmanaged>
        h_all(neighbor[0], nall, maxneigh);
    auto nbr = d_neighbor;
    auto nn = d_numneigh;
    const int mn = maxneigh;
    for (int r0 = 0; r0 < nall; r0 += chunk) {
      const int n = std::min(chunk, nall - r0);
      const auto rows = Kokkos::make_pair(r0, r0 + n);
      const auto srows = Kokkos::make_pair(0, n);
      Kokkos::deep_copy(Kokkos::subview(stage, srows, Kokkos::ALL),
                        Kokkos::subview(h_all, rows, Kokkos::ALL));
      Kokkos::parallel_for("neighbor_transpose", Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const int i) {
          const int ni = nn(r0 + i);
          for (int j = 0; j < mn; j++)
            if (j < ni) nbr((size_t) (r0 + i), j) = stage(i, j);
        });
    }
    Kokkos::fence();
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

template <class Base>
void LatticeKokkos<Base>::setup_color_lists_kokkos()
{
  // ncolors <= 1: one "colour" holding every local site, one sector.
  const bool colored = ncolors > 1;
  kokkos_ncolors = colored ? ncolors : 1;
  kokkos_nsectors = (colored && bothflag) ? nsector : 1;
  color_offsets.assign(kokkos_nsectors, std::vector<int>(kokkos_ncolors + 1, 0));
  d_color_sites_vec.assign(kokkos_nsectors, Kokkos::View<int*>());
  // iterate() then runs iterate_app_update_only, which never reads the lists.
  if (allow_app_update && app_update_only) return;

  for (int isector = 0; isector < kokkos_nsectors; isector++) {
    // set[c] holds colour c; with sectors, set[nsector + isector*ncolors + c]
    auto colour_set = [&](int c) -> Set & {
      return set[bothflag ? nsector + isector * ncolors + c : c];
    };
    std::vector<int> &off = color_offsets[isector];
    for (int c = 0; c < kokkos_ncolors; c++)
      off[c + 1] = off[c] + (colored ? colour_set(c).nlocal : nlocal);

    char name[64];
    snprintf(name, sizeof(name), "color_sites_%d", isector);
    d_color_sites_vec[isector] =
      Kokkos::View<int*>(name, std::max(off[kokkos_ncolors], 1));
    auto h_sites = Kokkos::create_mirror_view(d_color_sites_vec[isector]);
    if (colored) {
      int offset = 0;
      for (int c = 0; c < kokkos_ncolors; c++)
        for (int j = 0; j < colour_set(c).nlocal; j++)
          h_sites(offset++) = colour_set(c).site2i[j];
    } else {
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < nlocal; i++) h_sites(i) = i;
    }
    Kokkos::deep_copy(d_color_sites_vec[isector], h_sites);
  }
}

/* ----------------------------------------------------------------------
   main iteration loop - Kokkos version with color-based sweeps
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::iterate_rejection(double stoptime)
{
  // The device holds the authoritative field data for the whole run. The host
  // copy is refreshed only when something on the host reads it: the ghost
  // exchange (np > 1), app_update, output, and the end of the run. At np = 1
  // that removes a full device->host->device round trip per sector per sweep.
  //
  // Seed the device once here: App::run() calls init() only on the first run
  // (or `pre yes`), so host-side changes between runs (`set`, `run ... pre
  // no`) must be pushed now that there is no per-sector upload.
  // The upload may change the data (potts/quaternion/kk normalises), so the
  // host is stale from here on: the first exchange must ship what the device
  // holds, or other ranks' ghosts differ in the low bits from the owner's.
  timer->stamp();
  sync_app_data_to_device();
  host_data_stale = true;
  timer->stamp(TIME_SOLVE);

  // `sector nstop/tstop` sweeps each colour nloop times per pass; setup()
  // gives every active set the same nloop and folds it into dt_rkmc.
  const int nloop = set[bothflag ? nsector : 0].nloop;

  int done = 0;
  while (!done) {
    // For each sector (or single pass if no sectors)
    int nset_loop = sectorflag ? nsector : 1;

    for (int iset = 0; iset < nset_loop; iset++) {

      // Ghost exchange: owned sites to host, swap, received ghosts (the
      // contiguous range [nlocal,nall)) back to the device.
      if (nprocs > 1) {
        timer->stamp();
        ensure_host_data_current();
        timer->stamp(TIME_SOLVE);

        timer->stamp();
        if (sectorflag) comm->sector(iset);
        else comm->all();
        timer->stamp(TIME_COMM);

        timer->stamp();
        sync_ghosts_to_device();
        timer->stamp(TIME_SOLVE);
      }

      if (Lmask) boundary_clear_mask(iset);

      // Determine which sector's color lists to use
      // When bothflag=1, use iset (the current sector)
      // When bothflag=0, always use sector 0's lists
      int sector_idx = bothflag ? iset : 0;
      const std::vector<int> &h_offsets = color_offsets[sector_idx];
      // Set d_color_sites to the current sector's view - derived classes use this
      d_color_sites = d_color_sites_vec[sector_idx];

      // === CORE: Process each color sequentially ===
      for (int icolor = 0; icolor < kokkos_ncolors; icolor++) {

        // Colors WITHOUT sectors need a ghost refresh BETWEEN colors at np > 1:
        // color c-1 writes owned sites that are color c's neighbors on another
        // rank, and those updates have to cross before c is swept. Without it
        // the ghosts go stale between colors, which silently makes the result
        // decomposition-dependent and defeats `sweep color/strict`. This
        // mirrors AppLattice (nset == ncolors, one comm->all() per set,
        // app_lattice.cpp:660-667). icolor == 0 is covered by the exchange
        // above; sectoring needs only one exchange per sector.
        if (nprocs > 1 && !sectorflag && icolor > 0) {
          timer->stamp();
          ensure_host_data_current();
          timer->stamp(TIME_SOLVE);

          timer->stamp();
          comm->all();
          timer->stamp(TIME_COMM);

          timer->stamp();
          sync_ghosts_to_device();
          timer->stamp(TIME_SOLVE);
        }

        int color_start = h_offsets[icolor];
        int color_count = h_offsets[icolor + 1] - color_start;

        if (color_count > 0) {
          // Derived class implements the actual sweep kernel; fence so each
          // pass sees the previous one's writes
          for (int iloop = 0; iloop < nloop; iloop++) {
            sweep_color_kokkos(icolor, color_start, color_count);
            Kokkos::fence();
          }
          host_data_stale = true;

          // Count the sites this kernel actually attempted. Adding nlocal once
          // per sector instead overcounted by nsector (each sector's lists hold
          // only its share of the sites), inflating Nreject ~nsector-fold.
          nattempt += (bigint) nloop * color_count;
        }
      }

      // Reverse comm writes owned host sites, so the device needs them back.
      // Every Potts app has delevent = 0, which makes it a no-op (skipped).
      if (nprocs > 1 && comm->reverse_needed()) {
        timer->stamp();
        ensure_host_data_current();
        timer->stamp(TIME_SOLVE);

        timer->stamp();
        if (sectorflag) comm->reverse_sector(iset);
        else comm->all_reverse();
        timer->stamp(TIME_COMM);

        timer->stamp();
        sync_app_data_to_device();
        host_data_stale = true;
        timer->stamp(TIME_SOLVE);
      }
    }

    // Time advancement. app_update runs on the host arrays.
    if (allow_app_update) {
      ensure_host_data_current();
      app_update(dt_rkmc);
      sync_app_data_to_device();
      host_data_stale = true;
    }
    nsweeps++;
    time += dt_rkmc;

    if (time >= stoptime) done = 1;
    if (done || time >= nextoutput) {
      timer->stamp();
      ensure_host_data_current();
      timer->stamp(TIME_SOLVE);
      nextoutput = output->compute(time, done);
    }
    timer->stamp(TIME_OUTPUT);
  }

  // Leave the host current for whatever runs after (next command, dumps).
  ensure_host_data_current();
}

/* ----------------------------------------------------------------------
   device -> host copy of the owned field data, only if a sweep ran since
------------------------------------------------------------------------- */

template <class Base>
void LatticeKokkos<Base>::ensure_host_data_current()
{
  if (!host_data_stale) return;
  sync_app_data_to_host();
  host_data_stale = false;
}

}

#endif
