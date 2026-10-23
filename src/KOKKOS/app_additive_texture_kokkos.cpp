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

   Contributing author: AI-assisted implementation (Phase 2: Temperature Integration)

   Kokkos GPU-accelerated version of additive manufacturing application.
   Extends AppPottsQuaternionKokkos with temperature field integration
   for AM simulations with thermal gradient-driven grain growth.

------------------------------------------------------------------------- */

#include "app_additive_texture_kokkos.h"
#include "comm_lattice.h"
#include "domain.h"
#include "error.h"
#include "random_park.h"
#include "timer.h"
#include "math_const.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_moser.h"
#include "temperature_source_conduction.h"
#include "temperature_source_hdf5_unstructured.h"
#include "grain_walk.h"
#include "random_mars.h"
#include "crystalline_orientations/quaternion.h"
#include <Kokkos_Sort.hpp>
#include <type_traits>

// Grain-walk state provider for the device kernels: cached OU recurrence
// (see the cache views in the header); direct recurrence when rho == 0 or
// on cache overflow.
struct GwCached {
  Kokkos::View<int*> gid_of_spin, ngid, nready;
  Kokkos::View<double*[3]> tab;
  int maxgid, maxlev;
  double rms, rho;
  KOKKOS_INLINE_FUNCTION void operator()(int g, int k, double v[3]) const {
    if (rho <= 0.0 || maxgid == 0 || k < 0 || k >= maxlev) { gw_state(g, k, rms, rho, v); return; }
    int gid = Kokkos::atomic_load(&gid_of_spin(g));
    if (gid < 0) {
      int mine = Kokkos::atomic_fetch_add(&ngid(0), 1);
      if (mine >= maxgid) { gw_state(g, k, rms, rho, v); return; }
      int old = Kokkos::atomic_compare_exchange(&gid_of_spin(g), -1, mine);
      gid = (old == -1) ? mine : old;
    }
    if (gid >= maxgid) { gw_state(g, k, rms, rho, v); return; }
    const int base = gid * maxlev;
    int r = Kokkos::atomic_load(&nready(gid));
    if (r > k) { Kokkos::memory_fence(); v[0] = tab(base + k, 0); v[1] = tab(base + k, 1); v[2] = tab(base + k, 2); return; }
    const double sig = rms / 1.7320508075688772, w = sqrt(1.0 - rho * rho) * sig;
    double cur[3]; int j0;
    if (r == 0) {
      gw_random_gauss3(g, 0, cur); cur[0] *= sig; cur[1] *= sig; cur[2] *= sig;
      tab(base, 0) = cur[0]; tab(base, 1) = cur[1]; tab(base, 2) = cur[2]; j0 = 1;
    } else {
      Kokkos::memory_fence(); cur[0] = tab(base + r - 1, 0); cur[1] = tab(base + r - 1, 1); cur[2] = tab(base + r - 1, 2); j0 = r;
    }
    for (int j = j0; j <= k; j++) {
      double rr[3]; gw_random_gauss3(g, j, rr);
      cur[0] = rho * cur[0] + w * rr[0]; cur[1] = rho * cur[1] + w * rr[1]; cur[2] = rho * cur[2] + w * rr[2];
      tab(base + j, 0) = cur[0]; tab(base + j, 1) = cur[1]; tab(base + j, 2) = cur[2];
    }
    Kokkos::memory_fence();
    Kokkos::atomic_fetch_max(&nready(gid), k + 1);
    v[0] = cur[0]; v[1] = cur[1]; v[2] = cur[2];
  }
};
#include <cstring>
#include <limits>

using namespace SPPARKS_NS;
using namespace MathConst;

// Upload a host array into a fresh device View, casting element-wise. A
// zero-length source yields a zero-extent View (legal; never indexed).
template<typename U, typename T>
static Kokkos::View<U*> upload_vec(const char *name, const T *src, size_t n)
{
  Kokkos::View<U*> dv(name, n);
  auto hm = Kokkos::create_mirror_view(dv);
  for (size_t i = 0; i < n; i++) hm(i) = (U) src[i];
  Kokkos::deep_copy(dv, hm);
  return dv;
}
template<typename U, typename T>
static Kokkos::View<U*> upload_vec(const char *name, const std::vector<T> &v)
{
  return upload_vec<U>(name, v.data(), v.size());
}

/* ----------------------------------------------------------------------
   Helper to construct parent class args from user input
------------------------------------------------------------------------- */

static char** construct_parent_args(char **arg) {
  static char* parent_args[3];
  static char app_name[] = "potts/quaternion/kk";  // Use Kokkos quaternion parent
  static char crystal_type[] = "cubic";

  parent_args[0] = app_name;
  parent_args[1] = arg[1]; // nspins
  parent_args[2] = crystal_type;

  return parent_args;
}

/* ----------------------------------------------------------------------
   Constructor
------------------------------------------------------------------------- */

#ifdef KOKKOS_ENABLE_HIP
// SeriesMemory hooks for an integrated GPU: the HDF5 time series lives in pinned
// host memory the GPU reads in place. Non-coherent = coarse-grained, so the GPU
// may cache it; the host writes it once per layer, before any kernel reads it.
static void *series_alloc_pinned(size_t bytes)
{
  void *p = nullptr;
  if (hipHostMalloc(&p, bytes, hipHostMallocNonCoherent) != hipSuccess) return nullptr;
  return p;
}

static void series_release_pinned(void *p)
{
  hipDeviceSynchronize();   // a kernel may still be reading the previous layer
  hipHostFree(p);
}
#endif

/* ---------------------------------------------------------------------- */

AppAdditiveTextureKokkos::AppAdditiveTextureKokkos(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternionKokkos(spk, 3, construct_parent_args(arg))
{
  if (narg != 5)
    error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  spin_lo = 0;           // kernels draw (int)(nspins*drand()); CPU app accepts 0..nspins too
  dx = atof(arg[2]);     // Lattice spacing (meters)
  dt = atof(arg[3]);     // Timestep (seconds)
  nrefine = atoi(arg[4]); // Refinement MC steps after solidification

  // Configure array sizes (inherit from AppPottsQuaternion)
  // Parent AppPottsQuaternionKokkos called recreate_arrays() with ndouble=4
  // We need MORE arrays, so call recreate_arrays() again with larger ndouble
  ndouble = 10;  // q0, qx, qy, qz, mobility_out, T, solid_d, G, V, spath
  ninteger = 2; // spin, active_flag
  recreate_arrays();  // Reallocate iarray/darray pointers for ndouble=10

#ifdef KOKKOS_ENABLE_HIP
  // On an integrated GPU (APU) the HDF5 time series is allocated GPU-readable
  // and read in place (fill_temperature_hdf5_device), so a layer is held once
  // instead of as a host series plus a device copy. FP64 only: the FP32 build
  // narrows the temperatures into a device copy. Must precede the first layer
  // load, i.e. setup_temperature_source.
  if (std::is_same<fastreal, double>::value && !SeriesMemory::alloc && integrated_gpu) {
    SeriesMemory::alloc = series_alloc_pinned;
    SeriesMemory::release = series_release_pinned;
  }
#endif

  host_layout_sites = true;  // front kernels gather whole rows (see app_lattice_kokkos.h)
  allow_app_update = 1;
  app_update_only = 1;  // Skip solid-state growth for now
  allow_kmc = 0;        // all physics is in app_update; no KMC path

  // Initialize pointers
  active_flag = nullptr;
  mobility_out = nullptr;
  T = nullptr;
  solid_d = nullptr;
  G = nullptr;
  V = nullptr;
  nucleation_flags = nullptr;
  nucleation_temps = nullptr;
  nucleation_sizes = nullptr;

  // AdditiveParams holds the defaults shared with AppAdditiveTexture; the
  // nuclei volumes need dx, as there
  t_room = 300.0;  // Room temperature (K)
  size_norm = pow(dx,3) * 2;
  size_sig = pow(dx,3);
  t_cool_max = 100.0;    // placeholder; init_app sets tl - ts like the CPU app

  // Temperature-I/O orchestration (source setup/update, laser scan path,
  // multi-pass pause/reset scheduling, fast-forward predictors, smoothing)
  // lives in the shared ThermalManager, exactly as the CPU AppAdditiveTexture.
  // This Kokkos app populates a ThermalHost snapshot (refresh_thermal_host)
  // the manager reads/writes; the manager fills the host T[] which we then
  // sync to the device d_temp.
  thermal = std::unique_ptr<ThermalManager>(new ThermalManager(spk, &thermal_host_));

  // Initialize powder activation tracking
  last_powder_activation_time = -1.0;  // Force activation at init

  // RNG pool is inherited from AppLatticeKokkos - initialized in parent
}

/* ----------------------------------------------------------------------
   Destructor
------------------------------------------------------------------------- */

AppAdditiveTextureKokkos::~AppAdditiveTextureKokkos()
{
  // Every posted non-blocking request must complete before MPI_Finalize.
  ff_finish_pending();
  delete[] nucleation_flags;
  delete[] nucleation_temps;
  delete[] nucleation_sizes;
}

/* ----------------------------------------------------------------------
   Allocate memory for site arrays
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::grow_app()
{
  // Call parent chain - this sets up spin, quaternions, and neighbor arrays
  // AppPottsQuaternionKokkos::grow_app() -> AppLatticeKokkos::grow_app() -> allocate_app_views()
  AppPottsQuaternionKokkos::grow_app();

  // Set up additional AM pointers from iarray/darray
  active_flag = iarray[1];
  mobility_out = darray[4];
  T = darray[5];
  solid_d = darray[6];
  G = darray[7];
  V = darray[8];
  spath = darray[9];
}

/* ----------------------------------------------------------------------
   Allocate app-specific Kokkos views
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::allocate_app_views()
{
  // Call parent (allocates d_spin, d_q0, d_qx, d_qy, d_qz)
  AppPottsQuaternionKokkos::allocate_app_views();

  int nall = nlocal + nghost;

  // AM-specific device views
  d_active_flag = site_view("active_flag", active_flag, nall);
  d_solid_d = site_view("solid_d", solid_d, nall);

  // Temperature field (device fills, or host fills + upload)
  d_temp = site_view("temperature", T, nall);

  // Gradient cache (device-only, computed once per timestep on device)
  d_G = site_view("gradient_magnitude", G, nall);
  d_V = site_view("solidification_velocity", V, nall);
  d_spath = site_view("spath", spath, nall);
  d_mobility_out = site_view("mobility_out", mobility_out, nall);

  // Global parameter arrays (device-accessible constants); the per-spin
  // nucleation arrays and solid-front coefficients are uploaded at init.
}

/* ----------------------------------------------------------------------
   Initialize application
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::init_app()
{
  // Call parent initialization - sets up color lists, syncs neighbors, etc.
  // (The parent also builds rand_pool; this app's kernels draw from KeyedRNG.)
  AppPottsQuaternionKokkos::init_app();
  // The inherited colour sweep reads d_neighbor, which this app never fills
  // (it keeps the row-major d_neighbor_rm).
  if (!app_update_only)
    error->all(FLERR, "additive/texture/kk requires app_update_only yes");
  check_misorientation_step(dx, error, domain->me);
  check_material_complete(error, *thermal);
  check_sector_stop(error, nstop, tstop);

  // The neighbour-reading kernels colour sites by coordinate parity (see
  // shuffled_colors), which is race-free only if no two owned neighbours share
  // a colour, i.e. on a unit-spaced lattice. Check that directly rather than
  // race silently. (An odd periodic length is already refused by
  // Lattice::ncolors, "Color stencil is incommensurate".)
  {
    auto colour = [this](int i) {
      return (static_cast<int>(xyz[i][0]) & 1) | ((static_cast<int>(xyz[i][1]) & 1) << 1) |
             ((static_cast<int>(xyz[i][2]) & 1) << 2);
    };
    int flag = 0;
    #pragma omp parallel for schedule(static) reduction(|:flag)
    for (int i = 0; i < nlocal; i++)
      for (int j = 0; j < numneigh[i]; j++)
        if (neighbor[i][j] < nlocal && colour(neighbor[i][j]) == colour(i)) flag = 1;
    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
    if (flagall)
      error->all(FLERR, "additive/texture/kk needs a unit-spaced lattice (sc/26n 1.0): "
                 "its kernels colour sites by coordinate parity");
  }

  // Grain-walk recurrence cache (persistent walk only).
  if (grain_walk_D > 0.0 && grain_walk_rho > 0.0 && d_gw_tab.extent(0) == 0) {
    d_gw_gid = Kokkos::View<int*>("gw_gid", nspins + 1); Kokkos::deep_copy(d_gw_gid, -1);
    d_gw_ngid = Kokkos::View<int*>("gw_ngid", 1);
    d_gw_nready = Kokkos::View<int*>("gw_nready", gw_maxgid);
    d_gw_tab = Kokkos::View<double*[3]>("gw_tab", (size_t)gw_maxgid * gw_maxlev);
    if (domain->me == 0 && screen)
      fprintf(screen, "grain-walk cache: %d grains x %d levels (%.0f MB)\n", gw_maxgid, gw_maxlev,
              (double)gw_maxgid * gw_maxlev * 3 * sizeof(double) / 1e6);
  }

  // Timestep guard, same formula as AppAdditiveTexture::init_app (front must
  // not advance more than one voxel per step at the full freezing range).
  // The CPU app aborts; here it is a warning, because the front actually runs
  // well below Tl-Ts undercooling and the capture reach is sqrt(3) dx, and a
  // 2 vs 4 us comparison on the GP700 build agreed within seed noise
  // (docs/GP700_MOSER_SROD_DECK_PLAN.md). The reach-based bound is printed too.
  {
    const double vmax = front_velocity(tl - ts);
    if (vmax > 0.0 && domain->me == 0 && screen) {
      const double dt_cpu = dx / vmax, dt_reach = 1.7320508075688772 * dx / vmax;
      if (dt > dt_cpu)
        fprintf(screen, "WARNING: timestep %g s exceeds the CPU app's front limit dx/V(Tl-Ts) = %g s "
                "(reach-based limit sqrt(3)*dx/V = %g s); the CPU app would abort here\n", dt, dt_cpu, dt_reach);
      else
        fprintf(screen, "Maximum allowable timestep (CPU-app rule dx/V(Tl-Ts)) is %e s\n", dt_cpu);
    }
  }

  // Do NOT initialize T / G / V / active_flag / mobility_out / solid_d here.
  //
  // These are per-site arrays the INPUT DECK owns. init_app() runs from
  // AppLattice::init() (app_lattice.cpp:331), i.e. at `run` time -- AFTER every
  // `set` command has executed. A stitch-continuation deck does
  //
  //     set i1 stitch <file> last
  //     set i2 stitch <file> last      # i2 IS active_flag (iarray[1])
  //     set d1..d4 stitch <file> last
  //
  // to reload the previously built microstructure, so zeroing active_flag here
  // destroyed exactly the state that had just been read: every already-solid
  // site (i2 == 3) became 0, which is inactive/air. Multi-box builds
  // therefore never accumulated -- each box re-solidified its own layer and
  // lost every earlier one (measured: solid fraction flat at ~0.28 across 10
  // boxes where the CPU app climbs to 0.75), and the output carried a ~9.5%
  // i2 == 0 population the CPU app never produces.
  //
  // The CPU app (AppAdditiveTexture::init_app) initializes NONE of these and
  // instead READS active_flag -- `if (active_flag[i] != 3)` -- to decide which
  // sites keep their loaded orientation. Sites the deck never sets are already
  // zeroed at creation (app_lattice.cpp:1299-1300), which is what the CPU path
  // relies on, so matching it needs no replacement initialization.
  //
  // This was invisible to the temperature validation: d6 is refilled from the
  // temperature source every step and does not depend on active_flag.

  // Set timestep for framework (matches CPU version behavior)
  dt_sweep = dt;

  // Initialize nucleation parameters (per-spin arrays and global params).
  // The per-site arrays were uploaded by AppPottsQuaternionKokkos::init_app.
  initialize_nucleation_parameters();

  // Temperature field at the run's start time via the ThermalManager (fills
  // host T[]), as the CPU app's app_update(0.0) does. Not t=0: on a second run
  // that rewound the scan clock, and the next step moved the laser off the path.
  if (thermal->enabled()) {
    refresh_thermal_host();
    // The conduction source has no point query; its first device fill (step
    // 1) builds the grid. Until then host T is the ambient.
    if (ConductionTemperatureSource *c = thermal->conduction_source())
      for (int i = 0; i < nlocal + nghost; i++) T[i] = c->ambient();
    else
      thermal->update_temperatures(time);
  }

  // Note: Powder activation happens after meltpool appears, not during init
}

/* ----------------------------------------------------------------------
   Sync app data from host to device (implements AppLatticeKokkos pure virtual)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_app_data_to_device()
{
  // Call parent (syncs spin, quaternions)
  AppPottsQuaternionKokkos::sync_app_data_to_device();

  // AM arrays over the full owned+ghost range. T is synced separately
  // (sync_temperature_to_device). G and V are written on the device at
  // solidification, but the host copies can hold loaded values (`set d8/d9
  // stitch`): without the upload a stitched box zeroed them for all earlier
  // material it did not remelt.
  const int nall = nlocal + nghost;
  upload_range(d_active_flag, active_flag, 0, nall);
  upload_range(d_solid_d, solid_d, 0, nall);
  upload_range(d_G, G, 0, nall);
  upload_range(d_V, V, 0, nall);
  upload_range(d_spath, spath, 0, nall);
  upload_range(d_mobility_out, mobility_out, 0, nall);
}

/* ----------------------------------------------------------------------
   Sync app data from device to host (implements AppLatticeKokkos pure virtual)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_app_data_to_host()
{
  // Call parent (syncs spin, quaternions)
  AppPottsQuaternionKokkos::sync_app_data_to_host();

  // Owned slice only; host ghosts are refreshed by the ghost exchange.
  // G and V are device-only and come back for the STITCH dumps.
  download_range(active_flag, d_active_flag, nlocal);
  download_range(solid_d, d_solid_d, nlocal);
  download_range(G, d_G, nlocal);
  download_range(V, d_V, nlocal);
  download_range(spath, d_spath, nlocal);
  download_range(mobility_out, d_mobility_out, nlocal);
}

/* ----------------------------------------------------------------------
   Update temperature from external source (CPU operation)
------------------------------------------------------------------------- */

// Populate the ThermalHost snapshot ThermalManager reads/writes. Cheap pointer
// and scalar copies; called before any thermal-> use so the manager sees the
// current lattice state without a typed back-pointer into this app. Mirrors
// AppAdditiveTexture::refresh_thermal_host().
void AppAdditiveTextureKokkos::refresh_thermal_host()
{
  thermal_host_.T = T;
  thermal_host_.xyz = xyz;
  thermal_host_.numneigh = numneigh;
  thermal_host_.neighbor = neighbor;
  thermal_host_.nlocal = nlocal;
  thermal_host_.dx = dx;
  thermal_host_.tl = tl;
  thermal_host_.ts = ts;
  if (!thermal_host_.any_liquid)
    thermal_host_.any_liquid = [this]() {
      auto af = d_active_flag;
      int nl = 0, ng = 0;
      Kokkos::parallel_reduce("AnyLiquid", nlocal,
        KOKKOS_LAMBDA(int i, int &c) { if (af(i) == 2) c++; }, nl);
      // a flag, not a count: a sum over ranks can wrap past 2^31 liquid sites
      int any = nl > 0;
      MPI_Allreduce(&any, &ng, 1, MPI_INT, MPI_MAX, world);
      return ng > 0;
    };
  t_cool_max = tl - ts;  // undercooling normalization for misorient_alpha (matches CPU)
  thermal_host_.t_room = t_room;
  thermal_host_.time = time;
  thermal_host_.dt = dt;
  thermal_host_.comm = comm;
  // The manager's host ghost comm moves T only (darray[5]): its smoothing
  // passes read neighbour T. The device buffers carry the rest.
  static const int T_index[1] = {5};
  thermal_host_.ghost_iindices = nullptr;
  thermal_host_.nghost_iarray = 0;
  thermal_host_.ghost_dindices = T_index;
  thermal_host_.nghost_darray = 1;
}

/* ----------------------------------------------------------------------
   Sync temperature from host to device (after CPU update from HDF5)
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_temperature_to_device()
{
  upload_range(d_temp, T, 0, nlocal + nghost);
}

/* ----------------------------------------------------------------------
   Sync temperature from device to host (after a device-side fill). Copies
   d_temp back into the host T array (darray[5]) so host consumers -- the d6
   diagnostic, activate_powder_sites(), optional smoothing diagnostics -- see
   the current field. Cheap O(nall) copy; the expensive per-site evaluation
   stays on the device.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_temperature_to_host()
{
  download_range(T, d_temp, nlocal + nghost);
  host_T_stale = false;
}

/* ----------------------------------------------------------------------
   Selective ghost sync (Phase 2). Move ONLY the neighbor-read Monte Carlo
   state -- spin (iarray[0]), active_flag (iarray[1]), quaternions q0-3
   (darray[0-3]) -- between device and the raw host arrays, wrapping each raw
   pointer as an unmanaged host View so deep_copy writes straight into the
   comm-visible iarray/darray memory (one contiguous DMA per array, no serial
   loop, no per-call mirror allocation). T is device-resident (no ghost comm)
   and mobility/solid_d/G/V are output-only, so none of those are touched here.

   _to_host copies the OWNED slice (so comm->all_selective sends current values);
   _to_device copies the received GHOST slice back. Owned device state is left
   untouched (the kernels are authoritative for owned). Quaternions are moved
   raw (no renormalization); they stay near-unit through the physics, and the
   output-step full sync renormalizes for dumps.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::sync_ghost_to_host()
{
  download_range(spin, d_spin, nlocal);
  download_range(active_flag, d_active_flag, nlocal);
  download_range(q0, d_q0, nlocal);
  download_range(qx, d_qx, nlocal);
  download_range(qy, d_qy, nlocal);
  download_range(qz, d_qz, nlocal);
  download_range(spath, d_spath, nlocal);
}

void AppAdditiveTextureKokkos::sync_ghost_to_device()
{
  const int nall = nlocal + nghost;
  upload_range(d_spin, spin, nlocal, nall);
  upload_range(d_active_flag, active_flag, nlocal, nall);
  upload_range(d_q0, q0, nlocal, nall);
  upload_range(d_qx, qx, nlocal, nall);
  upload_range(d_qy, qy, nlocal, nall);
  upload_range(d_qz, qz, nlocal, nall);
  upload_range(d_spath, spath, nlocal, nall);
}

/* ----------------------------------------------------------------------
   Global peak temperature of the last device temperature fill. Used by the
   device fast-forward decision (Rosenthal/HDF5) and the Moser pause/reset
   pass firing. Valid only immediately after a fill_temperature_*_device().
------------------------------------------------------------------------- */

double AppAdditiveTextureKokkos::global_max_temperature_from_fill()
{
  // The owned-range Max is computed INSIDE the fill kernels (fused with the
  // d_temp write -- no separate device pass, no extra stream sync); this only
  // reduces the per-rank value across ranks.
  double global_max = fill_maxT_local;
  MPI_Allreduce(&fill_maxT_local, &global_max, 1, MPI_DOUBLE, MPI_MAX, world);
  return global_max;
}

/* ----------------------------------------------------------------------
   One-step-stale peak temperature (HDF5 path, nprocs>1).

   Posts a non-blocking Iallreduce for the CURRENT fill and returns the
   PREVIOUS fill's completed value, so the collective overlaps this step's
   microstructure kernels rather than blocking on the slowest rank inline.

   At nprocs==1 this degenerates to the exact blocking path -- no staleness,
   no MPI -- so single-rank runs keep byte-for-byte the behaviour every GPU
   result to date was validated against.

   Before the first result has landed the reported value is DBL_MAX ("assume
   hot"), which suppresses a fast-forward rather than inventing one. Not
   +infinity: Release builds use -ffast-math, under which infinity is UB.
------------------------------------------------------------------------- */

double AppAdditiveTextureKokkos::global_max_temperature_from_fill_stale()
{
  if (nprocs == 1) return global_max_temperature_from_fill();

  if (ff_ia_active) {
    MPI_Wait(&ff_ia_req, MPI_STATUS_IGNORE);
    ff_ia_active = false;
    ff_gmax_prev = ff_ia_recv;
    ff_gmax_primed = true;
  }

  // Post this fill's reduction; it is harvested on the next call.
  ff_ia_send = fill_maxT_local;
  MPI_Iallreduce(&ff_ia_send, &ff_ia_recv, 1, MPI_DOUBLE, MPI_MAX, world,
                 &ff_ia_req);
  ff_ia_active = true;

  return ff_gmax_primed ? ff_gmax_prev
                        : std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   Complete any outstanding fast-forward Iallreduce. MPI requires every
   posted non-blocking request to be completed before MPI_Finalize; leaving
   one in flight past the end of a run also leaks the request handle.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::ff_finish_pending()
{
  if (!ff_ia_active) return;
  MPI_Wait(&ff_ia_req, MPI_STATUS_IGNORE);
  ff_ia_active = false;
  ff_gmax_prev = ff_ia_recv;
  ff_gmax_primed = true;
}

/* ----------------------------------------------------------------------
   Evaluate the analytical Rosenthal temperature on the device, writing d_temp
   for all nlocal+nghost sites. The laser pose + material parameters come from
   the host-advanced RosenthalDevicePlan; the pool-local transform
   (am_raster.h compute_position_relative_to_pool) and the Rosenthal kernel
   (temperature_source_rosenthal.cpp rosenthal_pointwise/rosenthal_kernel) are
   reproduced here device-side -- keep them in sync.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::ensure_device_xyz()
{
  const int nall = nlocal + nghost;
  if (d_xyz_n == nall) return;

  // LayoutRight matches the host's row-major [nall][3] block (Memory::create).
  if (pinned_sites) {
    d_xyz = Kokkos::View<double*[3], Kokkos::LayoutRight>(xyz[0], nall);
    d_xyz_n = nall;
    return;
  }
  d_xyz = Kokkos::View<double*[3], Kokkos::LayoutRight>("d_xyz", nall);
  if (nall > 0)
    Kokkos::deep_copy(d_xyz,
        Kokkos::View<const double*[3], Kokkos::LayoutRight, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(xyz[0], nall));
  d_xyz_n = nall;
}

// Device reproduction of RosenthalTemperatureSource::rosenthal_kernel:
// Q_eff / (2 pi k R) * exp(-v (xi + R) / (2 alpha)), with the 1/R singularity
// clamped at r_min. (The image-source factor of 2 is applied by the caller.)
template<typename Real>
KOKKOS_INLINE_FUNCTION
static Real rosenthal_device_kernel(Real Qeff, Real v, Real xi, Real R,
                                    Real r_min, Real k_cond, Real alpha,
                                    Real two_pi)
{
  Real Rk = (R < r_min) ? r_min : R;
  if (Rk <= (Real)0) return (Real)0;
  return (Qeff / (two_pi * k_cond * Rk)) *
         Kokkos::exp(-v * (xi + Rk) / ((Real)2 * alpha));
}

void AppAdditiveTextureKokkos::fill_temperature_rosenthal_device(
    const RosenthalDevicePlan &plan)
{
  const int nall = nlocal + nghost;

  ensure_device_xyz();

  // KEYHOLE: upload the (fixed) line-integral quadrature once.
  if (plan.mode == 2 && d_ros_nquad != plan.n_quad) {
    std::vector<double> nodes, weights;
    thermal->rosenthal_source()->copy_keyhole_quadrature(nodes, weights);
    d_ros_qnodes = upload_vec<double>("ros_qnodes", nodes);
    d_ros_qweights = upload_vec<double>("ros_qweights", weights);
    d_ros_nquad = plan.n_quad;
  }

  auto d_T   = d_temp;
  auto d_xyz_local = d_xyz;
  auto qnodes = d_ros_qnodes;
  auto qweights = d_ros_qweights;

  // The plan is a trivially-copyable POD: capture it by value.
  const RosenthalDevicePlan p = plan;
  const double dx_l = this->dx;
  const double two_pi = 2.0 * 3.14159265358979323846;
  const int nlocal_l = this->nlocal;

  // Fused owned-range Max: the fast-forward decision needs the peak owned
  // temperature every step, and this kernel already has each T in a register.
  double local_max = 0.0;
  Kokkos::parallel_reduce("RosenthalFill", nall, KOKKOS_LAMBDA(const int i, double &m) {
    // Site physical coordinates (meters). The site-minus-laser differences are
    // computed in double: (site - laser) is cancellation-prone (both operands
    // are ~domain-scale) and cheap, so it stays FP64 regardless of fastreal.
    const double x = d_xyz_local(i,0) * dx_l;
    const double y = d_xyz_local(i,1) * dx_l;
    const double z = d_xyz_local(i,2) * dx_l;

    // Pool-local frame: project (site - laser) onto scan / transverse axes.
    const double rsx0 = x - p.px;
    const double rsx1 = y - p.py;
    const double z_rel_d = z - p.layer_z;
    const double xi_d = p.ux*rsx0 + p.uy*rsx1;          // along scan direction
    const double tsx0 = rsx0 - xi_d*p.ux;
    const double tsx1 = rsx1 - xi_d*p.uy;
    const double y_rel_d = -p.uy*tsx0 + p.ux*tsx1;      // transverse

    double Tval = p.T0;
    // Free-surface cutoff: vacuum above the laser plane. Inactive path -> p.T0.
    if (p.scan_active && z_rel_d <= 0.0) {
      // Cast the (small, pool-local) coordinates and material params to fastreal.
      // The transcendental kernel math below runs in fastreal -- FP32 when built
      // -DSPPARKS_FP32, otherwise identical FP64 to the original path.
      const fastreal xi = (fastreal)xi_d;
      const fastreal y_rel = (fastreal)y_rel_d;
      const fastreal z_rel = (fastreal)z_rel_d;
      const fastreal rv = (fastreal)p.v, rr_min = (fastreal)p.r_min;
      const fastreal rk_cond = (fastreal)p.k_cond, ralpha = (fastreal)p.alpha;
      const fastreal rtwo_pi = (fastreal)two_pi;
      if (p.mode == 2) {
        // KEYHOLE: image-doubled surface point source + buried line source
        // integrated over the Gauss-Legendre nodes (image-doubled per node).
        const fastreal R = Kokkos::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
        fastreal T_rise = (fastreal)2 * rosenthal_device_kernel<fastreal>(
                            (fastreal)p.lambda_pQ, rv, xi, R, rr_min, rk_cond, ralpha, rtwo_pi);
        for (int q = 0; q < p.n_quad; q++) {
          const fastreal zarg = (fastreal)qnodes(q) + z_rel;
          const fastreal Rp = Kokkos::sqrt(xi*xi + y_rel*y_rel + zarg*zarg);
          T_rise += (fastreal)2 * (fastreal)qweights(q) *
                    rosenthal_device_kernel<fastreal>(
                        (fastreal)p.lambda_lQ, rv, xi, Rp, rr_min, rk_cond, ralpha, rtwo_pi);
        }
        Tval = p.T0 + (double)T_rise;
      } else {
        fastreal R;
        if (p.mode == 1) {                            // ANISOTROPIC
          const fastreal yy = (fastreal)p.eta_y * y_rel;
          const fastreal zz = (fastreal)p.eta_z * z_rel;
          R = Kokkos::sqrt(xi*xi + yy*yy + zz*zz);
        } else {                                    // STANDARD
          R = Kokkos::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
        }
        Tval = p.T0 + 2.0 * (double)rosenthal_device_kernel<fastreal>(
                              (fastreal)p.lambdaQ, rv, xi, R, rr_min, rk_cond, ralpha, rtwo_pi);
      }
    }
    d_T(i) = Tval;
    if (i < nlocal_l && Tval > m) m = Tval;
  }, Kokkos::Max<double>(local_max));
  fill_maxT_local = (nlocal > 0) ? local_max : 0.0;
  Kokkos::fence();
}

/* ----------------------------------------------------------------------
   Build and upload the Gauss-Legendre roots/weights table for orders
   2..MOSER_GL_MAX_ORDER. Order N occupies N entries starting at offset(N).
   Roots/weights are generated by the same Newton solve on the Legendre roots
   used by GREENAM GaussLegendreQuadrature, so the device quadrature matches the
   host integrator. Order 1 is the midpoint rule (handled in the kernel).
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::build_gl_table()
{
  if (d_gl_built) return;
  const int maxN = MOSER_GL_MAX_ORDER;

  std::vector<int> offset(maxN + 1, 0);
  std::vector<double> roots, weights;
  for (int N = 2; N <= maxN; N++) {
    offset[N] = (int)roots.size();
    for (int i = 1; i <= N; i++) {
      // Initial guess for the i-th root, then Newton-refine on P_N.
      double x = std::cos(M_PI * (i - 0.25) / (N + 0.5));
      double d = 0.0;
      for (int it = 0; it < 100; it++) {
        // Evaluate P_N(x) value (v) and derivative (d) by recurrence.
        double vsub1 = x, vsub2 = 1.0, v = 1.0;
        const double f = 1.0 / (x * x - 1.0);
        for (int kk = 2; kk <= N; kk++) {
          v = ((2 * kk - 1) * x * vsub1 - (kk - 1) * vsub2) / kk;
          d = kk * f * (x * v - vsub1);
          vsub2 = vsub1;
          vsub1 = v;
        }
        const double dr = v / d;
        x -= dr;
        if (std::fabs(dr) <= 2e-16) break;
      }
      roots.push_back(x);
      weights.push_back(2.0 / ((1.0 - x * x) * d * d));
    }
  }

  d_gl_offset  = upload_vec<int>("gl_offset", offset);
  d_gl_roots   = upload_vec<double>("gl_roots", roots);
  d_gl_weights = upload_vec<double>("gl_weights", weights);
  d_gl_built = true;
}

/* ----------------------------------------------------------------------
   Device-side Moser temperature. Each site integrates the Green's-function
   over the scan-path history, reproducing GREENAM
   ScanIntegration::integrate_point_adaptive() + GaussGreenEllips (STANDARD
   single lobe, no fluctuations). The host advances pass scheduling and hands us
   scalar params (plan) + the scan segment arrays (uploaded on version change).
------------------------------------------------------------------------- */

// Device reproduction of FluctuationTable::get_at (binary search + clamped
// linear interpolation over the [foff, foff+fn) slice). fn <= 0 -> factors = 1.
KOKKOS_INLINE_FUNCTION
static void moser_fluct_get_at(
    const Kokkos::View<double*> &ft, const Kokkos::View<double*> &fw,
    const Kokkos::View<double*> &fd, const Kokkos::View<double*> &fp,
    int foff, int fn, double s, double &fW, double &fD, double &fP)
{
  if (fn <= 0)            { fW = fD = fP = 1.0; return; }
  if (s <= ft(foff))      { fW = fw(foff); fD = fd(foff); fP = fp(foff); return; }
  const int last = foff + fn - 1;
  if (s >= ft(last))      { fW = fw(last); fD = fd(last); fP = fp(last); return; }
  int lo = 0, hi = fn - 1;
  while (lo + 1 < hi) { int mid = (lo + hi) / 2;
    if (ft(foff + mid) <= s) lo = mid; else hi = mid; }
  const double t0 = ft(foff + lo), t1 = ft(foff + lo + 1);
  const double denom = t1 - t0;
  const double frac = (denom > 0.0) ? (s - t0) / denom : 0.0;
  const double om = 1.0 - frac;
  fW = om * fw(foff + lo) + frac * fw(foff + lo + 1);
  fD = om * fd(foff + lo) + frac * fd(foff + lo + 1);
  fP = om * fp(foff + lo) + frac * fp(foff + lo + 1);
}

void AppAdditiveTextureKokkos::fill_temperature_moser_device(
    const MoserDevicePlan &plan)
{
  const int nall = nlocal + nghost;
  ensure_device_xyz();
  build_gl_table();

  // (Re)upload the shared geometry, per-lobe powers, and per-lobe fluctuation
  // tables when the scan generation changes.
  if (d_seg_version != plan.version) {
    const MoserTemperatureSource::DeviceArrays a = thermal->moser_source()->copy_device_data();
    d_seg_bounds = upload_vec<double>("seg_bounds", a.bounds);
    d_seg_xl     = upload_vec<double>("seg_xl", a.xl);
    d_seg_yl     = upload_vec<double>("seg_yl", a.yl);
    d_seg_nx     = upload_vec<double>("seg_nx", a.normals_x);
    d_seg_ny     = upload_vec<double>("seg_ny", a.normals_y);
    d_seg_speeds = upload_vec<double>("seg_speeds", a.speeds);
    d_seg_powers = upload_vec<double>("seg_powers", a.powers);
    d_fluct_t    = upload_vec<double>("fluct_t", a.fluct_t);
    d_fluct_w    = upload_vec<double>("fluct_w", a.fluct_w);
    d_fluct_d    = upload_vec<double>("fluct_d", a.fluct_d);
    d_fluct_p    = upload_vec<double>("fluct_p", a.fluct_p);
    d_seg_version = plan.version;
  }

  auto d_T = d_temp;
  auto d_xyz_local = d_xyz;
  auto bounds = d_seg_bounds; auto xlv = d_seg_xl; auto ylv = d_seg_yl;
  auto powers = d_seg_powers;  auto nxv = d_seg_nx;  auto nyv = d_seg_ny;
  auto speeds = d_seg_speeds;
  auto ftv = d_fluct_t; auto fwv = d_fluct_w; auto fdv = d_fluct_d; auto fpv = d_fluct_p;
  auto gl_roots = d_gl_roots; auto gl_weights = d_gl_weights; auto gl_off = d_gl_offset;

  // The plan is a trivially-copyable POD: capture it by value. p.zl[0] (top
  // lobe, z_offset 0) is the real free surface.
  const MoserDevicePlan p = plan;
  const double dx_l = this->dx;
  const int maxN = MOSER_GL_MAX_ORDER;
  const double eps = 2.220446049250313e-16;
  const double invPi3_2 = 1.0 / (M_PI * 1.7724538509055160273); // 1/(pi*sqrt(pi))
  const int nlocal_l = this->nlocal;

  // Fused owned-range Max: feeds the fast-forward AND the Moser pause/reset
  // firing decisions, so every exit path must contribute its written T.
  double local_max = 0.0;
  Kokkos::parallel_reduce("MoserFill", nall, KOKKOS_LAMBDA(const int site, double &m) {
    const double x = d_xyz_local(site,0) * dx_l;
    const double y = d_xyz_local(site,1) * dx_l;
    const double z = d_xyz_local(site,2) * dx_l;

    // Vacuum above the real free surface; integral is empty before t=0.
    if (z > p.zl[0] || p.time <= 0.0 || p.nseg <= 0) {
      d_T(site) = p.T0;
      if (site < nlocal_l && p.T0 > m) m = p.T0;
      return;
    }

    // binary_search(bounds, p.time, p.nseg): largest i with bounds(i) <= p.time.
    int end_idx = -1;
    { int L = 0, R = p.nseg - 1;
      while (L <= R) { int m = (L + R) / 2;
        if (bounds(m) <= p.time) {
          if (m + 1 == p.nseg || bounds(m+1) >= p.time) { end_idx = m; break; }
          L = m + 1;
        } else R = m - 1; } }
    if (end_idx < 0) {
      d_T(site) = p.T0;
      if (site < nlocal_l && p.T0 > m) m = p.T0;
      return;
    }
    if (Kokkos::fabs(p.time - bounds(end_idx)) <= 4.0 * eps) end_idx--;

    // KEYHOLE sums the contributions of both lobes; STANDARD has one.
    double res = 0.0;
    for (int lobe = 0; lobe < p.n_lobes; lobe++) {
      const double sx = p.sx[lobe], sy = p.sy[lobe], sz = p.sz[lobe], zl = p.zl[lobe];
      const int fn = p.fluct_n[lobe];
      const int foff = (lobe == 0) ? 0 : p.fluct_n[0];   // fluct slice for this lobe
      const int poff = lobe * p.nseg;               // power slice for this lobe
      const double avg_sigma = (sx + sy + sz) / 3.0;
      const double avg_sigma_sq = avg_sigma * avg_sigma;

      for (int i = 0; i <= end_idx; i++) {
        const double p_i = powers(poff + i);
        if (Kokkos::fabs(p_i) <= 4.0 * eps) continue;
        const double xl_i = xlv(i), yl_i = ylv(i);
        const double nx_i = nxv(i), ny_i = nyv(i);
        const double sp_i = speeds(i), t0_i = bounds(i);

        const double tstart = bounds(i);
        const double tend = (i + 1 >= p.nseg) ? p.time
            : (p.time <= bounds(i+1) + 4.0*eps ? p.time : bounds(i+1));

        const double sigma_eff = 4.0*p.alpha*(p.time - tend) + avg_sigma_sq;
        double quad_size = p.char_length*p.char_length*sigma_eff/4.0/p.alpha;
        if (sp_i > 4.0*eps)
          quad_size = Kokkos::fmin(p.char_length*Kokkos::sqrt(sigma_eff)/sp_i, quad_size);

        const double n_quad_pts = (tend - tstart)/quad_size;
        int order = (int)Kokkos::ceil(n_quad_pts); if (order < 1) order = 1;
        if (order > maxN) order = maxN;
        const double step_size = quad_size*order;
        double n_int = Kokkos::ceil((tend - tstart)/step_size);
        if (n_int < 1.0) n_int = 1.0;
        const int nint = (int)n_int;
        const double dt = (tend - tstart)/n_int;

        for (int j = 0; j < nint; j++) {
          const double a = tstart + j*dt;
          const double bb = tstart + (j+1)*dt;
          const double pmid = 0.5*(bb - a), qmid = 0.5*(bb + a);
          // node count: 1 -> midpoint rule, else `order` Gauss-Legendre nodes.
          const int nnodes = (order == 1) ? 1 : order;
          const int goff = (order == 1) ? 0 : gl_off(order);
          double sum = 0.0;
          for (int nn = 0; nn < nnodes; nn++) {
            const double s = (order == 1) ? qmid : (pmid*gl_roots(goff+nn) + qmid);
            const double w = (order == 1) ? (bb - a) : gl_weights(goff+nn);
            // Fluctuation modulation of (sx,sy), sz, power at this emission time.
            double fW = 1.0, fD = 1.0, fP = 1.0;
            if (fn > 0) moser_fluct_get_at(ftv, fwv, fdv, fpv, foff, fn, s, fW, fD, fP);
            // Cancellation-prone time deltas / geometry differences stay double.
            const double st_d = 4.0*p.alpha*(p.time - s);
            const double cx = x - xl_i, cy = y - yl_i;
            const double xt = cx*nx_i + cy*ny_i;
            const double yt = -cx*ny_i + cy*nx_i;
            const double zt = z - zl;
            const double xeff_d = xt - sp_i*(s - t0_i);
            // Per-node Gaussian in fastreal (float exp/sqrt/reciprocals when built
            // -DSPPARKS_FP32; identical FP64 to the original otherwise). The
            // differenced coordinates above are already small, so the cast is safe.
            const fastreal sxu = (fastreal)sx*(fastreal)fW, syu = (fastreal)sy*(fastreal)fW;
            const fastreal szu = (fastreal)sz*(fastreal)fD, pu = (fastreal)p_i*(fastreal)fP;
            const fastreal st = (fastreal)st_d;
            const fastreal xeff = (fastreal)xeff_d, ytr = (fastreal)yt, ztr = (fastreal)zt;
            const fastreal vx = (fastreal)1/(sxu*sxu+st), vy = (fastreal)1/(syu*syu+st),
                         vz = (fastreal)1/(szu*szu+st);
            const fastreal e = Kokkos::exp(-xeff*xeff*vx - ytr*ytr*vy - ztr*ztr*vz);
            const fastreal term = ((fastreal)2*pu/(fastreal)p.c*(fastreal)invPi3_2 * e *
                                 Kokkos::sqrt(vx*vy*vz)) * (fastreal)w;
            sum += (double)term;
          }
          // GL needs the *pmid scale; the midpoint rule already used (bb-a) as w.
          res += (order == 1) ? sum : (pmid * sum);
        }
      }
    }
    const double Tval = p.T0 + res;
    d_T(site) = Tval;
    if (site < nlocal_l && Tval > m) m = Tval;
  }, Kokkos::Max<double>(local_max));
  fill_maxT_local = (nlocal > 0) ? local_max : 0.0;
  Kokkos::fence();
}

/* ----------------------------------------------------------------------
   Device-side HDF5 unstructured-mesh temperature. The host has already done the
   expensive work in thermal->hdf5_device_plan(): loaded the layer, (re)built the
   per-site element cache at a layer boundary, and precomputed the per-node
   temperature slice for this time (cached_nodal_temps; time interpolation +
   optional nodal mesh smoothing). Here we just upload that data and do the
   per-site barycentric interpolation on the GPU, reproducing
   HDF5UnstructuredTemperatureSource::get_temperature_at_site_fast() in the same
   operation order so the device field matches the host fast path bit-for-bit.

   Fills ALL nall sites (owned + ghost) POSITIONALLY, like the Rosenthal/Moser
   device fills. The element cache is built over nall (the chunk-load pad covers
   ghosts), so every ghost has a valid entry and computes its own T from this
   rank's haloed mesh. A ghost equals the owning rank's value by construction
   (same global field, same physical point), so multi-rank needs no T ghost comm
   and no host round-trip -- the field is fully device-resident. See hdf5_device.h.
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::fill_temperature_hdf5_device(
    const HDF5DevicePlan &plan)
{
  if (!plan.valid) return;
  const int n = plan.nsites;   // nall (owned + ghost); fills positionally
  const int nnodes = plan.n_nodes;

  // (Re)upload the STATIC per-site element cache only when the host rebuilds it
  // (layer boundary -> cache generation bumps) or the site count changes.
  if (d_hdf5_cache_version != plan.version || d_hdf5_nsites != n) {
    const HDF5UnstructuredTemperatureSource *src = thermal->hdf5_source();
    const auto noinit = Kokkos::WithoutInitializing;
    d_hdf5_node_idx = Kokkos::View<int*[4]>(Kokkos::view_alloc(noinit, "hdf5_node_idx"), n);
    d_hdf5_weights  = Kokkos::View<fastreal*[3]>(Kokkos::view_alloc(noinit, "hdf5_weights"), n);
    d_hdf5_valid    = Kokkos::View<int*>(Kokkos::view_alloc(noinit, "hdf5_valid"), n);
    auto h_idx = Kokkos::create_mirror_view(noinit, d_hdf5_node_idx);
    auto h_w   = Kokkos::create_mirror_view(noinit, d_hdf5_weights);
    auto h_v   = Kokkos::create_mirror_view(noinit, d_hdf5_valid);
    // Every entry is written: an invalid site (no containing element) gets
    // zero indices/weights and valid 0, so the kernel yields default_temp.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      const auto &c = src->element_cache(i);
      h_v(i) = c.valid ? 1 : 0;
      for (int j = 0; j < 4; j++) h_idx(i, j) = c.valid ? (int) c.nodeIndices[j] : 0;
      // Narrowed to fastreal under FP32 (weights are barycentric, in [0,1]);
      // a no-op cast in the default FP64 build.
      for (int j = 0; j < 3; j++) h_w(i, j) = c.valid ? (fastreal) c.weights[j] : (fastreal) 0;
    }
    Kokkos::deep_copy(d_hdf5_node_idx, h_idx);
    Kokkos::deep_copy(d_hdf5_weights, h_w);
    Kokkos::deep_copy(d_hdf5_valid, h_v);
    d_hdf5_cache_version = plan.version;
    d_hdf5_nsites = n;
  }

  // Fill the per-node temperature slice d_hdf5_nodal_temps that the blend reads.
  // PRIMARY (nodal_smooth OFF): the device interpolates it itself from the raw
  // per-node time series -- skipping the host precompute, the per-step host->
  // device slice copy, and (its dominant cost) the host time interpolation. This
  // is the lever for the host-bound GPU. FALLBACK (nodal_smooth ON): the host
  // precompute already filled the slice (the device doesn't reproduce the mesh
  // smoothing); upload it per step as before.
  const HDF5UnstructuredTemperatureSource *hdf5 = thermal->hdf5_source();
  const bool dti = hdf5->device_time_interp_active();
  if ((int) d_hdf5_nodal_temps.extent(0) != nnodes)
    d_hdf5_nodal_temps = Kokkos::View<fastreal*>("hdf5_nodal_temps", nnodes);

  if (dti) {
    // (Re)upload the time series ONCE per layer, compacted to the active nodes
    // (the tet nodes of valid sites, ~33k): the only rows the blend reads, and
    // the same subset the host precompute interpolates. Row k of the device CSR
    // is node an[k]. Uploading every loaded node instead (pad halo included)
    // cost a full extra copy of the layer's series, plus a same-size staging
    // mirror; on an APU (gfx1151, MI300A) both come out of host memory.
    if (d_ts_version != plan.version) {
      const auto &ts_t = hdf5->timeseries_times();
      const auto &ts_T = hdf5->timeseries_temps();
      const std::vector<size_t> &ts_o = hdf5->timeseries_offsets();
      const std::vector<unsigned> &an = hdf5->timeseries_active_nodes();
      // Free the previous layer's copy before allocating this one.
      d_ts_times = {}; d_ts_temps = {}; d_ts_offsets = {}; d_active_nodes = {};
      if (!ts_t.empty() && !an.empty()) {
        const size_t nact = an.size();
        // Row k = [start, end) of node an[k]. 64-bit: a large layer can exceed
        // 2^31 samples on one rank.
        Kokkos::View<int64_t*[2]> o("ts_offsets", nact);
        auto ho = Kokkos::create_mirror_view(o);
        bool in_place = false;
#ifdef KOKKOS_ENABLE_HIP
#ifndef SPPARKS_FP32
        // In-place needs the series in fastreal; a View<float> cannot wrap a double
        // array, and if constexpr would not discard it outside a template.
        {
          if (SeriesMemory::alloc == series_alloc_pinned) {
            // In place: rows index the host series directly, no copy.
            for (size_t k = 0; k < nact; k++) {
              ho(k, 0) = (int64_t) ts_o[an[k]];
              ho(k, 1) = (int64_t) ts_o[an[k] + 1];
            }
            Kokkos::deep_copy(o, ho);
            d_ts_offsets = o;
            d_ts_times = Kokkos::View<double*>(const_cast<double*>(ts_t.data()), ts_t.size());
            d_ts_temps = Kokkos::View<fastreal*>(const_cast<double*>(ts_T.data()), ts_T.size());
            in_place = true;
          }
        }
#endif
#endif
        if (!in_place) {
        // Compacted copy of the active rows.
        int64_t pos = 0;
        for (size_t k = 0; k < nact; k++) {
          ho(k, 0) = pos;
          pos += (int64_t) (ts_o[an[k] + 1] - ts_o[an[k]]);
          ho(k, 1) = pos;
        }
        const size_t nsamp = (size_t) pos;
        // Times ALWAYS stay FP64 (absolute sim times ~3.9e4 s on a 3e-6 s step)
        // -- see fastreal in spktype.h. Only the temperatures are narrowable.
        Kokkos::View<double*> tv("ts_times", nsamp);
        Kokkos::View<fastreal*> Tv("ts_temps", nsamp);
        auto ht = Kokkos::create_mirror_view(tv);
        auto hT = Kokkos::create_mirror_view(Tv);
        for (size_t k = 0; k < nact; k++) {
          const size_t src = ts_o[an[k]], dst = (size_t) ho(k, 0);
          const size_t n = (size_t) (ho(k, 1) - ho(k, 0));
          for (size_t j = 0; j < n; j++) {
            ht(dst + j) = ts_t[src + j];
            hT(dst + j) = (fastreal) ts_T[src + j];
          }
        }
        Kokkos::deep_copy(o, ho);
        Kokkos::deep_copy(tv, ht);
        Kokkos::deep_copy(Tv, hT);
        d_ts_offsets = o; d_ts_times = tv; d_ts_temps = Tv;
        }

        d_n_active_nodes = (int) nact;
        d_active_nodes = upload_vec<int>("active_nodes", an);

        d_ts_version = plan.version;
      } else {
        // This layer has no thermal data on this rank (load_layer found no
        // overlapping chunks and cleared the layer state). Drop the previous
        // layer's active-node list: it indexes a time series that is gone, and
        // d_hdf5_nodal_temps has just been resized to n_nodes == 0, so leaving
        // the count set would run ComputeNodalTemps off the end of both.
        d_n_active_nodes = 0;
        d_ts_version = plan.version;
      }
    }

    // Device per-node time interpolation -> d_hdf5_nodal_temps, active nodes only
    // (~n_active searches, NOT 4*nall). Bit-matches the host precompute
    // (lower_bound bracket + linear interp); inactive entries are never read.
    // Skipped entirely for an empty layer -- HDF5Fill below then sees valid==0
    // at every site and writes default_temp, which is the right answer.
    if (d_n_active_nodes > 0) {
    auto nt_out = d_hdf5_nodal_temps;
    auto d_act_nodes = d_active_nodes;
    auto tst = d_ts_times; auto tsT = d_ts_temps; auto tso = d_ts_offsets;
    const double defn = plan.default_temp; const double simtn = plan.sim_time;
    Kokkos::parallel_for("ComputeNodalTemps", d_n_active_nodes, KOKKOS_LAMBDA(const int k) {
      const int nd = d_act_nodes(k);
      const int64_t start = tso(k, 0);
      const int count = (int) (tso(k, 1) - start);
      const double tfirst = tst(start), tlast = tst(start + count - 1);
      if (simtn < tfirst || simtn > tlast) { nt_out(nd) = (fastreal) defn; return; }
      int lo = 0, hi = count;                  // first idx with tst >= simtn
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (tst(start + mid) < simtn) lo = mid + 1; else hi = mid;
      }
      int idx = lo; if (idx == 0) idx = 1;
      // Bracket + the (simtn - ti0) / (ti1 - ti0) ratio stay FP64: only the
      // temperature endpoints narrow. Under FP64 this is the identical
      // expression to before; under FP32 the time math is still exact and only
      // the interpolated result is stored narrowed.
      const double ti0 = tst(start + idx - 1), ti1 = tst(start + idx);
      const fastreal te0 = tsT(start + idx - 1), te1 = tsT(start + idx);
      nt_out(nd) = te0 + (te1 - te0) / (ti1 - ti0) * (simtn - ti0);
    });
    }
  } else if (nnodes > 0) {
    // Fallback: the host precompute filled the slice; upload it per step.
    const std::vector<double> &nt = hdf5->nodal_temps();
    d_hdf5_nodal_temps = upload_vec<fastreal>("hdf5_nodal_temps", nt.data(), nnodes);
  }

  auto d_T = d_temp;
  auto nidx = d_hdf5_node_idx;
  auto wts  = d_hdf5_weights;
  auto vld  = d_hdf5_valid;
  auto nt   = d_hdf5_nodal_temps;
  const double def  = plan.default_temp;

  // Full fusion: evaluate the active-front predicate here (T is in a register;
  // the fill already visits every site) and ATOMIC-APPEND active owned sites
  // straight into d_active_sites, so there is no separate compaction scan at all.
  if ((int) d_active_sites.extent(0) != nlocal)
    d_active_sites = Kokkos::View<int*>("active_sites", nlocal);
  if (d_active_count.data() == nullptr)
    d_active_count = Kokkos::View<int>("active_count");
  Kokkos::deep_copy(d_active_count, 0);
  auto d_asites = d_active_sites;
  auto d_acount = d_active_count;
  auto d_af = d_active_flag;
  auto d_sd = d_solid_d;
  const double tl_f = this->tl;
  const int nrefine_f = this->nrefine;
  const int nlocal_f = this->nlocal;

  // Fused owned-range Max for the fast-forward decision (T is in a register
  // here; this replaced a separate per-step Tmax pass over nlocal).
  double local_max = 0.0;
  Kokkos::parallel_reduce("HDF5Fill", n, KOKKOS_LAMBDA(const int i, double &m) {
    // The gather-and-blend runs in fastreal. Under FP64 this is the identical
    // expression (and the identical operation order) the host fast path is
    // validated against; under FP32 the four nodal reads and three weights are
    // half-width, which is the whole point -- this kernel streams them over
    // nall sites every step. The RESULT is always stored to the FP64 d_temp field.
    fastreal Ti;
    if (!vld(i)) {
      Ti = (fastreal) def;
    } else {
      const int i0 = nidx(i, 0), i1 = nidx(i, 1), i2 = nidx(i, 2), i3 = nidx(i, 3);
      const fastreal t0 = nt(i0);
      Ti = t0 + wts(i, 0) * (nt(i1) - t0)
              + wts(i, 1) * (nt(i2) - t0)
              + wts(i, 2) * (nt(i3) - t0);
    }
    d_T(i) = Ti;
    if (i < nlocal_f) {
      if ((double) Ti > m) m = (double) Ti;
      const int af = d_af(i);
      const double sd = d_sd(i);
      // i2 == 5 (void) never enters the list, so it never melts or solidifies
      if (af != 5 && (af == 2 || Ti >= tl_f ||
          (sd > -nrefine_f - 1 && sd < 0.0) ||
          (af == 3 && sd == -nrefine_f - 2))) {
        const int slot = Kokkos::atomic_fetch_add(&d_acount(), 1);
        d_asites(slot) = i;
      }
    }
  }, Kokkos::Max<double>(local_max));
  fill_maxT_local = (nlocal > 0) ? local_max : 0.0;
  Kokkos::fence();
  device_active_list_fresh = true;
}

/* ----------------------------------------------------------------------
   Initialize nucleation parameters on device
   Phase 3.1: Port per-spin nucleation arrays from CPU to device
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::initialize_nucleation_parameters()
{
  rng_seed = (uint64_t) ranmaster->initial_seed();   // KeyedRNG and the per-spin tables
  // Allocate host temp arrays for nucleation parameters
  delete[] nucleation_flags;
  delete[] nucleation_temps;
  delete[] nucleation_sizes;
  nucleation_flags = new int[nspins];
  nucleation_temps = new double[nspins];
  nucleation_sizes = new double[nspins];

  // Initialize nucleation_flags (which spins can nucleate)
  // Nucleation fraction = volume of voxel * nucleation density (grains/m^3)
  double nucleationFraction = dx * dx * dx * no;

  if (nucleationFraction >= 1.0 && domain->me == 0) {
    fprintf(screen, "Nucleation fraction is %g (>= 1). All spins can nucleate.\n", nucleationFraction);
  }

  // Rank 0 draws the tables and broadcasts them, as the CPU app does: drawn
  // on every rank, a spin would nucleate differently on each.
  if (domain->me == 0) {
    // Assign nucleation flags randomly based on nucleation density
    for (int i = 0; i < nspins; i++) {
      if (nucleationFraction >= 1.0) {
        nucleation_flags[i] = 1;  // All spins can nucleate
      } else {
        // keyed by spin with the CPU app's key: the same spins nucleate in both
        quaternion::SiteOrientationRNG r(rng_seed ^ 0x6E75636C65666C67ULL, (uint64_t) i);
        nucleation_flags[i] = (r.uniform() <= nucleationFraction) ? 1 : 0;
      }
    }

    // Critical undercooling (K) and nucleus volume (m^3) per spin: normals keyed
    // by spin (was std::random_device, different every run), the CPU app's
    // keys and Box-Muller pair, so both apps draw the same tables
    for (int i = 0; i < nspins; i++) {
      quaternion::SiteOrientationRNG r(rng_seed ^ 0x6E75636C74656D70ULL, (uint64_t) i);
      const double u1 = r.uniform(), u2 = r.uniform();
      const double rad = sqrt(-2.0 * log(1.0 - u1));
      nucleation_temps[i] = tc + tsig * rad * cos(2.0 * MathConst::MY_PI * u2);
      nucleation_sizes[i] = size_norm + size_sig * rad * sin(2.0 * MathConst::MY_PI * u2);
    }
  }
  MPI_Bcast(nucleation_flags, nspins, MPI_INT, 0, world);
  MPI_Bcast(nucleation_temps, nspins, MPI_DOUBLE, 0, world);
  MPI_Bcast(nucleation_sizes, nspins, MPI_DOUBLE, 0, world);

  d_nucleation_flags = upload_vec<int>("nucleation_flags", nucleation_flags, nspins);
  d_nucleation_temps = upload_vec<double>("nucleation_temps", nucleation_temps, nspins);
  d_nucleation_sizes = upload_vec<double>("nucleation_sizes", nucleation_sizes, nspins);
  d_solid_front_coeffs =
      upload_vec<double>("solid_front_coeffs", solid_front_coeffs, solid_front_length);

}

/* ----------------------------------------------------------------------
   Melting kernel: Transition sites to liquid state when T >= liquidus
   Phase 3.1: Simplest phase transition kernel (no neighbor interactions)
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::melting_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (no double buffering needed for self-only writes)
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = d_temp;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;
  auto d_G_local = d_G;
  auto d_V_local = d_V;
  auto d_spath_local = d_spath;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nspins_local = this->nspins;
  const uint64_t key_local = rng_key();   // KeyedRNG: this step's draws

  // Melting kernel: parallel over the active-front list with event tracking
  // Race-free: each site only reads/writes itself
  int naccept_local = 0;
  Kokkos::parallel_reduce("MeltingKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count) {
      const int i = d_active_sites_local(idx);
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);

      // Check if site should melt: T >= liquidus && not already liquid
      if (T_i >= tl_local && active_flag_i != 2) {
        KeyedRNG generator(key_local, i, RNG_MELT);

        // Transition to liquid state
        d_active_flag_local(i) = 2;  // LIQUID

        // Randomize grain ID (destroy grain structure)
        d_spin_local(i) = static_cast<int>(nspins_local * generator.drand());

        // Randomize orientation (destroy crystallographic texture)
        double q[4];
        QuaternionOps::generate_random_quaternion(q, generator);
        d_q0_local(i) = q[0];
        d_qx_local(i) = q[1];
        d_qy_local(i) = q[2];
        d_qz_local(i) = q[3];

        // Reset solidification tracking
        d_solid_d_local(i) = 0.0;
        d_G_local(i) = 0.0;
        d_V_local(i) = 0.0;
        d_spath_local(i) = 0.0;


        // Track melting event
        count++;
      }
      // No else branch needed - site keeps its current values if not melting
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   floor_check: lowest liquid site vs the box bottom. The active-front list
   holds every liquid site, so ranging over it sees them all.
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::check_melt_floor(int n_active)
{
  ensure_device_xyz();
  auto list = d_active_sites;
  auto af = d_active_flag;
  auto xyz_d = d_xyz;
  double zmin_local = 1.0e300, zmin = 1.0e300;
  Kokkos::parallel_reduce("MeltFloor", n_active,
    KOKKOS_LAMBDA(int idx, double &m) {
      const int i = list(idx);
      if (af(i) == 2 && xyz_d(i,2) < m) m = xyz_d(i,2);
    }, Kokkos::Min<double>(zmin_local));
  MPI_Allreduce(&zmin_local, &zmin, 1, MPI_DOUBLE, MPI_MIN, world);
  // Report each new deepest site; a driver sizes the next box from the last line.
  if (zmin < 1.0e299 && zmin <= floor_zmin_run - 1.0) {
    floor_zmin_run = zmin;
    if (domain->me == 0) {
      char line[160];
      snprintf(line, sizeof(line), "floor_check: deepest liquid z = %g (%g m above the box bottom) at t = %g\n",
               zmin, (zmin - domain->boxzlo) * dx, time);
      if (screen) fputs(line, screen);
      if (logfile) fputs(line, logfile);
    }
  }
  const double zfloor = domain->boxzlo + floor_check_d / dx;
  if (zmin < zfloor) {
    char msg[256];
    snprintf(msg, sizeof(msg), "floor_check: liquid at z = %g sites, within %g m of the box "
             "bottom (z = %g) at t = %g; deepen the box", zmin, floor_check_d, domain->boxzlo, time);
    error->all(FLERR, msg);
  }
}

/* ----------------------------------------------------------------------
   Nucleation kernel: Stochastic nucleation in undercooled liquid
   Phase 3.2: Sites transition to solid when undercooling exceeds critical value
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::nucleation_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views - simple Views (no double buffering needed for self-only writes)
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = d_temp;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  // Capture parameters for lambda
  double tl_local = this->tl;
  int nrefine_local = this->nrefine;
  const uint64_t key_local = rng_key();   // KeyedRNG: this step's draws
  auto d_nucleation_flags_local = this->d_nucleation_flags;
  auto d_nucleation_temps_local = this->d_nucleation_temps;
  auto d_solid_front_coeffs_local = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  const double solid_front_a_local = this->solid_front_a, solid_front_n_local = this->solid_front_n;
  auto d_V_local = this->d_V;
  auto d_G_local = this->d_G;
  auto k_numneigh_d = d_numneigh;
  auto k_neighbor_d = d_neighbor_rm;
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;
  const double dx_local = dx;

  // Nucleation kernel: parallel over all local sites with event tracking
  // Race-free: each site only reads/writes itself
  int naccept_local = 0;
  Kokkos::parallel_reduce("NucleationKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count) {
      const int i = d_active_sites_local(idx);
      // Read current state
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);
      int spin_i = d_spin_local(i);

      // Check if site is liquid and below liquidus (undercooled)
      if (active_flag_i == 2 && T_i < tl_local) {
        // Check if this spin can nucleate
        if (d_nucleation_flags_local(spin_i) == 1) {
          // Calculate undercooling
          double Tcool = tl_local - T_i;

          // Check if undercooling exceeds critical value for this spin
          if (Tcool >= d_nucleation_temps_local(spin_i)) {
            // Nucleate: transition to solid
            d_active_flag_local(i) = 3;  // SOLID
            count++;  // Track nucleation event

            // Set solid_d flag to prevent disappearance during smoothing
            // Negative value triggers smoothing countdown later
            d_solid_d_local(i) = -nrefine_local - 2;

            // Keep the assigned spin (already set during initialization)
            // d_spin_local(i) = spin_i;  // No-op, already has this value

            // Random unit quaternion for the new grain orientation
            KeyedRNG generator(key_local, i, RNG_NUCLEATE);
            double q[4];
            QuaternionOps::generate_random_quaternion(q, generator);
            d_q0_local(i) = q[0];
            d_qx_local(i) = q[1];
            d_qy_local(i) = q[2];
            d_qz_local(i) = q[3];

            d_V_local(i) = solid_front_velocity(d_solid_front_coeffs_local,
                                                solid_front_length_local, solid_front_a_local,
                                                solid_front_n_local, Tcool);

            // Gradient magnitude, as the CPU's normal_finder in
            // execute_nucleation_event; the flipper copies it to the nucleus.
            double grad_dir[3];
            double grad_magnitude = 0.0;
            ThermalOps::compute_temperature_gradient(
              i, d_T_local, d_active_flag_local, k_numneigh_d, k_neighbor_d, k_xyz_d, dx_local,
              grad_dir, &grad_magnitude);
            d_G_local(i) = grad_magnitude;
          }
          // No else branch - site keeps current values if not nucleating
        }
        // No else branch - site keeps current values if can't nucleate
      }
      // No else branch - site keeps current values if not liquid/undercooled
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Phase 3.5: Particle flipper) kernel - grow nuclei to specified volume

   Grows newly nucleated grains by flipping neighboring liquid sites to
   match the nucleus orientation and grain ID. This avoids lattice-size
   dependency where nuclei start as single sites.

   Algorithm:
   1. Identify newly nucleated sites (solid_d == -nrefine-2)
   2. Calculate target number of sites from nucleation_sizes
   3. Flip liquid neighbors in randomized order until target reached
   4. Copy nucleus spin, quaternion, G, and V to flipped sites
   5. Set flipped sites to solid_d = -nrefine-3 (ready for smoothing)

   The nuclei of a step are grown one after another in site order (one
   device thread) so a contested liquid site always goes to the same
   nucleus; the run is then repeatable at a fixed seed and rank count.
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::particle_flipper_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  // Get device views
  auto d_active_sites_local = d_active_sites;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_neighbor_d = d_neighbor_rm;
  auto k_numneigh_d = d_numneigh;
  auto d_G_local = d_G;
  auto d_V_local = d_V;
  auto d_spath_local = d_spath;
  auto d_nucleation_sizes_local = d_nucleation_sizes;
  const uint64_t key_local = rng_key();   // KeyedRNG: this step's draws

  int nrefine_local = nrefine;
  double dx_local = dx;
  const int nlocal_local = nlocal;

  // Nucleated-shell misorientation (mirrors the CPU
  // nucleation_particle_flipper): each claimed site takes one walk step of
  // sqrt(D_eff * distance_from_seed) about a random axis, relative to the
  // seed orientation, with the seed's undercooling.
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;
  auto d_T_local = d_temp;
  const double mis_diffusion_local = mis_diffusion;
  const double misorient_alpha_local = misorient_alpha;
  const double t_cool_max_local = t_cool_max;
  const double tl_local = tl;
  const double mis_thresh_local = mis_thresh;
  const int nspins_local = nspins;
  const int misori_mode_local = misorientation_target_mode;
  const bool nucl_rotate = (mis_diffusion > 0.0 && nucleation_misorient_enabled);

  // Terminal solid_d for a nucleus and for the neighbours it claims. `off`
  // (default) keeps -nrefine-3, which sits OUTSIDE the smoothing window
  // (solid_d < 0 && solid_d > -nrefine-1) so nuclei are never smoothed; `on`
  // uses -1, INSIDE that window, so they take the ordinary smoothing pass --
  // matching AppAdditiveTexture's `nuclei_smooth_enabled ? -1 : -nrefine-3`.
  // The -nrefine-2 marker the nucleation kernel writes is deliberately NOT
  // switched: unlike the CPU app (which grows the nucleus inline inside
  // execute_nucleation_event), this port splits nucleation and flipping into
  // two kernels, and the flipper re-identifies its work by that exact value.
  const double nucleus_done_d =
      nuclei_smooth_enabled ? -1.0 : (double)(-nrefine_local - 3);

  // This step's nuclei (marked solid_d = -nrefine-2), in site order. They
  // grow one after another in that order, in a single-thread launch: a liquid
  // site two nuclei reach goes to the lower-index one every run, where
  // concurrent CAS claims gave it to whichever thread got there first.
  // ponytail: serial over the step's nuclei, ~11-100 us per nucleus on
  // gfx1151 (0.1-1 s for a 1e4-nucleus step; unmeasurable on a GP700 box,
  // even at 10x nucleation). If a profile shows it: a nucleus reaches at most
  // R = round(V_max/dx^3) + 1 sites (claims + the carry-on reads), so bin the
  // nuclei into cells wider than 2R, colour the cells 2x2x2, and launch one
  // thread per occupied cell per colour (site order inside a cell), reusing
  // this body. Still repeatable; bits change (neighbouring cells grow in
  // colour order, not site order). A plain propose/atomic_min claim is
  // smaller but shrinks the losers (no retry), unlike the CPU.
  if ((int) d_nuclei.extent(0) < n_active)
    d_nuclei = Kokkos::View<int*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nuclei"),
                                  Kokkos::max(n_active, 2 * (int) d_nuclei.extent(0)));
  auto d_nuc = d_nuclei;
  int n_nuc = 0;
  Kokkos::parallel_scan("FindNuclei", n_active,
    KOKKOS_LAMBDA(int idx, int& off, bool final) {
      const int i = d_active_sites_local(idx);
      if (d_active_flag_local(i) == 3 && d_solid_d_local(i) == -nrefine_local - 2) {
        if (final) d_nuc(off) = i;
        off++;
      }
    }, n_nuc);
  if (n_nuc == 0) return 0;
  Kokkos::sort(d_nuc, 0, n_nuc);   // the active list's order varies run to run

  int naccept_local = 0;
  Kokkos::parallel_reduce("ParticleFlipperKernel", Kokkos::RangePolicy<>(0, 1),
    KOKKOS_LAMBDA(int, int& count) {
    for (int kn = 0; kn < n_nuc; kn++) {
      const int i = d_nuc(kn);

      // This is a newly nucleated site - attempt to grow it
      int spin_i = d_spin_local(i);
      double q_i[4] = {d_q0_local(i), d_qx_local(i), d_qy_local(i), d_qz_local(i)};
      double G_i = d_G_local(i);
      double V_i = d_V_local(i);

      const double Tcool_seed = tl_local - d_T_local(i);

      // Grow the nucleus as the CPU app's nucleation_particle_flipper does:
      // claim round(V/dx^3) liquid sites besides the seed, shell by shell
      // (6 faces, 12 edges, 8 corners), each shell in random order; when a
      // centre's shells run out, carry on from a random owned solid
      // neighbour of the same spin, until the nucleus is full or a round
      // claims nothing. A site an earlier nucleus already took is not
      // claimed again and does not count, as on the CPU.
      double nucleus_volume = d_nucleation_sizes_local(spin_i);
      double voxel_volume = dx_local * dx_local * dx_local;
      int remaining = static_cast<int>(Kokkos::round(nucleus_volume / voxel_volume));

      // sc/26n neighbour slots, nearest shell first (the CPU's tables).
      constexpr int shell_slot[26] = {4, 10, 12, 15, 21, 13,
                                      9, 16, 3, 22, 14, 11, 20, 5, 1, 24, 7, 18,
                                      0, 25, 23, 2, 6, 19, 17, 8};
      constexpr int shell_start[4] = {0, 6, 18, 26};
      constexpr int MAX_NEIGHBORS = 26;

      KeyedRNG generator(key_local, i, RNG_FLIP);
      int center = i;
      while (remaining > 0) {
        const int nn = Kokkos::min(k_numneigh_d(center), MAX_NEIGHBORS);
        int order[MAX_NEIGHBORS];
        if (nn == MAX_NEIGHBORS) {
          for (int j = 0; j < MAX_NEIGHBORS; j++) order[j] = shell_slot[j];
          for (int sh = 0; sh < 3; sh++) {        // Fisher-Yates within a shell
            const int lo = shell_start[sh];
            for (int j = shell_start[sh + 1] - 1; j > lo; j--) {
              int k = lo + Kokkos::min(static_cast<int>(generator.drand() * (j - lo + 1)), j - lo);
              int t = order[j]; order[j] = order[k]; order[k] = t;
            }
          }
        } else {
          for (int j = 0; j < nn; j++) order[j] = j;   // CPU: list order
        }

        const int before = remaining;
        for (int j = 0; j < nn && remaining > 0; j++) {
          const int nbr = k_neighbor_d((size_t) center, order[j]);
          // Owned sites only, as on the CPU: a claimed ghost never reaches its
          // owner and is overwritten by the next exchange.
          if (nbr >= nlocal_local) continue;
          if (Kokkos::atomic_compare_exchange(&d_active_flag_local(nbr), 2, 3) != 2) continue;
          remaining--;

          int spin_nbr = spin_i;
          double q_nbr[4] = {q_i[0], q_i[1], q_i[2], q_i[3]};

          if (nucl_rotate) {
            // Distance from the SEED (lattice units -> m), as on the CPU.
            const double ddx = k_xyz_d(nbr, 0) - k_xyz_d(i, 0);
            const double ddy = k_xyz_d(nbr, 1) - k_xyz_d(i, 1);
            const double ddz = k_xyz_d(nbr, 2) - k_xyz_d(i, 2);
            double mis_angle = diffusion_angle(mis_diffusion_local, misorient_alpha_local,
                                               t_cool_max_local, Tcool_seed,
                                               Kokkos::sqrt(ddx*ddx + ddy*ddy + ddz*ddz) * dx_local);

            double q_rot[4];
            perturb_orientation(misori_mode_local, q_nbr, mis_angle, generator, q_rot);

            // mis_thresh new-grain check (direct angle, matching the
            // solidification kernel's convention).
            if (direct_angle(q_nbr, q_rot) >= mis_thresh_local) {
              spin_nbr = static_cast<int>(nspins_local * generator.drand());
            }

            q_nbr[0] = q_rot[0];
            q_nbr[1] = q_rot[1];
            q_nbr[2] = q_rot[2];
            q_nbr[3] = q_rot[3];
          }

          d_spin_local(nbr) = spin_nbr;
          d_q0_local(nbr) = q_nbr[0];
          d_qx_local(nbr) = q_nbr[1];
          d_qy_local(nbr) = q_nbr[2];
          d_qz_local(nbr) = q_nbr[3];
          // active_flag already set to SOLID by CAS above
          d_solid_d_local(nbr) = nucleus_done_d;  // terminal, or -1 under nuclei_smooth
          d_spath_local(nbr) = 0.0;

          // Copy thermal properties from nucleus
          d_G_local(nbr) = G_i;
          d_V_local(nbr) = V_i;
        }
        if (remaining == 0 || remaining == before) break;

        // Carry on from a random solid neighbour of the centre's spin.
        const int spin_c = d_spin_local(center);
        int cand[MAX_NEIGHBORS];
        int nc = 0;
        for (int j = 0; j < nn; j++) {
          const int nbr = k_neighbor_d((size_t) center, j);
          if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) == spin_c) cand[nc++] = nbr;
        }
        if (nc == 0) break;
        const int next = cand[static_cast<int>(Kokkos::round((nc - 1) * generator.drand()))];
        if (next >= nlocal_local) break;   // the CPU recurses only into owned sites
        center = next;
      }

      // Update nucleus solid_d to ready state
      d_solid_d_local(i) = nucleus_done_d;  // terminal, or -1 under nuclei_smooth

      // Track nucleus growth event
      count++;
    }
    }, naccept_local);

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Solidification kernel: Epitaxial growth with texture weighting
   Phase 3.3: Liquid sites captured by solid neighbors based on texture
------------------------------------------------------------------------- */

// This step's KeyedRNG key: the deck seed, the rank and the step. A site's
// draws are then fixed by (key, site index, kernel) -- repeatable at a fixed
// seed and rank count, whatever the thread schedule.
uint64_t AppAdditiveTextureKokkos::rng_key() const
{
  return KeyedRNG::mix(KeyedRNG::mix(rng_seed ^ ((uint64_t)(domain->me + 1) * 0xD1B54A32D192ED03ULL))
                       + rng_step * 0x9E3779B97F4A7C15ULL);
}

// Multicolour ordering for the neighbour-reading kernels. Colour = parity
// class (x&1, y&1, z&1) of the lattice coordinates: no two SC_26N neighbours
// share a colour, so within one colour's launch every neighbour read is
// race-free and sees the results of the colours already processed (parallel
// Gauss-Seidel). The colour order is shuffled every step (keyed by the seed and step) so
// the update order is spatially random, like the CPU app's random sweep, and
// never the site-index order (which grew grains along +x). Non-periodic
// boxes only: parity would not match across a periodic wrap of odd length.
void AppAdditiveTextureKokkos::shuffled_colors(int order[8], int kernel)
{
  KeyedRNG r(rng_key(), 0, kernel);   // keyed by step, like every other draw
  for (int c = 0; c < 8; c++) order[c] = c;
  for (int c = 7; c > 0; c--) {
    const int k = static_cast<int>(r.drand() * (c + 1));
    const int t = order[c]; order[c] = order[k]; order[k] = t;
  }
}


int AppAdditiveTextureKokkos::solidification_kernel(Kokkos::View<int*> d_active_sites, int n_active)
{
  auto d_active_sites_local = d_active_sites;
  auto d_T_local = d_temp;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_numneigh_d = d_numneigh;
  auto k_neighbor_d = d_neighbor_rm;

  // Cached, layout-correct device copy of site coordinates (used for on-demand
  // gradient computation in the kernel).
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;

  // Capture parameters
  double tl_local = this->tl;
  double dt_local = this->dt;
  double dx_local = this->dx;
  auto d_solid_front_coeffs_local = this->d_solid_front_coeffs;
  int solid_front_length_local = this->solid_front_length;
  const double solid_front_a_local = this->solid_front_a, solid_front_n_local = this->solid_front_n;
  const double c1_local = c1, c2_local = c2, c3_local = c3;
  auto d_G_local = this->d_G;
  auto d_V_local = this->d_V;
  auto d_spath_local = this->d_spath;
  const uint64_t key_local = rng_key();   // KeyedRNG: this step's draws

  // Misorientation random walk parameters
  const double mis_diffusion_local = this->mis_diffusion;
  const double gw_angle_local = (grain_walk_D > 0.0) ? sqrt(grain_walk_D * dx) : 0.0;
  const double gw_rho_local = grain_walk_rho;
  const int gw_path_local = grain_walk_path;
  GwCached gw_local; gw_local.gid_of_spin = d_gw_gid; gw_local.ngid = d_gw_ngid; gw_local.nready = d_gw_nready; gw_local.tab = d_gw_tab;
  gw_local.maxgid = (d_gw_tab.extent(0) > 0) ? gw_maxgid : 0; gw_local.maxlev = gw_maxlev; gw_local.rms = gw_angle_local; gw_local.rho = gw_rho_local;
  const double t_cool_max_local = this->t_cool_max;
  const double mis_thresh_local = this->mis_thresh;
  const double misorient_alpha_local = this->misorient_alpha;
  const int misori_mode_local = this->misorientation_target_mode;
  int nspins_local = this->nspins;

  // Solid-front reach threshold: a neighbor becomes eligible once the front
  // distance solid_d has advanced past the farthest SC_26N neighbor (the cube
  // corner, sqrt(3) lattice units). This matches the CPU reference, which tests
  // the constant neigh_dist[2] == sqrt3*dx for every neighbor
  // (app_additive_texture.cpp:1143) -- NOT a per-neighbor-slot distance. Using a
  // slot-indexed table here was wrong twice over: it deviated from the CPU even
  // for interior sites, and for low-coordination boundary sites (compacted
  // neighbor list) the slot did not correspond to the neighbor's real offset.
  constexpr double solid_reach = 1.7320508075688772;  // sqrt(3), lattice units

  // Solidification kernel: parallel over all local sites with event tracking
  // Multicolour Gauss-Seidel: one launch per parity colour in a per-step
  // random order (see shuffled_colors); neighbour reads are live and race-free.
  int naccept_local = 0;
  int color_order[8]; shuffled_colors(color_order, RNG_COLORS_SOLIDIFY);
  for (int cc = 0; cc < 8; cc++) {
  const int color_local = color_order[cc];
  int naccept_color = 0;
  Kokkos::parallel_reduce("SolidificationKernel", n_active,
    KOKKOS_LAMBDA(int idx, int& count) {
      const int i = d_active_sites_local(idx);
      if (((static_cast<int>(k_xyz_d(i,0)) & 1) | ((static_cast<int>(k_xyz_d(i,1)) & 1) << 1) |
           ((static_cast<int>(k_xyz_d(i,2)) & 1) << 2)) != color_local) return;
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_local(i);

      // Check if site is liquid and below liquidus
      if (active_flag_i == 2 && T_i < tl_local) {
        double Tcool = tl_local - T_i;
        double solid_d_i = d_solid_d_local(i);

        // CPU parity (site_event_solidification): the front distance advances
        // only while the site is adjacent to solid (or sits on the box
        // boundary). Advancing it for every undercooled liquid site pre-charged
        // the whole pool, so the front then followed the isotherm instead of
        // the kinetics and captured along the scan direction one voxel per
        // step (grains hundreds of um long along x at dt 2 us).
        int nn = k_numneigh_d(i);
        bool adjacent_solid = (nn < 26);
        for (int j = 0; j < nn && j < 26 && !adjacent_solid; j++) {
          const int f = d_active_flag_local(k_neighbor_d((size_t) i, j));
          if (f == 1 || f == 3) adjacent_solid = true;
        }
        if (!adjacent_solid) return;

        // Advance solid front distance: solid_d += V(Tcool) * dt
        const double V_advance = solid_front_velocity(d_solid_front_coeffs_local,
                                                      solid_front_length_local, solid_front_a_local,
                                                      solid_front_n_local, Tcool);
        solid_d_i += V_advance * dt_local;

        // Find solid neighbors within reach; weights are computed per GRAIN below
        int num_solid_neighbors = 0;
        int solid_neighbor_idx[26];
        int solid_neighbor_spin[26];

        // On-demand gradient computation: only compute if we might have solid neighbors
        // This is the key optimization - gradients computed only when needed
        double grad_dir[3] = {0.0, 0.0, 0.0};
        double grad_magnitude = 0.0;
        bool gradient_computed = false;

        for (int j = 0; j < nn && j < 26; j++) {
          int nbr = k_neighbor_d((size_t) i, j);
          int nbr_flag = d_active_flag_local(nbr);   // live: same-colour sites are never neighbours

          // Check if neighbor is solid (active_flag == 1 or 3) and within reach
          if ((nbr_flag == 1 || nbr_flag == 3) && solid_reach * dx_local <= solid_d_i) {
            // Compute gradient on-demand (only once, on first solid neighbor)
            if (!gradient_computed) {
              // Use ThermalOps to compute temperature gradient at site i
              ThermalOps::compute_temperature_gradient(
                i, d_T_local, d_active_flag_local, k_numneigh_d, k_neighbor_d, k_xyz_d, dx_local,
                grad_dir, &grad_magnitude);

              // Store gradient magnitude for later use
              d_G_local(i) = grad_magnitude;
              gradient_computed = true;
            }

            solid_neighbor_idx[num_solid_neighbors] = nbr;
            solid_neighbor_spin[num_solid_neighbors] = d_spin_local(nbr);
            num_solid_neighbors++;
          }
        }

        // If we have solid neighbors, perform stochastic selection
        if (num_solid_neighbors > 0) {
          // Parent selection is per GRAIN, not per voxel (matches
          // AppAdditiveTexture::site_event_solidification): weighting each
          // candidate voxel by its own orientation let the mobility pick, among
          // the randomly rotated voxels of ONE grain, the ones rotated toward
          // gradient alignment, so whole grains drifted toward <001>||BD with
          // height. Group the candidates by spin, weight each grain by the
          // mobility c1 + c2*cos(c3*theta) of the sign-aligned mean orientation
          // of its candidate voxels against the gradient at this site (zero
          // gradient => full mobility; negative => zero weight), draw the grain,
          // then draw the parent voxel uniformly within it.
          const double c1 = c1_local;
          const double c2 = c2_local;
          const double c3 = c3_local;
          int gspin[26]; double gcum[26]; int ngr = 0; double gtotal = 0.0;
          for (int k = 0; k < num_solid_neighbors; k++) {
            bool seen = false;
            for (int g = 0; g < ngr; g++) if (gspin[g] == solid_neighbor_spin[k]) { seen = true; break; }
            if (seen) continue;
            double qm[4] = {0.0, 0.0, 0.0, 0.0}, qr[4] = {0.0, 0.0, 0.0, 0.0}; bool first = true;
            for (int m = 0; m < num_solid_neighbors; m++) {
              if (solid_neighbor_spin[m] != solid_neighbor_spin[k]) continue;
              const int n = solid_neighbor_idx[m];
              double qn[4] = {d_q0_local(n), d_qx_local(n), d_qy_local(n), d_qz_local(n)};
              if (first) { for (int c = 0; c < 4; c++) qr[c] = qn[c]; first = false; }
              else if (qr[0]*qn[0] + qr[1]*qn[1] + qr[2]*qn[2] + qr[3]*qn[3] < 0.0)
                for (int c = 0; c < 4; c++) qn[c] = -qn[c];   // q and -q are one orientation
              for (int c = 0; c < 4; c++) qm[c] += qn[c];
            }
            QuaternionOps::normalize(qm);
            double mobility = c1 + c2;
            if (grad_magnitude > 0.0) {
              const double theta = Kokkos::acos(QuaternionOps::get_cosine_angle_q_u(qm, grad_dir));
              mobility = c1 + c2 * Kokkos::cos(c3 * theta);
            }
            gtotal += Kokkos::fmax(mobility, 0.0);
            gspin[ngr] = solid_neighbor_spin[k]; gcum[ngr] = gtotal; ngr++;
          }

          KeyedRNG generator(key_local, i, RNG_SOLIDIFY);
          int g = 0;
          if (gtotal > 0.0) {
            const double rand_val = generator.drand() * gtotal;
            while (g < ngr - 1 && rand_val > gcum[g]) g++;
          } else {
            g = static_cast<int>(generator.drand() * ngr);
            if (g >= ngr) g = ngr - 1;
          }
          int nsame = 0;
          for (int m = 0; m < num_solid_neighbors; m++) if (solid_neighbor_spin[m] == gspin[g]) nsame++;
          int pick = static_cast<int>(generator.drand() * nsame);
          if (pick >= nsame) pick = nsame - 1;
          int selected_idx = 0;
          for (int m = 0; m < num_solid_neighbors; m++) {
            if (solid_neighbor_spin[m] != gspin[g]) continue;
            if (pick-- == 0) { selected_idx = m; break; }
          }

          int selected_nbr = solid_neighbor_idx[selected_idx];

          // Solidify: copy grain ID and quaternion from selected neighbor
          int new_spin = d_spin_local(selected_nbr);
          double q_nbr_inherited[4] = {
            d_q0_local(selected_nbr),
            d_qx_local(selected_nbr),
            d_qy_local(selected_nbr),
            d_qz_local(selected_nbr)
          };

          // Misorientation random walk: one step of sqrt(D_eff * hop) about a
          // random axis relative to the inheritance parent (matches
          // apply_misorientation in the CPU app); hop_distance() is in
          // lattice units.
          if (mis_diffusion_local > 0.0) {
            double mis_angle = diffusion_angle(mis_diffusion_local, misorient_alpha_local,
                                               t_cool_max_local, Tcool,
                                               hop_distance(k_xyz_d, i, selected_nbr) * dx_local);

            // Normalize the inherited quaternion (should already be unit length)
            QuaternionOps::normalize(q_nbr_inherited);

            double q_rotated[4];
            perturb_orientation(misori_mode_local, q_nbr_inherited,
                                mis_angle, generator, q_rotated);

            // If the rotation exceeds the threshold, create a new grain with a
            // random spin.
            if (direct_angle(q_nbr_inherited, q_rotated) >= mis_thresh_local) {
              new_spin = static_cast<int>(nspins_local * generator.drand());
            }

            // Use the rotated quaternion
            d_q0_local(i) = q_rotated[0];
            d_qx_local(i) = q_rotated[1];
            d_qy_local(i) = q_rotated[2];
            d_qz_local(i) = q_rotated[3];
          } else {
            // No misorientation - use neighbor's orientation as-is
            d_q0_local(i) = q_nbr_inherited[0];
            d_qx_local(i) = q_nbr_inherited[1];
            d_qy_local(i) = q_nbr_inherited[2];
            d_qz_local(i) = q_nbr_inherited[3];
          }
          d_spath_local(i) = d_spath_local(selected_nbr) + hop_distance(k_xyz_d, i, selected_nbr);   // d10
          if (gw_angle_local > 0.0) {   // coherent grain rotation between the parent's level and this one
            double qg[4] = {d_q0_local(i), d_qx_local(i), d_qy_local(i), d_qz_local(i)};
            if (gw_path_local) grain_walk_apply_f(gw_local, d_spin_local(selected_nbr), d_spath_local(selected_nbr), d_spath_local(i), qg);
            else               grain_walk_apply_f(gw_local, d_spin_local(selected_nbr), k_xyz_d(selected_nbr, 2), k_xyz_d(i, 2), qg);
            d_q0_local(i) = qg[0]; d_qx_local(i) = qg[1]; d_qy_local(i) = qg[2]; d_qz_local(i) = qg[3];
          }

          // Assign grain ID (may be new grain if rotation exceeded threshold)
          d_spin_local(i) = new_spin;

          // Set to solid state
          d_active_flag_local(i) = 3;  // SOLID
          count++;  // Track solidification event

          // Set solid_d flag to trigger smoothing countdown
          d_solid_d_local(i) = -1.0;

          // Record solidification parameters
          d_V_local(i) = V_advance;

        } else {
          // No solid neighbors within reach - update advanced distance only
          d_solid_d_local(i) = solid_d_i;
        }
      }
      // No else branch - site keeps current values if not liquid/undercooled
    }, naccept_color);
  naccept_local += naccept_color;
  }

  return naccept_local;
}

/* ----------------------------------------------------------------------
   Smoothing kernel: Potts MC grain-boundary refinement over the compact list
   of sites in the smoothing band (solid_d in (-nrefine-1, 0)).
------------------------------------------------------------------------- */

int AppAdditiveTextureKokkos::smoothing_kernel_efficient(
    Kokkos::View<int*> d_smoothing_sites, int n_sites)
{
  // Multicolour Gauss-Seidel (see shuffled_colors): neighbour reads are live
  // and race-free because same-colour sites are never neighbours.
  ensure_device_xyz();
  auto k_xyz_d = d_xyz;
  auto d_spin_local = d_spin;
  auto d_q0_local = d_q0;
  auto d_qx_local = d_qx;
  auto d_qy_local = d_qy;
  auto d_qz_local = d_qz;
  auto d_active_flag_local = d_active_flag;
  auto d_solid_d_local = d_solid_d;

  auto k_neighbor_d = d_neighbor_rm;
  auto k_numneigh_d = d_numneigh;
  const uint64_t key_local = rng_key();   // KeyedRNG: this step's draws

  double temperature_local = temperature;
  double t_inverse_local = t_inverse;
  const bool greedy_local = smooth_greedy_multiproposal_enabled;
  const bool copy_local = smooth_orientation_copy;
  const double gw_angle_smooth = (grain_walk_D > 0.0) ? sqrt(grain_walk_D * dx) : 0.0;
  const double gw_rho_smooth = grain_walk_rho;
  const int gw_path_smooth = grain_walk_path;
  GwCached gw_sm; gw_sm.gid_of_spin = d_gw_gid; gw_sm.ngid = d_gw_ngid; gw_sm.nready = d_gw_nready; gw_sm.tab = d_gw_tab;
  gw_sm.maxgid = (d_gw_tab.extent(0) > 0) ? gw_maxgid : 0; gw_sm.maxlev = gw_maxlev; gw_sm.rms = gw_angle_smooth; gw_sm.rho = gw_rho_smooth;
  auto d_spath_local = d_spath;
  auto d_mobility_out_local = d_mobility_out;

  // Parallel sweep over ONLY smoothing sites (not all sites!)
  int naccept_local = 0;
  int color_order[8]; shuffled_colors(color_order, RNG_COLORS_SMOOTH);
  for (int cc = 0; cc < 8; cc++) {
  const int color_local = color_order[cc];
  int naccept_color = 0;
  Kokkos::parallel_reduce("SmoothingKernelEfficient", n_sites,
    KOKKOS_LAMBDA(int idx, int& count) {
      // Get actual site index from compact list
      int i = d_smoothing_sites(idx);
      if (((static_cast<int>(k_xyz_d(i,0)) & 1) | ((static_cast<int>(k_xyz_d(i,1)) & 1) << 1) |
           ((static_cast<int>(k_xyz_d(i,2)) & 1) << 2)) != color_local) return;
      d_mobility_out_local(i) = 1.0;   // the CPU marks every site it smooths

      // All sites in this list need smoothing - no branching needed!
      const int active_flag_i = d_active_flag_local(i);
      const int spin_i = d_spin_local(i);
      const double solid_d_i = d_solid_d_local(i);

      // CPU-aligned Potts boundary smoothing (AppAdditiveTexture::smooth_site).
      // Energy = number of SOLID (active_flag==3) neighbors with a different spin
      // (site_energy_smooth). Candidates = distinct neighbor spins != self.
      // Proposal: greedy steepest-descent (evaluate every candidate, keep the
      // bond-energy minimizer; first-encounter tie-break -> deterministic) or a
      // random single proposal. Boltzmann accept. On accept the orientation
      // becomes the sign-aligned average of the selected grain's neighbor
      // quaternions (get_average_neighbor_quaternion). NB: smoothing sites are
      // swept in parallel here (vs the CPU's sequential per-site sweep), so the
      // microstructure is statistically -- not bit -- equal to the CPU even at
      // temperature 0.
      const int nn = k_numneigh_d(i);

      // Distinct neighbor spins != self. Voids (i2 == 5) carry no grain.
      int unique_spins[27];
      int nevent = 0;
      for (int n = 0; n < nn; n++) {
        const int nbr = k_neighbor_d((size_t) i, n);
        if (d_active_flag_local(nbr) == 5) continue;
        const int v = d_spin_local(nbr);
        if (v == spin_i) continue;
        bool found = false;
        for (int k = 0; k < nevent; k++)
          if (unique_spins[k] == v) { found = true; break; }
        if (!found && nevent < 27) unique_spins[nevent++] = v;
      }

      if (nevent > 0) {
        // einitial: bond energy of the current spin.
        int einitial = 0;
        for (int n = 0; n < nn; n++) {
          const int nbr = k_neighbor_d((size_t) i, n);
          if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != spin_i) einitial++;
        }

        KeyedRNG generator(key_local, i, RNG_SMOOTH);

        int new_spin, efinal;
        if (greedy_local) {
          new_spin = unique_spins[0];
          efinal = 0;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d((size_t) i, n);
            if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != new_spin) efinal++;
          }
          for (int c = 1; c < nevent; c++) {
            const int cand = unique_spins[c];
            int e = 0;
            for (int n = 0; n < nn; n++) {
              const int nbr = k_neighbor_d((size_t) i, n);
              if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != cand) e++;
            }
            if (e < efinal) { efinal = e; new_spin = cand; }  // strict -> first-encounter
          }
        } else {
          int iran = (int)(nevent * generator.drand());
          if (iran >= nevent) iran = nevent - 1;
          new_spin = unique_spins[iran];
          efinal = 0;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d((size_t) i, n);
            if (d_active_flag_local(nbr) == 3 && d_spin_local(nbr) != new_spin) efinal++;
          }
        }

        // Boltzmann accept (matches smooth_site).
        bool accept;
        if (efinal <= einitial) accept = true;
        else if (temperature_local == 0.0) accept = false;
        else accept = (generator.drand() <=
                       Kokkos::exp((double)(einitial - efinal) * t_inverse_local));


        if (accept) {
          // Orientation = sign-aligned average of neighbors with new_spin,
          // or (smooth_orientation copy) one random such neighbor.
          int pick = -1;
          if (copy_local) {
            int nc = 0;
            for (int n = 0; n < nn; n++) {
              const int nbr = k_neighbor_d((size_t) i, n);
              if (d_spin_local(nbr) == new_spin && d_active_flag_local(nbr) != 5) nc++;
            }
            pick = (int)(nc * generator.drand()); if (pick >= nc) pick = nc - 1;
          }
          double s0 = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
          double r0 = 0.0, rx = 0.0, ry = 0.0, rz = 0.0;
          int cq = 0; bool first = false; int seen = 0; int picked_nbr = -1;
          for (int n = 0; n < nn; n++) {
            const int nbr = k_neighbor_d((size_t) i, n);
            if (d_spin_local(nbr) != new_spin || d_active_flag_local(nbr) == 5) continue;
            if (copy_local && seen++ != pick) continue;
            picked_nbr = nbr;
            double n0 = d_q0_local(nbr), nx = d_qx_local(nbr),
                   ny = d_qy_local(nbr), nz = d_qz_local(nbr);
            if (!first) { r0 = n0; rx = nx; ry = ny; rz = nz; first = true; }
            else if (r0*n0 + rx*nx + ry*ny + rz*nz < 0.0) {
              n0 = -n0; nx = -nx; ny = -ny; nz = -nz;  // q and -q are the same orientation
            }
            s0 += n0; sx += nx; sy += ny; sz += nz; cq++;
          }
          if (cq > 0) {
            double a0 = s0/cq, ax = sx/cq, ay = sy/cq, az = sz/cq;
            const double nrm = Kokkos::sqrt(a0*a0 + ax*ax + ay*ay + az*az);
            if (nrm > 0.0) { a0/=nrm; ax/=nrm; ay/=nrm; az/=nrm; }
            if (copy_local && picked_nbr >= 0) d_spath_local(i) = d_spath_local(picked_nbr);   // path length travels with the orientation
            if (copy_local && gw_angle_smooth > 0.0 && !gw_path_smooth && picked_nbr >= 0) {   // height mode: copied neighbour may sit on another level
              double qg[4] = {a0, ax, ay, az};
              grain_walk_apply_f(gw_sm, new_spin, k_xyz_d(picked_nbr, 2), k_xyz_d(i, 2), qg);
              a0 = qg[0]; ax = qg[1]; ay = qg[2]; az = qg[3];
            }
            d_spin_local(i) = new_spin;
            d_q0_local(i) = a0; d_qx_local(i) = ax;
            d_qy_local(i) = ay; d_qz_local(i) = az;
            count++;
          }
        }
        // Reject / no-grain: spin and quaternion already hold the original
        // values (this kernel has not written site i up to here).
      }

      // Advance the smoothing window exactly like the CPU (solid_d--): a site
      // enters at -1 on solidification and leaves the band
      // (-nrefine-1, 0) after nrefine passes. (It used to be incremented and
      // clamped at 0, i.e. one pass regardless of nrefine.)
      d_solid_d_local(i) = solid_d_i - 1.0;
      d_active_flag_local(i) = active_flag_i;
    }, naccept_color);
  naccept_local += naccept_color;
  }

  return naccept_local;
}

/* ----------------------------------------------------------------------
   App update - main time integration loop
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::app_update(double dt_sweep)
{
  // PHASE 3.1-3.2: Melting and nucleation kernels
  // PHASE 3.3-3.4: Will add solidification and smoothing

  static int step_count = 0;   // cadence for maybe_run_diagnostics
  step_count++;
  rng_step++;

  // 1. Update temperature + fast-forward.
  //
  // Rosenthal (STANDARD/ANISOTROPIC): evaluate the per-site temperature on the
  // DEVICE -- the host only advances the laser pose (cheap) and we fill d_temp
  // directly on the GPU, then copy back to host T for the diagnostic and
  // host-side consumers. This replaces the dominant host exp/sqrt per-site
  // loop. Fast-forward uses a global max temperature + ThermalManager's
  // path-geometry predictor.
  //
  // temperature_smooth, or a source without a device plan: host path via
  // ThermalManager::advance(), which fills host T[]; we then sync to device.
  refresh_thermal_host();
  bool device_temp = false;
  device_active_list_fresh = false;  // set true only by a fill that built d_active_sites (HDF5)
  if (thermal->enabled()) {
    RosenthalDevicePlan rplan = thermal->rosenthal_device_plan(time);
    MoserDevicePlan mplan;
    if (!rplan.valid) mplan = thermal->moser_device_plan(time);

    // The device->host copy-back is deferred (host_T_stale) and done only before
    // a host consumer reads T -- powder activation below, or the d6 diagnostic
    // at an output step (end of app_update). The global peak temperature for the
    // fast-forward decision is reduced on the device, so no host T is needed.
    ThermalManager::ThermalStep s{time, false, false};
    if (ConductionTemperatureSource *csrc = thermal->conduction_source()) {
      // ---- DEVICE conduction path ----
      // Stateful: the filler advances the enthalpy field to t (sub-cycled) and
      // samples it into d_temp for all nall sites. Moser's pass scheduling runs
      // unchanged around it; the fast-forward predictor is skipped for it.
      device_temp = true;
      if (!cond) cond = std::make_unique<ConductionSolverKokkos>(spk, csrc);
      thermal->set_moser_device_filler([this](double t) -> double {
        ensure_device_xyz();
        fill_maxT_local = cond->fill(t, d_temp, d_xyz, d_active_flag,
                                     nlocal, nlocal + nghost, dx, ts, tl);
        Kokkos::fence();
        host_T_stale = true;
        return global_max_temperature_from_fill();
      });
      s = thermal->moser_advance_device(time, stoptime);
    } else if (rplan.valid) {
      // ---- DEVICE Rosenthal path ----
      device_temp = true;
      fill_temperature_rosenthal_device(rplan);
      host_T_stale = true;
      s = thermal->rosenthal_fast_forward(time, stoptime,
                                          global_max_temperature_from_fill());
    } else if (mplan.valid) {
      // ---- DEVICE Moser path ----
      // ThermalManager runs the per-pass pause/reset + fast-forward decisions on
      // the host (they rebuild the scan from temperature feedback) and calls back
      // here to do every per-site Green's-function fill on the GPU. The filler
      // fills d_temp at a given time and returns the device-reduced global max; it
      // also re-uploads segment/fluctuation Views when a fired pass bumps the
      // scan version. On return the device field is filled at s.new_time.
      device_temp = true;
      thermal->set_moser_device_filler([this](double t) -> double {
        fill_temperature_moser_device(thermal->moser_device_plan(t));
        host_T_stale = true;
        return global_max_temperature_from_fill();
      });
      s = thermal->moser_advance_device(time, stoptime);
    } else if (thermal->hdf5_device_eligible()) {
      // ---- DEVICE HDF5 (unstructured-mesh) path ----
      // The host still does the expensive HDF5 work (reads, layer load, element
      // search/cache, nodal precompute) inside hdf5_device_plan(); the GPU does
      // the per-site barycentric interpolation into d_temp for all nall sites
      // (owned + ghost, positional). The filler fills the device field at a
      // given time and returns the device-reduced global max for fast-forward.
      device_temp = true;
      // Device does the per-node time interpolation itself (see
      // fill_temperature_hdf5_device): tell the source to skip the host
      // precompute. Set once; must precede the first hdf5_device_plan() below.
      if (!device_time_interp_set) {
        thermal->hdf5_source()->set_device_time_interp(true);
        device_time_interp_set = true;
      }
      thermal->set_hdf5_device_filler([this](double t) -> double {
        fill_temperature_hdf5_device(thermal->hdf5_device_plan(t));
        host_T_stale = true;
        // One-step-stale at nprocs>1 (overlaps the collective with this
        // step's kernels); exactly the old blocking path at nprocs==1.
        return global_max_temperature_from_fill_stale();
      });
      s = thermal->hdf5_advance_device(time, stoptime);
      // Ghost temperatures need no special handling: the element cache covers
      // nall (owned + ghost via the chunk-load pad), so the device fill already
      // wrote correct ghost d_temp positionally. A ghost equals the owning rank's
      // value by construction (same global field, same point), so the field is
      // fully device-resident -- no host T round-trip, no T ghost comm.
    } else {
      // ---- HOST path (temperature_smooth / no device plan) ----
      s = thermal->advance(time, stoptime);
    }

    // Apply the manager's decision. The Moser/HDF5 fillers already refilled the
    // device field at s.new_time; the Rosenthal path refills here.
    if (s.end_run) {
      // The run ends here, and its final output reads the host arrays
      // (diagnostics run before stats(), the other sync), so bring all back.
      time = s.new_time;
      sync_app_data_to_host();
      if (host_T_stale) sync_temperature_to_host();
      return;
    }
    if (s.new_time > time) {
      time = s.new_time;
      if (rplan.valid) {
        fill_temperature_rosenthal_device(thermal->rosenthal_device_plan(time));
        host_T_stale = true;
      }
      if (s.activate_powder) {
        activate_powder_sites();
        last_powder_activation_time = time;
      }
    }
  }

  // 2. Sync temperature to device (host path only; the device path already
  //    wrote d_temp on the GPU). The host fill covers owned sites only, so ghost
  //    T is exchanged first: this step's kernels read the neighbour T of this
  //    step, as on the CPU.
  if (!device_temp) {
    timer->stamp();
    comm->all_selective(thermal_host_.ghost_iindices, thermal_host_.nghost_iarray,
                        thermal_host_.ghost_dindices, thermal_host_.nghost_darray);
    timer->stamp(TIME_COMM);
    sync_temperature_to_device();
  }

  // 3. Gradient computation moved to on-demand inside solidification kernel
  // (Removed global compute_gradients_kernel() - gradients now computed only when needed)

  // Build the compact "active-front" list once per step: the union superset of
  // every owned site any microstructure kernel could touch this step --
  //   active_flag==2 (liquid: can nucleate/solidify),
  //   T>=tl (hot: can melt),
  //   solid_d in (-nrefine-1, 0) (smoothing band),
  //   active_flag==3 && solid_d==-nrefine-2 (leftover nucleus for particle flip).
  // The phase kernels below run over this list (size n_active, the thin melt/
  // solidification shell) instead of nlocal. The list View is cached and
  // reallocated only when nlocal changes. Capture LOCAL View copies so the
  // lambda does not capture `this` (device page-fault hazard; see
  // BuildSmoothingList below). n_active==0 => no liquid, nothing hot, no band
  // site, no leftover nucleus => nothing for any microstructure kernel to do.
  int n_active = 0;
  if (device_active_list_fresh) {
    // Full fusion: the temperature fill already atomic-appended the active owned
    // sites into d_active_sites and counted them -- just read the count back.
    // h_active_count is a cached member, not a per-step create_mirror_view: this
    // runs every single step and the mirror is a single int.
    Kokkos::deep_copy(h_active_count, d_active_count);
    n_active = h_active_count();
  } else {
    // Full predicate scan (Rosenthal/Moser/host fills that didn't build the list).
    if ((int)d_active_sites.extent(0) != nlocal)
      d_active_sites = Kokkos::View<int*>("active_sites", nlocal);
    auto d_active_sites_local = d_active_sites;
    const int nlocal_act = this->nlocal;
    auto d_T_act = d_temp;
    auto d_active_flag_act = d_active_flag;
    auto d_solid_d_act = d_solid_d;
    const double tl_act = this->tl;
    const int nrefine_act = this->nrefine;
    Kokkos::parallel_scan("BuildActiveList", nlocal_act,
      KOKKOS_LAMBDA(int i, int& offset, bool final) {
        const int af = d_active_flag_act(i);
        const double sd = d_solid_d_act(i);
        // i2 == 5 (void) is permanent: keeping it off this list keeps it out
        // of every melt/nucleate/solidify/smooth kernel.
        const bool active = af != 5 && (
            (af == 2) ||
            (d_T_act(i) >= tl_act) ||
            (sd > -nrefine_act - 1 && sd < 0.0) ||
            (af == 3 && sd == -nrefine_act - 2));
        if (active) {
          if (final) d_active_sites_local(offset) = i;
          offset++;
        }
      }, n_active);
  }

  // 4. Phase transition kernels with event tracking
  // Phase 3.1: Melting (with Phase 1 optimization: early temperature scan)

  // Melt over the active-front list. Melt candidates (T>=tl && active_flag!=2)
  // are a subset of the active list (hot clause), so the old CheckMeltingCandidates
  // pre-scan over nlocal is gone -- the kernel's own naccept is the candidate
  // count (melting is deterministic given T>=tl: it always melts a candidate,
  // RNG only randomizes the destroyed grain/orientation). So naccept>0 iff there
  // were candidates, preserving the powder-activation trigger (melting_occurred).
  int nmelt = melting_kernel(d_active_sites, n_active);
  naccept += nmelt;
  if (floor_check_d > 0.0) check_melt_floor(n_active);
  bool melting_occurred = (nmelt > 0);

  // Phase 3.2: Nucleation (stochastic nucleation in undercooled liquid)

  // Nucleation ranges over the active-front list, which contains EVERY liquid
  // (active_flag==2) site, so nucleation still sees all liquid every step -- the
  // "always launch / never gate on a candidate sub-count" recovery invariant
  // (see the removed block below) is preserved. n_active==0 means no liquid
  // anywhere, where nucleation is impossible by its own guard, so ranging over 0
  // is correct.
  naccept += nucleation_kernel(d_active_sites, n_active);

  // Phase 3.5: Particle flipper (grow nuclei to specified size)

  // Grow nuclei over the active-front list. Nucleus sites (active_flag==3 &&
  // solid_d==-nrefine-2) are a subset of the active list (leftover-nucleus
  // clause), so the separate CheckNucleiSites pre-scan over nlocal is gone -- the
  // kernel's own guard rejects non-nuclei, and ranging over n_active==0 is a
  // correct no-op (no nuclei).
  naccept += particle_flipper_kernel(d_active_sites, n_active);

  // Phase 3.4: Smoothing kernel (Potts MC for grain boundary refinement)
  // The list is built BEFORE the solidification kernel, so a site is smoothed on
  // the steps AFTER it solidified (the CPU app's else-if chain does the same);
  // smoothing a voxel in the very step it solidified, with most of its
  // neighbours still liquid, let the greedy majority vote at the front hand
  // everything to the locally dominant grain (23 vs 732 grains on the GP700
  // single-hatch check).

  // Build compact list of smoothing sites using parallel_scan (Phase 1
  // optimization). The list View is cached (member) and reallocated only when
  // nlocal changes, rather than allocated on the GPU every timestep.
  int n_smoothing_sites = 0;
  if ((int)d_smoothing_sites.extent(0) != nlocal)
    d_smoothing_sites = Kokkos::View<int*>("smoothing_sites", nlocal);
  {
    auto d_solid_d_check = d_solid_d;
    int nrefine_check = nrefine;
    // Capture a LOCAL copy of the View handle. Using the member
    // d_smoothing_sites directly inside KOKKOS_LAMBDA ([=]) would capture
    // `this` and dereference the host `this` pointer on the device to reach
    // the View object -> GPU "page not present" fault (and the
    // -Wdeprecated-this-capture warning). The fault only fired once a site
    // first entered the smoothing band (final && needs_smoothing), which made
    // it look like a melt-triggered, run-dependent crash.
    auto d_smoothing_sites_local = d_smoothing_sites;
    auto d_active_sites_local = d_active_sites;

    // Scan the active-front list (size n_active) rather than nlocal: every
    // smoothing-band site is in d_active_sites (band clause), so this builds
    // the same compact smoothing list (in list order, which the parity-
    // coloured smoothing does not depend on) at O(n_active), not O(nlocal).
    Kokkos::parallel_scan("BuildSmoothingList", n_active,
      KOKKOS_LAMBDA(int idx, int& offset, bool final) {
        const int i = d_active_sites_local(idx);
        double sd = d_solid_d_check(i);
        bool needs_smoothing = (sd > -nrefine_check - 1 && sd < 0.0);

        if (needs_smoothing) {
          if (final) {
            d_smoothing_sites_local(offset) = i;  // Store site index
          }
          offset++;
        }
      }, n_smoothing_sites);
  }

  // Phase 3.3: Solidification (epitaxial growth with texture weighting)
  naccept += solidification_kernel(d_active_sites, n_active);

  // Only launch smoothing kernel if there are eligible sites
  // Process ONLY the sites in the compact list (not all 2.99M sites!)
  if (n_smoothing_sites > 0) {
    naccept += smoothing_kernel_efficient(d_smoothing_sites, n_smoothing_sites);
  }

  // 5. Single fence after all kernels complete (replaces individual kernel fences)
  // This ensures all device writes are visible before proceeding
  Kokkos::fence();

  // Activate powder once melting first occurs, BEFORE syncing to host so the
  // device activation (which writes d_active_flag) is reflected in the host
  // copy used for output. Runs on the device.
  // CRITICAL: MPI_Allreduce the melting flag so ALL ranks call
  // activate_powder_sites() together (it contains collective calls); otherwise
  // ranks with no local melting would skip the collective and deadlock.
  if (last_powder_activation_time < 0.0) {
    int local_melting = melting_occurred ? 1 : 0;
    int global_melting = 0;
    MPI_Allreduce(&local_melting, &global_melting, 1, MPI_INT, MPI_SUM, world);

    if (global_melting > 0) {
      activate_powder_sites();
      last_powder_activation_time = time;
    }
  }

  // Device->host / ghost sync. The device is the source of truth: the physics
  // kernels read and write device Views every step. Host iarray/darray are only
  // consumed by stats, dumps, and diagnostics, so a full device->host copy is
  // only needed on output steps -- doing it every step (the old behavior) was
  // the dominant per-step cost (a ~10-array, O(nall) malloc+copy over millions
  // of sites). "Output fires this step" is time + dt_sweep >= nextoutput (the
  // iterate loop advances time AFTER app_update; nextoutput spans stats/dumps/
  // diagnostics), the same gate the lazy temperature copy-back below uses.
  const bool host_output_imminent =
      (time + dt_sweep >= nextoutput) || (time + dt_sweep >= stoptime);

  if (nprocs > 1) {
    // Multi-rank ghost exchange (Phase 2: selective). The kernels read as
    // NEIGHBORS the Monte Carlo state -- spin (iarray[0]), active_flag
    // (iarray[1]), quaternions q0-3 (darray[0-3]) -- and, in the solidification
    // gradient, neighbor temperature T (darray[5]). mobility/solid_d/G/V are
    // never read as neighbors, so they are dropped (full-synced only at output).
    //
    // T is not exchanged here: on the host-temperature path (temperature_smooth /
    // no device plan) ghost T was exchanged right after the
    // fill, before the kernels; on a device-temp path the fill already wrote
    // d_temp over all nall positionally.
    //
    // sync_ghost_to_host/device move the OWNED out / ghost in for spin/
    // active_flag/q0-3 only (T lives in host T[]/d_temp, handled above). all_selective
    // uses the same swap as all(), so the same ghost sites are exchanged.
    static const int sel_i[2] = {0, 1};            // spin, active_flag
    static const int sel_d[5] = {0, 1, 2, 3, 9};             // q0-3 + spath
    sync_ghost_to_host();                       // device owned -> host raw
    timer->stamp();
    comm->all_selective(sel_i, 2, sel_d, 5);
    timer->stamp(TIME_COMM);
    sync_ghost_to_device();                     // host ghost -> device
  }
  // Full device->host of ALL arrays only on output steps (stats/dumps/diags read
  // the host iarray/darray). Applies to both single and multi rank.
  if (host_output_imminent) sync_app_data_to_host();

  // Optional isotherm-compactness diagnostic on its configured cadence, matching
  // the CPU app. A no-op unless temperature_smooth + a diag interval are set,
  // and temperature_smooth forces the host temperature path, so host T is
  // never stale when it runs -- no copy-back here.
  thermal->maybe_run_diagnostics(step_count);

  // Lazy device->host temperature copy-back: if a device fill left host T (d6)
  // stale and an output step is imminent, refresh host T now so the diagnostic
  // and dumps see the current field. This app sets app_update_only=1, so it runs
  // via AppLattice::iterate_app_update_only, which advances time once per step
  // (time += dt, dt == dt_sweep here) AFTER this returns -- so output fires next
  // when time + dt_sweep >= nextoutput (or the run ends).
  if (host_T_stale &&
      (time + dt_sweep >= nextoutput || time + dt_sweep >= stoptime)) {
    sync_temperature_to_host();
  }

  // NOTE: time is advanced by the iterate loop (iterate_app_update_only does
  // time += dt), exactly as in the CPU AppAdditiveTexture. Do NOT increment time
  // here -- an earlier version added a second `time += dt_sweep`, double-counting
  // time (advancing 2x per step), which halved the MC sweep resolution and
  // shifted the temperature sampling phase vs the CPU app.
}

/* ----------------------------------------------------------------------
   Activate powder sites that are at or below the meltpool surface

   Strategy: Find the maximum z-coordinate of all molten sites (meltpool top),
   then set all inactive sites below that level to powder (active_flag = 1).

   This function should be called:
   - After fast forwards > 1 second (new layer deposition)
   - During app_update when meltpool is present

   Runs entirely on the device, operating directly on d_active_flag (the
   authoritative phase-state array): a parallel_reduce for the meltpool-top z
   over molten sites, then a parallel_reduce that activates inactive sites at or
   below it. Keeping it on the device (rather than mutating host active_flag) is
   what makes it consistent -- a host write here would be overwritten by the
   device->host sync, and for nprocs==1 never reaches the device at all.
------------------------------------------------------------------------- */
void AppAdditiveTextureKokkos::activate_powder_sites() {
  ensure_device_xyz();
  auto d_af = d_active_flag;
  auto d_xyz_l = d_xyz;
  const int n = nlocal;

  // Local meltpool top z over molten sites (active_flag == 2), then the global
  // max across ranks. z is in lattice units, matching the host comparison.
  // local_max_molten_z is the reduction OUTPUT; Kokkos::Max seeds the reduction
  // with its own identity (-DBL_MAX), so with no molten sites the result stays
  // very negative and the -1e99 guard below detects "no meltpool".
  double local_max_molten_z;
  Kokkos::parallel_reduce("MeltpoolTopZ", n,
    KOKKOS_LAMBDA(const int i, double &m) {
      if (d_af(i) == 2) { const double z = d_xyz_l(i,2); if (z > m) m = z; }
    }, Kokkos::Max<double>(local_max_molten_z));

  double global_max_molten_z;
  MPI_Allreduce(&local_max_molten_z, &global_max_molten_z, 1, MPI_DOUBLE, MPI_MAX, world);
  if (global_max_molten_z <= -1e99) return;   // no meltpool anywhere

  // Activate all inactive sites (active_flag == 0) at or below the meltpool top;
  // the reduction returns the count for reporting.
  int powder_activated = 0;
  Kokkos::parallel_reduce("PowderActivate", n,
    KOKKOS_LAMBDA(const int i, int &cnt) {
      if (d_af(i) == 0 && d_xyz_l(i,2) <= global_max_molten_z) {
        d_af(i) = 1;  // POWDER state
        cnt++;
      }
    }, powder_activated);
  Kokkos::fence();

  bigint powder_me = powder_activated, global_powder_activated = 0;  // a layer can pass 2^31
  MPI_Allreduce(&powder_me, &global_powder_activated, 1, MPI_SPK_BIGINT, MPI_SUM, world);
  if (domain->me == 0 && global_powder_activated > 0) {
    fprintf(screen, "Powder activation at t=%.6e s: activated " BIGINT_FORMAT " sites (meltpool top at z=%.3f)\n",
            time, global_powder_activated, global_max_molten_z);
  }
}

/* ----------------------------------------------------------------------
   Override stats to sync device data before computing statistics
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::stats(char *strtmp)
{
  // Sync device data to host so stats can access current values. sync_app_data_
  // to_host() does not touch the temperature field, so refresh host T (d6) here
  // too when a device fill left it stale (the d6 diagnostic reads host T).
  sync_app_data_to_host();
  if (host_T_stale) sync_temperature_to_host();

  // Call parent class stats implementation
  AppPottsQuaternionKokkos::stats(strtmp);
}

/* ----------------------------------------------------------------------
   Process input commands
------------------------------------------------------------------------- */

void AppAdditiveTextureKokkos::input_app(char *command, int narg, char **arg)
{
  // Keep the thermal snapshot current before any thermal command handler
  // (e.g. laser_path reads host->time/dx when building the scan path).
  refresh_thermal_host();

  // floor_check d: stop if liquid reaches within d (m) of the box bottom. For
  // box-per-layer stitch builds: material below the box cannot melt, so a pool
  // reaching the floor is silently cut off. Error before the run's stitch dump.
  if (strcmp(command, "floor_check") == 0) {
    if (narg != 1) error->all(FLERR, "Illegal floor_check command: floor_check d");
    floor_check_d = atof(arg[0]);
    if (floor_check_d < 0.0) error->all(FLERR, "floor_check distance must be >= 0");
    return;
  }

  if (!parse_command(command, narg, arg, error, screen, domain->me, *thermal))
    AppPottsQuaternionKokkos::input_app(command, narg, arg);
}
