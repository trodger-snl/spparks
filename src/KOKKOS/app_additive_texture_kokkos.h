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

#ifdef APP_CLASS
AppStyle(additive/texture/kk,AppAdditiveTextureKokkos)

#else

#ifndef SPK_APP_ADDITIVE_TEXTURE_KOKKOS_H
#define SPK_APP_ADDITIVE_TEXTURE_KOKKOS_H

#include "app_potts_quaternion_kokkos.h"
#include "temperature_source.h"
#include "thermal_manager.h"
#include "thermal_host.h"
#include "additive_params.h"
#include "rosenthal_device.h"
#include "moser_device.h"
#include "hdf5_device.h"
#include "conduction_solver_kokkos.h"
#include "math_const.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace SPPARKS_NS {

// Counter-keyed RNG for the kernels (splitmix64, as the CPU app's step_rng).
// A site's draws in one kernel and step depend only on the key (deck seed,
// step, rank), the site's index and the kernel, never on which thread runs
// it or on what other sites drew, so a run repeats bit for bit at a fixed
// seed and rank count on any backend and thread count, and a void changes
// the microstructure only through the physics around it.
struct KeyedRNG {
  uint64_t s;
  KOKKOS_INLINE_FUNCTION static uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  KOKKOS_INLINE_FUNCTION KeyedRNG(uint64_t key, int i, int kernel)
    : s(mix(key ^ mix(((uint64_t)(uint32_t) i << 8) + (uint64_t) kernel + 0x9E3779B97F4A7C15ULL))) {}
  KOKKOS_INLINE_FUNCTION double drand() {   // (0,1): never 0, so log(drand()) is finite
    return ((mix(s += 0x9E3779B97F4A7C15ULL) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
};
// Where the baseplate is (conduction plate_material): below zplate (m) a site
// melts and freezes at the plate's liquidus and solidifies back to i2 = 6.
// Without a plate (on = false) every site takes tl and solidifies to 3.
// A flag, not an infinite zplate: the build uses -ffast-math.
struct PlateRule {
  bool on;
  double zplate, dx, tl, tl_plate;
  KOKKOS_INLINE_FUNCTION bool in(double z) const { return on && z * dx < zplate; }
  KOKKOS_INLINE_FUNCTION double tl_at(double z) const { return in(z) ? tl_plate : tl; }
  KOKKOS_INLINE_FUNCTION int solid(double z) const { return in(z) ? 6 : 3; }
};

enum { RNG_MELT = 1, RNG_NUCLEATE, RNG_FLIP, RNG_SOLIDIFY, RNG_SMOOTH,
       RNG_COLORS_SOLIDIFY, RNG_COLORS_SMOOTH };

class AppAdditiveTextureKokkos : public AppPottsQuaternionKokkos, public AdditiveParams {
 public:
  AppAdditiveTextureKokkos(class SPPARKS *, int, char **);
  virtual ~AppAdditiveTextureKokkos();

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;
  virtual void app_update(double) override;
  virtual void input_app(char *, int, char **) override;
  virtual void stats(char *) override;

  // Override required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  // Back to the full upload: the quaternion parent's ghost-range version
  // moves only spin/quaternions and would drop the AM arrays.
  void sync_ghosts_to_device() override { sync_app_data_to_device(); }
  virtual void allocate_app_views() override;

  // Temperature field management
  void refresh_thermal_host();
  void sync_temperature_to_device();
  void sync_temperature_to_host();   // device d_temp -> host T (for diag/host code)
  // Phase 2 selective ghost sync: move only neighbor-read MC state
  // (spin/active_flag/q0-3) device<->host raw arrays (owned out, ghost in).
  void sync_ghost_to_host();
  void sync_ghost_to_device();
  double global_max_temperature_from_fill();  // blocking Allreduce of the fill-fused owned max
  // One-step-stale variant for the HDF5 path at nprocs>1: posts a
  // non-blocking Iallreduce for THIS fill and returns the PREVIOUS fill's
  // completed value, so the collective overlaps the microstructure kernels
  // instead of blocking on the slowest rank every step. See the impl.
  double global_max_temperature_from_fill_stale();
  void   ff_finish_pending();   // complete any outstanding Iallreduce

  // Ensure the cached device copy of site coordinates (d_xyz) is built for the
  // current site count. The host xyz is row-major [nall][3]; this fills d_xyz
  // by explicit element copy so it is correct regardless of the device View
  // layout (a LayoutLeft device view would otherwise misread row-major data).
  void ensure_device_xyz();

  // Device-side analytical Rosenthal temperature: evaluate the per-site
  // temperature directly into d_temp on the GPU from the host-advanced laser pose,
  // replacing the host per-site exp/sqrt loop. Fills all nlocal+nghost sites
  // (Rosenthal is positional, so ghosts compute their own T -- no ghost comm).
  void fill_temperature_rosenthal_device(const RosenthalDevicePlan &plan);

  // Device-side Moser (time-domain Green's function) temperature: integrate the
  // scan-path history per site on the GPU. STANDARD mode, no fluctuations.
  void fill_temperature_moser_device(const MoserDevicePlan &plan);

  // Device-side HDF5 unstructured-mesh temperature: the host precomputes the
  // per-site element cache (static per layer) and the per-node temperature
  // slice (per step); the GPU does the per-site barycentric interpolation
  // straight into d_temp, reproducing get_temperature_at_site_fast() bit-for-bit.
  // Fills ALL nall sites (owned + ghost) positionally -- the cache is built over
  // nall via the chunk-load pad, so ghosts compute their own T with no host
  // round-trip and no T ghost comm (a ghost equals the owner's value by
  // construction). See the impl and hdf5_device.h.
  void fill_temperature_hdf5_device(const HDF5DevicePlan &plan);

  // Phase 3: Phase transition kernels
  void initialize_nucleation_parameters();
  int melting_kernel(Kokkos::View<int*> d_active_sites, int n_active);
  void check_melt_floor(int n_active);
  int nucleation_kernel(Kokkos::View<int*> d_active_sites, int n_active);
  int particle_flipper_kernel(Kokkos::View<int*> d_active_sites, int n_active);
  int solidification_kernel(Kokkos::View<int*> d_active_sites, int n_active);
  int smoothing_kernel_efficient(Kokkos::View<int*> d_smoothing_sites, int n_sites);

  // Powder phase activation
  void activate_powder_sites();

 protected:
  // Temperature field (darray d6). Written by the device fills, or by the
  // host fill + sync_temperature_to_device.
  Kokkos::View<double*> d_temp;

  // True when a device temperature fill has made the host T (darray d6) stale.
  // The copy-back is deferred (lazy): host T is only refreshed before a host
  // consumer reads it -- powder activation, or the d6 diagnostic at an output
  // step. See app_update().
  bool host_T_stale = false;

  // Peak temperature of the LAST device fill over OWNED sites. The Max
  // reduction is FUSED into the fill kernels themselves (RosenthalFill /
  // MoserFill / HDF5Fill fold it in while writing d_temp), replacing the
  // separate per-step "Tmax" parallel_reduce over nlocal + its stream sync
  // (~9% of box runtime on gfx1151). Consumed by
  // global_max_temperature_from_fill() for the fast-forward and Moser
  // pause/reset decisions; only valid immediately after a fill.
  double fill_maxT_local = 0.0;

  // One-step-stale fast-forward state (HDF5 path, nprocs>1 only). The peak-T
  // Allreduce is a per-step collective whose cost is almost entirely waiting
  // for the slowest rank: measured at 18.6% of the loop at np=8 vs 0.0% at
  // np=1. Posting it non-blocking and consuming the PREVIOUS step's result
  // removes that sync point. Safe here because hdf5_advance_device uses the
  // value only as the boolean (gmax <= threshold): a stale-cold reading can
  // at worst delay a fast-forward one step, and a stale-hot reading is caught
  // by the source's own get_next_time_with_temperature() query, which returns
  // a target <= current time and so skips nothing. NOT applied to the Moser
  // path, where this value also drives pause/reset pass firing.
  double      ff_gmax_prev = 0.0;   // last COMPLETED global max
  double      ff_ia_send = 0.0;     // Iallreduce buffers (must outlive the call)
  double      ff_ia_recv = 0.0;
  MPI_Request ff_ia_req = MPI_REQUEST_NULL;
  bool        ff_ia_active = false;
  bool        ff_gmax_primed = false;  // false until the first result lands

  // Cached device copy of site coordinates (lattice units). The lattice is
  // static for the run, so this is uploaded once and reused by the device
  // Rosenthal fill (rebuilt only if the site count changes via grow_app).
  Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz;
  int d_xyz_n = -1;                 // extent d_xyz was built for (nlocal+nghost)

  // Rosenthal KEYHOLE line-integral quadrature (fixed at setup; uploaded once).
  Kokkos::View<double*> d_ros_qnodes, d_ros_qweights;
  int d_ros_nquad = -1;

  // Device Moser support: scan segment arrays (re-uploaded when the scan
  // version changes) and a precomputed Gauss-Legendre roots/weights table
  // (orders 2..MOSER_GL_MAX_ORDER; order 1 is the midpoint rule). See
  // fill_temperature_moser_device().
  static constexpr int MOSER_GL_MAX_ORDER = 30;
  // Shared path geometry (length nseg) + per-lobe powers (length n_lobes*nseg,
  // lobe-major) + per-lobe fluctuation tables (concatenated, length sum fluct_n).
  Kokkos::View<double*> d_seg_bounds, d_seg_xl, d_seg_yl, d_seg_powers;
  Kokkos::View<double*> d_seg_nx, d_seg_ny, d_seg_speeds;
  Kokkos::View<double*> d_fluct_t, d_fluct_w, d_fluct_d, d_fluct_p;
  long d_seg_version = -1;          // scan version the device Views currently hold
  Kokkos::View<double*> d_gl_roots, d_gl_weights;  // flat tables, orders 2..30
  Kokkos::View<int*>    d_gl_offset;               // per-order start index (size 31)
  bool d_gl_built = false;
  void build_gl_table();            // host-build + upload the GL table once

  // Device HDF5 support. The per-site element cache (4 tet-node indices + 3
  // barycentric weights + a validity flag, length nlocal) is STATIC within a
  // layer -- re-uploaded only when the host cache generation changes. The
  // per-node temperature slice (length n_nodes) is re-uploaded every step.
  // See fill_temperature_hdf5_device().
  // KK_FLOAT (spktype.h) is double by default, float under
  // KOKKOS_PREC=mixed/single: this kernel is gather-bound, so narrowing the streamed
  // arrays is the win. Node indices stay int and TIMES stay double -- see the
  // header for why d_ts_times must not be narrowed.
  Kokkos::View<int*[4]>      d_hdf5_node_idx;   // [nlocal] tet-node indices into nodal slice
  Kokkos::View<KK_FLOAT*[3]> d_hdf5_weights;    // [nlocal] barycentric weights
  Kokkos::View<int*>         d_hdf5_valid;      // [nlocal] 1 = has containing element
  Kokkos::View<KK_FLOAT*>    d_hdf5_nodal_temps;// [n_nodes] per-node T for the step
  long d_hdf5_cache_version = -1;   // cache generation the static Views hold
  int  d_hdf5_nsites = -1;          // site count (nall) the static Views were sized for

  // Device-side time interpolation: the per-node raw time-series CSR, uploaded
  // once per layer (static within a layer). The fill binary-searches each tet
  // node's series at the current time + linear-interpolates (bit-matching the
  // host precompute), replacing the per-step host precompute + nodal-slice
  // upload. Only the active nodes' rows are uploaded; times/temps share one
  // offset array. On an integrated GPU the two series Views instead wrap the
  // host series in place (SeriesMemory) and the offsets index it directly.
  Kokkos::View<double*>   d_ts_times; // flat per-node sample times -- ALWAYS
                                      // double: absolute sim times ~3.9e4 s on
                                      // a 3e-6 s step; float would quantize to
                                      // ~4e-3 s and break the search bracket
  Kokkos::View<KK_FLOAT*> d_ts_temps; // flat per-node sample temperatures
  Kokkos::View<int64_t*[2]> d_ts_offsets; // [n_active] {start,end}; row k = node d_active_nodes(k)
  Kokkos::View<int*>    d_active_nodes;  // the ~33k nodes the blend reads (interp only these)
  int  d_n_active_nodes = 0;
  long d_ts_version = -1;             // cache generation the time-series Views hold
  bool device_time_interp_set = false;
  double floor_check_d = 0.0;   // floor_check: error if liquid comes within this (m) of the box bottom
  double floor_zmin_run = 1.0e300;  // deepest liquid z so far (lattice units), printed as it drops

  // Gradient cache (device-only, computed once per timestep)
  // Uses simple View because computed on device, no host sync needed
  Kokkos::View<double*> d_G;        // Gradient magnitude
  Kokkos::View<double*> d_V;        // Solidification velocity
  Kokkos::View<double*> d_spath;    // inheritance path length (d10)
  Kokkos::View<double*> d_mobility_out; // 1 once a site has been smoothed (d5), as on the CPU

  // Phase state arrays (device views - simple Views are race-safe for these kernels)
  Kokkos::View<int*> d_active_flag;      // 0=inactive, 1=active, 2=molten, 3=solidified, 5=void, 6=plate
  // Per-step random order of the 8 parity colours used by the solidification
  // and smoothing kernels (multicolour Gauss-Seidel; see the .cpp).
  void shuffled_colors(int order[8], int kernel);
  PlateRule plate_rule() const;   // z in lattice units, as in d_xyz
  // This step's RNG key (KeyedRNG): the deck seed, the step and the rank.
  uint64_t rng_key() const;
  uint64_t rng_seed = 0;          // the deck seed
  uint64_t rng_step = 0;          // app_update calls so far
  Kokkos::View<int*> d_nuclei;    // this step's nuclei, in site order (particle_flipper_kernel)
  Kokkos::View<double*> d_solid_d;       // Solid front distance

  // Compact list of sites needing smoothing, rebuilt each step by a parallel_scan.
  // Cached (sized to nlocal) so it is not reallocated on the GPU every timestep.
  Kokkos::View<int*> d_smoothing_sites;

  // Compact "active-front" list: the union superset of every owned site any
  // microstructure kernel (melt/nucleate/particle/solidify/smooth) could touch
  // this step. Rebuilt each step by a parallel_scan; the phase kernels run over
  // this list (size n_active, the thin melt/solidification shell) instead of
  // nlocal (up to 5M). Cached (sized to nlocal) so it is not reallocated per step.
  Kokkos::View<int*> d_active_sites;

  // Full fusion of the active-front build into the temperature fill: the device
  // fill already visits every site and has T in a register, so it evaluates the
  // active predicate and ATOMIC-APPENDS active owned sites straight into
  // d_active_sites (with d_active_count as the append counter) -- eliminating the
  // separate O(nlocal) compaction scan entirely (the scan is a multi-pass prefix
  // sum that is the GPU's #2 cost). device_active_list_fresh is set by a fill
  // that built the list this step (HDF5); when false (Rosenthal/Moser/host
  // fills), app_update falls back to the full predicate scan. The appended order
  // is non-deterministic (atomic order), which is fine -- each microstructure
  // kernel processes its sites independently and temperature is untouched.
  Kokkos::View<int> d_active_count;
  // Persistent host mirror for the count readback. app_update reads this every
  // step; a per-step create_mirror_view would allocate (and on HIP register) a
  // fresh host buffer each time for a single int.
  Kokkos::View<int, Kokkos::HostSpace> h_active_count{"h_active_count"};
  bool device_active_list_fresh = false;

  // Per-spin nucleation arrays (device-accessible, initialized once)
  Kokkos::View<int*> d_nucleation_flags;     // Can this spin nucleate? (1=yes, 0=no)
  Kokkos::View<double*> d_nucleation_temps;  // Critical undercooling for nucleation (K)
  Kokkos::View<double*> d_nucleation_sizes;  // Nucleus volume (m^3)

  // Global parameter arrays (device-accessible constants)
  Kokkos::View<double*> d_solid_front_coeffs;  // Solidification velocity polynomial coefficients

  // Host pointers (mapped to iarray/darray like CPU version)
  int *active_flag;
  double *mobility_out;
  double *T;
  double *solid_d;
  double *G;
  double *V;
  double *spath;   // d10: inheritance path length, lattice units

  // Temperature-I/O orchestration: shared with the CPU app. ThermalManager owns
  // the temperature_source, laser path, fast-forward predictors, and smoothing;
  // it reads/writes the lattice state through the ThermalHost snapshot, which
  // refresh_thermal_host() populates each step. See thermal_manager.h.
  std::unique_ptr<ThermalManager> thermal;
  std::unique_ptr<ConductionSolverKokkos> cond;   // conduction source state
  ThermalHost thermal_host_;

  // Physical parameters (same as CPU version)
  double dt;
  double dx;
  double t_room;

  // Nucleation parameters
  int nrefine;

  // misorientation_diffusion: the new-grain test here uses the direct
  // quaternion angle, the CPU app the symmetry-reduced disorientation.
  // Device cache of the persistent (rho > 0) grain-walk recurrence: rotation
  // vector per (compact grain id, level), filled lazily and incrementally by
  // the capture kernels (identical values on concurrent fills, so the races
  // are benign; fences order the table writes before the ready counter).
  // Grains beyond gw_maxgid or levels beyond gw_maxlev fall back to the
  // direct O(k) recurrence.
  int gw_maxgid = 16384, gw_maxlev = 768;
  Kokkos::View<int*> d_gw_gid;        // spin -> compact grain id (-1 unassigned), nspins+1
  Kokkos::View<int*> d_gw_ngid;       // [1] next free id
  Kokkos::View<int*> d_gw_nready;     // per gid: levels 0..nready-1 are in the table
  Kokkos::View<double*[3]> d_gw_tab;  // gid*maxlev + k -> rotation vector
  double t_cool_max;   // tl - ts, set in init_app (was a hardcoded 100 K)

  // Per-hop rotation angle of the walk: sqrt(D_eff * hop_m). The sqrt(hop)
  // scaling keeps RMS^2 = D * distance independent of lattice spacing.
  KOKKOS_INLINE_FUNCTION
  static double diffusion_angle(double D, double alpha, double t_cool_max,
                                double Tcool, double hop_m) {
    if (alpha > 0.0) {
      const double x = Kokkos::fmin(Kokkos::fmax(Tcool / t_cool_max, 0.0), 1.0);
      const double g = (Kokkos::exp(alpha * x) - 1.0) / (Kokkos::exp(alpha) - 1.0);
      D *= g * g;
    }
    return Kokkos::sqrt(D * hop_m);
  }

  // Nucleation data
  int *nucleation_flags;
  double *nucleation_temps;
  double *nucleation_sizes;

  // Powder activation tracking
  double last_powder_activation_time;  // Track when we last activated powder sites

  // Device-compatible thermal operations
  //
  // LATTICE TYPE ASSUMPTION: Simple Cubic with 26 Neighbors (sc/26n)
  // This implementation assumes a simple cubic lattice with 26 neighbors
  // (3×3×3 cube minus center). Neighbor offsets are taken from the actual site
  // coordinates (not a fixed stencil table indexed by neighbor slot), so the
  // gradient is correct for both interior and low-coordination boundary sites;
  // the sqrt(3) solid-front reach in the solidification kernel is the SC_26N
  // cube-corner distance. For other lattices these would need revisiting.
  //
  // Hop between two lattice sites in lattice units, clamped at the SC_26N
  // cube-corner distance sqrt(3) so a periodic-wrap neighbour pair never
  // reads as a box-length hop.
  KOKKOS_INLINE_FUNCTION
  static double hop_distance(const Kokkos::View<double*[3], Kokkos::LayoutRight> &xyz, int i, int j) {
    const double hdx = xyz(i, 0) - xyz(j, 0);
    const double hdy = xyz(i, 1) - xyz(j, 1);
    const double hdz = xyz(i, 2) - xyz(j, 2);
    return Kokkos::fmin(Kokkos::sqrt(hdx * hdx + hdy * hdy + hdz * hdz),
                        1.7320508075688772);
  }

  // Direct rotation angle between two unit quaternions (double cover folded).
  KOKKOS_INLINE_FUNCTION
  static double direct_angle(const double *a, const double *b) {
    double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    dot = Kokkos::fmin(1.0, Kokkos::fabs(dot));
    return 2.0 * Kokkos::acos(dot);
  }

  // Device twin of AdditiveParams::front_velocity: n == 0 is the power law
  // a * Tcool^p, otherwise the polynomial coeffs (highest power first).
  KOKKOS_INLINE_FUNCTION
  static double solid_front_velocity(const Kokkos::View<double*> &coeffs, int n,
                                     double a, double p, double Tcool) {
    if (n == 0) return a * Kokkos::pow(Tcool, p);
    double V = 0.0, Tcool_power = 1.0;
    for (int k = n - 1; k >= 0; k--) {
      V += coeffs(k) * Tcool_power;
      Tcool_power *= Tcool;
    }
    return V;
  }

  // Perturb orientation q by `angle` per a misorientation-target mode:
  // RANDOM_ROTATION composes a rotation about a uniformly random axis;
  // RANDOM rotates the nearest <100> toward a uniformly random unit vector
  // (same z/phi sampling as AppAdditiveTexture's sample_random_unit_vector).
  // Two RNG draws either way.
  template<typename Gen>
  KOKKOS_INLINE_FUNCTION
  static void perturb_orientation(int mode, const double *q,
                                  double angle, Gen &gen, double *out) {
    if (mode == MISORI_TARGET_RANDOM_ROTATION) {
      const double z = 2.0 * gen.drand() - 1.0;
      const double phi = 2.0 * MathConst::MY_PI * gen.drand();
      const double radial = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - z * z));
      const double half = 0.5 * angle;
      const double s = Kokkos::sin(half);
      const double dq[4] = {Kokkos::cos(half),
                            s * radial * Kokkos::cos(phi),
                            s * radial * Kokkos::sin(phi),
                            s * z};
      QuaternionOps::composition(dq, q, out);
      QuaternionOps::normalize(out);
      return;
    }
    // MISORI_TARGET_RANDOM
    double dir[3];
    {
      const double z = 2.0 * gen.drand() - 1.0;
      const double phi = 2.0 * MathConst::MY_PI * gen.drand();
      const double radial = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - z * z));
      dir[0] = radial * Kokkos::cos(phi);
      dir[1] = radial * Kokkos::sin(phi);
      dir[2] = z;
    }
    QuaternionOps::rotate_q_towards_u(q, dir, angle, out);
  }

  struct ThermalOps {
    // Compute temperature gradient at site i using weighted least-squares
    //
    // Algorithm: For each of 26 cardinal neighbors, build weighted least
    // squares system A*grad = b where A is [dx dy dz] and b is temperature
    // difference. Solve 3x3 system using Cramer's rule.
    //
    // CRITICAL: Matches CPU version's two-path logic for melt surface handling
    //
    // Args:
    //   i: Site index (0 <= i < nlocal)
    //   d_T: Temperature field (device view, read-only)
    //   d_active_flag: Active flag array (0=inactive, 1=active, 2=molten, 3=solid, 5=void)
    //   numneigh: Number of neighbors per site
    //   neighbor: Neighbor list [site][neighbor_index] -> neighbor_site
    //   xyz: Site coordinates [site][dimension]
    //   dx_lattice: Lattice spacing in physical units (meters)
    //   grad_out: Output gradient vector [3] (normalized direction)
    //   magnitude_out: Output gradient magnitude (K/m)
    //
    // Returns: void (modifies grad_out and magnitude_out)
    //
    // Thread safety: Read-only access to inputs, writes to separate output
    // pointers per thread (safe for parallel_for)
    //
    KOKKOS_INLINE_FUNCTION
    static void compute_temperature_gradient(
        int i,
        const Kokkos::View<const double*>& d_T,
        const Kokkos::View<const int*>& d_active_flag,
        const Kokkos::View<const int*>& numneigh,
        const Kokkos::View<int**, Kokkos::LayoutRight>& neighbor,
        const Kokkos::View<double*[3], Kokkos::LayoutRight>& xyz,
        double dx_lattice,
        double* grad_out,
        double* magnitude_out) {

      // Cache neighbor temperatures in registers for performance
      const int nn = numneigh(i);
      double T_neigh[26];
      for (int j = 0; j < nn && j < 26; j++) {
        T_neigh[j] = d_T(neighbor((size_t) i, j));
      }

      // Detect melt surface: the neighbor directly above (+z) is missing (a
      // top-boundary site) or inactive/active (active_flag <= 1), as in the
      // CPU normal_finder(). Found by coordinates, not slot 13: a boundary
      // site's neighbor list is compacted, so slot 13 need not be +z.
      int above = -1;
      for (int j = 0; j < nn && j < 26; j++) {
        const int nb = neighbor((size_t) i, j);
        if (xyz(nb, 0) == xyz(i, 0) && xyz(nb, 1) == xyz(i, 1) &&
            xyz(nb, 2) == xyz(i, 2) + 1.0) { above = nb; break; }
      }
      // A void above (i2 = 5) is air, so it is a free surface too.
      bool is_melt_surface = (above < 0) || (d_active_flag(above) <= 1) ||
                             (d_active_flag(above) == 5);

      // Get current site z-coordinate for melt surface filtering
      double site_z = xyz(i, 2);

      // Build weighted least squares matrices
      // A^T * W * A (3x3 symmetric matrix, store as 9 elements)
      double AtA[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      // A^T * W * b (3x1 vector)
      double Atb[3] = {0.0, 0.0, 0.0};

      double T_center = d_T(i);

      // Accumulate weighted least squares matrices
      for (int n = 0; n < nn && n < 26; n++) {
        int nbr = neighbor((size_t) i, n);

        // CRITICAL FIX: At melt surface, only use neighbors at or below current z
        // This matches CPU version lines 1211-1212
        if (is_melt_surface && xyz(nbr, 2) > site_z) {
          continue;  // Skip neighbors above current site
        }

        // Neighbor offset from the actual site coordinates (lattice units * dx).
        // Do NOT index a fixed stencil table by the neighbor slot n: low-
        // coordination global-boundary sites have a COMPACTED neighbor list, so
        // slot n is the n-th existing neighbor, not the n-th SC_26N stencil
        // position -- a table lookup would attach the wrong geometry. The
        // coordinate difference is exact for interior and boundary sites alike
        // (identical to the old offset_map for full-coordination sites).
        double dx_n = (xyz(nbr,0) - xyz(i,0)) * dx_lattice;
        double dy_n = (xyz(nbr,1) - xyz(i,1)) * dx_lattice;
        double dz_n = (xyz(nbr,2) - xyz(i,2)) * dx_lattice;

        // Temperature difference
        double dT = T_neigh[n] - T_center;

        // Distance-based weight (inverse distance squared)
        double dist = Kokkos::sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
        double weight = 1.0 / (dist*dist + 1.0e-12);

        // Accumulate normal equations: A^T * W * A
        AtA[0] += weight * dx_n * dx_n;  // (0,0) xx
        AtA[1] += weight * dx_n * dy_n;  // (0,1) xy
        AtA[2] += weight * dx_n * dz_n;  // (0,2) xz
        AtA[3] += weight * dy_n * dx_n;  // (1,0) yx
        AtA[4] += weight * dy_n * dy_n;  // (1,1) yy
        AtA[5] += weight * dy_n * dz_n;  // (1,2) yz
        AtA[6] += weight * dz_n * dx_n;  // (2,0) zx
        AtA[7] += weight * dz_n * dy_n;  // (2,1) zy
        AtA[8] += weight * dz_n * dz_n;  // (2,2) zz

        // Accumulate A^T * W * b
        Atb[0] += weight * dx_n * dT;
        Atb[1] += weight * dy_n * dT;
        Atb[2] += weight * dz_n * dT;
      }

      // Solve 3x3 system using Cramer's rule: grad = (A^T*W*A)^(-1) * (A^T*W*b)
      // Compute determinant
      double det = AtA[0] * (AtA[4] * AtA[8] - AtA[5] * AtA[7])
                 - AtA[1] * (AtA[3] * AtA[8] - AtA[5] * AtA[6])
                 + AtA[2] * (AtA[3] * AtA[7] - AtA[4] * AtA[6]);

      // Check for singular matrix (shouldn't happen with valid neighbors)
      if (Kokkos::fabs(det) < 1.0e-12) {
        grad_out[0] = 0.0;
        grad_out[1] = 0.0;
        grad_out[2] = 0.0;
        *magnitude_out = 0.0;
        return;
      }

      double inv_det = 1.0 / det;

      // Solve for gradient using Cramer's rule
      // grad_x = det([Atb[0] AtA[1] AtA[2]; Atb[1] AtA[4] AtA[5]; Atb[2] AtA[7] AtA[8]]) / det
      double grad_x = (Atb[0] * (AtA[4] * AtA[8] - AtA[5] * AtA[7])
                     - AtA[1] * (Atb[1] * AtA[8] - AtA[5] * Atb[2])
                     + AtA[2] * (Atb[1] * AtA[7] - AtA[4] * Atb[2])) * inv_det;

      // grad_y = det([AtA[0] Atb[0] AtA[2]; AtA[3] Atb[1] AtA[5]; AtA[6] Atb[2] AtA[8]]) / det
      double grad_y = (AtA[0] * (Atb[1] * AtA[8] - AtA[5] * Atb[2])
                     - Atb[0] * (AtA[3] * AtA[8] - AtA[5] * AtA[6])
                     + AtA[2] * (AtA[3] * Atb[2] - Atb[1] * AtA[6])) * inv_det;

      // grad_z = det([AtA[0] AtA[1] Atb[0]; AtA[3] AtA[4] Atb[1]; AtA[6] AtA[7] Atb[2]]) / det
      double grad_z = (AtA[0] * (AtA[4] * Atb[2] - Atb[1] * AtA[7])
                     - AtA[1] * (AtA[3] * Atb[2] - Atb[1] * AtA[6])
                     + Atb[0] * (AtA[3] * AtA[7] - AtA[4] * AtA[6])) * inv_det;

      // Compute magnitude
      double norm = Kokkos::sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
      *magnitude_out = norm;

      // Normalize gradient direction (with absolute values like CPU version)
      if (norm > 1.0e-12) {
        grad_out[0] = Kokkos::fabs(grad_x) / norm;
        grad_out[1] = Kokkos::fabs(grad_y) / norm;
        grad_out[2] = Kokkos::fabs(grad_z) / norm;
      } else {
        grad_out[0] = 0.0;
        grad_out[1] = 0.0;
        grad_out[2] = 0.0;
      }
    }
  };
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPPARKS to see the offending
line.

E: One or more sites have invalid values

The application only allows sites to be initialized with specific
values.

*/
