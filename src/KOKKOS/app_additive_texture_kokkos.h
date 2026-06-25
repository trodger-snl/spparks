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
#include "rosenthal_device.h"
#include "moser_device.h"
#include "hdf5_device.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <memory>

namespace SPPARKS_NS {

class AppAdditiveTextureKokkos : public AppPottsQuaternionKokkos {
 public:
  AppAdditiveTextureKokkos(class SPPARKS *, int, char **);
  virtual ~AppAdditiveTextureKokkos();

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;
  virtual void app_update(double) override;
  virtual void input_app(char *, int, char **) override;
  virtual void stats(char *) override;
  virtual void stats_header(char *) override;

  // Override required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  virtual void allocate_app_views() override;
  virtual void sweep_color_kokkos(int color, int start, int count) override;

  // Temperature field management
  void refresh_thermal_host();
  void update_temperature_from_source(double simulation_time);
  void sync_temperature_to_device();
  void sync_temperature_to_host();   // device k_T -> host T (for diag/host code)
  void sync_owned_temperature_to_host();  // device k_T[0,nlocal) -> host T (ghosts kept)
  double device_global_max_temperature();  // parallel_reduce max over k_T + Allreduce
  void compute_gradients_kernel();

  // Ensure the cached device copy of site coordinates (d_xyz) is built for the
  // current site count. The host xyz is row-major [nall][3]; this fills d_xyz
  // by explicit element copy so it is correct regardless of the device View
  // layout (a LayoutLeft device view would otherwise misread row-major data).
  void ensure_device_xyz();

  // Device-side analytical Rosenthal temperature: evaluate the per-site
  // temperature directly into k_T on the GPU from the host-advanced laser pose,
  // replacing the host per-site exp/sqrt loop. Fills all nlocal+nghost sites
  // (Rosenthal is positional, so ghosts compute their own T -- no ghost comm).
  void fill_temperature_rosenthal_device(const RosenthalDevicePlan &plan);

  // Device-side Moser (time-domain Green's function) temperature: integrate the
  // scan-path history per site on the GPU. STANDARD mode, no fluctuations.
  void fill_temperature_moser_device(const MoserDevicePlan &plan);

  // Device-side HDF5 unstructured-mesh temperature: the host precomputes the
  // per-site element cache (static per layer) and the per-node temperature
  // slice (per step); the GPU does the per-site barycentric interpolation
  // straight into k_T, reproducing get_temperature_at_site_fast() bit-for-bit.
  // Fills OWNED sites only (k_T[0,nlocal)); ghost T (which may lie outside this
  // rank's loaded mesh) is copied from the host T ghost slice -- see the impl.
  void fill_temperature_hdf5_device(const HDF5DevicePlan &plan);

  // Phase 3: Phase transition kernels
  void initialize_nucleation_parameters();
  int melting_kernel();
  int nucleation_kernel();
  int particle_flipper_kernel();
  int solidification_kernel();
  int smoothing_kernel();
  int smoothing_kernel_efficient(Kokkos::View<int*> d_smoothing_sites, int n_sites);

  // Powder phase activation
  void activate_powder_sites();

 protected:
  // Temperature fields (CPU-updated from HDF5, device-readable)
  // Uses DualView because temperature is read-only during device sweeps
  Kokkos::DualView<double*> k_T;  // Temperature field

  // True when a device temperature fill has made the host T (darray d6) stale.
  // The copy-back is deferred (lazy): host T is only refreshed before a host
  // consumer reads it -- powder activation, or the d6 diagnostic at an output
  // step. See app_update().
  bool host_T_stale = false;

  // Cached device copy of site coordinates (lattice units). The lattice is
  // static for the run, so this is uploaded once and reused by the device
  // Rosenthal fill (rebuilt only if the site count changes via grow_app).
  Kokkos::View<double*[3]> d_xyz;
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
  Kokkos::View<int*[4]>    d_hdf5_node_idx;   // [nlocal] tet-node indices into nodal slice
  Kokkos::View<double*[3]> d_hdf5_weights;    // [nlocal] barycentric weights
  Kokkos::View<int*>       d_hdf5_valid;      // [nlocal] 1 = has containing element
  Kokkos::View<double*>    d_hdf5_nodal_temps;// [n_nodes] per-node T for the step
  long d_hdf5_cache_version = -1;   // cache generation the static Views hold
  int  d_hdf5_nlocal = -1;          // nlocal the static Views were sized for

  // Gradient cache (device-only, computed once per timestep)
  // Uses simple View because computed on device, no host sync needed
  Kokkos::View<double*> d_G;        // Gradient magnitude
  Kokkos::View<double*> d_grad_x;   // Gradient x-component
  Kokkos::View<double*> d_grad_y;   // Gradient y-component
  Kokkos::View<double*> d_grad_z;   // Gradient z-component
  Kokkos::View<double*> d_V;        // Solidification velocity

  // Phase state arrays (device views - simple Views are race-safe for these kernels)
  Kokkos::View<int*> d_active_flag;      // 0=inactive, 1=active, 2=molten, 3=solidified, 5=void
  Kokkos::View<double*> d_mobility_out;  // Arrhenius mobility
  Kokkos::View<double*> d_solid_d;       // Solid front distance

  // Compact list of sites needing smoothing, rebuilt each step by a parallel_scan.
  // Cached (sized to nlocal) so it is not reallocated on the GPU every timestep.
  Kokkos::View<int*> d_smoothing_sites;

  // Per-spin nucleation arrays (device-accessible, initialized once)
  Kokkos::View<int*> d_nucleation_flags;     // Can this spin nucleate? (1=yes, 0=no)
  Kokkos::View<double*> d_nucleation_temps;  // Critical undercooling for nucleation (K)
  Kokkos::View<double*> d_nucleation_sizes;  // Nucleus volume (m^3)

  // Global parameter arrays (device-accessible constants)
  Kokkos::View<double*> d_solid_front_coeffs;  // Solidification velocity polynomial coefficients
  Kokkos::View<double[3]> d_texture_params;    // Texture model parameters: [c1, c2, c3]

  // Host pointers (mapped to iarray/darray like CPU version)
  int *active_flag;
  double *mobility_out;
  double *T;
  double *solid_d;
  double *G;
  double *V;
  double *neigh_dist;
  double *unique_dot;

  // Temperature-I/O orchestration: shared with the CPU app. ThermalManager owns
  // the temperature_source, laser path, fast-forward predictors, and smoothing;
  // it reads/writes the lattice state through the ThermalHost snapshot, which
  // refresh_thermal_host() populates each step. See thermal_manager.h.
  std::unique_ptr<ThermalManager> thermal;
  ThermalHost thermal_host_;
  bool use_temperature_source;        // vestigial; thermal->enabled() is canonical
  double fast_forward_search_window;

  // Index-based selective ghost communication (used by ThermalManager's
  // optional temperature-smoothing / diagnostics paths). Null/0 disables it,
  // which is the default for the Kokkos app (device buffers carry the KMC
  // ghost exchange); set when host-side temperature smoothing is enabled.
  const int *ghost_iindices = nullptr;
  int nghost_iarray = 0;
  const int *ghost_dindices = nullptr;
  int nghost_darray = 0;

  // Physical parameters (same as CPU version)
  double time_step;
  double dt;
  double dx;
  double tl;  // Liquidus temperature
  double ts;  // Solidus temperature
  double t_room;

  // Nucleation parameters
  int nrefine;
  double no;
  double tc;
  double tsig;
  double size_norm;
  double size_sig;
  double *solid_front_coeffs;
  int solid_front_length;

  // Texture model parameters
  double c1, c2, c3;
  double max_misorient;
  double t_cool_max;
  double mis_thresh;
  double misorient_alpha;

  // Nucleation data
  int *nucleation_flags;
  double *nucleation_temps;
  double *nucleation_sizes;

  // Void parameters
  double void_density;
  double void_pore_fraction;
  double void_radius_mean;
  double void_radius_std;
  double void_radius_min;
  double void_radius_max;
  int enable_voids;

  int t_active;

  // Powder activation tracking
  double last_powder_activation_time;  // Track when we last activated powder sites

  // Debug counters for gradient/velocity analysis
  int debug_nucleation_count;
  int debug_solidification_count;
  int debug_zero_G_count;
  int debug_nonzero_G_count;
  double debug_G_sum;
  double debug_V_sum;

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
  struct ThermalOps {
    // Maximum neighbors for sc/26n lattice
    static constexpr int MAX_NEIGHBORS_SC26N = 26;

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
        const Kokkos::View<int**>& neighbor,
        const Kokkos::View<double*[3]>& xyz,
        double dx_lattice,
        double* grad_out,
        double* magnitude_out) {

      // Cache neighbor temperatures in registers for performance
      const int nn = numneigh(i);
      double T_neigh[26];
      for (int j = 0; j < nn && j < 26; j++) {
        T_neigh[j] = d_T(neighbor(i,j));
      }

      // Detect melt surface: check if the neighbor directly above (slot 13 =
      // offset [0,0,1]) is inactive/active (active_flag <= 1). Low-coordination
      // global-boundary sites (numneigh <= 13) have no valid slot 13, so they
      // are treated as melt-surface -- matching the CPU normal_finder() and
      // avoiding an out-of-bounds neighbor read.
      bool is_melt_surface = (nn <= 13);
      if (!is_melt_surface) {
        is_melt_surface = (d_active_flag(neighbor(i, 13)) <= 1);
      }

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
        int nbr = neighbor(i,n);

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
