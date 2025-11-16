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
AppStyle(additive_temperature_texture/kk,AppAdditiveExtTempTextureKokkos)

#else

#ifndef SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_KOKKOS_H
#define SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_KOKKOS_H

#include "app_potts_quaternion_kokkos.h"
#include "kokkos_double_buffer.h"
#include "temperature_source.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>
#include <memory>

namespace SPPARKS_NS {

class AppAdditiveExtTempTextureKokkos : public AppPottsQuaternionKokkos {
 public:
  AppAdditiveExtTempTextureKokkos(class SPPARKS *, int, char **);
  virtual ~AppAdditiveExtTempTextureKokkos();

  // Override base class methods
  virtual void grow_app();
  virtual void init_app();
  virtual void app_update(double);
  virtual void input_app(char *, int, char **);

  // Temperature field management
  void update_temperature_from_source(double simulation_time);
  void sync_temperature_to_device();
  void compute_gradients_kernel();

  // Phase 3: Phase transition kernels
  void initialize_nucleation_parameters();
  void melting_kernel();
  void nucleation_kernel();
  void solidification_kernel();

 protected:
  // Temperature fields (CPU-updated from HDF5, device-readable)
  // Uses DualView because temperature is read-only during device sweeps
  Kokkos::DualView<double*> k_T;  // Temperature field

  // Gradient cache (device-only, computed once per timestep)
  // Uses simple View because computed on device, no host sync needed
  Kokkos::View<double*> d_G;        // Gradient magnitude
  Kokkos::View<double*> d_grad_x;   // Gradient x-component
  Kokkos::View<double*> d_grad_y;   // Gradient y-component
  Kokkos::View<double*> d_grad_z;   // Gradient z-component
  Kokkos::View<double*> d_V;        // Solidification velocity

  // Phase state arrays (modified during sweeps - need double buffering)
  KokkosDoubleBuffer<int> buffer_active_flag;      // 0=inactive, 1=active, 2=molten, 3=solidified, 5=void
  KokkosDoubleBuffer<double> buffer_mobility_out;  // Arrhenius mobility
  KokkosDoubleBuffer<double> buffer_solid_d;       // Solid front distance

  // Per-spin nucleation arrays (device-accessible, initialized once)
  Kokkos::View<int*> d_nucleation_flags;     // Can this spin nucleate? (1=yes, 0=no)
  Kokkos::View<double*> d_nucleation_temps;  // Critical undercooling for nucleation (K)
  Kokkos::View<double*> d_nucleation_sizes;  // Nucleus volume (m^3)

  // Global parameter arrays (device-accessible constants)
  Kokkos::View<double*> d_solid_front_coeffs;  // Solidification velocity polynomial coefficients
  Kokkos::View<double[3]> d_texture_params;    // Texture model parameters: [c1, c2, c3]

  // Random number generator pool (thread-safe device RNG)
  Kokkos::Random_XorShift64_Pool<> rand_pool;

  // Host pointers (mapped to iarray/darray like CPU version)
  int *active_flag;
  double *mobility_out;
  double *T;
  double *solid_d;
  double *G;
  double *V;
  double *neigh_dist;
  double *unique_dot;

  // Temperature source interface (CPU-only)
  std::unique_ptr<TemperatureSource> temperature_source;
  bool use_temperature_source;
  double fast_forward_search_window;

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

  // Sync helpers (extend parent class)
  void sync_data_to_device();
  void sync_data_to_host();
  void swap_buffers();

  // Device-compatible thermal operations
  struct ThermalOps {
    // 26-neighbor offset map for sc/26n lattice (device constant memory)
    // Maps neighbor index to [dx, dy, dz] offset in lattice units
    static constexpr int offset_map[26][3] = {
      {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},  // Layer z=-1
      {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
      {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
      { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},  // Layer z=0
      { 0, 0,-1},             { 0, 0, 1},  // Skip center (0,0,0)
      { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
      { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},  // Layer z=1
      { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
      { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}
    };

    // Compute temperature gradient at site i using weighted least-squares
    //
    // Algorithm: For each of 26 cardinal neighbors, build weighted least
    // squares system A*grad = b where A is [dx dy dz] and b is temperature
    // difference. Solve 3x3 system using Cramer's rule.
    //
    // Args:
    //   i: Site index (0 <= i < nlocal)
    //   d_T: Temperature field (device view, read-only)
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

      // Build weighted least squares matrices
      // A^T * W * A (3x3 symmetric matrix, store as 9 elements)
      double AtA[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      // A^T * W * b (3x1 vector)
      double Atb[3] = {0.0, 0.0, 0.0};

      double T_center = d_T(i);

      // Accumulate weighted least squares matrices
      for (int n = 0; n < nn && n < 26; n++) {
        int nbr = neighbor(i,n);

        // Get neighbor offset from lookup table
        double dx_n = offset_map[n][0] * dx_lattice;
        double dy_n = offset_map[n][1] * dx_lattice;
        double dz_n = offset_map[n][2] * dx_lattice;

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
