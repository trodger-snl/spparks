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
AppStyle(potts/quaternion/kk,AppPottsQuaternionKokkos)

#else

#ifndef SPK_APP_POTTS_QUATERNION_KOKKOS_H
#define SPK_APP_POTTS_QUATERNION_KOKKOS_H

#include "app_potts_quaternion.h"
#include "kokkos_double_buffer.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppPottsQuaternionKokkos : public AppPottsQuaternion {
 public:
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

  AppPottsQuaternionKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsQuaternionKokkos();

  // Neighbor data
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;

  // Double-buffer system to prevent race conditions
  // Uses generalized framework for spin and quaternion arrays
  KokkosDoubleBuffer<int> buffer_spin;
  KokkosDoubleBuffer<double> buffer_q0;
  KokkosDoubleBuffer<double> buffer_qx;
  KokkosDoubleBuffer<double> buffer_qy;
  KokkosDoubleBuffer<double> buffer_qz;

  // Symmetries (copied to device once, read-only)
  Kokkos::View<double*> d_symmetries;
  int num_symmetries;

  // Random number generation pool
  pool_type *rand_pool;

  // Override base class methods with Kokkos versions
  virtual void grow_app();
  virtual void init_app();
  virtual void iterate_rejection(double);

  // Kokkos computation kernels
  virtual double site_energy(int);
  virtual void site_event_rejection(int, class RandomPark *);

  // Kokkos-specific kernel implementations
  void sweep_rejection_kokkos();
  void sweep_rejection_kokkos(int, int *);

  // Input command processing
  virtual void input_app(char *, int, char **);

  // Profiling helpers
  void enable_sync_profiling(bool enable = true) {
    buffer_spin.enable_profiling(enable);
    buffer_q0.enable_profiling(enable);
    buffer_qx.enable_profiling(enable);
    buffer_qy.enable_profiling(enable);
    buffer_qz.enable_profiling(enable);
  }
  void reset_sync_profiling() {
    buffer_spin.reset_profiling();
    buffer_q0.reset_profiling();
    buffer_qx.reset_profiling();
    buffer_qy.reset_profiling();
    buffer_qz.reset_profiling();
  }
  void print_sync_profiling_stats(FILE* fp) {
    buffer_spin.print_profiling_stats(fp, "spin");
    buffer_q0.print_profiling_stats(fp, "q0");
    buffer_qx.print_profiling_stats(fp, "qx");
    buffer_qy.print_profiling_stats(fp, "qy");
    buffer_qz.print_profiling_stats(fp, "qz");
  }

private:
  // Device-side helper functors
  struct ComputeEnergyFunctor;
  struct RejectionSweepFunctor;
  struct SectorSweepFunctor;

  // Sync helpers
  void sync_data_to_device();          // MPI version - only updates read buffer
  void sync_data_to_device_full();     // Full version - updates both buffers (initialization only)
  void sync_data_to_host();
  void swap_buffers();

  // Profiling configuration
  bool sync_profiling_enabled;

  // Quaternion helper - normalizes arrays on host
  void normalize_quaternion_arrays(double* q0_out, double* qx_out, double* qy_out, double* qz_out,
                                   const double* q0_in, const double* qx_in,
                                   const double* qy_in, const double* qz_in,
                                   int n);

  // Device-compatible quaternion operations
  struct QuaternionOps {
    KOKKOS_INLINE_FUNCTION
    static void normalize(double* q) {
      // Normalize quaternion to unit length
      double norm = Kokkos::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
      if (norm > 1.0e-15) {
        double inv_norm = 1.0 / norm;
        q[0] *= inv_norm;
        q[1] *= inv_norm;
        q[2] *= inv_norm;
        q[3] *= inv_norm;
      }
    }

    KOKKOS_INLINE_FUNCTION
    static void composition(const double* a, const double* b, double* result) {
      // Quaternion multiplication: result = a * b
      double ar = a[0], ax = a[1], ay = a[2], az = a[3];
      double br = b[0], bx = b[1], by = b[2], bz = b[3];

      result[0] = ar * br - (ax * bx + ay * by + az * bz);
      result[1] = ar * bx + br * ax + (ay * bz - az * by);
      result[2] = ar * by + br * ay - (ax * bz - az * bx);
      result[3] = ar * bz + br * az + (ax * by - ay * bx);
    }

    KOKKOS_INLINE_FUNCTION
    static void conjugate(const double* q, double* result) {
      result[0] = q[0];
      result[1] = -q[1];
      result[2] = -q[2];
      result[3] = -q[3];
    }

    KOKKOS_INLINE_FUNCTION
    static double trace_of_rotation_matrix(const double* q) {
      double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];
      double q00 = q0 * q0;
      double qxx = qx * qx;
      double qyy = qy * qy;
      double qzz = qz * qz;
      double r11 = q00 + qxx - qyy - qzz;
      double r22 = q00 - qxx + qyy - qzz;
      double r33 = q00 - qxx - qyy + qzz;
      return r11 + r22 + r33;
    }

    KOKKOS_INLINE_FUNCTION
    static double compute_disorientation(const double* symmetries, int num_sym,
                                         const double* q1, const double* q2) {
      // Compute disorientation between q1 and q2 using symmetries
      double t = 0.0;

      for (int i = 0; i < num_sym; i++) {
        const double* qr = &symmetries[4 * i];

        // q = q2 * qr (active quaternions)
        double q[4];
        composition(q2, qr, q);

        // qc = conjugate(q)
        double qc[4];
        conjugate(q, qc);

        // m = q1 * qc
        double m[4];
        composition(q1, qc, m);

        // trace of rotation matrix from m
        double trace = trace_of_rotation_matrix(m);
        if (trace > t) t = trace;
      }

      // Compute disorientation angle in degrees
      double di;
      if (Kokkos::fabs(t - 3.0) <= 1.0e-6) {
        di = 0.0;
      } else {
        di = Kokkos::acos(0.5 * (t - 1.0)) * 180.0 / M_PI;
      }
      return di;
    }

    // NEW: Advanced operations for AM applications (AM_KOKKOS_PLAN.md Phase 1)

    KOKKOS_INLINE_FUNCTION
    static void vector_cross_product(const double* a, const double* b, double* result) {
      // Compute cross product: result = a × b
      result[0] = a[1] * b[2] - a[2] * b[1];
      result[1] = -(a[0] * b[2] - a[2] * b[0]);
      result[2] = a[0] * b[1] - a[1] * b[0];
    }

    KOKKOS_INLINE_FUNCTION
    static void rotate_vector(const double* q, const double* r, double* result) {
      // Rotate vector r by quaternion q
      // Formula: r' = r + 2*q0*(q_vec × r) + 2*(q_vec × (q_vec × r))
      // where q = (q0, qx, qy, qz)

      double q_vec[3] = {q[1], q[2], q[3]};
      double cross1[3], cross2[3];

      vector_cross_product(q_vec, r, cross1);
      vector_cross_product(q_vec, cross1, cross2);

      result[0] = r[0] + 2.0 * q[0] * cross1[0] + 2.0 * cross2[0];
      result[1] = r[1] + 2.0 * q[0] * cross1[1] + 2.0 * cross2[1];
      result[2] = r[2] + 2.0 * q[0] * cross1[2] + 2.0 * cross2[2];
    }

    KOKKOS_INLINE_FUNCTION
    static double get_cosine_angle_q_u(const double* q, const double* u) {
      // Compute maximum cosine of angle between crystal orientation (q) and vector (u)
      // Uses rotation matrix R derived from quaternion q
      // Returns max(|R_column · u|) over all three columns

      double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];

      // Compute rotation matrix components
      // Column 1: R * [1,0,0]
      double R00 = 1.0 - 2.0 * (qy * qy + qz * qz);
      double R10 = 2.0 * (qx * qy + q0 * qz);
      double R20 = 2.0 * (qx * qz - q0 * qy);

      // Column 2: R * [0,1,0]
      double R01 = 2.0 * (qx * qy - q0 * qz);
      double R11 = 1.0 - 2.0 * (qx * qx + qz * qz);
      double R21 = 2.0 * (qy * qz + q0 * qx);

      // Column 3: R * [0,0,1]
      double R02 = 2.0 * (qx * qz + q0 * qy);
      double R12 = 2.0 * (qy * qz - q0 * qx);
      double R22 = 1.0 - 2.0 * (qx * qx + qy * qy);

      // Dot each column with u and take absolute value
      double cos1 = Kokkos::fabs(R00 * u[0] + R10 * u[1] + R20 * u[2]);
      double cos2 = Kokkos::fabs(R01 * u[0] + R11 * u[1] + R21 * u[2]);
      double cos3 = Kokkos::fabs(R02 * u[0] + R12 * u[1] + R22 * u[2]);

      // Return maximum
      double max_cos = cos1;
      if (cos2 > max_cos) max_cos = cos2;
      if (cos3 > max_cos) max_cos = cos3;

      return max_cos;
    }

    KOKKOS_INLINE_FUNCTION
    static void rotate_q_towards_u(const double* q_in, const double* u,
                                   double angle, double* q_out) {
      // Rotate quaternion q_in by angle (radians) towards unit vector u
      // Algorithm:
      // 1. Rotate basis vectors e1, e2, e3 by quaternion q_in
      // 2. Find which rotated vector is closest to u
      // 3. Compute rotation axis perpendicular to both
      // 4. Create rotation quaternion around axis by angle
      // 5. Compose with original quaternion

      // Basis vectors
      double e1[3] = {1.0, 0.0, 0.0};
      double e2[3] = {0.0, 1.0, 0.0};
      double e3[3] = {0.0, 0.0, 1.0};

      // Rotate basis vectors
      double r1[3], r2[3], r3[3];
      rotate_vector(q_in, e1, r1);
      rotate_vector(q_in, e2, r2);
      rotate_vector(q_in, e3, r3);

      // Find which rotated vector is closest to u (maximum dot product)
      double dot1 = r1[0] * u[0] + r1[1] * u[1] + r1[2] * u[2];
      double dot2 = r2[0] * u[0] + r2[1] * u[1] + r2[2] * u[2];
      double dot3 = r3[0] * u[0] + r3[1] * u[1] + r3[2] * u[2];

      // Select the closest rotated basis vector
      double* closest_r = r1;
      double max_dot = dot1;
      if (dot2 > max_dot) {
        closest_r = r2;
        max_dot = dot2;
      }
      if (dot3 > max_dot) {
        closest_r = r3;
        max_dot = dot3;
      }

      // Compute rotation axis: perpendicular to both closest_r and u
      double axis[3];
      vector_cross_product(closest_r, u, axis);

      // Normalize axis
      double axis_mag = Kokkos::sqrt(axis[0] * axis[0] +
                                     axis[1] * axis[1] +
                                     axis[2] * axis[2]);

      // Handle parallel vectors (cross product ≈ 0)
      if (axis_mag < 1.0e-10) {
        // Vectors are parallel, no rotation needed
        q_out[0] = q_in[0];
        q_out[1] = q_in[1];
        q_out[2] = q_in[2];
        q_out[3] = q_in[3];
        return;
      }

      // Normalize axis
      axis[0] /= axis_mag;
      axis[1] /= axis_mag;
      axis[2] /= axis_mag;

      // Create rotation quaternion from axis and angle
      // q_rot = (cos(angle/2), sin(angle/2) * axis)
      double half_angle = angle * 0.5;
      double sin_half = Kokkos::sin(half_angle);
      double cos_half = Kokkos::cos(half_angle);

      double q_rot[4];
      q_rot[0] = cos_half;
      q_rot[1] = sin_half * axis[0];
      q_rot[2] = sin_half * axis[1];
      q_rot[3] = sin_half * axis[2];

      // Compose: q_out = q_rot * q_in
      composition(q_rot, q_in, q_out);

      // Normalize to ensure unit quaternion
      normalize(q_out);
    }

    template<typename RandGenType>
    KOKKOS_INLINE_FUNCTION
    static void generate_random_quaternion(double* q, RandGenType& rand_gen) {
      // Generate uniformly distributed random quaternion on SO(3)
      // Uses Box-Muller transform to generate normal variates

      // Box-Muller transform for generating pairs of normal variates
      // Generate 4 independent normal variates
      double u1, u2, mag1, mag2;
      double n1, n2, n3, n4;

      // Generate first pair
      u1 = rand_gen.drand();
      u2 = rand_gen.drand();
      mag1 = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
      n1 = mag1 * Kokkos::cos(2.0 * M_PI * u2);
      n2 = mag1 * Kokkos::sin(2.0 * M_PI * u2);

      // Generate second pair
      u1 = rand_gen.drand();
      u2 = rand_gen.drand();
      mag2 = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
      n3 = mag2 * Kokkos::cos(2.0 * M_PI * u2);
      n4 = mag2 * Kokkos::sin(2.0 * M_PI * u2);

      // Assign to quaternion
      q[0] = n1;
      q[1] = n2;
      q[2] = n3;
      q[3] = n4;

      // Normalize to unit quaternion
      normalize(q);
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
