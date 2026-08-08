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

#include "app_lattice_kokkos.h"
#include <vector>

namespace SPPARKS_NS {

class AppPottsQuaternionKokkos : public AppLatticeKokkos {
public:
  AppPottsQuaternionKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsQuaternionKokkos();

  // App-specific data
  int nspins;
  int *spin;                              // Host pointer (from iarray[0])
  double *q0, *qx, *qy, *qz;              // Host pointers (from darray[])

  // Device views
  Kokkos::View<int*> d_spin;
  Kokkos::View<double*> d_q0, d_qx, d_qy, d_qz;

  // Symmetries (copied to device once, read-only)
  Kokkos::View<double*> d_symmetries;
  std::vector<double> symmetries;
  int num_symmetries;
  double theta_cut;

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;

  // Implement required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  virtual void allocate_app_views() override;
  virtual void sweep_color_kokkos(int color, int start, int count) override;

  // Site methods (required by AppLattice base)
  virtual double site_energy(int) override;
  virtual void site_event_rejection(int, class RandomPark *) override;
  virtual double site_propensity(int) override;
  virtual void site_event(int, class RandomPark *) override;

protected:
  // Device-compatible quaternion operations (accessible to derived classes)
  //
  // QUATERNION CONVENTION: Scalar-first (Hamilton notation)
  // q = (q0, qx, qy, qz) where:
  //   q0 = cos(theta/2)          (scalar/real component)
  //   (qx, qy, qz) = sin(theta/2)*axis  (vector/imaginary components)
  // This matches the CPU reference implementation in quaternion.h
  //
  struct QuaternionOps {
    // Numerical thresholds for quaternion operations
    static constexpr double PARALLEL_VECTOR_THRESHOLD = 1.0e-10;
    static constexpr double QUATERNION_ZERO_THRESHOLD = 1.0e-15;

    KOKKOS_INLINE_FUNCTION
    static void normalize(double* q) {
      double norm = Kokkos::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
      if (norm > QUATERNION_ZERO_THRESHOLD) {
        double inv_norm = 1.0 / norm;
        q[0] *= inv_norm;
        q[1] *= inv_norm;
        q[2] *= inv_norm;
        q[3] *= inv_norm;
      }
    }

    KOKKOS_INLINE_FUNCTION
    static void composition(const double* a, const double* b, double* result) {
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
      double t = 0.0;

      for (int i = 0; i < num_sym; i++) {
        const double* qr = &symmetries[4 * i];

        double q[4];
        composition(q2, qr, q);

        double qc[4];
        conjugate(q, qc);

        double m[4];
        composition(q1, qc, m);

        double trace = trace_of_rotation_matrix(m);
        if (trace > t) t = trace;
      }

      double di;
      if (Kokkos::fabs(t - 3.0) <= 1.0e-6) {
        di = 0.0;
      } else {
        di = Kokkos::acos(0.5 * (t - 1.0)) * 180.0 / M_PI;
      }
      return di;
    }

    KOKKOS_INLINE_FUNCTION
    static void vector_cross_product(const double* a, const double* b, double* result) {
      result[0] = a[1] * b[2] - a[2] * b[1];
      result[1] = -(a[0] * b[2] - a[2] * b[0]);
      result[2] = a[0] * b[1] - a[1] * b[0];
    }

    KOKKOS_INLINE_FUNCTION
    static void rotate_vector(const double* q, const double* r, double* result) {
      double q_vec[3] = {q[1], q[2], q[3]};
      double cross1[3], cross2[3];

      vector_cross_product(q_vec, r, cross1);
      vector_cross_product(q_vec, cross1, cross2);

      result[0] = r[0] + 2.0 * q[0] * cross1[0] + 2.0 * cross2[0];
      result[1] = r[1] + 2.0 * q[0] * cross1[1] + 2.0 * cross2[1];
      result[2] = r[2] + 2.0 * q[0] * cross1[2] + 2.0 * cross2[2];
    }

    template<typename RandGenType>
    KOKKOS_INLINE_FUNCTION
    static void generate_random_quaternion(double* q, RandGenType& rand_gen) {
      double u1, u2, mag1, mag2;
      double n1, n2, n3, n4;

      u1 = rand_gen.drand();
      u2 = rand_gen.drand();
      mag1 = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
      n1 = mag1 * Kokkos::cos(2.0 * M_PI * u2);
      n2 = mag1 * Kokkos::sin(2.0 * M_PI * u2);

      u1 = rand_gen.drand();
      u2 = rand_gen.drand();
      mag2 = Kokkos::sqrt(-2.0 * Kokkos::log(u1));
      n3 = mag2 * Kokkos::cos(2.0 * M_PI * u2);
      n4 = mag2 * Kokkos::sin(2.0 * M_PI * u2);

      q[0] = n1;
      q[1] = n2;
      q[2] = n3;
      q[3] = n4;

      normalize(q);
    }

    // Convert quaternion to rotation matrix (row-major 3x3)
    KOKKOS_INLINE_FUNCTION
    static void to_rotation_matrix(const double* q, double* m) {
      const double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];
      const double q00 = q0 * q0;
      const double q0x = q0 * qx;
      const double q0y = q0 * qy;
      const double q0z = q0 * qz;
      const double qxx = qx * qx;
      const double qxy = qx * qy;
      const double qxz = qx * qz;
      const double qyy = qy * qy;
      const double qyz = qy * qz;
      const double qzz = qz * qz;
      m[0] = q00 + qxx - qyy - qzz;  // r11
      m[1] = 2 * (qxy - q0z);        // r12
      m[2] = 2 * (qxz + q0y);        // r13
      m[3] = 2 * (qxy + q0z);        // r21
      m[4] = q00 - qxx + qyy - qzz;  // r22
      m[5] = 2 * (qyz - q0x);        // r23
      m[6] = 2 * (qxz - q0y);        // r31
      m[7] = 2 * (qyz + q0x);        // r32
      m[8] = q00 - qxx - qyy + qzz;  // r33
    }

    // Get cosine of minimum angle between quaternion's <100> directions and vector u
    // Used for texture weighting in solidification
    KOKKOS_INLINE_FUNCTION
    static double get_cosine_angle_q_u(const double* q, const double* u) {
      // Normalize the target vector
      double n[3] = {u[0], u[1], u[2]};
      double d = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
      if (d > 1.0e-15) {
        double inv_d = 1.0 / d;
        n[0] *= inv_d;
        n[1] *= inv_d;
        n[2] *= inv_d;
      }

      // Get rotation matrix from quaternion
      double m[9];
      to_rotation_matrix(q, m);

      // Dot columns of 'm' with 'n' (matrix transpose times vector)
      // Find maximum absolute dot product across all 3 columns
      double max_dot = 0.0;
      for (int col = 0; col < 3; col++) {
        double dot = 0.0;
        for (int row = 0; row < 3; row++) {
          int ptr = col + 3 * row;
          dot += m[ptr] * n[row];
        }
        dot = Kokkos::fabs(dot);
        if (dot > max_dot) max_dot = dot;
      }
      return max_dot;
    }

    // Rotate quaternion q towards vector u by given angle (in radians)
    // If angle exceeds the misorientation, stops at alignment
    KOKKOS_INLINE_FUNCTION
    static void rotate_q_towards_u(const double* q, const double* u, double angle, double* result) {
      // Normalize the target vector
      double n[3] = {u[0], u[1], u[2]};
      double d = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
      if (d < 1.0e-15) {
        // Zero magnitude target, return original quaternion
        result[0] = q[0]; result[1] = q[1]; result[2] = q[2]; result[3] = q[3];
        return;
      }
      double inv_d = 1.0 / d;
      n[0] *= inv_d;
      n[1] *= inv_d;
      n[2] *= inv_d;

      // Get reference direction from the quaternion (rotate z-axis)
      double ref_axis[3] = {0.0, 0.0, 1.0};
      double current_dir[3];
      rotate_vector(q, ref_axis, current_dir);

      // Calculate the current misorientation angle
      double dot_product = current_dir[0]*n[0] + current_dir[1]*n[1] + current_dir[2]*n[2];
      dot_product = Kokkos::fmax(-1.0, Kokkos::fmin(1.0, dot_product));
      double current_misorientation = Kokkos::acos(dot_product);

      // Check if already aligned (within tolerance)
      if (current_misorientation < 1.0e-10) {
        result[0] = q[0]; result[1] = q[1]; result[2] = q[2]; result[3] = q[3];
        return;
      }

      // Limit the rotation angle to not exceed the current misorientation
      double actual_angle = Kokkos::fmin(Kokkos::fabs(angle), current_misorientation);

      // Calculate the rotation axis as the cross product
      double rot_axis[3];
      vector_cross_product(current_dir, n, rot_axis);

      // Check if vectors are parallel (cross product near zero)
      double axis_norm = Kokkos::sqrt(rot_axis[0]*rot_axis[0] + rot_axis[1]*rot_axis[1] + rot_axis[2]*rot_axis[2]);
      if (axis_norm < 1.0e-10) {
        // Vectors are parallel
        if (dot_product > 0.99999) {
          // Same direction, return original quaternion
          result[0] = q[0]; result[1] = q[1]; result[2] = q[2]; result[3] = q[3];
          return;
        } else {
          // Opposite directions (180° apart), find a perpendicular axis
          if (Kokkos::fabs(n[0]) < 0.9) {
            double tmp[3] = {1.0, 0.0, 0.0};
            vector_cross_product(n, tmp, rot_axis);
          } else {
            double tmp[3] = {0.0, 1.0, 0.0};
            vector_cross_product(n, tmp, rot_axis);
          }
          axis_norm = Kokkos::sqrt(rot_axis[0]*rot_axis[0] + rot_axis[1]*rot_axis[1] + rot_axis[2]*rot_axis[2]);
        }
      }

      // Normalize the rotation axis
      double inv_axis_norm = 1.0 / axis_norm;
      rot_axis[0] *= inv_axis_norm;
      rot_axis[1] *= inv_axis_norm;
      rot_axis[2] *= inv_axis_norm;

      // Create the rotation quaternion for the actual angle
      double half_angle = actual_angle * 0.5;
      double cos_half = Kokkos::cos(half_angle);
      double sin_half = Kokkos::sin(half_angle);

      double rot_q[4] = {cos_half, sin_half * rot_axis[0], sin_half * rot_axis[1], sin_half * rot_axis[2]};

      // Apply the rotation to the original quaternion
      composition(rot_q, q, result);

      // Normalize the result
      normalize(result);
    }
  };

private:
  // Sweep functor
  struct ColorSweepFunctor;

  // Temporary arrays for KMC
  int *sites, *unique_neigh;

  // Helper to normalize quaternion arrays on host
  void normalize_quaternion_arrays(double* q0_out, double* qx_out, double* qy_out, double* qz_out,
                                   const double* q0_in, const double* qx_in,
                                   const double* qy_in, const double* qz_in, int n);

  // Persistent host mirrors for the sync paths, sized to nall. Allocated only
  // when nall changes rather than per call: sync_app_data_to_host runs at every
  // output step, and on a discrete-memory backend create_mirror_view allocates
  // a fresh nall-sized host buffer for each of the four quaternion arrays. On a
  // host-space backend these alias the device Views, so this costs nothing.
  // decltype of the actual mirror, not View<...>::HostMirror -- the latter was
  // removed in Kokkos 5.
  decltype(Kokkos::create_mirror_view(d_q0)) h_q0_m, h_qx_m, h_qy_m, h_qz_m;
  int sync_mirror_n = -1;
  void ensure_sync_mirrors();
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

E: Kokkos app requires 'sweep color' or 'sweep color/strict' for correct GPU execution

GPU execution requires coloring to prevent race conditions. Add 'sweep color'
to your input script.

*/
