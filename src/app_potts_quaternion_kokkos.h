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
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppPottsQuaternionKokkos : public AppPottsQuaternion {
 public:
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

  AppPottsQuaternionKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsQuaternionKokkos();

  // Kokkos-specific data
  Kokkos::DualView<int*> k_spin;
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;

  // Quaternion data
  Kokkos::DualView<double*> k_q0;
  Kokkos::DualView<double*> k_qx;
  Kokkos::DualView<double*> k_qy;
  Kokkos::DualView<double*> k_qz;

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

private:
  // Device-side helper functors
  struct ComputeEnergyFunctor;
  struct RejectionSweepFunctor;
  struct SectorSweepFunctor;

  // Sync helpers
  void sync_data_to_device();
  void sync_data_to_host();

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
