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
#include "crystalline_orientations/quaternion_ops.h"
#include <vector>

namespace SPPARKS_NS {

class AppPottsQuaternionKokkos : public AppLatticeKokkos {
public:
  AppPottsQuaternionKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsQuaternionKokkos();

  // App-specific data
  int nspins;
  int spin_lo = 1;                        // lowest valid spin (highest is nspins)
  int *spin;                              // Host pointer (from iarray[0])
  double *q0, *qx, *qy, *qz;              // Host pointers (from darray[])

  // Device views
  Kokkos::View<int*> d_spin;
  Kokkos::View<double*> d_q0, d_qx, d_qy, d_qz;

  // Symmetries (copied to device once, read-only)
  Kokkos::View<double*> d_symmetries;
  std::vector<double> symmetries;
  int num_symmetries;
  int crystal_index = 0;    // CRYSTAL::index of the symmetries
  double theta_cut;

  // "crystal" -> &crystal_index, for dump image ipf
  virtual void *extract_app(char *) override;

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;

  // Implement required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  virtual void allocate_app_views() override;
  virtual void sweep_color_kokkos(int color, int start, int count) override;

  // Site methods (required by AppLattice base)
  virtual double site_energy(int) override;
  virtual void site_event_rejection(int, class RandomPark *) override;
  virtual double site_propensity(int) override;
  virtual void site_event(int, class RandomPark *) override;

protected:
  // KMC helpers
  int unique_neighbors(int);
  void copy_state(int, int);

  // Device-compatible quaternion operations (accessible to derived classes)
  //
  // QUATERNION CONVENTION: Scalar-first (Hamilton notation)
  // q = (q0, qx, qy, qz) where:
  //   q0 = cos(theta/2)          (scalar/real component)
  //   (qx, qy, qz) = sin(theta/2)*axis  (vector/imaginary components)
  // Shared host/device math (normalize, composition, rotate_q_towards_u, ...)
  // is in crystalline_orientations/quaternion_ops.h; these are kk-only.
  //
  struct QuaternionOps {
    // m = q1 * conj(q2 * s) is the misorientation under symmetry s, and for
    // unit quaternions trace(R(m)) = 4 m0^2 - 1. Its scalar part is
    // m0 = q1 . (q2 s) = (conj(q2) q1) . s, so d = conj(q2) q1 is formed once
    // and each symmetry costs a 4-term dot product instead of two products.
    // Returns degrees; angles below ~1e-3 rad (|t-3| <= 1e-6) count as 0.
    KOKKOS_INLINE_FUNCTION
    static double compute_disorientation(const double* symmetries, int num_sym,
                                         const double* q1, const double* q2) {
#ifndef SPPARKS_KOKKOS_DOUBLE_DOUBLE
      // float argmax of |m0| over symmetries, then the angle of that one
      // misorientation from its vector part: acos near 1 would lose small
      // angles in float, atan2 does not
      const float ar = (float)q2[0], ax = -(float)q2[1], ay = -(float)q2[2], az = -(float)q2[3];
      const float br = (float)q1[0], bx = (float)q1[1], by = (float)q1[2], bz = (float)q1[3];
      const float df[4] = {ar*br - (ax*bx + ay*by + az*bz),
                           ar*bx + br*ax + (ay*bz - az*by),
                           ar*by + br*ay - (ax*bz - az*bx),
                           ar*bz + br*az + (ax*by - ay*bx)};
      float best = -1.0f;
      int ibest = 0;
      for (int i = 0; i < num_sym; i++) {
        const double* s = &symmetries[4 * i];
        float m0 = df[0]*(float)s[0] + df[1]*(float)s[1] +
                   df[2]*(float)s[2] + df[3]*(float)s[3];
        m0 = Kokkos::fabs(m0);
        if (m0 > best) { best = m0; ibest = i; }
      }
      const double* s = &symmetries[4 * ibest];
      const float sf[4] = {(float)s[0], (float)s[1], (float)s[2], (float)s[3]};
      // vector part of conj(s) d (its scalar part is +-best)
      const float vx = sf[0]*df[1] - df[0]*sf[1] - (sf[2]*df[3] - sf[3]*df[2]);
      const float vy = sf[0]*df[2] - df[0]*sf[2] + (sf[1]*df[3] - sf[3]*df[1]);
      const float vz = sf[0]*df[3] - df[0]*sf[3] - (sf[1]*df[2] - sf[2]*df[1]);
      const float theta =
        2.0f * Kokkos::atan2(Kokkos::sqrt(vx*vx + vy*vy + vz*vz), best);
      if (theta <= 1.0e-3f) return 0.0;
      return (double)theta * 180.0 / M_PI;
#else
      double q2c[4], d[4];
      quaternion::conjugate(q2, q2c);
      quaternion::composition(q2c, q1, d);

      double t = 0.0;

      for (int i = 0; i < num_sym; i++) {
        const double* s = &symmetries[4 * i];
        double m0 = d[0]*s[0] + d[1]*s[1] + d[2]*s[2] + d[3]*s[3];
        double trace = 4.0 * m0 * m0 - 1.0;
        if (trace > t) t = trace;
      }

      double di;
      if (Kokkos::fabs(t - 3.0) <= 1.0e-6) {
        di = 0.0;
      } else {
        di = Kokkos::acos(0.5 * (t - 1.0)) * 180.0 / M_PI;
      }
      return di;
#endif
    }

    // Read-Shockley pair energy for disorientation di (degrees), cutoff
    // theta_cut: r (1 - ln r) for r = di/theta_cut < 1, else 1
    KOKKOS_INLINE_FUNCTION
    static double read_shockley(double di, double theta_cut) {
      KK_FLOAT ratio = (KK_FLOAT)(di / theta_cut);
      if (ratio <= (KK_FLOAT)0) return 0.0;
      if (ratio < (KK_FLOAT)1) return (double)(ratio * ((KK_FLOAT)1 - Kokkos::log(ratio)));
      return 1.0;
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

      quaternion::normalize(q);
    }
  };

public:
  // Public for nvcc: a __global__ kernel's template argument must be accessible.
  struct ColorSweepFunctor;

private:
  // Temporary arrays for KMC
  int *sites, *unique_neigh;

  // Helper to normalize quaternion arrays on host
  void normalize_host(int n, bool off_unit_only);
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
