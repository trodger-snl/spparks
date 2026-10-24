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
#ifndef SPK_QUATERNION_H
#define SPK_QUATERNION_H

#include <math.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "quaternion_ops.h"

using std::vector;

namespace SPPARKS_NS {

namespace quaternion {

// NOTE: this generator is wall-clock seeded, so generate_random_unit_quaternions
// is NOT reproducible run to run. Prefer generate_unit_quaternion(rng) below,
// which draws from a caller-supplied stream; keying that stream to a site's
// GLOBAL id (RandomPark::tagreset) makes the orientation field both repeatable
// and independent of the MPI decomposition.
inline unsigned seed =
    std::chrono::system_clock::now().time_since_epoch().count();
inline std::default_random_engine generator(seed);

/* ----------------------------------------------------------------------
   Per-site orientation RNG: splitmix64 keyed by (deck seed, GLOBAL site id).

   NOT RandomPark::tagreset. Park/Miller is a 31-bit LCG, and reseeding it at
   ADJACENT integers leaves the resulting streams correlated -- a 100-draw
   warmup does not clear it, 1000 does not either, and hashing the id merely
   moves the artifact. Fed through Marsaglia's rejection sampling that showed up
   as a spatial artifact in the orientation field: neighbouring sites came out
   systematically MORE misoriented than independent draws (mean 131.6 deg vs
   126.8 deg; |q.q| between adjacent ids 0.0359 below the shuffled-pair mean,
   where an independent field gives 0.0007). The marginal distribution was
   correct -- it is the joint structure over neighbouring ids that was wrong.

   splitmix64 is a hash/counter construction, so each site's stream is
   independent by design rather than by warmup. Measured over 100k sites it
   reproduces the independent reference on every check: adjacent-pair excess
   -0.00099, adjacent misorientation 126.505 deg, the uniform-SO(3) rotation
   angle histogram, and no structure at lattice strides of 1, one row or one
   plane. It is also cheaper -- no warmup draws at all.
------------------------------------------------------------------------- */

class SiteOrientationRNG {
public:
  // seed: the deck's `seed` value. site_id: the site's GLOBAL id, so the
  // stream depends on nothing that varies with the MPI decomposition.
  SiteOrientationRNG(uint64_t seed, uint64_t site_id)
    : state_((seed * 0x9E3779B97F4A7C15ULL) ^ (site_id + 0x9E3779B97F4A7C15ULL)) {}

  double uniform() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
  }

private:
  uint64_t state_;
};

/* ----------------------------------------------------------------------
   Uniformly-distributed unit quaternion from a caller-supplied RNG, using
   Marsaglia's method -- the same construction (and the same operation order)
   as the device kernel in app_additive_texture_kokkos.cpp, so a host-generated
   orientation is drawn from the identical distribution as a device-generated
   one. Templated on the RNG so this header keeps no dependency on RandomPark;
   any type with a `double uniform()` works.
------------------------------------------------------------------------- */

template <class RNG>
inline vector<double> generate_unit_quaternion(RNG &rng) {
  double x0, x1, x2, x3, r1, r2;
  do {
    x0 = 2.0 * rng.uniform() - 1.0;
    x1 = 2.0 * rng.uniform() - 1.0;
    r1 = x0 * x0 + x1 * x1;
  } while (r1 >= 1.0);

  do {
    x2 = 2.0 * rng.uniform() - 1.0;
    x3 = 2.0 * rng.uniform() - 1.0;
    r2 = x2 * x2 + x3 * x3;
  } while (r2 >= 1.0);

  const double s = std::sqrt((1.0 - r1) / r2);
  return vector<double>{x0, x1, x2 * s, x3 * s};
}

inline double norm(const vector<double> &q) {
  // if (q.size() != 4)
  //   error->all(FLERR, "vector storing quaternion must have size 4.");

  double init = 0.0;
  auto r = std::inner_product(q.cbegin(), q.cend(), q.cbegin(), init);
  return std::sqrt(r);
}

inline bool check_validity(const vector<double> &q, double epsilon = 1.0e-15) {
  double n = norm(q);
  if (std::fabs(n - 1.0) > epsilon)
    throw std::runtime_error(
        "FAILS unit quaternion test: abs(||q||-1.0)>1.0e-15");
  return true;
}

inline vector<double> vector_cross_product(const vector<double> &q,
                                           const vector<double> &r) {
  if (3 != q.size() || 3 != r.size())
    throw std::runtime_error("len(q)!=3 or len(r)!=3");
  vector<double> s(3);
  vector_cross_product(q.data(), r.data(), s.data());
  return s;
}
inline vector<double> composition(const vector<double> &a,
                                  const vector<double> &b, bool unit = true) {
  // Quaternion composition a b; if unit=true (default) a and b must be unit
  // quaternions.
  if (unit) {
    check_validity(a);
    check_validity(b);
  }
  vector<double> q(4);
  composition(a.data(), b.data(), q.data());
  return q;
}

inline vector<double> conjugate(const vector<double> &q) {
  check_validity(q);
  return vector<double>{q[0], -q[1], -q[2], -q[3]};
}

inline vector<double> rotate_vector(const vector<double> &q,
                                    const vector<double> &r) {
  // Rotates vector 'r' with unit quaternion q; returns len 3.
  if (4 != q.size() || 3 != r.size())
    throw std::runtime_error("len(q)!=4 or len(r)!=3");
  check_validity(q);
  vector<double> rot(3);
  rotate_vector(q.data(), r.data(), rot.data());
  return rot;
}

inline vector<double> to_rotation_matrix(const vector<double> &q) {
  // Row-major 3x3; columns express the local site orientation in the
  // global axes (transpose as needed).
  vector<double> m(9);
  to_rotation_matrix(q.data(), m.data());
  return m;
}

inline double trace_of_rotation_matrix(const vector<double> &q) {
  //     """
  //     Converts quaternion q to 3x3 unitary matrix m
  //     returns trace(m)
  //     """
  //     check_validity(q,unit=True)

  // if (q.size() != 4)
  //   error->all(FLERR, "vector storing quaternion must have size 4.");
  check_validity(q);
  const double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];
  const double q00 = q0 * q0;
  const double qxx = qx * qx;
  const double qyy = qy * qy;
  const double qzz = qz * qz;
  const double r11 = q00 + qxx - qyy - qzz;
  const double r22 = q00 - qxx + qyy - qzz;
  const double r33 = q00 - qxx - qyy + qzz;
  return r11 + r22 + r33;
}

inline vector<double> to_rodrigues_vector(const vector<double> &q,
                                          double epsilon = 1.0e-4) {
  // Converts quaternion q to 3-component Rodrigues vector.
  // see also https://doi.org/10.1107/S0108767391006864
  // If q=(cos(theta/2),sin(theta/2)*[nx,ny,nz])
  //    where n = nx,ny,nz is a unit vector, then
  // the Rodrigues vector  of q is:
  // d := (1/q_0)(q_1; q_2; q_3) = tan(theta/2)*n
  //    where tan(theta/2) = sin(theta/2)/cos(theta/2)
  //    and n is the same unit vector defined above.
  // Returns a vector of length 3.

  // confirm vector length
  if (4 != q.size())
    throw std::runtime_error("len(q)!=4");

  // assign quaternion
  const double q0 = q[0], qx = q[1], qy = q[2], qz = q[3];

  // find norm via angle (see Heinz 1991 eqns 3 & 7)
  const double rho = acos(q0); // theta*2
  const double rtan = tan(rho);
  const double norm = (q0 * rtan);

  // calculate and return Rodrigues vector
  if (norm < epsilon) {
    // if angle is nearing 0
    return std::move(vector<double>{0, 0, 0});
  } else {
    // convert quaternion components to Rodrigues components
    const double rx = qx / norm;
    const double ry = qy / norm;
    const double rz = qz / norm;

    return std::move(vector<double>{rx, ry, rz});
  };
}

inline vector<double> from_bunge_euler_angles(double phi1, double phi,
                                              double phi2) {
  // Get unit quaternion from input Bunge euler angles (degrees).
  // WARNING: this does not look for degeneracies in
  //   the input Euler angle.
  /**
   *
   *  DO NOT USE THIS FUNCTION
   *
   *  Un-tested and may be removed.
   *
   */
  const double c = M_PI / 180;
  const double p1 = phi1 * c;
  const double P = phi * c;
  const double p2 = phi2 * c;
  const double a = (p1 + p2);
  const double b = (p1 - p2);
  vector<double> q{cos(a / 2) * cos(P / 2), sin(a / 2) * cos(P / 2),
                   sin(P / 2) * cos(b / 2), sin(a / 2) * sin(P / 2)};
  // Magnitude of quaternion (should be close to 1)
  double sq = 0.0;
  for (std::size_t j = 0; j < 4; j++) {
    double dj = q[j];
    sq += dj * dj;
  }
  const double mag = sqrt(sq);
  const double d = 1.0 / mag;
  // Normalize
  for (std::size_t j = 0; j < 4; j++) {
    q[j] *= d;
  }
  const double epsilon = 1.0e-10;
  check_validity(q);
  return q;
}

inline vector<double>
generate_random_unit_quaternions(std::size_t n, double epsilon = 1.0e-15) {
  // Computes n random unit quaternions
  // returns vector length = 4 * n
  // ith quaternion uq[4*i:4*i+4] for i=0,1,2,...n
  vector<double> uq(4 * n);

  // Gaussian distribution
  double mean(0.0), std(1.0);
  std::normal_distribution<double> rng(mean, std);
  for (std::size_t i = 0; i < n; i++) {
    double mag = 0.0;
    vector<double> qi{0, 0, 0, 0};
    // 4 random numbers from normal (Gaussian) distribution
    // Make sure to generate non-trivial (non-zero) quaternion
    while (mag < epsilon) {
      double sq = 0.0;
      for (std::size_t j = 0; j < 4; j++) {
        double dj = rng(generator);
        qi[j] = dj;
        sq += dj * dj;
      }
      mag = sqrt(sq);
    }

    // Normalize random quaternion
    double d = 1.0 / mag;
    for (std::size_t j = 0; j < 4; j++) {
      qi[j] *= d;
    }

    // Equivalent to above normalization loop.
    // Normalize using less obvious c++ lambda function
    //   Requires #include<algorithm>
    // std::transform(qi.begin(), qi.end(), qi.begin(),
    //                [&d](auto &c) { return d * c; });

    // Copy normalized quaternion to array of unit quaternions
    check_validity(qi, epsilon);
    for (std::size_t j = 0; j < 4; j++)
      uq[4 * i + j] = qi[j];
  }
  return std::move(uq);
}

inline double
get_cosine_of_minumum_angle_between_q_and_u(const vector<double> &q,
                                            const vector<double> &u,
                                            const vector<double> &dirs) {
  // See cosine_angle_q_u in quaternion_ops.h.
  if (4 != q.size() || 3 != u.size())
    throw std::runtime_error("len(q)!=4 or len(u)!=3");
  return cosine_angle_q_u(q.data(), u.data(), dirs.data(), (int) dirs.size() / 3);
}

inline vector<double> rotate_q_towards_u(const vector<double> &q,
                                        const vector<double> &u,
                                        double angle,
                                        const vector<double> &dirs) {
  // See rotate_q_towards_u in quaternion_ops.h; returns the rotated unit
  // quaternion.
  if (4 != q.size() || 3 != u.size())
    throw std::runtime_error("len(q)!=4 or len(u)!=3");
  check_validity(q);
  if (norm(u) < 1.0e-15)
    throw std::runtime_error("target vector has zero magnitude");
  vector<double> result(4);
  rotate_q_towards_u(q.data(), u.data(), angle, dirs.data(), (int) dirs.size() / 3,
                     result.data());
  return result;
}

} // namespace quaternion

} // namespace SPPARKS_NS
#endif
