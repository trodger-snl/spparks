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
  double qx = q[0], qy = q[1], qz = q[2];
  double rx = r[0], ry = r[1], rz = r[2];
  vector<double> s{qy * rz - qz * ry, -(qx * rz - qz * rx), qx * ry - qy * rx};
  return std::move(s);
}
inline vector<double> composition(const vector<double> &a,
                                  const vector<double> &b, bool unit = true) {
  // Computes quaternion composition a b
  //   args: a quaternion array len=4.
  //   args: b quaternion array len=4.
  //   if unit=true (default) a and b must be unit quaternions
  //   returns: quaternion array len=4;

  // if (a.size() != 4 || b.size() != 4)
  //   error->all(FLERR, "vector storing quaternion must have size 4.");

  if (unit) {
    check_validity(a);
    check_validity(b);
  }
  double ar = a[0];
  double br = b[0];
  double ax = a[1], ay = a[2], az = a[3];
  double bx = b[1], by = b[2], bz = b[3];
  // scalar qr=ar br - a[1:] dot b[1:]
  double qr = ar * br - (ax * bx + ay * by + az * bz);
  vector<double> cross{ay * bz - az * by, -(ax * bz - az * bx),
                       ax * by - ay * bx};
  // 3d vector qv=ar b[1:] + br a[1:] - a[1:] cross b[1:]
  vector<double> qv{ar * bx + br * ax + cross[0], ar * by + br * ay + cross[1],
                    ar * bz + br * az + cross[2]};
  return std::move(vector<double>{qr, qv[0], qv[1], qv[2]});
}

inline vector<double> conjugate(const vector<double> &q) {
  // if (q.size() != 4)
  //   error->all(FLERR, "vector storing quaternion must have size 4.");
  check_validity(q);
  return std::move(vector<double>{q[0], -q[1], -q[2], -q[3]});
}

inline vector<double> rotate_vector(const vector<double> &q,
                                    const vector<double> &r) {
  // """
  // Rotates vector 'r' with input quaternion q.
  // args: q unit quaternion array len=4.
  // args: r array len=3.
  // return value array len=3
  // Vec(r)+2 Real(q)Vec(q) CROSS Vec(r)+2 Vec(q) CROSS (Vec(q) CROSS Vec(r))
  // """
  if (4 != q.size() || 3 != r.size())
    throw std::runtime_error("len(q)!=4 or len(r)!=3");
  check_validity(q);
  // Real and vector parts of q
  double qr = q[0];
  vector<double> qv{q[1], q[2], q[3]};
  // rot=r+2*qr*c+2*numpy.cross(qv,c)
  vector<double> c = vector_cross_product(qv, r);
  vector<double> c2 = vector_cross_product(qv, c);
  vector<double> rot{r[0] + 2.0 * qr * c[0] + 2.0 * c2[0],
                     r[1] + 2.0 * qr * c[1] + 2.0 * c2[1],
                     r[2] + 2.0 * qr * c[2] + 2.0 * c2[2]};
  return std::move(rot);
}

inline vector<double> to_rotation_matrix(const vector<double> &q) {
  //     """
  //     Converts quaternion q to 3x3 unitary matrix
  //     returns 'row major' 3x3 matrix.
  //
  //     Return value: Columns of matrix express local site orientation
  //     (per input quaternion) with respect to problem
  //     global coordinate axes.  Depending on context, it may
  //     be necessary to transpose this matrix.
  //     """
  //     check_validity(q,unit=True)

  // if (q.size() != 4)
  //   error->all(FLERR, "vector storing quaternion must have size 4.");
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
  const double r11 = q00 + qxx - qyy - qzz;
  const double r22 = q00 - qxx + qyy - qzz;
  const double r33 = q00 - qxx - qyy + qzz;
  const double r12 = 2 * (qxy - q0z);
  const double r21 = 2 * (qxy + q0z);
  const double r13 = 2 * (qxz + q0y);
  const double r31 = 2 * (qxz - q0y);
  const double r23 = 2 * (qyz - q0x);
  const double r32 = 2 * (qyz + q0x);
  vector<double> m{r11, r12, r13, r21, r22, r23, r31, r32, r33};
  return std::move(m);
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
                                            const vector<double> &u) {
  /**
   *
   *  DO NOT USE THIS FUNCTION
   *
   *  Un-tested and may be removed.
   *
      arg q: array shape=(4,) unit quaterion
      arg u: array shape=(3,) unit vector

      Measure/compute cosine of angle between input unit quaterion 'q'
      representing a cube and the direction 'u'.  Input vector makes 3 different
      angles with local cubic coordinate axes attached to cube.  The cubic
      coordinate axes are normal to each of the faces.  Calculation of angle
      makes no distinction between positive or negative local coordinate axes of
      cube.   Intended application is to determine angle between u and line
   taken normal to face of cube where u is temperature gradient.

      returns cos(angle)
   *
   *
   */
  if (4 != q.size() || 3 != u.size())
    throw std::runtime_error("len(q)!=4 or len(u)!=3");

  // Take a copy of 'u' and normalize
  vector<double> n{u[0], u[1], u[2]};
  {
    double d = norm(n);
    double b = 1 / d;
    n[0] *= b;
    n[1] *= b;
    n[2] *= b;
  }
  vector<double> m = to_rotation_matrix(q);
  // Dot columns of 'm' with 'n'; which is a matrix
  // multiplication of transpose(m) X n
  double max = 0.0;
  // Because 'm' is an array in 'row-major' form -- need
  // for matrix multiply 'transpose(m)' times 'n'.
  for (size_t col = 0; col < 3; col++) {
    double dot = 0.0;
    for (size_t row = 0; row < 3; row++) {
      size_t ptr = col + 3 * row;
      dot += m[ptr] * n[row];
    }
    // Take absolute value
    dot = std::fabs(dot);
    if (dot > max)
      max = dot;
  }
  return max;
}

inline vector<double> rotate_q_towards_u(const vector<double> &q,
                                        const vector<double> &u,
                                        double angle) {
  /**
  Rotates quaternion 'q' towards vector 'u' by the specified angle.
  If the input angle exceeds the misorientation between the current
  orientation and target vector, the rotation stops when they are aligned.
  
  arg q: array shape=(4,) unit quaternion to be rotated
  arg u: array shape=(3,) target vector direction
  arg angle: rotation angle in radians
  
  returns: array shape=(4,) rotated unit quaternion
  */
  if (4 != q.size() || 3 != u.size())
    throw std::runtime_error("len(q)!=4 or len(u)!=3");
  
  check_validity(q);
  
  // Normalize the target vector
  vector<double> n{u[0], u[1], u[2]};
  {
    double d = norm(n);
    if (d < 1.0e-15)
      throw std::runtime_error("target vector has zero magnitude");
    double b = 1.0 / d;
    n[0] *= b;
    n[1] *= b;
    n[2] *= b;
  }
  
  // Get the rotation axis
  // The rotation axis is perpendicular to both the current orientation
  // and the target direction. We'll use the quaternion's principal axis
  // (the vector part when angle is non-zero) and cross it with the target.
  
  // Find which <100> axis is closest to the target direction.
  // For a cubic crystal, [100], [010], [001] are equivalent growth
  // directions.  We want to rotate the nearest one toward the target,
  // not the hardcoded [001].
  vector<double> m = to_rotation_matrix(q);
  // Columns of m are the lab-frame directions of [100], [010], [001].
  // m is row-major: column 'col' has components m[col], m[col+3], m[col+6].
  double best_dot = 0.0;
  int best_col = 0;
  for (int col = 0; col < 3; col++) {
      double dot = m[col] * n[0] + m[col + 3] * n[1] + m[col + 6] * n[2];
      if (std::fabs(dot) > std::fabs(best_dot)) {
          best_dot = dot;
          best_col = col;
      }
  }
  // current_dir = lab-frame direction of the nearest <100>, sign-corrected
  // so it points toward (not away from) the target.
  double sign = (best_dot >= 0.0) ? 1.0 : -1.0;
  vector<double> current_dir{sign * m[best_col],
                             sign * m[best_col + 3],
                             sign * m[best_col + 6]};
  
  // Calculate the current misorientation angle
  double dot_product = current_dir[0]*n[0] + current_dir[1]*n[1] + current_dir[2]*n[2];
  // Clamp dot product to avoid numerical issues with acos
  dot_product = std::max(-1.0, std::min(1.0, dot_product));
  double current_misorientation = std::acos(dot_product);
  
  // Check if already aligned (within tolerance)
  if (current_misorientation < 1.0e-10) {
    return vector<double>{q[0], q[1], q[2], q[3]};
  }
  
  // Limit the rotation angle to not exceed the current misorientation
  double actual_angle = std::min(std::fabs(angle), current_misorientation);
  
  // Calculate the rotation axis as the cross product
  vector<double> rot_axis = vector_cross_product(current_dir, n);
  
  // Check if vectors are parallel (cross product near zero)
  double axis_norm = norm(rot_axis);
  if (axis_norm < 1.0e-10) {
    // Vectors are parallel, no rotation needed or 180° rotation
    // Check if they point in the same direction
    if (dot_product > 0.99999) {
      // Same direction, return original quaternion
      return vector<double>{q[0], q[1], q[2], q[3]};
    } else {
      // Opposite directions (180° apart), need to find a perpendicular axis
      // Use any perpendicular vector
      if (std::fabs(n[0]) < 0.9) {
        rot_axis = vector<double>{1.0, 0.0, 0.0};
      } else {
        rot_axis = vector<double>{0.0, 1.0, 0.0};
      }
      rot_axis = vector_cross_product(n, rot_axis);
      axis_norm = norm(rot_axis);
    }
  }
  
  // Normalize the rotation axis
  rot_axis[0] /= axis_norm;
  rot_axis[1] /= axis_norm;
  rot_axis[2] /= axis_norm;
  
  // Create the rotation quaternion for the actual angle
  double half_angle = actual_angle * 0.5;
  double cos_half = std::cos(half_angle);
  double sin_half = std::sin(half_angle);
  
  vector<double> rot_q{cos_half, 
                      sin_half * rot_axis[0],
                      sin_half * rot_axis[1], 
                      sin_half * rot_axis[2]};
  
  // Apply the rotation to the original quaternion
  vector<double> result = composition(rot_q, q);
  
  return result;
}

} // namespace quaternion

} // namespace SPPARKS_NS
#endif
