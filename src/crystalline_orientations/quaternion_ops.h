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
// Quaternion math on raw arrays, compiled for host and device: the one
// implementation behind the vector API in quaternion.h (CPU apps) and the
// Kokkos kernels. Scalar-first: q = (cos(theta/2), sin(theta/2) axis).
#ifndef SPK_QUATERNION_OPS_H
#define SPK_QUATERNION_OPS_H

#include <math.h>

#if defined(SPK_KOKKOS) && __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif
#ifdef KOKKOS_INLINE_FUNCTION
#define QOPS_FN KOKKOS_INLINE_FUNCTION
#else
#define QOPS_FN inline
#endif

namespace SPPARKS_NS {
namespace quaternion {

QOPS_FN void normalize(double *q) {
  double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n > 1.0e-15) {
    double inv = 1.0 / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
  }
}

// result = a b
QOPS_FN void composition(const double *a, const double *b, double *result) {
  double ar = a[0], ax = a[1], ay = a[2], az = a[3];
  double br = b[0], bx = b[1], by = b[2], bz = b[3];
  result[0] = ar * br - (ax * bx + ay * by + az * bz);
  result[1] = ar * bx + br * ax + (ay * bz - az * by);
  result[2] = ar * by + br * ay - (ax * bz - az * bx);
  result[3] = ar * bz + br * az + (ax * by - ay * bx);
}

QOPS_FN void conjugate(const double *q, double *result) {
  result[0] = q[0];
  result[1] = -q[1];
  result[2] = -q[2];
  result[3] = -q[3];
}

QOPS_FN void vector_cross_product(const double *a, const double *b, double *result) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = -(a[0] * b[2] - a[2] * b[0]);
  result[2] = a[0] * b[1] - a[1] * b[0];
}

// r rotated by q: r + 2 q0 (v x r) + 2 v x (v x r), v = vector part of q
QOPS_FN void rotate_vector(const double *q, const double *r, double *result) {
  double v[3] = {q[1], q[2], q[3]};
  double c[3], c2[3];
  vector_cross_product(v, r, c);
  vector_cross_product(v, c, c2);
  result[0] = r[0] + 2.0 * q[0] * c[0] + 2.0 * c2[0];
  result[1] = r[1] + 2.0 * q[0] * c[1] + 2.0 * c2[1];
  result[2] = r[2] + 2.0 * q[0] * c[2] + 2.0 * c2[2];
}

// Row-major 3x3 R(q); its columns are the crystal axes in the sample frame.
QOPS_FN void to_rotation_matrix(const double *q, double *m) {
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
  m[0] = q00 + qxx - qyy - qzz;
  m[1] = 2 * (qxy - q0z);
  m[2] = 2 * (qxz + q0y);
  m[3] = 2 * (qxy + q0z);
  m[4] = q00 - qxx + qyy - qzz;
  m[5] = 2 * (qyz - q0x);
  m[6] = 2 * (qxz - q0y);
  m[7] = 2 * (qyz + q0x);
  m[8] = q00 - qxx - qyy + qzz;
}

// Cosine of the smallest angle between u and any growth direction of the
// crystal q, ignoring sign. dirs: crystal-frame directions, 3 per direction,
// one per +-pair (cubic <100>: the unit x, y, z axes). Crystal-frame v lies
// along R(q) v in the sample frame, and (R(q) v).n = (R(q)^T n).v.
QOPS_FN double cosine_angle_q_u(const double *q, const double *u, const double *dirs,
                                int ndirs) {
  double n[3] = {u[0], u[1], u[2]};
  double d = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (d > 1.0e-15) {
    double inv = 1.0 / d;
    n[0] *= inv;
    n[1] *= inv;
    n[2] *= inv;
  }
  double m[9];
  to_rotation_matrix(q, m);
  // r = transpose(m) n
  double r[3];
  for (int col = 0; col < 3; col++) {
    double dot = 0.0;
    for (int row = 0; row < 3; row++) dot += m[col + 3 * row] * n[row];
    r[col] = dot;
  }
  double max = 0.0;
  for (int k = 0; k < ndirs; k++) {
    const double *v = &dirs[3 * k];
    const double dot = fabs(r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
    if (dot > max) max = dot;
  }
  return max;
}

// Rotate q by `angle` (radians) so that its growth direction nearest u
// (either sign) turns toward u, stopping once aligned; result is normalized.
// A zero u returns q unchanged.
QOPS_FN void rotate_q_towards_u(const double *q, const double *u, double angle,
                                const double *dirs, int ndirs, double *result) {
  result[0] = q[0]; result[1] = q[1]; result[2] = q[2]; result[3] = q[3];
  double n[3] = {u[0], u[1], u[2]};
  double d = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (d < 1.0e-15) return;
  double inv_d = 1.0 / d;
  n[0] *= inv_d;
  n[1] *= inv_d;
  n[2] *= inv_d;

  // nearest growth direction, in the sample frame, sign-corrected toward n
  double m[9];
  to_rotation_matrix(q, m);
  double r[3];
  for (int col = 0; col < 3; col++) r[col] = m[col] * n[0] + m[col + 3] * n[1] + m[col + 6] * n[2];
  double best_dot = 0.0;
  int best = 0;
  for (int k = 0; k < ndirs; k++) {
    const double *v = &dirs[3 * k];
    const double dot = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    if (fabs(dot) > fabs(best_dot)) { best_dot = dot; best = k; }
  }
  const double sign = (best_dot >= 0.0) ? 1.0 : -1.0;
  const double *v = &dirs[3 * best];
  double cur[3];
  for (int row = 0; row < 3; row++)
    cur[row] = sign * (m[3 * row] * v[0] + m[3 * row + 1] * v[1] + m[3 * row + 2] * v[2]);

  double dot_product = cur[0] * n[0] + cur[1] * n[1] + cur[2] * n[2];
  dot_product = fmax(-1.0, fmin(1.0, dot_product));
  double misorientation = acos(dot_product);
  if (misorientation < 1.0e-10) return;
  double actual_angle = fmin(fabs(angle), misorientation);

  double axis[3];
  vector_cross_product(cur, n, axis);
  double axis_norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  if (axis_norm < 1.0e-10) {
    if (dot_product > 0.99999) return;
    // antiparallel: any axis perpendicular to n
    double tmp[3] = {0.0, 0.0, 0.0};
    tmp[fabs(n[0]) < 0.9 ? 0 : 1] = 1.0;
    vector_cross_product(n, tmp, axis);
    axis_norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  }
  double inv_axis = 1.0 / axis_norm;
  axis[0] *= inv_axis;
  axis[1] *= inv_axis;
  axis[2] *= inv_axis;

  double half = actual_angle * 0.5;
  double s = sin(half);
  double rot_q[4] = {cos(half), s * axis[0], s * axis[1], s * axis[2]};
  composition(rot_q, q, result);
  normalize(result);
}

} // namespace quaternion
} // namespace SPPARKS_NS

#endif
