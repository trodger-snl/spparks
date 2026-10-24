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
#ifndef SPK_CRYSTAL_SYMMETRIES_H
#define SPK_CRYSTAL_SYMMETRIES_H

#include "cubic_symmetries.h"
#include "hcp_symmetries.h"
#include <cmath>
#include <cstring>
#include <vector>
using std::vector;

namespace SPPARKS_NS {

namespace TETRAGONAL {

inline vector<double> get_symmetries() {
  // Tetragonal (point group 422, Laue 4/mmm) rotations, unit quaternions:
  // identity, 90/180/270 degrees about c (z), 180 degrees about a_1 (x),
  // a_2 (y), [110] and [1-10]. All are members of the cubic set.
  const double c = 1.0 / std::sqrt(2.0);
  return {1, 0, 0, 0,   c, 0, 0, c,   0, 0, 0, 1,   c, 0, 0, -c,
          0, 1, 0, 0,   0, 0, 1, 0,   0, c, c, 0,   0, c, -c, 0};
}

} // namespace TETRAGONAL

namespace ORTHORHOMBIC {

inline vector<double> get_symmetries() {
  // Orthorhombic (point group 222, Laue mmm) rotations: identity and
  // 180 degrees about each of a (x), b (y), c (z).
  return {1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1};
}

} // namespace ORTHORHOMBIC

namespace CRYSTAL {

// Crystal names accepted by the quaternion apps, the additive `crystal`
// command and dump_modify ipf, in one place.
static const char *const names[] = {"cubic", "hcp", "trigonal", "tetragonal",
                                    "orthorhombic"};
static const int nnames = 5;

// Index of `name` in `names`, -1 if unknown.
inline int index(const char *name) {
  for (int i = 0; i < nnames; i++)
    if (strcmp(name, names[i]) == 0) return i;
  return -1;
}

// Rotation table (4 doubles per op) for `name`; empty if unknown.
inline vector<double> get_symmetries(const char *name) {
  switch (index(name)) {
    case 0: return CUBIC::get_symmetries();
    case 1: return HCP::get_symmetries();
    case 2: return TRIGONAL::get_symmetries();
    case 3: return TETRAGONAL::get_symmetries();
    case 4: return ORTHORHOMBIC::get_symmetries();
  }
  return {};
}

// Largest disorientation angle (degrees, rounded down) of the group: the
// upper bound for the Read-Shockley theta_cut. Exact values 62.80 (432),
// 93.84 (622), 104.48 (32), 98.42 (422), 120 (222).
inline double max_disorientation(const char *name) {
  static const double maxdeg[] = {62.7, 93.8, 104.4, 98.4, 120.0};
  const int i = index(name);
  return i < 0 ? 0.0 : maxdeg[i];
}

// Growth-direction family: the crystal-frame directions equivalent to g
// under the rotations `sym`, one per +-pair (the growth physics folds the
// sign). Components within 1e-12 of 0 are snapped to 0 before normalizing,
// so axis directions come out exact (cubic g = [001] gives exactly the
// unit x, y, z axes).
inline vector<double> growth_family(const vector<double> &sym, const double g[3]) {
  const double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  vector<double> dirs;
  for (std::size_t k = 0; k + 3 < sym.size(); k += 4) {
    const double w = sym[k], x = sym[k + 1], y = sym[k + 2], z = sym[k + 3];
    const double r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
    double v[3];
    for (int i = 0; i < 3; i++) {
      v[i] = (r[3 * i] * g[0] + r[3 * i + 1] * g[1] + r[3 * i + 2] * g[2]) / gn;
      if (std::fabs(v[i]) < 1e-12) v[i] = 0.0;
    }
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; i++) v[i] /= n;
    bool seen = false;
    for (std::size_t j = 0; j < dirs.size() && !seen; j += 3)
      seen = std::fabs(dirs[j] * v[0] + dirs[j + 1] * v[1] + dirs[j + 2] * v[2]) > 1.0 - 1e-9;
    if (!seen) dirs.insert(dirs.end(), v, v + 3);
  }
  return dirs;
}

// Reach of a growth family (radians): the largest angle between any sample
// direction and its nearest growth direction (cubic <100>: acos(1/sqrt 3)).
// The farthest direction is equidistant from three of the signed axes, so it
// is (a-b)x(a-c) for some triple; a single axis reaches 90 degrees.
inline double family_reach(const vector<double> &dirs) {
  const int n = (int) dirs.size() / 3;
  if (n < 2) return 2.0 * std::atan(1.0);
  vector<double> a;   // the 2n signed axes
  for (int k = 0; k < 3 * n; k++) a.push_back(dirs[k]);
  for (int k = 0; k < 3 * n; k++) a.push_back(-dirs[k]);
  double cmin = 1.0;
  for (int i = 0; i < 2 * n; i++)
    for (int j = i + 1; j < 2 * n; j++)
      for (int k = j + 1; k < 2 * n; k++) {
        const double *p = &a[3 * i], *q = &a[3 * j], *r = &a[3 * k];
        const double e[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
        const double f[3] = {p[0] - r[0], p[1] - r[1], p[2] - r[2]};
        double u[3] = {e[1] * f[2] - e[2] * f[1], e[2] * f[0] - e[0] * f[2],
                       e[0] * f[1] - e[1] * f[0]};
        const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (un < 1e-12) continue;
        double c = 0.0;
        for (int m = 0; m < n; m++)
          c = std::fmax(c, std::fabs(u[0] * dirs[3 * m] + u[1] * dirs[3 * m + 1] +
                                     u[2] * dirs[3 * m + 2]) / un);
        cmin = std::fmin(cmin, c);
      }
  return std::acos(cmin);
}

} // namespace CRYSTAL

} // namespace SPPARKS_NS

#endif
