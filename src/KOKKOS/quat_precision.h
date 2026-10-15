/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifndef SPK_QUAT_PRECISION_H
#define SPK_QUAT_PRECISION_H

/* ----------------------------------------------------------------------
   QReal: precision of the potts/quaternion/kk disorientation and
   Read-Shockley energy math (QuaternionOps::compute_disorientation,
   QuaternionOps::read_shockley), on device and host alike.

   Default double: reproduces the previous arithmetic exactly.
   -DSPK_QUAT_FP32 (CMake -DSPPARKS_QUAT_FP32=ON) makes it float: a large
   win on RDNA parts where FP64 rate is reduced, at the cost of bit-exact
   continuity with FP64 runs. The float path takes the angle as
   2 atan2(|v|, |m0|) of the best-symmetry misorientation rather than acos
   of the trace, so small angles stay accurate (~1e-5 deg) in float.

   Deliberately NOT QReal: the stored quaternions and the symmetry table
   (double views), and the per-site energy sums.
------------------------------------------------------------------------- */

namespace SPPARKS_NS {

#ifdef SPK_QUAT_FP32
typedef float  QReal;
#else
typedef double QReal;
#endif

}

#endif
