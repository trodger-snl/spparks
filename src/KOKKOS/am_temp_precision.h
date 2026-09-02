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

#ifndef SPK_AM_TEMP_PRECISION_H
#define SPK_AM_TEMP_PRECISION_H

/* ----------------------------------------------------------------------
   AMReal: precision of the additive/texture DEVICE temperature math -- the
   Rosenthal/Moser transcendental inner kernels and the HDF5 gather-and-blend
   (weights, nodal temperature slice, raw temperature samples).

   Default double: the HPC/CDNA production precision and the bit-for-bit
   reference against the CPU additive/texture app. -DSPK_AM_TEMP_FP32
   (CMake -DSPPARKS_AM_TEMP_FP32=ON) makes it float: a large win on RDNA
   parts where FP64 rate is reduced (Rosenthal/Moser) and halved traffic on
   the memory-bound HDF5 fill -- at the cost of the bit-exact match.

   Deliberately NOT AMReal: the k_T field (aliased to host d6, read
   everywhere), the plan PODs, the cancellation-prone site-minus-laser
   differences (computed in double, then cast), and absolute simulation
   times (d_ts_times, HDF5DevicePlan::sim_time: ~4e4 s on a 3e-6 s step,
   which float cannot bracket). Every AMReal use is written so that
   AMReal == double reproduces the previous arithmetic exactly.
------------------------------------------------------------------------- */

namespace SPPARKS_NS {

#ifdef SPK_AM_TEMP_FP32
typedef float  AMReal;
#else
typedef double AMReal;
#endif

}

#endif
