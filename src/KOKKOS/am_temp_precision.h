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
   Compile-time precision selector for the additive/texture DEVICE
   temperature-evaluation math (the per-site Rosenthal / Moser transcendental
   kernels -- the exp/sqrt-heavy inner work that dominates GPU cost).

   AMReal is the type those inner kernels accumulate in. It defaults to
   `double` (FP64):
     - FP64 is the HPC/CDNA (MI-series) production precision, where double is
       full- or high-rate, and
     - it is the bit-for-bit reference the CPU `additive/texture` app is
       validated against (see CLAUDE.md "Validating the GPU port").

   Build with  -DSPK_AM_TEMP_FP32  (CMake: -DSPPARKS_AM_TEMP_FP32=ON) to switch
   the eval math to `float` (FP32). On RDNA/consumer GPUs (gfx1151, Strix Halo)
   FP64 throughput is heavily reduced relative to FP32, so FP32 is a large
   speedup for these compute-bound kernels -- at the cost of the bit-exact
   temperature match (compare with a tolerance instead).

   Scope is deliberately narrow: only the transcendental kernel math is carried
   in AMReal. The k_T field and host-side plan PODs (RosenthalDevicePlan /
   MoserDevicePlan) stay double, and the cancellation-prone site-minus-laser
   geometry differences are computed in double before the small pool-local
   coordinates are cast to AMReal. This isolates the precision knob to exactly
   the work that benefits, and keeps the FP64 default bit-for-bit identical to
   the pre-existing code path.
------------------------------------------------------------------------- */

namespace SPPARKS_NS {

#ifdef SPK_AM_TEMP_FP32
typedef float  AMReal;
#else
typedef double AMReal;
#endif

}

#endif
