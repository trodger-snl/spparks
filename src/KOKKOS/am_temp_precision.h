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

   ----------------------------------------------------------------------
   HDF5Real is the same knob for the HDF5 unstructured-mesh device path
   (fill_temperature_hdf5_device). That path is NOT transcendental -- it is a
   gather-and-blend: per site, read 3 barycentric weights and 4 entries out of
   the per-node temperature slice. It is therefore MEMORY-bound, not FP64-rate
   bound, and FP32 buys halved traffic on the arrays the hot kernel streams
   rather than faster arithmetic. Same flag drives both so one switch describes
   "the AM device math runs in FP32".

   Carried in HDF5Real: the per-node temperature slice (d_hdf5_nodal_temps), the
   per-site barycentric weights (d_hdf5_weights), and the raw per-node
   temperature samples (d_ts_temps) -- all bounded, well-scaled quantities
   (T ~ 300-3000 K; weights in [0,1]) where float's ~7 significant digits are
   ample.

   DELIBERATELY still double, and NOT to be "optimized" into HDF5Real:

     - d_ts_times and HDF5DevicePlan::sim_time. These are ABSOLUTE simulation
       times (~3.9e4 s in the ct15 decks) advanced on a ~3e-6 s timestep. float
       resolves ~4e-3 s at that magnitude -- coarser than the timestep by three
       orders of magnitude and than a per-box dwell (0.1-0.3 s) by two. Casting
       them collapses the ComputeNodalTemps binary-search bracket and destroys
       the interpolation. The time math stays FP64 and only the interpolated
       RESULT is narrowed.

     - k_T / d_T (the temperature field itself). It is a DualView aliased to the
       host darray[5] (`d6`), read by every microstructure kernel and by the host
       diagnostics; narrowing it is a far wider blast radius for a second
       increment of bandwidth. The fill computes in HDF5Real and stores to
       double.

   As with AMReal, FP32 forfeits the bit-for-bit match against the CPU
   `additive/texture` app that this port is validated by -- compare on a
   tolerance instead. FP64 remains the default and is bit-for-bit unchanged:
   every use below is written so that HDF5Real == double reproduces the previous
   arithmetic exactly.
------------------------------------------------------------------------- */

namespace SPPARKS_NS {

#ifdef SPK_AM_TEMP_FP32
typedef float  AMReal;
typedef float  HDF5Real;
#else
typedef double AMReal;
typedef double HDF5Real;
#endif

}

#endif
