/* -*- c++ -*- ----------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifndef SPK_GPU_AWARE_MPI_H
#define SPK_GPU_AWARE_MPI_H

#include <Kokkos_Core.hpp>
#include <cstdlib>

namespace SPPARKS_NS {

// Hand MPI device buffers directly (GPU-aware MPI) when SPPARKS_GPU_AWARE_MPI
// is set to a nonzero value; otherwise Kokkos halo exchanges stage their
// packed buffers through host memory (MPIStagingSpace). Off by default: an MPI built
// without GPU support crashes on a device pointer.
inline bool gpu_aware_mpi()
{
  static const bool on = [] {
    const char *e = std::getenv("SPPARKS_GPU_AWARE_MPI");
    return e && std::atoi(e) != 0;
  }();
  return on;
}

// Host memory that MPI messages are staged through when it cannot read the
// device buffers: pinned on CUDA (full-speed, asynchronous copies). Not on
// HIP, whose pinned space is fine-grained: staging the conduction halo
// through it made a 4-rank run on one gfx1151 1.8x slower than pageable.
#ifdef KOKKOS_ENABLE_CUDA
using MPIStagingSpace = Kokkos::CudaHostPinnedSpace;
#else
using MPIStagingSpace = Kokkos::HostSpace;
#endif

}

#endif
