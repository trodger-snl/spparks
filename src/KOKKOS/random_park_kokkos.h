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

/* ----------------------------------------------------------------------
   RandomParkDevice -- a device-callable copy of SPPARKS' Park/Miller RNG
   (src/random_park.cpp), for the per-site "strict" sweep.

   WHY THIS EXISTS. The Kokkos sweeps normally draw from a
   Kokkos::Random_XorShift64_Pool. get_state() picks a generator by hardware
   thread index and, on contention, spins to the NEXT free one
   (Kokkos_Random.hpp get_state_idx). Which generator a site ends up with is
   therefore decided by a lock race, so the same site draws from a different
   stream on every run -- the seed is fixed (12345+me), but the site->stream
   mapping is not. That is what makes large runs irreproducible.

   Under `sweep color/strict` the stream is instead keyed to the site's own
   persistent seed (AppLattice seeds those from the GLOBAL site id via
   RandomPark::tagreset), so it depends on nothing but the site itself: no
   thread ids, no scheduling, no lock order.

   The arithmetic below is a line-for-line copy of RandomPark::uniform(). It
   must stay that way -- matching it is what lets the device sweep reproduce
   the CPU strict sweep bit-for-bit.
------------------------------------------------------------------------- */

#ifndef SPK_RANDOM_PARK_KOKKOS_H
#define SPK_RANDOM_PARK_KOKKOS_H

#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

struct RandomParkDevice {
  // Park/Miller constants, as in src/random_park.cpp
  static constexpr int PM_IA = 16807, PM_IM = 2147483647, PM_IQ = 127773,
                       PM_IR = 2836;

  int seed;

  KOKKOS_INLINE_FUNCTION
  RandomParkDevice() : seed(1) {}

  KOKKOS_INLINE_FUNCTION
  explicit RandomParkDevice(int s) : seed(s) {}

  KOKKOS_INLINE_FUNCTION
  double uniform() {
    const int k = seed / PM_IQ;
    seed = PM_IA * (seed - k * PM_IQ) - PM_IR * k;
    if (seed < 0) seed += PM_IM;
    return (1.0 / PM_IM) * seed;
  }

  // Name-compatible with Kokkos' generator so the sweep body can be written
  // once and templated over which RNG it is handed.
  KOKKOS_INLINE_FUNCTION
  double drand() { return uniform(); }
};

}  // namespace SPPARKS_NS

#endif
