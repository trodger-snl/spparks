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

#ifdef APP_CLASS
AppStyle(potts/neigh/kk,AppPottsNeighKokkos)

#else

#ifndef SPK_APP_POTTS_NEIGH_KOKKOS_H
#define SPK_APP_POTTS_NEIGH_KOKKOS_H

#include "app_potts_neigh.h"
#include "kokkos_double_buffer.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppPottsNeighKokkos : public AppPottsNeigh {
 public:
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

  AppPottsNeighKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsNeighKokkos();

  // Kokkos-specific data
  KokkosDoubleBuffer<int> buffer_spin;
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;
  Kokkos::DualView<double*> k_propensity;

  // Random number generation pool
  pool_type *rand_pool;

  // Override base class methods with Kokkos versions
  virtual void grow_app();
  virtual void init_app();
  virtual void iterate_rejection(double);

  // Kokkos-specific kernel implementations
  void sweep_rejection_kokkos();
  void sweep_rejection_kokkos(int, int *);

  // Profiling helpers
  void enable_sync_profiling(bool enable = true) { buffer_spin.enable_profiling(enable); }
  void reset_sync_profiling() { buffer_spin.reset_profiling(); }
  void print_sync_profiling_stats(FILE* fp) { buffer_spin.print_profiling_stats(fp, "spin"); }

private:
  // Device-side helper functors
  struct RejectionSweepFunctor;

  // Sync helpers
  void sync_data_to_device();           // MPI version - updates both buffers
  void sync_data_to_device_full();      // Full version - updates both buffers (initialization)
  void sync_data_to_host();
  void swap_buffers();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPPARKS to see the offending
line.

E: One or more sites have invalid values

The application only allows sites to be initialized with specific
values.

*/
