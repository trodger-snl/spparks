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
AppStyle(potts/neighonly/kk,AppPottsNeighOnlyKokkos)

#else

#ifndef SPK_APP_POTTS_NEIGHONLY_KOKKOS_H
#define SPK_APP_POTTS_NEIGHONLY_KOKKOS_H

#include "app_potts_neighonly.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppPottsNeighOnlyKokkos : public AppPottsNeighOnly {
 public:
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

  AppPottsNeighOnlyKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsNeighOnlyKokkos();

  // Kokkos-specific data
  Kokkos::DualView<int*> k_spin;
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

private:
  // Device-side helper functors
  struct RejectionSweepFunctor;

  // Sync helpers
  void sync_spin_to_device();
  void sync_spin_to_host();
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
