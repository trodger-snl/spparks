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

#ifndef SPK_APP_LATTICE_KOKKOS_H
#define SPK_APP_LATTICE_KOKKOS_H

#include "app_lattice.h"
#include "app_kokkos.h"

namespace SPPARKS_NS {

class AppLatticeKokkos : public AppLattice {
public:
  AppLatticeKokkos(class SPPARKS *, int, char **);
  virtual ~AppLatticeKokkos();

  // Kokkos neighbor data
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;
  Kokkos::DualView<double*> k_propensity;

  // Random number generation pool
  using pool_type = Kokkos::Random_XorShift64_Pool<>;
  pool_type *rand_pool;

  // Virtual functions to override with Kokkos implementations
  // These are the core computational kernels that apps must implement
  virtual double site_energy_kokkos(int) = 0;
  virtual void site_event_rejection_kokkos(int, class RandomPark *) = 0;
  virtual double site_propensity_kokkos(int) = 0;

  // Bulk operations on device - to be implemented by derived classes
  virtual void compute_all_energies_kokkos() {}
  virtual void compute_all_propensities_kokkos() {}
  virtual void sweep_rejection_kokkos() {}

  // Sync helpers
  void sync_neighbor_data_to_device();
  void sync_neighbor_data_to_host();
  void allocate_kokkos_neighbor_arrays();
  void sync_neighbor_arrays_host_to_device();

protected:
  // Helper to copy data between host arrays and DualViews
  void copy_neighbor_data_to_host();
};

}

#endif
