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

#ifndef SPK_APP_KOKKOS_H
#define SPK_APP_KOKKOS_H

#include "app.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppKokkos : public App {
public:
  AppKokkos(class SPPARKS *, int, char **);
  virtual ~AppKokkos();

  // Kokkos versions of site data
  Kokkos::DualView<tagint*> k_id;           // Site IDs
  Kokkos::DualView<double*[3]> k_xyz;       // Coordinates
  Kokkos::DualView<int**> k_iarray;         // Integer site arrays
  Kokkos::DualView<double**> k_darray;      // Double site arrays

  // Random number generation pool
  using pool_type = Kokkos::Random_XorShift64_Pool<>;
  pool_type *rand_pool;

  // Memory sync control
  void sync_to_device();
  void sync_to_host();
  void modify_device();
  void modify_host();

  // Execution space query
  bool is_device_execution();

protected:
  // Helper function to allocate Kokkos views
  void allocate_kokkos_arrays();
  void sync_arrays_host_to_device();
  void sync_arrays_device_to_host();
};

}

#endif
