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
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppLatticeKokkos : public AppLattice {
public:
  AppLatticeKokkos(class SPPARKS *, int, char **);
  virtual ~AppLatticeKokkos();

  // === Kokkos type aliases ===
  using pool_type = Kokkos::Random_XorShift64_Pool<>;
  using view_int = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>;
  using view_double = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

  // === Shared Kokkos data (all derived classes use these) ===
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;
  pool_type *rand_pool;

  // === Color-based site lists for GPU ===
  int kokkos_ncolors;                    // Number of colors (8 for 26-neighbor)
  Kokkos::View<int*> d_color_offsets;    // CSR offsets [ncolors+1]
  Kokkos::View<int*> d_color_sites;      // Site indices by color [nlocal]
  bool color_lists_initialized;

  // === Shared methods ===
  virtual void init_app() override;
  virtual void grow_app() override;
  void setup_color_lists_kokkos();
  virtual void iterate_rejection(double stoptime) override;

  // === Virtual methods derived classes MUST implement ===
  // Sync app-specific data (spin, quaternions, etc.)
  virtual void sync_app_data_to_device() = 0;
  virtual void sync_app_data_to_host() = 0;

  // Allocate app-specific Kokkos views
  virtual void allocate_app_views() = 0;

  // The actual sweep kernel for one color
  virtual void sweep_color_kokkos(int color, int start, int count) = 0;

protected:
  // Shared neighbor sync
  void sync_neighbors_to_device();
};

}

#endif
