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
AppStyle(sinter/kk,AppSinterKokkos)

#else

#ifndef SPK_APP_SINTER_KOKKOS_H
#define SPK_APP_SINTER_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_sinter.h"

namespace SPPARKS_NS {

// sinter on the device: the colour sweep runs sinter::event per site, as
// the CPU sweep does; of the end-of-sweep annihilation, the passes over the
// sites are kernels and the rest (paths, collapse) is the CPU app's.
class AppSinterKokkos : public LatticeKokkos<AppSinter> {
 public:
  AppSinterKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  // define device lambdas, so public (nvcc)
  void sweep_color_kokkos(int, int, int) override;
  int nmade() override;
  void update_ghosts() override;
  void candidates(std::vector<Vac> &) override;
  void grain_sums(const std::vector<int> &, std::vector<long long> &) override;
  void get_sites(const std::vector<int> &, std::vector<int> &) override;
  void set_sites(const std::vector<int> &, const std::vector<int> &) override;
  void gb_faces(long long &, std::vector<unsigned long long> &,
                std::vector<int> &) override;

  Kokkos::View<int*> d_spin;
  Kokkos::View<tagint*> d_id;
  Kokkos::View<int*> d_made;     // vacancies the sweep made
  Kokkos::View<int> d_nmade;
};

}

#endif
#endif
