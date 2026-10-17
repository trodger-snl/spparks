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
AppStyle(potts/kk,AppPottsKokkos)

#else

#ifndef SPK_APP_POTTS_KOKKOS_H
#define SPK_APP_POTTS_KOKKOS_H

#include "app_lattice_kokkos.h"

namespace SPPARKS_NS {

class AppPottsKokkos : public AppLatticeKokkos {
public:
  AppPottsKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsKokkos();

  // App-specific data
  int nspins;
  int *spin;                              // Host pointer (from iarray[0])
  Kokkos::View<int*> d_spin;              // Device view

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;

  // Implement required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  virtual void allocate_app_views() override;
  virtual void sweep_color_kokkos(int color, int start, int count) override;

  // Site methods (required by AppLattice base)
  virtual double site_energy(int) override;
  virtual void site_event_rejection(int, class RandomPark *) override;
  virtual double site_propensity(int) override;
  virtual void site_event(int, class RandomPark *) override;

protected:
  // Temporary arrays for KMC (allocated if allow_kmc)
  int *sites, *unique;

public:
  // Public for nvcc: a __global__ kernel's template argument must be accessible.
  struct ColorSweepFunctor;
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

E: Kokkos app requires 'sweep color' or 'sweep color/strict' for correct GPU execution

GPU execution requires coloring to prevent race conditions. Add 'sweep color'
to your input script.

*/
