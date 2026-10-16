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

#include "app_potts_kokkos.h"

namespace SPPARKS_NS {

// potts/kk with the potts/neigh move: flip to a random unlike neighbour
// spin, with a null bin of maxneigh. Everything else is inherited.
class AppPottsNeighKokkos : public AppPottsKokkos {
public:
  AppPottsNeighKokkos(class SPPARKS *, int, char **);

  void init_app() override;
  void site_event_rejection(int, class RandomPark *) override;
  void sweep_color_kokkos(int color, int start, int count) override;

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
