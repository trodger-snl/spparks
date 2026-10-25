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

#ifdef DIAG_CLASS
DiagStyle(sinter_free_energy_pore,DiagSinterFreeEnergyPore)

#else

#ifndef SPK_DIAG_SINTER_FREE_ENERGY_PORE_H
#define SPK_DIAG_SINTER_FREE_ENERGY_PORE_H

#include "diag.h"

namespace SPPARKS_NS {

class DiagSinterFreeEnergyPore : public Diag {
 public:
  DiagSinterFreeEnergyPore(class SPPARKS *, int, char **);
  void init();
  void compute();
  void stats(char *);
  void stats_header(char *);

 private:
  class AppSinter *appsinter;
  double interfacialFE;
};

}

#endif
#endif
