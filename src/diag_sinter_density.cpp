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

#include "mpi.h"
#include "stdio.h"
#include "diag_sinter_density.h"
#include "app_sinter.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

DiagSinterDensity::DiagSinterDensity(SPPARKS *spk, int narg, char **arg) :
  Diag(spk,narg,arg) {}

/* ---------------------------------------------------------------------- */

void DiagSinterDensity::init()
{
  appsinter = dynamic_cast<AppSinter *>(app);
  if (!appsinter) error->all(FLERR,"Diag style incompatible with app style");
  density = 0.0;
}

/* ---------------------------------------------------------------------- */

// fraction of grain sites in the central box (AppSinter::in_density_box)

void DiagSinterDensity::compute()
{
  density = appsinter->calculate_density();
}

/* ---------------------------------------------------------------------- */

void DiagSinterDensity::stats(char *strtmp)
{
  snprintf(strtmp,256," %10.6lf",density);
}

/* ---------------------------------------------------------------------- */

void DiagSinterDensity::stats_header(char *strtmp)
{
  snprintf(strtmp,256," %10s","Density");
}
