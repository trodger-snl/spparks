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
#include "diag_sinter_free_energy.h"
#include "app_sinter.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

DiagSinterFreeEnergy::DiagSinterFreeEnergy(SPPARKS *spk, int narg, char **arg) :
  Diag(spk,narg,arg) {}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergy::init()
{
  appsinter = dynamic_cast<AppSinter *>(app);
  if (!appsinter) error->all(FLERR,"Diag style incompatible with app style");
  interfacialFE = 0.0;
}

/* ---------------------------------------------------------------------- */

// pore surface: the # of (grain site, pore neighbour) pairs

void DiagSinterFreeEnergy::compute()
{
  appsinter->update_ghosts();
  int *spin = appsinter->spin;
  int *numneigh = appsinter->numneigh;
  int **neighbor = appsinter->neighbor;
  double surface = 0.0;
  for (int i = 0; i < appsinter->nlocal; i++)
    if (spin[i] > sinter::VACANT)
      for (int k = 0; k < numneigh[i]; k++)
        if (spin[neighbor[i][k]] == sinter::VACANT) surface++;
  MPI_Allreduce(&surface,&interfacialFE,1,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergy::stats(char *strtmp)
{
  snprintf(strtmp,256," %g",interfacialFE);
}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergy::stats_header(char *strtmp)
{
  snprintf(strtmp,256," %10s","InterfFE");
}
