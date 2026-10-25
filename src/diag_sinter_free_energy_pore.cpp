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
#include "diag_sinter_free_energy_pore.h"
#include "app_sinter.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

DiagSinterFreeEnergyPore::DiagSinterFreeEnergyPore(SPPARKS *spk, int narg, char **arg) :
  Diag(spk,narg,arg) {}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergyPore::init()
{
  appsinter = dynamic_cast<AppSinter *>(app);
  if (!appsinter) error->all(FLERR,"Diag style incompatible with app style");
  interfacialFE = 0.0;
}

/* ---------------------------------------------------------------------- */

// pore surface per site of the central box (AppSinter::in_density_box)

void DiagSinterFreeEnergyPore::compute()
{
  appsinter->update_ghosts();
  int *spin = appsinter->spin;
  int *numneigh = appsinter->numneigh;
  int **neighbor = appsinter->neighbor;
  double count[2] = {0.0,0.0};   // surface, sites
  for (int i = 0; i < appsinter->nlocal; i++) {
    if (!appsinter->in_density_box(i)) continue;
    count[1]++;
    if (spin[i] > sinter::VACANT)
      for (int k = 0; k < numneigh[i]; k++)
        if (spin[neighbor[i][k]] == sinter::VACANT) count[0]++;
  }
  MPI_Allreduce(MPI_IN_PLACE,count,2,MPI_DOUBLE,MPI_SUM,world);
  interfacialFE = count[0] / count[1];
}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergyPore::stats(char *strtmp)
{
  snprintf(strtmp,256," %10.6lf",interfacialFE);
}

/* ---------------------------------------------------------------------- */

void DiagSinterFreeEnergyPore::stats_header(char *strtmp)
{
  snprintf(strtmp,256," %10s","FE_psite");
}
