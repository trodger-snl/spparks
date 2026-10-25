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
#include <initializer_list>
#include "fd_stencil.h"
#include "app_lattice.h"
#include "domain.h"
#include "lattice.h"
#include "error.h"
#include "spparks.h"

using namespace SPPARKS_NS;

// same as in create_sites.cpp and diag_cluster.cpp and lattice.cpp

enum{NONE,LINE_2N,SQ_4N,SQ_8N,TRI,SC_6N,SC_26N,FCC,BCC,DIAMOND,
  FCC_OCTA_TETRA,RANDOM_1D,RANDOM_2D,RANDOM_3D};

/* ----------------------------------------------------------------------
   cmap[2d], cmap[2d+1] = neighbor-list slots of the -/+ neighbour along
   axis d, from create_sites' neighbour order for each lattice style
------------------------------------------------------------------------- */

void FDStencil::setup(SPPARKS *spk, AppLattice *app, int print)
{
  Domain *domain = spk->domain;
  Error *error = spk->error;
  Lattice *lattice = domain->lattice;

  if (lattice == NULL || lattice->nbasis > 1)
    error->all(FLERR,"Finite differences need a single-basis lattice");

  dim = domain->dimension;
  h = lattice->latconst;
  inv_h2 = 1.0 / (h * h);

  static const int maps[][6] = {
    {0,1,0,0,0,0},      // LINE_2N
    {0,3,1,2,0,0},      // SQ_4N
    {1,6,3,4,0,0},      // SQ_8N
    {0,5,1,4,2,3},      // SC_6N
    {4,21,10,15,12,13}  // SC_26N
  };
  int style = lattice->style, m;
  if (style == LINE_2N) m = 0;
  else if (style == SQ_4N) m = 1;
  else if (style == SQ_8N) m = 2;
  else if (style == SC_6N) m = 3;
  else if (style == SC_26N) m = 4;
  else error->all(FLERR,"Lattice style not compatible with finite differences");
  for (int k = 0; k < 6; k++) cmap[k] = maps[m][k];

  // check the map at one site away from the box faces (whose periodic
  // neighbours sit a box length away); a rank owning none skips the check

  double lim[3][2] = {{domain->boxxlo,domain->boxxhi},
                      {domain->boxylo,domain->boxyhi},
                      {domain->boxzlo,domain->boxzhi}};
  int bad = 0, checked = 0;
  for (int i = 0; i < app->nlocal && !checked; i++) {
    int inside = 1;
    for (int d = 0; d < dim; d++)
      if (app->xyz[i][d] == lim[d][0] || app->xyz[i][d] == lim[d][1]) inside = 0;
    if (!inside) continue;
    checked = 1;
    for (int k = 0; k < 2 * dim; k++) {
      int s = app->neighbor[i][cmap[k]];
      double want = app->xyz[i][k / 2] + (k % 2 ? h : -h);
      if (app->xyz[s][k / 2] != want) bad = 1;
    }
  }
  int all[2] = {bad, checked}, sum[2];
  MPI_Allreduce(all,sum,2,MPI_INT,MPI_SUM,spk->world);

  if ((sum[0] || print) && domain->me == 0) {
    const char *ax = "xyz";
    for (int k = 0; k < 2 * dim; k++)
      for (FILE *fp : {spk->screen, spk->logfile})
        if (fp) fprintf(fp,"  FD neighbour slot %c%c = %d\n",
                        k % 2 ? '+' : '-',ax[k / 2],cmap[k]);
  }
  if (sum[0]) error->all(FLERR,"Invalid finite-difference connectivity map");
  if (sum[1] == 0)
    error->all(FLERR,"No interior site to check the finite-difference map");
}
