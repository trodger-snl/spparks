/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifdef SPPARKS_HIGHFIVE

#include "dump_conduction_kokkos.h"
#include "app_additive_texture_kokkos.h"
#include "error.h"
#include <cstring>

using namespace SPPARKS_NS;

DumpConductionKokkos::DumpConductionKokkos(SPPARKS *spk, int narg, char **arg) :
  Dump(spk, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal dump conduction command: dump ID conduction DELTA FILE");
  if (multifile || multiproc)
    error->all(FLERR,"Dump conduction cannot use '*' or '%' wildcards");
}

void DumpConductionKokkos::init_style()
{
  if (!dynamic_cast<AppAdditiveTextureKokkos *>(app))
    error->all(FLERR,"Dump conduction needs app_style additive/texture/kk");
}

int DumpConductionKokkos::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"precision") != 0) return 0;
  if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
  if (strcmp(arg[1],"float") == 0) single = true;
  else if (strcmp(arg[1],"double") == 0) single = false;
  else error->all(FLERR,"Dump_modify precision must be float or double");
  return 2;
}

void DumpConductionKokkos::write(double time)
{
  ConductionSolverKokkos *c = static_cast<AppAdditiveTextureKokkos *>(app)->conduction();
  // no grid yet (a dump at the run's first step, before any advance): nothing to write
  if (!c || !c->ready()) return;
  c->write_grid(filename, time, single);
  firstflag = 0;
  idump++;
}

#endif
