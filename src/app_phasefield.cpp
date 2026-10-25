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
#include "stdlib.h"
#include "string.h"
#include "app_phasefield.h"
#include "comm_lattice.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "table.h"
#include "timer.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPhaseField::AppPhaseField(SPPARKS *spk, int narg, char **arg) :
  AppLattice(spk,narg,arg)
{
  if (narg < 3) error->all(FLERR,"Illegal app_style phasefield command");
  dt = atof(arg[1]);
  if (dt <= 0.0) error->all(FLERR,"Illegal app_style phasefield time step");

  const char *name = arg[2];
  int nargs = narg - 3;
  char **a = &arg[3];
  if (strcmp(name,"allen_cahn") == 0 && nargs == 3) {
    model = ALLEN_CAHN;
    ac = {atof(a[0]),atof(a[1]),atof(a[2]),dt};
  } else if (strcmp(name,"cahn_hilliard") == 0 && nargs == 3) {
    model = CAHN_HILLIARD;
    ch = {atof(a[0]),atof(a[1]),atof(a[2]),dt};
  } else if (strcmp(name,"grain_growth") == 0 && nargs == 6) {
    model = GRAIN_GROWTH;
    gg = {atoi(a[0]),atof(a[1]),atof(a[2]),atof(a[3]),atof(a[4]),atof(a[5]),dt};
    if (gg.n < 1 || gg.n > PF_MAXF)
      error->all(FLERR,"phasefield grain_growth needs 1 to 32 order parameters");
  } else if (strcmp(name,"ch_heat") == 0 && nargs >= 6) {
    model = CH_HEAT;
    chh.q = {dt,atof(a[0]),atof(a[1]),atof(a[2]),atof(a[3]),atof(a[4]),
             0.0,0.0,0};
    for (int i = 5; i < nargs; i++) {
      for (char *p : phase_names)
        if (strcmp(p,a[i]) == 0) error->all(FLERR,"Phase already exists");
      phase_names.push_back(dupstr(a[i]));
      tables.push_back(new Table(spk,a[i]));
    }
  } else error->all(FLERR,"Illegal app_style phasefield model or arguments");

  with_model([&](auto &m) { nfield = m.nfield(); naux = m.naux; });

  ninteger = 1;
  ndouble = nfield;
  delpropensity = naux ? 2 : 1;   // aux on the first ghost layer needs a second
  delevent = 0;
  allow_kmc = 0;
  allow_rejection = 0;
  allow_masking = 0;
  allow_app_update = 1;
  app_update_only = 1;
  dt_sweep = dt;
  create_arrays();

  phase = NULL;
  for (int k = 0; k < PF_MAXA; k++) aux[k] = NULL;
  for (int k = 0; k < PF_MAXF; k++) fnew[k] = NULL;
  nmax_app = 0;
  warned = 0;
  scale_free = scale_penalty = 1.0;
}

/* ---------------------------------------------------------------------- */

AppPhaseField::~AppPhaseField()
{
  for (int k = 0; k < PF_MAXA; k++) memory->destroy(aux[k]);
  for (int k = 0; k < PF_MAXF; k++) memory->destroy(fnew[k]);
  for (Table *t : tables) delete t;
  for (char *p : phase_names) delete [] p;
}

/* ---------------------------------------------------------------------- */

char *AppPhaseField::dupstr(const char *s)
{
  char *p = new char[strlen(s) + 1];
  strcpy(p,s);
  return p;
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"phasefield/command") != 0 || narg < 2)
    error->all(FLERR,"Unrecognized command");

  if (strcmp(arg[0],"enforce_concentration_limits") == 0 && narg == 2)
    chh.q.enforce = strcmp(arg[1],"yes") == 0;
  else if (strcmp(arg[0],"scale_energy") == 0 && narg % 2 == 1) {
    for (int i = 1; i < narg; i += 2) {
      if (strcmp(arg[i],"free") == 0) scale_free = atof(arg[i+1]);
      else if (strcmp(arg[i],"penalty") == 0) scale_penalty = atof(arg[i+1]);
      else error->all(FLERR,"Illegal phasefield/command scale_energy");
    }
  } else error->all(FLERR,"Illegal phasefield/command");
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::grow_app()
{
  phase = iarray[0];
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::init_app()
{
  for (int d = 0; d < domain->dimension; d++)
    if (domain->periodicity[d] != 1)
      error->all(FLERR,"app_style phasefield needs periodic boundaries");

  stencil.setup(spk,this,0);

  if (model == CH_HEAT) {
    pftable::pack_tables(error,tables.data(),tables.size(),scale_free,ptab,tab);
    chh.q.penalty = scale_penalty * scale_free;
    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (phase[i] < 0 || phase[i] >= (int) tables.size()) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
    if (flagall)
      error->all(FLERR,"phasefield ch_heat: i1 must be a phase index 0..nphases-1");
  }

  if (nmax_app < nlocal + nghost) {
    nmax_app = nlocal + nghost;
    for (int k = 0; k < naux; k++) {
      memory->destroy(aux[k]);
      memory->create(aux[k],nmax_app,"phasefield:aux");
    }
    for (int k = 0; k < nfield; k++) {
      memory->destroy(fnew[k]);
      memory->create(fnew[k],nmax_app,"phasefield:fnew");
    }
  }

  double dtmax;
  with_model([&](auto &m) { dtmax = m.dt_max(stencil); });
  if (dt > dtmax && domain->me == 0) {
    char str[128];
    snprintf(str,sizeof(str),"phasefield time step %g exceeds the explicit "
             "stability estimate %g",dt,dtmax);
    error->warning(FLERR,str);
  }
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::set_dt(double step)
{
  ac.dt = ch.dt = gg.dt = chh.q.dt = step;
}

PFCtx<FDHostNbr, double *, int *> AppPhaseField::host_ctx()
{
  PFCtx<FDHostNbr, double *, int *> x;
  x.st = stencil;
  x.nbr = FDHostNbr{neighbor};
  for (int k = 0; k < PF_MAXF; k++) x.f[k] = k < nfield ? darray[k] : NULL;
  for (int k = 0; k < PF_MAXA; k++) x.a[k] = aux[k];
  x.phase = phase;
  x.pt = ptab.data();
  x.tab = tab.data();
  return x;
}

/* ----------------------------------------------------------------------
   one explicit step of length step (the update-only loop's dt)
------------------------------------------------------------------------- */

void AppPhaseField::app_update(double step)
{
  set_dt(step);

  timer->stamp();
  comm->all();
  timer->stamp(TIME_COMM);

  auto x = host_ctx();
  const int nall = nlocal + nghost;
  int flags = 0;
  with_model([&](auto &m) {
    // aux on owned sites and on ghosts with full neighbour lists
    if (m.naux)
      for (int i = 0; i < nall; i++)
        if (i < nlocal || numneigh[i] == maxneigh) {
          double a[PF_MAXA];
          m.aux(x,i,a);
          for (int k = 0; k < m.naux; k++) aux[k][i] = a[k];
        }
    for (int i = 0; i < nlocal; i++) {
      double out[PF_MAXF];
      flags |= m.update(x,i,out);
      for (int k = 0; k < nfield; k++) fnew[k][i] = out[k];
    }
  });
  for (int k = 0; k < nfield; k++)
    memcpy(darray[k],fnew[k],nlocal*sizeof(double));

  check_flags(flags);
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::check_flags(int flags)
{
  int all;
  MPI_Allreduce(&flags,&all,1,MPI_INT,MPI_BOR,world);
  if (all & 2)
    error->all(FLERR,"Temperature reached zero: reduce the phase-field "
               "time step");
  if ((all & 1) && !warned) {
    warned = 1;
    if (domain->me == 0)
      error->warning(FLERR,"Concentration has deviated outside [0,1] range");
  }
}

/* ---------------------------------------------------------------------- */

double AppPhaseField::site_energy(int i)
{
  auto x = host_ctx();
  double e = 0.0;
  with_model([&](auto &m) { e = m.energy(x,i); });
  return e;
}

/* ---------------------------------------------------------------------- */

void *AppPhaseField::extract_app(char *name)
{
  for (size_t i = 0; i < phase_names.size(); i++)
    if (strcmp(name,phase_names[i]) == 0) return (void *) tables[i];
  error->all(FLERR,"Cannot find the phase");
  return NULL;
}
