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
#include "math.h"
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
  AppPhaseField(spk,narg,arg,1) {}

AppPhaseField::AppPhaseField(SPPARKS *spk, int narg, char **arg, int parse) :
  AppLattice(spk,narg,arg)
{
  phase = NULL;
  for (int k = 0; k < PF_MAXA; k++) aux[k] = NULL;
  for (int k = 0; k < PF_MAXF; k++) fnew[k] = NULL;
  nmax_app = 0;
  warned = 0;
  print_stencil = 0;
  scale_free = scale_penalty = 1.0;
  dt = 0.0;
  if (!parse) return;

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
    check_ch_heat();
    for (int i = 5; i < nargs; i++) add_phase(a[i]);
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
  exact_steps = 1;
  dt_sweep = dt;
  create_arrays();
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

void AppPhaseField::check_ch_heat()
{
  const PFParams &q = chh.q;
  if (q.mobility < 0.0 || q.ch_energy < 0.0 || q.conductivity < 0.0 ||
      q.specific_heat <= 0.0)
    error->all(FLERR,"Illegal app_style command -- ch_heat needs M, kappa, "
               "k >= 0 and Cp > 0");
}

/* ----------------------------------------------------------------------
   a ch_heat phase: its name and the table read_table fills
------------------------------------------------------------------------- */

void AppPhaseField::add_phase(const char *name)
{
  for (char *p : phase_names)
    if (strcmp(p,name) == 0) error->all(FLERR,"Phase already exists");
  char *p = new char[strlen(name) + 1];
  strcpy(p,name);
  phase_names.push_back(p);
  tables.push_back(new Table(spk,p));
}

/* ---------------------------------------------------------------------- */

void AppPhaseField::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"phasefield/command") != 0 || narg < 2)
    error->all(FLERR,"Unrecognized command");

  if (strcmp(arg[0],"enforce_concentration_limits") == 0 && narg == 2)
    chh.q.enforce = yes_no(arg[1],"phasefield/command");
  else if (strcmp(arg[0],"scale_energy") == 0 && narg % 2 == 1) {
    for (int i = 1; i < narg; i += 2) {
      if (strcmp(arg[i],"free") == 0) scale_free = atof(arg[i+1]);
      else if (strcmp(arg[i],"penalty") == 0) scale_penalty = atof(arg[i+1]);
      else error->all(FLERR,"Illegal phasefield/command scale_energy");
    }
  } else error->all(FLERR,"Illegal phasefield/command");
}

/* ---------------------------------------------------------------------- */

bool AppPhaseField::yes_no(const char *word, const char *command)
{
  if (strcmp(word,"yes") == 0) return true;
  if (strcmp(word,"no") != 0) {
    char str[128];
    snprintf(str,128,"Illegal %s: expected yes or no, got %s",command,word);
    error->all(FLERR,str);
  }
  return false;
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

  stencil.setup(spk,this,print_stencil);

  if (model == CH_HEAT) {
    pftable::pack_tables(error,tables.data(),tables.size(),scale_free,ptab,tab);
    chh.q.penalty = scale_penalty * scale_free;
    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (phase[i] < 0 || phase[i] >= (int) tables.size()) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
    if (flagall)
      error->all(FLERR,"phasefield ch_heat: a site's phase is not 0..nphases-1");
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
}

/* ----------------------------------------------------------------------
   dt is known here for phasefield/potts(/table) (a fraction of dt_rkmc)
------------------------------------------------------------------------- */

void AppPhaseField::setup_end_app()
{
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
  ac.dt = ch.dt = gg.dt = chh.q.dt = tp.dt = step;
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
  check_flags(step_fields());
}

/* ----------------------------------------------------------------------
   exchange ghosts, then one step of the model on the owned sites;
   returns this rank's flag bits
------------------------------------------------------------------------- */

int AppPhaseField::step_fields()
{
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
  return flags;
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

/* ----------------------------------------------------------------------
   check each table (T rows > 0, c columns 0 to 1, both evenly spaced) and
   pack every table's G and dG/dc, scaled by scale_free, into tab; spin
   ranges are left to the caller
------------------------------------------------------------------------- */

void pftable::pack_tables(Error *error, Table **tables, int n_tables,
                          double scale_free, std::vector<PFPhaseTable> &ptab,
                          std::vector<double> &tab)
{
  int i,j,k;

  ptab.assign(n_tables,PFPhaseTable());
  tab.clear();

  for (i=0; i<n_tables; i++) {
    Table *t = tables[i];
    if (!t->getTableReady())
      error->all(FLERR,"Not all tables loaded for phases");
    if (!t->hasRowColVals())
      error->all(FLERR,"Phase tables need rowcolvals (T rows, c columns)");

    int nRow,nCol;
    t->getTableDims(&nRow,&nCol);
    double **G = t->getTablePtr();
    double *rowvals = t->getRowPtr();
    double *colvals = t->getColPtr();
    if (nRow < 2 || nCol < 3)
      error->all(FLERR,"Illegal table for phase, need at least 2 rows "
                 "and 3 columns");

    double dC = colvals[1]-colvals[0];
    double dT = rowvals[1]-rowvals[0];

    // colvals must be the concentration and increase from 0 to 1

    if (colvals[0] !=0 || colvals[nCol-1] != 1)
      error->all(FLERR,"Illegal table for phase, concentration must "
                 "increase from 0 to 1");
    for (j=1; j<nCol; j++)
      if (colvals[j] < 0 || colvals[j] > 1 ||
          colvals[j] <= colvals[j-1] ||
          fabs(colvals[j]-colvals[j-1] - dC) > 1e-6)
        error->all(FLERR,"Illegal table for phase, concentration must "
                   "increase from 0 to 1 and have even intervals");

    // rowvals must be the temperature and increase
    // error if anything less than or equal zero

    if (rowvals[0] <= 0)
      error->all(FLERR,"Illegal table for phase, "
                 "temperature must be greater than zero");
    for (j=1; j<nRow; j++) {
      if (rowvals[j] <= 0)
        error->all(FLERR,"Illegal table for phase, "
                   "temperature must be greater than zero");
      if (rowvals[j] <= rowvals[j-1] ||
          fabs(rowvals[j]-rowvals[j-1] - dT) > 1e-6)
        error->all(FLERR,"Illegal table for phase, temperature must increase "
                   " and have even intervals");
    }

    PFPhaseTable &pt = ptab[i];
    pt.nrow = nRow;
    pt.ncol = nCol;
    pt.t0 = rowvals[0];
    pt.inv_dt = 1.0/dT;
    pt.inv_dc = 1.0/dC;

    // G scaled, then its dGdC: one-sided second-order differences at the
    // ends, central in between

    pt.g = tab.size();
    pt.dg = pt.g + nRow*nCol;
    tab.resize(pt.dg + nRow*nCol);
    double h=dC;
    int n=nCol - 1;    //index of the last value
    for (j=0; j<nRow; j++) {
      double *g = &tab[pt.g + j*nCol];
      double *dg = &tab[pt.dg + j*nCol];
      for (k=0; k<nCol; k++) g[k] = G[j][k] * scale_free;
      dg[0] = (-g[2] + 4*g[1] - 3*g[0])/(2*h);
      for (k=1; k<n; k++) dg[k] = (g[k+1] - g[k-1])/(2*h);
      dg[n] = (3*g[n] - 4*g[n-1] + g[n-2])/(2*h);
    }
  }
}
