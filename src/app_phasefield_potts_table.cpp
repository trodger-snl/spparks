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

/* ----------------------------------------------------------------------
   Contributing author: Eric Homer, ehomer@sandia.gov
     Mar 31, 2011 - Most recent version
     Most of this was copied from AppPotts and AppPottsNeighOnly
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "app_phasefield_potts_table.h"
#include "solve.h"
#include "random_park.h"
#include "error.h"
#include "memory.h"
#include "domain.h"
#include "region.h"
#include "comm_lattice.h"
#include "timer.h"
#include "table.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPhaseFieldPottsTable::
AppPhaseFieldPottsTable(SPPARKS *spk, int narg, char **arg) :
  AppPottsNeighOnly(spk,narg,arg)
{
  ninteger = 2;
  ndouble = 2;
  allow_kmc = 0;
  allow_rejection = 1;
  allow_masking = 0;
  allow_app_update = 1;
  numrandom = 3;
  delpropensity = 2;//need full neighbor lists of the 1st layer ghost sites

  // add the double array

  recreate_arrays();

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");

  int nphasestemp = atoi(arg[2]);
  if (nphasestemp < 1) error->all(FLERR,"Illegal app_style command");

  if ((9+2*nphasestemp) != narg)
    error->all(FLERR,"Illegal app_style command for phasefield/potts/table");

  // dt_phasefield is set as a multiple of dt_rkmc in setup_end_app();
  // set the multiple here. ( dt_phasefield = dt_rkmc / dt_phasefield_mult )

  dt_phasefield_mult = atof(arg[3]);

  // set the variables for the energetics and evolution

  params.dt = 0.0;
  params.mobility = atof(arg[4]);
  params.ch_energy = atof(arg[5]);
  params.conductivity = atof(arg[6]);
  params.heat_of_transport = atof(arg[7]);
  params.specific_heat = atof(arg[8]);
  params.scale_potts = 1.0;
  params.enforce = 0;

  phases = (Phase **)
    memory->smalloc(nphasestemp*sizeof(Phase *),"phasefield/potts/table:phases");

  int iarg=9;
  int currentspin=0;
  nphases = 0;

  for (int i=0; i<nphasestemp; i++) {
    int phaseID = find_phase(arg[iarg]);
    if (phaseID >= 0 ) error->all(FLERR,"Phase already exists");

    //create the new phase

    phases[nphases] = new Phase;
    int n = strlen(arg[iarg]) + 1;
    phases[nphases]->id = new char[n];
    strcpy(phases[nphases]->id,arg[iarg]);
    int spin_range = atoi(arg[iarg+1]);
    phases[nphases]->spin_start = currentspin+1;
    currentspin += spin_range;
    phases[nphases]->spin_end = currentspin;
    phases[nphases]->table = new Table(spk,phases[nphases]->id);

    iarg += 2;
    nphases++;
  }

  if (nphases != nphasestemp) error->all(FLERR,"Illegal app_style command");
  if (currentspin != nspins) error->all(FLERR,"Illegal app_style command -- "
                                        "phase spins don't add to nspins");

  //set other default values

  mu=cnew=Tnew=NULL;
  nmax_app=0;
  warn_concentration_deviation_all=0;
  print_cmap=false;
  init_site_phase=false;

  scale_free_energy = 1.0;
  scale_comp_energy_penalty = 1.0;
}

/* ---------------------------------------------------------------------- */

AppPhaseFieldPottsTable::~AppPhaseFieldPottsTable()
{
  memory->destroy(mu);
  memory->destroy(cnew);
  memory->destroy(Tnew);

  for (int i = 0; i < nphases; i++) {
    delete [] phases[i]->id;
    delete phases[i]->table;
    delete phases[i];
  }
  memory->sfree(phases);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::input_app(char *command, int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Invalid command for app_style");

  if (strcmp(command,"pottspf/command") == 0) {

    if (strcmp(arg[0],"reset_phasefield") == 0)
      set_pf_resetlist(narg,arg);

    else if (strcmp(arg[0],"print_connectivity") == 0) {
      if (strcmp(arg[1],"yes") == 0) print_cmap=true;
      else print_cmap=false;
    }

    else if (strcmp(arg[0],"set_site_phase") == 0) {
      if (strcmp(arg[1],"yes") == 0) init_site_phase=true;
      else init_site_phase=false;
    }

    else if (strcmp(arg[0],"enforce_concentration_limits") == 0) {
      params.enforce = strcmp(arg[1],"yes") == 0;

      // forcing the limits means there is nothing to warn about;
      // turning it off checks for deviations again

      warn_concentration_deviation_all = params.enforce;

      if (domain->me == 0) {
        const char *msg = params.enforce ?
          "Concentration limits will be enforced\n" :
          "Concentration limits will not be enforced\n";
        if (screen) fprintf(screen,"%s",msg);
        if (logfile) fprintf(logfile,"%s",msg);
      }
    }

    else if (strcmp(arg[0],"nucleation") == 0)
      error->all(FLERR,"Nucleation command not currently available");

    else if (strcmp(arg[0],"scale_energy") == 0) {
      int i=1;
      while (i < narg) {
        if (i+1 >= narg) error->all(FLERR,"Illegal scale_energy command");
        double *target;
        const char *what;
        if (strcmp(arg[i],"potts")==0) {
          target = &params.scale_potts;
          what = "Potts interface energy";
        } else if (strcmp(arg[i],"free")==0) {
          target = &scale_free_energy;
          what = "Free energy";
        } else if (strcmp(arg[i],"penalty")==0) {
          target = &scale_comp_energy_penalty;
          what = "Composition energy penalty";
        } else
          error->all(FLERR,"Illegal scale_energy command");
        *target = atof(arg[i+1]);
        if (domain->me == 0) {
          if (screen) fprintf(screen,"%s will be scaled by %f\n",what,*target);
          if (logfile) fprintf(logfile,"%s will be scaled by %f\n",what,*target);
        }
        i+=2;
      }
    }
    else error->all(FLERR,"Invalid command for app_style");
  }
  else error->all(FLERR,"Invalid command for app_style");
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::init_app()
{
  int i;

  // check to make sure that simulation is fully periodic

  for (i=0; i<domain->dimension; i++)
    if (domain->periodicity[i] != 1)
      error->all(FLERR,"app_style does not support non-periodic boundaries");

  // init the parent class first

  AppPottsNeighOnly::init_app();
  if (maxneigh > 32)
    error->all(FLERR,"phasefield/potts/table supports at most 32 neighbors");

  // axis neighbours for the finite differences

  stencil.setup(spk,this,print_cmap);

  // process phases

  process_phases();

  if (init_site_phase)
    for (i=0; i<nlocal+nghost; i++)
      set_site_phase(i);

  // scratch arrays: mu on all sites, the updated fields on owned sites

  if (nmax_app < nlocal+nghost) {
    nmax_app = nlocal+nghost;
    memory->destroy(mu);
    memory->destroy(cnew);
    memory->destroy(Tnew);
    memory->create(mu,nmax_app,"app_potts_pf:mu");
    memory->create(cnew,nmax_app,"app_potts_pf:cnew");
    memory->create(Tnew,nmax_app,"app_potts_pf:Tnew");
  }
}

/* ----------------------------------------------------------------------
   check the tables and pack every phase's G and dG/dc, scaled by
   scale_free_energy, into tab
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::process_phases()
{
  int i,j,k;

  ptab.resize(nphases);
  tab.clear();
  params.penalty = scale_comp_energy_penalty * scale_free_energy;

  for (i=0; i<nphases; i++) {
    Phase *p = phases[i];
    Table *t = p->table;
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
    pt.spin_lo = p->spin_start;
    pt.spin_hi = p->spin_end;
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
      for (k=0; k<nCol; k++) g[k] = G[j][k] * scale_free_energy;
      dg[0] = (-g[2] + 4*g[1] - 3*g[0])/(2*h);
      for (k=1; k<n; k++) dg[k] = (g[k+1] - g[k-1])/(2*h);
      dg[n] = (3*g[n] - 4*g[n-1] + g[n-2])/(2*h);
    }
  }
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::setup_end_app()
{
  // set the time step for phase field

  dt_phasefield = dt_rkmc / dt_phasefield_mult;
  params.dt = dt_phasefield;
}

/* ----------------------------------------------------------------------
 set site value ptrs each time iarray/darray are reallocated
 ------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::grow_app()
{
  spin = iarray[0];
  phase = iarray[1];
  c = darray[0];
  T = darray[1];
}

/* ----------------------------------------------------------------------
   full site energy: bulk + Potts + gradient energy kappa/2 |grad c|^2
------------------------------------------------------------------------- */

double AppPhaseFieldPottsTable::site_energy(int i)
{
  FDHostNbr nbr{neighbor};
  return pftable::site_energy(numneigh,nbr,spin,c,T,i,spin[i],phase[i],
                              ptab.data(),tab.data(),params) +
         0.5 * params.ch_energy * stencil.grad_dot(nbr,c,c,i);
}

/* ----------------------------------------------------------------------
   rKMC method
   perform a site event with no null bin rejection
   flip to random neighbor spin without null bin
   technically this is an incorrect rejection-KMC algorithm
   the gradient energy does not change with a flip, so it is left out
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::site_event_rejection(int i, RandomPark *random)
{
  FDHostNbr nbr{neighbor};
  if (pftable::rejection_event(i,*random,numneigh,nbr,spin,phase,c,T,
                               ptab.data(),nphases,tab.data(),params,
                               temperature,t_inverse))
    naccept++;
}

/* ----------------------------------------------------------------------
   iterate through the phase field solution
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::app_update(double stoptime)
{
  double localtime=0.0;
  FDHostNbr nbr{neighbor};
  const PFPhaseTable *pt = ptab.data();
  const double *tb = tab.data();
  int flags = 0;

  // communicate all sites to make sure it's up-to-date when it starts

  timer->stamp();
  comm->all();
  timer->stamp(TIME_COMM);

  int done = 0;
  while (!done) {

    // mu on owned sites and on ghosts with full neighbor lists (the first
    // ghost layer), so lap(mu) needs no extra exchange

    for (int i=0; i<nlocal+nghost; i++)
      if (i < nlocal || numneigh[i] == maxneigh)
        mu[i] = pftable::chem_potential(stencil,nbr,phase,c,T,i,pt,tb,params);

    for (int i=0; i<nlocal; i++)
      flags |= pftable::step(stencil,nbr,c,T,mu,i,params,cnew[i],Tnew[i]);

    // copy updated phase field into old concentration field

    for (int i=0; i<nlocal; i++) {
      c[i]=cnew[i];
      T[i]=Tnew[i];
    }

    // reset the certain values if appropriate

    for (size_t m=0; m < pf_resetlist.size(); m++)
      c[pf_resetlist[m]]=pf_resetlistvals[m];
    timer->stamp(TIME_SOLVE);

    // re-sync all the data

    comm->all();
    timer->stamp(TIME_COMM);

    // increment time and determine when to finish

    localtime += dt_phasefield;
    if (localtime >= (stoptime- 1e-6)) done = 1;

    // throw exception when PF has iterated too far

    if (localtime > (stoptime + 1e-6)) {
      char errorstr[128];
      snprintf(errorstr,sizeof(errorstr),
               "PF step time (%f) in excess of Potts step time (%f)",
               localtime,stoptime);
      error->all(FLERR,errorstr);
    }
  }

  check_step_flags(flags);
}

/* ----------------------------------------------------------------------
   flags from pftable::step over this app_update, all ranks
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::check_step_flags(int flags)
{
  int all;
  MPI_Allreduce(&flags,&all,1,MPI_INT,MPI_BOR,world);

  // 1/T in the next step would turn the field into NaN

  if (all & 2)
    error->all(FLERR,"Temperature reached zero: reduce the phase-field "
               "time step (raise the PF/Potts steps argument)");

  if ((all & 1) && !warn_concentration_deviation_all) {
    warn_concentration_deviation_all = 1;
    if (me == 0) {
      if (screen) fprintf(screen,
          "Warning: Concentration has deviated outside [0,1] range\n");
      if (logfile) fprintf(logfile,
          "Warning: Concentration has deviated outside [0,1] range\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

int AppPhaseFieldPottsTable::find_phase(char *name)
{
  for (int iphase = 0; iphase < nphases; iphase++)
    if (strcmp(name,phases[iphase]->id) == 0) return iphase;
  return -1;
}
/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::set_site_phase(int i)
{
  for (int j=0; j<nphases; j++)
    if (spin[i] >= phases[j]->spin_start &&
        spin[i] <= phases[j]->spin_end)
      phase[i]=j;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::set_pf_resetlist(int narg, char **arg)
{
  if (narg < 3)
    error->all(FLERR,"Illegal pottspf/command reset_phasefield");

  // value each owned site is reset to, seeded from any earlier
  // reset_phasefield, so a site in two regions is caught in O(n)
  // (a flag array, not a NaN sentinel: fast-math folds x == x to true)

  std::vector<double> val(nlocal,0.0);
  std::vector<char> isset(nlocal,0);
  for (size_t m=0; m<pf_resetlist.size(); m++) {
    val[pf_resetlist[m]] = pf_resetlistvals[m];
    isset[pf_resetlist[m]] = 1;
  }

  // process the arguments one by one

  int flag = 0;
  int iarg = 1;
  while (iarg < narg) {
    if (iarg+2 > narg) error->all(FLERR,"Illegal pottspf/command "
                                  "reset_phasefield");

    int iregion = domain->find_region(arg[iarg]);
    if (iregion < 0) error->all(FLERR,"reset_phasefield command region ID "
                                "does not exist");
    double v = atof(arg[iarg+1]);

    for (int i=0;i<nlocal;i++)
      if (domain->regions[iregion]->match(xyz[i][0],xyz[i][1],xyz[i][2])) {
        if (isset[i] && val[i] != v) flag = 1;
        val[i] = v;
        isset[i] = 1;
      }
    iarg += 2;
  }

  // make sure no site is reset more than once to a different value

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
  if (flagall)
    error->all(FLERR,"Invalid reset_phasefield command: attempting to "
               "reset a site more than once to different values");

  pf_resetlist.clear();
  pf_resetlistvals.clear();
  for (int i=0;i<nlocal;i++)
    if (isset[i]) {
      pf_resetlist.push_back(i);
      pf_resetlistvals.push_back(val[i]);
    }

  // print statistics

  tagint nbig = pf_resetlist.size();
  tagint allcount;
  MPI_Allreduce(&nbig,&allcount,1,MPI_SPK_TAGINT,MPI_SUM,world);

  if (domain->me == 0) {
    if (screen) fprintf(screen,"Setting reset phasefield list ...\n"
                        "  " TAGINT_FORMAT " sites identified to reset\n",
                        allcount);
    if (logfile) fprintf(logfile,"Setting reset phasefield list ...\n"
                         "  " TAGINT_FORMAT " sites identified to reset\n",
                         allcount);
  }
}

/* ----------------------------------------------------------------------
   get app properties or in this case the pointer to the table
------------------------------------------------------------------------- */

void *AppPhaseFieldPottsTable::extract_app(char * name) {

  int i=find_phase(name);
  if (i < 0)
    error->all(FLERR,"Cannot find the phase");
  else
    return (void *) phases[i]->table;

  return (void *) NULL;
}
