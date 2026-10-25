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
#include "random_park.h"
#include "error.h"
#include "domain.h"
#include "region.h"
#include "comm_lattice.h"
#include "timer.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPhaseFieldPottsTable::
AppPhaseFieldPottsTable(SPPARKS *spk, int narg, char **arg) :
  AppPhaseField(spk,narg,arg,0)
{
  if (narg < 9) error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");

  int nphasestemp = atoi(arg[2]);
  if (nphasestemp < 1) error->all(FLERR,"Illegal app_style command");

  if ((9+2*nphasestemp) != narg)
    error->all(FLERR,"Illegal app_style command for phasefield/potts/table");

  // dt (the phase-field step) is set as a fraction of dt_rkmc in
  // setup_end_app(): dt = dt_rkmc / pf_nsteps

  pf_nsteps = atoi(arg[3]);
  if (pf_nsteps < 1 || atof(arg[3]) != pf_nsteps)
    error->all(FLERR,"Illegal app_style command -- PF steps must be an "
               "integer >= 1");

  // the ch_heat model of AppPhaseField, plus the Potts interface energy

  model = CH_HEAT;
  chh.q = {0.0,atof(arg[4]),atof(arg[5]),atof(arg[6]),atof(arg[7]),atof(arg[8]),
           0.0,1.0,0};
  check_ch_heat();
  nfield = 2;
  naux = 1;

  int currentspin=0;
  for (int iarg=9; iarg<narg; iarg+=2) {
    add_phase(arg[iarg]);
    int nspin = atoi(arg[iarg+1]);
    if (nspin < 1) error->all(FLERR,"Illegal app_style command -- "
                              "a phase needs at least one spin");
    spin_lo.push_back(currentspin+1);
    currentspin += nspin;
    spin_hi.push_back(currentspin);
  }
  if (currentspin != nspins) error->all(FLERR,"Illegal app_style command -- "
                                        "phase spins don't add to nspins");

  ninteger = 2;
  ndouble = 2;
  delpropensity = 2;//need full neighbor lists of the 1st layer ghost sites
  delevent = 0;
  allow_kmc = 0;
  allow_rejection = 1;
  allow_masking = 0;
  allow_app_update = 1;
  numrandom = 3;
  create_arrays();

  init_site_phase=false;
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::input_app(char *command, int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Invalid command for app_style");

  if (strcmp(command,"pottspf/command") == 0) {

    if (strcmp(arg[0],"reset_phasefield") == 0)
      set_pf_resetlist(narg,arg);

    else if (strcmp(arg[0],"print_connectivity") == 0)
      print_stencil = strcmp(arg[1],"yes") == 0;

    else if (strcmp(arg[0],"set_site_phase") == 0) {
      if (strcmp(arg[1],"yes") == 0) init_site_phase=true;
      else init_site_phase=false;
    }

    else if (strcmp(arg[0],"enforce_concentration_limits") == 0) {
      chh.q.enforce = strcmp(arg[1],"yes") == 0;

      // forcing the limits means there is nothing to warn about;
      // turning it off checks for deviations again

      warned = chh.q.enforce;

      if (domain->me == 0) {
        const char *msg = chh.q.enforce ?
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
          target = &chh.q.scale_potts;
          what = "Potts interface energy";
        } else if (strcmp(arg[i],"free")==0) {
          target = &scale_free;
          what = "Free energy";
        } else if (strcmp(arg[i],"penalty")==0) {
          target = &scale_penalty;
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
  if (maxneigh > 32)
    error->all(FLERR,"phasefield/potts/table supports at most 32 neighbors");

  // as AppPottsNeighOnly: valid spins, a sweep visits every neighbor once

  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (spin[i] < 1 || spin[i] > nspins) flag = 1;
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
  dt_sweep = 1.0/maxneigh;

  if (init_site_phase)
    for (int i=0; i<nlocal+nghost; i++)
      set_site_phase(i);

  // periodicity, stencil, tables, scratch; then each phase's spin range

  AppPhaseField::init_app();
  for (size_t p=0; p<ptab.size(); p++) {
    ptab[p].spin_lo = spin_lo[p];
    ptab[p].spin_hi = spin_hi[p];
  }
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::setup_end_app()
{
  dt = dt_rkmc / pf_nsteps;
  set_dt(dt);
  AppPhaseField::setup_end_app();
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
                              ptab.data(),tab.data(),chh.q) +
         0.5 * chh.q.ch_energy * stencil.bond_sq(nbr,c,i);
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
                               ptab.data(),ptab.size(),tab.data(),chh.q,
                               temperature,t_inverse))
    naccept++;
}

/* ----------------------------------------------------------------------
   the phase-field steps of one Potts step, each followed by the reset list
------------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::app_update(double /*dt_rkmc*/)
{
  int flags = 0;
  for (int n=0; n<pf_nsteps; n++) {
    flags |= step_fields();

    // reset the certain values if appropriate

    for (size_t m=0; m < pf_resetlist.size(); m++)
      c[pf_resetlist[m]]=pf_resetlistvals[m];
    timer->stamp(TIME_SOLVE);
  }

  // ghosts current for the sweeps and output that follow

  timer->stamp();
  comm->all();
  timer->stamp(TIME_COMM);

  check_flags(flags);
}

/* ---------------------------------------------------------------------- */

void AppPhaseFieldPottsTable::set_site_phase(int i)
{
  for (size_t j=0; j<spin_lo.size(); j++)
    if (spin[i] >= spin_lo[j] && spin[i] <= spin_hi[j])
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
  pf_resetversion++;

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
