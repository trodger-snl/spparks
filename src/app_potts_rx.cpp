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

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include "app_potts_rx.h"
#include "comm_lattice.h"
#include "random_mars.h"
#include "random_park.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsRX::AppPottsRX(SPPARKS *spk, int narg, char **arg) :
  AppPotts(spk,narg,arg)
{
  ninteger = 2;
  ndouble = 2;
  allow_kmc = 0;
  allow_masking = 0;
  allow_app_update = 1;
  recreate_arrays();

  if (narg < 2) error->all(FLERR,"Illegal app_style command");
  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");
  dt_sweep = 1.0;   // time is in MCS

  nrate = hrate = hrecover = 0.0;
  nuc = rx::Nucleation{nspins, 0.0, 0.0, 0, 0, 0};
  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nucleation") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal app_style command");
      nrate = atof(arg[iarg+1]);
      nuc.hmin = atof(arg[iarg+2]);
      if (nrate < 0.0) error->all(FLERR,"Illegal app_style command");
      iarg += 3;
      if (iarg < narg && strcmp(arg[iarg],"boundary") == 0) {
        nuc.boundary = 1;
        iarg++;
      }
    } else if (strcmp(arg[iarg],"hardening") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal app_style command");
      hrate = atof(arg[iarg+1]);
      hrecover = atof(arg[iarg+2]);
      if (hrate <= 0.0 || hrecover < 0.0)
        error->all(FLERR,"Illegal app_style command");
      nuc.any = 1;
      iarg += 3;
    } else error->all(FLERR,"Illegal app_style command");
  }

  key = ranmaster->initial_seed();
}

/* ---------------------------------------------------------------------- */

void AppPottsRX::grow_app()
{
  spin = iarray[0];
  seedspin = iarray[1];
  h = darray[0];
  w = darray[1];
}

/* ----------------------------------------------------------------------
   check the sites; warn if stored energy makes the boundaries rough
------------------------------------------------------------------------- */

void AppPottsRX::init_app()
{
  if (maxneigh > rx::MAXNEIGH)
    error->all(FLERR,"potts/rx supports at most 32 neighbors");

  int flag = 0;
  double hmax = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (spin[i] < 1 || h[i] < 0.0) flag = 1;
    if (spin[i] <= nspins) hmax = MAX(hmax,h[i]);
  }
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");

  double hmaxall;
  MPI_Allreduce(&hmax,&hmaxall,1,MPI_DOUBLE,MPI_MAX,world);
  if (nrate > 0.0 && nglobal >= (tagint) INT_MAX - nspins)
    error->all(FLERR,"potts/rx nucleus spins (nspins + site id) overflow int");
  nuc.idbase = nspins + nglobal;

  // H >= maxneigh-2 lets a deformed site with any recrystallized neighbour
  // flip downhill whatever its boundary: curvature-blind (Eden) growth.
  // Hardening saturates at hrate/hrecover, or grows without bound.

  if (hrate > 0.0)
    hmaxall = hrecover > 0.0 ? MAX(hmaxall,hrate/hrecover) : HUGE_VAL;
  if (hmaxall >= maxneigh - 2 && me == 0)
    error->warning(FLERR,"potts/rx stored energy reaches maxneigh-2: "
                   "recrystallization fronts will be rough");
}

/* ----------------------------------------------------------------------
   this run's temperature: the Boltzmann table and w = exp(-H/T)
------------------------------------------------------------------------- */

void AppPottsRX::setup_end_app()
{
  par.nspins = nspins;
  par.temp = temperature;
  par.tinv = temperature > 0.0 ? t_inverse : 0.0;
  for (int d = -rx::MAXNEIGH; d <= rx::MAXNEIGH; d++)
    par.boltz[d + rx::MAXNEIGH] =
      temperature > 0.0 ? exp(-(double) d * t_inverse) : 0.0;
  for (int i = 0; i < nlocal + nghost; i++) w[i] = exp(-h[i] * par.tinv);
}

/* ----------------------------------------------------------------------
   # of unlike neighbours + stored energy
------------------------------------------------------------------------- */

double AppPottsRX::site_energy(int i)
{
  int isite = spin[i];
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (isite != spin[neighbor[i][j]]) eng++;
  return (double) eng + h[i];
}

/* ---------------------------------------------------------------------- */

void AppPottsRX::site_event_rejection(int i, RandomPark *random)
{
  if (rx::event(i,*random,numneigh,FDHostNbr{neighbor},spin,h,w,par))
    naccept++;
}

/* ----------------------------------------------------------------------
   after each sweep: hardening, then continuous nucleation. A site with
   H >= hmin (deformed, without hardening; on a boundary, with boundary)
   seeds with probability nrate*dt, and it and its neighbours (deformed
   ones, without hardening) become a new recrystallized grain, H = 0. The
   draws are keyed to (site id, sweep), the seeds decided before any site
   changes, so the result does not depend on the # of ranks.
------------------------------------------------------------------------- */

unsigned long long AppPottsRX::salt() const
{
  return gw_mix(key ^ ((unsigned long long) nsweeps << 32));
}

// H = (H + d)*f integrates dH/dt = hrate - hrecover*H over dt exactly
void AppPottsRX::hardening_step(double dt, double &d, double &f) const
{
  if (hrecover > 0.0) {
    f = exp(-hrecover * dt);
    d = hrate / hrecover * (1.0 - f) / f;
  } else {
    f = 1.0;
    d = hrate * dt;
  }
}

void AppPottsRX::app_update(double dt)
{
  if (hrate > 0.0) {
    double d, f;
    hardening_step(dt,d,f);
    for (int i = 0; i < nlocal; i++) rx::harden(i,h,w,d,f,par.tinv);
  }

  if (nrate == 0.0) return;
  if (nprocs > 1) comm->all();

  // every site decides its own seed, ghosts too, unless that takes its
  // neighbours (boundary): then the owners decide and send them

  const int nseed = nuc.boundary ? nlocal : nlocal + nghost;
  const unsigned long long s = salt();
  rx::Nucleation q = nuc;
  q.prob = nrate * dt;
  for (int c = 0; c < nseed; c++)
    seedspin[c] = rx::seed(c,id[c],numneigh,FDHostNbr{neighbor},spin,h,q,s);
  if (nuc.boundary && nprocs > 1) comm->all();
  for (int i = 0; i < nlocal; i++) {
    int snew = rx::nucleus(i,numneigh,FDHostNbr{neighbor},spin,seedspin,
                           id,q);
    if (snew) {
      spin[i] = snew;
      h[i] = 0.0;
      w[i] = 1.0;
    }
  }
}

/* ----------------------------------------------------------------------
   recrystallized fraction
------------------------------------------------------------------------- */

void AppPottsRX::stats(char *strtmp)
{
  char big[256];
  AppLattice::stats(big);
  bigint n = 0, nall;
  for (int i = 0; i < nlocal; i++)
    if (spin[i] > nspins) n++;
  MPI_Allreduce(&n,&nall,1,MPI_SPK_BIGINT,MPI_SUM,world);
  snprintf(strtmp,256,"%s %10.6f",big,(double) nall/nglobal);
}

void AppPottsRX::stats_header(char *strtmp)
{
  char big[256];
  AppLattice::stats_header(big);
  snprintf(strtmp,256,"%s %10s",big,"Xrx");
}
