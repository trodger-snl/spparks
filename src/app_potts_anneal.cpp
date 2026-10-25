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
#include <algorithm>
#include "app_potts_anneal.h"
#include "comm_lattice.h"
#include "random_mars.h"
#include "random_park.h"
#include "error.h"
#include "crystalline_orientations/crystal_symmetries.h"
#include "crystalline_orientations/quaternion.h"

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

AppPottsAnneal::AppPottsAnneal(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternion(spk,narg,arg)
{
  // i1 label, i2 site state, i3 seeds; d1..d4 orientation, d5 stored
  // energy, d8 d9 G V: the slots of additive/texture, so its output loads
  // as is (d6 d7 unused)

  ninteger = 3;
  ndouble = 9;
  allow_kmc = 0;
  allow_masking = 0;
  allow_app_update = 1;
  recreate_arrays();

  if (narg < 4) error->all(FLERR,"Illegal app_style command");
  nspins = atoi(arg[1]);
  symmetries = CRYSTAL::get_symmetries(arg[2]);
  crystal_index = CRYSTAL::index(arg[2]);
  theta_cut = atof(arg[3]);
  if (nspins <= 0 || symmetries.empty() || theta_cut <= 0.0)
    error->all(FLERR,"Illegal app_style command");
  dt_sweep = 1.0;   // time is in MCS

  nrate = rrecover = theta_m = 0.0;
  nradius = 1;
  nuc = rx::Nucleation{nspins, 0.0, 0.0, 0, 0, 0};
  keywords(narg-4,&arg[4],"app_style");

  key = ranmaster->initial_seed();

  // nucleus orientations: uniform random, from the deck seed

  orient.resize(4*NORIENT);
  for (int k = 0; k < NORIENT; k++) {
    quaternion::SiteOrientationRNG rng(key ^ 0x5bd1e995ULL,(uint64_t) k);
    std::vector<double> q = quaternion::generate_unit_quaternion(rng);
    for (int c = 0; c < 4; c++) orient[4*k+c] = q[c];
  }
}

/* ----------------------------------------------------------------------
   nucleation / recovery / mobility settings, from app_style or, between
   runs (one stage of a heat treatment to the next), anneal_modify
------------------------------------------------------------------------- */

void AppPottsAnneal::keywords(int narg, char **arg, const char *cmd)
{
  char msg[64];
  snprintf(msg,64,"Illegal %s command",cmd);
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nucleation") == 0) {
      if (iarg+3 > narg) error->all(FLERR,msg);
      nrate = atof(arg[iarg+1]);
      nuc.hmin = atof(arg[iarg+2]);
      if (nrate < 0.0) error->all(FLERR,msg);
      nuc.boundary = 0;
      nradius = 1;
      iarg += 3;
      while (iarg < narg) {
        if (strcmp(arg[iarg],"boundary") == 0) {
          nuc.boundary = 1;
          iarg++;
        } else if (strcmp(arg[iarg],"radius") == 0 && iarg+1 < narg) {
          nradius = atoi(arg[iarg+1]);
          if (nradius < 1) error->all(FLERR,msg);
          iarg += 2;
        } else break;
      }
    } else if (strcmp(arg[iarg],"recovery") == 0) {
      if (iarg+2 > narg) error->all(FLERR,msg);
      rrecover = atof(arg[iarg+1]);
      if (rrecover < 0.0) error->all(FLERR,msg);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mobility") == 0) {
      if (iarg+2 > narg) error->all(FLERR,msg);
      theta_m = atof(arg[iarg+1]);
      if (theta_m < 0.0) error->all(FLERR,msg);
      iarg += 2;
    } else error->all(FLERR,msg);
  }
}

/* ---------------------------------------------------------------------- */

void AppPottsAnneal::grow_app()
{
  AppPottsQuaternion::grow_app();
  phase = iarray[1];
  seedspin = iarray[2];
  h = darray[4];
}

/* ----------------------------------------------------------------------
   check the sites; orient those without an orientation by their label
------------------------------------------------------------------------- */

void AppPottsAnneal::init_app()
{
  if (maxneigh > anneal::MAXNEIGH)
    error->all(FLERR,"potts/anneal supports at most 32 neighbors");

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {
    if (spin[i] < 1 || h[i] < 0.0) flag = 1;
    // normalise only clearly non-unit input: renormalising is not
    // idempotent in the last bit (see QuaternionKokkos::normalize_host)
    double q[4] = {q0[i],qx[i],qy[i],qz[i]};
    const double n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (n2 < 1.0e-12) {
      quaternion::SiteOrientationRNG rng(key,(uint64_t) spin[i]);
      std::vector<double> u = quaternion::generate_unit_quaternion(rng);
      for (int c = 0; c < 4; c++) q[c] = u[c];
    }
    if (n2 < 1.0e-12 || fabs(sqrt(n2) - 1.0) > 1.0e-10) {
      quaternion::normalize(q);
      q0[i] = q[0]; qx[i] = q[1]; qy[i] = q[2]; qz[i] = q[3];
    }
  }
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");

  if (nrate > 0.0 && nglobal >= (tagint) INT_MAX - nspins)
    error->all(FLERR,"potts/anneal nucleus labels (nspins + site id) overflow int");
  nuc.idbase = nspins + nglobal;
}

/* ---------------------------------------------------------------------- */

void AppPottsAnneal::setup_end_app()
{
  // fronts roughen when H outgrows the boundary energy (see potts/rx);
  // checked every run, since stored_energy and set change H between runs

  double hmax = 0.0, hmaxall;
  for (int i = 0; i < nlocal; i++)
    if (phase[i] != anneal::VOID && spin[i] <= nspins) hmax = MAX(hmax,h[i]);
  MPI_Allreduce(&hmax,&hmaxall,1,MPI_DOUBLE,MPI_MAX,world);
  if (hmaxall >= maxneigh - 2 && me == 0)
    error->warning(FLERR,"potts/anneal stored energy >= maxneigh-2: "
                   "recrystallization fronts will be rough");

  par.sym = symmetries.data();
  par.nsym = symmetries.size()/4;
  par.theta_cut = theta_cut;
  par.temp = temperature;
  par.tinv = temperature > 0.0 ? t_inverse : 0.0;
  par.theta_m = theta_m;
}

/* ----------------------------------------------------------------------
   stored_energy pdas K rho0 theta_c b rho_ref Hmax
   H of the solidified sites (i2 = 3) from their G (d8) and V (d9):
   PDAS lambda = K G^-1/2 V^-1/4 (microns, as spparks-micro-analysis),
   rho = rho0 + theta_c/(b lambda), H = Hmax rho/rho_ref
------------------------------------------------------------------------- */

void AppPottsAnneal::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"stored_energy") == 0) stored_energy(narg,arg);
  else if (strcmp(command,"anneal_modify") == 0) keywords(narg,arg,command);
  else error->all(FLERR,"Unrecognized command");
}

void AppPottsAnneal::stored_energy(int narg, char **arg)
{
  if (narg != 7 || strcmp(arg[0],"pdas") != 0)
    error->all(FLERR,"Illegal stored_energy command");
  const double k = atof(arg[1]), rho0 = atof(arg[2]), thc = atof(arg[3]);
  const double b = atof(arg[4]), rref = atof(arg[5]), hmax = atof(arg[6]);
  if (k <= 0.0 || rho0 < 0.0 || thc < 0.0 || b <= 0.0 || rref <= 0.0 || hmax < 0.0)
    error->all(FLERR,"Illegal stored_energy command");

  const double *G = darray[7], *V = darray[8];
  bigint n = 0, nall;
  for (int i = 0; i < nlocal; i++) {
    if (phase[i] != 3 || G[i] <= 0.0 || V[i] <= 0.0) continue;
    const double lambda = k * pow(G[i],-0.5) * pow(V[i],-0.25) * 1.0e-6;
    h[i] = hmax * (rho0 + thc/(b*lambda)) / rref;
    n++;
  }
  MPI_Allreduce(&n,&nall,1,MPI_SPK_BIGINT,MPI_SUM,world);
  if (me == 0) {
    char str[128];
    snprintf(str,128,"  " BIGINT_FORMAT " stored energies from PDAS\n",nall);
    if (screen) fputs(str,screen);
    if (logfile) fputs(str,logfile);
  }
}

/* ----------------------------------------------------------------------
   boundary + stored energy
------------------------------------------------------------------------- */

double AppPottsAnneal::site_energy(int i)
{
  if (phase[i] == anneal::VOID) return 0.0;
  double q[4] = {q0[i],qx[i],qy[i],qz[i]};
  return anneal::boundary(i,spin[i],q,numneigh,FDHostNbr{neighbor},
                          host_sites(),par) + h[i];
}

/* ---------------------------------------------------------------------- */

void AppPottsAnneal::site_event_rejection(int i, RandomPark *random)
{
  if (anneal::event(i,*random,numneigh,FDHostNbr{neighbor},host_sites(),par))
    naccept++;
}

/* ----------------------------------------------------------------------
   after each sweep: recovery, then nucleation. Seeds as in potts/rx
   (static); each grows by nradius dilation passes over the deformed sites
   (lowest seed id wins), so a nucleus is a cube of half-width nradius,
   cut by recrystallized grains and voids. It takes label nspins + the
   seed's id and an orientation from the table keyed to that id.
------------------------------------------------------------------------- */

unsigned long long AppPottsAnneal::salt() const
{
  return gw_mix(key ^ ((unsigned long long) nsweeps << 32));
}

// # of seeds over all ranks
bigint AppPottsAnneal::nseeds() const
{
  bigint n = 0, nall;
  for (int i = 0; i < nlocal; i++) n += seedspin[i] != 0;
  MPI_Allreduce(&n,&nall,1,MPI_SPK_BIGINT,MPI_SUM,world);
  return nall;
}

void AppPottsAnneal::app_update(double dt)
{
  if (rrecover > 0.0) {
    const double f = exp(-rrecover * dt);
    for (int i = 0; i < nlocal; i++) anneal::recover(i,h,f);
  }

  if (nrate == 0.0) return;
  if (nprocs > 1) comm->all();

  const int nseed = nuc.boundary ? nlocal : nlocal + nghost;
  const unsigned long long s = salt();
  rx::Nucleation q = nuc;
  q.prob = nrate * dt;
  for (int c = 0; c < nseed; c++)
    seedspin[c] = phase[c] == anneal::VOID ? 0 :
      rx::seed(c,id[c],numneigh,FDHostNbr{neighbor},spin,h,q,s);
  if (nseeds() == 0) return;

  // ghosts of the seeds: computed in place, except with boundary (whose
  // test needs the ghosts' neighbours) and after the first pass

  std::vector<int> next(nlocal);
  for (int pass = 0; pass < nradius; pass++) {
    if ((pass > 0 || nuc.boundary) && nprocs > 1) comm->all();
    for (int i = 0; i < nlocal; i++)
      next[i] = anneal::grow_nucleus(i,numneigh,FDHostNbr{neighbor},spin,phase,
                                     seedspin,nspins);
    std::copy(next.begin(),next.end(),seedspin);
  }

  for (int i = 0; i < nlocal; i++) {
    const int snew = seedspin[i];
    if (!snew) continue;
    const double *o = &orient[4*(gw_mix(key ^ (unsigned long long)
                                        (snew - nspins)) % NORIENT)];
    spin[i] = snew;
    q0[i] = o[0]; qx[i] = o[1]; qy[i] = o[2]; qz[i] = o[3];
    h[i] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   recrystallized fraction of the non-void sites
------------------------------------------------------------------------- */

void AppPottsAnneal::stats(char *strtmp)
{
  char big[256];
  AppLattice::stats(big);
  bigint n[2] = {0,0}, nall[2];
  for (int i = 0; i < nlocal; i++) {
    if (phase[i] == anneal::VOID) continue;
    n[0]++;
    if (spin[i] > nspins) n[1]++;
  }
  MPI_Allreduce(n,nall,2,MPI_SPK_BIGINT,MPI_SUM,world);
  snprintf(strtmp,256,"%s %10.6f",big,nall[0] ? (double) nall[1]/nall[0] : 0.0);
}

void AppPottsAnneal::stats_header(char *strtmp)
{
  char big[256];
  AppLattice::stats_header(big);
  snprintf(strtmp,256,"%s %10s",big,"Xrx");
}
