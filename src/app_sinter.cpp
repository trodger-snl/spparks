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

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "app_sinter.h"
#include "domain.h"
#include "lattice.h"
#include "comm_lattice.h"
#include "random_mars.h"
#include "random_park.h"
#include "error.h"

#include <algorithm>
#include <climits>

using namespace SPPARKS_NS;
using namespace sinter;

static const int FRAME_DEPTH = 1;    // frame sites at the box faces
static const int BORDER_DEPTH = 2;   // then pore sites around the compact

// sort by bits lo..63: LSD radix, 16 bits a pass (std::sort of the path
// sites was 40% of a 16-rank 256^3 run)
static void radix_sort(std::vector<unsigned long long> &a, int lo)
{
  std::vector<unsigned long long> b(a.size());
  for (int shift = lo; shift < 64; shift += 16) {
    std::vector<size_t> count(65537,0);
    for (unsigned long long x : a) count[(x >> shift & 0xffff) + 1]++;
    for (int k = 0; k < 65536; k++) count[k+1] += count[k];
    for (unsigned long long x : a) b[count[x >> shift & 0xffff]++] = x;
    a.swap(b);
  }
}

/* ---------------------------------------------------------------------- */

AppSinter::AppSinter(SPPARKS *spk, int narg, char **arg) :
  AppLattice(spk,narg,arg)
{
  if (narg != 1) error->all(FLERR,"Illegal app_style command");

  ninteger = 1;
  ndouble = 0;
  delpropensity = 2;   // an event reads the energy of the neighbour it swaps
  delevent = 1;        // with, and changes that neighbour
  allow_kmc = 0;
  allow_rejection = 1;
  allow_masking = 0;
  allow_app_update = 1;
  numrandom = 1;

  create_arrays();

  par.gg = par.pm = par.anni = 1.0/3.0;
  par.gb_factor = 1.0;
  temperature = par.T[PM] = 1.0;
  t_inverse = 1.0;
  par.T[VAC] = 15.0;
  dt_sweep = 1.0/par.gg;
  time_sinter_start = 50.0;

  vac_made = 0;
  gb_ini = -1.0;
  delta_t = density = 0.0;
  next_gb = -1.0;
}

/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppSinter::grow_app()
{
  spin = iarray[0];
}

/* ---------------------------------------------------------------------- */

void AppSinter::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"event_ratios") == 0) {
    if (narg != 3) error->all(FLERR,"Illegal event_ratios command");
    double gg = atof(arg[0]), pm = atof(arg[1]), anni = atof(arg[2]);
    if (gg <= 0.0 || pm < 0.0 || anni < 0.0)
      error->all(FLERR,"Illegal event_ratios command");
    double total = gg + pm + anni;
    par.gg = gg/total;
    par.pm = pm/total;
    par.anni = anni/total;
    dt_sweep = 1.0/par.gg;

  } else if (strcmp(command,"event_temperatures") == 0 ||
             strcmp(command,"events_temperatures") == 0) {
    if (narg != 3) error->all(FLERR,"Illegal event_temperatures command");
    temperature = atof(arg[0]);
    par.T[PM] = atof(arg[1]);
    par.T[VAC] = atof(arg[2]);
    if (temperature < 0.0 || par.T[PM] < 0.0 || par.T[VAC] < 0.0)
      error->all(FLERR,"Illegal event_temperatures command");
    t_inverse = 1.0/temperature;

  } else if (strcmp(command,"time_sinter_start") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal time_sinter_start command");
    time_sinter_start = atof(arg[0]);
    if (time_sinter_start < 0.0)
      error->all(FLERR,"Illegal time_sinter_start command");

  } else error->all(FLERR,"Unrecognized command");
}

/* ---------------------------------------------------------------------- */

void AppSinter::init_app()
{
  // spins are ints and set i1 unique numbers them by site id
  if (nglobal > MAXSMALLINT)
    error->all(FLERR,"App sinter supports up to 2^31-1 sites");

  // the frame, density box and annihilation paths work on lattice cells
  // numbered as create_sites numbers a fully periodic box
  dimension = domain->dimension;
  nx = domain->nx;
  ny = domain->ny;
  nz = domain->nz;
  if (domain->lattice == NULL || nx == 0)
    error->all(FLERR,"App sinter requires create_sites box in a periodic box");
  nbasis = domain->lattice->nbasis;
  if ((bigint) nx*ny*nz*nbasis != nglobal || !contiguous_sites())
    error->all(FLERR,"App sinter requires create_sites box in a periodic box");
  if (maxneigh > MAXNEIGH)
    error->all(FLERR,"App sinter supports at most 32 neighbors per site");

  key = ranmaster->initial_seed();

  // owned sites by lattice cell: a rank owns a box of cells

  for (int d = 0; d < 3; d++) {
    clo[d] = INT_MAX;
    cn[d] = 0;
  }
  int c[3];
  for (int i = 0; i < nlocal; i++) {
    cell(id[i],nbasis,nx,ny,c);
    for (int d = 0; d < 3; d++) {
      clo[d] = MIN(clo[d],c[d]);
      cn[d] = MAX(cn[d],c[d]+1);
    }
  }
  for (int d = 0; d < 3; d++) cn[d] = nlocal ? cn[d] - clo[d] : 0;
  cell2local.assign((size_t) cn[0]*cn[1]*cn[2]*nbasis,-1);
  for (int i = 0; i < nlocal; i++) {
    cell(id[i],nbasis,nx,ny,c);
    cell2local[(((size_t) (c[2]-clo[2])*cn[1] + c[1]-clo[1])*cn[0] +
                c[0]-clo[0])*nbasis + (id[i]-1) % nbasis] = i;
  }
}

/* ---------------------------------------------------------------------- */

void AppSinter::setup_app()
{
  overimpose_frame();
  set_params(time);
}

/* ----------------------------------------------------------------------
   frame sites on the box faces, then (first run) pore sites, around the
   compact. Later runs keep the border: the compact grows into it as the
   vacancies leave, and nothing changes the frame.
------------------------------------------------------------------------- */

void AppSinter::overimpose_frame()
{
  const int n[3] = {nx,ny,nz};
  int c[3];
  for (int i = 0; i < nlocal; i++) {
    cell(id[i],nbasis,nx,ny,c);
    int depth = INT_MAX;
    for (int d = 0; d < dimension; d++)
      depth = MIN(depth,MIN(c[d],n[d]-1-c[d]));
    if (depth < FRAME_DEPTH) spin[i] = FRAME;
    else if (first_run && depth < FRAME_DEPTH + BORDER_DEPTH) spin[i] = VACANT;
  }
}

/* ----------------------------------------------------------------------
   the parameters of the next sweep, which starts at time t
------------------------------------------------------------------------- */

void AppSinter::set_params(double t)
{
  par.T[GG] = temperature;   // the temperature command may set it
  const double tinv[3] = {t_inverse,1.0/par.T[PM],1.0/par.T[VAC]};
  for (int k = 0; k < 3; k++)
    for (int de = 0; de <= MAXDE; de++)
      par.boltz[k][de] = par.T[k] > 0.0 ? exp(-de*tinv[k]) : 0.0;
  par.sintering = t > time_sinter_start;
}

/* ----------------------------------------------------------------------
   compute energy of site
------------------------------------------------------------------------- */

double AppSinter::site_energy(int i)
{
  return energy(i,numneigh,FDHostNbr{neighbor},spin);
}

/* ---------------------------------------------------------------------- */

void AppSinter::site_event_rejection(int i, RandomPark *random)
{
  int vac;
  if (event(i,*random,numneigh,FDHostNbr{neighbor},spin,par,vac)) naccept++;
  if (vac >= 0) made.push_back(vac);
}

/* ----------------------------------------------------------------------
   after each sweep: annihilate the vacancies it made, update gb_factor
------------------------------------------------------------------------- */

void AppSinter::app_update(double dt)
{
  const double t = time + dt;
  annihilate();
  if (t > time_sinter_start && t > next_gb) next_gb = gb_update(t);
  set_params(t);
}

/* ----------------------------------------------------------------------
   Annihilate the vacancies the sweep made (Tikare 2010): each one still
   a vacancy moves to the compact's surface along the line from it through
   the centre of mass of its adjacent grain, every site on the way taking
   the spin of the next. All ranks process every vacancy, in site id
   order, on one copy of the path sites, and keep the spins they own, so
   the result does not depend on the # of ranks.
------------------------------------------------------------------------- */

void AppSinter::annihilate()
{
  int n = nmade(), ntotal;
  MPI_Allreduce(&n,&ntotal,1,MPI_INT,MPI_SUM,world);
  if (ntotal == 0) return;

  update_ghosts();
  std::vector<Vac> mine, vac;
  candidates(mine);

  int nmine = mine.size()*sizeof(Vac);
  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Allgather(&nmine,1,MPI_INT,counts.data(),1,MPI_INT,world);
  for (int p = 1; p < nprocs; p++) displs[p] = displs[p-1] + counts[p-1];
  vac.resize((displs[nprocs-1] + counts[nprocs-1])/sizeof(Vac));
  MPI_Allgatherv(mine.data(),nmine,MPI_BYTE,vac.data(),counts.data(),
                 displs.data(),MPI_BYTE,world);
  std::sort(vac.begin(),vac.end());
  vac.erase(std::unique(vac.begin(),vac.end(),
                        [](const Vac &a, const Vac &b) {return a.gid == b.gid;}),
            vac.end());
  if (vac.empty()) return;

  // centres of mass of their grains, in lattice cells

  std::vector<int> grains;
  for (const Vac &v : vac) grains.push_back(v.grain);
  std::sort(grains.begin(),grains.end());
  grains.erase(std::unique(grains.begin(),grains.end()),grains.end());
  std::vector<long long> sums(4*grains.size(),0);
  grain_sums(grains,sums);
  MPI_Allreduce(MPI_IN_PLACE,sums.data(),sums.size(),MPI_LONG_LONG,MPI_SUM,
                world);

  // the paths, as (site id, place in the paths), then the current spins
  // of their distinct sites

  std::vector<size_t> start(vac.size()+1);
  std::vector<unsigned long long> occ;
  std::vector<tagint> p;
  for (size_t r = 0; r < vac.size(); r++) {
    const size_t g = std::lower_bound(grains.begin(),grains.end(),
                                      vac[r].grain) - grains.begin();
    p.clear();
    path(vac[r].gid,&sums[4*g],p);
    start[r] = occ.size();
    for (tagint s : p)
      occ.push_back((unsigned long long) s << 32 | occ.size());
  }
  start[vac.size()] = occ.size();

  std::vector<int> at(occ.size());    // place in the paths -> in sites
  std::vector<tagint> sites;
  radix_sort(occ,32);
  for (size_t k = 0; k < occ.size(); k++) {
    if (k == 0 || occ[k] >> 32 != occ[k-1] >> 32)
      sites.push_back(occ[k] >> 32);
    at[occ[k] & 0xffffffff] = sites.size() - 1;
  }

  std::vector<int> idx(sites.size()), val(sites.size(),INT_MIN);
  for (size_t s = 0; s < sites.size(); s++) idx[s] = local_of(sites[s]);
  get_sites(idx,val);
  MPI_Allreduce(MPI_IN_PLACE,val.data(),val.size(),MPI_INT,MPI_MAX,world);

  // collapse the paths in turn; the vacancy ends before the frame
  // ponytail: every rank replays every path (O(total path length) per
  // sweep, not divided by ranks); split into independent groups of paths
  // if that dominates at high rank counts

  for (size_t r = 0; r < vac.size(); r++) {
    const int *q = &at[start[r]], n = start[r+1] - start[r];
    if (n == 0 || val[q[0]] != VACANT) continue;   // an earlier path moved it
    int a = q[0];
    for (int s = 1; s < n; s++) {
      const int b = q[s];
      if (s+1 == n || val[b] == FRAME) break;
      val[a] = val[b];
      a = b;
    }
    val[a] = VACANT;
    if (me == 0) vac_made++;
  }

  set_sites(idx,val);
}

/* ----------------------------------------------------------------------
   the sites from vacancy gid towards its grain's centre of mass (sum =
   # of sites, sums of their cells) to the box face beyond it: the
   vacancy, then one basis-0 site per cell along the line, rounded to the
   nearest cell as the original app's xyz walk; empty if under 2 cells,
   so a vacancy next to the frame stays put. In integers, so that every
   build and device takes the same path.
   ponytail: cells ignore basis offsets (< 1 cell off for fcc/bcc)
------------------------------------------------------------------------- */

void AppSinter::path(tagint gid, const long long *sum, std::vector<tagint> &p)
{
  // nearest integer to a/b (b > 0), halves up; a/b is a cell offset, but
  // a = w D can pass 2^62 on a big 2D box (D ~ N n), so 128 bits
  auto near = [](__int128 a, __int128 b) {
    __int128 q = (2*a + b) / (2*b);
    if ((2*a + b) % (2*b) != 0 && 2*a + b < 0) q--;
    return (long long) q;
  };

  // direction N cm - N v = D; the line leaves the box at t = w_m/|D_m|
  // (in units of N), the least w_d/|D_d| over the axes it moves along
  const int n[3] = {nx,ny,nz};
  int v[3];
  cell(gid,nbasis,nx,ny,v);
  long long D[3] = {0,0,0}, w = 0, dm = 0;
  for (int d = 0; d < dimension; d++) {
    D[d] = sum[1+d] - sum[0]*v[d];
    if (D[d] == 0) continue;
    const long long wd = D[d] > 0 ? n[d] - v[d] : v[d];
    if (dm == 0 || (__int128) wd * llabs(dm) < (__int128) w * llabs(D[d])) {
      w = wd;
      dm = D[d];
    }
  }
  if (dm == 0) return;              // vacancy at the centre of mass

  long long e[3];
  int steps = 0;
  for (int d = 0; d < 3; d++) {
    e[d] = v[d] + near((__int128) w * D[d],llabs(dm));
    e[d] = MAX(0,MIN(n[d]-1,e[d]));
    steps = MAX(steps,(int) llabs(e[d]-v[d]));
  }
  if (steps < 2) return;

  // cell s = v + near((e-v) s, steps), stepped: near's numerator
  // 2 (e-v) s + steps grows by 2 (e-v) <= 2 steps a cell
  long long c[3], r[3];
  for (int d = 0; d < 3; d++) {
    c[d] = v[d];
    r[d] = steps;
  }
  p.push_back(gid);
  for (int s = 1; s <= steps; s++) {
    for (int d = 0; d < 3; d++) {
      r[d] += 2*(e[d]-v[d]);
      if (r[d] >= 2*steps) {
        r[d] -= 2*steps;
        c[d]++;
      } else if (r[d] < 0) {
        r[d] += 2*steps;
        c[d]--;
      }
    }
    p.push_back(((c[2]*ny + c[1])*nx + c[0])*nbasis + 1);
  }
}

/* ---------------------------------------------------------------------- */

int AppSinter::local_of(tagint gid) const
{
  int c[3];
  cell(gid,nbasis,nx,ny,c);
  for (int d = 0; d < 3; d++) {
    c[d] -= clo[d];
    if (c[d] < 0 || c[d] >= cn[d]) return -1;
  }
  return cell2local[(((size_t) c[2]*cn[1] + c[1])*cn[0] + c[0])*nbasis +
                    (gid-1) % nbasis];
}

/* ---------------------------------------------------------------------- */

unsigned long long AppSinter::salt() const
{
  return (unsigned long long) mix(key ^ ((unsigned long long) nsweeps << 32))
    << 32;
}

/* ----------------------------------------------------------------------
   hooks of annihilate(), on the host arrays (sinter/kk: the device's)
------------------------------------------------------------------------- */

void AppSinter::update_ghosts()
{
  if (nprocs > 1) comm->all();
}

// the made vacancies still vacancies, with their adjacent grains
void AppSinter::candidates(std::vector<Vac> &v)
{
  const unsigned long long s = salt();
  for (int j : made) {
    int g = adjacent_grain(j,mix(s + id[j]) % numneigh[j],numneigh,
                           FDHostNbr{neighbor},spin);
    if (g > 0) v.push_back({id[j],g});
  }
  made.clear();
}

// per grain (sorted): # of owned sites, then the sums of their cells
void AppSinter::grain_sums(const std::vector<int> &grains,
                           std::vector<long long> &sums)
{
  int c[3];
  for (int i = 0; i < nlocal; i++) {
    auto it = std::lower_bound(grains.begin(),grains.end(),spin[i]);
    if (it == grains.end() || *it != spin[i]) continue;
    const size_t m = 4*(it - grains.begin());
    cell(id[i],nbasis,nx,ny,c);
    sums[m]++;
    for (int d = 0; d < 3; d++) sums[m+1+d] += c[d];
  }
}

void AppSinter::get_sites(const std::vector<int> &idx, std::vector<int> &val)
{
  for (size_t s = 0; s < idx.size(); s++)
    if (idx[s] >= 0) val[s] = spin[idx[s]];
}

void AppSinter::set_sites(const std::vector<int> &idx,
                          const std::vector<int> &val)
{
  for (size_t s = 0; s < idx.size(); s++)
    if (idx[s] >= 0) spin[idx[s]] = val[s];
}

/* ----------------------------------------------------------------------
   gb_factor = (initial / current mean grain-boundary area)^2, rescaling
   how often vacancies are made; the next update comes sooner while the
   density changes fast (Garcia-Cardona 2011)
------------------------------------------------------------------------- */

double AppSinter::gb_update(double t)
{
  const double gb = gb_average();
  double prev = -1.0;
  if (gb_ini < 0.0) {
    gb_ini = gb;
    delta_t = 20.0;
  } else {
    if (gb > 0.0) par.gb_factor = (gb_ini*gb_ini) / (gb*gb);
    prev = density;
  }
  density = calculate_density();

  const double change = fabs(density - prev);
  if (change > 0.05) delta_t /= 2.0;
  else if (change < 0.01) delta_t *= 2.0;
  return t + delta_t;
}

/* ----------------------------------------------------------------------
   mean grain-boundary area: the # of (grain site, other grain beside it)
   pairs over the # of (grain, other grain beside it) pairs
------------------------------------------------------------------------- */

double AppSinter::gb_average()
{
  update_ghosts();

  long long count[2] = {0,0};    // face sites, faces
  std::vector<unsigned long long> faces;
  gb_faces(count[0],faces);

  // a face may be on several ranks (a grain spans ranks, or its pieces
  // do): count each at the rank its key hashes to

  std::vector<int> scount(nprocs,0), rcount(nprocs), sdispl(nprocs,0),
    rdispl(nprocs,0);
  for (unsigned long long f : faces) scount[mix(f) % nprocs]++;
  for (int p = 1; p < nprocs; p++) sdispl[p] = sdispl[p-1] + scount[p-1];
  std::vector<unsigned long long> send(faces.size()), recv;
  std::vector<int> next(sdispl);
  for (unsigned long long f : faces) send[next[mix(f) % nprocs]++] = f;
  MPI_Alltoall(scount.data(),1,MPI_INT,rcount.data(),1,MPI_INT,world);
  for (int p = 1; p < nprocs; p++) rdispl[p] = rdispl[p-1] + rcount[p-1];
  recv.resize(rdispl[nprocs-1] + rcount[nprocs-1]);
  MPI_Alltoallv(send.data(),scount.data(),sdispl.data(),
                MPI_UNSIGNED_LONG_LONG,recv.data(),rcount.data(),
                rdispl.data(),MPI_UNSIGNED_LONG_LONG,world);
  radix_sort(recv,0);
  count[1] = std::unique(recv.begin(),recv.end()) - recv.begin();

  MPI_Allreduce(MPI_IN_PLACE,count,2,MPI_LONG_LONG,MPI_SUM,world);
  return count[1] ? (double) count[0] / count[1] : 0.0;
}

/* ----------------------------------------------------------------------
   over the owned grain sites: the # of (site, other grain beside it)
   pairs, and the distinct (grain, other grain beside it) pairs, sorted
------------------------------------------------------------------------- */

void AppSinter::gb_faces(long long &fsites,
                         std::vector<unsigned long long> &faces)
{
  for (int i = 0; i < nlocal; i++) {
    const int s = spin[i];
    if (s <= VACANT) continue;
    int u[MAXNEIGH];
    const int nu = other_grains(i,numneigh,FDHostNbr{neighbor},spin,u);
    fsites += nu;
    for (int m = 0; m < nu; m++)
      faces.push_back((unsigned long long) s << 32 | (unsigned) u[m]);
  }
  radix_sort(faces,0);
  faces.erase(std::unique(faces.begin(),faces.end()),faces.end());
}

/* ----------------------------------------------------------------------
   site i is in the central box of a third of the cells along each axis
------------------------------------------------------------------------- */

int AppSinter::in_density_box(int i) const
{
  const int n[3] = {nx,ny,nz};
  int c[3];
  cell(id[i],nbasis,nx,ny,c);
  for (int d = 0; d < dimension; d++) {
    const int lo = n[d]*0.33, hi = lo + n[d]/3;
    if (c[d] <= lo || c[d] >= hi) return 0;
  }
  return 1;
}

/* ----------------------------------------------------------------------
   fraction of grain sites in the central box
------------------------------------------------------------------------- */

double AppSinter::calculate_density()
{
  double count[2] = {0.0,0.0};
  for (int i = 0; i < nlocal; i++)
    if (in_density_box(i)) {
      count[1]++;
      if (spin[i] > VACANT) count[0]++;
    }
  MPI_Allreduce(MPI_IN_PLACE,count,2,MPI_DOUBLE,MPI_SUM,world);
  return count[0] / count[1];
}

/* ---------------------------------------------------------------------- */

void AppSinter::stats_header(char *strtmp)
{
  snprintf(strtmp,256,"%10s %12s %14s %10s %12s",
           "Time","Naccept","Nreject","Nsweeps","Vmade");
}

/* ----------------------------------------------------------------------
   Vmade = vacancies annihilated since the last stats
------------------------------------------------------------------------- */

void AppSinter::stats(char *strtmp)
{
  double v[3] = {(double) naccept,(double) nattempt,(double) vac_made};
  MPI_Allreduce(MPI_IN_PLACE,v,3,MPI_DOUBLE,MPI_SUM,world);
  snprintf(strtmp,256,"%g %g %g %d %g",time,v[0],v[1]-v[0],nsweeps,v[2]);
  vac_made = 0;
}
