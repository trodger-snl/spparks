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

#include "conduction_solver_kokkos.h"
#include "temperature_source_conduction.h"
#include "domain.h"
#include "region.h"
#include "error.h"
#include "spktype.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace SPPARKS_NS;

namespace {

const double SIGMA_SB = 5.670374419e-8;

// Outward cell widths covering length L: dx*r, dx*r^2, ... capped at cap; a
// remainder under half a cell is merged into the last one.
std::vector<double> stretched(double dx, double r, double cap, double L)
{
  std::vector<double> w;
  double sum = 0.0, cur = dx;
  while (sum < L * (1.0 - 1e-12)) {
    cur *= r;
    if (cap > 0.0 && cur > cap) cur = std::max(cap, dx);
    const double take = std::min(cur, L - sum);
    if (take < 0.5 * cur && !w.empty()) { w.back() += take; break; }
    w.push_back(take);
    sum += take;
  }
  return w;
}

// Global cell faces along one axis: nlo stretched cells, the lattice sites
// g_lo..g_hi (spacing dx), then the high-side stretched cells.
struct Axis {
  std::vector<double> lo, hi;
  int nlo = 0;
};

Axis make_axis(int g_lo, int g_hi, double dx, double mlo, double mhi,
               double r, double cap)
{
  Axis a;
  const std::vector<double> wl = stretched(dx, r, cap, mlo);
  const std::vector<double> wh = stretched(dx, r, cap, mhi);
  a.nlo = (int) wl.size();
  double f = (g_lo - 0.5) * dx;
  for (int c = a.nlo - 1; c >= 0; c--) f -= wl[c];
  for (int c = a.nlo - 1; c >= 0; c--) { a.lo.push_back(f); f += wl[c]; a.hi.push_back(f); }
  for (int g = g_lo; g <= g_hi; g++) { a.lo.push_back((g - 0.5) * dx); a.hi.push_back((g + 0.5) * dx); }
  f = (g_hi + 0.5) * dx;
  for (double w : wh) { a.lo.push_back(f); f += w; a.hi.push_back(f); }
  return a;
}

template <typename R>
KOKKOS_INLINE_FUNCTION
R interp(const Kokkos::View<double*> &X, const Kokkos::View<double*> &V, R x)
{
  const int n = (int) X.extent(0);
  if (x <= R(X(0))) return R(V(0));
  if (x >= R(X(n-1))) return R(V(n-1));
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) { const int m = (lo + hi) / 2; if (R(X(m)) <= x) lo = m; else hi = m; }
  return R(V(lo)) + (x - R(X(lo))) / (R(X(hi)) - R(X(lo))) * (R(V(hi)) - R(V(lo)));
}

}   // namespace

/* ---------------------------------------------------------------------- */

ConductionSolverKokkos::ConductionSolverKokkos(SPPARKS *spk,
                                               ConductionTemperatureSource *s) :
  Pointers(spk), src(s) {}

/* ---------------------------------------------------------------------- */

double ConductionSolverKokkos::fill(double t, Kokkos::View<double*> d_T,
                                    Kokkos::View<double*[3]> d_xyz,
                                    Kokkos::View<int*> d_af, int nlocal,
                                    int nall, double dx_in, double ts, double tl_in)
{
  dx = dx_in;
  tl = tl_in;
  const double zl_now = src->get_scan_laser_plane_z();
  if (!built || nall != built_nall || zl_now != built_zl) {
    build(d_xyz, d_af, nlocal, nall, ts);
    reset_field(src->ambient());
    t_fd = t;
  }

  double Tr;
  if (src->take_reset(Tr)) { reset_field(Tr); t_fd = t; }
  if (t < t_fd) {
    if (domain->me == 0)
      error->warning(FLERR,"conduction: time moved backwards; field reset to ambient");
    reset_field(src->ambient());
    t_fd = t;
  }

  if (t > t_fd) {
    const int n = (int) std::ceil((t - t_fd) / dt_fd - 1e-9);
    // Laser off for the whole interval: RKL2 super-steps of at most s_max
    // stages, when that is fewer stage evaluations than n explicit steps.
    const int smax = src->config().sts_smax;
    double xl, yl;
    bool sts = smax >= 2 && n > 2 && src->laser_window(t_fd, t, xl, yl) == 0.0;
    int m = 0, s = 0;
    if (sts) {
      const double tau_max = dt_fd * (smax * smax + smax - 2) / 4.0;
      m = (int) std::ceil((t - t_fd) / tau_max - 1e-9);
      const double r = (t - t_fd) / m / dt_fd;
      s = std::max(2, (int) std::ceil((-1.0 + std::sqrt(9.0 + 16.0 * r)) / 2.0 - 1e-9));
      sts = (long) m * s < n;
    }
    if (sts) {
      const double tau = (t - t_fd) / m;
      for (int k = 0; k < m; k++) sts_step(tau, s);
    } else {
      const double h = (t - t_fd) / n;
      for (int k = 0; k < n; k++) step(h, t_fd + k * h);
    }
    t_fd = t;
  }
  return sample(d_T, nlocal, nall);
}

/* ---------------------------------------------------------------------- */

void ConductionSolverKokkos::build(Kokkos::View<double*[3]> d_xyz,
                                   Kokkos::View<int*> d_af, int nlocal,
                                   int nall, double ts)
{
  const auto &cfg = src->config();
  if (src->fluctuating())
    error->all(FLERR,"conduction: laser_fluctuations are not supported");
  if (domain->procgrid[2] != 1)
    error->all(FLERR,"conduction: processors must not split z (use processors * * 1)");
  if (domain->xperiodic || domain->yperiodic || domain->zperiodic)
    error->all(FLERR,"conduction: boundary must be non-periodic (n n n)");

  auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_xyz);
  auto haf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_af);

  // Sites must sit on integer lattice coordinates (sc lattice, spacing 1).
  int lo[3] = {INT_MAX, INT_MAX, INT_MAX}, hi[3] = {INT_MIN, INT_MIN, INT_MIN};
  std::vector<int> g(3 * nall);
  int bad = 0;
  for (int s = 0; s < nall; s++)
    for (int d = 0; d < 3; d++) {
      const double v = hx(s, d);
      g[3*s+d] = (int) std::lround(v);
      if (std::fabs(v - g[3*s+d]) > 1e-6) bad = 1;
      if (s < nlocal) { lo[d] = std::min(lo[d], g[3*s+d]); hi[d] = std::max(hi[d], g[3*s+d]); }
    }
  int any_bad;
  MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, world);
  if (any_bad) error->all(FLERR,"conduction: needs a simple cubic lattice with spacing 1");
  if (nlocal == 0) error->one(FLERR,"conduction: every rank needs sites");
  int glo[3], ghi[3];
  MPI_Allreduce(lo, glo, 3, MPI_INT, MPI_MIN, world);
  MPI_Allreduce(hi, ghi, 3, MPI_INT, MPI_MAX, world);

  // Grid top: the highest site layer at or below the laser plane.
  zl = src->get_scan_laser_plane_z();
  const int gz_top = std::min(ghi[2], (int) std::floor(zl / dx + 1e-9));
  if (gz_top < glo[2])
    error->all(FLERR,"conduction: laser plane is below the lattice");

  const double r = cfg.stretch, cap = cfg.max_cell;
  const Axis ax = make_axis(glo[0], ghi[0], dx, cfg.margin[0], cfg.margin[1], r, cap);
  const Axis ay = make_axis(glo[1], ghi[1], dx, cfg.margin[2], cfg.margin[3], r, cap);
  const Axis az = make_axis(glo[2], gz_top, dx, cfg.plate, 0.0, r, cap);
  const int NX = (int) ax.lo.size(), NY = (int) ay.lo.size();
  nz = (int) az.lo.size();
  // A laser plane above the top face (e.g. at ztop*dx, half a cell up) would
  // put the upper tail of the Gaussian outside the grid; anchor it at the
  // free surface instead so the absorbed power is all deposited.
  zsrc = std::min(zl, az.hi.back());

  const int *pg = domain->procgrid, *ml = domain->myloc;
  edge[0] = ml[0] == 0;  edge[1] = ml[0] == pg[0] - 1;
  edge[2] = ml[1] == 0;  edge[3] = ml[1] == pg[1] - 1;
  auto rank = [&](int i, int j) { return i + pg[0] * (j + pg[1] * ml[2]); };
  nbr[0] = edge[0] ? -1 : rank(ml[0]-1, ml[1]);
  nbr[1] = edge[1] ? -1 : rank(ml[0]+1, ml[1]);
  nbr[2] = edge[2] ? -1 : rank(ml[0], ml[1]-1);
  nbr[3] = edge[3] ? -1 : rank(ml[0], ml[1]+1);

  const int cx0 = edge[0] ? 0 : ax.nlo + lo[0] - glo[0];
  const int cx1 = edge[1] ? NX - 1 : ax.nlo + hi[0] - glo[0];
  const int cy0 = edge[2] ? 0 : ay.nlo + lo[1] - glo[1];
  const int cy1 = edge[3] ? NY - 1 : ay.nlo + hi[1] - glo[1];
  nxo = cx1 - cx0 + 1;
  nyo = cy1 - cy0 + 1;
  const int NXL = nxo + 2, NYL = nyo + 2;

  // Per-local-cell geometry (halo cells past a domain edge get dummies).
  auto axis_view = [](const char *nm, const Axis &a, int c0, int n, int which) {
    Kokkos::View<double*> v(nm, n);
    auto h = Kokkos::create_mirror_view(v);
    for (int i = 0; i < n; i++) {
      const int c = c0 + i;
      const bool in = c >= 0 && c < (int) a.lo.size();
      h(i) = !in ? (which == 0 ? 1.0 : 0.0)
           : which == 0 ? a.hi[c] - a.lo[c] : which == 1 ? a.lo[c] : a.hi[c];
    }
    Kokkos::deep_copy(v, h);
    return v;
  };
  wx = axis_view("cond_wx", ax, cx0 - 1, NXL, 0);
  fxlo = axis_view("cond_fxlo", ax, cx0 - 1, NXL, 1);
  fxhi = axis_view("cond_fxhi", ax, cx0 - 1, NXL, 2);
  wy = axis_view("cond_wy", ay, cy0 - 1, NYL, 0);
  fylo = axis_view("cond_fylo", ay, cy0 - 1, NYL, 1);
  fyhi = axis_view("cond_fyhi", ay, cy0 - 1, NYL, 2);
  wz = axis_view("cond_wz", az, 0, nz, 0);
  Ex = Kokkos::View<double*>("cond_Ex", NXL);
  Ey = Kokkos::View<double*>("cond_Ey", NYL);
  Ez = Kokkos::View<double*>("cond_Ez", nz);
  fzlo = axis_view("cond_fzlo", az, 0, nz, 1);
  fzhi = axis_view("cond_fzhi", az, 0, nz, 2);

  H = Kokkos::View<fastreal***>("cond_H", NXL, NYL, nz);
  T = Kokkos::View<fastreal***>("cond_T", NXL, NYL, nz);
  Tn = Kokkos::View<fastreal***>("cond_Tn", NXL, NYL, nz);
  P = Kokkos::View<unsigned char***>("cond_P", NXL, NYL, nz);
  Pn = Kokkos::View<unsigned char***>("cond_Pn", NXL, NYL, nz);

  // Phase: lattice cells from i2; elsewhere plate / layer / part rule.
  auto hP = Kokkos::create_mirror_view(P);
  const double plate_top = cfg.plate_top_set ? cfg.plate_top : (glo[2] - 0.5) * dx;
  Region *part = nullptr;
  if (!cfg.part_region.empty())
    part = domain->regions[domain->find_region((char *) cfg.part_region.c_str())];
  const int nix = ghi[0] - glo[0] + 1, niy = ghi[1] - glo[1] + 1;
  for (int i = 1; i <= nxo; i++)
    for (int j = 1; j <= nyo; j++)
      for (int k = 0; k < nz; k++) {
        const int cx = cx0 + i - 1, cy = cy0 + j - 1;
        const bool inner = cx >= ax.nlo && cx < ax.nlo + nix &&
                           cy >= ay.nlo && cy < ay.nlo + niy && k >= az.nlo;
        unsigned char p;
        if (inner) p = GAS;   // overwritten by the site below, if any
        else {
          const double xc = 0.5 * (ax.lo[cx] + ax.hi[cx]);
          const double yc = 0.5 * (ay.lo[cy] + ay.hi[cy]);
          const double zc = 0.5 * (az.lo[k] + az.hi[k]);
          if (zc < plate_top) p = DENSE;
          else if (zc > zl - cfg.layer_thickness) p = POWDER;
          else if (!part || part->match(xc / dx, yc / dx, zc / dx)) p = DENSE;
          else p = POWDER;
        }
        hP(i, j, k) = p;
      }

  // Site -> local cell; owned sites set the phase of their cell.
  site_cell = Kokkos::View<int*[3]>("cond_site_cell", nall);
  auto hsc = Kokkos::create_mirror_view(site_cell);
  int out = 0;
  for (int s = 0; s < nall; s++) {
    const int i = ax.nlo + g[3*s] - glo[0] - cx0 + 1;
    const int j = ay.nlo + g[3*s+1] - glo[1] - cy0 + 1;
    const int k = az.nlo + g[3*s+2] - glo[2];
    if (g[3*s+2] > gz_top) { hsc(s, 0) = -1; hsc(s, 1) = hsc(s, 2) = 0; continue; }
    if (i < 0 || i > nxo + 1 || j < 0 || j > nyo + 1) { out = 1; hsc(s, 0) = -1; continue; }
    hsc(s, 0) = i; hsc(s, 1) = j; hsc(s, 2) = k;
    if (s < nlocal) {
      // Inside the box i2 alone decides, at every depth: melted (2, 3) is
      // dense, never melted (0, 1) is powder, 5 is gas. So the part's walls
      // are wherever the pools melted, rough as they are. A substrate must be
      // loaded as i2 = 3.
      const int af = haf(s);
      hP(i, j, k) = (af == 2 || af == 3) ? DENSE : (af == 0 || af == 1) ? POWDER : GAS;
    }
  }
  if (out) error->one(FLERR,"conduction: a ghost site lies outside the 1-cell halo");
  // packing < 1: powder cells carry that fraction of the dense mass (LOOSE),
  // and keep it once melted (the bed melts in place, it does not shrink).
  if (cfg.packing < 1.0)
    for (size_t n = 0; n < hP.size(); n++)
      if (hP.data()[n] == POWDER) hP.data()[n] = POWDER | LOOSE;
  Kokkos::deep_copy(site_cell, hsc);
  Kokkos::deep_copy(P, hP);

  // Property tables.
  std::vector<double> Ht;
  src->enthalpy_table(ts, tl, Htab_T0, Htab_dT, Ht);
  double Href = 0.0;
  if (std::is_same<fastreal, float>::value) {
    const double u = (src->ambient() - Htab_T0) / Htab_dT;
    const int i = std::max(0, std::min((int) Ht.size() - 2, (int) std::floor(u)));
    Href = Ht[i] + (u - i) * (Ht[i+1] - Ht[i]);
  }
  Htab = Kokkos::View<fastreal*>("cond_Htab", Ht.size());
  {
    auto h = Kokkos::create_mirror_view(Htab);
    for (size_t i = 0; i < Ht.size(); i++) h(i) = Ht[i] - Href;
    Kokkos::deep_copy(Htab, h);
  }
  auto up = [](const char *nm, const std::vector<double> &v) {
    Kokkos::View<double*> d(nm, v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); i++) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  };
  kT = up("cond_kT", cfg.k_T);
  kV = up("cond_kV", cfg.k_v);

  // Stable step: C_min V / sum of face conductances at k_max (+ linearised
  // surface losses at 4000 K), global min so every rank count steps alike.
  double kmax = cfg.k_powder;
  for (double v : cfg.k_v) kmax = std::max(kmax, v);
  double cmin = 1e300;
  for (double v : cfg.cp_v) cmin = std::min(cmin, v);
  cmin *= src->density() * std::min(1.0, cfg.packing);
  const double hsurf = cfg.h + 4.0 * cfg.emissivity * SIGMA_SB * 4000.0 * 4000.0 * 4000.0;
  // an adiabatic far face carries no flux, so it does not limit the step
  const double farG = cfg.far_bc == ConductionTemperatureSource::FAR_FIXED ? kmax : 0.0;
  double dtl = 1e300;
  auto W = [](const Axis &a, int c) { return a.hi[c] - a.lo[c]; };
  for (int i = 0; i < nxo; i++)
    for (int j = 0; j < nyo; j++)
      for (int k = 0; k < nz; k++) {
        const int cx = cx0 + i, cy = cy0 + j;
        const double wxc = W(ax, cx), wyc = W(ay, cy), wzc = W(az, k);
        double G = 0.0;
        for (int d = 0; d < 2; d++) {
          const int nx = cx + (d ? 1 : -1), ny = cy + (d ? 1 : -1), nk = k + (d ? 1 : -1);
          const double ax_ = wyc * wzc, ay_ = wxc * wzc, az_ = wxc * wyc;
          G += (nx < 0 || nx >= NX) ? farG * ax_ / (0.5 * wxc)
                                    : ax_ * kmax / (0.5 * (wxc + W(ax, nx)));
          G += (ny < 0 || ny >= NY) ? farG * ay_ / (0.5 * wyc)
                                    : ay_ * kmax / (0.5 * (wyc + W(ay, ny)));
          G += (nk < 0) ? farG * az_ / (0.5 * wzc)
             : (nk >= nz) ? az_ * hsurf : az_ * kmax / (0.5 * (wzc + W(az, nk)));
        }
        dtl = std::min(dtl, cmin * wxc * wyc * wzc / G);
      }
  MPI_Allreduce(&dtl, &dt_fd, 1, MPI_DOUBLE, MPI_MIN, world);
  dt_fd *= 0.9;

  const size_t nplane = 2 * (size_t) std::max(NYL, NXL) * nz;   // T + phase
  sbuf = Kokkos::View<double*>("cond_sbuf", 2 * nplane);
  rbuf = Kokkos::View<double*>("cond_rbuf", 2 * nplane);

  built = true;
  built_nall = nall;
  built_zl = zl;

  long long ncell = (long long) nxo * nyo * nz, ntot;
  MPI_Allreduce(&ncell, &ntot, 1, MPI_LONG_LONG, MPI_SUM, world);
  if (domain->me == 0) {
    char msg[256];
    snprintf(msg, sizeof(msg), "conduction: grid %d x %d x %d = %lld cells, "
             "dt_fd = %g s, top face z = %g m (laser plane %g m, source plane %g m)\n",
             NX, NY, nz, ntot, dt_fd, az.hi.back(), zl, zsrc);
    if (screen) fputs(msg, screen);
    if (logfile) fputs(msg, logfile);
  }
}

/* ---------------------------------------------------------------------- */

void ConductionSolverKokkos::reset_field(double Tr)
{
  // H(Tr) from the table (linear interpolation, end slopes beyond).
  auto hH = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Htab);
  const int n = (int) hH.extent(0);
  const double u = (Tr - Htab_T0) / Htab_dT;
  const int i = std::max(0, std::min(n - 2, (int) std::floor(u)));
  const double Hr = hH(i) + (u - i) * ((double) hH(i+1) - hH(i));
  Kokkos::deep_copy(H, Hr);
  Kokkos::deep_copy(T, Tr);
  Kokkos::deep_copy(Tn, Tr);
  // phase is kept: a reset cools the part, it does not un-melt powder
  Kokkos::deep_copy(Pn, P);
  exchange();
}

/* ----------------------------------------------------------------------
   One explicit step of length h starting at t0.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::step(double h, double t0)
{
  Stage c;
  c.h = h;
  const double frac = src->laser_window(t0, t0 + h, c.xl, c.yl);
  c.Pq = 0.25 * frac * src->absorbed_power();   // P/4 * prod(d erf)
  stage(c, H, H, H);
}

/* ----------------------------------------------------------------------
   One RKL2 super-step of length tau in s stages (Meyer, Balsara & Aslam,
   J. Comput. Phys. 257 (2014) 594), laser off. Stable for
   tau <= dt_fd (s^2 + s - 2)/4; second order. The phase only updates on
   the last stage.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::sts_step(double tau, int s)
{
  if (Ya.extent(0) != H.extent(0) || Ya.extent(1) != H.extent(1) || Ya.extent(2) != H.extent(2)) {
    Ya = Kokkos::View<fastreal***>("cond_Ya", H.extent(0), H.extent(1), H.extent(2));
    Yb = Kokkos::View<fastreal***>("cond_Yb", H.extent(0), H.extent(1), H.extent(2));
    L0 = Kokkos::View<fastreal***>("cond_L0", H.extent(0), H.extent(1), H.extent(2));
  }
  auto b = [](int j) { return j < 2 ? 1.0 / 3.0 : (j*j + j - 2.0) / (2.0 * j * (j + 1)); };
  const double w1 = 4.0 / (s*s + s - 2.0);

  Stage c;
  c.sts = true; c.store_L0 = true; c.phase = (s == 1);
  c.mu = 1.0; c.nu = 0.0; c.mut = b(1) * w1 * tau; c.gat = 0.0;
  stage(c, H, H, Ya);                       // Y1 = H + mu~1 tau L(H)
  Kokkos::View<fastreal***> y1 = Ya, y2 = H, spare = Yb;
  for (int j = 2; j <= s; j++) {
    c.store_L0 = false; c.phase = (j == s);
    c.mu = (2.0*j - 1.0) / j * b(j) / b(j-1);
    c.nu = -(j - 1.0) / j * b(j) / b(j-2);
    c.mut = c.mu * w1 * tau;
    c.gat = -(1.0 - b(j-1)) * c.mut;
    // Y_j overwrites Y_{j-2} in place, except while Y_{j-2} is H (Y0)
    Kokkos::View<fastreal***> out = (j == 2) ? spare : y2;
    stage(c, y1, y2, out);
    y2 = y1; y1 = out;
  }
  // y1 = Y_s: make it H; keep the other two as stage buffers
  Kokkos::View<fastreal***> oldH = H;
  H = y1;
  Ya = (y2.data() == oldH.data()) ? spare : y2;
  Yb = oldH;
}

/* ----------------------------------------------------------------------
   Flux divergence + source at T (the state Y1), then the stage update.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::stage(const Stage &c, Kokkos::View<fastreal***> Y1,
                                   Kokkos::View<fastreal***> Y2,
                                   Kokkos::View<fastreal***> Yout)
{
  const auto &cfg = src->config();
  const double xl = c.xl, yl = c.yl, Pq = c.Pq, h = c.h;
  const bool sts = c.sts, store_L0 = c.store_L0, do_phase = c.phase;
  const double mu = c.mu, nu = c.nu, mut = c.mut, gat = c.gat;
  const double sx = src->sigma_x(), sy = src->sigma_y(), sz = src->sigma_z();
  const double zs = zsrc;

  auto H_ = H; auto Y1_ = Y1; auto Y2_ = Y2; auto Yo_ = Yout; auto L0_ = L0;
  auto T_ = T; auto Tn_ = Tn; auto P_ = P; auto Pn_ = Pn;
  auto wx_ = wx, wy_ = wy, wz_ = wz;
  auto fxlo_ = fxlo, fxhi_ = fxhi, fylo_ = fylo, fyhi_ = fyhi, fzlo_ = fzlo, fzhi_ = fzhi;
  auto Htab_ = Htab; auto kT_ = kT; auto kV_ = kV;
  const double H0 = Htab_T0, dTt = Htab_dT;
  const int nHt = (int) Htab.extent(0);
  const double kp = cfg.k_powder, hc = cfg.h, Tinf = cfg.T_inf;
  const double es = cfg.emissivity * SIGMA_SB;
  const bool fixed = cfg.far_bc == ConductionTemperatureSource::FAR_FIXED;
  const double Tfar = src->ambient();
  const bool exlo = edge[0], exhi = edge[1], eylo = edge[2], eyhi = edge[3];
  const int nxo_ = nxo, nyo_ = nyo, nz_ = nz;
  const double tl_ = tl;
  const double pack = cfg.packing;

  // The Gaussian separates: integrate it over each cell's extent per axis.
  auto Ex_ = Ex, Ey_ = Ey, Ez_ = Ez;
  if (Pq > 0.0) {
    const int nmax = std::max(std::max((int) Ex.extent(0), (int) Ey.extent(0)), nz_);
    Kokkos::parallel_for("ConductionSource", nmax, KOKKOS_LAMBDA(const int n) {
      if (n < (int) Ex_.extent(0))
        Ex_(n) = Kokkos::erf((fxhi_(n) - xl) / sx) - Kokkos::erf((fxlo_(n) - xl) / sx);
      if (n < (int) Ey_.extent(0))
        Ey_(n) = Kokkos::erf((fyhi_(n) - yl) / sy) - Kokkos::erf((fylo_(n) - yl) / sy);
      if (n < nz_) {
        const double zt = fzhi_(n) < zs ? fzhi_(n) : zs;   // half space z <= zs
        Ez_(n) = zt > fzlo_(n) ? Kokkos::erf((zt - zs) / sz) - Kokkos::erf((fzlo_(n) - zs) / sz)
                               : 0.0;
      }
    });
  }

  // All math in fastreal (float under SPPARKS_FP32; gfx1151 runs FP64 at a
  // small fraction of FP32 and this kernel is FP64-bound), H relative to
  // Href. With fastreal == double every expression is the original one.
  using R = fastreal;
  Kokkos::parallel_for("ConductionStep",
    Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>(
      {1, 1, 0}, {nxo_ + 1, nyo_ + 1, nz_}),
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
      // H is the dense-equivalent enthalpy H(T); a LOOSE cell holds
      // pack * H(T) * V, so its heat capacity is scaled by pack.
      const unsigned char pfull = P_(i, j, k), pc = pfull & 3;
      const double Tc = T_(i, j, k);
      if (pc == GAS) {
        if (sts) Yo_(i, j, k) = H_(i, j, k);
        Tn_(i, j, k) = Tc; Pn_(i, j, k) = pfull; return;
      }

      auto kof = [&](double Tv, unsigned char p) {
        return p == POWDER ? R(kp) : interp<R>(kT_, kV_, Tv);
      };
      const R Tr = R(Tc);
      const R kc = kof(Tr, pc);
      auto surf = [&](R A) {
        return -A * (R(hc) * (Tr - R(Tinf)) + R(es) * (Tr*Tr*Tr*Tr - R(Tinf)*R(Tinf)*R(Tinf)*R(Tinf)));
      };
      auto far = [&](R A, R half) {
        return fixed ? A * kc / half * (R(Tfar) - Tr) : R(0);
      };
      auto face = [&](R A, R half_c, int ii, int jj, int kk, R half_n) {
        const unsigned char pn = P_(ii, jj, kk) & 3;
        if (pn == GAS) return surf(A);
        const R Tv = R(T_(ii, jj, kk));
        const R G = A / (half_c / kc + half_n / kof(Tv, pn));
        return G * (Tv - Tr);
      };

      const R dxc = R(wx_(i)), dyc = R(wy_(j)), dzc = R(wz_(k));
      const R Ax = dyc * dzc, Ay = dxc * dzc, Az = dxc * dyc, half = R(0.5);
      R q = R(0);
      q += (i == 1 && exlo) ? far(Ax, half*dxc) : face(Ax, half*dxc, i-1, j, k, half*R(wx_(i-1)));
      q += (i == nxo_ && exhi) ? far(Ax, half*dxc) : face(Ax, half*dxc, i+1, j, k, half*R(wx_(i+1)));
      q += (j == 1 && eylo) ? far(Ay, half*dyc) : face(Ay, half*dyc, i, j-1, k, half*R(wy_(j-1)));
      q += (j == nyo_ && eyhi) ? far(Ay, half*dyc) : face(Ay, half*dyc, i, j+1, k, half*R(wy_(j+1)));
      q += (k == 0) ? far(Az, half*dzc) : face(Az, half*dzc, i, j, k-1, half*R(wz_(k-1)));
      q += (k == nz_ - 1) ? surf(Az) : face(Az, half*dzc, i, j, k+1, half*R(wz_(k+1)));

      if (Pq > 0.0) q += R(Pq) * R(Ex_(i)) * R(Ey_(j)) * R(Ez_(k));

      const R vol = dxc * dyc * dzc * ((pfull & LOOSE) ? R(pack) : R(1));
      R Hn;
      if (!sts) {
        Hn = H_(i, j, k) + R(h) * q / vol;
        H_(i, j, k) = Hn;
      } else {
        const R Lc = q / vol;
        Hn = R(mu) * Y1_(i, j, k) + R(nu) * Y2_(i, j, k) + R(1.0 - mu - nu) * H_(i, j, k)
           + R(mut) * Lc + R(gat) * (store_L0 ? Lc : L0_(i, j, k));
        if (store_L0) L0_(i, j, k) = Lc;
        Yo_(i, j, k) = Hn;
      }
      // T(H): table search, end slopes beyond.
      R Tv;
      const R H0r = R(H0), dTr = R(dTt);
      if (Hn <= Htab_(0)) {
        Tv = H0r + (Hn - Htab_(0)) / (Htab_(1) - Htab_(0)) * dTr;
      } else if (Hn >= Htab_(nHt - 1)) {
        Tv = H0r + R(nHt - 1) * dTr +
             (Hn - Htab_(nHt - 1)) / (Htab_(nHt - 1) - Htab_(nHt - 2)) * dTr;
      } else {
        // H rises strictly with T, so the bracket Htab(lo) <= Hn < Htab(lo+1)
        // is unique: walk to it from the cell's previous T (a few K away)
        // instead of a 13-level binary search.
        const R u = (Tr - H0r) / dTr;
        int lo = u <= R(0) ? 0 : (u >= R(nHt - 2) ? nHt - 2 : (int) u);
        while (lo > 0 && Htab_(lo) > Hn) lo--;
        while (lo < nHt - 2 && Htab_(lo + 1) <= Hn) lo++;
        const R dH = Htab_(lo + 1) - Htab_(lo);
        Tv = H0r + (R(lo) + (dH > R(0) ? (Hn - Htab_(lo)) / dH : R(0))) * dTr;
      }
      Tn_(i, j, k) = Tv;
      Pn_(i, j, k) = (do_phase && pc == POWDER && Tv >= tl_)
                   ? (unsigned char) (DENSE | (pfull & LOOSE)) : pfull;
    });

  std::swap(T, Tn);
  std::swap(P, Pn);
  exchange();
}

/* ----------------------------------------------------------------------
   Halo exchange of T and phase: x faces (owned y range plus y halo), then
   y faces including the x halo, so edges and corners arrive.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::exchange()
{
  auto T_ = T; auto P_ = P; auto sb = sbuf; auto rb = rbuf;
  const int nxo_ = nxo, nyo_ = nyo, nz_ = nz;

  for (int dir = 0; dir < 2; dir++) {
    const int lo_n = nbr[2*dir], hi_n = nbr[2*dir+1];
    if (lo_n < 0 && hi_n < 0) continue;
    const int nline = (dir == 0 ? nyo_ : nxo_) + 2;
    const int np = nline * nz_;   // cells per plane
    // layout: [side][field][cell], side 0 = lo, 1 = hi
    Kokkos::parallel_for("ConductionPack", np, KOKKOS_LAMBDA(const int p) {
      const int a = p / nz_, k = p % nz_;
      for (int side = 0; side < 2; side++) {
        const int i = dir == 0 ? (side ? nxo_ : 1) : a;
        const int j = dir == 0 ? a : (side ? nyo_ : 1);
        sb((2*side) * np + p) = T_(i, j, k);
        sb((2*side + 1) * np + p) = (double) P_(i, j, k);
      }
    });
    auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sbuf);
    auto hr = Kokkos::create_mirror_view(Kokkos::HostSpace(), rbuf);
    MPI_Request req[4];
    int nreq = 0;
    // tag = direction of travel: 0 toward lo, 1 toward hi
    if (lo_n >= 0) MPI_Irecv(hr.data(), 2*np, MPI_DOUBLE, lo_n, 1, world, &req[nreq++]);
    if (hi_n >= 0) MPI_Irecv(hr.data() + 2*np, 2*np, MPI_DOUBLE, hi_n, 0, world, &req[nreq++]);
    if (lo_n >= 0) MPI_Isend(hs.data(), 2*np, MPI_DOUBLE, lo_n, 0, world, &req[nreq++]);
    if (hi_n >= 0) MPI_Isend(hs.data() + 2*np, 2*np, MPI_DOUBLE, hi_n, 1, world, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    Kokkos::deep_copy(rbuf, hr);
    const bool has_lo = lo_n >= 0, has_hi = hi_n >= 0;
    Kokkos::parallel_for("ConductionUnpack", np, KOKKOS_LAMBDA(const int p) {
      const int a = p / nz_, k = p % nz_;
      for (int side = 0; side < 2; side++) {
        if (!(side ? has_hi : has_lo)) continue;
        const int i = dir == 0 ? (side ? nxo_ + 1 : 0) : a;
        const int j = dir == 0 ? a : (side ? nyo_ + 1 : 0);
        T_(i, j, k) = rb((2*side) * np + p);
        P_(i, j, k) = (unsigned char) rb((2*side + 1) * np + p);
      }
    });
  }
}

/* ---------------------------------------------------------------------- */

double ConductionSolverKokkos::sample(Kokkos::View<double*> d_T, int nlocal, int nall)
{
  auto T_ = T; auto sc = site_cell;
  const double Ta = src->ambient();
  double m = 0.0;
  Kokkos::parallel_reduce("ConductionSample", nall, KOKKOS_LAMBDA(const int s, double &mx) {
    const int i = sc(s, 0);
    const double v = i < 0 ? Ta : T_(i, sc(s, 1), sc(s, 2));
    d_T(s) = v;
    if (s < nlocal && v > mx) mx = v;
  }, Kokkos::Max<double>(m));
  return m;
}
