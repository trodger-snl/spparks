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
#include "gpu_aware_mpi.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <type_traits>
#include <vector>
#ifdef SPPARKS_HIGHFIVE
#include "highfive/highfive.hpp"
#endif

using namespace SPPARKS_NS;

namespace {

const double SIGMA_SB = 5.670374419e-8;

#ifdef SPPARKS_HIGHFIVE
// ranks take turns on the file (as dump hdf5 does); an HDF5 lock adds nothing
// and fails on file systems mounted without flock
struct NoFileLocking {
  void apply(hid_t fapl) const { H5Pset_file_locking(fapl, false, true); }
};
#endif

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

// S(i) = (V(i+1) - V(i)) / (X(i+1) - X(i)), precomputed. The tables are a
// handful of points and most cells sit in the first segment, so a walk from
// the bottom beats a binary search.
template <typename R, typename E>
KOKKOS_INLINE_FUNCTION
R interp(const Kokkos::View<E*> &X, const Kokkos::View<E*> &V,
         const Kokkos::View<E*> &S, R x)
{
  const int n = (int) X.extent(0);
  if (x <= R(X(0))) return R(V(0));
  if (x >= R(X(n-1))) return R(V(n-1));
  int lo = 0;
  while (R(X(lo + 1)) <= x) lo++;
  return R(V(lo)) + (x - R(X(lo))) * R(S(lo));
}

}   // namespace

/* ---------------------------------------------------------------------- */

ConductionSolverKokkos::ConductionSolverKokkos(SPPARKS *spk,
                                               ConductionTemperatureSource *s) :
  Pointers(spk), src(s) {}

/* ---------------------------------------------------------------------- */

void ConductionSolverKokkos::advance(double t,
                                     Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz,
                                     Kokkos::View<int*> d_af, Kokkos::View<double*> d_T,
                                     int nlocal, int nall, double dx_in, double ts,
                                     double tl_in)
{
  dx = dx_in;
  tl = tl_in;
  const double zl_now = src->get_scan_laser_plane_z();
  if (!built || nall != built_nall || zl_now != built_zl) {
    build(d_xyz, d_af, nlocal, nall, ts);
    reset_field(src->ambient());
    if (!seeded) {
      // grid first (margins and all), then d6 (the sites, exact) if both are given
      if (!src->config().init_grid.empty()) seed_from_grid(src->config().init_grid, t);
      if (src->config().init_d6) seed_from_sites(d_T, nlocal);
    }
    seeded = true;
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

  // far: the box is done; the field stays as it was
  if (src->take_freeze()) t_fd = t;

  if (t > t_fd) {
    const int n = (int) std::ceil((t - t_fd) / dt_fd - 1e-9);
    // Laser off for the whole interval, or (far) on but farther than far from
    // the box throughout: RKL2 super-steps of at most s_max stages, when that
    // is fewer stage evaluations than n explicit steps.
    const int smax = src->config().sts_smax;
    double xl, yl;
    const bool off = src->laser_window(t_fd, t, xl, yl) == 0.0;
    const bool far = !off && src->config().far > 0.0 && src->next_near(t_fd) >= t;
    bool sts = smax >= 2 && n > 2 && (off || far);
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
      for (int k = 0; k < m; k++) sts_step(tau, s, t_fd + k * tau);
    } else {
      const double h = (t - t_fd) / n;
      for (int k = 0; k < n; k++) step(h, t_fd + k * h);
    }
    t_fd = t;
  }
}

/* ---------------------------------------------------------------------- */

void ConductionSolverKokkos::build(Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz,
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
  src->set_box((glo[0] - 0.5) * dx, (ghi[0] + 0.5) * dx, (glo[1] - 0.5) * dx, (ghi[1] + 0.5) * dx);

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
  // ded: birth reaches one layer below the plane (the rise since the last
  // build; unlimited on the first) unless depth is given
  if (cfg.ded_depth > 0.0) birth_depth = cfg.ded_depth;
  else if (built && zl > built_zl) birth_depth = zl - built_zl;

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
  auto faces = [](const Axis &a) { std::vector<double> f(a.lo); f.push_back(a.hi.back()); return f; };
  gfx = faces(ax); gfy = faces(ay); gfz = faces(az);
  gcx0 = cx0; gcy0 = cy0;
  blo[0] = (glo[0] - 0.5) * dx; bhi[0] = (ghi[0] + 0.5) * dx;
  blo[1] = (glo[1] - 0.5) * dx; bhi[1] = (ghi[1] + 0.5) * dx;
  blo[2] = (glo[2] - 0.5) * dx; bhi[2] = az.hi.back();

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
  Ex = Kokkos::View<KK_FLOAT*>("cond_Ex", NXL);
  Ey = Kokkos::View<KK_FLOAT*>("cond_Ey", NYL);
  Ez = Kokkos::View<KK_FLOAT*>("cond_Ez", nz);
  fzlo = axis_view("cond_fzlo", az, 0, nz, 1);
  fzhi = axis_view("cond_fzhi", az, 0, nz, 2);

  // the step kernel indexes cells with an int
  if ((long long) NXL * NYL * nz > INT_MAX)
    error->one(FLERR,"conduction: more than 2^31 grid cells on one rank; use more ranks");
  H = Kokkos::View<KK_FLOAT***>("cond_H", NXL, NYL, nz);
  T = Kokkos::View<KK_FLOAT***>("cond_T", NXL, NYL, nz);
  Tn = Kokkos::View<KK_FLOAT***>("cond_Tn", NXL, NYL, nz);
  P = Kokkos::View<unsigned char***>("cond_P", NXL, NYL, nz);
  Pn = Kokkos::View<unsigned char***>("cond_Pn", NXL, NYL, nz);

  // Phase: lattice cells from i2; elsewhere plate / layer / part rule.
  auto hP = Kokkos::create_mirror_view(P);
  const double plate_top = cfg.plate_top_set ? cfg.plate_top : (glo[2] - 0.5) * dx;
  const bool plate_on = cfg.plate_mat.on;
  const double zplate = src->plate_surface();
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
          // ded: no powder bed, the part stands in gas
          const unsigned char bed = cfg.ded ? GAS : POWDER;
          if (zc < plate_top) p = plate_on ? (DENSE | PLATE) : DENSE;
          else if (zc > zl - cfg.layer_thickness) p = bed;
          else if (!part || part->match(xc / dx, yc / dx, zc / dx)) p = DENSE;
          else p = bed;
        }
        hP(i, j, k) = p;
      }

  // Site -> local cell; owned sites set the phase of their cell.
  // (T, Tn and H share one shape, so one offset serves whichever is T)
  site_cell = Kokkos::View<int*>("cond_site_cell", nall);
  auto hsc = Kokkos::create_mirror_view(site_cell);
  int out = 0, bad6 = 0;
  for (int s = 0; s < nall; s++) {
    const int i = ax.nlo + g[3*s] - glo[0] - cx0 + 1;
    const int j = ay.nlo + g[3*s+1] - glo[1] - cy0 + 1;
    const int k = az.nlo + g[3*s+2] - glo[2];
    if (g[3*s+2] > gz_top) { hsc(s) = -1; continue; }
    if (i < 0 || i > nxo + 1 || j < 0 || j > nyo + 1) { out = 1; hsc(s) = -1; continue; }
    hsc(s) = i * (int) T.stride(0) + j * (int) T.stride(1) + k * (int) T.stride(2);
    if (s < nlocal) {
      // Inside the box i2 alone decides, at every depth: melted (2, 3) is
      // dense, never melted (0, 1) is powder, 5 is gas. So the part's walls
      // are wherever the pools melted, rough as they are. A substrate must be
      // loaded as i2 = 3, or as i2 = 6 to take the plate material. Under ded
      // never-melted sites are unborn: gas until the laser deposits them.
      const int af = haf(s);
      if (af == 6 && !(plate_on && hx(s, 2) * dx < zplate)) bad6 = 1;
      const unsigned char unmelted = cfg.ded ? (POWDER | UNBORN) : POWDER;
      hP(i, j, k) = af == 6 ? (DENSE | PLATE)
                  : (af == 2 || af == 3) ? DENSE : (af == 0 || af == 1) ? unmelted : GAS;
    }
  }
  if (out) error->one(FLERR,"conduction: a ghost site lies outside the 1-cell halo");
  int any6;
  MPI_Allreduce(&bad6, &any6, 1, MPI_INT, MPI_MAX, world);
  if (any6)
    error->all(FLERR,"conduction: i2 = 6 (plate) needs plate_material, and only below "
               "plate_top + plate_growth");
  // packing < 1: powder cells carry that fraction of the dense mass (LOOSE),
  // and keep it once melted (the bed melts in place, it does not shrink).
  if (cfg.packing < 1.0)
    for (size_t n = 0; n < hP.size(); n++)
      if (hP.data()[n] == POWDER) hP.data()[n] = POWDER | LOOSE;
  Kokkos::deep_copy(site_cell, hsc);
  Kokkos::deep_copy(P, hP);

  // Property tables.
  // Both materials share the T grid (enthalpy_table: 0..6000 K at 1 K).
  auto htab = [&](const char *nm, bool plate, double ts_m, double tl_m) {
    std::vector<double> Ht;
    src->enthalpy_table(ts_m, tl_m, Htab_T0, Htab_dT, Ht, plate);
    double Href = 0.0;
    if (std::is_same<KK_FLOAT, float>::value) {
      const double u = (src->ambient() - Htab_T0) / Htab_dT;
      const int i = std::max(0, std::min((int) Ht.size() - 2, (int) std::floor(u)));
      Href = Ht[i] + (u - i) * (Ht[i+1] - Ht[i]);
    }
    Kokkos::View<KK_FLOAT*> d(nm, Ht.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < Ht.size(); i++) h(i) = Ht[i] - Href;
    Kokkos::deep_copy(d, h);
    return d;
  };
  const auto &pm = cfg.plate_mat;
  Htab = htab("cond_Htab", false, ts, tl);
  // no plate: the part's tables stand in, and no cell carries PLATE
  Htab2 = plate_on ? htab("cond_Htab2", true, pm.ts, pm.tl) : Htab;
  if (cfg.ded) Hbirth = H_at(Htab, cfg.T_birth);   // born cells are part powder
  auto up = [](const char *nm, const std::vector<double> &v) {
    Kokkos::View<double*> d(nm, v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); i++) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  };
  kT = up("cond_kT", cfg.k_T);
  kV = up("cond_kV", cfg.k_v);
  kT2 = plate_on ? up("cond_kT2", pm.k_T) : kT;
  kV2 = plate_on ? up("cond_kV2", pm.k_v) : kV;
  auto to_r = [](Kokkos::View<double*> d) {
    if constexpr (std::is_same<KK_FLOAT, double>::value) return d;
    else {
      Kokkos::View<KK_FLOAT*> r(d.label() + "_r", d.extent(0));
      auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
      auto hr = Kokkos::create_mirror_view(r);
      for (size_t i = 0; i < d.extent(0); i++) hr(i) = (KK_FLOAT) hd(i);
      Kokkos::deep_copy(r, hr);
      return r;
    }
  };
  kTr = to_r(kT); kVr = to_r(kV); kT2r = to_r(kT2); kV2r = to_r(kV2);
  wxr = to_r(wx); wyr = to_r(wy); wzr = to_r(wz);

  // Slopes and reciprocals, so the stencil multiplies where it divided:
  // k-table segment slopes, 1/width per cell, 1/(H(i+1) - H(i)) per T step.
  auto upr = [](const char *nm, const std::vector<double> &v) {
    Kokkos::View<KK_FLOAT*> d(nm, v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); i++) h(i) = (KK_FLOAT) v[i];
    Kokkos::deep_copy(d, h);
    return d;
  };
  auto slopes = [&](const char *nm, const std::vector<double> &X, const std::vector<double> &V) {
    std::vector<double> sl(std::max<size_t>(X.size(), 2) - 1, 0.0);
    for (size_t i = 0; i + 1 < X.size(); i++) sl[i] = (V[i+1] - V[i]) / (X[i+1] - X[i]);
    return upr(nm, sl);
  };
  auto recip = [&](const char *nm, Kokkos::View<double*> w) {
    auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    std::vector<double> v(hw.extent(0));
    for (size_t i = 0; i < v.size(); i++) v[i] = hw(i) > 0.0 ? 1.0 / hw(i) : 0.0;
    return upr(nm, v);
  };
  auto dHinv = [&](const char *nm, Kokkos::View<KK_FLOAT*> tab) {
    auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tab);
    std::vector<double> v(ht.extent(0), 0.0);
    for (size_t i = 0; i + 1 < v.size(); i++) {
      const KK_FLOAT dH = ht(i+1) - ht(i);   // as the stencil forms it
      v[i] = dH > 0 ? 1.0 / dH : 0.0;
    }
    return upr(nm, v);
  };
  kSr = slopes("cond_kS", cfg.k_T, cfg.k_v);
  kS2r = plate_on ? slopes("cond_kS2", pm.k_T, pm.k_v) : kSr;
  iwx = recip("cond_iwx", wx); iwy = recip("cond_iwy", wy); iwz = recip("cond_iwz", wz);
  iHtab = dHinv("cond_iHtab", Htab);
  iHtab2 = plate_on ? dHinv("cond_iHtab2", Htab2) : iHtab;

  const size_t nplane = 2 * (size_t) std::max(NYL, NXL) * nz;   // T + phase
  sbuf = Kokkos::View<double*>("cond_sbuf", 2 * nplane);
  rbuf = Kokkos::View<double*>("cond_rbuf", 2 * nplane);
  constexpr bool host_reads_device = Kokkos::SpaceAccessibility<
      Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;
  stage_host = !host_reads_device && !gpu_aware_mpi();
  if (stage_host) {
    hsbuf = decltype(hsbuf)(Kokkos::view_alloc("cond_hsbuf", Kokkos::WithoutInitializing), 2 * nplane);
    hrbuf = decltype(hrbuf)(Kokkos::view_alloc("cond_hrbuf", Kokkos::WithoutInitializing), 2 * nplane);
  }
  // halo phases, so a cell sees its neighbours' material across ranks
  exchange();
  Kokkos::deep_copy(hP, P);

  // Stable step, per cell: C_min V / sum of face conductances at each side's
  // k_max (in series across a material interface), + linearised surface
  // losses at 4000 K; global min so every rank count steps alike. Each
  // material is bounded over its own tables: the part's k_max covers its
  // powder (powder melts to dense), the plate has its own C_min and k_max.
  double kmax = cfg.k_powder;
  for (double v : cfg.k_v) kmax = std::max(kmax, v);
  double cmin = 1e300;
  for (double v : cfg.cp_v) cmin = std::min(cmin, v);
  cmin *= src->density() * std::min(1.0, cfg.packing);
  double kmax_p = kmax, cmin_p = cmin;
  if (plate_on) {
    kmax_p = 0.0;
    for (double v : pm.k_v) kmax_p = std::max(kmax_p, v);
    cmin_p = 1e300;
    for (double v : pm.cp_v) cmin_p = std::min(cmin_p, v * pm.rho);
  }
  const double hsurf = cfg.h + 4.0 * cfg.emissivity * SIGMA_SB * 4000.0 * 4000.0 * 4000.0;
  // an adiabatic far face carries no flux, so it does not limit the step
  const bool far_fixed = cfg.far_bc == ConductionTemperatureSource::FAR_FIXED;
  double dtl = 1e300;
  auto W = [](const Axis &a, int c) { return a.hi[c] - a.lo[c]; };
  // local cell (li, lj, k), halo included -> its material's k_max
  auto kof = [&](int li, int lj, int k) { return (hP(li, lj, k) & PLATE) ? kmax_p : kmax; };
  // face conductance; one material keeps the single-k form (bit-identical)
  auto face = [](double A, double wc, double kc, double wn, double kn) {
    return kc == kn ? A * kc / (0.5 * (wc + wn)) : A / (0.5 * wc / kc + 0.5 * wn / kn);
  };
  for (int i = 0; i < nxo; i++)
    for (int j = 0; j < nyo; j++)
      for (int k = 0; k < nz; k++) {
        const int cx = cx0 + i, cy = cy0 + j, li = i + 1, lj = j + 1;
        const double wxc = W(ax, cx), wyc = W(ay, cy), wzc = W(az, k);
        const double kc = kof(li, lj, k);
        const double farG = far_fixed ? kc : 0.0;
        double G = 0.0;
        for (int d = 0; d < 2; d++) {
          const int s = d ? 1 : -1;
          const int nx = cx + s, ny = cy + s, nk = k + s;
          const double ax_ = wyc * wzc, ay_ = wxc * wzc, az_ = wxc * wyc;
          G += (nx < 0 || nx >= NX) ? farG * ax_ / (0.5 * wxc)
                                    : face(ax_, wxc, kc, W(ax, nx), kof(li + s, lj, k));
          G += (ny < 0 || ny >= NY) ? farG * ay_ / (0.5 * wyc)
                                    : face(ay_, wyc, kc, W(ay, ny), kof(li, lj + s, k));
          G += (nk < 0) ? farG * az_ / (0.5 * wzc)
             : (nk >= nz) ? az_ * hsurf : face(az_, wzc, kc, W(az, nk), kof(li, lj, nk));
        }
        const double c = (hP(li, lj, k) & PLATE) ? cmin_p : cmin;
        dtl = std::min(dtl, c * wxc * wyc * wzc / G);
      }
  MPI_Allreduce(&dtl, &dt_fd, 1, MPI_DOUBLE, MPI_MIN, world);
  dt_fd *= 0.9;

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

// H(Tr) - Href from a material's table (linear interpolation, end slopes beyond).
double ConductionSolverKokkos::H_at(Kokkos::View<KK_FLOAT*> tab, double Tr) const
{
  auto hH = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tab);
  const int n = (int) hH.extent(0);
  const double u = (Tr - Htab_T0) / Htab_dT;
  const int i = std::max(0, std::min(n - 2, (int) std::floor(u)));
  return hH(i) + (u - i) * ((double) hH(i+1) - hH(i));
}

/* ----------------------------------------------------------------------
   init d6: each owned site's cell takes the site's temperature (a stitched
   box hands its field to the next), with H from the cell's material. Sites
   with d6 <= 0 (never written) and the margin cells keep the reset value.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::seed_from_sites(Kokkos::View<double*> d_T, int nlocal)
{
  auto H_ = H; auto T_ = T; auto Tn_ = Tn; auto P_ = P; auto sc = site_cell;
  auto tab = Htab, tab2 = Htab2;
  const double T0 = Htab_T0, dTt = Htab_dT;
  const int n1 = (int) tab.extent(0), n2 = (int) tab2.extent(0);
  Kokkos::parallel_for("ConductionSeedD6", nlocal, KOKKOS_LAMBDA(const int s) {
    const int o = sc(s);
    const double v = d_T(s);
    if (o < 0 || !(v > 0.0)) return;
    const bool plate = P_.data()[o] & PLATE;
    const int n = plate ? n2 : n1;
    const double u = (v - T0) / dTt;
    const int i = u < 0.0 ? 0 : (u > n - 2 ? n - 2 : (int) u);
    const double h = plate ? tab2(i) + (u - i) * ((double) tab2(i+1) - tab2(i))
                           : tab(i) + (u - i) * ((double) tab(i+1) - tab(i));
    H_.data()[o] = h;
    T_.data()[o] = v;
    Tn_.data()[o] = v;
  });
  exchange();
}

void ConductionSolverKokkos::reset_field(double Tr)
{
  const double Hr = H_at(Htab, Tr), Hr2 = H_at(Htab2, Tr);
  if (Hr2 == Hr) Kokkos::deep_copy(H, Hr);
  else {
    auto H_ = H; auto P_ = P;
    const KK_FLOAT h1 = Hr, h2 = Hr2;
    Kokkos::parallel_for("ConductionResetH",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {(int) H.extent(0), (int) H.extent(1), nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        H_(i, j, k) = (P_(i, j, k) & PLATE) ? h2 : h1;
      });
  }
  Kokkos::deep_copy(T, Tr);
  Kokkos::deep_copy(Tn, Tr);
  // phase is kept: a reset cools the part, it does not un-melt powder
  Kokkos::deep_copy(Pn, P);
  exchange();
}

/* ----------------------------------------------------------------------
   dump conduction: datasets T (K, double) and phase (uint8), [z][y][x] over
   the whole grid, the cell faces x_faces, y_faces, z_faces (m, N+1 each) and
   attributes time (s) and dx (m). Ranks write their owned cells in turn into
   FILE.tmp, which rank 0 then renames to FILE: a reader (the next box) never
   sees a half-written grid.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::write_grid(const std::string &file, double time, bool single)
{
#ifndef SPPARKS_HIGHFIVE
  (void) file; (void) time; (void) single;
  error->all(FLERR,"dump conduction needs HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
  if (!built) error->all(FLERR,"dump conduction: no conduction grid yet (a dump before the first step)");
  int me, np;
  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &np);
  const size_t NX = gfx.size() - 1, NY = gfy.size() - 1, NZ = nz;
  auto hT = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), T);
  auto hP = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P);
  std::vector<double> tb((size_t) nz * nyo * nxo);
  std::vector<unsigned char> pb(tb.size());
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < nyo; j++)
      for (int i = 0; i < nxo; i++) {
        const size_t n = ((size_t) k * nyo + j) * nxo + i;
        tb[n] = hT(i+1, j+1, k);
        pb[n] = hP(i+1, j+1, k);
      }
  const std::string tmp = file + ".tmp";
  const std::vector<size_t> off = {0, (size_t) gcy0, (size_t) gcx0};
  const std::vector<size_t> cnt = {NZ, (size_t) nyo, (size_t) nxo};
  int bad = 0;
  std::string why;
  if (me > 0) MPI_Recv(&bad, 1, MPI_INT, me - 1, 0, world, MPI_STATUS_IGNORE);
  if (!bad) {
    try {
      HighFive::FileAccessProps fapl;
      fapl.add(NoFileLocking{});
      const unsigned mode = me == 0
          ? (HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate)
          : HighFive::File::ReadWrite;
      HighFive::File f(tmp, mode, fapl);
      if (me == 0) {
        const HighFive::DataSpace sp({NZ, NY, NX});
        // single: T stored as float, HDF5 converts the double buffer on write (and back on read)
        if (single) f.createDataSet<float>("T", sp);
        else f.createDataSet<double>("T", sp);
        f.createDataSet<unsigned char>("phase", sp);
        f.createDataSet("x_faces", gfx);
        f.createDataSet("y_faces", gfy);
        f.createDataSet("z_faces", gfz);
        f.createAttribute("time", time);
        f.createAttribute("dx", dx);
      }
      f.getDataSet("T").select(off, cnt).write_raw(tb.data());
      f.getDataSet("phase").select(off, cnt).write_raw(pb.data());
      f.flush();
    } catch (const std::exception &e) {
      bad = 1;
      why = e.what();
    }
  }
  if (me < np - 1) MPI_Send(&bad, 1, MPI_INT, me + 1, 0, world);
  if (!why.empty()) error->one(FLERR, ("dump conduction: writing " + tmp + " failed: " + why).c_str());
  int any;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, world);
  if (any) error->all(FLERR, ("dump conduction: writing " + tmp + " failed").c_str());
  if (me == 0 && std::rename(tmp.c_str(), file.c_str()) != 0)
    error->one(FLERR, ("dump conduction: cannot rename " + tmp + " to " + file).c_str());
  MPI_Barrier(world);
#endif
}

/* ----------------------------------------------------------------------
   init grid FILE: each owned cell takes the T of a dump conduction file,
   trilinear between the file's cell centres at its own centre, with H from
   the cell's material, so the previous box's margin heat carries over too.
   Cells centred outside the file's grid keep the reset value. Outside the
   lattice box a powder cell becomes dense where the file's nearest cell is
   dense (powder the previous box melted); lattice cells keep i2's phase.
   ponytail: point interpolation, not a conservative remap; where the new
   cells are coarser than the file's, the heat is sampled, not summed.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::seed_from_grid(const std::string &file, double t)
{
#ifndef SPPARKS_HIGHFIVE
  (void) file; (void) t;
  error->all(FLERR,"conduction: init grid needs HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
  // per local cell along one axis: the file centre below its centre and the
  // weight of the one above; a < 0 = centred outside the file's grid
  struct Wt { int a, b; double w; };
  auto bracket = [](const std::vector<double> &f, const std::vector<double> &gf, int c0, int n) {
    std::vector<double> c(f.size() - 1);
    for (size_t m = 0; m + 1 < f.size(); m++) c[m] = 0.5 * (f[m] + f[m+1]);
    const int nc = (int) c.size();
    std::vector<Wt> out(n);
    for (int m = 0; m < n; m++) {
      const double x = 0.5 * (gf[c0 + m] + gf[c0 + m + 1]);
      if (x < f.front() || x > f.back()) { out[m] = {-1, -1, 0.0}; continue; }
      int a = (int) (std::upper_bound(c.begin(), c.end(), x) - c.begin()) - 1;
      a = std::max(0, std::min(nc - 1, a));
      const int b = std::min(nc - 1, a + 1);
      const double w = b == a ? 0.0 : std::max(0.0, std::min(1.0, (x - c[a]) / (c[b] - c[a])));
      out[m] = {a, b, w};
    }
    return out;
  };
  auto span = [](const std::vector<Wt> &v, int &lo, int &hi) {
    lo = INT_MAX; hi = -1;
    for (const Wt &q : v) if (q.a >= 0) { lo = std::min(lo, q.a); hi = std::max(hi, q.b); }
    return hi >= 0;
  };

  std::vector<Wt> bx, by, bz;
  std::vector<double> tb;
  std::vector<unsigned char> pb;
  int x0 = 0, x1 = -1, y0 = 0, y1 = -1, z0 = 0, z1 = -1;
  double tfile = 0.0;
  bool any = false;
  try {
    HighFive::FileAccessProps fapl;
    fapl.add(NoFileLocking{});
    HighFive::File f(file, HighFive::File::ReadOnly, fapl);
    std::vector<double> fx, fy, fz;
    f.getDataSet("x_faces").read(fx);
    f.getDataSet("y_faces").read(fy);
    f.getDataSet("z_faces").read(fz);
    f.getAttribute("time").read(tfile);
    bx = bracket(fx, gfx, gcx0, nxo);
    by = bracket(fy, gfy, gcy0, nyo);
    bz = bracket(fz, gfz, 0, nz);
    any = span(bx, x0, x1) && span(by, y0, y1) && span(bz, z0, z1);
    if (any) {
      const std::vector<size_t> off = {(size_t) z0, (size_t) y0, (size_t) x0};
      const std::vector<size_t> cnt = {(size_t) (z1 - z0 + 1), (size_t) (y1 - y0 + 1), (size_t) (x1 - x0 + 1)};
      tb.resize(cnt[0] * cnt[1] * cnt[2]);
      pb.resize(tb.size());
      f.getDataSet("T").select(off, cnt).read_raw(tb.data());
      f.getDataSet("phase").select(off, cnt).read_raw(pb.data());
    }
  } catch (const std::exception &e) {
    error->one(FLERR, ("conduction: init grid " + file + ": " + e.what()).c_str());
  }

  long long n = 0;
  if (any) {
    auto hH = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H);
    auto hT = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), T);
    auto hP = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P);
    auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Htab);
    auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Htab2);
    const int sx = x1 - x0 + 1, sy = y1 - y0 + 1;
    auto at = [&](int a, int b, int c) { return ((size_t) (c - z0) * sy + (b - y0)) * sx + (a - x0); };
    for (int i = 1; i <= nxo; i++)
      for (int j = 1; j <= nyo; j++)
        for (int k = 0; k < nz; k++) {
          const Wt &u = bx[i-1], &v = by[j-1], &w = bz[k];
          if (u.a < 0 || v.a < 0 || w.a < 0) continue;
          double Tv = 0.0;
          for (int c = 0; c < 8; c++) {
            const double wt = ((c & 1) ? u.w : 1.0 - u.w) * ((c & 2) ? v.w : 1.0 - v.w) *
                              ((c & 4) ? w.w : 1.0 - w.w);
            if (wt > 0.0) Tv += wt * tb[at((c & 1) ? u.b : u.a, (c & 2) ? v.b : v.a, (c & 4) ? w.b : w.a)];
          }
          const double xc = 0.5 * (gfx[gcx0 + i - 1] + gfx[gcx0 + i]);
          const double yc = 0.5 * (gfy[gcy0 + j - 1] + gfy[gcy0 + j]);
          const double zc = 0.5 * (gfz[k] + gfz[k + 1]);
          const bool lattice = xc > blo[0] && xc < bhi[0] && yc > blo[1] && yc < bhi[1] &&
                               zc > blo[2] && zc < bhi[2];
          unsigned char &p = hP(i, j, k);
          const unsigned char fp = pb[at(u.w < 0.5 ? u.a : u.b, v.w < 0.5 ? v.a : v.b, w.w < 0.5 ? w.a : w.b)];
          if (!lattice && p == POWDER && (fp & 3) == DENSE) p = DENSE;
          else if (!lattice && p == (POWDER | LOOSE) && (fp & 3) == DENSE) p = DENSE | LOOSE;
          const auto &tab = (p & PLATE) ? h2 : h1;
          const int nt = (int) tab.extent(0);
          const double q = (Tv - Htab_T0) / Htab_dT;
          const int t = q < 0.0 ? 0 : (q > nt - 2 ? nt - 2 : (int) q);
          hH(i, j, k) = tab(t) + (q - t) * ((double) tab(t+1) - tab(t));
          hT(i, j, k) = Tv;
          n++;
        }
    Kokkos::deep_copy(H, hH);
    Kokkos::deep_copy(T, hT);
    Kokkos::deep_copy(Tn, T);
    Kokkos::deep_copy(P, hP);
    Kokkos::deep_copy(Pn, P);
  }
  exchange();

  long long nall_ = n, ncell = (long long) nxo * nyo * nz, tot[2], loc[2] = {nall_, ncell};
  MPI_Allreduce(loc, tot, 2, MPI_LONG_LONG, MPI_SUM, world);
  if (domain->me == 0) {
    char msg[512];
    snprintf(msg, sizeof(msg), "conduction: init grid %s (t = %.10g s): %lld of %lld cells "
             "from the file\n", file.c_str(), tfile, tot[0], tot[1]);
    if (screen) fputs(msg, screen);
    if (logfile) fputs(msg, logfile);
    // the gap is not integrated: past the time heat takes to diffuse one cell
    // (dx^2 / alpha at T0) the seeded field is visibly too sharp, or, with the
    // file later than the run, already holds heat this run deposits again
    const double gap = t - tfile, tcell = dx * dx / src->diffusivity();
    if (std::fabs(gap) > tcell) {
      snprintf(msg, sizeof(msg), "conduction: init grid %s is %.3g s %s the run start "
               "(%.3g x the %.3g s one-cell diffusion time); the gap is not integrated",
               file.c_str(), std::fabs(gap), gap > 0 ? "before" : "after", std::fabs(gap) / tcell, tcell);
      error->warning(FLERR, msg);
    }
  }
#endif
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
   far: the laser's path over [t0, t1] as one source plane,
   S2(i, j) = sum over path samples of w Ex(i) Ey(j), samples at most sx/2
   apart. The stage adds Pq S2(i, j) Ez(k), held over the super-step.
------------------------------------------------------------------------- */

double ConductionSolverKokkos::path_source(double t0, double t1)
{
  const double sx = src->sigma_x(), sy = src->sigma_y();
  src->laser_samples(t0, t1, 0.5 * std::min(sx, sy), h_samp);
  const int ns = (int) h_samp.size() / 3;
  if (ns == 0) return 0.0;
  if ((int) d_samp.extent(0) < 3 * ns) d_samp = Kokkos::View<double*>("cond_samp", 3 * ns);
  Kokkos::deep_copy(Kokkos::subview(d_samp, std::make_pair(0, 3 * ns)),
                    Kokkos::View<const double*, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_samp.data(), 3 * ns));
  const int nxl = (int) fxlo.extent(0), nyl = (int) fylo.extent(0);
  if ((int) S2.extent(0) != nxl || (int) S2.extent(1) != nyl)
    S2 = Kokkos::View<KK_FLOAT**>("cond_S2", nxl, nyl);
  auto S2_ = S2; auto smp = d_samp;
  auto fxlo_ = fxlo, fxhi_ = fxhi, fylo_ = fylo, fyhi_ = fyhi;
  Kokkos::parallel_for("ConductionPathSource",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nxl, nyl}),
    KOKKOS_LAMBDA(const int i, const int j) {
      double v = 0.0;
      for (int k = 0; k < ns; k++) {
        const double xs = smp(3*k), ys = smp(3*k+1);
        const double ex = Kokkos::erf((fxhi_(i) - xs) / sx) - Kokkos::erf((fxlo_(i) - xs) / sx);
        if (ex == 0.0) continue;
        v += smp(3*k+2) * ex *
             (Kokkos::erf((fyhi_(j) - ys) / sy) - Kokkos::erf((fylo_(j) - ys) / sy));
      }
      S2_(i, j) = v;
    });
  return 0.25 * src->absorbed_power();
}

/* ----------------------------------------------------------------------
   One RKL2 super-step of length tau in s stages (Meyer, Balsara & Aslam,
   J. Comput. Phys. 257 (2014) 594), from t0. Stable for
   tau <= dt_fd (s^2 + s - 2)/4; second order. The phase only updates on
   the last stage. The laser is off, or (far) held as path_source's plane.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::sts_step(double tau, int s, double t0)
{
  if (Ya.extent(0) != H.extent(0) || Ya.extent(1) != H.extent(1) || Ya.extent(2) != H.extent(2)) {
    Ya = Kokkos::View<KK_FLOAT***>("cond_Ya", H.extent(0), H.extent(1), H.extent(2));
    Yb = Kokkos::View<KK_FLOAT***>("cond_Yb", H.extent(0), H.extent(1), H.extent(2));
    L0 = Kokkos::View<KK_FLOAT***>("cond_L0", H.extent(0), H.extent(1), H.extent(2));
  }
  auto b = [](int j) { return j < 2 ? 1.0 / 3.0 : (j*j + j - 2.0) / (2.0 * j * (j + 1)); };
  const double w1 = 4.0 / (s*s + s - 2.0);

  Stage c;
  c.sts = true; c.store_L0 = true; c.phase = (s == 1);
  if (src->config().far > 0.0) { c.Pq = path_source(t0, t0 + tau); c.path = c.Pq > 0.0; }
  c.mu = 1.0; c.nu = 0.0; c.mut = b(1) * w1 * tau; c.gat = 0.0;
  stage(c, H, H, Ya);                       // Y1 = H + mu~1 tau L(H)
  Kokkos::View<KK_FLOAT***> y1 = Ya, y2 = H, spare = Yb;
  for (int j = 2; j <= s; j++) {
    c.store_L0 = false; c.phase = (j == s);
    c.mu = (2.0*j - 1.0) / j * b(j) / b(j-1);
    c.nu = -(j - 1.0) / j * b(j) / b(j-2);
    c.mut = c.mu * w1 * tau;
    c.gat = -(1.0 - b(j-1)) * c.mut;
    // Y_j overwrites Y_{j-2} in place, except while Y_{j-2} is H (Y0)
    Kokkos::View<KK_FLOAT***> out = (j == 2) ? spare : y2;
    stage(c, y1, y2, out);
    y2 = y1; y1 = out;
  }
  // y1 = Y_s: make it H; keep the other two as stage buffers
  Kokkos::View<KK_FLOAT***> oldH = H;
  H = y1;
  Ya = (y2.data() == oldH.data()) ? spare : y2;
  Yb = oldH;
}

/* ----------------------------------------------------------------------
   Flux divergence + source at T (the state Y1), then the stage update.
------------------------------------------------------------------------- */

void ConductionSolverKokkos::stage(const Stage &c, Kokkos::View<KK_FLOAT***> Y1,
                                   Kokkos::View<KK_FLOAT***> Y2,
                                   Kokkos::View<KK_FLOAT***> Yout)
{
  const auto &cfg = src->config();
  const double xl = c.xl, yl = c.yl, Pq = c.Pq, h = c.h;
  const bool sts = c.sts, store_L0 = c.store_L0, do_phase = c.phase, path = c.path;
  const double mu = c.mu, nu = c.nu, mut = c.mut, gat = c.gat;
  const double sx = src->sigma_x(), sy = src->sigma_y(), sz = src->sigma_z();
  const double zs = zsrc;

  auto wx_ = wxr, wy_ = wyr, wz_ = wzr;
  auto fxlo_ = fxlo, fxhi_ = fxhi, fylo_ = fylo, fyhi_ = fyhi, fzlo_ = fzlo, fzhi_ = fzhi;
  auto Htab1_ = Htab, Htab2_ = Htab2; auto kT_ = kTr, kV_ = kVr, kT2_ = kT2r, kV2_ = kV2r;
  auto kS_ = kSr, kS2_ = kS2r, iwx_ = iwx, iwy_ = iwy, iwz_ = iwz;
  auto iHtab1_ = iHtab, iHtab2_ = iHtab2;
  const double H0 = Htab_T0, dTt = Htab_dT;
  const int nHt = (int) Htab.extent(0);
  const double kp = cfg.k_powder, hc = cfg.h, Tinf = cfg.T_inf;
  const double es = cfg.emissivity * SIGMA_SB;
  const bool fixed = cfg.far_bc == ConductionTemperatureSource::FAR_FIXED;
  const double Tfar = src->ambient();
  const bool exlo = edge[0], exhi = edge[1], eylo = edge[2], eyhi = edge[3];
  const int nxo_ = nxo, nyo_ = nyo, nz_ = nz;
  const double tl_ = tl;
  const double pack = cfg.packing, ipack = 1.0 / cfg.packing;
  // ded birth: laser on, axis within rb of the cell centre, centre in the band
  const bool birth = cfg.ded && Pq > 0.0;
  const double rb2 = cfg.ded_radius * cfg.ded_radius, zb = zs - birth_depth;
  const KK_FLOAT Tb = cfg.T_birth, Hb = Hbirth;

  // The Gaussian separates: integrate it over each cell's extent per axis.
  auto Ex_ = Ex, Ey_ = Ey, Ez_ = Ez;   // KK_FLOAT: rounded once, as the stencil used them
  auto S2_ = S2;                       // far: path plane in place of Ex Ey
  if (Pq > 0.0) {
    const int nmax = std::max(std::max((int) Ex.extent(0), (int) Ey.extent(0)), nz_);
    Kokkos::parallel_for("ConductionSource", nmax, KOKKOS_LAMBDA(const int n) {
      if (!path && n < (int) Ex_.extent(0))
        Ex_(n) = Kokkos::erf((fxhi_(n) - xl) / sx) - Kokkos::erf((fxlo_(n) - xl) / sx);
      if (!path && n < (int) Ey_.extent(0))
        Ey_(n) = Kokkos::erf((fyhi_(n) - yl) / sy) - Kokkos::erf((fylo_(n) - yl) / sy);
      if (n < nz_) {
        const double zt = fzhi_(n) < zs ? fzhi_(n) : zs;   // half space z <= zs
        Ez_(n) = zt > fzlo_(n) ? Kokkos::erf((zt - zs) / sz) - Kokkos::erf((fzlo_(n) - zs) / sz)
                               : 0.0;
      }
    });
  }

  // All math in KK_FLOAT (float under KOKKOS_PREC=mixed/single), H relative to Href.
  // With KK_FLOAT == double every expression is the original one.
  //
  // A team owns a TX x TY column of cells and marches it up through z. A
  // face conductance is A / (r_c + r_n) with r = (w/2) / k(T) of each side,
  // so each cell's r per axis is computed once (one k(T) table lookup and
  // one reciprocal) and shared: x/y neighbours through team scratch (plus a
  // ring of halo cells around the tile), z neighbours in registers as the
  // column advances. The only divisions left per cell are 1/k and the six
  // face conductances. One barrier per plane: the scratch is
  // double-buffered by plane parity.
  using R = KK_FLOAT;
  // host backends (OpenMP, Serial) cannot run 256-thread teams: 1 x 1 tiles.
  // GPU tile from CMake SPPARKS_COND_TILE (default 32x8, tuned on gfx1151).
#ifndef SPK_COND_TX
#define SPK_COND_TX 32
#define SPK_COND_TY 8
#endif
#ifndef SPK_COND_MINB
#define SPK_COND_MINB 0     // CMake SPPARKS_COND_MINB: min teams per SM, 0 = no cap
#endif
  constexpr bool host = std::is_same<Kokkos::DefaultExecutionSpace,
                                     Kokkos::DefaultHostExecutionSpace>::value;
  constexpr int TX = host ? 1 : SPK_COND_TX, TY = host ? 1 : SPK_COND_TY;
  static_assert(TX * TY <= 1024, "SPPARKS_COND_TILE: team size X*Y must be <= 1024");
  const int nbx = (nxo_ + TX - 1) / TX, nby = (nyo_ + TY - 1) / TY;
  const int nsx = TY * (TX + 2), nsy = (TY + 2) * TX;   // rx and ry slots per plane
  using Scratch = Kokkos::View<R*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const size_t shmem = Scratch::shmem_size(2 * (nsx + nsy));
  // Cells by one linear index: every 3D field shares T's extents (and so its
  // layout), so the x/y/z neighbours of cell c are c -+ si, sj, sk.
  const int si = (int) T.stride(0), sj = (int) T.stride(1), sk = (int) T.stride(2);
  const R *Tp = T.data(), *Y1p = Y1.data(), *Y2p = Y2.data();
  const unsigned char *Pp = P.data();
  R *Hp = H.data(), *Yop = Yout.data(), *L0p = L0.data(), *Tnp = Tn.data();
  unsigned char *Pnp = Pn.data();
  using Team = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("ConductionStep",
    Kokkos::TeamPolicy<Kokkos::LaunchBounds<SPK_COND_MINB ? TX * TY : 0, SPK_COND_MINB>>(
        nbx * nby, TX * TY).set_scratch_size(0, Kokkos::PerTeam(shmem)),
    KOKKOS_LAMBDA(const Team &team) {
      Scratch sbuf(team.team_scratch(0), 2 * (nsx + nsy));
      const int t = team.team_rank(), nt = team.team_size();
      const int i0 = 1 + (team.league_rank() % nbx) * TX, j0 = 1 + (team.league_rank() / nbx) * TY;
      const int tx = t % TX, ty = t / TX;
      const int i = i0 + tx, j = j0 + ty;
      // mine: an owned cell this thread updates. has: a cell in the local
      // arrays (owned or halo) whose r this thread publishes, so the halo
      // cells inside a partial tile at a rank boundary are covered too.
      const bool mine = t < TX * TY && i <= nxo_ && j <= nyo_;
      const bool has = t < TX * TY && i <= nxo_ + 1 && j <= nyo_ + 1;
      const R half = R(0.5);
      // k(T) of a cell; p is the full phase byte (PLATE picks the plate's table)
      auto kof = [&](R Tv, unsigned char p) {
        return (p & 3) == POWDER ? R(kp)
             : (p & PLATE) ? interp<R>(kT2_, kV2_, kS2_, Tv) : interp<R>(kT_, kV_, kS_, Tv);
      };
      auto kcell = [&](int cc) { return kof(Tp[cc], Pp[cc]); };
      // halo ring: 2*TY cells left/right (need rx), 2*TX below/above (need ry)
      const int nhalo = 2 * TY + 2 * TX;

      // r = (w/2) * (1/k), one reciprocal per cell; every r (own, scratch,
      // halo ring) is formed this way, so both sides of a face agree on it.
      const int c0 = has ? i * si + j * sj : 0;
      R kc_next = has ? kcell(c0) : R(1);
      R ik_next = R(1) / kc_next;
      R rz_next = half * R(wz_(0)) * ik_next;   // r_z of the plane, carried up
      R rz_prev = R(0);
      for (int k = 0; k < nz_; k++) {
        R *rx = sbuf.data() + (k & 1) * (nsx + nsy), *ry = rx + nsx;
        const int c = c0 + k * sk;
        const R ik = ik_next, rzc = rz_next;
        const int ic = i < nxo_ + 2 ? i : 0, jc = j < nyo_ + 2 ? j : 0;
        const R dxc = R(wx_(ic)), dyc = R(wy_(jc));
        const R dzc = R(wz_(k));
        const R rxc = half * dxc * ik, ryc = half * dyc * ik;
        if (has) {
          rx[ty * (TX + 2) + tx + 1] = rxc;
          ry[(ty + 1) * TX + tx] = ryc;
          kc_next = k + 1 < nz_ ? kcell(c + sk) : R(1);
        }
        ik_next = R(1) / kc_next;
        rz_next = k + 1 < nz_ ? half * R(wz_(k + 1)) * ik_next : R(0);
        for (int hcell = t; hcell < nhalo; hcell += nt) {
          if (hcell < 2 * TY) {                       // x halo: column i0-1 or i0+TX
            const int side = hcell / TY, hy = hcell % TY;
            const int ii = side ? i0 + TX : i0 - 1, jj = j0 + hy;
            if (ii <= nxo_ + 1 && jj <= nyo_)
              rx[hy * (TX + 2) + (side ? TX + 1 : 0)] = half * R(wx_(ii)) * (R(1) / kcell(ii * si + jj * sj + k * sk));
          } else {                                    // y halo: row j0-1 or j0+TY
            const int g = hcell - 2 * TY, side = g / TX, hx = g % TX;
            const int ii = i0 + hx, jj = side ? j0 + TY : j0 - 1;
            if (ii <= nxo_ && jj <= nyo_ + 1)
              ry[(side ? TY + 1 : 0) * TX + hx] = half * R(wy_(jj)) * (R(1) / kcell(ii * si + jj * sj + k * sk));
          }
        }
        team.team_barrier();

        if (mine) {
          // H is the dense-equivalent enthalpy H(T); a LOOSE cell holds
          // pack * H(T) * V, so its heat capacity is scaled by pack.
          const unsigned char pfull = Pp[c], pc = pfull & 3;
          const R Tr = Tp[c];
          if (pfull & UNBORN) {
            const double xc = 0.5 * (fxlo_(i) + fxhi_(i)), yc = 0.5 * (fylo_(j) + fyhi_(j));
            const double zc = 0.5 * (fzlo_(k) + fzhi_(k));
            if (birth && (xc - xl) * (xc - xl) + (yc - yl) * (yc - yl) <= rb2 &&
                zc < zs && zc > zb) {
              Hp[c] = Hb;
              Tnp[c] = Tb; Pnp[c] = (unsigned char) (pfull & ~UNBORN);
            } else {
              if (sts) Yop[c] = Hp[c];
              Tnp[c] = Tr; Pnp[c] = pfull;
            }
          } else if (pc == GAS) {
            if (sts) Yop[c] = Hp[c];
            Tnp[c] = Tr; Pnp[c] = pfull;
          } else {
            auto surf = [&](R A) {
              return -A * (R(hc) * (Tr - R(Tinf)) + R(es) * (Tr*Tr*Tr*Tr - R(Tinf)*R(Tinf)*R(Tinf)*R(Tinf)));
            };
            auto far = [&](R A, R halfw) {   // k looked up again: not carried for the few edge cells
              return fixed ? A * kcell(c) / halfw * (R(Tfar) - Tr) : R(0);
            };
            // r_c, r_n: (w/2)/k of this cell and of the neighbour across the face
            auto face = [&](R A, R rc, int cn, R rn) {
              const unsigned char pn = Pp[cn];
              if ((pn & 3) == GAS || (pn & UNBORN)) return surf(A);
              const R Tv = Tp[cn];
              const R G = A / (rc + rn);
              return G * (Tv - Tr);
            };
            const R Ax = dyc * dzc, Ay = dxc * dzc, Az = dxc * dyc;
            const int sx_ = ty * (TX + 2) + tx + 1, sy_ = (ty + 1) * TX + tx;
            R q = R(0);
            q += (i == 1 && exlo) ? far(Ax, half*dxc) : face(Ax, rxc, c - si, rx[sx_ - 1]);
            q += (i == nxo_ && exhi) ? far(Ax, half*dxc) : face(Ax, rxc, c + si, rx[sx_ + 1]);
            q += (j == 1 && eylo) ? far(Ay, half*dyc) : face(Ay, ryc, c - sj, ry[sy_ - TX]);
            q += (j == nyo_ && eyhi) ? far(Ay, half*dyc) : face(Ay, ryc, c + sj, ry[sy_ + TX]);
            q += (k == 0) ? far(Az, half*dzc) : face(Az, rzc, c - sk, rz_prev);
            q += (k == nz_ - 1) ? surf(Az) : face(Az, rzc, c + sk, rz_next);

            if (Pq > 0.0) q += R(Pq) * (path ? S2_(i, j) : Ex_(i) * Ey_(j)) * Ez_(k);

            const R ivol = iwx_(ic) * iwy_(jc) * iwz_(k) * ((pfull & LOOSE) ? R(ipack) : R(1));
            R Hn;
            if (!sts) {
              Hn = Hp[c] + R(h) * q * ivol;
              Hp[c] = Hn;
            } else {
              const R Lc = q * ivol;
              Hn = R(mu) * Y1p[c] + R(nu) * Y2p[c] + R(1.0 - mu - nu) * Hp[c]
                 + R(mut) * Lc + R(gat) * (store_L0 ? Lc : L0p[c]);
              if (store_L0) L0p[c] = Lc;
              Yop[c] = Hn;
            }
            // T(H): table search, end slopes beyond.
            const auto &Htab_ = (pfull & PLATE) ? Htab2_ : Htab1_;
            const auto &iHtab_ = (pfull & PLATE) ? iHtab2_ : iHtab1_;
            R Tv;
            const R H0r = R(H0), dTr = R(dTt), idTr = R(1.0 / dTt);
            if (Hn <= Htab_(0)) {
              Tv = H0r + (Hn - Htab_(0)) / (Htab_(1) - Htab_(0)) * dTr;
            } else if (Hn >= Htab_(nHt - 1)) {
              Tv = H0r + R(nHt - 1) * dTr +
                   (Hn - Htab_(nHt - 1)) / (Htab_(nHt - 1) - Htab_(nHt - 2)) * dTr;
            } else {
              // H rises strictly with T, so the bracket Htab(lo) <= Hn < Htab(lo+1)
              // is unique: walk to it from the cell's previous T (a few K away)
              // instead of a 13-level binary search.
              const R u = (Tr - H0r) * idTr;
              int lo = u <= R(0) ? 0 : (u >= R(nHt - 2) ? nHt - 2 : (int) u);
              while (lo > 0 && Htab_(lo) > Hn) lo--;
              while (lo < nHt - 2 && Htab_(lo + 1) <= Hn) lo++;
              Tv = H0r + (R(lo) + (Hn - Htab_(lo)) * iHtab_(lo)) * dTr;   // 1/dH, 0 if flat
            }
            Tnp[c] = Tv;
            Pnp[c] = (do_phase && pc == POWDER && Tv >= tl_)
                         ? (unsigned char) (DENSE | (pfull & LOOSE)) : pfull;
          }
        }
        rz_prev = rzc;
      }
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
    // MPI reads the device buffers where it can; else stage only this
    // direction's 4*np values through the staging buffers. Copies and kernels
    // share the default instance, so only it is fenced, never the device.
    Kokkos::DefaultExecutionSpace exec;
    const auto used = std::make_pair(0, 4 * np);
    double *hs = sb.data(), *hr = rb.data();
    if (stage_host) {
      Kokkos::deep_copy(exec, Kokkos::subview(hsbuf, used), Kokkos::subview(sbuf, used));
      hs = hsbuf.data(); hr = hrbuf.data();
    }
    exec.fence("ConductionExchange: packed");
    MPI_Request req[4];
    int nreq = 0;
    // tag = direction of travel: 0 toward lo, 1 toward hi
    if (lo_n >= 0) MPI_Irecv(hr, 2*np, MPI_DOUBLE, lo_n, 1, world, &req[nreq++]);
    if (hi_n >= 0) MPI_Irecv(hr + 2*np, 2*np, MPI_DOUBLE, hi_n, 0, world, &req[nreq++]);
    if (lo_n >= 0) MPI_Isend(hs, 2*np, MPI_DOUBLE, lo_n, 0, world, &req[nreq++]);
    if (hi_n >= 0) MPI_Isend(hs + 2*np, 2*np, MPI_DOUBLE, hi_n, 1, world, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    if (stage_host)
      Kokkos::deep_copy(exec, Kokkos::subview(rbuf, used), Kokkos::subview(hrbuf, used));
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

double ConductionSolverKokkos::ambient() const { return src->ambient(); }
