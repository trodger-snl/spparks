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

template <typename R, typename E>
KOKKOS_INLINE_FUNCTION
R interp(const Kokkos::View<E*> &X, const Kokkos::View<E*> &V, R x)
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

void ConductionSolverKokkos::advance(double t,
                                     Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz,
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
  site_cell = Kokkos::View<int*[3]>("cond_site_cell", nall);
  auto hsc = Kokkos::create_mirror_view(site_cell);
  int out = 0, bad6 = 0;
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
    Ya = Kokkos::View<KK_FLOAT***>("cond_Ya", H.extent(0), H.extent(1), H.extent(2));
    Yb = Kokkos::View<KK_FLOAT***>("cond_Yb", H.extent(0), H.extent(1), H.extent(2));
    L0 = Kokkos::View<KK_FLOAT***>("cond_L0", H.extent(0), H.extent(1), H.extent(2));
  }
  auto b = [](int j) { return j < 2 ? 1.0 / 3.0 : (j*j + j - 2.0) / (2.0 * j * (j + 1)); };
  const double w1 = 4.0 / (s*s + s - 2.0);

  Stage c;
  c.sts = true; c.store_L0 = true; c.phase = (s == 1);
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
  const bool sts = c.sts, store_L0 = c.store_L0, do_phase = c.phase;
  const double mu = c.mu, nu = c.nu, mut = c.mut, gat = c.gat;
  const double sx = src->sigma_x(), sy = src->sigma_y(), sz = src->sigma_z();
  const double zs = zsrc;

  auto wx_ = wxr, wy_ = wyr, wz_ = wzr;
  auto fxlo_ = fxlo, fxhi_ = fxhi, fylo_ = fylo, fyhi_ = fyhi, fzlo_ = fzlo, fzhi_ = fzhi;
  auto Htab1_ = Htab, Htab2_ = Htab2; auto kT_ = kTr, kV_ = kVr, kT2_ = kT2r, kV2_ = kV2r;
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
  // ded birth: laser on, axis within rb of the cell centre, centre in the band
  const bool birth = cfg.ded && Pq > 0.0;
  const double rb2 = cfg.ded_radius * cfg.ded_radius, zb = zs - birth_depth;
  const KK_FLOAT Tb = cfg.T_birth, Hb = Hbirth;

  // The Gaussian separates: integrate it over each cell's extent per axis.
  auto Ex_ = Ex, Ey_ = Ey, Ez_ = Ez;   // KK_FLOAT: rounded once, as the stencil used them
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

  // All math in KK_FLOAT (float under KOKKOS_PREC=mixed/single), H relative to Href.
  // With KK_FLOAT == double every expression is the original one.
  //
  // A team owns a TX x TY column of cells and marches it up through z. A
  // face conductance is A / (r_c + r_n) with r = (w/2) / k(T) of each side,
  // so each cell's r per axis is computed once (one k(T) table lookup and
  // three divisions) and shared: x/y neighbours through team scratch (plus a
  // ring of halo cells around the tile), z neighbours in registers as the
  // column advances. The operands are the ones a per-face evaluation uses,
  // so the result is bit-identical to it, at a seventh of the lookups and
  // about a third of the divisions. One barrier per plane: the scratch is
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
             : (p & PLATE) ? interp<R>(kT2_, kV2_, Tv) : interp<R>(kT_, kV_, Tv);
      };
      auto kcell = [&](int cc) { return kof(Tp[cc], Pp[cc]); };
      // halo ring: 2*TY cells left/right (need rx), 2*TX below/above (need ry)
      const int nhalo = 2 * TY + 2 * TX;

      const int c0 = has ? i * si + j * sj : 0;
      R kc_next = has ? kcell(c0) : R(1);
      // r_z of this plane's cell, carried from the plane below (same operands)
      R rz_next = half * R(wz_(0)) / kc_next;
      R rz_prev = R(0);
      for (int k = 0; k < nz_; k++) {
        R *rx = sbuf.data() + (k & 1) * (nsx + nsy), *ry = rx + nsx;
        const int c = c0 + k * sk;
        const R kc = kc_next, rzc = rz_next;
        const R dxc = R(wx_(i < nxo_ + 2 ? i : 0)), dyc = R(wy_(j < nyo_ + 2 ? j : 0));
        const R dzc = R(wz_(k));
        const R rxc = half * dxc / kc, ryc = half * dyc / kc;
        if (has) {
          rx[ty * (TX + 2) + tx + 1] = rxc;
          ry[(ty + 1) * TX + tx] = ryc;
          kc_next = k + 1 < nz_ ? kcell(c + sk) : R(1);
        }
        rz_next = k + 1 < nz_ ? half * R(wz_(k + 1)) / kc_next : R(0);
        for (int hcell = t; hcell < nhalo; hcell += nt) {
          if (hcell < 2 * TY) {                       // x halo: column i0-1 or i0+TX
            const int side = hcell / TY, hy = hcell % TY;
            const int ii = side ? i0 + TX : i0 - 1, jj = j0 + hy;
            if (ii <= nxo_ + 1 && jj <= nyo_)
              rx[hy * (TX + 2) + (side ? TX + 1 : 0)] = half * R(wx_(ii)) / kcell(ii * si + jj * sj + k * sk);
          } else {                                    // y halo: row j0-1 or j0+TY
            const int g = hcell - 2 * TY, side = g / TX, hx = g % TX;
            const int ii = i0 + hx, jj = side ? j0 + TY : j0 - 1;
            if (ii <= nxo_ && jj <= nyo_ + 1)
              ry[(side ? TY + 1 : 0) * TX + hx] = half * R(wy_(jj)) / kcell(ii * si + jj * sj + k * sk);
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
            auto far = [&](R A, R halfw) {
              return fixed ? A * kc / halfw * (R(Tfar) - Tr) : R(0);
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

            if (Pq > 0.0) q += R(Pq) * Ex_(i) * Ey_(j) * Ez_(k);

            const R vol = dxc * dyc * dzc * ((pfull & LOOSE) ? R(pack) : R(1));
            R Hn;
            if (!sts) {
              Hn = Hp[c] + R(h) * q / vol;
              Hp[c] = Hn;
            } else {
              const R Lc = q / vol;
              Hn = R(mu) * Y1p[c] + R(nu) * Y2p[c] + R(1.0 - mu - nu) * Hp[c]
                 + R(mut) * Lc + R(gat) * (store_L0 ? Lc : L0p[c]);
              if (store_L0) L0p[c] = Lc;
              Yop[c] = Hn;
            }
            // T(H): table search, end slopes beyond.
            const auto &Htab_ = (pfull & PLATE) ? Htab2_ : Htab1_;
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
