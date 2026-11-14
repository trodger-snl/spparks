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

#ifndef SPK_MELT_FLOW_KOKKOS_H
#define SPK_MELT_FLOW_KOKKOS_H

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Melt-pool flow: D3Q19 lattice Boltzmann on a uniform block of cells,
   one rank. It knows nothing of SPPARKS: the caller fills type, T and fl
   and reads u, so tools/melt_flow_check.cpp drives it against analytical
   solutions. R is the float type of the distributions and the math.

   Lattice units throughout (cell 1, step 1, density 1): the caller scales.
     nu        viscosity * dt / dx^2
     accel     uniform acceleration * dt^2 / dx
     buoy      a = buoy * (T - Tref): e.g. z = beta g dt^2 / dx (hot rises)
     marangoni a_xy = marangoni * (T's difference per cell along x, y), on
               surface cells only: dsigma/dT * dt^2 / (rho dx^3)
     darcy     a = -darcy (1 - fl)^2 / (fl^3 + 1e-3) u (mushy drag), * dt

   Collision: two relaxation times with Lambda = 3/16 (a bounce-back wall
   then sits half-way between the cells whatever nu is), Guo forcing, and
   an optional Smagorinsky eddy viscosity (smag = its constant, 0 = off).
   Distributions are stored as f - w, so single precision keeps the small
   velocities of a low-Mach flow.

   Cells: FLUID; WALL (half-way bounce-back; also every cell beyond the
   block in x, y and below it in z, unless that axis is periodic); GAS
   above a fluid cell is a free surface, flat and free-slip (specular
   reflection), and so is the block's top. GAS beside a fluid cell acts as
   a wall. Surface cells (GAS or the block's top above) take the Marangoni
   force. Cells that are not FLUID hold f = w (rest), so a cell that melts
   starts at rest.

   ux, uy, uz are the mean of the last two steps. Bounce-back walls conserve
   a staggered momentum, sum of (-1)^(x+t) j_x per axis, that no collision
   damps: any abrupt change of force leaves a checkerboard that flips sign
   every step (5e-5 for a force of 1e-4 switched on at rest). The two-step
   mean cancels it; callers should take an even number of steps between
   reads. The surface force acts on the whole top cell, so that cell's
   velocity lags the true surface profile by about tau / (8 mu) (first
   order in the cell size); the cells below are second order.

   A window passed to step() must hold every FLUID cell and every cell that
   was FLUID at either of the last two steps, so freezing cells are reset
   in both copies of f.
------------------------------------------------------------------------- */

template <class R>
struct MeltFlow {
  enum Type : unsigned char { FLUID = 0, WALL = 1, GAS = 2 };

  struct Params {
    double nu = 1.0 / 6.0;
    double smag = 0.0;
    double accel[3] = {0, 0, 0};
    double buoy[3] = {0, 0, 0};
    double Tref = 0.0;
    double marangoni = 0.0;
    double darcy = 0.0;
    bool periodic[3] = {false, false, false};
  } p;

  int nx = 0, ny = 0, nz = 0;
  Kokkos::View<R****> f, fn;                 // (x, y, z, q): f - w
  Kokkos::View<unsigned char***> type;
  Kokkos::View<R***> T, fl;                  // temperature, liquid fraction
  Kokkos::View<R***> ux, uy, uz, rho;        // output: two-step mean u, rho - 1
  Kokkos::View<R***> px, py, pz;             // the last step's u

  void resize(int nx_, int ny_, int nz_) {
    nx = nx_; ny = ny_; nz = nz_;
    f = Kokkos::View<R****>("flow_f", nx, ny, nz, 19);
    fn = Kokkos::View<R****>("flow_fn", nx, ny, nz, 19);
    type = Kokkos::View<unsigned char***>("flow_type", nx, ny, nz);
    T = Kokkos::View<R***>("flow_T", nx, ny, nz);
    fl = Kokkos::View<R***>("flow_fl", nx, ny, nz);
    ux = Kokkos::View<R***>("flow_ux", nx, ny, nz);
    uy = Kokkos::View<R***>("flow_uy", nx, ny, nz);
    uz = Kokkos::View<R***>("flow_uz", nx, ny, nz);
    rho = Kokkos::View<R***>("flow_rho", nx, ny, nz);
    px = Kokkos::View<R***>("flow_px", nx, ny, nz);
    py = Kokkos::View<R***>("flow_py", nx, ny, nz);
    pz = Kokkos::View<R***>("flow_pz", nx, ny, nz);
    Kokkos::deep_copy(fl, R(1));
  }

  // The 19 velocities: rest, then opposite pairs (q odd, q + 1).
  KOKKOS_INLINE_FUNCTION static int opp(int q) { return q == 0 ? 0 : (q & 1) ? q + 1 : q - 1; }
  KOKKOS_INLINE_FUNCTION static void vel(int q, int &x, int &y, int &z) {
    const signed char c[19][3] = {
      {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
      {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 0, 1}, {-1, 0, -1},
      {1, 0, -1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1}};
    x = c[q][0]; y = c[q][1]; z = c[q][2];
  }
  // the velocity mirrored in z (specular reflection off the flat surface)
  KOKKOS_INLINE_FUNCTION static int mirz(int q) {
    const signed char m[19] = {0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16};
    return m[q];
  }
  KOKKOS_INLINE_FUNCTION static R wgt(int q) {
    return q == 0 ? R(1.0 / 3.0) : q < 7 ? R(1.0 / 18.0) : R(1.0 / 36.0);
  }

  // f = equilibrium of (rho - 1, u) given by fn(i, j, k, drho, u, v, w)
  template <class F>
  void init(const F fnc) {
    auto f_ = f;
    Kokkos::parallel_for("FlowInit",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        R dr = 0, u = 0, v = 0, w = 0;
        fnc(i, j, k, dr, u, v, w);
        const R usq = u * u + v * v + w * w;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R cu = cx * u + cy * v + cz * w;
          f_(i, j, k, q) = wgt(q) * (dr + (R(1) + dr) * (R(3) * cu + R(4.5) * cu * cu - R(1.5) * usq));
        }
      });
  }

  // One step over the cells [lo, hi) per axis (the pool window; the whole
  // block by default). Cells outside it are not touched.
  void step() { const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz}; step(lo, hi); }
  void step(const int lo[3], const int hi[3]) {
    auto f_ = f; auto g_ = fn; auto ty = type; auto T_ = T; auto fl_ = fl;
    auto ux_ = ux; auto uy_ = uy; auto uz_ = uz; auto rho_ = rho;
    auto px_ = px; auto py_ = py; auto pz_ = pz;
    const int nx_ = nx, ny_ = ny, nz_ = nz;
    const bool px = p.periodic[0], py = p.periodic[1], pz = p.periodic[2];
    const R tau0 = R(3.0 * p.nu + 0.5), smag2 = R(18.0 * 1.4142135623730951 * p.smag * p.smag);
    const R a0x = R(p.accel[0]), a0y = R(p.accel[1]), a0z = R(p.accel[2]);
    const R bx = R(p.buoy[0]), by = R(p.buoy[1]), bz = R(p.buoy[2]), Tref = R(p.Tref);
    const R mar = R(p.marangoni), darcy = R(p.darcy);
    Kokkos::parallel_for("FlowStep",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != FLUID) {
          for (int q = 0; q < 19; q++) g_(i, j, k, q) = R(0);
          ux_(i, j, k) = uy_(i, j, k) = uz_(i, j, k) = rho_(i, j, k) = R(0);
          px_(i, j, k) = py_(i, j, k) = pz_(i, j, k) = R(0);
          return;
        }
        // type of a cell, the block's edges included; wraps periodic axes
        auto cell = [&](int &ii, int &jj, int &kk) -> int {
          if (ii < 0 || ii >= nx_) { if (!px) return WALL; ii = (ii + nx_) % nx_; }
          if (jj < 0 || jj >= ny_) { if (!py) return WALL; jj = (jj + ny_) % ny_; }
          if (kk < 0 || kk >= nz_) { if (!pz) return kk < 0 ? WALL : GAS; kk = (kk + nz_) % nz_; }
          return ty(ii, jj, kk);
        };
        // pull: what arrives here along each velocity
        R fq[19];
        fq[0] = f_(i, j, k, 0);
        for (int q = 1; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          int ii = i - cx, jj = j - cy, kk = k - cz;
          const int t = cell(ii, jj, kk);
          if (t == FLUID) fq[q] = f_(ii, jj, kk, q);
          else if (t == GAS && cz == -1) {
            // off the surface above: it left the cell beside this one going up
            int i2 = i - cx, j2 = j - cy, k2 = k;
            fq[q] = cell(i2, j2, k2) == FLUID ? f_(i2, j2, k2, mirz(q)) : f_(i, j, k, opp(q));
          } else fq[q] = f_(i, j, k, opp(q));
        }
        R dr = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          dr += fq[q]; mx += cx * fq[q]; my += cy * fq[q]; mz += cz * fq[q];
        }
        const R r = R(1) + dr, ir = R(1) / r;

        // acceleration: uniform + buoyancy + Marangoni (surface cells)
        const R dT = T_(i, j, k) - Tref;
        R ax = a0x + bx * dT, ay = a0y + by * dT, az = a0z + bz * dT;
        if (mar != R(0)) {
          int ii = i, jj = j, kk = k + 1;
          if (cell(ii, jj, kk) == GAS) {
            // T's difference per cell, one-sided at the block's edges
            auto grad = [&](int d) -> R {
              int a[3] = {i, j, k}, b[3] = {i, j, k};
              const int nd = d == 0 ? nx_ : ny_;
              const bool per = d == 0 ? px : py;
              a[d]--; b[d]++;
              R w = R(2);
              if (a[d] < 0) { if (per) a[d] += nd; else { a[d] = 0; w -= R(1); } }
              if (b[d] >= nd) { if (per) b[d] -= nd; else { b[d] = nd - 1; w -= R(1); } }
              return w > R(0) ? (T_(b[0], b[1], b[2]) - T_(a[0], a[1], a[2])) / w : R(0);
            };
            ax += mar * grad(0); ay += mar * grad(1);
          }
        }
        // mushy drag, implicit: u = (m / rho + a / 2) / (1 + D / 2)
        const R lf = fl_(i, j, k);
        const R D = darcy != R(0) ? darcy * (R(1) - lf) * (R(1) - lf) / (lf * lf * lf + R(1e-3)) : R(0);
        const R id = R(1) / (R(1) + R(0.5) * D);
        const R u = (mx * ir + R(0.5) * ax) * id, v = (my * ir + R(0.5) * ay) * id,
                w = (mz * ir + R(0.5) * az) * id;
        ax -= D * u; ay -= D * v; az -= D * w;
        ux_(i, j, k) = R(0.5) * (u + px_(i, j, k)); px_(i, j, k) = u;
        uy_(i, j, k) = R(0.5) * (v + py_(i, j, k)); py_(i, j, k) = v;
        uz_(i, j, k) = R(0.5) * (w + pz_(i, j, k)); pz_(i, j, k) = w;
        rho_(i, j, k) = dr;

        const R usq = u * u + v * v + w * w, ua = u * ax + v * ay + w * az;
        R feq[19], src[19];
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R cu = cx * u + cy * v + cz * w, ca = cx * ax + cy * ay + cz * az;
          feq[q] = wgt(q) * (dr + r * (R(3) * cu + R(4.5) * cu * cu - R(1.5) * usq));
          src[q] = wgt(q) * r * (R(3) * (ca - ua) + R(9) * cu * ca);
        }
        R tau = tau0;
        if (smag2 != R(0)) {
          // eddy viscosity from the non-equilibrium stress
          R pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
          for (int q = 1; q < 19; q++) {
            int cx, cy, cz; vel(q, cx, cy, cz);
            const R d = fq[q] - feq[q];
            pxx += cx * cx * d; pyy += cy * cy * d; pzz += cz * cz * d;
            pxy += cx * cy * d; pxz += cx * cz * d; pyz += cy * cz * d;
          }
          const R Q = Kokkos::sqrt(pxx * pxx + pyy * pyy + pzz * pzz +
                                   R(2) * (pxy * pxy + pxz * pxz + pyz * pyz));
          tau = R(0.5) * (tau0 + Kokkos::sqrt(tau0 * tau0 + smag2 * Q * ir));
        }
        const R wp = R(1) / tau;                                        // symmetric part
        const R wm = R(1) / (R(0.1875) / (tau - R(0.5)) + R(0.5));      // antisymmetric
        g_(i, j, k, 0) = fq[0] - wp * (fq[0] - feq[0]) + (R(1) - R(0.5) * wp) * src[0];
        for (int q = 1; q < 19; q += 2) {
          const int o = q + 1;
          const R fs = R(0.5) * (fq[q] + fq[o]), fa = R(0.5) * (fq[q] - fq[o]);
          const R es = R(0.5) * (feq[q] + feq[o]), ea = R(0.5) * (feq[q] - feq[o]);
          const R ss = R(0.5) * (src[q] + src[o]), sa = R(0.5) * (src[q] - src[o]);
          const R cs = -wp * (fs - es) + (R(1) - R(0.5) * wp) * ss;
          const R ca = -wm * (fa - ea) + (R(1) - R(0.5) * wm) * sa;
          g_(i, j, k, q) = fq[q] + cs + ca;
          g_(i, j, k, o) = fq[o] + cs - ca;
        }
      });
    std::swap(f, fn);
  }

  /* --------------------------------------------------------------------
     SI driver. run(h) advances the flow by h seconds in an even number of
     steps (so u, the two-step mean, is clean), setting the lattice
     parameters from ph. The step is chosen so the fastest cell moves at
     about 0.1 cell per step, where LBM is accurate: u_cap is the speed
     mapped to 0.1. It starts at ph.u_ref, doubles when the measured
     maximum passes 0.15 and halves again below 0.03; f is rescaled when
     the step changes. Returns the number of steps, or -1 if h would take
     more than ph.nmax of them (the flow is running away).
     After run(): u in m/s is ux * ph.dx / dt, and melt_advect's cfl is the
     returned step count.
  -------------------------------------------------------------------- */
  struct Phys {
    double dx = 1.0;          // cell (m)
    double nu = 1e-6;         // kinematic viscosity (m^2/s)
    double smag = 0.17;       // Smagorinsky constant
    double dsdT_rho = 0.0;    // dsigma/dT / rho (m^3 / s^2 / K)
    double beta_g = 0.0;      // thermal expansion * g (m / s^2 / K), hot rises
    double Tref = 0.0;        // buoyancy is zero at this T
    double darcy = 0.0;       // mushy drag constant (1/s)
    double u_ref = 1.0;       // expected speed (m/s): the first u_cap
    int nmax = 256;
  } ph;
  double dt = 0.0, u_cap = 0.0, umax = 0.0;   // step (s), cap and last max speed (m/s)

  int run(double h, const int lo[3], const int hi[3]) {
    if (u_cap == 0.0) u_cap = ph.u_ref;
    while (umax > 1.5 * u_cap) u_cap *= 2.0;
    if (umax < 0.3 * u_cap && u_cap > ph.u_ref) u_cap = std::max(ph.u_ref, 0.5 * u_cap);
    const int n = 2 * std::max(1, (int) std::ceil(0.5 * h * u_cap / (0.1 * ph.dx) - 1e-9));
    if (n > ph.nmax) return -1;
    const double dtn = h / n;
    if (dt > 0.0 && dtn != dt) rescale(dtn / dt, lo, hi);
    dt = dtn;
    p.nu = ph.nu * dt / (ph.dx * ph.dx);
    p.smag = ph.smag;
    p.marangoni = ph.dsdT_rho * dt * dt / (ph.dx * ph.dx * ph.dx);
    p.buoy[0] = p.buoy[1] = 0.0;
    p.buoy[2] = ph.beta_g * dt * dt / ph.dx;
    p.Tref = ph.Tref;
    p.darcy = ph.darcy * dt;
    for (int s = 0; s < n; s++) step(lo, hi);
    auto px_ = px; auto py_ = py; auto pz_ = pz;
    R m2 = 0;
    Kokkos::parallel_reduce("FlowMax",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, R &mx) {
        const R v = px_(i, j, k) * px_(i, j, k) + py_(i, j, k) * py_(i, j, k) + pz_(i, j, k) * pz_(i, j, k);
        if (v > mx) mx = v;
      }, Kokkos::Max<R>(m2));
    umax = std::sqrt((double) m2) * ph.dx / dt;
    return n;
  }

  // The step changes by the factor s: lattice velocities scale by s, the
  // pressure deviation by s^2, the non-equilibrium part (stress) by s.
  void rescale(double s_, const int lo[3], const int hi[3]) {
    auto f_ = f; auto ty = type;
    auto ux_ = ux; auto uy_ = uy; auto uz_ = uz; auto px_ = px; auto py_ = py; auto pz_ = pz;
    const R s = R(s_);
    Kokkos::parallel_for("FlowRescale",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != FLUID) return;
        R dr = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R v = f_(i, j, k, q);
          dr += v; mx += cx * v; my += cy * v; mz += cz * v;
        }
        const R r = R(1) + dr, u = mx / r, v = my / r, w = mz / r;
        const R dr2 = dr * s * s, r2 = R(1) + dr2, u2 = u * s, v2 = v * s, w2 = w * s;
        const R usq = u * u + v * v + w * w, usq2 = usq * s * s;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R cu = cx * u + cy * v + cz * w, cu2 = cu * s;
          const R e = wgt(q) * (dr + r * (R(3) * cu + R(4.5) * cu * cu - R(1.5) * usq));
          const R e2 = wgt(q) * (dr2 + r2 * (R(3) * cu2 + R(4.5) * cu2 * cu2 - R(1.5) * usq2));
          f_(i, j, k, q) = e2 + s * (f_(i, j, k, q) - e);
        }
        ux_(i, j, k) *= s; uy_(i, j, k) *= s; uz_(i, j, k) *= s;
        px_(i, j, k) *= s; py_(i, j, k) *= s; pz_(i, j, k) *= s;
      });
  }

  // sum of rho - 1 over the fluid cells
  double mass() const {
    auto f_ = f; auto ty = type;
    double m = 0.0;
    Kokkos::parallel_reduce("FlowMass",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, double &s) {
        if (ty(i, j, k) != FLUID) return;
        for (int q = 0; q < 19; q++) s += f_(i, j, k, q);
      }, m);
    return m;
  }
};

/* ----------------------------------------------------------------------
   Carry the cell field q with the flow's u for one step of `cfl` lattice
   steps (the Courant number is cfl * u; keep it under 1/2). Three 1-D
   sweeps, x then y then z, each upwind with the van Leer limiter in
   Lax-Wendroff form, so a sharp front stays sharp and no sweep makes a new
   maximum or minimum. Flux crosses only the faces between two FLUID cells
   (face velocity = the mean of the two); other cells keep their q.

   Each sweep is q - d(u q) + q d(u): the last term removes what the
   velocity's own divergence would add, so a uniform q stays uniform
   exactly. The price: where u is not divergence-free (LBM's is, to
   O(Mach^2)) the sum of q is conserved only to that order.
   ponytail: fixed sweep order; alternate it if a directional bias shows.

   tmp is scratch of q's shape. [lo, hi) must hold every FLUID cell and
   two cells around them (or reach the block's edge): the limiter reads
   that far, and tmp is only valid inside the window.
------------------------------------------------------------------------- */

template <class R>
void melt_advect(const MeltFlow<R> &m, Kokkos::View<R***> q, Kokkos::View<R***> tmp,
                 double cfl, const int lo[3], const int hi[3])
{
  auto ty = m.type;
  const int n[3] = {m.nx, m.ny, m.nz};
  const R c = R(cfl);
  for (int d = 0; d < 4; d++) {
    // sweeps 0..2 along x, y, z (q -> tmp -> q -> tmp), then tmp -> q
    auto src = (d & 1) ? tmp : q, dst = (d & 1) ? q : tmp;
    if (d == 3) {
      Kokkos::parallel_for("FlowAdvectCopy",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { dst(i, j, k) = src(i, j, k); });
      break;
    }
    auto u = d == 0 ? m.ux : d == 1 ? m.uy : m.uz;
    const int nd = n[d];
    const bool per = m.p.periodic[d];
    Kokkos::parallel_for("FlowAdvect",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const R q0 = src(i, j, k);
        if (ty(i, j, k) != MeltFlow<R>::FLUID) { dst(i, j, k) = q0; return; }
        const int x[3] = {i, j, k};
        // the cell s along the sweep axis: index in range, or -1 (none)
        auto at = [&](int s, int &ii, int &jj, int &kk) -> bool {
          int y[3] = {x[0], x[1], x[2]};
          y[d] += s;
          if (y[d] < 0 || y[d] >= nd) { if (!per) return false; y[d] = (y[d] + nd) % nd; }
          ii = y[0]; jj = y[1]; kk = y[2];
          return true;
        };
        // flux through the face between cells s and s + 1 (s = -1 or 0),
        // and that face's velocity; zero unless both are fluid
        auto face = [&](int s, R &uf) -> R {
          int ia, ja, ka, ib, jb, kb;
          uf = R(0);
          if (!at(s, ia, ja, ka) || !at(s + 1, ib, jb, kb)) return R(0);
          if (ty(ia, ja, ka) != MeltFlow<R>::FLUID || ty(ib, jb, kb) != MeltFlow<R>::FLUID) return R(0);
          uf = c * R(0.5) * (u(ia, ja, ka) + u(ib, jb, kb));
          const bool up = uf > R(0);
          const R qu = up ? src(ia, ja, ka) : src(ib, jb, kb);
          const R qd = up ? src(ib, jb, kb) : src(ia, ja, ka);
          int iu, ju, ku;
          R phi = R(0);
          if (at(up ? s - 1 : s + 2, iu, ju, ku) && qd != qu) {
            const R r = (qu - src(iu, ju, ku)) / (qd - qu);
            phi = r > R(0) ? R(2) * r / (R(1) + r) : R(0);
          }
          return uf * (qu + R(0.5) * phi * (R(1) - Kokkos::abs(uf)) * (qd - qu));
        };
        R ul, uh;
        const R fl = face(-1, ul), fh = face(0, uh);
        dst(i, j, k) = q0 - (fh - fl) + q0 * (uh - ul);
      });
  }
}

}

#endif
