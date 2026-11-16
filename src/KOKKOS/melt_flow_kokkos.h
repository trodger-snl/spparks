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
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Melt-pool flow: D3Q19 lattice Boltzmann on a uniform block of cells.
   It knows nothing of SPPARKS or MPI: the caller fills type, T and fl
   and reads u, so tools/melt_flow_check.cpp drives it against analytical
   solutions. R is the float type of the distributions and the math.

   Several ranks: each rank's block carries halo cells of its neighbours
   and steps only its own cells (the window). The caller keeps the halo
   current: type and T before run(), f through the `halo` callback (after
   every step and rescale), flux before melt_advect, and the carried
   fields through melt_advect's own callback.

   Lattice units throughout (cell 1, step 1, density 1): the caller scales.
     nu        viscosity * dt / dx^2
     accel     uniform acceleration * dt^2 / dx
     buoy      a = buoy * (T - Tref): e.g. z = beta g dt^2 / dx (hot rises)
     marangoni a_xy = marangoni * (T's difference per cell along x, y), on
               surface cells only: dsigma/dT * dt^2 / (rho dx^3)
     darcy     a = -darcy (1 - fl)^2 / (fl^3 + 1e-3) u (mushy drag), * dt

   props (set before resize()): the liquid's properties vary from cell to
   cell, and the caller fills three more fields:
     sig   surface tension. The Marangoni acceleration is marangoni * (sig's
           difference per cell) / (1 + dens), so it holds the thermal and
           the solutal part; the difference is taken between FLUID cells
           only (one-sided next to a wall: the solid's composition is not
           the liquid's). marangoni = dt^2 / (rho0 dx^3) for sig in N/m.
     dens  rho / rho0 - 1. Buoyancy is a = -buoy * dens (buoy z = g dt^2 /
           dx), instead of buoy * (T - Tref). The inertia stays that of
           rho0 (Boussinesq).
     visc  the cell's viscosity is nu * visc.

   Collision: two relaxation times with Lambda = 3/16 (a bounce-back wall
   then sits half-way between the cells whatever nu is), Guo forcing, and
   an optional Smagorinsky eddy viscosity (smag = its constant, 0 = off).
   Distributions are stored as f - w, so single precision keeps the small
   velocities of a low-Mach flow.

   Cells: FLUID; WALL (half-way bounce-back; also every cell beyond the
   block in x, y and below it in z, unless that axis is periodic); GAS
   above a fluid cell is a free surface, flat and free-slip (specular
   reflection), and so is the block's top. GAS beside a fluid cell acts as
   a wall; a reflection needs GAS above both cells it joins, so a step in
   the surface neither makes nor loses mass. Two fluid cells that touch
   only along an edge (both cells between them are not fluid) do not
   exchange: that link bounces back. Surface cells (GAS or the block's top
   above) take the Marangoni force. Cells that are not FLUID hold f = w (rest), so a cell that melts
   starts at rest.

   ux, uy, uz are the mean of the last two steps. Bounce-back walls conserve
   a staggered momentum, sum of (-1)^(x+t) j_x per axis, that no collision
   damps: any abrupt change of force leaves a checkerboard that flips sign
   every step (5e-5 for a force of 1e-4 switched on at rest). The two-step
   mean cancels it; callers should take an even number of steps between
   reads. The surface force acts on the whole top cell, so that cell's
   velocity lags the true surface profile by about tau / (8 mu) (first
   order in the cell size); the cells below are second order.

   flux(i, j, k, 2 d + s) is the mass (as f - w) that left the cell through
   its face on axis d, side s (0 low, 1 high), as this cell counted it,
   summed over the steps since the last step(lo, hi, true). A population
   on a diagonal link crosses two faces, by way of whichever of the two
   cells between is fluid (half each way if both are); the cell it leaves
   counts the first face, the cell it reaches the second, and an axis
   population is counted where it arrives. So the mass through the face
   between a and b = a + e_d is flux(a, 2d+1) - flux(b, 2d), and summed
   over a cell's six faces that is exactly its loss of rho: melt_advect
   carries scalars with these.

   A window passed to step() must hold every FLUID cell and every cell that
   was FLUID at either of the last two steps, so freezing cells are reset
   in both copies of f.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   A cell field held over a window of the block only: v covers the cells
   [o, o + extent) and is read by block index. Made from a plain view it is
   the whole block.
------------------------------------------------------------------------- */

template <class V>
struct FlowWin {
  V v;
  int o0 = 0, o1 = 0, o2 = 0;
  FlowWin() = default;
  FlowWin(const V &v_) : v(v_) {}
  template <class... Q>
  KOKKOS_INLINE_FUNCTION auto &operator()(int i, int j, int k, Q... q) const {
    return v(i - o0, j - o1, k - o2, q...);
  }
  // hold the cells [lo, hi) instead: a cell in both windows keeps its
  // value, a new one is 0
  void move(const int lo[3], const int hi[3]) {
    int n[3], a[3], b[3];
    const int o[3] = {o0, o1, o2};
    bool both = true;
    for (int d = 0; d < 3; d++) {
      n[d] = hi[d] > lo[d] ? hi[d] - lo[d] : 0;
      a[d] = lo[d] > o[d] ? lo[d] : o[d];
      b[d] = std::min(lo[d] + n[d], o[d] + (int) v.extent(d));
      both = both && b[d] > a[d];
    }
    FlowWin old = *this, nw;
    const int nc = V::rank() == 4 ? (int) v.extent(3) : 1;
    if constexpr (V::rank() == 4) nw.v = V(v.label(), n[0], n[1], n[2], nc);
    else nw.v = V(v.label(), n[0], n[1], n[2]);
    nw.o0 = lo[0]; nw.o1 = lo[1]; nw.o2 = lo[2];
    if (both)
      Kokkos::parallel_for("FlowWinMove",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({a[0], a[1], a[2]}, {b[0], b[1], b[2]}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          if constexpr (V::rank() == 4) { for (int q = 0; q < nc; q++) nw(i, j, k, q) = old(i, j, k, q); }
          else nw(i, j, k) = old(i, j, k);
        });
    *this = nw;
  }
};

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
  using W3 = FlowWin<Kokkos::View<R***>>;
  using W4 = FlowWin<Kokkos::View<R****>>;
  W4 f, fn;                                  // (x, y, z, q): f - w
  FlowWin<Kokkos::View<unsigned char***>> type;
  W3 T, fl;                                  // temperature, liquid fraction
  W3 ux, uy, uz, rho;                        // output: two-step mean u, rho - 1
  W3 px, py, pz;                             // the last step's u
  W4 flux;                                   // (x, y, z, face): mass out, see above
  bool props = false;                        // per-cell liquid properties (see above)
  // lazy: step() clears a non-FLUID cell only where the caller set fl < 0
  // (a cell that was FLUID at the last call; the clearing needs two steps).
  // Else every non-FLUID cell of the window is cleared at every step.
  bool lazy = false;
  W3 sig, dens, visc;
  int wlo[3] = {0, 0, 0}, whi[3] = {0, 0, 0};   // the cells the fields hold

  // A block of nx_ x ny_ x nz_ cells. all: the fields hold every cell;
  // else none yet, and window() says which.
  void resize(int nx_, int ny_, int nz_, bool all = true) {
    nx = nx_; ny = ny_; nz = nz_;
    const int a = all ? nx : 0, b = all ? ny : 0, c = all ? nz : 0;
    f = Kokkos::View<R****>("flow_f", a, b, c, 19);
    fn = Kokkos::View<R****>("flow_fn", a, b, c, 19);
    type = Kokkos::View<unsigned char***>("flow_type", a, b, c);
    T = Kokkos::View<R***>("flow_T", a, b, c);
    fl = Kokkos::View<R***>("flow_fl", a, b, c);
    ux = Kokkos::View<R***>("flow_ux", a, b, c);
    uy = Kokkos::View<R***>("flow_uy", a, b, c);
    uz = Kokkos::View<R***>("flow_uz", a, b, c);
    rho = Kokkos::View<R***>("flow_rho", a, b, c);
    px = Kokkos::View<R***>("flow_px", a, b, c);
    py = Kokkos::View<R***>("flow_py", a, b, c);
    pz = Kokkos::View<R***>("flow_pz", a, b, c);
    flux = Kokkos::View<R****>("flow_flux", a, b, c, 6);
    if (props) {
      sig = Kokkos::View<R***>("flow_sig", a, b, c);
      dens = Kokkos::View<R***>("flow_dens", a, b, c);
      visc = Kokkos::View<R***>("flow_visc", a, b, c);
      Kokkos::deep_copy(visc.v, R(1));
    }
    Kokkos::deep_copy(fl.v, R(1));
    for (int d = 0; d < 3; d++) wlo[d] = 0;
    whi[0] = a; whi[1] = b; whi[2] = c;
  }

  // The fields now hold the cells [lo, hi) only (see FlowWin::move). Every
  // cell a kernel reads must be inside: the windows passed to step() and
  // melt_advect() and, around FLUID cells, two more cells (or the block's
  // edge). A cell outside has no flow state: it must have been non-FLUID
  // for the last two steps, as one outside step()'s window.
  void window(const int lo[3], const int hi[3]) {
    f.move(lo, hi); fn.move(lo, hi); type.move(lo, hi); T.move(lo, hi); fl.move(lo, hi);
    ux.move(lo, hi); uy.move(lo, hi); uz.move(lo, hi); rho.move(lo, hi);
    px.move(lo, hi); py.move(lo, hi); pz.move(lo, hi); flux.move(lo, hi);
    if (props) { sig.move(lo, hi); dens.move(lo, hi); visc.move(lo, hi); }
    for (int d = 0; d < 3; d++) { wlo[d] = lo[d]; whi[d] = hi[d] > lo[d] ? hi[d] : lo[d]; }
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
  // first: flux starts again from this step (else it adds to it).
  void step() { const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz}; step(lo, hi); }
  void step(const int lo[3], const int hi[3], bool first = true) {
    auto f_ = f; auto g_ = fn; auto ty = type; auto T_ = T; auto fl_ = fl; auto flux_ = flux;
    auto ux_ = ux; auto uy_ = uy; auto uz_ = uz; auto rho_ = rho;
    auto px_ = px; auto py_ = py; auto pz_ = pz;
    const int nx_ = nx, ny_ = ny, nz_ = nz;
    const bool px = p.periodic[0], py = p.periodic[1], pz = p.periodic[2];
    const R tau00 = R(3.0 * p.nu + 0.5), smag2 = R(18.0 * 1.4142135623730951 * p.smag * p.smag);
    const R a0x = R(p.accel[0]), a0y = R(p.accel[1]), a0z = R(p.accel[2]);
    const R bx = R(p.buoy[0]), by = R(p.buoy[1]), bz = R(p.buoy[2]), Tref = R(p.Tref);
    const R mar = R(p.marangoni), darcy = R(p.darcy);
    // props: the Marangoni force follows sig, not T
    const bool hp = props, lazy_ = lazy;
    auto S_ = hp ? sig : T; auto dn_ = dens; auto vs_ = visc;
    const R nu3 = R(3.0 * p.nu);
    Kokkos::parallel_for("FlowStep",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != FLUID) {
          if (lazy_ && fl_(i, j, k) >= R(0)) return;
          for (int q = 0; q < 19; q++) g_(i, j, k, q) = R(0);
          ux_(i, j, k) = uy_(i, j, k) = uz_(i, j, k) = rho_(i, j, k) = R(0);
          px_(i, j, k) = py_(i, j, k) = pz_(i, j, k) = R(0);
          for (int g = 0; g < 6; g++) flux_(i, j, k, g) = R(0);
          return;
        }
        // type of a cell, the block's edges included; wraps periodic axes
        auto cell = [&](int &ii, int &jj, int &kk) -> int {
          if (ii < 0 || ii >= nx_) { if (!px) return WALL; ii = (ii + nx_) % nx_; }
          if (jj < 0 || jj >= ny_) { if (!py) return WALL; jj = (jj + ny_) % ny_; }
          if (kk < 0 || kk >= nz_) { if (!pz) return kk < 0 ? WALL : GAS; kk = (kk + nz_) % nz_; }
          return ty(ii, jj, kk);
        };
        // pull: what arrives here along each velocity. tq[q] is the type
        // of the cell it comes from (this cell - c_q); G the mass leaving
        // by each face (see flux above).
        R fq[19], G[6] = {0, 0, 0, 0, 0, 0};
        int tq[19];
        fq[0] = f_(i, j, k, 0); tq[0] = FLUID;
        for (int q = 1; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          int ii = i - cx, jj = j - cy, kk = k - cz;
          tq[q] = cell(ii, jj, kk);
          if (tq[q] == FLUID) fq[q] = f_(ii, jj, kk, q);
        }
        // the axis velocities: q = 1 + 2 A + (sign < 0); face 2 A + (side > 0)
        for (int q = 1; q < 7; q++) {
          if (tq[q] == FLUID) G[q - 1] -= fq[q];     // came in by the face toward its source
          else fq[q] = f_(i, j, k, opp(q));          // wall, or the surface above
        }
        for (int q = 7; q < 19; q++) {
          int c[3]; vel(q, c[0], c[1], c[2]);
          const int A = c[0] != 0 ? 0 : 1, B = c[2] != 0 ? 2 : 1, sA = c[A], sB = c[B];
          const int o = opp(q);
          // the two cells between this one and the source: this - sA e_A, this - sB e_B
          const bool inA = tq[1 + 2 * A + (sA < 0)] == FLUID, inB = tq[1 + 2 * B + (sB < 0)] == FLUID;
          if (tq[q] == FLUID && (inA || inB)) {
            const R wv = (inA && inB ? R(0.5) : R(1)) * fq[q];
            if (inA) G[2 * A + (sA < 0)] -= wv;
            if (inB) G[2 * B + (sB < 0)] -= wv;
          } else if (tq[q] == GAS && sB == -1 && B == 2 && tq[6] == GAS && inA) {
            // off the flat surface above this cell and the one beside it,
            // which it left going up
            int i2 = i - c[0], j2 = j - c[1], k2 = k;
            cell(i2, j2, k2);
            fq[q] = f_(i2, j2, k2, mirz(q));
            G[2 * A + (sA < 0)] -= fq[q];
          } else fq[q] = f_(i, j, k, o);
          // what leaves along q, first face: by way of this + sA e_A or this + sB e_B
          if (tq[o] == FLUID) {
            const bool outA = tq[1 + 2 * A + (sA > 0)] == FLUID, outB = tq[1 + 2 * B + (sB > 0)] == FLUID;
            if (outA || outB) {
              const R wv = (outA && outB ? R(0.5) : R(1)) * f_(i, j, k, q);
              if (outA) G[2 * A + (sA > 0)] += wv;
              if (outB) G[2 * B + (sB > 0)] += wv;
            }
          }
        }
        for (int g = 0; g < 6; g++) flux_(i, j, k, g) = first ? G[g] : flux_(i, j, k, g) + G[g];
        R dr = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          dr += fq[q]; mx += cx * fq[q]; my += cy * fq[q]; mz += cz * fq[q];
        }
        const R r = R(1) + dr, ir = R(1) / r;

        // acceleration: uniform + buoyancy + Marangoni (surface cells)
        const R dT = hp ? -dn_(i, j, k) : T_(i, j, k) - Tref;
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
              // one-sided where the neighbour holds no surface value: sigma
              // exists in FLUID cells only; T exists in walls too, but a gas
              // cell's T is not the surface's (it gave a false edge gradient)
              const unsigned char ta = ty(a[0], a[1], a[2]), tb = ty(b[0], b[1], b[2]);
              if (hp ? ta != FLUID : ta == GAS) { a[d] = d == 0 ? i : j; w -= R(1); }
              if (hp ? tb != FLUID : tb == GAS) { b[d] = d == 0 ? i : j; w -= R(1); }
              return w > R(0) ? (S_(b[0], b[1], b[2]) - S_(a[0], a[1], a[2])) / w : R(0);
            };
            const R ms = hp ? mar / (R(1) + dn_(i, j, k)) : mar;
            ax += ms * grad(0); ay += ms * grad(1);
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
        const R tau0 = hp ? nu3 * vs_(i, j, k) + R(0.5) : tau00;
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

  // Is the stored value NaN or infinite? Read as bits: the builds are
  // -ffast-math, where a floating-point test for NaN is compiled away.
  KOKKOS_INLINE_FUNCTION static bool bad(const R &x) {
    if (sizeof(R) == 4) { unsigned int u; memcpy(&u, &x, 4); return (u & 0x7f800000u) == 0x7f800000u; }
    unsigned long long u; memcpy(&u, &x, 8);
    return (u & 0x7ff0000000000000ull) == 0x7ff0000000000000ull;
  }

  // the largest |u|^2 of the last step (lattice units); 1e12 if any is not
  // finite
  R max_u2(const int lo[3], const int hi[3]) const {
    auto px_ = px; auto py_ = py; auto pz_ = pz;
    R m2 = 0;
    Kokkos::parallel_reduce("FlowMax",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, R &mx) {
        R v = px_(i, j, k) * px_(i, j, k) + py_(i, j, k) * py_(i, j, k) + pz_(i, j, k) * pz_(i, j, k);
        if (bad(px_(i, j, k)) || bad(py_(i, j, k)) || bad(pz_(i, j, k))) v = R(1e12);
        if (v > mx) mx = v;
      }, Kokkos::Max<R>(m2));
    return m2;
  }

  /* --------------------------------------------------------------------
     SI driver. run(h) advances the flow by h seconds in an even number of
     steps (so u, the two-step mean, is clean), setting the lattice
     parameters from ph. The step is chosen so the fastest cell moves at
     about 0.1 cell per step, where LBM is accurate: u_cap is the speed
     mapped to 0.1. It starts at ph.u_ref, doubles when the measured
     maximum passes 0.15 and halves again below 0.03; f is rescaled when
     the step changes. Returns the number of steps, or -1 if h would take
     more than ph.nmax of them or the measured speed is past half a cell
     per step or not finite (the flow is running away).
     After run(): u in m/s is ux * ph.dx / dt; melt_advect then carries the
     scalars with the fluxes of these steps.
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
  std::function<void()> halo;   // several ranks: brings the halo cells' f up to date
  long ncall = 0;

  int run(double h, const int lo[3], const int hi[3]) {
    if (u_cap == 0.0) u_cap = ph.u_ref;
    while (umax > 1.5 * u_cap) u_cap *= 2.0;
    if (umax < 0.3 * u_cap && u_cap > ph.u_ref) u_cap = std::max(ph.u_ref, 0.5 * u_cap);
    const int n = 2 * std::max(1, (int) std::ceil(0.5 * h * u_cap / (0.1 * ph.dx) - 1e-9));
    if (n > ph.nmax) return -1;
    const double dtn = h / n;
    if (dt > 0.0 && dtn != dt) { rescale(dtn / dt, lo, hi); if (halo) halo(); }
    dt = dtn;
    p.nu = ph.nu * dt / (ph.dx * ph.dx);
    p.smag = ph.smag;
    p.marangoni = ph.dsdT_rho * dt * dt / (ph.dx * ph.dx * ph.dx);
    p.buoy[0] = p.buoy[1] = 0.0;
    p.buoy[2] = ph.beta_g * dt * dt / ph.dx;
    p.Tref = ph.Tref;
    p.darcy = ph.darcy * dt;
    // A fence per step: inside spparks on HIP (gfx1151) host memory grows
    // during long runs; fenced plain launches grow least (measured, 3-layer
    // pillar: 1.1 GB; unfenced 1.8; every step a reduction 2.0; no flow 0.6).
    for (int s = 0; s < n; s++) { step(lo, hi, s == 0); Kokkos::fence(); if (halo) halo(); }
    // the speed is measured every 16th call (a reduction; it can at most
    // double in that time at the accelerations a pool has). Several ranks:
    // the caller takes the maximum of umax over them after every run().
    if (ncall++ % 16 == 0) {
      const double ul = std::sqrt((double) max_u2(lo, hi));
      umax = ul * ph.dx / dt;
      // past half a cell per step the flow has blown up (or is not finite):
      // no step size will bring it back
      if (ul > 0.5) return -1;
    }
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

  // flux as a uniform flow would leave it: (u, v, w) cells' mass per face
  // (for driving melt_advect without stepping)
  void set_flux(double u, double v, double w) {
    auto fl_ = flux;
    const R h[3] = {R(0.5 * u), R(0.5 * v), R(0.5 * w)};
    Kokkos::parallel_for("FlowSetFlux",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        for (int g = 0; g < 6; g++) fl_(i, j, k, g) = (g & 1) ? h[g / 2] : -h[g / 2];
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
   Carry the cell field q (an amount per unit mass: composition, specific
   enthalpy) with the flow's own mass fluxes, m.flux, as summed since the
   last step(lo, hi, true): call it once after each run(). Three 1-D
   sweeps, x then y then z, each upwind with the van Leer limiter in
   Lax-Wendroff form, so a sharp front stays sharp and no sweep makes a new
   maximum or minimum. Keep the mass through a face under half a cell's.

   With F the mass through a face and qf the limited face value, a sweep is
   q - (F_hi (qf_hi - q) - F_lo (qf_lo - q)) / m, m the cell's mass after
   the sweep (after the last one, 1 + rho). That is (m q)' = m q - d(F qf)
   with m' = m - d(F): the sum of (1 + rho) q over the fluid cells is
   conserved to round-off, and a uniform q stays uniform exactly, though
   the flow is slightly compressible. (Before 2026-11 the fluxes were the
   mean u of the two cells and m was taken as 1: the sum then drifted, 1.6%
   of the fed composition over a 4-layer DED wall.)
   ponytail: fixed sweep order; alternate it if a directional bias shows.

   tmp is scratch of q's shape. [lo, hi) must hold every FLUID cell and
   two cells around them (or reach the block's edge): the limiter reads
   that far, and tmp is only valid inside the window.
   Several ranks: halo(d, a, b) is called before the sweep along d and must
   bring the halo cells of a (and b, if not empty) up to date two cells
   deep along d; m.flux and m.type must be current in the halo already.
------------------------------------------------------------------------- */

// The carried fields are FlowWin (a plain view converts: the whole block);
// R is taken from m alone.
template <class R>
using FlowField = std::type_identity_t<FlowWin<Kokkos::View<R***>>>;

// Two fields at once (q2/tmp2 may be empty views: one field). With
// copy_back false the result is left in tmp/tmp2 (inside the window) and
// q/q2 hold the state after the y sweep: the caller reads tmp and saves a
// launch.
template <class R>
void melt_advect(const MeltFlow<R> &m, FlowField<R> q, FlowField<R> tmp,
                 FlowField<R> q2, FlowField<R> tmp2,
                 const int lo[3], const int hi[3], bool copy_back,
                 const std::function<void(int, FlowField<R>, FlowField<R>)> &halo = {})
{
  const bool two = q2.v.extent(0) > 0;
  auto ty = m.type; auto G = m.flux; auto rho = m.rho;
  const int n0 = m.nx, n1 = m.ny, n2 = m.nz;
  const bool p0 = m.p.periodic[0], p1 = m.p.periodic[1], p2 = m.p.periodic[2];
  for (int d = 0; d < (copy_back ? 4 : 3); d++) {
    // sweeps 0..2 along x, y, z (q -> tmp -> q -> tmp), then tmp -> q
    auto src = (d & 1) ? tmp : q, dst = (d & 1) ? q : tmp;
    auto src2 = (d & 1) ? tmp2 : q2, dst2 = (d & 1) ? q2 : tmp2;
    if (d == 3) {
      Kokkos::parallel_for("FlowAdvectCopy",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          dst(i, j, k) = src(i, j, k);
          if (two) dst2(i, j, k) = src2(i, j, k);
        });
      break;
    }
    if (halo) halo(d, src, src2);
    Kokkos::parallel_for("FlowAdvect",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != MeltFlow<R>::FLUID) {
          dst(i, j, k) = src(i, j, k);
          if (two) dst2(i, j, k) = src2(i, j, k);
          return;
        }
        const int x[3] = {i, j, k}, nn[3] = {n0, n1, n2};
        const bool per[3] = {p0, p1, p2};
        // the cell s along axis e: its index if it is there and FLUID
        auto at = [&](int e, int s, int &ii, int &jj, int &kk) -> bool {
          int y[3] = {x[0], x[1], x[2]};
          y[e] += s;
          if (y[e] < 0 || y[e] >= nn[e]) { if (!per[e]) return false; y[e] = (y[e] + nn[e]) % nn[e]; }
          ii = y[0]; jj = y[1]; kk = y[2];
          return ty(ii, jj, kk) == MeltFlow<R>::FLUID;
        };
        // mass through the face between cells s and s + 1 along e (s = -1, 0)
        auto mass = [&](int e, int s) -> R {
          int ia, ja, ka, ib, jb, kb;
          if (!at(e, s, ia, ja, ka) || !at(e, s + 1, ib, jb, kb)) return R(0);
          return G(ia, ja, ka, 2 * e + 1) - G(ib, jb, kb, 2 * e);
        };
        // that face's value of src along d, less this cell's: limited upwind
        auto face = [&](const typename MeltFlow<R>::W3 &src, int s, R uf) -> R {
          if (uf == R(0)) return R(0);
          int ia, ja, ka, ib, jb, kb;
          at(d, s, ia, ja, ka); at(d, s + 1, ib, jb, kb);
          const bool up = uf > R(0);
          const R qu = up ? src(ia, ja, ka) : src(ib, jb, kb);
          const R qd = up ? src(ib, jb, kb) : src(ia, ja, ka);
          int iu, ju, ku;
          // van Leer: phi (qd - qu) with phi = 2 r / (1 + r), r = a / b, as
          // 2 a b / (a + b). (Not through r: with b tiny, as in a well-mixed
          // pool in single precision, r overflows and phi is inf / inf.)
          R lim = R(0);
          if (at(d, up ? s - 1 : s + 2, iu, ju, ku)) {
            const R a = qu - src(iu, ju, ku), b = qd - qu;
            if (a * b > R(0)) lim = R(2) * a * b / (a + b);
          }
          return qu + R(0.5) * (R(1) - Kokkos::abs(uf)) * lim - src(i, j, k);
        };
        // the cell's mass after this sweep: the later sweeps' fluxes still in it
        R ms = R(1) + rho(i, j, k);
        for (int e = d + 1; e < 3; e++) ms += mass(e, 0) - mass(e, -1);
        const R ul = mass(d, -1), uh = mass(d, 0), im = R(1) / ms;
        dst(i, j, k) = src(i, j, k) - (uh * face(src, 0, uh) - ul * face(src, -1, ul)) * im;
        if (two) dst2(i, j, k) = src2(i, j, k) - (uh * face(src2, 0, uh) - ul * face(src2, -1, ul)) * im;
      });
  }
}

template <class R>
void melt_advect(const MeltFlow<R> &m, FlowField<R> q, FlowField<R> tmp,
                 const int lo[3], const int hi[3])
{
  melt_advect(m, q, tmp, FlowField<R>(), FlowField<R>(), lo, hi, true);
}

}

#endif
