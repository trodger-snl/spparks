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

// HIP launches a kernel whose lambda is 512 bytes or more (most here: they
// capture many Views) through constant memory, an async copy and an event
// per launch, and the ROCm runtime (7.1.1, gfx1151) keeps host memory for
// each: a free-surface track grew 12 MB/s. A light-weight policy passes the
// lambda as a kernel argument (up to 4 KB); other backends ignore the hint.
template <int N>
using LightMD = Kokkos::MDRangePolicy<Kokkos::Rank<N>, Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
using LightRange = Kokkos::RangePolicy<Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;

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

   Free surface (fs): the surface moves. Mass is tracked per cell (Koerner
   et al. 2005, Thuerey 2007): an INTERFACE cell holds distributions like a
   FLUID cell and a mass between 0 and its density; fill = mass / density.
   A layer of INTERFACE cells always separates FLUID from GAS. A link from
   GAS is not reflected but reconstructed from the gas's equilibrium (the
   gas is a pressure, not a fluid), and a link between liquid cells moves
   mass: all of the populations' difference, or the mean fill's part of it
   between two INTERFACE cells. Those are the fluxes of `flux`, so
   melt_advect carries scalars through partial cells with the same mass.
   After each step convert() turns a cell whose mass passed its density
   into FLUID and one whose mass fell below zero into GAS, makes INTERFACE
   cells of their gas and fluid neighbours, and hands the excess to the
   interface cells around. The caller calls, per step: step(), melt_advect
   (if it carries fields), convert(). After setting types by hand call
   fs_repair() once.
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
        LightMD<3>({a[0], a[1], a[2]}, {b[0], b[1], b[2]}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // (named before the if constexpr: nvcc will not capture there first)
          const FlowWin &dst = nw, &src = old;
          const int nq = nc;
          if constexpr (V::rank() == 4) { for (int q = 0; q < nq; q++) dst(i, j, k, q) = src(i, j, k, q); }
          else dst(i, j, k) = src(i, j, k);
        });
    *this = nw;
  }
};

template <class R>
struct MeltFlow {
  enum Type : unsigned char { FLUID = 0, WALL = 1, GAS = 2, INTERFACE = 3 };
  // liquid: a cell that holds distributions (INTERFACE only with fs)
  KOKKOS_INLINE_FUNCTION static bool liq(int t) { return t == FLUID || t == INTERFACE; }

  struct Params {
    double nu = 1.0 / 6.0;
    double smag = 0.0;
    double accel[3] = {0, 0, 0};
    double buoy[3] = {0, 0, 0};
    double Tref = 0.0;
    double marangoni = 0.0;
    double darcy = 0.0;
    // fs: the surface tension itself, sigma * dt^2 / (rho dx^3), at T = Tref
    // (it then follows T by `marangoni`). props: the N/m to add to `sig`,
    // which is stored less its first value.
    double sigma = 0.0;
    // stab: dt^2 / dx^3, the factor from sigma / rho (m^3/s^2) to lattice units
    double sigconv = 0.0;
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
  // fs, one liquid (no props): sigma(T) / rho (m^3/s^2) on the uniform grid
  // stab_T0 + m / stab_idT, linear beyond it (flow lbm ... sigma_table):
  // capillary pressure and Marangoni stress follow the surface cell's T.
  // Empty: sigma = p.sigma + p.marangoni (T - Tref).
  Kokkos::View<R*> stab;
  R stab_T0 = 0, stab_idT = 1;
  // fs (set before resize()): the surface moves; see "Free surface" below
  bool fs = false;
  W3 cmass, eps;                             // interface cells' mass; fill (1 fluid, 0 gas)
  FlowWin<Kokkos::View<unsigned char***>> mark, type2;   // convert()'s scratch
  W4 xs;                                     // convert()'s scratch: a marked cell's excess, q, q2
  double fs_kappa = 1e-3;                    // a cell converts when its mass passes the end by this
  // run(): dt <= this * sqrt(rho dx^3 / sigma), i.e. lattice sigma <= its
  // square. Measured on the pool of check 21 at rest: currents of 4e-5 m/s at
  // lattice sigma 0.021, 2e-3 at 0.032, 1.5 at 0.072.
  // In a track (316L, 367 W, 1.5 mm spot) 0.18 with the smoothing below let
  // one-column pits grow until the pool broke up at 150-200 ms; 0.08 with
  // no smoothing kept the surface within 1 cell of flat. A wider, hotter
  // pool (Ti-6Al-4V, 400 W, 1.5 mm spot) still grew a two-cell checkerboard
  // on the surface at 0.08 (spikes at 2-3 m/s by 110-250 ms); 0.04 was calm
  // to 200 ms, and half smoothing at 0.08 was not.
  double fs_cap = 0.04;
  // surface(): contact angle (degrees) where the liquid meets a lower, dry
  // column: that column counts as the liquid's height less tan(angle) cells,
  // not below its own (0: mirrored, the liquid wets it; 90: at its own
  // height). As a share of the height difference (the earlier fs_edge) any
  // value held a tall bead's edge too stiffly: at 0.5 a 316L bead 16 cells
  // high balled and the pool broke away from it. 316L powder track (367 W,
  // 1.5 mm spot, 450 ms): 0 spread the bead flat and wide over gas pockets
  // under its flanks; 45 humped (the pool broke away, 600 cells of
  // spatter); 20 gave a dense, round, continuous bead.
  double fs_angle = 20.0;
  // surface(): dry solid whose top cell is colder than this (K) pins the
  // contact line instead (90 degrees: the surface is held back). Molten
  // metal reaching cold solid freezes at the contact line and stops there
  // (Schiaffino & Sonin 1997); fs_angle then applies to hot solid only.
  double fs_pin_T = -1e30;
  // A floor on the relaxation time of interface cells. At a liquid metal's
  // viscosity (tau about 0.51) they are unstable under a Marangoni force:
  // the pool of check 21 breaks up at a third of its forcing with no floor,
  // is rough at 0.6 and holds at 1. It makes the one-cell surface layer
  // viscous (nu = 1/6 lattice at 1), so the surface is dragged along with
  // the cell under it: a known bias on the surface speed.
  double fs_tau = 1.0;
  // The unsmoothed part of the height the forces see. Smoothing lets a
  // larger step be stable but takes the restoring force from cell-sized
  // pits, which then never heal: off (1) by default.
  double fs_raw = 1.0;
  // (set before resize()) The surface's curvature and Marangoni force from
  // local height functions along each interface cell's normal (Popinet
  // 2009), not one height per column: see surface_hf(). fs_raw is not used.
  bool fs_hf = false;
  // fs_hf, per cell: (0) the gas density less 1 an INTERFACE cell's gas
  // links see; (1-3) the surface force as an acceleration (an INTERFACE
  // cell's frame column's; a FLUID cell's, gathered from the columns it is
  // the base of); (4) an INTERFACE cell's curvature. hfr: an INTERFACE
  // cell's method << 3 | frame, frame = 2 d + (s > 0) (heights along
  // s e_d), method HF_* below (heights in the dominant frame, in another,
  // a paraboloid fit, the mean of its neighbours, none).
  W4 hfs;
  FlowWin<Kokkos::View<unsigned char***>> hfr;
  enum { HF_DOM = 1, HF_OTHER = 2, HF_FIT = 3, HF_AVG = 4, HF_NONE = 5 };

  // per column (x, y), from surface(): the metal's height, the gas density
  // the surface sees (less 1), the surface force as an acceleration of the
  // cells kb..kt, and what the caller may set: the gas pressure on the
  // surface (lattice units; e.g. recoil), less the ambient
  Kokkos::View<R**> hcol, scol, slcol, mcol, drg, sax, say, saz, pgas, tcol;
  Kokkos::View<int**> skt, skb;
  // fs: per column, the mass inject() and convert() took and lost since
  // tallies() (with dev: no reduction to the host per flow step)
  Kokkos::View<double**> sgot, slost;
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
    if (fs) {
      cmass = Kokkos::View<R***>("flow_mass", a, b, c);
      eps = Kokkos::View<R***>("flow_eps", a, b, c);
      mark = Kokkos::View<unsigned char***>("flow_mark", a, b, c);
      type2 = Kokkos::View<unsigned char***>("flow_type2", a, b, c);
      xs = Kokkos::View<R****>("flow_xs", a, b, c, 3);
      for (auto v : {&hcol, &scol, &slcol, &mcol, &drg, &sax, &say, &saz, &pgas, &tcol})
        *v = Kokkos::View<R**>("flow_col", nx, ny);
      skt = Kokkos::View<int**>("flow_skt", nx, ny); skb = Kokkos::View<int**>("flow_skb", nx, ny);
      sgot = Kokkos::View<double**>("flow_sgot", nx, ny); slost = Kokkos::View<double**>("flow_slost", nx, ny);
      Kokkos::deep_copy(skt, -1);
      if (fs_hf) {
        hfs = Kokkos::View<R****>("flow_hfs", a, b, c, 5);
        hfr = Kokkos::View<unsigned char***>("flow_hfr", a, b, c);
      }
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
    if (fs) { cmass.move(lo, hi); eps.move(lo, hi); mark.move(lo, hi); type2.move(lo, hi); xs.move(lo, hi); }
    if (fs && fs_hf) { hfs.move(lo, hi); hfr.move(lo, hi); }
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
      LightMD<3>({0, 0, 0}, {nx, ny, nz}),
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
    const bool hp = props, lazy_ = lazy, fs_ = fs;
    auto ms_ = cmass; auto ep_ = eps;
    auto dg_ = drg; auto sax_ = sax; auto say_ = say; auto saz_ = saz; auto kt_ = skt; auto kb_ = skb;
    auto sl_ = slcol;
    // fs_hf: the gas density and the surface force per cell (surface_hf);
    // the force only where there is a Marangoni term
    const bool hf_ = fs && fs_hf, hfm = hf_ && hf_mg() != R(0);
    auto hs_ = hfs;
    auto S_ = hp ? sig : T; auto dn_ = dens; auto vs_ = visc;
    const R nu3 = R(3.0 * p.nu), tauif = R(fs_tau);
    Kokkos::parallel_for("FlowStep",
      LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (!liq(ty(i, j, k))) {
          if (lazy_ && fl_(i, j, k) >= R(0)) return;
          #pragma unroll
          for (int q = 0; q < 19; q++) g_(i, j, k, q) = R(0);
          ux_(i, j, k) = uy_(i, j, k) = uz_(i, j, k) = rho_(i, j, k) = R(0);
          px_(i, j, k) = py_(i, j, k) = pz_(i, j, k) = R(0);
          #pragma unroll
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
        R dm = R(0);   // fs: the mass the links brought, less what they took
        int tq[19];
        bool L[19];
        // ph: the part of a link's populations that is mass exchanged (the
        // mean fill between two interface cells, else all of it)
        R ph[19];
        const bool ifc = ty(i, j, k) == INTERFACE;
        const R e0 = ifc ? ep_(i, j, k) : R(1);
        fq[0] = f_(i, j, k, 0); tq[0] = FLUID; L[0] = true; ph[0] = R(1);
        #pragma unroll
        for (int q = 1; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          int ii = i - cx, jj = j - cy, kk = k - cz;
          tq[q] = cell(ii, jj, kk);
          L[q] = liq(tq[q]);
          ph[q] = R(1);
          if (L[q]) {
            fq[q] = f_(ii, jj, kk, q);
            if (ifc && tq[q] == INTERFACE) ph[q] = R(0.5) * (e0 + ep_(ii, jj, kk));
          }
        }
        // fs: what arrives from the gas is the gas's equilibrium at this
        // cell's velocity, less what left toward it (the pressure boundary)
        const R ug = px_(i, j, k), vg = py_(i, j, k), wg = pz_(i, j, k);
        // An interface cell under its column's surface (a bubble's wall, or
        // the underside of an overhang) is held a one-cell bubble's Laplace
        // pressure, 2 sigma / r with r one cell, below the liquid: such a void
        // closes, as a bubble that small does. At the surface's own pressure
        // it neither closed nor grew; bubbles caught in a large 316L pool
        // were stretched and merged until a slot opened through the pool and
        // it collapsed (Sattari DED set, 380 ms).
        R drg = R(0);
        if (hf_) { if (ifc) drg = hs_(i, j, k, 0); }
        else {
          drg = fs_ ? dg_(i, j) : R(0);
          if (fs_ && ifc && kt_(i, j) >= 0 && k < kb_(i, j)) drg -= R(6) * sl_(i, j);
        }
        const R usg = ug * ug + vg * vg + wg * wg;
        auto rec = [&](int q) -> R {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R cu = cx * ug + cy * vg + cz * wg;
          return R(2) * wgt(q) * (drg + (R(1) + drg) * (R(4.5) * cu * cu - R(1.5) * usg)) - f_(i, j, k, opp(q));
        };
        // the axis velocities: q = 1 + 2 A + (sign < 0); face 2 A + (side > 0)
        #pragma unroll
        for (int q = 1; q < 7; q++) {
          if (L[q]) {
            G[q - 1] -= ph[q] * fq[q];               // came in by the face toward its source
            dm += ph[q] * (fq[q] - f_(i, j, k, opp(q)));
          }
          else if (fs_ && tq[q] == GAS) fq[q] = rec(q);
          else fq[q] = f_(i, j, k, opp(q));          // wall, or the flat surface above
        }
        #pragma unroll
        for (int q = 7; q < 19; q++) {
          int c[3]; vel(q, c[0], c[1], c[2]);
          const int A = c[0] != 0 ? 0 : 1, B = c[2] != 0 ? 2 : 1, sA = c[A], sB = c[B];
          const int o = opp(q);
          // the two cells between this one and the source: this - sA e_A, this - sB e_B
          const int tA = tq[1 + 2 * A + (sA < 0)], tB = tq[1 + 2 * B + (sB < 0)];
          const bool inA = liq(tA), inB = liq(tB);
          if (L[q] && (inA || inB)) {
            const R wv = (inA && inB ? R(0.5) : R(1)) * ph[q] * fq[q];
            if (inA) G[2 * A + (sA < 0)] -= wv;
            if (inB) G[2 * B + (sB < 0)] -= wv;
            dm += ph[q] * (fq[q] - f_(i, j, k, o));
          } else if (fs_) {
            // gas, or liquid that touches this cell only across gas
            fq[q] = (tq[q] == GAS || (L[q] && (tA == GAS || tB == GAS))) ? rec(q) : f_(i, j, k, o);
          } else if (tq[q] == GAS && sB == -1 && B == 2 && tq[6] == GAS && inA) {
            // off the flat surface above this cell and the one beside it,
            // which it left going up
            int i2 = i - c[0], j2 = j - c[1], k2 = k;
            cell(i2, j2, k2);
            fq[q] = f_(i2, j2, k2, mirz(q));
            G[2 * A + (sA < 0)] -= fq[q];
          } else fq[q] = f_(i, j, k, o);
          // what leaves along q, first face: by way of this + sA e_A or this + sB e_B
          if (L[o]) {
            const bool outA = liq(tq[1 + 2 * A + (sA > 0)]), outB = liq(tq[1 + 2 * B + (sB > 0)]);
            if (outA || outB) {
              const R wv = (outA && outB ? R(0.5) : R(1)) * ph[o] * f_(i, j, k, q);
              if (outA) G[2 * A + (sA > 0)] += wv;
              if (outB) G[2 * B + (sB > 0)] += wv;
            }
          }
        }
        #pragma unroll
        for (int g = 0; g < 6; g++) flux_(i, j, k, g) = first ? G[g] : flux_(i, j, k, g) + G[g];
        R dr = 0, mx = 0, my = 0, mz = 0;
        #pragma unroll
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          dr += fq[q]; mx += cx * fq[q]; my += cy * fq[q]; mz += cz * fq[q];
        }
        const R r = R(1) + dr, ir = R(1) / r;
        if (fs_) ms_(i, j, k) = ifc ? ms_(i, j, k) + dm : r;

        // acceleration: uniform + buoyancy + Marangoni (surface cells)
        const R dT = hp ? -dn_(i, j, k) : T_(i, j, k) - Tref;
        R ax = a0x + bx * dT, ay = a0y + by * dT, az = a0z + bz * dT;
        if (fs_) {
          // the surface force, on the column's surface cells (see surface())
          if (hf_) { if (hfm) { ax += hs_(i, j, k, 1); ay += hs_(i, j, k, 2); az += hs_(i, j, k, 3); } }
          else if (k >= kb_(i, j) && k <= kt_(i, j)) { ax += sax_(i, j); ay += say_(i, j); az += saz_(i, j); }
          // a partial cell takes its fill's part of every force: the
          // pressure under the surface then follows the surface's height
          // within a cell (else a surface inside a cell layer feels no
          // restoring force), and the column's cells together take the
          // whole surface force
          if (ifc) { const R w = e0 < R(0) ? R(0) : e0 > R(1) ? R(1) : e0; ax *= w; ay *= w; az *= w; }
        } else if (mar != R(0)) {
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
        #pragma unroll
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R cu = cx * u + cy * v + cz * w, ca = cx * ax + cy * ay + cz * az;
          feq[q] = wgt(q) * (dr + r * (R(3) * cu + R(4.5) * cu * cu - R(1.5) * usq));
          src[q] = wgt(q) * r * (R(3) * (ca - ua) + R(9) * cu * ca);
        }
        R tau0 = hp ? nu3 * vs_(i, j, k) + R(0.5) : tau00;
        if (ifc && tau0 < tauif) tau0 = tauif;
        R tau = tau0;
        if (smag2 != R(0)) {
          // eddy viscosity from the non-equilibrium stress
          R pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
          #pragma unroll
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
        #pragma unroll
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
      LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
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
    double sig_rho = 0.0;     // fs: surface tension / rho (m^3 / s^2), the largest there is
    int nmax = 256;
  } ph;
  double dt = 0.0, u_cap = 0.0, umax = 0.0;   // step (s), cap and last max speed (m/s)
  std::function<void()> halo;   // several ranks: brings the halo cells' f up to date
  // fs, several ranks: called inside convert() after the marks (0) and the
  // new types (1) are made, to bring the halo's up to date. own: this
  // rank's cells; the mass that convert() loses is counted there only.
  std::function<void(int)> fs_halo;
  int own_lo[3] = {0, 0, 0}, own_hi[3] = {1 << 30, 1 << 30, 1 << 30};
  // fs: called after each step in place of convert() (the caller carries
  // its fields and then converts)
  std::function<void()> post;
  long ncall = 0;

  int run(double h, const int lo[3], const int hi[3]) {
    if (u_cap == 0.0) u_cap = ph.u_ref;
    while (umax > 1.5 * u_cap) u_cap *= 2.0;
    if (umax < 0.3 * u_cap && u_cap > ph.u_ref) u_cap = std::max(ph.u_ref, 0.5 * u_cap);
    int n = 2 * std::max(1, (int) std::ceil(0.5 * h * u_cap / (0.1 * ph.dx) - 1e-9));
    // fs: the shortest capillary wave (two cells) must not outrun the step
    if (fs && ph.sig_rho > 0.0)
      n = std::max(n, 2 * (int) std::ceil(0.5 * h / (fs_cap * std::sqrt(ph.dx * ph.dx * ph.dx / ph.sig_rho)) - 1e-9));
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
    if (!props) p.sigma = ph.sig_rho * dt * dt / (ph.dx * ph.dx * ph.dx);
    p.sigconv = dt * dt / (ph.dx * ph.dx * ph.dx);
    // No fence per step: one was added when host memory grew on HIP with
    // the launches, which was Kokkos' constant-memory launch path (LightMD).
    for (int s = 0; s < n; s++) {
      if (fs) surface(lo, hi);
      step(lo, hi, fs || s == 0);
      if (halo && !post) halo();     // fs: post() brings f over once the cells are converted
      if (fs) { if (post) post(); else convert(lo, hi); }
    }
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
    auto f_ = f; auto ty = type; auto ms_ = cmass;
    auto ux_ = ux; auto uy_ = uy; auto uz_ = uz; auto px_ = px; auto py_ = py; auto pz_ = pz;
    const R s = R(s_);
    Kokkos::parallel_for("FlowRescale",
      LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (!liq(ty(i, j, k))) return;
        R dr = 0, mx = 0, my = 0, mz = 0;
        for (int q = 0; q < 19; q++) {
          int cx, cy, cz; vel(q, cx, cy, cz);
          const R v = f_(i, j, k, q);
          dr += v; mx += cx * v; my += cy * v; mz += cz * v;
        }
        const R r = R(1) + dr, u = mx / r, v = my / r, w = mz / r;
        const R dr2 = dr * s * s, r2 = R(1) + dr2, u2 = u * s, v2 = v * s, w2 = w * s;
        if (ty(i, j, k) == INTERFACE) ms_(i, j, k) *= r2 / r;   // the fill stays
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
      LightMD<3>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        for (int g = 0; g < 6; g++) fl_(i, j, k, g) = (g & 1) ? h[g / 2] : -h[g / 2];
      });
  }

  // sum of rho - 1 over the fluid cells
  double mass() const {
    auto f_ = f; auto ty = type;
    double m = 0.0;
    Kokkos::parallel_reduce("FlowMass",
      LightMD<3>({0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, double &s) {
        if (ty(i, j, k) != FLUID) return;
        for (int q = 0; q < 19; q++) s += f_(i, j, k, q);
      }, m);
    return m;
  }

  /* --------------------------------------------------------------------
     Free surface (see the top of the file).
  -------------------------------------------------------------------- */

  // the liquid's mass: fluid cells' density and interface cells' mass
  double total_mass() const {
    auto f_ = f; auto ty = type; auto ms_ = cmass;
    double m = 0.0;
    Kokkos::parallel_reduce("FlowTotalMass",
      LightMD<3>({wlo[0], wlo[1], wlo[2]}, {whi[0], whi[1], whi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, double &s) {
        if (ty(i, j, k) == INTERFACE) s += ms_(i, j, k);
        if (ty(i, j, k) != FLUID) return;
        double r = 1.0;
        for (int q = 0; q < 19; q++) r += f_(i, j, k, q);
        s += r;
      }, m);
    return m;
  }

  // The cell this + c_q, if it is in the block and held (wraps periodic
  // axes). A cell that is not there is no neighbour.
  struct Nbr {
    int n[3], lo[3], hi[3];
    bool per[3];
    KOKKOS_INLINE_FUNCTION bool operator()(int q, int i, int j, int k, int &ii, int &jj, int &kk) const {
      int c[3]; vel(q, c[0], c[1], c[2]);
      int x[3] = {i + c[0], j + c[1], k + c[2]};
      for (int d = 0; d < 3; d++) {
        if (x[d] < 0 || x[d] >= n[d]) { if (!per[d]) return false; x[d] = (x[d] + n[d]) % n[d]; }
        if (x[d] < lo[d] || x[d] >= hi[d]) return false;
      }
      ii = x[0]; jj = x[1]; kk = x[2];
      return true;
    }
  };
  Nbr nbr() const {
    Nbr b;
    b.n[0] = nx; b.n[1] = ny; b.n[2] = nz;
    for (int d = 0; d < 3; d++) { b.lo[d] = wlo[d]; b.hi[d] = whi[d]; b.per[d] = p.periodic[d]; }
    return b;
  }
  // [lo - 1, hi + 1) within the cells held
  void grown(const int lo[3], const int hi[3], int a[3], int b[3]) const {
    for (int d = 0; d < 3; d++) { a[d] = std::max(wlo[d], lo[d] - 1); b[d] = std::max(a[d], std::min(whi[d], hi[d] + 1)); }
  }

  // Types set by hand (or by melting and freezing): a FLUID cell that
  // touches GAS becomes a full INTERFACE cell; fill follows the types.
  void fs_repair() { const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz}; fs_repair(lo, hi); }
  void fs_repair(const int lo[3], const int hi[3]) {
    auto ty = type; auto ms_ = cmass; auto ep_ = eps; auto rho_ = rho;
    const Nbr nb = nbr();
    int a[3], b[3];
    grown(lo, hi, a, b);
    // only FLUID cells are written and only GAS cells decide: no race
    Kokkos::parallel_for("FlowFsRepair",
      LightMD<3>({a[0], a[1], a[2]}, {b[0], b[1], b[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const int t = ty(i, j, k);
        if (t == INTERFACE) return;
        ep_(i, j, k) = t == FLUID ? R(1) : R(0);
        if (t != FLUID) return;
        for (int q = 1; q < 19; q++) {
          int ii, jj, kk;
          if (nb(q, i, j, k, ii, jj, kk) && ty(ii, jj, kk) == GAS) {
            ty(i, j, k) = INTERFACE; ms_(i, j, k) = R(1) + rho_(i, j, k);
            return;
          }
        }
      });
  }

  /* --------------------------------------------------------------------
     Before each step: the surface's shape and the forces it carries, per
     column. The metal's height h(x, y) is the top full cell's top plus the
     fill of the interface cells stacked above it, so the surface is one
     height per column.
     ponytail: no overhangs (a bead that wets past 90 degrees, a keyhole, a
     droplet over a gap): local height functions along the normal if needed.
     - Capillary pressure: curvature kappa = div(grad h / sqrt(1 + |grad
       h|^2)) (negative on a hump); the surface sees the gas at
       pgas - sigma kappa, as the density of the reconstructed populations.
       A column with no liquid surface mirrors its liquid neighbour (a
       right contact angle), as does the block's edge.
     - Marangoni: the tangential force per column, sqrt(g) grad_s sigma =
       [((1 + hy^2) sx - hx hy sy) (1, 0, hx) + ((1 + hx^2) sy - hx hy sx)
       (0, 1, hy)] / sqrt(1 + hx^2 + hy^2), shared as one acceleration by the
       column's interface cells and the fluid cell under them (so a cell
       holding next to nothing is not thrown). sigma's difference is taken
       as in the flat model: one-sided beside a column without it.
  -------------------------------------------------------------------- */
  void surface(const int lo[3], const int hi[3]) {
    auto ty = type; auto ep_ = eps; auto ms_ = cmass; auto rho_ = rho; auto T_ = T;
    auto h_ = hcol; auto S_ = scol; auto sl_ = slcol; auto M_ = mcol; auto dg_ = drg; auto pg_ = pgas;
    auto sax_ = sax; auto say_ = say; auto saz_ = saz; auto kt_ = skt; auto kb_ = skb; auto tc_ = tcol;
    const bool hp = props;
    auto sg_ = sig; auto dn_ = dens;
    const R pinT = R(fs_pin_T);
    const R mar = R(p.marangoni), sig0 = R(p.sigma), Tref = R(p.Tref), rawp = R(fs_raw);
    auto st_ = stab;
    const bool tab = !hp && stab.extent(0) > 1;
    const int nst = (int) stab.extent(0);
    const R sT0 = stab_T0, siT = stab_idT, sc = R(p.sigconv);
    const R mg = tab ? sc : mar;    // the lattice sigma per unit of S_
    const R edge = R(fs_angle >= 89.99 ? 1e30 : std::tan(fs_angle * M_PI / 180.0));
    const int z0 = wlo[2], z1 = whi[2];
    // the columns held, and those of them the step needs (one more around)
    int a[2], b[2], c[2], d[2];
    const int nn[2] = {nx, ny};
    const bool per[2] = {p.periodic[0], p.periodic[1]};
    for (int e = 0; e < 2; e++) {
      a[e] = wlo[e]; b[e] = whi[e];
      c[e] = std::max(a[e], lo[e] - 1); d[e] = std::max(c[e], std::min(b[e], hi[e] + 1));
    }
    const int a0 = a[0], a1 = a[1], b0 = b[0], b1 = b[1], n0 = nx, n1 = ny;
    const bool p0 = per[0], p1 = per[1];
    (void) nn;
    Kokkos::parallel_for("FlowFsColumns",
      LightMD<2>({a[0], a[1]}, {b[0], b[1]}),
      KOKKOS_LAMBDA(const int i, const int j) {
        int k = z1 - 1;
        while (k >= z0 && ty(i, j, k) == GAS) k--;
        kt_(i, j) = -1; kb_(i, j) = 0; M_(i, j) = R(1); S_(i, j) = R(0); sl_(i, j) = R(0);
        if (k < z0) { h_(i, j) = R(z0); kt_(i, j) = -2; return; }      // no metal: -2
        const int kt = k;
        R sv = T_(i, j, kt);
        tc_(i, j) = sv;
        if (tab) {     // sigma(T) / rho from the table, linear between (and beyond) its rows
          const R u = (sv - sT0) * siT;
          const int m = u < R(0) ? 0 : u >= R(nst - 1) ? nst - 2 : (int) u;
          sv = st_(m) + (u - R(m)) * (st_(m + 1) - st_(m));
        }
        S_(i, j) = hp ? sg_(i, j, kt) : sv;
        if (ty(i, j, kt) == WALL) { h_(i, j) = R(kt + 1); return; }  // solid on top: -1
        sl_(i, j) = hp ? mar * (sg_(i, j, kt) + sig0) : tab ? sc * sv : sig0 + mar * (T_(i, j, kt) - Tref);
        R sum = 0, M = 0;
        while (k >= z0 && ty(i, j, k) == INTERFACE) { sum += ep_(i, j, k); M += ms_(i, j, k); k--; }
        int kb = k + 1;
        if (k >= z0 && ty(i, j, k) == FLUID) { M += R(1) + rho_(i, j, k); kb = k; }
        h_(i, j) = R(k + 1) + sum;
        // Liquid hanging over gas (left on a frozen rim when what was under
        // it drained or froze away) is not a surface one height describes:
        // it carries no surface force, like a column with solid on top.
        if (k >= z0 && ty(i, j, k) == GAS) return;
        kt_(i, j) = kt; kb_(i, j) = kb;
        M_(i, j) = (M > R(0.5) ? M : R(0.5)) * (hp ? R(1) + dn_(i, j, kt) : R(1));
      });
    // the columns stay: powder and the laser come from above
    if (fs_hf) { surface_hf(lo, hi); return; }
    Kokkos::parallel_for("FlowFsSurface",
      LightMD<2>({c[0], c[1]}, {d[0], d[1]}),
      KOKKOS_LAMBDA(const int i, const int j) {
        dg_(i, j) = sax_(i, j) = say_(i, j) = saz_(i, j) = R(0);
        if (kt_(i, j) < 0) return;
        // a neighbouring column: wraps, or stays at the edge of what is held
        auto col = [&](int di, int dj, int &ii, int &jj) {
          ii = i + di; jj = j + dj;
          if (p0) ii = (ii + n0) % n0;
          if (p1) jj = (jj + n1) % n1;
          ii = ii < a0 ? a0 : ii >= b0 ? b0 - 1 : ii;
          jj = jj < a1 ? a1 : jj >= b1 ? b1 - 1 : jj;
        };
        // The height the forces see is smoothed over 3 x 3 columns (1 2 1
        // each way). A capillary wave two cells long would travel at
        // sqrt(pi sigma), near the lattice's speed of sound for a metal at
        // any affordable step, and the surface breaks up; smoothed, that
        // wave feels no force and one four cells long half of it.
        // A column without a liquid surface (solid on top, or no metal)
        // counts as this column's height: the liquid meets solid at a right
        // angle, neither pulled over it nor held back, and where it stops
        // is left to the flow and to freezing.
        auto Hr = [&](int di, int dj) -> R {
          int ii, jj; col(di, dj, ii, jj);
          if (kt_(ii, jj) >= 0) return h_(ii, jj);
          // Solid standing lower than this liquid (the frozen rim under a
          // bead): the surface meets it at the contact angle, tan(fs_angle)
          // cells down, not below the solid. Mirrored (fs_angle 0) nothing
          // held a bead's edge, and it crept out over the rim and hung in
          // the air beside the pool.
          // Cold (below fs_pin_T): 90 degrees, the solid's own height.
          if (kt_(ii, jj) == -1 && h_(ii, jj) < h_(i, j))
            return tc_(ii, jj) < pinT || h_(i, j) - edge < h_(ii, jj) ? h_(ii, jj) : h_(i, j) - edge;
          return h_(i, j);
        };
        auto H = [&](int di, int dj) -> R {
          if (rawp == R(1)) return Hr(di, dj);     // no smoothing (the default): 9 lookups, not 81
          const R sm = R(0.25) * Hr(di, dj)
               + R(0.125) * (Hr(di - 1, dj) + Hr(di + 1, dj) + Hr(di, dj - 1) + Hr(di, dj + 1))
               + R(0.0625) * (Hr(di - 1, dj - 1) + Hr(di + 1, dj - 1) + Hr(di - 1, dj + 1) + Hr(di + 1, dj + 1));
          return sm + rawp * (Hr(di, dj) - sm);
        };
        const R hc = H(0, 0);
        const R hx = R(0.5) * (H(1, 0) - H(-1, 0)), hy = R(0.5) * (H(0, 1) - H(0, -1));
        const R hxx = H(1, 0) - R(2) * hc + H(-1, 0), hyy = H(0, 1) - R(2) * hc + H(0, -1);
        const R hxy = R(0.25) * (H(1, 1) - H(1, -1) - H(-1, 1) + H(-1, -1));
        const R g = R(1) + hx * hx + hy * hy, sg = Kokkos::sqrt(g);
        const R kap = (hxx * (R(1) + hy * hy) + hyy * (R(1) + hx * hx) - R(2) * hxy * hx * hy) / (g * sg);
        dg_(i, j) = R(3) * (pg_(i, j) - sl_(i, j) * kap);
        if (mg == R(0)) return;
        // sigma's difference per cell along x (e = 0) or y
        auto grad = [&](int e) -> R {
          int ia, ja, ib, jb;
          col(e == 0 ? -1 : 0, e == 0 ? 0 : -1, ia, ja); col(e == 0 ? 1 : 0, e == 0 ? 0 : 1, ib, jb);
          R w = R((e == 0 ? ib - ia : jb - ja));
          if (w < R(0)) w += R(e == 0 ? n0 : n1);          // wrapped
          // props: sigma is the liquid's only; else T, wherever there is metal
          if (hp ? kt_(ia, ja) < 0 : kt_(ia, ja) == -2) { ia = i; ja = j; w -= R(1); }
          if (hp ? kt_(ib, jb) < 0 : kt_(ib, jb) == -2) { ib = i; jb = j; w -= R(1); }
          return w > R(0) ? mg * (S_(ib, jb) - S_(ia, ja)) / w : R(0);
        };
        const R sx = grad(0), sy = grad(1);
        const R fa = ((R(1) + hy * hy) * sx - hx * hy * sy) / sg, fb = ((R(1) + hx * hx) * sy - hx * hy * sx) / sg;
        const R im = R(1) / M_(i, j);
        sax_(i, j) = fa * im; say_(i, j) = fb * im; saz_(i, j) = (fa * hx + fb * hy) * im;
      });
  }

  // the lattice sigma per unit of the surface's source value (T, sigma(T)
  // / rho from the table, or sig): 0 when there is no Marangoni force
  R hf_mg() const { return R(!props && stab.extent(0) > 1 ? p.sigconv : p.marangoni); }

  /* --------------------------------------------------------------------
     fs_hf: the surface per INTERFACE cell, from local height functions
     along its normal (Popinet 2009; Cummins, Francois & Kothe 2005). See
     docs/FREE_SURFACE_LBM.md, section 3a.
     - The frame: the axis d of the largest component of n = -grad(fill)
       (Youngs' 3 x 3 x 3 weights; FLUID and WALL full), heights along
       s e_d with s the side of the gas.
     - Heights of the 3 x 3 columns of the frame: each walked from this
       cell's row down to a full cell and up through partial cells to gas,
       at most HF_L cells; h = (full row) + 1/2 + the partial cells' fills.
       A column is liquid, dry (full right under gas), empty (no full cell
       in reach), high (no gas in reach), at the edge of what is held, or
       bad (it crosses the surface twice in reach).
     - The column model's boundary rules, in the frame: an empty, high or
       edge column counts as this one's height (a right angle); a dry one
       lower than this one is met at fs_angle (90 degrees if colder than
       fs_pin_T).
     - The dominant frame is used if this cell's column is liquid and none
       is bad; else the other two axes (every column liquid or dry: a
       steeper frame overflows the reach, so no mirrors there); else, in a
       second kernel, the mean curvature of the INTERFACE cells around that
       their heights measured; else a least-squares paraboloid in the
       normal's frame through the surface points of every liquid column of
       the three frames (at least 6; why it comes last: FlowFsHFKappa);
       else none, and a cell under its column's surface keeps the column
       model's void rule (-6 sigma).
     - The reach HF_L is 5: at 3 (Popinet's 3 x 7 stencil) a column beside
       a cell near 45 degrees often needs 4 cells to reach a full one, and
       a sphere of radius 8 cells left 35% of its cells to the fallbacks.
     - kappa from the 9 heights as in the column model; the gas density
       3 (pgas - sigma kappa), sigma from this cell's own value.
     - Marangoni: the column model's force in the frame, shared as one
       acceleration by the frame column's partial cells and its FLUID base
       (that base gathers it in a third kernel).
     Several ranks: the halo of type, eps, cmass, rho and T (or sig) must
     be HF_L + 1 deep (a cell one beyond this rank's walks HF_L more).
     ponytail: cells given the mean or the fit carry no Marangoni force.
  -------------------------------------------------------------------- */
  static constexpr int HF_L = 5;
  void surface_hf(const int lo[3], const int hi[3]) {
    enum { CL_LIQ, CL_DRY, CL_EMPTY, CL_HIGH, CL_EDGE, CL_BAD };
    auto ty = type; auto ep_ = eps; auto ms_ = cmass; auto rho_ = rho; auto T_ = T;
    auto pg_ = pgas; auto kt_ = skt; auto kb_ = skb; auto hs_ = hfs; auto hr_ = hfr;
    const bool hp = props;
    auto sg_ = sig; auto dn_ = dens; auto st_ = stab;
    const R pinT = R(fs_pin_T), mar = R(p.marangoni), sig0 = R(p.sigma), Tref = R(p.Tref);
    const bool tab = !hp && stab.extent(0) > 1;
    const int nst = (int) stab.extent(0);
    const R sT0 = stab_T0, siT = stab_idT, sc = R(p.sigconv), mg = hf_mg();
    const R edge = R(fs_angle >= 89.99 ? 1e30 : std::tan(fs_angle * M_PI / 180.0));
    const int nn0 = nx, nn1 = ny, nn2 = nz;
    const bool pp0 = p.periodic[0], pp1 = p.periodic[1], pp2 = p.periodic[2];
    const int wl0 = wlo[0], wl1 = wlo[1], wl2 = wlo[2], wh0 = whi[0], wh1 = whi[1], wh2 = whi[2];
    int a[3], b[3];
    grown(lo, hi, a, b);
    Kokkos::parallel_for("FlowFsHF",
      LightMD<3>({a[0], a[1], a[2]}, {b[0], b[1], b[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != INTERFACE) return;
        const int nn[3] = {nn0, nn1, nn2}, wl[3] = {wl0, wl1, wl2}, wh[3] = {wh0, wh1, wh2};
        const bool pp[3] = {pp0, pp1, pp2};
        // a cell's class: 0 GAS, 1 partial (INTERFACE; e its fill, clamped),
        // 2 full (FLUID, WALL, below the block), 3 not known (beyond the
        // block's sides or the cells held). x is wrapped in place; below the
        // block (class 2) it stays out of range.
        auto at = [&](int x[3], R &e) -> int {
          e = R(0);
          for (int d = 0; d < 3; d++) {
            if (x[d] < 0 || x[d] >= nn[d]) {
              if (pp[d]) x[d] = (x[d] + nn[d]) % nn[d];
              else if (d < 2) return 3;
              else if (x[d] < 0) { e = R(1); return 2; }
              else return 0;
            }
            if (x[d] < wl[d] || x[d] >= wh[d]) return 3;
          }
          const int t = ty(x[0], x[1], x[2]);
          if (t == GAS) return 0;
          if (t == INTERFACE) { const R v = ep_(x[0], x[1], x[2]); e = v < R(0) ? R(0) : v > R(1) ? R(1) : v; return 1; }
          e = R(1);
          return 2;
        };
        // the surface's source value at a cell
        auto Sv = [&](const int x[3]) -> R {
          if (hp) return sg_(x[0], x[1], x[2]);
          R v = T_(x[0], x[1], x[2]);
          if (tab) {
            const R u = (v - sT0) * siT;
            const int m = u < R(0) ? 0 : u >= R(nst - 1) ? nst - 2 : (int) u;
            v = st_(m) + (u - R(m)) * (st_(m + 1) - st_(m));
          }
          return v;
        };
        const int xc[3] = {i, j, k};
        const R Sc = Sv(xc);
        const R sl = hp ? mar * (Sc + sig0) : tab ? sc * Sc : sig0 + mar * (Sc - Tref);
        // Youngs' normal, toward the gas
        R e0, nv[3] = {R(0), R(0), R(0)};
        { int x[3] = {i, j, k}; at(x, e0); }
        for (int dz = -1; dz <= 1; dz++)
          for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
              if (!dx && !dy && !dz) continue;
              int x[3] = {i + dx, j + dy, k + dz};
              R e;
              if (at(x, e) == 3) e = e0;
              nv[0] -= R(dx * (dy ? 1 : 2) * (dz ? 1 : 2)) * e;
              nv[1] -= R(dy * (dx ? 1 : 2) * (dz ? 1 : 2)) * e;
              nv[2] -= R(dz * (dx ? 1 : 2) * (dy ? 1 : 2)) * e;
            }
        // the axes by |n|, ties to z (the column model's frame), then x
        int ord[3] = {2, 0, 1};
        for (int u = 0; u < 2; u++)
          for (int q = 0; q < 2 - u; q++)
            if (Kokkos::abs(nv[ord[q + 1]]) > Kokkos::abs(nv[ord[q]])) { const int t = ord[q]; ord[q] = ord[q + 1]; ord[q + 1] = t; }
        // the cell of row m of column (u, v) of the frame (d, s)
        auto cellx = [&](int d, int s, int u, int v, int m, int x[3]) {
          const int d1 = d == 0 ? 1 : 0, d2 = d == 2 ? 1 : 2;
          x[0] = i; x[1] = j; x[2] = k;
          x[d1] += u; x[d2] += v; x[d] += s * m;
        };
        // column (u, v) of the frame (d, s): its class, height (from this
        // cell's centre along s e_d), full row and top non-gas row
        auto column = [&](int d, int s, int u, int v, R &h, int &mf, int &mt) -> int {
          auto get = [&](int m, R &e) -> int { int x[3]; cellx(d, s, u, v, m, x); return at(x, e); };
          R e;
          const int c0 = get(0, e);
          if (c0 == 3) return CL_EDGE;
          R sum = R(0);
          int m = 0, cm = c0, me = 1;
          if (c0 == 0) {
            while (cm == 0) { if (--m < -HF_L) return CL_EMPTY; cm = get(m, e); }
            me = m + 1;
          }
          while (cm == 1) { sum += e; if (--m < -HF_L) return CL_BAD; cm = get(m, e); }
          if (cm == 3) return CL_EDGE;
          if (cm == 0) return CL_BAD;                    // gas under a partial cell
          mf = m;
          if (c0 != 0) {
            m = 0; cm = c0;
            if (c0 == 2) { while (cm == 2) { mf = m; if (++m > HF_L) return CL_HIGH; cm = get(m, e); } }
            else cm = get(++m, e);
            while (cm == 1) { sum += e; if (++m > HF_L) return CL_BAD; cm = get(m, e); }
            if (cm == 3) return CL_EDGE;
            if (cm == 2) return CL_BAD;                  // full over a partial cell
            me = m;
          }
          h = R(mf) + R(0.5) + sum;
          mt = me - 1;
          return me > mf + 1 ? CL_LIQ : CL_DRY;
        };
        R H[9];
        int C[9], MF[9], MT[9];
        auto frame = [&](int d, int s, bool dom) -> bool {
          for (int q = 0; q < 9; q++) {
            H[q] = R(0); MF[q] = MT[q] = 0;
            C[q] = column(d, s, q / 3 - 1, q % 3 - 1, H[q], MF[q], MT[q]);
            if (C[q] == CL_BAD || (!dom && C[q] != CL_LIQ && C[q] != CL_DRY)) return false;
          }
          return C[4] == CL_LIQ;
        };
        int method = HF_NONE, fd = ord[0], fsd = nv[ord[0]] < R(0) ? -1 : 1;
        for (int t = 0; t < 3 && method == HF_NONE; t++) {
          const int d = ord[t], s = nv[d] < R(0) ? -1 : 1;
          if (frame(d, s, t == 0)) { method = t == 0 ? HF_DOM : HF_OTHER; fd = d; fsd = s; }
        }
        R kap = R(0), acc[3] = {R(0), R(0), R(0)};
        if (method != HF_NONE) {
          const R hc = H[4];
          // the top cell of column q, if it is in the block
          auto topx = [&](int q, int x[3]) -> bool {
            R e;
            cellx(fd, fsd, q / 3 - 1, q % 3 - 1, MT[q], x);
            at(x, e);
            return x[2] >= 0 && x[2] < nn[2];
          };
          R g9[9];
          for (int q = 0; q < 9; q++) {
            g9[q] = hc;     // empty, high, edge: a right angle
            if (C[q] == CL_LIQ) g9[q] = H[q];
            else if (C[q] == CL_DRY && H[q] < hc) {
              // dry solid lower than the liquid: met at the contact angle,
              // or held at its own height when it is cold
              int x[3];
              const bool cold = topx(q, x) && T_(x[0], x[1], x[2]) < pinT;
              g9[q] = cold || hc - edge < H[q] ? H[q] : hc - edge;
            }
          }
          // q = 3 (u + 1) + (v + 1)
          const R h1 = R(0.5) * (g9[7] - g9[1]), h2 = R(0.5) * (g9[5] - g9[3]);
          const R h11 = g9[7] - R(2) * hc + g9[1], h22 = g9[5] - R(2) * hc + g9[3];
          const R h12 = R(0.25) * (g9[8] - g9[6] - g9[2] + g9[0]);
          const R g = R(1) + h1 * h1 + h2 * h2, sg = Kokkos::sqrt(g);
          kap = (h11 * (R(1) + h2 * h2) + h22 * (R(1) + h1 * h1) - R(2) * h12 * h1 * h2) / (g * sg);
          if (mg != R(0)) {
            // the source value of a column's top cell (liquid; or dry, by T)
            auto Sq = [&](int q, R &v) -> bool {
              int x[3];
              if (!(C[q] == CL_LIQ || (C[q] == CL_DRY && !hp)) || !topx(q, x)) return false;
              v = Sv(x);
              return true;
            };
            R S4 = Sc;
            Sq(4, S4);
            auto grad = [&](int qa, int qb) -> R {
              R va, vb, w = R(2);
              if (!Sq(qa, va)) { va = S4; w -= R(1); }
              if (!Sq(qb, vb)) { vb = S4; w -= R(1); }
              return w > R(0) ? mg * (vb - va) / w : R(0);
            };
            const R s1 = grad(1, 7), s2 = grad(3, 5);
            const R fa = ((R(1) + h2 * h2) * s1 - h1 * h2 * s2) / sg, fb = ((R(1) + h1 * h1) * s2 - h1 * h2 * s1) / sg;
            // the frame column's mass: its partial cells and a FLUID base
            R M = R(0);
            for (int m = MF[4]; m <= MT[4]; m++) {
              int x[3]; R e;
              cellx(fd, fsd, 0, 0, m, x);
              const int cl = at(x, e);
              if (cl == 1) { const R v = ms_(x[0], x[1], x[2]); M += v > R(0) ? v : R(0); }
              else if (cl == 2 && m == MF[4] && x[2] >= 0 && ty(x[0], x[1], x[2]) == FLUID)
                M += R(1) + rho_(x[0], x[1], x[2]);
            }
            const R im = R(1) / ((M > R(0.5) ? M : R(0.5)) * (hp ? R(1) + dn_(i, j, k) : R(1)));
            const int d1 = fd == 0 ? 1 : 0, d2 = fd == 2 ? 1 : 2;
            acc[d1] = fa * im; acc[d2] = fb * im; acc[fd] = R(fsd) * (fa * h1 + fb * h2) * im;
          }
        } else {
          // Popinet's fallback: a paraboloid z' = c0 + c1 x' + c2 y' + c3 x'^2
          // + c4 y'^2 + c5 x'y' in the normal's frame, through the surface
          // points that the liquid columns of all three frames give
          const R nm = Kokkos::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
          R nh[3] = {R(0), R(0), R(1)};
          if (nm > R(0)) for (int d = 0; d < 3; d++) nh[d] = nv[d] / nm;
          const int ax = Kokkos::abs(nh[0]) < R(0.9) ? 0 : 1;
          R tu[3] = {R(0), R(0), R(0)};
          tu[ax] = R(1);
          {
            const R dp = nh[ax];
            R l = R(0);
            for (int d = 0; d < 3; d++) { tu[d] -= dp * nh[d]; l += tu[d] * tu[d]; }
            l = R(1) / Kokkos::sqrt(l);
            for (int d = 0; d < 3; d++) tu[d] *= l;
          }
          const R tv[3] = {nh[1] * tu[2] - nh[2] * tu[1], nh[2] * tu[0] - nh[0] * tu[2], nh[0] * tu[1] - nh[1] * tu[0]};
          // the normal equations, lower triangle by rows
          R A[21], r6[6];
          for (int q = 0; q < 21; q++) A[q] = R(0);
          for (int q = 0; q < 6; q++) r6[q] = R(0);
          int np = 0;
          for (int d = 0; d < 3; d++) {
            if (nv[d] == R(0)) continue;
            const int s = nv[d] < R(0) ? -1 : 1, d1 = d == 0 ? 1 : 0, d2 = d == 2 ? 1 : 2;
            for (int q = 0; q < 9; q++) {
              R h;
              int mf, mt;
              if (column(d, s, q / 3 - 1, q % 3 - 1, h, mf, mt) != CL_LIQ) continue;
              R r[3];
              r[d1] = R(q / 3 - 1); r[d2] = R(q % 3 - 1); r[d] = R(s) * h;
              const R X = r[0] * tu[0] + r[1] * tu[1] + r[2] * tu[2];
              const R Y = r[0] * tv[0] + r[1] * tv[1] + r[2] * tv[2];
              const R Z = r[0] * nh[0] + r[1] * nh[1] + r[2] * nh[2];
              const R f[6] = {R(1), X, Y, X * X, Y * Y, X * Y};
              int p = 0;
              for (int u = 0; u < 6; u++) { r6[u] += f[u] * Z; for (int v = 0; v <= u; v++) A[p++] += f[u] * f[v]; }
              np++;
            }
          }
          if (np >= 6) {
            // Cholesky in place; a singular set (points on a line) fails
            const R tiny = R(1e-6) * (A[0] + A[20]);
            bool ok = true;
            int q = 0;
            for (int u = 0; u < 6; u++)
              for (int v = 0; v <= u; v++, q++) {
                R s = A[q];
                for (int w = 0; w < v; w++) s -= A[u * (u + 1) / 2 + w] * A[v * (v + 1) / 2 + w];
                if (u == v) { if (s <= tiny) ok = false; A[q] = Kokkos::sqrt(s > tiny ? s : tiny); }
                else A[q] = s / A[v * (v + 1) / 2 + v];
              }
            if (ok) {
              R y[6];
              for (int u = 0; u < 6; u++) {
                R s = r6[u];
                for (int w = 0; w < u; w++) s -= A[u * (u + 1) / 2 + w] * y[w];
                y[u] = s / A[u * (u + 1) / 2 + u];
              }
              for (int u = 5; u >= 0; u--) {
                R s = y[u];
                for (int w = u + 1; w < 6; w++) s -= A[w * (w + 1) / 2 + u] * y[w];
                y[u] = s / A[u * (u + 1) / 2 + u];
              }
              const R b1 = y[1], b2 = y[2], gq = R(1) + b1 * b1 + b2 * b2;
              kap = R(2) * (y[3] * (R(1) + b2 * b2) + y[4] * (R(1) + b1 * b1) - y[5] * b1 * b2) / (gq * Kokkos::sqrt(gq));
              method = HF_FIT;
            }
          }
        }
        // the curvature, and this cell's sigma for the second pass
        hs_(i, j, k, 0) = sl; hs_(i, j, k, 4) = kap;
        hs_(i, j, k, 1) = acc[0]; hs_(i, j, k, 2) = acc[1]; hs_(i, j, k, 3) = acc[2];
        hr_(i, j, k) = (unsigned char) (method << 3 | (2 * fd + (fsd > 0)));
      });
    // A cell its own heights did not measure takes the mean curvature of
    // the INTERFACE cells around it that were, if any (Basilisk's last
    // resort, here before the fit): they are mostly cells holding next to
    // nothing at a corner of the surface, whose columns cross it twice
    // within reach. The fit's points are column means, not point heights:
    // off the surface by about (h_11 + h_22) / 24 cells, more on the
    // steeper frames, which the fit takes for curvature (4-6% on a sphere
    // of 8 or 16 cells, not converging; the mean is 1-2%). Then the gas
    // density from kappa.
    Kokkos::parallel_for("FlowFsHFKappa",
      LightMD<3>({a[0], a[1], a[2]}, {b[0], b[1], b[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != INTERFACE) return;
        const int c = hr_(i, j, k);
        R kap = hs_(i, j, k, 4);
        const R sl = hs_(i, j, k, 0);
        int m = c >> 3;
        if (m == HF_FIT || m == HF_NONE) {
          R s = R(0);
          int n = 0;
          for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
              for (int dx = -1; dx <= 1; dx++) {
                int x[3] = {i + dx, j + dy, k + dz};
                bool in = true;
                for (int d = 0; d < 3; d++) {
                  const int nd = d == 0 ? nn0 : d == 1 ? nn1 : nn2;
                  const bool pd = d == 0 ? pp0 : d == 1 ? pp1 : pp2;
                  if (x[d] < 0 || x[d] >= nd) { if (pd) x[d] = (x[d] + nd) % nd; else in = false; }
                }
                if (!in || x[0] < a[0] || x[0] >= b[0] || x[1] < a[1] || x[1] >= b[1] || x[2] < a[2] || x[2] >= b[2]) continue;
                if (ty(x[0], x[1], x[2]) != INTERFACE) continue;
                const int mn = hr_(x[0], x[1], x[2]) >> 3;
                if (mn == HF_DOM || mn == HF_OTHER) { s += hs_(x[0], x[1], x[2], 4); n++; }
              }
          if (n) { kap = s / R(n); m = HF_AVG; hr_(i, j, k) = (unsigned char) (m << 3 | (c & 7)); }
        }
        R dg = R(3) * (pg_(i, j) - sl * kap);
        // nothing resolves it (a void of a cell or two): the column model's void rule
        if (m == HF_NONE && kt_(i, j) >= 0 && k < kb_(i, j)) dg -= R(6) * sl;
        hs_(i, j, k, 0) = dg;
      });
    if (mg == R(0)) return;
    // a FLUID cell takes the force of each frame column it is the base of:
    // the neighbour along q = s e_d whose frame is (d, s)
    const Nbr nb = nbr();
    Kokkos::parallel_for("FlowFsHFBase",
      LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (ty(i, j, k) != FLUID) return;
        R acc[3] = {R(0), R(0), R(0)};
        for (int q = 1; q < 7; q++) {
          int ii, jj, kk;
          if (!nb(q, i, j, k, ii, jj, kk) || ty(ii, jj, kk) != INTERFACE) continue;
          const int c = hr_(ii, jj, kk), m = c >> 3;
          if ((m == HF_DOM || m == HF_OTHER) && (c & 7) == 2 * ((q - 1) / 2) + (q & 1))
            for (int d = 0; d < 3; d++) acc[d] += hs_(ii, jj, kk, 1 + d);
        }
        hs_(i, j, k, 1) = acc[0]; hs_(i, j, k, 2) = acc[1]; hs_(i, j, k, 3) = acc[2];
      });
  }

  // Mass arriving on the surface (powder): scale * dm(x, y), in cells of
  // liquid, goes into the lowest interface cell of each column's surface
  // stack (as surface() found it: call between step() and convert(), which
  // passes on what overfills a cell). Into the top one, a partial cell under
  // an empty one (made when a neighbour filled) was built over and left as
  // a void in the liquid: a two-cell foam in the pool's top layers. The carried fields mix with the values qf
  // and q2f hold for that cell. Returns the mass taken; what falls on a
  // column with no liquid surface is not (with dev: kept for tallies()).
  double inject(const int lo[3], const int hi[3], Kokkos::View<R**> dm, R scale, W3 q = W3(), W3 qf = W3(),
                W3 q2 = W3(), W3 q2f = W3(), bool dev = false) {
    auto ty = type; auto ms_ = cmass; auto kt_ = skt; auto kb_ = skb;
    const bool one = q.v.extent(0) > 0, two = q2.v.extent(0) > 0;
    auto body = KOKKOS_LAMBDA(const int i, const int j, double &g) {
        if (kt_(i, j) < 0) return;
        int k = kb_(i, j);
        if (ty(i, j, k) != INTERFACE) k++;      // kb is the fluid cell under the stack
        const R d = scale * dm(i, j);
        if (k > kt_(i, j) || d <= R(0) || ty(i, j, k) != INTERFACE) return;
        const R m = ms_(i, j, k), mn = m + d;
        // a cell overdrawn by more than arrives takes the feed's values
        const R w = m > R(0) ? d / mn : R(1);
        if (one) q(i, j, k) += w * (qf(i, j, k) - q(i, j, k));
        if (two) q2(i, j, k) += w * (q2f(i, j, k) - q2(i, j, k));
        ms_(i, j, k) = mn;
        g += d;
      };
    return run_sum(LightMD<2>({lo[0], lo[1]}, {hi[0], hi[1]}), "FlowFsInject", dev, sgot, body);
  }

  // A sum over a range, reduced to the host (dev false), or added per
  // column to acc (one value per column and call: a 2-D range writes its
  // own column; a 3-D one adds atomically, nonzero values only).
  template <class P, class B>
  double run_sum(const P &pol, const char *nm, bool dev, Kokkos::View<double**> acc, const B &body) {
    if (!dev) { double s = 0.0; Kokkos::parallel_reduce(nm, pol, body, s); return s; }
    if constexpr (P::rank == 2)
      Kokkos::parallel_for(nm, pol, KOKKOS_LAMBDA(const int i, const int j) {
        double v = 0.0; body(i, j, v); acc(i, j) += v;
      });
    else
      Kokkos::parallel_for(nm, pol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double v = 0.0; body(i, j, k, v);
        if (v != 0.0) Kokkos::atomic_add(&acc(i, j), v);
      });
    return 0.0;
  }

  // what inject() took and convert() lost (dev) since the last call
  void tallies(double &got, double &lost) {
    auto g = sgot; auto l = slost;
    double a = 0.0, b = 0.0;
    Kokkos::parallel_reduce("FlowFsTallies", LightMD<2>({0, 0}, {nx, ny}),
      KOKKOS_LAMBDA(const int i, const int j, double &x, double &y) {
        x += g(i, j); y += l(i, j); g(i, j) = 0.0; l(i, j) = 0.0;
      }, a, b);
    got = a; lost = b;
  }

  // After a step (and melt_advect): filled interface cells become FLUID,
  // emptied ones GAS, their neighbours INTERFACE; the excess mass goes to
  // the interface cells around, with its share of the carried fields q, q2
  // (may be empty). [lo, hi) is step()'s window; cells one beyond it can
  // become INTERFACE, so the next window must hold them. Returns the mass
  // that found no interface cell to go to (0 unless a cell is isolated;
  // with dev it is kept for tallies()).
  double convert() { const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz}; return convert(lo, hi); }
  double convert(const int lo[3], const int hi[3], W3 q = W3(), W3 q2 = W3(), bool dev = false) {
    auto f_ = f; auto ty = type; auto t2 = type2; auto mk = mark; auto ms_ = cmass; auto ep_ = eps; auto xs_ = xs;
    auto rho_ = rho; auto ux_ = ux; auto uy_ = uy; auto uz_ = uz; auto px_ = px; auto py_ = py; auto pz_ = pz;
    auto fl_ = fl; auto flux_ = flux;
    const Nbr nb = nbr();
    const R kap = R(fs_kappa);
    const bool one = q.v.extent(0) > 0, two = q2.v.extent(0) > 0;
    enum { FILL = 1, EMPTY = 2 };
    int a[3], b[3];
    grown(lo, hi, a, b);
    const LightMD<3> range({a[0], a[1], a[2]}, {b[0], b[1], b[2]});
    // 1. mark. A cell does not empty beside one that fills (the filled
    //    cell would touch gas). Interface cells with neither fluid nor wall
    //    beside them are a droplet too small for the grid: nothing would
    //    move its mass and it would hang, falling for ever. It empties into
    //    its interface neighbours (if it has none the mass is returned).
    const bool open = !p.periodic[2];
    const int ktop = nz - 4;
    Kokkos::parallel_for("FlowFsMark", range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
      mk(i, j, k) = 0;
      if (ty(i, j, k) != INTERFACE) return;
      const R m = ms_(i, j, k), r = R(1) + rho_(i, j, k);
      bool nf = false, held = false;
      for (int c = 1; c < 19; c++) {
        int ii, jj, kk;
        if (!nb(c, i, j, k, ii, jj, kk)) { held = true; continue; }   // the block's edge, or not known
        const int t = ty(ii, jj, kk);
        if (t == FLUID || t == WALL) held = true;
        if (t == INTERFACE && ms_(ii, jj, kk) > (R(1) + kap) * (R(1) + rho_(ii, jj, kk))) nf = true;
      }
      // liquid thrown to the block's top (spatter) leaves: it empties there,
      // filling or not (a filling cell there let a whole droplet through)
      if (open && k >= ktop) mk(i, j, k) = EMPTY;
      else if (m > (R(1) + kap) * r) mk(i, j, k) = FILL;
      else if (!nf && (!held || m < -kap * r)) mk(i, j, k) = EMPTY;
    });
    if (fs_halo) fs_halo(0);
    // 2. the new types, in type2. A gas cell that becomes interface starts
    //    at the equilibrium of its liquid neighbours' mean density and
    //    velocity; an emptied cell is cleared. A gas cell is entered from
    //    the side or below only where the cell under it is not gas, and
    //    always from straight above: liquid does not grow out over gas
    //    (left alone it made a lip that froze over a gap at a bead's toe,
    //    which the column heights cannot see). A filled cell with a gas
    //    neighbour that stays gas stays INTERFACE; its excess goes on (3.).
    //    ponytail: a droplet in flight cannot move sideways over gas; none
    //    seen in the DED decks. Relax to "liquid below the gas cell's
    //    column somewhere" if they matter.
    const bool sup = !p.periodic[2] && !fs_hf;   // fs_hf: side faces have their curvature
    Kokkos::parallel_for("FlowFsRetype", range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
      const int t = ty(i, j, k), m = mk(i, j, k);
      // what a marked cell hands on, kept: one kept as interface writes its own
      if (m) {
        xs_(i, j, k, 0) = ms_(i, j, k) - (m == FILL ? R(1) + rho_(i, j, k) : R(0));
        if (one) xs_(i, j, k, 1) = q(i, j, k);
        if (two) xs_(i, j, k, 2) = q2(i, j, k);
      }
      // a gas cell beside a filled one becomes interface
      auto opens = [&](int gi, int gj, int gk) -> bool {
        int x, y, z;
        if (!sup || !nb(6, gi, gj, gk, x, y, z) || ty(x, y, z) != GAS) return true;
        return nb(5, gi, gj, gk, x, y, z) && mk(x, y, z) == FILL;
      };
      int nt = t;
      if (t == INTERFACE) {
        nt = m == FILL ? FLUID : m == EMPTY ? GAS : INTERFACE;
        for (int c = 1; nt == FLUID && c < 19; c++) {
          int ii, jj, kk;
          if (nb(c, i, j, k, ii, jj, kk) && ty(ii, jj, kk) == GAS && !opens(ii, jj, kk)) nt = INTERFACE;
        }
      }
      else if (t == GAS || t == FLUID) {
        const int want = t == GAS ? FILL : EMPTY;
        bool hit = false;
        R dr = 0, u = 0, v = 0, w = 0;
        int n = 0;
        for (int c = 1; c < 19; c++) {
          int ii, jj, kk;
          if (!nb(c, i, j, k, ii, jj, kk)) continue;
          const int mn = mk(ii, jj, kk);
          if (mn == want) hit = true;
          if (t == GAS && liq(ty(ii, jj, kk)) && mn != EMPTY) {
            dr += rho_(ii, jj, kk); u += px_(ii, jj, kk); v += py_(ii, jj, kk); w += pz_(ii, jj, kk); n++;
          }
        }
        if (hit && (t != GAS || opens(i, j, k))) {
          nt = INTERFACE;
          if (t == GAS) {
            const R in = R(1) / n;
            dr *= in; u *= in; v *= in; w *= in;
            const R usq = u * u + v * v + w * w;
            for (int c = 0; c < 19; c++) {
              int cx, cy, cz; vel(c, cx, cy, cz);
              const R cu = cx * u + cy * v + cz * w;
              f_(i, j, k, c) = wgt(c) * (dr + (R(1) + dr) * (R(3) * cu + R(4.5) * cu * cu - R(1.5) * usq));
            }
            rho_(i, j, k) = dr; fl_(i, j, k) = R(1);
            px_(i, j, k) = ux_(i, j, k) = u; py_(i, j, k) = uy_(i, j, k) = v; pz_(i, j, k) = uz_(i, j, k) = w;
          }
        }
      }
      if (t == INTERFACE && nt == GAS) {
        for (int c = 0; c < 19; c++) f_(i, j, k, c) = R(0);
        ux_(i, j, k) = uy_(i, j, k) = uz_(i, j, k) = R(0);
        px_(i, j, k) = py_(i, j, k) = pz_(i, j, k) = R(0);
        for (int g = 0; g < 6; g++) flux_(i, j, k, g) = R(0);
      }
      t2(i, j, k) = (unsigned char) nt;
    });
    if (fs_halo) fs_halo(1);
    const int ol0 = own_lo[0], ol1 = own_lo[1], ol2 = own_lo[2], oh0 = own_hi[0], oh1 = own_hi[1], oh2 = own_hi[2];
    // 3. the excess: a converted cell's mass beyond its new state, in equal
    //    shares to the interface cells around it. Converted cells are read
    //    from xs (kept in 2.), so a filled cell kept as interface can write
    //    its own mass and q: no race.
    auto body = KOKKOS_LAMBDA(const int i, const int j, const int k, double &ls) {
      const int t = ty(i, j, k), nt = t2(i, j, k);
      // a converted cell's excess and the number of cells that take it
      auto excess = [&](int ci, int cj, int ck, int &cnt) -> R {
        cnt = 0;
        for (int c = 1; c < 19; c++) {
          int ii, jj, kk;
          if (nb(c, ci, cj, ck, ii, jj, kk) && t2(ii, jj, kk) == INTERFACE) cnt++;
        }
        return xs_(ci, cj, ck, 0);
      };
      if (mk(i, j, k) && i >= ol0 && i < oh0 && j >= ol1 && j < oh1 && k >= ol2 && k < oh2) {
        int cnt;
        const R e = excess(i, j, k, cnt);
        if (cnt == 0) ls += e;
      }
      ty(i, j, k) = (unsigned char) nt;
      if (nt != INTERFACE) { ep_(i, j, k) = nt == FLUID ? R(1) : R(0); return; }
      const R r = R(1) + rho_(i, j, k);
      R m = t == GAS ? R(0) : t == FLUID || mk(i, j, k) == FILL ? r : ms_(i, j, k);   // a filled cell kept as interface
      R mq = one && t != GAS ? m * q(i, j, k) : R(0), mq2 = two && t != GAS ? m * q2(i, j, k) : R(0);
      // the range of what is mixed: the result stays inside it
      R qa = one && t != GAS ? q(i, j, k) : R(1e30), qb = one && t != GAS ? q(i, j, k) : R(-1e30);
      R q2a = two && t != GAS ? q2(i, j, k) : R(1e30), q2b = two && t != GAS ? q2(i, j, k) : R(-1e30);
      for (int c = 1; c < 19; c++) {
        int ii, jj, kk;
        if (!nb(c, i, j, k, ii, jj, kk) || !mk(ii, jj, kk)) continue;
        int cnt;
        const R sh = excess(ii, jj, kk, cnt) / cnt;    // cnt >= 1: this cell is one of them
        m += sh;
        if (one) { const R v = xs_(ii, jj, kk, 1); mq += sh * v; qa = v < qa ? v : qa; qb = v > qb ? v : qb; }
        if (two) { const R v = xs_(ii, jj, kk, 2); mq2 += sh * v; q2a = v < q2a ? v : q2a; q2b = v > q2b ? v : q2b; }
      }
      auto mix = [&](R mqv, R lo_, R hi_) {
        const R v = m > R(1e-6) ? mqv / m : R(0.5) * (lo_ + hi_);
        return v < lo_ ? lo_ : v > hi_ ? hi_ : v;
      };
      if (one && qb >= qa) q(i, j, k) = mix(mq, qa, qb);
      if (two && q2b >= q2a) q2(i, j, k) = mix(mq2, q2a, q2b);
      ms_(i, j, k) = m; ep_(i, j, k) = m / r;
    };
    return run_sum(range, "FlowFsExcess", dev, slost, body);
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
  auto ty = m.type; auto G = m.flux; auto rho = m.rho; auto mass_ = m.cmass;
  const bool fs = m.fs;
  const int n0 = m.nx, n1 = m.ny, n2 = m.nz;
  const bool p0 = m.p.periodic[0], p1 = m.p.periodic[1], p2 = m.p.periodic[2];
  for (int d = 0; d < (copy_back ? 4 : 3); d++) {
    // sweeps 0..2 along x, y, z (q -> tmp -> q -> tmp), then tmp -> q
    auto src = (d & 1) ? tmp : q, dst = (d & 1) ? q : tmp;
    auto src2 = (d & 1) ? tmp2 : q2, dst2 = (d & 1) ? q2 : tmp2;
    if (d == 3) {
      Kokkos::parallel_for("FlowAdvectCopy",
        LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          dst(i, j, k) = src(i, j, k);
          if (two) dst2(i, j, k) = src2(i, j, k);
        });
      break;
    }
    if (halo) halo(d, src, src2);
    Kokkos::parallel_for("FlowAdvect",
      LightMD<3>({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        if (!MeltFlow<R>::liq(ty(i, j, k))) {
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
          return MeltFlow<R>::liq(ty(ii, jj, kk));
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
          // first order at a partial cell: it can pass more than half its mass
          const bool part = ty(ia, ja, ka) == MeltFlow<R>::INTERFACE || ty(ib, jb, kb) == MeltFlow<R>::INTERFACE;
          if (!part && at(d, up ? s - 1 : s + 2, iu, ju, ku)) {
            const R a = qu - src(iu, ju, ku), b = qd - qu;
            if (a * b > R(0)) lim = R(2) * a * b / (a + b);
          }
          return qu + R(0.5) * (R(1) - Kokkos::abs(uf)) * lim - src(i, j, k);
        };
        // the cell's mass after this sweep: the later sweeps' fluxes still in it
        R ms = fs ? mass_(i, j, k) : R(1) + rho(i, j, k);
        for (int e = d + 1; e < 3; e++) ms += mass(e, 0) - mass(e, -1);
        const R ul = mass(d, -1), uh = mass(d, 0);
        if (ty(i, j, k) == MeltFlow<R>::INTERFACE) {
          // a partial cell: what flows in mixes with what is left, and the
          // result stays between the values mixed (the cell may hold next
          // to nothing, or less)
          auto mixin = [&](const typename MeltFlow<R>::W3 &sv) -> R {
            const R q0 = sv(i, j, k);
            if (ms <= R(1e-6)) return q0;
            const R im = R(1) / ms;
            R v = q0, a = q0, b = q0;
            int ii, jj, kk;
            if (ul > R(0) && at(d, -1, ii, jj, kk)) { const R qn = sv(ii, jj, kk); v += ul * (qn - q0) * im; a = qn < a ? qn : a; b = qn > b ? qn : b; }
            if (uh < R(0) && at(d, 1, ii, jj, kk)) { const R qn = sv(ii, jj, kk); v -= uh * (qn - q0) * im; a = qn < a ? qn : a; b = qn > b ? qn : b; }
            return v < a ? a : v > b ? b : v;
          };
          dst(i, j, k) = mixin(src);
          if (two) dst2(i, j, k) = mixin(src2);
          return;
        }
        const R im = R(1) / ms;
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
