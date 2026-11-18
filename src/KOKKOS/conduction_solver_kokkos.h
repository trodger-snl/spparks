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

#ifndef SPK_CONDUCTION_SOLVER_KOKKOS_H
#define SPK_CONDUCTION_SOLVER_KOKKOS_H

#include "pointers.h"
#include "spktype.h"
#include "gpu_aware_mpi.h"
#include "melt_flow_kokkos.h"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Device state of the conduction temperature source: enthalpy on a
   cell-centred tensor-product grid. Inside the lattice box the cells ARE
   the lattice sites (spacing dx); outside, cells grow geometrically out to
   the margins and the plate depth. Explicit, sub-cycled, finite volume.

   MPI: x/y follow the lattice procgrid (z is not split); each rank owns the
   cells over its own sites, and edge ranks also own the margin cells. A
   1-cell halo of T and phase is exchanged in x then y every sub-step.

   ponytail: edge ranks carry all margin cells; rebalance by cell count if
   margin-heavy decks show the imbalance.
   Laser-off intervals use RKL2 super-steps (sts stages); laser-on
   intervals stay explicit at dt_fd, unless the laser stays farther than
   `far` from the box: then the super-steps carry it as a path-averaged
   source plane S2(x, y) times the depth profile Ez.
------------------------------------------------------------------------- */

class ConductionSolverKokkos : protected Pointers {
 public:
  ConductionSolverKokkos(class SPPARKS *, class ConductionTemperatureSource *);

  // Advance the field to t; then sample() writes T for all nall sites into
  // d_T and returns the max T over owned sites (the fused-max convention).
  // d_T: the sites' temperatures (d6), read once by `init d6` at the first build.
  // d_mat: each site's material (ConductionTemperatureSource::materials), read
  // at each grid build; i2 = 6 sites take the plate whatever it says.
  // d_comp: each site's composition (d11), empty unless it changes in a run;
  // mix flow reads it at each grid build and writes it and d_mat back.
  void advance(double t, Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz,
               Kokkos::View<int*> d_af, Kokkos::View<double*> d_T,
               Kokkos::View<unsigned char*> d_mat, Kokkos::View<double*> d_comp,
               int nlocal, int nall, double dx, double ts, double tl);

  // LOOSE: flag bit, see packing. PLATE: flag bit, an i2 = 6 site's cell or
  // a cell below plate_top (the material byte M says which material).
  // UNBORN: flag bit on POWDER under ded, gas until the laser deposits it
  // PART (free surface): flag bit, the cell also carries the heat of the
  // interface cell above it; its capacity factor is Cap.
  enum Phase { GAS = 0, POWDER = 1, DENSE = 3, LOOSE = 4, PLATE = 8, UNBORN = 16, PART = 32 };

  bool ready() const { return built; }
  // dump conduction: T and phase of every cell (margins too) and the cell
  // faces, to one HDF5 file; `init grid FILE` reads it back
  void write_grid(const std::string &file, double time, bool single = false);

 private:
  class ConductionTemperatureSource *src;

  bool built = false;
  bool seeded = false;   // the run's first grid has been built (init d6 reads d6 once)
  int built_nall = -1;
  double built_zl = 0.0;
  double t_fd = 0.0;
  double dt_fd = 0.0;
  double dx = 0.0;

  // local grid: owned x cells 1..nxo, y cells 1..nyo (0 and n+1 are halo),
  // z cells 0..nz-1
  int nxo = 0, nyo = 0, nz = 0;
  bool edge[4] = {false, false, false, false};   // xlo xhi ylo yhi
  int nbr[4] = {-1, -1, -1, -1};
  double zl = 0.0;
  // global cell faces per axis (NX+1, NY+1, nz+1), this rank's first owned
  // global x/y cell, and the lattice box (m) whose cells take i2's phase
  std::vector<double> gfx, gfy, gfz;
  int gcx0 = 0, gcy0 = 0;
  double blo[3] = {0, 0, 0}, bhi[3] = {0, 0, 0};

  // State in KK_FLOAT. H is stored relative to Href = H(T0) when
  // KK_FLOAT is float (it resolves small increments near T0 that way; 0 in
  // FP64, so the FP64 arithmetic is the original).
  Kokkos::View<KK_FLOAT***> H, T, Tn;
  Kokkos::View<KK_FLOAT***> Ya, Yb, L0;   // RKL2 stage buffers (allocated on demand)
  Kokkos::View<unsigned char***> P, Pn;
  Kokkos::View<unsigned char***> M;   // material index per cell (static between builds)
  // per local cell: width and lower/upper face per axis (halo included)
  Kokkos::View<double*> wx, wy, wz, fxlo, fxhi, fylo, fyhi, fzlo, fzhi;
  Kokkos::View<KK_FLOAT*> Ex, Ey, Ez;   // per-axis Gaussian integrals, per sub-step
  Kokkos::View<KK_FLOAT**> S2;          // far: path-averaged x-y source, per super-step
  Kokkos::View<double*> d_samp;         // far: (x, y, w) path samples
  std::vector<double> h_samp;
  double zsrc = 0.0;                  // source plane: laser plane, capped at the top face
  double birth_depth = 1e300;         // ded: cells born within this of zsrc
  // Per material m (row m): H(T) - Href, its k(T) table (kN(m) points) and
  // 1/(H(i+1) - H(i)) and the k-table slopes, so the stencil multiplies;
  // H(T_birth) - Href (ded) and the liquidus at which its powder turns dense.
  using Tab2 = Kokkos::View<KK_FLOAT**, Kokkos::LayoutRight>;
  Tab2 Htab, iHtab, kTr, kVr, kSr;
  Kokkos::View<int*> kN;
  Kokkos::View<KK_FLOAT*> Hbirth;
  // evaporation: per material, each species' flux prefactor (see build);
  // evB = Lv M / R and evTb per species; ev_on = the keyword was given
  Kokkos::View<KK_FLOAT*[2]> evA, evX;    // evX: the species' mole fractions * 1 atm / ambient
  double evB[2] = {0, 0}, evTb[2] = {1, 1}, ev_hsurf = 0.0;
  bool ev_on = false;
  Kokkos::View<double*> tlm;
  std::vector<KK_FLOAT> hHtab;   // host copy of Htab, row-major
  // KK_FLOAT copies for the stencil (aliases under FP64)
  Kokkos::View<KK_FLOAT*> wxr, wyr, wzr, iwx, iwy, iwz;
  double Htab_T0 = 0.0, Htab_dT = 1.0;
  // flow lbm: the flow's block is this rank's cells of the lattice box plus
  // fh halo cells (xlo xhi ylo yhi: 2 toward a neighbour rank, else 0);
  // local cell = block cell + fb0 (the outer halo layer has no local cell).
  // fwin is the window of this rank's fluid cells at the last step
  // (fwin_loc: it had any); fwin_on: some rank had fluid.
  // The flow's fields and fq, ftmp, fXw, fX2 hold only a window of the
  // block, which follows the pool: fW, in box cells (block cell + fg0, the
  // same on every rank); fG is the last step's fluid cells and two around,
  // in box cells. fX and fborn hold the whole block.
  // tsm, Hrefm: each material's solidus and the H(T0) its H is relative to.
  std::unique_ptr<MeltFlow<KK_FLOAT>> flow;
  int fb0[3] = {0, 0, 0};
  int fh[4] = {0, 0, 0, 0};
  static int flow_halo_depth(bool hf) { return hf ? MeltFlow<KK_FLOAT>::HF_L + 1 : 2; }
  int fwin[6] = {0, 0, 0, 0, 0, 0};
  int fg0[3] = {0, 0, 0}, fW[6] = {0, 0, 0, 0, 0, 0}, fG[6] = {0, 0, 0, 0, 0, 0};
  bool fwin_on = false, fwin_loc = false;
  // halo buffers of the flow's fields (hs, hr: host staging, as hsbuf)
  template <class S> struct FlowHalo {
    Kokkos::View<S*> s, r;
    Kokkos::View<S*, MPIStagingSpace> hs, hr;
  };
  FlowHalo<KK_FLOAT> fhR;
  FlowHalo<unsigned char> fhC;
  double flow_cap_seen = 0.0;
  MeltFlow<KK_FLOAT>::W3 fq, ftmp;
  Kokkos::View<KK_FLOAT***> fX;      // mix flow: composition of the block's cells
  MeltFlow<KK_FLOAT>::W3 fXw, fX2;   // the window's copy for the sweeps, and its scratch
  // Where liquid can be: a step's stage writes its stamp to fgx(i), fgy(j),
  // fgz(k) for every cell it leaves above the lowest solidus (ts_min), so
  // the next flow_step looks for fluid only in the box of stamped rows.
  // fg_ok: the field has not changed since that stage.
  Kokkos::View<int*> fgx, fgy, fgz;
  int fstamp = 0;
  bool fg_ok = false;
  double ts_min = 0.0;
  // flow properties cards: the tables of Config::Flow::Props on the device
  Kokkos::View<double*> fp_sig, fp_r0, fp_r1, fp_eta;
  // feed surface: fborn marks new cells (unborn, or born and not melted yet); the
  // reservoir holds fed composition not yet put into the pool's top
  // (in units of x * cells); xpool, xfeed: last pool mean and fed x
  Kokkos::View<unsigned char***> fborn;
  double feed_res = 0.0, xpool = 0.0, xfeed = 0.0, feed_xl = 0.0, feed_yl = 0.0;
  // flow free_surface. Thermally a cell is metal when it is full: an
  // interface cell is gas to the stencil, and its heat rides with the full
  // cell below it (its host: Cap = 1 + the interface cell's mass, PART),
  // sharing one enthalpy; one with no full cell below does not conduct. So
  // the stencil's surface is the staircase of full cells, every capacity is
  // >= 1 (the explicit bound holds), and a film on cold solid freezes with it.
  // A partial cell that freezes becomes a whole cell or gas; fs_res is the
  // liquid owed by that rounding and by lost droplets (cells; + = to return),
  // paid back through the powder feed.
  Kokkos::View<KK_FLOAT***> Cap;          // T's shape; read where PART is set
  int kz_end = 0;                         // free surface: cell layers the step sweeps (0: all)
  Kokkos::View<KK_FLOAT**> zsf;           // per local column: the surface the laser lands on
  Kokkos::View<KK_FLOAT***> fXfed, fqfed; // block: the composition and enthalpy powder brings to a cell
  Kokkos::View<KK_FLOAT**> fdm;           // block columns: powder per thermal step (cells of liquid)
  Kokkos::View<double**> fpile;           // fs_engulf: block columns, powder held ahead of the pool (cells)
  double fs_lx = 0.0, fs_ly = 0.0;          // fs_engulf: the laser axis at the last feeding step
  bool fs_lset = false;
  int fvalid[6] = {0, 0, 0, 0, 0, 0};     // block cells whose flow type is current (the window last filled)
  double fs_res = 0.0, fs_fed = 0.0, fs_got = 0.0, fs_lost = 0.0, fs_round = 0.0, fs_rcap = 0.0, fs_rset = 0.0, rho_feed = 0.0;
  bool fs_laser = false;
  bool has_pool = false;
  Kokkos::View<int*> site_blk;       // mix flow: a site's block cell (flat; ghosts too), -1 = none
  Kokkos::View<double*> tsm, Hrefm;
  Kokkos::View<int*> site_cell;   // offset of the site's cell in T, -1 = above grid

  Kokkos::View<double*> sbuf, rbuf;
  // host staging of the packed halo (MPIStagingSpace); unused (empty) where MPI
  // can read sbuf/rbuf itself: host-accessible device memory, or GPU-aware MPI
  Kokkos::View<double*, MPIStagingSpace> hsbuf, hrbuf;
  bool stage_host = false;

  void build(Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz, Kokkos::View<int*> d_af,
             Kokkos::View<unsigned char*> d_mat, Kokkos::View<double*> d_comp,
             int nlocal, int nall, double ts, double tl);
  double H_at(int m, double Tr) const;   // material m's H(Tr) - Href
  void integrate(double t);
  void step(double h, double t0);
  void sts_step(double tau, int s, double t0);

  // These define device lambdas (KOKKOS_LAMBDA), which nvcc rejects in a
  // private or protected member function, and Stage is a kernel argument,
  // which nvcc also needs public; not part of the class interface.
 public:
  // One stage update. Explicit Euler (sts = false): H += h L(H). RKL2 stage
  // (sts = true): Yout = mu Y1 + nu Y2 + (1-mu-nu) H + mut L(Y1) + gat L0,
  // with T holding T(Y1); store_L0 saves L(Y1) (stage 1, Y1 = H).
  struct Stage {
    bool sts = false, store_L0 = false, phase = true, path = false;
    double h = 0.0, mu = 1.0, nu = 0.0, mut = 0.0, gat = 0.0;
    double Pq = 0.0, xl = 0.0, yl = 0.0;
    int stamp = 0;   // flow lbm: marks the hot cells' rows (fgx, fgy, fgz), 0 = not
  };
  void stage(const Stage &c, Kokkos::View<KK_FLOAT***> Y1,
             Kokkos::View<KK_FLOAT***> Y2, Kokkos::View<KK_FLOAT***> Yout);
  // far: fill S2 with the laser's path over [t0, t1]; returns Pq (0 if off)
  double path_source(double t0, double t1);
  void exchange();
  void reset_field(double Tr);
  void remat(Kokkos::View<unsigned char*> d_mat, int nlocal);
  // flow lbm: h seconds of flow on the pool, then the enthalpy carried with it
  void flow_step(double h, bool laser_on, double xl, double yl);
  // flow free_surface: each column's surface for the laser; and the sites
  // follow the metal (i2 5 -> 2 where a cell filled, -> 5 where it emptied)
  void fs_surface_z();
  // spatter log: lost droplets grouped into bursts (a burst ends after
  // 1 ms without loss); one line per burst of at least one cell
  double sp_t0 = -1.0, sp_last = 0.0, sp_sum = 0.0, sp_x = 0.0, sp_y = 0.0;
  void fs_spatter(double lost, double x, double y, bool flush);
  void fs_report();   // the powder and reservoir tallies, at each dump conduction and when the laser goes off
  void phase_out(Kokkos::View<int*> d_af, int nall);
  // flow lbm, several ranks: the w owned layers next to each neighbour go
  // to its halo; dirs: 1 = x, 2 = y, 3 = x then y (corners arrive)
  template <class S>
  void flow_exchange(Kokkos::View<S****> v, FlowHalo<S> &b, int w, int dirs,
                     const int *vlo = nullptr, const int *vhi = nullptr);
  // mix flow: the cells' composition back to the sites (d11 and the bin)
  void comp_out(Kokkos::View<double*> d_comp, Kokkos::View<unsigned char*> d_mat, int nall);
  void carry_field(Kokkos::View<KK_FLOAT***> Told, const std::vector<double> &fx0,
                   const std::vector<double> &fy0, const std::vector<double> &fz0);
  void seed_from_sites(Kokkos::View<double*> d_T, int nlocal);
  void seed_from_grid(const std::string &file, double t);
  // Write T for all nall sites into d_T; return the max over owned sites.
  // Each owned site's T also goes to on_owned(s, T), so the caller's own
  // per-site pass (the app's active-front list) rides along instead of
  // reading the whole lattice again.
  template <class F>
  double sample(Kokkos::View<double*> d_T, int nlocal, int nall, const F on_owned) {
    auto T_ = T; auto sc = site_cell; auto P_ = P;
    const double Ta = ambient();
    // free surface: a gas cell's T is stale (a cell the liquid left, or an
    // interface cell rounded to gas as it froze, keeps its last T); its sites
    // are air (i2 = 5) and read ambient, or the hottest of them would hold
    // the max above the solidus and the next layer would never start
    const bool gas_amb = fs_on();
    double m = 0.0;
    Kokkos::parallel_reduce("ConductionSample", nall, KOKKOS_LAMBDA(const int s, double &mx) {
      const int o = sc(s);
      const double v = o < 0 || (gas_amb && (P_.data()[o] & 3) == GAS) ? Ta : T_.data()[o];
      d_T(s) = v;
      if (s < nlocal) {
        if (v > mx) mx = v;
        on_owned(s, v);
      }
    }, Kokkos::Max<double>(m));
    return m;
  }

 private:
  double ambient() const;
  bool fs_on() const;
};

}

#endif
