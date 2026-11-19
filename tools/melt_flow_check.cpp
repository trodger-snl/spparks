/* Self-check of src/KOKKOS/melt_flow_kokkos.h against analytical flows.
   Build and run (any Kokkos build dir, e.g. build-verify):

     cmake --build build-verify --target melt_flow_check
     build-verify/melt_flow_check

   Runs every problem in double and in single precision, prints each
   measured error beside its tolerance and exits 1 if any check fails.
   Lattice units (cell 1, step 1, density 1). */

#include "melt_flow_kokkos.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace SPPARKS_NS;

namespace {

int nfail = 0;

void check(const std::string &what, double value, double tol, bool below = true)
{
  const bool ok = std::isfinite(value) && (below ? std::fabs(value) <= tol : std::fabs(value) >= tol);
  if (!ok) nfail++;
  printf("  %-4s %-58s %10.3e  (%s %.1e)\n", ok ? "ok" : "FAIL", what.c_str(), value,
         below ? "<=" : ">=", tol);
}

template <class V>
auto host(const V &v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v); }

// tolerances: double, then single (round-off of the stored f - w limits it)
template <class R> double pick(double d, double s) { return sizeof(R) == 8 ? d : s; }

// 1. Poiseuille: uniform force along x between bounce-back walls at
//    k = 0 and nz-1, i.e. at z = 1/2 and nz - 3/2. u = a (z-z0)(z1-z) / 2 nu.
template <class R>
void poiseuille()
{
  MeltFlow<R> m;
  const int nz = 34;
  m.resize(1, 1, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.nu = 1.0 / 6.0;
  m.p.accel[0] = 1e-6;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  ht(0, 0, 0) = ht(0, 0, nz - 1) = MeltFlow<R>::WALL;
  Kokkos::deep_copy(m.type.v, ht);
  for (int s = 0; s < 60000; s++) m.step();
  auto u = host(m.ux.v);
  const double z0 = 0.5, z1 = nz - 1.5, umax = m.p.accel[0] * (z1 - z0) * (z1 - z0) / (8.0 * m.p.nu);
  double err = 0.0;
  for (int k = 1; k < nz - 1; k++)
    err = std::max(err, std::fabs(u(0, 0, k) - m.p.accel[0] * (k - z0) * (z1 - k) / (2.0 * m.p.nu)));
  check("1 Poiseuille: max |u - parabola| / u_max", err / umax, pick<R>(1e-10, 2e-3));
}

// 2. Taylor-Green vortex, periodic: amplitude decays as exp(-2 nu k^2 t).
//    Diffusive scaling (U ~ 1/N, t ~ N^2), so the error falls 4x from N to 2N.
template <class R>
double taylor_green(int N, double &rate_err)
{
  MeltFlow<R> m;
  m.resize(N, N, 1);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.nu = 0.05;
  const double kw = 2.0 * M_PI / N, U = 0.04 * 32.0 / N;
  const R kr = R(kw), Ur = R(U);
  m.init(KOKKOS_LAMBDA(int i, int j, int, R &dr, R &u, R &v, R &) {
    u = Ur * Kokkos::cos(kr * i) * Kokkos::sin(kr * j);
    v = -Ur * Kokkos::sin(kr * i) * Kokkos::cos(kr * j);
    dr = R(-0.75) * Ur * Ur * (Kokkos::cos(2 * kr * i) + Kokkos::cos(2 * kr * j));
  });
  const double rate = 2.0 * m.p.nu * kw * kw;
  const int nstep = (int) std::lround(std::log(2.0) / rate);
  for (int s = 0; s < nstep; s++) m.step();
  // u is the mean of the last two steps
  const double t = nstep - 0.5, A = 0.5 * U * (std::exp(-rate * nstep) + std::exp(-rate * (nstep - 1)));
  auto u = host(m.ux.v); auto v = host(m.uy.v);
  double e2 = 0.0, dot = 0.0, nrm = 0.0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      const double su = std::cos(kw * i) * std::sin(kw * j), sv = -std::sin(kw * i) * std::cos(kw * j);
      const double du = u(i, j, 0) - A * su, dv = v(i, j, 0) - A * sv;
      e2 += du * du + dv * dv;
      dot += u(i, j, 0) * su + v(i, j, 0) * sv;
      nrm += su * su + sv * sv;
    }
  rate_err = -std::log(dot / nrm / U) / t / rate - 1.0;
  return std::sqrt(e2 / nrm) / A;
}

// 3. Thermocapillary return flow: a shallow closed layer, T linear along x,
//    no-slip bottom, free-slip top carrying the Marangoni stress tau.
//    Far from the end walls u = tau (3 z^2 - 2 h z) / (4 mu h), no net flux.
//    The stress is a force on the top cell, so that cell lags the profile by
//    up to tau / (8 mu) = u_surface / (2 h); below it the error is 2nd order.
template <class R>
void thermocapillary(int nz, double &interior, double &top, double &flux)
{
  MeltFlow<R> m;
  const int nx = 8 * nz;
  m.resize(nx, 1, nz);
  m.p.periodic[1] = true;
  m.p.nu = 1.0 / 6.0;
  m.p.marangoni = -1e-6;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) hT(i, 0, k) = R(i);
  Kokkos::deep_copy(m.T.v, hT);
  for (int s = 0; s < 140 * nz * nz; s++) m.step();
  auto u = host(m.ux.v);
  const double tau = m.p.marangoni, h = nz, us = tau * h / (4.0 * m.p.nu);
  interior = flux = 0.0;
  for (int k = 0; k < nz; k++) {
    const double z = k + 0.5, e = (u(nx / 2, 0, k) - tau * (3.0 * z * z - 2.0 * h * z) / (4.0 * m.p.nu * h)) / us;
    if (k < nz - 1) interior = std::max(interior, std::fabs(e)); else top = e;
    flux += u(nx / 2, 0, k) / (h * us);
  }
}

// 4. Buoyant slot: walls at x = -1/2 and nx - 1/2, T linear across,
//    periodic along z. w = b L^2 / (48 nu) (s - s^3), s = 2 x / L - 1.
template <class R>
void buoyant_slot()
{
  MeltFlow<R> m;
  const int nx = 32;
  m.resize(nx, 1, 1);
  m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.nu = 1.0 / 6.0;
  m.p.buoy[2] = 1e-5;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  for (int i = 0; i < nx; i++) hT(i, 0, 0) = R((i + 0.5) / nx - 0.5);
  Kokkos::deep_copy(m.T.v, hT);
  for (int s = 0; s < 40000; s++) m.step();
  auto w = host(m.uz.v);
  const double amp = m.p.buoy[2] * nx * nx / (48.0 * m.p.nu), wmax = amp * 2.0 / (3.0 * std::sqrt(3.0));
  double err = 0.0, flux = 0.0;
  for (int i = 0; i < nx; i++) {
    const double s = 2.0 * (i + 0.5) / nx - 1.0;
    err = std::max(err, std::fabs(w(i, 0, 0) - amp * (s - s * s * s)));
    flux += w(i, 0, 0);
  }
  check("4 buoyant slot: max |w - cubic| / w_max", err / wmax, pick<R>(1e-10, 1e-3));
  check("4 buoyant slot: net flux / (L w_max)", flux / (nx * wmax), pick<R>(1e-10, 1e-3));
}

// 5. Hydrostatic rest: closed box, T = z, buoyancy along z. Nothing moves.
template <class R>
void hydrostatic()
{
  MeltFlow<R> m;
  const int n = 16;
  m.resize(n, n, n);
  m.p.buoy[2] = 1e-4;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++) hT(i, j, k) = R(k);
  Kokkos::deep_copy(m.T.v, hT);
  // the force starts abruptly: the checkerboard it leaves must cancel in u
  for (int s = 0; s < 30001; s++) m.step();
  auto u = host(m.ux.v); auto v = host(m.uy.v); auto w = host(m.uz.v);
  double umax = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        umax = std::max({umax, std::fabs((double) u(i, j, k)), std::fabs((double) v(i, j, k)),
                         std::fabs((double) w(i, j, k))});
  check("5 hydrostatic: max |u| (lattice)", umax, pick<R>(1e-8, 1e-5));
  auto pw = host(m.pz.v);
  check("5 hydrostatic: the checkerboard is there (one step's |w|)", std::fabs((double) pw(3, 3, 3)), 1e-5, false);
}

// 6. Darcy: uniform liquid fraction and force, periodic. u = a / D.
template <class R>
void darcy()
{
  MeltFlow<R> m;
  m.resize(4, 4, 4);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.accel[0] = 1e-6;
  m.p.darcy = 0.01;
  const double fl = 0.5, D = m.p.darcy * (1 - fl) * (1 - fl) / (fl * fl * fl + 1e-3);
  Kokkos::deep_copy(m.fl.v, R(fl));
  for (int s = 0; s < 5000; s++) m.step();
  auto u = host(m.ux.v);
  check("6 Darcy: |u D / a - 1|", u(1, 2, 3) * D / m.p.accel[0] - 1.0, pick<R>(1e-3, 5e-3));
}

// 7. A solid block in a forced channel: mass is conserved, the solid is still.
template <class R>
void block()
{
  MeltFlow<R> m;
  const int nx = 32, ny = 8, nz = 18;
  m.resize(nx, ny, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.accel[0] = 1e-5;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  int nfluid = 0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        const bool wall = k == 0 || k == nz - 1 || (i >= 12 && i < 20 && k <= 8);
        ht(i, j, k) = wall ? MeltFlow<R>::WALL : MeltFlow<R>::FLUID;
        nfluid += !wall;
      }
  Kokkos::deep_copy(m.type.v, ht);
  for (int s = 0; s < 5000; s++) m.step();
  auto u = host(m.ux.v);
  double usolid = 0.0, ufluid = 0.0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        (ht(i, j, k) ? usolid : ufluid) = std::max(ht(i, j, k) ? usolid : ufluid,
                                                   std::fabs((double) u(i, j, k)));
  check("7 block: |mass change| per fluid cell", m.mass() / nfluid, pick<R>(1e-12, 1e-5));
  check("7 block: max |u| in the solid", usolid, 0.0);
  check("7 block: the fluid moves (max |u|)", ufluid, 1e-5, false);
}

// Pool conditions: problem 3's layer at Ti-6Al-4V values (nu 1e-6 m^2/s,
// dx 20 um, dt 2 us, dsigma/dT -2.6e-4 N/m/K, 1e6 K/m) with the Smagorinsky
// term. No analytical answer: it must stay finite; the speeds are reported.
template <class R>
void pool_stability()
{
  MeltFlow<R> m;
  const int nx = 192, nz = 24;
  m.resize(nx, 1, nz);
  m.p.periodic[1] = true;
  m.p.nu = 0.005;
  m.p.smag = 0.17;
  m.p.marangoni = -3.17e-5;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) hT(i, 0, k) = R(20.0 * i);
  Kokkos::deep_copy(m.T.v, hT);
  for (int s = 0; s < 100000; s++) m.step();
  auto u = host(m.ux.v); auto w = host(m.uz.v);
  double umax = 0.0;
  bool finite = true;
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
      finite = finite && std::isfinite(u(i, 0, k)) && std::isfinite(w(i, 0, k));
      umax = std::max(umax, std::hypot((double) u(i, 0, k), (double) w(i, 0, k)));
    }
  check("pool conditions, 1e5 steps: max |u| (lattice; 0.1 = 1 m/s)", finite ? umax : NAN, 0.3);
}

// A 3-D pool at Ti-6Al-4V values through the SI driver: a half-ellipsoid of
// liquid (0.8 mm across, 0.24 mm deep, dx 20 um) under a Gaussian hot spot
// 1200 K above the liquidus, nu 8.5e-7 m^2/s, dsigma/dT -2.6e-4 N/m/K,
// rho 4100. 20 ms in thermal-sized steps of 5 us. No analytical answer.
// It must stay finite and keep the lattice speed in range; the speed should
// be of the inertial Marangoni scale, (dsigma^2 dT^2 / (rho mu L))^(1/3)
// = 2.6 m/s, and flow outward along the surface (dsigma/dT < 0).
template <class R>
double pool_si(double u_ref)
{
  MeltFlow<R> m;
  const int nx = 64, nz = 24;
  m.resize(nx, nx, nz);
  m.ph.dx = 20e-6; m.ph.nu = 8.5e-7; m.ph.dsdT_rho = -2.6e-4 / 4100.0;
  m.ph.beta_g = 1e-5 * 9.81; m.ph.Tref = 1928.0; m.ph.darcy = 1e4; m.ph.u_ref = u_ref;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  auto hT = Kokkos::create_mirror_view(m.T.v);
  auto hf = Kokkos::create_mirror_view(m.fl.v);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < nx; j++)
      for (int k = 0; k < nz; k++) {
        const double x = i - 31.5, y = j - 31.5, d = nz - 0.5 - k;   // depth below the surface
        const double e = (x * x + y * y) / (20.0 * 20.0) + d * d / (12.0 * 12.0);
        ht(i, j, k) = e < 1.0 ? MeltFlow<R>::FLUID : MeltFlow<R>::WALL;
        hT(i, j, k) = R(1928.0 + 1200.0 * std::exp(-(x * x + y * y) / (12.0 * 12.0) - d * d / (8.0 * 8.0)));
        hf(i, j, k) = R(std::min(1.0, std::max(0.05, 4.0 * (1.0 - e))));   // mushy near the wall
      }
  Kokkos::deep_copy(m.type.v, ht); Kokkos::deep_copy(m.T.v, hT); Kokkos::deep_copy(m.fl.v, hf);
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, nx, nz};
  double lat_max = 0.0, u_late = 0.0;
  int nmax = 0, bad = 0;
  for (int s = 0; s < 4000 && !bad; s++) {
    const int n = m.run(5e-6, lo, hi);
    if (n < 0 || !std::isfinite(m.umax)) { bad = 1; break; }
    nmax = std::max(nmax, n);
    if (s >= 200) lat_max = std::max(lat_max, m.umax * m.dt / m.ph.dx);
    if (s >= 3000) u_late = std::max(u_late, m.umax);
  }
  auto u = host(m.ux.v);
  // surface flow at half the pool radius on +x: outward means u > 0
  const double us = bad ? NAN : u(42, 32, nz - 1) * m.ph.dx / m.dt;
  if (u_ref != 1.0) {
    // started 4x too coarse: the cap must have doubled, f been rescaled
    check("pool from u_ref 0.25: the step adapted (u_cap, m/s) >= 1", bad ? NAN : m.u_cap, 1.0, false);
    check("pool from u_ref 0.25: max lattice speed after start-up", bad ? NAN : lat_max, 0.2);
    return bad ? NAN : u_late;
  }
  check("pool (SI driver, Ti-6Al-4V, 20 ms): stays finite, steps <= nmax", bad, 0.0);
  check("pool: max lattice speed after start-up", bad ? NAN : lat_max, 0.2);
  check("pool: max speed in the last 5 ms (m/s), >= 0.5", bad ? NAN : u_late, 0.5, false);
  check("pool: max speed in the last 5 ms (m/s), <= 10", bad ? NAN : u_late, 10.0);
  check("pool: surface flows outward, u at r/2 (m/s) > 0.1", us, 0.1, false);
  printf("       pool: %d flow steps per 5 us at most, u_cap %.3g m/s, dt %.3g s\n", nmax, m.u_cap, m.dt);
  return bad ? NAN : u_late;
}

// ---- scalar advection (melt_advect) ----

template <class R>
void advect_all(const MeltFlow<R> &m, Kokkos::View<R***> q, Kokkos::View<R***> tmp)
{
  const int lo[3] = {0, 0, 0}, hi[3] = {m.nx, m.ny, m.nz};
  melt_advect(m, q, tmp, lo, hi);
}

// 8. A uniform field in a real (slightly compressible) LBM flow stays uniform.
template <class R>
void advect_uniform()
{
  MeltFlow<R> m;
  const int N = 32;
  m.resize(N, N, 1);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.nu = 0.05;
  const R kr = R(2.0 * M_PI / N);
  m.init(KOKKOS_LAMBDA(int i, int j, int, R &, R &u, R &v, R &) {
    u = R(0.08) * Kokkos::cos(kr * i) * Kokkos::sin(kr * j);
    v = R(-0.08) * Kokkos::sin(kr * i) * Kokkos::cos(kr * j);
  });
  Kokkos::View<R***> q("q", N, N, 1), tmp("tmp", N, N, 1);
  Kokkos::deep_copy(q, R(0.37));
  double div = 0.0;
  for (int s = 0; s < 100; s++) {
    const int lo[3] = {0, 0, 0}, hi[3] = {N, N, 1};
    m.step(); m.step(lo, hi, false);
    advect_all(m, q, tmp);
  }
  auto h = host(q); auto u = host(m.ux.v); auto v = host(m.uy.v);
  double err = 0.0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      err = std::max(err, std::fabs(h(i, j, 0) - 0.37));
      div = std::max(div, std::fabs(0.5 * (u((i + 1) % N, j, 0) - u((i + N - 1) % N, j, 0) +
                                           v(i, (j + 1) % N, 0) - v(i, (j + N - 1) % N, 0))));
    }
  check("8 uniform field in an LBM flow: max |q - q0|", err, pick<R>(1e-13, 1e-5));
  check("8   (that flow is not divergence-free: max |div u|)", div, 1e-7, false);
}

// 9. Step and Gaussian carried once around a periodic line by a uniform flow.
template <class R>
double advect_line(int N, bool step, double &dsum, double &over, double &shift)
{
  MeltFlow<R> m;
  m.resize(N, 1, 1);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.set_flux(0.1, 0, 0);
  Kokkos::View<R***> q("q", N, 1, 1), tmp("tmp", N, 1, 1);
  // the start profile is kept apart: on a host backend a mirror view is q itself
  std::vector<R> q0(N);
  auto hq = Kokkos::create_mirror_view(q);
  const double sig = N / 16.0;
  for (int i = 0; i < N; i++)
    hq(i, 0, 0) = q0[i] = step ? R(i >= N / 4 && i < N / 2)
                               : R(std::exp(-0.5 * (i - N / 2.0) * (i - N / 2.0) / (sig * sig)));
  Kokkos::deep_copy(q, hq);
  for (int s = 0; s < 10 * N; s++) advect_all(m, q, tmp);
  auto h = host(q);
  double s0 = 0, s1 = 0, e2 = 0, n2 = 0, lo = 1e300, hi = -1e300, cr = 0, ci = 0, c0r = 0, c0i = 0;
  for (int i = 0; i < N; i++) {
    const double a = q0[i], b = h(i, 0, 0), ph = 2.0 * M_PI * i / N;
    s0 += a; s1 += b; e2 += (b - a) * (b - a); n2 += a * a;
    lo = std::min(lo, b); hi = std::max(hi, b);
    cr += b * std::cos(ph); ci += b * std::sin(ph); c0r += a * std::cos(ph); c0i += a * std::sin(ph);
  }
  dsum = (s1 - s0) / s0;
  over = std::max(hi - 1.0, -lo);
  shift = std::remainder(std::atan2(ci, cr) - std::atan2(c0i, c0r), 2.0 * M_PI) * N / (2.0 * M_PI);
  return std::sqrt(e2 / n2);
}

// 10. A cube of q = 1 carried by a diagonal uniform flow in a periodic box.
template <class R>
void advect_cube()
{
  MeltFlow<R> m;
  const int N = 32;
  m.resize(N, N, N);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.set_flux(0.08, 0.06, 0.04);
  Kokkos::View<R***> q("q", N, N, N), tmp("tmp", N, N, N);
  auto h0 = Kokkos::create_mirror_view(q);
  for (int i = 8; i < 16; i++)
    for (int j = 8; j < 16; j++)
      for (int k = 8; k < 16; k++) h0(i, j, k) = R(1);
  Kokkos::deep_copy(q, h0);
  for (int s = 0; s < 400; s++) advect_all(m, q, tmp);
  auto h = host(q);
  // centre of mass per axis from the phase of the first Fourier mode
  // (the box is periodic and the cube ends across its edge in y)
  double sum = 0, lo = 1e300, hi = -1e300, re[3] = {0, 0, 0}, im[3] = {0, 0, 0};
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++) {
        const double b = h(i, j, k);
        const int x[3] = {i, j, k};
        sum += b; lo = std::min(lo, b); hi = std::max(hi, b);
        for (int d = 0; d < 3; d++) {
          re[d] += b * std::cos(2.0 * M_PI * x[d] / N); im[d] += b * std::sin(2.0 * M_PI * x[d] / N);
        }
      }
  check("10 cube, diagonal flow: |sum change| / sum", sum / 512.0 - 1.0, pick<R>(1e-12, 1e-5));
  check("10 cube: over/undershoot beyond [0, 1]", std::max(hi - 1.0, -lo), pick<R>(1e-12, 1e-6));
  const double want[3] = {11.5 + 0.08 * 400, 11.5 + 0.06 * 400, 11.5 + 0.04 * 400};
  double err = 0.0;
  for (int d = 0; d < 3; d++)
    err = std::max(err, std::fabs(std::remainder(std::atan2(im[d], re[d]) * N / (2.0 * M_PI) - want[d], (double) N)));
  check("10 cube: centre of mass error (cells)", err, 0.5);
}

// 11. A Gaussian carried and diffused: variance grows as s0^2 + 2 alpha t.
//     The diffusion is a plain explicit step here; melt_advect must not add
//     more than a little of its own.
template <class R>
void advect_diffuse()
{
  MeltFlow<R> m;
  const int N = 256, nstep = 400;
  const double al = 0.05, s0 = 6.0, x0 = 60.0, uv = 0.2;
  m.resize(N, 1, 1);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.set_flux(uv, 0, 0);
  Kokkos::View<R***> q("q", N, 1, 1), tmp("tmp", N, 1, 1);
  auto h0 = Kokkos::create_mirror_view(q);
  for (int i = 0; i < N; i++) h0(i, 0, 0) = R(std::exp(-0.5 * (i - x0) * (i - x0) / (s0 * s0)));
  Kokkos::deep_copy(q, h0);
  const R a = R(al);
  for (int s = 0; s < nstep; s++) {
    advect_all(m, q, tmp);
    Kokkos::parallel_for("diffuse", N, KOKKOS_LAMBDA(const int i) {
      tmp(i, 0, 0) = q(i, 0, 0) + a * (q((i + 1) % N, 0, 0) - R(2) * q(i, 0, 0) + q((i + N - 1) % N, 0, 0));
    });
    Kokkos::deep_copy(q, tmp);
  }
  auto h = host(q);
  double sum = 0, mean = 0, var = 0;
  for (int i = 0; i < N; i++) { sum += h(i, 0, 0); mean += i * (double) h(i, 0, 0); }
  mean /= sum;
  for (int i = 0; i < N; i++) var += (i - mean) * (i - mean) * h(i, 0, 0);
  var /= sum;
  check("11 carried + diffused Gaussian: width / sqrt(s0^2 + 2 alpha t) - 1",
        std::sqrt(var / (s0 * s0 + 2.0 * al * nstep)) - 1.0, 2e-2);
  check("11 its centre: |x - (x0 + u t)| (cells)", mean - (x0 + uv * nstep), 0.5);
}

// 12. Conservation where the pool is awkward: a forced flow round a solid
//     block under a free surface with a step in it (GAS over half the
//     length) and a pillar touching the block only along an edge. Checked:
//     the flow's mass; the face fluxes against each cell's change of rho;
//     the sum of (1 + rho) q for a two-valued q; a uniform q; q's range.
template <class R>
void advect_conserve()
{
  MeltFlow<R> m;
  const int nx = 32, ny = 8, nz = 18;
  m.resize(nx, ny, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.accel[0] = 2e-5; m.p.accel[2] = -1e-5;
  m.p.nu = 0.02;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        const bool wall = k == 0 || (i >= 12 && i < 20 && k <= 8) || (i == 20 && j < 4 && k == 9);
        const bool gas = k >= 15 || (i < 16 && k >= 12);
        ht(i, j, k) = wall ? MeltFlow<R>::WALL : gas ? MeltFlow<R>::GAS : MeltFlow<R>::FLUID;
      }
  Kokkos::deep_copy(m.type.v, ht);
  Kokkos::View<R***> q("q", nx, ny, nz), tmp("tmp", nx, ny, nz), q2("q2", nx, ny, nz), tmp2("tmp2", nx, ny, nz);
  auto hq = Kokkos::create_mirror_view(q);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) hq(i, j, k) = R(k >= 9);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(q2, R(0.37));
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz};
  // sum of (1 + rho) q over the fluid, and its size
  auto total = [&](double &sum) {
    auto h = host(q); auto r = host(m.rho.v);
    sum = 0.0;
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
          if (ht(i, j, k) == MeltFlow<R>::FLUID) sum += (1.0 + r(i, j, k)) * h(i, j, k);
  };
  for (int s = 0; s < 50; s++) m.step();       // rho settles to the forcing first
  double s0, s1, ferr = 0.0;
  total(s0);
  for (int s = 0; s < 1500; s++) {
    // a copy of its own: on a host backend host() is m.rho.v itself
    auto r0 = Kokkos::create_mirror(m.rho.v);
    if (s % 500 == 499) Kokkos::deep_copy(r0, m.rho.v);
    m.step(); m.step(lo, hi, false);
    if (s % 500 == 499) {
      // the six faces against rho's change over the two steps
      auto r1 = host(m.rho.v); auto G = host(m.flux.v);
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++) {
            if (ht(i, j, k) != MeltFlow<R>::FLUID) continue;
            const int x[3] = {i, j, k}, n[3] = {nx, ny, nz};
            double out = 0.0;
            for (int d = 0; d < 3; d++)
              for (int sd = 0; sd < 2; sd++) {
                int y[3] = {i, j, k};
                y[d] += sd ? 1 : -1;
                if (d < 2) y[d] = (y[d] + n[d]) % n[d];
                else if (y[d] < 0 || y[d] >= nz) continue;
                if (ht(y[0], y[1], y[2]) != MeltFlow<R>::FLUID) {
                  ferr = std::max(ferr, std::fabs((double) G(x[0], x[1], x[2], 2 * d + sd)));
                  continue;
                }
                out += G(x[0], x[1], x[2], 2 * d + sd) - G(y[0], y[1], y[2], 2 * d + 1 - sd);
              }
            ferr = std::max(ferr, std::fabs(out - ((double) r0(i, j, k) - (double) r1(i, j, k))));
          }
    }
    melt_advect(m, q, tmp, q2, tmp2, lo, hi, true);
  }
  total(s1);
  auto h = host(q); auto h2 = host(q2); auto u = host(m.ux.v);
  int nfluid = 0;
  double qlo = 1e300, qhi = -1e300, uni = 0.0, umax = 0.0, moved = 0.0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        if (ht(i, j, k) != MeltFlow<R>::FLUID) continue;
        nfluid++;
        qlo = std::min(qlo, (double) h(i, j, k)); qhi = std::max(qhi, (double) h(i, j, k));
        uni = std::max(uni, std::fabs(h2(i, j, k) - 0.37));
        umax = std::max(umax, std::fabs((double) u(i, j, k)));
        moved += std::fabs(h(i, j, k) - (k >= 9));
      }
  check("12 stepped surface: |mass change| per fluid cell", m.mass() / nfluid, pick<R>(1e-12, 1e-5));
  check("12 face fluxes - a cell's loss of rho (max)", ferr, pick<R>(1e-14, 2e-6));
  check("12 sum of (1 + rho) q: |change| / sum", s1 / s0 - 1.0, pick<R>(1e-12, 1e-5));
  check("12 uniform q stays: max |q - q0|", uni, pick<R>(1e-13, 1e-5));
  check("12 q stays in [0, 1]: over/undershoot", std::max(qhi - 1.0, -qlo), pick<R>(1e-12, 1e-5));
  check("12   (the flow moves: max |u|)", umax, 1e-3, false);
  check("12   (and carries q: mean |q - q start|)", moved / nfluid, 1e-2, false);
}

template <class R>
void advect_suite()
{
  advect_uniform<R>();
  advect_conserve<R>();
  double ds, ov, sh, d2, o2, s2;
  advect_line<R>(128, true, ds, ov, sh);
  check("9 step, one pass: |sum change| / sum", ds, pick<R>(1e-12, 1e-5));
  check("9 step: over/undershoot beyond [0, 1]", ov, pick<R>(1e-12, 1e-6));
  check("9 step: centre shift (cells)", sh, 0.5);
  const double e64 = advect_line<R>(64, false, ds, ov, sh), e128 = advect_line<R>(128, false, d2, o2, s2);
  check("9 Gaussian N=128: |sum change| / sum", d2, pick<R>(1e-12, 1e-5));
  check("9 Gaussian N=128: over/undershoot", o2, pick<R>(1e-12, 1e-6));
  check("9 Gaussian N=128: centre shift (cells)", s2, 0.5);
  check("9 Gaussian N=128: L2 error / norm", e128, 5e-2);
  check("9 Gaussian: error(64) / error(128) >= 2", e64 / e128, 2.0, false);
  advect_cube<R>();
  advect_diffuse<R>();
}

// 13. Per-cell properties (props). A liquid layer periodic in x and y under
//     a free surface, T a wave along x plus a ramp in z: Marangoni, buoyancy
//     and viscosity at once. With sig = T, dens = -1e-7 (T - Tref) under a
//     buoyancy 1e7 times larger, and visc = 2 on half the viscosity, it must
//     give the flow of the plain model (but for the Marangoni force's
//     1 / (1 + dens), 1e-5 here). Then
//     a wall in the layer: the Marangoni difference must not reach into it.
template <class R>
void props_same()
{
  const int nx = 32, nz = 8;
  Kokkos::View<R***> u[2];
  double wall = 0.0;
  for (int v = 0; v < 3; v++) {
    MeltFlow<R> m;
    m.props = v > 0;
    m.resize(nx, 4, nz);
    m.p.periodic[0] = m.p.periodic[1] = true;
    m.p.nu = v ? 0.05 : 0.1; m.p.marangoni = 2e-6; m.p.buoy[2] = v ? 1.0 : 1e-7; m.p.Tref = 3.0;
    auto hT = Kokkos::create_mirror_view(m.T.v);
    auto ht = Kokkos::create_mirror_view(m.type.v);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < 4; j++)
        for (int k = 0; k < nz; k++) {
          hT(i, j, k) = R(100.0 * std::sin(2.0 * M_PI * (i + 0.5) / nx) + 10.0 * k);
          // v = 2: a wall column at i = 8 whose "T" is far off
          ht(i, j, k) = v == 2 && i == 8 ? MeltFlow<R>::WALL : MeltFlow<R>::FLUID;
          if (v == 2 && i == 8) hT(i, j, k) = R(1e6);
        }
    Kokkos::deep_copy(m.T.v, hT); Kokkos::deep_copy(m.type.v, ht);
    if (v) {
      auto hd = Kokkos::create_mirror_view(m.dens.v);
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < 4; j++)
          for (int k = 0; k < nz; k++) hd(i, j, k) = R(-1e-7) * (hT(i, j, k) - R(3.0));
      Kokkos::deep_copy(m.sig.v, hT); Kokkos::deep_copy(m.dens.v, hd); Kokkos::deep_copy(m.visc.v, R(2));
    }
    for (int s = 0; s < 400; s++) m.step();
    if (v < 2) u[v] = m.ux.v;
    else {
      auto h = host(m.ux.v);
      for (int i = 0; i < nx; i++) wall = std::max(wall, (double) std::fabs(h(i, 0, nz - 1)));
    }
  }
  auto a = host(u[0]), b = host(u[1]);
  double d = 0.0, top = 0.0;
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
      d = std::max(d, (double) std::fabs(a(i, 0, k) - b(i, 0, k)));
      top = std::max(top, (double) std::fabs(a(i, 0, k)));
    }
  check("13 props as the plain model: max |u - u plain| / max |u|", d / top, 1e-4);
  check("13   (the flow moves: max |u|)", top, 1e-4, false);
  // one-sided next to the wall: the 1e6 in it would give |u| of order 1
  check("13 props, wall with sig 1e6: max surface |u| / plain", wall / top, 3.0);
}

// ---- Free surface (fs) ------------------------------------------------

// is the interface layer closed: no FLUID cell touches GAS (count of those that do)
template <class R>
int fs_open(const MeltFlow<R> &m)
{
  auto t = host(m.type.v);
  int bad = 0;
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++) {
        if (t(i, j, k) != MeltFlow<R>::FLUID) continue;
        for (int q = 1; q < 19; q++) {
          int c[3]; MeltFlow<R>::vel(q, c[0], c[1], c[2]);
          int x[3] = {i + c[0], j + c[1], k + c[2]};
          const int n[3] = {m.nx, m.ny, m.nz};
          bool in = true;
          for (int d = 0; d < 3; d++) {
            if (x[d] < 0 || x[d] >= n[d]) { if (m.p.periodic[d]) x[d] = (x[d] + n[d]) % n[d]; else in = false; }
          }
          if (in && t(x[0], x[1], x[2]) == MeltFlow<R>::GAS) { bad++; break; }
        }
      }
  return bad;
}

// 14. A flat pool at rest under gravity, the surface half-way through a
//     cell layer: it stays flat and still, and keeps its mass.
template <class R>
void fs_rest()
{
  MeltFlow<R> m;
  const int nx = 12, ny = 6, nz = 16;
  m.fs = true; m.fs_tau = 0.0;
  m.resize(nx, ny, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.accel[2] = -1e-4;
  m.p.nu = 0.05;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  auto hm = Kokkos::create_mirror_view(m.cmass.v);
  auto he = Kokkos::create_mirror_view(m.eps.v);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        ht(i, j, k) = k == 0 ? MeltFlow<R>::WALL : k < 8 ? MeltFlow<R>::FLUID
                    : k == 8 ? MeltFlow<R>::INTERFACE : MeltFlow<R>::GAS;
        hm(i, j, k) = he(i, j, k) = k == 8 ? R(0.5) : k < 8 ? R(1) : R(0);
      }
  Kokkos::deep_copy(m.type.v, ht); Kokkos::deep_copy(m.cmass.v, hm); Kokkos::deep_copy(m.eps.v, he);
  const double m0 = m.total_mass();
  double lost = 0.0;
  for (int s = 0; s < 30000; s++) { m.step(); lost += m.convert(); }
  auto u = host(m.ux.v); auto v = host(m.uy.v); auto w = host(m.uz.v);
  auto e = host(m.eps.v); auto t = host(m.type.v);
  double umax = 0.0, hlo = 1e300, hhi = -1e300;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      double h = 0.0;
      for (int k = 0; k < nz; k++) {
        h += e(i, j, k);
        umax = std::max({umax, std::fabs((double) u(i, j, k)), std::fabs((double) v(i, j, k)),
                         std::fabs((double) w(i, j, k))});
      }
      hlo = std::min(hlo, h); hhi = std::max(hhi, h);
    }
  check("14 fs at rest: max |u| (lattice)", umax, pick<R>(1e-8, 1e-5));
  check("14 fs at rest: surface height, max - min (cells)", hhi - hlo, pick<R>(1e-8, 1e-4));
  check("14 fs at rest: height - 7.5 (cells; the liquid is compressed)", hhi - 7.5, 2e-2);
  check("14 fs at rest: |mass change| / mass", (m.total_mass() + lost) / m0 - 1.0, pick<R>(1e-12, 1e-5));
}

// 15, 16. Dam break: a column of liquid collapses along a floor between
//     walls, with many cells filling and emptying. The liquid's mass, the
//     closed interface layer, and the carried fields: a uniform one stays
//     uniform, a two-valued one keeps its total and its range.
template <class R>
void fs_dam()
{
  MeltFlow<R> m;
  const int nx = 48, ny = 4, nz = 32;
  m.fs = true; m.fs_tau = 0.0;
  m.resize(nx, ny, nz);
  m.p.periodic[1] = true;
  m.p.accel[2] = -5e-5;
  m.p.nu = 0.02;
  m.p.smag = 0.17;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        ht(i, j, k) = k == 0 ? MeltFlow<R>::WALL : (i < 16 && k < 24) ? MeltFlow<R>::FLUID : MeltFlow<R>::GAS;
  Kokkos::deep_copy(m.type.v, ht);
  m.fs_repair();
  Kokkos::View<R***> q("q", nx, ny, nz), tmp("tmp", nx, ny, nz), q2("q2", nx, ny, nz), tmp2("tmp2", nx, ny, nz);
  auto hq = Kokkos::create_mirror_view(q);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) hq(i, j, k) = R(k >= 12);
  Kokkos::deep_copy(q, hq);
  Kokkos::deep_copy(q2, R(0.37));
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz};
  // sum of mass x q over the liquid; the liquid's extent along x
  auto total = [&](double &sum, int &reach) {
    auto h = host(q); auto r = host(m.rho.v); auto ms = host(m.cmass.v); auto t = host(m.type.v);
    sum = 0.0; reach = 0;
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++) {
          if (t(i, j, k) == MeltFlow<R>::FLUID) sum += (1.0 + r(i, j, k)) * h(i, j, k);
          if (t(i, j, k) == MeltFlow<R>::INTERFACE) sum += (double) ms(i, j, k) * h(i, j, k);
          if (MeltFlow<R>::liq(t(i, j, k))) reach = std::max(reach, i);
        }
  };
  const double m0 = m.total_mass();
  double s0, s1, lost = 0.0, umax = 0.0;
  int r0, r1, open = 0;
  total(s0, r0);
  for (int s = 0; s < 8000; s++) {
    m.step();
    melt_advect(m, q, tmp, q2, tmp2, lo, hi, true);
    lost += m.convert(lo, hi, q, q2);
    if (s % 50 == 49) open += fs_open(m);
    if (s % 500 == 499) {
      umax = std::max(umax, std::sqrt((double) m.max_u2(lo, hi)));
    }
  }
  total(s1, r1);
  auto h = host(q); auto h2 = host(q2); auto t = host(m.type.v);
  double qlo = 1e300, qhi = -1e300, uni = 0.0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        if (!MeltFlow<R>::liq(t(i, j, k))) continue;
        qlo = std::min(qlo, (double) h(i, j, k)); qhi = std::max(qhi, (double) h(i, j, k));
        uni = std::max(uni, std::fabs(h2(i, j, k) - 0.37));
      }
  check("15 dam break: |mass change| / mass", (m.total_mass() + lost) / m0 - 1.0, pick<R>(1e-12, 1e-5));
  check("15 dam break: lone droplets' mass / mass", lost / m0, 1e-3);
  check("15 dam break: FLUID cells touching GAS", open, 0.0);
  check("15 dam break: max |u| (lattice) <= 0.3", umax, 0.3);
  check("15   (the front runs: cells along x)", r1 - r0, 16.0, false);
  // not to round-off: a cell can be overdrawn (pass more mass than it holds),
  // and its fields are then held between the values mixed (1.6e-5; 6.4e-4
  // under the removed rule that liquid enters a gas cell sideways only over
  // metal, a9ac6675: more cells were overdrawn as the column fell)
  check("16 moving surface, sum of mass x q: |change| / sum", s1 / s0 - 1.0, 1e-4);
  check("16 uniform q stays: max |q - q0|", uni, pick<R>(1e-13, 1e-5));
  check("16 q stays in [0, 1]: over/undershoot", std::max(qhi - 1.0, -qlo), pick<R>(1e-12, 1e-5));
}

// A free surface of height h(i) (cells, from the block's bottom; the row
// k = 0 is wall) in a block uniform along y; returns the liquid's mass.
template <class R, class F>
void fs_shape(MeltFlow<R> &m, const F h)
{
  auto ht = Kokkos::create_mirror_view(m.type.v);
  auto hm = Kokkos::create_mirror_view(m.cmass.v);
  auto he = Kokkos::create_mirror_view(m.eps.v);
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++) {
        const double f = std::min(1.0, std::max(0.0, h(i) - k));
        ht(i, j, k) = k == 0 ? MeltFlow<R>::WALL : f >= 1.0 ? MeltFlow<R>::FLUID
                    : f > 0.0 ? MeltFlow<R>::INTERFACE : MeltFlow<R>::GAS;
        hm(i, j, k) = he(i, j, k) = k == 0 ? R(0) : R(f);
      }
  Kokkos::deep_copy(m.type.v, ht); Kokkos::deep_copy(m.cmass.v, hm); Kokkos::deep_copy(m.eps.v, he);
  m.fs_repair();
}

// the surface's height per column i (at j = 0), as fs_shape took it
template <class R>
std::vector<double> fs_height(const MeltFlow<R> &m)
{
  auto e = host(m.eps.v);
  std::vector<double> h(m.nx, 1.0);
  for (int i = 0; i < m.nx; i++)
    for (int k = 1; k < m.nz; k++) h[i] += e(i, 0, k);
  return h;
}

// 17. Capillary wave: a standing wave of one wavelength on a deep layer,
//     periodic, no gravity. omega^2 = sigma k^3 tanh(k H) / rho, and it
//     decays as exp(-2 nu k^2 t).
template <class R>
void fs_wave(double floor)
{
  MeltFlow<R> m;
  const int nx = 64, nz = 48;
  const double H = 32.0, a0 = 1.0, kw = 2.0 * M_PI / nx;
  m.fs = true; m.fs_tau = floor;
  m.resize(nx, 2, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.nu = 0.005;
  m.p.sigma = 0.005;
  fs_shape(m, [&](int i) { return 1.0 + H + a0 * std::cos(kw * (i + 0.5)); });
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 2, nz};
  const double m0 = m.total_mass();
  // the wave's amplitude every 20 steps: its zero crossings and extrema
  std::vector<double> A;
  const int every = 20, nstep = 7000;
  double lost = 0.0;
  for (int s = 0; s <= nstep; s++) {
    if (s % every == 0) {
      const auto h = fs_height(m);
      double c = 0.0;
      for (int i = 0; i < nx; i++) c += h[i] * std::cos(kw * (i + 0.5));
      A.push_back(2.0 * c / nx);
    }
    m.surface(lo, hi); m.step(); lost += m.convert();
  }
  std::vector<double> zero, ext;
  for (size_t n = 1; n < A.size(); n++)
    if (A[n - 1] * A[n] < 0.0) zero.push_back(every * (n - 1 + A[n - 1] / (A[n - 1] - A[n])));
  for (size_t n = 1; n + 1 < A.size(); n++)
    if (std::fabs(A[n]) > std::fabs(A[n - 1]) && std::fabs(A[n]) >= std::fabs(A[n + 1])) ext.push_back(std::fabs(A[n]));
  const double gam = 2.0 * m.p.nu * kw * kw;
  const double w0 = std::sqrt(m.p.sigma * kw * kw * kw * std::tanh(kw * H)), w = std::sqrt(w0 * w0 - gam * gam);
  const bool got = zero.size() >= 3 && ext.size() >= 2;
  const double period = got ? zero[2] - zero[0] : NAN;
  const double eper = period * w / (2.0 * M_PI) - 1.0, edec = got ? std::log(ext[0] / ext[1]) / (0.5 * period) / gam - 1.0 : NAN;
  if (floor > 0.0) {
    // what the stabiliser costs (not a check: the floor is not physics)
    printf("       17 with the interface floor tau = %g: period %+.1f%%, decay rate %+.1f%%\n", floor, 100 * eper, 100 * edec);
    return;
  }
  check("17 capillary wave: period / (2 pi / omega) - 1", eper, 5e-2);
  check("17 capillary wave: decay rate / (2 nu k^2) - 1", edec, 0.2);
  check("17 capillary wave: |mass change| / mass", (m.total_mass() + lost) / m0 - 1.0, pick<R>(1e-12, 1e-5));
}

// 18. A Gaussian gas pressure on a layer, periodic, no gravity: at rest the
//     surface's curvature carries it, sigma h'' / (1 + h'^2)^(3/2) = p - mean.
template <class R>
void fs_load()
{
  MeltFlow<R> m;
  const int nx = 64, nz = 24;
  const double H = 12.0, P0 = 2e-4, wd = 6.0;
  m.fs = true; m.fs_tau = 0.0;
  m.resize(nx, 2, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.nu = 0.1;
  m.p.sigma = 0.01;
  fs_shape(m, [&](int) { return 1.0 + H + 0.5; });
  std::vector<double> pr(nx);
  auto hp = Kokkos::create_mirror_view(m.pgas);
  double mean = 0.0;
  for (int i = 0; i < nx; i++) {
    pr[i] = P0 * std::exp(-(i - 31.5) * (i - 31.5) / (wd * wd));
    mean += pr[i] / nx;
    hp(i, 0) = hp(i, 1) = R(pr[i]);
  }
  Kokkos::deep_copy(m.pgas, hp);
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 2, nz};
  for (int s = 0; s < 40000; s++) { m.surface(lo, hi); m.step(); m.convert(); }
  // the small-slope solution: integrate h'' = (p - mean) / sigma twice, zero mean
  std::vector<double> g(nx, 0.0), e(nx, 0.0);
  double sl = 0.0, gm = 0.0, em = 0.0;
  for (int i = 1; i < nx; i++) { sl += 0.5 * (pr[i - 1] + pr[i] - 2.0 * mean) / m.p.sigma; g[i] = sl; }
  for (int i = 0; i < nx; i++) gm += g[i] / nx;
  for (int i = 1; i < nx; i++) e[i] = e[i - 1] + 0.5 * (g[i - 1] + g[i]) - gm;
  for (int i = 0; i < nx; i++) em += e[i] / nx;
  const auto h = fs_height(m);
  double hm = 0.0, err = 0.0, depth = 0.0;
  for (int i = 0; i < nx; i++) hm += h[i] / nx;
  for (int i = 0; i < nx; i++) {
    err = std::max(err, std::fabs((h[i] - hm) - (e[i] - em)));
    depth = std::max(depth, em - e[i]);
  }
  auto u = host(m.px.v); auto w = host(m.pz.v);
  double umax = 0.0;
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) umax = std::max({umax, std::fabs((double) u(i, 0, k)), std::fabs((double) w(i, 0, k))});
  check("18 pressure load: max |h - Young-Laplace| / depth", err / depth, 5e-2);
  check("18   (the surface is pushed in under the load: depth, cells)", hm - h[31], 0.5, false);
  check("18 pressure load: at rest, max |u| (lattice)", umax, 1e-5);
}

static double usurf0 = 0.0;   // check 19's surface cell speed / theory, no floor

// 19. Check 3 with a surface that can move: T linear along x in a closed
//     layer under gravity. Far from the ends the profile is check 3's, and
//     the surface tilts to drive the return flow: dh/dx = 3 tau / (2 rho g h).
template <class R>
void fs_thermocapillary(double floor)
{
  MeltFlow<R> m;
  const int nx = 96, nz = 24;
  const double H = 16.0, g = 1e-4, tau = -1e-5;
  m.fs = true; m.fs_tau = floor;
  m.resize(nx, 1, nz);
  m.p.periodic[1] = true;
  m.p.nu = 0.1;
  m.p.accel[2] = -g;
  m.p.marangoni = tau; m.p.Tref = 48.0; m.p.sigma = 2e-3;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) hT(i, 0, k) = R(i);
  Kokkos::deep_copy(m.T.v, hT);
  fs_shape(m, [&](int) { return 1.0 + H; });
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 1, nz};
  for (int s = 0; s < 120000; s++) { m.surface(lo, hi); m.step(); m.convert(); }
  const auto h = fs_height(m);
  // the tilt over the middle half, by least squares
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  int n = 0;
  for (int i = nx / 4; i < 3 * nx / 4; i++) { sx += i; sy += h[i]; sxx += (double) i * i; sxy += i * h[i]; n++; }
  const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  auto u = host(m.ux.v);
  const double hl = h[nx / 2] - 1.0, us = tau * hl / (4.0 * m.p.nu);
  double err = 0.0, flux = 0.0;
  for (int k = 1; k < nz; k++) {
    const double z = k - 0.5;
    if (z < hl - 2.0) err = std::max(err, std::fabs(u(nx / 2, 0, k) - tau * (3.0 * z * z - 2.0 * hl * z) / (4.0 * m.p.nu * hl)) / std::fabs(us));
    flux += u(nx / 2, 0, k) / (hl * std::fabs(us));
  }
  if (floor > 0.0) {
    printf("       19 with the interface floor tau = %g: tilt %+.1f%%, surface cell speed / theory %.2f (without: %.2f)\n", floor,
           100 * (slope / (3.0 * tau / (2.0 * g * hl)) - 1.0), u(nx / 2, 0, (int) h[nx / 2] - 1) / us, usurf0);
    return;
  }
  usurf0 = u(nx / 2, 0, (int) h[nx / 2] - 1) / us;
  check("19 fs thermocapillary: tilt / (3 tau / (2 g h)) - 1", slope / (3.0 * tau / (2.0 * g * hl)) - 1.0, 5e-2);
  check("19 fs thermocapillary: max |u - profile| / u_surf, 2 cells below the surface", err, 3e-2);
  check("19 fs thermocapillary: net flux / (h u_surf)", flux, 2e-2);
}

// 21. The pool of the SI driver check with a free surface (surface tension
//     1.5 N/m): it must stay finite and in range; the deflection is reported.
template <class R>
void fs_pool()
{
  MeltFlow<R> m;
  const int nx = 64, nz = 30, ns = 24;       // the surface starts at k = ns
  m.fs = true;
  m.resize(nx, nx, nz);
  m.ph.dx = 20e-6; m.ph.nu = 8.5e-7; m.ph.dsdT_rho = -2.6e-4 / 4100.0; m.ph.sig_rho = 1.5 / 4100.0;
  m.ph.beta_g = 1e-5 * 9.81; m.ph.Tref = 1928.0; m.ph.darcy = 1e4; m.ph.u_ref = 1.0;
  auto ht = Kokkos::create_mirror_view(m.type.v);
  auto hT = Kokkos::create_mirror_view(m.T.v);
  auto hf = Kokkos::create_mirror_view(m.fl.v);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < nx; j++)
      for (int k = 0; k < nz; k++) {
        const double x = i - 31.5, y = j - 31.5, d = std::max(0.0, ns - 0.5 - k);
        const double e = (x * x + y * y) / (20.0 * 20.0) + d * d / (12.0 * 12.0);
        ht(i, j, k) = k >= ns ? MeltFlow<R>::GAS : e < 1.0 ? MeltFlow<R>::FLUID : MeltFlow<R>::WALL;
        hT(i, j, k) = R(1928.0 + 1200.0 * std::exp(-(x * x + y * y) / (12.0 * 12.0) - d * d / (8.0 * 8.0)));
        // nothing freezes here: liquid that spills over the solid is held as if it had (a liquid fraction of 1e-3)
        hf(i, j, k) = R(std::min(1.0, std::max(k >= ns ? 1e-3 : 0.05, 4.0 * (1.0 - e))));
      }
  Kokkos::deep_copy(m.type.v, ht); Kokkos::deep_copy(m.T.v, hT); Kokkos::deep_copy(m.fl.v, hf);
  m.fs_repair();
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, nx, nz};
  const double v0 = m.total_mass();
  double lat_max = 0.0, u_late = 0.0, lost = 0.0;
  int bad = 0, nmax = 0;
  // the lost mass kept on the device, read once per run() (as the conduction source does)
  m.post = [&]() { m.convert(lo, hi, typename MeltFlow<R>::W3(), typename MeltFlow<R>::W3(), true); };
  for (int s = 0; s < 4000 && !bad; s++) {
    const int n = m.run(5e-6, lo, hi);
    { double g, l; m.tallies(g, l); lost += l; }
    if (n < 0 || !std::isfinite(m.umax)) { bad = 1; break; }
    nmax = std::max(nmax, n);
    if (s >= 200) lat_max = std::max(lat_max, m.umax * m.dt / m.ph.dx);
    if (s >= 3000) u_late = std::max(u_late, m.umax);
  }
  auto e = host(m.eps.v); auto t = host(m.type.v);
  double up = 0.0, down = 0.0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < nx; j++) {
      if (ht(i, j, ns - 1) != MeltFlow<R>::FLUID) continue;
      double h = 0.0;
      for (int k = 0; k < nz; k++) h += t(i, j, k) == MeltFlow<R>::WALL ? 1.0 : e(i, j, k);
      up = std::max(up, h - ns); down = std::max(down, ns - h);
    }
  // the liquid's volume (the mass is not kept when the step changes: the
  // pressure's part of the density is rescaled)
  double vol = 0.0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < nx; j++)
      for (int k = 0; k < nz; k++) vol += t(i, j, k) == MeltFlow<R>::WALL ? 0.0 : e(i, j, k);
  check("21 fs pool (SI driver, 20 ms): stays finite, steps <= nmax", bad, 0.0);
  check("21 fs pool: max lattice speed after start-up", bad ? NAN : lat_max, 0.25);
  check("21 fs pool: max speed in the last 5 ms (m/s), >= 0.5", bad ? NAN : u_late, 0.5, false);
  check("21 fs pool: max speed in the last 5 ms (m/s), <= 10", bad ? NAN : u_late, 10.0);
  check("21 fs pool: |volume change| / volume, lost droplets included", bad ? NAN : (vol + lost) / v0 - 1.0, 2e-3);
  check("21 fs pool: lost droplets / volume", bad ? NAN : lost / v0, 2e-3);
  printf("       fs pool: surface up %.2f, down %.2f cells; %d flow steps per 5 us at most, dt %.3g s, lost %.3g\n",
         up, down, nmax, m.dt, lost);
}

// 20. Mass fed onto a pool at rest under a Gaussian footprint (powder):
//     the pool gains what was fed, and the carried field its share.
template <class R>
void fs_inject()
{
  MeltFlow<R> m;
  const int nx = 32, ny = 32, nz = 20;
  m.fs = true;
  m.resize(nx, ny, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.accel[2] = -1e-4;
  m.p.nu = 0.05;
  m.p.sigma = 0.01;
  fs_shape(m, [&](int) { return 9.5; });
  Kokkos::View<R**> dm("dm", nx, ny);
  auto hd = Kokkos::create_mirror_view(dm);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      hd(i, j) = R(1e-3 * std::exp(-((i - 15.5) * (i - 15.5) + (j - 15.5) * (j - 15.5)) / 36.0));
  Kokkos::deep_copy(dm, hd);
  Kokkos::View<R***> q("q", nx, ny, nz), tmp("tmp", nx, ny, nz), qfv("qf", nx, ny, nz);
  Kokkos::deep_copy(q, R(1));
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, ny, nz};
  auto total = [&]() {
    auto h = host(q); auto r = host(m.rho.v); auto ms = host(m.cmass.v); auto t = host(m.type.v);
    double sum = 0.0;
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
          sum += t(i, j, k) == MeltFlow<R>::FLUID ? (1.0 + r(i, j, k)) * h(i, j, k)
               : t(i, j, k) == MeltFlow<R>::INTERFACE ? (double) ms(i, j, k) * h(i, j, k) : 0.0;
    return sum;
  };
  const double m0 = m.total_mass(), s0 = total(), qf = 3.0;
  Kokkos::deep_copy(qfv, R(qf));
  double fed = 0.0, lost = 0.0;
  for (int s = 0; s < 6000; s++) {
    m.surface(lo, hi); m.step();
    melt_advect(m, q, tmp, lo, hi);
    // the sums kept on the device (as the conduction source runs it)
    using W3 = typename MeltFlow<R>::W3;
    if (s < 3000) m.inject(lo, hi, dm, R(1), q, qfv, W3(), W3(), true);
    m.convert(lo, hi, q, W3(), true);
    if (s % 100 == 99) { double g, l; m.tallies(g, l); fed += g; lost += l; }
  }
  { double g, l; m.tallies(g, l); fed += g; lost += l; }
  const auto h = fs_height(m);
  double hlo = 1e300, hhi = -1e300;
  for (int i = 0; i < nx; i++) { hlo = std::min(hlo, h[i]); hhi = std::max(hhi, h[i]); }
  check("20 injection: (mass gained - fed) / fed", (m.total_mass() + lost - m0) / fed - 1.0, pick<R>(1e-10, 1e-3));
  check("20 injection: (sum of mass x q gained) / (fed x q fed) - 1", (total() - s0) / (fed * qf) - 1.0, 1e-3);
  check("20   (the pool rose: cells)", 0.5 * (hlo + hhi) - 9.5, 0.1, false);
  check("20   (and levelled: height max - min, cells)", hhi - hlo, 0.05);
}

// 22. The surface force on a steep surface (no flow: surface() alone). A
//     wave of slope up to 2 with sigma linear along x: the force put on the
//     cells, summed over a wavelength, against the integral of
//     sigma_x (1, h_x) / sqrt(1 + h_x^2), and the largest acceleration of
//     any one cell against sigma_x (a cell holding little must not be thrown).
template <class R>
void fs_force()
{
  MeltFlow<R> m;
  const int nx = 64, nz = 48;
  const double H = 24.0, A = 8.0, kw = 2.0 * M_PI / nx, tau = 1e-5;
  m.fs = true;
  m.resize(nx, 2, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.marangoni = tau; m.p.sigma = 0.01; m.p.Tref = 0.0;
  auto hT = Kokkos::create_mirror_view(m.T.v);
  // T periodic with a uniform gradient over the middle half (sigma_x = tau there)
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < nz; k++) hT(i, j, k) = R(i < nx / 2 ? i : nx - i);
  Kokkos::deep_copy(m.T.v, hT);
  fs_shape(m, [&](int i) { return 1.0 + H + A * std::cos(kw * (i + 0.5)); });
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 2, nz};
  m.surface(lo, hi);
  auto t = host(m.type.v); auto e = host(m.eps.v);
  double fx = 0, fz = 0, ex = 0, ez = 0, amax = 0;
  auto hs = host(m.hfs.v);
  for (int i = 4; i < nx / 2 - 4; i++) {
    // what the step applies: each liquid cell's acceleration, by its fill
    for (int k = 0; k < nz; k++) {
      if (!MeltFlow<R>::liq(t(i, 0, k))) continue;
      const double w = t(i, 0, k) == MeltFlow<R>::INTERFACE ? std::min(1.0, std::max(0.0, (double) e(i, 0, k))) : 1.0;
      const double cx = hs(i, 0, k, 1), cz = hs(i, 0, k, 3);
      fx += w * cx; fz += w * cz;
      amax = std::max(amax, std::hypot(cx, cz));
    }
    const double hx = -A * kw * std::sin(kw * (i + 0.5)), g = std::sqrt(1.0 + hx * hx);
    ex += tau / g; ez += tau * hx / g;
  }
  check("22 steep surface: force along x, applied / exact - 1", fx / ex - 1.0, 5e-2);
  check("22 steep surface: force along z, (applied - exact) / |exact x|", (fz - ez) / ex, 5e-2);
  check("22 steep surface: largest cell acceleration / sigma_x", amax / tau, 1.5);
}

/* 23-25: curvature from local height functions.
   Shapes are set from their exact fills (the volume of each cell inside,
   by quadrature fine enough that its error is far below the curvature's),
   all cells FLUID, INTERFACE or GAS by fill, and WALL where wall(i, j, k). */
template <class R, class F, class W>
void fs_fill(MeltFlow<R> &m, const F fill, const W wall)
{
  auto ht = Kokkos::create_mirror_view(m.type.v);
  auto hm = Kokkos::create_mirror_view(m.cmass.v);
  auto he = Kokkos::create_mirror_view(m.eps.v);
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++) {
        double f = wall(i, j, k) ? 0.0 : fill(i, j, k);
        if (f < 1e-12) f = 0.0;
        if (f > 1.0 - 1e-12) f = 1.0;
        ht(i, j, k) = wall(i, j, k) ? MeltFlow<R>::WALL : f >= 1.0 ? MeltFlow<R>::FLUID
                    : f > 0.0 ? MeltFlow<R>::INTERFACE : MeltFlow<R>::GAS;
        hm(i, j, k) = he(i, j, k) = R(f);
      }
  Kokkos::deep_copy(m.type.v, ht); Kokkos::deep_copy(m.cmass.v, hm); Kokkos::deep_copy(m.eps.v, he);
  m.fs_repair();
}

// The volume of the cell [i, i+1) x [j, j+1) x [k, k+1) inside the sphere
// (c, r) (chords along z, midpoint rule over n x n in x, y), or inside the
// cylinder along y (c0, c2, r) for x >= xw.
double sphere_fill(int i, int j, int k, const double c[3], double r, int n = 256)
{
  const double d = std::sqrt(std::pow(i + 0.5 - c[0], 2) + std::pow(j + 0.5 - c[1], 2) + std::pow(k + 0.5 - c[2], 2));
  if (d > r + 0.9) return 0.0;
  if (d < r - 0.9) return 1.0;
  double s = 0.0;
  for (int a = 0; a < n; a++)
    for (int b = 0; b < n; b++) {
      const double x = i + (a + 0.5) / n - c[0], y = j + (b + 0.5) / n - c[1], q = r * r - x * x - y * y;
      if (q <= 0.0) continue;
      const double h = std::sqrt(q);
      s += std::max(0.0, std::min(k + 1.0, c[2] + h) - std::max((double) k, c[2] - h));
    }
  return s / (n * n);
}
double cylinder_fill(int i, int k, double c0, double c2, double r, double xw = -1e30, double rz = 0.0, int n = 4096)
{
  if (rz <= 0.0) rz = r;     // an ellipse of semi-axes r along x, rz along z
  double s = 0.0;
  for (int a = 0; a < n; a++) {
    const double x = i + (a + 0.5) / n;
    if (x < xw) continue;
    const double q = r * r - (x - c0) * (x - c0);
    if (q <= 0.0) continue;
    const double h = rz / r * std::sqrt(q);
    s += std::max(0.0, std::min(k + 1.0, c2 + h) - std::max((double) k, c2 - h));
  }
  return s / n;
}

// kappa per INTERFACE cell from surface() (-drg / (3 sigma), no gas
// pressure) against kexact: the rms, median and largest relative error,
// and how many cells each method took
template <class R>
void fs_kappa_err(MeltFlow<R> &m, double kexact, double &rms, double &med, double &mx, int cnt[6])
{
  const int lo[3] = {0, 0, 0}, hi[3] = {m.nx, m.ny, m.nz};
  m.surface(lo, hi);
  auto hs = host(m.hfs.v); auto hr = host(m.hfr.v); auto t = host(m.type.v); auto he = host(m.eps.v);
  double s2 = 0.0;
  int n = 0;
  std::vector<double> all;
  mx = 0.0;
  for (int c = 0; c < 6; c++) cnt[c] = 0;
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++) {
        if (t(i, j, k) != MeltFlow<R>::INTERFACE) continue;
        cnt[hr(i, j, k) >> 3]++;
        const double e = (-hs(i, j, k, 0) / (3.0 * m.p.sigma)) / kexact - 1.0;
        s2 += e * e; mx = std::max(mx, std::fabs(e)); n++;
        all.push_back(std::fabs(e));
      }
  rms = std::sqrt(s2 / std::max(n, 1));
  std::sort(all.begin(), all.end());
  med = all.empty() ? NAN : all[all.size() / 2];
}

// the mean rho - 1 of the FLUID cells within dist(i, j, k) < rin
template <class R, class D>
double fs_inner_rho(const MeltFlow<R> &m, const D dist, double rin)
{
  auto r = host(m.rho.v); auto t = host(m.type.v);
  double s = 0.0;
  int n = 0;
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++)
        if (t(i, j, k) == MeltFlow<R>::FLUID && dist(i, j, k) < rin) { s += r(i, j, k); n++; }
  return n ? s / n : NAN;
}

template <class R>
double fs_umax(const MeltFlow<R> &m)
{
  auto u = host(m.ux.v); auto v = host(m.uy.v); auto w = host(m.uz.v); auto t = host(m.type.v);
  double mx = 0.0;
  for (int i = 0; i < m.nx; i++)
    for (int j = 0; j < m.ny; j++)
      for (int k = 0; k < m.nz; k++)
        if (MeltFlow<R>::liq(t(i, j, k)))
          mx = std::max(mx, std::sqrt((double) u(i, j, k) * u(i, j, k) + (double) v(i, j, k) * v(i, j, k) + (double) w(i, j, k) * w(i, j, k)));
  return mx;
}

// 23. A sphere of radius r, periodic box, no gravity. Static: kappa = -2 / r
//     at every interface cell. At rest after the flow runs: the Laplace
//     pressure 2 sigma / r inside (rho - 1 = 6 sigma / r) and the spurious
//     currents, as a capillary number u nu / sigma.
template <class R>
void fs_sphere(double r, int nstep, double &rms, double &med, double &mx, int cnt[6], double &dp, double &ca)
{
  MeltFlow<R> m;
  const int n = 2 * (int) std::ceil(r) + 8;
  const double c[3] = {0.5 * n + 0.1, 0.5 * n - 0.2, 0.5 * n + 0.3};   // off the grid's symmetry
  m.fs = true; m.fs_tau = 0.0;
  m.resize(n, n, n);
  m.p.periodic[0] = m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.nu = 0.1;
  m.p.sigma = 0.002;
  fs_fill(m, [&](int i, int j, int k) { return sphere_fill(i, j, k, c, r); }, [](int, int, int) { return false; });
  fs_kappa_err(m, -2.0 / r, rms, med, mx, cnt);
  dp = ca = NAN;
  if (nstep <= 0) return;
  const int lo[3] = {0, 0, 0}, hi[3] = {n, n, n};
  for (int s = 0; s < nstep; s++) { m.surface(lo, hi); m.step(); m.convert(); }
  auto dist = [&](int i, int j, int k) { return std::sqrt(std::pow(i + 0.5 - c[0], 2) + std::pow(j + 0.5 - c[1], 2) + std::pow(k + 0.5 - c[2], 2)); };
  dp = fs_inner_rho(m, dist, r - 3.0) / (6.0 * m.p.sigma / r) - 1.0;
  ca = fs_umax(m) * m.p.nu / m.p.sigma;
}

// 24, 25. A horizontal cylinder along y (periodic in y), so the surface's
//     normal turns through every direction in x-z, under faces included:
//     kappa = -1 / r, and at rest the Laplace pressure sigma / r. wall: a
//     half-cylinder on a vertical wall (WALL at i = 0, the liquid at
//     x >= 1): an overhang that meets the wall at 90 degrees, the column
//     model's mirror at a wall, an exact equilibrium. Also its lowest point
//     after the run (it must not run down the wall).
template <class R>
void fs_cyl(bool wall, double r, int nstep, double &rms, double &med, double &mx, int cnt[6], double &dp, double &ca, double &sag)
{
  MeltFlow<R> m;
  const int nx = wall ? (int) std::ceil(r) + 6 : 2 * (int) std::ceil(r) + 8, nz = 2 * (int) std::ceil(r) + 8;
  const double c0 = wall ? 1.0 : 0.5 * nx + 0.2, c2 = 0.5 * nz + 0.3;
  m.fs = true; m.fs_tau = 0.0;
  m.resize(nx, 2, nz);
  m.p.periodic[1] = m.p.periodic[2] = true;
  m.p.periodic[0] = !wall;
  m.p.nu = 0.1;
  m.p.sigma = 0.002;
  fs_fill(m, [&](int i, int, int k) { return cylinder_fill(i, k, c0, c2, r, wall ? 1.0 : -1e30); },
          [&](int i, int, int) { return wall && i == 0; });
  fs_kappa_err(m, -1.0 / r, rms, med, mx, cnt);
  dp = ca = sag = NAN;
  if (nstep <= 0) return;
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 2, nz};
  for (int s = 0; s < nstep; s++) { m.surface(lo, hi); m.step(); m.convert(); }
  auto dist = [&](int i, int, int k) { return std::sqrt(std::pow(i + 0.5 - c0, 2) + std::pow(k + 0.5 - c2, 2)); };
  dp = fs_inner_rho(m, dist, r - 3.0) / (3.0 * m.p.sigma / r) - 1.0;
  ca = fs_umax(m) * m.p.nu / m.p.sigma;
  // the liquid's lowest point (cells) beside the wall, against c2 - r
  auto e = host(m.eps.v); auto t = host(m.type.v);
  double zlo = 1e30;
  for (int k = 0; k < nz; k++)
    for (int i = 1; i < nx; i++)
      if (MeltFlow<R>::liq(t(i, 0, k))) { zlo = std::min(zlo, k + 1.0 - std::min(1.0, std::max(0.0, (double) e(i, 0, k)))); break; }
  sag = (c2 - r) - zlo;
}

// 26. A horizontal elliptic cylinder of liquid (semi-axes a along x, b
//     along z; along y, periodic), floating, no gravity: it must round into
//     the cylinder of its area, radius r = sqrt(a b) > a, its sides growing
//     out over gas and holding the
//     Laplace pressure sigma / r. Returns the middle row's half-width / r
//     - 1 and that pressure's error. (Not stuck to a wall: on a vertical
//     wall face the contact angle depends on the frame: docs/
//     FREE_SURFACE_LBM.md, 3.) The column model's supported-conversion
//     rule (a9ac6675, removed) held the sides back: half-width -14%, and
//     the liquid lost its pressure.
template <class R>
void fs_relax(double &reach, double &dp)
{
  MeltFlow<R> m;
  const int nx = 24, nz = 32;
  const double a = 4.0, b = 10.0, c0 = 12.2, c2 = 16.3;
  m.fs = true; m.fs_tau = 0.0;
  m.resize(nx, 2, nz);
  m.p.periodic[0] = m.p.periodic[1] = true;
  m.p.nu = 0.1;
  m.p.sigma = 0.002;
  fs_fill(m, [&](int i, int, int k) { return cylinder_fill(i, k, c0, c2, a, -1e30, b); }, [](int, int, int) { return false; });
  const int lo[3] = {0, 0, 0}, hi[3] = {nx, 2, nz};
  for (int s = 0; s < 12000; s++) { m.surface(lo, hi); m.step(); m.convert(); }
  const double r = std::sqrt(a * b);
  // the liquid's centre (fill-weighted) and the width of its row there
  auto e = host(m.eps.v); auto t = host(m.type.v);
  double s = 0.0, sx = 0.0, sz = 0.0;
  for (int k = 0; k < nz; k++)
    for (int i = 0; i < nx; i++) {
      const double f = MeltFlow<R>::liq(t(i, 0, k)) ? std::min(1.0, std::max(0.0, (double) e(i, 0, k))) : 0.0;
      s += f; sx += f * (i + 0.5); sz += f * (k + 0.5);
    }
  const int kc = (int) (sz / s);
  double row = 0.0;
  for (int i = 0; i < nx; i++) row += MeltFlow<R>::liq(t(i, 0, kc)) ? std::min(1.0, std::max(0.0, (double) e(i, 0, kc))) : 0.0;
  reach = 0.5 * row / r - 1.0;
  auto dist = [&](int i, int, int k) { return std::sqrt(std::pow(i + 0.5 - sx / s, 2) + std::pow(k + 0.5 - sz / s, 2)); };
  dp = fs_inner_rho(m, dist, r - 3.0) / (3.0 * m.p.sigma / r) - 1.0;
}

template <class R>
void fs_hf_shapes()
{
  int cnt[6];
  double rms, med, mx, dp, ca, sag, e[4], em[4], cas[4];
  const double rs[4] = {4.0, 8.0, 16.0, 32.0};
  for (int s = 0; s < 4; s++) {
    fs_sphere<R>(rs[s], s == 1 || s == 2 ? 6000 : 0, rms, med, mx, cnt, dp, ca);
    e[s] = rms; em[s] = med; cas[s] = ca;
    printf("       23 sphere r = %g: kappa error rms %.2e, median %.2e, max %.2e; cells by heights (dominant frame, other), fit, mean, none: %d %d %d %d %d\n",
           rs[s], rms, med, mx, cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
    if (s == 1 || s == 2) {
      char nm[96];
      snprintf(nm, sizeof(nm), "23 sphere r = %g at rest: rho inside / (6 sigma / r) - 1", rs[s]);
      check(nm, dp, 3e-2);
      printf("       23 sphere r = %g at rest: spurious currents, u nu / sigma = %.2e\n", rs[s], ca);
    }
  }
  // the spurious currents have no analytical value but 0: they must fall
  // with the resolution (from conversions and the curvature's error)
  check("23 sphere r = 16 at rest: spurious currents, u nu / sigma", cas[2], 3e-3);
  check("23 sphere: spurious currents r = 8 / r = 16", cas[1] / cas[2], 4.0, false);
  check("23 sphere r = 8: kappa rms error / (2 / r)", e[1], 1e-2);
  check("23 sphere r = 16: kappa rms error / (2 / r)", e[2], 5e-3);
  check("23 sphere r = 32: kappa rms error / (2 / r)", e[3], 2e-3);
  // second order: the median cell from 8 cells on; the rms (cells with
  // slopes near 1 in both directions converge late) from 16 on
  if (sizeof(R) == 8) {
    check("23 sphere: median error order, r = 8 -> 16", std::log2(em[1] / em[2]), 1.8, false);
    check("23 sphere: median error order, r = 16 -> 32", std::log2(em[2] / em[3]), 1.8, false);
    check("23 sphere: rms error order, r = 16 -> 32", std::log2(e[2] / e[3]), 1.8, false);
  }
  for (int s = 1; s < 4; s++) {
    fs_cyl<R>(false, rs[s], s == 1 ? 6000 : 0, rms, med, mx, cnt, dp, ca, sag);
    e[s] = rms;
    printf("       24 cylinder r = %g: kappa error rms %.2e, median %.2e, max %.2e; methods %d %d %d %d %d\n",
           rs[s], rms, med, mx, cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
    if (s == 1) {
      check("24 cylinder r = 8 at rest: rho inside / (3 sigma / r) - 1", dp, 3e-2);
      check("24 cylinder r = 8 at rest: spurious currents, u nu / sigma", ca, 1e-2);
    }
  }
  check("24 cylinder r = 8: kappa rms error / (1 / r)", e[1], 1e-2);
  if (sizeof(R) == 8) {
    check("24 cylinder: rms error order, r = 8 -> 16", std::log2(e[1] / e[2]), 1.8, false);
    check("24 cylinder: rms error order, r = 16 -> 32", std::log2(e[2] / e[3]), 1.8, false);
  }
  fs_cyl<R>(true, 8.0, 6000, rms, med, mx, cnt, dp, ca, sag);
  printf("       25 half-cylinder on a wall r = 8: kappa error rms %.2e, max %.2e; methods %d %d %d %d %d\n", rms, mx, cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
  check("25 bulge on a wall r = 8: kappa rms error / (1 / r)", rms, 1e-2);
  check("25 bulge on a wall at rest: rho inside / (3 sigma / r) - 1", dp, 3e-2);
  check("25 bulge on a wall at rest: spurious currents, u nu / sigma", ca, 1e-2);
  check("25 bulge on a wall: its lowest point moved (cells)", sag, 0.5);
  double reach;
  fs_relax<R>(reach, dp);
  check("26 elliptic cylinder rounds: middle row's half-width / r - 1", reach, 5e-2);
  check("26 elliptic cylinder rounds: rho inside / (sigma / r) - 1", dp, 3e-2);
}

template <class R>
void fs_all()
{
  fs_rest<R>();
  fs_dam<R>();
  fs_wave<R>(0.0);
  fs_load<R>();
  fs_thermocapillary<R>(0.0);
  fs_inject<R>();
  fs_force<R>();
  fs_pool<R>();
  fs_hf_shapes<R>();
}

template <class R>
void suite(const char *name)
{
  printf("%s precision\n", name);
  poiseuille<R>();
  double r32, r64;
  const double e32 = taylor_green<R>(32, r32), e64 = taylor_green<R>(64, r64);
  check("2 Taylor-Green N=64: decay rate / (2 nu k^2) - 1", r64, 1e-2);
  check("2 Taylor-Green N=64: L2 error / amplitude", e64, 1e-2);
  if (sizeof(R) == 8) {
    check("2 Taylor-Green: error(32) / error(64) >= 3.5", e32 / e64, 3.5, false);
    check("2 Taylor-Green: error(32) / error(64) <= 4.5", e32 / e64, 4.5);
  }
  double i12, i24, t12, t24, f12, f24;
  thermocapillary<R>(12, i12, t12, f12);
  thermocapillary<R>(24, i24, t24, f24);
  check("3 thermocapillary h=24: max |u - profile| / u_surf below the top cell", i24, 5e-3);
  check("3 thermocapillary h=24: top cell lag / (u_surf / 2h), <= 1", t24 * 2 * 24, 1.0);
  check("3 thermocapillary h=24: net flux / (h u_surf)", f24, pick<R>(1e-6, 5e-3));
  if (sizeof(R) == 8)
    check("3 thermocapillary: interior error(12) / error(24) >= 3", i12 / i24, 3.0, false);
  buoyant_slot<R>();
  hydrostatic<R>();
  darcy<R>();
  block<R>();
  pool_stability<R>();
  props_same<R>();
  const double u1 = pool_si<R>(1.0), u2 = pool_si<R>(0.25);
  check("pool: speed from u_ref 0.25 / from u_ref 1 - 1", u2 / u1 - 1.0, 0.15);
  advect_suite<R>();
  fs_all<R>();
}

}   // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    printf("melt_flow_check on %s\n", Kokkos::DefaultExecutionSpace::name());
    if (argc > 1 && std::string(argv[1]) == "hf") {     // the height-function shapes alone
      printf("double precision\n"); fs_hf_shapes<double>();
      printf("single precision\n"); fs_hf_shapes<float>();
    } else if (argc > 1 && std::string(argv[1]) == "fs") {     // the free-surface checks alone
      printf("double precision\n"); fs_all<double>();
      printf("single precision\n"); fs_all<float>();
    } else {
      suite<double>("double");
      suite<float>("single");
    }
  }
  Kokkos::finalize();
  printf(nfail ? "melt_flow_check: %d check(s) FAILED\n" : "melt_flow_check ok\n", nfail);
  return nfail ? 1 : 0;
}
