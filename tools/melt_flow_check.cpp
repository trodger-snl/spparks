/* Self-check of src/KOKKOS/melt_flow_kokkos.h against analytical flows.
   Build and run (any Kokkos build dir, e.g. build-verify):

     cmake --build build-verify --target melt_flow_check
     build-verify/melt_flow_check

   Runs every problem in double and in single precision, prints each
   measured error beside its tolerance and exits 1 if any check fails.
   Lattice units (cell 1, step 1, density 1). */

#include "melt_flow_kokkos.h"

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
  const double u1 = pool_si<R>(1.0), u2 = pool_si<R>(0.25);
  check("pool: speed from u_ref 0.25 / from u_ref 1 - 1", u2 / u1 - 1.0, 0.15);
  advect_suite<R>();
}

}   // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    printf("melt_flow_check on %s\n", Kokkos::DefaultExecutionSpace::name());
    suite<double>("double");
    suite<float>("single");
  }
  Kokkos::finalize();
  printf(nfail ? "melt_flow_check: %d check(s) FAILED\n" : "melt_flow_check ok\n", nfail);
  return nfail ? 1 : 0;
}
