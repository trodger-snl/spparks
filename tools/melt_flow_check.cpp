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
  auto ht = Kokkos::create_mirror_view(m.type);
  ht(0, 0, 0) = ht(0, 0, nz - 1) = MeltFlow<R>::WALL;
  Kokkos::deep_copy(m.type, ht);
  for (int s = 0; s < 60000; s++) m.step();
  auto u = host(m.ux);
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
  auto u = host(m.ux); auto v = host(m.uy);
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
  auto hT = Kokkos::create_mirror_view(m.T);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) hT(i, 0, k) = R(i);
  Kokkos::deep_copy(m.T, hT);
  for (int s = 0; s < 140 * nz * nz; s++) m.step();
  auto u = host(m.ux);
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
  auto hT = Kokkos::create_mirror_view(m.T);
  for (int i = 0; i < nx; i++) hT(i, 0, 0) = R((i + 0.5) / nx - 0.5);
  Kokkos::deep_copy(m.T, hT);
  for (int s = 0; s < 40000; s++) m.step();
  auto w = host(m.uz);
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
  auto hT = Kokkos::create_mirror_view(m.T);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++) hT(i, j, k) = R(k);
  Kokkos::deep_copy(m.T, hT);
  // the force starts abruptly: the checkerboard it leaves must cancel in u
  for (int s = 0; s < 30001; s++) m.step();
  auto u = host(m.ux); auto v = host(m.uy); auto w = host(m.uz);
  double umax = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        umax = std::max({umax, std::fabs((double) u(i, j, k)), std::fabs((double) v(i, j, k)),
                         std::fabs((double) w(i, j, k))});
  check("5 hydrostatic: max |u| (lattice)", umax, pick<R>(1e-8, 1e-5));
  auto pw = host(m.pz);
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
  Kokkos::deep_copy(m.fl, R(fl));
  for (int s = 0; s < 5000; s++) m.step();
  auto u = host(m.ux);
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
  auto ht = Kokkos::create_mirror_view(m.type);
  int nfluid = 0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        const bool wall = k == 0 || k == nz - 1 || (i >= 12 && i < 20 && k <= 8);
        ht(i, j, k) = wall ? MeltFlow<R>::WALL : MeltFlow<R>::FLUID;
        nfluid += !wall;
      }
  Kokkos::deep_copy(m.type, ht);
  for (int s = 0; s < 5000; s++) m.step();
  auto u = host(m.ux);
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
  auto hT = Kokkos::create_mirror_view(m.T);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) hT(i, 0, k) = R(20.0 * i);
  Kokkos::deep_copy(m.T, hT);
  for (int s = 0; s < 100000; s++) m.step();
  auto u = host(m.ux); auto w = host(m.uz);
  double umax = 0.0;
  bool finite = true;
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++) {
      finite = finite && std::isfinite(u(i, 0, k)) && std::isfinite(w(i, 0, k));
      umax = std::max(umax, std::hypot((double) u(i, 0, k), (double) w(i, 0, k)));
    }
  check("pool conditions, 1e5 steps: max |u| (lattice; 0.1 = 1 m/s)", finite ? umax : NAN, 0.3);
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
