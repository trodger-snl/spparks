// Checks src/crystalline_orientations/crystal_symmetries.h: each rotation table
// is a group (closed under products, up to sign), its largest disorientation
// (sampled) sits just under CRYSTAL::max_disorientation and the exact value,
// and the growth families / reach come out as expected.
// build: g++ -O2 -I ../../src -o check_symmetries check_symmetries.cpp
// run:   ./check_symmetries   (exit 0 = pass)
#include "crystalline_orientations/crystal_symmetries.h"
#include "crystalline_orientations/disorientation.h"
#include <cstdio>
#include <random>
using namespace SPPARKS_NS;

static void mul(const double *a, const double *b, double *c) {
  c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  c[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  c[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
  c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

int main() {
  const int nops[] = {24, 12, 6, 8, 4};
  const double exact[] = {62.80, 93.84, 104.48, 98.42, 120.0};
  int bad = 0;
  std::mt19937 gen(1);
  std::normal_distribution<double> nd;
  for (int c = 0; c < CRYSTAL::nnames; c++) {
    const char *name = CRYSTAL::names[c];
    vector<double> s = CRYSTAL::get_symmetries(name);
    int n = (int) s.size() / 4, closed = 1;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        double p[4], best = 0;
        mul(&s[4*i], &s[4*j], p);
        for (int k = 0; k < n; k++) {
          double d = 0;
          for (int m = 0; m < 4; m++) d += p[m] * s[4*k+m];
          best = std::fmax(best, std::fabs(d));
        }
        if (best < 1 - 1e-12) closed = 0;
      }
    double mx = 0, one[4] = {1, 0, 0, 0};
    for (int t = 0; t < 400000; t++) {
      double q[4], qn = 0;
      for (double &x : q) { x = nd(gen); qn += x*x; }
      for (double &x : q) x /= std::sqrt(qn);
      mx = std::fmax(mx, disorientation::compute_disorientation(
                             s, vector<double>(q, q+4), vector<double>(one, one+4)));
    }
    const double cap = CRYSTAL::max_disorientation(name);
    int ok = n == nops[c] && closed && mx > exact[c] - 1.5 && mx < exact[c] + 0.01 &&
             cap <= exact[c] && cap > exact[c] - 0.15;
    bad += !ok;
    printf("%-12s ops %2d closed %d sampled max %.2f (exact %.2f, cap %.1f) %s\n", name, n,
           closed, mx, exact[c], cap, ok ? "ok" : "FAIL");
  }
  // growth families: cubic [001] must be exactly the unit axes (bit-identity
  // with the old <100> code), with reach acos(1/sqrt 3)
  const double z[3] = {0, 0, 1}, d111[3] = {1, 1, 1}, d110[3] = {1, 1, 0};
  vector<double> cub = CRYSTAL::get_symmetries("cubic");
  vector<double> f = CRYSTAL::growth_family(cub, z);
  int exact_axes = f.size() == 9;
  for (std::size_t k = 0; k < f.size(); k++) exact_axes &= f[k] == 0.0 || std::fabs(f[k]) == 1.0;
  struct { const char *crystal; const double *g; int ndir; double reach_deg; } cases[] = {
    {"cubic", z, 3, 54.7356}, {"cubic", d111, 4, 54.7356}, {"cubic", d110, 6, 45.0},
    {"hcp", z, 1, 90.0}, {"tetragonal", z, 1, 90.0}, {"orthorhombic", z, 1, 90.0}};
  bad += !exact_axes;
  printf("cubic [001] family exact unit axes: %s\n", exact_axes ? "ok" : "FAIL");
  for (auto &t : cases) {
    vector<double> d = CRYSTAL::growth_family(CRYSTAL::get_symmetries(t.crystal), t.g);
    double r = CRYSTAL::family_reach(d) * 180.0 / M_PI;
    int ok = (int) d.size() / 3 == t.ndir && std::fabs(r - t.reach_deg) < 1e-3;
    bad += !ok;
    printf("%-12s [%g %g %g]: %d axes, reach %.4f deg %s\n", t.crystal, t.g[0], t.g[1], t.g[2],
           (int) d.size() / 3, r, ok ? "ok" : "FAIL");
  }
  return bad != 0;
}
