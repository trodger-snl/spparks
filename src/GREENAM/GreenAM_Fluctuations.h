/* ----------------------------------------------------------------------
   GreenAM_Fluctuations.h

   Lightweight emission-time-keyed fluctuation table consumed by the
   GaussGreenEllips integrand. The temperature source builds the table
   once at scan start (with samples drawn from a PSD generator), and
   the integrator queries it at every emission-time `s` evaluation
   inside operator()(s).

   This is a SPPARKS extension to the upstream sierra::greenam library
   and lives in the vendored GREENAM/ tree alongside the original
   headers.
------------------------------------------------------------------------- */

#ifndef GREENAM_FLUCTUATIONS_H
#define GREENAM_FLUCTUATIONS_H

#include <cstddef>
#include <vector>

namespace sierra {
namespace greenam {

template <typename T>
struct FluctuationTable
{
  // Sorted, monotonically increasing emission-time samples (seconds).
  std::vector<T> t_grid;
  // Per-sample multiplicative factors applied inside the integrand.
  // factor_W scales sx AND sy together (lateral pool scaling).
  // factor_D scales sz                  (depth scaling).
  // factor_P scales the absorbed power  (laser power scaling).
  std::vector<T> factor_W;
  std::vector<T> factor_D;
  std::vector<T> factor_P;

  bool   empty() const { return t_grid.empty(); }
  size_t size()  const { return t_grid.size(); }

  void clear()
  {
    t_grid.clear();
    factor_W.clear();
    factor_D.clear();
    factor_P.clear();
  }

  // Linear interpolation at emission time s, clamped at the endpoints.
  // The integrator hits this at every Gauss-Legendre node, so it must
  // be cheap: one binary search + linear interpolation in each channel.
  inline void get_at(T s, T &fW, T &fD, T &fP) const
  {
    if (t_grid.empty()) {
      fW = T(1); fD = T(1); fP = T(1);
      return;
    }
    if (s <= t_grid.front()) {
      fW = factor_W.front();
      fD = factor_D.front();
      fP = factor_P.front();
      return;
    }
    if (s >= t_grid.back()) {
      fW = factor_W.back();
      fD = factor_D.back();
      fP = factor_P.back();
      return;
    }
    // Binary search for the bracketing pair (lo, lo+1).
    size_t lo = 0;
    size_t hi = t_grid.size() - 1;
    while (lo + 1 < hi) {
      const size_t mid = (lo + hi) / 2;
      if (t_grid[mid] <= s) lo = mid;
      else                  hi = mid;
    }
    const T t0 = t_grid[lo];
    const T t1 = t_grid[lo + 1];
    const T denom = t1 - t0;
    const T frac  = (denom > T(0)) ? (s - t0) / denom : T(0);
    const T one_minus = T(1) - frac;
    fW = one_minus * factor_W[lo] + frac * factor_W[lo + 1];
    fD = one_minus * factor_D[lo] + frac * factor_D[lo + 1];
    fP = one_minus * factor_P[lo] + frac * factor_P[lo + 1];
  }
};

}
}

#endif
