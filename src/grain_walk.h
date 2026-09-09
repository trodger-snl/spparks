/* ----------------------------------------------------------------------
   Grain-level (coherent) misorientation walk, shared by the CPU and Kokkos
   additive apps.

   Between lattice level k and k+1 every voxel of grain g receives the SAME
   rotation increment: a Gaussian rotation vector of RMS angle sqrt(D_g * dx)
   drawn from a hash of (g, k), optionally correlated between levels (rho). The increment is deterministic, so any inheritance path
   through the grain composes the same product and the coherent rotation of
   the grain at level z is a function of (g, z) only; no per-grain state is
   stored. The per-voxel walk (misorientation_diffusion) adds intragranular
   spread on top of it.
------------------------------------------------------------------------- */
#ifndef SPK_GRAIN_WALK_H
#define SPK_GRAIN_WALK_H

#include <stdint.h>
#include <math.h>

#ifdef KOKKOS_INLINE_FUNCTION
#define GW_FN KOKKOS_INLINE_FUNCTION
#else
#define GW_FN inline
#endif

GW_FN uint64_t gw_mix(uint64_t x) {           // splitmix64
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Three standard Gaussians hashed from (g, k) (Box-Muller on hashed uniforms).
GW_FN void gw_random_gauss3(int g, int k, double out[3]) {
  uint64_t h = gw_mix(((uint64_t)(uint32_t)g << 32) ^ (uint64_t)(uint32_t)k);
  double u[4];
  for (int i = 0; i < 4; i++) { u[i] = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0); h = gw_mix(h); }
  double r1 = sqrt(-2.0 * log(u[0])), r2 = sqrt(-2.0 * log(u[2]));
  out[0] = r1 * cos(6.283185307179586 * u[1]);
  out[1] = r1 * sin(6.283185307179586 * u[1]);
  out[2] = r2 * cos(6.283185307179586 * u[3]);
}

// Rotation VECTOR v_k (angle * axis, radians) of grain g for the increment
// from level k to k+1: a 3-D Gaussian with RMS |v| = rms per level.
// rho = 0: independent per level (memoryless walk). rho > 0: v_k is an
// AR(1) / Ornstein-Uhlenbeck sequence in the level index,
// v_k = rho v_{k-1} + sqrt(1-rho^2) v'_k, evaluated from level 0 so it stays a
// pure function of (g, k): persistent rotation (lattice bending) with
// correlation length ~ dx/(1-rho); the long-range accumulation is
// rms^2 (1+rho)/(1-rho) per level, and the accumulated rotation across
// grains is Gaussian (Maxwell magnitude) in both regimes.
GW_FN void gw_state(int g, int k, double rms, double rho, double v[3]) {
  double sig = rms / 1.7320508075688772;
  if (k < 0) k = 0;
  if (rho <= 0.0) { gw_random_gauss3(g, k, v); v[0] *= sig; v[1] *= sig; v[2] *= sig; return; }
  gw_random_gauss3(g, 0, v); v[0] *= sig; v[1] *= sig; v[2] *= sig;
  double w = sqrt(1.0 - rho * rho) * sig;
  for (int j = 1; j <= k; j++) {
    double r[3]; gw_random_gauss3(g, j, r);
    v[0] = rho * v[0] + w * r[0]; v[1] = rho * v[1] + w * r[1]; v[2] = rho * v[2] + w * r[2];
  }
}

// Unit quaternion (w first) of the rotation vector v.
GW_FN void gw_dq_from_v(const double v[3], double dq[4]) {
  double ang = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (ang < 1e-300) { dq[0] = 1.0; dq[1] = dq[2] = dq[3] = 0.0; return; }
  double s = sin(0.5 * ang) / ang;
  dq[0] = cos(0.5 * ang); dq[1] = s * v[0]; dq[2] = s * v[1]; dq[3] = s * v[2];
}

GW_FN void gw_increment(int g, int k, double rms, double rho, double dq[4]) {
  double v[3]; gw_state(g, k, rms, rho, v); gw_dq_from_v(v, dq);
}

// q <- dq (x) q  (rotation applied in the sample frame; conj = true applies dq^-1)
GW_FN void gw_premultiply(const double dq[4], bool conj, double q[4]) {
  double w1 = dq[0], x1 = conj ? -dq[1] : dq[1], y1 = conj ? -dq[2] : dq[2], z1 = conj ? -dq[3] : dq[3];
  double w2 = q[0], x2 = q[1], y2 = q[2], z2 = q[3];
  double r[4] = { w1*w2 - x1*x2 - y1*y2 - z1*z2,
                  w1*x2 + x1*w2 + y1*z2 - z1*y2,
                  w1*y2 - x1*z2 + y1*w2 + z1*x2,
                  w1*z2 + x1*y2 - y1*x2 + z1*w2 };
  double n = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3]);
  if (n > 0.0) { r[0] /= n; r[1] /= n; r[2] /= n; r[3] /= n; }
  q[0] = r[0]; q[1] = r[1]; q[2] = r[2]; q[3] = r[3];
}

// Move an orientation inherited at level coordinate z_from (lattice units of
// height or path length) to z_to within grain g: compose the level increments
// in between. `state(g, k, v)` supplies the rotation vector of level k -- the
// direct recurrence (GwDirect) or a device cache of it (Kokkos app).
template <class State>
GW_FN void grain_walk_apply_f(const State &state, int g, double z_from, double z_to, double q[4]) {
  int kf = (int)floor(z_from + 0.5), kt = (int)floor(z_to + 0.5);
  if (kf == kt) return;
  double v[3], dq[4];
  if (kt > kf) { for (int k = kf; k < kt; k++) { state(g, k, v); gw_dq_from_v(v, dq); gw_premultiply(dq, false, q); } }
  else         { for (int k = kf - 1; k >= kt; k--) { state(g, k, v); gw_dq_from_v(v, dq); gw_premultiply(dq, true, q); } }
}

struct GwDirect {
  double rms, rho;
  GW_FN void operator()(int g, int k, double v[3]) const { gw_state(g, k, rms, rho, v); }
};

GW_FN void grain_walk_apply(int g, double z_from, double z_to, double rms, double rho, double q[4]) {
  GwDirect d; d.rms = rms; d.rho = rho;
  grain_walk_apply_f(d, g, z_from, z_to, q);
}

#endif
