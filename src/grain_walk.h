/* ----------------------------------------------------------------------
   Grain-level (coherent) misorientation walk, shared by the CPU and Kokkos
   additive apps.

   Between lattice level k and k+1 every voxel of grain g receives the SAME
   rotation increment: a fixed angle sqrt(D_g * dx) about an axis drawn from
   a hash of (g, k). The increment is deterministic, so any inheritance path
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

// Rotation increment of grain g from level k to k+1 (unit quaternion, w first).
GW_FN void gw_increment(int g, int k, double angle, double dq[4]) {
  uint64_t h = gw_mix(((uint64_t)(uint32_t)g << 32) ^ (uint64_t)(uint32_t)k);
  double u1 = (double)(h >> 11) * (1.0 / 9007199254740992.0);
  uint64_t h2 = gw_mix(h);
  double u2 = (double)(h2 >> 11) * (1.0 / 9007199254740992.0);
  double z = 2.0 * u1 - 1.0, r = sqrt(1.0 - z * z), phi = 6.283185307179586 * u2;
  double s = sin(0.5 * angle);
  dq[0] = cos(0.5 * angle); dq[1] = s * r * cos(phi); dq[2] = s * r * sin(phi); dq[3] = s * z;
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

// Move an orientation inherited at lattice height z_from (lattice units) to
// height z_to within grain g: compose the level increments in between.
GW_FN void grain_walk_apply(int g, double z_from, double z_to, double angle, double q[4]) {
  int kf = (int)floor(z_from + 0.5), kt = (int)floor(z_to + 0.5);
  if (kf == kt) return;
  double dq[4];
  if (kt > kf) { for (int k = kf; k < kt; k++) { gw_increment(g, k, angle, dq); gw_premultiply(dq, false, q); } }
  else         { for (int k = kf - 1; k >= kt; k--) { gw_increment(g, k, angle, dq); gw_premultiply(dq, true, q); } }
}

#endif
