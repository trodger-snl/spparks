// Check src/ipf_color.h against orix IPFColorKeyTSL reference colours (m-3m by
// default; other Laue groups from gen_refs.py, crystal name as the 5th arg).
// Reference: 300 unit quaternions (w x y z r g b); r g b = orix 0.13 IPFColorKeyTSL colour of the
// CONJUGATE quaternion (orix applies R d; SPPARKS quaternions need R^T d), direction z / x.
// build: g++ -O2 -I ../../src -include stdlib.h -o check check.cpp
// run:   ./check ipf_ref_z.csv 0 0 1 && ./check ipf_ref_x.csv 1 0 0
//        for c in hcp trigonal tetragonal orthorhombic; do
//          ./check ipf_ref_${c}_z.csv 0 0 1 $c && ./check ipf_ref_${c}_x.csv 1 0 0 $c; done
#include "ipf_color.h"
#include <stdio.h>
int main(int argc, char **argv) {
  double d[3] = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
  int crystal = argc > 5 ? SPPARKS_NS::CRYSTAL::index(argv[5]) : 0;
  if (crystal < 0) { printf("unknown crystal %s\n", argv[5]); return 1; }
  FILE *f = fopen(argv[1], "r"); double q[4], ref[3], rgb[3], maxe = 0; int n = 0, bad = 0;
  while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf", q, q+1, q+2, q+3, ref, ref+1, ref+2) == 7) {
    SPPARKS_NS::ipf::quaternion_rgb(q[0], q[1], q[2], q[3], d, rgb, crystal);
    double e = 0; for (int k = 0; k < 3; k++) e = fmax(e, fabs(rgb[k]-ref[k]));
    if (e > maxe) maxe = e; if (e > 1e-3) bad++; n++;
  }
  printf("%s: n=%d max err %.2e bad %d\n", argv[1], n, maxe, bad); return bad != 0;
}
