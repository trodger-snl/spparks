/* ----------------------------------------------------------------------
   Inverse-pole-figure colour of a unit quaternion for a sample direction,
   cubic (m-3m) symmetry, EDAX/TSL colour key as implemented in MTEX and
   orix (IPFColorKeyTSL): red [001], green [101], blue [111], hue from the
   azimuth about the sector barycentre (corrected so each triangle edge
   spans one third of the hue circle), lightness from the distance to the
   sector boundary. Same convention as spparks-micro-analysis
   generate_ipf_colors: the crystal-frame direction is R(q) d.
------------------------------------------------------------------------- */
#ifndef SPK_IPF_COLOR_H
#define SPK_IPF_COLOR_H

#include <math.h>

namespace SPPARKS_NS {
namespace ipf {

static const double PI2 = 6.283185307179586;
static const int NTAB = 1000;

inline double dot3(const double *a, const double *b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline void cross3(const double *a, const double *b, double *c) {
  c[0] = a[1]*b[2]-a[2]*b[1]; c[1] = a[2]*b[0]-a[0]*b[2]; c[2] = a[0]*b[1]-a[1]*b[0];
}
inline double unit3(double *a) {
  double n = sqrt(dot3(a,a));
  if (n > 0.0) { a[0] /= n; a[1] /= n; a[2] /= n; }
  return n;
}
inline double angle3(const double *a, const double *b) {
  double c = dot3(a,b)/sqrt(dot3(a,a)*dot3(b,b));
  if (c > 1.0) c = 1.0; if (c < -1.0) c = -1.0;
  return acos(c);
}

// m-3m fundamental sector [001]-[101]-[111]: barycentre, vertices, normals (orix Oh)
struct Sector {
  double c[3], rx[3], ry[3], v[3][3], n[3][3];
  double az[NTAB], cum[NTAB];      // azimuth correction table (MTEX correctAngle)
  Sector() {
    double c0[3] = {0.42815235, 0.19245009, 0.76148568};
    for (int i = 0; i < 3; i++) c[i] = c0[i];
    unit3(c);
    double v0[3][3] = {{1,1,1},{1,0,1},{0,0,1}};
    double n0[3][3] = {{1,-1,0},{-1,0,1},{0,1,0}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { v[i][j] = v0[i][j]; n[i][j] = n0[i][j]; }
    for (int i = 0; i < 3; i++) unit3(v[i]);
    // rx = north pole - centre, made orthogonal to the centre; ry = c x rx
    double r0[3] = {-c[0], -c[1], 1.0-c[2]};
    double d = dot3(r0,c);
    for (int i = 0; i < 3; i++) rx[i] = r0[i] - d*c[i];
    unit3(rx); cross3(c,rx,ry); unit3(ry);
    // correction table: boundary distance vs raw azimuth, each edge renormalised to 1/3
    double base[3], polar[NTAB-1];
    cross3(r0,c,base); unit3(base);
    for (int i = 0; i < NTAB; i++) az[i] = PI2*i/(NTAB-1);
    for (int i = 0; i < NTAB-1; i++) {
      double t = az[i], ct = cos(t), st = sin(t), cb = dot3(c,base), cxb[3], nrm[3], x[3];
      cross3(c,base,cxb);
      for (int k = 0; k < 3; k++) nrm[k] = base[k]*ct + cxb[k]*st + c[k]*cb*(1.0-ct);
      double p = 1e300;
      for (int j = 0; j < 3; j++) { cross3(n[j],nrm,x); double a = angle3(x,c); if (a < p) p = a; }
      polar[i] = p;
    }
    double va[3];
    for (int j = 0; j < 3; j++) va[j] = raw_azimuth(v[j]);
    for (int a = 0; a < 3; a++) for (int b = a+1; b < 3; b++) if (va[b] < va[a]) { double t = va[a]; va[a] = va[b]; va[b] = t; }
    int a3[3], na = 0;
    for (int j = 0; j < 3; j++) a3[na++] = (int) floor(NTAB*va[j]/PI2 + 0.5);
    int lo0 = 0, b0 = a3[0], b1 = a3[1];
    if (a3[0] < 10) { b0 = a3[1]; b1 = a3[2]; }
    int lim[4] = {lo0, b0, b1, NTAB-1};
    for (int s = 0; s < 3; s++) {
      double sum = 0.0;
      for (int i = lim[s]; i < lim[s+1]; i++) sum += polar[i];
      for (int i = lim[s]; i < lim[s+1]; i++) polar[i] /= sum/3.0;
    }
    double tot = 0.0;
    for (int i = 0; i < NTAB-1; i++) tot += polar[i];
    cum[0] = 0.0;
    for (int i = 1; i < NTAB; i++) cum[i] = cum[i-1] + PI2*polar[i-1]/tot;
  }
  double raw_azimuth(const double *p) const {
    double d[3] = {p[0]-c[0], p[1]-c[1], p[2]-c[2]};
    unit3(d);
    double a = atan2(dot3(ry,d), dot3(rx,d));
    if (a != a) return 0.0;
    a = fmod(a, PI2); if (a < 0.0) a += PI2;
    return a;
  }
  double azimuth(const double *p) const {          // corrected (np.interp on the table)
    double a = raw_azimuth(p);
    int i = (int) (a/PI2*(NTAB-1)); if (i >= NTAB-1) i = NTAB-2; if (i < 0) i = 0;
    double f = (a - az[i])/(az[i+1]-az[i]);
    return cum[i] + f*(cum[i+1]-cum[i]);
  }
  double polar(const double *p) const {            // distance to the boundary, 0 at centre, 1 at edge
    double vcn[3], bp[3], mc[3] = {-c[0],-c[1],-c[2]}, mp[3] = {-p[0],-p[1],-p[2]};
    cross3(p,c,vcn); unit3(vcn);
    double best = 1e300;
    for (int j = 0; j < 3; j++) {
      cross3(vcn,n[j],bp);
      double d = 1.0;
      if (unit3(bp) > 0.0) { double den = angle3(mc,bp); d = den > 0.0 ? angle3(mp,bp)/den : 1.0; if (d != d) d = 1.0; }
      if (d < best) best = d;
    }
    return best;
  }
};

inline const Sector &sector() { static Sector s; return s; }

// direction p (any frame vector) -> rgb; p is folded into the sector first
inline void direction_rgb(const double *pin, double *rgb) {
  double a[3] = {fabs(pin[0]), fabs(pin[1]), fabs(pin[2])};
  for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++) if (a[j] > a[i]) { double t = a[i]; a[i] = a[j]; a[j] = t; }
  double p[3] = {a[1], a[2], a[0]};               // sector coordinates: z >= x >= y >= 0
  if (unit3(p) == 0.0) { rgb[0] = rgb[1] = rgb[2] = 0.0; return; }
  const Sector &s = sector();
  double h = fmod(s.azimuth(p)/PI2, 1.0);
  double l = 0.5 + 0.5*s.polar(p);
  // hsl (h, 1, l) -> hsv -> rgb (MTEX hsl2hsv, matplotlib hsv_to_rgb)
  double l2 = 2.0*l, s2 = (l2 <= 1.0) ? l2 : 2.0-l2;
  double sat = (l2+s2) > 0.0 ? 2.0*s2/(l2+s2) : 0.0, val = 0.5*(l2+s2);
  double hh = h*6.0; int i = (int) floor(hh); double f = hh - i;
  double pp = val*(1.0-sat), q = val*(1.0-sat*f), t = val*(1.0-sat*(1.0-f));
  switch (i % 6) {
    case 0: rgb[0] = val; rgb[1] = t;   rgb[2] = pp;  break;
    case 1: rgb[0] = q;   rgb[1] = val; rgb[2] = pp;  break;
    case 2: rgb[0] = pp;  rgb[1] = val; rgb[2] = t;   break;
    case 3: rgb[0] = pp;  rgb[1] = q;   rgb[2] = val; break;
    case 4: rgb[0] = t;   rgb[1] = pp;  rgb[2] = val; break;
    default: rgb[0] = val; rgb[1] = pp; rgb[2] = q;   break;
  }
}

// quaternion (w,x,y,z) and unit sample direction d -> rgb of R(q) d
inline void quaternion_rgb(double w, double x, double y, double z, const double *d, double *rgb) {
  double n = sqrt(w*w+x*x+y*y+z*z);
  if (n == 0.0) { rgb[0] = rgb[1] = rgb[2] = 0.0; return; }
  w /= n; x /= n; y /= n; z /= n;
  double p[3] = {(1.0-2.0*(y*y+z*z))*d[0] + 2.0*(x*y-w*z)*d[1] + 2.0*(x*z+w*y)*d[2],
                 2.0*(x*y+w*z)*d[0] + (1.0-2.0*(x*x+z*z))*d[1] + 2.0*(y*z-w*x)*d[2],
                 2.0*(x*z-w*y)*d[0] + 2.0*(y*z+w*x)*d[1] + (1.0-2.0*(x*x+y*y))*d[2]};
  direction_rgb(p, rgb);
}

} // namespace ipf
} // namespace SPPARKS_NS
#endif
