#ifndef GREENAM_LASER_SCAN_H
#define GREENAM_LASER_SCAN_H

#include "GreenAM_Properties.h"

#ifndef GREENAM_FUNCTION
#define GREENAM_FUNCTION
#endif
#ifndef GREENAM_FORCEINLINE_FUNCTION
#define GREENAM_FORCEINLINE_FUNCTION inline
#endif
#ifndef GREENAM_INLINE_FUNCTION
#define GREENAM_INLINE_FUNCTION inline
#endif

namespace sierra {
namespace greenam {

template <typename T>
struct GaussGreenEllips
{
  GREENAM_FUNCTION GaussGreenEllips(T xi = 0., T yi = 0., T zi = 0., 
    T ti =0., T xl = 0., T yl = 0., T zl = 0.,
    T p = 0., T sp = 0., T tzero = 0., 
    T nxi = 1., T nyi = 0., T a = 0., 
    T ci = 0., EllipsoidProperties<T> ep = EllipsoidProperties<T>()) :
      x(xi), y(yi), z(zi), t(ti), alpha(a), c(ci), z0(zl),  
      xl(xl), yl(yl), power(p), speed(sp), t0(tzero), nx(nxi), 
      ny(nyi), eprop(ep)
  {}
  GREENAM_INLINE_FUNCTION void coord_transform(T s, T & xt, T & yt, T & zt, T & tt) const
  {
    T c_x = x - xl;
    T c_y = y - yl;
    xt = c_x * nx + c_y * ny;
    yt = c_x * -ny + c_y * nx;
    zt = z - z0;
    tt = s - t0;
  }
  GREENAM_FUNCTION T operator()(T s) const
  {
    T st = 4.*alpha*(t-s);
    T xt, yt, zt, tt;
    coord_transform(s, xt, yt, zt, tt);
    T varx = 1. / (eprop.sx*eprop.sx + st);
    T vary = 1. / (eprop.sy*eprop.sy + st);
    T varz = 1. / (eprop.sz*eprop.sz + st);
    T xeff = xt-speed*tt;
    T res = 2. * power / c * invPi3_2 * 
      std::exp(-xeff * xeff * varx - yt*yt*vary 
      - zt*zt*varz);
    res *= std::sqrt(varx*vary*varz);
    return res;
  }
  T x, y, z, t; 
  T alpha, c, z0;
  unsigned my_idx = uint_max;
  T xl;
  T yl;
  T power; 
  T speed; 
  T t0;
  T nx;
  T ny;
  EllipsoidProperties<T> eprop;
};

//Class to compute the temperature response of a laser scan at a set of provided space-time points
//Template Parameters:
//MemorySpace : Memory space where computations will be carried out. Execution is done in this space's
//  associated execution space
template <typename RealType, typename ArrayType>
struct LaserScan {

  //constructor
  //Parameters:
  //therm [in] thermal properties of workpiece {k, rho, cp}
  //elips [in] laser properties {sigma_x, sigma_y, sigma_z}
  //ts, xs, ys, ps [in] Arrays defining laser scan pattern in the form {time, laser_x, laser_y, power}
  //z [in] location of laser in z (depth) dimension
  //nq [in] number of initial quadrature order to use for integration
  //i_size [in] maximum time interval to integrate over. Intervals larger than this will be sub-divided
  //nr [in] maximum number of times to refine the quadrature order if integral is not converged
  LaserScan(ThermalProperties<RealType> therm, 
    EllipsoidProperties<RealType> elips, 
    const ArrayType & ts, 
    const ArrayType & xs,
    const ArrayType & ys,
    const ArrayType & ps,
    double z) :
    eprop(std::move(elips)),
    zl(z),
    alpha(therm.k / therm.cp / therm.rho),
    c(therm.cp * therm.rho)
  {
    unsigned nlines = ts.size()-1;
    if(nlines <= 0) throw std::runtime_error("GreenAM::LaserScan::LaserScan: Not enough input lines to define scan");
    if(nlines+1 != xs.size()) throw std::runtime_error("GreenAM::LaserScan::LaserScan: All input arrays must be same length");
    if(nlines+1 != ys.size()) throw std::runtime_error("GreenAM::LaserScan::LaserScan: All input arrays must be same length");
    if(nlines+1 != ps.size()) throw std::runtime_error("GreenAM::LaserScan::LaserScan: All input arrays must be same length");

    bounds = ArrayType("Bounds", nlines);
    xl = ArrayType("XLaser", nlines);
    yl = ArrayType("YLaser", nlines);
    powers = ArrayType("Powers", nlines);
    normals_x = ArrayType("Normals_X", nlines);
    normals_y = ArrayType("Normals_Y", nlines);
    speeds = ArrayType("Speeds", nlines);

    for(unsigned p=0; p<nlines; p++)
    {
      bounds(p) = ts(p);
      xl(p) = xs(p);
      yl(p) = ys(p);
      powers(p) = ps(p);
      const RealType & t1 = ts(p);
      const RealType & t2 = ts(p+1);
      const RealType & x1 = xs(p);
      const RealType & x2 = xs(p+1);
      const RealType & y1 = ys(p);
      const RealType & y2 = ys(p+1);

      RealType vx = (x2-x1)/(t2-t1);
      RealType vy = (y2-y1)/(t2-t1);
      RealType speed = std::sqrt(vx*vx+vy*vy);
      if(speed > epsilon)
      {
        vx /= speed;
        vy /= speed;
      }
      else
      {
        vx = 1.;
        vy = 0.;
      }
      normals_x(p) = vx;
      normals_y(p) = vy;
      speeds(p) = speed;
    }

    bounds.commit();
    xl.commit();
    yl.commit();
    powers.commit();
    normals_x.commit();
    normals_y.commit();
    speeds.commit();  
  }

  //Get function to integrate for a provided spatial location and scan line
  //Parameters:
  //x [in] X coordinate
  //y [in] Y coordinate
  //z [in] Z coordinate
  //t [in] time
  //l [in] line number
  //Returns:
  //GaussGreenEllips at specified location and time for integration
  GREENAM_FUNCTION GaussGreenEllips<RealType> get_gg_ellipse_for_point(const RealType & x, const RealType & y, 
    const RealType & z, const RealType & t, unsigned l) const
  {
    return GaussGreenEllips<RealType>(x, y, z, t, xl(l), yl(l), zl, powers(l), speeds(l), bounds(l), 
      normals_x(l), normals_y(l), alpha, c, eprop);
  }

  GREENAM_FUNCTION void get_laser_location_at_time(const RealType & t, RealType & x, RealType & y)
  {
    unsigned idx = binary_search(bounds.data(), t, bounds.size());
    RealType dt = t - bounds(idx);

    x = xl(idx) + speeds(idx)*normals_x(idx)*dt;
    y = yl(idx) + speeds(idx)*normals_y(idx)*dt;
  }

  GREENAM_FUNCTION RealType get_laser_power_at_time(const RealType & t)
  {
    unsigned idx = binary_search(bounds.data(), t, bounds.size());
    
    return powers(idx);
  }

  GREENAM_FUNCTION RealType min_dist_to_scan_line(const RealType & x, const RealType & y, const RealType & z, unsigned i) const
  {
    if(std::fabs(powers(i)) < epsilon)
    {
      return double_max;
    }
    RealType vx = normals_x(i)*speeds(i);
    RealType vy = normals_y(i)*speeds(i);
    RealType & x0 = xl(i);
    RealType & y0 = yl(i);
    RealType dt;
    if(i==(bounds.size()-1)) dt = double_max;
    else dt = bounds(i+1)-bounds(i);
    auto dist_func = [&](RealType & t)
    {
      RealType xdist = (x-x0-vx*t)/eprop.sx;
      RealType ydist = (y-y0-vy*t)/eprop.sy;
      RealType zdist = (z-zl)/eprop.sz;
      RealType ans = xdist*xdist + ydist*ydist + zdist*zdist;
      return ans;
    };

    RealType sx2 = eprop.sx*eprop.sx;
    RealType sy2 = eprop.sy*eprop.sy;
    RealType mint = vx == 0. && vy == 0. ? 
      -1 : (sx2*vy*(y-y0)+sy2*vx*(x-x0))/(sx2*vy*vy+sy2*vx*vx);
    RealType dist = std::fmin(dist_func(0),dist_func(dt));
    if(mint > 0. && mint < dt) dist = std::fmin(dist, dist_func(mint));
    return dist;
  }
  
  ~LaserScan() = default;

  ArrayType normals_x;
  ArrayType normals_y;
  ArrayType speeds;
  ArrayType bounds;
  ArrayType xl;
  ArrayType yl;
  ArrayType powers;
  EllipsoidProperties<RealType> eprop;
  RealType zl;
  RealType alpha;
  RealType c;
};
}
}
#endif
