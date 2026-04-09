#ifndef GREENAM_SCAN_INTEGRATION_H
#define GREENAM_SCAN_INTEGRATION_H
#include "GreenAM_Constants.h"
#include "GreenAM_LaserScan.h"
#include "GreenAM_GaussLegendre.h"

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

template <typename FuncType> struct CallCounter
{
  GREENAM_FUNCTION CallCounter(FuncType & f, unsigned * c) :
    func(f),
    cnt(c)
  {}

  template <typename DataType>
  GREENAM_FUNCTION auto operator()(DataType t) const
  {
    if(cnt != nullptr) (*cnt)++;
    return func(t);
  }

  FuncType & func;
  unsigned * cnt;
};

template <unsigned O>
struct VariableOrderGaussQuadrature
{
  VariableOrderGaussQuadrature() {}
  template<unsigned N, typename F> 
  struct looper 
  {
    template<typename T>
    static auto loop(F & f, T a, T b, unsigned order)
    {
      if(order == N)
      {
        sierra::greenam::GaussLegendreQuadrature<N> glq;
        return glq.integrate(f, a, b);
      }
      else
        return looper<N-1,F>::loop(f,a,b,order);
    }
  };

  template<typename F>
  struct looper<1, F>
  {
    template<typename T>
    static auto loop(F & f, T a, T b, unsigned order)
    {
      T res = (b-a) * f((b+a)/2.);
      return res;
    }
  };

  template<typename T, class F>
  auto integrate(F & f, T a, T b, unsigned order) const
  {
    return looper<O,F>::loop(f,a,b,order);
  }
  unsigned max_order = O;
};

template <typename RealType, typename ArrayType, unsigned MaxOrder>
struct ScanIntegration
{
  GREENAM_FUNCTION ScanIntegration(LaserScan<RealType, ArrayType> & input_scan) :
    scan(input_scan)
  {}

  //Integrate a scan path to get temperature at a point using a variable order method 
  //where the order for each segment is set based on resolving the laser motion and diffusion
  //time to a specified degree
  //Parameters:
  //x [in] x coordinate
  //y [in] y coordinate
  //z [in] z coordinate
  //t [in] time
  //char_length [in] the resolution of the laser motion and diffusion time. Lower values
  //will give lower errors. Empirically, a value of 0.5 gave O(1e-4) error
  //cnt [in/out] if nullptr, do not perform call counts. If not nullptr, store call count
  //in variable pointed to by cnt
  //Returns:
  //integrated value
  GREENAM_FUNCTION RealType integrate_point_adaptive(RealType x, RealType y, RealType z, 
    RealType t, double char_length, unsigned * cnt = nullptr)
  {
    if(t==0.) return 0.;
    unsigned end_idx = binary_search(scan.bounds.data(), t, scan.bounds.size());
    if(std::fabs(t-scan.bounds(end_idx)) <= 4.*epsilon) end_idx--;

    RealType res = 0.;
    RealType avg_sigma = 1./3.*(scan.eprop.sx+scan.eprop.sy+scan.eprop.sz);
    RealType avg_sigma_sq = avg_sigma*avg_sigma;

    for(unsigned i=0; i<=end_idx; i++)
    {
      if(std::fabs(scan.powers(i)) <= 4*sierra::greenam::epsilon) continue;
      auto gge = scan.get_gg_ellipse_for_point(x, y, z, t, i);
      auto gge_cnt = CallCounter<GaussGreenEllips<RealType>>(gge, cnt);
      
      RealType & tstart = scan.bounds(i);
      RealType tend = i+1 >= scan.bounds.size() ? t : 
        (t <= scan.bounds(i+1) + 4.*epsilon ? t : scan.bounds(i+1));

      RealType sigma_eff = 4.*gge.alpha*(t-tend)+avg_sigma_sq;
      RealType quad_size = char_length*char_length*sigma_eff/4./gge.alpha;

      if(scan.speeds(i) > 4*sierra::greenam::epsilon)
      {
        quad_size = std::min(char_length*std::sqrt(sigma_eff)/scan.speeds(i),quad_size);
      }

      RealType n_quad_pts = (tend-tstart)/quad_size;
      unsigned order = std::max(1u, unsigned(std::ceil(n_quad_pts)));
      order = std::min(MaxOrder,order);
      RealType step_size = quad_size*order;

      RealType n_intervals_exact = (tend-tstart)/step_size;
      double n_intervals = std::ceil(n_intervals_exact);
      RealType dt = (tend-tstart)/n_intervals;
      unsigned nint = unsigned(n_intervals);
      for(unsigned j=0; j<nint; j++)
      {
        res += gauss.template integrate<RealType>(gge_cnt, tstart+j*dt, tstart+(j+1)*dt, order);
      }
    }
    return res;
  }

  //Integrate a scan path to get temperature at a point using a brute force method, calling the same 
  //order guass quadrature on each path segement
  //Parameters:
  //x [in] x coordinate
  //y [in] y coordinate
  //z [in] z coordinate
  //t [in] time
  //order [in] quadrature order to use
  //cnt [in/out] if nullptr, do not perform call counts. If not nullptr, store call count
  //in variable pointed to by cnt
  //Returns:
  //integrated value
  GREENAM_FUNCTION RealType integrate_point_fixed(RealType x, RealType y, RealType z, 
    RealType t, unsigned order, unsigned * cnt = nullptr)
  {
    if(order > MaxOrder) order = MaxOrder;
    if(t==0.) return 0.;
    unsigned end_idx = binary_search(scan.bounds.data(), t, scan.bounds.size());
    if(std::fabs(t-scan.bounds(end_idx)) <= 4.*epsilon) end_idx--;

    RealType res = 0.;
    for(unsigned i=0; i<=end_idx; i++)
    {
      if(std::fabs(scan.powers(i)) <= 4*sierra::greenam::epsilon) continue;
      RealType & tstart = scan.bounds(i);
      RealType tend = i+1 >= scan.bounds.size() ? t : 
        (t <= scan.bounds(i+1) + 4.*epsilon ? t : scan.bounds(i+1));

      auto gge = scan.get_gg_ellipse_for_point(x, y, z, t, i);
      auto gge_cnt = CallCounter<GaussGreenEllips<RealType>>(gge, cnt);

      res += gauss.template integrate<RealType>(gge_cnt, tstart, tend, order);
    }
    return res;
  }

  LaserScan<RealType, ArrayType> & scan;
  static const VariableOrderGaussQuadrature<MaxOrder> gauss;
};
template <typename RealType, typename ArrayType, unsigned MaxOrder> const VariableOrderGaussQuadrature<MaxOrder> ScanIntegration<RealType,ArrayType,MaxOrder>::gauss;
}
}
#endif
