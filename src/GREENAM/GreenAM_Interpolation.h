#ifndef GREENAM_INTERPOLATION_H
#define GREENAM_INTERPOLATION_H

#include "GreenAM_Constants.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sierra {
namespace greenam {

/*
 *  Copyright Nick Thompson, 2017
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 *  Given N samples (t_i, y_i) which are irregularly spaced, this routine constructs an
 *  interpolant s which is constructed in O(N) time, occupies O(N) space, and can be evaluated in O(N) time.
 *  The interpolation is stable, unless one point is incredibly close to another, and the next point is incredibly far.
 *  The measure of this stability is the "local mesh ratio", which can be queried from the routine.
 *  Pictorially, the following t_i spacing is bad (has a high local mesh ratio)
 *  ||             |      | |                           |
 *  and this t_i spacing is good (has a low local mesh ratio)
 *  |   |      |    |     |    |        |    |  |    |
 *
 *
 *  If f is C^{d+2}, then the interpolant is O(h^(d+1)) accurate, where d is the interpolation order.
 *  A disadvantage of this interpolant is that it does not reproduce rational functions; for example, 1/(1+x^2) is not interpolated exactly.
 *
 *  References:
 *  Floater, Michael S., and Kai Hormann. "Barycentric rational interpolation with no poles and high rates of approximation."
*      Numerische Mathematik 107.2 (2007): 315-331.
 *  Press, William H., et al. "Numerical recipes third edition: the art of scientific computing." Cambridge University Press 32 (2007): 10013-2473.
 */

template<class Real>
class barycentric_rational
{
public:
    template <class InputIterator1, class InputIterator2>
    barycentric_rational(const Real* const x, const Real* const y, size_t len, size_t approximation_order = 3)
    {
        if (approximation_order >= len)
        {
            throw std::runtime_error("Approximation order must be < data length.");
        }

        // Big sad memcpy.
        m_x.resize(len);
        m_y.resize(len);
        for(unsigned i = 0; i <= len; ++i)
        {
            m_x[i] = x[i];
            m_y[i] = y[i];
        }
        calculate_weights(approximation_order);
    }

    barycentric_rational(std::vector<Real>&& x, std::vector<Real>&& y, size_t approximation_order = 3) : m_x(std::move(x)), m_y(std::move(y))
    {
      if(m_x.size() != m_y.size()) throw std::runtime_error("There must be the same number of abscissas and ordinates.");
      if(approximation_order >= m_x.size()) throw std::runtime_error("Approximation order must be < data length.");
      if(!std::is_sorted(m_x.begin(), m_x.end())) throw std::runtime_error("The abscissas must be listed in increasing order x[0] < x[1] < ... < x[n-1].");
      calculate_weights(approximation_order);
    }

    Real operator()(Real x) const;

    Real prime(Real x) const;

    // The barycentric weights are not really that interesting; except to the unit tests!
    Real weight(size_t i) const { return m_w[i]; }

    std::vector<Real>&& return_x()
    {
        return std::move(m_x);
    }

    std::vector<Real>&& return_y()
    {
        return std::move(m_y);
    }

private:

    void calculate_weights(size_t approximation_order);

    std::vector<Real> m_x;
    std::vector<Real> m_y;
    std::vector<Real> m_w;
};

template<class Real>
void barycentric_rational<Real>::calculate_weights(size_t approximation_order)
{
    using std::abs;
    int64_t n = m_x.size();
    m_w.resize(n, 0);
    for(int64_t k = 0; k < n; ++k)
    {
        int64_t i_min = (std::max)(k - (int64_t) approximation_order, (int64_t) 0);
        int64_t i_max = k;
        if (k >= n - (std::ptrdiff_t)approximation_order)
        {
            i_max = n - approximation_order - 1;
        }

        for(int64_t i = i_min; i <= i_max; ++i)
        {
            Real inv_product = 1;
            int64_t j_max = (std::min)(static_cast<int64_t>(i + approximation_order), static_cast<int64_t>(n - 1));
            for(int64_t j = i; j <= j_max; ++j)
            {
                if (j == k)
                {
                    continue;
                }

                Real diff = m_x[k] - m_x[j];
                using std::numeric_limits;
                if (abs(diff) < epsilon)
                {
                   std::string msg = std::string("Spacing between  x[")
                      + std::to_string(k) + std::string("] and x[")
                      + std::to_string(i) + std::string("] is smaller than ")
                      + std::to_string(epsilon);
                    throw std::logic_error(msg);
                }
                inv_product *= diff;
            }
            if (i % 2 == 0)
            {
                m_w[k] += 1/inv_product;
            }
            else
            {
                m_w[k] -= 1/inv_product;
            }
        }
    }
}


template<class Real>
Real barycentric_rational<Real>::operator()(Real x) const
{
    Real numerator = 0;
    Real denominator = 0;
    for(size_t i = 0; i < m_x.size(); ++i)
    {
        // Presumably we should see if the accuracy is improved by using ULP distance of say, 5 here, instead of testing for floating point equality.
        // However, it has been shown that if x approx x_i, but x != x_i, then inaccuracy in the numerator cancels the inaccuracy in the denominator,
        // and the result is fairly accurate. See: http://epubs.siam.org/doi/pdf/10.1137/S0036144502417715
        if (x == m_x[i])
        {
            return m_y[i];
        }
        Real t = m_w[i]/(x - m_x[i]);
        numerator += t*m_y[i];
        denominator += t;
    }
    return numerator/denominator;
}

/*
 * A formula for computing the derivative of the barycentric representation is given in
 * "Some New Aspects of Rational Interpolation", by Claus Schneider and Wilhelm Werner,
 * Mathematics of Computation, v47, number 175, 1986.
 * http://www.ams.org/journals/mcom/1986-47-175/S0025-5718-1986-0842136-8/S0025-5718-1986-0842136-8.pdf
 * and reviewed in
 * Recent developments in barycentric rational interpolation
 * Jean-Paul Berrut, Richard Baltensperger and Hans D. Mittelmann
 *
 * Is it possible to complete this in one pass through the data?
 */

template<class Real>
Real barycentric_rational<Real>::prime(Real x) const
{
    Real rx = this->operator()(x);
    Real numerator = 0;
    Real denominator = 0;
    for(size_t i = 0; i < m_x.size(); ++i)
    {
        if (x == m_x[i])
        {
            Real sum = 0;
            for (size_t j = 0; j < m_x.size(); ++j)
            {
                if (j == i)
                {
                    continue;
                }
                sum += m_w[j]*(m_y[i] - m_y[j])/(m_x[i] - m_x[j]);
            }
            return -sum/m_w[i];
        }
        Real t = m_w[i]/(x - m_x[i]);
        Real diff = (rx - m_y[i])/(x-m_x[i]);
        numerator += t*diff;
        denominator += t;
    }

    return numerator/denominator;
}

//*******************
//End Boost Copyright
//*******************

template <typename ArrayType, typename T>
std::pair<ArrayType,ArrayType> fill_arrays_from_map(const std::map<T,T> & data,
  unsigned data_size)
{
  unsigned cnt = 0;
  ArrayType ts("Ts", data_size);
  ArrayType vals("Vals", data_size);
  for (auto it = data.begin(); it != data.end(); ++it)
  {
    ts(cnt) = it->first;
    vals(cnt) = it->second;
    cnt++;
  }
  return std::make_pair(ts,vals);
}

//Create a Chebyshev interpolant for function f using successive order refinement
//Parameters:
//f [in] Function to interpolate
//a [in] start of interpolation interval
//b [in] ending of interpolation interval
//tol [in] tolerance at which interpolate is considered converged
//conv [out] set to true if interpolant converged in specified number of max refinements, false otherwise
//initial_order [in] initial order of the interpolant
//max_refinements [in] maximum number of times to perform order refinement
//Returns:
//Pair of vectors <x values, function values>
template <typename FuncType>
std::pair<std::vector<double>, std::vector<double>> 
p_refinement(FuncType f, double a, double b, 
  double tol, bool & conv, unsigned initial_order = 2, unsigned max_refinements = 10)
{
  double apb = (a+b)/2.;
  double bma = (b-a)/2.;
  unsigned n = initial_order;

  std::vector<double> xs(n+1);
  std::vector<double> ys(n+1);
  for(unsigned i=0; i<=n; i++)
  {
    xs[i] = apb - bma * 
      cos(sierra::greenam::PI*static_cast<double>(i)/static_cast<double>(n));
    ys[i] = f(xs[i]);
  }
  barycentric_rational<double> interp(std::move(xs), std::move(ys), n);

  for(unsigned nref = 0; nref < max_refinements; nref++)
  {
    std::vector<double> new_xs(2*n+1);
    std::vector<double> new_ys(2*n+1);
    double on = 1./static_cast<double>(2*n);
    conv = true;
    for(unsigned i=0; i<n; i++)
    {
      double j = static_cast<double>(2*i+1);
      double new_x = apb - bma * cos(sierra::greenam::PI*j*on);
      double new_y = f(new_x);
      double interp_y = interp(new_x);
      if(std::fabs(new_y-interp_y) > tol + std::fabs(new_y*tol))
      {
        conv = false;
      }
      new_xs[2*i+1] = new_x;
      new_ys[2*i+1] = new_y;
    }
    if(conv) break;
    else
    {
      xs = interp.return_x();
      ys = interp.return_y();
      for(unsigned i=0; i<=n; i++)
      {
        new_xs[2*i] = xs[i];
        new_ys[2*i] = ys[i];
      }
      interp = barycentric_rational<double>(
        std::move(new_xs), std::move(new_ys));
      n *= 2;
    }
  }
  return std::make_pair(interp.return_x(), interp.return_y());
}

//Create an H0 continuous rational interpolant for function f using successive h refinement
//Parameters:
//f [in] Function to interpolate
//ti [in] start of interpolation interval
//tf [in] ending of interpolation interval
//tol [in] tolerance at which interpolate is considered converged
//is_converged [out] set to true if interpolant converged in specified number of max intervals, false otherwise
//interval_size [in] max initial interval size. If negative, set to tf-ti
//order [in] order of piecewise interpolants
//max_length [in] maximum number of points
//Returns:
//Pair of views <x values, function values>
template<typename T, typename ArrayType>
class h_refinement
{
public:
  template <typename FuncType>
  h_refinement(FuncType f, double ti, double tf, 
    double tol, double interval_size = -1., 
    unsigned order = 1, unsigned max_length = 1000)
  {
    if(interval_size <= 0.) interval_size = tf-ti;
    std::vector<T> xs(order+1);
    std::vector<T> yvec(order+1);
    T * ys = yvec.data();
    T dx = 1./T(order);
    for(unsigned i=0; i<=order; i++)
    {
      xs[i] = i*dx;
    }
    barycentric_rational<T> interp(std::move(xs), std::move(yvec), order);

    std::map<T, T> data;
    std::map<T, bool> conv;

    double dnint = std::ceil((tf - ti) / interval_size);
    unsigned nint = (unsigned)(std::ceil(dnint/order))*order;
    unsigned data_size = nint+1;

    double dt = (tf - ti)/nint;
    for(unsigned i=0; i<=nint; i++)
    {
      double t = ti + i*dt;
      data[T(t)] = f(T(t));

      if(i==nint) conv[T(t)] = true;
      else if((i % order) == 0u) conv[T(t)] = false;
    }

    bool new_insertions = true;
    std::vector<T> new_vals(order);
    std::vector<T> new_ts(order);

    while(new_insertions)
    {
      std::tie(ts,vals) = fill_arrays_from_map<ArrayType>(data, data_size);
      new_insertions = false;
      for (unsigned i=0; i<ts.size(); i += order) 
      {
        if(conv.at(ts(i))) continue;
        const T & t0 = ts(i);
        const T & t1 = ts(i+order);
        T oIsize = 1./(t1-t0);
        bool int_conv = true;
        for (unsigned p=0; p<=order; p++) ys[p] = vals(i+p);
        for (unsigned p=0; p<order; p++)
        {
          new_ts[p] = 0.5*(ts(i+p)+ts(i+p+1));
          new_vals[p] = f(new_ts[p]);
          T interp_val = interp((new_ts[p]-t0)*oIsize);

          if(std::fabs(new_vals[p]-interp_val) > tol + std::fabs(new_vals[p]*tol))
          {
            int_conv = false;
          }
        }
        if(int_conv) conv[ts(i)] = true;
        else
        {
          for(unsigned p=0; p<order; p++) data[new_ts[p]] = new_vals[p];
          auto it = data.find(ts(i));
          it = std::next(it, order);
          conv[it->first] = false;
          new_insertions = true;
          data_size += order;
          if(data_size >= max_length)
          {
            is_converged = false;
            std::tie(ts,vals) = fill_arrays_from_map<ArrayType>(data, data_size);
            return;
          }
        }
      }
    }
    is_converged = true;
    return;
  }
  ArrayType ts;
  ArrayType vals;
  bool is_converged;
};

//Create an HN continous rational interpolant for function f using successive point additions
//Parameters:
//f [in] Function to interpolate
//ti [in] start of interpolation interval
//tf [in] ending of interpolation interval
//tol [in] tolerance at which interpolate is considered converged
//is_converged [out] set to true if interpolant converged in specified number of max intervals, false otherwise
//interval_size [in] max initial interval size. If negative, set to tf-ti
//order [in] order of piecewise interpolants
//max_length [in] maximum number of points
//Returns:
//Pair of views <x values, function values>
template<typename ArrayType, typename FuncType>
std::pair<ArrayType,  ArrayType> 
piecewise_global_refinement(FuncType f, double ti, double tf, 
  double tol, bool & is_converged, double interval_size = -1., 
  unsigned order = 3, unsigned max_length = 1000)
{
  if(interval_size <= 0.) interval_size = tf-ti;
  std::map<double, double> data;
  std::map<double, bool> conv;

  double dnint = std::ceil((tf - ti) / interval_size);
  unsigned nint = (unsigned)(std::ceil(dnint/order))*order;
  unsigned data_size = nint+1;

  double dt = (tf - ti)/nint;
  for(unsigned i=0; i<=nint; i++)
  {
    double t = ti + i*dt;
    data[t] = f(t);

    if(i==nint) conv[t] = true;
    else conv[t] = false;
  }

  std::vector<double> times;
  std::vector<double> vals;
  bool new_insertions = true;

  while(new_insertions)
  {
    std::tie(times, vals) = fill_arrays_from_map<ArrayType>(data, data_size);
    double * ts = times.data();
    unsigned size = times.size();
    barycentric_rational<double> interp(std::move(times),std::move(vals),order);
    new_insertions = false;
    for (unsigned i=0; i<size; i++) 
    {
      if(conv.at(ts[i])) continue;
      double tnew = 0.5*(ts[i]+ts[i+1]);
      double interp_val = interp(tnew);
      double actual_val = f(tnew);
      data[tnew] = actual_val;
      new_insertions = true;
      data_size++;
      if(std::fabs(actual_val-interp_val) > tol + std::fabs(actual_val*tol))
        conv[tnew] = false;
      else
      {
        conv[ts[i]] = true;
        conv[tnew] = true;
      }
      if(data_size >= max_length)
      {
        is_converged = false;
        return fill_arrays_from_map<ArrayType>(data, data_size);
      }
    }
    times = interp.return_x();
    vals = interp.return_y();
  }
  is_converged = true;
  return std::make_pair(times,vals);
}

}
}
#endif
