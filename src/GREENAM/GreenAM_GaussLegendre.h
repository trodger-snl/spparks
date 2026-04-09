#ifndef GREENAM_GAUSSLEGENDRE_H
#define GREENAM_GAUSSLEGENDRE_H

#include "GreenAM_Constants.h"
#include <cmath>

namespace sierra {
namespace greenam {

template <int N>
class GaussLegendreQuadrature {
public:
  enum {eDEGREE = N};

  /*! Compute the integral of a functor
  *
  *   @param a    lower limit of integration
  *   @param b    upper limit of integration
  *   @param f    the function to integrate
  *   @param err  callback in case of problems
  */
  template <typename T, typename Function>
  auto integrate(Function & f, T a, T b)
  {
    typedef decltype(f(T(0))) K;
    T p = (b - a) / 2;
    T q = (b + a) / 2;
    const LegendrePolynomial& legpoly = s_LegendrePolynomial;

    K sum = 0.;
    for (int i = 1; i <= eDEGREE; ++i) 
    {
      sum += f(p * legpoly.root(i) + q) * legpoly.weight(i);
    }

    return K(p * sum);
  }
private:
  /*! Implementation of the Legendre polynomials that form
  *   the basis of this quadrature
  */
  class LegendrePolynomial {
  public:
    LegendrePolynomial () 
    {
      // Solve roots and weights
      for (int i = 0; i <= eDEGREE; ++i) 
      {
        double dr = 1;

        // Find zero
        Evaluation eval(cos(PI * (i - 0.25) / (eDEGREE + 0.5)));
        do 
        {
          dr = eval.v() / eval.d();
          eval.evaluate(eval.x() - dr);
        } 
        while (std::fabs (dr) > 2e-16);

        this->_r[i] = eval.x();
        this->_w[i] = 2 / ((1 - eval.x() * eval.x()) * eval.d() * eval.d());
      }
    }

    double root(int i) const { return this->_r[i]; }
    double weight(int i) const { return this->_w[i]; }
  private:
    double _r[eDEGREE + 1];
    double _w[eDEGREE + 1];

      /*! Evaluate the value *and* derivative of the
      *   Legendre polynomial
      */
    class Evaluation {
    public:
      explicit Evaluation (double x) : _x(x), _v(1), _d(0) 
      {
        this->evaluate(x);
      }

      void evaluate(double x) {
        this->_x = x;

        double vsub1 = x;
        double vsub2 = 1;
        double f     = 1 / (x * x - 1);

        for (int i = 2; i <= eDEGREE; ++i) 
        {
          this->_v = ((2 * i - 1) * x * vsub1 - (i - 1) * vsub2) / i;
          this->_d = i * f * (x * this->_v - vsub1);

          vsub2 = vsub1;
          vsub1 = this->_v;
        }
      }

      double v() const { return this->_v; }
      double d() const { return this->_d; }
      double x() const { return this->_x; }

    private:
      double _x;
      double _v;
      double _d;
    };
  };
  /*! Pre-compute the weights and abscissae of the Legendre polynomials
  */
  static LegendrePolynomial s_LegendrePolynomial;
};

template <int N>
typename GaussLegendreQuadrature<N>::LegendrePolynomial GaussLegendreQuadrature<N>::s_LegendrePolynomial;

}
}
#endif
