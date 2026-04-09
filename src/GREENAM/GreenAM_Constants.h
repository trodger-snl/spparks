#ifndef GREENAM_CONSTANTS_H
#define GREENAM_CONSTANTS_H
#include <limits>

namespace sierra {
namespace greenam {
const double PI = 3.14159265358979323846;
const double sqrtPI = 1.7724538509055160273;
const double invPi3_2 = 1./PI/sqrtPI;
const double epsilon = std::numeric_limits<double>::epsilon();
const double double_max = std::numeric_limits<double>::max();
const unsigned uint_max = std::numeric_limits<unsigned>::max();
const double double_min = std::numeric_limits<double>::min();
}
}
#endif
