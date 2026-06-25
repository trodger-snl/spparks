#ifndef GREENAM_POINT_H
#define GREENAM_POINT_H

#include <cmath>
#include <cstddef>
#include <ostream>
#ifndef GREENAM_FUNCTION
#define GREENAM_FUNCTION
#endif
#ifndef GREENAM_FORCEINLINE_FUNCTION
#define GREENAM_FORCEINLINE_FUNCTION inline
#endif
#ifndef GREENAM_INLINE_FUNCTION
#define GREENAM_INLINE_FUNCTION inline
#endif

namespace sierra { namespace greenam {

//Class for GPU-friednly fixed length array
//Template Parameters:
//T: type of data to store
//size: length of array
template <typename T, unsigned size>
class Point
{
public:
  typedef T value_type;

  //const element access
  //Parameters:
  //index [in] index of value to access
  //Returns:
  //const value at index
  GREENAM_FUNCTION value_type const& operator[](size_t index) const
  {
    #ifdef NGP_ThrowAssertMsg
    NGP_ThrowAssertMsg(index < size, "Point index out of range");
    #endif
    return m_value[index];
  }

  //non-const element access
  //Parameters:
  //index [in] index of value to access
  //Returns:
  //value at index
  GREENAM_FUNCTION value_type & operator[](size_t index)
  {
    #ifdef NGP_ThrowAssertMsg
    NGP_ThrowAssertMsg(index < size, "Point index out of range");
    #endif
    return m_value[index];
  }

  //Deep-copy assignment operator
  //Parameters:
  //pt [in] point to copy
  GREENAM_FORCEINLINE_FUNCTION void operator=(const Point<value_type,size> &pt)
  {
    for (unsigned i =0; i < size; ++i) {
      m_value[i] = pt.m_value[i];
    }
  }

  //Deep-copy assignment operator from single value
  //Parameters:
  //val [in] value to copy to all elements
  GREENAM_FORCEINLINE_FUNCTION void operator=(const value_type &val)
  {
    for (unsigned i =0; i < size; ++i) {
      m_value[i] = val;
    }
  }

  //Scalar multiplication
  //Parameters:
  //val [in] value to multiply all elements by
  //Returns:
  //point multiplied by val
  GREENAM_FORCEINLINE_FUNCTION Point<T, size> operator*(const T c) const
  {
    Point<T,size> result;
    for (unsigned i =0; i < size; ++i) {
      result[i] = c * m_value[i];
    }
    return result;
  }

  //Element-wise multiplication
  //Parameters:
  //b [in] array to multiply with
  //Returns:
  //point multiplied by b element-wise
  GREENAM_FORCEINLINE_FUNCTION Point<T, size> operator*(const Point<T,size> b) const
  {
    Point<T,size> result;
    for (unsigned i =0; i < size; ++i) {
      result[i] = b[i] * m_value[i];
    }
    return result;
  }

  //Element-wise addition
  //Parameters:
  //b [in] array to add with
  //Returns:
  //point added to b element-wise
  GREENAM_FORCEINLINE_FUNCTION Point<T, size> operator+(const Point<T,size> b) const
  {
    Point<T,size> result;
    for (unsigned i =0; i < size; ++i) {
      result[i] = b[i] + m_value[i];
    }
    return result;
  }

  //Element-wise subtraction
  //Parameters:
  //b [in] array to subtract
  //Returns:
  //point - b element-wise
  GREENAM_FORCEINLINE_FUNCTION Point<T, size> operator-(const Point<T,size> b) const
  {
    Point<T,size> result;
    for (unsigned i =0; i < size; ++i) {
      result[i] = m_value[i] - b[i];
    }
    return result;
  }

  //Element-wise equality
  //Parameters:
  //b [in] array to test
  //Returns:
  //true if all elements of b = corresponding element of point, false otherwise
  GREENAM_FUNCTION bool operator==(Point<value_type,size> const& p) const
  {
    bool result = (m_value[0] == p.m_value[0]);
    for (unsigned i=1; i < size; ++i) result = result && (m_value[i] == p.m_value[i]);
    return result;
  }

  //Element-wise in-equality
  //Parameters:
  //b [in] array to test
  //Returns:
  //true if any elements of b != corresponding element of point, false otherwise
  GREENAM_FUNCTION bool operator!=(Point<value_type,size> const& p) const
  { return !(*this == p); }

  //Element-wise between
  //Parameters:
  //a [in] left bounds
  //b [in] right bounds
  //Returns:
  //true if all elements of point are between the corresponding elements of a and b, false otherwise
  GREENAM_FUNCTION bool between(const Point<T,size> a, const Point<T,size> b) const
  {
    for (unsigned i =0; i < size; ++i) {
      if(m_value[i] < a[i] || m_value[i] > b[i]) return false;
    }
    return true;
  }

  //Element-wise between with tolerance
  //Parameters:
  //a [in] left bounds
  //b [in] right bounds
  //tol [in] tolerance
  //Returns:
  //true if all elements of point are between the corresponding elements of a and b within tol, false otherwise
  GREENAM_FUNCTION bool tolerant_between(const Point<T,size> a, const Point<T,size> b, T tol) const
  {
    for (unsigned i =0; i < size; ++i) {
      T dt = tol * (b[i] - a[i]);
      if((m_value[i] + dt) < a[i] || (m_value[i] - dt) > b[i]) return false;
    }
    return true;
  }
  
  value_type m_value[size];
};

//Magnitude of point
//Template Parameters:
//T: type of data stored in point
//size: length of array
//Parameters:
//pt [in] point to take magnitude of
//Returns:
//L2 norm of point
template <class T, unsigned size>
GREENAM_INLINE_FUNCTION T mag(const Point<T,size> & pt)
{
  T result = pt[0] * pt[0];
  for(unsigned i=1; i<size; i++)
  {
    result += pt[i] * pt[i];
  }
  return std::sqrt(result);
}

//Print point
//Template Parameters:
//T: type of data stored in point
//size: length of array
//Parameters:
//out [in] stream to print to
//pt [in] point to print
//Returns:
//point printed to stream out
template <class T, unsigned size>
std::ostream& operator<<(std::ostream & out, Point<T,size> const& p)
{
  out << "(";
  for(unsigned i=0; i<size; i++)
  {
    if(i != 0) out << ",";
    out << p[i];
  }
  out << ")";
  return out;
}

//Struct for hashing points for use with unordered maps
struct point_hash
{
  //Hash point
  //Template Parameters:
  //T: type of data stored in point
  //size: length of array
  //Parameters:
  //pt [in] point to print
  //Returns:
  //hash value of point
  template <class T, unsigned size>
  std::size_t operator() (const Point<T,size> & pt) const
  {
    std::size_t result = std::hash<T>()(pt[0]);
    for(unsigned i=1; i<size; i++)
    {
      result ^= std::hash<T>()(pt[i]) + 0x9e3779b9 + (result<<6) + (result>>2);
    }
    return result;
  }
};

typedef Point<double,1> Point1D;
typedef Point<double,2> Point2D;
typedef Point<double,3> Point3D;
typedef Point<double,4> Point4D;

}}

#endif
