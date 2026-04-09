#ifndef GREENAM_UTIL_H
#define GREENAM_UTIL_H
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include "GreenAM_Point.h"
#include "GreenAM_Constants.h"

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

template <typename RealType> 
struct VectorWrapper
{
  VectorWrapper(std::string name, unsigned sz) :
    my_data(std::vector<RealType>(sz)),
    my_name(name)
  {
  }

  VectorWrapper() :
    my_data(std::vector<RealType>())
  {
  }

  VectorWrapper(const VectorWrapper & copy)
  {
    my_data = copy.my_data;
  }

  RealType & operator()(unsigned l)
  {
    return my_data[l];
  }

  const RealType & operator()(unsigned l) const
  {
    return my_data[l];
  }

  unsigned size() const
  {
    return my_data.size();
  }

  void push_back(const RealType & value)
  {
    my_data.push_back(value);
  }

  void push_back(RealType && value)
  {
    my_data.push_back(value);
  }

  RealType* data()
  {
    return my_data.data();
  };

  void commit() {}


  std::vector<RealType> my_data;
  std::string my_name;
};

//Split string into tokens based on seperators
//Parameters:
//str [in] string to tokenize
//separators [in] characters to seperate by
//Returns:
//list of tokens string was broken into
inline std::vector<std::string> tokenize(const std::string & str, const std::string & separators)
{
  std::vector<std::string> tokens;
  auto first = std::begin(str);
  while (first != std::end(str)) {
    const auto second =
        std::find_first_of(first, std::end(str), std::begin(separators), std::end(separators));
    if (first != second) {
      tokens.emplace_back(first, second);
    }
    if (second == std::end(str)) {
      break;
    }
    first = std::next(second);
  }
  return tokens;
}

//Convert a string to a point
//Template Parameters:
//T: type contained in point
//size: length of point
//Parameters:
//str [in] string to convert
//pt [out] point to populate
//Returns:
//True if conversion successful, false otherwise
template <typename T, unsigned size>
bool strToPoint(std::string str, Point<T, size> & pt)
{
  std::vector<std::string> line = tokenize(str, " ,\t\r\n");
  if(line.size() == 0) return false;
  if(line.size() != size)
    throw std::runtime_error("GreenAM::strToPoint: Poorly formatted input line");
  for(unsigned i=0; i<size; i++)
  {
    T val;
    std::istringstream ss(line[i]);
    ss >> val;
    pt[i] = val;
  }
  return true;
}

//Read a laser path in {time, laser_x, laser_y, power} format in from an input file
//Parameters:
//filename [in] name of file to read from
//Returns:
//View of path in {time, laser_x, laser_y, power} format
template <typename ArrayType>
ArrayType read_laser_path_from_file(std::string filename)
{
  std::string line;
  std::string delim = " ,\t\r\n";
  std::ifstream myfile(filename);
  ArrayType path("Laser_Path", 0);
  if (myfile.is_open())
  {
    while (std::getline(myfile,line))
    {
      Point4D pt;
      if(strToPoint(line, pt))
        path.push_back(pt);
    }
    myfile.close();
  }
  return path;
}

//Determine if a line segment intersect an axis-aligned box in arbitrary dimensions
//Template Parameters:
//Dim: dimension
//Parameters:
//p1 [in] first point of line
//p2 [in] second point of line
//bmin [in] coords of box minimum
//bmax [in] coords of box maximum
//Returns:
//true if segment intersects box, false otherwise
template <unsigned Dim>
bool segment_box_intersection(const Point<double, Dim> p1, const Point<double, Dim> p2,
  const Point<double, Dim> bmin, const Point<double, Dim> bmax)
{
  Point<double, Dim> dir = p2 - p1;
  Point<double, Dim> invDir;
  for(unsigned i=0; i<Dim; i++)
  {
    if(dir[i] == 0.)
      invDir[i] = double_max;
    else invDir[i] = 1./dir[i];
  }
  double tmin = -double_max;
  double tmax = double_max;
  for(unsigned i=0; i<Dim; i++)
  {
    double tymin, tymax;
    if (invDir[i] >= 0.) {
      tymin = (bmin[i] - p1[i]) * invDir[i];
      tymax = (bmax[i] - p1[i]) * invDir[i];
    }
    else {
      tymin = (bmax[i] - p1[i]) * invDir[i];
      tymax = (bmin[i] - p1[i]) * invDir[i];
    }
    if ((tmin > tymax) || (tymin > tmax)) return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
  }
  return (tmin <= 1.) && (tmax >= 0.);
}

//Binary search a sorted list for value
//Template Parameters:
//DataType: type contained in list
//Parameters:
//data [in] sorted list to search
//x [in] value to search for
//len [in] length of list
//Returns:
//index at which x could be inserted into list and maintain sorting. -1 if search failed
template <typename DataType1, typename DataType2>
GREENAM_FUNCTION int binary_search(DataType1 * data, DataType2 x, unsigned len)
{
  int L = 0;
  int R = len-1;
  int result = -1;
  while(L <= R)
  {
    int m = (L+R)/2;
    if(data[m] <= x)
    {
      if(static_cast<unsigned>(m+1) == len || data[m+1] >= x)
      {
        result = m;
        break;
      }
      L = m + 1;
    }
    else R = m - 1;
  }
  return result;
}

}
}
#endif
