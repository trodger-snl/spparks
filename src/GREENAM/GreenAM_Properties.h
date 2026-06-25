#ifndef GREENAM_PROPERTIES_H
#define GREENAM_PROPERTIES_H

#include "GreenAM_Util.h"

namespace Teuchos { class XMLObject; }

namespace sierra {
namespace greenam {
//Struct holding thermal properties of workpiece
template <typename T>
struct ThermalProperties
{
  //constructor
  //Parameters:
  //k [in] thermal conductivity
  //rho [in] density
  //cp [in] specific heat
  ThermalProperties(T k = T(), T rho = T(), T cp = T()) :
    k(k),
    rho(rho),
    cp(cp)
  {}
  //constructor
  //Parameters:
  //obj [in] XML object containing k, rho, cp info
  ThermalProperties(Teuchos::XMLObject obj);
  T k;
  T rho;
  T cp;
};

//Struct holding laser size properties
template <typename T>
struct EllipsoidProperties
{
  //constructor
  //Parameters:
  //sx [in] laser standard deviation in travel direction
  //sy [in] laser standard deviation perpendicular to travel direction in plane of travel
  //sz [in] laser standard deviation in depth direction
  EllipsoidProperties(T sx = T(), T sy = T(), T sz = T()) :
    sx(sx),
    sy(sy),
    sz(sz)
  {}
  //constructor
  //Parameters:
  //obj [in] XML object containing size info
  EllipsoidProperties(Teuchos::XMLObject obj);
  T sx;
  T sy;
  T sz;
};

}}

#endif
