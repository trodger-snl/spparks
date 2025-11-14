/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#include "temperature_source.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_hdf5_unstructured.h"
#include "domain.h"
#include "lattice.h"
#include "error.h"
#include <iostream>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

TemperatureSource::TemperatureSource(SPPARKS *spk) : Pointers(spk)
{
  ambient_temperature = 300.0; // Default room temperature
  source_initialized = false;
}

/* ----------------------------------------------------------------------
   Default implementation for site-based temperature access
   Gets site coordinates and calls xyz-based method
------------------------------------------------------------------------- */

double TemperatureSource::get_temperature_at_site(int site_index, double time)
{
  check_initialization();
  
  // This default implementation should be overridden by derived classes
  // that have access to the specific app's coordinate arrays
  error->one(FLERR,"get_temperature_at_site must be overridden by derived temperature source classes");
  return ambient_temperature;
}

/* ----------------------------------------------------------------------
   Check if temperature source has been properly initialized
------------------------------------------------------------------------- */

void TemperatureSource::check_initialization() const
{
  if (!source_initialized) {
    error->one(FLERR,"Temperature source not initialized - call setup_temperature_source first");
  }
}

/* ----------------------------------------------------------------------
   Helper method for derived classes to indicate unimplemented methods
------------------------------------------------------------------------- */

void TemperatureSource::error_not_implemented(const std::string &method_name) const
{
  std::string msg = "Method " + method_name + " not implemented in " + get_source_type() + " temperature source";
  error->one(FLERR,msg.c_str());
}

/* ----------------------------------------------------------------------
   Factory function to create temperature sources based on type
------------------------------------------------------------------------- */

std::unique_ptr<TemperatureSource> SPPARKS_NS::create_temperature_source(
    const std::string &type, 
    SPPARKS *spk)
{
  if (type == "rosenthal") {
    return std::make_unique<RosenthalTemperatureSource>(spk);
  } else if (type == "hdf5_unstructured") {
    return std::make_unique<HDF5UnstructuredTemperatureSource>(spk);
  } else if (type == "finitediff") {
    // Will be implemented when FiniteDifferenceTemperatureSource is extracted
    spk->error->all(FLERR,"Finite difference temperature source not yet implemented");
    return nullptr;
  } else {
    std::string msg = "Unknown temperature source type: " + type;
    spk->error->all(FLERR,msg.c_str());
    return nullptr;
  }
}