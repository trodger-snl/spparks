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
#include "temperature_source_moser.h"
#include "temperature_source_conduction.h"
#include "temperature_source_hdf5_unstructured.h"
#include "temperature_source_hdf5_csr.h"
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
   Check if temperature source has been properly initialized
------------------------------------------------------------------------- */

void TemperatureSource::check_initialization() const
{
  if (!source_initialized) {
    error->one(FLERR,"Temperature source not initialized - call setup_temperature_source first");
  }
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
  } else if (type == "conduction") {
    return std::make_unique<ConductionTemperatureSource>(spk);
  } else if (type == "moser") {
    return std::make_unique<MoserTemperatureSource>(spk);
  } else if (type == "hdf5_unstructured") {
    return std::make_unique<HDF5UnstructuredTemperatureSource>(spk);
  } else if (type == "hdf5_csr") {
    return std::make_unique<HDF5CSRTemperatureSource>(spk);
  } else {
    std::string msg = "Unknown temperature source type: " + type;
    spk->error->all(FLERR,msg.c_str());
    return nullptr;
  }
}