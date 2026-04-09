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

#ifndef SPK_TEMPERATURE_SOURCE_H
#define SPK_TEMPERATURE_SOURCE_H

#include "pointers.h"
#include <vector>
#include <string>
#include <memory>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Abstract base class for temperature data sources
   
   This class provides a unified interface for different temperature
   data sources in additive manufacturing simulations:
   - External temperature fields (HDF5 files)
   - Internal thermal solvers (finite difference)
   - Analytical solutions (Rosenthal moving heat source)
   - Coupled external thermal solvers
   
   Derived classes must implement the pure virtual methods to provide
   temperature data to the microstructure evolution application.
------------------------------------------------------------------------- */

class TemperatureSource : protected Pointers {
 public:
  TemperatureSource(class SPPARKS *);
  virtual ~TemperatureSource() = default;

  // Pure virtual interface methods that derived classes must implement
  virtual void setup_temperature_source(const std::vector<std::string> &args) = 0;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) = 0;
  virtual void update_temperatures(double dt, double simulation_time) = 0;
  virtual bool needs_data_refresh(double simulation_time) = 0;
  virtual void cleanup() = 0;

  // Convenience method for site-based access (can be overridden for optimization)
  virtual double get_temperature_at_site(int site_index, double time);
  
  // Optional methods with default implementations
  virtual void validate_setup() {}
  virtual std::string get_source_type() const = 0;
  virtual void print_source_info() const {}
  
  // Time query interface for efficient fast forward
  virtual bool supports_time_queries() const { return false; }
  virtual double get_next_time_with_temperature(double current_time, double threshold_temp) { return current_time; }

  // Accessor for the source's notion of ambient/preheat temperature.
  // Set by setup_temperature_source() in derived classes (e.g. Rosenthal T0,
  // HDF5 default_temp). Used by driving apps to seed analytical evaluations
  // and to know what to set sites to outside the active region.
  double get_ambient_temperature() const { return ambient_temperature; }

 protected:
  // Common data members accessible to derived classes
  double ambient_temperature;
  bool source_initialized;
  
  // Helper methods for common operations
  void check_initialization() const;
  void error_not_implemented(const std::string &method_name) const;
};

/* ----------------------------------------------------------------------
   Factory function for creating temperature sources
   Creates appropriate temperature source based on type string
------------------------------------------------------------------------- */
std::unique_ptr<TemperatureSource> create_temperature_source(
    const std::string &type, 
    class SPPARKS *spk
);

}

#endif