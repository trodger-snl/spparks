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

#ifndef SPK_TEMPERATURE_SOURCE_ROSENTHAL_H
#define SPK_TEMPERATURE_SOURCE_ROSENTHAL_H

#include "temperature_source.h"
#include <vector>
#include <array>
#include <string>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Scan path types for Rosenthal moving heat source
------------------------------------------------------------------------- */
enum ScanPathType {
  LINEAR,      // Single linear pass
  SERPENTINE,  // Back-and-forth raster pattern
  SPIRAL,      // Spiral pattern
  CUSTOM       // User-defined path from file
};

/* ----------------------------------------------------------------------
   Rosenthal analytical temperature source for moving point heat source
   
   Implements the classical Rosenthal solution for a moving point heat source:
   T(x,y,z,t) = T_ambient + (Q/(2πkr)) * exp(-v(x-x_source(t)+r)/(2α))
   
   where:
   - r = distance from current heat source position
   - v = scanning velocity
   - α = thermal diffusivity
   - Q = laser power
   - k = thermal conductivity
   
   Supports multiple scan path patterns and moving heat source trajectories.
------------------------------------------------------------------------- */

class RosenthalTemperatureSource : public TemperatureSource {
 public:
  RosenthalTemperatureSource(class SPPARKS *);
  virtual ~RosenthalTemperatureSource();

  // Required virtual methods from base class
  virtual void setup_temperature_source(const std::vector<std::string> &args) override;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  virtual void update_temperatures(double dt, double simulation_time) override;
  virtual bool needs_data_refresh(double simulation_time) override;
  virtual void cleanup() override;
  virtual std::string get_source_type() const override { return "rosenthal"; }
  virtual void print_source_info() const override;
  virtual double get_temperature_at_site(int site_index, double time) override;

  // Rosenthal-specific methods
  void setup_scan_path(ScanPathType type, const std::vector<double> &params);
  void load_custom_path(const std::string &filename);
  std::array<double,3> get_laser_position(double time);
  double get_laser_velocity(double time);

 private:
  // Material and laser parameters
  double laser_power;           // Laser power (W)
  double scan_velocity;         // Nominal scanning velocity (m/s)
  double thermal_conductivity;  // Material thermal conductivity (W/m·K)
  double thermal_diffusivity;   // Material thermal diffusivity (m²/s)
  
  // Scan path parameters
  ScanPathType path_type;
  std::vector<std::array<double,4>> scan_path_points; // [x, y, z, time]
  std::vector<double> scan_velocities; // Velocity at each path segment
  
  // Path generation parameters
  double path_start_time;
  double path_end_time;
  
  // Linear path parameters
  std::array<double,3> linear_start, linear_end;
  
  // Serpentine/raster parameters
  double raster_x_min, raster_x_max;
  double raster_y_min, raster_y_max;
  double raster_z_height;
  double raster_spacing;
  int raster_num_passes;
  
  // Spiral parameters
  std::array<double,3> spiral_center;
  double spiral_radius_max;
  double spiral_pitch;
  double spiral_turns;
  
  // Cache for performance
  mutable double cached_time;
  mutable std::array<double,3> cached_laser_position;
  mutable double cached_laser_velocity;
  
  // Private helper methods
  void parse_setup_arguments(const std::vector<std::string> &args);
  void generate_linear_path();
  void generate_serpentine_path();
  void generate_spiral_path();
  void generate_custom_path();
  
  double calculate_distance(double x, double y, double z, const std::array<double,3> &laser_pos);
  double rosenthal_temperature(double x, double y, double z, double time);
  
  int find_path_segment(double time) const;
  std::array<double,3> interpolate_position(double time, int segment) const;
  double interpolate_velocity(double time, int segment) const;
  
  void validate_material_properties();
  void validate_scan_path();
};

}

#endif