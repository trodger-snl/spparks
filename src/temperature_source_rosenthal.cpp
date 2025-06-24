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

#include "temperature_source_rosenthal.h"
#include "error.h"
#include "domain.h"
#include "math_const.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace SPPARKS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

RosenthalTemperatureSource::RosenthalTemperatureSource(SPPARKS *spk) : TemperatureSource(spk)
{
  // Initialize default values
  laser_power = 1000.0;          // 1 kW default
  scan_velocity = 0.01;          // 1 cm/s default
  thermal_conductivity = 30.0;   // W/m·K (typical for steel)
  thermal_diffusivity = 5.3e-6;  // m²/s (typical for steel)
  ambient_temperature = 300.0;   // Room temperature
  
  path_type = LINEAR;
  path_start_time = 0.0;
  path_end_time = 1.0;
  
  // Initialize cache
  cached_time = -1.0;
  cached_laser_position = {0.0, 0.0, 0.0};
  cached_laser_velocity = 0.0;
  
  // Initialize path parameters with defaults
  linear_start = {0.0, 0.0, 0.0};
  linear_end = {0.01, 0.0, 0.0}; // 1 cm linear path
  
  // Raster defaults
  raster_x_min = 0.0; raster_x_max = 0.01;
  raster_y_min = 0.0; raster_y_max = 0.01;
  raster_z_height = 0.0;
  raster_spacing = 0.001; // 1 mm spacing
  raster_num_passes = 0;
  
  // Spiral defaults
  spiral_center = {0.0, 0.0, 0.0};
  spiral_radius_max = 0.005; // 5 mm
  spiral_pitch = 0.001;      // 1 mm
  spiral_turns = 5.0;
}

/* ---------------------------------------------------------------------- */

RosenthalTemperatureSource::~RosenthalTemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup temperature source from command line arguments
   Expected format: laser_power scan_velocity thermal_conductivity thermal_diffusivity ambient_temp
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  if (args.size() < 5) {
    error->all(FLERR,"Insufficient arguments for Rosenthal temperature source setup");
  }
  
  try {
    laser_power = std::stod(args[0]);
    scan_velocity = std::stod(args[1]);
    thermal_conductivity = std::stod(args[2]);
    thermal_diffusivity = std::stod(args[3]);
    ambient_temperature = std::stod(args[4]);
  } catch (const std::exception &e) {
    error->all(FLERR,"Invalid numeric arguments in Rosenthal temperature source setup");
  }
  
  validate_material_properties();
  
  // Generate default linear path if no path has been set up
  if (scan_path_points.empty()) {
    generate_linear_path();
  }
  
  validate_scan_path();
  
  source_initialized = true;
  
  if (domain->me == 0) {
    print_source_info();
  }
}

/* ----------------------------------------------------------------------
   Calculate temperature at given position and time using Rosenthal solution
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::get_temperature_at_xyz_and_time(double x, double y, double z, double time)
{
  check_initialization();
  
  // Check if time is within the scan period
  if (time < path_start_time || time > path_end_time) {
    return ambient_temperature;
  }
  
  return rosenthal_temperature(x, y, z, time);
}

/* ----------------------------------------------------------------------
   Rosenthal analytical solution for moving point heat source
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::rosenthal_temperature(double x, double y, double z, double time)
{
  // Get current laser position and velocity
  std::array<double,3> laser_pos = get_laser_position(time);
  double velocity = get_laser_velocity(time);
  
  // Calculate distance from laser position
  double r = calculate_distance(x, y, z, laser_pos);
  
  // Avoid singularity at laser position
  if (r < 1e-12) {
    r = 1e-12;
  }
  
  // Calculate relative position in scanning direction
  // Assume scanning primarily in x-direction for simplicity
  double dx_rel = x - laser_pos[0];
  
  // Rosenthal solution: T = T_amb + (Q/(2πkr)) * exp(-v(x-x_source+r)/(2α))
  double peclet_term = velocity * (dx_rel + r) / (2.0 * thermal_diffusivity);
  double temperature_rise = (laser_power / (2.0 * MY_PI * thermal_conductivity * r)) * 
                           std::exp(-peclet_term);
  
  return ambient_temperature + temperature_rise;
}

/* ----------------------------------------------------------------------
   Calculate distance between point and laser position
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::calculate_distance(double x, double y, double z, 
                                                     const std::array<double,3> &laser_pos)
{
  double dx = x - laser_pos[0];
  double dy = y - laser_pos[1]; 
  double dz = z - laser_pos[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

/* ----------------------------------------------------------------------
   Get laser position at given time with caching for performance
------------------------------------------------------------------------- */

std::array<double,3> RosenthalTemperatureSource::get_laser_position(double time)
{
  // Use cache if time hasn't changed
  if (std::abs(time - cached_time) < 1e-12) {
    return cached_laser_position;
  }
  
  // Find appropriate path segment
  int segment = find_path_segment(time);
  
  if (segment < 0 || segment >= static_cast<int>(scan_path_points.size()) - 1) {
    // Outside path range, return start or end position
    if (time <= path_start_time) {
      cached_laser_position = {scan_path_points[0][0], scan_path_points[0][1], scan_path_points[0][2]};
    } else {
      int last_idx = scan_path_points.size() - 1;
      cached_laser_position = {scan_path_points[last_idx][0], scan_path_points[last_idx][1], scan_path_points[last_idx][2]};
    }
  } else {
    // Interpolate position within segment
    cached_laser_position = interpolate_position(time, segment);
  }
  
  cached_time = time;
  return cached_laser_position;
}

/* ----------------------------------------------------------------------
   Get laser velocity at given time
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::get_laser_velocity(double time)
{
  // Use cache if time hasn't changed
  if (std::abs(time - cached_time) < 1e-12) {
    return cached_laser_velocity;
  }
  
  int segment = find_path_segment(time);
  
  if (segment < 0 || segment >= static_cast<int>(scan_velocities.size())) {
    cached_laser_velocity = scan_velocity; // Default velocity
  } else {
    cached_laser_velocity = scan_velocities[segment];
  }
  
  return cached_laser_velocity;
}

/* ----------------------------------------------------------------------
   Find path segment index for given time
------------------------------------------------------------------------- */

int RosenthalTemperatureSource::find_path_segment(double time) const
{
  for (int i = 0; i < static_cast<int>(scan_path_points.size()) - 1; i++) {
    if (time >= scan_path_points[i][3] && time <= scan_path_points[i+1][3]) {
      return i;
    }
  }
  return -1; // Time outside path range
}

/* ----------------------------------------------------------------------
   Interpolate position within a path segment
------------------------------------------------------------------------- */

std::array<double,3> RosenthalTemperatureSource::interpolate_position(double time, int segment) const
{
  if (segment < 0 || segment >= static_cast<int>(scan_path_points.size()) - 1) {
    error->one(FLERR,"Invalid segment in position interpolation");
  }
  
  double t0 = scan_path_points[segment][3];
  double t1 = scan_path_points[segment + 1][3];
  double dt = t1 - t0;
  
  if (dt < 1e-12) {
    // Zero time interval, return start position
    return {scan_path_points[segment][0], scan_path_points[segment][1], scan_path_points[segment][2]};
  }
  
  double alpha = (time - t0) / dt;
  alpha = std::max(0.0, std::min(1.0, alpha)); // Clamp to [0,1]
  
  std::array<double,3> pos;
  pos[0] = scan_path_points[segment][0] + alpha * (scan_path_points[segment + 1][0] - scan_path_points[segment][0]);
  pos[1] = scan_path_points[segment][1] + alpha * (scan_path_points[segment + 1][1] - scan_path_points[segment][1]);
  pos[2] = scan_path_points[segment][2] + alpha * (scan_path_points[segment + 1][2] - scan_path_points[segment][2]);
  
  return pos;
}

/* ----------------------------------------------------------------------
   Generate linear scan path between two points
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::generate_linear_path()
{
  scan_path_points.clear();
  scan_velocities.clear();
  
  // Calculate path length and time
  double dx = linear_end[0] - linear_start[0];
  double dy = linear_end[1] - linear_start[1];
  double dz = linear_end[2] - linear_start[2];
  double path_length = std::sqrt(dx*dx + dy*dy + dz*dz);
  
  double travel_time = path_length / scan_velocity;
  
  // Add start and end points
  scan_path_points.push_back({linear_start[0], linear_start[1], linear_start[2], path_start_time});
  scan_path_points.push_back({linear_end[0], linear_end[1], linear_end[2], path_start_time + travel_time});
  
  // Single velocity for entire path
  scan_velocities.push_back(scan_velocity);
  
  path_end_time = path_start_time + travel_time;
}

/* ----------------------------------------------------------------------
   Update method - no special updates needed for analytical solution
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::update_temperatures(double dt, double simulation_time)
{
  // Analytical solution doesn't require time stepping updates
  // This method is available for future extensions
}

/* ----------------------------------------------------------------------
   Check if data refresh is needed - always false for analytical solution
------------------------------------------------------------------------- */

bool RosenthalTemperatureSource::needs_data_refresh(double simulation_time)
{
  return false; // Analytical solution doesn't need data refresh
}

/* ----------------------------------------------------------------------
   Setup scan path with specified type and parameters
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::setup_scan_path(ScanPathType type, const std::vector<double> &params)
{
  path_type = type;
  
  switch (path_type) {
    case LINEAR:
      if (params.size() >= 6) {
        linear_start = {params[0], params[1], params[2]};
        linear_end = {params[3], params[4], params[5]};
      }
      generate_linear_path();
      break;
      
    case SERPENTINE:
      if (params.size() >= 6) {
        raster_x_min = params[0]; raster_x_max = params[1];
        raster_y_min = params[2]; raster_y_max = params[3];
        raster_z_height = params[4];
        raster_spacing = params[5];
      }
      generate_serpentine_path();
      break;
      
    case SPIRAL:
      if (params.size() >= 6) {
        spiral_center = {params[0], params[1], params[2]};
        spiral_radius_max = params[3];
        spiral_pitch = params[4];
        spiral_turns = params[5];
      }
      generate_spiral_path();
      break;
      
    case CUSTOM:
      // Custom path will be loaded separately
      break;
  }
}

/* ----------------------------------------------------------------------
   Generate serpentine/raster scan path
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::generate_serpentine_path()
{
  scan_path_points.clear();
  scan_velocities.clear();
  
  double y_range = raster_y_max - raster_y_min;
  raster_num_passes = static_cast<int>(y_range / raster_spacing) + 1;
  
  double current_time = path_start_time;
  
  for (int pass = 0; pass < raster_num_passes; pass++) {
    double y_pos = raster_y_min + pass * raster_spacing;
    
    if (pass % 2 == 0) {
      // Even passes: left to right
      scan_path_points.push_back({raster_x_min, y_pos, raster_z_height, current_time});
      double travel_time = (raster_x_max - raster_x_min) / scan_velocity;
      current_time += travel_time;
      scan_path_points.push_back({raster_x_max, y_pos, raster_z_height, current_time});
    } else {
      // Odd passes: right to left
      scan_path_points.push_back({raster_x_max, y_pos, raster_z_height, current_time});
      double travel_time = (raster_x_max - raster_x_min) / scan_velocity;
      current_time += travel_time;
      scan_path_points.push_back({raster_x_min, y_pos, raster_z_height, current_time});
    }
    
    scan_velocities.push_back(scan_velocity);
  }
  
  path_end_time = current_time;
}

/* ----------------------------------------------------------------------
   Generate spiral scan path
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::generate_spiral_path()
{
  scan_path_points.clear();
  scan_velocities.clear();
  
  int num_points = static_cast<int>(spiral_turns * 50); // 50 points per turn
  double angular_increment = 2.0 * MY_PI * spiral_turns / num_points;
  
  double current_time = path_start_time;
  
  for (int i = 0; i <= num_points; i++) {
    double angle = i * angular_increment;
    double radius = spiral_radius_max * angle / (2.0 * MY_PI * spiral_turns);
    
    double x = spiral_center[0] + radius * std::cos(angle);
    double y = spiral_center[1] + radius * std::sin(angle);
    double z = spiral_center[2];
    
    scan_path_points.push_back({x, y, z, current_time});
    
    if (i > 0) {
      // Calculate distance from previous point
      double dx = x - scan_path_points[i-1][0];
      double dy = y - scan_path_points[i-1][1];
      double dist = std::sqrt(dx*dx + dy*dy);
      current_time += dist / scan_velocity;
      scan_path_points[i][3] = current_time;
    }
    
    if (i < num_points) {
      scan_velocities.push_back(scan_velocity);
    }
  }
  
  path_end_time = current_time;
}

/* ----------------------------------------------------------------------
   Validate material properties
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::validate_material_properties()
{
  if (laser_power <= 0.0) {
    error->all(FLERR,"Laser power must be positive in Rosenthal temperature source");
  }
  if (scan_velocity <= 0.0) {
    error->all(FLERR,"Scan velocity must be positive in Rosenthal temperature source");
  }
  if (thermal_conductivity <= 0.0) {
    error->all(FLERR,"Thermal conductivity must be positive in Rosenthal temperature source");
  }
  if (thermal_diffusivity <= 0.0) {
    error->all(FLERR,"Thermal diffusivity must be positive in Rosenthal temperature source");
  }
}

/* ----------------------------------------------------------------------
   Validate scan path
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::validate_scan_path()
{
  if (scan_path_points.empty()) {
    error->all(FLERR,"No scan path points defined in Rosenthal temperature source");
  }
  if (scan_path_points.size() != scan_velocities.size() + 1) {
    error->all(FLERR,"Inconsistent scan path points and velocities in Rosenthal temperature source");
  }
}

/* ----------------------------------------------------------------------
   Print source information
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::print_source_info() const
{
  std::cout << "Rosenthal Temperature Source Configuration:" << std::endl;
  std::cout << "  Laser Power: " << laser_power << " W" << std::endl;
  std::cout << "  Scan Velocity: " << scan_velocity << " m/s" << std::endl;
  std::cout << "  Thermal Conductivity: " << thermal_conductivity << " W/m·K" << std::endl;
  std::cout << "  Thermal Diffusivity: " << thermal_diffusivity << " m²/s" << std::endl;
  std::cout << "  Ambient Temperature: " << ambient_temperature << " K" << std::endl;
  
  std::string path_type_str;
  switch (path_type) {
    case LINEAR: path_type_str = "Linear"; break;
    case SERPENTINE: path_type_str = "Serpentine"; break;
    case SPIRAL: path_type_str = "Spiral"; break;
    case CUSTOM: path_type_str = "Custom"; break;
  }
  std::cout << "  Scan Path Type: " << path_type_str << std::endl;
  std::cout << "  Path Duration: " << (path_end_time - path_start_time) << " s" << std::endl;
  std::cout << "  Number of Path Points: " << scan_path_points.size() << std::endl;
}

/* ----------------------------------------------------------------------
   Cleanup resources
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::cleanup()
{
  scan_path_points.clear();
  scan_velocities.clear();
  source_initialized = false;
}

/* ----------------------------------------------------------------------
   Site-based temperature access for lattice applications
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::get_temperature_at_site(int site_index, double time)
{
  check_initialization();
  
  // For Rosenthal source, we need access to site coordinates
  // This will need to be handled by the calling app since we don't have direct access
  // to the coordinate arrays. For now, return ambient temperature.
  // This method should be called through get_temperature_at_xyz_and_time instead.
  return ambient_temperature;
}