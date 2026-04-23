/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
   
   Contributing author: Theron Rodgers

   Additive manufacturing application with external temperature field and crystallographic texture modeling.
   Uses quaternion-based representation for crystal orientations and implements experimentally-derived
   parameters for nucleation and grain growth during solidification processes.
   
   Key features:
   - Modular temperature source system with HDF5 unstructured data support
   - Monte Carlo nucleation model with temperature-dependent kinetics
   - Quaternion-based crystallographic orientation tracking
   - Arrhenius mobility model for grain boundary kinetics
   - Epitaxial growth and competitive nucleation mechanisms
   - Support for both 2D and 3D additive manufacturing simulations
   
 ------------------------------------------------------------------------- */
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <limits>
#include "math.h"
#include "math_const.h"
#include "app_additive_ext_temp_texture.h"
#include "comm_lattice.h"
#include "lattice.h"
#include "timer.h"
#include "random_park.h"
#include "error.h"
#include "domain.h"
#include "potts_quaternion/cubic_symmetries.h"
#include "potts_quaternion/disorientation.h"
#include "potts_quaternion/hcp_symmetries.h"
#include "potts_quaternion/quaternion.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_moser.h"
#include "temperature_source_hdf5_unstructured.h"


using namespace SPPARKS_NS;
using namespace MathConst;

// File-scope timing variables for temperature update breakdown
// (accessible from both app_update and update_temperature_from_source)
static double g_t_temp_prepare = 0.0;   // Time in prepare_for_timestep()
static double g_t_temp_site_loop = 0.0; // Time in site temperature loop
static double g_t_cache_build = 0.0;    // One-time cache build time (per layer)
static bool g_cache_was_built = false;  // Flag to detect cache build in current call

static std::vector<double> sample_random_unit_vector(RandomPark *random)
{
  const double z = 2.0 * random->uniform() - 1.0;
  const double phi = 2.0 * MY_PI * random->uniform();
  const double radial = sqrt(std::max(0.0, 1.0 - z * z));
  return std::vector<double>{radial * cos(phi), radial * sin(phi), z};
}

/* ---------------------------------------------------------------------- */

static char** construct_parent_args(char **arg) {
  static char* parent_args[3];
  static char app_name[] = "additive_temperature_texture";
  static char crystal_type[] = "cubic";
  
  parent_args[0] = app_name;
  parent_args[1] = arg[1]; // nspins
  parent_args[2] = crystal_type; // crystal structure
  
  return parent_args;
}

/* ---------------------------------------------------------------------- */

AppAdditiveExtTempTexture::AppAdditiveExtTempTexture(SPPARKS *spk, int narg, char **arg) :
  AppPottsQuaternion(spk,3,construct_parent_args(arg))
{

    if (narg == 6)
    error->all(FLERR,"app_style command format changed: remove temperature file path argument and use 'setup_temperature_source' command instead");

    if (narg != 5  )
    error->all(FLERR,"Illegal app_style command");

    nspins = atoi(arg[1]);
    dx = atof(arg[2]); //The source lattice spacing ( in m)
    dt = atof(arg[3]); //The source timestep (in seconds)
    nrefine = atoi(arg[4]); //How many refinement MC steps to perform after a site solidifies
    
    // Integer arrays (iarray): 0=spin, 1=active_flag
    // Double arrays (darray):  0=q0, 1=qx, 2=qy, 3=qz,
    //   4=mobility_out, 5=T, 6=solid_d, 7=G, 8=V
    ndouble = 9;
    allow_app_update = 1;
    app_update_only = 1; //Skip solid-state growth for now.
    ninteger = 2;

    // Ghost-communicated arrays (index-based selective comm):
    //   iarray: {0,1} = spin, active_flag
    //   darray: {0,1,2,3,5} = q0,qx,qy,qz,T (skips 4=mobility_out)
    // Remaining arrays are local-only (used for I/O and physics).
    ghost_iindices = NULL;
    ghost_dindices = NULL;

    nghost_iarray = 2;
    int *gi = new int[nghost_iarray];
    gi[0] = 0;  // spin
    gi[1] = 1;  // active_flag
    ghost_iindices = gi;

    nghost_darray = 5;
    int *gd = new int[nghost_darray];
    gd[0] = 0;  // q0
    gd[1] = 1;  // qx
    gd[2] = 2;  // qy
    gd[3] = 3;  // qz
    gd[4] = 5;  // T (needed by normal_finder on ghost neighbors)
    ghost_dindices = gd;
    sites = unique = NULL;
    unique_dot = NULL;
    neigh_dist = NULL;
    nucleation_flags = NULL;
    nucleation_temps = NULL;
    nucleation_sizes = NULL;

    //Set default values    
    tl = 1723;
    ts = 1673;
    t_cool_max = tl - ts;
    no = 1e15;
    tc = 5;
    tsig = 3;
    size_norm = pow(dx,3) * 2;
    size_sig = pow(dx,3);
    solid_front_length = 4;
    // Set default values for coefficient arrays
    solid_front_coeffs = new double[4];    
    solid_front_coeffs[0] = 1.091e-5;
    solid_front_coeffs[1] = -2.034e-4;
    solid_front_coeffs[2] = 2.74e-3;
    solid_front_coeffs[3] = 1.151e-4;
    //Default texture parameters
    c1 = 0.5;
    c2 = 0.5;
    c3 = 2.5;
    time_step = dt;
    t_room = 300;
    max_misorient = 0;
    mis_thresh = 10.0 * MY_PI / 180.0; // Default 10 degrees converted to radians
    misorient_alpha = 5.0; // Default exponential steepness parameter
    misorientation_target_mode = MISORI_TARGET_GRADIENT;
    nucleation_mode = NUCLEATION_THRESHOLD;

    // Void generation defaults
    enable_voids = 0;              // Disabled by default
    void_density = 0.0;            // No voids by default
    void_pore_fraction = -1.0;     // -1.0 means not specified
    void_radius_mean = 75.0;       // 75 micrometers
    void_radius_std = 25.0;        // 25 micrometers
    void_radius_min = 0.0;         // 0 micrometers
    void_radius_max = 150.0;       // 150 micrometers

    // Initialize bounds checking variables

    // Initialize modular temperature source system
    temperature_source = nullptr;
    use_temperature_source = true;  // Always use modular temperature system
    fast_forward_search_window = 0.1;  // Default 100ms search window

    // Rosenthal scan-path state (only used when temperature_source is Rosenthal)
    scan_layer = RASTER::Layer();
    scan_layer_z = 0.0;
    scan_layer_time = 0.0;
    scan_layer_active = false;
    laser_path_set = false;

    // Temperature optimization flags (all enabled by default)
    opt_use_spatial_grid = true;
    opt_use_element_cache = true;
    opt_use_nodal_precompute = true;

    // Initialize powder activation tracking
    last_powder_activation_time = -1.0;  // Force activation at init

    // Solidification-band smoothing defaults (off until `temperature_smooth`
    // is issued in the input script).
    temperature_smooth_enabled = false;
    smooth_tmin  = 0.0;
    smooth_tmax  = 0.0;
    smooth_guard = 20.0;
    smooth_sigma_xy = 1.2;
    smooth_sigma_z  = 1.2;
    smooth_alpha = 0.4;
    smooth_passes = 1;
    smooth_diag_interval = 0;

    // Single-voxel grain cleanup is off until `single_voxel_cleanup on`
    // is issued in the input script.
    single_voxel_cleanup_enabled = false;
    n_single_voxel_flips = 0;
    single_voxel_aggressive_interval = 0;
    smooth_greedy_multiproposal_enabled = false;

    //add the double array
    recreate_arrays();
}

/* ---------------------------------------------------------------------- */

AppAdditiveExtTempTexture::~AppAdditiveExtTempTexture()
{
  // Only delete arrays that are specific to this class
  // Parent class destructors will handle sites, unique, unique_neigh
  if (unique_dot) delete[] unique_dot;
  if (neigh_dist) delete[] neigh_dist;
  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  if (solid_front_coeffs) delete[] solid_front_coeffs;
  if (ghost_iindices) delete [] ghost_iindices;
  if (ghost_dindices) delete [] ghost_dindices;
}

/* ----------------------------------------------------------------------
   input script commands unique to this app
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::input_app(char *command, int narg, char **arg)
{
  if (strcmp(command,"liquidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal liquidus command");
    tl = atof(arg[0]);
    if (tl <= 0) 
      error->all(FLERR,"Illegal liquidus temperature");
  }
  else if (strcmp(command,"solidus") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal solidus command");
    ts = atof(arg[0]);
    if (ts <= 0) 
      error->all(FLERR,"Illegal solidus temperature");
  }
  else if (strcmp(command,"nucleation_density") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nucleation density command");
    no = atof(arg[0]);
    if (no < 0) 
      error->all(FLERR,"Illegal nucleation density");
  }
  else if (strcmp(command,"critical_undercooling") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal critical_undercooling command");
    tc = atof(arg[0]);
    if (tc < 0) 
      error->all(FLERR,"Illegal critical undercooling");
  }
  else if (strcmp(command,"undercooling_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal undercooling_deviation command");
    tsig = atof(arg[0]);
    if (tsig < 0) 
      error->all(FLERR,"Illegal undercooling standard deviation");
  }
  else if (strcmp(command,"mean_nuclei_volume") == 0) {
    if (narg !=1) error->all(FLERR,"Illegal mean_nuclei_volume command");
    size_norm = atof(arg[0]);
    if (size_norm < 0) 
      error->all(FLERR,"Illegal mean nuclei volume");
  }
  else if (strcmp(command,"nuclei_volume_deviation") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nuclei_volume_deviation command");
    size_sig = atof(arg[0]);
    if (size_sig < 0) 
      error->all(FLERR,"Illegal nuclei volume standard deviation");
  }
  else if (strcmp(command,"solid_front_vel") == 0) {
    delete [] solid_front_coeffs;
    if (narg < 1) error->all(FLERR,"Illegal solid_front_vel command");
    int nVel;
    nVel = atoi(arg[0]);
    if (nVel <= 0) {
      error->all(FLERR,"Illegal solidification front velocity specification");
    }
    solid_front_coeffs = new double [nVel];
    int j = 0;
    for(int i = 1; i < nVel + 1; i++) {
        solid_front_coeffs[j] = atof(arg[i]);
        j++;
    }
  }
  else if (strcmp(command,"texture_parameters") == 0) {
     if (narg == 3)
       error->all(FLERR,
         "texture_parameters now takes 2 arguments (c2 c3), not 3. "
         "The baseline coefficient c1 has been fixed at 1.0 because "
         "only mobility ratios affect selection. To migrate an old "
         "input file: texture_parameters c2/c1 c3 (divide old c2 by "
         "old c1, drop old c1)");
     if (narg != 2) error->all(FLERR,"Illegal texture_parameters command");

     // c1 is fixed at 1.0 by normalization convention. The roulette-wheel
     // selection in site_event_solidification is invariant under uniform
     // scaling of mobility (only ratios matter), so c1 is mathematically
     // non-identifiable. Hardcoding it here removes the degenerate
     // parameter and makes input files unambiguous.
     c1 = 1.0;
     c2 = atof(arg[0]);
     c3 = atof(arg[1]);

     // -------- Validation --------
     // Mobility model:  mob(theta) = 1 + c2 * cos(c3 * theta)
     // theta is the misorientation angle between the grain's <100> family
     // and the local thermal gradient, folded into the cubic fundamental
     // zone by get_cosine_of_minumum_angle_between_q_and_u.
     const double theta_max = std::acos(1.0 / std::sqrt(3.0));  // ~0.9553 rad

     // Rule 1: signs.  c2 < 0 reverses the physical meaning (misaligned
     // grains preferred), which is unphysical for cubic solidification.
     // c3 < 0 is mathematically redundant since cosine is even.
     if (c2 < 0.0) {
       error->all(FLERR,
         "texture_parameters: c2 must be >= 0 (negative amplitude "
         "would mean misaligned grains grow faster than aligned ones, "
         "which is unphysical for cubic solidification)");
     }
     if (c3 < 0.0) {
       error->all(FLERR,
         "texture_parameters: c3 must be >= 0");
     }

     // Rule 2: monotonicity of mob(theta) on [0, theta_max].
     //   d/dtheta mob = -c2 * c3 * sin(c3 * theta) <= 0
     //   requires c3 * theta_max <= pi (with c2 >= 0 from Rule 1).
     if (c3 * theta_max > MY_PI + 1.0e-12) {
       char msg[256];
       snprintf(msg, sizeof(msg),
         "texture_parameters: c3 = %g exceeds monotonicity limit "
         "pi/theta_max ~ %g; mobility would be non-monotonic in "
         "misorientation angle",
         c3, MY_PI / theta_max);
       error->all(FLERR, msg);
     }

     // Rule 3: positivity of mobility at every angle in [0, theta_max].
     //   With c2 >= 0 and c3 * theta_max <= pi, the binding angle is
     //   theta = theta_max where cos is most negative on the interval.
     double mob_min = 1.0 + c2 * std::cos(c3 * theta_max);
     if (mob_min < -1.0e-12) {
       char msg[256];
       snprintf(msg, sizeof(msg),
         "texture_parameters: minimum mobility = %g < 0 at theta = "
         "theta_max (= %g rad); roulette-wheel selection in "
         "site_event_solidification requires non-negative mobility",
         mob_min, theta_max);
       error->all(FLERR, msg);
     }

     // -------- Soft warning --------
     if (c2 == 0.0 && screen) {
       fprintf(screen,
         "WARNING: texture_parameters c2 = 0; mobility has no "
         "orientation dependence\n");
     }
  }
  else if (strcmp(command,"misorientation_function") == 0) {
     if (narg != 2 && narg != 3) error->all(FLERR,"Illegal misorientation_function command");
     max_misorient = atof(arg[0]);
     mis_thresh = atof(arg[1]) * MY_PI / 180.0; // Convert from degrees to radians
     if (narg == 3) {
       misorient_alpha = atof(arg[2]);
       if (misorient_alpha <= 0)
         error->all(FLERR,"Illegal misorientation_function alpha value: must be greater than 0");
     }
     if (max_misorient < 0) 
       error->all(FLERR,"Illegal misorientation_function maximum value: must be greater than 0");
     if (mis_thresh < 0)
       error->all(FLERR,"Illegal misorientation_function threshold value: must be greater than 0");
  }
  else if (strcmp(command,"misorientation_target") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal misorientation_target command");
     if (strcmp(arg[0],"gradient") == 0) {
       misorientation_target_mode = MISORI_TARGET_GRADIENT;
     } else if (strcmp(arg[0],"random") == 0) {
       misorientation_target_mode = MISORI_TARGET_RANDOM;
     } else {
       error->all(FLERR,"Illegal misorientation_target value: use gradient or random");
     }
  }
  else if (strcmp(command,"nucleation_mode") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nucleation_mode command");
    if (strcmp(arg[0],"threshold") == 0) {
      nucleation_mode = NUCLEATION_THRESHOLD;
    } else if (strcmp(arg[0],"continuous") == 0) {
      nucleation_mode = NUCLEATION_CONTINUOUS;
    } else {
      error->all(FLERR,"Illegal nucleation_mode value: use threshold or continuous");
    }
  }
  else if (strcmp(command,"fast_forward_search_window") == 0) {
     if (narg != 1) error->all(FLERR,"Illegal fast_forward_search_window command");
     fast_forward_search_window = atof(arg[0]);
     if (fast_forward_search_window <= 0.0)
       error->all(FLERR,"Illegal fast_forward_search_window value: must be > 0");
  }
  else if (strcmp(command,"temperature_optimization") == 0) {
     // Usage: temperature_optimization <option> <on|off>
     // Options: spatial_grid, element_cache, nodal_precompute
     if (narg != 2) error->all(FLERR,"Illegal temperature_optimization command: requires 2 args (option on/off)");
     bool enable = (strcmp(arg[1],"on") == 0 || strcmp(arg[1],"1") == 0 || strcmp(arg[1],"true") == 0);

     // Get HDF5 source if available to pass flags
     HDF5UnstructuredTemperatureSource* hdf5_source = nullptr;
     if (temperature_source) {
       hdf5_source = dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
     }

     if (strcmp(arg[0],"spatial_grid") == 0) {
       opt_use_spatial_grid = enable;
       if (hdf5_source) hdf5_source->set_use_spatial_grid(enable);
       if (domain->me == 0) fprintf(screen,"Temperature optimization: spatial_grid = %s\n", enable ? "ON" : "OFF");
     } else if (strcmp(arg[0],"element_cache") == 0) {
       opt_use_element_cache = enable;
       if (!enable) opt_use_nodal_precompute = false;  // nodal_precompute requires element_cache
       if (hdf5_source) hdf5_source->set_use_element_cache(enable);
       if (domain->me == 0) fprintf(screen,"Temperature optimization: element_cache = %s\n", enable ? "ON" : "OFF");
     } else if (strcmp(arg[0],"nodal_precompute") == 0) {
       if (enable && !opt_use_element_cache)
         error->all(FLERR,"nodal_precompute requires element_cache to be enabled");
       opt_use_nodal_precompute = enable;
       if (domain->me == 0) fprintf(screen,"Temperature optimization: nodal_precompute = %s\n", enable ? "ON" : "OFF");
     } else {
       error->all(FLERR,"Unknown temperature_optimization option (use: spatial_grid, element_cache, nodal_precompute)");
     }
  }

  // Void generation commands
  else if (strcmp(command,"void_density") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal void_density command");
    if (void_pore_fraction > 0.0)
      error->all(FLERR,"Cannot specify both void_density and void_pore_fraction");
    void_density = atof(arg[0]);
    if (void_density < 0.0)
      error->all(FLERR,"Illegal void_density value: must be >= 0");
    enable_voids = (void_density > 0.0) ? 1 : 0;
  }
  else if (strcmp(command,"void_pore_fraction") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal void_pore_fraction command");
    if (void_density > 0.0)
      error->all(FLERR,"Cannot specify both void_density and void_pore_fraction");
    void_pore_fraction = atof(arg[0]);
    if (void_pore_fraction < 0.0 || void_pore_fraction >= 1.0)
      error->all(FLERR,"Illegal void_pore_fraction value: must be in range [0.0, 1.0)");
    if (void_pore_fraction > 0.3 && domain->me == 0)
      fprintf(screen,"Warning: void_pore_fraction = %.3f is unrealistically high (>30%%)\n", void_pore_fraction);
    enable_voids = (void_pore_fraction > 0.0) ? 1 : 0;
  }
  else if (strcmp(command,"void_size_distribution") == 0) {
    if (narg != 4) error->all(FLERR,"Illegal void_size_distribution command");
    void_radius_mean = atof(arg[0]);
    void_radius_std = atof(arg[1]);
    void_radius_min = atof(arg[2]);
    void_radius_max = atof(arg[3]);
    if (void_radius_mean <= 0.0)
      error->all(FLERR,"Illegal void_size_distribution: mean must be > 0");
    if (void_radius_std < 0.0)
      error->all(FLERR,"Illegal void_size_distribution: std must be >= 0");
    if (void_radius_min < 0.0)
      error->all(FLERR,"Illegal void_size_distribution: min must be >= 0");
    if (void_radius_max <= void_radius_min)
      error->all(FLERR,"Illegal void_size_distribution: max must be > min");
  }

  // Modular temperature source commands
  else if (strcmp(command,"setup_temperature_source") == 0) {
    setup_temperature_source_cmd(narg, arg);
  }
  else if (strcmp(command,"laser_path") == 0) {
    laser_path_cmd(narg, arg);
  }
  else if (strcmp(command,"laser_fluctuations") == 0) {
    laser_fluctuations_cmd(narg, arg);
  }

  else if (strcmp(command,"temperature_smooth") == 0) {
    // Usage: temperature_smooth <tmin> <tmax> <sigma_xy> <sigma_z>
    //                           [alpha] [guard] [passes] [diag_interval]
    //   tmin,tmax:     window inside which smoothing is at full strength (K)
    //   sigma_xy:      Gaussian width in x,y (lattice sites)
    //   sigma_z:       Gaussian width in z   (lattice sites)
    //   alpha:         blend factor, 0=off, 1=replace (default 0.4)
    //   guard:         half-width of taper band outside [tmin,tmax] (default 20 K)
    //   passes:        number of sequential smoothing passes (default 1)
    //   diag_interval: report isotherm compactness (P/sqrt(A) at T>=liquidus)
    //                  every N steps; 0 disables (default 0)
    if (narg < 4 || narg > 8)
      error->all(FLERR,"Illegal temperature_smooth command");
    smooth_tmin     = atof(arg[0]);
    smooth_tmax     = atof(arg[1]);
    smooth_sigma_xy = atof(arg[2]);
    smooth_sigma_z  = atof(arg[3]);
    if (narg >= 5) smooth_alpha         = atof(arg[4]);
    if (narg >= 6) smooth_guard         = atof(arg[5]);
    if (narg >= 7) smooth_passes        = atoi(arg[6]);
    if (narg >= 8) smooth_diag_interval = atoi(arg[7]);
    if (smooth_tmax <= smooth_tmin)
      error->all(FLERR,"temperature_smooth: tmax must be > tmin");
    if (smooth_sigma_xy <= 0.0)
      error->all(FLERR,"temperature_smooth: sigma_xy must be > 0");
    if (smooth_sigma_z <= 0.0)
      error->all(FLERR,"temperature_smooth: sigma_z must be > 0");
    if (smooth_alpha < 0.0 || smooth_alpha > 1.0)
      error->all(FLERR,"temperature_smooth: alpha must be in [0,1]");
    if (smooth_guard < 0.0)
      error->all(FLERR,"temperature_smooth: guard must be >= 0");
    if (smooth_passes < 1)
      error->all(FLERR,"temperature_smooth: passes must be >= 1");
    if (smooth_diag_interval < 0)
      error->all(FLERR,"temperature_smooth: diag_interval must be >= 0");
    temperature_smooth_enabled = true;
    if (domain->me == 0)
      fprintf(screen,
        "Temperature smoothing enabled: window=[%.1f,%.1f] K, "
        "guard=%.1f K, sigma_xy=%.2f sigma_z=%.2f sites, "
        "alpha=%.2f, passes=%d, diag_interval=%d\n",
        smooth_tmin, smooth_tmax, smooth_guard,
        smooth_sigma_xy, smooth_sigma_z,
        smooth_alpha, smooth_passes, smooth_diag_interval);
  }

  else if (strcmp(command,"nodal_smooth") == 0) {
    // Usage: nodal_smooth <sigma_xy> <sigma_z> <passes> <alpha>
    //   sigma_xy, sigma_z: Gaussian widths in lattice units (> 0)
    //   passes:            number of mesh-Laplacian passes (>= 1)
    //   alpha:             pass blend factor in [0,1]
    // Forwards to the temperature source; sources that don't implement
    // the hook silently ignore it (default base-class no-op).
    if (narg != 4)
      error->all(FLERR,"Illegal nodal_smooth command");
    if (!temperature_source)
      error->all(FLERR,"nodal_smooth requires setup_temperature_source first");
    const double sxy = atof(arg[0]);
    const double sz  = atof(arg[1]);
    const int    K   = atoi(arg[2]);
    const double al  = atof(arg[3]);
    if (sxy <= 0.0) error->all(FLERR,"nodal_smooth: sigma_xy must be > 0");
    if (sz  <= 0.0) error->all(FLERR,"nodal_smooth: sigma_z must be > 0");
    if (K    < 1)   error->all(FLERR,"nodal_smooth: passes must be >= 1");
    if (al < 0.0 || al > 1.0)
      error->all(FLERR,"nodal_smooth: alpha must be in [0,1]");
    temperature_source->enable_nodal_smoothing(sxy, sz, K, al);
    if (domain->me == 0)
      fprintf(screen,
        "Nodal mesh-Laplacian smoothing enabled: "
        "sigma_xy=%.3f sigma_z=%.3f passes=%d alpha=%.3f\n",
        sxy, sz, K, al);
  }

  else if (strcmp(command,"single_voxel_cleanup") == 0) {
    // Usage: single_voxel_cleanup on|off
    // Post-smoothing pass that flips 1-voxel grains into the
    // energy-minimizing neighboring grain. Default off.
    if (narg != 1)
      error->all(FLERR,"Illegal single_voxel_cleanup command");
    if (strcmp(arg[0],"on") == 0) single_voxel_cleanup_enabled = true;
    else if (strcmp(arg[0],"off") == 0) single_voxel_cleanup_enabled = false;
    else error->all(FLERR,"single_voxel_cleanup: argument must be on or off");
    if (domain->me == 0)
      fprintf(screen,"Single-voxel grain cleanup %s\n",
              single_voxel_cleanup_enabled ? "enabled" : "disabled");
  }

  else if (strcmp(command,"single_voxel_aggressive_interval") == 0) {
    // Usage: single_voxel_aggressive_interval N
    // After the main phase-transition loop and post-loop ghost sync,
    // sweep every active_flag == 3 site and call flip_single_voxel_grain
    // every N app_update() calls. 0 disables. Requires
    // `single_voxel_cleanup on` to take effect.
    if (narg != 1)
      error->all(FLERR,"Illegal single_voxel_aggressive_interval command");
    single_voxel_aggressive_interval = atoi(arg[0]);
    if (single_voxel_aggressive_interval < 0)
      error->all(FLERR,"single_voxel_aggressive_interval must be >= 0");
    if (domain->me == 0)
      fprintf(screen,"single_voxel_aggressive_interval = %d\n",
              single_voxel_aggressive_interval);
  }

  else if (strcmp(command,"smooth_greedy_multiproposal") == 0) {
    // Usage: smooth_greedy_multiproposal on|off
    // When on, smooth_site() picks the energy-minimizing distinct
    // neighbor spin instead of testing one random proposal per pass.
    // Each nrefine pass becomes a steepest-descent step. Default off
    // (preserves the historical single-random-proposal behavior).
    if (narg != 1)
      error->all(FLERR,"Illegal smooth_greedy_multiproposal command");
    if (strcmp(arg[0],"on") == 0) smooth_greedy_multiproposal_enabled = true;
    else if (strcmp(arg[0],"off") == 0) smooth_greedy_multiproposal_enabled = false;
    else error->all(FLERR,"smooth_greedy_multiproposal: argument must be on or off");
    if (domain->me == 0)
      fprintf(screen,"smooth_greedy_multiproposal %s\n",
              smooth_greedy_multiproposal_enabled ? "enabled" : "disabled");
  }

  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
  When app_update is called, read in an input file
 ------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::app_update(double dt)
{
  // CPU Performance Timing Instrumentation
  static int step_count = 0;
  static double t_temp_update = 0.0, t_fast_forward = 0.0;
  static double t_melting = 0.0, t_mushy = 0.0, t_smoothing = 0.0;
  static double t_comm = 0.0;
  double t_start, t_end;
  step_count++;

  //Reset t_active to assume we don't have a melt pool right now
  t_active = 0;

  // Reset per-call single-voxel cleanup counter
  n_single_voxel_flips = 0;

  // Update temperature first to get current temperatures
  if (use_temperature_source) {
    // Time: Temperature update from HDF5
    t_start = MPI_Wtime();
    // Use new modular temperature source system
    update_temperature_from_source(time);
    t_end = MPI_Wtime();
    t_temp_update += (t_end - t_start);

    // Check for active sites using temperature source's threshold.
    // HDF5 sources expose their own threshold; the Rosenthal source has
    // no intrinsic notion of "active" so we use the app's liquidus tl.
    double fast_forward_threshold = t_room; // Default to t_room

    HDF5UnstructuredTemperatureSource* hdf5_source =
      dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
    RosenthalTemperatureSource* ros_source =
      hdf5_source ? nullptr
                  : dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());
    MoserGreenTemperatureSource* moser_source =
      (hdf5_source || ros_source) ? nullptr
                  : dynamic_cast<MoserGreenTemperatureSource*>(temperature_source.get());
    if (hdf5_source) {
      fast_forward_threshold = hdf5_source->get_fast_forward_threshold();
    } else if (ros_source || moser_source) {
      fast_forward_threshold = tl;
    }

    for (int i = 0; i < nlocal; i++) {
      if (T[i] > fast_forward_threshold) t_active = 1;
    }

    // Use MPI_Allreduce to check if t_active is 0 on all processors
    int global_t_active;
    MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    // If no active sites and temperature source supports time queries, fast forward.
    // For Rosenthal we route through the app's own predictor (which knows
    // the scan_layer geometry); for HDF5 we use the source's predictor.
    // Time: Fast-forward logic
    t_start = MPI_Wtime();
    if (global_t_active == 0 && time > 1e-6 && temperature_source->supports_time_queries()) {
      double local_next_time;
      if (ros_source) {
        local_next_time = rosenthal_next_active_time(time, fast_forward_threshold);
      } else {
        local_next_time = temperature_source->get_next_time_with_temperature(time, fast_forward_threshold);
      }

      // Each rank predicts based on its own local subdomain, so the
      // earliest wake-up across the entire MPI communicator is the
      // single time we may safely advance to. Without this reduction,
      // ranks would skip to different target_times, desynchronizing
      // simulation time and risking MPI hangs in subsequent collectives.
      double next_thermal_time;
      MPI_Allreduce(&local_next_time, &next_thermal_time, 1, MPI_DOUBLE,
                    MPI_MIN, world);

      if (next_thermal_time > time && next_thermal_time < std::numeric_limits<double>::max()) {
        // Fast-forward to whichever comes first: next thermal activity or run end time
        // This prevents skipping past the intended end of the run while still
        // efficiently skipping idle periods with no thermal activity
        double target_time = std::min(next_thermal_time, stoptime);
        double time_skip = target_time - time;

        if (time_skip > 1e-6) {  // Only skip if meaningful time difference
          if (domain->me == 0) {
            if (target_time < next_thermal_time) {
              std::cout << "Fast-forward: Skipping " << time_skip << " seconds from time "
                        << time << " to run end " << target_time
                        << " (next thermal activity at " << next_thermal_time << ")" << std::endl;
            } else {
              std::cout << "Fast-forward: Skipping " << time_skip << " seconds from time "
                        << time << " to " << target_time
                        << " (all temperatures < " << fast_forward_threshold << "K)" << std::endl;
            }
          }

          // Update simulation time
          time = target_time;

          // Update temperatures at new time
          update_temperature_from_source(time);

          // Activate powder after significant time skip (new layer deposition)
          if (time_skip > 1.0) {
            activate_powder_sites();
            last_powder_activation_time = time;
          }
        }
      }
    }
    t_end = MPI_Wtime();
    t_fast_forward += (t_end - t_start);
  }

  // Common processing for both temperature systems

  // Track if any melting occurs this timestep
  bool melting_occurred = false;

  //iterate through all the sites for phase transitions (applies to both systems)
  // Time each phase-transition category around the loop, not per-site
  double t_melt_start = 0, t_melt_accum = 0;
  double t_mushy_start = 0, t_mushy_accum = 0;
  double t_smooth_start = 0, t_smooth_accum = 0;
  bool in_melting = false, in_mushy = false, in_smoothing = false;

  for (int i=0; i<nlocal; i++) {
    // Skip void sites - they never change state
    if (active_flag[i] == 5) continue;

    //Turn the sites on/off depending on the phase data and whether or not the
    //site's temperature has gone above tl. Only do this when melting the first time
    //to avoid repeated initialization
    if( (T[i] >= tl) && active_flag[i] != 2) {
        melting_occurred = true;
        if (!in_melting) { t_melt_start = MPI_Wtime(); in_melting = true; }
        active_flag[i] = 2;
        spin[i] = (int) (nspins * ranapp->uniform());
        // Create random orientation as each site
        vector<double> uq = quaternion::generate_random_unit_quaternions(1);
        q0[i] = uq[0];
        qx[i] = uq[1];
        qy[i] = uq[2];
        qz[i] = uq[3];
        solid_d[i] = 0;
        G[i] = 0;
        V[i] = 0;
        if (nucleation_mode == NUCLEATION_CONTINUOUS) {
          T_prev_map[i] = T[i];
        }
    }
    //If we're molten, call the mushy_phase function to figure out any phase change
    else if (active_flag[i] == 2 && T[i] <= tl) {
        // Flush melting timer if switching categories
        if (in_melting) { t_melt_accum += MPI_Wtime() - t_melt_start; in_melting = false; }
        if (!in_mushy) { t_mushy_start = MPI_Wtime(); in_mushy = true; }
        mushy_phase(i, ranapp);
        if (nucleation_mode == NUCLEATION_CONTINUOUS && active_flag[i] == 2) {
          T_prev_map[i] = T[i];
        }
    }
    //Call smoothing
    else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3)    {
            // Flush prior timers if switching categories
            if (in_melting) { t_melt_accum += MPI_Wtime() - t_melt_start; in_melting = false; }
            if (in_mushy) { t_mushy_accum += MPI_Wtime() - t_mushy_start; in_mushy = false; }
            if (!in_smoothing) { t_smooth_start = MPI_Wtime(); in_smoothing = true; }
            mobility_out[i] = 1;
            smooth_site(i);
            solid_d[i]--;
    }
    // Single-voxel grain cleanup (opt-in). Triggers exactly once per voxel
    // after it exits the smoothing window. Covers the epitaxial-end state
    // (solid_d == -nrefine-1), the nucleation-start state
    // (solid_d == -nrefine-2), and failed-nucleus flipped neighbors
    // (solid_d == -nrefine-3) that never got consumed into a larger
    // grain. A resolved voxel is marked with solid_d = -nrefine-4 so
    // the branch never re-fires. The flipped-neighbor sentinel at
    // -nrefine-3 must remain distinct from the resolved sentinel so a
    // failed nucleus cluster can still be cleaned up.
    else if (single_voxel_cleanup_enabled &&
             active_flag[i] == 3 &&
             solid_d[i] <= -nrefine - 1 &&
             solid_d[i] >  -nrefine - 4) {
            if (in_melting) { t_melt_accum += MPI_Wtime() - t_melt_start; in_melting = false; }
            if (in_mushy) { t_mushy_accum += MPI_Wtime() - t_mushy_start; in_mushy = false; }
            if (in_smoothing) { t_smooth_accum += MPI_Wtime() - t_smooth_start; in_smoothing = false; }
            if (flip_single_voxel_grain(i)) {
                solid_d[i] = -nrefine - 4;
            }
    }
    else {
        // Non-transition site: flush any open timer
        if (in_melting) { t_melt_accum += MPI_Wtime() - t_melt_start; in_melting = false; }
        if (in_mushy) { t_mushy_accum += MPI_Wtime() - t_mushy_start; in_mushy = false; }
        if (in_smoothing) { t_smooth_accum += MPI_Wtime() - t_smooth_start; in_smoothing = false; }
    }
  }
  // Flush any timer still open after the loop
  if (in_melting) t_melt_accum += MPI_Wtime() - t_melt_start;
  if (in_mushy) t_mushy_accum += MPI_Wtime() - t_mushy_start;
  if (in_smoothing) t_smooth_accum += MPI_Wtime() - t_smooth_start;
  t_melting += t_melt_accum;
  t_mushy += t_mushy_accum;
  t_smoothing += t_smooth_accum;

  // Activate powder once when melting first occurs
  // CRITICAL: Use MPI_Allreduce to check if ANY rank has melting, so all ranks call activate_powder_sites()
  // This prevents MPI deadlock where only some ranks enter the function that contains MPI_Allreduce calls
  if (last_powder_activation_time < 0.0) {
    int local_melting = melting_occurred ? 1 : 0;
    int global_melting = 0;
    MPI_Allreduce(&local_melting, &global_melting, 1, MPI_INT, MPI_SUM, world);

    if (global_melting > 0) {
      activate_powder_sites();
      last_powder_activation_time = time;
    }
  }

  // Communicate only KMC-relevant ghost arrays via index-based selection
  t_start = MPI_Wtime();
  timer->stamp();
  comm->all_selective(ghost_iindices, nghost_iarray,
                      ghost_dindices, nghost_darray);
  timer->stamp(TIME_COMM);
  t_end = MPI_Wtime();
  t_comm += (t_end - t_start);

  // Aggressive full-domain single-voxel grain sweep. Runs post-ghost-sync
  // so spin/active_flag ghosts are fresh. flip_single_voxel_grain(i)
  // no-ops on voxels with a same-spin 26-neighbor or an immature
  // neighborhood, so it is safe to call on every solidified site. The
  // narrow solid_d-gated cleanup in the main phase-transition loop above
  // is retained and independent; both contribute to n_single_voxel_flips.
  if (single_voxel_cleanup_enabled &&
      single_voxel_aggressive_interval > 0 &&
      step_count % single_voxel_aggressive_interval == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (active_flag[i] != 3) continue;
      if (flip_single_voxel_grain(i)) {
        // Mark resolved so the narrow main-loop branch won't re-fire,
        // but only on the sentinel range it would otherwise catch.
        // Immature-neighborhood sites return false and are retried
        // on the next sweep.
        if (solid_d[i] <= -nrefine - 1 && solid_d[i] > -nrefine - 4) {
          solid_d[i] = -nrefine - 4;
        }
      }
    }
  }

  // Optional isotherm-compactness diagnostic for tuning smoothing.
  // Ghosts are guaranteed fresh here (just synced above).
  if (temperature_smooth_enabled && smooth_diag_interval > 0 &&
      step_count % smooth_diag_interval == 0) {
    compute_smoothing_diagnostics();
  }

  // Report single-voxel cleanup activity. Reduced across ranks so the
  // log line reflects the global domain, not just rank 0.
  if (single_voxel_cleanup_enabled) {
    long long local_flips = n_single_voxel_flips;
    long long global_flips = 0;
    MPI_Allreduce(&local_flips, &global_flips, 1, MPI_LONG_LONG, MPI_SUM, world);
    if (global_flips > 0 && domain->me == 0) {
      fprintf(screen,"single_voxel_cleanup: flipped %lld voxels this step\n",
              global_flips);
    }
  }

  // Print timing summary every 100 steps (on rank 0 only)
  if (step_count % 100 == 0 && domain->me == 0) {
    double total_time = t_temp_update + t_fast_forward + t_melting + t_mushy + t_smoothing + t_comm;
    double t_temp_other = t_temp_update - g_t_temp_prepare - g_t_temp_site_loop - g_t_cache_build;
    double steady_state_time = total_time - g_t_cache_build;

    fprintf(screen, "\n=== CPU Timing Summary (after %d steps, %.4f total sec) ===\n",
            step_count, total_time);
    fprintf(screen, "  Temperature update:   %8.3f s (%5.1f%%)\n",
            t_temp_update, 100.0*t_temp_update/total_time);
    if (g_t_cache_build > 0.001) {
      fprintf(screen, "    - Cache build:      %8.3f s (%5.1f%%)  [One-time per layer]\n",
              g_t_cache_build, 100.0*g_t_cache_build/total_time);
    }
    fprintf(screen, "    - Prepare:          %8.3f s (%5.1f%%)  [Nodal temp precompute]\n",
            g_t_temp_prepare, 100.0*g_t_temp_prepare/total_time);
    fprintf(screen, "    - Site loop:        %8.3f s (%5.1f%%)  [Per-site interpolation]\n",
            g_t_temp_site_loop, 100.0*g_t_temp_site_loop/total_time);
    if (t_temp_other > 0.001) {
      fprintf(screen, "    - Other:            %8.3f s (%5.1f%%)  [Layer load, etc.]\n",
              t_temp_other, 100.0*t_temp_other/total_time);
    }
    fprintf(screen, "  Fast-forward logic:   %8.3f s (%5.1f%%)  [Time skip checks]\n",
            t_fast_forward, 100.0*t_fast_forward/total_time);
    fprintf(screen, "  Melting:              %8.3f s (%5.1f%%)  [T >= Tl transitions]\n",
            t_melting, 100.0*t_melting/total_time);
    fprintf(screen, "  Mushy phase:          %8.3f s (%5.1f%%)  [Nucleation + solidification + gradients]\n",
            t_mushy, 100.0*t_mushy/total_time);
    fprintf(screen, "  Smoothing:            %8.3f s (%5.1f%%)  [Grain boundary smoothing]\n",
            t_smoothing, 100.0*t_smoothing/total_time);
    fprintf(screen, "  Communication:        %8.3f s (%5.1f%%)  [MPI comm]\n",
            t_comm, 100.0*t_comm/total_time);
    fprintf(screen, "  --------------------------------------------------------\n");
    fprintf(screen, "  Average time/step:    %8.4f s (overall)\n", total_time / step_count);
    fprintf(screen, "  Steady-state avg:     %8.4f s (excl. cache build)\n", steady_state_time / step_count);
    fprintf(screen, "=========================================================\n\n");
  }

    timer->stamp(TIME_UPDATE);
}


/* ----------------------------------------------------------------------
   Nucleation site initializer. Find the volume of a voxel from dx^3 and Multiply by no.
   This will be the average number of nucleation sites in the voxel. This is also the
   fraction of spins that we want to be able to nucleate new grains. If the value is greater
   than 1 (which we should avoid), allow all spins to nucleate. If not, call a random number
   between zero and one. If the number is less than the fraction, make true. If not, make false.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_spins(RandomPark *random) {
    timer->stamp();
    
    nucleation_flags = new int[nspins];
    double nucleationFraction = dx * dx * dx * no;
    
    // Initialize all spins as nucleation sites (fallback behavior)
    if(nucleationFraction >= 1.0) {
        fprintf(screen,"Nucleation fraction is greater than 1. Decrease no or increase mesh resolution. no = %f\n", nucleationFraction);
        for (int i = 0; i < nspins; i++) {
            nucleation_flags[i] = 1;
        }
    }
    //Do a random number test and allow the spin to nucleate if less than
    else {
        for (int i = 0; i < nspins; i++) {
            if(random->uniform() <= nucleationFraction) {
                nucleation_flags[i] = 1;
            }
            else {
                nucleation_flags[i] = 0;
            }
        }
    }   
    
    timer->stamp(TIME_APP);
}


/* ----------------------------------------------------------------------
   set site value ptrs each time iarray/darray are reallocated
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::grow_app()
{
  // Call parent to set up quaternion arrays (q0, qx, qy, qz)
  AppPottsQuaternion::grow_app();
  
  // Set up additional arrays for this class
  active_flag = iarray[1];
  mobility_out = darray[4];
  T = darray[5];
  solid_d = darray[6];
  G = darray[7];
  V = darray[8];
}


/* ----------------------------------------------------------------------
   initialize before each run
   check validity of site values
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::init_app()
{
  // Duplicate parent's array allocation logic without quaternion randomization
  delete[] sites;
  delete[] unique;
  delete[] unique_neigh;
  sites = new int[1 + maxneigh];
  unique = new int[1 + maxneigh];
  unique_neigh = new int[1 + maxneigh];
  int invalid_count = 0;
  
  dt_sweep = 1.0 / maxneigh;
  
  // Clean up our own arrays
  if (unique_dot) delete [] unique_dot;
  if (neigh_dist) delete [] neigh_dist;
  
  // Allocate our own arrays
  unique_dot = new double[1 + maxneigh];
  
  double sqrt2 = 1.4142135624;
  double sqrt3 = 1.7320508076;
  RandomPark random(3000);

  dt_sweep = dt;

  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  nucleation_flags = new int[nspins];
  nucleation_temps = new double[nspins];
	nucleation_sizes = new double[nspins];

  int flag = 0;
	int flagall;
    for (int i = 0; i < nlocal; i++) {
			if (spin[i] < 0 || spin[i] > nspins) {
				flag = 1;
        fprintf(screen,"Bad spin %i\n",spin[i]);
			}
      if (active_flag[i] != 3) { 
        // Create random orientation at each site that wasn't read as solid from input file
        vector<double> uq = quaternion::generate_random_unit_quaternions(1);
        // std::cout << "Generating new quaternions" << std::endl;
        q0[i] = uq[0];
        qx[i] = uq[1];
        qy[i] = uq[2];
        qz[i] = uq[3];
      }
    }

  timer->stamp();
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  timer->stamp(TIME_COMM);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");
  
	//Initialize the nucleation_flags vector
	if (domain->me==0) {
		nucleation_spins(ranapp);    
	}

	timer->stamp();
	MPI_Bcast(nucleation_flags,nspins, MPI_INT,0,world);
	timer->stamp(TIME_COMM);

	//Initialize the nucleation_temps and nucleation_sizes vectors
	if (domain->me==0) {
			nucleation_init();    
	}

	timer->stamp();
	MPI_Bcast(nucleation_temps,nspins, MPI_DOUBLE,0,world);
	MPI_Bcast(nucleation_sizes,nspins, MPI_DOUBLE,0,world);
	timer->stamp(TIME_COMM);

	// Validate parameters for continuous nucleation mode
	if (nucleation_mode == NUCLEATION_CONTINUOUS) {
	  if (tsig <= 0.0)
	    error->all(FLERR, "undercooling_deviation must be > 0 for continuous nucleation mode");
	  if (!use_temperature_source)
	    error->all(FLERR, "continuous nucleation mode requires a temperature source");
	}

	if (domain->me == 0) {
	  if (nucleation_mode == NUCLEATION_THRESHOLD) {
	    fprintf(screen, "Nucleation mode: threshold (tc=%.2f, tsig=%.2f)\n", tc, tsig);
	  } else {
	    fprintf(screen, "Nucleation mode: continuous Gaussian (tc=%.2f, tsig=%.2f)\n", tc, tsig);
	  }
	}

	//Initialize the neigh_dist array need to fill with good values
	neigh_dist = new double[26];
	neigh_dist[0] = sqrt3 * dx;
	neigh_dist[1] = sqrt2 * dx;
	neigh_dist[2] = sqrt3 * dx;
	neigh_dist[3] = sqrt2 * dx;
	neigh_dist[4] = dx;
	neigh_dist[5] = sqrt2 * dx;
	neigh_dist[6] = sqrt3 * dx;
	neigh_dist[7] = sqrt2 * dx;
	neigh_dist[8] = sqrt3 * dx;
	neigh_dist[9] =  sqrt2 * dx;
	neigh_dist[10] = dx;
	neigh_dist[11] =  sqrt2 * dx;
	neigh_dist[12] = sqrt3 * dx;
	neigh_dist[13] = sqrt3 * dx;
	neigh_dist[14] =  sqrt2 * dx;
	neigh_dist[15] = dx;
	neigh_dist[16] =  sqrt2 * dx;
	neigh_dist[17] = sqrt3 * dx;
	neigh_dist[18] = sqrt2 * dx;
	neigh_dist[19] = sqrt3 * dx;
	neigh_dist[20] =  sqrt2 * dx;
	neigh_dist[21] = dx;
	neigh_dist[22] =  sqrt2 * dx;
	neigh_dist[23] = sqrt3 * dx;
	neigh_dist[24] = sqrt2 * dx;
	neigh_dist[25] = sqrt3 * dx;

  t_cool_max = tl - ts;

  //Check that our timestep is small enough	
  double max_front_vel = 0;
	int power = solid_front_length -1;
	for(int k = 0; k < solid_front_length; k++) {
		max_front_vel = max_front_vel + solid_front_coeffs[k] * pow(tl - ts, power);
		power--;
	}

  
  if(dt > dx/max_front_vel) {
      fprintf(screen,"Temperature timestep too large, reduce to %f\n", dx/max_front_vel);
          error->all(FLERR,"Temperature timestep too large");
  }
  else{
    if(domain->me == 0) {
      fprintf(screen,"Maximum allowable timestep is %e\n", dx/max_front_vel);
    }
  }
  
  // Always use modular temperature source
  if (domain->me == 0) {
    std::cout << "Using modular temperature source" << std::endl;
    std::cout << "Using liquidus temperature tl = " << tl << " K" << std::endl;
    std::cout << "Using solidus temperature ts = " << ts << " K" << std::endl;
  }

  // Generate voids if enabled
  if (enable_voids) {
    generate_voids(ranapp);
  }

  // Note: Powder activation happens after meltpool appears, not during init

	this->app_update(0.0);
}


/* ----------------------------------------------------------------------
   Override parent site_energy to only calculate energy for solidified sites
   and exclude non-solidified neighbors from energy calculation
------------------------------------------------------------------------- */

double AppAdditiveExtTempTexture::site_energy(int i) {
  timer->stamp();

  // Voids have no energy
  if (active_flag[i] == 5) {
    timer->stamp(TIME_SOLVE);
    return 0.0;
  }

  // Condition 1: If active_flag != 3 (not solidified), return zero energy
  if (active_flag[i] != 3) {
    timer->stamp(TIME_SOLVE);
    return 0.0;
  }
  
  // Same algorithm as parent class but with active_flag filtering
  double energy = 0.0;
  vector<double> qi{q0[i], qx[i], qy[i], qz[i]};
  
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];

    // Exclude void neighbors and non-solidified neighbors
    if (active_flag[nj] == 5 || (active_flag[nj] != 3 && active_flag[nj] != 1)) {
      continue;
    }
    
    vector<double> qj{q0[nj], qx[nj], qy[nj], qz[nj]};
    double di = disorientation::compute_disorientation(symmetries, qi, qj);
    double ratio = di / theta_cut;

    // Read-Shockley equation logic (same as parent)
    if (ratio <= 0)
      continue;
    else if (ratio < 1.0)
      energy += ratio * (1 - log(ratio));
    else
      energy += 1.0;
  }
  
  // Each site carries half the grain boundary energy
  // Neighbor sites carry the other half
  timer->stamp(TIME_SOLVE);
  return 0.5 * energy;
}

/* ----------------------------------------------------------------------
   site_event_rejection is intentionally not used by this app.

   AppAdditiveExtTempTexture overrides app_update() (see line ~400) to drive
   site evolution directly via mushy_phase()/site_event_solidification() and
   smooth_site(); the framework's rejection-KMC sweep loop in
   AppLattice::iterate_rejection is bypassed entirely. This override exists
   only to satisfy the pure virtual declared in AppLattice. If something
   ever calls it, that indicates the standard sweep path has been wired up
   on this app by mistake -- bail out loudly rather than silently running
   stale logic.
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::site_event_rejection(int /*i*/, RandomPark * /*random*/)
{
  error->one(FLERR,
    "AppAdditiveExtTempTexture::site_event_rejection should never be "
    "called; this app drives site evolution from app_update()");
}

/* ----------------------------------------------------------------------
   Execute a nucleation event at site i with undercooling Tcool.
   Sets site to solid, computes gradient and solidification rate,
   and grows the nucleus via nucleation_particle_flipper.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::execute_nucleation_event(int i, double Tcool) {
    active_flag[i] = 3;
    solid_d[i] = -nrefine - 2;

    std::vector<double> solid_G(4);
    solid_G = normal_finder(i);
    G[i] = solid_G[3];

    int power = solid_front_length - 1;
    for (int k = 0; k < solid_front_length; k++) {
        V[i] = V[i] + solid_front_coeffs[k] * pow(Tcool, power);
        power--;
    }

    naccept++;
    nucleation_particle_flipper(i, round(nucleation_sizes[spin[i]] / pow(dx, 3)), ranapp);

    if (nucleation_mode == NUCLEATION_CONTINUOUS) {
      T_prev_map.erase(i);
    }
}

/* ----------------------------------------------------------------------
   Perform evolution for sites in the mushy zone. There are several things that need to happen.
   1. Determine if the site is solid or liquid (from active_flag)
   2. Determine if the site should nucleate a new grain (from nucleation_flags)
   3. If our current site is liquid & can't nucleate, have it try to switch to a solid neighbor
      (with 4 calculated from the undercooling in someway, not temperature)
   4. If our current site is liquid & can nucleate, check if local undercooling is equal to its 
      critical temp. If so, change the active_flag value to solid. If not, see if there are any solid sites
      that should capture it.
   5. If our current site is solid, see if it should flip to a neighboring solid value (with
      mobility calculated from undercooling.)
THIS NEEDS UPDATED TO INCLUDE TEXTURE
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::mushy_phase(int i, RandomPark *random){
    // Skip void sites - they never change state
    if (active_flag[i] == 5) return;

    double Tcool = tl - T[i];


    //Our site should always be molten and below tl
    //Check if it's eligible to nucleate
    if(nucleation_flags[spin[i]]) {
        if (nucleation_mode == NUCLEATION_THRESHOLD) {
            // Threshold mode: nucleate if undercooling exceeds per-spin critical value
            if(Tcool >= nucleation_temps[spin[i]]){
                execute_nucleation_event(i, Tcool);
                return;
            }
            else {
                site_event_solidification(i, Tcool, random);
            }
        }
        else {
            // Continuous Gaussian nucleation mode: probabilistic activation
            auto it = T_prev_map.find(i);
            if (it != T_prev_map.end()) {
                double dT_increment = it->second - T[i];  // positive when cooling
                if (dT_increment > 0.0 && Tcool > 0.0) {
                    double exponent = (Tcool - tc) * (Tcool - tc) / (2.0 * tsig * tsig);
                    double dn_ddT = (1.0 / (sqrt(2.0 * MY_PI) * tsig)) * exp(-exponent);
                    double P_nucleate = dn_ddT * dT_increment;
                    if (P_nucleate > 1.0) P_nucleate = 1.0;

                    if (random->uniform() < P_nucleate) {
                        execute_nucleation_event(i, Tcool);
                        return;
                    }
                }
            }
            // Did not nucleate (or no previous temp): allow epitaxial growth
            site_event_solidification(i, Tcool, random);
        }
    }
    else {
        site_event_solidification(i, Tcool, random);
    }
}

/* ----------------------------------------------------------------------
   Perform solidification event for a site in the mushy zone.
   This handles the common logic for both nucleation and non-nucleation paths.
   Updates solid front distance and attempts epitaxial growth from neighboring sites.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::site_event_solidification(int i, double Tcool, RandomPark *random) {
    int nevent = 0;
    int value;
    double dotValue = 0;
    int solidNeigh = 0;

    //Add the distance of the front travel. Need to multiply by timestep to get distance
    //Only start growing if we're on the boundary or have a solid neighbor
    if (numneigh[i] < 26) solidNeigh = 1;
    else {
      for (int j = 0; j < numneigh[i]; j++) {
        // Exclude void neighbors
        if(active_flag[neighbor[i][j]] == 5) continue;
        if(active_flag[neighbor[i][j]] == 3 || active_flag[neighbor[i][j]] == 1) {
          solidNeigh = 1;
          break;
        }
      }
    }

    if(solidNeigh == 1) {
      int power = solid_front_length - 1;
      for(int k = 0; k < solid_front_length; k++) {
          solid_d[i] = solid_d[i] + solid_front_coeffs[k] * pow(Tcool, power) * time_step;
          power--;
      }
    
      //Go through neighbor list and add them to possible switches
      for (int j = 0; j < numneigh[i]; j++) {
          // Exclude void neighbors
          if(active_flag[neighbor[i][j]] == 5) continue;
          if(neigh_dist[2] <= solid_d[i] && (active_flag[neighbor[i][j]] == 1 || active_flag[neighbor[i][j]] == 3)) {
              // Calculate temperature gradient/grain misorientation and store in array
              // Use cumulative probability for random sampling
              //Exclude gas or molten sites from the Potts neighbor tally
              double melt_misori_val = melt_misorientation(neighbor[i][j],c1,c2,c3);
              dotValue += melt_misori_val;
              unique_dot[nevent] = dotValue;
              value = spin[neighbor[i][j]];
              unique[nevent] = value;
              unique_neigh[nevent] = neighbor[i][j];
              nevent++;
          }
      }
    }
    //If no neighbor is eligible, return before changing anything. Will try next sweep.
    if (nevent == 0) return;
    
    // unique_dot[nevent - 1] is the total cumulative mobility (last entry).
    double dran = (unique_dot[nevent - 1]*random->uniform());
    // Roulette-wheel selection: find the first j with dran <= unique_dot[j].
    // Loop must run through j = nevent - 1 so the last neighbor is reachable;
    // the early return after flip_site keeps the body single-shot.
    for( int j = 0; j < nevent; j++) {
        if(dran <= unique_dot[j]) {
            int neighran = unique_neigh[j];
            SiteState s1(unique[j],{q0[neighran],qx[neighran],qy[neighran],qz[neighran]});
            flip_site(i,s1);
            active_flag[i] = 3;
            solid_d[i] = -1;
            if (nucleation_mode == NUCLEATION_CONTINUOUS) {
              T_prev_map.erase(i);
            }

            std::vector<double> solid_G(4);  // Now returns [norm_x, norm_y, norm_z, magnitude]
            solid_G = normal_finder(i); //Update gradient
            G[i] = solid_G[3];  // Use the actual gradient magnitude directly

            int power = solid_front_length - 1;
            for(int k = 0; k < solid_front_length; k++) {
               V[i] = V[i] + solid_front_coeffs[k] * pow(Tcool, power); //Update solidification rate
                power--;
            }
            naccept++;

            // Apply misorientation based on temperature gradient
            apply_misorientation(i, Tcool, ranapp);
            return;
        }
    }
}

/* ----------------------------------------------------------------------
    Apply misorientation based on temperature gradient and cooling rate
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::apply_misorientation(int i, double Tcool, RandomPark *ranapp) {
    if(max_misorient <= 0) return;
    
    // Exponential function: grows slowly initially, then rapidly in last ~10%
    double normalized_temp = Tcool / t_cool_max;
    double mis_angle = max_misorient * (exp(misorient_alpha * normalized_temp) - 1.0) / (exp(misorient_alpha) - 1.0) * MY_PI/180;

    vector<double> target_dir;
    if (misorientation_target_mode == MISORI_TARGET_GRADIENT) {
        // Calculate the unit-vector surface norm (use voxel counting)
        vector<double> grad_full = normal_finder(i);  // Returns [norm_x, norm_y, norm_z, magnitude]
        target_dir = {grad_full[0], grad_full[1], grad_full[2]};
    } else if (misorientation_target_mode == MISORI_TARGET_RANDOM) {
        target_dir = sample_random_unit_vector(ranapp);
    } else {
        error->all(FLERR,"Unknown misorientation target mode");
    }

    // Create quaternion vector from site's orientation
    vector<double> q_site = {q0[i], qx[i], qy[i], qz[i]};
    
    // Normalize quaternion to ensure it's a unit quaternion
    double q_mag = sqrt(q_site[0]*q_site[0] + q_site[1]*q_site[1] + q_site[2]*q_site[2] + q_site[3]*q_site[3]);
    if (q_mag > 1e-15) {
        q_site[0] /= q_mag;
        q_site[1] /= q_mag;
        q_site[2] /= q_mag;
        q_site[3] /= q_mag;
    }

    vector<double> q_new = quaternion::rotate_q_towards_u(q_site,target_dir, mis_angle);

    q0[i] = q_new[0];
    qx[i] = q_new[1];
    qy[i] = q_new[2];
    qz[i] = q_new[3];

    // Compute crystal-symmetry-aware disorientation (consistent with site_energy)
    double di_rad;
    if (symmetries.empty()) {
        // Fallback: direct angle (no symmetry)
        double dot_product = q_site[0]*q_new[0] + q_site[1]*q_new[1]
                           + q_site[2]*q_new[2] + q_site[3]*q_new[3];
        di_rad = 2.0 * acos(std::min(1.0, std::abs(dot_product)));
    } else {
        double di_deg = disorientation::compute_disorientation(symmetries, q_site, q_new);
        di_rad = di_deg * MY_PI / 180.0;
    }

    if(di_rad >= mis_thresh) {
        spin[i] = (int) (nspins * ranapp->uniform()); //If greater than misorientation threshold, make new grain
    }
}

/* ----------------------------------------------------------------------
    Only nucleating one site at a time introduce lattice size dependency. Here we will
    use a user-defined nucleation particle size and flip neighboring sites until that size is met
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_particle_flipper(int i, int partRad, RandomPark *random) {
    
    //If one site is big enough to satisfy, skip evertyhing
    if(partRad <= 0) return;
    
    int nSites = partRad;
    int nSitesIn = nSites;
    int nearest_neigh [ ] = {4, 10, 12, 15, 21, 13};
    int second_nearest [ ] = {9,16,3,22,14,11,20,5,1,24,7,18};
    int third_nearest [ ] = {0,25,23,2,6,19,17,8};
    int nneigh = 0;
    int possible_neigh[26];
    SiteState s_in(spin[i],{q0[i], qx[i], qy[i], qz[i]});

    

    //Its hard to go through shells iteravely if the number of neighbors isn't the full 26.
    //Check if this is the case and just loop through the neihgbor list if so
    if(numneigh[i] != 26) {

        
        for(int j = 0; j < numneigh[i]; j++) {
            if(active_flag[neighbor[i][j]] == 2) {
                int i_chosen = neighbor[i][j];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                if (i_chosen < nlocal) {
                    solid_d[i_chosen] = -nrefine -3;
                    G[i_chosen] = G[i];
                    V[i_chosen] = V[i];
                }
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
    }
    else {
        //Go through neighbors and nucleate liquid ones
        //Create shuffled indices for each shell
        std::vector<int> shell1_indices(6);
        std::vector<int> shell2_indices(12);
        std::vector<int> shell3_indices(8);
        
        // Initialize index arrays
        for(int j = 0; j < 6; j++) shell1_indices[j] = j;
        for(int j = 0; j < 12; j++) shell2_indices[j] = j;
        for(int j = 0; j < 8; j++) shell3_indices[j] = j;
        
        // Shuffle each shell's indices using Fisher-Yates algorithm
        // Shell 1 (nearest neighbors)
        for(int j = 5; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell1_indices[j], shell1_indices[k]);
        }
        
        // Shell 2 (second nearest)
        for(int j = 11; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell2_indices[j], shell2_indices[k]);
        }
        
        // Shell 3 (third nearest)
        for(int j = 7; j > 0; j--) {
            int k = (int)(random->uniform() * (j + 1));
            std::swap(shell3_indices[j], shell3_indices[k]);
        }
        
        //Do 1st shell in randomized order
        for(int j = 0; j < 6; j++) {
            int idx = shell1_indices[j];
            if(active_flag[neighbor[i][nearest_neigh[idx]]] == 2) {
                int i_chosen = neighbor[i][nearest_neigh[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                if (i_chosen < nlocal) {
                    solid_d[i_chosen] = -nrefine -3;
                    G[i_chosen] = G[i];
                    V[i_chosen] = V[i];
                }
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
        //Do 2nd shell in randomized order
        for(int j = 0; j < 12; j++) {
            int idx = shell2_indices[j];
            if(active_flag[neighbor[i][second_nearest[idx]]] == 2) {
                int i_chosen = neighbor[i][second_nearest[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                if (i_chosen < nlocal) {
                    solid_d[i_chosen] = -nrefine -3;
                    G[i_chosen] = G[i];
                    V[i_chosen] = V[i];
                }
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }      
          }
        //Do 3rd shell in randomized order   
        for(int j = 0; j < 8; j++) {
            int idx = shell3_indices[j];
            if(active_flag[neighbor[i][third_nearest[idx]]] ==2) {
                int i_chosen = neighbor[i][third_nearest[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                if (i_chosen < nlocal) {
                    solid_d[i_chosen] = -nrefine -3;
                    G[i_chosen] = G[i];
                    V[i_chosen] = V[i];
                }
                nSites--;
                naccept++;
            }
            if(nSites <= 0) {
                return;
            }
        }
    }
    //If we didn't fill any sites this time, just quit
    if (nSites == nSitesIn) {
        return;
    }
    
    //If we still haven't satisfied the particle size, pick a neighbor at random and solidify from there.
    //Build a list of same-particle neighbors and pick one randomly
    for(int j =0; j < numneigh[i]; j++) {
        // Exclude void neighbors
        if(active_flag[neighbor[i][j]] == 5) continue;
        if(spin[neighbor[i][j]] == spin[i] && active_flag[neighbor[i][j]] == 3) {
            possible_neigh[nneigh] = j;
            nneigh++;
        }
    }
    //If no possible nieghbors, quit
    if(nneigh == 0) {
        return;
    }
    //Otherwise, randomly pick a possilbe neighbor
    int neighran =  round(((nneigh -1) * random->uniform()));
    
    int next_site = neighbor[i][possible_neigh[neighran]];
    if (next_site < nlocal)
        nucleation_particle_flipper(next_site, nSites, random);
    return;
    
}


//Texture specific functions start here

// Calculate the local direction of the largest temperature gradient
// using weighted least squares with multiple neighbors for improved accuracy
std::vector<double> AppAdditiveExtTempTexture::normal_finder(int site)
{
	// Initialize gradient components
	double grad_x = 0, grad_y = 0, grad_z = 0;
	
	// Set up neighbor offset mapping for 3D cubic lattice (SC_26N)
	// Order: i,j,k from -1 to 1 (excluding center at 0,0,0)
	int offset_map[26][3] = {
		{-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},  // i=-1, j=-1
		{-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},  // i=-1, j=0 
		{-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},  // i=-1, j=1
		{ 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},  // i=0, j=-1
		{ 0, 0,-1},             { 0, 0, 1},  // i=0, j=0 (skip center)
		{ 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},  // i=0, j=1
		{ 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},  // i=1, j=-1
		{ 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},  // i=1, j=0
		{ 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}   // i=1, j=1
	};

	// Special case: top of melt pool where some neighbors are inactive
	// Use only neighbors at or below current site's z-coordinate
	bool is_melt_surface = (active_flag[neighbor[site][13]] <= 1);

	if (is_melt_surface) {
		// Filter neighbors to only include those at or below current z-level
		double site_z = xyz[site][2];
		
		// Weighted least squares with z-filtered neighbors
		double AtA[9] = {0}; // 3x3 matrix A^T * W * A
		double Atb[3] = {0}; // 3x1 vector A^T * W * b
		double T_center = T[site];
		int valid_neighbors = 0;
		
		for (int n = 0; n < numneigh[site] && n < 26; n++) {
			int neighbor_site = neighbor[site][n];
			
			// Skip invalid neighbors
			if (neighbor_site < 0 || neighbor_site >= app->nlocal + app->nghost) continue;
			
			// Only use neighbors at or below current site's z-coordinate
			if (xyz[neighbor_site][2] > site_z) continue;
			
			// Get neighbor offset coordinates
			double dx_n = offset_map[n][0] * dx;
			double dy_n = offset_map[n][1] * dx;
			double dz_n = offset_map[n][2] * dx;
			
			// Temperature difference
			double dT = T[neighbor_site] - T_center;
			
			// Distance and weight
			double dist = sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
			double weight = 1.0 / (dist*dist + 1e-12);
			
			// Build weighted normal equations
			AtA[0] += weight * dx_n * dx_n;  // AtA[0,0]
			AtA[1] += weight * dx_n * dy_n;  // AtA[0,1]
			AtA[2] += weight * dx_n * dz_n;  // AtA[0,2]
			AtA[3] += weight * dy_n * dx_n;  // AtA[1,0] 
			AtA[4] += weight * dy_n * dy_n;  // AtA[1,1]
			AtA[5] += weight * dy_n * dz_n;  // AtA[1,2]
			AtA[6] += weight * dz_n * dx_n;  // AtA[2,0]
			AtA[7] += weight * dz_n * dy_n;  // AtA[2,1]
			AtA[8] += weight * dz_n * dz_n;  // AtA[2,2]
			
			Atb[0] += weight * dx_n * dT;
			Atb[1] += weight * dy_n * dT;
			Atb[2] += weight * dz_n * dT;
			
			valid_neighbors++;
		}

		// Compute gradient using weighted least squares with Cramer's rule
		// Note: Diagnostics (2M+ calls) showed matrix always non-singular and valid_neighbors always >= 3
		double det = AtA[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) -
		             AtA[1]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) +
		             AtA[2]*(AtA[3]*AtA[7] - AtA[4]*AtA[6]);

		grad_x = (Atb[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) -
		          AtA[1]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) +
		          AtA[2]*(Atb[1]*AtA[7] - AtA[4]*Atb[2])) / det;

		grad_y = (AtA[0]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) -
		          Atb[0]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) +
		          AtA[2]*(AtA[3]*Atb[2] - Atb[1]*AtA[6])) / det;

		grad_z = (AtA[0]*(AtA[4]*Atb[2] - Atb[1]*AtA[7]) -
		          AtA[1]*(AtA[3]*Atb[2] - Atb[1]*AtA[6]) +
		          Atb[0]*(AtA[3]*AtA[7] - AtA[4]*AtA[6])) / det;
		
		// Normalize and return (with magnitude as 4th element)
		// Note: Diagnostics (2M+ calls) showed norm always > 1e-12
		double norm = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
		std::vector<double> result(4);  // 4 elements: [norm_x, norm_y, norm_z, magnitude]

		result[0] = fabs(grad_x)/norm;
		result[1] = fabs(grad_y)/norm;
		result[2] = fabs(grad_z)/norm;
		result[3] = norm;  // Store the actual gradient magnitude

		return result;
	}

	// Use weighted least squares with all available neighbors
	
	// Weighted least squares matrices: A*grad = b
	// A is 26x3 matrix of [dx dy dz] for each neighbor
	// b is 26x1 vector of temperature differences
	// weights based on inverse distance
	double AtA[9] = {0}; // 3x3 matrix A^T * W * A
	double Atb[3] = {0}; // 3x1 vector A^T * W * b
	
	double T_center = T[site];
	
	for (int n = 0; n < numneigh[site] && n < 26; n++) {
		int neighbor_site = neighbor[site][n];
		
		// Skip invalid neighbors
		if (neighbor_site < 0 || neighbor_site >= app->nlocal + app->nghost) continue;
		
		// Get neighbor offset coordinates
		double dx_n = offset_map[n][0] * dx;  // x offset in physical units
		double dy_n = offset_map[n][1] * dx;  // y offset 
		double dz_n = offset_map[n][2] * dx;  // z offset
		
		// Temperature difference
		double dT = T[neighbor_site] - T_center;
		
		// Distance and weight (inverse distance squared with small regularization)
		double dist = sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
		double weight = 1.0 / (dist*dist + 1e-12);
		
		// Build weighted normal equations: AtA * grad = Atb
		// A matrix row: [dx_n, dy_n, dz_n]
		AtA[0] += weight * dx_n * dx_n;  // AtA[0,0]
		AtA[1] += weight * dx_n * dy_n;  // AtA[0,1]
		AtA[2] += weight * dx_n * dz_n;  // AtA[0,2]
		AtA[3] += weight * dy_n * dx_n;  // AtA[1,0] 
		AtA[4] += weight * dy_n * dy_n;  // AtA[1,1]
		AtA[5] += weight * dy_n * dz_n;  // AtA[1,2]
		AtA[6] += weight * dz_n * dx_n;  // AtA[2,0]
		AtA[7] += weight * dz_n * dy_n;  // AtA[2,1]
		AtA[8] += weight * dz_n * dz_n;  // AtA[2,2]
		
		Atb[0] += weight * dx_n * dT;    // Atb[0]
		Atb[1] += weight * dy_n * dT;    // Atb[1] 
		Atb[2] += weight * dz_n * dT;    // Atb[2]
	}
	
	// Solve 3x3 system AtA * grad = Atb using Cramer's rule
	// Note: Diagnostics (2M+ calls) showed matrix always non-singular
	double det = AtA[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) -
	             AtA[1]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) +
	             AtA[2]*(AtA[3]*AtA[7] - AtA[4]*AtA[6]);

	// Compute gradient using Cramer's rule
	grad_x = (Atb[0]*(AtA[4]*AtA[8] - AtA[5]*AtA[7]) -
	          AtA[1]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) +
	          AtA[2]*(Atb[1]*AtA[7] - AtA[4]*Atb[2])) / det;

	grad_y = (AtA[0]*(Atb[1]*AtA[8] - AtA[5]*Atb[2]) -
	          Atb[0]*(AtA[3]*AtA[8] - AtA[5]*AtA[6]) +
	          AtA[2]*(AtA[3]*Atb[2] - Atb[1]*AtA[6])) / det;

	grad_z = (AtA[0]*(AtA[4]*Atb[2] - Atb[1]*AtA[7]) -
	          AtA[1]*(AtA[3]*Atb[2] - Atb[1]*AtA[6]) +
	          Atb[0]*(AtA[3]*AtA[7] - AtA[4]*AtA[6])) / det;

	// Normalize and return gradient direction (with magnitude as 4th element)
	// Note: Diagnostics (2M+ calls) showed norm always > 1e-12
	double norm = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
	std::vector<double> result(4);  // 4 elements: [norm_x, norm_y, norm_z, magnitude]

	result[0] = fabs(grad_x)/norm;
	result[1] = fabs(grad_y)/norm;
	result[2] = fabs(grad_z)/norm;
	result[3] = norm;  // Store the actual gradient magnitude

	return result;
}

/* ----------------------------------------------------------------------
   Find all neighbors with a given spin value and return their average quaternion

   This function searches through all neighbors of a site, finds those with
   the specified spin value, and computes the average of their quaternion
   orientations.

   AVERAGING METHOD: Uses simple arithmetic averaging with hemisphere alignment.
   This is a simplified approach that is computationally efficient but has
   limitations compared to more sophisticated methods like Markley's algorithm
   (eigenvalue-based averaging).

   APPLICABILITY: Simple averaging is acceptable when:
   - Quaternions being averaged differ by < 10 degrees (typical for same-grain neighbors)
   - Computational efficiency is important
   - Use case: post-solidification smoothing within the same grain

   LIMITATIONS: For quaternions with large angular differences (> 10 degrees),
   simple averaging can introduce errors up to ~5 degrees. For higher accuracy
   requirements, consider implementing Markley's algorithm (2007) which uses
   eigenvalue decomposition of a 4x4 accumulator matrix.

   Reference: https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions

   Inputs:
     site - the site index to check neighbors of
     target_spin - the spin value to search for in neighbors

   Returns:
     std::vector<double> containing [q0, qx, qy, qz] - the average quaternion

   ERROR HANDLING:
     If no neighbors with target_spin are found, this is an error condition
     (function should only be called when such neighbors exist) and will
     terminate the simulation with an error message.
------------------------------------------------------------------------- */
std::vector<double> AppAdditiveExtTempTexture::get_average_neighbor_quaternion(int site, int target_spin)
{
    // Initialize storage for first quaternion as reference
    double ref_q0 = 0.0, ref_qx = 0.0, ref_qy = 0.0, ref_qz = 0.0;
    bool first_found = false;
    
    // Initialize sums
    double sum_q0 = 0.0, sum_qx = 0.0, sum_qy = 0.0, sum_qz = 0.0;
    int count = 0;
    
    // Check all neighbors
    for (int j = 0; j < numneigh[site]; j++) {
        int neighbor_site = neighbor[site][j];
        
        // Check if this neighbor has the target spin
        if (spin[neighbor_site] == target_spin) {
            // Get this neighbor's quaternion
            double nq0 = q0[neighbor_site];
            double nqx = qx[neighbor_site];
            double nqy = qy[neighbor_site];
            double nqz = qz[neighbor_site];
            
            if (!first_found) {
                // Store first quaternion as reference
                ref_q0 = nq0;
                ref_qx = nqx;
                ref_qy = nqy;
                ref_qz = nqz;
                first_found = true;

                // Add first quaternion to sum
                sum_q0 += nq0;
                sum_qx += nqx;
                sum_qy += nqy;
                sum_qz += nqz;
                count++;
            } else {
                // Check dot product with reference quaternion
                double dot = ref_q0*nq0 + ref_qx*nqx + ref_qy*nqy + ref_qz*nqz;

                // If dot product is negative, flip the quaternion to same hemisphere
                // q and -q represent the same orientation, so this is valid
                if (dot < 0.0) {
                    nq0 = -nq0;
                    nqx = -nqx;
                    nqy = -nqy;
                    nqz = -nqz;
                }

                // Add this neighbor's quaternion to the sum (possibly flipped)
                sum_q0 += nq0;
                sum_qx += nqx;
                sum_qy += nqy;
                sum_qz += nqz;
                count++;
            }
        }
    }
    
    // If no neighbors with target spin found, this is an error
    // This function should only be called when such neighbors exist
    if (count == 0) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                "get_average_neighbor_quaternion: No neighbors found with target spin %d at site %d. "
                "This function should only be called when neighbors with the target spin exist.",
                target_spin, site);
        error->all(FLERR, error_msg);
    }
    
    // Compute average
    double avg_q0 = sum_q0 / count;
    double avg_qx = sum_qx / count;
    double avg_qy = sum_qy / count;
    double avg_qz = sum_qz / count;
    
    // Normalize the average quaternion to ensure unit length
    double norm = sqrt(avg_q0*avg_q0 + avg_qx*avg_qx + avg_qy*avg_qy + avg_qz*avg_qz);
    if (norm > 0.0) {
        avg_q0 /= norm;
        avg_qx /= norm;
        avg_qy /= norm;
        avg_qz /= norm;
    }
    
    return std::vector<double>{avg_q0, avg_qx, avg_qy, avg_qz};
}

/* ----------------------------------------------------------------------
   Figure out the dot product between the temperature gradient and the x'tal orientation.
   We'll use this to change the Q-value at the lattice site...

   Inputs: lattice site :: site
   Outputs: double of max dot product
------------------------------------------------------------------------- */
double AppAdditiveExtTempTexture::melt_misorientation(int site, double c1, double c2, double c3)
{
	double dotMax;
	double theta;
	double mobOut;

	//Calculate the unit-vector surface norm (use voxel counting)
	vector<double> grad_full = normal_finder(site);  // Returns [norm_x, norm_y, norm_z, magnitude]
	vector<double> grad_vector = {grad_full[0], grad_full[1], grad_full[2]};  // Extract direction only
	
	// Create quaternion vector from site's orientation
	vector<double> q_site = {q0[site], qx[site], qy[site], qz[site]};
	
	// Use quaternion function to get cosine of minimum angle between
	// crystal orientation and temperature gradient
	dotMax = quaternion::get_cosine_of_minumum_angle_between_q_and_u(q_site, grad_vector);
	
	// Calculate angle from cosine
	theta = acos(dotMax);
	
	// Apply texture mobility model
	mobOut = c1 + c2 * cos(c3 * theta);
	
	return mobOut;
}

/* ----------------------------------------------------------------------
   This function is called after a site is solidified and nsmooth > 0. Its purpose is to smooth grain boundaries.
   It does this by using the traditional Potts model without including grain orientations. Quaternions are handled
   by switching the site to the average quaternion value of neighbors with the selected spin/grain ID.

   Inputs: lattice site :: site
   Outputs: None
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::smooth_site(int i) {

  //Adapt code from potts_neigh_only site_event_rejection.
  int oldstate = spin[i];
  double einitial = site_energy_smooth(i);

  // events = spin flips to neighboring site different than self

  int j,m,value;
  int nevent = 0;

  for (j = 0; j < numneigh[i]; j++) {
    value = spin[neighbor[i][j]];
    if (value == spin[i]) continue;
    for (m = 0; m < nevent; m++)
      if (value == unique[m]) break;
    if (m < nevent) continue;
    unique[nevent++] = value;
  }

  if (nevent == 0) return;

  double efinal;
  if (smooth_greedy_multiproposal_enabled) {
    // Steepest-descent proposal: evaluate every distinct neighbor spin
    // and pick the energy-minimizing one. First-encountered tie-break
    // keeps the choice deterministic under MPI.
    int best_spin = unique[0];
    spin[i] = unique[0];
    double best_energy = site_energy_smooth(i);
    for (int k = 1; k < nevent; k++) {
      spin[i] = unique[k];
      double e = site_energy_smooth(i);
      if (e < best_energy) {
        best_energy = e;
        best_spin = unique[k];
      }
    }
    spin[i] = best_spin;
    efinal = best_energy;
  } else {
    int iran = (int) (nevent*ranapp->uniform());
    if (iran >= nevent) iran = nevent-1;
    spin[i] = unique[iran];
    efinal = site_energy_smooth(i);
  }

  // accept or reject via Boltzmann criterion

  if (efinal <= einitial) {
  } else if (temperature == 0.0) {
    spin[i] = oldstate;
  } else if (ranapp->uniform() > exp((einitial-efinal)*t_inverse)) {
    spin[i] = oldstate;
  }

  //If the spin flips successfully, also change the rest of the values
  if (spin[i] != oldstate) {

    vector<double> q_new = get_average_neighbor_quaternion(i, spin[i]);
    q0[i] = q_new[0];
    qx[i] = q_new[1];
    qy[i] = q_new[2];
    qz[i] = q_new[3];
    naccept++;
  }
  // set mask if site could not have changed
  // if site changed, unset mask of sites with affected propensity
  // OK to change mask of ghost sites since never used

  if (Lmask) {
    if (einitial < 0.5*numneigh[i]) mask[i] = 1;
    if (spin[i] != oldstate)
      for (int j = 0; j < numneigh[i]; j++)
	mask[neighbor[i][j]] = 0;
  }
}

/* ----------------------------------------------------------------------
   compute energy of site using only spin values (for boundary smoothing after solidification)
------------------------------------------------------------------------- */

double AppAdditiveExtTempTexture::site_energy_smooth(int i)
{
  int isite = spin[i];
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    // NB: must check neighbor's active_flag, not center site's (bug fix: was active_flag[i])
    if(active_flag[neighbor[i][j]] != 3) continue;
    if (isite != spin[neighbor[i][j]]) eng++;
  }
  return (double) eng;
}

/* ----------------------------------------------------------------------
   Post-smoothing single-voxel grain cleanup. Returns true iff the voxel
   was resolved (flipped, or confirmed not a 1-voxel grain) this call;
   returns false to request a deferral when the 26-neighbor first shell
   is not yet fully solidified.

   A grain in this app is 26-connected through the Potts energy
   neighborhood, so a site with no same-spin 26-neighbor is by
   definition a 1-voxel grain. If such a voxel is found, it is flipped
   to the candidate spin that minimizes site_energy_smooth(i) (tie-break
   by first-encountered neighbor order → deterministic under MPI). The
   quaternion is updated via get_average_neighbor_quaternion() to adopt
   the chosen grain's local orientation. The caller is responsible for
   marking solid_d to suppress re-firing.
------------------------------------------------------------------------- */
bool AppAdditiveExtTempTexture::flip_single_voxel_grain(int i)
{
  int oldstate = spin[i];
  int nunique = 0;
  bool has_same_spin_neighbor = false;

  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    if (active_flag[nj] != 3) return false;  // defer: neighborhood immature
    int s = spin[nj];
    if (s == oldstate) {
      has_same_spin_neighbor = true;
      continue;
    }
    int m;
    for (m = 0; m < nunique; m++) if (s == unique[m]) break;
    if (m == nunique) unique[nunique++] = s;
  }

  // Fully mature neighborhood: decide.
  if (has_same_spin_neighbor) return true;   // not a 1-voxel grain
  if (nunique == 0)           return true;   // no solidified neighbors at all

  // Energy-minimizing pick.
  int best_spin = unique[0];
  spin[i] = unique[0];
  double best_energy = site_energy_smooth(i);
  for (int k = 1; k < nunique; k++) {
    spin[i] = unique[k];
    double e = site_energy_smooth(i);
    if (e < best_energy) {
      best_energy = e;
      best_spin = unique[k];
    }
  }
  spin[i] = best_spin;

  vector<double> q_new = get_average_neighbor_quaternion(i, best_spin);
  q0[i] = q_new[0];
  qx[i] = q_new[1];
  qy[i] = q_new[2];
  qz[i] = q_new[3];

  n_single_voxel_flips++;
  naccept++;
  return true;
}

/* ----------------------------------------------------------------------
    The first version of this just initialized critical nucleation temperatures.
    This version will also initialize nucleii size (starting with a normal dist)
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::nucleation_init() {
    std::normal_distribution<> dist_T{tc,tsig};
    std::normal_distribution<> dist_S{size_norm,size_sig};
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // Randomly assign critical temperature to every spin
    for(int i = 0; i < nspins; i++) {
        nucleation_temps[i] = dist_T(gen);
        nucleation_sizes[i] = dist_S(gen);
    }
}

/* ----------------------------------------------------------------------
   Check if site is eligible to be activated as powder

   Criteria:
   - Site must not be void (active_flag == 5)
   - Site must not be molten or solid (active_flag >= 2)
   - Site is eligible if:
     1. Temperature >= solidus (in thermal influence zone)
     2. Has molten or solidified neighbor (adjacent to meltpool)
     3. At bottom boundary (substrate)
     4. Below meltpool surface (has active material above)
------------------------------------------------------------------------- */
bool AppAdditiveExtTempTexture::is_powder_eligible_site(int i) {
  // Skip voids - they never participate
  if (active_flag[i] == 5) return false;

  // Skip already molten or solid sites
  if (active_flag[i] >= 2) return false;

  // Criterion 1: Hot enough (in thermal influence zone)
  if (T[i] >= ts) return true;

  // Criterion 2: Has solid or molten neighbor (adjacent to meltpool)
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    if (active_flag[nj] == 2 || active_flag[nj] == 3) {
      return true;  // Adjacent to meltpool/solidified region
    }
  }

  // Criterion 3: At bottom boundary (substrate layer)
  double site_z = xyz[i][2];
  if (site_z <= domain->boxzlo + dx * 0.5) return true;

  // Criterion 4: Below meltpool surface (has active material above)
  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    double neighbor_z = xyz[nj][2];

    // Check if neighbor is above and is molten or solid
    if (neighbor_z > site_z && active_flag[nj] >= 2) {
      return true;  // Material exists above this site
    }
  }

  return false;
}

/* ----------------------------------------------------------------------
   Activate powder sites that are at or below the meltpool surface

   Strategy: Find the maximum z-coordinate of all molten sites (meltpool top),
   then set all inactive sites below that level to powder (active_flag = 1).

   This function should be called:
   - After fast forwards > 1 second (new layer deposition)
   - During app_update when meltpool is present
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::activate_powder_sites() {
  // Find local maximum z-coordinate of molten sites
  double local_max_molten_z = -1e100;  // Very negative initial value
  bool has_molten_sites = false;

  for (int i = 0; i < nlocal; i++) {
    if (active_flag[i] == 2) {  // Molten site
      has_molten_sites = true;
      double site_z = xyz[i][2];
      if (site_z > local_max_molten_z) {
        local_max_molten_z = site_z;
      }
    }
  }

  // Find global maximum z-coordinate of molten sites across all ranks
  double global_max_molten_z = -1e100;
  MPI_Allreduce(&local_max_molten_z, &global_max_molten_z, 1, MPI_DOUBLE, MPI_MAX, world);

  // If no meltpool exists anywhere, don't activate powder
  if (global_max_molten_z <= -1e99) {
    return;
  }

  // Activate all inactive sites at or below meltpool top
  int powder_activated = 0;
  for (int i = 0; i < nlocal; i++) {
    if (active_flag[i] == 0) {  // Inactive site
      double site_z = xyz[i][2];
      if (site_z <= global_max_molten_z) {
        active_flag[i] = 1;  // Set to POWDER state
        powder_activated++;
      }
    }
  }

  // Sum across all MPI ranks
  int global_powder_activated = 0;
  MPI_Allreduce(&powder_activated, &global_powder_activated, 1, MPI_INT, MPI_SUM, world);

  // Report activation (rank 0 only)
  if (domain->me == 0 && global_powder_activated > 0) {
    fprintf(screen, "Powder activation at t=%.6e s: activated %d sites (meltpool top at z=%.3f)\n",
            time, global_powder_activated, global_max_molten_z);
  }
}

/* ----------------------------------------------------------------------
   Generate spherical voids in the domain based on volumetric density
   and size distribution. Voids are represented by activeFlag = 5.
   Generation happens on rank 0, then void list is broadcast to all ranks.
------------------------------------------------------------------------- */
void AppAdditiveExtTempTexture::generate_voids(RandomPark *random) {
    timer->stamp();

    if (!enable_voids) {
        timer->stamp(TIME_APP);
        return;
    }

    // Calculate domain volume and number of voids
    double domain_volume = 0.0;
    int num_voids = 0;

    if (domain->me == 0) {
        // Get global domain bounds
        double xlen = domain->boxxhi - domain->boxxlo;
        double ylen = domain->boxyhi - domain->boxylo;
        double zlen = domain->boxzhi - domain->boxzlo;
        domain_volume = xlen * ylen * zlen;  // in m^3

        // If pore fraction was specified, calculate void_density from it
        if (void_pore_fraction > 0.0) {
            double r_mean_m = void_radius_mean * 1e-6;  // Convert μm to m
            double avg_void_volume = (4.0/3.0) * M_PI * pow(r_mean_m, 3);
            void_density = void_pore_fraction / avg_void_volume;

            fprintf(screen, "Target pore fraction: %.4f (%.2f%%)\n",
                    void_pore_fraction, void_pore_fraction * 100.0);
            fprintf(screen, "Calculated void density: %.3e voids/m^3\n", void_density);
        }

        // Calculate expected number of voids
        num_voids = static_cast<int>(void_density * domain_volume + 0.5);

        if (num_voids > 0) {
            fprintf(screen, "Generating %d voids in domain volume %.3e m^3\n",
                    num_voids, domain_volume);
            fprintf(screen, "Void size distribution: mean=%.1f um, std=%.1f um, min=%.1f um, max=%.1f um\n",
                    void_radius_mean, void_radius_std, void_radius_min, void_radius_max);
        }
    }

    // Broadcast number of voids to all ranks
    MPI_Bcast(&num_voids, 1, MPI_INT, 0, world);

    if (num_voids == 0) {
        timer->stamp(TIME_APP);
        return;
    }

    // Generate voids on rank 0
    if (domain->me == 0) {
        voids.clear();
        voids.reserve(num_voids);

        // Setup random number generator for Gaussian distribution
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<double> radius_dist(void_radius_mean, void_radius_std);

        // Convert micrometers to meters for radius bounds
        double r_min_m = void_radius_min * 1e-6;
        double r_max_m = void_radius_max * 1e-6;

        int max_placement_attempts = 1000;
        int successful_placements = 0;
        int total_attempts = 0;

        for (int ivoid = 0; ivoid < num_voids; ivoid++) {
            bool placed = false;
            int attempts = 0;

            while (!placed && attempts < max_placement_attempts) {
                attempts++;
                total_attempts++;

                // Sample radius from truncated Gaussian (convert um to m)
                double radius_m;
                do {
                    radius_m = radius_dist(gen) * 1e-6;
                } while (radius_m < r_min_m || radius_m > r_max_m);

                // Pick random center location in domain
                double x = domain->boxxlo + random->uniform() * (domain->boxxhi - domain->boxxlo);
                double y = domain->boxylo + random->uniform() * (domain->boxyhi - domain->boxylo);
                double z = domain->boxzlo + random->uniform() * (domain->boxzhi - domain->boxzlo);

                // Check for collision with existing voids
                bool collides = false;
                for (const auto& existing_void : voids) {
                    double dx_v = x - existing_void.x;
                    double dy_v = y - existing_void.y;
                    double dz_v = z - existing_void.z;
                    double dist = sqrt(dx_v*dx_v + dy_v*dy_v + dz_v*dz_v);
                    double min_dist = radius_m + existing_void.radius;

                    if (dist < min_dist) {
                        collides = true;
                        break;
                    }
                }

                if (!collides) {
                    voids.emplace_back(x, y, z, radius_m);
                    placed = true;
                    successful_placements++;
                }
            }

            if (!placed) {
                fprintf(screen, "Warning: Could not place void %d after %d attempts\n",
                        ivoid, max_placement_attempts);
            }
        }

        fprintf(screen, "Successfully placed %d/%d voids (total attempts: %d)\n",
                successful_placements, num_voids, total_attempts);
    }

    // Broadcast void count (may be less than num_voids if placements failed)
    int actual_void_count = voids.size();
    MPI_Bcast(&actual_void_count, 1, MPI_INT, 0, world);

    // Prepare void data for broadcast (x, y, z, radius for each void)
    std::vector<double> void_data;
    if (domain->me == 0) {
        void_data.resize(actual_void_count * 4);
        for (int i = 0; i < actual_void_count; i++) {
            void_data[i*4 + 0] = voids[i].x;
            void_data[i*4 + 1] = voids[i].y;
            void_data[i*4 + 2] = voids[i].z;
            void_data[i*4 + 3] = voids[i].radius;
        }
    } else {
        void_data.resize(actual_void_count * 4);
    }

    // Broadcast void data to all ranks
    if (actual_void_count > 0) {
        MPI_Bcast(void_data.data(), actual_void_count * 4, MPI_DOUBLE, 0, world);

        // Reconstruct void list on other ranks
        if (domain->me != 0) {
            voids.clear();
            voids.reserve(actual_void_count);
            for (int i = 0; i < actual_void_count; i++) {
                voids.emplace_back(void_data[i*4 + 0], void_data[i*4 + 1],
                                   void_data[i*4 + 2], void_data[i*4 + 3]);
            }
        }
    }

    // Each rank sets activeFlag = 5 for sites within voids
    int sites_in_voids = 0;
    for (int i = 0; i < nlocal; i++) {
        // Get site coordinates
        double site_x = xyz[i][0];
        double site_y = xyz[i][1];
        double site_z = xyz[i][2];

        // Check if site is inside any void
        for (const auto& void_obj : voids) {
            double dx_s = site_x - void_obj.x;
            double dy_s = site_y - void_obj.y;
            double dz_s = site_z - void_obj.z;
            double dist_sq = dx_s*dx_s + dy_s*dy_s + dz_s*dz_s;

            if (dist_sq <= void_obj.radius * void_obj.radius) {
                // Site is inside this void
                active_flag[i] = 5;
                sites_in_voids++;
                break;  // No need to check other voids
            }
        }
    }

    // Report statistics
    int global_sites_in_voids = 0;
    MPI_Reduce(&sites_in_voids, &global_sites_in_voids, 1, MPI_INT, MPI_SUM, 0, world);

    if (domain->me == 0 && actual_void_count > 0) {
        fprintf(screen, "Total sites marked as voids: %d\n", global_sites_in_voids);

        // Post-generation verification: calculate actual pore fraction
        double total_void_volume = 0.0;
        for (const auto& v : voids) {
            total_void_volume += (4.0/3.0) * M_PI * pow(v.radius, 3);
        }

        double xlen = domain->boxxhi - domain->boxxlo;
        double ylen = domain->boxyhi - domain->boxylo;
        double zlen = domain->boxzhi - domain->boxzlo;
        double domain_vol = xlen * ylen * zlen;

        double actual_pore_fraction = total_void_volume / domain_vol;

        fprintf(screen, "\nVoid generation verification:\n");
        fprintf(screen, "  Actual pore fraction: %.4f (%.2f%%)\n",
                actual_pore_fraction, actual_pore_fraction * 100.0);

        if (void_pore_fraction > 0.0) {
            double error_pct = 100.0 * (actual_pore_fraction - void_pore_fraction) / void_pore_fraction;
            fprintf(screen, "  Target pore fraction: %.4f (%.2f%%)\n",
                    void_pore_fraction, void_pore_fraction * 100.0);
            fprintf(screen, "  Relative error: %.2f%%\n", error_pct);
        }
        fprintf(screen, "\n");
    }

    timer->stamp(TIME_APP);
}

/* ----------------------------------------------------------------------
   Setup modular temperature source command
   Format: setup_temperature_source type [arguments...]
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::setup_temperature_source_cmd(int narg, char **arg)
{
  if (narg < 1) {
    error->all(FLERR,"Illegal setup_temperature_source command: must specify type");
  }
  
  std::string source_type = arg[0];
  std::vector<std::string> source_args;
  
  // Convert remaining arguments to string vector
  for (int i = 1; i < narg; i++) {
    source_args.push_back(std::string(arg[i]));
  }
  
  // Create temperature source using factory
  temperature_source = SPPARKS_NS::create_temperature_source(source_type, spk);
  if (!temperature_source) {
    error->all(FLERR,"Failed to create temperature source");
  }

  // Pass optimization flags to temperature source before setup
  HDF5UnstructuredTemperatureSource* hdf5_source =
    dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
  if (hdf5_source) {
    hdf5_source->set_use_spatial_grid(opt_use_spatial_grid);
    hdf5_source->set_use_element_cache(opt_use_element_cache);
  }

  // Setup the temperature source with provided arguments
  temperature_source->setup_temperature_source(source_args);

  // Enable the new temperature source system
  use_temperature_source = true;

  if (domain->me == 0) {
    std::cout << "Modular temperature source (" << source_type << ") enabled" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_path command: define a single linear scan, optionally
   repeated N times. Coordinates and velocity are SI (meters, m/s).

   Format:
     laser_path start <X0> <Y0> <Z0> end <X1> <Y1> speed <V> [repeats <N>]

   Builds N copies of one RASTER::Path into scan_layer; scan_layer_z is
   set to Z0. The temperature sources are path-agnostic; this layer is
   consumed by update_temperature_from_source() each step (Rosenthal),
   or translated into a GREENAM LaserScan via build_scan() (Moser).
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::laser_path_cmd(int narg, char **arg)
{
  // Expected token layout (0-based, command name already stripped):
  //   start X0 Y0 Z0 end X1 Y1 speed V [repeats N]
  // Minimum is 9 tokens (no repeats); with repeats it's 11.
  if (narg < 9)
    error->all(FLERR,"Illegal laser_path command: expected start X0 Y0 Z0 end X1 Y1 speed V [repeats N]");
  if (strcmp(arg[0],"start") != 0)
    error->all(FLERR,"laser_path: expected keyword 'start'");
  const double x0 = atof(arg[1]);
  const double y0 = atof(arg[2]);
  const double z0 = atof(arg[3]);
  if (strcmp(arg[4],"end") != 0)
    error->all(FLERR,"laser_path: expected keyword 'end' after start coordinates");
  const double x1 = atof(arg[5]);
  const double y1 = atof(arg[6]);
  if (strcmp(arg[7],"speed") != 0)
    error->all(FLERR,"laser_path: expected keyword 'speed' after end coordinates");
  const double v = atof(arg[8]);
  if (v <= 0.0)
    error->all(FLERR,"laser_path: speed must be > 0");

  int repeats = 1;
  if (narg > 9) {
    if (strcmp(arg[9],"repeats") != 0)
      error->all(FLERR,"laser_path: expected keyword 'repeats' or end-of-command");
    if (narg < 11)
      error->all(FLERR,"laser_path: missing value after 'repeats'");
    repeats = atoi(arg[10]);
    if (repeats < 1)
      error->all(FLERR,"laser_path: repeats must be >= 1");
  }

  // Build N identical paths in the layer.
  RASTER::Point a(x0,y0,0.0), b(x1,y1,0.0);
  std::vector<RASTER::Path> paths;
  paths.reserve(repeats);
  for (int i = 0; i < repeats; ++i) paths.emplace_back(a, b, v);

  scan_layer = RASTER::Layer(paths, /*thickness*/0);
  scan_layer_z = z0;
  scan_layer_time = time;  // anchor the layer pose to the current sim time
  scan_layer_active = true;
  laser_path_set = true;

  // Size the singularity cutoff to the lattice so the 1/R prefactor at
  // the laser site stays finite. Only meaningful when the source is
  // already a Rosenthal source.
  if (auto* ros = dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get())) {
    ros->set_r_min(0.5 * dx);
  }

  // If the active source is Moser-Green, hand the path geometry over so
  // it can build its own GREENAM LaserScan/ScanIntegration. The Moser
  // source is path-aware and consumes simulation time + world-space
  // coordinates directly; it does not use scan_layer.
  if (auto* moser = dynamic_cast<MoserGreenTemperatureSource*>(temperature_source.get())) {
    moser->build_scan(time, x0, y0, x1, y1, z0, v, repeats);
  }

  if (domain->me == 0) {
    std::cout << "laser_path: (" << x0 << "," << y0 << "," << z0
              << ") -> (" << x1 << "," << y1 << "," << z0 << ")"
              << " speed=" << v << " m/s, repeats=" << repeats << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_fluctuations command. Currently supports the in-source PSD
   form only (file-loaded fluctuations are not yet implemented for the
   Moser source):

     laser_fluctuations psd <shape> sigma_W <s> sigma_D <s> \
                            [sigma_P <s>] [rho <r>] [seed <n>] [dt <s>] \
                            [shape-specific keys ...]
   Shapes:
     white
     lorentzian   tau <s>
     pink
     narrow_band  f0 <Hz>  df <Hz>

   Stores the spec on the active Moser source. The fluctuation table
   is materialized inside laser_path_cmd's call to build_scan(),
   because that's when the scan duration becomes known. Therefore the
   correct command order is:

     setup_temperature_source moser ...
     laser_fluctuations psd ...
     laser_path start ... end ... speed ...

   The active source must be the Moser/Green's-function source; the
   Rosenthal source no longer supports fluctuations (the steady-state
   kernel "snaps the entire pool" semantics is unphysical, and the
   Moser unsteady integral is the proper way to apply per-emission
   parameter variation).
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::laser_fluctuations_cmd(int narg, char **arg)
{
  if (narg < 1)
    error->all(FLERR,"Illegal laser_fluctuations command: expected psd <shape> ...");

  if (!temperature_source) {
    error->all(FLERR,"laser_fluctuations: setup_temperature_source must be called first");
  }
  MoserGreenTemperatureSource *moser =
    dynamic_cast<MoserGreenTemperatureSource*>(temperature_source.get());
  if (!moser) {
    error->all(FLERR,"laser_fluctuations: only the Moser/Green's-function source supports fluctuations");
  }

  if (strcmp(arg[0],"psd") != 0)
    error->all(FLERR,"laser_fluctuations: expected 'psd' subcommand");
  if (narg < 2)
    error->all(FLERR,"laser_fluctuations psd: missing <shape>");

  MoserGreenTemperatureSource::PsdSpec spec;
  spec.sigma_W = 0.0;
  spec.sigma_D = 0.0;
  spec.sigma_P = 0.0;
  spec.rho     = 0.0;
  spec.seed    = 12345UL;
  spec.dt_psd  = 5.0e-6;
  spec.tau     = 0.0;
  spec.f0      = 0.0;
  spec.df      = 0.0;

  if      (strcmp(arg[1],"white")       == 0) spec.shape = MoserGreenTemperatureSource::PsdShape::WHITE;
  else if (strcmp(arg[1],"lorentzian")  == 0) spec.shape = MoserGreenTemperatureSource::PsdShape::LORENTZIAN;
  else if (strcmp(arg[1],"pink")        == 0) spec.shape = MoserGreenTemperatureSource::PsdShape::PINK;
  else if (strcmp(arg[1],"narrow_band") == 0) spec.shape = MoserGreenTemperatureSource::PsdShape::NARROW_BAND;
  else error->all(FLERR,"laser_fluctuations psd: unknown shape (use white|lorentzian|pink|narrow_band)");

  int i = 2;
  while (i < narg) {
    const char *k = arg[i];
    if (i + 1 >= narg)
      error->all(FLERR,"laser_fluctuations psd: missing value for keyword");
    const double val = atof(arg[i+1]);
    if      (strcmp(k,"sigma_W") == 0) spec.sigma_W = val;
    else if (strcmp(k,"sigma_D") == 0) spec.sigma_D = val;
    else if (strcmp(k,"sigma_P") == 0) spec.sigma_P = val;
    else if (strcmp(k,"rho")     == 0) spec.rho     = val;
    else if (strcmp(k,"seed")    == 0) spec.seed    = static_cast<unsigned long>(atol(arg[i+1]));
    else if (strcmp(k,"dt")      == 0) spec.dt_psd  = val;
    else if (strcmp(k,"tau")     == 0) spec.tau     = val;
    else if (strcmp(k,"f0")      == 0) spec.f0      = val;
    else if (strcmp(k,"df")      == 0) spec.df      = val;
    else error->all(FLERR,"laser_fluctuations psd: unknown keyword");
    i += 2;
  }

  moser->set_psd_spec(spec);
  if (domain->me == 0) {
    const char *sname = "white";
    if      (spec.shape == MoserGreenTemperatureSource::PsdShape::LORENTZIAN)  sname = "lorentzian";
    else if (spec.shape == MoserGreenTemperatureSource::PsdShape::PINK)        sname = "pink";
    else if (spec.shape == MoserGreenTemperatureSource::PsdShape::NARROW_BAND) sname = "narrow_band";
    std::cout << "laser_fluctuations psd: shape=" << sname
              << " sigma_W=" << spec.sigma_W
              << " sigma_D=" << spec.sigma_D
              << " sigma_P=" << spec.sigma_P
              << " rho="     << spec.rho
              << " seed="    << spec.seed
              << " dt="      << spec.dt_psd << " s";
    if (spec.shape == MoserGreenTemperatureSource::PsdShape::LORENTZIAN)
      std::cout << " tau=" << spec.tau;
    if (spec.shape == MoserGreenTemperatureSource::PsdShape::NARROW_BAND)
      std::cout << " f0=" << spec.f0 << " df=" << spec.df;
    std::cout << std::endl;
  }
}

/* ----------------------------------------------------------------------
   Fast-forward predictor for the Rosenthal source.

   Walks a copy of scan_layer forward in time and returns the earliest
   time at which the conservative upper bound on the Rosenthal
   temperature on the local domain reaches threshold_temp.

   The upper bound is computed by:
     1. Finding the closest world-space point on the local bounding box
        to the current laser position. Distance to this point is the
        smallest possible R for any site in the local domain.
     2. Calling RosenthalTemperatureSource::rosenthal_peak_at_distance(R)
        which returns max(rosenthal_pointwise) over all (xi, y_rel, z_rel)
        with R fixed (exponent <= 1, line-source weights summed).

   Because this bound is provably >= the true T at any site in the box,
   crossing it cannot skip a real heating event. The predictor may
   under-estimate the next active time (i.e. wake up early), which is
   the safe direction.
------------------------------------------------------------------------- */

double AppAdditiveExtTempTexture::rosenthal_next_active_time(double current_time, double threshold_temp)
{
  if (!laser_path_set || !scan_layer_active)
    return std::numeric_limits<double>::max();

  auto* ros = dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());
  if (!ros) return std::numeric_limits<double>::max();

  // Local domain bounding box in physical (meters) coordinates.
  if (nlocal == 0) return std::numeric_limits<double>::max();
  double xmin = xyz[0][0]*dx, xmax = xmin;
  double ymin = xyz[0][1]*dx, ymax = ymin;
  double zmin = xyz[0][2]*dx, zmax = zmin;
  for (int i = 1; i < nlocal; ++i) {
    const double sx = xyz[i][0]*dx;
    const double sy = xyz[i][1]*dx;
    const double sz = xyz[i][2]*dx;
    if (sx < xmin) xmin = sx; else if (sx > xmax) xmax = sx;
    if (sy < ymin) ymin = sy; else if (sy > ymax) ymax = sy;
    if (sz < zmin) zmin = sz; else if (sz > zmax) zmax = sz;
  }

  const double T0 = ros->get_ambient_temperature();
  // Cheap early-out: if even the unmodulated peak at r_min can't reach
  // the threshold, no laser position ever will. Use r_min as the
  // tightest possible R_min (set by the app to 0.5*dx).
  // (rosenthal_peak_at_distance clamps R to r_min internally.)
  {
    const double T_max_possible = ros->rosenthal_peak_at_distance(0.0, T0);
    if (T_max_possible < threshold_temp) {
      return std::numeric_limits<double>::max();
    }
  }

  // Walk a copy of the layer forward in time using the search window.
  RASTER::Layer probe = scan_layer;
  const double ddt = (fast_forward_search_window > 0.0)
                       ? fast_forward_search_window : 0.01;
  // Cap the look-ahead so we don't spin forever on a never-active path.
  const double max_lookahead = 3600.0; // 1 hour of sim time
  double t = current_time;
  const double t_stop = current_time + max_lookahead;

  while (t < t_stop) {
    // Closest point on the local box to the current laser position.
    // Laser z is the scan plane (scan_layer_z); xy come from the probe.
    const RASTER::Point laser = probe.get_position();
    const double lx = laser[0];
    const double ly = laser[1];
    const double lz = scan_layer_z;
    const double cx = (lx < xmin) ? xmin : (lx > xmax ? xmax : lx);
    const double cy = (ly < ymin) ? ymin : (ly > ymax ? ymax : ly);
    const double cz = (lz < zmin) ? zmin : (lz > zmax ? zmax : lz);
    const double dxr = cx - lx;
    const double dyr = cy - ly;
    const double dzr = cz - lz;
    const double R_min_box = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);

    const double T_upper = ros->rosenthal_peak_at_distance(R_min_box, T0);
    if (T_upper >= threshold_temp) return t;

    // Advance the probe by ddt; stop if path is exhausted.
    if (!probe.move(ddt)) return std::numeric_limits<double>::max();
    t += ddt;
  }
  return std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   Update temperature from modular source
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::update_temperature_from_source(double simulation_time)
{
  timer->stamp();

  if (!use_temperature_source || !temperature_source) {
    timer->stamp(TIME_APP);
    return; // No temperature source configured
  }

  // Update temperature source (may load new data)
  temperature_source->update_temperatures(dt, simulation_time);

  // Cache the dynamic_cast results outside the loop — these casts are
  // expensive for subclasses (CSR) due to RTTI string comparisons.
  HDF5UnstructuredTemperatureSource* hdf5_source =
    dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
  RosenthalTemperatureSource* ros_source =
    hdf5_source ? nullptr
                : dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());

  // -------------------------------------------------------------------
  // HDF5 path: three optimization tiers, unchanged from prior behavior.
  // -------------------------------------------------------------------
  if (hdf5_source && opt_use_element_cache && opt_use_nodal_precompute) {
    // Path 3: Fast path with nodal precomputation
    double t0 = MPI_Wtime();
    hdf5_source->prepare_for_timestep(simulation_time);
    double t1 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      T[i] = hdf5_source->get_temperature_at_site_fast(i);
    }
    double t2 = MPI_Wtime();

    double prepare_time = t1 - t0;
    if (prepare_time > 1.0) {
      g_t_cache_build += prepare_time;
      g_cache_was_built = true;
    } else {
      g_t_temp_prepare += prepare_time;
      g_cache_was_built = false;
    }
    g_t_temp_site_loop += (t2 - t1);

  } else if (hdf5_source && opt_use_element_cache) {
    // Path 2: Lazy per-site lookup with element cache (no nodal precomputation)
    double t0 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      T[i] = hdf5_source->get_temperature_at_site(i, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  } else if (hdf5_source) {
    // Path 1: XYZ-based lookup (no element cache)
    double t0 = MPI_Wtime();
    const double src_dx = hdf5_source->get_dx();
    for (int i = 0; i < nlocal; i++) {
      double x_physical = xyz[i][0] * src_dx;
      double y_physical = xyz[i][1] * src_dx;
      double z_physical = xyz[i][2] * src_dx;
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  // -------------------------------------------------------------------
  // Rosenthal path: analytical evaluation in pool-local frame.
  // The laser position is advanced by scan_layer.move(dt) once per call;
  // each site is then evaluated against the current pool.
  // -------------------------------------------------------------------
  } else if (ros_source) {
    if (!laser_path_set) {
      error->all(FLERR,"Rosenthal source selected but no laser_path defined");
    }

    double t0 = MPI_Wtime();

    // Sync the laser pose to simulation_time. We may be called with a
    // simulation_time that is well ahead of scan_layer_time -- e.g. just
    // after a fast-forward jump -- so advance in small dt-sized chunks.
    // RASTER::Layer::move(dt) does not carry remainder across path
    // boundaries, so taking many small steps is also the safe way to
    // step across multiple repeats of a single path.
    if (scan_layer_active) {
      double advance = simulation_time - scan_layer_time;
      // tolerance: ignore tiny negative drift from FP arithmetic
      const double step = (dt > 0.0) ? dt : 1.0e-6;
      while (advance > step) {
        if (!scan_layer.move(step)) { scan_layer_active = false; break; }
        advance -= step;
      }
      if (scan_layer_active && advance > 0.0) {
        if (!scan_layer.move(advance)) scan_layer_active = false;
      }
    }
    scan_layer_time = simulation_time;

    const double v_laser = scan_layer.get_speed();
    // Use the source's own preheat/ambient (Rosenthal T0), not the app's
    // t_room — they may differ (e.g. T0=573 K with t_room=300 K).
    const double T0 = ros_source->get_ambient_temperature();
    for (int i = 0; i < nlocal; i++) {
      double phys[3] = { xyz[i][0]*dx, xyz[i][1]*dx, xyz[i][2]*dx };
      RASTER::Point loc =
        scan_layer.compute_position_relative_to_pool(phys, scan_layer_z);
      T[i] = scan_layer_active
               ? ros_source->rosenthal_pointwise(loc[0], loc[1], loc[2], v_laser, T0)
               : T0;
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);

  // -------------------------------------------------------------------
  // Generic xyz fallback for any other future source type.
  // -------------------------------------------------------------------
  } else {
    double t0 = MPI_Wtime();
    for (int i = 0; i < nlocal; i++) {
      double x_physical = xyz[i][0] * dx;
      double y_physical = xyz[i][1] * dx;
      double z_physical = xyz[i][2] * dx;
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
    double t1 = MPI_Wtime();
    g_t_temp_site_loop += (t1 - t0);
  }

  // Optional Gaussian smoothing of the solidification band. Runs only when
  // enabled via `temperature_smooth` in the input script.
  if (temperature_smooth_enabled) apply_temperature_smoothing();

  timer->stamp(TIME_APP);
}

/* ----------------------------------------------------------------------
   Gaussian smoothing of T[] inside the user-specified solidification
   window. Called from update_temperature_from_source() when enabled.

   Algorithm per pass:
     1. Sync T on ghost sites (T is already in ghost_dindices).
     2. For each local site whose T falls inside [tmin-guard, tmax+guard]:
          Tavg = weighted avg of self + lattice neighbors whose T also
                 lies in [tmin-guard, tmax+guard]. Neighbors outside the
                 window are skipped so the kernel cannot bleed across
                 the pool boundary into cold ambient sites.
                 w_j = exp(-|r_j - r_i|^2 / (2*sigma^2))
          blend alpha_eff * Tavg + (1-alpha_eff) * T[i], where alpha_eff
          ramps linearly from 0 at the guard edge to `smooth_alpha` at
          the window edge. This prevents a discontinuity when a site
          crosses in/out of the smoothing window across steps.
     3. Write smoothed values back to T[].

   Pass 2+ re-syncs ghosts so each pass sees the smoothed values from
   the previous one. Cost per step: one extra all_selective per pass.
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::apply_temperature_smoothing()
{
  const double two_sxy2 = 2.0 * smooth_sigma_xy * smooth_sigma_xy;
  const double two_sz2  = 2.0 * smooth_sigma_z  * smooth_sigma_z;
  const double lo_outer = smooth_tmin - smooth_guard;
  const double hi_outer = smooth_tmax + smooth_guard;
  const double ramp_lo = (smooth_guard > 0.0) ? smooth_guard : 1.0;
  const double ramp_hi = (smooth_guard > 0.0) ? smooth_guard : 1.0;

  if ((int)smooth_buffer.size() < nlocal) smooth_buffer.resize(nlocal);

  // Precompute Gaussian weights per 26-neighbor slot. On a uniform SC_26N
  // lattice with constant sigmas, the weight depends only on the neighbor
  // slot (all interior sites share identical lattice offsets). Use any
  // site with a full 26-neighbor complement as reference; fall back to
  // per-pair exp() if none exists on this rank (pathological small domain).
  constexpr int MAX_NBR = 26;
  double w_slot[MAX_NBR];
  bool have_weight_lut = false;
  for (int ref = 0; ref < nlocal && !have_weight_lut; ref++) {
    if (numneigh[ref] < MAX_NBR) continue;
    bool all_valid = true;
    for (int j = 0; j < MAX_NBR; j++) {
      if (neighbor[ref][j] < 0) { all_valid = false; break; }
    }
    if (!all_valid) continue;
    const double xr = xyz[ref][0];
    const double yr = xyz[ref][1];
    const double zr = xyz[ref][2];
    for (int j = 0; j < MAX_NBR; j++) {
      const int nj = neighbor[ref][j];
      const double dxn = xyz[nj][0] - xr;
      const double dyn = xyz[nj][1] - yr;
      const double dzn = xyz[nj][2] - zr;
      const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
      w_slot[j] = exp(-expo);
    }
    have_weight_lut = true;
  }

  for (int pass = 0; pass < smooth_passes; pass++) {
    // Ghosts need current T before each pass.
    comm->all_selective(ghost_iindices, nghost_iarray,
                        ghost_dindices, nghost_darray);

    for (int i = 0; i < nlocal; i++) {
      const double Ti = T[i];
      if (Ti < lo_outer || Ti > hi_outer) {
        smooth_buffer[i] = Ti;
        continue;
      }

      double wsum = 1.0;   // self
      double Tsum = Ti;    // self-contribution

      const int nn = numneigh[i];
      if (have_weight_lut) {
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          // Skip neighbors whose T is outside the smoothing window; they
          // are solid/ambient and would drag the in-band site's average
          // sharply toward cold temperatures.
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double w = w_slot[j];
          wsum += w;
          Tsum += w * Tj;
        }
      } else {
        const double xi = xyz[i][0];
        const double yi = xyz[i][1];
        const double zi = xyz[i][2];
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double dxn = xyz[nj][0] - xi;
          const double dyn = xyz[nj][1] - yi;
          const double dzn = xyz[nj][2] - zi;
          const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
          const double w = exp(-expo);
          wsum += w;
          Tsum += w * Tj;
        }
      }
      const double Tavg = Tsum / wsum;

      // Guard-band taper so the blend factor ramps to zero at the outer edges.
      double aeff = smooth_alpha;
      if (Ti < smooth_tmin) {
        aeff *= (Ti - lo_outer) / ramp_lo;
      } else if (Ti > smooth_tmax) {
        aeff *= (hi_outer - Ti) / ramp_hi;
      }
      if (aeff < 0.0) aeff = 0.0;

      smooth_buffer[i] = (1.0 - aeff) * Ti + aeff * Tavg;
    }

    // Commit this pass before the next one.
    for (int i = 0; i < nlocal; i++) T[i] = smooth_buffer[i];
  }

  // Leave ghosts consistent with the final committed T. Downstream code
  // (phase-transition loops, diagnostic) may depend on this.
  comm->all_selective(ghost_iindices, nghost_iarray,
                      ghost_dindices, nghost_darray);
}

/* ----------------------------------------------------------------------
   In-situ smoothing diagnostic.

   Measures how "clean" the T >= liquidus (tl) isotherm is by counting:
     V = # local sites with T[i] >= tl                           (pool volume)
     S = # of those sites with at least one neighbor T[nj] < tl  (pool surface)

   Reports S / V^(2/3) — a 3D isoperimetric ratio, size-independent.
   Perfect 3D sphere gives (36*pi)^(1/3) ~= 4.836; wiggly/fragmented
   isotherms push higher. Also reports pool volume, surface, mean and max T.

   Assumes T ghosts are current (caller must ensure this).
------------------------------------------------------------------------- */

void AppAdditiveExtTempTexture::compute_smoothing_diagnostics()
{
  bigint V_local = 0, S_local = 0;
  double sumT_local = 0.0;
  double maxT_local = -1.0e30;

  for (int i = 0; i < nlocal; i++) {
    const double Ti = T[i];
    if (Ti < tl) continue;

    V_local++;
    sumT_local += Ti;
    if (Ti > maxT_local) maxT_local = Ti;

    const int nn = numneigh[i];
    for (int j = 0; j < nn; j++) {
      const int nj = neighbor[i][j];
      if (nj < 0) continue;
      if (T[nj] < tl) { S_local++; break; }
    }
  }

  bigint V_global = 0, S_global = 0;
  double sumT_global = 0.0, maxT_global = 0.0;
  MPI_Allreduce(&V_local,    &V_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&S_local,    &S_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&sumT_local, &sumT_global, 1, MPI_DOUBLE,     MPI_SUM, world);
  MPI_Allreduce(&maxT_local, &maxT_global, 1, MPI_DOUBLE,     MPI_MAX, world);

  if (domain->me == 0) {
    const double ideal = cbrt(36.0 * MY_PI);  // ~= 4.8360
    fprintf(screen, "\n=== Smoothing diagnostic (T >= liquidus=%.1f K) ===\n", tl);
    if (V_global > 0) {
      const double V = static_cast<double>(V_global);
      const double ratio = static_cast<double>(S_global) / cbrt(V * V);  // V^(2/3)
      const double meanT = sumT_global / V;
      fprintf(screen,
        "  Pool volume:       " BIGINT_FORMAT " sites\n"
        "  Pool surface:      " BIGINT_FORMAT " sites\n"
        "  S/V^(2/3):         %.3f  (ideal 3D sphere = %.3f)\n"
        "  Pool mean T:       %.2f K   max T: %.2f K\n",
        V_global, S_global, ratio, ideal, meanT, maxT_global);
    } else {
      fprintf(screen, "  No sites above liquidus this step.\n");
    }
  }
}
