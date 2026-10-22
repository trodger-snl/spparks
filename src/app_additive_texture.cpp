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
#include <cstring>
#include "math.h"
#include "math_const.h"
#include "app_additive_texture.h"
#include "grain_walk.h"
#include "comm_lattice.h"
#include "lattice.h"
#include "timer.h"
#include "random_park.h"
#include "random_mars.h"
#include "error.h"
#include "domain.h"
#include "crystalline_orientations/cubic_symmetries.h"
#include "crystalline_orientations/disorientation.h"
#include "crystalline_orientations/hcp_symmetries.h"
#include "crystalline_orientations/quaternion.h"
#include "thermal_manager.h"


using namespace SPPARKS_NS;
using namespace MathConst;

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
  static char app_name[] = "additive/texture";
  static char crystal_type[] = "cubic";
  
  parent_args[0] = app_name;
  parent_args[1] = arg[1]; // nspins
  parent_args[2] = crystal_type; // crystal structure
  
  return parent_args;
}

/* ---------------------------------------------------------------------- */

AppAdditiveTexture::AppAdditiveTexture(SPPARKS *spk, int narg, char **arg) :
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
    ndouble = 10;
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

    nghost_darray = 6;
    int *gd = new int[nghost_darray];
    gd[0] = 0;  // q0
    gd[1] = 1;  // qx
    gd[2] = 2;  // qy
    gd[3] = 3;  // qz
    gd[4] = 5;  // T (needed by normal_finder on ghost neighbors)
    gd[5] = 9;  // spath (d10): a ghost parent's path length feeds the grain walk
    ghost_dindices = gd;
    sites = unique = NULL;
    neigh_dist = NULL;
    nucleation_flags = NULL;
    nucleation_temps = NULL;
    nucleation_sizes = NULL;

    // AdditiveParams holds the shared defaults; the nuclei volumes need dx
    t_cool_max = tl - ts;
    size_norm = pow(dx,3) * 2;
    size_sig = pow(dx,3);
    t_room = 300;
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

    // Temperature-I/O orchestration (source setup/update, laser scan path,
    // pause/reset scheduling, fast-forward predictors, smoothing) lives in
    // the ThermalManager. It initializes its own defaults in its ctor.
    thermal = std::unique_ptr<ThermalManager>(new ThermalManager(spk, &thermal_host_));

    // Initialize powder activation tracking
    last_powder_activation_time = -1.0;  // Force activation at init

    // Single-voxel grain cleanup is off until `single_voxel_cleanup on`
    // is issued in the input script.
    single_voxel_cleanup_enabled = false;
    n_single_voxel_flips = 0;
    smooth_greedy_multiproposal_enabled = false;
    nuclei_smooth_enabled = false;

    //add the double array
    recreate_arrays();
}

/* ---------------------------------------------------------------------- */

AppAdditiveTexture::~AppAdditiveTexture()
{
  // Only delete arrays that are specific to this class
  // Parent class destructors will handle sites, unique, unique_neigh
  if (neigh_dist) delete[] neigh_dist;
  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  if (ghost_iindices) delete [] ghost_iindices;
  if (ghost_dindices) delete [] ghost_dindices;
}

/* ----------------------------------------------------------------------
   input script commands unique to this app
------------------------------------------------------------------------- */

// Populate the ThermalHost snapshot ThermalManager reads/writes. Cheap pointer
// and scalar copies; called before any thermal-> use so the manager sees the
// current lattice state without a typed back-pointer into this app.
void AppAdditiveTexture::refresh_thermal_host()
{
  thermal_host_.T = T;
  thermal_host_.xyz = xyz;
  thermal_host_.numneigh = numneigh;
  thermal_host_.neighbor = neighbor;
  thermal_host_.nlocal = nlocal;
  thermal_host_.dx = dx;
  thermal_host_.tl = tl;
  thermal_host_.ts = ts;
  thermal_host_.t_room = t_room;
  thermal_host_.time = time;
  thermal_host_.dt = dt;
  thermal_host_.comm = comm;
  thermal_host_.ghost_iindices = ghost_iindices;
  thermal_host_.nghost_iarray = nghost_iarray;
  thermal_host_.ghost_dindices = ghost_dindices;
  thermal_host_.nghost_darray = nghost_darray;
}

void AppAdditiveTexture::input_app(char *command, int narg, char **arg)
{
  // Keep the thermal snapshot current before any thermal command handler
  // (e.g. laser_path reads host->time/dx when building the scan path).
  refresh_thermal_host();
  if (parse_command(command, narg, arg, error, screen, domain->me, *thermal)) return;

  if (strcmp(command,"nucleation_mode") == 0) {
    if (narg != 1) error->all(FLERR,"Illegal nucleation_mode command");
    if (strcmp(arg[0],"threshold") == 0) {
      nucleation_mode = NUCLEATION_THRESHOLD;
    } else if (strcmp(arg[0],"continuous") == 0) {
      nucleation_mode = NUCLEATION_CONTINUOUS;
    } else {
      error->all(FLERR,"Illegal nucleation_mode value: use threshold or continuous");
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

  else error->all(FLERR,"Unrecognized command");
}

/* ----------------------------------------------------------------------
  When app_update is called, read in an input file
 ------------------------------------------------------------------------- */
void AppAdditiveTexture::app_update(double dt)
{
  static int step_count = 0;   // cadence for maybe_run_diagnostics
  step_count++;

  // Reset per-call single-voxel cleanup counter
  n_single_voxel_flips = 0;

  // Refresh the ThermalHost snapshot so the manager sees the current lattice
  // pointers, sim time, and dt for this step.
  refresh_thermal_host();

  // Update the temperature field and run fast-forward, delegated to the
  // ThermalManager. advance() fills T[] from the modular source, runs the
  // Moser pause/reset + pause_below scheduling, and predicts idle-period
  // skips. We apply the returned decision here: advance sim time, activate
  // powder after a large skip, or end the run early. The app-level effects
  // (sim time, powder activation) stay in the app.
  if (thermal->enabled()) {
    ThermalManager::ThermalStep s = thermal->advance(time, stoptime);
    if (s.end_run) { time = s.new_time; return; }
    if (s.new_time > time) {
      time = s.new_time;
      if (s.activate_powder) {
        activate_powder_sites();
        last_powder_activation_time = time;
      }
    }
  }

  // Common processing for both temperature systems

  // Track if any melting occurs this timestep
  bool melting_occurred = false;

  //iterate through all the sites for phase transitions (applies to both systems)
  for (int i=0; i<nlocal; i++) {
    // Skip void sites - they never change state
    if (active_flag[i] == 5) continue;

    //Turn the sites on/off depending on the phase data and whether or not the
    //site's temperature has gone above tl. Only do this when melting the first time
    //to avoid repeated initialization
    if( (T[i] >= tl) && active_flag[i] != 2) {
        melting_occurred = true;
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
        spath[i] = 0;
        if (nucleation_mode == NUCLEATION_CONTINUOUS) {
          T_prev_map[i] = T[i];
        }
    }
    //If we're molten, call the mushy_phase function to figure out any phase change
    else if (active_flag[i] == 2 && T[i] <= tl) {
        mushy_phase(i, ranapp);
        if (nucleation_mode == NUCLEATION_CONTINUOUS && active_flag[i] == 2) {
          T_prev_map[i] = T[i];
        }
    }
    //Call smoothing
    else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3)    {
            mobility_out[i] = 1;
            smooth_site(i);
            solid_d[i]--;
    }
    // Single-voxel grain cleanup (opt-in). Triggers exactly once per voxel
    // after it exits the smoothing window. Covers both the epitaxial-end
    // state (solid_d == -nrefine-1) and the nucleation-start state
    // (solid_d == -nrefine-2). A resolved voxel is marked with
    // solid_d = -nrefine-3 so the branch never re-fires.
    else if (single_voxel_cleanup_enabled &&
             active_flag[i] == 3 &&
             solid_d[i] <= -nrefine - 1 &&
             solid_d[i] >  -nrefine - 3) {
            if (flip_single_voxel_grain(i)) {
                solid_d[i] = -nrefine - 3;
            }
    }
  }

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
  timer->stamp();
  comm->all_selective(ghost_iindices, nghost_iarray,
                      ghost_dindices, nghost_darray);
  timer->stamp(TIME_COMM);

  // Optional isotherm-compactness diagnostic for tuning smoothing.
  // Ghosts are guaranteed fresh here (just synced above).
  thermal->maybe_run_diagnostics(step_count);

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

    timer->stamp(TIME_UPDATE);
}


/* ----------------------------------------------------------------------
   Nucleation site initializer. Find the volume of a voxel from dx^3 and Multiply by no.
   This will be the average number of nucleation sites in the voxel. This is also the
   fraction of spins that we want to be able to nucleate new grains. If the value is greater
   than 1 (which we should avoid), allow all spins to nucleate. If not, call a random number
   between zero and one. If the number is less than the fraction, make true. If not, make false.
------------------------------------------------------------------------- */
void AppAdditiveTexture::nucleation_spins(RandomPark *random) {
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

void AppAdditiveTexture::grow_app()
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
  spath = darray[9];
}


/* ----------------------------------------------------------------------
   initialize before each run
   check validity of site values
------------------------------------------------------------------------- */

void AppAdditiveTexture::init_app()
{
  check_misorientation_step(dx, error, domain->me);
  check_material_complete(error, *thermal);
  check_sector_stop(error, nstop, tstop);

  // Duplicate parent's array allocation logic without quaternion randomization
  delete[] sites;
  delete[] unique;
  delete[] unique_neigh;
  sites = new int[1 + maxneigh];
  unique = new int[1 + maxneigh];
  unique_neigh = new int[1 + maxneigh];
  if (neigh_dist) delete [] neigh_dist;

  double sqrt2 = 1.4142135624;
  double sqrt3 = 1.7320508076;

  dt_sweep = dt;

  if (nucleation_flags) delete[] nucleation_flags;
  if (nucleation_temps) delete[] nucleation_temps;
  if (nucleation_sizes) delete[] nucleation_sizes;
  nucleation_flags = new int[nspins];
  nucleation_temps = new double[nspins];
	nucleation_sizes = new double[nspins];

  // Per-site orientation stream keyed to the deck seed only (not drawn from
  // ranmaster, which would shift every stream seeded after it).
  const uint64_t qseed = (uint64_t) ranmaster->initial_seed();

  int flag = 0;
	int flagall;
    for (int i = 0; i < nlocal; i++) {
			if (spin[i] < 0 || spin[i] > nspins) {
				flag = 1;
        fprintf(screen,"Bad spin %i\n",spin[i]);
			}
      if (active_flag[i] != 3) {
        // Create random orientation at each site that wasn't read as solid from
        // input file. Keyed to the site's GLOBAL id (see ranquat above), so the
        // starting field is repeatable and decomposition-independent. NOTE the
        // per-melt redraw in app_update() is deliberately left on the
        // wall-clock generator -- a site re-melts many times and would need a
        // per-site event counter to stay keyed, and this app is non-reproducible
        // anyway once nucleation/solidification start sharing an RNG.
        quaternion::SiteOrientationRNG rng(qseed, (uint64_t) id[i]);
        vector<double> uq = quaternion::generate_unit_quaternion(rng);
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
	  if (!thermal->enabled())
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
  double max_front_vel = front_velocity(tl - ts);

  
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
    enable_voids = 0;  // init_app runs every run: a second run must not add more voids
  }

  // Note: Powder activation happens after meltpool appears, not during init

	this->app_update(0.0);
}


/* ----------------------------------------------------------------------
   Override parent site_energy to only calculate energy for solidified sites
   and exclude non-solidified neighbors from energy calculation
------------------------------------------------------------------------- */

double AppAdditiveTexture::site_energy(int i) {
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

   AppAdditiveTexture overrides app_update() (see line ~400) to drive
   site evolution directly via mushy_phase()/site_event_solidification() and
   smooth_site(); the framework's rejection-KMC sweep loop in
   AppLattice::iterate_rejection is bypassed entirely. This override exists
   only to satisfy the pure virtual declared in AppLattice. If something
   ever calls it, that indicates the standard sweep path has been wired up
   on this app by mistake -- bail out loudly rather than silently running
   stale logic.
------------------------------------------------------------------------- */

void AppAdditiveTexture::site_event_rejection(int /*i*/, RandomPark * /*random*/)
{
  error->one(FLERR,
    "AppAdditiveTexture::site_event_rejection should never be "
    "called; this app drives site evolution from app_update()");
}

/* ----------------------------------------------------------------------
   Execute a nucleation event at site i with undercooling Tcool.
   Sets site to solid, computes gradient and solidification rate,
   and grows the nucleus via nucleation_particle_flipper.
------------------------------------------------------------------------- */
void AppAdditiveTexture::execute_nucleation_event(int i, double Tcool) {
    active_flag[i] = 3;
    solid_d[i] = nuclei_smooth_enabled ? -1 : -nrefine - 2;

    std::vector<double> solid_G(4);
    solid_G = normal_finder(i);
    G[i] = solid_G[3];

    V[i] = V[i] + front_velocity(Tcool);

    naccept++;
    nucleation_particle_flipper(i, round(nucleation_sizes[spin[i]] / pow(dx, 3)), ranapp, i, Tcool);

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
void AppAdditiveTexture::mushy_phase(int i, RandomPark *random){
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
void AppAdditiveTexture::site_event_solidification(int i, double Tcool, RandomPark *random) {
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
      solid_d[i] = solid_d[i] + front_velocity(Tcool) * dt;
    
      //Go through neighbor list and collect the eligible solid parents
      for (int j = 0; j < numneigh[i]; j++) {
          // Exclude void neighbors
          if(active_flag[neighbor[i][j]] == 5) continue;
          if(neigh_dist[2] <= solid_d[i] && (active_flag[neighbor[i][j]] == 1 || active_flag[neighbor[i][j]] == 3)) {
              unique_neigh[nevent] = neighbor[i][j];
              unique[nevent] = spin[neighbor[i][j]];
              nevent++;
          }
      }
    }
    //If no neighbor is eligible, return before changing anything. Will try next sweep.
    if (nevent == 0) return;

    // Parent selection is per GRAIN, not per voxel. Weighting each candidate
    // voxel by its own orientation let the mobility select, among the
    // randomly rotated voxels of ONE grain, the ones rotated toward gradient
    // alignment, so whole grains drifted toward <001>||BD with height (12-16
    // deg over 240-360 um on the GP700 build; Lamb et al. see no such drift).
    // Now: group the candidates by spin, weight each grain by the mobility of
    // the sign-aligned mean orientation of its candidate voxels against the
    // gradient at this site, draw the grain, then draw the parent voxel
    // uniformly within it. Grain-vs-grain competition (the <001> fibre) is kept.
    int gspin[27]; double gcum[27]; int ngr = 0; double gtotal = 0.0;
    std::vector<double> grad_full = normal_finder(i);   // [nx, ny, nz, magnitude]
    for (int k = 0; k < nevent; k++) {
        bool seen = false;
        for (int g = 0; g < ngr; g++) if (gspin[g] == unique[k]) { seen = true; break; }
        if (seen) continue;
        std::vector<double> qm = {0.0, 0.0, 0.0, 0.0}, qr;
        for (int m = 0; m < nevent; m++) {
            if (unique[m] != unique[k]) continue;
            int n = unique_neigh[m];
            std::vector<double> qn = {q0[n], qx[n], qy[n], qz[n]};
            if (qr.empty()) qr = qn;
            else if (qr[0]*qn[0] + qr[1]*qn[1] + qr[2]*qn[2] + qr[3]*qn[3] < 0.0)
                for (int c = 0; c < 4; c++) qn[c] = -qn[c];   // q and -q are one orientation
            for (int c = 0; c < 4; c++) qm[c] += qn[c];
        }
        double nrm = sqrt(qm[0]*qm[0] + qm[1]*qm[1] + qm[2]*qm[2] + qm[3]*qm[3]);
        if (nrm > 0.0) for (int c = 0; c < 4; c++) qm[c] /= nrm;
        gtotal += mobility_of_orientation(qm, grad_full);
        gspin[ngr] = unique[k]; gcum[ngr] = gtotal; ngr++;
    }
    int g = 0;
    if (gtotal > 0.0) {
        double dran = gtotal * random->uniform();
        while (g < ngr - 1 && dran > gcum[g]) g++;
    } else {
        g = (int)(ngr * random->uniform()); if (g >= ngr) g = ngr - 1;
    }
    int nsame = 0;
    for (int m = 0; m < nevent; m++) if (unique[m] == gspin[g]) nsame++;
    int pick = (int)(nsame * random->uniform()); if (pick >= nsame) pick = nsame - 1;
    for( int j = 0; j < nevent; j++) {
        if (unique[j] != gspin[g]) continue;
        if (pick-- > 0) continue;
        {
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

            V[i] = V[i] + front_velocity(Tcool); //Update solidification rate
            naccept++;

            // Apply misorientation based on temperature gradient
            apply_misorientation(i, neighran, Tcool, ranapp);
            return;
        }
    }
}

/* ----------------------------------------------------------------------
    Apply misorientation based on temperature gradient and cooling rate
------------------------------------------------------------------------- */
double AppAdditiveTexture::diffusion_angle(double Tcool, double hop_m) const {
    double D = mis_diffusion;
    if (misorient_alpha > 0.0) {
        double x = Tcool / t_cool_max;
        if (x < 0.0) x = 0.0;
        if (x > 1.0) x = 1.0;
        double g = (exp(misorient_alpha * x) - 1.0) / (exp(misorient_alpha) - 1.0);
        D *= g * g;
    }
    return sqrt(D * hop_m);
}

void AppAdditiveTexture::apply_misorientation(int i, int parent, double Tcool, RandomPark *ranapp) {
    // Hop to the inheritance parent. Coordinate differences are in lattice
    // units; the sqrt(3) clamp guards against a periodic-wrap neighbor pair
    // reading as a box-length hop (the true SC_26N stencil distance is at
    // most sqrt(3)).
    double ddx = xyz[i][0] - xyz[parent][0];
    double ddy = xyz[i][1] - xyz[parent][1];
    double ddz = xyz[i][2] - xyz[parent][2];
    double hop_lat = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    if (hop_lat > 1.7320508076) hop_lat = 1.7320508076;
    spath[i] = spath[parent] + hop_lat;       // inheritance path length (d10)

    if (grain_walk_D > 0.0) {   // coherent grain rotation between the parent's level and this one
        double q[4] = {q0[i], qx[i], qy[i], qz[i]};
        if (grain_walk_path) grain_walk_apply(spin[parent], spath[parent], spath[i], sqrt(grain_walk_D * dx), grain_walk_rho, q);
        else                 grain_walk_apply(spin[parent], xyz[parent][2], xyz[i][2], sqrt(grain_walk_D * dx), grain_walk_rho, q);
        q0[i] = q[0]; qx[i] = q[1]; qy[i] = q[2]; qz[i] = q[3];
    }
    if (mis_diffusion <= 0.0) return;

    rotate_toward_target(i, diffusion_angle(Tcool, hop_lat * dx),
                         misorientation_target_mode, ranapp);
}


/* ----------------------------------------------------------------------
    Rotate site i's orientation by mis_angle per the given target mode, then
    apply the mis_thresh new-grain check. Shared by the solidification path
    and the nucleation particle flipper.
------------------------------------------------------------------------- */
void AppAdditiveTexture::rotate_toward_target(int i, double mis_angle, int target_mode,
                                              RandomPark *ranapp) {
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

    vector<double> q_new;
    if (target_mode == MISORI_TARGET_RANDOM_ROTATION) {
        // Apply a rotation by mis_angle about a uniformly-random axis,
        // composed with the existing site quaternion. No <100>-toward-target
        // alignment step.
        vector<double> axis = sample_random_unit_vector(ranapp);
        double half = 0.5 * mis_angle;
        double s = sin(half);
        vector<double> dq = {cos(half), s*axis[0], s*axis[1], s*axis[2]};
        q_new = quaternion::composition(dq, q_site, /*unit=*/false);
        double m = sqrt(q_new[0]*q_new[0] + q_new[1]*q_new[1]
                      + q_new[2]*q_new[2] + q_new[3]*q_new[3]);
        if (m > 1e-15) {
            q_new[0] /= m; q_new[1] /= m; q_new[2] /= m; q_new[3] /= m;
        }
    } else {  // MISORI_TARGET_RANDOM: rotate the nearest <100> toward a random direction
        vector<double> target_dir = sample_random_unit_vector(ranapp);
        q_new = quaternion::rotate_q_towards_u(q_site, target_dir, mis_angle);
    }

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
void AppAdditiveTexture::nucleation_particle_flipper(int i, int partRad, RandomPark *random,
                                                     int seed, double Tcool) {
    
    //If one site is big enough to satisfy, skip evertyhing
    if(partRad <= 0) return;
    
    int nSites = partRad;
    int nSitesIn = nSites;
    int nearest_neigh [ ] = {4, 10, 12, 15, 21, 13};
    int second_nearest [ ] = {9,16,3,22,14,11,20,5,1,24,7,18};
    int third_nearest [ ] = {0,25,23,2,6,19,17,8};
    int nneigh = 0;
    int possible_neigh[26];
    // Every claimed site inherits the SEED orientation (not the current
    // growth center's, which the recursion re-roots on) and takes one walk
    // step of sqrt(D_eff * distance_from_seed) about a random axis -- the
    // random walk's RMS for that growth distance, so nuclei and epitaxial
    // growth obey one law. Same rule as the kk particle_flipper_kernel.
    SiteState s_in(spin[seed],{q0[seed], qx[seed], qy[seed], qz[seed]});
    bool nucl_rotate = (mis_diffusion > 0.0 && nucleation_misorient_enabled);
    auto rotate_flipped = [&](int i_chosen) {
        if (!nucl_rotate) return;
        double ddx = xyz[i_chosen][0] - xyz[seed][0];
        double ddy = xyz[i_chosen][1] - xyz[seed][1];
        double ddz = xyz[i_chosen][2] - xyz[seed][2];
        double r_m = sqrt(ddx*ddx + ddy*ddy + ddz*ddz) * dx;
        rotate_toward_target(i_chosen, diffusion_angle(Tcool, r_m),
                             misorientation_target_mode, random);
    };


    // Only OWNED sites are claimed: a ghost flipped here would never reach its
    // owning rank (the next ghost exchange overwrites it), yet this rank's
    // sites would see it as solid in the meantime.
    // ponytail: a nucleus at a rank boundary stops at it; growing across needs
    // a reverse comm of claimed ghosts.

    //Its hard to go through shells iteravely if the number of neighbors isn't the full 26.
    //Check if this is the case and just loop through the neihgbor list if so
    if(numneigh[i] != 26) {

        
        for(int j = 0; j < numneigh[i]; j++) {
            if(neighbor[i][j] < nlocal && active_flag[neighbor[i][j]] == 2) {
                int i_chosen = neighbor[i][j];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                spath[i_chosen] = 0.0;
                rotate_flipped(i_chosen);
                solid_d[i_chosen] = nuclei_smooth_enabled ? -1 : -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
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
            if(neighbor[i][nearest_neigh[idx]] < nlocal && active_flag[neighbor[i][nearest_neigh[idx]]] == 2) {
                int i_chosen = neighbor[i][nearest_neigh[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                spath[i_chosen] = 0.0;
                rotate_flipped(i_chosen);
                solid_d[i_chosen] = nuclei_smooth_enabled ? -1 : -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
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
            if(neighbor[i][second_nearest[idx]] < nlocal && active_flag[neighbor[i][second_nearest[idx]]] == 2) {
                int i_chosen = neighbor[i][second_nearest[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                spath[i_chosen] = 0.0;
                rotate_flipped(i_chosen);
                solid_d[i_chosen] = nuclei_smooth_enabled ? -1 : -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
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
            if(neighbor[i][third_nearest[idx]] < nlocal && active_flag[neighbor[i][third_nearest[idx]]] == 2) {
                int i_chosen = neighbor[i][third_nearest[idx]];
                flip_site(i_chosen, s_in);
                active_flag[i_chosen] = 3;
                spath[i_chosen] = 0.0;
                rotate_flipped(i_chosen);
                solid_d[i_chosen] = nuclei_smooth_enabled ? -1 : -nrefine -3;
                G[i_chosen] = G[i];
                V[i_chosen] = V[i];
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
        nucleation_particle_flipper(next_site, nSites, random, seed, Tcool);
    return;
    
}


//Texture specific functions start here

// Calculate the local direction of the largest temperature gradient
// using weighted least squares with multiple neighbors for improved accuracy
std::vector<double> AppAdditiveTexture::normal_finder(int site)
{
	// Initialize gradient components
	double grad_x = 0, grad_y = 0, grad_z = 0;
	
	// Special case: top of melt pool. If the neighbour directly above (+z) is
	// missing (a top-boundary site), not melted/solid (active_flag <= 1) or a void,
	// use only neighbours at or below the site's z. The +z neighbour is found
	// by its coordinates, not by slot 13: a boundary site's neighbour list is
	// compacted, so slot n is the n-th existing neighbour, not the n-th
	// SC_26N stencil position.
	int above = -1;
	for (int n = 0; n < numneigh[site]; n++) {
		const int nb = neighbor[site][n];
		if (xyz[nb][0] == xyz[site][0] && xyz[nb][1] == xyz[site][1] &&
		    xyz[nb][2] == xyz[site][2] + 1.0) { above = nb; break; }
	}
	// A void above (i2 = 5) is air, so it is a free surface too.
	bool is_melt_surface = (above < 0) || (active_flag[above] <= 1) || (active_flag[above] == 5);
	double site_z = xyz[site][2];

	// Weighted least squares: A*grad = b, A rows [dx dy dz] per neighbor, b the
	// temperature differences, weights inverse distance squared. At the melt
	// surface only neighbors at or below the site's z-level contribute.
	double AtA[9] = {0}; // 3x3 matrix A^T * W * A
	double Atb[3] = {0}; // 3x1 vector A^T * W * b
	double T_center = T[site];

	for (int n = 0; n < numneigh[site] && n < 26; n++) {
		int neighbor_site = neighbor[site][n];
		if (neighbor_site < 0 || neighbor_site >= app->nlocal + app->nghost) continue;
		if (is_melt_surface && xyz[neighbor_site][2] > site_z) continue;

		// Offset from the coordinates (slot n is not stencil position n at a
		// boundary site; identical to the stencil for interior sites).
		double dx_n = (xyz[neighbor_site][0] - xyz[site][0]) * dx;
		double dy_n = (xyz[neighbor_site][1] - xyz[site][1]) * dx;
		double dz_n = (xyz[neighbor_site][2] - xyz[site][2]) * dx;
		double dT = T[neighbor_site] - T_center;
		double dist = sqrt(dx_n*dx_n + dy_n*dy_n + dz_n*dz_n);
		double weight = 1.0 / (dist*dist + 1e-12);

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
	}

	// Solve the 3x3 system with Cramer's rule.
	// Note: Diagnostics (2M+ calls) showed matrix always non-singular
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

	// Normalize and return gradient direction (with magnitude as 4th element)
	// Note: Diagnostics (2M+ calls) showed norm always > 1e-12
	double norm = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
	std::vector<double> result(4);  // [norm_x, norm_y, norm_z, magnitude]
	result[0] = fabs(grad_x)/norm;
	result[1] = fabs(grad_y)/norm;
	result[2] = fabs(grad_z)/norm;
	result[3] = norm;
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
std::vector<double> AppAdditiveTexture::get_average_neighbor_quaternion(int site, int target_spin)
{
    if (smooth_orientation_copy) {   // one random neighbour of the new grain
        int cand[27]; int nc = 0;
        for (int j = 0; j < numneigh[site]; j++)
            if (spin[neighbor[site][j]] == target_spin && active_flag[neighbor[site][j]] != 5)
                cand[nc++] = neighbor[site][j];
        if (nc == 0) error->all(FLERR,"get_average_neighbor_quaternion: no neighbor with target spin");
        int k = (int)(nc * ranapp->uniform()); if (k >= nc) k = nc - 1;
        double q[4] = {q0[cand[k]], qx[cand[k]], qy[cand[k]], qz[cand[k]]};
        spath[site] = spath[cand[k]];   // copy the path length with the orientation
        if (grain_walk_D > 0.0 && !grain_walk_path)   // height mode: the copied neighbour may sit on another level
            grain_walk_apply(target_spin, xyz[cand[k]][2], xyz[site][2], sqrt(grain_walk_D * dx), grain_walk_rho, q);
        return {q[0], q[1], q[2], q[3]};
    }
    // Initialize storage for first quaternion as reference
    double ref_q0 = 0.0, ref_qx = 0.0, ref_qy = 0.0, ref_qz = 0.0;
    bool first_found = false;
    
    // Initialize sums
    double sum_q0 = 0.0, sum_qx = 0.0, sum_qy = 0.0, sum_qz = 0.0;
    int count = 0;
    
    // Check all neighbors
    for (int j = 0; j < numneigh[site]; j++) {
        int neighbor_site = neighbor[site][j];
        
        // Check if this neighbor has the target spin (voids carry no grain)
        if (spin[neighbor_site] == target_spin && active_flag[neighbor_site] != 5) {
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
/* ----------------------------------------------------------------------
   Mobility c1 + c2*cos(c3*theta) of an orientation q against a gradient
   [nx,ny,nz,magnitude]; theta = angle between the nearest <100> and the
   gradient. Zero gradient => no preferred direction => full mobility.
   Negative values clamp to 0 (zero-weight candidate).
------------------------------------------------------------------------- */
double AppAdditiveTexture::mobility_of_orientation(const std::vector<double> &q,
                                                   const std::vector<double> &grad_full)
{
	if (grad_full[3] <= 0.0) return c1 + c2;
	std::vector<double> dir = {grad_full[0], grad_full[1], grad_full[2]};
	double theta = acos(quaternion::get_cosine_of_minumum_angle_between_q_and_u(q, dir));
	double mob = c1 + c2 * cos(c3 * theta);
	return mob < 0.0 ? 0.0 : mob;
}

double AppAdditiveTexture::melt_misorientation(int site, double c1, double c2, double c3)
{
	double dotMax;
	double theta;
	double mobOut;

	//Calculate the unit-vector surface norm (use voxel counting)
	vector<double> grad_full = normal_finder(site);  // Returns [norm_x, norm_y, norm_z, magnitude]
	// A zero gradient (flat field) has no preferred growth direction: full
	// mobility for every candidate. The cosine helper returns 0 for a zero
	// vector (a bogus 90 deg), which with c3 = 2.5 can go negative. Same rule
	// as the kk SolidificationKernel.
	if (grad_full[3] <= 0.0) return c1 + c2;
	vector<double> grad_vector = {grad_full[0], grad_full[1], grad_full[2]};  // Extract direction only
	
	// Create quaternion vector from site's orientation
	vector<double> q_site = {q0[site], qx[site], qy[site], qz[site]};
	
	// Use quaternion function to get cosine of minimum angle between
	// crystal orientation and temperature gradient
	dotMax = quaternion::get_cosine_of_minumum_angle_between_q_and_u(q_site, grad_vector);
	
	// Calculate angle from cosine
	theta = acos(dotMax);
	
	// Apply texture mobility model; a negative mobility is a zero-weight
	// candidate (the roulette wheel below needs monotone cumulative weights).
	mobOut = c1 + c2 * cos(c3 * theta);
	if (mobOut < 0.0) mobOut = 0.0;
	
	return mobOut;
}

/* ----------------------------------------------------------------------
   This function is called after a site is solidified and nsmooth > 0. Its purpose is to smooth grain boundaries.
   It does this by using the traditional Potts model without including grain orientations. Quaternions are handled
   by switching the site to the average quaternion value of neighbors with the selected spin/grain ID.

   Inputs: lattice site :: site
   Outputs: None
------------------------------------------------------------------------- */
void AppAdditiveTexture::smooth_site(int i) {

  //Adapt code from potts_neigh_only site_event_rejection.
  int oldstate = spin[i];
  double einitial = site_energy_smooth(i);

  // events = spin flips to neighboring site different than self

  int j,m,value;
  int nevent = 0;

  for (j = 0; j < numneigh[i]; j++) {
    if (active_flag[neighbor[i][j]] == 5) continue;  // voids carry no grain
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

double AppAdditiveTexture::site_energy_smooth(int i)
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
bool AppAdditiveTexture::flip_single_voxel_grain(int i)
{
  int oldstate = spin[i];
  int nunique = 0;
  bool has_same_spin_neighbor = false;

  for (int j = 0; j < numneigh[i]; j++) {
    int nj = neighbor[i][j];
    if (active_flag[nj] == 5) continue;      // a void never matures and carries no grain
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
void AppAdditiveTexture::nucleation_init() {
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
   Activate powder sites that are at or below the meltpool surface

   Strategy: Find the maximum z-coordinate of all molten sites (meltpool top),
   then set all inactive sites below that level to powder (active_flag = 1).

   This function should be called:
   - After fast forwards > 1 second (new layer deposition)
   - During app_update when meltpool is present
------------------------------------------------------------------------- */
void AppAdditiveTexture::activate_powder_sites() {
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
  bigint powder_me = powder_activated, global_powder_activated = 0;  // a layer can pass 2^31
  MPI_Allreduce(&powder_me, &global_powder_activated, 1, MPI_SPK_BIGINT, MPI_SUM, world);

  // Report activation (rank 0 only)
  if (domain->me == 0 && global_powder_activated > 0) {
    fprintf(screen, "Powder activation at t=%.6e s: activated " BIGINT_FORMAT " sites (meltpool top at z=%.3f)\n",
            time, global_powder_activated, global_max_molten_z);
  }
}

/* ----------------------------------------------------------------------
   Generate spherical voids in the domain based on volumetric density
   and size distribution. Voids are represented by activeFlag = 5.
   Generation happens on rank 0, then void list is broadcast to all ranks.
------------------------------------------------------------------------- */
void AppAdditiveTexture::generate_voids(RandomPark *random) {
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
        domain_volume = xlen * ylen * zlen * dx * dx * dx;  // box is in lattice units; m^3

        // If pore fraction was specified, calculate void_density from it
        if (void_pore_fraction > 0.0) {
            // mean void volume = 4/3 pi E[r^3] over the truncated normal
            // (not at r_mean: E[r^3] > r_mean^3 overshot the target)
            double r3 = pow(void_radius_mean, 3);
            if (void_radius_std > 0.0) {
                double s3 = 0.0, sw = 0.0;
                const int nq = 2000;
                for (int k = 0; k < nq; k++) {
                    double r = void_radius_min + (k + 0.5) * (void_radius_max - void_radius_min) / nq;
                    double u = (r - void_radius_mean) / void_radius_std;
                    double p = exp(-0.5 * u * u);
                    s3 += p * r * r * r;
                    sw += p;
                }
                if (sw > 0.0) r3 = s3 / sw;
            }
            double avg_void_volume = (4.0/3.0) * M_PI * r3 * 1e-18;  // um^3 -> m^3
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

                // Sample radius from truncated Gaussian (convert um to m);
                // Box-Muller on the app RNG so the voids follow the deck seed
                double radius_m;
                do {
                    double u1 = random->uniform(), u2 = random->uniform();
                    double g = sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * M_PI * u2);
                    radius_m = (void_radius_mean + void_radius_std * g) * 1e-6;
                } while (radius_m < r_min_m || radius_m > r_max_m);
                // centres and xyz are lattice units: store the radius in them too
                const double radius = radius_m / dx;

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
                    double min_dist = radius + existing_void.radius;

                    if (dist < min_dist) {
                        collides = true;
                        break;
                    }
                }

                if (!collides) {
                    voids.emplace_back(x, y, z, radius);
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
    bigint voids_me = sites_in_voids, global_sites_in_voids = 0;
    MPI_Reduce(&voids_me, &global_sites_in_voids, 1, MPI_SPK_BIGINT, MPI_SUM, 0, world);

    if (domain->me == 0 && actual_void_count > 0) {
        fprintf(screen, "Total sites marked as voids: " BIGINT_FORMAT "\n", global_sites_in_voids);

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
   Deprecated app_style alias 'additive_temperature_texture'.
   Forwards to AppAdditiveTexture; emits a one-shot warning so existing
   input scripts keep running while users migrate to the canonical
   'additive/texture' app_style.
------------------------------------------------------------------------- */

AppAdditiveTextureDeprecated::AppAdditiveTextureDeprecated(SPPARKS *spk,
                                                           int narg,
                                                           char **arg)
  : AppAdditiveTexture(spk, narg, arg)
{
  if (domain->me == 0)
    error->warning(FLERR,
      "app_style 'additive_temperature_texture' is deprecated; "
      "use 'additive/texture' instead. The two are equivalent for now, "
      "but the deprecated alias will be removed in a future release.");
}
