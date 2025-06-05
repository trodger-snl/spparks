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

#ifdef APP_CLASS
AppStyle(additive_temperature_texture,AppAdditiveExtTempTexture)

#else

#ifndef SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_H
#define SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_H

#include "app_potts.h"
#include "temperatureQueues.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>

namespace SPPARKS_NS {

class AppAdditiveExtTempTexture : public AppPotts {
 public:
  AppAdditiveExtTempTexture(class SPPARKS *, int, char **);
  virtual void grow_app();
  virtual void init_app();
  virtual void site_event_rejection(int, class RandomPark *);
  virtual void app_update(double);
  virtual void nucleation_spins(class RandomPark *);
  //virtual void nucleation_temps();
  virtual void  input_app(char *, int , char **);
  //virtual double mushy_mobility(int, class RandomPark *);
  virtual void orientation_init(class RandomPark *);
  virtual void nucleation_init();
  virtual void nucleation_particle_flipper(int, int,class RandomPark *);
  virtual void mushy_phase(int, class RandomPark *);
  void normal_finder(int, double *);
	double melt_misorientation(int, double, double, double);
  void vec2euler(int, double *);
  void euler_init();
  virtual void reduced_temperature_hdf();
  int xyz_to_local( double x, double y, double z );
  virtual void temperature_time_interpolate(int, double);
	std::vector<std::vector<std::vector<int>>> convertTo3DArrayWithRange(std::vector<int>&,int,int,int,int,int,int);

 protected:

	double *MobilityOut;
	double *SolidD;
		double *neighDist;
    int time_index;
    double dtFD;
 	//To help improve ease of visualization, lets introduce another integer array. It will be 0 every to begin with
 	//When its layer becomes "active" we'll switch it to 1. This will help visualization and image dumping.
 	int *activeFlag;
 	double read_interval;
 	const double R = 8.314459848; //Define a constant gas constant
	/// parameters for the thermal diffusion eq
	double *T;
	double time_step;
	int totalTime; //Keep track of time for app_update
   int line_count;
  	double *temp_in_array;
  	double *phase_in_array;
	int xPeriod, yPeriod, zPeriod;
	int nrefine;
	//New inputs
	double dt;
	double No;
	double Tc;
	double Tsig;
	double dx;
    double Mo; //Arrhenius pre-factor
 	double Q; //Arrhenius exponential factor
 	double Tl; //Liquidus point
 	double Ts; //Solidus point
  double T_room;
    double *phi1;
    double *Phi;
    double *phi2;
    double *uniqueDot;
    double *spin_euler;
    double *x1, *x2, *x3;
    double *y1, *y2, *y3;
    double *orientation_vectors;
    
  //New stuff
  double sizeNorm;
	double sizeSig;
	double *solid_front_coeffs;
	int    solid_front_length;
	//Texture parameters
	double c1, c2, c3;
	int nlocal_app; //size of cnew, not sure if needed
 	int fully_periodic;
 	
 	//New MC model parameters
	int *nucleationFlags;
	double *nucleationTemps;
	double *nucleationSizes;
  std::string temp_file_str;

  //Variables for reduced temperature files
  DoubleQueueContainer temp_in;
  DoubleQueueContainer time_in;
  double priorTime;
  int t_active;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPPARKS to see the offending
line.

E: One or more sites have invalid values

The application only allows sites to be initialized with specific
values.

*/
