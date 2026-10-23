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

#ifndef SPK_MATERIAL_CARD_H
#define SPK_MATERIAL_CARD_H

#include <mpi.h>
#include <optional>
#include <string>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   A material parameter card (docs/MATERIAL_CARDS.md) resolved to SI
   numbers. Values the card does not give are empty optionals (tables,
   coefficient lists: empty vectors); not NaN, which -ffast-math folds away.
   Every error throws std::runtime_error.
------------------------------------------------------------------------- */

struct MaterialCard {
  using opt = std::optional<double>;
  struct Table { std::vector<double> T, v; };

  std::string material, version, modified, process;   // process: selected set, or ""

  // [thermal], [thermal.linear], [powder], [optical]
  opt density, solidus, liquidus, latent, emissivity;
  Table k, cp;
  opt k_lin, cp_lin, T_ref;
  opt k_powder, packing, absorptivity;

  // [microstructure]
  std::vector<double> front;   // solid_front_vel coefficients, highest power first,
  opt front_a, front_n;        // or the power law a * dT^n (front empty)
  opt nucleation_density, critical_undercooling, undercooling_deviation,
      mean_nuclei_volume, nuclei_volume_deviation;

  // [process.<name>]; absorptivity above is the process value when it gives one
  opt power, speed, hatch, layer_thickness, spot_sigma,
      source_depth, preheat, conv_h, conv_Tinf;

  std::vector<std::string> summary;   // one "key = value unit [ref]" line per value read

  static MaterialCard parse(const std::string &text, const std::string &path,
                            const std::string &process);
  // Rank 0 reads the file, every rank parses the broadcast text.
  static MaterialCard load(const std::string &path, const std::string &process,
                           MPI_Comm world);
  // setup_temperature_source tokens for <type> from this card; args are the
  // deck tokens after "card" (Q, lambda, T0, spot overrides; the rest appended)
  std::vector<std::string> source_tokens(const std::string &type,
                                         const std::vector<std::string> &args) const;
  static double table_at(const Table &t, double T);
  static std::string num(double v);   // shortest round-trip decimal
};

}

#endif
