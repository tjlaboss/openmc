#ifndef OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H
#define OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H

#include <string>

#include "openmc/tallies/filter_circumferential_expansion.h"

namespace openmc {

//==============================================================================
//! Gives Legendre moments of a particle's azimuthal position about a cylinder.
//==============================================================================

class CircumferentialLegendreFilter : public CircumferentialExpansionFilter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "circumferentiallegendre"; }
  FilterType type() const override
  {
    return FilterType::CIRCUMFERENTIAL_LEGENDRE;
  }

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_order(int order) override;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H
