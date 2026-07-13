#ifndef OPENMC_TALLIES_FILTER_SPTL_CHEBYSHEV_H
#define OPENMC_TALLIES_FILTER_SPTL_CHEBYSHEV_H

#include <string>

#include "openmc/tallies/filter_sptl_expansion.h"

namespace openmc {

//==============================================================================
//! Gives Chebyshev moments of the particle's normalized position along an
//! axis
//==============================================================================

class SpatialChebyshevFilter : public SpatialExpansionFilter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~SpatialChebyshevFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "spatialchebyshev"; }
  FilterType type() const override { return FilterType::SPATIAL_CHEBYSHEV; }

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_order(int order) override;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SPTL_CHEBYSHEV_H