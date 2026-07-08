#ifndef OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H
#define OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H

#include <cstdint>
#include <string>

#include "openmc/tallies/filter.h"

namespace openmc {

//==============================================================================
//! Gives Legendre moments of a particle's azimuthal position about a cylinder.
//==============================================================================

class CircumferentialLegendreFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~CircumferentialLegendreFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "circumferentiallegendre"; }
  FilterType type() const override { return FilterType::CIRCUMFERENTIAL_LEGENDRE; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  int order() const { return order_; }
  void set_order(int order);

  int32_t surface_index() const { return surface_index_; }
  void set_surface(int32_t surface_index);

private:
  //----------------------------------------------------------------------------
  // Methods

  //! Map the azimuthal angle about the cylinder onto [0, 1).
  double affine_transform(const Particle& p) const;

  //----------------------------------------------------------------------------
  // Data members

  int order_;

  //! Index of the cylinder surface in model::surfaces.
  int32_t surface_index_;

  //! Perpendicular axis (0 = x, 1 = y, 2 = z) and in-plane center, cached from
  //! the cylinder when the surface is set.
  int axis_;
  double c1_, c2_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_LEGENDRE_H
