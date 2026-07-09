#ifndef OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_EXPANSION_H
#define OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_EXPANSION_H

#include <cstdint>
#include <string>

#include "openmc/tallies/filter.h"

namespace openmc {

//==============================================================================
//! Abstract base class for filters that expand a particle's azimuthal
//! position about an axis-aligned cylinder in some functional basis.
//==============================================================================

class CircumferentialExpansionFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~CircumferentialExpansionFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  void from_xml(pugi::xml_node node) override;

  void to_statepoint(hid_t filter_group) const override;

  //----------------------------------------------------------------------------
  // Accessors

  int order() const { return order_; }
  virtual void set_order(int order) = 0;

  int32_t surface_index() const { return surface_index_; }
  void set_surface(int32_t surface_index);

protected:
  //----------------------------------------------------------------------------
  // Methods

  //! Map the azimuthal angle about the cylinder onto this filter's domain.
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
#endif // OPENMC_TALLIES_FILTER_CIRCUMFERENTIAL_EXPANSION_H
