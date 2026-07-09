#include "openmc/tallies/filter_circumferential_expansion.h"

#include <cmath> // for atan2

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/surface.h"
#include "openmc/xml_interface.h"

namespace openmc {

void CircumferentialExpansionFilter::set_surface(int32_t surface_index)
{
  surface_index_ = surface_index;
  const auto* s = model::surfaces[surface_index].get();
  if (auto* c = dynamic_cast<const SurfaceXCylinder*>(s)) {
    axis_ = 0;
    c1_ = c->y0();
    c2_ = c->z0();
  } else if (auto* c = dynamic_cast<const SurfaceYCylinder*>(s)) {
    axis_ = 1;
    c1_ = c->x0();
    c2_ = c->z0();
  } else if (auto* c = dynamic_cast<const SurfaceZCylinder*>(s)) {
    axis_ = 2;
    c1_ = c->x0();
    c2_ = c->y0();
  } else {
    throw std::runtime_error {
      "Circumferential expansion filter requires an axis-aligned cylinder."};
  }
}

double CircumferentialExpansionFilter::affine_transform(
  const Particle& p) const
{
  double a, b;
  if (axis_ == 0) {
    a = p.r().y - c1_;
    b = p.r().z - c2_;
  } else if (axis_ == 1) {
    a = p.r().x - c1_;
    b = p.r().z - c2_;
  } else {
    a = p.r().x - c1_;
    b = p.r().y - c2_;
  }
  double theta = std::atan2(b, a);
  if (theta < 0.0)
    theta += 2.0 * PI;
  return theta;
}

void CircumferentialExpansionFilter::from_xml(pugi::xml_node node)
{
  this->set_order(std::stoi(get_node_value(node, "order")));

  int32_t id = std::stoi(get_node_value(node, "surface"));
  auto search = model::surface_map.find(id);
  if (search == model::surface_map.end()) {
    throw std::runtime_error {
      fmt::format("Could not find surface {} specified on tally filter.", id)};
  }
  this->set_surface(search->second);
}

void CircumferentialExpansionFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "order", order_);
  write_dataset(filter_group, "surface", model::surfaces[surface_index_]->id_);
}

} // namespace openmc
