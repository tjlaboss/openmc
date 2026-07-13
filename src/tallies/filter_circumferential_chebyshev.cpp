#include "openmc/tallies/filter_circumferential_chebyshev.h"

#include <utility> // for pair

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/surface.h"

namespace openmc {

void CircumferentialChebyshevFilter::set_order(int order)
{
  if (order < 0) {
    throw std::invalid_argument {"Chebyshev order must be non-negative."};
  }
  order_ = order;
  n_bins_ = order_ + 1;
}

void CircumferentialChebyshevFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  if (p.surface_index() != surface_index_)
    return;

  // Chebyshev polynomials are evaluated on [-1, 1)
  double x = this->affine_transform(p) / PI - 1.0;

  vector<double> tnx(n_bins_);
  calc_tn_c(order_, x, tnx.data());
  for (int i = 0; i < n_bins_; ++i) {
    match.bins_.push_back(i);
    match.weights_.push_back(tnx[i]);
  }
}

std::string CircumferentialChebyshevFilter::text_label(int bin) const
{
  return fmt::format("Circumferential Chebyshev expansion, surface {}, T{}",
    model::surfaces[surface_index_]->id_, bin);
}

//==============================================================================
// C-API functions
//==============================================================================

std::pair<int, CircumferentialChebyshevFilter*>
check_circumferential_chebyshev_filter(int32_t index)
{
  int err = verify_filter(index);
  if (err) {
    return {err, nullptr};
  }

  const auto& filt_base = model::tally_filters[index].get();
  auto* filt = dynamic_cast<CircumferentialChebyshevFilter*>(filt_base);

  if (!filt) {
    set_errmsg("Not a circumferential Chebyshev filter.");
    err = OPENMC_E_INVALID_TYPE;
  }
  return {err, filt};
}

extern "C" int openmc_circumferential_chebyshev_filter_get_order(
  int32_t index, int* order)
{
  auto check_result = check_circumferential_chebyshev_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  *order = filt->order();
  return 0;
}

extern "C" int openmc_circumferential_chebyshev_filter_set_order(
  int32_t index, int order)
{
  auto check_result = check_circumferential_chebyshev_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  filt->set_order(order);
  return 0;
}

extern "C" int openmc_circumferential_chebyshev_filter_get_surface(
  int32_t index, int32_t* surface)
{
  auto check_result = check_circumferential_chebyshev_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  *surface = model::surfaces[filt->surface_index()]->id_;
  return 0;
}

extern "C" int openmc_circumferential_chebyshev_filter_set_surface(
  int32_t index, int32_t surface)
{
  auto check_result = check_circumferential_chebyshev_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  auto search = model::surface_map.find(surface);
  if (search == model::surface_map.end()) {
    set_errmsg(fmt::format(
      "Could not find surface {} specified on tally filter.", surface));
    return OPENMC_E_INVALID_ARGUMENT;
  }
  filt->set_surface(search->second);
  return 0;
}

} // namespace openmc