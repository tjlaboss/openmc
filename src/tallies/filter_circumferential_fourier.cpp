#include "openmc/tallies/filter_circumferential_fourier.h"

#include <utility> // for pair

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/surface.h"
#include "openmc/tallies/filter_fourier_basis.h"

namespace openmc {

void CircumferentialFourierFilter::set_order(int order)
{
  if (order < 0) {
    throw std::invalid_argument {"Fourier order must be non-negative."};
  }
  order_ = order;
  n_bins_ = 2 * order_ + 1;
}

void CircumferentialFourierFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  if (p.surface_index() != surface_index_)
    return;

  // Fourier basis operates on angle normalized to [0, 1)
  double x = this->affine_transform(p) / (2.0 * PI);

  vector<double> wgt(n_bins_);
  fourier_weights(x, order_, wgt);
  for (int i = 0; i < n_bins_; ++i) {
    match.bins_.push_back(i);
    match.weights_.push_back(wgt[i]);
  }
}

std::string CircumferentialFourierFilter::text_label(int bin) const
{
  return fmt::format("Circumferential Fourier expansion, surface {}, {}",
    model::surfaces[surface_index_]->id_, fourier_bin_label(bin));
}

//==============================================================================
// C-API functions
//==============================================================================

std::pair<int, CircumferentialFourierFilter*>
check_circumferential_fourier_filter(int32_t index)
{
  int err = verify_filter(index);
  if (err) {
    return {err, nullptr};
  }

  const auto& filt_base = model::tally_filters[index].get();
  auto* filt = dynamic_cast<CircumferentialFourierFilter*>(filt_base);

  if (!filt) {
    set_errmsg("Not a circumferential Fourier filter.");
    err = OPENMC_E_INVALID_TYPE;
  }
  return {err, filt};
}

extern "C" int openmc_circumferential_fourier_filter_get_order(
  int32_t index, int* order)
{
  auto check_result = check_circumferential_fourier_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  *order = filt->order();
  return 0;
}

extern "C" int openmc_circumferential_fourier_filter_set_order(
  int32_t index, int order)
{
  auto check_result = check_circumferential_fourier_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  filt->set_order(order);
  return 0;
}

extern "C" int openmc_circumferential_fourier_filter_get_surface(
  int32_t index, int32_t* surface)
{
  auto check_result = check_circumferential_fourier_filter(index);
  auto err = check_result.first;
  auto filt = check_result.second;
  if (err)
    return err;

  *surface = model::surfaces[filt->surface_index()]->id_;
  return 0;
}

extern "C" int openmc_circumferential_fourier_filter_set_surface(
  int32_t index, int32_t surface)
{
  auto check_result = check_circumferential_fourier_filter(index);
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
