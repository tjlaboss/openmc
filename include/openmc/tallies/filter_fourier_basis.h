#ifndef OPENMC_TALLIES_FILTER_FOURIER_BASIS_H
#define OPENMC_TALLIES_FILTER_FOURIER_BASIS_H

#include <cmath>
#include <string>

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Fourier basis evaluation shared by spatial and circumferential filters.
//==============================================================================

//! Fill wgt (size 2*order+1) with Fourier weights for x in [0, 1).
inline void fourier_weights(double x, int order, vector<double>& wgt)
{
  wgt[0] = 1.0;
  for (int n = 1; n <= order; ++n) {
    double arg = 2.0 * PI * n * x;
    wgt[2 * n - 1] = std::cos(arg);
    wgt[2 * n] = std::sin(arg);
  }
}

inline std::string fourier_bin_label(int bin)
{
  if (bin == 0)
    return "a0 (constant)";
  if (bin % 2 == 1)
    return fmt::format("a{} (cos)", (bin + 1) / 2);
  return fmt::format("b{} (sin)", bin / 2);
}

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_FOURIER_BASIS_H
