/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef HYPER_MATH_HPP
#define HYPER_MATH_HPP

/* Core Memory, Bitmask, PRNG, Parallel Thread Pool, and Diagnostics.  */
#include "hyper_math/core/config.hpp"
#include "hyper_math/core/memory.hpp"
#include "hyper_math/core/bitmask.hpp"
#include "hyper_math/core/prng.hpp"
#include "hyper_math/core/parallel.hpp"

/* Pillar 1: Hyper-Lattice N-Dimensional Tensor Engine (NumPy layer).  */
#include "hyper_math/tensor/lattice.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"
#include "hyper_math/tensor/ufunc.hpp"
#include "hyper_math/tensor/reductions.hpp"
#include "hyper_math/tensor/manipulation.hpp"

/* Pillar 2: Scientific Numerical Continuum (SciPy layer).  */
#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/scientific/fft.hpp"
#include "hyper_math/scientific/integrate.hpp"
#include "hyper_math/scientific/optimize.hpp"
#include "hyper_math/scientific/stats.hpp"
#include "hyper_math/scientific/interpolate.hpp"
#include "hyper_math/scientific/signal.hpp"
#include "hyper_math/scientific/sparse.hpp"
#include "hyper_math/scientific/spatial.hpp"
#include "hyper_math/scientific/cluster.hpp"
#include "hyper_math/scientific/autograd.hpp"

/* Pillar 3: Dual-Bitmask Columnar & Lazy Relational Engine (Pandas/Polars).  */
#include "hyper_math/tabular/index.hpp"
#include "hyper_math/tabular/series.hpp"
#include "hyper_math/tabular/data_frame.hpp"
#include "hyper_math/tabular/lazy.hpp"

#endif /* HYPER_MATH_HPP */
