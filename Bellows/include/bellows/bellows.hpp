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

#ifndef BELLOWS_BELLOWS_HPP
#define BELLOWS_BELLOWS_HPP

#include <bellows/version.hpp>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/operations.hpp>
#include <bellows/core/parameters.hpp>

#include <bellows/metrics/metrics.hpp>

#include <bellows/preprocess/encoding.hpp>
#include <bellows/preprocess/mending.hpp>
#include <bellows/preprocess/projection.hpp>
#include <bellows/preprocess/scaling.hpp>
#include <bellows/preprocess/selection.hpp>

#include <bellows/estimators/clustering.hpp>
#include <bellows/estimators/density.hpp>
#include <bellows/estimators/forest.hpp>
#include <bellows/estimators/linear.hpp>
#include <bellows/estimators/margin.hpp>
#include <bellows/estimators/proximity.hpp>

#include <bellows/boosting/booster.hpp>
#include <bellows/boosting/losses.hpp>
#include <bellows/boosting/policies.hpp>

#include <bellows/composition/chain.hpp>
#include <bellows/composition/ensemble.hpp>
#include <bellows/composition/validation.hpp>

#include <bellows/io/artifact.hpp>

#endif // BELLOWS_BELLOWS_HPP
