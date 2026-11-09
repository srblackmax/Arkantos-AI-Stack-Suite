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

#ifndef SURYA_IMGPROC_MORPHOLOGY_HPP
#define SURYA_IMGPROC_MORPHOLOGY_HPP

#include <surya/core.hpp>

namespace surya::inline v1::imgproc
{

/**
 * @brief Structuring element geometry for morphological operators.
 */
enum class Structuring_Element : uint8_t
{
  Rect = 0,
  Cross = 1,
  Ellipse = 2
};

/**
 * @brief Morphological operation kind.
 */
enum class Morph_Op : uint8_t
{
  Erode = 0,
  Dilate = 1,
  Open = 2,
  Close = 3,
  Gradient = 4,
  Top_Hat = 5,
  Black_Hat = 6
};

/**
 * @brief Erode a 1-channel uint8 image using a KWIDTH x KHEIGHT structuring
 *        element.  Uses separable AVX2 scanline min for rectangular kernels.
 */
SURYA_API core::Result<core::Image_U8C1>
erode (core::Const_View_U8C1 src,
       size_t kwidth,
       size_t kheight,
       Structuring_Element shape = Structuring_Element::Rect);

/**
 * @brief Dilate a 1-channel uint8 image using a KWIDTH x KHEIGHT structuring
 *        element.  Uses separable AVX2 scanline max for rectangular kernels.
 */
SURYA_API core::Result<core::Image_U8C1>
dilate (core::Const_View_U8C1 src,
        size_t kwidth,
        size_t kheight,
        Structuring_Element shape = Structuring_Element::Rect);

/**
 * @brief Apply a composite morphological operation (Open, Close, Gradient,
 *        Top_Hat, Black_Hat, Erode, or Dilate).
 */
SURYA_API core::Result<core::Image_U8C1>
morphology_ex (core::Const_View_U8C1 src,
               Morph_Op op,
               size_t kwidth,
               size_t kheight,
               Structuring_Element shape = Structuring_Element::Rect);

} /* namespace surya::inline v1::imgproc */

#endif /* SURYA_IMGPROC_MORPHOLOGY_HPP */
