# Surya — High-Performance Native C++ Computer Vision Library

> Surya is an independent GPLv3 C++ computer vision library.  It follows the
> GNU Coding Standards as a style convention; it is not part of the GNU Project
> and is not affiliated with the Free Software Foundation.

**Surya** is a clean-room, modern C++23 Computer Vision library designed from
scratch for low-latency, high-throughput production environments.  It provides
capabilities that architecturally rival OpenCV, Torchvision, Albumentations,
and Ultralytics without a single line of Python, Cython, or legacy reference-
counted wrapper code.

---

## 1. Architectural Highlights & From-Scratch Design Philosophy

1. **Deterministic Ownership & 64-Byte Cache-Line Row Alignment (`surya/core`)**
   - Unlike OpenCV's `cv::Mat` (which couples views and owning buffers via
     atomic reference counts and unaligned default row strides), Surya
     strictly separates owning `surya::core::Image<T, C, Layout>` buffers from
     non-owning, zero-overhead 32-byte `surya::core::Image_View<T, C, Layout>`
     descriptors.
   - Every scanline of an owning `Image` is padded to a 64-byte boundary
     (`SURYA_CACHELINE_BYTES = 64`), guaranteeing aligned AVX2/AVX-512 loads
     at the start of every row and eliminating cross-row cache-line splits.
   - Includes a thread-safe `Aligned_Buffer_Pool` and monotonic
     `Arena_Allocator` to eliminate heap allocations in real-time frame loops.

2. **SIMD-Dispatched & Cache-Blocked Image Processing (`surya/imgproc`)**
   - Separable 2D convolutions (Gaussian, Sobel, Scharr, box filter),
     fixed-point AVX2 color space conversions (RGB/BGR $\leftrightarrow$
     Grayscale, HSV, YCbCr, CIE Lab), precomputed-coefficient bilinear/bicubic
     resizing, inverse-mapped affine/perspective warps, van Herk / separable
     mathematical morphology, Otsu/adaptive thresholding, Canny edge detection,
     connected components, and integral images.

3. **Zero-Copy & Direct-Row-Table Image Codecs (`surya/io`)**
   - Zero-copy `mmap(2)` PNM (PGM/PPM/PAM), BMP, and QOI codecs plus native
     scanline-direct `libjpeg-turbo`, `libpng`, `libwebp`, and `libtiff`
     codecs that decode directly into 64-byte row-aligned `Image` buffers,
     alongside deterministic `Frame_Stream` video ingestion abstractions.

4. **Algebraically Fused Augmentation Engine (`surya/augment`)**
   - Inspired by Albumentations' batch pipeline ergonomics, but architected
     natively in C++: consecutive geometric stages (flip, rotate, scale, shear,
     shift, perspective, crop) are fused into a single $3 \times 3$ homography
     matrix $H_{\text{total}} = H_k \cdots H_1$ so pixels are resampled only
     once, while consecutive photometric operations (brightness, contrast,
     gamma, solarize, posterize) fuse into a single 768-byte ($3 \times 256$)
     L1-resident Lookup Table (LUT).  Includes batch-level `MixUp`, `CutMix`,
     and `Mosaic4` with synchronized bounding-box, keypoint, and mask updates.

5. **Deep Learning Vision Bridge (`surya/nn`)**
   - Single-pass SIMD `HWC -> NCHW` de-interleaving + `uint8 -> float32`
     normalization bridge (`(px * scale - mean) / std`) into **Basalt**
     (`basalt::Tensor`), native vision backbone blocks (`Conv_BN_SiLU`,
     `Bottleneck`, `C2f_Block`, `SPPF_Block`, `Residual_Stage`), and zero-copy
     POSIX `mmap` model loading via **Obsidian** (`obsidian::SafeTensors_File`).

6. **Native Detection, Segmentation, Pose & Tracking (`surya/detection`)**
   - Reversible `Letterbox_Transform`, anchor-free Distribution Focal Loss
     (DFL) 16-bin integral box decoding, vectorized IoU / GIoU / DIoU / CIoU,
     class-aware & agnostic NMS / Soft-NMS, prototype instance mask assembly,
     OKS pose keypoint decoding, and `Byte_Tracker` multi-object Kalman
     tracking powered by **Hyper Math** (`hyper_math::scientific::linalg`).

7. **Classical Computer Vision & ML Pipelines (`surya/classical`)**
   - FAST-9 & Harris/Shi-Tomasi corner detection, ORB keypoints with 256-bit
     rotated BRIEF descriptors and hardware POPCNT Hamming matching, HOG
     descriptor extraction, RANSAC homography estimation via **Hyper Math**
     SVD (`linalg::svd`), pyramidal Lucas-Kanade optical flow, template
     matching, and classical ML vision pipelines delegating to **Bellows**
     (`bellows::max_margin_classifier` SVM, `bellows::k_means_clusterer`
     Bag-of-Visual-Words & color quantization, and
     `bellows::random_forest_classifier`).

---

## 2. Custom Library Integration Summary

| Custom Library | Role in Surya | Consuming Modules |
| :--- | :--- | :--- |
| **Basalt Deep Learning** | Differentiable `Tensor`, `ops`, and `nn::Module` graph execution | `surya/nn`, `surya/detection` |
| **Obsidian** | Zero-copy `mmap` `.safetensors` checkpoint loading and serialization | `surya/nn`, `surya/detection` |
| **Hyper Math** | Numerical linear algebra (`lu_factor`, `lu_solve`, `svd`, `lstsq`) | `surya/detection`, `surya/classical` |
| **Bellows** | Classical ML (`max_margin_classifier`, `k_means_clusterer`, `random_forest_classifier`) | `surya/classical` |

---

## 3. Building and Verifying Surya

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSURYA_BUILD_TESTS=ON \
  -DSURYA_BUILD_BENCH=ON
cmake --build build -j20
ctest --test-dir build --output-on-failure
```

To run with AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSURYA_BUILD_TESTS=ON \
  -DSURYA_USE_SANITIZERS=ON
cmake --build build-asan -j20
ctest --test-dir build-asan --output-on-failure
```

---

## 4. License

Surya is free software licensed under the **GNU General Public License v3.0 or
later** (`GPL-3.0-or-later`).  See [LICENSE](LICENSE) for the full text.
