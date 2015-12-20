/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace {
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename dtype>
__global__ void ResizeNearestNeighborNCHW(const int nthreads, const dtype* bottom_data,
                                          const int channels, const int in_height,
                                          const int in_width, const int out_height,
                                          const int out_width, dtype* top_data) {
  const float width_scale = in_width / static_cast<float>(out_width);
  const float height_scale = in_height / static_cast<float>(out_height);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int out_x = index % out_width;
    int out_y = (index / out_width) % out_height;
    int c = (index / out_width / out_height) % channels;
    int n = index / out_width / out_height / channels;

    const dtype* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_x = min(static_cast<int>(floorf(out_x * width_scale)), in_width - 1);
    const int in_y = min(static_cast<int>(floorf(out_y * height_scale)), in_height - 1);
    int idx = c * in_height * in_width + in_y * in_width + in_x;
    top_data[index] = bottom_data_n[idx];
  }
}

template <typename dtype>
__global__ void ResizeNearestNeighborBackwardNHWC(
                                   const int nthreads, const dtype* bottom_data,
                                   const int channels, const int in_height,
                                   const int in_width, const int out_height,
                                   const int out_width, dtype* top_data) {
  const float width_scale = in_width / static_cast<float>(out_width);
  const float height_scale = in_height / static_cast<float>(out_height);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int out_x = index % out_width;
    int out_y = (index / out_width) % out_height;
    int c = (index / out_width / out_height) % channels;
    int n = index / out_width / out_height / channels;

    const dtype* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_x = min(static_cast<int>(floorf(out_x * width_scale)), in_width - 1);
    const int in_y = min(static_cast<int>(floorf(out_y * height_scale)), in_height - 1);
    int idx = c * in_height * in_width + in_y * in_width + in_x;
    top_data[index] = bottom_data_n[idx];
  }
}

#undef CUDA_1D_KERNEL_LOOP
}  // namespace

bool ResizeNearestNeighbor(const float* bottom_data, const int batch,
                           const int in_height, const int in_width,
                           const int channels, const int out_height,
                           const int out_width, float* top_data,
                           const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * out_height * out_width;

  ResizeNearestNeighborNCHW<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, in_height, in_width, channels, out_height,
      out_width, top_data);
  return d.ok();
}

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
