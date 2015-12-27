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
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

template <typename dtype>
__global__ void ResizeNearestNeighborNHWC(const int nthreads, const dtype* bottom_data,
                                          const int channels, const int in_height,
                                          const int in_width, const int out_height,
                                          const int out_width, dtype* top_data) {
  const float width_scale = in_width / static_cast<float>(out_width);
  const float height_scale = in_height / static_cast<float>(out_height);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const dtype* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_x = min(static_cast<int>(floorf(out_x * width_scale)), in_width - 1);
    const int in_y = min(static_cast<int>(floorf(out_y * height_scale)), in_height - 1);
    int idx = c * in_height * in_width + in_y * in_width + in_x;
    top_data[index] = bottom_data_n[idx];
  }
}

template <typename dtype>
__global__ void SetZero(const int nthreads, dtype* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) { *(bottom_diff + index) = dtype(0); }
}

template <typename dtype>
__global__ void ResizeNearestNeighborBackwardNHWC(
                                   const int nthreads, const dtype* top_diff,
                                   const int channels, const int in_height,
                                   const int in_width, const int out_height,
                                   const int out_width, dtype* bottom_diff) {
  const float width_scale = out_width / static_cast<float>(in_width);
  const float height_scale = out_height / static_cast<float>(in_height);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int in_x = n % in_width;
    n /= in_width;
    int in_y = n % in_height;
    n /= in_height;

    const dtype* top_diff_n = top_diff + n * channels * out_height * out_width;
    const int out_x = min(static_cast<int>(floorf(in_x * width_scale)), out_width - 1);
    const int out_y = min(static_cast<int>(floorf(in_y * height_scale)), out_height - 1);
    int idx = c * out_height * out_width + out_y * out_width + out_x;
    bottom_diff[index] += top_diff_n[idx];
  }
}

}  // namespace

bool ResizeNearestNeighbor(const float* bottom_data, const int batch,
                           const int in_height, const int in_width,
                           const int channels, const int out_height,
                           const int out_width, float* top_data,
                           const Eigen::GpuDevice& d) {
  const int output_size = batch * channels * out_height * out_width;
  CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);

  ResizeNearestNeighborNHWC<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      output_size, bottom_data, in_height, in_width, channels, out_height,
      out_width, top_data);
  return d.ok();
}

bool ResizeNearestNeighborBackward(const float* top_diff, const int batch,
                                   const int in_height, const int in_width,
                                   const int channels, const int out_height,
                                   const int out_width, float* bottom_diff,
                                   const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int input_size = batch * channels * in_height * in_width;
  const int output_size = batch * channels * out_height * out_width;

  SetZero<<<(input_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(input_size, bottom_diff);

  ResizeNearestNeighborBackwardNHWC<<<(input_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, in_height, in_width, channels, out_height,
      out_width, bottom_diff);
  return d.ok();
}
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
