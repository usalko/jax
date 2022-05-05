/* Copyright 2021 Google LLC

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

#include "jaxlib/hip/hip_prng_kernels.h"

#include "jaxlib/hip/hip_gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {
namespace {

absl::Status HipThreeFry2x32_(hipStream_t stream, void** buffers,
                              const char* opaque, std::size_t opaque_len) {
  auto s = UnpackDescriptor<ThreeFry2x32Descriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  LaunchThreeFry2x32Kernel(stream, buffers, **s);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipGetLastError()));
  return absl::OkStatus();
}

}  // namespace

void HipThreeFry2x32(hipStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = HipThreeFry2x32_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    absl::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

}  // namespace jax
