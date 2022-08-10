# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TODO(krishnahari): DO NOT SUBMIT without one-line documentation for program_eviction.

TODO(krishnahari): DO NOT SUBMIT without a detailed description of
program_eviction.
"""

import contextlib
from jax.lib import xla_bridge as xb
from jax._src.lib import program_eviction_lib


@contextlib.contextmanager
def loaded_program_eviction_scope():
  """Indicates that loaded programs have to be evicted at the end of the current scope."""
  state = program_eviction_lib.thread_local_state()
  prev_evict_loaded_programs = state.evict_loaded_programs
  prev_eviction_identifier = state.eviction_identifier
  state.evict_loaded_programs = True
  state.generate_eviction_identifier()
  try:
    yield
  finally:
    xb.get_backend().evict_loaded_programs_from_live_executables()
    state.evict_loaded_programs = prev_evict_loaded_programs
    state.eviction_identifier = prev_eviction_identifier
