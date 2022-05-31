# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import numpy as np
from typing import (Callable, Sequence, Tuple, Union, Mapping, Optional, List,
                    Dict, NamedTuple, Set)

from jax import core
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.config import config
from jax.interpreters import pxla, xla
from jax._src.util import prod, safe_zip, cache
from jax._src.api import device_put
from jax.interpreters.pxla import PartitionSpec as P
from jax.experimental.sharding import Sharding, SingleDeviceSharding

Shape = Tuple[int, ...]
Device = xc.Device
DeviceArray = xc.Buffer


class Array:
  aval: core.AbstractValue
  sticky: bool
  sharding: Sharding

  def __init__(self, aval: core.AbstractValue, sharding: Sharding,
               arrays: Sequence[DeviceArray], sticky: bool):
    self._aval = aval
    self._sharding = sharding
    self._arrays = arrays
    self._sticky = sticky

    # Rearrange arrays based on the device assignment.
    if not isinstance(self.sharding, SingleDeviceSharding):
      device_to_buffer = {db.device().id: db for db in self._arrays}
      self._arrays = [device_to_buffer[device.id]
                      for device in self.sharding._addressable_device_assignment()]

  @property
  def shape(self) -> Shape:
    return self._aval.shape

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return prod(self.shape)

  @property
  def dtype(self):
    return self._aval.dtype

  @property
  def sharding(self):
    return self._sharding

  def is_fully_addressable(self) -> bool:
    return len(self.sharding.device_set) == len(self.sharding.addressable_devices)

  def __repr__(self):
    return (f'Array(shape={self.shape}, dtype={self.dtype}, '
            f'sharding={self.sharding}, arrays={self._arrays}, '
            f'sticky={self._sticky}')

  @cache()
  def explode_into_addressable_arrays(self) -> Sequence[Array]:
    # Wrap the device arrays in `Array`.
    return [Array(da.aval, SingleDeviceSharding(da.device()), [da], sticky=True)
            for da in self._arrays]

  def copy_to_host_async(self):
    for arr in self.explode_into_addressable_arrays():
      device = arr.sharding.device_set[0]
      if self.sharding.device_replica_id_map(self.shape)[device] == 0:
        arr._arrays[0].copy_to_host_async()

  def _value(self):
    if not self.is_fully_addressable():
      raise RuntimeError("Fetching value for `jax.Array` that spans "
                         "non-addressable devices is not possible. You can use "
                         "`jax.experimental.multihost_utils.process_allgather` "
                         "for this use case.")
    self.copy_to_host_async()
    npy_value = np.empty(self.shape, self.dtype)
    for arr in self.explode_into_addressable_arrays():
      device = arr.sharding.device_set[0]
      if self.sharding.device_replica_id_map(self.shape)[device] == 0:
        npy_value[self.sharding.device_indices(device, self.shape)] = arr._arrays[0].to_py()
    return npy_value


def make_array_from_callback(shape: Shape, sharding: Sharding,
                             data_callback) -> Array:
  dbs = [
      device_put(data_callback(sharding.device_indices(device, shape)), device)
      for device in sharding.addressable_devices
  ]
  return assemble_array(shape, sharding, dbs)


def assemble_array(shape: Shape, sharding: Sharding,
                   arrays: Sequence[DeviceArray]) -> Array:
  aval = core.ShapedArray(shape, arrays[0].dtype)
  return Array(aval, sharding, arrays, sticky=True)
