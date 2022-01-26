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
"""GlobalDeviceArray serialization and deserialization."""

import asyncio
import os
import re
import threading
import time
from functools import partial
from typing import Callable, NamedTuple, Optional
from absl import logging

import jax
from jax._src.util import prod
from jax.experimental import global_device_array as gda
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
import tensorflow.compat.v2 as tf


async def create_async_gda_from_callback(
    global_shape: gda.Shape,
    global_mesh: Mesh,
    mesh_axes: gda.MeshAxes,
    data_callback: Callable[[gda.Index], asyncio.Future],
):
  global_idx_rid = gda.get_shard_indices_replica_ids(
      global_shape, global_mesh, mesh_axes)
  local_devices = global_mesh.local_devices
  future_arrays = [data_callback(global_idx_rid[d][0])
                   for d in local_devices]
  # Pause here and come back to `from_async_callback()` when future_arrays are
  # ready. device_put cannot happen with future_arrays.
  local_arrays = await asyncio.gather(*future_arrays)

  dbs = [jax.device_put(array, device)
         for array, device in zip(local_arrays, local_devices)]
  return gda.GlobalDeviceArray(global_shape, global_mesh, mesh_axes, dbs,
                               gda._GdaFastPathArgs(global_idx_rid, local_devices))


def _get_metadata(gda):
  if gda.dtype == jnp.bfloat16:
    # Tensorstore uses 'bfloat16', not '<V2'.
    dtype = 'bfloat16'
  else:
    dtype = np.dtype(gda.dtype).str

  return {
      'compressor': {
          'id': 'gzip'
      },
      'shape': gda.shape,
      'chunks': np.array(np.maximum(1, gda.local_data(0).shape)),
      'dtype': dtype,
  }


def _spec_has_metadata(tree):
  if not isinstance(tree, dict):
    return False
  return 'metadata' in tree or any(
      _spec_has_metadata(subtree) for _, subtree in tree.items())


def get_tensorstore_spec(ckpt_path: str):
  spec = {'driver': 'zarr', 'kvstore': {}}

  if ckpt_path.startswith('gs://'):
    m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
    if m is None:
      raise ValueError('The ckpt_path should contain the bucket name and the '
                       f'file path inside the bucket. Got: {ckpt_path}')
    gcs_bucket = m.group(1)
    path_without_bucket = m.group(2)
    spec['kvstore'] = {'driver': 'gcs', 'bucket': gcs_bucket,
                       'path': path_without_bucket}
  else:
    spec['kvstore'] = {'driver': 'file', 'path': ckpt_path}
  return spec


class AsyncInfo(NamedTuple):
  temp_ckpt_dir: str
  final_ckpt_dir: str


class AsyncCheckpointManager:
  _done: bool

  def __init__(self):
    logging.info('SerdeManager initialized.')

    self.commit_futures = None
    self.context = ts.Context({'file_io_concurrency': {'limit': 128}})
    self.t = None
    self.exceptions = []

  def wait_for_thread_to_finish(self):
    if self.t is not None:
      self.t.join()
      self.t = None

  def _thread_func(self, temp_ckpt_dir, final_ckpt_dir):
    try:
      for future in self.commit_futures:
        for f in future:
          f.result()
      logging.info('Commit has completed.')

      current_process = jax.process_index()
      lockfiles_dir = os.path.join(temp_ckpt_dir, 'lockfiles')
      all_lockfile_paths = [
          os.path.join(lockfiles_dir, f'lockfile_{p}')
          for p in range(jax.process_count())
      ]
      current_process_lockfile = os.path.join(lockfiles_dir,
                                              f'lockfile_{current_process}')

      while True:
        if tf.io.gfile.exists(current_process_lockfile):
          break
        else:
          logging.info('Waiting for current process %s lockfile to appear.',
                       current_process)
          time.sleep(60)

      # This while loop will not trigger until all commits have finished.
      while not self._done:
        if tf.io.gfile.exists(current_process_lockfile):
          tf.io.gfile.remove(current_process_lockfile)
          logging.info('Lockfile removed for process %s', current_process)

        # Mark as done when no lockfiles exist.
        if current_process == 0 and all(not tf.io.gfile.exists(f) for f in all_lockfile_paths):
          self._done = True

          tf.io.gfile.remove(lockfiles_dir)
          logging.info('Lockfiles directory removed.')

          logging.info('Renaming %s to %s', temp_ckpt_dir, final_ckpt_dir)
          tf.io.gfile.rename(temp_ckpt_dir, final_ckpt_dir)
          logging.info('Finished saving GDA checkpoint to `%s`.', final_ckpt_dir)

        elif current_process != 0 and not tf.io.gfile.exists(current_process_lockfile):
          self._done = True

        if not self._done:
          logging.info('Thread sleeping for 60 seconds.')
          time.sleep(60)

    except Exception as e:
      self.exceptions.append(e)

  def _start_commit_thread(self, temp_ckpt_dir, final_ckpt_dir):
    self._done = False
    self.t = threading.Thread(
        target=self._thread_func, args=(temp_ckpt_dir, final_ckpt_dir))
    self.t.start()

  def _write_lockfiles(self, temp_ckpt_dir):
    # Process 0 writes lock files for all processes.
    if jax.process_index() == 0:
      lockfiles_dir = os.path.join(temp_ckpt_dir, 'lockfiles')
      tf.io.gfile.mkdir(lockfiles_dir)

      for p in range(jax.process_count()):
        with tf.io.gfile.GFile(
            os.path.join(lockfiles_dir, f'lockfile_{p}'), mode='w') as f:
          f.write('File to track if all chunks have been written.')
      logging.info('Lock files for all processes have been written by process 0.')

  def run_serialization(self, gdas, tensorstore_specs, *,
                        async_info: Optional[AsyncInfo] = None):
    for e in self.exceptions:
      raise e

    if async_info is not None:
      if self.t is not None and self.t.is_alive():
        logging.info('Waiting for thread to finish -- run_serialization')
        self.wait_for_thread_to_finish()
      self._write_lockfiles(async_info.temp_ckpt_dir)

    self.commit_futures = [[] for _ in range(len(tensorstore_specs))]

    async def _run_serializer():
      future_writer = jax.tree_map(
          partial(self.async_serialize, async_info=async_info), gdas,
          tensorstore_specs, self.commit_futures)
      return await asyncio.gather(*future_writer)
    asyncio.run(_run_serializer())

    if async_info is not None:
      self._start_commit_thread(async_info.temp_ckpt_dir, async_info.final_ckpt_dir)

  async def async_serialize(
      self, gda_inp: gda.GlobalDeviceArray, tensorstore_spec,
      commit_future=None, async_info: Optional[AsyncInfo] = None):
    # 'metadata' may not be present at the top level (for example, if we are using
    # a 'cast' driver).
    if not _spec_has_metadata(tensorstore_spec):
      tensorstore_spec['metadata'] = _get_metadata(gda_inp)

    t = await ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=self.context)

    async def _write_array(shard):
      if shard.replica_id == 0:
        write_future = t[shard.index].write(shard.data)
        if async_info is not None:
          commit_future.append(write_future.commit)
          await write_future.copy
        else:
          await write_future.commit

    future_write_state = jax.tree_util.tree_map(_write_array,
                                                gda_inp.local_shards)
    return await asyncio.gather(*future_write_state)


  async def async_deserialize(self, mesh, mesh_axes, tensorstore_spec,
                              global_shape=None, dtype=None):
    t = ts.open(ts.Spec(tensorstore_spec), open=True, context=self.context).result()
    shape = t.shape if global_shape is None else global_shape
    requires_padding = prod(shape) > prod(t.shape)

    if requires_padding:
      new_shard_shape = gda.get_shard_shape(shape, mesh, mesh_axes)

    async def cb(index):
      if requires_padding:
        # This is needed because the shape the array was saved with is smaller
        # than the requested shape of the array in which it will be reloaded. So
        # the extra values will be filled with 0s.
        out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
        requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
        restricted_domain = t.domain.intersect(requested_domain)
        await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(t[restricted_domain])
      else:
        out = await t[index].read()

      if dtype is not None:
        # Cast while reloading on process to avoid 2 copies on device if the
        # casting is done on device.
        return out.astype(dtype)
      return out

    return await create_async_gda_from_callback(shape, mesh, mesh_axes, cb)


  def run_deserialization(self, global_meshes, mesh_axes, tensorstore_specs,
                          global_shapes=None, dtypes=None):
    async def _run_deserializer():
      future_gdas = jax.tree_map(
          self.async_deserialize, global_meshes, mesh_axes, tensorstore_specs,
          [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
          [None] * len(tensorstore_specs) if dtypes is None else dtypes)
      return await asyncio.gather(*future_gdas)
    return asyncio.run(_run_deserializer())
