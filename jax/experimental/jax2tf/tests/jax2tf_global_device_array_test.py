# Copyright 2020 Google LLC
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
"""Tests for jax2tf with GDA input."""
import numpy as np
from functools import partial

from absl import logging
from absl.testing import absltest

import tensorflow as tf
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax._src.util import prod
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
# pylint: disable=unused-import
from jax.experimental.pjit import pjit, FROM_GDA
# pylint: enable=unused-import
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import PartitionSpec as P

config.parse_flags_with_absl()


class Jax2TfTest(tf_test_util.JaxToTfTestCase):

  def test_global_device_array(self):

    def create_gda(global_shape, global_mesh, mesh_axes, global_data=None):
      if global_data is None:
        global_data = np.arange(prod(global_shape)).reshape(global_shape)
      return GlobalDeviceArray.from_callback(
          global_shape, global_mesh, mesh_axes,
          lambda idx: global_data[idx]), global_data

    @partial(
        pjit,
        in_axis_resources=(P("y", "x"), FROM_GDA),
        out_axis_resources=None)
    def jax_func(x, gda):
      result = jnp.matmul(x, gda)
      return result

    def fun(x):
      return jax_func(x, gda)

    global_mesh = jtu.create_global_mesh((4, 2), ("x", "y"))
    mesh_axes = P(("x", "y"))
    input_shape = (8, 2)
    gda, _ = create_gda(input_shape, global_mesh, mesh_axes)
    x = np.arange(16).reshape(2, 8)
    with global_mesh:
      logging.info("Run TF converted func.")
      tf_fun = tf.function(
          jax2tf.convert(fun, enable_xla=True),
          jit_compile=True,
      )
      logging.info("result = %s", tf_fun(x))
      # self.ConvertAndCompare(fun, x)


if __name__ == "__main__":
  absltest.main()
