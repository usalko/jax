# Copyright 2022 Google LLC
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
import io
import textwrap
import unittest

from typing import IO, Sequence, Tuple

from absl.testing import absltest
import jax
from jax.config import config
from jax._src import debugger
from jax._src import lib as jaxlib
from jax._src import test_util as jtu
import jax.numpy as jnp

config.parse_flags_with_absl()

def make_fake_stdin_stdout(commands: Sequence[str]) -> Tuple[IO[str], io.StringIO]:
  fake_stdin = io.StringIO()
  fake_stdin.truncate(0)
  for command in commands:
    fake_stdin.write(command + "\n")
  fake_stdin.seek(0)
  return fake_stdin, io.StringIO()

def _format_multiline(text):
  return textwrap.dedent(text).lstrip()

prev_xla_flags = None

def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()

# TODO(sharadmv): remove jaxlib guards for GPU tests when jaxlib minimum
#                 version is >= 0.3.11
# TODO(sharadmv): remove jaxlib guards for TPU tests when jaxlib minimum
#                 version is >= 0.3.15
disabled_backends = []
if jaxlib.version < (0, 3, 11):
  disabled_backends.append("gpu")
if jaxlib.version < (0, 3, 15):
  disabled_backends.append("tpu")

class CliDebuggerTest(jtu.JaxTestCase):

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_eof(self):
    stdin, stdout = make_fake_stdin_stdout([])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    with self.assertRaises(SystemExit):
      f(2.)
      jax.effects_barrier()

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_continue(self):
    stdin, stdout = make_fake_stdin_stdout(["c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    f(2.)
    jax.effects_barrier()
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) """)
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_print_value(self):
    stdin, stdout = make_fake_stdin_stdout(["p x", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) DeviceArray(2., dtype=float32)
    (jaxdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_print_value_in_jit(self):
    stdin, stdout = make_fake_stdin_stdout(["p x", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) array(2., dtype=float32)
    (jaxdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_print_multiple_values(self):
    stdin, stdout = make_fake_stdin_stdout(["p x, y", "c"])

    @jax.jit
    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) (array(2., dtype=float32), array(3., dtype=float32))
    (jaxdb) """)
    f(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_print_context(self):
    stdin, stdout = make_fake_stdin_stdout(["l", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    f(2.)
    jax.effects_barrier()
    expected = _format_multiline(r"""
    Entering jaxdb:
    \(jaxdb\) > .*debugger_test\.py\([0-9]+\)
            @jax\.jit
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout\)
              return y
    .*
    \(jaxdb\) """)
    self.assertRegex(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_print_backtrace(self):
    stdin, stdout = make_fake_stdin_stdout(["bt", "c"])

    @jax.jit
    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y
    expected = _format_multiline(r"""
    Entering jaxdb:.*
    \(jaxdb\) Traceback:.*
    """)
    f(2.)
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_can_work_with_multiple_stack_frames(self):
    stdin, stdout = make_fake_stdin_stdout(["l", "u", "p x", "d", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y

    @jax.jit
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jaxdb:
    \(jaxdb\) > .*debugger_test\.py\([0-9]+\)
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout\)
              return y
    .*
    \(jaxdb\) > .*debugger_test\.py\([0-9]+\).*
            @jax\.jit
            def g\(x\):
    ->        y = f\(x\)
              return jnp\.exp\(y\)
    .*
    \(jaxdb\) array\(2\., dtype=float32\)
    \(jaxdb\) > .*debugger_test\.py\([0-9]+\)
            def f\(x\):
              y = jnp\.sin\(x\)
    ->        debugger\.breakpoint\(stdin=stdin, stdout=stdout\)
              return y
    .*
    \(jaxdb\) """)
    g(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_can_use_multiple_breakpoints(self):
    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])

    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=True)
      return y

    @jax.jit
    def g(x):
      y = f(x) * 2.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=True)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) array(3., dtype=float32)
    (jaxdb) Entering jaxdb:
    (jaxdb) array(6., dtype=float32)
    (jaxdb) """)
    g(jnp.array(2., jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_works_with_vmap(self):
    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])
    # On TPU, the breakpoints can be reordered inside of vmap but can be fixed
    # by ordering sends.
    # TODO(sharadmv): change back to ordered = False when sends are ordered
    ordered = jax.default_backend == "tpu"

    def f(x):
      y = x + 1.
      debugger.breakpoint(stdin=stdin, stdout=stdout, ordered=ordered)
      return 2. * y

    @jax.jit
    @jax.vmap
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jaxdb:
    (jaxdb) array(1., dtype=float32)
    (jaxdb) Entering jaxdb:
    (jaxdb) array(2., dtype=float32)
    (jaxdb) """)
    g(jnp.arange(2., dtype=jnp.float32))
    jax.effects_barrier()
    self.assertEqual(stdout.getvalue(), expected)

  @jtu.skip_on_devices(*disabled_backends)
  def test_debugger_works_with_pmap(self):
    if jax.local_device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")
    stdin, stdout = make_fake_stdin_stdout(["p y", "c", "p y", "c"])

    def f(x):
      y = jnp.sin(x)
      debugger.breakpoint(stdin=stdin, stdout=stdout)
      return y

    @jax.pmap
    def g(x):
      y = f(x)
      return jnp.exp(y)
    expected = _format_multiline(r"""
    Entering jaxdb:
    \(jaxdb\) array\(.*, dtype=float32\)
    \(jaxdb\) Entering jaxdb:
    \(jaxdb\) array\(.*, dtype=float32\)
    \(jaxdb\) """)
    g(jnp.arange(2., dtype=jnp.float32))
    jax.effects_barrier()
    self.assertRegex(stdout.getvalue(), expected)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
