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
# limitations under the License

"""Tests for the library of QDWH-based SVD decomposition."""
import functools

from jax import test_util as jtu
from jax.config import config
import jax.numpy as jnp
import numpy as np
import scipy.linalg as osp_linalg
from jax._src.lax import svd

from absl.testing import absltest
from absl.testing import parameterized


config.parse_flags_with_absl()
_JAX_ENABLE_X64 = config.x64_enabled

# Input matrix data type for SvdTest.
_SVD_TEST_DTYPE = np.float64 if _JAX_ENABLE_X64 else np.float32

# Machine epsilon used by SvdTest.
_SVD_TEST_EPS = jnp.finfo(_SVD_TEST_DTYPE).eps

# SvdTest relative tolerance.
_SVD_RTOL = 1E-6 if _JAX_ENABLE_X64 else 1E-2

_MAX_LOG_CONDITION_NUM = 9 if _JAX_ENABLE_X64 else 4

config.update('jax_default_matmul_precision', 'float32')


@jtu.with_config(jax_numpy_rank_promotion='allow')
class SvdTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {    # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}'.format(m, n, log_cond),
          'm': m, 'n': n, 'log_cond': log_cond}
      for m, n in zip([2, 8, 10, 20], [4, 6, 10, 18])
      for log_cond in np.linspace(1, _MAX_LOG_CONDITION_NUM, 4)))
  def testSvdWithRectangularInput(self, m, n, log_cond):
    """Tests SVD with rectangular input."""
    a = np.random.uniform(
        low=0.3, high=0.9, size=(m, n)).astype(_SVD_TEST_DTYPE)
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.linspace(cond, 1, min(m, n))
    a = (u * s) @ v
    a = a + 1j * a

    osp_linalg_fn = functools.partial(osp_linalg.svd, full_matrices=False)
    actual_u, actual_s, actual_v = svd.svd(a)

    k = min(m, n)
    if m > n:
      unitary_u = jnp.abs(actual_u.T.conj() @ actual_u)
      unitary_v = jnp.abs(actual_v.T.conj() @ actual_v)
    else:
      unitary_u = jnp.abs(actual_u @ actual_u.T.conj())
      unitary_v = jnp.abs(actual_v @ actual_v.T.conj())

    _, expected_s, _ = osp_linalg_fn(a)

    args_maker = lambda: [a]

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(svd.svd, args_maker)

    with self.subTest('Test unitary u.'):
      self.assertAllClose(np.eye(k), unitary_u, rtol=_SVD_RTOL, atol=2E-3)

    with self.subTest('Test unitary v.'):
      self.assertAllClose(np.eye(k), unitary_v, rtol=_SVD_RTOL, atol=2E-3)

    with self.subTest('Test s.'):
      self.assertAllClose(
          expected_s, jnp.real(actual_s), rtol=_SVD_RTOL, atol=1E-6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_m={}_by_n={}'.format(m, n), 'm': m, 'n': n}
      for m, n in zip([50, 6], [3, 60])))
  def testSvdWithSkinnyTallInput(self, m, n):
    """Tests SVD with skinny and tall input."""
    # Generates a skinny and tall input
    np.random.seed(1235)
    a = np.random.randn(m, n).astype(_SVD_TEST_DTYPE)
    u, s, v = svd.svd(a, is_hermitian=False)

    relative_diff = np.linalg.norm(a - (u * s) @ v) / np.linalg.norm(a)

    np.testing.assert_almost_equal(relative_diff, 1E-6, decimal=6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {   # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_r={}_log_cond={}'.format(m, r, log_cond),
          'm': m, 'r': r, 'log_cond': log_cond}
      for m, r in zip([8, 8, 8, 10], [3, 5, 7, 9])
      for log_cond in np.linspace(1, 3, 3)))
  def testSvdWithOnRankDeficientInput(self, m, r, log_cond):
    """Tests SVD with rank-deficient input."""
    a = jnp.triu(jnp.ones((m, m))).astype(_SVD_TEST_DTYPE)

    # Generates a rank-deficient input.
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.linspace(cond, 1, m)
    s = s.at[r:m].set(jnp.zeros((m-r,)))
    a = (u * s) @ v

    u, s, v = svd.svd(a, is_hermitian=False)
    diff = np.linalg.norm(a - (u * s) @ v)

    np.testing.assert_almost_equal(diff, 1E-4, decimal=2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
