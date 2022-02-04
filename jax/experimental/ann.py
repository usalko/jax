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

"""ANN (Approximate Nearest Neighbor) computes top-k with a configurable recall rate.

This package only optimizes the TPU backend. For other device types it fallbacks
to sort and slice.

Usage::

  from jax.experimental import ann

  # MIPS := maximal inner product search
  # Inputs:
  #   qy: f32[qy_size, feature_dim]
  #   db: f32[db_size, feature_dim]
  #
  # Returns:
  #   (f32[qy_size, k], i32[qy_size, k])
  @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
  def mips(qy, db, k=10, recall_target=0.95):
    dists = jax.lax.dot(qy, db.transpose())
    # Computes max_k along the last dimension
    # returns (f32[qy_size, k], i32[qy_size, k])
    return ann.approx_max_k(dists, k=k, recall_target=recall_target)

  # Obtains the top-10 dot products and its offsets in db.
  dot_products, neighbors = mips(qy, db, k=10)
  # Computes the recall against the true neighbors.
  recall = ann.ann_recall(neighbors, true_neighbors)

  # Multi-core example
  # Inputs:
  #   qy: f32[num_devices, qy_size, feature_dim]
  #   db: f32[num_devices, per_device_db_size, feature_dim]
  #   db_offset: i32[num_devices]
  #
  # Returns:
  #   (f32[qy_size, num_devices, k], i32[qy_size, num_devices, k])
  @functools.partial(
      jax.pmap,
      # static args: db_size, k, recall_target
      static_broadcasted_argnums=[3, 4, 5],
      out_axes=(1, 1))
  def pmap_mips(qy, db, db_offset, db_size, k, recall_target):
    dists = jax.lax.dot(qy, db.transpose())
    dists, neighbors = ann.approx_max_k(
        dists, k=k, recall_target=recall_target,
        reduction_input_size_override=db_size)
    return (dists, neighbors + db_offset)

  # i32[qy_size, num_devices, k]
  pmap_neighbors = pmap_mips(qy, db, db_offset, db_size, 10, 0.95)[1]
  # i32[qy_size, num_devices * k]
  neighbors = jax.lax.collapse(pmap_neighbors, start_dimension=1, stop_dimension=3)

Todos::

  * On host top-k aggregation
  * Inaccurate but fast differentiation

"""

from functools import partial
from typing import (Any, Tuple)

import numpy as np
from jax import lax, core
from jax._src.lib import xla_client as xc
from jax._src import ad_util, dtypes

from jax.interpreters import ad, xla, batching

Array = Any


def approx_max_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95,
                 reduction_input_size_override: int = -1,
                 aggregate_to_topk: bool = True) -> Tuple[Array, Array]:
  """Returns max ``k`` values and their indices of the ``operand``.

  Args:
    operand : Array to search for max-k.
    k : Specifies the number of max-k.
    reduction_dimension : Integer dimension along which to search. Default: -1.
    recall_target : Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by operands[reduction_dim] for evaluating the recall.
      This option is useful when the given operand is only a subset of the
      overall computation in SPMD or distributed pipelines, where the true input
      size cannot be deferred by the operand shape.
    aggregate_to_topk: When true, aggregates approximate results to top-k. When
      false, returns the approximate results.

  Returns:
    Tuple[Array, Array] : Max k values and their indices of the inputs.
  """
  if xc._version < 45:
    aggregate_to_topk = True
  return approx_top_k_p.bind(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=True,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk)


def approx_min_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95,
                 reduction_input_size_override: int = -1,
                 aggregate_to_topk: bool = True) -> Tuple[Array, Array]:
  """Returns min ``k`` values and their indices of the ``operand``.

  Args:
    operand : Array to search for min-k.
    k : Specifies the number of min-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by operands[reduction_dim] for evaluating the recall.
      This option is useful when the given operand is only a subset of the
      overall computation in SPMD or distributed pipelines, where the true input
      size cannot be deferred by the operand shape.
    aggregate_to_topk: When true, aggregates approximate results to top-k. When
      false, returns the approximate results.

  Returns:
    Tuple[Array, Array] : Least k values and their indices of the inputs.
  """
  if xc._version < 45:
    aggregate_to_topk = True
  return approx_top_k_p.bind(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=False,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk)


def ann_recall(result_neighbors: Array, ground_truth_neighbors: Array) -> float:
  """Computes the recall of an approximate nearest neighbor search.

  Note, this is NOT a JAX-compatible op. This is a host function that only takes
  numpy arrays as the inputs.

  Args:
    result_neighbors: int32 numpy array of the shape
      [num_queries, neighbors_per_query] where the values are the indices of the
      dataset.
    ground_truth_neighbors: int32 numpy array of with shape
      [num_queries, ground_truth_neighbors_per_query] where the values are the
      indices of the dataset.

  Example:
    # shape [num_queries, 100]
    ground_truth_neighbors = ...  # pre-computed top 100 sorted neighbors
    # shape [num_queries, 200]
    result_neighbors = ...  # Retrieves 200 neighbors from some ann algorithms.

    # Computes the recall with k = 100
    ann_recall(result_neighbors, ground_truth_neighbors)

    # Computes the recall with k = 10
    ann_recall(result_neighbors, ground_truth_neighbors[:,0:10])

  Returns:
    The recall.
  """
  assert len(result_neighbors.shape) == 2, 'shape = [num_queries, neighbors_per_query]'
  assert len(ground_truth_neighbors.shape) == 2, 'shape = [num_queries, ground_truth_neighbors_per_query]'
  assert result_neighbors.shape[0] == ground_truth_neighbors.shape[0]
  gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]
  hits = sum(
      len(list(x for x in nn_per_q if x.item() in gt_sets[q]))
      for q, nn_per_q in enumerate(result_neighbors))
  return hits / ground_truth_neighbors.size


def _approx_top_k_abstract_eval(operand, *, k, reduction_dimension,
                                recall_target, is_max_k,
                                reduction_input_size_override,
                                aggregate_to_topk):
  if k <= 0:
    raise ValueError('k must be positive, got {}'.format(k))
  if len(operand.shape) == 0:
    raise TypeError('approx_top_k operand must have >= 1 dimension, got {}'.format(
        operand.shape))
  dims = list(operand.shape)
  if dims[reduction_dimension] < k:
    raise ValueError(
        'k must be smaller than the size of reduction_dim {}, got {}'.format(
            dims[reduction_dimension], k))
  if xc._version >= 45:
    reduction_input_size = dims[reduction_dimension]
    dims[reduction_dimension] = xc.ops.ApproxTopKReductionOutputSize(
        reduction_input_size, len(dims), k, recall_target, aggregate_to_topk,
        reduction_input_size_override)[0]
  else:
    dims[reduction_dimension] = k
  return (operand.update(
      shape=dims, dtype=operand.dtype, weak_type=operand.weak_type),
          operand.update(shape=dims, dtype=np.dtype(np.int32)))


def _comparator_builder(operand, op_type, is_max_k):
  c = xc.XlaBuilder(
      'top_k_{}_comparator'.format('gt' if is_max_k else 'lt'))
  p0 = xla.parameter(c, 0, xc.Shape.scalar_shape(op_type))
  p1 = xla.parameter(c, 1, xc.Shape.scalar_shape(op_type))
  xla.parameter(c, 2, xc.Shape.scalar_shape(np.dtype(np.int32)))
  xla.parameter(c, 3, xc.Shape.scalar_shape(np.dtype(np.int32)))
  if is_max_k:
    cmp_result = xc.ops.Gt(p0, p1)
  else:
    cmp_result = xc.ops.Lt(p0, p1)
  return c.build(cmp_result)


def _approx_top_k_tpu_translation(ctx, avals_in, avals_out, operand, *, k,
                                  reduction_dimension, recall_target, is_max_k,
                                  reduction_input_size_override,
                                  aggregate_to_topk):
  c = ctx.builder
  op_shape = c.get_shape(operand)
  if not op_shape.is_array():
    raise ValueError('operand must be an array, but was {}'.format(op_shape))
  op_dims = op_shape.dimensions()
  op_type = op_shape.element_type()
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension
  comparator = _comparator_builder(operand, op_type, is_max_k)
  if is_max_k:
    if dtypes.issubdtype(op_type, np.floating):
      init_literal = np.array(np.NINF, dtype=op_type)
    else:
      init_literal = np.iinfo(op_type).min()
  else:
    if dtypes.issubdtype(op_type, np.floating):
      init_literal = np.array(np.Inf, dtype=op_type)
    else:
      init_literal = np.iinfo(op_type).max()
  iota = xc.ops.Iota(c, xc.Shape.array_shape(np.dtype(np.int32), op_dims),
                     reduction_dimension)
  init_val = xc.ops.Constant(c, init_literal)
  init_arg = xc.ops.Constant(c, np.int32(-1))
  out = xc.ops.ApproxTopK(c, [operand, iota], [init_val, init_arg], k,
                          reduction_dimension, comparator, recall_target,
                          aggregate_to_topk, reduction_input_size_override)
  return xla.xla_destructure(c, out)


def _approx_top_k_fallback_translation(ctx, avals_in, avals_out, operand, *, k,
                                       reduction_dimension, recall_target,
                                       is_max_k, reduction_input_size_override,
                                       aggregate_to_topk):
  c = ctx.builder
  op_shape = c.get_shape(operand)
  if not op_shape.is_array():
    raise ValueError('operand must be an array, but was {}'.format(op_shape))
  op_dims = op_shape.dimensions()
  op_type = op_shape.element_type()
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension
  comparator = _comparator_builder(operand, op_type, is_max_k)
  iota = xc.ops.Iota(c, xc.Shape.array_shape(np.dtype(np.int32), op_dims),
                     reduction_dimension)
  val_arg = xc.ops.Sort(c, [operand, iota], comparator, reduction_dimension)
  vals = xc.ops.GetTupleElement(val_arg, 0)
  args = xc.ops.GetTupleElement(val_arg, 1)
  sliced_vals = xc.ops.SliceInDim(vals, 0,
                                  avals_out[0].shape[reduction_dimension], 1,
                                  reduction_dimension)
  sliced_args = xc.ops.SliceInDim(args, 0,
                                  avals_out[0].shape[reduction_dimension], 1,
                                  reduction_dimension)
  return sliced_vals, sliced_args


def _approx_top_k_batch_rule(batched_args, batch_dims, *, k,
                             reduction_dimension, recall_target, is_max_k,
                             reduction_input_size_override, aggregate_to_topk):
  prototype_arg, new_bdim = next(
      (a, b) for a, b in zip(batched_args, batch_dims) if b is not None)
  new_args = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is None:
      dims = np.delete(np.arange(prototype_arg.ndim), new_bdim)
      new_args.append(lax.broadcast_in_dim(arg, prototype_arg.shape, dims))
    else:
      new_args.append(batching.moveaxis(arg, bdim, new_bdim))
  new_reduction_dim = reduction_dimension + (new_bdim <= reduction_dimension)
  bdims = (new_bdim,) * len(new_args)
  return (approx_top_k_p.bind(
      *new_args,
      k=k,
      reduction_dimension=new_reduction_dim,
      recall_target=recall_target,
      is_max_k=False,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk), bdims)


# Slow jvp implementation using gather.
#
# TODO(fchern): Some optimization ideas
# 1. ApproxTopK is internally a variadic reduce, so we can simply call
#    ApproxTopK(operand, tangent, iota) for jvp.
# 2. vjp cannot benefit from the algorithm above. We must run scatter to
#    distribute the output cotangent to input cotangent. A reasonable way to do
#    this is to run it on CPU.
def _approx_top_k_jvp(primals, tangents, *, k, reduction_dimension,
                      recall_target, is_max_k, reduction_input_size_override,
                      aggregate_to_topk):
  operand, = primals
  tangent, = tangents
  if is_max_k:
    val_out, arg_out = approx_max_k(operand, k, reduction_dimension,
                                    recall_target,
                                    reduction_input_size_override,
                                    aggregate_to_topk)
  else:
    val_out, arg_out = approx_min_k(operand, k, reduction_dimension,
                                    recall_target,
                                    reduction_input_size_override,
                                    aggregate_to_topk)
  if type(tangent) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    arg_shape = arg_out.shape
    rank = len(arg_shape)
    if reduction_dimension < 0:
      reduction_dimension += rank
    iotas = [
        lax.broadcasted_iota(arg_out.dtype, arg_shape, i) for i in range(rank)
    ]
    idx = tuple(
        arg_out if i == reduction_dimension else iotas[i] for i in range(rank))
    tangent_out = tangent[idx]
  return (val_out, arg_out), (tangent_out, ad_util.Zero.from_value(arg_out))


approx_top_k_p = core.Primitive('approx_top_k')
approx_top_k_p.multiple_results = True
approx_top_k_p.def_impl(partial(xla.apply_primitive, approx_top_k_p))
approx_top_k_p.def_abstract_eval(_approx_top_k_abstract_eval)
xla.register_translation(approx_top_k_p, _approx_top_k_fallback_translation)
xla.register_translation(approx_top_k_p, _approx_top_k_tpu_translation,
                         platform='tpu')
batching.primitive_batchers[approx_top_k_p] = _approx_top_k_batch_rule
ad.primitive_jvps[approx_top_k_p] = _approx_top_k_jvp
