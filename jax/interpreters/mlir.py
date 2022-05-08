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

# Lowering and execution path that converts jaxprs into the MLIR MHLO/CHLO
# dialects.
from __future__ import annotations

import collections
import dataclasses
import functools
from functools import partial
import io
import itertools
import re
import typing
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Type, Union, FrozenSet)
from typing_extensions import Protocol
import warnings

import jax
from jax import core
from jax import linear_util as lu
from jax._src import ad_util
from jax._src import device_array
from jax._src import dtypes
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src import source_info_util
import jax._src.util as util
from jax.config import config
import jax.interpreters.ad as ad
import jax.interpreters.partial_eval as pe
import jax.interpreters.xla as xla
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = typing.TypeVar("T")

# mypy implicitly sets this variable to true when type checking.
MYPY = False

lowerable_effects: Set[core.Effect] = set()


# IR Helpers

def dense_int_elements(xs) -> ir.DenseIntElementsAttr:
  return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))

def dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
  a = np.packbits(np.array(xs, np.bool_), bitorder='little')
  # TODO(b/209005197): Work around for MLIR crash for non-splat single element
  # buffers.
  if len(xs) == 1:
    a = np.array(0 if a.item() == 0 else 0xff, np.uint8)
  return ir.DenseElementsAttr.get(
      a, type=ir.IntegerType.get_signless(1), shape=[len(xs)])

def i32_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
def i64_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)

def shape_tensor(sizes: Sequence[Union[int, ir.RankedTensorType]]
                 ) -> ir.RankedTensorType:
  sizes = [ir_constant(np.array(d, np.dtype('int32'))) if type(d) is int else d
           for d in sizes]
  int1d = aval_to_ir_type(core.ShapedArray((1,), np.dtype('int32')))
  return mhlo.ConcatenateOp([mhlo.ReshapeOp(int1d, d) for d in sizes],
                            i64_attr(0)).results


# IR Types

# Non-canonicalized dtype to IR type mapping.
_dtype_to_ir_type : Dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(dtypes.float0): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(dtypes.bfloat16): ir.BF16Type.get,
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}

def dtype_to_ir_type(dtype: Union[np.dtype, np.generic]) -> ir.Type:
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)
  try:
    ir_type_factory = _dtype_to_ir_type[dtype]
  except KeyError as err:
    raise TypeError(
        f"No dtype_to_ir_type handler for dtype: {dtype}") from err
  return ir_type_factory()

def _array_ir_types(aval: core.ShapedArray) -> Sequence[ir.Type]:
  return (ir.RankedTensorType.get(aval.shape, dtype_to_ir_type(aval.dtype)),)

def _dynamic_array_ir_types(aval: core.ShapedArray) -> Sequence[ir.Type]:
  shape = [d if type(d) is int else -1 for d in aval.shape]
  return (ir.RankedTensorType.get(shape, dtype_to_ir_type(aval.dtype)),)

def _bint_ir_types(aval: core.AbstractBInt) -> Sequence[ir.Type]:
  return (ir.RankedTensorType.get((), dtype_to_ir_type(dtypes.dtype('int32'))),)

ir_type_handlers: Dict[Type[core.AbstractValue],
                        Callable[[Any], Sequence[ir.Type]]] = {}

def aval_to_ir_types(aval: core.AbstractValue) -> Sequence[ir.Type]:
  """Converts a JAX aval to zero or more MLIR IR types.

  In general, a JAX value may be represented by multiple IR values, so this
  function returns multiple types."""
  try:
    return ir_type_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No ir_type_handler for aval type: {type(aval)}") from err

ir_type_handlers[core.AbstractBInt] = _bint_ir_types
ir_type_handlers[core.ShapedArray] = _array_ir_types
ir_type_handlers[core.ConcreteArray] = _array_ir_types
ir_type_handlers[core.AbstractToken] = lambda _: [mhlo.TokenType.get()]
ir_type_handlers[core.DShapedArray] = _dynamic_array_ir_types

def aval_to_ir_type(aval: core.AbstractValue) -> ir.Type:
  """Convenience wrapper around aval_to_ir_types for single types.

  For some common cases, e.g. dense arrays, we know JAX values are represented
  by a single IR value."""
  types = aval_to_ir_types(aval)
  if len(types) != 1:
    raise TypeError(f"aval_to_ir_type called on {aval} which corresponds to "
                    f"multiple IR types {types}")
  return types[0]


# Constants

class ConstantHandler(Protocol):
  def __call__(self, val: Any, canonicalize_types: bool) -> Sequence[ir.Value]:
    """Builds an IR representation for a constant `val`.

    A JAX value is represented by zero or more IR values."""

_constant_handlers : Dict[type, ConstantHandler] = {}

def register_constant_handler(type_: type, handler_fun: ConstantHandler):
  _constant_handlers[type_] = handler_fun

def ir_constants(val: Any,
                 canonicalize_types: bool = True) -> Sequence[ir.Value]:
  """Translate a Python `val` to an IR constant, canonicalizing its dtype.

  Args:
    val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant as a list of IR values.
  """
  for t in type(val).__mro__:
    handler = _constant_handlers.get(t)
    if handler:
      out = handler(val, canonicalize_types)
      assert all(isinstance(v, ir.Value) for v in out), (type(val), out)
      return out
  if hasattr(val, '__jax_array__'):
    return ir_constants(val.__jax_array__(), canonicalize_types)
  raise TypeError("No constant handler for type: {}".format(type(val)))

def ir_constant(val: Any, canonicalize_types: bool = True) -> ir.Value:
  """Convenience wrapper around ir_constants for singleton values."""
  values = ir_constants(val, canonicalize_types=canonicalize_types)
  if len(values) != 1:
    raise TypeError(f"ir_constant called on {val} which corresponds to "
                    f"multiple IR values {values}")
  return values[0]


def _numpy_array_constant(x: np.ndarray, canonicalize_types
                         ) -> Sequence[ir.Value]:
  if canonicalize_types:
    x = np.asarray(x, dtypes.canonicalize_dtype(x.dtype))
  element_type = dtype_to_ir_type(x.dtype)
  ir_type = ir.RankedTensorType.get(x.shape, element_type)
  shape = x.shape
  if x.dtype == np.bool_:
    nelems = x.size
    x = np.packbits(x, bitorder='little')
    # TODO(b/209005197): Work around for MLIR crash for non-splat single element
    # buffers.
    if nelems == 1:
      x = np.array(0 if x.item() == 0 else 0xff, np.uint8)
  elif x.dtype == dtypes.bfloat16:
    x = x.view(np.uint16)
  x = np.ascontiguousarray(x)
  attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)
  if jax._src.lib.xla_extension_version >= 64:
    return (mhlo.ConstOp(attr).result,)
  else:
    return (mhlo.ConstOp(ir_type, attr).result,)



def _ndarray_constant_handler(val: np.ndarray, canonicalize_types
                             ) -> Sequence[ir.Value]:
  """Constant handler for ndarray literals, handling zero-size strides.

  In most cases this function calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses np.broadcast_to which in turn
  uses size-zero strides).

  Args:
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  if dtypes.result_type(val) == dtypes.float0:
    return _numpy_array_constant(np.zeros(val.shape, dtype=np.bool_),
                                 canonicalize_types=False)
  elif np.any(np.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = np.where(np.equal(0, val.strides))
    other_axes, = np.where(np.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]  # type: ignore
    if canonicalize_types:
      collapsed_val = np.asarray(
          collapsed_val, dtypes.canonicalize_dtype(collapsed_val.dtype))
    out = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(
            val.shape, dtype_to_ir_type(collapsed_val.dtype)),
        _numpy_array_constant(collapsed_val, canonicalize_types=False)[0],
        dense_int_elements(other_axes)).result
    return (out,)
  else:
    return _numpy_array_constant(val, canonicalize_types)

register_constant_handler(np.ndarray, _ndarray_constant_handler)

for _scalar_type in [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     np.float16, np.float32, np.float64,
                     np.complex64, np.complex128,
                     np.bool_, np.longlong, dtypes.bfloat16]:
  register_constant_handler(_scalar_type, _ndarray_constant_handler)

def _python_scalar_handler(dtype, val, canonicalize_dtypes):
  return _numpy_array_constant(np.array(val, dtype), canonicalize_dtypes)

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))

def _device_array_constant_handler(val, canonicalize_types):
  return _ndarray_constant_handler(val.device_buffer.to_py(),
                                   canonicalize_types)
for t in device_array.device_array_types:
  register_constant_handler(t, _device_array_constant_handler)

register_constant_handler(
  core.Token, lambda _, __: [mhlo.CreateTokenOp(mhlo.TokenType.get()).result])

# Source locations

def _source_info_to_location(
    primitive: core.Primitive, params: Dict,
    source_info: source_info_util.SourceInfo,
    name_stack: Union[str, source_info_util.NameStack] = "") -> ir.Location:
  if config.jax_experimental_name_stack:
    eqn_str = (f'{str(source_info.name_stack)}/'
               f'{core.str_eqn_compact(primitive.name, params)}')
  else:
    assert isinstance(name_stack, str)
    eqn_str = name_stack + core.str_eqn_compact(primitive.name, params)
  frame = source_info_util.user_frame(source_info)
  if frame is None:
    loc = ir.Location.unknown()
  else:
    loc = ir.Location.file(xla._get_canonical_source_file(frame),
                           frame.line_num, 1)
  loc = ir.Location.name(eqn_str, childLoc=loc)
  # TODO(phawkins): also include primitive.name as the operator type.
  return loc


# Translation rules
NameStack = Union[str, source_info_util.NameStack]

def make_ir_context() -> ir.Context:
  """Creates an MLIR context suitable for JAX IR."""
  context = ir.Context()
  mhlo.register_mhlo_dialect(context)
  chlo.register_chlo_dialect(context)
  return context


Mesh = Any
MeshAxisName = Any

@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
  """A hardware axis context for parallel computations that use the GSPMD partitioner.

  This includes the mesh that will later by used to execute this computation,
  as well as a set of mesh axes that are currently (e.g. because the current lowering
  is invoked inside an xmap) lowered in the MANUAL sharding mode.
  """
  mesh: Mesh
  manual_axes: FrozenSet[MeshAxisName] = frozenset()

  @property
  def axis_env(self):
    # TODO: Should we restrict the mesh to manual axes? This depends on the
    # semantics of ReplicaId in partially manual computations, but I don't
    # know what they are...
    return xla.AxisEnv(nreps=1, names=(), sizes=())

  def extend_manual(self, axes: FrozenSet[MeshAxisName]) -> 'SPMDAxisContext':
    return SPMDAxisContext(self.mesh, self.manual_axes | axes)


@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
  """A hardware axis context for parallel computations that are partitioned by JAX.

  Unlike in the SPMDAxisContext, this means that JAX might need to emit calls to
  explicit collectives.
  """
  axis_env: xla.AxisEnv

AxisContext = Union[SPMDAxisContext, ReplicaAxisContext]

@dataclasses.dataclass
class ModuleContext:
  """Module-wide context information for MLIR lowering."""
  context: ir.Context
  module: ir.Module
  ip: ir.InsertionPoint
  symbol_table: ir.SymbolTable
  platform: str
  axis_context: AxisContext
  name_stack: NameStack
  keepalives: List[Any]

  # Cached primitive lowerings.
  cached_primitive_lowerings: Dict[Any, func_dialect.FuncOp]

  @property
  def axis_env(self) -> xla.AxisEnv:
    return self.axis_context.axis_env

  def __init__(
      self,
      platform: str,
      axis_context: AxisContext,
      name_stack: NameStack,
      keepalives: List[Any],
      context: Optional[ir.Context] = None,
      module: Optional[ir.Module] = None,
      ip: Optional[ir.InsertionPoint] = None,
      symbol_table: Optional[ir.SymbolTable] = None,
      cached_primitive_lowerings: Optional[Dict[Any, func_dialect.FuncOp]] = None):
    assert platform is not None
    self.context = context or make_ir_context()
    self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
    self.ip = ip or ir.InsertionPoint(self.module.body)
    self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
    self.platform = platform
    self.axis_context = axis_context
    self.name_stack = name_stack
    self.cached_primitive_lowerings = ({} if cached_primitive_lowerings is None
                                       else cached_primitive_lowerings)
    self.keepalives = keepalives

  def add_keepalive(self, keepalive: Any) -> None:
    self.keepalives.append(keepalive)

  def replace(self, **kw): return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class LoweringRuleContext:
  """Per-rule context information for MLIR lowering."""
  module_context: ModuleContext
  primitive: Optional[core.Primitive]
  avals_in: Sequence[core.AbstractValue]
  avals_out: Any  # Usually Sequence[core.AbstractValue], but sometimes None.
  tokens_in: TokenSet
  tokens_out: Optional[TokenSet]  # Mutable store for output containers

  def set_tokens_out(self, tokens_out: TokenSet):
    assert self.tokens_out is None, 'Should only set `tokens_out` once.'
    self.tokens_out = tokens_out

  def replace(self, **kw): return dataclasses.replace(self, **kw)


if not MYPY:
  class LoweringRule(Protocol):
    def __call__(self, ctx: LoweringRuleContext,
                 *args: Union[ir.Value, Sequence[ir.Value]],
                 **kw) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
      """Converts a JAX primitive invocation into MLIR."""
else:
  LoweringRule = Any

_lowerings: Dict[core.Primitive, LoweringRule] = {}
_platform_specific_lowerings: Dict[str, Dict[core.Primitive, LoweringRule]]
_platform_specific_lowerings = collections.defaultdict(dict)

def register_lowering(prim: core.Primitive, rule: LoweringRule,
                      platform: Optional[str] = None):
  if platform is None:
    _lowerings[prim] = rule
  else:
    # For backward compatibility reasons, we allow rules to be registered
    # under "gpu" even though the platforms are now called "cuda" and "rocm".
    # TODO(phawkins): fix up users to specify either "cuda" or "rocm" and remove
    # this expansion.
    for p in xb.expand_platform_alias(platform):
      _platform_specific_lowerings[p][prim] = rule


def _unwrap_singleton_ir_values(x): return x[0] if len(x) == 1 else x
def wrap_singleton_ir_values(x: Union[ir.Value, Sequence[ir.Value]]
                             ) -> Sequence[ir.Value]:
  """Adds a consistent tuples to a mixture of tupled and untuple values."""
  return (x,) if isinstance(x, ir.Value) else tuple(x)

def flatten_lowering_ir_args(
    xs: Sequence[Union[ir.Value, Sequence[ir.Value]]]
) -> Sequence[Sequence[ir.Value]]:
  return util.flatten(map(wrap_singleton_ir_values, xs))

_module_unique_id = itertools.count()
_module_name_regex = re.compile(r"[^\w.-]")

def sharded_aval(aval: core.ShapedArray,
                 sharding: Optional[xc.OpSharding]) -> core.ShapedArray:
  """Returns the new aval sharded based on sharding proto."""
  if sharding is None:
    return aval

  if (sharding.type == xc.OpSharding.Type.REPLICATED or
      sharding.type == xc.OpSharding.Type.MANUAL):
    return aval

  sharded_shape = []
  tile_rank = len(sharding.tile_assignment_dimensions)
  if sharding.replicate_on_last_tile_dim:
    tile_rank -= 1
  if sharding.last_tile_dims:
    tile_rank -= len(sharding.last_tile_dims)
  if tile_rank == 0:
    return aval

  for i in range(tile_rank):
    partitions = sharding.tile_assignment_dimensions[i]
    assert partitions > 0
    sharded_shape.append((aval.shape[i] + partitions - 1) // partitions)
  return aval.update(tuple(sharded_shape))


def lower_jaxpr_to_module(
    module_name: str, jaxpr: core.ClosedJaxpr,
    effects: List[core.Effect],
    platform: str,
    axis_context: AxisContext,
    name_stack: NameStack, donated_args: Sequence[bool],
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None
    ) -> Tuple[ir.Module, Optional[Any]]:
  """Lowers a top-level jaxpr to an MHLO module.

  Handles the quirks of the argument/return value passing conventions of the
  runtime.
  """
  platform = xb.canonicalize_platform(platform)
  if not xb.is_known_platform(platform):
    raise ValueError(f"Unknown platform {platform}")
  input_output_aliases = None
  in_avals = jaxpr.in_avals
  if arg_shardings is not None:
    in_avals = [
        sharded_aval(in_aval, in_sharding)
        for in_aval, in_sharding in zip(in_avals, arg_shardings)
    ]
  out_avals = jaxpr.out_avals
  if result_shardings is not None:
    out_avals = [
        sharded_aval(out_aval, out_sharding)
        for out_aval, out_sharding in zip(out_avals, result_shardings)
    ]
  platforms_with_donation = ("cuda", "rocm", "tpu")
  if platform in platforms_with_donation:
    input_output_aliases, donated_args = _set_up_aliases(
        in_avals, out_avals, donated_args)
  if any(eff not in lowerable_effects for eff in jaxpr.effects):
    raise ValueError(f'Cannot lower jaxpr with effects: {jaxpr.effects}')
  if any(donated_args):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(a) for a, d in zip(in_avals, donated_args)
                        if d]
    msg = "See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation."
    if platform not in platforms_with_donation:
      msg = f"Donation is not implemented for {platform}.\n{msg}"
    warnings.warn(f"Some donated buffers were not usable: {', '.join(unused_donations)}.\n{msg}")

  # Create a keepalives list that will be mutated during the lowering.
  keepalives: List[Any] = []
  ctx = ModuleContext(platform, axis_context, name_stack, keepalives)
  with ctx.context, ir.Location.unknown(ctx.context):
    # Remove module name characters that XLA would alter. This ensures that
    # XLA computation preserves the module name.
    module_name = _module_name_regex.sub("_", module_name)
    # Some clients expect modules to have unique names, e.g., in trace data.
    # This may or may not be a reasonable assumption.
    ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(
        f"{module_name}.{next(_module_unique_id)}")
    unlowerable_effects = {eff for eff in jaxpr.effects
                           if eff not in lowerable_effects}
    if unlowerable_effects:
      raise ValueError(
          f'Cannot lower jaxpr with unlowerable effects: {unlowerable_effects}')
    lower_jaxpr_to_fun(
        ctx, "main", jaxpr, effects, public=True, create_tokens=True,
        replace_tokens_with_dummy=True, replicated_args=replicated_args,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        input_output_aliases=input_output_aliases)

  ctx.module.operation.verify()
  return ctx.module, ctx.keepalives

def module_to_string(module: ir.Module) -> str:
  output = io.StringIO()
  module.operation.print(file=output, enable_debug_info=True,
                         print_generic_op_form=False)
  return output.getvalue()

def _set_up_aliases(avals_in, avals_out, donated_args):
  input_output_aliases = [None] * len(avals_in)
  # To match-up in-avals to out-avals we only care about the number of
  # bytes, so we strip off unrelated aval metadata (eg. the named shape)
  strip_metadata = lambda a: a.strip_named_shape().strip_weak_type()
  avals_in = map(strip_metadata, avals_in)
  avals_out = map(strip_metadata, avals_out)

  donations = collections.defaultdict(collections.deque)
  for i, (aval, donated) in enumerate(zip(avals_in, donated_args)):
    if donated:
      donations[aval].append(i)

  out_donated_args = list(donated_args)
  for i, aval in enumerate(avals_out):
    if donations.get(aval, ()):
      input_id = donations[aval].popleft()
      input_output_aliases[input_id] = i
      out_donated_args[input_id] = False

  return input_output_aliases, out_donated_args

Token = Sequence[ir.Value]

def token_type() -> Sequence[ir.Type]:
  return [mhlo.TokenType.get()]

def token() -> Token:
  return wrap_singleton_ir_values(
      mhlo.CreateTokenOp(mhlo.TokenType.get()).result)

class TokenSet:
  """An immutable container of tokens to be used to lower effectful jaxprs. When lowering
  effectful jaxprs, we need to thread MHLO tokens to sequence them. Each effect
  will need its own token that will be threaded in and out of the effectful
  primitives. A `TokenSet` encapsulates a set of MHLO tokens that will be
  used by the lowering rules.
  """
  _tokens: typing.OrderedDict[core.Effect, Token]

  def __init__(self, *args, **kwargs):
    self._tokens = collections.OrderedDict(*args, **kwargs)

  def __len__(self):
    return len(self._tokens)

  def get(self, effect: core.Effect) -> Token:
    return self._tokens[effect]

  @classmethod
  def create(cls, effects: Sequence[core.Effect]) -> TokenSet:
    """Creates a `TokenSet` corresponding to a list of `core.Effect`s."""
    tokens = [token() for _ in effects]
    return TokenSet(zip(effects, tokens))

  def items(self) -> Sequence[Tuple[core.Effect, Token]]:
    return tuple(self._tokens.items())

  def effects(self) -> Sequence[core.Effect]:
    return tuple(self._tokens.keys())

  def tokens(self) -> Sequence[Token]:
    return tuple(self._tokens.values())

  def subset(self, effects: Sequence[core.Effect]) -> TokenSet:
    """Return a subset of the `TokenSet` restricted to a set of `core.Effect`s."""
    return TokenSet((eff, self._tokens[eff]) for eff in effects)

  def update_tokens(self, tokens: TokenSet) -> TokenSet:
    """Returns a new `TokenSet` with tokens replaced with ones from the input `TokenSet`."""
    new_tokens = []
    for eff in self.effects():
      if eff in tokens._tokens:
        new_tokens.append(tokens._tokens[eff])
      else:
        new_tokens.append(self._tokens[eff])
    return TokenSet(zip(self.effects(), new_tokens))

def dummy_token_type() -> Sequence[ir.Type]:
  return aval_to_ir_types(core.ShapedArray((0,), np.bool_))

def dummy_token() -> Sequence[ir.Value]:
  return ir_constants(np.zeros(0, np.bool_))

def lower_jaxpr_to_fun(
    ctx: ModuleContext,
    name: str,
    jaxpr: core.ClosedJaxpr,
    effects: Sequence[core.Effect],
    *,
    create_tokens: bool = False,
    public: bool = False,
    replace_tokens_with_dummy: bool = False,
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    use_sharding_annotations: bool = True,
    input_output_aliases: Optional[Sequence[Optional[int]]] = None
) -> func_dialect.FuncOp:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    effects: a sequence of `core.Effect`s corresponding to an ordering of tokens
      that will be created in or used by the lowered function.
    create_tokens: if true, the MHLO will create tokens and ignore dummy input tokens.
    public: if true, the function's visibility is set to "public".
    replace_tokens_with_dummy: if true, token arguments/return values are
      replaced with bool arrays of size [0].
    replicated_args: if present, annotates arguments as replicated.
    arg_shardings: sharding annotations for each argument (optional).
    result_shardings: sharding annotations for each argument (optional).
    use_sharding_annotations: if True, use mhlo.sharding annotations on
      parameters and return values to express sharding. If False, use
      mhlo.custom_call operators with sharding annotations.
      TODO(b/228598865): remove this option when mhlo.sharding annotations are
      propagated on non-entry functions during MHLO->HLO conversion.
    input_output_aliases: optional sequence that maps argument numbers to the
      corresponding output that should alias them.
  Returns the name of the function.
  """
  def aval_to_types(aval):
    if replace_tokens_with_dummy and aval is core.abstract_token:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    return aval_to_ir_types(aval)

  input_types = map(aval_to_types, jaxpr.in_avals)
  output_types = map(aval_to_types, jaxpr.out_avals)
  num_tokens = len(effects)
  if create_tokens:
    # If we create the tokens they won't be inputs to the MLIR function.
    token_types = [dummy_token_type() for _ in effects]
  else:
    # If we aren't creating tokens they will be the initial inputs to the
    # MLIR function.
    num_tokens = len(effects)
    token_types = [token_type() for _ in effects]
  input_types = [*token_types, *input_types]
  output_types = [*token_types, *output_types]
  if input_output_aliases:
    token_input_output_aliases = [None] * num_tokens
    input_output_aliases = [*token_input_output_aliases, *input_output_aliases]
  if arg_shardings:
    token_shardings = [None] * num_tokens
    arg_shardings = [*token_shardings, *arg_shardings]
  if result_shardings:
    token_shardings = [None] * num_tokens
    result_shardings = [*token_shardings, *result_shardings]
  if replicated_args:
    token_replicated_args = [False] * num_tokens
    replicated_args = [*token_replicated_args, *replicated_args]
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  func_op = func_dialect.FuncOp(name, ftype, ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get(
      "public" if public else "private")
  ctx.symbol_table.insert(func_op)
  ir_arg_shardings = None
  if arg_shardings is not None:
    ir_arg_shardings = util.flatten(
        [[sharding] * len(types) for sharding, types
         in zip(arg_shardings, input_types)])
  ir_result_shardings = None
  if result_shardings is not None:
      ir_result_shardings = util.flatten(
        [[sharding] * len(types) for sharding, types
         in zip(result_shardings, output_types)])

  if (replicated_args is not None or ir_arg_shardings is not None
      or input_output_aliases is not None):
    arg_attrs: List[Dict[str, ir.Attribute]] = [
        {} for _ in range(len(flat_input_types))]

    if replicated_args is not None:
      replicated_ir_args = [[replicated] * len(types) for replicated, types
                            in zip(replicated_args, input_types)]
      for attrs, replicated in zip(arg_attrs, util.flatten(replicated_ir_args)):
        if replicated:
          attrs["mhlo.is_same_data_across_replicas"] = ir.UnitAttr.get()

    if use_sharding_annotations and ir_arg_shardings is not None:
      for attrs, sharding in zip(arg_attrs, ir_arg_shardings):
        if sharding is not None:
          attrs["mhlo.sharding"] = ir.StringAttr.get(
              sharding.SerializeToString())

    if input_output_aliases is not None:
      output_ids = util.unflatten(list(range(len(flat_output_types))),
                                  map(len, output_types))
      aliases: List[Optional[int]] = []
      for types, alias in zip(input_types, input_output_aliases):
        if alias is None:
          aliases.extend([None] * len(types))
        else:
          aliases.extend(output_ids[alias])

      for attrs, alias in zip(arg_attrs, aliases):
        if alias is not None:
          attrs["tf.aliasing_output"] = i32_attr(alias)

    func_op.arg_attrs = ir.ArrayAttr.get(
        [ir.DictAttr.get(attrs) for attrs in arg_attrs])

  if use_sharding_annotations and ir_result_shardings is not None:
    func_op.result_attrs = ir.ArrayAttr.get([
        ir.DictAttr.get(
            {} if sharding is None else
            {"mhlo.sharding": ir.StringAttr.get(sharding.SerializeToString())}
        ) for sharding in ir_result_shardings
    ])

  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    flat_args = entry_block.arguments
    if not use_sharding_annotations and ir_arg_shardings is not None:
      flat_args = map(wrap_with_sharding_op, flat_args, ir_arg_shardings)

    unflattened_args = util.unflatten(flat_args, map(len, input_types))
    # We separate out the token inputs and the usual inputs. The token inputs
    # will be passed to `jaxpr_subcomp` separately from the `args`.
    token_args, unflattened_args = util.split_list(unflattened_args, [num_tokens])
    if create_tokens:
      tokens_in = TokenSet.create(effects)
    else:
      tokens_in = TokenSet(zip(effects, token_args))
    args: List[List[ir.Value]] = []
    for aval, arg in zip(jaxpr.in_avals, unflattened_args):
      if replace_tokens_with_dummy and aval is core.abstract_token:
        args.append(mhlo.CreateTokenOp(mhlo.TokenType.get()).results)
      else:
        args.append(arg)
    callee_name_stack = xla.extend_name_stack(ctx.name_stack,
                                              util.wrap_name(name, 'jit'))
    out_vals, tokens_out = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
                                         jaxpr.jaxpr, tokens_in, map(ir_constants, jaxpr.consts),
                                         *args)
    outs = []
    if create_tokens:
      for _ in effects:
        outs.append(dummy_token())
    else:
      for token in tokens_out.tokens():
        outs.append(token)
    for aval, out in zip(jaxpr.out_avals, out_vals):
      if replace_tokens_with_dummy and aval is core.abstract_token:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      else:
        outs.append(out)
    flat_outputs = util.flatten(outs)
    if not use_sharding_annotations and ir_result_shardings is not None:
      flat_outputs = map(wrap_with_sharding_op, flat_outputs,
                         ir_result_shardings)

    func_dialect.ReturnOp(flat_outputs)

  return func_op

def _emit_lowering_rule_as_fun(lowering_rule,
                               ctx: LoweringRuleContext) -> func_dialect.FuncOp:
  """Emits the contents of a lowering rule as a private function."""
  input_types = map(aval_to_ir_types, ctx.avals_in)
  output_types = map(aval_to_ir_types, ctx.avals_out)
  token_types = [token_type() for _ in ctx.tokens_in.items()]
  input_types = [*token_types, *input_types]
  output_types = [*token_types, *output_types]

  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  assert ctx.primitive is not None
  func_op = func_dialect.FuncOp(ctx.primitive.name, ftype,
                                ip=ctx.module_context.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
  ctx.module_context.symbol_table.insert(func_op)
  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = util.unflatten(entry_block.arguments,
                                      map(len, input_types))
    token_args, unflattened_args = util.split_list(unflattened_args, [len(ctx.tokens_in)])
    sub_ctx = ctx.replace(tokens_in=TokenSet(zip(ctx.tokens_in.effects(), token_args)))
    outs = lowering_rule(sub_ctx, *_unwrap_singleton_ir_values(unflattened_args))
    if sub_ctx.tokens_out:
      outs = [*sub_ctx.tokens_out.tokens(), outs]
    func_dialect.ReturnOp(util.flatten(map(wrap_singleton_ir_values, outs)))
  return func_op

def jaxpr_subcomp(ctx: ModuleContext, jaxpr: core.Jaxpr,
                  tokens: TokenSet,
                  consts: Sequence[Sequence[ir.Value]],
                  *args: Sequence[ir.Value]
                  ) -> Tuple[Sequence[Sequence[ir.Value]], TokenSet]:
  """Lowers a jaxpr into mHLO, inlined into an existing function.

  Assumes that an MLIR context, location, and insertion point are set.
  """
  assert ctx.platform != "gpu"
  def read(v: core.Var) -> Sequence[ir.Value]:
    if type(v) is core.Literal:
      return ir_constants(v.val, canonicalize_types=True)
    else:
      return env[v]

  def aval(v: core.Var) -> core.AbstractValue:
    if type(v) is core.Literal:
      return xla.abstractify(v.val)
    else:
      return v.aval

  def write(v: core.Var, node: Sequence[ir.Value]):
    assert node is not None
    env[v] = tuple(node)


  env: Dict[core.Var, Tuple[ir.Value, ...]] = {}

  assert len(args) == len(jaxpr.invars), (jaxpr, args)
  assert len(consts) == len(jaxpr.constvars), (jaxpr, consts)
  assert all(isinstance(v, ir.Value) for vs in consts for v in vs), consts
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    if config.jax_experimental_name_stack:
      assert isinstance(ctx.name_stack, source_info_util.NameStack), type(ctx.name_stack)
      source_info = eqn.source_info.replace(
          name_stack=ctx.name_stack + eqn.source_info.name_stack)
    else:
      source_info = eqn.source_info
    loc = _source_info_to_location(eqn.primitive, eqn.params, source_info,
                                   name_stack=ctx.name_stack)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive in _platform_specific_lowerings[ctx.platform]:
        rule = _platform_specific_lowerings[ctx.platform][eqn.primitive]
      elif eqn.primitive in xla._backend_specific_translations[ctx.platform]:
        rule = xla_fallback_lowering(eqn.primitive)
      elif eqn.primitive in _lowerings:
        rule = _lowerings[eqn.primitive]
      elif eqn.primitive in xla._translations:
        rule = xla_fallback_lowering(eqn.primitive)
      else:
        raise NotImplementedError(
            f"MLIR translation rule for primitive '{eqn.primitive.name}' not "
            f"found for platform {ctx.platform}")

      eqn_ctx = (ctx.replace(name_stack=source_info.name_stack) if
          config.jax_experimental_name_stack else ctx)
      effects = [eff for eff in eqn.effects if eff in core.ordered_effects]
      tokens_in = tokens.subset(effects)
      rule_ctx = LoweringRuleContext(
          module_context=eqn_ctx, primitive=eqn.primitive,
          avals_in=map(aval, eqn.invars), avals_out=map(aval, eqn.outvars),
          tokens_in=tokens_in, tokens_out=None)
      ans = rule(rule_ctx, *map(_unwrap_singleton_ir_values, in_nodes),
                 **eqn.params)
      if effects:
        # If there were ordered effects in the primitive, there should be output
        # tokens we need for subsequent ordered effects.
        tokens_out = rule_ctx.tokens_out
        if tokens_out is None:
          raise ValueError(
              f'Lowering rule for `{eqn.primitive}` needs to set `tokens_out` '
              f'because it has effects: {eqn.effects}.')
        if tokens_out.effects() != tokens_in.effects():
          raise ValueError(
              f'Lowering rule for `{eqn.primitive}` '
              'returns incorrect set of output tokens. '
              f'Expected: {tuple(tokens_in.effects())} vs. Actual: {tuple(tokens_out.effects())}')
        tokens = tokens.update_tokens(tokens_out)

    try:
      out_nodes = tuple(map(wrap_singleton_ir_values, ans))
    except TypeError as e:
      raise ValueError("Output of translation rule must be iterable: "
                       f"{eqn}, got output {ans}") from e

    assert all(isinstance(v, tuple) for v in out_nodes), (ans, eqn)
    assert all(isinstance(v, ir.Value) for w in out_nodes for v in w), (ans, eqn)
    assert len(ans) == len(eqn.outvars), (ans, eqn)
    map(write, eqn.outvars, out_nodes)
  return map(read, jaxpr.outvars), tokens

def _ir_consts(consts):
  unique_consts = {id(const): const for const in consts}
  ir_consts = {
      id_: ir_constants(const) for id_, const in unique_consts.items()}
  return [ir_consts[id(const)] for const in consts]

def lower_fun(fun: Callable, multiple_results: bool = True) -> Callable:
  """Converts a traceable JAX function `fun` into a lowering rule.

  The returned function does not use `avals_out`, so callers may pass any value
  as `avals_out`."""
  def f_lowered(ctx, *args, **params):
    if multiple_results:
      f = fun
    else:
      f = lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    axis_env = ctx.module_context.axis_env
    with core.extend_axis_env_nd(zip(axis_env.names, axis_env.sizes)):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    out, tokens = jaxpr_subcomp(ctx.module_context, jaxpr, ctx.tokens_in, _ir_consts(consts),
                                *map(wrap_singleton_ir_values, args))
    ctx.set_tokens_out(tokens)
    return out

  return f_lowered



def _call_lowering(fn_name, stack_name, call_jaxpr, backend, ctx, avals_in,
                   avals_out, tokens_in, *args):
  xla.check_backend_matches(backend, ctx.platform)
  output_types = map(aval_to_ir_types, avals_out)
  flat_output_types = util.flatten(output_types)
  effects = tokens_in.effects()
  symbol_name = lower_jaxpr_to_fun(ctx, fn_name,
                                   core.ClosedJaxpr(call_jaxpr, ()),
                                   effects).name.value
  args = [*tokens_in.tokens(), *args]
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(symbol_name),
                             flatten_lowering_ir_args(args))
  out_nodes = util.unflatten(call.results, map(len, output_types))
  tokens, out_nodes = util.split_list(out_nodes, [len(effects)])
  tokens_out = tokens_in.update_tokens(TokenSet(zip(effects, tokens)))
  return out_nodes, tokens_out

def _xla_call_lower(ctx, *args,
                    backend=None, name, call_jaxpr, donated_invars, inline=None,
                    device=None, keep_unused=None):
  del device, donated_invars, inline, keep_unused  # Ignored.
  out_nodes, tokens = _call_lowering(
      name, util.wrap_name(name, "jit"), call_jaxpr, backend,
      ctx.module_context, ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args)
  ctx.set_tokens_out(tokens)
  return out_nodes

register_lowering(xla.xla_call_p, _xla_call_lower)

def _named_call_lowering(ctx, *args, name, backend=None,
                         call_jaxpr):
  out_nodes, tokens = _call_lowering(
      name, name, call_jaxpr, backend, ctx.module_context,
      ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args)
  ctx.set_tokens_out(tokens)
  return out_nodes

register_lowering(core.named_call_p, _named_call_lowering)
register_lowering(core.call_p, partial(_named_call_lowering, name="core_call"))


def full_like_aval(value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero = ir_constant(np.array(value, aval.dtype))
  if jax._src.lib.mlir_api_version < 9:
    return mhlo.BroadcastOp(aval_to_ir_type(aval), zero,
                            dense_int_elements(aval.shape)).result
  else:
    return mhlo.BroadcastOp(zero, dense_int_elements(aval.shape)).result


def zeros_like_lowering(ctx, x):
  aval, = ctx.avals_in
  assert isinstance(aval, core.ShapedArray), aval
  return [full_like_aval(0, aval)]
register_lowering(ad_util.zeros_like_p, zeros_like_lowering)

def add_jaxvals_lowering(ctx, x, y):
  return mhlo.AddOp(x, y).results
register_lowering(ad_util.add_jaxvals_p, add_jaxvals_lowering)

register_lowering(ad_util.stop_gradient_p, lambda ctx, x: [x])


def compare_mhlo(x, y, direction, type):
  """Creates mhlo.CompareOp."""
  return mhlo.CompareOp(x, y, mhlo.ComparisonDirectionAttr.get(direction),
                        mhlo.ComparisonTypeAttr.get(type))

def _minmax_mhlo(op, cmp, x, y):
  """Min/max that compares complex values lexicographically as pairs."""
  tensor_type = ir.RankedTensorType(x.type)
  if ir.ComplexType.isinstance(tensor_type.element_type):
    rx = mhlo.RealOp(x).result
    ry = mhlo.RealOp(y).result
    real_eq = compare_mhlo(rx, ry, "EQ", "FLOAT")
    real_cmp = compare_mhlo(rx, ry, cmp, "FLOAT")
    imag_cmp = compare_mhlo(
        mhlo.ImagOp(x).result,
        mhlo.ImagOp(y).result, cmp, "FLOAT")
    which = mhlo.SelectOp(real_eq, imag_cmp, real_cmp).result
    return mhlo.SelectOp(which, x, y)
  else:
    return op(x, y)

min_mhlo = partial(_minmax_mhlo, mhlo.MinOp, "LT")
max_mhlo = partial(_minmax_mhlo, mhlo.MaxOp, "GT")


def convert_mhlo(x, aval_in, aval_out):
  """Variant of convert that has XLA HLO semantics.

  In particular, treat casts to boolean as x != 0, rather than truncating
  integer values (b/209440332)."""
  if aval_out.dtype == np.dtype(np.bool_):
    if dtypes.issubdtype(aval_in.dtype, np.inexact):
      compare_type = "FLOAT"
    elif dtypes.issubdtype(aval_in.dtype, np.signedinteger):
      compare_type = "SIGNED"
    else:
      compare_type = "UNSIGNED"
    return compare_mhlo(x, full_like_aval(0, aval_in), "NE",
                         compare_type).result
  return mhlo.ConvertOp(aval_to_ir_type(aval_out), x).result

def _wrap_with_spmd_op(name: str,
                       result_type: ir.Type,
                       x: ir.Value,
                       sharding_proto: xc.OpSharding,
                       unspecified_dims: Optional[Set[int]] = None):
  # unspecified_dims indicate dimensions whose shardings are not specified and
  # XLA sharding propagation can change them.
  if unspecified_dims:
    backend_config = "unspecified_dims=[" + ",".join(
        [str(i) for i in sorted(unspecified_dims)]) + "]"
  else:
    backend_config = ""
  op = mhlo.CustomCallOp([result_type], [x],
                         call_target_name=ir.StringAttr.get(name),
                         has_side_effect=ir.BoolAttr.get(False),
                         backend_config=ir.StringAttr.get(backend_config),
                         api_version=i32_attr(1),
                         called_computations=ir.ArrayAttr.get([]),
                         operand_layouts=None,
                         result_layouts=None)
  op.attributes["mhlo.sharding"] = ir.StringAttr.get(
      sharding_proto.SerializeToString())
  return op.result

def wrap_with_sharding_op(x: ir.Value,
                          sharding_proto: xc.OpSharding,
                          unspecified_dims: Optional[Set[int]] = None):
  return _wrap_with_spmd_op("Sharding", x.type, x, sharding_proto,
                            unspecified_dims)

wrap_with_full_to_shard_op = partial(_wrap_with_spmd_op, "SPMDFullToShardShape")
wrap_with_shard_to_full_op = partial(_wrap_with_spmd_op, "SPMDShardToFullShape")

def set_sharding(op, sharding_proto: xc.OpSharding):
  op.attributes["mhlo.sharding"] = ir.StringAttr.get(
      sharding_proto.SerializeToString())

# MLIR lowerings for lax primitives

def cache_lowering(f):
  """Decorator that causes the contents of a lowering rule to be reused.

  The lowering will be emitted out-of-line in a separate function, together with
  a call to that function. If the same primitive is called with the same shapes
  and parameters, a new call to the original function will be added, without
  emitting a new function.
  """
  @functools.wraps(f)
  def cached_lowering(ctx, *args, **params):
    assert ctx.primitive is not None
    key = (ctx.primitive, tuple(ctx.avals_in), tuple(ctx.avals_out),
           tuple(params.items()))
    try:
      func = ctx.module_context.cached_primitive_lowerings.get(key)
    except TypeError:
      # If the parameters aren't hashable, give up on caching.
      # TODO(phawkins): switch to requiring hashability, when XLA fallback
      # computations have been ported to MHLO.
      return f(ctx, *args, **params)
    if func is None:
      func = _emit_lowering_rule_as_fun(partial(f, **params), ctx)
      ctx.module_context.cached_primitive_lowerings[key] = func

    output_types = map(aval_to_ir_types, ctx.avals_out)
    flat_output_types = util.flatten(output_types)
    call = func_dialect.CallOp(flat_output_types,
                               ir.FlatSymbolRefAttr.get(func.name.value),
                               flatten_lowering_ir_args(args))
    return util.unflatten(call.results, map(len, output_types))
  return cached_lowering



def xla_computation_to_mhlo_module(xla_computation: xc.XlaComputation
                                  ) -> ir.Module:
  module_str = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation)
  return ir.Module.parse(module_str)

def merge_mhlo_modules(dst_module: ir.Module,
                       sym_name: str,
                       src_module: ir.Module) -> str:
  """Returns the name of src_module's main() function, after renaming."""
  callee_name = None
  assert dst_module.context == src_module.context
  dst_symtab = ir.SymbolTable(dst_module.operation)

  n = len(dst_module.body.operations)
  for op in src_module.body.operations:
    dst_module.body.append(op)
  ops = list(dst_module.body.operations)[n:]

  for op in ops:
    op = typing.cast(func_dialect.FuncOp, op)
    old_name = op.name.value
    if op.name.value == "main":
      dst_symtab.set_symbol_name(op, sym_name)
      op.attributes["sym_visibility"] = ir.StringAttr.get("private")
      callee_name = ir.StringAttr(dst_symtab.insert(op)).value
      new_name = callee_name
    else:
      new_name = ir.StringAttr(dst_symtab.insert(op)).value

    # Replace references to the symbol with the new name
    for other_op in ops:
      dst_symtab.replace_all_symbol_uses(
          old_name, new_name, other_op.operation)


  assert callee_name is not None
  return callee_name


def xla_fallback_lowering(prim: core.Primitive):
  @cache_lowering
  def fallback(ctx: LoweringRuleContext, *args, **params):
    module_ctx = ctx.module_context
    xla_computation = xla.primitive_subcomputation(
        module_ctx.platform, module_ctx.axis_env, prim, ctx.avals_in,
        ctx.avals_out, **params)
    xla_module = xla_computation_to_mhlo_module(xla_computation)
    callee_name = merge_mhlo_modules(
        module_ctx.module, f"xla_fallback_{prim.name}", xla_module)
    output_types = map(aval_to_ir_types, ctx.avals_out)
    flat_output_types = util.flatten(output_types)
    output_type = (ir.TupleType.get_tuple(flat_output_types)
                   if prim.multiple_results else flat_output_types[0])

    call = func_dialect.CallOp([output_type],
                               ir.FlatSymbolRefAttr.get(callee_name),
                               flatten_lowering_ir_args(args)).result
    if not prim.multiple_results:
      return [call]
    flat_results = [mhlo.GetTupleElementOp(call, i32_attr(i)).result
                    for i in range(len(flat_output_types))]

    return util.unflatten(flat_results, map(len, output_types))
  return fallback

register_lowering(ad.custom_lin_p, ad._raise_custom_vjp_error_on_jvp)

def emit_python_callback(platform, callback,
                         operands: List[ir.Value],
                         operand_avals: List[core.AbstractValue],
                         result_avals: List[core.AbstractValue],
                         has_side_effect: bool) -> Tuple[List[ir.Value], Any]:
  """Creates an MHLO `CustomCallOp` that calls back to the provided function."""
  if platform != "cpu":
    raise ValueError('`EmitPythonCallback` not supported in non-CPU backends.')
  backend = xb.get_backend(platform)
  result_shapes = util.flatten(
      [xla.aval_to_xla_shapes(result_aval) for result_aval in result_avals])
  operand_shapes = util.flatten(
      [xla.aval_to_xla_shapes(op_aval) for op_aval in operand_avals])
  callback_descriptor, keepalive = backend.get_emit_python_callback_descriptor(
      callback, operand_shapes, result_shapes)
  descriptor_operand = ir_constant(
      callback_descriptor, canonicalize_types=False)
  callback_operands = [descriptor_operand, *operands]
  result_types = util.flatten(
      [aval_to_ir_types(aval) for aval in result_avals])
  result_type = ir.TupleType.get_tuple(result_types)
  result = mhlo.CustomCallOp(
      [result_type],
      callback_operands,
      call_target_name=ir.StringAttr.get("xla_python_cpu_callback"),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      api_version=i32_attr(2),
      called_computations=ir.ArrayAttr.get([]),
      backend_config=ir.StringAttr.get(""),
      operand_layouts=None,
      result_layouts=None)
  results = [
      mhlo.GetTupleElementOp(result, i32_attr(i)).result
      for i in range(len(result_types))
  ]
  return results, keepalive

# Lax ops missing MLIR lowerings.
# # TODO(b/203775215): these are missing from the cHLO dialect. Either add
# # them or port them to Python.
# lax.igamma_p,
# lax.igammac_p,
# lax.igamma_grad_a,
# lax.random_gamma_grad_p,
# lax.bessel_i0e_p,
# lax.bessel_i1e_p,
# lax.erf_inv_p,
# lax.regularized_incomplete_beta_p,

# # CHLO doesn't have a legalization for bf16 (b/203774470)
# lax.erf_p,
# lax.erfc_p,
