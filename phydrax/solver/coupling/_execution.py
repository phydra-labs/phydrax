#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Signature-grouped, bounded lane execution of spatial coupled problems.

Preparation groups coupled work that is one executable: components (and law
contribution blocks) whose traced programs are identical once their array
leaves are program arguments. Equal traced programs with equal hoisted
constants, equal first-order JVP/VJP programs where custom differentiation
rules occur, and equal host callbacks are equal functions with equal certified
derivatives, so the representative's program evaluated on a member's arrays is
that member's own evaluation; names, shapes, or classes alone never group
anything, and lanes refuse derivatives the signature does not certify.

Each group becomes one fixed-capacity workset of the execution-workset
substrate: its members are the canonical items, ``lane_capacity`` lanes form a
bucket, buckets run sequentially (``lax.map``) and lanes within a bucket are
vectorized, so one bucket is the live working set. The resource estimate of a
group is checked against the declared working-set bound at preparation, before
lane data are stacked or any batched program is traced. An execution group
places the lane axis of each bucket on its devices through the existing
execution-runtime binding; no second runtime is created. Caller-owned states
and arguments are only read, never donated.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.extend.core as jex_core
import jax.numpy as jnp
from jax import Array
from jax.core import ShapedArray
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree

from ..._execution_pool import PoolExecutionSignature
from ..._execution_resources import ExecutionGroupSpec
from ..._execution_runtime import bind_execution_group
from ..._execution_workset import ExecutionWorksetPlan, PreparedExecutionWorksets
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, positive_integer
from ...linalg import AbstractLinearOperator, AbstractVectorSpace, BlockLinearOperator
from ...typing import checked, parse
from ._components import AbstractSpatialComponent


LaneKind: TypeAlias = Literal["component", "interface"]
LaneAction: TypeAlias = Literal[
    "weak", "weak-transpose", "identified", "identified-transpose"
]

type OwnerPath = tuple[str, str]
type ArgumentSignature = tuple[PyTreeDef, tuple[Hashable, ...]]
type LaneSubject = AbstractSpatialComponent | InterfaceLaneSubject
type LaneProgram = Callable[[LaneSubject, PyTree[Array]], PyTree[Array]]

# The workset substrate bounds one bucket to this modest fixed range.
_MAX_LANE_CAPACITY = 64


@final
class CoupledExecutionPolicy(StrictModule):
    """Declared lane execution of homogeneous coupled work.

    ``lane_capacity`` lanes form one bucket: they are vectorized together and
    buckets run one after another, so a bucket is the live working set.
    ``max_working_set_bytes`` bounds every group's estimated bucket working
    set; preparation refuses a group above it. ``execution_group`` places the
    lane axis of each bucket on that group's devices (``lane_capacity`` must be
    a multiple of its device count); without it lanes run on the default
    device. Grouping traces every component with the preparation
    ``arguments`` of ``prepare_coupled_problem``, so owners whose residual
    needs runtime arguments must receive representative ones there.
    """

    lane_capacity: int = eqx.field(static=True)
    max_working_set_bytes: int | None = eqx.field(static=True)
    execution_group: ExecutionGroupSpec | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        lane_capacity: int = 8,
        max_working_set_bytes: int | None = None,
        execution_group: ExecutionGroupSpec | None = None,
    ) -> None:
        capacity = positive_integer(lane_capacity, "lane_capacity")
        if capacity > _MAX_LANE_CAPACITY:
            raise ValueError(
                f"lane_capacity must lie in the fixed bucket range [1, {_MAX_LANE_CAPACITY}]."
            )
        bound = (
            None
            if max_working_set_bytes is None
            else positive_integer(max_working_set_bytes, "max_working_set_bytes")
        )
        if execution_group is not None:
            if not isinstance(execution_group, ExecutionGroupSpec):
                raise TypeError("execution_group must be an ExecutionGroupSpec.")
            if capacity % execution_group.device_count:
                raise ValueError(
                    "lane_capacity must be a multiple of the execution group's "
                    f"{execution_group.device_count} devices."
                )
        self.lane_capacity = capacity
        self.max_working_set_bytes = bound
        self.execution_group = execution_group


@final
class InterfaceLaneSubject(StrictModule):
    """One law contribution block in solve coordinates and its row identification.

    ``operator`` is the contribution composed with its components' constraint
    maps exactly once; ``identified`` is the state block its rows are
    identified with by the inverse Riesz map in the native linear system.
    """

    operator: AbstractLinearOperator
    identified: AbstractVectorSpace

    @checked
    def __init__(
        self, operator: AbstractLinearOperator, identified: AbstractVectorSpace, /
    ) -> None:
        self.operator = operator
        self.identified = identified


@final
class LaneWorksetEstimate(StrictModule):
    """Host resource estimate of one lane workset.

    ``lane_data_bytes`` are the stacked lane leaves the workset retains (bucket
    layout, padded lanes included). ``lane_input_bytes``/``lane_output_bytes``
    are the dynamic arguments and results of one lane, and
    ``traced_intermediate_bytes`` sums every intermediate value of one lane's
    traced program without buffer reuse (an upper estimate of its
    temporaries). ``working_set_bytes`` is one bucket: ``lane_capacity`` times
    the per-lane data, inputs, outputs, and intermediates.
    """

    lane_count: int = eqx.field(static=True)
    bucket_count: int = eqx.field(static=True)
    lane_capacity: int = eqx.field(static=True)
    padded_lanes: int = eqx.field(static=True)
    lane_data_bytes: int = eqx.field(static=True)
    lane_input_bytes: int = eqx.field(static=True)
    lane_output_bytes: int = eqx.field(static=True)
    traced_intermediate_bytes: int = eqx.field(static=True)
    working_set_bytes: int = eqx.field(static=True)


@final
class LaneDerivatives(StrictModule):
    """Derivatives of a lane workset certified equal to its members' own.

    ``route`` is how lanes are differentiated: ``"primal"`` by the standard
    rules of the primal program (every derivative certified), ``"forward"``
    or ``"reverse"`` through a first-derivative rule wrapping the bucket loop
    (its own JVP or VJP), ``"refused"`` not at all. ``lane_data`` is whether
    derivatives in the members' stored arrays are certified (otherwise only
    in the lane inputs), ``higher_order`` whether derivatives beyond the
    first are. Uncertified derivatives raise instead of evaluating the
    representative's rules for another member.
    """

    route: Literal["primal", "forward", "reverse", "refused"] = eqx.field(static=True)
    lane_data: bool = eqx.field(static=True)
    higher_order: bool = eqx.field(static=True)


@final
class LaneWorkset(StrictModule):
    """Members of one executable signature evaluated as bounded lanes.

    ``members`` are component names or law contribution keys in canonical
    order (the workset items); ``routes`` hold, for interface members, the
    ``(row, identified state, column)`` solve paths of each contribution
    block. ``lanes`` are the members' stacked array leaves in the bucket
    layout of ``worksets``; ``template`` is the representative with its array
    leaves removed, and ``arguments`` the argument signature every member's
    runtime arguments must reproduce. ``derivatives`` is what the lanes
    certify equal to the members' own derivatives.
    """

    kind: LaneKind = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    members: tuple[str, ...] = eqx.field(static=True)
    routes: tuple[tuple[OwnerPath, OwnerPath, OwnerPath], ...] = eqx.field(static=True)
    template: LaneSubject
    treedef: PyTreeDef = eqx.field(static=True)
    lanes: tuple[Array, ...]
    worksets: PreparedExecutionWorksets
    arguments: ArgumentSignature | None = eqx.field(static=True)
    estimate: LaneWorksetEstimate
    execution_group: ExecutionGroupSpec | None = eqx.field(static=True)
    derivatives: LaneDerivatives = eqx.field(static=True)


@final
class PreparedCoupledExecution(StrictModule):
    """Prepared lane worksets of one coupled problem and their resource evidence.

    ``components`` group components by executable signature; ``interfaces``
    group law contribution blocks of the native linear system. Members of no
    group (singletons, eliminated components, nonlinear or affine-residual
    contributions) execute per owner.
    """

    policy: CoupledExecutionPolicy
    components: tuple[LaneWorkset, ...]
    interfaces: tuple[LaneWorkset, ...]
    execution_id: str = eqx.field(static=True)

    @property
    def grouped_components(self) -> frozenset[str]:
        return frozenset(name for group in self.components for name in group.members)

    @property
    def grouped_contributions(self) -> frozenset[str]:
        return frozenset(key for group in self.interfaces for key in group.members)

    @property
    def lane_data_bytes(self) -> int:
        """Stacked lane data retained by every workset."""
        return sum(
            group.estimate.lane_data_bytes
            for group in (*self.components, *self.interfaces)
        )

    @property
    def max_working_set_bytes(self) -> int:
        """Largest estimated bucket working set of any workset (0 without groups)."""
        return max(
            (
                group.estimate.working_set_bytes
                for group in (*self.components, *self.interfaces)
            ),
            default=0,
        )


# --- Signatures -----------------------------------------------------------------------------


def argument_signature(arguments: object, /) -> ArgumentSignature | None:
    """Structure, array shapes/dtypes, and typed static leaves of runtime arguments.

    A static leaf enters with its qualified type, so values that compare equal
    across types (``2``, ``2.0``, ``True``) are different signatures. ``None``
    when a static leaf is unhashable: such arguments cannot be shown equal
    across members, so their component executes alone.
    """
    leaves, treedef = jax.tree.flatten(arguments)
    items: list[Hashable] = []
    for leaf in leaves:
        if eqx.is_array(leaf):
            items.append(("array", tuple(leaf.shape), str(jnp.result_type(leaf))))
        elif isinstance(leaf, Hashable):
            kind = type(leaf)
            items.append(("static", f"{kind.__module__}.{kind.__qualname__}", leaf))
        else:
            return None
    return treedef, tuple(items)


def _aval_bytes(value: jex_core.Var, /) -> int:
    aval = value.aval
    if not isinstance(aval, ShapedArray) or jax.dtypes.issubdtype(
        aval.dtype, jax.dtypes.extended
    ):
        return 0
    return math.prod(aval.shape) * aval.dtype.itemsize


def _traced_bytes(jaxpr: jex_core.Jaxpr, /) -> int:
    """Bytes of every intermediate value of a traced program (no buffer reuse)."""
    total = 0
    for equation in jaxpr.eqns:
        total += sum(_aval_bytes(value) for value in equation.outvars)
        total += sum(
            _traced_bytes(inner) for inner in jex_core.jaxprs_in_params(equation.params)
        )
    return total


def _tree_bytes(tree: PyTree[object], /) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
            total += math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize
    return total


@final
class _Opaque:
    """Identity of one object a traced program cannot show equal by value.

    Two tokens are equal only when they hold the same object. The token keeps
    its object alive while members are grouped, so identities are never
    reused.
    """

    __slots__ = ("value",)

    def __init__(self, value: object, /) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Opaque) and other.value is self.value

    def __hash__(self) -> int:
        return id(self.value)


# Primitives whose ``callback`` parameter is a host callable: the printed
# program shows its name only, so the callable itself enters the signature.
_HOST_CALLBACKS = frozenset({"pure_callback", "io_callback", "debug_callback"})
# Custom differentiation rules: the printed program shows the rule's name only.
# Their first-order content is traced into the JVP and VJP programs; a rule
# that survives there is certified at first order only.
_CUSTOM_RULES = frozenset(
    {"custom_jvp_call", "custom_vjp_call", "custom_lin", "custom_transpose_call"}
)
_ATOMS = (type(None), bool, int, float, complex, str, bytes)


def _value_token(value: object, depth: int, /) -> Hashable:
    """Token of a value a host callable captures: equal tokens, equal values.

    Immutable atoms compare by type and value, containers item by item, JAX
    shape records and tree structures by their own equality, functions
    structurally (``_callable_token``); every other object by identity.
    """
    if isinstance(value, _ATOMS):
        return (type(value).__qualname__, value)
    if isinstance(value, (tuple, list)):
        return (
            type(value).__qualname__,
            tuple(_value_token(item, depth) for item in value),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (_value_token(key, depth), _value_token(item, depth))
                for key, item in value.items()
            ),
        )
    if isinstance(value, (jax.ShapeDtypeStruct, PyTreeDef)):
        return (type(value).__qualname__, value)
    return _callable_token(value, depth)


def _callable_token(value: object, depth: int = 0, /) -> Hashable:
    """Structural token of a host callable; identity when it cannot be opened.

    Two plain functions are the same function when they share code and
    globals and their defaults and captured values are equal (tokens above);
    ``functools.partial`` and bound methods are opened the same way, and
    JAX's dataclass wrappers (the flattened callback of ``pure_callback``)
    field by field. Anything else, or nesting beyond a fixed depth, compares
    by identity, so an unrecognized callable never merges two members.
    """
    if depth > 8:
        return _Opaque(value)
    depth += 1
    if isinstance(value, types.FunctionType):
        cells = []
        for cell in value.__closure__ or ():
            try:
                cells.append(_value_token(cell.cell_contents, depth))
            except ValueError:
                cells.append("empty-cell")
        return (
            "function",
            _Opaque(value.__code__),
            _Opaque(value.__globals__),
            _value_token(value.__defaults__, depth),
            _value_token(value.__kwdefaults__, depth),
            tuple(cells),
        )
    if isinstance(value, functools.partial):
        return (
            "partial",
            _callable_token(value.func, depth),
            _value_token(value.args, depth),
            _value_token(value.keywords, depth),
        )
    if isinstance(value, types.MethodType):
        return ("method", _callable_token(value.__func__, depth), _Opaque(value.__self__))
    kind = type(value)
    if dataclasses.is_dataclass(value) and kind.__module__.startswith("jax."):
        return (
            f"{kind.__module__}.{kind.__qualname__}",
            tuple(
                (field.name, _value_token(getattr(value, field.name), depth))
                for field in dataclasses.fields(value)
            ),
        )
    return _Opaque(value)


def _host_callables(jaxpr: jex_core.Jaxpr, /) -> tuple[Hashable, ...]:
    """Tokens of every host callback a program calls, in program order."""
    tokens: list[Hashable] = []
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in _HOST_CALLBACKS:
            tokens.append((name, _callable_token(equation.params["callback"])))
        for inner in jex_core.jaxprs_in_params(equation.params):
            tokens.extend(_host_callables(inner))
    return tuple(tokens)


def _contains(jaxpr: jex_core.Jaxpr, names: frozenset[str], /) -> bool:
    return any(
        equation.primitive.name in names
        or any(
            _contains(inner, names)
            for inner in jex_core.jaxprs_in_params(equation.params)
        )
        for equation in jaxpr.eqns
    )


def _derivative_program(
    function: Callable[..., PyTree[Array]], *specs: PyTree[jax.ShapeDtypeStruct]
) -> jex_core.ClosedJaxpr | None:
    """One derivative program, or ``None`` when the transformation is refused.

    A refused transformation fails the same way for the member and its lane
    (reverse mode of a while loop, a host callback without a rule); forward
    mode through a custom_vjp rule traces to a deferred linearization that
    refuses evaluation.
    """
    try:
        program = jax.make_jaxpr(function)(*specs)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if _contains(program.jaxpr, frozenset({"custom_lin"})):
        return None
    return program


def _derivative_programs(
    lane: Callable[[list[Array], PyTree[Array]], PyTree[Array]],
    leaves: list[Array],
    inputs: PyTree[Array],
    lane_data: bool,
    /,
) -> tuple[jex_core.ClosedJaxpr | None, jex_core.ClosedJaxpr | None]:
    """First-order JVP and VJP programs of ``lane`` in its inexact arrays.

    The derivative is taken in the inexact inputs and, with ``lane_data``, in
    the member's own inexact arrays too; every array stays a program argument
    (never a hoisted constant), as in the primal program. Custom
    differentiation rules are traced into these programs as ordinary
    equations, so equal derivative programs certify equal first derivatives
    even when the primal programs print the same rule name.
    """
    differentiated = (
        jax.tree.map(lambda leaf: lane_data and eqx.is_inexact_array(leaf), leaves),
        jax.tree.map(eqx.is_inexact_array, inputs),
    )
    diff, fixed = eqx.partition((leaves, inputs), differentiated)

    def primal(constant: PyTree[Array], values: PyTree[Array]) -> PyTree[Array]:
        member_leaves, member_inputs = eqx.combine(values, constant)
        return eqx.filter(lane(member_leaves, member_inputs), eqx.is_inexact_array)

    def specs(tree: PyTree[Array]) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), tree
        )

    fixed_specs, diff_specs = specs(fixed), specs(diff)
    cotangents = jax.eval_shape(primal, fixed_specs, diff_specs)
    forward = _derivative_program(
        lambda constant, values, tangents: jax.jvp(
            functools.partial(primal, constant), (values,), (tangents,)
        ),
        fixed_specs,
        diff_specs,
        diff_specs,
    )
    reverse = _derivative_program(
        lambda constant, values, cotangent: jax.vjp(
            functools.partial(primal, constant), values
        )[1](cotangent),
        fixed_specs,
        diff_specs,
        cotangents,
    )
    return forward, reverse


def _program_record(program: jex_core.ClosedJaxpr | None, /) -> object:
    if program is None:
        return "refused"
    return {
        "program": str(program.jaxpr),
        "constants": array_tree_fingerprint(list(program.consts)),
    }


def _certified_derivatives(
    lane: Callable[[list[Array], PyTree[Array]], PyTree[Array]],
    leaves: list[Array],
    inputs: PyTree[Array],
    primal: jex_core.ClosedJaxpr,
    /,
) -> tuple[LaneDerivatives, tuple[jex_core.ClosedJaxpr | None, ...]]:
    """The derivatives a lane program certifies and the programs certifying them.

    A program without custom rules is differentiated by the standard rules of
    its primitives, so its primal program certifies every derivative.
    Otherwise the first-order programs are traced in every inexact array,
    or, when that is refused (a rule that fixes the member's data), in the
    inputs only; a custom rule surviving in them (one applied to values with
    no first-order tangent) leaves higher orders uncertified.
    """
    if not _contains(primal.jaxpr, _CUSTOM_RULES):
        return LaneDerivatives(route="primal", lane_data=True, higher_order=True), ()
    for lane_data in (True, False):
        forward, reverse = _derivative_programs(lane, leaves, inputs, lane_data)
        present = tuple(item for item in (forward, reverse) if item is not None)
        if not present:
            continue
        higher_order = not any(_contains(item.jaxpr, _CUSTOM_RULES) for item in present)
        route: Literal["primal", "forward", "reverse"]
        if higher_order and lane_data:
            route = "primal"
        else:
            route = "forward" if forward is not None else "reverse"
        return (
            LaneDerivatives(route=route, lane_data=lane_data, higher_order=higher_order),
            (forward, reverse),
        )
    return LaneDerivatives(route="refused", lane_data=False, higher_order=False), ()


@final
class _Traced(StrictModule):
    """One member's lane program: signature, leaves, and traced sizes.

    ``callables`` holds the tokens of its host callbacks, which must match
    with ``signature_id``; ``derivatives`` is what its lanes certify.
    """

    key: str = eqx.field(static=True)
    route: tuple[OwnerPath, OwnerPath, OwnerPath] | None = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    callables: tuple[Hashable, ...] = eqx.field(static=True)
    derivatives: LaneDerivatives = eqx.field(static=True)
    arguments: ArgumentSignature | None = eqx.field(static=True)
    leaves: tuple[Array, ...]
    treedef: PyTreeDef = eqx.field(static=True)
    template: LaneSubject
    input_bytes: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    intermediate_bytes: int = eqx.field(static=True)


def _trace(
    key: str,
    route: tuple[OwnerPath, OwnerPath, OwnerPath] | None,
    subject: LaneSubject,
    program: LaneProgram,
    inputs: PyTree[Array],
    arguments: ArgumentSignature | None,
    /,
) -> _Traced:
    """Trace one member's lane program with its array leaves as arguments.

    The signature is the primal program, the first-order JVP and VJP programs
    of a program with custom differentiation rules (the rules expanded into
    ordinary equations), their hoisted constants, and the tokens of every
    host callback. Members sharing it compute the same values and the
    derivatives ``LaneDerivatives`` certifies; lanes refuse the others.
    """
    dynamic, template = eqx.partition(subject, eqx.is_array)
    leaves, treedef = jax.tree.flatten(dynamic)

    def lane(lane_leaves: list[Array], lane_inputs: PyTree[Array]) -> PyTree[Array]:
        member = eqx.combine(jax.tree.unflatten(treedef, lane_leaves), template)
        return program(member, lane_inputs)

    closed, outputs = jax.make_jaxpr(lane, return_shape=True)(leaves, inputs)
    derivatives, programs = _certified_derivatives(lane, leaves, inputs, closed)
    signature_id = canonical_fingerprint(
        {
            "kind": "coupled-lane-program",
            "primal": _program_record(closed),
            "derivatives": [
                derivatives.route,
                derivatives.lane_data,
                derivatives.higher_order,
            ],
            "first-order": [_program_record(item) for item in programs],
        }
    )
    return _Traced(
        key=key,
        route=route,
        signature_id=signature_id,
        callables=tuple(
            token
            for item in (closed, *programs)
            if item is not None
            for token in _host_callables(item.jaxpr)
        ),
        derivatives=derivatives,
        arguments=arguments,
        leaves=tuple(leaves),
        treedef=treedef,
        template=template,
        input_bytes=_tree_bytes(inputs),
        output_bytes=_tree_bytes(outputs),
        intermediate_bytes=_traced_bytes(closed.jaxpr),
    )


# --- Lane programs --------------------------------------------------------------------------


def _component_arguments(dynamic: PyTree[Array], shared: object, /) -> object:
    return eqx.combine(dynamic, shared)


def component_residual_program(shared: object, /) -> LaneProgram:
    """Native residual rows of one component lane (inputs ``(state, dynamic args)``)."""

    def program(subject: LaneSubject, inputs: PyTree[Array]) -> PyTree[Array]:
        component = _component(subject)
        state, dynamic = inputs
        return component.residual(state, _component_arguments(dynamic, shared))

    return program


def component_fields_program(shared: object, /) -> LaneProgram:
    """Full coefficients ``P z + g`` of every field of one component lane."""

    def program(subject: LaneSubject, inputs: PyTree[Array]) -> PyTree[Array]:
        component = _component(subject)
        state, dynamic = inputs
        args = _component_arguments(dynamic, shared)
        return tuple(
            component.expand(record.name, state, args) for record in component.fields
        )

    return program


def _identify(
    component: AbstractSpatialComponent, rows: tuple[Array, ...], /
) -> tuple[Array, ...]:
    return tuple(
        block.space.inverse_riesz(value)
        for block, value in zip(component.state_blocks, rows, strict=True)
    )


def transposed_inverse_riesz(
    space: AbstractVectorSpace, value: PyTree[Array], /
) -> PyTree[Array]:
    """Coordinate transpose ``M^{-T} y`` of the inverse Riesz map ``M^{-1}``.

    Riesz maps are Hermitian positive definite, so ``M^{-T} = conj(M^{-1})``
    and ``M^{-T} y = conj(M^{-1} conj(y))``; for real coordinates this is
    ``M^{-1} y``. It is never ``M``: that is the transpose of ``M``, not of
    the identification ``M^{-1}``.
    """
    return jax.tree.map(jnp.conj, space.inverse_riesz(jax.tree.map(jnp.conj, value)))


def _transposed_identify(
    component: AbstractSpatialComponent, states: tuple[Array, ...], /
) -> tuple[Array, ...]:
    return tuple(
        transposed_inverse_riesz(block.space, value)
        for block, value in zip(component.state_blocks, states, strict=True)
    )


def _affine_operator(
    component: AbstractSpatialComponent, args: object, /
) -> BlockLinearOperator:
    operator = component.linear_operator(args)
    if operator is None:
        raise ValueError(
            f"Component {component.name!r} publishes no affine operator; solve the "
            "coupled problem through its nonlinear residual."
        )
    return operator


def _operator_action(
    component: AbstractSpatialComponent,
    operator: BlockLinearOperator,
    value: tuple[Array, ...],
    action: LaneAction,
    /,
) -> tuple[Array, ...]:
    match action:
        case "weak":
            return operator.mv(value)
        case "weak-transpose":
            return operator.transpose_mv(value)
        case "identified":
            return _identify(component, operator.mv(value))
        case "identified-transpose":
            return operator.transpose_mv(_transposed_identify(component, value))
        case _:
            assert_never(action)


def component_action_program(shared: object, action: LaneAction, /) -> LaneProgram:
    """One action of a component's own affine operator block (inputs ``(x, args)``)."""
    action_ = parse(action, LaneAction, "action")

    def program(subject: LaneSubject, inputs: PyTree[Array]) -> PyTree[Array]:
        value, dynamic = inputs
        component = _component(subject)
        operator = _affine_operator(component, _component_arguments(dynamic, shared))
        return _operator_action(component, operator, value, action_)

    return program


def _component_signature_program(shared: object, affine: bool, /) -> LaneProgram:
    """Every lane operation of a component in one program (the grouping certificate)."""

    def program(subject: LaneSubject, inputs: PyTree[Array]) -> PyTree[Array]:
        component = _component(subject)
        state, rows, dynamic = inputs
        args = _component_arguments(dynamic, shared)
        residual = component.residual(state, args)
        fields = tuple(
            component.expand(record.name, state, args) for record in component.fields
        )
        if not affine:
            return residual, fields
        # One operator construction serves every action; each output is the
        # same function of the inputs as the per-action runtime programs.
        operator = _affine_operator(component, args)
        actions = (
            _operator_action(component, operator, state, "weak"),
            _operator_action(component, operator, rows, "weak-transpose"),
            _operator_action(component, operator, state, "identified"),
            _operator_action(component, operator, state, "identified-transpose"),
        )
        return residual, fields, actions

    return program


def _interface_action(
    subject: InterfaceLaneSubject, value: Array, action: LaneAction, /
) -> Array:
    operator = subject.operator
    match action:
        case "weak":
            return operator.mv(value)
        case "weak-transpose":
            return operator.transpose_mv(value)
        case "identified":
            return subject.identified.inverse_riesz(operator.mv(value))
        case "identified-transpose":
            return operator.transpose_mv(
                transposed_inverse_riesz(subject.identified, value)
            )
        case _:
            assert_never(action)


def interface_action_program(action: LaneAction, /) -> LaneProgram:
    """One action of a law contribution block in solve coordinates."""
    action_ = parse(action, LaneAction, "action")

    def program(subject: LaneSubject, inputs: PyTree[Array]) -> PyTree[Array]:
        return _interface_action(_interface(subject), inputs, action_)

    return program


def _interface_signature_program(
    subject: LaneSubject, inputs: PyTree[Array], /
) -> PyTree[Array]:
    interface = _interface(subject)
    source, row, state = inputs
    return (
        _interface_action(interface, source, "weak"),
        _interface_action(interface, row, "weak-transpose"),
        _interface_action(interface, source, "identified"),
        _interface_action(interface, state, "identified-transpose"),
    )


def _component(subject: LaneSubject, /) -> AbstractSpatialComponent:
    if not isinstance(subject, AbstractSpatialComponent):
        raise TypeError("A component lane program acts on a component.")
    return subject


def _interface(subject: LaneSubject, /) -> InterfaceLaneSubject:
    if not isinstance(subject, InterfaceLaneSubject):
        raise TypeError("An interface lane program acts on an InterfaceLaneSubject.")
    return subject


# --- Preparation ----------------------------------------------------------------------------


def _precision(leaves: Sequence[tuple[Array, ...]], /) -> str:
    dtypes = sorted({str(leaf.dtype) for member in leaves for leaf in member})
    return ",".join(dtypes) if dtypes else "no-lane-data"


def _estimate(
    members: Sequence[_Traced],
    capacity: int,
    bucket_count: int,
    /,
) -> LaneWorksetEstimate:
    representative = members[0]
    per_lane = _tree_bytes(representative.leaves)
    lanes = bucket_count * capacity
    per_lane_work = (
        per_lane
        + representative.input_bytes
        + representative.output_bytes
        + representative.intermediate_bytes
    )
    return LaneWorksetEstimate(
        lane_count=len(members),
        bucket_count=bucket_count,
        lane_capacity=capacity,
        padded_lanes=lanes - len(members),
        lane_data_bytes=lanes * per_lane,
        lane_input_bytes=representative.input_bytes,
        lane_output_bytes=representative.output_bytes,
        traced_intermediate_bytes=representative.intermediate_bytes,
        working_set_bytes=capacity * per_lane_work,
    )


def _require_bound(
    kind: LaneKind,
    members: Sequence[_Traced],
    estimate: LaneWorksetEstimate,
    policy: CoupledExecutionPolicy,
    /,
) -> None:
    bound = policy.max_working_set_bytes
    if bound is not None and estimate.working_set_bytes > bound:
        names = ", ".join(member.key for member in members)
        raise ValueError(
            f"The {kind} lane workset of ({names}) needs an estimated "
            f"{estimate.working_set_bytes} bytes per bucket of {estimate.lane_capacity} "
            f"lanes, above the declared working-set bound of {bound} bytes; lower "
            "lane_capacity or raise max_working_set_bytes."
        )


def _workset(
    kind: LaneKind,
    signature_id: str,
    members: Sequence[_Traced],
    policy: CoupledExecutionPolicy,
    /,
) -> LaneWorkset:
    """Bind one signature group to fixed-capacity worksets, refusing it above the bound."""
    ordered = sorted(members, key=lambda member: member.key)
    count = len(ordered)
    capacity = (
        policy.lane_capacity
        if policy.execution_group is not None
        else min(policy.lane_capacity, count)
    )
    bucket_count = -(-count // capacity)
    estimate = _estimate(ordered, capacity, bucket_count)
    _require_bound(kind, ordered, estimate, policy)
    representative = ordered[0]
    signature = PoolExecutionSignature(
        topology_id=signature_id,
        method_id=f"coupled-{kind}-lanes",
        precision_id=_precision([member.leaves for member in ordered]),
        backend_id=jax.default_backend(),
        execution_group=policy.execution_group,
    )
    keys = tuple(member.key for member in ordered)
    worksets = ExecutionWorksetPlan(
        keys, (signature,) * count, bucket_capacity=capacity
    ).prepare()
    if worksets.bucket_count != bucket_count:
        raise RuntimeError(
            "One signature group must fill ceil(count / capacity) buckets."
        )
    lanes = (
        tuple(
            worksets.gather(
                tuple(
                    jnp.stack([member.leaves[index] for member in ordered])
                    for index in range(len(representative.leaves))
                )
            )
        )
        if representative.leaves
        else ()
    )
    return LaneWorkset(
        kind=kind,
        signature_id=signature_id,
        members=keys,
        routes=tuple(member.route for member in ordered if member.route is not None),
        template=representative.template,
        treedef=representative.treedef,
        lanes=lanes,
        worksets=worksets,
        arguments=representative.arguments,
        estimate=estimate,
        execution_group=policy.execution_group,
        derivatives=representative.derivatives,
    )


def _grouped(
    kind: LaneKind, traced: Sequence[_Traced], policy: CoupledExecutionPolicy, /
) -> tuple[LaneWorkset, ...]:
    groups: dict[
        tuple[str, tuple[Hashable, ...], ArgumentSignature | None], list[_Traced]
    ] = {}
    for member in traced:
        groups.setdefault(
            (member.signature_id, member.callables, member.arguments), []
        ).append(member)
    worksets = tuple(
        _workset(kind, signature_id, members, policy)
        for (signature_id, _, _), members in groups.items()
        if len(members) > 1
    )
    return tuple(sorted(worksets, key=lambda workset: workset.members[0]))


def _zeros(blocks: Sequence[AbstractVectorSpace], /) -> tuple[Array, ...]:
    return tuple(space.zeros() for space in blocks)


def _component_traces(
    components: Sequence[AbstractSpatialComponent],
    args: Mapping[str, object],
    /,
) -> tuple[_Traced, ...]:
    traced: list[_Traced] = []
    for component in components:
        arguments = args[component.name]
        signature = argument_signature(arguments)
        if signature is None:
            continue
        dynamic, shared = eqx.partition(arguments, eqx.is_array)
        affine = component.linear_operator(arguments) is not None
        inputs = (
            _zeros([block.space for block in component.state_blocks]),
            _zeros([block.space for block in component.row_blocks]),
            dynamic,
        )
        traced.append(
            _trace(
                component.name,
                None,
                component,
                _component_signature_program(shared, affine),
                inputs,
                signature,
            )
        )
    return tuple(traced)


def _interface_traces(
    candidates: Sequence[
        tuple[str, tuple[OwnerPath, OwnerPath, OwnerPath], InterfaceLaneSubject]
    ],
    /,
) -> tuple[_Traced, ...]:
    return tuple(
        _trace(
            key,
            route,
            subject,
            _interface_signature_program,
            (
                subject.operator.source.zeros(),
                subject.operator.target.zeros(),
                subject.identified.zeros(),
            ),
            None,
        )
        for key, route, subject in candidates
    )


def prepare_coupled_execution(
    components: Sequence[AbstractSpatialComponent],
    args: Mapping[str, object],
    interfaces: Sequence[
        tuple[str, tuple[OwnerPath, OwnerPath, OwnerPath], InterfaceLaneSubject]
    ],
    policy: CoupledExecutionPolicy,
    /,
) -> PreparedCoupledExecution:
    """Group eligible components and contribution blocks into bounded lane worksets.

    ``components`` are the components whose solve blocks are their owner
    blocks (no eliminated rows) with their preparation ``args``;
    ``interfaces`` are ``(key, (row, identified state, column), subject)``
    contribution blocks of the native linear system. A group above the
    policy's working-set bound is refused before its lanes are stacked.
    """
    if not isinstance(policy, CoupledExecutionPolicy):
        raise TypeError("policy must be a CoupledExecutionPolicy.")
    for key, _, _ in interfaces:
        canonical_identifier(key, "contribution key")
    component_groups = _grouped("component", _component_traces(components, args), policy)
    interface_groups = _grouped("interface", _interface_traces(interfaces), policy)
    execution_id = canonical_fingerprint(
        {
            "kind": "prepared-coupled-execution",
            "lane_capacity": policy.lane_capacity,
            "max_working_set_bytes": policy.max_working_set_bytes,
            "execution_group": None
            if policy.execution_group is None
            else policy.execution_group.group_id,
            "groups": [
                {
                    "kind": group.kind,
                    "signature": group.signature_id,
                    "members": list(group.members),
                    "worksets": group.worksets.prepared_id,
                }
                for group in (*component_groups, *interface_groups)
            ],
        }
    )
    return PreparedCoupledExecution(
        policy=policy,
        components=component_groups,
        interfaces=interface_groups,
        execution_id=execution_id,
    )


# --- Evaluation -----------------------------------------------------------------------------


def lane_arguments(
    workset: LaneWorkset, arguments: Sequence[object], /
) -> tuple[object, tuple[PyTree[Array], ...]]:
    """Shared static part and per-lane arrays of the members' runtime arguments.

    Every member must reproduce the argument signature the group was prepared
    with; otherwise the certified executable does not cover the call.
    """
    if workset.arguments is None:
        raise ValueError("Interface lane worksets take no component arguments.")
    for name, value in zip(workset.members, arguments, strict=True):
        if argument_signature(value) != workset.arguments:
            raise ValueError(
                f"Runtime arguments of lane-grouped component {name!r} differ in "
                "structure, shape, dtype, or static leaves from the prepared lane "
                "signature; re-prepare with these arguments."
            )
    parts = [eqx.partition(value, eqx.is_array) for value in arguments]
    return parts[0][1], tuple(dynamic for dynamic, _ in parts)


def _lane_sharding(spec: ExecutionGroupSpec | None, /) -> NamedSharding | None:
    if spec is None:
        return None
    group = bind_execution_group(spec)
    axes = tuple(name for name, _ in spec.mesh_axes) or ("device",)
    return group.named_sharding(PartitionSpec(axes))


_UNCERTIFIED = (
    "This derivative of a lane-grouped coupled execution is not certified: its "
    "members share values and certified derivatives, but may differ in custom "
    "differentiation rules beyond them (see LaneWorkset.derivatives). "
    "Differentiate a per-owner execution (no CoupledExecutionPolicy) instead."
)


@jax.custom_jvp
def _underivable(values: PyTree[Array], /) -> PyTree[Array]:
    """Identity whose differentiation raises (an uncertified derivative)."""
    return values


@_underivable.defjvp
def _refuse_derivative(
    primals: tuple[PyTree[Array]], tangents: tuple[PyTree[Array]]
) -> tuple[PyTree[Array], PyTree[Array]]:
    del primals, tangents
    raise ValueError(_UNCERTIFIED)


def _symbolic_zero(tangent: object, /) -> bool:
    return isinstance(tangent, jax.custom_derivatives.SymbolicZero)


def _live_tangent(tangent: object, /) -> bool:
    return not _symbolic_zero(tangent) and (jnp.result_type(tangent) != jax.dtypes.float0)


type _LaneRun = Callable[[PyTree[Array], PyTree[Array]], PyTree[Array]]


def _guarded(run: _LaneRun, derivatives: LaneDerivatives, /) -> _LaneRun:
    """``run(data, inputs)`` differentiable exactly as ``derivatives`` certifies.

    A first-derivative rule applies ``run``'s own JVP or VJP at primals routed
    through ``_underivable`` where a further derivative is uncertified, so
    that derivative raises instead of evaluating the representative's custom
    rules for another member. The guard wraps the whole bucket loop: JAX's
    partial evaluation of a loop body differentiates custom rules nested in it
    through their primal functions at second order, which would bypass a
    guard placed inside the loop.
    """

    def certified(
        data: PyTree[Array], inputs: PyTree[Array]
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        if not derivatives.higher_order:
            return _underivable((data, inputs))
        return data, inputs

    match derivatives.route:
        case "primal":
            return run
        case "refused":

            @jax.custom_jvp
            def refused(data: PyTree[Array], inputs: PyTree[Array]) -> PyTree[Array]:
                return run(data, inputs)

            @refused.defjvp
            def _refused_rule(
                primals: tuple[PyTree[Array], PyTree[Array]],
                tangents: tuple[PyTree[Array], PyTree[Array]],
            ) -> tuple[PyTree[Array], PyTree[Array]]:
                del primals, tangents
                raise ValueError(_UNCERTIFIED)

            return refused
        case "forward":

            @jax.custom_jvp
            def forward(data: PyTree[Array], inputs: PyTree[Array]) -> PyTree[Array]:
                return run(data, inputs)

            def forward_rule(
                primals: tuple[PyTree[Array], PyTree[Array]],
                tangents: tuple[PyTree[Array], PyTree[Array]],
            ) -> tuple[PyTree[Array], PyTree[Array]]:
                live = jax.tree.map(_live_tangent, tangents, is_leaf=_symbolic_zero)
                if not derivatives.lane_data and any(jax.tree.leaves(live[0])):
                    raise ValueError(_UNCERTIFIED)
                primals = certified(*primals)
                if not derivatives.lane_data:
                    primals = (_underivable(primals[0]), primals[1])
                fixed = eqx.filter(primals, live, inverse=True)

                def restricted(values: PyTree[Array]) -> PyTree[Array]:
                    return run(*eqx.combine(values, fixed))

                return jax.jvp(
                    restricted,
                    (eqx.filter(primals, live),),
                    (eqx.filter(tangents, live, is_leaf=_symbolic_zero),),
                )

            forward.defjvp(forward_rule, symbolic_zeros=True)
            return forward
        case "reverse":

            def reverse(data: PyTree[Array], inputs: PyTree[Array]) -> PyTree[Array]:
                # Without certified lane-data derivatives the data is closed
                # over, and JAX refuses a derivative in it.
                @jax.custom_vjp
                def apply(values: tuple[PyTree[Array], PyTree[Array]]) -> PyTree[Array]:
                    return run(*_joined(data, values))

                def apply_forward(
                    values: tuple[PyTree[Array], PyTree[Array]],
                ) -> tuple[PyTree[Array], tuple[PyTree[Array], PyTree[Array]]]:
                    return run(*_joined(data, values)), values

                def apply_backward(
                    values: tuple[PyTree[Array], PyTree[Array]],
                    cotangent: PyTree[Array],
                ) -> tuple[tuple[PyTree[Array], PyTree[Array]]]:
                    values = certified(*values)
                    return jax.vjp(lambda item: run(*_joined(data, item)), values)[1](
                        cotangent
                    )

                apply.defvjp(apply_forward, apply_backward)
                return apply((data, inputs) if derivatives.lane_data else (None, inputs))

            return reverse
        case _:
            assert_never(derivatives.route)


def _joined(
    data: PyTree[Array], values: tuple[PyTree[Array], PyTree[Array]], /
) -> tuple[PyTree[Array], PyTree[Array]]:
    """``(data, inputs)`` from differentiated values, data closed over when absent."""
    differentiated, inputs = values
    return (data if differentiated is None else differentiated), inputs


def evaluate_lanes(
    workset: LaneWorkset,
    program: LaneProgram,
    inputs: Sequence[PyTree[Array]],
    /,
) -> tuple[PyTree[Array], ...]:
    """Evaluate ``program`` for every member: buckets in sequence, lanes vectorized.

    ``inputs`` holds one dynamic input tree per member in canonical member
    order; the result holds one output tree per member in the same order.
    Padded lanes repeat a valid lane and are masked out when scattering back.
    Derivatives ``workset.derivatives`` does not certify raise.
    """
    if len(inputs) != len(workset.members):
        raise ValueError("Lane inputs must hold one entry per workset member.")
    items = jax.tree.map(lambda *values: jnp.stack(values), *inputs)
    buckets = workset.worksets.gather(items)
    template, treedef = workset.template, workset.treedef
    sharding = _lane_sharding(workset.execution_group)

    def lane(leaves: tuple[Array, ...], lane_inputs: PyTree[Array]) -> PyTree[Array]:
        member = eqx.combine(jax.tree.unflatten(treedef, list(leaves)), template)
        return program(member, lane_inputs)

    def bucket(
        values: tuple[tuple[Array, ...], PyTree[Array]],
    ) -> PyTree[Array]:
        if sharding is not None:
            values = jax.lax.with_sharding_constraint(values, sharding)
        return jax.vmap(lane)(*values)

    def run(data: PyTree[Array], lane_inputs: PyTree[Array]) -> PyTree[Array]:
        return jax.lax.map(bucket, (data, lane_inputs))

    outputs = _guarded(run, workset.derivatives)(workset.lanes, buckets)
    scattered = workset.worksets.scatter(outputs)
    return tuple(
        jax.tree.map(lambda value, index=index: value[index], scattered)
        for index in range(len(inputs))
    )


__all__ = [
    "CoupledExecutionPolicy",
    "InterfaceLaneSubject",
    "LaneAction",
    "LaneDerivatives",
    "LaneKind",
    "LaneWorkset",
    "LaneWorksetEstimate",
    "PreparedCoupledExecution",
    "prepare_coupled_execution",
]
