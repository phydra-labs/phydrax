"""Structural identity of a fuel numerical execution template.

Two prepared rods share one compiled fuel step only when their numerical methods
have identical structure: types, static selectors, source IDs, field/space
roles, FE topology, compiler programs, and array shapes and dtypes. Per-case
scientific receipts (prepared, bundle, specification, ... IDs) are redacted, so
rods that differ only in those receipts and in numerical array values share the
compiled program. Array values never enter the structure.

The payload is a content-addressed node table: every shared container is
serialized once, and references between nodes use the node fingerprint, so
Python object identity never changes the result.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from inspect import Signature
from types import MethodType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox._ad import _ClosureConvert
from jax import Array
from jax.extend import core as jax_core

from ..._fingerprint import canonical_fingerprint
from ..._identity import _CompilerPayload
from ...discretization._spaces import DiscreteFieldSpace
from ...discretization.fem import PreparedFiniteElementPointInterpolation
from ...exterior import FormType, FormValueSpec
from ...nonlinear import PreparedNonlinearSolve
from ._coolant import PrescribedWallTemperature


# `Any` is confined to this module: the serializer visits heterogeneous
# registered JAX/Equinox leaves and arbitrary dataclass metadata. Numerical entry
# points remain array-only.

type ArrayLayout = tuple[tuple[tuple[int, ...], str], ...]

# Per-case scientific receipts. Their values are replaced by the field name so
# that cases differing only in these IDs share one compiled step.
_RECEIPT_FIELDS = frozenset(
    (
        "prepared_id",
        "admission_id",
        "specification_id",
        "geometry_id",
        "history_id",
        "bundle_id",
        "law_id",
        "method_id",
        "problem_id",
        "reference_id",
        "linear_plan_id",
        "template_id",
        "operator_id",
        "linearization_id",
        "plan_id",
    )
)

# Native refresh replaces these `PreparedNonlinearSolve` fields before every solve.
_REFRESHED_SOLVE_FIELDS = ("jacobian", "run")


def _execution_layout_signature(signature: tuple[Any, ...], /) -> tuple[Any, ...]:
    """Drop the bundle receipt from `PreparedFuelRod.execution_signature`.

    The signature is `(reduction, layer_count, fuel_node_count,
    cladding_node_count, quadrature_order, gas_mode_count, helium_cell_count,
    bundle_id)`; only the leading layout entries constrain numerical sharing.
    """
    (
        reduction,
        layer_count,
        fuel_node_count,
        cladding_node_count,
        quadrature_order,
        gas_mode_count,
        helium_cell_count,
        _bundle_id,
    ) = signature
    return (
        reduction,
        layer_count,
        fuel_node_count,
        cladding_node_count,
        quadrature_order,
        gas_mode_count,
        helium_cell_count,
    )


def _is_shared_node(value: Any, /) -> bool:
    """Whether `value` is a container serialized once into the node table."""
    return (
        dataclasses.is_dataclass(value)
        or eqx.is_array(value)
        or isinstance(value, (Mapping, tuple, list, MethodType, Signature))
    )


def _type_name(value: Any, /) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


class _FuelStructurePayload:
    """Content-addressed node table for one structural serialization."""

    def __init__(self) -> None:
        self.compiler = _CompilerPayload()
        self.nodes: dict[str, Any] = {}
        # Memo values retain the visited object so its `id` cannot be reused.
        self.memo: dict[tuple[int, bool, bool], tuple[Any, Any]] = {}
        self.active: set[tuple[int, bool, bool]] = set()

    def structure(
        self,
        value: Any,
        /,
        *,
        field_name: str = "",
        topology: bool = True,
        dynamic_captures: bool = False,
    ) -> Any:
        if field_name in _RECEIPT_FIELDS:
            return {"scientific_receipt": field_name}
        if field_name == "execution_signature":
            return _execution_layout_signature(value)
        if not _is_shared_node(value):
            return _structure_payload(
                value, self, topology=topology, dynamic_captures=dynamic_captures
            )
        key = (id(value), topology, dynamic_captures)
        visited = self.memo.get(key)
        if visited is not None:
            return visited[1]
        if key in self.active:
            raise TypeError("Fuel execution metadata must not contain reference cycles.")
        self.active.add(key)
        payload = _structure_payload(
            value, self, topology=topology, dynamic_captures=dynamic_captures
        )
        self.active.remove(key)
        node_id = canonical_fingerprint(payload)
        self.nodes.setdefault(node_id, payload)
        reference = {"native_node_id": node_id}
        self.memo[key] = (value, reference)
        return reference

    def finish(self, root: Any, /) -> Any:
        return {
            "native_structure": root,
            "native_nodes": tuple((key, self.nodes[key]) for key in sorted(self.nodes)),
            "execution_metadata": self.compiler.definitions(),
        }


def execution_structure(value: Any, /, *, topology: bool = True) -> Any:
    """Serialize the execution structure of `value`.

    With `topology`, integer arrays contribute the fingerprint of their values
    (FE connectivity and routes); otherwise only shapes and dtypes enter.
    """
    context = _FuelStructurePayload()
    return context.finish(context.structure(value, topology=topology))


def _structure_payload(
    value: Any,
    context: _FuelStructurePayload,
    /,
    *,
    topology: bool,
    dynamic_captures: bool,
) -> Any:
    # Order matters: specific fuel and solver records precede the generic
    # dataclass, mapping, sequence, and callable encodings they would match.
    if isinstance(value, PreparedNonlinearSolve):
        return _refreshed_solve_structure(value, context, topology=topology)
    if isinstance(value, PreparedFiniteElementPointInterpolation):
        return _point_route_structure(value, context, topology=topology)
    if isinstance(value, PrescribedWallTemperature):
        return (
            "prescribed-wall-temperature",
            context.structure(value.times, topology=topology),
            context.structure(value.wall_temperatures, topology=topology),
        )
    if isinstance(value, (FormType, FormValueSpec)):
        return {"type": _type_name(value), "identity": value.to_dict()}
    if isinstance(value, Signature):
        return _signature_structure(value, context, topology=topology)
    if isinstance(value, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
        return context.compiler.program(
            value, "execution", dynamic_captures=dynamic_captures
        )
    if isinstance(value, (jax.tree_util.PyTreeDef, jax.ShapeDtypeStruct)):
        return context.compiler.parameter(value, "execution")
    if eqx.is_array(value):
        return _array_structure(value, topology=topology)
    if isinstance(value, MethodType):
        return (
            context.structure(value.__func__, topology=topology),
            context.structure(value.__self__, topology=topology),
        )
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    if isinstance(value, float):
        return {"static_float": value.hex()}
    if isinstance(value, Mapping):
        return tuple(
            (
                context.structure(key, topology=topology),
                context.structure(item, field_name=str(key), topology=topology),
            )
            for key, item in sorted(value.items())
        )
    if dataclasses.is_dataclass(value):
        return _dataclass_structure(value, context, topology=topology)
    if isinstance(value, (tuple, list)):
        return tuple(context.structure(item, topology=topology) for item in value)
    if callable(value):
        return (value.__module__, value.__qualname__)
    return value


def _array_structure(value: Any, /, *, topology: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"shape": value.shape, "dtype": str(value.dtype)}
    if topology and jnp.issubdtype(value.dtype, jnp.integer):
        result["topology"] = canonical_fingerprint(np.asarray(value))
    return result


def _signature_structure(
    value: Signature, context: _FuelStructurePayload, /, *, topology: bool
) -> dict[str, Any]:
    return {
        "signature": tuple(
            (
                name,
                int(parameter.kind),
                context.structure(parameter.default, topology=topology),
                context.structure(parameter.annotation, topology=topology),
            )
            for name, parameter in value.parameters.items()
        ),
        "return_annotation": context.structure(
            value.return_annotation, topology=topology
        ),
    }


def _dataclass_structure(
    value: Any, context: _FuelStructurePayload, /, *, topology: bool
) -> dict[str, Any]:
    # A closure-converted function's jaxpr captures its closed-over arrays as
    # dynamic inputs; the compiler payload must record them as such.
    closure_converted = isinstance(value, _ClosureConvert)
    return {
        "type": _type_name(value),
        "fields": tuple(
            (
                field.name,
                context.structure(
                    getattr(value, field.name),
                    field_name=field.name,
                    topology=topology,
                    dynamic_captures=closure_converted and field.name == "jaxpr",
                ),
            )
            for field in dataclasses.fields(value)
        ),
    }


def _point_route_structure(
    route: PreparedFiniteElementPointInterpolation,
    context: _FuelStructurePayload,
    /,
    *,
    topology: bool,
) -> dict[str, Any]:
    fields = tuple(
        field
        for field in route.discretization.field_spaces
        if field.name == route.field_name
    )
    if len(fields) != 1:
        raise ValueError("Fuel numerical route requires one exact declared field role.")
    field = fields[0]
    standard = DiscreteFieldSpace(
        field.name,
        field.support_id,
        field.layout,
        field.vector_space,
        representation=field.representation,
        conformity=field.conformity,
        projection_id=field.projection_id,
        reconstruction_id=field.reconstruction_id,
        trace_space_id=field.trace_space_id,
        form_type=field.form_type,
    )
    if field.field_space_id != standard.field_space_id:
        raise ValueError(
            "Fuel numerical route has a custom scientific field role; regroup the workset."
        )
    discretization = route.discretization
    return {
        "native_route": type(route).__qualname__,
        "field_role": (
            field.name,
            field.representation,
            field.conformity,
            field.projection_id,
            field.reconstruction_id,
            field.trace_space_id,
            context.structure(field.form_type, topology=topology),
        ),
        "coefficient_space": context.structure(field.vector_space, topology=topology),
        "elements": context.structure(discretization.elements, topology=topology),
        "precision": context.structure(
            discretization.precision_policy, topology=topology
        ),
        "topology": discretization.mesh.topology.topology_id,
        "block_roles": context.structure(discretization.mesh.blocks, topology=topology),
        "derivative_axis": route.derivative_axis,
        "tolerance": route.tolerance,
        "routes": context.structure(route.dof_routes, topology=topology),
        "weights": context.structure(route.weights, topology=topology),
        "points": context.structure(route.reference_positions, topology=topology),
        "dof_points": context.structure(route.dof_reference_positions, topology=topology),
    }


def _refreshed_solve_structure(
    value: PreparedNonlinearSolve, context: _FuelStructurePayload, /, *, topology: bool
) -> dict[str, Any]:
    """Describe a nonlinear template that fuel refreshes before every solve.

    Refresh replaces the point-bound Jacobian and root state. Their partially
    evaluated programs can contain folded pellet areas, which are numerical
    cache data rather than a different execution recipe, so only their roles and
    array layout enter.
    """
    retained_fields = tuple(
        (
            field.name,
            context.structure(
                getattr(value, field.name), field_name=field.name, topology=topology
            ),
        )
        for field in dataclasses.fields(value)
        if field.name not in _REFRESHED_SOLVE_FIELDS
    )
    plan = value.linear_plan
    linear_recipe = tuple(
        (
            field.name,
            context.structure(
                getattr(plan, field.name), field_name=field.name, topology=topology
            ),
        )
        for field in dataclasses.fields(plan)
        if field.name != "problem_signature"
    )
    return {
        "type": _type_name(value),
        "fields": retained_fields,
        "derivative": value.jacobian.derivative_id,
        "source": context.structure(value.jacobian.operator.source, topology=topology),
        "target": context.structure(value.jacobian.operator.target, topology=topology),
        "linear_recipe": linear_recipe,
        "cache_layout": array_layout(dynamic_arrays((value.jacobian, value.run))),
        "recycling": context.structure(value.run.recycling, topology=topology),
        "preconditioner_refresh_kind": value.linear_refresh_state.preconditioner_refresh_kind,
    }


def dynamic_arrays(value: Any, /) -> tuple[Array, ...]:
    """Return the dynamic array leaves of `value` in PyTree order."""
    leaves = tuple(jax.tree.leaves(eqx.filter(value, eqx.is_array)))
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError("Fuel numerical execution requires array-only dynamic leaves.")
    return leaves


def array_layout(arrays: tuple[Array, ...], /) -> ArrayLayout:
    return tuple((array.shape, str(array.dtype)) for array in arrays)


__all__ = ["ArrayLayout", "array_layout", "dynamic_arrays", "execution_structure"]
