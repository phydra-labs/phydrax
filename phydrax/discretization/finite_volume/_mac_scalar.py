#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._diffusion import (
    ConservativeAdvectionPlan,
    ConservativeBoundaryCondition,
    ConservativeDiffusionPlan,
    PreparedConservativeAdvection,
    PreparedConservativeDiffusion,
)
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._precision import FiniteVolumePrecisionPolicy


MACScalarAdvection: TypeAlias = Literal["centered", "upwind"]
MACScalarBoundaryKind: TypeAlias = Literal["periodic", "dirichlet", "neumann", "flux"]
MACScalarSGSNumberKind: TypeAlias = Literal["prandtl", "schmidt", "none"]


def _canonical_names(names: Sequence[str], /) -> tuple[str, ...]:
    values = tuple(str(name) for name in names)
    if not values or any(not name for name in values) or len(set(values)) != len(values):
        raise ValueError("MAC scalar field names must be non-empty and unique.")
    return tuple(sorted(values))


def _finite_array(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)),
        f"{owner} must be finite.",
    )


def _boundary_slice(value: Array, axis: int, index: int, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value[tuple(location)]


class MACScalarLayout(StrictModule, NonTrainableState):
    """Canonical named cell-scalar coordinates on one prepared MAC grid."""

    operators: PreparedMACOperators
    field_names: tuple[str, ...] = eqx.field(static=True)
    cell_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        field_names: Sequence[str],
        /,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        names = _canonical_names(field_names)
        cell_size = operators.pressure_space.size
        self.operators = operators
        self.field_names = names
        self.cell_size = cell_size
        self.state_size = cell_size * len(names)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-layout",
                "operators": operators.prepared_id,
                "fields": list(names),
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.state_size,)

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.operators.discretization.cell_shape

    @property
    def dtype(self):
        return self.operators.pressure_space.dtype

    def _field_index(self, name: str, /) -> int:
        field = str(name)
        if field not in self.field_names:
            raise KeyError(f"Unknown MAC scalar field {field!r}.")
        return self.field_names.index(field)

    def validate_fields(self, fields: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        supplied = dict(fields)
        if set(supplied) != set(self.field_names):
            raise ValueError(
                "MAC scalar field keys must exactly match "
                f"{self.field_names}; got {tuple(sorted(supplied))}."
            )
        values: dict[str, Array] = {}
        for name in self.field_names:
            value = jnp.asarray(supplied[name])
            if value.shape != self.cell_shape:
                raise ValueError(
                    f"MAC scalar field {name!r} must have shape {self.cell_shape}; "
                    f"got {value.shape}."
                )
            if value.dtype != self.dtype:
                raise TypeError(
                    f"MAC scalar field {name!r} must have dtype {self.dtype}; "
                    f"got {value.dtype}."
                )
            values[name] = _finite_array(value, f"MAC scalar field {name!r}")
        return values

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC scalar coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        if value.dtype != self.dtype:
            raise TypeError(
                f"MAC scalar coordinates must have dtype {self.dtype}; got {value.dtype}."
            )
        return _finite_array(value, "MAC scalar coordinates")

    def pack(self, fields: Mapping[str, ArrayLike], /) -> Array:
        values = self.validate_fields(fields)
        return jnp.concatenate(
            tuple(values[name].reshape((-1,)) for name in self.field_names)
        )

    def field(self, state: ArrayLike, name: str, /) -> Array:
        value = self.validate_state(state)
        index = self._field_index(name)
        start = index * self.cell_size
        return value[start : start + self.cell_size].reshape(self.cell_shape)

    def unpack(self, state: ArrayLike, /) -> dict[str, Array]:
        value = self.validate_state(state)
        return {
            name: value[index * self.cell_size : (index + 1) * self.cell_size].reshape(
                self.cell_shape
            )
            for index, name in enumerate(self.field_names)
        }


class MACScalarBoundaryCondition(StrictModule, NonTrainableState):
    """One scalar wall value, gradient, or outward conservative flux."""

    kind: MACScalarBoundaryKind = eqx.field(static=True)
    value: Array
    function: Any = eqx.field(static=True)
    function_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MACScalarBoundaryKind,
        value: ArrayLike | Any = 0.0,
        /,
        *,
        function_id: str | None = None,
    ):
        if kind not in ("periodic", "dirichlet", "neumann", "flux"):
            raise ValueError("Unknown MAC scalar boundary kind.")
        if callable(value):
            if kind != "flux":
                raise ValueError(
                    "Dynamic MAC scalar boundary data are supported only for flux."
                )
            identifier = "" if function_id is None else str(function_id)
            if not identifier:
                raise ValueError(
                    "Dynamic MAC scalar boundary flux requires a function_id."
                )
            value_ = jnp.asarray(0.0)
            function = value
            host_value: Any = "dynamic"
        else:
            if function_id is not None:
                raise ValueError(
                    "function_id must be omitted for static MAC scalar boundaries."
                )
            value_ = jnp.asarray(value)
            host = np.asarray(value_)
            if np.iscomplexobj(host) or np.any(~np.isfinite(host)):
                raise ValueError("MAC scalar boundary values must be finite and real.")
            function = None
            identifier = "none"
            host_value = host.tolist()
        if kind == "periodic" and (
            function is not None or value_.shape != () or float(value_) != 0.0
        ):
            raise ValueError("Periodic MAC scalar boundaries cannot carry data.")
        self.kind = kind
        self.value = value_
        self.function = function
        self.function_id = identifier
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-boundary-condition",
                "boundary_kind": kind,
                "value": host_value,
                "function": identifier,
                "flux_sign": "outward-loss" if kind == "flux" else "not-applicable",
            }
        )

    def evaluate(
        self,
        time: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> Array:
        target_shape = coordinates.shape[:-1]
        if self.function is None:
            output = jnp.broadcast_to(
                jnp.asarray(self.value, dtype=coordinates.dtype), target_shape
            )
        else:
            output = jnp.asarray(
                self.function(time, coordinates, args),
                dtype=coordinates.dtype,
            )
            if output.shape == ():
                output = jnp.broadcast_to(output, target_shape)
            elif output.shape != target_shape:
                raise ValueError(
                    "Dynamic MAC scalar boundary flux must be scalar or match "
                    f"boundary shape {target_shape}."
                )
        return _finite_array(output, "MAC scalar boundary evaluation")


class MACScalarBoundarySet(StrictModule, NonTrainableState):
    """Named periodic or static-wall scalar boundary data."""

    layout: MACScalarLayout
    conditions: tuple[
        tuple[tuple[MACScalarBoundaryCondition, MACScalarBoundaryCondition], ...], ...
    ]
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: MACScalarLayout,
        /,
        *,
        walls: Mapping[
            str,
            Mapping[
                str,
                tuple[
                    MACScalarBoundaryCondition | MACScalarBoundaryKind,
                    MACScalarBoundaryCondition | MACScalarBoundaryKind,
                ],
            ],
        ]
        | None = None,
    ):
        if not isinstance(layout, MACScalarLayout):
            raise TypeError("layout must be MACScalarLayout.")
        supplied = (
            {}
            if walls is None
            else {str(name): dict(value) for name, value in walls.items()}
        )
        unknown_fields = set(supplied).difference(layout.field_names)
        if unknown_fields:
            raise ValueError(
                f"MAC scalar walls reference unknown fields {sorted(unknown_fields)!r}."
            )
        grid = layout.operators.discretization.grid
        all_conditions = []
        for field_name in layout.field_names:
            field_walls = supplied.get(field_name, {})
            unknown_axes = set(field_walls).difference(grid.axis_names)
            if unknown_axes:
                raise ValueError(
                    f"MAC scalar walls reference unknown axes {sorted(unknown_axes)!r}."
                )
            axis_conditions = []
            for axis_name, axis in zip(
                grid.axis_names, grid.structured_axes, strict=True
            ):
                if axis.periodic:
                    if axis_name in field_walls:
                        raise ValueError(
                            "Periodic MAC axes do not accept scalar wall data."
                        )
                    pair = (
                        MACScalarBoundaryCondition("periodic"),
                        MACScalarBoundaryCondition("periodic"),
                    )
                else:
                    raw_pair = field_walls.get(axis_name, ("neumann", "neumann"))
                    if len(raw_pair) != 2:
                        raise ValueError(
                            "Each MAC scalar wall axis requires lower and upper data."
                        )
                    pair = tuple(
                        value
                        if isinstance(value, MACScalarBoundaryCondition)
                        else MACScalarBoundaryCondition(value)
                        for value in raw_pair
                    )
                    if any(value.kind == "periodic" for value in pair):
                        raise ValueError(
                            "Static MAC walls cannot use periodic scalar data."
                        )
                    for condition in pair:
                        expected = (
                            grid.shape[: grid.axis_names.index(axis_name)]
                            + grid.shape[grid.axis_names.index(axis_name) + 1 :]
                        )
                        if condition.value.shape not in ((), expected):
                            raise ValueError(
                                "MAC scalar wall data must be scalar or match its tangential "
                                f"shape {expected}."
                            )
                axis_conditions.append(pair)
            all_conditions.append(tuple(axis_conditions))
        conditions = tuple(all_conditions)
        self.layout = layout
        self.conditions = conditions
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-boundary-set",
                "layout": layout.layout_id,
                "conditions": [
                    [
                        [lower.boundary_id, upper.boundary_id]
                        for lower, upper in field_conditions
                    ]
                    for field_conditions in conditions
                ],
            }
        )

    def _field_index(self, name: str, /) -> int:
        return self.layout._field_index(name)

    def diffusion_conditions(
        self, name: str, /
    ) -> dict[str, tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition]]:
        field_conditions = self.conditions[self._field_index(name)]
        return {
            axis_name: (
                ConservativeBoundaryCondition(
                    "neumann" if lower.kind == "flux" else lower.kind
                ),
                ConservativeBoundaryCondition(
                    "neumann" if upper.kind == "flux" else upper.kind
                ),
            )
            for axis_name, (lower, upper) in zip(
                self.layout.operators.discretization.grid.axis_names,
                field_conditions,
                strict=True,
            )
        }

    def boundary_values(self, name: str, /) -> dict[str, tuple[Array, Array]]:
        field_conditions = self.conditions[self._field_index(name)]
        return {
            axis_name: (
                jnp.asarray(0.0) if lower.kind == "flux" else lower.value,
                jnp.asarray(0.0) if upper.kind == "flux" else upper.value,
            )
            for axis_name, (lower, upper) in zip(
                self.layout.operators.discretization.grid.axis_names,
                field_conditions,
                strict=True,
            )
        }

    def field_conditions(
        self, name: str, /
    ) -> tuple[tuple[MACScalarBoundaryCondition, MACScalarBoundaryCondition], ...]:
        return self.conditions[self._field_index(name)]


class MACScalarTransport(StrictModule, NonTrainableState):
    """Named explicit advection, diffusion, and source declaration."""

    name: str = eqx.field(static=True)
    diffusivity: Array
    advection: MACScalarAdvection = eqx.field(static=True)
    source: Any
    source_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        diffusivity: ArrayLike,
        /,
        *,
        advection: MACScalarAdvection = "upwind",
        source: Any = None,
        source_id: str | None = None,
    ):
        field_name = str(name)
        if not field_name:
            raise ValueError("MAC scalar transport requires a non-empty field name.")
        diffusivity_ = jnp.asarray(diffusivity, dtype=float)
        if (
            diffusivity_.ndim > 1
            or diffusivity_.size == 0
            or not bool(
                jnp.all(jnp.isfinite(diffusivity_)) & jnp.all(diffusivity_ >= 0.0)
            )
        ):
            raise ValueError(
                "MAC scalar diffusivity must be one finite nonnegative scalar "
                "or one nonnegative value per grid axis."
            )
        if advection not in ("centered", "upwind"):
            raise ValueError("MAC scalar advection must be 'centered' or 'upwind'.")
        if source is not None and not callable(source):
            raise TypeError("MAC scalar source must be callable or None.")
        if source is None:
            if source_id is not None:
                raise ValueError("source_id must be omitted when source is None.")
            source_identifier = "none"
        else:
            source_identifier = "" if source_id is None else str(source_id)
            if not source_identifier:
                raise ValueError("A MAC scalar source requires a non-empty source_id.")
        self.name = field_name
        self.diffusivity = diffusivity_
        self.advection = advection
        self.source = source
        self.source_id = source_identifier
        self.transport_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-transport",
                "name": field_name,
                "diffusivity": np.asarray(diffusivity_).tolist(),
                "advection": advection,
                "source": source_identifier,
            }
        )


class MACScalarReaction(StrictModule, NonTrainableState):
    """Named local reaction network with declared explicit rate bounds."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    rate: Any
    rate_bounds: tuple[float, ...] = eqx.field(static=True)
    reaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        rate: Any,
        /,
        *,
        rate_bounds: Mapping[str, float],
        reaction_id: str,
    ):
        names = _canonical_names(field_names)
        if not callable(rate):
            raise TypeError("MAC scalar reaction rate must be callable.")
        supplied = {str(name): float(value) for name, value in rate_bounds.items()}
        if set(supplied) != set(names) or any(
            not np.isfinite(value) or value < 0.0 for value in supplied.values()
        ):
            raise ValueError(
                "MAC scalar reaction rate bounds must be finite, nonnegative, and "
                "provided for every reaction field."
            )
        identifier = str(reaction_id)
        if not identifier:
            raise ValueError("reaction_id must be non-empty.")
        bounds = tuple(supplied[name] for name in names)
        self.field_names = names
        self.rate = rate
        self.rate_bounds = bounds
        self.reaction_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-reaction",
                "declared_id": identifier,
                "fields": list(names),
                "rate_bounds": list(bounds),
            }
        )


class MACScalarSGSField(StrictModule, NonTrainableState):
    """Explicit named scalar response to one runtime SGS eddy viscosity."""

    name: str = eqx.field(static=True)
    number_kind: MACScalarSGSNumberKind = eqx.field(static=True)
    turbulent_number: float | None = eqx.field(static=True)
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        turbulent_prandtl_number: float | None = None,
        turbulent_schmidt_number: float | None = None,
        no_sgs: bool = False,
    ):
        field_name = str(name)
        if not field_name:
            raise ValueError("MAC scalar SGS declarations require a field name.")
        supplied = (
            turbulent_prandtl_number is not None,
            turbulent_schmidt_number is not None,
            bool(no_sgs),
        )
        if sum(supplied) != 1:
            raise ValueError(
                "Declare exactly one turbulent Prandtl number, turbulent Schmidt "
                "number, or no_sgs=True for every MAC LES scalar."
            )
        if turbulent_prandtl_number is not None:
            kind: MACScalarSGSNumberKind = "prandtl"
            number = float(turbulent_prandtl_number)
        elif turbulent_schmidt_number is not None:
            kind = "schmidt"
            number = float(turbulent_schmidt_number)
        else:
            kind = "none"
            number = None
        if number is not None and (not np.isfinite(number) or number <= 0.0):
            raise ValueError("Turbulent Prandtl and Schmidt numbers must be positive.")
        self.name = field_name
        self.number_kind = kind
        self.turbulent_number = number
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-sgs-field",
                "field": field_name,
                "number_kind": kind,
                "turbulent_number": number,
            }
        )


class MACScalarSGSPlan(StrictModule, NonTrainableState):
    """Complete named turbulent-number contract for runtime scalar SGS fluxes."""

    fields: tuple[MACScalarSGSField, ...]
    field_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, fields: Sequence[MACScalarSGSField], /):
        values = tuple(fields)
        if not values or any(
            not isinstance(value, MACScalarSGSField) for value in values
        ):
            raise TypeError("fields must contain MACScalarSGSField declarations.")
        names = _canonical_names(tuple(value.name for value in values))
        by_name = {value.name: value for value in values}
        ordered = tuple(by_name[name] for name in names)
        self.fields = ordered
        self.field_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-scalar-sgs-plan",
                "fields": [value.declaration_id for value in ordered],
            }
        )

    def prepare(
        self,
        transport: PreparedMACScalarTransport,
        /,
        *,
        field_names: Sequence[str] | None = None,
    ) -> PreparedMACScalarSGS:
        if not isinstance(transport, PreparedMACScalarTransport):
            raise TypeError("transport must be PreparedMACScalarTransport.")
        names = (
            transport.layout.field_names
            if field_names is None
            else _canonical_names(field_names)
        )
        return PreparedMACScalarSGS(self, transport, names)


class PreparedMACScalarSGS(StrictModule, NonTrainableState):
    """Prepared named conversion from eddy viscosity to scalar diffusivity."""

    plan: MACScalarSGSPlan
    transport: PreparedMACScalarTransport
    fields: tuple[MACScalarSGSField, ...]
    field_names: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACScalarSGSPlan,
        transport: PreparedMACScalarTransport,
        field_names: Sequence[str],
        /,
    ):
        if not isinstance(plan, MACScalarSGSPlan):
            raise TypeError("plan must be MACScalarSGSPlan.")
        if not isinstance(transport, PreparedMACScalarTransport):
            raise TypeError("transport must be PreparedMACScalarTransport.")
        names = _canonical_names(field_names)
        if not set(names).issubset(transport.layout.field_names):
            raise ValueError("MAC scalar SGS fields must belong to the transport.")
        if plan.field_names != names:
            raise ValueError(
                "MAC scalar SGS declarations must exactly match the required named "
                f"fields {names}; got {plan.field_names}."
            )
        by_name = {value.name: value for value in plan.fields}
        fields = tuple(by_name[name] for name in names)
        for name in names:
            for lower, upper in transport.boundaries.field_conditions(name):
                for condition in (lower, upper):
                    no_flux = (
                        condition.kind == "neumann"
                        and condition.function is None
                        and bool(jnp.all(condition.value == 0.0))
                    )
                    if condition.kind not in ("periodic", "flux") and not no_flux:
                        raise ValueError(
                            "MAC scalar SGS supports only periodic, impermeable "
                            "no-flux, and prescribed total-flux boundaries."
                        )
        self.plan = plan
        self.transport = transport
        self.fields = fields
        self.field_names = names
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-scalar-sgs",
                "plan": plan.plan_id,
                "transport": transport.prepared_id,
                "fields": list(names),
                "boundary_semantics": "periodic-no-flux-or-prescribed-total-flux",
            }
        )

    def diffusivities(self, kinematic_eddy_viscosity: ArrayLike, /) -> dict[str, Array]:
        viscosity = jnp.asarray(
            kinematic_eddy_viscosity,
            dtype=self.transport.layout.dtype,
        )
        if viscosity.shape not in ((), self.transport.layout.cell_shape):
            raise ValueError(
                "Runtime MAC eddy viscosity must be scalar or match the cell shape."
            )
        viscosity = eqx.error_if(
            viscosity,
            jnp.any(~jnp.isfinite(viscosity)) | jnp.any(viscosity < 0.0),
            "Runtime MAC eddy viscosity must be finite and nonnegative.",
        )
        return {
            field.name: (
                jnp.zeros_like(viscosity)
                if field.turbulent_number is None
                else viscosity
                / jnp.asarray(field.turbulent_number, dtype=viscosity.dtype)
            )
            for field in self.fields
        }


class MACScalarProblem(StrictModule, NonTrainableState):
    """Named explicit scalar system transported on a physical MAC velocity."""

    transports: tuple[MACScalarTransport, ...]
    reaction: MACScalarReaction | None
    field_names: tuple[str, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        transports: Sequence[MACScalarTransport],
        /,
        *,
        reaction: MACScalarReaction | None = None,
        problem_id: str | None = None,
    ):
        values = tuple(transports)
        if not values or any(
            not isinstance(value, MACScalarTransport) for value in values
        ):
            raise TypeError("transports must contain MACScalarTransport declarations.")
        names = _canonical_names(tuple(value.name for value in values))
        by_name = {value.name: value for value in values}
        ordered = tuple(by_name[name] for name in names)
        if reaction is not None and not isinstance(reaction, MACScalarReaction):
            raise TypeError("reaction must be MACScalarReaction or None.")
        if reaction is not None and not set(reaction.field_names).issubset(names):
            raise ValueError("MAC scalar reaction fields must belong to the problem.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mac-scalar-problem",
                    "transports": [value.transport_id for value in ordered],
                    "reaction": None if reaction is None else reaction.reaction_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.transports = ordered
        self.reaction = reaction
        self.field_names = names
        self.problem_id = identifier

    def prepare(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        boundaries: MACScalarBoundarySet | None = None,
    ) -> PreparedMACScalarTransport:
        layout = MACScalarLayout(operators, self.field_names)
        boundaries_ = MACScalarBoundarySet(layout) if boundaries is None else boundaries
        return PreparedMACScalarTransport(self, layout, boundaries_)


class MACScalarFluxResult(StrictModule):
    """One named scalar stage with separated molecular, SGS, and total fluxes."""

    face_values: FaceVelocity
    advective_fluxes: FaceVelocity
    molecular_diffusive_fluxes: FaceVelocity
    sgs_diffusive_fluxes: FaceVelocity
    boundary_diffusive_fluxes: FaceVelocity
    diffusive_fluxes: FaceVelocity
    molecular_diffusivity: Array
    sgs_diffusivity: Array
    total_diffusivity: Array
    advective_divergence: Array
    molecular_diffusive_divergence: Array
    sgs_diffusive_divergence: Array
    boundary_diffusive_divergence: Array
    diffusive_divergence: Array
    source: Array
    reaction: Array
    rate: Array
    molecular_finite: Array
    sgs_finite: Array
    boundary_finite: Array
    finite: Array
    success: Array
    field_name: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MACScalarFieldDiagnostics(StrictModule):
    """Content and variance ledger for one named scalar stage."""

    content: Array
    mean: Array
    variance: Array
    content_rate: Array
    advective_content_rate: Array
    molecular_diffusive_content_rate: Array
    sgs_diffusive_content_rate: Array
    boundary_diffusive_content_rate: Array
    diffusive_content_rate: Array
    source_content_rate: Array
    reaction_content_rate: Array
    content_balance_defect: Array
    variance_rate: Array
    advective_variance_rate: Array
    molecular_diffusive_variance_rate: Array
    sgs_diffusive_variance_rate: Array
    boundary_diffusive_variance_rate: Array
    diffusive_variance_rate: Array
    source_variance_rate: Array
    reaction_variance_rate: Array
    variance_balance_defect: Array
    finite: Array
    success: Array
    field_name: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACScalarDiagnostics(StrictModule):
    """Named content and variance ledgers for a complete scalar stage."""

    fields: dict[str, MACScalarFieldDiagnostics]
    finite: Array
    success: Array
    problem_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACScalarStepRestriction(StrictModule):
    """Named explicit scalar advective, diffusive, and reaction restrictions."""

    advective: dict[str, Array]
    molecular_diffusive: dict[str, Array]
    sgs_diffusive: dict[str, Array]
    diffusive: dict[str, Array]
    reaction: dict[str, Array]
    selected_by_field: dict[str, Array]
    selected: Array
    finite: Array
    success: Array
    transport_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class PreparedMACScalarTransport(StrictModule, NonTrainableState):
    """Prepared named conservative scalar transport on one MAC grid."""

    problem: MACScalarProblem
    layout: MACScalarLayout
    boundaries: MACScalarBoundarySet
    advection: tuple[PreparedConservativeAdvection, ...]
    diffusion: tuple[PreparedConservativeDiffusion, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MACScalarProblem,
        layout: MACScalarLayout,
        boundaries: MACScalarBoundarySet,
        /,
    ):
        if not isinstance(problem, MACScalarProblem):
            raise TypeError("problem must be MACScalarProblem.")
        if not isinstance(layout, MACScalarLayout):
            raise TypeError("layout must be MACScalarLayout.")
        if not isinstance(boundaries, MACScalarBoundarySet):
            raise TypeError("boundaries must be MACScalarBoundarySet.")
        if problem.field_names != layout.field_names:
            raise ValueError("MAC scalar problem and layout fields must agree by name.")
        if boundaries.layout.layout_id != layout.layout_id:
            raise ValueError("MAC scalar boundaries must use the same layout.")
        grid = layout.operators.discretization.grid
        precision = FiniteVolumePrecisionPolicy(np.dtype(layout.dtype).name)
        zero_velocity = tuple(
            jnp.zeros(face_layout.shape, dtype=layout.dtype)
            for face_layout in layout.operators.discretization.face_layouts
        )
        advection = []
        diffusion = []
        for declaration in problem.transports:
            conditions = boundaries.diffusion_conditions(declaration.name)
            reconstruction = (
                "arithmetic" if declaration.advection == "centered" else "upwind"
            )
            advection.append(
                ConservativeAdvectionPlan(
                    grid,
                    form="conservative",
                    reconstruction=reconstruction,
                    boundaries=conditions,
                    precision=precision,
                ).prepare(zero_velocity)
            )
            coefficient = declaration.diffusivity
            if coefficient.ndim == 1:
                dimension = len(grid.shape)
                if coefficient.size != dimension:
                    raise ValueError(
                        "Directional MAC scalar diffusivity must contain one "
                        f"value per grid axis; expected {dimension}."
                    )
                coefficient = jnp.broadcast_to(
                    coefficient,
                    grid.shape + (dimension,),
                )
            diffusion.append(
                ConservativeDiffusionPlan(
                    grid,
                    boundaries=conditions,
                    interpolation="harmonic",
                    precision=precision,
                ).prepare(coefficient)
            )
        advection_ = tuple(advection)
        diffusion_ = tuple(diffusion)
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-mac-scalar-transport",
                "problem": problem.problem_id,
                "layout": layout.layout_id,
                "boundaries": boundaries.boundary_id,
                "advection": [value.prepared_id for value in advection_],
                "diffusion": [value.operator_id for value in diffusion_],
            }
        )
        self.problem = problem
        self.layout = layout
        self.boundaries = boundaries
        self.advection = advection_
        self.diffusion = diffusion_
        self.prepared_id = identifier

    def _validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = self.layout.operators.validate_velocity(velocity)
        values = tuple(
            _finite_array(value, "MAC scalar transport velocity") for value in values
        )
        for axis, grid_axis in enumerate(
            self.layout.operators.discretization.grid.structured_axes
        ):
            if grid_axis.periodic:
                continue
            value = values[axis]
            defect = jnp.maximum(
                jnp.max(jnp.abs(_boundary_slice(value, axis, 0))),
                jnp.max(jnp.abs(_boundary_slice(value, axis, -1))),
            )
            values = (
                values[:axis]
                + (
                    eqx.error_if(
                        value,
                        defect > 0.0,
                        "MAC scalar transport requires impermeable static walls.",
                    ),
                )
                + values[axis + 1 :]
            )
        return values

    def _reaction_rates(
        self,
        time: Array,
        fields: dict[str, Array],
        args: Any,
        /,
    ) -> dict[str, Array]:
        reaction = self.problem.reaction
        output = {name: jnp.zeros_like(fields[name]) for name in self.layout.field_names}
        if reaction is None:
            return output
        raw = dict(reaction.rate(time, fields, args))
        if set(raw) != set(reaction.field_names):
            raise ValueError(
                "MAC scalar reaction output keys must exactly match its declared fields."
            )
        for name in reaction.field_names:
            value = jnp.asarray(raw[name])
            if value.shape != self.layout.cell_shape:
                raise ValueError(
                    f"MAC scalar reaction field {name!r} must have shape "
                    f"{self.layout.cell_shape}."
                )
            if value.dtype != self.layout.dtype:
                raise TypeError(
                    f"MAC scalar reaction field {name!r} must have dtype "
                    f"{self.layout.dtype}."
                )
            output[name] = _finite_array(value, f"MAC scalar reaction field {name!r}")
        return output

    def _prescribed_diffusive_fluxes(
        self,
        time: Array,
        name: str,
        fluxes: tuple[Array, ...],
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        output = list(fluxes)
        discretization = self.layout.operators.discretization
        for axis, (lower, upper) in enumerate(self.boundaries.field_conditions(name)):
            for index, condition, orientation in (
                (0, lower, 1.0),
                (-1, upper, -1.0),
            ):
                if condition.kind != "flux":
                    continue
                coordinates = jnp.take(
                    discretization.face_centers[axis],
                    index,
                    axis=axis,
                )
                outward_loss = condition.evaluate(
                    time,
                    coordinates,
                    args,
                )
                location = [slice(None)] * output[axis].ndim
                location[axis] = index
                output[axis] = (
                    output[axis].at[tuple(location)].set(orientation * outward_loss)
                )
        return tuple(output)

    def _runtime_sgs_diffusivities(
        self,
        values: Mapping[str, ArrayLike] | None,
        /,
    ) -> dict[str, Array]:
        if values is None:
            return {
                name: jnp.zeros((), dtype=self.layout.dtype)
                for name in self.layout.field_names
            }
        supplied = dict(values)
        if set(supplied) != set(self.layout.field_names):
            raise ValueError(
                "Runtime MAC scalar SGS diffusivities must exactly match "
                f"{self.layout.field_names}."
            )
        output: dict[str, Array] = {}
        for name in self.layout.field_names:
            value = jnp.asarray(supplied[name], dtype=self.layout.dtype)
            if value.shape not in ((), self.layout.cell_shape):
                raise ValueError(
                    f"Runtime scalar SGS diffusivity {name!r} must be scalar or "
                    f"have cell shape {self.layout.cell_shape}."
                )
            output[name] = eqx.error_if(
                value,
                jnp.any(~jnp.isfinite(value)) | jnp.any(value < 0.0),
                f"Runtime scalar SGS diffusivity {name!r} must be finite and nonnegative.",
            )
            for lower, upper in self.boundaries.field_conditions(name):
                for condition in (lower, upper):
                    if condition.kind not in ("periodic", "neumann", "flux"):
                        raise ValueError(
                            "Runtime MAC scalar SGS diffusion supports only periodic, "
                            "impermeable no-flux, and prescribed total-flux boundaries."
                        )
                    if condition.kind == "neumann":
                        output[name] = eqx.error_if(
                            output[name],
                            jnp.any(condition.value != 0.0),
                            "Runtime MAC scalar SGS diffusion requires zero Neumann flux.",
                        )
        return output

    def evaluate(
        self,
        time: ArrayLike,
        fields: Mapping[str, ArrayLike],
        velocity: FaceVelocity,
        args: Any = None,
        /,
        *,
        sgs_diffusivities: Mapping[str, ArrayLike] | None = None,
    ) -> dict[str, MACScalarFluxResult]:
        time_ = _finite_array(jnp.asarray(time), "MAC scalar stage time")
        if time_.shape != ():
            raise ValueError("MAC scalar stage time must be scalar.")
        fields_ = self.layout.validate_fields(fields)
        velocity_ = self._validate_velocity(velocity)
        sgs_active = sgs_diffusivities is not None
        sgs_coefficients = self._runtime_sgs_diffusivities(sgs_diffusivities)
        reactions = self._reaction_rates(time_, fields_, args)
        results: dict[str, MACScalarFluxResult] = {}
        grid_id = self.layout.operators.discretization.grid.prepared_id
        for declaration, advection, diffusion in zip(
            self.problem.transports,
            self.advection,
            self.diffusion,
            strict=True,
        ):
            name = declaration.name
            value = fields_[name]
            boundary_values = self.boundaries.boundary_values(name)
            face_values = advection.face_values(
                value,
                velocity=velocity_,
                boundary_values=boundary_values,
            )
            advective_fluxes = tuple(
                face_velocity * face_value
                for face_velocity, face_value in zip(velocity_, face_values, strict=True)
            )
            advective_divergence = advection.divergence(advective_fluxes)
            molecular_diffusivity = diffusion._coefficient(diffusion.coefficient)
            molecular_fluxes = diffusion.fluxes(
                value,
                molecular_diffusivity,
                boundary_values,
            )
            if sgs_active:
                sgs_diffusivity = diffusion._coefficient(sgs_coefficients[name])
                sgs_fluxes = diffusion.fluxes(
                    value,
                    sgs_diffusivity,
                    boundary_values,
                )
            else:
                sgs_diffusivity = jnp.zeros_like(molecular_diffusivity)
                sgs_fluxes = tuple(jnp.zeros_like(flux) for flux in molecular_fluxes)
            total_diffusivity = molecular_diffusivity + sgs_diffusivity
            internal_fluxes = tuple(
                molecular + sgs
                for molecular, sgs in zip(molecular_fluxes, sgs_fluxes, strict=True)
            )
            diffusive_fluxes = self._prescribed_diffusive_fluxes(
                time_,
                name,
                internal_fluxes,
                args,
            )
            boundary_fluxes = tuple(
                total - internal
                for total, internal in zip(diffusive_fluxes, internal_fluxes, strict=True)
            )
            molecular_divergence = diffusion.divergence(molecular_fluxes)
            sgs_divergence = diffusion.divergence(sgs_fluxes)
            boundary_divergence = diffusion.divergence(boundary_fluxes)
            diffusive_divergence = diffusion.divergence(diffusive_fluxes)
            if declaration.source is None:
                source = jnp.zeros_like(value)
            else:
                source = jnp.asarray(declaration.source(time_, fields_, velocity_, args))
                if source.shape != self.layout.cell_shape:
                    raise ValueError(
                        f"MAC scalar source {name!r} must have shape "
                        f"{self.layout.cell_shape}."
                    )
                if source.dtype != self.layout.dtype:
                    raise TypeError(
                        f"MAC scalar source {name!r} must have dtype {self.layout.dtype}."
                    )
                source = _finite_array(source, f"MAC scalar source {name!r}")
            reaction = reactions[name]
            rate = _finite_array(
                -advective_divergence + diffusive_divergence + source + reaction,
                f"MAC scalar rate {name!r}",
            )
            molecular_finite = (
                jnp.all(jnp.isfinite(molecular_diffusivity))
                & jnp.all(
                    jnp.stack(
                        tuple(jnp.all(jnp.isfinite(value)) for value in molecular_fluxes)
                    )
                )
                & jnp.all(jnp.isfinite(molecular_divergence))
            )
            sgs_finite = (
                jnp.all(jnp.isfinite(sgs_diffusivity))
                & jnp.all(
                    jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in sgs_fluxes))
                )
                & jnp.all(jnp.isfinite(sgs_divergence))
            )
            boundary_finite = jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in boundary_fluxes)
                )
            ) & jnp.all(jnp.isfinite(boundary_divergence))
            finite = (
                molecular_finite
                & sgs_finite
                & boundary_finite
                & jnp.all(jnp.isfinite(total_diffusivity))
                & jnp.all(jnp.isfinite(rate))
                & jnp.all(jnp.isfinite(advective_divergence))
                & jnp.all(jnp.isfinite(diffusive_divergence))
                & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in face_values)))
            )
            results[name] = MACScalarFluxResult(
                face_values=face_values,
                advective_fluxes=advective_fluxes,
                molecular_diffusive_fluxes=molecular_fluxes,
                sgs_diffusive_fluxes=sgs_fluxes,
                boundary_diffusive_fluxes=boundary_fluxes,
                diffusive_fluxes=diffusive_fluxes,
                molecular_diffusivity=molecular_diffusivity,
                sgs_diffusivity=sgs_diffusivity,
                total_diffusivity=total_diffusivity,
                advective_divergence=advective_divergence,
                molecular_diffusive_divergence=molecular_divergence,
                sgs_diffusive_divergence=sgs_divergence,
                boundary_diffusive_divergence=boundary_divergence,
                diffusive_divergence=diffusive_divergence,
                source=source,
                reaction=reaction,
                rate=rate,
                molecular_finite=molecular_finite,
                sgs_finite=sgs_finite,
                boundary_finite=boundary_finite,
                finite=finite,
                success=finite,
                field_name=name,
                grid_id=grid_id,
                layout_id=self.layout.layout_id,
                transport_id=self.prepared_id,
                result_id=canonical_fingerprint(
                    {
                        "kind": "mac-scalar-flux-result",
                        "transport": self.prepared_id,
                        "field": name,
                    }
                ),
            )
        return results

    def rates(
        self,
        time: ArrayLike,
        fields: Mapping[str, ArrayLike],
        velocity: FaceVelocity,
        args: Any = None,
        /,
        *,
        sgs_diffusivities: Mapping[str, ArrayLike] | None = None,
    ) -> dict[str, Array]:
        results = self.evaluate(
            time,
            fields,
            velocity,
            args,
            sgs_diffusivities=sgs_diffusivities,
        )
        return {name: results[name].rate for name in self.layout.field_names}

    def diagnostics_from_fluxes(
        self,
        fields: Mapping[str, ArrayLike],
        fluxes: Mapping[str, MACScalarFluxResult],
        /,
    ) -> MACScalarDiagnostics:
        fields_ = self.layout.validate_fields(fields)
        results = dict(fluxes)
        if set(results) != set(self.layout.field_names):
            raise ValueError("MAC scalar diagnostics require one result per named field.")
        volumes = self.layout.operators.discretization.cell_volumes
        total_volume = jnp.sum(volumes)
        diagnostics: dict[str, MACScalarFieldDiagnostics] = {}
        for name in self.layout.field_names:
            result = results[name]
            if (
                not isinstance(result, MACScalarFluxResult)
                or result.field_name != name
                or result.transport_id != self.prepared_id
                or result.layout_id != self.layout.layout_id
            ):
                raise ValueError(
                    "MAC scalar flux result provenance does not match transport."
                )
            value = fields_[name]
            content = jnp.sum(volumes * value)
            mean = content / total_volume
            centered = value - mean
            variance = jnp.sum(volumes * centered**2) / total_volume
            pieces = (
                -result.advective_divergence,
                result.molecular_diffusive_divergence,
                result.sgs_diffusive_divergence,
                result.boundary_diffusive_divergence,
                result.source,
                result.reaction,
            )
            content_rates = tuple(jnp.sum(volumes * piece) for piece in pieces)
            variance_rates = tuple(
                2.0 * jnp.sum(volumes * centered * piece) / total_volume
                for piece in pieces
            )
            diffusive_content_rate = sum(content_rates[1:4])
            diffusive_variance_rate = sum(variance_rates[1:4])
            content_rate = jnp.sum(volumes * result.rate)
            variance_rate = 2.0 * jnp.sum(volumes * centered * result.rate) / total_volume
            content_defect = content_rate - sum(content_rates)
            variance_defect = variance_rate - sum(variance_rates)
            finite = result.finite & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            content,
                            mean,
                            variance,
                            content_rate,
                            diffusive_content_rate,
                            variance_rate,
                            diffusive_variance_rate,
                            content_defect,
                            variance_defect,
                        )
                        + content_rates
                        + variance_rates
                    )
                )
            )
            diagnostics[name] = MACScalarFieldDiagnostics(
                content=content,
                mean=mean,
                variance=variance,
                content_rate=content_rate,
                advective_content_rate=content_rates[0],
                molecular_diffusive_content_rate=content_rates[1],
                sgs_diffusive_content_rate=content_rates[2],
                boundary_diffusive_content_rate=content_rates[3],
                diffusive_content_rate=diffusive_content_rate,
                source_content_rate=content_rates[4],
                reaction_content_rate=content_rates[5],
                content_balance_defect=content_defect,
                variance_rate=variance_rate,
                advective_variance_rate=variance_rates[0],
                molecular_diffusive_variance_rate=variance_rates[1],
                sgs_diffusive_variance_rate=variance_rates[2],
                boundary_diffusive_variance_rate=variance_rates[3],
                diffusive_variance_rate=diffusive_variance_rate,
                source_variance_rate=variance_rates[4],
                reaction_variance_rate=variance_rates[5],
                variance_balance_defect=variance_defect,
                finite=finite,
                success=finite,
                field_name=name,
                transport_id=self.prepared_id,
                grid_id=result.grid_id,
            )
        finite = jnp.all(
            jnp.stack(tuple(diagnostics[name].finite for name in self.layout.field_names))
        )
        return MACScalarDiagnostics(
            fields=diagnostics,
            finite=finite,
            success=finite,
            problem_id=self.problem.problem_id,
            transport_id=self.prepared_id,
            layout_id=self.layout.layout_id,
            grid_id=self.layout.operators.discretization.grid.prepared_id,
        )

    def diagnostics(
        self,
        time: ArrayLike,
        fields: Mapping[str, ArrayLike],
        velocity: FaceVelocity,
        args: Any = None,
        /,
        *,
        sgs_diffusivities: Mapping[str, ArrayLike] | None = None,
    ) -> MACScalarDiagnostics:
        fields_ = self.layout.validate_fields(fields)
        results = self.evaluate(
            time,
            fields_,
            velocity,
            args,
            sgs_diffusivities=sgs_diffusivities,
        )
        return self.diagnostics_from_fluxes(fields_, results)

    def step_restriction(
        self,
        velocity: FaceVelocity,
        /,
        *,
        sgs_diffusivities: Mapping[str, ArrayLike] | None = None,
    ) -> MACScalarStepRestriction:
        sgs_active = sgs_diffusivities is not None
        sgs_coefficients = self._runtime_sgs_diffusivities(sgs_diffusivities)
        velocity_ = self._validate_velocity(velocity)
        discretization = self.layout.operators.discretization
        grid = discretization.grid
        inverse_advective = jnp.zeros(
            self.layout.cell_shape,
            dtype=self.layout.dtype,
        )
        for axis_index, axis in enumerate(grid.structured_axes):
            oriented_flux = (
                velocity_[axis_index] * discretization.face_measures[axis_index]
            )
            if axis.periodic:
                lower = oriented_flux
                upper = jnp.roll(oriented_flux, -1, axis=axis_index)
            else:
                lower_slice = [slice(None)] * oriented_flux.ndim
                upper_slice = [slice(None)] * oriented_flux.ndim
                lower_slice[axis_index] = slice(0, oriented_flux.shape[axis_index] - 1)
                upper_slice[axis_index] = slice(1, oriented_flux.shape[axis_index])
                lower = oriented_flux[tuple(lower_slice)]
                upper = oriented_flux[tuple(upper_slice)]
            outgoing = jnp.maximum(-lower, 0.0) + jnp.maximum(upper, 0.0)
            inverse_advective = inverse_advective + outgoing / discretization.cell_volumes
        advective_rate = jnp.max(inverse_advective)
        safe_advective = jnp.where(advective_rate > 0.0, advective_rate, 1.0)
        advective_value = jnp.where(
            advective_rate > 0.0,
            1.0 / safe_advective,
            jnp.inf,
        )
        reaction_bounds = {name: 0.0 for name in self.layout.field_names}
        if self.problem.reaction is not None:
            reaction_bounds.update(
                dict(
                    zip(
                        self.problem.reaction.field_names,
                        self.problem.reaction.rate_bounds,
                        strict=True,
                    )
                )
            )
        advective: dict[str, Array] = {}
        molecular_diffusive: dict[str, Array] = {}
        sgs_diffusive: dict[str, Array] = {}
        diffusive: dict[str, Array] = {}
        reaction: dict[str, Array] = {}
        selected_by_field: dict[str, Array] = {}
        for declaration, diffusion_operator in zip(
            self.problem.transports,
            self.diffusion,
            strict=True,
        ):
            name = declaration.name
            molecular_coefficient = diffusion_operator._coefficient(
                diffusion_operator.coefficient
            )
            molecular_diagonal = diffusion_operator.diagonal_with_coefficient(
                molecular_coefficient
            )
            if sgs_active:
                sgs_coefficient = diffusion_operator._coefficient(sgs_coefficients[name])
                sgs_diagonal = diffusion_operator.diagonal_with_coefficient(
                    sgs_coefficient
                )
            else:
                sgs_diagonal = jnp.zeros_like(molecular_diagonal)
            molecular_rate = jnp.max(jnp.maximum(0.0, -molecular_diagonal))
            sgs_rate = jnp.max(jnp.maximum(0.0, -sgs_diagonal))
            diffusive_rate = jnp.max(
                jnp.maximum(0.0, -(molecular_diagonal + sgs_diagonal))
            )
            safe_molecular = jnp.where(molecular_rate > 0.0, molecular_rate, 1.0)
            safe_sgs = jnp.where(sgs_rate > 0.0, sgs_rate, 1.0)
            safe_diffusive = jnp.where(diffusive_rate > 0.0, diffusive_rate, 1.0)
            molecular_value = jnp.where(
                molecular_rate > 0.0,
                1.0 / safe_molecular,
                jnp.inf,
            )
            sgs_value = jnp.where(
                sgs_rate > 0.0,
                1.0 / safe_sgs,
                jnp.inf,
            )
            diffusive_value = jnp.where(
                diffusive_rate > 0.0,
                1.0 / safe_diffusive,
                jnp.inf,
            )
            reaction_rate = jnp.asarray(reaction_bounds[name], dtype=self.layout.dtype)
            safe_reaction = jnp.where(reaction_rate > 0.0, reaction_rate, 1.0)
            reaction_value = jnp.where(
                reaction_rate > 0.0,
                1.0 / safe_reaction,
                jnp.inf,
            )
            advective[name] = advective_value
            molecular_diffusive[name] = molecular_value
            sgs_diffusive[name] = sgs_value
            diffusive[name] = diffusive_value
            reaction[name] = reaction_value
            selected_by_field[name] = jnp.minimum(
                advective_value,
                jnp.minimum(diffusive_value, reaction_value),
            )
        selected = jnp.min(
            jnp.stack(tuple(selected_by_field[name] for name in self.layout.field_names))
        )
        finite = ~jnp.isnan(selected)
        return MACScalarStepRestriction(
            advective=advective,
            molecular_diffusive=molecular_diffusive,
            sgs_diffusive=sgs_diffusive,
            diffusive=diffusive,
            reaction=reaction,
            selected_by_field=selected_by_field,
            selected=selected,
            finite=finite,
            success=finite,
            transport_id=self.prepared_id,
            layout_id=self.layout.layout_id,
            grid_id=grid.prepared_id,
        )


__all__ = [
    "MACScalarAdvection",
    "MACScalarBoundaryCondition",
    "MACScalarBoundaryKind",
    "MACScalarSGSField",
    "MACScalarSGSNumberKind",
    "MACScalarSGSPlan",
    "MACScalarBoundarySet",
    "MACScalarDiagnostics",
    "MACScalarFieldDiagnostics",
    "MACScalarFluxResult",
    "MACScalarLayout",
    "MACScalarProblem",
    "MACScalarReaction",
    "MACScalarStepRestriction",
    "MACScalarTransport",
    "PreparedMACScalarSGS",
    "PreparedMACScalarTransport",
]
