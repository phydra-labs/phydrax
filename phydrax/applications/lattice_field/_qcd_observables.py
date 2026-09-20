#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite lattice-QCD observables with explicit normalization and identity."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...graph._gauge_transport import GaugeCovariantShiftPlan
from ...linalg import (
    ArraySpace,
    BiCGStab,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    RHSLayout,
    solve,
    SolveResourcePolicy,
    TolerancePolicy,
)
from ...metrix import SpecialUnitaryGroup
from ...operators.path_integral._improved_gauge import (
    GaugeGradientFlowEvidence,
    GaugeGradientFlowPlan,
)
from ...operators.path_integral._lattice_fermion import (
    AbstractLatticeDiracOperator,
    WilsonDiracOperator,
)
from ...operators.path_integral._wilson_gauge import WilsonGaugeAction
from ._qcd_ensembles import MeasurementWorkItem


SpatialNormalization: TypeAlias = Literal["sum", "mean"]
TopologyStatus: TypeAlias = Literal["measured", "not-requested", "unsupported-dimension"]
NoiseKind: TypeAlias = Literal["z2", "z4"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _positive_shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{name} must contain positive dimensions.")
    return shape


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _shift_site_field(value: Array, displacement: Sequence[int], /) -> Array:
    shifted = value
    for axis, amount in enumerate(displacement):
        if amount:
            shifted = jnp.roll(shifted, -int(amount), axis=axis)
    return shifted


def _oriented_structured_link(
    links: Array,
    axis: int,
    direction: int,
    offset: tuple[int, ...],
    /,
) -> tuple[Array, tuple[int, ...]]:
    next_offset = list(offset)
    if direction > 0:
        factor = _shift_site_field(links[..., axis, :, :], offset)
        next_offset[axis] += 1
    else:
        next_offset[axis] -= 1
        factor = _adjoint(_shift_site_field(links[..., axis, :, :], tuple(next_offset)))
    return factor, tuple(next_offset)


def _oriented_plaquette(
    links: Array,
    first_axis: int,
    first_direction: int,
    second_axis: int,
    second_direction: int,
    /,
) -> Array:
    dimension = links.ndim - 3
    offset = (0,) * dimension
    identity = jnp.eye(links.shape[-1], dtype=links.dtype)
    value = jnp.broadcast_to(identity, links.shape[:dimension] + identity.shape)
    for axis, direction in (
        (first_axis, first_direction),
        (second_axis, second_direction),
        (first_axis, -first_direction),
        (second_axis, -second_direction),
    ):
        factor, offset = _oriented_structured_link(links, axis, direction, offset)
        value = contract("...ij,...jk->...ik", value, factor)
    return value


class HypercubicGaugeObservablePlan(StrictModule, NonTrainableState):
    """Observable geometry for structured ``site..., direction, color, color`` links."""

    group: SpecialUnitaryGroup
    lattice_spacing: Array
    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    color_components: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_shape: Sequence[int],
        /,
        *,
        lattice_spacing: float = 1.0,
        color_components: int = 3,
        topology_id: str | None = None,
        field_space_id: str | None = None,
        maximum_sites: int = 1 << 22,
    ):
        shape = _positive_shape(lattice_shape, "lattice_shape")
        spacing = float(lattice_spacing)
        colors = int(color_components)
        maximum = int(maximum_sites)
        if len(shape) < 2:
            raise ValueError("Gauge observables require at least two lattice axes.")
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("lattice_spacing must be finite and positive.")
        if colors < 2:
            raise ValueError("color_components must be at least two.")
        if maximum <= 0 or prod(shape) > maximum:
            raise ValueError("Gauge-observable lattice exceeds maximum_sites.")
        topology = (
            canonical_fingerprint(
                {"kind": "periodic-hypercubic-topology", "axis_sizes": shape}
            )
            if topology_id is None
            else _identifier(topology_id, "topology_id")
        )
        field_space = (
            canonical_fingerprint(
                {
                    "kind": "structured-fundamental-gauge-links",
                    "topology": topology,
                    "dimension": len(shape),
                    "colors": colors,
                }
            )
            if field_space_id is None
            else _identifier(field_space_id, "field_space_id")
        )
        group = SpecialUnitaryGroup(colors)
        plan_id = canonical_fingerprint(
            {
                "kind": "hypercubic-gauge-observables",
                "topology": topology,
                "field_space": field_space,
                "lattice_spacing": spacing,
                "group": group.group_id,
                "plaquette_normalization": "mean-real-trace-over-color-and-oriented-plane",
                "topology_normalization": "clover-epsilon-ff-over-32-pi-squared",
            }
        )
        self.group = group
        self.lattice_spacing = jnp.asarray(spacing)
        self.lattice_shape = shape
        self.dimension = len(shape)
        self.color_components = colors
        self.topology_id = topology
        self.field_space_id = field_space
        self.plan_id = plan_id

    @property
    def configuration_shape(self) -> tuple[int, ...]:
        return self.lattice_shape + (
            self.dimension,
            self.color_components,
            self.color_components,
        )

    def validate(self, links: ArrayLike, /) -> Array:
        values = jnp.asarray(links)
        if values.shape != self.configuration_shape:
            raise ValueError(
                f"links must have structured shape {self.configuration_shape}; got {values.shape}."
            )
        return values


class GaugeObservableResult(StrictModule):
    """Normalized gauge observables and explicit topology availability."""

    mean_plaquette: Array
    wilson_action_density: Array
    topological_charge: Array | None
    topological_charge_density: Array | None
    finite: Array
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    topology_status: TopologyStatus = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)


def hypercubic_plaquette_holonomies(
    plan: HypercubicGaugeObservablePlan,
    links: ArrayLike,
    /,
) -> Array:
    """Return one positively oriented plaquette for every site and axis pair."""
    if not isinstance(plan, HypercubicGaugeObservablePlan):
        raise TypeError("plan must be HypercubicGaugeObservablePlan.")
    values = plan.validate(links)
    return jnp.stack(
        tuple(
            _oriented_plaquette(values, mu, 1, nu, 1)
            for mu in range(plan.dimension)
            for nu in range(mu + 1, plan.dimension)
        ),
        axis=plan.dimension,
    )


def clover_field_strength(
    plan: HypercubicGaugeObservablePlan,
    links: ArrayLike,
    /,
) -> Array:
    """Return Hermitian traceless clover field strengths for ordered axis pairs."""
    if not isinstance(plan, HypercubicGaugeObservablePlan):
        raise TypeError("plan must be HypercubicGaugeObservablePlan.")
    values = plan.validate(links)
    identity = jnp.eye(plan.color_components, dtype=values.dtype)
    fields: list[Array] = []
    for mu in range(plan.dimension):
        for nu in range(mu + 1, plan.dimension):
            clover = (
                _oriented_plaquette(values, mu, 1, nu, 1)
                + _oriented_plaquette(values, nu, 1, mu, -1)
                + _oriented_plaquette(values, mu, -1, nu, -1)
                + _oriented_plaquette(values, nu, -1, mu, 1)
            )
            field = (clover - _adjoint(clover)) / (8.0j * plan.lattice_spacing**2)
            trace = jnp.trace(field, axis1=-2, axis2=-1) / plan.color_components
            fields.append(field - trace[..., None, None] * identity)
    return jnp.stack(tuple(fields), axis=plan.dimension)


def clover_topological_charge(
    plan: HypercubicGaugeObservablePlan,
    links: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Return site density and integrated charge with the standard 4D normalization."""
    if not isinstance(plan, HypercubicGaugeObservablePlan):
        raise TypeError("plan must be HypercubicGaugeObservablePlan.")
    if plan.dimension != 4:
        raise ValueError(
            "Clover topological charge is defined here only in four dimensions."
        )
    field = clover_field_strength(plan, links)
    f01, f02, f03, f12, f13, f23 = tuple(
        jnp.take(field, index, axis=plan.dimension) for index in range(6)
    )
    density = jnp.real(
        jnp.trace(
            contract("...ij,...jk->...ik", f01, f23)
            - contract("...ij,...jk->...ik", f02, f13)
            + contract("...ij,...jk->...ik", f03, f12),
            axis1=-2,
            axis2=-1,
        )
    ) / (4.0 * jnp.pi**2)
    charge = plan.lattice_spacing**4 * jnp.sum(density)
    return density, charge


def _routed_plaquette(
    transport: GaugeCovariantShiftPlan,
    links: Array,
    first_axis: int,
    first_direction: int,
    second_axis: int,
    second_direction: int,
    /,
) -> Array:
    group = transport.representation.group
    sites = jnp.arange(transport.site_count, dtype=jnp.int32)
    identity = group.identity(dtype=links.dtype)
    holonomy = jnp.broadcast_to(identity, (transport.site_count,) + identity.shape)
    for axis, direction in (
        (first_axis, first_direction),
        (second_axis, second_direction),
        (first_axis, -first_direction),
        (second_axis, -second_direction),
    ):
        if direction > 0:
            edges = transport.forward_edges[sites, axis]
            orientations = transport.forward_orientations[sites, axis]
            next_sites = transport.forward_sites[sites, axis]
        else:
            edges = transport.backward_edges[sites, axis]
            orientations = transport.backward_orientations[sites, axis]
            next_sites = transport.backward_sites[sites, axis]
        factors = links[edges]
        oriented = jnp.where(
            (orientations > 0)[..., None, None],
            factors,
            group.inverse(factors),
        )
        holonomy = contract("...ij,...jk->...ik", holonomy, oriented)
        sites = next_sites
    return holonomy


def routed_clover_field_strength(
    transport: GaugeCovariantShiftPlan,
    links: ArrayLike,
    /,
    *,
    lattice_spacing: float,
    maximum_field_bytes: int = 256 * 1024 * 1024,
) -> Array:
    """Return native-route clover field strengths as ``(site, plane, color, color)``."""
    if not isinstance(transport, GaugeCovariantShiftPlan):
        raise TypeError("transport must be GaugeCovariantShiftPlan.")
    values = jnp.asarray(links)
    if values.shape != transport.link_space.configuration_shape:
        raise ValueError("links shape does not match the gauge-covariant transport.")
    spacing = float(lattice_spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice_spacing must be finite and positive.")
    colors = transport.representation.group.dimension
    field_bytes = (
        transport.site_count
        * (transport.dimension * (transport.dimension - 1) // 2)
        * colors
        * colors
        * values.dtype.itemsize
    )
    if int(maximum_field_bytes) <= 0 or field_bytes > int(maximum_field_bytes):
        raise ValueError("Routed clover field exceeds maximum_field_bytes.")
    identity = jnp.eye(colors, dtype=values.dtype)
    fields: list[Array] = []
    for mu in range(transport.dimension):
        for nu in range(mu + 1, transport.dimension):
            clover = (
                _routed_plaquette(transport, values, mu, 1, nu, 1)
                + _routed_plaquette(transport, values, nu, 1, mu, -1)
                + _routed_plaquette(transport, values, mu, -1, nu, -1)
                + _routed_plaquette(transport, values, nu, -1, mu, 1)
            )
            field = (clover - _adjoint(clover)) / (8.0j * spacing**2)
            trace = jnp.trace(field, axis1=-2, axis2=-1) / colors
            fields.append(field - trace[:, None, None] * identity[None, :, :])
    return jnp.stack(tuple(fields), axis=1)


class RoutedTopologicalChargeResult(StrictModule):
    density: Array
    charge: Array
    finite: Array
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)


def measure_routed_topological_charge(
    transport: GaugeCovariantShiftPlan,
    links: ArrayLike,
    /,
    *,
    lattice_spacing: float,
    maximum_field_bytes: int = 256 * 1024 * 1024,
) -> RoutedTopologicalChargeResult:
    """Measure the standard 4D clover charge on native flattened gauge links."""
    if not isinstance(transport, GaugeCovariantShiftPlan):
        raise TypeError("transport must be GaugeCovariantShiftPlan.")
    if transport.dimension != 4:
        raise ValueError("Native routed topological charge requires four dimensions.")
    spacing = float(lattice_spacing)
    field = routed_clover_field_strength(
        transport,
        links,
        lattice_spacing=spacing,
        maximum_field_bytes=maximum_field_bytes,
    )
    f01, f02, f03, f12, f13, f23 = tuple(field[:, index] for index in range(6))
    density = jnp.real(
        jnp.trace(
            contract("...ij,...jk->...ik", f01, f23)
            - contract("...ij,...jk->...ik", f02, f13)
            + contract("...ij,...jk->...ik", f03, f12),
            axis1=-2,
            axis2=-1,
        )
    ) / (4.0 * jnp.pi**2)
    charge = spacing**4 * jnp.sum(density)
    observable_id = canonical_fingerprint(
        {
            "kind": "native-routed-clover-topological-charge",
            "transport": transport.plan_id,
            "lattice_spacing": spacing,
            "normalization": "epsilon-ff-over-32-pi-squared",
        }
    )
    return RoutedTopologicalChargeResult(
        density=density,
        charge=charge,
        finite=jnp.all(jnp.isfinite(density)) & jnp.isfinite(charge),
        topology_id=transport.link_space.topology.topology_id,
        field_space_id=transport.link_space.field_space.field_space_id,
        boundary_id=transport.boundary_id,
        observable_id=observable_id,
    )


def native_wilson_clover_term(
    wilson: WilsonDiracOperator,
    /,
    *,
    maximum_clover_bytes: int | None = None,
) -> Array:
    """Build the coefficient-free local ``a sigma.F / 4`` clover insertion."""
    if not isinstance(wilson, WilsonDiracOperator):
        raise TypeError("wilson must be WilsonDiracOperator.")
    block_size = wilson.spin_components * wilson.color_components
    required_bytes = (
        wilson.site_count * block_size * block_size * wilson.source.dtype.itemsize
    )
    maximum = (
        wilson.resource_policy.maximum_local_inverse_bytes
        if maximum_clover_bytes is None
        else int(maximum_clover_bytes)
    )
    if maximum <= 0 or required_bytes > maximum:
        raise ValueError("Native clover insertion exceeds maximum_clover_bytes.")
    term = jnp.zeros(
        (
            wilson.site_count,
            wilson.spin_components,
            wilson.color_components,
            wilson.spin_components,
            wilson.color_components,
        ),
        dtype=wilson.source.dtype,
    )
    fields = routed_clover_field_strength(
        wilson.transport,
        wilson.links,
        lattice_spacing=wilson.lattice_spacing,
        maximum_field_bytes=maximum,
    )
    plane = 0
    for mu in range(wilson.dimension):
        for nu in range(mu + 1, wilson.dimension):
            gamma_mu = wilson.gamma_matrices[mu]
            gamma_nu = wilson.gamma_matrices[nu]
            sigma = -0.5j * (
                contract("ab,bc->ac", gamma_mu, gamma_nu)
                - contract("ab,bc->ac", gamma_nu, gamma_mu)
            )
            term = term + 0.25 * wilson.lattice_spacing * contract(
                "ab,sij->saibj", sigma, fields[:, plane]
            )
            plane += 1
    return term


def measure_hypercubic_gauge_observables(
    plan: HypercubicGaugeObservablePlan,
    links: ArrayLike,
    /,
    *,
    measure_topology: bool = True,
) -> GaugeObservableResult:
    """Measure plaquette/action density and, when supported, clover topology."""
    values = plan.validate(links)
    plaquettes = hypercubic_plaquette_holonomies(plan, values)
    normalized_traces = (
        jnp.real(jnp.trace(plaquettes, axis1=-2, axis2=-1)) / plan.color_components
    )
    mean_plaquette = jnp.mean(normalized_traces)
    action_density = jnp.mean(1.0 - normalized_traces)
    if not measure_topology:
        density = None
        charge = None
        topology_status: TopologyStatus = "not-requested"
    elif plan.dimension != 4:
        density = None
        charge = None
        topology_status = "unsupported-dimension"
    else:
        density, charge = clover_topological_charge(plan, values)
        topology_status = "measured"
    finite_values = (
        (mean_plaquette, action_density)
        if charge is None
        else (
            mean_plaquette,
            action_density,
            charge,
            density,
        )
    )
    observable_id = canonical_fingerprint(
        {
            "kind": "hypercubic-gauge-observable-result",
            "plan": plan.plan_id,
            "topology_status": topology_status,
        }
    )
    return GaugeObservableResult(
        mean_plaquette=mean_plaquette,
        wilson_action_density=action_density,
        topological_charge=charge,
        topological_charge_density=density,
        finite=jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in finite_values))
        ),
        topology_id=plan.topology_id,
        field_space_id=plan.field_space_id,
        topology_status=topology_status,
        observable_id=observable_id,
    )


def measure_wilson_gauge_action(
    action: WilsonGaugeAction,
    links: ArrayLike,
    /,
) -> GaugeObservableResult:
    """Measure a generic Wilson action with its declared plaquette normalization."""
    if not isinstance(action, WilsonGaugeAction):
        raise TypeError("action must be WilsonGaugeAction.")
    traces = action.plaquette_traces(links)
    plaquette = jnp.mean(traces)
    density = action.canonical_action(links) / action.num_plaquettes
    return GaugeObservableResult(
        mean_plaquette=plaquette,
        wilson_action_density=density,
        topological_charge=None,
        topological_charge_density=None,
        finite=jnp.all(jnp.isfinite(traces)) & jnp.isfinite(density),
        topology_id=action.topology_id,
        field_space_id=action.field_space_id,
        topology_status="not-requested",
        observable_id=canonical_fingerprint(
            {
                "kind": "wilson-gauge-observable-result",
                "action": action.action_id,
                "normalization": "plaquette-mean-and-canonical-action-per-plaquette",
            }
        ),
    )


class WilsonFlowPlan(StrictModule, NonTrainableState):
    """Wilson-action Lie-midpoint flow with bounded deterministic backtracking."""

    action: WilsonGaugeAction
    native: GaugeGradientFlowPlan
    num_steps: int = eqx.field(static=True)
    maximum_history_bytes: int = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: WilsonGaugeAction,
        /,
        *,
        step_size: float,
        num_steps: int,
        maximum_backtracks: int = 8,
        descent_tolerance: float = 1.0e-10,
        maximum_history_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(action, WilsonGaugeAction):
            raise TypeError("action must be WilsonGaugeAction.")
        if not isinstance(action.link_space.group, SpecialUnitaryGroup):
            raise TypeError("Wilson flow currently requires an SU(N) link space.")
        maximum = int(maximum_history_bytes)
        if maximum <= 0:
            raise ValueError("maximum_history_bytes must be positive.")
        native = GaugeGradientFlowPlan(
            action,
            step_size=step_size,
            steps=num_steps,
            maximum_backtracks=maximum_backtracks,
            descent_tolerance=descent_tolerance,
        )
        self.action = action
        self.native = native
        self.num_steps = native.steps
        self.maximum_history_bytes = maximum
        self.flow_id = canonical_fingerprint(
            {
                "kind": "wilson-gradient-flow-observable-plan",
                "native_flow": native.plan_id,
                "topology": action.topology_id,
                "field_space": action.field_space_id,
                "retained": [
                    "final-links",
                    "action-history",
                    "accepted-step-sizes",
                    "final-observables",
                ],
            }
        )


class WilsonFlowResult(StrictModule):
    links: Array
    flow_times: Array
    action_history: Array
    accepted_step_sizes: Array
    mean_plaquette: Array
    action_density: Array
    valid: Array
    evidence: GaugeGradientFlowEvidence
    topology_id: str = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)


def run_wilson_flow(
    plan: WilsonFlowPlan,
    initial_links: ArrayLike,
    /,
) -> WilsonFlowResult:
    """Run the native fixed-shape group flow and retain its complete evidence."""
    if not isinstance(plan, WilsonFlowPlan):
        raise TypeError("plan must be WilsonFlowPlan.")
    values = jnp.asarray(initial_links)
    if values.shape != plan.action.configuration_shape:
        raise ValueError("initial_links shape does not match the Wilson action.")
    real_itemsize = jnp.empty((), dtype=jnp.real(values).dtype).dtype.itemsize
    retained_bytes = (
        values.size * values.dtype.itemsize + (2 * plan.num_steps + 1) * real_itemsize
    )
    if retained_bytes > plan.maximum_history_bytes:
        raise ValueError("Wilson-flow result exceeds maximum_history_bytes.")
    flowed = plan.native.flow(values)
    observables = flowed.observables
    return WilsonFlowResult(
        links=flowed.links,
        flow_times=jnp.concatenate(
            (
                jnp.zeros((1,), dtype=flowed.accepted_step_sizes.dtype),
                jnp.cumsum(flowed.accepted_step_sizes),
            )
        ),
        action_history=flowed.action_history,
        accepted_step_sizes=flowed.accepted_step_sizes,
        mean_plaquette=observables.mean_loop,
        action_density=observables.canonical_action_density,
        valid=flowed.evidence.successful & observables.finite,
        evidence=flowed.evidence,
        topology_id=plan.action.topology_id,
        flow_id=plan.flow_id,
    )


class StochasticSourcePlan(StrictModule, NonTrainableState):
    """Semantically addressed stochastic spinor sources, independent of update RNG."""

    source_shape: tuple[int, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    noise_kind: NoiseKind = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    measurement_id: str = eqx.field(static=True)
    randomness_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_shape: Sequence[int],
        source_count: int,
        /,
        *,
        configuration_id: str,
        measurement_id: str,
        randomness_id: str,
        noise_kind: NoiseKind = "z4",
        maximum_source_values: int = 1 << 24,
    ):
        shape = _positive_shape(source_shape, "source_shape")
        count = int(source_count)
        maximum = int(maximum_source_values)
        configuration = _identifier(configuration_id, "configuration_id")
        measurement = _identifier(measurement_id, "measurement_id")
        randomness = _identifier(randomness_id, "randomness_id")
        if count <= 0:
            raise ValueError("source_count must be positive.")
        if maximum <= 0 or prod(shape) * count > maximum:
            raise ValueError("Stochastic sources exceed maximum_source_values.")
        if noise_kind not in ("z2", "z4"):
            raise ValueError("noise_kind must be 'z2' or 'z4'.")
        source_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "lattice-stochastic-source",
                    "configuration": configuration,
                    "measurement": measurement,
                    "randomness": randomness,
                    "noise_kind": noise_kind,
                    "source_index": index,
                }
            )
            for index in range(count)
        )
        self.source_shape = shape
        self.source_ids = source_ids
        self.noise_kind = noise_kind
        self.configuration_id = configuration
        self.measurement_id = measurement
        self.randomness_id = randomness
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stochastic-source-plan",
                "shape": shape,
                "sources": source_ids,
            }
        )


def stochastic_source_plan_for_measurement(
    work_item: MeasurementWorkItem,
    source_shape: Sequence[int],
    /,
    *,
    noise_kind: NoiseKind = "z4",
    maximum_source_values: int = 1 << 24,
) -> StochasticSourcePlan:
    """Bind one stochastic source directly to a prespecified measurement address."""
    if not isinstance(work_item, MeasurementWorkItem):
        raise TypeError("work_item must be MeasurementWorkItem.")
    if work_item.update_randomness_id == work_item.source_randomness_id:
        raise ValueError("Measurement work item reuses its update randomness address.")
    return StochasticSourcePlan(
        source_shape,
        1,
        configuration_id=work_item.configuration_id,
        measurement_id=work_item.measurement_id,
        randomness_id=work_item.source_randomness_id,
        noise_kind=noise_kind,
        maximum_source_values=maximum_source_values,
    )


class StochasticSourceRealization(StrictModule):
    sources: Array
    finite: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def realize_stochastic_sources(
    plan: StochasticSourcePlan,
    key: Key[Array, ""],
    /,
) -> StochasticSourceRealization:
    """Realize normalized Z2/Z4 noise without changing semantic source identity."""
    if not isinstance(plan, StochasticSourcePlan):
        raise TypeError("plan must be StochasticSourcePlan.")
    keys = jr.split(key, len(plan.source_ids))

    def one(source_key):
        if plan.noise_kind == "z2":
            values = (
                2 * jr.bernoulli(source_key, shape=plan.source_shape).astype("float64")
                - 1
            )
            return values.astype(jnp.complex64)
        real_key, imag_key = jr.split(source_key)
        real = 2 * jr.bernoulli(real_key, shape=plan.source_shape).astype("float64") - 1
        imag = 2 * jr.bernoulli(imag_key, shape=plan.source_shape).astype("float64") - 1
        return (real + 1j * imag) / jnp.sqrt(2.0)

    sources = jax.vmap(one)(keys)
    sources = jnp.moveaxis(sources, 0, -1)
    return StochasticSourceRealization(
        sources=sources,
        finite=jnp.all(jnp.isfinite(sources)),
        source_ids=plan.source_ids,
        plan_id=plan.plan_id,
    )


class PropagatorSolvePlan(StrictModule, NonTrainableState):
    """Immutable Dirac right-hand sides and a bounded matrix-free solve policy."""

    operator: AbstractLatticeDiracOperator
    sources: Array
    solve_policy: LinearSolvePolicy
    source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLatticeDiracOperator,
        sources: ArrayLike,
        source_ids: Sequence[str],
        /,
        *,
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 1.0e-10,
        maximum_steps: int = 4096,
        maximum_workspace_bytes: int = 256 * 1024 * 1024,
        maximum_source_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(operator, AbstractLatticeDiracOperator):
            raise TypeError("operator must be AbstractLatticeDiracOperator.")
        if not isinstance(operator.target, ArraySpace):
            raise TypeError("Reference propagators require an ArraySpace Dirac target.")
        source = jnp.asarray(sources)
        identifiers = tuple(_identifier(value, "source_id") for value in source_ids)
        expected = operator.target.shape + (len(identifiers),)
        if not identifiers or source.shape != expected:
            raise ValueError(f"sources must have shape {expected} with nonempty IDs.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Propagator source IDs must be unique.")
        if source.dtype != operator.target.dtype:
            raise TypeError("Propagator source dtype must match the Dirac target space.")
        steps = int(maximum_steps)
        workspace = int(maximum_workspace_bytes)
        source_bytes = int(maximum_source_bytes)
        if steps <= 0 or workspace <= 0 or source_bytes <= 0:
            raise ValueError(
                "Propagator step, workspace, and source byte limits must be positive."
            )
        if source.nbytes > source_bytes:
            raise ValueError("Propagator sources exceed maximum_source_bytes.")
        policy = LinearSolvePolicy(
            BiCGStab(),
            tolerance=TolerancePolicy(
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=steps,
            ),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
            resources=SolveResourcePolicy(
                factorization_bytes=0,
                workspace_bytes=workspace,
                krylov_basis_bytes=workspace,
                preconditioner_bytes=0,
                recycling_state_bytes=0,
            ),
        )
        self.operator = operator
        self.sources = source
        self.solve_policy = policy
        self.source_ids = identifiers
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matrix-free-dirac-propagator-solve",
                "operator": operator.operator_id,
                "sources": array_tree_fingerprint(np.asarray(source)),
                "source_ids": identifiers,
                "method": "bicgstab",
                "relative_tolerance": float(relative_tolerance),
                "absolute_tolerance": float(absolute_tolerance),
                "maximum_steps": steps,
                "maximum_workspace_bytes": workspace,
                "maximum_source_bytes": source_bytes,
            }
        )


class PreparedPropagatorSolve(StrictModule, NonTrainableState):
    plan: PropagatorSolvePlan
    solver: PreparedLinearSolve
    prepared_id: str = eqx.field(static=True)


class PropagatorResult(StrictModule):
    values: Array
    residual_norm: Array
    relative_residual: Array
    status: Array
    finite: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_propagator_solve(plan: PropagatorSolvePlan, /) -> PreparedPropagatorSolve:
    if not isinstance(plan, PropagatorSolvePlan):
        raise TypeError("plan must be PropagatorSolvePlan.")
    problem = LinearSystem(
        plan.operator,
        problem_id=f"{plan.plan_id}:dirac-system",
    )
    rhs_layout = RHSLayout((len(plan.source_ids),))
    solver = prepare(problem, plan.solve_policy, rhs_layout=rhs_layout)
    return PreparedPropagatorSolve(
        plan=plan,
        solver=solver,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-dirac-propagator-solve",
                "plan": plan.plan_id,
                "linear_plan": solver.plan.plan_id,
                "numeric_version": int(np.asarray(solver.numeric_version)),
            }
        ),
    )


def solve_propagators(prepared: PreparedPropagatorSolve, /) -> PropagatorResult:
    """Solve all declared sources through one reusable Phydrax Krylov plan."""
    if not isinstance(prepared, PreparedPropagatorSolve):
        raise TypeError("prepared must be PreparedPropagatorSolve.")
    linear = solve(prepared.solver, prepared.plan.sources)
    values = jnp.asarray(linear.value)
    return PropagatorResult(
        values=values,
        residual_norm=linear.diagnostics.residual_norm,
        relative_residual=linear.diagnostics.relative_residual,
        status=linear.status,
        finite=jnp.all(jnp.isfinite(values)),
        source_ids=prepared.plan.source_ids,
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
    )


def point_source_vectors(
    operator: AbstractLatticeDiracOperator,
    source_site: int,
    /,
    *,
    source_id_prefix: str,
    maximum_source_values: int = 1 << 24,
) -> tuple[Array, tuple[str, ...]]:
    """Build a complete spin-color point-source basis at one flattened site."""
    if not isinstance(operator, AbstractLatticeDiracOperator):
        raise TypeError("operator must be AbstractLatticeDiracOperator.")
    if not isinstance(operator.target, ArraySpace) or len(operator.target.shape) != 3:
        raise TypeError("Point sources require target shape (site, spin/taste, color).")
    sites, spin, color = operator.target.shape
    site = int(source_site)
    prefix = _identifier(source_id_prefix, "source_id_prefix")
    count = spin * color
    if site < 0 or site >= sites:
        raise ValueError("source_site lies outside the Dirac field.")
    if int(maximum_source_values) <= 0 or sites * spin * color * count > int(
        maximum_source_values
    ):
        raise ValueError("Point-source basis exceeds maximum_source_values.")
    sources = jnp.zeros(operator.target.shape + (count,), dtype=operator.target.dtype)
    source_spin = jnp.repeat(jnp.arange(spin), color)
    source_color = jnp.tile(jnp.arange(color), spin)
    sources = sources.at[site, source_spin, source_color, jnp.arange(count)].set(1)
    ids = tuple(
        canonical_fingerprint(
            {
                "kind": "dirac-point-source",
                "operator": operator.operator_id,
                "prefix": prefix,
                "site": site,
                "spin": source_spin_index,
                "color": source_color_index,
            }
        )
        for source_spin_index in range(spin)
        for source_color_index in range(color)
    )
    return sources, ids


def point_to_all_propagator(
    result: PropagatorResult,
    lattice_shape: Sequence[int],
    /,
) -> Array:
    """Reshape a complete point-source solve to ``site...,spin,color,spin,color``."""
    if not isinstance(result, PropagatorResult):
        raise TypeError("result must be PropagatorResult.")
    shape = _positive_shape(lattice_shape, "lattice_shape")
    values = result.values
    if values.ndim != 4 or prod(shape) != values.shape[0]:
        raise ValueError("Propagator values and lattice_shape are incompatible.")
    _, spin, color, sources = values.shape
    if sources != spin * color:
        raise ValueError(
            "Complete point source requires one RHS per spin-color component."
        )
    return values.reshape(shape + (spin, color, spin, color))


def _spatial_reduce(
    values: Array,
    time_axis: int,
    normalization: SpatialNormalization,
    /,
) -> Array:
    dimension = values.ndim
    time = int(time_axis) % dimension
    spatial_axes = tuple(axis for axis in range(dimension) if axis != time)
    reduced = jnp.sum(values, axis=spatial_axes)
    if normalization == "mean":
        reduced = reduced / prod(values.shape[axis] for axis in spatial_axes)
    elif normalization != "sum":
        raise ValueError("spatial_normalization must be 'sum' or 'mean'.")
    return reduced


def meson_correlator(
    propagator: ArrayLike,
    source_gamma: ArrayLike,
    sink_gamma: ArrayLike,
    /,
    *,
    time_axis: int = -1,
    spatial_normalization: SpatialNormalization = "sum",
) -> Array:
    """Contract ``Tr[Γ_sink S Γ_source S†]`` and reduce spatial sites."""
    values = jnp.asarray(propagator)
    if values.ndim < 6:
        raise ValueError(
            "propagator must have lattice axes followed by sink/source spin-color axes."
        )
    spin, color, source_spin, source_color = values.shape[-4:]
    if spin != source_spin or color != source_color:
        raise ValueError("Sink and source spin-color propagator dimensions must agree.")
    source = jnp.asarray(source_gamma, dtype=values.dtype)
    sink = jnp.asarray(sink_gamma, dtype=values.dtype)
    if source.shape != (spin, spin) or sink.shape != (spin, spin):
        raise ValueError("Meson gamma matrices must match the propagator spin dimension.")
    identity = jnp.eye(color, dtype=values.dtype)
    source_sc = contract("ab,ij->aibj", source, identity).reshape(
        (spin * color, spin * color)
    )
    sink_sc = contract("ab,ij->aibj", sink, identity).reshape(
        (spin * color, spin * color)
    )
    matrix = values.reshape(values.shape[:-4] + (spin * color, spin * color))
    site_values = contract(
        "ab,...bc,cd,...ad->...",
        sink_sc,
        matrix,
        source_sc,
        jnp.conj(matrix),
    )
    return _spatial_reduce(
        site_values,
        time_axis % (values.ndim - 4),
        spatial_normalization,
    )


def color_levi_civita_three(dtype=jnp.float32, /) -> Array:
    """Return ε_ijk for three-color baryon contractions."""
    epsilon = jnp.zeros((3, 3, 3), dtype=dtype)
    return epsilon.at[
        jnp.asarray((0, 1, 2, 1, 2, 0)),
        jnp.asarray((1, 2, 0, 0, 1, 2)),
        jnp.asarray((2, 0, 1, 2, 0, 1)),
    ].set(jnp.asarray((1, 1, 1, -1, -1, -1), dtype=dtype))


def baryon_correlator(
    first_propagator: ArrayLike,
    second_propagator: ArrayLike,
    third_propagator: ArrayLike,
    sink_spin_tensor: ArrayLike,
    source_spin_tensor: ArrayLike,
    /,
    *,
    time_axis: int = -1,
    spatial_normalization: SpatialNormalization = "sum",
    normalize_color: bool = True,
) -> Array:
    """Perform the exact two-epsilon three-quark color/spin contraction."""
    first = jnp.asarray(first_propagator)
    second = jnp.asarray(second_propagator)
    third = jnp.asarray(third_propagator)
    if first.shape != second.shape or first.shape != third.shape or first.ndim < 6:
        raise ValueError(
            "All baryon propagators must share one valid point-to-all shape."
        )
    spin, color, source_spin, source_color = first.shape[-4:]
    if color != 3 or source_color != 3 or spin != source_spin:
        raise ValueError(
            "Baryon contraction requires equal-spin, three-color propagators."
        )
    sink = jnp.asarray(sink_spin_tensor, dtype=first.dtype)
    source = jnp.asarray(source_spin_tensor, dtype=first.dtype)
    if sink.shape != (spin, spin, spin) or source.shape != sink.shape:
        raise ValueError("Baryon spin tensors must have shape (spin, spin, spin).")
    epsilon = color_levi_civita_three(first.dtype)
    site_values = contract(
        "abc,ijk,def,lmn,...aidl,...bjem,...ckfn->...",
        sink,
        epsilon,
        jnp.conj(source),
        epsilon,
        first,
        second,
        third,
    )
    if normalize_color:
        site_values = site_values / 6.0
    return _spatial_reduce(
        site_values,
        time_axis % (first.ndim - 4),
        spatial_normalization,
    )


__all__ = [
    "GaugeObservableResult",
    "HypercubicGaugeObservablePlan",
    "NoiseKind",
    "PreparedPropagatorSolve",
    "PropagatorResult",
    "PropagatorSolvePlan",
    "RoutedTopologicalChargeResult",
    "SpatialNormalization",
    "StochasticSourcePlan",
    "StochasticSourceRealization",
    "TopologyStatus",
    "WilsonFlowPlan",
    "WilsonFlowResult",
    "baryon_correlator",
    "clover_field_strength",
    "clover_topological_charge",
    "color_levi_civita_three",
    "hypercubic_plaquette_holonomies",
    "measure_hypercubic_gauge_observables",
    "measure_wilson_gauge_action",
    "measure_routed_topological_charge",
    "meson_correlator",
    "point_source_vectors",
    "native_wilson_clover_term",
    "point_to_all_propagator",
    "prepare_propagator_solve",
    "realize_stochastic_sources",
    "routed_clover_field_strength",
    "stochastic_source_plan_for_measurement",
    "run_wilson_flow",
    "solve_propagators",
]
