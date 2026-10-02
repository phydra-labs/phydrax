#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Overdamped, volume-preserving capillary relaxation of explicit foams.

The first-order (inertia-free) film motion is the constrained gradient flow

``zeta A_v dx_v/dt = -dE/dx_v - sum_r lambda_r dV_r/dx_v,   dV_r/dt = 0``

with the surface energy ``E = sum_f gamma_f A_f``, the lumped barycentric
vertex area ``A_v``, a film friction coefficient ``zeta`` (force per unit area
per unit velocity) and one multiplier per independent finite-region volume.
Eliminating the multipliers gives ``lambda = -(J M J^T)^{-1} J M grad E`` with the
mobility ``M = diag(1 / (zeta A_v))`` (zero on wire-fixed components), so the
velocity conserves every constrained volume to first order and the region
pressures relative to the boundary labels are ``p = -lambda`` (the equilibrium
module's convention). Each explicit step is followed by a mobility-weighted
Newton projection onto the volume targets. The step is the plan's time step
limited so that no vertex moves farther than ``maximum_displacement_fraction``
of the current shortest active edge; this geometric step control keeps the
fixed-topology flow away from element inversion between topology event passes
but is not a collision certificate (event passes certify their own motions).

This route drives quasi-static evolution toward equilibrium and through the
loss of an equilibrium (catenoid collapse) between `apply_surface_events`
passes. It carries no film inertia, air flow or drainage; those belong to the
constrained foam dynamics. No derivative is claimed.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ... import ein
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import positive_finite_float, positive_integer
from ...geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
)
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear,
    solve as solve_linear,
)
from ...typing import Bool, checked, Dim, Float, Float64, Identifier, Int32, Scalar
from ._contracts import FoamMaterialPlan
from ._equilibrium import _independent_rows, _volume_jacobian, _wire_coordinates


class FoamRelaxationStatus(IntEnum):
    """Outcome of one overdamped relaxation call."""

    COMPLETED = 0
    NONFINITE = 1
    VOLUME_PROJECTION_FAILED = 2


@final
class FoamRelaxationPlan(StrictModule):
    """Friction, step control and projection settings of overdamped relaxation.

    ``friction`` is ``zeta`` (force per area per velocity), ``time_step`` the
    largest step, ``steps`` the fixed number of steps per call,
    ``maximum_displacement_fraction`` the per-step vertex displacement limit
    relative to the shortest active edge, ``projection_iterations`` the Newton
    iterations of the volume projection and ``volume_tolerance`` the accepted
    final relative volume residual.
    """

    __strict_contract__ = True

    friction: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    projection_iterations: int = eqx.field(static=True)
    volume_tolerance: float = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        friction: float,
        time_step: float,
        steps: int,
        maximum_displacement_fraction: float = 0.1,
        projection_iterations: int = 4,
        volume_tolerance: float = 1.0e-9,
    ) -> None:
        zeta = positive_finite_float(friction, "friction")
        step = positive_finite_float(time_step, "time_step")
        count = positive_integer(steps, "steps")
        fraction = positive_finite_float(
            maximum_displacement_fraction, "maximum_displacement_fraction"
        )
        if fraction >= 0.5:
            raise ValueError("maximum_displacement_fraction must be below 0.5.")
        iterations = positive_integer(projection_iterations, "projection_iterations")
        tolerance = positive_finite_float(volume_tolerance, "volume_tolerance")
        self.friction = zeta
        self.time_step = step
        self.steps = count
        self.maximum_displacement_fraction = fraction
        self.projection_iterations = iterations
        self.volume_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "foam-relaxation-plan",
                "friction": float(zeta).hex(),
                "time_step": float(step).hex(),
                "steps": count,
                "maximum_displacement_fraction": float(fraction).hex(),
                "projection_iterations": iterations,
                "volume_tolerance": float(tolerance).hex(),
            }
        )


class _RelaxVertexDim(Dim, minimum=3):
    """Vertex slots."""


class _RelaxFaceDim(Dim, minimum=1):
    """Face slots."""


class _RelaxRegionDim(Dim, minimum=2):
    """Region slots."""


class _RelaxConstraintDim(Dim):
    """Independent finite-region volume rows."""


class _RelaxWireDim(Dim):
    """Wire-fixed coordinates."""


@final
class FoamRelaxationEvidence(StrictModule):
    """Step, energy, constraint and geometry evidence of one relaxation call.

    ``energy_monotone`` holds when no step increased the surface energy beyond
    roundoff; ``volume_residual`` is the final largest relative volume defect
    of the constrained regions; ``projection_successful`` records every native
    Gram solve of the velocity and projection systems.
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    elapsed_time: Float[Scalar]
    initial_energy: Float[Scalar]
    final_energy: Float[Scalar]
    energy_monotone: Bool[Scalar]
    volume_residual: Float[Scalar]
    projection_successful: Bool[Scalar]
    minimum_face_area: Float[Scalar]
    minimum_edge_length: Float[Scalar]
    finite: Bool[Scalar]
    steps: int = eqx.field(static=True)
    constrained_region_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)


@final
class FoamRelaxationResult(StrictModule):
    """Relaxed state, region pressures (boundary labels zero) and evidence."""

    __strict_contract__ = True

    state: MultiRegionSurfaceState
    pressures: Float[_RelaxRegionDim]
    region_volumes: Float[_RelaxRegionDim]
    energy: Float[Scalar]
    evidence: FoamRelaxationEvidence

    @property
    def successful(self) -> bool:
        """Host decision: completed with satisfied volume constraints."""
        return int(self.evidence.status) == FoamRelaxationStatus.COMPLETED


@final
class PreparedFoamRelaxation(StrictModule):
    """Prepared overdamped relaxation of one foam topology epoch."""

    __strict_contract__ = True

    surface: PreparedMultiRegionSurface
    material: FoamMaterialPlan
    plan: FoamRelaxationPlan
    face_tension: Float64[_RelaxFaceDim]
    mobility_mask: Float64[_RelaxVertexDim, Literal[3]]
    wire_flat_indices: Int32[_RelaxWireDim]
    wire_values: Float64[_RelaxWireDim]
    constrained_slots: Int32[_RelaxConstraintDim]
    constrained_region_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: FoamRelaxationPlan,
        surface: PreparedMultiRegionSurface,
        material: FoamMaterialPlan,
        state: MultiRegionSurfaceState,
        /,
    ) -> None:
        topology = surface.topology
        state.require_topology(topology)
        first, second = material.face_tension_indices(topology)
        tension = np.where(
            np.asarray(topology.face_active),
            np.asarray(
                material.tensions.pair_values(jnp.asarray(first), jnp.asarray(second))
            ),
            0.0,
        )
        wire_flat, wire_values = _wire_coordinates(material, topology)
        mask = np.zeros((topology.vertex_capacity * 3,), dtype=np.float64)
        mask[: 3 * topology.vertex_count] = 1.0
        mask[wire_flat] = 0.0
        wire_positions = (
            np.zeros((0,), dtype=np.float64)
            if material.wires is None
            else np.asarray(material.wires.positions, dtype=np.float64).reshape(-1)[
                wire_values
            ]
        )
        finite = topology.finite_region_indices
        positions = np.asarray(state.positions, dtype=np.float64)
        free = np.flatnonzero(mask > 0.0)
        rows = _volume_jacobian(surface, positions, free, finite)
        constrained = _independent_rows(rows) if finite else ()
        slots = tuple(finite[row] for row in constrained)
        self.surface = surface
        self.material = material
        self.plan = plan
        self.face_tension = jnp.asarray(tension, dtype=jnp.float64)
        self.mobility_mask = jnp.asarray(mask.reshape((-1, 3)))
        self.wire_flat_indices = jnp.asarray(wire_flat, dtype=jnp.int32)
        self.wire_values = jnp.asarray(wire_positions, dtype=jnp.float64)
        self.constrained_slots = jnp.asarray(slots, dtype=jnp.int32)
        self.constrained_region_ids = tuple(topology.region_ids[slot] for slot in slots)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-foam-relaxation",
                "plan": plan.plan_id,
                "surface": surface.prepared_id,
                "material": material.material_id,
                "constrained": list(self.constrained_region_ids),
            }
        )

    def relax(
        self, state: MultiRegionSurfaceState, volume_targets: ArrayLike, /
    ) -> FoamRelaxationResult:
        """Run ``plan.steps`` overdamped steps from ``state``.

        ``volume_targets`` lists every finite region in region-table order (as
        for the equilibrium); only the independent rows are imposed.
        """
        topology = self.surface.topology
        state.require_topology(topology)
        targets = jnp.asarray(volume_targets, dtype=jnp.float64)
        finite = topology.finite_region_indices
        if targets.shape != (len(finite),):
            raise ValueError(
                "volume_targets must list every finite region in table order."
            )
        full = jnp.zeros((topology.region_capacity,), dtype=jnp.float64)
        if finite:
            full = full.at[jnp.asarray(finite, dtype=jnp.int32)].set(targets)
        return _relax(self, state, full)


def _mobility(prepared: PreparedFoamRelaxation, positions: Array, /) -> Array:
    area = jnp.sum(prepared.surface.slot_areas(positions), axis=1)
    safe = jnp.where(area > 0.0, area, 1.0)
    return prepared.mobility_mask / (prepared.plan.friction * safe)[:, None]


def _gram_solve(jacobian: Array, mobility: Array, rhs: Array, /) -> tuple[Array, Array]:
    """Native dense solve of ``(J M J^T) y = rhs``."""
    weighted = jacobian * mobility[None]
    gram = ein.contract("rvk,svk->rs", weighted, jacobian)
    prepared = prepare_linear(
        LinearSystem(DenseLinearOperator(gram)), LinearSolvePolicy(DenseLU())
    )
    result = solve_linear(prepared, rhs)
    return result.value, result.successful


def _constraint_jacobian(prepared: PreparedFoamRelaxation, positions: Array, /) -> Array:
    def volumes(values: Array, /) -> Array:
        return prepared.surface.region_volumes(values)[prepared.constrained_slots]

    return jax.jacrev(volumes)(positions)


def _project(
    prepared: PreparedFoamRelaxation, positions: Array, targets: Array, mobility: Array, /
) -> tuple[Array, Array]:
    ok = jnp.asarray(True)
    for _ in range(prepared.plan.projection_iterations):
        defect = (
            prepared.surface.region_volumes(positions)[prepared.constrained_slots]
            - targets[prepared.constrained_slots]
        )
        jacobian = _constraint_jacobian(prepared, positions)
        correction, solved = _gram_solve(jacobian, mobility, defect)
        positions = positions - mobility * ein.contract("r,rvk->vk", correction, jacobian)
        ok = ok & solved
    return positions, ok


def _edge_minimum(prepared: PreparedFoamRelaxation, positions: Array, /) -> Array:
    topology = prepared.surface.topology
    edges = jnp.maximum(topology.edges, 0)
    lengths = jnp.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    return jnp.min(jnp.where(topology.edge_active, lengths, jnp.inf))


def _relax_impl(
    prepared: PreparedFoamRelaxation, state: MultiRegionSurfaceState, targets: Array, /
) -> FoamRelaxationResult:
    surface = prepared.surface
    plan = prepared.plan
    constrained = prepared.constrained_slots.size > 0
    flat = (
        state.positions.reshape(-1)
        .at[prepared.wire_flat_indices]
        .set(prepared.wire_values)
    )
    start = flat.reshape((-1, 3))

    def energy_of(values: Array, /) -> Array:
        return surface.surface_energy(values, prepared.face_tension)

    initial_energy = energy_of(start)

    def step(
        _: int, carry: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        positions, elapsed, previous, monotone, ok, pressures = carry
        energy, gradient = jax.value_and_grad(energy_of)(positions)
        mobility = _mobility(prepared, positions)
        velocity = -mobility * gradient
        multipliers = jnp.zeros((prepared.constrained_slots.size,), dtype=positions.dtype)
        if constrained:
            jacobian = _constraint_jacobian(prepared, positions)
            rhs = -ein.contract("rvk,vk->r", jacobian, mobility * gradient)
            multipliers, solved = _gram_solve(jacobian, mobility, rhs)
            velocity = -mobility * (
                gradient + ein.contract("r,rvk->vk", multipliers, jacobian)
            )
            ok = ok & solved
        speed = jnp.max(jnp.linalg.norm(velocity, axis=1))
        limit = plan.maximum_displacement_fraction * _edge_minimum(prepared, positions)
        dt = jnp.minimum(plan.time_step, limit / jnp.maximum(speed, 1e-300))
        moved = positions + dt * velocity
        if constrained:
            moved, solved = _project(prepared, moved, targets, mobility)
            ok = ok & solved
        tolerance = 1.0e-12 * jnp.maximum(jnp.abs(energy), 1e-300)
        monotone = monotone & (energy <= previous + tolerance)
        pressures = pressures.at[prepared.constrained_slots].set(-multipliers)
        return moved, elapsed + dt, energy, monotone, ok, pressures

    carry = (
        start,
        jnp.asarray(0.0, dtype=start.dtype),
        initial_energy,
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.zeros((surface.topology.region_capacity,), dtype=start.dtype),
    )
    positions, elapsed, _, monotone, ok, pressures = jax.lax.fori_loop(
        0, plan.steps, step, carry
    )
    energy = energy_of(positions)
    monotone = monotone & (energy <= initial_energy + 1.0e-12 * jnp.abs(initial_energy))
    volumes = surface.region_volumes(positions)
    scale = jnp.maximum(jnp.max(jnp.abs(targets)), 1e-300)
    residual = (
        jnp.max(jnp.abs(volumes - targets)[prepared.constrained_slots]) / scale
        if constrained
        else jnp.asarray(0.0, dtype=positions.dtype)
    )
    finite = jnp.all(jnp.isfinite(positions)) & jnp.isfinite(energy)
    status = jnp.where(
        ~finite,
        int(FoamRelaxationStatus.NONFINITE),
        jnp.where(
            ~ok | (residual > plan.volume_tolerance),
            int(FoamRelaxationStatus.VOLUME_PROJECTION_FAILED),
            int(FoamRelaxationStatus.COMPLETED),
        ),
    ).astype(jnp.int32)
    areas = surface.face_areas(positions)
    evidence = FoamRelaxationEvidence(
        status=status,
        elapsed_time=elapsed,
        initial_energy=initial_energy,
        final_energy=energy,
        energy_monotone=monotone,
        volume_residual=residual,
        projection_successful=ok,
        minimum_face_area=jnp.min(
            jnp.where(surface.topology.face_active, areas, jnp.inf)
        ),
        minimum_edge_length=_edge_minimum(prepared, positions),
        finite=finite,
        steps=plan.steps,
        constrained_region_ids=prepared.constrained_region_ids,
        prepared_id=prepared.prepared_id,
    )
    return FoamRelaxationResult(
        state=state.with_positions(positions),
        pressures=pressures,
        region_volumes=volumes,
        energy=energy,
        evidence=evidence,
    )


_relax = eqx.filter_jit(_relax_impl)


__all__ = [
    "FoamRelaxationEvidence",
    "FoamRelaxationPlan",
    "FoamRelaxationResult",
    "FoamRelaxationStatus",
    "PreparedFoamRelaxation",
]
