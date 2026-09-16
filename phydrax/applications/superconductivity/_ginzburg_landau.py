#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.simplicial._ddg import DDGOperators, discrete_operators
from ...geometry.simplicial._mesh import TriangleMesh
from ...graph._charged_scalar_gauge import ChargedGaugeState, ChargedScalarGaugePlan


class GinzburgLandauEnergy(StrictModule):
    condensation: Array
    kinetic: Array
    magnetic: Array
    total: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class GinzburgLandauState(StrictModule):
    gauge: ChargedGaugeState
    time: Array
    accepted_step: Array
    plan_id: str = eqx.field(static=True)


class GinzburgLandauEvidence(StrictModule):
    gradient_norm: Array
    gauge_energy_residual: Array
    gauge_covariance_residual: Array
    energy_change: Array
    finite: Array
    converged: Array
    derivative_valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GinzburgLandauResult(StrictModule):
    state: GinzburgLandauState
    energy: GinzburgLandauEnergy
    evidence: GinzburgLandauEvidence
    plan_id: str = eqx.field(static=True)


class TDGLStepResult(StrictModule):
    candidate: GinzburgLandauState
    accepted: GinzburgLandauState
    initial_energy: GinzburgLandauEnergy
    candidate_energy: GinzburgLandauEnergy
    evidence: GinzburgLandauEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class GaugeCovariantGLPlan(StrictModule, NonTrainableState):
    mesh: TriangleMesh
    operators: DDGOperators
    gauge: ChargedScalarGaugePlan
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    kinetic_coefficient: float = eqx.field(static=True)
    magnetic_coefficient: float = eqx.field(static=True)
    applied_normal_field: Array
    phase_anchor: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: TriangleMesh,
        /,
        *,
        alpha: float,
        beta: float,
        kinetic_coefficient: float,
        magnetic_coefficient: float,
        gauge_coupling: float,
        applied_normal_field: ArrayLike = 0.0,
        phase_anchor: int = 0,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("mesh must be TriangleMesh.")
        values = tuple(
            float(value)
            for value in (
                alpha,
                beta,
                kinetic_coefficient,
                magnetic_coefficient,
                gauge_coupling,
                tolerance,
            )
        )
        if (
            any(not isfinite(value) for value in values)
            or values[1] <= 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
            or values[4] == 0.0
            or values[5] <= 0.0
        ):
            raise ValueError("GL material, gauge, or tolerance values are invalid.")
        anchor = int(phase_anchor)
        if not 0 <= anchor < mesh.vertices.shape[0]:
            raise ValueError("GL phase anchor is outside the vertex range.")
        field = jnp.broadcast_to(
            jnp.asarray(applied_normal_field, dtype=mesh.vertices.dtype),
            (mesh.faces.shape[0],),
        )
        if not bool(jnp.all(jnp.isfinite(field))):
            raise ValueError("Applied GL field must be finite.")
        operators = discrete_operators(mesh)
        gauge = ChargedScalarGaugePlan(
            mesh.vertices,
            mesh.faces,
            mesh.topology.edges,
            coupling=values[4],
        )
        self.mesh = mesh
        self.operators = operators
        self.gauge = gauge
        self.alpha, self.beta = values[:2]
        self.kinetic_coefficient = values[2]
        self.magnetic_coefficient = values[3]
        self.applied_normal_field = field
        self.phase_anchor = anchor
        self.tolerance = values[5]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-mesh-gauge-covariant-gl",
                "mesh": mesh.source_id,
                "gauge": gauge.plan_id,
                "alpha": self.alpha,
                "beta": self.beta,
                "kinetic_coefficient": self.kinetic_coefficient,
                "magnetic_coefficient": self.magnetic_coefficient,
                "applied_field": np.asarray(field).tolist(),
                "phase_anchor": anchor,
                "tolerance": self.tolerance,
            }
        )

    def initialize(
        self, order_parameter: ArrayLike, vector_potential: ArrayLike | None = None, /
    ) -> GinzburgLandauState:
        scalar = jnp.asarray(order_parameter)
        if scalar.shape != (self.mesh.vertices.shape[0],) or not jnp.iscomplexobj(scalar):
            raise ValueError("GL order parameter must be one complex value per vertex.")
        potential = (
            jnp.zeros((self.mesh.topology.edges.shape[0],), dtype=jnp.real(scalar).dtype)
            if vector_potential is None
            else jnp.asarray(vector_potential, dtype=jnp.real(scalar).dtype)
        )
        gauge = self.gauge.state(scalar, potential)
        gauge = self._anchor(gauge)
        return GinzburgLandauState(
            gauge,
            jnp.asarray(0.0, dtype=potential.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def _anchor(self, state: ChargedGaugeState, /) -> ChargedGaugeState:
        phase = jnp.angle(state.scalar[self.phase_anchor])
        scalar = state.scalar * jnp.exp(-1.0j * phase)
        scalar = scalar.at[self.phase_anchor].set(
            jnp.real(scalar[self.phase_anchor]) + 0.0j
        )
        return ChargedGaugeState(scalar, state.vector_potential, self.gauge.plan_id)

    def energy(self, state: GinzburgLandauState, /) -> GinzburgLandauEnergy:
        if not isinstance(state, GinzburgLandauState) or state.plan_id != self.plan_id:
            raise TypeError("state must belong to this GaugeCovariantGLPlan.")
        scalar = state.gauge.scalar
        magnitude_squared = jnp.abs(scalar) ** 2
        condensation = jnp.sum(
            self.operators.vertex_mass
            * (self.alpha * magnitude_squared + 0.5 * self.beta * magnitude_squared**2)
        )
        covariant = self.gauge.covariant_difference(state.gauge)
        kinetic = self.kinetic_coefficient * jnp.sum(
            self.operators.edge_weights * jnp.abs(covariant) ** 2
        )
        flux_density = self.gauge.curvature(state.gauge) / self.operators.face_area
        magnetic = (
            0.5
            * self.magnetic_coefficient
            * jnp.sum(
                self.operators.face_area * (flux_density - self.applied_normal_field) ** 2
            )
        )
        total = jnp.real(condensation + kinetic + magnetic)
        finite = (
            jnp.isfinite(total)
            & jnp.isfinite(condensation)
            & jnp.isfinite(kinetic)
            & jnp.isfinite(magnetic)
        )
        return GinzburgLandauEnergy(
            jnp.real(condensation),
            jnp.real(kinetic),
            jnp.real(magnetic),
            total,
            finite,
            self.plan_id,
        )

    def _encode(self, state: ChargedGaugeState, /) -> Array:
        return jnp.concatenate(
            (
                jnp.real(state.scalar),
                jnp.imag(state.scalar),
                state.vector_potential,
            )
        )

    def _decode(self, coordinates: Array, /) -> ChargedGaugeState:
        count = self.mesh.vertices.shape[0]
        scalar = coordinates[:count] + 1.0j * coordinates[count : 2 * count]
        potential = coordinates[2 * count :]
        return self._anchor(ChargedGaugeState(scalar, potential, self.gauge.plan_id))

    def _coordinate_energy(self, coordinates: Array, template: GinzburgLandauState, /):
        gauge = self._decode(coordinates)
        return self.energy(
            GinzburgLandauState(
                gauge, template.time, template.accepted_step, self.plan_id
            )
        ).total

    def solve(
        self,
        initial: GinzburgLandauState,
        /,
        *,
        step_size: float = 0.05,
        iterations: int = 256,
    ) -> GinzburgLandauResult:
        if initial.plan_id != self.plan_id:
            raise ValueError("Initial GL state belongs to another plan.")
        step = float(step_size)
        count = int(iterations)
        if not isfinite(step) or step <= 0.0 or count < 1:
            raise ValueError("GL solve step_size or iterations are invalid.")
        coordinates = self._encode(initial.gauge)
        gradient_function = jax.grad(self._coordinate_energy)
        for _ in range(count):
            gradient = gradient_function(coordinates, initial)
            coordinates = coordinates - step * gradient
        gauge = self._decode(coordinates)
        state = GinzburgLandauState(
            gauge, initial.time, initial.accepted_step, self.plan_id
        )
        gradient = gradient_function(coordinates, initial)
        gradient_norm = jnp.linalg.norm(gradient)
        energy = self.energy(state)
        parameter = jnp.linspace(-0.2, 0.2, state.gauge.scalar.size)
        transformed = self.gauge.transform(state.gauge, parameter)
        transformed_energy = self.energy(
            GinzburgLandauState(
                transformed, state.time, state.accepted_step, self.plan_id
            )
        )
        gauge_evidence = self.gauge.validate_transform(state.gauge, parameter)
        gauge_residual = jnp.abs(transformed_energy.total - energy.total)
        finite = energy.finite & jnp.isfinite(gradient_norm) & transformed_energy.finite
        converged = gradient_norm <= self.tolerance
        successful = (
            finite
            & converged
            & gauge_evidence.successful
            & (gauge_residual <= self.tolerance * jnp.maximum(jnp.abs(energy.total), 1.0))
        )
        evidence = GinzburgLandauEvidence(
            gradient_norm,
            gauge_residual,
            gauge_evidence.scalar_covariance_residual,
            energy.total - self.energy(initial).total,
            finite,
            converged,
            successful,
            successful,
            self.plan_id,
        )
        return GinzburgLandauResult(state, energy, evidence, self.plan_id)

    def tdgl_step(
        self,
        state: GinzburgLandauState,
        step_size: ArrayLike,
        /,
        *,
        order_parameter_mobility: float = 1.0,
        vector_potential_mobility: float = 1.0,
    ) -> TDGLStepResult:
        step = jnp.asarray(step_size, dtype=state.time.dtype)
        scalar_mobility = float(order_parameter_mobility)
        vector_mobility = float(vector_potential_mobility)
        if (
            step.shape != ()
            or not isfinite(scalar_mobility)
            or scalar_mobility <= 0.0
            or not isfinite(vector_mobility)
            or vector_mobility <= 0.0
        ):
            raise ValueError("TDGL step or mobility is invalid.")
        coordinates = self._encode(state.gauge)
        gradient = jax.grad(self._coordinate_energy)(coordinates, state)
        vertex_count = self.mesh.vertices.shape[0]
        mobility = jnp.concatenate(
            (
                jnp.full((2 * vertex_count,), scalar_mobility),
                jnp.full((self.mesh.topology.edges.shape[0],), vector_mobility),
            )
        )
        candidate_coordinates = coordinates - step * mobility * gradient
        candidate_gauge = self._decode(candidate_coordinates)
        candidate = GinzburgLandauState(
            candidate_gauge,
            state.time + step,
            state.accepted_step + 1,
            self.plan_id,
        )
        initial_energy = self.energy(state)
        candidate_energy = self.energy(candidate)
        finite = (
            initial_energy.finite
            & candidate_energy.finite
            & jnp.all(jnp.isfinite(candidate_coordinates))
        )
        energy_change = candidate_energy.total - initial_energy.total
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & finite
            & (
                energy_change
                <= self.tolerance * jnp.maximum(jnp.abs(initial_energy.total), 1.0)
            )
        )
        accepted = jax.tree.map(
            lambda proposed, prior: jnp.where(successful, proposed, prior),
            candidate,
            state,
        )
        evidence = GinzburgLandauEvidence(
            jnp.linalg.norm(gradient),
            jnp.asarray(0.0, dtype=step.dtype),
            jnp.asarray(0.0, dtype=step.dtype),
            energy_change,
            finite,
            successful,
            successful,
            successful,
            self.plan_id,
        )
        return TDGLStepResult(
            candidate,
            accepted,
            initial_energy,
            candidate_energy,
            evidence,
            successful,
            self.plan_id,
        )


__all__ = [
    "GaugeCovariantGLPlan",
    "GinzburgLandauEnergy",
    "GinzburgLandauEvidence",
    "GinzburgLandauResult",
    "GinzburgLandauState",
    "TDGLStepResult",
]
