#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ....equations import (
    coefficient,
    CompiledFiniteElementProblem,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
    TensorDiffusionAction,
)
from ....units import AMPERE, convert_value, METER, UnitDefinition
from ._finite_patch import (
    _conductivity_tensor,
    _primal_operator,
    _validate_connected_tetrahedra,
    DC_CONDUCTIVITY_UNIT,
)


class PointElectrodeSurvey(StrictModule, NonTrainableState):
    positions_m: Array
    currents_A: Array
    receiver_weights: Array
    source_indices: Array
    survey_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        currents: ArrayLike,
        receiver_weights: ArrayLike,
        source_indices: ArrayLike,
        /,
        *,
        length_unit: UnitDefinition = METER,
        current_unit: UnitDefinition = AMPERE,
    ):
        positions_ = np.asarray(
            convert_value(positions, source=length_unit, target=METER), dtype=float
        )
        currents_ = np.asarray(
            convert_value(currents, source=current_unit, target=AMPERE), dtype=float
        )
        receivers = np.asarray(receiver_weights, dtype=float)
        indices = np.asarray(source_indices)
        if positions_.ndim != 2 or positions_.shape[1] != 3 or positions_.shape[0] < 4:
            raise ValueError(
                "Point-electrode survey requires at least four 3D electrode positions."
            )
        if (
            currents_.ndim != 2
            or currents_.shape[1] != positions_.shape[0]
            or currents_.shape[0] == 0
        ):
            raise ValueError(
                "Point current matrix must have shape (sources, electrodes)."
            )
        if (
            receivers.ndim != 2
            or receivers.shape[1] != positions_.shape[0]
            or receivers.shape[0] == 0
        ):
            raise ValueError(
                "Point receiver matrix must have shape (measurements, electrodes)."
            )
        if indices.shape != (receivers.shape[0],) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("Each point measurement needs one integer source index.")
        if (
            np.any(~np.isfinite(positions_))
            or np.any(~np.isfinite(currents_))
            or np.any(~np.isfinite(receivers))
            or np.any(indices < 0)
            or np.any(indices >= currents_.shape[0])
        ):
            raise ValueError(
                "Point-electrode survey arrays must be finite and indices valid."
            )
        tolerance = 64 * np.finfo(float).eps
        if np.any(
            np.abs(np.sum(currents_, axis=1))
            > tolerance * np.sum(np.abs(currents_), axis=1)
        ):
            raise ValueError("Every point current pattern must balance exactly.")
        if np.any(
            np.abs(np.sum(receivers, axis=1))
            > tolerance * np.sum(np.abs(receivers), axis=1)
        ):
            raise ValueError(
                "Every point receiver combination must reject the potential gauge."
            )
        for measurement, source in enumerate(indices):
            active = np.abs(currents_[source]) > 0
            if np.any(np.abs(receivers[measurement, active]) > tolerance):
                raise ValueError(
                    "Point receivers cannot evaluate an active singular source electrode."
                )
        self.positions_m = jnp.asarray(positions_)
        self.currents_A = jnp.asarray(currents_)
        self.receiver_weights = jnp.asarray(receivers)
        self.source_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.survey_id = canonical_fingerprint(
            {
                "kind": "point-electrode-survey",
                "positions_m": positions_,
                "currents_A": currents_,
                "receiver_weights": receivers,
                "source_indices": indices,
            }
        )


class PointElectrodeSolveResult(StrictModule):
    total_potentials: Array
    correction_potentials: Array
    electrode_potentials: Array
    electrode_valid: Array
    voltages: Array
    residual_norms: Array
    successful: Array


class PointElectrodeDCPlan(StrictModule, NonTrainableState):
    """Interior point-electrode DC with homogeneous primary singularity subtraction."""

    mesh: CellMesh
    survey: PointElectrodeSurvey
    background_conductivity_S_m: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        survey: PointElectrodeSurvey,
        background_conductivity: float,
        /,
        *,
        conductivity_unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ):
        _validate_connected_tetrahedra(mesh)
        if not isinstance(survey, PointElectrodeSurvey):
            raise TypeError("Point-electrode DC requires PointElectrodeSurvey.")
        background = float(
            convert_value(
                background_conductivity,
                source=conductivity_unit,
                target=DC_CONDUCTIVITY_UNIT,
            )
        )
        if not np.isfinite(background) or background <= 0:
            raise ValueError(
                "Point-electrode background conductivity must be positive finite."
            )
        self.mesh, self.survey = mesh, survey
        self.background_conductivity_S_m = background
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-electrode-dc-plan",
                "mesh": mesh.mesh_id,
                "survey": survey.survey_id,
                "background_conductivity_S_m": background,
                "primary_domain": "infinite-homogeneous-3d",
            }
        )

    def prepare(self) -> PreparedPointElectrodeDC:
        return PreparedPointElectrodeDC(self)


class PreparedPointElectrodeDC(StrictModule, NonTrainableState):
    plan: PointElectrodeDCPlan
    compiled: object
    point_cells: Array
    barycentric: Array
    gauge: Array
    space: la.ArraySpace
    policy: la.LinearSolvePolicy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PointElectrodeDCPlan, /):
        if not isinstance(plan, PointElectrodeDCPlan):
            raise TypeError("Prepared point-electrode model requires its plan.")
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in plan.mesh.blocks]
        )
        coordinates = np.asarray(plan.mesh.coordinates, dtype=float)
        point_cells: list[int] = []
        barycentric: list[np.ndarray] = []
        tolerance = 1e-10
        for point in np.asarray(plan.survey.positions_m):
            candidates = []
            for cell, vertices in enumerate(cells):
                tetrahedron = coordinates[vertices]
                local = np.linalg.solve(
                    (tetrahedron[1:] - tetrahedron[0]).T,
                    point - tetrahedron[0],
                )
                weights = np.concatenate(([1.0 - np.sum(local)], local))
                if np.min(weights) > tolerance and np.max(weights) < 1.0 - tolerance:
                    candidates.append((cell, weights))
            if len(candidates) != 1:
                raise ValueError(
                    "Point electrodes must lie strictly inside exactly one tetrahedron."
                )
            point_cells.append(candidates[0][0])
            barycentric.append(candidates[0][1])
        field = FiniteElementPlan(
            plan.mesh,
            FiniteElementFieldSpec("potential", lagrange_element("tetrahedron", 1)),
        ).prepare()
        tensor = jnp.tile(jnp.eye(3)[None, :, :], (cells.shape[0], 1, 1))
        properties = la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        diffusion = TensorDiffusionAction(
            "potential",
            coefficient(
                tensor,
                location="cell",
                support_id=field.support.support_id,
                entity_set_id=field.cell_domain.entity_set_id,
            ),
            properties=properties,
            action_id="point-dc-cell-conductivity",
        )
        compiled = CompiledFiniteElementProblem(
            FiniteElementForm(
                "point-electrode-dc-conduction",
                "potential",
                (diffusion,),
                properties=properties,
            ),
            field,
            execution_policy=FiniteElementExecutionPolicy(realization="sparse"),
        )
        node_count = coordinates.shape[0]
        gauge = np.ones(node_count)
        gauge /= np.sqrt(np.sum(gauge**2))
        self.plan, self.compiled = plan, compiled
        self.point_cells = jnp.asarray(point_cells, dtype=jnp.int32)
        self.barycentric = jnp.asarray(barycentric)
        self.gauge = jnp.asarray(gauge)
        self.space = la.ArraySpace((node_count + 1,), dtype=coordinates.dtype)
        self.policy = la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-point-electrode-dc", "plan": plan.plan_id}
        )

    def _point_interpolate(self, nodal: Array) -> Array:
        cells = jnp.concatenate(
            [block.vertices for block in self.plan.mesh.blocks], axis=0
        )
        nodes = cells[self.point_cells]
        return ein.contract("ei,ei->e", self.barycentric, nodal[nodes])

    def _point_load(self, currents: Array) -> Array:
        cells = jnp.concatenate(
            [block.vertices for block in self.plan.mesh.blocks], axis=0
        )
        nodes = cells[self.point_cells]
        return (
            jnp.zeros((self.gauge.size,), dtype=currents.dtype)
            .at[nodes]
            .add(self.barycentric * currents[:, None])
        )

    def _primary(
        self,
        locations: Array,
        currents: Array,
        /,
        *,
        allow_source_locations: bool = False,
    ) -> Array:
        displacement = locations[:, None, :] - self.plan.survey.positions_m[None, :, :]
        distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
        active = jnp.abs(currents)[None, :] > 0
        singular = active & (distance <= 0)
        if not allow_source_locations:
            distance = eqx.error_if(
                distance,
                jnp.any(singular),
                "Primary potential is singular at an active point electrode.",
            )
        contributing = active & ~singular
        safe = jnp.where(contributing, distance, 1.0)
        return jnp.sum(jnp.where(contributing, currents[None, :] / safe, 0.0), axis=1) / (
            4.0 * jnp.pi * self.plan.background_conductivity_S_m
        )

    def solve(
        self,
        conductivity: ArrayLike,
        /,
        *,
        unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ) -> PointElectrodeSolveResult:
        count = sum(block.cell_count for block in self.plan.mesh.blocks)
        tensor = _conductivity_tensor(conductivity, count, self.gauge.dtype, unit)
        compiled = eqx.tree_at(
            lambda problem: problem.form.actions[0].diffusivity.value,
            self.compiled,
            tensor,
        )
        bulk = _primal_operator(compiled)
        node_count = self.gauge.size

        def kkt_action(value):
            potential, multiplier = value[:node_count], value[-1]
            return jnp.concatenate(
                (
                    bulk.mv(potential) + multiplier * self.gauge,
                    jnp.vdot(self.gauge, potential)[None],
                )
            )

        operator = la.FunctionLinearOperator(
            kkt_action,
            source=self.space,
            target=self.space,
            properties=la.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "point-electrode-dc-kkt",
                    "prepared": self.prepared_id,
                    "conductivity": np.asarray(tensor),
                }
            ),
        )
        coordinates = self.plan.mesh.coordinates
        electrode_positions = self.plan.survey.positions_m

        def solve_current(current):
            primary_nodes = self._primary(coordinates, current)
            load = self._point_load(current)
            correction_load = load - bulk.mv(primary_nodes)
            rhs = jnp.concatenate((correction_load, jnp.zeros(1)))
            result = la.solve(la.LinearSystem(operator), rhs, policy=self.policy)
            correction = result.value[:node_count]
            total = primary_nodes + correction
            primary_electrodes = self._primary(
                electrode_positions, current, allow_source_locations=True
            )
            electrodes = primary_electrodes + self._point_interpolate(correction)
            electrode_valid = jnp.abs(current) == 0
            residual = bulk.mv(total) - load
            residual = residual - self.gauge * jnp.vdot(self.gauge, residual)
            norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
            successful = result.successful & jnp.isfinite(norm)
            return total, correction, electrodes, electrode_valid, norm, successful

        rows = [solve_current(current) for current in self.plan.survey.currents_A]
        total = jnp.stack([row[0] for row in rows])
        correction = jnp.stack([row[1] for row in rows])
        electrodes = jnp.stack([row[2] for row in rows])
        voltages = ein.contract(
            "me,me->m",
            self.plan.survey.receiver_weights,
            electrodes[self.plan.survey.source_indices],
        )
        return PointElectrodeSolveResult(
            total,
            correction,
            electrodes,
            jnp.stack([row[3] for row in rows]),
            voltages,
            jnp.stack([row[4] for row in rows]),
            jnp.all(jnp.stack([row[5] for row in rows])),
        )

    def predict(
        self, conductivity: ArrayLike, /, *, unit: UnitDefinition = DC_CONDUCTIVITY_UNIT
    ) -> Array:
        result = self.solve(conductivity, unit=unit)
        return eqx.error_if(
            result.voltages,
            ~result.successful,
            "Point-electrode response requires a converged singularity-subtracted solve.",
        )


__all__ = [
    "PointElectrodeDCPlan",
    "PointElectrodeSolveResult",
    "PointElectrodeSurvey",
    "PreparedPointElectrodeDC",
]
