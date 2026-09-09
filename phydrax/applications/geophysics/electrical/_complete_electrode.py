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
from ....discretization import TetrahedralConnectivity
from ....units import conversion_factor, derived_unit, METER, OHM, UnitDefinition
from ._finite_patch import DC_CONDUCTIVITY_UNIT, FinitePatchDCPlan, PreparedDC


CONTACT_IMPEDANCE_UNIT = derived_unit("ohm*m^2", ((OHM, 1), (METER, 2)))


class CompleteElectrodeSolveResult(StrictModule):
    potentials: Array
    electrode_potentials: Array
    voltages: Array
    residual_norms: Array
    current_residuals: Array
    gauge_residuals: Array
    dissipated_power_W: Array
    successful: Array


class CompleteElectrodeDCPlan(StrictModule, NonTrainableState):
    """Three-dimensional complete-electrode model with finite contact impedance."""

    finite_patch: FinitePatchDCPlan
    contact_impedance_ohm_m2: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        finite_patch: FinitePatchDCPlan,
        contact_impedance: ArrayLike,
        /,
        *,
        unit: UnitDefinition = CONTACT_IMPEDANCE_UNIT,
    ):
        if not isinstance(finite_patch, FinitePatchDCPlan):
            raise TypeError(
                "Complete electrode model requires FinitePatchDCPlan geometry/survey."
            )
        values = jnp.asarray(contact_impedance) * float(
            conversion_factor(unit, CONTACT_IMPEDANCE_UNIT)
        )
        count = len(finite_patch.survey.patches)
        values = jnp.broadcast_to(values, (count,))
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values <= 0),
            "Contact impedances must be finite and strictly positive.",
        )
        self.finite_patch = finite_patch
        self.contact_impedance_ohm_m2 = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "complete-electrode-dc-plan",
                "finite_patch": finite_patch.plan_id,
                "contact_impedance_ohm_m2": np.asarray(values),
            }
        )

    def prepare(self) -> PreparedCompleteElectrodeDC:
        return PreparedCompleteElectrodeDC(self)


class PreparedCompleteElectrodeDC(StrictModule, NonTrainableState):
    plan: CompleteElectrodeDCPlan
    base: PreparedDC
    mass_electrodes: Array
    mass_rows: Array
    mass_columns: Array
    mass_weights: Array
    integrated_rows: Array
    gauge: Array
    space: la.ArraySpace
    solve_policy: la.LinearSolvePolicy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CompleteElectrodeDCPlan, /):
        if not isinstance(plan, CompleteElectrodeDCPlan):
            raise TypeError("Prepared complete electrode model requires its plan.")
        base = plan.finite_patch.prepare()
        mesh = plan.finite_patch.mesh
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Complete electrode model requires tetrahedral connectivity.")
        faces = np.asarray(connectivity.faces, dtype=np.int32)
        coordinates = np.asarray(mesh.coordinates)
        node_count = coordinates.shape[0]
        electrode_count = len(plan.finite_patch.survey.patches)
        electrode_indices: list[int] = []
        row_indices: list[int] = []
        column_indices: list[int] = []
        weights: list[float] = []
        integrated = np.zeros((electrode_count, node_count), dtype=coordinates.dtype)
        for electrode, patch in enumerate(plan.finite_patch.survey.patches):
            for facet in np.asarray(patch.facet_indices):
                triangle = faces[facet]
                points = coordinates[triangle]
                area = 0.5 * np.sqrt(
                    np.sum(np.cross(points[1] - points[0], points[2] - points[0]) ** 2)
                )
                integrated[electrode, triangle] += area / 3.0
                for local_row, row in enumerate(triangle):
                    for local_column, column in enumerate(triangle):
                        electrode_indices.append(electrode)
                        row_indices.append(int(row))
                        column_indices.append(int(column))
                        weights.append(
                            area / (6.0 if local_row == local_column else 12.0)
                        )
        mass_electrodes = np.asarray(electrode_indices, dtype=np.int32)
        mass_rows = np.asarray(row_indices, dtype=np.int32)
        mass_columns = np.asarray(column_indices, dtype=np.int32)
        mass_weights = np.asarray(weights, dtype=coordinates.dtype)
        unknowns = node_count + electrode_count + 1
        gauge = np.concatenate(
            (np.ones(node_count), np.ones(electrode_count), np.zeros(1))
        )
        gauge /= np.sqrt(np.sum(gauge**2))
        self.plan, self.base = plan, base
        self.mass_electrodes = jnp.asarray(mass_electrodes)
        self.mass_rows = jnp.asarray(mass_rows, dtype=jnp.int32)
        self.mass_columns = jnp.asarray(mass_columns, dtype=jnp.int32)
        self.mass_weights = jnp.asarray(mass_weights, dtype=coordinates.dtype)
        self.integrated_rows = jnp.asarray(integrated)
        self.gauge = jnp.asarray(gauge)
        self.space = la.ArraySpace((unknowns,), dtype=coordinates.dtype)
        self.solve_policy = la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(
                relative=plan.finite_patch.relative_tolerance,
                absolute=plan.finite_patch.absolute_tolerance,
                max_steps=plan.finite_patch.max_steps,
            ),
            failure=la.FailurePolicy("status"),
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-complete-electrode-dc",
                "plan": plan.plan_id,
                "base": base.plan.plan_id,
                "mass_electrodes": mass_electrodes,
                "mass_rows": mass_rows,
                "mass_columns": mass_columns,
                "mass_weights": mass_weights,
            }
        )

    def _operator(self, conductivity: ArrayLike, unit: UnitDefinition):
        bound = self.base.bind_conductivity(conductivity, unit=unit)
        bulk = bound.linear_solve.problem.operator
        node_count = self.base.compiled.state_space.size
        electrode_count = len(self.plan.finite_patch.survey.patches)
        impedance = self.plan.contact_impedance_ohm_m2
        mass_rows = self.mass_rows
        mass_columns = self.mass_columns
        mass_weights = self.mass_weights

        def action(values):
            potential = values[:node_count]
            electrode = values[node_count : node_count + electrode_count]
            gauge_multiplier = values[-1]
            bulk_result = bulk.mv(potential)
            row = mass_rows
            column = mass_columns
            patch = self.mass_electrodes
            contact = (
                mass_weights * (potential[column] - electrode[patch]) / impedance[patch]
            )
            potential_result = bulk_result.at[row].add(contact)
            electrode_result = (
                self.base.electrode_areas * electrode - self.integrated_rows @ potential
            ) / impedance
            primal_gauge = self.gauge[:node_count]
            electrode_gauge = self.gauge[node_count : node_count + electrode_count]
            potential_result = potential_result + gauge_multiplier * primal_gauge
            electrode_result = electrode_result + gauge_multiplier * electrode_gauge
            gauge_result = jnp.vdot(primal_gauge, potential) + jnp.vdot(
                electrode_gauge, electrode
            )
            return jnp.concatenate(
                (potential_result, electrode_result, gauge_result[None])
            )

        return la.FunctionLinearOperator(
            action,
            source=self.space,
            target=self.space,
            properties=la.OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "complete-electrode-kkt-operator",
                    "prepared": self.prepared_id,
                    "conductivity": np.asarray(bound.conductivity),
                }
            ),
        )

    def solve(
        self,
        conductivity: ArrayLike,
        /,
        *,
        unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ) -> CompleteElectrodeSolveResult:
        operator = self._operator(conductivity, unit)
        survey = self.plan.finite_patch.survey
        node_count = self.base.compiled.state_space.size
        electrode_count = len(survey.patches)

        def solve_current(current):
            rhs = jnp.concatenate((jnp.zeros(node_count), current, jnp.zeros(1)))
            result = la.solve(la.LinearSystem(operator), rhs, policy=self.solve_policy)
            residual = operator.mv(result.value) - rhs
            potential = result.value[:node_count]
            electrode = result.value[node_count : node_count + electrode_count]
            gauge_residual = jnp.vdot(self.gauge[:-1], result.value[:-1])
            contact_drop = electrode[:, None] - potential[None, :]
            current_density_integral = (
                self.base.electrode_areas * electrode - self.integrated_rows @ potential
            ) / self.plan.contact_impedance_ohm_m2
            current_defect = current_density_integral - current
            current_residual = jnp.sqrt(
                jnp.real(jnp.vdot(current_defect, current_defect))
            )
            power = jnp.real(jnp.vdot(current, electrode))
            successful = (
                result.successful
                & jnp.all(jnp.isfinite(result.value))
                & (current_residual <= 1e-8 * jnp.maximum(jnp.linalg.norm(current), 1.0))
                & (jnp.abs(gauge_residual) <= 1e-8)
                & jnp.all(jnp.isfinite(contact_drop))
                & jnp.all(jnp.isfinite(current_density_integral))
                & (power >= -1e-10)
            )
            return (
                potential,
                electrode,
                jnp.sqrt(jnp.real(jnp.vdot(residual, residual))),
                current_residual,
                gauge_residual,
                power,
                successful,
            )

        rows = [solve_current(current) for current in survey.currents]
        potentials = jnp.stack([row[0] for row in rows])
        electrodes = jnp.stack([row[1] for row in rows])
        voltages = ein.contract(
            "me,me->m", survey.receiver_weights, electrodes[survey.source_indices]
        )
        return CompleteElectrodeSolveResult(
            potentials,
            electrodes,
            voltages,
            jnp.stack([row[2] for row in rows]),
            jnp.stack([row[3] for row in rows]),
            jnp.stack([row[4] for row in rows]),
            jnp.stack([row[5] for row in rows]),
            jnp.all(jnp.stack([row[6] for row in rows])),
        )

    def predict(
        self,
        conductivity: ArrayLike,
        /,
        *,
        unit: UnitDefinition = DC_CONDUCTIVITY_UNIT,
    ) -> Array:
        result = self.solve(conductivity, unit=unit)
        return eqx.error_if(
            result.voltages,
            ~result.successful,
            "Complete-electrode voltage requires a successful compatible KKT solve.",
        )


__all__ = [
    "CONTACT_IMPEDANCE_UNIT",
    "CompleteElectrodeDCPlan",
    "CompleteElectrodeSolveResult",
    "PreparedCompleteElectrodeDC",
]
