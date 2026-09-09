#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....interchange import BoreholeTrajectory
from ._point_electrode import PointElectrodeSurvey


class BoreholeElectrodeArray(StrictModule, NonTrainableState):
    trajectory: BoreholeTrajectory
    measured_depth_m: Array
    positions_m: Array
    array_id: str = eqx.field(static=True)

    def __init__(self, trajectory: BoreholeTrajectory, measured_depth_m: ArrayLike, /):
        if not isinstance(trajectory, BoreholeTrajectory):
            raise TypeError("Borehole electrodes require BoreholeTrajectory.")
        depths = np.asarray(measured_depth_m, dtype=float)
        if depths.ndim != 1 or depths.size == 0 or np.any(~np.isfinite(depths)):
            raise ValueError("Borehole electrode depths must be a finite vector.")
        if np.unique(depths).size != depths.size:
            raise ValueError("Borehole electrode depths must be unique.")
        positions = trajectory.sample_positions(depths)
        self.trajectory = trajectory
        self.measured_depth_m = jnp.asarray(depths)
        self.positions_m = positions
        self.array_id = canonical_fingerprint(
            {
                "kind": "borehole-electrode-array",
                "trajectory": trajectory.trajectory_id,
                "depth_m": depths,
                "positions_m": np.asarray(positions),
            }
        )

    def survey(
        self,
        currents_A: ArrayLike,
        receiver_weights: ArrayLike,
        source_indices: ArrayLike,
        /,
    ) -> PointElectrodeSurvey:
        return PointElectrodeSurvey(
            self.positions_m,
            currents_A,
            receiver_weights,
            source_indices,
        )


def marine_layer_conductivity(
    cell_centers_m: ArrayLike,
    water_surface_elevation_m: float,
    seafloor_elevation_m: ArrayLike,
    water_conductivity_S_m: ArrayLike,
    earth_conductivity_S_m: ArrayLike,
    /,
) -> Array:
    centers = jnp.asarray(cell_centers_m)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Marine conductivity needs three-dimensional cell centers.")
    seafloor = jnp.broadcast_to(jnp.asarray(seafloor_elevation_m), (centers.shape[0],))
    water = jnp.broadcast_to(jnp.asarray(water_conductivity_S_m), (centers.shape[0],))
    earth = jnp.broadcast_to(jnp.asarray(earth_conductivity_S_m), (centers.shape[0],))
    surface = jnp.asarray(water_surface_elevation_m)
    invalid = (
        jnp.any(~jnp.isfinite(centers))
        | jnp.any(~jnp.isfinite(seafloor))
        | ~jnp.isfinite(surface)
        | jnp.any(seafloor > surface)
        | jnp.any(~jnp.isfinite(water))
        | jnp.any(water <= 0)
        | jnp.any(~jnp.isfinite(earth))
        | jnp.any(earth <= 0)
    )
    centers = eqx.error_if(
        centers,
        invalid,
        "Marine geometry and conductivities must be finite and physical.",
    )
    in_water = (centers[:, 2] <= surface) & (centers[:, 2] >= seafloor)
    return jnp.where(in_water, water, earth)


class CasingSolveResult(StrictModule):
    formation_potential: Array
    casing_potential: Array
    leakage_current_A: Array
    residual_norm: Array
    current_balance_A: Array
    successful: Array


class MixedDimensionalCasingPlan(StrictModule, NonTrainableState):
    """Monolithic 3D formation/1D conductive casing coupling.

    ``coupling`` interpolates formation potential to casing stations. Positive
    leakage conductance transfers equal-and-opposite current. Segment axial
    conductance connects consecutive casing stations. Sources may enter either
    formation or casing but their combined current must balance.
    """

    formation_operator: la.AbstractLinearOperator
    coupling: Array
    axial_conductance_S: Array
    leakage_conductance_S: Array
    gauge: Array
    space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formation_operator: la.AbstractLinearOperator,
        coupling: ArrayLike,
        axial_conductance_S: ArrayLike,
        leakage_conductance_S: ArrayLike,
        /,
    ):
        if not isinstance(
            formation_operator, la.AbstractLinearOperator
        ) or not formation_operator.source.compatible(formation_operator.target):
            raise TypeError("Casing formation operator must be a square native operator.")
        matrix = np.asarray(coupling, dtype=float)
        axial = np.asarray(axial_conductance_S, dtype=float)
        leakage = np.asarray(leakage_conductance_S, dtype=float)
        stations = matrix.shape[0] if matrix.ndim == 2 else 0
        if (
            stations < 2
            or matrix.shape[1] != formation_operator.source.size
            or axial.shape != (stations - 1,)
            or leakage.shape != (stations,)
            or np.any(~np.isfinite(matrix))
            or np.any(matrix < 0)
            or not np.allclose(np.sum(matrix, axis=1), 1.0)
            or np.any(~np.isfinite(axial))
            or np.any(axial <= 0)
            or np.any(~np.isfinite(leakage))
            or np.any(leakage <= 0)
        ):
            raise ValueError("Casing coupling, axial, or leakage data are invalid.")
        formation_count = formation_operator.source.size
        gauge = np.concatenate((np.ones(formation_count), np.ones(stations), np.zeros(1)))
        gauge /= np.sqrt(np.sum(gauge**2))
        self.formation_operator = formation_operator
        self.coupling = jnp.asarray(matrix)
        self.axial_conductance_S = jnp.asarray(axial)
        self.leakage_conductance_S = jnp.asarray(leakage)
        self.gauge = jnp.asarray(gauge)
        self.space = la.ArraySpace((formation_count + stations + 1,), dtype=matrix.dtype)
        self.policy = la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mixed-dimensional-casing-plan",
                "formation": formation_operator.operator_id,
                "coupling": matrix,
                "axial_S": axial,
                "leakage_S": leakage,
            }
        )

    def _operator(self):
        formation_count = self.formation_operator.source.size
        stations = self.coupling.shape[0]
        gauge_formation = self.gauge[:formation_count]
        gauge_casing = self.gauge[formation_count : formation_count + stations]

        def action(values):
            formation = values[:formation_count]
            casing = values[formation_count : formation_count + stations]
            multiplier = values[-1]
            sampled = self.coupling @ formation
            leakage = self.leakage_conductance_S * (casing - sampled)
            formation_result = (
                self.formation_operator.mv(formation)
                - self.coupling.T @ leakage
                + multiplier * gauge_formation
            )
            axial_flux = self.axial_conductance_S * jnp.diff(casing)
            casing_result = leakage
            casing_result = casing_result.at[:-1].add(-axial_flux)
            casing_result = casing_result.at[1:].add(axial_flux)
            casing_result = casing_result + multiplier * gauge_casing
            gauge_result = jnp.vdot(gauge_formation, formation) + jnp.vdot(
                gauge_casing, casing
            )
            return jnp.concatenate((formation_result, casing_result, gauge_result[None]))

        return la.FunctionLinearOperator(
            action,
            source=self.space,
            target=self.space,
            properties=la.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=canonical_fingerprint(
                {"kind": "mixed-dimensional-casing-operator", "plan": self.plan_id}
            ),
        )

    def solve(
        self,
        formation_source_A: ArrayLike,
        casing_source_A: ArrayLike,
        /,
    ) -> CasingSolveResult:
        formation_count = self.formation_operator.source.size
        stations = self.coupling.shape[0]
        formation_source = jnp.asarray(formation_source_A)
        casing_source = jnp.asarray(casing_source_A)
        if formation_source.shape != (formation_count,) or casing_source.shape != (
            stations,
        ):
            raise ValueError("Casing source vectors have wrong shape.")
        balance = jnp.sum(formation_source) + jnp.sum(casing_source)
        scale = jnp.sum(jnp.abs(formation_source)) + jnp.sum(jnp.abs(casing_source))
        formation_source = eqx.error_if(
            formation_source,
            jnp.any(~jnp.isfinite(formation_source))
            | jnp.any(~jnp.isfinite(casing_source))
            | (jnp.abs(balance) > 64 * jnp.finfo(formation_source.dtype).eps * scale),
            "Combined formation/casing sources must be finite and balanced.",
        )
        rhs = jnp.concatenate((formation_source, casing_source, jnp.zeros(1)))
        operator = self._operator()
        result = la.solve(la.LinearSystem(operator), rhs, policy=self.policy)
        residual = operator.mv(result.value) - rhs
        formation = result.value[:formation_count]
        casing = result.value[formation_count : formation_count + stations]
        leakage = self.leakage_conductance_S * (casing - self.coupling @ formation)
        norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
        successful = result.successful & jnp.isfinite(norm)
        return CasingSolveResult(
            formation,
            casing,
            leakage,
            norm,
            balance,
            successful,
        )


__all__ = [
    "BoreholeElectrodeArray",
    "CasingSolveResult",
    "MixedDimensionalCasingPlan",
    "marine_layer_conductivity",
]
