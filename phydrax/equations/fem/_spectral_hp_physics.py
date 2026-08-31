#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._mortar import (
    FiniteElementMortarMetricData,
)
from ._conservation import certify_dgsem_mortar_compatibility


class HPOverintegrationPolicy(StrictModule, NonTrainableState):
    solution_degree: tuple[int, ...] = eqx.field(static=True)
    geometry_degree: tuple[int, ...] = eqx.field(static=True)
    nonlinearity_degree: int = eqx.field(static=True)
    quadrature_counts: tuple[int, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        solution_degree: Sequence[int],
        geometry_degree: Sequence[int],
        nonlinearity_degree: int = 2,
        /,
    ):
        solution = tuple(int(value) for value in solution_degree)
        geometry = tuple(int(value) for value in geometry_degree)
        nonlinear = int(nonlinearity_degree)
        if (
            len(solution) != len(geometry)
            or not solution
            or min(solution) < 1
            or min(geometry) < 1
            or nonlinear < 1
        ):
            raise ValueError("Overintegration degrees are invalid.")
        exact_degrees = tuple(
            nonlinear * p + len(solution) * g - 1
            for p, g in zip(solution, geometry, strict=True)
        )
        counts = tuple(max(2, (degree + 2) // 2) for degree in exact_degrees)
        self.solution_degree = solution
        self.geometry_degree = geometry
        self.nonlinearity_degree = nonlinear
        self.quadrature_counts = counts
        self.policy_id = canonical_fingerprint(
            {
                "kind": "hp-overintegration",
                "solution": solution,
                "geometry": geometry,
                "nonlinearity": nonlinear,
                "counts": counts,
            }
        )


class DGSEMCharacteristicBoundaryPlan(StrictModule, NonTrainableState):
    kind: Literal["inflow", "outflow", "slip-wall", "no-slip-wall"] = eqx.field(
        static=True
    )
    wavespeed_floor: float = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["inflow", "outflow", "slip-wall", "no-slip-wall"],
        /,
        *,
        wavespeed_floor: float = 1.0e-12,
    ):
        if (
            kind not in ("inflow", "outflow", "slip-wall", "no-slip-wall")
            or wavespeed_floor <= 0.0
        ):
            raise ValueError("Characteristic boundary kind or floor is invalid.")
        self.kind = kind
        self.wavespeed_floor = float(wavespeed_floor)

    def exterior_state(
        self, interior: ArrayLike, prescribed: ArrayLike, normal: ArrayLike, /
    ) -> Array:
        state = jnp.asarray(interior)
        target = jnp.asarray(prescribed)
        normal_ = jnp.asarray(normal)
        if state.shape != target.shape or state.shape[-1] < normal_.shape[-1] + 1:
            raise ValueError("Boundary state and normal shapes are incompatible.")
        dimension = normal_.shape[-1]
        momentum = state[..., 1 : 1 + dimension]
        normal_momentum = jnp.sum(momentum * normal_, axis=-1, keepdims=True)
        if self.kind == "inflow":
            return target
        if self.kind == "outflow":
            return state.at[..., -1].set(target[..., -1])
        reflected = momentum - 2.0 * normal_momentum * normal_
        if self.kind == "no-slip-wall":
            reflected = -momentum
        return state.at[..., 1 : 1 + dimension].set(reflected)


class EntropyStableWallEvidence(StrictModule):
    mass_flux: Array
    entropy_flux: Array
    passed: Array


def entropy_stable_wall_evidence(
    state: ArrayLike,
    flux: ArrayLike,
    entropy_variables: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> EntropyStableWallEvidence:
    state_ = jnp.asarray(state)
    flux_ = jnp.asarray(flux)
    entropy = jnp.asarray(entropy_variables)
    normal_ = jnp.asarray(normal)
    mass_flux = flux_[..., 0]
    entropy_flux = jnp.sum(entropy * flux_, axis=-1)
    momentum = state_[..., 1 : 1 + normal_.shape[-1]]
    wall_velocity = jnp.sum(momentum * normal_, axis=-1)
    passed = (
        (jnp.max(jnp.abs(mass_flux)) <= tolerance)
        & (jnp.max(jnp.abs(wall_velocity)) <= tolerance)
        & (jnp.max(entropy_flux) <= tolerance)
    )
    return EntropyStableWallEvidence(mass_flux, entropy_flux, passed)


class WellBalancedSourceLedger(StrictModule):
    flux_divergence: Array
    source: Array
    residual: Array
    balance_error: Array

    def __init__(self, flux_divergence: ArrayLike, source: ArrayLike, /):
        divergence = jnp.asarray(flux_divergence)
        source_ = jnp.asarray(source)
        if divergence.shape != source_.shape:
            raise ValueError("Well-balanced flux and source arrays must match.")
        residual = divergence - source_
        self.flux_divergence = divergence
        self.source = source_
        self.residual = residual
        self.balance_error = jnp.max(jnp.abs(residual))


class BR1ViscousPlan(StrictModule, NonTrainableState):
    derivative_matrices: tuple[Array, ...]
    penalty: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, derivative_matrices: Sequence[ArrayLike], /, *, penalty: float = 0.0
    ):
        matrices = tuple(jnp.asarray(value) for value in derivative_matrices)
        if not matrices or any(value.ndim != 2 for value in matrices) or penalty < 0.0:
            raise ValueError("BR1 derivative matrices or penalty are invalid.")
        self.derivative_matrices = matrices
        self.penalty = float(penalty)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "br1-viscous",
                "shapes": [list(value.shape) for value in matrices],
                "penalty": penalty,
            }
        )

    def gradient(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        gradients = [
            oe.contract("ij,...jv->...iv", matrix, value)
            for matrix in self.derivative_matrices
        ]
        return jnp.stack(gradients, axis=-2)

    def divergence(self, viscous_flux: ArrayLike, /) -> Array:
        flux = jnp.asarray(viscous_flux)
        if flux.shape[-2] != len(self.derivative_matrices):
            raise ValueError(
                "BR1 viscous flux lacks one spatial component per derivative."
            )
        result = jnp.zeros(flux.shape[:-2] + (flux.shape[-1],), dtype=flux.dtype)
        for axis, matrix in enumerate(self.derivative_matrices):
            result = result + oe.contract("ji,...jv->...iv", matrix, flux[..., axis, :])
        return result

    def mortar_flux(
        self,
        left_gradient: ArrayLike,
        right_gradient: ArrayLike,
        left_state: ArrayLike,
        right_state: ArrayLike,
        normal: ArrayLike,
        viscosity: ArrayLike,
        /,
    ) -> Array:
        left_g = jnp.asarray(left_gradient)
        right_g = jnp.asarray(right_gradient)
        jump = jnp.asarray(right_state) - jnp.asarray(left_state)
        normal_ = jnp.asarray(normal)
        averaged = 0.5 * (left_g + right_g)
        normal_gradient = oe.contract("...dv,...d->...v", averaged, normal_)
        return jnp.asarray(viscosity)[..., None] * normal_gradient - self.penalty * jump


class SplitFormPolicy(StrictModule, NonTrainableState):
    kind: Literal["entropy", "kinetic-energy", "skew-symmetric"] = eqx.field(static=True)

    def __init__(self, kind: Literal["entropy", "kinetic-energy", "skew-symmetric"], /):
        if kind not in ("entropy", "kinetic-energy", "skew-symmetric"):
            raise ValueError("Unknown DGSEM split form.")
        self.kind = kind

    def combine(
        self,
        conservative: ArrayLike,
        advective: ArrayLike,
        entropy: ArrayLike | None = None,
        /,
    ) -> Array:
        conservative_ = jnp.asarray(conservative)
        advective_ = jnp.asarray(advective)
        if conservative_.shape != advective_.shape:
            raise ValueError("Split-form residuals must have identical shapes.")
        if self.kind == "skew-symmetric":
            return 0.5 * (conservative_ + advective_)
        if self.kind == "kinetic-energy":
            return 0.25 * conservative_ + 0.75 * advective_
        if entropy is None:
            raise ValueError(
                "Entropy split form requires an entropy-conservative residual."
            )
        entropy_ = jnp.asarray(entropy)
        if entropy_.shape != conservative_.shape:
            raise ValueError("Entropy residual shape is incompatible.")
        return entropy_


class TroubledCellEvidence(StrictModule):
    modal_sensor: Array
    entropy_sensor: Array
    pressure_sensor: Array
    troubled: Array
    score: Array

    def __init__(
        self,
        modal_sensor: ArrayLike,
        entropy_sensor: ArrayLike,
        pressure_sensor: ArrayLike,
        /,
        *,
        threshold: float = 1.0,
    ):
        modal = jnp.asarray(modal_sensor)
        entropy = jnp.asarray(entropy_sensor)
        pressure = jnp.asarray(pressure_sensor)
        if (
            modal.shape != entropy.shape
            or modal.shape != pressure.shape
            or threshold <= 0.0
        ):
            raise ValueError(
                "Troubled-cell sensors must have matching shape and positive threshold."
            )
        score = jnp.maximum(modal, jnp.maximum(entropy, pressure))
        self.modal_sensor = modal
        self.entropy_sensor = entropy
        self.pressure_sensor = pressure
        self.troubled = score >= threshold
        self.score = score


class ConservativeModalLimiter(StrictModule, NonTrainableState):
    strength: float = eqx.field(static=True)

    def __init__(self, strength: float = 1.0, /):
        strength_ = float(strength)
        if not 0.0 <= strength_ <= 1.0:
            raise ValueError("Limiter strength must lie in [0, 1].")
        self.strength = strength_

    def apply(self, modal_coefficients: ArrayLike, troubled: ArrayLike, /) -> Array:
        coefficients = jnp.asarray(modal_coefficients)
        mask = jnp.asarray(troubled, dtype=bool)
        if coefficients.shape[0] != mask.shape[0]:
            raise ValueError("Limiter cells and troubled mask disagree.")
        degree = jnp.arange(coefficients.shape[1], dtype=coefficients.real.dtype)
        filter_ = jnp.exp(
            -self.strength * (degree / max(coefficients.shape[1] - 1, 1)) ** 8
        )
        filter_ = filter_.at[0].set(1.0)
        limited = (
            coefficients * filter_[None, :, None]
            if coefficients.ndim == 3
            else coefficients * filter_[None, :]
        )
        shape = mask.shape + (1,) * (coefficients.ndim - 1)
        return jnp.where(mask.reshape(shape), limited, coefficients)


class PositivityLimiter(StrictModule, NonTrainableState):
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)

    def __init__(
        self, density_floor: float = 1.0e-12, pressure_floor: float = 1.0e-12, /
    ):
        if density_floor <= 0.0 or pressure_floor <= 0.0:
            raise ValueError("Positivity floors must be positive.")
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)

    def apply(
        self,
        states: ArrayLike,
        cell_average: ArrayLike,
        pressure: Callable[[Array], Array],
        /,
    ) -> tuple[Array, Array]:
        values = jnp.asarray(states)
        average = jnp.asarray(cell_average)
        if values.shape[0] != average.shape[0] or values.shape[-1] != average.shape[-1]:
            raise ValueError("Positivity states and averages are incompatible.")
        density = values[..., 0]
        theta_density = jnp.minimum(
            1.0,
            (average[..., 0] - self.density_floor)
            / jnp.maximum(average[..., 0] - jnp.min(density, axis=1), 1.0e-30),
        )
        limited = average[:, None, :] + theta_density[:, None, None] * (
            values - average[:, None, :]
        )
        pressure_values = pressure(limited)
        theta_pressure = jnp.minimum(
            1.0,
            (pressure(average) - self.pressure_floor)
            / jnp.maximum(pressure(average) - jnp.min(pressure_values, axis=1), 1.0e-30),
        )
        limited = average[:, None, :] + theta_pressure[:, None, None] * (
            limited - average[:, None, :]
        )
        return limited, jnp.minimum(theta_density, theta_pressure)


class SubcellFiniteVolumePlan(StrictModule, NonTrainableState):
    dg_to_subcell: Array
    subcell_to_dg: Array
    subcell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dg_nodes: ArrayLike, subcell_points: ArrayLike, /):
        dg_nodes_ = np.asarray(dg_nodes)
        subcell = np.asarray(subcell_points)
        interpolation = np.asarray(_tensor_interpolation(dg_nodes_, subcell))
        reconstruction = np.linalg.pinv(interpolation)
        self.dg_to_subcell = jnp.asarray(interpolation)
        self.subcell_to_dg = jnp.asarray(reconstruction)
        self.subcell_count = subcell.shape[0]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "subcell-fv",
                "dg_nodes": list(dg_nodes_.shape),
                "subcells": list(subcell.shape),
            }
        )

    def project(self, dg_values: ArrayLike, /) -> Array:
        return oe.contract("si,...iv->...sv", self.dg_to_subcell, jnp.asarray(dg_values))

    def reconstruct(self, subcell_values: ArrayLike, /) -> Array:
        return oe.contract(
            "is,...sv->...iv", self.subcell_to_dg, jnp.asarray(subcell_values)
        )

    def advance(
        self,
        values: ArrayLike,
        left_flux: ArrayLike,
        right_flux: ArrayLike,
        dt: ArrayLike,
        volumes: ArrayLike,
        /,
    ) -> Array:
        value = jnp.asarray(values)
        left = jnp.asarray(left_flux)
        right = jnp.asarray(right_flux)
        volume = jnp.asarray(volumes)
        if (
            left.shape != value.shape
            or right.shape != value.shape
            or volume.shape != value.shape[:-1]
        ):
            raise ValueError("Subcell values, fluxes, and volumes are incompatible.")
        return value - jnp.asarray(dt) * (right - left) / volume[..., None]


def _tensor_interpolation(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    if points.ndim == 1:
        points = points[:, None]
    axes = tuple(np.unique(nodes[:, axis]) for axis in range(nodes.shape[1]))
    values = []
    for axis_values, coordinates in zip(axes, points.T, strict=True):
        differences = axis_values[:, None] - axis_values[None, :]
        np.fill_diagonal(differences, 1.0)
        barycentric = 1.0 / np.prod(differences, axis=1)
        delta = coordinates[:, None] - axis_values[None, :]
        exact = np.isclose(delta, 0.0)
        raw = barycentric[None, :] / np.where(exact, 1.0, delta)
        basis = raw / np.sum(raw, axis=1, keepdims=True)
        for row in np.flatnonzero(np.any(exact, axis=1)):
            basis[row] = 0.0
            basis[row, np.argmax(exact[row])] = 1.0
        values.append(basis)
    tensor = values[0]
    for basis in values[1:]:
        tensor = (tensor[..., :, None] * basis[..., None, :]).reshape(
            (points.shape[0], -1)
        )
    return tensor


class ALEMetricState(StrictModule):
    coordinates: Array
    mesh_velocity: Array
    jacobian_determinant: Array
    jacobian_rate: Array
    temporal_gcl_defect: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        mesh_velocity: ArrayLike,
        jacobian_determinant: ArrayLike,
        jacobian_rate: ArrayLike,
        metric_flux_divergence: ArrayLike,
        /,
    ):
        coordinates_ = jnp.asarray(coordinates)
        velocity = jnp.asarray(mesh_velocity)
        determinant = jnp.asarray(jacobian_determinant)
        rate = jnp.asarray(jacobian_rate)
        divergence = jnp.asarray(metric_flux_divergence)
        if (
            coordinates_.shape != velocity.shape
            or determinant.shape != rate.shape
            or determinant.shape != divergence.shape
        ):
            raise ValueError("ALE coordinates, velocities, and Jacobian data disagree.")
        self.coordinates = coordinates_
        self.mesh_velocity = velocity
        self.jacobian_determinant = determinant
        self.jacobian_rate = rate
        self.temporal_gcl_defect = rate + divergence


class MovingMortarMetricPlan(StrictModule, NonTrainableState):
    def update(
        self,
        metric: FiniteElementMortarMetricData,
        coordinate_velocity: ArrayLike,
        dt: ArrayLike,
        /,
    ) -> FiniteElementMortarMetricData:
        velocity = jnp.asarray(coordinate_velocity)
        if velocity.shape != metric.physical_coordinates.shape:
            raise ValueError("Moving mortar velocity and coordinates disagree.")
        coordinates = metric.physical_coordinates + jnp.asarray(dt) * velocity
        return FiniteElementMortarMetricData(
            coordinates,
            metric.physical_weights,
            metric.owner_scaled_normals,
            metric.neighbour_scaled_normals,
        )


class LocalTimeSteppingPlan(StrictModule, NonTrainableState):
    level_steps: Array
    level_ratios: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, stable_steps: ArrayLike, levels: ArrayLike, /):
        steps = np.asarray(stable_steps, dtype=float)
        levels_ = np.asarray(levels, dtype=np.int32)
        if steps.shape != levels_.shape or np.any(steps <= 0.0) or np.any(levels_ < 0):
            raise ValueError("Local stable steps and levels are invalid.")
        base = float(np.min(steps * 2.0**levels_))
        ratios = 2**levels_
        self.level_steps = jnp.asarray(base / ratios)
        self.level_ratios = jnp.asarray(ratios)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "local-time-stepping",
                "steps": steps.tolist(),
                "levels": levels_.tolist(),
            }
        )

    def reflux(self, coarse_flux: ArrayLike, fine_fluxes: ArrayLike, /) -> Array:
        coarse = jnp.asarray(coarse_flux)
        fine = jnp.asarray(fine_fluxes)
        if fine.shape[1:] != coarse.shape:
            raise ValueError("Fine flux samples do not match the coarse flux shape.")
        return jnp.sum(fine, axis=0) - coarse


class TemporalHPBudget(StrictModule, NonTrainableState):
    spatial_error: Array
    temporal_error: Array
    algebraic_error: Array
    total_error: Array
    refine_space: Array
    refine_time: Array

    def __init__(
        self,
        spatial_error: ArrayLike,
        temporal_error: ArrayLike,
        algebraic_error: ArrayLike,
        tolerance: float,
        /,
    ):
        spatial = jnp.asarray(spatial_error)
        temporal = jnp.asarray(temporal_error)
        algebraic = jnp.asarray(algebraic_error)
        if (
            spatial.shape != temporal.shape
            or spatial.shape != algebraic.shape
            or tolerance <= 0.0
        ):
            raise ValueError("Space-time error arrays or tolerance are invalid.")
        total = spatial + temporal + algebraic
        self.spatial_error = spatial
        self.temporal_error = temporal
        self.algebraic_error = algebraic
        self.total_error = total
        self.refine_space = (spatial >= temporal) & (total > tolerance)
        self.refine_time = (temporal > spatial) & (total > tolerance)


def derived_mortar_entropy_defect(
    left_state: ArrayLike,
    right_state: ArrayLike,
    left_entropy_variables: ArrayLike,
    right_entropy_variables: ArrayLike,
    numerical_flux: ArrayLike,
    left_entropy_potential: ArrayLike,
    right_entropy_potential: ArrayLike,
    /,
) -> Array:
    left = jnp.asarray(left_state)
    right = jnp.asarray(right_state)
    left_variables = jnp.asarray(left_entropy_variables)
    right_variables = jnp.asarray(right_entropy_variables)
    flux = jnp.asarray(numerical_flux)
    left_potential = jnp.asarray(left_entropy_potential)
    right_potential = jnp.asarray(right_entropy_potential)
    if (
        left.shape != right.shape
        or left.shape != left_variables.shape
        or left.shape != right_variables.shape
        or left.shape != flux.shape
    ):
        raise ValueError("Entropy mortar states, variables, and flux must match.")
    return jnp.sum((right_variables - left_variables) * flux, axis=-1) - (
        right_potential - left_potential
    )


def certify_derived_dgsem_mortar(
    mortar,
    metric: FiniteElementMortarMetricData,
    left_state: ArrayLike,
    right_state: ArrayLike,
    left_entropy_variables: ArrayLike,
    right_entropy_variables: ArrayLike,
    numerical_flux: ArrayLike,
    left_entropy_potential: ArrayLike,
    right_entropy_potential: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
):
    defect = derived_mortar_entropy_defect(
        left_state,
        right_state,
        left_entropy_variables,
        right_entropy_variables,
        numerical_flux,
        left_entropy_potential,
        right_entropy_potential,
    )
    return certify_dgsem_mortar_compatibility(
        mortar,
        metric,
        entropy_error=defect,
        tolerance=tolerance,
    )


__all__ = [
    "ALEMetricState",
    "BR1ViscousPlan",
    "certify_derived_dgsem_mortar",
    "derived_mortar_entropy_defect",
    "ConservativeModalLimiter",
    "DGSEMCharacteristicBoundaryPlan",
    "EntropyStableWallEvidence",
    "HPOverintegrationPolicy",
    "LocalTimeSteppingPlan",
    "MovingMortarMetricPlan",
    "PositivityLimiter",
    "SplitFormPolicy",
    "SubcellFiniteVolumePlan",
    "TemporalHPBudget",
    "TroubledCellEvidence",
    "WellBalancedSourceLedger",
    "entropy_stable_wall_evidence",
]
