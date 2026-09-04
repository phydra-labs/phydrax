#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ...discretization.finite_volume._mac_boundary import (
    MACBoundaryProvider,
    MACBoundarySide,
)
from ...discretization.spectral._channel import ChannelStokesDiagnostics
from ...equations._channel_les import (
    ChannelLESEnergyLedger,
    CompiledChannelLESDynamics,
)
from ...solver._channel_flow import (
    ChannelSBDF2Method,
    ChannelSBDF2State,
    PreparedChannelSBDF2Method,
)


InflowSpatialMode = Literal["compact", "spectral"]


def _velocity_plus(
    y_plus: Array,
    roughness_plus: Array,
    kappa: float,
    /,
) -> Array:
    """Reichardt composite law with an equivalent-sand-grain roughness shift."""
    smooth = jnp.log1p(kappa * y_plus) / kappa + 7.8 * (
        1.0 - jnp.exp(-y_plus / 11.0) - y_plus / 11.0 * jnp.exp(-y_plus / 3.0)
    )
    roughness_shift = jnp.log1p(roughness_plus) / kappa
    return jnp.maximum(smooth - roughness_shift, 0.0)


def _broadcast_scalar(
    value: ArrayLike, shape: tuple[int, ...], dtype, name: str, /
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if jnp.broadcast_shapes(array.shape, shape) != shape:
        raise ValueError(f"{name} must broadcast to the velocity batch shape {shape}.")
    return jnp.broadcast_to(array, shape)


class VectorEquilibriumWallStressEvidence(StrictModule):
    """Pointwise evidence for the attached equilibrium wall-law solve."""

    normal_norm: Array
    tangency_error: Array
    roughness_ratio: Array
    y_plus: Array
    relative_residual: Array
    input_finite: Array
    geometry_valid: Array
    fluid_properties_valid: Array
    tangential_velocity_valid: Array
    roughness_valid: Array
    bracketed: Array
    law_envelope_valid: Array
    converged: Array
    dissipative: Array
    finite: Array
    successful: Array


class VectorEquilibriumWallStressResult(StrictModule):
    """Wall-on-fluid traction and the evidence required before applying it."""

    traction: Array
    friction_velocity: Array
    wall_shear_magnitude: Array
    residual: Array
    converged: Array
    boundary_power: Array
    finite: Array
    successful: Array
    evidence: VectorEquilibriumWallStressEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class VectorEquilibriumWallStressPlan(StrictModule, NonTrainableState):
    """Attached, zero-pressure-gradient equilibrium wall-stress recipe.

    Molecular viscosity means dynamic viscosity. Roughness is an optional
    equivalent sand-grain height. This plan deliberately has no pressure-
    gradient input and makes no adverse-pressure-gradient or separation claim.
    """

    kappa: float = eqx.field(static=True)
    root_iterations: int = eqx.field(static=True)
    bracket_iterations: int = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    tangency_tolerance: float = eqx.field(static=True)
    minimum_y_plus: float = eqx.field(static=True)
    maximum_y_plus: float = eqx.field(static=True)
    maximum_roughness_ratio: float = eqx.field(static=True)
    support_model: str = eqx.field(static=True)
    pressure_gradient_support: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        kappa: float = 0.41,
        root_iterations: int = 48,
        bracket_iterations: int = 16,
        root_tolerance: float = 1.0e-8,
        tangency_tolerance: float = 1.0e-10,
        y_plus_envelope: tuple[float, float] = (0.0, 1.0e6),
        maximum_roughness_ratio: float = 0.2,
    ):
        kappa_ = float(kappa)
        iterations = int(root_iterations)
        bracket = int(bracket_iterations)
        tolerance = float(root_tolerance)
        tangency = float(tangency_tolerance)
        minimum_y, maximum_y = (float(value) for value in y_plus_envelope)
        roughness_ratio = float(maximum_roughness_ratio)
        if (
            not math.isfinite(kappa_)
            or kappa_ <= 0.0
            or iterations <= 0
            or bracket < 0
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not math.isfinite(tangency)
            or tangency < 0.0
            or not math.isfinite(minimum_y)
            or minimum_y < 0.0
            or not math.isfinite(maximum_y)
            or maximum_y <= minimum_y
            or not math.isfinite(roughness_ratio)
            or roughness_ratio <= 0.0
            or roughness_ratio >= 1.0
        ):
            raise ValueError("Vector equilibrium wall-stress controls are invalid.")
        self.kappa = kappa_
        self.root_iterations = iterations
        self.bracket_iterations = bracket
        self.root_tolerance = tolerance
        self.tangency_tolerance = tangency
        self.minimum_y_plus = minimum_y
        self.maximum_y_plus = maximum_y
        self.maximum_roughness_ratio = roughness_ratio
        self.support_model = (
            "attached incompressible isothermal Newtonian equilibrium boundary layer"
        )
        self.pressure_gradient_support = "none (zero-pressure-gradient law only)"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vector-equilibrium-wall-stress",
                "kappa": kappa_.hex(),
                "root_iterations": iterations,
                "bracket_iterations": bracket,
                "root_tolerance": tolerance.hex(),
                "tangency_tolerance": tangency.hex(),
                "y_plus_envelope": (minimum_y.hex(), maximum_y.hex()),
                "maximum_roughness_ratio": roughness_ratio.hex(),
                "support_model": self.support_model,
                "pressure_gradient_support": self.pressure_gradient_support,
            }
        )

    def prepare(self, spatial_dimension: int, /) -> PreparedVectorEquilibriumWallStress:
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Vector wall stress supports spatial dimension two or three."
            )
        return PreparedVectorEquilibriumWallStress(self, dimension)

    def prepare_channel(
        self,
        dynamics: CompiledChannelLESDynamics,
        step_size: ArrayLike,
        /,
        *,
        density: ArrayLike,
        sample_distance: ArrayLike,
        roughness_height: ArrayLike = 0.0,
        method: ChannelSBDF2Method | None = None,
    ) -> PreparedVectorEquilibriumWallStressChannel:
        """Bind this law as the sole tangential owner of two channel walls."""
        return PreparedVectorEquilibriumWallStressChannel(
            self.prepare(3),
            dynamics,
            step_size,
            density=density,
            sample_distance=sample_distance,
            roughness_height=roughness_height,
            method=method,
        )


class PreparedVectorEquilibriumWallStress(StrictModule, NonTrainableState):
    """Dimension-bound vector wall law with fixed-iteration JAX evaluation."""

    plan: VectorEquilibriumWallStressPlan
    spatial_dimension: int = eqx.field(static=True)
    traction_convention: str = eqx.field(static=True)
    roughness_support: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: VectorEquilibriumWallStressPlan, spatial_dimension: int, /):
        if not isinstance(plan, VectorEquilibriumWallStressPlan):
            raise TypeError("plan must be a VectorEquilibriumWallStressPlan.")
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Vector wall stress supports spatial dimension two or three."
            )
        self.plan = plan
        self.spatial_dimension = dimension
        self.traction_convention = (
            "wall-on-fluid traction; negative boundary power is removal"
        )
        self.roughness_support = (
            "equivalent sand-grain shift with roughness/sample-distance ratio bounded"
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vector-equilibrium-wall-stress",
                "plan": plan.plan_id,
                "spatial_dimension": dimension,
                "traction_convention": self.traction_convention,
                "roughness_support": self.roughness_support,
            }
        )

    def evaluate(
        self,
        tangential_velocity: ArrayLike,
        wall_normal: ArrayLike,
        sample_distance: ArrayLike,
        density: ArrayLike,
        molecular_viscosity: ArrayLike,
        /,
        *,
        roughness_height: ArrayLike = 0.0,
    ) -> VectorEquilibriumWallStressResult:
        velocity = jnp.asarray(tangential_velocity)
        normal = jnp.asarray(wall_normal)
        if velocity.ndim < 1 or velocity.shape[-1] != self.spatial_dimension:
            raise ValueError(
                "tangential_velocity must end in the prepared spatial dimension."
            )
        if normal.shape == (self.spatial_dimension,):
            normal = jnp.broadcast_to(normal, velocity.shape)
        elif normal.shape != velocity.shape:
            raise ValueError(
                "wall_normal must be one vector or match tangential_velocity."
            )
        dtype = jnp.result_type(
            velocity,
            normal,
            sample_distance,
            density,
            molecular_viscosity,
            roughness_height,
            0.0,
        )
        if jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError("Vector wall stress requires real-valued inputs.")
        velocity = velocity.astype(dtype)
        normal = normal.astype(dtype)
        batch_shape = velocity.shape[:-1]
        distance = _broadcast_scalar(
            sample_distance, batch_shape, dtype, "sample_distance"
        )
        density_ = _broadcast_scalar(density, batch_shape, dtype, "density")
        viscosity = _broadcast_scalar(
            molecular_viscosity, batch_shape, dtype, "molecular_viscosity"
        )
        roughness = _broadcast_scalar(
            roughness_height, batch_shape, dtype, "roughness_height"
        )

        vector_finite = jnp.all(jnp.isfinite(velocity), axis=-1) & jnp.all(
            jnp.isfinite(normal), axis=-1
        )
        scalar_finite = (
            jnp.isfinite(distance)
            & jnp.isfinite(density_)
            & jnp.isfinite(viscosity)
            & jnp.isfinite(roughness)
        )
        input_finite = vector_finite & scalar_finite
        clean_velocity = jnp.where(jnp.isfinite(velocity), velocity, 0.0)
        clean_normal = jnp.where(jnp.isfinite(normal), normal, 0.0)
        normal_norm = jnp.sqrt(jnp.sum(clean_normal * clean_normal, axis=-1))
        unit_normal = (
            clean_normal / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[..., None]
        )
        normal_component = jnp.sum(clean_velocity * unit_normal, axis=-1)
        tangent = clean_velocity - normal_component[..., None] * unit_normal
        speed = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
        velocity_norm = jnp.sqrt(jnp.sum(clean_velocity * clean_velocity, axis=-1))
        tangency_error = jnp.abs(normal_component) / jnp.maximum(
            velocity_norm, jnp.finfo(dtype).tiny
        )

        geometry_valid = (distance > 0.0) & (normal_norm > 0.0)
        fluid_valid = (density_ > 0.0) & (viscosity > 0.0)
        tangential_valid = tangency_error <= self.plan.tangency_tolerance
        safe_distance = jnp.where(distance > 0.0, distance, 1.0)
        safe_density = jnp.where(density_ > 0.0, density_, 1.0)
        safe_viscosity = jnp.where(viscosity > 0.0, viscosity, 1.0)
        clean_roughness = jnp.where(jnp.isfinite(roughness), roughness, 0.0)
        roughness_ratio = clean_roughness / safe_distance
        roughness_valid = (roughness >= 0.0) & (
            roughness_ratio <= self.plan.maximum_roughness_ratio
        )
        kinematic_viscosity = safe_viscosity / safe_density

        def prediction(friction_velocity: Array, /) -> Array:
            y_plus_ = safe_distance * friction_velocity / kinematic_viscosity
            roughness_plus = clean_roughness * friction_velocity / kinematic_viscosity
            return friction_velocity * _velocity_plus(
                y_plus_, roughness_plus, self.plan.kappa
            )

        lower = jnp.zeros_like(speed)
        viscous_estimate = jnp.sqrt(
            jnp.maximum(speed * kinematic_viscosity / safe_distance, 0.0)
        )
        upper = jnp.maximum(
            2.0 * jnp.maximum(speed, viscous_estimate),
            kinematic_viscosity / safe_distance,
        )
        for _ in range(self.plan.bracket_iterations):
            upper = jnp.where(prediction(upper) >= speed, upper, 2.0 * upper)
        bracketed = (prediction(upper) >= speed) | (speed == 0.0)
        for _ in range(self.plan.root_iterations):
            middle = 0.5 * (lower + upper)
            choose_upper = prediction(middle) >= speed
            upper = jnp.where(choose_upper, middle, upper)
            lower = jnp.where(choose_upper, lower, middle)
        friction_velocity = jnp.where(speed > 0.0, 0.5 * (lower + upper), 0.0)
        residual = prediction(friction_velocity) - speed
        residual_scale = jnp.maximum(
            jnp.maximum(speed, kinematic_viscosity / safe_distance),
            jnp.finfo(dtype).tiny,
        )
        relative_residual = jnp.abs(residual) / residual_scale
        converged = bracketed & (relative_residual <= self.plan.root_tolerance)
        y_plus = safe_distance * friction_velocity / kinematic_viscosity
        law_envelope_valid = (y_plus >= self.plan.minimum_y_plus) & (
            y_plus <= self.plan.maximum_y_plus
        )
        direction = tangent / jnp.where(speed > 0.0, speed, 1.0)[..., None]
        wall_shear_magnitude = safe_density * friction_velocity**2
        traction = -wall_shear_magnitude[..., None] * direction
        boundary_power = jnp.sum(traction * tangent, axis=-1)
        power_scale = jnp.maximum(wall_shear_magnitude * speed, 1.0)
        dissipative = boundary_power <= self.plan.root_tolerance * power_scale
        finite = (
            input_finite
            & jnp.isfinite(friction_velocity)
            & jnp.isfinite(residual)
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(y_plus)
            & jnp.isfinite(boundary_power)
            & jnp.all(jnp.isfinite(traction), axis=-1)
        )
        successful = (
            finite
            & geometry_valid
            & fluid_valid
            & tangential_valid
            & roughness_valid
            & converged
            & law_envelope_valid
            & dissipative
        )
        evidence = VectorEquilibriumWallStressEvidence(
            normal_norm=normal_norm,
            tangency_error=tangency_error,
            roughness_ratio=roughness_ratio,
            y_plus=y_plus,
            relative_residual=relative_residual,
            input_finite=input_finite,
            geometry_valid=geometry_valid,
            fluid_properties_valid=fluid_valid,
            tangential_velocity_valid=tangential_valid,
            roughness_valid=roughness_valid,
            bracketed=bracketed,
            law_envelope_valid=law_envelope_valid,
            converged=converged,
            dissipative=dissipative,
            finite=finite,
            successful=successful,
        )
        return VectorEquilibriumWallStressResult(
            traction=traction,
            friction_velocity=friction_velocity,
            wall_shear_magnitude=wall_shear_magnitude,
            residual=residual,
            converged=converged,
            boundary_power=boundary_power,
            finite=finite,
            successful=successful,
            evidence=evidence,
            plan_id=self.plan.plan_id,
            prepared_id=self.prepared_id,
        )


def _compact_mass_neutral_basis(
    coordinates: np.ndarray,
    support_radius: float,
    /,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], float]:
    """Build deterministic pair/triple blocks with unit row variance and zero sum."""
    count = coordinates.shape[0]
    remaining = list(range(count))
    groups: list[tuple[int, ...]] = []
    while len(remaining) > 3:
        first = remaining.pop(0)
        distances = np.linalg.norm(coordinates[remaining] - coordinates[first], axis=1)
        partner_position = min(
            range(len(remaining)), key=lambda index: (distances[index], remaining[index])
        )
        second = remaining.pop(partner_position)
        groups.append((first, second))
    groups.append(tuple(remaining))
    if len(groups[-1]) not in (2, 3):
        raise ValueError("Compact mass-neutral synthesis requires at least two nodes.")
    maximum_diameter = 0.0
    for group in groups:
        local = coordinates[np.asarray(group)]
        diameter = float(np.max(np.linalg.norm(local[:, None] - local[None, :], axis=-1)))
        maximum_diameter = max(maximum_diameter, diameter)
        if diameter > support_radius:
            raise ValueError(
                "Deterministic compact pair/triple grouping exceeds compact_support_radius."
            )
    column_count = sum(1 if len(group) == 2 else 2 for group in groups)
    basis = np.zeros((count, column_count), dtype=float)
    column = 0
    for group in groups:
        if len(group) == 2:
            basis[group[0], column] = 1.0
            basis[group[1], column] = -1.0
            column += 1
        else:
            basis[group[0], column] = 1.0
            basis[group[1], column] = -0.5
            basis[group[1], column + 1] = np.sqrt(3.0) / 2.0
            basis[group[2], column] = -0.5
            basis[group[2], column + 1] = -np.sqrt(3.0) / 2.0
            column += 2
    return basis, tuple(groups), maximum_diameter


def _relative_loading_defect(loading: np.ndarray, scale: float, /) -> float:
    if loading.size == 0:
        return 0.0
    return float(np.max(np.abs(loading))) / max(float(scale), np.finfo(float).tiny)


class StochasticTurbulentInflowPreparationEvidence(StrictModule):
    """Analytic covariance and boundary-compatibility evidence for preparation."""

    covariance_symmetry_error: Array
    minimum_covariance_eigenvalue: Array
    covariance_reconstruction_error: Array
    maximum_nodal_covariance_error: Array
    maximum_mass_loading: Array
    maximum_divergence_loading: Array
    finite: Array
    positive_semidefinite: Array
    covariance_exact: Array
    mass_compatible: Array
    divergence_available: Array
    divergence_compatible: Array
    successful: Array


class StochasticTurbulentInflowState(StrictModule):
    """Next-use typed JAX key and exact sample index for restart."""

    key: Array
    sample_index: Array
    prepared_id: str = eqx.field(static=True)


class StochasticTurbulentInflowEvidence(StrictModule):
    """PRNG lineage plus realized mass and represented-divergence evidence."""

    parent_key: Array
    draw_key: Array
    next_key: Array
    sample_index: Array
    fluctuation_volume_flux: Array
    total_volume_flux: Array
    divergence_residual: Array
    maximum_divergence_residual: Array
    covariance_exact: Array
    mass_compatible: Array
    divergence_available: Array
    divergence_compatible: Array
    finite: Array
    successful: Array


class StochasticTurbulentInflowResult(StrictModule):
    """One uncommitted deterministic inflow draw and its successor state."""

    state: StochasticTurbulentInflowState
    velocity: Array
    scalars: Array
    velocity_fluctuation: Array
    scalar_fluctuation: Array
    evidence: StochasticTurbulentInflowEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class StochasticTurbulentInflowPlan(StrictModule, NonTrainableState):
    """Prepare covariance-exact compact or divergence-certified spectral inflow.

    Covariance matrices are never symmetrized, clipped, jittered, or otherwise
    repaired. Compact mode is a nodal pair/triple construction: it certifies
    constant-density boundary flux, and only certifies divergence when a
    caller supplies the exact discrete divergence operator. Spectral mode
    uses caller-declared tangential wavevectors and certifies represented
    surface divergence analytically. Neither mode claims volume divergence
    away from the boundary or variable-density mass compatibility.
    """

    mode: InflowSpatialMode = eqx.field(static=True)
    compact_support_radius: float = eqx.field(static=True)
    covariance_tolerance: float = eqx.field(static=True)
    compatibility_tolerance: float = eqx.field(static=True)
    maximum_preparation_bytes: int = eqx.field(static=True)
    mass_support: str = eqx.field(static=True)
    divergence_support: str = eqx.field(static=True)
    distribution_support: str = eqx.field(static=True)
    temporal_support: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: InflowSpatialMode,
        /,
        *,
        compact_support_radius: float = 1.0,
        covariance_tolerance: float = 1.0e-10,
        compatibility_tolerance: float = 1.0e-10,
        maximum_preparation_bytes: int = 256 * 1024 * 1024,
    ):
        if mode not in ("compact", "spectral"):
            raise ValueError("Inflow mode must be 'compact' or 'spectral'.")
        radius = float(compact_support_radius)
        covariance_error = float(covariance_tolerance)
        compatibility = float(compatibility_tolerance)
        maximum_bytes = int(maximum_preparation_bytes)
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(covariance_error)
            or covariance_error <= 0.0
            or not math.isfinite(compatibility)
            or compatibility <= 0.0
            or maximum_bytes <= 0
        ):
            raise ValueError("Stochastic inflow controls are invalid.")
        self.mode = mode
        self.compact_support_radius = radius
        self.covariance_tolerance = covariance_error
        self.compatibility_tolerance = compatibility
        self.maximum_preparation_bytes = maximum_bytes
        self.mass_support = "constant-density quadrature volume flux of fluctuations"
        self.divergence_support = (
            "supplied discrete boundary operator"
            if mode == "compact"
            else "analytic represented surface divergence; no normal derivative claim"
        )
        self.distribution_support = (
            "zero-mean jointly Gaussian velocity/scalar fluctuations"
        )
        self.temporal_support = (
            "independent draws per accepted state advance; no temporal-correlation claim"
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stochastic-turbulent-inflow",
                "mode": mode,
                "compact_support_radius": radius.hex(),
                "covariance_tolerance": covariance_error.hex(),
                "compatibility_tolerance": compatibility.hex(),
                "maximum_preparation_bytes": maximum_bytes,
                "mass_support": self.mass_support,
                "divergence_support": self.divergence_support,
                "distribution_support": self.distribution_support,
                "temporal_support": self.temporal_support,
                "covariance_repair": "none",
                "randomness": "explicit-next-use-typed-jax-key",
            }
        )

    def prepare(
        self,
        coordinates: ArrayLike,
        wall_normal: ArrayLike,
        quadrature_weights: ArrayLike,
        velocity_covariance: ArrayLike,
        /,
        *,
        scalar_covariance: ArrayLike | None = None,
        velocity_scalar_covariance: ArrayLike | None = None,
        spectral_wavevectors: ArrayLike | None = None,
        divergence_operator: ArrayLike | None = None,
    ) -> PreparedStochasticTurbulentInflow:
        points = np.asarray(coordinates, dtype=float)
        normals = np.asarray(wall_normal, dtype=float)
        weights = np.asarray(quadrature_weights, dtype=float)
        velocity_covariance_ = np.asarray(velocity_covariance)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                "Inflow coordinates must be an N-by-2 or N-by-3 matrix, N >= 2."
            )
        count, dimension = points.shape
        if normals.shape == (dimension,):
            normals = np.broadcast_to(normals, points.shape).copy()
        if normals.shape != points.shape or weights.shape != (count,):
            raise ValueError("Inflow normals/weights do not match boundary coordinates.")
        if (
            np.any(~np.isfinite(points))
            or np.any(~np.isfinite(normals))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Inflow geometry must be finite with positive weights.")
        normal_norm = np.linalg.norm(normals, axis=-1)
        if np.any(normal_norm <= 0.0):
            raise ValueError("Every inflow normal must be nonzero.")
        unit_normals = normals / normal_norm[:, None]
        if velocity_covariance_.shape != (dimension, dimension) or np.iscomplexobj(
            velocity_covariance_
        ):
            raise ValueError("velocity_covariance must be a real spatial square matrix.")
        velocity_covariance_ = np.asarray(velocity_covariance_, dtype=float)
        if scalar_covariance is None:
            scalar_covariance_ = np.zeros((0, 0), dtype=float)
        else:
            scalar_covariance_ = np.asarray(scalar_covariance)
            if (
                scalar_covariance_.ndim != 2
                or scalar_covariance_.shape[0] != scalar_covariance_.shape[1]
                or np.iscomplexobj(scalar_covariance_)
            ):
                raise ValueError("scalar_covariance must be a real square matrix.")
            scalar_covariance_ = np.asarray(scalar_covariance_, dtype=float)
        scalar_count = scalar_covariance_.shape[0]
        if velocity_scalar_covariance is None:
            cross = np.zeros((dimension, scalar_count), dtype=float)
        else:
            cross = np.asarray(velocity_scalar_covariance)
            if cross.shape != (dimension, scalar_count) or np.iscomplexobj(cross):
                raise ValueError(
                    "velocity_scalar_covariance must have shape (dimension, scalar_count)."
                )
            cross = np.asarray(cross, dtype=float)
        joint_covariance = np.block(
            [[velocity_covariance_, cross], [cross.T, scalar_covariance_]]
        )
        if np.any(~np.isfinite(joint_covariance)):
            raise ValueError("The prescribed velocity/scalar covariance must be finite.")
        symmetry_error = float(np.max(np.abs(joint_covariance - joint_covariance.T)))
        if not np.array_equal(joint_covariance, joint_covariance.T):
            raise ValueError(
                "The prescribed covariance must be exactly symmetric; no repair is applied."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(joint_covariance)
        minimum_eigenvalue = float(np.min(eigenvalues))
        if minimum_eigenvalue < 0.0:
            raise ValueError(
                "The prescribed covariance must be positive semidefinite; no repair is applied."
            )
        positive = eigenvalues > 0.0
        covariance_root = (
            eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])[None, :]
        )
        covariance_rank = covariance_root.shape[1]
        reconstructed = covariance_root @ covariance_root.T
        covariance_scale = max(float(np.max(np.abs(joint_covariance))), 1.0)
        reconstruction_error = float(np.max(np.abs(reconstructed - joint_covariance)))
        if reconstruction_error > self.covariance_tolerance * covariance_scale:
            raise ValueError(
                "The prescribed covariance factorization is not accurate enough."
            )

        compact_groups: tuple[tuple[int, ...], ...] = ()
        compact_diameter = 0.0
        if self.mode == "compact":
            if spectral_wavevectors is not None:
                raise ValueError("spectral_wavevectors are only valid in spectral mode.")
            spatial_basis, compact_groups, compact_diameter = _compact_mass_neutral_basis(
                points, self.compact_support_radius
            )
            synthesis = np.einsum(
                "nm,ca->ncma", spatial_basis, covariance_root, optimize=True
            ).reshape((count, dimension + scalar_count, -1))
            wavevectors = np.zeros((0, dimension), dtype=float)
        else:
            if spectral_wavevectors is None:
                if covariance_rank:
                    raise ValueError(
                        "spectral_wavevectors must provide one wavevector per covariance mode."
                    )
                wavevectors = np.zeros((0, dimension), dtype=float)
            else:
                wavevectors = np.asarray(spectral_wavevectors, dtype=float)
            if wavevectors.shape != (covariance_rank, dimension) or np.any(
                ~np.isfinite(wavevectors)
            ):
                raise ValueError(
                    "spectral_wavevectors must have shape (covariance_rank, dimension)."
                )
            if covariance_rank:
                wave_norm = np.linalg.norm(wavevectors, axis=-1)
                if np.any(wave_norm <= 0.0):
                    raise ValueError("Every active spectral wavevector must be nonzero.")
                normal_wave = np.einsum(
                    "rd,nd->rn", wavevectors, unit_normals, optimize=True
                )
                if (
                    np.max(np.abs(normal_wave) / wave_norm[:, None])
                    > self.compatibility_tolerance
                ):
                    raise ValueError(
                        "Spectral wavevectors must be tangent to the boundary."
                    )
            synthesis = np.zeros(
                (count, dimension + scalar_count, 2 * covariance_rank), dtype=float
            )
            for mode_index in range(covariance_rank):
                phase = points @ wavevectors[mode_index]
                synthesis[:, :, 2 * mode_index] = (
                    np.cos(phase)[:, None] * covariance_root[:, mode_index][None, :]
                )
                synthesis[:, :, 2 * mode_index + 1] = (
                    np.sin(phase)[:, None] * covariance_root[:, mode_index][None, :]
                )

        nodal_covariance = np.einsum("ncl,ndl->ncd", synthesis, synthesis, optimize=True)
        nodal_error = float(
            np.max(np.abs(nodal_covariance - joint_covariance[None, :, :]))
        )
        covariance_exact = nodal_error <= self.covariance_tolerance * covariance_scale
        if not covariance_exact:
            raise ValueError(
                "Prepared inflow does not reproduce the prescribed covariance."
            )
        mass_loading = np.einsum(
            "n,nd,ndl->l",
            weights,
            unit_normals,
            synthesis[:, :dimension, :],
            optimize=True,
        )
        synthesis_scale = float(np.max(np.abs(synthesis), initial=0.0))
        mass_scale = float(np.sum(np.abs(weights))) * synthesis_scale
        mass_defect = _relative_loading_defect(mass_loading, mass_scale)
        mass_compatible = mass_defect <= self.compatibility_tolerance
        if not mass_compatible:
            raise ValueError(
                "Prepared inflow modes are not compatible with boundary volume flux."
            )

        divergence_blocks: list[np.ndarray] = []
        divergence_kind = "unavailable"
        if self.mode == "spectral":
            analytic = np.zeros((count, synthesis.shape[-1]), dtype=float)
            for mode_index in range(covariance_rank):
                phase = points @ wavevectors[mode_index]
                coefficient = float(
                    np.dot(
                        wavevectors[mode_index],
                        covariance_root[:dimension, mode_index],
                    )
                )
                analytic[:, 2 * mode_index] = -np.sin(phase) * coefficient
                analytic[:, 2 * mode_index + 1] = np.cos(phase) * coefficient
            divergence_blocks.append(analytic)
            divergence_kind = "analytic-represented-surface"
        operator = np.zeros((0, count * dimension), dtype=float)
        if divergence_operator is not None:
            operator_ = np.asarray(divergence_operator, dtype=float)
            if operator_.ndim == 3 and operator_.shape[1:] == (count, dimension):
                operator_ = operator_.reshape((operator_.shape[0], count * dimension))
            if (
                operator_.ndim != 2
                or operator_.shape[1] != count * dimension
                or np.any(~np.isfinite(operator_))
            ):
                raise ValueError(
                    "divergence_operator must map flattened boundary velocity values."
                )
            operator = operator_
            discrete = operator @ synthesis[:, :dimension, :].reshape(
                (count * dimension, synthesis.shape[-1])
            )
            divergence_blocks.append(discrete)
            divergence_kind = (
                "supplied-discrete"
                if divergence_kind == "unavailable"
                else f"{divergence_kind}+supplied-discrete"
            )
        divergence_loading = (
            np.concatenate(divergence_blocks, axis=0)
            if divergence_blocks
            else np.zeros((0, synthesis.shape[-1]), dtype=float)
        )
        divergence_available = bool(divergence_blocks)
        divergence_scale = max(synthesis_scale, np.finfo(float).tiny)
        if self.mode == "spectral" and covariance_rank:
            divergence_scale *= max(float(np.max(np.abs(wavevectors))), 1.0)
        if operator.size:
            divergence_scale *= max(float(np.max(np.abs(operator))), 1.0)
        divergence_defect = _relative_loading_defect(divergence_loading, divergence_scale)
        divergence_compatible = (
            divergence_available and divergence_defect <= self.compatibility_tolerance
        )
        if divergence_available and not divergence_compatible:
            raise ValueError(
                "Prepared inflow modes are incompatible with represented divergence."
            )

        preparation_bytes = int(
            points.nbytes
            + unit_normals.nbytes
            + weights.nbytes
            + joint_covariance.nbytes
            + covariance_root.nbytes
            + synthesis.nbytes
            + wavevectors.nbytes
            + operator.nbytes
            + mass_loading.nbytes
            + divergence_loading.nbytes
        )
        if preparation_bytes > self.maximum_preparation_bytes:
            raise ValueError(
                "Stochastic inflow preparation exceeds maximum_preparation_bytes."
            )
        finite = bool(
            np.all(np.isfinite(covariance_root))
            and np.all(np.isfinite(synthesis))
            and np.all(np.isfinite(mass_loading))
            and np.all(np.isfinite(divergence_loading))
        )
        preparation = StochasticTurbulentInflowPreparationEvidence(
            covariance_symmetry_error=jnp.asarray(symmetry_error),
            minimum_covariance_eigenvalue=jnp.asarray(minimum_eigenvalue),
            covariance_reconstruction_error=jnp.asarray(reconstruction_error),
            maximum_nodal_covariance_error=jnp.asarray(nodal_error),
            maximum_mass_loading=jnp.asarray(np.max(np.abs(mass_loading), initial=0.0)),
            maximum_divergence_loading=jnp.asarray(
                np.max(np.abs(divergence_loading), initial=0.0)
            ),
            finite=jnp.asarray(finite),
            positive_semidefinite=jnp.asarray(True),
            covariance_exact=jnp.asarray(covariance_exact),
            mass_compatible=jnp.asarray(mass_compatible),
            divergence_available=jnp.asarray(divergence_available),
            divergence_compatible=jnp.asarray(divergence_compatible),
            successful=jnp.asarray(
                finite
                and covariance_exact
                and mass_compatible
                and (not divergence_available or divergence_compatible)
            ),
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-stochastic-turbulent-inflow",
                "plan": self.plan_id,
                "coordinates": array_tree_fingerprint(points),
                "unit_normals": array_tree_fingerprint(unit_normals),
                "quadrature_weights": array_tree_fingerprint(weights),
                "joint_covariance": array_tree_fingerprint(joint_covariance),
                "covariance_root": array_tree_fingerprint(covariance_root),
                "synthesis": array_tree_fingerprint(synthesis),
                "spectral_wavevectors": array_tree_fingerprint(wavevectors),
                "divergence_operator": array_tree_fingerprint(operator),
                "compact_groups": compact_groups,
                "compact_maximum_diameter": compact_diameter,
                "divergence_kind": divergence_kind,
                "covariance_repair": "none",
            }
        )
        return PreparedStochasticTurbulentInflow(
            plan=self,
            coordinates=jnp.asarray(points),
            unit_normals=jnp.asarray(unit_normals),
            quadrature_weights=jnp.asarray(weights),
            joint_covariance=jnp.asarray(joint_covariance),
            covariance_root=jnp.asarray(covariance_root),
            synthesis=jnp.asarray(synthesis),
            spectral_wavevectors=jnp.asarray(wavevectors),
            divergence_operator=jnp.asarray(operator),
            mass_loading=jnp.asarray(mass_loading),
            divergence_loading=jnp.asarray(divergence_loading),
            preparation=preparation,
            spatial_dimension=dimension,
            scalar_count=scalar_count,
            covariance_rank=covariance_rank,
            latent_dimension=synthesis.shape[-1],
            compact_groups=compact_groups,
            compact_maximum_diameter=compact_diameter,
            divergence_kind=divergence_kind,
            preparation_bytes=preparation_bytes,
            prepared_id=prepared_id,
        )

    def prepare_mac_boundary(
        self,
        coordinates: ArrayLike,
        wall_normal: ArrayLike,
        quadrature_weights: ArrayLike,
        velocity_covariance: ArrayLike,
        /,
        *,
        axis: str,
        side: Literal["lower", "upper"],
        boundary_shape: tuple[int, ...],
        scalar_covariance: ArrayLike | None = None,
        velocity_scalar_covariance: ArrayLike | None = None,
        spectral_wavevectors: ArrayLike | None = None,
        divergence_operator: ArrayLike | None = None,
    ) -> PreparedStochasticTurbulentInflowMACBoundary:
        """Prepare an accepted-step owner for one structured-MAC velocity inflow."""
        inflow = self.prepare(
            coordinates,
            wall_normal,
            quadrature_weights,
            velocity_covariance,
            scalar_covariance=scalar_covariance,
            velocity_scalar_covariance=velocity_scalar_covariance,
            spectral_wavevectors=spectral_wavevectors,
            divergence_operator=divergence_operator,
        )
        return PreparedStochasticTurbulentInflowMACBoundary(
            inflow,
            axis,
            side,
            boundary_shape,
        )


class PreparedStochasticTurbulentInflow(StrictModule, NonTrainableState):
    """Immutable synthesis operator with explicit-key, exact-restart sampling."""

    plan: StochasticTurbulentInflowPlan
    coordinates: Array
    unit_normals: Array
    quadrature_weights: Array
    joint_covariance: Array
    covariance_root: Array
    synthesis: Array
    spectral_wavevectors: Array
    divergence_operator: Array
    mass_loading: Array
    divergence_loading: Array
    preparation: StochasticTurbulentInflowPreparationEvidence
    spatial_dimension: int = eqx.field(static=True)
    scalar_count: int = eqx.field(static=True)
    covariance_rank: int = eqx.field(static=True)
    latent_dimension: int = eqx.field(static=True)
    compact_groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    compact_maximum_diameter: float = eqx.field(static=True)
    divergence_kind: str = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self,
        key: ArrayLike,
        /,
        *,
        sample_index: int = 0,
    ) -> StochasticTurbulentInflowState:
        key_ = jnp.asarray(key)
        reference = jr.key(0)
        if key_.shape != reference.shape or key_.dtype != reference.dtype:
            raise ValueError("key must be one typed JAX PRNG key.")
        index = int(sample_index)
        if index < 0 or index > np.iinfo(np.uint32).max:
            raise ValueError("sample_index must fit a nonnegative uint32.")
        return StochasticTurbulentInflowState(
            key=key_,
            sample_index=jnp.asarray(index, dtype=jnp.uint32),
            prepared_id=self.prepared_id,
        )

    def _validate_state(self, state: StochasticTurbulentInflowState, /) -> None:
        if not isinstance(state, StochasticTurbulentInflowState):
            raise TypeError("state must be a StochasticTurbulentInflowState.")
        reference = jr.key(0)
        if state.prepared_id != self.prepared_id:
            raise ValueError("Inflow state belongs to another prepared plan.")
        if state.key.shape != reference.shape or state.key.dtype != reference.dtype:
            raise ValueError("Inflow state does not contain one typed JAX PRNG key.")
        if state.sample_index.shape:
            raise ValueError("Inflow state sample_index must be scalar.")

    def sample(
        self,
        state: StochasticTurbulentInflowState,
        /,
        *,
        mean_velocity: ArrayLike | None = None,
        mean_scalars: ArrayLike | None = None,
    ) -> StochasticTurbulentInflowResult:
        self._validate_state(state)
        count = self.coordinates.shape[0]
        dtype = self.synthesis.dtype
        if mean_velocity is None:
            velocity_mean = jnp.zeros((count, self.spatial_dimension), dtype=dtype)
        else:
            velocity_mean = jnp.asarray(mean_velocity, dtype=dtype)
            if velocity_mean.shape == (self.spatial_dimension,):
                velocity_mean = jnp.broadcast_to(
                    velocity_mean, (count, self.spatial_dimension)
                )
            elif velocity_mean.shape != (count, self.spatial_dimension):
                raise ValueError(
                    "mean_velocity must be one vector or one vector per boundary node."
                )
        if mean_scalars is None:
            scalar_mean = jnp.zeros((count, self.scalar_count), dtype=dtype)
        else:
            scalar_mean = jnp.asarray(mean_scalars, dtype=dtype)
            if scalar_mean.shape == (self.scalar_count,):
                scalar_mean = jnp.broadcast_to(scalar_mean, (count, self.scalar_count))
            elif scalar_mean.shape != (count, self.scalar_count):
                raise ValueError(
                    "mean_scalars must be one scalar vector or one per boundary node."
                )
        next_key, draw_key = jr.split(state.key)
        standard = jr.normal(draw_key, (self.latent_dimension,), dtype=dtype)
        fluctuation = ein.contract("ncl,l->nc", self.synthesis, standard, backend="jax")
        velocity_fluctuation = fluctuation[:, : self.spatial_dimension]
        scalar_fluctuation = fluctuation[:, self.spatial_dimension :]
        velocity = velocity_mean + velocity_fluctuation
        scalars = scalar_mean + scalar_fluctuation
        fluctuation_flux = ein.contract(
            "n,nd,nd->",
            self.quadrature_weights,
            self.unit_normals,
            velocity_fluctuation,
            backend="jax",
        )
        total_flux = ein.contract(
            "n,nd,nd->",
            self.quadrature_weights,
            self.unit_normals,
            velocity,
            backend="jax",
        )
        divergence_residual = self.divergence_loading @ standard
        maximum_divergence = jnp.max(
            jnp.abs(divergence_residual), initial=jnp.asarray(0.0, dtype=dtype)
        )
        flux_scale = jnp.maximum(
            jnp.sum(jnp.abs(self.quadrature_weights))
            * jnp.max(jnp.abs(velocity_fluctuation), initial=0.0),
            1.0,
        )
        mass_compatible = jnp.asarray(self.preparation.mass_compatible) & (
            jnp.abs(fluctuation_flux) <= self.plan.compatibility_tolerance * flux_scale
        )
        divergence_available = jnp.asarray(self.preparation.divergence_available)
        divergence_scale = jnp.maximum(
            jnp.max(jnp.abs(velocity_fluctuation), initial=0.0), 1.0
        )
        divergence_compatible = (
            divergence_available
            & jnp.asarray(self.preparation.divergence_compatible)
            & (maximum_divergence <= self.plan.compatibility_tolerance * divergence_scale)
        )
        finite = (
            jnp.all(jnp.isfinite(standard))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(scalars))
            & jnp.isfinite(fluctuation_flux)
            & jnp.isfinite(total_flux)
            & jnp.all(jnp.isfinite(divergence_residual))
        )
        successful = (
            finite
            & self.preparation.successful
            & mass_compatible
            & (~divergence_available | divergence_compatible)
        )
        next_state = StochasticTurbulentInflowState(
            key=next_key,
            sample_index=state.sample_index + jnp.asarray(1, dtype=jnp.uint32),
            prepared_id=self.prepared_id,
        )
        evidence = StochasticTurbulentInflowEvidence(
            parent_key=state.key,
            draw_key=draw_key,
            next_key=next_key,
            sample_index=state.sample_index,
            fluctuation_volume_flux=fluctuation_flux,
            total_volume_flux=total_flux,
            divergence_residual=divergence_residual,
            maximum_divergence_residual=maximum_divergence,
            covariance_exact=self.preparation.covariance_exact,
            mass_compatible=mass_compatible,
            divergence_available=divergence_available,
            divergence_compatible=divergence_compatible,
            finite=finite,
            successful=successful,
        )
        return StochasticTurbulentInflowResult(
            state=next_state,
            velocity=velocity,
            scalars=scalars,
            velocity_fluctuation=velocity_fluctuation,
            scalar_fluctuation=scalar_fluctuation,
            evidence=evidence,
            plan_id=self.plan.plan_id,
            prepared_id=self.prepared_id,
        )


CHANNEL_WALL_STRESS_FAILURE = -4


class VectorEquilibriumWallStressChannelState(StrictModule):
    """Complete SBDF2 and current wall traction for exact channel restart."""

    channel: ChannelSBDF2State
    current_lower: VectorEquilibriumWallStressResult
    current_upper: VectorEquilibriumWallStressResult
    prepared_id: str = eqx.field(static=True)


class VectorEquilibriumWallStressChannelEvidence(StrictModule):
    """Applied traction, work, and boundary-identity evidence for one step."""

    lower_wall_stress: VectorEquilibriumWallStressResult
    upper_wall_stress: VectorEquilibriumWallStressResult
    applied_lower_specific_traction: Array
    applied_upper_specific_traction: Array
    energy_ledger: ChannelLESEnergyLedger
    stokes: ChannelStokesDiagnostics
    energy_boundary_work_defect: Array
    finite: Array
    wall_law_successful: Array
    dissipative: Array
    boundary_identity_closed: Array
    successful: Array


class VectorEquilibriumWallStressChannelResult(StrictModule):
    """Atomic channel candidate/acceptance with wall-law evidence."""

    candidate_state: VectorEquilibriumWallStressChannelState
    accepted_state: VectorEquilibriumWallStressChannelState
    evidence: VectorEquilibriumWallStressChannelEvidence
    successful: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


class PreparedVectorEquilibriumWallStressChannel(StrictModule, NonTrainableState):
    """Channel SBDF2 with normal essential constraints and wall-law traction."""

    wall_stress: PreparedVectorEquilibriumWallStress
    dynamics: CompiledChannelLESDynamics
    integrator: PreparedChannelSBDF2Method
    density: Array
    sample_distances: Array
    sample_coordinates: Array
    sample_evaluation: Array
    roughness_heights: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        wall_stress: PreparedVectorEquilibriumWallStress,
        dynamics: CompiledChannelLESDynamics,
        step_size: ArrayLike,
        /,
        *,
        density: ArrayLike,
        sample_distance: ArrayLike,
        roughness_height: ArrayLike = 0.0,
        method: ChannelSBDF2Method | None = None,
    ):
        if not isinstance(wall_stress, PreparedVectorEquilibriumWallStress):
            raise TypeError("wall_stress must be a PreparedVectorEquilibriumWallStress.")
        if wall_stress.spatial_dimension != 3:
            raise ValueError("Channel wall stress must be prepared in three dimensions.")
        if not isinstance(dynamics, CompiledChannelLESDynamics):
            raise TypeError("Channel wall stress requires CompiledChannelLESDynamics.")
        stokes = dynamics.stokes_plan
        if stokes.tangential_boundary != "traction":
            raise ValueError(
                "Channel wall stress requires traction-owned tangential Stokes walls."
            )
        if stokes.mean_constraint.kind != "pressure_gradient" or not bool(
            jnp.array_equal(
                stokes.mean_constraint.values,
                jnp.zeros((2,), dtype=stokes.mean_constraint.values.dtype),
            )
        ):
            raise ValueError(
                "Equilibrium channel wall stress supports only exactly zero prescribed "
                "pressure gradient."
            )
        if not bool(
            jnp.array_equal(
                stokes.lower_wall_velocity,
                jnp.zeros((3,), dtype=stokes.lower_wall_velocity.dtype),
            )
            & jnp.array_equal(
                stokes.upper_wall_velocity,
                jnp.zeros((3,), dtype=stokes.upper_wall_velocity.dtype),
            )
        ):
            raise ValueError(
                "Equilibrium channel wall stress requires stationary impermeable walls."
            )
        density_ = np.asarray(density)
        distances = np.asarray(sample_distance)
        roughness = np.asarray(roughness_height)
        if distances.shape == ():
            distances = np.broadcast_to(distances, (2,))
        if roughness.shape == ():
            roughness = np.broadcast_to(roughness, (2,))
        wall_axis = dynamics.discretization.axes[1]
        wall_bounds = np.asarray(wall_axis.bounds, dtype=float)
        wall_length = float(wall_bounds[1] - wall_bounds[0])
        if (
            density_.shape != ()
            or not np.isfinite(density_)
            or float(density_) <= 0.0
            or distances.shape != (2,)
            or np.any(~np.isfinite(distances))
            or np.any(distances <= 0.0)
            or np.any(distances >= wall_length)
            or roughness.shape != (2,)
            or np.any(~np.isfinite(roughness))
            or np.any(roughness < 0.0)
            or np.any(roughness / distances > wall_stress.plan.maximum_roughness_ratio)
        ):
            raise ValueError(
                "Channel wall density, sample distances, and roughness are invalid."
            )
        selected = ChannelSBDF2Method() if method is None else method
        if not isinstance(selected, ChannelSBDF2Method):
            raise TypeError("method must be ChannelSBDF2Method or None.")
        integrator = selected.prepare(dynamics, step_size)
        sample_coordinates = np.asarray(
            (
                wall_bounds[0] + distances[0],
                wall_bounds[1] - distances[1],
            )
        )
        midpoint = 0.5 * (wall_bounds[0] + wall_bounds[1])
        half_length = 0.5 * wall_length
        reference_coordinates = (sample_coordinates - midpoint) / half_length
        sample_evaluation = np.polynomial.chebyshev.chebvander(
            reference_coordinates,
            wall_axis.mode_count - 1,
        )
        self.wall_stress = wall_stress
        self.dynamics = dynamics
        self.integrator = integrator
        self.density = jnp.asarray(float(density_))
        self.sample_distances = jnp.asarray(distances, dtype=float)
        self.sample_coordinates = jnp.asarray(sample_coordinates, dtype=float)
        self.sample_evaluation = jnp.asarray(
            sample_evaluation,
            dtype=jnp.dtype(dynamics.discretization.plan.precision.coefficient_dtype),
        )
        self.roughness_heights = jnp.asarray(roughness, dtype=float)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vector-equilibrium-wall-stress-channel",
                "wall_stress": wall_stress.prepared_id,
                "dynamics": dynamics.compilation_id,
                "integrator": integrator.method_id,
                "density": float(density_).hex(),
                "sample_distances": tuple(float(value).hex() for value in distances),
                "sample_coordinates": tuple(
                    float(value).hex() for value in sample_coordinates
                ),
                "sample_evaluation": array_tree_fingerprint(sample_evaluation),
                "roughness_heights": tuple(float(value).hex() for value in roughness),
                "tangential_owner": "wall-law-only",
            }
        )

    def _evaluate_walls(
        self, velocity_state: ArrayLike, /
    ) -> tuple[
        VectorEquilibriumWallStressResult,
        VectorEquilibriumWallStressResult,
    ]:
        value = self.dynamics.admissible_modes(velocity_state)
        sample_modes = ein.contract(
            "sy,xyzc->xszc",
            self.sample_evaluation.astype(value.dtype),
            value,
            backend="jax",
        )
        reconstruction = self.integrator.backward_euler.reconstruct_horizontal_boundary
        lower_velocity = reconstruction(sample_modes[:, 0, :, :]).at[..., 1].set(0.0)
        upper_velocity = reconstruction(sample_modes[:, 1, :, :]).at[..., 1].set(0.0)
        dtype = lower_velocity.dtype
        lower_normal = jnp.asarray((0.0, 1.0, 0.0), dtype=dtype)
        upper_normal = jnp.asarray((0.0, -1.0, 0.0), dtype=dtype)
        dynamic_viscosity = self.density.astype(dtype) * self.dynamics.problem.viscosity
        lower = self.wall_stress.evaluate(
            lower_velocity,
            lower_normal,
            self.sample_distances[0],
            self.density,
            dynamic_viscosity,
            roughness_height=self.roughness_heights[0],
        )
        upper = self.wall_stress.evaluate(
            upper_velocity,
            upper_normal,
            self.sample_distances[1],
            self.density,
            dynamic_viscosity,
            roughness_height=self.roughness_heights[1],
        )
        return lower, upper

    def _validate_state(self, state: VectorEquilibriumWallStressChannelState, /) -> None:
        if not isinstance(state, VectorEquilibriumWallStressChannelState):
            raise TypeError("state must be a VectorEquilibriumWallStressChannelState.")
        if (
            state.prepared_id != self.prepared_id
            or state.channel.current_velocity.shape != self.dynamics.state_shape
            or state.current_lower.prepared_id != self.wall_stress.prepared_id
            or state.current_upper.prepared_id != self.wall_stress.prepared_id
        ):
            raise ValueError("Channel wall-stress continuation is incompatible.")

    def initialize(
        self,
        initial_state: ArrayLike,
        time: ArrayLike,
        args: object = None,
        /,
    ) -> VectorEquilibriumWallStressChannelState:
        """Initialize SBDF2 and the exact current wall-traction restart state."""
        channel = self.integrator.initialize(initial_state, time, args)
        lower, upper = self._evaluate_walls(channel.current_velocity)
        return VectorEquilibriumWallStressChannelState(
            channel=channel,
            current_lower=lower,
            current_upper=upper,
            prepared_id=self.prepared_id,
        )

    def step(
        self,
        step_index: ArrayLike,
        time: ArrayLike,
        state: VectorEquilibriumWallStressChannelState,
        step_size: ArrayLike,
        args: object = None,
        /,
    ) -> VectorEquilibriumWallStressChannelResult:
        """Advance one atomic wall-owned channel LES transition."""
        self._validate_state(state)
        tangential = jnp.asarray((0, 2))
        current_lower = state.current_lower.traction[..., tangential] / self.density
        current_upper = state.current_upper.traction[..., tangential] / self.density
        applied_lower = current_lower
        applied_upper = current_upper
        projector = self.integrator.backward_euler
        lower_modes = projector.project_horizontal_boundary(applied_lower)
        upper_modes = projector.project_horizontal_boundary(applied_upper)
        transition = self.integrator.step_with_diagnostics(
            jnp.asarray(step_index),
            jnp.asarray(time),
            state.channel,
            jnp.asarray(step_size),
            args,
            lower_tangential_traction=lower_modes,
            upper_tangential_traction=upper_modes,
        )
        channel_candidate = transition.fixed_step.candidate_state
        next_lower, next_upper = self._evaluate_walls(channel_candidate.current_velocity)
        candidate = VectorEquilibriumWallStressChannelState(
            channel=channel_candidate,
            current_lower=next_lower,
            current_upper=next_upper,
            prepared_id=self.prepared_id,
        )
        ledger = self.dynamics.energy_ledger(channel_candidate.current_velocity)
        boundary_power = transition.diagnostics.boundary_power
        boundary_defect = jnp.abs(ledger.wall_power - boundary_power)
        power_scale = jnp.maximum(
            jnp.maximum(jnp.abs(ledger.wall_power), jnp.abs(boundary_power)),
            1.0,
        )
        boundary_identity_closed = (
            transition.diagnostics.tangential_traction_residual
            <= self.dynamics.stokes_plan.constraint_tolerance
        ) & (
            boundary_defect
            <= self.dynamics.stokes_plan.constraint_tolerance * power_scale
        )
        wall_law_successful = (
            jnp.all(state.current_lower.successful)
            & jnp.all(state.current_upper.successful)
            & jnp.all(next_lower.successful)
            & jnp.all(next_upper.successful)
        )
        dissipative = jnp.all(state.current_lower.evidence.dissipative) & jnp.all(
            state.current_upper.evidence.dissipative
        )
        finite = (
            ledger.finite
            & jnp.isfinite(boundary_power)
            & jnp.isfinite(boundary_defect)
            & jnp.all(jnp.isfinite(applied_lower))
            & jnp.all(jnp.isfinite(applied_upper))
        )
        boundary_evidence_successful = (
            wall_law_successful & boundary_identity_closed & dissipative & finite
        )
        successful = transition.fixed_step.successful & boundary_evidence_successful
        accepted = tree_where(successful, candidate, state)
        status = jnp.where(
            boundary_evidence_successful,
            transition.status,
            jnp.asarray(CHANNEL_WALL_STRESS_FAILURE, dtype=jnp.int32),
        )
        evidence = VectorEquilibriumWallStressChannelEvidence(
            lower_wall_stress=state.current_lower,
            upper_wall_stress=state.current_upper,
            applied_lower_specific_traction=applied_lower,
            applied_upper_specific_traction=applied_upper,
            energy_ledger=ledger,
            stokes=transition.diagnostics,
            energy_boundary_work_defect=boundary_defect,
            finite=finite,
            wall_law_successful=wall_law_successful,
            dissipative=dissipative,
            boundary_identity_closed=boundary_identity_closed,
            successful=successful,
        )
        return VectorEquilibriumWallStressChannelResult(
            candidate_state=candidate,
            accepted_state=accepted,
            evidence=evidence,
            successful=successful,
            status=status,
            prepared_id=self.prepared_id,
        )


class StochasticTurbulentInflowMACBoundaryState(StrictModule):
    """Complete accepted-step continuation for a stochastic MAC inflow side."""

    inflow_state: StochasticTurbulentInflowState
    velocity: Array
    scalars: Array
    time: Array
    accepted_steps: Array
    prepared_id: str = eqx.field(static=True)


class StochasticTurbulentInflowMACBoundaryEvidence(StrictModule):
    """Provider, rate, and compatibility evidence for one committed boundary state."""

    realization: StochasticTurbulentInflowEvidence
    elapsed_time: Array
    rate_closure_error: Array
    fluctuation_volume_flux: Array
    total_volume_flux: Array
    maximum_divergence_residual: Array
    covariance_exact: Array
    mass_compatible: Array
    divergence_compatible: Array
    finite: Array
    successful: Array


class StochasticTurbulentInflowMACBoundaryResult(StrictModule):
    """A velocity-inflow side and its exact next accepted-step continuation."""

    state: StochasticTurbulentInflowMACBoundaryState
    boundary: MACBoundarySide
    scalar_values: Array
    scalar_rates: Array
    realization: StochasticTurbulentInflowResult
    evidence: StochasticTurbulentInflowMACBoundaryEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def provider(self) -> MACBoundaryProvider:
        return self.boundary.provider


class PreparedStochasticTurbulentInflowMACBoundary(StrictModule, NonTrainableState):
    """Accepted-step stochastic owner for one structured-MAC velocity-inflow side.

    Each call draws exactly once from the continuation key and emits a concrete
    ``MACBoundaryProvider`` whose material rate is the exact accepted-step
    difference quotient. A caller commits ``result.state`` only with the fluid
    step that consumed ``result.boundary``; rejected attempts retain the input
    state and therefore consume no random draw.
    """

    inflow: PreparedStochasticTurbulentInflow
    boundary_shape: tuple[int, ...] = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        inflow: PreparedStochasticTurbulentInflow,
        axis: str,
        side: Literal["lower", "upper"],
        boundary_shape: tuple[int, ...],
        /,
    ):
        if not isinstance(inflow, PreparedStochasticTurbulentInflow):
            raise TypeError("inflow must be a PreparedStochasticTurbulentInflow.")
        axis_ = str(axis)
        if not axis_:
            raise ValueError("Structured-MAC inflow axis must be non-empty.")
        if side not in ("lower", "upper"):
            raise ValueError("Structured-MAC inflow side must be 'lower' or 'upper'.")
        shape = tuple(int(value) for value in boundary_shape)
        if (
            len(shape) != inflow.spatial_dimension - 1
            or any(value <= 0 for value in shape)
            or math.prod(shape) != inflow.coordinates.shape[0]
        ):
            raise ValueError(
                "boundary_shape must cover every prepared inflow node and have "
                "spatial_dimension - 1 axes."
            )
        if not bool(
            inflow.preparation.successful
            & inflow.preparation.divergence_available
            & inflow.preparation.divergence_compatible
        ):
            raise ValueError(
                "Structured-MAC inflow requires covariance, mass, and represented-"
                "divergence compatible prepared modes."
            )
        self.inflow = inflow
        self.boundary_shape = shape
        self.axis = axis_
        self.side = side
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-stochastic-turbulent-inflow-mac-boundary",
                "inflow": inflow.prepared_id,
                "axis": axis_,
                "side": side,
                "boundary_shape": shape,
                "commit": "accepted-fluid-step-only",
                "rate": "exact-accepted-step-difference-quotient",
            }
        )

    def _validate_state(
        self, state: StochasticTurbulentInflowMACBoundaryState, /
    ) -> None:
        if not isinstance(state, StochasticTurbulentInflowMACBoundaryState):
            raise TypeError("state must be a StochasticTurbulentInflowMACBoundaryState.")
        self.inflow._validate_state(state.inflow_state)
        count = self.inflow.coordinates.shape[0]
        if (
            state.prepared_id != self.prepared_id
            or state.velocity.shape != (count, self.inflow.spatial_dimension)
            or state.scalars.shape != (count, self.inflow.scalar_count)
            or state.time.shape
            or state.accepted_steps.shape
            or not jnp.issubdtype(state.accepted_steps.dtype, jnp.integer)
        ):
            raise ValueError("Structured-MAC inflow continuation is incompatible.")

    def _result(
        self,
        realization: StochasticTurbulentInflowResult,
        time: Array,
        elapsed: Array,
        previous_velocity: Array,
        previous_scalars: Array,
        accepted_steps: Array,
        /,
    ) -> StochasticTurbulentInflowMACBoundaryResult:
        safe_elapsed = jnp.where(elapsed > 0.0, elapsed, 1.0)
        velocity_rate = jnp.where(
            elapsed > 0.0,
            (realization.velocity - previous_velocity) / safe_elapsed,
            jnp.zeros_like(realization.velocity),
        )
        scalar_rate = jnp.where(
            elapsed > 0.0,
            (realization.scalars - previous_scalars) / safe_elapsed,
            jnp.zeros_like(realization.scalars),
        )
        closure_error = jnp.maximum(
            jnp.max(
                jnp.abs(
                    realization.velocity - previous_velocity - elapsed * velocity_rate
                ),
                initial=0.0,
            ),
            jnp.max(
                jnp.abs(realization.scalars - previous_scalars - elapsed * scalar_rate),
                initial=0.0,
            ),
        )
        vector_shape = self.boundary_shape + (self.inflow.spatial_dimension,)
        provider_value = jnp.moveaxis(realization.velocity.reshape(vector_shape), -1, 0)
        provider_rate = jnp.moveaxis(velocity_rate.reshape(vector_shape), -1, 0)
        provider = MACBoundaryProvider(provider_value, rate=provider_rate)
        boundary = MACBoundarySide(
            self.axis,
            self.side,
            "velocity-inflow",
            provider=provider,
        )
        scalar_shape = self.boundary_shape + (self.inflow.scalar_count,)
        scalar_values = jnp.moveaxis(realization.scalars.reshape(scalar_shape), -1, 0)
        scalar_rates = jnp.moveaxis(scalar_rate.reshape(scalar_shape), -1, 0)
        state = StochasticTurbulentInflowMACBoundaryState(
            inflow_state=realization.state,
            velocity=realization.velocity,
            scalars=realization.scalars,
            time=time,
            accepted_steps=accepted_steps,
            prepared_id=self.prepared_id,
        )
        finite = (
            realization.evidence.finite
            & jnp.isfinite(time)
            & jnp.isfinite(elapsed)
            & jnp.all(jnp.isfinite(provider_value))
            & jnp.all(jnp.isfinite(provider_rate))
            & jnp.all(jnp.isfinite(scalar_values))
            & jnp.all(jnp.isfinite(scalar_rates))
            & jnp.isfinite(closure_error)
        )
        successful = (
            realization.evidence.successful
            & realization.evidence.mass_compatible
            & realization.evidence.divergence_compatible
            & finite
            & (
                closure_error
                <= self.inflow.plan.compatibility_tolerance
                * jnp.maximum(jnp.max(jnp.abs(realization.velocity)), 1.0)
            )
        )
        evidence = StochasticTurbulentInflowMACBoundaryEvidence(
            realization=realization.evidence,
            elapsed_time=elapsed,
            rate_closure_error=closure_error,
            fluctuation_volume_flux=realization.evidence.fluctuation_volume_flux,
            total_volume_flux=realization.evidence.total_volume_flux,
            maximum_divergence_residual=(
                realization.evidence.maximum_divergence_residual
            ),
            covariance_exact=realization.evidence.covariance_exact,
            mass_compatible=realization.evidence.mass_compatible,
            divergence_compatible=realization.evidence.divergence_compatible,
            finite=finite,
            successful=successful,
        )
        return StochasticTurbulentInflowMACBoundaryResult(
            state=state,
            boundary=boundary,
            scalar_values=scalar_values,
            scalar_rates=scalar_rates,
            realization=realization,
            evidence=evidence,
            prepared_id=self.prepared_id,
        )

    def initialize(
        self,
        key: ArrayLike,
        time: ArrayLike,
        /,
        *,
        sample_index: int = 0,
        mean_velocity: ArrayLike | None = None,
        mean_scalars: ArrayLike | None = None,
    ) -> StochasticTurbulentInflowMACBoundaryResult:
        """Draw and own the initial concrete MAC inflow boundary."""
        time_ = jnp.asarray(time, dtype=self.inflow.synthesis.dtype).reshape(())
        time_ = eqx.error_if(
            time_,
            ~jnp.isfinite(time_),
            "Structured-MAC inflow initial time must be finite.",
        )
        realization = self.inflow.sample(
            self.inflow.initialize(key, sample_index=sample_index),
            mean_velocity=mean_velocity,
            mean_scalars=mean_scalars,
        )
        return self._result(
            realization,
            time_,
            jnp.zeros((), dtype=time_.dtype),
            realization.velocity,
            realization.scalars,
            jnp.asarray(0, dtype=jnp.uint32),
        )

    def advance(
        self,
        state: StochasticTurbulentInflowMACBoundaryState,
        time: ArrayLike,
        /,
        *,
        mean_velocity: ArrayLike | None = None,
        mean_scalars: ArrayLike | None = None,
    ) -> StochasticTurbulentInflowMACBoundaryResult:
        """Draw the boundary committed by the next accepted fluid step."""
        self._validate_state(state)
        time_ = jnp.asarray(time, dtype=state.time.dtype).reshape(())
        elapsed = time_ - state.time
        time_ = eqx.error_if(
            time_,
            ~(jnp.isfinite(time_) & jnp.isfinite(elapsed) & (elapsed > 0.0)),
            "Structured-MAC inflow accepted times must increase finitely.",
        )
        realization = self.inflow.sample(
            state.inflow_state,
            mean_velocity=mean_velocity,
            mean_scalars=mean_scalars,
        )
        return self._result(
            realization,
            time_,
            elapsed,
            state.velocity,
            state.scalars,
            state.accepted_steps + jnp.asarray(1, dtype=state.accepted_steps.dtype),
        )


__all__ = [
    "CHANNEL_WALL_STRESS_FAILURE",
    "PreparedStochasticTurbulentInflow",
    "PreparedStochasticTurbulentInflowMACBoundary",
    "PreparedVectorEquilibriumWallStress",
    "PreparedVectorEquilibriumWallStressChannel",
    "StochasticTurbulentInflowEvidence",
    "StochasticTurbulentInflowMACBoundaryEvidence",
    "StochasticTurbulentInflowMACBoundaryResult",
    "StochasticTurbulentInflowMACBoundaryState",
    "StochasticTurbulentInflowPlan",
    "StochasticTurbulentInflowPreparationEvidence",
    "StochasticTurbulentInflowResult",
    "StochasticTurbulentInflowState",
    "VectorEquilibriumWallStressEvidence",
    "VectorEquilibriumWallStressChannelEvidence",
    "VectorEquilibriumWallStressChannelResult",
    "VectorEquilibriumWallStressChannelState",
    "VectorEquilibriumWallStressPlan",
    "VectorEquilibriumWallStressResult",
]
