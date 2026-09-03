#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import ScientificArtifactEnvelope


class S3ManifoldPlan(StrictModule, NonTrainableState):
    radius: float = eqx.field(static=True)
    cut_locus_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, radius: float, /, *, cut_locus_tolerance: float = 1.0e-8):
        radius_ = float(radius)
        tolerance = float(cut_locus_tolerance)
        if (
            not np.isfinite(radius_)
            or radius_ <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("S3 manifold parameters are invalid.")
        self.radius = radius_
        self.cut_locus_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {"kind": "s3-manifold", "radius": radius_, "cut_locus_tolerance": tolerance}
        )

    def normalize(self, points: ArrayLike, /) -> Array:
        value = jnp.asarray(points)
        if value.shape[-1] != 4:
            raise ValueError("S3 points must use four embedding coordinates.")
        norm = jnp.sqrt(jnp.sum(value**2, axis=-1, keepdims=True))
        return self.radius * value / norm

    def project_tangent(self, points: Array, vectors: Array, /) -> Array:
        return (
            vectors
            - points * jnp.sum(points * vectors, axis=-1, keepdims=True) / self.radius**2
        )

    def distance(self, first: ArrayLike, second: ArrayLike, /) -> Array:
        left = self.normalize(first)
        right = self.normalize(second)
        cosine = jnp.sum(left * right, axis=-1) / self.radius**2
        return self.radius * jnp.arccos(jnp.clip(cosine, -1.0, 1.0))

    def exponential(self, point: Array, tangent: Array, /) -> Array:
        point_ = self.normalize(point)
        tangent_ = self.project_tangent(point_, tangent)
        norm = jnp.sqrt(jnp.sum(tangent_**2, axis=-1, keepdims=True))
        angle = norm / self.radius
        safe_norm = jnp.where(norm > 0.0, norm, 1.0)
        result = (
            jnp.cos(angle) * point_ + self.radius * jnp.sin(angle) * tangent_ / safe_norm
        )
        return jnp.where(norm > 0.0, self.normalize(result), point_)

    def logarithm(self, first: Array, second: Array, /) -> Array:
        point = self.normalize(first)
        target = self.normalize(second)
        cosine = jnp.clip(
            jnp.sum(point * target, axis=-1, keepdims=True) / self.radius**2, -1.0, 1.0
        )
        angle = jnp.arccos(cosine)
        sine = jnp.sin(angle)
        sine = eqx.error_if(
            sine,
            jnp.any(jnp.abs(jnp.pi - angle) < self.cut_locus_tolerance),
            "S3 logarithm is undefined at the antipodal cut locus.",
        )
        direction = target - cosine * point
        safe = jnp.where(jnp.abs(sine) > 1.0e-12, sine, 1.0)
        return jnp.where(jnp.abs(sine) > 1.0e-12, angle * direction / safe, 0.0)

    def parallel_transport(self, first: Array, second: Array, tangent: Array, /) -> Array:
        point = self.normalize(first)
        target = self.normalize(second)
        vector = self.project_tangent(point, tangent)
        denominator = self.radius**2 + jnp.sum(point * target, axis=-1, keepdims=True)
        denominator = eqx.error_if(
            denominator,
            jnp.any(jnp.abs(denominator) < self.cut_locus_tolerance),
            "S3 parallel transport is undefined at the antipode.",
        )
        transported = vector - (
            jnp.sum(vector * target, axis=-1, keepdims=True) / denominator
        ) * (point + target)
        return self.project_tangent(target, transported)


class S3ParticleState(StrictModule):
    positions: Array
    canonical_momenta: Array
    masses: Array
    scale_factor: Array


class S3KDKResult(StrictModule):
    state: S3ParticleState
    norm_defect: Array
    tangent_defect: Array
    maximum_drift_angle: Array
    finite: Array
    successful: Array


class S3GeodesicKDKPlan(StrictModule, NonTrainableState):
    manifold: S3ManifoldPlan

    def __init__(self, manifold: S3ManifoldPlan, /):
        self.manifold = manifold

    def initialize(
        self,
        positions: ArrayLike,
        canonical_momenta: ArrayLike,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> S3ParticleState:
        position = self.manifold.normalize(positions)
        momentum = self.manifold.project_tangent(
            position, jnp.asarray(canonical_momenta, dtype=position.dtype)
        )
        mass = jnp.asarray(masses, dtype=position.dtype)
        scale = jnp.asarray(scale_factor, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != 4
            or momentum.shape != position.shape
            or mass.shape != (position.shape[0],)
            or scale.shape != ()
        ):
            raise ValueError("S3 particle state shapes are invalid.")
        return S3ParticleState(position, momentum, mass, scale)

    def advance(
        self,
        state: S3ParticleState,
        end_scale_factor: ArrayLike,
        drift_factor: ArrayLike,
        first_kick_factor: ArrayLike,
        second_kick_factor: ArrayLike,
        acceleration_start: ArrayLike,
        acceleration_end: ArrayLike,
        /,
    ) -> S3KDKResult:
        end = jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype)
        drift = jnp.asarray(drift_factor, dtype=state.scale_factor.dtype)
        kick_0 = jnp.asarray(first_kick_factor, dtype=state.scale_factor.dtype)
        kick_1 = jnp.asarray(second_kick_factor, dtype=state.scale_factor.dtype)
        acceleration_0 = self.manifold.project_tangent(
            state.positions, jnp.asarray(acceleration_start, dtype=state.positions.dtype)
        )
        half = state.canonical_momenta + kick_0 * state.masses[:, None] * acceleration_0
        tangent_step = drift * half / state.masses[:, None]
        angles = jnp.sqrt(jnp.sum(tangent_step**2, axis=-1)) / self.manifold.radius
        proposed_position = self.manifold.exponential(state.positions, tangent_step)
        transported_half = self.manifold.parallel_transport(
            state.positions, proposed_position, half
        )
        acceleration_1 = self.manifold.project_tangent(
            proposed_position, jnp.asarray(acceleration_end, dtype=state.positions.dtype)
        )
        momentum = transported_half + kick_1 * state.masses[:, None] * acceleration_1
        momentum = self.manifold.project_tangent(proposed_position, momentum)
        norm_defect = jnp.max(
            jnp.abs(
                jnp.sqrt(jnp.sum(proposed_position**2, axis=-1)) - self.manifold.radius
            )
        )
        tangent_defect = jnp.max(jnp.abs(jnp.sum(proposed_position * momentum, axis=-1)))
        finite = jnp.all(jnp.isfinite(proposed_position)) & jnp.all(
            jnp.isfinite(momentum)
        )
        successful = (
            finite
            & (end > state.scale_factor)
            & jnp.all(angles < jnp.pi - self.manifold.cut_locus_tolerance)
        )
        accepted = S3ParticleState(
            jnp.where(successful, proposed_position, state.positions),
            jnp.where(successful, momentum, state.canonical_momenta),
            state.masses,
            jnp.where(successful, end, state.scale_factor),
        )
        return S3KDKResult(
            accepted, norm_defect, tangent_defect, jnp.max(angles), finite, successful
        )


class S3HarmonicBasisPlan(StrictModule, NonTrainableState):
    mode_indices: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    eigenvalues: Array
    evaluation_matrix: Array
    gradient_matrix: Array
    quadrature_weights: Array
    radius: float = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_indices: tuple[tuple[int, int, int], ...],
        evaluation_matrix: ArrayLike,
        gradient_matrix: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        radius: float,
        artifact: ScientificArtifactEnvelope,
    ):
        indices = tuple(tuple(int(value) for value in index) for index in mode_indices)
        evaluation = jax.lax.stop_gradient(jnp.asarray(evaluation_matrix))
        gradient = jax.lax.stop_gradient(
            jnp.asarray(gradient_matrix, dtype=evaluation.dtype)
        )
        weights = jax.lax.stop_gradient(
            jnp.asarray(quadrature_weights, dtype=evaluation.dtype)
        )
        radius_ = float(radius)
        if (
            not indices
            or evaluation.shape[1] != len(indices)
            or gradient.shape != (evaluation.shape[0], len(indices), 4)
            or weights.shape != (evaluation.shape[0],)
            or radius_ <= 0.0
        ):
            raise ValueError("S3 harmonic basis shapes are invalid.")
        n = jnp.asarray([index[0] for index in indices], dtype=evaluation.dtype)
        eigenvalues = n * (n + 2.0) / radius_**2
        self.mode_indices = indices
        self.eigenvalues = eigenvalues
        self.evaluation_matrix = evaluation
        self.gradient_matrix = gradient
        self.quadrature_weights = weights
        self.radius = radius_
        self.artifact = artifact
        self.basis_id = canonical_fingerprint(
            {
                "kind": "s3-harmonic-basis",
                "indices": [list(index) for index in indices],
                "radius": radius_,
                "artifact": artifact.artifact_id,
                "shape": list(evaluation.shape),
            }
        )

    def project(self, nodal_field: ArrayLike, /) -> Array:
        field = jnp.asarray(nodal_field, dtype=self.evaluation_matrix.dtype)
        if field.shape != (self.evaluation_matrix.shape[0],):
            raise ValueError("S3 nodal field does not match basis nodes.")
        return contract(
            "p,pm,p->m", self.quadrature_weights, self.evaluation_matrix, field
        )

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients, dtype=self.evaluation_matrix.dtype)
        if values.shape != (len(self.mode_indices),):
            raise ValueError("S3 coefficients do not match basis modes.")
        return contract("pm,m->p", self.evaluation_matrix, values)


class S3PoissonResult(StrictModule):
    density_coefficients: Array
    potential_coefficients: Array
    potential: Array
    gradient: Array
    zero_mode_removed: Array
    finite: Array
    successful: Array


class S3PoissonPlan(StrictModule, NonTrainableState):
    basis: S3HarmonicBasisPlan
    gravitational_constant: float = eqx.field(static=True)

    def __init__(self, basis: S3HarmonicBasisPlan, gravitational_constant: float, /):
        gravity = float(gravitational_constant)
        if not np.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("S3 gravitational constant must be finite and positive.")
        self.basis = basis
        self.gravitational_constant = gravity

    def solve(
        self,
        density_contrast: ArrayLike,
        scale_factor: ArrayLike,
        mean_density: ArrayLike,
        /,
    ) -> S3PoissonResult:
        scale = jnp.asarray(scale_factor, dtype=self.basis.evaluation_matrix.dtype)
        mean = jnp.asarray(mean_density, dtype=scale.dtype)
        density_coefficients = self.basis.project(density_contrast)
        zero = self.basis.eigenvalues == 0.0
        safe_eigenvalue = jnp.where(zero, 1.0, self.basis.eigenvalues)
        potential_coefficients = jnp.where(
            zero,
            0.0,
            -4.0
            * jnp.pi
            * self.gravitational_constant
            * scale**2
            * mean
            * density_coefficients
            / safe_eigenvalue,
        )
        potential = self.basis.synthesize(potential_coefficients)
        gradient = contract(
            "pmc,m->pc", self.basis.gradient_matrix, potential_coefficients
        )
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(gradient))
        return S3PoissonResult(
            density_coefficients,
            potential_coefficients,
            potential,
            gradient,
            jnp.all(potential_coefficients[zero] == 0.0),
            finite,
            finite,
        )


class S3ParticleMeshResult(StrictModule):
    nodal_density: Array
    nodal_acceleration: Array
    particle_acceleration: Array
    mass_defect: Array
    finite: Array
    successful: Array


class S3ParticleMeshPlan(StrictModule, NonTrainableState):
    poisson: S3PoissonPlan
    deposition_matrix: Array
    gather_matrix: Array

    def __init__(
        self,
        poisson: S3PoissonPlan,
        deposition_matrix: ArrayLike,
        gather_matrix: ArrayLike,
        /,
    ):
        deposit = jax.lax.stop_gradient(jnp.asarray(deposition_matrix))
        gather = jax.lax.stop_gradient(jnp.asarray(gather_matrix, dtype=deposit.dtype))
        node_count = poisson.basis.evaluation_matrix.shape[0]
        if (
            deposit.ndim != 2
            or deposit.shape[0] != node_count
            or gather.shape != (deposit.shape[1], node_count)
        ):
            raise ValueError("S3 particle deposition/gather matrices are invalid.")
        self.poisson = poisson
        self.deposition_matrix = deposit
        self.gather_matrix = gather

    def evaluate(
        self,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        mean_density: ArrayLike,
        /,
    ) -> S3ParticleMeshResult:
        mass = jnp.asarray(masses, dtype=self.deposition_matrix.dtype)
        if mass.shape != (self.deposition_matrix.shape[1],):
            raise ValueError("S3 particle masses do not match transfer capacity.")
        nodal_mass = contract("pn,n->p", self.deposition_matrix, mass)
        volume = 2.0 * jnp.pi**2 * self.poisson.basis.radius**3
        nodal_density = nodal_mass / (self.poisson.basis.quadrature_weights * volume)
        density_contrast = nodal_density / jnp.asarray(mean_density) - 1.0
        solved = self.poisson.solve(density_contrast, scale_factor, mean_density)
        nodal_acceleration = -solved.gradient
        particle_acceleration = contract(
            "np,pc->nc", self.gather_matrix, nodal_acceleration
        )
        mass_defect = jnp.sum(nodal_mass) - jnp.sum(mass)
        finite = jnp.all(jnp.isfinite(particle_acceleration))
        return S3ParticleMeshResult(
            nodal_density,
            nodal_acceleration,
            particle_acceleration,
            mass_defect,
            finite,
            finite & solved.successful,
        )


__all__ = [
    "S3GeodesicKDKPlan",
    "S3HarmonicBasisPlan",
    "S3KDKResult",
    "S3ManifoldPlan",
    "S3ParticleMeshPlan",
    "S3ParticleMeshResult",
    "S3ParticleState",
    "S3PoissonPlan",
    "S3PoissonResult",
]
