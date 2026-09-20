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

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._linear_boltzmann import MultigroupSlabTransportProblem


class DiscreteOrdinatesEvidence(StrictModule):
    iteration_count: Array
    source_iteration_residual: Array
    global_balance_residual: Array
    minimum_angular_flux: Array
    finite: Array
    nonnegative: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DiscreteOrdinatesResult(StrictModule, NonTrainableState):
    angular_flux: Array
    scalar_flux: Array
    current: Array
    left_outgoing: Array
    right_outgoing: Array
    evidence: DiscreteOrdinatesEvidence
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def response(self, cell_group_weights: ArrayLike, cell_widths: ArrayLike, /) -> Array:
        weights = jnp.asarray(cell_group_weights, dtype=self.scalar_flux.dtype)
        widths = jnp.asarray(cell_widths, dtype=self.scalar_flux.dtype)
        if (
            weights.shape != self.scalar_flux.shape
            or widths.shape != self.scalar_flux.shape[:1]
        ):
            raise ValueError("Transport response weights or cell widths are invalid.")
        return jnp.sum(widths[:, None] * weights * self.scalar_flux)


class DiscreteOrdinatesTransportPlan(StrictModule, NonTrainableState):
    problem: MultigroupSlabTransportProblem
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    dsa: bool = eqx.field(static=True)
    dsa_relaxation: float = eqx.field(static=True)
    dsa_matrices: Array
    positive_angles: tuple[int, ...] = eqx.field(static=True)
    negative_angles: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MultigroupSlabTransportProblem,
        /,
        *,
        maximum_iterations: int = 500,
        tolerance: float = 1.0e-9,
        dsa: bool = False,
        dsa_relaxation: float = 0.5,
    ):
        if not isinstance(problem, MultigroupSlabTransportProblem):
            raise TypeError("problem must be MultigroupSlabTransportProblem.")
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        relaxation = float(dsa_relaxation)
        if (
            iterations < 1
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(relaxation)
            or not 0.0 < relaxation <= 1.0
        ):
            raise ValueError("Discrete-ordinates iteration controls are invalid.")
        matrices = _dsa_matrices(problem)
        positive_angles = tuple(
            np.flatnonzero(np.asarray(problem.quadrature.ordinates) > 0.0)
        )
        negative_angles = tuple(
            np.flatnonzero(np.asarray(problem.quadrature.ordinates) < 0.0)
        )
        self.problem = problem
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.dsa = bool(dsa)
        self.dsa_relaxation = relaxation
        self.dsa_matrices = jnp.asarray(matrices)
        self.positive_angles = positive_angles
        self.negative_angles = negative_angles
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multigroup-slab-discrete-ordinates",
                "problem": problem.problem_id,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "dsa": bool(dsa),
                "dsa_relaxation": relaxation,
                "dsa_matrices": array_tree_fingerprint(matrices),
                "positive_angles": positive_angles,
                "negative_angles": negative_angles,
            }
        )

    def _incoming(self, angular: Array, group: int, angle: int, /) -> Array:
        boundary = self.problem.boundaries
        opposite = self.problem.quadrature.opposite_indices[angle]
        if angle in self.positive_angles:
            if boundary.left_kind == "incident":
                return boundary.left_incident[group, angle]
            if boundary.left_kind == "reflecting":
                return angular[0, group, opposite]
            return jnp.asarray(0.0, dtype=angular.dtype)
        if boundary.right_kind == "incident":
            return boundary.right_incident[group, angle]
        if boundary.right_kind == "reflecting":
            return angular[-1, group, opposite]
        return jnp.asarray(0.0, dtype=angular.dtype)

    def _sweep_group(self, angular: Array, group: int, angular_source: Array, /) -> Array:
        total = self.problem.total_cross_section[:, group]
        width = self.problem.cell_widths
        output = angular
        for angle in range(self.problem.quadrature.angle_count):
            mu = self.problem.quadrature.ordinates[angle]
            incoming = self._incoming(angular, group, angle)
            coefficient = jnp.abs(mu) / width

            def step(boundary_flux, values):
                source, attenuation, total_value = values
                flux = (source + attenuation * boundary_flux) / (
                    total_value + attenuation
                )
                return flux, flux

            if angle in self.positive_angles:
                _, flux = jax.lax.scan(
                    step, incoming, (angular_source, coefficient, total)
                )
            else:
                _, reversed_flux = jax.lax.scan(
                    step,
                    incoming,
                    (angular_source[::-1], coefficient[::-1], total[::-1]),
                )
                flux = reversed_flux[::-1]
            output = output.at[:, group, angle].set(flux)
        return output

    def _iterate(self, angular: Array, scalar: Array, /) -> tuple[Array, Array]:
        updated = angular
        current_scalar = scalar
        for group_set in self.problem.group_sets:
            for group in group_set:
                scattering = contract(
                    "ci,ci->c",
                    self.problem.scattering_cross_section[:, :, group],
                    current_scalar,
                    backend="jax",
                )
                angular_source = 0.5 * (
                    self.problem.fixed_isotropic_source[:, group] + scattering
                )
                updated = self._sweep_group(updated, group, angular_source)
                group_scalar = contract(
                    "a,ca->c",
                    self.problem.quadrature.weights,
                    updated[:, group, :],
                    backend="jax",
                )
                current_scalar = current_scalar.at[:, group].set(group_scalar)
        swept_scalar = contract(
            "a,cga->cg",
            self.problem.quadrature.weights,
            updated,
            backend="jax",
        )
        if self.dsa:
            residual = swept_scalar - scalar
            corrections = jnp.stack(
                tuple(
                    jnp.linalg.solve(self.dsa_matrices[group], residual[:, group])
                    for group in range(self.problem.group_count)
                ),
                axis=-1,
            )
            corrected = jnp.maximum(swept_scalar + self.dsa_relaxation * corrections, 0.0)
            ratio = corrected / jnp.maximum(
                swept_scalar, jnp.finfo(swept_scalar.dtype).tiny
            )
            updated = updated * ratio[..., None]
            swept_scalar = corrected
        return updated, swept_scalar

    def solve(
        self, initial_scalar_flux: ArrayLike | None = None, /
    ) -> DiscreteOrdinatesResult:
        shape = (
            self.problem.cell_count,
            self.problem.group_count,
            self.problem.quadrature.angle_count,
        )
        angular = jnp.zeros(shape, dtype=self.problem.total_cross_section.dtype)
        scalar = (
            jnp.zeros(shape[:2], dtype=angular.dtype)
            if initial_scalar_flux is None
            else jnp.asarray(initial_scalar_flux, dtype=angular.dtype)
        )
        if scalar.shape != shape[:2]:
            raise ValueError("initial_scalar_flux must be cell-by-group data.")
        initial_finite = jnp.all(jnp.isfinite(scalar))
        initial_nonnegative = jnp.all(scalar >= 0.0)
        scalar = jnp.where(initial_finite & initial_nonnegative, scalar, 0.0)

        def body(_, carry):
            previous_angular, previous_scalar = carry
            return self._iterate(previous_angular, previous_scalar)

        angular, scalar = jax.lax.fori_loop(
            0, self.maximum_iterations, body, (angular, scalar)
        )
        next_angular, next_scalar = self._iterate(angular, scalar)
        residual = jnp.max(jnp.abs(next_scalar - scalar)) / jnp.maximum(
            jnp.max(jnp.abs(next_scalar)), 1.0
        )
        angular, scalar = next_angular, next_scalar
        current = contract(
            "a,a,cga->cg",
            self.problem.quadrature.weights,
            self.problem.quadrature.ordinates,
            angular,
            backend="jax",
        )
        positive = jnp.asarray(self.positive_angles, dtype=jnp.int32)
        negative = jnp.asarray(self.negative_angles, dtype=jnp.int32)
        left_outgoing = jnp.take(angular[0], negative, axis=-1)
        right_outgoing = jnp.take(angular[-1], positive, axis=-1)
        source = jnp.sum(
            self.problem.cell_widths[:, None] * self.problem.fixed_isotropic_source
        )
        absorption = jnp.sum(
            self.problem.cell_widths[:, None]
            * self.problem.absorption_cross_section
            * scalar
        )
        negative_weight = jnp.take(self.problem.quadrature.weights, negative, axis=0)
        positive_weight = jnp.take(self.problem.quadrature.weights, positive, axis=0)
        negative_mu = jnp.take(self.problem.quadrature.ordinates, negative, axis=0)
        positive_mu = jnp.take(self.problem.quadrature.ordinates, positive, axis=0)
        left_leakage = jnp.sum(
            negative_weight[None, :] * -negative_mu[None, :] * left_outgoing
        )
        right_leakage = jnp.sum(
            positive_weight[None, :] * positive_mu[None, :] * right_outgoing
        )
        left_incident = jnp.take(self.problem.boundaries.left_incident, positive, axis=-1)
        right_incident = jnp.take(
            self.problem.boundaries.right_incident, negative, axis=-1
        )
        left_incoming = jnp.sum(
            positive_weight[None, :] * positive_mu[None, :] * left_incident
        )
        right_incoming = jnp.sum(
            negative_weight[None, :] * -negative_mu[None, :] * right_incident
        )
        balance = (
            source
            + left_incoming
            + right_incoming
            - absorption
            - left_leakage
            - right_leakage
        )
        finite = (
            initial_finite
            & jnp.all(jnp.isfinite(angular))
            & jnp.all(jnp.isfinite(scalar))
            & jnp.isfinite(residual)
            & jnp.isfinite(balance)
        )
        minimum = jnp.min(angular)
        nonnegative = initial_nonnegative & (
            minimum >= -64.0 * jnp.finfo(angular.dtype).eps
        )
        converged = residual <= self.tolerance
        successful = finite & nonnegative & converged
        evidence = DiscreteOrdinatesEvidence(
            jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            residual,
            balance,
            minimum,
            finite,
            nonnegative,
            converged,
            successful,
            self.plan_id,
        )
        return DiscreteOrdinatesResult(
            angular,
            scalar,
            current,
            left_outgoing,
            right_outgoing,
            evidence,
            self.problem.problem_id,
            self.plan_id,
        )


def _dsa_matrices(problem: MultigroupSlabTransportProblem) -> np.ndarray:
    total = np.asarray(problem.total_cross_section)
    absorption = np.asarray(problem.absorption_cross_section)
    widths = np.asarray(problem.cell_widths)
    cells, groups = total.shape
    output = np.zeros((groups, cells, cells), dtype=np.float64)
    for group in range(groups):
        diffusion = 1.0 / (3.0 * total[:, group])
        matrix = np.diag(absorption[:, group] + 1.0e-12)
        for cell in range(cells - 1):
            interface = (
                2.0
                * diffusion[cell]
                * diffusion[cell + 1]
                / (diffusion[cell] + diffusion[cell + 1])
            )
            distance = 0.5 * (widths[cell] + widths[cell + 1])
            coefficient = interface / distance
            matrix[cell, cell] += coefficient / widths[cell]
            matrix[cell + 1, cell + 1] += coefficient / widths[cell + 1]
            matrix[cell, cell + 1] -= coefficient / widths[cell]
            matrix[cell + 1, cell] -= coefficient / widths[cell + 1]
        if problem.boundaries.left_kind != "reflecting":
            matrix[0, 0] += diffusion[0] / widths[0] ** 2
        if problem.boundaries.right_kind != "reflecting":
            matrix[-1, -1] += diffusion[-1] / widths[-1] ** 2
        output[group] = matrix
    return output


__all__ = [
    "DiscreteOrdinatesEvidence",
    "DiscreteOrdinatesResult",
    "DiscreteOrdinatesTransportPlan",
]
