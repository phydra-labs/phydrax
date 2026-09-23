#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded multigroup slab S_N transport, criticality, and Bateman depletion."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from numpy.polynomial.legendre import leggauss

from .._fingerprint import canonical_fingerprint
from ..ein import contract
from ..linalg import ArraySpace, DenseLinearOperator, matrix_exponential_action
from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class SNTransportResult:
    scalar_flux: Array
    angular_flux: Array
    iterations: int
    residual_norm: Array
    converged: Array
    multiplication_factor: Array


class SlabSNTransportPlan:
    """Cellwise-constant multigroup 1D slab transport with vacuum boundaries."""

    def __init__(
        self,
        cell_widths: ArrayLike,
        total_cross_section: ArrayLike,
        scattering_cross_section: ArrayLike,
        /,
        *,
        ordinates: int = 8,
        maximum_iterations: int = 512,
        relative_tolerance: float = 1.0e-9,
    ):
        widths = np.asarray(cell_widths, dtype=np.float64)
        total = np.asarray(total_cross_section, dtype=np.float64)
        scatter = np.asarray(scattering_cross_section, dtype=np.float64)
        if (
            widths.ndim != 1
            or widths.size == 0
            or np.any(~np.isfinite(widths))
            or np.any(widths <= 0.0)
        ):
            raise ValueError("Transport cell widths must be a finite positive vector.")
        if (
            total.ndim != 2
            or total.shape[0] != widths.size
            or np.any(~np.isfinite(total))
            or np.any(total <= 0.0)
        ):
            raise ValueError(
                "Total cross sections must have shape (cell, group) and be finite positive."
            )
        groups = total.shape[1]
        if (
            scatter.shape != (widths.size, groups, groups)
            or np.any(~np.isfinite(scatter))
            or np.any(scatter < 0.0)
        ):
            raise ValueError(
                "Scattering cross sections must have finite nonnegative shape (cell, to, from)."
            )
        tolerance = 512.0 * np.finfo(np.float64).eps * np.maximum(total, 1.0)
        if np.any(np.sum(scatter, axis=1) > total + tolerance):
            raise ValueError("Scattering removal cannot exceed total cross section.")
        if isinstance(ordinates, bool) or not isinstance(ordinates, Integral):
            raise TypeError("ordinates must be an integer.")
        count = int(ordinates)
        if count < 2 or count % 2:
            raise ValueError("S_N ordinate count must be positive and even.")
        if (
            isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, Integral)
            or int(maximum_iterations) < 1
        ):
            raise ValueError("maximum_iterations must be a positive integer.")
        relative_tolerance_ = float(relative_tolerance)
        if not np.isfinite(relative_tolerance_) or relative_tolerance_ <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive.")
        directions, weights = leggauss(count)
        self.cell_widths = jnp.asarray(widths)
        self.total_cross_section = jnp.asarray(total)
        self.scattering_cross_section = jnp.asarray(scatter)
        self.directions = jnp.asarray(directions)
        self.weights = jnp.asarray(weights)
        self.maximum_iterations = int(maximum_iterations)
        self.relative_tolerance = relative_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "slab-sn-transport",
                "cell_widths": widths.tolist(),
                "total_cross_section": total.tolist(),
                "scattering_cross_section": scatter.tolist(),
                "ordinates": count,
                "maximum_iterations": self.maximum_iterations,
                "relative_tolerance": self.relative_tolerance,
            }
        )

    def _sweep(
        self, isotropic_source: Array, left: Array, right: Array
    ) -> tuple[Array, Array]:
        cells, groups = isotropic_source.shape
        angular = jnp.zeros(
            (self.directions.size, cells, groups), dtype=isotropic_source.dtype
        )
        for direction_index in range(self.directions.size):
            direction = float(self.directions[direction_index])
            incoming = left if direction > 0.0 else right
            indices = range(cells) if direction > 0.0 else range(cells - 1, -1, -1)
            for cell in indices:
                optical = (
                    self.total_cross_section[cell]
                    * self.cell_widths[cell]
                    / abs(direction)
                )
                attenuation = jnp.exp(-optical)
                equilibrium = isotropic_source[cell] / self.total_cross_section[cell]
                average_factor = jnp.where(
                    optical > 1.0e-12,
                    (1.0 - attenuation) / optical,
                    1.0 - 0.5 * optical,
                )
                average = equilibrium + (incoming - equilibrium) * average_factor
                outgoing = equilibrium + (incoming - equilibrium) * attenuation
                angular = angular.at[direction_index, cell].set(average)
                incoming = outgoing
        scalar = jnp.tensordot(self.weights, angular, axes=1)
        return scalar, angular

    def solve_fixed_source(
        self,
        external_source: ArrayLike,
        /,
        *,
        left_boundary: ArrayLike | None = None,
        right_boundary: ArrayLike | None = None,
    ) -> SNTransportResult:
        source_host = np.asarray(external_source)
        if source_host.shape != self.total_cross_section.shape:
            raise ValueError("External source must have shape (cell, group).")
        if np.any(~np.isfinite(source_host)) or np.any(source_host < 0.0):
            raise ValueError("External source must be finite and nonnegative.")
        source = jnp.asarray(source_host)
        groups = source.shape[1]
        left_host = (
            np.zeros((groups,), dtype=source_host.dtype)
            if left_boundary is None
            else np.asarray(left_boundary)
        )
        right_host = (
            np.zeros((groups,), dtype=source_host.dtype)
            if right_boundary is None
            else np.asarray(right_boundary)
        )
        if left_host.shape != (groups,) or right_host.shape != (groups,):
            raise ValueError("Boundary inflows must have exact shape (group,).")
        if (
            np.any(~np.isfinite(left_host))
            or np.any(left_host < 0.0)
            or np.any(~np.isfinite(right_host))
            or np.any(right_host < 0.0)
        ):
            raise ValueError("Boundary inflows must be finite and nonnegative.")
        left = jnp.asarray(left_host)
        right = jnp.asarray(right_host)
        scalar = jnp.ones_like(source)
        angular = jnp.zeros((self.directions.size, *source.shape), dtype=source.dtype)
        residual = jnp.asarray(jnp.inf)
        iterations = 0
        for iteration in range(self.maximum_iterations):
            scattering = contract(
                "cij,cj->ci", self.scattering_cross_section, scalar, backend="jax"
            )
            next_scalar, angular = self._sweep(0.5 * (source + scattering), left, right)
            residual = jnp.linalg.norm(next_scalar - scalar) / jnp.maximum(
                jnp.linalg.norm(next_scalar), 1.0e-30
            )
            scalar = next_scalar
            iterations = iteration + 1
            if float(residual) <= self.relative_tolerance:
                break
        converged = jnp.isfinite(residual) & (residual <= self.relative_tolerance)
        return SNTransportResult(
            scalar,
            angular,
            iterations,
            residual,
            converged,
            jnp.asarray(jnp.nan),
        )

    def solve_criticality(
        self,
        nu_fission_cross_section: ArrayLike,
        spectrum: ArrayLike,
        /,
        *,
        initial_factor: float = 1.0,
    ) -> SNTransportResult:
        nu_fission_host = np.asarray(nu_fission_cross_section)
        chi_host = np.asarray(spectrum)
        if nu_fission_host.shape != self.total_cross_section.shape or chi_host.shape != (
            nu_fission_host.shape[1],
        ):
            raise ValueError("Criticality data do not match transport groups.")
        if (
            np.any(~np.isfinite(nu_fission_host))
            or np.any(nu_fission_host < 0.0)
            or not np.any(nu_fission_host > 0.0)
        ):
            raise ValueError(
                "Fission cross sections must be finite, nonnegative, and productive."
            )
        if (
            np.any(~np.isfinite(chi_host))
            or np.any(chi_host < 0.0)
            or not np.isclose(np.sum(chi_host), 1.0, rtol=1.0e-10, atol=1.0e-12)
        ):
            raise ValueError(
                "Fission spectrum must be finite, nonnegative, and normalized."
            )
        factor_host = float(initial_factor)
        if not np.isfinite(factor_host) or factor_host <= 0.0:
            raise ValueError("initial_factor must be finite and positive.")
        nu_fission = jnp.asarray(nu_fission_host)
        chi = jnp.asarray(chi_host)
        factor = jnp.asarray(factor_host)
        scalar = jnp.ones_like(nu_fission)
        angular = jnp.zeros((self.directions.size, *scalar.shape))
        residual = jnp.asarray(jnp.inf)
        iterations = 0
        zero = jnp.zeros((chi.size,), dtype=scalar.dtype)
        for iteration in range(self.maximum_iterations):
            production = jnp.sum(nu_fission * scalar, axis=1)
            fission_source = production[:, None] * chi[None, :] / factor
            scattering = contract(
                "cij,cj->ci", self.scattering_cross_section, scalar, backend="jax"
            )
            next_scalar, angular = self._sweep(
                0.5 * (scattering + fission_source), zero, zero
            )
            next_production = jnp.sum(
                self.cell_widths[:, None] * nu_fission * next_scalar
            )
            previous_production = jnp.sum(self.cell_widths[:, None] * nu_fission * scalar)
            next_factor = factor * next_production / previous_production
            flux_error = jnp.linalg.norm(next_scalar - scalar) / jnp.maximum(
                jnp.linalg.norm(next_scalar), 1.0e-30
            )
            factor_error = jnp.abs(next_factor - factor) / jnp.maximum(
                jnp.abs(next_factor), 1.0e-30
            )
            residual = jnp.maximum(flux_error, factor_error)
            scalar = next_scalar / jnp.maximum(next_production, 1.0e-30)
            factor = next_factor
            iterations = iteration + 1
            if float(residual) <= self.relative_tolerance:
                break
        return SNTransportResult(
            scalar,
            angular,
            iterations,
            residual,
            jnp.isfinite(residual) & (residual <= self.relative_tolerance),
            factor,
        )


@dataclass(frozen=True, slots=True)
class DepletionResult:
    inventory: Array
    total_inventory: Array
    nonnegative: Array
    conservation_defect: Array
    plan_id: str


class BatemanDepletionPlan:
    """Linear nuclide inventory evolution with a native matrix-exponential action."""

    def __init__(
        self,
        transition_matrix: ArrayLike,
        /,
        *,
        conserved_weights: ArrayLike | None = None,
    ):
        matrix = np.asarray(transition_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("Depletion transition matrix must be non-empty and square.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Depletion transition matrix must be finite.")
        off_diagonal = matrix - np.diag(np.diag(matrix))
        if np.any(off_diagonal < 0.0) or np.any(np.diag(matrix) > 0.0):
            raise ValueError(
                "Depletion matrix must have nonnegative production and nonpositive loss."
            )
        weights = (
            np.ones((matrix.shape[0],))
            if conserved_weights is None
            else np.asarray(conserved_weights, dtype=np.float64)
        )
        if (
            weights.shape != (matrix.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "conserved_weights must be finite, positive, and match nuclides."
            )
        conservation_scale = np.maximum(
            1.0, np.linalg.norm(weights) * np.linalg.norm(matrix)
        )
        if not np.allclose(
            weights @ matrix,
            0.0,
            rtol=0.0,
            atol=512.0 * np.finfo(np.float64).eps * conservation_scale,
        ):
            raise ValueError("conserved_weights must be a left null vector.")
        self.transition_matrix = jnp.asarray(matrix)
        self.conserved_weights = jnp.asarray(weights)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bateman-depletion",
                "matrix": matrix.tolist(),
                "weights": weights.tolist(),
            }
        )

    def evolve(self, inventory: ArrayLike, duration_s: ArrayLike, /) -> DepletionResult:
        initial_host = np.asarray(inventory)
        duration_host = np.asarray(duration_s)
        if (
            initial_host.shape != self.conserved_weights.shape
            or np.any(~np.isfinite(initial_host))
            or np.any(initial_host < 0.0)
        ):
            raise ValueError(
                "Initial depletion inventory must be finite, nonnegative, and aligned."
            )
        if (
            duration_host.shape != ()
            or not np.isfinite(duration_host)
            or float(duration_host) <= 0.0
        ):
            raise ValueError("duration_s must be a finite positive scalar.")
        initial = jnp.asarray(initial_host)
        space = ArraySpace(initial.shape, dtype=initial.dtype)
        operator = DenseLinearOperator(
            self.transition_matrix,
            source=space,
            target=space,
        )
        evolved = matrix_exponential_action(
            operator, initial, jnp.asarray(duration_host)
        ).value
        total = self.conserved_weights @ evolved
        initial_total = self.conserved_weights @ initial
        defect = jnp.abs(total - initial_total) / jnp.maximum(
            jnp.abs(initial_total), 1.0e-30
        )
        return DepletionResult(
            evolved,
            total,
            jnp.all(evolved >= -1.0e-12),
            defect,
            self.plan_id,
        )


def decay_heat_w(
    inventory: ArrayLike,
    decay_constants_s_inv: ArrayLike,
    recoverable_energy_j: ArrayLike,
    /,
) -> Array:
    inventory_ = jnp.asarray(inventory)
    decay = jnp.asarray(decay_constants_s_inv)
    energy = jnp.asarray(recoverable_energy_j)
    if inventory_.shape != decay.shape or inventory_.shape != energy.shape:
        raise ValueError("Decay heat arrays must align.")
    if (
        np.any(~np.isfinite(np.asarray(inventory_)))
        or np.any(np.asarray(inventory_) < 0.0)
        or np.any(~np.isfinite(np.asarray(decay)))
        or np.any(np.asarray(decay) < 0.0)
        or np.any(~np.isfinite(np.asarray(energy)))
        or np.any(np.asarray(energy) < 0.0)
    ):
        raise ValueError("Decay heat inputs must be finite and nonnegative.")
    return jnp.sum(inventory_ * decay * energy)


def nuclear_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return exact unreleased transport and depletion candidate profiles."""

    specifications = (
        (
            "nuclear.transport.slab-sn-multigroup",
            {
                "geometry": "one-dimensional-slab",
                "angular": "gauss-legendre-sn",
                "energy": "multigroup",
                "boundary": "vacuum-or-prescribed-inflow",
            },
        ),
        (
            "nuclear.depletion.linear-bateman",
            {
                "inventory": "finite-nuclide-vector",
                "evolution": "matrix-exponential-action",
                "production": "linear-transition-matrix",
            },
        ),
    )
    gates = (
        "analytic-control",
        "balance",
        "refinement",
        "reference-comparison",
        "resource-envelope",
    )
    return tuple(
        CapabilityProfile(
            f"{capability}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(capability, attributes),),
            required_gates=gates,
        )
        for capability, attributes in specifications
    )


__all__ = [
    "nuclear_transport_candidate_profiles",
    "BatemanDepletionPlan",
    "DepletionResult",
    "SNTransportResult",
    "SlabSNTransportPlan",
    "decay_heat_w",
]
