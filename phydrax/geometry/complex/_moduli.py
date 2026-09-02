#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class TrainableHomogeneousHypersurface(StrictModule):
    """Fixed-support homogeneous polynomial with a declared projective gauge pivot."""

    exponents: Array
    coefficients: Array
    projective_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    pivot: int = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    pgl_slice: Array | None
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        exponents: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        pivot: int,
        pivot_tolerance: float = 1e-8,
        pgl_slice: ArrayLike | None = None,
        family_id: str,
    ):
        exponents_ = jnp.asarray(exponents, dtype=jnp.int32)
        coefficients_ = jnp.asarray(coefficients)
        if (
            exponents_.ndim != 2
            or coefficients_.shape != exponents_.shape[:1]
            or exponents_.shape[0] == 0
            or exponents_.shape[1] < 2
        ):
            raise ValueError("Homogeneous support/coefficients have incompatible shapes.")
        degrees = jnp.sum(exponents_, axis=1)
        if (
            bool(jnp.any(exponents_ < 0))
            or bool(jnp.any(degrees != degrees[0]))
            or int(degrees[0]) < 1
        ):
            raise ValueError(
                "Every monomial exponent must be nonnegative with one common positive degree."
            )
        if not jnp.issubdtype(coefficients_.dtype, jnp.complexfloating):
            raise TypeError(
                "Trainable homogeneous coefficients must use a complex dtype."
            )
        pivot_ = int(pivot)
        if (
            not 0 <= pivot_ < coefficients_.shape[0]
            or float(pivot_tolerance) <= 0.0
            or not family_id
        ):
            raise ValueError("Hypersurface pivot/tolerance/family_id are invalid.")
        if bool(jnp.abs(coefficients_[pivot_]) <= pivot_tolerance):
            raise ValueError(
                "Declared projective coefficient pivot is zero or ambiguous."
            )
        normalized = coefficients_ / coefficients_[pivot_]
        slice_ = (
            None
            if pgl_slice is None
            else jnp.asarray(pgl_slice, dtype=coefficients_.dtype)
        )
        if slice_ is not None and (
            slice_.ndim != 2 or slice_.shape[1] != coefficients_.shape[0]
        ):
            raise ValueError("PGL transverse slice must act on the coefficient vector.")
        self.exponents = exponents_
        self.coefficients = normalized
        self.projective_dimension = exponents_.shape[1] - 1
        self.degree = int(degrees[0])
        self.pivot = pivot_
        self.pivot_tolerance = float(pivot_tolerance)
        self.pgl_slice = slice_
        self.family_id = str(family_id)

    def with_coefficients(
        self, coefficients: ArrayLike, /
    ) -> "TrainableHomogeneousHypersurface":
        return TrainableHomogeneousHypersurface(
            self.exponents,
            coefficients,
            pivot=self.pivot,
            pivot_tolerance=self.pivot_tolerance,
            pgl_slice=self.pgl_slice,
            family_id=self.family_id,
        )

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        point = jnp.asarray(coordinates, dtype=self.coefficients.dtype)
        if point.shape[-1:] != (self.projective_dimension + 1,):
            raise ValueError("Homogeneous coordinates have the wrong trailing dimension.")
        monomials = jnp.prod(point[..., None, :] ** self.exponents, axis=-1)
        return jnp.sum(self.coefficients * monomials, axis=-1)

    def gradient(self, coordinates: ArrayLike, /) -> Array:
        point = jnp.asarray(coordinates, dtype=self.coefficients.dtype)
        if point.ndim == 1:
            return jax.jacfwd(self.__call__, holomorphic=True)(point)
        return jax.vmap(jax.jacfwd(self.__call__, holomorphic=True))(point)

    def slice_residual(self, /) -> Array:
        if self.pgl_slice is None:
            return jnp.asarray(0.0, dtype=self.coefficients.real.dtype)
        return jnp.max(jnp.abs(self.pgl_slice @ self.coefficients))


class HypersurfaceEpochEvidence(StrictModule):
    root_residuals: Array
    root_derivative_margins: Array
    pivot_margin: Array
    chart_valid: Array
    finite: Array
    valid: Array

    def __init__(
        self,
        *,
        root_residuals: ArrayLike,
        root_derivative_margins: ArrayLike,
        pivot_margin: ArrayLike,
        chart_valid: ArrayLike,
        finite: ArrayLike,
        valid: ArrayLike,
    ):
        self.root_residuals = jnp.asarray(root_residuals)
        self.root_derivative_margins = jnp.asarray(root_derivative_margins)
        self.pivot_margin = jnp.asarray(pivot_margin)
        self.chart_valid = jnp.asarray(chart_valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)


class PreparedHypersurfaceEpoch(StrictModule):
    """Fixed line/root/chart ancestry for one differentiable moduli epoch."""

    origins: Array
    directions: Array
    roots: Array
    chart_indices: Array
    root_tolerance: float = eqx.field(static=True)
    derivative_tolerance: float = eqx.field(static=True)
    newton_iterations: int = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        roots: ArrayLike,
        chart_indices: ArrayLike,
        /,
        *,
        root_tolerance: float = 1e-8,
        derivative_tolerance: float = 1e-8,
        newton_iterations: int = 8,
        epoch_id: str,
    ):
        origins_ = jnp.asarray(origins)
        directions_ = jnp.asarray(directions, dtype=origins_.dtype)
        roots_ = jnp.asarray(roots, dtype=origins_.dtype)
        charts = jnp.asarray(chart_indices, dtype=jnp.int32)
        if (
            origins_.ndim != 2
            or directions_.shape != origins_.shape
            or roots_.shape != origins_.shape[:1]
            or charts.shape != roots_.shape
        ):
            raise ValueError(
                "Hypersurface epoch line/root/chart arrays are incompatible."
            )
        if (
            min(float(root_tolerance), float(derivative_tolerance)) <= 0.0
            or int(newton_iterations) < 1
            or not epoch_id
        ):
            raise ValueError("Hypersurface epoch tolerances/iterations/id are invalid.")
        self.origins = origins_
        self.directions = directions_
        self.roots = roots_
        self.chart_indices = charts
        self.root_tolerance = float(root_tolerance)
        self.derivative_tolerance = float(derivative_tolerance)
        self.newton_iterations = int(newton_iterations)
        self.epoch_id = str(epoch_id)

    def points(self, roots: ArrayLike | None = None, /) -> Array:
        roots_ = (
            self.roots if roots is None else jnp.asarray(roots, dtype=self.roots.dtype)
        )
        return self.origins + roots_[:, None] * self.directions

    def continue_roots(
        self, hypersurface: TrainableHomogeneousHypersurface, /
    ) -> tuple[Array, HypersurfaceEpochEvidence]:
        roots = self.roots

        def line_value(root):
            return hypersurface(self.origins + root[:, None] * self.directions)

        for _ in range(self.newton_iterations):
            values = line_value(roots)
            complex_derivatives = jax.vmap(
                jax.jacfwd(
                    lambda root, origin, direction: hypersurface(
                        origin + root * direction
                    ),
                    holomorphic=True,
                )
            )(roots, self.origins, self.directions)
            roots = roots - values / complex_derivatives
        residuals = jnp.abs(line_value(roots))
        derivatives = jax.vmap(
            jax.jacfwd(
                lambda root, origin, direction: hypersurface(origin + root * direction),
                holomorphic=True,
            )
        )(roots, self.origins, self.directions)
        margins = jnp.abs(derivatives)
        pivot_margin = jnp.abs(hypersurface.coefficients[hypersurface.pivot])
        points = self.points(roots)
        chart_values = jnp.take_along_axis(
            jnp.abs(points), self.chart_indices[:, None], axis=1
        )[:, 0]
        chart_valid = jnp.all(chart_values > self.derivative_tolerance)
        finite = jnp.all(jnp.isfinite(roots)) & jnp.all(jnp.isfinite(residuals))
        valid = (
            finite
            & chart_valid
            & jnp.all(residuals <= self.root_tolerance)
            & jnp.all(margins > self.derivative_tolerance)
            & (pivot_margin > hypersurface.pivot_tolerance)
        )
        return roots, HypersurfaceEpochEvidence(
            root_residuals=residuals,
            root_derivative_margins=margins,
            pivot_margin=pivot_margin,
            chart_valid=chart_valid,
            finite=finite,
            valid=valid,
        )


class CalabiYauCertificate(StrictModule):
    exact_degree_hypothesis: Array
    nonzero_polynomial: Array
    cellular_cover_certified: Array
    gradient_lower_bound: Array
    transition_residual: Array
    residue_residual: Array
    metric_minimum_eigenvalue: Array
    volume_error_bound: Array
    monge_ampere_sup_bound: Array
    topology_certified: Array
    adjunction_conclusion: Array
    compactness_conclusion: Array
    completeness_conclusion: Array
    epsilon_candidate: Array
    ricci_flat_claim: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        projective_dimension: int,
        degree: int,
        nonzero_polynomial: ArrayLike,
        cellular_cover_certified: ArrayLike,
        gradient_lower_bound: ArrayLike,
        transition_residual: ArrayLike,
        residue_residual: ArrayLike,
        metric_minimum_eigenvalue: ArrayLike,
        volume_error_bound: ArrayLike,
        monge_ampere_sup_bound: ArrayLike,
        topology_certified: ArrayLike,
        tolerance: float,
    ):
        exact_degree = jnp.asarray(int(degree) == int(projective_dimension) + 1)
        nonzero = jnp.asarray(nonzero_polynomial, dtype=bool)
        cover = jnp.asarray(cellular_cover_certified, dtype=bool)
        gradient = jnp.asarray(gradient_lower_bound)
        transition = jnp.asarray(transition_residual)
        residue = jnp.asarray(residue_residual)
        metric = jnp.asarray(metric_minimum_eigenvalue)
        volume = jnp.asarray(volume_error_bound)
        monge = jnp.asarray(monge_ampere_sup_bound)
        topology = jnp.asarray(topology_certified, dtype=bool)
        smooth = cover & (gradient > float(tolerance))
        global_metric = (
            smooth
            & (transition <= tolerance)
            & (residue <= tolerance)
            & (metric > tolerance)
        )
        self.exact_degree_hypothesis = exact_degree
        self.nonzero_polynomial = nonzero
        self.cellular_cover_certified = cover
        self.gradient_lower_bound = gradient
        self.transition_residual = transition
        self.residue_residual = residue
        self.metric_minimum_eigenvalue = metric
        self.volume_error_bound = volume
        self.monge_ampere_sup_bound = monge
        self.topology_certified = topology
        self.adjunction_conclusion = exact_degree & nonzero & smooth
        self.compactness_conclusion = nonzero & smooth
        self.completeness_conclusion = self.compactness_conclusion & global_metric
        self.epsilon_candidate = (
            global_metric & jnp.isfinite(volume) & jnp.isfinite(monge)
        )
        self.ricci_flat_claim = False


class CalabiYauModuliProblem(StrictModule):
    objective: Callable[[TrainableHomogeneousHypersurface, Array], Array]
    epoch: PreparedHypersurfaceEpoch
    steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    backtracking_steps: int = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[TrainableHomogeneousHypersurface, Array], Array],
        epoch: PreparedHypersurfaceEpoch,
        /,
        *,
        steps: int,
        learning_rate: float,
        backtracking_steps: int = 8,
    ):
        if (
            not callable(objective)
            or int(steps) < 1
            or float(learning_rate) <= 0.0
            or int(backtracking_steps) < 1
        ):
            raise ValueError(
                "Moduli objective/steps/learning rate/backtracking are invalid."
            )
        self.objective = objective
        self.epoch = epoch
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.backtracking_steps = int(backtracking_steps)


class CalabiYauModuliResult(StrictModule):
    hypersurface: TrainableHomogeneousHypersurface
    roots: Array
    loss_history: Array
    accepted_steps: Array
    epoch_evidence: HypersurfaceEpochEvidence

    def __init__(
        self,
        hypersurface: TrainableHomogeneousHypersurface,
        roots: ArrayLike,
        loss_history: ArrayLike,
        accepted_steps: ArrayLike,
        epoch_evidence: HypersurfaceEpochEvidence,
        /,
    ):
        self.hypersurface = hypersurface
        self.roots = jnp.asarray(roots)
        self.loss_history = jnp.asarray(loss_history)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=bool)
        self.epoch_evidence = epoch_evidence


def solve_calabi_yau_moduli(
    problem: CalabiYauModuliProblem,
    initial: TrainableHomogeneousHypersurface,
    potential_parameters: ArrayLike,
    /,
) -> CalabiYauModuliResult:
    coefficients = initial.coefficients
    roots = problem.epoch.roots
    losses = []
    accepted = []
    evidence = problem.epoch.continue_roots(initial)[1]
    for _ in range(problem.steps):

        def loss(coefficient):
            family = initial.with_coefficients(coefficient)
            return jnp.real(problem.objective(family, jnp.asarray(potential_parameters)))

        current_loss, gradient = jax.value_and_grad(loss)(coefficients)
        candidate = coefficients
        candidate_roots = roots
        candidate_evidence = evidence
        accepted_step = jnp.asarray(False)
        rate = problem.learning_rate
        for _ in range(problem.backtracking_steps):
            proposed = initial.with_coefficients(coefficients - rate * jnp.conj(gradient))
            proposed_roots, proposed_evidence = problem.epoch.continue_roots(proposed)
            proposed_loss = loss(proposed.coefficients)
            take = (
                proposed_evidence.valid
                & jnp.isfinite(proposed_loss)
                & (proposed_loss <= current_loss)
            )
            candidate = jnp.where(take, proposed.coefficients, candidate)
            candidate_roots = jnp.where(take, proposed_roots, candidate_roots)
            candidate_evidence = jax.tree.map(
                lambda new, old: jnp.where(take, new, old),
                proposed_evidence,
                candidate_evidence,
            )
            accepted_step = accepted_step | take
            rate *= 0.5
        coefficients = candidate
        roots = candidate_roots
        evidence = candidate_evidence
        losses.append(loss(coefficients))
        accepted.append(accepted_step)
    return CalabiYauModuliResult(
        initial.with_coefficients(coefficients),
        roots,
        jnp.stack(losses),
        jnp.stack(accepted),
        evidence,
    )


__all__ = [
    "CalabiYauCertificate",
    "CalabiYauModuliProblem",
    "CalabiYauModuliResult",
    "HypersurfaceEpochEvidence",
    "PreparedHypersurfaceEpoch",
    "TrainableHomogeneousHypersurface",
    "solve_calabi_yau_moduli",
]
