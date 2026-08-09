#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Callable

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..domain import ComponentSum, DomainFunction
from ..integration._api import IntegrationRealization, reduce
from ..integration._estimates import IntegrationEstimate, IntegrationProvenance
from ..integration._status import IntegrationStatus
from ..integration._targets import (
    ComponentTarget,
    DensityTarget,
    DiscreteMeasureTarget,
    ProbabilityTarget,
    WeightedSampleTarget,
)
from ._costs import AbstractGroundCost
from ._measure import _FiniteTransportMeasure, EventEncoder, lower_transport_measure
from ._status import TransportStatus


class SemidiscreteProblemProvenance(StrictModule):
    """Continuous source, finite target, cost, and fixed-realization identity."""

    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)
    integration_method: str = eqx.field(static=True)
    integration_target: str = eqx.field(static=True)
    realization: str = eqx.field(static=True)

    def __init__(
        self,
        source: str,
        target: str,
        cost: str,
        integration: IntegrationProvenance,
        /,
    ):
        self.source = str(source)
        self.target = str(target)
        self.cost = str(cost)
        self.integration_method = integration.method
        self.integration_target = integration.target
        self.realization = integration.realization


class SemidiscreteTransportProblem(StrictModule):
    """A density-to-finite-measure problem bound to one explicit realization.

    The density is never converted to a public or private empirical source measure.
    Every source reduction continues to execute through ``realization``.
    """

    source: DensityTarget
    realization: IntegrationRealization
    cost: AbstractGroundCost
    source_mass_estimate: IntegrationEstimate
    _target: _FiniteTransportMeasure
    source_encoder: EventEncoder | None = eqx.field(static=True)
    mass_tolerance: float = eqx.field(static=True)
    provenance: SemidiscreteProblemProvenance

    def __init__(
        self,
        source: DensityTarget,
        realization: IntegrationRealization,
        target: DiscreteMeasureTarget | WeightedSampleTarget,
        cost: AbstractGroundCost,
        /,
        *,
        source_encoder: EventEncoder | None = None,
        target_encoder: EventEncoder | None = None,
        mass_tolerance: float = 1e-8,
    ):
        if not isinstance(source, DensityTarget):
            raise TypeError("source must be a DensityTarget.")
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("realization must be an IntegrationRealization.")
        if realization.target is not source:
            raise ValueError(
                "realization.target must be the exact DensityTarget passed as source."
            )
        if realization.batch is None:
            raise ValueError(
                "Semidiscrete transport requires a materialized fixed integration batch."
            )
        if not isinstance(cost, AbstractGroundCost):
            raise TypeError("cost must be an AbstractGroundCost.")
        tolerance = float(mass_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("mass_tolerance must be finite and nonnegative.")
        domain = _density_domain(source)
        target_ = lower_transport_measure(
            target,
            encoder=target_encoder,
            name="target",
        )
        mass_estimate = reduce(
            DomainFunction(
                domain=domain,
                deps=domain.labels,
                func=_ConstantOne(),
            ),
            realization,
        )
        mass_value = _estimate_array(mass_estimate)
        if mass_value.shape != ():
            raise ValueError(
                "The density realization must describe one unbatched source measure."
            )
        self.source = source
        self.realization = realization
        self.cost = cost
        self.source_mass_estimate = mass_estimate
        self._target = target_
        self.source_encoder = source_encoder
        self.mass_tolerance = tolerance
        self.provenance = SemidiscreteProblemProvenance(
            "continuous-density",
            target_.provenance,
            cost.cost_id,
            mass_estimate.provenance,
        )

    @property
    def source_mass(self) -> Array:
        """Numerically integrated physical source mass."""
        return _estimate_array(self.source_mass_estimate).reshape(())

    @property
    def target_mass(self) -> Array:
        """Declared physical target mass."""
        return self._target.mass

    @property
    def target_probabilities(self) -> Array:
        return self._target.probabilities

    @property
    def target_weights(self) -> Array:
        """Physical finite-target weights, including explicit zero mask entries."""
        return self._target.physical_weights

    @property
    def target_mask(self) -> Array:
        return self._target.active

    @property
    def target_event_shape(self) -> tuple[int, ...]:
        return self._target.event_shape

    @property
    def num_target_atoms(self) -> int:
        return self._target.num_atoms

    @property
    def target_support(self) -> Array:
        """Finite support restored to its declared event shape."""
        shape = (self._target.num_atoms,) + self._target.event_shape
        return self._target.points.reshape(shape)

    @property
    def masses_compatible(self) -> Array:
        mass = self.source_mass
        return (
            jnp.isfinite(mass)
            & (mass > 0.0)
            & jnp.isclose(
                mass,
                self.target_mass,
                rtol=self.mass_tolerance,
                atol=self.mass_tolerance,
            )
        )

    def with_target_support(self, support: ArrayLike, /) -> SemidiscreteTransportProblem:
        """Return this declared problem at new support locations.

        Weights, masks, event shape, density realization, and provenance are retained.
        Inactive locations are represented canonically but are never optimized or used.
        """
        values = jnp.asarray(support, dtype=self._target.points.dtype)
        expected = (self._target.num_atoms,) + self._target.event_shape
        if values.shape != expected:
            raise ValueError(f"support must have shape {expected}; got {values.shape}.")
        canonical = values.reshape(self._target.points.shape)
        canonical = eqx.error_if(
            canonical,
            jnp.any(self._target.active[:, None] & ~jnp.isfinite(canonical)),
            "Active target support coordinates must be finite.",
        )
        canonical = jnp.where(self._target.active[:, None], canonical, 0.0)
        return eqx.tree_at(lambda problem: problem._target.points, self, canonical)


def semidiscrete_problem(
    source: DensityTarget,
    realization: IntegrationRealization,
    target: DiscreteMeasureTarget | WeightedSampleTarget,
    /,
    *,
    cost: AbstractGroundCost,
    source_encoder: EventEncoder | None = None,
    target_encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> SemidiscreteTransportProblem:
    """Construct density-to-discrete transport without empirical source lowering."""
    return SemidiscreteTransportProblem(
        source,
        realization,
        target,
        cost,
        source_encoder=source_encoder,
        target_encoder=target_encoder,
        mass_tolerance=mass_tolerance,
    )


class SemidiscreteTransportDiagnostics(StrictModule):
    """Transport-iteration diagnostics, separate from integration error."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    normalized_target_marginal_residual: Array
    physical_target_marginal_residual: Array
    dual_residual: Array
    primal_dual_gap: Array
    num_checks: Array
    residual_history: Array


class SemidiscreteIntegrationDiagnostics(StrictModule):
    """Status and numerical error from the fixed continuous reductions."""

    status: Array
    mass_status: Array
    marginal_status: Array
    objective_status: Array
    mass_error_estimate: Array
    normalized_marginal_error_estimate: Array
    physical_objective_error_estimate: Array
    mass_error_available: Array
    marginal_error_available: Array
    objective_error_available: Array
    mass_num_evaluations: Array
    marginal_num_evaluations: Array
    objective_num_evaluations: Array
    mass_error_kind: str | None = eqx.field(static=True)
    marginal_error_kind: str | None = eqx.field(static=True)
    objective_error_kind: str | None = eqx.field(static=True)
    provenance: IntegrationProvenance


class SemidiscreteTransportProvenance(StrictModule):
    """Explicit numerical provenance for a semidiscrete result."""

    method: str = eqx.field(static=True)
    ground_cost: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    integration_method: str = eqx.field(static=True)
    realization: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    fixed_realization: bool = eqx.field(static=True)
    common_random_numbers: bool = eqx.field(static=True)
    deterministic_replay: bool = eqx.field(static=True)

    def __init__(self, problem: SemidiscreteTransportProblem, /):
        self.method = "semidiscrete-entropic-dual"
        self.ground_cost = problem.provenance.cost
        self.source = problem.provenance.source
        self.target = problem.provenance.target
        self.integration_method = problem.provenance.integration_method
        self.realization = problem.provenance.realization
        self.approximation = "fixed-integration-realization"
        self.fixed_realization = True
        self.common_random_numbers = True
        self.deterministic_replay = True


class SemidiscreteTransportResult(StrictModule):
    """Entropic density-to-discrete solution with two error contracts."""

    problem: SemidiscreteTransportProblem
    target_potential: Array
    target_marginal: Array
    epsilon: Array
    source_mass: Array
    transport_cost: Array
    regularization: Array
    regularized_cost: Array
    dual_cost: Array
    diagnostics: SemidiscreteTransportDiagnostics
    integration_diagnostics: SemidiscreteIntegrationDiagnostics
    provenance: SemidiscreteTransportProvenance

    @property
    def converged(self) -> Array:
        return (
            (self.diagnostics.status == int(TransportStatus.CONVERGED))
            & (
                self.integration_diagnostics.status
                == int(IntegrationStatus.CONVERGED)
            )
        )

    @property
    def approximate(self) -> bool:
        """Semidiscrete results always retain fixed-realization approximation."""
        return True

    @property
    def integration_status(self) -> Array:
        return self.integration_diagnostics.status

    @property
    def integration_error_estimate(self) -> Array:
        return self.integration_diagnostics.physical_objective_error_estimate

    @property
    def integration_provenance(self) -> IntegrationProvenance:
        return self.integration_diagnostics.provenance

    @property
    def target_support(self) -> Array:
        return self.problem.target_support

    @property
    def target_weights(self) -> Array:
        return self.problem.target_weights

    @property
    def target_mask(self) -> Array:
        return self.problem.target_mask

    @property
    def target_event_shape(self) -> tuple[int, ...]:
        return self.problem.target_event_shape

    def soft_c_transform(self, source_points: Any, /) -> Array:
        """Evaluate the source soft c-transform induced by target potentials."""
        encoded = (
            source_points
            if self.problem.source_encoder is None
            else self.problem.source_encoder(source_points)
        )
        points = jnp.asarray(encoded, dtype=float)
        if points.ndim == 0:
            points = points.reshape((1, 1))
        elif points.ndim == 1:
            points = points.reshape((-1, self.problem._target.feature_size))
        elif points.ndim != 2:
            raise ValueError("source_points must contain scalar or vector points.")
        costs = self.problem.cost.matrix(points, self.problem._target.points)
        logits = (
            _safe_log(self.problem.target_probabilities)[None, :]
            + (self.target_potential[None, :] - costs) / self.epsilon
        )
        return -self.epsilon * jsp.special.logsumexp(logits, axis=-1)


class SemidiscreteSinkhorn(StrictModule):
    """Fixed-realization entropic semidiscrete dual solver."""

    epsilon: Array
    tolerance: Array
    max_iterations: int = eqx.field(static=True)
    min_iterations: int = eqx.field(static=True)
    check_every: int = eqx.field(static=True)
    early_stop: bool = eqx.field(static=True)
    store_history: bool = eqx.field(static=True)

    def __init__(
        self,
        epsilon: ArrayLike,
        /,
        *,
        max_iterations: int = 200,
        min_iterations: int = 1,
        tolerance: ArrayLike = 1e-7,
        check_every: int = 1,
        early_stop: bool = False,
        store_history: bool = False,
    ):
        maximum = int(max_iterations)
        minimum = int(min_iterations)
        interval = int(check_every)
        if maximum < 1:
            raise ValueError("max_iterations must be positive.")
        if minimum < 0 or minimum > maximum:
            raise ValueError("min_iterations must lie in [0, max_iterations].")
        if interval < 1:
            raise ValueError("check_every must be positive.")
        epsilon_ = jnp.asarray(epsilon, dtype=float).reshape(())
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
        self.epsilon = eqx.error_if(
            epsilon_,
            ~jnp.isfinite(epsilon_) | (epsilon_ <= 0.0),
            "epsilon must be finite and positive.",
        )
        self.tolerance = eqx.error_if(
            tolerance_,
            ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
            "tolerance must be finite and nonnegative.",
        )
        self.max_iterations = maximum
        self.min_iterations = minimum
        self.check_every = interval
        self.early_stop = bool(early_stop)
        self.store_history = bool(store_history)

    def __call__(
        self,
        problem: SemidiscreteTransportProblem,
        /,
        *,
        initial_target_potential: ArrayLike | None = None,
    ) -> SemidiscreteTransportResult:
        if not isinstance(problem, SemidiscreteTransportProblem):
            raise TypeError("problem must be a SemidiscreteTransportProblem.")
        count = problem.num_target_atoms
        dtype = jnp.result_type(problem._target.points, self.epsilon)
        if initial_target_potential is None:
            initial = jnp.zeros((count,), dtype=dtype)
        else:
            initial = jnp.asarray(initial_target_potential, dtype=dtype)
            if initial.shape != (count,):
                raise ValueError(
                    "initial_target_potential must match the target atom count."
                )
            initial = eqx.error_if(
                initial,
                jnp.any(~jnp.isfinite(initial)),
                "initial_target_potential must be finite.",
            )
        initial = jnp.where(problem.target_mask, initial, 0.0)
        epsilon = self.epsilon.astype(dtype)
        tolerance = self.tolerance.astype(dtype)
        source_mass = problem.source_mass.astype(dtype)
        mass_status = problem.source_mass_estimate.status
        initial_failed = (
            (mass_status != int(IntegrationStatus.CONVERGED))
            | ~jnp.isfinite(source_mass)
            | (source_mass <= 0.0)
        )
        initial_carry = (
            initial,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            initial_failed,
            jnp.asarray(mass_status, dtype=jnp.int32),
        )

        def step(carry, index):
            (
                potential,
                marginal_residual,
                dual_residual,
                first_converged,
                converged,
                failed,
                integration_status,
            ) = carry
            frozen = failed | (converged if self.early_stop else False)

            def update(_):
                estimate = _marginal_estimate(problem, potential, epsilon)
                marginal = _estimate_array(estimate) / source_mass
                active_log_target = jnp.log(
                    jnp.where(
                        problem.target_mask,
                        problem.target_probabilities,
                        1.0,
                    )
                )
                active_log_marginal = jnp.log(
                    jnp.where(problem.target_mask, marginal, 1.0)
                )
                candidate = jnp.where(
                    problem.target_mask,
                    potential
                    + epsilon * (active_log_target - active_log_marginal),
                    0.0,
                )
                candidate = jnp.where(
                    problem.target_mask,
                    candidate
                    - jnp.sum(problem.target_probabilities * candidate),
                    0.0,
                )
                finite = jnp.all(
                    jnp.where(problem.target_mask, jnp.isfinite(candidate), True)
                )
                estimate_ok = estimate.status == int(IntegrationStatus.CONVERGED)
                valid = finite & estimate_ok
                next_potential = jnp.where(valid, candidate, potential)
                residual = jnp.sum(
                    jnp.abs(marginal - problem.target_probabilities)
                )
                residual = jnp.where(valid, residual, jnp.inf)
                dual = jnp.max(jnp.abs(next_potential - potential)) / epsilon
                return (
                    next_potential,
                    residual,
                    dual,
                    ~valid,
                    estimate.status,
                )

            def keep(_):
                return (
                    potential,
                    marginal_residual,
                    dual_residual,
                    failed,
                    integration_status,
                )

            (
                next_potential,
                next_residual,
                next_dual_residual,
                next_failed,
                next_integration_status,
            ) = jax.lax.cond(frozen, keep, update, operand=None)
            iteration = index + 1
            should_check = (
                (iteration % self.check_every == 0)
                | (iteration == self.max_iterations)
                | (iteration == self.min_iterations)
            )
            eligible = (
                should_check
                & (iteration >= self.min_iterations)
                & (next_residual <= tolerance)
                & ~next_failed
            )
            next_first = jnp.where(
                (first_converged < 0) & eligible,
                iteration.astype(jnp.int32),
                first_converged,
            )
            next_converged = converged | eligible
            return (
                next_potential,
                next_residual,
                next_dual_residual,
                next_first,
                next_converged,
                next_failed,
                next_integration_status,
            ), next_residual

        final_carry, residuals = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(self.max_iterations, dtype=jnp.int32),
        )
        (
            target_potential,
            _,
            dual_residual,
            first_converged,
            _,
            iteration_failed,
            marginal_status,
        ) = final_carry
        objective_estimate = _statistics_estimate(
            problem,
            target_potential,
            epsilon,
        )
        statistics = _estimate_array(objective_estimate) / source_mass
        target_marginal = statistics[:count]
        transport_probability = statistics[count]
        kl_probability = statistics[count + 1]
        source_potential_mean = statistics[count + 2]
        plan_mass = jnp.sum(target_marginal)
        normalized_residual = jnp.sum(
            jnp.abs(target_marginal - problem.target_probabilities)
        )
        transport_cost = source_mass * transport_probability
        regularization = source_mass * epsilon * kl_probability
        regularized_cost = transport_cost + regularization
        dual_cost = source_mass * (
            source_potential_mean
            + jnp.sum(problem.target_probabilities * target_potential)
            - epsilon * (plan_mass - 1.0)
        )
        objective_finite = (
            jnp.all(jnp.isfinite(statistics))
            & jnp.isfinite(regularized_cost)
            & jnp.isfinite(dual_cost)
        )
        integration_status = jnp.where(
            mass_status != int(IntegrationStatus.CONVERGED),
            mass_status,
            jnp.where(
                marginal_status != int(IntegrationStatus.CONVERGED),
                marginal_status,
                objective_estimate.status,
            ),
        ).astype(jnp.int32)
        integration_failed = integration_status != int(IntegrationStatus.CONVERGED)
        final_converged = (
            (normalized_residual <= tolerance)
            & (self.max_iterations >= self.min_iterations)
            & ~iteration_failed
            & ~integration_failed
            & objective_finite
            & problem.masses_compatible
        )
        status = jnp.where(
            integration_failed,
            int(TransportStatus.INTEGRATION_FAILURE),
            jnp.where(
                ~problem.masses_compatible,
                int(TransportStatus.MASS_MISMATCH),
                jnp.where(
                    iteration_failed,
                    int(TransportStatus.NONFINITE_ITERATE),
                    jnp.where(
                        ~objective_finite,
                        int(TransportStatus.NONFINITE_OBJECTIVE),
                        jnp.where(
                            final_converged,
                            int(TransportStatus.CONVERGED),
                            int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        check_indices = tuple(
            index
            for index in range(self.max_iterations)
            if (index + 1) % self.check_every == 0
            or (index + 1) == self.max_iterations
            or (index + 1) == self.min_iterations
        )
        history = (
            residuals[jnp.asarray(check_indices, dtype=jnp.int32)]
            if self.store_history
            else jnp.empty((0,), dtype=dtype)
        )
        actual_iterations = jnp.where(
            self.early_stop & (first_converged >= 0),
            first_converged,
            self.max_iterations,
        ).astype(jnp.int32)
        mass_error, mass_error_available = _error_array(
            problem.source_mass_estimate,
            dtype,
        )
        objective_error, objective_error_available = _error_array(
            objective_estimate,
            dtype,
        )
        transport_diagnostics = SemidiscreteTransportDiagnostics(
            status=status,
            num_iterations=actual_iterations,
            first_converged_iteration=first_converged,
            normalized_target_marginal_residual=normalized_residual,
            physical_target_marginal_residual=source_mass * normalized_residual,
            dual_residual=dual_residual,
            primal_dual_gap=jnp.abs(regularized_cost - dual_cost),
            num_checks=jnp.asarray(len(check_indices), dtype=jnp.int32),
            residual_history=history,
        )
        integration_diagnostics = SemidiscreteIntegrationDiagnostics(
            status=integration_status,
            mass_status=mass_status,
            marginal_status=marginal_status,
            objective_status=objective_estimate.status,
            mass_error_estimate=mass_error,
            normalized_marginal_error_estimate=objective_error / source_mass,
            physical_objective_error_estimate=objective_error,
            mass_error_available=mass_error_available,
            marginal_error_available=objective_error_available,
            objective_error_available=objective_error_available,
            mass_num_evaluations=problem.source_mass_estimate.num_evaluations,
            marginal_num_evaluations=objective_estimate.num_evaluations,
            objective_num_evaluations=objective_estimate.num_evaluations,
            mass_error_kind=problem.source_mass_estimate.error_kind,
            marginal_error_kind=objective_estimate.error_kind,
            objective_error_kind=objective_estimate.error_kind,
            provenance=objective_estimate.provenance,
        )
        return SemidiscreteTransportResult(
            problem=problem,
            target_potential=target_potential,
            target_marginal=source_mass * target_marginal,
            epsilon=epsilon,
            source_mass=source_mass,
            transport_cost=transport_cost,
            regularization=regularization,
            regularized_cost=regularized_cost,
            dual_cost=dual_cost,
            diagnostics=transport_diagnostics,
            integration_diagnostics=integration_diagnostics,
            provenance=SemidiscreteTransportProvenance(problem),
        )


class SemidiscreteQuantizationDiagnostics(StrictModule):
    """Fixed-step outer support-optimization diagnostics."""

    num_steps: Array
    objective_history: Array
    final_objective: Array
    gradient_norm_history: Array
    support_displacement: Array
    final_transport_status: Array
    final_integration_status: Array
    constrained: bool = eqx.field(static=True)


class SemidiscreteQuantizationResult(StrictModule):
    """Optimized support with retained optimizer, transport, and integration state."""

    parameters: Array
    support: Array
    optimizer_state: Any
    transport: SemidiscreteTransportResult
    diagnostics: SemidiscreteQuantizationDiagnostics

    @property
    def converged(self) -> Array:
        return self.transport.converged


class SemidiscreteQuantizer(StrictModule):
    """Compose Optax with a fixed-realization semidiscrete design objective.

    ``support_transform`` maps unconstrained optimizer parameters into the physical
    sensor, particle, or collocation support. Bounds and other domain constraints
    are therefore parameterized compositionally rather than enforced by clipping.
    """

    solver: SemidiscreteSinkhorn
    optimizer: optax.GradientTransformation
    support_transform: Callable[[Array], Array] | None
    num_steps: int = eqx.field(static=True)

    def __init__(
        self,
        solver: SemidiscreteSinkhorn,
        optimizer: optax.GradientTransformation,
        /,
        *,
        num_steps: int,
        support_transform: Callable[[Array], Array] | None = None,
    ):
        if not isinstance(solver, SemidiscreteSinkhorn):
            raise TypeError("solver must be a SemidiscreteSinkhorn.")
        if not isinstance(optimizer, optax.GradientTransformation):
            raise TypeError("optimizer must be an Optax GradientTransformation.")
        steps = int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be positive.")
        if support_transform is not None and not callable(support_transform):
            raise TypeError("support_transform must be callable or None.")
        self.solver = solver
        self.optimizer = optimizer
        self.support_transform = support_transform
        self.num_steps = steps

    def physical_support(self, parameters: ArrayLike, /) -> Array:
        """Map outer optimizer coordinates to physical support coordinates."""
        values = jnp.asarray(parameters, dtype=float)
        return values if self.support_transform is None else self.support_transform(values)

    def objective(
        self,
        problem: SemidiscreteTransportProblem,
        parameters: ArrayLike,
        /,
    ) -> Array:
        """Differentiable regularized design objective on one frozen realization."""
        candidate = problem.with_target_support(self.physical_support(parameters))
        result = self.solver(candidate)
        return eqx.error_if(
            result.regularized_cost,
            ~result.converged,
            "Semidiscrete support optimization requires converged integration and transport.",
        )

    def __call__(
        self,
        problem: SemidiscreteTransportProblem,
        /,
        *,
        initial_parameters: ArrayLike | None = None,
    ) -> SemidiscreteQuantizationResult:
        if not isinstance(problem, SemidiscreteTransportProblem):
            raise TypeError("problem must be a SemidiscreteTransportProblem.")
        parameters = jnp.asarray(
            problem.target_support if initial_parameters is None else initial_parameters,
            dtype=float,
        )
        initial_support = problem.with_target_support(
            self.physical_support(parameters)
        ).target_support
        optimizer_state = self.optimizer.init(parameters)

        def step(carry, _):
            current, state = carry
            value, gradient = jax.value_and_grad(
                lambda candidate: self.objective(problem, candidate)
            )(current)
            updates, next_state = self.optimizer.update(gradient, state, current)
            next_parameters = optax.apply_updates(current, updates)
            gradient_norm = jnp.linalg.norm(gradient)
            return (next_parameters, next_state), (value, gradient_norm)

        (parameters, optimizer_state), (objectives, gradient_norms) = jax.lax.scan(
            step,
            (parameters, optimizer_state),
            xs=None,
            length=self.num_steps,
        )
        final_problem = problem.with_target_support(self.physical_support(parameters))
        support = final_problem.target_support
        final_transport = self.solver(final_problem)
        final_cost = eqx.error_if(
            final_transport.regularized_cost,
            ~final_transport.converged,
            "Semidiscrete support optimization ended at a nonconverged solve.",
        )
        diagnostics = SemidiscreteQuantizationDiagnostics(
            num_steps=jnp.asarray(self.num_steps, dtype=jnp.int32),
            objective_history=objectives,
            final_objective=final_cost,
            gradient_norm_history=gradient_norms,
            support_displacement=jnp.linalg.norm(support - initial_support),
            final_transport_status=final_transport.diagnostics.status,
            final_integration_status=final_transport.integration_status,
            constrained=self.support_transform is not None,
        )
        return SemidiscreteQuantizationResult(
            parameters=parameters,
            support=support,
            optimizer_state=optimizer_state,
            transport=final_transport,
            diagnostics=diagnostics,
        )


class _ConstantOne(StrictModule):
    def __call__(self, *coordinates: Array, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        return jnp.asarray(coordinates[0]).reshape((-1,))[0] * 0.0 + 1.0


class _MarginalIntegrand(StrictModule):
    target_points: Array
    target_probabilities: Array
    target_mask: Array
    target_potential: Array
    epsilon: Array
    cost: AbstractGroundCost
    encoder: EventEncoder | None = eqx.field(static=True)

    def __call__(self, *coordinates: Array, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        costs = _coordinate_costs(
            coordinates, self.encoder, self.cost, self.target_points
        )
        logits = _safe_log(self.target_probabilities) + (
            self.target_potential - costs
        ) / self.epsilon
        normalizer = jsp.special.logsumexp(logits, axis=-1)
        probabilities = jnp.exp(logits - normalizer[..., None])
        return jnp.where(self.target_mask, probabilities, 0.0)


class _StatisticsIntegrand(StrictModule):
    target_points: Array
    target_probabilities: Array
    target_mask: Array
    target_potential: Array
    epsilon: Array
    cost: AbstractGroundCost
    encoder: EventEncoder | None = eqx.field(static=True)

    def __call__(self, *coordinates: Array, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        costs = _coordinate_costs(
            coordinates, self.encoder, self.cost, self.target_points
        )
        logits = _safe_log(self.target_probabilities) + (
            self.target_potential - costs
        ) / self.epsilon
        log_normalizer = jsp.special.logsumexp(logits, axis=-1)
        probabilities = jnp.where(
            self.target_mask,
            jnp.exp(logits - log_normalizer[..., None]),
            0.0,
        )
        source_potential = -self.epsilon * log_normalizer
        log_ratio = (
            source_potential[..., None] + self.target_potential - costs
        ) / self.epsilon
        transport = jnp.sum(probabilities * costs, axis=-1)
        entropy = jnp.sum(
            probabilities * jnp.where(self.target_mask, log_ratio, 0.0),
            axis=-1,
        )
        return jnp.concatenate(
            (
                probabilities,
                jnp.stack((transport, entropy, source_potential), axis=-1),
            ),
            axis=-1,
        )


def _density_domain(source: DensityTarget, /):
    base = source.base
    if isinstance(base, ComponentTarget):
        if isinstance(base.component, ComponentSum):
            raise TypeError(
                "Semidiscrete density sources require one component, not ComponentSum."
            )
        return base.component.domain
    if isinstance(base, ProbabilityTarget):
        return base.probability
    raise TypeError(
        "Semidiscrete density sources require a component or probability base target."
    )


def _estimate_array(estimate: IntegrationEstimate, /) -> Array:
    value = estimate.value
    return jnp.asarray(value.data if isinstance(value, cx.Field) else value)


def _error_array(estimate: IntegrationEstimate, dtype: Any, /) -> tuple[Array, Array]:
    if estimate.error_estimate is None:
        return jnp.asarray(jnp.nan, dtype=dtype), jnp.asarray(False)
    return jnp.asarray(estimate.error_estimate, dtype=dtype).reshape(()), jnp.asarray(True)


def _coordinate_costs(
    coordinates: tuple[Any, ...],
    encoder: EventEncoder | None,
    cost: AbstractGroundCost,
    target_points: Array,
    /,
) -> Array:
    points, batch_shape = _encode_points(coordinates, encoder)
    flat_points = points.reshape((-1, points.shape[-1]))
    costs = cost.matrix(flat_points, target_points)
    return costs.reshape(batch_shape + (target_points.shape[0],))


def _encode_points(
    coordinates: tuple[Any, ...],
    encoder: EventEncoder | None,
    /,
) -> tuple[Array, tuple[int, ...]]:
    if len(coordinates) == 1 and isinstance(coordinates[0], tuple):
        axes = tuple(jnp.asarray(axis, dtype=float) for axis in coordinates[0])
        if not axes or any(axis.ndim != 1 for axis in axes):
            raise ValueError("Separable source coordinates must be nonempty rank-one axes.")
        raw = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        batch_shape = raw.shape[:-1]
        encoded = raw if encoder is None else encoder(raw)
        points = jnp.asarray(encoded, dtype=float)
        if points.shape[: len(batch_shape)] != batch_shape:
            raise ValueError(
                "source_encoder must preserve the integration batch shape."
            )
        if points.ndim == len(batch_shape):
            points = points[..., None]
        if points.ndim != len(batch_shape) + 1 or points.shape[-1] == 0:
            raise ValueError(
                "source_encoder must return a nonempty trailing coordinate axis."
            )
        return points, batch_shape

    raw: Any = coordinates[0] if len(coordinates) == 1 else coordinates
    if encoder is not None:
        return jnp.asarray(encoder(raw), dtype=float).reshape((-1,)), ()
    point = jnp.concatenate(
        tuple(
            jnp.asarray(coordinate, dtype=float).reshape((-1,))
            for coordinate in coordinates
        )
    )
    return point, ()


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


def _marginal_estimate(
    problem: SemidiscreteTransportProblem,
    target_potential: Array,
    epsilon: Array,
    /,
) -> IntegrationEstimate:
    domain = _density_domain(problem.source)
    integrand = DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_MarginalIntegrand(
            problem._target.points,
            problem.target_probabilities,
            problem.target_mask,
            target_potential,
            epsilon,
            problem.cost,
            problem.source_encoder,
        ),
    )
    return reduce(integrand, problem.realization)


def _statistics_estimate(
    problem: SemidiscreteTransportProblem,
    target_potential: Array,
    epsilon: Array,
    /,
) -> IntegrationEstimate:
    domain = _density_domain(problem.source)
    integrand = DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_StatisticsIntegrand(
            problem._target.points,
            problem.target_probabilities,
            problem.target_mask,
            target_potential,
            epsilon,
            problem.cost,
            problem.source_encoder,
        ),
    )
    return reduce(integrand, problem.realization)


__all__ = [
    "SemidiscreteIntegrationDiagnostics",
    "SemidiscreteProblemProvenance",
    "SemidiscreteQuantizationDiagnostics",
    "SemidiscreteQuantizationResult",
    "SemidiscreteQuantizer",
    "SemidiscreteSinkhorn",
    "SemidiscreteTransportDiagnostics",
    "SemidiscreteTransportProblem",
    "SemidiscreteTransportProvenance",
    "SemidiscreteTransportResult",
    "semidiscrete_problem",
]
