#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Plan, preparation and transactional steps of hard-label threshold dynamics.

One step computes the kernel potentials ``psi_i = sum_j K_ij * u_j`` of the current
labels on per-site candidate-label slots (every active label on dense routes, the
brick-halo labels on the sparse route), assigns every site to its
``argmin psi`` candidate (or solves the count-constrained assignment by
capacitated auction), evaluates the Lyapunov energy of the candidate state, and
commits only when the step status allows it. A rolled-back step returns the input
state unchanged with its failure evidence. Label extinction is deterministic: a
label that loses all sites becomes inactive, never re-nucleates, and advances the
state epoch. Ties resolve to the lowest label index on every route.
"""

from __future__ import annotations

import math
from typing import assert_never, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from .._validation import finite_real_scalar, positive_integer
from ..combinatorial import CapacitatedAuctionPlan
from ..interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix
from ..sparse import KeyGroupPlan
from ..typing import AnyShape, as_host_array, Float64, HostInteger, parse, Scalar
from ._contracts import (
    _energy,
    _energy_tolerance,
    AbstractThresholdHeatKernel,
    decompose_threshold_kernel,
    HeatActionEvidence,
    LabelFieldState,
    SparseCandidateEvidence,
    ThresholdDynamicsEvidence,
    ThresholdDynamicsResourcePolicy,
    ThresholdDynamicsRunResult,
    ThresholdDynamicsStepResult,
    ThresholdKernelDecomposition,
    ThresholdKernelForm,
    ThresholdPotentials,
    VolumeConstraintEvidence,
)
from ._sparse import sparse_potentials, SparseLabelGrid
from ._status import _combine_status, _committed
from ._volume_constraints import LabelVolumeConstraint


ThresholdRoute: TypeAlias = AbstractThresholdHeatKernel | SparseLabelGrid

_RunCarry: TypeAlias = tuple[
    LabelFieldState,
    ThresholdPotentials,
    Array | None,
    Array,
    ThresholdDynamicsEvidence,
]


def _admissibility_failures(decomposition: ThresholdKernelDecomposition, /) -> list[str]:
    tension = decomposition.tension
    failures = []
    if not bool(tension.finite):
        failures.append("tension values are not finite")
    if not float(tension.minimum_pair_value) > 0.0:
        failures.append("every pair of distinct labels needs a positive tension")
    if not bool(tension.triangle_inequality):
        failures.append(
            "tensions violate the triangle inequality (excess "
            f"{float(tension.triangle_excess):.6g}), so the multiphase energy is not "
            "lower semicontinuous"
        )
    if not bool(decomposition.mobility.admissible):
        failures.append("mobilities must be finite and positive")
    if not bool(decomposition.consistent):
        failures.append(
            "the single-gaussian kernel needs one shared mu_ij * sigma_ij (relative "
            f"spread {float(decomposition.consistency_defect):.3g}); use "
            "kernel_form='two-gaussian'"
        )
    return failures

def _aggregate_heat_evidence(
    current: HeatActionEvidence, candidate: HeatActionEvidence, /
) -> HeatActionEvidence:
    """Retain worst diagnostics and total action work for both evaluations."""
    return HeatActionEvidence(
        successful=current.successful & candidate.successful,
        error_estimate=jnp.maximum(
            current.error_estimate, candidate.error_estimate
        ),
        native_status=jnp.maximum(current.native_status, candidate.native_status),
        converged=current.converged & candidate.converged,
        derivative_valid=current.derivative_valid & candidate.derivative_valid,
        iterations=jnp.maximum(current.iterations, candidate.iterations),
        setup_matvec_count=jnp.maximum(
            current.setup_matvec_count, candidate.setup_matvec_count
        ),
        action_matvec_count=current.action_matvec_count
        + candidate.action_matvec_count,
        transpose_matvec_count=jnp.maximum(
            current.transpose_matvec_count, candidate.transpose_matvec_count
        ),
        breakdown_status=jnp.maximum(
            current.breakdown_status, candidate.breakdown_status
        ),
        numeric_version=jnp.maximum(
            current.numeric_version, candidate.numeric_version
        ),
        method=current.method,
        exact=current.exact and candidate.exact,
        retained_storage_bytes=max(
            current.retained_storage_bytes, candidate.retained_storage_bytes
        ),
        workspace_bytes=max(current.workspace_bytes, candidate.workspace_bytes),
        operator_id=current.operator_id,
        prepared_id=current.prepared_id,
    )


def _aggregate_sparse_evidence(
    current: SparseCandidateEvidence, candidate: SparseCandidateEvidence, /
) -> SparseCandidateEvidence:
    """Expose every current/candidate overflow cause and worst capacity evidence."""
    return SparseCandidateEvidence(
        required_candidates=jnp.maximum(
            current.required_candidates, candidate.required_candidates
        ),
        candidate_overflow=current.candidate_overflow
        | candidate.candidate_overflow,
        overflowed_sites=jnp.maximum(
            current.overflowed_sites, candidate.overflowed_sites
        ),
        active_sites=jnp.maximum(current.active_sites, candidate.active_sites),
        active_label_count=jnp.maximum(
            current.active_label_count, candidate.active_label_count
        ),
        stencil_routes=current.stencil_routes + candidate.stencil_routes,
        truncated_kernel_mass=jnp.maximum(
            current.truncated_kernel_mass, candidate.truncated_kernel_mass
        ),
        kernel_symbol_minimum=jnp.minimum(
            current.kernel_symbol_minimum, candidate.kernel_symbol_minimum
        ),
        candidate_capacity=current.candidate_capacity,
    )


class ThresholdDynamicsPlan(StrictModule):
    """Physics of one threshold-dynamics system: tensions, mobilities and ``dt``.

    ``kernel_form="single-gaussian"`` is the Esedoglu–Otto scheme and requires one
    shared ``mu_ij * sigma_ij``; ``"two-gaussian"`` is the Salvador–Esedoglu scheme
    for arbitrary mobilities. Construction refuses tensions without the triangle
    inequality, zero tensions, nonpositive mobilities and inconsistent
    single-kernel mobilities. Tension, mobility and ``time_step`` stay dynamic
    leaves; they are re-certified inside every step.
    """

    __strict_contract__ = True

    tension: InterfaceTensionMatrix
    mobility: InterfaceMobilityMatrix
    time_step: Float64[Scalar] = fixed_field()
    volume_constraint: LabelVolumeConstraint | None
    resource_policy: ThresholdDynamicsResourcePolicy
    kernel_form: ThresholdKernelForm = eqx.field(static=True)
    minimum_resolution_ratio: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        tension: InterfaceTensionMatrix,
        mobility: InterfaceMobilityMatrix,
        time_step: float,
        /,
        *,
        kernel_form: ThresholdKernelForm = "single-gaussian",
        volume_constraint: LabelVolumeConstraint | None = None,
        resource_policy: ThresholdDynamicsResourcePolicy | None = None,
        minimum_resolution_ratio: float = 1.0,
    ) -> None:
        if not isinstance(tension, InterfaceTensionMatrix):
            raise TypeError("tension must be an InterfaceTensionMatrix.")
        if not isinstance(mobility, InterfaceMobilityMatrix):
            raise TypeError("mobility must be an InterfaceMobilityMatrix.")
        if mobility.label_ids != tension.label_ids:
            raise ValueError("tension and mobility must declare the same label_ids.")
        step = finite_real_scalar(time_step, "time_step")
        if step <= 0.0:
            raise ValueError("time_step must be positive.")
        form = parse(kernel_form, ThresholdKernelForm, "kernel_form")
        if volume_constraint is not None and not isinstance(
            volume_constraint, LabelVolumeConstraint
        ):
            raise TypeError("volume_constraint must be a LabelVolumeConstraint or None.")
        if (
            volume_constraint is not None
            and volume_constraint.label_count != tension.label_count
        ):
            raise ValueError("volume_constraint must bound every declared label.")
        policy = (
            ThresholdDynamicsResourcePolicy()
            if resource_policy is None
            else resource_policy
        )
        if not isinstance(policy, ThresholdDynamicsResourcePolicy):
            raise TypeError("resource_policy must be a ThresholdDynamicsResourcePolicy.")
        uniform = tension.structure == "uniform" and mobility.structure == "uniform"
        coefficient_bytes = policy.coefficient_bytes(
            tension.label_count, np.dtype(np.float64).itemsize, uniform=uniform
        )
        ratio = finite_real_scalar(minimum_resolution_ratio, "minimum_resolution_ratio")
        if ratio < 0.0:
            raise ValueError("minimum_resolution_ratio must be nonnegative.")
        failures = _admissibility_failures(
            decompose_threshold_kernel(
                tension, mobility, jnp.asarray(step, dtype=jnp.float64), form
            )
        )
        if failures:
            raise ValueError(
                "Inadmissible threshold-dynamics parameters: " + "; ".join(failures) + "."
            )
        self.tension = tension
        self.mobility = mobility
        self.time_step = jnp.asarray(step, dtype=jnp.float64)
        self.volume_constraint = volume_constraint
        self.resource_policy = policy
        self.kernel_form = form
        self.coefficient_bytes = coefficient_bytes
        self.minimum_resolution_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "threshold-dynamics-plan",
                "label_ids": list(tension.label_ids),
                "tension": tension.structure_id,
                "mobility": mobility.structure_id,
                "kernel_form": form,
                "volume_constraint": None
                if volume_constraint is None
                else {
                    "exact": volume_constraint.exact,
                    "epsilon_schedule": list(volume_constraint.epsilon_schedule),
                    "epsilon_scale": volume_constraint.epsilon_scale,
                    "maximum_rounds": volume_constraint.maximum_rounds,
                },
                "maximum_working_bytes": policy.maximum_working_bytes,
                "coefficient_bytes": coefficient_bytes,
                "minimum_resolution_ratio": ratio,
            }
        )

    @property
    def label_ids(self) -> tuple[str, ...]:
        return self.tension.label_ids

    @property
    def label_count(self) -> int:
        return self.tension.label_count

    def decomposition(self) -> ThresholdKernelDecomposition:
        """Kernel times, coefficients and admissibility for the current values."""
        return decompose_threshold_kernel(
            self.tension, self.mobility, self.time_step, self.kernel_form
        )

    def prepare(
        self, route: ThresholdRoute, /, *, candidate_capacity: int | None = None
    ) -> PreparedThresholdDynamics:
        """Bind the plan to a dense heat-kernel route or the sparse label grid."""
        return PreparedThresholdDynamics(
            self, route, candidate_capacity=candidate_capacity
        )


class PreparedThresholdDynamics(StrictModule):
    """Threshold dynamics bound to one site route.

    Dense heat-kernel routes carry every declared label as a candidate at every
    site; `SparseLabelGrid` carries ``candidate_capacity`` brick-halo candidates.
    The plan's resource policy admits the ``site x slot`` working storage before
    preparation. Volume constraints require equal site measure and solve one
    capacitated auction per step over the candidate slots.
    """

    plan: ThresholdDynamicsPlan
    route: ThresholdRoute
    candidate_groups: KeyGroupPlan | None
    auction: CapacitatedAuctionPlan | None
    site_measures: Array
    candidate_width: int = eqx.field(static=True)
    working_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        plan: ThresholdDynamicsPlan,
        route: ThresholdRoute,
        /,
        *,
        candidate_capacity: int | None = None,
    ) -> None:
        if not isinstance(plan, ThresholdDynamicsPlan):
            raise TypeError("plan must be a ThresholdDynamicsPlan.")
        labels = plan.label_count
        match route:
            case AbstractThresholdHeatKernel():
                if candidate_capacity is not None:
                    raise ValueError("Dense heat-kernel routes take every label.")
                width = labels
                storage = math.prod(route.site_shape)
                groups = None
            case SparseLabelGrid():
                if candidate_capacity is None:
                    raise ValueError("The sparse route needs a candidate_capacity.")
                width = positive_integer(candidate_capacity, "candidate_capacity")
                if width > labels:
                    raise ValueError("candidate_capacity cannot exceed the label count.")
                storage = route.storage_sites
                groups = KeyGroupPlan(
                    route.halo_size,
                    width,
                    labels - 1,
                    case_shape=(route.brick_count,),
                )
            case _:
                raise TypeError(
                    "route must be an AbstractThresholdHeatKernel or a SparseLabelGrid."
                )
        measures = route.site_measures()
        estimate = plan.resource_policy.admit(
            storage,
            width,
            measures.dtype.itemsize,
            coefficient_bytes=plan.coefficient_bytes,
        )
        constraint = plan.volume_constraint
        if constraint is not None and not route.equal_site_measure:
            raise ValueError(
                "Exact label volumes by capacitated auction require equal site "
                "measure; this route has unequal site measures."
            )
        sites = math.prod(route.site_shape)
        self.plan = plan
        self.route = route
        self.candidate_groups = groups
        self.auction = (
            None if constraint is None else constraint.auction_plan(sites, width)
        )
        self.site_measures = measures.reshape(-1)
        self.candidate_width = width
        self.working_bytes = estimate
        self.coefficient_bytes = plan.coefficient_bytes
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-threshold-dynamics",
                "plan": plan.plan_id,
                "route": route.route_id,
                "sites": route.site_id,
                "candidate_width": width,
            }
        )

    @property
    def dtype(self) -> jnp.dtype:
        return self.site_measures.dtype

    @property
    def site_shape(self) -> tuple[int, ...]:
        return self.route.site_shape

    @property
    def site_count(self) -> int:
        return self.site_measures.shape[0]

    def initial_state(
        self, labels: ArrayLike, /, *, time: float = 0.0
    ) -> LabelFieldState:
        """Validated state whose active labels are exactly the labels present."""
        host = as_host_array(labels, HostInteger[AnyShape], "labels")
        if host.shape != self.site_shape:
            raise ValueError(
                f"labels must have shape {self.site_shape}; got {host.shape}."
            )
        count = self.plan.label_count
        if np.any(host < 0) or np.any(host >= count):
            raise ValueError(f"labels must index the {count} declared labels.")
        present = np.bincount(host.reshape(-1), minlength=count) > 0
        return LabelFieldState(
            jnp.asarray(host, dtype=jnp.int32),
            jnp.asarray(present),
            jnp.zeros((), jnp.int32),
            jnp.asarray(finite_real_scalar(time, "time"), dtype=self.dtype),
            label_ids=self.plan.label_ids,
            route_id=self.route.route_id,
            prepared_id=self.prepared_id,
            site_id=self.route.site_id,
        )

    def potentials(self, state: LabelFieldState, /) -> ThresholdPotentials:
        """Kernel potentials of one state on its candidate-label slots."""
        state = self._validate_state(state)
        return self._potentials(state, self.plan.decomposition())

    def energy(self, state: LabelFieldState, /) -> Array:
        """Discrete Lyapunov energy, converging to ``sum sigma_ij |Gamma_ij|``."""
        return _energy(self.potentials(state).own, self.site_measures)

    def step(
        self, state: LabelFieldState, /, *, initial_prices: ArrayLike | None = None
    ) -> ThresholdDynamicsStepResult:
        """One transactional step; ``initial_prices`` warm-starts the auction."""
        state = self._validate_state(state)
        decomposition = self.plan.decomposition()
        potentials = self._potentials(state, decomposition)
        prices = self._prices(initial_prices)
        result, _, _ = self._advance(
            state, potentials, decomposition, prices, self._declared_bounds()
        )
        return result

    def run(self, state: LabelFieldState, steps: int, /) -> ThresholdDynamicsRunResult:
        """``steps`` transactional steps with one potential evaluation per step.

        Potentials of the committed state are carried into the next step and the
        auction is warm-started with the previous prices. The first step that does
        not commit halts the run.
        """
        state = self._validate_state(state)
        count = positive_integer(steps, "steps")
        decomposition = self.plan.decomposition()
        potentials = self._potentials(state, decomposition)
        initial_energy = _energy(potentials.own, self.site_measures)

        def advance(
            current: LabelFieldState, bundle: ThresholdPotentials, dual: Array | None
        ) -> tuple[ThresholdDynamicsStepResult, ThresholdPotentials, Array | None]:
            return self._advance(
                current, bundle, decomposition, dual, self._declared_bounds()
            )

        prices = self._prices(None)
        template = jax.eval_shape(advance, state, potentials, prices)[0].evidence
        blank = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), template)

        def body(
            carry: _RunCarry, _: None
        ) -> tuple[_RunCarry, ThresholdDynamicsEvidence]:
            current, bundle, dual, halted, last = carry

            def proceed() -> _RunCarry:
                result, bundle_next, dual_next = advance(current, bundle, dual)
                return (
                    result.state,
                    bundle_next,
                    dual_next,
                    ~result.committed,
                    result.evidence,
                )

            def hold() -> _RunCarry:
                frozen = eqx.tree_at(
                    lambda evidence: (
                        evidence.committed,
                        evidence.changed_sites,
                        evidence.extinct_labels,
                    ),
                    last,
                    (
                        jnp.zeros_like(last.committed),
                        jnp.zeros_like(last.changed_sites),
                        jnp.zeros_like(last.extinct_labels),
                    ),
                )
                return current, bundle, dual, halted, frozen

            updated = jax.lax.cond(halted, hold, proceed)
            return updated, updated[-1]

        initial: _RunCarry = (state, potentials, prices, jnp.asarray(False), blank)
        final, evidence = jax.lax.scan(body, initial, None, length=count)
        return ThresholdDynamicsRunResult(
            state=final[0], initial_energy=initial_energy, evidence=evidence
        )

    def _declared_bounds(self) -> tuple[Array, Array] | None:
        constraint = self.plan.volume_constraint
        if constraint is None:
            return None
        return constraint.lower_counts, constraint.upper_counts

    def _prices(self, initial_prices: ArrayLike | None, /) -> Array | None:
        if self.auction is None:
            if initial_prices is not None:
                raise ValueError("initial_prices require a volume constraint.")
            return None
        if initial_prices is None:
            return jnp.zeros((self.plan.label_count,), self.dtype)
        prices = jnp.asarray(initial_prices, dtype=self.dtype)
        if prices.shape != (self.plan.label_count,):
            raise ValueError("initial_prices must have one value per declared label.")
        return prices

    def _validate_state(self, state: LabelFieldState, /) -> LabelFieldState:
        """Refuse a foreign or invalid interpretation before numerical action."""
        if not isinstance(state, LabelFieldState):
            raise TypeError("state must be a LabelFieldState.")
        if state.label_ids != self.plan.label_ids:
            raise ValueError("state label_ids must exactly match the prepared label order.")
        if state.route_id != self.route.route_id:
            raise ValueError("state route_id does not match this prepared route.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("state prepared_id does not match this preparation.")
        if state.site_id != self.route.site_id:
            raise ValueError("state site_id does not match the prepared sites.")
        if state.labels.shape != self.site_shape:
            raise ValueError(
                f"state labels must have shape {self.site_shape}; got {state.labels.shape}."
            )
        if state.active_labels.shape != (self.plan.label_count,):
            raise ValueError(
                "state active_labels must have one entry per prepared label."
            )
        labels = state.labels
        invalid_range = jnp.any((labels < 0) | (labels >= self.plan.label_count))
        if isinstance(invalid_range, jax_core.Tracer):
            labels = eqx.error_if(
                labels,
                invalid_range,
                "state labels must index the prepared label_ids.",
            )
        elif bool(invalid_range):
            raise ValueError("state labels must index the prepared label_ids.")
        safe_labels = jnp.clip(labels.reshape(-1), 0, self.plan.label_count - 1)
        inactive_owner = jnp.any(~state.active_labels[safe_labels])
        if isinstance(inactive_owner, jax_core.Tracer):
            labels = eqx.error_if(
                labels,
                inactive_owner,
                "Every state label that owns sites must be active.",
            )
        elif bool(inactive_owner):
            raise ValueError("Every state label that owns sites must be active.")
        return eqx.tree_at(lambda item: item.labels, state, labels)

    def _potentials(
        self, state: LabelFieldState, decomposition: ThresholdKernelDecomposition, /
    ) -> ThresholdPotentials:
        match self.route:
            case AbstractThresholdHeatKernel():
                return self._dense_potentials(state, decomposition)
            case SparseLabelGrid():
                if self.candidate_groups is None:
                    raise RuntimeError("Sparse preparation lost its candidate groups.")
                return sparse_potentials(
                    self.route,
                    self.candidate_groups,
                    state.labels,
                    state.active_labels,
                    decomposition,
                )
            case _:
                assert_never(self.route)

    def _dense_potentials(
        self, state: LabelFieldState, decomposition: ThresholdKernelDecomposition, /
    ) -> ThresholdPotentials:
        route = self.route
        if not isinstance(route, AbstractThresholdHeatKernel):
            raise RuntimeError("Dense potentials need a heat-kernel route.")
        count = self.plan.label_count
        fields = jax.nn.one_hot(state.labels, count, dtype=self.dtype)
        kernels = 1 if decomposition.form == "single-gaussian" else 2
        psi, heat = route.combine(
            fields,
            decomposition.times[:kernels].astype(self.dtype),
            decomposition.coefficients[:kernels].astype(self.dtype),
        )
        values = psi.reshape((self.site_count, count))
        labels = state.labels.reshape(-1)
        valid = jnp.broadcast_to(state.active_labels[None, :], values.shape)
        return ThresholdPotentials(
            values=jnp.where(valid, values, jnp.inf),
            keys=jnp.broadcast_to(
                jnp.arange(count, dtype=jnp.int32)[None, :], values.shape
            ),
            valid=valid,
            own=jnp.take_along_axis(values, labels[:, None], axis=-1)[:, 0],
            heat=heat,
            sparse=None,
            kernel_admitted=jnp.asarray(True),
        )

    def _assign(
        self,
        state: LabelFieldState,
        potentials: ThresholdPotentials,
        prices: Array | None,
        counts_before: Array,
        bounds: tuple[Array, Array] | None,
        /,
    ) -> tuple[Array, VolumeConstraintEvidence | None, Array, Array | None]:
        """Candidate labels, volume evidence, energy gap allowance, next prices.

        ``bounds`` are integer ``(lower, upper)`` site counts; extinct labels are
        forced to zero count.
        """
        labels = state.labels.reshape(-1)
        if bounds is None or self.auction is None or prices is None:
            slot = jnp.argmin(potentials.values, axis=-1)
            candidate = jnp.take_along_axis(potentials.keys, slot[:, None], axis=-1)[:, 0]
            return candidate, None, jnp.zeros((), self.dtype), None
        zero = jnp.zeros_like(bounds[0])
        lower = jnp.where(state.active_labels, bounds[0], zero)
        upper = jnp.where(state.active_labels, bounds[1], zero)
        own = potentials.valid & (potentials.keys == labels[:, None])
        result = self.auction.solve(
            potentials.keys,
            potentials.valid,
            -jnp.where(potentials.valid, potentials.values, 0.0),
            lower,
            upper,
            initial_prices=prices,
            initial_slots=jnp.where(
                jnp.any(own, axis=1), jnp.argmax(own, axis=1), -1
            ).astype(jnp.int32),
        )
        success = result.success
        candidate = jnp.where(success, result.labels, labels)
        # An eps-optimal assignment can raise the linearized energy by at most its
        # duality gap: E(v) - E(u) <= sqrt(pi) m (sum psi_v - sum psi_u).
        allowance = jnp.where(
            success,
            jnp.sqrt(jnp.pi)
            * self.site_measures[0]
            * result.evidence.duality_gap.astype(self.dtype),
            0.0,
        )
        volume = VolumeConstraintEvidence(
            auction_status=result.status,
            auction_success=success,
            prices=result.prices,
            lower_counts=lower,
            upper_counts=upper,
            feasible_before=jnp.all(lower <= counts_before)
            & jnp.all(counts_before <= upper),
            auction=result.evidence,
        )
        next_prices = jnp.where(success, result.prices.astype(self.dtype), prices)
        return candidate, volume, allowance, next_prices

    def _advance(
        self,
        state: LabelFieldState,
        potentials: ThresholdPotentials,
        decomposition: ThresholdKernelDecomposition,
        prices: Array | None,
        bounds: tuple[Array, Array] | None,
        /,
    ) -> tuple[ThresholdDynamicsStepResult, ThresholdPotentials, Array | None]:
        labels = state.labels.reshape(-1)
        active = state.active_labels
        counts_before = state.label_counts()
        energy_before = _energy(potentials.own, self.site_measures)
        candidate, volume, allowance, next_prices = self._assign(
            state, potentials, prices, counts_before, bounds
        )
        candidate_state = eqx.tree_at(
            lambda item: item.labels,
            state,
            candidate.reshape(state.labels.shape),
        )
        after = self._potentials(candidate_state, decomposition)
        energy_after = _energy(after.own, self.site_measures)
        # Approximate kernels bound the potential error in max norm; the energy
        # comparison admits that error over the total site measure.
        heat_allowance = (
            jnp.sqrt(jnp.pi)
            * jnp.sum(self.site_measures)
            * (potentials.heat.error_estimate + after.heat.error_estimate)
        )
        tolerance = (
            _energy_tolerance(energy_before, energy_after) + allowance + heat_allowance
        )
        decreased = energy_after <= energy_before + tolerance
        dissipation = (
            decomposition.dissipation_admitted
            & potentials.kernel_admitted
            & after.kernel_admitted
        )
        volume_failed = jnp.asarray(False)
        if volume is not None:
            dissipation = dissipation & volume.feasible_before
            volume_failed = ~volume.auction_success
        sparse = None
        overflow = jnp.asarray(False)
        if potentials.sparse is not None and after.sparse is not None:
            sparse = _aggregate_sparse_evidence(potentials.sparse, after.sparse)
            overflow = sparse.candidate_overflow
        counts_after = jnp.bincount(candidate, length=self.plan.label_count).astype(
            jnp.int32
        )
        extinct = active & (counts_after == 0)
        resolution = jnp.sqrt(decomposition.times[0]) / self.route.resolution_length
        heat_successful = potentials.heat.successful & after.heat.successful
        status = _combine_status(
            nonfinite=~(jnp.isfinite(energy_before) & jnp.isfinite(energy_after)),
            heat_failed=~heat_successful,
            overflow=overflow,
            inadmissible=~decomposition.admissible,
            volume_failed=volume_failed,
            energy_increase=dissipation & ~decreased,
            under_resolved=resolution < self.plan.minimum_resolution_ratio,
        )
        committed = _committed(status)
        new_state = eqx.tree_at(
            lambda item: (
                item.labels,
                item.active_labels,
                item.epoch,
                item.time,
            ),
            state,
            (
                jnp.where(committed, candidate, labels).reshape(state.labels.shape),
                jnp.where(committed, active & ~extinct, active),
                state.epoch + (committed & jnp.any(extinct)).astype(jnp.int32),
                state.time
                + jnp.where(committed, self.plan.time_step, 0.0).astype(
                    state.time.dtype
                ),
            ),
        )
        heat = _aggregate_heat_evidence(potentials.heat, after.heat)
        evidence = ThresholdDynamicsEvidence(
            status=status,
            committed=committed,
            energy_before=energy_before,
            energy_after=energy_after,
            energy_tolerance=tolerance,
            energy_decreased=decreased,
            dissipation_admitted=dissipation,
            parameters_admissible=decomposition.admissible,
            resolution_ratio=resolution,
            kernel_times=decomposition.times,
            label_counts=jnp.where(committed, counts_after, counts_before),
            extinct_labels=extinct & committed,
            changed_sites=jnp.where(committed, jnp.sum(candidate != labels), 0).astype(
                jnp.int32
            ),
            heat=heat,
            volume=volume,
            sparse=sparse,
        )
        carried = jax.tree.map(
            lambda new, old: jnp.where(committed, new, old), after, potentials
        )
        return (
            ThresholdDynamicsStepResult(state=new_state, evidence=evidence),
            carried,
            next_prices,
        )


__all__ = ["PreparedThresholdDynamics", "ThresholdDynamicsPlan", "ThresholdRoute"]
