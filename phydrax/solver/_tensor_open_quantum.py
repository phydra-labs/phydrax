#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""MPO generator contracts and CP-preserving local-channel LPDO evolution."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..tensor_network import (
    apply_mpo,
    ChainCompressionEvidence,
    LocallyPurifiedDensity,
    MatrixProductOperator,
    MatrixProductState,
    mpo_hermiticity_residual,
)
from ..tensor_network._mpo import adjoint_mpo, apply_mpo_exact
from ._purified_lindblad import apply_local_kraus_channel, LocalKrausChannel


class MPOHamiltonian(StrictModule):
    """A finite-chain Hamiltonian MPO with native contraction evidence."""

    operator: MatrixProductOperator
    hermiticity_residual: Array
    finite: Array
    hermitian: Array
    valid: Array
    tolerance: float = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)

    def __init__(self, operator: MatrixProductOperator, /, *, tolerance: float = 1e-8):
        if not isinstance(operator, MatrixProductOperator):
            raise TypeError("operator must be MatrixProductOperator.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Hamiltonian tolerance must be finite and nonnegative.")
        residual = mpo_hermiticity_residual(operator)
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in operator.tensors])
        )
        hermitian = jnp.isfinite(residual) & (residual <= tolerance_)
        self.operator = operator
        self.hermiticity_residual = residual
        self.finite = finite
        self.hermitian = hermitian
        self.valid = finite & hermitian
        self.tolerance = tolerance_
        self.hamiltonian_id = canonical_fingerprint(
            {
                "kind": "mpo-hamiltonian",
                "operator": operator.structure_id,
                "tolerance": tolerance_,
            }
        )


class MPOLindbladian(StrictModule):
    """An explicit Liouville-space MPO; no implicit dense construction occurs."""

    generator: MatrixProductOperator
    finite: Array
    square_liouville_factors: Array
    trace_annihilation_residual: Array
    trace_preserving_generator: Array
    valid: Array
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    lindbladian_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: MatrixProductOperator,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        if not isinstance(generator, MatrixProductOperator):
            raise TypeError("generator must be MatrixProductOperator.")
        if generator.output_dimensions != generator.input_dimensions:
            raise ValueError("Lindbladian MPO must be square.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Lindbladian tolerance must be finite and nonnegative.")
        physical: list[int] = []
        for dimension in generator.output_dimensions:
            root = round(dimension**0.5)
            if root * root != dimension:
                raise ValueError(
                    "Every Lindbladian local dimension must be a Hilbert-space square."
                )
            physical.append(root)
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in generator.tensors])
        )
        trace_vector = MatrixProductState(
            tuple(
                jnp.eye(dimension, dtype=generator.tensors[0].dtype).reshape(
                    (1, dimension * dimension, 1)
                )
                for dimension in physical
            ),
            precision=generator.precision,
        )
        trace_derivative = apply_mpo_exact(adjoint_mpo(generator), trace_vector)
        trace_residual = trace_derivative.norm()
        trace_preserving = jnp.isfinite(trace_residual) & (trace_residual <= tolerance_)
        square = jnp.asarray(True)
        self.generator = generator
        self.finite = finite
        self.square_liouville_factors = square
        self.trace_annihilation_residual = trace_residual
        self.trace_preserving_generator = trace_preserving
        self.valid = finite & square & trace_preserving
        self.physical_dimensions = tuple(physical)
        self.tolerance = tolerance_
        self.lindbladian_id = canonical_fingerprint(
            {
                "kind": "mpo-lindbladian",
                "generator": generator.structure_id,
                "physical_dimensions": tuple(physical),
                "tolerance": tolerance_,
            }
        )


class MPOLindbladianActionResult(StrictModule):
    derivative: MatrixProductState
    compression: ChainCompressionEvidence
    finite: Array
    valid: Array
    lindbladian_id: str = eqx.field(static=True)


def apply_mpo_lindbladian(
    lindbladian: MPOLindbladian,
    vectorized_density: MatrixProductState,
    /,
    *,
    maximum_bond_dimension: int,
) -> MPOLindbladianActionResult:
    """Apply an explicit Liouville MPO using native finite-chain contraction."""
    if not isinstance(lindbladian, MPOLindbladian) or not isinstance(
        vectorized_density, MatrixProductState
    ):
        raise TypeError("lindbladian/vectorized_density types are invalid.")
    if vectorized_density.physical_dimensions != lindbladian.generator.input_dimensions:
        raise ValueError("Vectorized density dimensions do not match the Lindbladian.")
    derivative, compression = apply_mpo(
        lindbladian.generator,
        vectorized_density,
        maximum_bond_dimension=int(maximum_bond_dimension),
    )
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in derivative.tensors])
    )
    return MPOLindbladianActionResult(
        derivative,
        compression,
        finite,
        lindbladian.valid & compression.valid & finite,
        lindbladian.lindbladian_id,
    )


class LPDOChannelEvolutionPlan(StrictModule):
    """Static ordered local-channel sweeps and explicit truncation budgets."""

    channels: tuple[LocalKrausChannel, ...]
    steps: int = eqx.field(static=True)
    maximum_purification_dimension: int = eqx.field(static=True)
    trace_preservation_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    maximum_discarded_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        channels: Sequence[LocalKrausChannel],
        /,
        *,
        steps: int,
        maximum_purification_dimension: int,
        trace_preservation_tolerance: float = 1e-8,
        trace_tolerance: float = 1e-6,
        maximum_discarded_weight: float = 1e-6,
    ):
        selected = tuple(channels)
        steps_ = int(steps)
        capacity = int(maximum_purification_dimension)
        tolerances = tuple(
            float(value)
            for value in (
                trace_preservation_tolerance,
                trace_tolerance,
                maximum_discarded_weight,
            )
        )
        if not selected or any(
            not isinstance(channel, LocalKrausChannel) for channel in selected
        ):
            raise ValueError("At least one LocalKrausChannel is required.")
        if steps_ < 1 or capacity < 1:
            raise ValueError("Evolution steps/capacity must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Evolution tolerances must be finite and nonnegative.")
        self.channels = selected
        self.steps = steps_
        self.maximum_purification_dimension = capacity
        self.trace_preservation_tolerance = tolerances[0]
        self.trace_tolerance = tolerances[1]
        self.maximum_discarded_weight = tolerances[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lpdo-channel-evolution-plan",
                "channels": tuple(channel.channel_id for channel in selected),
                "steps": steps_,
                "purification_capacity": capacity,
                "tolerances": tolerances,
            }
        )


class LPDOChannelEvolutionEvidence(StrictModule):
    channel_trace_preservation_residuals: Array
    trace_history: Array
    trace_residual_history: Array
    purification_discarded_weights: Array
    completely_positive_by_construction: Array
    trace_preserving_channels: Array
    positive_semidefinite_by_construction: Array
    trace_within_tolerance: Array
    truncation_within_budget: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class LPDOChannelEvolutionResult(StrictModule):
    final_state: LocallyPurifiedDensity
    evidence: LPDOChannelEvolutionEvidence
    plan_id: str = eqx.field(static=True)


def evolve_lpdo_local_channels(
    state: LocallyPurifiedDensity,
    plan: LPDOChannelEvolutionPlan,
    /,
) -> LPDOChannelEvolutionResult:
    """Execute every CP channel without trace normalization or representation switching."""
    if not isinstance(state, LocallyPurifiedDensity) or not isinstance(
        plan, LPDOChannelEvolutionPlan
    ):
        raise TypeError("state/plan types are invalid.")
    for channel in plan.channels:
        if not 0 <= channel.site < state.site_count:
            raise ValueError("A channel site is outside the LPDO.")
        if state.physical_dimensions[channel.site] != channel.kraus.shape[-1]:
            raise ValueError("A channel physical dimension differs from the LPDO.")
    channel_residuals = jnp.stack(
        [channel.completeness_residual() for channel in plan.channels]
    )
    trace_values: list[Array] = [state.raw_trace()]
    discarded: list[Array] = []
    current = state
    for _ in range(plan.steps):
        for channel in plan.channels:
            current, local = apply_local_kraus_channel(
                current,
                channel,
                maximum_purification_dimension=plan.maximum_purification_dimension,
            )
            discarded.append(local.truncation.discarded_weight)
        trace_values.append(current.raw_trace())
    traces = jnp.stack(trace_values)
    trace_residuals = jnp.abs(traces - traces[0])
    discarded_ = jnp.stack(discarded)
    cp = jnp.full(channel_residuals.shape, True)
    tp = jnp.isfinite(channel_residuals) & (
        channel_residuals <= plan.trace_preservation_tolerance
    )
    psd = jnp.full((plan.steps,), True)
    trace_within = jnp.isfinite(trace_residuals[-1]) & (
        trace_residuals[-1] <= plan.trace_tolerance
    )
    truncation_within = jnp.isfinite(jnp.sum(discarded_)) & (
        jnp.sum(discarded_) <= plan.maximum_discarded_weight
    )
    finite = (
        jnp.all(jnp.isfinite(traces))
        & jnp.all(jnp.isfinite(channel_residuals))
        & jnp.all(jnp.isfinite(discarded_))
        & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in current.tensors])
        )
    )
    evidence = LPDOChannelEvolutionEvidence(
        channel_residuals,
        traces,
        trace_residuals,
        discarded_,
        cp,
        tp,
        psd,
        trace_within,
        truncation_within,
        finite,
        finite & jnp.all(tp) & trace_within & truncation_within,
        plan.plan_id,
    )
    return LPDOChannelEvolutionResult(current, evidence, plan.plan_id)


def _lpdo_hilbert_schmidt_inner(
    left: LocallyPurifiedDensity,
    right: LocallyPurifiedDensity,
    /,
) -> Array:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError(
            "LPDO Hilbert--Schmidt contraction requires matching dimensions."
        )
    left_tensors = left.precision.accumulation(left.tensors)
    right_tensors = right.precision.accumulation(right.tensors)
    environment = jnp.ones(
        (1, 1, 1, 1),
        dtype=jnp.result_type(left_tensors[0], right_tensors[0]),
    )
    for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
        environment = ein.contract(
            "abcd,axkr,byks,cyqt,dxqu->rstu",
            environment,
            left_tensor,
            jnp.conj(left_tensor),
            right_tensor,
            jnp.conj(right_tensor),
        )
    return jnp.real(environment.reshape(()))


def _lpdo_hilbert_schmidt_distance(
    left: LocallyPurifiedDensity,
    right: LocallyPurifiedDensity,
    /,
) -> Array:
    squared = (
        _lpdo_hilbert_schmidt_inner(left, left)
        + _lpdo_hilbert_schmidt_inner(right, right)
        - 2.0 * _lpdo_hilbert_schmidt_inner(left, right)
    )
    return jnp.sqrt(jnp.maximum(squared, 0.0))


class LPDOSteadyStateResult(StrictModule):
    state: LocallyPurifiedDensity
    fixed_point_residual_history: Array
    trace_residual: Array
    converged: Array
    finite: Array
    valid: Array
    iterations: int = eqx.field(static=True)
    solve_id: str = eqx.field(static=True)


def solve_lpdo_steady_state(
    initial_state: LocallyPurifiedDensity,
    channels: Sequence[LocalKrausChannel],
    /,
    *,
    maximum_iterations: int,
    maximum_purification_dimension: int,
    convergence_tolerance: float = 1e-7,
    trace_tolerance: float = 1e-6,
    maximum_discarded_weight_per_sweep: float = 1e-6,
) -> LPDOSteadyStateResult:
    """Run an explicitly bounded fixed-point solve and report every sweep residual."""
    iterations = int(maximum_iterations)
    tolerance = float(convergence_tolerance)
    if iterations < 1 or not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Steady-state iteration/tolerance policy is invalid.")
    current = initial_state
    residuals: list[Array] = []
    all_valid: list[Array] = []
    for _ in range(iterations):
        before = current
        step = evolve_lpdo_local_channels(
            current,
            LPDOChannelEvolutionPlan(
                channels,
                steps=1,
                maximum_purification_dimension=maximum_purification_dimension,
                trace_tolerance=trace_tolerance,
                maximum_discarded_weight=maximum_discarded_weight_per_sweep,
            ),
        )
        current = step.final_state
        residuals.append(_lpdo_hilbert_schmidt_distance(before, current))
        all_valid.append(step.evidence.valid)
    residuals_ = jnp.stack(residuals)
    trace_residual = jnp.abs(current.raw_trace() - initial_state.raw_trace())
    finite = jnp.all(jnp.isfinite(residuals_)) & jnp.isfinite(trace_residual)
    converged = finite & (residuals_[-1] <= tolerance)
    solve_id = canonical_fingerprint(
        {
            "kind": "lpdo-steady-state-solve",
            "initial": initial_state.structure_id,
            "channels": tuple(channel.channel_id for channel in channels),
            "maximum_iterations": iterations,
            "purification_capacity": int(maximum_purification_dimension),
            "convergence_tolerance": tolerance,
        }
    )
    return LPDOSteadyStateResult(
        current,
        residuals_,
        trace_residual,
        converged,
        finite,
        finite & jnp.all(jnp.stack(all_valid)) & converged,
        iterations,
        solve_id,
    )


__all__ = [
    "LPDOChannelEvolutionEvidence",
    "LPDOChannelEvolutionPlan",
    "LPDOChannelEvolutionResult",
    "LPDOSteadyStateResult",
    "MPOHamiltonian",
    "MPOLindbladian",
    "MPOLindbladianActionResult",
    "apply_mpo_lindbladian",
    "evolve_lpdo_local_channels",
    "solve_lpdo_steady_state",
]
