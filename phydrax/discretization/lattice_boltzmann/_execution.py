#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet


LatticeBoltzmannExecutionKind: TypeAlias = Literal["reference", "sharded", "fused"]
LatticeBoltzmannStep = Callable[
    [Array, Array, Array, Array, Any], "LatticeBoltzmannExecutionStep"
]


class LatticeBoltzmannExecutionStep(StrictModule):
    """One fail-closed-capable execution step with scalar diagnostic leaves."""

    candidate_populations: Array
    accepted_populations: Array
    successful: Array
    residual: Array
    work: Array
    diagnostics: Any


class LatticeBoltzmannExecutionProvenance(StrictModule, NonTrainableState):
    """Static identity of one JAX-native LBM realization."""

    execution_kind: LatticeBoltzmannExecutionKind = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    step_id: str = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_kind: LatticeBoltzmannExecutionKind,
        plan_id: str,
        step_id: str,
        lattice_id: str,
        step_count: int,
        /,
    ):
        if execution_kind not in ("reference", "sharded", "fused"):
            raise ValueError("Unknown LBM execution kind.")
        identifiers = tuple(str(value) for value in (plan_id, step_id, lattice_id))
        count = int(step_count)
        if any(not value for value in identifiers) or count <= 0:
            raise ValueError("LBM execution provenance requires complete identities.")
        self.execution_kind = execution_kind
        self.backend = "jax"
        self.plan_id, self.step_id, self.lattice_id = identifiers
        self.step_count = count
        self.realization_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-realization",
                "execution_kind": execution_kind,
                "backend": "jax",
                "plan": identifiers[0],
                "step": identifiers[1],
                "lattice": identifiers[2],
                "step_count": count,
            }
        )


class LatticeBoltzmannEquivalenceEvidence(StrictModule, NonTrainableState):
    """Observable reference/candidate agreement, including failures and diagnostics."""

    populations_equivalent: Array
    failures_equivalent: Array
    diagnostics_equivalent: Array
    equivalent: Array
    maximum_absolute_error: Array
    reference_plan_id: str = eqx.field(static=True)
    candidate_plan_id: str = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)


class LatticeBoltzmannRealizationResult(StrictModule, NonTrainableState):
    """Full deterministic realization plus execution and equivalence evidence."""

    final_populations: Array
    populations: Array
    valid: Array
    successful: Array
    successful_steps: Array
    residuals: Array
    work: Array
    diagnostics: Any
    provenance: LatticeBoltzmannExecutionProvenance
    equivalence: LatticeBoltzmannEquivalenceEvidence


def _validate_populations(
    populations: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    /,
) -> None:
    if not eqx.is_array(populations):
        raise TypeError("LBM execution accepts JAX arrays only.")
    if populations.ndim != velocity_set.dimension + 1:
        raise ValueError("LBM populations require one spatial axis per dimension.")
    if populations.shape[-1] != velocity_set.population_count:
        raise ValueError("LBM populations must use the trailing velocity-set Q axis.")
    if not jnp.issubdtype(populations.dtype, jnp.inexact):
        raise TypeError("LBM populations must have an inexact dtype.")


def _validate_scalar(name: str, value: Array, /, *, boolean: bool = False) -> None:
    if not eqx.is_array(value) or value.shape != ():
        raise TypeError(f"LBM execution {name} must be a scalar array.")
    if boolean and value.dtype != jnp.dtype(bool):
        raise TypeError(f"LBM execution {name} must be Boolean.")


def _validate_diagnostics(diagnostics: Any, /) -> None:
    leaves = jax.tree.leaves(diagnostics)
    if not leaves:
        raise ValueError("LBM execution diagnostics must contain scalar array leaves.")
    if any(not eqx.is_array(value) or value.shape != () for value in leaves):
        raise TypeError("Every LBM execution diagnostic leaf must be a scalar array.")


def _self_equivalence(plan_id: str, dtype: Any, /) -> LatticeBoltzmannEquivalenceEvidence:
    true = jnp.asarray(True)
    return LatticeBoltzmannEquivalenceEvidence(
        true,
        true,
        true,
        true,
        jnp.zeros((), dtype=dtype),
        plan_id,
        plan_id,
        0.0,
        0.0,
    )


def _realize_lattice_boltzmann(
    step: LatticeBoltzmannStep,
    velocity_set: LatticeBoltzmannVelocitySet,
    initial_populations: Array,
    *,
    step_count: int,
    step_size: Any,
    args: Any,
    t0: Any,
    plan_id: str,
    step_id: str,
    execution_kind: LatticeBoltzmannExecutionKind,
) -> LatticeBoltzmannRealizationResult:
    populations = initial_populations
    _validate_populations(populations, velocity_set)
    count = int(step_count)
    if count <= 0:
        raise ValueError("LBM realization step_count must be positive.")
    dt = jnp.asarray(step_size, dtype=populations.dtype)
    initial_time = jnp.asarray(t0, dtype=populations.dtype)
    _validate_scalar("step_size", dt)
    _validate_scalar("t0", initial_time)

    def advance(carry, step_index):
        state, previous_success = carry
        time = initial_time + step_index * dt
        result = step(step_index, time, state, dt, args)
        if not isinstance(result, LatticeBoltzmannExecutionStep):
            raise TypeError(
                "LBM step callbacks must return LatticeBoltzmannExecutionStep."
            )
        _validate_populations(result.candidate_populations, velocity_set)
        _validate_populations(result.accepted_populations, velocity_set)
        if (
            result.candidate_populations.shape != state.shape
            or result.accepted_populations.shape != state.shape
            or result.candidate_populations.dtype != state.dtype
            or result.accepted_populations.dtype != state.dtype
        ):
            raise ValueError(
                "LBM execution steps must preserve population shape and dtype."
            )
        _validate_scalar("successful", result.successful, boolean=True)
        _validate_scalar("residual", result.residual)
        _validate_scalar("work", result.work)
        _validate_diagnostics(result.diagnostics)
        accepted = jnp.where(previous_success, result.accepted_populations, state)
        successful = previous_success & result.successful
        return (accepted, successful), (
            accepted,
            successful,
            result.residual,
            result.work,
            result.diagnostics,
        )

    indices = jnp.arange(count, dtype=jnp.int32)
    (final, successful), outputs = jax.lax.scan(
        advance,
        (populations, jnp.asarray(True)),
        indices,
    )
    states, successful_steps, residuals, work, diagnostics = outputs
    trajectory = jnp.concatenate((populations[None, ...], states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), successful_steps), axis=0)
    provenance = LatticeBoltzmannExecutionProvenance(
        execution_kind,
        plan_id,
        step_id,
        velocity_set.lattice_id,
        count,
    )
    return LatticeBoltzmannRealizationResult(
        final,
        trajectory,
        valid,
        successful,
        successful_steps,
        residuals,
        work,
        diagnostics,
        provenance,
        _self_equivalence(plan_id, populations.real.dtype),
    )


def lattice_boltzmann_equivalence(
    reference: LatticeBoltzmannRealizationResult,
    candidate: LatticeBoltzmannRealizationResult,
    /,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> LatticeBoltzmannEquivalenceEvidence:
    """Compare complete realization behavior without erasing failure evidence."""

    if not isinstance(reference, LatticeBoltzmannRealizationResult) or not isinstance(
        candidate, LatticeBoltzmannRealizationResult
    ):
        raise TypeError("LBM equivalence requires two realization results.")
    relative = float(rtol)
    absolute = float(atol)
    if relative < 0.0 or absolute < 0.0:
        raise ValueError("LBM equivalence tolerances must be non-negative.")
    if reference.populations.shape != candidate.populations.shape:
        raise ValueError("LBM equivalence requires matching population trajectories.")
    population_error = jnp.max(
        jnp.abs(reference.populations - candidate.populations), initial=0.0
    )
    populations_equivalent = jnp.allclose(
        reference.populations,
        candidate.populations,
        rtol=relative,
        atol=absolute,
    ) & jnp.allclose(
        reference.residuals,
        candidate.residuals,
        rtol=relative,
        atol=absolute,
    )
    failures_equivalent = (
        jnp.array_equal(reference.valid, candidate.valid)
        & jnp.array_equal(reference.successful_steps, candidate.successful_steps)
        & jnp.array_equal(reference.successful, candidate.successful)
    )
    reference_diagnostics = jax.tree.leaves(reference.diagnostics)
    candidate_diagnostics = jax.tree.leaves(candidate.diagnostics)
    if jax.tree.structure(reference.diagnostics) != jax.tree.structure(
        candidate.diagnostics
    ):
        raise ValueError("LBM equivalence requires matching diagnostic structures.")
    diagnostics_equivalent = jnp.array_equal(reference.work, candidate.work)
    for left, right in zip(reference_diagnostics, candidate_diagnostics, strict=True):
        diagnostics_equivalent = diagnostics_equivalent & jnp.allclose(
            left, right, rtol=relative, atol=absolute
        )
    equivalent = populations_equivalent & failures_equivalent & diagnostics_equivalent
    return LatticeBoltzmannEquivalenceEvidence(
        populations_equivalent,
        failures_equivalent,
        diagnostics_equivalent,
        equivalent,
        population_error,
        reference.provenance.plan_id,
        candidate.provenance.plan_id,
        relative,
        absolute,
    )


def with_lattice_boltzmann_equivalence(
    result: LatticeBoltzmannRealizationResult,
    equivalence: LatticeBoltzmannEquivalenceEvidence,
    /,
) -> LatticeBoltzmannRealizationResult:
    if not isinstance(result, LatticeBoltzmannRealizationResult) or not isinstance(
        equivalence, LatticeBoltzmannEquivalenceEvidence
    ):
        raise TypeError("LBM result/equivalence types do not match.")
    return LatticeBoltzmannRealizationResult(
        result.final_populations,
        result.populations,
        result.valid,
        result.successful,
        result.successful_steps,
        result.residuals,
        result.work,
        result.diagnostics,
        result.provenance,
        equivalence,
    )


class ReferenceLatticeBoltzmannExecutionPlan(StrictModule, NonTrainableState):
    """Typed pure-JAX reference realization for a trailing-Q LBM step."""

    velocity_set: LatticeBoltzmannVelocitySet
    step: LatticeBoltzmannStep
    step_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        step: LatticeBoltzmannStep,
        /,
        *,
        step_id: str,
        backend: str = "jax",
    ):
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be a LatticeBoltzmannVelocitySet.")
        if not callable(step):
            raise TypeError("step must be callable.")
        identifier = str(step_id)
        if not identifier:
            raise ValueError("step_id must be non-empty.")
        if backend != "jax":
            raise ValueError("LBM execution supports only the JAX array backend.")
        self.velocity_set = velocity_set
        self.step = step
        self.step_id = identifier
        self.backend = "jax"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reference-lattice-boltzmann-execution",
                "backend": "jax",
                "lattice": velocity_set.lattice_id,
                "step": identifier,
            }
        )

    @classmethod
    def from_dynamics(cls, dynamics: Any, /) -> "ReferenceLatticeBoltzmannExecutionPlan":
        """Adapt the prepared dynamics result without changing its public contract."""

        from ._dynamics import PreparedLatticeBoltzmannDynamics

        if not isinstance(dynamics, PreparedLatticeBoltzmannDynamics):
            raise TypeError("dynamics must be PreparedLatticeBoltzmannDynamics.")

        def step(step_index, time, populations, step_size, args):
            result = dynamics.step_detailed(
                step_index, time, populations, step_size, args
            )
            return LatticeBoltzmannExecutionStep(
                result.candidate_state,
                result.accepted_state,
                result.successful,
                result.residual,
                result.work,
                result.diagnostics,
            )

        return cls(
            dynamics.discretization.velocity_set,
            step,
            step_id=dynamics.prepared_id,
        )

    def realize(
        self,
        initial_populations: Array,
        /,
        *,
        step_count: int,
        step_size: Any,
        args: Any = None,
        t0: Any = 0.0,
    ) -> LatticeBoltzmannRealizationResult:
        return _realize_lattice_boltzmann(
            self.step,
            self.velocity_set,
            initial_populations,
            step_count=step_count,
            step_size=step_size,
            args=args,
            t0=t0,
            plan_id=self.plan_id,
            step_id=self.step_id,
            execution_kind="reference",
        )


__all__ = [
    "LatticeBoltzmannEquivalenceEvidence",
    "LatticeBoltzmannExecutionKind",
    "LatticeBoltzmannExecutionProvenance",
    "LatticeBoltzmannExecutionStep",
    "LatticeBoltzmannRealizationResult",
    "LatticeBoltzmannStep",
    "ReferenceLatticeBoltzmannExecutionPlan",
    "lattice_boltzmann_equivalence",
    "with_lattice_boltzmann_equivalence",
]
