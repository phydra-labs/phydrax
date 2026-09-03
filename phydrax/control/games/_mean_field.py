#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate best-response evaluation against one supplied frozen law."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._strict import StrictModule
from ...domain import DomainFunction
from ...stochastic import (
    BSDEControlMode,
    BSDEEvaluation,
    BSDEPathBatch,
    BSDEQuadrature,
    EmpiricalMeanField,
    evaluate_bsde,
    evaluate_mean_field_bsde_control,
    MeanFieldBSDEControlAdapter,
    MeanFieldBSDEProblem,
)


FROZEN_LAW_BEST_RESPONSE = "FROZEN_LAW_BEST_RESPONSE"
MINIMUM_FROZEN_LAW_EFFECTIVE_SAMPLE_SIZE = 2.0


class FrozenLawBestResponseStatus(IntEnum):
    """Stable validity codes for frozen-law candidate evaluations."""

    SUCCESS = 0
    INVALID_LAW_EVIDENCE = 1
    LOW_EFFECTIVE_SAMPLE_SIZE = 2
    INVALID_BSDE_EVIDENCE = 3
    NONFINITE_HAMILTONIAN_EVIDENCE = 4


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class FrozenLawBestResponseProblem(StrictModule):
    """A control-adapted BSDE frozen against one explicitly supplied law.

    This wrapper adds provenance and claim boundaries only. It does not perform a
    mean-field fixed point, infer a population law, or alter the base BSDE.
    """

    base_problem: MeanFieldBSDEProblem
    adapter: MeanFieldBSDEControlAdapter
    supplied_law_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    law_consistency_evaluated: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    def __init__(
        self,
        base_problem: MeanFieldBSDEProblem,
        adapter: MeanFieldBSDEControlAdapter,
        /,
        *,
        supplied_law_id: str,
        problem_id: str,
    ):
        if not isinstance(base_problem, MeanFieldBSDEProblem):
            raise TypeError("base_problem must be a MeanFieldBSDEProblem.")
        if not isinstance(adapter, MeanFieldBSDEControlAdapter):
            raise TypeError("adapter must be a MeanFieldBSDEControlAdapter.")
        if base_problem.control_adapter is not adapter:
            raise ValueError(
                "base_problem must carry the supplied control adapter by identity."
            )
        if (
            adapter.output_shape != base_problem.output_shape
            or adapter.noise_shape != base_problem.noise_shape
        ):
            raise ValueError("adapter output/noise shapes must match base_problem.")
        self.base_problem = base_problem
        self.adapter = adapter
        self.supplied_law_id = _identifier(supplied_law_id, "supplied_law_id")
        self.problem_id = _identifier(problem_id, "problem_id")
        self.certificate_label = FROZEN_LAW_BEST_RESPONSE
        self.law_consistency_evaluated = False
        self.mean_field_game_equilibrium_claimed = False
        self.mean_field_control_optimum_claimed = False
        self.finite_population_game_claimed = False

    @property
    def mean_field(self) -> EmpiricalMeanField:
        """The exact empirical flow frozen into the base problem."""
        return self.base_problem.mean_field

    @property
    def flow_id(self) -> str:
        return self.mean_field.mean_field_id

    @property
    def process_id(self) -> str:
        return self.base_problem.process_id

    @property
    def support(self) -> tuple[float, float]:
        return self.mean_field.support

    @property
    def source_path_id(self) -> str | None:
        return self.mean_field.source_path_id


class FrozenLawHamiltonianEvidence(StrictModule):
    """Controls and Hamiltonian values produced by the public adapter paths."""

    selected_controls: Array
    values: Array
    finite: Array
    adapter_id: str = eqx.field(static=True)


class FrozenLawBestResponseResult(StrictModule):
    """Evidence for one candidate response to a supplied, unevaluated law flow."""

    problem: FrozenLawBestResponseProblem
    bsde_evaluation: BSDEEvaluation
    hamiltonian_evidence: FrozenLawHamiltonianEvidence
    law_snapshot_validity: Array
    law_particle_validity: Array
    law_weights: Array
    law_effective_sample_sizes: Array
    minimum_effective_sample_size: Array
    law_evidence_valid: Array
    effective_sample_size_sufficient: Array
    minimum_required_effective_sample_size: float = eqx.field(static=True)
    valid: Array
    status: Array
    supplied_law_id: str = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    support: tuple[float, float] = eqx.field(static=True)
    source_path_id: str | None = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    base_problem_id: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    law_consistency_evaluated: bool = eqx.field(static=True)
    best_response_optimality_evaluated: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    @property
    def base_problem(self) -> MeanFieldBSDEProblem:
        return self.problem.base_problem

    @property
    def adapter(self) -> MeanFieldBSDEControlAdapter:
        return self.problem.adapter

    @property
    def mean_field(self) -> EmpiricalMeanField:
        return self.problem.mean_field

    @property
    def selected_controls(self) -> Array:
        return self.hamiltonian_evidence.selected_controls

    @property
    def hamiltonian_values(self) -> Array:
        return self.hamiltonian_evidence.values

    @property
    def paths(self) -> BSDEPathBatch:
        return self.bsde_evaluation.paths


def _validate_evaluation_request(
    problem: FrozenLawBestResponseProblem,
    paths: BSDEPathBatch,
    value_predictor: Any,
    control_predictor: Any,
    control_mode: BSDEControlMode,
    quadrature: BSDEQuadrature,
    /,
) -> None:
    if not isinstance(problem, FrozenLawBestResponseProblem):
        raise TypeError("problem must be a FrozenLawBestResponseProblem.")
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("paths must be a BSDEPathBatch.")
    if not callable(value_predictor) and not isinstance(value_predictor, DomainFunction):
        raise TypeError("value_predictor must be callable or a DomainFunction.")
    if control_mode not in ("explicit", "autodiff"):
        raise ValueError("control_mode must be 'explicit' or 'autodiff'.")
    if quadrature not in ("left", "trapezoid"):
        raise ValueError("quadrature must be 'left' or 'trapezoid'.")
    if control_mode == "explicit" and control_predictor is None:
        raise ValueError("Explicit BSDE control requires control_predictor.")
    if (
        control_mode == "explicit"
        and not callable(control_predictor)
        and not isinstance(control_predictor, DomainFunction)
    ):
        raise TypeError("control_predictor must be callable or a DomainFunction.")
    base = problem.base_problem
    if base.control_adapter is not problem.adapter:
        raise ValueError("Frozen-law problem no longer carries its supplied adapter.")
    if paths.state_shape != base.state_shape or paths.noise_shape != base.noise_shape:
        raise ValueError("Path and frozen-law state/noise shapes do not match.")
    if paths.process_id != base.process_id:
        raise ValueError("Path and frozen-law process IDs do not match.")
    if (
        float(paths.times[0]) != problem.support[0]
        or float(paths.times[-1]) != problem.support[1]
    ):
        raise ValueError("Paths and frozen law must share time support.")


def solve_frozen_law_best_response(
    problem: FrozenLawBestResponseProblem,
    paths: BSDEPathBatch,
    value_predictor: Any,
    /,
    *,
    control_predictor: Any = None,
    control_mode: BSDEControlMode = "explicit",
    quadrature: BSDEQuadrature = "left",
    key: Key[Array, ""] = jr.key(0),
    minimum_effective_sample_size: float = MINIMUM_FROZEN_LAW_EFFECTIVE_SAMPLE_SIZE,
) -> FrozenLawBestResponseResult:
    """Evaluate a candidate response while holding the supplied law fixed.

    The public BSDE evaluator supplies residual evidence, and the public mean-field
    control evaluator maps each BSDE control to the physical Hamiltonian action.
    No outer law-consistency iteration or optimality proof is performed.
    """
    required_effective_sample_size = float(minimum_effective_sample_size)
    if (
        not math.isfinite(required_effective_sample_size)
        or required_effective_sample_size < 1.0
    ):
        raise ValueError("minimum_effective_sample_size must be finite and at least one.")
    _validate_evaluation_request(
        problem,
        paths,
        value_predictor,
        control_predictor,
        control_mode,
        quadrature,
    )
    base = problem.base_problem
    evaluation = evaluate_bsde(
        base.as_bsde_problem(),
        paths,
        value_predictor,
        control_predictor=control_predictor,
        control_mode=control_mode,
        quadrature=quadrature,
        key=key,
    )

    step_times = jnp.broadcast_to(
        paths.times[:-1], paths.sample_shape + (paths.num_steps,)
    ).reshape((-1,))
    state_index = (..., slice(None, -1), *([slice(None)] * len(base.state_shape)))
    value_index = (..., slice(None, -1), *([slice(None)] * len(base.output_shape)))
    step_states = paths.states[state_index].reshape((-1,) + base.state_shape)
    step_values = evaluation.values[value_index].reshape((-1,) + base.output_shape)
    step_bsde_controls = evaluation.controls.reshape(
        (-1,) + base.output_shape + base.noise_shape
    )
    selected_controls = jax.vmap(
        lambda time, state, value, control: evaluate_mean_field_bsde_control(
            base, time, state, value, control
        )
    )(step_times, step_states, step_values, step_bsde_controls).reshape(
        paths.sample_shape + (paths.num_steps,) + problem.adapter.control_shape
    )

    snapshots = jax.vmap(problem.mean_field.snapshot)(paths.times)
    snapshot_validity = snapshots.valid
    effective_sample_sizes = snapshots.effective_sample_size
    minimum_effective_sample_size = jnp.min(effective_sample_sizes)
    law_evidence_valid = (
        jnp.all(snapshot_validity)
        & jnp.all(problem.mean_field.valid)
        & jnp.all(jnp.isfinite(problem.mean_field.weights))
        & jnp.all(jnp.isfinite(effective_sample_sizes))
    )
    effective_sample_size_sufficient = (
        minimum_effective_sample_size >= required_effective_sample_size
    )
    bsde_valid = jnp.all(evaluation.valid_paths)
    hamiltonian_finite = jnp.all(jnp.isfinite(selected_controls)) & jnp.all(
        jnp.isfinite(evaluation.generator_values)
    )
    status = jnp.where(
        ~law_evidence_valid,
        int(FrozenLawBestResponseStatus.INVALID_LAW_EVIDENCE),
        jnp.where(
            ~effective_sample_size_sufficient,
            int(FrozenLawBestResponseStatus.LOW_EFFECTIVE_SAMPLE_SIZE),
            jnp.where(
                ~bsde_valid,
                int(FrozenLawBestResponseStatus.INVALID_BSDE_EVIDENCE),
                jnp.where(
                    ~hamiltonian_finite,
                    int(FrozenLawBestResponseStatus.NONFINITE_HAMILTONIAN_EVIDENCE),
                    int(FrozenLawBestResponseStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(FrozenLawBestResponseStatus.SUCCESS)
    hamiltonian_evidence = FrozenLawHamiltonianEvidence(
        selected_controls=selected_controls,
        values=evaluation.generator_values,
        finite=hamiltonian_finite,
        adapter_id=problem.adapter.adapter_id,
    )
    return FrozenLawBestResponseResult(
        problem=problem,
        bsde_evaluation=evaluation,
        hamiltonian_evidence=hamiltonian_evidence,
        law_snapshot_validity=snapshot_validity,
        law_particle_validity=problem.mean_field.valid,
        law_weights=problem.mean_field.weights,
        law_effective_sample_sizes=effective_sample_sizes,
        minimum_effective_sample_size=minimum_effective_sample_size,
        law_evidence_valid=law_evidence_valid,
        effective_sample_size_sufficient=effective_sample_size_sufficient,
        minimum_required_effective_sample_size=required_effective_sample_size,
        valid=valid,
        status=status,
        supplied_law_id=problem.supplied_law_id,
        flow_id=problem.flow_id,
        process_id=problem.process_id,
        support=problem.support,
        source_path_id=problem.source_path_id,
        adapter_id=problem.adapter.adapter_id,
        problem_id=problem.problem_id,
        base_problem_id=base.problem_id,
        certificate_label=FROZEN_LAW_BEST_RESPONSE,
        candidate_evaluation_only=True,
        law_consistency_evaluated=False,
        best_response_optimality_evaluated=False,
        mean_field_game_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        finite_population_game_claimed=False,
    )


__all__ = [
    "FROZEN_LAW_BEST_RESPONSE",
    "MINIMUM_FROZEN_LAW_EFFECTIVE_SAMPLE_SIZE",
    "FrozenLawBestResponseProblem",
    "FrozenLawBestResponseResult",
    "FrozenLawBestResponseStatus",
    "FrozenLawHamiltonianEvidence",
    "solve_frozen_law_best_response",
]
