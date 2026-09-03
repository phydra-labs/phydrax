#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-only mean-field-control planner evaluation.

The planner contract in this module is deliberately separate from the frozen-law
best-response and mean-field-game fixed-point contracts.  It augments a current
mean-field BSDE with the population externality terms in the planner adjoint; it
does not turn a stationary candidate into a global optimality claim.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

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
    MeanFieldBSDEProblem,
    MeanFieldSnapshot,
)


MEAN_FIELD_CONTROL_PLANNER_STATIONARITY = "MEAN_FIELD_CONTROL_PLANNER_STATIONARITY"
MINIMUM_MEAN_FIELD_CONTROL_EFFECTIVE_SAMPLE_SIZE = 2.0
MeanFieldExternalityMode: TypeAlias = Literal["analytic-lions", "finite-particle-adjoint"]


class MeanFieldControlStatus(IntEnum):
    """Stable evidence-validity codes for planner-stationarity evaluation."""

    SUCCESS = 0
    INVALID_LAW_EVIDENCE = 1
    LOW_EFFECTIVE_SAMPLE_SIZE = 2
    PATH_IDENTITY_MISMATCH = 3
    FINITE_PARTICLE_EVIDENCE_MISMATCH = 4
    INVALID_BSDE_EVIDENCE = 5
    NONFINITE_EXTERNALITY_EVIDENCE = 6
    NONFINITE_STATIONARITY_EVIDENCE = 7
    NONFINITE_WELFARE_EVIDENCE = 8


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _callback(value: Any, owner: str, /) -> Callable:
    if not callable(value):
        raise TypeError(f"{owner} must be callable.")
    return value


def _rms(values: Array, /) -> Array:
    return jnp.sqrt(jnp.mean(jnp.square(values)))


def _infinity_norm(values: Array, /) -> Array:
    return jnp.max(jnp.abs(values))


class MeanFieldExternality(StrictModule):
    """Explicit population contribution to the planner adjoint.

    ``running`` must return the population integral of the running Lions
    derivative at the supplied representative state.  ``terminal`` has the
    analogous terminal meaning.  Both outputs have the BSDE ``output_shape``.
    The callbacks use the signatures

    ``running(t, x, law, y, z, action, args)`` and
    ``terminal(x, law, args)``.

    In ``"analytic-lions"`` mode their distinct identifiers are mandatory and
    finite-particle evidence is forbidden.  In
    ``"finite-particle-adjoint"`` mode the callbacks are explicitly only a
    particle discretization: particle count, discretization identity, and a
    finite nonnegative bias bound are all mandatory.  Such an object never
    reports that it evaluated a Lions derivative.
    """

    running: Callable[[Array, Array, MeanFieldSnapshot, Array, Array, Array, Any], Array]
    terminal: Callable[[Array, MeanFieldSnapshot, Any], Array]
    bias_bound: float | None = eqx.field(static=True)
    particle_count: int | None = eqx.field(static=True)
    mode: MeanFieldExternalityMode = eqx.field(static=True)
    externality_id: str = eqx.field(static=True)
    running_id: str = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    analytic_lions_derivatives_supplied: bool = eqx.field(static=True)
    finite_particle_bias_audited: bool = eqx.field(static=True)

    def __init__(
        self,
        running: Callable,
        terminal: Callable,
        /,
        *,
        mode: MeanFieldExternalityMode,
        externality_id: str,
        running_id: str,
        terminal_id: str,
        particle_count: int | None = None,
        discretization_id: str | None = None,
        bias_bound: ArrayLike | None = None,
    ):
        running_callback = _callback(running, "running")
        terminal_callback = _callback(terminal, "terminal")
        if mode not in ("analytic-lions", "finite-particle-adjoint"):
            raise ValueError(
                "mode must be 'analytic-lions' or 'finite-particle-adjoint'."
            )

        if mode == "analytic-lions":
            if (
                particle_count is not None
                or discretization_id is not None
                or bias_bound is not None
            ):
                raise ValueError(
                    "analytic-lions mode must not carry finite-particle evidence."
                )
            resolved_particle_count = None
            resolved_discretization = None
            resolved_bias_bound = None
        else:
            if (
                not isinstance(particle_count, int)
                or isinstance(particle_count, bool)
                or particle_count <= 0
            ):
                raise ValueError(
                    "finite-particle-adjoint mode requires a positive particle_count."
                )
            if discretization_id is None:
                raise ValueError(
                    "finite-particle-adjoint mode requires discretization_id."
                )
            resolved_discretization = _identifier(discretization_id, "discretization_id")
            if bias_bound is None:
                raise ValueError(
                    "finite-particle-adjoint mode requires a finite bias_bound."
                )
            bias_value = jnp.asarray(bias_bound, dtype=float)
            if bias_value.shape != ():
                raise ValueError("bias_bound must be scalar.")
            resolved_bias_bound = float(bias_value)
            if not isfinite(resolved_bias_bound) or resolved_bias_bound < 0.0:
                raise ValueError("bias_bound must be finite and nonnegative.")
            resolved_particle_count = particle_count

        self.running = running_callback
        self.terminal = terminal_callback
        self.bias_bound = resolved_bias_bound
        self.particle_count = resolved_particle_count
        self.mode = mode
        self.externality_id = _identifier(externality_id, "externality_id")
        self.running_id = _identifier(running_id, "running_id")
        self.terminal_id = _identifier(terminal_id, "terminal_id")
        self.discretization_id = resolved_discretization
        self.analytic_lions_derivatives_supplied = mode == "analytic-lions"
        self.finite_particle_bias_audited = mode == "finite-particle-adjoint"


class MeanFieldControlProblem(StrictModule):
    """A social-planner candidate built on one current empirical-law BSDE.

    ``welfare_running(t, x, law, action, args)`` and
    ``welfare_terminal(x, law, args)`` return scalar social objective samples.
    ``hamiltonian_stationarity(t, x, law, y, z, action, args)`` returns one
    derivative with exactly the adapter's physical ``control_shape``.  The
    stationarity callback must use the planner adjoint represented by ``y, z``;
    the evaluator separately checks that this adjoint includes the declared
    measure externality.
    """

    base_problem: MeanFieldBSDEProblem
    externality: MeanFieldExternality
    welfare_running: Callable[[Array, Array, MeanFieldSnapshot, Array, Any], Array]
    welfare_terminal: Callable[[Array, MeanFieldSnapshot, Any], Array]
    hamiltonian_stationarity: Callable[
        [Array, Array, MeanFieldSnapshot, Array, Array, Array, Any], Array
    ]
    minimum_effective_sample_size: float = eqx.field(static=True)
    welfare_running_id: str = eqx.field(static=True)
    welfare_terminal_id: str = eqx.field(static=True)
    stationarity_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    global_optimality_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    def __init__(
        self,
        base_problem: MeanFieldBSDEProblem,
        externality: MeanFieldExternality,
        welfare_running: Callable,
        welfare_terminal: Callable,
        hamiltonian_stationarity: Callable,
        /,
        *,
        welfare_running_id: str,
        welfare_terminal_id: str,
        stationarity_id: str,
        problem_id: str,
        minimum_effective_sample_size: float = (
            MINIMUM_MEAN_FIELD_CONTROL_EFFECTIVE_SAMPLE_SIZE
        ),
    ):
        if not isinstance(base_problem, MeanFieldBSDEProblem):
            raise TypeError("base_problem must be a MeanFieldBSDEProblem.")
        if base_problem.control_adapter is None:
            raise ValueError(
                "base_problem must carry a MeanFieldBSDEControlAdapter so physical "
                "planner controls are explicit."
            )
        if not isinstance(externality, MeanFieldExternality):
            raise TypeError(
                "externality must be an explicit MeanFieldExternality; planner "
                "evaluation never drops the measure derivative."
            )
        minimum_ess = float(minimum_effective_sample_size)
        if not isfinite(minimum_ess) or minimum_ess <= 0.0:
            raise ValueError("minimum_effective_sample_size must be finite and positive.")

        self.base_problem = base_problem
        self.externality = externality
        self.welfare_running = _callback(welfare_running, "welfare_running")
        self.welfare_terminal = _callback(welfare_terminal, "welfare_terminal")
        self.hamiltonian_stationarity = _callback(
            hamiltonian_stationarity, "hamiltonian_stationarity"
        )
        self.minimum_effective_sample_size = minimum_ess
        self.welfare_running_id = _identifier(welfare_running_id, "welfare_running_id")
        self.welfare_terminal_id = _identifier(welfare_terminal_id, "welfare_terminal_id")
        self.stationarity_id = _identifier(stationarity_id, "stationarity_id")
        self.problem_id = _identifier(problem_id, "problem_id")
        self.certificate_label = MEAN_FIELD_CONTROL_PLANNER_STATIONARITY
        self.mean_field_game_equilibrium_claimed = False
        self.mean_field_control_optimum_claimed = False
        self.global_optimality_claimed = False
        self.finite_population_game_claimed = False

    @property
    def mean_field(self) -> EmpiricalMeanField:
        return self.base_problem.mean_field

    @property
    def adapter(self):
        adapter = self.base_problem.control_adapter
        if adapter is None:  # Constructor makes this unreachable after valid creation.
            raise RuntimeError("Planner control adapter is unavailable.")
        return adapter


class MeanFieldControlResult(StrictModule):
    """Planner stationarity, externality, welfare, and provenance evidence.

    ``valid`` means only that the requested evidence was finite, sufficiently
    supported, and evaluated on the exact paths inducing the empirical law.  It
    does not mean the residuals vanish and does not claim an MFC optimum, an MFG
    equilibrium, a frozen-law best response, or a finite-population result.
    """

    problem: MeanFieldControlProblem
    bsde_evaluation: BSDEEvaluation
    physical_controls: Array
    hamiltonian_stationarity: Array
    running_externality_contributions: Array
    terminal_externality_contributions: Array
    measure_adjoint_residuals: Array
    global_measure_adjoint_residual: Array
    terminal_residual: Array
    running_welfare_values: Array
    terminal_welfare_values: Array
    expected_running_welfare: Array
    running_welfare: Array
    terminal_welfare: Array
    welfare: Array
    stationarity_rms_norm: Array
    stationarity_infinity_norm: Array
    measure_adjoint_rms_norm: Array
    measure_adjoint_infinity_norm: Array
    terminal_rms_norm: Array
    terminal_infinity_norm: Array
    law_snapshot_validity: Array
    law_particle_validity: Array
    law_effective_sample_sizes: Array
    minimum_effective_sample_size: Array
    law_evidence_valid: Array
    effective_sample_size_sufficient: Array
    path_identity_valid: Array
    finite_particle_evidence_matches_law: Array
    finite_particle_bias_bound: Array
    valid: Array
    status: Array
    minimum_required_effective_sample_size: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    base_problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)
    source_path_id: str | None = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    externality_id: str = eqx.field(static=True)
    externality_mode: MeanFieldExternalityMode = eqx.field(static=True)
    running_externality_id: str = eqx.field(static=True)
    terminal_externality_id: str = eqx.field(static=True)
    finite_particle_count: int | None = eqx.field(static=True)
    finite_particle_discretization_id: str | None = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    planner_stationarity_evaluated: bool = eqx.field(static=True)
    analytic_lions_derivatives_evaluated: bool = eqx.field(static=True)
    finite_particle_adjoint_evaluated: bool = eqx.field(static=True)
    finite_particle_bias_audited: bool = eqx.field(static=True)
    frozen_law_best_response_claimed: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    global_optimality_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(MeanFieldControlStatus.SUCCESS)

    @property
    def paths(self) -> BSDEPathBatch:
        return self.bsde_evaluation.paths

    @property
    def bsde_controls(self) -> Array:
        """BSDE ``Z`` values, kept distinct from ``physical_controls``."""
        return self.bsde_evaluation.controls

    @property
    def mean_field(self) -> EmpiricalMeanField:
        return self.problem.mean_field


def _validate_request(
    problem: MeanFieldControlProblem,
    paths: BSDEPathBatch,
    value_predictor: Any,
    control_predictor: Any,
    control_mode: BSDEControlMode,
    quadrature: BSDEQuadrature,
    /,
) -> None:
    if not isinstance(problem, MeanFieldControlProblem):
        raise TypeError("problem must be a MeanFieldControlProblem.")
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
    if base.control_adapter is None:
        raise ValueError("Planner base problem no longer has a control adapter.")
    if paths.state_shape != base.state_shape or paths.noise_shape != base.noise_shape:
        raise ValueError("Path and planner BSDE state/noise shapes do not match.")
    if paths.process_id != base.process_id:
        raise ValueError("Path and planner BSDE process IDs do not match.")
    if (
        float(paths.times[0]) != problem.mean_field.support[0]
        or float(paths.times[-1]) != problem.mean_field.support[1]
    ):
        raise ValueError("Paths and planner empirical law must share time support.")


def _path_identity(problem: MeanFieldControlProblem, paths: BSDEPathBatch) -> Array:
    flow = problem.mean_field
    identifiers_match = flow.source_path_id is not None and (
        flow.source_path_id == paths.path_id
    )
    shapes_match = flow.sample_shape == paths.sample_shape
    if not shapes_match:
        arrays_match = jnp.asarray(False)
    else:
        arrays_match = (
            jnp.array_equal(flow.times, paths.times)
            & jnp.array_equal(flow.particles, paths.states)
            & jnp.array_equal(flow.valid, paths.valid)
        )
    return jnp.asarray(identifiers_match) & jnp.asarray(shapes_match) & arrays_match


def evaluate_mean_field_control_planner(
    problem: MeanFieldControlProblem,
    paths: BSDEPathBatch,
    value_predictor: Any,
    /,
    *,
    control_predictor: Any = None,
    control_mode: BSDEControlMode = "explicit",
    quadrature: BSDEQuadrature = "left",
    key: Key[Array, ""] = jr.key(0),
) -> MeanFieldControlResult:
    """Evaluate one social-planner candidate on its law-generating paths.

    The current BSDE residual is augmented by the running externality and its
    terminal target by the terminal externality.  Welfare is a weighted empirical
    left-rule estimate on the declared law; ``quadrature`` controls the composed
    BSDE evaluation and is recorded there.  The return value is evidence for the
    exact ``MEAN_FIELD_CONTROL_PLANNER_STATIONARITY`` contract only.
    """

    _validate_request(
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

    sample_count = paths.num_paths
    num_steps = paths.num_steps
    flat_states = paths.states.reshape((sample_count, num_steps + 1) + base.state_shape)
    flat_values = evaluation.values.reshape(
        (sample_count, num_steps + 1) + base.output_shape
    )
    flat_bsde_controls = evaluation.controls.reshape(
        (sample_count, num_steps) + base.output_shape + base.noise_shape
    )

    step_times = jnp.broadcast_to(
        paths.times[:-1], paths.sample_shape + (num_steps,)
    ).reshape((-1,))
    step_states = flat_states[:, :-1].reshape((-1,) + base.state_shape)
    step_values = flat_values[:, :-1].reshape((-1,) + base.output_shape)
    step_bsde_controls = flat_bsde_controls.reshape(
        (-1,) + base.output_shape + base.noise_shape
    )
    physical_controls = jax.vmap(
        lambda time, state, value, control: evaluate_mean_field_bsde_control(
            base, time, state, value, control
        )
    )(step_times, step_states, step_values, step_bsde_controls).reshape(
        (sample_count, num_steps) + problem.adapter.control_shape
    )

    stationarity_steps = []
    externality_steps = []
    welfare_steps = []
    for step in range(num_steps):
        time = paths.times[step]
        snapshot = problem.mean_field.snapshot(time)

        def stationarity_at_path(state, value, bsde_control, action):
            output = jnp.asarray(
                problem.hamiltonian_stationarity(
                    time,
                    state,
                    snapshot,
                    value,
                    bsde_control,
                    action,
                    base.args,
                )
            )
            if output.shape != problem.adapter.control_shape:
                raise ValueError(
                    "hamiltonian_stationarity returned an incompatible shape."
                )
            return output

        def externality_at_path(state, value, bsde_control, action):
            output = jnp.asarray(
                problem.externality.running(
                    time,
                    state,
                    snapshot,
                    value,
                    bsde_control,
                    action,
                    base.args,
                )
            )
            if output.shape != base.output_shape:
                raise ValueError("running externality returned an incompatible shape.")
            return output

        def welfare_at_path(state, action):
            output = jnp.asarray(
                problem.welfare_running(time, state, snapshot, action, base.args)
            )
            if output.shape != ():
                raise ValueError("welfare_running must return a scalar.")
            return output

        states = flat_states[:, step]
        values = flat_values[:, step]
        bsde_controls = flat_bsde_controls[:, step]
        actions = physical_controls[:, step]
        stationarity_steps.append(
            jax.vmap(stationarity_at_path)(states, values, bsde_controls, actions)
        )
        externality_steps.append(
            jax.vmap(externality_at_path)(states, values, bsde_controls, actions)
        )
        welfare_steps.append(jax.vmap(welfare_at_path)(states, actions))

    flat_stationarity = jnp.stack(stationarity_steps, axis=1)
    flat_running_externality = jnp.stack(externality_steps, axis=1)
    flat_running_welfare = jnp.stack(welfare_steps, axis=1)
    terminal_snapshot = problem.mean_field.snapshot(paths.times[-1])

    def terminal_externality_at_path(state):
        output = jnp.asarray(
            problem.externality.terminal(state, terminal_snapshot, base.args)
        )
        if output.shape != base.output_shape:
            raise ValueError("terminal externality returned an incompatible shape.")
        return output

    def terminal_welfare_at_path(state):
        output = jnp.asarray(
            problem.welfare_terminal(state, terminal_snapshot, base.args)
        )
        if output.shape != ():
            raise ValueError("welfare_terminal must return a scalar.")
        return output

    terminal_states = flat_states[:, -1]
    flat_terminal_externality = jax.vmap(terminal_externality_at_path)(terminal_states)
    flat_terminal_welfare = jax.vmap(terminal_welfare_at_path)(terminal_states)

    stationarity = flat_stationarity.reshape(
        paths.sample_shape + (num_steps,) + problem.adapter.control_shape
    )
    running_externality = flat_running_externality.reshape(
        paths.sample_shape + (num_steps,) + base.output_shape
    )
    terminal_externality = flat_terminal_externality.reshape(
        paths.sample_shape + base.output_shape
    )
    running_welfare_values = flat_running_welfare.reshape(
        paths.sample_shape + (num_steps,)
    )
    terminal_welfare_values = flat_terminal_welfare.reshape(paths.sample_shape)
    physical_controls = physical_controls.reshape(
        paths.sample_shape + (num_steps,) + problem.adapter.control_shape
    )

    dt = jnp.diff(paths.times)
    dt_event_shape = (
        (1,) * len(paths.sample_shape) + (num_steps,) + (1,) * len(base.output_shape)
    )
    externality_increments = running_externality * dt.reshape(dt_event_shape)
    measure_adjoint_residuals = evaluation.local_residuals + externality_increments
    global_measure_adjoint_residual = evaluation.global_residual + jnp.sum(
        externality_increments, axis=len(paths.sample_shape)
    )
    terminal_residual = evaluation.terminal_residual - terminal_externality

    snapshots = jax.vmap(problem.mean_field.snapshot)(paths.times)
    snapshot_validity = snapshots.valid
    effective_sample_sizes = snapshots.effective_sample_size
    minimum_effective_sample_size = jnp.min(effective_sample_sizes)
    law_evidence_valid = (
        jnp.all(snapshot_validity)
        & jnp.all(problem.mean_field.valid)
        & jnp.all(jnp.isfinite(problem.mean_field.particles))
        & jnp.all(jnp.isfinite(problem.mean_field.weights))
        & jnp.all(jnp.isfinite(effective_sample_sizes))
    )
    effective_sample_size_sufficient = (
        minimum_effective_sample_size >= problem.minimum_effective_sample_size
    )
    path_identity_valid = _path_identity(problem, paths)

    externality = problem.externality
    finite_particle_evidence_matches_law = jnp.asarray(
        externality.mode == "analytic-lions"
        or externality.particle_count == problem.mean_field.num_particles
    )
    finite_particle_bias_bound = jnp.asarray(
        jnp.nan if externality.bias_bound is None else externality.bias_bound,
        dtype=float,
    )

    flat_snapshot_weights = snapshots.weights
    expected_running_welfare = jnp.sum(
        flat_snapshot_weights[:-1] * flat_running_welfare.T,
        axis=-1,
    )
    running_welfare = jnp.sum(expected_running_welfare * dt)
    terminal_welfare = jnp.sum(flat_snapshot_weights[-1] * flat_terminal_welfare)
    welfare = running_welfare + terminal_welfare

    bsde_valid = jnp.all(evaluation.valid_paths)
    externality_finite = (
        jnp.all(jnp.isfinite(running_externality))
        & jnp.all(jnp.isfinite(terminal_externality))
        & jnp.all(jnp.isfinite(measure_adjoint_residuals))
        & jnp.all(jnp.isfinite(terminal_residual))
    )
    stationarity_finite = jnp.all(jnp.isfinite(stationarity))
    welfare_finite = (
        jnp.all(jnp.isfinite(running_welfare_values))
        & jnp.all(jnp.isfinite(terminal_welfare_values))
        & jnp.isfinite(welfare)
    )
    status = jnp.where(
        ~law_evidence_valid,
        int(MeanFieldControlStatus.INVALID_LAW_EVIDENCE),
        jnp.where(
            ~effective_sample_size_sufficient,
            int(MeanFieldControlStatus.LOW_EFFECTIVE_SAMPLE_SIZE),
            jnp.where(
                ~path_identity_valid,
                int(MeanFieldControlStatus.PATH_IDENTITY_MISMATCH),
                jnp.where(
                    ~finite_particle_evidence_matches_law,
                    int(MeanFieldControlStatus.FINITE_PARTICLE_EVIDENCE_MISMATCH),
                    jnp.where(
                        ~bsde_valid,
                        int(MeanFieldControlStatus.INVALID_BSDE_EVIDENCE),
                        jnp.where(
                            ~externality_finite,
                            int(MeanFieldControlStatus.NONFINITE_EXTERNALITY_EVIDENCE),
                            jnp.where(
                                ~stationarity_finite,
                                int(
                                    MeanFieldControlStatus.NONFINITE_STATIONARITY_EVIDENCE
                                ),
                                jnp.where(
                                    ~welfare_finite,
                                    int(
                                        MeanFieldControlStatus.NONFINITE_WELFARE_EVIDENCE
                                    ),
                                    int(MeanFieldControlStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(MeanFieldControlStatus.SUCCESS)

    return MeanFieldControlResult(
        problem=problem,
        bsde_evaluation=evaluation,
        physical_controls=physical_controls,
        hamiltonian_stationarity=stationarity,
        running_externality_contributions=running_externality,
        terminal_externality_contributions=terminal_externality,
        measure_adjoint_residuals=measure_adjoint_residuals,
        global_measure_adjoint_residual=global_measure_adjoint_residual,
        terminal_residual=terminal_residual,
        running_welfare_values=running_welfare_values,
        terminal_welfare_values=terminal_welfare_values,
        expected_running_welfare=expected_running_welfare,
        running_welfare=running_welfare,
        terminal_welfare=terminal_welfare,
        welfare=welfare,
        stationarity_rms_norm=_rms(stationarity),
        stationarity_infinity_norm=_infinity_norm(stationarity),
        measure_adjoint_rms_norm=_rms(measure_adjoint_residuals),
        measure_adjoint_infinity_norm=_infinity_norm(measure_adjoint_residuals),
        terminal_rms_norm=_rms(terminal_residual),
        terminal_infinity_norm=_infinity_norm(terminal_residual),
        law_snapshot_validity=snapshot_validity,
        law_particle_validity=problem.mean_field.valid,
        law_effective_sample_sizes=effective_sample_sizes,
        minimum_effective_sample_size=minimum_effective_sample_size,
        law_evidence_valid=law_evidence_valid,
        effective_sample_size_sufficient=effective_sample_size_sufficient,
        path_identity_valid=path_identity_valid,
        finite_particle_evidence_matches_law=finite_particle_evidence_matches_law,
        finite_particle_bias_bound=finite_particle_bias_bound,
        valid=valid,
        status=status,
        minimum_required_effective_sample_size=problem.minimum_effective_sample_size,
        problem_id=problem.problem_id,
        base_problem_id=base.problem_id,
        process_id=base.process_id,
        path_id=paths.path_id,
        flow_id=problem.mean_field.mean_field_id,
        source_path_id=problem.mean_field.source_path_id,
        adapter_id=problem.adapter.adapter_id,
        externality_id=externality.externality_id,
        externality_mode=externality.mode,
        running_externality_id=externality.running_id,
        terminal_externality_id=externality.terminal_id,
        finite_particle_count=externality.particle_count,
        finite_particle_discretization_id=externality.discretization_id,
        certificate_label=MEAN_FIELD_CONTROL_PLANNER_STATIONARITY,
        candidate_evaluation_only=True,
        planner_stationarity_evaluated=True,
        analytic_lions_derivatives_evaluated=(
            externality.analytic_lions_derivatives_supplied
        ),
        finite_particle_adjoint_evaluated=(externality.mode == "finite-particle-adjoint"),
        finite_particle_bias_audited=externality.finite_particle_bias_audited,
        frozen_law_best_response_claimed=False,
        mean_field_game_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        global_optimality_claimed=False,
        finite_population_game_claimed=False,
    )


__all__ = [
    "MEAN_FIELD_CONTROL_PLANNER_STATIONARITY",
    "MINIMUM_MEAN_FIELD_CONTROL_EFFECTIVE_SAMPLE_SIZE",
    "MeanFieldControlProblem",
    "MeanFieldControlResult",
    "MeanFieldControlStatus",
    "MeanFieldExternality",
    "MeanFieldExternalityMode",
    "evaluate_mean_field_control_planner",
]
