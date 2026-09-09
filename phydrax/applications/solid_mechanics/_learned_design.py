#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.design import DesignState
from ...optim._iterative._types import Bounds, OptimizationTermination
from ...optim._pde_constrained import (
    AbstractStateDesignMethod,
    solve_state_design,
    StateDesignConstraint,
    StateDesignProblem,
    StateDesignResult,
)
from ...optim._pde_constrained_mma import ReducedMMA
from ...optim._state_design_parameterization import (
    reparameterize_state_design,
    StateDesignParameterization,
)
from ...transport.diffusion._guidance import (
    AbstractScoreGuidance,
    GuidanceEvaluation,
    ScoreContext,
)
from ._topology import (
    TopologyContinuationStageEvidence,
    TopologyMechanicsProblem,
    TopologyOptimizationResult,
)
from ._topology_reanalysis import (
    reanalyse_topology_design,
    TopologyReanalysisPlan,
    TopologyReanalysisReport,
)
from ._topology_state import FiniteElementStateSolver, NeuralVariationalStateSolver


class _FixedDensityDecoder(StrictModule, NonTrainableState):
    decode: Callable
    design_mask: Array
    fixed_density: Array

    def __call__(self, latent):
        density = self.decode(latent)
        if not eqx.is_array(density) or density.shape != self.fixed_density.shape:
            raise ValueError("Density decoder must return the prepared cell array shape.")
        if density.dtype != self.fixed_density.dtype:
            raise TypeError("Density decoder must preserve the prepared density dtype.")
        # This is the declared fixed-region map, not a repair of invalid geometry.
        return jnp.where(self.design_mask, density, self.fixed_density)


class _DensityCoordinate(StrictModule):
    index: int = eqx.field(static=True)

    def __call__(self, state, density, args):
        del state, args
        return density[self.index]


class LearnedTopologyDesign(StrictModule, NonTrainableState):
    """A native mechanics problem restricted to one frozen decoder image."""

    topology_problem: TopologyMechanicsProblem
    parameterization: StateDesignParameterization


def prepare_learned_topology_design(
    problem: TopologyMechanicsProblem,
    decode: Callable,
    latent_template,
    /,
    *,
    latent_bounds: Bounds,
    decoder_id: str,
    realization_id: str,
) -> LearnedTopologyDesign:
    """Compose raw density decoding with the existing filter, projection and FE.

    Fixed cells are exact both before and after ``PreparedDensityTransform``.
    Existing volume constraints, physical bounds, load cases, branch gates and
    FE authority are retained; no new reanalysis protocol is introduced.
    """
    if not isinstance(problem, TopologyMechanicsProblem):
        raise TypeError("problem must be TopologyMechanicsProblem.")
    if not callable(decode):
        raise TypeError("decode must be callable.")
    prepared = problem.density_transform.prepared
    fixed = prepared.plan.filter.fixed_density
    decoder = _FixedDensityDecoder(decode, prepared.plan.filter.design_mask, fixed)
    physical_problem = problem.as_state_design_problem()
    free_indices = tuple(
        int(index)
        for index in np.flatnonzero(np.asarray(prepared.plan.filter.design_mask))
    )
    # The prepared fixed-region map proves the remaining cell bounds exactly.
    # Keep every free bound scalar so ReducedMMA retains its scalar-inequality
    # contract; large elementwise bound sets should use the structured route.
    free_bounds = tuple(
        StateDesignConstraint(
            _DensityCoordinate(index),
            lower=0.0,
            upper=1.0,
            constraint_id=f"{problem.problem_id}/free-density-bound:{index}",
            depends_on_state=False,
        )
        for index in free_indices
    )
    physical_problem = eqx.tree_at(
        lambda item: (item.design_bounds, item.constraints),
        physical_problem,
        (None, physical_problem.constraints + free_bounds),
    )
    parameterization = reparameterize_state_design(
        physical_problem,
        decoder,
        latent_template,
        fixed,
        latent_bounds=latent_bounds,
        decoder_id=decoder_id,
        realization_id=realization_id,
    )
    return LearnedTopologyDesign(problem, parameterization)


class _ShapeDecoder(StrictModule, NonTrainableState):
    decode: Callable
    reference: DesignState

    def __call__(self, latent):
        design = self.decode(latent)
        if not isinstance(design, DesignState) or design.schema != self.reference.schema:
            raise ValueError("Shape decoder must return the exact DesignState schema.")
        fixed_changed = jnp.asarray(False)
        for spec, value, reference in zip(
            design.schema.specs, design.values, self.reference.values, strict=True
        ):
            if not spec.trainable:
                fixed_changed = fixed_changed | jnp.any(value != reference)
        return eqx.error_if(
            design, fixed_changed, "Shape decoder changed a frozen parameter."
        )


class _DesignStateCoordinate(StrictModule):
    parameter_index: int = eqx.field(static=True)
    coordinate_index: int = eqx.field(static=True)

    def __call__(self, state, design, args):
        del state, args
        return jnp.ravel(design.values[self.parameter_index])[self.coordinate_index]


def prepare_learned_shape_design(
    problem: StateDesignProblem,
    decode: Callable,
    latent_template,
    physical_template: DesignState,
    /,
    *,
    latent_bounds: Bounds,
    decoder_id: str,
    realization_id: str,
    design_admissibility: Callable,
) -> StateDesignParameterization:
    """Lower exact-schema geometry with mandatory pre-FE admissibility.

    The physical state solver must already be FE-authoritative. The caller's
    admissibility predicate must test the realized geometry (e.g. positive cell
    Jacobians), not substitute a safe geometry. Parameter-schema bounds are
    retained in addition to all physical problem constraints and bounds.
    """
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be StateDesignProblem.")
    if not isinstance(
        problem.state_solver, (FiniteElementStateSolver, NeuralVariationalStateSolver)
    ):
        raise TypeError(
            "Shape design requires an FE-authoritative physical state solver."
        )
    if not isinstance(physical_template, DesignState):
        raise TypeError("physical_template must be a DesignState.")
    if not callable(design_admissibility):
        raise TypeError("design_admissibility must be callable.")
    schema_constraints = []
    for parameter_index, (spec, value) in enumerate(
        zip(
            physical_template.schema.specs,
            physical_template.values,
            strict=True,
        )
    ):
        if not spec.trainable:
            continue
        lower, upper = spec.bounds
        if lower is None and upper is None:
            continue
        for coordinate_index in range(value.size):
            schema_constraints.append(
                StateDesignConstraint(
                    _DesignStateCoordinate(parameter_index, coordinate_index),
                    lower=-jnp.inf if lower is None else lower,
                    upper=jnp.inf if upper is None else upper,
                    constraint_id=(
                        f"{problem.problem_id}/shape-schema-bound:"
                        f"{spec.parameter_id}:{coordinate_index}"
                    ),
                    depends_on_state=False,
                )
            )
    physical = eqx.tree_at(
        lambda item: item.constraints,
        problem,
        problem.constraints + tuple(schema_constraints),
    )
    return reparameterize_state_design(
        physical,
        _ShapeDecoder(decode, physical_template),
        latent_template,
        physical_template,
        latent_bounds=latent_bounds,
        decoder_id=decoder_id,
        realization_id=realization_id,
        design_admissibility=design_admissibility,
    )


class LearnedTopologyResult(StrictModule):
    """Latent optimality and independently recertified physical mechanics.

    ``topology_result.state_design`` deliberately remains the latent result; its
    certificate must never be interpreted as full-density stationarity.
    """

    latent_result: StateDesignResult
    topology_result: TopologyOptimizationResult
    reanalysis: TopologyReanalysisReport
    reference_volume_ratio: Array
    reference_feasible: Array
    decoder_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @property
    def accepted(self):
        return (
            self.latent_result.successful
            & self.reanalysis.accepted
            & self.reference_feasible
        )


def solve_learned_topology_design(
    design: LearnedTopologyDesign,
    initial_states: PyTree[Array],
    initial_latent,
    reanalysis_plan: TopologyReanalysisPlan,
    initial_reference_state,
    /,
    *,
    method: AbstractStateDesignMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> LearnedTopologyResult:
    """Solve in latent coordinates, then execute mandatory physical FE reanalysis.

    Hard extraction, if wanted, belongs in the existing reanalysis plan's
    transfer function and is therefore outside every derivative in this solve.
    """
    if not isinstance(design, LearnedTopologyDesign):
        raise TypeError("design must be LearnedTopologyDesign.")
    if not isinstance(reanalysis_plan, TopologyReanalysisPlan):
        raise TypeError("reanalysis_plan must be TopologyReanalysisPlan.")
    parameterization = design.parameterization
    method_ = ReducedMMA() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    result = solve_state_design(
        parameterization.problem,
        initial_states,
        initial_latent,
        method=method_,
        termination=termination_,
        args=args,
    )
    problem = design.topology_problem
    raw = parameterization.decode(result.design)
    physical = problem.physical_density(raw)
    adjoint_ok = (
        jnp.asarray(False)
        if result.adjoint_acceptance is None
        else result.adjoint_acceptance.accepted
    )
    accepted = result.successful & result.state_acceptance.accepted & adjoint_ok
    stage_evidence = TopologyContinuationStageEvidence(
        accepted,
        jnp.asarray(False),
        result.status,
        result.state_acceptance.accepted,
        adjoint_ok,
        parameterization.decoder_id,
    )
    topology = TopologyOptimizationResult(
        result,
        (result,),
        (stage_evidence,),
        raw,
        problem.density_transform.filtered(raw),
        physical,
        problem.material_parameters(raw),
        problem.load_values(result.state, raw, args=args),
        problem.volume_ratio(raw),
        jnp.sum(physical * problem.density_transform.measures),
        jnp.sum(problem.density_transform.measures),
        problem.density_transform.beta,
        problem.material_interpolation.penalty,
        accepted,
        problem.problem_id,
    )
    report = reanalyse_topology_design(
        topology,
        reanalysis_plan,
        initial_reference_state,
        args=args,
    )
    reference_volume = reanalysis_plan.reference_problem.volume_ratio(
        report.evidence.transfer.raw_density,
        reanalysis_plan.beta,
    )
    reference_feasible = (
        report.evidence.transfer.finite
        & jnp.isfinite(reference_volume)
        & (
            reference_volume
            <= reanalysis_plan.reference_problem.volume_fraction
            + termination_.absolute_optimality
        )
    )
    return LearnedTopologyResult(
        result,
        topology,
        report,
        reference_volume,
        reference_feasible,
        parameterization.decoder_id,
        parameterization.realization_id,
    )


class MechanicsPotentialGuidance(AbstractScoreGuidance):
    """Native ``GuidedScoreField`` correction from an accepted physical value/VJP.

    The log potential is minus ``scale`` times the physical scalar objective.
    When ``denoise`` is supplied it maps (noisy latent, time, ScoreContext) to a
    clean latent estimate: this guidance is explicitly heuristic, never an exact
    noised-state likelihood. Neither decoder Jacobians nor optimizer gradients
    are formed. Failed state or transpose certification yields invalid guidance
    and a nonfinite correction; callers must reject such proposals.
    """

    parameterization: StateDesignParameterization
    initial_state: PyTree[Array]
    args: Any
    linear_policy: Any
    denoise: Callable | None = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    guidance_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterization: StateDesignParameterization,
        initial_state,
        /,
        *,
        scale: float = 1.0,
        denoise: Callable | None = None,
        args=None,
        linear_policy=None,
        guidance_id: str = "mechanics-potential",
    ):
        if not isinstance(parameterization, StateDesignParameterization):
            raise TypeError("parameterization must be StateDesignParameterization.")
        if not isfinite(float(scale)) or float(scale) < 0.0:
            raise ValueError("scale must be finite and nonnegative.")
        if denoise is not None and not callable(denoise):
            raise TypeError("denoise must be callable or None.")
        if not guidance_id:
            raise ValueError("guidance_id must be nonempty.")
        self.parameterization = parameterization
        self.initial_state = initial_state
        self.args = args
        self.linear_policy = linear_policy
        self.denoise = denoise
        self.scale = float(scale)
        self.exactness = "heuristic" if denoise is not None else "approximate"
        self.guidance_id = guidance_id

    def evaluate(self, state, time, context, /, *, key=None):
        del key
        if not isinstance(context, ScoreContext):
            raise TypeError("context must be ScoreContext.")
        current = jnp.asarray(state)
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("One guidance evaluation requires scalar time.")
        denoiser = self.denoise
        if denoiser is None:
            estimate = lambda value: value
        else:
            estimate = lambda value: denoiser(value, time_, context)
        latent, pullback = jax.vjp(estimate, current)
        response = self.parameterization.response_vjp(
            latent,
            self.initial_state,
            args=self.args,
            linear_policy=self.linear_policy,
        )
        gradient = pullback(response.latent_cotangent)[0]
        if gradient.shape != current.shape:
            raise ValueError("Guidance VJP must preserve the score state shape.")
        valid = (
            response.accepted
            & jnp.isfinite(response.values)
            & jnp.all(jnp.isfinite(gradient))
        )
        correction = jnp.where(
            valid, -self.scale * gradient, jnp.full_like(gradient, jnp.nan)
        )
        return GuidanceEvaluation(correction, valid, self.exactness, self.guidance_id)


__all__ = [
    "LearnedTopologyDesign",
    "LearnedTopologyResult",
    "MechanicsPotentialGuidance",
    "prepare_learned_shape_design",
    "prepare_learned_topology_design",
    "solve_learned_topology_design",
]
