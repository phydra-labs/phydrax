#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.spatial import cKDTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import (
    AbstractStateDesignMethod,
    Bounds,
    OptimizationTermination,
    ReducedMMA,
    solve_state_design,
    StateDesignConstraint,
    StateDesignProblem,
    StateDesignResult,
)
from ...optim._pde_constrained import AbstractStateSolver
from ...sparse import EdgeRelation, SparseLinearMap


class SIMPInterpolation(StrictModule, NonTrainableState):
    """Solid-isotropic material interpolation on a bounded physical density."""

    minimum_modulus: float = eqx.field(static=True)
    solid_modulus: float = eqx.field(static=True)
    penalty: float = eqx.field(static=True)
    interpolation_id: str = eqx.field(static=True)

    def __init__(
        self,
        solid_modulus: float,
        /,
        *,
        minimum_modulus: float = 1.0e-9,
        penalty: float = 3.0,
    ):
        minimum = float(minimum_modulus)
        solid = float(solid_modulus)
        exponent = float(penalty)
        if any(not isfinite(value) for value in (minimum, solid, exponent)):
            raise ValueError("SIMP parameters must be finite.")
        if minimum <= 0.0 or solid <= minimum or exponent <= 0.0:
            raise ValueError(
                "SIMP requires 0 < minimum_modulus < solid_modulus and positive penalty."
            )
        self.minimum_modulus = minimum
        self.solid_modulus = solid
        self.penalty = exponent
        self.interpolation_id = canonical_fingerprint(
            {
                "kind": "simp-interpolation",
                "minimum_modulus": minimum,
                "solid_modulus": solid,
                "penalty": exponent,
            }
        )

    def __call__(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value < 0.0) | (value > 1.0)),
            "SIMP density must be finite and lie in [0, 1].",
        )
        return self.minimum_modulus + value**self.penalty * (
            self.solid_modulus - self.minimum_modulus
        )


class DensityFilterPlan(StrictModule, NonTrainableState):
    """Sparse physical-radius cell-density filter over fixed centers and measures."""

    centers: Array
    measures: Array
    radius: float = eqx.field(static=True)
    relation: EdgeRelation
    coefficients: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        centers: ArrayLike,
        measures: ArrayLike,
        radius: float,
        /,
    ):
        points = np.asarray(centers, dtype=float)
        volumes = np.asarray(measures, dtype=float)
        radius_ = float(radius)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("centers must have shape (cell_count, dimension).")
        if volumes.shape != (points.shape[0],):
            raise ValueError("measures must contain one value per cell.")
        if (
            np.any(~np.isfinite(points))
            or np.any(~np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or not isfinite(radius_)
            or radius_ <= 0.0
        ):
            raise ValueError(
                "Density-filter geometry and radius must be finite and positive."
            )

        tree = cKDTree(points)
        source_indices: list[int] = []
        target_indices: list[int] = []
        raw_weights: list[float] = []
        row_sums = np.zeros((points.shape[0],), dtype=float)
        for target, neighbours in enumerate(tree.query_ball_point(points, radius_)):
            for source in sorted(int(index) for index in neighbours):
                distance = float(np.linalg.norm(points[target] - points[source]))
                weight = max(radius_ - distance, 0.0) * volumes[source]
                if weight > 0.0:
                    source_indices.append(source)
                    target_indices.append(target)
                    raw_weights.append(weight)
                    row_sums[target] += weight
        if np.any(row_sums <= 0.0):
            raise ValueError("Every density-filter row must contain positive support.")
        coefficients = np.asarray(raw_weights) / row_sums[np.asarray(target_indices)]
        count = int(points.shape[0])
        relation = EdgeRelation(
            np.asarray(source_indices, dtype=np.int32),
            np.asarray(target_indices, dtype=np.int32),
            source_size=count,
            target_size=count,
        )
        self.centers = jnp.asarray(points)
        self.measures = jnp.asarray(volumes)
        self.radius = radius_
        self.relation = relation
        self.coefficients = jnp.asarray(coefficients)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "density-filter-plan",
                "centers": array_tree_fingerprint(self.centers),
                "measures": array_tree_fingerprint(self.measures),
                "radius": radius_,
            }
        )

    def prepare(self, /) -> "PreparedDensityFilter":
        return PreparedDensityFilter(self)


class PreparedDensityFilter(StrictModule, NonTrainableState):
    """Prepared sparse filter and constant-preservation evidence."""

    plan: DensityFilterPlan
    operator: SparseLinearMap
    constant_residual: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DensityFilterPlan, /):
        if not isinstance(plan, DensityFilterPlan):
            raise TypeError("plan must be a DensityFilterPlan.")
        operator = SparseLinearMap(
            plan.relation,
            plan.coefficients,
            operator_id=f"density-filter/{plan.plan_id}",
        )
        ones = jnp.ones((plan.centers.shape[0],), dtype=plan.coefficients.dtype)
        residual = jnp.max(jnp.abs(operator.mv(ones) - ones))
        self.plan = plan
        self.operator = operator
        self.constant_residual = residual
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-density-filter",
                "plan": plan.plan_id,
            }
        )

    def apply(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        if value.shape != self.operator.input_shape:
            raise ValueError(
                f"density must have shape {self.operator.input_shape}; got {value.shape}."
            )
        return self.operator.mv(value)


class ComplianceTopologyProblem(StrictModule, NonTrainableState):
    """Fixed-mesh compliance problem with filtered SIMP cell densities."""

    state_residual: Callable = eqx.field(static=True)
    load: Array
    density_filter: PreparedDensityFilter
    interpolation: SIMPInterpolation
    volume_fraction: float = eqx.field(static=True)
    state_solver: AbstractStateSolver
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_residual: Callable,
        load: ArrayLike,
        density_filter: PreparedDensityFilter,
        interpolation: SIMPInterpolation,
        volume_fraction: float,
        state_solver: AbstractStateSolver,
        /,
        *,
        problem_id: str = "compliance-topology",
    ):
        if not callable(state_residual):
            raise TypeError("state_residual must be callable.")
        if not isinstance(density_filter, PreparedDensityFilter):
            raise TypeError("density_filter must be a PreparedDensityFilter.")
        if not isinstance(interpolation, SIMPInterpolation):
            raise TypeError("interpolation must be a SIMPInterpolation.")
        if not isinstance(state_solver, AbstractStateSolver):
            raise TypeError("state_solver must be an AbstractStateSolver.")
        load_ = jnp.asarray(load)
        fraction = float(volume_fraction)
        identifier = str(problem_id)
        if not jnp.issubdtype(load_.dtype, jnp.inexact) or load_.size == 0:
            raise TypeError("load must be one nonempty inexact array.")
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("volume_fraction must lie strictly between zero and one.")
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.state_residual = state_residual
        self.load = load_
        self.density_filter = density_filter
        self.interpolation = interpolation
        self.volume_fraction = fraction
        self.state_solver = state_solver
        self.problem_id = identifier

    def physical_density(self, raw_density: ArrayLike, /) -> Array:
        return self.density_filter.apply(raw_density)

    def modulus(self, raw_density: ArrayLike, /) -> Array:
        return self.interpolation(self.physical_density(raw_density))

    def volume_ratio(self, raw_density: ArrayLike, /) -> Array:
        density = self.physical_density(raw_density)
        measures = self.density_filter.plan.measures
        return jnp.sum(density * measures) / jnp.sum(measures)

    def as_state_design_problem(self, /) -> StateDesignProblem:
        def residual(state, density, args):
            return self.state_residual(state, self.modulus(density), args)

        def objective(state, density, args):
            del density, args
            if state.shape != self.load.shape:
                raise ValueError("state and load must have one shape for compliance.")
            return jnp.real(jnp.vdot(self.load, state))

        volume = StateDesignConstraint(
            lambda state, density, args: self.volume_ratio(density),
            upper=self.volume_fraction,
            constraint_id=f"{self.problem_id}/volume",
            depends_on_state=False,
        )
        return StateDesignProblem(
            residual,
            objective,
            state_solver=self.state_solver,
            design_bounds=Bounds(1.0e-3, 1.0),
            constraints=(volume,),
            problem_id=self.problem_id,
        )


class TopologyOptimizationResult(StrictModule):
    """State-design result with raw, physical, and constitutive density views."""

    state_design: StateDesignResult
    raw_density: Array
    physical_density: Array
    modulus: Array
    volume_ratio: Array
    problem_id: str = eqx.field(static=True)


class DensityTransferResult(StrictModule):
    """Transferred physical density and relative material-measure defect."""

    density: Array
    relative_measure_error: Array


class TopologyReanalysisReport(StrictModule):
    """Independent reference-mesh compliance and distortion accounting."""

    optimized_compliance: Array
    reference_compliance: Array
    uniform_coarse_compliance: Array
    uniform_reference_compliance: Array
    discretization_ratio: Array
    excess_stiffness_overreport: Array
    transfer_measure_error: Array


def solve_topology_optimization(
    problem: ComplianceTopologyProblem,
    initial_state: ArrayLike,
    initial_density: ArrayLike,
    /,
    *,
    method: AbstractStateDesignMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> TopologyOptimizationResult:
    """Solve one fixed-mesh compliance topology problem."""

    if not isinstance(problem, ComplianceTopologyProblem):
        raise TypeError("problem must be a ComplianceTopologyProblem.")
    method_ = ReducedMMA() if method is None else method
    if not isinstance(method_, AbstractStateDesignMethod):
        raise TypeError("method must be an AbstractStateDesignMethod or None.")
    termination_ = (
        OptimizationTermination(maximum_steps=200) if termination is None else termination
    )
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    result = solve_state_design(
        problem.as_state_design_problem(),
        jnp.asarray(initial_state),
        jnp.asarray(initial_density),
        method=method_,
        termination=termination_,
        args=args,
    )
    physical = problem.physical_density(result.design)
    modulus = problem.interpolation(physical)
    return TopologyOptimizationResult(
        result,
        result.design,
        physical,
        modulus,
        problem.volume_ratio(result.design),
        problem.problem_id,
    )


def reanalyse_topology_design(
    result: TopologyOptimizationResult,
    transfer: Callable[[Array], DensityTransferResult],
    reference_compliance: Callable[[Array], ArrayLike],
    /,
    *,
    uniform_coarse_compliance: ArrayLike,
    uniform_reference_compliance: ArrayLike,
) -> TopologyReanalysisReport:
    """Reanalyse a physical design on an independent discretization."""

    if not isinstance(result, TopologyOptimizationResult):
        raise TypeError("result must be a TopologyOptimizationResult.")
    if not callable(transfer) or not callable(reference_compliance):
        raise TypeError("transfer and reference_compliance must be callable.")
    transferred = transfer(result.physical_density)
    if not isinstance(transferred, DensityTransferResult):
        raise TypeError("transfer must return DensityTransferResult.")
    own = jnp.asarray(result.state_design.objective)
    reference = jnp.asarray(reference_compliance(transferred.density))
    coarse_uniform = jnp.asarray(uniform_coarse_compliance)
    reference_uniform = jnp.asarray(uniform_reference_compliance)
    values = (own, reference, coarse_uniform, reference_uniform)
    if any(value.shape != () for value in values):
        raise ValueError("Reanalysis compliance values must be scalars.")
    ratio = reference_uniform / coarse_uniform
    excess = (reference / own) / ratio - 1.0
    return TopologyReanalysisReport(
        own,
        reference,
        coarse_uniform,
        reference_uniform,
        ratio,
        excess,
        transferred.relative_measure_error,
    )


__all__ = [
    "ComplianceTopologyProblem",
    "DensityFilterPlan",
    "DensityTransferResult",
    "PreparedDensityFilter",
    "SIMPInterpolation",
    "TopologyOptimizationResult",
    "TopologyReanalysisReport",
    "reanalyse_topology_design",
    "solve_topology_optimization",
]
