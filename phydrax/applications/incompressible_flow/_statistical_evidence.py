#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Finite-campaign evidence, not statistical-solution compactness certification.

The structure function convention is that of Lanthaler, Mishra and
Parés-Pulido, arXiv:1909.06615, Sections 2 and 4. Componentwise and sliced
transport are explicitly NOT the joint correlation-marginal Wasserstein metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ..._validation import (
    finite_real_scalar,
    integer,
    positive_integer,
    real_number_array,
)
from ...discretization.spectral import (
    HermitianSpectralCoordinates,
    PeriodicIncrementMomentResult,
    TensorSpectralDiscretization,
)
from ...equations import CompiledIncompressibleSpectralDynamics
from ...execution import compiled_memory_estimate, MemoryBudgetError
from ...solver import (
    CourantGuardedETDRKMethod,
    ETDRKMethod,
    execute_fixed_step_ensemble,
    FixedStepEnsemblePlan,
    FixedStepEnsembleResult,
    FixedStepProblem,
    LESStabilityGuardedETDRKMethod,
    PreparedCourantGuardedETDRKMethod,
    PreparedETDRKMethod,
    PreparedFixedStepEnsemble,
    PreparedLESStabilityGuardedETDRKMethod,
)
from ...solver._deterministic_ensemble import (
    deterministic_semantic_path_id,
    WeightedEnsembleReducer,
)
from ...transport import sliced_wasserstein_distance, wasserstein_distance_1d
from ...typing import parse, PRNGKey
from ._initial_conditions import PeriodicFlowInitialState
from ._production import _bind_periodic_method


ScalingEvidenceStatus: TypeAlias = Literal["complete", "abstained"]


def _member_labels(initial_state: PeriodicFlowInitialState, /) -> tuple[str, ...]:
    """Single owner of resolution-independent scientific member identity."""
    return tuple(
        canonical_fingerprint(
            {"coupling": initial_state.coupling_id, "member": int(member)}
        )
        for member in np.asarray(initial_state.member_indices)
    )


def _member_path_ids(initial_state: PeriodicFlowInitialState, /) -> tuple[str, ...]:
    return tuple(
        deterministic_semantic_path_id(label) for label in _member_labels(initial_state)
    )


@dataclass(frozen=True)
class PeriodicFlowEnsembleResult:
    """Native trajectories bound to their scientific initial-measure owner."""

    result: FixedStepEnsembleResult
    initial_state: PeriodicFlowInitialState

    def __post_init__(self) -> None:
        initial, result = self.initial_state, self.result
        if initial.discretization_id != initial.discretization.prepared_id:
            raise ValueError("Initial state discretization identity is inconsistent.")
        if initial.discretization_id not in result.discretization_ids:
            raise ValueError(
                "Result problem discretization identity differs from the initial state."
            )
        if result.path_ids != _member_path_ids(initial):
            raise ValueError(
                "Result semantic paths differ from the initial measure members."
            )
        if not isinstance(result.states, Array):
            raise TypeError("Periodic flow results require an array velocity state.")
        if result.states.shape != (
            initial.velocity.shape[0],
            result.times.size,
            *initial.velocity.shape[1:],
        ):
            raise ValueError(
                "Ensemble states do not match the owned initial discretization."
            )


@dataclass(frozen=True)
class PreparedPeriodicFlowEnsemble:
    """Prepared native execution plus its immutable scientific initial owner."""

    ensemble: PreparedFixedStepEnsemble
    initial_state: PeriodicFlowInitialState

    def __post_init__(self) -> None:
        if self.ensemble.plan.path_ids != _member_path_ids(self.initial_state):
            raise ValueError(
                "Prepared ensemble paths differ from the initial measure members."
            )
        bundle = self.ensemble.plan.problem.discretization_bundle
        if bundle is None or self.initial_state.discretization_id not in tuple(
            record.artifact_id for record in bundle.records
        ):
            raise ValueError(
                "Prepared problem discretization differs from the initial state."
            )

    def execute(self) -> PeriodicFlowEnsembleResult:
        return PeriodicFlowEnsembleResult(
            execute_fixed_step_ensemble(self.ensemble), self.initial_state
        )


@dataclass(frozen=True)
class CoupledEnsemblePair:
    """Validated campaigns with identical ordered members, weights and saves."""

    coarse: PeriodicFlowEnsembleResult
    fine: PeriodicFlowEnsembleResult

    def __post_init__(self) -> None:
        c, f = self.coarse, self.fine
        if (c.initial_state.measure_id, c.initial_state.coupling_id) != (
            f.initial_state.measure_id,
            f.initial_state.coupling_id,
        ):
            raise ValueError(
                "Coupled ensembles require the same measure and coupling root."
            )
        if c.result.path_ids != f.result.path_ids:
            raise ValueError(
                "Coupled ensembles require identical ordered semantic paths."
            )
        for left, right, name in (
            (
                c.initial_state.member_indices,
                f.initial_state.member_indices,
                "member indices",
            ),
            (c.result.reducer.weights, f.result.reducer.weights, "weights"),
            (c.result.times, f.result.times, "saved times"),
        ):
            if not np.array_equal(np.asarray(left), np.asarray(right)):
                raise ValueError(f"Coupled ensembles require identical {name}.")


def align_coupled_ensembles(
    coarse: PeriodicFlowEnsembleResult, fine: PeriodicFlowEnsembleResult, /
) -> CoupledEnsemblePair:
    """Refuse distinct measures, save grids, path sets or coupling roots."""
    c, f = coarse.result, fine.result
    if (
        coarse.initial_state.measure_id != fine.initial_state.measure_id
        or coarse.initial_state.coupling_id != fine.initial_state.coupling_id
    ):
        raise ValueError("Coupled ensembles require the same measure and coupling root.")
    if (
        len(set(c.path_ids)) != len(c.path_ids)
        or set(c.path_ids) != set(f.path_ids)
        or len(c.path_ids) != len(f.path_ids)
    ):
        raise ValueError("Coupled ensembles require identical semantic path sets.")
    if not np.array_equal(np.asarray(c.times), np.asarray(f.times)):
        raise ValueError("Coupled ensembles require identical saved times.")
    order = np.asarray([f.path_ids.index(path) for path in c.path_ids], dtype=np.int32)
    if not np.array_equal(
        np.asarray(c.reducer.weights), np.asarray(f.reducer.weights)[order]
    ):
        raise ValueError("Coupled ensembles require equal member measures (weights).")
    if not np.array_equal(
        np.asarray(coarse.initial_state.member_indices),
        np.asarray(fine.initial_state.member_indices)[order],
    ):
        raise ValueError("Semantic paths disagree with coupled member indices.")
    if np.array_equal(order, np.arange(len(order), dtype=np.int32)):
        return CoupledEnsemblePair(coarse, fine)
    solutions = jax.tree.map(lambda a: a[order], f.solutions)
    reducer = WeightedEnsembleReducer(f.reducer.weights[order], f.reducer.valid[order])
    # Static identifiers are changed by dataclass reconstruction, not tree_at.
    evidence = eqx.tree_at(
        lambda e: (e.status, e.backend_successful, e.finite_trajectory),
        f.evidence,
        (
            f.evidence.status[order],
            f.evidence.backend_successful[order],
            f.evidence.finite_trajectory[order],
        ),
    )
    reordered = FixedStepEnsembleResult(
        f.times,
        solutions.states,
        f.valid[order],
        f.status[order],
        solutions,
        reducer,
        evidence,
        c.path_ids,
        tuple(f.realization_ids[i] for i in order),
        f.prepared_id,
        canonical_fingerprint(
            {
                "kind": "aligned-fixed-step-ensemble-result",
                "source": f.result_id,
                "aligned_to": c.result_id,
                "paths": c.path_ids,
                "realizations": tuple(f.realization_ids[i] for i in order),
            }
        ),
        f.discretization_ids,
    )
    state = eqx.tree_at(
        lambda s: (
            s.velocity,
            s.member_indices,
            s.parameters,
            s.removed_coefficient_energy,
            s.projection_removed_energy,
            s.removed_mean_vorticity,
            s.quadrature_residual,
            s.baseline_coefficient_energy,
        ),
        fine.initial_state,
        (
            fine.initial_state.velocity[order],
            fine.initial_state.member_indices[order],
            jax.tree.map(lambda a: a[order], fine.initial_state.parameters),
            fine.initial_state.removed_coefficient_energy[order],
            fine.initial_state.projection_removed_energy[order],
            fine.initial_state.removed_mean_vorticity[order],
            fine.initial_state.quadrature_residual[order],
            fine.initial_state.baseline_coefficient_energy[order],
        ),
    )
    return CoupledEnsemblePair(coarse, PeriodicFlowEnsembleResult(reordered, state))


def _admit_compiled_workspace(
    executable: jax.stages.Compiled, retained_bytes: int, limit: int, /
) -> None:
    memory = compiled_memory_estimate(executable)
    if memory is None:
        raise RuntimeError("Periodic metrics require compiler memory analysis.")
    executable_bytes = max(
        memory.argument_bytes
        + memory.output_bytes
        - memory.alias_bytes
        + memory.temporary_bytes
        + memory.generated_code_bytes,
        0 if memory.peak_bytes is None else memory.peak_bytes,
    ) + (
        memory.host_argument_bytes
        + memory.host_output_bytes
        - memory.host_alias_bytes
        + memory.host_temporary_bytes
        + memory.host_generated_code_bytes
    )
    required = retained_bytes + executable_bytes
    if required > limit:
        raise MemoryBudgetError("maximum_workspace_bytes", required, limit)


def _shared_values(
    pair: CoupledEnsemblePair,
    time_index: int,
    /,
    *,
    member_chunk_size: int,
    maximum_workspace_bytes: int,
    extra_workspace_bytes: int = 0,
) -> tuple[Array, Array, Array]:
    """Evaluate native fine modes on coarse nodes by alias summation, not truncation."""
    c, f = pair.coarse, pair.fine
    slot = integer(time_index, "time_index")
    chunk = positive_integer(member_chunk_size, "member_chunk_size")
    cap = positive_integer(maximum_workspace_bytes, "maximum_workspace_bytes")
    if not (-c.result.times.size <= slot < c.result.times.size):
        raise ValueError("time_index is outside the saved grid.")
    if (
        not np.asarray(c.result.valid[:, slot]).all()
        or not np.asarray(f.result.valid[:, slot]).all()
    ):
        raise ValueError("Invalid members cannot be silently dropped from comparison.")
    cd, fd = c.initial_state.discretization, f.initial_state.discretization
    if len(cd.axes) != len(fd.axes):
        raise ValueError("Comparison requires equal spatial dimension.")
    for ca, fa in zip(cd.axes, fd.axes, strict=True):
        if (
            ca.family != "fourier"
            or fa.family != "fourier"
            or ca.domain.domain_id != fa.domain.domain_id
            or fa.nodes.size % ca.nodes.size
            or not np.allclose(
                np.asarray(ca.nodes),
                np.asarray(fa.nodes)[:: fa.nodes.size // ca.nodes.size],
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("Coarse axes must be nested fine-grid injection lattices.")
    members = c.result.states.shape[0]
    lanes = min(chunk, members)
    physical_bytes = (
        members
        * cd.quadrature_weights.size
        * len(cd.axes)
        * np.dtype(cd.plan.precision.physical_dtype).itemsize
    )
    modal_bytes = (
        lanes
        * (np.prod(fd.modal_shape) + 4 * np.prod(cd.modal_shape))
        * len(cd.axes)
        * f.result.states.dtype.itemsize
    )
    # Both host staging arrays and device values remain live during transport.
    # Eight modal buffers conservatively cover folding, transform and staging.
    required = 4 * physical_bytes + 8 * int(modal_bytes) + extra_workspace_bytes
    if required > cap:
        raise MemoryBudgetError("maximum_workspace_bytes", required, cap)

    @jax.jit
    def evaluate(coarse: Array, fine: Array) -> tuple[Array, Array]:
        folded = fine
        for axis, (ca, fa) in enumerate(zip(cd.axes, fd.axes, strict=True)):
            moved = jnp.moveaxis(folded, axis + 1, 0)
            indices = jnp.mod(fa.modes.mode_numbers, ca.nodes.size).astype(jnp.int32)
            target = jnp.zeros((ca.nodes.size, *moved.shape[1:]), dtype=moved.dtype)
            folded = jnp.moveaxis(target.at[indices].add(moved), 0, axis + 1)
        return jax.vmap(cd.reconstruct)(coarse), jax.vmap(cd.reconstruct)(folded)

    executable = evaluate.lower(
        jax.ShapeDtypeStruct(
            (lanes, *cd.modal_shape, len(cd.axes)), c.result.states.dtype
        ),
        jax.ShapeDtypeStruct(
            (lanes, *fd.modal_shape, len(fd.axes)), f.result.states.dtype
        ),
    ).compile()
    _admit_compiled_workspace(executable, 4 * physical_bytes + extra_workspace_bytes, cap)

    shape = (members, *cd.physical_shape, len(cd.axes))
    cv = np.empty(shape, dtype=cd.plan.precision.physical_dtype)
    fv = np.empty_like(cv)
    for start in range(0, members, chunk):
        end = min(start + chunk, members)
        coarse_modal = c.result.states[start:end, slot]
        fine_modal = f.result.states[start:end, slot]
        if end - start < lanes:
            coarse_modal = jnp.pad(
                coarse_modal,
                ((0, lanes - (end - start)), *((0, 0),) * (coarse_modal.ndim - 1)),
            )
            fine_modal = jnp.pad(
                fine_modal,
                ((0, lanes - (end - start)), *((0, 0),) * (fine_modal.ndim - 1)),
            )
        coarse_values, fine_values = executable(coarse_modal, fine_modal)
        cv[start:end], fv[start:end] = (
            np.asarray(coarse_values)[: end - start],
            np.asarray(fine_values)[: end - start],
        )
    if not np.isfinite(cv).all() or not np.isfinite(fv).all():
        raise ValueError("Comparison requires finite physical states.")
    return (
        jnp.asarray(cv).reshape((members, -1, len(cd.axes))),
        jnp.asarray(fv).reshape((members, -1, len(cd.axes))),
        cd.quadrature_weights.reshape(-1),
    )


@dataclass(frozen=True)
class ComponentwiseEmpiricalWassersteinResult:
    """Exact scalar marginals; component-averaged lower diagnostic, not joint Wp."""

    value: Array
    per_node_component: Array
    p: float
    nonclaim: str = (
        "Not joint velocity or k-point correlation-marginal Wasserstein distance."
    )


@dataclass(frozen=True)
class PairedCouplingUpperBoundResult:
    """Integrated coupled Euclidean p-cost, an upper bound for one-point Wp."""

    value: Array
    per_node_cost: Array
    p: float
    nonclaim: str = (
        "Only this empirical coupling; not an optimal joint transport distance."
    )


@dataclass(frozen=True)
class SlicedJointMarginalDistanceResult:
    """Declared point tuples and finite random-projection error evidence."""

    values: Array
    projection_standard_error: Array
    projection_distances: Array
    projections: Array
    node_tuples: tuple[tuple[int, ...], ...]
    p: float
    num_projections: int
    nonclaim: str = (
        "Sliced metric, not joint Wasserstein; error is projection Monte Carlo only."
    )


def _exponent(p: float, /) -> float:
    value = finite_real_scalar(p, "p")
    if value < 1.0:
        raise ValueError("Wasserstein p must be finite and at least one.")
    return value


def componentwise_empirical_wasserstein(
    pair: CoupledEnsemblePair,
    /,
    *,
    p: float = 1.0,
    time_index: int = -1,
    node_chunk_size: int = 128,
    member_chunk_size: int = 8,
    maximum_workspace_bytes: int = 2**28,
) -> ComponentwiseEmpiricalWassersteinResult:
    """Integrate the mean of componentwise exact empirical Wp^p costs."""
    power = _exponent(p)
    chunk = positive_integer(node_chunk_size, "node_chunk_size")
    extra = (
        32
        * pair.coarse.initial_state.velocity.shape[0]
        * min(chunk, pair.coarse.initial_state.discretization.quadrature_weights.size)
        * len(pair.coarse.initial_state.discretization.axes)
    )
    cv, fv, weights = _shared_values(
        pair,
        time_index,
        member_chunk_size=member_chunk_size,
        maximum_workspace_bytes=maximum_workspace_bytes,
        extra_workspace_bytes=extra,
    )
    probabilities = pair.coarse.result.reducer.weights

    def node(values: tuple[Array, Array]) -> Array:
        source, target = values
        return jax.vmap(
            lambda a, b: wasserstein_distance_1d(
                a, b, p=power, source_weights=probabilities, target_weights=probabilities
            ),
            in_axes=(1, 1),
        )(source, target)

    @jax.jit
    def transport(source: Array, target: Array) -> tuple[Array, Array]:
        distances = jax.lax.map(
            node, (source.swapaxes(0, 1), target.swapaxes(0, 1)), batch_size=chunk
        )
        value = jnp.sum(weights * jnp.mean(distances**power, axis=-1)) ** (1.0 / power)
        return value, distances

    executable = transport.lower(cv, fv).compile()
    _admit_compiled_workspace(executable, cv.nbytes + fv.nbytes, maximum_workspace_bytes)
    value, distances = executable(cv, fv)
    return ComponentwiseEmpiricalWassersteinResult(value, distances, power)


def paired_coupling_upper_bound(
    pair: CoupledEnsemblePair,
    /,
    *,
    p: float = 1.0,
    time_index: int = -1,
    node_chunk_size: int = 128,
    member_chunk_size: int = 8,
    maximum_workspace_bytes: int = 2**28,
) -> PairedCouplingUpperBoundResult:
    power = _exponent(p)
    chunk = positive_integer(node_chunk_size, "node_chunk_size")
    extra = (
        32
        * pair.coarse.initial_state.velocity.shape[0]
        * min(chunk, pair.coarse.initial_state.discretization.quadrature_weights.size)
        * len(pair.coarse.initial_state.discretization.axes)
    )
    cv, fv, weights = _shared_values(
        pair,
        time_index,
        member_chunk_size=member_chunk_size,
        maximum_workspace_bytes=maximum_workspace_bytes,
        extra_workspace_bytes=extra,
    )
    probabilities = pair.coarse.result.reducer.weights / jnp.sum(
        pair.coarse.result.reducer.weights
    )

    def node(values: tuple[Array, Array]) -> Array:
        source, target = values
        return jnp.sum(probabilities * jnp.linalg.norm(source - target, axis=-1) ** power)

    @jax.jit
    def transport(source: Array, target: Array) -> tuple[Array, Array]:
        costs = jax.lax.map(
            node, (source.swapaxes(0, 1), target.swapaxes(0, 1)), batch_size=chunk
        )
        return jnp.sum(weights * costs) ** (1.0 / power), costs

    executable = transport.lower(cv, fv).compile()
    _admit_compiled_workspace(executable, cv.nbytes + fv.nbytes, maximum_workspace_bytes)
    value, costs = executable(cv, fv)
    return PairedCouplingUpperBoundResult(value, costs, power)


def sliced_joint_marginal_distance(
    pair: CoupledEnsemblePair,
    node_tuples: Sequence[Sequence[int]],
    /,
    *,
    key: PRNGKey,
    num_projections: int,
    p: float = 1.0,
    time_index: int = -1,
    member_chunk_size: int = 8,
    maximum_workspace_bytes: int = 2**28,
) -> SlicedJointMarginalDistanceResult:
    power = _exponent(p)
    key = parse(key, PRNGKey, "key")
    count = positive_integer(num_projections, "num_projections")
    tuples = tuple(
        tuple(integer(i, "node_tuples") for i in nodes) for nodes in node_tuples
    )
    if (
        count < 2
        or not tuples
        or not tuples[0]
        or any(len(t) != len(tuples[0]) for t in tuples)
    ):
        raise ValueError(
            "Declare equal nonempty k-point tuples, at least two projections, and a positive byte cap."
        )
    members = pair.coarse.initial_state.velocity.shape[0]
    dimension = len(pair.coarse.initial_state.discretization.axes)
    features = len(tuples[0]) * dimension
    # Retained projections/results plus per-tuple sort and projected samples.
    required = 8 * (
        len(tuples) * count * (features + 1)
        + 16 * count * members
        + 4 * members * features
    )
    cv, fv, _ = _shared_values(
        pair,
        time_index,
        member_chunk_size=member_chunk_size,
        maximum_workspace_bytes=maximum_workspace_bytes,
        extra_workspace_bytes=required,
    )
    if any(i < 0 or i >= cv.shape[1] for t in tuples for i in t):
        raise ValueError("Declared tuple node is outside the coarse grid.")
    probabilities = pair.coarse.result.reducer.weights
    nodes = jnp.asarray(tuples, dtype=jnp.int32)
    keys = jax.random.split(key, len(tuples))

    def one(
        source: Array, target: Array, args: tuple[Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        ids, subkey = args
        result = sliced_wasserstein_distance(
            source[:, ids].reshape((members, -1)),
            target[:, ids].reshape((members, -1)),
            p=power,
            source_weights=probabilities,
            target_weights=probabilities,
            key=subkey,
            num_projections=count,
        )
        costs = result.projection_distances**power
        # Delta-method SE of the p-root of a projection mean; zero is exact.
        error = jnp.where(
            result.value > 0,
            jnp.std(costs, ddof=1)
            / jnp.sqrt(count)
            / (
                power
                * jnp.maximum(result.value, jnp.finfo(result.value.dtype).tiny)
                ** (power - 1)
            ),
            0.0,
        )
        return result.value, error, result.projection_distances, result.projections

    @jax.jit
    def transport(
        source: Array, target: Array, nodes: Array, keys: Array
    ) -> tuple[Array, Array, Array, Array]:
        return jax.lax.map(lambda args: one(source, target, args), (nodes, keys))

    executable = transport.lower(cv, fv, nodes, keys).compile()
    _admit_compiled_workspace(
        executable,
        cv.nbytes + fv.nbytes + nodes.nbytes + keys.nbytes,
        maximum_workspace_bytes,
    )
    values, errors, distances, projections = executable(cv, fv, nodes, keys)
    return SlicedJointMarginalDistanceResult(
        values, errors, distances, projections, tuples, power, count
    )


@dataclass(frozen=True)
class StructureFunctionScalingFit:
    exponent: float
    standard_error: float
    constant: float
    residual: float
    condition_number: float
    status: ScalingEvidenceStatus
    reason: str
    jackknife_exponents: Array


@dataclass(frozen=True)
class InstantaneousStructureFunctionScalingEvidence:
    """Finite-resolution, finite-member evidence; no compactness certification."""

    fits: tuple[StructureFunctionScalingFit, ...]
    common_exponent: float
    envelope_constant: float
    consistent_within_campaign: bool
    paired_exponent_standard_errors: Array
    nonclaim: str = "Finite-campaign evidence only; tightness, uniform limits and theorem hypotheses are not certified."


@dataclass(frozen=True)
class TimeIntegratedStructureFunctionScalingEvidence:
    """Explicit time-integrated finite-campaign evidence, not a snapshot."""

    fits: tuple[StructureFunctionScalingFit, ...]
    common_exponent: float
    envelope_constant: float
    consistent_within_campaign: bool
    paired_exponent_standard_errors: Array
    time_quadrature_weights: tuple[float, ...]
    nonclaim: str = "Finite time-integrated campaign only; compactness and continuum limits are not certified."


class StructureFunctionScalingPlan(StrictModule):
    radius_window: tuple[float, float] = eqx.field(static=True)
    minimum_points: int = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)

    def __init__(
        self,
        radius_window: tuple[float, float],
        /,
        *,
        minimum_points: int = 4,
        maximum_condition_number: float = 1e8,
    ) -> None:
        lower, upper = (finite_real_scalar(v, "radius_window") for v in radius_window)
        points = positive_integer(minimum_points, "minimum_points")
        condition = finite_real_scalar(
            maximum_condition_number, "maximum_condition_number"
        )
        if lower <= 0 or upper < lower or points < 2 or condition < 1:
            raise ValueError(
                "Scaling needs a positive finite radius window, at least two points, and conditioning limit >= 1."
            )
        self.radius_window = (lower, upper)
        self.minimum_points = points
        self.maximum_condition_number = condition

    def evaluate(
        self,
        results: Sequence[PeriodicIncrementMomentResult],
        discretizations: Sequence[TensorSpectralDiscretization],
        /,
        *,
        weights: ArrayLike | None = None,
    ) -> InstantaneousStructureFunctionScalingEvidence:
        """Fit sqrt(mean M2); delete the SAME member in every radius/resolution.

        Weights are declared radius-fit weights, not independent-radius error bars.
        Campaign consistency uses paired two-standard-error exponent differences;
        this descriptive criterion is not a hypothesis test or convergence claim.
        Input results must already have identical coupled member ordering.
        """
        moments, grids = tuple(results), tuple(discretizations)
        if not moments or len(moments) != len(grids):
            raise ValueError("Provide one moment result per resolution.")
        members = moments[0].values.shape[0]
        if any(
            m.values.ndim != 2 or m.values.shape != (members, m.radii.size)
            for m in moments
        ):
            raise ValueError(
                "Radial M2 requires shared member count and (members,radii) values."
            )
        fits: list[StructureFunctionScalingFit] = []
        windows: list[tuple[Array, Array]] = []
        for moment, grid in zip(moments, grids, strict=True):
            radii = np.asarray(moment.radii)
            selected = np.flatnonzero(
                (radii >= self.radius_window[0]) & (radii <= self.radius_window[1])
            )
            spacing = max(float(a.length) / a.nodes.size for a in grid.axes)
            reason = ""
            if members < 3:
                reason = "At least three coupled members are required for delete-one uncertainty."
            elif self.radius_window[0] < 2 * spacing:
                reason = "r_min must be at least twice the maximum grid spacing."
            elif selected.size < self.minimum_points:
                reason = "Insufficient radii in the declared window."
            elif (
                not np.asarray(moment.finite).all()
                or not np.isfinite(np.asarray(moment.values)).all()
                or np.any(np.asarray(moment.values) < 0)
            ):
                reason = "Radial second moments must be finite and nonnegative for every member."
            if reason:
                fits.append(
                    StructureFunctionScalingFit(
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.inf,
                        "abstained",
                        reason,
                        jnp.full((members,), jnp.nan),
                    )
                )
                windows.append(
                    (
                        jnp.empty((0,), dtype=jnp.float64),
                        jnp.empty((0,), dtype=jnp.float64),
                    )
                )
                continue
            r = moment.radii[selected]
            values = moment.values[:, selected]
            mean = jnp.mean(values, axis=0)
            replicate = (jnp.sum(values, axis=0)[None, :] - values) / (members - 1)
            response = 0.5 * jnp.log(mean)
            design = jnp.stack((jnp.ones_like(r), jnp.log(r)), axis=1)
            fit_weights = (
                jnp.ones_like(r)
                if weights is None
                else jnp.asarray(real_number_array(weights, "weights"))[selected]
            )
            fit = solve_weighted_least_squares(
                design, response, weights=fit_weights, min_samples=self.minimum_points
            )

            def deleted(y: Array) -> Array:
                return solve_weighted_least_squares(
                    design,
                    0.5 * jnp.log(y),
                    weights=fit_weights,
                    min_samples=self.minimum_points,
                ).coefficients

            deleted_coefficients = jax.vmap(deleted)(replicate)
            exponents = deleted_coefficients[:, 1]
            se = jnp.sqrt(
                (members - 1) / members * jnp.sum((exponents - jnp.mean(exponents)) ** 2)
            )
            condition = float(fit.condition_number)
            if (
                not bool(fit.valid)
                or not np.isfinite(np.asarray(deleted_coefficients)).all()
            ):
                reason = "Rank-deficient or nonfinite logarithmic least-squares fit."
            elif condition > self.maximum_condition_number:
                reason = "Least-squares conditioning exceeds the declared limit."
            status = parse(
                "abstained" if reason else "complete", ScalingEvidenceStatus, "status"
            )
            fits.append(
                StructureFunctionScalingFit(
                    float(fit.coefficients[1]),
                    float(se),
                    float(jnp.exp(fit.coefficients[0])),
                    float(jnp.sqrt(jnp.mean(fit.residual**2))),
                    condition,
                    status,
                    reason,
                    exponents,
                )
            )
            windows.append((r, jnp.sqrt(mean)))
        complete = all(f.status == "complete" for f in fits)
        if not complete:
            return InstantaneousStructureFunctionScalingEvidence(
                tuple(fits),
                np.nan,
                np.nan,
                False,
                jnp.full((len(fits), len(fits)), jnp.nan),
            )
        common = sum(f.exponent for f in fits) / len(fits)
        envelope = max(float(jnp.max(s / r**common)) for r, s in windows)
        exponents = jnp.stack(tuple(f.jackknife_exponents for f in fits))
        differences = exponents[:, None, :] - exponents[None, :, :]
        errors = jnp.sqrt(
            (members - 1)
            / members
            * jnp.sum(
                (differences - jnp.mean(differences, axis=-1, keepdims=True)) ** 2,
                axis=-1,
            )
        )
        point = np.asarray([f.exponent for f in fits])
        consistent = bool(
            np.all(
                np.abs(point[:, None] - point[None, :]) <= 2 * np.asarray(errors) + 1e-12
            )
        )
        return InstantaneousStructureFunctionScalingEvidence(
            tuple(fits), common, envelope, consistent, errors
        )

    def evaluate_time_quadrature(
        self,
        results: Sequence[Sequence[PeriodicIncrementMomentResult]],
        discretizations: Sequence[TensorSpectralDiscretization],
        time_weights: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
    ) -> TimeIntegratedStructureFunctionScalingEvidence:
        """Explicit weighted time-integrated M2; no implicit trajectory quadrature."""
        trajectories = tuple(tuple(items) for items in results)
        quadrature = real_number_array(time_weights, "time_weights")
        if (
            quadrature.ndim != 1
            or not quadrature.size
            or not np.isfinite(quadrature).all()
            or np.any(quadrature < 0)
            or quadrature.sum() <= 0
            or any(len(items) != quadrature.size for items in trajectories)
        ):
            raise ValueError(
                "Time quadrature requires finite nonnegative explicit weights and matching saves."
            )
        combined = []
        for items in trajectories:
            if any(
                not np.array_equal(np.asarray(m.radii), np.asarray(items[0].radii))
                or m.values.shape != items[0].values.shape
                for m in items
            ):
                raise ValueError(
                    "Time quadrature requires matching radial queries and member ordering."
                )
            value = jnp.sum(
                jnp.asarray(quadrature)[:, None, None]
                * jnp.stack(tuple(m.values for m in items)),
                axis=0,
            )
            combined.append(
                eqx.tree_at(
                    lambda m: (m.values, m.finite),
                    items[0],
                    (value, jnp.all(jnp.stack(tuple(m.finite for m in items)), axis=0)),
                )
            )
        fitted = self.evaluate(combined, discretizations, weights=weights)
        return TimeIntegratedStructureFunctionScalingEvidence(
            fitted.fits,
            fitted.common_exponent,
            fitted.envelope_constant,
            fitted.consistent_within_campaign,
            fitted.paired_exponent_standard_errors,
            tuple(float(w) for w in quadrature),
        )


def prepare_periodic_flow_ensemble(
    dynamics: CompiledIncompressibleSpectralDynamics,
    method: ETDRKMethod
    | LESStabilityGuardedETDRKMethod
    | CourantGuardedETDRKMethod
    | PreparedETDRKMethod
    | PreparedLESStabilityGuardedETDRKMethod
    | PreparedCourantGuardedETDRKMethod,
    initial_state: PeriodicFlowInitialState,
    /,
    *,
    start_time: float,
    end_time: float,
    step_size: float,
    save_every: int = 1,
    member_chunk_size: int = 8,
    maximum_output_bytes: int = 2**31,
) -> PreparedPeriodicFlowEnsemble:
    """Bind one native fixed-step template and its scientific member owner."""

    if dynamics.dynamic_les is not None:
        raise ValueError(
            "Transactional dynamic LES cannot run as a velocity-only fixed-step ensemble."
        )

    if dynamics.discretization.prepared_id != initial_state.discretization_id:
        raise ValueError(
            "Initial state and dynamics require identical discretization identity."
        )
    coordinates = HermitianSpectralCoordinates(
        dynamics.discretization, component_shape=(len(dynamics.discretization.axes),)
    )
    if isinstance(method, ETDRKMethod):
        bound = method.prepare(dynamics.semilinear_drift, coordinates=coordinates)
    elif isinstance(method, (LESStabilityGuardedETDRKMethod, CourantGuardedETDRKMethod)):
        bound = method.prepare(dynamics, coordinates=coordinates)
    else:
        bound = method
    base = _bind_periodic_method(dynamics, bound)
    if base.coordinates is None:
        raise ValueError("Periodic ensembles require Hermitian spectral coordinates.")
    if base.coordinates.discretization.prepared_id != initial_state.discretization_id:
        raise ValueError("Prepared coordinates have a different discretization identity.")
    states = initial_state.velocity
    problem = FixedStepProblem(
        bound,
        states[0],
        t0=start_time,
        t1=end_time,
        step_size=step_size,
        discretization_bundle=dynamics.discretization_bundle,
    )
    labels = _member_labels(initial_state)
    ensemble = FixedStepEnsemblePlan(
        problem,
        states,
        labels,
        member_chunk_size=member_chunk_size,
        maximum_output_bytes=maximum_output_bytes,
    ).prepare(save_every=save_every)
    return PreparedPeriodicFlowEnsemble(ensemble, initial_state)
