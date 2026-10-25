#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Diagnostic free-energy profiles of adaptive histories.

These are explicitly *adaptive* diagnostics, never equilibrium
``FreeEnergyResult`` values: every result records its estimator, owner
temperature, storage policy, support/connectivity and integrability evidence
and whether the usual profile formula is admitted for that history.

* ABF: ``A`` solves the least-squares gradient problem on the largest connected
  supported grid component (wrapping across every axis whose bins close a CV
  period), with one fixed gauge, through a native matrix-free conjugate-gradient
  solve of the grid-graph normal equations that stops only on the true
  residual: the finite-difference incidence is a masked shift action, so storage
  stays proportional to the grid.  Results retain native solve evidence and the
  gradient misfit (integrability residual).
* Metadynamics: ``A(s) = -V(s)`` (standard) or ``-(gamma/(gamma-1)) V(s)``
  (well-tempered) up to a constant, admitted only for an intact (non-FIFO)
  history deposited at the declared sampling temperature.
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    FailurePolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    PCG,
    solve,
    TolerancePolicy,
)
from ._abf import (
    _abf_state_content,
    ABFLinearSolveEvidence,
    AtomisticABFState,
    PreparedABF,
)
from ._bias_history import (
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    FrozenBiasHistory,
    HillKernel,
)
from ._collective_variable import CollectiveVariableMetric


class AdaptiveProfileEstimator(StrEnum):
    ABF = "abf"
    METADYNAMICS = "metadynamics"
    WELL_TEMPERED_METADYNAMICS = "well-tempered-metadynamics"


class AdaptiveProfileResult(StrictModule, NonTrainableState):
    """Grid profile ``A`` (NaN outside the admitted support) with assumptions.

    ``nodes`` are per-CV node coordinates; ``free_energy`` has the grid shape.
    ``periodic_axes`` marks the axes whose nodes close one CV period (the last
    and first node are neighbours through the seam).
    ``formula_admitted`` is false whenever the estimator's usual profile formula
    does not apply to this history; ``assumptions`` and ``refusals`` say why.
    ``gradient`` keeps the node mean-force data of ABF profiles (``None``
    otherwise) so a conservative integrated realization can use it.
    """

    nodes: tuple[Array, ...]
    free_energy: Array
    gradient: Array | None
    support: Array
    sample_counts: Array | None
    integrability_residual: Array
    integration_solve: ABFLinearSolveEvidence | None
    metrics: tuple[CollectiveVariableMetric, ...]
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    estimator: AdaptiveProfileEstimator = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    storage: str = eqx.field(static=True)
    connected: bool = eqx.field(static=True)
    formula_admitted: bool = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    refusals: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: tuple[np.ndarray, ...],
        free_energy: np.ndarray,
        support: np.ndarray,
        /,
        *,
        gradient: np.ndarray | None,
        sample_counts: np.ndarray | None,
        integrability_residual: float,
        integration_solve: ABFLinearSolveEvidence | None,
        metrics: tuple[CollectiveVariableMetric, ...],
        periodic_axes: tuple[bool, ...],
        estimator: AdaptiveProfileEstimator,
        temperature: float,
        storage: str,
        connected: bool,
        refusals: tuple[str, ...],
        assumptions: tuple[str, ...],
        source_id: str,
    ) -> None:
        periodic = tuple(bool(flag) for flag in periodic_axes)
        if len(periodic) != len(nodes):
            raise ValueError("periodic_axes must have one flag per node axis.")
        self.nodes = tuple(jnp.asarray(node) for node in nodes)
        self.free_energy = jnp.asarray(free_energy)
        self.gradient = None if gradient is None else jnp.asarray(gradient)
        self.support = jnp.asarray(support, dtype=jnp.bool_)
        self.sample_counts = None if sample_counts is None else jnp.asarray(sample_counts)
        self.integrability_residual = jnp.asarray(integrability_residual)
        self.integration_solve = integration_solve
        self.metrics = metrics
        self.periodic_axes = periodic
        self.estimator = estimator
        self.temperature = float(temperature)
        self.storage = str(storage)
        self.connected = bool(connected)
        self.formula_admitted = not refusals
        self.assumptions = tuple(assumptions)
        self.refusals = tuple(refusals)
        self.source_id = str(source_id)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "adaptive-profile",
                "estimator": estimator.value,
                "source": source_id,
                "arrays": array_tree_fingerprint(
                    {
                        "nodes": list(nodes),
                        "free_energy": free_energy,
                        "support": support,
                    }
                ),
                "periodic_axes": list(periodic),
                "temperature": float(temperature).hex(),
                "storage": storage,
                "refusals": list(refusals),
            }
        )


def _components(support: np.ndarray, periodic: tuple[bool, ...], /) -> list[list[tuple]]:
    shape = support.shape
    seen = np.zeros(shape, dtype=np.bool_)
    components: list[list[tuple]] = []
    for start in zip(*np.nonzero(support), strict=True):
        if seen[start]:
            continue
        component = []
        queue = deque([start])
        seen[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for axis in range(len(shape)):
                for step in (-1, 1):
                    neighbor = list(node)
                    neighbor[axis] += step
                    if periodic[axis]:
                        neighbor[axis] %= shape[axis]
                    elif not 0 <= neighbor[axis] < shape[axis]:
                        continue
                    candidate = tuple(neighbor)
                    if support[candidate] and not seen[candidate]:
                        seen[candidate] = True
                        queue.append(candidate)
        components.append(component)
    return components


def _integrate_gradient(
    means: np.ndarray,
    kept: np.ndarray,
    widths: np.ndarray,
    periodic: tuple[bool, ...],
    /,
) -> tuple[np.ndarray, float, ABFLinearSolveEvidence | None]:
    """Least-squares potential of a node gradient field on one grid component.

    Edges join each kept node to its kept successor along every axis
    (through the seam on periodic axes) with the mean of the two node
    gradients as the finite-difference target.  ``D`` is applied as a masked
    shift and ``D^T D + e_g e_g^T + diag(1 - kept)`` (gauge ``A = 0`` at the
    first kept node; nodes outside the component are decoupled identities)
    is solved matrix-free by native unpreconditioned PCG, so storage stays
    ``O(nodes * dimension)``.  On a connected component that operator is
    positive definite by construction.  Native PCG certifies the normwise true
    residual; a componentwise stopping test (Lineax CG) cannot be met at the
    gauge node or at zero crossings of ``A`` or of the load, where it demands
    an exactly zero step or residual and exhausts the step budget.
    """

    shape = kept.shape
    dimension = len(shape)
    dtype = np.dtype(means.dtype)
    masks = []
    targets = []
    for axis in range(dimension):
        mask = kept & np.roll(kept, -1, axis=axis)
        if shape[axis] == 1:
            mask = np.zeros(shape, dtype=np.bool_)
        elif not periodic[axis]:
            last: list[slice | int] = [slice(None)] * dimension
            last[axis] = -1
            mask[tuple(last)] = False
        component = means[..., axis]
        ahead = np.roll(component, -1, axis=axis)
        targets.append(np.where(mask, 0.5 * (component + ahead), 0.0))
        masks.append(mask)
    edges = int(np.sum(np.stack(masks)))
    if edges == 0:
        return np.where(kept, 0.0, np.nan).astype(dtype), 0.0, None
    edge_mask = jnp.asarray(np.stack(masks), dtype=dtype)
    target = jnp.asarray(np.stack(targets), dtype=dtype)
    scales = tuple(float(1.0 / width) for width in widths)
    gauge = np.zeros(shape, dtype=dtype)
    gauge[tuple(np.argwhere(kept)[0])] = 1.0
    diagonal = jnp.asarray(gauge + (~kept).astype(dtype))

    def difference(values: Array) -> Array:
        return edge_mask * jnp.stack(
            tuple(
                (jnp.roll(values, -1, axis=axis) - values) * scales[axis]
                for axis in range(dimension)
            )
        )

    def divergence(flux: Array) -> Array:
        total = jnp.zeros(shape, dtype=dtype)
        for axis in range(dimension):
            weighted = edge_mask[axis] * flux[axis] * scales[axis]
            total = total + jnp.roll(weighted, 1, axis=axis) - weighted
        return total

    def normal(values: Array) -> Array:
        return divergence(difference(values)) + diagonal * values

    space = ArraySpace(shape, dtype=dtype)
    operator_id = canonical_fingerprint(
        {
            "kind": "abf-profile-gradient-normal-operator",
            "shape": list(shape),
            "periodic": list(periodic),
            "widths": array_tree_fingerprint(np.asarray(widths)),
            "kept": array_tree_fingerprint(kept),
        }
    )
    operator = FunctionLinearOperator(
        normal,
        source=space,
        target=space,
        transpose_action=normal,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=operator_id,
    )
    nodes = int(np.prod(shape))
    # ~1.6e-10 in float64: above the recursive-residual stagnation floor of CG;
    # native PCG confirms it on the true residual ``b - N A``.
    tolerance = float(np.finfo(dtype).eps) ** 0.625
    result = solve(
        LinearSystem(operator, problem_id=f"{operator_id}:system"),
        divergence(target),
        policy=LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=tolerance,
                absolute=float(np.finfo(dtype).tiny),
                max_steps=10 * nodes,
            ),
            failure=FailurePolicy("status"),
        ),
    )
    solution = np.asarray(result.value)
    misfit = np.asarray(difference(jnp.asarray(solution)) - target)
    residual = float(np.sqrt(np.sum(misfit * misfit) / edges))
    return (
        np.where(kept, solution, np.nan),
        residual,
        ABFLinearSolveEvidence(result),
    )


def abf_profile(
    abf: PreparedABF,
    state: AtomisticABFState,
    /,
    *,
    minimum_samples: int | None = None,
) -> AdaptiveProfileResult:
    """Integrate the frozen ABF mean-force field into a diagnostic profile.

    Bins with at least ``minimum_samples`` (default: the plan's
    ``min_samples``, at least one) are supported; the largest connected
    supported component is integrated with ``A = 0`` at its first node.
    """

    if state.owner_identity.fingerprint() != abf.plan.abf_id:
        raise ValueError("ABF state belongs to another ABF plan.")
    plan = abf.plan
    threshold = max(
        1,
        int(np.asarray(plan.min_samples))
        if minimum_samples is None
        else int(minimum_samples),
    )
    temperature = float(np.asarray(plan.temperature))
    shape = plan.bins
    dimension = len(shape)
    counts = np.asarray(state.counts).reshape(shape)
    means = np.asarray(abf.mean_force(state)).reshape(shape + (dimension,))
    nodes = tuple(np.asarray(center) for center in plan.centers())
    widths = np.asarray(plan.widths)
    periodic = tuple(
        metric.period is not None
        and bool(
            np.isclose(float(plan.maximum[axis] - plan.minimum[axis]), metric.period)
        )
        for axis, metric in enumerate(plan.variables.metrics)
    )
    support = (counts >= threshold) & np.all(np.isfinite(means), axis=-1)
    components = _components(support, periodic)
    refusals: list[str] = []
    if not components:
        raise ValueError(
            "No ABF bin reaches the support threshold; nothing to integrate."
        )
    largest = max(components, key=len)
    connected = len(components) == 1
    if not connected:
        refusals.append("disconnected-support")
    kept = np.zeros(shape, dtype=np.bool_)
    for node in largest:
        kept[node] = True
    free_energy, residual, integration_solve = _integrate_gradient(
        means, kept, widths, periodic
    )
    if integration_solve is not None and not bool(
        np.all(np.asarray(integration_solve.status) == int(LinearSolveStatus.SUCCESS))
    ):
        refusals.append("integration-solve-failed")
    if (
        not np.all(np.isfinite(free_energy[kept]))
        and "integration-solve-failed" not in refusals
    ):
        refusals.append("integration-solve-failed")
    assumptions = (
        f"base-measure:{plan.measure.value}",
        f"base-density:{plan.measure.density}",
        f"temperature:{temperature!r}",
        f"minimum-samples:{threshold}",
        "adaptive-history:not-equilibrium-samples",
    )
    return AdaptiveProfileResult(
        nodes,
        free_energy,
        kept,
        gradient=means,
        sample_counts=counts,
        integrability_residual=residual,
        integration_solve=integration_solve,
        metrics=plan.variables.metrics,
        periodic_axes=periodic,
        estimator=AdaptiveProfileEstimator.ABF,
        temperature=temperature,
        storage="bins",
        connected=connected,
        refusals=tuple(refusals),
        assumptions=assumptions,
        source_id=canonical_fingerprint(
            {
                "abf": abf.prepared_id,
                "state": array_tree_fingerprint(_abf_state_content(state)),
            }
        ),
    )


def metadynamics_profile(
    frozen: FrozenBiasHistory,
    metrics: tuple[CollectiveVariableMetric, ...],
    nodes: tuple[ArrayLike, ...],
    /,
    *,
    sampled_temperature: float,
    owner_key: int = 0,
) -> AdaptiveProfileResult:
    """Profile of a frozen hill history on a tensor grid of CV nodes."""

    plan: AtomisticBiasHistoryPlan = frozen.plan
    state: AtomisticBiasHistoryState = frozen.state
    if plan.kernel is HillKernel.ALIGNED_MSD:
        raise ValueError(
            "A configuration-kernel history has no CV-grid profile; analyse it "
            "through frozen-history reweighting instead."
        )
    grid = tuple(np.asarray(node, dtype=np.float64).reshape((-1,)) for node in nodes)
    if len(grid) != plan.dimension or len(metrics) != plan.dimension:
        raise ValueError("Profile nodes and metrics must align with the history.")
    points = np.asarray(list(product(*grid)), dtype=np.float64)
    energy = np.asarray(
        jax.vmap(lambda point: plan.evaluate(point, metrics, state, owner_key).energy)(
            jnp.asarray(points)
        )
    ).reshape(tuple(node.size for node in grid))
    refusals: list[str] = []
    if frozen.fifo_overwritten:
        refusals.append("fifo-overwritten-history")
    if plan.ramp_depositions:
        refusals.append("ramped-kernels-not-at-full-height")
    if plan.bias_factor is None:
        estimator = AdaptiveProfileEstimator.METADYNAMICS
        scale = 1.0
        temperature = float(sampled_temperature)
    else:
        estimator = AdaptiveProfileEstimator.WELL_TEMPERED_METADYNAMICS
        gamma = float(np.asarray(plan.bias_factor))
        scale = gamma / (gamma - 1.0)
        temperature = float(np.asarray(plan.temperature))
        if not np.isclose(temperature, float(sampled_temperature)):
            refusals.append("history-owner-temperature-differs-from-sampling")
    free_energy = -scale * energy
    free_energy = free_energy - np.min(free_energy)
    assumptions = (
        f"storage:{plan.storage.value}",
        f"ownership:{plan.ownership.value}:{owner_key}",
        f"kernel:{plan.kernel.value}",
        f"deposits:{int(np.asarray(state.deposits))}",
        "adaptive-history:convergence-not-certified",
    )
    periodic = tuple(
        metric.period is not None
        and node.size >= 2
        and bool(np.allclose(np.diff(node), node[1] - node[0], rtol=1e-9, atol=0.0))
        and bool(np.isclose(node.size * (node[1] - node[0]), metric.period))
        for node, metric in zip(grid, metrics, strict=True)
    )
    return AdaptiveProfileResult(
        grid,
        free_energy,
        np.ones(free_energy.shape, dtype=np.bool_),
        gradient=None,
        sample_counts=None,
        integrability_residual=0.0,
        integration_solve=None,
        metrics=metrics,
        periodic_axes=periodic,
        estimator=estimator,
        temperature=temperature,
        storage=plan.storage.value,
        connected=True,
        refusals=tuple(refusals),
        assumptions=assumptions,
        source_id=frozen.digest,
    )


__all__ = [
    "AdaptiveProfileEstimator",
    "AdaptiveProfileResult",
    "abf_profile",
    "metadynamics_profile",
]
