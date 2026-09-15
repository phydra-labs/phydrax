#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._oriented_path import (
    CellBoundaryPathPlan,
    OrientedEdgePathPlan,
)
from ...graph._matrix_gauge import MatrixGaugeLinkSpace, path_holonomy
from ...metrix._complex_matrix_manifold import SpecialUnitaryGroup, UnitaryGroup
from ._lattice_action import (
    AbstractLatticeEuclideanAction,
    lattice_action_local_gradient,
    LatticeActionEvidence,
)
from ._wilson_gauge import WilsonGaugeAction


ImprovedGaugeFamily: TypeAlias = Literal["wilson", "symanzik", "iwasaki", "dbw2"]


class GaugeLoopTerm(StrictModule, NonTrainableState):
    """One named fixed path family and its path-aligned real coefficients."""

    paths: OrientedEdgePathPlan
    coefficients: Array
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        paths: OrientedEdgePathPlan,
        coefficients: ArrayLike,
        /,
        *,
        name: str,
    ):
        if not isinstance(paths, OrientedEdgePathPlan) or not paths.require_closed:
            raise TypeError("Gauge loop terms require closed OrientedEdgePathPlan paths.")
        name_ = str(name).strip()
        if not name_:
            raise ValueError("Gauge loop term name must be non-empty.")
        values = np.asarray(coefficients, dtype=float)
        if values.shape == ():
            values = np.full((paths.num_paths,), float(values))
        if values.shape != (paths.num_paths,):
            raise ValueError("coefficients must be scalar or provide one value per path.")
        if np.any(~np.isfinite(values)):
            raise ValueError("Gauge loop coefficients must be finite.")
        self.paths = paths
        self.coefficients = jnp.asarray(values)
        self.name = name_
        self.term_id = canonical_fingerprint(
            {
                "kind": "gauge-loop-term",
                "paths": paths.path_plan_id,
                "coefficients": array_tree_fingerprint(values),
                "name": name_,
                "trace": "fundamental-real-normalized",
            }
        )


class ImprovedGaugeAction(AbstractLatticeEuclideanAction):
    """Composable real normalized-trace action over closed Wilson-loop families."""

    link_space: MatrixGaugeLinkSpace
    terms: tuple[GaugeLoopTerm, ...]
    coefficients: Array
    edge_loops: Array
    edge_loops_valid: Array
    geometry: Any
    evidence: LatticeActionEvidence
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    loop_count: int = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        terms: Sequence[GaugeLoopTerm],
        /,
        *,
        maximum_loops_per_link: int = 128,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(link_space.group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Improved gauge actions require U(N) or SU(N).")
        terms_ = tuple(terms)
        if not terms_ or not all(isinstance(term, GaugeLoopTerm) for term in terms_):
            raise TypeError("terms must contain at least one GaugeLoopTerm.")
        if len({term.name for term in terms_}) != len(terms_):
            raise ValueError("Gauge loop term names must be unique.")
        if any(
            term.paths.topology_id != link_space.topology.topology_id for term in terms_
        ):
            raise ValueError("Every gauge loop term must share the link topology.")
        resource_limit = int(maximum_loops_per_link)
        if resource_limit < 1 or resource_limit > 4096:
            raise ValueError("maximum_loops_per_link must lie in [1, 4096].")
        coefficients_host = np.concatenate(
            tuple(np.asarray(term.coefficients) for term in terms_)
        )
        affected: list[list[int]] = [[] for _ in range(link_space.num_edges)]
        loop_offset = 0
        for term in terms_:
            edges = np.asarray(term.paths.edge_indices)
            valid = np.asarray(term.paths.valid)
            for path in range(term.paths.num_paths):
                for edge in np.unique(edges[path, valid[path]]):
                    affected[int(edge)].append(loop_offset + path)
            loop_offset += term.paths.num_paths
        capacity = max(1, max((len(indices) for indices in affected), default=0))
        if capacity > resource_limit:
            raise ValueError("Loop incidence exceeds maximum_loops_per_link.")
        edge_loops = np.zeros((link_space.num_edges, capacity), dtype=np.int32)
        edge_valid = np.zeros((link_space.num_edges, capacity), dtype=bool)
        for edge, indices in enumerate(affected):
            edge_loops[edge, : len(indices)] = indices
            edge_valid[edge, : len(indices)] = True
        action_id = canonical_fingerprint(
            {
                "kind": "composable-improved-gauge-action",
                "link_space": link_space.link_space_id,
                "terms": [term.term_id for term in terms_],
                "loop_count": int(coefficients_host.size),
                "maximum_loops_per_link": resource_limit,
                "convention": "minus-coefficient-times-real-normalized-trace",
            }
        )
        self.link_space = link_space
        self.terms = terms_
        self.coefficients = jnp.asarray(coefficients_host)
        self.edge_loops = jnp.asarray(edge_loops)
        self.edge_loops_valid = jnp.asarray(edge_valid)
        self.geometry = link_space.geometry
        self.evidence = LatticeActionEvidence(
            reference_measure="product-haar",
            real_valued=True,
            bounded_below=True,
            normalizable=True,
            additive_constant=float(np.sum(coefficients_host)),
            evidence_id=canonical_fingerprint(
                {
                    "kind": "compact-loop-action-measure-evidence",
                    "action": action_id,
                    "compact_group": link_space.group.group_id,
                    "finite_loop_count": int(coefficients_host.size),
                }
            ),
        )
        self.topology_id = link_space.topology.topology_id
        self.field_space_id = link_space.field_space.field_space_id
        self.configuration_shape = link_space.configuration_shape
        self.local_coordinate_shape = link_space.local_coordinate_shape
        self.loop_count = int(coefficients_host.size)
        self.action_id = action_id

    def _links(self, links: ArrayLike, /) -> Array:
        values = jnp.asarray(links)
        if values.shape != self.configuration_shape:
            raise ValueError(
                f"links must have shape {self.configuration_shape}; got {values.shape}."
            )
        return values

    def loop_traces(self, links: ArrayLike, /) -> Array:
        """Return real normalized traces in canonical term then path order."""
        values = self._links(links)
        dimension = self.link_space.point_shape[0]
        traces = tuple(
            jnp.real(
                jnp.trace(
                    path_holonomy(self.link_space, values, term.paths),
                    axis1=-2,
                    axis2=-1,
                )
            )
            / dimension
            for term in self.terms
        )
        return jnp.concatenate(traces)

    def loop_contributions(self, links: ArrayLike, /) -> Array:
        return -self.coefficients * self.loop_traces(links)

    def action(self, links: PyTree[Array], /) -> Array:
        return jnp.sum(self.loop_contributions(links))

    def canonical_action(self, links: ArrayLike, /) -> Array:
        return self.action(links) + self.evidence.additive_constant

    def local_action(self, links: ArrayLike, edge: ArrayLike, /) -> Array:
        """Return the exact full-action terms whose closed paths touch ``edge``."""
        contributions = self.loop_contributions(links)
        edge_ = jnp.asarray(edge, dtype=jnp.int32)
        indices = self.edge_loops[edge_]
        return jnp.sum(
            jnp.where(self.edge_loops_valid[edge_], contributions[indices], 0.0)
        )


MatrixGaugeLoopAction: TypeAlias = ImprovedGaugeAction | WilsonGaugeAction


_FAMILY_RECTANGLE_COEFFICIENT = {
    "wilson": 0.0,
    "symanzik": -1.0 / 12.0,
    "iwasaki": -0.331,
    "dbw2": -1.4088,
}


def standard_gauge_loop_action(
    link_space: MatrixGaugeLinkSpace,
    plaquettes: CellBoundaryPathPlan | OrientedEdgePathPlan,
    rectangles: OrientedEdgePathPlan | None = None,
    /,
    *,
    beta: float,
    family: ImprovedGaugeFamily,
    maximum_loops_per_link: int = 128,
) -> WilsonGaugeAction | ImprovedGaugeAction:
    """Build Wilson or standard plaquette-plus-rectangle improved actions.

    Coefficients use ``c0 + 8*c1 = 1``.  ``symanzik`` is tree-level
    Lüscher--Weisz; Iwasaki and DBW2 use their conventional rectangle values.
    """
    beta_ = float(beta)
    if not isfinite(beta_) or beta_ < 0.0:
        raise ValueError("beta must be finite and non-negative.")
    if family not in _FAMILY_RECTANGLE_COEFFICIENT:
        raise ValueError("Unknown improved gauge family.")
    if isinstance(plaquettes, CellBoundaryPathPlan):
        plaquette_paths = plaquettes.paths
        plaquette_boundaries = plaquettes
    elif isinstance(plaquettes, OrientedEdgePathPlan):
        plaquette_paths = plaquettes
        plaquette_boundaries = None
    else:
        raise TypeError("plaquettes must be cell-boundary or oriented-edge paths.")
    if family == "wilson":
        if plaquette_boundaries is not None:
            return WilsonGaugeAction(
                link_space, plaquette_boundaries, plaquette_couplings=beta_
            )
        return ImprovedGaugeAction(
            link_space,
            (GaugeLoopTerm(plaquette_paths, beta_, name="plaquette"),),
            maximum_loops_per_link=maximum_loops_per_link,
        )
    if rectangles is None:
        raise ValueError(f"{family} requires explicit certified rectangle paths.")
    if not isinstance(rectangles, OrientedEdgePathPlan) or not rectangles.require_closed:
        raise TypeError("rectangles must be a closed OrientedEdgePathPlan.")
    c1 = _FAMILY_RECTANGLE_COEFFICIENT[family]
    c0 = 1.0 - 8.0 * c1
    return ImprovedGaugeAction(
        link_space,
        (
            GaugeLoopTerm(plaquette_paths, beta_ * c0, name="plaquette"),
            GaugeLoopTerm(rectangles, beta_ * c1, name="rectangle"),
        ),
        maximum_loops_per_link=maximum_loops_per_link,
    )


class GaugeFlowObservables(StrictModule):
    """Finite scalar diagnostics at one gauge-flow time."""

    reduced_action_density: Array
    canonical_action_density: Array
    mean_loop: Array
    force_norm: Array
    membership: Array
    finite: Array


def gauge_flow_observables(
    action: MatrixGaugeLoopAction,
    links: ArrayLike,
    /,
) -> GaugeFlowObservables:
    if not isinstance(action, (ImprovedGaugeAction, WilsonGaugeAction)):
        raise TypeError("action must be ImprovedGaugeAction or WilsonGaugeAction.")
    values = action._links(links)
    if isinstance(action, ImprovedGaugeAction):
        traces = action.loop_traces(values)
        loop_count = action.loop_count
    else:
        traces = action.plaquette_traces(values)
        loop_count = action.num_plaquettes
    reduced = action.action(values) / loop_count
    canonical = action.canonical_action(values) / loop_count
    force = lattice_action_local_gradient(action, values)
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(traces))
        & jnp.isfinite(reduced)
        & jnp.all(jnp.isfinite(force))
    )
    return GaugeFlowObservables(
        reduced,
        canonical,
        jnp.mean(traces),
        jnp.linalg.norm(force),
        action.link_space.contains(values),
        finite,
    )


class GaugeGradientFlowStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_LINKS = 1
    NONFINITE_FLOW = 2
    GROUP_MEMBERSHIP_LOST = 3
    ACTION_DESCENT_FAILED = 4


class GaugeGradientFlowEvidence(StrictModule, NonTrainableState):
    status: Array
    finite: Array
    membership: Array
    action_nonincreasing: Array
    accepted_steps: Array
    maximum_action_increase: Array
    action_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(GaugeGradientFlowStatus.SUCCESS)


class GaugeGradientFlowResult(StrictModule):
    links: Array
    action_history: Array
    accepted_step_sizes: Array
    observables: GaugeFlowObservables
    evidence: GaugeGradientFlowEvidence
    flow_time: Array
    plan_id: str = eqx.field(static=True)


class GaugeGradientFlowPlan(StrictModule, NonTrainableState):
    """Fixed-step Lie-midpoint descent with bounded deterministic backtracking."""

    action: MatrixGaugeLoopAction
    step_size: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    descent_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: MatrixGaugeLoopAction,
        /,
        *,
        step_size: float,
        steps: int,
        maximum_backtracks: int = 8,
        descent_tolerance: float = 1.0e-10,
    ):
        if not isinstance(action, (ImprovedGaugeAction, WilsonGaugeAction)):
            raise TypeError("action must be ImprovedGaugeAction or WilsonGaugeAction.")
        dt = float(step_size)
        count = int(steps)
        backtracks = int(maximum_backtracks)
        tolerance = float(descent_tolerance)
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if count < 1 or count > 1_000_000:
            raise ValueError("steps must lie in [1, 1000000].")
        if backtracks < 0 or backtracks > 32:
            raise ValueError("maximum_backtracks must lie in [0, 32].")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("descent_tolerance must be finite and non-negative.")
        self.action = action
        self.step_size = dt
        self.steps = count
        self.maximum_backtracks = backtracks
        self.descent_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "group-preserving-gauge-gradient-flow",
                "action": action.action_id,
                "step_size": dt,
                "steps": count,
                "maximum_backtracks": backtracks,
                "descent_tolerance": tolerance,
                "integrator": "lie-midpoint-bounded-backtracking",
            }
        )

    def _candidate(self, links: Array, step_size: Array, /) -> Array:
        gradient = lattice_action_local_gradient(self.action, links)
        midpoint = self.action.geometry.retract(links, -0.5 * step_size * gradient)
        midpoint_gradient = lattice_action_local_gradient(self.action, midpoint)
        return self.action.geometry.retract(links, -step_size * midpoint_gradient)

    def flow(self, links: ArrayLike, /) -> GaugeGradientFlowResult:
        values = self.action._links(links)
        initial_action = self.action.action(values)
        initial_membership = self.action.link_space.contains(values)
        dtype = jnp.real(values).dtype

        def advance(
            carry: tuple[Array, Array, Array, Array], _: None
        ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
            current, current_action, all_members, all_finite = carry

            def attempt(
                selection: tuple[Array, Array, Array, Array], index: Array
            ) -> tuple[tuple[Array, Array, Array, Array], None]:
                chosen, chosen_action, chosen_step, found = selection
                step = jnp.asarray(self.step_size, dtype=dtype) * (
                    jnp.asarray(0.5, dtype=dtype) ** index
                )
                candidate = self._candidate(current, step)
                candidate_action = self.action.action(candidate)
                finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(candidate_action)
                member = self.action.link_space.contains(candidate)
                descent = candidate_action <= (
                    current_action
                    + jnp.asarray(self.descent_tolerance, dtype=current_action.dtype)
                )
                choose = (~found) & finite & member & descent
                next_selection = (
                    jnp.where(choose, candidate, chosen),
                    jnp.where(choose, candidate_action, chosen_action),
                    jnp.where(choose, step, chosen_step),
                    found | choose,
                )
                return next_selection, None

            selection, _ = jax.lax.scan(
                attempt,
                (
                    current,
                    current_action,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                ),
                jnp.arange(self.maximum_backtracks + 1),
            )
            proposed, proposed_action, accepted_step, accepted = selection
            member = self.action.link_space.contains(proposed)
            finite = jnp.all(jnp.isfinite(proposed)) & jnp.isfinite(proposed_action)
            next_carry = (
                proposed,
                proposed_action,
                all_members & member,
                all_finite & finite,
            )
            return next_carry, (proposed_action, accepted_step, accepted, member)

        final, history = jax.lax.scan(
            advance,
            (
                values,
                initial_action,
                initial_membership,
                jnp.all(jnp.isfinite(values)) & jnp.isfinite(initial_action),
            ),
            xs=None,
            length=self.steps,
        )
        final_links, _, membership, finite = final
        actions, step_sizes, accepted, _ = history
        action_history = jnp.concatenate((initial_action[None], actions))
        increases = action_history[1:] - action_history[:-1]
        nonincreasing = jnp.all(increases <= self.descent_tolerance)
        all_accepted = jnp.all(accepted)
        status = jnp.where(
            ~initial_membership,
            int(GaugeGradientFlowStatus.INVALID_INITIAL_LINKS),
            jnp.where(
                ~finite,
                int(GaugeGradientFlowStatus.NONFINITE_FLOW),
                jnp.where(
                    ~membership,
                    int(GaugeGradientFlowStatus.GROUP_MEMBERSHIP_LOST),
                    jnp.where(
                        ~(all_accepted & nonincreasing),
                        int(GaugeGradientFlowStatus.ACTION_DESCENT_FAILED),
                        int(GaugeGradientFlowStatus.SUCCESS),
                    ),
                ),
            ),
        )
        evidence = GaugeGradientFlowEvidence(
            status,
            finite,
            membership,
            nonincreasing,
            jnp.sum(accepted.astype(jnp.int32)),
            jnp.max(jnp.maximum(increases, 0.0)),
            self.action.action_id,
            self.plan_id,
        )
        return GaugeGradientFlowResult(
            final_links,
            action_history,
            step_sizes,
            gauge_flow_observables(self.action, final_links),
            evidence,
            jnp.sum(step_sizes),
            self.plan_id,
        )


__all__ = [
    "GaugeFlowObservables",
    "GaugeGradientFlowEvidence",
    "GaugeGradientFlowPlan",
    "GaugeGradientFlowResult",
    "GaugeGradientFlowStatus",
    "GaugeLoopTerm",
    "ImprovedGaugeAction",
    "MatrixGaugeLoopAction",
    "ImprovedGaugeFamily",
    "gauge_flow_observables",
    "standard_gauge_loop_action",
]
