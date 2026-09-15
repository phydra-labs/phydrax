#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...graph._matrix_gauge import gauge_transform_links, MatrixGaugeLinkSpace
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ...metrix._complex_matrix_manifold import SpecialUnitaryGroup, UnitaryGroup


GaugeFixingCondition: TypeAlias = Literal["landau", "coulomb"]


def _links(space: MatrixGaugeLinkSpace, links: ArrayLike, /) -> Array:
    values = jnp.asarray(links)
    if values.shape != space.configuration_shape:
        raise ValueError(
            f"links must have shape {space.configuration_shape}; got {values.shape}."
        )
    return values


def _functional(
    space: MatrixGaugeLinkSpace,
    edge_mask: Array,
    links: Array,
    /,
) -> Array:
    traces = jnp.real(jnp.trace(links, axis1=-2, axis2=-1))
    count = jnp.sum(edge_mask.astype(traces.dtype))
    return jnp.sum(jnp.where(edge_mask, traces, 0.0)) / (count * space.point_shape[0])


class GaugeFixingPlan(StrictModule, NonTrainableState):
    """Host plan for explicit Landau or selected-spatial-edge Coulomb fixing."""

    link_space: MatrixGaugeLinkSpace
    edge_mask: Array
    condition: GaugeFixingCondition = eqx.field(static=True)
    anchor_vertex: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    maximum_gribov_copies: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    gribov_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        /,
        *,
        condition: GaugeFixingCondition,
        spatial_edges: ArrayLike | None = None,
        anchor_vertex: int = 0,
        maximum_iterations: int = 256,
        maximum_backtracks: int = 8,
        maximum_gribov_copies: int = 16,
        step_size: float = 0.2,
        residual_tolerance: float = 1.0e-8,
        gribov_tolerance: float = 1.0e-8,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(link_space.group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Gauge fixing requires U(N) or SU(N) matrix links.")
        if condition not in ("landau", "coulomb"):
            raise ValueError("condition must be 'landau' or 'coulomb'.")
        active_edges = np.asarray(link_space.topology.entities(1).active_mask, dtype=bool)
        if condition == "landau":
            if spatial_edges is not None:
                raise ValueError("Landau fixing selects all active edges implicitly.")
            selected = active_edges
        else:
            if spatial_edges is None:
                raise ValueError("Coulomb fixing requires an explicit spatial edge mask.")
            spatial = np.asarray(spatial_edges, dtype=bool)
            if spatial.shape != (link_space.num_edges,):
                raise ValueError("spatial_edges must contain one flag per gauge link.")
            if np.any(spatial & ~active_edges):
                raise ValueError("Coulomb spatial_edges cannot select inactive links.")
            selected = spatial
        if not np.any(selected):
            raise ValueError("Gauge fixing requires at least one selected active edge.")
        anchor = int(anchor_vertex)
        iterations = int(maximum_iterations)
        backtracks = int(maximum_backtracks)
        copies = int(maximum_gribov_copies)
        step = float(step_size)
        residual = float(residual_tolerance)
        gribov = float(gribov_tolerance)
        if anchor < 0 or anchor >= link_space.num_vertices:
            raise ValueError("anchor_vertex lies outside the gauge-link vertices.")
        if iterations < 1 or iterations > 1_000_000:
            raise ValueError("maximum_iterations must lie in [1, 1000000].")
        if backtracks < 0 or backtracks > 32:
            raise ValueError("maximum_backtracks must lie in [0, 32].")
        if copies < 1 or copies > 256:
            raise ValueError("maximum_gribov_copies must lie in [1, 256].")
        if not isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not isfinite(residual) or residual <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        if not isfinite(gribov) or gribov < 0.0:
            raise ValueError("gribov_tolerance must be finite and non-negative.")
        self.link_space = link_space
        self.edge_mask = jnp.asarray(selected)
        self.condition = condition
        self.anchor_vertex = anchor
        self.maximum_iterations = iterations
        self.maximum_backtracks = backtracks
        self.maximum_gribov_copies = copies
        self.step_size = step
        self.residual_tolerance = residual
        self.gribov_tolerance = gribov
        self.plan_id = canonical_fingerprint(
            {
                "kind": "explicit-lattice-gauge-fixing-plan",
                "link_space": link_space.link_space_id,
                "condition": condition,
                "selected_edges": array_tree_fingerprint(selected),
                "anchor_vertex": anchor,
                "maximum_iterations": iterations,
                "maximum_backtracks": backtracks,
                "maximum_gribov_copies": copies,
                "step_size": step,
                "residual_tolerance": residual,
                "gribov_tolerance": gribov,
                "functional": "mean-real-normalized-link-trace",
            }
        )

    def prepare(
        self, initial_transformations: ArrayLike | None = None, /
    ) -> PreparedGaugeFixing:
        """Bind explicit Gribov starts; omitted starts mean one identity copy."""
        return PreparedGaugeFixing(self, initial_transformations)


class LandauGaugeFixingPlan(StrictModule, NonTrainableState):
    """Typed Landau-gauge facade over the common explicit fixing plan."""

    plan: GaugeFixingPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, link_space: MatrixGaugeLinkSpace, /, **kwargs):
        plan = GaugeFixingPlan(link_space, condition="landau", **kwargs)
        self.plan = plan
        self.plan_id = plan.plan_id

    @property
    def link_space(self) -> MatrixGaugeLinkSpace:
        return self.plan.link_space

    @property
    def anchor_vertex(self) -> int:
        return self.plan.anchor_vertex

    def prepare(
        self, initial_transformations: ArrayLike | None = None, /
    ) -> PreparedGaugeFixing:
        return self.plan.prepare(initial_transformations)


class CoulombGaugeFixingPlan(StrictModule, NonTrainableState):
    """Typed Coulomb-gauge facade with an explicit spatial-link selection."""

    plan: GaugeFixingPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        spatial_edges: ArrayLike,
        /,
        **kwargs,
    ):
        plan = GaugeFixingPlan(
            link_space,
            condition="coulomb",
            spatial_edges=spatial_edges,
            **kwargs,
        )
        self.plan = plan
        self.plan_id = plan.plan_id

    @property
    def link_space(self) -> MatrixGaugeLinkSpace:
        return self.plan.link_space

    @property
    def anchor_vertex(self) -> int:
        return self.plan.anchor_vertex

    def prepare(
        self, initial_transformations: ArrayLike | None = None, /
    ) -> PreparedGaugeFixing:
        return self.plan.prepare(initial_transformations)


class GaugeFixingStatus(IntEnum):
    SUCCESS = 0
    INVALID_LINKS = 1
    NONFINITE_ITERATE = 2
    GROUP_MEMBERSHIP_LOST = 3
    NOT_CONVERGED = 4


class GribovCopyEvidence(StrictModule, NonTrainableState):
    functionals: Array
    residuals: Array
    iterations: Array
    converged: Array
    finite: Array
    membership: Array
    selected_copy: Array
    functional_spread: Array
    ambiguous: Array
    prepared_id: str = eqx.field(static=True)


class GaugeTransformationEvidence(StrictModule, NonTrainableState):
    initial_functional: Array
    final_functional: Array
    functional_gain: Array
    residual: Array
    transformation_membership: Array
    reconstruction_residual: Array
    condition: GaugeFixingCondition = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class GaugeFixingEvidence(StrictModule, NonTrainableState):
    status: Array
    finite: Array
    membership: Array
    converged: Array
    iterations: Array
    transformation: GaugeTransformationEvidence
    gribov: GribovCopyEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(GaugeFixingStatus.SUCCESS)


class GaugeFixingResult(StrictModule):
    links: Array
    transformation: Array
    faddeev_popov: FaddeevPopovOperator
    evidence: GaugeFixingEvidence
    prepared_id: str = eqx.field(static=True)


class PreparedGaugeFixing(StrictModule, NonTrainableState):
    """Prepared fixed-shape multistart runtime for one gauge condition."""

    plan: GaugeFixingPlan
    initial_transformations: Array
    copy_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GaugeFixingPlan,
        initial_transformations: ArrayLike | None,
        /,
    ):
        if not isinstance(plan, GaugeFixingPlan):
            raise TypeError("plan must be GaugeFixingPlan.")
        identity = plan.link_space.group.identity()
        starts = (
            np.broadcast_to(
                np.asarray(identity),
                (1, plan.link_space.num_vertices) + plan.link_space.point_shape,
            ).copy()
            if initial_transformations is None
            else np.asarray(initial_transformations)
        )
        expected_tail = (plan.link_space.num_vertices,) + plan.link_space.point_shape
        if starts.ndim != 4 or starts.shape[1:] != expected_tail:
            raise ValueError(
                "initial_transformations must have shape "
                f"(copy_count, {expected_tail[0]}, {expected_tail[1]}, {expected_tail[2]})."
            )
        copy_count = int(starts.shape[0])
        if copy_count < 1 or copy_count > plan.maximum_gribov_copies:
            raise ValueError("Gribov copy count exceeds maximum_gribov_copies.")
        if not bool(np.asarray(plan.link_space.group.contains(jnp.asarray(starts)))):
            raise ValueError(
                "Every initial transformation must belong to the gauge group."
            )
        self.plan = plan
        self.initial_transformations = jnp.asarray(starts)
        self.copy_count = copy_count
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-explicit-gauge-fixing",
                "plan": plan.plan_id,
                "initial_transformations": array_tree_fingerprint(starts),
                "copy_count": copy_count,
            }
        )

    def _gradient_at_coordinates(
        self,
        links: Array,
        coordinates: Array,
        /,
    ) -> Array:
        group = self.plan.link_space.group

        def transformed_functional(local_coordinates: Array) -> Array:
            transformations = group.exp(group.hat(local_coordinates))
            transformed = gauge_transform_links(
                self.plan.link_space, links, transformations
            )
            return _functional(self.plan.link_space, self.plan.edge_mask, transformed)

        gradient = jax.grad(transformed_functional)(coordinates)
        return gradient.at[self.plan.anchor_vertex].set(0.0)

    def _gradient(self, links: Array, /) -> Array:
        coordinates = jnp.zeros(
            (self.plan.link_space.num_vertices,)
            + self.plan.link_space.group.algebra_shape,
            dtype=jnp.real(links).dtype,
        )
        return self._gradient_at_coordinates(links, coordinates)

    def residual(self, links: ArrayLike, /) -> Array:
        """Return the RMS gauge-condition gradient with the anchor mode removed."""
        values = _links(self.plan.link_space, links)
        gradient = self._gradient(values)
        degrees = max(
            1,
            (self.plan.link_space.num_vertices - 1)
            * self.plan.link_space.group.algebra_shape[0],
        )
        return jnp.linalg.norm(gradient) / jnp.sqrt(
            jnp.asarray(degrees, dtype=gradient.dtype)
        )

    def _fix_one(
        self,
        original: Array,
        initial_transformation: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        space = self.plan.link_space
        group = space.group
        initial_links = gauge_transform_links(space, original, initial_transformation)
        initial_functional = _functional(space, self.plan.edge_mask, initial_links)
        initial_residual = self.residual(initial_links)
        dtype = jnp.real(original).dtype

        def iteration(
            carry: tuple[Array, Array, Array, Array, Array], _: None
        ) -> tuple[tuple[Array, Array, Array, Array, Array], tuple[Array, Array]]:
            links, transformation, functional, finite, membership = carry
            gradient = self._gradient(links)
            residual = self.residual(links)
            already_converged = residual <= self.plan.residual_tolerance

            def attempt(
                selection: tuple[Array, Array, Array, Array, Array], index: Array
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                chosen_links, chosen_transform, chosen_functional, chosen, good = (
                    selection
                )
                step = jnp.asarray(self.plan.step_size, dtype=dtype) * (
                    jnp.asarray(0.5, dtype=dtype) ** index
                )
                increment = group.exp(group.hat(step * gradient))
                candidate_links = gauge_transform_links(space, links, increment)
                candidate_transform = group.compose(increment, transformation)
                candidate_functional = _functional(
                    space, self.plan.edge_mask, candidate_links
                )
                candidate_finite = (
                    jnp.all(jnp.isfinite(candidate_links))
                    & jnp.all(jnp.isfinite(candidate_transform))
                    & jnp.isfinite(candidate_functional)
                )
                candidate_member = space.contains(candidate_links) & group.contains(
                    candidate_transform
                )
                nondecreasing = candidate_functional >= functional
                choose = (
                    (~chosen)
                    & (~already_converged)
                    & candidate_finite
                    & candidate_member
                    & nondecreasing
                )
                return (
                    jnp.where(choose, candidate_links, chosen_links),
                    jnp.where(choose, candidate_transform, chosen_transform),
                    jnp.where(choose, candidate_functional, chosen_functional),
                    chosen | choose | already_converged,
                    good | (choose & candidate_finite & candidate_member),
                ), None

            selection, _ = jax.lax.scan(
                attempt,
                (links, transformation, functional, already_converged, already_converged),
                jnp.arange(self.plan.maximum_backtracks + 1),
            )
            next_links, next_transform, next_functional, _, accepted = selection
            next_finite = (
                finite & jnp.all(jnp.isfinite(next_links)) & jnp.isfinite(next_functional)
            )
            next_member = (
                membership & space.contains(next_links) & group.contains(next_transform)
            )
            return (
                next_links,
                next_transform,
                next_functional,
                next_finite,
                next_member,
            ), (self.residual(next_links), accepted)

        final, trace = jax.lax.scan(
            iteration,
            (
                initial_links,
                initial_transformation,
                initial_functional,
                jnp.all(jnp.isfinite(initial_links)),
                space.contains(initial_links),
            ),
            xs=None,
            length=self.plan.maximum_iterations,
        )
        links, transformation, functional, finite, membership = final
        residuals, _ = trace
        convergence_trace = residuals <= self.plan.residual_tolerance
        first_converged = jnp.argmax(convergence_trace).astype(jnp.int32) + 1
        iteration_count = jnp.where(
            initial_residual <= self.plan.residual_tolerance,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                jnp.any(convergence_trace),
                first_converged,
                jnp.asarray(self.plan.maximum_iterations, dtype=jnp.int32),
            ),
        )
        return (
            links,
            transformation,
            functional,
            residuals[-1],
            iteration_count,
            finite,
            membership,
        )

    def fix(self, links: ArrayLike, /) -> GaugeFixingResult:
        original = _links(self.plan.link_space, links)
        outputs = jax.vmap(lambda start: self._fix_one(original, start))(
            self.initial_transformations
        )
        (
            fixed_links,
            transformations,
            functionals,
            residuals,
            iterations,
            finite,
            membership,
        ) = outputs
        converged = residuals <= self.plan.residual_tolerance
        admissible = finite & membership
        preferred = admissible & converged
        use_preferred = jnp.any(preferred)
        selectable = jnp.where(use_preferred, preferred, admissible)
        scores = jnp.where(selectable, functionals, -jnp.inf)
        selected = jnp.argmax(scores).astype(jnp.int32)
        result_links = fixed_links[selected]
        result_transformation = transformations[selected]
        selected_finite = finite[selected]
        selected_membership = membership[selected]
        selected_converged = converged[selected]
        selected_functional = functionals[selected]
        initial_functional = _functional(
            self.plan.link_space, self.plan.edge_mask, original
        )
        reconstructed = gauge_transform_links(
            self.plan.link_space, original, result_transformation
        )
        reconstruction_residual = jnp.max(jnp.abs(reconstructed - result_links))
        preferred_count = jnp.sum(preferred.astype(jnp.int32))
        maximum = jnp.max(jnp.where(preferred, functionals, -jnp.inf))
        minimum = jnp.min(jnp.where(preferred, functionals, jnp.inf))
        spread = jnp.where(preferred_count > 1, maximum - minimum, 0.0)
        ambiguous = (preferred_count > 1) & (spread > self.plan.gribov_tolerance)
        gribov = GribovCopyEvidence(
            functionals,
            residuals,
            iterations,
            converged,
            finite,
            membership,
            selected,
            spread,
            ambiguous,
            self.prepared_id,
        )
        transformation_evidence = GaugeTransformationEvidence(
            initial_functional,
            selected_functional,
            selected_functional - initial_functional,
            residuals[selected],
            self.plan.link_space.group.contains(result_transformation),
            reconstruction_residual,
            self.plan.condition,
            self.plan.plan_id,
        )
        input_membership = self.plan.link_space.contains(original)
        status = jnp.where(
            ~input_membership,
            int(GaugeFixingStatus.INVALID_LINKS),
            jnp.where(
                ~selected_finite,
                int(GaugeFixingStatus.NONFINITE_ITERATE),
                jnp.where(
                    ~selected_membership,
                    int(GaugeFixingStatus.GROUP_MEMBERSHIP_LOST),
                    jnp.where(
                        ~selected_converged,
                        int(GaugeFixingStatus.NOT_CONVERGED),
                        int(GaugeFixingStatus.SUCCESS),
                    ),
                ),
            ),
        )
        evidence = GaugeFixingEvidence(
            status,
            selected_finite,
            selected_membership,
            selected_converged,
            iterations[selected],
            transformation_evidence,
            gribov,
            self.plan.plan_id,
        )
        return GaugeFixingResult(
            result_links,
            result_transformation,
            FaddeevPopovOperator(self, result_links),
            evidence,
            self.prepared_id,
        )


class FaddeevPopovOperator(AbstractLinearOperator):
    """Matrix-free negative Hessian of the selected lattice gauge functional."""

    fixing: PreparedGaugeFixing
    links: Array

    def __init__(self, fixing: PreparedGaugeFixing, links: ArrayLike, /):
        if not isinstance(fixing, PreparedGaugeFixing):
            raise TypeError("fixing must be PreparedGaugeFixing.")
        values = _links(fixing.plan.link_space, links)
        shape = (
            fixing.plan.link_space.num_vertices,
            fixing.plan.link_space.group.algebra_shape[0],
        )
        space = ArraySpace(
            shape,
            dtype=jnp.real(values).dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "gauge-parameter-coordinate-space",
                    "fixing": fixing.prepared_id,
                    "shape": list(shape),
                    "anchor_vertex": fixing.plan.anchor_vertex,
                }
            ),
        )
        self.fixing = fixing
        self.links = values
        self.source = space
        self.target = space
        self.properties = OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "matrix-free-faddeev-popov-action",
                "fixing": fixing.prepared_id,
                "link_shape": list(values.shape),
                "condition": fixing.plan.condition,
            }
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        direction = self.source.validate(vector)
        direction = direction.at[self.fixing.plan.anchor_vertex].set(0.0)
        zero = jnp.zeros_like(direction)
        _, hessian_action = jax.jvp(
            lambda coordinates: self.fixing._gradient_at_coordinates(
                self.links, coordinates
            ),
            (zero,),
            (direction,),
        )
        return (-hessian_action).at[self.fixing.plan.anchor_vertex].set(0.0)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return self.mv(vector)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError("Faddeev-Popov action is intentionally matrix-free.")


__all__ = [
    "CoulombGaugeFixingPlan",
    "FaddeevPopovOperator",
    "GaugeFixingCondition",
    "GaugeFixingEvidence",
    "GaugeFixingPlan",
    "GaugeFixingResult",
    "GaugeFixingStatus",
    "GaugeTransformationEvidence",
    "GribovCopyEvidence",
    "LandauGaugeFixingPlan",
    "PreparedGaugeFixing",
]
