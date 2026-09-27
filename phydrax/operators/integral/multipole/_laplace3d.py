#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared spherical-harmonic translations for the three-dimensional Laplace kernel."""

from __future__ import annotations

import math
from operator import index
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spatial import (
    AdaptiveOctree,
    AdaptiveOctreePlan,
    MortonAddressPlan,
)
from ....discretization.spatial._plane_interactions import (
    MortonPlaneInteractionPlan,
    MortonPlaneInteractionState,
)
from ....discretization.spatial._plane_schedule import (
    MortonPlaneSchedulePlan,
    MortonPlaneScheduleState,
)
from ....discretization.spectral._spherical_layout import SphericalModeLayout
from ....sparse import RelationExecutionPlan
from ....special._solid_harmonic import (
    solid_harmonic_irregular,
    solid_harmonic_regular,
)
from ....typing import parse


TranslationRoute3D = Literal["dense"]
LaplaceExecution3D = Literal["level_octree", "plane_dual"]

# Targets per batch of the per-target W/U gathers, bounding their working set.
_TARGET_BATCH_SIZE = 256


def _translation_quadrature(bandlimit: int) -> tuple[np.ndarray, np.ndarray]:
    """Product Gauss--Legendre/Fourier rule exact on degree ``2L-2`` products."""
    polar_nodes, polar_weights = np.polynomial.legendre.leggauss(bandlimit)
    azimuth_count = 2 * bandlimit - 1
    azimuth = 2.0 * np.pi * np.arange(azimuth_count) / azimuth_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - polar_nodes * polar_nodes))
    directions = np.stack(
        (
            np.repeat(sine, azimuth_count) * np.tile(np.cos(azimuth), bandlimit),
            np.repeat(sine, azimuth_count) * np.tile(np.sin(azimuth), bandlimit),
            np.repeat(polar_nodes, azimuth_count),
        ),
        axis=-1,
    )
    weights = np.repeat(polar_weights, azimuth_count) * (2.0 * np.pi / azimuth_count)
    return directions, weights


def _mode_basis(
    layout: SphericalModeLayout,
    vectors: Array,
    /,
    *,
    radial: Literal["regular", "irregular"],
) -> Array:
    """Return coefficient-leading solid harmonics with padded layout storage."""
    values = jnp.asarray(vectors)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Spherical basis vectors must end in dimension three.")
    function = solid_harmonic_regular if radial == "regular" else solid_harmonic_irregular
    output = jnp.zeros(
        layout.coefficient_shape + values.shape[:-1],
        dtype=jnp.result_type(values.dtype, 1j),
    )
    offset = layout.bandlimit - 1
    for degree in range(layout.bandlimit):
        for order in range(-degree, degree + 1):
            output = output.at[degree, offset + order].set(
                function(degree, order, values)
            )
    return output


def _flatten_payload(values: Array, leading_axes: int) -> tuple[Array, tuple[int, ...]]:
    payload_shape = tuple(values.shape[leading_axes:])
    payload_size = math.prod(payload_shape) if payload_shape else 1
    return values.reshape(values.shape[:leading_axes] + (payload_size,)), payload_shape


class MultipoleResourceEvidence3D(StrictModule, NonTrainableState):
    """Static allocation and translation-workspace evidence."""

    logical_mode_count: int = eqx.field(static=True)
    padded_mode_count: int = eqx.field(static=True)
    coefficient_bytes_per_expansion: int = eqx.field(static=True)
    required_coefficient_bytes: int = eqx.field(static=True)
    maximum_coefficient_bytes: int = eqx.field(static=True)
    quadrature_node_count: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    far_interaction_capacity: int = eqx.field(static=True)
    near_interaction_capacity: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MultipoleTruncationEvidence3D(StrictModule, NonTrainableState):
    """A posteriori geometric evidence for the accepted far routes."""

    geometric_tail_bound: Array
    maximum_separation_ratio: Array
    well_separated: Array
    expansion_order: int = eqx.field(static=True)


class MultipoleCapacityEvidence3D(StrictModule, NonTrainableState):
    """Occupied-tree and route-capacity completion evidence."""

    required_nodes: Array
    node_capacity: Array
    required_far_interactions: Array
    far_interaction_capacity: Array
    required_near_interactions: Array
    near_interaction_capacity: Array
    successful: Array


class MultipoleFarLocal3D(StrictModule, NonTrainableState):
    """Far-only local expansions centered at every target center."""

    coefficients: Array
    truncation: MultipoleTruncationEvidence3D
    capacity: MultipoleCapacityEvidence3D
    m2m_count: Array
    m2l_count: Array
    p2l_count: Array
    l2l_count: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class AbstractLaplaceMultipoleEvaluation3D(StrictModule, NonTrainableState):
    """Complete FMM values with exact near completion and bounded far evidence."""

    values: Array
    far_values: Array
    near_values: Array
    truncation: MultipoleTruncationEvidence3D
    capacity: MultipoleCapacityEvidence3D
    maximum_reference_displacement: Array
    stale_topology: Array
    finite: Array
    successful: Array
    p2m_count: Array
    m2m_count: Array
    m2l_count: Array
    p2l_count: Array
    l2l_count: Array
    l2p_count: Array
    m2p_count: Array
    p2p_count: Array
    expansion_order: int = eqx.field(static=True)
    source_convention: str = eqx.field(static=True)
    local_convention: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


class LaplaceMultipoleEvaluation3D(AbstractLaplaceMultipoleEvaluation3D):
    """Final complete Laplace FMM evaluation."""


class LaplaceMultipolePlan3D(StrictModule, NonTrainableState):
    """Fixed-capacity Laplace FMM policy over frozen reference topology.

    ``expansion_order=p`` stores degrees ``0 <= ell <= p`` in a complex,
    spin-zero :class:`SphericalModeLayout`.  Omitting ``reference_targets``
    selects same-support evaluation and exact diagonal exclusion.

    ``execution="level_octree"`` prepares an adaptive octree over the
    reference sources that subdivides cells holding more than
    ``source_leaf_occupancy`` sources down to ``depth``; targets are located in
    its leaves at evaluation. Its U/V/W/X interaction lists keep one cell of
    clearance for sources displaced by up to ``maximum_reference_displacement``.
    ``far_interaction_capacity`` then bounds each of the V, W, and X lists and
    ``near_interaction_capacity`` bounds the U list; ``None`` stores them
    completely.
    """

    reference_sources: Array
    reference_targets: Array
    lower: tuple[float, float, float] = eqx.field(static=True)
    upper: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    expansion_order: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    target_topology: Literal["same-support", "rectangular"] = eqx.field(static=True)
    maximum_reference_displacement: float = eqx.field(static=True)
    far_interaction_capacity: int | None = eqx.field(static=True)
    near_interaction_capacity: int | None = eqx.field(static=True)
    maximum_coefficient_bytes: int = eqx.field(static=True)
    translation_route: TranslationRoute3D = eqx.field(static=True)
    execution: LaplaceExecution3D = eqx.field(static=True)
    plane_queue_capacity: int | None = eqx.field(static=True)
    source_leaf_occupancy: int = eqx.field(static=True)
    target_leaf_occupancy: int = eqx.field(static=True)
    plane_coarsening_factor: int = eqx.field(static=True)
    plane_target_top_nodes: int = eqx.field(static=True)
    plane_opening_angle: float = eqx.field(static=True)
    plane_maximum_node_radius: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_sources: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        reference_targets: ArrayLike | None = None,
        depth: int = 3,
        expansion_order: int = 4,
        maximum_reference_displacement: float = 0.0,
        far_interaction_capacity: int | None = None,
        near_interaction_capacity: int | None = None,
        maximum_coefficient_bytes: int = 512 * 1024**2,
        translation_route: TranslationRoute3D = "dense",
        execution: LaplaceExecution3D = "level_octree",
        plane_queue_capacity: int | None = None,
        source_leaf_occupancy: int = 16,
        target_leaf_occupancy: int = 16,
        plane_coarsening_factor: int = 8,
        plane_target_top_nodes: int = 32,
        plane_opening_angle: float = 0.6,
        plane_maximum_node_radius: float | None = None,
    ) -> None:
        sources = np.asarray(reference_sources, dtype=np.float64)
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        same_support = reference_targets is None
        targets = (
            sources if same_support else np.asarray(reference_targets, dtype=np.float64)
        )
        depth_ = index(depth)
        order = index(expansion_order)
        displacement = float(maximum_reference_displacement)
        maximum_bytes = index(maximum_coefficient_bytes)
        if (
            sources.ndim != 2
            or sources.shape[0] == 0
            or sources.shape[1] != 3
            or targets.ndim != 2
            or targets.shape[0] == 0
            or targets.shape[1] != 3
        ):
            raise ValueError("Reference sources and targets must have shape (count, 3).")
        if (
            lower_.shape != (3,)
            or upper_.shape != (3,)
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(upper_ <= lower_)
            or np.any(~np.isfinite(sources))
            or np.any(~np.isfinite(targets))
            or np.any(sources < lower_)
            or np.any(sources >= upper_)
            or np.any(targets < lower_)
            or np.any(targets >= upper_)
        ):
            raise ValueError(
                "Multipole references must be finite and lie in [lower, upper)."
            )
        if depth_ < 2 or order < 0:
            raise ValueError(
                "depth must be at least two and expansion_order nonnegative."
            )
        if not math.isfinite(displacement) or displacement < 0.0:
            raise ValueError(
                "maximum_reference_displacement must be finite and nonnegative."
            )
        if maximum_bytes <= 0:
            raise ValueError("maximum_coefficient_bytes must be positive.")
        if translation_route != "dense":
            raise ValueError(
                "Only the complete dense Laplace translation route is available."
            )
        execution = parse(execution, LaplaceExecution3D, "execution")
        source_leaf = index(source_leaf_occupancy)
        target_leaf = index(target_leaf_occupancy)
        coarse = index(plane_coarsening_factor)
        top_nodes = index(plane_target_top_nodes)
        if source_leaf <= 0 or target_leaf <= 0 or coarse < 2 or top_nodes <= 0:
            raise ValueError("Plane execution geometry is invalid.")
        if same_support and source_leaf != target_leaf:
            raise ValueError(
                "same-support plane execution requires matching leaf occupancies."
            )
        opening = float(plane_opening_angle)
        maximum_node_radius = (
            None
            if plane_maximum_node_radius is None
            else float(plane_maximum_node_radius)
        )
        if not 0.0 < opening < 1.0:
            raise ValueError("plane_opening_angle must lie strictly in (0, 1).")
        if maximum_node_radius is not None and (
            not math.isfinite(maximum_node_radius) or maximum_node_radius <= 0.0
        ):
            raise ValueError("plane_maximum_node_radius must be finite and positive.")
        if execution == "plane_dual":
            source_leaf_count = math.ceil(sources.shape[0] / source_leaf)
            target_leaf_count = math.ceil(targets.shape[0] / target_leaf)
            pair_default = max(source_leaf_count * target_leaf_count, 1)
        else:
            pair_default = None
        far = (
            pair_default
            if far_interaction_capacity is None
            else index(far_interaction_capacity)
        )
        near = (
            pair_default
            if near_interaction_capacity is None
            else index(near_interaction_capacity)
        )
        queue = (
            pair_default if plane_queue_capacity is None else index(plane_queue_capacity)
        )
        if any(value is not None and value <= 0 for value in (far, near, queue)):
            raise ValueError("Far, near, and queue capacities must be positive.")
        self.reference_sources = jnp.asarray(sources)
        self.reference_targets = jnp.asarray(targets)
        # ty: ignore[invalid-assignment]
        self.lower = tuple(float(value) for value in lower_)
        # ty: ignore[invalid-assignment]
        self.upper = tuple(float(value) for value in upper_)
        self.depth = depth_
        self.expansion_order = order
        self.source_capacity = sources.shape[0]
        self.target_capacity = targets.shape[0]
        self.target_topology = "same-support" if same_support else "rectangular"
        self.maximum_reference_displacement = displacement
        self.far_interaction_capacity = far
        self.near_interaction_capacity = near
        self.maximum_coefficient_bytes = maximum_bytes
        self.translation_route = translation_route
        self.execution = execution
        self.plane_queue_capacity = queue
        self.source_leaf_occupancy = source_leaf
        self.target_leaf_occupancy = target_leaf
        self.plane_coarsening_factor = coarse
        self.plane_opening_angle = opening
        self.plane_maximum_node_radius = maximum_node_radius
        self.plane_target_top_nodes = top_nodes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "laplace-multipole-plan-3d",
                "reference_sources": array_tree_fingerprint(sources),
                "reference_targets": array_tree_fingerprint(targets),
                "lower": list(self.lower),
                "upper": list(self.upper),
                "depth": depth_,
                "expansion_order": order,
                "maximum_reference_displacement": displacement,
                "far_interaction_capacity": far,
                "near_interaction_capacity": near,
                "maximum_coefficient_bytes": maximum_bytes,
                "target_topology": self.target_topology,
                "translation_route": translation_route,
                "execution": execution,
                "plane_queue_capacity": queue,
                "source_leaf_occupancy": source_leaf,
                "target_leaf_occupancy": target_leaf,
                "plane_coarsening_factor": coarse,
                "plane_opening_angle": opening,
                "plane_maximum_node_radius": maximum_node_radius,
                "plane_target_top_nodes": top_nodes,
            }
        )

    def prepare(self, /) -> PreparedLaplaceMultipole3D:
        """Materialize topology, projection quadrature, and bounded workspaces."""
        # ty: ignore[missing-argument]
        return PreparedLaplaceMultipole3D(self)


class AbstractPreparedLaplaceMultipole3D(StrictModule, NonTrainableState):
    """Prepared complete Laplace FMM with level-octree or plane execution."""

    plan: LaplaceMultipolePlan3D
    layout: SphericalModeLayout
    topology: AdaptiveOctree | None
    quadrature_directions: Array
    quadrature_weights: Array
    quadrature_harmonics: Array
    equivalent_inverse: Array
    resources: MultipoleResourceEvidence3D
    source_plane_plan: MortonPlaneSchedulePlan | None = eqx.field(static=True)
    target_plane_plan: MortonPlaneSchedulePlan | None = eqx.field(static=True)
    source_plane: MortonPlaneScheduleState | None
    target_plane: MortonPlaneScheduleState | None
    plane_interactions: MortonPlaneInteractionState | None
    source_convention: str = eqx.field(static=True)
    local_convention: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: LaplaceMultipolePlan3D, /) -> None:
        if not isinstance(plan, LaplaceMultipolePlan3D):
            raise TypeError("plan must be LaplaceMultipolePlan3D.")
        layout = SphericalModeLayout(plan.expansion_order + 1, spin=0, reality=False)
        directions_, weights_ = _translation_quadrature(layout.bandlimit)
        directions = jnp.asarray(directions_)
        weights = jnp.asarray(weights_)
        regular = _mode_basis(layout, directions, radial="regular")
        degree_scale = 1.0 / (2.0 * jnp.arange(layout.bandlimit) + 1.0)
        p2m_matrix = (jnp.conj(regular) * degree_scale[:, None, None]).reshape(
            (-1, directions.shape[0])
        )[layout.valid_indices]
        equivalent_inverse = jnp.linalg.pinv(p2m_matrix)

        topology = None
        source_plane_plan = None
        target_plane_plan = None
        source_plane = None
        target_plane = None
        plane_interactions = None
        if plan.execution == "level_octree":
            topology = AdaptiveOctreePlan(
                MortonAddressPlan(plan.lower, plan.upper, plan.depth),
                leaf_capacity=plan.source_leaf_occupancy,
                separation_padding=plan.maximum_reference_displacement,
                u_capacity=plan.near_interaction_capacity,
                v_capacity=plan.far_interaction_capacity,
                w_capacity=plan.far_interaction_capacity,
                x_capacity=plan.far_interaction_capacity,
            ).prepare(plan.reference_sources)
            if not bool(topology.evidence.successful):
                raise ValueError(
                    "Adaptive octree interaction capacity is exhausted by the reference sources."
                )
        else:
            address = MortonAddressPlan(plan.lower, plan.upper, plan.depth)
            source_plane_plan = MortonPlaneSchedulePlan(
                address,
                plan.source_capacity,
                maximum_leaf_occupancy=plan.source_leaf_occupancy,
                coarsening_factor=plan.plane_coarsening_factor,
                target_top_nodes=plan.plane_target_top_nodes,
            )
            source_plane = source_plane_plan.build(
                plan.reference_sources,
                bounding_padding=plan.maximum_reference_displacement,
                stable_ids=jnp.arange(plan.source_capacity, dtype=jnp.int64),
            )
            if plan.target_topology == "same-support":
                target_plane_plan = source_plane_plan
                target_plane = source_plane
            else:
                target_plane_plan = MortonPlaneSchedulePlan(
                    address,
                    plan.target_capacity,
                    maximum_leaf_occupancy=plan.target_leaf_occupancy,
                    coarsening_factor=plan.plane_coarsening_factor,
                    target_top_nodes=plan.plane_target_top_nodes,
                )
                target_plane = target_plane_plan.build(
                    plan.reference_targets,
                    bounding_padding=plan.maximum_reference_displacement,
                    stable_ids=jnp.arange(plan.target_capacity, dtype=jnp.int64),
                )
            plane_interactions = MortonPlaneInteractionPlan(
                source_plane_plan,
                target_plane_plan,
                opening_angle=plan.plane_opening_angle,
                # ty: ignore[invalid-argument-type]
                queue_capacity=plan.plane_queue_capacity,
                # ty: ignore[invalid-argument-type]
                far_capacity=plan.far_interaction_capacity,
                # ty: ignore[invalid-argument-type]
                near_capacity=plan.near_interaction_capacity,
                maximum_node_radius=plan.plane_maximum_node_radius,
            ).build(
                source_plane,
                target_plane,
                same_support=plan.target_topology == "same-support",
            )
            if not bool(
                source_plane.evidence.successful
                & target_plane.evidence.successful
                & plane_interactions.evidence.successful
            ):
                raise ValueError(
                    "Plane schedule or interaction capacity is exhausted by the reference topology."
                )

        padded = math.prod(layout.coefficient_shape)
        coefficient_bytes = padded * np.dtype(np.complex128).itemsize
        if topology is not None:
            node_capacity = topology.node_count
            required_bytes = 2 * node_capacity * coefficient_bytes
            far_capacity = (
                topology.v_list.routes.capacity
                + topology.w_list.routes.capacity
                + topology.x_list.routes.capacity
            )
            near_capacity = topology.u_list.routes.capacity
            topology_id = topology.tree_id
        else:
            if source_plane_plan is None or target_plane_plan is None:
                raise RuntimeError("Plane topology is not prepared.")
            node_capacity = (
                source_plane_plan.node_capacity + target_plane_plan.node_capacity
            )
            required_bytes = node_capacity * coefficient_bytes
            far_capacity = plan.far_interaction_capacity
            near_capacity = plan.near_interaction_capacity
            topology_id = canonical_fingerprint(
                {
                    "source_plane": source_plane_plan.plan_id,
                    "target_plane": target_plane_plan.plan_id,
                    "queue_capacity": plan.plane_queue_capacity,
                    "far_capacity": plan.far_interaction_capacity,
                    "near_capacity": plan.near_interaction_capacity,
                }
            )
        within_budget = required_bytes <= plan.maximum_coefficient_bytes
        if not within_budget:
            raise ValueError(
                "Multipole coefficient workspaces exceed maximum_coefficient_bytes."
            )
        evidence_id = canonical_fingerprint(
            {
                "kind": "multipole-resource-evidence-3d",
                "plan": plan.plan_id,
                "logical_mode_count": layout.logical_mode_count,
                "padded_mode_count": padded,
                "coefficient_bytes_per_expansion": coefficient_bytes,
                "required_coefficient_bytes": required_bytes,
                "maximum_coefficient_bytes": plan.maximum_coefficient_bytes,
                "quadrature_node_count": directions.shape[0],
                "node_capacity": node_capacity,
                "far_interaction_capacity": far_capacity,
                "near_interaction_capacity": near_capacity,
                "execution": plan.execution,
            }
        )
        self.plan = plan
        self.layout = layout
        self.topology = topology
        self.quadrature_directions = directions
        self.quadrature_weights = weights
        self.quadrature_harmonics = regular
        self.equivalent_inverse = equivalent_inverse
        self.resources = MultipoleResourceEvidence3D(
            logical_mode_count=layout.logical_mode_count,
            padded_mode_count=padded,
            coefficient_bytes_per_expansion=coefficient_bytes,
            required_coefficient_bytes=required_bytes,
            maximum_coefficient_bytes=plan.maximum_coefficient_bytes,
            quadrature_node_count=directions.shape[0],
            node_capacity=node_capacity,
            # ty: ignore[invalid-argument-type]
            far_interaction_capacity=far_capacity,
            # ty: ignore[invalid-argument-type]
            near_interaction_capacity=near_capacity,
            within_budget=within_budget,
            evidence_id=evidence_id,
        )
        self.source_plane_plan = source_plane_plan
        self.target_plane_plan = target_plane_plan
        self.source_plane = source_plane
        self.target_plane = target_plane
        self.plane_interactions = plane_interactions
        self.source_convention = (
            "M[l,m]=sum(q*r**l*conj(Y[l,m]))/(2*l+1); field=sum(M[l,m]*Y[l,m]/r**(l+1))"
        )
        self.local_convention = (
            "L[l,m]=sum(q*conj(Y[l,m])/(2*l+1)/r**(l+1)); field=sum(L[l,m]*r**l*Y[l,m])"
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-laplace-multipole-3d",
                "plan": plan.plan_id,
                "layout": layout.layout_id,
                "topology": topology_id,
                "resources": evidence_id,
            }
        )

    def _validate_coefficients(self, coefficients: ArrayLike, name: str, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.ndim < 2 or tuple(values.shape[:2]) != self.layout.coefficient_shape:
            raise ValueError(
                f"{name} must begin with coefficient shape {self.layout.coefficient_shape}."
            )
        return jnp.where(
            self.layout.valid_mask.reshape(
                self.layout.coefficient_shape + (1,) * (values.ndim - 2)
            ),
            values,
            jnp.zeros((), dtype=values.dtype),
        )

    def _mask_coefficients(self, values: Array, /) -> Array:
        mask = self.layout.valid_mask.reshape(
            self.layout.coefficient_shape + (1,) * (values.ndim - 2)
        )
        return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))

    def _p2m_basis(self, relative: Array, /) -> Array:
        regular = _mode_basis(self.layout, relative, radial="regular")
        scale = 1.0 / (2.0 * jnp.arange(self.layout.bandlimit) + 1.0)
        return jnp.conj(regular) * scale.reshape(
            (self.layout.bandlimit, 1) + (1,) * (regular.ndim - 2)
        )

    def _p2l_basis(self, relative: Array, /) -> Array:
        irregular = _mode_basis(self.layout, relative, radial="irregular")
        scale = 1.0 / (2.0 * jnp.arange(self.layout.bandlimit) + 1.0)
        return jnp.conj(irregular) * scale.reshape(
            (self.layout.bandlimit, 1) + (1,) * (irregular.ndim - 2)
        )

    def p2m(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        center: ArrayLike,
        /,
        *,
        source_normals: ArrayLike | None = None,
    ) -> Array:
        """Accumulate point charges or source-normal dipoles into one multipole."""
        positions = jnp.asarray(source_positions)
        strengths = jnp.asarray(source_strengths)
        center_ = jnp.asarray(center, dtype=positions.dtype)
        if positions.ndim != 2 or positions.shape[1] != 3 or center_.shape != (3,):
            raise ValueError("P2M positions must have shape (source_count, 3).")
        if strengths.ndim < 1 or strengths.shape[0] != positions.shape[0]:
            raise ValueError("P2M strengths must begin with source_count.")
        relative = positions - center_
        if source_normals is None:
            basis = self._p2m_basis(relative)
        else:
            normals = jnp.asarray(source_normals, dtype=positions.dtype)
            if normals.shape != positions.shape:
                raise ValueError("source_normals must match source_positions.")

            def dipole_basis(vector: Any, normal: Any) -> Any:
                derivative = jax.jacfwd(self._p2m_basis)(vector)
                return jnp.sum(derivative * normal, axis=-1)

            basis = jnp.moveaxis(jax.vmap(dipole_basis)(relative, normals), 0, -1)
        strength_flat, payload_shape = _flatten_payload(strengths, 1)
        modal = basis.reshape((-1, positions.shape[0])) @ strength_flat
        return self._mask_coefficients(
            modal.reshape(self.layout.coefficient_shape + payload_shape)
        )

    def p2l(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        center: ArrayLike,
        /,
        *,
        source_normals: ArrayLike | None = None,
    ) -> Array:
        """Accumulate point charges or source-normal dipoles into one local expansion."""
        positions = jnp.asarray(source_positions)
        strengths = jnp.asarray(source_strengths)
        center_ = jnp.asarray(center, dtype=positions.dtype)
        if positions.ndim != 2 or positions.shape[1] != 3 or center_.shape != (3,):
            raise ValueError("P2L positions must have shape (source_count, 3).")
        if strengths.ndim < 1 or strengths.shape[0] != positions.shape[0]:
            raise ValueError("P2L strengths must begin with source_count.")
        relative = positions - center_
        if source_normals is None:
            basis = self._p2l_basis(relative)
        else:
            normals = jnp.asarray(source_normals, dtype=positions.dtype)
            if normals.shape != positions.shape:
                raise ValueError("source_normals must match source_positions.")

            def dipole_basis(vector: Any, normal: Any) -> Any:
                derivative = jax.jacfwd(self._p2l_basis)(vector)
                return jnp.sum(derivative * normal, axis=-1)

            basis = jnp.moveaxis(jax.vmap(dipole_basis)(relative, normals), 0, -1)
        strength_flat, payload_shape = _flatten_payload(strengths, 1)
        modal = basis.reshape((-1, positions.shape[0])) @ strength_flat
        return self._mask_coefficients(
            modal.reshape(self.layout.coefficient_shape + payload_shape)
        )

    def _evaluate_basis(self, coefficients: Array, relative: Array, radial: str) -> Array:
        modal = self._validate_coefficients(coefficients, "coefficients")
        # ty: ignore[invalid-argument-type]
        basis = _mode_basis(self.layout, relative, radial=radial)
        modal_flat, payload_shape = _flatten_payload(modal, 2)
        point_shape = tuple(relative.shape[:-1])
        basis_flat = basis.reshape((-1, math.prod(point_shape) if point_shape else 1))
        result = basis_flat.T @ modal_flat.reshape((-1, modal_flat.shape[-1]))
        return result.reshape(point_shape + payload_shape)

    def l2p(self, local: ArrayLike, center: ArrayLike, targets: ArrayLike, /) -> Array:
        """Evaluate a local expansion at arbitrary target points."""
        target = jnp.asarray(targets)
        center_ = jnp.asarray(center, dtype=target.dtype)
        if target.ndim < 1 or target.shape[-1] != 3 or center_.shape != (3,):
            raise ValueError("L2P targets must end in dimension three.")
        return self._evaluate_basis(jnp.asarray(local), target - center_, "regular")

    def multipole_to_point(
        self, multipole: ArrayLike, center: ArrayLike, targets: ArrayLike, /
    ) -> Array:
        """Evaluate a multipole expansion outside its source ball."""
        target = jnp.asarray(targets)
        center_ = jnp.asarray(center, dtype=target.dtype)
        if target.ndim < 1 or target.shape[-1] != 3 or center_.shape != (3,):
            raise ValueError("Multipole targets must end in dimension three.")
        return self._evaluate_basis(jnp.asarray(multipole), target - center_, "irregular")

    def m2m(
        self,
        multipole: ArrayLike,
        child_center: ArrayLike,
        parent_center: ArrayLike,
        /,
    ) -> Array:
        """Translate source moments exactly through the retained degree."""
        modal = self._validate_coefficients(multipole, "multipole")
        child = jnp.asarray(child_center)
        parent = jnp.asarray(parent_center, dtype=child.dtype)
        if child.shape != (3,) or parent.shape != (3,):
            raise ValueError("M2M centers must have shape (3,).")
        modal_flat, payload_shape = _flatten_payload(modal, 2)
        active = modal_flat.reshape((-1, modal_flat.shape[-1]))[self.layout.valid_indices]
        charges = self.equivalent_inverse.astype(active.dtype) @ active
        translated = self.p2m(
            child[None, :] + self.quadrature_directions.astype(child.dtype),
            charges.reshape((self.quadrature_directions.shape[0],) + payload_shape),
            parent,
        )
        return translated

    def _project_local(self, function: Any, center: Array, /) -> Array:
        directions = self.quadrature_directions.astype(center.dtype)
        weights = self.quadrature_weights.astype(center.dtype)
        payload_probe = function(center)
        payload_shape = tuple(payload_probe.shape)
        output = jnp.zeros(
            self.layout.coefficient_shape + payload_shape,
            dtype=jnp.result_type(payload_probe.dtype, 1j),
        )
        offset = self.layout.bandlimit - 1
        for degree in range(self.layout.bandlimit):

            def along(direction: Any, degree: Any = degree) -> Any:
                def radial(distance: Any) -> Any:
                    return function(center + distance * direction)

                derivative = radial
                for _ in range(degree):
                    derivative = jax.jacfwd(derivative)
                return derivative(jnp.asarray(0.0, dtype=center.dtype)) / math.factorial(
                    degree
                )

            homogeneous = jax.vmap(along)(directions)
            for order in range(-degree, degree + 1):
                harmonic = self.quadrature_harmonics[degree, offset + order]
                projection = jnp.tensordot(
                    weights * jnp.conj(harmonic),
                    homogeneous,
                    axes=((0,), (0,)),
                )
                output = output.at[degree, offset + order].set(projection)
        return self._mask_coefficients(output)

    def m2l(
        self,
        multipole: ArrayLike,
        source_center: ArrayLike,
        target_center: ArrayLike,
        /,
    ) -> Array:
        """Translate a multipole to a local expansion by exact Taylor projection."""
        modal = self._validate_coefficients(multipole, "multipole")
        source = jnp.asarray(source_center)
        target = jnp.asarray(target_center, dtype=source.dtype)
        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("M2L centers must have shape (3,).")
        return self._project_local(
            lambda point: self.multipole_to_point(modal, source, point), target
        )

    def l2l(
        self,
        local: ArrayLike,
        parent_center: ArrayLike,
        child_center: ArrayLike,
        /,
    ) -> Array:
        """Translate a retained local polynomial exactly to a child center."""
        modal = self._validate_coefficients(local, "local")
        parent = jnp.asarray(parent_center)
        child = jnp.asarray(child_center, dtype=parent.dtype)
        if parent.shape != (3,) or child.shape != (3,):
            raise ValueError("L2L centers must have shape (3,).")
        return self._project_local(lambda point: self.l2p(modal, parent, point), child)

    def p2p(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike,
        /,
        *,
        source_normals: ArrayLike | None = None,
        target_source_indices: ArrayLike | None = None,
    ) -> Array:
        """Evaluate exact point interactions, optionally excluding identified self pairs."""
        sources = jnp.asarray(source_positions)
        strengths = jnp.asarray(source_strengths)
        targets = jnp.asarray(target_positions, dtype=sources.dtype)
        if sources.ndim != 2 or sources.shape[1] != 3:
            raise ValueError("P2P sources must have shape (source_count, 3).")
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("P2P targets must have shape (target_count, 3).")
        if strengths.ndim < 1 or strengths.shape[0] != sources.shape[0]:
            raise ValueError("P2P strengths must begin with source_count.")
        pair_mask = jnp.ones((targets.shape[0], sources.shape[0]), dtype=jnp.bool_)
        if target_source_indices is not None:
            identities = jnp.asarray(target_source_indices, dtype=jnp.int32)
            if identities.shape != (targets.shape[0],):
                raise ValueError("target_source_indices must match target_count.")
            pair_mask = pair_mask & (
                identities[:, None]
                != jnp.arange(sources.shape[0], dtype=jnp.int32)[None, :]
            )
        return self._p2p_masked(
            sources,
            strengths,
            targets,
            pair_mask,
            source_normals=source_normals,
        )[0]

    def _p2p_masked(
        self,
        sources: Array,
        strengths: Array,
        targets: Array,
        pair_mask: Array,
        /,
        *,
        source_normals: ArrayLike | None,
    ) -> tuple[Array, Array]:
        differences = targets[:, None, :] - sources[None, :, :]
        squared = jnp.sum(differences * differences, axis=-1)
        strengths = eqx.error_if(
            strengths,
            jnp.any(pair_mask & (squared == 0.0)),
            "A non-excluded Laplace point pair is singular.",
        )
        safe_squared = jnp.where(pair_mask, squared, 1.0)
        radii = jnp.sqrt(safe_squared)
        if source_normals is None:
            kernels = 1.0 / (4.0 * jnp.pi * radii)
        else:
            normals = jnp.asarray(source_normals, dtype=sources.dtype)
            if normals.shape != sources.shape:
                raise ValueError("source_normals must match source_positions.")
            kernels = jnp.sum(differences * normals[None, :, :], axis=-1) / (
                4.0 * jnp.pi * safe_squared * radii
            )
        kernels = jnp.where(pair_mask, kernels, 0.0)
        strength_flat, payload_shape = _flatten_payload(strengths, 1)
        values = kernels @ strength_flat
        return values.reshape((targets.shape[0],) + payload_shape), jnp.sum(
            pair_mask, dtype=jnp.int32
        )

    def _plane_source_moments(
        self,
        positions: Array,
        strengths: Array,
        active: Array,
        normals: Array | None,
    ) -> tuple[Array, Array]:
        """Build source moments and absolute-strength bounds on plane nodes."""
        if self.source_plane is None or self.source_plane_plan is None:
            raise RuntimeError("Plane source topology is not prepared.")
        schedule = self.source_plane
        node_count = self.source_plane_plan.node_capacity
        leaf_capacity = self.source_plane_plan.plane_capacities[0]
        width = self.source_plane_plan.maximum_leaf_occupancy
        storage_to_logical = schedule.point_order.storage_to_logical
        sorted_positions = positions[storage_to_logical]
        sorted_strengths = strengths[storage_to_logical]
        sorted_active = schedule.point_order.sorted_active & active[storage_to_logical]
        sorted_normals = None if normals is None else normals[storage_to_logical]
        offsets = jnp.arange(width, dtype=jnp.int32)
        starts = schedule.node_item_starts[:leaf_capacity, None] + offsets[None, :]
        counts = schedule.node_item_counts[:leaf_capacity, None]
        safe = jnp.clip(starts, 0, self.plan.source_capacity - 1)
        valid = offsets[None, :] < counts
        valid = valid & sorted_active[safe]
        leaf_positions = sorted_positions[safe]
        strength_mask = valid.reshape(valid.shape + (1,) * (sorted_strengths.ndim - 1))
        leaf_strengths = jnp.where(
            strength_mask,
            sorted_strengths[safe],
            0.0,
        )
        leaf_normals = None if sorted_normals is None else sorted_normals[safe]
        leaf_centers = schedule.node_centers[:leaf_capacity]
        if leaf_normals is None:
            leaf_moments = jax.vmap(self.p2m)(
                leaf_positions,
                leaf_strengths,
                leaf_centers,
            )
        else:
            leaf_moments = jax.vmap(self.p2m)(
                leaf_positions,
                leaf_strengths,
                leaf_centers,
                source_normals=leaf_normals,
            )
        moments = (
            jnp.zeros(
                (node_count,) + leaf_moments.shape[1:],
                dtype=leaf_moments.dtype,
            )
            .at[:leaf_capacity]
            .set(leaf_moments)
        )
        leaf_weight = jnp.sum(
            jnp.abs(leaf_strengths),
            axis=tuple(range(1, leaf_strengths.ndim)),
        )
        weights = (
            jnp.zeros((node_count,), dtype=leaf_weight.dtype)
            .at[:leaf_capacity]
            .set(leaf_weight)
        )
        offsets = jnp.arange(self.source_plane_plan.coarsening_factor)
        for plane in range(1, self.source_plane_plan.plane_count):
            at_plane = schedule.node_active & (schedule.node_planes == plane)
            children = schedule.node_child_starts[:, None] + offsets[None, :]
            valid_children = at_plane[:, None] & (
                offsets[None, :] < schedule.node_child_counts[:, None]
            )
            safe_children = jnp.clip(children, 0, node_count - 1)
            child_centers = schedule.node_centers[safe_children]
            parent_centers = schedule.node_centers[:, None, :]
            child_values = moments[safe_children]
            translated = jax.vmap(
                jax.vmap(lambda value, child, parent: self.m2m(value, child, parent))
            )(
                child_values,
                child_centers,
                jnp.broadcast_to(parent_centers, child_centers.shape),
            )
            translated = jnp.where(
                valid_children.reshape(
                    valid_children.shape + (1,) * (translated.ndim - 2)
                ),
                translated,
                0.0,
            )
            parent_values = jnp.sum(translated, axis=1)
            moments = jnp.where(
                at_plane.reshape((node_count,) + (1,) * (moments.ndim - 1)),
                parent_values,
                moments,
            )
            parent_weights = jnp.sum(
                jnp.where(valid_children, weights[safe_children], 0.0),
                axis=1,
            )
            weights = jnp.where(at_plane, parent_weights, weights)
        return moments, weights

    def _plane_far_locals(
        self,
        sources: Array,
        strengths: Array,
        targets: Array,
        active: Array,
        normals: Array | None,
    ) -> tuple[Array, MultipoleTruncationEvidence3D, Array, Array, Array]:
        if (
            self.source_plane is None
            or self.target_plane is None
            or self.plane_interactions is None
            or self.target_plane_plan is None
        ):
            raise RuntimeError("Plane topology is not prepared.")
        source_moments, source_weight = self._plane_source_moments(
            sources, strengths, active, normals
        )
        source_schedule = self.source_plane
        target_schedule = self.target_plane
        far = self.plane_interactions.far
        source_nodes = far.source_indices
        target_nodes = far.target_indices
        far_values = jax.vmap(self.m2l)(
            source_moments[source_nodes],
            source_schedule.node_centers[source_nodes],
            target_schedule.node_centers[target_nodes],
        )
        far_values = jnp.where(
            far.valid.reshape((far.valid.shape[0],) + (1,) * (far_values.ndim - 1)),
            far_values,
            0.0,
        )
        execution = RelationExecutionPlan(
            maximum_active_targets=self.target_plane_plan.node_capacity
        ).prepare(far)
        locals_, _reduction = execution.reduce(
            far_values,
            accumulation="deterministic",
        )
        target_node_count = self.target_plane_plan.node_capacity
        for plane in range(self.target_plane_plan.plane_count - 2, -1, -1):
            at_plane = target_schedule.node_active & (
                target_schedule.node_planes == plane
            )
            parents = jnp.maximum(target_schedule.node_parents, 0)
            inherited = jax.vmap(self.l2l)(
                locals_[parents],
                target_schedule.node_centers[parents],
                target_schedule.node_centers,
            )
            locals_ = locals_ + jnp.where(
                at_plane.reshape((target_node_count,) + (1,) * (locals_.ndim - 1)),
                inherited,
                0.0,
            )
        source_radius = jnp.sqrt(
            jnp.sum(source_schedule.node_half_widths[source_nodes] ** 2, axis=-1)
        )
        target_radius = jnp.sqrt(
            jnp.sum(target_schedule.node_half_widths[target_nodes] ** 2, axis=-1)
        )
        distance = jnp.sqrt(
            jnp.sum(
                (
                    target_schedule.node_centers[target_nodes]
                    - source_schedule.node_centers[source_nodes]
                )
                ** 2,
                axis=-1,
            )
        )
        safe_distance = jnp.maximum(distance, jnp.finfo(distance.dtype).tiny)
        ratio = (source_radius + target_radius) / safe_distance
        gap = jnp.maximum(
            distance - source_radius - target_radius,
            jnp.finfo(distance.dtype).tiny,
        )
        route_tail = (
            source_weight[source_nodes]
            * ratio ** (self.plan.expansion_order + 1)
            / (4.0 * jnp.pi * gap * jnp.maximum(1.0 - ratio, jnp.finfo(ratio.dtype).eps))
        )
        route_tail = jnp.where(far.valid, route_tail, 0.0)
        truncation = MultipoleTruncationEvidence3D(
            geometric_tail_bound=jnp.sum(route_tail),
            maximum_separation_ratio=jnp.max(
                jnp.where(far.valid, ratio, 0.0), initial=0.0
            ),
            well_separated=jnp.all(~far.valid | (ratio < 1.0)),
            expansion_order=self.plan.expansion_order,
        )
        active_target_leaf = target_schedule.logical_point_leaf_slots
        target_leaf = jnp.maximum(active_target_leaf, 0)
        far_local_values = locals_[target_leaf]
        return (
            far_local_values,
            truncation,
            jnp.sum(
                source_schedule.node_active & (source_schedule.node_planes > 0),
                dtype=jnp.int32,
            ),
            jnp.sum(far.valid, dtype=jnp.int32),
            jnp.sum(
                target_schedule.node_active & (target_schedule.node_parents >= 0),
                dtype=jnp.int32,
            ),
        )

    def _plane_evaluate(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike | None,
        *,
        source_normals: ArrayLike | None,
        active_mask: ArrayLike | None,
        target_source_indices: ArrayLike | None,
    ) -> AbstractLaplaceMultipoleEvaluation3D:
        sources, strengths, targets, active, normals, displacement, stale = (
            self._validate_evaluation_inputs(
                source_positions,
                source_strengths,
                target_positions,
                active_mask,
                source_normals,
            )
        )
        if (
            self.source_plane is None
            or self.target_plane is None
            or self.plane_interactions is None
        ):
            raise RuntimeError("Plane topology is not prepared.")
        far_local, truncation, m2m_count, m2l_count, l2l_count = self._plane_far_locals(
            sources, strengths, targets, active, normals
        )
        target_leaf = jnp.maximum(self.target_plane.logical_point_leaf_slots, 0)
        target_centers = self.target_plane.node_centers[target_leaf]
        far_values = jax.vmap(self.l2p)(
            far_local,
            target_centers,
            targets,
        )
        source_leaf = self.source_plane.logical_point_leaf_slots
        pair_mask = jnp.zeros(
            (self.plan.target_capacity, self.plan.source_capacity), dtype=jnp.bool_
        )
        near = self.plane_interactions.near
        # ty: ignore[invalid-argument-type]
        for route in range(self.plan.near_interaction_capacity):
            pair_mask = pair_mask | (
                near.valid[route]
                & (target_leaf[:, None] == near.target_indices[route])
                & (source_leaf[None, :] == near.source_indices[route])
            )
        pair_mask = pair_mask & active[None, :]
        if target_source_indices is None:
            identities = (
                jnp.arange(self.plan.target_capacity, dtype=jnp.int32)
                if self.plan.target_topology == "same-support"
                else jnp.full((self.plan.target_capacity,), -1, dtype=jnp.int32)
            )
        else:
            identities = jnp.asarray(target_source_indices, dtype=jnp.int32)
            if identities.shape != (self.plan.target_capacity,):
                raise ValueError("target_source_indices must match target_capacity.")
        pair_mask = pair_mask & (
            identities[:, None]
            != jnp.arange(self.plan.source_capacity, dtype=jnp.int32)[None, :]
        )
        near_values, p2p_count = self._p2p_masked(
            sources,
            strengths,
            targets,
            pair_mask,
            source_normals=normals,
        )
        values = far_values + near_values
        capacity = self._capacity_evidence()
        finite = jnp.all(jnp.isfinite(values))
        successful = (
            capacity.successful
            & self.plane_interactions.evidence.successful
            & truncation.well_separated
            & finite
            & ~stale
        )
        return LaplaceMultipoleEvaluation3D(
            values=values,
            far_values=far_values,
            near_values=near_values,
            truncation=truncation,
            capacity=capacity,
            maximum_reference_displacement=displacement,
            stale_topology=stale,
            finite=finite,
            successful=successful,
            p2m_count=jnp.sum(active, dtype=jnp.int32),
            m2m_count=m2m_count,
            m2l_count=m2l_count,
            p2l_count=jnp.asarray(0, dtype=jnp.int32),
            l2l_count=l2l_count,
            l2p_count=jnp.asarray(self.plan.target_capacity, dtype=jnp.int32),
            m2p_count=jnp.asarray(0, dtype=jnp.int32),
            p2p_count=p2p_count,
            expansion_order=self.plan.expansion_order,
            source_convention=self.source_convention,
            local_convention=self.local_convention,
            evaluation_id=canonical_fingerprint(
                {
                    "kind": "laplace-multipole-plane-evaluation-3d",
                    "prepared": self.prepared_id,
                }
            ),
        )

    def _capacity_evidence(self) -> MultipoleCapacityEvidence3D:
        if self.plan.execution == "plane_dual":
            if (
                self.source_plane is None
                or self.target_plane is None
                or self.plane_interactions is None
            ):
                raise RuntimeError("Plane topology is not prepared.")
            source_nodes = self.source_plane.evidence
            target_nodes = self.target_plane.evidence
            interactions = self.plane_interactions.evidence
            return MultipoleCapacityEvidence3D(
                required_nodes=source_nodes.required_nodes + target_nodes.required_nodes,
                node_capacity=source_nodes.node_capacity + target_nodes.node_capacity,
                required_far_interactions=interactions.required_far,
                far_interaction_capacity=interactions.far_capacity,
                required_near_interactions=interactions.required_near,
                near_interaction_capacity=interactions.near_capacity,
                successful=interactions.successful,
            )
        if self.topology is None:
            raise RuntimeError("Level-octree topology is not prepared.")
        topology = self.topology
        far_lists = (topology.v_list, topology.w_list, topology.x_list)
        return MultipoleCapacityEvidence3D(
            required_nodes=topology.evidence.node_count,
            node_capacity=topology.evidence.node_count,
            # ty: ignore[invalid-argument-type]
            required_far_interactions=sum(
                interaction.required_routes for interaction in far_lists
            ),
            far_interaction_capacity=jnp.asarray(
                sum(interaction.routes.capacity for interaction in far_lists),
                dtype=jnp.int32,
            ),
            required_near_interactions=topology.u_list.required_routes,
            near_interaction_capacity=jnp.asarray(
                topology.u_list.routes.capacity, dtype=jnp.int32
            ),
            successful=topology.evidence.successful,
        )

    def _validate_evaluation_inputs(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike | None,
        active_mask: ArrayLike | None,
        source_normals: ArrayLike | None,
    ) -> tuple[Array, Array, Array, Array, Array | None, Array, Array]:
        sources = jnp.asarray(source_positions)
        strengths = jnp.asarray(source_strengths)
        targets = sources if target_positions is None else jnp.asarray(target_positions)
        if sources.shape != (self.plan.source_capacity, 3):
            raise ValueError(
                f"source_positions must have shape {(self.plan.source_capacity, 3)}."
            )
        if targets.shape != (self.plan.target_capacity, 3):
            raise ValueError(
                f"target_positions must have shape {(self.plan.target_capacity, 3)}."
            )
        if strengths.ndim < 1 or strengths.shape[0] != self.plan.source_capacity:
            raise ValueError("source_strengths must begin with source_capacity.")
        active = (
            jnp.ones((self.plan.source_capacity,), dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        if active.shape != (self.plan.source_capacity,):
            raise ValueError("active_mask must match source_capacity.")
        normals = None if source_normals is None else jnp.asarray(source_normals)
        if normals is not None and normals.shape != sources.shape:
            raise ValueError("source_normals must match source_positions.")
        references = jnp.concatenate(
            (self.plan.reference_sources, self.plan.reference_targets),
            axis=0,
        ).astype(sources.dtype)
        actual = jnp.concatenate((sources, targets), axis=0)
        displacement = jnp.max(
            jnp.linalg.norm(actual - references, axis=-1),
            initial=0.0,
        )
        finite = jnp.all(jnp.isfinite(actual))
        if self.plan.execution == "plane_dual":
            if self.source_plane is None or self.target_plane is None:
                raise RuntimeError("Plane topology is not prepared.")
            source_leaf = jnp.maximum(
                self.source_plane.logical_point_leaf_slots,
                0,
            )
            target_leaf = jnp.maximum(
                self.target_plane.logical_point_leaf_slots,
                0,
            )
            source_lower = (
                self.source_plane.node_centers[source_leaf]
                - self.source_plane.node_half_widths[source_leaf]
            )
            source_upper = (
                self.source_plane.node_centers[source_leaf]
                + self.source_plane.node_half_widths[source_leaf]
            )
            target_lower = (
                self.target_plane.node_centers[target_leaf]
                - self.target_plane.node_half_widths[target_leaf]
            )
            target_upper = (
                self.target_plane.node_centers[target_leaf]
                + self.target_plane.node_half_widths[target_leaf]
            )
            source_tolerance = (
                8.0
                * jnp.finfo(sources.dtype).eps
                * (1.0 + jnp.abs(source_lower) + jnp.abs(source_upper))
            )
            target_tolerance = (
                8.0
                * jnp.finfo(targets.dtype).eps
                * (1.0 + jnp.abs(target_lower) + jnp.abs(target_upper))
            )
            source_within = jnp.all(
                (sources >= source_lower - source_tolerance)
                & (sources <= source_upper + source_tolerance)
            )
            target_within = jnp.all(
                (targets >= target_lower - target_tolerance)
                & (targets <= target_upper + target_tolerance)
            )
            stale = (
                (displacement > self.plan.maximum_reference_displacement)
                | ~(source_within & target_within)
                | ~finite
            )
        else:
            if self.topology is None:
                raise RuntimeError("Level-octree topology is not prepared.")
            # Sources keep their reference leaves; the separation padding covers
            # their displacement. Targets are located, so they only need a leaf.
            stale = (
                (displacement > self.plan.maximum_reference_displacement)
                | jnp.any(self.topology.locate(targets) < 0)
                | ~finite
            )
        strengths = eqx.error_if(
            strengths,
            stale,
            "Multipole positions leave the prepared frozen-topology envelope.",
        )
        return sources, strengths, targets, active, normals, displacement, stale

    def _source_moments(
        self,
        sources: Array,
        strengths: Array,
        active: Array,
        normals: Array | None,
    ) -> tuple[Array, Array]:
        """P2M into the frozen reference leaves, then M2M up the adaptive octree."""
        topology = self.topology
        # ty: ignore[unresolved-attribute]
        node_count = topology.node_count
        # ty: ignore[unresolved-attribute]
        leaves = topology.point_leaves
        # ty: ignore[unresolved-attribute]
        relative = sources - topology.node_centers[leaves]
        if normals is None:
            basis = jnp.moveaxis(self._p2m_basis(relative), -1, 0)
        else:

            def dipole_basis(vector: Any, normal: Any) -> Any:
                derivative = jax.jacfwd(self._p2m_basis)(vector)
                return jnp.sum(derivative * normal, axis=-1)

            basis = jax.vmap(dipole_basis)(relative, normals)
        strength_flat, payload_shape = _flatten_payload(strengths, 1)
        per_source = basis[..., None] * strength_flat[:, None, None, :]
        per_source = jnp.where(active[:, None, None, None], per_source, 0.0)
        moments = (
            jnp.zeros(
                (node_count,)
                + self.layout.coefficient_shape
                + (strength_flat.shape[-1],),
                dtype=jnp.result_type(per_source.dtype, 1j),
            )
            .at[leaves]
            .add(per_source)
        )
        charge = jnp.where(active, jnp.max(jnp.abs(strength_flat), axis=-1), 0.0)
        charge_mass = jnp.zeros((node_count,), dtype=charge.dtype).at[leaves].add(charge)

        def translate(values: Any, child_centers: Any, parent_centers: Any) -> Any:
            child_moments, child_mass = values
            return (
                jax.vmap(self.m2m)(child_moments, child_centers, parent_centers),
                child_mass,
            )

        # ty: ignore[unresolved-attribute]
        moments, charge_mass = topology.upward_pass((moments, charge_mass), translate)
        return moments.reshape(
            (node_count,) + self.layout.coefficient_shape + payload_shape
        ), charge_mass

    def _far_locals(
        self,
        sources: Array,
        strengths: Array,
        active: Array,
        normals: Array | None,
    ) -> tuple[Array, Array, Array]:
        """Return node moments, node charge mass, and complete node locals.

        Node locals collect V-list M2L and X-list P2L routes and then inherit every
        ancestor local through L2L.
        """
        topology = self.topology
        moments, charge_mass = self._source_moments(sources, strengths, active, normals)
        # ty: ignore[unresolved-attribute]
        centers = topology.node_centers
        # Masked route slots are evaluated one unit away from their expansion
        # center so that every basis stays finite under differentiation.
        offset = jnp.asarray((1.0, 0.0, 0.0), dtype=centers.dtype)
        # ty: ignore[unresolved-attribute]
        far = topology.v_list.routes
        source_center = centers[far.source_indices]
        target_center = jnp.where(
            far.valid[:, None], centers[far.target_indices], source_center + offset
        )
        translated = jax.vmap(self.m2l)(
            moments[far.source_indices], source_center, target_center
        )
        translated = jnp.where(
            far.valid.reshape((-1,) + (1,) * (translated.ndim - 1)), translated, 0.0
        )
        # ty: ignore[unresolved-attribute]
        leaf_routes = topology.x_list.routes
        # ty: ignore[unresolved-attribute]
        points, point_valid = topology.leaf_points(leaf_routes.source_indices)
        point_valid = point_valid & leaf_routes.valid[:, None] & active[points]
        local_center = centers[leaf_routes.target_indices]
        positions = jnp.where(
            point_valid[..., None], sources[points], local_center[:, None, :] + offset
        )
        point_strengths = jnp.where(
            point_valid.reshape(point_valid.shape + (1,) * (strengths.ndim - 1)),
            strengths[points],
            0.0,
        )
        if normals is None:
            expanded = jax.vmap(self.p2l)(positions, point_strengths, local_center)
        else:
            expanded = jax.vmap(
                lambda position, strength, center, normal: self.p2l(
                    position, strength, center, source_normals=normal
                )
            )(positions, point_strengths, local_center, normals[points])
        locals_ = (
            # ty: ignore[unresolved-attribute]
            topology.v_list.execution.reduce(translated, accumulation="deterministic")[0]
            # ty: ignore[unresolved-attribute]
            + topology.x_list.execution.reduce(expanded, accumulation="deterministic")[0]
        )
        # ty: ignore[unresolved-attribute]
        locals_ = topology.downward_pass(locals_, jax.vmap(self.l2l))
        return moments, charge_mass, locals_

    def _octree_truncation(self, charge_mass: Array) -> MultipoleTruncationEvidence3D:
        """Geometric tail evidence of every V (M2L), W (M2P), and X (P2L) route."""
        # ty: ignore[unresolved-attribute]
        sources, radii, distances, valid = self.topology.far_route_geometry()
        tiny = jnp.finfo(radii.dtype).tiny
        distance = jnp.maximum(distances, tiny)
        route_mass = charge_mass[sources]
        ratio = radii / distance
        gap = jnp.maximum(distance - radii, tiny)
        route_tail = (
            route_mass
            * ratio ** (self.plan.expansion_order + 1)
            / (4.0 * jnp.pi * gap * jnp.maximum(1.0 - ratio, jnp.finfo(ratio.dtype).eps))
        )
        return MultipoleTruncationEvidence3D(
            geometric_tail_bound=jnp.sum(jnp.where(valid, route_tail, 0.0)),
            maximum_separation_ratio=jnp.max(jnp.where(valid, ratio, 0.0), initial=0.0),
            well_separated=jnp.all(~valid | (ratio < 1.0)),
            expansion_order=self.plan.expansion_order,
        )

    def _octree_target_routes(
        self,
        moments: Array,
        sources: Array,
        strengths: Array,
        active: Array,
        normals: Array | None,
        targets: Array,
        target_leaves: Array,
        identities: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Evaluate W-list multipoles and U-list direct pairs at every target."""
        topology = self.topology
        # ty: ignore[unresolved-attribute]
        centers = topology.node_centers
        offset = jnp.asarray((1.0, 0.0, 0.0), dtype=centers.dtype)

        def one_target(item: Any) -> Any:
            target, leaf, identity = item
            # ty: ignore[unresolved-attribute]
            nodes, valid = topology.w_list.rows(leaf)
            node_centers = jnp.where(valid[:, None], centers[nodes], target + offset)
            multipole_values = jax.vmap(
                lambda moment, center: self.multipole_to_point(moment, center, target)
            )(moments[nodes], node_centers)
            multipole_value = jnp.sum(
                jnp.where(
                    valid.reshape((-1,) + (1,) * (multipole_values.ndim - 1)),
                    multipole_values,
                    0.0,
                ),
                axis=0,
            )
            # ty: ignore[unresolved-attribute]
            leaves, leaf_valid = topology.u_list.rows(leaf)
            # ty: ignore[unresolved-attribute]
            points, point_valid = topology.leaf_points(leaves)
            points = points.reshape((-1,))
            pair_valid = (
                (point_valid & leaf_valid[:, None]).reshape((-1,))
                & active[points]
                & (points != identity)
            )
            near, count = self._p2p_masked(
                sources[points],
                strengths[points],
                target[None, :],
                pair_valid[None, :],
                source_normals=None if normals is None else normals[points],
            )
            return multipole_value, near[0], count, jnp.sum(valid, dtype=jnp.int32)

        return jax.lax.map(
            one_target,
            (targets, target_leaves, identities),
            batch_size=_TARGET_BATCH_SIZE,
        )

    def far_local(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_centers: ArrayLike | None = None,
        /,
        *,
        source_normals: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
    ) -> MultipoleFarLocal3D:
        """Return the far-only local expansion centered at every target center."""
        sources, strengths, expansion_centers, active, normals, _, stale = (
            self._validate_evaluation_inputs(
                source_positions,
                source_strengths,
                target_centers,
                active_mask,
                source_normals,
            )
        )
        if self.plan.execution == "plane_dual":
            (
                leaf_coefficients,
                truncation,
                m2m_count,
                m2l_count,
                l2l_count,
            ) = self._plane_far_locals(
                sources,
                strengths,
                expansion_centers,
                active,
                normals,
            )
            # ty: ignore[unresolved-attribute]
            target_leaf = jnp.maximum(self.target_plane.logical_point_leaf_slots, 0)
            coefficients = jax.vmap(self.l2l)(
                leaf_coefficients,
                # ty: ignore[unresolved-attribute]
                self.target_plane.node_centers[target_leaf],
                expansion_centers,
            )
            capacity = self._capacity_evidence()
            finite = jnp.all(jnp.isfinite(coefficients))
            successful = capacity.successful & truncation.well_separated & finite & ~stale
            return MultipoleFarLocal3D(
                coefficients=coefficients,
                truncation=truncation,
                capacity=capacity,
                m2m_count=m2m_count,
                m2l_count=m2l_count,
                p2l_count=jnp.asarray(0, dtype=jnp.int32),
                l2l_count=l2l_count + self.plan.target_capacity,
                successful=successful,
                prepared_id=self.prepared_id,
            )
        topology = self.topology
        moments, charge_mass, locals_ = self._far_locals(
            sources, strengths, active, normals
        )
        truncation = self._octree_truncation(charge_mass)
        # ty: ignore[unresolved-attribute]
        centers = topology.node_centers
        offset = jnp.asarray((1.0, 0.0, 0.0), dtype=centers.dtype)
        # ty: ignore[unresolved-attribute]
        target_leaves = jnp.maximum(topology.locate(expansion_centers), 0)
        inherited = jax.vmap(self.l2l)(
            locals_[target_leaves], centers[target_leaves], expansion_centers
        )

        def converted_locals(item: Any) -> Any:
            center, leaf = item
            # ty: ignore[unresolved-attribute]
            nodes, valid = topology.w_list.rows(leaf)
            source_centers = centers[nodes]
            local_centers = jnp.where(valid[:, None], center, source_centers + offset)
            translated = jax.vmap(self.m2l)(moments[nodes], source_centers, local_centers)
            return jnp.sum(
                jnp.where(
                    valid.reshape((-1,) + (1,) * (translated.ndim - 1)), translated, 0.0
                ),
                axis=0,
            ), jnp.sum(valid, dtype=jnp.int32)

        converted, conversions = jax.lax.map(
            converted_locals,
            (expansion_centers, target_leaves),
            batch_size=_TARGET_BATCH_SIZE,
        )
        coefficients = inherited + converted
        capacity = self._capacity_evidence()
        finite = jnp.all(jnp.isfinite(coefficients))
        successful = capacity.successful & truncation.well_separated & finite & ~stale
        # ty: ignore[unresolved-attribute]
        translations = jnp.asarray(topology.node_count - 1, dtype=jnp.int32)
        return MultipoleFarLocal3D(
            coefficients=coefficients,
            truncation=truncation,
            capacity=capacity,
            m2m_count=translations,
            # ty: ignore[unresolved-attribute]
            m2l_count=jnp.sum(topology.v_list.routes.valid, dtype=jnp.int32)
            + jnp.sum(conversions, dtype=jnp.int32),
            # ty: ignore[unresolved-attribute]
            p2l_count=jnp.sum(topology.x_list.routes.valid, dtype=jnp.int32),
            l2l_count=translations + self.plan.target_capacity,
            successful=successful,
            prepared_id=self.prepared_id,
        )

    def evaluate(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike | None = None,
        /,
        *,
        source_normals: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        target_source_indices: ArrayLike | None = None,
    ) -> AbstractLaplaceMultipoleEvaluation3D:
        """Execute every FMM pass with exact direct completion of near routes."""
        if self.plan.execution == "plane_dual":
            return self._plane_evaluate(
                source_positions,
                source_strengths,
                target_positions,
                source_normals=source_normals,
                active_mask=active_mask,
                target_source_indices=target_source_indices,
            )
        sources, strengths, targets, active, normals, displacement, stale = (
            self._validate_evaluation_inputs(
                source_positions,
                source_strengths,
                target_positions,
                active_mask,
                source_normals,
            )
        )
        if target_source_indices is None and self.plan.target_topology == "same-support":
            identities = jnp.arange(self.plan.target_capacity, dtype=jnp.int32)
        elif target_source_indices is None:
            identities = jnp.full((self.plan.target_capacity,), -1, dtype=jnp.int32)
        else:
            identities = jnp.asarray(target_source_indices, dtype=jnp.int32)
            if identities.shape != (self.plan.target_capacity,):
                raise ValueError("target_source_indices must match target_capacity.")
        topology = self.topology
        moments, charge_mass, locals_ = self._far_locals(
            sources, strengths, active, normals
        )
        truncation = self._octree_truncation(charge_mass)
        # ty: ignore[unresolved-attribute]
        target_leaves = jnp.maximum(topology.locate(targets), 0)
        local_values = jax.vmap(self.l2p)(
            locals_[target_leaves],
            # ty: ignore[unresolved-attribute]
            topology.node_centers[target_leaves],
            targets,
        )
        multipole_values, near_values, p2p_counts, m2p_counts = (
            self._octree_target_routes(
                moments,
                sources,
                strengths,
                active,
                normals,
                targets,
                target_leaves,
                identities,
            )
        )
        far_values = local_values + multipole_values
        values = far_values + near_values
        capacity = self._capacity_evidence()
        finite = jnp.all(jnp.isfinite(values))
        successful = capacity.successful & truncation.well_separated & finite & ~stale
        # ty: ignore[unresolved-attribute]
        translations = jnp.asarray(topology.node_count - 1, dtype=jnp.int32)
        return LaplaceMultipoleEvaluation3D(
            values=values,
            far_values=far_values,
            near_values=near_values,
            truncation=truncation,
            capacity=capacity,
            maximum_reference_displacement=displacement,
            stale_topology=stale,
            finite=finite,
            successful=successful,
            p2m_count=jnp.sum(active, dtype=jnp.int32),
            m2m_count=translations,
            # ty: ignore[unresolved-attribute]
            m2l_count=jnp.sum(topology.v_list.routes.valid, dtype=jnp.int32),
            # ty: ignore[unresolved-attribute]
            p2l_count=jnp.sum(topology.x_list.routes.valid, dtype=jnp.int32),
            l2l_count=translations,
            l2p_count=jnp.asarray(self.plan.target_capacity, dtype=jnp.int32),
            m2p_count=jnp.sum(m2p_counts, dtype=jnp.int32),
            p2p_count=jnp.sum(p2p_counts, dtype=jnp.int32),
            expansion_order=self.plan.expansion_order,
            source_convention=self.source_convention,
            local_convention=self.local_convention,
            evaluation_id=canonical_fingerprint(
                {
                    "kind": "laplace-multipole-evaluation-3d",
                    "prepared": self.prepared_id,
                }
            ),
        )

    def __call__(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike | None = None,
        /,
        **kwargs: Any,
    ) -> Array:
        """Return only values while preserving the full evidence API on ``evaluate``."""
        return self.evaluate(
            source_positions, source_strengths, target_positions, **kwargs
        ).values


class PreparedLaplaceMultipole3D(AbstractPreparedLaplaceMultipole3D):
    """Final prepared complete Laplace FMM pipeline."""


__all__ = [
    "LaplaceMultipoleEvaluation3D",
    "LaplaceExecution3D",
    "LaplaceMultipolePlan3D",
    "MultipoleCapacityEvidence3D",
    "MultipoleFarLocal3D",
    "MultipoleResourceEvidence3D",
    "MultipoleTruncationEvidence3D",
    "PreparedLaplaceMultipole3D",
]
