#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared spherical-harmonic translations for the three-dimensional Laplace kernel."""

from __future__ import annotations

import math
from operator import index
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spatial import MortonAddressPlan, SparseLevelOctreePlan
from ....discretization.spatial._level_octree import SparseLevelOctree
from ....discretization.spectral._spherical_layout import SphericalModeLayout
from ....special._solid_harmonic import (
    solid_harmonic_irregular,
    solid_harmonic_regular,
)


TranslationRoute3D = Literal["dense"]


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
    """Target-associated far local expansions and their completion evidence."""

    coefficients: Array
    truncation: MultipoleTruncationEvidence3D
    capacity: MultipoleCapacityEvidence3D
    m2m_count: Array
    m2l_count: Array
    l2l_count: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class LaplaceMultipoleEvaluation3D(eqx.Module, NonTrainableState):
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
    l2l_count: Array
    l2p_count: Array
    p2p_count: Array
    expansion_order: int = eqx.field(static=True)
    source_convention: str = eqx.field(static=True)
    local_convention: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


class LaplaceMultipolePlan3D(StrictModule, NonTrainableState):
    """Fixed-capacity Laplace FMM policy over a frozen sparse level octree.

    ``expansion_order=p`` stores degrees ``0 <= ell <= p`` in a complex,
    spin-zero :class:`SphericalModeLayout`.  Omitting ``reference_targets``
    selects same-support evaluation and exact diagonal exclusion.
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
    far_interaction_capacity: int = eqx.field(static=True)
    near_interaction_capacity: int = eqx.field(static=True)
    maximum_coefficient_bytes: int = eqx.field(static=True)
    translation_route: TranslationRoute3D = eqx.field(static=True)
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
    ):
        sources = np.asarray(reference_sources, dtype=float)
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        same_support = reference_targets is None
        targets = sources if same_support else np.asarray(reference_targets, dtype=float)
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
        combined = int(sources.shape[0] + targets.shape[0])
        far_default = combined * max(depth_ - 1, 1) * min(189, combined)
        near_default = combined * min(27, combined)
        far = (
            far_default
            if far_interaction_capacity is None
            else index(far_interaction_capacity)
        )
        near = (
            near_default
            if near_interaction_capacity is None
            else index(near_interaction_capacity)
        )
        if far <= 0 or near <= 0:
            raise ValueError("Far and near interaction capacities must be positive.")
        self.reference_sources = jnp.asarray(sources)
        self.reference_targets = jnp.asarray(targets)
        self.lower = tuple(float(value) for value in lower_)
        self.upper = tuple(float(value) for value in upper_)
        self.depth = depth_
        self.expansion_order = order
        self.source_capacity = int(sources.shape[0])
        self.target_capacity = int(targets.shape[0])
        self.target_topology = "same-support" if same_support else "rectangular"
        self.maximum_reference_displacement = displacement
        self.far_interaction_capacity = far
        self.near_interaction_capacity = near
        self.maximum_coefficient_bytes = maximum_bytes
        self.translation_route = translation_route
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
            }
        )

    def prepare(self, /) -> PreparedLaplaceMultipole3D:
        """Materialize topology, projection quadrature, and bounded workspaces."""
        return PreparedLaplaceMultipole3D(self)


class PreparedLaplaceMultipole3D(eqx.Module, NonTrainableState):
    """Prepared complete Laplace P2M/M2M/M2L/L2L/L2P/P2P pipeline."""

    plan: LaplaceMultipolePlan3D
    layout: SphericalModeLayout
    topology: SparseLevelOctree
    quadrature_directions: Array
    quadrature_weights: Array
    quadrature_harmonics: Array
    equivalent_inverse: Array
    resources: MultipoleResourceEvidence3D
    prepared_id: str = eqx.field(static=True)
    source_convention: str = eqx.field(static=True)
    local_convention: str = eqx.field(static=True)

    def __init__(self, plan: LaplaceMultipolePlan3D, /):
        if not isinstance(plan, LaplaceMultipolePlan3D):
            raise TypeError("plan must be LaplaceMultipolePlan3D.")
        layout = SphericalModeLayout(plan.expansion_order + 1, spin=0, reality=False)
        combined_reference = jnp.concatenate(
            (plan.reference_sources, plan.reference_targets), axis=0
        )
        combined_capacity = plan.source_capacity + plan.target_capacity
        octree_plan = SparseLevelOctreePlan(
            MortonAddressPlan(plan.lower, plan.upper, plan.depth),
            combined_capacity,
            far_interaction_capacity=plan.far_interaction_capacity,
            near_interaction_capacity=plan.near_interaction_capacity,
        )
        topology = octree_plan.prepare(
            combined_reference,
            stable_ids=jnp.arange(combined_capacity, dtype=jnp.int64),
        )
        if not bool(topology.evidence.successful):
            raise ValueError(
                "SparseLevelOctreePlan capacity is exhausted by the reference topology."
            )
        directions_, weights_ = _translation_quadrature(layout.bandlimit)
        directions = jnp.asarray(directions_)
        weights = jnp.asarray(weights_)
        regular = _mode_basis(layout, directions, radial="regular")
        degree_scale = 1.0 / (2.0 * jnp.arange(layout.bandlimit) + 1.0)
        p2m_matrix = (jnp.conj(regular) * degree_scale[:, None, None]).reshape(
            (-1, directions.shape[0])
        )[layout.valid_indices]
        equivalent_inverse = jnp.linalg.pinv(p2m_matrix)
        quadrature_harmonics = regular
        padded = math.prod(layout.coefficient_shape)
        coefficient_bytes = padded * np.dtype(np.complex128).itemsize
        node_capacity = int(topology.hierarchy.node_active.size)
        required_bytes = 2 * node_capacity * coefficient_bytes
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
                "quadrature_node_count": int(directions.shape[0]),
                "node_capacity": node_capacity,
                "far_interaction_capacity": plan.far_interaction_capacity,
                "near_interaction_capacity": plan.near_interaction_capacity,
            }
        )
        self.plan = plan
        self.layout = layout
        self.topology = topology
        self.quadrature_directions = directions
        self.quadrature_weights = weights
        self.quadrature_harmonics = quadrature_harmonics
        self.equivalent_inverse = equivalent_inverse
        self.resources = MultipoleResourceEvidence3D(
            logical_mode_count=layout.logical_mode_count,
            padded_mode_count=padded,
            coefficient_bytes_per_expansion=coefficient_bytes,
            required_coefficient_bytes=required_bytes,
            maximum_coefficient_bytes=plan.maximum_coefficient_bytes,
            quadrature_node_count=int(directions.shape[0]),
            node_capacity=node_capacity,
            far_interaction_capacity=plan.far_interaction_capacity,
            near_interaction_capacity=plan.near_interaction_capacity,
            within_budget=within_budget,
            evidence_id=evidence_id,
        )
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
                "topology": topology.tree_id,
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

            def dipole_basis(vector, normal):
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

            def dipole_basis(vector, normal):
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

    def _project_local(self, function, center: Array, /) -> Array:
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

            def along(direction, degree=degree):
                def radial(distance):
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
        pair_mask = jnp.ones((targets.shape[0], sources.shape[0]), dtype=bool)
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

    def _capacity_evidence(self) -> MultipoleCapacityEvidence3D:
        hierarchy = self.topology.hierarchy
        topology = self.topology.evidence
        return MultipoleCapacityEvidence3D(
            required_nodes=hierarchy.evidence.required_nodes,
            node_capacity=hierarchy.evidence.node_capacity,
            required_far_interactions=topology.required_far_interactions,
            far_interaction_capacity=topology.far_interaction_capacity,
            required_near_interactions=topology.required_near_interactions,
            near_interaction_capacity=topology.near_interaction_capacity,
            successful=topology.successful,
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
            jnp.ones((self.plan.source_capacity,), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != (self.plan.source_capacity,):
            raise ValueError("active_mask must match source_capacity.")
        normals = None if source_normals is None else jnp.asarray(source_normals)
        if normals is not None and normals.shape != sources.shape:
            raise ValueError("source_normals must match source_positions.")
        references = jnp.concatenate(
            (self.plan.reference_sources, self.plan.reference_targets), axis=0
        ).astype(sources.dtype)
        actual = jnp.concatenate((sources, targets), axis=0)
        displacement = jnp.max(jnp.linalg.norm(actual - references, axis=-1), initial=0.0)
        address = MortonAddressPlan(self.plan.lower, self.plan.upper, self.plan.depth)
        actual_codes = address.encode(actual).codes
        reference_codes = address.encode(references).codes
        stale = (
            (displacement > self.plan.maximum_reference_displacement)
            | jnp.any(actual_codes != reference_codes)
            | jnp.any(~jnp.isfinite(actual))
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
        hierarchy = self.topology.hierarchy
        node_count = hierarchy.node_active.size
        source_leaf = hierarchy.logical_point_leaf_slots[: self.plan.source_capacity]
        safe_leaf = jnp.maximum(source_leaf, 0)
        centers = hierarchy.node_centers[safe_leaf]
        relative = sources - centers
        if normals is None:
            basis = jnp.moveaxis(self._p2m_basis(relative), -1, 0)
        else:

            def dipole_basis(vector, normal):
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
            .at[safe_leaf]
            .add(per_source)
        )
        charge = jnp.max(jnp.abs(strength_flat), axis=-1)
        charge_mass = (
            jnp.zeros((node_count,), dtype=charge.dtype)
            .at[safe_leaf]
            .add(jnp.where(active, charge, 0.0))
        )
        children = hierarchy.node_children
        safe_children = jnp.maximum(children, 0)
        valid_children = children >= 0
        for level in range(self.plan.depth - 1, -1, -1):
            at_level = (
                hierarchy.node_active
                & ~hierarchy.node_is_leaf
                & (hierarchy.node_levels == level)
            )
            child_moments = moments[safe_children]
            child_centers = hierarchy.node_centers[safe_children]
            parent_centers = jnp.broadcast_to(
                hierarchy.node_centers[:, None, :], child_centers.shape
            )
            translated = jax.vmap(
                jax.vmap(self.m2m, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
            )(child_moments, child_centers, parent_centers)
            translated = jnp.where(
                valid_children.reshape(
                    valid_children.shape + (1,) * (translated.ndim - 2)
                ),
                translated,
                0.0,
            )
            parent_moments = jnp.sum(translated, axis=1)
            moments = jnp.where(
                at_level.reshape((node_count,) + (1,) * (moments.ndim - 1)),
                parent_moments,
                moments,
            )
            parent_mass = jnp.sum(
                jnp.where(valid_children, charge_mass[safe_children], 0.0), axis=1
            )
            charge_mass = jnp.where(at_level, parent_mass, charge_mass)
        return moments.reshape(
            (node_count,) + self.layout.coefficient_shape + payload_shape
        ), charge_mass

    def _far_locals(
        self,
        sources: Array,
        strengths: Array,
        active: Array,
        normals: Array | None,
    ) -> tuple[Array, MultipoleTruncationEvidence3D, Array, Array, Array]:
        hierarchy = self.topology.hierarchy
        moments, charge_mass = self._source_moments(sources, strengths, active, normals)
        node_count = hierarchy.node_active.size
        safe_source = jnp.maximum(self.topology.far_sources, 0)
        safe_target = jnp.maximum(self.topology.far_targets, 0)
        active_route = self.topology.far_active
        source_center = hierarchy.node_centers[safe_source]
        raw_target_center = hierarchy.node_centers[safe_target]
        fallback_offset = jnp.asarray((1.0, 0.0, 0.0), dtype=source_center.dtype)
        target_center = jnp.where(
            active_route[:, None],
            raw_target_center,
            source_center + fallback_offset,
        )
        translated = jax.vmap(self.m2l)(
            moments[safe_source], source_center, target_center
        )
        translated = jnp.where(
            active_route.reshape((active_route.size,) + (1,) * (translated.ndim - 1)),
            translated,
            0.0,
        )
        locals_ = (
            jnp.zeros((node_count,) + translated.shape[1:], dtype=translated.dtype)
            .at[safe_target]
            .add(translated)
        )
        source_radius = jnp.linalg.norm(hierarchy.node_half_widths[safe_source], axis=-1)
        target_radius = jnp.linalg.norm(hierarchy.node_half_widths[safe_target], axis=-1)
        distance = jnp.linalg.norm(target_center - source_center, axis=-1)
        safe_distance = jnp.maximum(distance, jnp.finfo(distance.dtype).tiny)
        ratio = (source_radius + target_radius) / safe_distance
        gap = jnp.maximum(
            distance - source_radius - target_radius,
            jnp.finfo(distance.dtype).tiny,
        )
        route_tail = (
            charge_mass[safe_source]
            * ratio ** (self.plan.expansion_order + 1)
            / (4.0 * jnp.pi * gap * jnp.maximum(1.0 - ratio, jnp.finfo(ratio.dtype).eps))
        )
        route_tail = jnp.where(active_route, route_tail, 0.0)
        maximum_ratio = jnp.max(jnp.where(active_route, ratio, 0.0), initial=0.0)
        well_separated = jnp.all(~active_route | (ratio < 1.0))
        parent = jnp.maximum(hierarchy.node_parents, 0)
        for level in range(1, self.plan.depth + 1):
            at_level = hierarchy.node_active & (hierarchy.node_levels == level)
            inherited = jax.vmap(self.l2l)(
                locals_[parent], hierarchy.node_centers[parent], hierarchy.node_centers
            )
            locals_ = locals_ + jnp.where(
                at_level.reshape((node_count,) + (1,) * (locals_.ndim - 1)),
                inherited,
                0.0,
            )
        truncation = MultipoleTruncationEvidence3D(
            geometric_tail_bound=jnp.sum(route_tail),
            maximum_separation_ratio=maximum_ratio,
            well_separated=well_separated,
            expansion_order=self.plan.expansion_order,
        )
        m2m_count = jnp.sum(
            hierarchy.node_active & ~hierarchy.node_is_leaf, dtype=jnp.int32
        )
        m2l_count = jnp.sum(active_route, dtype=jnp.int32)
        l2l_count = jnp.maximum(hierarchy.evidence.active_nodes - 1, 0)
        return locals_, truncation, m2m_count, m2l_count, l2l_count

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
        """Return the far-only local expansion associated with every target leaf."""
        sources, strengths, _targets, active, normals, _, stale = (
            self._validate_evaluation_inputs(
                source_positions,
                source_strengths,
                target_centers,
                active_mask,
                source_normals,
            )
        )
        locals_, truncation, m2m_count, m2l_count, l2l_count = self._far_locals(
            sources, strengths, active, normals
        )
        hierarchy = self.topology.hierarchy
        target_leaf = hierarchy.logical_point_leaf_slots[
            self.plan.source_capacity : self.plan.source_capacity
            + self.plan.target_capacity
        ]
        coefficients = locals_[jnp.maximum(target_leaf, 0)]
        capacity = self._capacity_evidence()
        finite = jnp.all(jnp.isfinite(coefficients))
        successful = capacity.successful & truncation.well_separated & finite & ~stale
        return MultipoleFarLocal3D(
            coefficients=coefficients,
            truncation=truncation,
            capacity=capacity,
            m2m_count=m2m_count,
            m2l_count=m2l_count,
            l2l_count=l2l_count,
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
    ) -> LaplaceMultipoleEvaluation3D:
        """Execute every FMM pass with exact direct completion of near routes."""
        sources, strengths, targets, active, normals, displacement, stale = (
            self._validate_evaluation_inputs(
                source_positions,
                source_strengths,
                target_positions,
                active_mask,
                source_normals,
            )
        )
        locals_, truncation, m2m_count, m2l_count, l2l_count = self._far_locals(
            sources, strengths, active, normals
        )
        hierarchy = self.topology.hierarchy
        source_leaf = hierarchy.logical_point_leaf_slots[: self.plan.source_capacity]
        target_leaf = hierarchy.logical_point_leaf_slots[
            self.plan.source_capacity : self.plan.source_capacity
            + self.plan.target_capacity
        ]
        target_local = locals_[jnp.maximum(target_leaf, 0)]
        far_values = jax.vmap(self.l2p)(
            target_local, hierarchy.node_centers[jnp.maximum(target_leaf, 0)], targets
        )
        pair_mask = jnp.zeros(
            (self.plan.target_capacity, self.plan.source_capacity), dtype=bool
        )
        for route in range(self.plan.near_interaction_capacity):
            pair_mask = pair_mask | (
                self.topology.near_active[route]
                & (target_leaf[:, None] == self.topology.near_targets[route])
                & (source_leaf[None, :] == self.topology.near_sources[route])
            )
        pair_mask = pair_mask & active[None, :]
        if target_source_indices is None and self.plan.target_topology == "same-support":
            identities = jnp.arange(self.plan.target_capacity, dtype=jnp.int32)
        elif target_source_indices is None:
            identities = jnp.full((self.plan.target_capacity,), -1, dtype=jnp.int32)
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
        successful = capacity.successful & truncation.well_separated & finite & ~stale
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
            l2l_count=l2l_count,
            l2p_count=jnp.asarray(self.plan.target_capacity, dtype=jnp.int32),
            p2p_count=p2p_count,
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
        **kwargs,
    ) -> Array:
        """Return only values while preserving the full evidence API on ``evaluate``."""
        return self.evaluate(
            source_positions, source_strengths, target_positions, **kwargs
        ).values


__all__ = [
    "LaplaceMultipoleEvaluation3D",
    "LaplaceMultipolePlan3D",
    "MultipoleCapacityEvidence3D",
    "MultipoleFarLocal3D",
    "MultipoleResourceEvidence3D",
    "MultipoleTruncationEvidence3D",
    "PreparedLaplaceMultipole3D",
]
