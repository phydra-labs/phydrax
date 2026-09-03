#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import BoundaryAtlas
from ...linalg import (
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
    ConstraintMap,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ._identity import InterfaceId


def _positive_finite(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _coordinate_dtype(space: AbstractVectorSpace, /) -> np.dtype:
    dtypes = {np.dtype(spec.dtype) for spec in jax.tree.leaves(space.structure())}
    if len(dtypes) != 1:
        raise TypeError("Interface spaces must have one coordinate dtype.")
    return next(iter(dtypes))


class TraceRoute(StrictModule, NonTrainableState):
    """One oriented tensor-patch boundary route selected by axis and side."""

    axis: int = eqx.field(static=True)
    side: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, axis: int, side: int, /):
        axis_ = int(axis)
        side_ = int(side)
        if axis_ < 0 or side_ not in (-1, 1):
            raise ValueError("Trace routes require a nonnegative axis and side +/-1.")
        self.axis = axis_
        self.side = side_
        self.route_id = canonical_fingerprint(
            {"kind": "iga-trace-route", "axis": axis_, "side": side_}
        )

    def assert_dimension(self, dimension: int, /) -> None:
        if not 0 <= self.axis < int(dimension):
            raise ValueError("Trace route axis is outside the patch dimension.")


class InterfaceParameterMap(StrictModule, NonTrainableState):
    """Full-column-rank affine map from common to one trace parameter chart."""

    matrix: Array
    offset: Array
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    minimum_rank_ratio: float = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        offset: ArrayLike,
        /,
        *,
        rank_tolerance: float = 1.0e-12,
    ):
        matrix_host = np.asarray(matrix, dtype=float)
        offset_host = np.asarray(offset, dtype=float)
        tolerance = _positive_finite("rank_tolerance", rank_tolerance)
        if matrix_host.ndim != 2 or min(matrix_host.shape) <= 0:
            raise ValueError("Interface parameter matrix must be nonempty and rank two.")
        target_dimension, source_dimension = matrix_host.shape
        if target_dimension < source_dimension:
            raise ValueError(
                "Interface parameter maps cannot reduce parameter dimension."
            )
        if offset_host.shape != (target_dimension,):
            raise ValueError("Interface parameter-map offset has the wrong shape.")
        if np.any(~np.isfinite(matrix_host)) or np.any(~np.isfinite(offset_host)):
            raise ValueError("Interface parameter maps must be finite.")
        singular_values = np.linalg.svd(matrix_host, compute_uv=False)
        rank_ratio = float(singular_values[-1] / singular_values[0])
        if not np.isfinite(rank_ratio) or rank_ratio <= tolerance:
            raise ValueError("Interface parameter map is not full column rank.")
        orientation = (
            int(np.sign(np.linalg.det(matrix_host)))
            if target_dimension == source_dimension
            else 0
        )
        self.matrix = jnp.asarray(matrix_host)
        self.offset = jnp.asarray(offset_host)
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        self.orientation = orientation
        self.minimum_rank_ratio = rank_ratio
        self.map_id = canonical_fingerprint(
            {
                "kind": "iga-interface-parameter-map",
                "matrix": array_tree_fingerprint(matrix_host),
                "offset": array_tree_fingerprint(offset_host),
                "rank_tolerance": tolerance,
            }
        )

    @classmethod
    def identity(cls, dimension: int, /) -> InterfaceParameterMap:
        dimension_ = int(dimension)

        if dimension_ <= 0:
            raise ValueError("Interface parameter dimension must be positive.")
        return cls(np.eye(dimension_), np.zeros((dimension_,)))

    def map(self, reference: ArrayLike, /) -> Array:
        reference_ = jnp.asarray(reference, dtype=self.matrix.dtype)
        if reference_.shape[-1:] != (self.source_dimension,):
            raise ValueError(
                f"Common interface reference must end in {self.source_dimension} coordinates."
            )
        return contract("...i,ji->...j", reference_, self.matrix) + self.offset

    def differential(self, reference: ArrayLike, /) -> Array:
        reference_ = jnp.asarray(reference)
        if reference_.shape[-1:] != (self.source_dimension,):
            raise ValueError(
                f"Common interface reference must end in {self.source_dimension} coordinates."
            )
        return jnp.broadcast_to(self.matrix, (*reference_.shape[:-1], *self.matrix.shape))


class PeriodicSelfInterface(StrictModule, NonTrainableState):
    """Explicit periodic identification of two distinct traces on one patch."""

    patch_id: str = eqx.field(static=True)
    left_route: TraceRoute
    right_route: TraceRoute
    identity: InterfaceId
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_id: str,
        left_route: TraceRoute,
        right_route: TraceRoute,
        /,
        *,
        dimension: int | None = None,
    ):
        patch = str(patch_id)
        if not patch:
            raise ValueError("patch_id must be non-empty.")
        if not isinstance(left_route, TraceRoute) or not isinstance(
            right_route, TraceRoute
        ):
            raise TypeError("Periodic self interfaces require TraceRoute values.")
        if left_route.route_id == right_route.route_id:
            raise ValueError("A periodic self interface requires two distinct traces.")
        if dimension is not None:
            left_route.assert_dimension(dimension)
            right_route.assert_dimension(dimension)
        identity = InterfaceId(
            patch,
            patch,
            (left_route.axis, left_route.side),
            (right_route.axis, right_route.side),
            periodic=True,
        )
        self.patch_id = patch
        self.left_route = left_route
        self.right_route = right_route
        self.identity = identity
        self.interface_id = identity.interface_id


class PatchInterface(StrictModule, NonTrainableState):
    """Two atlas traces parameterized by one exact common interface chart."""

    left_atlas: BoundaryAtlas
    right_atlas: BoundaryAtlas
    left_parameter_map: InterfaceParameterMap
    right_parameter_map: InterfaceParameterMap
    left_patch_id: str = eqx.field(static=True)
    right_patch_id: str = eqx.field(static=True)
    left_chart: int = eqx.field(static=True)
    right_chart: int = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_patch_id: str,
        right_patch_id: str,
        left_atlas: BoundaryAtlas,
        right_atlas: BoundaryAtlas,
        /,
        *,
        left_chart: int,
        right_chart: int,
        left_parameter_map: InterfaceParameterMap,
        right_parameter_map: InterfaceParameterMap,
        orientation: int,
        periodic: bool = False,
    ):
        left_id = str(left_patch_id)
        right_id = str(right_patch_id)
        left_chart_ = int(left_chart)
        right_chart_ = int(right_chart)
        orientation_ = int(orientation)
        if not left_id or not right_id:
            raise ValueError("Patch interface IDs must be non-empty.")
        if not isinstance(left_atlas, BoundaryAtlas) or not isinstance(
            right_atlas, BoundaryAtlas
        ):
            raise TypeError("Patch interfaces require two BoundaryAtlas values.")
        if not isinstance(left_parameter_map, InterfaceParameterMap) or not isinstance(
            right_parameter_map, InterfaceParameterMap
        ):
            raise TypeError("Patch interfaces require two InterfaceParameterMap values.")
        if (
            not 0 <= left_chart_ < left_atlas.num_charts
            or not 0 <= right_chart_ < right_atlas.num_charts
        ):
            raise ValueError("Patch interface chart index is outside its atlas.")
        if left_atlas.ambient_dimension != right_atlas.ambient_dimension:
            raise ValueError("Patch interface ambient dimensions must agree.")
        if left_parameter_map.target_dimension != left_atlas.reference_dimension:
            raise ValueError("Left parameter map does not target the left atlas chart.")
        if right_parameter_map.target_dimension != right_atlas.reference_dimension:
            raise ValueError("Right parameter map does not target the right atlas chart.")
        if left_parameter_map.source_dimension != right_parameter_map.source_dimension:
            raise ValueError("Interface parameter maps need one common source dimension.")
        if left_atlas.reference_dimension != right_atlas.reference_dimension:
            raise ValueError("Patch traces must have equal parameter dimension.")
        if orientation_ not in (-1, 1):
            raise ValueError("Patch interface orientation must be +1 or -1.")
        self.left_atlas = left_atlas
        self.right_atlas = right_atlas
        self.left_parameter_map = left_parameter_map
        self.right_parameter_map = right_parameter_map
        self.left_patch_id = left_id
        self.right_patch_id = right_id
        self.left_chart = left_chart_
        self.right_chart = right_chart_
        self.orientation = orientation_
        self.periodic = bool(periodic)
        self.interface_id = canonical_fingerprint(
            {
                "kind": "iga-patch-interface",
                "left_patch": left_id,
                "right_patch": right_id,
                "left_atlas": left_atlas.source_id,
                "right_atlas": right_atlas.source_id,
                "left_chart": left_chart_,
                "right_chart": right_chart_,
                "left_map": left_parameter_map.map_id,
                "right_map": right_parameter_map.map_id,
                "orientation": orientation_,
                "periodic": bool(periodic),
            }
        )

    @property
    def reference_dimension(self) -> int:
        return self.left_parameter_map.source_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.left_atlas.ambient_dimension

    def _references(self, reference: ArrayLike, /) -> tuple[Array, Array, Array]:
        common = jnp.asarray(reference, dtype=self.left_parameter_map.matrix.dtype)
        if common.ndim < 1 or common.shape[-1] != self.reference_dimension:
            raise ValueError(
                "Common interface reference has the wrong trailing dimension."
            )
        return (
            common,
            self.left_parameter_map.map(common),
            self.right_parameter_map.map(common),
        )

    def traces(self, reference: ArrayLike, /) -> tuple[Array, Array]:
        common, left_reference, right_reference = self._references(reference)
        leading = common.shape[:-1]
        left_indices = jnp.full(leading, self.left_chart, dtype=jnp.int32)
        right_indices = jnp.full(leading, self.right_chart, dtype=jnp.int32)
        return (
            self.left_atlas.map(left_indices, left_reference),
            self.right_atlas.map(right_indices, right_reference),
        )

    def trace_differentials(self, reference: ArrayLike, /) -> tuple[Array, Array]:
        common, left_reference, right_reference = self._references(reference)
        leading = common.shape[:-1]
        left_indices = jnp.full(leading, self.left_chart, dtype=jnp.int32)
        right_indices = jnp.full(leading, self.right_chart, dtype=jnp.int32)
        left_atlas_differential = self.left_atlas.differential(
            left_indices, left_reference
        )
        right_atlas_differential = self.right_atlas.differential(
            right_indices, right_reference
        )
        return (
            contract(
                "...ai,ij->...aj",
                left_atlas_differential,
                self.left_parameter_map.matrix,
            ),
            contract(
                "...ai,ij->...aj",
                right_atlas_differential,
                self.right_parameter_map.matrix,
            ),
        )


class InterfaceQualificationEvidence(StrictModule, NonTrainableState):
    """Scale-free trace, projective, orientation, normal, and rank evidence."""

    maximum_trace_residual: Array
    maximum_projective_residual: Array
    minimum_orientation_alignment: Array
    minimum_normal_opposition: Array
    minimum_rank_ratio: Array
    projective_weight_scale: Array
    sample_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class InterfaceCertificate(StrictModule, NonTrainableState):
    """Qualification token required before strong or weak interface lowering."""

    evidence: InterfaceQualificationEvidence
    interface_id: str = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    weight_tolerance: float = eqx.field(static=True)
    orientation_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def assert_matches(self, interface: PatchInterface, /) -> None:
        if not isinstance(interface, PatchInterface):
            raise TypeError("interface must be a PatchInterface.")
        if interface.interface_id != self.interface_id:
            raise ValueError("Interface certificate does not match this patch interface.")

    def lower_c0_constraint_map(
        self,
        interface: PatchInterface,
        left_space: AbstractVectorSpace,
        right_space: AbstractVectorSpace,
        left_trace_dofs: Sequence[int],
        right_trace_dofs: Sequence[int],
        /,
    ) -> ConstraintMap:
        self.assert_matches(interface)
        return _lower_c0_constraint_map(
            self,
            left_space,
            right_space,
            left_trace_dofs,
            right_trace_dofs,
        )


def _trace_orientation_alignment(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    dimension = left.shape[-1]
    values = []
    for left_value, right_value in zip(left, right, strict=True):
        left_gram = left_value.T @ left_value
        right_gram = right_value.T @ right_value
        if dimension == 1:
            value = float(left_value[:, 0] @ right_value[:, 0])
            value /= float(np.sqrt(left_gram[0, 0] * right_gram[0, 0]))
        else:
            transition = np.linalg.solve(left_gram, left_value.T @ right_value)
            determinant = float(np.linalg.det(transition))
            scale = float(np.sqrt(np.linalg.det(right_gram) / np.linalg.det(left_gram)))
            value = determinant / scale
        values.append(value)
    return np.asarray(values)


def certify_patch_interface(
    interface: PatchInterface,
    reference: ArrayLike,
    /,
    *,
    left_projective_weights: ArrayLike | None = None,
    right_projective_weights: ArrayLike | None = None,
    trace_tolerance: float = 1.0e-9,
    weight_tolerance: float = 1.0e-9,
    orientation_tolerance: float = 1.0e-8,
    rank_tolerance: float = 1.0e-10,
) -> InterfaceCertificate:
    """Check the complete geometric/projective interface contract or fail closed."""
    if not isinstance(interface, PatchInterface):
        raise TypeError("interface must be a PatchInterface.")
    trace_tol = _positive_finite("trace_tolerance", trace_tolerance)
    weight_tol = _positive_finite("weight_tolerance", weight_tolerance)
    orientation_tol = _positive_finite("orientation_tolerance", orientation_tolerance)
    rank_tol = _positive_finite("rank_tolerance", rank_tolerance)
    common, left_reference, right_reference = interface._references(reference)
    if common.ndim != 2 or common.shape[0] == 0:
        raise ValueError("Interface qualification requires one non-empty sample batch.")
    left_trace, right_trace = interface.traces(common)
    left_host = np.asarray(left_trace)
    right_host = np.asarray(right_trace)
    coordinate_scale = max(
        float(np.max(np.linalg.norm(left_host, axis=-1))),
        float(np.max(np.linalg.norm(right_host, axis=-1))),
        1.0,
    )
    trace_residual = float(
        np.max(np.linalg.norm(left_host - right_host, axis=-1)) / coordinate_scale
    )
    if not np.isfinite(trace_residual) or trace_residual > trace_tol:
        raise ValueError("Patch interface trace equality qualification failed.")
    if (left_projective_weights is None) != (right_projective_weights is None):
        raise ValueError("Projective interface weights must be supplied for both traces.")
    projective_residual = 0.0
    projective_scale = 1.0
    if left_projective_weights is not None and right_projective_weights is not None:
        left_weights = np.asarray(left_projective_weights, dtype=float)
        right_weights = np.asarray(right_projective_weights, dtype=float)
        if (
            left_weights.shape != common.shape[:-1]
            or right_weights.shape != left_weights.shape
        ):
            raise ValueError(
                "Projective weights must contain one scalar per interface sample."
            )
        if (
            np.any(~np.isfinite(left_weights))
            or np.any(~np.isfinite(right_weights))
            or np.any(left_weights <= 0.0)
            or np.any(right_weights <= 0.0)
        ):
            raise ValueError("Projective interface weights must be finite and positive.")
        ratios = left_weights / right_weights
        projective_scale = float(np.mean(ratios))
        projective_residual = float(np.max(np.abs(ratios / projective_scale - 1.0)))
        if projective_residual > weight_tol:
            raise ValueError("Patch interface projective-weight qualification failed.")
    left_differential, right_differential = interface.trace_differentials(common)
    left_differential_host = np.asarray(left_differential)
    right_differential_host = np.asarray(right_differential)
    left_singular = np.linalg.svd(left_differential_host, compute_uv=False)
    right_singular = np.linalg.svd(right_differential_host, compute_uv=False)
    rank_ratios = np.concatenate(
        (
            left_singular[..., -1] / left_singular[..., 0],
            right_singular[..., -1] / right_singular[..., 0],
        )
    )
    minimum_rank = float(np.min(rank_ratios))
    if not np.isfinite(minimum_rank) or minimum_rank <= rank_tol:
        raise ValueError("Patch interface parameterized trace is rank deficient.")
    alignments = _trace_orientation_alignment(
        left_differential_host, right_differential_host
    )
    signed_alignments = interface.orientation * alignments
    minimum_alignment = float(np.min(signed_alignments))
    if not np.isfinite(minimum_alignment) or minimum_alignment < 1.0 - orientation_tol:
        raise ValueError("Patch interface orientation qualification failed.")
    count = int(common.shape[0])
    left_indices = jnp.full((count,), interface.left_chart, dtype=jnp.int32)
    right_indices = jnp.full((count,), interface.right_chart, dtype=jnp.int32)
    left_normal = np.asarray(
        interface.left_atlas.frame(left_indices, left_reference).normal
    )
    right_normal = np.asarray(
        interface.right_atlas.frame(right_indices, right_reference).normal
    )
    opposition = -np.sum(left_normal * right_normal, axis=-1)
    minimum_opposition = float(np.min(opposition))
    if not np.isfinite(minimum_opposition) or minimum_opposition < 1.0 - orientation_tol:
        raise ValueError("Patch interface outward normals are not opposed.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "iga-interface-qualification-evidence",
            "interface": interface.interface_id,
            "samples": array_tree_fingerprint(np.asarray(common)),
            "maximum_trace_residual": trace_residual,
            "maximum_projective_residual": projective_residual,
            "minimum_orientation_alignment": minimum_alignment,
            "minimum_normal_opposition": minimum_opposition,
            "minimum_rank_ratio": minimum_rank,
            "projective_weight_scale": projective_scale,
        }
    )
    evidence = InterfaceQualificationEvidence(
        jnp.asarray(trace_residual),
        jnp.asarray(projective_residual),
        jnp.asarray(minimum_alignment),
        jnp.asarray(minimum_opposition),
        jnp.asarray(minimum_rank),
        jnp.asarray(projective_scale),
        count,
        evidence_id,
    )
    return InterfaceCertificate(
        evidence,
        interface.interface_id,
        trace_tol,
        weight_tol,
        orientation_tol,
        rank_tol,
        interface.periodic,
        canonical_fingerprint(
            {
                "kind": "iga-interface-certificate",
                "interface": interface.interface_id,
                "evidence": evidence_id,
                "trace_tolerance": trace_tol,
                "weight_tolerance": weight_tol,
                "orientation_tolerance": orientation_tol,
                "rank_tolerance": rank_tol,
            }
        ),
    )


def _lower_c0_constraint_map(
    certificate: InterfaceCertificate,
    left_space: AbstractVectorSpace,
    right_space: AbstractVectorSpace,
    left_trace_dofs: Sequence[int],
    right_trace_dofs: Sequence[int],
    /,
) -> ConstraintMap:
    if not isinstance(left_space, AbstractVectorSpace) or not isinstance(
        right_space, AbstractVectorSpace
    ):
        raise TypeError("C0 lowering requires two AbstractVectorSpace values.")
    left_dtype = _coordinate_dtype(left_space)
    right_dtype = _coordinate_dtype(right_space)
    if left_dtype != right_dtype:
        raise TypeError("C0 interface spaces must have equal coordinate dtype.")
    left = tuple(int(value) for value in left_trace_dofs)
    right = tuple(int(value) for value in right_trace_dofs)
    if not left or len(left) != len(right):
        raise ValueError("C0 trace routes must be nonempty and have equal length.")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("C0 trace routes must be one-to-one.")
    if any(value < 0 or value >= left_space.size for value in left) or any(
        value < 0 or value >= right_space.size for value in right
    ):
        raise ValueError("C0 trace route is outside its vector space.")
    full_space = BlockSpace(
        (left_space, right_space),
        names=("left", "right"),
        space_id=canonical_fingerprint(
            {
                "kind": "iga-c0-full-space",
                "interface": certificate.interface_id,
                "left": left_space.space_id,
                "right": right_space.space_id,
            }
        ),
    )
    full_size = full_space.size
    parent = np.arange(full_size, dtype=np.int32)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left_index, right_index in zip(left, right, strict=True):
        left_root = root(left_index)
        right_root = root(left_space.size + right_index)
        parent[right_root] = left_root
    representatives = [root(index) for index in range(full_size)]
    reduced_by_representative: dict[int, int] = {}
    full_to_reduced = []
    for representative in representatives:
        if representative not in reduced_by_representative:
            reduced_by_representative[representative] = len(reduced_by_representative)
        full_to_reduced.append(reduced_by_representative[representative])
    route = np.asarray(full_to_reduced, dtype=np.int32)
    reduced_space = ArraySpace(
        (len(reduced_by_representative),),
        dtype=left_dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "iga-c0-reduced-space",
                "interface": certificate.interface_id,
                "full_space": full_space.space_id,
                "trace_pairs": list(zip(left, right, strict=True)),
            }
        ),
    )
    relation = EdgeRelation(
        route,
        np.arange(full_size, dtype=np.int32),
        source_size=reduced_space.size,
        target_size=full_size,
    )
    operator = SparseCoordinateOperator(
        relation,
        jnp.ones((full_size,), dtype=left_dtype),
        source=reduced_space,
        target=full_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "iga-c0-prolongation",
                "certificate": certificate.certificate_id,
                "full_to_reduced": full_to_reduced,
            }
        ),
    )
    return ConstraintMap(
        full_space,
        reduced_space,
        operator,
        constraint_id=canonical_fingerprint(
            {
                "kind": "iga-certified-c0-interface-constraint",
                "certificate": certificate.certificate_id,
                "prolongation": operator.operator_id,
            }
        ),
    )


class PeriodicCompatibilityPayload(StrictModule, NonTrainableState):
    """Certified periodic relation consumable by boundary/coupling providers."""

    left_map: InterfaceParameterMap
    right_map: InterfaceParameterMap
    orientation: int = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class PeriodicCompatibilityAdapter(StrictModule, NonTrainableState):
    """Expose an already-certified periodic interface without a second boundary stack."""

    interface: PatchInterface
    certificate: InterfaceCertificate
    adapter_id: str = eqx.field(static=True)

    def __init__(self, interface: PatchInterface, certificate: InterfaceCertificate, /):
        if not isinstance(interface, PatchInterface):
            raise TypeError("interface must be a PatchInterface.")
        if not isinstance(certificate, InterfaceCertificate):
            raise TypeError("Periodic compatibility requires an InterfaceCertificate.")
        certificate.assert_matches(interface)
        if not interface.periodic or not certificate.periodic:
            raise ValueError("Periodic compatibility requires a periodic interface.")
        self.interface = interface
        self.certificate = certificate
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "iga-periodic-compatibility-adapter",
                "interface": interface.interface_id,
                "certificate": certificate.certificate_id,
            }
        )

    def provider_payload(self, /) -> PeriodicCompatibilityPayload:
        return PeriodicCompatibilityPayload(
            self.interface.left_parameter_map,
            self.interface.right_parameter_map,
            self.interface.orientation,
            self.interface.interface_id,
            self.certificate.certificate_id,
            canonical_fingerprint(
                {
                    "kind": "iga-periodic-compatibility-payload",
                    "adapter": self.adapter_id,
                }
            ),
        )

    def lower_c0_constraint_map(
        self,
        left_space: AbstractVectorSpace,
        right_space: AbstractVectorSpace,
        left_trace_dofs: Sequence[int],
        right_trace_dofs: Sequence[int],
        /,
    ) -> ConstraintMap:
        return self.certificate.lower_c0_constraint_map(
            self.interface,
            left_space,
            right_space,
            left_trace_dofs,
            right_trace_dofs,
        )


class H1NitscheCoercivityCertificate(StrictModule, NonTrainableState):
    """Checked lower gate for symmetric H1 Nitsche coupling."""

    interface_certificate: InterfaceCertificate
    trace_constant: float = eqx.field(static=True)
    inverse_constant: float = eqx.field(static=True)
    penalty: float = eqx.field(static=True)
    minimum_penalty: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


def certify_h1_nitsche_coercivity(
    interface_certificate: InterfaceCertificate,
    /,
    *,
    trace_constant: float,
    inverse_constant: float,
    penalty: float,
    safety_factor: float = 2.0,
) -> H1NitscheCoercivityCertificate:
    if not isinstance(interface_certificate, InterfaceCertificate):
        raise TypeError("Nitsche coercivity requires an InterfaceCertificate.")
    trace = _positive_finite("trace_constant", trace_constant)
    inverse = _positive_finite("inverse_constant", inverse_constant)
    penalty_ = _positive_finite("penalty", penalty)
    safety = _positive_finite("safety_factor", safety_factor)
    if safety <= 1.0:
        raise ValueError("Nitsche safety_factor must exceed one.")
    minimum = safety * trace * trace * inverse
    if penalty_ < minimum:
        raise ValueError("Nitsche penalty does not satisfy the coercivity gate.")
    return H1NitscheCoercivityCertificate(
        interface_certificate,
        trace,
        inverse,
        penalty_,
        minimum,
        canonical_fingerprint(
            {
                "kind": "iga-h1-nitsche-coercivity-certificate",
                "interface_certificate": interface_certificate.certificate_id,
                "trace_constant": trace,
                "inverse_constant": inverse,
                "penalty": penalty_,
                "minimum_penalty": minimum,
            }
        ),
    )


class H1NitscheProviderPayload(StrictModule, NonTrainableState):
    """Geometry-independent coefficients for the existing variational provider stack."""

    jump_signs: Array
    average_weights: Array
    penalty: Array
    interface_id: str = eqx.field(static=True)
    coercivity_certificate_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class H1NitscheInterfacePlan(StrictModule, NonTrainableState):
    """Symmetric H1 Nitsche route guarded by explicit coercivity evidence."""

    interface: PatchInterface
    certificate: InterfaceCertificate
    coercivity: H1NitscheCoercivityCertificate
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface: PatchInterface,
        certificate: InterfaceCertificate,
        coercivity: H1NitscheCoercivityCertificate,
        /,
    ):
        if not isinstance(interface, PatchInterface):
            raise TypeError("interface must be a PatchInterface.")
        if not isinstance(certificate, InterfaceCertificate):
            raise TypeError("Nitsche lowering requires an InterfaceCertificate.")
        if not isinstance(coercivity, H1NitscheCoercivityCertificate):
            raise TypeError("Nitsche lowering requires a coercivity stability gate.")
        certificate.assert_matches(interface)
        if coercivity.interface_certificate.certificate_id != certificate.certificate_id:
            raise ValueError("Nitsche coercivity evidence belongs to another interface.")
        self.interface = interface
        self.certificate = certificate
        self.coercivity = coercivity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-h1-nitsche-interface-plan",
                "interface": interface.interface_id,
                "certificate": certificate.certificate_id,
                "coercivity": coercivity.certificate_id,
                "form": "symmetric-consistent-adjoint-consistent",
            }
        )

    def provider_payload(self, /) -> H1NitscheProviderPayload:
        return H1NitscheProviderPayload(
            jnp.asarray((1.0, -1.0)),
            jnp.asarray((0.5, 0.5)),
            jnp.asarray(self.coercivity.penalty),
            self.interface.interface_id,
            self.coercivity.certificate_id,
            canonical_fingerprint(
                {"kind": "iga-h1-nitsche-provider-payload", "plan": self.plan_id}
            ),
        )


class MortarCrosspointPlan(StrictModule, NonTrainableState):
    """Single-owner crosspoint elimination for one mortar multiplier trace."""

    primal_trace_size: int = eqx.field(static=True)
    multiplier_size: int = eqx.field(static=True)
    crosspoint_trace_dofs: tuple[int, ...] = eqx.field(static=True)
    excluded_multiplier_dofs: tuple[int, ...] = eqx.field(static=True)
    retained_multiplier_dofs: Array
    owner_patch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal_trace_size: int,
        multiplier_size: int,
        /,
        *,
        crosspoint_trace_dofs: Sequence[int] = (),
        excluded_multiplier_dofs: Sequence[int] = (),
        owner_patch_id: str,
    ):
        primal_size = int(primal_trace_size)
        multiplier_size_ = int(multiplier_size)
        crosspoints = tuple(int(value) for value in crosspoint_trace_dofs)
        excluded = tuple(int(value) for value in excluded_multiplier_dofs)
        owner = str(owner_patch_id)
        if primal_size <= 0 or multiplier_size_ <= 0 or not owner:
            raise ValueError("Mortar crosspoint dimensions and owner must be valid.")
        if len(set(crosspoints)) != len(crosspoints) or any(
            value < 0 or value >= primal_size for value in crosspoints
        ):
            raise ValueError("Mortar crosspoint trace routes are invalid.")
        if len(set(excluded)) != len(excluded) or any(
            value < 0 or value >= multiplier_size_ for value in excluded
        ):
            raise ValueError("Mortar excluded multiplier routes are invalid.")
        if crosspoints and not excluded:
            raise ValueError(
                "Mortar crosspoints require explicit multiplier elimination."
            )
        retained = np.asarray(
            [value for value in range(multiplier_size_) if value not in set(excluded)],
            dtype=np.int32,
        )
        if retained.size == 0:
            raise ValueError("Mortar crosspoint elimination removed every multiplier.")
        self.primal_trace_size = primal_size
        self.multiplier_size = multiplier_size_
        self.crosspoint_trace_dofs = crosspoints
        self.excluded_multiplier_dofs = excluded
        self.retained_multiplier_dofs = jnp.asarray(retained)
        self.owner_patch_id = owner
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-mortar-crosspoint-plan",
                "primal_trace_size": primal_size,
                "multiplier_size": multiplier_size_,
                "crosspoint_trace_dofs": list(crosspoints),
                "excluded_multiplier_dofs": list(excluded),
                "owner_patch_id": owner,
                "policy": "single-owner",
            }
        )


class MortarInfSupCertificate(StrictModule, NonTrainableState):
    """Discrete normalized mortar inf-sup evidence after crosspoint elimination."""

    interface_certificate: InterfaceCertificate
    crosspoint_plan: MortarCrosspointPlan
    minimum_singular_value: float = eqx.field(static=True)
    required_lower_bound: float = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


def certify_mortar_inf_sup(
    interface_certificate: InterfaceCertificate,
    normalized_coupling: ArrayLike,
    crosspoint_plan: MortarCrosspointPlan,
    /,
    *,
    required_lower_bound: float,
) -> MortarInfSupCertificate:
    if not isinstance(interface_certificate, InterfaceCertificate):
        raise TypeError("Mortar stability requires an InterfaceCertificate.")
    if not isinstance(crosspoint_plan, MortarCrosspointPlan):
        raise TypeError("Mortar stability requires a MortarCrosspointPlan.")
    lower_bound = _positive_finite("required_lower_bound", required_lower_bound)
    coupling = np.asarray(normalized_coupling, dtype=float)
    expected_shape = (
        crosspoint_plan.multiplier_size,
        crosspoint_plan.primal_trace_size,
    )
    if coupling.shape != expected_shape or np.any(~np.isfinite(coupling)):
        raise ValueError("Normalized mortar coupling has an invalid shape or value.")
    retained = np.asarray(crosspoint_plan.retained_multiplier_dofs)
    reduced_coupling = coupling[retained]
    if reduced_coupling.shape[0] > reduced_coupling.shape[1]:
        raise ValueError("Mortar multiplier trace is overconstrained after elimination.")
    singular_values = np.linalg.svd(reduced_coupling, compute_uv=False)
    minimum = float(singular_values[-1])
    maximum = float(singular_values[0])
    if not np.isfinite(minimum) or minimum < lower_bound:
        raise ValueError("Mortar inf-sup stability gate failed.")
    condition = maximum / minimum
    return MortarInfSupCertificate(
        interface_certificate,
        crosspoint_plan,
        minimum,
        lower_bound,
        condition,
        canonical_fingerprint(
            {
                "kind": "iga-mortar-inf-sup-certificate",
                "interface_certificate": interface_certificate.certificate_id,
                "crosspoint_plan": crosspoint_plan.plan_id,
                "coupling": array_tree_fingerprint(coupling),
                "minimum_singular_value": minimum,
                "required_lower_bound": lower_bound,
                "condition_number": condition,
            }
        ),
    )


class MortarProviderPayload(StrictModule, NonTrainableState):
    """Stable mortar coupling routed into the existing coupling/assembly stack."""

    normalized_coupling: Array
    retained_multiplier_dofs: Array
    interface_id: str = eqx.field(static=True)
    inf_sup_certificate_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class MortarInterfacePlan(StrictModule, NonTrainableState):
    """One certified mortar route with an explicit single-owner crosspoint policy."""

    interface: PatchInterface
    certificate: InterfaceCertificate
    normalized_coupling: Array
    stability: MortarInfSupCertificate
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface: PatchInterface,
        certificate: InterfaceCertificate,
        normalized_coupling: ArrayLike,
        stability: MortarInfSupCertificate,
        /,
    ):
        if not isinstance(interface, PatchInterface):
            raise TypeError("interface must be a PatchInterface.")
        if not isinstance(certificate, InterfaceCertificate):
            raise TypeError("Mortar lowering requires an InterfaceCertificate.")
        if not isinstance(stability, MortarInfSupCertificate):
            raise TypeError("Mortar lowering requires an inf-sup stability gate.")
        certificate.assert_matches(interface)
        if stability.interface_certificate.certificate_id != certificate.certificate_id:
            raise ValueError("Mortar inf-sup evidence belongs to another interface.")
        coupling = jnp.asarray(normalized_coupling)
        expected_shape = (
            stability.crosspoint_plan.multiplier_size,
            stability.crosspoint_plan.primal_trace_size,
        )
        if coupling.shape != expected_shape:
            raise ValueError("Mortar coupling does not match its crosspoint plan.")
        self.interface = interface
        self.certificate = certificate
        self.normalized_coupling = coupling
        self.stability = stability
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-mortar-interface-plan",
                "interface": interface.interface_id,
                "certificate": certificate.certificate_id,
                "stability": stability.certificate_id,
                "coupling": array_tree_fingerprint(np.asarray(coupling)),
            }
        )

    def provider_payload(self, /) -> MortarProviderPayload:
        return MortarProviderPayload(
            self.normalized_coupling,
            self.stability.crosspoint_plan.retained_multiplier_dofs,
            self.interface.interface_id,
            self.stability.certificate_id,
            canonical_fingerprint(
                {"kind": "iga-mortar-provider-payload", "plan": self.plan_id}
            ),
        )


__all__ = [
    "H1NitscheCoercivityCertificate",
    "H1NitscheInterfacePlan",
    "H1NitscheProviderPayload",
    "InterfaceCertificate",
    "InterfaceParameterMap",
    "InterfaceQualificationEvidence",
    "MortarCrosspointPlan",
    "MortarInfSupCertificate",
    "MortarInterfacePlan",
    "MortarProviderPayload",
    "PatchInterface",
    "PeriodicSelfInterface",
    "TraceRoute",
    "PeriodicCompatibilityAdapter",
    "PeriodicCompatibilityPayload",
    "certify_h1_nitsche_coercivity",
    "certify_mortar_inf_sup",
    "certify_patch_interface",
]
