#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import EntitySet
from ....geometry import MeshRegion
from ....integration import IntegrationPrecisionPolicy
from ....linalg import (
    AbstractLinearOperator,
    DenseLinearOperator,
    DualSpace,
    LinearCapabilityError,
    MaterializationPolicy,
    OperatorCapabilities,
    OperatorProperties,
)
from ....linalg._operators import _AbstractCostedLinearOperator
from ._galerkin_quadrature3d import (
    _class_workspace_byte_estimates,
    _PAIR_CLASS_NAMES,
    _preparation_workspace_byte_estimate,
    _prepare_surface_pairs_3d,
    _resident_byte_estimate,
    _SurfacePairData3D,
)
from ._laplace3d import LaplaceLayerPotential3D
from ._surface3d import SurfacePanelization3D
from ._surface_fem3d import _SurfaceFEMBinding3D


class LaplaceSingleLayerDP0GalerkinPolicy3D(StrictModule, NonTrainableState):
    """Accuracy, resource, and blocking policy for 3D Laplace DP0 Galerkin."""

    regular_order: int = eqx.field(static=True)
    singular_order: int = eqx.field(static=True)
    near_order: int = eqx.field(static=True)
    near_ratio: float = eqx.field(static=True)
    near_max_depth: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    target_block_size: int = eqx.field(static=True)
    source_block_size: int = eqx.field(static=True)
    max_exception_pairs: int = eqx.field(static=True)
    max_preparation_workspace_bytes: int = eqx.field(static=True)
    max_resident_bytes: int = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy
    dense_oracle: MaterializationPolicy | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regular_order: int = 4,
        singular_order: int = 5,
        near_order: int = 4,
        near_ratio: float = 2.0,
        near_max_depth: int = 2,
        absolute_tolerance: float = 1.0e-6,
        relative_tolerance: float = 1.0e-4,
        target_block_size: int = 16,
        source_block_size: int = 16,
        max_exception_pairs: int = 100_000,
        max_preparation_workspace_bytes: int = 256 * 1024 * 1024,
        max_resident_bytes: int = 256 * 1024 * 1024,
        precision: IntegrationPrecisionPolicy | None = None,
        dense_oracle: MaterializationPolicy | None = None,
    ):
        orders = tuple(
            int(value) for value in (regular_order, singular_order, near_order)
        )
        if any(value < 2 for value in orders):
            raise ValueError("Galerkin quadrature orders must be at least two.")
        ratio = float(near_ratio)
        depth = int(near_max_depth)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        blocks = (int(target_block_size), int(source_block_size))
        limits = (
            int(max_exception_pairs),
            int(max_preparation_workspace_bytes),
            int(max_resident_bytes),
        )
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("near_ratio must be finite and positive.")
        if depth < 0:
            raise ValueError("near_max_depth must be nonnegative.")
        if any(not math.isfinite(value) or value < 0.0 for value in (absolute, relative)):
            raise ValueError("Galerkin tolerances must be finite and nonnegative.")
        if any(value <= 0 for value in blocks + limits):
            raise ValueError("Galerkin block and resource limits must be positive.")
        precision_ = IntegrationPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, IntegrationPrecisionPolicy):
            raise TypeError("precision must be IntegrationPrecisionPolicy or None.")
        if dense_oracle is not None and not isinstance(
            dense_oracle, MaterializationPolicy
        ):
            raise TypeError("dense_oracle must be MaterializationPolicy or None.")
        self.regular_order, self.singular_order, self.near_order = orders
        self.near_ratio = ratio
        self.near_max_depth = depth
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.target_block_size, self.source_block_size = blocks
        (
            self.max_exception_pairs,
            self.max_preparation_workspace_bytes,
            self.max_resident_bytes,
        ) = limits
        self.precision = precision_
        self.dense_oracle = dense_oracle
        self.policy_id = canonical_fingerprint(
            {
                "kind": "laplace-single-layer-dp0-galerkin-policy-3d-v1",
                "orders": orders,
                "near_ratio": ratio,
                "near_max_depth": depth,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "blocks": blocks,
                "limits": limits,
                "precision": precision_.policy_id,
                "dense_oracle": (
                    None
                    if dense_oracle is None
                    else {
                        "entries": dense_oracle.max_entries,
                        "bytes": dense_oracle.max_bytes,
                    }
                ),
            }
        )


class LaplaceSingleLayerDP0AssemblyReport3D(StrictModule, NonTrainableState):
    """Pair coverage, quadrature error, and resource evidence for weak V."""

    binding_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    pair_class_names: tuple[str, str, str, str, str] = eqx.field(static=True)
    pair_counts: tuple[int, int, int, int, int] = eqx.field(static=True)
    exception_count: int = eqx.field(static=True)
    maximum_errors: Array
    pair_class_tolerances: Array
    pair_class_supported: Array
    evaluations: Array
    pair_class_workspace_bytes: tuple[int, int, int, int, int] = eqx.field(static=True)
    pair_class_resident_bytes: tuple[int, int, int, int, int] = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    dense_oracle_available: bool = eqx.field(static=True)
    dense_oracle_bytes: int = eqx.field(static=True)
    materializable: bool = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    finite: Array
    accuracy_supported: Array
    report_id: str = eqx.field(static=True)


class _LaplaceDP0WeakOperator3D(_AbstractCostedLinearOperator):
    pair_data: _SurfacePairData3D
    target_block_size: int = eqx.field(static=True)
    source_block_size: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        pair_data: _SurfacePairData3D,
        space,
        /,
        *,
        target_block_size: int,
        source_block_size: int,
        operator_id: str,
    ):
        self.pair_data = pair_data
        self.target_block_size = int(target_block_size)
        self.source_block_size = int(source_block_size)
        self.face_count = int(space.size)
        quadrature_count = int(pair_data.regular_points.shape[1])
        itemsize = np.dtype(pair_data.regular_points.dtype).itemsize
        self.action_workspace_bytes = int(
            itemsize
            * (
                self.target_block_size * quadrature_count * 4
                + self.source_block_size * quadrature_count * 4
                + self.target_block_size
                * self.source_block_size
                * quadrature_count
                * quadrature_count
                * 4
                + 3 * self.target_block_size * self.source_block_size
            )
        )
        self.source = space
        self.target = DualSpace(space)
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def _regular_action(self, vector: Array, /, *, transpose: bool) -> Array:
        face_count = self.face_count
        target_block = self.target_block_size
        source_block = self.source_block_size
        points = self.pair_data.regular_points
        weights = self.pair_data.regular_weights
        keys = self.pair_data.exception_keys
        output = jnp.zeros((face_count,), dtype=vector.dtype)
        for target_start in range(0, face_count, target_block):
            target_ids = jnp.arange(target_start, target_start + target_block)
            target_valid = target_ids < face_count
            safe_targets = jnp.minimum(target_ids, face_count - 1)
            target_points = points[safe_targets]
            target_weights = weights[safe_targets]
            if not transpose:
                accumulator = jnp.zeros((target_block,), dtype=vector.dtype)
            for source_start in range(0, face_count, source_block):
                source_ids = jnp.arange(source_start, source_start + source_block)
                source_valid = source_ids < face_count
                safe_sources = jnp.minimum(source_ids, face_count - 1)
                source_points = points[safe_sources]
                source_weights = weights[safe_sources]
                pair_keys = target_ids[:, None] * face_count + source_ids[None, :]
                positions = jnp.searchsorted(keys, pair_keys)
                safe_positions = jnp.minimum(positions, keys.shape[0] - 1)
                is_exception = (positions < keys.shape[0]) & (
                    keys[safe_positions] == pair_keys
                )
                active = target_valid[:, None] & source_valid[None, :] & ~is_exception
                differences = (
                    target_points[:, None, :, None, :]
                    - source_points[None, :, None, :, :]
                )
                safe_differences = jnp.where(
                    active[:, :, None, None, None],
                    differences,
                    jnp.ones_like(differences),
                )
                kernel = 1.0 / (4.0 * jnp.pi * jnp.linalg.norm(safe_differences, axis=-1))
                pair_matrix = ein.contract(
                    "tq,sr,tsqr->ts",
                    target_weights,
                    source_weights,
                    kernel,
                    backend="jax",
                )
                pair_matrix = jnp.where(active, pair_matrix, 0.0)
                if transpose:
                    contribution = ein.contract(
                        "ts,t->s",
                        pair_matrix,
                        vector[safe_targets] * target_valid,
                        backend="jax",
                    )
                    output = output.at[safe_sources].add(contribution * source_valid)
                else:
                    accumulator = accumulator + ein.contract(
                        "ts,s->t",
                        pair_matrix,
                        vector[safe_sources] * source_valid,
                        backend="jax",
                    )
            if not transpose:
                output = output.at[safe_targets].add(accumulator * target_valid)
        return output

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        output = self._regular_action(value, transpose=False)
        correction = self.pair_data.values * value[self.pair_data.sources]
        return self.target.validate(output.at[self.pair_data.targets].add(correction))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        output = self._regular_action(value, transpose=True)
        correction = self.pair_data.values * value[self.pair_data.targets]
        return self.source.validate(output.at[self.pair_data.sources].add(correction))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.transpose_mv(vector)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError("Blocked Galerkin operators cannot materialize.")

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return self.action_workspace_bytes, "blocked-surface-galerkin-action"


class _LaplaceDP0StrongOperator3D(_AbstractCostedLinearOperator):
    weak: _LaplaceDP0WeakOperator3D
    inverse_areas: Array
    diagonal: Array

    def __init__(
        self,
        weak: _LaplaceDP0WeakOperator3D,
        areas: Array,
        diagonal: Array,
        /,
        *,
        operator_id: str,
    ):
        self.weak = weak
        self.inverse_areas = jnp.reciprocal(jnp.asarray(areas))
        self.diagonal = jnp.asarray(diagonal)
        self.source = weak.source
        self.target = weak.source
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.target.validate(self.weak.mv(vector) * self.inverse_areas)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(self.weak.transpose_mv(value * self.inverse_areas))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.transpose_mv(vector)

    def _assemble_diagonal(self, /) -> Array:
        return self.diagonal

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError("Blocked Galerkin operators cannot materialize.")

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        vector_bytes = self.source.size * np.dtype(self.inverse_areas.dtype).itemsize
        return (
            self.weak.action_workspace_bytes + int(vector_bytes),
            "strong-blocked-surface-galerkin-action",
        )


class LaplaceSingleLayerDP0Galerkin3D(StrictModule, NonTrainableState):
    """Prepared weak/strong DP0 Galerkin operators and field reconstruction."""

    weak_operator: AbstractLinearOperator
    strong_operator: AbstractLinearOperator
    dense_oracle: DenseLinearOperator | None
    panelization: SurfacePanelization3D
    surface_entities: EntitySet
    assembly_report: LaplaceSingleLayerDP0AssemblyReport3D
    _binding: _SurfaceFEMBinding3D

    @property
    def face_count(self) -> int:
        return self._binding.face_count

    @property
    def component_count(self) -> int:
        return self._binding.component_count

    @property
    def face_component_ids(self) -> Array:
        return self._binding.face_component_ids

    @property
    def face_areas(self) -> Array:
        return self._binding.face_areas

    def potential(self, coefficients: ArrayLike, /) -> LaplaceLayerPotential3D:
        values = self.strong_operator.source.validate(coefficients)
        node_density = jnp.repeat(values, self.panelization.nodes_per_panel)
        return LaplaceLayerPotential3D(
            self.panelization,
            kind="single",
            density=node_density,
        )


def prepare_laplace_single_layer_dp0_3d(
    region: MeshRegion,
    /,
    *,
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
    numeric_version: str = "0",
) -> LaplaceSingleLayerDP0Galerkin3D:
    """Prepare closed-surface DP0 weak and strong Laplace single-layer maps."""
    selected = LaplaceSingleLayerDP0GalerkinPolicy3D() if policy is None else policy
    if not isinstance(selected, LaplaceSingleLayerDP0GalerkinPolicy3D):
        raise TypeError("policy must be LaplaceSingleLayerDP0GalerkinPolicy3D or None.")
    if not isinstance(region, MeshRegion):
        raise TypeError("[geometry] 3D Galerkin preparation requires a MeshRegion.")
    triangle_mesh = region.triangle_mesh
    face_count = int(triangle_mesh.faces.shape[0])
    dense_bytes = 0
    if selected.dense_oracle is not None:
        entries = face_count * face_count
        dense_bytes = entries * np.dtype(jnp.float64).itemsize
        if entries > selected.dense_oracle.max_entries:
            raise LinearCapabilityError(
                "[dense-oracle-entries] Requested Galerkin dense oracle exceeds "
                "max_entries."
            )
        if dense_bytes > selected.dense_oracle.max_bytes:
            raise LinearCapabilityError(
                "[dense-oracle-bytes] Requested Galerkin dense oracle exceeds max_bytes."
            )
    if face_count > selected.max_exception_pairs:
        raise ValueError(
            "[exception-capacity] Surface pair exceptions exceed max_exception_pairs."
        )
    class_workspace_bytes, regular_point_count = _class_workspace_byte_estimates(
        selected.regular_order,
        selected.singular_order,
        selected.near_order,
    )
    minimum_workspace = _preparation_workspace_byte_estimate(
        face_count,
        face_count,
        class_workspace_bytes,
    )
    if minimum_workspace > selected.max_preparation_workspace_bytes:
        raise ValueError(
            "[preparation-bytes] Surface pair preparation exceeds its "
            "workspace-byte budget."
        )
    minimum_resident = _resident_byte_estimate(
        face_count,
        face_count,
        regular_point_count,
    )
    if minimum_resident > selected.max_resident_bytes:
        raise ValueError(
            "[resident-bytes] Surface pair state exceeds its resident-byte budget."
        )
    panel_order = max(
        selected.regular_order,
        selected.singular_order,
        selected.near_order,
    )
    binding = _SurfaceFEMBinding3D(
        region,
        quadrature_order=panel_order,
        numeric_version=numeric_version,
    )
    pair_data = _prepare_surface_pairs_3d(
        triangle_mesh.vertices,
        triangle_mesh.faces,
        regular_order=selected.regular_order,
        singular_order=selected.singular_order,
        near_order=selected.near_order,
        near_ratio=selected.near_ratio,
        near_max_depth=selected.near_max_depth,
        absolute_tolerance=selected.absolute_tolerance,
        relative_tolerance=selected.relative_tolerance,
        max_exception_pairs=selected.max_exception_pairs,
        max_preparation_workspace_bytes=selected.max_preparation_workspace_bytes,
        max_resident_bytes=selected.max_resident_bytes,
    )
    pair_data = eqx.tree_at(
        lambda data: (
            data.values,
            data.regular_points,
            data.regular_weights,
            data.maximum_errors,
            data.maximum_tolerances,
        ),
        pair_data,
        (
            selected.precision.accumulation(pair_data.values),
            selected.precision.evaluation(pair_data.regular_points),
            selected.precision.accumulation(pair_data.regular_weights),
            selected.precision.decision(pair_data.maximum_errors),
            selected.precision.decision(pair_data.maximum_tolerances),
        ),
    )
    class_counts = jnp.asarray(pair_data.counts)
    stored_supported = (
        jnp.isfinite(pair_data.maximum_errors)
        & jnp.isfinite(pair_data.maximum_tolerances)
        & (
            (class_counts == 0)
            | (pair_data.maximum_errors <= pair_data.maximum_tolerances)
        )
    )
    pair_data = eqx.tree_at(
        lambda data: data.supported,
        pair_data,
        stored_supported,
    )
    space = binding.discretization.field_spaces[0].vector_space
    weak_id = canonical_fingerprint(
        {
            "kind": "laplace-single-layer-dp0-weak-3d-v1",
            "binding": binding.binding_id,
            "policy": selected.policy_id,
        }
    )
    weak = _LaplaceDP0WeakOperator3D(
        pair_data,
        space,
        target_block_size=selected.target_block_size,
        source_block_size=selected.source_block_size,
        operator_id=weak_id,
    )
    self_mask = pair_data.targets == pair_data.sources
    weak_diagonal = (
        jnp.zeros((binding.face_count,))
        .at[pair_data.targets[self_mask]]
        .set(pair_data.values[self_mask])
    )
    strong_diagonal = weak_diagonal / binding.face_areas
    strong = _LaplaceDP0StrongOperator3D(
        weak,
        binding.face_areas,
        strong_diagonal,
        operator_id=canonical_fingerprint(
            {
                "kind": "laplace-single-layer-dp0-strong-3d-v1",
                "weak": weak.operator_id,
                "gram": array_tree_fingerprint(binding.face_areas),
            }
        ),
    )

    dense_oracle = None
    if selected.dense_oracle is not None:
        basis = jnp.eye(face_count, dtype=jnp.float64)
        weak_matrix = jax.vmap(weak.mv, in_axes=1, out_axes=1)(basis)
        dense_oracle = DenseLinearOperator(
            weak_matrix * jnp.reciprocal(binding.face_areas)[:, None],
            source=space,
            target=space,
        )

    finite = (
        jnp.all(jnp.isfinite(pair_data.values))
        & jnp.all(jnp.isfinite(binding.face_areas))
        & jnp.all(binding.face_areas > 0.0)
    )
    report = LaplaceSingleLayerDP0AssemblyReport3D(
        binding_id=binding.binding_id,
        policy_id=selected.policy_id,
        kernel_id="laplace-single-layer-3d",
        numeric_version=binding.numeric_version,
        face_count=binding.face_count,
        component_count=binding.component_count,
        pair_counts=pair_data.counts,
        pair_class_names=_PAIR_CLASS_NAMES,
        exception_count=int(pair_data.targets.shape[0]),
        maximum_errors=pair_data.maximum_errors,
        pair_class_tolerances=pair_data.maximum_tolerances,
        pair_class_supported=pair_data.supported,
        evaluations=pair_data.evaluations,
        pair_class_workspace_bytes=pair_data.class_workspace_bytes,
        pair_class_resident_bytes=pair_data.class_resident_bytes,
        preparation_workspace_bytes=pair_data.preparation_workspace_bytes,
        resident_bytes=pair_data.resident_bytes,
        action_workspace_bytes_per_rhs=strong._action_workspace_cost()[0],
        dense_oracle_available=dense_oracle is not None,
        dense_oracle_bytes=dense_bytes,
        materializable=False,
        continuum_discretization_error_estimated=False,
        finite=finite,
        accuracy_supported=finite & jnp.all(pair_data.supported),
        report_id=canonical_fingerprint(
            {
                "kind": "laplace-single-layer-dp0-assembly-report-3d-v1",
                "binding": binding.binding_id,
                "policy": selected.policy_id,
                "pair_counts": pair_data.counts,
                "errors": array_tree_fingerprint(pair_data.maximum_errors),
                "tolerances": array_tree_fingerprint(pair_data.maximum_tolerances),
                "supported": array_tree_fingerprint(pair_data.supported),
                "class_workspace_bytes": pair_data.class_workspace_bytes,
                "class_resident_bytes": pair_data.class_resident_bytes,
            }
        ),
    )
    return LaplaceSingleLayerDP0Galerkin3D(
        weak_operator=weak,
        strong_operator=strong,
        dense_oracle=dense_oracle,
        panelization=binding.panelization,
        surface_entities=binding.surface_entities,
        assembly_report=report,
        _binding=binding,
    )


__all__ = [
    "LaplaceSingleLayerDP0AssemblyReport3D",
    "LaplaceSingleLayerDP0Galerkin3D",
    "LaplaceSingleLayerDP0GalerkinPolicy3D",
    "prepare_laplace_single_layer_dp0_3d",
]
