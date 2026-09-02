#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    AbstractLinearOperator,
    estimate_operator_action_cost,
    LinearCapabilityError,
)
from ._fmm2d import LaplaceFMMBackend2D
from ._galerkin3d import (
    _LaplaceDP0StrongOperator3D,
    _LaplaceDP0WeakOperator3D,
    LaplaceSingleLayerDP0Galerkin3D,
)
from ._laplace2d import LaplaceLayerPotential2D


BoundaryFastProviderName: TypeAlias = Literal[
    "blocked-direct-dp0-galerkin-3d",
    "laplace-fmm-2d",
    "laplace-fmm-3d",
    "scalar-h-matrix-3d",
    "scalar-h2-matrix-3d",
]
BoundaryGalerkinFormulation: TypeAlias = Literal["weak", "strong"]


class BEMFastCapabilityError(LinearCapabilityError):
    """Raised when a requested BEM acceleration is not genuinely implemented."""


class BEMBlockActionStatus(IntEnum):
    """Per-right-hand-side status for one fused blocked action."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONFINITE_OUTPUT = 2


class BEMExecutionEnvelope(StrictModule, NonTrainableState):
    """Declared scientific and computational envelope for one BEM action.

    The envelope is evidence about a prepared discrete computation. It is not a
    continuum error certificate and does not claim geometry repair or CAD support.
    """

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: tuple[str, ...] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    accelerated: bool = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ambient_dimension: int,
        pde: str,
        geometry: str,
        formulation: str,
        provider: str,
        precision: str,
        resource_evidence: tuple[str, ...],
        error_evidence: tuple[str, ...],
        non_goals: tuple[str, ...],
        accelerated: bool,
    ):
        dimension = int(ambient_dimension)
        strings = tuple(
            str(value).strip()
            for value in (pde, geometry, formulation, provider, precision)
        )
        resources = tuple(str(value).strip() for value in resource_evidence)
        errors = tuple(str(value).strip() for value in error_evidence)
        excluded = tuple(str(value).strip() for value in non_goals)
        if dimension not in (2, 3):
            raise ValueError("The BEM execution envelope supports only 2D or 3D.")
        if (
            not resources
            or not errors
            or not excluded
            or any(not value for value in strings + resources + errors + excluded)
        ):
            raise ValueError("BEM execution-envelope declarations must be non-empty.")
        self.ambient_dimension = dimension
        self.pde, self.geometry, self.formulation, self.provider, self.precision = strings
        self.resource_evidence = resources
        self.error_evidence = errors
        self.non_goals = excluded
        self.continuum_certified = False
        self.accelerated = bool(accelerated)
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "bem-execution-envelope",
                "ambient_dimension": dimension,
                "pde": self.pde,
                "geometry": self.geometry,
                "formulation": self.formulation,
                "provider": self.provider,
                "precision": self.precision,
                "resource_evidence": list(resources),
                "error_evidence": list(errors),
                "non_goals": list(excluded),
                "continuum_certified": False,
                "accelerated": self.accelerated,
            }
        )


class BEMFastProviderCapabilities(StrictModule, NonTrainableState):
    """Truthful immutable capabilities of one concrete layer-potential provider."""

    name: BoundaryFastProviderName = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    accelerated: bool = eqx.field(static=True)
    exact_prepared_near: bool = eqx.field(static=True)
    exact_transpose: bool = eqx.field(static=True)
    multiple_rhs: bool = eqx.field(static=True)
    arbitrary_targets: bool = eqx.field(static=True)


_BOUNDARY_FAST_PROVIDER_CATALOG = (
    BEMFastProviderCapabilities(
        name="blocked-direct-dp0-galerkin-3d",
        ambient_dimension=3,
        pde="laplace",
        formulation="dp0-galerkin-single-layer",
        accelerated=False,
        exact_prepared_near=True,
        exact_transpose=True,
        multiple_rhs=True,
        arbitrary_targets=False,
    ),
    BEMFastProviderCapabilities(
        name="laplace-fmm-2d",
        ambient_dimension=2,
        pde="laplace",
        formulation="quadrature-source-potential-evaluation",
        accelerated=True,
        exact_prepared_near=True,
        exact_transpose=False,
        multiple_rhs=False,
        arbitrary_targets=True,
    ),
    BEMFastProviderCapabilities(
        name="laplace-fmm-3d",
        ambient_dimension=3,
        pde="laplace",
        formulation="dp0-galerkin-single-layer",
        accelerated=True,
        exact_prepared_near=True,
        exact_transpose=True,
        multiple_rhs=True,
        arbitrary_targets=False,
    ),
    BEMFastProviderCapabilities(
        name="scalar-h-matrix-3d",
        ambient_dimension=3,
        pde="laplace",
        formulation="dp0-galerkin-single-layer",
        accelerated=True,
        exact_prepared_near=True,
        exact_transpose=True,
        multiple_rhs=True,
        arbitrary_targets=False,
    ),
    BEMFastProviderCapabilities(
        name="scalar-h2-matrix-3d",
        ambient_dimension=3,
        pde="laplace",
        formulation="dp0-galerkin-single-layer",
        accelerated=True,
        exact_prepared_near=True,
        exact_transpose=True,
        multiple_rhs=True,
        arbitrary_targets=False,
    ),
)


def boundary_fast_provider_capabilities(
    name: str,
    /,
    *,
    ambient_dimension: int | None = None,
    require_acceleration: bool = False,
) -> BEMFastProviderCapabilities:
    """Return a truthful provider declaration or reject the unsupported request."""
    identifier = str(name).strip()
    dimension = None if ambient_dimension is None else int(ambient_dimension)
    for capabilities in _BOUNDARY_FAST_PROVIDER_CATALOG:
        if capabilities.name != identifier:
            continue
        if dimension is not None and dimension != capabilities.ambient_dimension:
            raise BEMFastCapabilityError(
                f"Provider {identifier!r} is {capabilities.ambient_dimension}D, not "
                f"{dimension}D."
            )
        if require_acceleration and not capabilities.accelerated:
            raise BEMFastCapabilityError(
                f"Provider {identifier!r} is a blocked direct action, not an accelerator."
            )
        return capabilities
    if dimension == 3 or any(
        token in identifier.lower() for token in ("fmm", "h2", "h²")
    ):
        raise BEMFastCapabilityError(
            f"No catalogued 3D fast provider matches {identifier!r}."
        )
    raise BEMFastCapabilityError(f"Unknown BEM fast provider {identifier!r}.")


def prepare_laplace_fmm_provider_2d(
    potential: LaplaceLayerPotential2D,
    /,
    *,
    expansion_order: int = 8,
    leaf_size: int = 32,
    opening_angle: float = 0.5,
) -> LaplaceFMMBackend2D:
    """Prepare the existing real 2D Laplace FMM within its declared envelope."""
    boundary_fast_provider_capabilities(
        "laplace-fmm-2d", ambient_dimension=2, require_acceleration=True
    )
    return LaplaceFMMBackend2D(
        potential,
        expansion_order=expansion_order,
        leaf_size=leaf_size,
        opening_angle=opening_angle,
    )


class FusedBEMBlockActionResult(StrictModule, NonTrainableState):
    """Fused DP0 Galerkin columns with status and bounded-envelope evidence."""

    values: Array
    column_status: Array
    finite: Array
    envelope: BEMExecutionEnvelope
    transpose: bool = eqx.field(static=True)
    action_id: str = eqx.field(static=True)


class FusedBlockedBEMAction3D(StrictModule, NonTrainableState):
    """Fixed-column fused action for the current 3D Laplace DP0 Galerkin map.

    ``apply`` uses one ``jax.vmap`` transformation over the prepared blocked
    operator. It therefore batches every source/target block contraction rather
    than invoking one Python action per column. Forward and algebraic transpose
    call the exact same prepared quadrature and exception-pair routes as the
    serial operator.
    """

    operator: AbstractLinearOperator
    rhs_count: int = eqx.field(static=True)
    formulation: BoundaryGalerkinFormulation = eqx.field(static=True)
    envelope: BEMExecutionEnvelope
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: LaplaceSingleLayerDP0Galerkin3D,
        rhs_count: int,
        /,
        *,
        formulation: BoundaryGalerkinFormulation = "strong",
        provider: str = "blocked-direct-dp0-galerkin-3d",
    ):
        if not isinstance(prepared, LaplaceSingleLayerDP0Galerkin3D):
            raise TypeError("Fused blocked action requires prepared 3D Laplace DP0 BEM.")
        if not bool(prepared.assembly_report.accuracy_supported):
            raise BEMFastCapabilityError(
                "Fused blocked action requires every prepared pair class to satisfy "
                "its declared quadrature tolerance."
            )
        columns = int(rhs_count)
        if columns <= 0:
            raise ValueError("rhs_count must be positive and fixed at preparation.")
        if formulation not in ("weak", "strong"):
            raise ValueError("formulation must be 'weak' or 'strong'.")
        capabilities = boundary_fast_provider_capabilities(provider, ambient_dimension=3)
        if capabilities.name != "blocked-direct-dp0-galerkin-3d":
            raise BEMFastCapabilityError(
                "The fused 3D route only has a blocked direct provider."
            )
        operator = (
            prepared.weak_operator if formulation == "weak" else prepared.strong_operator
        )
        if formulation == "weak" and not isinstance(operator, _LaplaceDP0WeakOperator3D):
            raise TypeError(
                "Prepared weak operator is not the supported blocked DP0 route."
            )
        if formulation == "strong" and not isinstance(
            operator, _LaplaceDP0StrongOperator3D
        ):
            raise TypeError(
                "Prepared strong operator is not the supported blocked DP0 route."
            )
        pair_data = (
            operator.pair_data
            if isinstance(operator, _LaplaceDP0WeakOperator3D)
            else operator.weak.pair_data
        )
        precision = np.dtype(pair_data.regular_points.dtype).name
        action_cost = estimate_operator_action_cost(operator)
        workspace = action_cost.apply_workspace_bytes_per_rhs * columns
        envelope = BEMExecutionEnvelope(
            ambient_dimension=3,
            pde="laplace",
            geometry="closed-oriented-triangular-surface",
            formulation=f"dp0-galerkin-single-layer-{formulation}",
            provider=capabilities.name,
            precision=precision,
            resource_evidence=(
                f"fixed-rhs-count={columns}",
                f"conservative-fused-workspace-bytes={workspace}",
                f"prepared-resident-bytes={pair_data.resident_bytes}",
            ),
            error_evidence=(
                "prepared-pair-quadrature-error-estimates",
                "finite-column-status",
            ),
            non_goals=(
                "continuum-discretization-certification",
                "3d-fmm-or-h-matrix-acceleration",
                "geometry-repair-or-cad",
            ),
            accelerated=False,
        )
        self.operator = operator
        self.rhs_count = columns
        self.formulation = formulation
        self.envelope = envelope
        self.action_id = canonical_fingerprint(
            {
                "kind": "fused-blocked-bem-action-3d",
                "operator": operator.operator_id,
                "rhs_count": columns,
                "formulation": formulation,
                "envelope": envelope.envelope_id,
            }
        )

    def _apply(self, right_hand_side: ArrayLike, /, *, transpose: bool) -> Array:
        values = jnp.asarray(right_hand_side)
        expected_rows = (
            self.operator.target.size if transpose else self.operator.source.size
        )
        if values.ndim != 2 or values.shape != (expected_rows, self.rhs_count):
            raise ValueError(
                "Fused BEM right-hand side has incompatible fixed row/column shape."
            )
        finite_input = jnp.all(jnp.isfinite(values), axis=0)
        safe_values = jnp.where(finite_input[None, :], values, jnp.zeros_like(values))
        action = self.operator.transpose_mv if transpose else self.operator.mv
        computed = jax.vmap(action, in_axes=1, out_axes=1)(safe_values)
        finite_output = jnp.all(jnp.isfinite(computed), axis=0)
        successful = finite_input & finite_output
        failed_values = jnp.full_like(computed, jnp.nan)
        return jnp.where(successful[None, :], computed, failed_values)

    def apply(
        self, right_hand_side: ArrayLike, /, *, transpose: bool = False
    ) -> FusedBEMBlockActionResult:
        """Apply the exact prepared forward or transpose route to fixed columns."""
        values = jnp.asarray(right_hand_side)
        expected_rows = (
            self.operator.target.size if transpose else self.operator.source.size
        )
        if values.ndim != 2 or values.shape != (expected_rows, self.rhs_count):
            raise ValueError(
                "Fused BEM right-hand side has incompatible fixed row/column shape."
            )
        finite_input = jnp.all(jnp.isfinite(values), axis=0)
        output = self._apply(values, transpose=transpose)
        finite_output = jnp.all(jnp.isfinite(output), axis=0)
        status = jnp.where(
            ~finite_input,
            int(BEMBlockActionStatus.NONFINITE_INPUT),
            jnp.where(
                finite_output,
                int(BEMBlockActionStatus.SUCCESS),
                int(BEMBlockActionStatus.NONFINITE_OUTPUT),
            ),
        ).astype(jnp.int32)
        return FusedBEMBlockActionResult(
            values=output,
            column_status=status,
            finite=finite_output,
            envelope=self.envelope,
            transpose=bool(transpose),
            action_id=self.action_id,
        )

    def transpose_apply(self, right_hand_side: ArrayLike, /) -> FusedBEMBlockActionResult:
        """Apply the exact algebraic transpose with per-column status."""
        return self.apply(right_hand_side, transpose=True)


class BEMLocalBlock3D(StrictModule, NonTrainableState):
    """One exact prepared-discrete local block and its quadrature evidence."""

    values: Array
    target_indices: Array
    source_indices: Array
    exact_near_mask: Array
    pair_classes: Array
    quadrature_error_bounds: Array
    accuracy_supported: Array
    envelope: BEMExecutionEnvelope
    provider_id: str = eqx.field(static=True)
    block_id: str = eqx.field(static=True)


class AbstractExactNearProvider3D(StrictModule, NonTrainableState):
    """Contract for bounded local extraction with exact prepared near entries."""

    envelope: BEMExecutionEnvelope
    provider_id: str
    max_block_entries: int
    max_block_workspace_bytes: int

    @abc.abstractmethod
    def local_block(
        self, target_indices: ArrayLike, source_indices: ArrayLike, /
    ) -> BEMLocalBlock3D:
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal(self, /) -> BEMLocalBlock3D:
        raise NotImplementedError


class LaplaceDP0ExactNearProvider3D(AbstractExactNearProvider3D):
    """Local blocks from the exact prepared DP0 near/exception pair table.

    Far entries use the prepared regular tensor quadrature. Coincident,
    edge-adjacent, vertex-adjacent, and geometrically near entries are replaced
    by their already-prepared singular/adaptive values. "Exact" therefore means
    exact parity with this prepared discrete operator, not analytic integration.
    """

    weak: _LaplaceDP0WeakOperator3D
    inverse_areas: Array
    formulation: BoundaryGalerkinFormulation = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    max_block_entries: int = eqx.field(static=True)
    max_block_workspace_bytes: int = eqx.field(static=True)
    envelope: BEMExecutionEnvelope
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: LaplaceSingleLayerDP0Galerkin3D,
        /,
        *,
        formulation: BoundaryGalerkinFormulation = "strong",
        max_block_entries: int = 1_000_000,
        max_block_workspace_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(prepared, LaplaceSingleLayerDP0Galerkin3D):
            raise TypeError("Exact-near provider requires prepared 3D Laplace DP0 BEM.")
        if formulation not in ("weak", "strong"):
            raise ValueError("formulation must be 'weak' or 'strong'.")
        limit = int(max_block_entries)
        workspace_limit = int(max_block_workspace_bytes)
        if limit <= 0 or workspace_limit <= 0:
            raise ValueError("Exact-near block resource limits must be positive.")
        weak = prepared.weak_operator
        if not isinstance(weak, _LaplaceDP0WeakOperator3D):
            raise TypeError(
                "Prepared BEM does not use the supported blocked weak operator."
            )
        pair_data = weak.pair_data
        face_count = weak.face_count
        diagonal_keys = np.arange(face_count, dtype=np.int64) * (face_count + 1)
        keys = np.asarray(pair_data.exception_keys, dtype=np.int64)
        positions = np.searchsorted(keys, diagonal_keys)
        if (
            keys.size == 0
            or np.any(positions >= keys.size)
            or np.any(keys[np.minimum(positions, keys.size - 1)] != diagonal_keys)
        ):
            raise ValueError(
                "Prepared pair data does not contain every singular diagonal."
            )
        precision = np.dtype(pair_data.regular_points.dtype).name
        envelope = BEMExecutionEnvelope(
            ambient_dimension=3,
            pde="laplace",
            geometry="closed-oriented-triangular-surface",
            formulation=f"dp0-galerkin-single-layer-{formulation}",
            provider="blocked-direct-prepared-pair-data",
            precision=precision,
            resource_evidence=(
                f"max-local-block-entries={limit}",
                f"max-local-block-workspace-bytes={workspace_limit}",
                f"prepared-resident-bytes={pair_data.resident_bytes}",
                f"prepared-exception-pairs={pair_data.exception_keys.size}",
            ),
            error_evidence=(
                "per-pair-class-maximum-quadrature-errors",
                "prepared-exception-table-membership",
            ),
            non_goals=(
                "continuum-discretization-certification",
                "analytic-exactness",
                "3d-fast-multipole-acceleration",
            ),
            accelerated=False,
        )
        self.weak = weak
        self.inverse_areas = jnp.reciprocal(prepared.face_areas)
        self.formulation = formulation
        self.face_count = face_count
        self.max_block_entries = limit
        self.max_block_workspace_bytes = workspace_limit
        self.envelope = envelope
        self.provider_id = canonical_fingerprint(
            {
                "kind": "laplace-dp0-exact-near-provider-3d",
                "operator": weak.operator_id,
                "formulation": formulation,
                "max_block_entries": limit,
                "max_block_workspace_bytes": workspace_limit,
                "envelope": envelope.envelope_id,
            }
        )

    def _validated_indices(self, values: ArrayLike, name: str, /) -> np.ndarray:
        indices = np.asarray(values, dtype=np.int64)
        if (
            indices.ndim != 1
            or indices.size == 0
            or np.unique(indices).size != indices.size
            or np.any(indices < 0)
            or np.any(indices >= self.face_count)
        ):
            raise ValueError(f"{name} must be nonempty unique in-range face indices.")
        return indices

    def local_block(
        self, target_indices: ArrayLike, source_indices: ArrayLike, /
    ) -> BEMLocalBlock3D:
        targets_host = self._validated_indices(target_indices, "target_indices")
        sources_host = self._validated_indices(source_indices, "source_indices")
        entries = int(targets_host.size * sources_host.size)
        if entries > self.max_block_entries:
            raise LinearCapabilityError(
                f"Local BEM block requires {entries} entries, exceeding the "
                f"declared limit {self.max_block_entries}."
            )
        quadrature_count = int(self.weak.pair_data.regular_points.shape[1])
        itemsize = np.dtype(self.weak.pair_data.regular_points.dtype).itemsize
        workspace = (
            itemsize
            * (
                12 * entries * quadrature_count * quadrature_count
                + 8 * entries
                + 3 * quadrature_count * (targets_host.size + sources_host.size)
            )
            + 16 * entries
        )
        if workspace > self.max_block_workspace_bytes:
            raise LinearCapabilityError(
                f"Local BEM block conservative workspace bound is {workspace} bytes, "
                f"exceeding the declared limit {self.max_block_workspace_bytes}."
            )
        targets = jnp.asarray(targets_host, dtype=jnp.int32)
        sources = jnp.asarray(sources_host, dtype=jnp.int32)
        pair_data = self.weak.pair_data
        keys = pair_data.exception_keys
        pair_keys = targets[:, None].astype(keys.dtype) * self.face_count + sources[
            None, :
        ].astype(keys.dtype)
        positions = jnp.searchsorted(keys, pair_keys)
        safe_positions = jnp.minimum(positions, keys.shape[0] - 1)
        exact_near = (positions < keys.shape[0]) & (keys[safe_positions] == pair_keys)
        target_points = pair_data.regular_points[targets]
        source_points = pair_data.regular_points[sources]
        differences = (
            target_points[:, None, :, None, :] - source_points[None, :, None, :, :]
        )
        safe_differences = jnp.where(
            exact_near[:, :, None, None, None],
            jnp.ones_like(differences),
            differences,
        )
        kernel = 1.0 / (4.0 * jnp.pi * jnp.linalg.norm(safe_differences, axis=-1))
        regular = oe.contract(
            "tq,sr,tsqr->ts",
            pair_data.regular_weights[targets],
            pair_data.regular_weights[sources],
            kernel,
            backend="jax",
        )
        values = jnp.where(exact_near, pair_data.values[safe_positions], regular)
        pair_classes = jnp.where(exact_near, pair_data.classes[safe_positions], 4)
        error_bounds = pair_data.maximum_errors[pair_classes]
        accuracy_supported = pair_data.supported[pair_classes]
        if self.formulation == "strong":
            scale = self.inverse_areas[targets, None]
            values = values * scale
            error_bounds = error_bounds * scale
        return BEMLocalBlock3D(
            values=values,
            target_indices=targets,
            source_indices=sources,
            exact_near_mask=exact_near,
            pair_classes=pair_classes,
            quadrature_error_bounds=error_bounds,
            accuracy_supported=accuracy_supported,
            envelope=self.envelope,
            provider_id=self.provider_id,
            block_id=canonical_fingerprint(
                {
                    "kind": "bem-local-block-3d",
                    "provider": self.provider_id,
                    "targets": array_tree_fingerprint(targets_host),
                    "sources": array_tree_fingerprint(sources_host),
                }
            ),
        )

    def diagonal(self, /) -> BEMLocalBlock3D:
        if self.face_count > self.max_block_entries:
            raise LinearCapabilityError(
                f"BEM diagonal requires {self.face_count} entries, exceeding the "
                f"declared limit {self.max_block_entries}."
            )
        indices = jnp.arange(self.face_count, dtype=jnp.int32)
        pair_data = self.weak.pair_data
        keys = pair_data.exception_keys
        diagonal_keys = indices.astype(keys.dtype) * (self.face_count + 1)
        positions = jnp.searchsorted(keys, diagonal_keys)
        safe_positions = jnp.minimum(positions, keys.shape[0] - 1)
        mask = (positions < keys.shape[0]) & (keys[safe_positions] == diagonal_keys)
        diagonal = jnp.where(mask, pair_data.values[safe_positions], jnp.nan)
        classes = jnp.where(mask, pair_data.classes[safe_positions], 4)
        errors = pair_data.maximum_errors[classes]
        accuracy_supported = mask & pair_data.supported[classes]
        if self.formulation == "strong":
            diagonal = diagonal * self.inverse_areas
            errors = errors * self.inverse_areas
        return BEMLocalBlock3D(
            values=diagonal,
            target_indices=indices,
            source_indices=indices,
            exact_near_mask=mask,
            pair_classes=classes,
            quadrature_error_bounds=errors,
            accuracy_supported=accuracy_supported,
            envelope=self.envelope,
            provider_id=self.provider_id,
            block_id=canonical_fingerprint(
                {
                    "kind": "bem-diagonal-3d",
                    "provider": self.provider_id,
                    "face_count": self.face_count,
                }
            ),
        )


__all__ = [
    "AbstractExactNearProvider3D",
    "BEMBlockActionStatus",
    "BEMExecutionEnvelope",
    "BEMFastCapabilityError",
    "BEMFastProviderCapabilities",
    "BEMLocalBlock3D",
    "BoundaryFastProviderName",
    "FusedBEMBlockActionResult",
    "FusedBlockedBEMAction3D",
    "LaplaceDP0ExactNearProvider3D",
    "boundary_fast_provider_capabilities",
    "prepare_laplace_fmm_provider_2d",
]
