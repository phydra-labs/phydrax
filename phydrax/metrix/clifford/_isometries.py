#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._spec import CliffordAlgebraSpec


def _matrix_index(matrices: np.ndarray, candidate: np.ndarray, tolerance: float) -> int:
    distances = np.max(np.abs(matrices - candidate[None, ...]), axis=(1, 2))
    matches = np.flatnonzero(distances <= tolerance)
    if matches.size != 1:
        raise ValueError(
            "Finite metric-isometry matrices are not uniquely closed under composition."
        )
    return int(matches[0])


def _validated_matrix(
    algebra: CliffordAlgebraSpec,
    matrix: ArrayLike,
    tolerance: float,
    /,
) -> tuple[np.ndarray, float]:
    if not isinstance(algebra, CliffordAlgebraSpec):
        raise TypeError("algebra must be a CliffordAlgebraSpec.")
    if not algebra.nondegenerate:
        raise ValueError("Metric isometry actions require a nondegenerate signature.")
    tolerance_ = float(tolerance)
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("Metric-isometry tolerance must be finite and nonnegative.")
    host = np.asarray(matrix)
    expected = (algebra.dimension, algebra.dimension)
    if host.shape != expected:
        raise ValueError(f"Metric isometry must have shape {expected}; got {host.shape}.")
    if np.iscomplexobj(host):
        raise TypeError("Metric-isometry matrices must be real-valued.")
    if not np.issubdtype(host.dtype, np.floating):
        host = host.astype(float)
    if np.any(~np.isfinite(host)):
        raise ValueError("Metric-isometry matrix must be finite.")
    metric = np.diag(np.asarray(algebra.diagonal, dtype=host.dtype))
    defect = float(np.max(np.abs(host.T @ metric @ host - metric)))
    if defect > tolerance_:
        raise ValueError(
            f"Matrix does not preserve Clifford metric; defect {defect} exceeds "
            f"tolerance {tolerance_}."
        )
    determinant = float(np.linalg.det(host))
    if not math.isfinite(determinant) or abs(determinant) <= tolerance_:
        raise ValueError("Metric-isometry matrix must be invertible.")
    return host, defect


class MetricIsometryAction(StrictModule, NonTrainableState):
    """One validated metric-preserving linear transformation, without closure claim."""

    algebra: CliffordAlgebraSpec
    matrix: Array
    inverse_matrix: Array
    metric_defect: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ):
        host, defect = _validated_matrix(algebra, matrix, tolerance)
        inverse = np.linalg.solve(
            host,
            np.eye(algebra.dimension, dtype=host.dtype),
        )
        tolerance_ = float(tolerance)
        inverse_defect = float(np.max(np.abs(inverse @ host - np.eye(algebra.dimension))))
        if inverse_defect > tolerance_:
            raise ValueError("Metric-isometry inverse failed its identity audit.")
        self.algebra = algebra
        self.matrix = jnp.asarray(host)
        self.inverse_matrix = jnp.asarray(inverse)
        self.metric_defect = defect
        self.tolerance = tolerance_
        self.action_id = canonical_fingerprint(
            {
                "kind": "metric-isometry-action-v1",
                "algebra": algebra.algebra_id,
                "matrix": host.tolist(),
                "tolerance": tolerance_,
            }
        )

    def inverse(self) -> "MetricIsometryAction":
        return MetricIsometryAction(
            self.algebra,
            self.inverse_matrix,
            tolerance=self.tolerance,
        )

    def compose(self, right: "MetricIsometryAction", /) -> "MetricIsometryAction":
        if not isinstance(right, MetricIsometryAction):
            raise TypeError("right must be a MetricIsometryAction.")
        self.algebra.require_compatible(right.algebra)
        return MetricIsometryAction(
            self.algebra,
            self.matrix @ right.matrix,
            tolerance=max(self.tolerance, right.tolerance),
        )


class MetricIsometryAuditSet(StrictModule, NonTrainableState):
    """Explicit standalone isometries used for audits, with no group-closure claim."""

    algebra: CliffordAlgebraSpec
    actions: tuple[MetricIsometryAction, ...]
    audit_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        actions: Sequence[MetricIsometryAction],
        /,
    ):
        resolved = tuple(actions)
        if not resolved:
            raise ValueError(
                "Metric isometry audit set must contain at least one action."
            )
        for action in resolved:
            if not isinstance(action, MetricIsometryAction):
                raise TypeError("Audit-set entries must be MetricIsometryAction values.")
            algebra.require_compatible(action.algebra)
        if len({action.action_id for action in resolved}) != len(resolved):
            raise ValueError("Metric isometry audit actions must be unique.")
        self.algebra = algebra
        self.actions = resolved
        self.audit_set_id = canonical_fingerprint(
            {
                "kind": "metric-isometry-audit-set-v1",
                "algebra": algebra.algebra_id,
                "actions": [action.action_id for action in resolved],
            }
        )


class FiniteMetricIsometryGroup(StrictModule, NonTrainableState):
    """Validated finite subgroup preserving one constant nondegenerate metric."""

    algebra: CliffordAlgebraSpec
    matrices: Array
    multiplication_table: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    inverse_indices: tuple[int, ...] = eqx.field(static=True)
    identity_index: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        matrices: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ):
        host = np.asarray(matrices)
        expected = (algebra.dimension, algebra.dimension)
        if host.ndim != 3 or host.shape[0] == 0 or host.shape[1:] != expected:
            raise ValueError(
                "Finite metric-isometry group matrices must have shape "
                f"(order, {algebra.dimension}, {algebra.dimension})."
            )
        tolerance_ = float(tolerance)
        validated = tuple(
            _validated_matrix(algebra, matrix, tolerance_)[0] for matrix in host
        )
        host = np.stack(validated, axis=0)
        identity = np.eye(algebra.dimension, dtype=host.dtype)
        pairwise = np.max(np.abs(host[:, None, :, :] - host[None, :, :, :]), axis=(2, 3))
        duplicate_mask = (pairwise <= tolerance_) & ~np.eye(host.shape[0], dtype=bool)
        if np.any(duplicate_mask):
            raise ValueError("Finite metric-isometry group matrices must be unique.")
        identity_index = _matrix_index(host, identity, tolerance_)
        table = tuple(
            tuple(
                _matrix_index(host, host[left] @ host[right], tolerance_)
                for right in range(host.shape[0])
            )
            for left in range(host.shape[0])
        )
        inverse_indices = tuple(
            _matrix_index(
                host,
                np.linalg.solve(
                    host[index],
                    np.eye(algebra.dimension, dtype=host.dtype),
                ),
                tolerance_,
            )
            for index in range(host.shape[0])
        )
        self.algebra = algebra
        self.matrices = jnp.asarray(host)
        self.multiplication_table = table
        self.inverse_indices = inverse_indices
        self.identity_index = identity_index
        self.tolerance = tolerance_
        self.group_id = canonical_fingerprint(
            {
                "kind": "finite-metric-isometry-group-v1",
                "algebra": algebra.algebra_id,
                "matrices": host.tolist(),
                "table": [list(row) for row in table],
                "inverses": list(inverse_indices),
                "identity": identity_index,
                "tolerance": tolerance_,
            }
        )

    @property
    def order(self) -> int:
        return int(self.matrices.shape[0])

    def action(self, index: int, /) -> MetricIsometryAction:
        index_ = int(index)
        if index_ < 0 or index_ >= self.order:
            raise IndexError("Finite metric-isometry group index is out of range.")
        return MetricIsometryAction(
            self.algebra,
            self.matrices[index_],
            tolerance=self.tolerance,
        )


def lorentz_boost_action(
    algebra: CliffordAlgebraSpec,
    spatial_axis: int,
    rapidity: float,
    /,
    *,
    time_axis: int = 0,
    tolerance: float = 1e-10,
) -> MetricIsometryAction:
    """Construct one coordinate-axis boost as a standalone metric isometry."""
    time = int(time_axis)
    space = int(spatial_axis)
    if (
        time == space
        or not 0 <= time < algebra.dimension
        or not 0 <= space < algebra.dimension
    ):
        raise ValueError(
            "Lorentz boost requires distinct in-range time and spatial axes."
        )
    if algebra.diagonal[time] != -algebra.diagonal[space] or algebra.diagonal[time] == 0:
        raise ValueError("Lorentz boost axes must have opposite nonzero metric signs.")
    rapidity_ = float(rapidity)
    if not math.isfinite(rapidity_):
        raise ValueError("Lorentz rapidity must be finite.")
    matrix = np.eye(algebra.dimension)
    cosine = math.cosh(rapidity_)
    sine = math.sinh(rapidity_)
    matrix[time, time] = cosine
    matrix[space, space] = cosine
    matrix[time, space] = sine
    matrix[space, time] = sine
    return MetricIsometryAction(algebra, matrix, tolerance=tolerance)


__all__ = [
    "FiniteMetricIsometryGroup",
    "lorentz_boost_action",
    "MetricIsometryAction",
    "MetricIsometryAuditSet",
]
