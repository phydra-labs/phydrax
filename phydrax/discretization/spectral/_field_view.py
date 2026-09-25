#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Arbitrary-point tensor spectral synthesis as an evidenced field reconstruction.

The kernel evaluates the canonical per-axis basis rows of a prepared
`TensorSpectralDiscretization` (`PreparedSpectralAxis.evaluate_basis`), so
reconstructed values and coordinate derivatives agree with `reconstruct` and
`derivative_values` on the grid. The support is the closed tensor box of the
axes (the periodic cell on periodic axes); points outside it are
`OUTSIDE_SUPPORT` and are never wrapped or extrapolated.
"""

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._view_support import tensor_box_support_geometry
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from ._basis import PreparedSpectralAxis
from ._precision import SpectralPrecisionPolicy
from ._space import (
    _point_support_mask,
    _tensor_point_rows,
    _tensor_point_synthesis,
    _tensor_point_transpose,
    TensorSpectralDiscretization,
)


@final
class _SpectralRoute(StrictModule):
    rows: tuple[Array, ...]


@final
class SpectralFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Canonical tensor spectral synthesis at arbitrary points of the axis box.

    Routes hold the per-axis basis rows (`points x modes_a`) of the requested
    derivative multi-index; `apply` contracts them with the modal coefficients
    and `transpose` is the exact bilinear transpose of that contraction. For a
    real physical dtype, values are the real part of the complex synthesis
    (the `reconstruct` real-output semantics); the transpose then satisfies
    `<R c, w> = Re(sum(c * R^T w))`.
    """

    axes: tuple[PreparedSpectralAxis, ...]
    precision: SpectralPrecisionPolicy
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    real_output: bool = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        field_space_id: str,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        precision = discretization.plan.precision
        real = not precision.physical_dtype.startswith("complex")
        self.axes = discretization.axes
        self.precision = precision
        self.modal_shape = discretization.modal_shape
        self.real_output = real
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "tensor-spectral-field-reconstruction-kernel",
                "axes": [axis.axis_id for axis in discretization.axes],
                "precision": precision.policy_id,
                "modal_shape": list(discretization.modal_shape),
                "real_output": real,
                "field_space": field_space_id,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return 0

    @property
    def support_coverage(self) -> str:
        return "complete"

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[_SpectralRoute, FieldQueryEvidence]:
        # Single-valued smooth synthesis: every trace side is the evaluation itself.
        del side
        query = points.astype(jnp.finfo(jnp.dtype(self.precision.physical_dtype)).dtype)
        finite = jnp.all(jnp.isfinite(query), axis=1)
        inside = _point_support_mask(self.axes, query, include_periodic=True)
        status = jnp.where(
            inside,
            int(FieldQueryStatus.VALID),
            jnp.where(
                finite,
                int(FieldQueryStatus.OUTSIDE_SUPPORT),
                int(FieldQueryStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        route = _SpectralRoute(_tensor_point_rows(self.axes, query, derivative))
        evidence = FieldQueryEvidence(
            status,
            jnp.ones(query.shape[:1], dtype=query.dtype),
            inside.astype(jnp.int32),
            kernel_id=self.kernel_id,
        )
        return route, evidence

    def apply(self, route: _SpectralRoute, coefficients: Array, /) -> Array:
        values = _tensor_point_synthesis(
            route.rows, self.precision.coefficients(coefficients)
        )
        if self.real_output:
            return self.precision.output(jnp.real(values))
        return values

    def transpose(self, route: _SpectralRoute, cotangent: Array, /) -> Array:
        return _tensor_point_transpose(route.rows, self.precision.coefficients(cotangent))

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        del sites, side, cell_ids
        raise ValueError("Single-valued spectral reconstructions have no trace cells.")


def _axis_box(
    discretization: TensorSpectralDiscretization, /
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for name, axis in zip(
        discretization.plan.axis_names, discretization.axes, strict=True
    ):
        bounds = axis.bounds
        if bounds is None:
            raise ValueError(
                f"Spectral axis {name!r} ({axis.family}) is unbounded; a field view "
                "needs a bounded or periodic tensor box support."
            )
        values = np.asarray(bounds, dtype=np.float64)
        lower.append(values[0])
        upper.append(values[1])
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def prepare_spectral_field_reconstruction(
    discretization: TensorSpectralDiscretization,
    /,
    *,
    value_port: ValuePort | None = None,
    support_geometry: Any = None,
    maximum_derivative_order: int = 2,
    support_tolerance: float = 1.0e-9,
) -> PreparedFieldReconstruction:
    """Prepare an evidenced coordinate reconstruction of one tensor spectral field.

    Arbitrary points are synthesized by the canonical basis rows of every axis
    (Fourier, sine, cosine, Chebyshev, Legendre); constrained and rational axes
    refuse with `ValueError`. Regularity is smooth and derivatives are exact up
    to `maximum_derivative_order`. The support is the tensor box of the axes
    (`support_geometry` defaults to an `Orthotope`; an explicit geometry must
    have the box's bounds and measure). `value_port.event_shape` declares
    trailing component axes, so coefficients have shape
    `modal_shape + event_shape`. Values are real for real physical dtypes (the
    real part of the complex modal synthesis, as in `reconstruct`) and complex
    otherwise.
    """
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    if isinstance(maximum_derivative_order, bool) or not isinstance(
        maximum_derivative_order, (int, np.integer)
    ):
        raise TypeError("maximum_derivative_order must be an int.")
    if maximum_derivative_order < 0:
        raise ValueError("maximum_derivative_order must be non-negative.")
    tolerance = float(support_tolerance)
    lower, upper = _axis_box(discretization)
    for axis in discretization.axes:
        # Refuse axes without canonical point synthesis at preparation time.
        axis.evaluate_basis(axis.nodes[:1])
    support_id = canonical_fingerprint(
        {
            "kind": "tensor-spectral-support",
            "support": discretization.support.support_id,
            "box": array_tree_fingerprint(np.stack((lower, upper))),
            "periodic": [axis.periodic for axis in discretization.axes],
        }
    )
    geometry = tensor_box_support_geometry(
        lower,
        upper,
        support_geometry,
        feature_id=f"tensor-spectral-support:{support_id}",
        tolerance=tolerance,
    )
    field_name = discretization.plan.field_name
    if value_port is None:
        components: tuple[int, ...] = ()
    elif isinstance(value_port, ValuePort):
        components = value_port.event_shape
    else:
        raise TypeError("value_port must be a ValuePort or None.")
    field_space_id = canonical_fingerprint(
        {
            "kind": "tensor-spectral-field-space",
            "discretization": discretization.prepared_id,
            "modal_space": discretization.modal_space.field_space_id,
            "components": list(components),
        }
    )
    port = (
        ValuePort(
            field_name,
            event_shape=(),
            component_ids=(field_name,),
            representation="spectral-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    kernel = SpectralFieldReconstructionKernel(
        discretization, field_space_id=field_space_id
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=DerivativeRegularity.smooth(),
        trace_policy=FieldTracePolicy("single-valued"),
        coefficient_shape=(*discretization.modal_shape, *components),
        physical_dimension=len(discretization.axes),
        maximum_derivative_order=int(maximum_derivative_order),
        field_space_id=field_space_id,
        support_id=support_id,
    )


__all__ = [
    "SpectralFieldReconstructionKernel",
    "prepare_spectral_field_reconstruction",
]
