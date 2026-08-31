#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics import fejer_first_data
from ..._polynomial._orthogonal import (
    standard_derivative_matrix,
    standard_vandermonde,
)
from ...linalg import DenseLinearTransform
from .._axis_domain import AxisDomain
from .._spectral import ModalTransform
from ._basis import (
    _analysis_from_synthesis,
    AbstractSpectralBasisPlan,
    PreparedSpectralAxis,
    SpectralModeLayout,
)
from ._precision import SpectralPrecisionPolicy


class RationalChebyshevLineBasisPlan(AbstractSpectralBasisPlan):
    """Rational Chebyshev modes on the full real line."""

    scale: Array
    maximum_construction_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        scale: ArrayLike = 1.0,
        /,
        *,
        maximum_construction_bytes: int = 512 * 1024**2,
    ):
        count, scale_, maximum = _plan_values(
            mode_count,
            scale,
            maximum_construction_bytes,
        )
        self.mode_count = count
        self.scale = scale_
        self.maximum_construction_bytes = maximum
        self.family = "rational_chebyshev_line"
        self.periodic = False
        self.boundary = "unconstrained"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rational-chebyshev-line-basis-plan",
                "mode_count": count,
                "scale": array_tree_fingerprint(scale_),
                "maximum_construction_bytes": maximum,
            }
        )

    def resized(self, mode_count: int, /) -> "RationalChebyshevLineBasisPlan":
        return RationalChebyshevLineBasisPlan(
            mode_count,
            self.scale,
            maximum_construction_bytes=self.maximum_construction_bytes,
        )

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        if not isinstance(domain, AxisDomain) or domain.kind != "real_line":
            raise ValueError(
                "RationalChebyshevLineBasisPlan requires a real-line domain."
            )
        scale = float(np.asarray(self.scale))

        def mapping(reference):
            complement = 1.0 - reference * reference
            nodes = scale * reference / np.sqrt(complement)
            jacobian = scale / complement**1.5
            return nodes, jacobian

        return _prepare_rational(self, domain, precision, mapping)


class RationalChebyshevHalfLineBasisPlan(AbstractSpectralBasisPlan):
    """Rational Chebyshev modes on a positive or negative half-line."""

    scale: Array
    maximum_construction_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        scale: ArrayLike = 1.0,
        /,
        *,
        maximum_construction_bytes: int = 512 * 1024**2,
    ):
        count, scale_, maximum = _plan_values(
            mode_count,
            scale,
            maximum_construction_bytes,
        )
        self.mode_count = count
        self.scale = scale_
        self.maximum_construction_bytes = maximum
        self.family = "rational_chebyshev_half_line"
        self.periodic = False
        self.boundary = "unconstrained"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rational-chebyshev-half-line-basis-plan",
                "mode_count": count,
                "scale": array_tree_fingerprint(scale_),
                "maximum_construction_bytes": maximum,
            }
        )

    def resized(self, mode_count: int, /) -> "RationalChebyshevHalfLineBasisPlan":
        return RationalChebyshevHalfLineBasisPlan(
            mode_count,
            self.scale,
            maximum_construction_bytes=self.maximum_construction_bytes,
        )

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        if not isinstance(domain, AxisDomain) or domain.kind != "half_line":
            raise ValueError(
                "RationalChebyshevHalfLineBasisPlan requires a half-line domain."
            )
        scale = float(np.asarray(self.scale))
        endpoint = float(
            np.asarray(domain.lower if domain.direction == "positive" else domain.upper)
        )
        direction = domain.direction

        def mapping(reference):
            if direction == "positive":
                denominator = 1.0 - reference
                nodes = endpoint + scale * (1.0 + reference) / denominator
                jacobian = 2.0 * scale / denominator**2
            else:
                denominator = 1.0 + reference
                nodes = endpoint - scale * (1.0 - reference) / denominator
                jacobian = 2.0 * scale / denominator**2
            return nodes, jacobian

        return _prepare_rational(self, domain, precision, mapping)


def _plan_values(
    mode_count: int,
    scale: ArrayLike,
    maximum_construction_bytes: int,
    /,
) -> tuple[int, Array, int]:
    count = int(mode_count)
    scale_ = jnp.asarray(scale, dtype=float).reshape(())
    maximum = int(maximum_construction_bytes)
    if count < 2 or maximum <= 0:
        raise ValueError("Rational Chebyshev mode count and budget must be positive.")
    scale_ = eqx.error_if(
        scale_,
        ~(jnp.isfinite(scale_) & (scale_ > 0.0)),
        "Rational Chebyshev scale must be finite and positive.",
    )
    return count, scale_, maximum


def _prepare_rational(
    plan: RationalChebyshevLineBasisPlan | RationalChebyshevHalfLineBasisPlan,
    domain: AxisDomain,
    precision: SpectralPrecisionPolicy,
    mapping,
    /,
) -> PreparedSpectralAxis:
    count = plan.mode_count
    itemsize = np.dtype(precision.coefficient_dtype).itemsize
    estimate = 8 * count * count * itemsize
    if estimate > plan.maximum_construction_bytes:
        raise ValueError("Rational Chebyshev construction exceeds its byte budget.")
    rule = fejer_first_data(count)
    reference = np.asarray(rule.nodes, dtype=float)
    reference_weights = np.asarray(rule.weights, dtype=float)
    nodes, jacobian = mapping(reference)
    weights = reference_weights * jacobian
    if (
        np.any(~np.isfinite(nodes))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise FloatingPointError(
            "Rational Chebyshev nodes and physical weights must remain finite."
        )
    synthesis = np.asarray(standard_vandermonde("chebyshev", reference, count - 1))
    analysis = _analysis_from_synthesis(synthesis, precision.coefficient_dtype)
    derivative_reference = np.asarray(
        standard_derivative_matrix(
            "chebyshev",
            count,
            dtype=precision.physical_dtype,
        )
    )
    derivative_values = (synthesis @ derivative_reference) / jacobian[:, None]
    derivative = analysis @ derivative_values
    residual = _derivative_closure_residual(
        count,
        derivative,
        mapping,
        precision,
    )
    mode_ids = tuple(f"{plan.family}:{degree}" for degree in range(count))
    modal = ModalTransform(
        analysis,
        synthesis,
        weights,
        mode_ids=mode_ids,
    )
    execution = DenseLinearTransform(
        np.asarray(analysis, dtype=precision.coefficient_dtype),
        np.asarray(synthesis, dtype=precision.coefficient_dtype),
        transform_id=modal.transform_id,
    )
    return PreparedSpectralAxis(
        plan,
        nodes,
        reference,
        weights,
        domain,
        SpectralModeLayout(
            plan.family,
            np.arange(count),
            mode_ids=mode_ids,
        ),
        execution,
        precision,
        lower_endpoint_included=False,
        upper_endpoint_included=False,
        derivative_matrix=derivative,
        derivative_exact=False,
        derivative_residual=residual,
        modal_transform=modal,
    )


def _derivative_closure_residual(
    count: int,
    derivative: np.ndarray,
    mapping,
    precision: SpectralPrecisionPolicy,
    /,
) -> float:
    validation = fejer_first_data(2 * count + 1)
    reference = np.asarray(validation.nodes, dtype=float)
    _, jacobian = mapping(reference)
    synthesis = np.asarray(standard_vandermonde("chebyshev", reference, count - 1))
    derivative_reference = np.asarray(
        standard_derivative_matrix(
            "chebyshev",
            count,
            dtype=precision.physical_dtype,
        )
    )
    exact = (synthesis @ derivative_reference) / jacobian[:, None]
    projected = synthesis @ derivative
    scale = max(1.0, float(np.max(np.abs(exact), initial=0.0)))
    return float(np.max(np.abs(projected - exact), initial=0.0) / scale)


__all__ = [
    "RationalChebyshevHalfLineBasisPlan",
    "RationalChebyshevLineBasisPlan",
]
