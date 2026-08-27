#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
)
from ..._holomorphic_linear import HolomorphicLinearFrame
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._holomorphic_frame import HolomorphicPolynomialFrame


HolomorphicTraceEvidenceKind = Literal[
    "finite-functional-exact",
    "continuous-subspace-exact",
    "continuous-validated-bound",
    "sampled-audit",
]


class HolomorphicTraceCertificate(StrictModule, NonTrainableState):
    """Explicit scope and geometry for one holomorphic boundary trace claim."""

    evidence_kind: HolomorphicTraceEvidenceKind = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    trace_space_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    topology_assumptions: tuple[str, ...] = eqx.field(static=True)
    residual_bound: Array
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        evidence_kind: HolomorphicTraceEvidenceKind,
        geometry_id: str,
        trace_space_id: str,
        field_id: str,
        topology_assumptions: tuple[str, ...] = (),
        residual_bound: ArrayLike = 0.0,
    ):
        if evidence_kind not in (
            "finite-functional-exact",
            "continuous-subspace-exact",
            "continuous-validated-bound",
            "sampled-audit",
        ):
            raise ValueError("Unknown holomorphic trace evidence kind.")
        identifiers = (str(geometry_id), str(trace_space_id), str(field_id))
        assumptions = tuple(str(value) for value in topology_assumptions)
        bound = jnp.asarray(residual_bound)
        if any(not value for value in identifiers) or any(
            not value for value in assumptions
        ):
            raise ValueError("Holomorphic trace identifiers must be nonempty.")
        if bound.shape != () or not bool(jnp.isfinite(bound)) or bool(bound < 0.0):
            raise ValueError(
                "Holomorphic trace residual bound must be finite and nonnegative."
            )
        if evidence_kind.endswith("exact") and bool(bound != 0.0):
            raise ValueError(
                "Exact holomorphic trace evidence requires zero residual bound."
            )
        self.evidence_kind = evidence_kind
        self.geometry_id = identifiers[0]
        self.trace_space_id = identifiers[1]
        self.field_id = identifiers[2]
        self.topology_assumptions = assumptions
        self.residual_bound = bound
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "holomorphic-trace-certificate",
                "evidence_kind": evidence_kind,
                "geometry_id": identifiers[0],
                "trace_space_id": identifiers[1],
                "field_id": identifiers[2],
                "topology_assumptions": list(assumptions),
                "residual_bound": array_tree_fingerprint(bound),
            }
        )


class HolomorphicContourFunctional(StrictModule, NonTrainableState):
    """Explicit quadrature contour moment of one holomorphic frame derivative."""

    nodes: Array
    weights: Array
    derivative_multi_index: tuple[int, ...] = eqx.field(static=True)
    output_index: int = eqx.field(static=True)
    component_weight: complex = eqx.field(static=True)
    construction: str = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        derivative_multi_index: tuple[int, ...],
        output_index: int = 0,
        component_weight: complex = 1.0 + 0.0j,
        construction: str = "holomorphic-contour-moment",
    ):
        nodes_raw = np.asarray(nodes, dtype=np.complex128)
        if nodes_raw.ndim == 1:
            nodes_raw = nodes_raw[:, None]
        weights_raw = np.asarray(weights, dtype=np.complex128)
        derivative = tuple(int(value) for value in derivative_multi_index)
        output = int(output_index)
        component = complex(component_weight)
        construction_ = str(construction)
        if (
            nodes_raw.ndim != 2
            or nodes_raw.shape[0] == 0
            or weights_raw.shape != (nodes_raw.shape[0],)
        ):
            raise ValueError("Contour nodes and weights have incompatible shapes.")
        if not np.all(np.isfinite(nodes_raw)) or not np.all(np.isfinite(weights_raw)):
            raise ValueError("Contour nodes and weights must be finite.")
        if (
            len(derivative) != nodes_raw.shape[1]
            or any(value < 0 for value in derivative)
            or output < 0
            or not math.isfinite(abs(component))
            or component == 0.0j
            or not construction_
        ):
            raise ValueError("Contour functional metadata is invalid.")
        nodes_ = jnp.asarray(nodes_raw)
        weights_ = jnp.asarray(weights_raw)
        self.nodes = nodes_
        self.weights = weights_
        self.derivative_multi_index = derivative
        self.output_index = output
        self.component_weight = component
        self.construction = construction_
        self.functional_id = canonical_fingerprint(
            {
                "kind": "holomorphic-contour-functional",
                "nodes": array_tree_fingerprint(nodes_),
                "weights": array_tree_fingerprint(weights_),
                "derivative_multi_index": list(derivative),
                "output_index": output,
                "component_weight": array_tree_fingerprint(jnp.asarray(component)),
                "construction": construction_,
            }
        )

    def assemble_row(self, frame: HolomorphicLinearFrame, /) -> Array:
        if not isinstance(frame, HolomorphicLinearFrame):
            raise TypeError("frame must implement HolomorphicLinearFrame.")
        certificate = frame.linear_frame_certificate()
        if int(self.nodes.shape[1]) != certificate.complex_input_size:
            raise ValueError("Contour and frame input dimensions differ.")
        if self.output_index >= certificate.complex_output_size:
            raise ValueError("Contour output index exceeds the frame output size.")
        if sum(self.derivative_multi_index) > certificate.maximum_derivative_order:
            raise ValueError("Contour derivative order exceeds the frame evidence.")
        basis = jax.vmap(
            lambda node: frame.basis_derivative(
                node,
                self.derivative_multi_index,
            )[self.output_index]
        )(self.nodes)
        moment = jnp.sum(self.weights[:, None] * basis, axis=0)
        return jnp.real(self.component_weight * moment)


def holomorphic_period_functional(
    nodes: ArrayLike,
    differential_weights: ArrayLike,
    /,
    *,
    output_index: int = 0,
    component: Literal["real", "imaginary"] = "real",
) -> HolomorphicContourFunctional:
    """Closed-contour period of a scalar-input holomorphic derivative."""
    if component not in ("real", "imaginary"):
        raise ValueError("Period component must be real or imaginary.")
    return HolomorphicContourFunctional(
        nodes,
        differential_weights,
        derivative_multi_index=(1,),
        output_index=output_index,
        component_weight=1.0 if component == "real" else -1j,
        construction="holomorphic-closed-contour-period",
    )


class DiskHolomorphicTracePlan(StrictModule, NonTrainableState):
    """Exact finite Fourier-to-Taylor lift on one physical circle."""

    center: complex = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    maximum_mode: int = eqx.field(static=True)
    frame: HolomorphicPolynomialFrame
    geometry_id: str = eqx.field(static=True)
    trace_space_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_mode: int,
        /,
        *,
        center: complex = 0.0j,
        radius: float = 1.0,
    ):
        mode = int(maximum_mode)
        center_ = complex(center)
        radius_ = float(radius)
        if mode < 0 or not math.isfinite(abs(center_)):
            raise ValueError("Disk trace mode and center are invalid.")
        if not math.isfinite(radius_) or radius_ <= 0.0:
            raise ValueError("Disk trace radius must be finite and positive.")
        normalization = ComplexAffineNormalization.scalar(
            center=center_,
            scale=radius_,
        )
        frame = HolomorphicPolynomialFrame.one_variable(
            mode,
            normalization=normalization,
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "complex-disk-boundary",
                "center": array_tree_fingerprint(jnp.asarray(center_)),
                "radius": radius_,
            }
        )
        trace_space_id = canonical_fingerprint(
            {
                "kind": "real-finite-fourier-trace-space",
                "geometry": geometry_id,
                "maximum_mode": mode,
            }
        )
        self.center = center_
        self.radius = radius_
        self.maximum_mode = mode
        self.frame = frame
        self.geometry_id = geometry_id
        self.trace_space_id = trace_space_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "disk-holomorphic-trace-plan",
                "geometry": geometry_id,
                "trace_space": trace_space_id,
                "frame": frame.linear_frame_certificate().frame_id,
            }
        )

    def lift(
        self,
        cosine_coefficients: ArrayLike,
        sine_coefficients: ArrayLike,
        /,
    ) -> DiskHolomorphicTraceLift:
        cosine = jnp.asarray(cosine_coefficients)
        sine = jnp.asarray(sine_coefficients)
        shape = (self.maximum_mode + 1,)
        if cosine.shape != shape or sine.shape != shape:
            raise ValueError(
                "Disk trace coefficients must include modes zero through maximum."
            )
        if jnp.iscomplexobj(cosine) or jnp.iscomplexobj(sine):
            raise TypeError("Disk trace coefficients must be real.")
        if not bool(jnp.all(jnp.isfinite(cosine))) or not bool(
            jnp.all(jnp.isfinite(sine))
        ):
            raise ValueError("Disk trace coefficients must be finite.")
        if not bool(sine[0] == 0.0):
            raise ValueError("The zero Fourier sine coefficient must be zero.")
        complex_coefficients = cosine - 1j * sine
        coefficient_vector = jnp.concatenate(
            (jnp.real(complex_coefficients), jnp.imag(complex_coefficients))
        )
        return DiskHolomorphicTraceLift(self, coefficient_vector)


class DiskHolomorphicTraceLift(StrictModule, NonTrainableState):
    """Globally holomorphic polynomial with an exact finite Fourier circle trace."""

    __hash__ = object.__hash__

    plan: DiskHolomorphicTracePlan
    coefficient_vector: Array
    _holomorphic_certificate: HolomorphicMapCertificate
    _trace_certificate: HolomorphicTraceCertificate

    def __init__(
        self,
        plan: DiskHolomorphicTracePlan,
        coefficient_vector: ArrayLike,
        /,
    ):
        if not isinstance(plan, DiskHolomorphicTracePlan):
            raise TypeError("plan must be DiskHolomorphicTracePlan.")
        coefficients = jnp.asarray(coefficient_vector)
        frame_certificate = plan.frame.linear_frame_certificate()
        if coefficients.shape != (frame_certificate.real_coefficient_count,):
            raise ValueError("Disk trace coefficient vector has invalid shape.")
        if jnp.iscomplexobj(coefficients):
            raise TypeError("Disk trace coefficient vector must be real Cartesian.")
        field_id = canonical_fingerprint(
            {
                "kind": "disk-holomorphic-trace-lift",
                "plan": plan.plan_id,
                "coefficients": array_tree_fingerprint(coefficients),
            }
        )
        self.plan = plan
        self.coefficient_vector = coefficients
        self._holomorphic_certificate = HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=1,
            construction="disk-fourier-to-holomorphic-polynomial-lift",
            normalization_id=frame_certificate.normalization_id,
            maximum_derivative_order=frame_certificate.maximum_derivative_order,
            operations=("complex-affine", "complex-polynomial"),
            parameter_coverage="finite-parametric-family",
            linear_in_parameters=False,
            parameter_mode="fixed-real-cartesian-coefficients",
            construction_dependencies=(frame_certificate.frame_id, field_id),
        )
        self._trace_certificate = HolomorphicTraceCertificate(
            evidence_kind="continuous-subspace-exact",
            geometry_id=plan.geometry_id,
            trace_space_id=plan.trace_space_id,
            field_id=field_id,
            topology_assumptions=("single circular boundary component",),
        )

    def __call__(self, coordinate: ArrayLike, /) -> Array:
        return self.plan.frame.evaluate(coordinate, self.coefficient_vector)

    def jet(self, coordinate: ArrayLike, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if order_ < 0 or order_ > self.plan.maximum_mode:
            raise ValueError("Requested disk trace jet order is unavailable.")
        value = self(coordinate)
        derivatives = tuple(
            self.plan.frame.basis_derivative(coordinate, (current,))
            @ self.coefficient_vector
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._holomorphic_certificate

    def trace_certificate(self) -> HolomorphicTraceCertificate:
        return self._trace_certificate


__all__ = [
    "DiskHolomorphicTraceLift",
    "DiskHolomorphicTracePlan",
    "HolomorphicContourFunctional",
    "HolomorphicTraceCertificate",
    "HolomorphicTraceEvidenceKind",
    "holomorphic_period_functional",
]
