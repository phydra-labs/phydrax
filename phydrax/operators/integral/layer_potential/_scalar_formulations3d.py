#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    AbstractLinearOperator,
    DiagonalLinearOperator,
    estimate_operator_action_cost,
    IdentityLinearOperator,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ....linalg._operators import _AbstractCostedLinearOperator
from ._scalar_calderon3d import ScalarCalderonDP0Galerkin3D
from ._scalar_trace import ScalarTraceSide3D, UnsupportedScalarBoundarySpaceError


ScalarLayerRepresentation3D = Literal[
    "single-layer", "double-layer", "direct-trace", "combined-field"
]


class ScalarBoundaryFormulationMetadata3D(StrictModule, NonTrainableState):
    """Exact envelope, signs, provider evidence, and non-goals of a BIE system."""

    ambient_dimension: int = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation_name: str = eqx.field(static=True)
    side: ScalarTraceSide3D = eqx.field(static=True)
    boundary_condition: str = eqx.field(static=True)
    unknown: str = eqx.field(static=True)
    given_data: str = eqx.field(static=True)
    representation: ScalarLayerRepresentation3D = eqx.field(static=True)
    discrete_space: str = eqx.field(static=True)
    weak_strong_form: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    assembly_report_id: str = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    action_cost_exact: bool = eqx.field(static=True)
    action_cost_reason: str = eqx.field(static=True)
    quadrature_error_evidence: str = eqx.field(static=True)
    double_layer_jump: float = eqx.field(static=True)
    single_layer_neumann_jump: float = eqx.field(static=True)
    compatibility_requirement: str = eqx.field(static=True)
    gauge: str = eqx.field(static=True)
    resonance_risk: str = eqx.field(static=True)
    coupling_parameter: float | None = eqx.field(static=True)
    hypersingular_required: bool = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)


class ScalarBoundaryFormulation3D(StrictModule, NonTrainableState):
    """Prepared fixed-size boundary system and exact map from data to its RHS.

    This is a discrete DP0 Galerkin formulation for a closed oriented triangle
    mesh in 3D. Its metadata records PDE, side, trace jumps, provider,
    precision, resource/error evidence, resonance/compatibility limitations,
    and non-goals. It does not certify continuum discretization error.
    """

    calderon: ScalarCalderonDP0Galerkin3D
    operator: AbstractLinearOperator
    data_operator: AbstractLinearOperator
    metadata: ScalarBoundaryFormulationMetadata3D
    robin_alpha: Array
    robin_beta: Array

    def right_hand_side(self, boundary_data: ArrayLike, /) -> Array:
        values = jnp.asarray(boundary_data, dtype=self.calderon.space.dtype)
        values = self.calderon.space.validate(values)
        return self.calderon.space.validate(self.data_operator.mv(values))


class _ComponentMeanProjector3D(_AbstractCostedLinearOperator):
    areas: Array
    component_ids: Array
    component_measures: Array
    component_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        space,
        areas: Array,
        component_ids: Array,
        component_count: int,
        /,
        *,
        operator_id: str,
    ):
        areas_ = jnp.asarray(areas, dtype=space.dtype)
        components = jnp.asarray(component_ids, dtype=jnp.int32)
        count = int(component_count)
        if areas_.shape != (space.size,) or components.shape != (space.size,):
            raise ValueError("Component projector geometry must match the DP0 space.")
        measures = jnp.stack(
            tuple(
                jnp.sum(jnp.where(components == component, areas_, 0.0))
                for component in range(count)
            )
        )
        if not bool(jnp.all(jnp.isfinite(measures) & (jnp.real(measures) > 0.0))):
            raise ValueError("Every surface component must have positive finite area.")
        self.areas = areas_
        self.component_ids = components
        self.component_measures = measures
        self.component_count = count
        self.workspace_bytes = int(
            np.dtype(space.dtype).itemsize * (2 * space.size + 2 * count)
        )
        self.source = space
        self.target = space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        means = jnp.stack(
            tuple(
                jnp.sum(
                    jnp.where(
                        self.component_ids == component,
                        self.areas * value,
                        0.0,
                    )
                )
                / self.component_measures[component]
                for component in range(self.component_count)
            )
        )
        return self.target.validate(means[self.component_ids])

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        totals = jnp.stack(
            tuple(
                jnp.sum(jnp.where(self.component_ids == component, value, 0.0))
                for component in range(self.component_count)
            )
        )
        return self.source.validate(
            self.areas
            * totals[self.component_ids]
            / self.component_measures[self.component_ids]
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError("Component gauge projectors cannot materialize.")

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return self.workspace_bytes, "surface-component-mean-projector-action"


def _checked(calderon: ScalarCalderonDP0Galerkin3D, /):
    if not isinstance(calderon, ScalarCalderonDP0Galerkin3D):
        raise TypeError("calderon must be ScalarCalderonDP0Galerkin3D.")
    if not bool(calderon.assembly_report.accuracy_supported):
        raise ValueError("Scalar Calderon quadrature does not support this formulation.")
    return calderon


def _resonance_risk(
    calderon: ScalarCalderonDP0Galerkin3D,
    boundary_condition: str,
    representation: ScalarLayerRepresentation3D,
    /,
) -> str:
    if calderon.kernel.family != "outgoing-helmholtz":
        return "not-applicable-to-this-coercive-or-static-kernel"
    if representation == "combined-field":
        return (
            "positive-real-eta-CFIE-removes-the-standard-raw-exterior-Dirichlet-"
            "resonance;-discrete-conditioning-is-not-certified"
        )
    if boundary_condition == "Dirichlet" and representation == "single-layer":
        return "raw-single-layer-is-singular-at-interior-Dirichlet-eigenfrequencies"
    if boundary_condition == "Dirichlet" and representation == "double-layer":
        return "raw-double-layer-is-singular-at-interior-Neumann-eigenfrequencies"
    return "raw-Helmholtz-formulation-has-unscreened-frequency-dependent-resonance-risk"


def _exterior_condition(calderon: ScalarCalderonDP0Galerkin3D, /) -> str:
    if calderon.kernel.family == "laplace":
        return "harmonic-field-decays-at-infinity"
    if calderon.kernel.family == "modified-helmholtz":
        return "modified-Helmholtz-field-decays-exponentially-at-infinity"
    return "outgoing-Sommerfeld-radiation-condition"


def _metadata(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
    *,
    operator: AbstractLinearOperator,
    name: str,
    side: ScalarTraceSide3D,
    boundary_condition: str,
    unknown: str,
    given_data: str,
    representation: ScalarLayerRepresentation3D,
    compatibility: str,
    gauge: str,
    resonance_risk: str | None = None,
    coefficient_id: str | None = None,
    coupling_parameter: float | None = None,
) -> ScalarBoundaryFormulationMetadata3D:
    cost = estimate_operator_action_cost(operator)
    report = calderon.assembly_report
    convention = calderon.trace_convention
    jump_d = convention.double_layer_dirichlet_jump(side)
    jump_s = convention.single_layer_neumann_jump(side)
    risk = (
        _resonance_risk(calderon, boundary_condition, representation)
        if resonance_risk is None
        else resonance_risk
    )
    non_goals = (
        "no-continuum-discretization-error-certificate",
        "no-automatic-Helmholtz-eigenfrequency-detection",
        "no-hypersingular-W-or-normal-trace-of-double-layer",
        "no-off-surface-reconstruction-for-direct-trace-or-modified-Helmholtz",
        "no-open-curved-or-higher-order-surfaces",
        "no-dense-materialization",
    )
    formulation_id = canonical_fingerprint(
        {
            "kind": "scalar-boundary-formulation-3d-v1",
            "assembly": report.report_id,
            "name": name,
            "side": side,
            "boundary_condition": boundary_condition,
            "unknown": unknown,
            "representation": representation,
            "jumps": (jump_d, jump_s),
            "compatibility": compatibility,
            "gauge": gauge,
            "resonance_risk": risk,
            "coefficients": coefficient_id,
            "coupling_parameter": coupling_parameter,
        }
    )
    return ScalarBoundaryFormulationMetadata3D(
        ambient_dimension=3,
        boundary_dimension=2,
        pde=report.pde,
        geometry=report.geometry,
        formulation_name=name,
        side=side,
        boundary_condition=boundary_condition,
        unknown=unknown,
        given_data=given_data,
        representation=representation,
        discrete_space=report.trial_space,
        weak_strong_form=(
            "DP0-weak-Galerkin-operators-followed-by-diagonal-face-mass-inverse"
        ),
        provider=report.provider,
        precision_policy_id=report.precision_policy_id,
        assembly_report_id=report.report_id,
        preparation_workspace_bytes=report.preparation_workspace_bytes,
        resident_bytes=max(report.resident_bytes, cost.storage_bytes),
        action_workspace_bytes_per_rhs=cost.apply_workspace_bytes_per_rhs,
        action_cost_exact=cost.exact,
        action_cost_reason=cost.reason,
        quadrature_error_evidence="assembly_report.quadrature_maximum_errors",
        double_layer_jump=jump_d,
        single_layer_neumann_jump=jump_s,
        compatibility_requirement=compatibility,
        gauge=gauge,
        resonance_risk=risk,
        coupling_parameter=coupling_parameter,
        hypersingular_required=False,
        continuum_certified=False,
        non_goals=non_goals,
        formulation_id=formulation_id,
    )


def _empty_coefficients(calderon: ScalarCalderonDP0Galerkin3D, /) -> Array:
    return jnp.zeros((0,), dtype=calderon.space.dtype)


def scalar_interior_dirichlet_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
    *,
    representation: Literal["single-layer", "double-layer"] = "single-layer",
) -> ScalarBoundaryFormulation3D:
    """Prepare ``V phi=g`` or ``(K-I/2)phi=g`` for interior Dirichlet data."""
    prepared = _checked(calderon)
    identity = IdentityLinearOperator(prepared.space)
    if representation == "single-layer":
        operator = prepared.single_layer
        name = "interior-Dirichlet-single-layer"
    elif representation == "double-layer":
        operator = prepared.double_layer - 0.5 * identity
        name = "interior-Dirichlet-double-layer"
    else:
        raise ValueError("Interior Dirichlet representation is invalid.")
    metadata = _metadata(
        prepared,
        operator=operator,
        name=name,
        side="interior",
        boundary_condition="Dirichlet",
        unknown="layer-density-DP0",
        given_data="gamma0(u)-DP0-projection",
        representation=representation,
        compatibility="none",
        gauge="none",
    )
    empty = _empty_coefficients(prepared)
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=identity,
        metadata=metadata,
        robin_alpha=empty,
        robin_beta=empty,
    )


def scalar_exterior_dirichlet_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
    *,
    representation: Literal["single-layer", "double-layer"] = "single-layer",
) -> ScalarBoundaryFormulation3D:
    """Prepare a raw exterior Dirichlet equation with explicit resonance risk."""
    prepared = _checked(calderon)
    identity = IdentityLinearOperator(prepared.space)
    if representation == "single-layer":
        operator = prepared.single_layer
        name = "exterior-Dirichlet-single-layer-raw"
    elif representation == "double-layer":
        operator = prepared.double_layer + 0.5 * identity
        name = "exterior-Dirichlet-double-layer-raw"
    else:
        raise ValueError("Exterior Dirichlet representation is invalid.")
    risk = None
    if prepared.kernel.family == "laplace" and representation == "double-layer":
        risk = (
            "raw-exterior-Laplace-double-layer-has-a-componentwise-constant-"
            "density-nullspace;-use-single-layer"
        )
    metadata = _metadata(
        prepared,
        operator=operator,
        name=name,
        side="exterior",
        boundary_condition="Dirichlet",
        unknown="layer-density-DP0",
        given_data="gamma0(u)-DP0-projection",
        representation=representation,
        compatibility="none",
        gauge=_exterior_condition(prepared),
        resonance_risk=risk,
    )
    empty = _empty_coefficients(prepared)
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=identity,
        metadata=metadata,
        robin_alpha=empty,
        robin_beta=empty,
    )


def scalar_interior_neumann_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
) -> ScalarBoundaryFormulation3D:
    """Prepare ``(I/2+K)u=Vq`` and complete the Laplace constant gauge."""
    prepared = _checked(calderon)
    identity = IdentityLinearOperator(prepared.space)
    operator = 0.5 * identity + prepared.double_layer
    if prepared.kernel.family == "laplace":
        projector = _ComponentMeanProjector3D(
            prepared.space,
            prepared.face_areas,
            prepared.face_component_ids,
            prepared.component_count,
            operator_id=canonical_fingerprint(
                {
                    "kind": "laplace-interior-Neumann-component-gauge-3d-v1",
                    "assembly": prepared.assembly_report.report_id,
                }
            ),
        )
        operator = operator + projector
        compatibility = "area-integral-of-q-is-zero-on-each-closed-component"
        gauge = "area-mean-of-gamma0(u)-is-zero-on-each-component"
    else:
        compatibility = "none-away-from-resonance"
        gauge = "none"
    metadata = _metadata(
        prepared,
        operator=operator,
        name="interior-Neumann-direct-Calderon-trace",
        side="interior",
        boundary_condition="Neumann",
        unknown="gamma0(u)-DP0-trace",
        given_data="gamma1(u)-DP0-flux-density",
        representation="direct-trace",
        compatibility=compatibility,
        gauge=gauge,
    )
    empty = _empty_coefficients(prepared)
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=prepared.single_layer,
        metadata=metadata,
        robin_alpha=empty,
        robin_beta=empty,
    )


def scalar_exterior_neumann_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
) -> ScalarBoundaryFormulation3D:
    """Prepare ``(I/2-K)u=-Vq`` without invoking hypersingular W."""
    prepared = _checked(calderon)
    identity = IdentityLinearOperator(prepared.space)
    operator = 0.5 * identity - prepared.double_layer
    metadata = _metadata(
        prepared,
        operator=operator,
        name="exterior-Neumann-direct-Calderon-trace",
        side="exterior",
        boundary_condition="Neumann",
        unknown="gamma0(u)-DP0-trace",
        given_data="gamma1(u)-DP0-flux-density",
        representation="direct-trace",
        compatibility="none-for-decaying-or-outgoing-exterior-field",
        gauge=_exterior_condition(prepared),
    )
    empty = _empty_coefficients(prepared)
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=-prepared.single_layer,
        metadata=metadata,
        robin_alpha=empty,
        robin_beta=empty,
    )


def scalar_robin_mixed_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    alpha: ArrayLike,
    beta: ArrayLike,
    /,
    *,
    side: ScalarTraceSide3D = "interior",
) -> ScalarBoundaryFormulation3D:
    """Prepare ``alpha*gamma0(S phi)+beta*gamma1(S phi)=f`` in DP0.

    Scalar or facewise DP0 coefficients are admitted. On every face at least
    one coefficient must be nonzero. Pure interior Laplace Neumann data must
    use the named compatibility/gauge formulation instead.
    """
    prepared = _checked(calderon)
    if side not in ("interior", "exterior"):
        raise ValueError("Robin/mixed side must be 'interior' or 'exterior'.")

    def coefficient(value: ArrayLike, name: str, /) -> Array:
        result = jnp.asarray(value, dtype=prepared.space.dtype)
        if result.shape == ():
            result = jnp.full((prepared.face_count,), result, dtype=result.dtype)
        if result.shape != (prepared.face_count,) or not bool(
            jnp.all(jnp.isfinite(result))
        ):
            raise ValueError(f"{name} must be finite scalar or facewise DP0 data.")
        return result

    alpha_ = coefficient(alpha, "alpha")
    beta_ = coefficient(beta, "beta")
    if bool(jnp.any((jnp.abs(alpha_) == 0.0) & (jnp.abs(beta_) == 0.0))):
        raise ValueError("Robin/mixed coefficients cannot both vanish on a face.")
    pure_neumann_component = any(
        bool(
            jnp.all(
                jnp.where(
                    prepared.face_component_ids == component,
                    jnp.abs(alpha_) == 0.0,
                    True,
                )
            )
        )
        for component in range(prepared.component_count)
    )
    if (
        side == "interior"
        and prepared.kernel.family == "laplace"
        and pure_neumann_component
    ):
        raise UnsupportedScalarBoundarySpaceError(
            "A pure interior Laplace Neumann component requires the named "
            "compatibility and component-gauge formulation."
        )
    identity = IdentityLinearOperator(prepared.space)
    jump = prepared.trace_convention.single_layer_neumann_jump(side)
    alpha_operator = DiagonalLinearOperator(alpha_, space=prepared.space)
    beta_operator = DiagonalLinearOperator(beta_, space=prepared.space)
    operator = alpha_operator @ prepared.single_layer + beta_operator @ (
        prepared.adjoint_double_layer + jump * identity
    )
    coefficient_id = canonical_fingerprint(
        {
            "alpha": array_tree_fingerprint(alpha_),
            "beta": array_tree_fingerprint(beta_),
        }
    )
    metadata = _metadata(
        prepared,
        operator=operator,
        name=f"{side}-Robin-mixed-single-layer",
        side=side,
        boundary_condition="Robin-or-facewise-mixed",
        unknown="single-layer-density-DP0",
        given_data="alpha*gamma0(u)+beta*gamma1(u)-DP0-projection",
        representation="single-layer",
        compatibility="none-within-the-declared-non-pure-Neumann-component-envelope",
        gauge=(
            _exterior_condition(prepared)
            if side == "exterior"
            else (
                "nonzero-alpha-on-each-component-removes-the-constant-gauge"
                if prepared.kernel.family == "laplace"
                else "none"
            )
        ),
        coefficient_id=coefficient_id,
    )
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=identity,
        metadata=metadata,
        robin_alpha=alpha_,
        robin_beta=beta_,
    )


def scalar_helmholtz_cfie_formulation_3d(
    calderon: ScalarCalderonDP0Galerkin3D,
    /,
    *,
    eta: float | None = None,
) -> ScalarBoundaryFormulation3D:
    """Prepare exterior Dirichlet Brakhage--Werner CFIE ``I/2+K-i eta V``."""
    prepared = _checked(calderon)
    if prepared.kernel.family != "outgoing-helmholtz":
        raise ValueError("CFIE requires the outgoing Helmholtz kernel family.")
    coupling = prepared.kernel.parameter if eta is None else float(eta)
    if not math.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("CFIE eta must be finite and positive.")
    identity = IdentityLinearOperator(prepared.space)
    operator = (
        0.5 * identity + prepared.double_layer - (1j * coupling) * prepared.single_layer
    )
    metadata = _metadata(
        prepared,
        operator=operator,
        name="exterior-Dirichlet-Brakhage-Werner-CFIE",
        side="exterior",
        boundary_condition="Dirichlet",
        unknown="combined-field-density-DP0",
        given_data="gamma0(u)-DP0-projection",
        representation="combined-field",
        compatibility="none",
        gauge="outgoing-Sommerfeld-radiation-condition",
        coupling_parameter=coupling,
    )
    empty = _empty_coefficients(prepared)
    return ScalarBoundaryFormulation3D(
        calderon=prepared,
        operator=operator,
        data_operator=identity,
        metadata=metadata,
        robin_alpha=empty,
        robin_beta=empty,
    )


__all__ = [
    "ScalarBoundaryFormulation3D",
    "ScalarBoundaryFormulationMetadata3D",
    "ScalarLayerRepresentation3D",
    "scalar_exterior_dirichlet_formulation_3d",
    "scalar_exterior_neumann_formulation_3d",
    "scalar_helmholtz_cfie_formulation_3d",
    "scalar_interior_dirichlet_formulation_3d",
    "scalar_interior_neumann_formulation_3d",
    "scalar_robin_mixed_formulation_3d",
]
