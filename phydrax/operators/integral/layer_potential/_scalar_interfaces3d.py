#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    BlockLinearOperator,
    BlockSpace,
    estimate_operator_action_cost,
    IdentityLinearOperator,
)
from ._scalar_calderon3d import ScalarCalderonDP0Galerkin3D


ScalarTransmissionSide3D = Literal["minus", "plus"]
ScalarTransmissionOrientation3D = Literal["calderon", "reversed"]


class ScalarTransmissionMaterial3D(StrictModule, NonTrainableState):
    """One scalar material and its boundary calculus.

    ``flux_coefficient`` is the positive coefficient in the conormal
    ``a * d_n u``.  The Calderón kernel supplies the homogeneous PDE in that
    material; its parameter may differ from the other material's parameter.
    """

    name: str = eqx.field(static=True)
    flux_coefficient: float = eqx.field(static=True)
    calderon: ScalarCalderonDP0Galerkin3D
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        calderon: ScalarCalderonDP0Galerkin3D,
        /,
        *,
        flux_coefficient: float = 1.0,
    ):
        name_ = str(name)
        coefficient = float(flux_coefficient)
        if not name_:
            raise ValueError("Scalar transmission materials require a name.")
        if not isinstance(calderon, ScalarCalderonDP0Galerkin3D):
            raise TypeError("calderon must be ScalarCalderonDP0Galerkin3D.")
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("flux_coefficient must be finite and positive.")
        self.name = name_
        self.flux_coefficient = coefficient
        self.calderon = calderon
        self.material_id = canonical_fingerprint(
            {
                "kind": "scalar-transmission-material-3d-v1",
                "name": name_,
                "flux_coefficient": coefficient,
                "assembly_report": calderon.assembly_report.report_id,
            }
        )


class ScalarTransmissionSideConvention3D(StrictModule, NonTrainableState):
    """Oriented two-side convention for a matching closed interface.

    The oriented normal ``n`` points from ``minus`` to ``plus``. Both normal
    derivatives use that same ``n``; the plus-domain outward normal on its
    inner boundary is therefore ``-n``. ``normal_sign`` relates ``n`` to the
    source normal used to assemble K/K': +1 for the landed Calderón orientation
    and -1 after an explicit side reversal.
    """

    normal_orientation: ScalarTransmissionOrientation3D = eqx.field(static=True)
    normal_sign: int = eqx.field(static=True)
    oriented_normal: str = eqx.field(static=True)
    minus_domain_outward_normal: str = eqx.field(static=True)
    plus_domain_outward_normal: str = eqx.field(static=True)
    derivative_convention: str = eqx.field(static=True)
    dirichlet_jump: str = eqx.field(static=True)
    weighted_flux_jump: str = eqx.field(static=True)
    unbounded_side: ScalarTransmissionSide3D = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self, normal_orientation: ScalarTransmissionOrientation3D = "calderon", /
    ):
        if normal_orientation not in ("calderon", "reversed"):
            raise ValueError("normal_orientation must be 'calderon' or 'reversed'.")
        sign = 1 if normal_orientation == "calderon" else -1
        unbounded: ScalarTransmissionSide3D = (
            "plus" if normal_orientation == "calderon" else "minus"
        )
        self.normal_orientation = normal_orientation
        self.normal_sign = sign
        self.oriented_normal = (
            "calderon-source-normal-minus-to-plus"
            if sign == 1
            else "negative-calderon-source-normal-minus-to-plus"
        )
        self.minus_domain_outward_normal = "+n"
        self.plus_domain_outward_normal = "-n"
        self.derivative_convention = "q_side=d(u_side)/d(n)-on-both-sides"
        self.dirichlet_jump = "u_plus-u_minus"
        self.weighted_flux_jump = "a_plus*q_plus-a_minus*q_minus"
        self.unbounded_side = unbounded
        self.convention_id = canonical_fingerprint(
            {
                "kind": "scalar-transmission-side-convention-3d-v1",
                "normal_sign": sign,
                "derivative": self.derivative_convention,
                "dirichlet_jump": self.dirichlet_jump,
                "weighted_flux_jump": self.weighted_flux_jump,
                "unbounded_side": unbounded,
            }
        )

    def reversed(self) -> ScalarTransmissionSideConvention3D:
        """Return the convention obtained by swapping sides and negating n."""

        return ScalarTransmissionSideConvention3D(
            "reversed" if self.normal_orientation == "calderon" else "calderon"
        )


class ScalarCauchyTraceBundle3D(StrictModule, NonTrainableState):
    """DP0 Dirichlet and same-oriented-normal traces for one material side."""

    dirichlet: Array
    normal_derivative: Array
    side: ScalarTransmissionSide3D = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        dirichlet: ArrayLike,
        normal_derivative: ArrayLike,
        /,
        *,
        side: ScalarTransmissionSide3D,
        material_id: str,
    ):
        if side not in ("minus", "plus"):
            raise ValueError("Cauchy bundle side must be 'minus' or 'plus'.")
        material = str(material_id)
        if not material:
            raise ValueError("Cauchy bundles require a material identity.")
        dtype = jnp.result_type(dirichlet, normal_derivative)
        value = jnp.asarray(dirichlet, dtype=dtype)
        derivative = jnp.asarray(normal_derivative, dtype=dtype)
        if value.ndim != 1 or derivative.shape != value.shape:
            raise ValueError("Cauchy traces must share one rank-1 DP0 shape.")
        self.dirichlet = value
        self.normal_derivative = derivative
        self.side = side
        self.material_id = material


class ScalarTransmissionData3D(StrictModule, NonTrainableState):
    """Four block right-hand sides for the direct multitrace system.

    The first two blocks are forcing in the minus/plus Calderón relations. The
    last two prescribe ``u_plus-u_minus`` and
    ``a_plus*q_plus-a_minus*q_minus``. Homogeneous physical transmission uses
    four zero blocks; exposing all four supports exact manufactured-block tests.
    """

    minus_calderon: Array
    plus_calderon: Array
    dirichlet_jump: Array
    weighted_flux_jump: Array
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        minus_calderon: ArrayLike,
        plus_calderon: ArrayLike,
        dirichlet_jump: ArrayLike,
        weighted_flux_jump: ArrayLike,
        /,
    ):
        dtype = jnp.result_type(
            minus_calderon,
            plus_calderon,
            dirichlet_jump,
            weighted_flux_jump,
        )
        values = tuple(
            jnp.asarray(value, dtype=dtype)
            for value in (
                minus_calderon,
                plus_calderon,
                dirichlet_jump,
                weighted_flux_jump,
            )
        )
        if values[0].ndim != 1 or any(value.shape != values[0].shape for value in values):
            raise ValueError("Transmission data blocks must share one rank-1 shape.")
        (
            self.minus_calderon,
            self.plus_calderon,
            self.dirichlet_jump,
            self.weighted_flux_jump,
        ) = values
        self.data_id = canonical_fingerprint(
            {
                "kind": "scalar-transmission-data-3d-v1",
                "blocks": array_tree_fingerprint(values),
            }
        )

    @property
    def blocks(self) -> tuple[Array, Array, Array, Array]:
        return (
            self.minus_calderon,
            self.plus_calderon,
            self.dirichlet_jump,
            self.weighted_flux_jump,
        )


class ScalarTransmissionAssemblyReport3D(StrictModule, NonTrainableState):
    """Accuracy, resource, uniqueness, radiation, and scope evidence."""

    pdes: tuple[str, str] = eqx.field(static=True)
    kernel_families: tuple[str, str] = eqx.field(static=True)
    material_ids: tuple[str, str] = eqx.field(static=True)
    assembly_report_ids: tuple[str, str] = eqx.field(static=True)
    interface_source_id: str = eqx.field(static=True)
    side_convention_id: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    flux_continuity: str = eqx.field(static=True)
    flux_coefficients: tuple[float, float] = eqx.field(static=True)
    unbounded_side: ScalarTransmissionSide3D = eqx.field(static=True)
    unbounded_condition: str = eqx.field(static=True)
    resonance_evidence: str = eqx.field(static=True)
    nullspace_evidence: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    quadrature_maximum_errors: Array
    quadrature_evaluations: Array
    precision_policy_ids: tuple[str, str] = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    action_cost_exact: bool = eqx.field(static=True)
    action_cost_reason: str = eqx.field(static=True)
    finite: Array
    accuracy_supported: Array
    continuum_certified: bool = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class ScalarTransmissionFormulation3D(StrictModule, NonTrainableState):
    """Prepared two-material direct Calderón multitrace block.

    Unknown order is ``(u_minus, q_minus, u_plus, q_plus)``. With K assembled
    using its source normal and ``s=normal_sign``, the rows are

    ``(I/2+s K_minus)u_minus - V_minus q_minus``,
    ``(I/2-s K_plus)u_plus + V_plus q_plus``,
    ``u_plus-u_minus``, and
    ``a_plus q_plus-a_minus q_minus``.

    This uses only V and K and therefore never pretends DP0 supplies W.
    """

    minus: ScalarTransmissionMaterial3D
    plus: ScalarTransmissionMaterial3D
    convention: ScalarTransmissionSideConvention3D
    operator: BlockLinearOperator
    report: ScalarTransmissionAssemblyReport3D
    formulation_id: str = eqx.field(static=True)

    def right_hand_side(
        self,
        data: ScalarTransmissionData3D
        | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
        /,
    ) -> tuple[Array, Array, Array, Array]:
        values = data.blocks if isinstance(data, ScalarTransmissionData3D) else data
        if not isinstance(values, tuple) or len(values) != 4:
            raise TypeError(
                "Transmission data must be ScalarTransmissionData3D or four blocks."
            )
        return self.operator.target.validate(values)

    def bundles(
        self, value: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], /
    ) -> tuple[ScalarCauchyTraceBundle3D, ScalarCauchyTraceBundle3D]:
        u_minus, q_minus, u_plus, q_plus = self.operator.source.validate(value)
        return (
            ScalarCauchyTraceBundle3D(
                u_minus,
                q_minus,
                side="minus",
                material_id=self.minus.material_id,
            ),
            ScalarCauchyTraceBundle3D(
                u_plus,
                q_plus,
                side="plus",
                material_id=self.plus.material_id,
            ),
        )

    def reversed(self) -> ScalarTransmissionFormulation3D:
        """Swap materials/sides and negate the oriented normal exactly."""

        return scalar_transmission_formulation_3d(
            self.plus,
            self.minus,
            convention=self.convention.reversed(),
        )


def _unbounded_condition(material: ScalarTransmissionMaterial3D, /) -> str:
    family = material.calderon.kernel.family
    if family == "laplace":
        return "harmonic-field-decays-at-infinity"
    if family == "modified-helmholtz":
        return "modified-Helmholtz-field-decays-exponentially-at-infinity"
    return "outgoing-Sommerfeld-radiation-condition"


def _validate_matching_materials(
    minus: ScalarTransmissionMaterial3D,
    plus: ScalarTransmissionMaterial3D,
    /,
) -> None:
    if not isinstance(minus, ScalarTransmissionMaterial3D) or not isinstance(
        plus, ScalarTransmissionMaterial3D
    ):
        raise TypeError("minus and plus must be ScalarTransmissionMaterial3D values.")
    if minus.name == plus.name:
        raise ValueError("Transmission sides must name distinct materials.")
    left = minus.calderon
    right = plus.calderon
    reports = (left.assembly_report, right.assembly_report)
    if any(not bool(report.finite & report.accuracy_supported) for report in reports):
        raise ValueError("Both scalar Calderón assemblies must be finite and supported.")
    if left.component_count != 1 or right.component_count != 1:
        raise ValueError(
            "Two-domain transmission requires one connected closed interface."
        )
    if left._binding.region.feature_id != right._binding.region.feature_id:
        raise ValueError("Transmission Calderón blocks must share the exact interface.")
    if left.face_count != right.face_count or not left.space.compatible(right.space):
        raise ValueError("Transmission Calderón blocks must share matching DP0 routes.")
    if left.kernel.family != right.kernel.family:
        raise ValueError(
            "Transmission materials must use one kernel family; material parameters may differ."
        )
    if left.trace_convention.convention_id != right.trace_convention.convention_id:
        raise ValueError("Transmission Calderón trace conventions do not match.")
    if any(report.hypersingular_supported for report in reports):
        raise ValueError("The direct DP0 transmission route must not claim W support.")


def scalar_transmission_formulation_3d(
    minus: ScalarTransmissionMaterial3D,
    plus: ScalarTransmissionMaterial3D,
    /,
    *,
    convention: ScalarTransmissionSideConvention3D | None = None,
    formulation: str = "two-sided-direct-Calderon-multitrace",
) -> ScalarTransmissionFormulation3D:
    """Prepare the matching closed-interface V/K transmission block.

    The direct multitrace route avoids raw single-layer inversion and its closed
    Helmholtz interior-resonance failure. Other formulations are rejected before
    any linear-solver preparation because DP0 W/P1 support is not available.
    """

    if formulation != "two-sided-direct-Calderon-multitrace":
        raise ValueError(
            "Only the two-sided direct Calderón V/K multitrace formulation is supported; "
            "hypersingular, single-trace, and nonmatching routes are unavailable."
        )
    _validate_matching_materials(minus, plus)
    convention_ = (
        ScalarTransmissionSideConvention3D() if convention is None else convention
    )
    if not isinstance(convention_, ScalarTransmissionSideConvention3D):
        raise TypeError("convention must be ScalarTransmissionSideConvention3D or None.")

    space = minus.calderon.space
    identity = IdentityLinearOperator(space)
    sign = float(convention_.normal_sign)
    minus_relation = 0.5 * identity + sign * minus.calderon.double_layer
    plus_relation = 0.5 * identity - sign * plus.calderon.double_layer
    block_space = BlockSpace(
        (space, space, space, space),
        names=("u_minus", "q_minus", "u_plus", "q_plus"),
    )
    operator_id = canonical_fingerprint(
        {
            "kind": "scalar-transmission-direct-multitrace-3d-v1",
            "minus": minus.material_id,
            "plus": plus.material_id,
            "convention": convention_.convention_id,
        }
    )
    operator = BlockLinearOperator(
        (
            (minus_relation, -minus.calderon.single_layer, None, None),
            (None, None, plus_relation, plus.calderon.single_layer),
            (-identity, None, identity, None),
            (
                None,
                -minus.flux_coefficient * identity,
                None,
                plus.flux_coefficient * identity,
            ),
        ),
        source=block_space,
        target=block_space,
        operator_id=operator_id,
    )
    cost = estimate_operator_action_cost(operator)
    left_report = minus.calderon.assembly_report
    right_report = plus.calderon.assembly_report
    unbounded_material = plus if convention_.unbounded_side == "plus" else minus
    family = unbounded_material.calderon.kernel.family
    resonance = (
        "two-sided-direct-Calderon-multitrace-uses-no-raw-single-layer-inverse;"
        "physical-outgoing-transmission-uniqueness-assumed;discrete-conditioning-not-certified"
        if family == "outgoing-helmholtz"
        else "not-applicable-to-static-or-coercive-kernel"
    )
    nullspace = (
        "positive-material-flux-coefficients-plus-unbounded-decay/radiation-remove-"
        "the-closed-interface-constant-nullspace"
    )
    errors = jnp.stack(
        (
            left_report.quadrature_maximum_errors,
            right_report.quadrature_maximum_errors,
        )
    )
    evaluations = jnp.stack(
        (
            left_report.quadrature_evaluations,
            right_report.quadrature_evaluations,
        )
    )
    finite = left_report.finite & right_report.finite & jnp.all(jnp.isfinite(errors))
    supported = left_report.accuracy_supported & right_report.accuracy_supported & finite
    non_goals = (
        "no-continuum-discretization-error-certificate",
        "no-hypersingular-W-or-P1-H1/2-route",
        "no-nonmatching-mortar-or-junction-interface",
        "no-automatic-discrete-condition-number-or-resonance-certificate",
        "no-dense-materialization",
    )
    report_id = canonical_fingerprint(
        {
            "kind": "scalar-transmission-assembly-report-3d-v1",
            "operator": operator_id,
            "assemblies": (left_report.report_id, right_report.report_id),
            "errors": array_tree_fingerprint(errors),
            "unbounded_condition": _unbounded_condition(unbounded_material),
        }
    )
    report = ScalarTransmissionAssemblyReport3D(
        pdes=(minus.calderon.kernel.pde, plus.calderon.kernel.pde),
        kernel_families=(
            minus.calderon.kernel.family,
            plus.calderon.kernel.family,
        ),
        material_ids=(minus.material_id, plus.material_id),
        assembly_report_ids=(left_report.report_id, right_report.report_id),
        interface_source_id=minus.calderon._binding.region.feature_id,
        side_convention_id=convention_.convention_id,
        normal_convention=convention_.oriented_normal,
        flux_continuity=convention_.weighted_flux_jump,
        flux_coefficients=(minus.flux_coefficient, plus.flux_coefficient),
        unbounded_side=convention_.unbounded_side,
        unbounded_condition=_unbounded_condition(unbounded_material),
        resonance_evidence=resonance,
        nullspace_evidence=nullspace,
        formulation=formulation,
        quadrature_maximum_errors=errors,
        quadrature_evaluations=evaluations,
        precision_policy_ids=(
            left_report.precision_policy_id,
            right_report.precision_policy_id,
        ),
        preparation_workspace_bytes=(
            left_report.preparation_workspace_bytes
            + right_report.preparation_workspace_bytes
        ),
        resident_bytes=max(
            left_report.resident_bytes + right_report.resident_bytes,
            cost.storage_bytes,
        ),
        action_workspace_bytes_per_rhs=cost.apply_workspace_bytes_per_rhs,
        action_cost_exact=cost.exact,
        action_cost_reason=cost.reason,
        finite=finite,
        accuracy_supported=supported,
        continuum_certified=False,
        non_goals=non_goals,
        report_id=report_id,
    )
    formulation_id = canonical_fingerprint(
        {
            "kind": "scalar-transmission-formulation-3d-v1",
            "operator": operator.operator_id,
            "report": report.report_id,
        }
    )
    return ScalarTransmissionFormulation3D(
        minus=minus,
        plus=plus,
        convention=convention_,
        operator=operator,
        report=report,
        formulation_id=formulation_id,
    )


__all__ = [
    "ScalarCauchyTraceBundle3D",
    "ScalarTransmissionAssemblyReport3D",
    "ScalarTransmissionData3D",
    "ScalarTransmissionFormulation3D",
    "ScalarTransmissionMaterial3D",
    "ScalarTransmissionOrientation3D",
    "ScalarTransmissionSide3D",
    "ScalarTransmissionSideConvention3D",
    "scalar_transmission_formulation_3d",
]
