#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import weakref
from collections.abc import Sequence
from fractions import Fraction
from math import inf, isfinite, nextafter
from numbers import Integral, Real

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import (
    ASTSCertificate,
    ASTSTransferPlan,
    LocalExtractedBernsteinRealization,
    TMesh2D,
)
from ..._interpolation._tspline import _validated_asts_certificate
from ...typing import checked, ConvertibleToArray
from ._patches import AbstractSurfacePatch


def _exact_scalar(value: Fraction | Integral | Real, name: str, /) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, Integral, Real)):
        raise TypeError(f"{name} entries must be exact rational-compatible scalars.")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} entries must be finite.")
    return Fraction.from_float(number)


def _exact_bank(
    values: Sequence[Sequence[Fraction | Integral | Real]],
    name: str,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    rows = tuple(tuple(_exact_scalar(value, name) for value in row) for row in values)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a nonempty rectangular coefficient bank.")
    return rows


def _float_bank(values: tuple[tuple[Fraction, ...], ...], /) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in row] for row in values], dtype=np.float64
    )


# Proved exact transfer chains, keyed by patch object identity. Each entry holds
# a weak reference to its patch (never the patch itself, so the entry cannot keep
# its key alive) and the content identity of the exact chain the proof consumed.
# A hit requires the same live object AND the same recomputed content identity,
# so constructor-bypassing restores or edits that change the chain re-prove it.
_PROVED_TRANSFER_CHAINS: dict[int, tuple[weakref.ref[TSplineSurfacePatch], str]] = {}


def _transfer_chain_identity(patch: TSplineSurfacePatch, /) -> str:
    """Fingerprint the exact inputs the transfer-chain proof consumes."""
    return canonical_fingerprint(
        {
            "kind": "tspline-transfer-chain-proof-input",
            "definition": _surface_definition_id(
                patch.source_mesh.mesh_id,
                patch.mesh.mesh_id,
                tuple(plan.plan_id for plan in patch.transfer_plans),
                tuple(mesh.mesh_id for mesh in patch.transfer_meshes),
                patch.exact_source_control_points,
                patch.exact_source_weights,
                patch.exact_target_homogeneous,
            ),
            "basis_certificates": [
                certificate.certificate_id
                for certificate in patch.transfer_basis_certificates
            ],
        }
    )


def _proved_transfer_chain(patch: TSplineSurfacePatch, identity: str, /) -> bool:
    proved = _PROVED_TRANSFER_CHAINS.get(id(patch))
    return proved is not None and proved[0]() is patch and proved[1] == identity


def _record_proved_transfer_chain(patch: TSplineSurfacePatch, identity: str, /) -> None:
    key = id(patch)

    def _forget(reference: weakref.ref[TSplineSurfacePatch], /) -> None:
        entry = _PROVED_TRANSFER_CHAINS.get(key)
        if entry is not None and entry[0] is reference:
            del _PROVED_TRANSFER_CHAINS[key]

    _PROVED_TRANSFER_CHAINS[key] = (weakref.ref(patch, _forget), identity)


def _exact_digest(values: Sequence[Sequence[Fraction]], /) -> list[list[list[int]]]:
    return [[[value.numerator, value.denominator] for value in row] for row in values]


def _surface_definition_id(
    source_mesh_id: str,
    target_mesh_id: str,
    transfer_plan_ids: Sequence[str],
    transfer_mesh_ids: Sequence[str],
    exact_control_points: Sequence[Sequence[Fraction]],
    exact_weights: Sequence[Fraction],
    exact_target_homogeneous: Sequence[Sequence[Fraction]],
    /,
) -> str:
    """Fingerprint one exact target realization and its retained transfer chain."""
    return canonical_fingerprint(
        {
            "kind": "tspline-surface-source",
            "source_mesh": source_mesh_id,
            "target_mesh": target_mesh_id,
            "transfer_plans": tuple(transfer_plan_ids),
            "transfer_meshes": list(transfer_mesh_ids),
            "exact_control_points": _exact_digest(exact_control_points),
            "exact_weights": [
                [value.numerator, value.denominator] for value in exact_weights
            ],
            "exact_target_homogeneous": _exact_digest(exact_target_homogeneous),
        }
    )


def _source_authority_id(
    source_mesh: TMesh2D,
    exact_control_points: Sequence[Sequence[Fraction]],
    exact_weights: Sequence[Fraction],
    /,
) -> str:
    """Fingerprint the authored root basis and its exact CAD geometry bank."""
    return canonical_fingerprint(
        {
            "kind": "asts2d-root-source-authority",
            "source_mesh": source_mesh.mesh_id,
            "exact_control_points": _exact_digest(exact_control_points),
            "exact_weights": [
                [value.numerator, value.denominator] for value in exact_weights
            ],
        }
    )


def _numeric_bank_id(control_points: object, weights: object, /) -> str:
    points = np.asarray(control_points)
    raw_weights = np.asarray(weights)
    return canonical_fingerprint(
        {
            "kind": "asts2d-numeric-geometry-bank",
            "control_points": array_tree_fingerprint(points),
            "raw_weights": array_tree_fingerprint(raw_weights),
        }
    )


class TSplineSurfacePatch(AbstractSurfacePatch):
    """One source-parametric ASTS surface backed by the shared spline substrate."""

    mesh: TMesh2D
    source_mesh: TMesh2D
    realization: LocalExtractedBernsteinRealization
    transfer_plans: tuple[ASTSTransferPlan, ...]
    transfer_meshes: tuple[TMesh2D, ...]
    transfer_basis_certificates: tuple[ASTSCertificate, ...] = eqx.field(static=True)
    control_points: Array
    weights: Array
    source_basis_certificate: ASTSCertificate = eqx.field(static=True)
    source_numeric_bank_id: str = eqx.field(static=True)
    exact_source_control_points: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    exact_source_weights: tuple[Fraction, ...] = eqx.field(static=True)
    exact_target_homogeneous: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    source_definition_id: str = eqx.field(static=True)
    maximum_homogeneous_realization_error: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: TMesh2D,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        exact_control_points: Sequence[Sequence[Fraction | Integral | Real]]
        | None = None,
        exact_weights: Sequence[Fraction | Integral | Real] | None = None,
        source_mesh: TMesh2D | None = None,
        transfer_plans: Sequence[ASTSTransferPlan] = (),
        source_basis_certificate: ASTSCertificate | None = None,
        basis_certificate: ASTSCertificate | None = None,
        transfer_meshes: Sequence[TMesh2D] | None = None,
        transfer_basis_certificates: Sequence[ASTSCertificate] | None = None,
    ) -> None:
        source_mesh_ = mesh if source_mesh is None else source_mesh
        if not isinstance(source_mesh_, TMesh2D):
            raise TypeError("source_mesh must be a TMesh2D or None.")
        plans = tuple(transfer_plans)
        if any(not isinstance(plan, ASTSTransferPlan) for plan in plans):
            raise TypeError("transfer_plans must contain ASTSTransferPlan values.")
        control_values = np.asarray(control_points, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        if (
            control_values.ndim != 2
            or control_values.shape != (source_mesh_.coefficient_count, 3)
            or not np.all(np.isfinite(control_values))
        ):
            raise ValueError(
                "control_points must be finite with shape (source_anchor_count, 3)."
            )
        if (
            weight_values.shape != (source_mesh_.coefficient_count,)
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values <= 0.0)
        ):
            raise ValueError(
                "weights must be finite, positive, and aligned with source anchors."
            )
        exact_points = (
            _exact_bank(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in row)
                    for row in control_values
                ),
                "exact_control_points",
            )
            if exact_control_points is None
            else _exact_bank(exact_control_points, "exact_control_points")
        )
        exact_weight_values = (
            tuple(Fraction.from_float(float(value)) for value in weight_values)
            if exact_weights is None
            else tuple(_exact_scalar(value, "exact_weights") for value in exact_weights)
        )
        if len(exact_points) != source_mesh_.coefficient_count or any(
            len(row) != 3 for row in exact_points
        ):
            raise ValueError(
                "exact_control_points must align with the source 3D anchor bank."
            )
        if len(exact_weight_values) != source_mesh_.coefficient_count or any(
            value <= 0 for value in exact_weight_values
        ):
            raise ValueError(
                "exact_weights must be positive and align with source anchors."
            )
        if not np.array_equal(
            _float_bank(exact_points), control_values
        ) or not np.array_equal(
            np.asarray([float(value) for value in exact_weight_values], dtype=np.float64),
            weight_values,
        ):
            raise ValueError(
                "Numerical source leaves must be the one binary64 realization of exact source values."
            )
        if plans:
            if source_mesh is None:
                raise ValueError("A refined T-spline requires its exact source mesh.")
            current_id = source_mesh_.mesh_id
            current_count = source_mesh_.coefficient_count
            for plan in plans:
                if (
                    plan.source_mesh_id != current_id
                    or plan.source_count != current_count
                ):
                    raise ValueError(
                        "T-spline transfer plans must form one ordered source-to-target chain."
                    )
                current_id = plan.target_mesh_id
                current_count = plan.target_count
            if current_id != mesh.mesh_id or current_count != mesh.coefficient_count:
                raise ValueError(
                    "T-spline transfer chain must terminate at the declared target mesh."
                )
        elif source_mesh_.mesh_id != mesh.mesh_id:
            raise ValueError(
                "Different source and target T-meshes require exact transfer plans."
            )
        mesh_chain = (
            tuple(transfer_meshes)
            if transfer_meshes is not None
            else ((source_mesh_, mesh) if len(plans) == 1 else (mesh,))
        )
        if (
            len(mesh_chain) != len(plans) + 1
            or any(not isinstance(item, TMesh2D) for item in mesh_chain)
            or mesh_chain[0].mesh_id != source_mesh_.mesh_id
            or mesh_chain[-1].mesh_id != mesh.mesh_id
        ):
            raise ValueError(
                "Exact T-spline transfer meshes must cover the complete chain."
            )
        for index, plan in enumerate(plans):
            if (
                plan.source_mesh_id != mesh_chain[index].mesh_id
                or plan.target_mesh_id != mesh_chain[index + 1].mesh_id
                or plan.source_count != mesh_chain[index].coefficient_count
                or plan.target_count != mesh_chain[index + 1].coefficient_count
            ):
                raise ValueError(
                    "T-spline transfer plan does not match its retained mesh link."
                )
        source_certificate = source_basis_certificate
        if source_mesh_.mesh_id != mesh.mesh_id:
            if source_certificate is None:
                source_certificate = source_mesh_.certify()
            else:
                source_certificate = _validated_asts_certificate(
                    source_mesh_, source_certificate
                )
            if not source_certificate.passed:
                raise ValueError(
                    "The exact source T-mesh requires matching passing ASTS evidence."
                )
        elif source_certificate is not None:
            source_certificate = _validated_asts_certificate(
                source_mesh_, source_certificate
            )
            if not source_certificate.passed:
                raise ValueError(
                    "The source basis certificate does not match its immutable T-mesh."
                )
        if basis_certificate is not None and not isinstance(
            basis_certificate, ASTSCertificate
        ):
            raise TypeError("basis_certificate must be ASTSCertificate or None.")
        realization = (
            mesh.realize()
            if basis_certificate is None
            else mesh.realize(
                tolerance=basis_certificate.tolerance,
                certificate=basis_certificate,
            )
        )
        if source_certificate is None:
            source_certificate = realization.basis_certificate
        elif source_mesh_.mesh_id == mesh.mesh_id and (
            source_certificate.certificate_id
            != realization.basis_certificate.certificate_id
        ):
            raise ValueError(
                "The retained source basis evidence differs from the target realization."
            )
        if transfer_basis_certificates is None:
            transfer_certificates = tuple(
                source_certificate
                if index == 0
                else realization.basis_certificate
                if index == len(mesh_chain) - 1
                else basis_mesh.certify()
                for index, basis_mesh in enumerate(mesh_chain)
            )
        else:
            retained_certificates = tuple(transfer_basis_certificates)
            if len(retained_certificates) != len(mesh_chain):
                raise ValueError(
                    "Retained transfer certificates must cover their exact mesh chain."
                )
            transfer_certificates = tuple(
                _validated_asts_certificate(basis_mesh, certificate)
                for basis_mesh, certificate in zip(
                    mesh_chain, retained_certificates, strict=True
                )
            )
        if (
            len(transfer_certificates) != len(mesh_chain)
            or any(
                not isinstance(certificate, ASTSCertificate)
                or certificate.mesh_id != basis_mesh.mesh_id
                or not certificate.passed
                for certificate, basis_mesh in zip(
                    transfer_certificates, mesh_chain, strict=True
                )
            )
            or transfer_certificates[0].certificate_id
            != source_certificate.certificate_id
            or transfer_certificates[-1].certificate_id
            != realization.basis_certificate.certificate_id
        ):
            raise ValueError(
                "Retained transfer certificates must cover their exact mesh chain."
            )
        exact_points_current = exact_points
        exact_weights_current = exact_weight_values
        for index, plan in enumerate(plans):
            homogeneous = plan.apply_exact_homogeneous_geometry(
                exact_points_current, exact_weights_current
            )
            plan.verify_exact_homogeneous_geometry_preservation(
                mesh_chain[index],
                mesh_chain[index + 1],
                exact_points_current,
                exact_weights_current,
                homogeneous,
                source_certificate=transfer_certificates[index],
                target_certificate=transfer_certificates[index + 1],
            )
            exact_weights_current = tuple(row[-1] for row in homogeneous)
            exact_points_current = tuple(
                tuple(value / weight for value in row[:-1])
                for row, weight in zip(homogeneous, exact_weights_current, strict=True)
            )
        exact_homogeneous = tuple(
            tuple(weight * value for value in point) + (weight,)
            for point, weight in zip(
                exact_points_current, exact_weights_current, strict=True
            )
        )
        numeric_points = jnp.asarray(control_values, dtype=jnp.float64)
        numeric_weights = jnp.asarray(weight_values, dtype=jnp.float64)
        for plan in plans:
            numeric_points, numeric_weights = plan.apply_homogeneous_geometry(
                numeric_points, numeric_weights
            )
        numeric_homogeneous = np.concatenate(
            (
                np.asarray(numeric_points) * np.asarray(numeric_weights)[:, None],
                np.asarray(numeric_weights)[:, None],
            ),
            axis=1,
        )
        if not np.all(np.isfinite(numeric_homogeneous)):
            raise ValueError(
                "Numerical T-spline homogeneous coefficients must remain finite."
            )
        errors = tuple(
            abs(Fraction.from_float(float(numeric_homogeneous[row, column])) - value)
            for row, values in enumerate(exact_homogeneous)
            for column, value in enumerate(values)
        )
        maximum_error = max(errors, default=Fraction(0))
        error_float = float(maximum_error)
        if Fraction.from_float(error_float) < maximum_error:
            error_float = nextafter(error_float, inf)
        exact_identity = _surface_definition_id(
            source_mesh_.mesh_id,
            mesh.mesh_id,
            tuple(plan.plan_id for plan in plans),
            tuple(basis_mesh.mesh_id for basis_mesh in mesh_chain),
            exact_points,
            exact_weight_values,
            exact_homogeneous,
        )
        self.mesh = mesh
        self.source_mesh = source_mesh_
        self.realization = realization
        self.transfer_plans = plans
        self.transfer_meshes = mesh_chain
        self.transfer_basis_certificates = transfer_certificates
        self.control_points = jnp.asarray(control_values, dtype=jnp.float64)
        self.weights = jnp.asarray(weight_values, dtype=jnp.float64)
        self.source_basis_certificate = source_certificate
        self.source_numeric_bank_id = _numeric_bank_id(control_values, weight_values)
        self.exact_source_control_points = exact_points
        self.exact_source_weights = exact_weight_values
        self.exact_target_homogeneous = exact_homogeneous
        self.source_definition_id = exact_identity
        self.maximum_homogeneous_realization_error = error_float

    @property
    def parameter_domain(self, /) -> tuple[tuple[float, float], tuple[float, float]]:
        u0, u1, v0, v1 = self.mesh.parameter_domain
        return ((float(u0), float(v0)), (float(u1), float(v1)))

    @property
    def source_authority_id(self, /) -> str:
        """Identify the exact authored root source independent of target refinement."""
        return _source_authority_id(
            self.source_mesh,
            self.exact_source_control_points,
            self.exact_source_weights,
        )

    def validate_source_bank(self, /) -> None:
        """Require concrete control and weight leaves to match this exact source bank."""
        points, raw_weights = self.control_points, self.weights
        if isinstance(points, jax_core.Tracer) or isinstance(
            raw_weights, jax_core.Tracer
        ):
            return
        if _numeric_bank_id(points, raw_weights) != self.source_numeric_bank_id:
            raise ValueError(
                "Concrete source-bank edits require a newly authored T-spline patch and source revision."
            )
        if not np.array_equal(
            np.asarray(points), _float_bank(self.exact_source_control_points)
        ) or not np.array_equal(
            np.asarray(raw_weights),
            np.asarray([float(value) for value in self.exact_source_weights]),
        ):
            raise ValueError(
                "Concrete T-spline source bank does not realize its retained exact data."
            )

    def validate_exact_transfer_chain(self, /) -> None:
        """Revalidate the retained exact source-to-target geometry chain."""
        self.validate_source_bank()
        chain_identity = _transfer_chain_identity(self)
        if _proved_transfer_chain(self, chain_identity):
            return
        if len(self.transfer_meshes) != len(self.transfer_plans) + 1 or len(
            self.transfer_basis_certificates
        ) != len(self.transfer_meshes):
            raise ValueError(
                "Retained T-spline transfer evidence does not cover its mesh chain."
            )
        points = self.exact_source_control_points
        weights = self.exact_source_weights
        for index, plan in enumerate(self.transfer_plans):
            homogeneous = plan.apply_exact_homogeneous_geometry(points, weights)
            plan.verify_exact_homogeneous_geometry_preservation(
                self.transfer_meshes[index],
                self.transfer_meshes[index + 1],
                points,
                weights,
                homogeneous,
                source_certificate=self.transfer_basis_certificates[index],
                target_certificate=self.transfer_basis_certificates[index + 1],
            )
            weights = tuple(row[-1] for row in homogeneous)
            points = tuple(
                tuple(value / weight for value in row[:-1])
                for row, weight in zip(homogeneous, weights, strict=True)
            )
        expected_target = tuple(
            tuple(weight * value for value in point) + (weight,)
            for point, weight in zip(points, weights, strict=True)
        )
        if expected_target != self.exact_target_homogeneous:
            raise ValueError(
                "Retained exact T-spline target bank differs from its transfer chain."
            )
        _record_proved_transfer_chain(self, chain_identity)

    @property
    def basis_certificate(self, /) -> ASTSCertificate:
        return self.realization.basis_certificate

    def _current_geometry(self, /) -> tuple[Array, Array]:
        self.validate_source_bank()
        control_points, weights = self.control_points, self.weights
        for plan in self.transfer_plans:
            control_points, weights = plan.apply_homogeneous_geometry(
                control_points, weights
            )
        return control_points, weights

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 0 or values.shape[-1] != 2:
            raise ValueError("T-spline parameters must have shape (..., 2).")
        values = eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), "T-spline parameters must be finite."
        )
        control_points, weights = self._current_geometry()
        leading = values.shape[:-1]
        points = values.reshape((-1, 2))
        jet = self.realization.evaluate_rational(
            points, control_points, weights, maximum_order=0
        )
        return jet.values.reshape((*leading, 3))

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        self.validate_source_bank()
        self.validate_parameter_box(parameter_box)
        certificate = self.basis_certificate
        return (
            certificate.mesh_id == self.mesh.mesh_id
            and certificate.passed
            and certificate.trace_continuous
            and certificate.continuity_order >= 1
        )

    def is_c1_along(
        self,
        fixed_axis: int,
        fixed_value: float,
        first: float,
        last: float,
        /,
    ) -> bool:
        if isinstance(fixed_axis, bool) or not isinstance(fixed_axis, Integral):
            raise TypeError("fixed_axis must be an integer.")
        axis = int(fixed_axis)
        if axis not in (0, 1):
            raise ValueError("fixed_axis must be zero or one.")
        coordinates = (fixed_value, first, last)
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in coordinates
        ):
            raise TypeError("T-spline continuity coordinates must be real scalars.")
        fixed, first_, last_ = (float(value) for value in coordinates)
        if any(not isfinite(value) for value in (fixed, first_, last_)):
            raise ValueError("T-spline continuity coordinates must be finite.")
        if first_ >= last_:
            raise ValueError("The T-spline continuity range must be increasing.")
        lower, upper = self.parameter_domain
        variable_axis = 1 - axis
        if (
            fixed < lower[axis]
            or fixed > upper[axis]
            or first_ < lower[variable_axis]
            or last_ > upper[variable_axis]
        ):
            raise ValueError(
                "The T-spline continuity range must lie in its source domain."
            )
        self.validate_source_bank()
        certificate = self.basis_certificate
        return (
            certificate.mesh_id == self.mesh.mesh_id
            and certificate.passed
            and certificate.trace_continuous
            and certificate.continuity_order >= 1
        )


__all__ = ["TSplineSurfacePatch"]
