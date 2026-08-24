#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...domain import DomainFunction, GridBatch, PointBatch
from ...operators.differential import laplacian
from ._core import (
    AbstractTrialSpaceAdmissibility,
    TRIAL_SPACE_CERTIFICATE_KEY,
    trial_target_fingerprint,
    TrialSpaceAuditReport,
    TrialSpaceCertificate,
)


def _audit_target_points(
    batch: PointBatch | GridBatch,
    var: str,
    ambient_dimension: int,
    /,
) -> jnp.ndarray:
    payload = batch.points[var]
    if isinstance(payload, tuple):
        if len(payload) != ambient_dimension:
            raise ValueError(
                "Coordinate-separable audit inputs must match the ambient dimension."
            )
        coordinate_axes = tuple(jnp.asarray(field.data) for field in payload)
        meshes = jnp.meshgrid(*coordinate_axes, indexing="ij")
        return jnp.stack(meshes, axis=-1)
    values = jnp.asarray(payload.data)
    if values.ndim < 1 or values.shape[-1] != ambient_dimension:
        raise ValueError("Audit target points must end in the ambient dimension.")
    return values


def _validate_admissibility(
    certificate: TrialSpaceCertificate,
    batch: PointBatch | GridBatch,
    var: str,
    admissibility: AbstractTrialSpaceAdmissibility | None,
    /,
) -> tuple[jnp.ndarray, jnp.ndarray, str | None]:
    if certificate.validity_region == "all-space":
        if admissibility is not None:
            raise ValueError(
                "All-space trial certificates do not accept admissibility reports."
            )
        return jnp.asarray(True), jnp.asarray(True), None

    if not isinstance(admissibility, AbstractTrialSpaceAdmissibility):
        raise TypeError(
            "Off-singular-support trial audits require matching target "
            "admissibility evidence."
        )
    if admissibility.singular_support_id != certificate.singular_support_id:
        raise ValueError(
            "Target admissibility evidence does not match the certified "
            "singular support."
        )
    points = _audit_target_points(batch, var, certificate.ambient_dimension)
    fingerprint = trial_target_fingerprint(points, certificate.ambient_dimension)
    if admissibility.target_fingerprint != fingerprint:
        raise ValueError(
            "Target admissibility evidence does not match the audit batch."
        )
    point_count = int(points.size) // certificate.ambient_dimension
    if admissibility.target_count != point_count:
        raise ValueError(
            "Target admissibility evidence has the wrong number of audit points."
        )
    if not bool(np.asarray(admissibility.pde_membership_valid)):
        raise ValueError(
            "Trial-space audit targets intersect or lie on the certified "
            "singular support."
        )
    return (
        jnp.asarray(admissibility.pde_membership_valid, dtype=bool),
        jnp.asarray(admissibility.accuracy_supported, dtype=bool),
        admissibility.report_id,
    )


def trial_space_certificate(field: DomainFunction, /) -> TrialSpaceCertificate:
    """Return the exact trial-space certificate attached to a bound field."""

    if not isinstance(field, DomainFunction):
        raise TypeError("trial_space_certificate requires a DomainFunction.")
    value = field.metadata.get(TRIAL_SPACE_CERTIFICATE_KEY)
    if not isinstance(value, TrialSpaceCertificate):
        raise TypeError("DomainFunction has no TrialSpaceCertificate.")
    return value


def audit_trial_space(
    field: DomainFunction,
    batch: PointBatch | GridBatch,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "forward",
    tolerance: float | None = None,
    admissibility: AbstractTrialSpaceAdmissibility | None = None,
) -> TrialSpaceAuditReport:
    """Audit one certified PDE residual after validating its target domain."""

    certificate = trial_space_certificate(field)
    if not isinstance(batch, (PointBatch, GridBatch)):
        raise TypeError("audit_trial_space requires a PointBatch or GridBatch.")
    if var not in field.deps:
        raise ValueError(f"Audit variable {var!r} is not a field dependency.")
    (
        pde_membership_valid,
        evaluation_accuracy_supported,
        admissibility_report_id,
    ) = _validate_admissibility(certificate, batch, var, admissibility)

    residual = field
    if certificate.equation_family == "laplace":
        residual = laplacian(field, var=var, mode=mode)
        differential_order = 2
    elif certificate.equation_family == "polyharmonic":
        parameters = dict(certificate.equation_parameters)
        order = int(parameters["order"])
        for _ in range(order):
            residual = laplacian(residual, var=var, mode=mode)
        differential_order = 2 * order
    elif certificate.equation_family == "helmholtz":
        parameters = dict(certificate.equation_parameters)
        wavenumber = float(parameters["wavenumber"])
        residual = laplacian(field, var=var, mode=mode) + (wavenumber**2) * field
        differential_order = 2
    else:
        raise ValueError(f"Unsupported trial-space family {certificate.equation_family!r}.")

    residual_values = jnp.asarray(residual(batch).data)
    field_values = jnp.asarray(field(batch).data)
    absolute = jnp.abs(residual_values)
    finite = jnp.all(jnp.isfinite(residual_values)) & jnp.all(jnp.isfinite(field_values))
    maximum = jnp.max(absolute)
    rms = jnp.sqrt(jnp.mean(absolute * absolute))
    scale = jnp.maximum(jnp.max(jnp.abs(field_values)), 1.0)
    if tolerance is None:
        epsilon = np.finfo(np.dtype(jnp.real(field_values).dtype)).eps
        tolerance_value = (
            2048.0
            * epsilon
            * max(certificate.ambient_dimension, differential_order, 1)
            * scale
        )
    else:
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Audit tolerance must be finite and nonnegative.")
        tolerance_value = jnp.asarray(tolerance_, dtype=jnp.real(field_values).dtype)

    output_count = math.prod(certificate.field_shape) if certificate.field_shape else 1
    point_count = int(field_values.size) // int(output_count)
    point_fingerprint = canonical_fingerprint(
        {
            "kind": "trefftz-audit-points-v1",
            "batch": array_tree_fingerprint(batch),
        }
    )
    return TrialSpaceAuditReport(
        finite=finite,
        maximum_residual=maximum,
        root_mean_square_residual=rms,
        reference_scale=scale,
        tolerance=tolerance_value,
        point_count=point_count,
        certificate_id=certificate.certificate_id,
        point_fingerprint=point_fingerprint,
        pde_membership_valid=pde_membership_valid,
        evaluation_accuracy_supported=evaluation_accuracy_supported,
        admissibility_report_id=admissibility_report_id,
    )


__all__ = ["audit_trial_space", "trial_space_certificate"]
