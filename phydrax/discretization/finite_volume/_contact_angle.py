#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ContactAngleStatus(IntEnum):
    """Validity status for an explicit contact-angle reconstruction."""

    SUCCESS = 0
    FAILED = 1


class ContactAngleCondition(StrictModule, NonTrainableState):
    """Explicit contact-angle policy for one tagged embedded wall."""

    body_tag: int = eqx.field(static=True)
    angle: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_tag: int,
        angle: float,
        tolerance: float,
        condition_id: str,
        /,
    ):
        if not isinstance(body_tag, (int, np.integer)) or isinstance(
            body_tag, (bool, np.bool_)
        ):
            raise TypeError("Contact-angle body_tag must be a nonnegative integer.")
        tag = int(body_tag)
        if tag < 0:
            raise ValueError("Contact-angle body_tag must be a nonnegative integer.")

        angle_ = float(angle)
        if not np.isfinite(angle_) or not 0.0 < angle_ < float(np.pi):
            raise ValueError(
                "Contact-angle angle must be finite and strictly in (0, pi)."
            )

        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Contact-angle tolerance must be finite and nonnegative.")

        identifier = str(condition_id)
        if not identifier:
            raise ValueError("Contact-angle condition_id must be non-empty.")

        self.body_tag = tag
        self.angle = angle_
        self.tolerance = tolerance_
        self.condition_id = identifier


class ContactAngleEvidence(StrictModule, NonTrainableState):
    """Numerical evidence for a reconstructed wall/interface normal."""

    declared_cosine: Array
    realized_cosine: Array
    cosine_defect: Array
    angle_defect: Array
    tolerance: Array
    passed: Array
    status: Array


class ContactAngleReconstructionResult(StrictModule, NonTrainableState):
    """Explicit contact-angle normal and its identity/evidence payload."""

    normal: Array
    evidence: ContactAngleEvidence
    status: Array
    angle_defect: Array
    body_tag: int = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    plic_id: str = eqx.field(static=True)


class EmbeddedBoundaryContactAngleSet(StrictModule, NonTrainableState):
    """Complete explicit contact-angle policy coverage for tagged walls."""

    body_tags: tuple[int, ...] = eqx.field(static=True)
    conditions: tuple[ContactAngleCondition, ...]
    geometry_id: str = eqx.field(static=True)
    plic_id: str = eqx.field(static=True)
    contact_angle_set_id: str = eqx.field(static=True)
    status: ContactAngleStatus = eqx.field(static=True)

    def __init__(
        self,
        conditions: Mapping[int, ContactAngleCondition] | Sequence[ContactAngleCondition],
        /,
        *,
        geometry_id: str,
        plic_id: str,
    ):
        geometry = str(geometry_id)
        plic = str(plic_id)
        if not geometry:
            raise ValueError("Contact-angle geometry_id must be non-empty.")
        if not plic:
            raise ValueError("Contact-angle plic_id must be non-empty.")

        if isinstance(conditions, Mapping):
            supplied = list(conditions.items())
            if not supplied:
                raise ValueError("Contact-angle policy coverage must not be empty.")
            normalized: dict[int, ContactAngleCondition] = {}
            for raw_tag, condition in supplied:
                if not isinstance(raw_tag, (int, np.integer)) or isinstance(
                    raw_tag, (bool, np.bool_)
                ):
                    raise TypeError(
                        "Contact-angle policy body tags must be nonnegative integers."
                    )
                tag = int(raw_tag)
                if tag < 0:
                    raise ValueError(
                        "Contact-angle policy body tags must be nonnegative integers."
                    )
                if tag in normalized:
                    raise ValueError("Contact-angle policy body tags must be unique.")
                if not isinstance(condition, ContactAngleCondition):
                    raise TypeError(
                        "Contact-angle policies must be ContactAngleCondition values."
                    )
                if condition.body_tag != tag:
                    raise ValueError(
                        "Contact-angle policy key must match condition.body_tag."
                    )
                normalized[tag] = condition
        elif isinstance(conditions, Sequence) and not isinstance(
            conditions, (str, bytes)
        ):
            if not conditions:
                raise ValueError("Contact-angle policy coverage must not be empty.")
            normalized = {}
            for condition in conditions:
                if not isinstance(condition, ContactAngleCondition):
                    raise TypeError(
                        "Contact-angle policies must be ContactAngleCondition values."
                    )
                if condition.body_tag in normalized:
                    raise ValueError("Contact-angle policy body tags must be unique.")
                normalized[condition.body_tag] = condition
        else:
            raise TypeError(
                "conditions must map body tags to ContactAngleCondition values or be "
                "a sequence of ContactAngleCondition values."
            )

        tags = tuple(sorted(normalized))
        policies = tuple(normalized[tag] for tag in tags)
        self.body_tags = tags
        self.conditions = policies
        self.geometry_id = geometry
        self.plic_id = plic
        self.status = ContactAngleStatus.SUCCESS
        self.contact_angle_set_id = canonical_fingerprint(
            {
                "kind": "embedded-boundary-contact-angle-set",
                "geometry": geometry,
                "plic": plic,
                "conditions": [
                    {
                        "body_tag": condition.body_tag,
                        "condition_id": condition.condition_id,
                        "angle": condition.angle,
                        "tolerance": condition.tolerance,
                    }
                    for condition in policies
                ],
            }
        )

    def validate_body_tags(self, body_tags: ArrayLike, /) -> None:
        """Require exact policy coverage of the supplied embedded body tags."""

        values = np.asarray(body_tags)
        if values.ndim == 0 or values.dtype.kind not in "iu":
            raise TypeError("Embedded body tags must be an integer array.")
        if values.dtype.kind == "b" or np.any(values < 0):
            raise ValueError("Embedded body tags must be nonnegative integers.")
        observed = frozenset(int(value) for value in values.reshape(-1))
        expected = frozenset(self.body_tags)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise ValueError(
                "Contact-angle body-tag policy coverage must match exactly; "
                f"missing policy tags={missing!r}; extra policy tags={extra!r}."
            )

    def validate_bindings(self, geometry_id: str, plic_id: str, /) -> None:
        """Reject use with stale geometry or PLIC reconstruction identities."""

        geometry = str(geometry_id)
        plic = str(plic_id)
        if geometry != self.geometry_id:
            raise ValueError(
                "Contact-angle policies belong to stale or different geometry_id."
            )
        if plic != self.plic_id:
            raise ValueError(
                "Contact-angle policies belong to stale or different plic_id."
            )

    def condition_for(self, body_tag: int, /) -> ContactAngleCondition:
        """Return the explicit condition for one body tag."""

        if not isinstance(body_tag, (int, np.integer)) or isinstance(
            body_tag, (bool, np.bool_)
        ):
            raise TypeError("body_tag must be a nonnegative integer.")
        tag = int(body_tag)
        if tag not in self.body_tags:
            raise KeyError(f"No explicit contact-angle condition for body tag {tag}.")
        return self.conditions[self.body_tags.index(tag)]

    def reconstruct(
        self,
        plic_normal: ArrayLike,
        wall_normal: ArrayLike,
        body_tag: int,
        /,
    ) -> ContactAngleReconstructionResult:
        """Reconstruct a normal while carrying this set's geometry/PLIC identity."""

        condition = self.condition_for(body_tag)
        return reconstruct_wall_interface_normal(
            plic_normal,
            wall_normal,
            condition,
            geometry_id=self.geometry_id,
            plic_id=self.plic_id,
        )


def reconstruct_wall_interface_normal(
    plic_normal: ArrayLike,
    wall_normal: ArrayLike,
    condition: ContactAngleCondition,
    /,
    *,
    geometry_id: str | None = None,
    plic_id: str | None = None,
) -> ContactAngleReconstructionResult:
    """Rotate a PLIC normal in the wall tangent plane to an explicit angle.

    The returned normal has ``dot(normal, wall_normal) == cos(condition.angle)``
    up to the supplied tolerance.  The tangential direction is inherited from the
    PLIC normal, so the contact-angle policy changes only its wall-normal component.
    """

    if not isinstance(condition, ContactAngleCondition):
        raise TypeError("condition must be a ContactAngleCondition.")
    if (geometry_id is None) != (plic_id is None):
        raise ValueError("geometry_id and plic_id must be supplied together.")
    geometry = "" if geometry_id is None else str(geometry_id)
    plic = "" if plic_id is None else str(plic_id)
    if geometry_id is not None and not geometry:
        raise ValueError("geometry_id must be non-empty when supplied.")
    if plic_id is not None and not plic:
        raise ValueError("plic_id must be non-empty when supplied.")

    plic_ = jnp.asarray(plic_normal)
    wall_ = jnp.asarray(wall_normal)
    if plic_.ndim == 0 or wall_.ndim == 0 or plic_.shape[-1] != 2 or wall_.shape[-1] != 2:
        raise ValueError(
            "plic_normal and wall_normal must have matching trailing shape (..., 2)."
        )
    try:
        plic_, wall_ = jnp.broadcast_arrays(plic_, wall_)
    except ValueError as error:
        raise ValueError(
            "plic_normal and wall_normal must be broadcast-compatible (..., 2)."
        ) from error
    dtype = jnp.result_type(plic_, wall_, jnp.float32)
    plic_ = plic_.astype(dtype)
    wall_ = wall_.astype(dtype)

    wall_squared_norm = oe.contract("...i,...i->...", wall_, wall_)
    wall_squared_norm = eqx.error_if(
        wall_squared_norm,
        jnp.any(~jnp.isfinite(wall_squared_norm) | (wall_squared_norm <= 0.0)),
        "wall_normal must be finite and nonzero.",
    )
    plic_squared_norm = oe.contract("...i,...i->...", plic_, plic_)
    plic_squared_norm = eqx.error_if(
        plic_squared_norm,
        jnp.any(~jnp.isfinite(plic_squared_norm) | (plic_squared_norm <= 0.0)),
        "plic_normal must be finite and nonzero.",
    )
    wall_unit = wall_ / jnp.sqrt(wall_squared_norm)[..., None]
    plic_unit = plic_ / jnp.sqrt(plic_squared_norm)[..., None]

    plic_wall_cosine = oe.contract("...i,...i->...", plic_unit, wall_unit)
    tangent = plic_unit - plic_wall_cosine[..., None] * wall_unit
    tangent_norm = jnp.sqrt(oe.contract("...i,...i->...", tangent, tangent))
    degeneracy_floor = jnp.asarray(
        64.0 * np.finfo(np.dtype(dtype)).eps,
        dtype=dtype,
    )
    tangent_norm = eqx.error_if(
        tangent_norm,
        jnp.any(~jnp.isfinite(tangent_norm) | (tangent_norm <= degeneracy_floor)),
        "PLIC normal has a degenerate projection into the wall tangent plane.",
    )

    angle = jnp.asarray(condition.angle, dtype=dtype)
    sine = jnp.sin(angle)
    cosine = jnp.cos(angle)
    tangent_unit = tangent / tangent_norm[..., None]
    candidate = cosine * wall_unit + sine * tangent_unit
    candidate_norm = jnp.sqrt(oe.contract("...i,...i->...", candidate, candidate))
    candidate_norm = eqx.error_if(
        candidate_norm,
        jnp.any(~jnp.isfinite(candidate_norm) | (candidate_norm <= 0.0)),
        "Contact-angle reconstruction produced a non-finite normal.",
    )
    normal = candidate / candidate_norm[..., None]
    realized_cosine = oe.contract("...i,...i->...", normal, wall_unit)
    realized_sine = oe.contract("...i,...i->...", normal, tangent_unit)
    declared_cosine = jnp.broadcast_to(cosine, realized_cosine.shape)
    cosine_defect = jnp.abs(realized_cosine - declared_cosine)
    angle_defect = jnp.abs(jnp.arctan2(realized_sine, realized_cosine) - angle)
    tolerance = jnp.asarray(condition.tolerance, dtype=dtype)
    passed = jnp.isfinite(cosine_defect) & jnp.isfinite(angle_defect)
    passed = passed & (cosine_defect <= tolerance) & (angle_defect <= tolerance)
    status = jnp.where(
        passed,
        int(ContactAngleStatus.SUCCESS),
        int(ContactAngleStatus.FAILED),
    ).astype(jnp.int32)
    evidence = ContactAngleEvidence(
        declared_cosine=declared_cosine,
        realized_cosine=realized_cosine,
        cosine_defect=cosine_defect,
        angle_defect=angle_defect,
        tolerance=jnp.broadcast_to(tolerance, realized_cosine.shape),
        passed=passed,
        status=status,
    )
    return ContactAngleReconstructionResult(
        normal=normal,
        evidence=evidence,
        status=status,
        angle_defect=angle_defect,
        body_tag=condition.body_tag,
        condition_id=condition.condition_id,
        geometry_id=geometry,
        plic_id=plic,
    )


__all__ = [
    "ContactAngleCondition",
    "ContactAngleEvidence",
    "ContactAngleReconstructionResult",
    "ContactAngleStatus",
    "EmbeddedBoundaryContactAngleSet",
    "reconstruct_wall_interface_normal",
]
