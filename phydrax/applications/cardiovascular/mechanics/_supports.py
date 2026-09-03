#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....equations import finite_element_form_from_functional, FiniteElementForm
from ....variational import FieldJetSpec, Functional, LocalIntegralTerm


def _nonnegative_stiffness(value: ArrayLike, name: str, /) -> Array:
    stiffness = jnp.asarray(value)
    if (
        stiffness.shape != ()
        or jnp.issubdtype(stiffness.dtype, jnp.complexfloating)
        or not bool(jnp.isfinite(stiffness))
        or bool(stiffness < 0.0)
    ):
        raise ValueError(f"{name} must be one nonnegative finite real scalar.")
    return stiffness


def _unit_vector(value: ArrayLike, name: str, /) -> Array:
    vector = jnp.asarray(value)
    if vector.shape != (3,) or jnp.issubdtype(vector.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real three-vector.")
    norm = jnp.sqrt(jnp.sum(vector * vector))
    if (
        not bool(jnp.all(jnp.isfinite(vector)))
        or not bool(jnp.isfinite(norm))
        or bool(norm <= 0.0)
    ):
        raise ValueError(f"{name} must be finite and nonzero.")
    return vector / norm


def _anchor(value: ArrayLike | None, /) -> Array:
    anchor = jnp.zeros((3,)) if value is None else jnp.asarray(value)
    if (
        anchor.shape != (3,)
        or jnp.issubdtype(anchor.dtype, jnp.complexfloating)
        or not bool(jnp.all(jnp.isfinite(anchor)))
    ):
        raise ValueError("anchor_displacement must be one finite real three-vector.")
    return anchor


class SupportResponse(StrictModule):
    """Pointwise Robin support energy and sign-explicit restoring traction."""

    energy_density: Array
    energy_gradient: Array
    restoring_traction: Array
    energy_hessian: Array
    traction_tangent: Array
    normal_displacement: Array
    tangential_displacement: Array
    finite: Array
    valid: Array
    support_id: str = eqx.field(static=True)
    support_kind: str = eqx.field(static=True)
    reference_configuration: str = eqx.field(static=True)


class SurfaceRobinSupport(StrictModule, NonTrainableState):
    """An anisotropic surface foundation, explicitly not a contact law.

    The potential per reference area is ``kn (u.n-u0.n)^2/2 +
    kt ||(I-nn)(u-u0)||^2/2``. ``restoring_traction`` is minus the energy
    gradient. Zero stiffness gives the exact traction-free limit. Finite large
    stiffness is only a Robin approximation; exact kinematic restraint belongs
    in the FEM essential-boundary-condition substrate.
    """

    direction: Array
    normal_stiffness: Array
    tangential_stiffness: Array
    anchor_displacement: Array
    support_id: str = eqx.field(static=True)
    support_kind: str = eqx.field(static=True)

    def __init__(
        self,
        direction: ArrayLike,
        normal_stiffness: ArrayLike,
        tangential_stiffness: ArrayLike,
        /,
        *,
        support_kind: str,
        anchor_displacement: ArrayLike | None = None,
        support_id: str | None = None,
    ):
        normal = _unit_vector(direction, "direction")
        normal_value = _nonnegative_stiffness(normal_stiffness, "normal_stiffness")
        tangential_value = _nonnegative_stiffness(
            tangential_stiffness, "tangential_stiffness"
        )
        anchor = _anchor(anchor_displacement)
        kind = str(support_kind)
        if not kind:
            raise ValueError("support_kind must be non-empty.")
        generated = canonical_fingerprint(
            {
                "kind": "cardiac-surface-robin-support",
                "support_kind": kind,
                "direction": array_tree_fingerprint(normal),
                "normal_stiffness": float(normal_value).hex(),
                "tangential_stiffness": float(tangential_value).hex(),
                "anchor_displacement": array_tree_fingerprint(anchor),
                "reference_configuration": "reference-surface-area",
            }
        )
        identifier = generated if support_id is None else str(support_id)
        if not identifier:
            raise ValueError("support_id must be non-empty or None.")
        self.direction = normal
        self.normal_stiffness = normal_value
        self.tangential_stiffness = tangential_value
        self.anchor_displacement = anchor
        self.support_id = identifier
        self.support_kind = kind

    def energy_density(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        if value.shape[-1:] != (3,):
            raise ValueError("Support displacement values must end in three components.")
        relative = value - self.anchor_displacement
        normal_value = contract("...i,i->...", relative, self.direction)
        tangential = relative - normal_value[..., None] * self.direction
        return 0.5 * (
            self.normal_stiffness * normal_value**2
            + self.tangential_stiffness
            * contract("...i,...i->...", tangential, tangential)
        )

    def evaluate(self, displacement: ArrayLike, /) -> SupportResponse:
        value = jnp.asarray(displacement)
        if value.shape[-1:] != (3,):
            raise ValueError("Support displacement values must end in three components.")
        relative = value - self.anchor_displacement
        normal_value = contract("...i,i->...", relative, self.direction)
        tangential = relative - normal_value[..., None] * self.direction
        normal_projector = self.direction[:, None] * self.direction[None, :]
        hessian = self.normal_stiffness * normal_projector + self.tangential_stiffness * (
            jnp.eye(3, dtype=value.dtype) - normal_projector
        )
        gradient = (
            self.normal_stiffness * normal_value[..., None] * self.direction
            + self.tangential_stiffness * tangential
        )
        energy = 0.5 * contract("...i,...i->...", relative, gradient)
        finite = (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(gradient), axis=-1)
        )
        return SupportResponse(
            energy,
            gradient,
            -gradient,
            hessian,
            -hessian,
            normal_value,
            tangential,
            finite,
            finite,
            self.support_id,
            self.support_kind,
            "reference-surface-area",
        )


class BasalSupport(StrictModule, NonTrainableState):
    """Basal-plane axial/tangential Robin support in a declared reference axis."""

    law: SurfaceRobinSupport

    def __init__(
        self,
        basal_axis: ArrayLike,
        axial_stiffness: ArrayLike,
        in_plane_stiffness: ArrayLike,
        /,
        *,
        anchor_displacement: ArrayLike | None = None,
        support_id: str | None = None,
    ):
        self.law = SurfaceRobinSupport(
            basal_axis,
            axial_stiffness,
            in_plane_stiffness,
            support_kind="basal",
            anchor_displacement=anchor_displacement,
            support_id=support_id,
        )

    def evaluate(self, displacement: ArrayLike, /) -> SupportResponse:
        return self.law.evaluate(displacement)

    def energy_density(self, displacement: ArrayLike, /) -> Array:
        return self.law.energy_density(displacement)


class VascularSupport(StrictModule, NonTrainableState):
    """Vessel-cut axial/transverse Robin support around a reference vessel axis."""

    law: SurfaceRobinSupport

    def __init__(
        self,
        vessel_axis: ArrayLike,
        axial_stiffness: ArrayLike,
        transverse_stiffness: ArrayLike,
        /,
        *,
        anchor_displacement: ArrayLike | None = None,
        support_id: str | None = None,
    ):
        self.law = SurfaceRobinSupport(
            vessel_axis,
            axial_stiffness,
            transverse_stiffness,
            support_kind="vascular",
            anchor_displacement=anchor_displacement,
            support_id=support_id,
        )

    def evaluate(self, displacement: ArrayLike, /) -> SupportResponse:
        return self.law.evaluate(displacement)

    def energy_density(self, displacement: ArrayLike, /) -> Array:
        return self.law.energy_density(displacement)


class EpicardialSupport(StrictModule, NonTrainableState):
    """Epicardial normal/tangential Robin foundation on reference area."""

    law: SurfaceRobinSupport

    def __init__(
        self,
        epicardial_normal: ArrayLike,
        normal_stiffness: ArrayLike,
        tangential_stiffness: ArrayLike,
        /,
        *,
        anchor_displacement: ArrayLike | None = None,
        support_id: str | None = None,
    ):
        self.law = SurfaceRobinSupport(
            epicardial_normal,
            normal_stiffness,
            tangential_stiffness,
            support_kind="epicardial",
            anchor_displacement=anchor_displacement,
            support_id=support_id,
        )

    def evaluate(self, displacement: ArrayLike, /) -> SupportResponse:
        return self.law.evaluate(displacement)

    def energy_density(self, displacement: ArrayLike, /) -> Array:
        return self.law.energy_density(displacement)


class PericardialSupport(StrictModule, NonTrainableState):
    """Pericardial Robin foundation; it does not detect or enforce contact."""

    law: SurfaceRobinSupport

    def __init__(
        self,
        pericardial_normal: ArrayLike,
        normal_stiffness: ArrayLike,
        tangential_stiffness: ArrayLike,
        /,
        *,
        anchor_displacement: ArrayLike | None = None,
        support_id: str | None = None,
    ):
        self.law = SurfaceRobinSupport(
            pericardial_normal,
            normal_stiffness,
            tangential_stiffness,
            support_kind="pericardial",
            anchor_displacement=anchor_displacement,
            support_id=support_id,
        )

    def evaluate(self, displacement: ArrayLike, /) -> SupportResponse:
        return self.law.evaluate(displacement)

    def energy_density(self, displacement: ArrayLike, /) -> Array:
        return self.law.energy_density(displacement)


CardiacSupport = BasalSupport | VascularSupport | EpicardialSupport | PericardialSupport


def _support_law(support: CardiacSupport, /) -> SurfaceRobinSupport:
    if isinstance(support, BasalSupport):
        return support.law
    if isinstance(support, VascularSupport):
        return support.law
    if isinstance(support, EpicardialSupport):
        return support.law
    if isinstance(support, PericardialSupport):
        return support.law
    raise TypeError("support must be a named cardiac support.")


def cardiac_support_functional(
    field_name: str,
    support: CardiacSupport,
    /,
    *,
    region: str,
    functional_id: str = "cardiac-surface-support",
) -> Functional:
    """Build a reference-surface support energy on the generic variational substrate."""
    field = str(field_name)
    region_ = str(region)
    identifier = str(functional_id)
    if not field or not region_ or not identifier:
        raise ValueError("field_name, region, and functional_id must be non-empty.")
    law = _support_law(support)

    def density(fields, geometry, context):
        del geometry, context
        return law.energy_density(fields[field].value)

    return Functional(
        identifier,
        (
            LocalIntegralTerm(
                f"{law.support_kind}-support-energy",
                region=region_,
                fields=(FieldJetSpec(field),),
                density=density,
                density_id=canonical_fingerprint(
                    {
                        "kind": "cardiac-support-energy",
                        "support_id": law.support_id,
                    }
                ),
            ),
        ),
        variable_fields=(field,),
    )


def cardiac_support_form(
    field_name: str,
    support: CardiacSupport,
    /,
    *,
    region: str,
    form_id: str = "cardiac-surface-support",
) -> FiniteElementForm:
    """Lower a named cardiac support energy to a finite-element form."""
    functional = cardiac_support_functional(
        field_name,
        support,
        region=region,
        functional_id=form_id,
    )
    return finite_element_form_from_functional(
        functional,
        {field_name: field_name},
        {region: None},
        form_id=form_id,
    )


__all__ = [
    "BasalSupport",
    "CardiacSupport",
    "EpicardialSupport",
    "PericardialSupport",
    "SupportResponse",
    "SurfaceRobinSupport",
    "VascularSupport",
    "cardiac_support_form",
    "cardiac_support_functional",
]
