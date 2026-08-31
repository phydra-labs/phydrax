#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._properties import BeamSection


class WarpingBeamSection(StrictModule, NonTrainableState):
    """Beam section plus shear-center and nonuniform-torsion properties."""

    base: BeamSection
    shear_center_y: Array
    shear_center_z: Array
    warping_constant: Array
    monosymmetry_y: Array
    monosymmetry_z: Array
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: BeamSection,
        shear_center_y: ArrayLike,
        shear_center_z: ArrayLike,
        warping_constant: ArrayLike,
        /,
        *,
        monosymmetry_y: ArrayLike = 0.0,
        monosymmetry_z: ArrayLike = 0.0,
        section_id: str | None = None,
    ):
        if not isinstance(base, BeamSection):
            raise TypeError("base must be a BeamSection.")
        values = tuple(
            jnp.asarray(value, dtype=base.area.dtype)
            for value in (
                shear_center_y,
                shear_center_z,
                warping_constant,
                monosymmetry_y,
                monosymmetry_z,
            )
        )
        if any(value.shape != () or not bool(jnp.isfinite(value)) for value in values):
            raise ValueError("Warping section properties must be finite scalars.")
        if not bool(values[2] > 0.0):
            raise ValueError("warping_constant must be positive.")
        (
            self.shear_center_y,
            self.shear_center_z,
            self.warping_constant,
            self.monosymmetry_y,
            self.monosymmetry_z,
        ) = values
        self.base = base
        self.section_id = str(section_id or f"{base.section_id}:warping")


class WarpingBeamState(StrictModule):
    axial_energy: Array
    bending_energy: Array
    torsion_energy: Array
    warping_energy: Array
    bimoment: Array
    torsional_moment: Array
    load_height_coupling: Array
    valid: Array


def evaluate_warping_beam(
    length: ArrayLike,
    axial_extension: ArrayLike,
    end_rotations: ArrayLike,
    end_warping: ArrayLike,
    material_young: ArrayLike,
    material_shear: ArrayLike,
    section: WarpingBeamSection,
    /,
    *,
    load_height_force: ArrayLike = 0.0,
) -> WarpingBeamState:
    """Return condensed elastic Vlasov beam energy and resultants."""
    length_ = jnp.asarray(length)
    extension = jnp.asarray(axial_extension, dtype=length_.dtype)
    rotations = jnp.asarray(end_rotations, dtype=length_.dtype)
    warping = jnp.asarray(end_warping, dtype=length_.dtype)
    young = jnp.asarray(material_young, dtype=length_.dtype)
    shear = jnp.asarray(material_shear, dtype=length_.dtype)
    load_force = jnp.asarray(load_height_force, dtype=length_.dtype)
    if rotations.shape != (2, 3) or warping.shape != (2,):
        raise ValueError("Warping beam end rotations/warping have invalid shapes.")
    axial_stiffness = young * section.base.area / length_
    bending_y = young * section.base.inertia_y / length_
    bending_z = young * section.base.inertia_z / length_
    torsion = shear * section.base.torsion_constant / length_
    warping_stiffness = young * section.warping_constant / length_**3
    relative_rotation = rotations[1] - rotations[0]
    relative_warping = warping[1] - warping[0]
    axial_energy = 0.5 * axial_stiffness * extension**2
    bending_energy = 0.5 * (
        bending_y * relative_rotation[1] ** 2 + bending_z * relative_rotation[2] ** 2
    )
    torsion_energy = 0.5 * torsion * relative_rotation[0] ** 2
    warping_energy = 0.5 * warping_stiffness * relative_warping**2
    eccentricity = jnp.sqrt(section.shear_center_y**2 + section.shear_center_z**2)
    coupling = -load_force * eccentricity * relative_rotation[0]
    return WarpingBeamState(
        axial_energy,
        bending_energy,
        torsion_energy,
        warping_energy,
        warping_stiffness * relative_warping,
        torsion * relative_rotation[0] - load_force * eccentricity,
        coupling,
        jnp.isfinite(
            axial_energy + bending_energy + torsion_energy + warping_energy + coupling
        )
        & (length_ > 0.0)
        & (young > 0.0)
        & (shear > 0.0),
    )


class BracingState(StrictModule):
    lateral_energy: Array
    torsional_energy: Array
    warping_energy: Array
    total_reaction: Array


def evaluate_bracing(
    lateral_displacement: ArrayLike,
    twist: ArrayLike,
    warping: ArrayLike,
    lateral_stiffness: ArrayLike,
    torsional_stiffness: ArrayLike,
    warping_stiffness: ArrayLike,
    /,
) -> BracingState:
    lateral = jnp.asarray(lateral_displacement)
    twist_ = jnp.asarray(twist, dtype=lateral.dtype)
    warping_ = jnp.asarray(warping, dtype=lateral.dtype)
    kl = jnp.asarray(lateral_stiffness, dtype=lateral.dtype)
    kt = jnp.asarray(torsional_stiffness, dtype=lateral.dtype)
    kw = jnp.asarray(warping_stiffness, dtype=lateral.dtype)
    lateral_energy = 0.5 * jnp.sum(kl * lateral**2)
    torsional_energy = 0.5 * jnp.sum(kt * twist_**2)
    warping_energy = 0.5 * jnp.sum(kw * warping_**2)
    reaction = jnp.sqrt(
        jnp.sum((kl * lateral) ** 2)
        + jnp.sum((kt * twist_) ** 2)
        + jnp.sum((kw * warping_) ** 2)
    )
    return BracingState(lateral_energy, torsional_energy, warping_energy, reaction)


__all__ = [
    "BracingState",
    "WarpingBeamSection",
    "WarpingBeamState",
    "evaluate_bracing",
    "evaluate_warping_beam",
]
