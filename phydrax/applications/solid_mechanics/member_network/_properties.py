#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import pi
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....optim import FiniteAxis


def _positive_scalar(name: str, value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.inexact):
        raise TypeError(f"{name} must be one real inexact scalar.")
    if not bool(jnp.isfinite(array) & (array > 0.0)):
        raise ValueError(f"{name} must be finite and positive.")
    return array


def _optional_positive(name: str, value: ArrayLike | None, dtype: Any, /) -> Array:
    if value is None:
        return jnp.asarray(jnp.inf, dtype=dtype)
    return _positive_scalar(name, value).astype(dtype)


class LinearElasticMaterial(StrictModule, NonTrainableState):
    """Isotropic elastic stiffness, density, and optional strength evidence."""

    young_modulus: Array
    shear_modulus: Array
    density: Array
    yield_strength: Array
    tension_allowable: Array
    compression_allowable: Array
    thermal_expansion: Array
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        young_modulus: ArrayLike,
        shear_modulus: ArrayLike,
        density: ArrayLike,
        /,
        *,
        yield_strength: ArrayLike | None = None,
        tension_allowable: ArrayLike | None = None,
        compression_allowable: ArrayLike | None = None,
        thermal_expansion: ArrayLike = 0.0,
        material_id: str | None = None,
    ):
        young = _positive_scalar("young_modulus", young_modulus)
        shear = _positive_scalar("shear_modulus", shear_modulus).astype(young.dtype)
        density_ = _positive_scalar("density", density).astype(young.dtype)
        thermal = jnp.asarray(thermal_expansion, dtype=young.dtype)
        if thermal.shape != () or not bool(jnp.isfinite(thermal)):
            raise ValueError("thermal_expansion must be one finite scalar.")
        identifier = material_id or canonical_fingerprint(
            {
                "kind": "linear-elastic-material",
                "young": float(young),
                "shear": float(shear),
                "density": float(density_),
                "yield": None
                if yield_strength is None
                else float(jnp.asarray(yield_strength)),
                "tension": None
                if tension_allowable is None
                else float(jnp.asarray(tension_allowable)),
                "compression": None
                if compression_allowable is None
                else float(jnp.asarray(compression_allowable)),
                "thermal": float(thermal),
            }
        )
        self.young_modulus = young
        self.shear_modulus = shear
        self.density = density_
        self.yield_strength = _optional_positive(
            "yield_strength", yield_strength, young.dtype
        )
        self.tension_allowable = _optional_positive(
            "tension_allowable", tension_allowable, young.dtype
        )
        self.compression_allowable = _optional_positive(
            "compression_allowable", compression_allowable, young.dtype
        )
        self.thermal_expansion = thermal
        self.material_id = str(identifier)

    @classmethod
    def from_young_poisson(
        cls,
        young_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        density: ArrayLike,
        /,
        **kwargs,
    ) -> LinearElasticMaterial:
        young = _positive_scalar("young_modulus", young_modulus)
        poisson = jnp.asarray(poisson_ratio, dtype=young.dtype)
        if poisson.shape != () or not bool(
            jnp.isfinite(poisson) & (poisson > -1.0) & (poisson < 0.5)
        ):
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5.")
        shear = young / (2.0 * (1.0 + poisson))
        return cls(young, shear, density, **kwargs)


class AxialSection(StrictModule, NonTrainableState):
    """Axial area with optional economic and environmental intensities."""

    area: Array
    cost_per_mass: Array
    carbon_per_mass: Array
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        area: ArrayLike,
        /,
        *,
        cost_per_mass: ArrayLike = 0.0,
        carbon_per_mass: ArrayLike = 0.0,
        section_id: str | None = None,
    ):
        area_ = _positive_scalar("area", area)
        cost = jnp.asarray(cost_per_mass, dtype=area_.dtype)
        carbon = jnp.asarray(carbon_per_mass, dtype=area_.dtype)
        if any(
            value.shape != () or not bool(jnp.isfinite(value) & (value >= 0.0))
            for value in (cost, carbon)
        ):
            raise ValueError("Section intensities must be finite and nonnegative.")
        self.area = area_
        self.cost_per_mass = cost
        self.carbon_per_mass = carbon
        self.section_id = str(
            section_id
            or canonical_fingerprint(
                {
                    "kind": "axial-section",
                    "area": float(area_),
                    "cost": float(cost),
                    "carbon": float(carbon),
                }
            )
        )


class BeamSection(StrictModule, NonTrainableState):
    """Principal-axis Timoshenko section properties."""

    area: Array
    cost_per_mass: Array
    carbon_per_mass: Array
    section_id: str = eqx.field(static=True)
    inertia_y: Array
    inertia_z: Array
    torsion_constant: Array
    shear_area_y: Array
    shear_area_z: Array
    warping_constant: Array

    def __init__(
        self,
        area: ArrayLike,
        inertia_y: ArrayLike,
        inertia_z: ArrayLike,
        torsion_constant: ArrayLike,
        shear_area_y: ArrayLike,
        shear_area_z: ArrayLike,
        /,
        *,
        warping_constant: ArrayLike | None = None,
        cost_per_mass: ArrayLike = 0.0,
        carbon_per_mass: ArrayLike = 0.0,
        section_id: str | None = None,
    ):
        area_ = _positive_scalar("area", area)
        properties = tuple(
            _positive_scalar(name, value).astype(area_.dtype)
            for name, value in (
                ("inertia_y", inertia_y),
                ("inertia_z", inertia_z),
                ("torsion_constant", torsion_constant),
                ("shear_area_y", shear_area_y),
                ("shear_area_z", shear_area_z),
            )
        )
        warping = (
            jnp.asarray(0.0, dtype=area_.dtype)
            if warping_constant is None
            else _positive_scalar("warping_constant", warping_constant).astype(
                area_.dtype
            )
        )
        identifier = section_id or canonical_fingerprint(
            {
                "kind": "beam-section",
                "area": float(area_),
                "properties": [float(value) for value in properties],
                "warping": float(warping),
            }
        )
        cost = jnp.asarray(cost_per_mass, dtype=area_.dtype)
        carbon = jnp.asarray(carbon_per_mass, dtype=area_.dtype)
        if any(
            value.shape != () or not bool(jnp.isfinite(value) & (value >= 0.0))
            for value in (cost, carbon)
        ):
            raise ValueError("Section intensities must be finite and nonnegative.")
        self.area = area_
        self.cost_per_mass = cost
        self.carbon_per_mass = carbon
        self.section_id = str(identifier)
        (
            self.inertia_y,
            self.inertia_z,
            self.torsion_constant,
            self.shear_area_y,
            self.shear_area_z,
        ) = properties
        self.warping_constant = warping


class AbstractSectionFamily(StrictModule, NonTrainableState):
    family_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def section(self, parameters: ArrayLike, /) -> BeamSection:
        raise NotImplementedError


class RectangularSectionFamily(AbstractSectionFamily):
    """Rectangular section generated from width and depth."""

    def __init__(self, *, family_id: str = "section-family:rectangle"):
        self.family_id = str(family_id)

    def section(self, parameters: ArrayLike, /) -> BeamSection:
        values = jnp.asarray(parameters)
        if values.shape != (2,):
            raise ValueError("Rectangular parameters must be [width, depth].")
        width, depth = values
        width = _positive_scalar("width", width)
        depth = _positive_scalar("depth", depth).astype(width.dtype)
        area = width * depth
        inertia_y = width * depth**3 / 12.0
        inertia_z = depth * width**3 / 12.0
        ratio = jnp.minimum(width, depth) / jnp.maximum(width, depth)
        torsion = (
            jnp.maximum(width, depth)
            * jnp.minimum(width, depth) ** 3
            * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))
        )
        return BeamSection(
            area,
            inertia_y,
            inertia_z,
            torsion,
            (5.0 / 6.0) * area,
            (5.0 / 6.0) * area,
            section_id=f"{self.family_id}:{float(width)}:{float(depth)}",
        )


class CircularSectionFamily(AbstractSectionFamily):
    """Solid circular section generated from diameter."""

    def __init__(self, *, family_id: str = "section-family:circle"):
        self.family_id = str(family_id)

    def section(self, parameters: ArrayLike, /) -> BeamSection:
        diameter = _positive_scalar("diameter", parameters)
        area = pi * diameter**2 / 4.0
        inertia = pi * diameter**4 / 64.0
        torsion = 2.0 * inertia
        shear_area = 0.9 * area
        return BeamSection(
            area,
            inertia,
            inertia,
            torsion,
            shear_area,
            shear_area,
            section_id=f"{self.family_id}:{float(diameter)}",
        )


class TubeSectionFamily(AbstractSectionFamily):
    """Circular tube generated from outer diameter and thickness."""

    def __init__(self, *, family_id: str = "section-family:tube"):
        self.family_id = str(family_id)

    def section(self, parameters: ArrayLike, /) -> BeamSection:
        values = jnp.asarray(parameters)
        if values.shape != (2,):
            raise ValueError("Tube parameters must be [outer_diameter, thickness].")
        outer = _positive_scalar("outer_diameter", values[0])
        thickness = _positive_scalar("thickness", values[1]).astype(outer.dtype)
        if not bool(2.0 * thickness < outer):
            raise ValueError("Tube thickness must be less than half the diameter.")
        inner = outer - 2.0 * thickness
        area = pi * (outer**2 - inner**2) / 4.0
        inertia = pi * (outer**4 - inner**4) / 64.0
        torsion = 2.0 * inertia
        return BeamSection(
            area,
            inertia,
            inertia,
            torsion,
            0.5 * area,
            0.5 * area,
            section_id=f"{self.family_id}:{float(outer)}:{float(thickness)}",
        )


class MemberPropertyMap(StrictModule, NonTrainableState):
    """Static mapping from members to materials, sections, and design groups."""

    materials: tuple[LinearElasticMaterial, ...]
    sections: tuple[AxialSection | BeamSection, ...]
    member_material: Array
    member_section: Array
    fabrication_group: Array
    actuator_group: Array
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        materials: Sequence[LinearElasticMaterial],
        sections: Sequence[AxialSection | BeamSection],
        member_material: ArrayLike,
        member_section: ArrayLike,
        /,
        *,
        fabrication_group: ArrayLike | None = None,
        actuator_group: ArrayLike | None = None,
    ):
        materials_ = tuple(materials)
        sections_ = tuple(sections)
        if not materials_ or any(
            not isinstance(value, LinearElasticMaterial) for value in materials_
        ):
            raise TypeError("materials must contain LinearElasticMaterial values.")
        if not sections_ or any(
            not isinstance(value, (AxialSection, BeamSection)) for value in sections_
        ):
            raise TypeError("sections must contain AxialSection or BeamSection values.")
        material_index = np.asarray(member_material)
        section_index = np.asarray(member_section)
        if (
            material_index.ndim != 1
            or section_index.shape != material_index.shape
            or not np.issubdtype(material_index.dtype, np.integer)
            or not np.issubdtype(section_index.dtype, np.integer)
        ):
            raise TypeError(
                "Member material/section mappings must be aligned integer arrays."
            )
        if (
            np.any(material_index < 0)
            or np.any(material_index >= len(materials_))
            or np.any(section_index < 0)
            or np.any(section_index >= len(sections_))
        ):
            raise ValueError("Member property indices are out of range.")
        count = material_index.size
        fabrication = (
            np.arange(count, dtype=np.int32)
            if fabrication_group is None
            else np.asarray(fabrication_group, dtype=np.int32)
        )
        actuator = (
            np.arange(count, dtype=np.int32)
            if actuator_group is None
            else np.asarray(actuator_group, dtype=np.int32)
        )
        if fabrication.shape != (count,) or actuator.shape != (count,):
            raise ValueError("Member group mappings must match the member count.")
        self.materials = materials_
        self.sections = sections_
        self.member_material = jnp.asarray(material_index, dtype=jnp.int32)
        self.member_section = jnp.asarray(section_index, dtype=jnp.int32)
        self.fabrication_group = jnp.asarray(fabrication)
        self.actuator_group = jnp.asarray(actuator)
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "member-property-map",
                "materials": [value.material_id for value in materials_],
                "sections": [value.section_id for value in sections_],
                "indices": array_tree_fingerprint(
                    (material_index, section_index, fabrication, actuator)
                ),
            }
        )

    @property
    def member_count(self) -> int:
        return int(self.member_material.size)

    def arrays(self, /) -> dict[str, Array]:
        young = jnp.asarray([value.young_modulus for value in self.materials])
        shear = jnp.asarray([value.shear_modulus for value in self.materials])
        density = jnp.asarray([value.density for value in self.materials])
        area = jnp.asarray([value.area for value in self.sections])
        return {
            "young": young[self.member_material],
            "shear": shear[self.member_material],
            "density": density[self.member_material],
            "area": area[self.member_section],
        }

    def structural_arrays(self, /) -> dict[str, Array]:
        """Return aligned axial, beam, density, and strength properties."""
        material = self.arrays()
        inertia_y = jnp.asarray(
            [
                value.inertia_y if isinstance(value, BeamSection) else 0.0
                for value in self.sections
            ]
        )
        inertia_z = jnp.asarray(
            [
                value.inertia_z if isinstance(value, BeamSection) else 0.0
                for value in self.sections
            ]
        )
        torsion = jnp.asarray(
            [
                value.torsion_constant if isinstance(value, BeamSection) else 0.0
                for value in self.sections
            ]
        )
        shear_y = jnp.asarray(
            [
                value.shear_area_y if isinstance(value, BeamSection) else 0.0
                for value in self.sections
            ]
        )
        shear_z = jnp.asarray(
            [
                value.shear_area_z if isinstance(value, BeamSection) else 0.0
                for value in self.sections
            ]
        )
        yield_strength = jnp.asarray([value.yield_strength for value in self.materials])
        tension = jnp.asarray([value.tension_allowable for value in self.materials])
        compression = jnp.asarray(
            [value.compression_allowable for value in self.materials]
        )
        return {
            **material,
            "inertia_y": inertia_y[self.member_section],
            "inertia_z": inertia_z[self.member_section],
            "torsion": torsion[self.member_section],
            "shear_area_y": shear_y[self.member_section],
            "shear_area_z": shear_z[self.member_section],
            "yield_strength": yield_strength[self.member_material],
            "tension_allowable": tension[self.member_material],
            "compression_allowable": compression[self.member_material],
        }


class SectionCatalog(StrictModule, NonTrainableState):
    """Correlated finite section candidates with stable external labels."""

    axis: FiniteAxis
    labels: tuple[str, ...] = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: dict[str, ArrayLike],
        labels: Sequence[str],
        /,
        *,
        catalog_id: str | None = None,
    ):
        axis = FiniteAxis(values)
        labels_ = tuple(str(value) for value in labels)
        if len(labels_) != axis.size or len(set(labels_)) != axis.size:
            raise ValueError("Catalog labels must be unique and match candidate count.")
        self.axis = axis
        self.labels = labels_
        self.catalog_id = str(
            catalog_id
            or canonical_fingerprint(
                {
                    "kind": "section-catalog",
                    "labels": list(labels_),
                    "values": array_tree_fingerprint(values),
                }
            )
        )


__all__ = [
    "AbstractSectionFamily",
    "AxialSection",
    "BeamSection",
    "CircularSectionFamily",
    "LinearElasticMaterial",
    "MemberPropertyMap",
    "RectangularSectionFamily",
    "SectionCatalog",
    "TubeSectionFamily",
]
