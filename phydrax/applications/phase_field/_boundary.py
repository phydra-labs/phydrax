#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FiniteElementDiscretization, IntegrationDomain
from ...discretization.fem import FiniteElementBoundarySet


class AbstractPhaseFieldSurfaceEnergy(StrictModule, NonTrainableState):
    surface_energy_id: AbstractAttribute[str]
    time_dependent: AbstractAttribute[bool]

    @abc.abstractmethod
    def density(
        self,
        phase: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> Array:
        raise NotImplementedError


class PolynomialSurfaceEnergy(AbstractPhaseFieldSurfaceEnergy):
    coefficients: Array
    surface_energy_id: str = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, coefficients: Sequence[float] | ArrayLike, /):
        values = np.asarray(coefficients, dtype=float)
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("Surface-energy coefficients must be finite and rank one.")
        self.coefficients = jnp.asarray(values)
        self.time_dependent = False
        self.surface_energy_id = canonical_fingerprint(
            {
                "kind": "polynomial-phase-field-surface-energy",
                "coefficients": array_tree_fingerprint(values),
            }
        )

    def density(
        self,
        phase: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> Array:
        del points, time, args
        value = jnp.asarray(phase)
        result = jnp.zeros_like(value)
        for coefficient in self.coefficients[::-1]:
            result = result * value + coefficient.astype(value.dtype)
        return result


class YoungAngleSurfaceEnergy(AbstractPhaseFieldSurfaceEnergy):
    liquid_interface_tension: Array
    contact_angle_radians: Array
    surface_energy_id: str = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)

    def __init__(
        self,
        liquid_interface_tension: ArrayLike,
        contact_angle_radians: ArrayLike,
        /,
    ):
        tension = np.asarray(liquid_interface_tension)
        angle = np.asarray(contact_angle_radians)
        if (
            tension.shape != ()
            or angle.shape != ()
            or not np.isfinite(tension)
            or tension <= 0.0
            or not np.isfinite(angle)
            or angle < 0.0
            or angle > np.pi
        ):
            raise ValueError("Young-angle wetting parameters are invalid.")
        self.liquid_interface_tension = jnp.asarray(tension)
        self.contact_angle_radians = jnp.asarray(angle)
        self.time_dependent = False
        self.surface_energy_id = canonical_fingerprint(
            {
                "kind": "young-angle-phase-field-surface-energy",
                "liquid_interface_tension": float(tension),
                "contact_angle_radians": float(angle),
            }
        )

    def density(
        self,
        phase: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> Array:
        del points, time, args
        value = jnp.asarray(phase)
        interpolation = 0.5 + 0.75 * value - 0.25 * value**3
        difference = self.liquid_interface_tension.astype(value.dtype) * jnp.cos(
            self.contact_angle_radians.astype(value.dtype)
        )
        return -difference * interpolation


class PrescribedMicrotractionEnergy(AbstractPhaseFieldSurfaceEnergy):
    traction: Callable = eqx.field(static=True)
    surface_energy_id: str = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, traction: Callable, /, *, traction_id: str):
        if not callable(traction):
            raise TypeError("Prescribed microtraction requires a callable.")
        identifier = str(traction_id)
        if not identifier:
            raise ValueError("Prescribed microtraction requires a stable ID.")
        self.traction = traction
        self.time_dependent = True
        self.surface_energy_id = canonical_fingerprint(
            {"kind": "prescribed-phase-field-microtraction", "id": identifier}
        )

    def density(
        self,
        phase: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> Array:
        load = jnp.asarray(self.traction(points, time, args), dtype=phase.dtype)
        return -jnp.broadcast_to(load, phase.shape) * phase


class PrescribedPhaseFieldFlux(StrictModule, NonTrainableState):
    evaluator: Callable = eqx.field(static=True)
    flux_id: str = eqx.field(static=True)

    def __init__(self, evaluator: Callable, /, *, flux_id: str):
        if not callable(evaluator):
            raise TypeError("Prescribed phase-field flux requires a callable.")
        identifier = str(flux_id)
        if not identifier:
            raise ValueError("Prescribed phase-field flux requires a stable flux_id.")
        self.evaluator = evaluator
        self.flux_id = canonical_fingerprint(
            {"kind": "prescribed-phase-field-flux", "id": identifier}
        )

    def evaluate(
        self,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> Array:
        values = jnp.asarray(self.evaluator(points, time, args))
        return jnp.broadcast_to(values, points.shape[:-1])


class PhaseFieldBoundaryPatch(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    domain: IntegrationDomain
    surface_energy: AbstractPhaseFieldSurfaceEnergy | None
    mass_flux: PrescribedPhaseFieldFlux | None
    patch_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        domain: IntegrationDomain,
        /,
        *,
        surface_energy: AbstractPhaseFieldSurfaceEnergy | None = None,
        mass_flux: PrescribedPhaseFieldFlux | None = None,
    ):
        patch_name = str(name)
        if not patch_name:
            raise ValueError("Phase-field boundary patch name must be nonempty.")
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise TypeError("Phase-field boundary patches require exterior facets.")
        if surface_energy is not None and not isinstance(
            surface_energy, AbstractPhaseFieldSurfaceEnergy
        ):
            raise TypeError("surface_energy has an invalid type.")
        if mass_flux is not None and not isinstance(mass_flux, PrescribedPhaseFieldFlux):
            raise TypeError("mass_flux has an invalid type.")
        if surface_energy is None and mass_flux is None:
            raise ValueError("A boundary patch must declare surface energy or flux.")
        self.name = patch_name
        self.domain = domain
        self.surface_energy = surface_energy
        self.mass_flux = mass_flux
        self.patch_id = canonical_fingerprint(
            {
                "kind": "phase-field-boundary-patch",
                "name": patch_name,
                "domain": domain.domain_id,
                "surface": (
                    None if surface_energy is None else surface_energy.surface_energy_id
                ),
                "mass_flux": None if mass_flux is None else mass_flux.flux_id,
            }
        )


class PhaseFieldBoundaryPlan(StrictModule, NonTrainableState):
    patches: tuple[PhaseFieldBoundaryPatch, ...]
    patch_names: tuple[str, ...] = eqx.field(static=True)
    boundary_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        patches: Mapping[
            str,
            tuple[
                Sequence[int],
                AbstractPhaseFieldSurfaceEnergy | None,
                PrescribedPhaseFieldFlux | None,
            ],
        ],
        /,
        *,
        boundary_set: FiniteElementBoundarySet | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        exterior = discretization.exterior_facet_domain
        exterior_ids = np.asarray(exterior.entity_indices, dtype=np.int32)
        position = {int(facet): row for row, facet in enumerate(exterior_ids)}
        periodic = set()
        if boundary_set is not None:
            if not isinstance(boundary_set, FiniteElementBoundarySet):
                raise TypeError("boundary_set must be FiniteElementBoundarySet or None.")
            periodic = {
                facet
                for pair in boundary_set.periodic_pairs
                for facet in (pair.owner_facet, pair.neighbour_facet)
            }
        names = tuple(sorted(str(name) for name in patches))
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Phase-field boundary patch names must be unique.")
        used: set[int] = set()
        prepared = []
        for name in names:
            value = patches[name]
            if not isinstance(value, tuple) or len(value) != 3:
                raise TypeError(
                    "Boundary entries must be (facet_ids, surface_energy, mass_flux)."
                )
            raw_facets, surface, flux = value
            facets = tuple(int(facet) for facet in raw_facets)
            if (
                not facets
                or len(facets) != len(set(facets))
                or any(facet not in position for facet in facets)
                or set(facets) & periodic
                or set(facets) & used
            ):
                raise ValueError(
                    "Phase-field boundary facets must be unique physical exterior facets."
                )
            used.update(facets)
            rows = np.asarray([position[facet] for facet in facets], dtype=np.int32)
            domain = IntegrationDomain(
                "exterior_facet",
                exterior_ids[rows],
                exterior.support_id,
                exterior.entity_set_id,
                owner_cells=np.asarray(exterior.owner_cells)[rows],
                neighbour_cells=np.asarray(exterior.neighbour_cells)[rows],
                owner_local_entities=np.asarray(exterior.owner_local_entities)[rows],
                neighbour_local_entities=np.asarray(exterior.neighbour_local_entities)[
                    rows
                ],
                selection_id=canonical_fingerprint(
                    {
                        "kind": "phase-field-boundary-domain",
                        "name": name,
                        "facets": facets,
                    }
                ),
            )
            prepared.append(
                PhaseFieldBoundaryPatch(
                    name,
                    domain,
                    surface_energy=surface,
                    mass_flux=flux,
                )
            )
        self.patches = tuple(prepared)
        self.patch_names = names
        self.boundary_plan_id = canonical_fingerprint(
            {
                "kind": "phase-field-boundary-plan",
                "discretization": discretization.prepared_id,
                "patches": [patch.patch_id for patch in prepared],
                "periodic": (
                    None if boundary_set is None else boundary_set.boundary_set_id
                ),
            }
        )

    @property
    def surface_patches(self) -> tuple[PhaseFieldBoundaryPatch, ...]:
        return tuple(patch for patch in self.patches if patch.surface_energy is not None)

    @property
    def flux_patches(self) -> tuple[PhaseFieldBoundaryPatch, ...]:
        return tuple(patch for patch in self.patches if patch.mass_flux is not None)


__all__ = [
    "AbstractPhaseFieldSurfaceEnergy",
    "PhaseFieldBoundaryPatch",
    "PhaseFieldBoundaryPlan",
    "PolynomialSurfaceEnergy",
    "PrescribedMicrotractionEnergy",
    "PrescribedPhaseFieldFlux",
    "YoungAngleSurfaceEnergy",
]
