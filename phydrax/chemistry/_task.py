#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed electronic tasks; physical methods remain independent of requested work."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ElectronicProperty(StrEnum):
    ENERGY = "energy"
    FORCES = "forces"
    HESSIAN = "hessian"
    DIPOLE = "dipole"
    POLARIZABILITY = "polarizability"
    STRESS = "stress"
    DENSITY_MATRIX = "density-matrix"
    ORBITALS = "orbitals"
    TRANSITION_MOMENTS = "transition-moments"
    NONADIABATIC_COUPLING = "nonadiabatic-coupling"
    BAND_ENERGIES = "band-energies"
    DIELECTRIC_TENSOR = "dielectric-tensor"
    BORN_EFFECTIVE_CHARGES = "born-effective-charges"
    POLARIZATION = "polarization"


class ElectronicTaskKind(StrEnum):
    GROUND_STATE = "ground-state"
    CORRELATION = "correlation"
    LINEAR_RESPONSE = "linear-response"
    EXCITED_MANIFOLD = "excited-manifold"
    NONADIABATIC_COUPLING = "nonadiabatic-coupling"
    BAND_STRUCTURE = "band-structure"
    OPTICAL_RESPONSE = "optical-response"


def _properties(
    values: Sequence[ElectronicProperty],
    /,
    *,
    require_energy: bool,
) -> tuple[ElectronicProperty, ...]:
    requested = tuple(values)
    if not requested or any(
        not isinstance(value, ElectronicProperty) for value in requested
    ):
        raise TypeError("properties must contain ElectronicProperty values.")
    normalized = tuple(sorted(set(requested), key=lambda value: value.value))
    if require_energy and ElectronicProperty.ENERGY not in normalized:
        raise ValueError("This electronic task must include energy.")
    if (
        ElectronicProperty.HESSIAN in normalized
        and ElectronicProperty.FORCES not in normalized
    ):
        raise ValueError("Hessian tasks must also request forces.")
    return normalized


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class GroundStateTaskPlan(StrictModule, NonTrainableState):
    """One exact ground-state observable request."""

    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(self, properties: Sequence[ElectronicProperty], /):
        normalized = _properties(properties, require_energy=True)
        self.properties = normalized
        self.task_kind = ElectronicTaskKind.GROUND_STATE
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-ground-state-task",
                "properties": [value.value for value in normalized],
            }
        )

    @classmethod
    def energy(cls) -> GroundStateTaskPlan:
        return cls((ElectronicProperty.ENERGY,))

    @classmethod
    def energy_and_forces(cls) -> GroundStateTaskPlan:
        return cls((ElectronicProperty.ENERGY, ElectronicProperty.FORCES))

    @classmethod
    def energy_forces_and_hessian(cls) -> GroundStateTaskPlan:
        return cls(
            (
                ElectronicProperty.ENERGY,
                ElectronicProperty.FORCES,
                ElectronicProperty.HESSIAN,
            )
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        if not isinstance(property_, ElectronicProperty):
            raise TypeError("property_ must be ElectronicProperty.")
        return property_ in self.properties


class CorrelationTaskPlan(StrictModule, NonTrainableState):
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    relaxed_density: bool = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        properties: Sequence[ElectronicProperty] = (ElectronicProperty.ENERGY,),
        /,
        *,
        relaxed_density: bool = False,
    ):
        normalized = _properties(properties, require_energy=True)
        self.properties = normalized
        self.relaxed_density = bool(relaxed_density)
        self.task_kind = ElectronicTaskKind.CORRELATION
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-correlation-task",
                "properties": [value.value for value in normalized],
                "relaxed_density": self.relaxed_density,
            }
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        return property_ in self.properties


class LinearResponseTaskPlan(StrictModule, NonTrainableState):
    perturbation: str = eqx.field(static=True)
    frequencies: Array
    damping: Array
    gauge: str = eqx.field(static=True)
    response_order: int = eqx.field(static=True)
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        perturbation: str,
        frequencies: ArrayLike | tuple[float, ...] = (0.0,),
        /,
        *,
        damping: ArrayLike = 0.0,
        gauge: str = "length",
        response_order: int = 1,
        properties: Sequence[ElectronicProperty] = (ElectronicProperty.POLARIZABILITY,),
    ):
        perturbation_ = _identifier(perturbation, "perturbation")
        gauge_ = _identifier(gauge, "gauge")
        frequency = np.asarray(frequencies, dtype=float).reshape((-1,))
        damping_ = np.broadcast_to(
            np.asarray(damping, dtype=float), frequency.shape
        ).copy()
        order = int(response_order)
        normalized = _properties(properties, require_energy=False)
        if (
            frequency.size == 0
            or np.any(~np.isfinite(frequency))
            or np.any(frequency < 0.0)
            or np.any(~np.isfinite(damping_))
            or np.any(damping_ < 0.0)
        ):
            raise ValueError(
                "Response frequencies and damping must be finite and non-negative."
            )
        if order not in (1, 2, 3):
            raise ValueError("response_order must be one, two, or three.")
        self.perturbation = perturbation_
        self.frequencies = jnp.asarray(frequency)
        self.damping = jnp.asarray(damping_)
        self.gauge = gauge_
        self.response_order = order
        self.properties = normalized
        self.task_kind = ElectronicTaskKind.LINEAR_RESPONSE
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-linear-response-task",
                "perturbation": perturbation_,
                "gauge": gauge_,
                "response_order": order,
                "properties": [value.value for value in normalized],
                "arrays": array_tree_fingerprint(
                    {"frequencies": frequency, "damping": damping_}
                ),
            }
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        return property_ in self.properties


class ExcitedManifoldTaskPlan(StrictModule, NonTrainableState):
    root_count: int = eqx.field(static=True)
    spin_sector: str = eqx.field(static=True)
    symmetry_sector: str | None = eqx.field(static=True)
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_count: int,
        /,
        *,
        spin_sector: str = "singlet",
        symmetry_sector: str | None = None,
        properties: Sequence[ElectronicProperty] = (
            ElectronicProperty.TRANSITION_MOMENTS,
        ),
    ):
        roots = int(root_count)
        spin = _identifier(spin_sector, "spin_sector")
        symmetry = (
            None
            if symmetry_sector is None
            else _identifier(symmetry_sector, "symmetry_sector")
        )
        normalized = _properties(properties, require_energy=False)
        if roots <= 0:
            raise ValueError("root_count must be positive.")
        self.root_count = roots
        self.spin_sector = spin
        self.symmetry_sector = symmetry
        self.properties = normalized
        self.task_kind = ElectronicTaskKind.EXCITED_MANIFOLD
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-excited-manifold-task",
                "root_count": roots,
                "spin_sector": spin,
                "symmetry_sector": symmetry,
                "properties": [value.value for value in normalized],
            }
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        return property_ in self.properties


class NonadiabaticCouplingTaskPlan(StrictModule, NonTrainableState):
    state_indices: tuple[int, ...] = eqx.field(static=True)
    route: str = eqx.field(static=True)
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_indices: Sequence[int],
        /,
        *,
        route: str = "analytic",
    ):
        indices = tuple(int(value) for value in state_indices)
        if len(indices) < 2 or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError(
                "Nonadiabatic coupling states must be unique non-negative indices."
            )
        route_ = _identifier(route, "route")
        self.state_indices = indices
        self.route = route_
        self.properties = (ElectronicProperty.NONADIABATIC_COUPLING,)
        self.task_kind = ElectronicTaskKind.NONADIABATIC_COUPLING
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-nonadiabatic-coupling-task",
                "state_indices": list(indices),
                "route": route_,
            }
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        return property_ in self.properties


class BandStructureTaskPlan(StrictModule, NonTrainableState):
    fractional_k_points: Array
    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    task_kind: ElectronicTaskKind = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(self, fractional_k_points: ArrayLike, /):
        points = np.asarray(fractional_k_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("Band-structure k points must have finite shape (K, 3).")
        self.fractional_k_points = jnp.asarray(points)
        self.properties = (ElectronicProperty.BAND_ENERGIES,)
        self.task_kind = ElectronicTaskKind.BAND_STRUCTURE
        self.task_id = canonical_fingerprint(
            {
                "kind": "electronic-band-structure-task",
                "k_points": array_tree_fingerprint(points),
            }
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        return property_ in self.properties


ElectronicTaskPlan: TypeAlias = (
    GroundStateTaskPlan
    | CorrelationTaskPlan
    | LinearResponseTaskPlan
    | ExcitedManifoldTaskPlan
    | NonadiabaticCouplingTaskPlan
    | BandStructureTaskPlan
)


__all__ = [
    "BandStructureTaskPlan",
    "CorrelationTaskPlan",
    "ElectronicProperty",
    "ElectronicTaskKind",
    "ElectronicTaskPlan",
    "ExcitedManifoldTaskPlan",
    "GroundStateTaskPlan",
    "LinearResponseTaskPlan",
    "NonadiabaticCouplingTaskPlan",
]
