#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import LatticeHarmonicDiscretization


class AbstractFourierFactorizationPlan(StrictModule, NonTrainableState):
    """Fourier material-factorization policy."""

    plan_id: str = eqx.field(static=True)

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        raise NotImplementedError


class FrequencyMaxwellMaterial(StrictModule):
    """Sampled frequency-domain permittivity and permeability."""

    permittivity: Array
    permeability: Array
    material_id: str = eqx.field(static=True)
    passive: bool | None = eqx.field(static=True)
    reciprocal: bool | None = eqx.field(static=True)

    def __init__(
        self,
        permittivity: ArrayLike,
        permeability: ArrayLike = 1.0,
        /,
        *,
        material_id: str | None = None,
        passive: bool | None = None,
        reciprocal: bool | None = None,
    ):
        epsilon = jnp.asarray(permittivity)
        mu = jnp.asarray(permeability)
        if not jnp.issubdtype(epsilon.dtype, jnp.number) or not jnp.issubdtype(
            mu.dtype, jnp.number
        ):
            raise TypeError("Permittivity and permeability must be numeric arrays.")
        identity = (
            canonical_fingerprint(
                {
                    "kind": "frequency-maxwell-material",
                    "epsilon_shape": list(epsilon.shape),
                    "mu_shape": list(mu.shape),
                }
            )
            if material_id is None
            else str(material_id)
        )
        if not identity:
            raise ValueError("material_id must be non-empty.")
        self.permittivity = epsilon
        self.permeability = mu
        self.material_id = identity
        self.passive = None if passive is None else bool(passive)
        self.reciprocal = None if reciprocal is None else bool(reciprocal)


class HomogeneousMaxwellPort(StrictModule):
    """Homogeneous semi-infinite exterior medium and reference plane."""

    material: FrequencyMaxwellMaterial
    reference_plane: Array
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: FrequencyMaxwellMaterial,
        /,
        *,
        reference_plane: ArrayLike = 0.0,
        port_id: str,
    ):
        if not isinstance(material, FrequencyMaxwellMaterial):
            raise TypeError("material must be a FrequencyMaxwellMaterial.")
        identifier = str(port_id)
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        plane = jnp.asarray(reference_plane)
        if plane.ndim > 0:
            raise ValueError("reference_plane must be scalar.")
        self.material = material
        self.reference_plane = plane
        self.port_id = identifier


class FourierModalLayer(StrictModule):
    """Finite z-invariant periodic layer."""

    material: FrequencyMaxwellMaterial
    thickness: Array
    factorization: AbstractFourierFactorizationPlan
    translation: Array
    layer_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: FrequencyMaxwellMaterial,
        thickness: ArrayLike,
        factorization: AbstractFourierFactorizationPlan,
        /,
        *,
        translation: ArrayLike = (0.0, 0.0),
        layer_id: str,
    ):
        if not isinstance(material, FrequencyMaxwellMaterial):
            raise TypeError("material must be a FrequencyMaxwellMaterial.")
        if not isinstance(factorization, AbstractFourierFactorizationPlan):
            raise TypeError("factorization must be an AbstractFourierFactorizationPlan.")
        identifier = str(layer_id)
        if not identifier:
            raise ValueError("layer_id must be non-empty.")
        thickness_ = jnp.asarray(thickness)
        if thickness_.ndim > 0:
            raise ValueError("Layer thickness must be scalar for one prepared case.")
        translation_ = jnp.asarray(translation)
        if translation_.shape != (2,):
            raise ValueError("Layer translation must have shape (2,).")
        self.material = material
        self.thickness = thickness_
        self.factorization = factorization
        self.translation = translation_
        self.layer_id = identifier


class FourierModalSourcePlane(StrictModule, NonTrainableState):
    """Named zero-thickness plane at which surface-current jumps may be applied."""

    source_id: str = eqx.field(static=True)

    def __init__(self, source_id: str, /):
        identifier = str(source_id)
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        self.source_id = identifier


FourierModalStackElement: TypeAlias = FourierModalLayer | FourierModalSourcePlane


class FourierModalMaxwellProblem(StrictModule):
    """One frequency/Bloch case with a static ordered stack topology."""

    harmonics: LatticeHarmonicDiscretization
    angular_frequency: Array
    bloch_wavevector: Array
    superstrate: HomogeneousMaxwellPort
    elements: tuple[FourierModalStackElement, ...]
    substrate: HomogeneousMaxwellPort
    problem_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        harmonics: LatticeHarmonicDiscretization,
        angular_frequency: ArrayLike,
        bloch_wavevector: ArrayLike,
        superstrate: HomogeneousMaxwellPort,
        elements: tuple[FourierModalStackElement, ...],
        substrate: HomogeneousMaxwellPort,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(harmonics, LatticeHarmonicDiscretization):
            raise TypeError("harmonics must be a LatticeHarmonicDiscretization.")
        if not isinstance(superstrate, HomogeneousMaxwellPort) or not isinstance(
            substrate, HomogeneousMaxwellPort
        ):
            raise TypeError("superstrate and substrate must be HomogeneousMaxwellPort.")
        omega = jnp.asarray(angular_frequency)
        wavevector = jnp.asarray(bloch_wavevector)
        if omega.ndim > 0:
            raise ValueError("angular_frequency must be scalar for one prepared case.")
        if wavevector.shape != (2,):
            raise ValueError("bloch_wavevector must have shape (2,).")
        element_tuple = tuple(elements)
        if not all(
            isinstance(element, FourierModalLayer | FourierModalSourcePlane)
            for element in element_tuple
        ):
            raise TypeError("elements must contain layers or source planes.")
        source_ids = tuple(
            element.source_id
            for element in element_tuple
            if isinstance(element, FourierModalSourcePlane)
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source-plane IDs must be unique within a problem.")
        version = str(numeric_version)
        self.harmonics = harmonics
        self.angular_frequency = omega
        self.bloch_wavevector = wavevector
        self.superstrate = superstrate
        self.elements = element_tuple
        self.substrate = substrate
        self.numeric_version = version
        self.problem_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-maxwell-problem",
                "harmonics": harmonics.preparation_id,
                "superstrate": superstrate.port_id,
                "elements": [
                    (
                        {"kind": "layer", "id": element.layer_id}
                        if isinstance(element, FourierModalLayer)
                        else {"kind": "source", "id": element.source_id}
                    )
                    for element in element_tuple
                ],
                "substrate": substrate.port_id,
                "numeric_version": version,
            }
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            element.source_id
            for element in self.elements
            if isinstance(element, FourierModalSourcePlane)
        )

    @property
    def layer_count(self) -> int:
        return sum(isinstance(element, FourierModalLayer) for element in self.elements)


__all__ = [
    "AbstractFourierFactorizationPlan",
    "FourierModalLayer",
    "FourierModalMaxwellProblem",
    "FourierModalSourcePlane",
    "FourierModalStackElement",
    "FrequencyMaxwellMaterial",
    "HomogeneousMaxwellPort",
]
