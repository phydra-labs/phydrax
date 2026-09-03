#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

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
    """Sampled frequency-domain constitutive data in one logical material slot."""

    permittivity: Array
    permeability: Array
    magnetoelectric_xi: Array
    magnetoelectric_zeta: Array
    material_id: str = eqx.field(static=True)
    material_role: Literal["physical", "artificial_pml"] = eqx.field(static=True)
    origin_evidence_id: str = eqx.field(static=True)
    passive: bool | None = eqx.field(static=True)
    reciprocal: bool | None = eqx.field(static=True)

    def __init__(
        self,
        permittivity: ArrayLike,
        permeability: ArrayLike = 1.0,
        /,
        magnetoelectric_xi: ArrayLike = 0.0,
        magnetoelectric_zeta: ArrayLike = 0.0,
        *,
        material_id: str,
        material_role: Literal["physical", "artificial_pml"] = "physical",
        origin_evidence_id: str | None = None,
        passive: bool | None = None,
        reciprocal: bool | None = None,
    ):
        epsilon = jnp.asarray(permittivity)
        mu = jnp.asarray(permeability)
        xi = jnp.asarray(magnetoelectric_xi)
        zeta = jnp.asarray(magnetoelectric_zeta)
        if any(
            not jnp.issubdtype(value.dtype, jnp.number)
            for value in (epsilon, mu, xi, zeta)
        ):
            raise TypeError("Maxwell constitutive blocks must be numeric arrays.")
        identifier = str(material_id)
        if not identifier:
            raise ValueError("material_id must be non-empty.")
        if material_role not in ("physical", "artificial_pml"):
            raise ValueError("material_role must be 'physical' or 'artificial_pml'.")
        origin = identifier if origin_evidence_id is None else str(origin_evidence_id)
        if not origin:
            raise ValueError("origin_evidence_id must be non-empty.")
        self.permittivity = epsilon
        self.permeability = mu
        self.magnetoelectric_xi = xi
        self.magnetoelectric_zeta = zeta
        self.material_id = identifier
        self.material_role = material_role
        self.origin_evidence_id = origin
        self.passive = None if passive is None else bool(passive)
        self.reciprocal = None if reciprocal is None else bool(reciprocal)


class AbstractFourierModalPort(StrictModule):
    """Prepared-interface contract for one semi-infinite Fourier-modal exterior."""

    material: FrequencyMaxwellMaterial
    reference_plane: Array
    port_id: str = eqx.field(static=True)


class HomogeneousMaxwellPort(AbstractFourierModalPort):
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


class PeriodicMaxwellPort(AbstractFourierModalPort):
    """Patterned/anisotropic z-invariant semi-infinite periodic exterior."""

    factorization: AbstractFourierFactorizationPlan
    mode_policy: Literal["frozen", "spectral-subspace"] = eqx.field(static=True)

    def __init__(
        self,
        material: FrequencyMaxwellMaterial,
        factorization: AbstractFourierFactorizationPlan,
        /,
        *,
        reference_plane: ArrayLike = 0.0,
        mode_policy: Literal["frozen", "spectral-subspace"] = "frozen",
        port_id: str,
    ):
        if not isinstance(material, FrequencyMaxwellMaterial):
            raise TypeError("material must be FrequencyMaxwellMaterial.")
        if not isinstance(factorization, AbstractFourierFactorizationPlan):
            raise TypeError("factorization must be AbstractFourierFactorizationPlan.")
        if mode_policy not in ("frozen", "spectral-subspace"):
            raise ValueError("Unknown periodic Maxwell port mode policy.")
        plane = jnp.asarray(reference_plane)
        if plane.ndim:
            raise ValueError("reference_plane must be scalar.")
        identifier = str(port_id)
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        self.material = material
        self.factorization = factorization
        self.reference_plane = plane
        self.mode_policy = mode_policy
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


class ContinuousZIntegrationPolicy(StrictModule, NonTrainableState):
    """Bounded embedded commutator-free Magnus preparation policy."""

    order: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_segments: int = eqx.field(static=True)
    minimum_segment_fraction: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: int = 4,
        /,
        *,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        maximum_segments: int = 64,
        minimum_segment_fraction: float = 1.0e-6,
    ):
        if int(order) != 4:
            raise ValueError("Continuous-z Fourier modal integration uses order four.")
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or absolute_tolerance + relative_tolerance <= 0.0
            or int(maximum_segments) < 1
            or not 0.0 < minimum_segment_fraction <= 1.0
        ):
            raise ValueError("Continuous-z integration policy is invalid.")
        self.order = 4
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.maximum_segments = int(maximum_segments)
        self.minimum_segment_fraction = float(minimum_segment_fraction)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "continuous-z-integration-policy",
                "order": 4,
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
                "maximum_segments": self.maximum_segments,
                "minimum_segment_fraction": self.minimum_segment_fraction,
            }
        )


class ContinuousFourierModalLayer(StrictModule):
    """Finite continuously varying z-profile with a prepared segment epoch."""

    material_profile: object
    thickness: Array
    factorization: AbstractFourierFactorizationPlan
    integration_policy: ContinuousZIntegrationPolicy
    layer_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_profile,
        thickness: ArrayLike,
        factorization: AbstractFourierFactorizationPlan,
        integration_policy: ContinuousZIntegrationPolicy,
        /,
        *,
        layer_id: str,
    ):
        if not callable(material_profile):
            raise TypeError("material_profile must be callable.")
        if not isinstance(factorization, AbstractFourierFactorizationPlan):
            raise TypeError("factorization must be AbstractFourierFactorizationPlan.")
        if not isinstance(integration_policy, ContinuousZIntegrationPolicy):
            raise TypeError("integration_policy must be ContinuousZIntegrationPolicy.")
        thickness_ = jnp.asarray(thickness)
        if thickness_.ndim:
            raise ValueError("Continuous layer thickness must be scalar.")
        identifier = str(layer_id)
        if not identifier:
            raise ValueError("layer_id must be non-empty.")
        self.material_profile = material_profile
        self.thickness = thickness_
        self.factorization = factorization
        self.integration_policy = integration_policy
        self.layer_id = identifier


class FourierModalSourcePlane(StrictModule, NonTrainableState):
    """Named zero-thickness plane at which surface-current jumps may be applied."""

    source_id: str = eqx.field(static=True)

    def __init__(self, source_id: str, /):
        identifier = str(source_id)
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        self.source_id = identifier


FourierModalStackElement: TypeAlias = (
    FourierModalLayer | ContinuousFourierModalLayer | FourierModalSourcePlane
)


class FourierModalMaxwellProblem(StrictModule):
    """One frequency/Bloch case with a static ordered stack topology."""

    harmonics: LatticeHarmonicDiscretization
    angular_frequency: Array
    bloch_wavevector: Array
    superstrate: AbstractFourierModalPort
    elements: tuple[FourierModalStackElement, ...]
    substrate: AbstractFourierModalPort
    problem_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        harmonics: LatticeHarmonicDiscretization,
        angular_frequency: ArrayLike,
        bloch_wavevector: ArrayLike,
        superstrate: AbstractFourierModalPort,
        elements: tuple[FourierModalStackElement, ...],
        substrate: AbstractFourierModalPort,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(harmonics, LatticeHarmonicDiscretization):
            raise TypeError("harmonics must be a LatticeHarmonicDiscretization.")
        if not isinstance(superstrate, AbstractFourierModalPort) or not isinstance(
            substrate, AbstractFourierModalPort
        ):
            raise TypeError("superstrate and substrate must be Fourier-modal ports.")
        omega = jnp.asarray(angular_frequency)
        wavevector = jnp.asarray(bloch_wavevector)
        if omega.ndim > 0:
            raise ValueError("angular_frequency must be scalar for one prepared case.")
        if wavevector.shape != (2,):
            raise ValueError("bloch_wavevector must have shape (2,).")
        element_tuple = tuple(elements)
        if not all(
            isinstance(
                element,
                FourierModalLayer | ContinuousFourierModalLayer | FourierModalSourcePlane,
            )
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
                        if isinstance(
                            element, FourierModalLayer | ContinuousFourierModalLayer
                        )
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
        return sum(
            isinstance(element, FourierModalLayer | ContinuousFourierModalLayer)
            for element in self.elements
        )


__all__ = [
    "AbstractFourierModalPort",
    "AbstractFourierFactorizationPlan",
    "ContinuousFourierModalLayer",
    "ContinuousZIntegrationPolicy",
    "FourierModalLayer",
    "FourierModalMaxwellProblem",
    "FourierModalSourcePlane",
    "PeriodicMaxwellPort",
    "FourierModalStackElement",
    "FrequencyMaxwellMaterial",
    "HomogeneousMaxwellPort",
]
