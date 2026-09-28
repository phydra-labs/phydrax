#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._numerics._quadrature_rules import gauss_kronrod_data
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import LatticeHarmonicDiscretization
from ....typing import Complex128, ConvertibleToArray, Dim, Float64, parse, Scalar, Scope


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
    ) -> None:
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
    reference_distance: Array
    port_id: str = eqx.field(static=True)


def _reference_distance(value: ArrayLike, /) -> Array:
    distance = jnp.asarray(value)
    if distance.ndim != 0 or not jnp.issubdtype(distance.dtype, jnp.number):
        raise ValueError("reference_distance must be one numeric scalar.")
    if jnp.issubdtype(distance.dtype, jnp.complexfloating):
        raise ValueError("reference_distance must be real.")
    return eqx.error_if(
        distance,
        (~jnp.isfinite(distance)) | (distance < 0.0),
        "reference_distance must be finite and nonnegative.",
    )


class HomogeneousMaxwellPort(AbstractFourierModalPort):
    """Homogeneous semi-infinite exterior with an outward reference distance."""

    material: FrequencyMaxwellMaterial
    reference_distance: Array
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: FrequencyMaxwellMaterial,
        /,
        *,
        reference_distance: ArrayLike = 0.0,
        port_id: str,
    ) -> None:
        if not isinstance(material, FrequencyMaxwellMaterial):
            raise TypeError("material must be a FrequencyMaxwellMaterial.")
        identifier = str(port_id)
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        self.material = material
        self.reference_distance = _reference_distance(reference_distance)
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
        reference_distance: ArrayLike = 0.0,
        mode_policy: Literal["frozen", "spectral-subspace"] = "frozen",
        port_id: str,
    ) -> None:
        if not isinstance(material, FrequencyMaxwellMaterial):
            raise TypeError("material must be FrequencyMaxwellMaterial.")
        if not isinstance(factorization, AbstractFourierFactorizationPlan):
            raise TypeError("factorization must be AbstractFourierFactorizationPlan.")
        if mode_policy not in ("frozen", "spectral-subspace"):
            raise ValueError("Unknown periodic Maxwell port mode policy.")
        identifier = str(port_id)
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        self.material = material
        self.factorization = factorization
        self.reference_distance = _reference_distance(reference_distance)
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
        translation: ArrayLike | Sequence[float] = (0.0, 0.0),
        layer_id: str,
    ) -> None:
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
    ) -> None:
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

    material_profile: Callable[[Array], FrequencyMaxwellMaterial]
    thickness: Array
    factorization: AbstractFourierFactorizationPlan
    integration_policy: ContinuousZIntegrationPolicy
    layer_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_profile: Callable[[Array], FrequencyMaxwellMaterial],
        thickness: ArrayLike,
        factorization: AbstractFourierFactorizationPlan,
        integration_policy: ContinuousZIntegrationPolicy,
        /,
        *,
        layer_id: str,
    ) -> None:
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

    def __init__(self, source_id: str, /) -> None:
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
    ) -> None:
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


class _HarmonicDim(Dim, minimum=1):
    """Lattice harmonics."""


class _WavenumberDim(Dim, minimum=1):
    """Transverse-wavenumber quadrature nodes."""


def _finite_real(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real.")
    result = array.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


class MovingLineChargeSource(StrictModule, NonTrainableState):
    """Line charge in uniform motion inside a Fourier-modal source plane.

    A charge per unit length ``λ`` (``charge``) along the in-plane direction
    ``ê_⊥ = ẑ × d̂`` moves with ``speed`` along the in-plane unit ``direction``
    ``d̂`` through ``position`` at ``t = 0``. Its transformed current is the sheet
    ``K̃ = λ d̂ exp(i k_B·(r − r₀))`` with Bloch wavevector
    ``k_B = (ω/v) d̂ + k_⊥ ê_⊥``: one zeroth-harmonic surface current. A nonzero
    ``transverse_wavenumber`` ``k_⊥`` is one ``exp(i k_⊥ ê_⊥·r)`` component of a
    point charge (see `MovingPointChargeQuadrature`), whose field decays from the
    plane as ``exp(−Γ|z|)`` with ``Γ² = ω²/v² − ω² ε μ + k_⊥²``.
    """

    __strict_contract__ = True

    charge: Float64[Scalar]
    speed: Float64[Scalar]
    direction: Float64[Literal[2]]
    position: Float64[Literal[2]]
    transverse_wavenumber: Float64[Scalar]
    source_id: str = eqx.field(static=True)
    source_key: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        /,
        *,
        charge: ConvertibleToArray,
        speed: ConvertibleToArray,
        direction: ConvertibleToArray = (1.0, 0.0),
        position: ConvertibleToArray = (0.0, 0.0),
        transverse_wavenumber: ConvertibleToArray = 0.0,
    ) -> None:
        identifier = str(source_id)
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        charge_ = _finite_real(charge, "charge")
        speed_ = _finite_real(speed, "speed")
        direction_ = _finite_real(direction, "direction")
        position_ = _finite_real(position, "position")
        transverse = _finite_real(transverse_wavenumber, "transverse_wavenumber")
        if charge_.shape != () or speed_.shape != () or transverse.shape != ():
            raise ValueError("charge, speed, and transverse_wavenumber must be scalars.")
        if not speed_ > 0.0:
            raise ValueError("speed must be positive.")
        if direction_.shape != (2,) or position_.shape != (2,):
            raise ValueError("direction and position are in-plane (2,) vectors.")
        norm = float(np.linalg.norm(direction_))
        if norm == 0.0:
            raise ValueError("direction must be nonzero.")
        scope = Scope()
        self.charge = parse(jnp.asarray(charge_), Float64[Scalar], "charge", scope=scope)
        self.speed = parse(jnp.asarray(speed_), Float64[Scalar], "speed", scope=scope)
        self.direction = parse(
            jnp.asarray(direction_ / norm), Float64[Literal[2]], "direction", scope=scope
        )
        self.position = parse(
            jnp.asarray(position_), Float64[Literal[2]], "position", scope=scope
        )
        self.transverse_wavenumber = parse(
            jnp.asarray(transverse),
            Float64[Scalar],
            "transverse_wavenumber",
            scope=scope,
        )
        self.source_id = identifier
        self.source_key = canonical_fingerprint(
            {
                "kind": "moving-line-charge-source",
                "source_id": identifier,
                "values": array_tree_fingerprint(
                    (charge_, speed_, direction_ / norm, position_, transverse)
                ),
            }
        )

    def bloch_wavevector(self, angular_frequency: ConvertibleToArray, /) -> Array:
        """``k_B = (ω/v) d̂ + k_⊥ ê_⊥``, the only wavevector the source excites."""
        omega = jnp.asarray(angular_frequency, dtype=jnp.float64)
        normal = jnp.stack((-self.direction[1], self.direction[0]))
        return omega / self.speed * self.direction + self.transverse_wavenumber * normal

    def surface_current(
        self,
        harmonics: LatticeHarmonicDiscretization,
        angular_frequency: ConvertibleToArray,
        /,
    ) -> Complex128[Literal[3], _HarmonicDim]:
        """Electric sheet coefficients ``(3, harmonic_count)``: harmonic zero only."""
        if not isinstance(harmonics, LatticeHarmonicDiscretization):
            raise TypeError("harmonics must be a LatticeHarmonicDiscretization.")
        amplitude = self.charge * jnp.exp(
            -1j * jnp.dot(self.bloch_wavevector(angular_frequency), self.position)
        )
        vector = jnp.concatenate((self.direction, jnp.zeros((1,)))).astype(jnp.complex128)
        return (
            jnp.zeros((3, harmonics.harmonic_count), dtype=jnp.complex128)
            .at[:, harmonics.plan.layout.zero_index]
            .set(amplitude * vector)
        )


class MovingPointChargeIntegral(StrictModule):
    """Kronrod estimate, embedded Gauss estimate, and their difference."""

    __strict_contract__ = True

    value: Array
    gauss_value: Array
    error_estimate: Array
    truncation_bound: Float64[Scalar]


class MovingPointChargeQuadrature(StrictModule):
    """Transverse-wavenumber decomposition of a moving point charge.

    ``q δ(y) = q ∫ dk_⊥/(2π) exp(i k_⊥ y)``: a point charge is the superposition
    of `MovingLineChargeSource` components of density ``q`` at transverse
    wavenumbers ``k_⊥``. Their coupling to structures a distance ``height``
    away decays as ``exp(−Γ height)`` with
    ``Γ = √(ω²/(β²γ²c²) + k_⊥²)`` in the source medium ``(ε, μ)``, so the
    integral is truncated at ``maximum_wavenumber`` or where
    ``exp(−2 Γ height)`` reaches ``tolerance`` (``truncation_bound`` reports the
    neglected relative power). Panels split at ``breakpoints`` (for example the
    radiation cutoffs of diffraction orders) carry cosine-mapped embedded
    Gauss–Kronrod rules, which absorb square-root cutoff singularities at panel
    ends; `integrate` reports the Kronrod–Gauss difference. ``symmetric``
    integrates ``[0, K]`` twice for mirror-symmetric structures. ``weights``
    include the ``1/(2π)`` of the transverse Fourier integral.
    """

    __strict_contract__ = True

    charge: Float64[Scalar]
    speed: Float64[Scalar]
    angular_frequency: Float64[Scalar]
    wavenumbers: Float64[_WavenumberDim]
    decay_rates: Float64[_WavenumberDim]
    weights: Float64[_WavenumberDim]
    gauss_weights: Float64[_WavenumberDim]
    maximum_wavenumber: Float64[Scalar]
    truncation_bound: Float64[Scalar]
    symmetric: bool = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        charge: ConvertibleToArray,
        speed: ConvertibleToArray,
        angular_frequency: ConvertibleToArray,
        /,
        *,
        height: ConvertibleToArray,
        permittivity: ConvertibleToArray = 1.0,
        permeability: ConvertibleToArray = 1.0,
        maximum_wavenumber: ConvertibleToArray | None = None,
        breakpoints: Sequence[float] = (),
        tolerance: float = 1e-12,
        order: int = 15,
        symmetric: bool = False,
    ) -> None:
        charge_ = float(_finite_real(charge, "charge"))
        speed_ = float(_finite_real(speed, "speed"))
        omega = float(_finite_real(angular_frequency, "angular_frequency"))
        height_ = float(_finite_real(height, "height"))
        index_squared = float(
            _finite_real(permittivity, "permittivity")
            * _finite_real(permeability, "permeability")
        )
        if speed_ <= 0.0 or omega <= 0.0 or height_ <= 0.0:
            raise ValueError("speed, angular_frequency, and height must be positive.")
        if not 0.0 < float(tolerance) < 1.0:
            raise ValueError("tolerance must lie in (0, 1).")
        longitudinal = omega**2 / speed_**2 - omega**2 * index_squared
        if longitudinal <= 0.0:
            raise ValueError(
                "The source medium is above the Cherenkov threshold; point-charge "
                "components do not decay away from the path."
            )
        # exp(-2 Γ(K) height) = tolerance fixes the evanescent truncation.
        decay = np.log(1.0 / float(tolerance)) / (2.0 * height_)
        limit = float(np.sqrt(max(decay**2 - longitudinal, 0.0)))
        if maximum_wavenumber is not None:
            limit = min(
                limit, float(_finite_real(maximum_wavenumber, "maximum_wavenumber"))
            )
        if not limit > 0.0:
            raise ValueError(
                "No transverse wavenumber couples above tolerance; raise tolerance or "
                "lower height."
            )
        interior = sorted(
            {float(value) for value in breakpoints if 0.0 < abs(value) < limit}
        )
        edges = np.asarray([0.0, *(abs(value) for value in interior), limit])
        edges = np.unique(edges)
        if not symmetric:
            edges = np.unique(np.concatenate((-edges[::-1], edges)))
        rule = gauss_kronrod_data(int(order))
        if rule.embedded_weights is None:
            raise RuntimeError(
                "The Gauss–Kronrod rule carries no embedded Gauss weights."
            )
        unit = 0.5 * (np.asarray(rule.nodes) + 1.0)
        # τ ↦ (1 − cos πτ)/2 has zero slope at both panel ends, removing the
        # square-root cutoff singularities of propagating diffraction orders.
        mapped = 0.5 * (1.0 - np.cos(np.pi * unit))
        jacobian = 0.5 * np.pi * np.sin(np.pi * unit)
        lower, upper = edges[:-1, None], edges[1:, None]
        nodes = (lower + (upper - lower) * mapped[None, :]).reshape(-1)
        scale = (upper - lower) * jacobian[None, :] * (2.0 if symmetric else 1.0)
        scale = scale / (2.0 * np.pi)
        kronrod = (scale * 0.5 * np.asarray(rule.weights)[None, :]).reshape(-1)
        gauss = (scale * 0.5 * np.asarray(rule.embedded_weights)[None, :]).reshape(-1)
        scope = Scope()
        self.charge = parse(jnp.asarray(charge_), Float64[Scalar], "charge", scope=scope)
        self.speed = parse(jnp.asarray(speed_), Float64[Scalar], "speed", scope=scope)
        self.angular_frequency = parse(
            jnp.asarray(omega), Float64[Scalar], "angular_frequency", scope=scope
        )
        self.wavenumbers = parse(
            jnp.asarray(nodes), Float64[_WavenumberDim], "wavenumbers", scope=scope
        )
        self.decay_rates = parse(
            jnp.asarray(np.sqrt(longitudinal + nodes**2)),
            Float64[_WavenumberDim],
            "decay_rates",
            scope=scope,
        )
        self.weights = parse(
            jnp.asarray(kronrod), Float64[_WavenumberDim], "weights", scope=scope
        )
        self.gauss_weights = parse(
            jnp.asarray(gauss), Float64[_WavenumberDim], "gauss_weights", scope=scope
        )
        self.maximum_wavenumber = parse(
            jnp.asarray(limit), Float64[Scalar], "maximum_wavenumber", scope=scope
        )
        self.truncation_bound = parse(
            jnp.asarray(np.exp(-2.0 * np.sqrt(longitudinal + limit**2) * height_)),
            Float64[Scalar],
            "truncation_bound",
            scope=scope,
        )
        self.symmetric = bool(symmetric)
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "moving-point-charge-quadrature",
                "values": array_tree_fingerprint((nodes, kronrod, gauss)),
                "charge": charge_,
                "speed": speed_,
                "angular_frequency": omega,
            }
        )

    def line_sources(
        self,
        source_id: str,
        /,
        *,
        direction: ConvertibleToArray = (1.0, 0.0),
        position: ConvertibleToArray = (0.0, 0.0),
    ) -> tuple[MovingLineChargeSource, ...]:
        """One line-charge component (density ``q``) per quadrature node."""
        return tuple(
            MovingLineChargeSource(
                source_id,
                charge=self.charge,
                speed=self.speed,
                direction=direction,
                position=position,
                transverse_wavenumber=value,
            )
            for value in np.asarray(self.wavenumbers)
        )

    def integrate(self, values: ConvertibleToArray, /) -> MovingPointChargeIntegral:
        """``∫ dk_⊥/(2π) f(k_⊥)`` of per-node ``values[node, ...]``."""
        array = jnp.asarray(values)
        if array.shape[:1] != self.wavenumbers.shape:
            raise ValueError("values must lead with one entry per quadrature node.")
        shape = (-1,) + (1,) * (array.ndim - 1)
        kronrod = jnp.sum(self.weights.reshape(shape) * array, axis=0)
        gauss = jnp.sum(self.gauss_weights.reshape(shape) * array, axis=0)
        return MovingPointChargeIntegral(
            kronrod, gauss, jnp.abs(kronrod - gauss), self.truncation_bound
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
    "MovingLineChargeSource",
    "MovingPointChargeIntegral",
    "MovingPointChargeQuadrature",
]
