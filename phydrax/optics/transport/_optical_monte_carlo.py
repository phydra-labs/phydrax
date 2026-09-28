#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spectral, polarized optical photon Monte Carlo through triangle media.

One owner transports photon packets with position, direction, a complex Jones
vector in a transverse frame carried with the direction, vacuum wavelength,
time, statistical weight, and a persistent global identity. Media, surfaces,
sources, and detectors are extension protocols; the tissue configuration
(scalar Henyey--Greenstein media, unpolarized-mean Fresnel surfaces, explicit
photon batches, unit detector response) implements them here.

Frame convention: a lane carries ``transverse_axes`` ``e1`` with
``e2 = direction × e1`` so ``(e1, e2, direction)`` is right-handed and the
Jones vector is ``J = J1 e1 + J2 e2``. Scattering re-expresses ``J`` in the
scattering-plane frame ``(s, p)`` with ``s = direction × u`` for the sampled
azimuthal unit vector ``u`` and carries ``e1' = s``; interfaces use
``s = direction × normal``. Every frame change is a real rotation, so the
Jones norm is preserved exactly and the weight alone carries power.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import assert_never, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._triangle_ray import (
    intersect_triangle_rays,
    prepare_triangle_ray_query,
    PreparedTriangleRayQuery,
    TriangleRayIntersectionStatus,
    TriangleRayQueryPlan,
)
from ...typing import (
    Bool,
    Complex128,
    Dim,
    Float64,
    Int32,
    parse,
    PRNGKey,
    Size,
    UInt32,
    VariadicDim,
)
from ..geometric._interface import evaluate_refractive_interface
from ..geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)


OpticalInterfaceBranching: TypeAlias = Literal["stochastic", "expected-split"]


class _LaneDims(VariadicDim):
    """Photon lane batch: ``(photons,)`` at launch, ``(photons, branches)`` at exit."""


class _LaneDim(Dim):
    """Lanes handed to one medium, surface, or detector interaction."""


class _MediumDim(Dim, minimum=1):
    """Number of homogeneous media."""


class _PhotonDim(Dim):
    """Photons of one transport result."""


class _ArrivalDim(Dim):
    """Detector-arrival slots retained per photon."""


_FREE_PATH_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="free-path")
_SCATTERING_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="scattering")
_INTERFACE_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="interface")
_ROULETTE_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="roulette")
_SOURCE_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="source")
_SURFACE_ADDRESS = SampleAddress("optics", "optical-monte-carlo", target="surface")

# Both identity words all-ones is the reserved particle identity (F6 rule).
_RESERVED_IDENTITY = (1 << 64) - 1
_WORD = 1 << 32


class OpticalTransportStatus(IntEnum):
    """Terminal status of one bounded photon-packet history."""

    SUCCESS = 0
    INVALID_INPUT = 1
    AMBIGUOUS_INTERSECTION = 2
    TRAVERSAL_CAPACITY_EXHAUSTED = 3
    MEDIUM_MISMATCH = 4
    INTERFACE_FAILURE = 5
    BRANCH_CAPACITY_EXHAUSTED = 6
    INTERACTION_CAPACITY_EXHAUSTED = 7
    NONFINITE_RESULT = 8
    SPECTRAL_SUPPORT_EXCEEDED = 9
    DETECTOR_ARRIVAL_CAPACITY_EXHAUSTED = 10


class OpticalPhotonState(StrictModule, NonTrainableState):
    """Photon packets: kinematics, polarization frame, spectrum, weight, identity.

    ``jones_vectors`` are unit-norm components on ``(transverse_axes,
    direction × transverse_axes)``; ``wavelengths`` are vacuum wavelengths;
    ``(id_hi, id_lo)`` is the persistent 64-bit identity that keys every random
    draw of the packet and its branches.
    """

    __strict_contract__ = True

    positions: Float64[_LaneDims, Literal[3]]
    directions: Float64[_LaneDims, Literal[3]]
    transverse_axes: Float64[_LaneDims, Literal[3]]
    jones_vectors: Complex128[_LaneDims, Literal[2]]
    wavelengths: Float64[_LaneDims]
    times: Float64[_LaneDims]
    weights: Float64[_LaneDims]
    medium_indices: Int32[_LaneDims]
    id_hi: UInt32[_LaneDims]
    id_lo: UInt32[_LaneDims]

    @property
    def photon_count(self) -> int:
        return math.prod(self.weights.shape)


class OpticalScatteringSample(StrictModule, NonTrainableState):
    """One volume event per lane in the scattering-plane frame.

    ``wavelengths`` are the vacuum wavelengths after the event (unchanged by
    elastic scattering, re-drawn by wavelength shifting) and ``delays`` the
    non-negative re-emission delays added to the lane time.
    """

    __strict_contract__ = True

    cosines: Float64[_LaneDim]
    azimuths: Float64[_LaneDim]
    jones_vectors: Complex128[_LaneDim, Literal[2]]
    wavelengths: Float64[_LaneDim]
    delays: Float64[_LaneDim]


class OpticalSurfaceHit(StrictModule, NonTrainableState):
    """Lanes arriving at oriented triangles in the ``(s, p)`` incidence frame.

    ``normals`` point from the incident to the transmitted medium,
    ``frame_axes`` is the unit ``s`` axis (``direction × normal`` normalized,
    the carried axis at normal incidence), ``jones_vectors`` are expressed on
    ``(s, direction × s)``, and ``surface_ids`` are the physical surface
    identities of the struck triangles.
    """

    __strict_contract__ = True

    directions: Float64[_LaneDim, Literal[3]]
    normals: Float64[_LaneDim, Literal[3]]
    incident_indices: Float64[_LaneDim]
    transmitted_indices: Float64[_LaneDim]
    surface_kinds: Int32[_LaneDim]
    wavelengths: Float64[_LaneDim]
    jones_vectors: Complex128[_LaneDim, Literal[2]]
    frame_axes: Float64[_LaneDim, Literal[3]]
    surface_ids: Int32[_LaneDim]


class OpticalSurfaceInteraction(StrictModule, NonTrainableState):
    """Reflected and transmitted candidates with power fractions and frames.

    Each candidate carries its own unit transverse axis ``a`` and its Jones
    vector is expressed on ``(a, candidate_direction × a)``. Power fractions
    satisfy ``reflectance + transmittance <= 1``; the transport absorbs the
    remainder at the surface in the incident medium.
    """

    __strict_contract__ = True

    reflected_directions: Float64[_LaneDim, Literal[3]]
    transmitted_directions: Float64[_LaneDim, Literal[3]]
    reflected_jones: Complex128[_LaneDim, Literal[2]]
    transmitted_jones: Complex128[_LaneDim, Literal[2]]
    reflectance: Float64[_LaneDim]
    transmittance: Float64[_LaneDim]
    reflection_valid: Bool[_LaneDim]
    transmission_valid: Bool[_LaneDim]
    reflected_axes: Float64[_LaneDim, Literal[3]]
    transmitted_axes: Float64[_LaneDim, Literal[3]]


class OpticalMedium(Protocol):
    """Homogeneous-per-index medium interaction seen by the transport owner."""

    medium_count: int
    medium_id: str

    def refractive_index(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        """Real refractive index per lane."""
        ...

    def extinction(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        """Inverse-length extinction ``mu_a + mu_s`` per lane."""
        ...

    def albedo(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        """Single-scattering albedo ``mu_s / (mu_a + mu_s)`` per lane."""
        ...

    def spectral_support(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        """Whether the medium's tables cover each lane's wavelength.

        The transport refuses unsupported lanes with
        ``SPECTRAL_SUPPORT_EXCEEDED`` and never consumes their coefficients.
        """
        ...

    def scatter(
        self,
        medium_indices: Array,
        wavelengths: Array,
        jones_vectors: Array,
        keys: Array,
        /,
    ) -> OpticalScatteringSample:
        """Sample a volume event per lane from the carried-frame Jones vector."""
        ...


class OpticalSurfaceModel(Protocol):
    """Mirror and dielectric response for lanes hitting oriented triangles."""

    surface_model_id: str

    def validate_surfaces(self, surfaces: NonSequentialSurfaceTable, /) -> None:
        """Refuse a surface table this model cannot describe."""
        ...

    def interact(
        self, hit: OpticalSurfaceHit, keys: Array, /
    ) -> OpticalSurfaceInteraction:
        """Respond per lane; ``keys`` are the lanes' identity-addressed keys."""
        ...


class OpticalDetectorResponse(Protocol):
    """Fraction of accepted weight a detector records per lane."""

    detector_response_id: str

    def respond(
        self,
        detector_indices: Array,
        wavelengths: Array,
        times: Array,
        jones_vectors: Array,
        incidence_cosines: Array,
        /,
    ) -> Array: ...


class OpticalPhotonSource(Protocol):
    """Deterministic or sampled photon launch."""

    photon_count: int

    def launch(self, key: PRNGKey, /) -> OpticalPhotonState: ...


def _unit_rows(vectors: Array) -> tuple[Array, Array]:
    norm = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    valid = jnp.isfinite(norm) & (norm > 0.0)
    return vectors / jnp.where(valid, norm, 1.0)[..., None], valid


def rotate_jones_to_scattering_frame(jones_vectors: Array, azimuths: Array, /) -> Array:
    """Re-express carried-frame Jones components on the scattering-plane frame.

    For sampled azimuth ``phi`` the azimuthal unit vector is ``u = cos(phi) e1 +
    sin(phi) e2``, the frame normal is ``s = direction × u`` and the incident
    p axis is ``direction × s = -u``. The map is a rotation, so the norm is
    preserved.
    """

    cosine = jnp.cos(azimuths)
    sine = jnp.sin(azimuths)
    first = jones_vectors[:, 0]
    second = jones_vectors[:, 1]
    return jnp.stack(
        (-sine * first + cosine * second, -(cosine * first + sine * second)), axis=-1
    )


def _henyey_greenstein_cosine(anisotropy: Array, uniforms: Array) -> Array:
    """Exact inverse-CDF Henyey--Greenstein cosine; isotropic as ``g -> 0``."""

    small = jnp.abs(anisotropy) < 1e-6
    denominator = 1.0 - anisotropy + 2.0 * anisotropy * uniforms
    ratio = (1.0 - anisotropy * anisotropy) / jnp.where(small, 1.0, denominator)
    hg_cosine = (1.0 + anisotropy * anisotropy - ratio * ratio) / jnp.where(
        small, 1.0, 2.0 * anisotropy
    )
    return jnp.clip(jnp.where(small, 2.0 * uniforms - 1.0, hg_cosine), -1.0, 1.0)


class TissueOpticalMedium(StrictModule, NonTrainableState):
    """Wavelength-independent scalar tissue optics per medium.

    ``mu_a`` and ``mu_s`` are inverse-length coefficients, ``g`` is the
    Henyey--Greenstein first moment, and ``n`` the positive real refractive
    index. Scattering is scalar: the Jones vector is rotated into the
    scattering frame without amplitude change, so polarization is transported
    but not modified by the medium.
    """

    __strict_contract__ = True

    mu_a: Float64[_MediumDim]
    mu_s: Float64[_MediumDim]
    g: Float64[_MediumDim]
    n: Float64[_MediumDim]
    medium_count: Size[_MediumDim] = eqx.field(static=True)
    medium_id: str = eqx.field(static=True)

    def __init__(
        self,
        mu_a: ArrayLike,
        mu_s: ArrayLike,
        g: ArrayLike,
        n: ArrayLike,
        /,
    ) -> None:
        absorption = np.asarray(mu_a)
        scattering = np.asarray(mu_s)
        anisotropy = np.asarray(g)
        refractive = np.asarray(n)
        values = (absorption, scattering, anisotropy, refractive)
        if any(np.iscomplexobj(value) for value in values):
            raise TypeError("Tissue optical coefficients must be real-valued.")
        if any(value.ndim != 1 for value in values):
            raise ValueError("Tissue optical coefficient arrays must be rank one.")
        if (
            not (
                absorption.shape
                == scattering.shape
                == anisotropy.shape
                == refractive.shape
            )
            or absorption.size < 1
        ):
            raise ValueError(
                "Tissue optical coefficient arrays must have one matching "
                "non-empty shape."
            )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Tissue optical coefficients must be finite.")
        if np.any(absorption < 0.0) or np.any(scattering < 0.0):
            raise ValueError("mu_a and mu_s must be non-negative.")
        if np.any(np.abs(anisotropy) >= 1.0):
            raise ValueError("Henyey--Greenstein g must lie strictly between -1 and 1.")
        if np.any(refractive <= 0.0):
            raise ValueError("Tissue refractive indices must be positive.")
        host = tuple(np.asarray(value, dtype=np.float64) for value in values)
        self.mu_a = jnp.asarray(host[0])
        self.mu_s = jnp.asarray(host[1])
        self.g = jnp.asarray(host[2])
        self.n = jnp.asarray(host[3])
        self.medium_count = absorption.size
        self.medium_id = canonical_fingerprint(
            {
                "kind": "tissue-optical-medium",
                "content": array_tree_fingerprint(
                    {"mu_a": host[0], "mu_s": host[1], "g": host[2], "n": host[3]}
                ),
            }
        )

    def refractive_index(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del wavelengths
        return self.n[medium_indices]

    def extinction(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del wavelengths
        return self.mu_a[medium_indices] + self.mu_s[medium_indices]

    def albedo(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del wavelengths
        total = self.mu_a[medium_indices] + self.mu_s[medium_indices]
        return self.mu_s[medium_indices] / jnp.where(total > 0.0, total, 1.0)

    def spectral_support(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del medium_indices
        return jnp.ones(wavelengths.shape, dtype=jnp.bool_)

    def scatter(
        self,
        medium_indices: Array,
        wavelengths: Array,
        jones_vectors: Array,
        keys: Array,
        /,
    ) -> OpticalScatteringSample:
        uniforms = jax.vmap(lambda key: jr.uniform(key, (2,), dtype=jnp.float64))(keys)
        azimuth = 2.0 * jnp.pi * uniforms[:, 1]
        return OpticalScatteringSample(
            _henyey_greenstein_cosine(self.g[medium_indices], uniforms[:, 0]),
            azimuth,
            rotate_jones_to_scattering_frame(jones_vectors, azimuth),
            wavelengths,
            jnp.zeros(wavelengths.shape, dtype=jnp.float64),
        )


class ScalarFresnelSurfaceModel(StrictModule, NonTrainableState):
    """Unpolarized-mean Fresnel dielectrics and ideal mirrors.

    Power fractions are the mean of the ``s`` and ``p`` Fresnel reflectances
    (the MCML convention); the Jones vector is transported through the
    ``(s, p)`` frames without amplitude change.
    """

    surface_model_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.surface_model_id = "scalar-fresnel"

    def validate_surfaces(self, surfaces: NonSequentialSurfaceTable, /) -> None:
        del surfaces

    def interact(
        self, hit: OpticalSurfaceHit, keys: Array, /
    ) -> OpticalSurfaceInteraction:
        del keys
        interface = evaluate_refractive_interface(
            hit.directions, hit.normals, hit.incident_indices, hit.transmitted_indices
        )
        mirror = hit.surface_kinds == int(NonSequentialSurfaceKind.MIRROR)
        reflectance = jnp.where(mirror, 1.0, jnp.mean(interface.reflectance, axis=-1))
        transmittance = jnp.where(mirror, 0.0, jnp.mean(interface.transmittance, axis=-1))
        return OpticalSurfaceInteraction(
            interface.reflected_directions,
            interface.transmitted_directions,
            hit.jones_vectors,
            hit.jones_vectors,
            reflectance,
            transmittance,
            interface.reflection_valid,
            interface.transmission_valid & ~mirror,
            hit.frame_axes,
            hit.frame_axes,
        )


class UnitDetectorResponse(StrictModule, NonTrainableState):
    """Detectors that record the full accepted weight regardless of spectrum."""

    detector_response_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.detector_response_id = "unit"

    def respond(
        self,
        detector_indices: Array,
        wavelengths: Array,
        times: Array,
        jones_vectors: Array,
        incidence_cosines: Array,
        /,
    ) -> Array:
        del detector_indices, times, jones_vectors, incidence_cosines
        return jnp.ones(wavelengths.shape, dtype=jnp.float64)


def _default_transverse_axes(directions: np.ndarray) -> np.ndarray:
    reference = np.where(
        (np.abs(directions[:, 2]) < 0.9)[:, None],
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    )
    axes = np.cross(reference, directions)
    return axes / np.linalg.norm(axes, axis=-1)[:, None]


def _identity_words(first_identity: tuple[int, int], count: int) -> tuple[Array, Array]:
    """Consecutive identities from ``first_identity`` in launch order (F6 rule)."""

    hi, lo = first_identity
    if type(hi) is not int or type(lo) is not int:
        raise TypeError("first_identity must be a pair of int words.")
    if not (0 <= hi < _WORD and 0 <= lo < _WORD):
        raise ValueError("first_identity words must lie in [0, 2**32).")
    first = hi * _WORD + lo
    if first + count - 1 >= _RESERVED_IDENTITY:
        raise ValueError("Photon identities would reach the reserved all-ones identity.")
    identities = np.arange(count, dtype=np.uint64) + np.uint64(first)
    return (
        jnp.asarray((identities >> np.uint64(32)).astype(np.uint32)),
        jnp.asarray((identities & np.uint64(_WORD - 1)).astype(np.uint32)),
    )


def launch_optical_photons(
    positions: ArrayLike,
    directions: ArrayLike,
    medium_indices: ArrayLike,
    /,
    *,
    wavelengths: ArrayLike,
    jones_vectors: ArrayLike | None = None,
    transverse_axes: ArrayLike | None = None,
    times: ArrayLike = 0.0,
    weights: ArrayLike = 1.0,
    first_identity: tuple[int, int] = (0, 0),
) -> OpticalPhotonState:
    """Prepare one launch batch with consecutive identities from ``first_identity``.

    Directions are normalized. The default transverse axis is the unit cross
    product of a fixed reference vector with the direction; a supplied axis
    must be transverse. The default Jones vector is linear along the transverse
    axis; a supplied one is normalized to unit norm.
    """

    positions_ = np.asarray(positions, dtype=np.float64)
    directions_ = np.asarray(directions, dtype=np.float64)
    if positions_.ndim != 2 or positions_.shape[1:] != (3,):
        raise ValueError("positions must have shape (photons, 3).")
    if directions_.shape != positions_.shape:
        raise ValueError("directions must match the positions shape.")
    count = positions_.shape[0]
    if count < 1:
        raise ValueError("At least one photon must be launched.")
    if not np.all(np.isfinite(positions_)) or not np.all(np.isfinite(directions_)):
        raise ValueError("positions and directions must be finite.")
    direction_norm = np.linalg.norm(directions_, axis=-1)
    if np.any(direction_norm <= 0.0):
        raise ValueError("directions must be nonzero.")
    directions_ = directions_ / direction_norm[:, None]
    media = np.broadcast_to(np.asarray(medium_indices, dtype=np.int32), (count,))
    wavelengths_ = np.broadcast_to(np.asarray(wavelengths, dtype=np.float64), (count,))
    times_ = np.broadcast_to(np.asarray(times, dtype=np.float64), (count,))
    weights_ = np.broadcast_to(np.asarray(weights, dtype=np.float64), (count,))
    if not np.all(np.isfinite(wavelengths_)) or np.any(wavelengths_ <= 0.0):
        raise ValueError("wavelengths must be finite and positive.")
    if not np.all(np.isfinite(times_)):
        raise ValueError("times must be finite.")
    if not np.all(np.isfinite(weights_)) or np.any(weights_ < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if transverse_axes is None:
        axes = _default_transverse_axes(directions_)
    else:
        axes = np.asarray(transverse_axes, dtype=np.float64)
        if axes.shape != positions_.shape or not np.all(np.isfinite(axes)):
            raise ValueError("transverse_axes must be finite with shape (photons, 3).")
        axis_norm = np.linalg.norm(axes, axis=-1)
        if np.any(axis_norm <= 0.0):
            raise ValueError("transverse_axes must be nonzero.")
        axes = axes / axis_norm[:, None]
        if np.any(np.abs(np.sum(axes * directions_, axis=-1)) > 1e-9):
            raise ValueError("transverse_axes must be perpendicular to directions.")
    if jones_vectors is None:
        jones = np.zeros((count, 2), dtype=np.complex128)
        jones[:, 0] = 1.0
    else:
        jones = np.asarray(jones_vectors, dtype=np.complex128)
        if jones.shape != (count, 2) or not np.all(np.isfinite(jones)):
            raise ValueError("jones_vectors must be finite with shape (photons, 2).")
        jones_norm = np.sqrt(np.sum(np.abs(jones) ** 2, axis=-1))
        if np.any(jones_norm <= 0.0):
            raise ValueError("jones_vectors must be nonzero.")
        jones = jones / jones_norm[:, None]
    id_hi, id_lo = _identity_words(first_identity, count)
    return OpticalPhotonState(
        jnp.asarray(positions_),
        jnp.asarray(directions_),
        jnp.asarray(axes),
        jnp.asarray(jones),
        jnp.asarray(wavelengths_),
        jnp.asarray(times_),
        jnp.asarray(weights_),
        jnp.asarray(media),
        id_hi,
        id_lo,
    )


class ExplicitPhotonSource(StrictModule, NonTrainableState):
    """Source that launches one prepared photon batch exactly."""

    state: OpticalPhotonState
    photon_count: int = eqx.field(static=True)

    def __init__(self, state: OpticalPhotonState, /) -> None:
        if not isinstance(state, OpticalPhotonState):
            raise TypeError("state must be an OpticalPhotonState.")
        if state.weights.ndim != 1:
            raise ValueError("An explicit source needs a rank-one photon batch.")
        self.state = state
        self.photon_count = state.photon_count

    def launch(self, key: PRNGKey, /) -> OpticalPhotonState:
        del key
        return self.state


class OpticalVarianceReduction(StrictModule, NonTrainableState):
    """Explicit interface branching and Russian-roulette policy.

    ``"stochastic"`` branching keeps one lane per interface event with the
    Fresnel reflectance as the reflection probability; ``"expected-split"``
    launches both lanes weighted by reflectance and transmittance. Roulette
    kills lanes below ``roulette_threshold`` with probability
    ``1 - roulette_survival_probability`` and boosts survivors, which keeps
    the expected weight unchanged.
    """

    interface_branching: OpticalInterfaceBranching = eqx.field(static=True)
    roulette_threshold: float = eqx.field(static=True)
    roulette_survival_probability: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        interface_branching: OpticalInterfaceBranching = "stochastic",
        roulette_threshold: float = 0.0,
        roulette_survival_probability: float = 0.1,
    ) -> None:
        branching = parse(
            interface_branching, OpticalInterfaceBranching, "interface_branching"
        )
        threshold = float(roulette_threshold)
        survival = float(roulette_survival_probability)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("roulette_threshold must be finite and non-negative.")
        if not math.isfinite(survival) or not 0.0 < survival <= 1.0:
            raise ValueError("roulette_survival_probability must lie in (0, 1].")
        self.interface_branching = branching
        self.roulette_threshold = threshold
        self.roulette_survival_probability = survival
        self.policy_id = canonical_fingerprint(
            {
                "kind": "optical-variance-reduction",
                "interface_branching": branching,
                "roulette_threshold": threshold,
                "roulette_survival_probability": survival,
            }
        )


class OpticalMonteCarloPlan(StrictModule, NonTrainableState):
    """Host plan for fixed-capacity photon transport through triangle media.

    Lengths are in the medium's inverse-coefficient unit; ``relativity``
    supplies the exact speed of light in that same length unit per time unit
    so lane times are optical path length over ``c``. The medium's refractive
    indices are the ones used at interfaces and for time of flight.
    ``detector_arrival_capacity`` is the number of accepted detector arrivals
    retained per photon in :class:`OpticalDetectorArrivals`; zero retains
    none, and a positive capacity that a history exceeds reports
    ``DETECTOR_ARRIVAL_CAPACITY_EXHAUSTED``.
    """

    surfaces: NonSequentialSurfaceTable
    medium: OpticalMedium
    surface_model: OpticalSurfaceModel
    detector_response: OpticalDetectorResponse
    variance_reduction: OpticalVarianceReduction
    relativity: RelativityScaleContract = eqx.field(static=True)
    maximum_interactions: int = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    photon_batch_size: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    triangle_leaf_size: int = eqx.field(static=True)
    ray_tolerance: float = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    weight_tolerance: float = eqx.field(static=True)
    detector_arrival_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        surfaces: NonSequentialSurfaceTable,
        medium: OpticalMedium,
        /,
        *,
        relativity: RelativityScaleContract,
        maximum_interactions: int,
        variance_reduction: OpticalVarianceReduction | None = None,
        surface_model: OpticalSurfaceModel | None = None,
        detector_response: OpticalDetectorResponse | None = None,
        branch_capacity: int = 1,
        photon_batch_size: int = 1024,
        traversal_stack_capacity: int = 64,
        triangle_leaf_size: int = 8,
        ray_tolerance: float = 1e-9,
        tie_tolerance: float = 1e-9,
        weight_tolerance: float = 0.0,
        detector_arrival_capacity: int = 0,
    ) -> None:
        if not isinstance(surfaces, NonSequentialSurfaceTable):
            raise TypeError("surfaces must be a NonSequentialSurfaceTable.")
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        policy = (
            OpticalVarianceReduction()
            if variance_reduction is None
            else variance_reduction
        )
        if not isinstance(policy, OpticalVarianceReduction):
            raise TypeError("variance_reduction must be an OpticalVarianceReduction.")
        maximum = int(maximum_interactions)
        branches = int(branch_capacity)
        batch = int(photon_batch_size)
        stack = int(traversal_stack_capacity)
        leaf = int(triangle_leaf_size)
        if maximum < 1 or branches < 1 or batch < 1 or stack < 1 or leaf < 1:
            raise ValueError("All optical transport capacities must be positive.")
        arrivals = int(detector_arrival_capacity)
        if arrivals < 0:
            raise ValueError("detector_arrival_capacity must be non-negative.")
        tolerances = (float(ray_tolerance), float(tie_tolerance), float(weight_tolerance))
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Transport tolerances must be finite and non-negative.")
        if medium.medium_count != surfaces.refractive_indices.shape[0]:
            raise ValueError(
                "Surface and medium tables must have matching medium counts."
            )
        model = ScalarFresnelSurfaceModel() if surface_model is None else surface_model
        model.validate_surfaces(surfaces)
        self.surfaces = surfaces
        self.medium = medium
        self.surface_model = model
        self.detector_response = (
            UnitDetectorResponse() if detector_response is None else detector_response
        )
        self.variance_reduction = policy
        self.relativity = relativity
        self.maximum_interactions = maximum
        self.branch_capacity = branches
        self.photon_batch_size = batch
        self.traversal_stack_capacity = stack
        self.triangle_leaf_size = leaf
        self.ray_tolerance = tolerances[0]
        self.tie_tolerance = tolerances[1]
        self.weight_tolerance = tolerances[2]
        self.detector_arrival_capacity = arrivals


class PreparedOpticalMonteCarlo(StrictModule, NonTrainableState):
    """Prepared exact geometry and explicit fixed-work transport evidence."""

    plan: OpticalMonteCarloPlan
    triangle_query: PreparedTriangleRayQuery
    speed_of_light: float = eqx.field(static=True)
    maximum_triangle_tests: int = eqx.field(static=True)
    maximum_random_draws: int = eqx.field(static=True)
    required_bytes_per_photon: int = eqx.field(static=True)
    working_set_bytes: int = eqx.field(static=True)
    exact_visibility: bool = eqx.field(static=True)
    remaining_optical_depth: bool = eqx.field(static=True)
    implicit_capture: bool = eqx.field(static=True)
    identity_keyed_sampling: bool = eqx.field(static=True)
    pathwise_differentiability_claimed: bool = eqx.field(static=True)


# Per-lane scalar bytes: position, direction, axis (3 float64 each), Jones
# (2 complex128), wavelength, time, weight, optical depth (float64), medium
# (int32), branch (uint32), live (bool).
_LANE_BYTES = 9 * 8 + 2 * 16 + 4 * 8 + 4 + 4 + 1
# Per-arrival bytes: detector index (int32), position (3 float64), time,
# wavelength, weight, incidence cosine (float64), active (bool).
_ARRIVAL_BYTES = 4 + 3 * 8 + 4 * 8 + 1
# Keyed uniforms per lane interaction: free path, two scattering, interface,
# roulette, and the surface-model key (the model's own draws derive from it).
_DRAWS_PER_LANE_INTERACTION = 6


def prepare_optical_monte_carlo(
    plan: OpticalMonteCarloPlan,
    /,
) -> PreparedOpticalMonteCarlo:
    """Prepare exact triangle traversal and fixed work/resource evidence."""

    if not isinstance(plan, OpticalMonteCarloPlan):
        raise TypeError("plan must be an OpticalMonteCarloPlan.")
    triangle_plan = TriangleRayQueryPlan(
        plan.surfaces.vertices,
        plan.surfaces.triangles,
        entity_ids=plan.surfaces.surface_ids,
        leaf_size=plan.triangle_leaf_size,
        traversal_stack_capacity=plan.traversal_stack_capacity,
        acceleration="bvh",
        forward_tolerance=plan.ray_tolerance,
        tie_tolerance=plan.tie_tolerance,
    )
    triangle_query = prepare_triangle_ray_query(triangle_plan)
    tally_scalars = (
        plan.medium.medium_count
        + plan.surfaces.surface_count
        + plan.surfaces.detector_count
        + 5
    )
    per_photon = (
        plan.branch_capacity * _LANE_BYTES
        + tally_scalars * 8
        + plan.detector_arrival_capacity * _ARRIVAL_BYTES
        + 4
    )
    lanes = plan.maximum_interactions * plan.branch_capacity
    return PreparedOpticalMonteCarlo(
        plan,
        triangle_query,
        float(plan.relativity.speed_of_light),
        lanes * triangle_query.triangle_count,
        lanes * _DRAWS_PER_LANE_INTERACTION,
        per_photon,
        per_photon * plan.photon_batch_size,
        True,
        True,
        True,
        True,
        False,
    )


class OpticalTransportTallies(StrictModule, NonTrainableState):
    """Absorption, surface flux, detector, escape, roulette, and residual tallies."""

    absorption: Array
    surface_flux: Array
    detector: Array
    escape: Array
    roulette: Array
    live: Array
    truncated: Array
    launched: Array
    ledger_residual: Array


class OpticalDetectorArrivals(StrictModule, NonTrainableState):
    """Accepted detector arrivals per photon in history order.

    Slot ``k`` of photon ``i`` holds that history's ``k``-th accepted
    detector crossing: detector index, crossing position, lane time,
    wavelength, the weight the detector response recorded, and the
    ``|direction · normal|`` incidence cosine. Unused slots are inactive with
    detector index ``-1``; ``dropped`` counts arrivals beyond the capacity.
    ``(id_hi, id_lo)`` are the photon identities keying later draws.
    """

    __strict_contract__ = True

    detector_indices: Int32[_PhotonDim, _ArrivalDim]
    positions: Float64[_PhotonDim, _ArrivalDim, Literal[3]]
    times: Float64[_PhotonDim, _ArrivalDim]
    wavelengths: Float64[_PhotonDim, _ArrivalDim]
    weights: Float64[_PhotonDim, _ArrivalDim]
    incidence_cosines: Float64[_PhotonDim, _ArrivalDim]
    active: Bool[_PhotonDim, _ArrivalDim]
    dropped: Int32[_PhotonDim]
    id_hi: UInt32[_PhotonDim]
    id_lo: UInt32[_PhotonDim]


class OpticalTransportResult(StrictModule, NonTrainableState):
    """Per-photon and estimator-level output with uncertainty and evidence.

    ``terminal_state`` has lane shape ``(photons, branch_capacity)``;
    ``maximum_polarization_defect`` bounds the deviation of live lanes from a
    unit-norm, transverse polarization frame; ``detector_arrivals`` has
    ``plan.detector_arrival_capacity`` slots per photon.
    """

    per_photon_tallies: OpticalTransportTallies
    tallies: OpticalTransportTallies
    standard_errors: OpticalTransportTallies
    terminal_state: OpticalPhotonState
    terminal_optical_depths: Array
    terminal_live: Array
    interaction_counts: Array
    status: Array
    successful: Array
    sample_count: Array
    maximum_absolute_ledger_residual: Array
    maximum_polarization_defect: Array
    finite: Array
    all_successful: Array
    detector_arrivals: OpticalDetectorArrivals


class _Lanes(StrictModule):
    """Fixed-capacity lane bank of one photon packet inside the scan."""

    positions: Array
    directions: Array
    transverse_axes: Array
    jones_vectors: Array
    wavelengths: Array
    times: Array
    weights: Array
    media: Array
    optical_depths: Array
    branches: Array
    live: Array


class _Tallies(StrictModule):
    absorption: Array
    surface_flux: Array
    detector: Array
    escape: Array
    roulette: Array


class _Flags(StrictModule):
    ambiguity: Array
    traversal: Array
    medium: Array
    interface: Array
    branch_capacity: Array
    spectral: Array


class _Arrivals(StrictModule):
    """Per-photon detector-arrival slots filled in history order."""

    detector_indices: Array
    positions: Array
    times: Array
    wavelengths: Array
    weights: Array
    incidence_cosines: Array
    active: Array
    count: Array
    dropped: Array


type _Carry = tuple[_Lanes, _Tallies, Array, _Flags, _Arrivals]


def _lane_keys(
    root_key: PRNGKey,
    address: SampleAddress,
    interaction: Array,
    id_hi: Array,
    id_lo: Array,
    branches: Array,
) -> Array:
    def one(branch: Array) -> PRNGKey:
        return derive_key(root_key, address, interaction, id_hi, id_lo, branch)

    return jax.vmap(one)(branches)


def _uniforms(keys: Array) -> Array:
    return jax.vmap(lambda key: jr.uniform(key, dtype=jnp.float64))(keys)


def _sample_optical_depth(uniform: Array) -> Array:
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    return -jnp.log(jnp.maximum(uniform, tiny))


def _compact_lanes(capacity: int, candidates: _Lanes) -> tuple[_Lanes, Array]:
    """Keep the first ``capacity`` live candidates in candidate order."""

    count = candidates.live.shape[0]
    source_indices = jnp.arange(count, dtype=jnp.int32)
    sentinel = jnp.asarray(count, dtype=jnp.int32)
    ordering = jnp.sort(jnp.where(candidates.live, source_indices, sentinel))[:capacity]
    selected = ordering < count
    safe = jnp.minimum(ordering, count - 1)
    rank = jnp.cumsum(candidates.live.astype(jnp.int32)) - 1
    accepted = candidates.live & (rank < capacity)
    dropped = jnp.sum(jnp.where(candidates.live & ~accepted, candidates.weights, 0.0))

    def gather(values: Array) -> Array:
        mask = selected.reshape((capacity,) + (1,) * (values.ndim - 1))
        return jnp.where(mask, values[safe], jnp.zeros((), dtype=values.dtype))

    compacted = _Lanes(
        gather(candidates.positions),
        gather(candidates.directions),
        gather(candidates.transverse_axes),
        gather(candidates.jones_vectors),
        gather(candidates.wavelengths),
        gather(candidates.times),
        gather(candidates.weights),
        gather(candidates.media),
        gather(candidates.optical_depths),
        ordering.astype(jnp.uint32),
        selected,
    )
    return compacted, dropped


def _interleave(first: Array, second: Array) -> Array:
    """Candidate ``2 * slot + side`` ordering of two lane banks."""

    stacked = jnp.stack((first, second), axis=1)
    return stacked.reshape((2 * first.shape[0],) + first.shape[1:])


def _jones_to_incidence_frame(
    jones_vectors: Array,
    transverse_axes: Array,
    directions: Array,
    frame_normals: Array,
) -> Array:
    """Express carried-frame components on ``(s, direction × s)``.

    Both frames are transverse to ``directions``; leading batch axes broadcast.
    """

    second_axes = jnp.cross(directions, transverse_axes)
    p_axes = jnp.cross(directions, frame_normals)
    j_s = jones_vectors[..., 0] * jnp.sum(transverse_axes * frame_normals, axis=-1) + (
        jones_vectors[..., 1] * jnp.sum(second_axes * frame_normals, axis=-1)
    )
    j_p = jones_vectors[..., 0] * jnp.sum(transverse_axes * p_axes, axis=-1) + (
        jones_vectors[..., 1] * jnp.sum(second_axes * p_axes, axis=-1)
    )
    return jnp.stack((j_s, j_p), axis=-1)


def _record_arrivals(
    arrivals: _Arrivals,
    accepted: Array,
    detector_indices: Array,
    positions: Array,
    times: Array,
    wavelengths: Array,
    weights: Array,
    incidence_cosines: Array,
) -> _Arrivals:
    """Append accepted lanes to the next free slots in lane order."""

    capacity = arrivals.active.shape[0]
    slots = arrivals.count + jnp.cumsum(accepted.astype(jnp.int32)) - 1
    stored = accepted & (slots < capacity)
    target = jnp.where(stored, slots, capacity)

    def put(values: Array, lane_values: Array) -> Array:
        return values.at[target].set(lane_values, mode="drop")

    return _Arrivals(
        put(arrivals.detector_indices, detector_indices),
        put(arrivals.positions, positions),
        put(arrivals.times, times),
        put(arrivals.wavelengths, wavelengths),
        put(arrivals.weights, weights),
        put(arrivals.incidence_cosines, incidence_cosines),
        put(arrivals.active, stored),
        arrivals.count + jnp.sum(accepted, dtype=jnp.int32),
        arrivals.dropped + jnp.sum(accepted & ~stored, dtype=jnp.int32),
    )


def _transport_one(
    prepared: PreparedOpticalMonteCarlo,
    root_key: PRNGKey,
    photon: OpticalPhotonState,
) -> tuple[OpticalTransportTallies, _Lanes, Array, Array, Array, Array, Array, _Arrivals]:
    """Transport one photon bank through a bounded interaction scan.

    Branch creation, identity-addressed draws, polarization-basis transport,
    roulette, surface response, arrival allocation, and energy tallies stay in
    one fused scan so capacity refusal cannot expose partially committed lanes
    and the random-event address remains independent of batching.
    """
    plan = prepared.plan
    medium = plan.medium
    surfaces = plan.surfaces
    policy = plan.variance_reduction
    capacity = plan.branch_capacity
    medium_count = medium.medium_count
    initial_weight = photon.weights
    input_valid = (
        (photon.medium_indices >= 0)
        & (photon.medium_indices < medium_count)
        & (initial_weight >= 0.0)
    )
    initial_branch = jnp.zeros((), dtype=jnp.uint32)
    initial_key = derive_key(
        root_key,
        _FREE_PATH_ADDRESS,
        jnp.zeros((), dtype=jnp.int32),
        photon.id_hi,
        photon.id_lo,
        initial_branch,
    )
    initial_depth = _sample_optical_depth(jr.uniform(initial_key, dtype=jnp.float64))
    lanes = _Lanes(
        jnp.zeros((capacity, 3), dtype=jnp.float64).at[0].set(photon.positions),
        jnp.zeros((capacity, 3), dtype=jnp.float64).at[0].set(photon.directions),
        jnp.zeros((capacity, 3), dtype=jnp.float64).at[0].set(photon.transverse_axes),
        jnp.zeros((capacity, 2), dtype=jnp.complex128).at[0].set(photon.jones_vectors),
        jnp.zeros((capacity,), dtype=jnp.float64).at[0].set(photon.wavelengths),
        jnp.zeros((capacity,), dtype=jnp.float64).at[0].set(photon.times),
        jnp.zeros((capacity,), dtype=jnp.float64).at[0].set(initial_weight),
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[0]
        .set(jnp.clip(photon.medium_indices, 0, medium_count - 1)),
        jnp.zeros((capacity,), dtype=jnp.float64).at[0].set(initial_depth),
        jnp.zeros((capacity,), dtype=jnp.uint32),
        jnp.zeros((capacity,), dtype=jnp.bool_)
        .at[0]
        .set(input_valid & (initial_weight > 0.0)),
    )
    zero = jnp.zeros((), dtype=jnp.float64)
    tallies = _Tallies(
        jnp.zeros((medium_count,), dtype=jnp.float64),
        jnp.zeros((surfaces.surface_count,), dtype=jnp.float64),
        jnp.zeros((surfaces.detector_count,), dtype=jnp.float64),
        zero,
        zero,
    )
    false = jnp.zeros((), dtype=jnp.bool_)
    arrival_capacity = plan.detector_arrival_capacity
    initial: _Carry = (
        lanes,
        tallies,
        jnp.zeros((), dtype=jnp.int32),
        _Flags(false, false, false, false, false, false),
        _Arrivals(
            jnp.full((arrival_capacity,), -1, dtype=jnp.int32),
            jnp.zeros((arrival_capacity, 3), dtype=jnp.float64),
            jnp.zeros((arrival_capacity,), dtype=jnp.float64),
            jnp.zeros((arrival_capacity,), dtype=jnp.float64),
            jnp.zeros((arrival_capacity,), dtype=jnp.float64),
            jnp.zeros((arrival_capacity,), dtype=jnp.float64),
            jnp.zeros((arrival_capacity,), dtype=jnp.bool_),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        ),
    )

    def keys_for(address: SampleAddress, interaction: Array, branches: Array) -> Array:
        return _lane_keys(
            root_key, address, interaction, photon.id_hi, photon.id_lo, branches
        )

    def step(carry: _Carry, interaction: Array) -> tuple[_Carry, Array]:
        """Advance every live branch through one atomic optical interaction."""
        lanes_, tallies_, interaction_count, flags, arrivals = carry
        draw_index = interaction + 1
        hit = intersect_triangle_rays(
            prepared.triangle_query, lanes_.positions, lanes_.directions
        )
        # Lanes outside the medium's spectral tables are refused before any
        # coefficient is consumed; their weight is reported as truncated.
        spectral_failure = lanes_.live & ~medium.spectral_support(
            lanes_.media, lanes_.wavelengths
        )
        live = lanes_.live & ~spectral_failure
        query_success = live & (hit.status == int(TriangleRayIntersectionStatus.SUCCESS))
        query_miss = live & (hit.status == int(TriangleRayIntersectionStatus.MISS))
        ambiguous_hit = live & (
            hit.status == int(TriangleRayIntersectionStatus.AMBIGUOUS_HIT)
        )
        traversal_failure = live & (
            hit.status == int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED)
        )
        other_failure = live & ~(
            query_success | query_miss | ambiguous_hit | traversal_failure
        )
        failure_weight = jnp.sum(
            jnp.where(
                ambiguous_hit | traversal_failure | other_failure | spectral_failure,
                lanes_.weights,
                0.0,
            )
        )
        safe_triangle = jnp.maximum(hit.triangle_indices, 0)
        normal = hit.oriented_normals
        positive_orientation = jnp.sum(lanes_.directions * normal, axis=-1) > 0.0
        incident_medium = jnp.where(
            positive_orientation,
            surfaces.negative_medium_indices[safe_triangle],
            surfaces.positive_medium_indices[safe_triangle],
        )
        transmitted_medium = jnp.where(
            positive_orientation,
            surfaces.positive_medium_indices[safe_triangle],
            surfaces.negative_medium_indices[safe_triangle],
        )
        interface_normal = jnp.where(positive_orientation[:, None], normal, -normal)
        medium_match = query_success & (lanes_.media == incident_medium)
        mismatch = query_success & ~medium_match
        failure_weight = failure_weight + jnp.sum(
            jnp.where(mismatch, lanes_.weights, 0.0)
        )
        valid_boundary = medium_match

        extinction = medium.extinction(lanes_.media, lanes_.wavelengths)
        refractive = medium.refractive_index(lanes_.media, lanes_.wavelengths)
        boundary_distance = jnp.where(valid_boundary, hit.intersection.distances, jnp.inf)
        boundary_depth = extinction * boundary_distance
        transportable = query_miss | valid_boundary
        volume_event = (
            live
            & transportable
            & (extinction > 0.0)
            & (lanes_.optical_depths < boundary_depth)
        )
        surface_event = valid_boundary & ~volume_event
        unbounded_escape = query_miss & (extinction == 0.0)
        moved = volume_event | surface_event
        volume_distance = lanes_.optical_depths / jnp.where(
            extinction > 0.0, extinction, 1.0
        )
        event_distance = jnp.where(volume_event, volume_distance, boundary_distance)
        event_positions = lanes_.positions + (
            jnp.where(moved, event_distance, 0.0)[:, None] * lanes_.directions
        )
        event_times = lanes_.times + jnp.where(
            moved, refractive * event_distance / prepared.speed_of_light, 0.0
        )
        remaining_depth = jnp.where(
            surface_event,
            jnp.maximum(lanes_.optical_depths - boundary_depth, 0.0),
            lanes_.optical_depths,
        )
        interaction_count = (interaction_count + jnp.sum(moved, dtype=jnp.int32)).astype(
            jnp.int32
        )

        # Volume event: implicit capture, then a scattering sample in the
        # scattering-plane frame that becomes the carried frame.
        albedo = medium.albedo(lanes_.media, lanes_.wavelengths)
        absorbed_weight = jnp.where(volume_event, lanes_.weights * (1.0 - albedo), 0.0)
        absorption = tallies_.absorption.at[lanes_.media].add(absorbed_weight)
        scattered_weight = lanes_.weights * albedo
        sample = medium.scatter(
            lanes_.media,
            lanes_.wavelengths,
            lanes_.jones_vectors,
            keys_for(_SCATTERING_ADDRESS, draw_index, lanes_.branches),
        )
        second_axes = jnp.cross(lanes_.directions, lanes_.transverse_axes)
        azimuthal = (
            jnp.cos(sample.azimuths)[:, None] * lanes_.transverse_axes
            + jnp.sin(sample.azimuths)[:, None] * second_axes
        )
        sine = jnp.sqrt(jnp.maximum(0.0, 1.0 - sample.cosines * sample.cosines))
        scattered_direction, _ = _unit_rows(
            sample.cosines[:, None] * lanes_.directions + sine[:, None] * azimuthal
        )
        scattered_axis, _ = _unit_rows(jnp.cross(lanes_.directions, azimuthal))
        next_depth = _sample_optical_depth(
            _uniforms(keys_for(_FREE_PATH_ADDRESS, draw_index, lanes_.branches))
        )
        volume_live = volume_event & (scattered_weight > plan.weight_tolerance)
        failure_weight = failure_weight + jnp.sum(
            jnp.where(
                volume_event & ~volume_live & (scattered_weight > 0.0),
                scattered_weight,
                0.0,
            )
        )

        # Surface event: detector/absorber tallies are owner semantics; mirror
        # and dielectric candidates come from the surface model in the
        # incidence frame s = direction × normal (carried axis at normal incidence).
        surface_kind = surfaces.surface_kinds[safe_triangle]
        detector_surface = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.DETECTOR)
        )
        incidence_cosine = jnp.abs(jnp.sum(lanes_.directions * normal, axis=-1))
        detector_accepted = detector_surface & (
            incidence_cosine >= surfaces.detector_acceptance_cosines[safe_triangle]
        )
        detector_rejected = detector_surface & ~detector_accepted
        detector_index = jnp.maximum(surfaces.detector_indices[safe_triangle], 0)
        detector = tallies_.detector
        detector_remainder = jnp.zeros_like(lanes_.weights)
        if surfaces.detector_count > 0:
            response = plan.detector_response.respond(
                detector_index,
                lanes_.wavelengths,
                event_times,
                lanes_.jones_vectors,
                incidence_cosine,
            )
            recorded = jnp.where(detector_accepted, lanes_.weights * response, 0.0)
            detector = detector.at[detector_index].add(recorded)
            # Weight the detector does not record is absorbed at its surface.
            detector_remainder = jnp.where(
                detector_accepted, lanes_.weights - recorded, 0.0
            )
            if arrival_capacity > 0:
                arrivals = _record_arrivals(
                    arrivals,
                    detector_accepted,
                    detector_index,
                    hit.intersection.points,
                    event_times,
                    lanes_.wavelengths,
                    recorded,
                    incidence_cosine,
                )
        absorber = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.ABSORBER)
        )
        absorption = absorption.at[lanes_.media].add(
            jnp.where(absorber, lanes_.weights, 0.0) + detector_remainder
        )
        mirror = surface_event & (surface_kind == int(NonSequentialSurfaceKind.MIRROR))
        dielectric_kind = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.DIELECTRIC)
        )
        transmitted_unsupported = dielectric_kind & ~medium.spectral_support(
            jnp.maximum(transmitted_medium, 0), lanes_.wavelengths
        )
        spectral_failure = spectral_failure | transmitted_unsupported
        failure_weight = failure_weight + jnp.sum(
            jnp.where(transmitted_unsupported, lanes_.weights, 0.0)
        )
        dielectric = dielectric_kind & ~transmitted_unsupported
        interface_active = mirror | dielectric
        frame_normal, oblique = _unit_rows(jnp.cross(lanes_.directions, interface_normal))
        frame_normal = jnp.where(oblique[:, None], frame_normal, lanes_.transverse_axes)
        incidence_jones = _jones_to_incidence_frame(
            lanes_.jones_vectors, lanes_.transverse_axes, lanes_.directions, frame_normal
        )
        surface_id = surfaces.surface_ids[safe_triangle]
        interaction_result = plan.surface_model.interact(
            OpticalSurfaceHit(
                lanes_.directions,
                interface_normal,
                medium.refractive_index(
                    jnp.maximum(incident_medium, 0), lanes_.wavelengths
                ),
                medium.refractive_index(
                    jnp.maximum(transmitted_medium, 0), lanes_.wavelengths
                ),
                surface_kind,
                lanes_.wavelengths,
                incidence_jones,
                frame_normal,
                surface_id,
            ),
            keys_for(_SURFACE_ADDRESS, draw_index, lanes_.branches),
        )
        reflection_valid = interaction_result.reflection_valid
        transmission_valid = interaction_result.transmission_valid
        reflectance = interaction_result.reflectance
        transmittance = interaction_result.transmittance
        responded = interface_active & reflection_valid
        transmission_open = dielectric & transmission_valid
        interface_uniform = _uniforms(
            keys_for(_INTERFACE_ADDRESS, draw_index, lanes_.branches)
        )
        # Power the surface neither reflects nor transmits is absorbed at the
        # surface; transmitted power without a valid transmitted candidate is
        # an interface failure.
        match policy.interface_branching:
            case "stochastic":
                choose_reflection = responded & (interface_uniform < reflectance)
                transmission_drawn = (
                    responded
                    & ~choose_reflection
                    & (interface_uniform < reflectance + transmittance)
                )
                choose_transmission = transmission_drawn & transmission_open
                transmission_failure = transmission_drawn & ~transmission_open
                reflection_weight = lanes_.weights
                transmission_weight = lanes_.weights
                failed_transmission_weight = lanes_.weights
                surface_absorbed = jnp.where(
                    responded & ~choose_reflection & ~transmission_drawn,
                    lanes_.weights,
                    0.0,
                )
            case "expected-split":
                choose_reflection = responded & (reflectance > 0.0)
                choose_transmission = transmission_open & (transmittance > 0.0)
                transmission_failure = (
                    responded & (transmittance > 0.0) & ~transmission_open
                )
                reflection_weight = lanes_.weights * reflectance
                transmission_weight = lanes_.weights * transmittance
                failed_transmission_weight = transmission_weight
                surface_absorbed = jnp.where(
                    responded,
                    lanes_.weights * jnp.maximum(1.0 - reflectance - transmittance, 0.0),
                    0.0,
                )
            case _:
                assert_never(policy.interface_branching)
        interface_failure = (interface_active & ~reflection_valid) | transmission_failure
        failure_weight = failure_weight + jnp.sum(
            jnp.where(interface_active & ~reflection_valid, lanes_.weights, 0.0)
            + jnp.where(transmission_failure, failed_transmission_weight, 0.0)
        )
        absorption = absorption.at[lanes_.media].add(surface_absorbed)
        crossing_sign = jnp.where(positive_orientation, 1.0, -1.0)
        surface_flux = tallies_.surface_flux.at[surface_id].add(
            jnp.where(choose_transmission, crossing_sign * transmission_weight, 0.0)
        )
        passthrough = detector_rejected
        escape = tallies_.escape + jnp.sum(
            jnp.where(unbounded_escape, lanes_.weights, 0.0)
        )

        first = _Lanes(
            jnp.where(volume_event[:, None], event_positions, hit.intersection.points),
            jnp.where(
                passthrough[:, None],
                lanes_.directions,
                jnp.where(
                    volume_event[:, None],
                    scattered_direction,
                    interaction_result.reflected_directions,
                ),
            ),
            jnp.where(
                passthrough[:, None],
                lanes_.transverse_axes,
                jnp.where(
                    volume_event[:, None],
                    scattered_axis,
                    interaction_result.reflected_axes,
                ),
            ),
            jnp.where(
                passthrough[:, None],
                lanes_.jones_vectors,
                jnp.where(
                    volume_event[:, None],
                    sample.jones_vectors,
                    interaction_result.reflected_jones,
                ),
            ),
            jnp.where(volume_event, sample.wavelengths, lanes_.wavelengths),
            event_times + jnp.where(volume_event, sample.delays, 0.0),
            jnp.where(
                passthrough,
                lanes_.weights,
                jnp.where(volume_event, scattered_weight, reflection_weight),
            ),
            lanes_.media,
            jnp.where(
                passthrough,
                remaining_depth,
                jnp.where(volume_event, next_depth, remaining_depth),
            ),
            lanes_.branches,
            volume_live | choose_reflection | passthrough,
        )
        second = _Lanes(
            hit.intersection.points,
            interaction_result.transmitted_directions,
            interaction_result.transmitted_axes,
            interaction_result.transmitted_jones,
            lanes_.wavelengths,
            event_times,
            transmission_weight,
            transmitted_medium,
            remaining_depth,
            lanes_.branches,
            choose_transmission,
        )
        candidates = jax.tree_util.tree_map(_interleave, first, second)
        lanes_, capacity_dropped = _compact_lanes(capacity, candidates)
        failure_weight = failure_weight + capacity_dropped

        roulette_uniform = _uniforms(
            keys_for(_ROULETTE_ADDRESS, draw_index, lanes_.branches)
        )
        roulette_candidate = (
            lanes_.live
            & (policy.roulette_threshold > 0.0)
            & (lanes_.weights < policy.roulette_threshold)
        )
        survive = roulette_candidate & (
            roulette_uniform < policy.roulette_survival_probability
        )
        killed = roulette_candidate & ~survive
        boosted = lanes_.weights / policy.roulette_survival_probability
        roulette = tallies_.roulette + jnp.sum(
            jnp.where(killed, lanes_.weights, 0.0)
            + jnp.where(survive, lanes_.weights - boosted, 0.0)
        )
        lanes_ = _Lanes(
            lanes_.positions,
            lanes_.directions,
            lanes_.transverse_axes,
            lanes_.jones_vectors,
            lanes_.wavelengths,
            lanes_.times,
            jnp.where(survive, boosted, lanes_.weights),
            lanes_.media,
            lanes_.optical_depths,
            lanes_.branches,
            lanes_.live & ~killed,
        )
        flags = _Flags(
            flags.ambiguity | jnp.any(ambiguous_hit),
            flags.traversal | jnp.any(traversal_failure | other_failure),
            flags.medium | jnp.any(mismatch),
            flags.interface | jnp.any(interface_failure),
            flags.branch_capacity | (capacity_dropped > plan.weight_tolerance),
            flags.spectral | jnp.any(spectral_failure),
        )
        return (
            lanes_,
            _Tallies(absorption, surface_flux, detector, escape, roulette),
            interaction_count,
            flags,
            arrivals,
        ), failure_weight

    (
        (lanes, tallies, interaction_count, flags, arrivals),
        truncated_increments,
    ) = jax.lax.scan(
        step, initial, jnp.arange(plan.maximum_interactions, dtype=jnp.int32)
    )
    truncated = jnp.sum(truncated_increments)
    live_weight = jnp.sum(jnp.where(lanes.live, lanes.weights, 0.0))
    ledger_residual = initial_weight - (
        jnp.sum(tallies.absorption)
        + jnp.sum(tallies.detector)
        + tallies.escape
        + tallies.roulette
        + live_weight
        + truncated
    )
    jones_norm = jnp.sum(jnp.abs(lanes.jones_vectors) ** 2, axis=-1)
    axis_norm = jnp.sum(lanes.transverse_axes * lanes.transverse_axes, axis=-1)
    axis_projection = jnp.sum(lanes.transverse_axes * lanes.directions, axis=-1)
    polarization_defect = jnp.max(
        jnp.where(
            lanes.live,
            jnp.maximum(
                jnp.abs(jones_norm - 1.0),
                jnp.maximum(jnp.abs(axis_norm - 1.0), jnp.abs(axis_projection)),
            ),
            0.0,
        )
    )
    finite = (
        jnp.all(jnp.isfinite(tallies.absorption))
        & jnp.all(jnp.isfinite(tallies.surface_flux))
        & jnp.all(jnp.isfinite(tallies.detector))
        & jnp.isfinite(tallies.escape)
        & jnp.isfinite(tallies.roulette)
        & jnp.isfinite(live_weight)
        & jnp.isfinite(truncated)
        & jnp.isfinite(ledger_residual)
        & jnp.all(jnp.isfinite(lanes.positions))
        & jnp.all(jnp.isfinite(lanes.directions))
        & jnp.all(jnp.isfinite(lanes.transverse_axes))
        & jnp.all(jnp.isfinite(lanes.jones_vectors))
        & jnp.all(jnp.isfinite(lanes.times))
        & jnp.all(jnp.isfinite(lanes.weights))
        & jnp.all(jnp.isfinite(lanes.optical_depths))
        & jnp.isfinite(polarization_defect)
    )
    status = jnp.select(
        (
            ~input_valid,
            flags.ambiguity,
            flags.traversal,
            flags.medium,
            flags.interface,
            flags.branch_capacity,
            flags.spectral,
            arrivals.dropped > 0,
            live_weight > plan.weight_tolerance,
            ~finite,
        ),
        (
            int(OpticalTransportStatus.INVALID_INPUT),
            int(OpticalTransportStatus.AMBIGUOUS_INTERSECTION),
            int(OpticalTransportStatus.TRAVERSAL_CAPACITY_EXHAUSTED),
            int(OpticalTransportStatus.MEDIUM_MISMATCH),
            int(OpticalTransportStatus.INTERFACE_FAILURE),
            int(OpticalTransportStatus.BRANCH_CAPACITY_EXHAUSTED),
            int(OpticalTransportStatus.SPECTRAL_SUPPORT_EXCEEDED),
            int(OpticalTransportStatus.DETECTOR_ARRIVAL_CAPACITY_EXHAUSTED),
            int(OpticalTransportStatus.INTERACTION_CAPACITY_EXHAUSTED),
            int(OpticalTransportStatus.NONFINITE_RESULT),
        ),
        int(OpticalTransportStatus.SUCCESS),
    ).astype(jnp.int32)
    successful = status == int(OpticalTransportStatus.SUCCESS)
    per_photon = OpticalTransportTallies(
        tallies.absorption,
        tallies.surface_flux,
        tallies.detector,
        tallies.escape,
        tallies.roulette,
        live_weight,
        truncated,
        initial_weight,
        ledger_residual,
    )
    return (
        per_photon,
        lanes,
        interaction_count,
        status,
        successful,
        finite,
        polarization_defect,
        arrivals,
    )


def _standard_error(values: Array) -> Array:
    count = values.shape[0]
    mean = jnp.mean(values, axis=0)
    squared = jnp.sum((values - mean) ** 2, axis=0)
    return jnp.where(count > 1, jnp.sqrt(squared / (count * (count - 1))), 0.0)


def simulate_optical_photons(
    prepared: PreparedOpticalMonteCarlo,
    source: OpticalPhotonSource,
    key: PRNGKey,
    /,
) -> OpticalTransportResult:
    """Transport a launched photon batch with fixed work and explicit uncertainty.

    Every random draw is keyed by ``derive_key`` on the draw purpose, the
    interaction index, the photon identity words, and the lane's candidate
    branch, so every photon replays the same history regardless of batching,
    lane placement, and photon order; floating-point fields agree up to the
    rounding of differently vectorized arithmetic. Photons are processed in
    chunks of ``photon_batch_size``. This stochastic solver makes no pathwise
    differentiability claim.
    """

    if not isinstance(prepared, PreparedOpticalMonteCarlo):
        raise TypeError("prepared must be a PreparedOpticalMonteCarlo.")
    key_ = parse(key, PRNGKey, "key")
    launched = source.launch(derive_key(key_, _SOURCE_ADDRESS))
    if not isinstance(launched, OpticalPhotonState):
        raise TypeError("source.launch must return an OpticalPhotonState.")
    if launched.weights.ndim != 1:
        raise ValueError("Launched photon batches must be rank one.")
    plan = prepared.plan
    count = launched.photon_count
    batch = min(plan.photon_batch_size, count)
    chunks = -(-count // batch)
    padding = chunks * batch - count

    def simulate_one(
        photon: OpticalPhotonState,
    ) -> tuple[
        OpticalTransportTallies, _Lanes, Array, Array, Array, Array, Array, _Arrivals
    ]:
        return _transport_one(prepared, key_, photon)

    # Bounded vectorization: chunks of `batch` photons run under one vmap inside
    # a scan. Padding lanes copy photon zero with zero weight, so they launch
    # dead, and are sliced away; every real photon's draws depend only on its
    # own identity.
    def chunked(values: Array) -> Array:
        padded = jnp.concatenate(
            (values, jnp.broadcast_to(values[:1], (padding,) + values.shape[1:])),
            axis=0,
        )
        return padded.reshape((chunks, batch) + values.shape[1:])

    photons = jax.tree_util.tree_map(chunked, launched)
    photons = OpticalPhotonState(
        photons.positions,
        photons.directions,
        photons.transverse_axes,
        photons.jones_vectors,
        photons.wavelengths,
        photons.times,
        jnp.where(
            jnp.arange(chunks * batch).reshape((chunks, batch)) < count,
            photons.weights,
            0.0,
        ),
        photons.medium_indices,
        photons.id_hi,
        photons.id_lo,
    )

    def unchunk(values: Array) -> Array:
        return values.reshape((chunks * batch,) + values.shape[2:])[:count]

    (
        per_photon,
        lanes,
        interactions,
        status,
        successful,
        finite,
        polarization_defect,
        arrivals,
    ) = jax.tree_util.tree_map(unchunk, jax.lax.map(jax.vmap(simulate_one), photons))
    tallies = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), per_photon)
    standard_errors = jax.tree_util.tree_map(_standard_error, per_photon)
    terminal = OpticalPhotonState(
        lanes.positions,
        lanes.directions,
        lanes.transverse_axes,
        lanes.jones_vectors,
        lanes.wavelengths,
        lanes.times,
        lanes.weights,
        lanes.media,
        jnp.broadcast_to(launched.id_hi[:, None], lanes.weights.shape),
        jnp.broadcast_to(launched.id_lo[:, None], lanes.weights.shape),
    )
    return OpticalTransportResult(
        per_photon,
        tallies,
        standard_errors,
        terminal,
        lanes.optical_depths,
        lanes.live,
        interactions,
        status,
        successful,
        jnp.asarray(count, dtype=jnp.int32),
        jnp.max(jnp.abs(per_photon.ledger_residual)),
        jnp.max(polarization_defect),
        jnp.all(finite),
        jnp.all(successful),
        OpticalDetectorArrivals(
            arrivals.detector_indices,
            arrivals.positions,
            arrivals.times,
            arrivals.wavelengths,
            arrivals.weights,
            arrivals.incidence_cosines,
            arrivals.active,
            arrivals.dropped,
            launched.id_hi,
            launched.id_lo,
        ),
    )


__all__ = [
    "ExplicitPhotonSource",
    "OpticalDetectorArrivals",
    "OpticalDetectorResponse",
    "OpticalInterfaceBranching",
    "OpticalMedium",
    "OpticalMonteCarloPlan",
    "OpticalPhotonSource",
    "OpticalPhotonState",
    "OpticalScatteringSample",
    "OpticalSurfaceHit",
    "OpticalSurfaceInteraction",
    "OpticalSurfaceModel",
    "OpticalTransportResult",
    "OpticalTransportStatus",
    "OpticalTransportTallies",
    "OpticalVarianceReduction",
    "PreparedOpticalMonteCarlo",
    "ScalarFresnelSurfaceModel",
    "TissueOpticalMedium",
    "UnitDetectorResponse",
    "launch_optical_photons",
    "prepare_optical_monte_carlo",
    "rotate_jones_to_scattering_frame",
    "simulate_optical_photons",
]
