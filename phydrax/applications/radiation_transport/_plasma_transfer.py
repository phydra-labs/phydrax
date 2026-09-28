#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mode-resolved polarized transfer along cold-plasma rays.

In a refracting, birefringent medium the transported quantity is ``I/n_r²``
(Bekefi 1966), with ``n_r`` the ray refractive index of `ColdPlasmaRayPath`.
Coefficients per unit length along the wave normal and per wave-normal solid
angle (`MagnetobremsstrahlungPlan`, ``α_σ`` and ``j_σ``) enter as

    d(S/n_r²)/dℓ = Σ_σ (j_σ/n_σ²)(1, ŝ_σ) − K S/n_r²,    dℓ = cos α ds,

which reaches ``I_σ/n_r² = j_σ/(n_σ² α_σ)`` (Kirchhoff) independently of
``n_r``. Two mode-coupling limits are provided:

- ``"weak"``: the modes propagate independently (geometric optics of each mode);
  the ray carries only its own mode, whose polarization follows the local mode
  ``ŝ_σ``; valid while `ColdPlasmaRayPath.coupling_parameter` stays small.
- ``"strong"``: the full Stokes vector is transported with the coupled
  propagation matrix (mode absorption plus cold-plasma Faraday rotation and
  conversion), so the polarization need not follow the modes; the two modes
  must share one ray (weak anisotropy, small ``index_splitting``).

Both limits compose the canonical polarized transfer
(`PolarizedRadiativeTransferPlan`, exact exponential per segment).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from math import isfinite
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...electromagnetics import ColdPlasmaDielectric, MagnetobremsstrahlungPlan
from ...optics.geometric import ColdPlasmaHamiltonian, ColdPlasmaRayPath
from ...typing import (
    as_array,
    Bool,
    Dim,
    Float64,
    Int32,
    parse,
    Scalar,
    Scope,
)
from ..astrophysics._radiative_transfer import PolarizedRadiativeTransferPlan


ModeCouplingLimit: TypeAlias = Literal["weak", "strong"]


class _RayDim(Dim, minimum=1):
    """Rays of the sampled plasma path."""


class _SegmentDim(Dim, minimum=1):
    """Segments of each ray."""


class PlasmaRayTransferStatus(IntFlag):
    """Per-ray evidence of `PlasmaRayTransferResult.status`.

    ``QUASI_TRANSVERSE`` qualifies a result (the path crosses a region where the
    quasi-transverse part of the discriminant dominates, where mode coupling is
    decided); every other flag makes the ray unsuccessful.
    """

    NONE = 0
    PATH_INVALID = 1
    COEFFICIENT_UNSUPPORTED = 2
    WEAK_COUPLING_VIOLATED = 4
    ANISOTROPY_VIOLATED = 8
    NONFINITE = 16
    EXPONENTIAL_UNCONVERGED = 32
    QUASI_TRANSVERSE = 64


_TRANSFER_FAILURE = (
    PlasmaRayTransferStatus.PATH_INVALID
    | PlasmaRayTransferStatus.COEFFICIENT_UNSUPPORTED
    | PlasmaRayTransferStatus.WEAK_COUPLING_VIOLATED
    | PlasmaRayTransferStatus.ANISOTROPY_VIOLATED
    | PlasmaRayTransferStatus.NONFINITE
    | PlasmaRayTransferStatus.EXPONENTIAL_UNCONVERGED
).value


class PlasmaPathCoefficients(StrictModule):
    """Mode-resolved coefficients on each path segment, ordered (ray mode, companion).

    ``emission`` is ``j_σ`` per volume, angular frequency and wave-normal
    steradian and ``absorption`` ``α_σ`` per length along the wave normal (the
    `MagnetobremsstrahlungPlan` convention); unsupported values are NaN and
    ``status`` carries the producer's per-mode status bits.
    """

    __strict_contract__ = True

    emission: Float64[_RayDim, _SegmentDim, Literal[2]]
    absorption: Float64[_RayDim, _SegmentDim, Literal[2]]
    status: Int32[_RayDim, _SegmentDim, Literal[2]]


def magnetobremsstrahlung_path_coefficients(
    hamiltonian: ColdPlasmaHamiltonian,
    path: ColdPlasmaRayPath,
    plan_factory: Callable[[ColdPlasmaDielectric, np.ndarray], MagnetobremsstrahlungPlan],
    /,
) -> PlasmaPathCoefficients:
    """Evaluate `MagnetobremsstrahlungPlan` at every segment of a plasma path.

    Host preparation: for each segment midpoint the local homogeneous
    `ColdPlasmaDielectric` is built from the path's profile and
    ``plan_factory(dielectric, midpoint)`` supplies the emitting population;
    coefficients are evaluated at the ray's ``(ω, θ)`` and matched to the ray
    mode and its companion by refractive index.
    """
    if not isinstance(hamiltonian, ColdPlasmaHamiltonian):
        raise TypeError("hamiltonian must be a ColdPlasmaHamiltonian.")
    if not isinstance(path, ColdPlasmaRayPath):
        raise TypeError("path must be a ColdPlasmaRayPath.")
    if path.hamiltonian_id != hamiltonian.hamiltonian_id:
        raise ValueError("path was not sampled from this ColdPlasmaHamiltonian.")
    if not callable(plan_factory):
        raise TypeError("plan_factory must be callable.")
    midpoints = np.asarray(path.midpoints)
    angles = np.asarray(path.angle)
    indices = np.asarray(path.refractive_index)
    omega = float(np.asarray(path.angular_frequency))
    shape = angles.shape
    emission = np.full(shape + (2,), np.nan, dtype=np.float64)
    absorption = np.full(shape + (2,), np.nan, dtype=np.float64)
    status = np.zeros(shape + (2,), dtype=np.int32)
    for ray, segment in np.ndindex(shape):
        point = midpoints[ray, segment]
        plan = plan_factory(hamiltonian.profile.dielectric_at(point), point)
        if not isinstance(plan, MagnetobremsstrahlungPlan):
            raise TypeError("plan_factory must return a MagnetobremsstrahlungPlan.")
        result = plan.evaluate(omega, float(angles[ray, segment]))
        roots = np.asarray(result.faraday.wave.refractive_index.real)
        first = int(
            np.argmin(np.abs(roots - indices[ray, segment, 0]), axis=-1).reshape(())
        )
        order = np.asarray((first, 1 - first))
        emission[ray, segment] = np.asarray(result.emission)[order]
        absorption[ray, segment] = np.asarray(result.absorption)[order]
        status[ray, segment] = np.asarray(result.status)[order]
    return PlasmaPathCoefficients(
        emission=jnp.asarray(emission),
        absorption=jnp.asarray(absorption),
        status=jnp.asarray(status),
    )


class PlasmaRayTransferResult(StrictModule):
    """Emergent Stokes vectors and coupling evidence per ray.

    ``stokes`` is ``(I, Q, U, V)`` at the last segment in its transported basis
    ``polarization_basis``; ``invariant`` is ``stokes / n_r²`` there and
    ``incident_invariant`` the transported quantity at the first segment, so a
    lossless source-free path returns ``invariant == incident_invariant``.
    ``faraday_rotation = ∫ ρ_V dℓ`` is the rotation of the plane of linear
    polarization (rotation measure times ``λ²``), ``faraday_conversion``
    ``∫ |ρ_lin| dℓ``.
    """

    __strict_contract__ = True

    stokes: Float64[_RayDim, Literal[4]]
    invariant: Float64[_RayDim, Literal[4]]
    incident_invariant: Float64[_RayDim, Literal[4]]
    faraday_rotation: Float64[_RayDim]
    faraday_conversion: Float64[_RayDim]
    maximum_coupling_parameter: Float64[_RayDim]
    maximum_index_splitting: Float64[_RayDim]
    quasi_transverse_segments: Int32[_RayDim]
    status: Int32[_RayDim]
    successful: Bool[Scalar]
    coupling: ModeCouplingLimit = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _propagation_matrix(
    absorption: Array, mode_stokes: Array, faraday: Array, /
) -> Array:
    """``K`` of ``dS/dℓ = j − K S`` from mode absorptions and ``2 Re ρ``."""
    absorption_i = 0.5 * jnp.sum(absorption, axis=-1)
    a_q, a_u, a_v = jnp.moveaxis(
        0.5 * jnp.sum(absorption[..., None] * mode_stokes, axis=-2), -1, 0
    )
    r_q, r_u, r_v = jnp.moveaxis(faraday, -1, 0)
    return jnp.stack(
        (
            jnp.stack((absorption_i, a_q, a_u, a_v), axis=-1),
            jnp.stack((a_q, absorption_i, r_v, -r_u), axis=-1),
            jnp.stack((a_u, -r_v, absorption_i, r_q), axis=-1),
            jnp.stack((a_v, r_u, -r_q, absorption_i), axis=-1),
        ),
        axis=-2,
    )


class PlasmaRayTransferPlan(StrictModule, NonTrainableState):
    """Polarized transfer of ``S/n_r²`` along every ray of a `ColdPlasmaRayPath`.

    ``coupling`` selects the mode-coupling limit (see the module docstring).
    ``coupling_tolerance`` bounds the path's coupling parameter in the weak
    limit and ``anisotropy_tolerance`` the relative index splitting of the two
    modes in the strong limit. Segment lengths along the wave normal are host
    plan structure; coefficients and incident Stokes vectors are dynamic.
    """

    path: ColdPlasmaRayPath
    transfers: tuple[PolarizedRadiativeTransferPlan, ...]
    coupling: ModeCouplingLimit = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)
    anisotropy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: ColdPlasmaRayPath,
        /,
        *,
        coupling: ModeCouplingLimit,
        coupling_tolerance: float = 0.1,
        anisotropy_tolerance: float = 1.0e-2,
    ) -> None:
        if not isinstance(path, ColdPlasmaRayPath):
            raise TypeError("path must be a ColdPlasmaRayPath.")
        coupling_ = parse(coupling, ModeCouplingLimit, "coupling")
        tolerances = (float(coupling_tolerance), float(anisotropy_tolerance))
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError(
                "coupling_tolerance and anisotropy_tolerance must be finite and positive."
            )
        lengths = np.asarray(path.normal_lengths)
        if not np.all(np.isfinite(lengths)) or np.any(lengths < 0.0):
            raise ValueError(
                "path normal lengths must be finite and nonnegative; resample rays "
                "whose segments are invalid."
            )
        self.path = path
        self.transfers = tuple(
            PolarizedRadiativeTransferPlan(
                lengths[ray], plan_id=f"{path.hamiltonian_id}:plasma-ray:{ray}"
            )
            for ray in range(lengths.shape[0])
        )
        self.coupling = coupling_
        self.coupling_tolerance, self.anisotropy_tolerance = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "plasma-ray-transfer",
                "hamiltonian": path.hamiltonian_id,
                "rays": path.ray_plan_id,
                "lengths": array_tree_fingerprint(lengths),
                "coupling": coupling_,
                "coupling_tolerance": tolerances[0],
                "anisotropy_tolerance": tolerances[1],
            }
        )

    def evaluate(
        self, emission: ArrayLike, absorption: ArrayLike, incident: ArrayLike, /
    ) -> PlasmaRayTransferResult:
        """Transfer incident Stokes vectors ``[ray, 4]`` (first-segment basis).

        ``emission`` and ``absorption`` are ``[ray, segment, 2]`` mode-resolved
        coefficients ordered (ray mode, companion), for example
        `PlasmaPathCoefficients` fields.
        """
        path = self.path
        scope = Scope()
        emission_ = as_array(
            emission, Float64[_RayDim, _SegmentDim, Literal[2]], "emission", scope=scope
        )
        absorption_ = as_array(
            absorption,
            Float64[_RayDim, _SegmentDim, Literal[2]],
            "absorption",
            scope=scope,
        )
        incident_ = as_array(
            incident, Float64[_RayDim, Literal[4]], "incident", scope=scope
        )
        if emission_.shape != path.refractive_index.shape:
            raise ValueError("Coefficients must match the path's ray and segment axes.")
        index_squared = path.refractive_index * path.refractive_index
        entry = path.ray_index_squared[:, 0]
        exit_ = path.ray_index_squared[:, -1]
        mode_stokes = path.mode_stokes
        match self.coupling:
            case "weak":
                ray_stokes = mode_stokes[..., 0, :]
                source = jnp.zeros(emission_.shape[:2] + (4,), dtype=jnp.float64)
                source = source.at[..., 0].set(emission_[..., 0] / index_squared[..., 0])
                eye = jnp.eye(4, dtype=jnp.float64)
                matrix = absorption_[..., 0, None, None] * eye[None, None]
                intensity = 0.5 * (
                    incident_[:, 0]
                    + jnp.sum(ray_stokes[:, 0] * incident_[:, 1:], axis=-1)
                )
                initial = jnp.zeros_like(incident_).at[:, 0].set(intensity / entry)
                incident_invariant = initial[:, 0, None] * jnp.concatenate(
                    (jnp.ones_like(entry)[:, None], ray_stokes[:, 0]), axis=-1
                )
                used = jnp.isfinite(emission_[..., 0]) & jnp.isfinite(absorption_[..., 0])
            case "strong":
                weighted = emission_ / index_squared
                source = jnp.concatenate(
                    (
                        jnp.sum(weighted, axis=-1)[..., None],
                        jnp.sum(weighted[..., None] * mode_stokes, axis=-2),
                    ),
                    axis=-1,
                )
                matrix = _propagation_matrix(absorption_, mode_stokes, path.faraday)
                initial = incident_ / entry[:, None]
                incident_invariant = initial
                used = jnp.all(
                    jnp.isfinite(emission_) & jnp.isfinite(absorption_), axis=-1
                )
            case _:
                assert_never(self.coupling)
        results = tuple(
            transfer.evaluate(source[ray], matrix[ray], initial[ray])
            for ray, transfer in enumerate(self.transfers)
        )
        emergent = jnp.stack(tuple(value.emergent for value in results))
        converged = jnp.stack(tuple(value.valid for value in results))
        match self.coupling:
            case "weak":
                invariant = emergent[:, 0, None] * jnp.concatenate(
                    (jnp.ones_like(exit_)[:, None], mode_stokes[:, -1, 0]), axis=-1
                )
            case "strong":
                invariant = emergent
            case _:
                assert_never(self.coupling)
        stokes = invariant * exit_[:, None]
        rotation = 0.5 * jnp.sum(path.faraday[..., 2] * path.normal_lengths, axis=-1)
        conversion = 0.5 * jnp.sum(
            jnp.sqrt(path.faraday[..., 0] ** 2 + path.faraday[..., 1] ** 2)
            * path.normal_lengths,
            axis=-1,
        )
        coupling = jnp.max(path.coupling_parameter, axis=-1)
        splitting = jnp.max(path.index_splitting, axis=-1)
        quasi_transverse = jnp.sum(path.quasi_transverse, axis=-1, dtype=jnp.int32)
        finite = jnp.all(jnp.isfinite(stokes), axis=-1)
        status = PlasmaRayTransferStatus
        weak = self.coupling == "weak"
        strong = self.coupling == "strong"
        flags = (
            (~jnp.all(path.valid, axis=-1)).astype(jnp.int32) * status.PATH_INVALID.value
            | (~jnp.all(used, axis=-1)).astype(jnp.int32)
            * status.COEFFICIENT_UNSUPPORTED.value
            | (weak & ~(coupling <= self.coupling_tolerance)).astype(jnp.int32)
            * status.WEAK_COUPLING_VIOLATED.value
            | (strong & ~(splitting <= self.anisotropy_tolerance)).astype(jnp.int32)
            * status.ANISOTROPY_VIOLATED.value
            | (~finite).astype(jnp.int32) * status.NONFINITE.value
            | (finite & ~converged).astype(jnp.int32)
            * status.EXPONENTIAL_UNCONVERGED.value
            | (quasi_transverse > 0).astype(jnp.int32) * status.QUASI_TRANSVERSE.value
        )
        return PlasmaRayTransferResult(
            stokes=stokes,
            invariant=invariant,
            incident_invariant=incident_invariant,
            faraday_rotation=rotation,
            faraday_conversion=conversion,
            maximum_coupling_parameter=coupling,
            maximum_index_splitting=splitting,
            quasi_transverse_segments=quasi_transverse,
            status=flags,
            successful=jnp.all((flags & _TRANSFER_FAILURE) == 0),
            coupling=self.coupling,
            plan_id=self.plan_id,
        )


__all__ = [
    "magnetobremsstrahlung_path_coefficients",
    "ModeCouplingLimit",
    "PlasmaPathCoefficients",
    "PlasmaRayTransferPlan",
    "PlasmaRayTransferResult",
    "PlasmaRayTransferStatus",
]
