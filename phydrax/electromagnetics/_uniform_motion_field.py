#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-domain field of a charge in uniform motion through a homogeneous medium.

Conventions: phasors ``exp(-i ω t)`` and transforms ``F(ω) = ∫ f(t) exp(+i ω t) dt``
of a single charge on an infinite straight path ``r(t) = r₀ + v t d̂``. The medium
is homogeneous and isotropic with absolute ``ε(ω)`` and ``μ(ω)`` in any consistent
unit system (SI users pass ``ε₀ ε_r`` and ``μ₀ μ_r``).

With ``k² = ω² ε μ`` and the transverse wavenumber ``k_ρ² = k² − ω²/v²``, the
Lorenz-gauge potentials of the transformed source ``ρ̃ = (q/v) δ_⊥ exp(i ω ζ/v)``
give, with ``s = −i k_ρ`` and ``P = exp(i ω ζ / v)``,

- point charge ``q`` (3-D, cylindrical ``ζ, ρ, φ``)::

      E_ζ = −i q s² K₀(sρ) P / (2π ε ω),  E_ρ = q s K₁(sρ) P / (2π ε v),
      H_φ = q s K₁(sρ) P / (2π);

- line charge ``λ`` per unit length along ``ẑ`` moving in the plane (2-D, signed
  in-plane offset ``η`` along ``n̂ = ẑ × d̂``)::

      E_ζ = −i λ s e^{−s|η|} P / (2 ε ω),  E_η = sgn(η) λ e^{−s|η|} P / (2 ε v),
      H_z = sgn(η) λ e^{−s|η|} P / 2.

The branch is ``Im k_ρ ≥ 0``: below the Cherenkov threshold the field is bound
(``K₀, K₁`` decay over ``1/Im k_ρ``, i.e. ``γβλ/2π`` in vacuum); above it,
``K_ν(−i k_ρ ρ) ∝ H_ν⁽¹⁾(k_ρ ρ)`` is the outgoing Hankel wave. A lossless
negative-index medium takes the limit of vanishing passive loss, which reverses
the phase flow (``Re k_ρ < 0``) while energy still flows outward.
"""

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..special import kve
from ..typing import (
    Bool,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float64,
    parse,
    Scalar,
    Scope,
)


UniformMotionGeometry: TypeAlias = Literal["point", "line"]


class _FrequencyDim(Dim, minimum=1):
    """Angular frequencies."""


class _PointDim(Dim, minimum=1):
    """Evaluation points."""


class _SpaceDim(Dim, minimum=2):
    """Ambient coordinates: three for a point charge, two for a line charge."""


class _MagneticDim(Dim, minimum=1):
    """Magnetic components: three for a point charge, the out-of-plane one for a line."""


def _real_vector(value: ConvertibleToArray, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array) or not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must be real.")
    return array.astype(jnp.float64)


def _positive_scalar(value: ConvertibleToArray, name: str, /) -> Array:
    array = _real_vector(value, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    return eqx.error_if(
        array,
        ~jnp.isfinite(array) | (array <= 0.0),
        f"{name} must be finite and positive.",
    )


class UniformMotionMedium(StrictModule):
    """Homogeneous isotropic medium sampled at angular frequencies ``ω > 0``.

    ``permittivity`` and ``permeability`` are the absolute complex ``ε(ω)`` and
    ``μ(ω)`` of a passive medium (``Im ε ≥ 0``, ``Im μ ≥ 0`` for ``exp(-i ω t)``),
    with any conductivity folded into ``ε`` as ``iσ/ω``.
    """

    __strict_contract__ = True

    angular_frequencies: Float64[_FrequencyDim]
    permittivity: Complex128[_FrequencyDim]
    permeability: Complex128[_FrequencyDim]
    medium_id: str = eqx.field(static=True)

    def __init__(
        self,
        angular_frequencies: ConvertibleToArray,
        permittivity: ConvertibleToArray,
        permeability: ConvertibleToArray,
        /,
    ) -> None:
        omega = _real_vector(angular_frequencies, "angular_frequencies")
        if omega.ndim != 1:
            raise ValueError("angular_frequencies must be a vector.")
        epsilon = jnp.broadcast_to(
            jnp.asarray(permittivity).astype(jnp.complex128), omega.shape
        )
        mu = jnp.broadcast_to(
            jnp.asarray(permeability).astype(jnp.complex128), omega.shape
        )
        scope = Scope()
        self.angular_frequencies = parse(
            eqx.error_if(
                omega,
                ~jnp.isfinite(omega) | (omega <= 0.0),
                "angular_frequencies must be finite and positive.",
            ),
            Float64[_FrequencyDim],
            "angular_frequencies",
            scope=scope,
        )
        passive = (
            ~jnp.isfinite(epsilon)
            | ~jnp.isfinite(mu)
            | (jnp.imag(epsilon) < 0.0)
            | (jnp.imag(mu) < 0.0)
            | (epsilon == 0.0)
            | (mu == 0.0)
        )
        self.permittivity = parse(
            eqx.error_if(
                epsilon,
                passive,
                "The medium must be finite, nonzero, and passive (Im ε, Im μ ≥ 0).",
            ),
            Complex128[_FrequencyDim],
            "permittivity",
            scope=scope,
        )
        self.permeability = parse(
            mu, Complex128[_FrequencyDim], "permeability", scope=scope
        )
        self.medium_id = canonical_fingerprint(
            {
                "kind": "uniform-motion-medium",
                "values": array_tree_fingerprint((omega, epsilon, mu)),
            }
        )

    def transverse_wavenumber(self, speed: ConvertibleToArray, /) -> Array:
        """``k_ρ = √(ω² ε μ − ω²/v²)`` on the passive branch ``Im k_ρ ≥ 0``."""
        velocity = _positive_scalar(speed, "speed")
        omega = self.angular_frequencies
        squared = omega**2 * (self.permittivity * self.permeability - 1.0 / velocity**2)
        root = jnp.sqrt(squared)
        # Lossless media sit on the cut; the vanishing-loss limit adds i0⁺ times
        # d(εμ)/d(loss) ∝ Re ε + Re μ, which is negative in a negative-index band.
        lossless_reversed = (jnp.imag(squared) == 0.0) & (
            jnp.real(self.permittivity) + jnp.real(self.permeability) < 0.0
        )
        flip = (jnp.imag(root) < 0.0) | lossless_reversed
        return jnp.where(flip, -root, root)


class UniformMotionEvidence(StrictModule):
    """Branch and scale evidence of one uniform-motion field.

    ``radiating`` marks frequencies above the Cherenkov threshold (a propagating
    transverse wave, ``Re(εμ) v² > 1``). ``decay_length = 1/Im k_ρ`` is the
    transverse reach of the field (``γβλ/2π`` in vacuum; infinite for a lossless
    radiating medium) and ``bound_extent = 2π·decay_length`` the domain extent
    ``γβλ`` a truncated simulation must contain. ``cherenkov_cosine`` is
    ``Re(ω/v)/Re|k|``, the cosine of the cone angle for radiating frequencies.
    """

    __strict_contract__ = True

    transverse_wavenumber: Complex128[_FrequencyDim]
    radiating: Bool[_FrequencyDim]
    decay_length: Float64[_FrequencyDim]
    bound_extent: Float64[_FrequencyDim]
    cherenkov_cosine: Float64[_FrequencyDim]
    lorentz_factor: Float64[_FrequencyDim]


class UniformMotionField(StrictModule):
    """Field phasors ``[frequency, point, component]`` and their support mask.

    Points on the path itself are unsupported; their values are NaN.
    """

    __strict_contract__ = True

    electric: Complex128[_FrequencyDim, _PointDim, _SpaceDim]
    magnetic: Complex128[_FrequencyDim, _PointDim, _MagneticDim]
    supported: Bool[_PointDim]
    distance: Float64[_PointDim]


class UniformMotionFieldPlan(StrictModule):
    """Analytic field of a point (3-D) or line (2-D) charge in uniform motion.

    ``charge`` is ``q`` for ``"point"`` and the charge per unit length ``λ`` for
    ``"line"``; ``origin`` is the position at ``t = 0`` and ``direction`` the
    motion direction, both with three coordinates for a point and two in-plane
    coordinates for a line (whose field is invariant along ``ẑ``).
    """

    __strict_contract__ = True

    medium: UniformMotionMedium
    charge: Float64[Scalar]
    speed: Float64[Scalar]
    origin: Float64[_SpaceDim]
    direction: Float64[_SpaceDim]
    geometry: UniformMotionGeometry = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: UniformMotionGeometry,
        medium: UniformMotionMedium,
        /,
        *,
        charge: ConvertibleToArray,
        speed: ConvertibleToArray,
        origin: ConvertibleToArray,
        direction: ConvertibleToArray,
    ) -> None:
        geometry = parse(geometry, UniformMotionGeometry, "geometry")
        if not isinstance(medium, UniformMotionMedium):
            raise TypeError("medium must be a UniformMotionMedium.")
        match geometry:
            case "point":
                dimension = 3
            case "line":
                dimension = 2
            case _:
                assert_never(geometry)
        charge_ = _real_vector(charge, "charge")
        if charge_.shape != ():
            raise ValueError("charge must be a scalar.")
        origin_ = _real_vector(origin, "origin")
        direction_ = _real_vector(direction, "direction")
        if origin_.shape != (dimension,) or direction_.shape != (dimension,):
            raise ValueError(
                f"A {geometry} charge takes {dimension} origin and direction coordinates."
            )
        norm = jnp.linalg.norm(direction_)
        direction_ = eqx.error_if(
            direction_ / norm,
            ~jnp.isfinite(norm) | (norm == 0.0),
            "direction must be a finite nonzero vector.",
        )
        scope = Scope()
        self.medium = medium
        self.charge = parse(charge_, Float64[Scalar], "charge", scope=scope)
        self.speed = parse(
            _positive_scalar(speed, "speed"), Float64[Scalar], "speed", scope=scope
        )
        self.origin = parse(origin_, Float64[_SpaceDim], "origin", scope=scope)
        self.direction = parse(direction_, Float64[_SpaceDim], "direction", scope=scope)
        self.geometry = geometry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uniform-motion-field-plan",
                "geometry": geometry,
                "medium": medium.medium_id,
                "values": array_tree_fingerprint(
                    (charge_, self.speed, origin_, direction_)
                ),
            }
        )

    @property
    def dimension(self) -> int:
        return self.origin.shape[0]

    def evidence(self) -> UniformMotionEvidence:
        omega = self.medium.angular_frequencies
        wavenumber = self.medium.transverse_wavenumber(self.speed)
        product = self.medium.permittivity * self.medium.permeability
        radiating = jnp.real(product) * self.speed**2 > 1.0
        decay = 1.0 / jnp.imag(wavenumber)
        longitudinal = omega / self.speed
        total = jnp.sqrt(longitudinal**2 + jnp.real(wavenumber) ** 2)
        speed_ratio = jnp.real(product) * self.speed**2
        lorentz = jnp.where(
            speed_ratio < 1.0,
            1.0 / jnp.sqrt(jnp.maximum(1.0 - speed_ratio, 0.0)),
            jnp.inf,
        )
        return UniformMotionEvidence(
            transverse_wavenumber=wavenumber,
            radiating=radiating,
            decay_length=decay,
            bound_extent=2.0 * jnp.pi * decay,
            cherenkov_cosine=jnp.where(radiating, longitudinal / total, jnp.nan),
            lorentz_factor=lorentz,
        )

    def evaluate(self, points: ConvertibleToArray, /) -> UniformMotionField:
        """Evaluate ``E`` and ``H`` phasors at ``points[N, dimension]``."""
        points_ = _real_vector(points, "points")
        if points_.ndim != 2 or points_.shape[1] != self.dimension:
            raise ValueError(
                f"points must have shape (count, {self.dimension}) for a "
                f"{self.geometry} charge."
            )
        relative = points_ - self.origin
        along = relative @ self.direction
        omega = self.medium.angular_frequencies[:, None]
        epsilon = self.medium.permittivity[:, None]
        decay = (-1j * self.medium.transverse_wavenumber(self.speed))[:, None]
        phase = 1j * omega * along[None, :] / self.speed
        match self.geometry:
            case "point":
                return self._point(relative, along, omega, epsilon, decay, phase)
            case "line":
                return self._line(relative, along, omega, epsilon, decay, phase)
            case _:
                assert_never(self.geometry)

    def _point(
        self,
        relative: Array,
        along: Array,
        omega: Array,
        epsilon: Array,
        decay: Array,
        phase: Array,
        /,
    ) -> UniformMotionField:
        radial_vector = relative - along[:, None] * self.direction
        distance = jnp.linalg.norm(radial_vector, axis=1)
        supported = (distance > 0.0) & jnp.all(decay != 0.0)
        safe = jnp.where(distance > 0.0, distance, 1.0)
        radial = radial_vector / safe[:, None]
        azimuthal = jnp.cross(self.direction, radial)
        argument = decay * safe[None, :]
        # K_ν(z) = kve(z)·e^{-z}; the exponential joins the path phase so that
        # far bound fields underflow to zero instead of overflowing kve's scale.
        carrier = jnp.exp(phase - argument)
        k0 = kve(jnp.asarray(0.0, dtype=jnp.complex128), argument) * carrier
        k1 = kve(jnp.asarray(1.0, dtype=jnp.complex128), argument) * carrier
        q = self.charge
        longitudinal = -1j * q * decay**2 * k0 / (2.0 * jnp.pi * epsilon * omega)
        transverse = q * decay * k1 / (2.0 * jnp.pi * epsilon * self.speed)
        magnetic = q * decay * k1 / (2.0 * jnp.pi)
        electric = (
            longitudinal[..., None] * self.direction
            + transverse[..., None] * radial[None, :, :]
        )
        magnetic_field = magnetic[..., None] * azimuthal[None, :, :]
        invalid = ~supported[None, :, None]
        return UniformMotionField(
            electric=jnp.where(invalid, jnp.nan, electric),
            magnetic=jnp.where(invalid, jnp.nan, magnetic_field),
            supported=supported,
            distance=distance,
        )

    def _line(
        self,
        relative: Array,
        along: Array,
        omega: Array,
        epsilon: Array,
        decay: Array,
        phase: Array,
        /,
    ) -> UniformMotionField:
        normal = jnp.stack((-self.direction[1], self.direction[0]))
        offset = relative @ normal
        distance = jnp.abs(offset)
        supported = distance > 0.0
        sign = jnp.sign(offset)[None, :]
        carrier = jnp.exp(phase - decay * distance[None, :])
        density = self.charge
        longitudinal = -1j * density * decay * carrier / (2.0 * epsilon * omega)
        transverse = sign * density * carrier / (2.0 * epsilon * self.speed)
        magnetic = sign * density * carrier / 2.0
        electric = (
            longitudinal[..., None] * self.direction + transverse[..., None] * normal
        )
        invalid = ~supported[None, :, None]
        return UniformMotionField(
            electric=jnp.where(invalid, jnp.nan, electric),
            magnetic=jnp.where(invalid, jnp.nan, magnetic[..., None]),
            supported=supported,
            distance=distance,
        )

    def radial_energy_flux(self, distance: ConvertibleToArray, /) -> Array:
        """One-sided ``d²W/(dω dl)`` leaving a cylinder (point) or slab (line).

        ``(1/π) Re ∮ (Ẽ × H̃*)·n̂`` per unit path length through the cylinder of
        radius ``distance`` about the path, or per unit path and ``ẑ`` length
        through the two planes at ``±distance``. Lossless media give the
        distance-independent Cherenkov spectrum above threshold and zero below.
        """
        radius = _positive_scalar(distance, "distance")
        offset = np.zeros((1, self.dimension))
        match self.geometry:
            case "point":
                probe = jnp.asarray(offset) + self.origin
                transverse = jnp.asarray(np.cross(np.asarray(self.direction), np.eye(3)))
                # Any unit vector normal to the path: the column with the largest norm.
                column = jnp.argmax(jnp.linalg.norm(transverse, axis=1))
                normal = transverse[column] / jnp.linalg.norm(transverse[column])
                field = self.evaluate(probe + radius * normal[None, :])
                longitudinal = field.electric[:, 0] @ self.direction
                azimuthal = field.magnetic[:, 0] @ jnp.cross(self.direction, normal)
                return -2.0 * radius * jnp.real(longitudinal * jnp.conj(azimuthal))
            case "line":
                normal = jnp.stack((-self.direction[1], self.direction[0]))
                field = self.evaluate(self.origin[None, :] + radius * normal[None, :])
                longitudinal = field.electric[:, 0] @ self.direction
                return (
                    -2.0
                    / jnp.pi
                    * jnp.real(longitudinal * jnp.conj(field.magnetic[:, 0, 0]))
                )
            case _:
                assert_never(self.geometry)


__all__ = [
    "UniformMotionEvidence",
    "UniformMotionField",
    "UniformMotionFieldPlan",
    "UniformMotionGeometry",
    "UniformMotionMedium",
]
