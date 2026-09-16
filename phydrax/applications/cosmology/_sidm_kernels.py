#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import ReferenceArtifactManifest
from ._dark_sector_species import DarkSectorSpeciesPlan


IdenticalParticleConvention: TypeAlias = Literal[
    "distinguishable-full-sphere",
    "labelled-full-sphere",
    "exchange-quotient",
]
ScreeningConvention: TypeAlias = Literal[
    "finite-full-support",
    "hard-angular-cutoff",
]


def _canonical_kernel_table_bytes(
    relative_speeds: ArrayLike,
    cosines: ArrayLike,
    differential_cross_section: ArrayLike,
    azimuths: ArrayLike | None,
    /,
) -> bytes:
    arrays = (
        ("relative_speeds", np.asarray(relative_speeds, dtype="<f8")),
        ("cosines", np.asarray(cosines, dtype="<f8")),
        (
            "differential_cross_section",
            np.asarray(differential_cross_section, dtype="<f8"),
        ),
        (
            "azimuths",
            np.asarray((), dtype="<f8")
            if azimuths is None
            else np.asarray(azimuths, dtype="<f8"),
        ),
    )
    metadata = [
        {"name": name, "shape": list(value.shape), "dtype": "<f8"}
        for name, value in arrays
    ]
    header = json.dumps(
        metadata, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return (
        b"phydrax-two-body-differential-kernel-v1\0"
        + len(header).to_bytes(8, "big")
        + header
        + b"".join(np.ascontiguousarray(value).tobytes(order="C") for _, value in arrays)
    )


class DifferentialKernelEvaluation(NamedTuple):
    differential_cross_section: Array
    supported: Array


class TwoBodyKernelMoments(NamedTuple):
    total: Array
    transfer: Array
    viscosity: Array
    modified_transfer: Array
    supported: Array


class AngularSample(NamedTuple):
    cosine: Array
    azimuth: Array
    supported: Array
    normalization_residual: Array


class SmallAngleSplitEvaluation(NamedTuple):
    total: TwoBodyKernelMoments
    small: TwoBodyKernelMoments
    rare: TwoBodyKernelMoments
    reconstruction_error: Array
    supported: Array
    successful: Array


def directions_from_angles(
    relative_velocity: ArrayLike,
    cosine: ArrayLike,
    azimuth: ArrayLike,
    /,
) -> Array:
    """Map kernel angles into one canonical incoming-relative-velocity frame."""

    relative = jnp.asarray(relative_velocity)
    if relative.ndim < 1 or relative.shape[-1] != 3:
        raise ValueError("Relative velocity must end in three Cartesian components.")
    mu = jnp.asarray(cosine, dtype=relative.dtype)
    phi = jnp.asarray(azimuth, dtype=relative.dtype)
    if mu.shape != relative.shape[:-1] or phi.shape != relative.shape[:-1]:
        raise ValueError("Cosine and azimuth shapes must match the velocity batch shape.")
    speed = jnp.sqrt(ein.contract("...i,...i->...", relative, relative))
    fallback = jnp.asarray((1.0, 0.0, 0.0), dtype=relative.dtype)
    incoming = jnp.where(
        (speed > 0.0)[..., None],
        relative / jnp.where(speed > 0.0, speed, 1.0)[..., None],
        fallback,
    )
    z_axis = jnp.asarray((0.0, 0.0, 1.0), dtype=relative.dtype)
    y_axis = jnp.asarray((0.0, 1.0, 0.0), dtype=relative.dtype)
    reference = jnp.where((jnp.abs(incoming[..., 2]) < 0.9)[..., None], z_axis, y_axis)
    first = jnp.cross(reference, incoming)
    first_norm = jnp.sqrt(ein.contract("...i,...i->...", first, first))
    first = first / jnp.where(first_norm > 0.0, first_norm, 1.0)[..., None]
    second = jnp.cross(incoming, first)
    sine = jnp.sqrt(jnp.maximum(1.0 - mu * mu, 0.0))
    transverse = jnp.cos(phi)[..., None] * first + jnp.sin(phi)[..., None] * second
    return mu[..., None] * incoming + sine[..., None] * transverse


def angles_from_direction(
    relative_velocity: ArrayLike,
    direction: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Resolve one Cartesian direction in the canonical recoil frame."""

    relative = jnp.asarray(relative_velocity)
    outgoing = jnp.asarray(direction, dtype=relative.dtype)
    if relative.ndim < 1 or relative.shape[-1] != 3 or outgoing.shape != relative.shape:
        raise ValueError("Relative velocity and direction must share shape (..., 3).")
    speed = jnp.sqrt(ein.contract("...i,...i->...", relative, relative))
    fallback = jnp.asarray((1.0, 0.0, 0.0), dtype=relative.dtype)
    incoming = jnp.where(
        (speed > 0.0)[..., None],
        relative / jnp.where(speed > 0.0, speed, 1.0)[..., None],
        fallback,
    )
    z_axis = jnp.asarray((0.0, 0.0, 1.0), dtype=relative.dtype)
    y_axis = jnp.asarray((0.0, 1.0, 0.0), dtype=relative.dtype)
    reference = jnp.where((jnp.abs(incoming[..., 2]) < 0.9)[..., None], z_axis, y_axis)
    first = jnp.cross(reference, incoming)
    first_norm = jnp.sqrt(ein.contract("...i,...i->...", first, first))
    first = first / jnp.where(first_norm > 0.0, first_norm, 1.0)[..., None]
    second = jnp.cross(incoming, first)
    cosine = jnp.clip(ein.contract("...i,...i->...", outgoing, incoming), -1.0, 1.0)
    first_component = ein.contract("...i,...i->...", outgoing, first)
    second_component = ein.contract("...i,...i->...", outgoing, second)
    azimuth = jnp.mod(jnp.arctan2(second_component, first_component), 2.0 * jnp.pi)
    return cosine, azimuth


def _integrate_piecewise_linear(
    nodes: np.ndarray,
    values: np.ndarray,
    lower: float,
    upper: float,
    moment: int,
) -> float:
    """Integrate a piecewise-linear angular density against one exact moment."""

    gauss_nodes = np.asarray((-0.7745966692414834, 0.0, 0.7745966692414834), dtype=float)
    gauss_weights = np.asarray((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0), dtype=float)
    total = 0.0
    for index in range(nodes.size - 1):
        left = max(float(nodes[index]), lower)
        right = min(float(nodes[index + 1]), upper)
        if right <= left:
            continue
        # Split the only non-polynomial weight at its cusp.  Three-point Gauss
        # is then exact for every linear-density moment used by this module.
        boundaries = (left, 0.0, right) if left < 0.0 < right else (left, right)
        for first, second in zip(boundaries[:-1], boundaries[1:], strict=True):
            midpoint = 0.5 * (first + second)
            half_width = 0.5 * (second - first)
            points = midpoint + half_width * gauss_nodes
            fraction = (points - nodes[index]) / (nodes[index + 1] - nodes[index])
            density = values[index] + fraction * (values[index + 1] - values[index])
            if moment == 0:
                weight = np.ones_like(points)
            elif moment == 1:
                weight = 1.0 - points
            elif moment == 2:
                weight = 1.0 - points * points
            elif moment == 3:
                weight = 1.0 - np.abs(points)
            else:
                raise ValueError("Unknown two-body angular moment.")
            total += half_width * float(np.sum(gauss_weights * density * weight))
    return total


def _node_moments(
    cosines: np.ndarray,
    differential: np.ndarray,
    azimuths: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if azimuths is None:
        marginal = 2.0 * np.pi * differential
    else:
        widths = np.diff(azimuths)
        marginal = np.sum(
            0.5
            * (differential[..., :-1] + differential[..., 1:])
            * widths[None, None, :],
            axis=-1,
        )
    rows = []
    for speed_index, values in enumerate(marginal):
        rows.append(
            tuple(
                _integrate_piecewise_linear(
                    cosines,
                    values,
                    float(lower[speed_index]),
                    float(upper[speed_index]),
                    moment,
                )
                for moment in range(4)
            )
        )
    return np.asarray(rows)


def _interval(nodes: Array, query: Array, /) -> tuple[Array, Array, Array]:
    upper = jnp.searchsorted(nodes, query, side="right")
    lower_index = jnp.clip(upper - 1, 0, nodes.size - 2).astype(jnp.int32)
    upper_index = lower_index + 1
    left = nodes[lower_index]
    right = nodes[upper_index]
    fraction = (query - left) / (right - left)
    return lower_index, upper_index, fraction


def _piecewise_linear_sample(
    key: Array,
    nodes: Array,
    values: Array,
    lower: Array,
    upper: Array,
    /,
) -> tuple[Array, Array, Array]:
    density_scale = jnp.max(values)
    normalized = values / jnp.where(density_scale > 0.0, density_scale, 1.0)
    segment_left = jnp.maximum(nodes[:-1], lower)
    segment_right = jnp.minimum(nodes[1:], upper)
    widths = jnp.maximum(segment_right - segment_left, 0.0)
    node_widths = nodes[1:] - nodes[:-1]
    left_fraction = (segment_left - nodes[:-1]) / node_widths
    right_fraction = (segment_right - nodes[:-1]) / node_widths
    left_density = normalized[:-1] + left_fraction * (normalized[1:] - normalized[:-1])
    right_density = normalized[:-1] + right_fraction * (normalized[1:] - normalized[:-1])
    masses = 0.5 * (left_density + right_density) * widths
    masses = jnp.where(widths > 0.0, masses, 0.0)
    cumulative = jnp.cumsum(masses)
    total = jnp.sum(masses)
    finite = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower < upper)
        & jnp.all(jnp.isfinite(values))
        & jnp.all(values >= 0.0)
        & jnp.isfinite(density_scale)
        & (density_scale > 0.0)
        & jnp.isfinite(total)
        & (total > 0.0)
    )
    safe_total = jnp.where(finite, total, 1.0)
    target = jr.uniform(key, (), dtype=values.dtype) * safe_total
    segment = jnp.clip(
        jnp.searchsorted(cumulative, target, side="right"), 0, masses.size - 1
    )
    previous = jnp.where(segment > 0, cumulative[segment - 1], 0.0)
    local_mass = target - previous
    width = widths[segment]
    density_left = left_density[segment]
    density_right = right_density[segment]
    delta = density_right - density_left
    target_density = local_mass / jnp.where(width > 0.0, width, 1.0)
    discriminant = jnp.maximum(
        density_left * density_left + 2.0 * delta * target_density, 0.0
    )
    root_denominator = density_left + jnp.sqrt(discriminant)
    linear_fraction = target_density / jnp.maximum(
        density_left, jnp.finfo(values.dtype).tiny
    )
    quadratic_fraction = (
        2.0 * target_density / jnp.where(root_denominator != 0.0, root_denominator, 1.0)
    )
    relative_scale = jnp.maximum(
        jnp.maximum(jnp.abs(density_left), jnp.abs(density_right)),
        jnp.finfo(values.dtype).tiny,
    )
    fraction = jnp.where(
        jnp.abs(delta) <= 32.0 * jnp.finfo(values.dtype).eps * relative_scale,
        linear_fraction,
        quadratic_fraction,
    )
    fraction = jnp.clip(fraction, 0.0, 1.0)
    reconstructed_mass = width * (
        density_left * fraction + 0.5 * delta * fraction * fraction
    )
    inversion_defect = jnp.abs(reconstructed_mass - local_mass)
    sample = segment_left[segment] + fraction * width
    sample = jnp.where(finite, sample, lower)
    cdf_total = cumulative[-1]
    monotonic_defect = jnp.maximum(
        -jnp.min(
            jnp.diff(jnp.concatenate((jnp.zeros((1,), dtype=values.dtype), cumulative)))
        ),
        0.0,
    )
    normalization_residual = jnp.where(
        finite,
        jnp.maximum(
            jnp.maximum(jnp.abs(cdf_total - total), monotonic_defect),
            inversion_defect,
        )
        / jnp.maximum(total, jnp.finfo(values.dtype).tiny),
        jnp.inf,
    )
    return sample, finite, normalization_residual


class TwoBodyDifferentialKernelPlan(StrictModule, NonTrainableState):
    """Bounded differential two-body cross section on ``dΩ = dμ dφ``.

    Axisymmetric tables have shape ``(speed, cosine)`` and are interpreted as
    ``dσ/dΩ`` with the azimuth integrated analytically over ``2π``.  A supplied
    azimuth axis includes both 0 and 2π and the table has shape
    ``(speed, cosine, azimuth)``. Direct tables require verified reference rights
    and provenance; arrays are stop-gradient and evaluation fails outside support.
    The constant-isotropic class factory is a closed in-code analytic specialization.
    """

    first_species: DarkSectorSpeciesPlan
    second_species: DarkSectorSpeciesPlan
    relative_speeds: Array
    cosines: Array
    differential_cross_section: Array
    azimuths: Array | None
    total_cross_sections: Array
    transfer_cross_sections: Array
    viscosity_cross_sections: Array
    modified_transfer_cross_sections: Array
    identical_particle_convention: IdenticalParticleConvention = eqx.field(static=True)
    source_artifact: ScientificArtifactEnvelope | None
    reference_manifest: ReferenceArtifactManifest | None
    commercial_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    screening_convention: ScreeningConvention = eqx.field(static=True)
    minimum_scattering_angle: float = eqx.field(static=True)
    speed_unit: str = eqx.field(static=True)
    cross_section_unit: str = eqx.field(static=True)
    normalization_tolerance: float = eqx.field(static=True)
    unbounded_speed: bool = eqx.field(static=True)
    isotropic_specialization: bool = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_species: DarkSectorSpeciesPlan,
        second_species: DarkSectorSpeciesPlan,
        relative_speeds: ArrayLike,
        cosines: ArrayLike,
        differential_cross_section: ArrayLike,
        /,
        *,
        azimuths: ArrayLike | None = None,
        identical_particle_convention: IdenticalParticleConvention = (
            "distinguishable-full-sphere"
        ),
        source_artifact: ScientificArtifactEnvelope,
        reference_manifest: ReferenceArtifactManifest,
        commercial_use: bool,
        redistribution: bool,
        training_use: bool,
        export: bool,
        screening_convention: ScreeningConvention = "finite-full-support",
        minimum_scattering_angle: float = 0.0,
        speed_unit: str = "physical-length/physical-time",
        cross_section_unit: str = "physical-area",
        normalization_tolerance: float = 1.0e-10,
        unbounded_speed: bool = False,
        isotropic_specialization: bool = False,
    ):
        if not isinstance(first_species, DarkSectorSpeciesPlan) or not isinstance(
            second_species, DarkSectorSpeciesPlan
        ):
            raise TypeError("Two-body kernels require two DarkSectorSpeciesPlan objects.")
        speeds = np.asarray(relative_speeds, dtype=float)
        mu = np.asarray(cosines, dtype=float)
        values = np.asarray(differential_cross_section, dtype=float)
        phi = None if azimuths is None else np.asarray(azimuths, dtype=float)
        if not isinstance(source_artifact, ScientificArtifactEnvelope):
            raise TypeError(
                "External differential kernels require ScientificArtifactEnvelope."
            )
        if not isinstance(reference_manifest, ReferenceArtifactManifest):
            raise TypeError(
                "External differential kernels require ReferenceArtifactManifest."
            )
        requested = (commercial_use, redistribution, training_use, export)
        if any(not isinstance(value, bool) for value in requested):
            raise TypeError(
                "External differential kernels require four explicit boolean "
                "requested-use flags."
            )
        rights = tuple(requested)
        if source_artifact.status != "complete":
            raise ValueError("Differential-kernel source artifacts must be complete.")
        payload = _canonical_kernel_table_bytes(speeds, mu, values, phi)
        reference_manifest.verify_bytes(payload)
        if (
            source_artifact.content_digest != reference_manifest.checksum
            or source_artifact.license_id != reference_manifest.license_id
            or tuple(sorted(source_artifact.parent_artifact_ids))
            != reference_manifest.lineage_ids
        ):
            raise ValueError(
                "Differential-kernel manifest and artifact digest, license, or "
                "lineage disagree."
            )
        reference_manifest.require_rights(
            commercial_use=rights[0],
            redistribution=rights[1],
            training_use=rights[2],
            export=rights[3],
        )
        requested_use_id = canonical_fingerprint(
            {
                "kind": "differential-kernel-requested-use",
                "reference_manifest": reference_manifest.manifest_id,
                "commercial_use": rights[0],
                "redistribution": rights[1],
                "training_use": rights[2],
                "export": rights[3],
            }
        )
        convention = str(identical_particle_convention)
        screening = str(screening_convention)
        minimum_angle = float(minimum_scattering_angle)
        tolerance = float(normalization_tolerance)
        speed_unit_ = str(speed_unit).strip()
        cross_section_unit_ = str(cross_section_unit).strip()
        same_species = first_species.species_plan_id == second_species.species_plan_id

        if speeds.ndim != 1 or speeds.size < 2:
            raise ValueError(
                "Kernel relative_speeds must be a vector with at least two nodes."
            )
        if (
            not np.all(np.isfinite(speeds))
            or np.any(speeds < 0.0)
            or np.any(np.diff(speeds) <= 0.0)
        ):
            raise ValueError(
                "Kernel speed nodes must be finite, nonnegative, and increasing."
            )
        if mu.ndim != 1 or mu.size < 2:
            raise ValueError("Kernel cosines must be a vector with at least two nodes.")
        if not np.all(np.isfinite(mu)) or np.any(np.diff(mu) <= 0.0):
            raise ValueError(
                "Kernel cosine nodes must be finite and strictly increasing."
            )
        if convention not in (
            "distinguishable-full-sphere",
            "labelled-full-sphere",
            "exchange-quotient",
        ):
            raise ValueError("Unknown identical-particle convention.")
        if same_species and convention == "distinguishable-full-sphere":
            raise ValueError(
                "Identical incoming species require an explicit identical convention."
            )
        if not same_species and convention != "distinguishable-full-sphere":
            raise ValueError(
                "Distinct incoming species require distinguishable-full-sphere."
            )
        quotient = convention == "exchange-quotient"
        expected_lower = 0.0 if quotient else -1.0
        if not np.isclose(mu[0], expected_lower, rtol=0.0, atol=tolerance):
            raise ValueError("Kernel cosine support has the wrong lower endpoint.")
        if screening not in ("finite-full-support", "hard-angular-cutoff"):
            raise ValueError("Unknown forward-screening convention.")
        if not np.isfinite(minimum_angle) or minimum_angle < 0.0:
            raise ValueError("minimum_scattering_angle must be finite and nonnegative.")
        if screening == "finite-full-support":
            if minimum_angle != 0.0 or not np.isclose(
                mu[-1], 1.0, rtol=0.0, atol=tolerance
            ):
                raise ValueError(
                    "Finite full-support kernels must cover the forward endpoint."
                )
        else:
            maximum_angle = 0.5 * np.pi if quotient else np.pi
            if not 0.0 < minimum_angle < maximum_angle or not np.isclose(
                mu[-1], np.cos(minimum_angle), rtol=0.0, atol=tolerance
            ):
                raise ValueError(
                    "Hard-cutoff kernels must end exactly at cos(minimum_scattering_angle)."
                )
        expected_shape = (speeds.size, mu.size)
        if phi is not None:
            if phi.ndim != 1 or phi.size < 3:
                raise ValueError(
                    "Kernel azimuths must have at least three periodic nodes."
                )
            if (
                not np.all(np.isfinite(phi))
                or np.any(np.diff(phi) <= 0.0)
                or not np.isclose(phi[0], 0.0, rtol=0.0, atol=tolerance)
                or not np.isclose(phi[-1], 2.0 * np.pi, rtol=0.0, atol=tolerance)
            ):
                raise ValueError("Kernel azimuths must increase from 0 through 2π.")
            expected_shape += (phi.size,)
        if values.shape != expected_shape:
            raise ValueError(
                f"Differential cross-section table must have shape {expected_shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Differential cross sections must be finite and nonnegative."
            )
        if phi is not None and not np.allclose(
            values[..., 0], values[..., -1], rtol=tolerance, atol=0.0
        ):
            raise ValueError(
                "Full-azimuth differential tables must close at the periodic seam."
            )
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("normalization_tolerance must be finite and positive.")
        if not speed_unit_ or not cross_section_unit_:
            raise ValueError("Kernel speed and cross-section units must be explicit.")
        if bool(unbounded_speed) and not np.allclose(
            values,
            np.broadcast_to(values[0], values.shape),
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError("Unbounded-speed kernels must be exactly speed independent.")
        if bool(isotropic_specialization) and (
            phi is not None
            or not np.allclose(
                values,
                np.full(values.shape, values.reshape((-1,))[0]),
                rtol=tolerance,
                atol=0.0,
            )
        ):
            raise ValueError(
                "The isotropic specialization must be constant over speed and angle."
            )

        lower = np.full(speeds.size, mu[0])
        upper = np.full(speeds.size, mu[-1])
        moments = _node_moments(mu, values, phi, lower, upper)
        if not np.all(np.isfinite(moments)) or np.any(moments < -tolerance):
            raise ValueError(
                "Kernel angular integration produced invalid cross sections."
            )
        moments = np.maximum(moments, 0.0)

        speed_array = jax.lax.stop_gradient(jnp.asarray(speeds))
        cosine_array = jax.lax.stop_gradient(jnp.asarray(mu, dtype=speed_array.dtype))
        value_array = jax.lax.stop_gradient(jnp.asarray(values, dtype=speed_array.dtype))
        azimuth_array = (
            None
            if phi is None
            else jax.lax.stop_gradient(jnp.asarray(phi, dtype=speed_array.dtype))
        )
        moment_array = jax.lax.stop_gradient(
            jnp.asarray(moments, dtype=speed_array.dtype)
        )
        self.first_species = first_species
        self.second_species = second_species
        self.relative_speeds = speed_array
        self.cosines = cosine_array
        self.differential_cross_section = value_array
        self.azimuths = azimuth_array
        self.source_artifact = source_artifact
        self.reference_manifest = reference_manifest
        self.commercial_use = rights[0]
        self.redistribution = rights[1]
        self.training_use = rights[2]
        self.export = rights[3]
        self.requested_use_id = requested_use_id
        self.total_cross_sections = moment_array[:, 0]
        self.transfer_cross_sections = moment_array[:, 1]
        self.viscosity_cross_sections = moment_array[:, 2]
        self.modified_transfer_cross_sections = moment_array[:, 3]
        self.identical_particle_convention = convention
        self.screening_convention = screening
        self.minimum_scattering_angle = minimum_angle
        self.speed_unit = speed_unit_
        self.cross_section_unit = cross_section_unit_
        self.normalization_tolerance = tolerance
        self.unbounded_speed = bool(unbounded_speed)
        self.isotropic_specialization = bool(isotropic_specialization)
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "two-body-differential-kernel",
                "first_species": first_species.species_plan_id,
                "second_species": second_species.species_plan_id,
                "identical_particle_convention": convention,
                "source_artifact": (
                    None if source_artifact is None else source_artifact.artifact_id
                ),
                "reference_manifest": (
                    None if reference_manifest is None else reference_manifest.manifest_id
                ),
                "requested_use": requested_use_id,
                "screening_convention": screening,
                "minimum_scattering_angle": minimum_angle,
                "speed_unit": speed_unit_,
                "cross_section_unit": cross_section_unit_,
                "normalization_tolerance": tolerance,
                "unbounded_speed": bool(unbounded_speed),
                "isotropic_specialization": bool(isotropic_specialization),
                "arrays": array_tree_fingerprint(
                    (speed_array, cosine_array, value_array, azimuth_array, moment_array)
                ),
            }
        )

    @staticmethod
    def canonical_table_bytes(
        relative_speeds: ArrayLike,
        cosines: ArrayLike,
        differential_cross_section: ArrayLike,
        /,
        *,
        azimuths: ArrayLike | None = None,
    ) -> bytes:
        """Return the exact canonical bytes governed by a reference manifest."""

        return _canonical_kernel_table_bytes(
            relative_speeds, cosines, differential_cross_section, azimuths
        )

    @classmethod
    def constant_isotropic(
        cls,
        species: DarkSectorSpeciesPlan,
        cross_section: float,
        /,
        *,
        cross_section_unit: str = "physical-area",
    ) -> TwoBodyDifferentialKernelPlan:
        """Create the unbounded-speed constant-isotropic specialization."""

        if not isinstance(species, DarkSectorSpeciesPlan):
            raise TypeError("Constant isotropic kernels require DarkSectorSpeciesPlan.")
        value = float(cross_section)
        unit = str(cross_section_unit).strip()
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "Constant isotropic cross section must be finite and nonnegative."
            )
        if not unit:
            raise ValueError("Constant isotropic cross-section unit must be explicit.")
        speeds = jnp.asarray((0.0, 1.0))
        cosines = jnp.asarray((-1.0, 0.0, 1.0), dtype=speeds.dtype)
        differential = jnp.full((2, 3), value / (4.0 * jnp.pi), dtype=speeds.dtype)
        moments = jnp.asarray(
            ((value, value, 2.0 * value / 3.0, value / 2.0),) * 2,
            dtype=speeds.dtype,
        )
        requested_use_id = canonical_fingerprint(
            {"kind": "internal-constant-isotropic-kernel-use"}
        )
        plan = object.__new__(cls)
        object.__setattr__(plan, "first_species", species)
        object.__setattr__(plan, "second_species", species)
        object.__setattr__(plan, "relative_speeds", speeds)
        object.__setattr__(plan, "cosines", cosines)
        object.__setattr__(plan, "differential_cross_section", differential)
        object.__setattr__(plan, "azimuths", None)
        object.__setattr__(plan, "source_artifact", None)
        object.__setattr__(plan, "reference_manifest", None)
        object.__setattr__(plan, "commercial_use", False)
        object.__setattr__(plan, "redistribution", False)
        object.__setattr__(plan, "training_use", False)
        object.__setattr__(plan, "export", False)
        object.__setattr__(plan, "requested_use_id", requested_use_id)
        object.__setattr__(plan, "total_cross_sections", moments[:, 0])
        object.__setattr__(plan, "transfer_cross_sections", moments[:, 1])
        object.__setattr__(plan, "viscosity_cross_sections", moments[:, 2])
        object.__setattr__(plan, "modified_transfer_cross_sections", moments[:, 3])
        object.__setattr__(plan, "identical_particle_convention", "labelled-full-sphere")
        object.__setattr__(plan, "screening_convention", "finite-full-support")
        object.__setattr__(plan, "minimum_scattering_angle", 0.0)
        object.__setattr__(plan, "speed_unit", "physical-length/physical-time")
        object.__setattr__(plan, "cross_section_unit", unit)
        object.__setattr__(plan, "normalization_tolerance", 1.0e-10)
        object.__setattr__(plan, "unbounded_speed", True)
        object.__setattr__(plan, "isotropic_specialization", True)
        object.__setattr__(
            plan,
            "kernel_id",
            canonical_fingerprint(
                {
                    "kind": "internal-constant-isotropic-two-body-kernel",
                    "species": species.species_plan_id,
                    "cross_section": value,
                    "cross_section_unit": unit,
                    "requested_use": requested_use_id,
                    "arrays": array_tree_fingerprint(
                        (speeds, cosines, differential, moments)
                    ),
                }
            ),
        )
        return plan

    def _speed_row(self, table: Array, relative_speed: Array, /) -> tuple[Array, Array]:
        speed = jnp.asarray(relative_speed, dtype=self.relative_speeds.dtype).reshape(())
        finite = jnp.isfinite(speed) & (speed >= 0.0)
        supported = finite & (
            self.unbounded_speed
            | ((speed >= self.relative_speeds[0]) & (speed <= self.relative_speeds[-1]))
        )
        query = jnp.clip(speed, self.relative_speeds[0], self.relative_speeds[-1])
        lower, upper, fraction = _interval(self.relative_speeds, query)
        fraction = jnp.where(self.unbounded_speed, 0.0, fraction)
        row = table[lower] + fraction * (table[upper] - table[lower])
        return row, supported

    def _evaluate_scalar(
        self, relative_speed: Array, cosine: Array, azimuth: Array
    ) -> tuple[Array, Array]:
        row, speed_supported = self._speed_row(
            self.differential_cross_section, relative_speed
        )
        mu = jnp.asarray(cosine, dtype=self.cosines.dtype).reshape(())
        mu_supported = (
            jnp.isfinite(mu) & (mu >= self.cosines[0]) & (mu <= self.cosines[-1])
        )
        mu_query = jnp.clip(mu, self.cosines[0], self.cosines[-1])
        lower, upper, fraction = _interval(self.cosines, mu_query)
        if self.azimuths is None:
            value = row[lower] + fraction * (row[upper] - row[lower])
            phi_supported = jnp.isfinite(azimuth)
        else:
            angular_row = row[lower] + fraction * (row[upper] - row[lower])
            phi = jnp.asarray(azimuth, dtype=self.azimuths.dtype).reshape(())
            phi_supported = (
                jnp.isfinite(phi) & (phi >= self.azimuths[0]) & (phi <= self.azimuths[-1])
            )
            phi_query = jnp.clip(phi, self.azimuths[0], self.azimuths[-1])
            phi_lower, phi_upper, phi_fraction = _interval(self.azimuths, phi_query)
            value = angular_row[phi_lower] + phi_fraction * (
                angular_row[phi_upper] - angular_row[phi_lower]
            )
        supported = speed_supported & mu_supported & phi_supported
        return jnp.where(supported, value, 0.0), supported

    def evaluate(
        self,
        relative_speed: ArrayLike,
        cosine: ArrayLike,
        azimuth: ArrayLike | None = None,
        /,
    ) -> DifferentialKernelEvaluation:
        """Interpolate ``dσ/dΩ`` and return explicit support evidence."""

        speed = jnp.asarray(relative_speed, dtype=self.relative_speeds.dtype)
        mu = jnp.asarray(cosine, dtype=self.cosines.dtype)
        if azimuth is None:
            phi = jnp.zeros_like(jnp.broadcast_arrays(speed, mu)[0])
        else:
            phi = jnp.asarray(azimuth, dtype=self.cosines.dtype)
        speed, mu, phi = jnp.broadcast_arrays(speed, mu, phi)
        values, supported = jax.vmap(self._evaluate_scalar)(
            speed.reshape((-1,)), mu.reshape((-1,)), phi.reshape((-1,))
        )
        return DifferentialKernelEvaluation(
            values.reshape(speed.shape), supported.reshape(speed.shape)
        )

    def _moments_scalar(self, relative_speed: Array, table: Array) -> tuple[Array, Array]:
        return self._speed_row(table, relative_speed)

    def moments(self, relative_speed: ArrayLike, /) -> TwoBodyKernelMoments:
        """Interpolate total, transfer, viscosity, and modified-transfer moments."""

        speed = jnp.asarray(relative_speed, dtype=self.relative_speeds.dtype)
        table = jnp.stack(
            (
                self.total_cross_sections,
                self.transfer_cross_sections,
                self.viscosity_cross_sections,
                self.modified_transfer_cross_sections,
            ),
            axis=-1,
        )
        values, supported = jax.vmap(lambda item: self._moments_scalar(item, table))(
            speed.reshape((-1,))
        )
        values = values.reshape(speed.shape + (4,))
        supported = supported.reshape(speed.shape)
        return TwoBodyKernelMoments(
            values[..., 0],
            values[..., 1],
            values[..., 2],
            values[..., 3],
            supported,
        )

    def _marginal_density(self, relative_speed: Array, /) -> tuple[Array, Array]:
        row, supported = self._speed_row(self.differential_cross_section, relative_speed)
        if self.azimuths is None:
            return 2.0 * jnp.pi * row, supported
        widths = jnp.diff(self.azimuths)
        marginal = jnp.sum(0.5 * (row[:, :-1] + row[:, 1:]) * widths[None, :], axis=-1)
        return marginal, supported

    def _sample_angles_scalar(
        self,
        key: Array,
        relative_speed: Array,
        cosine_minimum: Array,
        cosine_maximum: Array,
    ) -> AngularSample:
        mu_key, phi_key = jr.split(key)
        marginal, speed_supported = self._marginal_density(relative_speed)
        bounds_supported = (
            jnp.isfinite(cosine_minimum)
            & jnp.isfinite(cosine_maximum)
            & (cosine_minimum >= self.cosines[0])
            & (cosine_maximum <= self.cosines[-1])
            & (cosine_minimum < cosine_maximum)
        )
        mu, mu_supported, mu_residual = _piecewise_linear_sample(
            mu_key,
            self.cosines,
            marginal,
            cosine_minimum,
            cosine_maximum,
        )
        if self.azimuths is None:
            phi = 2.0 * jnp.pi * jr.uniform(phi_key, (), dtype=self.cosines.dtype)
            phi_supported = jnp.asarray(True)
            phi_residual = jnp.asarray(0.0, dtype=self.cosines.dtype)
        else:
            row, _ = self._speed_row(self.differential_cross_section, relative_speed)
            lower, upper, fraction = _interval(self.cosines, mu)
            phi_density = row[lower] + fraction * (row[upper] - row[lower])
            phi, phi_supported, phi_residual = _piecewise_linear_sample(
                phi_key,
                self.azimuths,
                phi_density,
                self.azimuths[0],
                self.azimuths[-1],
            )
        supported = speed_supported & bounds_supported & mu_supported & phi_supported
        residual = jnp.maximum(mu_residual, phi_residual)
        return AngularSample(mu, phi, supported, residual)

    def sample_angles(
        self,
        key: Array,
        relative_speed: ArrayLike,
        /,
        *,
        cosine_minimum: ArrayLike | None = None,
        cosine_maximum: ArrayLike | None = None,
    ) -> AngularSample:
        """Sample a bounded monotone inverse CDF from the tabulated angular law."""

        speed = jnp.asarray(relative_speed, dtype=self.relative_speeds.dtype)
        minimum = (
            jnp.asarray(self.cosines[0], dtype=self.cosines.dtype)
            if cosine_minimum is None
            else jnp.asarray(cosine_minimum, dtype=self.cosines.dtype)
        )
        maximum = (
            jnp.asarray(self.cosines[-1], dtype=self.cosines.dtype)
            if cosine_maximum is None
            else jnp.asarray(cosine_maximum, dtype=self.cosines.dtype)
        )
        if speed.shape != () or minimum.shape != () or maximum.shape != ():
            raise ValueError(
                "sample_angles consumes one key and one scalar angular query."
            )
        return self._sample_angles_scalar(key, speed, minimum, maximum)


class SmallAngleSplitPlan(StrictModule, NonTrainableState):
    """Disjoint small/rare angular partition with moment reconstruction evidence."""

    kernel: TwoBodyDifferentialKernelPlan
    split_cosines: Array
    rare_cosine_bounds: Array
    small_cosine_bounds: Array
    small_moments: Array
    rare_moments: Array
    reconstruction_error: Array
    gap_width: Array
    overlap_width: Array
    no_gap: Array
    no_overlap: Array
    successful: Array
    reconstruction_tolerance: float = eqx.field(static=True)
    split_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel: TwoBodyDifferentialKernelPlan,
        split_cosines: ArrayLike,
        /,
        *,
        reconstruction_tolerance: float = 1.0e-10,
    ):
        if not isinstance(kernel, TwoBodyDifferentialKernelPlan):
            raise TypeError(
                "Small-angle splitting requires TwoBodyDifferentialKernelPlan."
            )
        split = np.asarray(split_cosines, dtype=float)
        if split.shape == ():
            split = np.full(kernel.relative_speeds.shape, float(split))
        expected = tuple(kernel.relative_speeds.shape)
        tolerance = float(reconstruction_tolerance)
        if split.shape != expected:
            raise ValueError(f"split_cosines must be scalar or have shape {expected}.")
        if not np.all(split == split[0]):
            raise ValueError(
                "Small-angle split cosine must be speed independent so sampling and "
                "moment partitions share one exact angular measure."
            )
        if (
            not np.all(np.isfinite(split))
            or np.any(split <= float(kernel.cosines[0]))
            or np.any(split >= float(kernel.cosines[-1]))
        ):
            raise ValueError(
                "Every split cosine must lie strictly inside kernel support."
            )
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("reconstruction_tolerance must be finite and positive.")

        cosine_host = np.asarray(kernel.cosines)
        differential_host = np.asarray(kernel.differential_cross_section)
        azimuth_host = None if kernel.azimuths is None else np.asarray(kernel.azimuths)
        minimum = np.full(split.shape, cosine_host[0])
        maximum = np.full(split.shape, cosine_host[-1])
        rare = _node_moments(cosine_host, differential_host, azimuth_host, minimum, split)
        small = _node_moments(
            cosine_host, differential_host, azimuth_host, split, maximum
        )
        total = np.stack(
            (
                np.asarray(kernel.total_cross_sections),
                np.asarray(kernel.transfer_cross_sections),
                np.asarray(kernel.viscosity_cross_sections),
                np.asarray(kernel.modified_transfer_cross_sections),
            ),
            axis=-1,
        )
        scale = np.maximum(np.abs(total), np.finfo(total.dtype).tiny)
        error = np.abs((small + rare) - total) / scale
        rare_bounds = np.stack((minimum, split), axis=-1)
        small_bounds = np.stack((split, maximum), axis=-1)
        gap = np.maximum(small_bounds[:, 0] - rare_bounds[:, 1], 0.0)
        overlap = np.maximum(rare_bounds[:, 1] - small_bounds[:, 0], 0.0)
        no_gap = np.all(gap == 0.0)
        no_overlap = np.all(overlap == 0.0)
        successful = no_gap and no_overlap and bool(np.all(error <= tolerance))

        dtype = kernel.relative_speeds.dtype
        split_array = jax.lax.stop_gradient(jnp.asarray(split, dtype=dtype))
        rare_bounds_array = jax.lax.stop_gradient(jnp.asarray(rare_bounds, dtype=dtype))
        small_bounds_array = jax.lax.stop_gradient(jnp.asarray(small_bounds, dtype=dtype))
        small_array = jax.lax.stop_gradient(jnp.asarray(small, dtype=dtype))
        rare_array = jax.lax.stop_gradient(jnp.asarray(rare, dtype=dtype))
        error_array = jax.lax.stop_gradient(jnp.asarray(error, dtype=dtype))
        self.kernel = kernel
        self.split_cosines = split_array
        self.rare_cosine_bounds = rare_bounds_array
        self.small_cosine_bounds = small_bounds_array
        self.small_moments = small_array
        self.rare_moments = rare_array
        self.reconstruction_error = error_array
        self.gap_width = jnp.asarray(gap, dtype=dtype)
        self.overlap_width = jnp.asarray(overlap, dtype=dtype)
        self.no_gap = jnp.asarray(no_gap)
        self.no_overlap = jnp.asarray(no_overlap)
        self.successful = jnp.asarray(successful)
        self.reconstruction_tolerance = tolerance
        self.split_id = canonical_fingerprint(
            {
                "kind": "small-angle-kernel-split",
                "kernel": kernel.kernel_id,
                "reconstruction_tolerance": tolerance,
                "arrays": array_tree_fingerprint(
                    (
                        split_array,
                        rare_bounds_array,
                        small_bounds_array,
                        small_array,
                        rare_array,
                        error_array,
                    )
                ),
            }
        )

    def _table_moments(self, speed: Array, table: Array) -> TwoBodyKernelMoments:
        values, supported = self.kernel._speed_row(table, speed)
        return TwoBodyKernelMoments(values[0], values[1], values[2], values[3], supported)

    def _moments_scalar(self, speed: Array) -> SmallAngleSplitEvaluation:
        total = self.kernel.moments(speed)
        small = self._table_moments(speed, self.small_moments)
        rare = self._table_moments(speed, self.rare_moments)
        error, error_supported = self.kernel._speed_row(self.reconstruction_error, speed)
        supported = total.supported & small.supported & rare.supported & error_supported
        successful = (
            supported & self.successful & jnp.all(error <= self.reconstruction_tolerance)
        )
        return SmallAngleSplitEvaluation(total, small, rare, error, supported, successful)

    def moments(self, relative_speed: ArrayLike, /) -> SmallAngleSplitEvaluation:
        """Return the total and complementary small/rare moment partitions."""

        speed = jnp.asarray(relative_speed, dtype=self.kernel.relative_speeds.dtype)
        if speed.shape == ():
            return self._moments_scalar(speed)
        flat = jax.vmap(self._moments_scalar)(speed.reshape((-1,)))

        def reshape_moments(value: TwoBodyKernelMoments) -> TwoBodyKernelMoments:
            return TwoBodyKernelMoments(
                value.total.reshape(speed.shape),
                value.transfer.reshape(speed.shape),
                value.viscosity.reshape(speed.shape),
                value.modified_transfer.reshape(speed.shape),
                value.supported.reshape(speed.shape),
            )

        return SmallAngleSplitEvaluation(
            reshape_moments(flat.total),
            reshape_moments(flat.small),
            reshape_moments(flat.rare),
            flat.reconstruction_error.reshape(speed.shape + (4,)),
            flat.supported.reshape(speed.shape),
            flat.successful.reshape(speed.shape),
        )

    def sample_rare_angles(
        self, key: Array, relative_speed: ArrayLike, /
    ) -> AngularSample:
        """Sample only the rare complement ``μ ∈ [μ_min, μ_split]``."""

        speed = jnp.asarray(relative_speed, dtype=self.kernel.relative_speeds.dtype)
        if speed.shape != ():
            raise ValueError("sample_rare_angles consumes one key and one scalar speed.")
        split, supported = self.kernel._speed_row(self.split_cosines, speed)
        sample = self.kernel.sample_angles(
            key,
            speed,
            cosine_minimum=self.kernel.cosines[0],
            cosine_maximum=split,
        )
        return AngularSample(
            sample.cosine,
            sample.azimuth,
            sample.supported & supported & self.successful,
            sample.normalization_residual,
        )


__all__ = [
    "AngularSample",
    "angles_from_direction",
    "directions_from_angles",
    "DifferentialKernelEvaluation",
    "IdenticalParticleConvention",
    "ScreeningConvention",
    "SmallAngleSplitEvaluation",
    "SmallAngleSplitPlan",
    "TwoBodyDifferentialKernelPlan",
    "TwoBodyKernelMoments",
]
