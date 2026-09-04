#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_VACUUM_PERMITTIVITY = 8.8541878128e-12


class SBSStatus(IntEnum):
    """Portable status for one stimulated-Brillouin interaction."""

    SUCCESS = 0
    NONFINITE_OVERLAP = 1
    INVALID_INTERFACE_EVIDENCE = 2
    INVALID_NORMALIZATION = 3
    NONFINITE_INTERACTION = 4


class SBSSharedDomainMap(StrictModule, NonTrainableState):
    """Explicit interpolation of three native meshes to common SBS quadrature.

    Boundary normals point from the material-minus side to the material-plus
    side. Interface jumps supplied to :class:`SBSOverlapPlan` are therefore
    always ``minus - plus``. Requiring these literal conventions prevents an
    unoriented moving-boundary overlap from being mistaken for a shape gradient.
    """

    volume_weights: Array
    pump_to_volume: Array
    stokes_to_volume: Array
    acoustic_to_volume: Array
    boundary_weights: Array
    pump_to_boundary: Array
    stokes_to_boundary: Array
    acoustic_to_boundary: Array
    boundary_normals: Array
    normal_unit_error: Array
    spatial_dimension: int = eqx.field(static=True)
    normal_orientation: str = eqx.field(static=True)
    jump_convention: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_weights: ArrayLike,
        pump_to_volume: ArrayLike,
        stokes_to_volume: ArrayLike,
        acoustic_to_volume: ArrayLike,
        boundary_weights: ArrayLike,
        pump_to_boundary: ArrayLike,
        stokes_to_boundary: ArrayLike,
        acoustic_to_boundary: ArrayLike,
        boundary_normals: ArrayLike,
        /,
        *,
        normal_orientation: str,
        jump_convention: str,
        normal_tolerance: float = 1e-8,
        map_id: str | None = None,
    ):
        volume = np.asarray(volume_weights)
        pump_volume = np.asarray(pump_to_volume)
        stokes_volume = np.asarray(stokes_to_volume)
        acoustic_volume = np.asarray(acoustic_to_volume)
        boundary = np.asarray(boundary_weights)
        pump_boundary = np.asarray(pump_to_boundary)
        stokes_boundary = np.asarray(stokes_to_boundary)
        acoustic_boundary = np.asarray(acoustic_to_boundary)
        normals = np.asarray(boundary_normals)
        if volume.ndim != 1 or volume.size < 1:
            raise ValueError("volume_weights must be one non-empty vector.")
        if boundary.ndim != 1:
            raise ValueError("boundary_weights must be one vector.")
        if np.iscomplexobj(volume) or np.iscomplexobj(boundary):
            raise ValueError("SBS quadrature weights must be real.")
        if (
            np.any(~np.isfinite(volume))
            or np.any(volume < 0.0)
            or not np.any(volume > 0.0)
            or np.any(~np.isfinite(boundary))
            or np.any(boundary < 0.0)
        ):
            raise ValueError("SBS quadrature weights must be finite and non-negative.")
        _validate_interpolation_map(pump_volume, volume.size, "pump_to_volume")
        _validate_interpolation_map(stokes_volume, volume.size, "stokes_to_volume")
        _validate_interpolation_map(acoustic_volume, volume.size, "acoustic_to_volume")
        _validate_interpolation_map(pump_boundary, boundary.size, "pump_to_boundary")
        _validate_interpolation_map(stokes_boundary, boundary.size, "stokes_to_boundary")
        _validate_interpolation_map(
            acoustic_boundary, boundary.size, "acoustic_to_boundary"
        )
        if (
            pump_boundary.shape[1] != pump_volume.shape[1]
            or stokes_boundary.shape[1] != stokes_volume.shape[1]
            or acoustic_boundary.shape[1] != acoustic_volume.shape[1]
        ):
            raise ValueError(
                "Each field must use one native sample axis for volume and boundary maps."
            )
        if normals.ndim != 2 or normals.shape[0] != boundary.size or normals.shape[1] < 1:
            raise ValueError(
                "boundary_normals must have shape (boundary point, dimension)."
            )
        if np.iscomplexobj(normals) or np.any(~np.isfinite(normals)):
            raise ValueError("boundary_normals must be finite and real.")
        tolerance = float(normal_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("normal_tolerance must be finite and non-negative.")
        normal_error = (
            0.0
            if boundary.size == 0
            else float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)))
        )
        if normal_error > tolerance:
            raise ValueError(
                "boundary_normals must be unit vectors within normal_tolerance."
            )
        if normal_orientation != "material-minus-to-plus":
            raise ValueError("normal_orientation must be 'material-minus-to-plus'.")
        if jump_convention != "minus-minus-plus":
            raise ValueError("jump_convention must be 'minus-minus-plus'.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "sbs-shared-domain-map",
                    "volume_weights": array_tree_fingerprint(volume),
                    "pump_to_volume": array_tree_fingerprint(pump_volume),
                    "stokes_to_volume": array_tree_fingerprint(stokes_volume),
                    "acoustic_to_volume": array_tree_fingerprint(acoustic_volume),
                    "boundary_weights": array_tree_fingerprint(boundary),
                    "pump_to_boundary": array_tree_fingerprint(pump_boundary),
                    "stokes_to_boundary": array_tree_fingerprint(stokes_boundary),
                    "acoustic_to_boundary": array_tree_fingerprint(acoustic_boundary),
                    "boundary_normals": array_tree_fingerprint(normals),
                    "normal_orientation": normal_orientation,
                    "jump_convention": jump_convention,
                }
            )
            if map_id is None
            else str(map_id)
        )
        if not identifier:
            raise ValueError("map_id must be non-empty.")
        self.volume_weights = jnp.asarray(volume)
        self.pump_to_volume = jnp.asarray(pump_volume)
        self.stokes_to_volume = jnp.asarray(stokes_volume)
        self.acoustic_to_volume = jnp.asarray(acoustic_volume)
        self.boundary_weights = jnp.asarray(boundary)
        self.pump_to_boundary = jnp.asarray(pump_boundary)
        self.stokes_to_boundary = jnp.asarray(stokes_boundary)
        self.acoustic_to_boundary = jnp.asarray(acoustic_boundary)
        self.boundary_normals = jnp.asarray(normals)
        self.normal_unit_error = jnp.asarray(normal_error)
        self.spatial_dimension = int(normals.shape[1])
        self.normal_orientation = normal_orientation
        self.jump_convention = jump_convention
        self.map_id = identifier

    def map_pump_volume(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.pump_to_volume, values, "pump volume")

    def map_stokes_volume(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.stokes_to_volume, values, "Stokes volume")

    def map_acoustic_volume(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.acoustic_to_volume, values, "acoustic volume")

    def map_pump_boundary(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.pump_to_boundary, values, "pump boundary")

    def map_stokes_boundary(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.stokes_to_boundary, values, "Stokes boundary")

    def map_acoustic_boundary(self, values: ArrayLike, /) -> Array:
        return _map_samples(self.acoustic_to_boundary, values, "acoustic boundary")


class SBSInteractionCoefficients(StrictModule, NonTrainableState):
    """Frequencies, propagation constants, loss, and finite interaction length.

    The quality factor is the energy-decay convention, so the angular
    linewidth is ``Ω/Q`` and the acoustic power attenuation is
    ``(Ω/Q)/abs(v_g)``.
    """

    pump_angular_frequency: Array
    stokes_angular_frequency: Array
    acoustic_angular_frequency: Array
    pump_propagation_constant: Array
    stokes_propagation_constant: Array
    acoustic_wavenumber: Array
    acoustic_quality_factor: Array
    acoustic_group_velocity: Array
    interaction_length: Array
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pump_angular_frequency: float,
        stokes_angular_frequency: float,
        acoustic_angular_frequency: float,
        pump_propagation_constant: complex,
        stokes_propagation_constant: complex,
        acoustic_wavenumber: complex,
        acoustic_quality_factor: float,
        acoustic_group_velocity: float,
        interaction_length: float,
        coefficient_id: str | None = None,
    ):
        frequencies = tuple(
            float(value)
            for value in (
                pump_angular_frequency,
                stokes_angular_frequency,
                acoustic_angular_frequency,
            )
        )
        quality = float(acoustic_quality_factor)
        velocity = float(acoustic_group_velocity)
        length = float(interaction_length)
        propagation = tuple(
            complex(value)
            for value in (
                pump_propagation_constant,
                stokes_propagation_constant,
                acoustic_wavenumber,
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
            raise ValueError("SBS angular frequencies must be positive and finite.")
        if not math.isfinite(quality) or quality <= 0.0:
            raise ValueError("acoustic_quality_factor must be positive and finite.")
        if not math.isfinite(velocity) or velocity == 0.0:
            raise ValueError("acoustic_group_velocity must be finite and nonzero.")
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("interaction_length must be positive and finite.")
        if any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in propagation
        ):
            raise ValueError("SBS propagation constants must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "sbs-interaction-coefficients",
                    "frequencies": frequencies,
                    "propagation": [[value.real, value.imag] for value in propagation],
                    "quality": quality,
                    "group_velocity": velocity,
                    "interaction_length": length,
                }
            )
            if coefficient_id is None
            else str(coefficient_id)
        )
        if not identifier:
            raise ValueError("coefficient_id must be non-empty.")
        self.pump_angular_frequency = jnp.asarray(frequencies[0])
        self.stokes_angular_frequency = jnp.asarray(frequencies[1])
        self.acoustic_angular_frequency = jnp.asarray(frequencies[2])
        self.pump_propagation_constant = jnp.asarray(propagation[0])
        self.stokes_propagation_constant = jnp.asarray(propagation[1])
        self.acoustic_wavenumber = jnp.asarray(propagation[2])
        self.acoustic_quality_factor = jnp.asarray(quality)
        self.acoustic_group_velocity = jnp.asarray(velocity)
        self.interaction_length = jnp.asarray(length)
        self.coefficient_id = identifier


class SBSOverlapPlan(StrictModule):
    """Native optical/acoustic samples and constitutive SBS coupling data.

    Electric fields use V/m, electric displacements C/m², acoustic
    displacements m, volume/boundary weights m²/m, optical powers W, and
    acoustic energy per length J/m. With peak-phasor time averaging both
    overlaps have units J/m.

    ``relative_permittivity_jump`` and ``inverse_relative_permittivity_jump``
    are the oriented material-minus value minus the material-plus value. This
    plan evaluates the moving-interface overlap only; it deliberately makes no
    claim to be an interface shape derivative.
    """

    domain_map: SBSSharedDomainMap
    pump_electric: Array
    pump_electric_displacement: Array
    stokes_electric: Array
    stokes_electric_displacement: Array
    acoustic_displacement: Array
    acoustic_strain: Array
    relative_permittivity: Array
    photoelastic_tensor: Array
    relative_permittivity_jump: Array
    inverse_relative_permittivity_jump: Array
    pump_power: Array
    stokes_power: Array
    acoustic_energy_per_length: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain_map: SBSSharedDomainMap,
        /,
        *,
        pump_electric: ArrayLike,
        pump_electric_displacement: ArrayLike,
        stokes_electric: ArrayLike,
        stokes_electric_displacement: ArrayLike,
        acoustic_displacement: ArrayLike,
        acoustic_strain: ArrayLike,
        relative_permittivity: ArrayLike,
        photoelastic_tensor: ArrayLike,
        relative_permittivity_jump: ArrayLike,
        inverse_relative_permittivity_jump: ArrayLike,
        pump_power: ArrayLike,
        stokes_power: ArrayLike,
        acoustic_energy_per_length: ArrayLike,
    ):
        if not isinstance(domain_map, SBSSharedDomainMap):
            raise TypeError("domain_map must be an SBSSharedDomainMap.")
        dimension = domain_map.spatial_dimension
        pump_e = _native_vector_field(
            pump_electric, domain_map.pump_to_volume.shape[1], dimension, "pump_electric"
        )
        pump_d = _native_vector_field(
            pump_electric_displacement,
            domain_map.pump_to_volume.shape[1],
            dimension,
            "pump_electric_displacement",
        )
        stokes_e = _native_vector_field(
            stokes_electric,
            domain_map.stokes_to_volume.shape[1],
            dimension,
            "stokes_electric",
        )
        stokes_d = _native_vector_field(
            stokes_electric_displacement,
            domain_map.stokes_to_volume.shape[1],
            dimension,
            "stokes_electric_displacement",
        )
        acoustic_u = _native_vector_field(
            acoustic_displacement,
            domain_map.acoustic_to_volume.shape[1],
            dimension,
            "acoustic_displacement",
        )
        strain = np.asarray(acoustic_strain)
        if strain.shape != (
            domain_map.acoustic_to_volume.shape[1],
            dimension,
            dimension,
        ):
            raise ValueError(
                "acoustic_strain must have shape (sample, dimension, dimension)."
            )
        if np.any(~np.isfinite(strain)):
            raise ValueError("acoustic_strain must be finite.")
        if not np.allclose(strain, np.swapaxes(strain, -1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError("acoustic_strain must be symmetric.")
        volume_count = domain_map.volume_weights.size
        boundary_count = domain_map.boundary_weights.size
        epsilon_r = _point_values(
            relative_permittivity,
            volume_count,
            "relative_permittivity",
            real=True,
        )
        if np.any(epsilon_r <= 0.0):
            raise ValueError("relative_permittivity must be positive.")
        photoelastic = np.asarray(photoelastic_tensor)
        tensor_shape = (dimension, dimension, dimension, dimension)
        if photoelastic.shape == tensor_shape:
            photoelastic = np.broadcast_to(photoelastic, (volume_count,) + tensor_shape)
        if photoelastic.shape != (volume_count,) + tensor_shape:
            raise ValueError(
                "photoelastic_tensor must have shape (dimension,)^4 or "
                "(volume point, dimension, dimension, dimension, dimension)."
            )
        if np.any(~np.isfinite(photoelastic)):
            raise ValueError("photoelastic_tensor must be finite.")
        epsilon_jump = _point_values(
            relative_permittivity_jump,
            boundary_count,
            "relative_permittivity_jump",
        )
        inverse_jump = _point_values(
            inverse_relative_permittivity_jump,
            boundary_count,
            "inverse_relative_permittivity_jump",
        )
        powers = tuple(
            np.asarray(value)
            for value in (pump_power, stokes_power, acoustic_energy_per_length)
        )
        if any(
            value.ndim != 0
            or np.iscomplexobj(value)
            or not np.isfinite(value)
            or float(value) <= 0.0
            for value in powers
        ):
            raise ValueError(
                "Pump/Stokes powers and acoustic energy must be positive real scalars."
            )
        self.domain_map = domain_map
        self.pump_electric = jnp.asarray(pump_e)
        self.pump_electric_displacement = jnp.asarray(pump_d)
        self.stokes_electric = jnp.asarray(stokes_e)
        self.stokes_electric_displacement = jnp.asarray(stokes_d)
        self.acoustic_displacement = jnp.asarray(acoustic_u)
        self.acoustic_strain = jnp.asarray(strain)
        self.relative_permittivity = jnp.asarray(epsilon_r)
        self.photoelastic_tensor = jnp.asarray(photoelastic)
        self.relative_permittivity_jump = jnp.asarray(epsilon_jump)
        self.inverse_relative_permittivity_jump = jnp.asarray(inverse_jump)
        self.pump_power = jnp.asarray(powers[0])
        self.stokes_power = jnp.asarray(powers[1])
        self.acoustic_energy_per_length = jnp.asarray(powers[2])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sbs-overlap-plan",
                "domain_map": domain_map.map_id,
                "pump_electric": array_tree_fingerprint(pump_e),
                "pump_displacement": array_tree_fingerprint(pump_d),
                "stokes_electric": array_tree_fingerprint(stokes_e),
                "stokes_displacement": array_tree_fingerprint(stokes_d),
                "acoustic_displacement": array_tree_fingerprint(acoustic_u),
                "acoustic_strain": array_tree_fingerprint(strain),
                "relative_permittivity": array_tree_fingerprint(epsilon_r),
                "photoelastic": array_tree_fingerprint(photoelastic),
                "relative_permittivity_jump": array_tree_fingerprint(epsilon_jump),
                "inverse_relative_permittivity_jump": array_tree_fingerprint(
                    inverse_jump
                ),
                "normalizations": [float(value) for value in powers],
            }
        )

    def prepare(self, /) -> "PreparedSBSOverlap":
        return prepare_sbs_overlap(self)


class PreparedSBSOverlap(StrictModule):
    """All three fields represented on the same volume and interface rules."""

    plan: SBSOverlapPlan
    pump_electric_volume: Array
    stokes_electric_volume: Array
    acoustic_strain_volume: Array
    pump_electric_boundary: Array
    pump_displacement_boundary: Array
    stokes_electric_boundary: Array
    stokes_displacement_boundary: Array
    acoustic_displacement_boundary: Array
    normalization: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


class SBSResult(StrictModule):
    """Complex PE/MB interference and SI-normalized SBS gain evidence.

    Under field rescalings ``(a_p, a_s, a_b)``, every complex overlap transforms
    as ``a_p conj(a_s) a_b``. Gain uses ``abs(Q_PE + Q_MB)**2`` only after this
    coherent sum and is invariant when powers/energy use the same rescaling.
    """

    Q_PE: Array
    Q_MB: Array
    Q_total: Array
    normalized_Q_PE: Array
    normalized_Q_MB: Array
    normalized_Q_total: Array
    acoustic_quality_factor: Array
    acoustic_linewidth: Array
    acoustic_power_attenuation: Array
    frequency_detuning: Array
    phase_mismatch: Array
    frequency_lineshape: Array
    phase_matching_factor: Array
    resonant_gain: Array
    gain: Array
    status: Array
    overlap_units: str = eqx.field(static=True)
    normalized_overlap_units: str = eqx.field(static=True)
    linewidth_units: str = eqx.field(static=True)
    attenuation_units: str = eqx.field(static=True)
    phase_mismatch_units: str = eqx.field(static=True)
    gain_units: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(SBSStatus.SUCCESS)


def prepare_sbs_overlap(plan: SBSOverlapPlan, /) -> PreparedSBSOverlap:
    if not isinstance(plan, SBSOverlapPlan):
        raise TypeError("plan must be an SBSOverlapPlan.")
    mapping = plan.domain_map
    normalization = jnp.sqrt(
        plan.pump_power * plan.stokes_power * plan.acoustic_energy_per_length
    )
    finite = (
        jnp.isfinite(normalization)
        & (normalization > 0.0)
        & jnp.isfinite(mapping.normal_unit_error)
    )
    status = jnp.where(
        ~jnp.isfinite(mapping.normal_unit_error),
        int(SBSStatus.INVALID_INTERFACE_EVIDENCE),
        jnp.where(
            ~finite,
            int(SBSStatus.INVALID_NORMALIZATION),
            int(SBSStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return PreparedSBSOverlap(
        plan=plan,
        pump_electric_volume=mapping.map_pump_volume(plan.pump_electric),
        stokes_electric_volume=mapping.map_stokes_volume(plan.stokes_electric),
        acoustic_strain_volume=mapping.map_acoustic_volume(plan.acoustic_strain),
        pump_electric_boundary=mapping.map_pump_boundary(plan.pump_electric),
        pump_displacement_boundary=mapping.map_pump_boundary(
            plan.pump_electric_displacement
        ),
        stokes_electric_boundary=mapping.map_stokes_boundary(plan.stokes_electric),
        stokes_displacement_boundary=mapping.map_stokes_boundary(
            plan.stokes_electric_displacement
        ),
        acoustic_displacement_boundary=mapping.map_acoustic_boundary(
            plan.acoustic_displacement
        ),
        normalization=normalization,
        status=status,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-sbs-overlap",
                "plan": plan.plan_id,
                "domain_map": mapping.map_id,
            }
        ),
    )


def solve_sbs(
    prepared: PreparedSBSOverlap,
    coefficients: SBSInteractionCoefficients,
    /,
) -> SBSResult:
    if not isinstance(prepared, PreparedSBSOverlap):
        raise TypeError("prepared must be a PreparedSBSOverlap.")
    if not isinstance(coefficients, SBSInteractionCoefficients):
        raise TypeError("coefficients must be SBSInteractionCoefficients.")
    plan = prepared.plan
    mapping = plan.domain_map
    photoelastic_image = ein.contract(
        "qijkl,qkl,qj->qi",
        plan.photoelastic_tensor,
        prepared.acoustic_strain_volume,
        prepared.pump_electric_volume,
        backend="jax",
    )
    photoelastic_density = ein.contract(
        "qi,qi->q",
        jnp.conj(prepared.stokes_electric_volume),
        photoelastic_image,
        backend="jax",
    )
    Q_PE = (
        -0.5
        * _VACUUM_PERMITTIVITY
        * jnp.sum(
            mapping.volume_weights * plan.relative_permittivity**2 * photoelastic_density
        )
    )
    normals = mapping.boundary_normals
    pump_normal_e = ein.contract(
        "bi,bi->b", normals, prepared.pump_electric_boundary, backend="jax"
    )
    stokes_normal_e = ein.contract(
        "bi,bi->b", normals, prepared.stokes_electric_boundary, backend="jax"
    )
    pump_tangential_e = prepared.pump_electric_boundary - (
        pump_normal_e[:, None] * normals
    )
    stokes_tangential_e = prepared.stokes_electric_boundary - (
        stokes_normal_e[:, None] * normals
    )
    pump_normal_d = ein.contract(
        "bi,bi->b", normals, prepared.pump_displacement_boundary, backend="jax"
    )
    stokes_normal_d = ein.contract(
        "bi,bi->b", normals, prepared.stokes_displacement_boundary, backend="jax"
    )
    acoustic_normal = ein.contract(
        "bi,bi->b", normals, prepared.acoustic_displacement_boundary, backend="jax"
    )
    tangential_pairing = ein.contract(
        "bi,bi->b",
        jnp.conj(stokes_tangential_e),
        pump_tangential_e,
        backend="jax",
    )
    normal_displacement_pairing = jnp.conj(stokes_normal_d) * pump_normal_d
    boundary_density = acoustic_normal * (
        _VACUUM_PERMITTIVITY * plan.relative_permittivity_jump * tangential_pairing
        - plan.inverse_relative_permittivity_jump
        * normal_displacement_pairing
        / _VACUUM_PERMITTIVITY
    )
    Q_MB = 0.5 * jnp.sum(mapping.boundary_weights * boundary_density)
    Q_total = Q_PE + Q_MB
    normalized_Q_PE = Q_PE / prepared.normalization
    normalized_Q_MB = Q_MB / prepared.normalization
    normalized_Q_total = Q_total / prepared.normalization
    linewidth = (
        coefficients.acoustic_angular_frequency / coefficients.acoustic_quality_factor
    )
    attenuation = linewidth / jnp.abs(coefficients.acoustic_group_velocity)
    detuning = (
        coefficients.pump_angular_frequency
        - coefficients.stokes_angular_frequency
        - coefficients.acoustic_angular_frequency
    )
    mismatch = (
        coefficients.pump_propagation_constant
        - coefficients.stokes_propagation_constant
        - coefficients.acoustic_wavenumber
    )
    frequency_lineshape = 1.0 / (1.0 + (2.0 * detuning / linewidth) ** 2)
    phase_argument = 0.5 * jnp.real(mismatch) * coefficients.interaction_length
    phase_matching_factor = jnp.sinc(phase_argument / jnp.pi) ** 2
    resonant_gain = (
        2.0
        * coefficients.stokes_angular_frequency
        * coefficients.acoustic_angular_frequency
        * jnp.abs(Q_total) ** 2
        / (
            plan.pump_power
            * plan.stokes_power
            * plan.acoustic_energy_per_length
            * linewidth
        )
    )
    gain = resonant_gain * frequency_lineshape * phase_matching_factor
    overlap_finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                [
                    Q_PE,
                    Q_MB,
                    Q_total,
                    normalized_Q_PE,
                    normalized_Q_MB,
                    normalized_Q_total,
                ]
            )
        )
    )
    interaction_finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                [
                    linewidth,
                    attenuation,
                    detuning,
                    mismatch,
                    frequency_lineshape,
                    phase_matching_factor,
                    resonant_gain,
                    gain,
                ]
            )
        )
    )
    status = jnp.where(
        prepared.status != int(SBSStatus.SUCCESS),
        prepared.status,
        jnp.where(
            ~overlap_finite,
            int(SBSStatus.NONFINITE_OVERLAP),
            jnp.where(
                ~interaction_finite,
                int(SBSStatus.NONFINITE_INTERACTION),
                int(SBSStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return SBSResult(
        Q_PE=Q_PE,
        Q_MB=Q_MB,
        Q_total=Q_total,
        normalized_Q_PE=normalized_Q_PE,
        normalized_Q_MB=normalized_Q_MB,
        normalized_Q_total=normalized_Q_total,
        acoustic_quality_factor=coefficients.acoustic_quality_factor,
        acoustic_linewidth=linewidth,
        acoustic_power_attenuation=attenuation,
        frequency_detuning=detuning,
        phase_mismatch=mismatch,
        frequency_lineshape=frequency_lineshape,
        phase_matching_factor=phase_matching_factor,
        resonant_gain=resonant_gain,
        gain=gain,
        status=status,
        overlap_units="J m^-1",
        normalized_overlap_units="s J^-1/2 m^-1/2",
        linewidth_units="rad s^-1",
        attenuation_units="m^-1",
        phase_mismatch_units="rad m^-1",
        gain_units="W^-1 m^-1",
        result_id=canonical_fingerprint(
            {
                "kind": "sbs-result",
                "prepared": prepared.prepared_id,
                "coefficients": coefficients.coefficient_id,
            }
        ),
    )


def _validate_interpolation_map(value, target_count, name):
    if value.ndim != 2 or value.shape[0] != target_count or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape (shared point, native sample).")
    if np.iscomplexobj(value) or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be finite and real.")
    row_sums = np.sum(value, axis=1)
    if value.shape[0] and not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} rows must preserve constants exactly.")


def _map_samples(matrix, values, name):
    value = jnp.asarray(values)
    if value.ndim < 1 or value.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} samples do not match the native map axis.")
    return ein.contract("qn,n...->q...", matrix, value, backend="jax")


def _native_vector_field(value, sample_count, dimension, name):
    field = np.asarray(value)
    if field.shape != (sample_count, dimension):
        raise ValueError(f"{name} must have shape (native sample, dimension).")
    if np.any(~np.isfinite(field)):
        raise ValueError(f"{name} must be finite.")
    return field


def _point_values(value, count, name, *, real=False):
    points = np.asarray(value)
    if points.ndim == 0:
        points = np.broadcast_to(points, (count,))
    if points.shape != (count,):
        raise ValueError(f"{name} must be scalar or have one value per shared point.")
    if (real and np.iscomplexobj(points)) or np.any(~np.isfinite(points)):
        qualifier = "finite and real" if real else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return points


__all__ = [
    "PreparedSBSOverlap",
    "SBSInteractionCoefficients",
    "SBSOverlapPlan",
    "SBSResult",
    "SBSSharedDomainMap",
    "SBSStatus",
    "prepare_sbs_overlap",
    "solve_sbs",
]
