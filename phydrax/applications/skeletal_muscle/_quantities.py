#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from .._quantity_contract import (
    canonical_quantity_text,
    resolve_application_quantity,
)


_SUPPORTED_UNIT_CONVERSIONS = MappingProxyType(
    {
        ("angle", "rad", "rad"): Fraction(1),
        ("amount_concentration", "uM", "mol/m3"): Fraction(1, 1_000),
        ("dimensionless", "1", "1"): Fraction(1),
        ("electric_current", "A", "A"): Fraction(1),
        ("electric_potential", "V", "V"): Fraction(1),
        ("energy", "J", "J"): Fraction(1),
        ("force", "N", "N"): Fraction(1),
        ("frequency", "Hz", "Hz"): Fraction(1),
        ("frequency", "pps", "Hz"): Fraction(1),
        ("length", "m", "m"): Fraction(1),
        ("power", "W", "W"): Fraction(1),
        ("pressure", "Pa", "Pa"): Fraction(1),
        ("rate", "1/s", "1/s"): Fraction(1),
        ("surface_current_density", "uA/cm2", "A/m2"): Fraction(1, 100),
        ("time", "ms", "s"): Fraction(1, 1_000),
        ("time", "s", "s"): Fraction(1),
        ("velocity", "m/s", "m/s"): Fraction(1),
    }
)


@dataclass(frozen=True, slots=True, init=False)
class SkeletalMuscleQuantitySpec:
    """Immutable physical meaning and exact SI scale for a skeletal quantity."""

    name: str
    physical_dimension: str
    kernel_unit: str
    si_unit: str
    si_factor: Fraction
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str = field(init=False)

    def __init__(
        self,
        name: str,
        physical_dimension: str,
        kernel_unit: str,
        si_unit: str,
        si_factor: Any,
        axes: tuple[str, ...] = (),
        sign_convention: str = "",
        support_association: str = "",
        reference_configuration: str = "",
    ):
        resolved = resolve_application_quantity(
            domain="skeletal-muscle",
            supported_conversions=_SUPPORTED_UNIT_CONVERSIONS,
            name=name,
            physical_dimension=physical_dimension,
            kernel_unit=kernel_unit,
            si_unit=si_unit,
            si_factor=si_factor,
            axes=axes,
            sign_convention=sign_convention,
            support_association=support_association,
            reference_configuration=reference_configuration,
        )
        object.__setattr__(self, "name", resolved.name)
        object.__setattr__(self, "physical_dimension", resolved.physical_dimension)
        object.__setattr__(self, "kernel_unit", resolved.kernel_unit)
        object.__setattr__(self, "si_unit", resolved.si_unit)
        object.__setattr__(self, "si_factor", resolved.si_factor)
        object.__setattr__(self, "axes", resolved.axes)
        object.__setattr__(self, "sign_convention", resolved.sign_convention)
        object.__setattr__(self, "support_association", resolved.support_association)
        object.__setattr__(
            self, "reference_configuration", resolved.reference_configuration
        )
        object.__setattr__(self, "quantity_id", resolved.quantity_id)

    @property
    def spec_id(self) -> str:
        return self.quantity_id

    def to_si(self, value: Any, /) -> Any:
        return value * self.si_factor.numerator / self.si_factor.denominator

    def from_si(self, value: Any, /) -> Any:
        return value * self.si_factor.denominator / self.si_factor.numerator


def _quantity(
    name: str,
    dimension: str,
    unit: str,
    *,
    si_unit: str | None = None,
    axes: tuple[str, ...] = (),
    sign: str,
    support: str,
    reference: str,
) -> SkeletalMuscleQuantitySpec:
    si = unit if si_unit is None else si_unit
    return SkeletalMuscleQuantitySpec(
        name,
        dimension,
        unit,
        si,
        _SUPPORTED_UNIT_CONVERSIONS[(dimension, unit, si)],
        axes=axes,
        sign_convention=sign,
        support_association=support,
        reference_configuration=reference,
    )


_MOTOR_UNIT_REFERENCE = "Potvin--Fuglevand 2017 deterministic isometric model"
_MOTOR_UNIT_AXIS = ("motor_unit",)
_NONNEGATIVE = "nonnegative magnitude"
_MUSCLE_AXIS = ("muscle",)
_CELL_AXIS = ("cell",)
_COMPARTMENT_AXIS = ("compartment",)
_CHANNEL_SAMPLE_AXES = ("channel", "sample")

_SPECS = (
    _quantity(
        "time",
        "time",
        "s",
        sign="increases forward",
        support="global",
        reference="protocol time origin",
    ),
    _quantity(
        "common_excitation",
        "dimensionless",
        "1",
        sign="nonnegative common excitatory drive in the owning model scale",
        support="motor-unit population",
        reference=_MOTOR_UNIT_REFERENCE,
    ),
    _quantity(
        "motor_unit_firing_rate",
        "frequency",
        "Hz",
        axes=_MOTOR_UNIT_AXIS,
        sign=_NONNEGATIVE,
        support="motor-unit population",
        reference=_MOTOR_UNIT_REFERENCE,
    ),
    _quantity(
        "recruitment_duration",
        "time",
        "s",
        axes=_MOTOR_UNIT_AXIS,
        sign=_NONNEGATIVE,
        support="motor-unit population state",
        reference="time since first recruitment without recovery or reset",
    ),
    _quantity(
        "contraction_time",
        "time",
        "s",
        axes=_MOTOR_UNIT_AXIS,
        sign="positive twitch contraction time",
        support="motor-unit population",
        reference=_MOTOR_UNIT_REFERENCE,
    ),
    _quantity(
        "relative_twitch_force",
        "dimensionless",
        "1",
        axes=_MOTOR_UNIT_AXIS,
        sign=_NONNEGATIVE,
        support="motor-unit population",
        reference="relative to the rested twitch force of motor unit one",
    ),
    _quantity(
        "relative_muscle_force",
        "dimensionless",
        "1",
        sign=_NONNEGATIVE,
        support="motor-unit population aggregate",
        reference="sum in rested motor-unit-one twitch-force units",
    ),
    _quantity(
        "force_capacity_fraction",
        "dimensionless",
        "1",
        axes=_MOTOR_UNIT_AXIS,
        sign="zero exhausted and one rested",
        support="motor-unit population state",
        reference=_MOTOR_UNIT_REFERENCE,
    ),
    _quantity(
        "independent_excitation", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign="nonnegative excitation on the owning [0, 1] actuator scale",
        support="lumped musculotendon actuator",
        reference="De Groote--Fregly or provider-native actuator",
    ),
    _quantity(
        "muscle_activation", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign="zero inactive and one fully active",
        support="lumped musculotendon actuator",
        reference="owning activation dynamics",
    ),
    _quantity(
        "musculotendon_length", "length", "m", axes=_MUSCLE_AXIS,
        sign="nonnegative route or transmission length",
        support="musculotendon line of action",
        reference="current multibody configuration",
    ),
    _quantity(
        "musculotendon_velocity", "velocity", "m/s", axes=_MUSCLE_AXIS,
        sign="positive lengthening",
        support="musculotendon line of action",
        reference="current multibody configuration",
    ),
    _quantity(
        "normalized_tendon_force", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign="positive tensile", support="lumped tendon state",
        reference="maximum isometric muscle force",
    ),
    _quantity(
        "normalized_tendon_length", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="lumped tendon",
        reference="tendon slack length",
    ),
    _quantity(
        "tendon_length", "length", "m", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="lumped tendon",
        reference="current configuration",
    ),
    _quantity(
        "tendon_velocity", "velocity", "m/s", axes=_MUSCLE_AXIS,
        sign="positive lengthening", support="lumped tendon",
        reference="current configuration",
    ),
    _quantity(
        "normalized_muscle_fiber_length", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="contractile fiber",
        reference="optimal fiber length",
    ),
    _quantity(
        "muscle_fiber_length", "length", "m", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="contractile fiber",
        reference="current configuration",
    ),
    _quantity(
        "normalized_muscle_fiber_velocity", "dimensionless", "1", axes=_MUSCLE_AXIS,
        sign="positive lengthening", support="contractile fiber",
        reference="owning maximum contraction velocity",
    ),
    _quantity(
        "muscle_fiber_velocity", "velocity", "m/s", axes=_MUSCLE_AXIS,
        sign="positive lengthening", support="contractile fiber",
        reference="current configuration",
    ),
    _quantity(
        "pennation_angle", "angle", "rad", axes=_MUSCLE_AXIS,
        sign="nonnegative fiber-to-tendon angle below pi/2",
        support="lumped musculotendon geometry",
        reference="current configuration",
    ),
    _quantity(
        "tendon_force", "force", "N", axes=_MUSCLE_AXIS,
        sign="positive tensile", support="lumped tendon",
        reference="owning physical force model",
    ),
    _quantity(
        "muscle_fiber_force", "force", "N", axes=_MUSCLE_AXIS,
        sign="positive tensile along the fiber", support="contractile fiber",
        reference="owning physical force model",
    ),
    _quantity(
        "raw_provider_force", "force", "N", axes=_MUSCLE_AXIS,
        sign="provider-defined signed actuator force",
        support="external provider actuator",
        reference="immutable provider descriptor",
    ),
    _quantity(
        "tendon_elastic_energy", "energy", "J", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="lumped tendon",
        reference="zero at normalized tendon length one",
    ),
    _quantity(
        "passive_fiber_elastic_energy", "energy", "J", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE, support="passive contractile fiber",
        reference="owning musculotendon potential",
    ),
    _quantity(
        "mechanical_power", "power", "W", axes=_MUSCLE_AXIS,
        sign="positive energy increase or declared work output",
        support="musculotendon or continuum owner",
        reference="declared line-of-action convention",
    ),
    _quantity(
        "skeletal_continuum_stiffness", "pressure", "Pa",
        sign="positive material energy and stress scale",
        support="skeletal continuum material",
        reference="reference configuration",
    ),
    _quantity(
        "skeletal_peak_active_nominal_stress", "pressure", "Pa",
        sign="positive tension along the reference fiber",
        support="skeletal continuum material",
        reference="Engelhardt GASAM 2025",
    ),
    _quantity(
        "skeletal_prescribed_activation", "dimensionless", "1",
        sign="closed support [0, 1]",
        support="skeletal continuum material point",
        reference="owning prescribed-activation plan",
    ),
    _quantity(
        "skeletal_muscle_stimulus_current_density", "surface_current_density",
        "uA/cm2", si_unit="A/m2", axes=_CELL_AXIS, sign="positive inward",
        support="skeletal sarcolemma or endplate node",
        reference="Shorten 2007 I_HH convention",
    ),
    _quantity(
        "skeletal_muscle_membrane_current_density", "surface_current_density",
        "uA/cm2", si_unit="A/m2", axes=_CELL_AXIS, sign="positive outward",
        support="sarcolemma or transverse tubule",
        reference="Shorten 2007 ionic current decomposition",
    ),
    _quantity(
        "skeletal_muscle_cytosolic_calcium_concentration", "amount_concentration",
        "uM", si_unit="mol/m3", axes=_CELL_AXIS, sign=_NONNEGATIVE,
        support="skeletal cellular cytosol",
        reference="Shorten 2007 Ca_1 and Ca_2",
    ),
    _quantity(
        "skeletal_muscle_sr_calcium_concentration", "amount_concentration",
        "uM", si_unit="mol/m3", axes=_CELL_AXIS, sign=_NONNEGATIVE,
        support="skeletal sarcoplasmic reticulum",
        reference="Shorten 2007 Ca_SR1 and Ca_SR2",
    ),
    _quantity(
        "skeletal_muscle_force_bearing_crossbridge_concentration",
        "amount_concentration", "uM", si_unit="mol/m3", axes=_CELL_AXIS,
        sign=_NONNEGATIVE, support="skeletal cellular crossbridge pool",
        reference="Shorten 2007 A_2; not physical force or stress",
    ),
    _quantity(
        "motor_unit_event_time", "time", "ms", si_unit="s",
        axes=("motor_unit", "event_slot"),
        sign="forward absolute discharge time",
        support="fixed-capacity motor-unit event block",
        reference="Fuglevand--Winter--Patla 1993",
    ),
    _quantity(
        "brain_effort_rate", "rate", "1/s", sign=_NONNEGATIVE,
        support="macroscopic fatigue compartments",
        reference="Liu--Brown--Yue 2002",
    ),
    _quantity(
        "motor_unit_compartment_fraction", "dimensionless", "1",
        axes=_COMPARTMENT_AXIS, sign="closed support [0, 1] with sum one",
        support="macroscopic fatigue state",
        reference="Liu--Brown--Yue 2002",
    ),
    _quantity(
        "relative_isometric_force", "dimensionless", "1",
        sign="positive declared force-producing direction",
        support="isometric force observation",
        reference="source-named relative force model",
    ),
    _quantity(
        "physical_force_scale", "force", "N",
        sign="positive N per relative-force unit",
        support="force calibration observation model",
        reference="explicit calibration asset",
    ),
    _quantity(
        "observed_force", "force", "N", axes=("sample",),
        sign="declared measurement-axis direction",
        support="force transducer samples",
        reference="calibration asset and protocol",
    ),
    _quantity(
        "force_standard_uncertainty", "force", "N", axes=("sample",),
        sign="strictly positive standard uncertainty",
        support="force transducer samples",
        reference="measurement uncertainty model",
    ),
    _quantity(
        "surface_electric_potential", "electric_potential", "V",
        axes=_CHANNEL_SAMPLE_AXES,
        sign="signed electrode montage potential",
        support="surface EMG channels",
        reference="declared electrode and reference montage",
    ),
    _quantity(
        "muscle_metabolic_power", "power", "W", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE,
        support="phenomenological muscle energetic observation",
        reference="Uchida--Umberger 2010 pinned policy",
    ),
    _quantity(
        "muscle_metabolic_energy", "energy", "J", axes=_MUSCLE_AXIS,
        sign=_NONNEGATIVE,
        support="integrated muscle energetic observation",
        reference="time integral of declared metabolic power",
    ),
    _quantity(
        "spindle_afferent_rate", "frequency", "pps", si_unit="Hz",
        sign=_NONNEGATIVE, support="Ia or group-II spindle afferent",
        reference="Mileusnic et al. 2006 feline model",
    ),
    _quantity(
        "gamma_drive_frequency", "frequency", "pps", si_unit="Hz",
        sign=_NONNEGATIVE, support="dynamic or static fusimotor input",
        reference="Mileusnic et al. 2006 feline model",
    ),
    _quantity(
        "normalized_fascicle_length", "dimensionless", "1",
        sign="positive fascicle length divided by optimal length",
        support="muscle spindle", reference="Mileusnic et al. 2006",
    ),
)

if len({spec.name for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("Canonical skeletal-muscle quantity names must be unique.")
if len({spec.quantity_id for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("Canonical skeletal-muscle quantity IDs must be unique.")

SKELETAL_MUSCLE_QUANTITIES = MappingProxyType({spec.name: spec for spec in _SPECS})


def skeletal_muscle_quantity(name: str, /) -> SkeletalMuscleQuantitySpec:
    """Return one canonical skeletal-muscle quantity by stable name."""
    name_ = canonical_quantity_text(name, "name")
    if name_ not in SKELETAL_MUSCLE_QUANTITIES:
        raise KeyError(f"Unknown skeletal-muscle quantity {name_!r}.")
    return SKELETAL_MUSCLE_QUANTITIES[name_]


__all__ = [
    "SKELETAL_MUSCLE_QUANTITIES",
    "SkeletalMuscleQuantitySpec",
    "skeletal_muscle_quantity",
]
