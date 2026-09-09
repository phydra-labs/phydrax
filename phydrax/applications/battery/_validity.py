#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ex-ante equation envelopes, deliberately independent of release evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True)
class BatteryValidityEnvelope:
    model_id: str
    equation_schema: str
    property_schema: str
    geometry_bounds: tuple[tuple[str, float, float], ...]
    operating_bounds: tuple[tuple[str, float, float], ...]
    parameter_ranges: tuple[tuple[str, str], ...]
    discretization_policy: str
    replay_policy: str
    resource_policy: str
    property_support_ids: tuple[str, ...]
    production_parameter_bounds: tuple[tuple[str, float, float], ...] = ()
    production_initial_bounds: tuple[tuple[str, float, float], ...] = ()
    production_protocol_bounds: tuple[tuple[str, float, float], ...] = ()
    production_layout_bounds: tuple[tuple[str, float, float], ...] = ()
    resource_limits: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "equation_schema",
            "property_schema",
            "discretization_policy",
            "replay_policy",
            "resource_policy",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a canonical schema identifier.")
        for name in ("geometry_bounds", "operating_bounds"):
            bounds = getattr(self, name)
            if not isinstance(bounds, tuple) or not bounds:
                raise ValueError(f"{name} requires immutable declared bounds.")
            if len({row[0] for row in bounds}) != len(bounds):
                raise ValueError("Envelope bounds must have unique names.")
            for key, lower, upper in bounds:
                if (
                    not isinstance(key, str)
                    or not key
                    or type(lower) not in (int, float)
                    or type(upper) not in (int, float)
                    or not math.isfinite(lower)
                    or not math.isfinite(upper)
                    or lower > upper
                ):
                    raise ValueError("Envelope bounds must be finite ordered intervals.")
        if (
            not isinstance(self.parameter_ranges, tuple)
            or not self.parameter_ranges
            or len(dict(self.parameter_ranges)) != len(self.parameter_ranges)
        ):
            raise ValueError("Parameter schema ranges must be immutable and unique.")
        if any(
            not isinstance(value, str) or not value
            for pair in self.parameter_ranges
            for value in pair
        ):
            raise ValueError("Parameter ranges must declare named schema constraints.")
        if (
            not isinstance(self.property_support_ids, tuple)
            or not self.property_support_ids
            or len(set(self.property_support_ids)) != len(self.property_support_ids)
        ):
            raise ValueError("Property support schema identifiers must be unique.")
        if any(
            not isinstance(value, str) or not value for value in self.property_support_ids
        ):
            raise ValueError("Property support identifiers must be nonempty.")
        for name in (
            "production_parameter_bounds",
            "production_initial_bounds",
            "production_protocol_bounds",
            "production_layout_bounds",
        ):
            bounds = getattr(self, name)
            if not isinstance(bounds, tuple) or len({row[0] for row in bounds}) != len(
                bounds
            ):
                raise ValueError(
                    "Production bounds must be immutable and uniquely named."
                )
            for key, lower, upper in bounds:
                if (
                    not isinstance(key, str)
                    or not key
                    or type(lower) not in (int, float)
                    or type(upper) not in (int, float)
                    or not math.isfinite(lower)
                    or not math.isfinite(upper)
                    or lower > upper
                ):
                    raise ValueError(
                        "Production bounds must be numeric finite intervals."
                    )
        if not isinstance(self.resource_limits, tuple) or len(
            dict(self.resource_limits)
        ) != len(self.resource_limits):
            raise ValueError("Absolute resource limits must be immutable and unique.")
        if any(
            not isinstance(name, str)
            or not name
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            for name, value in self.resource_limits
        ):
            raise ValueError(
                "Absolute resource limits must be finite nonnegative numbers."
            )

    def require_production_bounds(self) -> None:
        if any(
            not getattr(self, name)
            for name in (
                "production_parameter_bounds",
                "production_initial_bounds",
                "production_protocol_bounds",
                "production_layout_bounds",
                "resource_limits",
            )
        ):
            raise ValueError(
                "Finite precommitted production parameter/initial/protocol/layout/resource bounds are unavailable."
            )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-validity-envelope",
            "model_id": self.model_id,
            "equation_schema": self.equation_schema,
            "property_schema": self.property_schema,
            "geometry_bounds": [list(row) for row in self.geometry_bounds],
            "operating_bounds": [list(row) for row in self.operating_bounds],
            "parameter_ranges": dict(self.parameter_ranges),
            "discretization_policy": self.discretization_policy,
            "replay_policy": self.replay_policy,
            "resource_policy": self.resource_policy,
            "property_support_ids": list(self.property_support_ids),
            **{
                name: [list(row) for row in getattr(self, name)]
                for name in (
                    "production_parameter_bounds",
                    "production_initial_bounds",
                    "production_protocol_bounds",
                    "production_layout_bounds",
                    "resource_limits",
                )
            },
        }

    @property
    def envelope_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    def require_operating_values(self, **values: float) -> None:
        bounds = {name: (lower, upper) for name, lower, upper in self.operating_bounds}
        if set(values) != set(bounds):
            raise ValueError(
                "Operating values must cover the exact envelope coordinates."
            )
        for name, value in values.items():
            lower, upper = bounds[name]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Operating coordinate {name} is outside its envelope.")


# Schema-level normalized margins do not claim any product parameterization.
# Current is normalized by the property-supported current bound, not cell capacity.
_COMMON_OPERATING = (
    ("normalized_current", -1.0, 1.0),
    ("normalized_property_margin", 0.0, 1.0),
)
_COMMON_REPLAY = "native-fixed-topology-tangent-adjoint-and-independent-clean-execution"
_COMMON_RESOURCE = "precommitted-absolute-workloads-cpu-x64-no-global-dense"
_ELECTROCHEMICAL_RANGES = (
    ("porosity", "0 < ε < 1"),
    ("transport", "finite-strictly-positive-property-support"),
    ("concentration", "0 < cₛ,surf < cₘₐₓ; cₑ > 0"),
    ("temperature", "fixed-isothermal-property-support"),
)
_ECM_RANGES = (
    ("capacity", "finite-positive"),
    ("series_resistance", "finite-strictly-positive"),
    ("rc_branches", "one-positive-R-and-C"),
    ("thermal_capacity", "finite-positive"),
    ("ambient_conductance", "finite-nonnegative"),
)

MARQUIS_2019_SPME_ENVELOPE = BatteryValidityEnvelope(
    "battery:spme:marquis-2019:isothermal-prescribed-current",
    "marquis-2019-eq49-table6",
    "marquis-2019-distinguished-electrolyte-χ=1",
    (("dimension", 1, 1),),
    _COMMON_OPERATING,
    _ELECTROCHEMICAL_RANGES,
    "independent-x-r-time-refinement-planar-asymptotic-electrolyte",
    _COMMON_REPLAY,
    _COMMON_RESOURCE,
    ("marquis-2019-eq49-table6-property-support",),
)
NEWMAN_DFN_ENVELOPE = BatteryValidityEnvelope(
    "battery:dfn:newman:isothermal-1d-prescribed-current",
    "newman-1d-symmetric-bv-gauge-negative-collector",
    "symmetric-bv-positive-transport-bruggeman-regions",
    (("dimension", 1, 1),),
    _COMMON_OPERATING,
    _ELECTROCHEMICAL_RANGES,
    "conservative-x-spherical-r-scaled-array-dae-fgmres-banded",
    _COMMON_REPLAY,
    _COMMON_RESOURCE,
    ("newman-interior-concentrations-positive-exchange-current",),
)
CIRCUIT_ECM_ENVELOPE = BatteryValidityEnvelope(
    "battery:ecm:circuit-connected-electrothermal",
    "passive-two-terminal-q-w-T-I-dae",
    "thermal-ecm-pure-model-terms",
    (("cell_count", 1, 1), ("rc_branch_count", 1, 1)),
    _COMMON_OPERATING,
    _ECM_RANGES,
    "one-cell-nodal-circuit-one-node-linear-ambient",
    _COMMON_REPLAY,
    _COMMON_RESOURCE,
    ("thermal-ecm-interior-charge-temperature-positive-resistance",),
)
SERIES_PACK_ENVELOPE = BatteryValidityEnvelope(
    "battery:pack:homogeneous-series-ecm-electrothermal",
    "ordered-homogeneous-series-ecm-thermal-graph-dae",
    "thermal-ecm-pure-model-terms",
    (("cell_count", 1, 96), ("rc_branch_count", 1, 1)),
    _COMMON_OPERATING,
    (
        *_ECM_RANGES,
        ("link_resistance", "uniform-finite-strictly-positive"),
        ("thermal_edges", "uniform-finite-nonnegative"),
        ("link_heat_allocation", "nonnegative-column-sum-one"),
    ),
    "ordered-nearest-neighbor-path-scaled-array-sparse-dae",
    _COMMON_REPLAY,
    _COMMON_RESOURCE,
    ("thermal-ecm-interior-charge-temperature-positive-resistance",),
)
THERMAL_ECM_ENVELOPE = BatteryValidityEnvelope(
    "battery:ecm:thermal-prescribed-current",
    "thermal-ecm-q-w-T-ode",
    "thermal-ecm-pure-model-terms",
    (("cell_count", 1, 1),),
    _COMMON_OPERATING,
    _ECM_RANGES,
    "standalone-prescribed-current-ode",
    _COMMON_REPLAY,
    _COMMON_RESOURCE,
    ("thermal-ecm-interior-charge-temperature-positive-resistance",),
)

BATTERY_NUMERICAL_ENVELOPES = (
    THERMAL_ECM_ENVELOPE,
    MARQUIS_2019_SPME_ENVELOPE,
    NEWMAN_DFN_ENVELOPE,
    CIRCUIT_ECM_ENVELOPE,
    SERIES_PACK_ENVELOPE,
)
