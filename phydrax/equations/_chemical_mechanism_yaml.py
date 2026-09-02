#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import yaml

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._chemical_components import ChemicalComponentCatalog
from ._chemical_mechanism import ChemicalMechanismIR, ChemicalReactionSpec
from ._chemical_rates import (
    ArrheniusRatePlan,
    ChebyshevRatePlan,
    LindemannRatePlan,
    PhotolysisRatePlan,
    PLogRatePlan,
    StickingRatePlan,
    SurfaceCoverageRatePlan,
    ThirdBodyRatePlan,
    TroeRatePlan,
)
from ._chemical_species import (
    ChemicalPhaseKind,
    ChemicalPhaseSpec,
    ChemicalSpeciesSchema,
)
from ._chemical_thermodynamics import (
    NASAPolynomialKind,
    NASASpeciesThermodynamicsPlan,
    PolynomialSpeciesThermodynamicsPlan,
)


class ChemicalMechanismImportReport(StrictModule):
    mechanism: ChemicalMechanismIR
    source_path: str = eqx.field(static=True)
    normalized_units: tuple[tuple[str, str], ...] = eqx.field(static=True)
    converted_fields: tuple[str, ...] = eqx.field(static=True)
    import_id: str = eqx.field(static=True)


_LENGTH = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}
_TIME = {"s": 1.0, "ms": 1.0e-3, "min": 60.0}
_AMOUNT = {"mol": 1.0, "kmol": 1000.0}
_MASS = {"kg": 1.0, "g": 1.0e-3}
_ENERGY = {"J": 1.0, "kJ": 1000.0, "cal": 4.184, "kcal": 4184.0}
_PRESSURE = {"Pa": 1.0, "kPa": 1000.0, "bar": 1.0e5, "atm": 101325.0}


def load_chemical_mechanism_yaml(path: str | Path, /) -> ChemicalMechanismImportReport:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Chemical mechanism YAML root must be a mapping.")
    units = _units(payload.get("units", {}))
    phases_payload = _sequence(payload, "phases")
    phase_specs = tuple(_phase(value, units) for value in phases_payload)
    phase_by_name = {value.name: value for value in phase_specs}
    if len(phase_by_name) != len(phase_specs):
        raise ValueError("Chemical phase names must be unique.")
    species_payload = _sequence(payload, "species")
    names = tuple(_string(value, "name") for value in species_payload)
    if len(set(names)) != len(names):
        raise ValueError("Chemical species names must be unique.")
    elements = []
    for species in species_payload:
        composition = _mapping(species, "composition")
        for element in composition:
            if str(element) not in elements:
                elements.append(str(element))
    phase_indices = []
    charges = []
    component_names = []
    component_indices = []
    component_by_name = {}
    component_masses = []
    component_charges = []
    component_compositions = []
    composition_matrix = np.zeros((len(elements), len(names)), dtype=np.int32)
    element_index = {name: index for index, name in enumerate(elements)}
    phase_index = {value.name: index for index, value in enumerate(phase_specs)}
    for species_index, species in enumerate(species_payload):
        phase_name = _string(species, "phase")
        if phase_name not in phase_by_name:
            raise ValueError(
                f"Species {names[species_index]!r} references unknown phase {phase_name!r}."
            )
        phase_indices.append(phase_index[phase_name])
        mass = float(species["molar-mass"]) * units["mass"] / units["amount"]
        charge = species.get("charge")
        if charge is None or int(charge) != charge:
            raise ValueError(f"Species {names[species_index]!r} requires integer charge.")
        charges.append(int(charge))
        for element, count in _mapping(species, "composition").items():
            if int(count) != count or int(count) < 0:
                raise ValueError("Element composition must be nonnegative integers.")
            composition_matrix[element_index[str(element)], species_index] = int(count)

        component_name = str(species.get("component", names[species_index]))
        if not component_name:
            raise ValueError("Species component names must be nonempty.")
        if component_name not in component_by_name:
            component_by_name[component_name] = len(component_names)
            component_names.append(component_name)
            component_masses.append(mass)
            component_charges.append(int(charge))
            component_compositions.append(composition_matrix[:, species_index].copy())
        else:
            component_id = component_by_name[component_name]
            if (
                component_masses[component_id] != mass
                or component_charges[component_id] != int(charge)
                or not np.array_equal(
                    component_compositions[component_id],
                    composition_matrix[:, species_index],
                )
            ):
                raise ValueError(
                    f"Component {component_name!r} has inconsistent species identity."
                )
        component_indices.append(component_by_name[component_name])

    catalog = ChemicalComponentCatalog(
        tuple(component_names),
        jnp.asarray(component_masses),
        tuple(elements),
        jnp.asarray(np.stack(component_compositions, axis=1)),
        charges=jnp.asarray(component_charges),
        provenance=str(source),
    )
    schema = ChemicalSpeciesSchema(
        catalog,
        names,
        jnp.asarray(component_indices, dtype=jnp.int32),
        phase_specs,
        jnp.asarray(phase_indices, dtype=jnp.int32),
    )
    thermodynamics_payload = _mapping(payload, "thermodynamics")
    thermodynamics = _thermodynamics(
        schema,
        thermodynamics_payload,
        species_payload,
        units,
    )
    reactions = tuple(
        _reaction(value, schema, units) for value in _sequence(payload, "reactions")
    )
    mechanism = ChemicalMechanismIR(
        str(payload.get("name", source.stem)),
        schema,
        thermodynamics,
        reactions,
    )
    normalized = (
        ("length", "m"),
        ("time", "s"),
        ("amount", "mol"),
        ("mass", "kg"),
        ("energy", "J"),
        ("pressure", "Pa"),
        ("temperature", "K"),
    )
    converted = tuple(
        name for name, factor in units.items() if name != "temperature" and factor != 1.0
    )
    identifier = canonical_fingerprint(
        {
            "kind": "chemical-mechanism-yaml-import",
            "source_content": canonical_fingerprint(payload),
            "mechanism": mechanism.name,
            "schema": schema.schema_id,
            "units": dict(normalized),
            "converted": list(converted),
        }
    )
    return ChemicalMechanismImportReport(
        mechanism,
        str(source),
        normalized,
        converted,
        identifier,
    )


def _units(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("units must be a mapping.")
    names = {
        "length": str(payload.get("length", "m")),
        "time": str(payload.get("time", "s")),
        "amount": str(payload.get("amount", "mol")),
        "mass": str(payload.get("mass", "kg")),
        "energy": str(payload.get("energy", "J")),
        "pressure": str(payload.get("pressure", "Pa")),
        "temperature": str(payload.get("temperature", "K")),
    }
    if names["temperature"] != "K":
        raise ValueError("Only absolute kelvin temperatures are supported.")
    tables = {
        "length": _LENGTH,
        "time": _TIME,
        "amount": _AMOUNT,
        "mass": _MASS,
        "energy": _ENERGY,
        "pressure": _PRESSURE,
    }
    factors = {key: tables[key][value] for key, value in names.items() if key in tables}
    factors["temperature"] = 1.0
    return factors


def _phase(payload, units):
    if not isinstance(payload, Mapping):
        raise ValueError("Each phase must be a mapping.")
    kind = ChemicalPhaseKind(_string(payload, "kind"))
    density = payload.get("site-density")
    pressure = payload.get("standard-pressure")
    if kind is ChemicalPhaseKind.GAS and pressure is None:
        raise ValueError("Gas phases require standard-pressure.")
    return ChemicalPhaseSpec(
        _string(payload, "name"),
        kind,
        int(
            payload.get(
                "measure-dimension", 2 if kind is ChemicalPhaseKind.SURFACE else 3
            )
        ),
        standard_concentration=float(payload.get("standard-concentration", 1.0)),
        standard_pressure=None
        if pressure is None
        else float(pressure) * units["pressure"],
        site_density=None if density is None else float(density),
    )


def _thermodynamics(schema, payload, species_payload, units):
    model = str(payload.get("model", "nasa7"))
    by_species = _mapping(payload, "species")
    if model in ("nasa7", "nasa9"):
        coefficient_count = 7 if model == "nasa7" else 9
        intervals = []
        for species in species_payload:
            name = _string(species, "name")
            if name not in by_species:
                raise ValueError(f"Missing thermodynamics for species {name!r}.")
            records = _sequence(by_species, name)
            parsed = []
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("NASA intervals must be mappings.")
                coefficients = np.asarray(record["coefficients"], dtype=float)
                if coefficients.shape != (coefficient_count,):
                    raise ValueError("NASA coefficient count does not match model.")
                parsed.append(
                    (
                        float(record["minimum-temperature"]),
                        float(record["maximum-temperature"]),
                        coefficients,
                    )
                )
            intervals.append(parsed)
        counts = {len(value) for value in intervals}
        if len(counts) != 1:
            raise ValueError("All species must provide the same NASA interval count.")
        lower = np.asarray([[entry[0] for entry in values] for values in intervals])
        upper = np.asarray([[entry[1] for entry in values] for values in intervals])
        coefficients = np.asarray(
            [[entry[2] for entry in values] for values in intervals]
        )
        return NASASpeciesThermodynamicsPlan(
            schema,
            NASAPolynomialKind(model),
            coefficients,
            lower,
            upper,
        )
    if model == "polynomial-internal-energy":
        coefficients = []
        energy = []
        entropy = []
        for species in species_payload:
            record = _mapping(by_species, _string(species, "name"))
            coefficients.append(np.asarray(record["heat-capacity-volume"], dtype=float))
            energy.append(
                float(record.get("reference-internal-energy", 0.0))
                * units["energy"]
                / units["amount"]
            )
            entropy.append(
                float(record.get("reference-entropy", 0.0))
                * units["energy"]
                / units["amount"]
            )
        return PolynomialSpeciesThermodynamicsPlan(
            schema,
            np.asarray(coefficients) * units["energy"] / units["amount"],
            np.asarray(energy),
            reference_molar_entropy=np.asarray(entropy),
            reference_temperature=float(payload.get("reference-temperature", 298.15)),
            minimum_temperature=float(payload.get("minimum-temperature", 1.0)),
            maximum_temperature=float(payload.get("maximum-temperature", 5000.0)),
        )
    raise ValueError(f"Unsupported thermodynamic model {model!r}.")


def _reaction(payload, schema, units):
    if not isinstance(payload, Mapping):
        raise ValueError("Each reaction must be a mapping.")
    reactants = _mapping(payload, "reactants")
    products = _mapping(payload, "products")
    orders = payload.get("orders", reactants)
    total_order = sum(float(value) for value in _mapping_value(orders).values())
    concentration_factor = units["amount"] / units["length"] ** 3
    rate = _rate(
        _mapping(payload, "rate"),
        schema,
        units,
        concentration_factor ** (1.0 - total_order) / units["time"],
        concentration_factor,
    )
    reverse_payload = payload.get("reverse-rate")
    reverse = (
        None
        if reverse_payload is None
        else _rate(
            _mapping_value(reverse_payload),
            schema,
            units,
            concentration_factor
            ** (1.0 - sum(float(value) for value in products.values()))
            / units["time"],
            concentration_factor,
        )
    )
    reversible = str(payload.get("reversible", "none"))
    if reversible not in ("none", "thermodynamic", "explicit"):
        raise ValueError("reversible must be none, thermodynamic, or explicit.")
    if reversible == "explicit" and reverse is None:
        raise ValueError("Explicit reversible reaction requires reverse-rate.")
    return ChemicalReactionSpec(
        _string(payload, "name"),
        reactants,
        products,
        rate,
        forward_orders=orders,
        reverse_rate=reverse,
        thermodynamic_reversible=reversible == "thermodynamic",
        duplicate_group=payload.get("duplicate-group"),
    )


def _rate(payload, schema, units, pre_factor, concentration_factor):
    kind = str(payload.get("type", "arrhenius"))
    if kind == "arrhenius":
        return ArrheniusRatePlan(
            float(payload["A"]) * pre_factor,
            float(payload.get("b", 0.0)),
            float(payload.get("Ea", 0.0)) * units["energy"] / units["amount"],
        )
    efficiencies = np.ones(schema.species_count)
    for name, value in payload.get("efficiencies", {}).items():
        efficiencies[schema.species_names.index(str(name))] = float(value)
    if kind == "third-body":
        return ThirdBodyRatePlan(
            _rate(
                _mapping(payload, "base"),
                schema,
                units,
                pre_factor / concentration_factor,
                concentration_factor,
            ),
            efficiencies,
        )
    if kind in ("lindemann", "troe"):
        low = _rate(
            _mapping(payload, "low"),
            schema,
            units,
            pre_factor / concentration_factor,
            concentration_factor,
        )
        high = _rate(
            _mapping(payload, "high"),
            schema,
            units,
            pre_factor,
            concentration_factor,
        )
        if not isinstance(low, ArrheniusRatePlan) or not isinstance(
            high, ArrheniusRatePlan
        ):
            raise ValueError("Falloff limits must be Arrhenius rates.")
        if kind == "lindemann":
            return LindemannRatePlan(low, high, efficiencies)
        troe = _mapping(payload, "troe")
        return TroeRatePlan(
            low,
            high,
            efficiencies,
            float(troe["alpha"]),
            float(troe["T1"]),
            float(troe["T2"]),
            float(troe["T3"]),
        )
    if kind == "surface-coverage":
        base = _rate(
            _mapping(payload, "base"),
            schema,
            units,
            pre_factor,
            concentration_factor,
        )
        if not isinstance(base, ArrheniusRatePlan):
            raise ValueError("Surface coverage base must be Arrhenius.")
        species = _string(payload, "species")
        if species not in schema.species_names:
            raise ValueError(f"Unknown surface coverage species {species!r}.")
        return SurfaceCoverageRatePlan(
            base,
            schema.species_names.index(species),
            exponential_coefficient=float(payload.get("a", 0.0)),
            power_exponent=float(payload.get("m", 0.0)),
            activation_energy_coefficient=float(payload.get("E", 0.0))
            * units["energy"]
            / units["amount"],
        )
    if kind == "sticking":
        species = _string(payload, "gas-species")
        if species not in schema.species_names:
            raise ValueError(f"Unknown sticking gas species {species!r}.")
        return StickingRatePlan(
            float(payload["coefficient"]),
            schema.molar_masses[schema.species_names.index(species)],
        )
    if kind == "plog":
        entries = _sequence(payload, "entries")
        return PLogRatePlan(
            [float(value["pressure"]) * units["pressure"] for value in entries],
            [
                _rate(
                    {"type": "arrhenius", **_mapping(value, "rate")},
                    schema,
                    units,
                    pre_factor,
                    concentration_factor,
                )
                for value in entries
            ],
        )
    if kind == "chebyshev":
        coefficients = np.asarray(payload["coefficients"], dtype=float).copy()
        coefficients[0, 0] = coefficients[0, 0] + np.log10(pre_factor)
        return ChebyshevRatePlan(
            coefficients,
            float(payload["minimum-temperature"]),
            float(payload["maximum-temperature"]),
            float(payload["minimum-pressure"]) * units["pressure"],
            float(payload["maximum-pressure"]) * units["pressure"],
        )
    if kind == "photolysis":
        return PhotolysisRatePlan(int(payload["channel"]))
    raise ValueError(f"Unsupported reaction rate type {kind!r}.")


def _mapping(payload, key):
    if key not in payload or not isinstance(payload[key], Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return payload[key]


def _mapping_value(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Expected mapping value.")
    return payload


def _sequence(payload, key):
    if key not in payload or isinstance(payload[key], (str, bytes, Mapping)):
        raise ValueError(f"{key} must be a sequence.")
    return tuple(payload[key])


def _string(payload, key):
    if key not in payload or not str(payload[key]):
        raise ValueError(f"{key} must be a nonempty string.")
    return str(payload[key])


__all__ = ["ChemicalMechanismImportReport", "load_chemical_mechanism_yaml"]
