#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import yaml
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import ChemicalMechanismIR, ChemicalReactionSpec
from ...equations._chemical_rates import (
    ArrheniusRatePlan,
    ChebyshevRatePlan,
    LindemannRatePlan,
    PLogRatePlan,
    ThirdBodyRatePlan,
    TroeRatePlan,
)
from ...equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ...equations._chemical_thermodynamics import (
    NASAPolynomialKind,
    NASASpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
)


_ATOMIC_MASS = {
    "H": 1.00794,
    "He": 4.002602,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Si": 28.0855,
    "P": 30.973761998,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
}


class CanteraAdapterError(RuntimeError):
    pass


class CanteraUnsupportedFeatureError(CanteraAdapterError):
    def __init__(self, features: Sequence[str], /):
        self.features = tuple(str(value) for value in features)
        super().__init__(
            "Unsupported Cantera features: " + ", ".join(self.features) + "."
        )


class CanteraNonDifferentiableBoundaryError(CanteraAdapterError):
    pass


class CanteraImportFeatureReport(StrictModule, NonTrainableState):
    phase_name: str = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    reaction_count: int = eqx.field(static=True)
    thermodynamic_model: str = eqx.field(static=True)
    transport_model: str | None = eqx.field(static=True)
    supported_features: tuple[str, ...] = eqx.field(static=True)
    unsupported_features: tuple[str, ...] = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class CanteraMechanismImport(StrictModule, NonTrainableState):
    mechanism: ChemicalMechanismIR
    report: CanteraImportFeatureReport
    source_path: str = eqx.field(static=True)
    import_id: str = eqx.field(static=True)


class CanteraReferenceState(StrictModule, NonTrainableState):
    temperature: float = eqx.field(static=True)
    pressure: float = eqx.field(static=True)
    density: float = eqx.field(static=True)
    mean_molar_mass: float = eqx.field(static=True)
    mass_fractions: tuple[float, ...] = eqx.field(static=True)
    mole_fractions: tuple[float, ...] = eqx.field(static=True)
    specific_heat_capacity_pressure: float = eqx.field(static=True)
    specific_heat_capacity_volume: float = eqx.field(static=True)
    specific_enthalpy: float = eqx.field(static=True)
    specific_internal_energy: float = eqx.field(static=True)
    species_molar_production_rate: tuple[float, ...] = eqx.field(static=True)
    heat_release_rate: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)


class CanteraYAMLAdapter(StrictModule, NonTrainableState):
    """Host-only importer for a deliberate Cantera gas-phase YAML subset."""

    phase_name: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(self, phase_name: str = "gas", /):
        name = str(phase_name)
        if not name:
            raise ValueError("phase_name must be nonempty.")
        self.phase_name = name
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "cantera-yaml-host-adapter",
                "phase_name": name,
                "differentiable": False,
            }
        )

    def inspect(self, path: str | Path, /) -> CanteraImportFeatureReport:
        source, payload = _load_yaml(path)
        del source
        phases = _sequence(payload, "phases")
        unsupported: list[str] = []
        units = payload.get("units")
        required_units = {
            "length": "m",
            "quantity": "mol",
            "activation-energy": "J/mol",
            "pressure": "Pa",
        }
        if not isinstance(units, Mapping) or any(
            str(units.get(name, "")) != value for name, value in required_units.items()
        ):
            unsupported.append("unit-system:requires-m-mol-J/mol-Pa")
        matching = [value for value in phases if _text(value, "name") == self.phase_name]
        if len(matching) != 1:
            unsupported.append(f"phase-selection:{self.phase_name}")
            phase: Mapping[str, Any] = {}
        else:
            phase = matching[0]
        thermo_model = str(phase.get("thermo", ""))
        if thermo_model != "ideal-gas":
            unsupported.append(f"phase-thermo:{thermo_model or 'missing'}")
        kinetics = str(phase.get("kinetics", "gas"))
        if kinetics not in ("gas", "bulk"):
            unsupported.append(f"phase-kinetics:{kinetics}")
        transport_value = phase.get("transport")
        transport = None if transport_value is None else str(transport_value)
        if transport not in (None, "mixture-averaged", "multicomponent"):
            unsupported.append(f"phase-transport:{transport}")
        species_entries = _sequence(payload, "species")
        selected_names = _phase_species_names(phase, species_entries)
        species_by_name = {_text(value, "name"): value for value in species_entries}
        selected = [
            species_by_name[name] for name in selected_names if name in species_by_name
        ]
        if len(selected) != len(selected_names):
            unsupported.append("phase-species-selection")
        models = {str(_mapping(value, "thermo").get("model", "")) for value in selected}
        if not models:
            thermo_species_model = "missing"
            unsupported.append("species-thermodynamics:missing")
        elif len(models) != 1:
            thermo_species_model = "mixed"
            unsupported.append("species-thermodynamics:mixed-models")
        else:
            thermo_species_model = next(iter(models))
            if thermo_species_model not in ("NASA7", "NASA9"):
                unsupported.append(f"species-thermodynamics:{thermo_species_model}")
        for species in selected:
            composition = _mapping(species, "composition")
            for element in composition:
                if str(element) not in _ATOMIC_MASS:
                    unsupported.append(f"atomic-mass:{element}")
        reactions = _sequence_or_empty(payload, "reactions")
        supported_reactions = {
            "elementary",
            "three-body",
            "falloff",
            "pressure-dependent-Arrhenius",
            "Chebyshev",
        }
        for index, reaction in enumerate(reactions):
            kind = str(reaction.get("type", "elementary"))
            if kind not in supported_reactions:
                unsupported.append(f"reaction:{index}:type:{kind}")
            if "coverage-dependencies" in reaction:
                unsupported.append(f"reaction:{index}:coverage-dependencies")
            if "sticking-coefficient" in reaction:
                unsupported.append(f"reaction:{index}:sticking")
        supported = (
            "ideal-gas",
            "si-mol-unit-system",
            thermo_species_model,
            *(f"reaction:{value}" for value in sorted(supported_reactions)),
        )
        unsupported_values = tuple(dict.fromkeys(unsupported))
        report_id = canonical_fingerprint(
            {
                "kind": "cantera-import-feature-report",
                "adapter": self.adapter_id,
                "phase": self.phase_name,
                "species": len(selected_names),
                "reactions": len(reactions),
                "thermodynamics": thermo_species_model,
                "transport": transport,
                "unsupported": list(unsupported_values),
            }
        )
        return CanteraImportFeatureReport(
            self.phase_name,
            len(selected_names),
            len(reactions),
            thermo_species_model,
            transport,
            supported,
            unsupported_values,
            False,
            not unsupported_values,
            report_id,
        )

    def import_mechanism(self, path: str | Path, /) -> CanteraMechanismImport:
        source, payload = _load_yaml(path)
        report = self.inspect(source)
        if not report.supported:
            raise CanteraUnsupportedFeatureError(report.unsupported_features)
        phase = next(
            value
            for value in _sequence(payload, "phases")
            if _text(value, "name") == self.phase_name
        )
        species_entries = _sequence(payload, "species")
        species_by_name = {_text(value, "name"): value for value in species_entries}
        names = _phase_species_names(phase, species_entries)
        selected = tuple(species_by_name[name] for name in names)
        elements = tuple(
            dict.fromkeys(
                str(element)
                for species in selected
                for element in _mapping(species, "composition")
            )
        )
        composition = np.zeros((len(elements), len(names)), dtype=np.int32)
        element_index = {name: index for index, name in enumerate(elements)}
        masses = np.zeros(len(names), dtype=float)
        charges = np.zeros(len(names), dtype=np.int32)
        for species_index, species in enumerate(selected):
            for element, count_value in _mapping(species, "composition").items():
                count = int(count_value)
                if count != count_value or count < 0:
                    raise CanteraAdapterError(
                        f"Species {names[species_index]!r} has invalid composition."
                    )
                composition[element_index[str(element)], species_index] = count
                masses[species_index] += 1.0e-3 * _ATOMIC_MASS[str(element)] * count
            charge = species.get("charge", 0)
            if int(charge) != charge:
                raise CanteraAdapterError(
                    f"Species {names[species_index]!r} has nonintegral charge."
                )
            charges[species_index] = int(charge)
        schema = ChemicalSpeciesSchema(
            names,
            (ChemicalPhaseKind.GAS,) * len(names),
            masses,
            elements,
            composition,
            charges,
        )
        thermodynamics = _cantera_thermodynamics(schema, selected)
        reactions = tuple(
            _cantera_reaction(value, names, reaction_index)
            for reaction_index, value in enumerate(
                _sequence_or_empty(payload, "reactions")
            )
        )
        if not reactions:
            raise CanteraAdapterError("Cantera reacting mechanisms require reactions.")
        mechanism = ChemicalMechanismIR(
            str(payload.get("description", source.stem)),
            schema,
            thermodynamics,
            reactions,
        )
        import_id = canonical_fingerprint(
            {
                "kind": "cantera-mechanism-import",
                "adapter": self.adapter_id,
                "report": report.report_id,
                "source": canonical_fingerprint(payload),
                "schema": schema.schema_id,
            }
        )
        return CanteraMechanismImport(
            mechanism,
            report,
            str(source),
            import_id,
        )


class CanteraReferenceAdapter(StrictModule, NonTrainableState):
    """Explicitly non-differentiable host boundary to a Cantera Solution."""

    solution: Any = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(self, solution: Any, /, *, solution_id: str):
        identifier = str(solution_id)
        if not identifier:
            raise ValueError("solution_id must be nonempty.")
        self.solution = solution
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "cantera-reference-host-adapter",
                "solution": identifier,
                "differentiable": False,
            }
        )

    def evaluate(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass_fractions: ArrayLike,
        /,
    ) -> CanteraReferenceState:
        _refuse_device_value(temperature, "temperature")
        _refuse_device_value(pressure, "pressure")
        _refuse_device_value(mass_fractions, "mass_fractions")
        temperature_ = np.asarray(temperature, dtype=float)
        pressure_ = np.asarray(pressure, dtype=float)
        mass = np.asarray(mass_fractions, dtype=float)
        if temperature_.shape != () or pressure_.shape != () or mass.ndim != 1:
            raise CanteraAdapterError(
                "Cantera reference evaluation accepts one host scalar state at a time."
            )
        if (
            not np.isfinite(temperature_)
            or temperature_ <= 0.0
            or not np.isfinite(pressure_)
            or pressure_ <= 0.0
            or np.any(~np.isfinite(mass))
            or np.any(mass < 0.0)
            or not np.isclose(np.sum(mass), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise CanteraAdapterError("Cantera reference state is invalid.")
        self.solution.TPY = (float(temperature_), float(pressure_), mass)
        mole = np.asarray(self.solution.X, dtype=float)
        production = np.asarray(self.solution.net_production_rates, dtype=float)
        values = (
            float(self.solution.density),
            float(self.solution.mean_molecular_weight) * 1.0e-3,
            float(self.solution.cp_mass),
            float(self.solution.cv_mass),
            float(self.solution.enthalpy_mass),
            float(self.solution.int_energy_mass),
            float(self.solution.heat_release_rate),
        )
        if any(not np.isfinite(value) for value in values) or np.any(
            ~np.isfinite(production)
        ):
            raise CanteraAdapterError("Cantera returned a nonfinite reference state.")
        reference_id = canonical_fingerprint(
            {
                "kind": "cantera-reference-state",
                "adapter": self.adapter_id,
                "temperature": float(temperature_),
                "pressure": float(pressure_),
                "mass_fractions": mass.tolist(),
            }
        )
        return CanteraReferenceState(
            float(temperature_),
            float(pressure_),
            values[0],
            values[1],
            tuple(float(value) for value in mass),
            tuple(float(value) for value in mole),
            values[2],
            values[3],
            values[4],
            values[5],
            tuple(float(value) for value in production),
            values[6],
            reference_id,
        )


def _refuse_device_value(value: Any, name: str, /) -> None:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        raise CanteraNonDifferentiableBoundaryError(
            f"{name} crossed the host-only, non-differentiable Cantera boundary."
        )


def _load_yaml(path: str | Path, /) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CanteraAdapterError(f"Cantera YAML file does not exist: {source}.")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CanteraAdapterError("Cantera YAML root must be a mapping.")
    return source, payload


def _sequence(payload: Mapping[str, Any], key: str, /) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CanteraAdapterError(f"Cantera YAML {key!r} must be a sequence.")
    if any(not isinstance(item, Mapping) for item in value):
        raise CanteraAdapterError(f"Cantera YAML {key!r} entries must be mappings.")
    return tuple(value)


def _sequence_or_empty(
    payload: Mapping[str, Any], key: str, /
) -> tuple[Mapping[str, Any], ...]:
    if key not in payload:
        return ()
    return _sequence(payload, key)


def _mapping(payload: Mapping[str, Any], key: str, /) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CanteraAdapterError(f"Cantera YAML {key!r} must be a mapping.")
    return value


def _text(payload: Mapping[str, Any], key: str, /) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise CanteraAdapterError(f"Cantera YAML {key!r} must be nonempty.")
    return value


def _phase_species_names(
    phase: Mapping[str, Any], species_entries: Sequence[Mapping[str, Any]], /
) -> tuple[str, ...]:
    selected = phase.get("species", "all")
    if selected == "all":
        return tuple(_text(value, "name") for value in species_entries)
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise CanteraAdapterError("Cantera phase species must be 'all' or a sequence.")
    names = tuple(str(value) for value in selected)
    if not names or any(not value for value in names) or len(set(names)) != len(names):
        raise CanteraAdapterError("Cantera phase species selection is invalid.")
    return names


def _cantera_thermodynamics(
    schema: ChemicalSpeciesSchema,
    species_entries: Sequence[Mapping[str, Any]],
    /,
) -> NASASpeciesThermodynamicsPlan:
    thermo_entries = tuple(_mapping(value, "thermo") for value in species_entries)
    model = str(thermo_entries[0]["model"])
    kind = NASAPolynomialKind.NASA7 if model == "NASA7" else NASAPolynomialKind.NASA9
    coefficient_count = 7 if kind is NASAPolynomialKind.NASA7 else 9
    ranges = [
        np.asarray(value.get("temperature-ranges"), dtype=float)
        for value in thermo_entries
    ]
    data = [np.asarray(value.get("data"), dtype=float) for value in thermo_entries]
    interval_count = len(ranges[0]) - 1
    if (
        interval_count < 1
        or any(value.shape != (interval_count + 1,) for value in ranges)
        or any(value.shape != (interval_count, coefficient_count) for value in data)
    ):
        raise CanteraAdapterError(
            "Cantera NASA species must share one valid interval structure."
        )
    lower = np.stack([value[:-1] for value in ranges])
    upper = np.stack([value[1:] for value in ranges])
    return NASASpeciesThermodynamicsPlan(
        schema,
        kind,
        np.stack(data),
        lower,
        upper,
    )


def _cantera_reaction(
    payload: Mapping[str, Any],
    species_names: Sequence[str],
    reaction_index: int,
    /,
) -> ChemicalReactionSpec:
    equation = _text(payload, "equation")
    if "<=>" in equation:
        left, right = equation.split("<=>", maxsplit=1)
        reversible = True
    elif "=>" in equation:
        left, right = equation.split("=>", maxsplit=1)
        reversible = False
    else:
        raise CanteraAdapterError(
            f"Reaction equation has no supported arrow: {equation!r}."
        )
    reactants = _equation_side(left, species_names)
    products = _equation_side(right, species_names)
    kind = str(payload.get("type", "elementary"))
    efficiencies = np.ones(len(species_names), dtype=float)
    species_index = {name: index for index, name in enumerate(species_names)}
    for name, value in _mapping_or_empty(payload, "efficiencies").items():
        if str(name) not in species_index:
            raise CanteraAdapterError(
                f"Third-body efficiency names unknown species {name!r}."
            )
        efficiencies[species_index[str(name)]] = float(value)
    if kind == "elementary":
        rate = _arrhenius(_mapping(payload, "rate-constant"))
    elif kind == "three-body":
        rate = ThirdBodyRatePlan(
            _arrhenius(_mapping(payload, "rate-constant")), efficiencies
        )
    elif kind == "falloff":
        low = _arrhenius(_mapping(payload, "low-P-rate-constant"))
        high = _arrhenius(_mapping(payload, "high-P-rate-constant"))
        troe = payload.get("Troe")
        if troe is None:
            rate = LindemannRatePlan(low, high, efficiencies)
        elif isinstance(troe, Mapping):
            rate = TroeRatePlan(
                low,
                high,
                efficiencies,
                float(troe["A"]),
                _temperature(troe["T1"]),
                _temperature(troe.get("T2", 1.0e30)),
                _temperature(troe["T3"]),
            )
        else:
            raise CanteraAdapterError("Troe falloff parameters must be a mapping.")
    elif kind == "pressure-dependent-Arrhenius":
        entries = payload.get("rate-constants")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise CanteraAdapterError("PLOG rate-constants must be a sequence.")
        pressures = [_pressure(_mapping_value(value)["P"]) for value in entries]
        rates = [_arrhenius(_mapping_value(value)) for value in entries]
        order = np.argsort(np.asarray(pressures))
        rate = PLogRatePlan(
            np.asarray(pressures)[order], tuple(rates[index] for index in order)
        )
    elif kind == "Chebyshev":
        temperature_range = payload.get("temperature-range")
        pressure_range = payload.get("pressure-range")
        if (
            not isinstance(temperature_range, Sequence)
            or len(temperature_range) != 2
            or not isinstance(pressure_range, Sequence)
            or len(pressure_range) != 2
        ):
            raise CanteraAdapterError(
                "Chebyshev temperature/pressure ranges are invalid."
            )
        rate = ChebyshevRatePlan(
            np.asarray(payload.get("data"), dtype=float),
            _temperature(temperature_range[0]),
            _temperature(temperature_range[1]),
            _pressure(pressure_range[0]),
            _pressure(pressure_range[1]),
        )
    else:
        raise CanteraUnsupportedFeatureError((f"reaction-type:{kind}",))
    orders = payload.get("orders")
    order_values = None
    if orders is not None:
        if not isinstance(orders, Mapping):
            raise CanteraAdapterError("Reaction orders must be a mapping.")
        order_values = {str(name): float(value) for name, value in orders.items()}
    return ChemicalReactionSpec(
        f"{reaction_index}:{equation}",
        reactants,
        products,
        rate,
        forward_orders=order_values,
        thermodynamic_reversible=reversible,
        duplicate_group=(equation if bool(payload.get("duplicate", False)) else None),
    )


def _equation_side(side: str, species_names: Sequence[str], /) -> dict[str, float]:
    cleaned = side.replace("(+M)", "").replace("+ M", "").strip()
    values: dict[str, float] = {}
    known = set(species_names)
    for term in cleaned.split("+"):
        tokens = term.strip().split()
        if len(tokens) == 1:
            coefficient = 1.0
            name = tokens[0]
        elif len(tokens) == 2:
            coefficient = float(tokens[0])
            name = tokens[1]
        else:
            raise CanteraAdapterError(f"Cannot parse reaction term {term!r}.")
        if name not in known:
            raise CanteraAdapterError(f"Reaction references unknown species {name!r}.")
        values[name] = values.get(name, 0.0) + coefficient
    return values


def _mapping_or_empty(payload: Mapping[str, Any], key: str, /) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise CanteraAdapterError(f"Cantera YAML {key!r} must be a mapping.")
    return value


def _mapping_value(value: Any, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanteraAdapterError("Expected a Cantera mapping entry.")
    return value


def _arrhenius(payload: Mapping[str, Any], /) -> ArrheniusRatePlan:
    return ArrheniusRatePlan(
        _number(payload["A"]),
        _number(payload.get("b", 0.0)),
        _activation_energy(payload.get("Ea", 0.0)),
    )


def _number(value: Any, /) -> float:
    if isinstance(value, str):
        tokens = value.split()
        if len(tokens) != 1:
            raise CanteraAdapterError(
                f"Unitful scalar {value!r} is unsupported at this field."
            )
        return float(tokens[0])
    return float(value)


def _temperature(value: Any, /) -> float:
    if isinstance(value, str):
        tokens = value.split()
        if len(tokens) != 2 or tokens[1] != "K":
            raise CanteraAdapterError(f"Unsupported temperature unit in {value!r}.")
        return float(tokens[0])
    return float(value)


def _pressure(value: Any, /) -> float:
    if not isinstance(value, str):
        return float(value)
    tokens = value.split()
    if len(tokens) != 2:
        raise CanteraAdapterError(f"Unsupported pressure value {value!r}.")
    factors = {"Pa": 1.0, "kPa": 1.0e3, "bar": 1.0e5, "atm": 101325.0}
    if tokens[1] not in factors:
        raise CanteraAdapterError(f"Unsupported pressure unit {tokens[1]!r}.")
    return float(tokens[0]) * factors[tokens[1]]


def _activation_energy(value: Any, /) -> float:
    if not isinstance(value, str):
        return float(value)
    tokens = value.split()
    if len(tokens) != 2:
        raise CanteraAdapterError(f"Unsupported activation energy {value!r}.")
    factors = {
        "J/mol": 1.0,
        "kJ/mol": 1.0e3,
        "cal/mol": 4.184,
        "kcal/mol": 4184.0,
        "K": UNIVERSAL_GAS_CONSTANT,
    }
    if tokens[1] not in factors:
        raise CanteraAdapterError(f"Unsupported activation-energy unit {tokens[1]!r}.")
    return float(tokens[0]) * factors[tokens[1]]


__all__ = [
    "CanteraAdapterError",
    "CanteraImportFeatureReport",
    "CanteraMechanismImport",
    "CanteraNonDifferentiableBoundaryError",
    "CanteraReferenceAdapter",
    "CanteraReferenceState",
    "CanteraUnsupportedFeatureError",
    "CanteraYAMLAdapter",
]
