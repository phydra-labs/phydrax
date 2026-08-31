#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._components import Capacitor, Inductor, Resistor
from .._elements import (
    CircuitElement,
    IndependentCurrentSourceLaw,
    IndependentVoltageSourceLaw,
    VoltageControlledCurrentLaw,
    VoltageControlledVoltageLaw,
)
from .._mna import CircuitInstance, NodalCircuit, NodalPort


_SUFFIXES = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


class SpiceImportResult(StrictModule):
    circuit: NodalCircuit
    directives: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


def _number(token: str, /) -> float:
    value = token.strip().lower()
    for suffix in ("meg", "t", "g", "k", "m", "u", "n", "p", "f"):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * _SUFFIXES[suffix]
    return float(value)


def _normalized_lines(text: str, /) -> tuple[str, ...]:
    lines: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            if not current:
                raise ValueError("SPICE continuation appears before a statement.")
            current = f"{current} {line[1:].strip()}"
        else:
            if current:
                lines.append(current)
            current = line
    if current:
        lines.append(current)
    return tuple(lines)


def _subcircuits(lines: tuple[str, ...], /):
    definitions: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    top: list[str] = []
    active_name: str | None = None
    active_ports: tuple[str, ...] = ()
    active_lines: list[str] = []
    for line in lines:
        tokens = tuple(line.split())
        command = tokens[0].lower()
        if command == ".subckt":
            if active_name is not None or len(tokens) < 3:
                raise ValueError("Malformed or nested .subckt definition.")
            active_name = tokens[1].lower()
            active_ports = tokens[2:]
            active_lines = []
        elif command == ".ends":
            if active_name is None:
                raise ValueError(".ends appears outside a subcircuit.")
            definitions[active_name] = (active_ports, tuple(active_lines))
            active_name = None
            active_ports = ()
            active_lines = []
        elif active_name is None:
            top.append(line)
        else:
            active_lines.append(line)
    if active_name is not None:
        raise ValueError("Unterminated .subckt definition.")
    return definitions, tuple(top)


def _expand(
    lines: Sequence[str],
    definitions,
    /,
    *,
    prefix: str = "",
    node_map: dict[str, str] | None = None,
) -> tuple[str, ...]:
    mapping = {} if node_map is None else node_map
    expanded: list[str] = []
    for line in lines:
        tokens = list(line.split())
        name = tokens[0]
        if name[0].lower() == "x":
            if len(tokens) < 3:
                raise ValueError("Malformed SPICE subcircuit instance.")
            definition_name = tokens[-1].lower()
            if definition_name not in definitions:
                raise KeyError(f"Unknown SPICE subcircuit {definition_name!r}.")
            formal, body = definitions[definition_name]
            actual = tuple(mapping.get(node, node) for node in tokens[1:-1])
            if len(formal) != len(actual):
                raise ValueError(
                    "Subcircuit instance node count does not match definition."
                )
            child_mapping = dict(zip(formal, actual, strict=True))
            expanded.extend(
                _expand(
                    body,
                    definitions,
                    prefix=f"{prefix}{name}/",
                    node_map=child_mapping,
                )
            )
            continue
        tokens[0] = f"{prefix}{name}"
        for index in range(1, len(tokens) - 1):
            tokens[index] = mapping.get(tokens[index], tokens[index])
        expanded.append(" ".join(tokens))
    return tuple(expanded)


def read_spice_netlist(
    text: str,
    ports: Sequence[NodalPort],
    /,
    *,
    ground: str = "0",
    circuit_id: str = "spice-circuit",
) -> SpiceImportResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("SPICE netlist text must be nonempty.")
    definitions, top = _subcircuits(_normalized_lines(text))
    lines = _expand(top, definitions)
    instances: list[CircuitInstance] = []
    directives: list[str] = []
    for line in lines:
        tokens = tuple(line.split())
        name = tokens[0]
        kind = name[0].lower()
        if kind == ".":
            directives.append(line)
            continue
        expected = 6 if kind in ("e", "g") else 4
        if kind in ("r", "c", "l", "i", "v", "e", "g") and len(tokens) != expected:
            raise ValueError(f"Malformed SPICE element statement {line!r}.")
        if kind == "r":
            component = Resistor(_number(tokens[3]), component_id=name)
        elif kind == "c":
            component = Capacitor(_number(tokens[3]), component_id=name)
        elif kind == "l":
            component = Inductor(_number(tokens[3]), component_id=name)
        elif kind == "i":
            component = CircuitElement(
                IndependentCurrentSourceLaw(_number(tokens[3]), law_id=name),
                element_id=name,
            )
        elif kind == "v":
            component = CircuitElement(
                IndependentVoltageSourceLaw(_number(tokens[3]), law_id=name),
                element_id=name,
            )
        elif kind == "g":
            component = CircuitElement(
                VoltageControlledCurrentLaw(_number(tokens[5]), law_id=name),
                element_id=name,
            )
        elif kind == "e":
            component = CircuitElement(
                VoltageControlledVoltageLaw(_number(tokens[5]), law_id=name),
                element_id=name,
            )
        else:
            raise ValueError(f"Unsupported SPICE element kind {kind!r}.")
        nodes = tokens[1:5] if kind in ("e", "g") else tokens[1:3]
        instances.append(CircuitInstance(name, component, nodes))
    if not instances:
        raise ValueError("SPICE netlist contains no supported circuit elements.")
    identifier = str(circuit_id)
    if not identifier:
        raise ValueError("circuit_id must be non-empty.")
    circuit = NodalCircuit(
        tuple(instances),
        tuple(ports),
        ground=ground,
        circuit_id=identifier,
    )
    source_id = canonical_fingerprint(
        {"kind": "spice-import", "text": text, "circuit": identifier}
    )
    return SpiceImportResult(circuit, tuple(directives), source_id)


__all__ = ["SpiceImportResult", "read_spice_netlist"]
