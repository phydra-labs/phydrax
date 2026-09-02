#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import prod

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class ContractionLeg(StrictModule):
    """One ordered tensor-axis incidence on a named network edge."""

    label: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(self, label: str, dimension: int, /):
        label_ = str(label)
        dimension_ = int(dimension)
        if not label_ or dimension_ < 1:
            raise ValueError("Contraction legs require a label and positive dimension.")
        self.label = label_
        self.dimension = dimension_


class ContractionOperand(StrictModule):
    """An immutable arbitrary-incidence node; an empty leg tuple is a scalar node."""

    operand_id: str = eqx.field(static=True)
    legs: tuple[ContractionLeg, ...] = eqx.field(static=True)

    def __init__(
        self,
        operand_id: str,
        legs: Sequence[ContractionLeg] = (),
        /,
    ):
        identifier = str(operand_id)
        values = tuple(legs)
        if not identifier:
            raise ValueError("Contraction operands require a nonempty ID.")
        if any(not isinstance(leg, ContractionLeg) for leg in values):
            raise TypeError("legs must contain ContractionLeg values.")
        self.operand_id = identifier
        self.legs = values


class ContractionStructure(StrictModule):
    """Immutable labelled tensor topology with explicit ordered output incidences.

    Repeated labels on one node express a diagonal/trace. A label may occur on any
    positive number of nodes, so hyperedges have ordinary copy-tensor semantics.
    Labels omitted from ``outputs`` are summed, including labels with one incidence.
    """

    operands: tuple[ContractionOperand, ...] = eqx.field(static=True)
    outputs: tuple[str, ...] = eqx.field(static=True)
    arithmetic_domain: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    dimensions: tuple[int, ...] = eqx.field(static=True)
    incidence_counts: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        operands: Sequence[ContractionOperand],
        outputs: Sequence[str],
        /,
        *,
        arithmetic_domain: str = "ordinary",
    ):
        values = tuple(operands)
        output_labels = tuple(str(label) for label in outputs)
        if not values or any(
            not isinstance(value, ContractionOperand) for value in values
        ):
            raise TypeError(
                "operands must be a nonempty sequence of ContractionOperand values."
            )
        operand_ids = tuple(value.operand_id for value in values)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("Contraction operand IDs must be unique.")
        if any(not label for label in output_labels):
            raise ValueError("Contraction output labels must be nonempty.")
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("Contraction output labels must be unique.")
        if arithmetic_domain != "ordinary":
            raise ValueError("Only ordinary product-sum contraction is supported.")

        ordered_labels: list[str] = []
        dimensions: dict[str, int] = {}
        occurrences: Counter[str] = Counter()
        for operand in values:
            for leg in operand.legs:
                if leg.label not in dimensions:
                    ordered_labels.append(leg.label)
                    dimensions[leg.label] = leg.dimension
                elif dimensions[leg.label] != leg.dimension:
                    raise ValueError(
                        f"Contraction label {leg.label!r} has inconsistent dimensions."
                    )
                occurrences[leg.label] += 1
        if any(label not in dimensions for label in output_labels):
            raise ValueError("Every output label must occur on at least one operand.")

        self.operands = values
        self.outputs = output_labels
        self.arithmetic_domain = arithmetic_domain
        self.labels = tuple(ordered_labels)
        self.dimensions = tuple(dimensions[label] for label in ordered_labels)
        self.incidence_counts = tuple(occurrences[label] for label in ordered_labels)
        self.structure_id = canonical_fingerprint(
            {
                "kind": "ordinary-arbitrary-incidence-contraction",
                "operands": tuple(
                    (
                        operand.operand_id,
                        tuple((leg.label, leg.dimension) for leg in operand.legs),
                    )
                    for operand in values
                ),
                "outputs": output_labels,
            }
        )

    def dimension(self, label: str, /) -> int:
        label_ = str(label)
        if label_ not in self.labels:
            raise KeyError(label_)
        return self.dimensions[self.labels.index(label_)]

    @property
    def output_elements(self) -> int:
        return prod(self.dimension(label) for label in self.outputs)

    @property
    def has_hyperedges(self) -> bool:
        return any(count > 2 for count in self.incidence_counts)

    @property
    def has_diagonals(self) -> bool:
        return any(
            len({leg.label for leg in operand.legs}) != len(operand.legs)
            for operand in self.operands
        )


class CircuitGate(StrictModule):
    gate_id: str = eqx.field(static=True)
    wires: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, gate_id: str, wires: Sequence[int], /):
        identifier = str(gate_id)
        wires_ = tuple(int(wire) for wire in wires)
        if not identifier or not wires_ or len(set(wires_)) != len(wires_):
            raise ValueError("Circuit gates require an ID and distinct wire indices.")
        if any(wire < 0 for wire in wires_):
            raise ValueError("Circuit wire indices must be non-negative.")
        self.gate_id = identifier
        self.wires = wires_


def chain_contraction_structure(
    site_count: int,
    /,
    *,
    bond_dimension: int = 2,
    physical_dimension: int | None = 2,
    expose_physical: bool = True,
) -> ContractionStructure:
    """Build a deterministic open chain in left-to-right node order."""

    count = int(site_count)
    bond = int(bond_dimension)
    physical = None if physical_dimension is None else int(physical_dimension)
    if count < 1 or bond < 1 or (physical is not None and physical < 1):
        raise ValueError("Chain dimensions and site_count must be positive.")
    nodes = []
    outputs = []
    for site in range(count):
        legs = []
        if site:
            legs.append(ContractionLeg(f"bond:{site - 1}:{site}", bond))
        if physical is not None:
            label = f"physical:{site}"
            legs.append(ContractionLeg(label, physical))
            if expose_physical:
                outputs.append(label)
        if site + 1 < count:
            legs.append(ContractionLeg(f"bond:{site}:{site + 1}", bond))
        nodes.append(ContractionOperand(f"site:{site}", tuple(legs)))
    return ContractionStructure(tuple(nodes), tuple(outputs))


def lattice_contraction_structure(
    rows: int,
    columns: int,
    /,
    *,
    bond_dimension: int = 2,
    physical_dimension: int | None = 2,
    expose_physical: bool = True,
) -> ContractionStructure:
    """Build a row-major rectangular open-boundary square lattice."""

    rows_ = int(rows)
    columns_ = int(columns)
    bond = int(bond_dimension)
    physical = None if physical_dimension is None else int(physical_dimension)
    if rows_ < 1 or columns_ < 1 or bond < 1 or (physical is not None and physical < 1):
        raise ValueError("Lattice extents and dimensions must be positive.")
    nodes = []
    outputs = []
    for row in range(rows_):
        for column in range(columns_):
            legs = []
            if row:
                legs.append(ContractionLeg(f"vertical:{row - 1}:{column}", bond))
            if column + 1 < columns_:
                legs.append(ContractionLeg(f"horizontal:{row}:{column}", bond))
            if row + 1 < rows_:
                legs.append(ContractionLeg(f"vertical:{row}:{column}", bond))
            if column:
                legs.append(ContractionLeg(f"horizontal:{row}:{column - 1}", bond))
            if physical is not None:
                label = f"physical:{row}:{column}"
                legs.append(ContractionLeg(label, physical))
                if expose_physical:
                    outputs.append(label)
            nodes.append(ContractionOperand(f"site:{row}:{column}", tuple(legs)))
    return ContractionStructure(tuple(nodes), tuple(outputs))


def tree_contraction_structure(
    leaf_count: int,
    /,
    *,
    bond_dimension: int = 2,
    physical_dimension: int = 2,
) -> ContractionStructure:
    """Build a deterministic, balanced binary TTN with exposed leaf legs."""

    leaves = int(leaf_count)
    bond = int(bond_dimension)
    physical = int(physical_dimension)
    if leaves < 1 or bond < 1 or physical < 1:
        raise ValueError("Tree sizes and dimensions must be positive.")
    nodes: list[ContractionOperand] = []
    outputs: list[str] = []
    frontier: list[tuple[str, str]] = []
    for leaf in range(leaves):
        physical_label = f"physical:{leaf}"
        outputs.append(physical_label)
        edge = f"tree:leaf:{leaf}"
        nodes.append(
            ContractionOperand(
                f"leaf:{leaf}",
                (ContractionLeg(physical_label, physical), ContractionLeg(edge, bond)),
            )
        )
        frontier.append((f"leaf:{leaf}", edge))
    level = 0
    while len(frontier) > 1:
        next_frontier: list[tuple[str, str]] = []
        for offset in range(0, len(frontier), 2):
            left = frontier[offset]
            if offset + 1 == len(frontier):
                next_frontier.append(left)
                continue
            right = frontier[offset + 1]
            parent_id = f"branch:{level}:{offset // 2}"
            parent_edge = f"tree:{level}:{offset // 2}"
            nodes.append(
                ContractionOperand(
                    parent_id,
                    (
                        ContractionLeg(left[1], bond),
                        ContractionLeg(right[1], bond),
                        ContractionLeg(parent_edge, bond),
                    ),
                )
            )
            next_frontier.append((parent_id, parent_edge))
        frontier = next_frontier
        level += 1
    nodes.append(
        ContractionOperand(
            "root",
            (ContractionLeg(frontier[0][1], bond),),
        )
    )
    return ContractionStructure(tuple(nodes), tuple(outputs))


def circuit_contraction_structure(
    wire_dimensions: Sequence[int],
    gates: Sequence[CircuitGate],
    /,
) -> ContractionStructure:
    """Lower a gate list to an operator tensor network with explicit I/O legs."""

    dimensions = tuple(int(value) for value in wire_dimensions)
    gates_ = tuple(gates)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("Circuit wire dimensions must be positive.")
    if any(not isinstance(gate, CircuitGate) for gate in gates_):
        raise TypeError("gates must contain CircuitGate values.")
    gate_ids = tuple(gate.gate_id for gate in gates_)
    if len(set(gate_ids)) != len(gate_ids):
        raise ValueError("Circuit gate IDs must be unique.")
    versions = [0] * len(dimensions)
    nodes = []
    for gate in gates_:
        if any(wire >= len(dimensions) for wire in gate.wires):
            raise ValueError("A circuit gate references an unavailable wire.")
        incoming = tuple(
            ContractionLeg(f"wire:{wire}:{versions[wire]}", dimensions[wire])
            for wire in gate.wires
        )
        for wire in gate.wires:
            versions[wire] += 1
        outgoing = tuple(
            ContractionLeg(f"wire:{wire}:{versions[wire]}", dimensions[wire])
            for wire in gate.wires
        )
        nodes.append(ContractionOperand(gate.gate_id, outgoing + incoming))
    for wire, dimension in enumerate(dimensions):
        if versions[wire] == 0:
            nodes.append(
                ContractionOperand(
                    f"identity-wire:{wire}",
                    (
                        ContractionLeg(f"wire:{wire}:1", dimension),
                        ContractionLeg(f"wire:{wire}:0", dimension),
                    ),
                )
            )
            versions[wire] = 1
    outputs = tuple(f"wire:{wire}:{versions[wire]}" for wire in range(len(dimensions)))
    outputs += tuple(f"wire:{wire}:0" for wire in range(len(dimensions)))
    return ContractionStructure(tuple(nodes), outputs)


def factor_graph_contraction_structure(
    variable_dimensions: Mapping[str, int],
    factors: Mapping[str, Sequence[str]],
    /,
    *,
    outputs: Sequence[str] = (),
) -> ContractionStructure:
    """Build a deterministic factor graph; shared variable labels are hyperedges."""

    dimensions = {str(name): int(value) for name, value in variable_dimensions.items()}
    if not dimensions or any(not name or value < 1 for name, value in dimensions.items()):
        raise ValueError("Factor-graph variables require names and positive dimensions.")
    nodes = []
    for factor_id in sorted(str(name) for name in factors):
        scope = tuple(str(name) for name in factors[factor_id])
        if not scope:
            nodes.append(ContractionOperand(factor_id, ()))
            continue
        if any(name not in dimensions for name in scope):
            raise ValueError("A factor references an undeclared variable.")
        nodes.append(
            ContractionOperand(
                factor_id,
                tuple(ContractionLeg(name, dimensions[name]) for name in scope),
            )
        )
    return ContractionStructure(tuple(nodes), tuple(str(name) for name in outputs))


__all__ = [
    "CircuitGate",
    "ContractionLeg",
    "ContractionOperand",
    "ContractionStructure",
    "chain_contraction_structure",
    "circuit_contraction_structure",
    "factor_graph_contraction_structure",
    "lattice_contraction_structure",
    "tree_contraction_structure",
]
