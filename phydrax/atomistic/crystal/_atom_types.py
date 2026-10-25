#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Topology/stereo-aware atom types and bounded molecular correspondence.

Everything here is host-only exact preparation over explicit molecular graphs.
Three declared typing modes exist:

* ``"connectivity"`` uses element identity and bond presence only.  This is the
  pinned NVIDIA ALCHEMI Toolkit ``topological_atom_types`` contract (exact
  label-preserving automorphism orbits).
* ``"chemical"`` additionally preserves formal charges and bond orders.
* ``"stereo"`` additionally preserves declared tetrahedral parities and
  double-bond configurations.

Types are *exact* automorphism orbits, never only color-refinement classes:
refinement supplies candidates and pruning, and a bounded backtracking search
proves every orbit.  Exhausting a declared node budget is reported (or refused
where a construction requires exact orbits); it is never treated as proof of
distinction.

Stereo conventions (shared with `phydrax.atomistic.crystal.MolecularComponent`):

* A tetrahedral entry ``(center, n1, n2, n3, sign)`` lists the three
  smallest-index neighbors in ascending order; ``sign`` is the sign of
  ``det[x(n1)-x(c), x(n2)-x(c), x(n3)-x(c)]``.  For four-coordinate centers the
  fourth neighbor is implied; three-coordinate centers carry an implicit lone
  pair.  The orientation of an ordered neighbor tuple is alternating under its
  permutations, so a mapping transports a parity by the sign of the
  permutation that sorts the mapped neighbors.
* A double-bond entry ``(a, b, c, d, sign)`` has ``b < c`` bonded with order 2,
  ``a`` the smallest substituent of ``b`` and ``d`` the smallest substituent of
  ``c``; ``sign`` is ``+1`` for cis (``cos(phi_abcd) > 0``) and ``-1`` for
  trans.
* A mirror inverts every tetrahedral parity and preserves double-bond
  configurations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import assert_never, final, Literal, NamedTuple, TYPE_CHECKING, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import positive_integer
from ...typing import parse


if TYPE_CHECKING:
    from ._input import MolecularComponent


AtomTypeMode: TypeAlias = Literal["connectivity", "chemical", "stereo"]
"""Graph invariants an atom type or correspondence preserves."""

CorrespondenceStatus: TypeAlias = Literal[
    "complete", "incompatible", "mapping-limit", "node-limit"
]
"""Search outcome.

``"complete"``: every mapping was found and returned.  ``"incompatible"``: the
search finished and no mapping exists.  ``"mapping-limit"``: more than
``max_mappings`` mappings exist; the first ``max_mappings`` in search order
are returned.  ``"node-limit"``: the node budget ended the search; this proves
neither existence nor absence beyond the returned mappings.
"""

type TetrahedralSpec = tuple[int, int, int, int, int]
type DoubleBondSpec = tuple[int, int, int, int, int]

DEFAULT_MAXIMUM_SEARCH_NODES = 200_000
DEFAULT_MAXIMUM_AUTOMORPHISMS = 4096


class LabeledGraph(NamedTuple):
    """Host molecular graph under one typing mode.

    ``neighbors[i]`` maps each neighbor of atom ``i`` to its edge label;
    ``tetrahedral`` maps a center to its sorted full neighbor tuple and parity;
    ``double_bonds`` maps ``(b, c)`` to ``(a, d, sign)``.
    """

    labels: tuple[tuple[int, ...], ...]
    neighbors: tuple[dict[int, int], ...]
    tetrahedral: dict[int, tuple[tuple[int, ...], int]]
    double_bonds: dict[tuple[int, int], tuple[int, int, int]]


def labeled_graph(
    atomic_numbers: Sequence[int],
    formal_charges: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    bond_orders: Sequence[int],
    tetrahedral_stereo: Sequence[TetrahedralSpec],
    double_bond_stereo: Sequence[DoubleBondSpec],
    mode: AtomTypeMode,
    /,
) -> LabeledGraph:
    """Build the mode-specific labeled graph of validated canonical inputs."""
    count = len(atomic_numbers)
    neighbors: list[dict[int, int]] = [{} for _ in range(count)]
    match mode:
        case "connectivity":
            for left, right in bonds:
                neighbors[left][right] = neighbors[right][left] = 1
            labels = tuple((number,) for number in atomic_numbers)
            return LabeledGraph(labels, tuple(neighbors), {}, {})
        case "chemical" | "stereo":
            for (left, right), order in zip(bonds, bond_orders, strict=True):
                neighbors[left][right] = neighbors[right][left] = order
        case _ as unreachable:
            assert_never(unreachable)
    if mode == "chemical":
        labels = tuple(
            (number, charge)
            for number, charge in zip(atomic_numbers, formal_charges, strict=True)
        )
        return LabeledGraph(labels, tuple(neighbors), {}, {})
    tetrahedral = {
        center: (tuple(sorted(neighbors[center])), sign)
        for center, _, _, _, sign in tetrahedral_stereo
    }
    double_bonds = {(b, c): (a, d, sign) for a, b, c, d, sign in double_bond_stereo}
    stereo_atoms = set(tetrahedral)
    double_atoms = {atom for key in double_bonds for atom in key}
    labels = tuple(
        (number, charge, int(atom in stereo_atoms), int(atom in double_atoms))
        for atom, (number, charge) in enumerate(
            zip(atomic_numbers, formal_charges, strict=True)
        )
    )
    return LabeledGraph(labels, tuple(neighbors), tetrahedral, double_bonds)


def refine_colors(graphs: Sequence[LabeledGraph], /) -> tuple[tuple[int, ...], ...]:
    """Jointly refine atom colors of graphs to their stable partition.

    Colors are ranks of sorted signatures over the disjoint union, so equal
    colors in different graphs carry the same refinement history and are valid
    correspondence candidates.  Color refinement is coarser than or equal to the
    automorphism orbit partition; it never proves equivalence by itself.
    """
    distinct = sorted({label for graph in graphs for label in graph.labels})
    rank = {label: index for index, label in enumerate(distinct)}
    colors = [[rank[label] for label in graph.labels] for graph in graphs]
    count = len(distinct)
    while True:
        signatures = [
            [
                (
                    color[atom],
                    tuple(
                        sorted(
                            (edge, color[neighbor])
                            for neighbor, edge in graph.neighbors[atom].items()
                        )
                    ),
                )
                for atom in range(len(color))
            ]
            for graph, color in zip(graphs, colors, strict=True)
        ]
        unique = sorted({signature for row in signatures for signature in row})
        index = {signature: position for position, signature in enumerate(unique)}
        colors = [[index[signature] for signature in row] for row in signatures]
        if len(unique) == count:
            return tuple(tuple(row) for row in colors)
        count = len(unique)


def _permutation_sign(values: Sequence[int], /) -> int:
    inversions = sum(
        1
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if values[left] > values[right]
    )
    return -1 if inversions % 2 else 1


def _tetrahedral_consistent(
    source: LabeledGraph,
    target: LabeledGraph,
    mapping: Sequence[int],
    center: int,
    mirror: bool,
    /,
) -> bool:
    neighbors, sign = source.tetrahedral[center]
    entry = target.tetrahedral.get(mapping[center])
    if entry is None:
        return False
    transported = sign * _permutation_sign([mapping[atom] for atom in neighbors])
    return (-transported if mirror else transported) == entry[1]


def _double_bond_consistent(
    source: LabeledGraph,
    target: LabeledGraph,
    mapping: Sequence[int],
    key: tuple[int, int],
    /,
) -> bool:
    a, d, sign = source.double_bonds[key]
    first, second = mapping[a], mapping[d]
    begin, end = mapping[key[0]], mapping[key[1]]
    if begin > end:
        begin, end, first, second = end, begin, second, first
    entry = target.double_bonds.get((begin, end))
    if entry is None:
        return False
    target_first, target_second, target_sign = entry
    if first != target_first:
        sign = -sign
    if second != target_second:
        sign = -sign
    return sign == target_sign


class _SearchOutcome(NamedTuple):
    mappings: tuple[tuple[int, ...], ...]
    status: CorrespondenceStatus
    nodes: int


def _search_order(
    graph: LabeledGraph, colors: Sequence[int], first: int | None, /
) -> list[int]:
    count = len(colors)
    sizes: dict[int, int] = {}
    for color in colors:
        sizes[color] = sizes.get(color, 0) + 1
    seen = [False] * count
    order: list[int] = []
    starts = sorted(range(count), key=lambda atom: (sizes[colors[atom]], atom))
    if first is not None:
        starts.insert(0, first)
    for start in starts:
        if seen[start]:
            continue
        seen[start] = True
        queue = deque((start,))
        while queue:
            atom = queue.popleft()
            order.append(atom)
            for neighbor in sorted(graph.neighbors[atom]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return order


def _correspond(
    source: LabeledGraph,
    target: LabeledGraph,
    /,
    *,
    fixed: tuple[int, int] | None,
    mirror: bool,
    max_mappings: int,
    max_nodes: int,
) -> _SearchOutcome:
    """Bounded exact search for label/edge/stereo-preserving bijections."""
    count = len(source.labels)
    if count != len(target.labels):
        return _SearchOutcome((), "incompatible", 0)
    if count == 0:
        return _SearchOutcome(((),), "complete", 0)
    source_colors, target_colors = refine_colors((source, target))
    if sorted(source_colors) != sorted(target_colors):
        return _SearchOutcome((), "incompatible", 0)
    if fixed is not None and source_colors[fixed[0]] != target_colors[fixed[1]]:
        return _SearchOutcome((), "incompatible", 0)
    order = _search_order(source, source_colors, None if fixed is None else fixed[0])
    position = {atom: depth for depth, atom in enumerate(order)}
    tetrahedral_checks: list[list[int]] = [[] for _ in range(count)]
    double_checks: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for center, (neighbors, _) in source.tetrahedral.items():
        tetrahedral_checks[max(position[atom] for atom in (center, *neighbors))].append(
            center
        )
    for key, (a, d, _) in source.double_bonds.items():
        double_checks[max(position[atom] for atom in (a, *key, d))].append(key)
    by_color: dict[int, list[int]] = {}
    for atom, color in enumerate(target_colors):
        by_color.setdefault(color, []).append(atom)
    mapping = [-1] * count
    inverse = [-1] * count
    found: list[tuple[int, ...]] = []
    nodes = 0
    exhausted = False

    def candidates(atom: int) -> list[int]:
        if fixed is not None and atom == fixed[0]:
            return [fixed[1]]
        anchor = next(
            (
                neighbor
                for neighbor in sorted(
                    source.neighbors[atom], key=lambda value: position[value]
                )
                if mapping[neighbor] >= 0
            ),
            None,
        )
        if anchor is None:
            return [
                value for value in by_color[source_colors[atom]] if inverse[value] < 0
            ]
        edge = source.neighbors[atom][anchor]
        return sorted(
            value
            for value, label in target.neighbors[mapping[anchor]].items()
            if label == edge and inverse[value] < 0
        )

    def admissible(atom: int, image: int) -> bool:
        if source_colors[atom] != target_colors[image]:
            return False
        mapped_source = 0
        for neighbor, edge in source.neighbors[atom].items():
            if mapping[neighbor] < 0:
                continue
            mapped_source += 1
            if target.neighbors[image].get(mapping[neighbor]) != edge:
                return False
        mapped_target = sum(
            1 for neighbor in target.neighbors[image] if inverse[neighbor] >= 0
        )
        return mapped_source == mapped_target

    def stereo_admissible(depth: int) -> bool:
        return all(
            _tetrahedral_consistent(source, target, mapping, center, mirror)
            for center in tetrahedral_checks[depth]
        ) and all(
            _double_bond_consistent(source, target, mapping, key)
            for key in double_checks[depth]
        )

    def extend(depth: int) -> bool:
        nonlocal nodes, exhausted
        if depth == count:
            found.append(tuple(mapping))
            return len(found) <= max_mappings
        atom = order[depth]
        for image in candidates(atom):
            nodes += 1
            if nodes > max_nodes:
                exhausted = True
                return False
            if not admissible(atom, image):
                continue
            mapping[atom] = image
            inverse[image] = atom
            proceed = not stereo_admissible(depth) or extend(depth + 1)
            mapping[atom] = -1
            inverse[image] = -1
            if not proceed:
                return False
        return True

    finished = extend(0)
    if exhausted:
        return _SearchOutcome(tuple(found), "node-limit", nodes)
    if not finished:
        return _SearchOutcome(tuple(found[:max_mappings]), "mapping-limit", nodes)
    if not found:
        return _SearchOutcome((), "incompatible", nodes)
    return _SearchOutcome(tuple(sorted(found)), "complete", nodes)


class GraphSymmetry(NamedTuple):
    """Exact orbits, bounded automorphisms and chirality of one labeled graph."""

    classes: tuple[int, ...]
    automorphisms: tuple[tuple[int, ...], ...]
    automorphisms_complete: bool
    stereo_chiral: bool
    mirror_automorphism: tuple[int, ...] | None
    nodes_visited: int


def _orbits_from_mappings(
    count: int, mappings: Sequence[tuple[int, ...]], /
) -> tuple[int, ...]:
    parent = list(range(count))

    def root(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for mapping in mappings:
        for atom, image in enumerate(mapping):
            left, right = root(atom), root(image)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return _first_occurrence([root(atom) for atom in range(count)])


def _first_occurrence(values: Sequence[int], /) -> tuple[int, ...]:
    relabel: dict[int, int] = {}
    return tuple(relabel.setdefault(value, len(relabel)) for value in values)


def graph_symmetry(
    graph: LabeledGraph,
    /,
    *,
    max_automorphisms: int,
    max_nodes: int,
) -> GraphSymmetry:
    """Return exact orbits, bounded automorphisms and declared-stereo chirality.

    Orbits come from the complete automorphism list when it fits in
    ``max_automorphisms``; otherwise every same-color pair is decided by a
    rooted existence search.  Raises `ValueError` when the node budget leaves an
    orbit or the chirality undetermined, because consumers require them exact.
    """
    limit = positive_integer(max_automorphisms, "max_automorphisms")
    budget = positive_integer(max_nodes, "max_nodes")
    count = len(graph.labels)
    identity = tuple(range(count))
    outcome = _correspond(
        graph, graph, fixed=None, mirror=False, max_mappings=limit, max_nodes=budget
    )
    nodes = outcome.nodes
    if outcome.status == "complete":
        automorphisms = outcome.mappings
        classes = _orbits_from_mappings(count, automorphisms)
        complete = True
    else:
        found = set(outcome.mappings)
        found.discard(identity)
        automorphisms = (identity, *sorted(found)[: limit - 1])
        (colors,) = refine_colors((graph,))
        # Found automorphisms already merge orbits; each remaining same-color
        # pair is decided by one rooted existence search.
        seeded = list(_orbits_from_mappings(count, outcome.mappings))
        representative: dict[int, int] = {}
        for atom, orbit in enumerate(seeded):
            representative.setdefault(orbit, atom)
        roots = sorted(representative.values())
        merged = {root: root for root in roots}
        for position, root in enumerate(roots):
            if merged[root] != root:
                continue
            for image in roots[position + 1 :]:
                if merged[image] != image or colors[image] != colors[root]:
                    continue
                rooted = _correspond(
                    graph,
                    graph,
                    fixed=(root, image),
                    mirror=False,
                    max_mappings=1,
                    max_nodes=budget,
                )
                nodes += rooted.nodes
                if rooted.mappings:
                    merged[image] = root
                elif rooted.status == "node-limit":
                    raise ValueError(
                        "Molecular atom orbits are undetermined within max_nodes; "
                        "raise the declared search budget."
                    )
        classes = _first_occurrence([merged[representative[orbit]] for orbit in seeded])
        complete = False
    if not graph.tetrahedral:
        return GraphSymmetry(classes, automorphisms, complete, False, identity, nodes)
    mirrored = _correspond(
        graph, graph, fixed=None, mirror=True, max_mappings=1, max_nodes=budget
    )
    nodes += mirrored.nodes
    if mirrored.mappings:
        return GraphSymmetry(
            classes, automorphisms, complete, False, mirrored.mappings[0], nodes
        )
    if mirrored.status == "node-limit":
        raise ValueError(
            "Molecular chirality is undetermined within max_nodes; raise the "
            "declared search budget."
        )
    return GraphSymmetry(classes, automorphisms, complete, True, None, nodes)


def bond_ring_size(
    neighbors: Sequence[dict[int, int]], left: int, right: int, /, *, limit: int
) -> int | None:
    """Return the smallest ring size through bond ``left-right`` up to ``limit``."""
    return _ring_through(neighbors, (left, right), limit)


def angle_ring_size(
    neighbors: Sequence[dict[int, int]],
    first: int,
    center: int,
    last: int,
    /,
    *,
    limit: int,
) -> int | None:
    """Return the smallest ring containing the path ``first-center-last``."""
    return _ring_through(neighbors, (first, center, last), limit)


def _ring_through(
    neighbors: Sequence[dict[int, int]], path: tuple[int, ...], limit: int, /
) -> int | None:
    """Shortest cycle containing the simple ``path`` (BFS avoiding its interior)."""
    start, goal = path[-1], path[0]
    blocked = set(path[1:-1])
    distance = {start: 0}
    queue = deque((start,))
    budget = limit - (len(path) - 1)
    while queue:
        atom = queue.popleft()
        if distance[atom] >= budget:
            continue
        for neighbor in sorted(neighbors[atom]):
            if neighbor in blocked or neighbor in distance:
                continue
            if atom == start and neighbor == path[-2]:
                continue
            distance[neighbor] = distance[atom] + 1
            if neighbor == goal:
                return distance[neighbor] + len(path) - 1
            queue.append(neighbor)
    return None


class StereoCandidates(NamedTuple):
    """Topological stereo elements detected from exact chemical orbits."""

    tetrahedral: tuple[int, ...]
    double_bonds: tuple[tuple[int, int], ...]


_PYRAMIDAL_STEREO_ELEMENTS = frozenset({15, 33})
_MAXIMUM_CIS_ONLY_RING = 7


def stereo_candidates(graph: LabeledGraph, orbits: Sequence[int], /) -> StereoCandidates:
    """Detect tetrahedral and double-bond stereo candidates of a chemical graph.

    Rule set: a center with four single bonds, or three single bonds on a
    configurationally stable pyramidal P/As center or a cationic S center,
    whose neighbors lie in pairwise distinct chemical orbits; a nonring (or
    ring of at least eight atoms) order-2 bond whose ends carry no other
    multiple bond and whose two substituents, when present, lie in distinct
    orbits.  Nitrogen inversion is treated as fast.  Pseudoasymmetric and
    ring cis/trans relationships with constitutionally equivalent neighbors
    are not detected and must be declared explicitly when intended.
    """
    tetrahedral: list[int] = []
    for atom, adjacent in enumerate(graph.neighbors):
        if any(order != 1 for order in adjacent.values()):
            continue
        number, charge = graph.labels[atom][0], graph.labels[atom][1]
        pyramidal = len(adjacent) == 3 and (
            number in _PYRAMIDAL_STEREO_ELEMENTS or (number == 16 and charge == 1)
        )
        if (len(adjacent) == 4 or pyramidal) and len(
            {orbits[neighbor] for neighbor in adjacent}
        ) == len(adjacent):
            tetrahedral.append(atom)
    double_bonds: list[tuple[int, int]] = []
    for begin, adjacent in enumerate(graph.neighbors):
        for end, order in adjacent.items():
            if end <= begin or order != 2:
                continue
            ring = bond_ring_size(
                graph.neighbors, begin, end, limit=_MAXIMUM_CIS_ONLY_RING
            )
            if ring is not None:
                continue
            admissible = True
            for atom, other in ((begin, end), (end, begin)):
                substituents = [n for n in graph.neighbors[atom] if n != other]
                if not substituents or len(substituents) > 2:
                    admissible = False
                elif any(graph.neighbors[atom][n] != 1 for n in substituents) or (
                    len(substituents) == 2
                    and orbits[substituents[0]] == orbits[substituents[1]]
                ):
                    admissible = False
            if admissible:
                double_bonds.append((begin, end))
    return StereoCandidates(tuple(tetrahedral), tuple(double_bonds))


def _component_graph(
    component: MolecularComponent, mode: AtomTypeMode, /
) -> LabeledGraph:
    return labeled_graph(
        component.atomic_numbers,
        component.formal_charges,
        component.bonds,
        component.bond_orders,
        component.tetrahedral_stereo,
        component.double_bond_stereo,
        mode,
    )


@final
class CorrespondenceResult(StrictModule):
    """Bounded atom correspondences between two molecular components.

    ``mappings[k][i]`` is the atom of ``other`` corresponding to atom ``i`` of
    ``component``.  Mappings are sorted lexicographically when the search is
    complete.  ``mirror`` mappings carry ``component``'s mirror image onto
    ``other`` (an improper relation).
    """

    mappings: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    status: CorrespondenceStatus = eqx.field(static=True)
    mode: AtomTypeMode = eqx.field(static=True)
    mirror: bool = eqx.field(static=True)
    nodes_visited: int = eqx.field(static=True)
    max_mappings: int = eqx.field(static=True)
    max_nodes: int = eqx.field(static=True)
    source_component_id: str = eqx.field(static=True)
    target_component_id: str = eqx.field(static=True)

    @property
    def complete(self) -> bool:
        """Whether the search finished (all mappings returned, or none exist)."""
        return self.status in ("complete", "incompatible")


def molecular_correspondences(
    component: MolecularComponent,
    other: MolecularComponent,
    /,
    *,
    max_mappings: int,
    max_nodes: int,
    mode: AtomTypeMode = "stereo",
    mirror: bool = False,
) -> CorrespondenceResult:
    """Enumerate bounded atom correspondences preserving ``mode`` invariants.

    ``mirror=True`` (stereo mode only) maps the mirror image of ``component``
    onto ``other``.  Exhaustion of ``max_nodes`` returns ``"node-limit"`` with
    any mappings found so far; it is not evidence of distinct molecules.
    """
    mode_ = parse(mode, AtomTypeMode, "mode")
    if not isinstance(mirror, bool):
        raise TypeError("mirror must be a bool.")
    if mirror and mode_ != "stereo":
        raise ValueError("Mirror correspondences require mode='stereo'.")
    limit = positive_integer(max_mappings, "max_mappings")
    budget = positive_integer(max_nodes, "max_nodes")
    outcome = _correspond(
        _component_graph(component, mode_),
        _component_graph(other, mode_),
        fixed=None,
        mirror=mirror,
        max_mappings=limit,
        max_nodes=budget,
    )
    return CorrespondenceResult(
        mappings=outcome.mappings,
        status=outcome.status,
        mode=mode_,
        mirror=mirror,
        nodes_visited=outcome.nodes,
        max_mappings=limit,
        max_nodes=budget,
        source_component_id=component.component_id,
        target_component_id=other.component_id,
    )


def component_atom_orbits(
    component: MolecularComponent,
    mode: AtomTypeMode,
    /,
    *,
    max_automorphisms: int = DEFAULT_MAXIMUM_AUTOMORPHISMS,
    max_nodes: int = DEFAULT_MAXIMUM_SEARCH_NODES,
) -> tuple[int, ...]:
    """Exact automorphism orbits of ``component`` under ``mode`` invariants."""
    mode_ = parse(mode, AtomTypeMode, "mode")
    if mode_ == "stereo":
        return component.atom_classes
    return graph_symmetry(
        _component_graph(component, mode_),
        max_automorphisms=max_automorphisms,
        max_nodes=max_nodes,
    ).classes


@final
class AtomTypeAssignment(StrictModule):
    """Shared atom types of an ordered formula unit of molecular components.

    Atoms receive the same type exactly when a ``mode``-preserving
    correspondence of whole components carries one onto the other (including
    automorphisms within one component).  ``component_type_ids[c][i]`` is the
    type of atom ``i`` of component ``c``; ``type_ids`` lists formula-unit atoms
    in order: component ``c`` repeated ``formula_counts[c]`` times.  Type IDs
    follow first occurrence.  ``representatives[t]`` is the first
    ``(component, atom)`` of type ``t``.
    """

    component_type_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    type_ids: tuple[int, ...] = eqx.field(static=True)
    representatives: tuple[tuple[int, int], ...] = eqx.field(static=True)
    formula_counts: tuple[int, ...] = eqx.field(static=True)
    component_ids: tuple[str, ...] = eqx.field(static=True)
    mode: AtomTypeMode = eqx.field(static=True)
    max_nodes: int = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    @property
    def type_count(self) -> int:
        return len(self.representatives)


def atom_type_assignment(
    components: Sequence[MolecularComponent],
    formula_counts: Sequence[int],
    /,
    *,
    mode: AtomTypeMode = "stereo",
    max_automorphisms: int = DEFAULT_MAXIMUM_AUTOMORPHISMS,
    max_nodes: int = DEFAULT_MAXIMUM_SEARCH_NODES,
) -> AtomTypeAssignment:
    """Assign exact shared atom types to a formula unit of components.

    Raises `ValueError` when a component pair cannot be decided within
    ``max_nodes``; a partial search never merges or separates types.
    """
    mode_ = parse(mode, AtomTypeMode, "mode")
    budget = positive_integer(max_nodes, "max_nodes")
    counts = tuple(positive_integer(value, "formula_counts") for value in formula_counts)
    members = tuple(components)
    if len(members) != len(counts) or not members:
        raise ValueError("Every component requires one positive formula count.")
    graphs = tuple(_component_graph(member, mode_) for member in members)
    orbits = tuple(
        component_atom_orbits(
            member, mode_, max_automorphisms=max_automorphisms, max_nodes=budget
        )
        for member in members
    )
    next_type = 0
    component_types: list[tuple[int, ...]] = []
    representatives: list[tuple[int, int]] = []
    for index, (graph, orbit) in enumerate(zip(graphs, orbits, strict=True)):
        assigned: tuple[int, ...] | None = None
        for earlier in range(index):
            outcome = _correspond(
                graph,
                graphs[earlier],
                fixed=None,
                mirror=False,
                max_mappings=1,
                max_nodes=budget,
            )
            if outcome.mappings:
                mapping = outcome.mappings[0]
                assigned = tuple(
                    component_types[earlier][mapping[atom]] for atom in range(len(orbit))
                )
                break
            if outcome.status == "node-limit":
                raise ValueError(
                    "Component correspondence is undetermined within max_nodes; "
                    "raise the declared search budget."
                )
        if assigned is None:
            local: dict[int, int] = {}
            types: list[int] = []
            for atom, orbit_id in enumerate(orbit):
                if orbit_id not in local:
                    local[orbit_id] = next_type
                    representatives.append((index, atom))
                    next_type += 1
                types.append(local[orbit_id])
            assigned = tuple(types)
        component_types.append(assigned)
    type_ids = tuple(
        value
        for types, count in zip(component_types, counts, strict=True)
        for _ in range(count)
        for value in types
    )
    component_ids = tuple(member.component_id for member in members)
    return AtomTypeAssignment(
        component_type_ids=tuple(component_types),
        type_ids=type_ids,
        representatives=tuple(representatives),
        formula_counts=counts,
        component_ids=component_ids,
        mode=mode_,
        max_nodes=budget,
        assignment_id=canonical_fingerprint(
            {
                "kind": "crystal-atom-type-assignment",
                "components": list(component_ids),
                "formula_counts": list(counts),
                "mode": mode_,
                "types": [list(types) for types in component_types],
            }
        ),
    )


def map_connectivity_types(
    assignment: AtomTypeAssignment,
    components: Sequence[MolecularComponent],
    atomic_numbers: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    /,
) -> tuple[int, ...]:
    """Map every connected component of a target graph onto reference types.

    The pinned Toolkit ``TopologicalAtomTypeMap.for_connectivity`` contract:
    connectivity mode only; each connected target component must match one
    reference component by an element- and bond-preserving bijection; target
    atom order is preserved.  Component proportions are not checked.  Raises
    `ValueError` for an unmatched component or an undetermined search.
    """
    if assignment.mode != "connectivity":
        raise ValueError("Raw-graph type mapping is defined for connectivity mode only.")
    references = tuple(components)
    if tuple(member.component_id for member in references) != assignment.component_ids:
        raise ValueError("components must be the assignment's reference components.")
    numbers = tuple(int(value) for value in atomic_numbers)
    count = len(numbers)
    neighbors: list[dict[int, int]] = [{} for _ in range(count)]
    for left, right in bonds:
        if not (0 <= left < count and 0 <= right < count) or left == right:
            raise ValueError("Target bonds must reference distinct target atoms.")
        neighbors[left][right] = neighbors[right][left] = 1
    reference_graphs = tuple(
        _component_graph(member, "connectivity") for member in references
    )
    result = [-1] * count
    seen = [False] * count
    for start in range(count):
        if seen[start]:
            continue
        members = _connected(neighbors, start, seen)
        local = {atom: position for position, atom in enumerate(members)}
        graph = LabeledGraph(
            tuple((numbers[atom],) for atom in members),
            tuple({local[n]: 1 for n in neighbors[atom]} for atom in members),
            {},
            {},
        )
        for reference_index, reference in enumerate(reference_graphs):
            outcome = _correspond(
                graph,
                reference,
                fixed=None,
                mirror=False,
                max_mappings=1,
                max_nodes=assignment.max_nodes,
            )
            if outcome.mappings:
                mapping = outcome.mappings[0]
                for position, atom in enumerate(members):
                    result[atom] = assignment.component_type_ids[reference_index][
                        mapping[position]
                    ]
                break
            if outcome.status == "node-limit":
                raise ValueError(
                    "Target component correspondence is undetermined within max_nodes."
                )
        else:
            raise ValueError(
                "Target graph contains a component with no isomorphic reference component."
            )
    return tuple(result)


def _connected(
    neighbors: Sequence[dict[int, int]], start: int, seen: list[bool], /
) -> list[int]:
    seen[start] = True
    members = [start]
    for atom in members:
        for neighbor in sorted(neighbors[atom]):
            if not seen[neighbor]:
                seen[neighbor] = True
                members.append(neighbor)
    return sorted(members)


__all__ = [
    "AtomTypeAssignment",
    "AtomTypeMode",
    "CorrespondenceResult",
    "CorrespondenceStatus",
    "atom_type_assignment",
    "component_atom_orbits",
    "map_connectivity_types",
    "molecular_correspondences",
]
