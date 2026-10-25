#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Donor/acceptor contact rules adapted from NVIDIA nvalchemi-toolkit:
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Apache-2.0; see LICENSES/NVALCHEMI-TOOLKIT-APACHE-2.0.txt and NOTICE.
# Modifications: explicit native molecular graphs replace RDKit objects.

"""Explicit molecular crystal input: chemistry, conformer pools, contacts and volume.

A `MolecularComponent` binds one connected molecule's atom/bond/stereo
identity, explicit hydrogens, formal charges, spin multiplicity and masses.
Chemical meaning is never inferred: hydrogens are explicit, bond orders are
declared (``1``, ``2``, ``3`` or ``4`` for aromatic), undeclared topological
stereo candidates are refused unless the caller explicitly declares them
unspecified, and every element table (masses, contact radii, atomic volumes,
covalent radii) is caller-admitted data with provenance.  Unknown elements
refuse instead of silently falling back to a provider table.

A `MolecularConformer` stores one admitted 3D geometry in the component's
mass-centered principal-axes local frame (a proper rotation of the input).
Input geometries must realize every declared stereo element.

A `MolecularCrystalInput` binds the formula unit: conformer pools, integer
component counts, a contact-distance matrix over formula-unit atoms and the
formula-unit volume.  Everything here is host-only immutable preparation;
numerical consumers receive dynamic float64 leaves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    finite_real_scalar,
    integer,
    positive_finite_float,
    positive_integer,
)
from ...linalg import HermitianSpectrum
from ...typing import Dim, Float64, HostFloat64, parse
from ._atom_types import (
    atom_type_assignment,
    AtomTypeAssignment,
    AtomTypeMode,
    bond_ring_size,
    DEFAULT_MAXIMUM_AUTOMORPHISMS,
    DEFAULT_MAXIMUM_SEARCH_NODES,
    graph_symmetry,
    labeled_graph,
    stereo_candidates,
)


ElementQuantity: TypeAlias = Literal[
    "mass",
    "covalent-radius",
    "double-bond-radius",
    "triple-bond-radius",
    "contact-radius",
    "atomic-volume",
]
"""Admitted per-element quantity; units are fixed per quantity (see `ELEMENT_UNITS`)."""

ValencePolicy: TypeAlias = Literal["standard", "unchecked"]
"""``"standard"`` checks explicit-hydrogen valences and resolves a Kekulé
structure for aromatic bonds; ``"unchecked"`` records that no valence model was
applied (aromatic bonds are then refused)."""

UndeclaredStereoPolicy: TypeAlias = Literal["refuse", "unspecified"]
ConformerSource: TypeAlias = Literal["input", "native-distance-geometry"]
ContactRule: TypeAlias = Literal["radius-sum", "donor-acceptor-reduced"]
"""``"radius-sum"`` uses ``r_i + r_j``; ``"donor-acceptor-reduced"`` subtracts the
pinned Toolkit topology rule's donor/acceptor reduction (capped)."""

ELEMENT_UNITS: Mapping[str, str] = {
    "mass": "dalton",
    "covalent-radius": "angstrom",
    "double-bond-radius": "angstrom",
    "triple-bond-radius": "angstrom",
    "contact-radius": "angstrom",
    "atomic-volume": "angstrom^3",
}

AROMATIC_BOND = 4
_MAXIMUM_ATOMIC_NUMBER = 118


class MolecularAtomDim(Dim, minimum=1):
    """Atoms of one molecular component, in component order."""


class FormulaAtomDim(Dim, minimum=1):
    """Atoms of one formula unit, components repeated in declared order."""


@final
class ElementTable(StrictModule):
    """Caller-admitted per-element data with explicit unit and provenance.

    ``lookup`` refuses elements that are absent; there is no fallback table.
    """

    quantity: ElementQuantity = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    values: tuple[float, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantity: ElementQuantity,
        values: Mapping[int, float],
        /,
        *,
        provenance: str,
    ) -> None:
        quantity_ = parse(quantity, ElementQuantity, "quantity")
        if not isinstance(values, Mapping) or not values:
            raise TypeError("values must be a non-empty mapping of atomic numbers.")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Element data requires a non-empty provenance statement.")
        entries: list[tuple[int, float]] = []
        for key, value in values.items():
            number = integer(key, "atomic number")
            if not 1 <= number <= _MAXIMUM_ATOMIC_NUMBER:
                raise ValueError("Element table keys must be atomic numbers in [1, 118].")
            entries.append((number, positive_finite_float(value, f"value[{number}]")))
        entries.sort()
        self.quantity = quantity_
        self.unit = ELEMENT_UNITS[quantity_]
        self.atomic_numbers = tuple(number for number, _ in entries)
        self.values = tuple(value for _, value in entries)
        self.provenance = provenance.strip()
        self.table_id = canonical_fingerprint(
            {
                "kind": "crystal-element-table",
                "quantity": quantity_,
                "unit": self.unit,
                "entries": [[number, value] for number, value in entries],
                "provenance": self.provenance,
            }
        )

    def lookup(self, atomic_numbers: Sequence[int], /) -> np.ndarray:
        """Return float64 values per atom; refuse any undeclared element."""
        table = dict(zip(self.atomic_numbers, self.values, strict=True))
        missing = sorted({number for number in atomic_numbers if number not in table})
        if missing:
            raise ValueError(
                f"Element table {self.quantity!r} ({self.provenance}) has no data for "
                f"atomic numbers {missing}; declare them explicitly."
            )
        return np.asarray([table[number] for number in atomic_numbers], dtype=np.float64)


# Default valences of the explicit-hydrogen organic valence model.  The charge
# adjustment follows the isoelectronic convention: groups 15-17 gain one bond
# per positive charge, group 13 gains one per negative charge, and H and group
# 14 lose one per unit of either sign.
_BASE_VALENCES: Mapping[int, tuple[int, ...]] = {
    1: (1,),
    5: (3,),
    6: (4,),
    7: (3,),
    8: (2,),
    9: (1,),
    14: (4,),
    15: (3, 5),
    16: (2, 4, 6),
    17: (1,),
    33: (3, 5),
    34: (2, 4, 6),
    35: (1,),
    53: (1, 3, 5),
}
_CHARGE_RULE: Mapping[int, Literal["donor", "acceptor", "symmetric"]] = {
    1: "symmetric",
    5: "acceptor",
    6: "symmetric",
    14: "symmetric",
    7: "donor",
    15: "donor",
    33: "donor",
    8: "donor",
    16: "donor",
    34: "donor",
    9: "donor",
    17: "donor",
    35: "donor",
    53: "donor",
}


def _allowed_valences(number: int, charge: int, /) -> frozenset[int]:
    base = _BASE_VALENCES.get(number)
    if base is None:
        raise ValueError(
            f"The standard valence model does not cover atomic number {number}; "
            "use valence='unchecked' to record that no valence model applies."
        )
    match _CHARGE_RULE[number]:
        case "donor":
            shifted = tuple(value + charge for value in base)
        case "acceptor":
            shifted = tuple(value - charge for value in base)
        case "symmetric":
            shifted = tuple(value - abs(charge) for value in base)
        case _ as unreachable:
            assert_never(unreachable)
    return frozenset(value for value in shifted if value >= 0)


def _kekule_orders(
    atomic_numbers: Sequence[int],
    formal_charges: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    orders: Sequence[int],
    /,
    *,
    max_nodes: int,
) -> tuple[int, ...]:
    """Check standard valences and resolve aromatic bonds to one Kekulé structure.

    Each aromatic atom receives one single bond per aromatic bond plus at most
    one extra double bond; atoms whose valence requires the extra bond are
    perfectly matched over aromatic bonds by a bounded deterministic search.
    """
    count = len(atomic_numbers)
    fixed = [0] * count
    aromatic_bonds: list[list[int]] = [[] for _ in range(count)]
    for index, ((left, right), order) in enumerate(zip(bonds, orders, strict=True)):
        if order == AROMATIC_BOND:
            aromatic_bonds[left].append(index)
            aromatic_bonds[right].append(index)
            fixed[left] += 1
            fixed[right] += 1
        else:
            fixed[left] += order
            fixed[right] += order
    required: list[int] = []
    flexible: set[int] = set()
    for atom in range(count):
        allowed = _allowed_valences(atomic_numbers[atom], formal_charges[atom])
        if not aromatic_bonds[atom]:
            if fixed[atom] not in allowed:
                raise ValueError(
                    f"Atom {atom} (Z={atomic_numbers[atom]}, charge "
                    f"{formal_charges[atom]}) has valence {fixed[atom]}, not one of "
                    f"{sorted(allowed)}; hydrogens and charges must be explicit."
                )
            continue
        without, with_extra = fixed[atom] in allowed, fixed[atom] + 1 in allowed
        if not without and not with_extra:
            raise ValueError(
                f"Aromatic atom {atom} (Z={atomic_numbers[atom]}) admits no Kekulé "
                "valence; hydrogens and charges must be explicit."
            )
        if with_extra and not without:
            required.append(atom)
        elif with_extra:
            flexible.add(atom)
    doubled: set[int] = set()
    matched = [False] * count
    nodes = 0

    def match(position: int) -> bool | None:
        nonlocal nodes
        if position == len(required):
            return True
        atom = required[position]
        if matched[atom]:
            return match(position + 1)
        for bond in aromatic_bonds[atom]:
            left, right = bonds[bond]
            partner = right if left == atom else left
            if matched[partner] or (partner not in flexible and partner not in required):
                continue
            nodes += 1
            if nodes > max_nodes:
                return None
            matched[atom] = matched[partner] = True
            doubled.add(bond)
            outcome = match(position + 1)
            if outcome is not False:
                return outcome
            matched[atom] = matched[partner] = False
            doubled.discard(bond)
        return False

    outcome = match(0)
    if outcome is None:
        raise ValueError("Kekulé resolution is undetermined within max_search_nodes.")
    if not outcome:
        raise ValueError(
            "Aromatic bonds admit no Kekulé structure; hydrogens or charges are "
            "missing or the aromatic system is inconsistent."
        )
    return tuple(
        (2 if index in doubled else 1) if order == AROMATIC_BOND else order
        for index, order in enumerate(orders)
    )


def _canonical_tetrahedral(
    spec: Sequence[int],
    neighbors: Sequence[dict[int, int]],
    /,
) -> tuple[int, int, int, int, int]:
    values = tuple(integer(value, "tetrahedral_stereo") for value in spec)
    if len(values) != 5:
        raise ValueError("Tetrahedral stereo entries are (center, n1, n2, n3, sign).")
    center, *declared, sign = values
    if not 0 <= center < len(neighbors):
        raise ValueError("Tetrahedral stereo center is outside the component.")
    adjacent = sorted(neighbors[center])
    if len(adjacent) not in (3, 4):
        raise ValueError(
            f"Tetrahedral stereo center {center} must have 3 or 4 neighbors."
        )
    if any(order != 1 for order in neighbors[center].values()):
        raise ValueError(
            f"Tetrahedral stereo at {center} with a multiple bond is unsupported "
            "chemical meaning (hypervalent centers are not admitted)."
        )
    if len(set(declared)) != 3 or any(atom not in neighbors[center] for atom in declared):
        raise ValueError(
            f"Tetrahedral stereo at {center} must list 3 distinct neighbors."
        )
    if sign not in (-1, 1):
        raise ValueError("Tetrahedral stereo sign must be +1 or -1.")
    ordered = list(declared) + [atom for atom in adjacent if atom not in declared]
    inversions = sum(
        1
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
        if ordered[left] > ordered[right]
    )
    canonical = -sign if inversions % 2 else sign
    return (center, adjacent[0], adjacent[1], adjacent[2], canonical)


def _canonical_double_bond(
    spec: Sequence[int],
    neighbors: Sequence[dict[int, int]],
    /,
) -> tuple[int, int, int, int, int]:
    values = tuple(integer(value, "double_bond_stereo") for value in spec)
    if len(values) != 5:
        raise ValueError("Double-bond stereo entries are (a, b, c, d, sign).")
    a, b, c, d, sign = values
    count = len(neighbors)
    if not all(0 <= atom < count for atom in (a, b, c, d)):
        raise ValueError("Double-bond stereo atoms are outside the component.")
    if neighbors[b].get(c) != 2:
        raise ValueError(f"Double-bond stereo requires an order-2 bond {b}-{c}.")
    if a == c or d == b or a not in neighbors[b] or d not in neighbors[c]:
        raise ValueError("Double-bond stereo substituents must be bonded to the ends.")
    for end, other in ((b, c), (c, b)):
        if any(order != 1 for atom, order in neighbors[end].items() if atom != other):
            raise ValueError(
                f"Double-bond stereo end {end} carries another multiple bond; "
                "cumulated systems are unsupported chemical meaning."
            )
    if sign not in (-1, 1):
        raise ValueError("Double-bond stereo sign must be +1 (cis) or -1 (trans).")
    if b > c:
        a, b, c, d = d, c, b, a
    first = min(atom for atom in neighbors[b] if atom != c)
    last = min(atom for atom in neighbors[c] if atom != b)
    if a != first:
        sign = -sign
    if d != last:
        sign = -sign
    return (first, b, c, last, sign)


class _Connectivity(NamedTuple):
    bonds: tuple[tuple[int, int], ...]
    orders: tuple[int, ...]
    neighbors: tuple[dict[int, int], ...]


def _connectivity(
    count: int, bonds: Sequence[Sequence[int]], bond_orders: Sequence[int], /
) -> _Connectivity:
    pairs = [tuple(integer(value, "bonds") for value in bond) for bond in bonds]
    orders = [integer(value, "bond_orders") for value in bond_orders]
    if len(pairs) != len(orders):
        raise ValueError("bond_orders must provide one order per bond.")
    entries: dict[tuple[int, int], int] = {}
    for pair, order in zip(pairs, orders, strict=True):
        if len(pair) != 2:
            raise ValueError("Bonds are atom pairs.")
        left, right = sorted(pair)
        if left == right or left < 0 or right >= count:
            raise ValueError("Bonds must join two distinct component atoms.")
        if order not in (1, 2, 3, AROMATIC_BOND):
            raise ValueError("Bond orders are 1, 2, 3 or 4 (aromatic).")
        if (left, right) in entries:
            raise ValueError(f"Bond {left}-{right} is declared twice.")
        entries[(left, right)] = order
    ordered = sorted(entries)
    neighbors: list[dict[int, int]] = [{} for _ in range(count)]
    for left, right in ordered:
        neighbors[left][right] = neighbors[right][left] = entries[(left, right)]
    seen = {0}
    frontier = [0]
    while frontier:
        atom = frontier.pop()
        for neighbor in neighbors[atom]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    if len(seen) != count:
        raise ValueError(
            "A molecular component must be one connected molecule; declare "
            "disconnected species as separate components."
        )
    for left, right in ordered:
        if entries[(left, right)] == AROMATIC_BOND and (
            bond_ring_size(neighbors, left, right, limit=count) is None
        ):
            raise ValueError(f"Aromatic bond {left}-{right} does not lie on a ring.")
    return _Connectivity(
        tuple(ordered), tuple(entries[pair] for pair in ordered), tuple(neighbors)
    )


@final
class MolecularComponent(StrictModule):
    """One connected molecule's explicit chemical identity.

    Atom order is the indexing contract of every conformer, correspondence and
    contact row.  ``component_id`` fingerprints the chemistry (elements,
    charges, bonds, orders, declared and unspecified stereo, spin, masses and
    valence policy); ``name`` is display-only.  ``mass_table`` retains the
    admitted mass `ElementTable` (``None`` for per-atom masses) as a host
    restoration witness and ``mass_provenance`` is its identity.
    ``atom_classes`` are exact
    stereo-preserving automorphism orbits; ``automorphisms`` lists up to
    ``max_automorphisms`` stereo-preserving permutations (identity first;
    ``p[i]`` is the image of atom ``i``) and ``automorphisms_complete`` states
    whether the list is the whole group.  ``stereo_chiral`` is true when the
    mirror image of the declared tetrahedral stereo is not related to it by
    any automorphism; ``mirror_automorphism`` is then ``None``, otherwise a
    permutation carrying the mirror image onto the declared stereo (identity
    without tetrahedral stereo).
    """

    name: str = eqx.field(static=True)
    atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    formal_charges: tuple[int, ...] = eqx.field(static=True)
    bonds: tuple[tuple[int, int], ...] = eqx.field(static=True)
    bond_orders: tuple[int, ...] = eqx.field(static=True)
    kekule_bond_orders: tuple[int, ...] = eqx.field(static=True)
    tetrahedral_stereo: tuple[tuple[int, int, int, int, int], ...] = eqx.field(
        static=True
    )
    double_bond_stereo: tuple[tuple[int, int, int, int, int], ...] = eqx.field(
        static=True
    )
    unspecified_tetrahedral: tuple[int, ...] = eqx.field(static=True)
    unspecified_double_bonds: tuple[tuple[int, int], ...] = eqx.field(static=True)
    total_charge: int = eqx.field(static=True)
    spin_multiplicity: int = eqx.field(static=True)
    masses: tuple[float, ...] = eqx.field(static=True)
    mass_table: ElementTable | None
    mass_provenance: str = eqx.field(static=True)
    valence: ValencePolicy = eqx.field(static=True)
    atom_classes: tuple[int, ...] = eqx.field(static=True)
    automorphisms: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    automorphisms_complete: bool = eqx.field(static=True)
    stereo_chiral: bool = eqx.field(static=True)
    mirror_automorphism: tuple[int, ...] | None = eqx.field(static=True)
    max_automorphisms: int = eqx.field(static=True)
    max_search_nodes: int = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    # Chemistry, stereo and bounded automorphism admission stay in declared field
    # order: the first refusal and the canonical molecular identity are one contract.
    def __init__(
        self,
        atomic_numbers: Sequence[int],
        bonds: Sequence[Sequence[int]],
        /,
        *,
        bond_orders: Sequence[int],
        masses: ElementTable | Sequence[float],
        spin_multiplicity: int,
        formal_charges: Sequence[int] | None = None,
        tetrahedral_stereo: Sequence[Sequence[int]] = (),
        double_bond_stereo: Sequence[Sequence[int]] = (),
        undeclared_stereo: UndeclaredStereoPolicy = "refuse",
        valence: ValencePolicy = "standard",
        name: str = "molecule",
        max_automorphisms: int = DEFAULT_MAXIMUM_AUTOMORPHISMS,
        max_search_nodes: int = DEFAULT_MAXIMUM_SEARCH_NODES,
    ) -> None:
        numbers = tuple(integer(value, "atomic_numbers") for value in atomic_numbers)
        count = len(numbers)
        if not count or any(
            not 1 <= value <= _MAXIMUM_ATOMIC_NUMBER for value in numbers
        ):
            raise ValueError(
                "atomic_numbers must be non-empty atomic numbers in [1, 118]."
            )
        charges = (
            (0,) * count
            if formal_charges is None
            else tuple(integer(value, "formal_charges") for value in formal_charges)
        )
        if len(charges) != count:
            raise ValueError("formal_charges must provide one charge per atom.")
        policy = parse(undeclared_stereo, UndeclaredStereoPolicy, "undeclared_stereo")
        valence_ = parse(valence, ValencePolicy, "valence")
        multiplicity = positive_integer(spin_multiplicity, "spin_multiplicity")
        automorphism_limit = positive_integer(max_automorphisms, "max_automorphisms")
        budget = positive_integer(max_search_nodes, "max_search_nodes")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        connectivity = _connectivity(count, bonds, bond_orders)
        total_charge = sum(charges)
        electrons = sum(numbers) - total_charge
        if (
            electrons < 0
            or multiplicity - 1 > electrons
            or ((electrons - (multiplicity - 1)) % 2)
        ):
            raise ValueError(
                f"Spin multiplicity {multiplicity} is inconsistent with {electrons} "
                "electrons."
            )
        match valence_:
            case "standard":
                kekule = _kekule_orders(
                    numbers,
                    charges,
                    connectivity.bonds,
                    connectivity.orders,
                    max_nodes=budget,
                )
            case "unchecked":
                if AROMATIC_BOND in connectivity.orders:
                    raise ValueError(
                        "Aromatic bonds require valence='standard' to resolve a "
                        "Kekulé structure."
                    )
                kekule = connectivity.orders
            case _ as unreachable:
                assert_never(unreachable)
        if isinstance(masses, ElementTable):
            if masses.quantity != "mass":
                raise ValueError("masses must be an ElementTable of quantity 'mass'.")
            mass_values = tuple(float(value) for value in masses.lookup(numbers))
            mass_table: ElementTable | None = masses
            mass_provenance = masses.table_id
        else:
            mass_values = tuple(
                positive_finite_float(value, "masses") for value in masses
            )
            if len(mass_values) != count:
                raise ValueError("masses must provide one mass per atom.")
            mass_table = None
            mass_provenance = "caller-declared-per-atom"
        tetrahedral = tuple(
            sorted(
                _canonical_tetrahedral(spec, connectivity.neighbors)
                for spec in tetrahedral_stereo
            )
        )
        double = tuple(
            sorted(
                _canonical_double_bond(spec, connectivity.neighbors)
                for spec in double_bond_stereo
            )
        )
        if len({entry[0] for entry in tetrahedral}) != len(tetrahedral) or len(
            {entry[1:3] for entry in double}
        ) != len(double):
            raise ValueError("Each stereo element may be declared once.")
        chemical = labeled_graph(
            numbers, charges, connectivity.bonds, connectivity.orders, (), (), "chemical"
        )
        chemical_orbits = graph_symmetry(
            chemical, max_automorphisms=automorphism_limit, max_nodes=budget
        ).classes
        candidates = stereo_candidates(chemical, chemical_orbits)
        declared_centers = {entry[0] for entry in tetrahedral}
        declared_bonds = {entry[1:3] for entry in double}
        missing_centers = tuple(
            atom for atom in candidates.tetrahedral if atom not in declared_centers
        )
        missing_bonds = tuple(
            bond for bond in candidates.double_bonds if bond not in declared_bonds
        )
        if policy == "refuse" and (missing_centers or missing_bonds):
            raise ValueError(
                f"Undeclared stereo candidates: tetrahedral centers {list(missing_centers)}, "
                f"double bonds {[list(bond) for bond in missing_bonds]}; declare them or "
                "use undeclared_stereo='unspecified'."
            )
        stereo_graph = labeled_graph(
            numbers,
            charges,
            connectivity.bonds,
            connectivity.orders,
            tetrahedral,
            double,
            "stereo",
        )
        symmetry = graph_symmetry(
            stereo_graph, max_automorphisms=automorphism_limit, max_nodes=budget
        )
        self.name = name.strip()
        self.atomic_numbers = numbers
        self.formal_charges = charges
        self.bonds = connectivity.bonds
        self.bond_orders = connectivity.orders
        self.kekule_bond_orders = kekule
        self.tetrahedral_stereo = tetrahedral
        self.double_bond_stereo = double
        self.unspecified_tetrahedral = missing_centers
        self.unspecified_double_bonds = missing_bonds
        self.total_charge = total_charge
        self.spin_multiplicity = multiplicity
        self.masses = mass_values
        self.mass_table = mass_table
        self.mass_provenance = mass_provenance
        self.valence = valence_
        self.atom_classes = symmetry.classes
        self.automorphisms = symmetry.automorphisms
        self.automorphisms_complete = symmetry.automorphisms_complete
        self.stereo_chiral = symmetry.stereo_chiral
        self.mirror_automorphism = symmetry.mirror_automorphism
        self.max_automorphisms = automorphism_limit
        self.max_search_nodes = budget
        self.component_id = canonical_fingerprint(
            {
                "kind": "crystal-molecular-component",
                "atomic_numbers": list(numbers),
                "formal_charges": list(charges),
                "bonds": [list(bond) for bond in connectivity.bonds],
                "bond_orders": list(connectivity.orders),
                "tetrahedral_stereo": [list(entry) for entry in tetrahedral],
                "double_bond_stereo": [list(entry) for entry in double],
                "unspecified_tetrahedral": list(missing_centers),
                "unspecified_double_bonds": [list(bond) for bond in missing_bonds],
                "spin_multiplicity": multiplicity,
                "masses": list(mass_values),
                "valence": valence_,
            }
        )

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)

    def neighbor_orders(self) -> tuple[dict[int, int], ...]:
        """Per-atom neighbor-to-declared-bond-order maps."""
        neighbors: list[dict[int, int]] = [{} for _ in self.atomic_numbers]
        for (left, right), order in zip(self.bonds, self.bond_orders, strict=True):
            neighbors[left][right] = neighbors[right][left] = order
        return tuple(neighbors)

    def mass_source(self) -> ElementTable | tuple[float, ...]:
        """The admitted mass input: the retained table or the per-atom masses."""
        return self.masses if self.mass_table is None else self.mass_table

    def enantiomer(self, *, name: str | None = None) -> MolecularComponent:
        """Return the mirror-image component (every tetrahedral parity inverted).

        Refuses achiral components, whose mirror image is the same compound.
        """
        if not self.stereo_chiral:
            raise ValueError("An achiral component has no distinct enantiomer.")
        return MolecularComponent(
            self.atomic_numbers,
            self.bonds,
            bond_orders=self.bond_orders,
            masses=self.mass_source(),
            spin_multiplicity=self.spin_multiplicity,
            formal_charges=self.formal_charges,
            tetrahedral_stereo=tuple(
                (center, a, b, c, -sign)
                for center, a, b, c, sign in self.tetrahedral_stereo
            ),
            double_bond_stereo=self.double_bond_stereo,
            undeclared_stereo=(
                "unspecified"
                if self.unspecified_tetrahedral or self.unspecified_double_bonds
                else "refuse"
            ),
            valence=self.valence,
            name=self.name if name is None else name,
            max_automorphisms=self.max_automorphisms,
            max_search_nodes=self.max_search_nodes,
        )


class StereoRealization(NamedTuple):
    """Signed stereo evidence of one geometry.

    ``tetrahedral_volumes[k]`` is ``sign_k * det`` for declared entry ``k``
    (positive when realized); ``double_bond_cosines[k]`` is ``sign_k *
    cos(phi_abcd)`` (positive when realized).
    """

    tetrahedral_volumes: np.ndarray
    double_bond_cosines: np.ndarray


def stereo_realization(
    component: MolecularComponent, positions: np.ndarray, /
) -> StereoRealization:
    """Evaluate declared stereo elements of host float64 ``positions``."""
    volumes = np.asarray(
        [
            sign
            * np.linalg.det(
                np.stack(
                    (
                        positions[a] - positions[center],
                        positions[b] - positions[center],
                        positions[c] - positions[center],
                    )
                )
            )
            for center, a, b, c, sign in component.tetrahedral_stereo
        ],
        dtype=np.float64,
    )
    cosines = np.asarray(
        [
            sign
            * _dihedral_cosine(positions[a], positions[b], positions[c], positions[d])
            for a, b, c, d, sign in component.double_bond_stereo
        ],
        dtype=np.float64,
    )
    return StereoRealization(volumes, cosines)


def _dihedral_cosine(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, /
) -> float:
    axis = c - b
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return 0.0
    unit = axis / norm
    first = (a - b) - np.dot(a - b, unit) * unit
    last = (d - c) - np.dot(d - c, unit) * unit
    scale = np.linalg.norm(first) * np.linalg.norm(last)
    return 0.0 if scale == 0.0 else float(np.dot(first, last) / scale)


@final
class MolecularConformer(StrictModule, NonTrainableState):
    """One admitted geometry in the component's local frame.

    ``positions`` (angstrom) are mass-centered and expressed in the principal
    axes of the inertia tensor, ascending moments, with a proper (det +1)
    rotation; the first two axes are oriented so that the mass-weighted third
    moment along them is nonnegative.  ``frame_degenerate`` reports a relative
    principal-moment gap below ``1e-8``: the frame is then one admissible
    choice, not unique.  Admission requires finite coordinates, every pair
    farther apart than ``minimum_distance`` and every declared stereo element
    realized with signed volume at least ``stereo_volume`` (angstrom^3) and
    signed double-bond cosine at least ``double_bond_cosine``.

    ``admitted_positions`` retains the host float64 copy of the constructor
    coordinates read-only as a host restoration witness, so a persisted recipe
    replays this constructor in the same numerical profile; ``conformer_id``
    remains the identity of the local geometry, source and provenance.
    """

    __strict_contract__ = True

    positions: Float64[MolecularAtomDim, Literal[3]]
    principal_moments: Float64[Literal[3]]
    admitted_positions: HostFloat64[MolecularAtomDim, Literal[3]]
    component_id: str = eqx.field(static=True)
    conformer_id: str = eqx.field(static=True)
    source: ConformerSource = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    frame_degenerate: bool = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    stereo_volume: float = eqx.field(static=True)
    double_bond_cosine: float = eqx.field(static=True)

    def __init__(
        self,
        component: MolecularComponent,
        positions: ArrayLike,
        /,
        *,
        provenance: str,
        source: ConformerSource = "input",
        minimum_distance: float = 0.5,
        stereo_volume: float = 0.1,
        double_bond_cosine: float = 0.5,
    ) -> None:
        if not isinstance(component, MolecularComponent):
            raise TypeError("component must be a MolecularComponent.")
        source_ = parse(source, ConformerSource, "source")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Conformers require a non-empty provenance statement.")
        threshold = positive_finite_float(minimum_distance, "minimum_distance")
        volume_threshold = positive_finite_float(stereo_volume, "stereo_volume")
        cosine_threshold = finite_real_scalar(double_bond_cosine, "double_bond_cosine")
        if not 0.0 <= cosine_threshold < 1.0:
            raise ValueError("double_bond_cosine must lie in [0, 1).")
        host = np.array(positions, dtype=np.float64)
        if host.shape != (component.atom_count, 3):
            raise ValueError("positions must have shape (atom_count, 3).")
        if not np.all(np.isfinite(host)):
            raise ValueError("Conformer positions must be finite.")
        if component.atom_count > 1:
            differences = host[:, None, :] - host[None, :, :]
            distances = np.sqrt(np.sum(differences**2, axis=-1))
            upper = distances[np.triu_indices(component.atom_count, 1)]
            if np.min(upper) <= threshold:
                raise ValueError(
                    f"Conformer atoms are closer than minimum_distance={threshold} Å."
                )
        realized = stereo_realization(component, host)
        if np.any(realized.tetrahedral_volumes < volume_threshold):
            failed = [
                component.tetrahedral_stereo[index][0]
                for index in np.flatnonzero(
                    realized.tetrahedral_volumes < volume_threshold
                )
            ]
            raise ValueError(
                f"Conformer does not realize the declared tetrahedral stereo at {failed}."
            )
        if np.any(realized.double_bond_cosines < cosine_threshold):
            failed_bonds = [
                list(component.double_bond_stereo[index][1:3])
                for index in np.flatnonzero(
                    realized.double_bond_cosines < cosine_threshold
                )
            ]
            raise ValueError(
                f"Conformer does not realize the declared double-bond stereo at {failed_bonds}."
            )
        local, moments, degenerate = _principal_frame(
            host, np.asarray(component.masses, dtype=np.float64)
        )
        host.setflags(write=False)
        self.positions = jnp.asarray(local)
        self.principal_moments = jnp.asarray(moments)
        self.admitted_positions = host
        self.component_id = component.component_id
        self.source = source_
        self.provenance = provenance.strip()
        self.frame_degenerate = degenerate
        self.minimum_distance = threshold
        self.stereo_volume = volume_threshold
        self.double_bond_cosine = cosine_threshold
        self.conformer_id = canonical_fingerprint(
            {
                "kind": "crystal-molecular-conformer",
                "component": component.component_id,
                "positions": array_tree_fingerprint(local),
                "source": source_,
                "provenance": self.provenance,
            }
        )


_FRAME_DEGENERACY = 1.0e-8


def _principal_frame(
    positions: np.ndarray, masses: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Mass-centered principal-axes coordinates through the native spectrum."""
    center = masses @ positions / np.sum(masses)
    relative = positions - center
    squared = np.sum(relative**2, axis=-1)
    inertia = np.eye(3) * np.sum(masses * squared) - (relative.T * masses) @ relative
    spectrum = HermitianSpectrum(jnp.asarray(inertia), tolerance=1.0e-9)
    moments = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    axes = np.array(spectrum.eigenvectors, dtype=np.float64)
    scale = max(float(np.max(np.abs(moments))), np.finfo(np.float64).tiny)
    degenerate = bool(np.min(np.diff(moments)) <= _FRAME_DEGENERACY * scale)
    for column in range(2):
        projection = relative @ axes[:, column]
        skew = float(masses @ projection**3)
        if abs(skew) <= 1.0e-12 * max(scale, 1.0):
            nonzero = np.flatnonzero(np.abs(projection) > 1.0e-9)
            skew = float(projection[nonzero[0]]) if nonzero.size else 1.0
        if skew < 0.0:
            axes[:, column] = -axes[:, column]
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return relative @ axes, moments, degenerate


@final
class ConformerPool(StrictModule, NonTrainableState):
    """Admitted conformers of one component with explicit selection weights.

    ``declared_weights`` are the admitted positive weights (all one when none
    are declared), a host restoration witness; ``weights`` are their
    normalization and ``pool_id`` binds the normalized law.
    """

    component: MolecularComponent
    conformers: tuple[MolecularConformer, ...]
    declared_weights: tuple[float, ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    pool_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: MolecularComponent,
        conformers: Sequence[MolecularConformer],
        /,
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(component, MolecularComponent):
            raise TypeError("component must be a MolecularComponent.")
        members = tuple(conformers)
        if not members or any(
            not isinstance(item, MolecularConformer) for item in members
        ):
            raise TypeError(
                "conformers must be a non-empty sequence of MolecularConformer."
            )
        if any(item.component_id != component.component_id for item in members):
            raise ValueError("Every pooled conformer must belong to the component.")
        if len({item.conformer_id for item in members}) != len(members):
            raise ValueError("Pooled conformers must be distinct.")
        raw = (
            (1.0,) * len(members)
            if weights is None
            else tuple(positive_finite_float(value, "weights") for value in weights)
        )
        if len(raw) != len(members):
            raise ValueError("weights must provide one weight per conformer.")
        total = sum(raw)
        normalized = tuple(value / total for value in raw)
        self.component = component
        self.conformers = members
        self.declared_weights = raw
        self.weights = normalized
        self.pool_id = canonical_fingerprint(
            {
                "kind": "crystal-conformer-pool",
                "component": component.component_id,
                "conformers": [item.conformer_id for item in members],
                "weights": list(normalized),
            }
        )


class _ContactAtom(NamedTuple):
    number: int
    charge: int
    aromatic: bool
    hybridization: Literal["sp", "sp2", "sp3"]
    total_valence: int
    hydrogens: int
    neighbors: tuple[int, ...]
    double_neighbors: tuple[int, ...]
    triple_neighbors: tuple[int, ...]


def _contact_atoms(component: MolecularComponent, /) -> tuple[_ContactAtom, ...]:
    count = component.atom_count
    declared: list[dict[int, int]] = [{} for _ in range(count)]
    kekule: list[dict[int, int]] = [{} for _ in range(count)]
    for (left, right), order, resolved in zip(
        component.bonds, component.bond_orders, component.kekule_bond_orders, strict=True
    ):
        declared[left][right] = declared[right][left] = order
        kekule[left][right] = kekule[right][left] = resolved
    pi_atoms = {
        atom
        for atom in range(count)
        if any(order != 1 for order in declared[atom].values())
    }
    atoms: list[_ContactAtom] = []
    for atom in range(count):
        number = component.atomic_numbers[atom]
        aromatic = AROMATIC_BOND in declared[atom].values()
        resolved_orders = tuple(kekule[atom].values())
        if 3 in resolved_orders or resolved_orders.count(2) >= 2:
            hybridization: Literal["sp", "sp2", "sp3"] = "sp"
        elif aromatic or 2 in resolved_orders:
            hybridization = "sp2"
        elif (
            number in (7, 8, 16)
            and len(declared[atom]) <= 3
            and any(neighbor in pi_atoms for neighbor in declared[atom])
        ):
            # Lone-pair atoms conjugated with an adjacent pi system.
            hybridization = "sp2"
        else:
            hybridization = "sp3"
        atoms.append(
            _ContactAtom(
                number=number,
                charge=component.formal_charges[atom],
                aromatic=aromatic,
                hybridization=hybridization,
                total_valence=sum(resolved_orders),
                hydrogens=sum(
                    1
                    for neighbor in declared[atom]
                    if component.atomic_numbers[neighbor] == 1
                ),
                neighbors=tuple(sorted(declared[atom])),
                double_neighbors=tuple(
                    sorted(n for n, order in declared[atom].items() if order == 2)
                ),
                triple_neighbors=tuple(
                    sorted(n for n, order in declared[atom].items() if order == 3)
                ),
            )
        )
    return tuple(atoms)


# Donor/acceptor contact reductions ported from the pinned NVIDIA ALCHEMI
# Toolkit ``nvalchemi/csp/_chem/_contacts.py`` (Apache-2.0).  Aromatic bonds are
# not double bonds in these rules, matching the source's RDKit bond types.
_HALOGENS = frozenset({9, 17, 35, 53})
_ELECTRON_WITHDRAWING = frozenset({5, 7, 8, 9, 15, 16, 17, 35, 53})
_MAXIMUM_CONTACT_REDUCTION = 0.90


def _has_double_neighbor(
    atoms: Sequence[_ContactAtom], atom: int, elements: frozenset[int], excluded: int, /
) -> bool:
    return any(
        atoms[n].number in elements and n != excluded
        for n in atoms[atom].double_neighbors
    )


def _cyano_carbon(atoms: Sequence[_ContactAtom], atom: int, /) -> bool:
    return atoms[atom].number == 6 and any(
        atoms[n].number == 7 for n in atoms[atom].triple_neighbors
    )


def _amide_like_nitrogen(atoms: Sequence[_ContactAtom], atom: int, /) -> bool:
    return atoms[atom].number == 7 and any(
        atoms[n].number == 6
        and _has_double_neighbor(atoms, n, frozenset({7, 8, 16}), atom)
        for n in atoms[atom].neighbors
    )


def _acceptor(atoms: Sequence[_ContactAtom], atom: int, /) -> float:
    view = atoms[atom]
    number, charge = view.number, view.charge
    if number in _HALOGENS and charge < 0:
        return 0.30
    if number == 8:
        if charge < 0:
            return 0.30
        if _has_double_neighbor(atoms, atom, frozenset({6, 7, 15, 16, 34}), -1):
            return 0.25
        if charge == 0 and view.total_valence == 2:
            return 0.15
        return 0.0
    nitro = (
        number == 7
        and charge > 0
        and sum(atoms[n].number == 8 for n in view.neighbors) >= 2
    )
    if (
        number == 7
        and charge <= 0
        and not (view.aromatic and view.hydrogens > 0)
        and not _amide_like_nitrogen(atoms, atom)
        and not nitro
    ):
        if view.aromatic or view.hybridization in ("sp", "sp2"):
            return 0.20
        if charge == 0 and view.hybridization == "sp3":
            return 0.10
    if (
        number in (16, 34)
        and charge == 0
        and view.total_valence == 2
        and not any(atoms[n].number == 8 for n in view.neighbors)
    ):
        return 0.10
    return 0.0


def _activated_parent(atoms: Sequence[_ContactAtom], parent: int, child: int, /) -> bool:
    view = atoms[parent]
    if view.aromatic or view.hybridization in ("sp", "sp2") or view.charge > 0:
        return True
    return any(
        n != child and (atoms[n].charge > 0 or atoms[n].number in _ELECTRON_WITHDRAWING)
        for n in view.neighbors
    )


# Keep the source-faithful chemical decision tree explicit. Its branch count
# represents distinct element/charge/valence rules, not a generic learned heuristic.
def _donor(atoms: Sequence[_ContactAtom], atom: int, /) -> float:
    view = atoms[atom]
    number, charge = view.number, view.charge
    degree = len(view.neighbors)
    if number == 1 and degree == 1:
        parent = view.neighbors[0]
        parent_number = atoms[parent].number
        if parent_number == 8:
            return 0.55
        if parent_number == 7:
            return 0.35
        if parent_number == 16:
            return 0.15
        if parent_number == 6 and _activated_parent(atoms, parent, atom):
            return 0.05
        return 0.0
    if number in (17, 35, 53) and charge == 0 and degree == 1:
        activated = _activated_parent(atoms, view.neighbors[0], atom)
        match number:
            case 17:
                return 0.10 if activated else 0.0
            case 35:
                return 0.20 if activated else 0.10
            case _:
                return 0.35 if activated else 0.25
    if number in (16, 34) and charge >= 0:
        activated = (
            charge > 0
            or view.total_valence > 2
            or any(
                atoms[n].charge > 0
                or atoms[n].number in (8, 9, 17, 35, 53)
                or _cyano_carbon(atoms, n)
                for n in view.neighbors
            )
        )
        if number == 16:
            return 0.10 if activated else 0.05
        return 0.15 if activated else 0.10
    if number == 15:
        if charge > 0:
            return 0.15
        if view.total_valence > 3 or any(
            atoms[n].number in (7, 8, 9, 17, 35, 53) or _cyano_carbon(atoms, n)
            for n in view.neighbors
        ):
            return 0.05
    if number == 14:
        if charge > 0 or view.total_valence > 4:
            return 0.15
        if any(atoms[n].number in (7, 8, 9, 17, 35, 53) for n in view.neighbors):
            return 0.05
    if number == 5 and charge in (0, 1) and degree == 3 and view.total_valence == 3:
        return 0.45
    return 0.0


@final
class ContactPolicy(StrictModule):
    """Admitted contact radii and pair rule for formula-unit contact distances.

    ``distances`` returns the symmetric matrix ``r_i + r_j - reduction_ij``
    (angstrom) over the given ordered molecules, covering every intra- and
    intermolecular pair; consumers decide which pairs are excluded.
    """

    radii: ElementTable
    rule: ContactRule = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, radii: ElementTable, /, *, rule: ContactRule = "radius-sum"
    ) -> None:
        if not isinstance(radii, ElementTable) or radii.quantity != "contact-radius":
            raise ValueError(
                "radii must be an ElementTable of quantity 'contact-radius'."
            )
        rule_ = parse(rule, ContactRule, "rule")
        self.radii = radii
        self.rule = rule_
        self.policy_id = canonical_fingerprint(
            {"kind": "crystal-contact-policy", "radii": radii.table_id, "rule": rule_}
        )

    def distances(self, molecules: Sequence[MolecularComponent], /) -> np.ndarray:
        numbers = [number for molecule in molecules for number in molecule.atomic_numbers]
        radius = self.radii.lookup(numbers)
        matrix = radius[:, None] + radius[None, :]
        match self.rule:
            case "radius-sum":
                return matrix
            case "donor-acceptor-reduced":
                donor: list[float] = []
                acceptor: list[float] = []
                for molecule in molecules:
                    atoms = _contact_atoms(molecule)
                    donor.extend(_donor(atoms, atom) for atom in range(len(atoms)))
                    acceptor.extend(_acceptor(atoms, atom) for atom in range(len(atoms)))
                donors = np.asarray(donor, dtype=np.float64)
                acceptors = np.asarray(acceptor, dtype=np.float64)
                forward = np.where(
                    (donors[:, None] > 0.0) & (acceptors[None, :] > 0.0),
                    donors[:, None] + acceptors[None, :],
                    0.0,
                )
                reduction = np.minimum(
                    np.maximum(forward, forward.T), _MAXIMUM_CONTACT_REDUCTION
                )
                return matrix - reduction
            case _ as unreachable:
                assert_never(unreachable)


@final
class VolumePolicy(StrictModule):
    """Formula-unit volume: an explicit value or a sum of admitted atomic volumes."""

    formula_unit_volume: float | None = eqx.field(static=True)
    atomic_volumes: ElementTable | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        formula_unit_volume: float | None = None,
        atomic_volumes: ElementTable | None = None,
    ) -> None:
        if (formula_unit_volume is None) == (atomic_volumes is None):
            raise ValueError(
                "Declare exactly one of formula_unit_volume and atomic_volumes."
            )
        volume = (
            None
            if formula_unit_volume is None
            else positive_finite_float(formula_unit_volume, "formula_unit_volume")
        )
        if atomic_volumes is not None and (
            not isinstance(atomic_volumes, ElementTable)
            or atomic_volumes.quantity != "atomic-volume"
        ):
            raise ValueError("atomic_volumes must be an 'atomic-volume' ElementTable.")
        self.formula_unit_volume = volume
        self.atomic_volumes = atomic_volumes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-volume-policy",
                "formula_unit_volume": volume,
                "atomic_volumes": None
                if atomic_volumes is None
                else atomic_volumes.table_id,
            }
        )

    def resolve(self, atomic_numbers: Sequence[int], /) -> float:
        """Formula-unit volume in angstrom^3 for the ordered formula-unit atoms."""
        if self.formula_unit_volume is not None:
            return self.formula_unit_volume
        if self.atomic_volumes is None:
            raise ValueError("VolumePolicy declares no volume source.")
        return float(np.sum(self.atomic_volumes.lookup(atomic_numbers)))


_CONTACT_SYMMETRY_TOLERANCE = 1.0e-3


@final
class MolecularCrystalInput(StrictModule, NonTrainableState):
    """Formula unit of molecular components with contact and volume policies.

    Formula-unit atoms list component ``c`` repeated ``formula_counts[c]``
    times in declared order; ``formula_atom_sources[k]`` is ``(component,
    copy, atom)`` for formula atom ``k``.  ``contact_distances`` (angstrom)
    covers every formula-atom pair; ``formula_unit_volume`` is angstrom^3.
    ``contact_policy`` (``None`` for a caller-declared matrix) and
    ``volume_policy`` retain the admitted policies, whose identities
    ``contact_provenance`` and ``volume_provenance`` already bind.
    """

    __strict_contract__ = True

    pools: tuple[ConformerPool, ...]
    contact_distances: Float64[FormulaAtomDim, FormulaAtomDim]
    contact_policy: ContactPolicy | None
    volume_policy: VolumePolicy
    formula_counts: tuple[int, ...] = eqx.field(static=True)
    formula_atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    formula_atom_sources: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    contact_provenance: str = eqx.field(static=True)
    formula_unit_volume: float = eqx.field(static=True)
    volume_provenance: str = eqx.field(static=True)
    total_charge: int = eqx.field(static=True)
    input_id: str = eqx.field(static=True)

    def __init__(
        self,
        pools: Sequence[ConformerPool],
        /,
        *,
        contact: ContactPolicy | ArrayLike,
        volume: VolumePolicy,
        formula_counts: Sequence[int] | None = None,
    ) -> None:
        members = tuple(pools)
        if not members or any(not isinstance(pool, ConformerPool) for pool in members):
            raise TypeError("pools must be a non-empty sequence of ConformerPool.")
        if len({pool.component.component_id for pool in members}) != len(members):
            raise ValueError("Each component may appear in one pool; use formula_counts.")
        counts = (
            (1,) * len(members)
            if formula_counts is None
            else tuple(
                positive_integer(value, "formula_counts") for value in formula_counts
            )
        )
        if len(counts) != len(members):
            raise ValueError("formula_counts must provide one count per pool.")
        if not isinstance(volume, VolumePolicy):
            raise TypeError("volume must be a VolumePolicy.")
        molecules = tuple(
            pool.component
            for pool, count in zip(members, counts, strict=True)
            for _ in range(count)
        )
        sources = tuple(
            (index, copy, atom)
            for index, (pool, count) in enumerate(zip(members, counts, strict=True))
            for copy in range(count)
            for atom in range(pool.component.atom_count)
        )
        numbers = tuple(
            number for molecule in molecules for number in molecule.atomic_numbers
        )
        if isinstance(contact, ContactPolicy):
            matrix = contact.distances(molecules)
            contact_provenance = contact.policy_id
            contact_policy: ContactPolicy | None = contact
        else:
            matrix = np.array(contact, dtype=np.float64)
            if matrix.shape != (len(numbers), len(numbers)):
                raise ValueError(
                    "Contact distances must cover every formula-unit atom pair."
                )
            if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
                raise ValueError("Contact distances must be finite and positive.")
            if not np.allclose(
                matrix,
                matrix.T,
                rtol=_CONTACT_SYMMETRY_TOLERANCE,
                atol=_CONTACT_SYMMETRY_TOLERANCE,
            ):
                raise ValueError("Contact distances must be symmetric within 1e-3.")
            matrix = 0.5 * (matrix + matrix.T)
            contact_provenance = "caller-declared-matrix:" + canonical_fingerprint(
                array_tree_fingerprint(matrix)
            )
            contact_policy = None
        formula_volume = volume.resolve(numbers)
        total_charge = sum(
            pool.component.total_charge * count
            for pool, count in zip(members, counts, strict=True)
        )
        self.pools = members
        self.contact_distances = jnp.asarray(matrix)
        self.contact_policy = contact_policy
        self.volume_policy = volume
        self.formula_counts = counts
        self.formula_atomic_numbers = numbers
        self.formula_atom_sources = sources
        self.contact_provenance = contact_provenance
        self.formula_unit_volume = formula_volume
        self.volume_provenance = volume.policy_id
        self.total_charge = total_charge
        self.input_id = canonical_fingerprint(
            {
                "kind": "crystal-molecular-input",
                "pools": [pool.pool_id for pool in members],
                "formula_counts": list(counts),
                "contacts": array_tree_fingerprint(matrix),
                "contact_provenance": contact_provenance,
                "formula_unit_volume": formula_volume,
                "volume_provenance": volume.policy_id,
            }
        )

    @property
    def formula_unit_atom_count(self) -> int:
        return len(self.formula_atomic_numbers)

    def atom_types(
        self,
        *,
        mode: AtomTypeMode = "stereo",
        max_nodes: int = DEFAULT_MAXIMUM_SEARCH_NODES,
    ) -> AtomTypeAssignment:
        """Exact shared atom types of this formula unit under ``mode``."""
        return atom_type_assignment(
            tuple(pool.component for pool in self.pools),
            self.formula_counts,
            mode=mode,
            max_nodes=max_nodes,
        )


__all__ = [
    "AROMATIC_BOND",
    "ConformerPool",
    "ConformerSource",
    "ContactPolicy",
    "ContactRule",
    "ElementQuantity",
    "ElementTable",
    "FormulaAtomDim",
    "MolecularAtomDim",
    "MolecularComponent",
    "MolecularConformer",
    "MolecularCrystalInput",
    "StereoRealization",
    "UndeclaredStereoPolicy",
    "ValencePolicy",
    "VolumePolicy",
    "stereo_realization",
]
