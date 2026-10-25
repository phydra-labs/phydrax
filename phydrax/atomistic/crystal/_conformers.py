#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native bounded distance-geometry conformer generation and qualification.

Workflow, per conformer ordinal and bounded attempt:

1. *Constraints* (host preparation, once per component): admitted covalent
   radii give bond lengths; local geometries (linear/trigonal/tetrahedral,
   ring-strain angles) give 1-3 bounds by the law of cosines; declared
   double-bond stereo, small-ring planarity and rotatable torsions give 1-4
   bounds; declared tetrahedral stereo gives signed chiral-volume bounds and
   trigonal centers/planar bonds give coplanarity constraints.  Native triangle
   bound smoothing (Dress-Havel, ``U_ij <= U_ik + U_kj`` and ``L_ij >= L_ik -
   U_kj``) propagates them; an inconsistency is reported with its pair, never
   repaired.  Scaled admitted nonbonded radii raise 1-5+ lower bounds only up
   to the smoothed upper bound, and the clipped count is evidence.
2. *Embedding*: distances are drawn uniformly within the smoothed bounds from
   a key addressed by (component, ordinal, attempt).  The centered metric
   matrix is decomposed by the native Hermitian spectrum and the top
   eigenpairs give a low-rank embedding.  It is labeled exact only when the
   discarded spectrum is numerically zero; an indefinite metric is never
   repaired, its negative spectral mass is evidence.
3. *Refinement*: native Levenberg-Marquardt minimizes the bound-violation,
   chiral-volume and planarity residuals, first in four dimensions with a
   penalized fourth coordinate, then in three.
4. *Qualification*: smoothed bounds, declared stereo, planarity and minimum
   separation are checked on the result; failures retain their reason.
5. *Optional energetic cleanup* by a caller-supplied native energy through the
   native minimizer, followed by topology/stereo requalification.

Every attempt key depends only on the root key, the component and the
(ordinal, attempt) address, so any ordinal range can be (re)generated
independently and resumed exactly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from math import acos, cos, degrees, radians, sin, sqrt
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    finite_real_scalar,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ...linalg import HermitianSpectrum
from ...optim import (
    AbstractMinimizationMethod,
    least_squares,
    LevenbergMarquardt,
    minimize,
    OptimizationStatus,
    OptimizationTermination,
)
from ...typing import Dim, Float64, Int32, parse, PRNGKey
from ._atom_types import angle_ring_size
from ._input import (
    AROMATIC_BOND,
    ConformerPool,
    ElementTable,
    MolecularAtomDim,
    MolecularComponent,
    MolecularConformer,
    stereo_realization,
)


LocalGeometry: TypeAlias = Literal["terminal", "linear", "trigonal", "tetrahedral"]
ConformerAttemptStatus: TypeAlias = Literal[
    "accepted",
    "embedding-rank-deficient",
    "embedding-nonfinite",
    "optimizer-failed",
    "bounds-violated",
    "stereo-violated",
    "planarity-violated",
    "minimum-distance",
    "cleanup-failed",
]
ConformerGenerationStatus: TypeAlias = Literal[
    "complete", "attempt-budget-exhausted", "bounds-inconsistent", "empty"
]
"""``"complete"``: every requested ordinal produced a qualified conformer.
``"attempt-budget-exhausted"``: some ordinals failed every attempt (their
attempts are retained).  ``"bounds-inconsistent"``: triangle smoothing proved
the constraints infeasible; no attempt ran.  ``"empty"``: zero ordinals were
requested."""

_TETRAHEDRAL_ANGLE = degrees(acos(-1.0 / 3.0))
# |det| of three unit vectors at the tetrahedral angle: sqrt(16 / 27).
_UNIT_TETRAHEDRAL_VOLUME = sqrt(16.0 / 27.0)
_STRAINED_ANGLE_TOLERANCE = 8.0
_MAXIMUM_ANGLE_RING = 6
_MAXIMUM_PLANAR_RING = 7
_CHIRAL_VOLUME_RANGE = (0.5, 1.5)
_TOLERATED_OPTIMIZER_STATUSES = frozenset(
    {
        int(OptimizationStatus.SUCCESS),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
        int(OptimizationStatus.STAGNATION),
    }
)
_ADDRESS_NAMESPACE = "phydrax.atomistic.crystal"
_ADMITTED_DEGREES: Mapping[LocalGeometry, frozenset[int]] = {
    "terminal": frozenset({0, 1}),
    "linear": frozenset({2}),
    "trigonal": frozenset({2, 3}),
    "tetrahedral": frozenset({2, 3, 4}),
}
# Exterior (non-ring) angles at strained tetrahedral ring atoms by smallest ring.
_STRAINED_EXTERIOR_ANGLE: Mapping[int, float] = {3: 118.0, 4: 114.0, 5: 111.0}


class ChiralCenterDim(Dim):
    """Declared tetrahedral stereo constraints."""


class PlanarityDim(Dim):
    """Coplanarity constraints of trigonal centers and planar bonds."""


@final
class ConformerGeometryTable(StrictModule):
    """Admitted geometric data for native distance-geometry bounds.

    Bond lengths are additive radii sums of the declared order: single bonds
    use ``single_radii``, double and triple bonds their own tables, aromatic
    bonds the mean of single and double sums.  Missing radii for a present
    bond order refuse.  ``nonbonded_scale * (r_i + r_j)`` of
    ``nonbonded_radii`` lower-bounds pairs four or more bonds apart.
    ``bond_tolerance`` (angstrom) and ``angle_tolerance`` (degrees) are bound
    half-widths.
    """

    single_radii: ElementTable
    double_radii: ElementTable | None
    triple_radii: ElementTable | None
    nonbonded_radii: ElementTable
    nonbonded_scale: float = eqx.field(static=True)
    bond_tolerance: float = eqx.field(static=True)
    angle_tolerance: float = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        single_radii: ElementTable,
        nonbonded_radii: ElementTable,
        /,
        *,
        double_radii: ElementTable | None = None,
        triple_radii: ElementTable | None = None,
        nonbonded_scale: float = 0.7,
        bond_tolerance: float = 0.01,
        angle_tolerance: float = 3.0,
    ) -> None:
        for table, quantity, name in (
            (single_radii, "covalent-radius", "single_radii"),
            (nonbonded_radii, "contact-radius", "nonbonded_radii"),
            (double_radii, "double-bond-radius", "double_radii"),
            (triple_radii, "triple-bond-radius", "triple_radii"),
        ):
            if table is None and name in ("double_radii", "triple_radii"):
                continue
            if not isinstance(table, ElementTable) or table.quantity != quantity:
                raise ValueError(
                    f"{name} must be an ElementTable of quantity {quantity!r}."
                )
        scale = positive_finite_float(nonbonded_scale, "nonbonded_scale")
        if scale > 1.0:
            raise ValueError("nonbonded_scale must not exceed one.")
        bond = positive_finite_float(bond_tolerance, "bond_tolerance")
        angle = positive_finite_float(angle_tolerance, "angle_tolerance")
        if angle >= 30.0:
            raise ValueError("angle_tolerance must be below 30 degrees.")
        self.single_radii = single_radii
        self.double_radii = double_radii
        self.triple_radii = triple_radii
        self.nonbonded_radii = nonbonded_radii
        self.nonbonded_scale = scale
        self.bond_tolerance = bond
        self.angle_tolerance = angle
        self.table_id = canonical_fingerprint(
            {
                "kind": "crystal-conformer-geometry-table",
                "single": single_radii.table_id,
                "double": None if double_radii is None else double_radii.table_id,
                "triple": None if triple_radii is None else triple_radii.table_id,
                "nonbonded": nonbonded_radii.table_id,
                "nonbonded_scale": scale,
                "bond_tolerance": bond,
                "angle_tolerance": angle,
            }
        )

    def bond_length(self, first: int, second: int, order: int, /) -> float:
        def radius_sum(table: ElementTable | None, name: str) -> float:
            if table is None:
                raise ValueError(f"A bond of order {order} requires {name}.")
            values = table.lookup((first, second))
            return float(values[0] + values[1])

        match order:
            case 1:
                return radius_sum(self.single_radii, "single_radii")
            case 2:
                return radius_sum(self.double_radii, "double_radii")
            case 3:
                return radius_sum(self.triple_radii, "triple_radii")
            case 4:
                return 0.5 * (
                    radius_sum(self.single_radii, "single_radii")
                    + radius_sum(self.double_radii, "double_radii")
                )
            case _:
                raise ValueError(f"Unsupported bond order {order}.")


def _classify(
    component: MolecularComponent,
    neighbors: Sequence[dict[int, int]],
    kekule: Sequence[dict[int, int]],
    overrides: Mapping[int, LocalGeometry],
    /,
) -> tuple[LocalGeometry, ...]:
    pi_atoms = {
        atom
        for atom in range(len(neighbors))
        if any(o != 1 for o in neighbors[atom].values())
    }
    result: list[LocalGeometry] = []
    for atom, adjacent in enumerate(neighbors):
        degree = len(adjacent)
        if atom in overrides:
            override = overrides[atom]
            if degree not in _ADMITTED_DEGREES[override]:
                raise ValueError(
                    f"Local geometry {override!r} does not admit atom {atom} with "
                    f"{degree} neighbors."
                )
            result.append(override)
            continue
        resolved = tuple(kekule[atom].values())
        if degree <= 1:
            geometry: LocalGeometry = "terminal"
        elif degree > 4:
            raise ValueError(
                f"Atom {atom} has {degree} neighbors; no local geometry above four "
                "neighbors is admitted."
            )
        elif degree == 2 and (3 in resolved or resolved.count(2) == 2):
            geometry = "linear"
        elif degree <= 3 and (
            2 in resolved or AROMATIC_BOND in adjacent.values() or 3 in resolved
        ):
            geometry = "trigonal"
        elif (
            degree == 3
            and component.atomic_numbers[atom] == 7
            and any(neighbor in pi_atoms for neighbor in adjacent)
        ):
            # Lone-pair nitrogen conjugated with an adjacent pi system (amides,
            # anilines, enamines) is planar.
            geometry = "trigonal"
        else:
            geometry = "tetrahedral"
        result.append(geometry)
    for center, *_ in component.tetrahedral_stereo:
        if result[center] != "tetrahedral":
            raise ValueError(
                f"Tetrahedral stereo at {center} conflicts with its {result[center]} geometry."
            )
    for _, b, c, _, _ in component.double_bond_stereo:
        if result[b] != "trigonal" or result[c] != "trigonal":
            raise ValueError(f"Double-bond stereo {b}={c} requires trigonal ends.")
    return tuple(result)


def _angle_target(geometry: LocalGeometry, ring: int | None, /) -> tuple[float, bool]:
    """Ideal angle (degrees) and whether it is ring-strained."""
    match geometry:
        case "linear":
            return 180.0, False
        case "trigonal":
            if ring is None or ring > _MAXIMUM_ANGLE_RING:
                return 120.0, False
            return 180.0 * (ring - 2) / ring, ring < 6
        case "tetrahedral":
            match ring:
                case 3:
                    return 60.0, True
                case 4:
                    return 88.0, True
                case 5:
                    return 104.0, True
                case _:
                    return _TETRAHEDRAL_ANGLE, False
        case "terminal":
            raise ValueError("Terminal atoms have no angle.")
        case _ as unreachable:
            assert_never(unreachable)


def _cosine_distance(first: float, second: float, angle: float, /) -> float:
    return sqrt(
        max(first**2 + second**2 - 2.0 * first * second * cos(radians(angle)), 0.0)
    )


def _torsion_distance(
    a: float, b: float, c: float, first_angle: float, second_angle: float, cis: bool, /
) -> float:
    theta1, theta2 = radians(first_angle), radians(second_angle)
    i = np.asarray((a * cos(theta1), a * sin(theta1), 0.0))
    sign = 1.0 if cis else -1.0
    l = np.asarray((b - c * cos(theta2), sign * c * sin(theta2), 0.0))
    return float(np.linalg.norm(i - l))


def _hop_distances(neighbors: Sequence[dict[int, int]], /) -> np.ndarray:
    count = len(neighbors)
    hops = np.full((count, count), -1, dtype=np.int64)
    for start in range(count):
        hops[start, start] = 0
        queue = deque((start,))
        while queue:
            atom = queue.popleft()
            for neighbor in neighbors[atom]:
                if hops[start, neighbor] < 0:
                    hops[start, neighbor] = hops[start, atom] + 1
                    queue.append(neighbor)
    return hops


def _smooth(lower: Array, upper: Array, /) -> tuple[Array, Array]:
    """Native Dress-Havel triangle bound smoothing over every pivot."""

    def pivot(k: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        low, high = state
        high = jnp.minimum(high, high[:, k, None] + high[None, k, :])
        low = jnp.maximum(
            low,
            jnp.maximum(
                low[:, k, None] - high[None, k, :], low[None, k, :] - high[:, k, None]
            ),
        )
        return low, high

    return lax.fori_loop(0, lower.shape[0], pivot, (lower, upper))


@final
class ConformerConstraints(StrictModule, NonTrainableState):
    """Smoothed distance bounds and stereo/planarity constraints of a component.

    ``lower``/``upper`` (angstrom) are triangle-smoothed; ``consistent`` is
    false when some ``lower_ij - upper_ij`` exceeds ``1e-6`` angstrom, with
    the worst pair and excess retained.  ``chiral_atoms[k]`` is ``(center, n1,
    n2, n3)`` with required signed volume ``chiral_signs[k] * det`` inside
    ``chiral_bounds[k]`` (angstrom^3).  ``planar_atoms[k]`` is ``(p, q, r, s)``
    whose volume ``det[q-p, r-p, s-p] / planar_scales[k]`` must vanish.
    """

    __strict_contract__ = True

    lower: Float64[MolecularAtomDim, MolecularAtomDim]
    upper: Float64[MolecularAtomDim, MolecularAtomDim]
    chiral_atoms: Int32[ChiralCenterDim, Literal[4]]
    chiral_signs: Float64[ChiralCenterDim]
    chiral_bounds: Float64[ChiralCenterDim, Literal[2]]
    planar_atoms: Int32[PlanarityDim, Literal[4]]
    planar_scales: Float64[PlanarityDim]
    bond_lengths: tuple[float, ...] = eqx.field(static=True)
    local_geometries: tuple[LocalGeometry, ...] = eqx.field(static=True)
    consistent: bool = eqx.field(static=True)
    maximum_inconsistency: float = eqx.field(static=True)
    inconsistent_pair: tuple[int, int] | None = eqx.field(static=True)
    clipped_nonbonded_pairs: int = eqx.field(static=True)
    required_rank: int = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)


_INCONSISTENCY_TOLERANCE = 1.0e-6


def prepare_conformer_constraints(
    component: MolecularComponent,
    table: ConformerGeometryTable,
    /,
    *,
    local_geometries: Mapping[int, LocalGeometry] | None = None,
) -> ConformerConstraints:
    """Derive and smooth the distance-geometry constraints of ``component``.

    ``local_geometries`` overrides the graph-derived local geometry of listed
    atoms (degree permitting); overrides contradicting declared stereo refuse.
    """
    if not isinstance(component, MolecularComponent):
        raise TypeError("component must be a MolecularComponent.")
    if not isinstance(table, ConformerGeometryTable):
        raise TypeError("table must be a ConformerGeometryTable.")
    count = component.atom_count
    overrides: dict[int, LocalGeometry] = {}
    for atom, geometry in ({} if local_geometries is None else local_geometries).items():
        parsed = parse(geometry, LocalGeometry, "local_geometries")
        if not 0 <= atom < count:
            raise ValueError("local_geometries keys must be component atoms.")
        overrides[atom] = parsed
    neighbors = component.neighbor_orders()
    kekule: list[dict[int, int]] = [{} for _ in range(count)]
    for (left, right), order in zip(
        component.bonds, component.kekule_bond_orders, strict=True
    ):
        kekule[left][right] = kekule[right][left] = order
    geometries = _classify(component, neighbors, kekule, overrides)
    numbers = component.atomic_numbers
    lengths = {
        (left, right): table.bond_length(numbers[left], numbers[right], order)
        for (left, right), order in zip(
            component.bonds, component.bond_orders, strict=True
        )
    }

    def length(first: int, second: int) -> float:
        return lengths[(min(first, second), max(first, second))]

    angles = _angle_bounds(neighbors, geometries, table.angle_tolerance)
    hops = _hop_distances(neighbors)
    total = sum(lengths.values()) + 1.0
    lower = np.zeros((count, count), dtype=np.float64)
    upper = np.full((count, count), total, dtype=np.float64)
    np.fill_diagonal(upper, 0.0)

    def tighten(first: int, second: int, low: float, high: float) -> None:
        lower[first, second] = lower[second, first] = max(lower[first, second], low)
        upper[first, second] = upper[second, first] = min(upper[first, second], high)

    for (left, right), value in lengths.items():
        tighten(left, right, value - table.bond_tolerance, value + table.bond_tolerance)
    for (first, center, last), (low_angle, high_angle) in angles.items():
        a, b = length(first, center), length(center, last)
        if hops[first, last] == 2:
            tighten(
                first,
                last,
                _cosine_distance(a, b, low_angle) - table.bond_tolerance,
                _cosine_distance(a, b, high_angle) + table.bond_tolerance,
            )
    _torsion_bounds(component, neighbors, geometries, angles, hops, length, tighten)
    smoothed_lower, smoothed_upper = _smooth(jnp.asarray(lower), jnp.asarray(upper))
    host_lower = np.asarray(smoothed_lower, dtype=np.float64)
    host_upper = np.asarray(smoothed_upper, dtype=np.float64)
    clipped = 0
    excess = host_lower - host_upper
    if float(np.max(excess)) <= _INCONSISTENCY_TOLERANCE:
        radii = table.nonbonded_radii.lookup(numbers)
        vdw = table.nonbonded_scale * (radii[:, None] + radii[None, :])
        distant = hops >= 4
        clipped = int(np.count_nonzero(np.triu(distant & (vdw > host_upper), 1)))
        raised = np.where(
            distant, np.maximum(host_lower, np.minimum(vdw, host_upper)), host_lower
        )
        smoothed_lower, smoothed_upper = _smooth(
            jnp.asarray(raised), jnp.asarray(host_upper)
        )
        host_lower = np.asarray(smoothed_lower, dtype=np.float64)
        host_upper = np.asarray(smoothed_upper, dtype=np.float64)
        excess = host_lower - host_upper
    worst = int(np.argmax(excess))
    maximum_excess = float(excess.reshape(-1)[worst])
    consistent = maximum_excess <= _INCONSISTENCY_TOLERANCE
    pair = None if consistent else tuple(sorted(divmod(worst, count)))
    chiral_atoms, chiral_signs, chiral_bounds = _chiral_constraints(component, length)
    planar_atoms, planar_scales = _planar_constraints(neighbors, geometries, length)
    required_rank = _required_rank(count, geometries)
    constraints_id = canonical_fingerprint(
        {
            "kind": "crystal-conformer-constraints",
            "component": component.component_id,
            "table": table.table_id,
            "local_geometries": list(geometries),
            "bounds": array_tree_fingerprint((host_lower, host_upper)),
        }
    )
    return ConformerConstraints(
        lower=jnp.asarray(host_lower),
        upper=jnp.asarray(host_upper),
        chiral_atoms=jnp.asarray(chiral_atoms),
        chiral_signs=jnp.asarray(chiral_signs),
        chiral_bounds=jnp.asarray(chiral_bounds),
        planar_atoms=jnp.asarray(planar_atoms),
        planar_scales=jnp.asarray(planar_scales),
        bond_lengths=tuple(lengths[bond] for bond in component.bonds),
        local_geometries=geometries,
        consistent=consistent,
        maximum_inconsistency=max(maximum_excess, 0.0),
        inconsistent_pair=None if pair is None else (pair[0], pair[1]),
        clipped_nonbonded_pairs=clipped,
        required_rank=required_rank,
        component_id=component.component_id,
        table_id=table.table_id,
        constraints_id=constraints_id,
    )


def _angle_bounds(
    neighbors: Sequence[dict[int, int]],
    geometries: Sequence[LocalGeometry],
    tolerance: float,
    /,
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Per-angle ``(low, high)`` degrees for every ``first < last`` around a center."""
    bounds: dict[tuple[int, int, int], tuple[float, float]] = {}
    for center, adjacent in enumerate(neighbors):
        geometry = geometries[center]
        if geometry == "terminal":
            continue
        pairs = [
            (first, last)
            for first in sorted(adjacent)
            for last in sorted(adjacent)
            if first < last
        ]
        rings = {
            pair: angle_ring_size(
                neighbors, pair[0], center, pair[1], limit=_MAXIMUM_ANGLE_RING
            )
            for pair in pairs
        }
        targets = {pair: _angle_target(geometry, rings[pair]) for pair in pairs}
        ring_pairs = [pair for pair in pairs if rings[pair] is not None]
        strained_center = any(strained for _, strained in targets.values())
        if geometry == "trigonal" and len(pairs) == 3 and 0 < len(ring_pairs) < 3:
            # Planar trigonal centers close their angle sum to 360 degrees.
            known = sum(targets[pair][0] for pair in ring_pairs)
            free = [pair for pair in pairs if pair not in ring_pairs]
            remainder = (360.0 - known) / len(free)
            for pair in free:
                targets[pair] = (remainder, strained_center)
        elif geometry == "tetrahedral" and strained_center:
            smallest = min(size for size in rings.values() if size is not None)
            exterior = _STRAINED_EXTERIOR_ANGLE[min(smallest, 5)]
            for pair in pairs:
                if rings[pair] is None:
                    targets[pair] = (exterior, True)
        for pair in pairs:
            value, strained = targets[pair]
            width = max(tolerance, _STRAINED_ANGLE_TOLERANCE) if strained else tolerance
            bounds[(pair[0], center, pair[1])] = (
                max(value - width, 0.0),
                min(value + width, 180.0),
            )
    return bounds


def _torsion_bounds(
    component: MolecularComponent,
    neighbors: Sequence[dict[int, int]],
    geometries: Sequence[LocalGeometry],
    angles: Mapping[tuple[int, int, int], tuple[float, float]],
    hops: np.ndarray,
    length: Callable[[int, int], float],
    tighten: Callable[[int, int, float, float], None],
    /,
) -> None:
    """Apply 1-4 bounds of every torsion path ``i-j-k-l`` with ``hops[i, l] == 3``."""
    declared = {(b, c): (a, d, sign) for a, b, c, d, sign in component.double_bond_stereo}

    def angle_range(first: int, center: int, last: int) -> tuple[float, float]:
        return angles[(min(first, last), center, max(first, last))]

    for (j, k), order in zip(component.bonds, component.bond_orders, strict=True):
        planar = (
            order in (2, AROMATIC_BOND)
            and geometries[j] == "trigonal"
            and geometries[k] == "trigonal"
        )
        first_substituents = [atom for atom in sorted(neighbors[j]) if atom != k]
        last_substituents = [atom for atom in sorted(neighbors[k]) if atom != j]
        # Each reference (x, y, sign) states that substituents x and y are cis
        # (+1) or trans (-1); a planar bond with two substituents per end fixes
        # every other pair by flipping.  Declared stereo and small-ring closure
        # are both applied, so contradicting references intersect to an
        # infeasible bound instead of one silently winning.
        references: list[tuple[int, int, int]] = []
        if (j, k) in declared:
            references.append(declared[(j, k)])
        if planar:
            ring_pair = next(
                (
                    (i, l)
                    for i in first_substituents
                    for l in last_substituents
                    if i != l
                    and _path_ring(neighbors, (i, j, k, l), _MAXIMUM_PLANAR_RING)
                ),
                None,
            )
            if ring_pair is not None:
                references.append((ring_pair[0], ring_pair[1], 1))
        for i in first_substituents:
            for l in last_substituents:
                if i == l or hops[i, l] != 3:
                    continue
                a, b, c = length(i, j), length(j, k), length(k, l)
                extremes = [
                    (theta1, theta2)
                    for theta1 in angle_range(i, j, k)
                    for theta2 in angle_range(j, k, l)
                ]
                cis = [_torsion_distance(a, b, c, t1, t2, True) for t1, t2 in extremes]
                trans = [_torsion_distance(a, b, c, t1, t2, False) for t1, t2 in extremes]
                if not references:
                    tighten(i, l, min(cis) - 0.05, max(trans) + 0.05)
                for x, y, sign in references:
                    relation = sign * (1 if i == x else -1) * (1 if l == y else -1)
                    chosen = cis if relation > 0 else trans
                    tighten(i, l, min(chosen) - 0.05, max(chosen) + 0.05)


def _path_ring(
    neighbors: Sequence[dict[int, int]], path: tuple[int, ...], limit: int, /
) -> bool:
    start, goal = path[-1], path[0]
    blocked = set(path[1:-1])
    distance = {start: 0}
    queue = deque((start,))
    budget = limit - (len(path) - 1)
    while queue:
        atom = queue.popleft()
        if distance[atom] >= budget:
            continue
        for neighbor in neighbors[atom]:
            if neighbor in blocked or neighbor in distance:
                continue
            if atom == start and neighbor == path[-2]:
                continue
            distance[neighbor] = distance[atom] + 1
            if neighbor == goal:
                return True
            queue.append(neighbor)
    return False


def _chiral_constraints(
    component: MolecularComponent, length: Callable[[int, int], float], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = np.zeros((len(component.tetrahedral_stereo), 4), dtype=np.int32)
    signs = np.zeros((len(component.tetrahedral_stereo),), dtype=np.float64)
    bounds = np.zeros((len(component.tetrahedral_stereo), 2), dtype=np.float64)
    for index, (center, a, b, c, sign) in enumerate(component.tetrahedral_stereo):
        ideal = (
            _UNIT_TETRAHEDRAL_VOLUME
            * length(center, a)
            * length(center, b)
            * length(center, c)
        )
        atoms[index] = (center, a, b, c)
        signs[index] = sign
        bounds[index] = (_CHIRAL_VOLUME_RANGE[0] * ideal, _CHIRAL_VOLUME_RANGE[1] * ideal)
    return atoms, signs, bounds


def _planar_constraints(
    neighbors: Sequence[dict[int, int]],
    geometries: Sequence[LocalGeometry],
    length: Callable[[int, int], float],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    quads: list[tuple[int, int, int, int]] = []
    scales: list[float] = []
    for center, adjacent in enumerate(neighbors):
        if geometries[center] == "trigonal" and len(adjacent) == 3:
            first, second, third = sorted(adjacent)
            quads.append((center, first, second, third))
            scales.append(
                length(center, first) * length(center, second) * length(center, third)
            )
    for begin, adjacent in enumerate(neighbors):
        for end, order in adjacent.items():
            if end <= begin or order not in (2, AROMATIC_BOND):
                continue
            if geometries[begin] != "trigonal" or geometries[end] != "trigonal":
                continue
            for first in sorted(n for n in neighbors[begin] if n != end):
                for last in sorted(n for n in neighbors[end] if n != begin):
                    if first == last:
                        continue
                    quads.append((begin, first, end, last))
                    scales.append(
                        length(begin, first) * length(begin, end) * length(end, last)
                    )
    return (
        np.asarray(quads, dtype=np.int32).reshape((-1, 4)),
        np.asarray(scales, dtype=np.float64),
    )


def _required_rank(count: int, geometries: Sequence[LocalGeometry], /) -> int:
    if count <= 2 or all(geometry in ("terminal", "linear") for geometry in geometries):
        return min(count - 1, 1)
    if count == 3 or all(
        geometry in ("terminal", "linear", "trigonal") for geometry in geometries
    ):
        return 2
    return 3


@final
class MetricEmbedding(StrictModule):
    """Low-rank metric-matrix embedding of one distance matrix.

    ``eigenvalues`` are the centered metric spectrum, descending.
    ``coordinates`` use the leading ``dimension`` eigenpairs with nonpositive
    eigenvalues contributing zero columns.  ``positive_rank`` counts leading
    eigenvalues above ``tolerance * max(|lambda|)``.  ``negative_mass`` is the
    fraction of absolute spectral mass that is negative (indefiniteness of the
    distances as a Euclidean matrix).  ``exact`` is true only when the
    discarded spectrum is within tolerance of zero, i.e. the distances are
    Euclidean of rank at most ``dimension``; otherwise the embedding is an
    explicit approximation.  ``rank_deficient`` is ``positive_rank <
    required_rank``.
    """

    coordinates: Array
    eigenvalues: Array
    positive_rank: int = eqx.field(static=True)
    negative_mass: float = eqx.field(static=True)
    discarded_mass: float = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    rank_deficient: bool = eqx.field(static=True)
    spectrum_valid: bool = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    required_rank: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


def metric_embedding(
    distances: ArrayLike,
    /,
    *,
    dimension: int,
    required_rank: int,
    tolerance: float = 1.0e-9,
) -> MetricEmbedding:
    """Embed a symmetric distance matrix through its centered metric spectrum."""
    matrix = jnp.asarray(distances, dtype=jnp.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("distances must be a non-empty square matrix.")
    dimension_ = positive_integer(dimension, "dimension")
    required = nonnegative_integer(required_rank, "required_rank")
    if required > dimension_:
        raise ValueError("required_rank must not exceed dimension.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    count = matrix.shape[0]
    squared = matrix**2
    centering = jnp.eye(count, dtype=jnp.float64) - jnp.full(
        (count, count), 1.0 / count, dtype=jnp.float64
    )
    metric = -0.5 * centering @ squared @ centering
    spectrum = HermitianSpectrum(metric, tolerance=tolerance_)
    order = jnp.arange(count - 1, -1, -1)
    values = spectrum.eigenvalues[order]
    vectors = spectrum.eigenvectors[:, order]
    kept = min(dimension_, count)
    leading = values[:kept]
    columns = vectors[:, :kept] * jnp.sqrt(jnp.maximum(leading, 0.0))[None, :]
    if kept < dimension_:
        columns = jnp.concatenate(
            (columns, jnp.zeros((count, dimension_ - kept), dtype=jnp.float64)), axis=1
        )
    host = np.asarray(values, dtype=np.float64)
    scale = max(float(np.max(np.abs(host))), np.finfo(np.float64).tiny)
    threshold = tolerance_ * scale
    positive_rank = int(np.count_nonzero(host[:kept] > threshold))
    total = float(np.sum(np.abs(host))) or 1.0
    negative_mass = float(np.sum(np.abs(host[host < -threshold]))) / total
    discarded_mass = float(np.sum(np.abs(host[kept:]))) / total
    valid = bool(spectrum.valid) and bool(np.all(np.isfinite(host)))
    return MetricEmbedding(
        coordinates=columns,
        eigenvalues=values,
        positive_rank=positive_rank,
        negative_mass=negative_mass,
        discarded_mass=discarded_mass,
        exact=valid
        and float(np.max(np.abs(host[kept:]), initial=0.0)) <= threshold
        and not np.any(host[:kept] < -threshold),
        rank_deficient=positive_rank < required,
        spectrum_valid=valid,
        dimension=dimension_,
        required_rank=required,
        tolerance=tolerance_,
    )


@final
class _ResidualData(StrictModule):
    first: Array
    second: Array
    lower_squared: Array
    upper_squared: Array
    chiral_atoms: Array
    chiral_signs: Array
    chiral_lower: Array
    chiral_upper: Array
    planar_atoms: Array
    planar_scales: Array
    chiral_weight: Array
    planar_weight: Array
    fourth_weight: Array


def _triple(first: Array, second: Array, third: Array, /) -> Array:
    return jnp.sum(first * jnp.cross(second, third), axis=-1)


def _residual(coordinates: Array, data: _ResidualData, /) -> Array:
    """Bound-violation, chiral-volume, planarity and fourth-dimension residuals."""
    delta = coordinates[data.first] - coordinates[data.second]
    squared = jnp.sum(delta**2, axis=-1)
    upper = jnp.maximum(squared / data.upper_squared - 1.0, 0.0)
    lower = jnp.maximum(
        2.0 * data.lower_squared / (data.lower_squared + squared + 1.0e-12) - 1.0, 0.0
    )
    spatial = coordinates[:, :3]
    chiral_center = spatial[data.chiral_atoms[:, 0]]
    volume = data.chiral_signs * _triple(
        spatial[data.chiral_atoms[:, 1]] - chiral_center,
        spatial[data.chiral_atoms[:, 2]] - chiral_center,
        spatial[data.chiral_atoms[:, 3]] - chiral_center,
    )
    chiral = (
        data.chiral_weight
        * (
            jnp.maximum(data.chiral_lower - volume, 0.0)
            + jnp.maximum(volume - data.chiral_upper, 0.0)
        )
        / data.chiral_upper
    )
    origin = spatial[data.planar_atoms[:, 0]]
    planar = (
        data.planar_weight
        * _triple(
            spatial[data.planar_atoms[:, 1]] - origin,
            spatial[data.planar_atoms[:, 2]] - origin,
            spatial[data.planar_atoms[:, 3]] - origin,
        )
        / data.planar_scales
    )
    fourth = data.fourth_weight * coordinates[:, 3:].reshape((-1,))
    return jnp.concatenate((upper, lower, chiral, planar, fourth))


@final
class ConformerQualification(StrictModule):
    """Acceptance thresholds of generated geometries.

    ``bound_tolerance`` (angstrom) bounds every smoothed-bound violation;
    ``planarity_tolerance`` bounds every normalized coplanarity volume;
    ``minimum_distance``, ``stereo_volume`` and ``double_bond_cosine`` are the
    `MolecularConformer` admission thresholds.
    """

    bound_tolerance: float = eqx.field(static=True)
    planarity_tolerance: float = eqx.field(static=True)
    minimum_distance: float = eqx.field(static=True)
    stereo_volume: float = eqx.field(static=True)
    double_bond_cosine: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        bound_tolerance: float = 0.1,
        planarity_tolerance: float = 0.05,
        minimum_distance: float = 0.5,
        stereo_volume: float = 0.1,
        double_bond_cosine: float = 0.5,
    ) -> None:
        self.bound_tolerance = positive_finite_float(bound_tolerance, "bound_tolerance")
        self.planarity_tolerance = positive_finite_float(
            planarity_tolerance, "planarity_tolerance"
        )
        self.minimum_distance = positive_finite_float(
            minimum_distance, "minimum_distance"
        )
        self.stereo_volume = positive_finite_float(stereo_volume, "stereo_volume")
        cosine = finite_real_scalar(double_bond_cosine, "double_bond_cosine")
        if not 0.0 <= cosine < 1.0:
            raise ValueError("double_bond_cosine must lie in [0, 1).")
        self.double_bond_cosine = cosine


@final
class ConformerEnergyCleanup(StrictModule):
    """Energetic cleanup by a caller-supplied native potential.

    ``energy`` maps component positions (angstrom, ``(atoms, 3)``) to a scalar
    energy in ``energy_unit``; ``model_id`` is the potential's explicit
    identity.  The cleaned geometry must keep every bond within
    ``bond_stretch_limit`` times its table length and requalify stereo and
    minimum separation; otherwise the attempt fails as ``"cleanup-failed"``.
    """

    energy: Callable[[Array], Array]
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    energy_unit: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    bond_stretch_limit: float = eqx.field(static=True)

    def __init__(
        self,
        energy: Callable[[Array], Array],
        /,
        *,
        method: AbstractMinimizationMethod,
        termination: OptimizationTermination,
        energy_unit: str,
        model_id: str,
        bond_stretch_limit: float = 1.25,
    ) -> None:
        if not callable(energy):
            raise TypeError("energy must be callable.")
        if not isinstance(method, AbstractMinimizationMethod):
            raise TypeError("method must be an AbstractMinimizationMethod.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination.")
        for value, name in ((energy_unit, "energy_unit"), (model_id, "model_id")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        limit = positive_finite_float(bond_stretch_limit, "bond_stretch_limit")
        if limit <= 1.0:
            raise ValueError("bond_stretch_limit must exceed one.")
        self.energy = energy
        self.method = method
        self.termination = termination
        self.energy_unit = energy_unit
        self.model_id = model_id
        self.bond_stretch_limit = limit


@final
class ConformerAttempt(StrictModule):
    """Host evidence of one addressed attempt (accepted or failed).

    Nonapplicable float evidence is NaN; optimizer statuses are
    `phydrax.optim.OptimizationStatus` codes (``-1`` when the stage did not
    run).
    """

    ordinal: int = eqx.field(static=True)
    attempt: int = eqx.field(static=True)
    status: ConformerAttemptStatus = eqx.field(static=True)
    embedding_positive_rank: int = eqx.field(static=True)
    embedding_negative_mass: float = eqx.field(static=True)
    embedding_exact: bool = eqx.field(static=True)
    embed_optimizer_status: int = eqx.field(static=True)
    refine_optimizer_status: int = eqx.field(static=True)
    refine_steps: int = eqx.field(static=True)
    maximum_bound_violation: float = eqx.field(static=True)
    minimum_stereo_volume: float = eqx.field(static=True)
    minimum_double_bond_cosine: float = eqx.field(static=True)
    maximum_planarity: float = eqx.field(static=True)
    minimum_pair_distance: float = eqx.field(static=True)
    cleanup_status: int = eqx.field(static=True)
    cleanup_energy: float = eqx.field(static=True)


@final
class ConformerGenerationResult(StrictModule, NonTrainableState):
    """Conformers and complete attempt evidence of one addressed ordinal range.

    ``conformers[k]`` belongs to ``accepted_ordinals[k]``; ordinals whose
    attempts all failed are listed in ``failed_ordinals``.  ``next_ordinal`` is
    the continuation cursor.
    """

    conformers: tuple[MolecularConformer, ...]
    attempts: tuple[ConformerAttempt, ...]
    accepted_ordinals: tuple[int, ...] = eqx.field(static=True)
    failed_ordinals: tuple[int, ...] = eqx.field(static=True)
    start_ordinal: int = eqx.field(static=True)
    requested: int = eqx.field(static=True)
    next_ordinal: int = eqx.field(static=True)
    status: ConformerGenerationStatus = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    root_key_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def pool(
        self, component: MolecularComponent, /, *, weights: Sequence[float] | None = None
    ) -> ConformerPool:
        """Pool the accepted conformers; refuses an empty result."""
        if not self.conformers:
            raise ValueError(f"No conformer was accepted (status {self.status!r}).")
        return ConformerPool(component, self.conformers, weights=weights)


@final
class ConformerGenerationPlan(StrictModule, NonTrainableState):
    """Bounded native distance-geometry conformer generation for one component.

    ``max_attempts`` bounds attempts per ordinal and ``max_atoms`` the
    admitted molecule size.  Stage terminations bound the four-dimensional and
    three-dimensional Levenberg-Marquardt refinements.
    """

    component: MolecularComponent
    constraints: ConformerConstraints
    qualification: ConformerQualification
    method: LevenbergMarquardt
    embed_termination: OptimizationTermination
    refine_termination: OptimizationTermination
    cleanup: ConformerEnergyCleanup | None
    max_attempts: int = eqx.field(static=True)
    max_atoms: int = eqx.field(static=True)
    chiral_weight: float = eqx.field(static=True)
    planarity_weight: float = eqx.field(static=True)
    fourth_dimension_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: MolecularComponent,
        constraints: ConformerConstraints,
        /,
        *,
        max_attempts: int = 16,
        max_atoms: int = 256,
        qualification: ConformerQualification | None = None,
        embed_termination: OptimizationTermination | None = None,
        refine_termination: OptimizationTermination | None = None,
        method: LevenbergMarquardt | None = None,
        cleanup: ConformerEnergyCleanup | None = None,
        chiral_weight: float = 1.0,
        planarity_weight: float = 1.0,
        fourth_dimension_weight: float = 0.1,
    ) -> None:
        if not isinstance(component, MolecularComponent):
            raise TypeError("component must be a MolecularComponent.")
        if not isinstance(constraints, ConformerConstraints):
            raise TypeError("constraints must be ConformerConstraints.")
        if constraints.component_id != component.component_id:
            raise ValueError("constraints belong to another component.")
        attempts = positive_integer(max_attempts, "max_attempts")
        atoms = positive_integer(max_atoms, "max_atoms")
        if component.atom_count > atoms:
            raise ValueError(
                f"Component has {component.atom_count} atoms, above max_atoms={atoms}."
            )
        qualification_ = (
            ConformerQualification() if qualification is None else qualification
        )
        method_ = LevenbergMarquardt() if method is None else method
        embed = (
            OptimizationTermination(maximum_steps=200)
            if embed_termination is None
            else embed_termination
        )
        refine = (
            OptimizationTermination(maximum_steps=400)
            if refine_termination is None
            else refine_termination
        )
        if not isinstance(qualification_, ConformerQualification):
            raise TypeError("qualification must be ConformerQualification or None.")
        if not isinstance(method_, LevenbergMarquardt):
            raise TypeError("method must be LevenbergMarquardt or None.")
        if not isinstance(embed, OptimizationTermination) or not isinstance(
            refine, OptimizationTermination
        ):
            raise TypeError("Stage terminations must be OptimizationTermination.")
        if cleanup is not None and not isinstance(cleanup, ConformerEnergyCleanup):
            raise TypeError("cleanup must be ConformerEnergyCleanup or None.")
        weights = tuple(
            positive_finite_float(value, name)
            for value, name in (
                (chiral_weight, "chiral_weight"),
                (planarity_weight, "planarity_weight"),
                (fourth_dimension_weight, "fourth_dimension_weight"),
            )
        )
        self.component = component
        self.constraints = constraints
        self.qualification = qualification_
        self.method = method_
        self.embed_termination = embed
        self.refine_termination = refine
        self.cleanup = cleanup
        self.max_attempts = attempts
        self.max_atoms = atoms
        self.chiral_weight, self.planarity_weight, self.fourth_dimension_weight = weights
        self.plan_id = canonical_fingerprint(
            {
                "kind": "crystal-conformer-generation-plan",
                "component": component.component_id,
                "constraints": constraints.constraints_id,
                "max_attempts": attempts,
                "max_atoms": atoms,
                "qualification": [
                    qualification_.bound_tolerance,
                    qualification_.planarity_tolerance,
                    qualification_.minimum_distance,
                    qualification_.stereo_volume,
                    qualification_.double_bond_cosine,
                ],
                "method": method_.method_id,
                "embed_steps": embed.maximum_steps,
                "refine_steps": refine.maximum_steps,
                "weights": list(weights),
                "cleanup": None
                if cleanup is None
                else [cleanup.model_id, cleanup.energy_unit, cleanup.method.method_id],
            }
        )

    @property
    def address(self) -> SampleAddress:
        """Semantic RNG address of this component's distance draws."""
        return SampleAddress(
            _ADDRESS_NAMESPACE,
            "conformer-generation",
            target=(self.component.component_id,),
            role="distance-draw",
        )

    def generate(
        self, root_key: PRNGKey, /, *, count: int, start_ordinal: int = 0
    ) -> ConformerGenerationResult:
        """Generate ordinals ``[start_ordinal, start_ordinal + count)``.

        Each ordinal runs at most ``max_attempts`` addressed attempts and stops
        at its first qualified conformer.  Results for an ordinal do not depend
        on the range it was requested in.
        """
        if (
            not isinstance(root_key, jax.Array)
            or not jax.dtypes.issubdtype(root_key.dtype, jax.dtypes.prng_key)
            or root_key.shape != ()
        ):
            raise TypeError("root_key must be one typed JAX key from jax.random.key.")
        root = root_key
        requested = nonnegative_integer(count, "count")
        start = nonnegative_integer(start_ordinal, "start_ordinal")
        if start + requested > 2**32:
            raise ValueError("Conformer ordinals must fit one uint32 address word.")
        root_key_record = array_tree_fingerprint(np.asarray(jax.random.key_data(root)))
        root_key_id = canonical_fingerprint(root_key_record)
        conformers: list[MolecularConformer] = []
        attempts: list[ConformerAttempt] = []
        accepted: list[int] = []
        failed: list[int] = []
        if requested == 0:
            status: ConformerGenerationStatus = "empty"
        elif not self.constraints.consistent:
            status = "bounds-inconsistent"
            failed.extend(range(start, start + requested))
        else:
            stages = (self._residual_data("embed"), self._residual_data("refine"))
            for ordinal in range(start, start + requested):
                conformer = None
                for attempt in range(self.max_attempts):
                    record, conformer = self._attempt(root, ordinal, attempt, stages)
                    attempts.append(record)
                    if conformer is not None:
                        break
                if conformer is None:
                    failed.append(ordinal)
                else:
                    conformers.append(conformer)
                    accepted.append(ordinal)
            status = "attempt-budget-exhausted" if failed else "complete"
        return ConformerGenerationResult(
            conformers=tuple(conformers),
            attempts=tuple(attempts),
            accepted_ordinals=tuple(accepted),
            failed_ordinals=tuple(failed),
            start_ordinal=start,
            requested=requested,
            next_ordinal=start + requested,
            status=status,
            plan_id=self.plan_id,
            root_key_id=root_key_id,
            result_id=canonical_fingerprint(
                {
                    "kind": "crystal-conformer-generation-result",
                    "plan": self.plan_id,
                    "root_key": root_key_record,
                    "start": start,
                    "requested": requested,
                    "conformers": [item.conformer_id for item in conformers],
                    "failed": failed,
                }
            ),
        )

    def _residual_data(self, stage: Literal["embed", "refine"], /) -> _ResidualData:
        count = self.component.atom_count
        first, second = np.triu_indices(count, 1)
        constraints = self.constraints
        lower = np.asarray(constraints.lower)[first, second]
        upper = np.asarray(constraints.upper)[first, second]
        return _ResidualData(
            first=jnp.asarray(first, dtype=jnp.int32),
            second=jnp.asarray(second, dtype=jnp.int32),
            lower_squared=jnp.asarray(lower**2),
            upper_squared=jnp.asarray(upper**2),
            chiral_atoms=constraints.chiral_atoms,
            chiral_signs=constraints.chiral_signs,
            chiral_lower=constraints.chiral_bounds[:, 0],
            chiral_upper=constraints.chiral_bounds[:, 1],
            planar_atoms=constraints.planar_atoms,
            planar_scales=constraints.planar_scales,
            chiral_weight=jnp.asarray(self.chiral_weight, dtype=jnp.float64),
            planar_weight=jnp.asarray(
                0.0 if stage == "embed" else self.planarity_weight, dtype=jnp.float64
            ),
            fourth_weight=jnp.asarray(self.fourth_dimension_weight, dtype=jnp.float64),
        )

    def _attempt(
        self,
        root: Array,
        ordinal: int,
        attempt: int,
        stages: tuple[_ResidualData, _ResidualData],
        /,
    ) -> tuple[ConformerAttempt, MolecularConformer | None]:
        count = self.component.atom_count
        nan = float("nan")
        evidence: dict[str, float | int | bool] = {
            "embedding_positive_rank": 0,
            "embedding_negative_mass": nan,
            "embedding_exact": False,
            "embed_optimizer_status": -1,
            "refine_optimizer_status": -1,
            "refine_steps": 0,
            "maximum_bound_violation": nan,
            "minimum_stereo_volume": nan,
            "minimum_double_bond_cosine": nan,
            "maximum_planarity": nan,
            "minimum_pair_distance": nan,
            "cleanup_status": -1,
            "cleanup_energy": nan,
        }

        def record(status: ConformerAttemptStatus) -> ConformerAttempt:
            return ConformerAttempt(
                ordinal=ordinal,
                attempt=attempt,
                status=status,
                embedding_positive_rank=int(evidence["embedding_positive_rank"]),
                embedding_negative_mass=float(evidence["embedding_negative_mass"]),
                embedding_exact=bool(evidence["embedding_exact"]),
                embed_optimizer_status=int(evidence["embed_optimizer_status"]),
                refine_optimizer_status=int(evidence["refine_optimizer_status"]),
                refine_steps=int(evidence["refine_steps"]),
                maximum_bound_violation=float(evidence["maximum_bound_violation"]),
                minimum_stereo_volume=float(evidence["minimum_stereo_volume"]),
                minimum_double_bond_cosine=float(evidence["minimum_double_bond_cosine"]),
                maximum_planarity=float(evidence["maximum_planarity"]),
                minimum_pair_distance=float(evidence["minimum_pair_distance"]),
                cleanup_status=int(evidence["cleanup_status"]),
                cleanup_energy=float(evidence["cleanup_energy"]),
            )

        if count == 1:
            positions = np.zeros((1, 3), dtype=np.float64)
            return self._qualify(positions, ordinal, attempt, evidence, record)
        key = derive_key(root, self.address, ordinal, attempt)
        lower, upper = self.constraints.lower, self.constraints.upper
        draw = jax.random.uniform(key, lower.shape, dtype=jnp.float64)
        draw = jnp.triu(draw, 1)
        draw = draw + draw.T
        distances = lower + draw * (upper - lower)
        embedding = metric_embedding(
            distances, dimension=4, required_rank=self.constraints.required_rank
        )
        evidence["embedding_positive_rank"] = embedding.positive_rank
        evidence["embedding_negative_mass"] = embedding.negative_mass
        evidence["embedding_exact"] = embedding.exact
        if not embedding.spectrum_valid:
            return record("embedding-nonfinite"), None
        if embedding.rank_deficient:
            return record("embedding-rank-deficient"), None
        embedded = least_squares(
            _residual,
            embedding.coordinates,
            method=self.method,
            termination=self.embed_termination,
            args=stages[0],
        )
        embed_status = int(embedded.status)
        evidence["embed_optimizer_status"] = embed_status
        if embed_status not in _TOLERATED_OPTIMIZER_STATUSES:
            return record("optimizer-failed"), None
        refined = least_squares(
            _residual,
            jnp.asarray(embedded.parameters)[:, :3],
            method=self.method,
            termination=self.refine_termination,
            args=stages[1],
        )
        refine_status = int(refined.status)
        evidence["refine_optimizer_status"] = refine_status
        evidence["refine_steps"] = int(refined.diagnostics.iterations)
        if refine_status not in _TOLERATED_OPTIMIZER_STATUSES:
            return record("optimizer-failed"), None
        positions = np.asarray(refined.parameters, dtype=np.float64)
        if not np.all(np.isfinite(positions)):
            return record("optimizer-failed"), None
        return self._qualify(positions, ordinal, attempt, evidence, record)

    def _qualify(
        self,
        positions: np.ndarray,
        ordinal: int,
        attempt: int,
        evidence: dict[str, float | int | bool],
        record: Callable[[ConformerAttemptStatus], ConformerAttempt],
        /,
    ) -> tuple[ConformerAttempt, MolecularConformer | None]:
        geometry = qualify_geometry(self.component, self.constraints, positions)
        evidence.update(
            maximum_bound_violation=geometry.maximum_bound_violation,
            minimum_stereo_volume=geometry.minimum_stereo_volume,
            minimum_double_bond_cosine=geometry.minimum_double_bond_cosine,
            maximum_planarity=geometry.maximum_planarity,
            minimum_pair_distance=geometry.minimum_pair_distance,
        )
        failure = geometry.failure(self.qualification)
        if failure is not None:
            return record(failure), None
        provenance = (
            f"native-distance-geometry plan={self.plan_id} ordinal={ordinal} "
            f"attempt={attempt}"
        )
        if self.cleanup is not None:
            cleaned = self._cleanup(positions, evidence)
            if cleaned is None:
                return record("cleanup-failed"), None
            positions = cleaned
            provenance += f" cleanup={self.cleanup.model_id}"
        conformer = MolecularConformer(
            self.component,
            positions,
            provenance=provenance,
            source="native-distance-geometry",
            minimum_distance=self.qualification.minimum_distance,
            stereo_volume=self.qualification.stereo_volume,
            double_bond_cosine=self.qualification.double_bond_cosine,
        )
        return record("accepted"), conformer

    def _cleanup(
        self, positions: np.ndarray, evidence: dict[str, float | int | bool], /
    ) -> np.ndarray | None:
        cleanup = self.cleanup
        if cleanup is None:
            raise ValueError("No energetic cleanup is declared.")
        energy = cleanup.energy

        def objective(value: Array, _: None) -> Array:
            return energy(value)

        optimized = minimize(
            objective,
            jnp.asarray(positions),
            method=cleanup.method,
            termination=cleanup.termination,
        )
        status = int(optimized.status)
        evidence["cleanup_status"] = status
        cleaned = np.asarray(optimized.parameters, dtype=np.float64)
        if not np.all(np.isfinite(cleaned)):
            return None
        evidence["cleanup_energy"] = float(energy(jnp.asarray(cleaned)))
        if status not in _TOLERATED_OPTIMIZER_STATUSES:
            return None
        bonds = np.asarray(self.component.bonds, dtype=np.int64).reshape((-1, 2))
        if bonds.size:
            stretched = np.linalg.norm(
                cleaned[bonds[:, 0]] - cleaned[bonds[:, 1]], axis=-1
            )
            reference = np.asarray(self.constraints.bond_lengths, dtype=np.float64)
            if np.any(stretched > cleanup.bond_stretch_limit * reference):
                return None
        realized = stereo_realization(self.component, cleaned)
        if np.any(
            realized.tetrahedral_volumes < self.qualification.stereo_volume
        ) or np.any(realized.double_bond_cosines < self.qualification.double_bond_cosine):
            return None
        if _minimum_pair_distance(cleaned) <= self.qualification.minimum_distance:
            return None
        return cleaned


def _minimum_pair_distance(positions: np.ndarray, /) -> float:
    if positions.shape[0] < 2:
        return float("inf")
    first, second = np.triu_indices(positions.shape[0], 1)
    return float(np.min(np.linalg.norm(positions[first] - positions[second], axis=-1)))


@final
class GeometryQualification(StrictModule):
    """Geometry evidence against smoothed bounds, stereo and planarity."""

    maximum_bound_violation: float = eqx.field(static=True)
    minimum_stereo_volume: float = eqx.field(static=True)
    minimum_double_bond_cosine: float = eqx.field(static=True)
    maximum_planarity: float = eqx.field(static=True)
    minimum_pair_distance: float = eqx.field(static=True)

    def failure(
        self, qualification: ConformerQualification, /
    ) -> ConformerAttemptStatus | None:
        """First failed criterion in a fixed order, or ``None`` when qualified."""
        if not self.minimum_pair_distance > qualification.minimum_distance:
            return "minimum-distance"
        if not self.minimum_stereo_volume >= qualification.stereo_volume or not (
            self.minimum_double_bond_cosine >= qualification.double_bond_cosine
        ):
            return "stereo-violated"
        if not self.maximum_planarity <= qualification.planarity_tolerance:
            return "planarity-violated"
        if not self.maximum_bound_violation <= qualification.bound_tolerance:
            return "bounds-violated"
        return None


def qualify_geometry(
    component: MolecularComponent,
    constraints: ConformerConstraints,
    positions: ArrayLike,
    /,
) -> GeometryQualification:
    """Evaluate a geometry against the component's prepared constraints.

    Absent stereo or planarity constraints report ``+inf`` stereo margins and
    zero planarity.
    """
    if constraints.component_id != component.component_id:
        raise ValueError("constraints belong to another component.")
    host = np.array(positions, dtype=np.float64)
    if host.shape != (component.atom_count, 3) or not np.all(np.isfinite(host)):
        raise ValueError("positions must be finite with shape (atom_count, 3).")
    first, second = np.triu_indices(component.atom_count, 1)
    distances = np.linalg.norm(host[first] - host[second], axis=-1)
    lower = np.asarray(constraints.lower)[first, second]
    upper = np.asarray(constraints.upper)[first, second]
    violation = np.maximum(np.maximum(distances - upper, lower - distances), 0.0)
    realized = stereo_realization(component, host)
    planar = np.asarray(constraints.planar_atoms, dtype=np.int64)
    if planar.shape[0]:
        origin = host[planar[:, 0]]
        volumes = np.einsum(
            "ij,ij->i",
            host[planar[:, 1]] - origin,
            np.cross(host[planar[:, 2]] - origin, host[planar[:, 3]] - origin),
        )
        planarity = float(np.max(np.abs(volumes) / np.asarray(constraints.planar_scales)))
    else:
        planarity = 0.0
    return GeometryQualification(
        maximum_bound_violation=float(np.max(violation, initial=0.0)),
        minimum_stereo_volume=float(np.min(realized.tetrahedral_volumes, initial=np.inf)),
        minimum_double_bond_cosine=float(
            np.min(realized.double_bond_cosines, initial=np.inf)
        ),
        maximum_planarity=planarity,
        minimum_pair_distance=_minimum_pair_distance(host),
    )


@final
class ConformerRMSD(StrictModule):
    """Minimum proper-rotation RMSD over the component's automorphisms.

    ``complete`` is the component's ``automorphisms_complete``: an incomplete
    list makes ``rmsd`` an upper bound of the symmetry-minimized value.
    """

    rmsd: float = eqx.field(static=True)
    permutation: tuple[int, ...] = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    include_hydrogens: bool = eqx.field(static=True)


def _horn_rmsd(first: np.ndarray, second: np.ndarray, /) -> float:
    """Proper-rotation RMSD of centered point sets via Horn's quaternion spectrum."""
    left = first - first.mean(axis=0)
    right = second - second.mean(axis=0)
    m = left.T @ right
    key = np.asarray(
        [
            [m[0, 0] + m[1, 1] + m[2, 2], m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]],
            [m[1, 2] - m[2, 1], m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]],
            [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], -m[0, 0] + m[1, 1] - m[2, 2], m[1, 2] + m[2, 1]],
            [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], -m[0, 0] - m[1, 1] + m[2, 2]],
        ]
    )  # fmt: skip
    spectrum = HermitianSpectrum(jnp.asarray(key), tolerance=1.0e-9)
    largest = float(np.asarray(spectrum.eigenvalues)[-1])
    squared = (np.sum(left**2) + np.sum(right**2) - 2.0 * largest) / first.shape[0]
    return sqrt(max(squared, 0.0))


def conformer_rmsd(
    component: MolecularComponent,
    first: MolecularConformer,
    second: MolecularConformer,
    /,
    *,
    include_hydrogens: bool = False,
) -> ConformerRMSD:
    """Symmetry-aware proper-rotation RMSD of two conformers of ``component``."""
    for conformer in (first, second):
        if conformer.component_id != component.component_id:
            raise ValueError("Conformers belong to another component.")
    selected = np.asarray(
        [
            atom
            for atom, number in enumerate(component.atomic_numbers)
            if include_hydrogens or number != 1
        ],
        dtype=np.int64,
    )
    if selected.size == 0:
        selected = np.arange(component.atom_count, dtype=np.int64)
    left = np.asarray(first.positions, dtype=np.float64)
    right = np.asarray(second.positions, dtype=np.float64)
    best = float("inf")
    best_permutation = tuple(range(component.atom_count))
    for permutation in component.automorphisms:
        mapped = np.asarray(permutation, dtype=np.int64)
        value = _horn_rmsd(left[selected], right[mapped[selected]])
        if value < best:
            best, best_permutation = value, permutation
    return ConformerRMSD(
        rmsd=best,
        permutation=best_permutation,
        complete=component.automorphisms_complete,
        include_hydrogens=include_hydrogens,
    )


def prune_conformers(
    component: MolecularComponent,
    conformers: Sequence[MolecularConformer],
    /,
    *,
    rmsd_threshold: float,
    include_hydrogens: bool = False,
) -> tuple[tuple[MolecularConformer, ...], tuple[int, ...]]:
    """Greedy in-order RMSD pruning; returns kept conformers and removed indices.

    A conformer is kept when its symmetry-aware RMSD to every kept conformer
    exceeds ``rmsd_threshold`` (angstrom).  Order defines the result; with an
    incomplete automorphism list some symmetric duplicates may be kept.
    """
    threshold = positive_finite_float(rmsd_threshold, "rmsd_threshold")
    kept: list[MolecularConformer] = []
    removed: list[int] = []
    for index, conformer in enumerate(conformers):
        if any(
            conformer_rmsd(
                component, existing, conformer, include_hydrogens=include_hydrogens
            ).rmsd
            <= threshold
            for existing in kept
        ):
            removed.append(index)
        else:
            kept.append(conformer)
    return tuple(kept), tuple(removed)


__all__ = [
    "ChiralCenterDim",
    "ConformerAttempt",
    "ConformerAttemptStatus",
    "ConformerConstraints",
    "ConformerEnergyCleanup",
    "ConformerGenerationPlan",
    "ConformerGenerationResult",
    "ConformerGenerationStatus",
    "ConformerGeometryTable",
    "ConformerQualification",
    "ConformerRMSD",
    "GeometryQualification",
    "LocalGeometry",
    "MetricEmbedding",
    "PlanarityDim",
    "conformer_rmsd",
    "metric_embedding",
    "prepare_conformer_constraints",
    "prune_conformers",
    "qualify_geometry",
]
