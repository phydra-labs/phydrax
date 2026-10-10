#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Private preparation of scientific nested reference integration pairs."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import permutations
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ._cell_geometry import CellGeometryElement, CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import affine_arguments, Polynomial
from ._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from ._cell_geometry_transfer import CellGeometryTransition, NestedReferenceWitnesses


class _NestedReferencePair(NamedTuple):
    source_cell: int
    target_cell: int
    fine_is_source: bool
    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]


class _ReferenceMap(NamedTuple):
    kind: str
    arguments: tuple[Polynomial, ...]


def _fraction_vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(value) for value in row)
        for row in reference_cell_topology(kind).vertices
    )


def _dot(first: Sequence[Fraction], second: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _normal(
    points: Sequence[Sequence[Fraction]],
    center: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    dimension = len(points[0])
    if dimension == 1:
        return (Fraction(1) if points[0][0] > center[0] else Fraction(-1),)
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    if dimension == 2:
        normal: tuple[Fraction, ...] = (first[1], -first[0])
    else:
        second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
    outward = tuple(a - b for a, b in zip(points[0], center, strict=True))
    return normal if _dot(normal, outward) > 0 else tuple(-value for value in normal)


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    dimension = len(matrix)
    result = Fraction(0)
    for order in permutations(range(dimension)):
        inversions = sum(
            order[i] > order[j] for i in range(dimension) for j in range(i + 1, dimension)
        )
        term = Fraction((-1) ** inversions)
        for row, column in enumerate(order):
            term *= matrix[row][column]
        result += term
    return result


def _reference_measure(kind: str) -> Fraction:
    return {
        "interval": Fraction(1),
        "triangle": Fraction(1, 2),
        "quadrilateral": Fraction(1),
        "tetrahedron": Fraction(1, 6),
        "prism": Fraction(1, 2),
        "hexahedron": Fraction(1),
        "pyramid": Fraction(1, 3),
    }[kind]


def _require_reference_partition(
    coarse_kind: str,
    fine_maps: Sequence[_ReferenceMap],
) -> None:
    """Prove an oriented exact reference tiling, rather than sampling it.

    Positive fine cells are contained in the convex parent. Every interior
    polygon is paired with its opposite orientation, and unpaired polygons
    lie on the parent boundary. The closed outward boundary chain therefore
    has constant nonnegative integer multiplicity on that boundary; the exact
    reference-volume equality fixes it to one. Its winding number is one in
    the parent and zero outside, so positive cells cannot leave gaps/overlap.
    """
    vertices = _fraction_vertices(coarse_kind)
    dimension = len(vertices[0])
    center = tuple(
        sum((row[axis] for row in vertices), Fraction(0)) / len(vertices)
        for axis in range(dimension)
    )
    planes: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for face in reference_cell_topology(coarse_kind).entities[dimension - 1]:
        points = tuple(vertices[index] for index in face)
        normal = _normal(points, center)
        planes.append((normal, _dot(normal, points[0])))
    faces: dict[tuple[tuple[Fraction, ...], ...], list[tuple[Fraction, ...]]] = {}
    measure = Fraction(0)
    from ._cell_geometry_transfer import _mapped_polynomial_integral_fraction
    from ._coordinate_enclosure import (
        bernstein_coefficients,
        derivative,
        determinant as polynomial_determinant,
        evaluate,
    )

    for kind, arguments in fine_maps:
        if len(arguments) != dimension:
            raise ValueError(
                "Nested reference maps must preserve the declared reference dimension."
            )
        affine = all(sum(index) <= 1 for value in arguments for index in value)
        if affine:
            coefficients = tuple(
                tuple(
                    value.get(
                        tuple(int(axis == column) for axis in range(dimension)),
                        Fraction(0),
                    )
                    for column in range(dimension)
                )
                for value in arguments
            )
            determinant = _determinant(coefficients)
            piece_measure = determinant * _reference_measure(kind)
        else:
            if (
                dimension != 2
                or kind != "quadrilateral"
                or any(
                    any(exponent > 1 for exponent in index)
                    for value in arguments
                    for index in value
                )
            ):
                raise ValueError(
                    "Nonlinear nested reference partitions require exact bilinear quad charts."
                )
            jacobian = polynomial_determinant(
                tuple(
                    tuple(derivative(value, axis) for axis in range(2))
                    for value in arguments
                )
            )
            determinant = min(bernstein_coefficients(jacobian, "box", 2))
            piece_measure = _mapped_polynomial_integral_fraction(jacobian, kind)
        if determinant <= 0:
            raise ValueError("Nested reference integration needs positive fine maps.")
        fine = tuple(
            tuple(evaluate(value, vertex) for value in arguments)
            for vertex in _fraction_vertices(kind)
        )
        if any(_dot(normal, point) > bound for point in fine for normal, bound in planes):
            raise ValueError("A nested reference fine cell is outside its parent.")
        fine_center = tuple(
            sum((row[axis] for row in fine), Fraction(0)) / len(fine)
            for axis in range(dimension)
        )
        for face in reference_cell_topology(kind).entities[dimension - 1]:
            points = tuple(fine[index] for index in face)
            faces.setdefault(tuple(sorted(points)), []).append(
                _normal(points, fine_center)
            )
        measure += piece_measure
    if measure != _reference_measure(coarse_kind):
        raise ValueError(
            "Nested reference cells do not cover the complete parent measure."
        )
    for points, normals in faces.items():
        if len(normals) == 2:
            if _dot(normals[0], normals[1]) >= 0:
                raise ValueError(
                    "Nested reference interior facets have inconsistent orientation."
                )
        elif len(normals) == 1:
            if not any(
                all(_dot(normal, point) == bound for point in points)
                and _dot(normal, normals[0]) > 0
                for normal, bound in planes
            ):
                raise ValueError(
                    "An unpaired nested reference facet is not on the parent boundary."
                )
        else:
            raise ValueError(
                "Nested reference facets have overlapping or nonmanifold multiplicity."
            )


def _nested_reference_pairs(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
) -> tuple[_NestedReferencePair, ...]:
    """Bind immediate refinement/coarsening witnesses to actual geometry IDs.

    Each matrix/offset maps the finer reference cell into the coarser one.
    The field/volume owner additionally proves equality of the represented
    maps under that composition; this preparation never treats root-source
    coefficients as target nodal coordinates.
    """
    source_geometry.resolve(source_mesh)
    target_geometry.resolve(target_mesh)
    source_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    source_index = {int(value): index for index, value in enumerate(source_ids)}
    target_index = {int(value): index for index, value in enumerate(target_ids)}
    source_kinds = tuple(
        block.cell_kind for block in source_mesh.blocks for _ in range(block.cell_count)
    )
    target_kinds = tuple(
        block.cell_kind for block in target_mesh.blocks for _ in range(block.cell_count)
    )
    if source_mesh.topological_dimension != target_mesh.topological_dimension:
        raise ValueError("Nested reference transfers need one common dimension.")
    dimension = source_mesh.topological_dimension
    records: list[tuple[int, int, bool, NDArray[np.float64]]] = []
    if geometry_transition is not None:
        from ._cell_geometry_transfer import CellGeometryTransition

        if not isinstance(geometry_transition, CellGeometryTransition):
            raise TypeError("geometry_transition must be a CellGeometryTransition.")
        if (
            geometry_transition.source_topology_id != source_mesh.topology_id
            or geometry_transition.target_topology_id != target_mesh.topology_id
            or geometry_transition.source_geometry_id != cell_geometry_id(source_geometry)
            or geometry_transition.target_geometry_id != cell_geometry_id(target_geometry)
        ):
            raise ValueError(
                "Nested geometry witnesses do not bind the actual source/target maps."
            )
        for target_id, source_id, corners in zip(
            np.asarray(geometry_transition.target_cell_ids),
            np.asarray(geometry_transition.parent_cell_ids),
            np.asarray(geometry_transition.parent_reference_vertices),
            strict=True,
        ):
            if int(target_id) not in target_index:
                raise ValueError(
                    "A nested witness names an absent target scientific cell."
                )
            if source_id >= 0:
                if int(source_id) not in source_index:
                    raise ValueError(
                        "A nested witness names an absent source scientific cell."
                    )
                records.append(
                    (
                        source_index[int(source_id)],
                        target_index[int(target_id)],
                        False,
                        corners,
                    )
                )
        for source_id, target_id, corners in zip(
            np.asarray(geometry_transition.coarsened_cell_ids),
            np.asarray(geometry_transition.coarsened_into_ids),
            np.asarray(geometry_transition.coarsened_reference_vertices),
            strict=True,
        ):
            if int(source_id) not in source_index or int(target_id) not in target_index:
                raise ValueError("A coarsening witness names an absent scientific cell.")
            records.append(
                (
                    source_index[int(source_id)],
                    target_index[int(target_id)],
                    True,
                    corners,
                )
            )
    elif coarsening_witnesses is not None:
        from ._cell_geometry_transfer import NestedReferenceWitnesses

        if not isinstance(coarsening_witnesses, NestedReferenceWitnesses):
            raise TypeError("coarsening_witnesses must be NestedReferenceWitnesses.")
        fine_ids = np.asarray(coarsening_witnesses.fine_cell_ids, dtype=np.int64)
        if np.unique(fine_ids).size != fine_ids.size:
            raise ValueError("A coarsening witness repeats a source scientific cell.")
        for source_id, target_id, corners in zip(
            fine_ids,
            np.asarray(coarsening_witnesses.coarse_cell_ids),
            np.asarray(coarsening_witnesses.fine_reference_vertices),
            strict=True,
        ):
            if int(source_id) not in source_index or int(target_id) not in target_index:
                raise ValueError("A coarsening witness names an absent scientific cell.")
            records.append(
                (
                    source_index[int(source_id)],
                    target_index[int(target_id)],
                    True,
                    corners,
                )
            )
    else:
        if parent_cells is None or parent_reference_vertices is None:
            raise ValueError(
                "Nested refinement requires parent cells and reference corners."
            )
        parents = np.asarray(parent_cells)
        corners = np.asarray(parent_reference_vertices, dtype=np.float64)
        if parents.shape != (target_ids.size,) or not np.issubdtype(
            parents.dtype, np.integer
        ):
            raise ValueError(
                "Nested refinement requires one integer source parent per target cell."
            )
        if np.any(parents < 0) or np.any(parents >= source_ids.size):
            raise ValueError("Nested source parent indices are outside the source cells.")
        if (
            corners.ndim != 3
            or corners.shape[0] != target_ids.size
            or corners.shape[2] != dimension
        ):
            raise ValueError(
                "Nested reference corners must align with target cells and dimension."
            )
        records = [
            (int(parent), index, False, corners[index])
            for index, parent in enumerate(parents)
        ]
    if not records:
        raise ValueError("Nested transfer has no reference integration pieces.")
    covered_source = {record[0] for record in records}
    covered_target = {record[1] for record in records}
    if covered_source != set(range(source_ids.size)) or covered_target != set(
        range(target_ids.size)
    ):
        raise ValueError(
            "Nested transfer witnesses leave source or target cells uncovered."
        )
    frame_indices = {
        "interval": (1,),
        "triangle": (1, 2),
        "quadrilateral": (1, 3),
        "tetrahedron": (1, 2, 3),
        "prism": (1, 2, 3),
        "hexahedron": (1, 3, 4),
        "pyramid": (1, 3, 4),
    }
    built: list[_NestedReferencePair | None] = [None] * len(records)
    groups: dict[str, list[tuple[int, NDArray[np.float64]]]] = {}
    for index, (source, target, fine_is_source, corners) in enumerate(records):
        kind = source_kinds[source] if fine_is_source else target_kinds[target]
        if kind not in frame_indices or not np.all(np.isfinite(corners)):
            raise ValueError("A nested fine reference map is unsupported or nonfinite.")
        arity = len(reference_cell_topology(kind).vertices)
        if corners.shape[0] < arity or corners.shape[1] != dimension:
            raise ValueError(
                "A nested reference witness does not contain its fine-cell corners."
            )
        groups.setdefault(kind, []).append((index, np.asarray(corners[:arity])))
    for kind, entries in groups.items():
        vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        corners = np.stack([row for _, row in entries])
        selected = list(frame_indices[kind])
        frame = (vertices[selected] - vertices[0]).T
        images = np.swapaxes(corners[:, selected] - corners[:, :1], -1, -2)
        solved = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            np.broadcast_to(frame.T, (len(entries), dimension, dimension)),
            np.swapaxes(images, -1, -2),
        )
        if not np.all(np.asarray(solved.successful)):
            raise ValueError("A nested reference witness has a singular affine frame.")
        matrices = np.swapaxes(np.asarray(solved.value, dtype=np.float64), -1, -2)
        offsets = corners[:, 0] - matrices @ vertices[0]
        reconstructed = vertices[None] @ np.swapaxes(matrices, -1, -2) + offsets[:, None]
        if not np.array_equal(reconstructed, corners):
            raise ValueError(
                "A nested reference witness is not an exact represented affine map."
            )
        for row, (index, _) in enumerate(entries):
            source, target, fine_is_source, _ = records[index]
            built[index] = _NestedReferencePair(
                source, target, fine_is_source, matrices[row], offsets[row]
            )
    output: list[_NestedReferencePair] = []
    for pair in built:
        if pair is None:
            raise ValueError("A nested reference witness has no supported affine map.")
        output.append(pair)
    partitions: dict[tuple[bool, int], list[_ReferenceMap]] = {}
    for pair in output:
        coarse = (
            pair.fine_is_source,
            pair.target_cell if pair.fine_is_source else pair.source_cell,
        )
        fine_kind = (
            source_kinds[pair.source_cell]
            if pair.fine_is_source
            else target_kinds[pair.target_cell]
        )
        partitions.setdefault(coarse, []).append(
            _ReferenceMap(fine_kind, affine_arguments(pair.offset, pair.matrix))
        )
    for (fine_is_source, index), maps in partitions.items():
        coarse_kind = target_kinds[index] if fine_is_source else source_kinds[index]
        _require_reference_partition(coarse_kind, maps)
    return tuple(output)


class _PolynomialReferencePair(NamedTuple):
    source_cell: int
    target_cell: int
    fine_is_source: bool
    arguments: tuple[Polynomial, ...]
    source_kind: str
    target_kind: str
    jacobian_bounds: tuple[Fraction, Fraction]


def _polynomial_reference_arguments(
    coarse: CellGeometryElement,
    fine: CellGeometryElement,
) -> tuple[Polynomial, ...]:
    from ._coordinate_enclosure import (
        _solve_exact,
        add,
        compose,
        constant,
        coordinate_reference_chain,
        coordinate_source_signature,
        RationalPolynomial,
    )

    coarse_root, coarse_expressions = coordinate_reference_chain(coarse)
    fine_root, fine_expressions = coordinate_reference_chain(fine)
    if any(
        isinstance(value, RationalPolynomial)
        for value in (*coarse_expressions, *fine_expressions)
    ):
        raise ValueError(
            "Polynomial nested-reference actions cannot contain a rational quotient."
        )
    coarse_arguments = tuple(
        value for value in coarse_expressions if not isinstance(value, RationalPolynomial)
    )
    fine_arguments = tuple(
        value for value in fine_expressions if not isinstance(value, RationalPolynomial)
    )
    if (
        coarse_root.element_id != fine_root.element_id
        or coarse_root.topological_dimension != 2
        or coordinate_source_signature(coarse_root)
        != coordinate_source_signature(fine_root)
    ):
        raise ValueError(
            "Nonlinear charts must bind one live two-dimensional coordinate source root."
        )
    if any(sum(index) > 1 for value in coarse_arguments for index in value):
        _, expressions = coordinate_reference_chain(fine, ancestor=coarse)
        if any(isinstance(value, RationalPolynomial) for value in expressions):
            raise ValueError(
                "Polynomial nested-reference actions cannot contain a rational quotient."
            )
        return tuple(
            value for value in expressions if not isinstance(value, RationalPolynomial)
        )
    matrix = [
        [
            value.get(tuple(int(axis == column) for axis in range(2)), Fraction(0))
            for column in range(2)
        ]
        for value in coarse_arguments
    ]
    differences = tuple(
        add(value, constant(-old.get((0, 0), Fraction(0)), 2))
        for value, old in zip(fine_arguments, coarse_arguments, strict=True)
    )
    indices = sorted({index for value in differences for index in value})
    solved = _solve_exact(
        matrix,
        [[value.get(index, Fraction(0)) for index in indices] for value in differences],
    )
    arguments: tuple[Polynomial, ...] = tuple(
        {
            index: coefficient
            for index, coefficient in zip(indices, row, strict=True)
            if coefficient
        }
        for row in solved
    )
    if any(
        compose(value, arguments) != expected
        for value, expected in zip(coarse_arguments, fine_arguments, strict=True)
    ):
        raise ValueError(
            "The nonlinear DG chart is not an exact action of its declared coarse coordinate root."
        )
    return arguments


def _has_nonlinear_reference_action(element: CellGeometryElement) -> bool:
    """Classify the complete authored reference law, not its Python owner kind."""
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
    )
    from ._coordinate_enclosure import (
        _has_cartesian_reference_chain,
        coordinate_reference_chain,
        expression_parts,
        source_expressions,
    )

    if _has_cartesian_reference_chain(element):
        # A chain of physical affine restrictions remains an affine reference
        # action. In particular, the canonical collapsed-cube chart makes an
        # affine pyramid restriction look polynomial; that chart is the root
        # parameterization, not a nonlinear authored action.
        current = element
        while isinstance(current, RestrictedCellGeometryElement):
            current = current.source_element
        if not isinstance(
            current,
            (
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        ):
            return False
        _, arguments = coordinate_reference_chain(element)
    else:
        # Full coefficient actions are not Cartesian ancestry. Their existing
        # nested route is still checked against the actual whole physical maps.
        arguments = source_expressions(element)
        if arguments is None:
            raise ValueError(
                "Nested source actions require their complete actual scalar source law."
            )
    for value in arguments:
        numerator, denominator = expression_parts(
            value, reference_cell_topology(element.cell_kind).dimension
        )
        if any(sum(index) != 0 for index in denominator) or not denominator:
            return True
        if any(sum(index) > 1 for index in numerator):
            return True
    return False


def _rooted_nested_reference_pairs(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
) -> tuple[tuple[_NestedReferencePair | _PolynomialReferencePair, ...], float]:
    """Bind actual polynomial fine actions and their complete oriented reference tiling."""
    # Reference semantics belong to each actual block element; physical cell
    # coefficient banks are authenticated by the whole-map proof below.
    from ._cell_geometry_transfer import (
        _certify_nested_geometry_pairs,
        _mapped_geometry_cells,
        _mapped_root_identities,
        CellGeometryTransition,
        NestedReferenceWitnesses,
    )
    from ._coordinate_enclosure import (
        bernstein_coefficients,
        derivative,
        determinant,
        evaluate,
        rounded_point,
    )

    source_cells, target_cells = (
        _mapped_geometry_cells(source, source_geometry),
        _mapped_geometry_cells(target, target_geometry),
    )
    nonlinear = any(
        _has_nonlinear_reference_action(element) for element in target_geometry.elements
    ) or (
        (coarsening_witnesses is not None or geometry_transition is not None)
        and any(
            _has_nonlinear_reference_action(element)
            for element in source_geometry.elements
        )
    )
    if not nonlinear or (
        source.topology_id == target.topology_id
        and cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry)
    ):
        pairs = _nested_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
        )
        return pairs, _certify_nested_geometry_pairs(
            source, source_geometry, target, target_geometry, pairs
        )
    if (
        source.topological_dimension != 2
        or target.topological_dimension != 2
        or source.ambient_dimension != target.ambient_dimension
    ):
        raise ValueError(
            "Nonlinear reference transfer requires one common declared surface coordinate space."
        )
    source_ids = {
        int(value): index
        for index, value in enumerate(
            np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
        )
    }
    target_ids = {
        int(value): index
        for index, value in enumerate(
            np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
        )
    }
    records: list[tuple[int, int, bool, NDArray[np.float64] | None]] = []
    if geometry_transition is not None:
        if not isinstance(geometry_transition, CellGeometryTransition) or (
            geometry_transition.source_topology_id != source.topology_id
            or geometry_transition.target_topology_id != target.topology_id
            or geometry_transition.source_geometry_id != cell_geometry_id(source_geometry)
            or geometry_transition.target_geometry_id != cell_geometry_id(target_geometry)
        ):
            raise ValueError(
                "Nonlinear geometry transition does not bind the actual endpoint geometry."
            )
        for fine, coarse, vertices in zip(
            np.asarray(geometry_transition.target_cell_ids),
            np.asarray(geometry_transition.parent_cell_ids),
            np.asarray(geometry_transition.parent_reference_vertices),
            strict=True,
        ):
            if int(coarse) >= 0:
                if int(coarse) not in source_ids or int(fine) not in target_ids:
                    raise ValueError(
                        "A nonlinear refinement witness names absent scientific cells."
                    )
                records.append(
                    (
                        source_ids[int(coarse)],
                        target_ids[int(fine)],
                        False,
                        np.asarray(vertices, dtype=np.float64),
                    )
                )
        for fine, coarse, vertices in zip(
            np.asarray(geometry_transition.coarsened_cell_ids),
            np.asarray(geometry_transition.coarsened_into_ids),
            np.asarray(geometry_transition.coarsened_reference_vertices),
            strict=True,
        ):
            if int(fine) not in source_ids or int(coarse) not in target_ids:
                raise ValueError(
                    "A nonlinear coarsening witness names absent scientific cells."
                )
            records.append(
                (
                    source_ids[int(fine)],
                    target_ids[int(coarse)],
                    True,
                    np.asarray(vertices, dtype=np.float64),
                )
            )
    elif coarsening_witnesses is not None:
        if not isinstance(coarsening_witnesses, NestedReferenceWitnesses):
            raise TypeError(
                "Nonlinear coarsening requires owning NestedReferenceWitnesses."
            )
        for fine, coarse, vertices in zip(
            np.asarray(coarsening_witnesses.fine_cell_ids),
            np.asarray(coarsening_witnesses.coarse_cell_ids),
            np.asarray(coarsening_witnesses.fine_reference_vertices),
            strict=True,
        ):
            if int(fine) not in source_ids or int(coarse) not in target_ids:
                raise ValueError(
                    "A nonlinear coarsening witness names absent scientific cells."
                )
            records.append(
                (
                    source_ids[int(fine)],
                    target_ids[int(coarse)],
                    True,
                    np.asarray(vertices, dtype=np.float64),
                )
            )
    else:
        if parent_cells is None:
            raise ValueError(
                "Nonlinear refinement requires explicit source-parent cell indices."
            )
        parents = np.asarray(parent_cells)
        if (
            parents.shape != (len(target_cells),)
            or not np.issubdtype(parents.dtype, np.integer)
            or np.any(parents < 0)
            or np.any(parents >= len(source_cells))
        ):
            raise ValueError(
                "Nonlinear refinement requires one present integer source parent per target cell."
            )
        witnesses = (
            None
            if parent_reference_vertices is None
            else np.asarray(parent_reference_vertices, dtype=np.float64)
        )
        if witnesses is not None and (
            witnesses.ndim != 3
            or witnesses.shape[0] != len(target_cells)
            or witnesses.shape[2] != 2
        ):
            raise ValueError(
                "Nonlinear corner witnesses must align with actual target cells and reference dimension."
            )
        records = [
            (int(parent), index, False, None if witnesses is None else witnesses[index])
            for index, parent in enumerate(parents)
        ]
    if {row[0] for row in records} != set(range(len(source_cells))) or {
        row[1] for row in records
    } != set(range(len(target_cells))):
        raise ValueError(
            "Nonlinear reference witnesses leave source or target cells uncovered."
        )
    source_roots, target_roots = (
        _mapped_root_identities(source, source_geometry),
        _mapped_root_identities(target, target_geometry),
    )
    built: list[_PolynomialReferencePair] = []
    partitions: dict[tuple[bool, int], list[_ReferenceMap]] = {}
    for source_index, target_index, fine_is_source, witness in records:
        source_element, source_bank = source_cells[source_index]
        target_element, target_bank = target_cells[target_index]
        if (
            source_roots[source_index] != target_roots[target_index]
            or source_bank != target_bank
        ):
            raise ValueError(
                "Nonlinear reference actions have stale roots or changed actual source coefficients."
            )
        coarse, fine = (
            (target_element, source_element)
            if fine_is_source
            else (source_element, target_element)
        )
        arguments = _polynomial_reference_arguments(coarse, fine)
        vertices = reference_cell_topology(fine.cell_kind).vertices
        if witness is not None:
            exact = np.asarray(
                [
                    rounded_point(
                        tuple(
                            evaluate(value, tuple(Fraction(float(x)) for x in vertex))
                            for value in arguments
                        )
                    )
                    for vertex in vertices
                ],
                dtype=np.float64,
            )
            if (
                witness.ndim != 2
                or witness.shape[0] < len(vertices)
                or witness.shape[1] != 2
                or not np.array_equal(exact, witness[: len(vertices)])
            ):
                raise ValueError(
                    "A nonlinear corner witness differs from its owning exact polynomial action."
                )
        partitions.setdefault(
            (fine_is_source, target_index if fine_is_source else source_index), []
        ).append(_ReferenceMap(fine.cell_kind, arguments))
        jacobian = determinant(
            tuple(
                tuple(derivative(value, axis) for axis in range(2)) for value in arguments
            )
        )
        controls = bernstein_coefficients(
            jacobian, "box" if fine.cell_kind == "quadrilateral" else "simplex", 2
        )
        built.append(
            _PolynomialReferencePair(
                source_index,
                target_index,
                fine_is_source,
                arguments,
                source_element.cell_kind,
                target_element.cell_kind,
                (min(controls), max(controls)),
            )
        )
    for (fine_is_source, parent), maps in partitions.items():
        _require_reference_partition(
            (target_cells if fine_is_source else source_cells)[parent][0].cell_kind, maps
        )
    return tuple(built), 0.0
