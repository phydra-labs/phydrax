from __future__ import annotations

from fractions import Fraction

from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._global_embedding import _contact_is_declared


type _Point = tuple[Fraction, Fraction, Fraction]


def _point(x: int, y: int, z: int, /) -> _Point:
    return Fraction(x), Fraction(y), Fraction(z)


def _polyhedron(
    global_id: int,
    vertices: dict[int, _Point],
    faces: tuple[tuple[int, ...], ...],
    /,
) -> EmbeddingCell:
    vertex_ids = tuple(sorted(vertices))
    row = {identifier: index for index, identifier in enumerate(vertex_ids)}
    images = tuple(vertices[identifier] for identifier in vertex_ids)
    return EmbeddingCell(
        global_id=global_id,
        block_name="cells",
        local_index=global_id,
        kind="polyhedron",
        dimension=3,
        domain="polyhedron",
        vertex_indices=tuple(range(len(vertex_ids))),
        vertex_ids=vertex_ids,
        reference_vertices=(),
        coordinate_expressions=(),
        corner_images=images,
        affine_offset=None,
        affine_matrix=None,
        image_lower=tuple(
            float(min(point[axis] for point in images)) for axis in range(3)
        ),
        image_upper=tuple(
            float(max(point[axis] for point in images)) for axis in range(3)
        ),
        facet_routes=tuple(
            tuple(row[identifier] for identifier in face) for face in faces
        ),
    )


# Unit square z = 1 at ids 1..4, split into two triangles along 1-3.
_INTERFACE = {
    1: _point(0, 0, 1),
    2: _point(1, 0, 1),
    3: _point(1, 1, 1),
    4: _point(0, 1, 1),
}
_BOTTOM_APEX = {5: _point(0, 0, 0)}
_TOP_APEX = {6: _point(0, 0, 2)}


def test_coplanar_split_interface_declares_its_convex_union() -> None:
    lower = _polyhedron(
        10,
        {**_INTERFACE, **_BOTTOM_APEX},
        ((1, 2, 3), (1, 3, 4), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)),
    )
    upper = _polyhedron(
        11,
        {**_INTERFACE, **_TOP_APEX},
        ((1, 3, 2), (1, 4, 3), (6, 1, 2), (6, 2, 3), (6, 3, 4), (6, 4, 1)),
    )
    # The contact square's corners all lie in the union of the two triangles.
    contacts = tuple(_INTERFACE.values())
    assert (
        _contact_is_declared(lower, upper, contacts, lower.vertex_ids, upper.vertex_ids)
        is True
    )
    # A point on the shared plane outside the unit square is not declared.
    outside = (_point(2, 2, 1),)
    assert (
        _contact_is_declared(lower, upper, outside, lower.vertex_ids, upper.vertex_ids)
        is False
    )


def test_nonadjacent_shared_faces_are_not_a_declared_union() -> None:
    # Two shared triangles on different planes with no common edge: their vertex
    # union is the shared set, but contact need not be their face union.
    first_face = {1: _point(0, 0, 0), 2: _point(1, 0, 0), 3: _point(0, 1, 0)}
    second_face = {7: _point(0, 0, 3), 8: _point(1, 0, 3), 9: _point(0, 0, 4)}
    first = _polyhedron(
        20,
        {**first_face, **second_face},
        ((1, 2, 3), (7, 8, 9), (1, 7, 2), (2, 8, 3), (3, 9, 1)),
    )
    second = _polyhedron(
        21,
        {**first_face, **second_face},
        ((1, 3, 2), (7, 9, 8), (1, 2, 7), (2, 3, 8), (3, 1, 9)),
    )
    contact = (_point(0, 0, 1),)
    assert (
        _contact_is_declared(first, second, contact, first.vertex_ids, second.vertex_ids)
        is not True
    )


def test_extra_shared_vertex_outside_face_union_is_not_declared() -> None:
    lower = _polyhedron(
        30,
        {**_INTERFACE, **_BOTTOM_APEX},
        ((1, 2, 3), (1, 3, 4), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)),
    )
    # The upper cell also names the lower apex through a face the lower cell
    # lacks, so vertex 5 is shared outside every shared face.
    upper = _polyhedron(
        31,
        {**_INTERFACE, **_BOTTOM_APEX, **_TOP_APEX},
        ((1, 3, 2), (1, 4, 3), (6, 1, 2), (6, 2, 3), (6, 3, 4), (6, 4, 1), (5, 6, 1)),
    )
    contacts = (_point(0, 0, 1),)
    assert (
        _contact_is_declared(lower, upper, contacts, lower.vertex_ids, upper.vertex_ids)
        is False
    )


def _edge_touching_boxes(
    first_chain: tuple[tuple[int, int], ...],
    second_chain: tuple[tuple[int, int], ...],
) -> tuple[EmbeddingCell, EmbeddingCell]:
    low, high = first_chain[0][0], first_chain[-1][0]
    z_low, z_high = first_chain[0][1], first_chain[-1][1]
    first_ids = tuple(identifier for identifier, _ in first_chain)
    second_ids = tuple(identifier for identifier, _ in second_chain)
    first = _polyhedron(
        40,
        {
            **{identifier: _point(0, 0, z) for identifier, z in first_chain},
            100: _point(-1, -1, z_low),
            101: _point(0, -1, z_low),
            102: _point(-1, 0, z_low),
            103: _point(-1, -1, z_high),
            104: _point(0, -1, z_high),
            105: _point(-1, 0, z_high),
        },
        (
            (100, 102, low, 101),
            (103, 104, high, 105),
            (100, 101, 104, 103),
            (101, *first_ids, 104),
            (102, 105, *reversed(first_ids)),
            (100, 103, 105, 102),
        ),
    )
    second = _polyhedron(
        41,
        {
            **{identifier: _point(0, 0, z) for identifier, z in second_chain},
            200: _point(1, 0, z_low),
            201: _point(1, 1, z_low),
            202: _point(0, 1, z_low),
            203: _point(1, 0, z_high),
            204: _point(1, 1, z_high),
            205: _point(0, 1, z_high),
        },
        (
            (low, 202, 201, 200),
            (high, 203, 204, 205),
            (low, 200, 203, *reversed(second_ids[1:])),
            (200, 201, 204, 203),
            (202, 205, 204, 201),
            (*second_ids, 205, 202),
        ),
    )
    return first, second


def test_collinear_shared_edges_declare_their_complete_segment() -> None:
    chain = ((0, 0), (1, 1), (2, 2))
    first, second = _edge_touching_boxes(chain, chain)
    contacts = (
        _point(0, 0, 0),
        (Fraction(0), Fraction(0), Fraction(1, 2)),
        _point(0, 0, 2),
    )
    assert (
        _contact_is_declared(first, second, contacts, first.vertex_ids, second.vertex_ids)
        is True
    )
    assert (
        _contact_is_declared(
            first, second, (_point(0, 0, 3),), first.vertex_ids, second.vertex_ids
        )
        is False
    )


def test_disconnected_shared_edges_do_not_declare_their_convex_hull() -> None:
    first, second = _edge_touching_boxes(
        ((0, 0), (1, 1), (10, 2), (2, 3), (3, 4)),
        ((0, 0), (1, 1), (11, 2), (2, 3), (3, 4)),
    )
    # The intersection segment's extreme points are declared, but its middle
    # contains distinct vertex identities and no shared edge.
    assert (
        _contact_is_declared(
            first,
            second,
            (_point(0, 0, 0), _point(0, 0, 4)),
            first.vertex_ids,
            second.vertex_ids,
        )
        is False
    )
