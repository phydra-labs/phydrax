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


def test_extra_shared_vertex_declares_only_single_shared_strata() -> None:
    lower = _polyhedron(
        30,
        {**_INTERFACE, **_BOTTOM_APEX},
        ((1, 2, 3), (1, 3, 4), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)),
    )
    # The upper cell also names the lower apex through a face the lower cell
    # lacks, so vertex 5 is shared outside every shared face and the pair has
    # no single shared stratum: each contact must lie in one shared vertex,
    # edge or face.
    upper = _polyhedron(
        31,
        {**_INTERFACE, **_BOTTOM_APEX, **_TOP_APEX},
        ((1, 3, 2), (1, 4, 3), (6, 1, 2), (6, 2, 3), (6, 3, 4), (6, 4, 1), (5, 6, 1)),
    )

    def declared(points: tuple[_Point, ...], /) -> bool | None:
        return _contact_is_declared(
            lower, upper, points, lower.vertex_ids, upper.vertex_ids
        )

    assert declared((_point(0, 0, 1),)) is True
    assert declared((_point(0, 0, 1), _point(1, 0, 1), _point(1, 1, 1))) is True
    # Corners 2 and 4 share no face: their segment crosses two shared faces.
    assert declared((_point(1, 0, 1), _point(0, 1, 1))) is False
    assert declared((_point(0, 0, 2),)) is False


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


def test_bent_shared_edge_chain_declares_its_vertex_and_edge_union() -> None:
    # Two pyramids on opposite sides of z = 0 share only the bent chain
    # 1-2-3 (2 a hanging corner); their square bases have distinct far
    # corners 4 and 7, so no face, and no straight chain, is shared.
    chain = {1: _point(0, 0, 0), 2: _point(2, 0, 0), 3: _point(2, 2, 0)}
    upper = _polyhedron(
        50,
        {**chain, 4: _point(0, 2, 0), 5: _point(1, 1, 2)},
        ((1, 4, 3, 2), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1)),
    )
    lower = _polyhedron(
        51,
        {**chain, 7: _point(0, 2, 0), 8: _point(1, 1, -2)},
        ((1, 2, 3, 7), (8, 2, 1), (8, 3, 2), (8, 7, 3), (8, 1, 7)),
    )

    def declared(points: tuple[_Point, ...], /) -> bool | None:
        return _contact_is_declared(
            upper, lower, points, upper.vertex_ids, lower.vertex_ids
        )

    # Exactly the hanging corner, and contacts within one shared edge.
    assert declared((_point(2, 0, 0),)) is True
    assert declared((_point(0, 0, 0), _point(1, 0, 0), _point(2, 0, 0))) is True
    assert declared((_point(2, 0, 0), _point(2, 1, 0))) is True
    # Vertices on both edges span the triangle 1-2-3, which is not shared.
    assert declared((_point(1, 0, 0), _point(2, 1, 0))) is False
    # A genuine overlap next to the hanging corner, off the chain.
    near = (Fraction(19, 10), Fraction(1, 10), Fraction(0))
    assert declared((near,)) is False
    assert declared((_point(2, 0, 0), _point(1, 1, 0))) is False


# A non-convex pentagon 1-2-3-4-5 in z = 0, reflex at 2: its notch is the
# triangle 1-3-2, with 6 the midpoint of edge 1-3.
_NOTCHED = {
    1: _point(0, 0, 0),
    2: (Fraction(2), Fraction(1, 2), Fraction(0)),
    3: _point(4, 0, 0),
    4: _point(4, 4, 0),
    5: _point(0, 4, 0),
    6: _point(2, 0, 0),
}
_BELOW = {
    11: _point(0, 0, -1),
    12: _point(4, 0, -1),
    13: _point(4, 4, -1),
    14: _point(0, 4, -1),
}
_ABOVE = {
    21: _point(0, 0, 1),
    22: _point(4, 0, 1),
    23: _point(4, 4, 1),
    24: _point(0, 4, 1),
}
_NOTCH_POINT = (Fraction(2), Fraction(1, 4), Fraction(0))


def _upper_with_quad_notch() -> EmbeddingCell:
    return _polyhedron(
        71,
        {**_NOTCHED, **_ABOVE},
        (
            (5, 4, 3, 2, 1),
            (1, 2, 3, 6),
            (21, 22, 23, 24),
            (1, 6, 3, 22, 21),
            (3, 4, 23, 22),
            (4, 5, 24, 23),
            (5, 1, 21, 24),
        ),
    )


def test_nonconvex_shared_face_is_its_polygon_not_its_hull() -> None:
    # One shared stratum, the notched pentagon: the notch is not shared.
    lower = _polyhedron(
        72,
        {**{k: v for k, v in _NOTCHED.items() if k != 6}, **_BELOW},
        (
            (1, 2, 3, 4, 5),
            (1, 3, 2),
            (14, 13, 12, 11),
            (1, 11, 12, 3),
            (3, 12, 13, 4),
            (4, 13, 14, 5),
            (5, 14, 11, 1),
        ),
    )
    upper = _upper_with_quad_notch()
    assert (
        _contact_is_declared(
            lower, upper, (_NOTCH_POINT,), lower.vertex_ids, upper.vertex_ids
        )
        is False
    )
    assert (
        _contact_is_declared(
            lower, upper, (_point(2, 2, 0),), lower.vertex_ids, upper.vertex_ids
        )
        is True
    )


def test_notch_split_differently_is_not_a_declared_union() -> None:
    # The notch is split at the shared hanging vertex 6 differently in each
    # cell: no single shared stratum, and the notch interior is in none.
    lower = _polyhedron(
        70,
        {**_NOTCHED, **_BELOW},
        (
            (1, 2, 3, 4, 5),
            (1, 6, 2),
            (6, 3, 2),
            (14, 13, 12, 11),
            (1, 11, 12, 3, 6),
            (3, 12, 13, 4),
            (4, 13, 14, 5),
            (5, 14, 11, 1),
        ),
    )
    upper = _upper_with_quad_notch()

    def declared(points: tuple[_Point, ...], /) -> bool | None:
        return _contact_is_declared(
            lower, upper, points, lower.vertex_ids, upper.vertex_ids
        )

    assert declared((_NOTCH_POINT,)) is False
    assert declared((_point(2, 2, 0),)) is True
    assert declared((_NOTCHED[1], _NOTCHED[6])) is True


def test_one_undecided_shared_stratum_does_not_hide_another() -> None:
    # Shared triangles in z = 0 and y = 0 plus a shared vertex outside both.
    shared = {
        1: _point(0, 0, 0),
        2: _point(2, 0, 0),
        3: _point(0, 2, 0),
        4: _point(0, 0, 2),
        9: _point(5, 5, 5),
    }
    first = _polyhedron(
        60,
        {**shared, 10: _point(-1, -1, -1)},
        ((1, 2, 3), (1, 4, 2), (10, 1, 3), (10, 9, 1)),
    )
    second = _polyhedron(
        61,
        {**shared, 11: _point(3, 3, 3)},
        ((1, 3, 2), (1, 2, 4), (11, 2, 3), (11, 9, 2)),
    )
    for one, other in ((first, second), (second, first)):
        for points in (
            (shared[1], shared[2], shared[3]),
            (shared[1], shared[2], shared[4]),
            ((Fraction(1, 2), Fraction(0), Fraction(1, 2)),),
        ):
            assert (
                _contact_is_declared(one, other, points, one.vertex_ids, other.vertex_ids)
                is True
            )
