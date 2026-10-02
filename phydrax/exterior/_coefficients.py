#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization._cell_complex import CubicalCellComplex
from ..discretization._topology import CellComplexTopology
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    adjoint,
    ArraySpace,
    ComposedLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
)
from ..linalg._spaces import _coordinate_dtype
from ..sparse import EdgeRelation, route_reduce, SparseCoordinateOperator
from ..typing import Bool, checked, Dim, Float, Scalar


class CurvatureDegreeDim(Dim):
    pass


def _apply_coordinates(operator: AbstractLinearOperator, values: ArrayLike, /) -> Array:
    coordinates = jnp.asarray(values, dtype=_coordinate_dtype(operator.source))
    return operator.target.flatten(operator.mv(operator.source.unflatten(coordinates)))


def _transport_blocks(
    topology: CellComplexTopology, transports: Sequence[ArrayLike], /
) -> tuple[Array, ...]:
    values = tuple(jnp.asarray(value) for value in transports)
    if len(values) != topology.dimension:
        raise ValueError("One transport array is required for each incidence degree.")
    blocks: list[Array] = []
    for incidence, value in zip(topology.incidences, values, strict=True):
        if value.shape == incidence.relation.route_shape:
            value = value[..., None, None]
        if (
            value.ndim != 3
            or value.shape[0] != incidence.relation.route_shape[0]
            or min(value.shape[1:]) < 1
        ):
            raise ValueError(
                "Transports must have shape (routes,) or (routes, target fiber, source fiber)."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float64)
        value = eqx.error_if(
            value,
            jnp.any(incidence.relation.valid[:, None, None] & ~jnp.isfinite(value)),
            "Active coefficient transports must be finite.",
        )
        blocks.append(value)
    for lower, upper in zip(blocks[:-1], blocks[1:], strict=True):
        if lower.shape[1] != upper.shape[2]:
            raise ValueError(
                "Consecutive coefficient fibers must agree at their common degree."
            )
    return tuple(blocks)


def _curvature_routes(
    topology: CellComplexTopology, /
) -> tuple[tuple[Array, Array, EdgeRelation, EdgeRelation], ...]:
    plans: list[tuple[Array, Array, EdgeRelation, EdgeRelation]] = []
    for lower, upper in zip(
        topology.incidences[:-1], topology.incidences[1:], strict=True
    ):
        low_valid = np.asarray(lower.relation.valid)
        up_valid = np.asarray(upper.relation.valid)
        low_targets = np.asarray(lower.relation.target_indices)
        up_sources = np.asarray(upper.relation.source_indices)
        groups: dict[int, list[int]] = {}
        for index in np.flatnonzero(low_valid):
            groups.setdefault(int(low_targets[index]), []).append(int(index))
        left: list[int] = []
        right: list[int] = []
        for index in np.flatnonzero(up_valid):
            for low_index in groups.get(int(up_sources[index]), ()):
                left.append(low_index)
                right.append(int(index))
        left_host = np.asarray(left, dtype=np.int32)
        right_host = np.asarray(right, dtype=np.int32)
        endpoints = np.stack(
            (
                np.asarray(lower.relation.source_indices)[left_host],
                np.asarray(upper.relation.target_indices)[right_host],
            ),
            axis=1,
        )
        unique, inverse = np.unique(endpoints, axis=0, return_inverse=True)
        relation = EdgeRelation(
            unique[:, 0],
            unique[:, 1],
            source_size=lower.relation.source_size,
            target_size=upper.relation.target_size,
        )
        reduction = EdgeRelation(
            np.arange(left_host.size, dtype=np.int32),
            inverse.astype(np.int32),
            source_size=left_host.size,
            target_size=unique.shape[0],
        )
        plans.append(
            (jnp.asarray(left_host), jnp.asarray(right_host), reduction, relation)
        )
    return tuple(plans)


def _coefficient_admission(
    topology: CellComplexTopology,
    blocks: tuple[Array, ...],
    spaces: Sequence[AbstractVectorSpace] | None,
    key: str | None,
    /,
) -> tuple[tuple[int, ...], tuple[AbstractVectorSpace, ...] | None, str, DTypeLike]:
    """Bind fiber ranks and scientific identity before constructing operators."""
    ranks = (
        ((blocks[0].shape[2],) + tuple(value.shape[1] for value in blocks))
        if blocks
        else (1,)
    )
    if key is None:
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(blocks)):
            raise ValueError(
                "Traced coefficient construction requires an explicit stable key."
            )
        numeric_identity: str | dict[str, object] = array_tree_fingerprint(blocks)
    else:
        if not key:
            raise ValueError("key must be nonempty.")
        numeric_identity = key
    provided_spaces = None if spaces is None else tuple(spaces)
    if provided_spaces is not None and any(
        not isinstance(space, AbstractVectorSpace) for space in provided_spaces
    ):
        raise TypeError("spaces must contain AbstractVectorSpace values.")
    identity = canonical_fingerprint(
        {
            "kind": "coefficient-system",
            "topology": topology.topology_id,
            "ranks": ranks,
            "binding": numeric_identity,
            "spaces": None
            if provided_spaces is None
            else tuple(space.space_id for space in provided_spaces),
        }
    )
    dtype = jnp.result_type(*(value.dtype for value in blocks), jnp.float64)
    return ranks, provided_spaces, identity, dtype


def _coefficient_spaces(
    topology: CellComplexTopology,
    ranks: tuple[int, ...],
    provided_spaces: tuple[AbstractVectorSpace, ...] | None,
    identity: str,
    dtype: DTypeLike,
    /,
) -> tuple[AbstractVectorSpace, ...]:
    """Admit coordinate spaces against the already bound cellular fibers."""
    spaces = (
        provided_spaces
        if provided_spaces is not None
        else tuple(
            ArraySpace(
                (entities.count * rank,),
                dtype=dtype,
                space_id=canonical_fingerprint({"system": identity, "degree": degree}),
            )
            for degree, (entities, rank) in enumerate(
                zip(topology.entity_sets, ranks, strict=True)
            )
        )
    )
    if len(spaces) != len(ranks) or any(
        not isinstance(space, AbstractVectorSpace) for space in spaces
    ):
        raise TypeError("spaces must provide an AbstractVectorSpace for each degree.")
    for entities, rank, space in zip(topology.entity_sets, ranks, spaces, strict=True):
        if space.size != entities.count * rank:
            raise ValueError(
                "Coefficient space size must equal cell count times fiber rank."
            )
    return spaces


@final
class CoefficientSystem(StrictModule):
    """Linear cellular coefficient routes, including curved connections.

    A transport maps a lower-cell fiber to an upper-cell fiber; incidence signs
    are supplied by the topology, not encoded a second time in transports.
    Curved systems deliberately do not claim to be de Rham/Hilbert complexes.
    ``key`` names a stable numerical binding when construction is traced.
    """

    topology: CellComplexTopology
    differentials: tuple[SparseCoordinateOperator, ...]
    spaces: tuple[AbstractVectorSpace, ...]
    curvature_routes: tuple[tuple[Array, Array, EdgeRelation, EdgeRelation], ...]
    curvature_templates: tuple[SparseCoordinateOperator, ...]
    fiber_ranks: tuple[int, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: CellComplexTopology,
        transports: Sequence[ArrayLike],
        /,
        *,
        spaces: Sequence[AbstractVectorSpace] | None = None,
        key: str | None = None,
    ) -> None:
        blocks = _transport_blocks(topology, transports)
        ranks, provided_spaces, identity, dtype = _coefficient_admission(
            topology, blocks, spaces, key
        )
        spaces_ = _coefficient_spaces(topology, ranks, provided_spaces, identity, dtype)
        differentials = tuple(
            SparseCoordinateOperator(
                incidence.relation,
                value * incidence.signs[:, None, None],
                source=spaces_[degree],
                target=spaces_[degree + 1],
                block_shape=(value.shape[1], value.shape[2]),
                accumulation_dtype=jnp.result_type(
                    value.dtype,
                    _coordinate_dtype(spaces_[degree]),
                    _coordinate_dtype(spaces_[degree + 1]),
                ),
                operator_id=canonical_fingerprint(
                    {"system": identity, "differential": degree}
                ),
            )
            for degree, (incidence, value) in enumerate(
                zip(topology.incidences, blocks, strict=True)
            )
        )
        curvature_routes = _curvature_routes(topology)
        curvature_templates = tuple(
            SparseCoordinateOperator(
                relation,
                jnp.zeros(
                    relation.route_shape + (ranks[degree + 2], ranks[degree]), dtype=dtype
                ),
                source=spaces_[degree],
                target=spaces_[degree + 2],
                block_shape=(ranks[degree + 2], ranks[degree]),
                accumulation_dtype=jnp.result_type(
                    dtype,
                    _coordinate_dtype(spaces_[degree]),
                    _coordinate_dtype(spaces_[degree + 2]),
                ),
                operator_id=canonical_fingerprint(
                    {"system": identity, "curvature": degree}
                ),
            )
            for degree, (_, _, _, relation) in enumerate(curvature_routes)
        )
        self.topology = topology
        self.differentials = differentials
        self.spaces = spaces_
        self.curvature_routes = curvature_routes
        self.curvature_templates = curvature_templates
        self.fiber_ranks = ranks
        self.system_id = identity

    @property
    def dimension(self) -> int:
        return self.topology.dimension

    def refresh(self, transports: Sequence[ArrayLike], /) -> CoefficientSystem:
        """Refresh numerical leaves without changing relation or binding identity."""
        blocks = _transport_blocks(self.topology, transports)
        coefficients = tuple(
            value * incidence.signs[:, None, None]
            for value, incidence in zip(blocks, self.topology.incidences, strict=True)
        )
        for operator, value in zip(self.differentials, coefficients, strict=True):
            if (
                value.shape != operator.coefficients.shape
                or value.dtype != operator.coefficients.dtype
            ):
                raise ValueError("Refresh must preserve coefficient shape and dtype.")
        return eqx.tree_at(
            lambda system: tuple(
                operator.coefficients for operator in system.differentials
            ),
            self,
            coefficients,
        )

    def exterior_derivative(self, degree: int, values: ArrayLike, /) -> Array:
        return _apply_coordinates(twisted_differential(self, degree), values)

    def codifferential(self, degree: int, values: ArrayLike, /) -> Array:
        return _apply_coordinates(adjoint(twisted_differential(self, degree - 1)), values)

    def chain_residual(self, degree: int, values: ArrayLike, /) -> Array:
        return self.exterior_derivative(
            degree + 1, self.exterior_derivative(degree, values)
        )

    def hodge_laplacian(self, degree: int, values: ArrayLike, /) -> Array:
        return _apply_coordinates(sheaf_laplacian(self, degree), values)


def twisted_differential(
    system: CoefficientSystem, degree: int, /
) -> SparseCoordinateOperator:
    if not isinstance(system, CoefficientSystem):
        raise TypeError("system must be CoefficientSystem.")
    if degree < 0 or degree >= system.dimension:
        raise ValueError("Differential degree must be below the top degree.")
    return system.differentials[degree]


@final
class CurvatureEvidence(StrictModule):
    """Actual signed two-step route compositions and their coordinate norms."""

    __strict_contract__ = True

    operators: tuple[SparseCoordinateOperator, ...]
    residual_norms: Float[CurvatureDegreeDim]
    flat: Bool[Scalar]
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: tuple[SparseCoordinateOperator, ...],
        system_id: str,
        tolerance: float,
        /,
    ) -> None:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Curvature tolerance must be finite and nonnegative.")
        norms = (
            jnp.stack(
                tuple(jnp.linalg.norm(operator.coefficients) for operator in operators)
            )
            if operators
            else jnp.zeros((0,), dtype=jnp.float64)
        )
        self.operators = operators
        self.residual_norms = norms
        self.flat = jnp.all(norms <= tolerance)
        self.system_id = system_id


def curvature_evidence(
    system: CoefficientSystem, /, *, tolerance: float = 1e-12
) -> CurvatureEvidence:
    if not isinstance(system, CoefficientSystem):
        raise TypeError("system must be CoefficientSystem.")
    operators: list[SparseCoordinateOperator] = []
    for degree, (left, right, reduction, _) in enumerate(system.curvature_routes):
        lower = system.differentials[degree]
        upper = system.differentials[degree + 1]
        blocks = route_reduce(
            reduction, upper.coefficients[right] @ lower.coefficients[left]
        )
        operators.append(
            eqx.tree_at(
                lambda operator: operator.coefficients,
                system.curvature_templates[degree],
                blocks,
            )
        )
    return CurvatureEvidence(tuple(operators), system.system_id, tolerance)


def sheaf_laplacian(system: CoefficientSystem, degree: int, /) -> AbstractLinearOperator:
    if not isinstance(system, CoefficientSystem):
        raise TypeError("system must be CoefficientSystem.")
    if degree < 0 or degree > system.dimension:
        raise ValueError("Laplacian degree must lie in the coefficient system.")
    terms: list[AbstractLinearOperator] = []
    if degree > 0:
        lower = twisted_differential(system, degree - 1)
        terms.append(ComposedLinearOperator(lower, adjoint(lower)))
    if degree < system.dimension:
        upper = twisted_differential(system, degree)
        terms.append(ComposedLinearOperator(adjoint(upper), upper))
    if not terms:
        return ScaledLinearOperator(IdentityLinearOperator(system.spaces[0]), 0.0)
    return terms[0] if len(terms) == 1 else SumLinearOperator(terms[0], terms[1])


def _cubical_wrap_routes(cubical: CubicalCellComplex, /) -> tuple[np.ndarray, ...]:
    offsets = cubical.orientation_offsets
    wraps: list[np.ndarray] = []
    for degree, incidence in enumerate(cubical.topology.incidences):
        relation = incidence.relation
        source = np.asarray(relation.source_indices)
        target = np.asarray(relation.target_indices)
        valid = np.asarray(relation.valid)
        signs = np.asarray(incidence.signs)
        value = np.zeros((source.size, len(cubical.shape)), dtype=np.int32)
        for upper_orientation_index, upper_axes in enumerate(
            cubical.orientations[degree + 1]
        ):
            upper_start = offsets[degree + 1][upper_orientation_index]
            upper_end = upper_start + int(
                np.prod(cubical.orientation_shapes[degree + 1][upper_orientation_index])
            )
            upper_routes = valid & (target >= upper_start) & (target < upper_end)
            for position, axis in enumerate(upper_axes):
                if not cubical.periodic[axis]:
                    continue
                lower_axes = upper_axes[:position] + upper_axes[position + 1 :]
                lower_index = cubical.orientations[degree].index(lower_axes)
                lower_start = offsets[degree][lower_index]
                lower_end = lower_start + int(
                    np.prod(cubical.orientation_shapes[degree][lower_index])
                )
                selected = np.flatnonzero(
                    upper_routes
                    & (source >= lower_start)
                    & (source < lower_end)
                    & (signs == (-1) ** position)
                )
                upper_coordinates = np.asarray(cubical.cell_multi_indices[degree + 1])[
                    target[selected]
                ]
                value[
                    selected[upper_coordinates[:, axis] == cubical.shape[axis] - 1], axis
                ] = 1
        wraps.append(value)
    return tuple(wraps)


def bloch_coefficient_system(
    cubical: CubicalCellComplex,
    wavevector: ArrayLike,
    /,
    *,
    periods: ArrayLike | None = None,
    spaces: Sequence[AbstractVectorSpace] | None = None,
    key: str | None = None,
) -> CoefficientSystem:
    """Quasi-periodic quotient routes with nontrivial wrap holonomy.

    ``wavevector`` has inverse-length units; default periods are the cell counts
    (unit spacing). Values on a wrapped upper face acquire exp(+i k L).
    """
    if not isinstance(cubical, CubicalCellComplex):
        raise TypeError("cubical must be CubicalCellComplex.")
    if not all(cubical.periodic):
        raise ValueError("Bloch coefficients require periodic axes.")
    wave = jnp.asarray(wavevector, dtype=jnp.float64)
    lengths = jnp.asarray(
        cubical.shape if periods is None else periods, dtype=jnp.float64
    )
    if wave.shape != (cubical.topology.dimension,) or lengths.shape != wave.shape:
        raise ValueError("Wavevector and periods require one entry per cubical axis.")
    wave = eqx.error_if(wave, jnp.any(~jnp.isfinite(wave)), "Wavevector must be finite.")
    lengths = eqx.error_if(
        lengths,
        jnp.any(~jnp.isfinite(lengths)) | jnp.any(lengths <= 0),
        "Periods must be finite and positive.",
    )
    transports = tuple(
        jnp.exp(1j * (jnp.asarray(wrap) @ (wave * lengths)))
        for wrap in _cubical_wrap_routes(cubical)
    )
    return CoefficientSystem(cubical.topology, transports, spaces=spaces, key=key)


def _coface_stars(
    topology: CellComplexTopology, coface_degree: int, /, *, lowest_degree: int = 0
) -> tuple[tuple[set[int], ...], ...]:
    """Close each cell's selected-degree cofaces through native incidences."""
    stars: list[tuple[set[int], ...]] = [
        tuple() for _ in topology.entity_sets[: coface_degree + 1]
    ]
    stars[-1] = tuple({cell} for cell in range(topology.entity_sets[coface_degree].count))
    for degree in range(coface_degree - 1, lowest_degree - 1, -1):
        current = tuple(set() for _ in range(topology.entity_sets[degree].count))
        incidence = topology.incidences[degree]
        for source, target, valid in zip(
            np.asarray(incidence.relation.source_indices),
            np.asarray(incidence.relation.target_indices),
            np.asarray(incidence.relation.valid),
            strict=True,
        ):
            if valid:
                current[int(source)].update(stars[degree + 1][int(target)])
        stars[degree] = current
    return tuple(stars)


def _orientation_dual_edges(
    topology: CellComplexTopology, /
) -> list[tuple[int, int, int, int]]:
    """Admit manifold top-face gluings and their orientation character."""
    top = topology.incidences[-1]
    face_routes: dict[int, list[tuple[int, int]]] = {}
    for lower, upper, sign, valid in zip(
        np.asarray(top.relation.source_indices),
        np.asarray(top.relation.target_indices),
        np.asarray(top.signs),
        np.asarray(top.relation.valid),
        strict=True,
    ):
        if valid:
            face_routes.setdefault(int(lower), []).append((int(upper), int(sign)))
    dual_edges: list[tuple[int, int, int, int]] = []
    for face, occurrences in face_routes.items():
        if len(occurrences) > 2:
            raise ValueError("Orientation coefficients require manifold top-face stars.")
        if len(occurrences) == 2:
            (left, a), (right, b) = occurrences
            dual_edges.append((face, left, right, -a * b))
    return dual_edges


def _orientation_face_membership(
    topology: CellComplexTopology, degree: int, cell_count: int, /
) -> tuple[set[int], ...]:
    if degree == topology.dimension:
        return tuple(set() for _ in range(cell_count))
    if degree == topology.dimension - 1:
        return tuple({cell} for cell in range(cell_count))
    return _coface_stars(topology, topology.dimension - 1, lowest_degree=degree)[degree]


def _extend_orientation_frame(
    frame: dict[int, int], pending: list[int], neighbor: int, expected: int, /
) -> None:
    """Require path-independent local orientation before extending a frame."""
    if neighbor in frame:
        if frame[neighbor] != expected:
            raise ValueError(
                "A cell star has orientation holonomy; subdivide nonregular attaching cells first."
            )
    else:
        frame[neighbor] = expected
        pending.append(neighbor)


def _walk_orientation_star(
    star: set[int], faces: set[int], dual_edges: list[tuple[int, int, int, int]], /
) -> dict[int, int]:
    if not star:
        raise ValueError("Orientation system requires a pure manifold cell complex.")
    frame = {min(star): 1}
    pending = [min(star)]
    while pending:
        top_cell = pending.pop()
        for face, left, right, parity in dual_edges:
            if face not in faces:
                continue
            if left == top_cell:
                neighbor = right
            elif right == top_cell:
                neighbor = left
            else:
                continue
            _extend_orientation_frame(frame, pending, neighbor, frame[top_cell] * parity)
    if set(frame) != star:
        raise ValueError("Orientation system requires connected local manifold stars.")
    return frame


def _orientation_frames(
    topology: CellComplexTopology, /
) -> tuple[tuple[dict[int, int], ...], ...]:
    stars = _coface_stars(topology, topology.dimension)
    dual_edges = _orientation_dual_edges(topology)
    frames: list[tuple[dict[int, int], ...]] = []
    for degree, degree_stars in enumerate(stars):
        face_membership = _orientation_face_membership(
            topology, degree, len(degree_stars)
        )
        degree_frames = tuple(
            _walk_orientation_star(star, face_membership[cell], dual_edges)
            for cell, star in enumerate(degree_stars)
        )
        frames.append(degree_frames)
    return tuple(frames)


def _polygon_orientation_system(topology: CellComplexTopology, /) -> CoefficientSystem:
    """Orientation character for a one-vertex polygon presentation of a surface.

    Paired polygon sides with equal incidence signs reverse local orientation;
    opposite signs preserve it. Attaching occurrences remain separate routes.
    """
    edges, faces = topology.incidences
    edge_targets = np.asarray(edges.relation.target_indices)
    edge_signs = np.asarray(edges.signs)
    edge_valid = np.asarray(edges.relation.valid)
    face_sources = np.asarray(faces.relation.source_indices)
    face_signs = np.asarray(faces.signs)
    face_valid = np.asarray(faces.relation.valid)
    edge_maps = np.ones(edges.relation.route_shape, dtype=np.float64)
    face_maps = np.ones(faces.relation.route_shape, dtype=np.float64)
    for edge in range(topology.entity_sets[1].count):
        endpoints = np.flatnonzero(edge_valid & (edge_targets == edge))
        occurrences = np.flatnonzero(face_valid & (face_sources == edge))
        if (
            endpoints.size != 2
            or occurrences.size != 2
            or set(edge_signs[endpoints]) != {-1.0, 1.0}
        ):
            raise ValueError(
                "A surface polygon presentation requires paired sides and closed oriented edges."
            )
        orientation_character = -face_signs[occurrences[0]] * face_signs[occurrences[1]]
        edge_maps[endpoints[edge_signs[endpoints] > 0]] = orientation_character
        face_maps[occurrences[1]] = orientation_character
    return CoefficientSystem(topology, (edge_maps, face_maps))


def orientation_coefficient_system(topology: CellComplexTopology, /) -> CoefficientSystem:
    """Canonical orientation local system from signed manifold face gluings."""
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be CellComplexTopology.")
    if topology.dimension == 0:
        return CoefficientSystem(topology, ())
    if (
        topology.dimension == 2
        and topology.entity_sets[0].count == topology.entity_sets[2].count == 1
    ):
        return _polygon_orientation_system(topology)
    frames = _orientation_frames(topology)
    transports: list[np.ndarray] = []
    for degree, incidence in enumerate(topology.incidences):
        value = np.ones(incidence.relation.route_shape, dtype=np.float64)
        for route, (lower, upper, valid) in enumerate(
            zip(
                np.asarray(incidence.relation.source_indices),
                np.asarray(incidence.relation.target_indices),
                np.asarray(incidence.relation.valid),
                strict=True,
            )
        ):
            if valid:
                lower_frame = frames[degree][int(lower)]
                upper_frame = frames[degree + 1][int(upper)]
                common = min(upper_frame)
                value[route] = lower_frame[common] * upper_frame[common]
        transports.append(value)
    return CoefficientSystem(topology, transports)


__all__ = [
    "CoefficientSystem",
    "CurvatureEvidence",
    "twisted_differential",
    "curvature_evidence",
    "bloch_coefficient_system",
    "orientation_coefficient_system",
    "sheaf_laplacian",
]
