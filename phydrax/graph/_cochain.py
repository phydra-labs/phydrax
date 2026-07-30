#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric cochain complexes over the canonical sparse :class:`GraphIR`."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array
from scipy.sparse.linalg import eigsh

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._ir import GraphIR


CochainBoundaryKind: TypeAlias = Literal["absolute", "relative"]
CochainSide: TypeAlias = Literal["primal", "dual"]
CochainCellOrientation: TypeAlias = Literal["invariant", "signed"]
CochainSampling: TypeAlias = Literal["point_value", "cell_average", "cell_integral"]


class CochainFieldSpec(StrictModule):
    """Discrete differential-form semantics shared by fields and operators."""

    degree: int = eqx.field(static=True)
    complex_side: CochainSide = eqx.field(static=True)
    cell_orientation: CochainCellOrientation = eqx.field(static=True)
    sampling: CochainSampling = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        complex_side: CochainSide = "primal",
        cell_orientation: CochainCellOrientation,
        sampling: CochainSampling,
    ):
        resolved_degree = int(degree)
        if resolved_degree < 0:
            raise ValueError("Cochain degree must be non-negative.")
        if complex_side not in ("primal", "dual"):
            raise ValueError("complex_side must be 'primal' or 'dual'.")
        if cell_orientation not in ("invariant", "signed"):
            raise ValueError("cell_orientation must be 'invariant' or 'signed'.")
        if sampling not in ("point_value", "cell_average", "cell_integral"):
            raise ValueError(
                "sampling must be 'point_value', 'cell_average', or 'cell_integral'."
            )
        self.degree = resolved_degree
        self.complex_side = complex_side
        self.cell_orientation = cell_orientation
        self.sampling = sampling

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible cochain semantics."""
        return {
            "degree": self.degree,
            "complex_side": self.complex_side,
            "cell_orientation": self.cell_orientation,
            "sampling": self.sampling,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "CochainFieldSpec":
        """Restore cochain semantics from a canonical dictionary."""
        return cls(
            int(value["degree"]),
            complex_side=value.get("complex_side", "primal"),
            cell_orientation=value["cell_orientation"],
            sampling=value["sampling"],
        )



def _host_array(name: str, value: Any, /, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _array_digest(digest: Any, value: Any, /) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))


def _fingerprint(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
        elif isinstance(value, tuple):
            digest.update(repr(len(value)).encode("utf-8"))
            for item in value:
                _array_digest(digest, item)
        else:
            _array_digest(digest, value)
    return digest.hexdigest()


class CochainBoundaryPolicy(StrictModule, NonTrainableState):
    """Boundary realization used by metric cochain operators."""

    kind: CochainBoundaryKind = eqx.field(static=True)

    def __init__(self, kind: CochainBoundaryKind = "absolute"):
        if kind not in ("absolute", "relative"):
            raise ValueError("Cochain boundary policy must be 'absolute' or 'relative'.")
        self.kind = kind

    @property
    def code(self) -> int:
        return 0 if self.kind == "absolute" else 1


class CochainIncidence(StrictModule, NonTrainableState):
    """Sparse signed boundary incidence ``B_degree`` in COO form."""

    degree: int = eqx.field(static=True)
    lower_count: int = eqx.field(static=True)
    upper_count: int = eqx.field(static=True)
    lower_indices: Array
    upper_indices: Array
    signs: Array

    def __init__(
        self,
        degree: int,
        lower_count: int,
        upper_count: int,
        lower_indices: Any,
        upper_indices: Any,
        signs: Any,
        /,
    ):
        resolved_degree = int(degree)
        lower_size = int(lower_count)
        upper_size = int(upper_count)
        if resolved_degree <= 0:
            raise ValueError("Cochain incidence degree must be positive.")
        if lower_size < 0 or upper_size < 0:
            raise ValueError("Cochain cell counts must be non-negative.")
        lower = np.asarray(lower_indices)
        upper = np.asarray(upper_indices)
        coefficient = _host_array("incidence signs", signs, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1 or coefficient.ndim != 1:
            raise ValueError("Cochain incidence arrays must be rank-1.")
        if lower.shape != upper.shape or lower.shape != coefficient.shape:
            raise ValueError("Cochain incidence arrays must have identical shapes.")
        if not np.issubdtype(lower.dtype, np.integer) or not np.issubdtype(
            upper.dtype, np.integer
        ):
            raise TypeError("Cochain incidence indices must have integer dtype.")
        lower = lower.astype(np.int32, copy=False)
        upper = upper.astype(np.int32, copy=False)
        if np.any(lower < 0) or np.any(lower >= lower_size):
            raise ValueError("Lower incidence indices are out of range.")
        if np.any(upper < 0) or np.any(upper >= upper_size):
            raise ValueError("Upper incidence indices are out of range.")
        if np.any(np.abs(coefficient) != 1.0):
            raise ValueError("Cell-complex incidence coefficients must be ±1.")
        pairs = np.stack((lower, upper), axis=1)
        if pairs.shape[0] and np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("Cochain incidence pairs must be unique.")
        self.degree = resolved_degree
        self.lower_count = lower_size
        self.upper_count = upper_size
        self.lower_indices = jnp.asarray(lower)
        self.upper_indices = jnp.asarray(upper)
        self.signs = jnp.asarray(coefficient)

    @classmethod
    def from_dense(cls, degree: int, matrix: Any, /) -> "CochainIncidence":
        """Construct one incidence from a dense lower-by-upper boundary matrix."""
        dense = _host_array("boundary matrix", matrix, dtype=float)
        if dense.ndim != 2:
            raise ValueError("Boundary matrices must be rank-2.")
        lower, upper = np.nonzero(dense)
        return cls(
            degree,
            dense.shape[0],
            dense.shape[1],
            lower,
            upper,
            dense[lower, upper],
        )

    def scipy_matrix(self) -> sp.csr_matrix:
        """Return the host-side sparse boundary matrix."""
        return sp.coo_matrix(
            (
                np.asarray(self.signs),
                (np.asarray(self.lower_indices), np.asarray(self.upper_indices)),
            ),
            shape=(self.lower_count, self.upper_count),
        ).tocsr()


class HarmonicSubspace(StrictModule, NonTrainableState):
    """Degree-wise metric-orthonormal bases for exact Hodge kernels."""

    bases: tuple[Array, ...]
    eigenvalues: tuple[Array, ...]
    ranks: tuple[int, ...] = eqx.field(static=True)
    max_modes: int = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)
    complex_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        bases: Sequence[Any],
        eigenvalues: Sequence[Any],
        ranks: Sequence[int],
        /,
        *,
        max_modes: int,
        boundary_policy: CochainBoundaryKind,
        complex_fingerprint: str,
    ):
        basis_tuple = tuple(jnp.asarray(value) for value in bases)
        eigenvalue_tuple = tuple(jnp.asarray(value) for value in eigenvalues)
        rank_tuple = tuple(int(value) for value in ranks)
        if len(basis_tuple) != len(rank_tuple) or len(eigenvalue_tuple) != len(rank_tuple):
            raise ValueError("Harmonic basis, eigenvalue, and rank counts must match.")
        if int(max_modes) < 0 or any(rank < 0 or rank > int(max_modes) for rank in rank_tuple):
            raise ValueError("Harmonic ranks must lie in [0, max_modes].")
        for basis, values, rank in zip(
            basis_tuple, eigenvalue_tuple, rank_tuple, strict=True
        ):
            if basis.ndim != 2 or int(basis.shape[1]) != int(max_modes):
                raise ValueError("Harmonic bases must have shape (cells, max_modes).")
            if values.shape != (int(max_modes),):
                raise ValueError("Harmonic eigenvalues must have shape (max_modes,).")
            if rank and bool(jnp.any(~jnp.isfinite(basis[:, :rank]))):
                raise ValueError("Harmonic bases must be finite.")
        if boundary_policy not in ("absolute", "relative"):
            raise ValueError("Unknown harmonic boundary policy.")
        if not str(complex_fingerprint):
            raise ValueError("Harmonic complex fingerprint must not be empty.")
        self.bases = basis_tuple
        self.eigenvalues = eigenvalue_tuple
        self.ranks = rank_tuple
        self.max_modes = int(max_modes)
        self.boundary_policy = boundary_policy
        self.complex_fingerprint = str(complex_fingerprint)


class CochainComplexIR(StrictModule, NonTrainableState):
    """Validated oriented metric cell complex backed by one canonical ``GraphIR``."""

    graph: GraphIR
    incidences: tuple[CochainIncidence, ...]
    hodge_stars: tuple[Array, ...]
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    boundary_masks: tuple[Array, ...]
    coordinates: tuple[Array | None, ...]
    harmonic_subspace: HarmonicSubspace | None
    cell_counts: tuple[int, ...] = eqx.field(static=True)
    cell_offsets: tuple[int, ...] = eqx.field(static=True)
    incidence_fingerprint: str = eqx.field(static=True)
    metric_fingerprint: str = eqx.field(static=True)
    boundary_fingerprint: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        cell_counts: Sequence[int],
        incidences: Sequence[CochainIncidence],
        hodge_stars: Sequence[Any],
        /,
        *,
        primal_measures: Sequence[Any] | None = None,
        dual_measures: Sequence[Any] | None = None,
        boundary_masks: Sequence[Any] | None = None,
        coordinates: Sequence[Any | None] | None = None,
        harmonic_subspace: HarmonicSubspace | None = None,
        validate: bool = True,
    ):
        counts = tuple(int(value) for value in cell_counts)
        if not counts or any(value <= 0 for value in counts):
            raise ValueError("Cochain complexes require positive cell counts by degree.")
        max_degree = len(counts) - 1
        incidence_tuple = tuple(incidences)
        if len(incidence_tuple) != max_degree:
            raise ValueError("One incidence is required between every consecutive degree.")
        for degree, incidence in enumerate(incidence_tuple, start=1):
            if not isinstance(incidence, CochainIncidence):
                raise TypeError("incidences must contain CochainIncidence objects.")
            if (
                incidence.degree != degree
                or incidence.lower_count != counts[degree - 1]
                or incidence.upper_count != counts[degree]
            ):
                raise ValueError("Incidence degrees and dimensions must match cell counts.")

        stars = self._degree_values("hodge_stars", hodge_stars, counts, positive=True)
        primal = self._degree_values(
            "primal_measures",
            tuple(jnp.ones((count,)) for count in counts)
            if primal_measures is None
            else primal_measures,
            counts,
            positive=True,
        )
        dual = self._degree_values(
            "dual_measures",
            tuple(np.asarray(primal[k]) * np.asarray(stars[k]) for k in range(len(counts)))
            if dual_measures is None
            else dual_measures,
            counts,
            positive=True,
        )
        masks = self._boundary_values(boundary_masks, counts)
        points = self._coordinate_values(coordinates, counts)
        offsets = tuple(np.cumsum((0,) + counts[:-1], dtype=np.int64).tolist())

        incidence_fingerprint = self._incidence_fingerprint(counts, incidence_tuple)
        metric_fingerprint = _fingerprint(tuple(stars), tuple(primal), tuple(dual))
        boundary_fingerprint = _fingerprint(tuple(masks))
        fingerprint = _fingerprint(
            incidence_fingerprint, metric_fingerprint, boundary_fingerprint
        )
        if harmonic_subspace is not None:
            if not isinstance(harmonic_subspace, HarmonicSubspace):
                raise TypeError("harmonic_subspace must be a HarmonicSubspace or None.")
            if harmonic_subspace.complex_fingerprint != fingerprint:
                raise ValueError("Harmonic subspace belongs to a different complex.")
            if len(harmonic_subspace.bases) != len(counts):
                raise ValueError("Harmonic subspace must cover every cochain degree.")
            if any(
                int(basis.shape[0]) != count
                for basis, count in zip(harmonic_subspace.bases, counts, strict=True)
            ):
                raise ValueError("Harmonic basis cell counts do not match the complex.")

        self.incidences = incidence_tuple
        self.hodge_stars = stars
        self.primal_measures = primal
        self.dual_measures = dual
        self.boundary_masks = masks
        self.coordinates = points
        self.harmonic_subspace = harmonic_subspace
        self.cell_counts = counts
        self.cell_offsets = offsets
        self.incidence_fingerprint = incidence_fingerprint
        self.metric_fingerprint = metric_fingerprint
        self.boundary_fingerprint = boundary_fingerprint
        self.fingerprint = fingerprint
        self.graph = self._build_graph()
        if validate:
            self.validate()

    @staticmethod
    def _degree_values(
        name: str,
        values: Sequence[Any],
        counts: tuple[int, ...],
        /,
        *,
        positive: bool,
    ) -> tuple[Array, ...]:
        resolved = tuple(values)
        if len(resolved) != len(counts):
            raise ValueError(f"{name} must provide one array per cochain degree.")
        arrays: list[Array] = []
        for degree, (value, count) in enumerate(zip(resolved, counts, strict=True)):
            array = _host_array(f"{name}[{degree}]", value, dtype=float)
            if array.shape != (count,):
                raise ValueError(f"{name}[{degree}] must have shape ({count},).")
            if positive and np.any(array <= 0.0):
                raise ValueError(f"{name}[{degree}] must be strictly positive.")
            arrays.append(jnp.asarray(array))
        return tuple(arrays)

    @staticmethod
    def _boundary_values(
        values: Sequence[Any] | None,
        counts: tuple[int, ...],
        /,
    ) -> tuple[Array, ...]:
        resolved = (
            tuple(np.zeros((count,), dtype=bool) for count in counts)
            if values is None
            else tuple(values)
        )
        if len(resolved) != len(counts):
            raise ValueError("boundary_masks must provide one mask per degree.")
        masks = []
        for degree, (value, count) in enumerate(zip(resolved, counts, strict=True)):
            mask = np.asarray(value, dtype=bool)
            if mask.shape != (count,):
                raise ValueError(f"boundary_masks[{degree}] must have shape ({count},).")
            masks.append(jnp.asarray(mask))
        return tuple(masks)

    @staticmethod
    def _coordinate_values(
        values: Sequence[Any | None] | None,
        counts: tuple[int, ...],
        /,
    ) -> tuple[Array | None, ...]:
        resolved = (None,) * len(counts) if values is None else tuple(values)
        if len(resolved) != len(counts):
            raise ValueError("coordinates must provide one entry per degree.")
        dimensions = set()
        points: list[Array | None] = []
        for degree, (value, count) in enumerate(zip(resolved, counts, strict=True)):
            if value is None:
                points.append(None)
                continue
            array = _host_array(f"coordinates[{degree}]", value, dtype=float)
            if array.ndim != 2 or int(array.shape[0]) != count:
                raise ValueError(
                    f"coordinates[{degree}] must have leading cell count {count}."
                )
            dimensions.add(int(array.shape[1]))
            points.append(jnp.asarray(array))
        if len(dimensions) > 1 or (dimensions and any(value is None for value in points)):
            raise ValueError("Coordinates must be present with one common dimension at all degrees.")
        return tuple(points)

    @staticmethod
    def _incidence_fingerprint(
        counts: tuple[int, ...], incidences: tuple[CochainIncidence, ...], /
    ) -> str:
        digest = hashlib.sha256()
        digest.update(repr(counts).encode("utf-8"))
        for incidence in incidences:
            digest.update(repr(incidence.degree).encode("utf-8"))
            _array_digest(digest, incidence.lower_indices)
            _array_digest(digest, incidence.upper_indices)
            _array_digest(digest, incidence.signs)
        return digest.hexdigest()

    @property
    def max_degree(self) -> int:
        return len(self.cell_counts) - 1

    @property
    def num_cells(self) -> int:
        return sum(self.cell_counts)

    def cell_entities(self, degree: int, /) -> Array:
        resolved = int(degree)
        if resolved < 0 or resolved > self.max_degree:
            raise ValueError(f"Cochain degree must lie in [0, {self.max_degree}].")
        start = self.cell_offsets[resolved]
        return jnp.arange(start, start + self.cell_counts[resolved], dtype=jnp.int32)

    def active_mask(
        self,
        degree: int,
        boundary_policy: CochainBoundaryPolicy | CochainBoundaryKind = "absolute",
        /,
    ) -> Array:
        policy = (
            boundary_policy
            if isinstance(boundary_policy, CochainBoundaryPolicy)
            else CochainBoundaryPolicy(boundary_policy)
        )
        if policy.kind == "absolute":
            return jnp.ones((self.cell_counts[int(degree)],), dtype=bool)
        return ~self.boundary_masks[int(degree)]

    def with_harmonic_subspace(self, subspace: HarmonicSubspace, /) -> "CochainComplexIR":
        return CochainComplexIR(
            self.cell_counts,
            self.incidences,
            self.hodge_stars,
            primal_measures=self.primal_measures,
            dual_measures=self.dual_measures,
            boundary_masks=self.boundary_masks,
            coordinates=self.coordinates,
            harmonic_subspace=subspace,
        )

    def _build_graph(self) -> GraphIR:
        degrees = np.concatenate(
            [np.full((count,), degree, dtype=np.int32) for degree, count in enumerate(self.cell_counts)]
        )
        local_indices = np.concatenate(
            [np.arange(count, dtype=np.int32) for count in self.cell_counts]
        )
        nodes: dict[str, Array] = {
            "cell_dim": jnp.asarray(degrees),
            "local_index": jnp.asarray(local_indices),
            "hodge_star": jnp.concatenate(self.hodge_stars),
            "primal_measure": jnp.concatenate(self.primal_measures),
            "dual_measure": jnp.concatenate(self.dual_measures),
            "boundary": jnp.concatenate(self.boundary_masks),
        }
        if self.coordinates[0] is not None:
            nodes["coordinates"] = jnp.concatenate(
                tuple(value for value in self.coordinates if value is not None), axis=0
            )
        if self.harmonic_subspace is not None:
            max_modes = self.harmonic_subspace.max_modes
            packed = np.zeros(
                (self.num_cells, self.max_degree + 1, max_modes), dtype=float
            )
            for degree, basis in enumerate(self.harmonic_subspace.bases):
                start = self.cell_offsets[degree]
                packed[start : start + self.cell_counts[degree], degree, :] = np.asarray(
                    basis
                )
            nodes["harmonic_basis"] = jnp.asarray(packed)

        senders: list[np.ndarray] = []
        receivers: list[np.ndarray] = []
        signs: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        incidence_degrees: list[np.ndarray] = []
        for incidence in self.incidences:
            lower = np.asarray(incidence.lower_indices) + self.cell_offsets[incidence.degree - 1]
            upper = np.asarray(incidence.upper_indices) + self.cell_offsets[incidence.degree]
            coefficient = np.asarray(incidence.signs)
            count = lower.size
            senders.extend((lower, upper))
            receivers.extend((upper, lower))
            signs.extend((coefficient, coefficient))
            directions.extend(
                (
                    np.ones((count,), dtype=np.int8),
                    -np.ones((count,), dtype=np.int8),
                )
            )
            incidence_degrees.extend(
                (
                    np.full((count,), incidence.degree, dtype=np.int32),
                    np.full((count,), incidence.degree, dtype=np.int32),
                )
            )
        sender_array = np.concatenate(senders) if senders else np.zeros((0,), dtype=np.int32)
        receiver_array = (
            np.concatenate(receivers) if receivers else np.zeros((0,), dtype=np.int32)
        )
        sign_array = np.concatenate(signs) if signs else np.zeros((0,), dtype=float)
        direction_array = (
            np.concatenate(directions) if directions else np.zeros((0,), dtype=np.int8)
        )
        incidence_degree_array = (
            np.concatenate(incidence_degrees)
            if incidence_degrees
            else np.zeros((0,), dtype=np.int32)
        )
        edges = {
            "cochain_incidence": jnp.ones(sender_array.shape, dtype=bool),
            "incidence_degree": jnp.asarray(incidence_degree_array),
            "incidence_direction": jnp.asarray(direction_array),
            "incidence_sign": jnp.asarray(sign_array),
        }
        globals_: dict[str, Array] = {
            "max_degree": jnp.asarray([[self.max_degree]], dtype=jnp.int32),
            "harmonic_rank": jnp.asarray(
                [
                    (0,) * (self.max_degree + 1)
                    if self.harmonic_subspace is None
                    else self.harmonic_subspace.ranks
                ],
                dtype=jnp.int32,
            ),
            "harmonic_boundary_policy": jnp.asarray(
                [
                    [
                        -1
                        if self.harmonic_subspace is None
                        else (
                            0
                            if self.harmonic_subspace.boundary_policy == "absolute"
                            else 1
                        )
                    ]
                ],
                dtype=jnp.int32,
            ),
        }
        return GraphIR(
            nodes=nodes,
            edges=edges,
            senders=jnp.asarray(sender_array),
            receivers=jnp.asarray(receiver_array),
            globals=globals_,
            n_node=jnp.asarray([self.num_cells], dtype=jnp.int32),
            n_edge=jnp.asarray([sender_array.size], dtype=jnp.int32),
        )

    def validate(self) -> None:
        self.graph.validate()
        for lower, upper in zip(self.incidences, self.incidences[1:], strict=False):
            composition = lower.scipy_matrix() @ upper.scipy_matrix()
            if composition.nnz and np.max(np.abs(composition.data)) > 0.0:
                raise ValueError("Boundary incidences violate B_k B_(k+1) = 0.")
        for incidence in self.incidences:
            upper_boundary = np.asarray(self.boundary_masks[incidence.degree])[
                np.asarray(incidence.upper_indices)
            ]
            lower_boundary = np.asarray(self.boundary_masks[incidence.degree - 1])[
                np.asarray(incidence.lower_indices)
            ]
            if np.any(upper_boundary & ~lower_boundary):
                raise ValueError("Boundary masks must define a closed boundary subcomplex.")


def cochain_complex_from_incidences(
    cell_counts: Sequence[int],
    incidences: Sequence[CochainIncidence | Any],
    hodge_stars: Sequence[Any],
    /,
    **kwargs: Any,
) -> CochainComplexIR:
    """Build a metric cochain complex from sparse or dense boundary incidences."""
    resolved: list[CochainIncidence] = []
    counts = tuple(int(value) for value in cell_counts)
    for degree, value in enumerate(incidences, start=1):
        resolved.append(
            value
            if isinstance(value, CochainIncidence)
            else CochainIncidence.from_dense(degree, value)
        )
    return CochainComplexIR(counts, resolved, hodge_stars, **kwargs)


def cochain_complex_from_simplicial(
    complex_graph: Any,
    hodge_stars: Sequence[Any],
    /,
    *,
    primal_measures: Sequence[Any] | None = None,
    dual_measures: Sequence[Any] | None = None,
    boundary_masks: Sequence[Any] | None = None,
    coordinates: Sequence[Any | None] | None = None,
) -> CochainComplexIR:
    """Attach metric data to an existing two-dimensional simplicial complex."""
    from ._simplicial import SimplicialComplexGraph

    if not isinstance(complex_graph, SimplicialComplexGraph):
        raise TypeError("cochain_complex_from_simplicial requires SimplicialComplexGraph.")
    edge_vertices = np.asarray(complex_graph.edge_vertices, dtype=np.int32)
    edge_ids = np.repeat(np.arange(edge_vertices.shape[0], dtype=np.int32), 2)
    b1 = CochainIncidence(
        1,
        int(complex_graph.vertex_cells.size),
        int(complex_graph.edge_cells.size),
        edge_vertices.reshape((-1,)),
        edge_ids,
        np.tile(np.asarray((-1.0, 1.0)), edge_vertices.shape[0]),
    )
    face_edges = np.asarray(complex_graph.face_edges, dtype=np.int32)
    b2 = CochainIncidence(
        2,
        int(complex_graph.edge_cells.size),
        int(complex_graph.face_cells.size),
        face_edges.reshape((-1,)),
        np.repeat(np.arange(face_edges.shape[0], dtype=np.int32), face_edges.shape[1]),
        np.asarray(complex_graph.face_edge_signs).reshape((-1,)),
    )
    return CochainComplexIR(
        (
            int(complex_graph.vertex_cells.size),
            int(complex_graph.edge_cells.size),
            int(complex_graph.face_cells.size),
        ),
        (b1, b2),
        hodge_stars,
        primal_measures=primal_measures,
        dual_measures=dual_measures,
        boundary_masks=boundary_masks,
        coordinates=coordinates,
    )


def triangle_mesh_to_cochain_complex(
    mesh_vertices: Any,
    mesh_faces: Any,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
) -> CochainComplexIR:
    """Build a positive barycentric metric cochain complex from a triangle mesh."""
    from ._simplicial import triangle_mesh_to_simplicial_graph

    vertices = _host_array("mesh_vertices", mesh_vertices, dtype=float)
    faces = np.asarray(mesh_faces)
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError("mesh_vertices must have shape (vertices, embedding_dim >= 2).")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(
        faces.dtype, np.integer
    ):
        raise ValueError("mesh_faces must have integer shape (faces, 3).")
    faces = faces.astype(np.int32, copy=False)
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("mesh_faces contain out-of-range vertex indices.")
    bundle = triangle_mesh_to_simplicial_graph(
        faces,
        num_vertices=int(vertices.shape[0]),
    )
    edge_vertices = np.asarray(bundle.edge_vertices, dtype=np.int32)
    edge_points = 0.5 * (
        vertices[edge_vertices[:, 0]] + vertices[edge_vertices[:, 1]]
    )
    face_points = np.mean(vertices[faces], axis=1)
    first = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    second = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    gram = np.sum(first * first, axis=1) * np.sum(second * second, axis=1) - np.square(
        np.sum(first * second, axis=1)
    )
    areas = 0.5 * np.sqrt(np.maximum(gram, 0.0))
    if np.any(areas <= 0.0):
        raise ValueError("Triangle meshes must not contain degenerate faces.")
    edge_lengths = np.linalg.norm(
        vertices[edge_vertices[:, 1]] - vertices[edge_vertices[:, 0]], axis=1
    )
    if np.any(edge_lengths <= 0.0):
        raise ValueError("Triangle meshes must not contain zero-length edges.")

    vertex_dual = np.zeros((vertices.shape[0],), dtype=float)
    for local in range(3):
        np.add.at(vertex_dual, faces[:, local], areas / 3.0)
    edge_dual = np.zeros((edge_vertices.shape[0],), dtype=float)
    face_edges = np.asarray(bundle.face_edges, dtype=np.int32)
    for local in range(3):
        ids = face_edges[:, local]
        np.add.at(edge_dual, ids, np.linalg.norm(face_points - edge_points[ids], axis=1))

    edge_face_count = np.bincount(
        face_edges.reshape((-1,)), minlength=edge_vertices.shape[0]
    )
    boundary_edge = edge_face_count == 1
    boundary_vertex = np.zeros((vertices.shape[0],), dtype=bool)
    boundary_vertex[edge_vertices[boundary_edge].reshape((-1,))] = True
    boundary_face = np.zeros((faces.shape[0],), dtype=bool)
    CochainBoundaryPolicy(boundary_policy)
    return cochain_complex_from_simplicial(
        bundle,
        (
            vertex_dual,
            edge_dual / edge_lengths,
            1.0 / areas,
        ),
        primal_measures=(
            np.ones((vertices.shape[0],), dtype=float),
            edge_lengths,
            areas,
        ),
        dual_measures=(
            vertex_dual,
            edge_dual,
            np.ones((faces.shape[0],), dtype=float),
        ),
        boundary_masks=(boundary_vertex, boundary_edge, boundary_face),
        coordinates=(vertices, edge_points, face_points),
    )


def _restricted_boundary_matrix(
    complex_ir: CochainComplexIR,
    degree: int,
    policy: CochainBoundaryPolicy,
    /,
) -> sp.csr_matrix:
    boundary = complex_ir.incidences[degree - 1].scipy_matrix()
    if policy.kind == "absolute":
        return boundary
    lower_active = np.asarray(complex_ir.active_mask(degree - 1, policy))
    upper_active = np.asarray(complex_ir.active_mask(degree, policy))
    return boundary[lower_active][:, upper_active].tocsr()


def compute_harmonic_subspace(
    complex_ir: CochainComplexIR,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    max_modes: int = 8,
    tolerance: float = 1e-9,
    dense_threshold: int = 256,
) -> HarmonicSubspace:
    """Precompute exact metric harmonic bases without target-dependent information."""
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("compute_harmonic_subspace requires a CochainComplexIR.")
    policy = CochainBoundaryPolicy(boundary_policy)
    if int(max_modes) < 0:
        raise ValueError("max_modes must be non-negative.")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")
    bases: list[Array] = []
    eigenvalues: list[Array] = []
    ranks: list[int] = []
    for degree, count in enumerate(complex_ir.cell_counts):
        active = np.asarray(complex_ir.active_mask(degree, policy), dtype=bool)
        active_count = int(np.sum(active))
        metric = np.asarray(complex_ir.hodge_stars[degree])[active]
        inverse_sqrt = sp.diags(1.0 / np.sqrt(metric))
        sqrt_metric = sp.diags(np.sqrt(metric))
        laplacian = sp.csr_matrix((active_count, active_count), dtype=float)
        if degree > 0:
            boundary = _restricted_boundary_matrix(complex_ir, degree, policy)
            transformed = sqrt_metric @ boundary.T @ sp.diags(
                1.0 / np.sqrt(
                    np.asarray(complex_ir.hodge_stars[degree - 1])[
                        np.asarray(complex_ir.active_mask(degree - 1, policy), dtype=bool)
                    ]
                )
            )
            laplacian = laplacian + transformed @ transformed.T
        if degree < complex_ir.max_degree:
            boundary = _restricted_boundary_matrix(complex_ir, degree + 1, policy)
            transformed = sp.diags(
                np.sqrt(
                    np.asarray(complex_ir.hodge_stars[degree + 1])[
                        np.asarray(complex_ir.active_mask(degree + 1, policy), dtype=bool)
                    ]
                )
            ) @ boundary.T @ inverse_sqrt
            laplacian = laplacian + transformed.T @ transformed
        laplacian = 0.5 * (laplacian + laplacian.T)
        scale = max(1.0, float(np.max(np.abs(laplacian.data), initial=0.0)))
        threshold = float(tolerance) * scale
        if active_count == 0:
            values = np.zeros((0,), dtype=float)
            vectors = np.zeros((0, 0), dtype=float)
        elif active_count <= int(dense_threshold) or active_count <= int(max_modes) + 1:
            values, vectors = np.linalg.eigh(laplacian.toarray())
        else:
            requested = min(active_count - 1, int(max_modes) + 1)
            values, vectors = eigsh(laplacian, k=requested, which="SM", tol=tolerance)
            order = np.argsort(values, kind="stable")
            values = values[order]
            vectors = vectors[:, order]
        rank = int(np.sum(np.abs(values) <= threshold))
        if rank > int(max_modes) or (
            values.size and rank == values.size and values.size < active_count
        ):
            raise ValueError(
                "Harmonic nullspace exceeds max_modes or is not separated from nonzero modes."
            )
        physical_active = np.asarray(inverse_sqrt @ vectors[:, :rank])
        physical = np.zeros((count, int(max_modes)), dtype=float)
        if rank:
            physical[np.flatnonzero(active), :rank] = physical_active
            gram = physical_active.T @ (metric[:, None] * physical_active)
            if not np.allclose(gram, np.eye(rank), rtol=1e-7, atol=1e-9):
                raise ValueError("Computed harmonic basis is not metric orthonormal.")
        stored_values = np.full((int(max_modes),), np.inf, dtype=float)
        stored_values[: min(values.size, int(max_modes))] = values[: int(max_modes)]
        bases.append(jnp.asarray(physical))
        eigenvalues.append(jnp.asarray(stored_values))
        ranks.append(rank)
    return HarmonicSubspace(
        bases,
        eigenvalues,
        ranks,
        max_modes=int(max_modes),
        boundary_policy=policy.kind,
        complex_fingerprint=complex_ir.fingerprint,
    )


def reorient_cochain(
    values: Any,
    orientation_signs: Any,
    /,
    *,
    cell_axis: int | None = None,
) -> Array:
    """Apply a diagonal cell-orientation change to cochain coefficients."""
    array = jnp.asarray(values)
    signs = jnp.asarray(orientation_signs)
    if signs.ndim != 1:
        raise ValueError("Orientation signs must be a rank-1 array.")
    if cell_axis is None:
        candidates = tuple(
            axis for axis, size in enumerate(array.shape) if int(size) == int(signs.size)
        )
        if len(candidates) != 1:
            raise ValueError(
                "Could not infer one unique cochain cell axis; pass cell_axis explicitly."
            )
        axis = candidates[0]
    else:
        axis = int(cell_axis)
        if axis < 0:
            axis += array.ndim
        if axis < 0 or axis >= array.ndim:
            raise ValueError("cell_axis is out of range.")
    if int(array.shape[axis]) != int(signs.size):
        raise ValueError("Orientation signs must match the cochain cell axis.")
    if bool(jnp.any(jnp.abs(signs) != 1)):
        raise ValueError("Orientation signs must be ±1.")
    shape = [1] * array.ndim
    shape[axis] = int(signs.size)
    return array * signs.reshape(shape)


def reorient_cochain_complex(
    complex_ir: CochainComplexIR,
    orientation_signs: Sequence[Any],
    /,
) -> CochainComplexIR:
    """Return an equivalent complex under independent cell-orientation changes."""
    signs = tuple(np.asarray(value, dtype=float) for value in orientation_signs)
    if len(signs) != len(complex_ir.cell_counts):
        raise ValueError("orientation_signs must provide one vector per degree.")
    for degree, (value, count) in enumerate(zip(signs, complex_ir.cell_counts, strict=True)):
        if value.shape != (count,) or np.any(np.abs(value) != 1.0):
            raise ValueError(f"orientation_signs[{degree}] must contain {count} values ±1.")
    incidences = []
    for incidence in complex_ir.incidences:
        coefficient = (
            signs[incidence.degree - 1][np.asarray(incidence.lower_indices)]
            * np.asarray(incidence.signs)
            * signs[incidence.degree][np.asarray(incidence.upper_indices)]
        )
        incidences.append(
            CochainIncidence(
                incidence.degree,
                incidence.lower_count,
                incidence.upper_count,
                incidence.lower_indices,
                incidence.upper_indices,
                coefficient,
            )
        )
    return CochainComplexIR(
        complex_ir.cell_counts,
        incidences,
        complex_ir.hodge_stars,
        primal_measures=complex_ir.primal_measures,
        dual_measures=complex_ir.dual_measures,
        boundary_masks=complex_ir.boundary_masks,
        coordinates=complex_ir.coordinates,
    )


__all__ = [
    "CochainBoundaryKind",
    "CochainBoundaryPolicy",
    "CochainComplexIR",
    "CochainIncidence",
    "HarmonicSubspace",
    "cochain_complex_from_incidences",
    "cochain_complex_from_simplicial",
    "compute_harmonic_subspace",
    "reorient_cochain",
    "reorient_cochain_complex",
    "triangle_mesh_to_cochain_complex",
]
