#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Neural implicit regions whose trainable weights live in the design state."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array

from ..._differentiation import CapabilityEvidenceKind, DerivativeRegularity
from ..._fingerprint import canonical_fingerprint
from ..._model._array import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState, partition_parameters
from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import (
    CompiledGeometry,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._sampling import bounded_rejection_sample, RejectionSamplingPlan
from .._validity import GeometryValidityEvidence
from ..analytic._primitives import _check_points, _feature_id
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._policy import ImplicitSurfacePolicy
from ._projection import _field_and_gradient


_DEFAULT_POLICY = ImplicitSurfacePolicy()


def _activation_lipschitz(function: Any, /) -> float | None:
    """Global Lipschitz constant (sup |sigma'|) of an activation matched by identity."""
    # Imported here: importing the network package eagerly from geometry cycles.
    from ...nn._utils import _identity

    # `jax.nn.leaky_relu` is listed at its default slope of 0.01.
    known = (
        (_identity, 1.0),
        (jax.nn.identity, 1.0),
        (jnp.tanh, 1.0),
        (jnp.sin, 1.0),
        (jnp.cos, 1.0),
        (jax.nn.sigmoid, 0.25),
        (jax.nn.softplus, 1.0),
        (jax.nn.relu, 1.0),
        (jax.nn.relu6, 1.0),
        (jax.nn.hard_tanh, 1.0),
        (jax.nn.leaky_relu, 1.0),
    )
    for candidate, constant in known:
        if function is candidate:
            return constant
    return None


@final
@dataclass(frozen=True, slots=True)
class ImplicitRegionTopology:
    """Topological type of an implicit region resolved on a sampling lattice.

    `betti_numbers` are the region's Betti numbers `(b0, b1)` in two dimensions
    and `(b0, b1, b2)` in three. `boundary_components` lists each closed
    boundary component as `(role, euler_characteristic)` in canonical sorted
    order, where `role` is `"outer"` for a component bounding a region
    component from outside and `"inner"` for a hole or cavity boundary.
    """

    ambient_dimension: int
    betti_numbers: tuple[int, ...]
    boundary_components: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.ambient_dimension not in (2, 3):
            raise ValueError(
                "Implicit region topology is defined in two or three dimensions."
            )
        if len(self.betti_numbers) != self.ambient_dimension or any(
            value < 0 for value in self.betti_numbers
        ):
            raise ValueError(
                "betti_numbers must hold one non-negative entry per dimension below the ambient dimension."
            )
        if tuple(sorted(self.boundary_components)) != self.boundary_components or any(
            role not in ("outer", "inner") for role, _ in self.boundary_components
        ):
            raise ValueError(
                "boundary_components must be sorted (role, euler_characteristic) pairs "
                "with role 'outer' or 'inner'."
            )

    @property
    def topology_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "implicit-region-topology",
                "ambient_dimension": self.ambient_dimension,
                "betti_numbers": list(self.betti_numbers),
                "boundary_components": [list(item) for item in self.boundary_components],
            }
        )


@final
class NeuralImplicitCertificate(StrictModule, NonTrainableState):
    """Host-side sampled evidence of one neural implicit region state.

    The evidence is sampled, not a proof: `field` reports
    `SignReliability.LOCAL`, `ZeroSetAccuracy.APPROXIMATE`, and no
    `topology_identity`, and carries the Lipschitz upper bound and the declared
    evaluation-error bound. `lipschitz_evidence` records whether the Lipschitz
    bound was `CONSTRUCTED` from the network or `DECLARED` by the caller.
    `topology` is the region topology resolved on the sampling lattice; a
    zero-set feature between lattice nodes can escape it. `boundary_points` are
    the lattice zero crossings at which the gradient margin is checked,
    `clearance_points` the lattice nodes on the declared bounds, and `evidence`
    the validity evidence at the checked state.
    """

    field: FieldCertificate = eqx.field(static=True)
    lipschitz_evidence: CapabilityEvidenceKind = eqx.field(static=True)
    topology: ImplicitRegionTopology = eqx.field(static=True)
    capabilities: frozenset[GeometryCapability] = eqx.field(static=True)
    boundary_points: Array
    clearance_points: Array
    evidence: GeometryValidityEvidence

    def __init__(
        self,
        *,
        field: FieldCertificate,
        lipschitz_evidence: CapabilityEvidenceKind,
        topology: ImplicitRegionTopology,
        capabilities: frozenset[GeometryCapability],
        boundary_points: Array,
        clearance_points: Array,
        evidence: GeometryValidityEvidence,
    ):
        if field.topology_identity is not None:
            raise ValueError(
                "Sampled neural implicit evidence carries no field topology identity."
            )
        self.field = field
        self.lipschitz_evidence = lipschitz_evidence
        self.topology = topology
        self.capabilities = capabilities
        self.boundary_points = boundary_points
        self.clearance_points = clearance_points
        self.evidence = evidence


def _activation_lipschitz_factors(
    network: AbstractArrayModel, /
) -> tuple[float, ...] | None:
    """Per-layer activation constants and the final one, when a bound is constructible.

    The product of layer operator-norm bounds and activation Lipschitz constants
    bounds a plain feed-forward `MLP`. Residual projections, low-rank or
    transformed weights, and activations without a known constant are not
    covered; such networks require a declared bound.
    """
    # Imported here for the same import-cycle reason as `_activation_lipschitz`.
    from ...nn.models import MLP

    if not isinstance(network, MLP) or network.skip_connection:
        return None
    factors: list[float] = []
    for layer in network.layers:
        constant = _activation_lipschitz(layer.activation)
        if (
            constant is None
            or layer.weight_transform is not None
            or not isinstance(layer.weight, jax.Array)
        ):
            return None
        factors.append(constant)
    final_constant = _activation_lipschitz(network.final_activation)
    if final_constant is None:
        return None
    return (*factors, final_constant)


def _constructed_lipschitz_bound(network: Any, factors: tuple[float, ...], /) -> Array:
    # The Frobenius norm bounds the operator 2-norm of every layer matrix.
    bound = jnp.asarray(factors[-1], dtype=jnp.float64)
    for layer, constant in zip(network.layers, factors[:-1], strict=True):
        weight = layer.weight
        if layer.random_weight_factorization:
            weight = jnp.exp(layer.rwf_log_scales)[:, None] * weight
        bound = bound * constant * jnp.sqrt(jnp.sum(weight * weight))
    return bound


def _field_regularity(regularity: DerivativeRegularity | None, /) -> FieldRegularity:
    if regularity is None:
        return FieldRegularity.NONSMOOTH
    if regularity.continuity == "smooth":
        return FieldRegularity.SMOOTH
    if regularity.continuity >= 0 and regularity.pieces in ("polynomial", "smooth"):
        return FieldRegularity.PIECEWISE_SMOOTH
    return FieldRegularity.NONSMOOTH


def _network_parts(network: AbstractArrayModel, /):
    """Split `network` by array role into design parameters and fixed data.

    Returns the PARAMETER lane as `(names, leaves, treedef)`, the FIXED lane's
    arrays (dynamic data, never design parameters), and its array-free
    remainder (static structure).
    """
    parameters, model_state, fixed = partition_parameters(network)
    if jax.tree_util.tree_leaves(model_state):
        raise ValueError(
            "Neural implicit geometry requires a network without model state."
        )
    flat, treedef = jax.tree_util.tree_flatten_with_path(parameters)
    names = tuple(jax.tree_util.keystr(path).lstrip(".") for path, _ in flat)
    leaves = tuple(leaf for _, leaf in flat)
    fixed_arrays, static = eqx.partition(fixed, eqx.is_array)
    return names, leaves, treedef, fixed_arrays, static


def _assemble_network(
    treedef: Any, leaves: Any, fixed_arrays: Any, static: Any, /
) -> AbstractArrayModel:
    parameters = jax.tree_util.tree_unflatten(treedef, list(leaves))
    return eqx.combine(parameters, eqx.combine(fixed_arrays, static))


@eqx.filter_jit
def _network_values(network: AbstractArrayModel, points: Array, /) -> Array:
    return jax.vmap(network)(points)


def _evaluate_network(network: AbstractArrayModel, points: Array, dimension: int, /):
    points_ = _check_points(points, dimension)
    values = _network_values(network, points_.reshape((-1, dimension)))
    return values.reshape(points_.shape[:-1])


@final
class _NeuralImplicitKernel(GeometryKernel):
    """Region `{x in bounds : network(x; weights) <= 0}` with state-owned weights.

    Only the network's PARAMETER lane is read from the design state; its FIXED
    arrays are kernel data and its array-free remainder is static structure.
    """

    network_fixed: Any = fixed_field()
    network_static: Any = eqx.field(static=True)
    network_treedef: Any = eqx.field(static=True)
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)
    activation_lipschitz: tuple[float, ...] | None = eqx.field(static=True)
    region_bounds: Array
    reference_values: tuple[Array, ...]
    interior_points: Array
    exterior_points: Array
    clearance_points: Array
    boundary_points: Array
    sign_margin: float = eqx.field(static=True)
    gradient_margin: float | None = eqx.field(static=True)
    certificate: FieldCertificate = eqx.field(static=True)
    region_capabilities: frozenset[GeometryCapability] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        network_fixed: Any,
        network_static: Any,
        network_treedef: Any,
        bindings: tuple[ParameterBinding, ...],
        activation_lipschitz: tuple[float, ...] | None,
        region_bounds: Array,
        reference_values: tuple[Array, ...],
        interior_points: Array,
        exterior_points: Array,
        clearance_points: Array,
        boundary_points: Array,
        sign_margin: float,
        gradient_margin: float | None,
        certificate: FieldCertificate,
        capabilities: frozenset[GeometryCapability],
        source_id: str,
    ):
        self.network_fixed = network_fixed
        self.network_static = network_static
        self.network_treedef = network_treedef
        self.bindings = bindings
        self.activation_lipschitz = activation_lipschitz
        self.region_bounds = region_bounds
        self.reference_values = reference_values
        self.interior_points = interior_points
        self.exterior_points = exterior_points
        self.clearance_points = clearance_points
        self.boundary_points = boundary_points
        self.sign_margin = sign_margin
        self.gradient_margin = gradient_margin
        self.certificate = certificate
        self.region_capabilities = capabilities
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return self.region_bounds.shape[1]

    @property
    def intrinsic_dimension(self) -> int:
        return self.region_bounds.shape[1]

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return self.region_capabilities

    @property
    def field_certificate(self) -> FieldCertificate:
        return self.certificate

    def network(self, state: DesignState, /) -> AbstractArrayModel:
        return _assemble_network(
            self.network_treedef,
            [binding.read(state) for binding in self.bindings],
            self.network_fixed,
            self.network_static,
        )

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        network = self.network(state)
        dimension = self.ambient_dimension
        error = self.certificate.evaluation_error
        lipschitz = self.certificate.lipschitz_upper_bound
        interior = _evaluate_network(network, self.interior_points, dimension)
        exterior = _evaluate_network(network, self.exterior_points, dimension)
        clearance = _evaluate_network(network, self.clearance_points, dimension)
        samples = jnp.concatenate(
            (
                self.interior_points,
                self.exterior_points,
                self.clearance_points,
                self.boundary_points,
            )
        )
        _, gradients = _field_and_gradient(self, state, samples)
        gradient_norms = jnp.linalg.norm(gradients, axis=-1)
        margins = [
            -jnp.max(interior) - error - self.sign_margin,
            jnp.min(exterior) - error - self.sign_margin,
            jnp.min(clearance) - error - self.sign_margin,
            lipschitz - jnp.max(gradient_norms),
        ]
        names = [
            "interior_sign",
            "exterior_sign",
            "bounds_clearance",
            "lipschitz_sampled_gradient",
        ]
        if self.gradient_margin is not None:
            boundary_count = self.boundary_points.shape[0]
            margins.append(
                jnp.min(gradient_norms[-boundary_count:]) - self.gradient_margin
            )
            names.append("boundary_gradient")
        if self.activation_lipschitz is not None:
            margins.append(
                lipschitz
                - _constructed_lipschitz_bound(network, self.activation_lipschitz)
            )
            names.append("lipschitz_construction")
        margin_array = jnp.stack(margins)
        weights_finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(binding.read(state))) for binding in self.bindings]
            )
        )
        certified_state = jnp.all(
            jnp.stack(
                [
                    jnp.all(binding.read(state) == reference)
                    for binding, reference in zip(
                        self.bindings, self.reference_values, strict=True
                    )
                ]
            )
        )
        # Sampled margins are rechecked at every state; the sampled topology is
        # resolved only at the checked weights, so any other state stays
        # inconclusive until it is recertified.
        return GeometryValidityEvidence(
            finite=weights_finite & jnp.all(jnp.isfinite(margin_array)),
            conditions_satisfied=jnp.all(margin_array >= 0.0),
            resolved=certified_state,
            margins=margin_array,
            margin_names=tuple(names),
            contract_id="neural_implicit",
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return _evaluate_network(self.network(state), points, self.ambient_dimension)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.ambient_dimension)
        inside_bounds = jnp.all(
            (points_ >= self.region_bounds[0]) & (points_ <= self.region_bounds[1]),
            axis=-1,
        )
        return inside_bounds & (self.boundary_field(state, points_) <= 0.0)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, self.ambient_dimension)
        flat = points_.reshape((-1, self.ambient_dimension))
        _, gradient = _field_and_gradient(self, state, flat)
        norm = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
        normal = gradient / jnp.maximum(norm, jnp.finfo(flat.dtype).eps)
        return normal.reshape(points_.shape)

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return self.region_bounds

    def measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            "Neural implicit regions have no evidenced interior measure route."
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        raise NotImplementedError(
            "Neural implicit regions have no evidenced boundary measure route."
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.region_bounds
        dimension = self.ambient_dimension

        def proposal(proposal_key, count):
            return jr.uniform(
                proposal_key,
                shape=(count, dimension),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            )

        return bounded_rejection_sample(
            proposal,
            lambda points: self.contains(state, points),
            num_points=int(num_points),
            point_dimension=dimension,
            key=key,
            plan=RejectionSamplingPlan() if plan is None else plan,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        del state, num_points, key
        raise NotImplementedError(
            "Neural implicit regions do not provide boundary sampling."
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        raise NotImplementedError(
            "Neural implicit regions do not provide a boundary atlas."
        )


def _compile_kernel(
    context: _ParameterCollector,
    network: AbstractArrayModel,
    bounds: Array,
    interior_points: Array,
    exterior_points: Array,
    /,
    *,
    feature_id: str,
    sign_margin: float,
    gradient_margin: float | None,
    certificate: FieldCertificate,
    lipschitz_evidence: CapabilityEvidenceKind,
    capabilities: frozenset[GeometryCapability],
    clearance_points: Array,
    boundary_points: Array,
) -> _NeuralImplicitKernel:
    names, leaves, treedef, fixed, static = _network_parts(network)
    bindings = tuple(
        context.bind(ParameterId(feature_id, name), leaf, role="network_weight")
        for name, leaf in zip(names, leaves, strict=True)
    )
    return _NeuralImplicitKernel(
        network_fixed=fixed,
        network_static=static,
        network_treedef=treedef,
        bindings=bindings,
        activation_lipschitz=(
            _activation_lipschitz_factors(network)
            if lipschitz_evidence is CapabilityEvidenceKind.CONSTRUCTED
            else None
        ),
        region_bounds=bounds,
        reference_values=tuple(jnp.asarray(leaf, dtype=jnp.float64) for leaf in leaves),
        interior_points=interior_points,
        exterior_points=exterior_points,
        clearance_points=clearance_points,
        boundary_points=boundary_points,
        sign_margin=sign_margin,
        gradient_margin=gradient_margin,
        certificate=certificate,
        capabilities=capabilities,
        source_id=feature_id,
    )


def _require_accepted(evidence: GeometryValidityEvidence, /) -> None:
    margins = np.asarray(evidence.margins, dtype=np.float64)
    failed = tuple(
        name
        for name, margin in zip(evidence.margin_names, margins, strict=True)
        if not np.isfinite(margin) or margin < 0.0
    )
    if failed or not bool(np.asarray(evidence.finite)):
        raise ValueError(
            "Neural implicit sampled evidence failed: "
            + (", ".join(failed) if failed else "nonfinite weights")
            + "."
        )


# Square corners are cyclic; square edge k joins corners k and k + 1.
_SQUARE_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
_SQUARE_EDGES = ((0, 1), (1, 2), (3, 2), (0, 3))
_CUBE_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_CUBE_EDGES = (
    (0, 1),
    (1, 2),
    (3, 2),
    (0, 3),
    (4, 5),
    (5, 6),
    (7, 6),
    (4, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
# Each cube face as a cycle of cube corners.
_CUBE_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)
_BISECTION_STEPS = 64


def _square_arcs(inside: tuple[bool, ...], /) -> tuple[tuple[int, int], ...]:
    """Zero-set arcs of one lattice square as pairs of its cyclic edge indices.

    An alternating (ambiguous) square keeps its inside corners apart, the
    convention under which inside lattice corners connect only along edges.
    """
    crossing = tuple(k for k in range(4) if inside[k] != inside[(k + 1) % 4])
    if len(crossing) == 4:
        return tuple(((k - 1) % 4, k) for k in range(4) if inside[k])
    return (crossing,) if crossing else ()


@functools.cache
def _cell_pieces(dimension: int, pattern: int, /) -> tuple[tuple[int, ...], ...]:
    """Zero-set pieces of one lattice cell as tuples of local crossing-edge indices.

    Bit `c` of `pattern` marks corner `c` inside. A square's pieces are its arcs;
    a cube's pieces are the boundary loops of its surface patches, one disk per
    loop, joined from the arcs of its six faces.
    """
    if dimension == 2:
        return _square_arcs(tuple(bool(pattern >> c & 1) for c in range(4)))
    edge_index = {frozenset(edge): k for k, edge in enumerate(_CUBE_EDGES)}
    parent = list(range(len(_CUBE_EDGES)))

    def root(k: int) -> int:
        while parent[k] != k:
            k = parent[k]
        return k

    crossing: set[int] = set()
    for face in _CUBE_FACES:
        local = tuple(
            edge_index[frozenset((face[k], face[(k + 1) % 4]))] for k in range(4)
        )
        for first, second in _square_arcs(tuple(bool(pattern >> c & 1) for c in face)):
            parent[root(local[first])] = root(local[second])
            crossing.update((local[first], local[second]))
    loops: dict[int, list[int]] = {}
    for k in sorted(crossing):
        loops.setdefault(root(k), []).append(k)
    return tuple(tuple(loop) for loop in loops.values())


def _crossing_ids(inside: np.ndarray, /) -> tuple[np.ndarray, ...]:
    """Per axis, the index of every sign-changing lattice edge (`-1` elsewhere)."""
    ids: list[np.ndarray] = []
    count = 0
    for axis in range(inside.ndim):
        crossing = np.diff(inside.astype(np.int8), axis=axis) != 0
        axis_ids = np.full(crossing.shape, -1, dtype=np.int64)
        found = int(np.count_nonzero(crossing))
        axis_ids[crossing] = np.arange(count, count + found, dtype=np.int64)
        ids.append(axis_ids)
        count += found
    return tuple(ids)


def _sampled_topology(
    inside: np.ndarray, ids: tuple[np.ndarray, ...], /
) -> ImplicitRegionTopology:
    """Topology of the zero set resolved from lattice signs, as sampled evidence.

    The zero set is taken to cross each sign-changing lattice edge once and no
    other edge, with the piece structure of `_cell_pieces` in each cell. Its
    components are closed curves (2D) or surfaces (3D). A surface's Euler
    characteristic is `loops - crossings`: every crossing lies on four lattice
    faces, so arcs number twice the crossings. A component is `"outer"` when
    the region lies on its bounded side, decided by crossing parity along an
    axis-0 lattice line entered from the positive bounds.
    """
    dimension = inside.ndim
    count = sum(int(np.count_nonzero(axis_ids >= 0)) for axis_ids in ids)
    corners, edges = (
        (_SQUARE_CORNERS, _SQUARE_EDGES)
        if dimension == 2
        else (_CUBE_CORNERS, _CUBE_EDGES)
    )
    offsets = np.asarray(corners, dtype=np.int64)
    cells = (
        np.indices(tuple(size - 1 for size in inside.shape)).reshape((dimension, -1)).T
    )
    pattern = np.zeros(cells.shape[0], dtype=np.int64)
    for corner, offset in enumerate(offsets):
        pattern |= inside[tuple((cells + offset).T)].astype(np.int64) << corner
    local_ids = np.stack(
        [
            ids[int(np.flatnonzero(offsets[first] != offsets[second])[0])][
                tuple((cells + np.minimum(offsets[first], offsets[second])).T)
            ]
            for first, second in edges
        ],
        axis=1,
    )
    parent = list(range(count))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    pieces: list[int] = []
    full = (1 << len(corners)) - 1
    for cell in np.flatnonzero((pattern != 0) & (pattern != full)).tolist():
        for piece in _cell_pieces(dimension, int(pattern[cell])):
            members = local_ids[cell, list(piece)].tolist()
            for member in members[1:]:
                parent[root(member)] = root(members[0])
            pieces.append(members[0])
    labels: dict[int, int] = {}
    component = np.asarray(
        [labels.setdefault(root(k), len(labels)) for k in range(count)],
        dtype=np.int64,
    )
    crossings = np.bincount(component, minlength=len(labels))
    loops = np.bincount(
        component[np.asarray(pieces, dtype=np.int64)], minlength=len(labels)
    )
    first_axis = np.where(ids[0] >= 0, component[ids[0]], -1)
    boundary: list[tuple[str, int]] = []
    for label in range(len(labels)):
        position = tuple(np.argwhere(first_axis == label)[0])
        line = first_axis[(slice(None), *position[1:])]
        parity = int(np.count_nonzero(line[: position[0]] == label)) % 2
        # The lower node lies on the component's bounded side iff it is preceded
        # by an odd number of the component's crossings along the line.
        role = "outer" if bool(inside[position]) == (parity == 1) else "inner"
        euler = 0 if dimension == 2 else int(loops[label] - crossings[label])
        boundary.append((role, euler))
    outer = sum(role == "outer" for role, _ in boundary)
    inner = len(boundary) - outer
    if dimension == 2:
        betti = (outer, inner)
    else:
        betti = (outer, sum((2 - euler) // 2 for _, euler in boundary), inner)
    return ImplicitRegionTopology(dimension, betti, tuple(sorted(boundary)))


def _zero_crossings(
    network: AbstractArrayModel,
    nodes: np.ndarray,
    values: np.ndarray,
    ids: tuple[np.ndarray, ...],
    /,
) -> Array:
    """Zero crossings of the field on every sign-changing lattice edge, by bisection."""
    dimension = nodes.shape[-1]
    lower = []
    upper = []
    for axis, axis_ids in enumerate(ids):
        positions = np.argwhere(axis_ids >= 0)
        lower.append(positions)
        upper.append(positions + np.eye(dimension, dtype=np.int64)[axis])
    lower_ = np.concatenate(lower)
    upper_ = np.concatenate(upper)
    left = jnp.asarray(nodes[tuple(lower_.T)])
    right = jnp.asarray(nodes[tuple(upper_.T)])
    left_inside = jnp.asarray(values[tuple(lower_.T)] < 0.0)
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (left + right)
        keep_left = (_evaluate_network(network, middle, dimension) < 0.0) != left_inside
        left = jnp.where(keep_left[:, None], left, middle)
        right = jnp.where(keep_left[:, None], middle, right)
    return 0.5 * (left + right)


def _lattice_nodes(bounds: np.ndarray, resolution: tuple[int, ...], /) -> np.ndarray:
    axes = tuple(
        np.linspace(bounds[0, axis], bounds[1, axis], count)
        for axis, count in enumerate(resolution)
    )
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _bounds_nodes(nodes: np.ndarray, /) -> np.ndarray:
    shape = nodes.shape[:-1]
    indices = np.indices(shape)
    on_bounds = np.zeros(shape, dtype=bool)
    for axis, count in enumerate(shape):
        on_bounds |= (indices[axis] == 0) | (indices[axis] == count - 1)
    return nodes[on_bounds]


def _network_contract(network: AbstractArrayModel, dimension: int, /):
    if not isinstance(network, AbstractArrayModel):
        raise TypeError("network must be an AbstractArrayModel.")
    if network.in_size != dimension or network.out_size != "scalar":
        raise ValueError(
            f"network must map {dimension}-vectors to scalars (in_size={dimension}, "
            "out_size='scalar')."
        )
    contract = network.model_execution_contract()
    randomness = contract.randomness
    if (
        randomness is None
        or randomness.mode != "deterministic"
        or randomness.requires_inference_state
    ):
        raise ValueError(
            "Neural implicit geometry requires a declared deterministic network."
        )
    _, leaves, _, _, _ = _network_parts(network)
    if not leaves or any(
        not np.issubdtype(np.asarray(leaf).dtype, np.floating) for leaf in leaves
    ):
        raise ValueError(
            "network must hold one or more real floating-point weight arrays."
        )
    return contract.regularity


def _lipschitz_evidence(
    network: AbstractArrayModel,
    declared: float | None,
    /,
) -> tuple[float, CapabilityEvidenceKind]:
    if declared is not None:
        bound = float(declared)
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("lipschitz_upper_bound must be finite and positive.")
        return bound, CapabilityEvidenceKind.DECLARED
    factors = _activation_lipschitz_factors(network)
    if factors is None:
        raise ValueError(
            "A Lipschitz bound cannot be constructed for this network; declare "
            "lipschitz_upper_bound explicitly."
        )
    bound = float(np.asarray(_constructed_lipschitz_bound(network, factors)))
    if not np.isfinite(bound):
        raise ValueError("The constructed Lipschitz bound is not finite.")
    return bound, CapabilityEvidenceKind.CONSTRUCTED


def _lattice_topology(
    network: AbstractArrayModel,
    nodes: np.ndarray,
    policy: ImplicitSurfacePolicy,
    /,
) -> tuple[ImplicitRegionTopology, Array]:
    """Sampled lattice topology and the zero crossings on the lattice edges."""
    values = np.asarray(
        _evaluate_network(network, jnp.asarray(nodes), nodes.shape[-1]),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Neural implicit lattice field values must be finite.")
    if np.any(np.abs(values) <= policy.lattice_zero_tolerance):
        raise ValueError(
            "Neural implicit lattice node lies on the zero set; shift the bounds "
            "or change discovery_resolution."
        )
    inside = values < 0.0
    ids = _crossing_ids(inside)
    count = sum(int(np.count_nonzero(axis_ids >= 0)) for axis_ids in ids)
    if count == 0:
        raise ValueError("Neural implicit lattice contains no zero crossing.")
    if count > policy.maximum_crossings:
        raise ValueError("Neural implicit lattice exceeds maximum_crossings.")
    return _sampled_topology(inside, ids), _zero_crossings(network, nodes, values, ids)


def _certify(
    network: AbstractArrayModel,
    bounds: Array,
    interior_points: Array,
    exterior_points: Array,
    /,
    *,
    sign_margin: float,
    gradient_margin: float | None,
    evaluation_error: float,
    lipschitz_upper_bound: float | None,
    discovery_resolution: tuple[int, ...],
    topology: ImplicitRegionTopology | None,
    policy: ImplicitSurfacePolicy,
    feature_id: str,
) -> NeuralImplicitCertificate:
    """Check the sampled evidence host-side at the network's current weights."""
    dimension = bounds.shape[1]
    if dimension not in (2, 3):
        raise ValueError(
            "Neural implicit regions are defined in two or three dimensions."
        )
    regularity = _network_contract(network, dimension)
    lipschitz, lipschitz_evidence = _lipschitz_evidence(network, lipschitz_upper_bound)
    nodes = _lattice_nodes(np.asarray(bounds), discovery_resolution)
    if nodes[..., 0].size > policy.maximum_lattice_points:
        raise ValueError("Neural implicit lattice exceeds maximum_lattice_points.")
    clearance_points = jnp.asarray(_bounds_nodes(nodes))
    # No covering argument bounds the field between samples, so neither the
    # sign nor the topology is certified: both are sampled evidence.
    field = FieldCertificate(
        zero_set_accuracy=ZeroSetAccuracy.APPROXIMATE,
        sign_reliability=SignReliability.LOCAL,
        distance_semantics=DistanceSemantics.LEVEL_SET,
        regularity=_field_regularity(regularity),
        safe_step_factor=None,
        validity_region=(
            "declared axis-aligned bounds, sampled at the checked design state"
        ),
        parameter_differentiable=True,
        provenance=("neural_implicit",),
        lipschitz_upper_bound=lipschitz,
        evaluation_error=evaluation_error,
    )
    region_capabilities = frozenset(
        {GeometryCapability.REGION_QUERY, GeometryCapability.INTERIOR_SAMPLING}
    )

    def compiled(certificate, capabilities, boundary_points, margin):
        context = _ParameterCollector()
        kernel = _compile_kernel(
            context,
            network,
            bounds,
            interior_points,
            exterior_points,
            feature_id=feature_id,
            sign_margin=sign_margin,
            gradient_margin=margin,
            certificate=certificate,
            lipschitz_evidence=lipschitz_evidence,
            capabilities=capabilities,
            clearance_points=clearance_points,
            boundary_points=boundary_points,
        )
        _, state = context.finish()
        return CompiledGeometry(kernel, state)

    provisional = compiled(
        field,
        region_capabilities,
        jnp.zeros((0, dimension), dtype=jnp.float64),
        None,
    )
    _require_accepted(provisional.validity())
    sampled, boundary_points = _lattice_topology(network, nodes, policy)
    if topology is not None and sampled != topology:
        raise ValueError(
            "Neural implicit topology changed: expected Betti numbers "
            f"{topology.betti_numbers}, sampled {sampled.betti_numbers}."
        )
    normals = gradient_margin is not None and (
        regularity is not None
        and (regularity.continuity == "smooth" or regularity.continuity >= 1)
    )
    if normals:
        region_capabilities = region_capabilities | {GeometryCapability.BOUNDARY_NORMAL}
    certified = compiled(field, region_capabilities, boundary_points, gradient_margin)
    evidence = certified.validity()
    _require_accepted(evidence)
    return NeuralImplicitCertificate(
        field=field,
        lipschitz_evidence=lipschitz_evidence,
        topology=sampled,
        capabilities=region_capabilities,
        boundary_points=boundary_points,
        clearance_points=clearance_points,
        evidence=evidence,
    )


def _points(value: Any, bounds: np.ndarray, name: str, /) -> Array:
    if value is None:
        raise ValueError(f"Neural implicit evidence requires {name}.")
    host = np.asarray(value, dtype=np.float64)
    if (
        host.ndim != 2
        or host.shape[0] == 0
        or host.shape[1] != bounds.shape[1]
        or not np.all(np.isfinite(host))
    ):
        raise ValueError(
            f"{name} must be a non-empty finite array of shape (points, {bounds.shape[1]})."
        )
    if np.any(host < bounds[0]) or np.any(host > bounds[1]):
        raise ValueError(f"{name} must lie inside the declared bounds.")
    return jnp.asarray(host)


def _positive(value: Any, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@final
class NeuralImplicitRegion(GeometrySource):
    r"""Region bounded by the zero set of a neural field, with sampled evidence.

    The region is $\{x \in B : \phi(x; w) \le 0\}$ for the declared
    axis-aligned bounds $B$ and the scalar network $\phi(\cdot; w)$, negative
    inside. The PARAMETER lane of `network` (its resolved array roles) is
    registered as design parameters `ParameterId(feature_id, path)`, so the
    trainable weights live in the compiled `DesignState` and update through
    `with_parameters`/`with_state` or a design-state objective. FIXED arrays,
    such as `fixed_field` data, stay fixed kernel data and are never design
    parameters; networks with MODEL_STATE leaves are refused.

    Construction checks sampled evidence at the current weights and refuses
    when a check fails:

    - `lipschitz_upper_bound`: constructed from a plain `MLP` as the product of
      layer Frobenius norms and activation Lipschitz constants, otherwise an
      explicit declaration; either must dominate the sampled gradient norms;
    - `evaluation_error`: the declared absolute field evaluation-error bound;
    - sign margins: `interior_points` satisfy $\phi \le -m - \varepsilon$ and
      `exterior_points` and the lattice nodes on $\partial B$ satisfy
      $\phi \ge m + \varepsilon$ for `sign_margin` $m$;
    - sampled topology: the Betti numbers and boundary components of the zero
      set resolved from the field signs on a uniform `discovery_resolution`
      lattice over $B$. When `topology` is given, a different sampled topology
      is refused.

    The evidence is sampled, not a proof: no covering argument bounds the field
    between lattice nodes, so a zero-set component or sign change between
    samples can go undetected. The field certificate therefore reports
    `SignReliability.LOCAL`, `ZeroSetAccuracy.APPROXIMATE`, and no
    `topology_identity`, whether the Lipschitz bound is constructed or declared.

    `BOUNDARY_NORMAL` is advertised only when the network regularity is at
    least $C^1$ and `gradient_margin` holds at the lattice zero crossings.
    Measures, boundary sampling, and closest points are not advertised.
    """

    network: AbstractArrayModel
    bounds: Array
    interior_points: Array
    exterior_points: Array
    sign_margin: float = eqx.field(static=True)
    gradient_margin: float | None = eqx.field(static=True)
    evaluation_error: float = eqx.field(static=True)
    lipschitz_upper_bound: float | None = eqx.field(static=True)
    discovery_resolution: tuple[int, ...] = eqx.field(static=True)
    policy: ImplicitSurfacePolicy = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    certificate: NeuralImplicitCertificate

    def __init__(
        self,
        network: AbstractArrayModel,
        bounds: Any,
        /,
        *,
        interior_points: Any = None,
        exterior_points: Any = None,
        sign_margin: float | None = None,
        evaluation_error: float | None = None,
        discovery_resolution: int | tuple[int, ...] | None = None,
        lipschitz_upper_bound: float | None = None,
        gradient_margin: float | None = None,
        topology: ImplicitRegionTopology | None = None,
        policy: ImplicitSurfacePolicy = _DEFAULT_POLICY,
        feature_id: str | None = None,
    ):
        bounds_ = np.asarray(bounds, dtype=np.float64)
        if (
            bounds_.ndim != 2
            or bounds_.shape[0] != 2
            or not np.all(np.isfinite(bounds_))
            or np.any(bounds_[0] >= bounds_[1])
        ):
            raise ValueError("bounds must be finite (2, dimension) lower/upper corners.")
        dimension = bounds_.shape[1]
        interior = _points(interior_points, bounds_, "interior_points")
        exterior = _points(exterior_points, bounds_, "exterior_points")
        if sign_margin is None:
            raise ValueError("Neural implicit evidence requires sign_margin.")
        margin = _positive(sign_margin, "sign_margin")
        if evaluation_error is None:
            raise ValueError("Neural implicit evidence requires evaluation_error.")
        error = float(evaluation_error)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("evaluation_error must be finite and non-negative.")
        if discovery_resolution is None:
            raise ValueError(
                "Neural implicit sampled topology requires discovery_resolution."
            )
        resolution = (
            (discovery_resolution,) * dimension
            if isinstance(discovery_resolution, int)
            else tuple(discovery_resolution)
        )
        if len(resolution) != dimension or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 3
            for count in resolution
        ):
            raise ValueError(
                "discovery_resolution must give at least three lattice points per axis."
            )
        gradient = (
            None
            if gradient_margin is None
            else _positive(gradient_margin, "gradient_margin")
        )
        if topology is not None and not isinstance(topology, ImplicitRegionTopology):
            raise TypeError("topology must be an ImplicitRegionTopology or None.")
        if not isinstance(policy, ImplicitSurfacePolicy):
            raise TypeError("policy must be an ImplicitSurfacePolicy.")
        identifier = _feature_id(feature_id, "neural_implicit")
        region_bounds = jnp.asarray(bounds_)
        certificate = _certify(
            network,
            region_bounds,
            interior,
            exterior,
            sign_margin=margin,
            gradient_margin=gradient,
            evaluation_error=error,
            lipschitz_upper_bound=lipschitz_upper_bound,
            discovery_resolution=resolution,
            topology=topology,
            policy=policy,
            feature_id=identifier,
        )
        self.network = network
        self.bounds = region_bounds
        self.interior_points = interior
        self.exterior_points = exterior
        self.sign_margin = margin
        self.gradient_margin = gradient
        self.evaluation_error = error
        self.lipschitz_upper_bound = lipschitz_upper_bound
        self.discovery_resolution = resolution
        self.policy = policy
        self.feature_id = identifier
        self.certificate = certificate

    @property
    def parameter_ids(self) -> tuple[ParameterId, ...]:
        """Design-state identities of the network's PARAMETER arrays, in tree order."""
        names, _, _, _, _ = _network_parts(self.network)
        return tuple(ParameterId(self.feature_id, name) for name in names)

    def recertify(self, state: DesignState, /) -> NeuralImplicitRegion:
        """Recheck the sampled evidence at the weights held by `state`.

        Reruns the Lipschitz, sign-margin, gradient-margin, and lattice
        topology checks at the updated PARAMETER arrays, keeping the network's
        FIXED data, and refuses (`ValueError`) when a margin fails or the
        sampled topology differs from the recorded sampled topology. `state`
        may belong to any compiled geometry containing this region.
        """
        if not isinstance(state, DesignState):
            raise TypeError("state must be a DesignState.")
        _, _, treedef, fixed, static = _network_parts(self.network)
        leaves = [state.values[state.schema.index(item)] for item in self.parameter_ids]
        network = _assemble_network(treedef, leaves, fixed, static)
        return NeuralImplicitRegion(
            network,
            self.bounds,
            interior_points=self.interior_points,
            exterior_points=self.exterior_points,
            sign_margin=self.sign_margin,
            evaluation_error=self.evaluation_error,
            discovery_resolution=self.discovery_resolution,
            lipschitz_upper_bound=self.lipschitz_upper_bound,
            gradient_margin=self.gradient_margin,
            topology=self.certificate.topology,
            policy=self.policy,
            feature_id=self.feature_id,
        )

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        certificate = self.certificate
        return _compile_kernel(
            context,
            self.network,
            self.bounds,
            self.interior_points,
            self.exterior_points,
            feature_id=self.feature_id,
            sign_margin=self.sign_margin,
            gradient_margin=self.gradient_margin,
            certificate=certificate.field,
            lipschitz_evidence=certificate.lipschitz_evidence,
            capabilities=certificate.capabilities,
            clearance_points=certificate.clearance_points,
            boundary_points=certificate.boundary_points,
        )


__all__ = [
    "ImplicitRegionTopology",
    "NeuralImplicitCertificate",
    "NeuralImplicitRegion",
]
