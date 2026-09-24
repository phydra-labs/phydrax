#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified neural implicit regions whose weights live in the design state."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from ..._trainable import NonTrainableState
from ...discretization._axis import TensorGridPlan, UniformAxisSpec
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
from ..simplicial import TriangleTopology
from ._curve_discovery import discover_implicit_curve, ImplicitCurvePlan
from ._discovery import discover_implicit_surface
from ._policy import ImplicitSurfacePolicy
from ._projection import _field_and_gradient
from ._realization import ImplicitSurfacePlan


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
    """Topological type of a certified implicit region.

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
    """Host-side certification evidence of one neural implicit region state.

    `field` carries the Lipschitz upper bound, evaluation-error bound, and
    topology identity; `lipschitz_evidence` records whether the Lipschitz bound
    was `CONSTRUCTED` from the network or `DECLARED` by the caller.
    `boundary_points` are the zero-set points of the discovered boundary on
    which the gradient margin is certified, `clearance_points` the lattice
    nodes on the declared bounds, and `evidence` the validity evidence at the
    certified state.
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
        if field.topology_identity != topology.topology_id:
            raise ValueError(
                "Field certificate topology identity must match the topology."
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
    dynamic, static = eqx.partition(network, eqx.is_inexact_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)
    names = tuple(jax.tree_util.keystr(path).lstrip(".") for path, _ in flat)
    leaves = tuple(leaf for _, leaf in flat)
    return names, leaves, treedef, static


@eqx.filter_jit
def _network_values(network: AbstractArrayModel, points: Array, /) -> Array:
    return jax.vmap(network)(points)


def _evaluate_network(network: AbstractArrayModel, points: Array, dimension: int, /):
    points_ = _check_points(points, dimension)
    values = _network_values(network, points_.reshape((-1, dimension)))
    return values.reshape(points_.shape[:-1])


@final
class _NeuralImplicitKernel(GeometryKernel):
    """Region `{x in bounds : network(x; weights) <= 0}` with state-owned weights."""

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
        leaves = [binding.read(state) for binding in self.bindings]
        dynamic = jax.tree_util.tree_unflatten(self.network_treedef, leaves)
        return eqx.combine(dynamic, self.network_static)

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
        # Sampled margins are rechecked at every state; the topology identity is
        # established only at the certified weights, so any other state stays
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
    names, leaves, treedef, static = _network_parts(network)
    bindings = tuple(
        context.bind(ParameterId(feature_id, name), leaf, role="network_weight")
        for name, leaf in zip(names, leaves, strict=True)
    )
    return _NeuralImplicitKernel(
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
            "Neural implicit certification failed: "
            + (", ".join(failed) if failed else "nonfinite weights")
            + "."
        )


def _curve_topology(plan: ImplicitCurvePlan, /) -> ImplicitRegionTopology:
    vertices = np.asarray(plan.base_vertices, dtype=np.float64)
    edges = np.asarray(plan.edges, dtype=np.int32)
    successor = {int(start): int(stop) for start, stop in edges}
    if len(successor) != edges.shape[0] or set(successor.values()) != set(successor):
        raise ValueError("Discovered implicit contour is not consistently oriented.")
    remaining = set(successor)
    components: list[tuple[str, int]] = []
    while remaining:
        current = min(remaining)
        loop: list[int] = []
        while current in remaining:
            remaining.remove(current)
            loop.append(current)
            current = successor[current]
        points = vertices[np.asarray(loop, dtype=np.int32)]
        following = np.roll(points, -1, axis=0)
        area = 0.5 * np.sum(
            points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
        )
        if not np.isfinite(area) or area == 0.0:
            raise ValueError("Discovered implicit contour encloses no area.")
        # Discovery orients the outward field gradient to the right of each
        # segment, so outer boundaries run counterclockwise and holes clockwise.
        components.append(("outer" if area > 0.0 else "inner", 0))
    outer = sum(role == "outer" for role, _ in components)
    return ImplicitRegionTopology(
        2,
        (outer, len(components) - outer),
        tuple(sorted(components)),
    )


def _surface_topology(plan: ImplicitSurfacePlan, /) -> ImplicitRegionTopology:
    vertices = np.asarray(plan.base_vertices, dtype=np.float64)
    faces = np.asarray(plan.faces, dtype=np.int32)
    topology = TriangleTopology(faces, num_vertices=vertices.shape[0])
    component_ids = np.asarray(topology.face_component_ids)
    components: list[tuple[str, int]] = []
    genus_total = 0
    for component in range(topology.num_face_components):
        selected = faces[component_ids == component]
        edges = np.sort(
            np.concatenate(
                (selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]])
            ),
            axis=1,
        )
        euler = (
            np.unique(selected).size
            - np.unique(edges, axis=0).shape[0]
            + selected.shape[0]
        )
        if euler > 2 or euler % 2:
            raise ValueError("Discovered implicit surface component is not orientable.")
        triangles = vertices[selected]
        volume = np.sum(
            triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]), axis=-1
        ).sum()
        if not np.isfinite(volume) or volume == 0.0:
            raise ValueError("Discovered implicit surface component encloses no volume.")
        # Faces are oriented along the outward field gradient: outer shells
        # enclose positive volume and cavity shells negative volume.
        components.append(("outer" if volume > 0.0 else "inner", int(euler)))
        genus_total += (2 - int(euler)) // 2
    outer = sum(role == "outer" for role, _ in components)
    return ImplicitRegionTopology(
        3,
        (outer, genus_total, len(components) - outer),
        tuple(sorted(components)),
    )


def _lattice(bounds: np.ndarray, resolution: tuple[int, ...], /):
    axes = tuple(
        np.linspace(bounds[0, axis], bounds[1, axis], count)
        for axis, count in enumerate(resolution)
    )
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    indices = np.stack(
        np.meshgrid(*(np.arange(count) for count in resolution), indexing="ij"),
        axis=-1,
    )
    on_bounds = np.any(
        (indices == 0) | (indices == np.asarray(resolution) - 1),
        axis=-1,
    )
    grid = TensorGridPlan(tuple(UniformAxisSpec(count) for count in resolution)).prepare(
        jnp.asarray(bounds)
    )
    return grid, mesh[on_bounds]


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
    _, leaves, _, _ = _network_parts(network)
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


def _discover_topology(
    geometry: CompiledGeometry,
    grid: Any,
    policy: ImplicitSurfacePolicy,
    source_id: str,
    /,
) -> tuple[ImplicitRegionTopology, np.ndarray]:
    match geometry.ambient_dimension:
        case 2:
            plan = discover_implicit_curve(
                geometry, grid, policy=policy, source_id=source_id
            )
            return _curve_topology(plan), np.asarray(plan.base_vertices)
        case 3:
            plan = discover_implicit_surface(
                geometry, grid, policy=policy, source_id=source_id
            )
            return _surface_topology(plan), np.asarray(plan.projection.anchors)
        case _:
            raise ValueError(
                "Neural implicit geometry is certified in two or three dimensions."
            )


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
    """Establish every certificate host-side at the network's current weights."""
    dimension = bounds.shape[1]
    regularity = _network_contract(network, dimension)
    lipschitz, lipschitz_evidence = _lipschitz_evidence(network, lipschitz_upper_bound)
    grid, clearance = _lattice(np.asarray(bounds), discovery_resolution)
    clearance_points = jnp.asarray(clearance)
    field = FieldCertificate(
        zero_set_accuracy=ZeroSetAccuracy.TOLERANCE_BOUND,
        sign_reliability=SignReliability.RELIABLE,
        distance_semantics=DistanceSemantics.LEVEL_SET,
        regularity=_field_regularity(regularity),
        safe_step_factor=None,
        validity_region="declared axis-aligned bounds at the certified design state",
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
    discovered, boundary = _discover_topology(
        provisional, grid, policy, f"{feature_id}:certification"
    )
    if topology is not None and discovered != topology:
        raise ValueError(
            "Neural implicit topology changed: expected Betti numbers "
            f"{topology.betti_numbers}, discovered {discovered.betti_numbers}."
        )
    normals = gradient_margin is not None and (
        regularity is not None
        and (regularity.continuity == "smooth" or regularity.continuity >= 1)
    )
    if normals:
        region_capabilities = region_capabilities | {GeometryCapability.BOUNDARY_NORMAL}
    field = replace(field, topology_identity=discovered.topology_id)
    boundary_points = jnp.asarray(boundary)
    certified = compiled(field, region_capabilities, boundary_points, gradient_margin)
    evidence = certified.validity()
    _require_accepted(evidence)
    return NeuralImplicitCertificate(
        field=field,
        lipschitz_evidence=lipschitz_evidence,
        topology=discovered,
        capabilities=region_capabilities,
        boundary_points=boundary_points,
        clearance_points=clearance_points,
        evidence=evidence,
    )


def _points(value: Any, bounds: np.ndarray, name: str, /) -> Array:
    if value is None:
        raise ValueError(f"Neural implicit certification requires {name}.")
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
    r"""Region bounded by the zero set of a certified neural field.

    The region is $\{x \in B : \phi(x; w) \le 0\}$ for the declared
    axis-aligned bounds $B$ and the scalar network $\phi(\cdot; w)$, negative
    inside. Every inexact weight array of `network` is registered as a design
    parameter `ParameterId(feature_id, path)`, so the weights live in the
    compiled `DesignState` and update through `with_parameters`/`with_state`
    or a design-state objective; the compiled kernel keeps only static network
    structure.

    Construction certifies the current weights and refuses without every
    required certificate:

    - `lipschitz_upper_bound`: constructed from a plain `MLP` as the product of
      layer Frobenius norms and activation Lipschitz constants, otherwise an
      explicit declaration that must dominate the sampled gradient norms;
    - `evaluation_error`: the declared absolute field evaluation-error bound;
    - sign margins: `interior_points` satisfy $\phi \le -m - \varepsilon$ and
      `exterior_points` and the lattice nodes on $\partial B$ satisfy
      $\phi \ge m + \varepsilon$ for `sign_margin` $m$;
    - topology identity: the Betti numbers of the region discovered on a
      uniform `discovery_resolution` lattice over $B$ by the implicit
      curve/surface discovery. When `topology` is given, a different
      discovered topology is refused.

    `BOUNDARY_NORMAL` is advertised only when the network regularity is at
    least $C^1$ and `gradient_margin` is certified on the discovered zero set.
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
            raise ValueError("Neural implicit certification requires sign_margin.")
        margin = _positive(sign_margin, "sign_margin")
        if evaluation_error is None:
            raise ValueError("Neural implicit certification requires evaluation_error.")
        error = float(evaluation_error)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("evaluation_error must be finite and non-negative.")
        if discovery_resolution is None:
            raise ValueError(
                "Neural implicit topology identity requires discovery_resolution."
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
        """Design-state identities of the network weight arrays, in tree order."""
        names, _, _, _ = _network_parts(self.network)
        return tuple(ParameterId(self.feature_id, name) for name in names)

    def recertify(self, state: DesignState, /) -> NeuralImplicitRegion:
        """Certify the weights held by `state` under the same certificates.

        Reruns the Lipschitz, sign-margin, gradient-margin, and discovery
        checks at the updated weights and refuses (`ValueError`) when a margin
        fails or the discovered topology differs from the certified topology.
        `state` may belong to any compiled geometry containing this region.
        """
        if not isinstance(state, DesignState):
            raise TypeError("state must be a DesignState.")
        _, _, treedef, static = _network_parts(self.network)
        leaves = [state.values[state.schema.index(item)] for item in self.parameter_ids]
        network = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)
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
