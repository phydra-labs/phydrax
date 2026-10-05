#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared conservative edge transport on the meshfree radius graph.

``ConservativeTransport`` owns the instantaneous spatial transport invariant:
geometric nodal velocity -> integrated edge/boundary volume flux, boundary
inflow/outflow states, the low-order donor-cell flux, a GMLS-gradient MUSCL
reconstruction, paired Zalesak/Kuzmin antidiffusive limiting with local graph
bounds, and the forward-Euler CFL admission certificate of the selected
scheme. It publishes rates only; native temporal methods own integration.

Boundary volume flux is closed by the graph itself. Prescribed (boundary) nodes
carry second-moment rows declared through the exterior's
``boundary_area_vectors``, which keeps their tangential edges; the boundary
normal measure is the first-moment deficit ``S_i = sum_j w_ij (x_i - x_j)``
and the boundary volume flux ``u_i · S_i`` makes a uniform translation
divergence-free at every node, so a uniform field is preserved exactly.
Exteriors whose prescribed nodes carry no moment rows are refused.
"""

from __future__ import annotations

from enum import IntEnum
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import gather_routes
from ...typing import Bool, Float64, Int32, parse, Scalar
from .._point_cloud import PreparedPointCloudDiscretization
from ..spatial._neighbor_query import _minimum_image
from ._exterior import (
    _ExteriorEdgeDim,
    _ExteriorNodeDim,
    PreparedMeshfreeExteriorCalculus,
)
from ._exterior_transport import (
    _edge_volume_flux,
    _limited_coefficients,
    _neighbor_extrema,
    _upwind_edge_flux,
)
from ._neighbors import PreparedMeshfreeEdgeRelation


TransportScheme: TypeAlias = Literal["upwind", "reconstructed", "limited"]


class TransportStatus(IntEnum):
    ACCEPTED = 0
    METRIC_REFUSED = 1
    NONFINITE = 2


class TransportRefreshStatus(IntEnum):
    ACCEPTED = 0
    INVALID_COORDINATES = 1
    TOPOLOGY_TRUST_EXCEEDED = 2
    RECONSTRUCTION_REFUSED = 3
    METRIC_REFUSED = 4


@final
class TransportVolumeFlux(StrictModule):
    """Integrated volume flux: oriented edges and outward boundary nodes."""

    __strict_contract__ = True
    edge: Float64[_ExteriorEdgeDim]
    boundary: Float64[_ExteriorNodeDim]


@final
class TransportRate(StrictModule):
    """Instantaneous conservative content rate with flux and limiter evidence.

    ``edge_flux = low_order_flux + limiter * antidiffusive_flux`` is the content
    flux along canonical orientation; ``content_rate = B^T edge_flux +
    boundary_content_rate + source``. ``limiter`` is ``0`` for ``upwind``, ``1``
    for ``reconstructed`` and the frozen Zalesak/Kuzmin coefficient for
    ``limited``; ``limiter_switched`` counts edges with a nonzero antidiffusive
    flux whose coefficient is below one, the points where the rate is only
    piecewise differentiable. ``conservation_residual`` audits the content
    ledger against boundary exchange and sources.
    """

    __strict_contract__ = True
    content_rate: Float64[_ExteriorNodeDim]
    value_rate: Float64[_ExteriorNodeDim]
    edge_flux: Float64[_ExteriorEdgeDim]
    low_order_flux: Float64[_ExteriorEdgeDim]
    antidiffusive_flux: Float64[_ExteriorEdgeDim]
    limiter: Float64[_ExteriorEdgeDim]
    volume_flux: Float64[_ExteriorEdgeDim]
    boundary_volume_flux: Float64[_ExteriorNodeDim]
    boundary_content_rate: Float64[_ExteriorNodeDim]
    node_outflow: Float64[_ExteriorNodeDim]
    lower_bound: Float64[_ExteriorNodeDim]
    upper_bound: Float64[_ExteriorNodeDim]
    limiter_switched: Int32[Scalar]
    conservation_residual: Float64[Scalar]
    finite: Bool[Scalar]
    metric_admitted: Bool[Scalar]
    status: Int32[Scalar]


@final
class TransportCFL(StrictModule):
    """Forward-Euler admission certificate of the selected spatial scheme.

    ``node_cfl = dt (outflow + capacity) / V`` with the limiter capacity of
    ``limited`` and zero capacity for ``upwind``. ``certified`` is false for
    ``reconstructed``: an unlimited high-order flux has no positivity or local
    bound certificate. ``admitted`` requires ``certified`` and ``cfl <= 1``.
    The certificate covers the forward-Euler base step; an SSP method inherits
    it only with its own SSP coefficient, and no other method inherits it.
    """

    __strict_contract__ = True
    node_cfl: Float64[_ExteriorNodeDim]
    cfl: Float64[Scalar]
    step_bound: Float64[Scalar]
    certified: Bool[Scalar]
    admitted: Bool[Scalar]


def minimum_image_edge_charts(
    edge_relation: PreparedMeshfreeEdgeRelation, points: ArrayLike, /
) -> Array:
    """Endpoint charts ``(capacity, 2, d)`` of a periodic radius relation.

    Endpoint 0 holds the minimum-image ``x_target - x_source`` and endpoint 1 its
    negative, in edge-relation capacity order, as required by
    ``MeshfreeExteriorCalculusPlan(intrinsic_displacements=...)``.
    """
    if not isinstance(edge_relation, PreparedMeshfreeEdgeRelation):
        raise TypeError("edge_relation must be a PreparedMeshfreeEdgeRelation.")
    coordinates = jnp.asarray(points, dtype=jnp.float64)
    relation = edge_relation.relation
    if coordinates.ndim != 2 or coordinates.shape != (
        relation.source_size,
        edge_relation.address.dimension,
    ):
        raise ValueError("Edge charts require capacity points of the address dimension.")
    delta = coordinates[relation.target_indices] - coordinates[relation.source_indices]
    if any(edge_relation.address.periodic_axes):
        delta = _minimum_image(delta, edge_relation.address)
    delta = jnp.where(relation.valid[:, None], delta, 0.0)
    return jnp.stack((delta, -delta), axis=1)


@final
class ConservativeTransport(StrictModule):
    """Prepared high/low-order conservative transport on one exterior graph.

    ``scheme`` selects the published edge flux: ``"upwind"`` (donor cell),
    ``"reconstructed"`` (unlimited high-order flux) or ``"limited"`` (its
    antidiffusive part paired-limited against local graph bounds with nodal
    capacity ``limiter_capacity * outflow``). ``reconstruction`` must be a
    prepared point cloud on exactly the exterior's compact nodes; it is
    refused for ``"upwind"``.

    Edge fluxes of a nodal field ``q`` (velocity for volume, ``c u`` for
    content) use the Hermite-corrected midpoint rule
    ``w_e [ (q_i + q_j)/2 · e - 1/4 e · (grad q_j - grad q_i) e ]`` with GMLS
    gradients. Expanded about either endpoint its cubic part vanishes, so with
    only degree-two metric moments the nodal flux sum is ``V div q + O(V h^2)``
    whatever the metric's third moments or sign; the plain midpoint average
    leaves ``1/4 grad^2 q : sum w e e e``, an ``O(h)`` truncation that stalls
    nonuniform flows on irregular clouds. The correction is antisymmetric in
    ``(i, j)``, so conservation is exact, and it vanishes for affine ``q`` (the
    translation/affine GCL are unchanged). A supplied ``TransportVolumeFlux``
    carries no nodal velocity; its high-order content flux is the
    upwind-biased MUSCL face value ``c_up + 1/2 grad c_up · (x_down - x_up)``,
    whose truncation is ``O(h^2)`` only where the metric's third moments vanish.
    The upwind scheme has no gradients and keeps the plain midpoint volume flux.
    """

    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    reconstruction: PreparedPointCloudDiscretization | None
    boundary_mask: Bool[_ExteriorNodeDim]
    scheme: TransportScheme = eqx.field(static=True)
    limiter_capacity: float = eqx.field(static=True)
    has_boundary: bool = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        exterior: PreparedMeshfreeExteriorCalculus,
        /,
        *,
        scheme: TransportScheme = "limited",
        reconstruction: PreparedPointCloudDiscretization | None = None,
        limiter_capacity: float = 1.0,
    ) -> None:
        if not isinstance(exterior, PreparedMeshfreeExteriorCalculus):
            raise TypeError("Transport requires a prepared meshfree exterior owner.")
        scheme_ = parse(scheme, TransportScheme, "scheme")
        capacity = float(limiter_capacity)
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("limiter_capacity must be finite and positive.")
        points = np.asarray(jax.device_get(exterior.points))
        if exterior.endpoint_displacements.shape[2] != points.shape[1]:
            raise ValueError(
                "Bulk transport needs endpoint charts in the ambient coordinates."
            )
        match scheme_:
            case "upwind":
                if reconstruction is not None:
                    raise ValueError("Upwind transport takes no reconstruction.")
            case "reconstructed" | "limited":
                _require_reconstruction(reconstruction, points)
            case unknown:
                assert_never(unknown)
        boundary = ~np.asarray(jax.device_get(exterior.equation_mask))
        closed = np.zeros(boundary.shape, dtype=np.bool_)
        closed[np.asarray(jax.device_get(exterior.moment_node_indices))] = True
        if np.any(boundary & ~closed):
            # A prescribed node without moment rows has no tangential edges:
            # its transport rate would miss the tangential flux (O(1) error).
            raise ValueError(
                "Graph transport needs moment rows at every prescribed node; declare "
                "boundary_area_vectors on MeshfreeExteriorCalculusPlan."
            )
        self.exterior = exterior
        self.reconstruction = reconstruction
        self.boundary_mask = jnp.asarray(boundary)
        self.scheme = scheme_
        self.limiter_capacity = capacity
        self.has_boundary = bool(boundary.any())
        self.transport_id = canonical_fingerprint(
            {
                "kind": "meshfree-conservative-transport",
                "graph": exterior.incidence.source.space_id,
                "reconstruction": None
                if reconstruction is None
                else reconstruction.prepared_id,
                "scheme": scheme_,
                "limiter_capacity": capacity,
            }
        )

    @property
    def node_volumes(self) -> Array:
        return self.exterior.node_volumes

    @property
    def edge_vectors(self) -> Array:
        """Oriented ``x_j - x_i`` in the exterior endpoint chart (minimum image)."""
        return self.exterior.endpoint_displacements[:, 0, :]

    @property
    def boundary_normal_measure(self) -> Array:
        """Graph-closing outward normal measure ``S_i`` at boundary nodes, else 0.

        The exterior's deficit ``S_i = sum_j w_ij (x_i - x_j)``; interior nodes
        are closed exactly by their moments and contribute no boundary flux.
        """
        return jnp.where(
            self.boundary_mask[:, None], self.exterior.boundary_normal_measure, 0.0
        )

    def volume_flux(self, velocity: ArrayLike, /) -> TransportVolumeFlux:
        value = self._velocity(velocity)
        exterior = self.exterior
        edge = (
            _edge_volume_flux(
                exterior.incidence,
                exterior.metric_result.weights,
                self.edge_vectors,
                value,
            )
            if self.reconstruction is None
            else self._hermite_flux(value, self.reconstruction)
        )
        return TransportVolumeFlux(
            edge=edge,
            boundary=jnp.sum(self.boundary_normal_measure * value, axis=1),
        )

    def _hermite_flux(
        self, field: Array, reconstruction: PreparedPointCloudDiscretization, /
    ) -> Array:
        """Hermite-corrected integrated edge flux of a nodal vector field."""
        exterior = self.exterior
        relation = exterior.incidence.relation
        edges = exterior.lengths.shape[0]
        dimension = field.shape[1]
        gradient = reconstruction.gradient(field)
        ends = gather_routes(relation, field).reshape((edges, 2, dimension))
        slopes = gather_routes(relation, gradient).reshape(
            (edges, 2, dimension, dimension)
        )
        vectors = self.edge_vectors
        mean = 0.5 * jnp.sum((ends[:, 0] + ends[:, 1]) * vectors, axis=1)
        jump = slopes[:, 1] - slopes[:, 0]
        correction = 0.25 * jnp.sum(
            vectors[:, :, None] * jump * vectors[:, None, :], axis=(1, 2)
        )
        return exterior.metric_result.weights * (mean - correction)

    def volume_rate(self, advection: ArrayLike | TransportVolumeFlux, /) -> Array:
        """Net outgoing volume flux per node, the discrete ``V div u``."""
        flux = self._advecting_flux(advection)
        return -self.exterior.incidence.transpose_mv(flux.edge) + flux.boundary

    def rate(
        self,
        values: ArrayLike,
        advection: ArrayLike | TransportVolumeFlux,
        /,
        *,
        inflow: ArrayLike | None = None,
        inflow_rate: ArrayLike | None = None,
        source: ArrayLike | None = None,
    ) -> TransportRate:
        """Content rate of nodal concentrations under an advecting volume flux.

        ``advection`` is a nodal velocity ``(nodes, dimension)``, converted by
        ``volume_flux``, or an already integrated ``TransportVolumeFlux`` (for
        example an exactly projected solenoidal flux), used as given.
        ``inflow`` is the declared nodal boundary state entering at boundary
        nodes with inward volume flux; it is required exactly when the graph
        has boundary nodes. ``source`` is an extensive content rate.

        Without ``inflow_rate`` the inflow is weak (upwind): inflow boundary
        nodes relax toward ``inflow``. Boundary nodes carry no tangential graph
        edges (no moment equation), so their own rate misses the tangential
        flux; a weak inflow node therefore lags the inflow state by ``O(h)``,
        which the interior inherits and transport is only first order
        globally. ``inflow_rate`` (the time derivative of ``inflow``) makes the
        inflow nodes strong prescribed rows: their value rate is
        ``inflow_rate`` and the boundary supplies exactly the content the
        graph fluxes remove, recorded in ``boundary_content_rate`` so the
        ledger stays exact. Interior second-order reconstruction then reaches
        the boundary.
        """
        exterior = self.exterior
        value = jnp.asarray(values, dtype=jnp.float64)
        volumes = exterior.node_volumes
        if value.shape != volumes.shape:
            raise ValueError("Transport values must match compact graph nodes.")
        incoming = self._inflow(inflow, value)
        held_rate = None
        if inflow_rate is not None:
            if inflow is None:
                raise ValueError("Prescribed inflow rows need the declared inflow state.")
            held_rate = jnp.asarray(inflow_rate, dtype=jnp.float64)
            if held_rate.shape != value.shape:
                raise ValueError("Inflow rate must be a compact nodal vector.")
        supplied = (
            jnp.zeros_like(value)
            if source is None
            else jnp.asarray(source, dtype=jnp.float64)
        )
        if supplied.shape != value.shape:
            raise ValueError("Transport source must be a compact nodal content rate.")
        flux = self._advecting_flux(advection)
        reconstruction = self.reconstruction
        low, graph_outflow = _upwind_edge_flux(exterior.incidence, flux.edge, value)
        boundary_out = jnp.maximum(flux.boundary, 0.0)
        boundary_in = jnp.maximum(-flux.boundary, 0.0)
        boundary_rate = boundary_in * incoming - boundary_out * value
        outflow = graph_outflow + boundary_out
        inflow_node = self.boundary_mask & (boundary_in > 0)
        lower, upper = _neighbor_extrema(exterior.incidence, value)
        lower = jnp.where(inflow_node, jnp.minimum(lower, incoming), lower)
        upper = jnp.where(inflow_node, jnp.maximum(upper, incoming), upper)
        if reconstruction is None:
            antidiffusive = jnp.zeros_like(flux.edge)
        elif isinstance(advection, TransportVolumeFlux):
            antidiffusive = self._antidiffusive(flux.edge, value, reconstruction)
        else:
            content = value[:, None] * self._velocity(advection)
            antidiffusive = self._hermite_flux(content, reconstruction) - low
        match self.scheme:
            case "upwind":
                limiter = jnp.zeros_like(flux.edge)
            case "reconstructed":
                limiter = jnp.ones_like(flux.edge)
            case "limited":
                limiter = _limited_coefficients(
                    exterior.incidence,
                    antidiffusive,
                    value,
                    lower,
                    upper,
                    self.limiter_capacity * outflow,
                )
            case unknown:
                assert_never(unknown)
        edge_flux = low + limiter * antidiffusive
        graph_rate = exterior.incidence.transpose_mv(edge_flux)
        if held_rate is not None:
            # Strong inflow rows: the boundary supplies whatever content holds
            # the prescribed state against the graph fluxes; it is ledgered.
            boundary_rate = jnp.where(
                inflow_node, volumes * held_rate - graph_rate - supplied, boundary_rate
            )
        content_rate = graph_rate + boundary_rate + supplied
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(incoming))
            & jnp.all(jnp.isfinite(content_rate))
        )
        metric = exterior.metric_result.accepted
        status = jnp.where(
            ~finite,
            int(TransportStatus.NONFINITE),
            jnp.where(
                ~metric,
                int(TransportStatus.METRIC_REFUSED),
                int(TransportStatus.ACCEPTED),
            ),
        ).astype(jnp.int32)
        return TransportRate(
            content_rate=content_rate,
            value_rate=content_rate / volumes,
            edge_flux=edge_flux,
            low_order_flux=low,
            antidiffusive_flux=antidiffusive,
            limiter=limiter,
            volume_flux=flux.edge,
            boundary_volume_flux=flux.boundary,
            boundary_content_rate=boundary_rate,
            node_outflow=outflow,
            lower_bound=lower,
            upper_bound=upper,
            limiter_switched=jnp.sum(
                (antidiffusive != 0) & (limiter < 1), dtype=jnp.int32
            ),
            conservation_residual=jnp.sum(content_rate)
            - jnp.sum(boundary_rate)
            - jnp.sum(supplied),
            finite=finite,
            metric_admitted=metric,
            status=status,
        )

    def cfl(
        self, advection: ArrayLike | TransportVolumeFlux, step_size: ArrayLike, /
    ) -> TransportCFL:
        """Forward-Euler CFL certificate of the scheme at one advecting flux."""
        dt = jnp.asarray(step_size, dtype=jnp.float64)
        if dt.shape != ():
            raise ValueError("CFL step size must be a scalar.")
        exterior = self.exterior
        flux = self._advecting_flux(advection)
        _, graph_outflow = _upwind_edge_flux(
            exterior.incidence, flux.edge, jnp.ones_like(exterior.node_volumes)
        )
        outflow = graph_outflow + jnp.maximum(flux.boundary, 0.0)
        match self.scheme:
            case "upwind":
                demand, certified = outflow, True
            case "limited":
                demand, certified = (1.0 + self.limiter_capacity) * outflow, True
            case "reconstructed":
                demand, certified = outflow, False
            case unknown:
                assert_never(unknown)
        volumes = exterior.node_volumes
        node_cfl = dt * demand / volumes
        cfl = jnp.max(node_cfl)
        certified_ = jnp.asarray(certified)
        return TransportCFL(
            node_cfl=node_cfl,
            cfl=cfl,
            step_bound=jnp.min(
                jnp.where(
                    demand > 0, volumes / jnp.where(demand > 0, demand, 1.0), jnp.inf
                )
            ),
            certified=certified_,
            admitted=certified_ & jnp.isfinite(dt) & (dt >= 0) & (cfl <= 1),
        )

    def refresh(
        self, points: ArrayLike, /, *, node_volumes: ArrayLike | None = None
    ) -> TransportRefresh:
        """Refresh moved geometry on the frozen graph topology and support.

        The exterior radius-topology witness and the reconstruction support are
        checked before any metric solve; a refusal returns the unchanged owner
        with status, never a repaired candidate. Periodic exteriors recompute
        minimum-image endpoint charts from the edge relation's address.

        This is an eager geometry-preparation boundary: the re-solved metric
        carries static rank-certificate metadata that differs from the anchored
        owner, so admission is decided on the host and the selected owner is
        returned as a whole rather than merged leafwise.
        """
        exterior = self.exterior
        coordinates = jnp.asarray(points, dtype=jnp.float64)
        if coordinates.shape != exterior.points.shape:
            raise ValueError("Transport refresh preserves compact node coordinates.")
        finite = jnp.all(jnp.isfinite(coordinates))
        # The exterior witness measures raw coordinate motion from its anchor;
        # periodic consumers pass continuous (unwrapped) trajectories.
        trusted = exterior.within_topology_trust(coordinates)
        reconstruction = self.reconstruction
        reconstructed = jnp.asarray(True)
        if reconstruction is not None:
            candidate = reconstruction.refresh(coordinates)
            reconstructed = candidate.accepted
        geometric = finite & trusted & reconstructed

        def refreshed(owner: ConservativeTransport) -> ConservativeTransport:
            charts = self._charts(coordinates) if owner.exterior.intrinsic else None
            moved = owner.exterior.refresh(
                points=coordinates,
                node_volumes=node_volumes,
                intrinsic_displacements=charts,
            )
            bound = (
                None
                if owner.reconstruction is None
                else owner.reconstruction.refresh(coordinates).discretization
            )
            return eqx.tree_at(
                lambda item: (item.exterior, item.reconstruction),
                owner,
                (moved, bound),
                is_leaf=lambda leaf: leaf is None,
            )

        # Host decision at the preparation boundary: no metric solve runs for a
        # refused geometry, and the metric admission selects the whole owner.
        candidate_owner = refreshed(self) if bool(geometric) else self
        metric = candidate_owner.exterior.metric_result.accepted
        accepted = geometric & metric
        status = jnp.where(
            ~finite,
            int(TransportRefreshStatus.INVALID_COORDINATES),
            jnp.where(
                ~trusted,
                int(TransportRefreshStatus.TOPOLOGY_TRUST_EXCEEDED),
                jnp.where(
                    ~reconstructed,
                    int(TransportRefreshStatus.RECONSTRUCTION_REFUSED),
                    jnp.where(
                        ~metric,
                        int(TransportRefreshStatus.METRIC_REFUSED),
                        int(TransportRefreshStatus.ACCEPTED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        owner = candidate_owner if bool(accepted) else self
        return TransportRefresh(
            transport=owner,
            status=status,
            accepted=accepted,
            displacement=exterior.displacement(coordinates),
            trust_margin=jnp.asarray(exterior.topology_trust_margin, dtype=jnp.float64),
        )

    def _charts(self, coordinates: Array, /) -> Array:
        exterior = self.exterior
        relation = exterior.edge_relation.relation
        capacity = (
            jnp.zeros(
                (relation.source_size, coordinates.shape[1]), dtype=coordinates.dtype
            )
            .at[exterior.capacity_indices]
            .set(coordinates)
        )
        charts = minimum_image_edge_charts(exterior.edge_relation, capacity)
        return charts[exterior.edge_capacity_indices]

    def _advecting_flux(
        self, advection: ArrayLike | TransportVolumeFlux, /
    ) -> TransportVolumeFlux:
        if not isinstance(advection, TransportVolumeFlux):
            return self.volume_flux(advection)
        exterior = self.exterior
        if (
            advection.edge.shape != exterior.lengths.shape
            or advection.boundary.shape != exterior.node_volumes.shape
        ):
            raise ValueError("Volume flux must match compact graph edges and nodes.")
        return advection

    def _velocity(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity, dtype=jnp.float64)
        if value.shape != self.exterior.points.shape:
            raise ValueError(
                "Transport velocity must be nodal, shape (nodes, dimension)."
            )
        return value

    def _inflow(self, inflow: ArrayLike | None, value: Array, /) -> Array:
        if inflow is None:
            if self.has_boundary:
                raise ValueError(
                    "A transport graph with boundary nodes requires a declared inflow state."
                )
            return jnp.zeros_like(value)
        if not self.has_boundary:
            raise ValueError("A closed transport graph has no inflow boundary.")
        incoming = jnp.asarray(inflow, dtype=jnp.float64)
        if incoming.shape != value.shape:
            raise ValueError("Inflow state must be a compact nodal vector.")
        return incoming

    def _antidiffusive(
        self,
        flux: Array,
        value: Array,
        reconstruction: PreparedPointCloudDiscretization,
        /,
    ) -> Array:
        """``F (c_face - c_up)`` with upwind-biased GMLS-gradient face values."""
        relation = self.exterior.incidence.relation
        edges = flux.shape[0]
        gradient = reconstruction.gradient(value)
        endpoint_value = gather_routes(relation, value).reshape((edges, 2))
        endpoint_gradient = gather_routes(relation, gradient).reshape(
            (edges, 2, gradient.shape[1])
        )
        half = 0.5 * self.edge_vectors
        forward = flux >= 0
        upwind = jnp.where(forward, endpoint_value[:, 0], endpoint_value[:, 1])
        face = jnp.where(
            forward,
            endpoint_value[:, 0] + jnp.sum(endpoint_gradient[:, 0] * half, axis=1),
            endpoint_value[:, 1] - jnp.sum(endpoint_gradient[:, 1] * half, axis=1),
        )
        return flux * (face - upwind)


@final
class TransportRefresh(StrictModule):
    """Fixed-topology geometry refresh candidate of a transport owner.

    ``transport`` is the refreshed owner when ``accepted`` and otherwise the
    unchanged owner (rollback); a refused topology trust is an epoch boundary,
    never repaired here.
    """

    __strict_contract__ = True
    transport: ConservativeTransport
    status: Int32[Scalar]
    accepted: Bool[Scalar]
    displacement: Float64[Scalar]
    trust_margin: Float64[Scalar]


def _require_reconstruction(
    reconstruction: PreparedPointCloudDiscretization | None, points: np.ndarray, /
) -> None:
    if not isinstance(reconstruction, PreparedPointCloudDiscretization):
        raise TypeError(
            "Reconstructed transport requires a PreparedPointCloudDiscretization."
        )
    cloud = np.asarray(jax.device_get(reconstruction.points))
    scale = max(1.0, float(np.max(np.abs(points))))
    if cloud.shape != points.shape or not np.allclose(
        cloud, points, rtol=0.0, atol=1e-12 * scale
    ):
        raise ValueError(
            "Reconstruction must be prepared on exactly the exterior's compact nodes."
        )


__all__ = [
    "ConservativeTransport",
    "TransportCFL",
    "TransportRate",
    "TransportRefresh",
    "TransportRefreshStatus",
    "TransportScheme",
    "TransportStatus",
    "TransportVolumeFlux",
    "minimum_image_edge_charts",
]
