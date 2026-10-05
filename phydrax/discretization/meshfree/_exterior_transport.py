#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Oriented graph volume-flux transport kernels and extensive conservation ledgers.

The instantaneous conservative edge rate is separate from time integration:
``MeshfreeAdvection.rate`` and the prepared ``_transport`` owner publish content
rates for native temporal methods, while ``edge_upwind_content`` is the explicit
forward-Euler admission certificate of the low-order rate (the SSP base step),
retained for one-step remaps such as the surface shift.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._admissibility import guard_derivative_validity
from ..._strict import StrictModule
from ...sparse import (
    EdgeRelation,
    gather_routes,
    linear_apply,
    linear_transpose_apply,
    route_reduce,
    SparseCoordinateOperator,
)
from ...typing import Bool, checked, Float64, Int32, Scalar
from ._exterior import (
    _ExteriorEdgeDim,
    _ExteriorNodeDim,
    PreparedMeshfreeExteriorCalculus,
)


class MeshfreeAdvectionStatus(IntEnum):
    ACCEPTED = 0
    METRIC_REFUSED = 1
    CFL_EXCEEDED = 2
    POSITIVITY_NOT_ADMITTED = 3
    NONFINITE = 4


@final
class MeshfreeAdvectionResult(StrictModule):
    """Accepted public update, raw candidate diagnostics, and independent evidence.

    `value` and `content` are the accepted public nodal values and extensive
    content: the candidate when `status` is `ACCEPTED`, otherwise the unchanged
    input. They are derivative-guarded, so a refused step yields invalid (NaN)
    tangents instead of a successful derivative through a held state.
    `candidate_value` and `candidate_content` keep the raw, unclipped candidate
    for inspection; the content ledger, edge flux, CFL, and positivity evidence
    describe that candidate.
    """

    __strict_contract__ = True
    value: Float64[_ExteriorNodeDim]
    content: Float64[_ExteriorNodeDim]
    candidate_value: Float64[_ExteriorNodeDim]
    candidate_content: Float64[_ExteriorNodeDim]
    content_before: Float64[Scalar]
    content_after: Float64[Scalar]
    source_content: Float64[Scalar]
    conservation_residual: Float64[Scalar]
    edge_flux: Float64[_ExteriorEdgeDim]
    cfl: Float64[Scalar]
    node_cfl: Float64[_ExteriorNodeDim]
    node_outflow: Float64[_ExteriorNodeDim]
    positivity_admitted: Bool[Scalar]
    accepted: Bool[Scalar]
    status: Int32[Scalar]


@final
class MeshfreeAdvectionRate(StrictModule):
    """Instantaneous low-order content rate of a prescribed oriented volume flux.

    ``content_rate`` is ``+B^T(edge_flux) + source``; ``value_rate`` divides it by
    the node volumes. ``stable_step`` is the forward-Euler positivity bound
    ``min_i V_i / outflow_i`` of this rate (``inf`` without outflow); it is a
    spatial certificate, not a property of every temporal method. ``status`` is
    ``ACCEPTED``, ``METRIC_REFUSED`` or ``NONFINITE``.
    """

    __strict_contract__ = True
    content_rate: Float64[_ExteriorNodeDim]
    value_rate: Float64[_ExteriorNodeDim]
    edge_flux: Float64[_ExteriorEdgeDim]
    node_outflow: Float64[_ExteriorNodeDim]
    source_rate: Float64[Scalar]
    conservation_residual: Float64[Scalar]
    stable_step: Float64[Scalar]
    finite: Bool[Scalar]
    metric_admitted: Bool[Scalar]
    status: Int32[Scalar]


def _require_incidence(incidence: SparseCoordinateOperator, /) -> EdgeRelation:
    if (
        not isinstance(incidence, SparseCoordinateOperator)
        or not isinstance(incidence.relation, EdgeRelation)
        or incidence.block_shape is not None
    ):
        raise TypeError("Upwind content requires prepared scalar native graph incidence.")
    return incidence.relation


def _upwind_edge_flux(
    incidence: SparseCoordinateOperator, flux: Array, value: Array, /
) -> tuple[Array, Array]:
    """Donor-cell content flux along canonical orientation and nodal outflow.

    Routes follow the native incidence ``B=(-1,+1)``; the donor of a positive
    oriented flux is the tail. Gather and accumulation are native relation
    actions; the outgoing volume flux is the per-node CFL denominator.
    """
    relation = _require_incidence(incidence)
    route_flux = gather_routes(relation.transpose(), flux)
    donor = ((route_flux >= 0) & (incidence.coefficients < 0)) | (
        (route_flux < 0) & (incidence.coefficients > 0)
    )
    donor_coefficients = donor.astype(jnp.float64)
    edge_flux = flux * linear_apply(relation, donor_coefficients, value)
    outflow = linear_transpose_apply(
        relation,
        jnp.abs(route_flux) * donor_coefficients,
        jnp.ones_like(flux),
    )
    return edge_flux, outflow


def _edge_volume_flux(
    incidence: SparseCoordinateOperator,
    weights: Array,
    edge_vectors: Array,
    velocity: Array,
    /,
) -> Array:
    """Integrated volume flux ``w_e (u_i+u_j)/2 · e_ij`` of a nodal velocity.

    With an accepted degree-two metric, ``sum_j w_ij e_ij = 0`` and
    ``sum_j w_ij e_ij e_ij^T = 2 V_i I`` at equation nodes, so the outgoing
    graph flux of an affine velocity equals ``V_i div u`` exactly. The flux is
    already integrated: it is never multiplied by the metric again.
    """
    endpoints = gather_routes(incidence.relation, velocity).reshape(
        (weights.shape[0], 2, velocity.shape[1])
    )
    mean = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
    return weights * jnp.sum(mean * edge_vectors, axis=1)


def _neighbor_extrema(
    incidence: SparseCoordinateOperator, value: Array, /
) -> tuple[Array, Array]:
    """Nodal minimum/maximum over each node and its graph neighbors."""
    relation = _require_incidence(incidence)
    edges = relation.target_size
    other = jnp.flip(gather_routes(relation, value).reshape((edges, 2)), axis=1)
    incoming = relation.transpose()
    degree = route_reduce(incoming, jnp.ones((2 * edges,), dtype=jnp.int32))
    upper = route_reduce(incoming, other.reshape(-1), reduction="max")
    lower = route_reduce(incoming, other.reshape(-1), reduction="min")
    connected = degree > 0
    return (
        jnp.where(connected, jnp.minimum(value, lower), value),
        jnp.where(connected, jnp.maximum(value, upper), value),
    )


def _limited_coefficients(
    incidence: SparseCoordinateOperator,
    antidiffusive: Array,
    value: Array,
    lower: Array,
    upper: Array,
    capacity: Array,
    /,
) -> Array:
    """Semi-discrete Zalesak/Kuzmin edge coefficients ``alpha in [0, 1]``.

    Node ``i`` may receive limited antidiffusive content at most
    ``capacity_i (upper_i - c_i)`` and lose at most
    ``capacity_i (c_i - lower_i)``; paired edge coefficients take the
    minimum of the receiving node's and the donating node's ratios, so the
    corrected flux remains conservative. The coefficients are a nonsmooth
    limiter decision: they are returned without derivative (frozen-limiter
    tangents), and consumers report where they switched.
    """
    relation = _require_incidence(incidence)
    incoming = relation.transpose()
    contribution = incidence.coefficients * gather_routes(incoming, antidiffusive)
    gained = route_reduce(incoming, jnp.maximum(contribution, 0.0))
    lost = route_reduce(incoming, jnp.minimum(contribution, 0.0))
    room_up = capacity * (upper - value)
    room_down = capacity * (lower - value)
    ratio_up = jnp.where(
        gained > 0, jnp.minimum(1.0, room_up / jnp.where(gained > 0, gained, 1.0)), 1.0
    )
    ratio_down = jnp.where(
        lost < 0, jnp.minimum(1.0, room_down / jnp.where(lost < 0, lost, -1.0)), 1.0
    )
    edges = relation.target_size
    up = gather_routes(relation, ratio_up).reshape((edges, 2))
    down = gather_routes(relation, ratio_down).reshape((edges, 2))
    alpha = jnp.where(
        antidiffusive >= 0,
        jnp.minimum(up[:, 1], down[:, 0]),
        jnp.minimum(up[:, 0], down[:, 1]),
    )
    return jax.lax.stop_gradient(alpha)


def edge_upwind_content(
    concentration: ArrayLike,
    measures: ArrayLike,
    incidence: SparseCoordinateOperator,
    oriented_volume_flux: ArrayLike,
    dt: ArrayLike,
    *,
    metric_nonnegative: ArrayLike,
    metric_accepted: ArrayLike = True,
    source: ArrayLike | None = None,
) -> MeshfreeAdvectionResult:
    """Forward-Euler admission certificate of the low-order conservative rate.

    ``incidence`` is the already-prepared canonical native graph B=(-1,+1).
    ``source`` is an extensive content rate. This is one explicit remap step
    (the SSP base step), not a time integrator: transient problems publish
    ``MeshfreeAdvection.rate`` or the prepared transport rate to native
    temporal methods. The raw candidate is retained as
    ``candidate_value``/``candidate_content`` on any refusal, never clipped into
    a fictitious positive result; the public ``value``/``content`` are accepted
    only for ``ACCEPTED`` status and otherwise hold the input, derivative-guarded.
    """
    value = jnp.asarray(concentration, dtype=jnp.float64)
    volumes = jnp.asarray(measures, dtype=jnp.float64)
    flux = jnp.asarray(oriented_volume_flux, dtype=jnp.float64)
    timestep = jnp.asarray(dt, dtype=jnp.float64)
    sign_ok = jnp.asarray(metric_nonnegative, dtype=jnp.bool_)
    accepted = jnp.asarray(metric_accepted, dtype=jnp.bool_)
    _require_incidence(incidence)
    if (
        value.shape != (incidence.source.size,)
        or volumes.shape != value.shape
        or flux.shape != (incidence.target.size,)
        or timestep.shape != ()
        or sign_ok.shape != ()
        or accepted.shape != ()
    ):
        raise ValueError(
            "Upwind content requires native nodal/edge vectors and scalar controls."
        )
    volumes = eqx.error_if(
        volumes,
        jnp.any(~jnp.isfinite(volumes) | (volumes <= 0)),
        "Upwind measures must be finite and positive.",
    )
    timestep = eqx.error_if(
        timestep,
        ~jnp.isfinite(timestep) | (timestep < 0),
        "Advection dt must be finite and nonnegative.",
    )
    source_value = (
        jnp.zeros_like(value)
        if source is None
        else jnp.asarray(source, dtype=jnp.float64)
    )
    if source_value.shape != value.shape:
        raise ValueError("Advection source must be a compact nodal content-rate vector.")
    edge_flux, outflow = _upwind_edge_flux(incidence, flux, value)
    content_rate = incidence.transpose_mv(edge_flux) + source_value
    content = volumes * value
    updated_content = content + timestep * content_rate
    updated = updated_content / volumes
    node_cfl = timestep * outflow / volumes
    cfl = jnp.max(node_cfl)
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(flux))
        & jnp.all(jnp.isfinite(source_value))
        & jnp.all(jnp.isfinite(updated))
    )
    metric_admitted = accepted & sign_ok
    positivity = (
        finite
        & metric_admitted
        & (cfl <= 1)
        & jnp.all(value >= 0)
        & jnp.all(source_value >= 0)
    )
    before = jnp.sum(content)
    after = jnp.sum(updated_content)
    supplied = timestep * jnp.sum(source_value)
    status = jnp.where(
        ~finite,
        int(MeshfreeAdvectionStatus.NONFINITE),
        jnp.where(
            ~metric_admitted,
            int(MeshfreeAdvectionStatus.METRIC_REFUSED),
            jnp.where(
                cfl > 1,
                int(MeshfreeAdvectionStatus.CFL_EXCEEDED),
                jnp.where(
                    ~positivity,
                    int(MeshfreeAdvectionStatus.POSITIVITY_NOT_ADMITTED),
                    int(MeshfreeAdvectionStatus.ACCEPTED),
                ),
            ),
        ),
    ).astype(jnp.int32)
    step_accepted = status == int(MeshfreeAdvectionStatus.ACCEPTED)
    public_value, public_content = guard_derivative_validity(
        (
            jnp.where(step_accepted, updated, value),
            jnp.where(step_accepted, updated_content, content),
        ),
        step_accepted,
        dependencies=(value, volumes, flux, timestep, source_value),
        failure="status",
        message="Meshfree advection derivatives require an accepted step.",
    )
    return MeshfreeAdvectionResult(
        value=public_value,
        content=public_content,
        candidate_value=updated,
        candidate_content=updated_content,
        content_before=before,
        content_after=after,
        source_content=supplied,
        conservation_residual=after - before - supplied,
        edge_flux=edge_flux,
        cfl=cfl,
        node_cfl=node_cfl,
        node_outflow=outflow,
        positivity_admitted=positivity,
        accepted=step_accepted,
        status=status,
    )


@final
class MeshfreeAdvection(StrictModule):
    """Low-order upwind rate provider in canonical edge orientation, i<j.

    Positive oriented_volume_flux flows i->j. With native B=(-1,+1),
    divergence=-B^T flux/V and content_dot=+B^T flux. Flux is already an
    integrated volume flux: multiplying by the metric a second time is wrong.
    The provider publishes the instantaneous rate only; native temporal
    methods own integration. Positivity of the forward-Euler base step needs
    nonnegative data and the *outgoing*, not net, CFL bound at every node
    (``stable_step``); SSP methods inherit it only under their own coefficient.
    """

    exterior: PreparedMeshfreeExteriorCalculus

    @checked
    def __init__(self, exterior: PreparedMeshfreeExteriorCalculus, /) -> None:
        self.exterior = exterior

    def rate(
        self,
        values: ArrayLike,
        oriented_volume_flux: ArrayLike,
        /,
        *,
        source: ArrayLike | None = None,
    ) -> MeshfreeAdvectionRate:
        exterior = self.exterior
        value = jnp.asarray(values, dtype=jnp.float64)
        flux = jnp.asarray(oriented_volume_flux, dtype=jnp.float64)
        volumes = exterior.node_volumes
        if value.shape != volumes.shape or flux.shape != exterior.lengths.shape:
            raise ValueError(
                "Advection rates require compact nodal values and canonical edge fluxes."
            )
        source_value = (
            jnp.zeros_like(value)
            if source is None
            else jnp.asarray(source, dtype=jnp.float64)
        )
        if source_value.shape != value.shape:
            raise ValueError(
                "Advection source must be a compact nodal content-rate vector."
            )
        edge_flux, outflow = _upwind_edge_flux(exterior.incidence, flux, value)
        content_rate = exterior.incidence.transpose_mv(edge_flux) + source_value
        supplied = jnp.sum(source_value)
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(content_rate))
        )
        metric = exterior.metric_result.accepted
        status = jnp.where(
            ~finite,
            int(MeshfreeAdvectionStatus.NONFINITE),
            jnp.where(
                ~metric,
                int(MeshfreeAdvectionStatus.METRIC_REFUSED),
                int(MeshfreeAdvectionStatus.ACCEPTED),
            ),
        ).astype(jnp.int32)
        return MeshfreeAdvectionRate(
            content_rate=content_rate,
            value_rate=content_rate / volumes,
            edge_flux=edge_flux,
            node_outflow=outflow,
            source_rate=supplied,
            conservation_residual=jnp.sum(content_rate) - supplied,
            stable_step=jnp.min(
                jnp.where(
                    outflow > 0, volumes / jnp.where(outflow > 0, outflow, 1.0), jnp.inf
                )
            ),
            finite=finite,
            metric_admitted=metric,
            status=status,
        )
