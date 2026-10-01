#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Oriented graph volume-flux transport and extensive conservation ledgers."""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    SparseCoordinateOperator,
)
from ...typing import Bool, Float64, Int32, Scalar
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
    """Actual extensive update and independent conservation/positivity evidence."""

    __strict_contract__ = True
    value: Float64[_ExteriorNodeDim]
    content: Float64[_ExteriorNodeDim]
    content_before: Float64[Scalar]
    content_after: Float64[Scalar]
    source_content: Float64[Scalar]
    conservation_residual: Float64[Scalar]
    edge_flux: Float64[_ExteriorEdgeDim]
    cfl: Float64[Scalar]
    node_cfl: Float64[_ExteriorNodeDim]
    node_outflow: Float64[_ExteriorNodeDim]
    positivity_admitted: Bool[Scalar]
    status: Int32[Scalar]


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
    """Shared conservative edge kernel for fixed and moving meshfree owners.

    ``incidence`` is the already-prepared canonical native graph B=(-1,+1).
    ``source`` is an extensive content rate. The raw candidate is retained on
    CFL or metric refusal, never clipped into a fictitious positive result.
    """
    value = jnp.asarray(concentration, dtype=jnp.float64)
    volumes = jnp.asarray(measures, dtype=jnp.float64)
    flux = jnp.asarray(oriented_volume_flux, dtype=jnp.float64)
    timestep = jnp.asarray(dt, dtype=jnp.float64)
    sign_ok = jnp.asarray(metric_nonnegative, dtype=jnp.bool_)
    accepted = jnp.asarray(metric_accepted, dtype=jnp.bool_)
    if (
        not isinstance(incidence, SparseCoordinateOperator)
        or not isinstance(incidence.relation, EdgeRelation)
        or incidence.block_shape is not None
    ):
        raise TypeError("Upwind content requires prepared scalar native graph incidence.")
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
    relation = incidence.relation
    route_flux = flux[relation.target_indices]
    donor = ((route_flux >= 0) & (incidence.coefficients < 0)) | (
        (route_flux < 0) & (incidence.coefficients > 0)
    )
    donor_coefficients = donor.astype(jnp.float64)
    edge_flux = flux * linear_apply(relation, donor_coefficients, value)
    content_rate = incidence.transpose_mv(edge_flux) + source_value
    content = volumes * value
    updated_content = content + timestep * content_rate
    updated = updated_content / volumes
    outflow = linear_transpose_apply(
        relation,
        jnp.abs(route_flux) * donor_coefficients,
        jnp.ones_like(flux),
    )
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
    return MeshfreeAdvectionResult(
        value=updated,
        content=updated_content,
        content_before=before,
        content_after=after,
        source_content=supplied,
        conservation_residual=after - before - supplied,
        edge_flux=edge_flux,
        cfl=cfl,
        node_cfl=node_cfl,
        node_outflow=outflow,
        positivity_admitted=positivity,
        status=status,
    )


@final
class MeshfreeAdvection(StrictModule):
    """First-order upwind in canonical edge orientation, i<j.

    Positive oriented_volume_flux flows i->j. With native B=(-1,+1),
    divergence=-B^T flux/V and content_dot=+B^T flux. Flux is already an
    integrated volume flux: multiplying by the metric a second time is wrong.
    Positivity needs accepted nonnegative metric, nonnegative initial/source
    content and the *outgoing*, not net, CFL bound at every node.
    """

    exterior: PreparedMeshfreeExteriorCalculus

    def __init__(self, exterior: PreparedMeshfreeExteriorCalculus, /) -> None:
        if not isinstance(exterior, PreparedMeshfreeExteriorCalculus):
            raise TypeError("Advection requires one prepared meshfree exterior owner.")
        self.exterior = exterior

    def step(
        self,
        values: ArrayLike,
        oriented_volume_flux: ArrayLike,
        dt: ArrayLike,
        *,
        source: ArrayLike | None = None,
    ) -> MeshfreeAdvectionResult:
        return edge_upwind_content(
            values,
            self.exterior.node_volumes,
            self.exterior.incidence,
            oriented_volume_flux,
            dt,
            metric_nonnegative=self.exterior.metric_result.nonnegative,
            metric_accepted=self.exterior.metric_result.accepted,
            source=source,
        )
