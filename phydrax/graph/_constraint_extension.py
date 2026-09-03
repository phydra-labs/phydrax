#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..enforcement._geometry_support import BoundaryCover
from ..geometry._trace_extension import (
    DiscreteTraceCorrectionProvider,
    PreparedTraceExtension,
)
from ..linalg._operators import AbstractLinearOperator


class GraphHarmonicCorrectionProvider(StrictModule, NonTrainableState):
    """Exact finite graph-harmonic extension from anchored restrictions.

    ``candidate_operator`` owns the native free-block Laplacian solve.  This
    wrapper certifies its restriction response jointly and refuses an uncontrolled
    constant mode on any connected component.
    """

    restriction_operator: AbstractLinearOperator
    candidate_operator: AbstractLinearOperator
    harmonic_residual_operator: AbstractLinearOperator
    cover: BoundaryCover
    topology_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    restriction_provider_id: str = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    anchored_components: tuple[int, ...] = eqx.field(static=True)
    gauged_components: tuple[int, ...] = eqx.field(static=True)
    gauge_certificate_id: str | None = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        restriction_operator: AbstractLinearOperator,
        candidate_operator: AbstractLinearOperator,
        harmonic_residual_operator: AbstractLinearOperator,
        cover: BoundaryCover,
        /,
        *,
        topology_id: str,
        metric_id: str,
        restriction_provider_id: str,
        component_count: int,
        anchored_components: tuple[int, ...],
        gauged_components: tuple[int, ...] = (),
        gauge_certificate_id: str | None = None,
    ):
        operators = (
            restriction_operator,
            candidate_operator,
            harmonic_residual_operator,
        )
        if not all(isinstance(value, AbstractLinearOperator) for value in operators):
            raise TypeError(
                "Graph restriction, extension, and residual maps must be linear."
            )
        if not isinstance(cover, BoundaryCover):
            raise TypeError("cover must be BoundaryCover.")
        count = int(component_count)
        if count < 1:
            raise ValueError("component_count must be positive.")
        anchored = tuple(sorted(set(int(value) for value in anchored_components)))
        gauged = tuple(sorted(set(int(value) for value in gauged_components)))
        valid = set(range(count))
        if not set(anchored).issubset(valid) or not set(gauged).issubset(valid):
            raise ValueError(
                "Anchored and gauged components must be valid component IDs."
            )
        if set(anchored) & set(gauged):
            raise ValueError("A graph component cannot be both anchored and gauge-only.")
        uncontrolled = valid - set(anchored) - set(gauged)
        if uncontrolled:
            raise ValueError(
                f"Graph harmonic extension leaves components {tuple(sorted(uncontrolled))} "
                "without anchors or gauges."
            )
        gauge_id = None if gauge_certificate_id is None else str(gauge_certificate_id)
        if gauged and (gauge_id is None or not gauge_id):
            raise ValueError("Gauged graph components require a gauge certificate.")
        topology = str(topology_id)
        metric = str(metric_id)
        restriction_provider = str(restriction_provider_id)
        if not topology or not metric or not restriction_provider:
            raise ValueError(
                "Graph topology, metric, and restriction-provider IDs are required."
            )
        if not harmonic_residual_operator.source.compatible(restriction_operator.source):
            raise ValueError("Harmonic residual and restriction act on different graphs.")
        self.restriction_operator = restriction_operator
        self.candidate_operator = candidate_operator
        self.harmonic_residual_operator = harmonic_residual_operator
        self.cover = cover
        self.topology_id = topology
        self.metric_id = metric
        self.restriction_provider_id = restriction_provider
        self.component_count = count
        self.anchored_components = anchored
        self.gauged_components = gauged
        self.gauge_certificate_id = gauge_id
        self.provider_id = canonical_fingerprint(
            {
                "kind": "graph-harmonic-correction-v1",
                "restriction": restriction_operator.operator_id,
                "candidate": candidate_operator.operator_id,
                "harmonic_residual": harmonic_residual_operator.operator_id,
                "cover": cover.cover_id,
                "topology": topology,
                "metric": metric,
                "restriction_provider": restriction_provider,
                "component_count": count,
                "anchored": anchored,
                "gauged": gauged,
                "gauge_certificate": gauge_id,
            }
        )

    def prepare(
        self, /, *, rank=None, resources=None, numeric_version=0
    ) -> PreparedTraceExtension:
        construction_id = canonical_fingerprint(
            {
                "kind": "graph-harmonic-system-v1",
                "provider": self.provider_id,
                "topology": self.topology_id,
                "metric": self.metric_id,
                "gauge": self.gauge_certificate_id,
            }
        )
        provider = DiscreteTraceCorrectionProvider(
            self.restriction_operator,
            self.candidate_operator,
            self.cover,
            representation_id=f"graph-harmonic/{self.topology_id}",
            representation_certificate_id=construction_id,
            preservation_operator=self.harmonic_residual_operator,
            preservation_certificate_id=canonical_fingerprint(
                {
                    "kind": "graph-free-harmonic-residual-v1",
                    "operator": self.harmonic_residual_operator.operator_id,
                    "topology": self.topology_id,
                }
            ),
            stability_owner_ids=(self.metric_id,),
        )
        return provider.prepare(
            rank=rank,
            resources=resources,
            numeric_version=numeric_version,
        )


__all__ = ["GraphHarmonicCorrectionProvider"]
