#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._interface_extension import (
    InterfaceGauge,
    OrientedInterfaceSupport,
    TwoSidedInterfaceCorrectionProvider,
)
from ...geometry._trace_extension import PreparedTraceExtension
from ...linalg._operators import AbstractLinearOperator
from ._core import BlockInterface, PreparedMultiblockGrid


class MultiblockInterfaceCorrectionProvider(StrictModule, NonTrainableState):
    """Exact admissible-state correction that preserves SBP/SAT ownership.

    This provider parameterizes a trial or initial state only.  It neither
    evaluates numerical fluxes nor replaces ``MultiblockSATCoupling`` during
    evolution; its certificate therefore retains the SAT stability owner ID.
    """

    multiblock: PreparedMultiblockGrid
    interface: BlockInterface
    support: OrientedInterfaceSupport
    trace_operator: AbstractLinearOperator
    candidate_operator: AbstractLinearOperator
    preservation_operator: AbstractLinearOperator | None
    sat_stability_owner_id: str = eqx.field(static=True)
    interpolation_certificate_id: str = eqx.field(static=True)
    gauge: InterfaceGauge = eqx.field(static=True)
    gauge_certificate_id: str | None = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        multiblock: PreparedMultiblockGrid,
        interface: BlockInterface,
        support: OrientedInterfaceSupport,
        trace_operator: AbstractLinearOperator,
        candidate_operator: AbstractLinearOperator,
        /,
        *,
        sat_stability_owner_id: str,
        interpolation_certificate_id: str,
        gauge: InterfaceGauge = "minimum_energy",
        gauge_certificate_id: str | None = None,
        preservation_operator: AbstractLinearOperator | None = None,
    ):
        if not isinstance(multiblock, PreparedMultiblockGrid):
            raise TypeError("multiblock must be PreparedMultiblockGrid.")
        if not isinstance(interface, BlockInterface):
            raise TypeError("interface must be BlockInterface.")
        if interface.name not in tuple(value.name for value in multiblock.plan.interfaces):
            raise ValueError("BlockInterface does not belong to the prepared multiblock grid.")
        report_index = tuple(value.name for value in multiblock.plan.interfaces).index(
            interface.name
        )
        report = multiblock.interface_reports[report_index]
        if not report.passed:
            raise ValueError("Multiblock physical interface evidence did not pass.")
        if not isinstance(support, OrientedInterfaceSupport):
            raise TypeError("support must be OrientedInterfaceSupport.")
        if not isinstance(trace_operator, AbstractLinearOperator) or not isinstance(
            candidate_operator, AbstractLinearOperator
        ):
            raise TypeError("Multiblock trace and candidate maps must be linear.")
        sat_owner = str(sat_stability_owner_id)
        interpolation = str(interpolation_certificate_id)
        if not sat_owner or not interpolation:
            raise ValueError("SAT stability and interpolation certificates are required.")
        if sat_owner not in support.stability_owner_ids:
            raise ValueError(
                "The oriented interface support must retain its SAT stability owner."
            )
        gauge_ = str(gauge)
        if gauge_ not in ("minimum_energy", "minus_only", "plus_only"):
            raise ValueError("Unknown multiblock interface gauge.")
        gauge_id = None if gauge_certificate_id is None else str(gauge_certificate_id)
        if gauge_ != "minimum_energy" and (gauge_id is None or not gauge_id):
            raise ValueError("One-sided multiblock gauges require a certificate.")
        self.multiblock = multiblock
        self.interface = interface
        self.support = support
        self.trace_operator = trace_operator
        self.candidate_operator = candidate_operator
        self.preservation_operator = preservation_operator
        self.sat_stability_owner_id = sat_owner
        self.interpolation_certificate_id = interpolation
        self.gauge = gauge_  # type: ignore[assignment]
        self.gauge_certificate_id = gauge_id
        self.provider_id = canonical_fingerprint(
            {
                "kind": "multiblock-interface-correction-v1",
                "multiblock": multiblock.prepared_id,
                "interface": interface.interface_id,
                "support": support.interface_id,
                "trace": trace_operator.operator_id,
                "candidate": candidate_operator.operator_id,
                "sat_owner": sat_owner,
                "interpolation": interpolation,
                "gauge": gauge_,
                "gauge_certificate": gauge_id,
            }
        )

    def prepare(self, /, *, rank=None, resources=None, numeric_version=0) -> PreparedTraceExtension:
        provider = TwoSidedInterfaceCorrectionProvider(
            self.trace_operator,
            self.candidate_operator,
            self.support,
            construction_certificate_id=self.interpolation_certificate_id,
            gauge=self.gauge,
            gauge_certificate_id=self.gauge_certificate_id,
            preservation_operator=self.preservation_operator,
        )
        return provider.prepare(
            rank=rank,
            resources=resources,
            numeric_version=numeric_version,
        )


__all__ = ["MultiblockInterfaceCorrectionProvider"]
