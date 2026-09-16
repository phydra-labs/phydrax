#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ...solver._photon_transport import PhotonTransportResult
from ..radiation_biophysics._interactions import (
    InteractionLedger,
    PhysicalInteraction,
    PrimaryHistoryKey,
    RadiationEventKey,
    RadiationSource,
)


_PROCESS_NAMES = ("photoelectric", "compton", "rayleigh")


def photon_result_to_interaction_ledger(
    result: PhotonTransportResult,
    source: RadiationSource,
    /,
    *,
    run_id: str,
    fraction_id: str,
) -> InteractionLedger:
    """Materialize deposited native photon events as governed host records.

    This is an explicit host boundary. Aggregate KERMA, escaped energy, virtual
    collisions, and events with zero local deposition are not fabricated as
    physical interaction records.
    """

    if not isinstance(result, PhotonTransportResult):
        raise TypeError("result must be PhotonTransportResult.")
    if not isinstance(source, RadiationSource):
        raise TypeError("source must be RadiationSource.")
    if not run_id or not fraction_id:
        raise ValueError("run_id and fraction_id must be nonempty.")
    history_ids = np.asarray(result.history_ids)
    active = np.asarray(result.event_active)
    positions = np.asarray(result.event_positions)
    deposited = np.asarray(result.event_deposited_energy)
    processes = np.asarray(result.event_process)
    materials = np.asarray(result.event_material)
    records = []
    for history_index, history_id in enumerate(history_ids):
        history = PrimaryHistoryKey(
            source.artifact.artifact_id,
            str(run_id),
            str(int(history_id)),
            str(fraction_id),
        )
        for event_index in np.flatnonzero(active[history_index]):
            process_index = int(processes[history_index, event_index])
            if not 0 <= process_index < len(_PROCESS_NAMES):
                raise ValueError(
                    "Native photon result contains an unknown process index."
                )
            records.append(
                PhysicalInteraction(
                    RadiationEventKey(history, "physical", f"event:{event_index}"),
                    tuple(
                        float(value) for value in positions[history_index, event_index]
                    ),
                    float(deposited[history_index, event_index]),
                    track_id=f"photon:{int(history_id)}",
                    process=_PROCESS_NAMES[process_index],
                    species="photon",
                    material=f"material:{int(materials[history_index, event_index])}",
                )
            )
    return InteractionLedger(source, tuple(records))


__all__ = ["photon_result_to_interaction_ledger"]
