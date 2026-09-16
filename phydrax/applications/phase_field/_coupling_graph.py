#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PhaseFieldCouplingTerm(StrictModule, NonTrainableState):
    term_id: str = eqx.field(static=True)
    input_fields: tuple[str, ...] = eqx.field(static=True)
    output_fields: tuple[str, ...] = eqx.field(static=True)
    storage_channels: tuple[str, ...] = eqx.field(static=True)
    dissipation_channels: tuple[str, ...] = eqx.field(static=True)
    work_channels: tuple[str, ...] = eqx.field(static=True)
    exchange_inputs: tuple[str, ...] = eqx.field(static=True)
    exchange_outputs: tuple[str, ...] = eqx.field(static=True)
    conservation_channels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        term_id: str,
        /,
        *,
        input_fields: tuple[str, ...] = (),
        output_fields: tuple[str, ...] = (),
        storage_channels: tuple[str, ...] = (),
        dissipation_channels: tuple[str, ...] = (),
        work_channels: tuple[str, ...] = (),
        exchange_inputs: tuple[str, ...] = (),
        exchange_outputs: tuple[str, ...] = (),
        conservation_channels: tuple[str, ...] = (),
    ):
        identifier = str(term_id)
        groups = tuple(
            tuple(str(item) for item in values)
            for values in (
                input_fields,
                output_fields,
                storage_channels,
                dissipation_channels,
                work_channels,
                exchange_inputs,
                exchange_outputs,
                conservation_channels,
            )
        )
        if not identifier or any(
            not item or len(set(values)) != len(values)
            for values in groups
            for item in values
        ):
            raise ValueError("Coupling term identities must be nonempty and unique.")
        self.term_id = identifier
        (
            self.input_fields,
            self.output_fields,
            self.storage_channels,
            self.dissipation_channels,
            self.work_channels,
            self.exchange_inputs,
            self.exchange_outputs,
            self.conservation_channels,
        ) = groups


class PhaseFieldCouplingGraph(StrictModule, NonTrainableState):
    terms: tuple[PhaseFieldCouplingTerm, ...]
    field_names: tuple[str, ...] = eqx.field(static=True)
    storage_channels: tuple[str, ...] = eqx.field(static=True)
    exchange_channels: tuple[str, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(self, terms: tuple[PhaseFieldCouplingTerm, ...], /):
        values = tuple(terms)
        if not values or any(
            not isinstance(value, PhaseFieldCouplingTerm) for value in values
        ):
            raise ValueError("Coupling graph requires phase-field coupling terms.")
        term_ids = tuple(value.term_id for value in values)
        if len(set(term_ids)) != len(term_ids):
            raise ValueError("Coupling term IDs must be unique.")
        storage = tuple(channel for value in values for channel in value.storage_channels)
        dissipation = tuple(
            channel for value in values for channel in value.dissipation_channels
        )
        work = tuple(channel for value in values for channel in value.work_channels)
        if (
            len(set(storage)) != len(storage)
            or len(set(dissipation)) != len(dissipation)
            or len(set(work)) != len(work)
        ):
            raise ValueError(
                "Storage, dissipation, and external-work channels need one owner."
            )
        exchange_inputs = Counter(
            channel for value in values for channel in value.exchange_inputs
        )
        exchange_outputs = Counter(
            channel for value in values for channel in value.exchange_outputs
        )
        all_exchanges = tuple(sorted(set(exchange_inputs) | set(exchange_outputs)))
        if any(
            exchange_inputs[channel] != 1 or exchange_outputs[channel] != 1
            for channel in all_exchanges
        ):
            raise ValueError(
                "Every internal exchange needs exactly one source and one target."
            )
        output_owners = Counter(
            field for value in values for field in value.output_fields
        )
        if any(count != 1 for count in output_owners.values()):
            raise ValueError("Every coupled residual field needs one output owner.")
        inputs = {field for value in values for field in value.input_fields}
        outputs = set(output_owners)
        missing = inputs - outputs
        if missing:
            raise ValueError(
                f"Coupling inputs lack residual owners: {sorted(missing)!r}."
            )
        self.terms = values
        self.field_names = tuple(sorted(outputs))
        self.storage_channels = storage
        self.exchange_channels = all_exchanges
        self.graph_id = canonical_fingerprint(
            {
                "kind": "phase-field-coupling-graph",
                "terms": [
                    {
                        "id": value.term_id,
                        "inputs": value.input_fields,
                        "outputs": value.output_fields,
                        "storage": value.storage_channels,
                        "dissipation": value.dissipation_channels,
                        "work": value.work_channels,
                        "exchange_inputs": value.exchange_inputs,
                        "exchange_outputs": value.exchange_outputs,
                        "conservation": value.conservation_channels,
                    }
                    for value in values
                ],
            }
        )


__all__ = ["PhaseFieldCouplingGraph", "PhaseFieldCouplingTerm"]
