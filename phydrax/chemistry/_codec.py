#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact typed conversion between chemistry results and lifecycle array archives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import LifecycleArchive, ResultManifest
from ._production_lifecycle import ProductionChemistryArchivePlan


ChemistryResultEncoder = Callable[[Any], Mapping[str, object]]
ChemistryResultDecoder = Callable[[Mapping[str, np.ndarray]], Any]


class ChemistryResultCodec(StrictModule, NonTrainableState):
    """One exact result type, field layout, and host reconstruction boundary."""

    result_type: type = eqx.field(static=True)
    archive: ProductionChemistryArchivePlan
    encoder: ChemistryResultEncoder = eqx.field(static=True)
    decoder: ChemistryResultDecoder = eqx.field(static=True)
    encoder_id: str = eqx.field(static=True)
    decoder_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self,
        result_type: type,
        archive: ProductionChemistryArchivePlan,
        encoder: ChemistryResultEncoder,
        decoder: ChemistryResultDecoder,
        /,
        *,
        encoder_id: str,
        decoder_id: str,
    ):
        if not isinstance(result_type, type):
            raise TypeError("result_type must be a concrete type.")
        if not isinstance(archive, ProductionChemistryArchivePlan):
            raise TypeError("archive must be ProductionChemistryArchivePlan.")
        if not callable(encoder) or not callable(decoder):
            raise TypeError("encoder and decoder must be callable.")
        encoder_id_ = str(encoder_id).strip()
        decoder_id_ = str(decoder_id).strip()
        if not encoder_id_ or not decoder_id_:
            raise ValueError("Codec callable identities must be non-empty.")
        self.result_type = result_type
        self.archive = archive
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_id = encoder_id_
        self.decoder_id = decoder_id_
        self.codec_id = canonical_fingerprint(
            {
                "kind": "chemistry-result-codec",
                "result_type": f"{result_type.__module__}.{result_type.__qualname__}",
                "archive": archive.archive_plan_id,
                "encoder": encoder_id_,
                "decoder": decoder_id_,
            }
        )

    def encode(self, result: Any, /) -> dict[str, np.ndarray]:
        if type(result) is not self.result_type:
            raise TypeError(
                f"Codec requires exact result type {self.result_type.__name__}."
            )
        arrays = {name: np.asarray(value) for name, value in self.encoder(result).items()}
        expected = {name for name, _ in self.archive.field_units}
        if set(arrays) != expected:
            raise ValueError("Encoded result fields do not match the archive plan.")
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("Chemistry result codecs cannot emit object arrays.")
        return arrays

    def write(
        self,
        path: str | Path,
        result: Any,
        run_id: str,
        /,
        *,
        evidence_ids: tuple[str, ...] = (),
        diagnostic_ids: tuple[str, ...] = (),
    ) -> LifecycleArchive:
        arrays = self.encode(result)
        result_id = str(result.result_id).strip()
        if not result_id:
            raise ValueError("Encoded chemistry results require a result_id.")
        return self.archive.write(
            path,
            result_id,
            run_id,
            arrays,
            evidence_ids=evidence_ids,
            diagnostic_ids=diagnostic_ids,
        )

    def read(
        self,
        path: str | Path,
        /,
        *,
        expected_result_id: str | None = None,
    ) -> Any:
        archive = self.archive.open(path, expected_result_id=expected_result_id)
        result = self.decoder(archive.arrays)
        if type(result) is not self.result_type:
            raise TypeError("Chemistry codec decoder returned the wrong result type.")
        if not isinstance(archive.manifest, ResultManifest):
            raise TypeError("Chemistry result archive requires ResultManifest.")
        if str(result.result_id) != archive.manifest.result_id:
            raise ValueError(
                "Decoded chemistry result identity differs from its archive."
            )
        return result


__all__ = ["ChemistryResultCodec"]
