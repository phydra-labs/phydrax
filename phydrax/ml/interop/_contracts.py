#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from ..._model import AbstractArrayModel, FrozenModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ConversionError(ValueError):
    """Base class for fail-closed external fitted-model conversion errors."""


class UnsupportedConversionError(ConversionError):
    """Raised when a source model uses semantics Phydrax cannot preserve exactly."""


class ConversionProvenance(StrictModule, NonTrainableState):
    """Immutable source identity and configuration copied at conversion time."""

    source: str = eqx.field(static=True)
    source_version: str = eqx.field(static=True)
    source_model: str = eqx.field(static=True)
    configuration: tuple[tuple[str, str], ...] = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    class_labels: tuple[str, ...] = eqx.field(static=True)
    license_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: str,
        source_version: str,
        source_model: str,
        configuration: Mapping[str, object] | Sequence[tuple[str, object]] = (),
        feature_names: Sequence[object] = (),
        class_labels: Sequence[object] = (),
        license_id: str,
    ):
        if isinstance(configuration, Mapping):
            items = configuration.items()
        else:
            items = configuration
        source_ = str(source)
        version_ = str(source_version)
        model_ = str(source_model)
        license_ = str(license_id)
        if not source_ or not version_ or not model_ or not license_:
            raise ValueError(
                "Conversion provenance source, version, model, and license are required."
            )
        self.source = source_
        self.source_version = version_
        self.source_model = model_
        self.configuration = tuple(
            sorted((str(name), repr(value)) for name, value in items)
        )
        self.feature_names = tuple(str(name) for name in feature_names)
        self.class_labels = tuple(str(label) for label in class_labels)
        self.license_id = license_


class ConversionResult(StrictModule):
    """A frozen native model and the audited provenance of its one-time conversion."""

    model: FrozenModel
    provenance: ConversionProvenance

    def __init__(
        self,
        model: AbstractArrayModel,
        provenance: ConversionProvenance,
        /,
    ):
        if not isinstance(provenance, ConversionProvenance):
            raise TypeError("provenance must be ConversionProvenance.")
        self.model = model if isinstance(model, FrozenModel) else FrozenModel(model)
        self.provenance = provenance


__all__ = [
    "ConversionError",
    "ConversionProvenance",
    "ConversionResult",
    "UnsupportedConversionError",
]
