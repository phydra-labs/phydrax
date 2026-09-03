#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import FieldTransfer


class CardiacTransferConfiguration(StrictModule, NonTrainableState):
    """Semantic binding for a prepared cardiac field transfer."""

    quantity_id: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    component_axes: tuple[str, ...] = eqx.field(static=True)
    source_reference_id: str = eqx.field(static=True)
    target_reference_id: str = eqx.field(static=True)
    source_field_space_id: str = eqx.field(static=True)
    target_field_space_id: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantity_id: str,
        value_unit: str,
        source_reference_id: str,
        target_reference_id: str,
        source_field_space_id: str,
        target_field_space_id: str,
        /,
        *,
        component_axes: Sequence[str] = (),
        configuration_id: str | None = None,
    ):
        identifiers = tuple(
            str(value)
            for value in (
                quantity_id,
                value_unit,
                source_reference_id,
                target_reference_id,
                source_field_space_id,
                target_field_space_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError(
                "Cardiac transfer configuration identifiers must be non-empty."
            )
        axes = tuple(str(axis) for axis in component_axes)
        if any(not axis for axis in axes) or len(set(axes)) != len(axes):
            raise ValueError("component_axes must contain unique non-empty names.")
        self.quantity_id = identifiers[0]
        self.value_unit = identifiers[1]
        self.source_reference_id = identifiers[2]
        self.target_reference_id = identifiers[3]
        self.source_field_space_id = identifiers[4]
        self.target_field_space_id = identifiers[5]
        self.component_axes = axes
        payload = {
            "kind": "cardiac-transfer-configuration",
            "quantity": identifiers[0],
            "unit": identifiers[1],
            "component_axes": list(axes),
            "source_reference": identifiers[2],
            "target_reference": identifiers[3],
            "source_field_space": identifiers[4],
            "target_field_space": identifiers[5],
        }
        if configuration_id is None:
            self.configuration_id = canonical_fingerprint(payload)
        else:
            identifier = str(configuration_id)
            if not identifier:
                raise ValueError("configuration_id must be non-empty.")
            self.configuration_id = identifier

    @classmethod
    def for_transfer(
        cls,
        transfer: FieldTransfer,
        quantity_id: str,
        value_unit: str,
        source_reference_id: str,
        target_reference_id: str,
        /,
        *,
        component_axes: Sequence[str] = (),
        configuration_id: str | None = None,
    ) -> CardiacTransferConfiguration:
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        return cls(
            quantity_id,
            value_unit,
            source_reference_id,
            target_reference_id,
            transfer.source.field_space_id,
            transfer.target.field_space_id,
            component_axes=component_axes,
            configuration_id=configuration_id,
        )


class CardiacTransferEpoch(StrictModule, NonTrainableState):
    """Geometry and reference epochs on both sides of a transfer."""

    source_geometry: Array
    target_geometry: Array
    source_reference: Array
    target_reference: Array

    def __init__(
        self,
        source_geometry: int | ArrayLike,
        target_geometry: int | ArrayLike,
        source_reference: int | ArrayLike,
        target_reference: int | ArrayLike,
        /,
    ):
        host_values = tuple(
            np.asarray(value)
            for value in (
                source_geometry,
                target_geometry,
                source_reference,
                target_reference,
            )
        )
        if any(value.shape != () for value in host_values):
            raise ValueError("Transfer epochs must be scalar values.")
        if any(not np.issubdtype(value.dtype, np.integer) for value in host_values):
            raise TypeError("Transfer epochs must be integers.")
        if any(int(value) < 0 for value in host_values):
            raise ValueError("Transfer epochs must be non-negative.")
        values = tuple(jnp.asarray(value, dtype=jnp.int32) for value in host_values)
        self.source_geometry = values[0]
        self.target_geometry = values[1]
        self.source_reference = values[2]
        self.target_reference = values[3]

    def matches(self, other: CardiacTransferEpoch, /) -> Array:
        if not isinstance(other, CardiacTransferEpoch):
            raise TypeError("other must be a CardiacTransferEpoch.")
        return (
            (self.source_geometry == other.source_geometry)
            & (self.target_geometry == other.target_geometry)
            & (self.source_reference == other.source_reference)
            & (self.target_reference == other.target_reference)
        )


class CardiacTransferEvidence(StrictModule, NonTrainableState):
    """Fail-closed runtime evidence for one cardiac transfer action."""

    source_covered: Array
    target_covered: Array
    source_coverage_fraction: Array
    target_coverage_fraction: Array
    constant_error: Array
    adjoint_error: Array
    configuration_matches: Array
    epoch_matches: Array
    coverage_complete: Array
    constant_preserved: Array
    adjoint_consistent: Array
    finite: Array
    accepted: Array
    transfer_id: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)


class CardiacTransferResult(StrictModule):
    """Candidate transferred value and the evidence required to accept it."""

    value: Any
    evidence: CardiacTransferEvidence


class CardiacFieldTransfer(StrictModule, NonTrainableState):
    """Cardiac semantic and epoch guard around a generic ``FieldTransfer``."""

    transfer: FieldTransfer
    configuration: CardiacTransferConfiguration
    prepared_epoch: CardiacTransferEpoch
    source_covered: Array
    target_covered: Array
    constant_tolerance: float = eqx.field(static=True)
    adjoint_tolerance: float = eqx.field(static=True)
    cardiac_transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        configuration: CardiacTransferConfiguration,
        prepared_epoch: CardiacTransferEpoch,
        /,
        *,
        source_covered: ArrayLike | None = None,
        target_covered: ArrayLike | None = None,
        constant_tolerance: float = 1.0e-7,
        adjoint_tolerance: float = 1.0e-7,
        cardiac_transfer_id: str | None = None,
    ):
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        if not isinstance(configuration, CardiacTransferConfiguration):
            raise TypeError("configuration must be a CardiacTransferConfiguration.")
        if not isinstance(prepared_epoch, CardiacTransferEpoch):
            raise TypeError("prepared_epoch must be a CardiacTransferEpoch.")
        if configuration.source_field_space_id != transfer.source.field_space_id:
            raise ValueError(
                "The configured source field space does not match the transfer."
            )
        if configuration.target_field_space_id != transfer.target.field_space_id:
            raise ValueError(
                "The configured target field space does not match the transfer."
            )
        if (
            transfer.source.vector_space.size <= 0
            or transfer.target.vector_space.size <= 0
        ):
            raise ValueError(
                "Cardiac transfers require non-empty source and target spaces."
            )
        source_mask_host = (
            np.ones((transfer.source.vector_space.size,), dtype=bool)
            if source_covered is None
            else np.asarray(source_covered)
        )
        target_mask_host = (
            np.ones((transfer.target.vector_space.size,), dtype=bool)
            if target_covered is None
            else np.asarray(target_covered)
        )
        if source_mask_host.shape != (transfer.source.vector_space.size,):
            raise ValueError("source_covered must have one entry per source coefficient.")
        if target_mask_host.shape != (transfer.target.vector_space.size,):
            raise ValueError("target_covered must have one entry per target coefficient.")
        if not np.issubdtype(source_mask_host.dtype, np.bool_) or not np.issubdtype(
            target_mask_host.dtype, np.bool_
        ):
            raise TypeError("Coverage arrays must be boolean.")
        constant_tolerance_ = float(constant_tolerance)
        adjoint_tolerance_ = float(adjoint_tolerance)
        if (
            not np.isfinite(constant_tolerance_)
            or constant_tolerance_ < 0.0
            or not np.isfinite(adjoint_tolerance_)
            or adjoint_tolerance_ < 0.0
        ):
            raise ValueError("Transfer tolerances must be finite and non-negative.")
        source_mask = jnp.asarray(source_mask_host, dtype=bool)
        target_mask = jnp.asarray(target_mask_host, dtype=bool)
        payload = {
            "kind": "cardiac-field-transfer",
            "field_transfer": transfer.transfer_id,
            "configuration": configuration.configuration_id,
            "epochs": [
                int(value)
                for value in (
                    np.asarray(prepared_epoch.source_geometry),
                    np.asarray(prepared_epoch.target_geometry),
                    np.asarray(prepared_epoch.source_reference),
                    np.asarray(prepared_epoch.target_reference),
                )
            ],
            "source_covered": array_tree_fingerprint(source_mask),
            "target_covered": array_tree_fingerprint(target_mask),
            "constant_tolerance": constant_tolerance_,
            "adjoint_tolerance": adjoint_tolerance_,
        }
        if cardiac_transfer_id is None:
            identifier = canonical_fingerprint(payload)
        else:
            identifier = str(cardiac_transfer_id)
            if not identifier:
                raise ValueError("cardiac_transfer_id must be non-empty.")
        self.transfer = transfer
        self.configuration = configuration
        self.prepared_epoch = prepared_epoch
        self.source_covered = source_mask
        self.target_covered = target_mask
        self.constant_tolerance = constant_tolerance_
        self.adjoint_tolerance = adjoint_tolerance_
        self.cardiac_transfer_id = identifier

    @staticmethod
    def _probe(space: Any, /) -> PyTree[Array]:
        zero_coordinates = space.flatten(space.zeros())
        size = zero_coordinates.shape[0]
        coordinates = jnp.arange(1, size + 1, dtype=zero_coordinates.dtype)
        scale = jnp.sqrt(jnp.sum(jnp.real(coordinates * jnp.conj(coordinates))))
        return space.unflatten(coordinates / scale)

    def evidence(
        self,
        source_probe: PyTree[Any],
        current_epoch: CardiacTransferEpoch,
        /,
        *,
        configuration_id: str,
    ) -> CardiacTransferEvidence:
        if not isinstance(current_epoch, CardiacTransferEpoch):
            raise TypeError("current_epoch must be a CardiacTransferEpoch.")
        source = self.transfer.source.vector_space.validate(source_probe)
        image = self.transfer.operator(source)
        image = self.transfer.target.vector_space.validate(image)
        source_space = self.transfer.source.vector_space
        target_space = self.transfer.target.vector_space
        source_coordinates = source_space.flatten(source)
        image_coordinates = target_space.flatten(image)

        one_source = source_space.unflatten(jnp.ones_like(source_coordinates))
        one_target_coordinates = jnp.ones_like(image_coordinates)
        constant_image = target_space.flatten(self.transfer.operator(one_source))
        constant_error = jnp.max(jnp.abs(constant_image - one_target_coordinates))
        constant_claimed = self.transfer.properties.constant_preserving
        constant_preserved = jnp.asarray(constant_claimed) & (
            constant_error <= self.constant_tolerance
        )

        adjoint_claimed = self.transfer.properties.adjoint_paired
        if adjoint_claimed:
            adjoint = self.transfer.adjoint_operator
            if adjoint is None:
                raise ValueError(
                    "An adjoint-paired transfer must contain an adjoint operator."
                )
            target_probe = self._probe(target_space)
            left = target_space.inner(image, target_probe)
            right = source_space.inner(source, adjoint(target_probe))
            adjoint_scale = jnp.maximum(
                jnp.asarray(1.0, dtype=jnp.abs(left).dtype),
                jnp.maximum(jnp.abs(left), jnp.abs(right)),
            )
            adjoint_error = jnp.abs(left - right) / adjoint_scale
            adjoint_consistent = adjoint_error <= self.adjoint_tolerance
        else:
            adjoint_error = jnp.asarray(jnp.inf, dtype=constant_error.dtype)
            adjoint_consistent = jnp.asarray(False)

        source_fraction = jnp.mean(self.source_covered.astype(constant_error.dtype))
        target_fraction = jnp.mean(self.target_covered.astype(constant_error.dtype))
        coverage_complete = jnp.all(self.source_covered) & jnp.all(self.target_covered)
        configuration_matches = jnp.asarray(
            str(configuration_id) == self.configuration.configuration_id
        )
        epoch_matches = self.prepared_epoch.matches(current_epoch)
        finite = (
            jnp.all(jnp.isfinite(source_coordinates))
            & jnp.all(jnp.isfinite(image_coordinates))
            & jnp.isfinite(constant_error)
            & ((~jnp.asarray(adjoint_claimed)) | jnp.isfinite(adjoint_error))
        )
        accepted = (
            finite
            & coverage_complete
            & configuration_matches
            & epoch_matches
            & ((~jnp.asarray(constant_claimed)) | constant_preserved)
            & ((~jnp.asarray(adjoint_claimed)) | adjoint_consistent)
        )
        return CardiacTransferEvidence(
            source_covered=self.source_covered,
            target_covered=self.target_covered,
            source_coverage_fraction=source_fraction,
            target_coverage_fraction=target_fraction,
            constant_error=constant_error,
            adjoint_error=adjoint_error,
            configuration_matches=configuration_matches,
            epoch_matches=epoch_matches,
            coverage_complete=coverage_complete,
            constant_preserved=constant_preserved,
            adjoint_consistent=adjoint_consistent,
            finite=finite,
            accepted=accepted,
            transfer_id=self.cardiac_transfer_id,
            configuration_id=self.configuration.configuration_id,
        )

    def apply(
        self,
        source_value: PyTree[Any],
        current_epoch: CardiacTransferEpoch,
        /,
        *,
        configuration_id: str,
    ) -> CardiacTransferResult:
        value = self.transfer.operator(source_value)
        evidence = self.evidence(
            source_value,
            current_epoch,
            configuration_id=configuration_id,
        )
        return CardiacTransferResult(value=value, evidence=evidence)

    def apply_adjoint(
        self,
        target_value: PyTree[Any],
        current_epoch: CardiacTransferEpoch,
        /,
        *,
        configuration_id: str,
    ) -> CardiacTransferResult:
        if not self.transfer.properties.adjoint_paired:
            raise ValueError("This field transfer does not declare an adjoint pair.")
        adjoint = self.transfer.adjoint_operator
        if adjoint is None:
            raise ValueError(
                "An adjoint-paired transfer must contain an adjoint operator."
            )
        value = adjoint(target_value)
        source_probe = self._probe(self.transfer.source.vector_space)
        evidence = self.evidence(
            source_probe,
            current_epoch,
            configuration_id=configuration_id,
        )
        return CardiacTransferResult(value=value, evidence=evidence)


__all__ = [
    "CardiacFieldTransfer",
    "CardiacTransferConfiguration",
    "CardiacTransferEpoch",
    "CardiacTransferEvidence",
    "CardiacTransferResult",
]
