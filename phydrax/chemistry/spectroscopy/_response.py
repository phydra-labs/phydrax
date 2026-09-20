#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed raw spectroscopy responses, before any instrument transformation."""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition


class SpectralResponseConvention(StrEnum):
    """Frequency, transfer, and density convention of a raw response."""

    RETARDED_EXP_MINUS_IWT_POSITIVE_LOSS = "retarded-exp-minus-iwt-positive-loss"


class SpectralResponseRepresentation(StrEnum):
    """Meaning of samples in a :class:`SpectralResponseProduct`."""

    LINES = "integrated-lines"
    DENSITY = "density-per-coordinate"


class SpectralResponseEvidence(StrictModule, NonTrainableState):
    """Scientific checks retained with an unmodified forward response."""

    passivity_residual: Array
    sum_rule_residual: Array
    detailed_balance_residual: Array
    selection_rule_residual: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        passivity_residual: ArrayLike,
        sum_rule_residual: ArrayLike,
        detailed_balance_residual: ArrayLike,
        selection_rule_residual: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        residuals = jnp.asarray(
            [
                passivity_residual,
                sum_rule_residual,
                detailed_balance_residual,
                selection_rule_residual,
            ],
            dtype=jnp.float64,
        ).reshape((4,))
        if bool(jnp.any(~jnp.isfinite(residuals))) or bool(jnp.any(residuals < 0.0)):
            raise ValueError(
                "Spectral response residuals must be finite and non-negative."
            )
        self.passivity_residual = residuals[0]
        self.sum_rule_residual = residuals[1]
        self.detailed_balance_residual = residuals[2]
        self.selection_rule_residual = residuals[3]
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spectral-response-evidence",
                "residuals": array_tree_fingerprint(np.asarray(residuals)),
                "successful": bool(self.successful),
            }
        )


class SpectralResponseProduct(StrictModule, NonTrainableState):
    """Immutable raw line strengths or gridded densities from forward physics.

    The final axis is spectral coordinate; the first axis is channel.  A line
    product stores integrated line strengths, whereas a density product stores
    density per coordinate.  Neither is an instrument-convolved observation.
    """

    coordinates: Array
    values: Array
    active: Array
    coordinate_unit: UnitDefinition = eqx.field(static=True)
    response_unit: UnitDefinition = eqx.field(static=True)
    channels: tuple[str, ...] = eqx.field(static=True)
    representation: SpectralResponseRepresentation = eqx.field(static=True)
    convention: SpectralResponseConvention = eqx.field(static=True)
    source_profile_id: str = eqx.field(static=True)
    source_product_id: str = eqx.field(static=True)
    evidence: SpectralResponseEvidence
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        values: ArrayLike,
        active: ArrayLike,
        coordinate_unit: UnitDefinition,
        response_unit: UnitDefinition,
        channels: tuple[str, ...],
        representation: SpectralResponseRepresentation,
        source_profile_id: str,
        source_product_id: str,
        evidence: SpectralResponseEvidence,
        /,
        *,
        convention: SpectralResponseConvention = SpectralResponseConvention.RETARDED_EXP_MINUS_IWT_POSITIVE_LOSS,
    ):
        coordinate = jnp.asarray(coordinates)
        response = jnp.asarray(values)
        mask = jnp.asarray(active, dtype=jnp.bool_)
        labels = tuple(str(label).strip() for label in channels)
        profile_id = str(source_profile_id).strip()
        parent_id = str(source_product_id).strip()
        if coordinate.ndim != 1 or coordinate.size == 0:
            raise ValueError("Spectral coordinates must be a non-empty vector.")
        if response.ndim == 1:
            response = response[None, :]
        if response.shape != (len(labels), coordinate.size):
            raise ValueError("Spectral values must have shape (channels, coordinates).")
        if mask.shape != coordinate.shape:
            raise ValueError("The active mask must match the spectral coordinate axis.")
        if (
            not isinstance(coordinate_unit, UnitDefinition)
            or not isinstance(response_unit, UnitDefinition)
            or not isinstance(representation, SpectralResponseRepresentation)
            or not isinstance(convention, SpectralResponseConvention)
            or not isinstance(evidence, SpectralResponseEvidence)
            or not labels
            or len(set(labels)) != len(labels)
            or any(not label for label in labels)
            or not profile_id
            or not parent_id
        ):
            raise ValueError("Spectral response metadata are invalid.")
        if bool(jnp.any(~jnp.isfinite(coordinate))) or bool(
            jnp.any(~jnp.isfinite(response))
        ):
            raise ValueError("Spectral coordinates and values must be finite.")
        if bool(jnp.any(response < 0.0)):
            raise ValueError("Released spectral response values must be non-negative.")
        active_coordinates = np.asarray(coordinate)[np.asarray(mask)]
        if active_coordinates.size and np.any(np.diff(active_coordinates) <= 0.0):
            raise ValueError("Active spectral coordinates must be strictly increasing.")
        self.coordinates = coordinate
        self.values = response
        self.active = mask
        self.coordinate_unit = coordinate_unit
        self.response_unit = response_unit
        self.channels = labels
        self.representation = representation
        self.convention = convention
        self.source_profile_id = profile_id
        self.source_product_id = parent_id
        self.evidence = evidence
        self.product_id = canonical_fingerprint(
            {
                "kind": "raw-spectral-response",
                "representation": representation.value,
                "convention": convention.value,
                "coordinate_unit": coordinate_unit.unit_id,
                "response_unit": response_unit.unit_id,
                "channels": list(labels),
                "source_profile": profile_id,
                "source_product": parent_id,
                "evidence": evidence.evidence_id,
                "arrays": array_tree_fingerprint(
                    {
                        "coordinates": np.asarray(coordinate),
                        "values": np.asarray(response),
                        "active": np.asarray(mask),
                    }
                ),
            }
        )


__all__ = [
    "SpectralResponseConvention",
    "SpectralResponseEvidence",
    "SpectralResponseProduct",
    "SpectralResponseRepresentation",
]
