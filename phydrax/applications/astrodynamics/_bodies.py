#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext


class CelestialBodyCatalog(StrictModule, NonTrainableState):
    """Fixed-capacity body constants with explicit context identity."""

    gravitational_parameters: Array
    reference_radii: Array
    active_mask: Array
    context: AstrodynamicsContext
    body_ids: tuple[str, ...] = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_ids: tuple[str, ...],
        gravitational_parameters: ArrayLike,
        reference_radii: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ):
        identifiers = tuple(str(value).strip() for value in body_ids)
        if not identifiers or any(not value for value in identifiers):
            raise ValueError("body_ids must be non-empty identifiers.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("body_ids must be unique.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        coupling = np.asarray(gravitational_parameters, dtype=float)
        radii = np.asarray(reference_radii, dtype=float)
        count = len(identifiers)
        if coupling.shape != (count,) or radii.shape != (count,):
            raise ValueError("Body constants must match body capacity.")
        active = (
            np.ones((count,), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError("active_mask must match body capacity.")
        if (
            np.any(~np.isfinite(coupling))
            or np.any(~np.isfinite(radii))
            or np.any(active & (coupling <= 0.0))
            or np.any(active & (radii < 0.0))
        ):
            raise ValueError("Active body constants must be finite and physical.")
        self.gravitational_parameters = jnp.asarray(coupling)
        self.reference_radii = jnp.asarray(radii)
        self.active_mask = jnp.asarray(active)
        self.context = context
        self.body_ids = identifiers
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "celestial-body-catalog",
                "body_ids": list(identifiers),
                "gravitational_parameters": coupling.tolist(),
                "reference_radii": radii.tolist(),
                "active_mask": active.tolist(),
                "context": context.context_id,
            }
        )

    @property
    def capacity(self) -> int:
        return len(self.body_ids)

    def index(self, body_id: str, /) -> int:
        identifier = str(body_id)
        if identifier not in self.body_ids:
            raise KeyError(f"Unknown celestial body {identifier!r}.")
        return self.body_ids.index(identifier)


__all__ = ["CelestialBodyCatalog"]
