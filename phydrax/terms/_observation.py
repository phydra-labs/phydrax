#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from jaxtyping import ArrayLike

from phydrax.domain import DomainFunction

from ..conditions._base import Observation
from ..integration import IntegrationSource
from ._residual import ResidualPenalty


def ObservationPenalty(
    condition: Observation,
    source: IntegrationSource,
    /,
    *,
    scale: ArrayLike = 1.0,
    density: DomainFunction | None = None,
    label: str | None = None,
    data_accuracy_eps: float = 1e-12,
) -> ResidualPenalty:
    """Construct a finite- or continuously-realized observation penalty."""
    if not isinstance(condition, Observation):
        raise TypeError("ObservationPenalty requires an Observation.")
    return ResidualPenalty(
        condition,
        source,
        scale=scale,
        density=density,
        label=label,
        data_accuracy_eps=data_accuracy_eps,
    )


__all__ = ["ObservationPenalty"]
