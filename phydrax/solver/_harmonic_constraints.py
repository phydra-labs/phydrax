#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import KernelCertificate
from ..typing import parse


if TYPE_CHECKING:
    from ..exterior._cohomology import HarmonicClassFrame


HarmonicConstraintPolicy: TypeAlias = Literal["prescribed", "free", "deflated"]


@final
class HarmonicConstraint(StrictModule, NonTrainableState):
    """Solver-owned period choice backed by an exact/numerical kernel certificate."""

    frame: HarmonicClassFrame
    target_periods: Array | None
    policy: HarmonicConstraintPolicy = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: HarmonicClassFrame,
        /,
        *,
        policy: HarmonicConstraintPolicy = "prescribed",
        target_periods: ArrayLike | None = None,
    ) -> None:
        from ..exterior._cohomology import HarmonicClassFrame

        if not isinstance(frame, HarmonicClassFrame):
            raise TypeError("frame must be a HarmonicClassFrame.")
        resolved = parse(policy, HarmonicConstraintPolicy, "policy")
        if (resolved == "prescribed") != (target_periods is not None):
            raise ValueError(
                "target_periods is required exactly for prescribed constraints."
            )
        periods = None if target_periods is None else jnp.asarray(target_periods)
        if periods is not None:
            if periods.shape != (frame.exact_basis.generator_count,):
                raise ValueError(
                    "Harmonic target periods do not match the exact class count."
                )
            periods = eqx.error_if(
                periods,
                jnp.any(~jnp.isfinite(periods)),
                "Harmonic target periods must be finite.",
            )
        self.frame, self.target_periods, self.policy = frame, periods, resolved
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "harmonic-constraint",
                "frame": frame.frame_id,
                "policy": resolved,
                "period_count": frame.exact_basis.generator_count,
            }
        )

    @property
    def certificate(self) -> KernelCertificate:
        return self.frame.kernel_certificate

    def apply(self, cochain: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        match self.policy:
            case "free":
                return values
            case "deflated":
                return self.frame.with_periods(
                    values,
                    jnp.zeros(
                        (self.frame.exact_basis.generator_count,), dtype=values.dtype
                    ),
                )
            case "prescribed":
                if self.target_periods is None:
                    raise ValueError(
                        "Prescribed harmonic constraint has no target periods."
                    )
                return self.frame.with_periods(values, self.target_periods)
            case _ as unreachable:
                assert_never(unreachable)

    def residual(self, cochain: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        return jnp.linalg.norm(
            self.frame.periods(values) - self.frame.periods(self.apply(values))
        )


__all__ = ["HarmonicConstraint", "HarmonicConstraintPolicy"]
