#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerical-Cherenkov guard for explicit electromagnetic PIC.

A relativistic species drifting through a Yee/cochain grid can resonate with
grid modes slowed by discrete dispersion and radiate numerical Cherenkov
radiation that the continuum medium forbids. The guard binds one species of a
PIC run to a B3 `CherenkovRegimePlan` whose dispersion audit is the run's own
prepared Maxwell update: construction refuses any numerical-only emission and
any regime that fails to certify its resonance roots, and every step is
rejected (`PICRejectionReason.NUMERICAL_CHERENKOV`) unless it uses the audited
step size and the species stays at or below the audited drift speed. Numerical
Cherenkov *instability* of drifting plasmas (aliasing, spectral PIC) is owned
by the spectral-PIC NCI analysis and is not certified here.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._maxwell_dispersion import CherenkovRegimeEvidence, CherenkovRegimePlan
from ._pic_field_solver import AbstractPreparedPICFieldSolver


# Relative slack of the runtime step-size and drift-speed admission.
_ADMISSION_TOLERANCE = 1.0e-12


class PICCherenkovGuard(StrictModule, NonTrainableState):
    """Numerical-Cherenkov admission of one drifting PIC species."""

    regime: CherenkovRegimePlan
    species: int = eqx.field(static=True)
    guard_id: str = eqx.field(static=True)

    def __init__(self, regime: CherenkovRegimePlan, /, *, species: int) -> None:
        if not isinstance(regime, CherenkovRegimePlan):
            raise TypeError("regime must be a CherenkovRegimePlan.")
        index = int(species)
        if index < 0:
            raise ValueError("species must be a nonnegative species index.")
        self.regime = regime
        self.species = index
        self.guard_id = canonical_fingerprint(
            {
                "kind": "pic-cherenkov-guard",
                "regime": regime.plan_id,
                "species": index,
            }
        )

    @property
    def step_size(self) -> float:
        """Audited field step; the only step size the guard admits."""
        return float(np.asarray(self.regime.audit.step_size))

    @property
    def drift_speed(self) -> float:
        """Audited drift speed bounding the guarded species."""
        return float(np.linalg.norm(np.asarray(self.regime.velocity)))

    def certify(
        self,
        solver: AbstractPreparedPICFieldSolver,
        species_count: int,
        speed_of_light: float,
        /,
    ) -> CherenkovRegimeEvidence:
        """Refuse a foreign audit or numerical-only emission; return the evidence."""
        if self.species >= int(species_count):
            raise ValueError("Cherenkov guard references a species outside the run.")
        if not isinstance(solver, CochainMaxwellPICFieldSolver):
            raise TypeError(
                "The numerical-Cherenkov guard requires the full 3-D cochain PIC "
                "field solver, the only solver with a compatible dispersion audit."
            )
        if self.regime.audit.prepared.prepared_id != solver.maxwell.prepared_id:
            raise ValueError(
                "The Cherenkov regime audits a different prepared Maxwell update."
            )
        if not np.isclose(
            self.regime.vacuum_speed_of_light,
            float(speed_of_light),
            rtol=_ADMISSION_TOLERANCE,
            atol=0.0,
        ):
            raise ValueError(
                "The Cherenkov regime and the PIC pusher use different speeds of light."
            )
        if not self.step_size <= float(solver.stable_step):
            raise ValueError("The audited Cherenkov step exceeds the stable PIC step.")
        evidence = self.regime.evaluate()
        if not bool(jnp.all(evidence.root_converged)):
            raise ValueError(
                "The Cherenkov regime did not converge every resonance root; the "
                "PIC run cannot be certified free of numerical Cherenkov emission."
            )
        numerical_only = np.asarray(evidence.numerical_only)
        if np.any(numerical_only):
            frequencies = np.asarray(evidence.angular_frequencies)[
                np.any(numerical_only, axis=1)
            ]
            raise ValueError(
                "Numerical Cherenkov emission without a physical counterpart at "
                f"angular frequencies {frequencies.tolist()}: the grid dispersion "
                "would radiate where the medium does not."
            )
        return evidence

    def admissible(self, velocity: Array, active: Array, step_size: Array, /) -> Array:
        """Step admission: audited step size and drift speed within the audit."""
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        within = jnp.all(
            jnp.where(
                active, speed <= self.drift_speed * (1.0 + _ADMISSION_TOLERANCE), True
            )
        )
        audited = jnp.abs(step_size - self.step_size) <= (
            _ADMISSION_TOLERANCE * self.step_size
        )
        return within & audited


__all__ = ["PICCherenkovGuard"]
