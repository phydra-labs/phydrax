#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._numerics._checkpointed_scan import CheckpointedScanMode
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....observation import (
    CholeskyCovarianceAction,
    CirculantCovarianceAction,
    CovarianceAction,
    DiagonalCovarianceAction,
    KroneckerCholeskyCovarianceAction,
    LinearNuisancePlan,
)
from ._acquisition import SeismicAcquisition
from ._constant_density import ConstantDensityAcousticPlan


class AcousticShot(StrictModule, NonTrainableState):
    acquisition: SeismicAcquisition
    source_rates: Array
    observed_pressure_Pa: Array
    covariance: CovarianceAction
    trace_weights: Array
    shot_id: str = eqx.field(static=True)

    def __init__(
        self,
        acquisition: SeismicAcquisition,
        source_rates: ArrayLike,
        observed_pressure_Pa: ArrayLike,
        covariance: CovarianceAction,
        /,
        *,
        trace_weights: ArrayLike = 1.0,
    ):
        if not isinstance(acquisition, SeismicAcquisition):
            raise TypeError("Acoustic shot requires SeismicAcquisition.")
        source = jnp.asarray(source_rates)
        observed = jnp.asarray(observed_pressure_Pa)
        expected = (acquisition.receivers.count, source.shape[0] + 1)
        if source.ndim != 2 or source.shape[1] != acquisition.sources.count:
            raise ValueError("Acoustic shot source history has wrong shape.")
        if observed.shape != expected:
            raise ValueError(
                "Acoustic shot observed traces must include the initial sample."
            )
        weights = jnp.broadcast_to(jnp.asarray(trace_weights), observed.shape)
        if covariance.layout.size != observed.size:
            raise ValueError(
                "Acoustic shot covariance layout does not match trace samples."
            )
        source = eqx.error_if(
            source,
            jnp.any(~jnp.isfinite(source))
            | jnp.any(~jnp.isfinite(observed))
            | jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights < 0),
            "Acoustic shot source, observations, and weights must be finite.",
        )
        self.acquisition, self.source_rates = acquisition, source
        self.observed_pressure_Pa, self.covariance, self.trace_weights = (
            observed,
            covariance,
            weights,
        )
        self.shot_id = canonical_fingerprint(
            {
                "kind": "acoustic-shot",
                "acquisition": acquisition.acquisition_id,
                "source_rates": source,
                "observed_pressure_Pa": observed,
                "covariance": covariance.action_id,
                "trace_weights": weights,
            }
        )


class WaveformInversionResult(StrictModule):
    objective: Array
    gradient: Array
    residuals: tuple[Array, ...]
    finite: Array


class RTMResult(StrictModule):
    image: Array
    objective: Array
    gradient_slowness_squared: Array
    finite: Array


class AcousticWaveformInversionPlan(StrictModule, NonTrainableState):
    forward: ConstantDensityAcousticPlan
    shots: tuple[AcousticShot, ...]
    replay: CheckpointedScanMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: ConstantDensityAcousticPlan,
        shots: Sequence[AcousticShot],
        /,
        *,
        replay: CheckpointedScanMode = "block",
        block_size: int | None = None,
    ):
        if not isinstance(forward, ConstantDensityAcousticPlan):
            raise TypeError("Waveform inversion requires ConstantDensityAcousticPlan.")
        shots_ = tuple(shots)
        if not shots_ or any(not isinstance(shot, AcousticShot) for shot in shots_):
            raise TypeError("Waveform inversion requires nonempty AcousticShot values.")
        if any(
            shot.source_rates.shape[0] != forward.step_count
            or shot.acquisition.sources.grid_id != forward.grid.grid_id
            for shot in shots_
        ):
            raise ValueError(
                "Waveform inversion shots must share the prepared forward grid/clock."
            )
        if replay not in ("full", "step", "block", "scheduled"):
            raise ValueError("Unsupported waveform replay mode.")
        if replay == "scheduled":
            raise ValueError(
                "Scheduled waveform inversion needs an explicit per-call replay schedule."
            )
        block = block_size
        if replay == "block" and (block is None or int(block) <= 0):
            raise ValueError("Block replay requires a positive block_size.")
        self.forward, self.shots = forward, shots_
        self.replay, self.block_size = replay, None if block is None else int(block)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "acoustic-waveform-inversion",
                "forward": forward.plan_id,
                "shots": [shot.shot_id for shot in shots_],
                "replay": replay,
                "block_size": self.block_size,
            }
        )

    def predict(self, wavespeed_m_s: ArrayLike, /) -> tuple[Array, ...]:
        speed = jnp.asarray(wavespeed_m_s)
        return tuple(
            self.forward.simulate(
                speed,
                shot.acquisition,
                shot.source_rates,
                replay=self.replay,
                block_size=self.block_size,
            ).traces.values
            for shot in self.shots
        )

    def residuals(self, wavespeed_m_s: ArrayLike, /) -> tuple[Array, ...]:
        predictions = self.predict(wavespeed_m_s)
        return tuple(
            shot.trace_weights * (prediction - shot.observed_pressure_Pa)
            for shot, prediction in zip(self.shots, predictions, strict=True)
        )

    def objective(self, wavespeed_m_s: ArrayLike, /) -> Array:
        residuals = self.residuals(wavespeed_m_s)
        terms = tuple(
            shot.covariance.quadratic(residual.reshape(-1))
            for shot, residual in zip(self.shots, residuals, strict=True)
        )
        return 0.5 * jnp.sum(jnp.stack(terms))

    def evaluate(self, wavespeed_m_s: ArrayLike, /) -> WaveformInversionResult:
        speed = jnp.asarray(wavespeed_m_s)
        objective, gradient = jax.value_and_grad(self.objective)(speed)
        residuals = self.residuals(speed)
        finite = (
            jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in residuals]))
        )
        return WaveformInversionResult(objective, gradient, residuals, finite)

    def gauss_newton_action(
        self, wavespeed_m_s: ArrayLike, direction: ArrayLike, /
    ) -> Array:
        speed, tangent = jnp.asarray(wavespeed_m_s), jnp.asarray(direction)

        def whitened(value):
            residuals = self.residuals(value)
            outputs = []
            whitening_types = (
                CholeskyCovarianceAction,
                CirculantCovarianceAction,
                DiagonalCovarianceAction,
                KroneckerCholeskyCovarianceAction,
            )
            for shot, residual in zip(self.shots, residuals, strict=True):
                if not isinstance(shot.covariance, whitening_types):
                    raise TypeError(
                        "Waveform Gauss-Newton action requires an explicit whitening covariance."
                    )
                outputs.append(shot.covariance.whiten(residual.reshape(-1)))
            return jnp.concatenate(outputs)

        linearization = la.prepare_linearization(whitened, speed)
        projected = linearization.jvp(tangent)
        return linearization.vjp(projected)

    def rtm(self, background_wavespeed_m_s: ArrayLike, /) -> RTMResult:
        speed = jnp.asarray(background_wavespeed_m_s)
        slowness_squared = 1.0 / speed**2

        def objective(slowness):
            return self.objective(1.0 / jnp.sqrt(slowness))

        value, gradient = jax.value_and_grad(objective)(slowness_squared)
        image = -gradient
        finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(image))
        return RTMResult(image, value, gradient, finite)


class AcousticSourceProjectionPlan(StrictModule, NonTrainableState):
    """Variable-project linear source-basis coefficients through the wave solver."""

    inversion: AcousticWaveformInversionPlan
    shot_index: int = eqx.field(static=True)
    basis_source_rates: Array
    nuisance: LinearNuisancePlan

    def __init__(
        self,
        inversion: AcousticWaveformInversionPlan,
        shot_index: int,
        basis_source_rates: ArrayLike,
        reference_wavespeed_m_s: ArrayLike,
        /,
    ):
        if not isinstance(inversion, AcousticWaveformInversionPlan):
            raise TypeError("Source projection requires waveform inversion plan.")
        index = int(shot_index)
        if not 0 <= index < len(inversion.shots):
            raise ValueError("Source projection shot index is invalid.")
        shot = inversion.shots[index]
        basis = jnp.asarray(basis_source_rates)
        if basis.ndim != 3 or basis.shape[1:] != shot.source_rates.shape:
            raise ValueError("Source basis must have shape (basis, steps, sources).")
        traces = []
        for source in basis:
            traces.append(
                inversion.forward.simulate(
                    reference_wavespeed_m_s,
                    shot.acquisition,
                    source,
                    replay=inversion.replay,
                    block_size=inversion.block_size,
                ).traces.values.reshape(-1)
            )
        design = jnp.stack(traces, axis=1)
        layout = shot.covariance.layout
        names = tuple(f"source-basis-{index}" for index in range(basis.shape[0]))
        self.inversion, self.shot_index, self.basis_source_rates = inversion, index, basis
        self.nuisance = LinearNuisancePlan(design, shot.covariance, layout, names)

    def evaluate(self) -> object:
        shot = self.inversion.shots[self.shot_index]
        return self.nuisance.evaluate(
            shot.observed_pressure_Pa.reshape(-1),
            jnp.zeros(shot.observed_pressure_Pa.size),
        )


__all__ = [
    "AcousticShot",
    "AcousticSourceProjectionPlan",
    "AcousticWaveformInversionPlan",
    "RTMResult",
    "WaveformInversionResult",
]
