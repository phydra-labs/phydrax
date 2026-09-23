#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Model-specific adaptive RGE, threshold, EWSB, mass, vacuum, and scan workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


NativeSpectrumModel: TypeAlias = Literal["sm-one-loop", "mssm-third-family-one-loop"]

_SM_LABELS = ("g1", "g2", "g3", "yt", "yb", "ytau", "lambda", "m2")
_MSSM_LABELS = (
    "g1",
    "g2",
    "g3",
    "yt",
    "yb",
    "ytau",
    "M1",
    "M2",
    "M3",
    "At",
    "Ab",
    "Atau",
    "mHu2",
    "mHd2",
    "mQ32",
    "mU32",
    "mD32",
    "mL32",
    "mE32",
    "mu",
    "Bmu",
    "tanbeta",
)


class NativeSpectrumStatus(IntEnum):
    SUCCESS = 0
    INTEGRATION_FAILURE = 1
    NO_EWSB = 2
    TACHYON = 3
    NONPERTURBATIVE = 4
    VACUUM_WARNING = 5
    NO_ROOT = 6
    INVALID_INPUT = 7


class NativeSpectrumModelPlan(StrictModule):
    """One convention-pinned SM or third-family MSSM perturbative model."""

    model: NativeSpectrumModel = eqx.field(static=True)
    parameter_labels: tuple[str, ...] = eqx.field(static=True)
    scheme: str = eqx.field(static=True)
    loop_order: int = eqx.field(static=True)
    perturbativity_limit: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    minimum_log_step: float = eqx.field(static=True)
    maximum_log_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: NativeSpectrumModel,
        /,
        *,
        scheme: str,
        loop_order: int = 1,
        perturbativity_limit: float = 4.0 * np.pi,
        relative_tolerance: float = 1e-7,
        absolute_tolerance: float = 1e-10,
        minimum_log_step: float = 1e-6,
        maximum_log_step: float = 0.1,
        maximum_steps: int = 100_000,
    ):
        if model not in ("sm-one-loop", "mssm-third-family-one-loop"):
            raise ValueError("Unknown native particle-spectrum model.")
        scheme_ = str(scheme).strip()
        order = int(loop_order)
        perturbative = float(perturbativity_limit)
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        minimum = float(minimum_log_step)
        maximum = float(maximum_log_step)
        steps = int(maximum_steps)
        if not scheme_ or order != 1:
            raise ValueError(
                "Current native model support is one-loop with explicit scheme."
            )
        if (
            perturbative <= 0.0
            or relative <= 0.0
            or absolute <= 0.0
            or minimum <= 0.0
            or maximum < minimum
            or steps < 1
        ):
            raise ValueError("Native spectrum integration controls are invalid.")
        labels = _SM_LABELS if model == "sm-one-loop" else _MSSM_LABELS
        content = {
            "kind": "native-spectrum-model-plan",
            "model": model,
            "parameter_labels": labels,
            "scheme": scheme_,
            "loop_order": order,
            "perturbativity_limit": perturbative,
            "relative_tolerance": relative,
            "absolute_tolerance": absolute,
            "minimum_log_step": minimum,
            "maximum_log_step": maximum,
            "maximum_steps": steps,
        }
        self.model = model
        self.parameter_labels = labels
        self.scheme = scheme_
        self.loop_order = order
        self.perturbativity_limit = perturbative
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.minimum_log_step = minimum
        self.maximum_log_step = maximum
        self.maximum_steps = steps
        self.plan_id = canonical_fingerprint(content)

    def index(self, label: str, /) -> int:
        name = str(label)
        if name not in self.parameter_labels:
            raise ValueError(f"Unknown {self.model} parameter {label!r}.")
        return self.parameter_labels.index(name)


@dataclass(frozen=True, slots=True)
class SpectrumThreshold:
    """One affine matching map applied at a fixed physical scale."""

    scale: float
    matrix: np.ndarray
    offset: np.ndarray
    source_id: str
    threshold_id: str

    def __init__(
        self,
        scale: float,
        matrix: ArrayLike,
        offset: ArrayLike,
        source_id: str,
        /,
    ):
        scale_ = float(scale)
        matrix_ = np.array(matrix, dtype=np.float64, copy=True)
        offset_ = np.array(offset, dtype=np.float64, copy=True)
        source = str(source_id).strip()
        if (
            not math.isfinite(scale_)
            or scale_ <= 0.0
            or matrix_.ndim != 2
            or matrix_.shape[0] != matrix_.shape[1]
            or np.any(~np.isfinite(matrix_))
        ):
            raise ValueError("Threshold scale and matching matrix are invalid.")
        if (
            offset_.shape != (matrix_.shape[0],)
            or np.any(~np.isfinite(offset_))
            or not source
        ):
            raise ValueError("Threshold offset or source identity is invalid.")
        matrix_.setflags(write=False)
        offset_.setflags(write=False)
        content = {
            "kind": "spectrum-threshold",
            "scale": scale_,
            "matrix": array_tree_fingerprint(matrix_),
            "offset": array_tree_fingerprint(offset_),
            "source_id": source,
        }
        object.__setattr__(self, "scale", scale_)
        object.__setattr__(self, "matrix", matrix_)
        object.__setattr__(self, "offset", offset_)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "threshold_id", canonical_fingerprint(content))


class NativeRGEHistory(StrictModule):
    log_scales: Array
    parameters: Array
    accepted_step_sizes: Array
    threshold_indices: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)


def _sm_beta(values: np.ndarray, /) -> np.ndarray:
    g1, g2, g3, yt, yb, ytau, quartic, mass2 = values
    loop = 1.0 / (16.0 * np.pi**2)
    beta = np.empty_like(values)
    beta[0] = loop * (41.0 / 6.0) * g1**3
    beta[1] = loop * (-19.0 / 6.0) * g2**3
    beta[2] = loop * (-7.0) * g3**3
    beta[3] = (
        loop
        * yt
        * (
            4.5 * yt**2
            + 1.5 * yb**2
            + ytau**2
            - 17.0 / 12.0 * g1**2
            - 9.0 / 4.0 * g2**2
            - 8.0 * g3**2
        )
    )
    beta[4] = (
        loop
        * yb
        * (
            4.5 * yb**2
            + 1.5 * yt**2
            + ytau**2
            - 5.0 / 12.0 * g1**2
            - 9.0 / 4.0 * g2**2
            - 8.0 * g3**2
        )
    )
    beta[5] = (
        loop
        * ytau
        * (2.5 * ytau**2 + 3.0 * yb**2 - 15.0 / 4.0 * g1**2 - 9.0 / 4.0 * g2**2)
    )
    beta[6] = loop * (
        24.0 * quartic**2
        - 6.0 * yt**4
        - 6.0 * yb**4
        - 2.0 * ytau**4
        + quartic
        * (12.0 * yt**2 + 12.0 * yb**2 + 4.0 * ytau**2 - 3.0 * g1**2 - 9.0 * g2**2)
        + 3.0 / 8.0 * (2.0 * g2**4 + (g1**2 + g2**2) ** 2)
    )
    beta[7] = (
        loop
        * mass2
        * (
            12.0 * quartic
            + 6.0 * yt**2
            + 6.0 * yb**2
            + 2.0 * ytau**2
            - 1.5 * g1**2
            - 4.5 * g2**2
        )
    )
    return beta


def _mssm_beta(values: np.ndarray, /) -> np.ndarray:
    (
        g1,
        g2,
        g3,
        yt,
        yb,
        ytau,
        m1,
        m2,
        m3,
        at,
        ab,
        atau,
        mhu2,
        mhd2,
        mq2,
        mu2,
        md2,
        ml2,
        me2,
        higgsino,
        bmu,
        tanbeta,
    ) = values
    loop = 1.0 / (16.0 * np.pi**2)
    beta = np.zeros_like(values)
    coefficients = (33.0 / 5.0, 1.0, -3.0)
    for index, (gauge, coefficient) in enumerate(
        zip((g1, g2, g3), coefficients, strict=True)
    ):
        beta[index] = loop * coefficient * gauge**3
        beta[6 + index] = loop * 2.0 * coefficient * gauge**2 * values[6 + index]
    beta[3] = (
        loop
        * yt
        * (6.0 * yt**2 + yb**2 - 13.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2)
    )
    beta[4] = (
        loop
        * yb
        * (
            6.0 * yb**2
            + yt**2
            + ytau**2
            - 7.0 / 15.0 * g1**2
            - 3.0 * g2**2
            - 16.0 / 3.0 * g3**2
        )
    )
    beta[5] = (
        loop * ytau * (4.0 * ytau**2 + 3.0 * yb**2 - 9.0 / 5.0 * g1**2 - 3.0 * g2**2)
    )
    beta[9] = loop * (
        at
        * (18.0 * yt**2 + yb**2 - 13.0 / 15.0 * g1**2 - 3.0 * g2**2 - 16.0 / 3.0 * g3**2)
        + 2.0 * yb**2 * ab
        + 26.0 / 15.0 * g1**2 * m1
        + 6.0 * g2**2 * m2
        + 32.0 / 3.0 * g3**2 * m3
    )
    beta[10] = loop * (
        ab
        * (
            18.0 * yb**2
            + yt**2
            + ytau**2
            - 7.0 / 15.0 * g1**2
            - 3.0 * g2**2
            - 16.0 / 3.0 * g3**2
        )
        + 2.0 * yt**2 * at
        + 2.0 * ytau**2 * atau
        + 14.0 / 15.0 * g1**2 * m1
        + 6.0 * g2**2 * m2
        + 32.0 / 3.0 * g3**2 * m3
    )
    beta[11] = loop * (
        atau * (12.0 * ytau**2 + 3.0 * yb**2 - 9.0 / 5.0 * g1**2 - 3.0 * g2**2)
        + 6.0 * yb**2 * ab
        + 18.0 / 5.0 * g1**2 * m1
        + 6.0 * g2**2 * m2
    )
    xt = mhu2 + mq2 + mu2 + at**2
    xb = mhd2 + mq2 + md2 + ab**2
    xtau = mhd2 + ml2 + me2 + atau**2
    beta[12] = loop * (6.0 * yt**2 * xt - 6.0 / 5.0 * g1**2 * m1**2 - 6.0 * g2**2 * m2**2)
    beta[13] = loop * (
        6.0 * yb**2 * xb
        + 2.0 * ytau**2 * xtau
        - 6.0 / 5.0 * g1**2 * m1**2
        - 6.0 * g2**2 * m2**2
    )
    beta[14] = loop * (
        2.0 * yt**2 * xt
        + 2.0 * yb**2 * xb
        - 2.0 / 15.0 * g1**2 * m1**2
        - 6.0 * g2**2 * m2**2
        - 32.0 / 3.0 * g3**2 * m3**2
    )
    beta[15] = loop * (
        4.0 * yt**2 * xt - 32.0 / 15.0 * g1**2 * m1**2 - 32.0 / 3.0 * g3**2 * m3**2
    )
    beta[16] = loop * (
        4.0 * yb**2 * xb - 8.0 / 15.0 * g1**2 * m1**2 - 32.0 / 3.0 * g3**2 * m3**2
    )
    beta[17] = loop * (
        2.0 * ytau**2 * xtau - 6.0 / 5.0 * g1**2 * m1**2 - 6.0 * g2**2 * m2**2
    )
    beta[18] = loop * (4.0 * ytau**2 * xtau - 24.0 / 5.0 * g1**2 * m1**2)
    beta[19] = (
        loop
        * higgsino
        * (3.0 * yt**2 + 3.0 * yb**2 + ytau**2 - 3.0 / 5.0 * g1**2 - 3.0 * g2**2)
    )
    beta[20] = (
        loop
        * bmu
        * (3.0 * yt**2 + 3.0 * yb**2 + ytau**2 - 3.0 / 5.0 * g1**2 - 3.0 * g2**2)
    )
    beta[21] = loop * tanbeta * (3.0 * yt**2 - 3.0 * yb**2 - ytau**2)
    return beta


def native_beta_function(
    plan: NativeSpectrumModelPlan, parameters: ArrayLike, /
) -> Array:
    if not isinstance(plan, NativeSpectrumModelPlan):
        raise TypeError("plan must be NativeSpectrumModelPlan.")
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(plan.parameter_labels),):
        raise ValueError("Native spectrum parameters have the wrong shape.")
    beta = _sm_beta(values) if plan.model == "sm-one-loop" else _mssm_beta(values)
    return jnp.asarray(beta)


def _rk45_step(plan: NativeSpectrumModelPlan, values: np.ndarray, step: float, /):
    beta = _sm_beta if plan.model == "sm-one-loop" else _mssm_beta
    k1 = beta(values)
    k2 = beta(values + step * (1.0 / 5.0) * k1)
    k3 = beta(values + step * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2))
    k4 = beta(values + step * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3))
    k5 = beta(
        values
        + step
        * (
            19372.0 / 6561.0 * k1
            - 25360.0 / 2187.0 * k2
            + 64448.0 / 6561.0 * k3
            - 212.0 / 729.0 * k4
        )
    )
    k6 = beta(
        values
        + step
        * (
            9017.0 / 3168.0 * k1
            - 355.0 / 33.0 * k2
            + 46732.0 / 5247.0 * k3
            + 49.0 / 176.0 * k4
            - 5103.0 / 18656.0 * k5
        )
    )
    fifth = values + step * (
        35.0 / 384.0 * k1
        + 500.0 / 1113.0 * k3
        + 125.0 / 192.0 * k4
        - 2187.0 / 6784.0 * k5
        + 11.0 / 84.0 * k6
    )
    k7 = beta(fifth)
    fourth = values + step * (
        5179.0 / 57600.0 * k1
        + 7571.0 / 16695.0 * k3
        + 393.0 / 640.0 * k4
        - 92097.0 / 339200.0 * k5
        + 187.0 / 2100.0 * k6
        + 1.0 / 40.0 * k7
    )
    return fifth, fifth - fourth


def integrate_native_rge(
    plan: NativeSpectrumModelPlan,
    initial_parameters: ArrayLike,
    initial_scale: float,
    final_scale: float,
    /,
    *,
    thresholds: Sequence[SpectrumThreshold] = (),
) -> NativeRGEHistory:
    """Adaptively integrate log-scale RGEs and hit every matching threshold exactly."""

    if not isinstance(plan, NativeSpectrumModelPlan):
        raise TypeError("plan must be NativeSpectrumModelPlan.")
    values = np.asarray(initial_parameters, dtype=np.float64)
    if values.shape != (len(plan.parameter_labels),) or not np.all(np.isfinite(values)):
        raise ValueError("Initial native spectrum parameters are invalid.")
    initial = float(initial_scale)
    final = float(final_scale)
    if initial <= 0.0 or final <= 0.0 or initial == final:
        raise ValueError("RGE scales must be positive and distinct.")
    thresholds_ = tuple(
        sorted(thresholds, key=lambda value: value.scale, reverse=final < initial)
    )
    if any(not isinstance(value, SpectrumThreshold) for value in thresholds_):
        raise TypeError("thresholds must contain SpectrumThreshold values.")
    if any(value.matrix.shape != (values.size, values.size) for value in thresholds_):
        raise ValueError("Threshold maps do not match the model dimension.")
    if len({value.scale for value in thresholds_}) != len(thresholds_):
        raise ValueError("Spectrum threshold scales must be unique.")
    lower_scale, upper_scale = sorted((initial, final))
    if any(not lower_scale < value.scale < upper_scale for value in thresholds_):
        raise ValueError(
            "Every threshold must lie strictly inside the integration interval."
        )
    direction = 1.0 if final > initial else -1.0
    targets = [math.log(value.scale) for value in thresholds_]
    targets.append(math.log(final))
    threshold_by_log = {math.log(value.scale): value for value in thresholds_}
    time = math.log(initial)
    step = direction * min(plan.maximum_log_step, abs(targets[0] - time))
    times = [time]
    history = [values.copy()]
    step_sizes: list[float] = []
    threshold_indices: list[int] = []
    status = NativeSpectrumStatus.SUCCESS
    iteration = 0
    for target in targets:
        while direction * (target - time) > 0.0:
            iteration += 1
            if iteration > plan.maximum_steps:
                status = NativeSpectrumStatus.INTEGRATION_FAILURE
                break
            step = direction * min(abs(step), abs(target - time), plan.maximum_log_step)
            candidate, error = _rk45_step(plan, values, step)
            scale = plan.absolute_tolerance + plan.relative_tolerance * np.maximum(
                np.abs(values), np.abs(candidate)
            )
            error_norm = float(np.sqrt(np.mean((error / scale) ** 2)))
            if not np.all(np.isfinite(candidate)) or not math.isfinite(error_norm):
                status = NativeSpectrumStatus.INTEGRATION_FAILURE
                break
            if error_norm <= 1.0:
                time += step
                values = candidate
                times.append(time)
                history.append(values.copy())
                step_sizes.append(step)
                if np.max(np.abs(values[:6])) > plan.perturbativity_limit:
                    status = NativeSpectrumStatus.NONPERTURBATIVE
                    break
                factor = (
                    5.0
                    if error_norm == 0.0
                    else min(5.0, max(0.2, 0.9 * error_norm ** (-0.2)))
                )
                step *= factor
            else:
                factor = max(0.1, 0.9 * error_norm ** (-0.25))
                step *= factor
                if abs(step) < plan.minimum_log_step:
                    status = NativeSpectrumStatus.INTEGRATION_FAILURE
                    break
        if status is not NativeSpectrumStatus.SUCCESS:
            break
        threshold = threshold_by_log.get(target)
        if threshold is not None:
            values = threshold.matrix @ values + threshold.offset
            threshold_indices.append(len(history))
            times.append(time)
            history.append(values.copy())
    history_array = np.stack(history)
    history_id = canonical_fingerprint(
        {
            "kind": "native-rge-history",
            "plan": plan.plan_id,
            "initial_scale": initial,
            "final_scale": final,
            "thresholds": [value.threshold_id for value in thresholds_],
            "parameters": array_tree_fingerprint(history_array),
            "status": int(status),
        }
    )
    return NativeRGEHistory(
        log_scales=jnp.asarray(times),
        parameters=jnp.asarray(history_array),
        accepted_step_sizes=jnp.asarray(step_sizes),
        threshold_indices=jnp.asarray(threshold_indices, dtype=jnp.int32),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        plan_id=plan.plan_id,
        history_id=history_id,
    )


class ElectroweakSymmetryBreakingEvidence(StrictModule):
    mu_squared: Array
    b_mu: Array
    residuals: Array
    stable_denominator: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


def solve_mssm_electroweak_breaking(
    parameters: ArrayLike,
    /,
    *,
    z_mass: float = 91.1876,
) -> ElectroweakSymmetryBreakingEvidence:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(_MSSM_LABELS),):
        raise ValueError(
            "MSSM electroweak breaking requires the native MSSM parameter vector."
        )
    mhu2 = values[_MSSM_LABELS.index("mHu2")]
    mhd2 = values[_MSSM_LABELS.index("mHd2")]
    tanbeta = values[_MSSM_LABELS.index("tanbeta")]
    denominator = tanbeta * tanbeta - 1.0
    stable = abs(denominator) > 1e-8 and tanbeta > 0.0
    mu_squared = (
        (mhd2 - mhu2 * tanbeta * tanbeta) / denominator - 0.5 * z_mass * z_mass
        if stable
        else math.nan
    )
    sin_two_beta = 2.0 * tanbeta / (1.0 + tanbeta * tanbeta)
    b_mu = 0.5 * sin_two_beta * (mhu2 + mhd2 + 2.0 * mu_squared)
    residuals = np.asarray(
        (
            mu_squared * denominator
            - (mhd2 - mhu2 * tanbeta**2)
            + 0.5 * z_mass**2 * denominator,
            2.0 * b_mu - sin_two_beta * (mhu2 + mhd2 + 2.0 * mu_squared),
        )
    )
    successful = stable and mu_squared > 0.0 and np.all(np.isfinite(residuals))
    evidence_id = canonical_fingerprint(
        {
            "kind": "mssm-electroweak-breaking-evidence",
            "parameters": array_tree_fingerprint(values),
            "z_mass": float(z_mass),
        }
    )
    return ElectroweakSymmetryBreakingEvidence(
        mu_squared=jnp.asarray(mu_squared),
        b_mu=jnp.asarray(b_mu),
        residuals=jnp.asarray(residuals),
        stable_denominator=jnp.asarray(stable),
        successful=jnp.asarray(successful),
        evidence_id=evidence_id,
    )


class NativePoleMassEvidence(StrictModule):
    labels: tuple[str, ...] = eqx.field(static=True)
    tree_masses: Array
    pole_masses: Array
    loop_corrections: Array
    tachyonic: Array
    finite: Array
    evidence_id: str = eqx.field(static=True)


def compute_native_mssm_pole_masses(
    parameters: ArrayLike,
    /,
    *,
    electroweak_vev: float = 246.21965,
) -> NativePoleMassEvidence:
    """Compute bounded tree electroweakinos/stops and leading one-loop light Higgs."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(_MSSM_LABELS),):
        raise ValueError("Native MSSM masses require the native MSSM parameter vector.")
    index = {label: position for position, label in enumerate(_MSSM_LABELS)}
    g1, g2, yt = values[index["g1"]], values[index["g2"]], values[index["yt"]]
    m1, m2, mu = values[index["M1"]], values[index["M2"]], values[index["mu"]]
    tanbeta = values[index["tanbeta"]]
    beta = math.atan(tanbeta)
    sin_beta, cos_beta = math.sin(beta), math.cos(beta)
    z_mass = 0.5 * electroweak_vev * math.sqrt(g1 * g1 + g2 * g2)
    w_mass = 0.5 * electroweak_vev * g2
    sine_w = g1 / math.sqrt(g1 * g1 + g2 * g2)
    cosine_w = g2 / math.sqrt(g1 * g1 + g2 * g2)
    neutralino = np.asarray(
        (
            (m1, 0.0, -z_mass * sine_w * cos_beta, z_mass * sine_w * sin_beta),
            (0.0, m2, z_mass * cosine_w * cos_beta, -z_mass * cosine_w * sin_beta),
            (-z_mass * sine_w * cos_beta, z_mass * cosine_w * cos_beta, 0.0, -mu),
            (z_mass * sine_w * sin_beta, -z_mass * cosine_w * sin_beta, -mu, 0.0),
        )
    )
    chargino = np.asarray(
        (
            (m2, math.sqrt(2.0) * w_mass * sin_beta),
            (math.sqrt(2.0) * w_mass * cos_beta, mu),
        )
    )
    neutralino_masses = np.sort(np.abs(np.linalg.eigvalsh(neutralino)))
    chargino_masses = np.sort(np.linalg.svd(chargino, compute_uv=False))
    top_mass = yt * electroweak_vev * sin_beta / math.sqrt(2.0)
    mq2, mu2, at = values[index["mQ32"]], values[index["mU32"]], values[index["At"]]
    stop = np.asarray(
        (
            (mq2 + top_mass**2, top_mass * (at - mu / tanbeta)),
            (top_mass * (at - mu / tanbeta), mu2 + top_mass**2),
        )
    )
    stop_squared = np.linalg.eigvalsh(stop)
    stop_masses = np.sqrt(np.maximum(stop_squared, 0.0))
    tree_higgs_squared = z_mass**2 * math.cos(2.0 * beta) ** 2
    scale_squared = max(stop_masses[0] * stop_masses[1], top_mass**2 * (1.0 + 1e-12))
    mixing = at - mu / tanbeta
    loop_higgs_squared = (
        3.0
        * top_mass**4
        / (2.0 * np.pi**2 * electroweak_vev**2)
        * (
            math.log(scale_squared / top_mass**2)
            + mixing**2 / scale_squared * (1.0 - mixing**2 / (12.0 * scale_squared))
        )
    )
    tree = np.concatenate(
        (
            neutralino_masses,
            chargino_masses,
            stop_masses,
            (math.sqrt(max(tree_higgs_squared, 0.0)),),
        )
    )
    pole = tree.copy()
    pole[-1] = math.sqrt(max(tree_higgs_squared + loop_higgs_squared, 0.0))
    corrections = pole - tree
    labels = (
        "neutralino-1",
        "neutralino-2",
        "neutralino-3",
        "neutralino-4",
        "chargino-1",
        "chargino-2",
        "stop-1",
        "stop-2",
        "higgs-light",
    )
    tachyonic = np.concatenate(
        (
            np.zeros(6, dtype=np.bool_),
            stop_squared < 0.0,
            (tree_higgs_squared + loop_higgs_squared < 0.0,),
        )
    )
    finite = bool(np.all(np.isfinite(pole)))
    evidence_id = canonical_fingerprint(
        {
            "kind": "native-mssm-pole-mass-evidence",
            "parameters": array_tree_fingerprint(values),
            "tree": array_tree_fingerprint(tree),
            "pole": array_tree_fingerprint(pole),
        }
    )
    return NativePoleMassEvidence(
        labels=labels,
        tree_masses=jnp.asarray(tree),
        pole_masses=jnp.asarray(pole),
        loop_corrections=jnp.asarray(corrections),
        tachyonic=jnp.asarray(tachyonic),
        finite=jnp.asarray(finite),
        evidence_id=evidence_id,
    )


class VacuumStabilityEvidence(StrictModule):
    higgs_quartic_margin: Array
    stop_ccb_margin: Array
    stau_ccb_margin: Array
    tachyon_count: Array
    stable: Array
    status: Array
    evidence_id: str = eqx.field(static=True)


def assess_native_vacuum_stability(
    plan: NativeSpectrumModelPlan,
    parameters: ArrayLike,
    /,
) -> VacuumStabilityEvidence:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(plan.parameter_labels),):
        raise ValueError("Vacuum parameters do not match the model.")
    if plan.model == "sm-one-loop":
        quartic = values[plan.index("lambda")]
        margins = (quartic, math.inf)
        tachyons = int(values[plan.index("m2")] < 0.0)
    else:
        index = {label: position for position, label in enumerate(_MSSM_LABELS)}
        mu = values[index["mu"]]
        stop_margin = (
            3.0
            * (
                values[index["mQ32"]]
                + values[index["mU32"]]
                + values[index["mHu2"]]
                + mu**2
            )
            - values[index["At"]] ** 2
        )
        stau_margin = (
            3.0
            * (
                values[index["mL32"]]
                + values[index["mE32"]]
                + values[index["mHd2"]]
                + mu**2
            )
            - values[index["Atau"]] ** 2
        )
        margins = (stop_margin, stau_margin)
        soft = values[
            [index[label] for label in ("mQ32", "mU32", "mD32", "mL32", "mE32")]
        ]
        tachyons = int(np.sum(soft < 0.0))
        quartic = math.inf
    stable = margins[0] >= 0.0 and margins[1] >= 0.0 and tachyons == 0 and quartic >= 0.0
    status = (
        NativeSpectrumStatus.SUCCESS if stable else NativeSpectrumStatus.VACUUM_WARNING
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "native-vacuum-stability-evidence",
            "plan": plan.plan_id,
            "parameters": array_tree_fingerprint(values),
        }
    )
    return VacuumStabilityEvidence(
        higgs_quartic_margin=jnp.asarray(quartic),
        stop_ccb_margin=jnp.asarray(margins[0]),
        stau_ccb_margin=jnp.asarray(margins[1]),
        tachyon_count=jnp.asarray(tachyons, dtype=jnp.int32),
        stable=jnp.asarray(stable),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence_id=evidence_id,
    )


class NativeSpectrumBVPPlan(StrictModule):
    model: NativeSpectrumModelPlan = eqx.field(static=True)
    initial_scale: float = eqx.field(static=True)
    final_scale: float = eqx.field(static=True)
    unknown_indices: tuple[int, ...] = eqx.field(static=True)
    target_indices: tuple[int, ...] = eqx.field(static=True)
    target_values: Array
    thresholds: tuple[SpectrumThreshold, ...] = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    finite_difference_step: float = eqx.field(static=True)
    trust_radius: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: NativeSpectrumModelPlan,
        initial_scale: float,
        final_scale: float,
        unknown_indices: Sequence[int],
        target_indices: Sequence[int],
        target_values: ArrayLike,
        /,
        *,
        thresholds: Sequence[SpectrumThreshold] = (),
        residual_tolerance: float = 1e-8,
        maximum_iterations: int = 20,
        finite_difference_step: float = 1e-5,
        trust_radius: float = 1.0,
    ):
        if not isinstance(model, NativeSpectrumModelPlan):
            raise TypeError("model must be NativeSpectrumModelPlan.")
        unknown_raw = tuple(unknown_indices)
        target_raw = tuple(target_indices)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (*unknown_raw, *target_raw)
        ):
            raise TypeError("Native BVP indices must be integers.")
        unknown = tuple(int(value) for value in unknown_raw)
        target = tuple(int(value) for value in target_raw)
        target_values_ = np.asarray(target_values, dtype=np.float64)
        if (
            not unknown
            or len(unknown) != len(target)
            or len(set(unknown)) != len(unknown)
            or len(set(target)) != len(target)
            or target_values_.shape != (len(target),)
            or np.any(~np.isfinite(target_values_))
        ):
            raise ValueError(
                "Native BVP must be a finite nondegenerate square boundary system."
            )
        if any(
            value < 0 or value >= len(model.parameter_labels)
            for value in (*unknown, *target)
        ):
            raise ValueError("Native BVP indices leave the model parameter roster.")
        initial_scale_ = float(initial_scale)
        final_scale_ = float(final_scale)
        tolerance = float(residual_tolerance)
        if isinstance(maximum_iterations, bool) or not isinstance(
            maximum_iterations, Integral
        ):
            raise TypeError("maximum_iterations must be an integer.")
        iterations = int(maximum_iterations)
        difference = float(finite_difference_step)
        radius = float(trust_radius)
        if (
            not math.isfinite(initial_scale_)
            or not math.isfinite(final_scale_)
            or initial_scale_ <= 0.0
            or final_scale_ <= 0.0
            or initial_scale_ == final_scale_
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or iterations < 1
            or not math.isfinite(difference)
            or difference <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise ValueError("Native BVP scales and numerical controls are invalid.")
        thresholds_ = tuple(thresholds)
        if any(not isinstance(value, SpectrumThreshold) for value in thresholds_):
            raise TypeError("thresholds must contain SpectrumThreshold values.")
        if len({value.scale for value in thresholds_}) != len(thresholds_):
            raise ValueError("Native BVP threshold scales must be unique.")
        content = {
            "kind": "native-spectrum-bvp-plan",
            "model": model.plan_id,
            "initial_scale": initial_scale_,
            "final_scale": final_scale_,
            "unknown_indices": unknown,
            "target_indices": target,
            "target_values": array_tree_fingerprint(target_values_),
            "thresholds": [value.threshold_id for value in thresholds_],
            "residual_tolerance": tolerance,
            "maximum_iterations": iterations,
            "finite_difference_step": difference,
            "trust_radius": radius,
        }
        self.model = model
        self.initial_scale = initial_scale_
        self.final_scale = final_scale_
        self.unknown_indices = unknown
        self.target_indices = target
        self.target_values = jnp.asarray(target_values_)
        self.thresholds = thresholds_
        self.residual_tolerance = tolerance
        self.maximum_iterations = iterations
        self.finite_difference_step = difference
        self.trust_radius = radius
        self.plan_id = canonical_fingerprint(content)


class NativeSpectrumBVPRoots(StrictModule):
    initial_parameters: Array
    final_parameters: Array
    residual_norms: Array
    iteration_counts: Array
    statuses: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _native_bvp_residual(plan: NativeSpectrumBVPPlan, parameters: np.ndarray, /):
    history = integrate_native_rge(
        plan.model,
        parameters,
        plan.initial_scale,
        plan.final_scale,
        thresholds=plan.thresholds,
    )
    final = np.asarray(history.parameters[-1])
    residual = final[np.asarray(plan.target_indices)] - np.asarray(plan.target_values)
    return residual, final, int(history.status)


def solve_native_spectrum_bvp(
    plan: NativeSpectrumBVPPlan,
    base_parameters: ArrayLike,
    seeds: ArrayLike,
    /,
) -> NativeSpectrumBVPRoots:
    """Solve and deduplicate every prespecified boundary-value seed."""

    if not isinstance(plan, NativeSpectrumBVPPlan):
        raise TypeError("plan must be NativeSpectrumBVPPlan.")
    base = np.asarray(base_parameters, dtype=np.float64)
    seeds_ = np.asarray(seeds, dtype=np.float64)
    if base.shape != (len(plan.model.parameter_labels),) or np.any(~np.isfinite(base)):
        raise ValueError("Base parameters must be finite with the model shape.")
    if (
        seeds_.ndim != 2
        or seeds_.shape[0] == 0
        or seeds_.shape[1] != len(plan.unknown_indices)
        or np.any(~np.isfinite(seeds_))
    ):
        raise ValueError("BVP seeds must be a non-empty finite (seeds, unknowns) table.")
    roots: list[np.ndarray] = []
    finals: list[np.ndarray] = []
    norms: list[float] = []
    iterations: list[int] = []
    statuses: list[int] = []
    for seed in seeds_:
        candidate = base.copy()
        candidate[np.asarray(plan.unknown_indices)] = seed
        status = NativeSpectrumStatus.NO_ROOT
        final = np.full_like(base, np.nan)
        norm = math.inf
        used = 0
        for iteration in range(plan.maximum_iterations):
            used = iteration + 1
            residual, final, integration_status = _native_bvp_residual(plan, candidate)
            norm = float(np.linalg.norm(residual))
            if integration_status != int(NativeSpectrumStatus.SUCCESS):
                status = NativeSpectrumStatus.INTEGRATION_FAILURE
                break
            if norm <= plan.residual_tolerance:
                status = NativeSpectrumStatus.SUCCESS
                break
            jacobian = np.empty((residual.size, residual.size), dtype=np.float64)
            for column, parameter_index in enumerate(plan.unknown_indices):
                displacement = plan.finite_difference_step * max(
                    1.0, abs(candidate[parameter_index])
                )
                shifted = candidate.copy()
                shifted[parameter_index] += displacement
                shifted_residual, _, shifted_status = _native_bvp_residual(plan, shifted)
                if shifted_status != int(NativeSpectrumStatus.SUCCESS):
                    status = NativeSpectrumStatus.INTEGRATION_FAILURE
                    break
                jacobian[:, column] = (shifted_residual - residual) / displacement
            if status is NativeSpectrumStatus.INTEGRATION_FAILURE:
                break
            update, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            update_norm = float(np.linalg.norm(update))
            if update_norm > plan.trust_radius:
                update *= plan.trust_radius / update_norm
            candidate[np.asarray(plan.unknown_indices)] += update
        if status is NativeSpectrumStatus.SUCCESS and any(
            np.linalg.norm(
                candidate[np.asarray(plan.unknown_indices)]
                - value[np.asarray(plan.unknown_indices)]
            )
            <= 10.0 * plan.residual_tolerance
            for value in roots
        ):
            continue
        roots.append(candidate)
        finals.append(final)
        norms.append(norm)
        iterations.append(used)
        statuses.append(int(status))
    root_table = np.stack(roots) if roots else np.empty((0, base.size))
    final_table = np.stack(finals) if finals else np.empty((0, base.size))
    result_id = canonical_fingerprint(
        {
            "kind": "native-spectrum-bvp-roots",
            "plan": plan.plan_id,
            "roots": array_tree_fingerprint(root_table),
            "finals": array_tree_fingerprint(final_table),
            "residual_norms": norms,
            "statuses": statuses,
        }
    )
    return NativeSpectrumBVPRoots(
        initial_parameters=jnp.asarray(root_table),
        final_parameters=jnp.asarray(final_table),
        residual_norms=jnp.asarray(norms),
        iteration_counts=jnp.asarray(iterations, dtype=jnp.int32),
        statuses=jnp.asarray(statuses, dtype=jnp.int32),
        plan_id=plan.plan_id,
        result_id=result_id,
    )


class SpectrumUncertaintyEvidence(StrictModule):
    central_observables: Array
    input_covariance: Array
    propagated_covariance: Array
    scale_uncertainty: Array
    order_uncertainty: Array
    provider_uncertainty: Array
    total_covariance: Array
    evidence_id: str = eqx.field(static=True)


def propagate_spectrum_uncertainty(
    central_inputs: ArrayLike,
    input_covariance: ArrayLike,
    observable_samples: ArrayLike,
    input_displacements: ArrayLike,
    /,
    *,
    scale_variations: ArrayLike,
    order_variations: ArrayLike,
    provider_variations: ArrayLike,
) -> SpectrumUncertaintyEvidence:
    """Separate parametric, scale, perturbative-order, and provider components."""

    inputs = np.asarray(central_inputs, dtype=np.float64)
    covariance = np.asarray(input_covariance, dtype=np.float64)
    observables = np.asarray(observable_samples, dtype=np.float64)
    displacements = np.asarray(input_displacements, dtype=np.float64)
    if inputs.ndim != 1 or inputs.size == 0 or np.any(~np.isfinite(inputs)):
        raise ValueError("central_inputs must be a non-empty finite vector.")
    if (
        covariance.shape != (inputs.size, inputs.size)
        or np.any(~np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
    ):
        raise ValueError("input_covariance must be finite and symmetric.")
    covariance_tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(covariance, ord=2)))
    )
    if np.min(np.linalg.eigvalsh(covariance)) < -covariance_tolerance:
        raise ValueError("input_covariance must be positive semidefinite.")
    if (
        observables.ndim != 2
        or observables.shape[0] < inputs.size + 1
        or displacements.shape != (observables.shape[0], inputs.size)
        or np.any(~np.isfinite(observables))
        or np.any(~np.isfinite(displacements))
    ):
        raise ValueError(
            "Observable finite-difference samples must be finite and identify the input Jacobian."
        )
    design = np.concatenate((np.ones((displacements.shape[0], 1)), displacements), axis=1)
    if np.linalg.matrix_rank(design) != inputs.size + 1:
        raise ValueError("Spectrum uncertainty design is rank deficient.")
    coefficients, _, _, _ = np.linalg.lstsq(design, observables, rcond=None)
    central = coefficients[0]
    jacobian = coefficients[1:].T
    propagated = jacobian @ covariance @ jacobian.T

    def variation_covariance(values: ArrayLike) -> np.ndarray:
        table = np.asarray(values, dtype=np.float64)
        if (
            table.ndim != 2
            or table.shape[0] == 0
            or table.shape[1] != central.size
            or np.any(~np.isfinite(table))
        ):
            raise ValueError(
                "Spectrum variation tables must be non-empty finite tables on the observable axis."
            )
        centered = table - central
        return centered.T @ centered / table.shape[0]

    scale = variation_covariance(scale_variations)
    order = variation_covariance(order_variations)
    provider = variation_covariance(provider_variations)
    total = propagated + scale + order + provider
    evidence_id = canonical_fingerprint(
        {
            "kind": "spectrum-uncertainty-evidence",
            "inputs": array_tree_fingerprint(inputs),
            "input_covariance": array_tree_fingerprint(covariance),
            "observables": array_tree_fingerprint(observables),
            "displacements": array_tree_fingerprint(displacements),
            "scale": array_tree_fingerprint(np.asarray(scale_variations)),
            "order": array_tree_fingerprint(np.asarray(order_variations)),
            "provider": array_tree_fingerprint(np.asarray(provider_variations)),
        }
    )
    return SpectrumUncertaintyEvidence(
        central_observables=jnp.asarray(central),
        input_covariance=jnp.asarray(covariance),
        propagated_covariance=jnp.asarray(propagated),
        scale_uncertainty=jnp.asarray(scale),
        order_uncertainty=jnp.asarray(order),
        provider_uncertainty=jnp.asarray(provider),
        total_covariance=jnp.asarray(total),
        evidence_id=evidence_id,
    )


class SpectrumScanResult(StrictModule):
    final_parameters: Array
    statuses: Array
    successful_mask: Array
    plan_id: str = eqx.field(static=True)
    scan_id: str = eqx.field(static=True)


def scan_native_spectrum(
    plan: NativeSpectrumModelPlan,
    parameter_points: ArrayLike,
    initial_scale: float,
    final_scale: float,
    /,
    *,
    thresholds: Sequence[SpectrumThreshold] = (),
) -> SpectrumScanResult:
    points = np.asarray(parameter_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(plan.parameter_labels):
        raise ValueError("Spectrum scan points have the wrong shape.")
    histories = tuple(
        integrate_native_rge(
            plan,
            point,
            initial_scale,
            final_scale,
            thresholds=thresholds,
        )
        for point in points
    )
    finals = np.stack([np.asarray(value.parameters[-1]) for value in histories])
    statuses = np.asarray([int(value.status) for value in histories], dtype=np.int32)
    successful = statuses == int(NativeSpectrumStatus.SUCCESS)
    scan_id = canonical_fingerprint(
        {
            "kind": "native-spectrum-scan",
            "plan": plan.plan_id,
            "points": array_tree_fingerprint(points),
            "thresholds": [value.threshold_id for value in thresholds],
            "statuses": statuses.tolist(),
        }
    )
    return SpectrumScanResult(
        final_parameters=jnp.asarray(finals),
        statuses=jnp.asarray(statuses),
        successful_mask=jnp.asarray(successful),
        plan_id=plan.plan_id,
        scan_id=scan_id,
    )


@dataclass(frozen=True, slots=True)
class SpectrumCalculatorAdapter:
    """Calculator-specific semantic binding above the pinned process provider."""

    provider_id: str
    model: NativeSpectrumModel
    required_blocks: tuple[str, ...]
    approximation_id: str
    adapter_id: str

    def __init__(
        self,
        provider_id: str,
        model: NativeSpectrumModel,
        required_blocks: Sequence[str],
        approximation_id: str,
        /,
    ):
        provider = str(provider_id).strip()
        blocks = tuple(sorted(str(value).upper() for value in required_blocks))
        approximation = str(approximation_id).strip()
        if (
            not provider
            or not blocks
            or len(set(blocks)) != len(blocks)
            or not approximation
        ):
            raise ValueError("Spectrum calculator adapter semantics are incomplete.")
        content = {
            "kind": "spectrum-calculator-adapter",
            "provider_id": provider,
            "model": model,
            "required_blocks": blocks,
            "approximation_id": approximation,
        }
        object.__setattr__(self, "provider_id", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "required_blocks", blocks)
        object.__setattr__(self, "approximation_id", approximation)
        object.__setattr__(self, "adapter_id", canonical_fingerprint(content))


class SpectrumCrossQualificationEvidence(StrictModule):
    labels: tuple[str, ...] = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    relative_residuals: Array
    maximum_relative_residual: Array
    accepted: Array
    status: Array
    evidence_id: str = eqx.field(static=True)


def cross_qualify_spectra(
    labels: Sequence[str],
    native_values: ArrayLike,
    provider_values: Mapping[str, ArrayLike],
    /,
    *,
    relative_tolerance: float,
) -> SpectrumCrossQualificationEvidence:
    labels_ = tuple(str(value).strip() for value in labels)
    native = np.asarray(native_values, dtype=np.float64)
    if not isinstance(provider_values, Mapping):
        raise TypeError("provider_values must be a mapping.")
    provider_items = tuple(
        sorted(
            (
                (
                    str(key).strip(),
                    np.asarray(value, dtype=np.float64),
                    str(key),
                )
                for key, value in provider_values.items()
            ),
            key=lambda item: item[0],
        )
    )
    providers = tuple(item[0] for item in provider_items)
    if native.shape != (len(labels_),) or not providers:
        raise ValueError("Spectrum cross-qualification inputs are incomplete.")
    table = np.stack([item[1] for item in provider_items])
    if table.shape != (len(providers), len(labels_)):
        raise ValueError("Provider spectra do not share the native observable roster.")
    tolerance = float(relative_tolerance)
    input_valid = (
        bool(labels_)
        and all(labels_)
        and len(set(labels_)) == len(labels_)
        and all(providers)
        and len(set(providers)) == len(providers)
        and math.isfinite(tolerance)
        and all(item[0] == item[2] for item in provider_items)
        and tolerance >= 0.0
        and np.all(np.isfinite(native))
        and np.all(np.isfinite(table))
    )
    if input_valid:
        residuals = np.abs(table - native[None, :]) / np.maximum(
            1.0, np.abs(native[None, :])
        )
        maximum = float(np.max(residuals))
        accepted = maximum <= tolerance
        status = NativeSpectrumStatus.SUCCESS
    else:
        residuals = np.zeros_like(table)
        maximum = math.inf
        accepted = False
        status = NativeSpectrumStatus.INVALID_INPUT
    evidence_id = canonical_fingerprint(
        {
            "kind": "spectrum-cross-qualification-evidence",
            "labels": labels_,
            "native": array_tree_fingerprint(native),
            "providers": [
                [name, original, array_tree_fingerprint(value)]
                for name, value, original in provider_items
            ],
            "relative_tolerance": tolerance,
        }
    )
    return SpectrumCrossQualificationEvidence(
        labels=labels_,
        provider_ids=providers,
        relative_residuals=jnp.asarray(residuals),
        maximum_relative_residual=jnp.asarray(maximum),
        accepted=jnp.asarray(accepted),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence_id=evidence_id,
    )


__all__ = [
    "ElectroweakSymmetryBreakingEvidence",
    "NativePoleMassEvidence",
    "NativeRGEHistory",
    "NativeSpectrumBVPPlan",
    "NativeSpectrumBVPRoots",
    "NativeSpectrumModel",
    "NativeSpectrumModelPlan",
    "NativeSpectrumStatus",
    "SpectrumCalculatorAdapter",
    "SpectrumCrossQualificationEvidence",
    "SpectrumScanResult",
    "SpectrumThreshold",
    "SpectrumUncertaintyEvidence",
    "VacuumStabilityEvidence",
    "assess_native_vacuum_stability",
    "compute_native_mssm_pole_masses",
    "cross_qualify_spectra",
    "integrate_native_rge",
    "native_beta_function",
    "propagate_spectrum_uncertainty",
    "scan_native_spectrum",
    "solve_mssm_electroweak_breaking",
    "solve_native_spectrum_bvp",
]
