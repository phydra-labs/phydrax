# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import assert_never, Literal

import numpy as np
from numpy.typing import NDArray


type Rule = Literal["off", "mass", "distortion", "removal"]
type Policy = Literal["mix", "frozen", "restart", "settle"]
type Split = Literal["radial", "multiplicative"]
type BatchLaw = Literal["quality", "throughput"]
type Array = NDArray[np.floating]


def parse_rule(value: str) -> Rule:
    match value:
        case "off" | "mass" | "distortion" | "removal":
            return value
        case _:
            raise ValueError(f"Invalid revival rule: {value}")


def parse_policy(value: str) -> Policy:
    match value:
        case "mix" | "frozen" | "restart" | "settle":
            return value
        case _:
            raise ValueError(f"Invalid last epoch policy: {value}")


def parse_split(value: str) -> Split:
    match value:
        case "radial" | "multiplicative":
            return value
        case _:
            raise ValueError(f"Invalid split: {value}")


def parse_batch_law(value: str) -> BatchLaw:
    match value:
        case "quality" | "throughput":
            return value
        case _:
            raise ValueError(f"Invalid batch law: {value}")


@dataclass(frozen=True, slots=True)
class EMAConfig:
    k: int
    epochs: float
    seed: int = 0
    batch_size: int | None = None
    alpha0: float | None = None
    batch_law: BatchLaw = "quality"
    rule: Rule = "mass"
    policy: Policy = "frozen"
    split: Split = "radial"
    centered: bool = True
    block_size: int = 2048

    def __post_init__(self) -> None:
        parse_rule(self.rule)
        parse_policy(self.policy)
        parse_split(self.split)
        parse_batch_law(self.batch_law)
        if (
            self.k < 1
            or self.epochs <= 0
            or not math.isfinite(self.epochs)
            or self.block_size < 1
        ):
            raise ValueError("Invalid cluster count, epochs, or block size")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.alpha0 is not None and not 0 < self.alpha0 <= 1:
            raise ValueError("alpha0 must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EMAResult:
    centers: Array
    ema_centers: Array
    initial_centers: Array
    weights: Array
    closing_objective: float
    epoch_objectives: tuple[float, ...]
    batch_size: int
    alpha0: float
    updates: int
    revivals: int
    closing_gain: float
    imbalance: float
    dead_count: int
    wall_seconds: float


def round_pow2(value: float) -> int:
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    high = 2 * low
    return low if value / low < high / value else high


def resolve_batch_size(
    n: int,
    k: int,
    epochs: float,
    law: BatchLaw = "quality",
    batch_size: int | None = None,
) -> int:
    if batch_size is not None:
        return min(batch_size, n)
    match law:
        case "quality":
            coefficient, cap = 0.00439175, n // 2
        case "throughput":
            coefficient, cap = 0.035134, round_pow2(n / 2)
        case _:
            assert_never(law)
    raw = coefficient * k**0.134749 * n**0.876473 * epochs**0.813344
    return max(1, min(round_pow2(raw), cap, n))


def resolve_alpha0(n: int, k: int, epochs: float, batch_size: int) -> float:
    return float(
        np.clip(
            0.8244 * k**-0.4617 * batch_size**0.9003 * n**-0.4498 * epochs**-0.1057,
            1e-4,
            1,
        )
    )


def epoch_rows(n: int, epochs: float) -> list[int]:
    whole = math.floor(epochs)
    remainder = round((epochs - whole) * n)
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def nearest_centroids(
    x: Array, centers: Array, block_size: int = 2048, second: bool = False
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64] | None]:
    n = x.shape[0]
    labels = np.empty(n, dtype=np.intp)
    first = np.empty(n, dtype=np.float64)
    second_dist = np.empty(n, dtype=np.float64) if second else None
    c = centers.astype(x.dtype, copy=False)
    norms = np.sum(c * c, axis=1)
    for start in range(0, n, block_size):
        block = x[start : start + block_size]
        scores = norms[None, :] - 2 * (block @ c.T)
        idx = np.argmin(scores, axis=1)
        rows = np.arange(block.shape[0])
        xnorm = np.sum(block * block, axis=1)
        labels[start : start + len(block)] = idx
        first[start : start + len(block)] = np.maximum(scores[rows, idx] + xnorm, 0)
        if second_dist is not None:
            if centers.shape[0] < 2:
                second_dist[start : start + len(block)] = first[
                    start : start + len(block)
                ]
            else:
                partitioned = np.partition(scores, 1, axis=1)
                second_dist[start : start + len(block)] = np.maximum(
                    partitioned[:, 1] + xnorm, 0
                )
    return labels, first, second_dist


def _sums(
    x: Array, labels: NDArray[np.intp], weights: Array, k: int
) -> tuple[Array, Array]:
    mass = np.bincount(labels, weights=weights, minlength=k)
    sums = np.column_stack(
        [
            np.bincount(labels, weights=weights * x[:, j], minlength=k)
            for j in range(x.shape[1])
        ]
    )
    return mass, sums


def _revive(
    c: Array,
    mass: Array,
    distortion: Array,
    removal: Array,
    config: EMAConfig,
    donors: np.random.Generator,
    directions: np.random.Generator,
) -> NDArray[np.intp]:
    """Split donors into dead slots; return every slot the splits rewrote."""
    match config.rule:
        case "off":
            return np.empty(0, dtype=np.intp)
        case "mass":
            score, donor_weight = mass, mass
        case "distortion":
            score, donor_weight = distortion, distortion
        case "removal":
            score, donor_weight = removal, distortion
        case _:
            assert_never(config.rule)
    dead = score < 0.1 * np.mean(score)
    eligible = np.flatnonzero(~dead & (donor_weight > 0))
    children = np.flatnonzero(dead)
    count = min(len(eligible), len(children))
    if count == 0:
        return np.empty(0, dtype=np.intp)
    keys = np.log(donor_weight[eligible]) + donors.gumbel(size=len(eligible))
    selected = eligible[np.argsort(-keys)[:count]]
    for child, donor in zip(children[:count], selected, strict=True):
        match config.split:
            case "radial":
                direction = directions.normal(size=c.shape[1])
                direction /= np.linalg.norm(direction)
                delta = 2**-7 * math.sqrt(distortion[donor] / mass[donor]) * direction
            case "multiplicative":
                delta = c[donor] * (
                    1e-4 * np.where(np.arange(c.shape[1]) % 2 == 0, 1, -1)
                )
            case _:
                assert_never(config.split)
        c[child] = c[donor] + delta
        c[donor] -= delta
        for state in (mass, distortion, removal):
            state[donor] *= 0.5
            state[child] = state[donor]
    return np.concatenate((children[:count], selected))


def reference_ema_kmeans(
    x: Array, config: EMAConfig, weights: Array | None = None
) -> EMAResult:
    started = time.perf_counter()
    if (
        x.ndim != 2
        or x.shape[0] < config.k
        or not np.issubdtype(x.dtype, np.floating)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid data or cluster count")
    n, d = x.shape
    w = (
        np.ones(n, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        w.shape != (n,)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or np.count_nonzero(w) < config.k
    ):
        raise ValueError("Weights must have at least k positive finite entries")
    offset = (
        np.average(x, axis=0, weights=w)
        if config.centered
        else np.zeros(d, dtype=np.float64)
    )
    work = (x - offset).astype(x.dtype)
    streams = [
        np.random.Generator(np.random.PCG64(s))
        for s in np.random.SeedSequence(config.seed).spawn(4)
    ]
    init_rng, order_rng, donor_rng, direction_rng = streams
    positive = np.flatnonzero(w > 0)
    keys = np.log(w[positive]) + init_rng.gumbel(size=len(positive))
    initial = work[positive[np.argsort(-keys)[: config.k]]].astype(np.float64)
    c = initial.copy()
    mass = np.zeros(config.k, dtype=np.float64)
    distortion = mass.copy()
    removal = mass.copy()
    closing_w = mass.copy()
    closing_mean = np.zeros((config.k, d), dtype=np.float64)
    closing_q = mass.copy()
    batch = resolve_batch_size(
        n, config.k, config.epochs, config.batch_law, config.batch_size
    )
    alpha0 = (
        config.alpha0
        if config.alpha0 is not None
        else resolve_alpha0(n, config.k, config.epochs, batch)
    )
    rows = epoch_rows(n, config.epochs)
    updates = sum((r + batch - 1) // batch for r in rows)
    if updates == 0:
        raise ValueError("Epoch schedule processes no rows")
    t = 0
    revivals = 0
    objectives = []
    for epoch, count in enumerate(rows):
        order = order_rng.permutation(n)[:count]
        epoch_dist, epoch_weight = 0.0, 0.0
        for start in range(0, count, batch):
            indices = order[start : start + batch]
            xb, wb = work[indices], w[indices]
            labels, d1, d2 = nearest_centroids(
                xb, c, config.block_size, config.rule == "removal"
            )
            sizes, sums = _sums(xb, labels, wb, config.k)
            delta = np.bincount(labels, weights=wb * d1, minlength=config.k)
            utility = (
                np.bincount(labels, weights=wb * (d2 - d1), minlength=config.k)
                if d2 is not None
                else np.zeros(config.k, dtype=np.float64)
            )
            epoch_dist += float(np.dot(wb, d1))
            epoch_weight += float(np.sum(wb))
            alpha = alpha0 * (0.1 + 0.45 * (1 + math.cos(math.pi * t / updates)))
            previous = mass.copy()
            mass *= 1 - alpha
            mass += alpha * sizes
            distortion *= 1 - alpha
            distortion += alpha * delta
            removal *= 1 - alpha
            removal += alpha * utility
            active = sizes > 0
            means = np.zeros_like(c)
            means[active] = sums[active] / sizes[active, None]
            fresh = active & (previous == 0)
            c[fresh] = means[fresh]
            old = active & ~fresh
            c[old] += (alpha * sizes[old] / mass[old])[:, None] * (means[old] - c[old])
            if epoch == len(rows) - 1:
                residual = xb.astype(np.float64) - means[labels]
                m2 = np.bincount(
                    labels,
                    weights=wb * np.sum(residual * residual, axis=1),
                    minlength=config.k,
                )
                combined = closing_w + sizes
                diff = means[active] - closing_mean[active]
                closing_q[active] += (
                    m2[active]
                    + np.sum(diff * diff, axis=1)
                    * closing_w[active]
                    * sizes[active]
                    / combined[active]
                )
                closing_mean[active] += diff * (sizes[active] / combined[active])[:, None]
                closing_w = combined
            t += 1
            last = epoch == len(rows) - 1
            # "settle" freezes the closing pass after a leading pass and restarts
            # rewritten slots when the closing pass is the only one.
            frozen = config.policy == "frozen" or (
                config.policy == "settle" and len(rows) > 1
            )
            if t % 5 == 0 and not (frozen and last):
                touched = _revive(
                    c, mass, distortion, removal, config, donor_rng, direction_rng
                )
                revivals += len(touched) // 2
                if last and config.policy != "mix":
                    # Rewritten slots close over their post-split rows only.
                    closing_w[touched] = 0.0
                    closing_mean[touched] = 0.0
                    closing_q[touched] = 0.0
        objectives.append(epoch_dist / epoch_weight if epoch_weight else 0.0)
    active = closing_w > 0
    final = np.where(active[:, None], closing_mean, c)
    q = float(np.sum(closing_q))
    total = float(np.sum(closing_w))
    gain_numerator = float(np.sum(closing_w[:, None] * (closing_mean - c) ** 2))
    gain = gain_numerator / q if q else (math.inf if gain_numerator else 0.0)
    return EMAResult(
        final + offset,
        c + offset,
        initial + offset,
        closing_w,
        q / total,
        tuple(objectives),
        batch,
        alpha0,
        updates,
        revivals,
        gain,
        float(config.k * np.sum((closing_w / total) ** 2)),
        int(np.count_nonzero(~active)),
        time.perf_counter() - started,
    )


def reference_lloyd(
    x: Array,
    initial_centers: Array,
    iterations: int,
    weights: Array | None = None,
    block_size: int = 2048,
) -> tuple[Array, tuple[float, ...]]:
    w = (
        np.ones(x.shape[0], dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    centers = initial_centers.astype(np.float64, copy=True)
    objectives = []
    for _ in range(iterations):
        labels, distances, _ = nearest_centroids(x, centers, block_size)
        objectives.append(float(np.dot(w, distances) / np.sum(w)))
        mass, sums = _sums(x, labels, w, len(centers))
        active = mass > 0
        centers[active] = sums[active] / mass[active, None]
    return centers, tuple(objectives)


def test_mse(x: Array, centers: Array, block_size: int = 2048) -> float:
    _, distances, _ = nearest_centroids(x, centers, block_size)
    return float(np.mean(distances))
