# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Qualify filtered, exact predicates and symbolic predicates against integer references."""

from __future__ import annotations

from collections.abc import Callable
from typing import final, Protocol, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import StrictModule
from phydrax._meshcore import (
    meshcore_exact_domain,
    symbolic_incircle,
    symbolic_insphere,
    symbolic_orient2d,
    symbolic_orient3d,
)


_PREDICATE_ARITY = {"orient2d": 3, "orient3d": 4, "incircle": 4, "insphere": 5}


class _PredicateStageRecord(TypedDict):
    """Wall seconds and post-stage process peak bytes, or None when unavailable."""

    wall_seconds: float
    peak_resident_bytes: int | None


class PredicateStageMeasurement(Protocol):
    def __call__[T](
        self, call: Callable[[], T], /
    ) -> tuple[T, _PredicateStageRecord]: ...


@final
class PredicateFamilyQualification(StrictModule):
    """Predicate agreement and resource evidence for one adversarial family batch."""

    records: dict[str, object] = eqx.field(static=True)
    resources: dict[str, object] = eqx.field(static=True)
    exact_signs: np.ndarray
    antisymmetry_violations: int = eqx.field(static=True)
    exact_exponent_range: tuple[int, int] = eqx.field(static=True)


def _predicate_integers(arrays: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, ...]:
    """Represent float64 coordinates as Python integers on one common binary scale."""
    mantissas, exponents = [], []
    for array in arrays:
        fraction, exponent = np.frexp(array)
        mantissas.append(np.ldexp(fraction, 53).astype(np.int64))
        exponents.append(exponent.astype(np.int64) - 53)
    scale = min(
        np.min(np.where(mantissa != 0, exponent, np.iinfo(np.int64).max))
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )
    return tuple(
        np.left_shift(
            mantissa.astype(np.object_),
            np.where(mantissa != 0, exponent - scale, 0).astype(np.object_),
        )
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )


def _integer_determinant(rows: list[list[np.ndarray]], /) -> np.ndarray:
    """Laplace expansion of a small matrix whose entries are integer object arrays."""
    if len(rows) == 1:
        return rows[0][0]
    total = np.zeros(rows[0][0].shape, dtype=np.object_)
    for column, entry in enumerate(rows[0]):
        minor = [row[:column] + row[column + 1 :] for row in rows[1:]]
        term = entry * _integer_determinant(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


def reference_predicate_signs(name: str, arrays: tuple[np.ndarray, ...], /) -> np.ndarray:
    """Predicate signs in the phydrax convention from exact integer arithmetic."""
    points = _predicate_integers(arrays)
    width = arrays[0].shape[-1]
    match name:
        case "orient2d" | "orient3d":
            # det[b - a, c - a(, d - a)]
            determinant = _integer_determinant(
                [
                    [point[:, axis] - points[0][:, axis] for axis in range(width)]
                    for point in points[1:]
                ]
            )
        case "incircle" | "insphere":
            # Lifted rows (p - q, |p - q|^2) against the query q; insphere is
            # positive inside the sphere of a positively oriented tetrahedron.
            rows = [
                [point[:, axis] - points[-1][:, axis] for axis in range(width)]
                for point in points[:-1]
            ]
            determinant = _integer_determinant(
                [row + [sum(value * value for value in row)] for row in rows]
            )
            if name == "insphere":
                determinant = -determinant
        case _:
            raise ValueError(
                "qualification.predicates.name must be orient2d, orient3d, incircle, "
                f"or insphere; got {name!r}"
            )
    return (determinant > 0).astype(np.int8) - (determinant < 0).astype(np.int8)


def _predicate_family_record(
    routes: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: np.ndarray,
    perturbed: np.ndarray,
    /,
) -> dict[str, object]:
    """Certainty, correctness, agreement, and degeneracy counts of one family."""
    rows = reference.size
    exact = routes["exact"][0]
    record: dict[str, object] = {"rows": rows}
    for route, (signs, certain) in routes.items():
        record[route] = {
            "certain": int(np.count_nonzero(certain)),
            "uncertain_fraction": 1.0 - int(np.count_nonzero(certain)) / rows,
            "certified_zeros": int(np.count_nonzero(certain & (signs == 0))),
            "wrong_certain_signs": int(np.count_nonzero(certain & (signs != reference))),
        }
    pairs = (("filtered", "device"), ("filtered", "exact"), ("device", "exact"))
    record["certain_disagreements"] = {
        f"{first}_{second}": int(
            np.count_nonzero(
                routes[first][1]
                & routes[second][1]
                & (routes[first][0] != routes[second][0])
            )
        )
        for first, second in pairs
    }
    record["exact_signs"] = {
        "negative": int(np.count_nonzero(exact == -1)),
        "zero": int(np.count_nonzero(exact == 0)),
        "positive": int(np.count_nonzero(exact == 1)),
    }
    record["reference_zeros"] = int(np.count_nonzero(reference == 0))
    record["symbolic"] = {
        "zeros": int(np.count_nonzero(perturbed == 0)),
        "changed_strict_signs": int(
            np.count_nonzero((exact != 0) & (perturbed != exact))
        ),
    }
    return record


def qualify_predicate_families(
    name: str,
    families: dict[str, tuple[np.ndarray, ...]],
    /,
    *,
    measure: PredicateStageMeasurement,
) -> PredicateFamilyQualification:
    """Evaluate one predicate in every mode on all families against the reference."""

    function = {
        "orient2d": phx.geometry.orient2d,
        "orient3d": phx.geometry.orient3d,
        "incircle": phx.geometry.incircle,
        "insphere": phx.geometry.insphere,
    }[name]
    mode = phx.geometry.PredicateMode
    arity = _PREDICATE_ARITY[name]
    arrays = tuple(
        np.concatenate([rows[index] for rows in families.values()])
        for index in range(arity)
    )
    bounds = np.cumsum([0, *(rows[0].shape[0] for rows in families.values())])
    filtered, filtered_stage = measure(lambda: function(*arrays, mode=mode.FILTERED))
    device_entry = jax.jit(lambda *values: function(*values, mode=mode.FILTERED_DEVICE))
    device_arrays = tuple(jnp.asarray(array) for array in arrays)
    _, device_cold = measure(lambda: jax.block_until_ready(device_entry(*device_arrays)))
    device, device_warm = measure(
        lambda: jax.block_until_ready(device_entry(*device_arrays))
    )
    exact, exact_stage = measure(lambda: function(*arrays, mode=mode.EXACT))
    reference, reference_stage = measure(
        lambda: np.concatenate(
            [reference_predicate_signs(name, rows) for rows in families.values()]
        )
    )
    swapped = function(arrays[1], arrays[0], *arrays[2:], mode=mode.EXACT).signs
    # Distinct ids: the index-ordered perturbation used inside the triangulations.
    identifiers = np.arange(arity, dtype=np.int64)
    match name:
        case "orient2d":
            perturbed = symbolic_orient2d(arrays[0], arrays[1], arrays[2], identifiers)
        case "orient3d":
            perturbed = symbolic_orient3d(
                arrays[0], arrays[1], arrays[2], arrays[3], identifiers
            )
        case "incircle":
            perturbed = symbolic_incircle(
                arrays[0], arrays[1], arrays[2], arrays[3], identifiers
            )
        case "insphere":
            perturbed = symbolic_insphere(
                arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], identifiers
            )
        case _:
            raise ValueError(
                "qualification.predicates.name must be orient2d, orient3d, incircle, "
                f"or insphere; got {name!r}"
            )
    routes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "filtered": (np.asarray(filtered.signs), np.asarray(filtered.certain)),
        "device": (np.asarray(device.signs), np.asarray(device.certain)),
        "exact": (np.asarray(exact.signs), np.asarray(exact.certain)),
    }
    records: dict[str, object] = {
        family: _predicate_family_record(
            {
                route: (signs[start:stop], certain[start:stop])
                for route, (signs, certain) in routes.items()
            },
            reference[start:stop],
            perturbed[start:stop],
        )
        for family, start, stop in zip(families, bounds[:-1], bounds[1:], strict=True)
    }
    resources: dict[str, object] = {
        "filtered": filtered_stage,
        "device_cold": device_cold,
        "device_warm": device_warm,
        "exact": exact_stage,
        "exact_native_rows": int(np.count_nonzero(~filtered.certain)),
        "integer_reference": reference_stage,
    }
    return PredicateFamilyQualification(
        records=records,
        resources=resources,
        exact_signs=np.asarray(exact.signs),
        antisymmetry_violations=int(np.count_nonzero(swapped != -exact.signs)),
        exact_exponent_range=meshcore_exact_domain(),
    )
