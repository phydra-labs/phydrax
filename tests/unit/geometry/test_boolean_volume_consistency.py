"""Seeded, bounded host placements check finite-additivity of certified volume.

The domain stratifies separated mixed primitives, overlapping boxes, and nested
curved primitives. Dyadic placements avoid introducing an unrelated source
rounding uncertainty. These fixtures are independently authored native solids.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._boolean_core import BRepBooleanFailure, BRepBooleanPolicy
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    brep_sphere,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._intersection import ParametricIntersectionPolicy
from phydrax.geometry.brep._measure import brep_mass_properties
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import PlanePatch


_CONTRACT = SpatialCoordinateContract.si()
_LAZY = BRepTessellationPolicy(realize=False)
_SEEDS = (290030, 290031, 290032)


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=512 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=512 << 20
        )
        with budget.activate():
            yield budget


def _operand(kind: str, center: np.ndarray, scale: float) -> BRepModel:
    if kind == "box":
        return brep_box(
            center - scale,
            center + scale,
            coordinate_contract=_CONTRACT,
            tessellation=_LAZY,
        )
    if kind == "cylinder":
        return brep_cylinder(
            scale,
            2 * scale,
            base_center=center - np.asarray((0.0, 0.0, scale)),
            coordinate_contract=_CONTRACT,
            tessellation=_LAZY,
        )
    if kind == "sphere":
        return brep_sphere(
            scale, center=center, coordinate_contract=_CONTRACT, tessellation=_LAZY
        )
    raise ValueError(f"corpus.primitive must be box, cylinder or sphere; got {kind}")


def _volume(
    model: BRepModel, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, Fraction]:
    if model.topology.num_solids == 0:
        return Fraction(), Fraction()
    subdivisions = (
        2 if all(isinstance(patch, PlanePatch) for patch in model.patches) else 16
    )
    properties = brep_mass_properties(
        model,
        model.require_geometry(),
        subdivisions=subdivisions,
        coordinate_budget=budget,
    )
    bounds = np.asarray(properties.volume_enclosures)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 0] > 0)
    assert np.all(bounds[:, 0] <= bounds[:, 1])
    # Rational endpoint sums avoid inward floating-point rounding in the oracle.
    return (
        sum((Fraction(float(value)) for value in bounds[:, 0]), Fraction()),
        sum((Fraction(float(value)) for value in bounds[:, 1]), Fraction()),
    )


def _overlap(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], label: str
) -> None:
    assert max(first[0], second[0]) <= min(first[1], second[1]), label


def _check_pair(first: BRepModel, second: BRepModel) -> None:
    policy = BRepBooleanPolicy(
        tessellation=_LAZY,
        intersection=ParametricIntersectionPolicy(
            maximum_boxes=200_000, maximum_march_steps=256
        ),
    )
    union = boolean_brep(first, second, "union", policy=policy).model
    intersection = boolean_brep(first, second, "intersection", policy=policy).model
    difference = boolean_brep(first, second, "difference", policy=policy).model
    with _budget() as budget:
        a, b, u, i, d = (
            _volume(model, budget)
            for model in (first, second, union, intersection, difference)
        )
    _overlap(
        (u[0] + i[0], u[1] + i[1]),
        (a[0] + b[0], a[1] + b[1]),
        "union + intersection = operands",
    )
    _overlap((d[0] + i[0], d[1] + i[1]), a, "difference + intersection = first")


@pytest.mark.parametrize("seed", _SEEDS, ids=lambda seed: f"numpy-seed-{seed}")
def test_seeded_boolean_volume_enclosures_are_finitely_additive(
    seed: int, record_property: Callable[[str, object], None]
) -> None:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str, str, np.ndarray, np.ndarray, float, float]] = []
    for first_kind, second_kind in combinations_with_replacement(
        ("box", "cylinder", "sphere"), 2
    ):
        first = rng.integers(-8, 9, size=3).astype(np.float64) / 8
        second = first + np.asarray((4.0, 0.0, 0.0))
        second += rng.integers(-2, 3, size=3).astype(np.float64) / 8
        pairs.append(
            (
                f"separated-{first_kind}-{second_kind}",
                first_kind,
                second_kind,
                first,
                second,
                0.5,
                0.5,
            )
        )
    for index in range(2):
        first = rng.integers(-8, 9, size=3).astype(np.float64) / 8
        second = first + rng.integers(1, 5, size=3).astype(np.float64) / 8
        pairs.append((f"overlap-box-{index}", "box", "box", first, second, 0.5, 0.5))
    for kind in ("sphere", "cylinder"):
        first = rng.integers(-8, 9, size=3).astype(np.float64) / 8
        second = first + rng.integers(-1, 2, size=3).astype(np.float64) / 16
        pairs.append((f"nested-{kind}", kind, kind, first, second, 1.0, 0.25))
    refused: list[str] = []
    for label, first_kind, second_kind, first, second, first_scale, second_scale in pairs:
        print(f"seed={seed}, case={label}", flush=True)
        try:
            _check_pair(
                _operand(first_kind, first, first_scale),
                _operand(second_kind, second, second_scale),
            )
        except BRepBooleanFailure as error:
            assert error.reason
            refused.append(
                f"{label}: unresolved: {error.reason}; sources={error.source_entities}"
            )
        except CoordinateEnclosureResourceError as error:
            refused.append(f"{label}: resource-limit: {error}")
        except AssertionError as error:
            raise AssertionError(
                f"boolean.volume_identity must hold; got seed={seed}, case={label}: {error}"
            ) from error
    certified = len(pairs) - len(refused)
    record_property("certified_cases", certified)
    record_property("generated_cases", len(pairs))
    record_property("typed_refusals", "\n".join(refused))
    print(
        f"seed={seed}: certified {certified}/{len(pairs)} ({certified / len(pairs):.0%}); refusals={refused}"
    )
    assert certified / len(pairs) >= 0.8, (
        f"seed={seed}: {certified}/{len(pairs)} certified; {refused}"
    )
