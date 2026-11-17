from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.geometry._interval_enclosure import (
    interval_cos,
    interval_sin,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from phydrax.geometry.brep import _root_bindings as roots
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch, TorusPatch


def _trigonometry(parameters: Array) -> Array:
    return jnp.stack((jnp.cos(parameters), jnp.sin(parameters)))


@pytest.mark.parametrize("torus", (False, True))
def test_patch_box_randomized_host_parity_and_partition_independence(
    torus: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch = (
        TorusPatch(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            float(Fraction(7, 3)),
            float(Fraction(2, 7)),
            major_radius_exact=Fraction(7, 3),
            minor_radius_exact=Fraction(2, 7),
        )
        if torus
        else PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    )
    rng = np.random.default_rng(8729)
    lower = rng.uniform(-3.0, 3.0, size=(24, 2))
    upper = lower + rng.uniform(0.0, 0.5, size=(24, 2))
    host_trig = prepare_interval_function(_trigonometry, 2, batch_capacity=1)
    host_patch = prepare_interval_function(patch.evaluate, 2, batch_capacity=1)
    with monkeypatch.context() as host:
        host.setattr(roots, "_torus_trigonometric_program", lambda: host_trig)
        expected = np.stack(
            [
                roots._patch_box(
                    patch,
                    np.asarray(lo, dtype=np.float64),
                    np.asarray(hi, dtype=np.float64),
                    interval_program=lambda _: host_patch,
                )
                for lo, hi in zip(lower, upper, strict=True)
            ]
        )

    def evaluate(index: int) -> np.ndarray:
        return roots._patch_box(patch, lower[index], upper[index])

    sequential = np.stack([evaluate(index) for index in range(len(lower))])
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = np.stack(list(pool.map(evaluate, range(len(lower)))))
    np.testing.assert_array_equal(sequential.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(parallel.view(np.uint64), expected.view(np.uint64))


def test_torus_trigonometric_batch_partition_invariance() -> None:
    rng = np.random.default_rng(492)
    lower = rng.uniform(-6.0, 6.0, size=(31, 2))
    upper = lower + rng.uniform(0.0, 1.0, size=(31, 2))
    program = roots._torus_trigonometric_program()
    whole = program.evaluate(lower, upper)
    parts = [
        program.evaluate(lower[a:b], upper[a:b]) for a, b in ((0, 3), (3, 19), (19, 31))
    ]
    for endpoint in range(2):
        split = np.concatenate([part[endpoint] for part in parts])
        np.testing.assert_array_equal(
            whole[endpoint].view(np.uint64), split.view(np.uint64)
        )


def test_inward_trigonometric_fault_is_visible_at_torus_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3, 1)
    origin = np.zeros(2, dtype=np.float64)
    correct = roots._patch_box(patch, origin, origin)
    assert correct[0, 0] <= 4.0 <= correct[1, 0]
    original = PreparedIntervalFunction.evaluate

    def inward(
        self: PreparedIntervalFunction, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = original(self, lower, upper)
        lo, hi = lo.copy(), hi.copy()
        lo[:, 0, :] = np.nextafter(lo[:, 0, :], np.inf)
        hi[:, 0, :] = np.nextafter(hi[:, 0, :], np.inf)
        return lo, hi

    monkeypatch.setattr(PreparedIntervalFunction, "evaluate", inward)
    faulty = roots._patch_box(patch, origin, origin)
    assert faulty[0, 0] > 4.0


def test_inward_fault_is_visible_at_plane_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    origin = np.zeros(2, dtype=np.float64)
    correct = roots._patch_box(patch, origin, origin)
    assert correct[0, 0] <= 0.0 <= correct[1, 0]
    original = PreparedIntervalFunction.evaluate

    def inward(
        self: PreparedIntervalFunction, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = original(self, lower, upper)
        lo = lo.copy()
        lo[:, 0] = np.nextafter(0.0, np.inf)
        return lo, hi

    monkeypatch.setattr(PreparedIntervalFunction, "evaluate", inward)
    faulty = roots._patch_box(patch, origin, origin)
    assert faulty[0, 0] > 0.0


class _IndividualTrigonometricIntervals:
    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        cosine = interval_cos((lower, upper))
        sine = interval_sin((lower, upper))
        return (
            np.stack((cosine[0], sine[0]), axis=1),
            np.stack((cosine[1], sine[1]), axis=1),
        )


def test_exact_cylinder_randomized_individual_operation_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = CylinderPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        float(Fraction(1, 3)),
        radius_exact=Fraction(1, 3),
    )
    rng = np.random.default_rng(5428)
    lower = rng.uniform(-4.0, 4.0, size=(32, 2))
    upper = lower + rng.uniform(0.0, 0.75, size=(32, 2))
    original = _IndividualTrigonometricIntervals()
    program = roots._cylinder_trigonometric_program()
    expected_trig = original.evaluate(lower[:, :1], upper[:, :1])
    actual_trig = program.evaluate(lower[:, :1], upper[:, :1])
    for expected, actual in zip(expected_trig, actual_trig, strict=True):
        np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))
    partitions = [
        program.evaluate(lower[a:b, :1], upper[a:b, :1])
        for a, b in ((0, 5), (5, 21), (21, 32))
    ]
    for endpoint in range(2):
        split = np.concatenate([part[endpoint] for part in partitions])
        np.testing.assert_array_equal(
            actual_trig[endpoint].view(np.uint64), split.view(np.uint64)
        )
    with monkeypatch.context() as host:
        host.setattr(roots, "_cylinder_trigonometric_program", lambda: original)
        expected_boxes = np.stack(
            [
                roots._patch_box(
                    patch,
                    np.asarray(lo, dtype=np.float64),
                    np.asarray(hi, dtype=np.float64),
                )
                for lo, hi in zip(lower, upper, strict=True)
            ]
        )
    boxes = np.stack(
        [
            roots._patch_box(
                patch, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
            )
            for lo, hi in zip(lower, upper, strict=True)
        ]
    )
    np.testing.assert_array_equal(boxes.view(np.uint64), expected_boxes.view(np.uint64))


def test_inward_trigonometric_fault_is_visible_at_cylinder_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = CylinderPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        float(Fraction(1, 3)),
        radius_exact=Fraction(1, 3),
    )
    origin = np.zeros(2, dtype=np.float64)
    correct = roots._patch_box(patch, origin, origin)
    assert Fraction.from_float(float(correct[0, 0])) <= Fraction(1, 3)
    original = PreparedIntervalFunction.evaluate

    def inward(
        self: PreparedIntervalFunction, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = original(self, lower, upper)
        lo, hi = lo.copy(), hi.copy()
        lo[:, 0, :] = np.nextafter(lo[:, 0, :], np.inf)
        hi[:, 0, :] = np.nextafter(hi[:, 0, :], np.inf)
        return lo, hi

    monkeypatch.setattr(PreparedIntervalFunction, "evaluate", inward)
    faulty = roots._patch_box(patch, origin, origin)
    assert Fraction.from_float(float(faulty[0, 0])) > Fraction(1, 3)
