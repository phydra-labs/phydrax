#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned elegant CSR oracle: deck translation, refusals, SDDS parsing, live run.

The deck is checked against independently computed SI conversions (slopes from
canonical momenta, arrival time ``ζ/(β₀c)``, ``p = (1+δ)β₀γ₀``, total charge)
and the chicane geometry. The parser is checked on a real elegant 2026.3.0
output (``tests/data/providers/elegant``; provenance in its JSON sidecar)
against the exact rectangular-chicane ``R₅₆ = 4ρ(θ − tanθ) − 2d tan²θ − L/γ²`` and closed
dispersion. The live comparison skips only when ``PHYDRAX_ELEGANT`` and
``PHYDRAX_ELEGANT_VERSION`` are absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._external_runtime import pin_executable
from phydrax.applications import accelerator
from phydrax.discretization import PreparedTensorGrid, TensorGridPlan, UniformCellAxisSpec
from phydrax.interchange import AdapterStatus


SCALE = phx.ElectromagneticScaleContract.si()
ELEMENTARY_CHARGE = 1.602176634e-19
LIGHT = 299792458.0
REST_ENERGY = float(SCALE.electron_mass) * LIGHT**2
FIXTURE = Path(__file__).resolve().parents[2] / "data/providers/elegant"
THETA, BEND, DRIFT, GAMMA, SIGMA = 0.05, 0.5, 2.0, 1000.0, 5.0e-5


def _momentum(gamma: float) -> float:
    return REST_ENERGY * math.sqrt(gamma * gamma - 1.0)


def _grid(
    lower: tuple[float, ...], upper: tuple[float, ...], counts: tuple[int, ...]
) -> PreparedTensorGrid:
    return TensorGridPlan(tuple(UniformCellAxisSpec(count) for count in counts)).prepare(
        np.asarray([lower, upper])
    )


def _chicane() -> accelerator.CSRLattice:
    curvature = THETA / BEND
    return accelerator.CSRLattice(
        [BEND, DRIFT, BEND, 0.5, BEND, DRIFT, BEND, 0.5],
        [curvature, 0.0, -curvature, 0.0, -curvature, 0.0, curvature, 0.0],
        element_ids=["b1", "d1", "b2", "d2", "b3", "d3", "b4", "d4"],
        entrance_edges=[0.0, 0.0, -THETA, 0.0, 0.0, 0.0, THETA, 0.0],
        exit_edges=[THETA, 0.0, 0.0, 0.0, -THETA, 0.0, 0.0, 0.0],
    )


def _plan(
    model: accelerator.CSRModel,
    capacity: int,
    substeps: int | list[int] = 2,
    *,
    plate_gap: float | None = None,
) -> accelerator.CSRTrackingPlan:
    transient = model == "1d-transient-shielded"
    plan = accelerator.CSRPlan(
        model,
        _chicane(),
        SCALE,
        _grid((-8 * SIGMA,), (8 * SIGMA,), (64,)),
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(GAMMA),
        capacity=capacity,
        smoothing=SIGMA / 8,
        plate_gap=plate_gap,
        image_count=0 if plate_gap is None else 2,
        history_capacity=128 if transient else 256,
        far_nodes=48 if transient else 96,
    )
    return accelerator.CSRTrackingPlan(plan, substeps=substeps)


def _bunch(
    coordinates: np.ndarray,
    charge: float,
    *,
    identifiers: np.ndarray | None = None,
    active: np.ndarray | None = None,
    reference_charge: float = -1.0,
    rest_energy: float = REST_ENERGY,
    sign: str = "positive-late",
) -> accelerator.AcceleratorBunch:
    count = coordinates.shape[0]
    return accelerator.AcceleratorBunch(
        coordinates,
        np.full((count,), charge / ELEMENTARY_CHARGE / count),
        np.arange(count) if identifiers is None else identifiers,
        active=active,
        reference_rest_energy=rest_energy,
        reference_momentum=rest_energy * math.sqrt(GAMMA * GAMMA - 1.0),
        reference_charge=reference_charge,
        convention=accelerator.AcceleratorConvention(longitudinal_sign=sign),
        bunch_id="elegant-test",
    )


def _lattice_elements(text: str) -> dict[str, tuple[str, dict[str, str]]]:
    elements: dict[str, tuple[str, dict[str, str]]] = {}
    for line in text.splitlines():
        name, _, body = line.partition(":")
        kind, *fields = [part.strip() for part in body.split(",")]
        if kind.startswith("LINE"):
            continue
        elements[name.strip()] = (
            kind,
            dict(field.split("=", 1) for field in fields),
        )
    return elements


def _beam_columns(data: bytes) -> dict[str, np.ndarray]:
    header, _, payload = data.partition(b"&data mode=binary, endian=little, &end\n")
    names = re.findall(rb"&column name=(\w+),[^&]*type=(\w+)", header)
    codes = {b"double": "<f8", b"long": "<i4"}
    dtype = np.dtype([(name.decode(), codes[kind]) for name, kind in names])
    count = int(np.frombuffer(payload, "<i4", 1)[0])
    table = np.frombuffer(payload, dtype, count, 4)
    assert len(payload) == 4 + count * dtype.itemsize
    return {name: np.asarray(table[name]) for name in dtype.names or ()}


@pytest.mark.parametrize("sign", ["positive-late", "positive-early"])
def test_deck_maps_chicane_geometry_charge_and_coordinates_to_si(sign: str) -> None:
    coordinates = np.asarray(
        [
            [1.0e-4, 2.0e-5, -3.0e-5, 1.0e-5, 4.0e-5, 1.0e-3],
            [-2.0e-4, -1.0e-5, 5.0e-5, -2.0e-5, -6.0e-5, -2.0e-3],
            [0.0, 3.0e-3, 0.0, -4.0e-3, 1.0e-5, 5.0e-2],
        ]
    )
    bunch = _bunch(coordinates, 1.0e-9, sign=sign)
    plan = _plan("1d-transient-shielded", 3, [10, 4, 10, 2, 10, 4, 10, 2])
    deck = accelerator.elegant_csr_input(plan, bunch)

    elements = _lattice_elements(deck["csr.lte"].decode())
    assert elements["Q"][0] == "CHARGE"
    assert float(elements["Q"][1]["TOTAL"]) == pytest.approx(1.0e-9, rel=1.0e-12)
    angles = [THETA, 0.0, -THETA, 0.0, -THETA, 0.0, THETA, 0.0]
    lengths = [BEND, DRIFT, BEND, 0.5, BEND, DRIFT, BEND, 0.5]
    entrance = [0.0, 0.0, -THETA, 0.0, 0.0, 0.0, THETA, 0.0]
    exit_ = [THETA, 0.0, 0.0, 0.0, -THETA, 0.0, 0.0, 0.0]
    substeps = [10, 4, 10, 2, 10, 4, 10, 2]
    extent = float(np.ptp(coordinates[:, 4]))
    for index in range(8):
        kind, fields = elements[f"E{index}"]
        assert float(fields["L"]) == pytest.approx(lengths[index], rel=1.0e-15)
        if angles[index] == 0.0:
            assert kind == "CSRDRIFT"
            assert int(fields["N_KICKS"]) == substeps[index]
            assert fields["USE_STUPAKOV"] == "1"
        else:
            assert kind == "CSRCSBEND"
            assert float(fields["ANGLE"]) == pytest.approx(angles[index], rel=1.0e-14)
            assert float(fields["E1"]) == entrance[index]
            assert float(fields["E2"]) == exit_[index]
            assert int(fields["N_SLICES"]) == substeps[index]
            assert fields["STEADY_STATE"] == "0"
            # The initial histogram bin is the plan's cell width 16σ/64.
            bins = int(fields["BINS"])
            assert 1.2 * extent / bins <= 0.25 * SIGMA < 1.2 * extent / (bins - 1)

    columns = _beam_columns(deck["beam.sdds"])
    px, py, delta = coordinates[:, 1], coordinates[:, 3], coordinates[:, 5]
    forward = np.sqrt((1.0 + delta) ** 2 - px**2 - py**2)
    beta = math.sqrt(1.0 - 1.0 / GAMMA**2)
    late = 1.0 if sign == "positive-late" else -1.0
    np.testing.assert_array_equal(columns["x"], coordinates[:, 0])
    np.testing.assert_array_equal(columns["y"], coordinates[:, 2])
    np.testing.assert_allclose(columns["xp"], px / forward, rtol=1.0e-15)
    np.testing.assert_allclose(columns["yp"], py / forward, rtol=1.0e-15)
    np.testing.assert_allclose(
        columns["t"], late * coordinates[:, 4] / (beta * LIGHT), rtol=1.0e-14
    )
    np.testing.assert_allclose(
        columns["p"], (1.0 + delta) * math.sqrt(GAMMA**2 - 1.0), rtol=1.0e-14
    )
    np.testing.assert_array_equal(columns["particleID"], [0, 1, 2])
    run = deck["csr.ele"].decode()
    central = re.search(r"p_central = ([^,]+),", run)
    assert central is not None
    p_central = float(central.group(1))
    assert p_central == pytest.approx(math.sqrt(GAMMA**2 - 1.0), rel=1.0e-14)


def test_steady_deck_turns_off_drift_csr() -> None:
    coordinates = np.zeros((2, 6))
    coordinates[:, 4] = [-SIGMA, SIGMA]
    deck = accelerator.elegant_csr_input(
        _plan("1d-steady", 2), _bunch(coordinates, 1.0e-9)
    )
    elements = _lattice_elements(deck["csr.lte"].decode())
    assert elements["E1"][1]["CSR"] == "0"
    assert elements["E0"][1]["STEADY_STATE"] == "1"


def test_refuses_inputs_outside_the_supported_subset() -> None:
    coordinates = np.zeros((4, 6))
    coordinates[:, 4] = [-2 * SIGMA, -SIGMA, SIGMA, 2 * SIGMA]
    steady = _plan("1d-steady", 4)
    with pytest.raises(ValueError, match="parallel-plate"):
        accelerator.elegant_csr_input(
            _plan("1d-transient-shielded", 4, plate_gap=0.02), _bunch(coordinates, 1.0e-9)
        )
    igf = accelerator.CSRPlan(
        "3d-steady-igf",
        _chicane(),
        SCALE,
        _grid((-1.0e-4, -1.0e-4, -8 * SIGMA), (1.0e-4, 1.0e-4, 8 * SIGMA), (4, 4, 8)),
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(GAMMA),
        capacity=4,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        accelerator.elegant_csr_input(
            accelerator.CSRTrackingPlan(igf, substeps=1), _bunch(coordinates, 1.0e-9)
        )
    with pytest.raises(ValueError, match="electron"):
        accelerator.elegant_csr_input(
            steady, _bunch(coordinates, 1.0e-9, reference_charge=1.0)
        )
    with pytest.raises(ValueError, match="electron"):
        accelerator.elegant_csr_input(
            steady, _bunch(coordinates, 1.0e-9, rest_energy=1836.15 * REST_ENERGY)
        )
    unequal = accelerator.AcceleratorBunch(
        coordinates,
        np.asarray([1.0, 2.0, 1.0, 1.0]),
        np.arange(4),
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(GAMMA),
        reference_charge=-1.0,
        bunch_id="unequal",
    )
    with pytest.raises(ValueError, match="equal charge"):
        accelerator.elegant_csr_input(steady, unequal)
    with pytest.raises(ValueError, match="unique"):
        accelerator.elegant_csr_input(
            steady, _bunch(coordinates, 1.0e-9, identifiers=np.asarray([0, 1, 1, 2]))
        )
    mismatched = accelerator.AcceleratorBunch(
        coordinates,
        np.ones((4,)),
        np.arange(4),
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(2.0 * GAMMA),
        reference_charge=-1.0,
        bunch_id="mismatched",
    )
    with pytest.raises(ValueError, match="does not match"):
        accelerator.elegant_csr_input(steady, mismatched)
    with pytest.raises(ValueError, match="nonzero length"):
        accelerator.elegant_csr_input(steady, _bunch(np.zeros((4, 6)), 1.0e-9))
    with pytest.raises(TypeError, match="PinnedExecutable"):
        accelerator.ElegantCSRProvider("elegant")  # ty: ignore[invalid-argument-type]


def _fixture_case() -> tuple[accelerator.CSRTrackingPlan, accelerator.AcceleratorBunch]:
    """Zero-charge chicane bunch whose elegant output is the stored fixture."""
    count = 49
    rng = np.random.default_rng(7)
    coordinates = 1.0e-5 * rng.standard_normal((count, 6))
    active = np.ones((count,), dtype=bool)
    active[3] = False
    bunch = _bunch(coordinates, 0.0, identifiers=5 + 7 * np.arange(count), active=active)
    return _plan("1d-steady", count), bunch


def test_parser_reads_pinned_elegant_chicane_as_first_order_optics() -> None:
    plan, bunch = _fixture_case()
    provenance = json.loads((FIXTURE / "chicane_zero_charge.json").read_text())
    deck = accelerator.elegant_csr_input(plan, bunch)
    assert hashlib.sha256(deck["beam.sdds"]).hexdigest() == provenance["beam_sha256"]
    data = (FIXTURE / "chicane_zero_charge.out").read_bytes()
    final = accelerator.elegant_csr_bunch(data, plan, bunch)

    active = np.asarray(bunch.active)
    initial = np.asarray(bunch.coordinates)
    tracked = np.asarray(final.coordinates)
    np.testing.assert_array_equal(np.asarray(final.active), active)
    np.testing.assert_array_equal(tracked[~active], initial[~active])
    # elegant's 2-slice symplectic bends add a constant time-of-flight offset
    # (≈ −0.13 µm here), so the first-order map is fitted with an intercept.
    affine = np.concatenate((initial[active], np.ones((int(active.sum()), 1))), axis=1)
    matrix = np.linalg.lstsq(affine, tracked[active], rcond=None)[0].T
    assert abs(matrix[4, 6]) < 1.0e-2 * SIGMA
    total = 4 * BEND + 2 * DRIFT + 1.0
    # Exact rectangular chicane: 4ρ(θ − tanθ) − 2d tan²θ − L/γ² (path lengths of
    # an off-momentum orbit through parallel-faced magnets).
    radius = BEND / THETA
    r56 = (
        4.0 * radius * (THETA - math.tan(THETA))
        - 2.0 * DRIFT * math.tan(THETA) ** 2
        - total / GAMMA**2
    )
    assert matrix[4, 5] == pytest.approx(r56, rel=1.0e-4)
    # Closed dispersion (0.11 m at mid-chicane) up to the second-order terms
    # the linear fit absorbs from 10 µm-scale amplitudes.
    np.testing.assert_allclose(matrix[0:2, 5], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(tracked[active, 5], initial[active, 5], atol=1.0e-15)
    with pytest.raises(ValueError, match="Truncated"):
        accelerator.elegant_csr_bunch(data[:-8], plan, bunch)


def _pinned_elegant() -> accelerator.ElegantCSRProvider | None:
    path = os.environ.get("PHYDRAX_ELEGANT")
    version = os.environ.get("PHYDRAX_ELEGANT_VERSION")
    if path is None or version is None:
        return None
    return accelerator.ElegantCSRProvider(
        pin_executable(path, version=version, license_id="EPICS")
    )


@pytest.mark.parametrize(
    ("model", "spread", "growth", "loss"),
    [("1d-steady", 0.05, 0.15, 0.1), ("1d-transient-shielded", 0.25, 0.5, 0.05)],
)
def test_chicane_csr_matches_pinned_elegant(
    model: accelerator.CSRModel,
    spread: float,
    growth: float,
    loss: float,
    tmp_path: Path,
) -> None:
    """1 nC, σ_z = 50 µm through a θ = 50 mrad chicane with 4000 particles.

    Tolerances: both codes bin 4000 macroparticles at the same initial cell
    width, but elegant histograms (nearest-bin, Savitzky–Golay derivative) at
    1.2× the current extent, which moves its spread and emittance growth by
    ~5 % and ~15 % per bin doubling (steady). The transient model additionally
    replaces the exact retarded integral by Saldin's entrance and Stupakov's
    exit asymptotics, so only the mean loss is held tightly there.
    """
    provider = _pinned_elegant()
    if provider is None:
        pytest.skip("set PHYDRAX_ELEGANT and PHYDRAX_ELEGANT_VERSION to a pinned elegant")
    count = 4000
    rng = np.random.default_rng(1)
    coordinates = np.zeros((count, 6))
    coordinates[:, 0] = math.sqrt(1.0e-9 * 10.0) * rng.standard_normal(count)
    coordinates[:, 1] = math.sqrt(1.0e-9 / 10.0) * rng.standard_normal(count)
    coordinates[:, 2] = 1.0e-4 * rng.standard_normal(count)
    coordinates[:, 3] = 1.0e-5 * rng.standard_normal(count)
    coordinates[:, 4] = SIGMA * rng.standard_normal(count)
    bunch = _bunch(coordinates, 1.0e-9)
    plan = _plan(model, count, [10, 4, 10, 2, 10, 4, 10, 2])
    ours = np.asarray(accelerator.track_csr(plan, bunch).bunch.coordinates)
    result = accelerator.run_elegant_csr(provider, plan, bunch, tmp_path)
    theirs = np.asarray(result.bunch.coordinates)

    def emittance(values: np.ndarray) -> float:
        return float(math.sqrt(max(np.linalg.det(np.cov(values[:, 0:2].T)), 0.0)))

    initial = emittance(coordinates)
    assert np.asarray(result.bunch.active).all()
    assert np.std(ours[:, 5]) == pytest.approx(np.std(theirs[:, 5]), rel=spread)
    assert np.mean(ours[:, 5]) == pytest.approx(np.mean(theirs[:, 5]), rel=loss)
    assert emittance(ours) - initial == pytest.approx(
        emittance(theirs) - initial, rel=growth
    )
    report = result.report
    assert report.status == AdapterStatus.DECLARED_LOSS
    assert report.source_id == result.output_sha256
    assert ("csr.model" in {item.path for item in report.losses}) == (
        model == "1d-transient-shielded"
    )
    assert result.license_id == "EPICS"
    assert result.executable_sha256 == provider.executable.sha256
