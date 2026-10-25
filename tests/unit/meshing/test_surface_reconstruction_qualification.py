#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Evidence-derived completion of the curved-surface and reconstruction scenarios."""

from __future__ import annotations

import copy
from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.reconstruction._robustness import remove_statistical_outliers
from phydrax.typing import parse
from tools.meshing_qualification._controls import QualificationControls, ReconstructionCase
from tools.meshing_qualification.scenarios.envelope import _envelope_qualification_limits
from tools.meshing_qualification.scenarios.reconstruction import (
    _coverage_distance,
    _CENTER,
    _MAXIMUM_DISCARDED_GENUINE_FRACTION,
    _NEIGHBORHOOD_SIZE,
    _OUTLIER_OFFSETS,
    _OUTLIER_STD_RATIO,
    _RECONSTRUCTION_SOURCES,
    _reconstruction_qualification_domain,
    _reconstruction_qualification_request,
    _reconstruction_qualification_samples,
    _spacing_ratio,
    _statistical_outliers,
    _supported_band_areas,
    _surface_topology,
    _vertex_components,
    reconstruction_research_gate,
)
from tools.meshing_qualification.scenarios.surface import (
    _can_rim_distance,
    _COLD_SEED_ORIGIN,
    _combined_feature_request,
    _surface_qualification_distance,
    _surface_qualification_material_values,
    _surface_qualification_obligations,
    _surface_qualification_prepare,
    _surface_qualification_state_inventory,
)
from tests._support.meshing_fixtures import box_plc


_FIELDS = ["u", "density", "history", "curl", "flux"]
# Sample noise is isotropic N(0, 8e-4) per coordinate; six standard deviations
# per coordinate bound the displacement norm of every generated genuine sample.
_NOISE_BOUND = 6.0 * np.sqrt(3.0) * 8.0e-4


def _compiler() -> dict[str, object]:
    return {
        "temporary_bytes": 1,
        "output_bytes": 1,
        "generated_code_bytes": 1,
        "warm_samples_seconds": [0.1],
    }


_SPHERE_LAW = {
    "name": "sphere",
    "area": float(4.0 * np.pi),
    "curvature": 1.0,
    "closed": True,
    "euler": 2,
}


def _source_acceptance() -> dict[str, Any]:
    return {
        "passed": True,
        "fidelity_status": "certified",
        "global_embedding_status": "certified",
        "complete_source_associations": True,
        "certified_mesh_to_source_upper": 0.01,
        "fidelity_tolerance": 0.02,
        "independent_analytic_source_distance": 0.008,
        "analytic_distance_roundoff": 1.0e-14,
        "physical_area": float(4.0 * np.pi) - 0.05,
        "euler_characteristic": 2,
        "feature_controls": None,
    }


def _surface_record(rounds: int = 1) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = [
        {
            "source_acceptance": _source_acceptance(),
            "physical_error": 0.01,
            "compiler": _compiler(),
            "retained_bytes": 10,
        }
        for _ in range(rounds + 1)
    ]
    for epoch in accepted[:-1]:
        epoch["state_transition"] = {
            "receipt_id": "receipt",
            "field_names": list(_FIELDS),
            "field_transfers": [{"passed": True} for _ in _FIELDS],
            "material_transport": {"passed": True},
            "rollback": {"passed": True},
        }
    cold = {
        "seed_origin": _COLD_SEED_ORIGIN,
        "l2_error": 0.01,
        "target_error_met": True,
        "compiler": _compiler(),
        "retained_bytes": 10,
    }
    return {
        "profile_law": dict(_SPHERE_LAW),
        # Area bound 4 pi * 0.25 * 0.4^2 ~ 0.50 admits the 0.05 area defect.
        "qualification_scale": {"requested_target_size": 0.4},
        "epochs": [*accepted, cold],
        "checkpoint": {
            "archives": [{} for _ in accepted],
            "cold_restart": {"renewed_epochs": [{} for _ in accepted]},
        },
    }


def _surface_args(rounds: int = 1, **controls: Any) -> QualificationControls:
    return QualificationControls(
        scenario="native-surface-pde",
        adaptation_rounds=rounds,
        target_error=0.05,
        timeout=120.0,
        **controls,
    )


def test_surface_obligations_complete_only_with_all_emitted_evidence() -> None:
    obligations = _surface_qualification_obligations(_surface_record(), _surface_args(), 1.0)

    assert all(obligations.values())


def _unembedded(record: dict[str, Any]) -> None:
    record["epochs"][1]["source_acceptance"]["global_embedding_status"] = "unresolved"


def _uncertified_fidelity(record: dict[str, Any]) -> None:
    record["epochs"][0]["source_acceptance"]["fidelity_status"] = "sampled"


def _certified_bound_above_tolerance(record: dict[str, Any]) -> None:
    record["epochs"][0]["source_acceptance"]["certified_mesh_to_source_upper"] = 0.03


def _sampled_distance_exceeds_certificate(record: dict[str, Any]) -> None:
    # A sampled distance alone cannot carry fidelity: one beyond the certified
    # bound falsifies the certificate even though it is under the tolerance.
    record["epochs"][0]["source_acceptance"]["independent_analytic_source_distance"] = 0.015


def _area_defect(record: dict[str, Any]) -> None:
    record["epochs"][0]["source_acceptance"]["physical_area"] = float(4.0 * np.pi) - 0.6


def _wrong_topology(record: dict[str, Any]) -> None:
    record["epochs"][1]["source_acceptance"]["euler_characteristic"] = 0


def _rollback_failed(record: dict[str, Any]) -> None:
    record["epochs"][0]["state_transition"]["rollback"]["passed"] = False


def _field_skipped(record: dict[str, Any]) -> None:
    record["epochs"][0]["state_transition"]["field_names"] = _FIELDS[:-1]


def _material_failed(record: dict[str, Any]) -> None:
    record["epochs"][0]["state_transition"]["material_transport"]["passed"] = False


def _final_error_missed(record: dict[str, Any]) -> None:
    record["epochs"][1]["physical_error"] = 0.1


def _epoch_not_renewed(record: dict[str, Any]) -> None:
    record["checkpoint"]["cold_restart"]["renewed_epochs"].pop()


def _cold_target_missed(record: dict[str, Any]) -> None:
    record["epochs"][-1]["target_error_met"] = False


def _retained_bytes_missing(record: dict[str, Any]) -> None:
    record["epochs"][-1]["retained_bytes"] = None


def _compiler_unmeasured(record: dict[str, Any]) -> None:
    record["epochs"][0]["compiler"]["temporary_bytes"] = None


@pytest.mark.parametrize(
    ("mutate", "obligation"),
    [
        pytest.param(
            _unembedded,
            "original_source_fidelity_and_global_embedding",
            id="embedding",
        ),
        pytest.param(
            _uncertified_fidelity,
            "original_source_fidelity_and_global_embedding",
            id="uncertified-fidelity",
        ),
        pytest.param(
            _certified_bound_above_tolerance,
            "original_source_fidelity_and_global_embedding",
            id="certified-above-tolerance",
        ),
        pytest.param(
            _sampled_distance_exceeds_certificate,
            "original_source_fidelity_and_global_embedding",
            id="sampled-exceeds-certificate",
        ),
        pytest.param(
            _area_defect, "original_source_fidelity_and_global_embedding", id="area"
        ),
        pytest.param(
            _wrong_topology,
            "original_source_fidelity_and_global_embedding",
            id="topology",
        ),
        pytest.param(
            _rollback_failed,
            "adaptation_all_field_material_transfer_and_rollback",
            id="rollback",
        ),
        pytest.param(
            _field_skipped,
            "adaptation_all_field_material_transfer_and_rollback",
            id="skipped-field",
        ),
        pytest.param(
            _material_failed,
            "adaptation_all_field_material_transfer_and_rollback",
            id="material",
        ),
        pytest.param(_final_error_missed, "independent_manufactured_pde", id="pde"),
        pytest.param(
            _epoch_not_renewed, "checkpoint_cold_restart_continuation", id="renewal"
        ),
        pytest.param(
            _cold_target_missed, "checkpoint_cold_restart_continuation", id="cold"
        ),
        pytest.param(
            _retained_bytes_missing,
            "runtime_phase_and_deadline_evidence",
            id="retained-bytes",
        ),
        pytest.param(
            _compiler_unmeasured,
            "runtime_phase_and_deadline_evidence",
            id="compiler",
        ),
    ],
)
def test_surface_obligation_fails_when_its_evidence_is_lost(
    mutate: Callable[[dict[str, Any]], None], obligation: str
) -> None:
    record = _surface_record()
    mutate(record)

    obligations = _surface_qualification_obligations(record, _surface_args(), 1.0)

    assert not obligations[obligation]
    assert all(value for name, value in obligations.items() if name != obligation)


def test_surface_obligations_refuse_missing_adaptation_and_deadline() -> None:
    without_adaptation = _surface_qualification_obligations(
        _surface_record(rounds=0), _surface_args(rounds=0), 1.0
    )
    late = _surface_qualification_obligations(_surface_record(), _surface_args(), 121.0)

    assert not without_adaptation["adaptation_all_field_material_transfer_and_rollback"]
    assert not late["runtime_phase_and_deadline_evidence"]


_COMBINED_RESOLUTION = 8


def _feature_controls() -> dict[str, Any]:
    request = _combined_feature_request(_COMBINED_RESOLUTION)
    size, deviation = request["target_size"], request["maximum_deviation"]
    return {
        "target_size": size,
        "maximum_size": request["maximum_size"],
        "relative_tolerance": request["relative_tolerance"],
        "absolute_tolerance": 0.0,
        "surface_deviation_request": deviation,
        "curve_deviation_request": deviation,
        "measured_edges": 900,
        "measured_maximum_edge": 1.4 * size,
        "measured_p50_edge": 0.9 * size,
        "rim_curve_edges": 64,
        "sampled_surface_deviation": 0.5 * deviation,
        "sampled_rim_curve_deviation": 0.6 * deviation,
        "certified_surface_deviation": 0.7 * deviation,
        "certified_curve_deviation": 0.8 * deviation,
        "distance_roundoff": 1.0e-14,
    }


def _combined_record() -> dict[str, Any]:
    record = _surface_record()
    request = _combined_feature_request(_COMBINED_RESOLUTION)
    record["profile_law"] = {
        "name": "capped_cylinder",
        "area": 2.0 * np.pi * 0.8 * 1.5 + 2.0 * np.pi * 0.8**2,
        "curvature": 1.0 / 0.8,
        "closed": True,
        "euler": 2,
    }
    record["qualification_scale"]["requested_target_size"] = request["target_size"]
    for epoch in record["epochs"][:-1]:
        epoch["source_acceptance"]["physical_area"] = record["profile_law"]["area"]
    record["epochs"][0]["source_acceptance"]["feature_controls"] = _feature_controls()
    return record


def _combined_args() -> QualificationControls:
    return _surface_args(
        surface_case="capped_cylinder_features", resolution=_COMBINED_RESOLUTION
    )


def test_combined_feature_obligation_recomputes_measured_controls() -> None:
    obligations = _surface_qualification_obligations(
        _combined_record(), _combined_args(), 1.0
    )

    assert obligations["combined_size_and_feature_controls_measured"]
    assert all(obligations.values())
    assert "combined_size_and_feature_controls_measured" not in (
        _surface_qualification_obligations(_surface_record(), _surface_args(), 1.0)
    )


@pytest.mark.parametrize(
    ("name", "scale"),
    [
        # The hard maximum is 1.5 h itself, not 1.5 h widened by the 0.5 tolerance.
        pytest.param("measured_maximum_edge", 1.6, id="hard-maximum"),
        pytest.param("measured_p50_edge", 1.6, id="p50"),
        pytest.param("certified_surface_deviation", 1.1, id="surface-deviation"),
        pytest.param("certified_curve_deviation", 1.1, id="curve-deviation"),
        # Under the request bound but above the certified bound it must not exceed.
        pytest.param("sampled_rim_curve_deviation", 0.9, id="rim-beyond-certificate"),
        pytest.param("sampled_surface_deviation", 0.8, id="surface-beyond-certificate"),
        pytest.param("curve_deviation_request", 2.0, id="loosened-request"),
        pytest.param("rim_curve_edges", 0, id="no-rim"),
    ],
)
def test_combined_feature_obligation_refuses_unmet_measurements(
    name: str, scale: float
) -> None:
    record = _combined_record()
    features = record["epochs"][0]["source_acceptance"]["feature_controls"]
    request = _combined_feature_request(_COMBINED_RESOLUTION)
    base = {
        "measured_maximum_edge": request["target_size"],
        "measured_p50_edge": request["target_size"],
        "certified_surface_deviation": request["maximum_deviation"],
        "certified_curve_deviation": request["maximum_deviation"],
        "sampled_rim_curve_deviation": request["maximum_deviation"],
        "sampled_surface_deviation": request["maximum_deviation"],
        "curve_deviation_request": request["maximum_deviation"],
        "rim_curve_edges": 1,
    }[name]
    features[name] = type(features[name])(base * scale)

    obligations = _surface_qualification_obligations(record, _combined_args(), 1.0)

    assert not obligations["combined_size_and_feature_controls_measured"]


def test_combined_feature_obligation_rederives_request_from_resolution() -> None:
    # Evidence measured against the r=4 request cannot close the r=8 profile.
    record = _combined_record()
    routine = _combined_feature_request(4)
    features = record["epochs"][0]["source_acceptance"]["feature_controls"]
    features.update(
        target_size=routine["target_size"],
        maximum_size=routine["maximum_size"],
        surface_deviation_request=routine["maximum_deviation"],
        curve_deviation_request=routine["maximum_deviation"],
        measured_p50_edge=routine["target_size"],
    )

    obligations = _surface_qualification_obligations(record, _combined_args(), 1.0)

    assert not obligations["combined_size_and_feature_controls_measured"]


def test_can_rim_distance_measures_chord_sagitta() -> None:
    angle = 0.3
    ends = 0.8 * np.asarray(((1.0, 0.0), (np.cos(angle), np.sin(angle))))
    midpoint = np.append(np.mean(ends, axis=0), 1.5)
    rim = np.asarray(((0.8, 0.0, 0.0), (0.0, 0.8, 1.5)))

    distances = _can_rim_distance(np.vstack((rim, midpoint[None, :])))

    np.testing.assert_allclose(distances[:2], 0.0, atol=1.0e-15)
    assert distances[2] == pytest.approx(0.8 * (1.0 - np.cos(angle / 2.0)), rel=1.0e-12)


def test_combined_surface_profile_hardens_controls_on_the_frozen_source() -> None:
    routine = _surface_qualification_prepare(
        QualificationControls(scenario="native-surface-pde", surface_case="capped_cylinder")
    )
    source, request, _, law = _surface_qualification_prepare(
        QualificationControls(
            scenario="native-surface-pde",
            surface_case="capped_cylinder_features",
            resolution=8,
        )
    )
    size = 0.3 * 4 / 8
    control = request.size_controls[0]

    assert routine[1].size_controls[0].target_size == 0.3
    assert (source.source_id, source.source_revision) == (
        routine[0].source_id,
        routine[0].source_revision,
    )
    assert law == routine[3] and law["name"] == "capped_cylinder"
    assert control.strength is phx.meshing.SizeControlStrength.HARD
    assert control.target_size == size and control.maximum_size == 1.5 * size
    assert request.size_compliance is not None
    assert {feature.feature_kind for feature in request.protected_features} == {
        phx.meshing.FeatureKind.SURFACE,
        phx.meshing.FeatureKind.CURVE,
    }
    assert all(
        feature.maximum_deviation == 0.25 * size * size
        for feature in request.protected_features
    )


@pytest.mark.parametrize(
    ("name", "on_surface", "off_surface", "distance"),
    [
        pytest.param("sphere", (0.0, 1.0, 0.0), (0.0, 1.25, 0.0), 0.25, id="sphere"),
        pytest.param("torus", (2.6, 0.0, 0.0), (2.0, 0.0, 0.9), 0.3, id="torus"),
        pytest.param(
            "capped_cylinder", (0.8, 0.0, 0.7), (0.3, 0.0, 1.7), 0.2, id="can"
        ),
        pytest.param("annulus", (0.7, 0.0, 0.0), (0.7, 0.0, -0.1), 0.1, id="annulus"),
        pytest.param(
            "cylinder_sheet", (0.0, 1.0, 0.5), (0.0, 1.5, 0.5), 0.5, id="sheet"
        ),
        pytest.param(
            "folded_plates", (0.0, 0.5, 0.5), (-0.5, 0.5, 0.2), 0.2, id="plates"
        ),
    ],
)
def test_surface_distance_oracle_is_closed_form_source_distance(
    name: str,
    on_surface: tuple[float, float, float],
    off_surface: tuple[float, float, float],
    distance: float,
) -> None:
    oracle = _surface_qualification_distance(name)
    values = oracle(np.asarray((on_surface, off_surface), dtype=np.float64))

    assert values[0] == pytest.approx(0.0, abs=1.0e-15)
    assert values[1] == pytest.approx(distance, rel=1.0e-12)
    assert _surface_qualification_distance(name) is oracle


@pytest.mark.parametrize(("name", "size"), [("sphere", 0.4), ("folded_plates", 1.0)])
def test_surface_checkpoint_retains_actual_nonconstant_fields_and_material(
    name: str, size: float,
) -> None:
    from phydrax.lifecycle import MeshingAcceptedEpoch, validate_meshing_accepted_epochs
    from tests._support.coordinate_references import coordinate_reference_budget
    from tests._support.meshing_fixtures import surface_mesh, surface_specification

    original = getattr(phx.meshing.qualification, f"{name}_surface")()
    request = surface_specification(original.domain, size)
    result = surface_mesh(original.domain, request)
    with coordinate_reference_budget(request.limits) as budget:
        owner = phx.meshing.qualification.prepare_surface_fields(result, budget=budget)
        # These are actual accepted coefficients, deliberately not the authored
        # initial density/history/compatible seeds or a freshly solved scalar.
        fields = tuple(
            jnp.asarray(0.125 + np.arange(space.layout.size, dtype=np.float64) / 32.0)
            for space in owner.field_spaces
        )
        committed = (
            jnp.arange(result.mesh.entity_set(2).count, dtype=jnp.float64)[:, None] + 3.0
        )
        trial = committed + 0.25
        material = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    phx.equations.MaterialSiteId("surface-history"),
                    "surface-history",
                    committed,
                    trial=trial,
                    state_version=7,
                ),
            )
        )
        carried, density, norms, checkpoint = _surface_qualification_state_inventory(
            result, fields, material, epoch=0, error=0.125, budget=budget
        )
        assert carried is material
        assert density > 0.0 and all(value > 0.0 for value in norms.values())
        assert checkpoint.material_identity == {
            "state_version": 7,
            "transaction_id": material.transaction_id,
        }
        for index, name in enumerate(_FIELDS):
            assert checkpoint.fields[f"surface:{name}"] is fields[index]
        state = material.state("surface-history")
        assert checkpoint.fields["surface:material"] is state.committed
        assert checkpoint.fields["surface:material-trial"] is state.trial
        accepted = MeshingAcceptedEpoch(
            result, (checkpoint.declaration,), checkpoint.fields
        )
        records = {
            "accepted_target": result,
            "accepted_data": {"accepted_epochs": (accepted,)},
        }
        assert validate_meshing_accepted_epochs(records) == (accepted,)
        assert set(records["accepted_data"]) == {"accepted_epochs"}


def test_surface_fv_remap_preserves_distinct_accepted_committed_and_trial_banks() -> None:
    from tests._support.coordinate_references import coordinate_reference_budget
    from tests._support.mapped_cells import curved_surface_chart_fv

    source, target, chart, _ = curved_surface_chart_fv()
    with coordinate_reference_budget() as budget:
        remap = phx.discretization.prepare_unstructured_conservative_remap(
            source, target,
            provenance="accepted-surface-material",
            source_geometry=chart.source_geometry,
            target_geometry=chart.target_geometry,
            surface_chart_deformation=chart,
            budget=budget,
        )
        assert remap.succeeded and remap.plan is not None
        committed = jnp.arange(source.cell_count, dtype=jnp.float64)[:, None] + 1.25
        trial = committed + 0.375
        moved, committed_evidence = _surface_qualification_material_values(remap, committed)
        moved_trial, trial_evidence = _surface_qualification_material_values(remap, trial)
        np.testing.assert_array_equal(moved, remap.plan.apply(committed))
        np.testing.assert_array_equal(moved_trial, remap.plan.apply(trial))
        assert committed_evidence["passed"] is True
        assert trial_evidence["passed"] is True
        assert not np.array_equal(moved, moved_trial)


_CASES = tuple(_RECONSTRUCTION_SOURCES)


@pytest.mark.parametrize("case", _CASES)
def test_defective_samples_carry_every_declared_defect(case: str) -> None:
    samples = _reconstruction_qualification_samples(case, 4)
    again = _reconstruction_qualification_samples(case, 4)
    points = samples.points
    genuine = points[: samples.genuine_count] - _CENTER
    outliers = points[samples.genuine_count :] - _CENTER
    genuine_radius = np.linalg.norm(genuine, axis=1)
    outlier_radius = np.linalg.norm(outliers, axis=1)
    clean = samples.genuine_count + samples.removed_cap_samples + samples.removed_band_samples

    assert samples.case == case
    assert np.array_equal(points, again.points)
    assert outliers.shape[0] == max(8, clean // 32)
    assert samples.removed_cap_samples > 0 and samples.removed_band_samples > 0
    assert samples.thinned.shape == (samples.genuine_count,)
    assert np.any(samples.thinned) and np.any(~samples.thinned)
    assert np.max(genuine_radius) <= samples.bounding_radius + _NOISE_BOUND
    assert np.all(outlier_radius >= samples.bounding_radius + _OUTLIER_OFFSETS[0])
    assert np.all(outlier_radius <= samples.bounding_radius + _OUTLIER_OFFSETS[1])


def test_registered_sources_are_geometrically_distinct() -> None:
    lobed = _reconstruction_qualification_samples("lobed-sphere", 4)
    radius = np.linalg.norm(lobed.points[: lobed.genuine_count] - _CENTER, axis=1)
    torus = _reconstruction_qualification_samples("torus", 4)
    relative = torus.points[: torus.genuine_count] - _CENTER
    tube = np.hypot(np.hypot(relative[:, 0], relative[:, 1]) - 0.32, relative[:, 2])

    assert np.all(np.abs(radius - 0.2) <= 0.2 * 0.04 + _NOISE_BOUND)
    assert np.all(np.abs(tube - 0.17) <= _NOISE_BOUND)
    # The torus hole is empty of genuine samples: no sample on the axis.
    assert np.min(np.hypot(relative[:, 0], relative[:, 1])) >= 0.32 - 0.17 - _NOISE_BOUND


@pytest.mark.parametrize("case", _CASES)
def test_independent_outlier_rule_matches_native_policy_on_defective_samples(
    case: str,
) -> None:
    samples = _reconstruction_qualification_samples(case, 4)

    removed, threshold = _statistical_outliers(
        samples.points, _NEIGHBORHOOD_SIZE, _OUTLIER_STD_RATIO
    )
    retained, evidence = remove_statistical_outliers(
        samples.points, _NEIGHBORHOOD_SIZE, _OUTLIER_STD_RATIO
    )

    assert np.array_equal(removed, np.flatnonzero(~retained))
    assert threshold == pytest.approx(evidence.threshold, rel=1.0e-12)
    # Generator diagnostics (truth labels): the instance carries the declared
    # outlier defect. Acceptance itself never reads these labels.
    injected = np.arange(samples.genuine_count, samples.points.shape[0])
    assert np.all(np.isin(injected, removed))
    discarded = np.count_nonzero(removed < samples.genuine_count)
    assert discarded <= _MAXIMUM_DISCARDED_GENUINE_FRACTION * samples.genuine_count


def test_vertex_components_count_only_the_selected_triangles() -> None:
    faces = np.asarray(((0, 1, 2), (2, 3, 4), (5, 6, 7)), dtype=np.int64)

    assert _vertex_components(faces, 9) == 2
    assert _vertex_components(faces[:2], 9) == 1


_TETRAHEDRON = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)), dtype=np.int64)


def test_surface_topology_is_counted_from_the_repaired_triangles() -> None:
    closed = _surface_topology(_TETRAHEDRON)
    flipped = _surface_topology(np.vstack((_TETRAHEDRON[:3], _TETRAHEDRON[3, ::-1])))
    opened = _surface_topology(_TETRAHEDRON[:3])

    assert closed == {
        "vertices": 4,
        "edges": 6,
        "faces": 4,
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "repeated_directed_edges": 0,
        "components": 1,
        "euler_characteristic": 2,
    }
    assert flipped["repeated_directed_edges"] == 3 and flipped["boundary_edges"] == 0
    assert opened["boundary_edges"] == 3 and opened["euler_characteristic"] == 1


def test_spacing_ratio_of_a_twofold_density_drop_is_sqrt_two() -> None:
    assert _spacing_ratio(50, 300, 1.0, 3.0) == pytest.approx(np.sqrt(2.0), rel=1.0e-15)
    assert _spacing_ratio(0, 300, 1.0, 3.0) == 0.0


def _volume_acceptance() -> dict[str, object]:
    return {
        "embedding_status": "certified",
        "coverage_status": "certified",
        "source_boundary_volume": 0.25,
        "mesh_volume": 0.25 + 1.0e-14,
        "minimum_cell_volume": 1.0e-6,
    }


def _gate_record() -> dict[str, Any]:
    domain = _reconstruction_qualification_domain(4)
    coverage = domain["repair_policy"]["coverage_distance"]
    return {
        "status": "passed",
        "reconstruction_case": "torus",
        "acceptance_domain": domain,
        "reconstruction": {"input_points": 1000},
        "independent_repair": {
            "case": "torus",
            "outliers": {
                "removed": 40,
                "independent_removed": 40,
                "removed_digest": "d",
                "independent_removed_digest": "d",
                "threshold": 0.05,
                "independent_threshold": 0.05,
                "retained": 960,
                "removed_within_coverage": 6,
                "removed_beyond_coverage": 34,
            },
            # A genus-one repair: V - E + F = 600 - 1800 + 1200 = 0.
            "repaired_topology": {
                "vertices": 600,
                "edges": 1800,
                "faces": 1200,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "repeated_directed_edges": 0,
                "components": 1,
                "euler_characteristic": 0,
            },
            "supported_fidelity": {
                "retained_sample_to_surface_max": 0.5 * coverage,
                "coverage_distance": coverage,
            },
            "repair_region": {
                "unsupported_triangles": 12,
                "unsupported_patches": 1,
                "unsupported_area": 0.1,
                "surface_area": 2.0,
                "retained_bounding_diagonal": 1.0,
                "excursion_from_data": 0.2,
            },
            # Half the regular density in the band: spacing ratio sqrt(2).
            "density_variation": {
                "band_samples": 60,
                "regular_samples": 900,
                "band_supported_area": 0.2,
                "regular_supported_area": 1.5,
            },
            "generator_diagnostics": {"injected_outliers": 34, "injected_removed": 34},
        },
        "independent_acceptance": _volume_acceptance(),
        "epochs": [
            {
                "l2_error": 0.02,
                "transfer": {"transfer_id": "t"},
                "target_acceptance": _volume_acceptance(),
            },
            {"l2_error": 0.01},
        ],
        "checkpoint": {
            "status": "passed",
            "accepted_fields_bitwise_preserved": True,
            "field_names": ["u", "density", "history"],
            "source_authority_stripped": False,
            "source_acceptance": _volume_acceptance(),
            "continued_physical_L2_error": 0.01,
            "density_max_defect": 0.0,
            "history_max_defect": 1.0e-16,
        },
        "end_to_end_seconds": 10.0,
    }


def _gate_args(case: str = "torus") -> QualificationControls:
    return QualificationControls(
        scenario="native-reconstruction-lifecycle",
        reconstruction_case=parse(case, ReconstructionCase, "reconstruction_case"),
        adaptation_rounds=1,
        target_error=0.05,
        timeout=120.0,
    )


def test_reconstruction_gate_closes_only_on_complete_numeric_evidence() -> None:
    gate = reconstruction_research_gate(_gate_record(), _gate_args())

    assert gate["status"] == "complete"
    assert all(gate["obligations"].values())
    assert gate["domain"] == "underspecified-defective-sampled-closed-surface"
    assert gate["case"] == "torus"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "unreached"),
        ("accepted_fields_bitwise_preserved", False),
        ("field_names", ["u"]),
        ("source_authority_stripped", True),
        ("source_acceptance", {}),
        ("continued_physical_L2_error", 0.051),
        ("continued_physical_L2_error", float("nan")),
        ("density_max_defect", 1.0e-11),
        ("history_max_defect", 1.0e-11),
    ],
)
def test_reconstruction_gate_requires_independent_cold_allfield_continuation(
    key: str, value: object,
) -> None:
    record = _gate_record()
    record["checkpoint"][key] = value
    gate = reconstruction_research_gate(record, _gate_args())
    assert gate["status"] == "incomplete"
    obligations = gate["obligations"]
    assert not obligations["independent_cold_allfield_continuation"]


def test_reconstruction_gate_never_reads_generator_truth() -> None:
    record = _gate_record()
    record["independent_repair"]["generator_diagnostics"] = {
        "injected_outliers": 34,
        "injected_removed": 0,
        "genuine_discarded": 10**6,
    }

    assert reconstruction_research_gate(record, _gate_args())["status"] == "complete"


@pytest.mark.parametrize("case", _CASES)
def test_reconstruction_gate_applies_one_preregistered_domain_to_every_case(
    case: str,
) -> None:
    record = _gate_record()
    record["reconstruction_case"] = case
    record["independent_repair"]["case"] = case
    other = next(name for name in _CASES if name != case)

    assert reconstruction_research_gate(record, _gate_args(case))["status"] == "complete"
    assert reconstruction_research_gate(record, _gate_args(other))["status"] == "incomplete"


def test_reconstruction_gate_rederives_bounds_instead_of_trusting_the_record() -> None:
    record = _gate_record()
    record["acceptance_domain"]["preregistered_bounds"][
        "maximum_repair_excursion_fraction"
    ] = 1.0
    record["independent_repair"]["repair_region"]["excursion_from_data"] = 0.5

    gate = reconstruction_research_gate(record, _gate_args())

    assert not gate["obligations"]["preregistered_domain_and_case"]
    assert not gate["obligations"]["native_repair_with_independent_geometry"]


def test_reconstruction_gate_bounds_discarded_supported_samples() -> None:
    record = _gate_record()
    repair = record["independent_repair"]
    # 50 removed samples lie on the repair: 50 > 0.05 * (900 + 50).
    repair["outliers"].update(
        removed=100,
        independent_removed=100,
        retained=900,
        removed_within_coverage=50,
        removed_beyond_coverage=50,
    )
    repair["density_variation"]["regular_samples"] = 840

    assert reconstruction_research_gate(record, _gate_args())["status"] == "incomplete"
    repair["outliers"].update(removed_within_coverage=47, removed_beyond_coverage=53)
    assert reconstruction_research_gate(record, _gate_args())["status"] == "complete"


_REPAIR = "independent_repair"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("status",), "failed", id="status"),
        pytest.param((_REPAIR, "case"), "lobed-sphere", id="case"),
        pytest.param(("reconstruction", "input_points"), 999, id="unbound-input"),
        pytest.param(
            (_REPAIR, "outliers", "independent_removed_digest"), "e", id="rule-set"
        ),
        pytest.param(
            (_REPAIR, "outliers", "independent_threshold"), 0.06, id="rule-threshold"
        ),
        pytest.param((_REPAIR, "outliers", "removed_beyond_coverage"), 0, id="no-outlier"),
        pytest.param((_REPAIR, "repaired_topology", "components"), 2, id="components"),
        pytest.param((_REPAIR, "repaired_topology", "boundary_edges"), 3, id="open"),
        pytest.param(
            (_REPAIR, "repaired_topology", "repeated_directed_edges"), 1, id="winding"
        ),
        pytest.param(
            (_REPAIR, "repaired_topology", "euler_characteristic"), 2, id="euler-copy"
        ),
        pytest.param((_REPAIR, "repaired_topology", "edges"), 1801, id="incidence"),
        pytest.param(
            (_REPAIR, "supported_fidelity", "retained_sample_to_surface_max"),
            1.0,
            id="unrepresented-sample",
        ),
        pytest.param(
            (_REPAIR, "supported_fidelity", "coverage_distance"), 1.0, id="loosened"
        ),
        pytest.param(
            (_REPAIR, "repair_region", "unsupported_triangles"), 0, id="no-repair"
        ),
        pytest.param((_REPAIR, "repair_region", "unsupported_patches"), 2, id="patches"),
        pytest.param(
            (_REPAIR, "repair_region", "excursion_from_data"), 0.5, id="excursion"
        ),
        pytest.param(
            (_REPAIR, "density_variation", "band_supported_area"),
            0.1,
            id="no-density-defect",
        ),
        pytest.param(
            (_REPAIR, "density_variation", "regular_supported_area"),
            1.75,
            id="area-overcount",
        ),
        pytest.param(
            ("independent_acceptance", "mesh_volume"), 0.25 + 1.0e-9, id="volume-closure"
        ),
        pytest.param(
            ("epochs", 0, "target_acceptance", "embedding_status"),
            "unresolved",
            id="adapted-embedding",
        ),
        pytest.param(
            ("epochs", 0, "target_acceptance", "minimum_cell_volume"),
            0.0,
            id="inverted-cell",
        ),
        pytest.param(("epochs", 0, "transfer"), None, id="transfer"),
        pytest.param(("epochs", 1, "l2_error"), 0.1, id="physics"),
        pytest.param(("end_to_end_seconds",), 121.0, id="deadline"),
    ],
)
def test_reconstruction_gate_stays_incomplete_without_its_evidence(
    path: tuple[str | int, ...], value: object
) -> None:
    record = copy.deepcopy(_gate_record())
    target: Any = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert reconstruction_research_gate(record, _gate_args())["status"] == "incomplete"


@pytest.mark.parametrize("boundary", ("fixed", "conforming"))
@pytest.mark.parametrize("rounds", (0, 1))
def test_reconstruction_schedule_prepares_only_requested_fixed_adaptation(
    boundary: Literal["fixed", "conforming"], rounds: int,
) -> None:
    source = phx.meshing.NativePlcSource(
        box_plc(boundary=boundary), "authored-reconstructed-source", "original-r1"
    )
    specification, options = _reconstruction_qualification_request(
        source,
        QualificationControls(
            capacity=20_000, timeout=120.0, adaptation_rounds=rounds
        ),
    )
    schedule = options.volume_schedule
    if schedule is None:
        pytest.fail("Reconstruction request lost its authored native volume schedule.")
    assert schedule.prepare_fixed_boundary_adaptation == (
        boundary == "fixed" and rounds > 0
    )
    assert schedule.refinement_rounds == 1
    assert schedule.improvement_passes == 1
    assert schedule.minimum_dihedral_degrees == 0.0
    assert specification.limits.maximum_work_units == 20_000_000
    assert specification.limits.maximum_cells == 100_000
    assert specification.limits.maximum_wall_seconds == 120.0


def test_reconstruction_qualification_limits_honor_absolute_resource_controls() -> None:
    source = phx.meshing.NativePlcSource(
        box_plc(), "authored-reconstructed-source", "original-r1"
    )
    args = QualificationControls(
        capacity=20_000,
        timeout=120.0,
        maximum_work_units=123_456,
        maximum_geometry_queries=234_567,
        maximum_scratch_bytes=345_678,
    )
    specification, _ = _reconstruction_qualification_request(source, args)

    assert specification.limits.maximum_work_units == 123_456
    assert specification.limits.maximum_geometry_queries == 234_567
    assert specification.limits.maximum_scratch_bytes == 345_678
    assert specification.limits.maximum_cells == 100_000
    assert specification.limits.maximum_wall_seconds == 120.0


def test_envelope_qualification_limits_honor_absolute_resource_controls() -> None:
    args = QualificationControls(
        capacity=20,
        timeout=17.0,
        maximum_work_units=123_456,
        maximum_geometry_queries=234_567,
        maximum_scratch_bytes=345_678,
    )
    limits = _envelope_qualification_limits(args)

    assert limits.maximum_work_units == 123_456
    assert limits.maximum_geometry_queries == 234_567
    assert limits.maximum_scratch_bytes == 345_678
    assert limits.maximum_cells == 100
    assert limits.maximum_wall_seconds == 17.0


def test_reconstruction_sparse_band_coverage_scales_with_sample_spacing() -> None:
    resolution = 12
    domain = _reconstruction_qualification_domain(resolution)
    policy = domain["repair_policy"]

    assert policy["sample_spacing"] == pytest.approx(0.44 / resolution)
    assert _coverage_distance(resolution) == pytest.approx(
        np.sqrt(2.0) * 0.44 / resolution
    )
    assert policy["coverage_distance"] == pytest.approx(
        _coverage_distance(resolution)
    )


@pytest.mark.parametrize(
    ("triangle", "cut", "band", "regular"),
    [
        (((0, 0, 0), (1, 0, 0), (0, 1, 0)), Fraction(1, 2), Fraction(3, 8), Fraction(1, 8)),
        (((0, 0, 0), (1, 0, 0), (0, 1, 0)), Fraction(1, 4), Fraction(7, 32), Fraction(9, 32)),
        (((-1, 0, 0), (1, 0, 0), (1, 1, 0)), Fraction(0), Fraction(1, 4), Fraction(3, 4)),
        (((0, 0, 0), (1, 0, 0), (0, 1, 0)), Fraction(0), Fraction(0), Fraction(1, 2)),
        (((0, 0, 0), (1, 0, 0), (0, 1, 0)), Fraction(1), Fraction(1, 2), Fraction(0)),
        (((0, 0, 0), (0, 1, 0), (0, 0, 1)), Fraction(0), Fraction(0), Fraction(1, 2)),
        (((0, 0, 0), (1, 0, 0), (2, 0, 0)), Fraction(1, 2), Fraction(0), Fraction(0)),
    ],
)
@pytest.mark.parametrize("order", ((0, 1, 2), (1, 2, 0), (2, 1, 0)))
@pytest.mark.parametrize("supported", (False, True))
def test_supported_density_band_integrates_actual_clipped_triangle_area(
    triangle: tuple[tuple[int, int, int], ...], cut: Fraction,
    band: Fraction, regular: Fraction, order: tuple[int, ...], supported: bool,
) -> None:
    points = np.asarray(triangle, dtype=np.float64)[np.asarray(order)]
    area = float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2)
    actual = _supported_band_areas(
        points[None, :], np.asarray((area,)), np.asarray((supported,)), cut
    )
    assert actual == (
        (float(band), float(regular)) if supported else (0.0, 0.0)
    )


def test_density_band_clipping_does_not_invent_density_variation() -> None:
    triangle = np.asarray((((0, 0, 0), (1, 0, 0), (0, 1, 0)),), dtype=np.float64)
    band, regular = _supported_band_areas(
        triangle, np.asarray((0.5,)), np.asarray((True,)), Fraction(1, 2)
    )
    # Six and two samples have equal physical density on areas 3/8 and 1/8.
    assert _spacing_ratio(6, 2, band, regular) == 1.0
    assert _spacing_ratio(6, 2, band, regular) < 1.2
