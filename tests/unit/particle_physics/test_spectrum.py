#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.interchange.hep import (
    parse_slha,
    serialize_slha,
    spectrum_observables_from_slha,
)
from phydrax.particle_physics import (
    cross_qualify_spectra,
    evaluate_scale_bvp,
    NativeSpectrumStatus,
    ScaleBVPPlan,
    ScaleBVPStatus,
    solve_scale_bvp,
    SpectrumApproximationProfile,
    SpectrumCalculationResult,
    SpectrumDiagnostics,
    SpectrumObservableTable,
    SpectrumStatus,
)


def _profile():
    return SpectrumApproximationProfile(
        model_id="analytic-one-coupling-control",
        renormalization_scheme="MSbar",
        rge_loop_order=1,
        threshold_loop_order=0,
        pole_mass_loop_order=0,
        matching_scale_rule="fixed-endpoints",
        electroweak_scale_rule="not-applicable-control",
        correction_ids=(),
        source_ids=("analytic-exponential-flow",),
    )


def test_spectrum_status_keeps_numerical_physical_and_warning_axes_separate():
    diagnostics = SpectrumDiagnostics(
        SpectrumStatus.SUCCESS,
        provider_available=True,
        finite=True,
        root_found=True,
        electroweak_minimum=True,
        perturbative=True,
        running_tachyons=False,
        pole_tachyons=False,
        approximation_warning=True,
        residual_norm=1e-12,
        warning_ids=("missing-two-loop-control",),
    )
    assert bool(diagnostics.physical_admissible)
    assert not bool(diagnostics.successful)
    result = SpectrumCalculationResult(
        SpectrumObservableTable(("pdg:25",), ("pole-mass",), ("GeV",), (125.1,)),
        (100.0, 1000.0),
        ((0.1,), (0.2,)),
        diagnostics,
        _profile(),
        provider_id="native-control",
        input_artifact_id="input-control",
        output_artifact_id="output-control",
    )
    np.testing.assert_allclose(
        result.observables.value("pdg:25", kind="pole-mass"), 125.1
    )
    assert "approximation" in result.claim


def test_cross_qualification_rejects_noncanonical_provider_keys_without_lookup_failure():
    evidence = cross_qualify_spectra(
        ("mass",),
        np.asarray((1.0,)),
        {" provider-a ": np.asarray((1.0,))},
        relative_tolerance=1.0e-8,
    )

    assert not bool(evidence.accepted)
    assert int(evidence.status) == int(NativeSpectrumStatus.INVALID_INPUT)


def test_strict_slha_roundtrip_preserves_unknown_blocks_comments_and_decays():
    source = (
        b"# spectrum control\n"
        b"BLOCK MASS # pole masses\n"
        b"  25 1.251000000D+02 # h0\n"
        b"  1000022 2.000000000E+02 # neutralino\n"
        b"BLOCK XUNKNOWN Q= 1.000000000E+03 # retained extension\n"
        b"  1 2 3.5 # arbitrary\n"
        b"DECAY 25 4.000000000E-03 # total width\n"
        b"  1.0 2 5 -5 # bb\n"
    )
    document = parse_slha(source)
    assert document.diagnostics.unknown_block_names == ("XUNKNOWN",)
    np.testing.assert_allclose(document.block("MASS").entry(25).value, 125.1)
    assert document.decays[0].channels[0].daughters == (5, -5)
    serialized = serialize_slha(document)
    restored = parse_slha(serialized)
    assert restored.diagnostics.unknown_block_names == ("XUNKNOWN",)
    observables = spectrum_observables_from_slha(restored)
    np.testing.assert_allclose(observables.value("pdg:25", kind="pole-mass"), 125.1)
    np.testing.assert_allclose(observables.value("pdg:25", kind="total-width"), 0.004)
    assert b"retained extension" in serialized


def test_slha_duplicate_entries_are_rejected_without_repinning():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_slha(b"BLOCK MASS\n 25 125.0\n 25 126.0\n")


def test_native_scale_bvp_recovers_analytic_exponential_flow():
    coefficient = 0.2
    lower = 10.0
    upper = 1000.0
    target = 3.0
    plan = ScaleBVPPlan(
        ("g",),
        lambda log_scale, parameters: coefficient * parameters,
        lambda parameters: jnp.zeros((0,), dtype=parameters.dtype),
        lambda parameters: jnp.asarray((parameters[0] - target,)),
        low_residual_count=0,
        lower_scale=lower,
        upper_scale=upper,
        integration_steps=128,
        maximum_newton_steps=8,
        residual_tolerance=1e-11,
        source_ids=("beta-g-equals-0.2g",),
    )
    result = solve_scale_bvp(plan, jnp.asarray((1.0,)))
    expected_initial = target / np.exp(coefficient * np.log(upper / lower))
    assert int(result.status) == int(ScaleBVPStatus.SUCCESS)
    assert bool(result.converged)
    np.testing.assert_allclose(result.final_parameters, (expected_initial,), rtol=1e-10)
    np.testing.assert_allclose(result.trajectory[-1], (target,), rtol=1e-10)
    assert float(jnp.linalg.norm(result.residual)) < 1e-11
    evaluation = evaluate_scale_bvp(plan, result.final_parameters)
    assert bool(evaluation.finite)
    assert "no-model-loop" in result.claim
