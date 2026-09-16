#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib

import numpy as np
import pytest

import phydrax.algebraic._isolated as isolated_module
from phydrax._external_runtime import pin_energy_executable
from phydrax.applications.power._network import (
    Branch,
    Bus,
    BusControl,
    compile_network,
    Generator,
    Load,
    PowerNetwork,
    PowerStudy,
)
from phydrax.applications.power._polynomial_power_flow import (
    compile_fixed_mode_power_flow_polynomial,
    enumerate_prepared_fixed_mode_power_flow_roots,
    prepare_fixed_mode_power_flow_roots,
    refresh_fixed_mode_power_flow_roots,
)
from phydrax.backends.homotopy_continuation import (
    HomotopyContinuationEnvironment,
    HomotopyContinuationExecution,
    HomotopyContinuationExecutionStatus,
    HomotopyContinuationPathRecord,
    HomotopyContinuationPathStatus,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
)


_UUID = "f213a82b-91d6-5c58-9dd6-746a5d3f2e4c"
_COUNT_KEYS = (
    "at_infinity",
    "excess_solution",
    "invalid_endpoint",
    "regular_endpoint",
    "singular_endpoint_candidate",
    "tracking_failed",
)


def _provider(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    project_bytes = b"[deps]\n"
    manifest_bytes = b'manifest_format = "2.0"\n'
    (project / "Project.toml").write_bytes(project_bytes)
    (project / "Manifest.toml").write_bytes(manifest_bytes)
    executable = tmp_path / "julia"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = HomotopyContinuationEnvironment(
        project,
        hashlib.sha256(project_bytes).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
        _UUID,
        "2.15.0",
    )
    return HomotopyContinuationProvider(
        pin_energy_executable(executable, version="1.11.7", license_id="MIT"),
        environment,
    )


def _two_bus():
    network = PowerNetwork(
        (Bus("source", 110), Bus("load", 110)),
        (Branch("line", "source", "load", 0.0, 0.1),),
        (Generator("g", "source"),),
        (Load("d", "load", 0.5, 0.0),),
    )
    study = PowerStudy((BusControl("source", "reference"), BusControl("load", "pq")))
    return compile_network(network, study)


def _three_bus():
    network = PowerNetwork(
        (Bus("r"), Bus("v"), Bus("d")),
        (
            Branch("rv", "r", "v", 0.0, 0.1),
            Branch("vd", "v", "d", 0.0, 0.1),
        ),
        (
            Generator("slack", "r"),
            Generator("pv", "v", p=0.4, q_min=-0.02, q_max=0.02),
        ),
        (Load("load", "d", 0.8, 0.3),),
    )
    study = PowerStudy(
        (
            BusControl("r", "reference"),
            BusControl("v", "pv", voltage=1.03),
            BusControl("d", "pq"),
        )
    )
    return compile_network(network, study)


def _execution(prepared, endpoints):
    paths = tuple(
        HomotopyContinuationPathRecord(
            index,
            "success",
            HomotopyContinuationPathStatus.REGULAR_ENDPOINT,
            tuple(complex(value) for value in endpoint),
            0.0,
            1.0,
        )
        for index, endpoint in enumerate(endpoints)
    )
    counts = {key: 0 for key in _COUNT_KEYS}
    counts["regular_endpoint"] = len(paths)
    isolated = prepared.isolated
    return HomotopyContinuationExecution(
        HomotopyContinuationExecutionStatus.COMPLETE,
        paths,
        tuple((key, counts[key]) for key in _COUNT_KEYS),
        len(paths),
        len(paths),
        isolated.request.request_id,
        isolated.request.support_id,
        isolated.request.system_id,
        isolated.plan.provider.provider_id,
        isolated.plan.policy.policy_id,
        None,
        "",
        "fake-power-flow-execution",
    )


def test_sparse_rectangular_polynomial_matches_native_power_residual():
    compiled = _three_bus()
    voltage = np.asarray([0.98 + 0.07j, 1.01 - 0.04j, 0.91 - 0.11j])
    for modes in (
        ("reference", "pv", "pq"),
        ("reference", "q_max", "pq"),
    ):
        polynomial = compile_fixed_mode_power_flow_polynomial(compiled, modes=modes)
        polynomial_residual = polynomial.system.evaluate(polynomial.coordinates(voltage))
        native_residual = polynomial.native_residual(voltage)

        np.testing.assert_allclose(polynomial_residual, native_residual, atol=2e-6)
        np.testing.assert_allclose(
            native_residual[0],
            voltage[0].real - compiled.initial_voltage[0].real,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            native_residual[3],
            voltage[0].imag - compiled.initial_voltage[0].imag,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            native_residual[2],
            (compiled.bus_power(voltage) - polynomial.injections)[2].real,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            native_residual[5],
            (compiled.bus_power(voltage) - polynomial.injections)[2].imag,
            atol=2e-6,
        )
    pv = compile_fixed_mode_power_flow_polynomial(
        compiled, modes=("reference", "pv", "pq")
    )
    np.testing.assert_allclose(
        pv.native_residual(voltage)[4],
        abs(voltage[1]) ** 2 - compiled.voltage_setpoints[1] ** 2,
        atol=2e-7,
    )
    q_max = compile_fixed_mode_power_flow_polynomial(
        compiled, modes=("reference", "q_max", "pq")
    )
    np.testing.assert_allclose(
        q_max.injections[1].imag + compiled.load_power[1].imag, 0.02, atol=1e-8
    )


def test_two_bus_high_and_low_branches_are_retained_with_separate_physical_masks(
    tmp_path, monkeypatch
):
    compiled = _two_bus()
    polynomial = compile_fixed_mode_power_flow_polynomial(compiled)
    prepared = prepare_fixed_mode_power_flow_roots(
        polynomial,
        _provider(tmp_path),
        policy=HomotopyContinuationPolicy(start_system="total-degree", path_capacity=4),
        polynomial_residual_tolerance=1e-6,
        physical_residual_tolerance=1e-6,
    )
    discriminant = np.sqrt(1 - 4 * 0.05**2)
    high = 0.5 * (1 + discriminant) - 0.05j
    low = 0.5 * (1 - discriminant) - 0.05j
    endpoints = (
        np.asarray([1.0, high.real, 0.0, high.imag]),
        np.asarray([1.0, low.real, 0.0, low.imag]),
    )
    monkeypatch.setattr(
        isolated_module,
        "execute_homotopy_continuation",
        lambda provider, policy, request: _execution(prepared, endpoints),
    )

    roots = enumerate_prepared_fixed_mode_power_flow_roots(prepared)

    assert roots.candidate_mask.tolist() == [True, True, False, False]
    assert roots.near_real_mask.tolist() == [True, True, False, False]
    assert roots.original_residual_mask.tolist() == [True, True, False, False]
    assert roots.mode_complementarity_mask.tolist() == [True, True, False, False]
    assert roots.generator_limit_mask.tolist() == [True, True, False, False]
    assert roots.branch_limit_mask.tolist() == [True, True, False, False]
    assert roots.voltage_limit_mask.tolist() == [False, True, False, False]
    assert roots.operational_mask.tolist() == [False, True, False, False]
    assert roots.physical_mask.tolist() == [False, True, False, False]
    np.testing.assert_allclose(roots.voltage[0, 1], low, atol=2e-7)
    np.testing.assert_allclose(roots.voltage[1, 1], high, atol=2e-7)
    assert len(roots.candidates) == 2


def test_power_root_refresh_rejects_mode_structure_change(tmp_path):
    compiled = _three_bus()
    prepared = prepare_fixed_mode_power_flow_roots(
        compile_fixed_mode_power_flow_polynomial(
            compiled, modes=("reference", "pv", "pq")
        ),
        _provider(tmp_path),
    )
    changed = compile_fixed_mode_power_flow_polynomial(
        compiled, modes=("reference", "q_max", "pq")
    )

    with pytest.raises(ValueError, match="structural/mode mismatch"):
        refresh_fixed_mode_power_flow_roots(prepared, changed)
