#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from phydrax import algebraic
from phydrax._external_runtime import PinnedExecutable
from phydrax.applications import power
from phydrax.backends.homotopy_continuation import (
    HomotopyContinuationEnvironment,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
)


pytestmark = pytest.mark.homotopy_continuation_live


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provider():
    required = (
        "PHYDRAX_JULIA_EXECUTABLE",
        "PHYDRAX_JULIA_SHA256",
        "PHYDRAX_JULIA_VERSION",
        "PHYDRAX_HC_PROJECT",
        "PHYDRAX_HC_PROJECT_SHA256",
        "PHYDRAX_HC_MANIFEST_SHA256",
        "PHYDRAX_HC_UUID",
        "PHYDRAX_HC_VERSION",
    )
    missing = tuple(name for name in required if not os.environ.get(name))
    if missing:
        pytest.skip(
            "live HomotopyContinuation capability is absent: " + ", ".join(missing)
        )
    executable_path = Path(os.environ["PHYDRAX_JULIA_EXECUTABLE"]).resolve(strict=True)
    executable_digest = _digest(executable_path)
    if executable_digest != os.environ["PHYDRAX_JULIA_SHA256"]:
        pytest.fail("The live Julia executable does not match its explicit pin.")
    project = Path(os.environ["PHYDRAX_HC_PROJECT"]).resolve(strict=True)
    project_digest = _digest(project / "Project.toml")
    manifest_digest = _digest(project / "Manifest.toml")
    if project_digest != os.environ["PHYDRAX_HC_PROJECT_SHA256"]:
        pytest.fail("The live Julia Project.toml does not match its explicit pin.")
    if manifest_digest != os.environ["PHYDRAX_HC_MANIFEST_SHA256"]:
        pytest.fail("The live Julia Manifest.toml does not match its explicit pin.")
    executable = PinnedExecutable(
        str(executable_path),
        executable_digest,
        os.environ["PHYDRAX_JULIA_VERSION"],
        os.environ.get("PHYDRAX_JULIA_LICENSE", "MIT"),
        "https://julialang.org/",
    )
    environment = HomotopyContinuationEnvironment(
        project,
        project_digest,
        manifest_digest,
        os.environ["PHYDRAX_HC_UUID"],
        os.environ["PHYDRAX_HC_VERSION"],
        depot_path=os.environ.get("PHYDRAX_JULIA_DEPOT", ""),
    )
    return HomotopyContinuationProvider(executable, environment)


def test_live_isolated_roots_and_fixed_mode_power_flow():
    provider = _provider()
    system = algebraic.SparsePolynomialSystem.from_coo(
        ("x",),
        ("x-squared-minus-one",),
        (0, 0),
        ((0,), (2,)),
        np.asarray((-1.0, 1.0)),
    )
    prepared = algebraic.prepare_isolated_roots(
        algebraic.plan_isolated_roots(
            system,
            provider,
            policy=HomotopyContinuationPolicy(
                start_system="total-degree",
                path_capacity=4,
                timeout_seconds=300,
            ),
        )
    )
    result = algebraic.solve_prepared_isolated_roots(prepared)
    assert result.successful
    assert result.coverage.all_tracked_paths_accounted
    assert result.coverage.tracking_failed_count == 0
    roots = np.sort(np.asarray(result.roots)[np.asarray(result.root_mask), 0].real)
    np.testing.assert_allclose(roots, (-1.0, 1.0), atol=1e-8)
    quotient = algebraic.solve_quotient_roots(system, polish=True)
    assert quotient.successful
    np.testing.assert_allclose(
        np.sort(np.asarray(quotient.roots[:, 0]).real),
        roots,
        atol=1e-8,
    )

    network = power.PowerNetwork(
        (power.Bus("source", 110), power.Bus("load", 110)),
        (power.Branch("line", "source", "load", 0.0, 0.1),),
        (power.Generator("g", "source"),),
        (power.Load("d", "load", 0.5, 0.0),),
    )
    study = power.PowerStudy(
        (power.BusControl("source", "reference"), power.BusControl("load", "pq"))
    )
    compiled = power.compile_network(network, study)
    root_set = power.enumerate_fixed_mode_power_flow_roots(
        compiled,
        provider,
        policy=HomotopyContinuationPolicy(
            start_system="total-degree",
            path_capacity=8,
            timeout_seconds=300,
        ),
        polynomial_residual_tolerance=1e-7,
        physical_residual_tolerance=1e-7,
    )
    candidates = np.asarray(root_set.voltage)[np.asarray(root_set.candidate_mask)]
    assert candidates.shape[0] >= 2
    load_voltage = np.sort_complex(candidates[:, 1])
    discriminant = np.sqrt(1.0 - 4.0 * 0.05**2)
    expected = np.sort_complex(
        np.asarray(
            (
                0.5 * (1.0 - discriminant) - 0.05j,
                0.5 * (1.0 + discriminant) - 0.05j,
            )
        )
    )
    distances = np.min(np.abs(load_voltage[:, None] - expected[None, :]), axis=0)
    assert np.all(distances < 1e-7)
