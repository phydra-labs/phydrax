#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from phydrax._external_runtime import PinnedExecutable
from phydrax.algebraic import SparsePolynomialSystem
from phydrax.algebraic._positive_dimensional import WitnessSet
from phydrax.backends.homotopy_continuation import (
    HomotopyContinuationEnvironment,
    HomotopyContinuationProvider,
)
from phydrax.backends.homotopy_geometry import (
    execute_homotopy_geometry,
    HomotopyGeometryPolicy,
    HomotopyGeometryRequest,
    HomotopyGeometryStatus,
)


pytestmark = pytest.mark.homotopy_continuation_live


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provider():
    if os.environ.get("PHYDRAX_RUN_HOMOTOPY_CONTINUATION_LIVE") != "1":
        pytest.skip(
            "Set PHYDRAX_RUN_HOMOTOPY_CONTINUATION_LIVE=1 for the real provider test."
        )
    executable_path = Path(os.environ["PHYDRAX_JULIA_EXECUTABLE"]).resolve(strict=True)
    project = Path(os.environ["PHYDRAX_HC_PROJECT"]).resolve(strict=True)
    executable = PinnedExecutable(
        str(executable_path),
        os.environ["PHYDRAX_JULIA_SHA256"],
        os.environ["PHYDRAX_JULIA_VERSION"],
        os.environ.get("PHYDRAX_JULIA_LICENSE", "MIT"),
        "https://julialang.org/",
    )
    assert _digest(executable_path) == executable.sha256
    environment = HomotopyContinuationEnvironment(
        project,
        os.environ["PHYDRAX_HC_PROJECT_SHA256"],
        os.environ["PHYDRAX_HC_MANIFEST_SHA256"],
        os.environ["PHYDRAX_HC_UUID"],
        os.environ["PHYDRAX_HC_VERSION"],
        depot_path=os.environ.get("PHYDRAX_JULIA_DEPOT", ""),
    )
    return HomotopyContinuationProvider(executable, environment)


def test_live_generic_slice_builds_a_qualified_witness_set():
    provider = _provider()
    system = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("parabola",),
        (0, 0),
        ((2, 0), (0, 1)),
        np.asarray((1.0, -1.0)),
    )
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    result = execute_homotopy_geometry(
        provider,
        HomotopyGeometryPolicy(path_capacity=2, timeout_seconds=300),
        request,
    )
    assert result.status is HomotopyGeometryStatus.SUCCESS
    assert result.paths is not None
    assert isinstance(result.output, WitnessSet)
    assert result.successful
    assert result.paths.successful
    assert result.output.dimension == 1
    assert result.output.degree == 2
    points = np.asarray(result.output.points)
    np.testing.assert_allclose(np.sort(points[:, 0].real), (-1.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(points[:, 1], 1.0, atol=1e-7)
    assert np.max(np.asarray(result.output.residual_norms)) < 1e-7
