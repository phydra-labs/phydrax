#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from phydrax._external_runtime import PinnedExecutable
from phydrax.algebraic import (
    ExactSparsePolynomialSystem,
    ExactSymbolicOperation,
    execute_exact_symbolic,
    plan_exact_symbolic,
    prepare_exact_symbolic,
    QQ,
    SparsePolynomialSupport,
    SparsePolynomialSystem,
)
from phydrax.algebraic._isolated import (
    plan_isolated_roots,
    prepare_isolated_roots,
    solve_prepared_isolated_roots,
)
from phydrax.algebraic._quotient import solve_quotient_roots
from phydrax.backends.homotopy_continuation import (
    HomotopyContinuationEnvironment,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
)
from phydrax.backends.macaulay2 import Macaulay2Environment, Macaulay2Provider


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _homotopy_provider():
    executable_path = Path(os.environ["PHYDRAX_JULIA_EXECUTABLE"]).resolve(strict=True)
    project = Path(os.environ["PHYDRAX_HC_PROJECT"]).resolve(strict=True)
    executable = PinnedExecutable(
        str(executable_path),
        os.environ["PHYDRAX_JULIA_SHA256"],
        os.environ["PHYDRAX_JULIA_VERSION"],
        os.environ.get("PHYDRAX_JULIA_LICENSE", "MIT"),
        "https://julialang.org/",
    )
    if _digest(executable_path) != executable.sha256:
        raise ValueError("Julia executable differs from its benchmark pin.")
    environment = HomotopyContinuationEnvironment(
        project,
        os.environ["PHYDRAX_HC_PROJECT_SHA256"],
        os.environ["PHYDRAX_HC_MANIFEST_SHA256"],
        os.environ["PHYDRAX_HC_UUID"],
        os.environ["PHYDRAX_HC_VERSION"],
        depot_path=os.environ.get("PHYDRAX_JULIA_DEPOT", ""),
    )
    return HomotopyContinuationProvider(executable, environment)


def _symbolic_provider():
    executable_path = Path(os.environ["PHYDRAX_MACAULAY2_EXECUTABLE"]).resolve(
        strict=True
    )
    executable = PinnedExecutable(
        str(executable_path),
        os.environ["PHYDRAX_MACAULAY2_SHA256"],
        os.environ["PHYDRAX_MACAULAY2_VERSION"],
        os.environ.get("PHYDRAX_MACAULAY2_LICENSE", "GPL-3.0-only"),
        "https://macaulay2.com/",
    )
    if _digest(executable_path) != executable.sha256:
        raise ValueError("Macaulay2 executable differs from its benchmark pin.")
    return Macaulay2Provider(Macaulay2Environment(executable))


def _system():
    return SparsePolynomialSystem.from_coo(
        ("x",),
        ("x-squared-minus-one",),
        (0, 0),
        ((0,), (2,)),
        np.asarray((-1.0, 1.0)),
    )


def main():
    system = _system()
    homotopy = _homotopy_provider()
    start = time.perf_counter()
    prepared = prepare_isolated_roots(
        plan_isolated_roots(
            system,
            homotopy,
            policy=HomotopyContinuationPolicy(
                start_system="total-degree",
                path_capacity=4,
                timeout_seconds=300,
            ),
        )
    )
    prepare_seconds = time.perf_counter() - start
    start = time.perf_counter()
    roots = solve_prepared_isolated_roots(prepared)
    solve_seconds = time.perf_counter() - start
    if not bool(roots.successful):
        raise RuntimeError(
            f"Homotopy benchmark failed: {roots.status.value}: {roots.provider.error}"
        )
    quotient = solve_quotient_roots(system, polish=True)
    if not bool(quotient.successful):
        raise RuntimeError("Native quotient root recovery failed its benchmark control.")
    provider_roots = np.asarray(roots.roots)[np.asarray(roots.root_mask)]
    quotient_roots = np.asarray(quotient.roots)
    quotient_provider_distance = float(
        np.max(
            np.min(
                np.abs(quotient_roots[:, None, :] - provider_roots[None, :, :]),
                axis=1,
            ),
            initial=0.0,
        )
    )

    support = SparsePolynomialSupport(("x",), ("f",), (0, 0), ((0,), (2,)))
    exact = ExactSparsePolynomialSystem(support, ("-1", "1"), QQ)
    symbolic = _symbolic_provider()
    start = time.perf_counter()
    exact_prepared = prepare_exact_symbolic(
        plan_exact_symbolic(exact, ExactSymbolicOperation.GROEBNER_BASIS),
        symbolic,
    )
    symbolic_prepare_seconds = time.perf_counter() - start
    start = time.perf_counter()
    exact_result = execute_exact_symbolic(exact_prepared)
    symbolic_solve_seconds = time.perf_counter() - start

    if (
        exact_result.status.name != "SUCCESS"
        or exact_result.output is None
        or exact_result.evidence is None
    ):
        raise RuntimeError(
            f"Exact symbolic benchmark failed: {exact_result.status.name}: "
            f"{exact_result.diagnostic}"
        )
    output = {
        "claim": "single-system-provider-cost-and-path-evidence-not-scaling-guarantee",
        "homotopy": {
            "prepare_seconds": prepare_seconds,
            "solve_seconds": solve_seconds,
            "status": roots.status.value,
            "planned_paths": roots.coverage.start_count,
            "tracked_paths": roots.coverage.tracked_path_count,
            "regular_paths": roots.coverage.regular_endpoint_count,
            "failed_paths": roots.coverage.tracking_failed_count,
            "clusters": roots.coverage.cluster_count,
            "quotient_provider_maximum_distance": quotient_provider_distance,
            "run_artifact_id": roots.provider.run_artifact_id,
        },
        "symbolic": {
            "prepare_seconds": symbolic_prepare_seconds,
            "solve_seconds": symbolic_solve_seconds,
            "status": int(exact_result.status),
            "output_polynomials": exact_result.output.equation_count,
            "run_artifact_id": exact_result.evidence.run_artifact_id,
        },
    }
    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
