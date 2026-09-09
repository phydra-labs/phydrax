#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.applications.compressible_flow import (
    NormalShockReferencePlan,
    ObliqueShockReferencePlan,
    PrandtlMeyerReferencePlan,
)
from phydrax.equations import SpalartAllmarasNegativePlan


def _references() -> dict[str, object]:
    normal = NormalShockReferencePlan(1.4).evaluate(2.0)
    oblique = ObliqueShockReferencePlan(1.4, "weak").evaluate(2.0, np.deg2rad(10.0))
    expansion = PrandtlMeyerReferencePlan(1.4).evaluate(2.0, np.deg2rad(10.0))
    checks = {
        "normal_shock": bool(normal.successful),
        "oblique_shock": bool(oblique.successful),
        "prandtl_meyer": bool(expansion.successful),
    }
    return {
        "suite": "high-speed-reference-relations",
        "checks": checks,
        "successful": all(checks.values()),
        "normal_shock": {
            "upstream_mach": float(normal.upstream_mach),
            "downstream_mach": float(normal.downstream_mach),
            "density_ratio": float(normal.density_ratio),
            "pressure_ratio": float(normal.pressure_ratio),
            "temperature_ratio": float(normal.temperature_ratio),
        },
        "oblique_shock": {
            "upstream_mach": 2.0,
            "deflection_degrees": 10.0,
            "shock_angle_degrees": float(np.rad2deg(oblique.shock_angle)),
            "downstream_mach": float(oblique.downstream_mach),
            "residual": float(oblique.residual),
        },
        "prandtl_meyer": {
            "upstream_mach": 2.0,
            "turning_degrees": 10.0,
            "downstream_mach": float(expansion.downstream_mach),
            "residual": float(expansion.residual),
        },
    }


def _rans() -> dict[str, object]:
    plan = SpalartAllmarasNegativePlan()
    velocity_gradient = np.asarray(((0.0, 2.0), (0.0, 0.0)))
    working_gradient = np.asarray((0.01, -0.02))
    positive = plan.evaluate(
        1.0, 1.0e-5, 2.0e-4, velocity_gradient, working_gradient, 0.1
    )
    negative = plan.evaluate(
        1.0, 1.0e-5, -5.0e-6, velocity_gradient, working_gradient, 0.1
    )
    checks = {
        "positive_branch": bool(positive.successful)
        and float(positive.eddy_viscosity) > 0.0,
        "negative_branch": bool(negative.successful)
        and float(negative.eddy_viscosity) == 0.0
        and float(negative.source) > 0.0,
        "positive_diffusion": float(positive.diffusion_coefficient) > 0.0
        and float(negative.diffusion_coefficient) > 0.0,
    }
    return {
        "suite": "sa-negative-closure",
        "checks": checks,
        "successful": all(checks.values()),
        "positive_eddy_viscosity": float(positive.eddy_viscosity),
        "negative_recovery_source": float(negative.source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("references", "rans", "all"), default="all")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.suite == "references":
        report = _references()
    elif arguments.suite == "rans":
        report = _rans()
    else:
        reports = (_references(), _rans())
        report = {
            "suite": "high-speed-flow",
            "reports": reports,
            "successful": all(item["successful"] for item in reports),
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
