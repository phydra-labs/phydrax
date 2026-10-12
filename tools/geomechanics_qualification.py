#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded geomechanics candidate comparisons; never release qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from math import atan, cos, exp, expm1, isfinite, log1p, pi, sqrt, tan
from pathlib import Path
from time import perf_counter
from typing import Any, assert_never, get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

import phydrax as phx
from phydrax.applications.solid_mechanics import MechanicalLoadState
from phydrax.discretization import (
    CompliantDEMMethodPlan,
    DEMContactModelPlan,
    DEMRuntimeState,
    DenseParticleNeighborhoodPlan,
    HertzNormalContactPlan,
    ParticleSetPlan,
    RigidSphereSetPlan,
)
from phydrax.equations import (
    compile_discrete_element_problem,
    CompiledDiscreteElementProblem,
    DEMMaterialTable,
    DiscreteElementProblemIR,
    InterphaseDragLawPlan,
)
from phydrax.integration import DeformedMeasureState
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    PublicationReference,
    QualificationCriterion,
    QualificationEvidence,
    ScientificCase,
    SupportTuple,
    validate_qualification_causality,
)
from phydrax.rheology import MuILaw, MuIParameters
from phydrax.typing import parse


CaseName: TypeAlias = Literal[
    "lame-thick-cylinder",
    "prandtl-footing-tresca",
    "prandtl-footing-frictional",
    "hertz-normal-impact",
    "ergun-packed-bed",
    "schiller-naumann-terminal-velocity",
    "bagnold-inclined-plane",
    "terzaghi-1d-consolidation",
    "wu-thornton-li-oblique-impact",
]


MAX_REPORT_BYTES = 2_000_000


MAX_CASES = len(get_args(CaseName))


CANDIDATE_NOTICE = "Proposed candidate criteria; no approval, release authorization, or field-validation claim."


def _json_scalar(value: Any) -> Any:
    """Convert host NumPy scalar observations without repairing their values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported qualification JSON value: {type(value).__name__}.")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_scalar,
        ).encode()
    ).hexdigest()


def encode_report(report: dict[str, Any]) -> str:
    if len(report["cases"]) > MAX_CASES:
        raise ValueError("Report exceeds the registered case bound.")
    encoded = (
        json.dumps(
            report, sort_keys=True, indent=2, allow_nan=False, default=_json_scalar
        )
        + "\n"
    )
    if len(encoded.encode()) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the 2 MB metadata bound.")
    return encoded


def _drag_inputs(
    porosity: float, speed: float | jax.Array, diameter: float
) -> tuple[jax.Array, ...]:
    return tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            porosity,
            speed,
            diameter,
            0.001,
            1000.0,
            math.pi * diameter**3 / 6.0,
        )
    )


def run_ergun_packed_bed() -> dict[str, Any]:
    law = InterphaseDragLawPlan("ergun")
    porosity, diameter, viscosity, density = 0.4, 0.002, 0.001, 1000.0
    levels = []
    for count in (4, 8):
        superficial = np.geomspace(1e-6, 0.01, count, dtype=np.float64)
        intrinsic = jnp.asarray(superficial / porosity, dtype=jnp.float64)
        result = law.evaluate(
            jnp.asarray(porosity, dtype=jnp.float64),
            intrinsic,
            jnp.asarray(diameter, dtype=jnp.float64),
            jnp.asarray(viscosity, dtype=jnp.float64),
            jnp.asarray(density, dtype=jnp.float64),
            jnp.asarray(math.pi * diameter**3 / 6, dtype=jnp.float64),
        )
        gradient = np.asarray(result.beta * intrinsic / porosity)
        reference = 150 * viscosity * (1 - porosity) ** 2 * superficial / (
            porosity**3 * diameter**2
        ) + 1.75 * density * (1 - porosity) * superficial**2 / (porosity**3 * diameter)
        error = float(np.max(np.abs(gradient / reference - 1)))
        levels.append(
            {
                "resolution": count,
                "metrics": {"pressure-gradient-relative-error": error},
                "successful": bool(np.all(np.asarray(result.successful))),
                "superficial-velocities": superficial.tolist(),
                "pressure-gradients": gradient.tolist(),
            }
        )
    return {
        "resolutions": [4, 8],
        "metrics": levels[-1]["metrics"],
        "levels": levels,
        "successful": all(level["successful"] for level in levels),
        "inconclusive": False,
        "reason": "Direct closure comparison; sampling refinement is not spatial convergence.",
        "resources": {"maximum-flow-points": 8, "evaluations": 12},
    }


def run_schiller_naumann_terminal_velocity() -> dict[str, Any]:
    diameter, viscosity, density, particle_density = 0.001, 0.001, 1000.0, 2500.0
    volume = math.pi * diameter**3 / 6
    weight = (particle_density - density) * volume * 9.81

    def host_residual(speed: float) -> float:
        reynolds = density * diameter * speed / viscosity
        return (
            3 * math.pi * viscosity * diameter * (1 + 0.15 * reynolds**0.687) * speed
            - weight
        )

    reference = float(brentq(host_residual, 0, 0.5, xtol=1e-14, maxiter=100))
    law = InterphaseDragLawPlan("schiller_naumann")

    def solve(iterations: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        def body(
            _: int, state: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            left, right, successful = state
            speed = (left + right) / 2
            result = law.evaluate(
                jnp.asarray(1.0, dtype=jnp.float64),
                speed,
                jnp.asarray(diameter, dtype=jnp.float64),
                jnp.asarray(viscosity, dtype=jnp.float64),
                jnp.asarray(density, dtype=jnp.float64),
                jnp.asarray(volume, dtype=jnp.float64),
            )
            force = result.particle_drag_coefficient * speed
            return (
                jnp.where(force < weight, speed, left),
                jnp.where(force >= weight, speed, right),
                successful & result.successful,
            )

        left, right, successful = jax.lax.fori_loop(
            0,
            iterations,
            body,
            (
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(0.5, dtype=jnp.float64),
                jnp.asarray(True, dtype=jnp.bool_),
            ),
        )
        speed = (left + right) / 2
        force = (
            law.evaluate(*_drag_inputs(1.0, speed, diameter)).particle_drag_coefficient
            * speed
        )
        return speed, force, successful

    levels = []
    for iterations in (20, 30, 40):
        speed, force, successful = jax.jit(solve, static_argnums=0)(iterations)
        speed_value, force_value = float(speed), float(force)
        levels.append(
            {
                "resolution": iterations,
                "terminal-velocity": speed_value,
                "reynolds": density * diameter * speed_value / viscosity,
                "successful": bool(successful),
                "metrics": {
                    "terminal-velocity-relative-error": abs(speed_value / reference - 1),
                    "force-relative-residual": abs(force_value / weight - 1),
                },
            }
        )
    return {
        "resolutions": [20, 30, 40],
        "metrics": levels[-1]["metrics"],
        "levels": levels,
        "successful": all(level["successful"] for level in levels),
        "inconclusive": False,
        "reason": "Device-law bisection versus independent host Brent root on Re <= 1000.",
        "resources": {"maximum-root-iterations": 40, "host-root-bound": 100},
        "reference-velocity": reference,
    }


def _spec(
    name: CaseName,
) -> tuple[PublicationReference, dict[str, float], dict[str, Any]]:
    match name:
        case "lame-thick-cylinder":
            return (
                PublicationReference(
                    "Lamé thick-cylinder solution; Timoshenko & Goodier, Theory of Elasticity",
                    "Theory of Elasticity, 3rd ed., 1970",
                ),
                {"stress-relative-l2-error": 0.01},
                {
                    "geometry": "axisymmetric-full-ring-plane-strain",
                    "load": "inner-pressure",
                    "stabilization": "none",
                },
            )
        case "prandtl-footing-tresca" | "prandtl-footing-frictional":
            return (
                PublicationReference(
                    "Prandtl (1920), Über die Härte plastischer Körper; Reissner (1924), Zum Erddruckproblem",
                    "https://doi.org/10.1002/zamm.19240040605",
                ),
                {"bearing-capacity-relative-error": 0.02},
                {
                    "geometry": "plane-strain-rough-strip",
                    "self-weight": 0.0,
                    "dilation": "associated",
                    "stabilization": "b_bar",
                },
            )
        case "hertz-normal-impact":
            return (
                PublicationReference(
                    "Hertz (1881); Johnson, Contact Mechanics (1985)",
                    "https://doi.org/10.1017/CBO9781139171731",
                ),
                {"duration-relative-error": 0.005, "peak-force-relative-error": 0.005},
                {"geometry": "two-identical-spheres", "damping": 0.0, "friction": 0.0},
            )
        case "ergun-packed-bed":
            return (
                PublicationReference(
                    "Ergun (1952), Fluid flow through packed columns",
                    "Chemical Engineering Progress 48, 89–94",
                ),
                {"pressure-gradient-relative-error": 1e-6},
                {
                    "porosity": 0.4,
                    "diameter": 0.002,
                    "density": 1000.0,
                    "viscosity": 0.001,
                    "superficial-speed-range": (1e-6, 0.01),
                },
            )
        case "schiller-naumann-terminal-velocity":
            return (
                PublicationReference(
                    "Schiller & Naumann, Über die grundlegenden Berechnungen bei der Schwerkraftaufbereitung",
                    "VDI-Zeitschrift 77 (1933), 318–320; original-journal library holdings: https://catalog.hathitrust.org/Record/000637388",
                ),
                {
                    "terminal-velocity-relative-error": 1e-7,
                    "force-relative-residual": 1e-8,
                },
                {
                    "porosity": 1.0,
                    "diameter": 0.001,
                    "particle-density": 2500.0,
                    "fluid-density": 1000.0,
                    "viscosity": 0.001,
                    "gravity": 9.81,
                    "maximum-reynolds": 1000.0,
                },
            )
        case "bagnold-inclined-plane":
            return (
                PublicationReference(
                    "GDR MiDi (2004), On dense granular flows; Jop et al. (2006)",
                    "https://doi.org/10.1140/epje/i2003-10153-0",
                ),
                {"profile-relative-l2-error": 0.02},
                {
                    "geometry": "periodic-plane-strain-layer",
                    "base": "rough-no-slip",
                    "route": "public-implicit-mu-i-mpm",
                    "pressure": "coupled-cauchy-stress",
                },
            )
        case "terzaghi-1d-consolidation":
            return (
                PublicationReference(
                    "Terzaghi (1925), Erdbaumechanik auf bodenphysikalischer Grundlage",
                    "Franz Deuticke, Leipzig/Vienna, 1925",
                ),
                {"pressure-relative-l2-error": 0.01, "settlement-relative-error": 0.02},
                {
                    "geometry": "single-drained-column",
                    "route": "full-advance-double-point-mpm-step",
                    "biot": 1.0,
                },
            )
        case "wu-thornton-li-oblique-impact":
            return (
                PublicationReference(
                    "Wu, Thornton & Li (2009), A semi-analytical model for oblique impacts of elastoplastic spheres",
                    "https://doi.org/10.1098/rspa.2008.0221",
                ),
                {"velocity-relative-error": 0.03},
                {"route": "omitted-no-rights-cleared-quantitative-reference"},
            )
        case _:
            assert_never(name)


def _settings(name: CaseName) -> dict[str, Any]:
    match name:
        case (
            "lame-thick-cylinder"
            | "prandtl-footing-tresca"
            | "prandtl-footing-frictional"
        ):
            return mpm_case_settings(name)
        case "hertz-normal-impact" | "bagnold-inclined-plane":
            return granular_case_settings(name)
        case "terzaghi-1d-consolidation":
            return terzaghi_1d_consolidation_settings()
        case "ergun-packed-bed":
            return {
                "resolutions": [4, 8],
                "resource-bounds": {"maximum-flow-points": 8, "evaluations": 12},
                "superficial-velocity-sampling": "geometric-inclusive-1e-6-to-0.01",
            }
        case "schiller-naumann-terminal-velocity":
            return {
                "resolutions": [20, 30, 40],
                "speed-bracket": [0.0, 0.5],
                "host-root-xtol": 1e-14,
                "resource-bounds": {
                    "maximum-root-iterations": 40,
                    "host-root-bound": 100,
                },
            }
        case "wu-thornton-li-oblique-impact":
            return {
                "resolutions": [],
                "resource-bounds": {"steps": 0},
                "reference-status": "omitted-no-rights-clearance",
            }
        case _:
            assert_never(name)


def run_case(name: CaseName) -> dict[str, Any]:
    match name:
        case "lame-thick-cylinder":
            return run_lame_thick_cylinder()
        case "prandtl-footing-tresca":
            return run_prandtl_footing_tresca()
        case "prandtl-footing-frictional":
            return run_prandtl_footing_frictional()
        case "hertz-normal-impact":
            return run_hertz_normal_impact()
        case "ergun-packed-bed":
            return run_ergun_packed_bed()
        case "schiller-naumann-terminal-velocity":
            return run_schiller_naumann_terminal_velocity()
        case "bagnold-inclined-plane":
            return run_bagnold_inclined_plane()
        case "terzaghi-1d-consolidation":
            return run_terzaghi_1d_consolidation()
        case "wu-thornton-li-oblique-impact":
            return {
                "resolutions": [],
                "metrics": {"velocity-relative-error": None},
                "levels": [],
                "successful": True,
                "inconclusive": True,
                "omitted": True,
                "reason": "No rights-cleared quantitative reference identified; no curves digitized or redistributed.",
                "resources": {"steps": 0},
            }
        case _:
            assert_never(name)


def _outcome(raw: dict[str, Any], criteria: tuple[QualificationCriterion, ...]) -> str:
    if raw.get("overflow", False):
        return "failed"
    if raw.get("model_range_limited", False):
        return "inconclusive"
    if not raw["successful"]:
        return "failed"
    values = [raw["metrics"].get(criterion.metric) for criterion in criteria]
    if raw["inconclusive"] or any(
        value is None or not math.isfinite(value) for value in values
    ):
        return "inconclusive"
    return (
        "passed"
        if all(
            value <= criterion.target
            for value, criterion in zip(values, criteria, strict=True)
        )
        else "failed"
    )


def qualify(
    case_names: Sequence[CaseName],
    /,
    *,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    names = tuple(parse(name, CaseName, "case") for name in case_names)
    if not names or len(names) > MAX_CASES or len(set(names)) != len(names):
        raise ValueError("Choose one to nine distinct registered cases.")
    if not jax.config.x64_enabled:
        raise ValueError(
            "Qualification requires JAX float64; enable jax_enable_x64 before execution."
        )
    runtime = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
        "precision": "float64",
        "contention": "shared-worktree-measurements",
    }
    report: dict[str, Any] = {
        "kind": "geomechanics-candidate-qualification",
        "notice": CANDIDATE_NOTICE,
        "runtime": runtime,
        "maximum-report-bytes": MAX_REPORT_BYTES,
        "cases": [],
    }
    for name in names:
        publication, targets, attributes = _spec(name)
        settings = _settings(name)
        source = {
            "title": publication.title,
            "locator": publication.locator,
            "data-rights": "Independent equation implementation; no publication data redistributed.",
        }
        source_id = _digest(source)
        support = SupportTuple(
            "geomechanics.reference-comparison",
            {
                "case": name,
                "precision": "float64",
                "numerical-settings-id": _digest(settings),
                **{
                    key: value
                    if isinstance(value, (str, int))
                    else json.dumps(value, allow_nan=False)
                    for key, value in attributes.items()
                },
            },
        )
        scientific_case = ScientificCase(
            name,
            name,
            "analytic-geomechanics-comparison",
            name,
            support.support_tuple_id,
            "geomechanics-substrates",
            (source_id,),
        )
        issued_at = clock()
        criteria = tuple(
            QualificationCriterion(
                support_tuple_id=support.support_tuple_id,
                metric=metric,
                unit="dimensionless",
                comparison="less-than-or-equal",
                target=target,
                aggregation="finest-registered-resolution",
                uncertainty="candidate-bound-not-validated",
                applicability="only-exact-recorded-support",
                approval_id="proposed-candidate-NOT-approved",
                issued_at=issued_at,
            )
            for metric, target in sorted(targets.items())
        )
        started_at = clock()
        if started_at <= issued_at:
            raise ValueError("Criterion must be issued strictly before campaign start.")
        campaign_id = _digest(
            {
                "case": scientific_case.to_record(),
                "criteria": [c.to_record() for c in criteria],
            }
        )
        run_id = _digest({"campaign": campaign_id, "runtime": runtime})
        starts = tuple(
            CampaignStartRecord(
                campaign_spec_id=campaign_id,
                criterion_id=c.criterion_id,
                resolved_run_spec_id=run_id,
                support_tuple_id=support.support_tuple_id,
                started_at=started_at,
            )
            for c in criteria
        )
        entry: dict[str, Any] = {
            "case": name,
            "scientific-case": scientific_case.to_record(),
            "support": support.to_record(),
            "source": {**source, "source-manifest-id": source_id},
            "settings": settings,
            "criterion-status": "proposed-candidate-not-approved",
            "criteria": [c.to_record() for c in criteria],
            "starts": [start.to_record() for start in starts],
            "status": "started",
        }
        report["cases"].append(entry)
        if checkpoint is not None:
            checkpoint(report)
        raw = run_case(name)
        observed_at = clock()
        raw_id = _digest(raw)
        observations = tuple(
            CampaignObservationRecord(
                start_record_id=start.start_record_id,
                campaign_spec_id=campaign_id,
                criterion_id=start.criterion_id,
                resolved_run_spec_id=run_id,
                support_tuple_id=support.support_tuple_id,
                raw_artifact_ids=(raw_id,),
                observed_at=observed_at,
            )
            for start in starts
        )
        outcome = _outcome(raw, criteria)
        evidence_at = clock()
        evidence = QualificationEvidence(
            "scientific",
            outcome,
            (name, support.support_tuple_id),
            build_id="uncommitted-geomechanics-worktree",
            environment_id=_digest(runtime),
            backend=runtime["backend"],
            topology="natural-local-devices",
            precision="float64",
            reduction="deterministic-case-local",
            replay_id=run_id,
            criteria_ids=tuple(c.criterion_id for c in criteria),
            raw_artifact_ids=(raw_id,),
            campaign_start_record_ids=tuple(start.start_record_id for start in starts),
            campaign_observation_record_ids=tuple(
                o.observation_record_id for o in observations
            ),
            reviewer_id="automated-candidate-comparison-NOT-human-reviewed",
            issued_at=evidence_at,
            expires_at=evidence_at + 86_400_000_000_000,
            reason=raw["reason"] or CANDIDATE_NOTICE,
        )
        for criterion, start, observation in zip(
            criteria, starts, observations, strict=True
        ):
            validate_qualification_causality(criterion, start, observation, evidence)
        entry.update(
            {
                "status": "observed",
                "outcome": outcome,
                "raw": raw,
                "raw-artifact-id": raw_id,
                "observations": [o.to_record() for o in observations],
                "evidence": evidence.to_record(),
            }
        )
        if checkpoint is not None:
            checkpoint(report)
    report["outcome"] = (
        "failed"
        if any(case["outcome"] == "failed" for case in report["cases"])
        else "inconclusive"
        if any(case["outcome"] == "inconclusive" for case in report["cases"])
        else "passed"
    )
    encode_report(report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--case", action="append", choices=get_args(CaseName))
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    names = get_args(CaseName) if arguments.all else arguments.case
    jax.config.update("jax_enable_x64", True)

    def checkpoint(report: dict[str, Any]) -> None:
        if arguments.output is not None:
            arguments.output.write_text(encode_report(report), encoding="utf-8")

    report = qualify(names, checkpoint=checkpoint)
    if arguments.output is None:
        print(encode_report(report), end="")
    else:
        checkpoint(report)
    raise SystemExit(0 if report["outcome"] == "passed" else 1)


_MPMS16CaseName: TypeAlias = Literal[
    "lame-thick-cylinder", "prandtl-footing-tresca", "prandtl-footing-frictional"
]


_FOOTING_SHEAR_MODULUS = 5000.0
_FOOTING_BULK_MODULUS = 50000.0
_FOOTING_COHESION = 1.0
_FOOTING_TARGET_INDENTATION = 0.006
_FOOTING_DT_COEFFICIENT = 0.0009


def _footing_time_grid(n: int, speed: float) -> tuple[int, float]:
    maximum_dt = _FOOTING_DT_COEFFICIENT / n
    steps = math.ceil(_FOOTING_TARGET_INDENTATION / (speed * maximum_dt))
    return steps, _FOOTING_TARGET_INDENTATION / (speed * steps)


def _footing_forecast_seconds(name: str, n: int) -> float:
    """Scale the old measured n8 cost by particle count and actual step count."""
    steps = _footing_time_grid(n, 0.02)[0] + _footing_time_grid(n, 0.01)[0]
    return _mpm_footing_baseline_seconds(name) * (n / 8.0) ** 2 * steps / (1152 * 8)


_MPMS16FootingCaseName: TypeAlias = Literal[
    "prandtl-footing-tresca", "prandtl-footing-frictional"
]


def _mpm_footing_phi(name: str) -> float:
    name_ = parse(name, _MPMS16FootingCaseName, "name")
    match name_:
        case "prandtl-footing-tresca":
            return 0.0
        case "prandtl-footing-frictional":
            return pi / 9.0
        case _ as unreachable:
            assert_never(unreachable)


def _mpm_footing_baseline_seconds(name: str) -> float:
    """Observed old n8 TWO-rate elapsed totals; scheduling metadata only."""
    name_ = parse(name, _MPMS16FootingCaseName, "name")
    match name_:
        case "prandtl-footing-tresca":
            return 64.88939254102297 + 147.1411449171137
        case "prandtl-footing-frictional":
            return 78.84120870800689 + 150.6420390419662
        case _ as unreachable:
            assert_never(unreachable)


def _mpm_footing_resolutions(name: str, resolutions: Sequence[int]) -> tuple[int, ...]:
    """Validate fixed resolution/work bounds before allocating solver state."""
    levels = tuple(resolutions)
    if not levels:
        raise ValueError("A footing campaign requires at least one fixed resolution.")
    previous = 0
    for n in levels:
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError("Footing resolutions must be Python integer cell counts.")
        if n < 4 or n > 16 or n % 2 != 0 or (n <= previous):
            raise ValueError(
                "Footing resolutions must be strictly increasing even full-width cell counts in [4,16]."
            )
        previous = n
    estimate = sum((_footing_forecast_seconds(name, n) for n in levels))
    if estimate > 2100.0:
        raise ValueError(
            "Requested footing levels exceed the 2100-second cubic affordability-model ceiling; split into separate causal campaign parts."
        )
    return levels


def _mpm_footing_settings(
    name: str, resolutions: Sequence[int] = (8, 16)
) -> dict[str, Any]:
    """Exact new-run support; historical data enters only a labeled cost forecast."""
    phi = _mpm_footing_phi(name)
    levels = _mpm_footing_resolutions(name, resolutions)
    baseline = _mpm_footing_baseline_seconds(name)
    nc = _mpm_footing_reference(phi)
    return {
        "resolutions": list(levels),
        "resolution_meaning": "n cells across FULL footing width B=1; h=B/n; B/2 contains n/2 cells",
        "half_width_cell_counts": [n // 2 for n in levels],
        "cell_sizes": [1.0 / n for n in levels],
        "campaign_role": "fresh fixed-level refinement; historical coarse observations retained as separate causal campaign parts",
        "historical_capacity_input": "none; no recorded reaction or material state feeds the solver",
        "steps_per_level_fast": [_footing_time_grid(n, 0.02)[0] for n in levels],
        "steps_per_level_slow": [_footing_time_grid(n, 0.01)[0] for n in levels],
        "step_sizes_fast": [_footing_time_grid(n, 0.02)[1] for n in levels],
        "total_step_bound": sum(
            (
                _footing_time_grid(n, 0.02)[0] + _footing_time_grid(n, 0.01)[0]
                for n in levels
            )
        ),
        "geometry": "3D Cartesian periodic extrusion, every nodal v_z=0 (plane strain)",
        "stabilization": "b_bar",
        "material": "Mohr-Coulomb; phi=psi=0 is Tresca",
        "cohesion": _FOOTING_COHESION,
        "shear_modulus": _FOOTING_SHEAR_MODULUS,
        "bulk_modulus": _FOOTING_BULK_MODULUS,
        "friction_angle_radians": phi,
        "dilation_angle_radians": phi,
        "hardening_modulus": 0.0,
        "surcharge": 0.0,
        "footing_width": 1.0,
        "box_width": 6.0,
        "box_depth": 3.0,
        "extrusion_length_per_level": [3.0 / n for n in levels],
        "indentation": _FOOTING_TARGET_INDENTATION,
        "platen_speeds": [0.02, 0.01],
        "physical_durations": [
            _FOOTING_TARGET_INDENTATION / 0.02,
            _FOOTING_TARGET_INDENTATION / 0.01,
        ],
        "maximum_steps_per_solve": _footing_time_grid(levels[-1], 0.01)[0],
        "maximum_particles": 36 * levels[-1] ** 2,
        "maximum_grid_nodes": 3 * (6 * levels[-1] + 5) * (3 * levels[-1] + 5),
        "particles_per_xy_cell": 1,
        "axial_particles": 2,
        "axial_grid_nodes": 3,
        "plane_strain_diagnostic": "relative axial motion and paired-extrusion transverse particle kinematics; not assumed B-bar F",
        "boundary": "rough vx=0, vy=-speed on reference y=0 strip nodes; base fixed; side rollers; z periodic",
        "reaction": "negative strip sum[m*(v_after-v_before)/dt - f_internal - f_external], divided by nominal width and extrusion length",
        "capacity_estimator": "mean reaction over last 20% of indentation; not largest transient spike",
        "candidate_tolerance": 0.02,
        "resolution_gap_limit": 0.02,
        "rate_gap_limit": 0.02,
        "plateau_drift_limit": 0.02,
        "plateau_variation_limit": 0.02,
        "kinetic_to_reference_work_limit": 0.01,
        "reference": "Nc=pi+2 for phi=0; otherwise Nq=exp(pi*tan(phi))*tan(pi/4+phi/2)^2, Nc=(Nq-1)/tan(phi); q_u=c*Nc",
        "reference_Nc": nc,
        "reference_Nq": 1.0
        if phi == 0.0
        else exp(pi * tan(phi)) * tan(pi / 4.0 + phi / 2.0) ** 2,
        "Nq_status": "algebraically inferred diagnostic; NOT independently measured or qualified",
        "Nq_diagnostic_derivation": "Nq_inferred=Nc_reaction*tan(phi)+1; at phi=0 this is identically1 and has no capacity sensitivity",
        "Nq_independent_run": "not executed: requires independently equilibrated prestress and maintained surface surcharge, plus matched incremental capacities; two additional rate solves exceed current affordability budget",
        "convergence_analysis": "measured signed Nc errors, pairwise absolute-error ratios and log(error_coarse/error_fine)/log(n_fine/n_coarse); unresolved plateau/rate pairs are diagnostic only; no extrapolated capacities",
        "affordability": {
            "target_seconds_per_case": 1800,
            "maximum_modeled_seconds_per_case": 2100,
            "model": "old n8 measured two-rate cost scaled by n^2 particle count and corrected actual step count; includes prior compilation/contention; forecast only",
            "baseline_resolution_full_width": 8,
            "baseline_two_rate_elapsed_seconds": baseline,
            "baseline_evidence": "local://s16-results.json; original /tmp/s16-geomechanics-footings.json causal part",
            "modeled_seconds_per_level": [
                _footing_forecast_seconds(name, n) for n in levels
            ],
            "modeled_seconds_total": sum(
                (_footing_forecast_seconds(name, n) for n in levels)
            ),
            "runtime_policy": "parent executes mostly idle CPU with requested threads2; approximately30min target, contended model up to35min; exact fixed steps, no time-based silent truncation",
            "full_width_n32_cost_multiplier_vs_n8": 64,
        },
        "reference_url": "https://doi.org/10.1016/j.ijsolstr.2004.08.002",
        "reference_location": "Hjiaj, Lyamin & Sloan (2005), Eq. (30); rough boundary Eq. (11); nodal reaction guidance p.1695",
        "limitations": "finite width/depth not independently enlarged; fixed reference nodal platen approximates infinitesimal-indentation rough contact; no full finite-penetration contact claim; zero-surcharge run identifies Nc only; inferred Nq is not independent evidence; a requested singleton remains mesh-inconclusive; default8->16 gap is measured afresh within this part",
        "rigidity_index_G_over_c": _FOOTING_SHEAR_MODULUS / _FOOTING_COHESION,
        "poisson_ratio": (3 * _FOOTING_BULK_MODULUS - 2 * _FOOTING_SHEAR_MODULUS)
        / (2 * (3 * _FOOTING_BULK_MODULUS + _FOOTING_SHEAR_MODULUS)),
        "half_cell_platen_support_limits": [0.5 / n for n in levels],
        "target_over_half_cell_support_limit": [
            2 * n * _FOOTING_TARGET_INDENTATION for n in levels
        ],
        "explicit_dt_bound_coefficient": _FOOTING_DT_COEFFICIENT,
        "maximum_load_settlement_curve_samples": 33,
        "initial_elastic_cfl": 1.5
        * _FOOTING_DT_COEFFICIENT
        * sqrt(_FOOTING_BULK_MODULUS + 4 * _FOOTING_SHEAR_MODULUS / 3),
        "undeformed_kernel_wave_factor": 1.5,
        "step_sizes_slow": [_footing_time_grid(n, 0.01)[1] for n in levels],
        "rigid_plastic_limit_note": "G/c=5000, K/G=10 preserves nu; Prandtl collapse is independent of G in rigid-perfectly-plastic limit. Actual curve plateau before h/2 is required, not assumed. Previous G/c=100, indentation0.05B configuration is superseded.",
    }


def _mpm_footing_convergence(
    name: str, campaign_parts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Read-only host analysis of original solver outputs; no evidence replacement."""
    phi = _mpm_footing_phi(name)
    reference_nc = _mpm_footing_reference(phi)
    reference_nq = (
        1.0 if phi == 0.0 else exp(pi * tan(phi)) * tan(pi / 4.0 + phi / 2.0) ** 2
    )
    levels: list[dict[str, Any]] = []
    seen: set[int] = set()
    all_successful = True
    # Do not merge differing physical supports merely because cell counts match.
    support_keys = (
        "geometry",
        "stabilization",
        "material",
        "cohesion",
        "shear_modulus",
        "bulk_modulus",
        "friction_angle_radians",
        "dilation_angle_radians",
        "hardening_modulus",
        "surcharge",
        "footing_width",
        "box_width",
        "box_depth",
        "indentation",
        "platen_speeds",
        "particles_per_xy_cell",
        "axial_particles",
        "axial_grid_nodes",
        "boundary",
        "capacity_estimator",
        "plane_strain_diagnostic",
        "candidate_tolerance",
        "resolution_gap_limit",
        "rate_gap_limit",
        "plateau_drift_limit",
        "plateau_variation_limit",
        "kinetic_to_reference_work_limit",
        "dtype",
        "density",
        "gravity",
        "transfer",
        "schedule",
        "assignment",
    )
    expected: Mapping[str, Any] | None = None
    for part_index, part in enumerate(campaign_parts):
        settings = part["settings"]
        if settings["friction_angle_radians"] != phi:
            raise ValueError(
                "Historical part belongs to a different footing friction angle."
            )
        if expected is None:
            expected = settings
        elif any(settings[key] != expected[key] for key in support_keys):
            raise ValueError(
                "Footing parts have different physical/numerical support; cannot compare their mesh sequence."
            )
        if part["reference_capacity"] != reference_nc:
            raise ValueError("Footing part reference does not match the recorded case.")
        all_successful = all_successful and bool(part["successful"])
        for level in part["levels"]:
            n = level["resolution"]
            if n in seen:
                raise ValueError(
                    "Repeated cell count needs an explicitly chosen observation, not hidden averaging."
                )
            seen.add(n)
            fast_runs = [rate for rate in level["rates"] if rate["speed"] == 0.02]
            slow_runs = [rate for rate in level["rates"] if rate["speed"] == 0.01]
            if len(fast_runs) != 1 or len(slow_runs) != 1 or len(level["rates"]) != 2:
                raise ValueError(
                    "Every compared footing level requires exactly the two recorded load rates."
                )
            fast, slow = fast_runs[0], slow_runs[0]
            successful = bool(
                level["solver_successful"]
                and fast["solver_successful"]
                and slow["solver_successful"]
            )
            all_successful = all_successful and successful
            capacity = slow["measured_capacity"]
            fast_capacity = fast["measured_capacity"]
            measured_nc = (
                float(capacity)
                if successful and capacity is not None and isfinite(capacity)
                else None
            )
            rate_gap = (
                abs(measured_nc - fast_capacity) / reference_nc
                if measured_nc is not None
                and fast_capacity is not None
                and isfinite(fast_capacity)
                else None
            )
            steady = bool(
                successful
                and fast["plateau_resolved"]
                and slow["plateau_resolved"]
                and rate_gap is not None
                and rate_gap <= 0.02
            )
            inferred_nq = (
                measured_nc * tan(phi) + 1.0 if measured_nc is not None else None
            )
            levels.append(
                {
                    "resolution_full_width": n,
                    "resolution_half_width": n / 2.0,
                    "campaign_part_index": part_index,
                    "measured_Nc": measured_nc,
                    "Nc_signed_relative_error": measured_nc / reference_nc - 1.0
                    if measured_nc is not None
                    else None,
                    "Nc_absolute_relative_error": abs(measured_nc / reference_nc - 1.0)
                    if measured_nc is not None
                    else None,
                    "Nq_algebraically_inferred_NOT_independent": inferred_nq,
                    "Nq_inferred_absolute_relative_error_NOT_qualification": abs(
                        inferred_nq / reference_nq - 1.0
                    )
                    if inferred_nq is not None
                    else None,
                    "solver_successful": successful,
                    "rate_plateau_resolved": steady,
                    "loading_rate_relative_gap": rate_gap,
                }
            )
    levels.sort(key=lambda level: level["resolution_full_width"])
    pairs: list[dict[str, Any]] = []
    for coarse, fine in zip(levels[:-1], levels[1:], strict=True):
        coarse_nc, fine_nc = coarse["measured_Nc"], fine["measured_Nc"]
        coarse_error, fine_error = (
            coarse["Nc_absolute_relative_error"],
            fine["Nc_absolute_relative_error"],
        )
        resolution_ratio = fine["resolution_full_width"] / coarse["resolution_full_width"]
        valid_values = coarse_nc is not None and fine_nc is not None
        gap = abs(fine_nc - coarse_nc) / reference_nc if valid_values else None
        error_ratio = (
            fine_error / coarse_error
            if coarse_error is not None and fine_error is not None and coarse_error > 0.0
            else None
        )
        slope = (
            math.log(coarse_error / fine_error) / math.log(resolution_ratio)
            if coarse_error is not None
            and fine_error is not None
            and coarse_error > 0.0
            and fine_error > 0.0
            else None
        )
        coarse_nq_error = coarse["Nq_inferred_absolute_relative_error_NOT_qualification"]
        fine_nq_error = fine["Nq_inferred_absolute_relative_error_NOT_qualification"]
        inferred_nq_error_ratio = (
            fine_nq_error / coarse_nq_error
            if coarse_nq_error is not None
            and fine_nq_error is not None
            and coarse_nq_error > 0.0
            else None
        )
        inferred_nq_slope = (
            math.log(coarse_nq_error / fine_nq_error) / math.log(resolution_ratio)
            if coarse_nq_error is not None
            and fine_nq_error is not None
            and coarse_nq_error > 0.0
            and fine_nq_error > 0.0
            else None
        )
        qualifies = bool(
            coarse["rate_plateau_resolved"] and fine["rate_plateau_resolved"]
        )
        pairs.append(
            {
                "coarse_resolution_full_width": coarse["resolution_full_width"],
                "fine_resolution_full_width": fine["resolution_full_width"],
                "resolution_ratio": resolution_ratio,
                "measured_Nc_fine_over_coarse": fine_nc / coarse_nc
                if valid_values and coarse_nc != 0.0
                else None,
                "successive_resolution_relative_gap": gap,
                "Nc_error_fine_over_coarse": error_ratio,
                "Nc_observed_reference_error_slope": slope,
                "Nc_reference_error_decreased": fine_error < coarse_error
                if coarse_error is not None and fine_error is not None
                else None,
                "pair_rate_plateau_resolved": qualifies,
                "interpretation": "resolved two-rate plateau observations"
                if qualifies
                else "diagnostic only: unresolved plateau/rate or numerical failure; not observed capacity mesh convergence",
                "Nq_slope_status": "no independent Nq measurement; inferred Nq errors inherit Nc algebraically (phi0 has identically zero error)",
                "Nq_inferred_error_fine_over_coarse_NOT_independent": inferred_nq_error_ratio,
                "Nq_inferred_reference_error_slope_NOT_independent": inferred_nq_slope,
            }
        )
    finest = levels[-1] if levels else None
    last_pair = pairs[-1] if pairs else None
    gap = (
        last_pair["successive_resolution_relative_gap"] if last_pair is not None else None
    )
    finest_resolved = finest is not None and finest["rate_plateau_resolved"]
    mesh_gap_resolved = gap is not None and gap <= 0.02
    comparable = bool(all_successful and finest_resolved and mesh_gap_resolved)
    reduced = bool(
        pairs and all(pair["Nc_reference_error_decreased"] is True for pair in pairs)
    )
    trend_resolved = bool(
        all_successful
        and pairs
        and all(pair["pair_rate_plateau_resolved"] for pair in pairs)
    )
    return {
        "analysis_kind": "host-only comparative diagnostic; not a new solver observation or replacement qualification evidence",
        "reference_Nc": reference_nc,
        "reference_Nq": reference_nq,
        "levels": levels,
        "pairs": pairs,
        "all_parts_solver_successful": all_successful,
        "finest_two_rate_plateau_resolved": bool(finest_resolved),
        "finest_measured_mesh_gap_within_candidate_limit": mesh_gap_resolved,
        "candidate_mesh_rate_plateau_gates_resolved": comparable,
        "measured_reference_error_reduction_over_all_pairs": reduced if pairs else None,
        "resolved_capacity_reference_error_reduction_observed": bool(
            trend_resolved and reduced
        ),
        "convergence_claim": "measured reference-error reduction trend only; no asymptotic convergence order or zero-bias limit is asserted",
        "Nc_candidate_error_within_0_02": bool(
            finest is not None
            and finest["Nc_absolute_relative_error"] is not None
            and finest["Nc_absolute_relative_error"] <= 0.02
        ),
        "Nq_independently_measured": False,
        "Nq_independently_qualified": False,
        "Nq_derivation": "Nq_inferred=Nc_reaction*tan(phi)+1; phi0 gives identically1, not an independently observed agreement",
        "limitations": "pairwise slopes use measured reference errors, not Richardson extrapolation or an assumed order; no plateau/rate failure is promoted to mesh convergence; finite-domain bias remains unseparated; keep every causal campaign part unchanged",
    }


_BG_HEIGHT = 0.08


_BG_GRAVITY = 9.81


_BG_PACKING = 0.6


_BG_GRAIN_DENSITY = 2500.0


_BG_DIAMETER = 0.001


_BG_SLOPE = 0.5


_BG_MU_S = 0.383


_BG_MU_2 = 0.643


_BG_I_0 = 0.279


def bagnold_solver_settings() -> dict[str, Any]:
    """Expose complete MPM support and limits before campaign execution."""
    return {
        "runner": "run_bagnold_inclined_plane",
        "resolutions": list(_BG_MPM_RESOLUTIONS),
        "resolution_quantity": "initial liquid depth cells; two y samples per cell",
        "support": "public MuIMPMConstitutivePlan and PreparedImplicitMPMDynamics full backward-Euler grid transaction; 3-D extruded plane-strain layer periodic in x/z, three periodic nodes each for complete quadratic stencils; natural zero-traction particle free surface; rough stationary y=0 essential no-slip base; initial transient linear shear and zero shear stress; hydrostatic physical pressure encoded only in admitted initial elastic metric history",
        "public_apis": [
            "MuIMPMConstitutivePlan",
            "MuIParameters",
            "TensorGridPlan",
            "ParticleGridSplatPlan",
            "TensorBSplineSplatAssignment",
            "ParticleSetPlan",
            "MPMParticleDomainPlan",
            "PrescribedGridVelocityPlan",
            "compile_material_point_problem",
            "CompiledMaterialPointProblem.initialize_state",
            "ImplicitMPMMethodPlan",
            "PreparedImplicitMPMDynamics.step_detailed",
            "PreparedImplicitMPMDynamics.prepare_residual",
            "NewtonKrylov",
            "GMRES",
            "LinearSolvePolicy",
            "NonlinearTermination",
        ],
        "constants": {
            "mu_s": _BG_MU_S,
            "mu_2": _BG_MU_2,
            "I_0": _BG_I_0,
            "grain_diameter_m": _BG_DIAMETER,
            "grain_density_kg_m3": _BG_GRAIN_DENSITY,
            "packing_fraction": _BG_PACKING,
            "bulk_reference_density_kg_m3": _BG_PACKING * _BG_GRAIN_DENSITY,
            "initial_depth_m": _BG_HEIGHT,
            "gravity_m_s2": _BG_GRAVITY,
            "tan_inclination": _BG_SLOPE,
            "shear_modulus_pa": _BG_MPM_SHEAR,
            "bulk_modulus_pa": _BG_MPM_BULK,
            "periodic_x_length_m": _BG_MPM_PERIODIC_LENGTH,
            "periodic_z_length_m": _BG_MPM_PERIODIC_LENGTH,
            "initial_velocity": "u=sqrt(g/H)*y, v=w=0; non-reference linear transient shear",
            "initial_shear_rate_s_inv": sqrt(_BG_GRAVITY / _BG_HEIGHT),
            "maximum_initial_pressure_over_G": _BG_PACKING
            * _BG_GRAIN_DENSITY
            * _BG_GRAVITY
            * _BG_HEIGHT
            / sqrt(1.0 + _BG_SLOPE**2)
            / _BG_MPM_SHEAR,
            "initial_shear_stress_pa": 0.0,
        },
        "domain": {
            "axes": ["x downslope", "y outward normal", "z periodic extrusion"],
            "grid": "3 x (N+7) x 3 vertices; y in [-3dy,H+3dy], x/z periodic",
            "particle_domain": "[0,Lx] x [0,H+dy] x [0,Lz], periodic x/z; y support halo 2dy",
            "assignment": "quadratic tensor B-spline, boundary=reject; complete support halo",
            "quadrature": "one cell-center x/z sample per periodic cell; two y samples at offsets .25/.75; 18N particles",
            "base": "all velocity components zero on nodes y<=0; v_z=0 on every node for plane strain",
            "top": "unloaded natural material-particle free surface; no top pressure or velocity prescription",
            "initial_metric": "public initialize_state_from_stress(-p0*I, MuIParameters), p0=rho_bulk*g*cos(theta)*(H-y_particle); packed Be(6), row-major F_previous(9)=I; 15-component relative-deformation history",
            "stress_convention": "tension-positive Kirchhoff tau=P*F^T; compression-positive Cauchy p=-tr(tau)/(3J)",
            "rate_normalization": "t=sqrt(s:s/2)=q/sqrt(3); r=sqrt(2*Dp:Dp); I=r*d*sqrt(rho_s/p_cauchy)",
            "no_seed": "initial hydrostatic compression and non-reference linear transient velocity; zero initial deviatoric stress; no analytic steady profile or shear-stress solution injection",
        },
        "method": {
            "scheme": "public backward-Euler implicit MPM; default APIC; no volumetric stabilization",
            "assignment_degree": 2,
            "apic_maximum_condition": 100000000.0,
            "maximum_dt_s": _BG_MPM_MAX_DT,
            "directional_displacement_fraction": _BG_MPM_DISPLACEMENT_FRACTION,
            "maximum_newton_steps": _BG_MPM_MAX_NEWTON,
            "maximum_predicted_shear_increment": _BG_MPM_PREDICTED_SHEAR_FRACTION
            * _BG_MPM_MAX_SHEAR_INCREMENT,
            "shear_increment_control": "dt*max sqrt(2*sym(C):sym(C)) <=0.1 using accepted APIC affine gradient; actual candidate increment reported, not repaired",
            "maximum_krylov_iterations": _BG_MPM_MAX_KRYLOV,
            "gmres_restart": 32,
            "newton_absolute_residual": "force-relative threshold times initial particle-body-force L2 norm",
            "newton_force_relative_threshold": _BG_MPM_NEWTON_FORCE_RELATIVE,
            "initial_force_norm_N_per_level": [
                (
                    _BG_PACKING
                    * _BG_GRAIN_DENSITY
                    * _BG_MPM_PERIODIC_LENGTH**2
                    * _BG_HEIGHT
                    * _BG_GRAVITY
                )
                / sqrt(2 * _BG_MPM_PERIODIC_NODES**2 * n)
                for n in _BG_MPM_RESOLUTIONS
            ],
            "newton_relative_residual": _BG_MPM_NEWTON_RELATIVE,
            "krylov_absolute_residual": _BG_MPM_KRYLOV_ABSOLUTE,
            "krylov_relative_residual": _BG_MPM_KRYLOV_RELATIVE,
            "explicit_restrictions": "explicit substrate only prepares storage/transfer; actual implicit owner does not enforce elastic-wave or explicit plastic-relaxation CFL",
            "retry": "none; first rejected transaction stops the level",
            "maximum_candidate_shear_increment": _BG_MPM_MAX_SHEAR_INCREMENT,
        },
        "stopping": {
            "minimum_gravity_times": _BG_MPM_MIN_GRAVITY_TIMES,
            "maximum_gravity_times": _BG_MPM_MAX_GRAVITY_TIMES,
            "maximum_physical_time_s": _BG_MPM_MAX_GRAVITY_TIMES
            * sqrt(_BG_HEIGHT / _BG_GRAVITY),
            "observation_window_gravity_times": 1.0,
            "consecutive_steady_windows": _BG_MPM_STEADY_WINDOWS,
            "profile_relative_change": _BG_MPM_PROFILE_GAP,
            "acceleration_over_g": _BG_MPM_ACCELERATION_GAP,
            "kinetic_relative_change": _BG_MPM_KINETIC_GAP,
            "transverse_kinetic_fraction": _BG_MPM_TRANSVERSE_FRACTION,
            "maximum_elapsed_wall_s_per_level": _BG_MPM_MAX_ELAPSED_S,
            "wall_bound_semantics": "checked at observation-block boundaries; one block/compilation may overrun, reported honestly",
            "horizon_rationale": "original dimensional yielded startup about7-8s and tangent viscous relaxation about8.5s; maximum about69s (768 gravity times), with early measured steadiness authoritative",
        },
        "references": [
            {
                "title": "GDR MiDi (2004), On dense granular flows",
                "doi": "10.1140/epje/i2003-10153-0",
                "url": "https://doi.org/10.1140/epje/i2003-10153-0",
                "used_for": "Eqs. (22)-(26), independent Bagnold continuum reference",
            },
            {
                "title": "Jop, Forterre and Pouliquen (2006), A constitutive law for dense granular flows",
                "doi": "10.1038/nature04801",
                "url": "https://doi.org/10.1038/nature04801",
                "used_for": "Eqs. (1)-(3), original unregularized mu(I) constants",
            },
            {
                "title": "Dunatunga and Kamrin (2015), Simulating the dynamics of dense granular flows with a continuum model",
                "doi": "10.1017/jfm.2015.383",
                "url": "https://arxiv.org/abs/1411.5447",
                "used_for": "S11b t and plastic shear-rate conventions; material is multiplicative Hencky, not the paper's Jaumann implementation",
            },
        ],
        "reference": {
            "profile": "same original independent host Bagnold continuum formula and intrinsic grain-density normalization",
            "sampling": "x/z-averaged admitted solver nodal u at y=0,dy,...,H",
            "criterion": "profile-relative-l2-error <= 0.02, candidate unchanged",
            "model_caveat": "finite compressibility and elastic-viscoplastic realization recorded, not fitted or suppressed",
        },
        "resource_bounds": {
            "levels": len(_BG_MPM_RESOLUTIONS),
            "maximum_particles": 2 * _BG_MPM_PERIODIC_NODES**2 * max(_BG_MPM_RESOLUTIONS),
            "maximum_grid_nodes": _BG_MPM_PERIODIC_NODES**2
            * (max(_BG_MPM_RESOLUTIONS) + 7),
            "maximum_grid_unknowns": 3
            * _BG_MPM_PERIODIC_NODES**2
            * (max(_BG_MPM_RESOLUTIONS) + 7),
            "maximum_steps_per_level": _BG_MPM_MAX_STEPS,
            "maximum_total_steps": len(_BG_MPM_RESOLUTIONS) * _BG_MPM_MAX_STEPS,
            "maximum_steps_per_observation_block": _BG_MPM_BLOCK_STEPS,
            "maximum_state_bytes": _BG_MPM_STATE_BYTES,
            "maximum_workspace_bytes": _BG_MPM_WORKSPACE_BYTES,
        },
        "diagnostics": "law-only helpers remain explicitly invoked diagnostics; prior full-FLIP evidence preserved in repro/history artifacts; neither is a primary fallback",
    }


_BG_MPM_RESOLUTIONS = (8, 16, 32)


_BG_MPM_SHEAR = 200000.0


_BG_MPM_BULK = 1000000.0


_BG_MPM_PERIODIC_NODES = 3


_BG_MPM_PERIODIC_LENGTH = 0.08


_BG_MPM_MAX_STEPS = 131072


_BG_MPM_BLOCK_STEPS = 32


_BG_MPM_MAX_DT = 0.001


_BG_MPM_MAX_SHEAR_INCREMENT = 0.1


_BG_MPM_DISPLACEMENT_FRACTION = 0.25


_BG_MPM_MAX_ELAPSED_S = 1800.0


_BG_MPM_STATE_BYTES = 32 * 1024**2


_BG_MPM_WORKSPACE_BYTES = 128 * 1024**2


_BG_MPM_MAX_NEWTON = 25


_BG_MPM_MAX_KRYLOV = 1024


_BG_MPM_NEWTON_FORCE_RELATIVE = 1.0e-6


_BG_MPM_NEWTON_RELATIVE = 1.0e-6


_BG_MPM_KRYLOV_ABSOLUTE = 1.0e-11


_BG_MPM_KRYLOV_RELATIVE = 1.0e-6


_BG_MPM_MIN_GRAVITY_TIMES = 64.0


_BG_MPM_MAX_GRAVITY_TIMES = 768.0


_BG_MPM_STEADY_WINDOWS = 3


_BG_MPM_PROFILE_GAP = 1.0e-4


_BG_MPM_ACCELERATION_GAP = 1.0e-4


_BG_MPM_KINETIC_GAP = 1.0e-4


_BG_MPM_TRANSVERSE_FRACTION = 1.0e-6


_BG_MPM_PREDICTED_SHEAR_FRACTION = 0.8


def _bg_mpm_gravity(
    time: Array, position: Array, velocity: Array, arguments: None
) -> Array:
    del time, velocity, arguments
    cosine = 1.0 / sqrt(1.0 + _BG_SLOPE**2)
    return jnp.broadcast_to(
        jnp.asarray(
            [_BG_GRAVITY * _BG_SLOPE * cosine, -_BG_GRAVITY * cosine, 0.0],
            dtype=position.dtype,
        ),
        position.shape,
    )


@eqx.filter_jit
def _bg_mpm_advance_block(
    dynamics: phx.solver.PreparedImplicitMPMDynamics,
    state: phx.discretization.MPMRuntimeState,
    arguments: phx.equations.MaterialPointArguments,
    previous_grid: Array,
    widths: Array,
    end_time: Array,
    remaining_steps: Array,
) -> tuple[
    phx.discretization.MPMRuntimeState, Array, Array, Array, Array, Array, Array, Array
]:
    codes = jnp.zeros((7,), dtype=jnp.int32)
    evidence = jnp.zeros((7,), dtype=jnp.float64).at[3].set(jnp.inf)
    initial = (
        state,
        previous_grid,
        evidence,
        codes,
        jnp.asarray(0, dtype=jnp.int32),
        state.particles.material_state,
        state.particles.deformation_gradient,
        jnp.asarray(0.0, dtype=jnp.float64),
    )

    def condition(
        carry: tuple[
            phx.discretization.MPMRuntimeState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> Array:
        current, _, _, status, attempts, _, _, _ = carry
        roundoff = 8.0 * jnp.finfo(jnp.float64).eps * jnp.abs(end_time)
        return (
            (end_time - current.time > roundoff)
            & (attempts < _BG_MPM_BLOCK_STEPS)
            & (attempts < remaining_steps)
            & (status[0] == 0)
            & (status[1] == 0)
        )

    def body(
        carry: tuple[
            phx.discretization.MPMRuntimeState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[
        phx.discretization.MPMRuntimeState,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        current, old_grid, maxima, _, attempts, _, _, _ = carry
        speed = jnp.max(jnp.abs(current.particles.velocity), axis=0)
        cosine = 1.0 / sqrt(1.0 + _BG_SLOPE**2)
        allowance = (
            jnp.asarray(
                [_BG_GRAVITY * _BG_SLOPE * cosine, _BG_GRAVITY * cosine, 0.0],
                dtype=jnp.float64,
            )
            * _BG_MPM_MAX_DT
        )
        speed_bound = speed + allowance
        motion_dt = (
            jnp.min(
                jnp.where(
                    speed_bound > 0.0,
                    widths / jnp.where(speed_bound > 0.0, speed_bound, 1.0),
                    jnp.inf,
                )
            )
            * _BG_MPM_DISPLACEMENT_FRACTION
        )
        symmetric_gradient = 0.5 * (
            current.particles.affine_velocity
            + jnp.swapaxes(current.particles.affine_velocity, -1, -2)
        )
        maximum_rate = jnp.max(
            jnp.sqrt(2.0 * jnp.sum(symmetric_gradient**2, axis=(-2, -1)))
        )
        rate_dt = jnp.where(
            maximum_rate > 0.0,
            _BG_MPM_PREDICTED_SHEAR_FRACTION
            * _BG_MPM_MAX_SHEAR_INCREMENT
            / jnp.where(maximum_rate > 0.0, maximum_rate, 1.0),
            jnp.inf,
        )
        dt = jnp.minimum(
            end_time - current.time,
            jnp.minimum(_BG_MPM_MAX_DT, jnp.minimum(motion_dt, rate_dt)),
        )
        detail = dynamics.step_detailed(current, dt, arguments)
        candidate = detail.candidate_state.particles
        support_changed = ~jnp.all(
            dynamics.explicit.particle_domain.contains(candidate.position)
        )
        displacement_fraction = jnp.max(
            jnp.abs(candidate.position - current.particles.position) / widths
        )
        candidate_symmetric_gradient = 0.5 * (
            candidate.affine_velocity + jnp.swapaxes(candidate.affine_velocity, -1, -2)
        )
        candidate_rate = jnp.max(
            jnp.sqrt(2.0 * jnp.sum(candidate_symmetric_gradient**2, axis=(-2, -1)))
        )
        new_values = jnp.stack(
            (
                detail.diagnostics.residual_norm,
                detail.diagnostics.nonlinear_steps.astype(jnp.float64),
                detail.diagnostics.linear_iterations.astype(jnp.float64),
                detail.diagnostics.minimum_jacobian,
                displacement_fraction,
                dt * maximum_rate,
                dt * candidate_rate,
            )
        )
        aggregate = (
            jnp.maximum(maxima, new_values)
            .at[3]
            .set(jnp.minimum(maxima[3], new_values[3]))
        )
        status = jnp.stack(
            (
                (~detail.successful).astype(jnp.int32),
                support_changed.astype(jnp.int32),
                detail.diagnostics.nonlinear_status.astype(jnp.int32),
                detail.diagnostics.material_successful.astype(jnp.int32),
                detail.diagnostics.tangent_successful.astype(jnp.int32),
                detail.diagnostics.finite.astype(jnp.int32),
                detail.accepted_state.last_status,
            )
        )
        return (
            detail.accepted_state,
            jnp.where(detail.successful, detail.grid.velocity_after[0], old_grid),
            aggregate,
            status,
            attempts + 1,
            current.particles.material_state,
            candidate.deformation_gradient,
            dt,
        )

    return jax.lax.while_loop(condition, body, initial)


def _bg_mpm_trial_failure(
    history: Array,
    candidate_f: Array,
    positions: NDArray[np.float64],
    material_diagnostics: Mapping[str, Array],
    material_successful: NDArray[np.bool_],
    cells: int,
) -> list[dict[str, Any]]:
    """Expose the actual last failed trial; never infer failure from a large norm."""
    indices = np.flatnonzero(~material_successful)
    if indices.size == 0:
        return []
    previous = history[:, 6:15].reshape((-1, 3, 3))
    solve_plan = phx.linalg.SmallLinearSolvePlan(3, singular_tolerance=1e-15)
    relative = phx.linalg.solve_small_linear(
        solve_plan, jnp.swapaxes(previous, -1, -2), jnp.swapaxes(candidate_f, -1, -2)
    )
    current = phx.linalg.solve_small_linear(
        solve_plan, candidate_f, jnp.zeros_like(candidate_f)
    )
    previous_host, candidate_host, relative_host, current_host, diagnostics_host = (
        jax.device_get((previous, candidate_f, relative, current, material_diagnostics))
    )
    records: list[dict[str, Any]] = []
    for index in indices[:8]:
        previous_matrix = np.asarray(previous_host[index], dtype=np.float64)
        candidate_matrix = np.asarray(candidate_host[index], dtype=np.float64)
        diagnostic_values: dict[str, Any] = {}
        for name, values in diagnostics_host.items():
            value = np.asarray(values)[index]
            if value.ndim == 0:
                scalar = value.item()
                diagnostic_values[name] = (
                    _dg_finite(scalar) if isinstance(scalar, float) else scalar
                )
        scale = float(np.max(np.abs(previous_matrix)))
        candidate_scale = float(np.max(np.abs(candidate_matrix)))
        determinant = float(relative_host.determinant[index])
        candidate_determinant = float(current_host.determinant[index])
        initial_y = (
            ((index // _BG_MPM_PERIODIC_NODES) % (2 * cells) + 0.5)
            * (_BG_HEIGHT / cells)
            / 2
        )
        records.append(
            {
                "particle_index": int(index),
                "current_position_m": positions[index].tolist(),
                "initial_y_m": float(initial_y),
                "current_y_over_H": float(positions[index, 1] / _BG_HEIGHT),
                "gamma_F_xy": _dg_finite(float(previous_matrix[0, 1])),
                "gamma_F_xy_over_F_yy": _dg_finite(
                    float(previous_matrix[0, 1] / previous_matrix[1, 1])
                ),
                "gamma_note": "total-F shear components, not a fitted integral or assumption of pure simple shear; full F is recorded",
                "previous_F": previous_matrix.tolist(),
                "candidate_F": candidate_matrix.tolist(),
                "previous_F_condition_2": _dg_finite(
                    float(np.linalg.cond(previous_matrix))
                ),
                "previous_F_determinant": _dg_finite(determinant),
                "previous_F_scaled_determinant": _dg_finite(determinant / scale**3)
                if scale > 0
                else None,
                "relative_small_solve_status": int(relative_host.status[index]),
                "relative_small_solve_successful": bool(relative_host.successful[index]),
                "relative_small_solve_condition_inf": _dg_finite(
                    float(relative_host.condition_estimate[index])
                ),
                "relative_small_solve_residual": _dg_finite(
                    float(relative_host.residual_norm[index])
                ),
                "candidate_small_solve_status": int(current_host.status[index]),
                "candidate_small_solve_successful": bool(current_host.successful[index]),
                "candidate_F_determinant": _dg_finite(candidate_determinant),
                "candidate_F_scaled_determinant": _dg_finite(
                    candidate_determinant / candidate_scale**3
                )
                if candidate_scale > 0
                else None,
                "native_policy": {
                    "singular_tolerance": 1e-15,
                    "maximum_condition": 1e12,
                    "refinement_iterations": 1,
                    "plan_id": solve_plan.plan_id,
                },
                "material_diagnostics": diagnostic_values,
            }
        )
    return records


def _bg_mpm_level(cells: int, parameters: MuIParameters) -> dict[str, Any]:
    started = perf_counter()
    dy = _BG_HEIGHT / cells
    periodic_spacing = _BG_MPM_PERIODIC_LENGTH / _BG_MPM_PERIODIC_NODES
    gravity_time = sqrt(_BG_HEIGHT / _BG_GRAVITY)
    rho = _BG_PACKING * _BG_GRAIN_DENSITY
    cosine = 1.0 / sqrt(1.0 + _BG_SLOPE**2)
    xz = (np.arange(_BG_MPM_PERIODIC_NODES, dtype=np.float64) + 0.5) * periodic_spacing
    particle_y = (np.arange(2 * cells, dtype=np.float64) + 0.5) * dy / 2.0
    xx, yy, zz = np.meshgrid(xz, particle_y, xz, indexing="ij")
    host_position = np.stack((xx, yy, zz), axis=-1).reshape((-1, 3))
    position = jnp.asarray(host_position, dtype=jnp.float64)
    volume = jnp.full(
        (position.shape[0],), periodic_spacing**2 * dy / 2.0, dtype=jnp.float64
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), rho * volume, ambient_dimension=3
    ).prepare()
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                _BG_MPM_PERIODIC_NODES, periodic=True, endpoint=False
            ),
            phx.discretization.UniformAxisSpec(cells + 7, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(
                _BG_MPM_PERIODIC_NODES, periodic=True, endpoint=False
            ),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            [
                [0.0, -3.0 * dy, 0.0],
                [_BG_MPM_PERIODIC_LENGTH, _BG_HEIGHT + 3.0 * dy, _BG_MPM_PERIODIC_LENGTH],
            ],
            dtype=jnp.float64,
        )
    )
    coordinates = np.stack(
        np.meshgrid(
            np.arange(_BG_MPM_PERIODIC_NODES, dtype=np.float64) * periodic_spacing,
            np.linspace(-3.0 * dy, _BG_HEIGHT + 3.0 * dy, cells + 7, dtype=np.float64),
            np.arange(_BG_MPM_PERIODIC_NODES, dtype=np.float64) * periodic_spacing,
            indexing="ij",
        ),
        axis=-1,
    )
    mask = np.zeros(coordinates.shape, dtype=np.bool_)
    mask[:, :4, :, :] = True
    mask[..., 2] = True
    boundary = phx.discretization.PrescribedGridVelocityPlan(
        mask, np.zeros(coordinates.shape, dtype=np.float64)
    )
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    material = phx.applications.solid_mechanics.MuIMPMConstitutivePlan(
        jnp.asarray(_BG_MPM_SHEAR, dtype=jnp.float64),
        jnp.asarray(_BG_MPM_BULK, dtype=jnp.float64),
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "s16-bagnold-coupled-mu-i-mpm",
            material,
            external_acceleration=_bg_mpm_gravity,
            external_acceleration_id="inclined-gravity-original-constants",
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [_BG_MPM_PERIODIC_LENGTH, _BG_HEIGHT + dy, _BG_MPM_PERIODIC_LENGTH],
                ],
                dtype=jnp.float64,
            ),
            periodic=(True, False, True),
            support_margin=(0.0, 2.0 * dy, 0.0),
        ),
        boundary=boundary,
        resource_policy=phx.discretization.MPMResourcePolicy(
            maximum_state_bytes=_BG_MPM_STATE_BYTES,
            maximum_step_workspace_bytes=_BG_MPM_WORKSPACE_BYTES,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    p0 = rho * _BG_GRAVITY * cosine * (_BG_HEIGHT - position[:, 1])
    initial_stress = -p0[:, None, None] * jnp.eye(3, dtype=jnp.float64)[None, :, :]
    prestress = material.initialize_state_from_stress(initial_stress, parameters)
    if not bool(jax.device_get(jnp.all(prestress.admissible))):
        raise ValueError("Initial hydrostatic mu(I) MPM stress is inadmissible.")
    history = prestress.state
    initial_velocity = (
        jnp.zeros_like(position)
        .at[:, 0]
        .set(sqrt(_BG_GRAVITY / _BG_HEIGHT) * position[:, 1])
    )
    initial_affine = (
        jnp.zeros((position.shape[0], 3, 3), dtype=jnp.float64)
        .at[:, 0, 1]
        .set(sqrt(_BG_GRAVITY / _BG_HEIGHT))
    )
    state = compiled.initialize_state(
        position,
        initial_velocity,
        volume,
        arguments,
        affine_velocity=initial_affine,
        material_state=history,
    )
    initial_energy = float(
        jax.device_get(0.5 * jnp.sum(rho * volume[:, None] * state.particles.velocity**2))
    )
    force_scale = (
        rho
        * _BG_MPM_PERIODIC_LENGTH**2
        * _BG_HEIGHT
        * _BG_GRAVITY
        / sqrt(position.shape[0])
    )
    newton = phx.nonlinear.NewtonKrylov(
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=32),
            tolerance=phx.linalg.TolerancePolicy(
                relative=_BG_MPM_KRYLOV_RELATIVE,
                absolute=_BG_MPM_KRYLOV_ABSOLUTE,
                max_steps=_BG_MPM_MAX_KRYLOV,
            ),
        )
    )
    implicit = phx.solver.PreparedImplicitMPMDynamics(
        compiled.dynamics,
        phx.solver.ImplicitMPMMethodPlan(
            newton,
            phx.nonlinear.NonlinearTermination(
                absolute_residual=_BG_MPM_NEWTON_FORCE_RELATIVE * force_scale,
                relative_residual=_BG_MPM_NEWTON_RELATIVE,
                maximum_steps=_BG_MPM_MAX_NEWTON,
                maximum_linear_iterations=_BG_MPM_MAX_KRYLOV,
            ),
        ),
    )
    grid_velocity = implicit.prepare_residual(
        state, _BG_MPM_MAX_DT, arguments
    ).initial_velocity
    attempts = 0
    physical_time = 0.0
    horizon = _BG_MPM_MAX_GRAVITY_TIMES * gravity_time
    horizon_tolerance = 8.0 * np.finfo(np.float64).eps * max(abs(horizon), gravity_time)
    initial_nodal = np.mean(
        np.asarray(jax.device_get(grid_velocity), dtype=np.float64), axis=(0, 2)
    )
    previous_profile = initial_nodal[3 : cells + 4, 0]
    previous_energy = initial_energy
    profile_gap: float | None = None
    acceleration_gap: float | None = None
    kinetic_gap: float | None = None
    transverse_fraction: float | None = None
    steady_windows = 0
    windows = 0
    rejected = False
    support_valid = True
    steady = False
    wall_limit = False
    codes = np.zeros((7,), dtype=np.int32)
    maxima = np.zeros((7,), dtype=np.float64)
    maxima[3] = np.inf
    samples: list[dict[str, Any]] = []
    last_history = state.particles.material_state
    last_candidate_f = state.particles.deformation_gradient
    last_dt = jnp.asarray(0.0, dtype=jnp.float64)
    while horizon - physical_time > horizon_tolerance and attempts < _BG_MPM_MAX_STEPS:
        window_start = physical_time
        window_end = min(window_start + gravity_time, horizon)
        while window_end - physical_time > 8.0 * np.finfo(np.float64).eps * abs(
            window_end
        ):
            (
                state,
                grid_velocity,
                device_evidence,
                device_codes,
                device_attempts,
                last_history,
                last_candidate_f,
                last_dt,
            ) = _bg_mpm_advance_block(
                implicit,
                state,
                arguments,
                grid_velocity,
                jnp.asarray([periodic_spacing, dy, periodic_spacing], dtype=jnp.float64),
                jnp.asarray(window_end, dtype=jnp.float64),
                jnp.asarray(_BG_MPM_MAX_STEPS - attempts, dtype=jnp.int32),
            )
            observed_time, observed_evidence, observed_codes, observed_attempts = (
                jax.device_get(
                    (state.time, device_evidence, device_codes, device_attempts)
                )
            )
            physical_time = float(observed_time)
            attempts += int(observed_attempts)
            values = np.asarray(observed_evidence, dtype=np.float64)
            minimum_j = min(float(maxima[3]), float(values[3]))
            maxima = np.maximum(maxima, values)
            maxima[3] = minimum_j
            codes = np.asarray(observed_codes, dtype=np.int32)
            rejected = bool(codes[0])
            support_valid = not bool(codes[1])
            wall_limit = perf_counter() - started >= _BG_MPM_MAX_ELAPSED_S
            if (
                rejected
                or not support_valid
                or wall_limit
                or (attempts >= _BG_MPM_MAX_STEPS)
                or (int(observed_attempts) == 0)
            ):
                break
        if rejected or not support_valid:
            break
        host_grid, host_velocity = jax.device_get(
            (grid_velocity, state.particles.velocity)
        )
        nodal = np.mean(np.asarray(host_grid, dtype=np.float64), axis=(0, 2))
        profile = nodal[3 : cells + 4, 0]
        velocities = np.asarray(host_velocity, dtype=np.float64)
        energy = 0.5 * rho * periodic_spacing**2 * dy / 2.0 * float(np.sum(velocities**2))
        norm = float(np.linalg.norm(profile))
        elapsed = physical_time - window_start
        difference = float(np.linalg.norm(profile - previous_profile))
        profile_gap = _dg_finite(difference / norm) if norm > 0.0 else None
        acceleration_gap = (
            _dg_finite(difference / (sqrt(cells + 1) * _BG_GRAVITY * elapsed))
            if elapsed > 0.0
            else None
        )
        kinetic_gap = (
            _dg_finite(abs(energy - previous_energy) / energy) if energy > 0.0 else None
        )
        total_squared_speed = float(np.sum(velocities**2))
        transverse_fraction = (
            _dg_finite(float(np.sum(velocities[:, 1:] ** 2)) / total_squared_speed)
            if total_squared_speed > 0.0
            else None
        )
        windows += 1
        observed_steady = (
            physical_time >= _BG_MPM_MIN_GRAVITY_TIMES * gravity_time
            and elapsed >= 0.99 * gravity_time
            and (profile_gap is not None)
            and (profile_gap <= _BG_MPM_PROFILE_GAP)
            and (acceleration_gap is not None)
            and (acceleration_gap <= _BG_MPM_ACCELERATION_GAP)
            and (kinetic_gap is not None)
            and (kinetic_gap <= _BG_MPM_KINETIC_GAP)
            and (transverse_fraction is not None)
            and (transverse_fraction <= _BG_MPM_TRANSVERSE_FRACTION)
        )
        steady_windows = steady_windows + 1 if observed_steady else 0
        if windows == 1 or windows % 16 == 0 or observed_steady:
            samples.append(
                {
                    "time_s": physical_time,
                    "profile_relative_change": profile_gap,
                    "acceleration_over_g": acceleration_gap,
                    "kinetic_relative_change": kinetic_gap,
                    "kinetic_energy_j": _dg_finite(energy),
                }
            )
        previous_profile, previous_energy = (profile, energy)
        if steady_windows >= _BG_MPM_STEADY_WINDOWS:
            steady = True
            break
        if wall_limit or attempts >= _BG_MPM_MAX_STEPS:
            break
    host_grid, host_position, host_velocity, host_f, host_p, host_history = (
        jax.device_get(
            (
                grid_velocity,
                state.particles.position,
                state.particles.velocity,
                state.particles.deformation_gradient,
                state.particles.first_piola,
                state.particles.material_state,
            )
        )
    )
    final_grid = np.asarray(host_grid, dtype=np.float64)
    profile = np.mean(final_grid, axis=(0, 2))[3 : cells + 4, 0]
    y_nodes = np.linspace(0.0, _BG_HEIGHT, cells + 1, dtype=np.float64)
    reference_i = _BG_I_0 * (_BG_SLOPE - _BG_MU_S) / (_BG_MU_2 - _BG_SLOPE)
    coefficient = reference_i * sqrt(_BG_PACKING * _BG_GRAVITY * cosine) / _BG_DIAMETER
    reference = (
        2.0 / 3.0 * coefficient * (_BG_HEIGHT**1.5 - (_BG_HEIGHT - y_nodes) ** 1.5)
    )
    profile_error = _dg_finite(
        float(np.linalg.norm(profile - reference) / np.linalg.norm(reference))
    )
    flux = dy * float(np.sum((profile[:-1] + profile[1:]) * 0.5))
    reference_flux = 2.0 / 5.0 * coefficient * _BG_HEIGHT**2.5
    deformation = np.asarray(host_f, dtype=np.float64)
    metric_entries = np.asarray(host_history, dtype=np.float64)
    metric = np.zeros((position.shape[0], 3, 3), dtype=np.float64)
    metric[:, 0, 0] = metric_entries[:, 0]
    metric[:, 1, 1] = metric_entries[:, 1]
    metric[:, 2, 2] = metric_entries[:, 2]
    metric[:, 1, 2] = metric[:, 2, 1] = metric_entries[:, 3]
    metric[:, 0, 2] = metric[:, 2, 0] = metric_entries[:, 4]
    metric[:, 0, 1] = metric[:, 1, 0] = metric_entries[:, 5]
    metric_eigenvalues = np.linalg.eigvalsh(metric)
    metric_condition = (
        _dg_finite(float(np.max(metric_eigenvalues[:, -1] / metric_eigenvalues[:, 0])))
        if np.all(metric_eigenvalues[:, 0] > 0.0)
        else None
    )
    deformation_condition = _dg_finite(float(np.max(np.linalg.cond(deformation))))
    piola = np.asarray(host_p, dtype=np.float64)
    jacobian = np.linalg.det(deformation)
    kirchhoff = piola @ np.swapaxes(deformation, -1, -2)
    cauchy = kirchhoff / jacobian[:, None, None]
    pressure = -np.trace(cauchy, axis1=-2, axis2=-1) / 3.0
    deviator = cauchy + pressure[:, None, None] * np.eye(3, dtype=np.float64)
    shear_magnitude = np.sqrt(np.sum(deviator**2, axis=(-2, -1)) / 2.0)
    pressure_reference = (
        rho
        * _BG_GRAVITY
        * cosine
        * (_BG_HEIGHT - np.asarray(host_position, dtype=np.float64)[:, 1])
    )
    pressure_error = _dg_finite(
        float(
            np.linalg.norm(pressure - pressure_reference)
            / np.linalg.norm(pressure_reference)
        )
    )
    current_height_from_volume = (
        float(np.sum(jacobian))
        * periodic_spacing**2
        * dy
        / 2.0
        / _BG_MPM_PERIODIC_LENGTH**2
    )
    final_energy = (
        0.5
        * rho
        * periodic_spacing**2
        * dy
        / 2.0
        * float(np.sum(np.asarray(host_velocity, dtype=np.float64) ** 2))
    )
    rate_evidence: dict[str, Any] = {"observed": False}
    trial_failures: list[dict[str, Any]] = []
    if attempts > 0:
        response = material.evaluate(
            last_candidate_f,
            last_history,
            jnp.full((position.shape[0],), rho, dtype=jnp.float64),
            parameters,
            state.time,
            last_dt,
        )
        rates, inertial, branches, successful_material = jax.device_get(
            (
                response.diagnostics["plastic_shear_rate"],
                response.diagnostics["inertial_number"],
                response.branch_code,
                response.successful,
            )
        )
        rate_evidence = {
            "observed": True,
            "last_trial_admitted": bool(np.all(successful_material)),
            "plastic_shear_rate_s_inv": [
                _dg_finite(float(value)) for value in np.asarray(rates).reshape(-1)
            ],
            "inertial_number": [
                _dg_finite(float(value)) for value in np.asarray(inertial).reshape(-1)
            ],
            "elastic_particles": int(np.count_nonzero(np.asarray(branches) == 0)),
            "flowing_particles": int(np.count_nonzero(np.asarray(branches) == 1)),
            "disconnected_particles": int(np.count_nonzero(np.asarray(branches) == 2)),
            "failed_particles": int(np.count_nonzero(np.asarray(branches) == -1)),
            "normalization": "r=sqrt(2*Dp:Dp), t=sqrt(s:s/2), I uses Cauchy p; no q-as-shear substitution",
        }
        trial_failures = _bg_mpm_trial_failure(
            last_history,
            last_candidate_f,
            np.asarray(host_position, dtype=np.float64),
            response.diagnostics,
            np.asarray(successful_material, dtype=np.bool_),
            cells,
        )
    finite = bool(
        np.all(np.isfinite(profile))
        and np.all(np.isfinite(host_position))
        and np.all(np.isfinite(host_velocity))
        and np.all(np.isfinite(cauchy))
    )
    model_range_limited = bool(
        rejected
        and finite
        and trial_failures
        and len(trial_failures) == rate_evidence.get("failed_particles", 0)
        and all(
            failure["candidate_small_solve_status"] == 1
            and failure["relative_small_solve_successful"]
            and failure["candidate_F_determinant"] is not None
            and failure["candidate_F_determinant"] > 0.0
            and failure["material_diagnostics"]["deformation_condition"] is not None
            and failure["material_diagnostics"]["deformation_condition"]
            <= failure["native_policy"]["maximum_condition"]
            for failure in trial_failures
        )
    )
    successful = not rejected and finite
    temporal_resolved = (
        attempts > 0
        and isfinite(float(maxima[6]))
        and (maxima[6] <= _BG_MPM_MAX_SHEAR_INCREMENT)
    )
    inconclusive = model_range_limited or (
        successful and (not steady or not support_valid or (not temporal_resolved))
    )
    if attempts == 0:
        maxima[:] = np.nan
    return {
        "resolution": cells,
        "metrics": {"profile-relative-l2-error": profile_error},
        "successful": successful,
        "inconclusive": inconclusive,
        "model_range_limited": model_range_limited,
        "reason": "Declared total-deformation native scaled-determinant range reached before steady admission; native gate unchanged, solver rejection retained."
        if model_range_limited
        else "Coupled public implicit mu(I) MPM established registered steady diagnostics; unchanged candidate reference criterion applies."
        if successful and steady and support_valid and temporal_resolved
        else "Public implicit MPM rejected a numerical/material/tangent transaction or produced nonfinite state."
        if not successful
        else "Admitted coupled MPM did not establish steady registered layer/temporal support within bounded physical time, steps, or wall time.",
        "diagnostics": {
            "flux_relative_error": _dg_finite(abs(flux / reference_flux - 1.0)),
            "pressure_profile_relative_l2_error": pressure_error,
            "steady_profile_relative_change": profile_gap,
            "steady_acceleration_over_g": acceleration_gap,
            "steady_kinetic_relative_change": kinetic_gap,
            "transverse_kinetic_energy_fraction": transverse_fraction,
            "volume_depth_relative_change": _dg_finite(
                abs(current_height_from_volume / _BG_HEIGHT - 1.0)
            ),
        },
        "evidence": {
            "steady": steady,
            "steady_windows": steady_windows,
            "declared_particle_domain_preserved": support_valid,
            "wall_time_limit_reached": wall_limit,
            "numerically_rejected": rejected,
            "last_nonlinear_status": int(codes[2]) if attempts > 0 else None,
            "last_material_admitted": bool(codes[3]) if attempts > 0 else None,
            "last_tangent_admitted": bool(codes[4]) if attempts > 0 else None,
            "last_finite": bool(codes[5]) if attempts > 0 else None,
            "last_run_status": int(codes[6]) if attempts > 0 else None,
            "maximum_residual_norm": _dg_finite(float(maxima[0])),
            "maximum_newton_steps": _dg_finite(float(maxima[1])),
            "maximum_krylov_iterations": _dg_finite(float(maxima[2])),
            "minimum_jacobian": _dg_finite(float(maxima[3])),
            "maximum_displacement_fraction": _dg_finite(float(maxima[4])),
            "maximum_predicted_shear_increment": _dg_finite(float(maxima[5])),
            "maximum_candidate_shear_increment": _dg_finite(float(maxima[6])),
            "temporal_control_resolved": temporal_resolved,
            "final_maximum_deformation_condition": deformation_condition,
            "final_maximum_elastic_Be_condition": metric_condition,
            "final_minimum_elastic_Be_eigenvalue": _dg_finite(
                float(np.min(metric_eigenvalues))
            ),
            "physical_time_s": physical_time,
            "gravity_times": physical_time / gravity_time,
            "initial_kinetic_energy_j": _dg_finite(initial_energy),
            "final_kinetic_energy_j": _dg_finite(final_energy),
            "profile_y_m": y_nodes.tolist(),
            "solver_profile_u_m_s": [_dg_finite(float(value)) for value in profile],
            "independent_reference_u_m_s": reference.tolist(),
            "solver_particle_pressure_pa": [
                _dg_finite(float(value)) for value in pressure
            ],
            "solver_particle_cauchy_shear_magnitude_pa": [
                _dg_finite(float(value)) for value in shear_magnitude
            ],
            "solver_flux_m2_s": _dg_finite(flux),
            "independent_reference_flux_m2_s": reference_flux,
            "last_material_rate_evidence": rate_evidence,
            "failed_trial_diagnostics": trial_failures,
            "temporal_samples": samples,
            "compiled_problem_id": compiled.compilation_id,
            "implicit_dynamics_id": implicit.prepared_id,
        },
        "resources": {
            "particles": position.shape[0],
            "grid_nodes": _BG_MPM_PERIODIC_NODES**2 * (cells + 7),
            "grid_unknowns": 3 * _BG_MPM_PERIODIC_NODES**2 * (cells + 7),
            "attempted_steps": attempts,
            "accepted_steps": int(jax.device_get(state.accepted_step)),
            "observed_windows": windows,
            "maximum_steps": _BG_MPM_MAX_STEPS,
            "state_bytes_estimated_by_solver": compiled.dynamics.resource_evidence.state_bytes,
            "workspace_bytes_estimated_by_solver": compiled.dynamics.resource_evidence.step_workspace_bytes,
            "elapsed_s_including_compilation": perf_counter() - started,
            "wall_time_limit_s": _BG_MPM_MAX_ELAPSED_S,
        },
    }


_LAME_RESOLUTIONS = (8, 16, 32, 64)


def mpm_case_settings(
    name: str, *, footing_resolutions: Sequence[int] = (8, 16)
) -> dict[str, Any]:
    """Numerical/support metadata available before any compilation or stepping."""
    name_ = parse(name, _MPMS16CaseName, "name")
    common: dict[str, Any] = {
        "dtype": "float64",
        "device": "CPU",
        "execution": "serial levels and rates",
        "shared_cpu_budget": 10,
        "requested_cpu_threads": 2,
        "maximum_state_bytes": 256 * 1024**2,
        "maximum_workspace_bytes": 512 * 1024**2,
        "cpu_thread_policy": "requested two threads; configure environment before JAX initialization",
        "reference_status": "independent host formulas; proposed tolerances only",
        "transfer": "APIC",
        "schedule": "USL",
        "assignment": "linear tensor B-spline",
        "gravity": 0.0,
        "density": 1.0,
    }
    match name_:
        case "lame-thick-cylinder":
            return common | {
                "resolutions": list(_LAME_RESOLUTIONS),
                "steps_per_level": [150 * n for n in _LAME_RESOLUTIONS],
                "step_sizes": [0.08 / n for n in _LAME_RESOLUTIONS],
                "physical_duration": 12.0,
                "geometry": "axisymmetric full rings; axial plane strain",
                "inner_radius": 1.0,
                "outer_radius": 2.0,
                "axial_length": 0.25,
                "inner_pressure": 0.001,
                "outer_pressure": 0.0,
                "shear_modulus": 1.0,
                "bulk_modulus": 2.0,
                "body_drag_rate": 2.0,
                "stabilization": "none",
                "particles_per_radial_cell": 2,
                "axial_particles": 4,
                "maximum_particles": 8 * max(_LAME_RESOLUTIONS),
                "maximum_steps_per_solve": 150 * max(_LAME_RESOLUTIONS),
                "boundary": "canonical follower-pressure ring quadrature; all axial nodal velocities zero",
                "stress_metric": "full-domain volume-weighted combined radial/hoop Cauchy relative L2",
                "candidate_tolerance": 0.01,
                "relaxation_limits": {
                    "energy": 0.001,
                    "displacement_drift": 0.002,
                    "force_residual": 0.01,
                },
                "reference": "sigma_r=A-B/r^2; sigma_theta=A+B/r^2; A=pi*a^2/(b^2-a^2), B=pi*a^2*b^2/(b^2-a^2)",
                "limitations": "finite-strain solution compared with small-pressure linear Lamé reference; no boundary stress exclusions",
            }
        case "prandtl-footing-tresca" | "prandtl-footing-frictional":
            return common | _mpm_footing_settings(name_, footing_resolutions)
        case _ as unreachable:
            assert_never(unreachable)


def _mpm_finite(value: float) -> float | None:
    return value if isfinite(value) else None


def _mpm_relative_l2(
    actual: NDArray[np.float64],
    reference: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    numerator = np.sum(weights[:, None] * (actual - reference) ** 2)
    denominator = np.sum(weights[:, None] * reference**2)
    return float(np.sqrt(numerator / denominator))


def _mpm_scan(
    dynamics: phx.discretization.PreparedMPMDynamics,
    arguments: phx.equations.MaterialPointArguments,
    initial: phx.discretization.MPMRuntimeState,
    steps: int,
    dt: float,
    observe: Callable[[phx.discretization.MPMStepResult], Array],
    *,
    observation_width: int = 3,
) -> tuple[
    phx.discretization.MPMRuntimeState, NDArray[np.float64], NDArray[np.bool_], int
]:
    """One bounded compiled loop; no host transfers inside individual steps."""

    def advance(
        carry: tuple[phx.discretization.MPMRuntimeState, Array, Array], unused: None
    ) -> tuple[
        tuple[phx.discretization.MPMRuntimeState, Array, Array], tuple[Array, Array]
    ]:
        del unused
        state, live, reasons = carry

        def attempt(
            value: phx.discretization.MPMRuntimeState,
        ) -> tuple[phx.discretization.MPMRuntimeState, Array, Array, Array]:
            detail = dynamics.step_detailed(value, dt, arguments)
            observation = observe(detail)
            return (
                detail.accepted_state,
                detail.successful,
                reasons | detail.rejection_reasons,
                observation,
            )

        def stopped(
            value: phx.discretization.MPMRuntimeState,
        ) -> tuple[phx.discretization.MPMRuntimeState, Array, Array, Array]:
            return (
                value,
                jnp.asarray(False),
                reasons,
                jnp.full((observation_width,), jnp.nan, dtype=jnp.float64),
            )

        accepted, successful, rejected, observation = jax.lax.cond(
            live, attempt, stopped, state
        )
        return (accepted, successful, rejected), (observation, successful)

    def execute(
        state: phx.discretization.MPMRuntimeState,
    ) -> tuple[
        tuple[phx.discretization.MPMRuntimeState, Array, Array], tuple[Array, Array]
    ]:
        return jax.lax.scan(
            advance,
            (state, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32)),
            None,
            length=steps,
        )

    (final, _, rejected), (history, accepted) = jax.jit(execute)(initial)
    host_history, host_accepted, host_rejected = jax.device_get(
        (history, accepted, rejected)
    )
    return (
        final,
        np.asarray(host_history, dtype=np.float64),
        np.asarray(host_accepted, dtype=np.bool_),
        int(host_rejected),
    )


def _mpm_pressure(
    reference: Array,
    current: Array,
    measure: DeformedMeasureState,
    state: MechanicalLoadState,
    args: Array | None,
) -> Array:
    del reference, current, args
    return -state.parameter * measure.normal("current")


def _mpm_drag(time: Array, position: Array, velocity: Array, args: Array) -> Array:
    del time, position
    return -args * velocity


def _mpm_resources(
    dynamics: phx.discretization.PreparedMPMDynamics,
    started: float,
    steps: int,
    particles: int,
) -> dict[str, Any]:
    return {
        "elapsed_seconds": perf_counter() - started,
        "step_bound": steps,
        "particles": particles,
        "state_bytes_estimated_by_solver": dynamics.resource_evidence.state_bytes,
        "workspace_bytes_estimated_by_solver": dynamics.resource_evidence.step_workspace_bytes,
        "execution": "one compiled scan; serial campaign; shared machine contention unmeasured",
    }


def _mpm_lame_level(n: int) -> dict[str, Any]:
    started = perf_counter()
    h, length, pressure = 1.0 / n, 0.25, 0.001
    r, z = np.meshgrid(
        1.0 + (np.arange(2 * n, dtype=np.float64) + 0.5) * h / 2.0,
        (np.arange(4, dtype=np.float64) + 0.5) * length / 4.0,
        indexing="ij",
    )
    host_position = np.stack((r, z), axis=-1).reshape(-1, 2)
    weights = (2.0 * pi * r * h / 2.0 * length / 4.0).reshape(-1)
    position, volume = (
        jnp.asarray(host_position, dtype=jnp.float64),
        jnp.asarray(weights, dtype=jnp.float64),
    )
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                9 * n // 4 + 1, periodic=False, endpoint=True
            ),
            phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False),
        ),
        axis_names=("r", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0), (2.25, length)), dtype=jnp.float64))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    geometry = phx.discretization.MPMGeometryPlan("axisymmetric", spatial_dimension=2)
    boundary_position = np.stack(
        (
            np.ones(4, dtype=np.float64),
            (np.arange(4, dtype=np.float64) + 0.5) * length / 4.0,
        ),
        axis=-1,
    )
    boundary_area = np.tile(
        np.asarray((-2.0 * pi * length / 4.0, 0.0), dtype=np.float64), (4, 1)
    )
    load = phx.discretization.PreparedMPMBoundaryLoad(
        splat,
        boundary_position,
        boundary_area,
        phx.applications.solid_mechanics.GeneralFollowerLoad(
            _mpm_pressure,
            support="boundary",
            measure_frame="current",
            load_id="qualification-lame-pressure",
        ),
        geometry=geometry,
    )
    mask = jnp.zeros(grid.vertices().shape + (2,), dtype=jnp.bool_).at[..., 1].set(True)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "qualification-lame",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3),
            external_acceleration=_mpm_drag,
            external_acceleration_id="qualification-linear-body-drag",
            boundary_loads=(load,),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(geometry=geometry),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.75, 0.0), (2.125, length)), dtype=jnp.float64),
            periodic=(False, True),
            support_margin=(h, 0.0),
        ),
        boundary=phx.discretization.PrescribedGridVelocityPlan(mask),
        resource_policy=phx.discretization.MPMResourcePolicy(
            maximum_state_bytes=256 * 1024**2, maximum_step_workspace_bytes=512 * 1024**2
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(1.0, 2.0),
        jnp.asarray(2.0, dtype=jnp.float64),
        mechanical_load_states=(
            MechanicalLoadState(jnp.asarray(pressure, dtype=jnp.float64)),
        ),
    )
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    force_scale = 2.0 * pi * length * pressure

    def observe(detail: phx.discretization.MPMStepResult) -> Array:
        state = detail.accepted_state.particles
        kinetic = 0.5 * jnp.sum(volume * jnp.sum(state.velocity**2, axis=-1))
        displacement = jnp.sum(
            volume * (state.position[:, 0] - position[:, 0])
        ) / jnp.sum(volume)
        residual = (
            jnp.linalg.norm(
                (detail.grid.internal_force + detail.grid.external_force)[..., 0]
            )
            / force_scale
        )
        return jnp.stack((kinetic, displacement, residual))

    steps, dt = 150 * n, 0.08 / n
    final, history, accepted, reasons = _mpm_scan(
        compiled.dynamics, arguments, initial, steps, dt, observe
    )
    deformation, piola, final_position = (
        np.asarray(x, dtype=np.float64)
        for x in jax.device_get(
            (
                final.particles.deformation_gradient,
                final.particles.first_piola,
                final.particles.position,
            )
        )
    )
    cauchy = (piola @ np.swapaxes(deformation, -1, -2)) / np.linalg.det(deformation)[
        :, None, None
    ]
    radius = host_position[:, 0]
    constant_a, constant_b = pressure / 3.0, 4.0 * pressure / 3.0
    reference = np.stack(
        (constant_a - constant_b / radius**2, constant_a + constant_b / radius**2),
        axis=-1,
    )
    stress_error = _mpm_relative_l2(cauchy[:, (0, 2), (0, 2)], reference, weights)
    reference_u = constant_a * radius / (2.0 * (1.0 + 4.0 / 3.0)) + constant_b / (
        2.0 * radius
    )
    displacement_error = _mpm_relative_l2(
        (final_position[:, 0] - radius)[:, None], reference_u[:, None], weights
    )
    peak = float(np.nanmax(history[:, 0]))
    energy_ratio = float(history[-1, 0] / peak) if peak > 0.0 else float("nan")
    drift = (
        float(abs(history[-1, 1] - history[-steps // 10, 1]) / abs(history[-1, 1]))
        if history[-1, 1] != 0.0
        else float("nan")
    )
    residual = float(history[-1, 2])
    relaxed = (
        isfinite(energy_ratio)
        and energy_ratio < 0.001
        and drift < 0.002
        and residual < 0.01
    )
    metrics = {"stress-relative-l2-error": _mpm_finite(stress_error)}
    diagnostics = {
        "displacement-relative-l2-error": _mpm_finite(displacement_error),
        "kinetic-energy-decay-ratio": _mpm_finite(energy_ratio),
        "late-displacement-relative-drift": _mpm_finite(drift),
        "equilibrium-force-relative-residual": _mpm_finite(residual),
    }
    complete = bool(np.all(accepted))
    return {
        "resolution": n,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "successful": complete,
        "inconclusive": complete and not relaxed,
        "solver_successful": complete,
        "accepted_steps": int(np.count_nonzero(accepted)),
        "rejection_reasons": reasons,
        "reason": "solver campaign completed; candidate stress criterion evaluated by caller"
        if complete and relaxed
        else "step rejection or unresolved pressure equilibrium",
        "resources": _mpm_resources(compiled.dynamics, started, steps, position.shape[0]),
    }


def _mpm_footing_reference(phi: float) -> float:
    if phi == 0.0:
        return pi + 2.0
    nq = exp(pi * tan(phi)) * tan(pi / 4.0 + phi / 2.0) ** 2
    return (nq - 1.0) / tan(phi)


def _mpm_footing_solve(n: int, phi: float, speed: float) -> dict[str, Any]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 4 or (n > 16) or n % 2:
        raise ValueError(
            "Footing n must be an even full-width integer cell count in [4,16]."
        )
    if speed != 0.02 and speed != 0.01:
        raise ValueError("Footing speed must be exactly0.02 or0.01.")
    steps, dt = _footing_time_grid(n, speed)
    started = perf_counter()
    h, length = (1.0 / n, 3.0 / n)
    x, y, z = np.meshgrid(
        -3.0 + (np.arange(6 * n, dtype=np.float64) + 0.5) * h,
        -3.0 + (np.arange(3 * n, dtype=np.float64) + 0.5) * h,
        (np.arange(2, dtype=np.float64) + 0.5) * length / 2.0,
        indexing="ij",
    )
    host_position = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    position = jnp.asarray(host_position, dtype=jnp.float64)
    volume = jnp.full((position.shape[0],), h * h * length / 2.0, dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(6 * n + 5, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(3 * n + 5, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(3, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            ((-3.0 - 2.0 * h, -3.0 - 2.0 * h, 0.0), (3.0 + 2.0 * h, 2.0 * h, length)),
            dtype=jnp.float64,
        )
    )
    coordinates = np.stack(
        np.meshgrid(
            np.linspace(-3.0 - 2.0 * h, 3.0 + 2.0 * h, 6 * n + 5, dtype=np.float64),
            np.linspace(-3.0 - 2.0 * h, 2.0 * h, 3 * n + 5, dtype=np.float64),
            np.arange(3, dtype=np.float64) * length / 3.0,
            indexing="ij",
        ),
        axis=-1,
    )
    strip = (np.abs(coordinates[..., 0]) <= 0.5 + 1e-12) & (
        np.abs(coordinates[..., 1]) < h / 10.0
    )
    side = np.abs(coordinates[..., 0]) >= 3.0 - h / 10.0
    base = coordinates[..., 1] <= -3.0 + h / 10.0
    mask = np.zeros(coordinates.shape, dtype=np.bool_)
    mask[..., 0] = strip | side | base
    mask[..., 1] = strip | base
    mask[..., 2] = True
    values = np.zeros(coordinates.shape, dtype=np.float64)
    values[..., 1] = np.where(strip, -speed, 0.0)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), volume, ambient_dimension=3
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    parameters = phx.applications.solid_mechanics.MohrCoulombParameters(
        *(
            jnp.asarray(v, dtype=jnp.float64)
            for v in (
                _FOOTING_SHEAR_MODULUS,
                _FOOTING_BULK_MODULUS,
                _FOOTING_COHESION,
                phi,
                phi,
                0.0,
            )
        )
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "qualification-rough-strip",
            phx.applications.solid_mechanics.MohrCoulombMPMConstitutivePlan(),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
                "b_bar"
            )
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((-3.0, -3.0, 0.0), (3.0, h, length)), dtype=jnp.float64),
            periodic=(False, False, True),
            support_margin=(h, h, 0.0),
        ),
        boundary=phx.discretization.PrescribedGridVelocityPlan(mask, values),
        resource_policy=phx.discretization.MPMResourcePolicy(
            maximum_state_bytes=256 * 1024**2, maximum_step_workspace_bytes=512 * 1024**2
        ),
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    strip_array = jnp.asarray(strip, dtype=jnp.bool_)
    nc = _mpm_footing_reference(phi)

    def observe(detail: phx.discretization.MPMStepResult) -> Array:
        """Measure constraint reactions and physical paired-extrusion motion, not assumed B-bar constitutive F."""
        grid_state = detail.grid
        nodal_constraint_force = (
            grid_state.mass[..., None]
            * (grid_state.velocity_after - grid_state.velocity_before)
            / dt
            - grid_state.internal_force
            - grid_state.external_force
        )
        capacity = -jnp.sum(
            jnp.where(strip_array, nodal_constraint_force[0, ..., 1], 0.0)
        ) / (length * _FOOTING_COHESION)
        state = detail.accepted_state.particles
        kinetic = 0.5 * jnp.sum(volume * jnp.sum(state.velocity**2, axis=-1))
        kinetic_ratio = kinetic / (nc * _FOOTING_TARGET_INDENTATION * length)
        plane_strain_defect = (
            jnp.max(jnp.abs(state.position[:, 2] - position[:, 2])) / length
            + jnp.max(jnp.abs(state.velocity[:, 2])) / speed
            + jnp.max(jnp.abs(state.position[::2, :2] - state.position[1::2, :2]))
            + jnp.max(jnp.abs(state.velocity[::2, :2] - state.velocity[1::2, :2])) / speed
        )
        return jnp.stack(
            (
                capacity,
                kinetic_ratio,
                plane_strain_defect,
                detail.suggested_step,
                detail.stability_margin,
                detail.restriction.limiting_process.astype(jnp.float64),
            )
        )

    final, history, accepted, reasons = _mpm_scan(
        compiled.dynamics, arguments, initial, steps, dt, observe, observation_width=6
    )
    del final
    complete = bool(np.all(accepted))
    tail = history[-steps // 5 :, 0]
    capacity = float(np.mean(tail)) if complete else float("nan")
    midpoint = tail.size // 2
    drift = (
        float(abs(np.mean(tail[:midpoint]) - np.mean(tail[midpoint:])) / nc)
        if complete
        else float("nan")
    )
    variation = float(np.std(tail) / nc) if complete else float("nan")
    energy = float(np.mean(history[-steps // 5 :, 1])) if complete else float("nan")
    plane_strain_defect = float(np.nanmax(history[:, 2]))
    plateau = (
        complete
        and drift <= 0.02
        and (variation <= 0.02)
        and (energy <= 0.01)
        and (capacity > 0.0)
        and (plane_strain_defect <= 1e-10)
    )
    measured_nc = _mpm_finite(capacity)
    accepted_count = int(np.count_nonzero(accepted))
    last_attempt = steps - 1 if complete else accepted_count
    curve_indices = (
        np.unique(
            np.linspace(0, accepted_count - 1, min(33, accepted_count), dtype=np.int64)
        )
        if accepted_count > 0
        else np.zeros((0,), dtype=np.int64)
    )
    plateau_interval = (
        [0.8 * _FOOTING_TARGET_INDENTATION, _FOOTING_TARGET_INDENTATION]
        if complete
        else None
    )
    return {
        "resolution_full_width": n,
        "resolution_half_width": n // 2,
        "cell_size": h,
        "speed": speed,
        "measured_capacity": measured_nc,
        "measured_Nc": measured_nc,
        "reference_capacity": nc,
        "capacity_error": _mpm_finite(abs(capacity / nc - 1.0)),
        "Nq_algebraically_inferred": _mpm_finite(capacity * tan(phi) + 1.0),
        "Nq_status": "derived from measured Nc, NOT independently measured or qualified",
        "plateau_drift": _mpm_finite(drift),
        "plateau_variation": _mpm_finite(variation),
        "kinetic_ratio": _mpm_finite(energy),
        "plane_strain_defect": _mpm_finite(plane_strain_defect),
        "peak_reaction_over_reference": _mpm_finite(float(np.nanmax(history[:, 0]) / nc)),
        "solver_successful": complete,
        "plateau_resolved": plateau,
        "accepted_steps": accepted_count,
        "rejection_reasons": reasons,
        "last_step_restriction": {
            "requested_step_s": dt,
            "suggested_step_s": _mpm_finite(float(history[last_attempt, 3])),
            "stability_margin_s": _mpm_finite(float(history[last_attempt, 4])),
            "limiting_process": int(history[last_attempt, 5])
            if isfinite(float(history[last_attempt, 5]))
            else None,
        },
        "resources": _mpm_resources(compiled.dynamics, started, steps, position.shape[0]),
        "load_settlement_curve": {
            "settlement_over_B": ((curve_indices + 1) * speed * dt).tolist(),
            "reaction_Nc": [
                _mpm_finite(float(value)) for value in history[curve_indices, 0]
            ],
            "note": "accepted reactions only; initial inertial spike is not ultimate capacity",
        },
        "half_cell_support_limit": 0.5 * h,
        "plateau_settlement_interval": plateau_interval,
        "plateau_before_half_cell_support_limit": bool(
            plateau and _FOOTING_TARGET_INDENTATION < 0.5 * h
        ),
    }


def _mpm_footing_campaign(
    name: str, phi: float, *, resolutions: Sequence[int] = (8, 16)
) -> dict[str, Any]:
    if phi != _mpm_footing_phi(name):
        raise ValueError("Footing case identity and friction angle disagree.")
    requested = _mpm_footing_resolutions(name, resolutions)
    settings = mpm_case_settings(name, footing_resolutions=requested)
    levels: list[dict[str, Any]] = []
    previous_capacity: float | None = None
    previous_solver_ok = False
    nc = _mpm_footing_reference(phi)
    for n in requested:
        fast = _mpm_footing_solve(n, phi, 0.02)
        slow = _mpm_footing_solve(n, phi, 0.01)
        capacity = slow["measured_capacity"]
        fast_capacity = fast["measured_capacity"]
        rate_gap = (
            abs(capacity - fast_capacity) / nc
            if capacity is not None and fast_capacity is not None
            else None
        )
        solver_ok = bool(fast["solver_successful"] and slow["solver_successful"])
        resolution_gap = (
            abs(capacity - previous_capacity) / nc
            if solver_ok
            and previous_solver_ok
            and (capacity is not None)
            and (previous_capacity is not None)
            else None
        )
        rate_resolved = rate_gap is not None and rate_gap <= 0.02
        steady = bool(
            fast["plateau_resolved"] and slow["plateau_resolved"] and rate_resolved
        )
        mesh_resolved = resolution_gap is not None and resolution_gap <= 0.02
        levels.append(
            {
                "resolution": n,
                "resolution_half_width": n // 2,
                "metrics": {"bearing-capacity-relative-error": slow["capacity_error"]},
                "diagnostics": {
                    "loading-rate-relative-gap": _mpm_finite(rate_gap)
                    if rate_gap is not None
                    else None,
                    "successive-resolution-relative-gap": _mpm_finite(resolution_gap)
                    if resolution_gap is not None
                    else None,
                    "plateau-relative-drift": slow["plateau_drift"],
                    "plateau-relative-variation": slow["plateau_variation"],
                    "kinetic-to-reference-work-ratio": slow["kinetic_ratio"],
                    "measured-Nc": capacity,
                    "Nq-algebraically-inferred-NOT-independent": slow[
                        "Nq_algebraically_inferred"
                    ],
                },
                "successful": solver_ok,
                "inconclusive": solver_ok and (not steady or not mesh_resolved),
                "solver_successful": solver_ok,
                "rate_plateau_resolved": steady,
                "mesh_resolved_within_this_part": mesh_resolved,
                "reason": "step rejection"
                if not solver_ok
                else "solver campaign completed; candidate capacity criterion evaluated by caller"
                if steady and mesh_resolved
                else "unresolved plateau/rate/mesh; no convergence or candidate pass asserted",
                "rates": [fast, slow],
                "resources": {
                    "step_bound": _footing_time_grid(n, 0.02)[0]
                    + _footing_time_grid(n, 0.01)[0],
                    "maximum_particles": 36 * n * n,
                },
            }
        )
        previous_capacity = capacity
        previous_solver_ok = solver_ok
    finest = levels[-1]
    result: dict[str, Any] = {
        "resolutions": list(requested),
        "metrics": finest["metrics"],
        "levels": levels,
        "successful": all((level["solver_successful"] for level in levels)),
        "inconclusive": finest["inconclusive"],
        "reason": finest["reason"],
        "diagnostics": finest["diagnostics"],
        "resources": settings,
        "settings": settings,
        "support_notes": settings["limitations"],
        "reference_capacity": nc,
        "Nq_status": settings["Nq_status"],
    }
    result["convergence_diagnostic"] = _mpm_footing_convergence(name, (result,))
    return result


def run_lame_thick_cylinder() -> dict[str, Any]:
    """Pressure-load an unstressed full-ring cylinder, then measure equilibrium."""
    levels = [_mpm_lame_level(n) for n in _LAME_RESOLUTIONS]
    finest = levels[-1]
    settings = mpm_case_settings("lame-thick-cylinder")
    return {
        "resolutions": list(_LAME_RESOLUTIONS),
        "metrics": finest["metrics"],
        "levels": levels,
        "successful": all(level["solver_successful"] for level in levels),
        "inconclusive": finest["inconclusive"],
        "reason": finest["reason"],
        "diagnostics": finest["diagnostics"],
        "resources": settings,
        "settings": settings,
        "support_notes": settings["limitations"],
    }


def run_prandtl_footing_tresca(*, resolutions: Sequence[int] = (8, 16)) -> dict[str, Any]:
    """Fresh bounded refinement of weightless rough-strip Tresca capacity."""
    return _mpm_footing_campaign("prandtl-footing-tresca", 0.0, resolutions=resolutions)


def run_prandtl_footing_frictional(
    *, resolutions: Sequence[int] = (8, 16)
) -> dict[str, Any]:
    """Fresh bounded refinement of weightless associated-MC20-degree capacity."""
    return _mpm_footing_campaign(
        "prandtl-footing-frictional", pi / 9.0, resolutions=resolutions
    )


GranularCaseName: TypeAlias = Literal["hertz-normal-impact", "bagnold-inclined-plane"]


DEM_GRANULAR_METADATA: dict[str, dict[str, Any]] = {
    "hertz-normal-impact": {
        "runner": "run_hertz_normal_impact",
        "resolutions": [256, 512, 1024],
        "resolution_quantity": "fixed integration steps over 1.25 reference contact durations",
        "support": "3-D two-sphere elastic normal impact; public compliant DEM velocity-Verlet transaction; no gravity, friction, damping, walls, or shape approximation",
        "public_apis": [
            "compile_discrete_element_problem",
            "CompliantDEMMethodPlan",
            "HertzNormalContactPlan",
            "CompiledDiscreteElementProblem.initialize_state",
            "PreparedCompliantDEMDynamics.step_detailed",
        ],
        "constants": {
            "mass_kg_each": 1.0,
            "radius_m_each": 0.5,
            "young_modulus_pa": 2000000.0,
            "poisson_ratio": 0.25,
            "relative_impact_speed_m_s": 1.0,
            "restitution": 1.0,
            "horizon_contact_durations": 1.25,
            "quadrature_epsabs": 1e-12,
            "quadrature_epsrel": 1e-12,
        },
        "formulas": [
            "m*=m1*m2/(m1+m2); R*=R1*R2/(R1+R2); 1/E*=(1-nu1^2)/E1+(1-nu2^2)/E2",
            "k=(4/3) E* sqrt(R*); F=k delta^(3/2); U=(2/5) k delta^(5/2)",
            "delta_max=(5 m* v0^2/(4 k))^(2/5); F_max=k delta_max^(3/2)",
            "t_c=(2 delta_max/v0) integral_0^1 [1-x^(5/2)]^(-1/2) dx; x=1-s^2 removes the endpoint singularity",
            "errors: |measured/reference-1|; energy=max_t |K+U-K0|/K0; restitution=|v_separation/v0-1|",
        ],
        "references": [
            {
                "title": "Sun et al. (2021), Hertz elastic dynamics of two colliding elastic spheres",
                "doi": "10.1016/j.rinp.2021.104870",
                "url": "https://doi.org/10.1016/j.rinp.2021.104870",
                "used_for": "elastic Hertz force and contact-duration energy integral; peak independently derived from energy conservation",
            }
        ],
        "resource_bounds": {
            "levels": 3,
            "bodies": 2,
            "maximum_contact_routes": 1,
            "maximum_steps_per_level": 1024,
            "total_steps": 1792,
            "host_reference_quadrature_limit": 100,
        },
    }
}


def granular_case_settings(name: GranularCaseName) -> dict[str, Any]:
    """Return deterministic support and resource settings without executing JAX."""
    name = parse(name, GranularCaseName, "name")
    match name:
        case "hertz-normal-impact":
            return DEM_GRANULAR_METADATA["hertz-normal-impact"]
        case "bagnold-inclined-plane":
            return bagnold_solver_settings()
        case _:
            assert_never(name)


def _dg_finite(value: float) -> float | None:
    return value if isfinite(value) else None


def _hertz_host_reference() -> tuple[float, float, float, float]:
    # This host reference never evaluates the public DEM force law.
    reduced_mass = 0.5
    effective_radius = 0.25
    effective_modulus = 2.0e6 / (2.0 * (1.0 - 0.25**2))
    stiffness = (4.0 / 3.0) * effective_modulus * sqrt(effective_radius)
    maximum_overlap = (5.0 * reduced_mass / (4.0 * stiffness)) ** 0.4

    def integrand(s: float) -> float:
        if s == 0.0:
            return sqrt(8.0 / 5.0)
        if s == 1.0:
            return 2.0
        return 2.0 * s / sqrt(-expm1(2.5 * log1p(-(s * s))))

    integral, quadrature_error = quad(
        integrand, 0.0, 1.0, epsabs=1.0e-12, epsrel=1.0e-12, limit=100
    )
    duration = 2.0 * maximum_overlap * integral
    peak_force = stiffness * maximum_overlap**1.5
    return duration, maximum_overlap, peak_force, quadrature_error / integral


def _hertz_public_problem() -> CompiledDiscreteElementProblem:
    particles = ParticleSetPlan(
        jnp.asarray([0, 1], dtype=jnp.int64),
        jnp.ones((2,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    materials = DEMMaterialTable(
        jnp.asarray([2.0e6], dtype=jnp.float64),
        jnp.asarray([0.25], dtype=jnp.float64),
        jnp.asarray([[1.0]], dtype=jnp.float64),
        jnp.asarray([[0.0]], dtype=jnp.float64),
    )
    return compile_discrete_element_problem(
        DiscreteElementProblemIR(
            "qualification-hertz-normal-impact",
            materials,
            gravity=jnp.zeros((3,), dtype=jnp.float64),
        ),
        particles,
        RigidSphereSetPlan(
            jnp.full((2,), 0.5, dtype=jnp.float64), jnp.zeros((2,), dtype=jnp.int32)
        ),
        CompliantDEMMethodPlan(DEMContactModelPlan(HertzNormalContactPlan())),
        neighborhood=DenseParticleNeighborhoodPlan(1),
    )


@eqx.filter_jit
def _hertz_history(
    compiled: CompiledDiscreteElementProblem,
    initial: DEMRuntimeState,
    step_size: Array,
    steps: int,
) -> tuple[DEMRuntimeState, tuple[Array, Array, Array, Array, Array]]:
    def advance(
        state: DEMRuntimeState, index: Array
    ) -> tuple[DEMRuntimeState, tuple[Array, Array, Array, Array, Array]]:
        detail = compiled.dynamics.step_detailed(
            index, index.astype(jnp.float64) * step_size, state, step_size, None
        )
        accepted = detail.accepted_state
        gap = jnp.linalg.norm(
            accepted.kinematics.position[1] - accepted.kinematics.position[0]
        ) - jnp.asarray(1.0, dtype=jnp.float64)
        force = jnp.linalg.norm(accepted.loads.total.force[1])
        energy = (
            jnp.asarray(0.5, dtype=jnp.float64) * jnp.sum(accepted.kinematics.velocity**2)
            + accepted.energy.contact_energy
        )
        return accepted, (detail.successful, gap, force, energy, detail.rejection_reasons)

    return jax.lax.scan(advance, initial, jnp.arange(steps, dtype=jnp.int32))


def _hertz_level(
    compiled: CompiledDiscreteElementProblem,
    initial: DEMRuntimeState,
    steps: int,
    reference: tuple[float, float, float, float],
    initial_successful: bool,
) -> dict[str, Any]:
    duration, maximum_overlap, peak_force, _ = reference
    dt = 1.25 * duration / steps
    started = perf_counter()
    final, history = _hertz_history(
        compiled, initial, jnp.asarray(dt, dtype=jnp.float64), steps
    )
    statuses, gaps_device, forces_device, energies_device, reasons_device = (
        jax.device_get(history)
    )
    gaps = np.concatenate(
        (np.zeros((1,), dtype=np.float64), np.asarray(gaps_device, dtype=np.float64))
    )
    forces = np.asarray(forces_device, dtype=np.float64)
    energies = np.asarray(energies_device, dtype=np.float64)
    separation_speed = float(
        jax.device_get(final.kinematics.velocity[1, 0] - final.kinematics.velocity[0, 0])
    )
    crossings = np.flatnonzero((gaps[:-1] < 0.0) & (gaps[1:] >= 0.0))
    event_found = crossings.size == 1
    measured_duration: float | None = None
    if event_found:
        index = int(crossings[0])
        measured_duration = (index - gaps[index] / (gaps[index + 1] - gaps[index])) * dt
    raw_metrics: dict[str, float | None] = {
        "duration-relative-error": None
        if measured_duration is None
        else _dg_finite(abs(measured_duration / duration - 1.0)),
        "peak-force-relative-error": _dg_finite(
            abs(float(np.max(forces)) / peak_force - 1.0)
        ),
        "maximum_overlap_relative_error": _dg_finite(
            abs(-float(np.min(gaps)) / maximum_overlap - 1.0)
        ),
        "restitution_relative_error": _dg_finite(abs(separation_speed - 1.0)),
        "energy_relative_error": _dg_finite(
            float(np.max(np.abs(energies - 0.25))) / 0.25
        ),
    }
    finite = all(value is not None and isfinite(value) for value in raw_metrics.values())
    successful = initial_successful and bool(np.all(statuses)) and event_found and finite
    return {
        "resolution": steps,
        "metrics": {
            name: raw_metrics[name]
            for name in ("duration-relative-error", "peak-force-relative-error")
        },
        "diagnostics": {
            name: raw_metrics[name]
            for name in (
                "maximum_overlap_relative_error",
                "restitution_relative_error",
                "energy_relative_error",
            )
        },
        "successful": successful,
        "inconclusive": not successful,
        "reason": "Accepted complete elastic impact with a resolved separation crossing."
        if successful
        else "Initial/step admission, finite output, or a unique separation crossing failed.",
        "evidence": {
            "initial_successful": initial_successful,
            "accepted_steps": int(np.count_nonzero(statuses)),
            "rejected_steps": int(steps - np.count_nonzero(statuses)),
            "rejection_reasons": np.unique(np.asarray(reasons_device)).tolist(),
            "contact_samples": int(np.count_nonzero(gaps < 0.0)),
            "separation_crossings": crossings.size,
            "contact_duration_s": measured_duration,
            "peak_force_n": _dg_finite(float(np.max(forces))),
            "maximum_overlap_m": _dg_finite(-float(np.min(gaps))),
            "separation_speed_m_s": _dg_finite(separation_speed),
        },
        "resources": {
            "steps": steps,
            "step_size_s": dt,
            "horizon_s": steps * dt,
            "elapsed_s_including_compilation": perf_counter() - started,
        },
    }


def run_hertz_normal_impact() -> dict[str, Any]:
    started = perf_counter()
    reference = _hertz_host_reference()
    compiled = _hertz_public_problem()
    initial = compiled.initialize_state(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64),
        jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=jnp.float64),
    )
    initial_successful = bool(
        jax.device_get(
            compiled.diagnostics(jnp.asarray(0.0, dtype=jnp.float64), initial).successful
        )
    )
    resolutions = (256, 512, 1024)
    levels = [
        _hertz_level(compiled, initial, steps, reference, initial_successful)
        for steps in resolutions
    ]
    successful = all(level["successful"] for level in levels)
    return {
        "resolutions": list(resolutions),
        "metrics": levels[-1]["metrics"],
        "levels": levels,
        "diagnostics": levels[-1]["diagnostics"],
        "successful": successful,
        "inconclusive": not successful,
        "reason": "Actual compliant DEM impact; errors against independent host energy/quadrature references. Candidate criteria only."
        if successful
        else "At least one bounded compliant DEM level failed; inspect level evidence.",
        "resources": {
            "bodies": 2,
            "maximum_contact_routes": 1,
            "total_steps": sum(resolutions),
            "maximum_steps_per_level": max(resolutions),
            "elapsed_s_including_compilation": perf_counter() - started,
        },
        "reference": {
            "contact_duration_s": reference[0],
            "maximum_overlap_m": reference[1],
            "peak_force_n": reference[2],
            "quadrature_relative_error_estimate": reference[3],
        },
    }


@eqx.filter_jit
def _bagnold_profile(
    law: MuILaw,
    parameters: MuIParameters,
    pressure: Array,
    target_friction: Array,
    rate_scale: Array,
    dy: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    lower = jnp.full_like(pressure, rate_scale * jnp.asarray(1.0e-8, dtype=jnp.float64))
    upper = jnp.full_like(pressure, rate_scale * jnp.asarray(1.0e4, dtype=jnp.float64))
    lower_evaluation = law.evaluate(lower, pressure, parameters)
    upper_evaluation = law.evaluate(upper, pressure, parameters)
    bracketed = (
        lower_evaluation.successful
        & upper_evaluation.successful
        & (lower_evaluation.friction_coefficient < target_friction)
        & (upper_evaluation.friction_coefficient > target_friction)
    )

    def bisect(_: int | Array, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bracket
        mid = (lo + hi) * jnp.asarray(0.5, dtype=jnp.float64)
        result = law.evaluate(mid, pressure, parameters)
        above = result.friction_coefficient > target_friction
        return jnp.where(above, lo, mid), jnp.where(above, mid, hi)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    rates = (lo + hi) * jnp.asarray(0.5, dtype=jnp.float64)
    evaluation = law.evaluate(rates, pressure, parameters)
    velocity = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(dy * rates))
    )
    flux = dy * jnp.sum(
        (velocity[:-1] + velocity[1:]) * jnp.asarray(0.5, dtype=jnp.float64)
    )
    return (
        velocity,
        flux,
        evaluation.friction_coefficient,
        evaluation.successful & bracketed,
        evaluation.hadamard_margin,
        evaluation.hadamard_applicable,
    )


def _bagnold_level(law: MuILaw, parameters: MuIParameters, cells: int) -> dict[str, Any]:
    started = perf_counter()
    height, gravity, packing, diameter = 0.08, 9.81, 0.6, 0.001
    theta, target = atan(0.5), 0.5
    dy = height / cells
    midpoints = (
        jnp.arange(cells, dtype=jnp.float64) + jnp.asarray(0.5, dtype=jnp.float64)
    ) * jnp.asarray(dy, dtype=jnp.float64)
    pressure = jnp.asarray(packing * 2500.0 * gravity * cos(theta), dtype=jnp.float64) * (
        jnp.asarray(height, dtype=jnp.float64) - midpoints
    )
    result = _bagnold_profile(
        law,
        parameters,
        pressure,
        jnp.asarray(target, dtype=jnp.float64),
        jnp.asarray(sqrt(gravity / height), dtype=jnp.float64),
        jnp.asarray(dy, dtype=jnp.float64),
    )
    velocity, flux, friction, statuses, margins, applicable = jax.device_get(result)
    # Analytic inversion and closed-form integration exist ONLY in this oracle.
    inertial_reference = 0.279 * (target - 0.383) / (0.643 - target)
    coefficient = inertial_reference * sqrt(packing * gravity * cos(theta)) / diameter
    nodes = np.linspace(0.0, height, cells + 1, dtype=np.float64)
    reference = (2.0 / 3.0) * coefficient * (height**1.5 - (height - nodes) ** 1.5)
    reference_flux = (2.0 / 5.0) * coefficient * height**2.5
    values = np.asarray(velocity, dtype=np.float64)
    metrics = {
        "profile-relative-l2-error": _dg_finite(
            float(np.linalg.norm(values - reference) / np.linalg.norm(reference))
        ),
        "flux_relative_error": _dg_finite(abs(float(flux) / reference_flux - 1.0)),
        "friction_relative_error": _dg_finite(
            float(np.max(np.abs(np.asarray(friction) / target - 1.0)))
        ),
    }
    finite = all(value is not None for value in metrics.values())
    residual = metrics["friction_relative_error"]
    converged = residual is not None and residual * target <= 1.0e-10
    successful = bool(np.all(statuses)) and finite and converged
    return {
        "resolution": cells,
        "metrics": {"profile-relative-l2-error": metrics["profile-relative-l2-error"]},
        "successful": successful,
        "inconclusive": not successful,
        "diagnostics": {
            name: metrics[name]
            for name in ("flux_relative_error", "friction_relative_error")
        },
        "reason": "Admitted bracketed public-law steady profile; momentum balance and no-slip integration, not dynamic FLIP."
        if successful
        else "Public constitutive admission, bracketing, finite output, or steady friction residual failed.",
        "evidence": {
            "admitted_cells": int(np.count_nonzero(statuses)),
            "constitutive_successful": bool(np.all(statuses)),
            "steady_residual_converged": converged,
            "surface_velocity_m_s": _dg_finite(float(values[-1])),
            "flux_m2_s": _dg_finite(float(flux)),
            "minimum_hadamard_margin": _dg_finite(float(np.min(margins))),
            "hadamard_applicable_cells": int(np.count_nonzero(applicable)),
            "base_no_slip_error": _dg_finite(abs(float(values[0]))),
            "free_surface_pressure_pa": 0.0,
            "free_surface_shear_traction_pa": 0.0,
            "surface_zero_rate_law_evaluated": False,
        },
        "resources": {
            "cells": cells,
            "bisection_iterations_per_cell": 64,
            "dynamic_timesteps": 0,
            "elapsed_s_including_compilation": perf_counter() - started,
        },
    }


def run_bagnold_inclined_plane() -> dict[str, Any]:
    started = perf_counter()
    parameters = MuIParameters(
        *(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (_BG_MU_S, _BG_MU_2, _BG_I_0, _BG_DIAMETER, _BG_GRAIN_DENSITY)
        )
    )
    levels = [_bg_mpm_level(cells, parameters) for cells in _BG_MPM_RESOLUTIONS]
    errors = [level["metrics"]["profile-relative-l2-error"] for level in levels]
    successful = all((level["successful"] for level in levels))
    inconclusive = any((level["inconclusive"] for level in levels))
    model_range_limited = any(
        level.get("model_range_limited", False) for level in levels
    ) and all(
        level["successful"] or level.get("model_range_limited", False) for level in levels
    )
    return {
        "resolutions": list(_BG_MPM_RESOLUTIONS),
        "metrics": levels[-1]["metrics"],
        "levels": levels,
        "diagnostics": levels[-1]["diagnostics"],
        "successful": successful,
        "inconclusive": inconclusive,
        "model_range_limited": model_range_limited,
        "reason": "Primary full public coupled implicit mu(I) MPM against independent host Bagnold continuum reference; finite elastic support recorded; candidate profile bound unchanged."
        if successful and (not inconclusive)
        else "At least one coupled MPM level rejected or did not establish registered steady support; no FLIP/law substitution.",
        "convergence": {
            "profile_errors": errors,
            "all_levels_steady": all((level["evidence"]["steady"] for level in levels)),
            "observed_error_reduction_factors": [
                _dg_finite(coarse / fine)
                if coarse is not None and fine is not None and (fine > 0.0)
                else None
                for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
            ],
        },
        "law_only_diagnostic": {
            "primary_qualification": False,
            "levels": [
                _bagnold_level(MuILaw(), parameters, cells)
                for cells in _BG_MPM_RESOLUTIONS
            ],
            "note": "pointwise inversion only; excluded from primary profile, initialization and outcome; no FLIP fallback",
        },
        "resources": {
            "maximum_depth_cells": max(_BG_MPM_RESOLUTIONS),
            "total_attempted_steps": sum(
                (level["resources"]["attempted_steps"] for level in levels)
            ),
            "maximum_total_steps": len(_BG_MPM_RESOLUTIONS) * _BG_MPM_MAX_STEPS,
            "maximum_wall_s_per_level": _BG_MPM_MAX_ELAPSED_S,
            "elapsed_s_including_compilation": perf_counter() - started,
        },
    }


_TERZAGHI_RESOLUTIONS = (8, 16)


_TERZAGHI_LENGTH = 1.0


_TERZAGHI_POROSITY = 0.4


_TERZAGHI_SOLID_DENSITY = 2650.0


_TERZAGHI_FLUID_DENSITY = 1000.0


_TERZAGHI_VISCOSITY = 1.0e-3


_TERZAGHI_MODULUS = 1.0e6


_TERZAGHI_LOAD_STRAIN = 1.0e-5


_TERZAGHI_TIME_RATIO = 100.0


_TERZAGHI_TIME_FACTORS = (0.05, 0.2)


_TERZAGHI_CFL = 0.2


_TERZAGHI_SOLVER_TOLERANCE = 1.0e-8


_TERZAGHI_MAXIMUM_STEPS = 12000


_TERZAGHI_QUASI_STATIC_GAP_LIMIT = 0.1


def _terzaghi_properties() -> tuple[float, float, float, float, float]:
    bulk_density = (1.0 - _TERZAGHI_POROSITY) * _TERZAGHI_SOLID_DENSITY
    mixture_density = bulk_density + _TERZAGHI_POROSITY * _TERZAGHI_FLUID_DENSITY
    solid_wave_speed = math.sqrt(_TERZAGHI_MODULUS / bulk_density)
    mixture_wave_speed = math.sqrt(_TERZAGHI_MODULUS / mixture_density)
    wave_time = _TERZAGHI_LENGTH / mixture_wave_speed
    cv = _TERZAGHI_LENGTH**2 / (_TERZAGHI_TIME_RATIO * wave_time)
    beta = _TERZAGHI_MODULUS * _TERZAGHI_POROSITY**2 / cv
    diameter = math.sqrt(18.0 * _TERZAGHI_VISCOSITY * (1.0 - _TERZAGHI_POROSITY) / beta)
    return solid_wave_speed, wave_time, cv, beta, diameter


def terzaghi_1d_consolidation_settings() -> dict[str, Any]:
    """Static scientific identity and hard bounds, available before execution.

    The high-permeability Stokes diameter is chosen from material/density scales,
    not from the reference pressure. It makes nominal consolidation time 100
    mixture wave transits; the final observation is about 18--19 transits. Phase
    inertia, implicit drag, mixed pressure solve, solid constitutive update,
    boundary-load advection, and both particle advections remain enabled.
    """
    speed, wave_time, cv, beta, diameter = _terzaghi_properties()
    planned: list[dict[str, Any]] = []
    for cells in _TERZAGHI_RESOLUTIONS:
        h = _TERZAGHI_LENGTH / cells
        drainage_length = _TERZAGHI_LENGTH - 0.5 * h
        target = _TERZAGHI_TIME_FACTORS[-1] * drainage_length**2 / cv
        dt_bound = _TERZAGHI_CFL * h / (1.01 * speed)
        steps = math.ceil(target / dt_bound)
        planned.append(
            {
                "resolution": cells,
                "drainage_length_m": drainage_length,
                "target_time_s": target,
                "wave_transits_at_target": target / wave_time,
                "planned_steps": steps,
                "step_size_s": target / steps,
                "acoustic_step_bound_s": dt_bound,
                "maximum_krylov_iterations_per_solve": max(128, 8 * cells),
                "particle_count_per_phase": cells,
                "solid_nodal_count": cells + 5,
                "mac_computational_cells": cells + 2,
            }
        )
    return {
        "resolutions": list(_TERZAGHI_RESOLUTIONS),
        "parameters": {
            "length_m": _TERZAGHI_LENGTH,
            "initial_porosity": _TERZAGHI_POROSITY,
            "solid_intrinsic_density_kg_m3": _TERZAGHI_SOLID_DENSITY,
            "fluid_intrinsic_density_kg_m3": _TERZAGHI_FLUID_DENSITY,
            "fluid_viscosity_pa_s": _TERZAGHI_VISCOSITY,
            "constrained_modulus_pa": _TERZAGHI_MODULUS,
            "shear_modulus_pa": _TERZAGHI_MODULUS / 4.0,
            "lame_lambda_pa": _TERZAGHI_MODULUS / 2.0,
            "load_pa": _TERZAGHI_MODULUS * _TERZAGHI_LOAD_STRAIN,
            "load_to_modulus_ratio": _TERZAGHI_LOAD_STRAIN,
            "grain_diameter_m": diameter,
            "initial_stokes_beta_pa_s_m2": beta,
            "consolidation_coefficient_m2_s": cv,
            "darcy_mobility_m2_pa_s": _TERZAGHI_POROSITY**2 / beta,
            "intrinsic_permeability_m2": _TERZAGHI_VISCOSITY
            * _TERZAGHI_POROSITY**2
            / beta,
            "mixture_wave_transit_time_s": wave_time,
            "nominal_consolidation_to_wave_time_ratio": _TERZAGHI_TIME_RATIO,
            "observation_time_factors": list(_TERZAGHI_TIME_FACTORS),
            "biot_coefficient": 1.0,
            "constituent_storage": 0.0,
        },
        "support": {
            "solver": "advance_double_point_mpm_step",
            "geometry": "cartesian-one-dimensional",
            "solid_material": "NeoHookeanMPMConstitutivePlan(1)",
            "solid_schedule": "USL",
            "solid_transfer": "APIC",
            "assignment": "linear-tensor-b-spline",
            "fluid_transfer": "FLIP",
            "drag_law": "stokes",
            "bottom": "stationary-solid-and-impermeable-fluid-face",
            "top": "sustained-reference-dead-traction-and-zero-excess-pressure-last-cell-center",
            "constituents": "incompressible-alpha-1-storage-0",
            "pressure_reference": "independent-host-160-term-Terzaghi-series",
            "settlement_observable": "advected-loaded-top-boundary-marker",
            "computational_halo": "one-MAC-cell-per-end-with-exterior-datum-rows",
            "exterior_velocity_masks": "solid-and-fluid-velocity-zero-outside-physical-column",
            "mac_cell_pressure_mask": "bottom-halo-top-physical-reservoir-top-halo",
        },
        "resource_limits": {
            "maximum_attempts_per_level": _TERZAGHI_MAXIMUM_STEPS,
            "maximum_attempts_total": len(_TERZAGHI_RESOLUTIONS)
            * _TERZAGHI_MAXIMUM_STEPS,
            "maximum_resolution": max(_TERZAGHI_RESOLUTIONS),
            "reference_series_terms": 160,
            "compiled_device_loop": True,
            "retry_count": 0,
            "planned_levels": planned,
        },
        "quasi_static_admission": {
            "maximum_load_scaled_pressure_equilibrium_gap": _TERZAGHI_QUASI_STATIC_GAP_LIMIT,
            "minimum_target_wave_transits": 10.0,
            "maximum_load_to_modulus_ratio": 1.0e-3,
            "maximum_observed_stokes_reynolds_number": 1.0,
        },
        "reference_boundary_note": (
            "Drainage lies at L_eff=L-h/2. The physical top marker lies at L. "
            "The half-cell cap above L_eff has zero excess pressure, so the independent "
            "settlement is load/H*[L-L_eff*mean_pressure_ratio]. No pressure-derived "
            "strain or reference value is fed back to either solver phase."
        ),
        "qualification_status": "candidate-not-approved",
    }


def _terzaghi_reference(
    coordinates: NDArray[np.float64],
    elapsed: float,
    drainage_length: float,
    cv: float,
    load: float,
) -> tuple[NDArray[np.float64], float]:
    """Host-only solution: p_x(0)=0, p(L_eff)=0, uniform initial p=load."""
    if elapsed == 0.0:
        pressure = np.where(coordinates < drainage_length, load, 0.0)
        # The half-cell drained cap is already part of this boundary idealization.
        return pressure, load * (_TERZAGHI_LENGTH - drainage_length) / _TERZAGHI_MODULUS
    modes = np.arange(160, dtype=np.float64)
    dimensionless_k = (modes + 0.5) * np.pi
    decay = np.exp(-(dimensionless_k**2) * cv * elapsed / drainage_length**2)
    coefficient = 2.0 * load * np.cos(modes * np.pi) / dimensionless_k
    pressure = np.sum(
        coefficient[:, None]
        * decay[:, None]
        * np.cos(dimensionless_k[:, None] * coordinates[None, :] / drainage_length),
        axis=0,
    )
    pressure = np.where(coordinates >= drainage_length, 0.0, pressure)
    average_ratio = float(np.sum(2.0 * decay / dimensionless_k**2))
    settlement = (
        load / _TERZAGHI_MODULUS * (_TERZAGHI_LENGTH - drainage_length * average_ratio)
    )
    return pressure, settlement


@eqx.filter_jit
def _terzaghi_advance_bounded(
    plan: phx.solver.DoublePointMPMPlan,
    state: phx.solver.DoublePointMPMState,
    arguments: phx.equations.MaterialPointArguments,
    dt: Array,
    attempt_limit: Array,
) -> tuple[phx.solver.DoublePointMPMState, Array, Array, Array]:
    """Stop at the first atomic rejection; never replace or retry the solver."""
    initial = (
        state,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.zeros((9,), dtype=jnp.float64),
    )

    def condition(
        carry: tuple[phx.solver.DoublePointMPMState, Array, Array, Array],
    ) -> Array:
        _, count, successful, _ = carry
        return (count < attempt_limit) & successful

    def advance(
        carry: tuple[phx.solver.DoublePointMPMState, Array, Array, Array],
    ) -> tuple[phx.solver.DoublePointMPMState, Array, Array, Array]:
        committed, count, _, diagnostics = carry
        result = phx.solver.advance_double_point_mpm_step(
            plan, committed, dt, arguments=arguments
        )
        maximum_reynolds = jnp.max(
            jnp.stack(tuple(jnp.max(value.reynolds_number) for value in result.drag))
        )
        updated = jnp.stack(
            (
                result.status.astype(jnp.float64),
                result.projection.linear_status.astype(jnp.float64),
                jnp.maximum(
                    diagnostics[2], result.projection.iterations.astype(jnp.float64)
                ),
                jnp.maximum(diagnostics[3], result.projection.residual_norm),
                jnp.maximum(diagnostics[4], result.drag_residual_norm),
                jnp.maximum(diagnostics[5], maximum_reynolds),
                jnp.min(result.porosity),
                jnp.max(result.porosity),
                result.solid_step.rejection_reasons.astype(jnp.float64),
            )
        )
        return result.accepted_state, count + 1, result.successful, updated

    return jax.lax.while_loop(condition, advance, initial)


def _terzaghi_finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _terzaghi_errors(
    state: phx.solver.DoublePointMPMState,
    coordinates: NDArray[np.float64],
    drainage_length: float,
    reference_time: float,
    cv: float,
    load: float,
) -> dict[str, float | None]:
    pressure = np.asarray(jax.device_get(state.fluid.pressure), dtype=np.float64)[1:-1]
    expected, expected_settlement = _terzaghi_reference(
        coordinates, reference_time, drainage_length, cv, load
    )
    settlement = _TERZAGHI_LENGTH - float(state.solid.boundary_states[0].position[0, 0])
    particle_x = np.asarray(
        jax.device_get(state.solid.particles.position), dtype=np.float64
    )[:, 0]
    effective = np.asarray(
        jax.device_get(state.solid.particles.first_piola), dtype=np.float64
    )[:, 0, 0]
    # In one dimension Cauchy stress = P*F/J = P. Host interpolation is only
    # for the measured equilibrium diagnostic; it never supplies solver forces.
    sampled_pressure = np.interp(
        particle_x, coordinates, pressure, left=pressure[0], right=0.0
    )
    equilibrium_gap = float(np.max(np.abs(sampled_pressure - (load + effective))) / load)
    return {
        "pressure-relative-l2-error": _terzaghi_finite(
            float(np.linalg.norm(pressure - expected) / np.linalg.norm(expected))
        ),
        "settlement-relative-error": _terzaghi_finite(
            abs(settlement - expected_settlement) / expected_settlement
        ),
        "quasi-static-pressure-gap": _terzaghi_finite(equilibrium_gap),
        "pressure-load-scaled-linf-error": _terzaghi_finite(
            float(np.max(np.abs(pressure - expected))) / load
        ),
        "settlement-m": _terzaghi_finite(settlement),
        "reference-settlement-m": _terzaghi_finite(expected_settlement),
    }


def _terzaghi_run_level(cells: int) -> dict[str, Any]:
    speed, wave_time, cv, beta, diameter = _terzaghi_properties()
    h = _TERZAGHI_LENGTH / cells
    drainage_length = _TERZAGHI_LENGTH - 0.5 * h
    load = _TERZAGHI_MODULUS * _TERZAGHI_LOAD_STRAIN
    bounds = jnp.asarray([[0.0], [_TERZAGHI_LENGTH]], dtype=jnp.float64)
    # The declared computational halo retains complete cell-assignment support
    # when physical particles move off their initial MAC cell centers.
    cell_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells + 2),), axis_names=("x",)
    ).prepare(jnp.asarray([[-h], [_TERZAGHI_LENGTH + h]], dtype=jnp.float64))
    mac = phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(cell_grid).prepare()
    ).prepare()
    nodal_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(cells + 5, endpoint=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[-2.0 * h], [_TERZAGHI_LENGTH + 2.0 * h]], dtype=jnp.float64))
    coordinates = (np.arange(cells, dtype=np.float64) + 0.5) * h
    position = jnp.asarray(coordinates[:, None], dtype=jnp.float64)
    volume = jnp.full((cells,), h, dtype=jnp.float64)
    solid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(cells, dtype=jnp.int32),
        _TERZAGHI_SOLID_DENSITY * (1.0 - _TERZAGHI_POROSITY) * volume,
        ambient_dimension=1,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        nodal_grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(solid_particles)
    top_load = phx.discretization.mpm.PreparedMPMBoundaryLoad(
        splat,
        jnp.asarray([[_TERZAGHI_LENGTH]], dtype=jnp.float64),
        jnp.asarray([[1.0]], dtype=jnp.float64),
        phx.applications.solid_mechanics.ReferenceDeadTraction(
            jnp.asarray([-load], dtype=jnp.float64)
        ),
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "terzaghi-full-double-point-column",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
            boundary_loads=(top_load,),
        ),
        solid_particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(acoustic_cfl=0.4),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(False,), support_margin=h
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters(
            jnp.asarray(_TERZAGHI_MODULUS / 4.0, dtype=jnp.float64),
            jnp.asarray(_TERZAGHI_MODULUS / 2.0, dtype=jnp.float64),
        ),
        mechanical_load_states=(phx.applications.solid_mechanics.MechanicalLoadState(),),
    )
    solid = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    fluid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(cells, dtype=jnp.int32),
        _TERZAGHI_FLUID_DENSITY * _TERZAGHI_POROSITY * volume,
        ambient_dimension=1,
    ).prepare()
    fluid_transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(
        fluid_particles
    )
    transfer = phx.applications.solid_mechanics.MPMToMACFaceTransferPlan(
        compiled.dynamics, mac, maximum_routes=2 * (cells + 3)
    ).prepare()
    reservoir = np.zeros((cells + 2,), dtype=np.bool_)
    # Exterior rows are declared datums, not unconstrained mixture continuity.
    # Masks at the INTERNAL physical base prevent artificial bottom drainage.
    reservoir[0] = True
    reservoir[-2:] = True
    nodal_x = np.asarray(splat.layout.coordinates_by_axis[0], dtype=np.float64)
    solid_free = ((nodal_x > 0.0) & (nodal_x <= _TERZAGHI_LENGTH))[:, None]
    face_x = (np.arange(cells + 3, dtype=np.float64) - 1.0) * h
    fluid_free = (face_x > 0.0) & (face_x <= _TERZAGHI_LENGTH)
    plan = phx.solver.DoublePointMPMPlan(
        compiled.dynamics,
        fluid_transfer,
        transfer,
        phx.equations.InterphaseDragLawPlan("stokes"),
        solid_density=_TERZAGHI_SOLID_DENSITY,
        fluid_density=_TERZAGHI_FLUID_DENSITY,
        fluid_viscosity=_TERZAGHI_VISCOSITY,
        grain_diameter=diameter,
        pressure_mask=reservoir,
        pressure_values=0.0,
        solid_free=solid_free,
        fluid_free=(fluid_free,),
        tolerance=_TERZAGHI_SOLVER_TOLERANCE,
        maximum_iterations=max(128, 8 * cells),
    )
    fluid = phx.discretization.flip.FLIPRuntimeState(
        particles=phx.discretization.flip.FLIPParticleState(
            position, jnp.zeros_like(position)
        ),
        pressure=jnp.where(
            jnp.asarray(reservoir, dtype=jnp.bool_),
            0.0,
            jnp.asarray(load, dtype=jnp.float64),
        ),
        time=jnp.asarray(0.0, dtype=jnp.float64),
        accepted_step=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(0, dtype=jnp.int32),
        geometry_epoch=jnp.asarray(-1, dtype=jnp.int32),
        geometry_id="",
    )
    state = plan.initialize(solid, fluid)
    initial_state_bytes = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree_util.tree_leaves(state)
        if isinstance(leaf, Array)
    )
    target = _TERZAGHI_TIME_FACTORS[-1] * drainage_length**2 / cv
    dt_bound = _TERZAGHI_CFL * h / (1.01 * speed)
    planned_steps = math.ceil(target / dt_bound)
    dt = target / planned_steps
    attempted = 0
    all_successful = True
    diagnostics = np.zeros((9,), dtype=np.float64)
    snapshots: list[dict[str, Any]] = []
    started = time.perf_counter()
    for time_factor in _TERZAGHI_TIME_FACTORS:
        desired_count = math.ceil(
            time_factor / _TERZAGHI_TIME_FACTORS[-1] * planned_steps
        )
        count = min(desired_count - attempted, _TERZAGHI_MAXIMUM_STEPS - attempted)
        if count <= 0 or not all_successful:
            break
        state, device_count, success, device_diagnostics = _terzaghi_advance_bounded(
            plan,
            state,
            arguments,
            jnp.asarray(dt, dtype=jnp.float64),
            jnp.asarray(count, dtype=jnp.int32),
        )
        actual_count = int(device_count)
        attempted += actual_count
        all_successful = bool(success)
        observed = np.asarray(jax.device_get(device_diagnostics), dtype=np.float64)
        diagnostics[[0, 1, 6, 7, 8]] = observed[[0, 1, 6, 7, 8]]
        diagnostics[2:6] = np.maximum(diagnostics[2:6], observed[2:6])
        elapsed = float(state.solid.time)
        snapshots.append(
            {
                "requested_time_factor": time_factor,
                "actual_time_s": elapsed,
                "actual_time_factor": cv * elapsed / drainage_length**2,
                "metrics": _terzaghi_errors(
                    state, coordinates, drainage_length, elapsed, cv, load
                ),
                "successful": all_successful,
            }
        )
    elapsed = float(state.solid.time)
    metrics = _terzaghi_errors(state, coordinates, drainage_length, target, cv, load)
    accepted_time_metrics = _terzaghi_errors(
        state, coordinates, drainage_length, elapsed, cv, load
    )
    completed = int(state.solid.accepted_step) == planned_steps
    gap = metrics["quasi-static-pressure-gap"]
    quasi_static = (
        gap is not None
        and gap <= _TERZAGHI_QUASI_STATIC_GAP_LIMIT
        and elapsed / wave_time >= 10.0
        and _TERZAGHI_LOAD_STRAIN <= 1.0e-3
        and math.isfinite(float(diagnostics[5]))
        and diagnostics[5] <= 1.0
    )
    finite_metrics = all(value is not None for value in metrics.values())
    successful = all_successful and finite_metrics
    inconclusive = successful and (not completed or not quasi_static)
    if not all_successful:
        reason = "Full coupled solver atomically rejected; no retry, pressure reduction, or replacement solver was used."
    elif not finite_metrics:
        reason = "Full coupled solver produced a nonfinite qualification observable."
    elif not completed:
        reason = "Explicit attempt cap exhausted before target consolidation time; target pressure gap is measured, not substituted."
    elif not quasi_static:
        reason = "Full coupled run completed but measured pressure-equilibrium gap or wave/strain/Re bound does not admit the Terzaghi quasi-static limit."
    else:
        reason = "Full coupled run reached the target and admitted the measured quasi-static limit; errors remain candidate evidence, not approval."
    return {
        "resolution": cells,
        "metrics": {
            key: metrics[key]
            for key in ("pressure-relative-l2-error", "settlement-relative-error")
        },
        "successful": successful,
        "inconclusive": inconclusive,
        "reason": reason,
        "raw": {
            "quasi-static-pressure-gap": gap,
            "target-time-observables": metrics,
            "accepted-time-observables": accepted_time_metrics,
            "convergence_observations": snapshots,
            "actual_time_s": elapsed,
            "target_time_s": target,
            "target_time_fraction_reached": elapsed / target,
            "wave_transits_reached": elapsed / wave_time,
            "quasi_static_admitted": quasi_static,
            "solver_status": int(diagnostics[0]),
            "projection_linear_status": int(diagnostics[1]),
            "solid_rejection_reasons": int(diagnostics[8]),
            "maximum_projection_iterations": int(diagnostics[2]),
            "maximum_projection_residual_norm": _terzaghi_finite(float(diagnostics[3])),
            "maximum_drag_residual_norm": _terzaghi_finite(float(diagnostics[4])),
            "maximum_drag_reynolds_number": _terzaghi_finite(float(diagnostics[5])),
            "last_attempt_porosity_min": _terzaghi_finite(float(diagnostics[6])),
            "last_attempt_porosity_max": _terzaghi_finite(float(diagnostics[7])),
            "pressure_initial_pa": load,
            "initial_drag_beta_pa_s_m2": beta,
            "drainage_length_m": drainage_length,
            "settlement_reference_includes_drained_half_cell_cap": True,
        },
        "resources": {
            "planned_steps": planned_steps,
            "attempted_steps": attempted,
            "accepted_steps": int(state.solid.accepted_step),
            "maximum_attempts": _TERZAGHI_MAXIMUM_STEPS,
            "step_size_s": dt,
            "acoustic_step_bound_s": dt_bound,
            "particle_count_per_phase": cells,
            "solid_nodal_count": cells + 5,
            "mac_computational_cells": cells + 2,
            "initial_state_array_bytes": initial_state_bytes,
            "wall_seconds_including_compilation": time.perf_counter() - started,
            "compiled_device_loop": True,
            "retry_count": 0,
        },
    }


def run_terzaghi_1d_consolidation() -> dict[str, Any]:
    """Run the bounded full S13 solver; never run the operator-only reduction.

    A solver rejection is failed evidence. A completed/in-cap campaign that has
    not attained its measured quasi-static limit is successful but inconclusive.
    Pressure and actual loaded-boundary settlement are compared to independent
    host references at the prescribed target; partial runs also report errors
    at their actual accepted time and target-time fraction. The refinement
    sequence does not turn an inconclusive finite-inertia run into qualification.
    """
    if not jax.config.x64_enabled:
        raise ValueError(
            "Terzaghi qualification requires jax_enable_x64 before execution."
        )
    levels = [_terzaghi_run_level(cells) for cells in _TERZAGHI_RESOLUTIONS]
    successful = all(level["successful"] for level in levels)
    inconclusive = successful and any(level["inconclusive"] for level in levels)
    finest = levels[-1]
    return {
        "resolutions": list(_TERZAGHI_RESOLUTIONS),
        "metrics": finest["metrics"],
        "levels": levels,
        "successful": successful,
        "inconclusive": inconclusive,
        "reason": (
            "One or more full-solver levels rejected or produced invalid observables."
            if not successful
            else (
                "Bounded full-solver evidence is inconclusive: target time or quasi-static admission was not attained."
                if inconclusive
                else "Full coupled refinement campaign completed; proposed criteria are not approval claims."
            )
        ),
        "settings": terzaghi_1d_consolidation_settings(),
        "raw": {"quasi-static-pressure-gap": finest["raw"]["quasi-static-pressure-gap"]},
        "resources": {
            "maximum_attempts_total": len(_TERZAGHI_RESOLUTIONS)
            * _TERZAGHI_MAXIMUM_STEPS,
            "actual_attempts_total": sum(
                level["resources"]["attempted_steps"] for level in levels
            ),
            "accepted_steps_total": sum(
                level["resources"]["accepted_steps"] for level in levels
            ),
            "wall_seconds_including_compilation": sum(
                level["resources"]["wall_seconds_including_compilation"]
                for level in levels
            ),
            "compiled_device_loop": True,
            "operator_only_reduction": False,
        },
    }


if __name__ == "__main__":
    main()
