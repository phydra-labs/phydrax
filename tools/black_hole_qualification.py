#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run bounded deterministic qualification profiles for black-hole closure APIs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


_PROFILE_METADATA: dict[str, dict[str, object]] = {
    "geometry": {
        "support": {
            "exercised": {
                "mass": 2.0,
                "specific_spin": 0.7,
                "exterior_point": [0.3, 7.0, 1.1, -0.2],
                "horizon_polar_angle": 1.0,
                "charts": ["boyer-lindquist", "future-ingoing-kerr"],
                "convention": "mostly-plus",
            },
            "api_domain": "finite stationary Kerr inputs; horizon helpers distinguish "
            "subextremal, extremal, overextremal, and invalid branches",
        },
        "references": [
            "Exact Kerr chart pullback and analytic horizon-determinant identities."
        ],
        "limitations": [
            "Only the listed subextremal point and horizon are qualified; the broader API domain is not swept.",
            "This profile does not qualify a dynamical spacetime or numerical metric.",
        ],
    },
    "thermodynamics": {
        "support": {
            "exercised": {
                "mass": 2.0,
                "angular_momentum": 0.8,
                "mass_tangent": 0.3,
                "angular_momentum_tangent": -0.2,
                "scale": "SI",
            },
            "api_domain": "stationary Kerr branch classification and explicitly scaled equilibrium thermodynamics",
        },
        "references": [
            "Kerr Killing-horizon first-law and Smarr identities evaluated from the landed implementation."
        ],
        "limitations": [
            "Only the listed smooth subextremal directional derivative is qualified.",
            "Stationary Killing-horizon evidence is not an apparent, dynamical, trapping, or event-horizon claim.",
            "No quantum or higher-curvature entropy correction is claimed.",
        ],
    },
    "rays": {
        "support": {
            "exercised": {
                "metric": "four-dimensional-Minkowski-mostly-plus",
                "scale": "geometric-G=c=1-with-kilogram-reference",
                "convention": "canonical-mostly-plus",
                "coordinate_unit": "declared-geometric-length-unit",
                "affine_parameter_unit": "declared-geometric-length-unit",
                "initial_coordinates": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                "initial_tangents": [[1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                "history_affine_nodes": [0.0, 0.125, 0.25, 0.375, 0.5],
                "capture_and_escape_affine_root": 0.25,
            },
            "api_domain": "fixed-capacity null and timelike rays on four-dimensional "
            "Lorentzian metrics with ordered events",
        },
        "references": [
            "Analytic straight null trajectories, event roots, and metric-orthonormal observer-screen identities."
        ],
        "limitations": [
            "Only two analytic Minkowski null lanes are qualified.",
            "The control does not establish image convergence in a curved production spacetime.",
        ],
    },
    "grrt-interferometry": {
        "support": {
            "exercised": {
                "scalar_slab": {
                    "length": 2.0,
                    "emission": 4.0,
                    "extinction": 0.5,
                    "incident_intensity": 1.0,
                },
                "point_source": {
                    "angular_position_rad": [0.125, -0.25],
                    "flux_jy": 3.5,
                    "frequency_hz": 230000000000.0,
                    "uv_wavelengths": [[0.0, 0.0], [1.5, -0.5]],
                },
                "closure_stations": ["A", "B", "C", "D"],
            },
            "api_domain": "fixed Stokes screens, invariant transfer, direct visibility "
            "sums, and fixed-topology closure products",
        },
        "references": [
            "Closed-form homogeneous transfer slab, point-source Fourier phase, "
            "and station-gain cancellation identities."
        ],
        "limitations": [
            "Synthetic analytic controls are not an observational calibration or an external GRRT comparison.",
            "No adaptive camera, slow-light worldtube, or polarized microphysics convergence claim is made.",
        ],
    },
    "perturbations-scattering-hawking": {
        "support": {
            "exercised": {
                "qnm": {
                    "background": "Schwarzschild-M1",
                    "mode": "spin-minus2-l2-m2-n0-Regge-Wheeler",
                    "reference": "schwarzschild-leaver-Momega-reference-v1",
                    "reference_Momega": [0.373671684418042, -0.088962315688936],
                    "node_count": 65,
                    "outer_radius": 30.0,
                    "integration_substeps": 32,
                    "infinity_asymptotic_order": 12,
                    "matching_tolerance": 1.0e-7,
                    "relative_ode_residual_tolerance": 1.0e-5,
                },
                "radial": {
                    "background": "Schwarzschild-M1",
                    "mode": "scalar-l0-m0",
                    "frequency": [0.4, -0.05],
                    "node_count": 17,
                    "outer_radius": 40.0,
                },
                "scattering": {
                    "background": "Schwarzschild-M1",
                    "mode": "scalar-l0-m0",
                    "central_frequency": 0.02,
                    "neighboring_frequencies": [0.019, 0.0195, 0.0205, 0.021],
                    "node_count": 65,
                    "inner_radius": 2.00002,
                    "outer_radius": 80.0,
                    "integration_substeps": [16, 32],
                    "infinity_asymptotic_order": 3,
                    "absolute_flux_tolerance": 1.0e-8,
                    "refinement_tolerance": 1.0e-5,
                    "dimensionless_slope_refinement_tolerance": 5.0e-3,
                    "low_frequency_relative_tolerance": 0.25,
                },
                "hawking": {
                    "species": "massless-scalar",
                    "mode": "l0-m0",
                    "frequency_nodes": [0.015, 0.02, 0.025],
                    "greybody_source": "computed-qualified-radial-solves",
                    "tail_evidence_qualified": False,
                },
            },
            "api_domain": "separated radial perturbations, coupled QNM roots, "
            "real-frequency flux ledgers, and explicitly evidenced bounded Hawking quadrature",
        },
        "references": [
            "Maintained Schwarzschild Leaver Mω reference v1 for the spin -2, l=2, n=0 fundamental.",
            "Strict two-sided QNM residuals plus computed Schwarzschild scattering "
            "decomposition, refinement, asymptotic, low-frequency, Wronskian, and "
            "flux evidence.",
        ],
        "limitations": [
            "No external greybody reference artifact is used; greybody factors "
            "come from the bounded native radial solve.",
            "The QNM evidence qualifies only the listed Schwarzschild fundamental.",
            "The Hawking spectrum remains unqualified because omitted-tail evidence is deliberately absent.",
            "No astrophysical evaporation-time prediction is claimed.",
        ],
    },
    "grhd-grmhd": {
        "support": {
            "exercised": {
                "eos": ["gamma-law-5/3", "gamma-law-4/3"],
                "geometry": "Minkowski-Cartesian",
                "grhd": {
                    "cells": 6,
                    "topology": "one-dimensional-periodic",
                    "integrator": "SSPRK33",
                    "step_size": 0.001,
                },
                "grmhd": {
                    "cells": [2, 2],
                    "topology": "two-dimensional-periodic",
                    "integrator": "vector-potential-CT-SSPRK33",
                    "step_size": 0.0001,
                },
                "accretion": "Michel-Bondi-critical-radius-M1",
            },
            "api_domain": "SRHD, Valencia GRHD, ideal Valencia GRMHD, constrained "
            "transport, and stationary accretion initial data",
        },
        "references": [
            "Uniform-state conservation, Valencia primitive recovery, "
            "constrained-transport Faraday balance, and Michel integrals."
        ],
        "limitations": [
            "Only the listed uniform/runtime and critical-radius controls are qualified.",
            "The controls are not shock-convergence, turbulent-accretion, or multi-dimensional production evidence.",
        ],
    },
    "z4c-initial-data-horizons-waves": {
        "support": {
            "exercised": {
                "z4c_grid_shape": [5, 5, 5],
                "z4c_topology": "periodic-fixed-grid-SSPRK33",
                "z4c_step_size": 0.05,
                "initial_data": "isotropic-Schwarzschild-M1-at-r4",
                "horizon": "spherical-Schwarzschild-isotropic-r0.5-bandlimit3",
                "wave": "vacuum-Psi4-sign-control-bandlimit3",
            },
            "api_domain": "fixed-grid Z4c, analytic and puncture initial data, spherical "
            "MOTS geometry, and fixed-shape wave extraction",
        },
        "references": [
            "Flat-vacuum Z4c fixed point, isotropic Schwarzschild ADM constraints, "
            "exact area/expansion, and Psi4 sign identities."
        ],
        "limitations": [
            "Only the listed flat and single-hole controls are qualified.",
            "This is not binary-black-hole evolution, waveform convergence, or merger-remnant validation.",
        ],
    },
    "coupling": {
        "support": {
            "exercised": {
                "integrator": "same-stage-SSPRK33",
                "matter": "analytic-GRHD-participant",
                "topology": "fixed-two-lane",
                "initial_z4c": 0.4,
                "initial_matter": 1.0,
                "step_size": 0.1,
            },
            "api_domain": "atomic same-stage Z4c coupling to GRHD or GRMHD participant adapters",
        },
        "references": [
            "Analytic SSPRK33 recurrence with exact stage addresses, participant updates, and closed ledgers."
        ],
        "limitations": [
            "The coordinator control uses analytic participant callbacks; physical "
            "GRHD and Z4c kernels are qualified in their own profiles."
        ],
    },
    "scaling-restart": {
        "support": {
            "exercised": {
                "formulation": "Z4c",
                "global_shape": [4, 4, 4],
                "decomposition": [1, 1, 1],
                "device_count": 1,
                "checkpoint_state_shape": [2, 2, 2],
                "restart_relation": "exact-same-topology",
            },
            "api_domain": "named fixed-grid distribution plus exact and tolerance-controlled restart relations",
        },
        "references": [
            "Named sharding/halo layout and exact content-addressed checkpoint round trip."
        ],
        "limitations": [
            "Single-device execution is not multi-host scaling or performance evidence.",
            "Only exact same-topology restart is admitted by this control.",
        ],
    },
    "production-interchange": {
        "support": {
            "exercised": {
                "runtime": "5x5x5-periodic-flat-vacuum-Z4c-SSPRK33",
                "step_size": 0.01,
                "artifact": "local-checksum-pinned-opaque-inert-field-bytes",
                "source_format": "opaque-binary",
                "license": "CC0-1.0",
                "backend": "local-default-JAX",
            },
            "api_domain": "exact production bindings and host-only rights-aware neutral artifact admission",
        },
        "references": [
            "Exact production-domain/support/resolved-run binding and checksum-first inert artifact mapping."
        ],
        "limitations": [
            "The opaque bytes are admitted and bound but are not parsed or claimed to be openPMD/HDF5 field data.",
            "Technical production qualification is not PNPL deployment authorization.",
            "The control neither downloads nor deserializes executable or pickle content.",
        ],
    },
    "waveforms-remnants": {
        "support": {
            "exercised": {
                "surrogate": {
                    "representation": "two-node-polynomial-EIM-l2m2",
                    "mass_ratio": 2.0,
                    "aligned_spins": [0.2, -0.1],
                    "geometric_times": [-0.5, 0.0, 0.5],
                    "frame": "qualification-inertial",
                },
                "match": {
                    "sample_count": 64,
                    "sample_interval_seconds": 0.015625,
                    "injected_lag_samples": 5,
                    "global_phase_radians": 0.7,
                },
                "remnant": {
                    "fit": "UIB2016v2",
                    "component_masses": [30.0, 30.0],
                    "aligned_spins": [0.0, 0.0],
                },
            },
            "api_domain": "normalized aligned-spin polynomial-EIM mode surrogates, "
            "canonical-grid PSD-weighted comparison, and bounded nonprecessing "
            "binary-remnant fits",
        },
        "references": [
            "Analytic polynomial-node reconstruction and nonprecessing mode symmetry.",
            "Exact phase/time-shifted self-match on a canonical one-sided FFT grid.",
            "UIB2016v2 equal-mass nonspinning final-spin and radiated-energy reference.",
        ],
        "limitations": [
            "Caller-supplied surrogate arrays are numerically exercised but remain unauthenticated and unqualified.",
            "The match search is discrete and does not establish differentiability through argmax.",
            "The remnant fit is qualified only inside the declared q<=8 and |chi|<=0.8 envelope.",
            "No external model file, waveform table, or runtime is loaded.",
        ],
    },
    "advanced": {
        "support": {
            "exercised": {
                "thermodynamics": "Kerr-Newman-AdS-M2-a0.3-Q0.2-L10",
                "massive_field": "Schwarzschild-M2-mu0.1-l1-n0-hydrogenic-seed",
                "self_force": "first-order-mode-sum-ell-max14",
                "inverse": "subextremal-Kerr-M2-a0.4-direction-[0.25,-0.1]",
                "plasma": "two-temperature-[100,400]-dt2",
                "radiation": "single-cell-GR-grey-M1",
                "electromagnetism": "single-cell-Ohm-and-force-free",
                "characteristic": "flat-nine-time-three-radius-l2m2",
                "bms": "octahedral-l<=1-rest-mass2",
                "event_horizon": "complete-flat-five-time-two-generator-unqualified-terminal",
            },
            "api_domain": "advanced compact-object, inverse/UQ, radiation/plasma, "
            "characteristic/BMS, and offline horizon branches",
        },
        "references": [
            "Analytic compact-object, plasma, radiation, characteristic, BMS, and fixed-branch derivative identities."
        ],
        "limitations": [
            "Only the listed advanced controls are qualified.",
            "No second-order self-force, branch-cut Green function, full kinetic "
            "plasma, or multifrequency radiation validation is supplied.",
            "The offline event-horizon result remains unqualified without an independently qualified terminal surface.",
            "No learned-closure admission, posterior inference, or external reference artifact is claimed.",
        ],
    },
}

PROFILE_NAMES = tuple(_PROFILE_METADATA)
_RUNTIME_DEPENDENCIES = (
    "diffrax",
    "equinox",
    "jax",
    "jaxlib",
    "jaxtyping",
    "lineax",
    "numpy",
    "optimistix",
)


def _content_id(kind: str, payload: Mapping[str, object], /) -> str:
    encoded = json.dumps(
        {"kind": kind, **payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_identity() -> dict[str, object]:
    import phydrax

    package_root = Path(phydrax.__file__).resolve().parent
    source_files = tuple(sorted(package_root.rglob("*.py")))
    digest = hashlib.sha256()
    for source in source_files:
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        content = source.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    runner_bytes = Path(__file__).read_bytes()
    return {
        "package": "phydrax",
        "package_version": importlib.metadata.version("phydrax"),
        "python_source_file_count": len(source_files),
        "python_source_tree_sha256": digest.hexdigest(),
        "qualification_runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
    }


def _runtime_manifest(valid_at: int, /) -> dict[str, object]:
    if isinstance(valid_at, bool) or not isinstance(valid_at, int) or valid_at < 0:
        raise ValueError("valid_at must be a nonnegative Unix timestamp integer.")
    devices = tuple(jax.devices())
    if len(devices) > 256:
        raise ValueError("Qualification runtime manifests support at most 256 devices.")
    validity = {
        "evaluated_at_unix_seconds": valid_at,
        "policy": "observation-time-only; no future validity is implied",
        "expires_at_unix_seconds": valid_at,
    }
    validity["policy_id"] = _content_id("black-hole-validity-policy", validity)
    manifest: dict[str, object] = {
        "source": _source_tree_identity(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in _RUNTIME_DEPENDENCIES
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "jax_backend": jax.default_backend(),
            "jax_process_count": jax.process_count(),
            "jax_process_index": jax.process_index(),
        },
        "devices": [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
        "precision": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "scientific_profiles": "float64",
            "production_and_native_electromagnetic_controls": "float32",
        },
        "validity": validity,
    }
    manifest["manifest_id"] = _content_id("black-hole-runtime-manifest", manifest)
    return manifest


def _all_true(value: object) -> bool:
    return bool(np.all(np.asarray(value, dtype=bool)))


def _maximum_absolute(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _profile_result(
    name: str,
    /,
    *,
    checks: Mapping[str, object],
    flags: Mapping[str, object],
    residuals: Mapping[str, object],
    statuses: Mapping[str, object],
    identities: Mapping[str, object],
) -> dict[str, object]:
    normalized_checks = {key: _all_true(value) for key, value in checks.items()}
    normalized_flags = {key: _all_true(value) for key, value in flags.items()}
    normalized_residuals: dict[str, float | None] = {}
    residuals_finite = True
    for key, value in residuals.items():
        scalar = float(value)
        finite = math.isfinite(scalar)
        normalized_residuals[key] = scalar if finite else None
        residuals_finite = residuals_finite and finite
    normalized_checks["reported_residuals_finite"] = residuals_finite
    normalized_statuses = {key: int(value) for key, value in statuses.items()}
    metadata = _PROFILE_METADATA[name]
    return {
        "profile": name,
        "passed": all(normalized_checks.values()),
        "checks": normalized_checks,
        "flags": normalized_flags,
        "residuals": normalized_residuals,
        "statuses": normalized_statuses,
        "identities": dict(identities),
        "support": metadata["support"],
        "references": metadata["references"],
        "limitations": metadata["limitations"],
    }


def _flat_adm_geometry(
    shape: tuple[int, ...],
    /,
    *,
    scale_id: str,
    convention_id: str,
    topology_id: str,
    geometry_lineage_id: str,
    dtype: Any = jnp.float64,
):
    from phydrax.metrix._adm_exchange import ADMGridGeometry

    identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape, dtype=dtype),
        jnp.zeros(shape + (3,), dtype=dtype),
        identity,
        identity,
        jnp.ones(shape, dtype=dtype),
        jnp.zeros(shape + (3, 3), dtype=dtype),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        chart_id="cartesian",
        convention_id=convention_id,
        scale_id=scale_id,
        topology_id=topology_id,
        geometry_lineage_id=geometry_lineage_id,
        snapshot_token=0,
    )


def qualify_geometry() -> dict[str, object]:
    from phydrax.metrix._black_hole_metrics import (
        boyer_lindquist_to_ingoing_kerr_transition,
        ingoing_kerr_domain_evidence,
        ingoing_kerr_metric,
        kerr_boyer_lindquist_domain_evidence,
        kerr_boyer_lindquist_metric,
        kerr_horizon_radii,
        kerr_kretschmann_scalar,
        kerr_pontryagin_scalar,
    )
    from phydrax.metrix._chart import CoordinateChart

    mass = 2.0
    spin = 0.7
    boyer_lindquist = CoordinateChart(
        "qualification-kerr-boyer-lindquist", ("t", "r", "theta", "phi")
    )
    ingoing = CoordinateChart(
        "qualification-kerr-future-ingoing", ("v", "r", "theta", "phi_tilde")
    )
    point = jnp.asarray((0.3, 7.0, 1.1, -0.2), dtype=jnp.float64)
    transition = boyer_lindquist_to_ingoing_kerr_transition(
        mass, spin, source=boyer_lindquist, target=ingoing
    )
    mapped = transition(point)
    jacobian = transition.jacobian(point)
    boyer_metric = kerr_boyer_lindquist_metric(mass, spin, chart=boyer_lindquist)
    ingoing_metric = ingoing_kerr_metric(mass, spin, chart=ingoing)
    pullback = jacobian.T @ ingoing_metric(mapped) @ jacobian
    pullback_residual = _maximum_absolute(pullback - boyer_metric(point))
    inverse_residual = _maximum_absolute(transition.inverse(mapped) - point)

    outer = kerr_horizon_radii(mass, spin)[1]
    horizon_point = jnp.asarray((0.0, outer, 1.0, 0.0), dtype=jnp.float64)
    horizon_matrix = ingoing_metric(horizon_point)
    sigma = outer**2 + spin**2 * jnp.cos(horizon_point[2]) ** 2
    expected_determinant = -((sigma * jnp.sin(horizon_point[2])) ** 2)
    determinant_residual = float(
        jnp.abs(jnp.linalg.det(horizon_matrix) - expected_determinant)
    )
    exterior_domain = kerr_boyer_lindquist_domain_evidence(
        mass, spin, point, chart=boyer_lindquist
    )
    horizon_domain = ingoing_kerr_domain_evidence(
        mass, spin, horizon_point, chart=ingoing
    )
    kretschmann = kerr_kretschmann_scalar(mass, spin, point[1], point[2])
    pontryagin = kerr_pontryagin_scalar(mass, spin, point[1], point[2])
    invariant_finite = jnp.isfinite(kretschmann) & jnp.isfinite(pontryagin)

    return _profile_result(
        "geometry",
        checks={
            "exterior_domain_qualified": exterior_domain.qualified,
            "future_horizon_domain_valid": horizon_domain.physically_valid,
            "future_horizon_metric_finite": jnp.all(jnp.isfinite(horizon_matrix)),
            "chart_pullback_identity": pullback_residual < 5.0e-12,
            "chart_inverse_identity": inverse_residual < 5.0e-13,
            "horizon_determinant_identity": determinant_residual < 5.0e-12,
            "curvature_invariants_finite": invariant_finite,
        },
        flags={
            "exterior_domain_finite": exterior_domain.finite,
            "exterior_domain_physically_valid": exterior_domain.physically_valid,
            "exterior_domain_derivative_valid": exterior_domain.derivative_valid,
            "future_horizon_domain_finite": horizon_domain.finite,
            "future_horizon_domain_physically_valid": horizon_domain.physically_valid,
            "curvature_invariants_finite": invariant_finite,
        },
        residuals={
            "chart_pullback_max_abs": pullback_residual,
            "chart_inverse_max_abs": inverse_residual,
            "future_horizon_determinant_abs": determinant_residual,
            "kretschmann_scalar": float(kretschmann),
            "pontryagin_scalar": float(pontryagin),
        },
        statuses={
            "exterior_domain": exterior_domain.status,
            "future_horizon_domain": horizon_domain.status,
        },
        identities={
            "source_chart": boyer_lindquist.name,
            "target_chart": ingoing.name,
            "exterior_domain_id": exterior_domain.domain_id,
            "future_horizon_domain_id": horizon_domain.domain_id,
        },
    )


def qualify_thermodynamics() -> dict[str, object]:
    from phydrax._physical import RelativityScaleContract
    from phydrax.applications.compact_objects._black_hole_thermodynamics import (
        evaluate_kerr_entropy_temperature,
        evaluate_kerr_first_law,
        evaluate_stationary_kerr_horizon,
        KerrInput,
    )

    parameters = KerrInput(2.0, 0.8)
    horizon = evaluate_stationary_kerr_horizon(parameters)
    scale = RelativityScaleContract.si()
    thermal = evaluate_kerr_entropy_temperature(horizon, scale)
    first_law = evaluate_kerr_first_law(parameters, 0.3, -0.2)
    area_identity = float(
        jnp.abs(horizon.area - 8.0 * jnp.pi * parameters.mass * horizon.outer_radius)
    )

    return _profile_result(
        "thermodynamics",
        checks={
            "stationary_horizon_qualified": horizon.qualified,
            "scaled_entropy_temperature_qualified": thermal.qualified,
            "first_law_qualified": first_law.qualified,
            "first_law_identity": jnp.abs(first_law.first_law_residual) < 5.0e-13,
            "smarr_identity": jnp.abs(first_law.smarr_residual) < 5.0e-13,
            "area_identity": area_identity < 5.0e-13,
        },
        flags={
            "horizon_finite": horizon.finite,
            "horizon_converged": horizon.converged,
            "horizon_physically_valid": horizon.physically_valid,
            "horizon_derivative_valid": horizon.derivative_valid,
            "thermal_finite": thermal.finite,
            "thermal_physically_valid": thermal.physically_valid,
            "first_law_satisfied": first_law.first_law_satisfied,
            "smarr_satisfied": first_law.smarr_satisfied,
            "first_law_derivative_valid": first_law.derivative_valid,
        },
        residuals={
            "first_law_abs": float(jnp.abs(first_law.first_law_residual)),
            "normalized_first_law_abs": float(
                jnp.abs(first_law.normalized_first_law_residual)
            ),
            "smarr_abs": float(jnp.abs(first_law.smarr_residual)),
            "normalized_smarr_abs": float(jnp.abs(first_law.normalized_smarr_residual)),
            "area_identity_abs": area_identity,
            "temperature": float(thermal.temperature),
            "entropy": float(thermal.entropy),
        },
        statuses={"kerr_branch": horizon.branch.code},
        identities={
            "kerr_input_id": parameters.input_id,
            "scale_id": scale.scale_id,
        },
    )


def qualify_rays() -> dict[str, object]:
    from phydrax._physical import RelativityScaleContract
    from phydrax.applications.astrophysics._gr_events import GRRayEventCode
    from phydrax.applications.astrophysics._gr_rays import (
        GRRayPlan,
        GRRayState,
        trace_gr_rays,
    )
    from phydrax.applications.astrophysics._gr_screens import GRObserverScreenPlan
    from phydrax.applications.astrophysics._gr_status import GRRayStatus
    from phydrax.metrix._chart import CoordinateChart
    from phydrax.metrix._lorentzian import minkowski_metric
    from phydrax.metrix._spacetime_conventions import RelativityConvention
    from phydrax.units import KILOGRAM

    chart = CoordinateChart("qualification-minkowski", ("t", "x", "y", "z"))
    metric = minkowski_metric(chart)
    scale = RelativityScaleContract.geometric(KILOGRAM)
    convention = RelativityConvention.canonical()
    coordinate_unit = scale.dimensional_scale.length_unit
    affine_parameter_unit = scale.dimensional_scale.length_unit
    screen_plan = GRObserverScreenPlan(
        metric,
        jnp.zeros(4),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 0.0, 1.0)),
        jnp.asarray((0.0, 0.0, 1.0, 0.0)),
        jnp.asarray(((0.0, 0.0), (0.1, -0.2))),
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        temporal_direction="past",
        plan_id="qualification:observer-screen:flat",
    )
    screen = screen_plan.initialize()

    state = GRRayState(
        jnp.zeros((2, 4), dtype=jnp.float64),
        jnp.asarray(((1.0, -1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0))),
    )
    plan = GRRayPlan(
        metric,
        state,
        jnp.linspace(0.0, 0.5, 5),
        capture_margin=lambda affine, point, tangent: point[1] + 0.25,
        escape_margin=lambda affine, point, tangent: 0.25 - point[1],
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        affine_budget_is_event=True,
        plan_id="qualification:null-rays:flat-events",
    )
    result = trace_gr_rays(plan)
    expected_status = jnp.asarray(
        (int(GRRayStatus.CAPTURED), int(GRRayStatus.ESCAPED)), dtype=jnp.int32
    )
    expected_events = jnp.asarray(
        (int(GRRayEventCode.CAPTURE), int(GRRayEventCode.ESCAPE)), dtype=jnp.int32
    )
    screen_tetrad_residual = float(jnp.abs(screen.tetrad_residual))
    screen_null_residual = _maximum_absolute(screen.null_residual)
    ray_null_residual = _maximum_absolute(result.null_residual)
    expected_trajectory = (
        state.coordinates[:, None, :]
        + result.affine_parameter[..., None] * state.tangents[:, None, :]
    )
    through_event = (
        jnp.arange(result.coordinates.shape[1])[None, :]
        <= result.event_ledger.history_index[:, None]
    )
    trajectory_residual = _maximum_absolute(
        jnp.where(
            through_event[..., None],
            result.coordinates - expected_trajectory,
            0.0,
        )
    )
    event_affine_residual = _maximum_absolute(result.event_ledger.affine_parameter - 0.25)

    return _profile_result(
        "rays",
        checks={
            "observer_screen_valid": screen.valid,
            "observer_tetrad_orthonormal": screen_tetrad_residual < 1.0e-13,
            "observer_rays_null": screen_null_residual < 1.0e-13,
            "terminal_statuses_exact": jnp.all(result.status == expected_status),
            "terminal_events_exact": jnp.all(result.event_code == expected_events),
            "ray_batch_qualified": result.qualified,
            "null_constraint_closed": ray_null_residual < 1.0e-10,
            "analytic_trajectory_identity": trajectory_residual < 2.0e-8,
            "analytic_event_affine_roots": event_affine_residual < 2.0e-8,
            "ray_context_identity_exact": (
                screen.metric_id == plan.metric_id
                and screen.chart_id == plan.chart_id
                and screen.convention_id == plan.convention_id
                and screen.scale_id == plan.scale_id
                and screen.coordinate_unit_id == plan.coordinate_unit_id
                and screen.affine_parameter_unit_id == plan.affine_parameter_unit_id
            ),
        },
        flags={
            "screen_all_valid": screen.valid,
            "ray_batch_finite": result.finite,
            "ray_batch_converged": result.converged,
            "ray_batch_physically_valid": result.physically_valid,
            "ray_batch_qualified": result.qualified,
            "ray_batch_derivative_valid": result.derivative_valid,
            "event_ledger_recorded": result.event_ledger.recorded,
        },
        residuals={
            "screen_tetrad_abs": screen_tetrad_residual,
            "screen_null_max_abs": screen_null_residual,
            "ray_null_max_abs": ray_null_residual,
            "analytic_trajectory_max_abs": trajectory_residual,
            "event_affine_root_max_abs": event_affine_residual,
        },
        statuses={
            "capture_lane": result.status[0],
            "escape_lane": result.status[1],
        },
        identities={
            "chart": chart.name,
            "screen_plan_id": screen.plan_id,
            "screen_result_id": screen.result_id,
            "ray_plan_id": result.plan_id,
            "ray_result_id": result.result_id,
            "event_surface_id": plan.events.event_id,
            "metric_id": plan.metric_id,
            "chart_id": plan.chart_id,
            "convention_id": plan.convention_id,
            "scale_id": plan.scale_id,
            "coordinate_unit_id": plan.coordinate_unit_id,
            "affine_parameter_unit_id": plan.affine_parameter_unit_id,
        },
    )


def qualify_grrt_interferometry() -> dict[str, object]:
    from phydrax.applications.astrophysics._gr_products import GRImageScreen, StokesImage
    from phydrax.applications.astrophysics._gr_transfer import (
        InvariantScalarTransferPlan,
        InvariantTransferUnitContract,
    )
    from phydrax.applications.astrophysics._interferometry import (
        apply_station_gains,
        closure_products,
        ClosureTopology,
        direct_stokes_visibilities,
        StokesVisibilityData,
        VisibilitySampling,
    )
    from phydrax.applications.astrophysics._photometry import ObservationDataProvenance
    from phydrax.units import (
        derived_unit,
        HERTZ,
        JANSKY,
        RADIAN,
    )

    transfer_units = InvariantTransferUnitContract.si_affine_length()
    transfer_plan = InvariantScalarTransferPlan(
        jnp.asarray((2.0,)), transfer_units, path_id="qualification:homogeneous-slab"
    )
    transfer = transfer_plan.evaluate(
        jnp.asarray((4.0,)), jnp.asarray((0.5,)), jnp.asarray(1.0)
    )
    expected_transfer = np.exp(-1.0) + 8.0 * (1.0 - np.exp(-1.0))
    transfer_residual = float(jnp.abs(transfer.invariant_intensity - expected_transfer))

    steradian = derived_unit("sr", ((RADIAN, 2),))
    flux_density = JANSKY
    intensity = derived_unit("Jy/sr", ((JANSKY, 1), (steradian, -1)))
    frequency = 230.0e9
    position = np.asarray((0.125, -0.25))
    solid_angle = np.asarray(((0.2,),))
    stokes = np.zeros((4, 1, 1))
    stokes[0, 0, 0] = 3.5 / solid_angle[0, 0]
    screen = GRImageScreen(
        position.reshape((1, 1, 2)),
        solid_angle,
        np.ones((1, 1), dtype=bool),
        angular_unit=RADIAN,
        solid_angle_unit=steradian,
    )
    image = StokesImage(
        stokes,
        screen,
        ObservationDataProvenance.native("qualification:point-source"),
        frequency=frequency,
        redshift=np.ones((1, 1)),
        redshift_valid=np.ones((1, 1), dtype=bool),
        lensing_masks=np.ones((1, 1, 1), dtype=bool),
        lensing_labels=("direct",),
        intensity_unit=intensity,
        flux_density_unit=flux_density,
        frequency_unit=HERTZ,
    )
    uv = np.asarray(((0.0, 0.0), (1.5, -0.5)))
    direct_sampling = VisibilitySampling(uv, ((0, 1), (0, 1)), frequency, ("A", "B"))
    visibility = direct_stokes_visibilities(image, direct_sampling)
    expected_visibility = 3.5 * np.exp(-2.0j * np.pi * (uv @ position))
    visibility_residual = _maximum_absolute(
        visibility.total_intensity - expected_visibility
    )

    station_ids = ("A", "B", "C", "D")
    pairs = np.asarray(((0, 1), (1, 2), (0, 2), (2, 3), (1, 3)))
    closure_sampling = VisibilitySampling(np.zeros((5, 2)), pairs, frequency, station_ids)
    topology = ClosureTopology.from_station_cycles(
        closure_sampling,
        phase_cycles=(("A", "B", "C"),),
        amplitude_cycles=(("A", "B", "C", "D"),),
    )
    values = np.zeros((4, 5), dtype=complex)
    values[0] = np.asarray(
        (
            2.0 * np.exp(0.2j),
            3.0 * np.exp(-0.4j),
            5.0 * np.exp(0.1j),
            7.0 * np.exp(0.3j),
            11.0 * np.exp(-0.2j),
        )
    )
    closure_data = StokesVisibilityData(
        values,
        closure_sampling,
        flux_density,
        ObservationDataProvenance.native("qualification:closure-data"),
    )
    closures = closure_products(closure_data, topology)
    gains = np.asarray(
        (
            1.2 * np.exp(0.7j),
            0.8 * np.exp(-0.5j),
            1.7 * np.exp(0.1j),
            0.6 * np.exp(-0.8j),
        )
    )
    transformed = closure_products(apply_station_gains(closure_data, gains), topology)
    phase_gain_residual = _maximum_absolute(
        transformed.closure_phase - closures.closure_phase
    )
    amplitude_gain_residual = _maximum_absolute(
        transformed.closure_amplitude - closures.closure_amplitude
    )

    return _profile_result(
        "grrt-interferometry",
        checks={
            "transfer_qualified": transfer.evidence.qualified,
            "homogeneous_slab_identity": transfer_residual < 2.0e-6,
            "visibility_finite": visibility.finite,
            "point_source_fourier_identity": visibility_residual < 2.0e-6,
            "closure_phase_valid": closures.phase_physically_valid,
            "closure_amplitude_valid": closures.amplitude_physically_valid,
            "closure_phase_gain_invariant": phase_gain_residual < 3.0e-6,
            "closure_amplitude_gain_invariant": amplitude_gain_residual < 3.0e-6,
        },
        flags={
            "transfer_finite": transfer.evidence.finite,
            "transfer_converged": transfer.evidence.converged,
            "transfer_physically_valid": transfer.evidence.physically_valid,
            "transfer_in_support": transfer.evidence.in_support,
            "transfer_derivative_valid": transfer.evidence.derivative_valid,
            "visibility_finite": visibility.finite,
            "closure_phase_physically_valid": closures.phase_physically_valid,
            "closure_amplitude_physically_valid": closures.amplitude_physically_valid,
            "closure_phase_derivative_valid": closures.phase_derivative_valid,
            "closure_amplitude_derivative_valid": closures.amplitude_derivative_valid,
        },
        residuals={
            "homogeneous_slab_abs": transfer_residual,
            "point_source_visibility_max_abs": visibility_residual,
            "closure_phase_gain_max_abs": phase_gain_residual,
            "closure_amplitude_gain_max_abs": amplitude_gain_residual,
            "optical_depth": float(transfer.optical_depth),
        },
        statuses={
            "closure_phase": closures.phase_status[0],
            "closure_amplitude": closures.amplitude_status[0],
        },
        identities={
            "transfer_units_id": transfer_units.units_id,
            "transfer_plan_id": transfer.plan_id,
            "image_screen_id": screen.screen_id,
            "image_content_id": image.content_id,
            "visibility_topology_id": direct_sampling.topology_id,
            "visibility_content_id": visibility.content_id,
            "closure_topology_id": topology.topology_id,
            "closure_content_id": closures.content_id,
        },
    )


def qualify_perturbations_scattering_hawking() -> dict[str, object]:
    from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
    from phydrax.applications.compact_objects._black_hole_scattering import (
        BlackHoleScatteringPlan,
        SchwarzschildScatteringSolvePlan,
        SchwarzschildScatteringStatus,
        solve_schwarzschild_scattering,
    )
    from phydrax.applications.compact_objects._black_hole_thermodynamics import (
        evaluate_kerr_entropy_temperature,
        evaluate_stationary_kerr_horizon,
        KerrInput,
    )
    from phydrax.applications.compact_objects._hawking import (
        evaluate_hawking_spectrum,
        HawkingScatteringData,
        HawkingSpectrumPlan,
        HawkingTailEvidence,
        KerrEvaporationState,
        QuantumFieldSpecies,
    )
    from phydrax.applications.compact_objects._perturbation import SeparatedMode
    from phydrax.applications.compact_objects._qnm import (
        BoundedContinuedFractionPlan,
        QnmDerivativeStatus,
        QnmSolvePlan,
        QnmStatus,
        schwarzschild_qnm_reference,
        solve_qnm,
    )
    from phydrax.applications.compact_objects._radial_perturbation import (
        evaluate_schwarzschild_radial,
        SchwarzschildRadialPlan,
    )
    from phydrax.applications.compact_objects._spheroidal import (
        SpheroidalAngularPlan,
    )
    from phydrax.linalg import DenseLU, LinearSolvePolicy
    from phydrax.nonlinear import (
        NewtonKrylov,
        NonlinearTermination,
        SensitivityPolicy,
    )

    qnm_mode = SeparatedMode(
        -2,
        2,
        2,
        overtone=0,
        sector="regge-wheeler",
        family="qnm",
        background_id="qualification:schwarzschild-M1",
    )
    qnm_angular = SpheroidalAngularPlan(
        qnm_mode,
        5,
        residual_tolerance=1.0e-9,
        isolation_tolerance=1.0e-9,
        minimum_target_overlap=1.0e-6,
        maximum_condition=1.0e10,
    )
    qnm_radial = SchwarzschildRadialPlan(
        qnm_mode,
        1.0,
        node_count=65,
        outer_radius=30.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=32,
        infinity_asymptotic_order=12,
    )
    qnm_plan = QnmSolvePlan(
        qnm_mode,
        qnm_angular,
        qnm_radial,
        jnp.asarray(0.0),
        NewtonKrylov(),
        NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=1.0e-10,
            maximum_steps=20,
        ),
        SensitivityPolicy("implicit-forward", condition_limit=1.0e12),
        BoundedContinuedFractionPlan(48, 96, 0, 1.0e-9, 1.0e-9),
        BoundedContinuedFractionPlan(96, 192, 0, 1.0e-9, 1.0e-9),
        1.0e-8,
        1.0e12,
        (-0.5, 0.5),
        branch_id="schwarzschild-s-2-l2-n0",
    )
    qnm_reference = schwarzschild_qnm_reference(qnm_mode, 2.0e-8, 2.0e-8)
    qnm = solve_qnm(
        qnm_plan,
        qnm_reference.angular_frequency * (1.0 + 1.0e-4),
        jnp.asarray(4.001 + 0.0j),
        qualification=qnm_reference,
        continuation_active=True,
    )
    qnm_matching_residual = float(jnp.abs(qnm.radial_resolution.residual))
    qnm_ode_residual = float(qnm.radial_resolution.residual_evidence.residual_norm)
    qnm_relative_ode_residual = float(
        qnm.radial_resolution.residual_evidence.relative_residual
    )
    radial_mode = SeparatedMode(
        0,
        0,
        0,
        sector="scalar",
        family="qnm",
        background_id="qualification:schwarzschild",
    )
    radial_plan = SchwarzschildRadialPlan(
        radial_mode,
        1.0,
        node_count=17,
        outer_radius=40.0,
        asymptotic_tolerance=0.2,
    )
    radial = evaluate_schwarzschild_radial(radial_plan, jnp.asarray(0.4 - 0.05j))

    scattering_mode = SeparatedMode(
        0,
        0,
        0,
        sector="scalar",
        family="scattering",
        background_id="qualification:computed-schwarzschild-scattering",
    )
    scattering_radial_plan = SchwarzschildRadialPlan(
        scattering_mode,
        1.0,
        node_count=65,
        inner_radius=2.00002,
        outer_radius=80.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=16,
        infinity_asymptotic_order=3,
    )
    scattering_flux_plan = BlackHoleScatteringPlan(scattering_mode, 0.0, 1.0e-7, 1.0e-10)
    scattering_plan = SchwarzschildScatteringSolvePlan(
        scattering_radial_plan,
        scattering_flux_plan,
        LinearSolvePolicy(DenseLU()),
        refined_integration_substeps=32,
        frequency_step=1.0e-3,
        decomposition_tolerance=1.0e-10,
        refinement_tolerance=1.0e-5,
        slope_refinement_tolerance=5.0e-3,
        absolute_flux_tolerance=1.0e-8,
        incident_amplitude_tolerance=1.0e-12,
        low_frequency_maximum=5.0e-2,
        low_frequency_relative_tolerance=2.5e-1,
    )
    scattering_frequencies = (0.015, 0.020, 0.025)
    scattering_results = tuple(
        solve_schwarzschild_scattering(scattering_plan, jnp.asarray(frequency))
        for frequency in scattering_frequencies
    )
    central_scattering = scattering_results[1]
    greybody_factors = jnp.stack(
        tuple(result.greybody_factor for result in scattering_results)
    )
    neighboring_frequency_slope = (
        central_scattering.evidence.neighboring_greybody_factors[2]
        - central_scattering.evidence.neighboring_greybody_factors[1]
    ) / (
        central_scattering.evidence.neighboring_frequencies[2]
        - central_scattering.evidence.neighboring_frequencies[1]
    )
    slope_residual = float(
        jnp.abs(central_scattering.corotation_slope - neighboring_frequency_slope)
    )
    flux_residual = max(
        float(jnp.abs(result.flux_residual)) for result in scattering_results
    )
    wronskian_residual = max(
        float(jnp.abs(result.wronskian_residual)) for result in scattering_results
    )
    decomposition_residual = max(
        float(result.evidence.decomposition_residual) for result in scattering_results
    )
    refinement_residual = max(
        float(result.evidence.greybody_refinement_error) for result in scattering_results
    )
    slope_refinement_residual = max(
        float(result.evidence.dimensionless_slope_refinement_error)
        for result in scattering_results
    )
    low_frequency_relative_error = max(
        float(result.evidence.low_frequency_relative_error)
        for result in scattering_results
    )

    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1.0, 1.0, 1.0, 1.0)
    species = QuantumFieldSpecies("qualification:massless-scalar", 0.0, "boson")
    hawking_plan = HawkingSpectrumPlan(
        scale,
        (species,),
        scattering_frequencies,
        (species.species_id,),
        (0,),
        (0,),
        mode_ids=("scalar:l0:m0",),
    )
    scattering_finite = jnp.asarray(
        (jnp.all(jnp.stack(tuple(result.finite for result in scattering_results))),)
    )
    scattering_converged = jnp.asarray(
        (jnp.all(jnp.stack(tuple(result.converged for result in scattering_results))),)
    )
    scattering_physical = jnp.asarray(
        (
            jnp.all(
                jnp.stack(tuple(result.physically_valid for result in scattering_results))
            ),
        )
    )
    scattering_qualified = jnp.asarray(
        (jnp.all(jnp.stack(tuple(result.qualified for result in scattering_results))),)
    )
    scattering_derivative = jnp.asarray(
        (
            jnp.all(
                jnp.stack(tuple(result.derivative_valid for result in scattering_results))
            ),
        )
    )
    tail = HawkingTailEvidence(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        qualified=False,
        derivative_valid=False,
        qualification_id="qualification:no-external-tail-evidence",
    )
    scattering_data = HawkingScatteringData(
        hawking_plan,
        greybody_factors[None, :],
        jnp.asarray((central_scattering.corotation_slope,)),
        finite=scattering_finite,
        converged=scattering_converged,
        physically_valid=scattering_physical,
        qualified=scattering_qualified,
        derivative_valid=scattering_derivative,
        tail_evidence=tail,
        source_ids=(
            tuple(
                f"{result.source_id}:omega={frequency:.3f}"
                for result, frequency in zip(
                    scattering_results, scattering_frequencies, strict=True
                )
            ),
        ),
        qualification_id=scattering_plan.plan_id,
    )
    source_state = KerrEvaporationState(
        1.0,
        0.0,
        state_id="qualification:schwarzschild-evaporation-source",
    )
    horizon = evaluate_stationary_kerr_horizon(KerrInput(1.0, 0.0))
    thermal = evaluate_kerr_entropy_temperature(horizon, scale)
    spectrum = evaluate_hawking_spectrum(
        hawking_plan,
        source_state,
        scattering_data,
        thermal.temperature,
        horizon.angular_velocity,
        horizon_source_id=horizon.parameters.input_id,
        horizon_finite=horizon.finite,
        horizon_converged=horizon.converged,
        horizon_physically_valid=horizon.physically_valid,
        horizon_qualified=horizon.qualified,
        horizon_derivative_valid=horizon.derivative_valid,
    )

    return _profile_result(
        "perturbations-scattering-hawking",
        checks={
            "qnm_successful": qnm.successful,
            "qnm_qualified": qnm.qualified,
            "qnm_derivative_valid": qnm.derivative_valid,
            "qnm_status_success": qnm.status == int(QnmStatus.SUCCESS),
            "qnm_derivative_status_valid": qnm.derivative_status
            == int(QnmDerivativeStatus.VALID),
            "qnm_maintained_reference_bound": (
                qnm.reference_id == qnm_reference.reference_id
                and qnm.qualification_source_id == qnm_reference.source_id
            ),
            "qnm_coupled_root_residual": qnm.residual_norm < 1.0e-9,
            "qnm_radial_resolution_qualified": qnm.radial_resolution.qualified,
            "qnm_radial_asymptotics_qualified": qnm.radial_resolution.asymptotic.qualified,
            "qnm_radial_matching_strict": qnm_matching_residual
            < qnm_radial.matching_tolerance,
            "qnm_radial_ode_residual_strict": qnm_relative_ode_residual
            < qnm_radial.residual_tolerance,
            "qnm_radial_source_exact": qnm.radial_source_id == qnm_radial.plan_id,
            "radial_finite": radial.finite,
            "radial_physically_valid": radial.physically_valid,
            "radial_asymptotics_qualified": radial.asymptotic.qualified,
            "scattering_numerically_successful": [
                result.successful for result in scattering_results
            ],
            "scattering_all_qualified": [
                result.qualified for result in scattering_results
            ],
            "scattering_all_derivative_valid": [
                result.derivative_valid for result in scattering_results
            ],
            "scattering_statuses_success": [
                result.status == int(SchwarzschildScatteringStatus.SUCCESS)
                for result in scattering_results
            ],
            "scattering_asymptotics_qualified": [
                result.asymptotic.qualified for result in scattering_results
            ],
            "scattering_sources_converged": [
                result.evidence.source_converged for result in scattering_results
            ],
            "scattering_neighbors_resolved": [
                result.evidence.neighboring_resolved for result in scattering_results
            ],
            "scattering_neighbor_ledgers_valid": [
                result.evidence.neighboring_ledger_valid for result in scattering_results
            ],
            "scattering_slope_refinement_valid": [
                result.evidence.slope_refinement_valid for result in scattering_results
            ],
            "scattering_decomposition_resolved": decomposition_residual
            < scattering_plan.decomposition_tolerance,
            "scattering_refinement_resolved": refinement_residual
            < scattering_plan.refinement_tolerance,
            "scattering_flux_closed": flux_residual
            < scattering_plan.absolute_flux_tolerance,
            "scattering_wronskian_closed": wronskian_residual
            < scattering_plan.absolute_flux_tolerance,
            "scattering_low_frequency_control": low_frequency_relative_error
            < scattering_plan.low_frequency_relative_tolerance,
            "scattering_frequency_slope_from_neighbors": slope_residual < 1.0e-12,
            "scattering_dimensionless_slope_refinement": slope_refinement_residual
            < scattering_plan.slope_refinement_tolerance,
            "hawking_uses_computed_scattering": spectrum.scattering_id
            == scattering_data.scattering_id,
            "hawking_finite": spectrum.finite,
            "hawking_converged": spectrum.converged,
            "hawking_physically_valid": spectrum.physically_valid,
            "hawking_rejects_missing_external_qualification": ~spectrum.qualified,
            "hawking_positive_number_flux": spectrum.number_flux > 0.0,
            "hawking_source_state_exact": spectrum.bound_to(source_state),
        },
        flags={
            "qnm_finite": qnm.finite,
            "qnm_converged": qnm.converged,
            "qnm_physically_valid": qnm.physically_valid,
            "qnm_qualified": qnm.qualified,
            "qnm_derivative_valid": qnm.derivative_valid,
            "qnm_angular_depth_resolved": qnm.angular_depth_evidence.resolved,
            "qnm_radial_depth_resolved": qnm.radial_depth_evidence.resolved,
            "qnm_radial_resolution_qualified": qnm.radial_resolution.qualified,
            "radial_finite": radial.finite,
            "radial_physically_valid": radial.physically_valid,
            "radial_asymptotics_qualified": radial.asymptotic.qualified,
            "scattering_all_finite": [result.finite for result in scattering_results],
            "scattering_all_converged": [
                result.converged for result in scattering_results
            ],
            "scattering_all_physically_valid": [
                result.physically_valid for result in scattering_results
            ],
            "scattering_all_qualified": [
                result.qualified for result in scattering_results
            ],
            "scattering_all_derivative_valid": [
                result.derivative_valid for result in scattering_results
            ],
            "scattering_asymptotics_qualified": [
                result.asymptotic.qualified for result in scattering_results
            ],
            "scattering_neighboring_resolved": [
                result.evidence.neighboring_resolved for result in scattering_results
            ],
            "hawking_scattering_modes_qualified": spectrum.mode_qualified,
            "hawking_finite": spectrum.finite,
            "hawking_converged": spectrum.converged,
            "hawking_physically_valid": spectrum.physically_valid,
            "hawking_coverage_satisfied": spectrum.coverage_satisfied,
            "hawking_external_qualified": spectrum.qualified,
            "hawking_derivative_valid": spectrum.derivative_valid,
        },
        residuals={
            "qnm_coupled_residual_norm": float(qnm.residual_norm),
            "qnm_reference_frequency_abs": float(qnm.reference_frequency_error),
            "qnm_reference_separation_abs": float(qnm.reference_separation_error),
            "qnm_radial_matching_abs": qnm_matching_residual,
            "qnm_radial_ode_residual_norm": qnm_ode_residual,
            "qnm_radial_relative_ode_residual": qnm_relative_ode_residual,
            "qnm_root_condition": float(qnm.root_condition),
            "radial_maximum_residual": float(jnp.abs(radial.residual)),
            "scattering_flux_max_abs": flux_residual,
            "scattering_wronskian_max_abs": wronskian_residual,
            "scattering_decomposition_max_abs": decomposition_residual,
            "scattering_greybody_refinement_max_abs": refinement_residual,
            "scattering_low_frequency_relative_max": low_frequency_relative_error,
            "scattering_neighbor_slope_abs": slope_residual,
            "scattering_dimensionless_slope_refinement_max_abs": (
                slope_refinement_residual
            ),
            "scattering_greybody_minimum": float(jnp.min(greybody_factors)),
            "scattering_greybody_maximum": float(jnp.max(greybody_factors)),
            "hawking_number_flux": float(spectrum.number_flux),
            "hawking_energy_flux": float(spectrum.energy_flux),
            "hawking_tail_number_upper": float(spectrum.tail_remainder_upper[0]),
        },
        statuses={
            "qnm": qnm.status,
            "qnm_derivative": qnm.derivative_status,
            "scattering_frequency_0": scattering_results[0].status,
            "scattering_frequency_1": scattering_results[1].status,
            "scattering_frequency_2": scattering_results[2].status,
        },
        identities={
            "qnm_mode_id": qnm_mode.mode_id,
            "qnm_plan_id": qnm.plan_id,
            "qnm_reference_id": qnm_reference.reference_id,
            "qnm_reference_source_id": qnm_reference.source_id,
            "qnm_radial_plan_id": qnm_radial.plan_id,
            "qnm_angular_plan_id": qnm_angular.plan_id,
            "qnm_branch_id": qnm.branch_id,
            "radial_mode_id": radial_mode.mode_id,
            "radial_plan_id": radial.plan_id,
            "scattering_mode_id": scattering_mode.mode_id,
            "scattering_plan_id": scattering_plan.plan_id,
            "scattering_radial_plan_id": scattering_radial_plan.plan_id,
            "scattering_refined_radial_plan_id": scattering_plan.refined_radial_plan.plan_id,
            "scattering_flux_plan_id": scattering_flux_plan.plan_id,
            "scattering_decomposition_plan_id": central_scattering.evidence.decomposition_plan_id,
            "scattering_source_id": central_scattering.source_id,
            "hawking_scattering_source_ids": list(scattering_data.source_ids[0]),
            "hawking_plan_id": hawking_plan.plan_id,
            "hawking_scattering_id": scattering_data.scattering_id,
            "hawking_tail_evidence_id": tail.evidence_id,
            "hawking_spectrum_id": spectrum.spectrum_id,
            "hawking_source_state_id": source_state.state_id,
            "horizon_source_id": spectrum.horizon_source_id,
        },
    )


def qualify_grhd_grmhd() -> dict[str, object]:
    from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
    from phydrax.applications.compact_objects._accretion import MichelBondiAccretionPlan
    from phydrax.discretization import (
        StructuredCochainBridge,
        TensorGridPlan,
        UniformCellAxisSpec,
    )
    from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
    from phydrax.equations._relativistic_eos import GammaLawEOS
    from phydrax.equations._relativistic_hydrodynamics import ValenciaGRHDSystem
    from phydrax.equations._relativistic_mhd import IdealValenciaGRMHDSystem
    from phydrax.metrix._spacetime_conventions import RelativityConvention
    from phydrax.solver._grmhd_ct import GRMHDConstrainedTransportPlan
    from phydrax.solver._grmhd_runtime import GRMHDRunStatus, GRMHDSSPRK3Plan
    from phydrax.solver._relativistic_finite_volume import (
        FixedGridGRHDSSPRK3Plan,
        lower_valencia_stage_geometry,
    )
    from phydrax.solver._relativistic_primitive import GRHDC2PPolicy
    from phydrax.units import KILOGRAM, METER, SECOND

    grhd_scale = RelativityScaleContract.geometric(KILOGRAM)
    grhd_eos = GammaLawEOS(grhd_scale, 5.0 / 3.0)
    grhd_system = ValenciaGRHDSystem(grhd_eos)
    grhd_grid = TensorGridPlan(
        (UniformCellAxisSpec(6, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    grhd_discretization = FiniteVolumePlan(
        grhd_grid, component_names=grhd_system.component_names
    ).prepare()
    grhd_geometry = _flat_adm_geometry(
        grhd_discretization.cell_shape,
        scale_id=grhd_scale.scale_id,
        convention_id=grhd_system.convention.convention_id,
        topology_id=grhd_grid.topology.topology_id,
        geometry_lineage_id="qualification:minkowski-periodic-line",
    )
    grhd_runtime = FixedGridGRHDSSPRK3Plan(
        grhd_system, GRHDC2PPolicy(grhd_system), grhd_discretization
    )
    grhd_step_size = 1.0e-3
    grhd_stages = (
        lower_valencia_stage_geometry(grhd_discretization, grhd_geometry, 0.0),
        lower_valencia_stage_geometry(grhd_discretization, grhd_geometry, grhd_step_size),
        lower_valencia_stage_geometry(
            grhd_discretization, grhd_geometry, 0.5 * grhd_step_size
        ),
    )
    grhd_primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.2, 0.15, 0.0, 0.0), dtype=jnp.float64),
        grhd_geometry.leading_shape + (5,),
    )
    grhd_conserved = grhd_system.primitive_to_conserved(grhd_primitive, grhd_geometry)
    grhd_initial = grhd_runtime.initialize(grhd_conserved, grhd_stages[0])
    grhd_step = grhd_runtime.advance(grhd_initial, 0.0, grhd_step_size, grhd_stages)
    grhd_state_residual = _maximum_absolute(grhd_step.accepted.conserved - grhd_conserved)
    grhd_ledger_residual = _maximum_absolute(grhd_step.ledger.content_defect)

    grmhd_scale = RelativityScaleContract(
        DimensionalScaleContract(METER, KILOGRAM, SECOND), 1.0, 1.0, 1.0, 1.0
    )
    convention = RelativityConvention.canonical()
    grmhd_eos = GammaLawEOS(grmhd_scale, 4.0 / 3.0, minimum_density=1.0e-15)
    grmhd_system = IdealValenciaGRMHDSystem(
        grmhd_eos,
        grmhd_scale,
        convention=convention,
        pressure_ceiling=1.0e5,
        recovery_iterations=28,
        enthalpy_iterations=28,
    )
    grmhd_grid = TensorGridPlan(
        (
            UniformCellAxisSpec(2, periodic=True),
            UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    bridge = StructuredCochainBridge(grmhd_grid)
    grmhd_geometry = _flat_adm_geometry(
        grmhd_grid.shape,
        scale_id=grmhd_scale.scale_id,
        convention_id=convention.convention_id,
        topology_id=grmhd_grid.topology.topology_id,
        geometry_lineage_id="qualification:minkowski-periodic-plane",
    )
    constrained_transport = GRMHDConstrainedTransportPlan(bridge)
    grmhd_runtime = GRMHDSSPRK3Plan(grmhd_system, constrained_transport, cfl=0.2)
    grmhd_primitive = jnp.zeros(grmhd_grid.shape + (8,), dtype=jnp.float64)
    grmhd_primitive = grmhd_primitive.at[..., 0].set(1.0)
    grmhd_primitive = grmhd_primitive.at[..., 1].set(0.05)
    grmhd_primitive = grmhd_primitive.at[..., 4].set(0.1)
    grmhd_primitive = grmhd_primitive.at[..., 5].set(0.2)
    grmhd_primitive = grmhd_primitive.at[..., 6].set(-0.1)
    grmhd_conserved = grmhd_system.primitive_to_conserved(grmhd_primitive, grmhd_geometry)
    recovered = grmhd_system.recover(grmhd_conserved, grmhd_geometry)
    recovery_residual = _maximum_absolute(recovered.primitive - grmhd_primitive)
    magnetic_flux = constrained_transport.pack_densitized_face_flux(
        tuple(grmhd_primitive[..., 5 + axis] for axis in range(2))
    )
    grmhd_initial = grmhd_runtime.initialize(
        grmhd_conserved,
        grmhd_geometry,
        magnetic_flux=magnetic_flux,
        step_size=1.0e-4,
    )
    grmhd_step = grmhd_runtime.advance(grmhd_initial, 0.0, 1.0e-4, grmhd_geometry)
    material_balance = _maximum_absolute(
        grmhd_step.attempted_ledger.material_balance_defect
    )
    faraday_balance = _maximum_absolute(
        grmhd_step.attempted_ledger.faraday_balance_defect
    )

    accretion_eos = GammaLawEOS(grmhd_scale, 4.0 / 3.0, minimum_density=0.0)
    accretion_plan = MichelBondiAccretionPlan(
        accretion_eos,
        1.0,
        1.0,
        0.01,
        root_iterations=48,
        bracket_samples=64,
    )
    accretion = accretion_plan.evaluate(jnp.asarray((accretion_plan.critical_radius,)))
    accretion_continuity = _maximum_absolute(accretion.continuity_residual)
    accretion_bernoulli = _maximum_absolute(accretion.bernoulli_residual)

    return _profile_result(
        "grhd-grmhd",
        checks={
            "grhd_step_successful": grhd_step.successful,
            "grhd_step_qualified": grhd_step.qualified,
            "grhd_uniform_state_preserved": grhd_state_residual < 1.0e-12,
            "grhd_conservation_closed": grhd_ledger_residual < 1.0e-12,
            "grmhd_recovery_qualified": recovered.qualified,
            "grmhd_recovery_round_trip": recovery_residual < 5.0e-5,
            "grmhd_step_accepted": grmhd_step.accepted,
            "grmhd_status_success": grmhd_step.status == int(GRMHDRunStatus.SUCCESS),
            "grmhd_material_balance_closed": material_balance < 5.0e-6,
            "grmhd_faraday_balance_closed": faraday_balance < 2.0e-7,
            "accretion_solution_qualified": accretion.qualified,
            "accretion_integrals_closed": (
                accretion_continuity < 3.0e-5 * float(accretion_plan.mass_accretion_rate)
            )
            & (accretion_bernoulli < 3.0e-5),
        },
        flags={
            "grhd_finite": grhd_step.finite,
            "grhd_converged": grhd_step.converged,
            "grhd_physically_valid": grhd_step.physically_valid,
            "grhd_qualified": grhd_step.qualified,
            "grmhd_recovery_finite": recovered.finite,
            "grmhd_recovery_converged": recovered.converged,
            "grmhd_recovery_physically_valid": recovered.physically_valid,
            "grmhd_recovery_qualified": recovered.qualified,
            "grmhd_step_accepted": grmhd_step.accepted,
            "grmhd_ledger_qualified": grmhd_step.attempted_ledger.qualified,
            "accretion_finite": accretion.finite,
            "accretion_converged": accretion.converged,
            "accretion_physically_valid": accretion.physically_valid,
            "accretion_qualified": accretion.qualified,
        },
        residuals={
            "grhd_uniform_state_max_abs": grhd_state_residual,
            "grhd_content_ledger_max_abs": grhd_ledger_residual,
            "grmhd_primitive_recovery_max_abs": recovery_residual,
            "grmhd_material_balance_max_abs": material_balance,
            "grmhd_faraday_balance_max_abs": faraday_balance,
            "accretion_continuity_max_abs": accretion_continuity,
            "accretion_bernoulli_max_abs": accretion_bernoulli,
        },
        statuses={
            "grhd": grhd_step.status,
            "grmhd": grmhd_step.status,
            "grmhd_recovery": recovered.status.reshape((-1,))[0],
            "accretion": accretion.status[0],
        },
        identities={
            "grhd_system_id": grhd_system.system_id,
            "grhd_runtime_id": grhd_runtime.runtime_id,
            "grhd_geometry_lineage_id": grhd_geometry.geometry_lineage_id,
            "grmhd_system_id": grmhd_system.system_id,
            "grmhd_ct_plan_id": constrained_transport.plan_id,
            "grmhd_runtime_id": grmhd_runtime.plan_id,
            "grmhd_geometry_lineage_id": grmhd_geometry.geometry_lineage_id,
            "accretion_plan_id": accretion_plan.plan_id,
        },
    )


def qualify_z4c_initial_data_horizons_waves() -> dict[str, object]:
    from phydrax._physical import RelativityScaleContract
    from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
    from phydrax.applications.numerical_relativity._derivatives import (
        FourthOrderDerivatives,
    )
    from phydrax.applications.numerical_relativity._enforcement import (
        Z4cAlgebraicEnforcement,
    )
    from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
    from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
    from phydrax.applications.numerical_relativity._horizon_tracking import (
        quasilocal_horizon_geometry,
    )
    from phydrax.applications.numerical_relativity._initial_data import (
        adm_constraint_diagnostics,
        IsotropicSchwarzschildInitialData,
    )
    from phydrax.applications.numerical_relativity._mots import (
        null_expansions,
        schwarzschild_isotropic_outgoing_expansion,
    )
    from phydrax.applications.numerical_relativity._state import flat_z4c_state
    from phydrax.applications.numerical_relativity._surfaces import (
        schwarzschild_isotropic_spatial_metric,
        SphericalSurfacePlan,
    )
    from phydrax.applications.numerical_relativity._temporal import FixedGridZ4cRuntime
    from phydrax.applications.numerical_relativity._wave_extraction import (
        Psi4ExtractionPlan,
        Psi4TetradConvention,
        SpinWeightedMultipolePlan,
        vacuum_weyl_curvature,
    )
    from phydrax.applications.numerical_relativity._z4c import Z4cSystem
    from phydrax.metrix._spacetime_conventions import RelativityConvention
    from phydrax.units import KILOGRAM

    grid = FixedGridGeometry(
        (5, 5, 5), (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), periodic=True
    )
    system = Z4cSystem(
        RelativityScaleContract.geometric(KILOGRAM),
        RelativityConvention.canonical(),
        chart_id="qualification:cartesian",
        constraint_tolerance=1.0e-6,
    )
    derivatives = FourthOrderDerivatives(grid.shape, grid.spacing)
    runtime = FixedGridZ4cRuntime(
        system,
        grid,
        derivatives,
        GeodesicGauge(),
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(),
        time_step=0.05,
        integrator="ssprk33",
    )
    initial = runtime.initialize(flat_z4c_state(grid.shape, grid_id=grid.grid_id))
    proposal = runtime.evaluate(initial)
    accepted = runtime.accept(proposal)
    z4c_state_residual = _maximum_absolute(accepted.state.values - initial.state.values)
    z4c_constraint = float(proposal.constraints.maximum_norm)

    initial_data_plan = IsotropicSchwarzschildInitialData(1.0)
    initial_point = jnp.asarray((4.0, 0.0, 0.0))
    initial_data = initial_data_plan(initial_point)
    initial_constraints = adm_constraint_diagnostics(
        initial_data_plan, initial_point, tolerance=2.0e-5
    )
    initial_hamiltonian = _maximum_absolute(initial_constraints.hamiltonian)
    initial_momentum = _maximum_absolute(initial_constraints.momentum)

    surface_plan = SphericalSurfacePlan(3)
    surface = surface_plan.constant(0.5)
    coordinate_geometry = surface_plan.geometry(surface)
    spatial_metric = schwarzschild_isotropic_spatial_metric(
        coordinate_geometry.points, 1.0
    )
    horizon_geometry = surface_plan.geometry(surface, spatial_metric)
    zero_curvature = jnp.zeros(surface_plan.sample_shape + (3, 3))
    expansion = schwarzschild_isotropic_outgoing_expansion(horizon_geometry.radius, 1.0)
    null = null_expansions(
        spatial_metric,
        zero_curvature,
        horizon_geometry.outward_normal,
        expansion,
    )
    horizon = quasilocal_horizon_geometry(
        horizon_geometry, zero_curvature, horizon_geometry.phi_tangent
    )
    horizon_area_residual = float(jnp.abs(horizon_geometry.area - 16.0 * jnp.pi))
    horizon_expansion_residual = _maximum_absolute(null.outgoing)

    multipole_plan = SpinWeightedMultipolePlan(3)
    sample_shape = multipole_plan.transform.sample_shape
    metric = jnp.broadcast_to(jnp.eye(3), sample_shape + (3, 3))
    ricci = jnp.broadcast_to(
        jnp.diag(jnp.asarray((0.0, 1.0, -1.0))), sample_shape + (3, 3)
    )
    curvature = vacuum_weyl_curvature(
        metric,
        ricci,
        jnp.zeros(sample_shape + (3, 3)),
        jnp.zeros(sample_shape + (3, 3, 3)),
    )
    psi4_plan = Psi4ExtractionPlan(multipole_plan)
    canonical = psi4_plan.extract(
        curvature,
        surface_plan.unit_radial,
        surface_plan.unit_theta,
        surface_plan.unit_phi,
    )
    reversed_plan = Psi4ExtractionPlan(multipole_plan, Psi4TetradConvention(psi4_sign=-1))
    reversed_sign = reversed_plan.extract(
        curvature,
        surface_plan.unit_radial,
        surface_plan.unit_theta,
        surface_plan.unit_phi,
    )
    psi4_sign_residual = _maximum_absolute(reversed_sign.psi4 + canonical.psi4)

    return _profile_result(
        "z4c-initial-data-horizons-waves",
        checks={
            "z4c_flat_step_successful": proposal.successful,
            "z4c_flat_state_preserved": z4c_state_residual < 2.0e-6,
            "z4c_constraints_closed": z4c_constraint < 2.0e-6,
            "initial_data_physically_valid": initial_data.status.physically_valid,
            "initial_constraints_converged": initial_constraints.converged,
            "initial_hamiltonian_closed": initial_hamiltonian < 2.0e-5,
            "initial_momentum_closed": initial_momentum < 2.0e-5,
            "horizon_geometry_valid": horizon_geometry.physically_valid,
            "horizon_area_identity": horizon_area_residual < 1.0e-10,
            "horizon_expansion_closed": horizon_expansion_residual < 1.0e-11,
            "quasilocal_horizon_qualified": horizon.qualified,
            "psi4_qualified": canonical.qualified,
            "psi4_sign_identity": psi4_sign_residual < 1.0e-11,
        },
        flags={
            "z4c_finite": proposal.finite,
            "z4c_physically_valid": proposal.physically_valid,
            "z4c_qualified": proposal.qualified,
            "initial_data_finite": initial_data.status.finite,
            "initial_data_physically_valid": initial_data.status.physically_valid,
            "initial_constraints_converged": initial_constraints.converged,
            "horizon_geometry_physically_valid": horizon_geometry.physically_valid,
            "null_expansion_qualified": null.qualified,
            "quasilocal_horizon_qualified": horizon.qualified,
            "psi4_physically_valid": canonical.physically_valid,
            "psi4_multipoles_converged": canonical.multipoles.converged,
            "psi4_qualified": canonical.qualified,
        },
        residuals={
            "z4c_state_max_abs": z4c_state_residual,
            "z4c_constraint_maximum": z4c_constraint,
            "initial_hamiltonian_max_abs": initial_hamiltonian,
            "initial_momentum_max_abs": initial_momentum,
            "horizon_area_abs": horizon_area_residual,
            "horizon_outgoing_expansion_max_abs": horizon_expansion_residual,
            "psi4_sign_max_abs": psi4_sign_residual,
        },
        statuses={
            "z4c": proposal.status,
            "initial_data": initial_data.status.status,
        },
        identities={
            "z4c_system_id": system.system_id,
            "z4c_grid_id": grid.grid_id,
            "z4c_runtime_id": runtime.runtime_id,
            "initial_data_id": initial_data.data_id,
            "surface_plan_id": surface_plan.plan_id,
            "psi4_plan_id": psi4_plan.plan_id,
            "psi4_convention_id": canonical.convention_id,
        },
    )


def qualify_coupling() -> dict[str, object]:
    from phydrax.applications.numerical_relativity._coupled_runtime import (
        Z4cMatterCoupledRuntime,
    )
    from phydrax.applications.numerical_relativity._matter_coupling import (
        ConservationLedger,
        ConstraintLedger,
        CoupledParticipantStatus,
        CoupledStageAddress,
        FloorLedger,
        HorizonFluxLedger,
        MatterCouplingPolicy,
        MatterStageProposal,
        SourceExchangeLedger,
        Z4cStageProposal,
    )
    from phydrax.metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection

    topology_id = "qualification:coupled-two-lane"
    weights = jnp.asarray((1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0))

    def geometry_at_stage(z4c, address, args):
        del args
        dtype = jnp.asarray(z4c).dtype
        identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
        return ADMGridGeometry(
            jnp.ones((2,), dtype=dtype),
            jnp.zeros((2, 3), dtype=dtype),
            identity,
            identity,
            jnp.ones((2,), dtype=dtype),
            jnp.asarray(z4c, dtype=dtype) * identity,
            jnp.ones((2,), dtype=bool),
            jnp.ones((2,), dtype=bool),
            chart_id="qualification:cartesian",
            convention_id="mostly-plus",
            scale_id="qualification:geometric",
            topology_id=topology_id,
            geometry_lineage_id="qualification:coupled-adm-snapshot",
            snapshot_token=address.step_id * 3 + address.stage_id + 1,
        )

    def projection(matter, geometry, address, args):
        del address, args
        dtype = jnp.asarray(matter).dtype
        return StressEnergyProjection(
            jnp.full((2,), matter, dtype=dtype),
            jnp.zeros((2, 3), dtype=dtype),
            jnp.zeros((2, 3, 3), dtype=dtype),
            geometry.active,
            jnp.ones((2,), dtype=bool),
            jnp.zeros((2,), dtype=dtype),
            jnp.zeros((2,), dtype=dtype),
            geometry_lineage_id=geometry.geometry_lineage_id,
            snapshot_token=geometry.snapshot_token,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=topology_id,
            projection_id="qualification:coupled-projection",
        )

    def participant_status():
        return CoupledParticipantStatus(0, True, True, True, True, True)

    def stage_address(address):
        return CoupledStageAddress(
            address.step_start_time,
            address.stage_time,
            address.step_end_time,
            address.step_id,
            address.stage_id,
            topology_id=address.topology_id,
        )

    def candidate(base, current, rate, address):
        euler = current + address.step_size * rate
        return jnp.where(
            address.stage_id == 0,
            euler,
            jnp.where(
                address.stage_id == 1,
                0.75 * base + 0.25 * euler,
                (1.0 / 3.0) * base + (2.0 / 3.0) * euler,
            ),
        )

    def propose_z4c(base, current, geometry, stress_energy, address, args):
        del args
        result = candidate(
            base, current, 2.0 + jnp.mean(stress_energy.energy_density), address
        )
        weight = weights[address.stage_id]
        source = SourceExchangeLedger(
            weight * address.step_size * jnp.sum(stress_energy.energy_density),
            jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
            0.0,
            jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
        )
        return Z4cStageProposal(
            result,
            geometry,
            stage_address(address),
            source,
            ConstraintLedger(0.0, jnp.zeros((3,)), 0.0),
            participant_status(),
        )

    def propose_matter(base, current, geometry, stress_energy, address, args):
        del args
        curvature = jnp.mean(jnp.abs(geometry.extrinsic_curvature[..., 0, 0]))
        result = candidate(base, current, (3.0 + curvature) * current, address)
        weight = weights[address.stage_id]
        horizon = weight * address.step_size * jnp.sum(stress_energy.energy_density)
        return MatterStageProposal(
            result,
            stress_energy,
            stage_address(address),
            ConservationLedger(0.0, 0.0, jnp.zeros((3,))),
            FloorLedger(0.0, 0.0, jnp.zeros((3,)), 0),
            HorizonFluxLedger(horizon, horizon, jnp.zeros((3,)), jnp.zeros((3,))),
            participant_status(),
            matter_kind="grhd",
        )

    policy = MatterCouplingPolicy(
        source_consistency_tolerance=1.0e-8,
        constraint_tolerance=1.0e-8,
        conservation_tolerance=1.0e-8,
        maximum_floor_rest_mass=0.0,
        maximum_floor_energy=0.0,
        maximum_consecutive_failures=2,
    )
    runtime = Z4cMatterCoupledRuntime(
        geometry_at_stage,
        projection,
        propose_z4c,
        propose_matter,
        policy,
        topology_id=topology_id,
        matter_kind="grhd",
        z4c_runtime_id="qualification:z4c-analytic-ssprk33",
        matter_runtime_id="qualification:grhd-analytic-ssprk33",
    )
    initial = runtime.initialize(jnp.asarray(0.4), jnp.asarray(1.0))
    result = runtime.advance(initial, jnp.asarray(0.1), None)
    expected_z4c = initial.z4c
    expected_matter = initial.matter
    for stage_id in range(3):
        z4c_euler = expected_z4c + 0.1 * (2.0 + expected_matter)
        matter_euler = (
            expected_matter + 0.1 * (3.0 + jnp.abs(expected_z4c)) * expected_matter
        )
        if stage_id == 0:
            expected_z4c, expected_matter = z4c_euler, matter_euler
        elif stage_id == 1:
            expected_z4c = 0.75 * initial.z4c + 0.25 * z4c_euler
            expected_matter = 0.75 * initial.matter + 0.25 * matter_euler
        else:
            expected_z4c = (1.0 / 3.0) * initial.z4c + (2.0 / 3.0) * z4c_euler
            expected_matter = (1.0 / 3.0) * initial.matter + (2.0 / 3.0) * matter_euler
    z4c_update_residual = float(jnp.abs(result.accepted.z4c - expected_z4c))
    matter_update_residual = float(jnp.abs(result.accepted.matter - expected_matter))
    stage_time_residual = _maximum_absolute(
        jnp.stack(tuple(value.stage_time for value in result.addresses))
        - jnp.asarray((0.0, 0.1, 0.05))
    )
    stage_id_residual = _maximum_absolute(
        jnp.stack(tuple(value.stage_id for value in result.addresses))
        - jnp.arange(3, dtype=jnp.int32)
    )

    return _profile_result(
        "coupling",
        checks={
            "coupled_step_successful": result.successful,
            "coupled_step_accepted": result.accepted_step,
            "coupled_step_qualified": result.qualified,
            "stage_times_exact": stage_time_residual == 0.0,
            "stage_ids_exact": stage_id_residual == 0.0,
            "source_ledger_closed": result.ledgers.source.maximum_defect <= 1.0e-8,
            "constraint_ledger_closed": result.ledgers.constraint.maximum <= 1.0e-8,
            "conservation_ledger_closed": result.ledgers.conservation.maximum <= 1.0e-8,
            "atomic_commit_advanced_both_fields": (result.accepted.z4c != initial.z4c)
            & (result.accepted.matter != initial.matter),
            "z4c_matches_analytic_ssprk33": z4c_update_residual < 1.0e-12,
            "matter_matches_analytic_ssprk33": matter_update_residual < 1.0e-12,
        },
        flags={
            "finite": result.finite,
            "converged": result.converged,
            "physically_valid": result.physically_valid,
            "qualified": result.qualified,
            "derivative_valid": result.derivative_valid,
            "successful": result.successful,
            "accepted_step": result.accepted_step,
            "all_stages_successful": result.stage_successful,
        },
        residuals={
            "stage_time_max_abs": stage_time_residual,
            "stage_id_max_abs": stage_id_residual,
            "source_maximum_defect": float(result.ledgers.source.maximum_defect),
            "constraint_maximum": float(result.ledgers.constraint.maximum),
            "conservation_maximum": float(result.ledgers.conservation.maximum),
            "floor_rest_mass": float(result.ledgers.floor.rest_mass_added),
            "z4c_analytic_update_abs": z4c_update_residual,
            "matter_analytic_update_abs": matter_update_residual,
        },
        statuses={"coupled_step": result.status},
        identities={
            "operator_id": runtime.operator_id,
            "result_operator_id": result.operator_id,
            "policy_id": policy.policy_id,
            "topology_id": topology_id,
            "z4c_runtime_id": runtime.z4c_runtime_id,
            "matter_runtime_id": runtime.matter_runtime_id,
        },
    )


def qualify_scaling_restart() -> dict[str, object]:
    from phydrax.applications.numerical_relativity._checkpoint import (
        evaluate_numerical_relativity_restart,
        NumericalRelativityCheckpointPlan,
        NumericalRelativityRestartPolicy,
        NumericalRelativityRestartState,
        read_numerical_relativity_checkpoint,
        write_numerical_relativity_checkpoint,
    )
    from phydrax.applications.numerical_relativity._distributed import (
        NumericalRelativityDistributedPlan,
    )
    from phydrax.applications.numerical_relativity._state import flat_z4c_state
    from phydrax.applications.numerical_relativity._temporal import Z4cRuntimeState

    devices = jax.devices()[:1]
    distributed = NumericalRelativityDistributedPlan(
        "z4c",
        (4, 4, 4),
        (1, 1, 1),
        halo_width=1,
        periodic=(True, True, True),
        grid_id="qualification:grid-4",
    ).prepare(devices)
    z4c_array = distributed.shard_z4c(jnp.zeros((25, 4, 4, 4)))
    material_array = distributed.shard_material(jnp.zeros((4, 4, 4, 5)))
    z4c_halo = distributed.periodic_z4c_halo(z4c_array, 0)
    material_halo = distributed.periodic_material_halo(material_array, 1)

    z4c = flat_z4c_state((2, 2, 2), grid_id="qualification:restart-grid")
    runtime_state = Z4cRuntimeState(
        z4c,
        jnp.asarray(0.5),
        jnp.asarray(4, dtype=jnp.int32),
        "qualification:restart-runtime",
    )
    state = NumericalRelativityRestartState.from_z4c(
        runtime_state,
        topology_id="qualification:restart-topology",
        topology_epoch=2,
    )
    restart_policy = NumericalRelativityRestartPolicy("exact")
    checkpoint_plan = NumericalRelativityCheckpointPlan(
        "z4c",
        "qualification:restart-runtime",
        "qualification:restart-grid",
        "qualification:restart-topology",
        analysis_plan_id="qualification:restart-analysis",
        numeric_revision_id="qualification:numeric-revision",
        execution_plan_id="qualification:execution",
        topology_epoch=2,
        state_template=state,
        restart=restart_policy,
    )
    with tempfile.TemporaryDirectory(prefix="phydrax-black-hole-restart-") as directory:
        path = Path(directory) / "qualification.phxcheckpoint"
        written = write_numerical_relativity_checkpoint(path, checkpoint_plan, state)
        restored = read_numerical_relativity_checkpoint(path, checkpoint_plan, state)
        restart = evaluate_numerical_relativity_restart(
            state, restored.state, restart_policy
        )
        checkpoint_byte_count = path.stat().st_size

    return _profile_result(
        "scaling-restart",
        checks={
            "single_device_authority": distributed.single_device_authority,
            "z4c_named_sharding_exact": z4c_array.sharding == distributed.z4c_sharding,
            "material_named_sharding_exact": material_array.sharding
            == distributed.material_sharding,
            "z4c_halo_shape_exact": z4c_halo.shape == (25, 6, 4, 4),
            "material_halo_shape_exact": material_halo.shape == (4, 6, 4, 5),
            "checkpoint_content_identity": written.content_id == restored.content_id,
            "restart_exact": restart.exact,
            "restart_within_tolerance": restart.within_tolerance,
            "restart_qualified": restart.qualified,
            "restart_admitted": restart.restart_admitted,
        },
        flags={
            "single_device_authority": distributed.single_device_authority,
            "restart_exact": restart.exact,
            "restart_topology_matches": restart.topology_matches,
            "restart_finite": restart.finite,
            "restart_within_tolerance": restart.within_tolerance,
            "restart_qualified": restart.qualified,
            "restart_derivative_valid": restart.derivative_valid,
            "restart_admitted": restart.restart_admitted,
        },
        residuals={
            "restart_maximum_absolute_error": float(restart.maximum_absolute_error),
            "restart_maximum_relative_error": float(restart.maximum_relative_error),
            "checkpoint_bytes": float(checkpoint_byte_count),
        },
        statuses={"device_count": len(devices)},
        identities={
            "distributed_plan_id": distributed.plan.plan_id,
            "distributed_prepared_id": distributed.prepared_id,
            "checkpoint_plan_id": checkpoint_plan.plan_id,
            "checkpoint_id": checkpoint_plan.checkpoint_id,
            "checkpoint_content_id": written.content_id,
            "restart_policy_id": restart_policy.policy_id,
            "backend": jax.default_backend(),
        },
    )


def qualify_production_interchange() -> dict[str, object]:
    from phydrax._execution_plan import (
        ExecutionCandidate,
        ExecutionRequirements,
        ProviderBinding,
        resolve_execution_plan,
    )
    from phydrax._execution_resources import (
        DeviceResource,
        ExecutionGroupSpec,
        ExecutionPolicy,
        ExecutionResourceEvidence,
        ResourceInventory,
        ResourceRequest,
    )
    from phydrax._physical import RelativityScaleContract
    from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
    from phydrax.applications.numerical_relativity._derivatives import (
        FourthOrderDerivatives,
    )
    from phydrax.applications.numerical_relativity._enforcement import (
        Z4cAlgebraicEnforcement,
    )
    from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
    from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
    from phydrax.applications.numerical_relativity._production import (
        compile_numerical_relativity_production,
        FixedGridZ4cProductionMethod,
        NumericalRelativityDomainBinding,
        NumericalRelativityProductionLimits,
        NumericalRelativitySupportBinding,
    )
    from phydrax.applications.numerical_relativity._state import flat_z4c_state
    from phydrax.applications.numerical_relativity._temporal import FixedGridZ4cRuntime
    from phydrax.applications.numerical_relativity._z4c import Z4cSystem
    from phydrax.interchange._black_hole import (
        BlackHoleArtifactRights,
        BlackHoleArtifactUsePolicy,
        map_field_artifact,
    )
    from phydrax.interchange._resource import ResourceLimits
    from phydrax.lifecycle._resolved_run import ResolvedRunSpec
    from phydrax.metrix._spacetime_conventions import RelativityConvention
    from phydrax.qualification._registry import SupportTuple
    from phydrax.solver._fixed_step import RobustRetryPolicy
    from phydrax.solver._production_runtime import (
        CheckpointGenerationPolicy,
        DurableCheckpointStore,
        ProductionCaseManifest,
        ProductionRunPlan,
    )
    from phydrax.units import KILOGRAM

    with tempfile.TemporaryDirectory(
        prefix="phydrax-black-hole-production-"
    ) as directory:
        root = Path(directory)
        grid = FixedGridGeometry(
            (5, 5, 5),
            (0.0, 0.0, 0.0),
            (0.1, 0.1, 0.1),
            periodic=True,
        )
        payload = b"qualification-bounded-opaque-field-bytes"
        artifact_path = root / "opaque-field-bytes.bin"
        artifact_path.write_bytes(payload)
        rights = BlackHoleArtifactRights(
            "field",
            "qualification:generated-z4c-field",
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            "CC0-1.0",
            "qualification:generated-locally",
            "qualification:no-attribution-required",
            producer="qualification:phydrax",
            producer_version=importlib.metadata.version("phydrax"),
            model_id="qualification:opaque-field-byte-admission",
            coverage="checksum-rights-resource-and-production-binding-only",
            commercial_use=True,
            training_use=False,
            redistribution=True,
            derivative_use=True,
            model_execution=False,
            export=True,
        )
        use_policy = BlackHoleArtifactUsePolicy(
            "numerical-relativity technical qualification",
            ("CC0-1.0",),
            commercial_use=True,
        )
        artifact = map_field_artifact(
            artifact_path.name,
            trusted_root=root,
            limits=ResourceLimits(1024, 4, 32, 16, 2),
            rights=rights,
            use_policy=use_policy,
            source_format="opaque-binary",
            chart_id="qualification:cartesian",
            coordinate_frame_id="qualification:simulation-frame",
            quantity_id="qualification:opaque-inert-field-bytes",
            topology_id=grid.grid_id,
            unit_id="qualification:uninterpreted",
        )

        convention = RelativityConvention.canonical()
        system = Z4cSystem(
            RelativityScaleContract.geometric(KILOGRAM),
            convention,
            chart_id="qualification:cartesian",
        )
        gauge = GeodesicGauge()
        runtime = FixedGridZ4cRuntime(
            system,
            grid,
            FourthOrderDerivatives(grid.shape, grid.spacing),
            gauge,
            PeriodicBoundary(),
            Z4cAlgebraicEnforcement(),
            time_step=0.01,
            integrator="ssprk33",
        )
        runtime_initial = runtime.initialize(
            flat_z4c_state(grid.shape, grid_id=grid.grid_id, dtype=jnp.float32)
        )
        runtime_proposal = runtime.evaluate(runtime_initial)
        method = FixedGridZ4cProductionMethod(runtime)
        initial = method.initialize(runtime_initial)
        step = method.step(0, 0.0, initial, 0.01, None)

        domain = NumericalRelativityDomainBinding(
            formulation_id=system.system_id,
            chart_id=system.chart_id,
            gauge_id=gauge.gauge_id,
            eos_id="eos:vacuum",
            topology_id=grid.grid_id,
            precision_id="precision:float32",
            input_artifacts=(artifact,),
        )
        support_tuple = SupportTuple(
            "numerical-relativity.production",
            {
                "formulation_id": domain.formulation_id,
                "chart_id": domain.chart_id,
                "gauge_id": domain.gauge_id,
                "eos_id": domain.eos_id,
                "topology_id": domain.topology_id,
                "precision_id": domain.precision_id,
            },
        )
        support = NumericalRelativitySupportBinding(
            "scientific", "qualification:fixed-grid-z4c", support_tuple
        )
        resources = ResourceRequest(
            1,
            1_000_000,
            maximum_checkpoint_staging_bytes=500_000,
            maximum_output_backlog_bytes=500_000,
        )
        execution_policy = ExecutionPolicy(resources=resources)
        limits = NumericalRelativityProductionLimits(
            execution_policy,
            maximum_input_artifacts=1,
            maximum_input_bytes=1024,
            maximum_output_manifests=1,
            maximum_output_artifacts=1,
            maximum_output_bytes=1024,
            maximum_cancellation_detail_bytes=64,
        )
        checkpoint_policy = CheckpointGenerationPolicy(1)
        run_plan = ProductionRunPlan(
            method,
            RobustRetryPolicy(maximum_retries=0),
            step_size=0.01,
            end_time=0.01,
            maximum_steps=1,
            checkpoint_interval=1,
            segment_steps=1,
        )
        case = ProductionCaseManifest(
            problem_id="qualification:flat-z4c",
            method_id=method.method_id,
            precision_id=domain.precision_id,
            topology_id=domain.topology_id,
            geometry_layout_id=domain.chart_id,
            dtype="float32",
        )
        requirements = ExecutionRequirements(
            "qualification:numerical-relativity",
            dtypes=("float32",),
        )
        group = ExecutionGroupSpec(
            "qualification:local-device-group",
            (0,),
            ((0, 0),),
            mesh_axes=(("device", 1),),
        )
        resource_evidence = ExecutionResourceEvidence(
            per_device_peak_bytes=2048,
            per_device_reserve_bytes=1024,
            per_host_peak_bytes=2048,
            per_host_reserve_bytes=1024,
            checkpoint_staging_bytes=resources.maximum_checkpoint_staging_bytes,
            output_backlog_bytes=resources.maximum_output_backlog_bytes,
            dtypes=("float32",),
            backends=(jax.default_backend(),),
        )
        candidate = ExecutionCandidate(
            "qualification:local-device-candidate",
            requirements.requirements_id,
            group,
            providers=(
                ProviderBinding(
                    "backend",
                    jax.default_backend(),
                    "qualification:numerical-relativity",
                ),
            ),
            estimated_memory_bytes=4096,
            resource_evidence=resource_evidence,
        )
        device = jax.devices()[0]
        inventory = ResourceInventory(
            1,
            0,
            (
                DeviceResource(
                    0,
                    0,
                    0,
                    jax.default_backend(),
                    device.device_kind,
                    memory_bytes=resources.memory_bytes,
                ),
            ),
        )
        execution = resolve_execution_plan(
            execution_policy,
            inventory,
            (candidate,),
            precision_policy_id=domain.precision_id,
            solver_policy_id=run_plan.plan_id,
        )
        resolved = ResolvedRunSpec(
            (support.dependency,),
            (),
            release_index_id="qualification:release-index",
            profile_ids=(support.profile_id,),
            trust_policy_id="qualification:local-trust",
            valid_at=10,
            valid_from=0,
            valid_until=20,
            prepared_configuration_id=domain.binding_id,
            precision_policy_id=domain.precision_id,
            resource_policy_id=limits.resource_policy_id,
            checkpoint_policy_id=checkpoint_policy.policy_id,
            output_policy_id=limits.output_policy_id,
            repository_id="qualification:local-repository",
            scheduler_id="qualification:inline-scheduler",
            auth_policy_id="qualification:local-auth",
        )
        production = compile_numerical_relativity_production(
            domain,
            (support,),
            execution,
            resolved,
            case,
            run_plan,
            checkpoint_policy,
            limits,
        )
        store = DurableCheckpointStore(root / "checkpoints", case, checkpoint_policy)
        prepared = production.prepare(store)
        artifact_size = artifact.rights.size_bytes

    return _profile_result(
        "production-interchange",
        checks={
            "artifact_mapping_valid": artifact.report.valid,
            "artifact_checksum_exact": artifact.rights.content_sha256
            == hashlib.sha256(payload).hexdigest(),
            "artifact_size_exact": artifact_size == len(payload),
            "artifact_source_format_honest": artifact.schema.source_format
            == "opaque-binary",
            "runtime_step_successful": step.successful,
            "runtime_step_advanced": step.accepted_state.runtime_state.step_index == 1,
            "runtime_scientific_status_qualified": runtime_proposal.qualified,
            "production_domain_bound": production.resolved_run_spec.prepared_configuration_id
            == production.domain.binding_id,
            "production_execution_bound": production.execution_plan.solver_policy_id
            == production.run_plan.plan_id,
            "execution_policy_bound": execution.policy_id == execution_policy.policy_id,
            "execution_requirements_bound": execution.requirements_id
            == requirements.requirements_id,
            "execution_inventory_bound": execution.inventory_id == inventory.inventory_id,
            "execution_group_bound": execution.group == group,
            "execution_resource_evidence_bound": execution.resource_evidence.evidence_id
            == resource_evidence.evidence_id,
            "prepared_resolved_run_exact": prepared.resolved_run_spec.spec_id
            == production.resolved_run_spec.spec_id,
        },
        flags={
            "artifact_report_valid": artifact.report.valid,
            "artifact_rights_commercial_use": artifact.rights.commercial_use,
            "artifact_model_execution_permitted": artifact.rights.model_execution,
            "runtime_step_finite": runtime_proposal.finite,
            "runtime_step_converged": runtime_proposal.converged,
            "runtime_step_physically_valid": runtime_proposal.physically_valid,
            "runtime_step_qualified": runtime_proposal.qualified,
            "runtime_step_derivative_valid": runtime_proposal.derivative_valid,
        },
        residuals={
            "artifact_bytes": float(artifact_size),
            "runtime_residual": float(step.residual),
        },
        statuses={
            "runtime_work": step.work,
            "runtime_z4c": runtime_proposal.status,
            "input_artifact_count": len(production.domain.input_artifacts),
        },
        identities={
            "artifact_id": artifact.artifact_id,
            "artifact_schema_id": artifact.schema.schema_id,
            "artifact_rights_id": artifact.rights.rights_id,
            "domain_binding_id": domain.binding_id,
            "support_binding_id": support.binding_id,
            "execution_plan_id": execution.execution_plan_id,
            "execution_policy_id": execution_policy.policy_id,
            "execution_requirements_id": requirements.requirements_id,
            "execution_inventory_id": inventory.inventory_id,
            "execution_group_id": group.group_id,
            "execution_resource_evidence_id": resource_evidence.evidence_id,
            "resolved_run_spec_id": resolved.spec_id,
            "run_plan_id": run_plan.plan_id,
            "production_id": production.production_id,
            "prepared_run_id": prepared.run_id,
        },
    )


def qualify_waveforms_remnants() -> dict[str, object]:
    import phydrax as phx

    gw = phx.applications.astrophysics.gravitational_waves
    compact = phx.applications.compact_objects
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "qualification:nr-polynomial-surrogate"
    )
    reconstruction = jnp.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=jnp.float64)
    coefficients = jnp.asarray([[1.0, 0.1], [2.0, 0.2]], dtype=jnp.float64)
    orders = jnp.asarray(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
        ],
        dtype=jnp.int32,
    )
    active = jnp.ones((2, 2), dtype=bool)
    real_field = gw.PolynomialEmpiricalField(
        reconstruction,
        coefficients,
        orders,
        active,
        field_id="qualification:real-l2m2",
    )
    imaginary_field = gw.PolynomialEmpiricalField(
        reconstruction,
        jnp.zeros((2, 1), dtype=jnp.float64),
        jnp.zeros((2, 1, 3), dtype=jnp.int32),
        jnp.ones((2, 1), dtype=bool),
        field_id="qualification:imaginary-l2m2",
    )
    artifact = gw.AlignedNRSurrogateArtifact(
        jnp.asarray([-1.0, 0.0, 1.0]),
        ((2, 2),),
        (real_field,),
        (imaginary_field,),
        jnp.asarray([0.0, -1.0, -1.0]),
        jnp.asarray([3.0, 1.0, 1.0]),
        provenance,
        maximum_mass_ratio=8.0,
        maximum_spin_magnitude=0.8,
        frame_id="qualification:inertial",
        time_origin_id="qualification:analytic-grid-origin",
        mode_normalization="s2fft-orthonormal-condon-shortley",
        artifact_id="qualification:analytic-polynomial-mode",
    )
    precision = phx.discretization.spectral.SpectralPrecisionPolicy(
        jnp.complex128,
        output_dtype=jnp.complex128,
    )
    angular = phx.discretization.spectral.SphericalSpectralPlan(
        3,
        spin=-2,
        reality=False,
        precision=precision,
    ).prepare()
    surrogate = gw.AlignedNRSurrogatePlan(
        artifact,
        angular,
        phx.RelativityScaleContract.si(),
    )
    parameters = {
        "mass_ratio": jnp.asarray(2.0),
        "primary_spin": jnp.asarray(0.2),
        "secondary_spin": jnp.asarray(-0.1),
    }
    coordinates = gw.aligned_spin_surrogate_coordinates(2.0, 0.2, -0.1)
    nodes = jnp.asarray([1.0 + 0.1 * coordinates[0], 2.0 + 0.2 * coordinates[1]])
    expected_mode = jnp.asarray([nodes[0], 0.5 * (nodes[0] + nodes[1]), nodes[1]])
    modes = surrogate.evaluate_modes(parameters)
    center = angular.layout.bandlimit - 1
    mode_error = jnp.max(jnp.abs(modes.coefficients[2, center + 2] - expected_mode))
    symmetry_error = jnp.max(
        jnp.abs(
            modes.coefficients[2, center - 2]
            - jnp.conj(modes.coefficients[2, center + 2])
        )
    )
    polarizations = surrogate.evaluate_geometric(
        jnp.asarray([-0.5, 0.0, 0.5]),
        parameters,
        inclination=jnp.asarray(0.6),
    )

    sample_count = 64
    sample_interval = 1.0 / sample_count
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    psd = gw.OneSidedPowerSpectralDensity(
        frequency,
        jnp.ones_like(frequency),
        provenance,
        sample_count=sample_count,
        sample_interval=sample_interval,
        psd_id="qualification:unit-psd",
    )
    match_plan = gw.WaveformMatchPlan(psd)
    first = jnp.exp(-(((frequency - 10.0) / 4.0) ** 2)).astype(jnp.complex128)
    expected_lag = 5.0 * sample_interval
    second = first * jnp.exp(-2.0j * jnp.pi * frequency * expected_lag + 0.7j)
    overlap = match_plan.overlap(
        first,
        second,
        time_shift_seconds=expected_lag,
    )
    match = match_plan.match(first, second)

    remnant_plan = compact.AlignedBinaryRemnantPlan()
    remnant = remnant_plan.evaluate(30.0, 30.0, 0.0, 0.0)
    remnant_spin_reference = 0.6863698793893069
    remnant_energy_reference = 0.04841605714434129
    remnant_spin_error = jnp.abs(
        remnant.final_dimensionless_spin - remnant_spin_reference
    )
    remnant_energy_error = jnp.abs(
        remnant.radiated_energy_fraction - remnant_energy_reference
    )
    remnant_derivative = jax.grad(
        lambda spin: remnant_plan.evaluate(36.0, 24.0, spin, 0.2).final_dimensionless_spin
    )(jnp.asarray(0.5))

    return _profile_result(
        "waveforms-remnants",
        checks={
            "surrogate_modes_valid": modes.valid,
            "surrogate_caller_source_unqualified": ~modes.qualified,
            "surrogate_source_not_authenticated": ~jnp.asarray(
                artifact.source_authenticated
            ),
            "surrogate_polynomial_reconstruction": mode_error < 2.0e-14,
            "surrogate_nonprecessing_symmetry": symmetry_error < 2.0e-14,
            "surrogate_polarizations_supported": jnp.all(polarizations.valid),
            "surrogate_polarizations_finite": jnp.all(jnp.isfinite(polarizations.values)),
            "fixed_time_overlap_unity": jnp.abs(overlap.overlap - 1.0) < 2.0e-14,
            "time_phase_match_unity": jnp.abs(match.match - 1.0) < 2.0e-14,
            "time_phase_match_lag_exact": jnp.abs(match.lag_seconds - expected_lag)
            < 2.0e-14,
            "remnant_successful": remnant.successful,
            "remnant_reference_spin": remnant_spin_error < 2.0e-14,
            "remnant_reference_energy": remnant_energy_error < 2.0e-14,
            "remnant_kerr_bound": jnp.abs(remnant.final_dimensionless_spin) < 1.0,
            "remnant_interior_derivative_finite": jnp.isfinite(remnant_derivative),
        },
        flags={
            "surrogate_mode_finite": jnp.all(jnp.isfinite(modes.coefficients)),
            "surrogate_mode_valid": modes.valid,
            "surrogate_polarization_valid": jnp.all(polarizations.valid),
            "surrogate_mode_qualified": modes.qualified,
            "surrogate_source_authenticated": artifact.source_authenticated,
            "overlap_valid": overlap.valid,
            "match_valid": match.valid,
            "remnant_finite": remnant.finite,
            "remnant_physically_valid": remnant.physically_valid,
        },
        residuals={
            "surrogate_mode_max_abs": mode_error,
            "surrogate_symmetry_max_abs": symmetry_error,
            "fixed_time_overlap_abs": jnp.abs(overlap.overlap - 1.0),
            "time_phase_match_abs": jnp.abs(match.match - 1.0),
            "time_phase_lag_abs_seconds": jnp.abs(match.lag_seconds - expected_lag),
            "remnant_spin_reference_abs": remnant_spin_error,
            "remnant_energy_reference_abs": remnant_energy_error,
            "remnant_spin_derivative": remnant_derivative,
        },
        statuses={
            "surrogate": modes.status,
            "overlap": overlap.status,
            "match": match.status,
            "remnant": remnant.status,
        },
        identities={
            "surrogate_artifact_id": artifact.artifact_id,
            "surrogate_plan_id": surrogate.plan_id,
            "surrogate_trust_id": artifact.trust_id,
            "angular_prepared_id": angular.prepared_id,
            "match_plan_id": match_plan.plan_id,
            "remnant_plan_id": remnant_plan.plan_id,
            "remnant_source_id": remnant.source_id,
        },
    )


def qualify_advanced() -> dict[str, object]:
    from phydrax._physical import RelativityScaleContract
    from phydrax.applications.compact_objects._advanced_perturbations import (
        ExcitationResiduePlan,
        MassiveFieldQuasiBoundPlan,
    )
    from phydrax.applications.compact_objects._extended_thermodynamics import (
        KerrNewmanAdSThermodynamicsPlan,
    )
    from phydrax.applications.compact_objects._inverse import (
        FixedBranchModelEvaluation,
        kerr_geometry_inverse_adapter,
    )
    from phydrax.applications.compact_objects._plasma_closures import (
        TwoTemperatureElectronIonClosure,
        TwoTemperaturePlasmaState,
    )
    from phydrax.applications.compact_objects._self_force import (
        FirstOrderSelfForceModeSum,
        ModeSumRegularizationParameters,
    )
    from phydrax.applications.numerical_relativity._bms import (
        BMSQuadraturePlan,
        BMSScriData,
    )
    from phydrax.applications.numerical_relativity._characteristic import (
        CharacteristicEvolutionPlan,
        CharacteristicWorldtubeHistory,
    )
    from phydrax.applications.numerical_relativity._event_horizon import (
        CompletedSpacetimeHistory,
        EventHorizonStatus,
        OfflineEventHorizonTracingPlan,
    )
    from phydrax.applications.numerical_relativity._surfaces import (
        SphericalSpectralSurface,
    )
    from phydrax.equations._force_free import GRForceFreeSystem
    from phydrax.equations._relativistic_radiation import GRGreyM1RadiationSystem
    from phydrax.equations._relativistic_radiation_interaction import (
        ConstantGRGreyOpacityPlan,
        GRGreyRadiationInteractionPlan,
    )
    from phydrax.equations._resistive_grmhd import ResistiveGRMHDOhmicClosure
    from phydrax.metrix._adm_exchange import ADMGridGeometry
    from phydrax.metrix._spacetime_conventions import RelativityConvention
    from phydrax.units import KILOGRAM

    scale = RelativityScaleContract.si()
    ads_plan = KerrNewmanAdSThermodynamicsPlan(
        2.0, 0.3, 0.2, 10.0, scale, ensemble="grand-canonical"
    )
    ads = ads_plan.evaluate()
    ads_residual = max(
        float(jnp.abs(ads.enthalpy_constraint_residual)),
        float(jnp.abs(ads.temperature_constraint_residual)),
        float(jnp.abs(ads.smarr_residual)),
    )

    bound_plan = MassiveFieldQuasiBoundPlan(
        2.0,
        0.0,
        0.0,
        0.1,
        jnp.geomspace(4.01, 120.0, 32),
        ell=1,
        overtone=0,
        newton_steps=2,
    )
    bound_seed = bound_plan.hydrogenic_seed()
    bound_parameters = bound_plan.mode_parameters(bound_seed)
    seed_expected = 0.1 * np.sqrt(1.0 - (2.0 * 0.1 / 2.0) ** 2)
    seed_residual = float(jnp.abs(jnp.real(bound_seed) - seed_expected))

    poles = jnp.asarray((1.0 - 0.1j, 2.0 - 0.2j))
    numerators = jnp.asarray((2.0 + 1.0j, -1.0 + 0.5j))
    derivatives = jnp.asarray((4.0 + 0.0j, 2.0 - 1.0j))
    residue_plan = ExcitationResiduePlan(poles, numerators, derivatives)
    residues = residue_plan.residues()
    residue_residual = _maximum_absolute(residues.residue - numerators / derivatives)

    regularization = ModeSumRegularizationParameters(
        jnp.asarray((1.0, -0.5)),
        jnp.asarray((2.0, 0.25)),
        jnp.asarray((3.0, -0.75)),
        jnp.asarray((0.1, -0.2)),
        side="plus",
        gauge="Lorenz",
        worldline_id="qualification:circular-r10M",
        component_basis="Schwarzschild-coordinate",
    )
    ell_max = 14
    angular_order = jnp.arange(ell_max + 1, dtype=jnp.float64) + 0.5
    coefficient_2 = jnp.asarray((0.2, -0.05))
    coefficient_4 = jnp.asarray((0.05, 0.025))
    retarded = (
        angular_order[:, None] * regularization.A
        + regularization.B
        + regularization.C / angular_order[:, None]
        + coefficient_2 / angular_order[:, None] ** 2
        + coefficient_4 / angular_order[:, None] ** 4
    )
    self_force_plan = FirstOrderSelfForceModeSum(
        regularization,
        ell_max,
        tail_window=7,
        tail_fit_tolerance=1.0e-9,
        maximum_tail_fraction=1.0,
    )
    self_force = self_force_plan.calculate(retarded)
    expected_self_force = (
        coefficient_2 * (np.pi**2 / 2.0)
        + coefficient_4 * (np.pi**4 / 6.0)
        - regularization.D
    )
    self_force_residual = _maximum_absolute(self_force.self_force - expected_self_force)

    def evaluate_geometry(parameters):
        mass, spin = parameters
        discriminant = mass * mass - spin * spin
        valid = (
            jnp.isfinite(mass) & jnp.isfinite(spin) & (mass > 0.0) & (discriminant > 0.0)
        )
        safe = jnp.maximum(discriminant, jnp.finfo(parameters.dtype).tiny)
        radius = mass + jnp.sqrt(safe)
        area = 4.0 * jnp.pi * (radius * radius + spin * spin)
        return FixedBranchModelEvaluation(
            jnp.stack((radius, area)),
            jnp.asarray((0,), dtype=jnp.int32),
            finite=jnp.isfinite(radius) & jnp.isfinite(area),
            converged=True,
            physically_valid=valid,
            qualified=valid,
            derivative_valid=valid,
            realization_id="qualification:kerr-analytic",
            branch_id="subextremal",
        )

    inverse = kerr_geometry_inverse_adapter(
        evaluate_geometry,
        jnp.asarray((0,), dtype=jnp.int32),
        parameter_count=2,
        output_count=2,
        realization_id="qualification:kerr-analytic",
        branch_id="subextremal",
        adapter_id="qualification:kerr-geometry-sensitivity",
        evaluator_semantic_id="qualification:kerr-geometry-observable:v1",
        evaluator_numeric_id="qualification:kerr-geometry-implementation:r1",
    )
    sensitivity = inverse.sensitivity(
        jnp.asarray((2.0, 0.4)),
        jnp.asarray((0.25, -0.1)),
        epsilon=2.0e-4,
    )

    plasma_scale = RelativityScaleContract.geometric(KILOGRAM)
    plasma_plan = TwoTemperatureElectronIonClosure(
        plasma_scale,
        equilibration_time=2.0,
        minimum_temperature=1.0,
        maximum_temperature=1.0e9,
    )
    plasma = plasma_plan.advance(
        jnp.asarray(2.0e20),
        jnp.asarray(1.0e20),
        TwoTemperaturePlasmaState(jnp.asarray(100.0), jnp.asarray(400.0)),
        jnp.asarray(2.0),
    )

    convention = RelativityConvention.canonical()
    electromagnetic_geometry = _flat_adm_geometry(
        (),
        scale_id=plasma_scale.scale_id,
        convention_id=convention.convention_id,
        topology_id="qualification:single-cell",
        geometry_lineage_id="qualification:flat-single-cell",
        dtype=jnp.float32,
    )
    radiation_system = GRGreyM1RadiationSystem(plasma_scale, convention)
    radiation_interaction = GRGreyRadiationInteractionPlan(
        radiation_system,
        ConstantGRGreyOpacityPlan(
            planck_absorption=2.0,
            scattering=3.0,
        ),
        radiation_constant=1.0,
    )
    radiation = radiation_interaction.matter_exchange(
        jnp.asarray(2.0),
        jnp.asarray((0.25, 0.0, 0.0)),
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.asarray(1.0),
        electromagnetic_geometry,
    )
    resistive = ResistiveGRMHDOhmicClosure(plasma_scale, convention, conductivity=1.0e4)
    magnetic = jnp.asarray((0.0, 0.0, 2.0))
    velocity = jnp.asarray((0.2, 0.0, 0.0))
    ideal_electric = resistive.ideal_electric_field(
        magnetic, velocity, electromagnetic_geometry
    )
    ideal_ohm = resistive.evaluate(
        ideal_electric,
        magnetic,
        velocity,
        jnp.asarray(3.0),
        electromagnetic_geometry,
    )
    force_free = GRForceFreeSystem(plasma_scale, convention, dominance_margin=1.0e-4)
    projected = force_free.project_constraints(
        jnp.asarray((3.0, 0.0, 1.0)),
        magnetic,
        electromagnetic_geometry,
    )

    sample_count = 9
    times = np.linspace(-1.0, 1.0, sample_count)
    inverse_radius = np.asarray((0.25, 0.125, 0.0))
    characteristic_plan = CharacteristicEvolutionPlan(sample_count, 3, 1)
    flat_history = CharacteristicWorldtubeHistory(
        times,
        inverse_radius,
        np.asarray((2,)),
        np.asarray((2,)),
        np.zeros((sample_count, 1), dtype=complex),
        np.zeros((sample_count, 3, 1), dtype=complex),
        history_name="qualification:minkowski-worldtube",
    )
    characteristic = characteristic_plan.evolve(flat_history)

    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    quadrature = BMSQuadraturePlan(
        directions,
        np.full((6,), 4.0 * np.pi / 6.0),
        np.concatenate((np.ones((1, 6)), directions.T), axis=0),
        np.diag((4.0 * np.pi, 4.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)),
        supported_bandlimit=1,
        quadrature_tolerance=2.0e-6,
    )
    scri = BMSScriData(
        (0.0, 1.0),
        np.full((2, quadrature.direction_capacity), 2.0),
        np.zeros((2, quadrature.direction_capacity), dtype=complex),
        np.zeros((2, quadrature.direction_capacity, 4, 4)),
        data_name="qualification:rest-bondi-data",
    )
    charges = quadrature.charges(scri)
    bms_mass_residual = _maximum_absolute(charges.four_momentum[:, 0] - 2.0)
    bms_momentum_residual = _maximum_absolute(charges.four_momentum[:, 1:])

    event_times = np.linspace(0.0, 1.0, 5)
    axes = tuple(np.asarray((-2.0, 0.0, 2.0)) for _ in range(3))
    history_shape = (len(event_times), 3, 3, 3)
    identity = np.broadcast_to(np.eye(3), history_shape + (3, 3)).copy()
    history_geometry = ADMGridGeometry(
        np.ones(history_shape),
        np.zeros(history_shape + (3,)),
        identity,
        identity,
        np.ones(history_shape),
        np.zeros(history_shape + (3, 3)),
        np.ones(history_shape, dtype=bool),
        np.ones(history_shape, dtype=bool),
        chart_id="qualification:minkowski-cartesian",
        convention_id="mostly-plus",
        scale_id="qualification:geometric",
        topology_id="qualification:event-grid",
        geometry_lineage_id="qualification:completed-flat-history",
        snapshot_token=0,
    )
    terminal_surface = SphericalSpectralSurface(
        np.asarray(((1.0 + 0.0j,),)),
        np.zeros(3),
        "qualification:unqualified-terminal-surface-plan",
    )
    terminal_positions = np.asarray(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)))
    terminal_covectors = np.broadcast_to(
        np.asarray((1.0, 0.0, 0.0)), terminal_positions.shape
    )
    completed_history = CompletedSpacetimeHistory(
        event_times,
        *axes,
        history_geometry,
        completed=True,
        completion_id="qualification:completed-flat-history",
    )
    event_plan = OfflineEventHorizonTracingPlan(
        terminal_surface,
        terminal_positions,
        terminal_covectors,
        np.ones((2,), dtype=bool),
        np.asarray(((1,), (0,))),
        time_capacity=len(event_times),
        grid_shape=(3, 3, 3),
        terminal_surface_qualified=False,
        absolute_tolerance=2.0e-7,
        relative_tolerance=2.0e-6,
        null_tolerance=2.0e-6,
        caustic_distance=0.1,
    )
    event_horizon = event_plan.trace(completed_history)
    expected_event_status = int(
        EventHorizonStatus.TERMINAL_SURFACE_UNQUALIFIED
        | EventHorizonStatus.DERIVATIVE_INVALID
    )

    return _profile_result(
        "advanced",
        checks={
            "ads_thermodynamics_qualified": ads.qualified,
            "ads_identities_closed": ads_residual < 2.0e-13,
            "massive_bound_state_physical": bound_parameters.physically_valid,
            "massive_seed_identity": seed_residual < 1.0e-14,
            "excitation_residues_qualified": residues.qualified,
            "excitation_residue_identity": residue_residual < 1.0e-13,
            "self_force_qualified": self_force.qualified,
            "self_force_identity": self_force_residual < 2.0e-10,
            "inverse_sensitivity_valid": sensitivity.derivative_valid,
            "inverse_jvp_finite_difference": sensitivity.jvp_finite_difference_residual
            < 2.0e-3,
            "inverse_vjp_pairing": sensitivity.vjp_pairing_residual < 1.0e-5,
            "plasma_exchange_qualified": plasma.qualified,
            "plasma_energy_closed": jnp.abs(plasma.conservation_residual) < 1.0e-12,
            "radiation_exchange_qualified": radiation.qualified,
            "radiation_energy_closed": jnp.abs(radiation.energy_balance_residual)
            < 1.0e-6,
            "radiation_momentum_closed": jnp.max(
                jnp.abs(radiation.momentum_balance_residual)
            )
            < 1.0e-6,
            "ideal_ohm_limit_qualified": ideal_ohm.qualified,
            "force_free_projection_qualified": projected.qualified,
            "force_free_degeneracy_closed": jnp.abs(
                projected.constraints.degeneracy_residual
            )
            < 1.0e-6,
            "characteristic_flat_control_qualified": characteristic.qualified,
            "characteristic_flat_wave_zero": jnp.max(
                jnp.abs(characteristic.waveform.news_modes)
            )
            < 1.0e-13,
            "bms_charges_qualified": charges.qualified,
            "bms_rest_mass_identity": bms_mass_residual < 2.0e-6,
            "bms_rest_momentum_identity": bms_momentum_residual < 2.0e-6,
            "event_history_complete": event_horizon.global_history_complete,
            "event_coverage_complete": event_horizon.coverage_complete,
            "event_numerically_converged": event_horizon.converged,
            "event_trace_finite": event_horizon.finite,
            "event_physical_claim_refused": ~event_horizon.physically_valid,
            "event_refusal_status_exact": event_horizon.status == expected_event_status,
            "event_trace_rejects_unqualified_terminal": ~event_horizon.qualified,
            "event_derivative_refused": ~event_horizon.derivative_valid,
        },
        flags={
            "ads_finite": ads.finite,
            "ads_physically_valid": ads.physically_valid,
            "ads_qualified": ads.qualified,
            "massive_bound_state": bound_parameters.bound_state,
            "massive_parameters_physically_valid": bound_parameters.physically_valid,
            "massive_superradiant": bound_parameters.superradiant,
            "self_force_finite": self_force.finite,
            "self_force_converged": self_force.converged,
            "self_force_physically_valid": self_force.physically_valid,
            "self_force_qualified": self_force.qualified,
            "self_force_derivative_valid": self_force.derivative_valid,
            "inverse_branch_stable": sensitivity.branch_stable,
            "inverse_derivative_valid": sensitivity.derivative_valid,
            "plasma_finite": plasma.finite,
            "plasma_converged": plasma.converged,
            "plasma_physically_valid": plasma.physically_valid,
            "plasma_qualified": plasma.qualified,
            "radiation_finite": radiation.finite,
            "radiation_converged": radiation.converged,
            "radiation_physically_valid": radiation.physically_valid,
            "radiation_qualified": radiation.qualified,
            "ideal_ohm_limit": ideal_ohm.ideal_limit,
            "force_free_projection_qualified": projected.qualified,
            "characteristic_qualified": characteristic.qualified,
            "bms_qualified": charges.qualified,
            "event_history_complete": event_horizon.global_history_complete,
            "event_coverage_complete": event_horizon.coverage_complete,
            "event_numerically_converged": event_horizon.converged,
            "event_physically_valid": event_horizon.physically_valid,
            "event_terminal_surface_qualified": False,
            "event_qualified": event_horizon.qualified,
        },
        residuals={
            "ads_identity_max_abs": ads_residual,
            "massive_seed_abs": seed_residual,
            "excitation_residue_max_abs": residue_residual,
            "self_force_max_abs": self_force_residual,
            "self_force_tail_fit_max_abs": _maximum_absolute(
                self_force.tail.fit_residual
            ),
            "inverse_jvp_finite_difference_max_abs": float(
                sensitivity.jvp_finite_difference_residual
            ),
            "inverse_vjp_pairing_abs": float(sensitivity.vjp_pairing_residual),
            "plasma_energy_balance_abs": float(jnp.abs(plasma.conservation_residual)),
            "radiation_energy_balance_abs": float(
                jnp.abs(radiation.energy_balance_residual)
            ),
            "radiation_momentum_balance_max_abs": _maximum_absolute(
                radiation.momentum_balance_residual
            ),
            "ideal_ohm_conduction_max_abs": _maximum_absolute(
                ideal_ohm.conduction_current
            ),
            "force_free_degeneracy_abs": float(
                jnp.abs(projected.constraints.degeneracy_residual)
            ),
            "characteristic_news_max_abs": _maximum_absolute(
                characteristic.waveform.news_modes
            ),
            "bms_rest_mass_max_abs": bms_mass_residual,
            "bms_rest_momentum_max_abs": bms_momentum_residual,
            "event_null_maximum": _maximum_absolute(event_horizon.null_residual),
        },
        statuses={
            "self_force": self_force.status,
            "event_horizon": event_horizon.status,
        },
        identities={
            "ads_plan_id": ads.plan_id,
            "massive_field_plan_id": bound_plan.plan_id,
            "excitation_plan_id": residue_plan.plan_id,
            "self_force_calculator_id": self_force.calculator_id,
            "self_force_parameter_id": regularization.parameter_id,
            "inverse_adapter_id": sensitivity.adapter_id,
            "plasma_closure_id": plasma_plan.closure_id,
            "radiation_system_id": radiation_system.system_id,
            "resistive_closure_id": resistive.closure_id,
            "force_free_system_id": force_free.system_id,
            "characteristic_plan_id": characteristic_plan.plan_id,
            "bms_quadrature_id": quadrature.plan_id,
            "event_horizon_plan_id": event_plan.plan_id,
            "event_history_id": completed_history.history_id,
        },
    )


PROFILE_RUNNERS: dict[str, Callable[[], dict[str, object]]] = {
    "geometry": qualify_geometry,
    "thermodynamics": qualify_thermodynamics,
    "rays": qualify_rays,
    "grrt-interferometry": qualify_grrt_interferometry,
    "perturbations-scattering-hawking": qualify_perturbations_scattering_hawking,
    "grhd-grmhd": qualify_grhd_grmhd,
    "z4c-initial-data-horizons-waves": qualify_z4c_initial_data_horizons_waves,
    "coupling": qualify_coupling,
    "scaling-restart": qualify_scaling_restart,
    "production-interchange": qualify_production_interchange,
    "waveforms-remnants": qualify_waveforms_remnants,
    "advanced": qualify_advanced,
}


def _requested_profiles(profiles: Sequence[str] | None, /) -> tuple[str, ...]:
    if profiles is None or len(profiles) == 0:
        return PROFILE_NAMES
    requested = tuple(str(profile).strip() for profile in profiles)
    unknown = tuple(profile for profile in requested if profile not in PROFILE_RUNNERS)
    if unknown:
        raise ValueError(f"Unknown black-hole qualification profiles: {unknown}.")
    if len(set(requested)) != len(requested):
        raise ValueError("Black-hole qualification profiles must be unique.")
    return requested


def run_qualification(
    profiles: Sequence[str] | None = None,
    /,
    *,
    valid_at: int,
) -> dict[str, object]:
    """Run requested profiles independently and bind them to one runtime manifest."""

    requested = _requested_profiles(profiles)
    manifest = _runtime_manifest(valid_at)
    manifest_id = str(manifest["manifest_id"])
    results: dict[str, dict[str, object]] = {}
    for name in requested:
        try:
            result = PROFILE_RUNNERS[name]()
        except Exception as error:
            metadata = _PROFILE_METADATA[name]
            result = {
                "profile": name,
                "passed": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error)[:512],
                },
                "checks": {},
                "flags": {},
                "residuals": {},
                "statuses": {},
                "identities": {},
                "support": metadata["support"],
                "references": metadata["references"],
                "limitations": metadata["limitations"],
            }
        result["manifest_id"] = manifest_id
        results[name] = result
    failures = [name for name, result in results.items() if not result["passed"]]
    report: dict[str, object] = {
        "qualification": "black-hole-closure",
        "requested_profiles": list(requested),
        "passed": not failures,
        "failed_profiles": failures,
        "manifest": manifest,
        "profiles": results,
        "references": [
            "All numeric evidence is produced locally by the named landed PHYDRAX APIs.",
            "Analytic controls are identified per profile; no external artifact "
            "is silently treated as reference evidence.",
        ],
        "limitations": [
            "A passing bounded profile is technical evidence only and does not "
            "authorize PNPL deployment.",
            "No hidden download, pickle loading, benchmark result, or external "
            "validation is used.",
        ],
    }
    report["report_id"] = _content_id("black-hole-qualification-report", report)
    return report


def serialize_report(report: Mapping[str, object], /) -> str:
    """Serialize a report as standards-compliant bounded JSON."""

    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic black-hole closure qualification profiles."
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_NAMES,
        dest="profiles",
        help="Run one profile; repeat for multiple profiles. The default runs all profiles.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--valid-at",
        type=int,
        required=True,
        help="Unix timestamp binding this observation-time-only evidence.",
    )
    arguments = parser.parse_args(argv)
    report = run_qualification(arguments.profiles, valid_at=arguments.valid_at)
    payload = serialize_report(report)
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(arguments.output)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
