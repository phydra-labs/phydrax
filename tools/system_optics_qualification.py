"""Deterministic beamlet and atmospheric-optics qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.beamlets import (
    BeamletFrame,
    BeamletReconstructionPlan,
    gaussian_beamlets_at_waist,
    GaussianWaistSpecification,
    reconstruct_gaussian_beamlets,
)
from phydrax.optics.geometric import OpticalRayState
from phydrax.optics.wave import (
    PlaneFieldSpace,
    sample_von_karman_phase_screen,
    VonKarmanPhaseScreenPlan,
)


jax.config.update("jax_enable_x64", True)


def _finite_space(count: int, extent: float) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (UniformAxisSpec(count), UniformAxisSpec(count)), axis_names=("u", "v")
    ).prepare(jnp.asarray(((-extent, -extent), (extent, extent))))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window")


def _periodic_space(count: int, extent: float) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (
            UniformAxisSpec(count, endpoint=False, periodic=True),
            UniformAxisSpec(count, endpoint=False, periodic=True),
        ),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-0.5 * extent, -0.5 * extent), (0.5 * extent, 0.5 * extent))))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _beamlet_case() -> dict[str, float | int | bool]:
    space = _finite_space(65, 2.0)
    ray = OpticalRayState(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        geometric_path_lengths=0.0,
        optical_path_lengths=0.0,
    )
    beamlet = gaussian_beamlets_at_waist(
        ray,
        GaussianWaistSpecification((0.7, 0.7)),
        BeamletFrame(RigidFrame.identity(3)),
        2.0 * jnp.pi,
        3.0,
        topology_id="qualification-branch",
        source_prepared_id="qualification-system",
    )
    result = reconstruct_gaussian_beamlets(
        BeamletReconstructionPlan(space, 0.0, tile_size=256).prepare(), beamlet
    )
    coordinates = space.transverse_coordinates
    expected = jnp.exp(-jnp.sum(coordinates * coordinates, axis=-1) / 0.7**2)
    maximum_error = jnp.max(jnp.abs(result.field.values - expected))
    return {
        "status": int(result.evidence.status),
        "successful": bool(result.successful),
        "maximum_field_error": float(maximum_error),
        "maximum_condition_estimate": float(result.evidence.maximum_condition_estimate),
    }


def _atmosphere_case() -> dict[str, float | int | bool]:
    prepared = VonKarmanPhaseScreenPlan(
        _periodic_space(32, 8.0),
        0.2,
        10.0,
        inner_scale=0.02,
    ).prepare()
    result = sample_von_karman_phase_screen(prepared, jax.random.key(17))
    keys = jax.random.split(jax.random.key(1701), 64)
    ensemble_variances = jax.vmap(
        lambda key: (
            sample_von_karman_phase_screen(prepared, key).evidence.realized_variance
        )
    )(keys)
    ensemble_variance = jnp.mean(ensemble_variances)
    variance_relative_error = (
        jnp.abs(ensemble_variance - result.evidence.predicted_variance)
        / result.evidence.predicted_variance
    )
    return {
        "status": int(result.evidence.status),
        "valid": bool(result.valid),
        "hermitian_error": float(result.evidence.hermitian_error),
        "parseval_relative_error": float(result.evidence.parseval_relative_error),
        "piston": float(result.evidence.piston),
        "predicted_variance": float(result.evidence.predicted_variance),
        "realized_variance": float(result.evidence.realized_variance),
        "ensemble_realized_variance": float(ensemble_variance),
        "ensemble_variance_relative_error": float(variance_relative_error),
    }


def qualify() -> dict[str, object]:
    beamlet = _beamlet_case()
    atmosphere = _atmosphere_case()
    accepted = (
        beamlet["successful"]
        and beamlet["maximum_field_error"] < 5.0e-5
        and atmosphere["valid"]
        and atmosphere["hermitian_error"] < 1.0e-5
        and atmosphere["parseval_relative_error"] < 1.0e-5
        and abs(atmosphere["piston"]) < 1.0e-5
        and atmosphere["predicted_variance"] > 0.0
        and atmosphere["ensemble_realized_variance"] > 0.0
        and atmosphere["ensemble_variance_relative_error"] < 0.25
    )
    return {
        "accepted": bool(accepted),
        "cases": {"beamlet": beamlet, "atmosphere": atmosphere},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/system_optics_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
