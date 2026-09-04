"""Beamlet, nonlinear-wave, transport, and guided-mode benchmark producer."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.artifacts import ArtifactManifest
from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.beamlets import (
    BeamletFrame,
    BeamletReconstructionPlan,
    gaussian_beamlets_at_waist,
    GaussianWaistSpecification,
    reconstruct_gaussian_beamlets,
)
from phydrax.optics.geometric import NonSequentialSurfaceTable, OpticalRayState
from phydrax.optics.materials import (
    AngularFrequencyValidity,
    ConstantRefractiveIndex,
    RefractiveIndexProvenance,
)
from phydrax.optics.transport import (
    prepare_tissue_transport,
    simulate_tissue_transport,
    TissueTransportCoefficients,
    TissueTransportPlan,
)
from phydrax.optics.wave import (
    AnalyticPulseField,
    InstantaneousScalarSusceptibility,
    PlaneFieldSpace,
    propagate_unidirectional,
    UnidirectionalPropagationPlan,
)
from phydrax.solver.maxwell import FixedFrequencyGuidedModePlan


jax.config.update("jax_enable_x64", True)


def _timed(function):
    start = time.perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def _beamlet_case():
    grid = TensorGridPlan(
        (UniformAxisSpec(65), UniformAxisSpec(65)), axis_names=("u", "v")
    ).prepare(jnp.asarray(((-2.0, -2.0), (2.0, 2.0))))
    space = PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window")
    ray = OpticalRayState(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        geometric_path_lengths=0.0,
        optical_path_lengths=0.0,
    )
    frame = BeamletFrame(RigidFrame.identity(3))
    beamlet = gaussian_beamlets_at_waist(
        ray,
        GaussianWaistSpecification((0.7, 0.9), 0.15),
        frame,
        2.0 * jnp.pi,
        3.0,
        topology_id="benchmark-branch",
        source_prepared_id="benchmark-system",
    )
    prepared = BeamletReconstructionPlan(space, 0.0, tile_size=256).prepare()
    return _timed(lambda: reconstruct_gaussian_beamlets(prepared, beamlet))


def _nonlinear_case():
    plane = TensorGridPlan(
        (FourierAxisSpec(4), FourierAxisSpec(4)), axis_names=("u", "v")
    ).prepare(jnp.asarray(((-20.0, -20.0), (20.0, 20.0))))
    temporal = TensorGridPlan((FourierAxisSpec(32),), axis_names=("time",)).prepare(
        jnp.asarray(((0.0,), (2.0 * jnp.pi,)))
    )
    space = PlaneFieldSpace(plane, RigidFrame.identity(3), "periodic-cell")
    mode = 7.0
    values = jnp.exp(-1j * mode * temporal.axes[0].nodes)
    values = jnp.broadcast_to(values, space.shape + temporal.shape)
    field = AnalyticPulseField(space, temporal, values, mode, 0.0, polarization="scalar")
    manifest = ArtifactManifest(
        artifact_id="advanced-optics-benchmark-index",
        producer="phydrax",
        version="current",
        sha256="0" * 64,
        byte_size=0,
        source_uri="generated://advanced-optics-benchmark",
        license_id="LicenseRef-PHYDRA",
        model="constant benchmark index",
        coverage="positive benchmark frequencies",
    )
    law = ConstantRefractiveIndex(
        1.5,
        validity=AngularFrequencyValidity(0.5, 20.0),
        reference_wave_speed=1.0,
        provenance=RefractiveIndexProvenance(manifest, record_id="benchmark-index"),
        law_id="benchmark-index",
    )
    prepared = UnidirectionalPropagationPlan(
        space,
        temporal,
        mode,
        step_count=8,
        maximum_spectral_edge_fraction=1.0,
        maximum_analytic_signal_defect=1.0,
        maximum_hermitian_reconstruction_defect=1.0,
        maximum_nonlinear_rejected_fraction=1.0,
        maximum_refinement_error=1.0,
        maximum_backward_wave_estimate=1.0,
    ).prepare(law)
    return _timed(
        lambda: propagate_unidirectional(
            prepared,
            field,
            InstantaneousScalarSusceptibility(0.0, 0.05),
            0.25,
        )
    )


def _transport_case():
    vertices = jnp.asarray(
        ((-10.0, -10.0, 1.0), (10.0, -10.0, 1.0), (10.0, 10.0, 1.0), (-10.0, 10.0, 1.0))
    )
    surfaces = NonSequentialSurfaceTable(
        vertices,
        jnp.asarray(((0, 1, 2), (0, 2, 3)), dtype=jnp.int32),
        jnp.asarray((0, 0)),
        jnp.asarray((1, 1)),
        jnp.asarray((1.0, 1.0)),
        surface_ids=jnp.asarray((0, 0)),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            surfaces,
            TissueTransportCoefficients(
                jnp.asarray((0.4, 0.0)),
                jnp.asarray((0.6, 0.0)),
                jnp.asarray((0.5, 0.0)),
                jnp.asarray((1.0, 1.0)),
            ),
            maximum_interactions=4,
        )
    )
    count = 2048
    origins = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 0.0)), (count, 3))
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3))
    media = jnp.zeros((count,), dtype=jnp.int32)
    return _timed(
        lambda: simulate_tissue_transport(
            prepared,
            origins,
            directions,
            media,
            jr.PRNGKey(0),
            photon_ids=jnp.arange(count, dtype=jnp.uint32),
        )
    )


def _guided_case():
    propagation = np.linspace(1.0, 4.0, 12)
    identity = np.eye(propagation.size, dtype=np.complex128)
    plan = FixedFrequencyGuidedModePlan(
        -np.diag(propagation**2),
        np.zeros_like(identity),
        identity,
        propagation.size,
        angular_frequency=8.0,
        right_electric_trace_coefficients=(identity,),
        right_magnetic_trace_coefficients=(identity,),
        left_electric_trace_coefficients=(identity,),
        left_magnetic_trace_coefficients=(identity,),
        divergence_coefficients=(np.zeros((1, propagation.size)),),
        power_pairing=identity,
        target_propagation_constant=2.5,
    )
    return _timed(plan.solve)


def benchmark() -> dict[str, object]:
    beamlet, beamlet_seconds = _beamlet_case()
    nonlinear, nonlinear_seconds = _nonlinear_case()
    transport, transport_seconds = _transport_case()
    guided, guided_seconds = _guided_case()
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "timings_seconds": {
            "beamlet_reconstruction": beamlet_seconds,
            "nonlinear_propagation": nonlinear_seconds,
            "tissue_transport_2048": transport_seconds,
            "guided_mode_solve_12": guided_seconds,
        },
        "evidence": {
            "beamlet_status": int(beamlet.evidence.status),
            "nonlinear_status": int(nonlinear.status),
            "transport_maximum_ledger_residual": float(
                transport.maximum_absolute_ledger_residual
            ),
            "guided_mode_status": int(guided.status),
            "guided_mode_maximum_residual": float(jnp.max(guided.polynomial_residuals)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/advanced_optics.json")
    )
    arguments = parser.parse_args()
    payload = benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
