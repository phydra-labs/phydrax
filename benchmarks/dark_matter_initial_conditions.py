#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Report correlated-mode covariance and wave-phase reconstruction residuals."""

import hashlib
import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
import phydrax.ein as ein
from phydrax.applications.cosmology._mixed_initial_conditions import (
    component_transfer_payload_bytes,
    ComponentTransferMatrixProduct,
    PrimordialModeRealization,
    WavePhaseSeedPlan,
)
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import GaussianCoefficientRealization


cosmology = phx.applications.cosmology


def _space(count=8):
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(count),
            phx.discretization.FourierBasisPlan(count),
        ),
        axis_names=("x", "y"),
        field_name="mixed-initial-condition",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )


def _transfer(space):
    background = cosmology.FLRWBackground(1.0, 1.0)
    matrix = jnp.asarray(((1.0, 0.35), (-0.2, 0.8)))
    values = jnp.broadcast_to(matrix[:, :, None, None], (2, 2, 2, 2))
    payload = component_transfer_payload_bytes((0.1, 1.0), (0.1, 100.0), values)
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = ReferenceArtifactManifest(
        "component-transfer-matrix-benchmark",
        checksum_algorithm="sha256",
        checksum=checksum,
        size_bytes=len(payload),
        license_id="internal-benchmark",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="internal",
        nondimensionalization={"code": 1.0},
        uncertainty={"benchmark": 0.0},
        lineage_ids=("benchmark-lineage",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="component-transfer-matrix-benchmark",
        content_digest=checksum,
        producer="PHYDRA-benchmark",
        producer_version="current",
        build_id="contract",
        license_id=manifest.license_id,
        resource_id="in-memory",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id, *manifest.lineage_ids),
    )
    provenance = cosmology.CosmologyProductProvenance(
        producer=artifact.producer,
        producer_version=artifact.producer_version,
        model_form_id=background.model_form_id,
        request_id="dark-matter-initial-condition-benchmark",
        numerical_policy_id="fixed-fourier-grid",
        physics_policy_id="correlated-two-component-transfer",
        scale_id=background.scale.scale_id,
        source_kind="external",
        differentiation="constant",
        parent_product_ids=(artifact.artifact_id, manifest.manifest_id),
    )
    transfer = ComponentTransferMatrixProduct(
        (0.1, 1.0),
        (0.1, 100.0),
        values,
        components=("cold_baryon", "wave_dark_matter"),
        primordial_components=("adiabatic", "isocurvature"),
        scale=background.scale,
        provenance=provenance,
        realization=background.realization,
        artifact=artifact,
        manifest=manifest,
        gauge="synchronous",
        spatial_dimension=len(space.axes),
    )
    return background, transfer, matrix


def run(sample_count=256):
    space = _space()
    background, transfer, matrix = _transfer(space)
    mode_ids = PrimordialModeRealization.required_mode_ids(
        space, transfer.primordial_components
    )
    samples = []
    for index in range(int(sample_count)):
        gaussian = GaussianCoefficientRealization.sample(
            jr.fold_in(jr.key(20260915), index),
            mode_ids,
            coupling_id="dark-matter-ic-benchmark",
        )
        primordial = PrimordialModeRealization.from_gaussian_modes(
            space,
            gaussian,
            transfer.primordial_components,
            scale=transfer.scale,
        )
        realized = transfer.realize(primordial, 0.1)
        samples.append(np.asarray(realized.modes[1, 0]))
    samples_array = np.asarray(samples)
    empirical = (
        np.asarray(
            ein.contract(
                "sc,sd->cd",
                jnp.asarray(samples_array),
                jnp.conj(jnp.asarray(samples_array)),
            )
        ).real
        / sample_count
    )
    expected = np.asarray(ein.contract("cp,dp->cd", matrix, matrix))
    empirical_covariance_relative_residual = float(
        np.sqrt(np.sum((empirical - expected) ** 2)) / np.sqrt(np.sum(expected**2))
    )
    exact = np.asarray(transfer.covariance_values[:, :, 0, 0])
    exact_covariance_relative_residual = float(
        np.sqrt(np.sum((exact - expected) ** 2)) / np.sqrt(np.sum(expected**2))
    )

    prepared = WaveDarkMatterPlan(
        1.0,
        (0.1, 0.1001),
        gravitational_constant=0.1,
        reduced_planck_constant=1.0,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
        ),
    ).prepare(space, background)
    phase_plan = WavePhaseSeedPlan(prepared)
    x = space.axes[0].nodes[:, None]
    density = jnp.ones(space.physical_shape) * 2.0
    velocity_x = 0.04 * jnp.pi * jnp.cos(2.0 * jnp.pi * x) / 0.1**2
    current = jnp.stack(
        (
            jnp.broadcast_to(density * velocity_x, space.physical_shape),
            jnp.zeros(space.physical_shape),
        ),
        axis=-1,
    )
    phase = phase_plan.realize(
        density,
        current,
        0.1,
        density_unit=phase_plan.density_unit,
        current_unit=phase_plan.current_unit,
        current_convention=phase_plan.current_convention,
    )
    if not bool(phase.successful):
        raise AssertionError("Compatible benchmark current was not reconstructed.")
    return {
        "sample_count": int(sample_count),
        "transfer_product_id": transfer.product_id,
        "source_artifact_id": transfer.artifact.artifact_id,
        "source_manifest_id": transfer.manifest.manifest_id,
        "requested_use_id": transfer.requested_use_id,
        "exact_covariance_relative_residual": exact_covariance_relative_residual,
        "empirical_covariance_relative_residual": empirical_covariance_relative_residual,
        "current_reconstruction_relative_residual": float(
            phase.evidence.current_relative_residual
        ),
        "curl_relative_residual": float(phase.evidence.curl_relative_residual),
        "phase_gauge_absolute": float(phase.evidence.phase_gauge_absolute),
        "wave_mass_relative_residual": float(phase.evidence.mass_relative_residual),
        "de_broglie_nyquist_fraction": float(phase.evidence.de_broglie_nyquist_fraction),
        "scope": (
            "Fixed-grid correlated linear modes and node-free irrotational wave-phase "
            "reconstruction only; no nonlinear evolution or production claim."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
