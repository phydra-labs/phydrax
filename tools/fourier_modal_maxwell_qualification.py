"""Deterministic Fourier-modal Maxwell qualification artifact."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


jax.config.update("jax_enable_x64", True)


def _interface_case() -> dict[str, float | int]:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    dielectric = fm.FrequencyMaxwellMaterial(4.0, material_id="dielectric")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (),
        fm.HomogeneousMaxwellPort(dielectric, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    expected_reflection = 1.0 / 9.0
    expected_transmission = 8.0 / 9.0
    return {
        "left_incoming_power": float(result.left_incoming_power[0]),
        "right_incoming_power": float(result.right_incoming_power[0]),
        "left_outgoing_power": float(result.left_outgoing_power[0]),
        "right_outgoing_power": float(result.right_outgoing_power[0]),
        "left_outgoing_error": float(
            abs(result.left_outgoing_power[0] - expected_reflection)
        ),
        "right_outgoing_error": float(
            abs(result.right_outgoing_power[0] - expected_transmission)
        ),
        "net_port_power_into_stack": float(result.net_port_power_into_stack[0]),
        "status": int(result.status),
    }


def _propagation_case() -> dict[str, float | int]:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    material = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.0, 0.0)),
    )
    boundary = fm.prepare_layer_boundary(
        operator,
        0.125,
        fm.BoundaryCascadePolicy(
            doublings=9,
            initializer_order=6,
            paired_error=True,
            relative_tolerance=1e-8,
        ),
    )
    modal = fm.prepare_modal_boundary(operator, 0.125)
    difference = jnp.sqrt(
        jnp.sum(jnp.abs(boundary.a - modal.boundary.a) ** 2)
        + jnp.sum(jnp.abs(boundary.b - modal.boundary.b) ** 2)
        + jnp.sum(jnp.abs(boundary.c - modal.boundary.c) ** 2)
        + jnp.sum(jnp.abs(boundary.d - modal.boundary.d) ** 2)
    )
    return {
        "boundary_modal_difference": float(difference),
        "paired_error": float(boundary.diagnostics.paired_error),
        "constitutive_residual": float(operator.diagnostics.constitutive_residual),
        "modal_status": int(modal.status),
    }


def _loss_and_convergence_case() -> dict[str, object]:
    evidence = []
    for bandwidth, samples in ((1, 3), (3, 7)):
        harmonics = LatticeHarmonicPlan.parallelogramic((bandwidth,), (samples,)).prepare(
            jnp.asarray(((1.0, 0.0),))
        )
        vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="loss-vacuum", passive=True)
        film = fm.FrequencyMaxwellMaterial(
            2.25 + 0.08j, material_id="lossy-film", passive=True
        )
        problem = fm.FourierModalMaxwellProblem(
            harmonics,
            2.0 * jnp.pi,
            jnp.asarray((0.0, 0.0)),
            fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
            (
                fm.FourierModalLayer(
                    film,
                    0.2,
                    fm.DirectFourierFactorizationPlan(),
                    layer_id="film",
                ),
            ),
            fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
            numeric_version=f"loss-{bandwidth}",
        )
        prepared = fm.prepare_fourier_modal_maxwell(problem)
        excitation = fm.plane_wave_excitation(
            prepared.scattering,
            harmonics.plan.layout.mode_ids[0],
            "te",
        )
        result = fm.solve_fourier_modal_maxwell(prepared, excitation)
        revision = fm.fourier_modal_numeric_revision(prepared)
        evidence.append(
            fm.evaluate_fourier_modal_loss(
                prepared,
                result,
                fm.FourierModalLossPolicy(z_quadrature_order=8),
                numeric_revision=revision,
            )
        )
    convergence = fm.assess_fourier_modal_loss_convergence(tuple(evidence))
    finest = evidence[-1]
    return {
        "net_port_power_into_stack": float(finest.net_port_power_into_stack[0]),
        "volume_material_loss": float(finest.total_volume_material_loss[0]),
        "unresolved_numerical_closure": float(finest.unresolved_numerical_closure[0]),
        "z_quadrature_defect": float(jnp.max(finest.z_quadrature_defect)),
        "loss_status": int(finest.status),
        "convergence_accepted": bool(convergence.accepted),
        "port_power_difference": float(convergence.port_power_differences[-1]),
        "material_loss_difference": float(convergence.material_loss_differences[-1]),
        "closure_difference": float(convergence.closure_differences[-1]),
    }


def _retrieval_case() -> dict[str, object]:
    cases = []
    revisions = []
    frequencies = jnp.asarray((4.0, 5.0, 6.0))
    thickness = 0.2
    harmonic_mode_id = ""
    for index, frequency in enumerate(frequencies):
        harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
            jnp.asarray(((1.0, 0.0),)),
            numeric_version=f"retrieval-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
        vacuum = fm.FrequencyMaxwellMaterial(
            1.0, material_id="retrieval-vacuum", passive=True
        )
        slab = fm.FrequencyMaxwellMaterial(
            2.25, material_id="retrieval-slab", passive=True
        )
        problem = fm.FourierModalMaxwellProblem(
            harmonics,
            frequency,
            jnp.asarray((0.0, 0.0)),
            fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
            (
                fm.FourierModalLayer(
                    slab,
                    thickness,
                    fm.DirectFourierFactorizationPlan(),
                    layer_id="slab",
                ),
            ),
            fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
            numeric_version=f"retrieval-{index}",
        )
        prepared = fm.prepare_fourier_modal_maxwell(problem)
        cases.append(prepared)
        revisions.append(fm.fourier_modal_numeric_revision(prepared))
    sweep = fm.prepare_maxwell_modal_sweep(
        tuple(cases),
        tuple(revisions),
        slab_thickness=thickness,
        harmonic_mode_id=harmonic_mode_id,
        polarization="te",
    )
    retrieval = fm.retrieve_equivalent_slab(
        sweep,
        fm.EquivalentSlabRetrievalPlan(
            (-2, 2),
            anchor="known-index",
            anchor_refractive_index=1.5,
            passive_claim=True,
        ),
    )
    return {
        "status": int(retrieval.status),
        "ambiguous": bool(retrieval.ambiguous),
        "minimum_branch_margin": float(jnp.min(retrieval.branch_margin)),
        "maximum_reconstruction_residual": float(
            jnp.max(retrieval.reconstruction_residual)
        ),
        "maximum_index_error": float(jnp.max(jnp.abs(retrieval.refractive_index - 1.5))),
        "finite_band_kramers_kronig_residual": float(
            retrieval.finite_band_kramers_kronig_residual
        ),
        "stable_fit_residual": float(retrieval.stable_fit_residual),
    }


def _local_sweep(thickness: float, polarization: str, bloch_x: float):
    cases = []
    revisions = []
    harmonic_mode_id = ""
    for index, frequency in enumerate((4.0, 5.0, 6.0)):
        harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
            jnp.asarray(((1.0, 0.0),)),
            numeric_version=f"local-{thickness}-{polarization}-{bloch_x}-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
        vacuum = fm.FrequencyMaxwellMaterial(
            1.0, material_id="retrieval-vacuum", passive=True
        )
        slab = fm.FrequencyMaxwellMaterial(
            2.25, material_id="retrieval-slab", passive=True
        )
        problem = fm.FourierModalMaxwellProblem(
            harmonics,
            frequency,
            jnp.asarray((bloch_x, 0.0)),
            fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
            (
                fm.FourierModalLayer(
                    slab,
                    thickness,
                    fm.DirectFourierFactorizationPlan(),
                    layer_id="slab",
                ),
            ),
            fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
            numeric_version=f"local-{thickness}-{polarization}-{bloch_x}-{index}",
        )
        prepared = fm.prepare_fourier_modal_maxwell(problem)
        cases.append(prepared)
        revisions.append(fm.fourier_modal_numeric_revision(prepared))
    sweep = fm.prepare_maxwell_modal_sweep(
        tuple(cases),
        tuple(revisions),
        slab_thickness=thickness,
        harmonic_mode_id=harmonic_mode_id,
        polarization=polarization,
    )
    if bloch_x != 0.0:
        return sweep
    return fm.retrieve_equivalent_slab(
        sweep,
        fm.EquivalentSlabRetrievalPlan(
            (-2, 2),
            anchor="known-index",
            anchor_refractive_index=1.5,
            passive_claim=True,
        ),
    )


def _local_loss_evidence():
    evidence = []
    for mode_count, sample_count in ((1, 3), (3, 7)):
        harmonics = LatticeHarmonicPlan.parallelogramic(
            (mode_count,), (sample_count,)
        ).prepare(
            jnp.asarray(((1.0, 0.0),)),
            numeric_version=f"local-loss-{mode_count}",
        )
        vacuum = fm.FrequencyMaxwellMaterial(
            1.0, material_id="retrieval-vacuum", passive=True
        )
        slab = fm.FrequencyMaxwellMaterial(
            2.25, material_id="retrieval-slab", passive=True
        )
        problem = fm.FourierModalMaxwellProblem(
            harmonics,
            5.0,
            jnp.zeros((2,)),
            fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
            (
                fm.FourierModalLayer(
                    slab,
                    0.2,
                    fm.DirectFourierFactorizationPlan(),
                    layer_id="slab",
                ),
            ),
            fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
            numeric_version=f"local-loss-{mode_count}",
        )
        prepared = fm.prepare_fourier_modal_maxwell(problem)
        result = fm.solve_fourier_modal_maxwell(
            prepared,
            fm.plane_wave_excitation(
                prepared.scattering,
                harmonics.plan.layout.mode_ids[0],
                "te",
            ),
        )
        evidence.append(
            fm.evaluate_fourier_modal_loss(
                prepared,
                result,
                fm.FourierModalLossPolicy(
                    relative_tolerance=1.0e-5,
                    absolute_tolerance=1.0e-8,
                ),
                numeric_revision=fm.fourier_modal_numeric_revision(prepared),
            )
        )
    convergence = fm.assess_fourier_modal_loss_convergence(
        tuple(evidence), relative_tolerance=1.0e-5
    )
    return tuple(evidence), convergence


def _local_isotropic_case() -> dict[str, object]:
    retrievals = tuple(
        _local_sweep(thickness, polarization, 0.0)
        for thickness in (0.1, 0.2)
        for polarization in ("te", "tm")
    )
    angle_sweeps = tuple(
        _local_sweep(0.15, polarization, 0.5) for polarization in ("te", "tm")
    )
    loss, convergence = _local_loss_evidence()
    qualification = fm.qualify_local_isotropic_medium(
        retrievals,
        angle_sweeps,
        loss,
        convergence,
        fm.LocalIsotropicQualificationPolicy(
            parameter_relative_tolerance=2.0e-4,
            reconstructed_scattering_tolerance=2.0e-5,
            angle_prediction_tolerance=2.0e-5,
            minimum_branch_margin=1.0e-5,
        ),
    )
    return {
        "status": int(qualification.status),
        "qualified": bool(qualification.qualified),
        "reasons": list(qualification.reasons),
        "thickness_invariance_residual": float(
            qualification.thickness_invariance_residual
        ),
        "polarization_invariance_residual": float(
            qualification.polarization_invariance_residual
        ),
        "angle_prediction_residual": float(qualification.angle_prediction_residual),
        "reconstructed_scattering_residual": float(
            qualification.reconstructed_scattering_residual
        ),
        "minimum_branch_margin": float(qualification.minimum_branch_margin),
        "loss_evidence_accepted": bool(qualification.loss_evidence_accepted),
        "loss_convergence_accepted": bool(qualification.loss_convergence_accepted),
    }


def qualification() -> dict[str, object]:
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "directional_interface_power": _interface_case(),
        "physical_loss": _loss_and_convergence_case(),
        "equivalent_slab_retrieval": _retrieval_case(),
        "local_isotropic_qualification": _local_isotropic_case(),
        "propagation": _propagation_case(),
        "reference": {
            "upstream_release": "v1.7.1",
            "upstream_commit": "e13d422cbb8b77820a5e375eb9f5c415be01b81e",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fourier_modal_maxwell_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
