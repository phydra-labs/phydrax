from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _slab_case(
    permittivity,
    permeability=1.0,
    *,
    frequency: float = 5.0,
    thickness: float = 0.2,
    bandwidth: int = 1,
    sample_count: int = 3,
    bloch_x: float = 0.0,
    material_role: str = "physical",
    passive: bool | None = True,
    exterior_permittivity=1.0,
    exterior_permeability=1.0,
    numeric_version: str = "case",
):
    harmonics = LatticeHarmonicPlan.parallelogramic(
        (bandwidth,), (sample_count,)
    ).prepare(jnp.asarray(((0.1, 0.0),)), numeric_version=numeric_version)
    exterior = fm.FrequencyMaxwellMaterial(
        exterior_permittivity,
        exterior_permeability,
        material_id="exterior",
        passive=True,
    )
    slab = fm.FrequencyMaxwellMaterial(
        permittivity,
        permeability,
        material_id="slab",
        material_role=material_role,
        passive=passive,
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        frequency,
        jnp.asarray((bloch_x, 0.0)),
        fm.HomogeneousMaxwellPort(exterior, port_id="left"),
        (
            fm.FourierModalLayer(
                slab,
                thickness,
                fm.DirectFourierFactorizationPlan(),
                layer_id="slab",
            ),
        ),
        fm.HomogeneousMaxwellPort(exterior, port_id="right"),
        numeric_version=numeric_version,
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    return harmonics, prepared


def _solve(prepared, polarization: str = "te", side: str = "left"):
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        prepared.problem.harmonics.plan.layout.mode_ids[0],
        polarization,
        side=side,
    )
    return fm.solve_fourier_modal_maxwell(prepared, excitation)


def test_directional_power_supports_right_only_and_coherent_two_sided_incidence() -> None:
    _, prepared = _slab_case(2.25, numeric_version="directional")
    right_result = _solve(prepared, side="right")
    np.testing.assert_allclose(
        np.asarray(right_result.left_incoming_power), 0.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        np.asarray(right_result.right_incoming_power),
        1.0,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    left_excitation = fm.plane_wave_excitation(
        prepared.scattering,
        prepared.problem.harmonics.plan.layout.mode_ids[0],
        "te",
        amplitude=1.0,
    )
    right_excitation = fm.plane_wave_excitation(
        prepared.scattering,
        prepared.problem.harmonics.plan.layout.mode_ids[0],
        "te",
        side="right",
        amplitude=1.0j,
    )
    coherent = fm.FourierModalExcitation(
        left_excitation.left_incident + right_excitation.left_incident,
        left_excitation.right_incident + right_excitation.right_incident,
    )
    result = fm.solve_fourier_modal_maxwell(prepared, coherent)
    np.testing.assert_allclose(
        np.asarray(result.left_incoming_power), 1.0, rtol=2.0e-6, atol=2.0e-7
    )
    np.testing.assert_allclose(
        np.asarray(result.right_incoming_power), 1.0, rtol=2.0e-6, atol=2.0e-7
    )
    assert np.all(np.isfinite(np.asarray(result.net_port_power_into_stack)))


def test_volume_loss_is_independent_and_passive_claims_fail_closed() -> None:
    _, prepared = _slab_case(2.25 + 0.08j, numeric_version="lossy")
    result = _solve(prepared)
    revision = fm.fourier_modal_numeric_revision(prepared)
    evidence = fm.evaluate_fourier_modal_loss(
        prepared,
        result,
        fm.FourierModalLossPolicy(
            z_quadrature_order=8,
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-8,
        ),
        numeric_revision=revision,
    )
    assert bool(evidence.eligible)
    assert float(evidence.total_volume_material_loss[0]) > 0.0
    assert float(evidence.z_quadrature_defect[0, 0]) >= 0.0
    corrupted = eqx.tree_at(
        lambda value: value.net_port_power_into_stack,
        result,
        result.net_port_power_into_stack + 0.1,
    )
    broken = fm.evaluate_fourier_modal_loss(
        prepared,
        corrupted,
        fm.FourierModalLossPolicy(z_quadrature_order=8),
        numeric_revision=revision,
    )
    assert int(broken.status) == int(fm.FourierModalLossStatus.CLOSURE_TOLERANCE_NOT_MET)

    active_tensor = jnp.diag(jnp.asarray((2.0 - 0.1j, 2.0 + 0.1j, 2.0 + 0.1j)))
    _, contradicted_prepared = _slab_case(
        active_tensor, numeric_version="passive-contradiction", passive=True
    )
    contradicted = fm.evaluate_fourier_modal_loss(
        contradicted_prepared,
        _solve(contradicted_prepared),
        fm.FourierModalLossPolicy(relative_tolerance=1.0),
        numeric_revision=fm.fourier_modal_numeric_revision(contradicted_prepared),
    )
    assert bool(jnp.any(contradicted.passive_claim_violation))
    assert int(contradicted.status) == int(
        fm.FourierModalLossStatus.PASSIVE_CLAIM_VIOLATED
    )


def test_loss_rejects_artificial_pml_and_has_differentiable_observable() -> None:
    _, pml_prepared = _slab_case(
        2.25 + 0.1j,
        material_role="artificial_pml",
        numeric_version="pml",
    )
    pml_evidence = fm.evaluate_fourier_modal_loss(
        pml_prepared,
        _solve(pml_prepared),
        fm.FourierModalLossPolicy(),
    )
    assert not bool(pml_evidence.eligible)

    def objective(loss):
        _, prepared = _slab_case(
            2.25 + 1.0j * loss,
            passive=False,
            numeric_version="gradient-loss",
        )
        evidence = fm.evaluate_fourier_modal_loss(
            prepared,
            _solve(prepared),
            fm.FourierModalLossPolicy(z_quadrature_order=4),
        )
        return evidence.total_volume_material_loss[0]

    derivative = jax.grad(objective)(jnp.asarray(0.05))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_loss_convergence_requires_distinct_nested_discretizations() -> None:
    evidence = []
    for bandwidth, sample_count in ((1, 3), (3, 7)):
        _, prepared = _slab_case(
            2.25 + 0.05j,
            bandwidth=bandwidth,
            sample_count=sample_count,
            numeric_version=f"convergence-{bandwidth}",
        )
        evidence.append(
            fm.evaluate_fourier_modal_loss(
                prepared,
                _solve(prepared),
                fm.FourierModalLossPolicy(
                    relative_tolerance=1.0e-5, absolute_tolerance=1.0e-8
                ),
                numeric_revision=fm.fourier_modal_numeric_revision(prepared),
            )
        )
    convergence = fm.assess_fourier_modal_loss_convergence(
        tuple(evidence), relative_tolerance=1.0e-5
    )
    assert bool(convergence.nested_refinement)
    assert bool(convergence.port_converged)
    assert bool(convergence.material_converged)


def _retrieval(
    thickness: float,
    polarization: str,
    *,
    bloch_x: float = 0.0,
    bandwidth: int = 1,
):
    cases = []
    revisions = []
    harmonic_mode_id = ""
    for index, frequency in enumerate((4.0, 5.0, 6.0)):
        harmonics, prepared = _slab_case(
            2.25,
            frequency=frequency,
            thickness=thickness,
            bloch_x=bloch_x,
            bandwidth=bandwidth,
            sample_count=max(3, 2 * bandwidth + 1),
            numeric_version=f"retrieval-{thickness}-{polarization}-{bloch_x}-{bandwidth}-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
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
    retrieval = fm.retrieve_equivalent_slab(
        sweep,
        fm.EquivalentSlabRetrievalPlan(
            (-2, 2),
            anchor="known-index",
            anchor_refractive_index=1.5,
            passive_claim=True,
        ),
    )
    return retrieval


def test_retrieval_boundary_rejects_unbound_numeric_revision() -> None:
    harmonics, prepared = _slab_case(2.25, numeric_version="revision-target")
    _, other = _slab_case(3.0, numeric_version="revision-other")
    with pytest.raises(ValueError, match="numeric_revision"):
        fm.prepare_maxwell_modal_sweep(
            (prepared,),
            (fm.fourier_modal_numeric_revision(other),),
            slab_thickness=0.2,
            harmonic_mode_id=harmonics.plan.layout.mode_ids[0],
            polarization="te",
        )


def test_retrieval_uses_absolute_impedance_for_nonvacuum_equal_terminations() -> None:
    cases = []
    revisions = []
    harmonic_mode_id = ""
    for index, frequency in enumerate((0.5, 0.75, 1.0)):
        harmonics, prepared = _slab_case(
            9.0,
            permeability=4.0,
            exterior_permittivity=4.0,
            exterior_permeability=1.0,
            frequency=frequency,
            numeric_version=f"nonvacuum-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
        cases.append(prepared)
        revisions.append(fm.fourier_modal_numeric_revision(prepared))
    sweep = fm.prepare_maxwell_modal_sweep(
        tuple(cases),
        tuple(revisions),
        slab_thickness=0.2,
        harmonic_mode_id=harmonic_mode_id,
        polarization="te",
    )
    retrieval = fm.retrieve_equivalent_slab(
        sweep,
        fm.EquivalentSlabRetrievalPlan(
            (-1, 1),
            anchor="known-index",
            anchor_refractive_index=6.0,
            passive_claim=True,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(retrieval.relative_impedance), 4.0 / 3.0, rtol=2.0e-5
    )
    np.testing.assert_allclose(
        np.asarray(retrieval.effective_impedance), 2.0 / 3.0, rtol=2.0e-5
    )
    np.testing.assert_allclose(np.asarray(retrieval.permittivity), 9.0, rtol=2.0e-5)
    np.testing.assert_allclose(np.asarray(retrieval.permeability), 4.0, rtol=2.0e-5)


def test_modal_sweep_accepts_one_provenance_bound_dispersive_response() -> None:
    cases = []
    revisions = []
    harmonic_mode_id = ""
    for index, frequency in enumerate((4.0, 5.0, 6.0)):
        harmonics, prepared = _slab_case(
            2.0 + 0.1 * frequency,
            frequency=frequency,
            numeric_version=f"dispersive-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
        cases.append(prepared)
        revisions.append(fm.fourier_modal_numeric_revision(prepared))
    sweep = fm.prepare_maxwell_modal_sweep(
        tuple(cases),
        tuple(revisions),
        slab_thickness=0.2,
        harmonic_mode_id=harmonic_mode_id,
        polarization="te",
    )
    assert len(set(sweep.physical_state_digests)) == len(cases)
    assert sweep.physical_stack_digest


def test_equivalent_slab_retrieval_valid_branch_zero_and_multimode_cases() -> None:
    retrieval = _retrieval(0.2, "te")
    assert isinstance(retrieval, fm.EquivalentSlabRetrieval)
    assert int(retrieval.status) == int(fm.EquivalentSlabRetrievalStatus.VALID)
    np.testing.assert_allclose(
        np.asarray(retrieval.refractive_index), 1.5, rtol=2.0e-5, atol=2.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(retrieval.permittivity), 2.25, rtol=2.0e-5, atol=2.0e-6
    )
    zero_transmission = eqx.tree_at(
        lambda value: (
            value.deembedded_left_to_right_transmission,
            value.deembedded_right_to_left_transmission,
        ),
        retrieval.sweep,
        (jnp.zeros((3,), dtype=complex), jnp.zeros((3,), dtype=complex)),
    )
    invalid = fm.retrieve_equivalent_slab(
        zero_transmission,
        fm.EquivalentSlabRetrievalPlan(
            (-1, 1), anchor="known-index", anchor_refractive_index=1.5
        ),
    )
    assert int(invalid.status) == int(fm.EquivalentSlabRetrievalStatus.INELIGIBLE)
    asymmetric = eqx.tree_at(
        lambda value: value.symmetric_termination,
        retrieval.sweep,
        jnp.zeros_like(retrieval.sweep.symmetric_termination),
    )
    asymmetric_result = fm.retrieve_equivalent_slab(
        asymmetric,
        fm.EquivalentSlabRetrievalPlan(
            (-1, 1),
            anchor="known-index",
            anchor_refractive_index=1.5,
            passive_claim=True,
        ),
    )
    assert int(asymmetric_result.status) == int(
        fm.EquivalentSlabRetrievalStatus.INELIGIBLE
    )

    ambiguous = fm.retrieve_equivalent_slab(
        retrieval.sweep,
        fm.EquivalentSlabRetrievalPlan(
            (-2, 2),
            anchor="known-index",
            anchor_refractive_index=1.5,
            anchor_tolerance=10.0,
        ),
    )
    assert int(ambiguous.status) == int(fm.EquivalentSlabRetrievalStatus.AMBIGUOUS)

    multimode_cases = []
    multimode_revisions = []
    harmonic_mode_id = ""
    for index, frequency in enumerate((80.0, 90.0)):
        harmonics, prepared = _slab_case(
            2.25,
            frequency=frequency,
            bandwidth=3,
            sample_count=7,
            numeric_version=f"multimode-{index}",
        )
        harmonic_mode_id = harmonics.plan.layout.mode_ids[0]
        multimode_cases.append(prepared)
        multimode_revisions.append(fm.fourier_modal_numeric_revision(prepared))
    multimode = fm.prepare_maxwell_modal_sweep(
        tuple(multimode_cases),
        tuple(multimode_revisions),
        slab_thickness=0.2,
        harmonic_mode_id=harmonic_mode_id,
        polarization="te",
    )
    rejected = fm.retrieve_equivalent_slab(
        multimode,
        fm.EquivalentSlabRetrievalPlan(
            (-2, 2), anchor="known-index", anchor_refractive_index=1.5
        ),
    )
    assert bool(jnp.any(multimode.additional_propagating_orders))
    assert int(rejected.status) == int(fm.EquivalentSlabRetrievalStatus.INELIGIBLE)


def test_local_isotropic_qualification_accepts_slab_and_rejects_disagreement() -> None:
    retrievals = tuple(
        _retrieval(thickness, polarization)
        for thickness in (0.1, 0.2)
        for polarization in ("te", "tm")
    )
    angle_sweeps = tuple(
        _retrieval(0.15, polarization, bloch_x=0.5) for polarization in ("te", "tm")
    )
    losses = []
    for bandwidth, sample_count in ((1, 3), (3, 7)):
        _, prepared = _slab_case(
            2.25,
            bandwidth=bandwidth,
            sample_count=sample_count,
            numeric_version=f"qualification-loss-{bandwidth}",
        )
        losses.append(
            fm.evaluate_fourier_modal_loss(
                prepared,
                _solve(prepared),
                fm.FourierModalLossPolicy(relative_tolerance=1.0e-5),
                numeric_revision=fm.fourier_modal_numeric_revision(prepared),
            )
        )
    convergence = fm.assess_fourier_modal_loss_convergence(
        tuple(losses), relative_tolerance=1.0e-5
    )
    policy = fm.LocalIsotropicQualificationPolicy(
        parameter_relative_tolerance=2.0e-4,
        reconstructed_scattering_tolerance=2.0e-5,
        angle_prediction_tolerance=2.0e-5,
        minimum_branch_margin=1.0e-5,
    )
    qualified = fm.qualify_local_isotropic_medium(
        retrievals, angle_sweeps, tuple(losses), convergence, policy
    )
    assert int(qualified.status) == int(fm.LocalIsotropicQualificationStatus.QUALIFIED)
    disagreed = eqx.tree_at(
        lambda value: value.permittivity,
        retrievals[-1],
        retrievals[-1].permittivity * 1.2,
    )
    rejected = fm.qualify_local_isotropic_medium(
        (*retrievals[:-1], disagreed),
        angle_sweeps,
        tuple(losses),
        convergence,
        policy,
    )
    assert int(rejected.status) == int(fm.LocalIsotropicQualificationStatus.INELIGIBLE)
    assert (
        "thickness_disagreement" in rejected.reasons
        or "polarization_disagreement" in rejected.reasons
    )
    unrelated_angle = replace(
        angle_sweeps[-1],
        physical_stack_digest="unrelated-stack-digest",
    )
    unrelated = fm.qualify_local_isotropic_medium(
        retrievals,
        (*angle_sweeps[:-1], unrelated_angle),
        tuple(losses),
        convergence,
        policy,
    )
    assert int(unrelated.status) == int(fm.LocalIsotropicQualificationStatus.INELIGIBLE)
    assert "unrelated_physical_stack" in unrelated.reasons
