# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Semidiscrete meshfree ADR problems on GMLS clouds under native temporal methods.

Manufactured references are independent closed forms: for
``c* = exp(-t) (1 + 0.5 sin(2 pi x) cos(2 pi y))`` and ``c_t = k lap c + r`` the
reaction is ``r = -c + 4 pi^2 k exp(-t) sin(2 pi x) cos(2 pi y)``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    ParticlePopulationPlan,
    ParticleSetPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    HyperviscosityPlan,
    LocalStencilPolicy,
    MaterialParticleMeasure,
    MeshfreeDiffusionLaw,
    MeshfreeEvolutionPlan,
    MeshfreeEvolutionStatus,
    MeshfreeMotion,
    MeshfreeReactionLaw,
    PreparedMeshfreeEvolution,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.dynamics import TimeGrid
from phydrax.solver import (
    BDFMethod,
    DAESolvePolicy,
    FixedStepProblem,
    RosenbrockAdaptivePolicy,
    solve_dae,
    solve_fixed_step,
    solve_rosenbrock,
)


_KAPPA = 0.05
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)


def _mode(points: jax.Array) -> jax.Array:
    return jnp.sin(2 * jnp.pi * points[:, 0]) * jnp.cos(2 * jnp.pi * points[:, 1])


def _exact(time: float, points: jax.Array) -> jax.Array:
    return jnp.exp(-time) * (1.0 + 0.5 * _mode(points))


def _mms_reaction(
    time: jax.Array, points: jax.Array, value: jax.Array, args: Any
) -> jax.Array:
    del args
    return -value + 4 * jnp.pi**2 * _KAPPA * jnp.exp(-time) * _mode(points)


_REACTION = MeshfreeReactionLaw(_mms_reaction, law_id="mms-decay")


def _random_periodic(count: int, seed: int = 1) -> PreparedPointCloudDiscretization:
    spacing = 1.0 / count
    axis = (np.arange(count) + 0.5) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    points = points + 0.2 * spacing * np.random.default_rng(seed).uniform(
        -1, 1, points.shape
    )
    return PointCloudPlan(
        points,
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        address=_PERIODIC,
    ).prepare()


def _shell_lattice(
    count: int, *, offset: float = 0.5, point_ids: np.ndarray | None = None
) -> PreparedPointCloudDiscretization:
    """Periodic jittered lattice whose 21 neighbors close a distance shell.

    Complete shells keep a positive neighbor-selection gap, i.e. a positive
    fixed-support motion trust.
    """
    spacing = 1.0 / count
    axis = (np.arange(count) + offset) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(3).uniform(-1, 1, points.shape)
    points = np.mod(points + 0.005 * spacing * jitter, 1.0)
    return PointCloudPlan(
        points,
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=21,
        point_ids=point_ids,
        address=_PERIODIC,
    ).prepare()


@pytest.fixture(scope="module")
def lattice() -> PreparedPointCloudDiscretization:
    return _shell_lattice(12)


def _uniform(vector: np.ndarray) -> Any:
    def field(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
        del time, args
        return jnp.broadcast_to(jnp.asarray(vector), points.shape)

    return field


def _final(
    evolution: PreparedMeshfreeEvolution,
    method: Any,
    initial: jax.Array,
    final: float,
    steps: int,
) -> tuple[jax.Array, bool]:
    solution = solve_fixed_step(
        FixedStepProblem(method, initial, t0=0.0, t1=final, step_size=final / steps),
        save_every=steps,
    )
    return solution.states[-1], bool(solution.successful)


def test_diffusion_reaction_mms_converges_in_space() -> None:
    final = 0.2
    rows = []
    for count in (10, 14, 20):
        cloud = _random_periodic(count)
        evolution = MeshfreeEvolutionPlan(
            cloud,
            diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
            reaction=_REACTION,
            plan_id=f"mms-space-{count}",
        ).prepare()
        state, successful = _final(
            evolution,
            evolution.ssprk_method("ssprk33"),
            evolution.initial_state(_exact(0.0, cloud.points)),
            final,
            40,
        )
        assert successful
        error = float(jnp.max(jnp.abs(state - _exact(final, cloud.points))))
        rows.append((1.0 / count, error))
    orders = [
        np.log(coarse / fine) / np.log(h_coarse / h_fine)
        for (h_coarse, coarse), (h_fine, fine) in zip(rows, rows[1:], strict=False)
    ]
    print("h, max error:", rows, "observed orders:", orders)
    assert min(orders) > 1.6


def test_ssp_and_imex_methods_reach_their_temporal_orders(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    evolution = MeshfreeEvolutionPlan(
        lattice,
        diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
        reaction=_REACTION,
        plan_id="mms-time",
    ).prepare()
    initial = evolution.initial_state(_exact(0.0, lattice.points))
    final = 0.2
    reference, _ = _final(
        evolution, evolution.ssprk_method("ssprk54"), initial, final, 160
    )
    observed = {}
    for name, method in (
        ("ssprk33", evolution.ssprk_method("ssprk33")),
        ("ars-222", evolution.imex_method("ars-222")),
    ):
        errors = []
        for steps in (5, 10, 20):
            state, successful = _final(evolution, method, initial, final, steps)
            assert successful
            errors.append(float(jnp.max(jnp.abs(state - reference))))
        observed[name] = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    print("temporal orders:", observed)
    assert min(observed["ssprk33"]) > 2.8
    assert min(observed["ars-222"]) > 1.85


def test_anisotropic_nonlinear_diffusion_uses_converged_implicit_stage_solves(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    law = MeshfreeDiffusionLaw(
        np.asarray([[0.04, 0.015], [0.015, 0.02]]),
        kind="tensor",
        constitutive=lambda value: 1.0 + value**2,
        law_id="anisotropic-quadratic",
    )
    evolution = MeshfreeEvolutionPlan(
        lattice, diffusion=law, plan_id="nonlinear"
    ).prepare()
    points = lattice.points
    initial = evolution.initial_state(
        0.5
        * jnp.exp(jnp.cos(2 * jnp.pi * points[:, 0]) - 1.0)
        * (1.0 + 0.3 * jnp.sin(2 * jnp.pi * points[:, 1]))
    )
    method = evolution.imex_method("ars-222")
    attempt = method.step(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.02), None
    )
    assert bool(attempt.successful)
    evidence = attempt.evidence
    assert evidence is not None
    assert bool(jnp.all(evidence["stage_status"] == 0))
    # Implicit stages ran Newton-Krylov iterations; the explicit stage did not.
    assert int(evidence["stage_iterations"][0]) == 0
    assert bool(jnp.all(evidence["stage_iterations"][1:] > 0))
    state, successful = _final(evolution, method, initial, 0.1, 5)
    reference, _ = _final(evolution, evolution.ssprk_method("ssprk54"), initial, 0.1, 40)
    assert successful
    np.testing.assert_allclose(state, reference, atol=1e-4)


def test_adaptive_rosenbrock_retains_its_accepted_grid_and_bdf_consumes_the_residual() -> (
    None
):
    cloud = _random_periodic(8)
    evolution = MeshfreeEvolutionPlan(
        cloud,
        diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
        reaction=_REACTION,
        plan_id="adaptive",
    ).prepare()
    initial = evolution.initial_state(_exact(0.0, cloud.points))
    final = 0.2
    reference, _ = _final(
        evolution, evolution.ssprk_method("ssprk54"), initial, final, 80
    )
    solution = solve_rosenbrock(
        evolution.differential_problem(initial, t0=0.0, t1=final),
        TimeGrid(jnp.asarray([0.0, 0.1, final]), time_id="adaptive-grid"),
        adaptive=RosenbrockAdaptivePolicy(
            relative_tolerance=1e-6,
            absolute_tolerance=1e-9,
            initial_step=0.01,
            maximum_accepted_steps=64,
            maximum_attempts=128,
        ),
    )
    assert solution.successful
    mesh = solution.temporal_mesh
    assert mesh is not None
    assert mesh.adaptive
    count = int(mesh.count)
    accepted = np.asarray(mesh.accepted_times)[:count]
    assert count == int(solution.stats["accepted_steps"]) >= 2
    assert np.all(np.diff(accepted) > 0.0)
    np.testing.assert_allclose(accepted[-1], final, atol=1e-12)
    np.testing.assert_allclose(solution.states[-1], reference, atol=1e-5)
    dae = solve_dae(
        evolution.dae_problem(initial),
        TimeGrid(jnp.linspace(0.0, final, 21), time_id="bdf-grid"),
        policy=DAESolvePolicy(method=BDFMethod(2)),
    )
    assert dae.successful
    np.testing.assert_allclose(dae.states[-1], reference, atol=5e-4)


def test_translation_preserves_a_uniform_state_and_the_volumes(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    trust = float(jnp.min(lattice.trust_radius))
    shift = np.asarray([0.4 * trust, -0.2 * trust])
    evolution = MeshfreeEvolutionPlan(
        lattice,
        motion=MeshfreeMotion("ale", mesh_velocity=_uniform(shift), law_id="translate"),
        plan_id="gcl-translation",
    ).prepare()
    initial = evolution.initial_state(jnp.ones(lattice.points.shape[0]))
    state, successful = _final(
        evolution, evolution.ssprk_method("ssprk33"), initial, 1.0, 4
    )
    assert successful
    fields = evolution.fields(state)
    np.testing.assert_allclose(fields.concentration, 1.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(fields.volumes, lattice.quadrature_weights, rtol=1e-13)
    np.testing.assert_allclose(fields.points, lattice.points + shift, atol=1e-14)


def test_affine_dilation_volumes_follow_the_geometric_jacobian() -> None:
    count = 10
    spacing = 1.0 / count
    axis = (np.arange(count) + 0.5) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    points = points + 0.2 * spacing * np.random.default_rng(5).uniform(
        -1, 1, points.shape
    )
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=2),
    ).prepare()
    rate = 0.5 * float(jnp.min(cloud.trust_radius))

    def dilation(time: jax.Array, coordinates: jax.Array, args: Any) -> jax.Array:
        del time, args
        return rate * (coordinates - 0.5)

    evolution = MeshfreeEvolutionPlan(
        cloud,
        motion=MeshfreeMotion("ale", mesh_velocity=dilation, law_id="dilate"),
        plan_id="gcl-dilation",
    ).prepare()
    initial = evolution.initial_state(jnp.ones(points.shape[0]))
    state, successful = _final(
        evolution, evolution.ssprk_method("ssprk33"), initial, 1.0, 4
    )
    assert successful
    fields = evolution.fields(state)
    # Free stream is preserved, and div_h of an affine field is exact at every
    # node, so V(t) = V(0) det F(t) = V(0) exp(2 rate t) to temporal accuracy.
    np.testing.assert_allclose(fields.concentration, 1.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        fields.volumes / spacing**2, np.exp(2.0 * rate), rtol=1e-12, atol=0.0
    )


def test_ale_transport_uses_material_minus_mesh_velocity(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    points = lattice.points
    trust = float(jnp.min(lattice.trust_radius))
    material = np.asarray([0.3 * trust, 0.1 * trust])
    pulse = jnp.exp(
        jnp.cos(2 * jnp.pi * points[:, 0]) + 0.5 * jnp.sin(2 * jnp.pi * points[:, 1])
    )
    lagrangian = MeshfreeEvolutionPlan(
        lattice,
        velocity=_uniform(material),
        motion=MeshfreeMotion(
            "ale", mesh_velocity=_uniform(material), law_id="with-material"
        ),
        plan_id="lagrangian",
    ).prepare()
    state, successful = _final(
        lagrangian,
        lagrangian.ssprk_method("ssprk33"),
        lagrangian.initial_state(pulse),
        1.0,
        4,
    )
    assert successful
    # A mesh moving with the material carries nodal values unchanged.
    np.testing.assert_allclose(lagrangian.fields(state).concentration, pulse, atol=1e-13)
    mesh = 0.5 * material
    ale = MeshfreeEvolutionPlan(
        lattice,
        velocity=_uniform(material),
        motion=MeshfreeMotion("ale", mesh_velocity=_uniform(mesh), law_id="half"),
        plan_id="half-mesh",
    ).prepare()
    eulerian = MeshfreeEvolutionPlan(
        lattice, velocity=_uniform(material - mesh), plan_id="relative"
    ).prepare()
    initial = ale.initial_state(pulse)
    count = points.shape[0]
    concentration_rate = ale.rate(0.0, initial)[:count] / lattice.quadrature_weights
    np.testing.assert_allclose(
        concentration_rate, eulerian.rate(0.0, pulse), rtol=0.0, atol=1e-12
    )


def test_material_particle_motion_keeps_mass_and_identity() -> None:
    count = 64
    masses = 0.5 + jnp.linspace(0.0, 1.0, count)
    particles = ParticleSetPlan(jnp.arange(count), masses, ambient_dimension=2).prepare()
    population = ParticlePopulationPlan(particles).initialize(masses=masses)
    measure = MaterialParticleMeasure(particles, population, 1000.0)
    identities = np.asarray(measure.identities)
    cloud = _shell_lattice(8, point_ids=identities)
    trust = float(jnp.min(cloud.trust_radius))
    evolution = MeshfreeEvolutionPlan(
        cloud,
        velocity=_uniform(np.asarray([0.4 * trust, 0.0])),
        motion=MeshfreeMotion("material", material=measure, law_id="material"),
        plan_id="material",
    ).prepare()
    assert evolution.capacity.measure == "material-mass"
    fraction = 0.2 + 0.1 * jnp.sin(2 * jnp.pi * cloud.points[:, 0])
    initial = evolution.initial_state(fraction)
    state, successful = _final(
        evolution, evolution.ssprk_method("ssprk33"), initial, 1.0, 4
    )
    assert successful
    fields = evolution.fields(state)
    # Content is particle mass times mass fraction; it never moves between nodes.
    np.testing.assert_allclose(fields.content, masses * fraction, rtol=1e-14)
    np.testing.assert_allclose(fields.measure, masses)
    np.testing.assert_allclose(fields.volumes, masses / 1000.0, rtol=1e-13)
    relabeled = PointCloudPlan(
        np.asarray(cloud.points),
        np.asarray(cloud.quadrature_weights),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=21,
        point_ids=identities[::-1],
        address=_PERIODIC,
    ).prepare()
    with pytest.raises(ValueError, match="relabeled"):
        MeshfreeEvolutionPlan(
            relabeled,
            motion=MeshfreeMotion("material", material=measure, law_id="material"),
            plan_id="relabeled",
        )


def test_periodic_seam_motion_is_an_accepted_refresh() -> None:
    cloud = _shell_lattice(12, offset=0.99)
    points = cloud.points
    trust = float(jnp.min(cloud.trust_radius))
    seam = float(jnp.min(1.0 - points[:, 0]))
    assert seam < 0.4 * trust
    velocity = _uniform(np.asarray([0.6 * trust, 0.0]))
    pulse = jnp.exp(
        jnp.cos(2 * jnp.pi * points[:, 0]) + 0.5 * jnp.sin(2 * jnp.pi * points[:, 1])
    )
    evolution = MeshfreeEvolutionPlan(
        cloud,
        velocity=velocity,
        motion=MeshfreeMotion("ale", mesh_velocity=velocity, law_id="seam"),
        plan_id="seam",
    ).prepare()
    state, successful = _final(
        evolution,
        evolution.ssprk_method("ssprk33"),
        evolution.initial_state(pulse),
        1.0,
        2,
    )
    assert successful
    fields = evolution.fields(state)
    assert int(jnp.sum(fields.points[:, 0] >= 1.0)) > 0
    np.testing.assert_allclose(fields.concentration, pulse, atol=1e-13)
    successor = evolution.rebase(state)
    wrapped = evolution.repacked(state)
    assert float(jnp.max(successor.fields(wrapped).points[:, 0])) < 1.0
    assert bool(successor.admission(wrapped).accepted)


def test_support_trust_exhaustion_refuses_and_rolls_back_the_full_step(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    trust = float(jnp.max(lattice.trust_radius))
    evolution = MeshfreeEvolutionPlan(
        lattice,
        diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
        motion=MeshfreeMotion(
            "ale", mesh_velocity=_uniform(np.asarray([4.0 * trust, 0.0])), law_id="far"
        ),
        plan_id="far",
    ).prepare()
    initial = evolution.initial_state(_exact(0.0, lattice.points))
    method = evolution.ssprk_method("ssprk33")
    attempt = method.step(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(1.0), None
    )
    assert not bool(attempt.successful)
    admission = evolution.admission(attempt.candidate_state)
    assert int(admission.status) == int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED)
    assert not bool(admission.support_accepted)
    rollout = solve_fixed_step(
        FixedStepProblem(method, initial, t0=0.0, t1=2.0, step_size=1.0)
    )
    assert not rollout.successful
    np.testing.assert_array_equal(rollout.states[-1], initial)
    # A new support epoch at the held state admits a step within its own trust.
    successor = evolution.rebase(initial)
    step = 0.2 * float(successor.capacity.support_trust) / (4.0 * trust)
    resumed = successor.ssprk_method("ssprk33").step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        successor.repacked(initial),
        jnp.asarray(step),
        None,
    )
    assert bool(resumed.successful)


def test_declared_hyperviscosity_dissipates_with_estimated_spectral_evidence(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    plain = MeshfreeEvolutionPlan(
        lattice, diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"), plan_id="plain"
    ).prepare()
    stabilized = MeshfreeEvolutionPlan(
        lattice,
        diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
        hyperviscosity=HyperviscosityPlan(1e-6, order=2),
        plan_id="stabilized",
    ).prepare()
    assert stabilized.hyperviscosity is not None
    points = lattice.points
    value = jnp.sin(6 * jnp.pi * points[:, 0]) * jnp.cos(4 * jnp.pi * points[:, 1])
    extra = stabilized.implicit_rate(0.0, value) - plain.implicit_rate(0.0, value)
    weights = lattice.quadrature_weights
    assert float(jnp.sum(weights * value * extra)) < 0.0
    evidence = stabilized.hyperviscosity.evidence
    assert bool(evidence.dissipative)
    assert float(evidence.explicit_step(2.5)) > 0.0
    estimate = stabilized.spectral_estimate(0.0, value)
    assert estimate.scope == "power-iteration-estimate"
    assert float(estimate.radius) > 0.0


def test_ssp_attempts_publish_admission_evidence_for_accepted_and_refused_steps(
    lattice: PreparedPointCloudDiscretization,
) -> None:
    trust = float(jnp.min(lattice.trust_radius))
    evolution = MeshfreeEvolutionPlan(
        lattice,
        diffusion=MeshfreeDiffusionLaw(_KAPPA, law_id="k"),
        motion=MeshfreeMotion(
            "ale", mesh_velocity=_uniform(np.asarray([0.6 * trust, 0.0])), law_id="drift"
        ),
        plan_id="drift",
    ).prepare()
    initial = evolution.initial_state(_exact(0.0, lattice.points))
    method = evolution.ssprk_method("ssprk33")
    # The first step moves 0.6 trust (accepted), the second exhausts the support.
    rollout = solve_fixed_step(
        FixedStepProblem(method, initial, t0=0.0, t1=3.0, step_size=1.0)
    )
    assert not rollout.successful
    retained = rollout.evidence
    assert retained is not None
    assert (int(retained.accepted_step), int(retained.refused_step)) == (0, 1)
    accepted, refused = retained.accepted.admission, retained.refused.admission
    assert int(accepted.status) == int(MeshfreeEvolutionStatus.ACCEPTED)
    assert bool(accepted.accepted & accepted.support_accepted)
    assert int(refused.status) == int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED)
    assert not bool(refused.accepted | refused.support_accepted)
    assert float(refused.support_margin) < float(accepted.support_margin)
    # The collocation route has no transport CFL certificate.
    assert retained.accepted.transport_cfl is None
    assert retained.refused.transport_cfl is None

    first = method.step(jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(1.0), None)
    second = method.step(
        jnp.asarray(1), jnp.asarray(1.0), first.accepted_state, jnp.asarray(1.0), None
    )
    window = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), first.evidence, second.evidence
    )
    # A coupling window reports the refusing substep and extremes over executed ones.
    refusing = method.reduce_evidence(
        window, jnp.asarray([True, True]), jnp.asarray([True, False])
    ).admission
    assert int(refusing.status) == int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED)
    assert not bool(refusing.accepted)
    assert float(refusing.support_margin) == float(refused.support_margin)
    # A substep that never ran contributes nothing.
    held = method.reduce_evidence(
        window, jnp.asarray([True, False]), jnp.asarray([True, False])
    ).admission
    assert int(held.status) == int(MeshfreeEvolutionStatus.ACCEPTED)
    assert bool(held.accepted & held.support_accepted)
    assert float(held.support_margin) == float(accepted.support_margin)
