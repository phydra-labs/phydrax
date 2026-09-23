#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax.linalg import (
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
    RationalFunctionPolicy,
    ShiftedSolvePolicy,
    SpectralInterval,
)
from phydrax.metrix import FlatTorusStateGeometry
from phydrax.operators.path_integral._lattice_fermion import (
    AbstractLatticeDiracOperator,
    LatticeFermionResourcePolicy,
)
from phydrax.operators.path_integral._pseudofermion import (
    dirac_normal_operator,
    evaluate_pseudofermion_action,
    FractionalPowerPseudofermionTerm,
    HasenbuschRatioPseudofermionTerm,
    pseudofermion_force,
    PseudofermionSolveRoles,
    refresh_pseudofermion,
    TwoFlavorPseudofermionTerm,
)
from phydrax.operators.path_integral._rational_approximation import (
    generate_minimax_rational_approximation,
    plan_minimax_rational_approximation,
    power_rational_target,
    RationalApproximationTarget,
)
from phydrax.sampling._rhmc import (
    initialize_rhmc_state,
    integrate_rhmc_trajectory,
    NestedForcePartition,
    NestedForcePlan,
    plan_rhmc,
    prepare_rhmc,
    rhmc_transition,
    sample_rhmc,
    SeparableActionRegistry,
    SeparableActionTerm,
)


class _ToyLatticeDiracOperator(AbstractLatticeDiracOperator):
    mass: float

    def __init__(self, links, /, *, mass: float = 2.0):
        links_ = jnp.asarray(links, dtype=jnp.float64)
        if links_.shape != (2,):
            raise ValueError("Toy links must have shape (2,).")
        space = ArraySpace((2,), dtype=jnp.float64, space_id="toy-spinor-space")
        self.mass = float(mass)
        self.source = space
        self.target = space
        self.boundary = None
        self.links = links_
        self.gamma_matrices = jnp.ones((1, 1, 1), dtype=jnp.float64)
        self.gamma5 = jnp.ones((1, 1), dtype=jnp.float64)
        self.resources = None
        self.resource_policy = LatticeFermionResourcePolicy()
        self.lattice_shape = (2,)
        self.site_count = 2
        self.dimension = 1
        self.spin_components = 1
        self.color_components = 1
        self.boundary_id = "toy-periodic-boundary"
        self.representation_id = "toy-trivial-representation"
        self.gauge_field_id = "toy-flat-torus-links"
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = f"toy-dirac-mass-{self.mass.hex()}"

    def _apply(self, vector):
        return (self.mass + 0.25 * jnp.cos(self.links)) * vector

    def _adjoint_apply(self, vector):
        return self._apply(vector)

    def _gamma5_apply(self, vector):
        return vector

    def diagonal_mv(self, vector):
        return self._apply(self.source.validate(vector))

    def diagonal_inverse_mv(self, vector):
        value = self.source.validate(vector)
        return value / (self.mass + 0.25 * jnp.cos(self.links))

    def with_links(self, links):
        return _ToyLatticeDiracOperator(links, mass=self.mass)


def _dirac_interval():
    dirac = _ToyLatticeDiracOperator(jnp.asarray((0.2, -0.4)))
    interval = SpectralInterval(
        dirac_normal_operator(dirac),
        3.0,
        5.1,
        evidence="verified",
        scope="structural",
    )
    return dirac, interval


def _approximation(interval, target, poles=8):
    plan = plan_minimax_rational_approximation(
        target,
        num_poles=poles,
        verification_points=4097,
        error_metric="relative",
    )
    return generate_minimax_rational_approximation(interval, plan)


def _streaming_rational_policy():
    return RationalFunctionPolicy(
        shifted=ShiftedSolvePolicy(
            "lanczos",
            execution="streaming",
            differentiation="none",
            orthogonalization="three-term",
            max_dimension=2,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-12,
        )
    )


def test_remez_certificate_bounds_error_on_its_spectral_interval():
    _, interval = _dirac_interval()
    target = power_rational_target(-0.5)
    approximation = _approximation(interval, target)
    points = jnp.geomspace(interval.lower, interval.upper, 20_001)
    relative_error = jnp.max(jnp.abs(approximation(points) / target(points) - 1.0))
    assert bool(approximation.successful)
    assert float(relative_error) <= 1.0001 * float(approximation.maximum_relative_error)
    assert float(approximation.maximum_relative_error) < 1.0e-6
    assert float(approximation.witness) >= float(interval.lower)
    assert float(approximation.witness) <= float(interval.upper)


def test_two_flavor_and_rhmc_refresh_actions_recover_gaussian_identity():
    dirac, interval = _dirac_interval()
    two_flavor = TwoFlavorPseudofermionTerm(dirac, interval)
    two_refresh = refresh_pseudofermion(two_flavor, jax.random.key(1))
    two_action = evaluate_pseudofermion_action(
        two_flavor,
        two_refresh.field,
        role="acceptance",
    )
    np.testing.assert_allclose(
        two_action.value,
        two_refresh.gaussian_action,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert bool(two_action.successful)

    shift = 0.7
    ratio_refresh_approximation = _approximation(
        interval,
        RationalApproximationTarget(((0.0, 0.5), (shift, -0.5))),
        poles=10,
    )
    ratio = HasenbuschRatioPseudofermionTerm(
        dirac,
        interval,
        ratio_refresh_approximation,
        mass_shift=shift,
        solves=PseudofermionSolveRoles(refresh=_streaming_rational_policy()),
    )
    ratio_refresh = refresh_pseudofermion(ratio, jax.random.key(2))
    ratio_action = evaluate_pseudofermion_action(
        ratio,
        ratio_refresh.field,
        role="acceptance",
    )
    np.testing.assert_allclose(
        ratio_action.value,
        ratio_refresh.gaussian_action,
        rtol=5.0 * float(ratio_refresh_approximation.maximum_relative_error),
        atol=1.0e-10,
    )
    assert bool(ratio_refresh.solve_error_bound_available)
    assert bool(ratio_refresh.solve_error_bound_certified)
    assert not bool(ratio_action.solve_error_bound_available)

    power = 0.5
    fractional = FractionalPowerPseudofermionTerm(
        dirac,
        interval,
        _approximation(interval, power_rational_target(-power), poles=10),
        _approximation(interval, power_rational_target(0.5 * power), poles=10),
        determinant_power=power,
    )
    fractional_refresh = refresh_pseudofermion(fractional, jax.random.key(3))
    fractional_action = evaluate_pseudofermion_action(
        fractional,
        fractional_refresh.field,
        role="acceptance",
    )
    np.testing.assert_allclose(
        fractional_action.value,
        fractional_refresh.gaussian_action,
        rtol=2.0e-5,
        atol=1.0e-10,
    )


def test_pseudofermion_force_matches_action_directional_derivative():
    dirac, interval = _dirac_interval()
    term = TwoFlavorPseudofermionTerm(dirac, interval)
    refresh = refresh_pseudofermion(term, jax.random.key(4))
    links = dirac.links
    direction = jnp.asarray((0.3, -0.5), dtype=links.dtype)
    force = pseudofermion_force(term, refresh.field, links)
    epsilon = 1.0e-5
    forward = evaluate_pseudofermion_action(
        term,
        refresh.field,
        role="force",
        links=links + epsilon * direction,
    ).value
    backward = evaluate_pseudofermion_action(
        term,
        refresh.field,
        role="force",
        links=links - epsilon * direction,
    ).value
    directional_derivative = (forward - backward) / (2.0 * epsilon)
    np.testing.assert_allclose(
        directional_derivative,
        -jnp.vdot(force.force, direction).real,
        rtol=2.0e-7,
        atol=2.0e-9,
    )
    assert bool(force.successful)


def test_streaming_solve_evidence_reaches_rhmc_transition_and_samples():
    dirac, interval = _dirac_interval()
    streaming = _streaming_rational_policy()
    term = TwoFlavorPseudofermionTerm(
        dirac,
        interval,
        solves=PseudofermionSolveRoles(
            refresh=streaming,
            action=streaming,
            force=streaming,
            acceptance=streaming,
        ),
    )
    registry = SeparableActionRegistry((), (term,))
    plan = plan_rhmc(
        step_size=0.01,
        trajectory_steps=1,
        force_plan=NestedForcePlan((NestedForcePartition((0,), substeps=1),)),
        divergence_threshold=100.0,
    )
    kernel = prepare_rhmc(
        registry,
        plan,
        dirac.links,
        geometry=FlatTorusStateGeometry(2.0 * np.pi),
        local_coordinate_shape=(2,),
    )
    state = initialize_rhmc_state(kernel, dirac.links, key=jax.random.key(17))
    transition = rhmc_transition(kernel, state)
    evidence = transition.evidence

    assert jnp.all(evidence.refresh_solve_error_bound_certified)
    assert jnp.all(evidence.initial_term_solve_error_bound_certified)
    assert jnp.all(evidence.proposed_term_solve_error_bound_certified)
    assert evidence.force_solve_error_bound_certified
    assert jnp.isfinite(evidence.maximum_force_shifted_solution_error_upper_bound)

    samples = sample_rhmc(kernel, state, num_draws=1)
    assert jnp.all(samples.refresh_solve_error_bound_certified)
    assert jnp.all(samples.initial_term_solve_error_bound_certified)
    assert jnp.all(samples.proposed_term_solve_error_bound_certified)
    assert jnp.all(samples.force_solve_error_bound_certified)


def _bosonic_kernel(*, step_size=0.08, divergence_threshold=1000.0):
    geometry = FlatTorusStateGeometry(2.0 * np.pi)
    action = SeparableActionTerm(
        lambda q: 0.7 * jnp.sum(1.0 - jnp.cos(q)),
        term_id="toy-cosine-action",
    )
    registry = SeparableActionRegistry((action,))
    forces = NestedForcePlan((NestedForcePartition((0,), substeps=3),))
    plan = plan_rhmc(
        step_size=step_size,
        trajectory_steps=4,
        force_plan=forces,
        divergence_threshold=divergence_threshold,
    )
    return prepare_rhmc(
        registry,
        plan,
        jnp.zeros((2,), dtype=jnp.float64),
        geometry=geometry,
        local_coordinate_shape=(2,),
    )


def test_nested_force_map_is_reversible():
    kernel = _bosonic_kernel()
    position = jnp.asarray((0.2, -0.4))
    momentum = jnp.asarray((0.7, -0.3))
    forward = integrate_rhmc_trajectory(kernel, position, momentum, ())
    backward = integrate_rhmc_trajectory(
        kernel,
        forward.configuration,
        forward.momentum,
        (),
        step_size=-kernel.plan.step_size,
    )
    np.testing.assert_allclose(backward.configuration, position, atol=2.0e-12)
    np.testing.assert_allclose(backward.momentum, momentum, atol=2.0e-12)
    assert int(forward.force_evaluations) == kernel.plan.force_evaluations


def test_nested_force_work_counts_every_term_evaluation():
    forces = NestedForcePlan(
        (
            NestedForcePartition((0, 1), substeps=2),
            NestedForcePartition((2,), substeps=3),
        )
    )

    assert forces.force_evaluations_per_step == 2 * 2 * 2 + 2 * 6


def test_acceptance_uses_exact_endpoint_energy_and_rejection_rolls_back():
    kernel = _bosonic_kernel(step_size=0.9, divergence_threshold=1.0e-16)
    initial = jnp.asarray((0.9, -0.7))
    state = initialize_rhmc_state(kernel, initial, key=jax.random.key(8))
    transition = rhmc_transition(kernel, state)
    evidence = transition.evidence
    np.testing.assert_allclose(
        evidence.initial_energy,
        evidence.initial_action + evidence.initial_kinetic,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evidence.proposed_energy,
        evidence.proposed_action + evidence.proposed_kinetic,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evidence.energy_error,
        evidence.proposed_energy - evidence.initial_energy,
        rtol=0.0,
        atol=0.0,
    )
    assert bool(evidence.divergent)
    assert not bool(evidence.accepted)
    np.testing.assert_array_equal(transition.state.configuration, state.configuration)
    np.testing.assert_array_equal(transition.state.bosonic_action, state.bosonic_action)


def test_production_restart_is_bitwise_deterministic():
    kernel = _bosonic_kernel()
    state = initialize_rhmc_state(
        kernel,
        jnp.asarray((0.1, -0.2)),
        key=jax.random.key(11),
    )
    uninterrupted = sample_rhmc(kernel, state, num_draws=4)
    prefix = sample_rhmc(kernel, state, num_draws=2)
    restarted = sample_rhmc(kernel, prefix.final_state, num_draws=2)
    np.testing.assert_array_equal(
        uninterrupted.configurations[2:],
        restarted.configurations,
    )
    np.testing.assert_array_equal(
        uninterrupted.accepted[2:],
        restarted.accepted,
    )
    np.testing.assert_array_equal(
        uninterrupted.energy_error[2:],
        restarted.energy_error,
    )
    np.testing.assert_array_equal(
        uninterrupted.final_state.configuration,
        restarted.final_state.configuration,
    )
    np.testing.assert_array_equal(
        uninterrupted.final_state.step_index,
        restarted.final_state.step_index,
    )
