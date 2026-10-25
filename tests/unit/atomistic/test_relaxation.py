#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native relaxation: analytic minima, chart pullbacks, residuals and admission.

Independent controls: closed-form bond and simple-cubic Lennard-Jones minima
(the latter solved by host bisection of ``phi'(a) = -P a^2``), central finite
differences of the enthalpy through the chart, the crystal symmetry tangent
operator's adjoint action against the consumed chart gradient, and the
closed-form implicit derivative of a bond length with respect to its
equilibrium parameter.
"""

from collections.abc import Callable, Sequence
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.atomistic._potential_program import PreparedAtomisticPotentialProgram
from phydrax.atomistic._relaxation import (
    AtomisticCartesianChart,
    AtomisticCellChartKind,
    AtomisticCoordinateConstraint,
    AtomisticDeformationChart,
    AtomisticGaugePolicy,
    AtomisticHamiltonianTransitionKind,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
    AtomisticRelaxationState,
    AtomisticRelaxationStatus,
    AtomisticSensitivityStatus,
    AtomisticStrainSubspace,
    AtomisticSymmetryChart,
    AtomisticTrialOutcome,
    PreparedAtomisticRelaxation,
    program_energy,
    RelaxationMethod,
    trial_geometry,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    atomic_state_from_expansion,
    AtomicSymmetryPlan,
    expand_asymmetric_unit,
    metric_from_cell_parameters,
    RigidSymmetryLayout,
    site_of_point,
    standard_space_group_setting,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.optim import FIRE, TrustScaledLBFGS, TrustScaledLBFGSState
from tests._support.crystal_geometry_molecules import (
    argon,
    chlorofluorobromomethane,
    nontrivial_automorphism,
    trans_dichloroethene,
)


_EMPTY = AtomisticNeighborhoodBundlePlan((), (), ())


def _units() -> AtomisticUnitSystem:
    return phx.atomistic.AtomisticUnitSystem.reduced()


def _bonded_system(
    positions_count: int,
    bonds: Sequence[tuple[int, int]],
    *,
    cell: PeriodicCell | None = None,
    mobile: ArrayLike | None = None,
) -> PreparedAtomisticSystem:
    ids = np.arange(1, positions_count + 1)
    return phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones(ids.shape, dtype=np.int32),
        np.ones(ids.shape),
        _units(),
        atom_type_ids=np.zeros(ids.shape, dtype=np.int32),
        topology=phx.atomistic.MolecularTopologyPlan(
            bonds=[list(bond) for bond in bonds]
        ),
        cell=cell,
        mobile_mask=mobile,
    ).prepare()


def _bond_program(
    system: PreparedAtomisticSystem, stiffness: float = 4.0, length: float = 1.0
) -> PreparedAtomisticPotentialProgram:
    return phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.HarmonicBondPotential(
                np.asarray([stiffness]), np.asarray([length])
            )
        ]
    ).prepare(system, neighborhood_bundle=_EMPTY)


def _lj_program(
    system: PreparedAtomisticSystem,
    *,
    cutoff: float,
    widths: tuple[int, ...] = (1024,),
    roles: tuple[tuple[int, int], ...] = ((4096, 256),),
    epsilon: float = 1.0,
) -> PreparedAtomisticPotentialProgram:
    plan = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([epsilon]), np.asarray([1.0]), cutoff
            )
        ]
    )
    (requirement,) = plan.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticCellListImageSearch(
            maximum_candidate_slots=10**6, deformation_margin=1.0
        ),
        ParticleImageCapacityLadder(
            tuple(
                ParticleImageCapacity(
                    maximum_particles_per_cell=32,
                    maximum_edges=width,
                    maximum_degree=width,
                    maximum_images=4096,
                    maximum_case_edges=width,
                )
                for width in widths
            )
        ),
        skin=0.3,
    )
    ladder = AtomisticRoleCapacityLadder(
        requirement,
        tuple(
            AtomisticRoleCapacity(maximum_routes=routes, maximum_degree=degree)
            for routes, degree in roles
        ),
    )
    bundle = AtomisticNeighborhoodBundlePlan((requirement,), (source,), (ladder,))
    return plan.prepare(system, neighborhood_bundle=bundle)


def _lj_system(positions: np.ndarray, cell: PeriodicCell) -> PreparedAtomisticSystem:
    count = positions.shape[0]
    ids = np.arange(1, count + 1)
    return phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones(ids.shape, dtype=np.int32),
        np.ones(ids.shape),
        _units(),
        atom_type_ids=np.zeros(ids.shape, dtype=np.int32),
        cell=cell,
    ).prepare()


def _simple_cubic(spacing: float, repeats: int = 3) -> np.ndarray:
    grid = np.stack(
        np.meshgrid(*(np.arange(repeats),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    return spacing * grid.astype(np.float64)


def _lj_derivative(distance: float) -> float:
    return 4.0 * (-12.0 * distance**-13 + 6.0 * distance**-7)


def _bisect(function: Callable[[float], float], low: float, high: float) -> float:
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.sign(function(middle)) == np.sign(function(low)):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# -- contracts ------------------------------------------------------------------------


def test_strain_subspaces_and_constraints_are_explicit_contracts() -> None:
    for subspace, size in (
        (AtomisticStrainSubspace.symmetric(), 6),
        (AtomisticStrainSubspace.hydrostatic(), 1),
        (AtomisticStrainSubspace.fixed_volume(), 5),
        (AtomisticStrainSubspace.selected([True, False, False, False, False, True]), 2),
    ):
        generators = subspace.generators
        assert generators.shape == (size, 3, 3)
        np.testing.assert_allclose(
            np.einsum("kij,lij->kl", generators, generators), np.eye(size), atol=1e-12
        )
    assert AtomisticStrainSubspace.fixed_volume().traceless
    with pytest.raises(ValueError, match="logarithmic"):
        AtomisticDeformationChart(
            AtomisticCellChartKind.AFFINE, strain=AtomisticStrainSubspace.fixed_volume()
        )
    with pytest.raises(ValueError, match="idempotent"):
        AtomisticCoordinateConstraint(projectors=2.0 * np.eye(3)[None])
    with pytest.raises(ValueError, match="separate contracts"):
        AtomisticCoordinateConstraint(projectors=np.eye(3)[None], remove_translation=True)
    fixed = AtomisticCoordinateConstraint.fixed_components([[True, False, False]])
    assert fixed.projectors is not None
    np.testing.assert_array_equal(fixed.projectors[0], np.diag([0.0, 1.0, 1.0]))


def test_constrained_systems_and_fixed_cell_pressure_are_refused() -> None:
    system = _bonded_system(2, [(1, 2)])
    program = _bond_program(system)
    convergence = AtomisticRelaxationConvergence(
        projected_tolerance=1e-8, maximum_evaluations=10
    )
    with pytest.raises(ValueError, match="variable-cell"):
        AtomisticRelaxationPlan(
            program,
            AtomisticCartesianChart(),
            method=FIRE(),
            convergence=convergence,
            target_pressure=1.0,
        )
    with pytest.raises(ValueError, match="periodic"):
        AtomisticRelaxationPlan(
            program,
            AtomisticDeformationChart(),
            method=FIRE(),
            convergence=convergence,
        )


# -- analytic minima ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [FIRE(maximum_step=0.05), TrustScaledLBFGS(maximum_step=0.05)],
    ids=["fire", "trust-scaled-lbfgs"],
)
def test_grouped_bond_relaxation_reaches_the_analytic_minimum(
    method: RelaxationMethod,
) -> None:
    system = _bonded_system(2, [(1, 2)])
    program = _bond_program(system, stiffness=4.0, length=1.0)
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=method,
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-9,
            full_force_tolerance=1e-9,
            require_full=True,
            maximum_evaluations=2000,
        ),
        group_count=3,
    )
    relaxation = plan.prepare()
    starts = np.zeros((3, 2, 3))
    starts[:, 1, 0] = [1.4, 0.7, 1.0]
    starts[:, 1, 1] = [0.2, -0.1, 0.0]
    state = relaxation.init(positions=starts)
    runtime, final = relaxation.solve(state, maximum_rounds=2000)
    status = np.asarray(final.status)
    np.testing.assert_array_equal(status, int(AtomisticRelaxationStatus.CONVERGED))
    distance = np.linalg.norm(
        np.asarray(final.positions)[:, 1] - np.asarray(final.positions)[:, 0], axis=-1
    )
    np.testing.assert_allclose(distance, 1.0, atol=1e-8)
    assert np.all(np.asarray(final.full_force_residual) <= 1e-9)
    assert np.all(np.asarray(final.projected_converged))
    # One evaluation per consumed trial; no capacity replay for a bonded program.
    np.testing.assert_array_equal(np.asarray(final.capacity_replays), 0)
    evaluation = runtime.accepted_evaluation(final, 0)
    np.testing.assert_allclose(evaluation.energy, final.energy[0], atol=1e-14)
    np.testing.assert_allclose(evaluation.forces, final.forces[0], atol=1e-12)


def test_fixed_components_separate_projected_and_full_residuals() -> None:
    system = _bonded_system(2, [(1, 2)], mobile=np.asarray([False, True]))
    program = _bond_program(system, stiffness=4.0, length=1.0)
    chart = AtomisticCartesianChart(
        constraint=AtomisticCoordinateConstraint.fixed_components(
            [[False, False, False], [True, False, False]]
        )
    )
    start = np.asarray([[[0.0, 0.0, 0.0], [1.5, 0.4, 0.0]]])

    def relax(
        require_full: bool,
    ) -> tuple[PreparedAtomisticRelaxation, AtomisticRelaxationState]:
        plan = AtomisticRelaxationPlan(
            program,
            chart,
            method=TrustScaledLBFGS(maximum_step=0.05),
            convergence=AtomisticRelaxationConvergence(
                projected_tolerance=1e-9,
                full_force_tolerance=1e-6,
                require_full=require_full,
                maximum_evaluations=300,
            ),
        )
        runtime = plan.prepare()
        return runtime.solve(runtime.init(positions=start), maximum_rounds=300)

    _, projected = relax(False)
    assert int(projected.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    position = np.asarray(projected.positions[0])
    np.testing.assert_allclose(position[1], [1.5, 0.0, 0.0], atol=1e-8)
    # The constrained x direction keeps the full bond force k (d - r0) = 2.
    np.testing.assert_allclose(projected.full_force_residual[0], 2.0, atol=1e-7)
    assert not bool(projected.full_converged[0])
    _, full = relax(True)
    assert int(full.status[0]) == int(AtomisticRelaxationStatus.EVALUATION_LIMIT)


def test_hydrostatic_log_chart_reaches_the_target_pressure_lattice() -> None:
    pressure = 0.4
    spacing = 1.16
    positions = _simple_cubic(spacing)
    cell = PeriodicCell(3 * spacing * np.eye(3))
    system = _lj_system(positions, cell)
    program = _lj_program(system, cutoff=1.4)
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticDeformationChart(
            AtomisticCellChartKind.LOGARITHMIC,
            strain=AtomisticStrainSubspace.hydrostatic(),
        ),
        method=TrustScaledLBFGS(maximum_step=0.02),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-8,
            full_force_tolerance=1e-8,
            full_stress_tolerance=1e-6,
            require_full=True,
            maximum_evaluations=500,
        ),
        target_pressure=pressure,
    )
    relaxation = plan.prepare()
    state = relaxation.init(positions=positions[None])
    _, final = relaxation.solve(state, maximum_rounds=500)
    assert int(final.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    expected = _bisect(lambda a: _lj_derivative(a) + pressure * a * a, 1.0, 1.3)
    assert final.cell_vectors is not None
    assert final.stress is not None
    assert final.volume is not None
    lattice = np.asarray(final.cell_vectors[0])
    np.testing.assert_allclose(lattice, 3 * expected * np.eye(3), atol=1e-6)
    # Hydrostatic balance sigma = -P I of the same accepted iterate.
    np.testing.assert_allclose(final.stress[0], -pressure * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(final.volume[0], (3 * expected) ** 3, rtol=1e-6)


# -- chart pullbacks ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [AtomisticCellChartKind.AFFINE, AtomisticCellChartKind.LOGARITHMIC]
)
def test_cell_chart_pullback_matches_enthalpy_finite_differences(
    kind: AtomisticCellChartKind,
) -> None:
    pressure = 0.3
    spacing = 1.15
    rng = np.random.default_rng(11)
    positions = _simple_cubic(spacing) + 0.03 * rng.normal(size=(27, 3))
    cell = PeriodicCell(3 * spacing * np.eye(3))
    system = _lj_system(positions, cell)
    program = _lj_program(system, cutoff=1.4)
    chart = AtomisticDeformationChart(kind, cell_factor=5.0)
    plan = AtomisticRelaxationPlan(
        program,
        chart,
        method=TrustScaledLBFGS(maximum_step=0.02),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-12, maximum_evaluations=50
        ),
        target_pressure=pressure,
    )
    relaxation = plan.prepare()
    state = relaxation.init(positions=positions[None])
    for _ in range(4):
        state = relaxation.step(state)
    parameters = jax.tree.map(lambda leaf: leaf[0], state.optimizer.accepted_parameters)
    reference = jax.tree.map(lambda leaf: leaf[0], state.reference)
    resources = jax.tree.map(lambda leaf: leaf[0], state.resources)
    gradient = jax.tree.map(lambda leaf: leaf[0], state.optimizer.accepted_gradient)
    # The checked point is away from the identity chart.
    assert float(jnp.max(jnp.abs(parameters["strain"]))) > 1e-4

    def enthalpy(point: dict[str, Array]) -> float:
        (r, h), _ = chart.realize(point, reference)
        geometry, _, images = trial_geometry(program, program.neighborhood_bundle, r, h)
        bound = program.neighborhood_bundle.bind(resources, geometry)
        energy, successful = program_energy(program, bound, r, h, images)
        assert bool(successful)
        return float(energy + pressure * jnp.abs(jnp.linalg.det(h)))

    step = 1e-6
    for component in range(6):
        plus = dict(parameters, strain=parameters["strain"].at[component].add(step))
        minus = dict(parameters, strain=parameters["strain"].at[component].add(-step))
        difference = (enthalpy(plus) - enthalpy(minus)) / (2 * step)
        np.testing.assert_allclose(
            gradient["strain"][component], difference, rtol=1e-6, atol=1e-7
        )
    for atom, axis in ((0, 0), (13, 2), (26, 1)):
        plus = dict(parameters, atoms=parameters["atoms"].at[atom, axis].add(step))
        minus = dict(parameters, atoms=parameters["atoms"].at[atom, axis].add(-step))
        difference = (enthalpy(plus) - enthalpy(minus)) / (2 * step)
        np.testing.assert_allclose(
            gradient["atoms"][atom, axis], difference, rtol=1e-6, atol=1e-7
        )


def test_compiled_nonlinear_chart_observations_use_the_accepted_geometry() -> None:
    setting = standard_space_group_setting(1)
    pool = chlorofluorobromomethane()
    plan = AsymmetricUnitPlan(
        setting,
        [
            AsymmetricUnitSite(
                pool,
                site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7))),
            )
        ],
        chirality="enantiopure",
    )
    quaternion = np.asarray([0.7, 0.1, 0.2, 0.3], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    chart_state = AsymmetricUnitState(
        jnp.asarray([[0.21, 0.13, 0.64]], dtype=jnp.float64),
        jnp.asarray(quaternion[None], dtype=jnp.float64),
        plan.chart.coefficients(
            metric_from_cell_parameters(
                jnp.asarray((7.0, 7.2, 7.1), dtype=jnp.float64),
                jnp.asarray((89.0, 91.0, 90.5), dtype=jnp.float64),
            )
        ),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    expansion = expand_asymmetric_unit(plan, chart_state)
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(1, plan.atom_count + 1, dtype=np.int64),
        np.asarray(pool.component.atomic_numbers, dtype=np.int32),
        np.asarray(pool.component.masses, dtype=np.float64),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.zeros((plan.atom_count,), dtype=np.int32),
        cell=PeriodicCell(np.asarray(expansion.cell.vectors)),
    ).prepare()
    runtime = AtomisticRelaxationPlan(
        _lj_program(system, cutoff=1.4),
        AtomisticSymmetryChart(
            RigidSymmetryLayout(plan), np.arange(plan.atom_count, dtype=np.int64)
        ),
        method=TrustScaledLBFGS(maximum_step=0.02),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-8, maximum_evaluations=50
        ),
    ).prepare()
    initial = runtime.init(
        chart_states=jax.tree.map(lambda value: value[None], chart_state)
    )
    accepted = eqx.filter_jit(lambda owner, state: owner.step(state))(runtime, initial)
    assert bool(accepted.optimizer.has_accepted[0])
    evaluation = runtime.accepted_evaluation(accepted, 0)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(evaluation.energy, accepted.energy[0], atol=1e-11)
    np.testing.assert_allclose(evaluation.forces, accepted.forces[0], atol=1e-10)
    assert accepted.stress is not None
    assert evaluation.stress is not None
    np.testing.assert_allclose(evaluation.stress, accepted.stress[0], atol=1e-11)


# -- failure and resources -------------------------------------------------------------------


def test_failed_cell_and_scientific_trials_roll_back_to_the_accepted_iterate() -> None:
    spacing = 1.15
    positions = _simple_cubic(spacing)
    cell = PeriodicCell(3 * spacing * np.eye(3))
    system = _lj_system(positions, cell)
    program = _lj_program(system, cutoff=1.4)
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticDeformationChart(
            AtomisticCellChartKind.AFFINE, strain=AtomisticStrainSubspace.hydrostatic()
        ),
        method=FIRE(),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-12, maximum_evaluations=50
        ),
        group_count=2,
    )
    relaxation = plan.prepare()
    state = relaxation.step(relaxation.init(positions=np.stack((positions, positions))))
    snapshot = [np.array(leaf) for leaf in jax.tree.leaves(state.resources)]
    # Group 0: an orientation-reversing deformation; group 1: nonfinite atoms.
    proposal = state.optimizer.parameters
    strain = proposal["strain"].at[0, 0].set(-1000.0)
    atoms = proposal["atoms"].at[1, 4, 0].set(jnp.nan)
    bad = eqx.tree_at(
        lambda value: value.optimizer.parameters,
        state,
        {"atoms": atoms, "strain": strain},
    )
    failed = relaxation.step(bad)
    np.testing.assert_array_equal(
        np.asarray(failed.status),
        [
            int(AtomisticRelaxationStatus.CELL_FAILURE),
            int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE),
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(failed.trial_outcome),
        [int(AtomisticTrialOutcome.CELL), int(AtomisticTrialOutcome.SCIENTIFIC)],
    )
    for before, after in zip(snapshot, jax.tree.leaves(failed.resources), strict=True):
        np.testing.assert_array_equal(before, np.asarray(after))
    for name in ("positions", "cell_vectors", "energy", "objective", "forces"):
        np.testing.assert_array_equal(getattr(state, name), getattr(failed, name))
    for left, right in zip(
        jax.tree.leaves(failed.optimizer.parameters),
        jax.tree.leaves(state.optimizer.accepted_parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.asarray(failed.rejected_trials), [1, 1])


def test_capacity_failure_is_replayed_after_growth_without_consuming_budget() -> None:
    clustered = np.asarray(
        [[3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0], [3.0, 3.0, 4.0]]
    )
    cell = PeriodicCell(6.0 * np.eye(3))
    system = _lj_system(clustered, cell)
    program = _lj_program(system, cutoff=2.0, widths=(4, 64), roles=((1, 4), (16, 16)))
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=TrustScaledLBFGS(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-8, maximum_evaluations=400
        ),
    )
    relaxation = plan.prepare()
    state = relaxation.init(positions=clustered[None])
    first = relaxation.step(state)
    assert int(first.status[0]) == int(AtomisticRelaxationStatus.CAPACITY_PENDING)
    assert int(first.trial_outcome[0]) == int(AtomisticTrialOutcome.CAPACITY)
    assert int(first.optimizer.evaluations[0]) == 0
    for left, right in zip(
        jax.tree.leaves(first.optimizer.parameters),
        jax.tree.leaves(state.optimizer.parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    grown, replay = relaxation.grow(first)
    assert grown.prepared_id != relaxation.prepared_id
    assert grown.program.prepared_id != relaxation.program.prepared_id
    assert int(replay.status[0]) == int(AtomisticRelaxationStatus.ITERATING)
    with pytest.raises(ValueError):
        relaxation.step(replay)
    runtime, final = grown.solve(replay, maximum_rounds=400)
    assert int(final.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    assert int(final.capacity_replays[0]) >= 1
    assert int(final.optimizer.evaluations[0]) <= 400
    assert runtime.bundle.source_levels != relaxation.bundle.source_levels or (
        runtime.bundle.role_levels != relaxation.bundle.role_levels
    )


# -- symmetry charts ------------------------------------------------------------------------


def _crystal() -> tuple[AsymmetricUnitPlan, AsymmetricUnitState]:
    setting = standard_space_group_setting(3)
    planar = trans_dichloroethene()
    swap = nontrivial_automorphism(planar)
    general = site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7)))
    plan = AsymmetricUnitPlan(
        setting,
        [
            AsymmetricUnitSite(
                planar,
                site_of_point(setting, (0, 0, 0)),
                site_permutations=[tuple(range(6)), swap],
            ),
            AsymmetricUnitSite(chlorofluorobromomethane(), general),
            AsymmetricUnitSite(argon(), general),
        ],
        chirality="enantiopure",
    )
    quaternion = np.asarray((0.6, 0.3, -0.5, 0.55))
    quaternion /= np.linalg.norm(quaternion)
    half = float(np.sqrt(0.5))
    metric = metric_from_cell_parameters(
        jnp.asarray((6.4, 7.3, 8.1), dtype=jnp.float64),
        jnp.asarray((90.0, 104.0, 90.0), dtype=jnp.float64),
    )
    state = AsymmetricUnitState(
        jnp.asarray([(0.0, 0.37, 0.0), (0.21, 0.13, 0.64), (0.55, 0.81, 0.29)]),
        jnp.asarray([(half, -half, 0.0, 0.0), quaternion, (1.0, 0.0, 0.0, 0.0)]),
        plan.chart.coefficients(metric),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    return plan, state


def test_atomic_symmetry_chart_uses_the_sparse_tangent_adjoint_and_keeps_symmetry() -> (
    None
):
    plan, asu_state = _crystal()
    layout = AtomicSymmetryPlan(plan)
    expansion = expand_asymmetric_unit(plan, asu_state)
    chart_state = atomic_state_from_expansion(layout, asu_state, expansion)
    count = plan.atom_count
    numbers = [
        plan.components[int(component)].atomic_numbers[int(local)]
        for component, local in zip(plan.atom_component, plan.atom_local, strict=True)
    ]
    bonds = []
    for a in range(count):
        for b in range(a + 1, count):
            if plan.atom_copy[a] != plan.atom_copy[b]:
                continue
            component = plan.components[int(plan.atom_component[a])]
            pair = tuple(sorted((int(plan.atom_local[a]), int(plan.atom_local[b]))))
            if pair in {tuple(sorted(bond)) for bond in component.bonds}:
                bonds.append((a + 1, b + 1))
    positions = np.asarray(expansion.positions)
    ids = np.arange(1, count + 1)
    system = phx.atomistic.AtomisticSystemPlan(
        ids,
        np.asarray(numbers),
        np.ones((count,)),
        _units(),
        atom_type_ids=np.zeros((count,), dtype=np.int32),
        topology=phx.atomistic.MolecularTopologyPlan(
            bonds=[list(bond) for bond in bonds]
        ),
        cell=PeriodicCell(np.asarray(expansion.cell.vectors)),
    ).prepare()
    lengths = [
        0.97 * float(np.linalg.norm(positions[a - 1] - positions[b - 1]))
        for a, b in bonds
    ]
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.HarmonicBondPotential(
                np.asarray([1.0]), np.asarray([float(np.mean(lengths))])
            )
        ]
    ).prepare(system, neighborhood_bundle=_EMPTY)
    chart = AtomisticSymmetryChart(layout, np.arange(count))
    swapped = np.arange(count)
    argon_atom = int(np.flatnonzero(np.asarray(numbers) == 18)[0])
    swapped[[0, argon_atom]] = swapped[[argon_atom, 0]]
    with pytest.raises(ValueError, match="element"):
        AtomisticRelaxationPlan(
            program,
            AtomisticSymmetryChart(layout, swapped),
            method=FIRE(),
            convergence=AtomisticRelaxationConvergence(
                projected_tolerance=1e-6, maximum_evaluations=10
            ),
        )
    relaxation = AtomisticRelaxationPlan(
        program,
        chart,
        method=TrustScaledLBFGS(maximum_step=0.01),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-7,
            full_force_tolerance=1e-7,
            maximum_evaluations=800,
        ),
    ).prepare()
    start = relaxation.init(
        chart_states=jax.tree.map(lambda leaf: leaf[None], chart_state)
    )
    first = relaxation.step(start)
    evidence = relaxation.symmetry_evidence(first, 0)
    # At the anchor the consumed chart gradient is the tangent operator adjoint.
    np.testing.assert_allclose(
        evidence.residual.projected, evidence.chart_force, rtol=1e-9, atol=1e-10
    )
    assert float(jnp.max(jnp.abs(evidence.chart_force))) > 1e-3
    _, final = relaxation.solve(first, maximum_rounds=800)
    assert int(final.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    final_evidence = relaxation.symmetry_evidence(final, 0)
    np.testing.assert_allclose(
        final_evidence.residual.projected,
        final_evidence.chart_force,
        rtol=1e-8,
        atol=1e-9,
    )
    assert float(final_evidence.residual.projected_norm) <= 1e-6
    assert float(jnp.max(final_evidence.deviation)) < 1e-9


# -- stationary sensitivity --------------------------------------------------------------------


def _relaxed_dimer() -> tuple[
    PreparedAtomisticSystem,
    PreparedAtomisticPotentialProgram,
    PreparedAtomisticRelaxation,
    AtomisticRelaxationState,
    AtomisticRelaxationState,
]:
    system = _bonded_system(2, [(1, 2)])
    program = _bond_program(system, stiffness=4.0, length=1.0)
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=TrustScaledLBFGS(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-11, maximum_evaluations=500
        ),
    )
    relaxation = plan.prepare()
    start = np.asarray([[[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]]])
    runtime, final = relaxation.solve(
        relaxation.init(positions=start), maximum_rounds=500
    )
    return system, program, runtime, final, relaxation.init(positions=start)


def test_bond_length_sensitivity_is_the_closed_form_implicit_derivative() -> None:
    system, program, runtime, final, unconverged = _relaxed_dimer()
    assert int(final.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    shifted = _bond_program(system, stiffness=4.0, length=2.0)
    dynamic, structure = jax.tree.flatten(eqx.filter(program, eqx.is_inexact_array))
    moved = jax.tree.leaves(eqx.filter(shifted, eqx.is_inexact_array))
    tangent = jax.tree.unflatten(
        structure, [right - left for left, right in zip(dynamic, moved, strict=True)]
    )
    gauge = AtomisticGaugePolicy(translations=True, rotations=True)
    result = runtime.sensitivity(final, 0, program_tangent=tangent, gauge=gauge)
    assert result.admitted
    assert int(result.evidence.negative_count) == 0
    assert float(result.evidence.curvatures[0]) > 0.0
    positions = np.asarray(final.positions[0])
    axis = (positions[1] - positions[0]) / np.linalg.norm(positions[1] - positions[0])
    stretch = float(np.dot(np.asarray(result.positions[1] - result.positions[0]), axis))
    np.testing.assert_allclose(stretch, 1.0, atol=1e-8)
    # Envelope: d E*/d r0 = -k (d* - r0) = 0 at the minimum.
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-8)

    # Undeclared gauge leaves zero modes: the Hessian is not regular.
    singular = runtime.sensitivity(final, 0, program_tangent=tangent)
    assert int(singular.evidence.status) == int(AtomisticSensitivityStatus.WRONG_INERTIA)
    assert np.all(np.isnan(np.asarray(singular.positions)))
    # An unconverged state is refused before any curvature claim.
    refused = runtime.sensitivity(
        unconverged,
        0,
        program_tangent=tangent,
        gauge=gauge,
    )
    assert int(refused.evidence.status) == int(AtomisticSensitivityStatus.NOT_CONVERGED)
    assert np.all(np.isnan(np.asarray(refused.positions)))


# -- Hamiltonian transitions ----------------------------------------------------------------


def test_hamiltonian_transition_keeps_geometry_resets_history_and_budget() -> None:
    system = _bonded_system(2, [(1, 2)])
    program = _bond_program(system, stiffness=4.0, length=1.0)
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=TrustScaledLBFGS(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-10, maximum_evaluations=1000
        ),
        group_count=2,
    )
    relaxation = plan.prepare()
    start = np.zeros((2, 2, 3))
    start[:, 1, 0] = 1.2
    runtime, final = relaxation.solve(
        relaxation.init(positions=start, active=np.asarray([True, False], dtype=bool)),
        maximum_rounds=1000,
    )
    np.testing.assert_array_equal(
        np.asarray(final.status),
        [
            int(AtomisticRelaxationStatus.CONVERGED),
            int(AtomisticRelaxationStatus.INACTIVE),
        ],
    )
    rebound = runtime.program.with_neighborhood_bundle(
        runtime.program.neighborhood_bundle
    )
    with pytest.raises(ValueError, match="same scientific identity"):
        runtime.transition_hamiltonian(rebound, final)

    longer = _bond_program(system, stiffness=4.0, length=1.3)
    moved = runtime.transition_hamiltonian(longer, final)
    record = moved.record
    assert record.successful and record.reason == ""
    assert record.optimizer_policy == "reset-history"
    assert record.kind is AtomisticHamiltonianTransitionKind.SCIENTIFIC_MODEL
    assert record.scientific_id != record.previous_scientific_id
    assert record.plan_id != record.previous_plan_id
    assert moved.runtime.prepared_id != runtime.prepared_id
    assert moved.state.hamiltonian_lineage == (record.transition_id,)
    np.testing.assert_array_equal(np.asarray(record.reprimed), [True, False])
    np.testing.assert_array_equal(
        np.asarray(moved.state.positions), np.asarray(final.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(moved.state.evaluations), np.asarray(final.evaluations)
    )
    np.testing.assert_array_equal(
        np.asarray(moved.state.status),
        [
            int(AtomisticRelaxationStatus.ITERATING),
            int(AtomisticRelaxationStatus.INACTIVE),
        ],
    )
    distance = float(
        np.linalg.norm(np.asarray(final.positions[0, 1] - final.positions[0, 0]))
    )
    np.testing.assert_allclose(
        moved.state.energy[0], 2.0 * (distance - 1.3) ** 2, rtol=1e-12
    )
    # No secant pair of the old Hamiltonian survives.
    assert isinstance(moved.state.optimizer, TrustScaledLBFGSState)
    assert int(np.asarray(moved.state.optimizer.history.count)[0]) == 0
    refused = runtime.step(moved.state)
    assert int(refused.status[0]) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)
    assert int(refused.trial_outcome[0]) == int(AtomisticTrialOutcome.SCIENTIFIC)
    np.testing.assert_array_equal(refused.positions, moved.state.positions)
    np.testing.assert_array_equal(refused.evaluations, moved.state.evaluations)

    updated, continued = moved.runtime.solve(moved.state, maximum_rounds=1000)
    assert int(continued.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    stretched = np.asarray(continued.positions[0, 1] - continued.positions[0, 0])
    np.testing.assert_allclose(np.linalg.norm(stretched), 1.3, atol=1e-8)
    assert int(continued.evaluations[0]) > int(final.evaluations[0])
    # New lanes join the transitioned bank with its Hamiltonian lineage.
    joining = updated.init(
        positions=start,
        active=np.asarray([False, True], dtype=bool),
        hamiltonian_lineage=continued.hamiltonian_lineage,
    )
    assert jax.tree.structure(joining) == jax.tree.structure(continued)


def test_refused_hamiltonian_transition_returns_the_previous_runtime_and_state() -> None:
    clustered = np.asarray(
        [[3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0], [3.0, 3.0, 4.0]]
    )
    system = _lj_system(clustered, PeriodicCell(6.0 * np.eye(3)))
    program = _lj_program(system, cutoff=2.0, widths=(64,), roles=((16, 16),))
    relaxation = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=TrustScaledLBFGS(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-8, maximum_evaluations=400
        ),
    ).prepare()
    runtime, final = relaxation.solve(
        relaxation.init(positions=clustered[None]), maximum_rounds=400
    )
    assert int(final.status[0]) == int(AtomisticRelaxationStatus.CONVERGED)
    # The new Hamiltonian declares capacities too small for the accepted cluster.
    cramped = _lj_program(system, cutoff=2.0, widths=(4,), roles=((1, 4),), epsilon=0.5)
    moved = runtime.transition_hamiltonian(cramped, final)
    assert not moved.record.successful
    assert "capacities" in moved.record.reason
    assert moved.runtime is runtime
    assert moved.state is final
    assert final.hamiltonian_lineage == ()
