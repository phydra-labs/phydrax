#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._axis import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization._spaces import DiscreteFieldSpace, TensorDofLayout
from phydrax.discretization._transfer import FieldTransfer, TransferProperties
from phydrax.discretization.discrete_velocity._hybrid import (
    ConformingFVKineticState,
    FixedConformingFVKineticInterfacePlan,
    KineticShockSensorPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)
from phydrax.discretization.discrete_velocity._semi_lagrangian import (
    PreparedOffLatticeSemiLagrangianDVM,
    SemiLagrangianTransferRequirements,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.finite_volume._boundary import FiniteVolumeBoundarySet
from phydrax.discretization.finite_volume._dynamics import FiniteVolumeMethodPlan
from phydrax.discretization.finite_volume._reconstruction import (
    PiecewiseConstantReconstruction,
)
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._discrete_velocity import (
    ConservativeRelaxationDVMSource,
    DiscreteVelocitySourceComposition,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.linalg import ArraySpace, DenseLinearOperator
from phydrax.solver._discrete_velocity import ConservativeFiniteVolumeDVMPlan


def _mean_equilibrium(state, args):
    del args
    return jnp.broadcast_to(jnp.mean(state, axis=-1, keepdims=True), state.shape)


def _identity_equilibrium(state, args):
    del args
    return state


def _compressible_method(quadrature=None):
    return SmoothCompressibleD2VKineticMethod(
        d2v17_quadrature() if quadrature is None else quadrature,
        IdealGasMaterial(1.4, 1.0),
        ConstantTransport(0.03, 0.04),
    )


def _conserved_state():
    return jnp.asarray((1.0, 0.03, -0.02, 2.5))


def _identity_field_transfer(*, conservative=True, positive=True):
    space = ArraySpace((3,), dtype=jnp.float64)
    field = DiscreteFieldSpace(
        "departure_values",
        "periodic-line",
        TensorDofLayout(("x",), (3,)),
        space,
        representation="cell_average",
        conformity="discontinuous",
    )
    operator = DenseLinearOperator(jnp.eye(3), source=space, target=space)
    return FieldTransfer(
        field,
        field,
        operator,
        properties=TransferProperties(
            constant_preserving=True,
            conservative=conservative,
            positivity_preserving=positive,
            exact_on=("cell_average",),
        ),
    )


def test_d2v_quadratures_certify_centered_maxwellian_moments():
    d2v17 = d2v17_quadrature()
    d2v37 = d2v37_off_lattice_quadrature()

    assert d2v17.population_count == 17
    assert d2v17.transport_kind == "integer_lattice"
    assert d2v37.population_count == 37
    assert d2v37.transport_kind == "off_lattice"
    assert d2v17.certification.passed
    assert d2v37.certification.passed
    np.testing.assert_allclose(
        d2v17.certification.measured_moments,
        d2v17.certification.expected_moments,
        atol=d2v17.certification.tolerance,
    )
    np.testing.assert_allclose(
        d2v37.certification.measured_moments,
        d2v37.certification.expected_moments,
        atol=d2v37.certification.tolerance,
    )


def test_quadrature_rejects_false_lattice_and_failed_moment_claims():
    rule = d2v17_quadrature()
    off_grid_velocities = np.asarray(rule.velocities).copy()
    off_grid_velocities[1, 0] += 0.1
    with pytest.raises(ValueError, match="integer velocities"):
        CertifiedDiscreteVelocityQuadrature(
            "mislabelled",
            off_grid_velocities,
            rule.weights,
            reference_temperature=0.5,
            certified_degree=2,
            transport_kind="integer_lattice",
        )
    with pytest.raises(ValueError, match="failed centered-Maxwellian"):
        CertifiedDiscreteVelocityQuadrature(
            "bad-weights",
            rule.velocities,
            np.asarray(rule.weights) * 1.01,
            reference_temperature=0.5,
            certified_degree=2,
            transport_kind="integer_lattice",
        )


def test_conservative_source_and_composition_preserve_declared_moments():
    quadrature = d2v17_quadrature()
    moment_matrix = quadrature.hydrodynamic_moment_matrix()
    source = ConservativeRelaxationDVMSource(
        quadrature,
        moment_matrix,
        _mean_equilibrium,
        moment_names=("mass", "momentum_x", "momentum_y", "kinetic_energy"),
        equilibrium_id="test-mean-equilibrium",
        relaxation_rate=1.7,
    )
    composed = DiscreteVelocitySourceComposition((source, source))
    populations = jnp.linspace(0.1, 1.7, quadrature.population_count).reshape(
        (1, quadrature.population_count)
    )
    coordinates = jnp.zeros((1, 2))

    evidence = source.evidence(jnp.asarray(0.0), populations, coordinates)
    composed_evidence = composed.evidence(jnp.asarray(0.0), populations, coordinates)

    np.testing.assert_allclose(evidence.moment_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(composed_evidence.moment_residual, 0.0, atol=4e-6)


def test_prepared_off_lattice_semi_lagrangian_transport_and_capabilities():
    quadrature = d2v37_off_lattice_quadrature()
    transfer = _identity_field_transfer()
    prepared = PreparedOffLatticeSemiLagrangianDVM(
        quadrature,
        (transfer,) * quadrature.population_count,
        0.05,
        requirements=SemiLagrangianTransferRequirements(exact_on=("cell_average",)),
    )
    populations = jnp.arange(3 * quadrature.population_count, dtype=jnp.float64).reshape(
        (3, quadrature.population_count)
    )

    transported, evidence = prepared.transport_with_evidence(populations)

    np.testing.assert_allclose(transported, populations)
    np.testing.assert_allclose(evidence.conservation_residual, 0.0, atol=2e-6)

    with pytest.raises(ValueError, match="explicitly off-lattice"):
        PreparedOffLatticeSemiLagrangianDVM(
            d2v17_quadrature(),
            (_identity_field_transfer(),) * 17,
            0.05,
        )
    with pytest.raises(ValueError, match="conservative"):
        PreparedOffLatticeSemiLagrangianDVM(
            quadrature,
            (_identity_field_transfer(conservative=False),) * quadrature.population_count,
            0.05,
        )


def test_finite_volume_dvm_constant_transport_is_conservative():
    quadrature = d2v17_quadrature()
    component_names = tuple(
        f"population_{index}" for index in range(quadrature.population_count)
    )
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(3, periodic=True),
            UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = FiniteVolumePlan(grid, component_names=component_names).prepare()
    method = FiniteVolumeMethodPlan(PiecewiseConstantReconstruction(), RusanovFluxPlan())
    source = DiscreteVelocitySourceComposition(
        (
            ConservativeRelaxationDVMSource(
                quadrature,
                quadrature.hydrodynamic_moment_matrix(),
                _identity_equilibrium,
                moment_names=(
                    "mass",
                    "momentum_x",
                    "momentum_y",
                    "kinetic_energy",
                ),
                equilibrium_id="test-identity-equilibrium",
                relaxation_rate=0.8,
            ),
        )
    )
    prepared = ConservativeFiniteVolumeDVMPlan(
        quadrature,
        discretization,
        method,
        FiniteVolumeBoundarySet.periodic(("x", "y")),
        source=source,
    ).prepare()
    state = jnp.broadcast_to(
        quadrature.weights, discretization.cell_shape + (quadrature.population_count,)
    )

    residual, evidence = prepared.residual_with_evidence(jnp.asarray(0.0), state)

    np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(
        evidence.declared_moment_conservation_defect, 0.0, atol=2e-6
    )


@pytest.mark.parametrize(
    "quadrature_factory", (d2v17_quadrature, d2v37_off_lattice_quadrature)
)
def test_total_energy_equilibrium_and_collision_are_coupled_and_conservative(
    quadrature_factory,
):
    method = _compressible_method(quadrature_factory())
    conserved = _conserved_state()
    equilibrium, equilibrium_evidence = method.equilibrium_with_evidence(conserved)

    np.testing.assert_allclose(
        equilibrium_evidence.recovered_conserved, conserved, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        equilibrium_evidence.recovered_total_energy_flux,
        equilibrium_evidence.target_total_energy_flux,
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(equilibrium_evidence.realizability.realizable)

    particle_perturbation = 1e-5 * method.particle_nullspace_projector[0]
    energy_perturbation = (
        jnp.zeros_like(method.quadrature.weights).at[0].set(1e-5).at[1].set(-1e-5)
    )
    nonequilibrium = SmoothCompressibleKineticState(
        equilibrium.particle_populations + particle_perturbation,
        equilibrium.total_energy_populations + energy_perturbation,
    )
    collided, collision_evidence = method.collide_with_evidence(
        nonequilibrium, jnp.asarray(0.01)
    )

    np.testing.assert_allclose(
        collision_evidence.post_collision_conserved,
        collision_evidence.pre_collision_conserved,
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(collision_evidence.post_collision_realizability.realizable)
    assert not np.allclose(
        np.asarray(collided.total_energy_populations),
        np.asarray(nonequilibrium.total_energy_populations),
    )


def test_realizability_and_shock_sensor_evidence_are_explicit():
    method = _compressible_method()
    conserved = _conserved_state()
    equilibrium = method.equilibrium(conserved)
    invalid = SmoothCompressibleKineticState(
        equilibrium.particle_populations.at[-1].set(-1e-5),
        equilibrium.total_energy_populations,
    )
    invalid_evidence = method.realizability(invalid)
    assert bool(invalid_evidence.macroscopic_admissible)
    assert not bool(invalid_evidence.populations_nonnegative)
    assert not bool(invalid_evidence.realizable)

    sensor = KineticShockSensorPlan(method.material, threshold=0.1)
    smooth = sensor.evaluate(conserved, conserved, equilibrium, equilibrium)
    shock_state = conserved.at[-1].set(20.0)
    shock = sensor.evaluate(conserved, shock_state, equilibrium, equilibrium)

    assert not bool(smooth.fv_owned)
    assert bool(smooth.kinetic_eligible)
    assert bool(shock.fv_owned)
    assert not bool(shock.kinetic_eligible)
    assert sensor.shock_owner == "finite_volume"


def test_fixed_hybrid_interface_uses_common_flux_and_atomic_rollback():
    method = _compressible_method()
    system = EulerSystem(2, material=method.material)
    plan = FixedConformingFVKineticInterfacePlan(
        method, system, jnp.asarray((1.0, 0.0)), "face-7"
    )
    conserved = _conserved_state()
    kinetic = method.equilibrium(conserved)
    flux = plan.common_flux(conserved, kinetic)

    np.testing.assert_allclose(flux.flux_equality_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(flux.moment_lift_residual, 0.0, atol=2e-6)
    previous = ConformingFVKineticState(conserved, kinetic)
    rejected = plan.atomic_update(previous, flux, 1e6, 1e6)

    assert not bool(rejected.evidence.accepted)
    assert bool(rejected.evidence.rollback_applied)
    np.testing.assert_allclose(
        rejected.committed.finite_volume_conserved,
        previous.finite_volume_conserved,
    )
    np.testing.assert_allclose(
        rejected.committed.kinetic.particle_populations,
        previous.kinetic.particle_populations,
    )
    np.testing.assert_allclose(
        rejected.committed.kinetic.total_energy_populations,
        previous.kinetic.total_energy_populations,
    )
    assert plan.shock_owner == "finite_volume"


def test_hybrid_interface_rejects_nonunit_normal_and_wrong_system_layout():
    method = _compressible_method()
    with pytest.raises(ValueError, match="unit length"):
        FixedConformingFVKineticInterfacePlan(
            method,
            EulerSystem(2, material=method.material),
            jnp.asarray((2.0, 0.0)),
            "bad-normal",
        )
    with pytest.raises(ValueError, match="matching compressible"):
        FixedConformingFVKineticInterfacePlan(
            method,
            EulerSystem(1, material=method.material),
            jnp.asarray((1.0, 0.0)),
            "bad-system",
        )
