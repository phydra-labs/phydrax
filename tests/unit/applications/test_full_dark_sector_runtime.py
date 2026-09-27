#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._full_dark_sector_runtime import (
    assemble_full_dark_sector_stress,
    FullDarkSectorRuntimePlan,
    FullDarkSectorStageToken,
    NamedStressEnergyComponent,
)
from phydrax.applications.cosmology._quantum_dark_kinetics import (
    QuantumDarkKineticsPlan,
    QuantumKineticState,
)
from phydrax.applications.cosmology._thermal_dark_rates import (
    ThermalDarkRatePlan,
    ThermalKernelArtifact,
)
from phydrax.applications.cosmology._weak_field_relativistic_pm import (
    WeakFieldRelativisticPMPlan,
)
from phydrax.applications.curved_spacetime_qft._coherent_transport import (
    CoherentTransportPlan,
)
from phydrax.applications.curved_spacetime_qft._gauge_covariant_wigner import (
    GaugeCovariantWignerPlan,
)
from phydrax.applications.curved_spacetime_qft._kadanoff_baym import (
    KadanoffBaymTransportPlan,
    WignerGradientPlan,
)
from phydrax.applications.curved_spacetime_qft._off_shell_transport import (
    OffShellTransportPlan,
)
from phydrax.applications.relativistic_scattering._decays import TwoBodyDecayPlan
from phydrax.applications.relativistic_scattering._matrix_element_revision import (
    MatrixElementRevision,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    OrientedEdgePathPlan,
    polygonal_cell_complex,
    prepare_cell_boundary_paths,
    TensorSpectralPlan,
)
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.discretization.splatting import ParticleGridSplatPlan
from phydrax.equations._dark_radiation_moments import DarkRadiationFourForce
from phydrax.equations._uehling_uhlenbeck import UehlingUhlenbeckPlan
from phydrax.graph import MatrixGaugeLinkSpace
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
    StressEnergyProjection,
    U1ChargeRepresentation,
    UnitaryGroup,
)
from phydrax.particle_physics._bound_states import (
    DarkBoundStateLevel,
    DarkBoundStateSpectrum,
    RadiativeCapturePlan,
)
from phydrax.particle_physics._dark_shower import (
    DarkColorRule,
    DarkShowerEpochPlan,
    DarkSplittingChannel,
    DarkSplittingKernelKind,
)
from phydrax.particle_physics._decay_cascade import (
    DarkDecayCascadePlan,
    DarkDecayChannel,
    DarkDecaySpeciesOwner,
)
from phydrax.particle_physics._hadronization import (
    DarkHadronPairChannel,
    DarkStringFragmentationPlan,
)
from phydrax.particle_physics._identity import ParticleCatalogReference
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.solver._dark_radiation_packets import DarkRadiationPacketPlan
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)
from phydrax.units import BARN, COULOMB


def _frame(snapshot_token: int = 7) -> LocalRelativisticFramePlan:
    scale = RelativityScaleContract(
        DimensionalScaleContract.si(),
        1,
        3,
        2,
        1,
        True,
    )
    units = RelativisticUnitContract(
        scale,
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(snapshot_token, dtype=jnp.int32),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-adm",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="observer",
    )
    return LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.asarray((0.5, 0.0, 0.0, 0.0)),
        jnp.asarray(0.5),
        jnp.asarray(0.75),
        observer_id="observer",
        orientation_id="right-handed-future",
    )


def _epoch_plan() -> DarkSectorEpochPlan:
    return DarkSectorEpochPlan(
        packet_capacity=2,
        event_capacity=2,
        product_capacity=2,
        radiation_capacity=2,
        work_capacity=2,
        frontier_capacity=2,
        packet_width=4,
        event_width=4,
        product_width=4,
        radiation_width=4,
        work_width=4,
        frontier_width=4,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
    )


def _matrix_revision() -> MatrixElementRevision:
    return MatrixElementRevision(
        "process",
        "model",
        "provider",
        "rights",
        "normalization",
        "support",
        "proposal",
        "adaptation",
        "training",
        "optimizer",
        "error-model",
        "3" * 64,
        differentiation_id="fixed-profile",
    )


def _stage(snapshot_token: int = 7) -> FullDarkSectorStageToken:
    frame = _frame(snapshot_token)
    epoch = empty_dark_sector_epoch_state(
        _epoch_plan(), epoch_sequence=0, parent_epoch_manifest_id=None
    )
    return FullDarkSectorStageToken(frame, epoch, _matrix_revision())


def _projection(
    stage: FullDarkSectorStageToken,
    energy: float,
    momentum: tuple[float, float, float],
    pressure: float,
    /,
    *,
    projection_id: str,
    snapshot_token: Any = None,
) -> StressEnergyProjection:
    return StressEnergyProjection(
        jnp.asarray(energy),
        jnp.asarray(momentum),
        pressure * jnp.eye(3),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        snapshot_token=(
            stage.geometry_snapshot_token
            if snapshot_token is None
            else jnp.asarray(snapshot_token, dtype=jnp.int32)
        ),
        geometry_lineage_id="flat-adm",
        convention_id=_frame().geometry.convention_id,
        scale_id=_frame().geometry.scale_id,
        topology_id=stage.topology_id,
        projection_id=projection_id,
    )


def _component(
    stage: FullDarkSectorStageToken,
    name: str,
    energy: float,
    /,
    *,
    projection_id: str,
    snapshot_token: Any = None,
) -> NamedStressEnergyComponent:
    return NamedStressEnergyComponent(
        name,
        _projection(
            stage,
            energy,
            (0.1 * energy, 0.0, 0.0),
            0.2 * energy,
            projection_id=projection_id,
            snapshot_token=snapshot_token,
        ),
        conservation_defect=0.0,
        constraint_defect=0.0,
        gauge_defect=0.0,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=True,
        evidence_id=f"evidence-{name}",
    )


def test_total_stress_has_exact_component_only_and_additive_limits() -> None:
    stage = _stage()
    matter = _component(stage, "matter", 2.0, projection_id="matter-stress")
    radiation = _component(stage, "radiation", 0.5, projection_id="radiation-stress")

    matter_only = assemble_full_dark_sector_stress((matter,), stage)
    coupled = assemble_full_dark_sector_stress((matter, radiation), stage)

    assert jnp.array_equal(
        matter_only.total.energy_density, matter.projection.energy_density
    )
    assert jnp.array_equal(
        matter_only.total.momentum_covector, matter.projection.momentum_covector
    )
    assert jnp.allclose(coupled.total.energy_density, 2.5)
    assert jnp.allclose(
        coupled.total.stress_covariant,
        matter.projection.stress_covariant + radiation.projection.stress_covariant,
    )
    assert bool(coupled.successful)


def test_shared_stage_mismatch_is_refused() -> None:
    stage = _stage(snapshot_token=7)
    stale = _component(
        stage,
        "matter",
        1.0,
        projection_id="stale",
        snapshot_token=6,
    )

    with pytest.raises(Exception, match="stage snapshot token"):
        assemble_full_dark_sector_stress((stale,), stage)


def test_dark_radiation_four_force_is_exactly_opposite() -> None:
    stage = _stage()
    radiation = jnp.asarray((0.3, 0.1, -0.2, 0.4))

    exchange = DarkRadiationFourForce.paired(
        radiation,
        stage.time,
        stage.frame_token,
        source_state_id="radiation-state",
        frame_id=stage.frame_id,
        frame_realization_id=stage.frame_realization_id,
        unit_contract_id=stage.unit_contract_id,
    )

    assert bool(jnp.all(exchange.exact_opposite))
    assert jnp.array_equal(exchange.matter_four_force, -radiation)
    assert jnp.array_equal(exchange.balance_residual, jnp.zeros((4,)))


def _packet_runtime_owners() -> Any:
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(3),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="observer",
        orientation_id="right-handed-future",
    )
    catalog = ParticleCatalogReference(
        source_id="full-dark-sector-species",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/full-dark-sector",
    )
    particles = ParticleSpeciesTable(
        jnp.asarray((10, -10, 30, 32, 100, -100, 101, 200, -200)),
        jnp.asarray((5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
        jnp.asarray((1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0)),
        catalog=catalog,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    epoch = DarkSectorEpochPlan(
        packet_capacity=2,
        event_capacity=4,
        product_capacity=8,
        radiation_capacity=2,
        work_capacity=8,
        frontier_capacity=8,
        packet_width=2,
        event_width=8,
        product_width=4,
        radiation_width=20,
        work_width=8,
        frontier_width=8,
        species_revision_id=particles.table_id,
        topology_revision_id="2" * 64,
    )
    species = tuple(
        DarkSectorSpeciesPlan(
            name,
            np.sqrt(3.0),
            charge_names=("dark",),
            charges=(charge,),
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        )
        for name, charge in (("a", 1.0), ("b", -1.0), ("c", 1.0), ("d", -1.0))
    )
    collision = UehlingUhlenbeckPlan(
        jnp.asarray((-1, -1, -1, -1), dtype=jnp.int8),
        jnp.ones((4, 1)),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.asarray((0.1,)),
        jnp.asarray(
            (
                ((2.0, 1.0, 0.0, 0.0),),
                ((2.0, -1.0, 0.0, 0.0),),
                ((2.0, 0.0, 1.0, 0.0),),
                ((2.0, 0.0, -1.0, 0.0),),
            )
        ),
        jnp.asarray(((1.0,), (-1.0,), (1.0,), (-1.0,))),
        time_unit_id=units.scale.dimensional_scale.time_unit.unit_id,
        invariant_tolerance=1.0e-6,
        entropy_tolerance=1.0e-6,
    )
    quantum = QuantumDarkKineticsPlan(species, units, frame, collision)
    temperature = jnp.asarray((1.0, 2.0, 3.0, 4.0))
    spectral_shape = (4, 4, 2, 3)
    pressure = temperature**2
    entropy = 2.0 * temperature
    thermal = ThermalDarkRatePlan(
        ThermalKernelArtifact(
            temperature,
            jnp.asarray((0.0, 1.0)),
            jnp.asarray((-1.0, 0.0, 1.0)),
            jnp.ones((4, 4)),
            0.1 * jnp.ones((4, 4)),
            0.1j * jnp.ones(spectral_shape),
            jnp.ones(spectral_shape),
            jnp.ones((4, 2, 3), dtype="complex128"),
            jnp.ones((4, 2, 3), dtype="complex128"),
            temperature[None, :],
            pressure,
            temperature * entropy - pressure,
            entropy,
            jnp.eye(12),
            units,
            frame,
            species_plan_ids=quantum.species_plan_ids,
            rate_channel_ids=("runtime-2to2-rate",),
            source_kind="native-analytic",
            thermodynamic_tolerance=1.0e-6,
        )
    )
    support = QuantumKineticState(
        jnp.full((4, 3, 3), 0.2),
        None,
        species=species,
        statistics=jnp.asarray((-1, -1, -1, -1), dtype=jnp.int8),
        spatial_active=jnp.asarray((True, True, True)),
        momentum_active=jnp.asarray((True, True, True)),
        units=units,
        frame=frame,
    )
    coherent = CoherentTransportPlan(
        support,
        jnp.ones((3,)),
        jnp.ones((3,)),
        momentum_quadrature_id="three-mode",
        internal_dimension=2,
    )
    nodes = np.linspace(-4.0, 6.0, 101)
    weights = np.full(nodes.shape, nodes[1] - nodes[0])
    weights[[0, -1]] *= 0.5
    off_shell = OffShellTransportPlan(
        support,
        nodes,
        weights,
        jnp.ones((3,)),
        jnp.ones((3,)),
        energy_quadrature_id="energy-trapezoid",
        momentum_quadrature_id="three-mode",
        spectral_tolerance=0.2,
        dyson_tolerance=1.0e-10,
        kms_tolerance=1.0e-10,
    )
    kadanoff_baym = KadanoffBaymTransportPlan(
        off_shell,
        WignerGradientPlan((3, 3), (1.0,), (1.0,)),
        epoch,
        time_step=0.05,
        memory_depth=2,
    )
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    group = UnitaryGroup(1)
    boundary = prepare_cell_boundary_paths(topology).paths
    edges = boundary.edge_indices[0]
    signs = boundary.orientations[0]
    wigner = GaugeCovariantWignerPlan(
        support,
        MatrixGaugeLinkSpace(topology, group),
        U1ChargeRepresentation(group, 2),
        OrientedEdgePathPlan(
            topology,
            jnp.asarray([[edges[0], edges[1]]]),
            jnp.asarray([[signs[0], signs[1]]]),
            valid=jnp.asarray([[True, True]]),
            path_names=("nearest",),
        ),
    )
    shower = DarkShowerEpochPlan(
        epoch,
        particles,
        units,
        frame,
        (
            DarkSplittingChannel(
                100,
                (100, 101),
                kernel_kind=DarkSplittingKernelKind.FERMION_VECTOR,
                color_rule=DarkColorRule.FUNDAMENTAL_EMISSION,
                kernel_coefficient=1.0,
                envelope_coefficient=2.0,
            ),
        ),
        ordering="transverse-momentum",
        model_id="dark-u1",
        model_revision_id="dark-u1-r1",
        tune_id="runtime-tune",
        alpha_reference=0.2,
        reference_scale=10.0,
        beta0=1.0,
        infrared_cutoff=1.0,
        maximum_scale=10.0,
        z_bounds=(0.1, 0.9),
        proposal_capacity=1,
        production_evidence_ids=("runtime-shower-control",),
    )
    hadronization = DarkStringFragmentationPlan(
        epoch,
        particles,
        units,
        frame,
        (DarkHadronPairChannel((200, -200), 1.0, spectrum_label="dark-meson-pair"),),
        model_id="dark-string",
        model_revision_id="dark-string-r1",
        tune_id="runtime-string-tune",
        string_tension=10.0,
        longitudinal_shape=(0.3, 0.8),
        production_evidence_ids=("runtime-string-control",),
    )
    bound_states = DarkBoundStateSpectrum(
        epoch,
        particles,
        units,
        frame,
        (
            DarkBoundStateLevel(
                20,
                (10, -10),
                rest_energy=9.0,
                charge=0.0,
                degeneracy=1,
                radial_quantum_number=1,
                orbital_angular_momentum=0,
                spin_twice=0,
                level_label="dark-1s",
            ),
        ),
        model_id="dark-coulomb",
        model_revision_id="dark-coulomb-r1",
        spectrum_source_id="analytic-control",
        production_evidence_ids=("spectrum-control",),
    )
    radiative_capture = RadiativeCapturePlan(
        bound_states,
        20,
        capture_coefficient=0.25,
        photo_dissociation_coefficient=0.25,
        emitted_radiation_degeneracy=2,
        constituent_degeneracies=(2, 2),
        multipole_order=1,
        cross_section_unit=BARN,
        coefficient_source_id="dipole-control",
    )
    decay_cascade = DarkDecayCascadePlan(
        epoch,
        particles,
        units,
        frame,
        (
            DarkDecaySpeciesOwner(
                30,
                (
                    DarkDecayChannel(
                        TwoBodyDecayPlan(30, (30, 32), (5.0, 0.0), branching_fraction=1.0)
                    ),
                ),
                owner_id="native-dark-decay",
                mean_proper_lifetime=1.0,
            ),
        ),
        model_id="recursive-dark-decay",
        model_revision_id=canonical_fingerprint({"model": "recursive-dark-decay-r1"}),
        prompt_lifetime_cutoff=0.0,
        production_evidence_ids=("cascade-control",),
    )
    spectral = TensorSpectralPlan(
        tuple(FourierBasisPlan(3) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="weak-field-metric",
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(3)))
    capacity = spectral.grid.points.shape[0]
    stress = RelativisticStressDepositPlan(
        ParticleGridSplatPlan(spectral.grid).prepare(
            ParticleSetPlan(
                jnp.arange(capacity), jnp.ones((capacity,)), ambient_dimension=3
            ).prepare()
        ),
        units,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1.0]),
        mass_shell_relative_tolerance=1.0e-5,
        conservation_tolerance=1.0e-5,
        frame_momentum_relative_tolerance=1.0e-5,
    )
    gravity = WeakFieldRelativisticPMPlan(
        stress, spectral, units, gravitational_constant=1.0e-4
    )
    radiation = DarkRadiationPacketPlan(
        2,
        2,
        units,
        jnp.asarray((0.1, 1.5, 3.0)),
        jnp.asarray((-10.0, -10.0, -10.0)),
        jnp.asarray((10.0, 10.0, 10.0)),
        epoch_plan=epoch,
    )
    return (
        units,
        gravity,
        epoch,
        quantum,
        thermal,
        coherent,
        off_shell,
        kadanoff_baym,
        wigner,
        shower,
        hadronization,
        bound_states,
        radiative_capture,
        decay_cascade,
        radiation,
    )


def test_packet_radiation_runtime_binds_to_the_shared_quantum_frame() -> None:
    owners = _packet_runtime_owners()
    radiation = owners[-1]

    # ty: ignore[too-many-positional-arguments]
    plan = FullDarkSectorRuntimePlan(*owners, _matrix_revision())

    assert plan.radiation is radiation
    assert dict(plan.profile_ids)["radiation"] == radiation.plan_id
