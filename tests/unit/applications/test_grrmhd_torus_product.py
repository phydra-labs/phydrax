#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx
from phydrax._physical import RelativityScaleContract
from phydrax.applications.compact_objects._accretion import FishboneMoncriefTorusPlan
from phydrax.applications.compact_objects._grrmhd_product import (
    grrmhd_fast_light_snapshot,
    GRRMHDTorusInitialDataPlan,
    IngoingKerrGridPlan,
)
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_mhd import IdealValenciaGRMHDSystem
from phydrax.equations._relativistic_radiation import GRGrayM1RadiationSystem
from phydrax.equations._relativistic_radiation_interaction import (
    ConstantGRGrayOpacityPlan,
    GRGrayRadiationInteractionPlan,
)
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.solver._gr_m1_finite_volume import (
    FixedGridGRM1SSPRK3Plan,
    GRM1BoundaryCondition,
    GRM1BoundaryPair,
)
from phydrax.solver._grmhd_boundary import (
    GRMHDBoundaryCondition,
    GRMHDBoundaryPair,
)
from phydrax.solver._grmhd_ct import (
    GRMHDConstrainedTransportPlan,
    GRMHDVectorPotentialGauge,
)
from phydrax.solver._grmhd_runtime import GRMHDSSPRK3Plan
from phydrax.solver._grrmhd_runtime import FixedGridGRRMHDIMEXPlan
from phydrax.solver._grrmhd_source import GRRMHDImplicitSourcePlan
from phydrax.units import KILOGRAM


def test_kerr_torus_initialization_evolves_and_exports_fast_light_snapshot():
    scale = RelativityScaleContract.geometric(KILOGRAM)
    convention = RelativityConvention.canonical()
    eos = GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-12)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=False),
            phx.discretization.UniformCellAxisSpec(2, periodic=False),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("radius", "polar", "ingoing_azimuth"),
    ).prepare(
        jnp.asarray(
            (
                (5.0, 0.6, 0.0),
                (15.0, 2.5, 2.0 * jnp.pi),
            )
        )
    )
    bridge = phx.discretization.StructuredCochainBridge(grid)
    constrained_transport = GRMHDConstrainedTransportPlan(
        bridge,
        gauge=GRMHDVectorPotentialGauge("weyl"),
        divergence_tolerance=1.0e-7,
        compatibility_tolerance=1.0e-7,
    )
    material_system = IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        density_floor=1.0e-14,
        pressure_floor=1.0e-16,
        maximum_magnetization=1.0e8,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    material_boundaries = (
        GRMHDBoundaryPair(
            GRMHDBoundaryCondition("horizon_outflow"),
            GRMHDBoundaryCondition("outflow"),
        ),
        GRMHDBoundaryPair(
            GRMHDBoundaryCondition("reflective"),
            GRMHDBoundaryCondition("reflective"),
        ),
        None,
    )
    material_transport = GRMHDSSPRK3Plan(
        material_system,
        constrained_transport,
        boundaries=material_boundaries,
        cfl=0.1,
    )
    radiation_system = GRGrayM1RadiationSystem(scale, convention)
    discretization = FiniteVolumePlan(
        grid, component_names=radiation_system.component_names
    ).prepare()
    radiation_boundaries = (
        GRM1BoundaryPair(
            GRM1BoundaryCondition("vacuum"),
            GRM1BoundaryCondition("outflow"),
        ),
        GRM1BoundaryPair(
            GRM1BoundaryCondition("reflective"),
            GRM1BoundaryCondition("reflective"),
        ),
        None,
    )
    radiation_transport = FixedGridGRM1SSPRK3Plan(
        radiation_system,
        discretization,
        boundaries=radiation_boundaries,
        cfl=0.1,
    )
    interaction = GRGrayRadiationInteractionPlan(
        radiation_system, ConstantGRGrayOpacityPlan()
    )
    source = GRRMHDImplicitSourcePlan(
        material_system, interaction, caloric_temperature_scale=1.0
    )
    runtime = FixedGridGRRMHDIMEXPlan(material_transport, radiation_transport, source)
    kerr = IngoingKerrGridPlan(discretization, scale, convention, 1.0, 0.5)
    torus = FishboneMoncriefTorusPlan(
        eos,
        1.0,
        0.5,
        6.0,
        12.0,
        0.01,
        atmosphere_density=1.0e-8,
        atmosphere_pressure=1.0e-10,
        magnetic_seed_amplitude=1.0e-3,
        magnetic_seed_cutoff=0.2,
    )
    initial = GRRMHDTorusInitialDataPlan(runtime, kerr, torus).initialize(
        step_size=1.0e-7
    )

    assert bool(jnp.all(initial.qualified))
    stages = (initial.geometry, kerr.stage(1.0e-7, 2))
    evolved = runtime.advance(initial.state, 0.0, 1.0e-7, stages)
    assert bool(evolved.accepted)

    axes = tuple(axis.interval_centers for axis in grid.structured_axes)
    snapshot = grrmhd_fast_light_snapshot(
        runtime,
        evolved.state,
        stages[-1],
        axes,
        electron_mass_per_particle=1.0,
        caloric_temperature_scale=1.0,
        source_id="unit:kerr-grrmhd-torus",
    )
    assert snapshot.spatial_shape == (2, 2, 2)
    assert bool(jnp.any(snapshot.source_mask))
    assert snapshot.chart_id == kerr.chart.name
