import jax
import jax.numpy as jnp

from phydrax import ein
from phydrax._physical import RelativityScaleContract
from phydrax.discretization._axis import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization.finite_volume._reconstruction import (
    MUSCLReconstruction,
    UnlimitedLimiter,
)
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_hydrodynamics import ValenciaGRHDSystem
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.solver._relativistic_finite_volume import (
    FixedGridGRHDSSPRK3Plan,
    GRHDBoundaryCondition,
    GRHDFaceFluxPlan,
    lower_valencia_stage_geometry,
    metric_aware_grhd_boundary_trace,
)
from phydrax.solver._relativistic_primitive import GRHDC2PPolicy
from phydrax.units import KILOGRAM


def _system():
    scale = RelativityScaleContract.geometric(KILOGRAM)
    return ValenciaGRHDSystem(GammaLawEOS(scale, 5.0 / 3.0))


def _periodic_runtime(cells=16, *, reconstruction=None):
    system = _system()
    grid = TensorGridPlan(
        (UniformCellAxisSpec(cells, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    shape = discretization.cell_shape
    identity = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), shape + (3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(shape, dtype=jnp.float64),
        jnp.zeros(shape + (3,), dtype=jnp.float64),
        identity,
        identity,
        jnp.ones(shape, dtype=jnp.float64),
        jnp.zeros(shape + (3, 3), dtype=jnp.float64),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=system.convention.convention_id,
        scale_id=system.eos.scale.scale_id,
        topology_id="periodic-line",
        geometry_lineage_id="minkowski-lineage",
    )
    runtime = FixedGridGRHDSSPRK3Plan(
        system,
        GRHDC2PPolicy(system),
        discretization,
        reconstruction=reconstruction,
    )
    return runtime, geometry


def _stages(runtime, geometry, step):
    return (
        lower_valencia_stage_geometry(runtime.discretization, geometry, 0.0),
        lower_valencia_stage_geometry(runtime.discretization, geometry, step),
        lower_valencia_stage_geometry(runtime.discretization, geometry, 0.5 * step),
    )


def test_fixed_grid_uniform_state_is_preserved_with_closed_conservation_ledger():
    runtime, geometry = _periodic_runtime()
    stages = _stages(runtime, geometry, 1.0e-3)
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.2, 0.15, 0.0, 0.0), dtype=jnp.float64),
        geometry.leading_shape + (5,),
    )
    conserved = runtime.system.primitive_to_conserved(primitive, geometry)
    state = runtime.initialize(conserved, stages[0])
    result = runtime.advance(state, 0.0, 1.0e-3, stages)

    assert bool(result.successful)
    assert bool(result.finite)
    assert bool(result.converged)
    assert bool(result.physically_valid)
    assert bool(result.qualified)
    assert jnp.allclose(result.accepted.conserved, conserved, atol=1.0e-13)
    assert jnp.allclose(result.ledger.content_defect, 0.0, atol=1.0e-13)
    assert bool(result.ledger.stationary_geometry)
    assert bool(result.ledger.eligible_for_conservation_claim)
    assert jnp.all(result.ledger.atmosphere_increment == 0.0)
    assert jnp.all(result.ledger.c2p_replacement_increment == 0.0)
    assert result.candidate.time == 1.0e-3
    assert result.accepted.accepted_step == 1
    assert result.accepted.content.geometry_version == stages[1].cell.snapshot_token
    assert (
        result.stage_evaluations[0].stress_energy.snapshot_token
        == stages[0].cell.snapshot_token
    )
    assert (
        result.stage_evaluations[0].stress_energy.geometry_lineage_id
        == stages[0].cell.geometry_lineage_id
    )


def test_fixed_grid_smooth_and_shock_paths_are_finite_and_conservative():
    runtime, geometry = _periodic_runtime(32)
    step = 2.0e-4
    stages = _stages(runtime, geometry, step)
    x = runtime.discretization.grid.structured_axes[0].interval_centers
    smooth_primitive = jnp.stack(
        (
            1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * x),
            jnp.full_like(x, 0.2),
            0.1 * jnp.cos(2.0 * jnp.pi * x),
            jnp.zeros_like(x),
            jnp.zeros_like(x),
        ),
        axis=-1,
    )
    smooth = runtime.system.primitive_to_conserved(smooth_primitive, geometry)
    smooth_result = runtime.advance(
        runtime.initialize(smooth, stages[0]), 0.0, step, stages
    )

    shock_primitive = jnp.where(
        (x < 0.5)[:, None],
        jnp.asarray((1.0, 1.0, 0.0, 0.0, 0.0)),
        jnp.asarray((0.125, 0.1, 0.0, 0.0, 0.0)),
    )
    shock = runtime.system.primitive_to_conserved(shock_primitive, geometry)
    shock_result = runtime.advance(
        runtime.initialize(shock, stages[0]), 0.0, step, stages
    )

    assert bool(smooth_result.successful)
    assert bool(shock_result.successful)
    assert bool(jnp.all(jnp.isfinite(smooth_result.accepted.conserved)))
    assert bool(jnp.all(jnp.isfinite(shock_result.accepted.conserved)))
    assert jnp.allclose(
        smooth_result.ledger.content_defect, 0.0, rtol=1.0e-8, atol=1.0e-11
    )
    assert jnp.allclose(
        shock_result.ledger.content_defect, 0.0, rtol=1.0e-8, atol=1.0e-11
    )
    assert float(jnp.max(jnp.abs(smooth_result.accepted.conserved - smooth))) > 0.0
    assert float(jnp.max(jnp.abs(shock_result.accepted.conserved - shock))) > 0.0


def test_invalid_high_order_face_uses_explicit_first_order_fallback_mask():
    runtime, geometry = _periodic_runtime(
        8, reconstruction=MUSCLReconstruction(UnlimitedLimiter())
    )
    stage = lower_valencia_stage_geometry(runtime.discretization, geometry, 0.0)
    density = jnp.asarray((1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 1.0))
    primitive = jnp.stack(
        (
            density,
            jnp.full_like(density, 0.2),
            jnp.zeros_like(density),
            jnp.zeros_like(density),
            jnp.zeros_like(density),
        ),
        axis=-1,
    )
    conserved = runtime.system.primitive_to_conserved(primitive, geometry)
    evaluation = runtime.evaluate_stage(conserved, stage)

    assert bool(evaluation.successful)
    assert bool(jnp.any(evaluation.fallback_masks[0]))
    assert bool(jnp.all(jnp.isfinite(evaluation.face_fluxes[0])))


def test_metric_aware_reflection_uses_spatial_unit_normal_and_reports_incoming_modes():
    system = _system()
    metric = jnp.diag(jnp.asarray((4.0, 1.0, 1.0), dtype=jnp.float64))
    inverse = jnp.diag(jnp.asarray((0.25, 1.0, 1.0), dtype=jnp.float64))
    geometry = ADMGridGeometry(
        jnp.asarray(0.8),
        jnp.asarray((0.1, 0.0, 0.0)),
        metric,
        inverse,
        jnp.asarray(2.0),
        jnp.zeros((3, 3), dtype=jnp.float64),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="scaled-cartesian",
        convention_id=system.convention.convention_id,
        scale_id=system.eos.scale.scale_id,
        topology_id="boundary-face",
        geometry_lineage_id="scaled-face",
    )
    c2p = GRHDC2PPolicy(system)
    interior = jnp.asarray((1.0, 0.2, 0.2, 0.05, 0.0), dtype=jnp.float64)
    trace = metric_aware_grhd_boundary_trace(
        system,
        c2p,
        GRHDFaceFluxPlan("hlle"),
        interior,
        geometry,
        0,
        "lower",
        GRHDBoundaryCondition("reflective"),
    )

    assert jnp.allclose(trace.exterior_primitive[:2], interior[:2])
    assert jnp.allclose(trace.exterior_primitive[2], -interior[2])
    assert jnp.allclose(trace.exterior_primitive[3:], interior[3:])
    assert jnp.allclose(
        ein.contract(
            "i,ij,j->", trace.outward_unit_covector, inverse, trace.outward_unit_covector
        ),
        1.0,
    )
    assert bool(trace.finite)
    assert bool(trace.physically_valid)
    assert trace.compatible_with(geometry)
    assert trace.incoming_characteristic_count.shape == ()


def test_stage_kernel_is_fixed_shape_under_jit_for_coupled_interleaving():
    runtime, geometry = _periodic_runtime(8)
    stage = lower_valencia_stage_geometry(runtime.discretization, geometry, 0.0)
    primitive = jnp.broadcast_to(jnp.asarray((1.0, 0.2, 0.1, 0.0, 0.0)), (8, 5))
    conserved = runtime.system.primitive_to_conserved(primitive, geometry)
    evaluate = jax.jit(lambda state: runtime.evaluate_stage(state, stage))
    result = evaluate(conserved)

    assert result.residual.shape == (8, 5)
    assert result.primitive_recovery.candidates.attempted.shape == (8, 3)
    assert result.stress_energy.energy_density.shape == (8,)
    assert result.face_fluxes[0].shape == (8, 5)
    assert result.fallback_masks[0].shape == (8,)
