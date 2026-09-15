import jax.numpy as jnp

from phydrax._physical import RelativityScaleContract
from phydrax.discretization._axis import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization.finite_volume._boundary import FiniteVolumeBoundarySet
from phydrax.discretization.finite_volume._dynamics import (
    FiniteVolumeMethodPlan,
    PreparedFiniteVolumeDynamics,
)
from phydrax.discretization.finite_volume._reconstruction import (
    PiecewiseConstantReconstruction,
)
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_hydrodynamics import (
    SRHDSystem,
    ValenciaGeometrySource,
    ValenciaGRHDSystem,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


def _eos():
    return GammaLawEOS(RelativityScaleContract.geometric(KILOGRAM), 5.0 / 3.0)


def _geometry(shape, scale_id, *, alpha=None, extrinsic=None):
    dtype = jnp.float64
    identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape, dtype=dtype) if alpha is None else alpha,
        jnp.zeros(shape + (3,), dtype=dtype),
        identity,
        identity,
        jnp.ones(shape, dtype=dtype),
        jnp.zeros(shape + (3, 3), dtype=dtype) if extrinsic is None else extrinsic,
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=RelativityConvention().convention_id,
        scale_id=scale_id,
        topology_id="fixed-grid",
        geometry_lineage_id="minkowski-or-local",
    )


def test_srhd_primitive_round_trip_and_causal_characteristic_bounds():
    system = SRHDSystem(_eos(), 3)
    primitive = jnp.asarray(
        (
            (1.0, 0.2, 0.1, -0.05, 0.02),
            (0.3, 0.5, 0.8, 0.1, -0.1),
            (2.0, 0.01, 0.0, 0.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    conserved = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(conserved)
    lower, upper = system.signal_bounds(conserved, conserved, 0)

    assert jnp.allclose(recovered, primitive, rtol=1.0e-9, atol=1.0e-11)
    assert bool(jnp.all(system.admissible(conserved)))
    assert bool(jnp.all(jnp.isfinite(system.physical_flux(conserved, 0))))
    assert bool(jnp.all(lower >= -1.0))
    assert bool(jnp.all(upper <= 1.0))
    assert bool(jnp.all(lower <= upper))


def test_minkowski_valencia_is_exactly_the_densitized_srhd_specialization():
    eos = _eos()
    srhd = SRHDSystem(eos, 3)
    grhd = ValenciaGRHDSystem(eos)
    primitive = jnp.asarray(
        ((1.2, 0.3, 0.2, -0.1, 0.05), (0.7, 0.8, -0.3, 0.1, 0.2)),
        dtype=jnp.float64,
    )
    geometry = _geometry((2,), eos.scale.scale_id)
    srhd_conserved = srhd.primitive_to_conserved(primitive)
    grhd_conserved = grhd.primitive_to_conserved(primitive, geometry)
    covector = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (2, 3))

    assert jnp.array_equal(grhd_conserved, srhd_conserved)
    assert jnp.allclose(
        grhd.physical_flux_from_primitive(primitive, geometry, 0),
        srhd.physical_flux(srhd_conserved, 0),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    grhd_bounds = grhd.characteristic_bounds_from_primitive(
        primitive, primitive, geometry, covector
    )
    srhd_bounds = srhd.signal_bounds(srhd_conserved, srhd_conserved, 0)
    assert jnp.allclose(grhd_bounds[0], srhd_bounds[0])
    assert jnp.allclose(grhd_bounds[1], srhd_bounds[1])


def test_valencia_local_lapse_gradient_and_extrinsic_curvature_sources():
    eos = _eos()
    system = ValenciaGRHDSystem(eos)
    primitive = jnp.asarray(((1.0, 0.3, 0.0, 0.0, 0.0),), dtype=jnp.float64)
    extrinsic = jnp.zeros((1, 3, 3), dtype=jnp.float64).at[0, 0, 0].set(0.2)
    geometry = _geometry((1,), eos.scale.scale_id, extrinsic=extrinsic)
    alpha_gradient = jnp.asarray(((0.4, 0.0, 0.0),), dtype=jnp.float64)
    source_geometry = ValenciaGeometrySource(
        geometry,
        alpha_gradient,
        jnp.zeros((1, 3, 3), dtype=jnp.float64),
        jnp.zeros((1, 3, 3, 3), dtype=jnp.float64),
    )
    source = system.source_from_primitive(primitive, source_geometry)
    evaluation = system.primitive_evaluation(primitive, geometry)

    expected_momentum_x = (
        -(
            evaluation.rest_mass_density * evaluation.specific_enthalpy
            - evaluation.pressure
        )
        * alpha_gradient[0, 0]
    )
    expected_energy = evaluation.pressure[0] * extrinsic[0, 0, 0]
    assert source[0, 0] == 0.0
    assert jnp.allclose(source[0, 1], expected_momentum_x)
    assert jnp.allclose(source[0, 2:4], 0.0)
    assert jnp.allclose(source[0, -1], expected_energy)

    projection = system.stress_energy_projection(primitive, geometry)
    assert projection.compatible_with(geometry)
    assert projection.snapshot_token == geometry.snapshot_token
    assert projection.geometry_lineage_id == geometry.geometry_lineage_id
    assert bool(projection.all_active_valid)
    assert jnp.allclose(projection.stress_covariant[0], jnp.eye(3) * evaluation.pressure)


def test_srhd_runs_on_native_finite_volume_smooth_and_shock_paths():
    system = SRHDSystem(_eos(), 1)
    grid = TensorGridPlan(
        (UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    dynamics = PreparedFiniteVolumeDynamics(
        system,
        discretization,
        FiniteVolumeMethodPlan(PiecewiseConstantReconstruction(), RusanovFluxPlan()),
        FiniteVolumeBoundarySet.periodic(("x",)),
    )
    smooth_primitive = jnp.broadcast_to(jnp.asarray((1.0, 0.2, 0.15)), (16, 3))
    smooth = system.primitive_to_conserved(smooth_primitive)
    shock_primitive = jnp.where(
        (jnp.arange(16) < 8)[:, None],
        jnp.asarray((1.0, 1.0, 0.0)),
        jnp.asarray((0.125, 0.8, 0.0)),
    )
    shock = system.primitive_to_conserved(shock_primitive)
    smooth_rate = dynamics(jnp.asarray(0.0), smooth)
    shock_rate = dynamics(jnp.asarray(0.0), shock)

    assert jnp.allclose(smooth_rate, 0.0, atol=1.0e-12)
    assert bool(jnp.all(jnp.isfinite(shock_rate)))
    assert float(jnp.max(jnp.abs(shock_rate))) > 0.0
    assert jnp.allclose(
        jnp.sum(shock_rate * discretization.cell_volumes[:, None], axis=0),
        0.0,
        atol=1.0e-11,
    )
