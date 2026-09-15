import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _astrodynamics_context(scale=None):
    astro = phx.applications.astrodynamics
    resolved_scale = astro.AstrodynamicsScaleContract.si() if scale is None else scale
    return astro.AstrodynamicsContext(
        resolved_scale,
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _qnm_table(
    frequency=(0.1, 0.2),
    damping_time=(10.0, 5.0),
    mode_indices=((2, 2, 0), (2, 2, 1)),
    *,
    source="catalog-a",
    time_unit="geometric-time",
    frequency_convention="cycles-per-time",
):
    physics = phx.applications.astrophysics
    return physics.QnmModeTable(
        frequency,
        damping_time,
        mode_indices,
        physics.ObservationDataProvenance.native(source),
        time_unit=time_unit,
        frequency_convention=frequency_convention,
    )


def test_schwarzschild_1pn_identity_binds_parameters_and_scale_contract():
    astro = phx.applications.astrodynamics
    context = _astrodynamics_context()
    baseline = astro.Schwarzschild1PNForce(4.0, context, speed_of_light=10.0)
    changed_mu = astro.Schwarzschild1PNForce(5.0, context, speed_of_light=10.0)
    changed_light = astro.Schwarzschild1PNForce(4.0, context, speed_of_light=11.0)
    kilometre_scale = astro.AstrodynamicsScaleContract(
        phx.units.KILOMETER,
        phx.units.KILOGRAM,
        phx.units.SECOND,
    )
    changed_scale = astro.Schwarzschild1PNForce(
        4.0, _astrodynamics_context(kilometre_scale), speed_of_light=10.0
    )

    assert (
        len(
            {
                baseline.force_id,
                changed_mu.force_id,
                changed_light.force_id,
                changed_scale.force_id,
            }
        )
        == 4
    )
    result = baseline.evaluate(0.0, jnp.asarray([2.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(result.acceleration, jnp.asarray([0.07, 0.0, 0.0]))
    assert bool(result.valid)

    with pytest.raises(ValueError, match="finite and positive"):
        astro.Schwarzschild1PNForce(0.0, context)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        baseline.evaluate(0.0, jnp.ones(5))


def test_eos_and_tov_ids_bind_numeric_tables_labels_units_and_radial_grid():
    compact = phx.applications.compact_objects
    pressure = np.asarray([0.0, 0.1, 0.2])
    energy = np.asarray([1.0, 1.2, 1.4])
    baseline = compact.EquationOfStateTable(pressure, energy, eos_id="source-a")
    changed_content = compact.EquationOfStateTable(
        pressure, np.asarray([1.0, 1.21, 1.4]), eos_id="source-a"
    )
    changed_label = compact.EquationOfStateTable(pressure, energy, eos_id="source-b")

    assert len({baseline.eos_id, changed_content.eos_id, changed_label.eos_id}) == 3
    assert baseline.unit_system == "geometric"
    np.testing.assert_allclose(baseline.sound_speed_squared, [0.5, 0.5])

    radial_grid = np.asarray([0.01, 0.1, 0.5, 1.0])
    plan = compact.TovPlan(baseline, radial_grid)
    changed_grid = compact.TovPlan(baseline, np.asarray([0.01, 0.2, 0.5, 1.0]))
    changed_eos = compact.TovPlan(changed_content, radial_grid)
    assert len({plan.plan_id, changed_grid.plan_id, changed_eos.plan_id}) == 3


def test_malformed_eos_tables_fail_before_entering_tov_workflows():
    compact = phx.applications.compact_objects
    with pytest.raises(ValueError, match="finite monotone"):
        compact.EquationOfStateTable([0.0, 0.1], [1.0, np.nan])
    with pytest.raises(ValueError, match="positive energy density"):
        compact.EquationOfStateTable([0.0, 0.1], [0.0, 1.0])
    with pytest.raises(ValueError, match="stable and causal"):
        compact.EquationOfStateTable([0.0, 1.0], [1.0, 1.1])
    with pytest.raises(ValueError, match="Every piecewise-linear EOS segment"):
        compact.EquationOfStateTable(
            [0.0, 0.01, 1.51, 1.52],
            [1.0, 2.0, 3.0, 4.0],
        )
    with pytest.raises(ValueError, match="eos_id"):
        compact.EquationOfStateTable([0.0, 0.1], [1.0, 1.2], eos_id=" ")
    with pytest.raises(ValueError, match="geometric units"):
        compact.EquationOfStateTable([0.0, 0.1], [1.0, 1.2], unit_system="SI")


def test_qnm_and_ringdown_ids_bind_content_units_convention_and_provenance():
    baseline = _qnm_table()
    changed_frequency = _qnm_table(frequency=(0.11, 0.2))
    changed_damping = _qnm_table(damping_time=(11.0, 5.0))
    changed_modes = _qnm_table(mode_indices=((2, 1, 0), (2, 2, 1)))
    changed_units = _qnm_table(time_unit="seconds")
    changed_convention = _qnm_table(frequency_convention="angular-frequency")
    changed_provenance = _qnm_table(source="catalog-b")

    tables = (
        baseline,
        changed_frequency,
        changed_damping,
        changed_modes,
        changed_units,
        changed_convention,
        changed_provenance,
    )
    assert len({table.table_id for table in tables}) == len(tables)

    physics = phx.applications.astrophysics
    plans = tuple(physics.RingdownPlan(table) for table in tables)
    assert len({plan.plan_id for plan in plans}) == len(plans)

    cyclic = physics.RingdownPlan(
        _qnm_table(frequency=(0.5,), damping_time=(2.0,), mode_indices=((2, 2, 0),))
    ).time_domain(jnp.asarray([1.0]), jnp.asarray([1.0 + 0.0j]))
    angular = physics.RingdownPlan(
        _qnm_table(
            frequency=(0.5,),
            damping_time=(2.0,),
            mode_indices=((2, 2, 0),),
            frequency_convention="angular-frequency",
        )
    ).time_domain(jnp.asarray([1.0]), jnp.asarray([1.0 + 0.0j]))
    np.testing.assert_allclose(cyclic, -np.exp(-0.5), atol=1.0e-7)
    np.testing.assert_allclose(angular, np.exp(-0.5 + 0.5j), atol=1.0e-7)


def test_malformed_qnm_catalog_data_fail_explicitly():
    physics = phx.applications.astrophysics
    provenance = physics.ObservationDataProvenance.native("catalog")

    with pytest.raises(ValueError, match="positive damping times"):
        physics.QnmModeTable([np.nan], [1.0], [[2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="positive damping times"):
        physics.QnmModeTable([0.1], [0.0], [[2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="indices must be integers"):
        physics.QnmModeTable([0.1], [1.0], [[2.0, 2.0, 0.5]], provenance)
    with pytest.raises(ValueError, match="unique valid modes"):
        physics.QnmModeTable([0.1, 0.2], [1.0, 2.0], [[2, 2, 0], [2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="unique valid modes"):
        physics.QnmModeTable([0.1], [1.0], [[-1, 0, 0]], provenance)
    with pytest.raises(TypeError, match="ObservationDataProvenance"):
        physics.QnmModeTable([0.1], [1.0], [[2, 2, 0]], "catalog")
