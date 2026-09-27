from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _astrodynamics_context(scale: Any = None) -> Any:
    astro = phx.applications.astrodynamics
    resolved_scale = astro.AstrodynamicsScaleContract.si() if scale is None else scale
    return astro.AstrodynamicsContext(
        resolved_scale,
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _qnm_table(
    frequency: Any = (0.1, 0.2),
    damping_time: Any = (10.0, 5.0),
    mode_indices: Any = ((2, 2, 0), (2, 2, 1)),
    *,
    source: Any = "catalog-a",
    time_unit: Any = "geometric-time",
    frequency_convention: Any = "cycles-per-time",
) -> Any:
    physics = phx.applications.astrophysics
    return physics.QnmModeTable(
        frequency,
        damping_time,
        mode_indices,
        physics.ObservationDataProvenance.native(source),
        time_unit=time_unit,
        frequency_convention=frequency_convention,
    )


def test_black_hole_identity_cutovers_scenario_1() -> None:
    astro = phx.applications.astrodynamics
    context = _astrodynamics_context()
    baseline = astro.Schwarzschild1PNForce(4.0, context, speed_of_light=10.0)
    changed_mu = astro.Schwarzschild1PNForce(5.0, context, speed_of_light=10.0)
    changed_light = astro.Schwarzschild1PNForce(4.0, context, speed_of_light=11.0)
    kilometer_scale = astro.AstrodynamicsScaleContract(
        phx.units.KILOMETER,
        phx.units.KILOGRAM,
        phx.units.SECOND,
    )
    changed_scale = astro.Schwarzschild1PNForce(
        4.0, _astrodynamics_context(kilometer_scale), speed_of_light=10.0
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
    compact = phx.applications.compact_objects
    with pytest.raises(ValueError, match="finite monotone"):
        # ty: ignore[invalid-argument-type]
        compact.EquationOfStateTable([0.0, 0.1], [1.0, np.nan])
    with pytest.raises(ValueError, match="positive energy density"):
        # ty: ignore[invalid-argument-type]
        compact.EquationOfStateTable([0.0, 0.1], [0.0, 1.0])
    with pytest.raises(ValueError, match="stable and causal"):
        # ty: ignore[invalid-argument-type]
        compact.EquationOfStateTable([0.0, 1.0], [1.0, 1.1])
    with pytest.raises(ValueError, match="Every piecewise-linear EOS segment"):
        compact.EquationOfStateTable(
            # ty: ignore[invalid-argument-type]
            [0.0, 0.01, 1.51, 1.52],
            # ty: ignore[invalid-argument-type]
            [1.0, 2.0, 3.0, 4.0],
        )
    with pytest.raises(ValueError, match="eos_id"):
        # ty: ignore[invalid-argument-type]
        compact.EquationOfStateTable([0.0, 0.1], [1.0, 1.2], eos_id=" ")
    with pytest.raises(ValueError, match="geometric units"):
        # ty: ignore[invalid-argument-type]
        compact.EquationOfStateTable([0.0, 0.1], [1.0, 1.2], unit_system="SI")
    physics = phx.applications.astrophysics
    provenance = physics.ObservationDataProvenance.native("catalog")

    with pytest.raises(ValueError, match="positive damping times"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([np.nan], [1.0], [[2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="positive damping times"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([0.1], [0.0], [[2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="indices must be integers"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([0.1], [1.0], [[2.0, 2.0, 0.5]], provenance)
    with pytest.raises(ValueError, match="unique valid modes"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([0.1, 0.2], [1.0, 2.0], [[2, 2, 0], [2, 2, 0]], provenance)
    with pytest.raises(ValueError, match="unique valid modes"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([0.1], [1.0], [[-1, 0, 0]], provenance)
    with pytest.raises(TypeError, match="ObservationDataProvenance"):
        # ty: ignore[invalid-argument-type]
        physics.QnmModeTable([0.1], [1.0], [[2, 2, 0]], "catalog")


def test_qnm_and_ringdown_ids_bind_content_units_convention_and_provenance() -> None:
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
