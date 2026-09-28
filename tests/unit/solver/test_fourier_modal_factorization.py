from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _patterned_lattice() -> Any:
    return LatticeHarmonicPlan.parallelogramic((3,), (9,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )


def test_fourier_modal_factorization_scenario_1() -> None:
    lattice = _patterned_lattice()
    material = fm.FrequencyMaxwellMaterial(
        jnp.full(lattice.sample_shape, 2.5),
        material_id="uniform-grid",
    )
    direct = fm.prepare_fourier_material(
        material,
        lattice,
        fm.DirectFourierFactorizationPlan(),
    )
    inverse = fm.prepare_fourier_material(
        material,
        lattice,
        fm.InverseFourierFactorizationPlan(),
    )
    np.testing.assert_allclose(
        np.asarray(inverse.permittivity),
        np.asarray(direct.permittivity),
        rtol=1e-11,
        atol=1e-11,
    )
    lattice = _patterned_lattice()
    coordinate = lattice.fractional_coordinates[..., 0]
    material = fm.FrequencyMaxwellMaterial(
        jnp.where(coordinate < 0.5, 4.0, 1.0),
        material_id="lamellar",
    )
    tangent = jnp.broadcast_to(jnp.asarray((0.0, 1.0)), lattice.sample_shape + (2,))
    prepared = fm.prepare_fourier_material(
        material,
        lattice,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(tangent, frame_id="analytic")
        ),
    )
    assert prepared.tangent_field is not None
    np.testing.assert_allclose(
        np.asarray(jnp.sum(jnp.abs(prepared.tangent_field) ** 2, axis=-1)),
        1.0,
        atol=1e-12,
    )
    assert not bool(prepared.diagnostics.frame_gradient_omitted)
    lattice = _patterned_lattice()
    coordinate = lattice.fractional_coordinates[..., 0]
    material = fm.FrequencyMaxwellMaterial(
        1.0 + 3.0 * jnp.exp(-(((coordinate - 0.5) / 0.15) ** 2)),
        material_id="smooth-pattern",
    )
    prepared = fm.prepare_fourier_material(
        material,
        lattice,
        fm.VectorFourierFactorizationPlan(
            fm.JonesDirectFramePlan(differentiation="frozen")
        ),
    )
    assert prepared.tangent_field is not None
    assert bool(prepared.diagnostics.frame_gradient_omitted)
    assert bool(jnp.all(jnp.isfinite(prepared.tangent_field)))


def test_dynamic_analytic_frame_id_cannot_collide_by_shape() -> None:
    lattice = _patterned_lattice()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="frame-material")
    first_tangent = jnp.broadcast_to(jnp.asarray((1.0, 0.0)), lattice.sample_shape + (2,))
    second_tangent = jnp.broadcast_to(
        jnp.asarray((0.0, 1.0)), lattice.sample_shape + (2,)
    )
    first = fm.FourierModalLayer(
        material,
        0.1,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(first_tangent, frame_id="shared-frame")
        ),
        layer_id="first",
    )
    second = fm.FourierModalLayer(
        material,
        0.1,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(second_tangent, frame_id="shared-frame")
        ),
        layer_id="second",
    )
    port = fm.HomogeneousMaxwellPort(
        fm.FrequencyMaxwellMaterial(1.0, material_id="frame-vacuum"),
        port_id="port",
    )
    problem = fm.FourierModalMaxwellProblem(
        lattice,
        2.0 * jnp.pi,
        jnp.zeros((2,)),
        port,
        (first, second),
        port,
    )
    with pytest.raises(ValueError, match="frame_id"):
        fm.prepare_fourier_modal_maxwell(problem)


# Lalanne & Hugonin, J. Opt. Soc. Am. A 17, 1033 (2000), Tables 1-2, RCWA column
# (the Lalanne-Morris / Granet-Guizal inverse-rule formulation): lamellar grating of
# period, wavelength, and depth 1 um, fill 0.5, ridges and substrate of index
# 0.22 + 6.71i, 30 deg incidence from air. Only reflected orders 0 and -1 propagate.
# The TM entries are order 0 and the TE entries order -1; both converge to the exact
# modal values 0.84848 and 0.73428.
METALLIC_LAMELLAR_TM_ORDER_ZERO = {21: 0.84211, 41: 0.84425, 81: 0.84677}
METALLIC_LAMELLAR_TE_ORDER_MINUS_ONE = {21: 0.76227, 41: 0.73857, 81: 0.73485}
METALLIC_LAMELLAR_EXACT = {"tm": 0.84848, "te": 0.73428}


def _lamellar_reflection(
    harmonics_count: int,
    permittivity: complex,
    polarization: str,
    policy: Any = None,
) -> tuple[dict[int, float], Any]:
    # Fine sampling makes the sampled Fourier coefficients those of the exact profile.
    lattice = LatticeHarmonicPlan.parallelogramic((harmonics_count,), (16384,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    fraction = lattice.fractional_coordinates[..., 0]
    grating = fm.FrequencyMaxwellMaterial(
        jnp.where(fraction < 0.5, permittivity, 1.0 + 0j), material_id="lamellar"
    )
    walls = jnp.broadcast_to(jnp.asarray((0.0, 1.0)), lattice.sample_shape + (2,))
    wavenumber = 2.0 * np.pi
    problem = fm.FourierModalMaxwellProblem(
        lattice,
        wavenumber,
        jnp.asarray((wavenumber * 0.5, 0.0)),
        fm.HomogeneousMaxwellPort(
            fm.FrequencyMaxwellMaterial(1.0 + 0j, material_id="air"), port_id="air"
        ),
        (
            fm.FourierModalLayer(
                grating,
                1.0,
                fm.VectorFourierFactorizationPlan(
                    fm.AnalyticInterfaceFramePlan(walls, frame_id="lamellar-walls")
                ),
                layer_id="ridges",
            ),
        ),
        fm.HomogeneousMaxwellPort(
            fm.FrequencyMaxwellMaterial(permittivity, material_id="metal"),
            port_id="metal",
        ),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem, policy)
    layout = lattice.plan.layout
    result = fm.solve_fourier_modal_maxwell(
        prepared,
        fm.plane_wave_excitation(
            prepared.scattering, layout.mode_ids[layout.zero_index], polarization
        ),
    )
    far = fm.diffraction_order_far_field(prepared, result, side="left")
    power = np.asarray(jnp.sum(far.power[..., 0], axis=1))
    orders = np.asarray(layout.coefficients[:, 0])
    return {int(order): float(power[orders == order][0]) for order in (0, -1)}, result


def test_metallic_lamellar_grating_matches_published_inverse_rule_efficiencies() -> None:
    permittivity = (0.22 + 6.71j) ** 2
    errors: dict[str, list[float]] = {"tm": [], "te": []}
    for count in (21, 41, 81):
        tm, tm_result = _lamellar_reflection(count, permittivity, "tm")
        te, te_result = _lamellar_reflection(count, permittivity, "te")
        for result in (tm_result, te_result):
            assert bool(result.diagnostics.finite)
            assert bool(result.diagnostics.propagation_converged)
        np.testing.assert_allclose(
            tm[0], METALLIC_LAMELLAR_TM_ORDER_ZERO[count], atol=1e-5
        )
        np.testing.assert_allclose(
            te[-1], METALLIC_LAMELLAR_TE_ORDER_MINUS_ONE[count], atol=1e-5
        )
        errors["tm"].append(abs(tm[0] - METALLIC_LAMELLAR_EXACT["tm"]))
        errors["te"].append(abs(te[-1] - METALLIC_LAMELLAR_EXACT["te"]))
        assert tm[0] + tm[-1] < 1.0 and te[0] + te[-1] < 1.0
    for sequence in errors.values():
        assert sequence[0] > sequence[1] > sequence[2]


def test_perfect_conductor_limit_grating_conserves_energy() -> None:
    # Lossless ε = −10⁴ (the bulk perfect-conductor surrogate of Szczepkowicz et al.
    # 2020) has skin depth λ/630; every order below the ridges is evanescent.
    policy = fm.FourierModalSolvePolicy(boundary=fm.BoundaryCascadePolicy(doublings=16))
    for polarization in ("te", "tm"):
        efficiency, result = _lamellar_reflection(21, -1.0e4 + 0j, polarization, policy)
        assert int(result.status) == int(fm.FourierModalSolveStatus.SUCCESS)
        np.testing.assert_allclose(efficiency[0] + efficiency[-1], 1.0, atol=1e-9)
