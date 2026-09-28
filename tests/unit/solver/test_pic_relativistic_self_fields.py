#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Non-neutral bounded PIC initialization: electrostatic and relativistic self-fields.

References are independent of the runtime:

* Gauss: the certified relative Poisson residual bound ``tol·‖ρ_free‖⋆`` of the
  default solver in the Hodge norm (grounded walls carry zero potential, so the
  assembled right-hand side is the charge on free vertices), converted to a
  max-norm bound by ``h^{-3/2}``.
* Field: the rest-frame Coulomb field of a Gaussian bunch from the Green's
  function integral ``(2/√π)∫₀^∞ ds Π_i (1 + 2s²σ_i²)^{-1/2}
  exp(−x_i²s²/(1 + 2s²σ_i²))`` (NumPy quadrature), Lorentz-boosted to the lab:
  ``E⊥(x, y, z) = γ E'⊥(x, y, γz)`` with rest length ``γσ_z``, ``B = β × E/c``.
* Transient: the Poynting flux ``E × H`` through four planes around the beam,
  integrated over the run, for relativistic versus electrostatic
  initialization of the same beam.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_SIGMA = 0.08
_GAMMA = 10.0
_BETA = float(np.sqrt(1.0 - 1.0 / _GAMMA**2))
_TOLERANCE = 1.0e-10


def _gaussian_field(points: np.ndarray, sigma: tuple[float, float, float]) -> np.ndarray:
    """Coulomb field (ε = 1) of a unit Gaussian charge at ``points[..., 3]``."""
    t = np.linspace(0.0, 1.0, 8001)[1:-1]
    s, jacobian = t / (1.0 - t), 1.0 / (1.0 - t) ** 2
    widths = np.asarray(sigma)[:, None]
    coordinates = points[..., None]
    denominator = 1.0 + 2.0 * s**2 * widths**2
    kernel = np.prod(
        denominator**-0.5 * np.exp(-(coordinates**2) * s**2 / denominator), axis=-2
    )
    integrand = (2.0 * coordinates * s**2 / denominator) * kernel[..., None, :]
    return (
        (2.0 / np.sqrt(np.pi))
        / (4.0 * np.pi)
        * np.trapezoid(integrand * jacobian, t, axis=-1)
    )


def _species(count: int, specific_charge: float) -> tuple[Any, Any]:
    support = D.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    species = PIC.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PIC.PICChargeModelPlan(
            specific_charge,
            "beam",
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )
    charged = D.ChargedParticlePlan(
        jnp.sign(specific_charge) * jnp.ones((count,)), "beam"
    )
    return species, charged.prepare(support)


def _beam(cells: int, periodic_z: bool, **walls: Any) -> dict[str, Any]:
    """Non-neutral Gaussian electron bunch drifting along ``z`` at ``γ = 10``.

    Macroparticles sit on grid vertices with Gaussian weights (total charge
    −1), so the deposited charge is the vertex-sampled Gaussian.
    """
    h = 1.0 / cells
    axes = tuple(D.UniformCellAxisSpec(cells) for _ in range(2)) + (
        D.UniformCellAxisSpec(cells, periodic=periodic_z),
    )
    grid = D.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray([[-0.5] * 3, [0.5] * 3])
    )
    bridge = D.StructuredCochainBridge(grid)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),), **walls
    ).prepare()
    nodes = np.arange(-(cells // 2), cells // 2 + 1) * h
    z_nodes = nodes[:-1] if periodic_z else nodes
    x, y, z = np.meshgrid(nodes, nodes, z_nodes, indexing="ij")
    density = np.exp(-(x**2 + y**2 + z**2) / (2.0 * _SIGMA**2))
    wall = np.maximum(np.abs(x), np.abs(y)) < 0.5 - h
    keep = (density > 1.0e-6) & wall & (periodic_z | (np.abs(z) < 0.5 - h))
    positions = np.stack([x[keep], y[keep], z[keep]], axis=-1)
    weights = density[keep] / np.sum(density[keep])
    # A tiny specific charge keeps the bunch rigid; macrocharge = mass·q/m = −w.
    specific = -1.0e-6
    species, charged = _species(positions.shape[0], specific)
    permittivity = maxwell.constitutive.electric_displacement(
        jnp.ones((maxwell.layout.electric_count,)),
        maxwell.constitutive.initialize_state(),
    )
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge,
        phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(bridge),
        permittivity=permittivity,
        tolerance=_TOLERANCE,
    )
    transfer = PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell, electrostatic, (transfer,), (PIC.ChargeConservingCurrentPlan(transfer),)
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver, species=(species,), maximum_displacement_fraction=1.0
    )
    velocities = np.zeros_like(positions)
    velocities[:, 2] = _BETA
    return {
        "bridge": bridge,
        "maxwell": maxwell,
        "solver": solver,
        "pic": pic,
        "positions": jnp.asarray(positions),
        "velocities": jnp.asarray(velocities),
        "masses": jnp.asarray(weights / -specific),
        "h": h,
        "cells": cells,
    }


def _initialize(beam: dict[str, Any], mode: str, step: float) -> Any:
    return beam["pic"].initialize(
        (beam["positions"],),
        (beam["velocities"],),
        step,
        masses=(beam["masses"],),
        self_fields=mode,
    )


def _gauss_bound(beam: dict[str, Any], field: Any) -> float:
    """Max-norm Gauss bound implied by the certified Hodge-norm Poisson residual."""
    stars = np.asarray(beam["bridge"].cochain.hodge_stars[0])
    free = ~np.asarray(beam["solver"].electrostatic.boundary.dirichlet_mask)
    charge = np.where(free, np.asarray(field.primary.charge), 0.0)
    return _TOLERANCE * np.sqrt(np.sum(stars * charge**2)) / beam["h"] ** 1.5


def _axis_fields(beam: dict[str, Any], field: Any) -> tuple[np.ndarray, ...]:
    """Physical ``E_x`` and ``B_y`` on the ``x`` axis (``y = z = 0``) and positions."""
    bridge, h, cells = beam["bridge"], beam["h"], beam["cells"]
    centre = cells // 2
    electric = bridge.unpack(1, beam["maxwell"].electric_field(field))[0] / h
    # The (x, z) face orientation dx∧dz carries −B_y.
    flux = -bridge.unpack(2, field.primary.magnetic_flux)[1] / h**2
    x = (np.arange(cells) + 0.5) * h - 0.5
    near = np.abs(x) < 2.5 * _SIGMA
    along = np.asarray(electric)[near, centre, centre]
    # B_y faces straddle the z = 0 vertex plane.
    across = 0.5 * np.asarray(flux[near, centre, centre - 1] + flux[near, centre, centre])
    return x[near], along, across


@pytest.mark.parametrize(
    ("cells", "field_tolerance"), [(16, 0.05), (32, 0.015)], ids=["16", "32"]
)
def test_bounded_non_neutral_electrostatic_initialization_on_large_grids(
    cells: int, field_tolerance: float
) -> None:
    beam = _beam(cells, False, boundaries=(mx.MaxwellBoundaryPlan("pec"),))
    state = _initialize(beam, "electrostatic", 0.5 * float(beam["maxwell"].stable_dt))
    field = state.field
    gauss = float(jnp.max(jnp.abs(beam["maxwell"].electric_constraint(field))))
    assert gauss <= _gauss_bound(beam, field)
    x, along, _ = _axis_fields(beam, field)
    points = np.stack([x, 0.0 * x, 0.0 * x], axis=-1)
    reference = -_gaussian_field(points, (_SIGMA, _SIGMA, _SIGMA))[:, 0]
    error = np.max(np.abs(along - reference)) / np.max(np.abs(reference))
    assert error < field_tolerance, error
    assert float(jnp.max(jnp.abs(field.primary.magnetic_flux))) == 0.0


@pytest.mark.parametrize("permittivity", [1.0, 11.7 * 8.8541878188e-12])
@pytest.mark.parametrize("cells", [(8,), (12, 12, 12)], ids=["1d", "3d"])
def test_charge_free_dirichlet_driven_solve_reaches_the_exact_linear_profile(
    cells: tuple[int, ...], permittivity: float
) -> None:
    """``ρ = 0`` with the potential 1 − x on the walls: the solution is exactly linear.

    The discrete Laplacian of a linear function vanishes, so the converged
    potential is ``1 − x`` at every vertex; the tolerance is relative to the
    Dirichlet-driven right-hand side, so an SI permittivity does not loosen it.
    """
    dimension = len(cells)
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count) for count in cells),
        axis_names=("x", "y", "z")[:dimension],
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    bridge = D.StructuredCochainBridge(grid)
    vertices = np.meshgrid(
        *(np.linspace(0.0, 1.0, count + 1) for count in cells), indexing="ij"
    )
    exact = (1.0 - vertices[0]).reshape(-1)
    boundary = phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(
        bridge, jnp.asarray(exact)
    )
    plan = phx.solver.CochainElectrostaticPlan(
        bridge, boundary, permittivity=permittivity, tolerance=1.0e-9
    )
    result = plan.solve(jnp.zeros((bridge.cochain.cell_counts[0],)))
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(result.potential).reshape(-1), exact, rtol=0.0, atol=1.0e-8
    )


@pytest.mark.parametrize(
    ("cells", "field_tolerance", "magnetic_tolerance"),
    [(16, 0.05, 0.15), (32, 0.025, 0.05)],
    ids=["16", "32"],
)
def test_relativistic_self_fields_match_the_boosted_gaussian_bunch(
    cells: int, field_tolerance: float, magnetic_tolerance: float
) -> None:
    beam = _beam(cells, False, boundaries=(mx.MaxwellBoundaryPlan("pec"),))
    charge, ok = beam["solver"].deposit_charge(
        0,
        beam["positions"],
        -beam["masses"] * 1.0e-6,
        jnp.ones((beam["positions"].shape[0],), dtype=jnp.bool_),
    )
    assert bool(ok)
    result = beam["solver"].initialize_relativistic_field(
        (charge,), ((0.0, 0.0, _BETA),), 1.0
    )
    assert bool(result.successful)
    assert float(result.gauss_residual) <= _gauss_bound(beam, result.field)
    # B = d(βφ/c) is closed exactly.
    assert float(result.magnetic_divergence) < 1.0e-12
    state = _initialize(
        beam, "relativistic-per-species", 0.5 * float(beam["maxwell"].stable_dt)
    )
    x, along, across = _axis_fields(beam, state.field)
    points = np.stack([x, 0.0 * x, 0.0 * x], axis=-1)
    rest = _gaussian_field(points, (_SIGMA, _SIGMA, _GAMMA * _SIGMA))[:, 0]
    reference = -_GAMMA * rest
    error = np.max(np.abs(along - reference)) / np.max(np.abs(reference))
    assert error < field_tolerance, error
    # Transverse fields are γ-enhanced over the electrostatic field of the bunch.
    static = -_gaussian_field(points, (_SIGMA, _SIGMA, _SIGMA))[:, 0]
    assert np.max(np.abs(along)) > 1.4 * np.max(np.abs(static))
    magnetic = np.max(np.abs(across - _BETA * along)) / np.max(np.abs(_BETA * along))
    assert magnetic < magnetic_tolerance, magnetic


def _outward_flux(beam: dict[str, Any], field: Any, half_width: int) -> Array:
    """``∮ E × H · n dA`` through the planes ``x, y = ±half_width·h`` (all z)."""
    bridge, h, cells = beam["bridge"], beam["h"], beam["cells"]
    ex, ey, ez = (
        value / h for value in bridge.unpack(1, beam["maxwell"].electric_field(field))
    )
    bz, minus_by, bx = (
        value / h**2 for value in bridge.unpack(2, field.primary.magnetic_flux)
    )
    by = -minus_by
    lower, upper = cells // 2 - half_width, cells // 2 + half_width
    total = jnp.zeros(())
    for sign, index in ((-1.0, lower), (1.0, upper)):
        # Magnetic faces sit half a cell off the plane; average both sides.
        hz = 0.5 * (bz[index - 1] + bz[index])
        hy = 0.5 * (by[index - 1] + by[index])
        flux_x = jnp.sum(ey[index, lower:upper] * hz[lower:upper]) - jnp.sum(
            ez[index, lower : upper + 1] * hy[lower : upper + 1]
        )
        hx = 0.5 * (bx[:, index - 1] + bx[:, index])
        hz_y = 0.5 * (bz[:, index - 1] + bz[:, index])
        flux_y = jnp.sum(ez[lower : upper + 1, index] * hx[lower : upper + 1]) - jnp.sum(
            ex[lower:upper, index] * hz_y[lower:upper]
        )
        total = total + sign * (flux_x + flux_y) * h * h
    return total


def test_relativistic_initialization_emits_no_start_up_transient() -> None:
    beam = _beam(16, True, boundaries=(mx.MaxwellBoundaryPlan("pec"),))
    pic = beam["pic"]
    step = 0.5 * float(beam["maxwell"].stable_dt)
    # The start-up pulse crosses the planes at 4h before its PEC echo returns.
    steps = int(0.5 / step)

    @eqx.filter_jit
    def radiated(state: Any) -> Array:
        def body(current: Any, _: None) -> tuple[Any, Array]:
            result = pic.step_detailed(current, jnp.asarray(step))
            return result.accepted_state, _outward_flux(
                beam, result.accepted_state.field, 4
            )

        _, flux = jax.lax.scan(body, state, None, length=steps)
        return step * jnp.sum(flux)

    energies = {}
    for mode in ("relativistic-per-species", "electrostatic"):
        energies[mode] = float(radiated(_initialize(beam, mode, step)))
    # A static field with B = 0 around a γ = 10 beam radiates a pulse carrying
    # more energy than its whole initial field; the boosted field co-moves.
    assert energies["electrostatic"] > 0.1
    assert abs(energies["relativistic-per-species"]) < 0.03 * energies["electrostatic"]


def test_relativistic_self_fields_refusals() -> None:
    beam = _beam(8, False, boundaries=(mx.MaxwellBoundaryPlan("pec"),))
    solver = beam["solver"]
    charge = jnp.zeros((beam["bridge"].cochain.cell_counts[0],))
    with pytest.raises(ValueError, match="one grid axis"):
        solver.initialize_relativistic_field((charge,), ((0.3, 0.0, 0.6),), 1.0)
    with pytest.raises(ValueError, match="subluminal"):
        solver.initialize_relativistic_field((charge,), ((0.0, 0.0, 1.0),), 1.0)
    with pytest.raises(ValueError, match="magnetic must be None"):
        beam["pic"].initialize(
            (beam["positions"],),
            (beam["velocities"],),
            0.01,
            masses=(beam["masses"],),
            magnetic=jnp.zeros((beam["maxwell"].layout.magnetic_count,)),
            self_fields="relativistic-per-species",
        )
    with pytest.raises(ValueError, match="drifts require"):
        beam["pic"].initialize(
            (beam["positions"],),
            (beam["velocities"],),
            0.01,
            masses=(beam["masses"],),
            drifts=((0.0, 0.0, _BETA),),
        )
