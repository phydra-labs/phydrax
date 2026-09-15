"""Differentiable flat-periodic wave-dark-matter Schrödinger--Poisson rollout."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def build_workflow():
    cosmology = phx.applications.cosmology
    shape = (12, 12, 12)
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for count in shape),
        axis_names=("x", "y", "z"),
        field_name="psi",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in shape))
    background = cosmology.FLRWBackground(1.0, 1.0)
    policy = cosmology.WaveDarkMatterStepPolicy(
        maximum_phase_radians=2.0,
        minimum_de_broglie_cells=2.0,
        norm_relative_tolerance=1.0e-7,
    )
    prepared = cosmology.WaveDarkMatterPlan(
        1.0,
        jnp.linspace(0.5, 0.5005, 6),
        gravitational_constant=0.05,
        reduced_planck_constant=0.03,
        step_policy=policy,
    ).prepare(space, background)

    coordinates = jnp.meshgrid(
        *(axis.nodes for axis in space.axes),
        indexing="ij",
    )
    radius_squared = sum(
        jnp.minimum(jnp.abs(axis - 0.5), 1.0 - jnp.abs(axis - 0.5)) ** 2
        for axis in coordinates
    )
    psi = jnp.exp(-radius_squared / (2.0 * 0.16**2)).astype(jnp.complex128)
    norm = jnp.sum(space.quadrature_weights * jnp.abs(psi) ** 2)
    state = prepared.initialize(psi / jnp.sqrt(norm))
    return prepared, state


def main() -> None:
    prepared, state = build_workflow()
    result = eqx.filter_jit(prepared.solve)(state)
    global_phase_tangent = 0.01j * state.psi
    tangent = eqx.filter_jit(prepared.jvp)(state, global_phase_tangent)

    print("completed", bool(result.successful))
    print("accepted_steps", int(result.diagnostics.accepted_steps))
    print("maximum_norm_error", float(jnp.max(result.diagnostics.norm_relative_error)))
    print(
        "maximum_poisson_residual",
        float(jnp.max(result.diagnostics.poisson_relative_residual)),
    )
    print(
        "maximum_de_broglie_nyquist_fraction",
        float(jnp.max(result.diagnostics.de_broglie_nyquist_fraction)),
    )
    print("final_total_energy", float(result.diagnostics.total_energy[-1]))
    print("finite_jvp", bool(jnp.all(jnp.isfinite(tangent))))
    print("prepared_id", prepared.prepared_id)


if __name__ == "__main__":
    main()
