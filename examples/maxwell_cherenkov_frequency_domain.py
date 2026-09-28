#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cherenkov radiation of a moving line charge, solved in the frequency domain.

A line charge crosses a commensurate periodic cell (``ωL/v = 2π``) of glass
(``ε = 4``) at ``v = 0.8c`` (above threshold, ``βn = 1.6``) and of vacuum (below
threshold). The exact Whitney load of ``λ x̂ δ(y) exp(iωx/v)`` drives the cochain
curl-curl operator with stretched-coordinate absorbers; the sparse-direct solve is
compared with the analytic uniform-motion field and with the 2-D Cherenkov energy
``(λ²/2π) η √(1 − 1/(β²n²))`` per unit length. Code units ε₀ = μ₀ = c = 1.
"""

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    SparseLU,
)


mx = phx.solver.maxwell
omega, speed = 2.0 * np.pi, 0.8
length = 2.0 * np.pi * speed / omega
cells, transverse = 32, 144
step = length / cells
grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(cells, periodic=True),
        phx.discretization.UniformCellAxisSpec(transverse),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray([[0.0, -transverse * step / 2], [length, transverse * step / 2]]))
bridge = phx.discretization.StructuredCochainBridge(grid)
layout = mx.MaxwellCochainLayout(bridge, "tez")
origin = [0.3 * step, 0.5 * step]
source = mx.MaxwellMovingChargePlan(
    bridge, layout, charge=1.0, speed=speed, origin=origin, direction=[1.0, 0.0]
)
policy = LinearSolvePolicy(
    SparseLU(provider="scipy-superlu"),
    differentiation=DifferentiationPolicy("none"),
    failure=FailurePolicy("status"),
)
centers = (np.arange(cells) + 0.5) * step
row = transverse // 2 + 12
probe = np.stack((centers, np.full(cells, -transverse * step / 2 + row * step)), axis=1)

for permittivity in (4.0, 1.0):
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity).prepare(
        bridge.cochain, layout
    )
    result = (
        mx.FrequencyMovingChargePlan(
            source,
            material,
            stretching=mx.MaxwellCPMLPlan((0, 16), target_reflection=1e-6),
            policy=policy,
        )
        .prepare()
        .solve(omega)
    )
    evidence = result.evidence
    energy = 2.0 / np.pi * float(result.ledger.source_power) / length
    beta_n = speed * np.sqrt(permittivity)
    reference = (
        np.sqrt(1.0 / permittivity) * np.sqrt(1.0 - 1.0 / beta_n**2) / (2.0 * np.pi)
        if beta_n > 1.0
        else 0.0
    )
    analytic = phx.electromagnetics.UniformMotionFieldPlan(
        "line",
        phx.electromagnetics.UniformMotionMedium([omega], permittivity, 1.0),
        charge=1.0,
        speed=speed,
        origin=origin,
        direction=[1.0, 0.0],
    )
    along, _ = bridge.unpack(1, result.electric)
    exact = analytic.evaluate(jnp.asarray(probe)).electric[0, :, 0] * step
    error = float(jnp.linalg.norm(along[:, row] - exact) / jnp.linalg.norm(exact))
    print(f"ε = {permittivity}: radiating = {bool(evidence.radiating)}")
    print(f"  d²W/(dω dl) solved   = {energy:.6f}")
    print(f"  d²W/(dω dl) analytic = {reference:.6f}")
    print(f"  field error at 12 cells from the path = {error:.4f}")
    print(f"  continuity defect = {float(evidence.continuity_defect):.2e}")
    print(f"  bound-field reach γβλ = {float(evidence.bound_extent):.4f}")
