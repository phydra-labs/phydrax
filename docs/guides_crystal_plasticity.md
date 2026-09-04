# Crystal plasticity

`phydrax.applications.crystal_plasticity` implements a three-dimensional,
finite-strain crystal-plasticity material and a block-exact finite-element route.
The supported envelope is deliberately narrow: static cell-block phase routing,
support-bound crystal orientations, multiplicative kinematics, power-law slip,
isotropic hardening, and accepted-step material transactions. Grain-map import,
DADF5, DREAM.3D, VTI, evolving texture, and mixed-phase quadrature are not part
of this contract.

## Point law

The deformation is split as $F=F_eF_p$. Each `CrystalSlipSystem` normalizes a
finite, nonzero direction and plane normal and requires them to be orthogonal.
Duplicate systems, including sign-equivalent systems, are rejected. An explicit
`crystal_to_sample` matrix rotates each crystal-frame Schmid tensor into the
sample frame. The matrix must belong to SO(3); an orthogonal reflection is not a
crystal orientation.

The compressible elastic energy is

$$
\Psi_e(F_e)=\frac{\mu}{2}
  \left(F_e:F_e-3-2\log J_e\right)
  +\frac{\kappa}{2}(\log J_e)^2,
\qquad J_e=\det F_e>0.
$$

The first-Piola stress is the derivative of this energy at fixed internal state.
Slip is driven by the Mandel stress, so a superposed spatial rotation preserves
the local root and rotates the first-Piola stress covariantly. The exponential
plastic update uses traceless Schmid tensors. Consequently, admissible updates
retain $\det F_p=1$ while reporting both $\det F_p$ and $\det F_e$.

For accumulated slip $\Gamma$, the stored hardening energy and current strength
are

$$
\Psi_h=\tfrac12 H\Gamma^2,
\qquad g=g_0+H\Gamma.
$$

The backward-Euler root uses the effective resolved shear
$\max(|\tau_\alpha|-H\Gamma,0)$ and the configured reference rate and rate
sensitivity. Each candidate reports plastic work and

$$
\mathcal D_{n\rightarrow n+1}
  =\sum_\alpha \tau_\alpha\,\Delta\gamma_\alpha
   -\left(\Psi_h^{n+1}-\Psi_h^n\right).
$$

Thermodynamic admissibility requires this incremental dissipation to be
nonnegative up to a dtype-scaled roundoff tolerance. Root convergence remains a
separate decision: `update.converged` describes the implicit solve,
`update.admissible` describes physical and increment bounds, and
`update.accepted` is their conjunction.

```python
import jax.numpy as jnp
import phydrax as phx

cp = phx.applications.crystal_plasticity
model = cp.CrystalPlasticityModel(
    (
        cp.CrystalSlipSystem(
            jnp.asarray((1.0, 0.0, 0.0)),
            jnp.asarray((0.0, 1.0, 0.0)),
        ),
    ),
    cp.CrystalPlasticityParameters(8.0, 20.0, 0.1, 1.0, 1.5, 1.0),
)
F = jnp.eye(3).at[0, 1].set(0.3)
Q = jnp.eye(3)  # crystal frame to sample frame
candidate = model.update(F, model.initial_state(), Q, 0.1)
```

`model.free_energy(F, state)` and `model.first_piola(F, state)` expose the fixed
internal-state energy/stress relation. Differentiating `update(...).first_piola`
uses the same `LocalImplicitMaterial` custom-root contract as the primal update;
the root is not recomputed through a second derivative path.

## Block-exact finite-element routing

`CrystalPlasticityRoute` binds a prepared three-dimensional vector field to one
entry per mesh cell block. Every entry is a tuple of block name, model, and
crystal-to-sample orientation. The orientation may be one `(3, 3)` matrix,
which is broadcast over the block, or an exact
`(cell_count, quadrature_count, 3, 3)` static texture field. Every site must
belong to SO(3). Entries must cover the blocks exactly: overlap, gaps, and
unknown block names are errors.

```python
route = cp.CrystalPlasticityRoute(
    discretization,
    "u",
    (
        ("phase-a", phase_a_model, phase_a_orientation),
        ("phase-b", phase_b_model, phase_b_orientation),
    ),
)
materials = route.initialize()
form = cp.cpfem_equilibrium_form(
    discretization,
    "u",
    route,
    materials,
    step_size,
)
```

Each block is phase homogeneous in its constitutive model while its static
orientation field may vary by cell and quadrature site. Different block routes
may use different slip counts; the transaction stores a tuple of route-local
arrays whose trailing width is `10 + model.slip_count`. States are never padded
or concatenated across phases. Residual and auxiliary evaluation use the same
block domain, model, site orientation, committed state, quadrature rule, and
point-update function.

`route.initialize()` returns the one authoritative `MaterialTransaction`.
Auxiliary evaluation returns one candidate transaction containing trials for
all routes. A global attempt may commit all trials or roll back all trials; no
route-local partial promotion is provided. A rejected local root or physical
bound requests cutback using the minimum suggested factor across every
quadrature point and route.

## Identity and restart boundaries

Model identity fingerprints normalized slip systems and material parameters.
Each block-route identity additionally binds the explicit orientation, prepared
support, field, block, state shape, and model. Transaction layout therefore
changes when a route-local slip count, model, orientation, support, or block
layout changes.

Use `route.checkpoint(materials)` and `route.restore(checkpoint)` for committed
material state. The checkpoint payload is bound to the route fingerprint and to
its committed content. A foreign route, changed orientation/model/support, or
incompatible state layout is rejected rather than interpreted or padded.

## Qualification boundary

Focused tests exercise initialization and packing, SO(3) rejection of
reflections, elastic and active-slip response, the incremental thermodynamic
inequality, the fixed-state energy/stress derivative, custom-root JVPs, frame
covariance, ragged two-block routing, global cutback and rollback, route
coverage, and checkpoint/layout mismatch. The runnable
`examples/crystal_plasticity_routed.py` and
`tools/crystal_plasticity_qualification.py` stay within that same envelope.
