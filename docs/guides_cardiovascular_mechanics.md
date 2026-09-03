# Cardiovascular passive mechanics

The cardiovascular mechanics layer specializes PhydraX's existing finite-strain,
variational, mixed-FEM, load, optimization, and archive substrates. It does not
introduce another mesh, nonlinear solver, optimizer, or surface owner.

## Kernel units and configurations

Cardiovascular kernels use millimetres, milliseconds, milligrams, millivolts,
kilopascals, and cubic millimetres. Passive material stress and reference energy
per reference volume are therefore in kPa. Surface-foundation stiffness is in
kPa/mm when displacement is in mm. Cavity pressure is in kPa; pressure-volume
work is in kPa mm3 (one microjoule per kPa mm3).

Every passive law consumes a reference material frame with ordered columns
`(fiber, sheet, sheet-normal)`. The preferred source is anatomy's cellwise
`CardiacMaterialFrame`; select a cell explicitly with `cell_index`. A structural
3-by-3 array is also accepted when accompanied by an explicit `frame_id`.
Frames are checked for finite values, orthonormality, and positive determinant.
The frame is fixed in the reference configuration. A superposed current-frame
rotation therefore changes stress covariantly but does not change energy.

## Passive energy conventions

### Guccione 1991

`Guccione1991Energy` implements the named Green-Lagrange-strain convention
exactly:

```text
W = C/2 (exp(Q) - 1)
Q = bf Eff^2 + bt (Ess^2 + Enn^2 + 2 Esn^2)
    + 2 bfs (Efs^2 + Efn^2)
```

The material components are tensor shear components, not engineering shear.
`C` is in kPa and all `b` coefficients are dimensionless.

### Holzapfel-Ogden 2009 tension-only fiber/sheet convention

`HolzapfelOgden2009TensionOnlyEnergy` intentionally carries its complete
convention in its name. It is the eight-parameter fiber/sheet model:

```text
W = a/(2b) expm1(b (I1 - 3))
  + af/(2bf) expm1(bf <I4f - 1>+^2)
  + as/(2bs) expm1(bs <I4s - 1>+^2)
  + afs/(2bfs) expm1(bfs I8fs^2)
```

Fiber and sheet families are tension-only. The fiber-sheet invariant is signed
before squaring. This is not a no-gating variant and it does not add a separate
sheet-normal family.

For either energy, construct a fidelity route explicitly:

```python
energy = mechanics.Guccione1991Energy(
    mechanics.Guccione1991Parameters(0.9, 8.0, 2.0, 4.0),
    ventricular_microstructure.material_frame,
    cell_index=cell_index,
)
finite_bulk = energy.finite_bulk(80.0)
exact_mixed = energy.exact_incompressible()
```

These routes are different Python types, not string modes.

## Finite-bulk and exact-incompressible routes

`FiniteBulkCardiacMaterial` is displacement-only and uses

```text
Wtotal(F) = Wiso(J^(-1/3) F) + K g(F)^2 / 2.
```

The default constraint is `g(F) = J - 1`; the logarithmic
`VolumetricConstraint` may be selected explicitly. `evaluate(F)` returns
reference energy, first Piola stress, Cauchy stress, the exact material tangent,
and finite/orientation admissibility evidence. `cardiac_passive_functional` and
`cardiac_passive_form` lower the same pointwise energy to PhydraX's generic
variational and FEM substrates.

Finite bulk modulus is not exact incompressibility. Raising `K` in a low-order
displacement space can lock; this route makes no no-locking claim.

`ExactIncompressibleCardiacMaterial` uses the generic mixed potential and pressure
constraint. Its only FEM preparation path is
`MixedFiniteElementConstraintPlan`, which provides Taylor-Hood P2/P1 on simplex
cells or Q2/Q1 on tensor-product cells, requires an explicit mean-zero or pinned
pressure gauge, rejects unqualified stabilization, assembles the saddle blocks,
and computes inf-sup evidence. Call `prepare_qualified`; it fails closed unless
all of the following hold:

- the route and FE plan are exact mixed u-p;
- the pressure gauge is explicit;
- the space is P2/P1 Taylor-Hood or Q2/Q1;
- assembled inf-sup and adjoint-block evidence passes;
- the generic substrate reports locking-safe evidence.

There is deliberately no displacement P1 exact-incompressibility route.

## Basal, vascular, epicardial, and pericardial supports

The four named support types all expose a sign-consistent Robin surface energy.
For a declared reference direction `n`, relative displacement `r = u - u0`,
normal stiffness `kn`, and tangential stiffness `kt`,

```text
psi = kn (r.n)^2 / 2 + kt ||(I - n nT) r||^2 / 2
restoring traction = -d psi / d u.
```

Zero stiffness is exactly traction-free. A finite large stiffness is only a
Robin approximation; use the generic FEM essential-boundary-condition machinery
for an exact kinematic restraint. In particular, `PericardialSupport` is a
foundation model. It is not contact, does not detect a gap, and must not be
reported as a contact formulation. Use the contact application when separation,
impenetrability, or friction is required.

`cardiac_support_functional` and `cardiac_support_form` bind support energy to an
explicit reference-surface region.

## Oriented chamber volume and follower pressure

Anatomy owns closure, connectedness, fixed triangles, outward orientation,
reference coordinates, vertex IDs, and `surface_id` through
`OrientedChamberSurface`. Mechanics consumes that committed surface:

```python
volume_plan = mechanics.ChamberVolumePlan(oriented_surface)
pressure_plan = mechanics.FollowerPressurePlan(
    volume_plan,
    load_id="lv-cavity-pressure",
)
response = pressure_plan.evaluate(current_coordinates, pressure_kpa)
```

Anatomy's outward normal points from cavity fluid into myocardium. Its signed
volume is positive. Positive cavity pressure therefore has nodal force
`p dV/dx`, external virtual work `p dV`, and conservative pressure potential
`-p V`. `FollowerPressureResponse.force_tangent` is the exact current-geometry
follower tangent at fixed surface topology. `work_between` is exact for constant
pressure. For varying pressure, integrate the declared pressure-volume path;
do not apply the constant-pressure shortcut.

`MechanicsChamber` is executor-free. It owns no circulation DAE storage and
provides `volume`, `volume_rate`, and `pressure_response`. A circulation
`MechanicsChamberCoupling` may consume the `chamber_id` and `volume_rate`
callback while retaining exclusive mechanics storage ownership.

## Unloaded reference recovery

`UnloadedReferenceRecoveryPlan` performs inverse reference recovery against a
forward mechanics continuation path. Load factors must be finite, strictly
increasing, begin at zero, and end at one. The callback contract is

```python
def forward_path(reference_coordinates, load_factors, args):
    # Solve each station, then return explicit equilibrium evidence.
    return mechanics.ForwardContinuationResult(
        coordinates,                 # (num_load_factors, num_nodes, 3)
        equilibrium_residual_norm,   # (num_load_factors,)
        stage_successful,            # (num_load_factors,)
    )
```

The callback should run the prepared solid-mechanics equilibrium at each factor,
warm-starting each station from the preceding accepted solution. The inverse
coordinates are solved by PhydraX's native nonlinear least-squares optimizer.
The candidate records the complete fixed-shape continuation path and every
stage's solver status/residual. Commit is fail-closed on optimizer status,
finite path evidence, all stage statuses, equilibrium tolerance, zero-load
consistency, and final loaded-geometry residual. Pickle-free checkpoints retain
the exact prepared identity and reject shape or preparation changes.

## Qualification and performance

Run the focused scientific qualification artifact:

```text
python tools/cardiovascular_mechanics_qualification.py --output mechanics.json
```

It reports objectivity, energy-stress-tangent consistency, mixed gauge/inf-sup
and no-locking evidence, cavity derivative and pressure work, support limits,
and unloaded-reference recovery.

The benchmark separates batch energy evaluation, full point response including
tangent, and chamber follower-load execution:

```text
python benchmarks/cardiovascular_mechanics.py --points 128 1024 --repeats 5
```
