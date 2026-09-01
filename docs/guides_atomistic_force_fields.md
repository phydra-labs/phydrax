# Atomistic force fields

PhydraX represents a classical force field as an immutable `AtomisticForceFieldPlan`.
Preparation validates every term against one prepared atomistic system and returns one
`PreparedAtomisticForceField` with a stable identity and an auditable preparation report.

## Coordinate domains

Dynamics state stores physical degrees of freedom. `AtomisticCoordinateMapPlan` realizes
interaction sites for an evaluation. Identity maps preserve ordinary atomistic systems;
weighted and local-frame virtual sites derive extra coordinates without adding masses or
momenta. Forces on derived sites return to degrees of freedom through the coordinate-map
pullback, so force conservation and differentiation use one path.

## Terms and nonbonded policy

A bundle may combine bonded terms, torsion series, CMAP, pair overrides, Morse,
Buckingham, tabulated pairs, reaction-field electrostatics, and dispersion treatment.
`AtomisticNonbondedPolicy` is the single source of truth for exclusions, 1-4 scaling,
cutoffs, switching, combining rules, and reciprocal-space choices. Do not duplicate those
rules in individual potentials.

```python
force_field = phx.atomistic.AtomisticForceFieldPlan(
    terms=(
        phx.atomistic.MorsePotential(depth, width, equilibrium, cutoff),
        phx.atomistic.BuckinghamPotential(a, b, c, cutoff),
    ),
    provenance=phx.atomistic.AtomisticForceFieldProvenance(
        source="project-parameters",
        source_version="2026.1",
    ),
)
prepared = force_field.prepare(system)
```

Preparation is fail-closed: unsupported term kinds, mismatched capacities, invalid tables,
and incompatible policies raise before compilation. Runtime evaluation reports a success
flag in addition to energy and forces.

## Rigid water

`SETTLEPlan` provides an analytic three-site water position projection and momentum tangent
projection. It is appropriate only for the geometry encoded by the plan; use general
distance constraints for other molecules.

## Interchange

Use `phydrax.atomistic.interchange` for OpenMM, OpenFF Interchange, and ParmEd boundaries.
Adapters return an `AtomisticInterchangeReport`; persist it with force-field provenance so
unsupported or approximated content cannot be hidden.
