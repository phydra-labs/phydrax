# Member-network structural verification

The member-network application adds constitutive reference states, cable
unilateral behavior, bending, buckling, construction stages, sizing, and
prestress realizability to force-density geometry. It does not change the
scientific meaning of force-density form-finding: a force-density result remains
geometric axial equilibrium evidence and becomes an input target for separate
structural verification.

## Architecture

A `MemberNetworkDefinition` combines:

- `ForceDensityStructure` topology, external IDs, masks, and restraints;
- `MemberReferenceState` stress-free lengths, rotations, curvature, twist, and
  installation metadata;
- `MemberPropertyMap` materials, sections, fabrication groups, and actuator
  groups;
- `MemberDOFLayout` translation and rotational coordinates.

A `MemberNetworkAssembly` is a static tuple of homogeneous mechanics blocks:

- `AxialMemberBlock`;
- `CorotationalFrameBlock`;
- `DiscreteRodBlock`;
- `HingeBendingBlock`.

Each block returns strain energy, resultants, validity, and unilateral evidence.
The prepared nonlinear equilibrium residual is the gradient of total strain
energy minus declared nodal forces and moments. Followers or other
nonconservative member loads must enter through an explicit residual extension;
they may not be inserted into a conservative energy.

## Materials and sections

`LinearElasticMaterial` carries independent stiffness, density, and optional
strength evidence. No strength is inferred from Young's modulus.

`AxialSection` carries area. `BeamSection` carries area, two principal second
moments, torsion constant, shear areas, and optional warping constant. Continuous
sizing should prefer physical section families:

- `RectangularSectionFamily`;
- `CircularSectionFamily`;
- `TubeSectionFamily`.

This keeps area and stiffness properties geometrically coupled. `SectionCatalog`
represents correlated finite candidates with stable external labels.

## Prepared equilibrium

```python
plan = mn.plan_member_network(problem, inputs, initial_kinematics)
prepared = mn.prepare_member_network(plan, inputs, initial_kinematics)
result = mn.solve_member_network(prepared)

changed = mn.refresh_member_network(prepared, changed_inputs, result.state.kinematics)
changed_result = mn.solve_member_network(changed)
```

The nonlinear root uses fixed state and residual spaces, numerical refresh, and
mathematical implicit derivatives. Result status requires the numerical solve,
constitutive geometry, cable active-set evidence, and physical equilibrium
residual to agree.

## Cable slackness

`TensionOnlyCableLaw` uses exact positive-part elastic energy. The active-set
wrapper solves smooth fixed-mask roots, reclassifies members by stress-free
extension, and repeats until the unilateral set closes.

`CableSlacknessResult` records:

- active and slack members;
- activation/deactivation changes;
- switching margins;
- complementarity residual;
- ambiguous members;
- derivative mode.

Implicit sensitivities are physically unique only when every tension-only member
is separated from the switching surface. An ambiguous active set is reported,
not smoothed silently.

A straight two-node cable does not model subspan catenary sag. Discretize the
cable or use a separately qualified catenary element when available.

## Bending

`CorotationalFrameBlock` supports objective 2-D and 3-D frame energy. It uses
large-motion chord and section frames with small local Timoshenko strains,
including axial, shear, bending, and torsional resultants. Section directors or
nodal reference rotations are required in 3-D; branching frames are not assigned
an arbitrary orientation.

`DiscreteRodBlock` supports ordered chains with stretching, discrete curvature,
and nodal twist energy. Antiparallel or degenerate adjacent edges are outside the
valid geometry domain.

`HingeBendingBlock` adds explicit rest-dihedral energy to triangulated surfaces.
A fairness or planarity objective remains a design preference, not bending
stiffness.

## Buckling

Three evidence levels remain distinct:

1. `local_euler_buckling` uses explicit member effective-length factors.
2. `linearized_buckling` solves a generalized conservative proportional-load
   eigenproblem.
3. `member_network_continuation_problem` connects nonlinear equilibrium to
   pseudo-arclength, fold localization, branch-point certification, and branch
   switching.

`tangent_stability` reports the current conservative tangent spectrum. A linear
buckling factor is not a nonlinear collapse load. Follower-load tangents may be
nonsymmetric and are rejected from the self-adjoint eigen route.

## Prestress realizability

`member_network_from_force_density` maps a force-density target into a
constitutive reference state. Under linear engineering strain,

```text
N  = EA (L - L0) / L0
L0 = EA L / (EA + N)
```

`assess_prestress_realizability` checks:

- positive, finite stress-free lengths;
- constitutive target-force reconstruction;
- tension/compression member roles;
- equilibrium;
- fabrication length bounds and shared groups;
- actuator stroke bounds;
- required stability evidence;
- required construction-sequence evidence.

The verdict is `CERTIFIED`, `FAILED`, or `INCOMPLETE`. Missing stability or
sequence information cannot become success.

## Construction sequence

A `ConstructionSequencePlan` is a static tuple of independently prepared stages.
Each stage declares a complete topology/support problem, loads, installation
rules, and actuator operations. Stable node and member IDs transfer state between
different stage topologies.

Installation rules include:

- declared stress-free length;
- stress-free installation at the current geometry;
- declared initial strain;
- actuator-controlled installation.

Load operations include replace, add, remove, and ramp. Jack force, jack stroke,
rest-length change, release, and lock-off are distinct actuator operations.

Fixed sequences compose differentiable continuous stage maps. Stage order and
activation membership are discrete; bounded candidate tuples can be enumerated,
but gradients do not cross sequence changes.

## Material sizing

`evaluate_member_sizing` reports mass, cost, carbon, axial stress, tension and
compression utilization, displacements, local buckling utilization, and the
governing member.

`ContinuousMemberSizingProblem` accepts a design evaluator that performs the
required member-network analyses and exposes ordinary PhydraX objectives and
constraints. `select_catalog_member_sizing` performs deterministic bounded
catalog enumeration.

Sizing must cover every declared service case and construction stage. Rounding a
continuous section does not create a discrete optimum.

## Unified evidence

`verify_member_structure` aggregates explicitly required blocks:

- equilibrium;
- prestress realizability;
- construction sequence;
- sizing;
- local buckling.

It returns:

```text
CERTIFIED
FAILED
INCOMPLETE
```

Missing required evidence always produces `INCOMPLETE`.

## Scientific limitations

The core does not infer effective-length factors, design-code safety factors,
lateral-torsional buckling, plate local buckling, catenary behavior, construction
stage order, or 3-D section orientation. Those require explicit physical inputs
or separately qualified models.

Runnable examples:

- `examples/member_network_cable_prestress.py`;
- `examples/member_network_frame_buckling.py`;
- `examples/member_network_construction_sizing.py`.
