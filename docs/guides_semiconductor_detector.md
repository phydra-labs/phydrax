# Semiconductor detector fields and prescribed response

The detector route composes the existing structured cochain electrostatic solver with
semiconductor-owned electrode, extrusion, evidence, and signal semantics. It covers a
fixed linear dielectric, prescribed volume charge, and prescribed point-charge
trajectories. It does not simulate carrier drift, diffusion, trapping, collection,
avalanche multiplication, or circuit loading.

## Separate physical fields

`SemiconductorDetectorPlan` binds a bounded `StructuredCochainBridge`, positive
physical permittivity, disjoint boundary-electrode masks, and a
`DetectorResourcePolicy`. One- and two-dimensional models must also provide the
physical transverse measure at every node and edge. A parallel-plate 1D reduction
uses its cross-sectional area. A radial coaxial reduction uses cylindrical
cross-sectional measures; geometry must not be hidden in a fitted permittivity.
Three-dimensional supports use unit transverse measures.

Two APIs deliberately produce incompatible types:

- `DetectorBiasElectrostaticPlan.solve(space_charge_density)` applies physical
  electrode voltages and prescribed volume charge. Its
  `DetectorBiasElectrostaticResult` reports electrode charges and total-charge
  closure.
- `DetectorWeightingFieldPlan.solve()` performs one zero-space-charge solve per
  electrode. Excitation `j` is one volt on electrode `j` and zero on every other
  physical boundary vertex. Its `DetectorWeightingFieldResult` contains the
  one-hot potential and electric-field basis plus the Maxwell capacitance matrix.

A bias result cannot be passed where a weighting result is required. The first
route has no carrier state and advances no carriers.

For weighting excitation `j`, the capacitance entry `C[i,j]` is the integrated
cochain boundary reaction on electrode `i`. The evidence retains the reciprocity
defect `max(abs(C - transpose(C)))` and the complete-electrode column-charge
defect. The sum of weighting potentials is certified as one only when the declared
electrodes exactly cover all Dirichlet boundary vertices. If any boundary vertices
are undeclared, they are an explicit grounded remainder: the fields remain
observable, but partition-of-unity and complete-electrode charge closure are not
certified.

```python
import jax.numpy as jnp
import numpy as np
from phydrax.applications import semiconductor as sc
from phydrax.discretization import (
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)

length = 1.0e-3
area = 1.0e-4
grid = TensorGridPlan(
    (UniformCellAxisSpec(32),), axis_names=("x",)
).prepare(jnp.asarray([[0.0], [length]]))
bridge = StructuredCochainBridge(grid)
x = np.asarray(bridge.cochain.coordinates[0])[:, 0]
resources = sc.DetectorResourcePolicy(
    maximum_nodes=33,
    maximum_edges=32,
    maximum_electrodes=2,
    maximum_linear_iterations=256,
    maximum_trajectory_cases=1,
    maximum_trajectory_samples=64,
    maximum_interpolation_routes=128,
)
detector = sc.SemiconductorDetectorPlan(
    bridge,
    (
        sc.DetectorElectrode("anode", np.isclose(x, x[0])),
        sc.DetectorElectrode("cathode", np.isclose(x, x[-1])),
    ),
    resources,
    permittivity=11.7 * sc.VACUUM_PERMITTIVITY_SI,
    node_transverse_measure=area,
    edge_transverse_measure=area,
    transverse_unit=sc.SQUARE_METER,
)
weighting_plan = sc.DetectorWeightingFieldPlan(detector)
weighting = weighting_plan.solve()
if not bool(weighting.evidence.certified):
    raise RuntimeError("weighting basis failed")
```

## Shockley–Ramo sign and trajectory contract

`PrescribedShockleyRamoPlan` consumes a certified complete-electrode weighting
result and generic `phydrax.dynamics.TrajectoryData`. `DetectorTrajectoryRoute`
explicitly selects flattened state components containing position and declares the
position and time units. No component name, carrier species, velocity, or reset
meaning is inferred.

For a fixed carrier charge `q` and weighting potential `phi_w`, the external
electrode charge convention is:

```text
Q_induced = -q * phi_w
I_into_electrode = dQ_induced / dt
```

The result uses the exact interval secant of induced charge. Therefore summing
`interval_current * dt` closes `endpoint_charge_change` without estimating a
velocity. A valid first-profile route has at least two samples, a contiguous valid
prefix, every corresponding transition valid, no active reset, finite fixed charge,
and positions inside the weighting grid. Padded samples are inert. Invalid routes
remain explicitly invalid and cannot produce a successful response.

```python
from phydrax.dynamics import StateLayout, TrajectoryData

layout = StateLayout((1,), component_names=("x",))
trajectory = TrajectoryData(
    jnp.linspace(0.0, 1.0e-6, 5),
    jnp.linspace(1.0e-6, length - 1.0e-6, 5)[:, None],
    state_layout=layout,
    source_id="prescribed-detector-path",
)
response = sc.PrescribedShockleyRamoPlan(
    weighting_plan,
    weighting,
    sc.DetectorTrajectoryRoute(layout, (0,)),
).evaluate(trajectory, -sc.ELEMENTARY_CHARGE_SI)
```

Evaluation does not modify or advance `trajectory`. The prescribed path remains the
caller’s artifact and its `dataset_id` is retained in the response.

## Lifecycle artifacts

`write_detector_artifact_archive` stores bias, weighting-field, or prescribed
Shockley–Ramo result arrays together with exact source/profile/unit provenance
and plan, trajectory, route, resource, and sign-convention identities.
`read_detector_artifact_archive` requires a matching caller-prepared result
structure; it does not serialize detector dynamics or providers. See the
[production-evidence guide](guides_condensed_matter_production_evidence.md).

## Resource and qualification boundaries

The resource policy rejects node, edge, electrode, linear-iteration, trajectory
case, sample, and multilinear-route overflow before the corresponding detector
allocation. These values are caller code policy, not released capacity claims.
Candidate and released support tuples must remain identical; maturity exists only
in qualification profiles.

Run the separated performance harness with:

```text
python benchmarks/cm_semiconductor_detector.py --output /tmp/detector-benchmark.json
```

Run the prospective detector and existing chain-transport campaigns with:

```text
python tools/cm_semiconductor_detector_qualification.py --output /tmp/detector-qualification.json
```

The detector campaign contains parallel-plate and coaxial calibrations, a segmented
locked case, and incomplete-electrode, invalid-route, and resource-overflow controls.
The same tool independently executes the existing scalar-chain Landauer, coherent AC,
finite-lead transient, and local optical-phonon Fock SCBA routes. Those profiles retain
their distinct physical boundaries: stationary Landauer has no displacement current;
coherent AC is connected-equilibrium Kubo plus capacitive Hartree and does not infer
ballistic DC from its adiabatic rate; finite-lead transients are valid only before the
reported recurrence window; and SCBA has no Hartree tadpole, numerical eta, or vertex
correction. Bound-state preparation and resource overflow remain locked refusals.

Campaign output is prospective evidence and never grants a release. No foundry
calibration, carrier-packet dynamics, trapping lifetime, avalanche gain, collection
efficiency, generic multiorbital transport, or detector-circuit response is claimed.
No candidate profile is emitted for an unimplemented detector execution path.
