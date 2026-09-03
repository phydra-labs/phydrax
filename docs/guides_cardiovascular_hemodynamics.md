# Fixed-wall cardiovascular hemodynamics

PhydraX's cardiovascular hemodynamics workflow composes the existing certified D3Q19 lattice-Boltzmann discretization, TRT collision, staged open-boundary program, precision policy, and circulation DAE ports. It does not define another mesh, time integrator, checkpoint archive, or lumped circulation model.

The supported claim is deliberately narrow: a stationary voxel lumen with halfway bounce-back walls and weakly compressible athermal flow. The workflow does **not** claim moving-wall or fluid--structure interaction support, curved-wall accuracy, clinical validity, or suitability for clinical decisions.

## Units, axes, signs, and references

The cardiovascular kernel uses the following hemodynamics units:

| Quantity | Kernel unit | Exact SI factor |
| --- | --- | --- |
| length | mm | 0.001 m |
| time | ms | 0.001 s |
| density | mg/mm3 | 1000 kg/m3 |
| velocity | mm/ms | 1 m/s |
| pressure | kPa | 1000 Pa |
| flow | mm3/ms | 0.000001 m3/s |
| dynamic viscosity | kPa ms | 1 Pa s |
| shear rate | 1/ms | 1000 1/s |
| power | mg mm2/ms3 | 0.001 W |

`HemodynamicsScaling` binds one physical cell size, LBM time step, and reference density. Its pressure scale is

```text
p_scale = rho_reference * (cell_size / time_step)^2
```

because `mg/(mm ms2)` is numerically `kPa`. Its volume-flow scale is `cell_size^3 / time_step`. Velocity, kinematic viscosity, density, gauge pressure, volume flow, shear rate, mass, momentum, and power all have explicit forward and inverse conversion methods. The constructor refuses a reference velocity whose lattice Mach number exceeds the declared weak-compressibility envelope.

The tensor grid's named `x`, `y`, and `z` axes define the spatial frame. Every terminal is an exterior lower or upper face. The geometric outward normal points out of the 3D lumen. `TerminalDirection.INTO_LUMEN` declares positive circulation flow into the 3D domain; `TerminalDirection.OUT_OF_LUMEN` declares it out of the domain. Thus positive directed flow maps to negative outward flux at an inlet and positive outward flux at an outlet.

Pressure is a gauge value relative to each terminal's explicit `pressure_reference_kpa`. Pressure measurements return the area-weighted boundary pressure plus that reference. Flow measurements integrate cell velocity against the outward normal and the fixed cell-face areas.

## Plan and prepare a fixed topology

Prepare a native uniform tensor grid and D3Q19 discretization first. The physical grid spacing must equal the `HemodynamicsScaling.cell_size_mm` value.

```python
import jax.numpy as jnp
import numpy as np
import phydrax as phx

from phydrax.applications.cardiovascular import hemodynamics
from phydrax.applications.cardiovascular import circulation

shape = (48, 24, 24)
grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(n) for n in shape),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray(((0.0, 0.0, 0.0), (24.0, 12.0, 12.0))))

discretization = phx.discretization.LatticeBoltzmannPlan(
    grid,
    phx.discretization.D3Q19(),
).prepare()

scaling = hemodynamics.HemodynamicsScaling(
    0.5,
    0.05,
    1.06,
    reference_velocity_mm_per_ms=0.2,
    maximum_lattice_mach=0.1,
)
```

`FixedWallLumenRegion` is an immutable Boolean cell classification. `True` means fluid. Solid neighbors are compiled once into stationary halfway bounce-back links; the mask cannot change during a run. This is a voxel boundary rule, not a curved-boundary reconstruction claim.

Terminals bind to the actual pressure/flow ports of circulation-owned components. The binding stores identity only: all 0D pressure, flow, resistance, compliance, inertance, and volume state remains in the circulation DAE.

```python
lumen_mask = np.ones(shape, dtype=bool)
lumen = hemodynamics.FixedWallLumenRegion(lumen_mask, lumen_name="example_lumen")

terminal_load = circulation.Resistance("terminal_load", 0.8)
inlet = hemodynamics.FlowTerminalPort(
    "inlet",
    hemodynamics.TerminalFace(
        "x", "lower", hemodynamics.TerminalDirection.INTO_LUMEN
    ),
    hemodynamics.CirculationPortBinding(terminal_load, "inlet"),
)
outlet = hemodynamics.PressureTerminalPort(
    "outlet",
    hemodynamics.TerminalFace(
        "x", "upper", hemodynamics.TerminalDirection.OUT_OF_LUMEN
    ),
    hemodynamics.CirculationPortBinding(terminal_load, "outlet"),
)

rheology = hemodynamics.NewtonianRheology(
    0.004,
    maximum_shear_rate_per_ms=2.0,
)
plan = hemodynamics.FixedWallLBMPlan(
    discretization,
    scaling,
    lumen,
    (inlet, outlet),
    rheology,
)
prepared = plan.prepare()
state = prepared.initialize_state()
```

Preparation refuses any lattice other than the certified D3Q19 set, a grid/lumen shape mismatch, a scaling/grid-spacing mismatch, duplicate terminal face or circulation bindings, periodic terminal faces, an empty terminal region, a pressure-controlled inflow (the native pressure boundary does not enforce inflow direction), a reference Mach envelope above the numerical limits, or rheology/scaling combinations whose relaxation rates leave the declared subinterval of `(0, 2)`.

## Circulation coupling and fail-closed commits

`TerminalPortValues` is a fixed-order runtime record containing both pressure and directed flow from the circulation solve. The order is exactly the terminal order in `FixedWallLBMPlan`. A pressure-controlled terminal consumes its pressure; a flow-controlled terminal consumes its flow. Both values remain available for interface evidence.

```python
port_values = hemodynamics.TerminalPortValues(
    pressure_kpa=jnp.asarray((12.0, 10.0)),
    directed_flow_mm3_per_ms=jnp.asarray((0.12, 0.12)),
)

candidate = prepared.candidate(state, port_values)
state = prepared.commit(state, candidate)
```

`candidate` performs a local-rheology TRT collision, applies the compiled stationary walls and open boundaries, reconstructs physical fields, measures all terminal p/Q pairs, and constructs `HemodynamicsEvidence`. It does not mutate the accepted state. Nonfinite, nonpositive-pressure-density, and super-lattice-speed terminal iterates are replaced only for the native boundary evaluation so the boundary kernel cannot raise before evidence is produced; the original values remain in the audit and force rejection. `commit` advances the state only when every mandatory check passes; otherwise it returns the previous populations, time, step index, boundary history, mass, momentum, and terminal accumulators. This allows a coupled nonlinear driver to revise the 0D/3D interface iterate without corrupting accepted history.

Evidence includes:

- finite populations, density, velocity, relaxation rates, and terminal values;
- maximum lattice Mach number;
- control-volume storage plus outward terminal-volume defect;
- relative total-momentum change;
- minimum population and density plus maximum density deviation;
- cellwise rheology and relaxation-rate admissibility;
- per-terminal circulation/3D pressure residuals and tolerances, terminal flow and hydraulic-power defects; and
- the fixed-wall scope identity and wall impulse ledger.

Status is fail-closed. `HemodynamicsStatus.SUCCESS` is emitted only if every check and the native collision succeeds. `FixedWallLBMCheckpoint` carries the immutable prepared and boundary-topology IDs; restore refuses a checkpoint from another preparation.

## Rheology

`NewtonianRheology` returns constant dynamic viscosity. `CarreauYasudaRheology` evaluates

```text
mu(gamma) = mu_inf + (mu_0 - mu_inf)
            * (1 + (lambda * gamma)^a)^((n - 1) / a)
```

with `mu` in kPa ms, `gamma` in 1/ms, and `lambda` in ms. The constructor enforces `mu_0 >= mu_inf > 0`, `lambda > 0`, `a > 0`, and `0 < n <= 1`. Equal zero- and infinite-shear viscosities recover the Newtonian law exactly. `evaluate` returns cellwise validity evidence for candidate auditing; `dynamic_viscosity` refuses a direct query outside the finite nonnegative declared shear-rate envelope. No clipping or uncertified extrapolation is reported as valid.

The prepared workflow computes the symmetric velocity-gradient invariant `sqrt(2 D:D)` on the immutable fluid mask. It uses centered differences where both neighbors are fluid and one-sided differences beside a voxel wall. The resulting local dynamic viscosity is divided by reference density and converted to a cellwise TRT relaxation rate.

## Terminal measurements and balances

`PreparedTerminalMeasurements` owns paired `PressureMeasurementDefinition` and `FlowMeasurementDefinition` objects for every immutable terminal region. The definitions record fixed area weights, outward normals, sign conventions, pressure references, circulation port IDs, and cardiovascular quantity-spec IDs.

For a measurement snapshot:

```python
macroscopic = prepared.macroscopic_state(state)
measurements = prepared.terminal_measurements.measure(
    macroscopic.gauge_pressure_kpa,
    macroscopic.velocity_mm_per_ms,
)
```

`terminal_balance_evidence` checks four distinct contracts:

1. measured pressure equals circulation-owned pressure within an explicit tolerance at every terminal;
2. measured directed flow equals the circulation-owned terminal flow;
3. storage-volume change plus integrated outward volume is zero; and
4. measured boundary power `-sum(p * Q_outward)` equals the power implied by circulation p/Q values.

Do not hide a terminal mismatch by loosening only the global mass tolerance. Per-terminal pressure, flow, volume, and power have independent thresholds.

## Poiseuille, Womersley, and LBM/MAC qualification

`PoiseuillePipeReference` provides the circular-pipe axial profile, centerline velocity, and exact flow rate in kernel units. `WomersleyPipeReference` provides the harmonic circular-pipe solution with an explicit `exp(+i omega t)` convention and reports the Womersley number. The deterministic complex Bessel reference is intended for qualification, not the fixed-step cell kernel.

`compare_lbm_mac` accepts co-registered physical velocity and pressure fields plus sample weights. It returns relative L2 and maximum-absolute discrepancies. The evidence records different route IDs for fixed-wall D3Q19 LBM and staggered MAC; it never turns the route into a string-selected solver mode or treats one result as the other.

Run the deterministic qualification report with:

```text
python tools/cardiovascular_hemodynamics_qualification.py --output hemodynamics.json
```

The report covers physical/lattice roundtrips, the Carreau--Yasuda Newtonian limit, Poiseuille flow integration, the low-Womersley quasi-steady limit, terminal outlet volume and power, a fixed-wall candidate/commit, and a native D3Q19/MAC comparison evolved from rest under the same body acceleration and time horizon. A failed case gives a nonzero exit status.

For candidate-kernel timing of both rheology types:

```text
python benchmarks/cardiovascular_hemodynamics.py \
  --shape 32 24 24 --repeats 20 --output hemodynamics_benchmark.json
```

Benchmark output reports population-state bytes, compile-and-first-call time, steady execution time, cell updates per second, Mach evidence, relaxation-rate extrema, and success. It is a numerical throughput measurement for the fixed-wall D3Q19 candidate path only; it is not evidence of physiological or clinical validity.
