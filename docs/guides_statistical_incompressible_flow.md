# Statistical periodic incompressible flow

This workflow evolves a finite, coupled sample of periodic incompressible initial
conditions with spectral vanishing viscosity (SVV), then records increment
moments, cross-resolution distances, and leakage-safe operator-learning data.
It is **finite-campaign evidence**, not a proof of convergence, compactness,
uniqueness, or existence of a statistical Euler solution.

The mathematical motivation is [Lanthaler–Mishra, rough-data spectral viscosity](https://arxiv.org/abs/1903.12361),
[Lanthaler–Mishra–Parés-Pulido, statistical solutions](https://arxiv.org/abs/1909.06615),
and [Rohner–Mishra, large-scale statistical solutions](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2024/2024-04_rev1.pdf).
The implemented contracts, rather than an external solver configuration, define
this recipe. See the [incompressible API](api/solver/incompressible.md),
[spectral statistics API](api/discretization/spectral.md), and
[ETDRK API](api/solver/etdrk.md).

## Scientific and execution identities

Prepare one initial measure at the finest generation resolution, realize a root
key and explicit member indices, and pass that state to
`restrict_periodic_flow_initial_state(state, coarse_discretization)` for coarse
initial states. Independent preparation at each resolution is not the same coupling:
the scientific generation discretization participates in the measure identity.
Restriction gathers the retained band and zeros target-forbidden modes without
reapplying Leray projection. Member identities and root/coupling identities
survive restriction; removed-energy evidence records the discarded tail.

A power-law velocity measure must generate a solenoidal vector realization,
not reuse an identical scalar sample for every component. Gaussian coefficients
with exponent `H + d/2` are the fractional-Gaussian specialization; uniform
coefficients are not Gaussian. The example below uses a deliberately small,
smooth, truncated random measure for execution, not a rough-data benchmark.

SVV is a diagonal linear decay integrated by ETDRK. For a retained integer
resolution `K`, its scale is `strength * K**(1 - 2*order)` and its cutoff is
`cutoff_scale * K**cutoff_exponent`. The `tadmor_step` operator-scaling condition
requires `order >= 1` and strictly
`0 < cutoff_exponent < (2*order - 1)/(2*order)`. Even when that condition holds,
`cited_convergence_supported` is false: operator scaling does not establish the
rough-data initialization/mollification hypotheses or the full convergence
theorem. `exponential_highpass` is practical high-pass hyperviscosity, not the
exact zero-low-mode Tadmor profile. SVV and LES closures are refused together.

The Courant guard supports ETDRK4 only. Its declared bound uses the native-grid
frequency `max_x sum_a abs(u_a(x))/dx_a` and the frozen uniform-advection RK4
reference ceiling `2*sqrt(2)/pi`. An oversized or nonfinite step is refused with
failure evidence and unchanged state. This is not nonlinear stability
certification: inspect path validity and failure status rather than treating a
returned state as a successful integration.

## Small executable campaign

All parameters below are explicitly chosen demonstration policies. Six members,
two short steps, bounded chunks, and small Fourier grids keep this recipe small;
they do not constitute statistical or spatial convergence evidence.

```python executable
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax.applications import incompressible_flow as flow
from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    PaddingDealiasingPlan,
    PeriodicRadialIncrementMomentPlan,
    PseudospectralMethodPlan,
    SpectralVanishingViscosityPlan,
    TensorSpectralPlan,
)
from phydrax.equations import (
    IncompressibleFlowProblem,
    compile_periodic_incompressible_flow,
)
from phydrax.nn.operator.training import split_operator_dataset
from phydrax.solver import (
    CourantGuardedETDRKMethod,
    ETDRKMethod,
)
from phydrax.stochastic import (
    PeriodicPowerLawFieldPlan,
    PowerLawSpectralAmplitude,
)

domains = (AxisDomain.periodic(0.0, 1.0),) * 2
coarse_space = TensorSpectralPlan(
    (FourierBasisPlan(16), FourierBasisPlan(16))
).prepare(domains)
fine_space = TensorSpectralPlan(
    (FourierBasisPlan(32), FourierBasisPlan(32))
).prepare(domains)

measure = flow.PeriodicFlowInitialMeasure(
    flow.PowerLawVelocityFamily(
        PeriodicPowerLawFieldPlan(
            PowerLawSpectralAmplitude(
                1.5, scale=0.02, maximum_wavenumber=20.0
            ),
            component_shape=(2,),
            solenoidal=True,
        )
    )
).prepare(fine_space, member_chunk_size=2, maximum_bytes=2**26)
fine_initial = measure.realize(
    jax.random.key(41), np.arange(6, dtype=np.uint32)
)
coarse_initial = flow.restrict_periodic_flow_initial_state(fine_initial, coarse_space)

svv = SpectralVanishingViscosityPlan(
    strength=0.01,
    order=1.0,
    cutoff_scale=1.0,
    cutoff_exponent=0.25,
    kernel="tadmor_step",
)
method = CourantGuardedETDRKMethod(
    ETDRKMethod(4), maximum_courant_number=0.5
)
radii = jnp.asarray([0.125, 0.18, 0.25, 0.35], dtype=jnp.float64)
results = []
moments = []
wrappers = []
for space, initial in (
    (coarse_space, coarse_initial),
    (fine_space, fine_initial),
):
    dynamics = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(2, 0.0),
        space,
        PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2)),
        spectral_viscosity=svv,
    )
    prepared = flow.prepare_periodic_flow_ensemble(
        dynamics,
        method,
        initial,
        start_time=0.0,
        end_time=2e-4,
        step_size=1e-4,
        member_chunk_size=2,
        maximum_output_bytes=2**27,
    )
    bound_result = prepared.execute()
    result = bound_result.result
    assert bool(jnp.all(result.valid))
    radial = PeriodicRadialIncrementMomentPlan(
        space,
        radii,
        averaging="ball",
        component_shape=(2,),
        member_chunk_size=2,
        maximum_kernel_bytes=2**20,
    )(result.states[:, -1])
    assert bool(jnp.all(radial.finite))
    results.append(result)
    moments.append(radial)
    wrappers.append(bound_result)

pair = flow.align_coupled_ensembles(wrappers[0], wrappers[1])
componentwise = flow.componentwise_empirical_wasserstein(
    pair, p=1, node_chunk_size=32
)
upper = flow.paired_coupling_upper_bound(pair, p=1, node_chunk_size=32)
sliced = flow.sliced_joint_marginal_distance(
    pair, ((0, 1), (16, 17)), key=jax.random.key(42),
    num_projections=8, p=1, member_chunk_size=2,
    maximum_workspace_bytes=2**24,
)
assert bool(jnp.isfinite(componentwise.value))
assert bool(jnp.isfinite(upper.value))
assert bool(jnp.all(jnp.isfinite(sliced.values)))

scaling = flow.StructureFunctionScalingPlan((0.125, 0.35)).evaluate(
    tuple(moments), (coarse_space, fine_space)
)
assert all(fit.status in ("complete", "abstained") for fit in scaling.fits)
print([(fit.status, fit.reason) for fit in scaling.fits])

dataset = flow.periodic_flow_operator_dataset(
    wrappers[1],
    target_time_indices=(1, 2),
    output_discretization=coarse_space,
    member_chunk_size=2,
    maximum_dataset_bytes=2**26,
)
split = split_operator_dataset(
    dataset, train_fraction=0.5, validation_fraction=0.25
)
assert split.group_keys == ("realization",)
groups = [
    {record.identities["realization"] for record in part.provenance}
    for part in (split.train, split.validation, split.test)
]
assert groups[0].isdisjoint(groups[1])
assert groups[0].isdisjoint(groups[2])
assert groups[1].isdisjoint(groups[2])
```

## What the distance records mean

`prepare_periodic_flow_ensemble` returns a `PreparedPeriodicFlowEnsemble`, whose
`execute()` produces a `PeriodicFlowEnsembleResult` bound to its initial measure.
Its `.result` is the native fixed-step result. Construction rejects foreign
semantic member labels or problem discretization identities, even for equal
array shapes. `CoupledEnsemblePair` validates ordered members, weights and saves;
`align_coupled_ensembles` restores order and returns the original owners unchanged
when the permutation is the identity.

The comparisons fold fine Fourier modes onto the coarse lattice before a
coarse inverse transform. This is exactly fine-grid injection on nested grids,
**without truncating the evolved fine state to the coarse band** or allocating
the full fine physical grid. All three metrics accept `member_chunk_size=8` and
`maximum_workspace_bytes=2**28`. Admission covers staged shared-node values,
chunked modal/transform buffers and transport workspaces; refusal raises
`MemoryBudgetError` with the cap name and required/allowed byte counts.
Conversion and transport executables are also admitted against their compiler
memory plans; a provider without memory analysis is refused. These operation
budgets do not cover caller-owned trajectories, unrelated live arrays, allocator
fragmentation or external runtime-provider workspaces, and are not process-RSS
certificates.

`periodic_flow_operator_dataset(result, /, ...)` accepts only this bound result,
not an independently supplied initial state. Its `member_chunk_size=8` policy
bounds conversion transients, and `maximum_dataset_bytes` admits both retained
and transient buffers with the same typed refusal. Downsampling uses retained-band
restriction, unlike the metric's alias folding. Vorticity follows
`PeriodicLerayProjector.vorticity_from_velocity`; leakage groups derive from the
coupling/member identity and are resolution-independent but measure-specific.

The three distance APIs have intentionally different meanings:

- `componentwise_empirical_wasserstein` computes exact one-dimensional empirical
  Wasserstein distances for each node and velocity component. Its scalar is the
  p-root of the coarse-quadrature integral of the **component mean** of Wp^p.
  This lower diagnostic ignores joint dependence; it is not exact joint or
  correlation-marginal Wasserstein distance.
- `paired_coupling_upper_bound` uses the declared member pairing and the mean
  velocity-vector norm cost (or the corresponding p-cost). It bounds the joint
  empirical transport cost at each node for that coupling; pairing does not
  solve an optimal-transport problem.
- `sliced_joint_marginal_distance` projects joint velocity vectors at explicitly
  declared node tuples onto random directions. A key and projection count are
  required. Its projection standard error measures projection Monte Carlo
  variability, not the finite-member sampling error, tuple-sampling error, or a
  certified confidence interval for exact multidimensional Wasserstein distance.

Compare diagnostics only with their declared norm, `p`, time index, tuple
selection, and projection count. No one of these APIs computes the cited exact
integral of multidimensional correlation-marginal W1. Small observed differences
can be below a Monte Carlo floor; an inconclusive finite campaign is not a failed
convergence theorem.

## Scaling, uncertainty, and validity

`PeriodicRadialIncrementMomentPlan` evaluates the continuous sphere or ball
average of the second increment moment of the retained trigonometric polynomial.
This is exact for that finite modal field, not for an unknown continuum solution.
It reports finiteness and forbidden-coefficient energy; an exact algebraic
formula is not permission to ignore invalid inputs. Sphere and ball averages are
different statistics and must not be pooled under one label.

`StructureFunctionScalingPlan` fits `S2(r) = sqrt(mean_member M2(r))`, not M2
itself, using a declared physical radius window and a tuple of per-resolution
radial results with their discretizations. Its optional `weights` are
radius-fit weights, not ensemble probability weights. The lower
radius must resolve every grid (`r_min >= 2 * max(dx)`); declare an upper radius
rather than fitting arbitrary radii after seeing a slope. Minimum-point, member-
count, rank, conditioning, positivity, and validity checks can cause abstention.
Inspect each fit's `status` and `reason` before reading its exponent or uncertainty.
An abstained fit is evidence of insufficient information, not a zero exponent.

Member jackknife replicates use the same deleted member jointly across radii and
coupled resolutions, preserving those dependencies. They are not independent-
radius chi-square errors. Exponent uncertainty and cross-resolution comparisons
therefore describe this finite, paired sample; they do not certify an asymptotic
scaling law. The snapshot route is instantaneous: it does not establish a
uniform-in-time or time-integrated structure-function bound. Report the saved
time, radius window, averaging policy, weights, sample count, and any abstention
alongside `common_exponent`, `envelope_constant`, and
`consistent_within_campaign`; the last field is not a compactness certificate.

For explicit time-integrated evidence, use `evaluate_time_quadrature` with
results nested as `(resolution, saved_time)`, the discretizations, and declared
nonnegative time-quadrature weights. It returns the distinct
`TimeIntegratedStructureFunctionScalingEvidence`, retaining those weights, and
sums weighted M2 before taking the square root and fitting. Their normalization
and time units affect the constant. It is not an instantaneous snapshot and
still does not certify a continuum limit or uniform-in-time theorem hypothesis.

## Dataset transfer is not metric alignment

The operator-data adapter may transfer coefficients to a declared observation
discretization using retained-band restriction before materializing sampled
inputs and targets. This defines the **dataset observation policy**. It is
appropriate when training data intentionally represent a common retained band,
but is not the native-grid comparison used by the statistical distance APIs.
Do not use restricted dataset targets to claim that unresolved solution
components agree.

The adapter admits the member-by-saved-time product against
`maximum_dataset_bytes` before materialization. Selected input/target saves must
be valid and finite or conversion is refused; failure at an unselected later
save does not invalidate earlier selected cases. Target times must be strictly
later than the selected input time. Provenance grouping uses only a
resolution-independent `realization` identity derived from the root realization
and member. Time, resolution, measure, and dynamics belong in case IDs
or metadata, not separate leakage-group identities. The default grouped split
therefore keeps every saved time and every coupled resolution of a member in one
partition. Never replace it with a random row split.

## Finite-campaign qualification

The `periodic-statistical-solution` route of
`tools/incompressible_flow_qualification.py` executes a coupled power-law campaign
on at least three nested resolutions (default `--mode-counts 16 32 64`).
Consecutive pairs report componentwise empirical W1, paired coupling upper
bounds, and sliced joint-marginal distances with projection errors. Each pair
samples its coarse injection nodes, integrating the same component-mean W1 over
the same physical-domain quadrature measure. Delete-one-member jackknife removes
the **same coupled member** at every resolution, recording SEs of the distances
and their consecutive differences. With declared `--refinement-z 2`, all
differences `d_j-d_(j+1) > z*SE` pass refinement; any significant increase fails;
otherwise the campaign is inconclusive.

SVV strength/order/cutoff/kernel, power-law exponent/scale, and reference Courant
number are declared scientific parameters. The power-law family is motivated by
Herde et al. (2024), Appendix B.2.2, Eq. (42); LMPP (arXiv:1909.06615) supplies
metric/structure-function motivation, not the campaign exponent. Inconsistent
scaling or any abstention is recorded with reasons and cannot yield a pass.
Ball-averaged radial moments require `coarse_count > 8`. Solver success, finite
trajectories/metrics, divergence and energy-budget tolerances, and positive SVV
dissipation remain gates. A shared step size preserves matching save grids;
runtime failure returns the artifact with guard/invariant evidence and no
statistical metrics. Inputs bind measure/coupling and per-resolution compilation,
prepared method/ensemble, result ensemble, discretization, and SVV plan/prepared
IDs. Support binds the kernel, spatial dimensions and methods with
`convergence_certified=false`: finite-campaign passing is not a continuum proof.
