# Battery production admission

## Availability is not authorization

The public construction APIs expose thermal ECM, canonical isothermal
Marquis-2019 SPMe, and one-cell circuit-connected electrothermal ECM. An explicit
candidate profile permits development execution; it never confers numerical or
predictive release status. Results report the admission scope and separate
numerical/predictive admission IDs.

DFN and series-pack implementation remain behind their respective authenticated
entry decisions. An eligible entry profile is not a model release. The DFN gate
requires a live SPMe release and exactly two converged, independently identified
reference families; the pack gate requires the exact released circuit ECM. Missing
rights, disagreement, overlapping uncertainty intervals, missing signatures, or
stale dependencies do not authorize construction.

The built-in candidate envelopes describe mathematical domains. They do not
invent finite production operating, parameter, initialization, discretization, or
resource limits. A production release requires those bounds to be supplied,
approved before observations, reflected in the exact support identity, exercised
by the complete campaign matrix, and enforced at dispatch. Empty production
bounds are a release refusal, not an unbounded guarantee.

## Native model execution

### Circuit ECM

The implicit cell owns differential charge `q`, RC polarization `w`, temperature
`T`, and algebraic terminal current `I`. Positive current enters the positive
terminal. Its equations are:

```text
q̇ = I
ẇ = −w/(RC) + I/C
Cₜ Ṫ = Qᵢᵣᵣ + Qᵣₑᵥ − ambient heat loss
V₊ − V₋ = U + I R₀ + Σw
```

The native nodal circuit owns voltage/current coupling. The thermal graph adds
ambient and internal-edge heat exactly once, using the cell's existing heat
capacity. Resistor heat uses actual constitutive voltage drop times current,
not commanded current. Current-source, voltage-source, and resistive-load
boundaries share the same augmented DAE for initialization, execution, events,
replay, diagnostics, and ledgers.

`BatteryDAESolvePlan` binds the native solver policy and the exact model guard
IDs. Every current/rest transition ends the old hold, preserves physical
inventories, recomputes algebraic state and rates with zero physical correction,
and restarts BDF history. The result retains original segment solutions,
one-sided forcing, restart consistency, and event evidence. Observation-side
selection does not erase the opposite transition side from ledger evidence.

`MODEL_LEDGER_FAILED` makes a finite but physically inconsistent trajectory an
application failure. Saved-state energy quadrature defects remain explicit;
local residual success is not an exact integrated-energy claim.

### Canonical SPMe

The isothermal SPMe uses conservative radial solid transport and the
Marquis-2019 asymptotic through-cell electrolyte model. Electrolyte transport
coefficients use the declared distinguished-limit evaluation point. Eq. 49
applicability is checked alongside property support and conservation.

For current density `i` in A/m², charge-concentration scale
`q_scale = i L / Dₑ` has units C/m³. Its OCP-curvature
term is `q_scale² |U″| / (F R T)`. Equivalently, with molar scale
`δc = q_scale/F`, it is `δc² |U″| / (R T/F)`. Confusing these scales introduces
an erroneous factor of `F²`.

The reference tooling constructs an empty PyBaMM parameter mapping from
self-authored SI data; it does not import bundled measured parameter sets. A
separate SciPy sparse-exponential implementation evaluates the paper equations.
Reference files bind actual executed package/source/binary bytes and dependency
versions. An upstream commit citation is not substituted for installed-byte
identity. Native FV surface reconstruction and reference sample conventions are
kept explicit rather than shifting outputs to force agreement.

## Retained evidence and asymmetric trust

`BatteryReleaseBundle` retains criteria, campaign starts and observations,
per-criterion evidence, raw artifacts, reference rights, resource records,
independent replay comparison, and complete coverage. `build_battery_release`
validates the entire chain and constructs the gate internally. Opaque evidence
IDs and caller-selected gate names cannot replace retained proof. Expiry is
bounded by the earliest cited prerequisite expiry.
The model-owned release contract fixes the required cases, metric definitions,
reference comparisons, independent replay observables, and workload matrix.
A self-consistent, signed subset is still incomplete and is rejected.
The retained distribution manifest content-addresses the source build, wheel,
sdist, and SBOM mapping. Scientific evidence uses its source build ID; runtime
admission uses its distinct distribution ID. Both identities are authenticated.

`QualificationRoleTrust` assigns distinct keys to criterion approver, executor,
scientific reviewer, entry-decision authority, release authority, and channel
promoter. Ed25519/KMS purpose signatures use the existing signing trust store's
activation, expiration, rotation, and revocation records. Production tooling
never creates an authority key implicitly. HMAC remains suitable only for
isolated tests.

`RuntimeDistributionAttestation` binds a distribution ID to the actual imported
package file manifest and Python identity under an executor signature. Admission
rehashes the executing package; repeating a caller-provided distribution ID is
not sufficient. The executor must independently verify the wheel/manifest
before attesting an installation.

`load_battery_execution_admission` reads retained deployment proofs while trust
roots are supplied separately. `BatteryExecutionAdmission.validate_run` checks
actual model coordinates, parameter/initial/protocol bounds, intrinsic model
domains, and predictive parameter binding before dispatch. A numerical profile
alone makes no claim about a particular commercial cell or safety envelope.

## Reproducible unpromoted distributions

After source, facades, examples, docs, runners, and harnesses are finalized:

```text
python -m tools.battery_distribution --source-root /path/to/worktree --output /outside/worktree/build-execution
```

The tool builds wheel and source archive, installs locked runtime dependencies
including asymmetric signing support into a fresh environment, verifies imported
package files against the wheel, and records an installed software bill of
materials. It retains a frozen source snapshot separately from a package-free
tools/benchmarks/examples harness so clean execution cannot import the checkout.
The receipt contains the isolated campaign command with its distribution
manifest and an example command prefix. Append an example module such as
`examples.battery_circuit_ecm` and its admission arguments to that prefix.
Output is immutable and
must be outside the source tree; changes during assembly abort publication.

The SBOM is a dependency/byte inventory, not license clearance. The clean-install
receipt is not independent replay. Source or harness changes invalidate the
frozen sweep. Final campaigns use one distribution and incremental signed,
unpromoted staging indexes; downstream campaigns authenticate their exact
prerequisites. A second independent actor/install must execute and compare the
full required matrix before public promotion.

## Campaign outcomes and resources

Scientific execution uses:

```text
python -m tools.battery_qualification --campaign-spec campaign.json --criteria criteria.json --output result.json
```

The exact registry owns case inputs, metric formula/unit/aggregation/uncertainty,
finite targets, reference count/rights, and raw-output schema. Every criterion
gets its own causal start/observation/evidence record. Valid observed violations
are failed; unavailable results after start are inconclusive. Preflight refusals
produce administrative attempt records, not scientific observations. All outcomes
are retained atomically under `qualification-artifacts`.

Registered scientific runners are `ecm-analytic`,
`spme-marquis2019-scientific`, `circuit-ecm-analytic`, and
`circuit-ecm-lifecycle`. The circuit lifecycle runner uses a declared
0.01-second grid on `[0, 2]` seconds for standalone parity and fixed-grid
JVP/VJP checks, with a separate 0.005-second terminal-event refinement.
The analytic and lifecycle matrices are both required for circuit release.
The circuit example and analytic runner use adaptive BDF2; internal step-size
control is independent of the requested output spacing. Native implicit stages
solve for increments about retained physical states, so small Newton corrections
are not rounded away against large charge or temperature offsets. Stored rates
come from those increments, not from subtracting rounded output states.

Circuit conservation ledgers independently integrate saved current, power, and
heat with nonuniform piecewise-quadratic quadrature. Stencils do not cross
held-current jumps, and prescribed-current charge remains integrated exactly.
Two-node intervals use trapezoids and must still satisfy the same qualification
limits. Invalid terminal-output suffixes retain their validity masks and JSON
`null` values, not nonfinite JSON numbers or fabricated samples.

Absolute resource campaigns use `BatteryResourcePlan` and at least five
synchronized warm executions. Preparation, lowering, compilation, first/warm
execution, host/device/scratch/output/checkpoint bytes, solver work, archive and
event capacity, and prohibited dense intermediates are distinct observations.
Missing telemetry remains unavailable/inconclusive; it is never replaced with
zero. Relative regression comparisons require matching workload, environment,
and harness identities.

## Deployment and channel state

The circuit ECM and SPMe examples require either `--development` or externally
authenticated inputs. The separate optimization demonstration is development-only:

```text
python examples/battery_circuit_ecm.py --deployment deployment.json --trust-roots public-roots.json --runtime-attestation runtime.json --profile-id PROFILE_ID --distribution-id DISTRIBUTION_ID
```

No example validates a historical gate at its issue time instead of current UTC
time. Runtime admission has a bounded lifetime and does not bypass a profile's
own expiry or revocation.

Production dispatch verifies time and runtime bytes on every host call.
Whole-run `jit`, `grad`, and `vmap` transformations with production admission are
refused: a cached executable must not bypass expiry or runtime checks.
Development candidates retain those transformations. A deployed transformed
optimization/control workflow requires its own qualified execution boundary;
a numerical-model token does not grant that capability.

`PromotionRepository` atomically commits signed `PromotionState` records,
channel heads, and generation floors with compare-and-swap. Rollback revalidates
the target's live trust, dependencies, distribution, envelopes, and rights. An
invalid rollback target cannot be silently reactivated; withdrawal is a signed
empty refusal state. Consumers retain their generation floor to detect replay.

None of these mechanisms establishes safety certification, abuse prediction,
thermal-runaway modeling, warranty, regulatory acceptance, or suitability for
fast charging.
