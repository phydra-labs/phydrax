# Electronic transport conventions and profiles

Electronic transport is split by physical approximation. Periodic independent-particle
Kubo response, constant-relaxation-time Boltzmann transport, coherent finite-device
Landauer transport, interacting SCBA, and two-time fermionic Keldysh data are not
interchangeable names for one solver.

## Shared finite-region convention

`ElectronicTransportConvention` fixes four choices:

- Bloch sums use `exp(+i k·R)`.
- Retarded functions use the `exp(-i omega t)` time transform.
- Contact particle, charge, energy, and heat currents are positive into the finite
  region.
- The electron charge is signed and negative. A positive electron-particle current
  therefore gives a negative conventional charge current.

`retarded_embedding` evaluates the nonorthogonal Schur complement
`Sigma = (z S_dl - H_dl) g_surface (z S_ld - H_ld)`. Both coupling factors use the
same retarded complex energy `z`; the reverse block does not conjugate `z`.
`RetardedEmbeddingEvidence` retains the minimum eigenvalues and Hermiticity residuals
of the surface spectral function and `Gamma = i(Sigma - Sigma†)`. A causal admitted
embedding has positive-semidefinite Gamma within its declared tolerance.

`retarded_open_system_point` performs one native dense linear solve. It is an
algebraic primitive, not a device model or another NEGF hierarchy. The returned
spectral identity includes `2 Im(z) S` separately as numerical broadening. That term
is explicitly not a reservoir, collision process, or relaxation time.

`electronic_contact_current_kernels` returns energy-resolved kernels before an
energy measure and Planck prefactor. Particle current at contact `p` is
`Tr[Gamma_p (f_p A - G^n)]`, charge current is the signed electron charge times that
kernel, energy current is `E` times it, and contact heat current is `(E-mu_p)` times
it.

## Periodic independent-particle Kubo

`PeriodicKuboPlan` takes energies in joules, full Hermitian band-basis velocity
matrices in physical Cartesian meters per second, normalized k weights, primitive
cell volume in cubic meters, chemical potential, temperature, and an independent
`KuboDiamagneticSumRule`.

`raw_transitions()` retains positive-energy interband transitions without a line
shape. Degenerate subspaces are excluded from the regular transitions and included
in the Drude trace using the complete velocity block, so unitary rotations within a
degenerate subspace do not change the result. The returned `drude_weight` is a
zero-frequency delta-distribution weight. It is not a finite conductivity number.
The f-sum evidence compares one half of the zero-frequency Drude weight plus the
positive-frequency regular spectral weight against the independently supplied
diamagnetic weight.

`finite_frequency_kubo_response` accepts only strictly positive angular frequencies
and a `KuboLinewidth` with a nonempty physical mechanism identity. It broadens only
the regular interband poles. The response reports passivity from the dissipative
Hermitian conductivity and retains the f-sum residual. It never broadens the Drude
distribution into a finite ballistic DC value and never infers a relaxation time
from the linewidth.

Frequency execution is explicit: `frequency_policy="chunked"` (the default) uses
`frequency_workspace_bytes=72 * 1024 * 1024` and
`frequency_chunk_size=None`. With `bytes_per_frequency =
16 * K * bands**2 * dimension**2`, its batch size is
`min(F, max(1, frequency_workspace_bytes // bytes_per_frequency))`.
An explicit positive `frequency_chunk_size` overrides the budget.
`"parallel"` evaluates all frequencies together regardless of the budget or
chunk override. Whenever the resolved batch covers all frequencies, the chunked
policy also calls the original parallel computation directly, without a map or
scan wrapper. Both routes contract transition axes directly, retain resonant and
antiresonant poles, and return the required `(frequency, Cartesian, Cartesian)`
conductivity plus unchanged raw transition data. They do not form six-dimensional
frequency-by-transition-by-Cartesian products. Changing batching does not change
the physical approximation or per-frequency transition reduction schedule.

`KuboResponseEvidence` records the requested `frequency_policy`, actual
`frequency_execution`, `frequency_workspace_bytes`,
`requested_frequency_chunk_size`, effective `frequency_chunk_size`, and
`logical_frequency_workspace_bytes`. The latter is the conservative complex128
logical bound `bytes_per_frequency * effective_chunk`, not compiler temporary
bytes or process RSS; raw transition storage and required outputs are excluded.
The minimum one-frequency batch can exceed a smaller positive budget, and explicit
chunks and the parallel policy may also exceed it. Boolean/noninteger or
nonpositive budgets and explicit chunks are refused even for the parallel route.
`execution_policy_id` includes the requested budget and optional chunk separately
from the scientific `plan_id`, which remains a fingerprint of physical inputs.

The default is grounded in the quiet crossover table
`/tmp/quiet-wave/periodic_kubo-paired.json`: at 32 frequencies the old eight-wide
map took 0.558 ms versus 0.351 ms parallel, and at 128 it took 1.781 ms versus
0.761 ms parallel; at 512 it took 6.464 ms versus 2.340 ms parallel.
At the benchmark's 16 k points, 8 bands, and 3 axes, 72 MiB fits exactly 512
frequencies, preserving parallel execution across the measured moderate range.
Larger frequency workloads still tile under the fixed byte budget. Historical records
`benchmarks/memory_periodic_kubo_baseline.json` and the original candidate recorded
16,859,288 versus 803,448 compiler temporary bytes at 512 with an eight-wide chunk;
those numbers are historical evidence, not a measurement of the new default.
The updated candidate benchmark compares the default against explicit parallel
execution in the same run at above-budget capacities, including larger
compile-only cases. The memory gate is evaluated only where tiling engages.
Below-budget equal memory is expected, not a gate failure. Its above-budget
frequency-transition workspace is constant; full-response scratch also includes
frequency-sized passivity checks. New compiler and quiet-timing results must be
captured by running the benchmark.

`conserved_collinear_spin_evidence` checks the supplied band-basis generator by the
actual commutator `[diag(E), Sz]` and by Hermiticity. A consumer must call
`require_conserved_collinear_spin` before interpreting a response as conserved
collinear-spin transport. Spin-current or spin-torque claims with spin-orbit mixing
are outside this profile.

## Constant-tau Boltzmann thermoelectrics

`PeriodicBoltzmannPlan` takes diagonal band velocities and a
`ConstantRelaxationTime`. The relaxation time must be a positive scalar in seconds
and carry a physical mechanism identity. The plan forms the zero-field transport
moments `L0`, `L1`, and `L2` from `-df/dE` and returns:

- electrical conductivity `q² L0`,
- Seebeck tensor `(q T)^-1 L0^-1 L1` with signed electron charge `q`,
- Peltier tensor `T S`, and
- open-circuit electronic thermal conductivity
  `T^-1 (L2 - L1 L0^-1 L1)`.

The native linear solve is admitted only when `L0` has full sampled Cartesian rank.
The evidence retains rank, moment symmetry, Kelvin-Onsager residual, and minimum
eigenvalues of electrical and electronic-thermal conductivity. Scaling the supplied
relaxation time scales electrical and thermal conductivity but not Seebeck. No Kubo
linewidth or Green-function numerical eta enters this API.

## Disorder and nonequilibrium adapters

`ElasticDisorderEnsemblePlan` is an explicit finite probability measure over unique
realization identities. Evaluation retains every value and success flag. Failed or
nonfinite realizations are never dropped and the declared probabilities are never
renormalized around them.

`FermionicKeldyshTransportState` is a consumer-side contract for fermionic
`(time,time,mode,mode)` lesser, greater, retarded, and advanced functions with an
explicit fermion mode-order identity. It checks the CAR identity
`Ggreater - Glesser = GR - GA`, retarded/advanced causality, advanced-retarded
adjoint consistency, and a provider-supplied particle-continuity residual. It does
not relabel the existing real-scalar Keldysh types in
`applications.nonequilibrium_field`.

## Existing semiconductor profiles

The scalar effective-mass chain remains under `applications.semiconductor.quantum`:
`SemiInfiniteLead` and `CoherentDevice` own the analytic scalar lead and selected-
source coherent Landauer path; the response module owns zero-frequency noise,
quasistatics, and the separately derived coherent finite-lead AC response; the
dynamic module owns finite-lead transient evolution; and the scattering module owns
local optical-phonon Fock SCBA. These profiles are qualified separately. The generic
operator primitives do not replace them, and this guide does not claim that a
passing periodic Kubo or Boltzmann case qualifies any semiconductor profile.

## Evidence and nonclaims

`phydrax/chemistry/periodic/_transport_support.py` declares exact candidate support
coordinates. `phydrax/chemistry/periodic/_transport_campaigns.py` fixes disjoint
calibration and locked cases. Campaign declarations, unit tests, the smoke-sized
qualification tool, and `benchmarks/cm_transport.py` are not release evidence.

The current profiles do not claim interacting matrix NEGF, energy- or band-dependent
collision operators, a relaxation time inferred from spectroscopy, finite ballistic
DC conductivity, spin torque without a nonconserved-spin treatment, a generic
long-time Keldysh solution, or material agreement without independently qualified
Hamiltonians, velocities, collision parameters, and linewidth mechanisms.
