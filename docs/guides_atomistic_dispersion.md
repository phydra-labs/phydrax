# Atomistic dispersion

PhydraX has several dispersion definitions. They share some symbols, such as `C6`, `r4r2`,
Becke-Johnson damping, and a coordination number, but they are separate scientific
methods with separate identities. Nothing converts one into another, and nothing falls back
from one to another:

| Definition | Public owner | Coefficients | Coordination | Pair sum |
| --- | --- | --- | --- | --- |
| Lennard-Jones / Tang-Toennies | `PairOverrideLennardJonesPotential`, `LennardJonesDispersionCorrection`, `DampedDispersionPotential` | fixed per type | none | real-space cutoff |
| LJ-PME | `LennardJonesPMEPotential` | fixed per-type `C6` matrix, **positive semidefinite** | none | real space plus reciprocal mesh |
| AIMNet2 `D3TS` | `TSDispersionReference`, `TSDispersionPotential` | network-predicted atomic `(C6, alpha)` | none | hard-radius directed routes |
| Conventional D3(BJ) | `D3BJPotential` | reference-table interpolation | truncated logistic CN | image-pair sum with a cutoff or S5 switch |
| FourierD3 | `FourierD3Potential` | **signed** low-rank factors of the reference block | modified counting function | particle mesh with no real-space cutoff |

None of these definitions includes a three-body Axilrod-Teller-Muto (ATM/C9) term, and
`D3BJPotential` has no zero-damping variant. Neither is substituted silently. The D3 and
FourierD3 term identities record `"three_body": "excluded"`.

## Reference tables and data rights

`D3BJPotential` and `FourierD3Potential` read `D3ReferenceTables`. PhydraX **does not bundle
or download** D3 reference data, and it has no fallback table. Grimme's `dftd3.tgz` data
carry the GPL-1.0-or-later notice of `dftd3.f`. If you hold those files and have the right to
use them, you build the tables yourself in one of two ways:

- `D3ReferenceTables.from_grimme_fortran(dftd3_source, pars_source, species=..., length_unit=..., energy_unit=..., provenance=...)`
  parses caller-held `dftd3.f` and `pars.f` on the host. The SHA-256 of each file must equal
  the digest declared in the provenance under the artifact names `"dftd3.f"` and `"pars.f"`.
  The provenance kind must be `"grimme-dftd3-reference"`. Only atomic numbers 1-94 are
  covered. The archive's units (bohr, and hartree bohr^6 for `C6`) must be declared by the
  caller. The covalent radii arrive already scaled by `k2 = 4/3`.
- Explicit arrays passed to the `D3ReferenceTables(species, covalent_radii=..., r4r2=..., c6_reference=..., reference_coordination=..., reference_valid=..., length_unit=..., energy_unit=..., covalent_radius_scale=..., provenance=...)`
  constructor.

Either way, an explicit `D3TableProvenance` is required. Its `kind` is one of
`"grimme-dftd3-reference"`, `"caller-derived"` or `"synthetic-fixture"`. Kinds other than
synthetic fixtures must declare `(artifact, sha256)` digests. Three rights fields are
recorded separately:

- `use_admitted` records that the caller holds the right to use the data numerically.
  Preparing either D3 term calls `require_use()`, which refuses tables without this flag.
- `redistribution_admitted` is a separate right. A digest or a use admission never implies
  it.
- `license_id` and `admission` are declared labels. They are recorded, not verified.

Synthetic fixtures need no digest. They qualify only their own invented species and never
establish element fidelity. PhydraX's qualification source catalog lists the inspected NVIDIA
operations source as Apache-2.0 code. It also states that the D3 reference tables remain
under their own GPL notice.

The tables are compact over species. `species_lookup[Z]` is `-1` for an atomic number with
no row. If an active atom has an uncovered element, preparation refuses it; the atom is
never treated as padding. Reference blocks are validated as follows:

- `c6_reference[S, S, R, R]` must be exactly symmetric under `(a,p) <-> (b,q)`.
- `C6` must be positive on every populated slot pair and exactly zero elsewhere.
- Unpopulated slots must carry zero coordination.

Restored tables are checked again with `validate_restored()`.

`D3BJDamping(a1=..., a2=..., s8=..., s6=1.0, a2_unit=BOHR, functional=...)` holds the
published Becke-Johnson parameters. `a2` is a length. The four parameters are dynamic leaves
with no gradient by default.

## Conventional D3(BJ)

`D3BJPotential(tables, damping, coordination, dispersion, *, coordination_cutoff, route_chunk=4096)`
implements the two-body treatment of Grimme, Antony, Ehrlich and Krieg (J. Chem. Phys. 132,
154104, 2010), with Becke-Johnson damping from Grimme, Ehrlich and Goerigk (J. Comput. Chem.
32, 1456, 2011). The energy is

$$
E = -\tfrac12 \sum'_{i,j,\mathbf T} S(r)\left[\frac{s_6\,C_6^{ij}}{r^6+R_0^6}
  + \frac{s_8\,C_8^{ij}}{r^8+R_0^8}\right],\qquad
C_8^{ij} = 3\,C_6^{ij}\,r4r2_a\,r4r2_b,\quad
R_0 = a_1\sqrt{3\,r4r2_a\,r4r2_b} + a_2 .
$$

The sum runs over every directed image route $(i,j,\mathbf T)$ with
$0<r=|x_i-x_j+\mathbf T|<$ dispersion cutoff. Repeated images and nonzero self images are
included.

**Coordination number.** `D3CoordinationPolicy` defines the conventional truncated count:

$$
CN_i = \sum_{\text{routes},\,0<r<r_\text{cn}} \frac{1}{1+\exp\!\left(-k_1\,(R_{\text{cov},ab}/r-1)\right)},
\qquad k_1 = 16 .
$$

The sum runs over every full directed image route inside the coordination cutoff.
$R_{\text{cov},ab}$ is the sum of the two table radii, with `covalent_radius_scale` already
folded in. The cutoff leaves a **value step**: it is not a smooth switch.

**C6 interpolation.** The pair coefficient is

$$
C_6^{ij} = \frac{\sum_{p,q} L_{pq}\,C_6^{\text{ref}}(a,p;b,q)}{\sum_{p,q} L_{pq}},
\qquad
L_{pq} = \exp\!\left(-k_3\left[(CN_i-CN_a^p)^2+(CN_j-CN_b^q)^2\right]\right),\quad k_3=4 .
$$

The sums run over populated slot pairs. A stop-gradient log-sum-exp shift normalizes the
weights; this shift is an exact identity. The policy's `interpolation` argument selects one
of two rules:

- `"normalized-gaussian"`: every populated pair contributes. The result is smooth in both
  coordination numbers.
- `"nvalchemi-relative-threshold"`: the rule of the inspected NVIDIA kernel. It skips pairs
  whose shifted log weight is below `-threshold` (default `12`). This is a distinct
  definition and is discontinuous where a pair crosses the threshold.

Neither rule reproduces the underflow branch in Grimme's Fortran `getc6`, which falls back to
the nearest reference.

**Resources, counting and switching.** Coordination and dispersion are two separate
neighborhood requirements with separate radii:

- `coordination_requirement` uses the `COORDINATION` role. It is a full directed route set
  and its radius is `coordination_cutoff`.
- `dispersion_requirement` uses the `IMAGE_PAIR_SUM` role. It is a set of canonical image
  halves, with `directed_multiplicity == 2`, and its radius is `D3DispersionCutoff.cutoff`.

Both radii are system lengths. A program must provide both neighborhoods explicitly.
Coordination numbers refuse half route sets.

`D3DispersionCutoff(cutoff)` truncates the energy hard at `cutoff`. Alternatively,
`D3DispersionCutoff(cutoff, switch_on=..., switch_off=...)` with
`switch_on < switch_off <= cutoff` applies the S5 switch
$S = 1-(10t^3-15t^4+6t^5)$ on $[r_\text{on}, r_\text{off}]$. The switch is C2 at both joins and
its third derivative jumps.

Evaluation runs in the table's units and returns system units through exact host
conversion factors. A counted coincident route ($r=0$) is a geometry failure. The NVIDIA
source skips such routes silently ($r<10^{-12}$); PhydraX does not. `atom_energy` splits
each route energy equally between its two endpoints, so the per-atom energies add up to the
total.

## FourierD3

`FourierD3Potential(tables, decomposition, damping, grid_shape, *, coordination_cutoff, definition=None, spline_degree=3, modulus="continuous-sinc-window", nyquist="symmetrized", frame="cell-origin", anchor_index=None, rank_chunk=None, budget=None)`
implements the fast summation method of Valeeva et al. ("A fast summation method for the
DFT-D3 dispersion correction", [arXiv:2607.15103](https://arxiv.org/html/2607.15103)), as
implemented by NVIDIA nvalchemi-toolkit-ops `10e12340c39fd12a2375d6301768e72d2cd12868`.
It is a different scientific definition from conventional D3. Its energies are not
interchangeable with conventional D3 energies.

**Modified coordination number.** `FourierD3CoordinationDefinition(steepness=16, epsilon=1e-6, epsilon_length_unit=BOHR, width=4)`
defines the counting function used here. Let $R_t = (R_\text{cov}/\text{scale} + R_c)/2$,
$s=\max(r,R_t)$ and $g=R_c-s$. Then

$$
k(r) = k_1 + \frac{(s-R_t)^2}{g^2+\epsilon},\qquad
f(r) = \frac{1}{1+\exp\!\left(-k(r)\,(R_\text{cov}/r-1)\right)}\quad(0<r<R_c),
$$

and $f=0$ otherwise. $R_c$ is `coordination_cutoff`. $\epsilon$ is an **area** in
`epsilon_length_unit**2`, and preparation converts it to the table's length unit. Preparation
refuses a covered species pair whose transition $R_\text{cov}/\text{scale}$ does not lie inside
$R_c$.

**Signed C6 factors.** `FourierC6Decomposition(tables, species, *, tolerance, max_rank=None, maximum_block_size=1024)`
eigendecomposes the reference **species/slot block matrix**
$M_{(a,p),(b,q)} = C_6^\text{ref}(a,p;b,q)$ on the host. This is never an instantaneous
atom-by-atom coefficient matrix. The decomposition works as follows:

- Eigenpairs are ordered by eigenvalue magnitude. **Negative eigenvalues are kept, never
  clipped**, and `negative_eigenvalues` counts them.
- Unpopulated slots are set exactly to zero before any error is measured.
- The rank is the smallest `r <= max_rank` whose maximum relative error over populated
  entries is at most `tolerance`. Signed errors are not monotone in rank, so every rank is
  scanned. `rank_errors[r-1]` is the error achieved at rank `r`.
- If no admitted rank meets the tolerance, `meets_tolerance` is false and
  `require_admitted()` refuses the decomposition. Constructing a `FourierD3Potential` with
  such a decomposition raises an error.
- `reconstruction_error(tables)` measures the stored factors again.
- `validate_restored(tables)` checks a restored decomposition against its error and identity.

Per-atom coefficients interpolate these factors:

$$
c_{il} = \frac{\sum_p v_{a,p,l}\,L_p}{\sum_p L_p},\quad L_p = \exp(-4\,(CN_i-CN_a^p)^2),\qquad
C_6^{ij} = \sum_l \lambda_l\,c_{il}\,c_{jl}.
$$

The interpolation has no threshold. The factors are signed, so this path is **not** the
LJ-PME dispersion definition. `LennardJonesPMEPotential` requires its per-type `C6` matrix to
be positive semidefinite and factors it with $\sqrt{\lambda}$. FourierD3 never passes through
that path.

**Reciprocal sum and self energy.** The energy is

$$
E = -\frac{1}{2V}\sum_{\mathbf k}\sum_{a,b}\sum_l \lambda_l\,K_{ab}(|\mathbf k|)\,
S_{al}(\mathbf k)\,\overline{S_{bl}(\mathbf k)} + E_\text{self},\qquad
S_{al}(\mathbf k)=\sum_{i\in a} c_{il}\,e^{i\mathbf k\cdot\mathbf r_i}.
$$

The terms of this expression are defined as follows:

- The sum over $\mathbf k$ **includes $\mathbf k=0$**. It uses the Hermitian half-spectrum mode
  weights of the shared mesh owner.
- $K_{ab}$ is the closed-form radial transform of
  $\phi_{ab}(r) = s_6/(r^6+R_0^6) + 3 s_8 q_{ab}/(r^8+R_0^8)$, where
  $q_{ab}=r4r2_a\,r4r2_b$. The transform is `damped_dispersion_fourier_kernel`.
- $K$ uses a 24-term Taylor series below $x=|\mathbf k|R_0=1$ and the closed form above it.
  The two branches join at roundoff level.
- $E_\text{self} = +\tfrac12\sum_i\sum_l\lambda_l c_{il}^2\,\phi_{aa}(0)$ cancels the
  zero-separation self pair contained in the reciprocal sum. It depends on the coefficients.
  It is part of the same scalar closure as the coordination-dependent coefficients, so its
  CN force and virial contributions are kept.

The sum has no real-space cutoff and no direct-sum fallback. FourierD3 requires a fully
periodic rank-3 cell and declares only the `COORDINATION` neighborhood requirement.

The implementation differs from the inspected source in three declared ways:

- **Nyquist convention.** `"symmetrized"` averages over Nyquist sign variants. The source
  evaluates only `+N/2`, which differs for triclinic cells.
- **Modulus floor.** A vanishing modulus window is refused, whereas the source applies a
  floor of `1e-10`.
- **Kernel crossover.** The series/closed-form crossover point of the radial kernel differs.

**Residency.** Species-by-rank-chunk channel spectra are transformed once per rank chunk.
Every species pair reuses them through the mesh `cross_quadratic`. Rank chunks run in a
checkpointed scan, and species pairs run in a nested checkpointed scan. `FourierD3Resources`
records the retained bytes exactly: one chunk's density block, one pair kernel, and the
per-atom rank coefficients with their cotangents. These bytes are admitted against the mesh
budget's retained-byte limit at preparation. Memory is bounded by species × `rank_chunk` ×
grid, plus the per-atom coefficients. It is not constant in system size.

**Domain execution.** `AtomisticDomainFourierD3Term(prepared, reciprocal_plan)` runs the
same scalar on a distributed domain: modified coordination numbers and coefficients per
owner, then the reciprocal sum and self energy through an
`AtomisticReciprocalDistributionPlan` of the same prepared term on the slab or pencil
schedule. Domain LJ-PME (`AtomisticDomainLennardJonesPMETerm`) is admitted only when the
interaction sites are a proven identity image of the particles; virtual sites are refused.
See [canonical domain terms](guides_atomistic_distributed_execution.md#canonical-domain-terms).
Process-level checks of these routes are forced-CPU functional evidence only.

## AIMNet2 TS dispersion

`TSDispersionReference` and `TSDispersionPotential` implement the `DispParam` and `D3TS`
output stages of AIMNetcentral 0.2.0. These stages are source specific. They are not
coordination-dependent D3, and they read no `D3ReferenceTables`.

The predicted atomic coefficients are `table[Z] * exp(clamp(raw, -4, 4))`. Here `table[Z]`
is a caller-supplied species reference `(C6, alpha)`, and row 0 must be the padding dummy
`(0, 1)`. Pairs use the Tkatchenko-Scheffler combination rule (Tkatchenko and Scheffler,
Phys. Rev. Lett. 102, 073005, 2009), with the source's denominator floor:

$$
C_6^{ij} = \frac{2\,C_6^i C_6^j}{\max\!\left(C_6^i\alpha_j/\alpha_i + C_6^j\alpha_i/\alpha_j,\ 10^{-4}\right)} .
$$

The pair energy uses a Becke-Johnson-like rational damping with $d$ in bohr:

$$
e_{ij} = C_6^{ij}\left(\frac{s_6}{d^6+r_0^6} + \frac{3\,s_8\,r4r2_i r4r2_j}{d^8+r_0^8}\right),\qquad
r_0 = a_1\sqrt{3\,r4r2_i r4r2_j}+a_2 .
$$

Each directed route contributes $-\tfrac{E_h}{2}e_{ij}$ in eV. The constants come from the
source:

- `SOURCE_BOHR_ANGSTROM = 0.5291772105638411` converts angstrom distances to bohr.
- `SOURCE_HARTREE_ELECTRONVOLT = 27.211386024367243` is $E_h$ in eV.

**Counting.** A full directed list counts every unordered pair, including each periodic image
route and nonzero self image, exactly once in total. For half-stored lists,
`route_energies(..., multiplicity=...)` takes an explicit integer multiplicity.

**Cutoff and limiter kinks.** `pair_cutoff` (in angstrom) is a hard truncation, so the energy
jumps where a route crosses it. The two clamps are declared model nonlinearities. Their kinks
are at `raw = ±4` and at the combination denominator floor. They follow the torch `clamp`
derivative convention: the gradient passes at and inside a bound and is zero outside
(`source_clamp`).

**`r4r2` table.** `TSDispersionPotential(..., r4r2=table)` requires an explicit
caller-supplied buffer indexed by atomic number. No elemental table is bundled
or reconstructed implicitly. Source conversion preserves the exact buffer in
the admitted source inventory, including its dtype.

AIMNet2 models bind these stages through `phx.nn.atomistic.AIMNet2OutputKind.DISPERSION_PARAMETERS`
and `TS_DISPERSION`. A `DispParam` stage requires an explicit reference table;
a native TS stage requires its `TSDispersionPotential` in `stage_values`.
Because the coefficients are network outputs, coordinate and cell derivatives
include the predicted-field response by composition.

## Error budget

Each of the following is a separate error source, and none of them bounds another:

| Source | Applies to | Evidence carried |
| --- | --- | --- |
| Coefficient reconstruction residual | FourierD3 | `FourierC6Decomposition.achieved_error`, `rank_errors`, `FourierD3Evidence.coefficient_residual` (maximum relative populated-entry error of the reference block, not of any observable) |
| CN-definition difference | FourierD3 vs conventional D3 | The definitions differ, so no tolerance applies. Compare against conventional D3 only as a different method. |
| Interpolation-rule difference | D3 `"normalized-gaussian"` vs `"nvalchemi-relative-threshold"` | Distinct identities; `threshold_relative_weight` |
| Real-space truncation | D3 (hard or S5), TS (hard `pair_cutoff`) | `D3BJRegularity.dispersion_cutoff_step` bounds the largest reference-`C6` pair energy at a hard cutoff |
| Mesh interpolation and reciprocal truncation | FourierD3 | Grid shape and spline degree; the mesh Nyquist limit. Converge the energy by refining the mesh. |
| Roundoff | all | Including the radial kernel's series/closed-form join |
| Force/stress discrepancy | all | Measure it separately, for example with finite differences against an independently converged reference. A coefficient tolerance is not a force or stress certificate. |

## Derivative matrix

In all three definitions, forces and stress are derivatives of one scalar energy. That
includes the coordination-number and coefficient chains, and for FourierD3 the self energy.
Stress is the tension-positive strain derivative $\operatorname{sym}(\partial E/\partial\varepsilon)/V$
at fixed fractional coordinates and integer image shifts.

| | Conventional D3(BJ) | FourierD3 | AIMNet2 TS |
| --- | --- | --- | --- |
| Geometry | finite, orthorhombic, triclinic; partial-periodic is listed in its profile | fully periodic rank-3 only (orthorhombic, triclinic) | as admitted by the AIMNet2 model profile |
| Energy / forces | yes | yes | yes, through the model |
| Strain stress | periodic geometries (the finite profile is energy-force only) | yes | through the model's cell derivative |
| Per-atom energy | complete (equal endpoint split) | not available (`atom_energy_complete=False`) | route energies |
| Parameter derivatives | damping `s6, s8, a1, a2`, table arrays, coordination policy scalars | damping, stored signed factors and eigenvalues, reference coordination numbers, `r4r2`, covalent radii; **not** the reference `C6` table (a changed table needs a new decomposition) | `s6, s8, a1, a2`, predicted coefficients |
| Higher order | defined by the closure inside the regularity domain below | defined by the closure inside the regularity domain below | almost everywhere; kinks at clamp bounds |

The capability profiles `atomistic.dispersion-d3` list energy-force-stress (energy-force
for finite D3) under the stress convention
`tensile-positive-strain-gradient-per-cell-volume`. Their open gates include table-rights
admission, regularity-locus reference, element-table fidelity and, for FourierD3, mesh
convergence. No release claim is made. Higher-order or mixed derivatives are mathematical
properties of the closure inside the domain below. They are not a qualified profile entry.

`benchmarks/atomistic_dispersion.py` keeps coefficient, cutoff, reciprocal and
observable discrepancies separate from timing. Warm kernel records retain the
canonical `PhaseMemoryEvidence` payload with host RSS, device allocator samples,
sample count and sampling gaps. Interval samples, allocator lifetime high-water
marks, logical retained arrays and compiler estimates are distinct evidence;
sampled peaks are not certified upper bounds.

**Regularity limits**, derived from the formulas:

- **D3(BJ):**
  - The coordination cutoff is a value step of up to `D3BJRegularity.coordination_cutoff_step`
    per route.
  - A hard dispersion cutoff is a value step (`dispersion_cutoff="value-step"`).
  - With the S5 switch the energy is C2 (`"c2-s5-switch"`).
  - The threshold interpolation rule steps at its threshold set.
  - Coincident atoms are refused.
  - Derivatives of every order hold only away from these sets.
- **FourierD3:**
  - The counting steepness is C1 at $R_t$. Its second derivative jumps by
    $2/((R_c-R_t)^2+\epsilon)$, reported as `transition_curvature_jump`.
  - At $R_c$ the counting value steps from $\exp(\texttt{log\_cutoff\_step})$ to zero. That
    value is small but nonzero, so the energy is neither exactly C0 nor exactly C1 there. A
    tiny step is not smoothness.
  - The B-spline assignment of degree $p$ makes the mesh energy $C^{p-1}$ in positions
    (`mesh_coordinate_continuity`).
- **TS:**
  - Value jump at `pair_cutoff`.
  - Kinks at `raw = ±4` and at the denominator floor, with one-sided clamp derivatives.

## Example: conventional D3(BJ) on a finite dimer

The tables below are **non-physical illustrative data**. They use a `"synthetic-fixture"`
provenance and qualify no element. For real calculations, build the tables with
`D3ReferenceTables.from_grimme_fortran` from files you hold under an admitted use right.

```python
import jax
import jax.numpy as jnp
import numpy as np
import phydrax as phx
from phydrax.units import BOHR, HARTREE

phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))

provenance = phx.atomistic.D3TableProvenance(
    "illustrative synthetic D3 tables",
    kind="synthetic-fixture",
    license_id="synthetic-illustration",
    use_admitted=True,
    redistribution_admitted=True,
    admission="invented values; qualifies no element",
)
tables = phx.atomistic.D3ReferenceTables(
    (1, 6),
    covalent_radii=np.asarray([0.8, 1.9]),
    r4r2=np.asarray([2.0, 3.1]),
    c6_reference=np.asarray([[3.0, 12.0], [12.0, 40.0]]).reshape(2, 2, 1, 1),
    reference_coordination=np.zeros((2, 1)),
    reference_valid=np.ones((2, 1), dtype=np.bool_),
    length_unit=BOHR,
    energy_unit=HARTREE,
    covalent_radius_scale=4.0 / 3.0,
    provenance=provenance,
)
damping = phx.atomistic.D3BJDamping(
    a1=0.4289, a2=4.4407, s8=0.7875, functional="illustrative BJ parameters"
)
term = phx.atomistic.D3BJPotential(
    tables,
    damping,
    phx.atomistic.D3CoordinationPolicy("normalized-gaussian"),
    phx.atomistic.D3DispersionCutoff(8.0, switch_on=6.0, switch_off=7.5),
    coordination_cutoff=5.0,
)
units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
system = phx.atomistic.AtomisticSystemPlan(
    np.arange(2), np.asarray([6, 1]), np.asarray([12.011, 1.008]), units
).prepare()
prepared = term.prepare(system)


def routes(x, receivers, sources, radius, multiplicity):
    receivers, sources = jnp.asarray(receivers), jnp.asarray(sources)
    displacement = x[receivers] - x[sources]
    counted = jax.lax.stop_gradient(jnp.sum(displacement**2, axis=-1) < radius**2)
    return phx.atomistic.D3Routes(
        receivers, sources, displacement, counted, jnp.asarray(True),
        directed_multiplicity=multiplicity,
    )


def energy(x):
    coordination = routes(x, [0, 1], [1, 0], 5.0, 1)  # full directed
    dispersion = routes(x, [0], [1], 8.0, 2)  # canonical half
    return prepared.evaluate(coordination, dispersion).energy


positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
forces = -jax.grad(energy)(positions)
regularity = prepared.regularity  # declared cutoff steps and switch kind
```

In a full potential program, `term.coordination_requirement` and
`term.dispersion_requirement` declare the two neighborhoods, and the program's explicit
neighborhood resources must satisfy both.

## Example: FourierD3 signed factors

The same illustrative tables have an indefinite block, because $12^2 > 3\cdot 40$. The
decomposition therefore keeps one negative eigenvalue. `LennardJonesPMEPotential` would refuse
the same matrix as non-PSD.

```python
decomposition = phx.atomistic.FourierC6Decomposition(tables, (1, 6), tolerance=1.0e-12)
assert int(decomposition.negative_eigenvalues) == 1
decomposition.require_admitted()

fourier = phx.atomistic.FourierD3Potential(
    tables, decomposition, damping, (16, 16, 16), coordination_cutoff=4.0
)
periodic = phx.atomistic.AtomisticSystemPlan(
    np.arange(2), np.asarray([6, 1]), np.asarray([12.011, 1.008]), units,
    cell=phx.discretization.PeriodicCell(10.0 * np.eye(3)),
).prepare()
prepared_fourier = fourier.prepare(periodic)
cell = jnp.asarray(10.0 * np.eye(3))
# The 10 Å cell exceeds twice the 4 Å coordination radius, so the only counted
# coordination routes are the zero-shift directed pair.
result = prepared_fourier.evaluate(positions, cell, routes(positions, [0, 1], [1, 0], 4.0, 1))
evidence = result.evidence  # reciprocal/self energies, coefficient_residual, resources
```

`evidence.reciprocal_energy` and `evidence.self_energy` are reported separately.
`evidence.regularity` and `evidence.resources` hold the declared regularity and the exact
residency.
