# Computational topology

`phydrax.topology` analyzes the immutable oriented cell complexes owned by
`phydrax.discretization`. It does not introduce another mesh, graph, or complex
representation.

Three claims remain separate:

1. exact algebra for a supplied finite complex or filtration;
2. metric-dependent numerical Hodge evidence;
3. approximation of a continuum geometry or sampled field.

A persistence diagram from sampled values is exact for that discrete filtration. A
continuum-topology claim additionally needs support, reconstruction, and perturbation
or discretization evidence.

## Compact active complexes

Every exact calculation first removes inactive fixed-capacity cells. Results preserve
ambient entity IDs, but inactive slots cannot create generators, persistence bars, or
Hodge zero modes.

`CellSubcomplex` validates algebraic closure: every nonzero boundary cell of a selected
cell must also be selected. `CellComplexPair(K, A)` represents relative chains on
`K \ A`; boundary contributions landing in `A` vanish in the quotient. A relative
representative may therefore have nonzero ambient boundary, provided that boundary is
supported entirely in `A`.

```text
K = phx.topology.CellSubcomplex.full(mesh.topology)
A = phx.topology.CellSubcomplex.from_subsets(mesh.topology, "boundary")
pair = phx.topology.CellComplexPair(K, A)
```

## Exact homology

Coefficient domains are mandatory. `PrimeField(p)` validates that `p` is prime and
uses exact host arithmetic. `RationalField()` is the rank-only exact path used for
field-independent Betti dimensions and Hodge-nullity comparison.

```text
homology = phx.topology.compute_homology(
    K,
    coefficients=phx.topology.PrimeField(2),
    representatives="both",
)
rational = phx.topology.compute_betti_dimensions(
    K,
    coefficients=phx.topology.RationalField(),
)
```

Homology results record exact boundary ranks, optional sparsely stored cycle/cocycle
generators, Euler–Poincaré evidence, resource counts, and topology identities.
Sparsely stored output does not imply sparse elimination: exact reduction may fill in
and fails closed when its explicit resource policy is exceeded.

Reduced absolute homology uses an explicit augmentation. Reduced relative homology is
not inferred by adjusting degree zero and is intentionally rejected.

## Geometric support and star filtrations

Boundary incidence does not, by itself, certify the geometric vertex closure of a
nonregular cell complex. Lower- and upper-star builders therefore require an explicit
`CellVertexSupport`.

```text
support = phx.topology.cell_vertex_support(
    mesh.topology,
    (vertex_vertices, edge_vertices, face_vertices),
)
filtration = phx.topology.lower_star_filtration(
    K,
    support,
    vertex_values,
    source_id="phase-field-step-12",
)
```

Lower-star cells receive the maximum value of their vertices. Upper-star filtration is
a genuine superlevel filtration and uses the minimum vertex value while internally
ordering by its negative. Every filtration validates face monotonicity. Equal values
are valid; deterministic cell ordering is reduction provenance, not a claim that tied
cell pairings are canonical.

`PreparedVertexFiltration` evaluates the same fixed vertex-to-cell relation in JAX and
preserves leading batch dimensions. Exact persistence remains a host preparation.

## Persistent homology

`compute_persistence` supports ordinary persistence and induced relative persistence
with `A_t = A ∩ K_t`. It does not silently implement extended persistence or a pair
whose cells enter `A_t` at independent times.

```text
persistence = phx.topology.compute_persistence(
    filtration,
    coefficients=phx.topology.PrimeField(2),
    relative_to=A,
)
diagram = persistence.diagram()
```

The exact reduction retains cell-level zero-length pairs. The mathematical diagram
omits them by default. Essential bars use an explicit `has_finite_death` mask; they do
not share a sentinel with inactive packed slots.

Host diagrams have their natural interval count. `result.pack(capacity)` creates a
fixed-capacity JAX representation and raises rather than truncating when capacity is
insufficient.

## Local derivatives from a frozen pairing

The exact reduction pairing is discontinuous when filtration order changes. A frozen
pairing exposes only the valid local derivative: birth and death endpoints are gathers
from dynamic values while the complete frozen order remains admissible.

```text
frozen = phx.topology.freeze_persistence_pairing(persistence, filtration)
evaluation = frozen.evaluate(dynamic_cell_values)
```

`evaluation.ordering_valid` checks every adjacent cell in the frozen order. Endpoint
gathering is differentiable inside that chamber. The validity predicate and order
changes are not differentiable topology operations.

## Hodge realization and solver nullspaces

Exact rational Betti dimensions give structural expected nullity. They do not alone
certify a floating Hodge kernel. `phydrax.graph.validate_hodge_homology` additionally
checks the same compact active complex, boundary policy, harmonic rank, metric
orthonormality, kernel residuals, and the next observed eigenvalue.

`cochain_harmonic_kernel_certificate` builds a compact `phydrax.linalg.LinearSubspace`
and `KernelCertificate` only after those numerical checks. It does not select a
compatibility projection or gauge; that remains a solver/physics decision.

## Execution boundary

- Complex validation, exact rank, exact reduction, and persistence pairing are host
  preprocessing.
- DEC actions, prepared vertex filtrations, packed diagrams, and frozen endpoint
  evaluation are JAX operations.
- Different complexes require host preparation or an explicit heterogeneous padded
  layout. Coefficient batching does not make different incidence patterns identical.
