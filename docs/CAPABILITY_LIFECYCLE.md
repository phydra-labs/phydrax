# Capability lifecycle

PhydraX records implementation maturity, scientific evidence, and release authority as separate facts. A numerical implementation, passing test, benchmark, or qualification producer is not a release claim.

## Global dispositions

Every public or intentionally internal capability has exactly one canonical declaration and one disposition:

- **Internal**: intentionally unavailable through a supported public workflow.
- **Research**: implemented for bounded research use without an exact release candidate.
- **Candidate**: exact unreleased support profiles exist and may collect evidence.
- **Released**: every required evidence dimension and exact signed release profile passed.
- **Retired**: removed from the public surface and recommendation paths.

Domain-specific scales such as operator maturity, ROM maturity, particle maturity, tensor-network maturity, boundary Q-levels, and field-theory research levels remain descriptive metadata. They never override the global disposition or the signed release index.

The word **production** is reserved for an exact released support tuple. Implementation-complete, numerically qualified, scientifically qualified, operations-qualified, and release candidate should be used for all earlier stages.

## Evidence dimensions

Evidence is conjunctive and orthogonal:

1. implementation completeness;
2. numerical qualification;
3. derivative qualification;
4. performance qualification;
5. hardware and provider qualification;
6. scientific validation;
7. operations qualification;
8. rights and security qualification;
9. release authorization.

Passing one dimension never implies another. Synthetic or manufactured controls can satisfy numerical verification, but do not establish scientific validation. A generated benchmark or qualification program is not retained evidence. A retained artifact without exact source/build identity cannot satisfy release authorization.

## Breadth freeze

Until the stable release portfolios are admitted, accepted changes are limited to:

- correctness and security fixes;
- clean migrations and deletion;
- public API completion;
- capability-catalog migration;
- numerical qualification;
- application promotion against existing support tuples;
- mainstream methods named by the closure roadmap;
- release-enabling documentation and governance.

New application namespaces, operator architectures, frontier families, provider abstractions, or umbrella production claims require a closure exception naming their owner, exact support tuple, dependencies, evidence obligations, and permanent nonclaims.

## Pull-request obligations

A capability-affecting pull request must state:

- affected capability IDs and public symbols;
- added, removed, or changed support tuples;
- dependencies and providers affected;
- evidence invalidated by the change;
- generated inventory changes;
- public examples and documentation changed;
- disposition before and after the change;
- whether any release profile must be withdrawn or requalified.

A change cannot promote itself. Promotion requires current retained evidence and the configured independent trust policy.

## Canonical authority

`phydrax.qualification.CapabilityCatalog` owns the repository inventory. `CapabilityProfile`, `ReleaseIndex`, and the trust policy own exact release admission. Generated capability tables are derived views. Narrative documentation must not invent a different maturity state.

The checked capability inventory is intentionally not a release index and confers no license, support, scientific-validity, safety, regulatory, clinical, or commercial claim.
