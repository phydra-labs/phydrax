# Sampling

`phydrax.sampling` contains stateless reference-space designs and persistent Markov
kernels. These are different numerical contracts: a reference design generates an IID
or quasi-Monte Carlo point set, while a Markov kernel advances correlated chain state.

These stochastic chains are distinct from deterministic weak-law
`phydrax.solver.solve_markov_cubature`; `markov_chain_measure` is named explicitly so
the two Markov contracts cannot be confused.

## Proposals

::: phydrax.sampling.AbstractProposal

::: phydrax.sampling.CallableProposal

::: phydrax.sampling.GaussianRandomWalkProposal

A normalized proposal supplies both sampling and `log q(proposed | current)`. The
Metropolis--Hastings kernel evaluates both directions, so asymmetric proposals retain
their exact Hastings correction.

## Persistent chains

::: phydrax.sampling.MetropolisHastings
    options:
        members:
            - initialize
            - refresh
            - step

::: phydrax.sampling.MarkovState

::: phydrax.sampling.MarkovTransitionInfo

::: phydrax.sampling.MarkovSampleResult

::: phydrax.sampling.sample_markov

Every position leaf has a leading chain axis. Retained sample leaves have leading
`(chain, draw)` axes, while transition evidence has leading
`(chain, draw, transition)` axes. Random keys are derived from semantic chain and
global-step addresses; extending a chain or draw count therefore preserves existing
prefixes.

The target callback must return one real scalar per chain position. A parameter update
may keep the current positions, but must call `MetropolisHastings.refresh` before the
next transition so stored target values cannot become stale.

Proposal adaptation is explicit between completed warmup chunks. Every
production chunk freezes its normalized proposal; final model evaluation
disables adaptation. `ProposalMove` carries complete forward/reverse density,
validity, and an optional fixed-shape local payload.

`FullMarkovTarget` and `IncrementalMarkovTarget` are the only root target
contracts. Incremental cache selection follows the same acceptance mask, and a
scheduled exact refresh fails closed on mismatch.

::: phydrax.sampling.ProposalMove

::: phydrax.sampling.RobbinsMonroScalePolicy

::: phydrax.sampling.IncrementalMarkovTarget

::: phydrax.sampling.MarkovChunkPlan

Chunk plans use global semantic transition addresses. The final partial chunk
has an explicit inactive mask, and continuation advances only active draws.

## Hamiltonian kernels

::: phydrax.sampling.prepare_hamiltonian_kernel

::: phydrax.sampling.adapt_hamiltonian_kernel

::: phydrax.sampling.sample_hamiltonian

The target-generic finite HMC/bounded-U-turn kernel uses a declared positive
mass factor and retains divergence, nonfinite-gradient, depth, and
factorization evidence per transition. Adaptation is a finite warmup epoch;
the returned production step size is frozen. Discrete accept/tree decisions
are not differentiable.

## Conditional update programs

`phydrax.sampling.conditional` composes arbitrary PyTree-valued node groups,
batched interaction routes, stateful exact or approximate kernels, and immutable
parallel update stages.

::: phydrax.sampling.conditional.ConditionalVariableGroup

---

::: phydrax.sampling.conditional.ConditionalInteractionGroup

---

::: phydrax.sampling.conditional.AbstractConditionalKernel

---

::: phydrax.sampling.conditional.CallableConditionalKernel

---

::: phydrax.sampling.conditional.MetropolisWithinConditionalKernel

---

::: phydrax.sampling.conditional.ConditionalUpdateStage

---

::: phydrax.sampling.conditional.prepare_conditional_program

---

::: phydrax.sampling.conditional.initialize_conditional_program

---

::: phydrax.sampling.conditional.conditional_program_step

---

::: phydrax.sampling.conditional.sample_conditional_program

## Integration bridge

::: phydrax.integration.markov_chain_measure

`markov_chain_measure` converts retained draws to an equal-weight
`WeightedSampleTarget` with named chain/draw axes, one replicate ID per chain, and
`independent=False`. Consequently `phydrax.integration.integrate` does not report an
IID standard error. Rank-normalized R-hat and effective sample sizes require a frozen
target and remain part of the UQ chain-diagnostic layer.

## Adaptive collocation

`ResidualAttentionCollocation` owns a fixed-capacity population, detached raw-score
EMA, stable point IDs/ages, and an optional fixed-size candidate/replacement route.
Retained points carry their EMA; inserted points initialize from their candidate
score. Stable index ties make replacement replayable. The configured initial
low-discrepancy anchors are never replaced and each retains a strictly positive
probability floor.

Scores define a probability measure with an explicit uniform floor and minimum-ESS
guard. Returned multipliers retain arithmetic mean one. This remains an adaptive
biased training measure, not importance-corrected quadrature; independent fixed
evaluation terms remain required for model selection. Controlled rollback includes
the batch, EMA, IDs, ages, anchors, and candidate-evaluation accounting.

::: phydrax.sampling.collocation.ResidualAttentionCollocation

::: phydrax.sampling.collocation.ResidualAttentionPopulation

## Free-boundary collocation

`CausalTimeSlabSchedule` supplies ordered, optionally overlapping time slabs
and detached causal loss weights. `NarrowBandCollocationPolicy` composes an
ordinary `CollocationPolicy`; it adds a compact level-set band score without
changing the base R3, RAR-D, or periodic population lifecycle.

::: phydrax.sampling.collocation.CausalTimeSlabSchedule

---

::: phydrax.sampling.collocation.NarrowBandCollocationPolicy
