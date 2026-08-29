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

The initial implementation deliberately uses fixed kernels. Proposal adaptation and
parameter updates are separate phases rather than hidden mutations of production-chain
stationarity.

## Integration bridge

::: phydrax.integration.markov_chain_measure

`markov_chain_measure` converts retained draws to an equal-weight
`WeightedSampleTarget` with named chain/draw axes, one replicate ID per chain, and
`independent=False`. Consequently `phydrax.integration.integrate` does not report an
IID standard error. Rank-normalized R-hat and effective sample sizes require a frozen
target and remain part of the UQ chain-diagnostic layer.

## Free-boundary collocation

`CausalTimeSlabSchedule` supplies ordered, optionally overlapping time slabs
and detached causal loss weights. `NarrowBandCollocationPolicy` composes an
ordinary `CollocationPolicy`; it adds a compact level-set band score without
changing the base R3, RAR-D, or periodic population lifecycle.

::: phydrax.sampling.collocation.CausalTimeSlabSchedule

---

::: phydrax.sampling.collocation.NarrowBandCollocationPolicy
