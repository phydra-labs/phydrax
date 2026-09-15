# Production QFT substrates

The public surface is distributed across canonical numerical owners. DLR and
variable-sector modules remain explicit public submodules to preserve the
acyclic root import boundary:

```python
from phydrax.discretization import dlr
from phydrax.operators.quantum import variable_sector
from phydrax.nn.quantum import variable_sector as variable_amplitudes
from phydrax.backends import lattice as lattice_backends
```

## Lattice structure

::: phydrax.discretization.LatticeBoundaryPhasePlan

::: phydrax.discretization.CheckerboardEntityLayout

::: phydrax.discretization.LatticeDecompositionPlan

::: phydrax.discretization.LatticeHaloPlan

::: phydrax.discretization.LatticeStencilExecutionPlan

::: phydrax.discretization.dlr

## Gauge geometry and entanglement

::: phydrax.metrix.AbstractGaugeRepresentation

::: phydrax.metrix.FundamentalGaugeRepresentation

::: phydrax.metrix.AdjointGaugeRepresentation

::: phydrax.metrix.GaussianEntanglementPlan

::: phydrax.graph.GaugeCovariantShiftPlan

::: phydrax.graph.GaugeStaplePlan

## Euclidean actions and fermions

::: phydrax.operators.path_integral.ImprovedGaugeAction

::: phydrax.operators.path_integral.GaugeGradientFlowPlan

::: phydrax.operators.path_integral.AbstractLatticeDiracOperator

::: phydrax.operators.path_integral.WilsonDiracOperator

::: phydrax.operators.path_integral.CloverWilsonDiracOperator

::: phydrax.operators.path_integral.StaggeredDiracOperator

::: phydrax.operators.path_integral.NaikStaggeredDiracOperator

::: phydrax.operators.path_integral.DomainWallDiracOperator

::: phydrax.operators.path_integral.OverlapDiracOperator

::: phydrax.operators.path_integral.CertifiedRationalApproximation

::: phydrax.operators.path_integral.TwoFlavorPseudofermionTerm

::: phydrax.operators.path_integral.FractionalPowerPseudofermionTerm

Pseudofermion terms expose independent refresh, proposal-action, force, and
acceptance solve policies. Eligible normal-operator systems may opt into
streaming three-term shifted Lanczos with direct residual and shifted-solution
error evidence. Force differentiation stops the solve vectors and
differentiates the physical normal-operator action; no Krylov recurrence or
force-error certificate is implied.

## Gauge and learned sampling

::: phydrax.sampling.GaugeUpdatePlan

::: phydrax.sampling.SplitGroupDynamicsPlan

::: phydrax.sampling.RHMCPlan

::: phydrax.sampling.DelayedAcceptanceHMCPlan

::: phydrax.sampling.GaugeFlowProposalPlan

## Integration and complex weights

::: phydrax.integration.VegasPlan

::: phydrax.integration.ComplexWeightMeasure

::: phydrax.integration.PhaseQuenchedReweightingPlan

## Hamiltonian and thermal operators

::: phydrax.operators.quantum.FermionicFockBasis

::: phydrax.operators.quantum.TruncatedU1LinkHilbertSpace

::: phydrax.operators.quantum.SU2IrrepTruncatedLinkHilbertSpace

::: phydrax.operators.quantum.GaussConstraintNetwork

::: phydrax.operators.quantum.ThermalLehmannPlan

::: phydrax.operators.quantum.DLRGreenFunction

::: phydrax.operators.quantum.PadeContinuationPlan

::: phydrax.operators.quantum.MaximumEntropyPlan

::: phydrax.operators.quantum.SparseContinuationPlan

::: phydrax.operators.quantum.variable_sector

## Tensor representations

::: phydrax.tensor_network.AbelianFusionBasis

::: phydrax.tensor_network.FiniteFusionCategory

::: phydrax.tensor_network.StringNetPlan

::: phydrax.tensor_network.AnyonicTensor

## Ensemble and variable-sector solvers

::: phydrax.solver.DeterministicEnsemblePlan

::: phydrax.solver.VariableSectorVMCPlan

::: phydrax.solver.VariableSectorTDVPPlan

::: phydrax.nn.quantum.variable_sector

## Lattice kernel providers

::: phydrax.backends.lattice
