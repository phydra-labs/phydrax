# Tensor-network platform

See [Tensor platform](../guides_tensor_platform.md) for representation selection,
precision, approximation, resource, physicality, persistence, and maturity contracts.

## Finite chains

::: phydrax.tensor_network.MatrixProductState

::: phydrax.tensor_network.MatrixProductOperator

::: phydrax.tensor_network.PreparedChainEnvironments

::: phydrax.tensor_network.FiniteLocalTerm

::: phydrax.tensor_network.mps_local_observable_expectation

::: phydrax.tensor_network.lpdo_local_observable_expectation

::: phydrax.tensor_network.FiniteThermalPolicy

::: phydrax.tensor_network.UniformMatrixProductState

## Tensor topology and execution

::: phydrax.tensor_network.ContractionStructure

::: phydrax.tensor_network.ContractionPlannerPolicy

::: phydrax.tensor_network.ContractionPlan

::: phydrax.tensor_network.SlicedContractionPlan

::: phydrax.tensor_network.SlicePlacementPlan

## Thermodynamic tensor renormalization

::: phydrax.tensor_network.UniformSquareTensor

::: phydrax.tensor_network.build_uniform_pair_partition_tensor

::: phydrax.tensor_network.UniformSquareTensorBuildResult


::: phydrax.tensor_network.TRGMethod

::: phydrax.tensor_network.HOTRGMethod

::: phydrax.tensor_network.TensorRenormalizationProblem

::: phydrax.tensor_network.TensorRenormalizationPolicy

::: phydrax.tensor_network.TensorRenormalizationResourcePolicy

::: phydrax.tensor_network.TensorRenormalizationPlan

::: phydrax.tensor_network.TensorRenormalizationStatus

::: phydrax.tensor_network.TensorRenormalizationDiagnostics


::: phydrax.tensor_network.PreparedTensorRenormalization

::: phydrax.tensor_network.TensorRenormalizationResult

::: phydrax.tensor_network.plan_tensor_renormalization

::: phydrax.tensor_network.prepare_tensor_renormalization

::: phydrax.tensor_network.refresh_tensor_renormalization

::: phydrax.tensor_network.run_tensor_renormalization

## Two-dimensional and hierarchical networks

::: phydrax.tensor_network.PEPS

::: phydrax.tensor_network.BoundaryMPSPolicy

::: phydrax.tensor_network.CTMRGPolicy

::: phydrax.tensor_network.PEPSUpdatePolicy

::: phydrax.tensor_network.TreeTensorNetwork

::: phydrax.tensor_network.NetworkBPPolicy

::: phydrax.tensor_network.BinaryMERA

## Symmetry and statistics

::: phydrax.operators.quantum.AbelianGroup

::: phydrax.tensor_network.AbelianContractionPlan

::: phydrax.tensor_network.FermionGrading

::: phydrax.tensor_network.FermionModeOrder

::: phydrax.tensor_network.RepresentationCategory

::: phydrax.tensor_network.SU2ReducedTensor

## Open systems

::: phydrax.tensor_network.LocallyPurifiedDensity

::: phydrax.tensor_network.AbelianLPDO

::: phydrax.tensor_network.ChargeCovariantKrausMap

## Platform envelope

::: phydrax.tensor_network.TensorNetworkSupportTuple

::: phydrax.tensor_network.TensorNetworkResourcePolicy

::: phydrax.tensor_network.TensorNetworkExecutionManifest

::: phydrax.tensor_network.TensorNetworkAcceptedCheckpointBoundary

::: phydrax.tensor_network.TensorNetworkQualificationProfile

::: phydrax.tensor_network.TensorNetworkInterchangeManifest

## Compact prefix-square operators

::: phydrax.tensor_network.build_prefix_quadratic_mpo

::: phydrax.tensor_network.PrefixQuadraticMPOResult

`build_prefix_quadratic_mpo` represents
`sum_n weight_n (offset_n + sum_{j <= n} Q_j)^2` exactly with maximum
prefix-sector bond dimension three. It is used by the open Schwinger-chain
application without expanding all long-range charge pairs.
