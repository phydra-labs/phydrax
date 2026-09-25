#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Literal

import equinox as eqx
import jax
from jaxtyping import Array

from .._differentiation import (
    _REGULARITY_UNDECLARED,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from .._strict import StrictModule
from .._trainable import ParameterOwner
from ._binding import ModelBinding
from ._component import (
    ComponentPrecisionContract,
    ExecutionCapabilities,
    ModelExecutionContract,
    RandomnessContract,
)
from ._ports import ModelPorts, PortProvider
from ._protocols import (
    MODEL_CONSTRUCTION_CERTIFICATE_KEYS,
    ModelEvaluator,
    ModelMetadataProvider,
)


def value_derivative_contract(
    regularity: DerivativeRegularity | None, /
) -> DerivativeContract:
    """Return the `DIRECT` `INPUT`/`MODEL_PARAMETER` contract of a value regularity.

    Undeclared regularity (`None`) makes both surfaces `CONDITIONAL` on
    `"regularity-undeclared"`. Otherwise both surfaces take the first-order level
    of `regularity.admits_order(1)`: `SMOOTH` within the continuity class and
    `ALMOST_EVERYWHERE` for piecewise maps. A discontinuous map has no `INPUT`
    derivative (`NONE`) while its parameters stay `ALMOST_EVERYWHERE`.
    """
    if regularity is None:
        conditions = (_REGULARITY_UNDECLARED,)
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT,
                    GradientLevel.CONDITIONAL,
                    conditions=conditions,
                ),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER,
                    GradientLevel.CONDITIONAL,
                    conditions=conditions,
                ),
            ),
            route=DerivativeRoute.DIRECT,
        )
    level, _ = regularity.admits_order(1)
    parameter_level = (
        GradientLevel.SMOOTH
        if level is GradientLevel.SMOOTH
        else GradientLevel.ALMOST_EVERYWHERE
    )
    input_level = GradientLevel.NONE if regularity.continuity == -1 else parameter_level
    return DerivativeContract(
        (
            SurfaceDerivative(DerivativeSurface.INPUT, input_level),
            SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, parameter_level),
        ),
        route=DerivativeRoute.DIRECT,
        regularity=regularity,
    )


def native_parameter_precision(tree: Any, /) -> ComponentPrecisionContract | None:
    """Return `ComponentPrecisionContract.native` of the uniform inexact leaf dtype.

    `None` (undeclared) when `tree` holds no inexact array leaf or mixes dtypes.
    Error floors stay undeclared.
    """
    dtypes = {
        leaf.dtype
        for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    }
    return ComponentPrecisionContract.native(dtypes.pop()) if len(dtypes) == 1 else None


class AbstractArrayModel(StrictModule, ModelEvaluator, ParameterOwner):
    """Callable array model with explicit input/output sizes and domain binding.

    Unannotated inexact array leaves are PARAMETER (`ParameterOwner`); subclasses
    declare fixed data such as nodes, training sets and fitted statistics with
    `fixed_field`.
    """

    # JAX hashes callable objects when a module itself is passed to ``jax.jit``.
    # Equinox's structural hash cannot hash array leaves; identity is stable because
    # Phydrax models are immutable and their leaves remain explicit PyTree state.
    __hash__ = object.__hash__

    in_size: eqx.AbstractVar[int | tuple[int, ...] | Literal["scalar"]]
    out_size: eqx.AbstractVar[int | tuple[int, ...] | Literal["scalar"]]

    _input_binding: ClassVar[
        ModelBinding | Callable[["AbstractArrayModel"], ModelBinding]
    ] = ModelBinding.pointwise()

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: Any = None,
    ) -> Array:
        raise NotImplementedError

    def input_binding(self) -> ModelBinding:
        """Return the model's domain input packing and batch execution contract."""
        binding = self._input_binding
        return binding if isinstance(binding, ModelBinding) else binding(self)

    def model_execution_contract(self) -> ModelExecutionContract:
        """Return the model's intrinsic execution contract.

        The default is conservative: value regularity, precision, and randomness
        are undeclared; `INPUT` and `MODEL_PARAMETER` derivatives are
        `CONDITIONAL` on `"regularity-undeclared"` through the `DIRECT` route;
        execution is native JAX with `jit` and `vmap`. Ports come from
        `model_ports()` when the model is a `PortProvider`, and certificates from
        the construction-certificate entries of `model_metadata()` when it is a
        `ModelMetadataProvider`. Families with declared regularity override it.
        """
        return self._execution_contract(value_derivative_contract(None))

    def _execution_contract(
        self,
        derivative: DerivativeContract,
        /,
        *,
        precision: ComponentPrecisionContract | None = None,
        randomness: RandomnessContract | None = None,
        execution: ExecutionCapabilities | None = None,
    ) -> ModelExecutionContract:
        """Contract with `derivative` and this model's ports and certificates.

        `execution` defaults to native JAX; external model tiers pass their
        declared capabilities.
        """
        metadata = (
            self.model_metadata() if isinstance(self, ModelMetadataProvider) else {}
        )
        return ModelExecutionContract(
            derivative=derivative,
            execution=ExecutionCapabilities("native-jax")
            if execution is None
            else execution,
            precision=precision,
            randomness=randomness,
            ports=self._contract_ports(),
            certificates=tuple(
                metadata[key]
                for key in sorted(MODEL_CONSTRUCTION_CERTIFICATE_KEYS)
                if key in metadata
            ),
        )

    def _contract_ports(self) -> ModelPorts | None:
        """Intrinsic ports recorded in the execution contract (`None` if undeclared)."""
        return self.model_ports() if isinstance(self, PortProvider) else None


__all__ = ["AbstractArrayModel"]
