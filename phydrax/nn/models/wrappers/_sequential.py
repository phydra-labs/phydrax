#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
from jax import Array

from ...._differentiation import DerivativeRegularity
from ...._doc import DOC_KEY0
from ...._model import (
    ModelBinding,
    ModelMetadataProvider,
    PERIODIC_INPUT_CERTIFICATE_KEY,
    PeriodicInputCertificate,
)
from ..._base import _AbstractBaseModel, _AbstractStructuredInputModel
from ..._contracts import compose_regularity, model_regularity
from ..._keys import EvalKey, split_eval_key
from ..._utils import _canonical_size, _get_size


class Sequential(_AbstractStructuredInputModel):
    r"""Compose models in sequence.

    Given models $(m_1,\dots,m_K)$, this wrapper evaluates

    $$
    y(x)=m_K(\cdots m_2(m_1(x))\cdots).
    $$

    Adjacent models must have compatible sizes:
    `canonical(prev.out_size) == canonical(next.in_size)`.

    Domain inputs are packed as the first stage packs them: flat stages receive
    the concatenated dependency vector and structured stages a tuple.
    """

    models: tuple[_AbstractBaseModel, ...]
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(self, models: Sequence[_AbstractBaseModel]) -> None:
        if not models:
            raise ValueError("Sequential requires at least one model.")

        models_t = tuple(models)
        for idx, model in enumerate(models_t):
            binding = model.input_binding()
            if binding.batch_mode != "pointwise":
                raise ValueError(
                    f"Sequential stage {idx} must use pointwise binding; "
                    f"got {binding.batch_mode!r}."
                )
            if idx > 0 and binding.input_mode != "flat":
                raise ValueError(
                    f"Sequential stage {idx} consumes one array output and must "
                    "declare flat input binding."
                )
        for idx in range(1, len(models_t)):
            prev = models_t[idx - 1]
            curr = models_t[idx]
            prev_out = _canonical_size(prev.out_size)
            curr_in = _canonical_size(curr.in_size)
            if prev_out != curr_in:
                raise ValueError(
                    "Sequential size mismatch at stage "
                    f"{idx - 1}->{idx}: prev.out_size={prev.out_size!r} "
                    f"is not compatible with curr.in_size={curr.in_size!r}."
                )

        self.models = models_t
        self.in_size = _canonical_size(models_t[0].in_size)
        self.out_size = _canonical_size(models_t[-1].out_size)

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        r"""Evaluate the model pipeline.

        If tuple input is provided, the first stage must support structured input.
        """
        if len(self.models) == 1:
            keys = (key,)
        else:
            keys = split_eval_key(key, len(self.models))

        first_model = self.models[0]
        if isinstance(x, tuple):
            if first_model.input_binding().input_mode != "structured":
                raise TypeError(
                    "Sequential received tuple input, but the first stage does not support structured inputs."
                )
            first_input = x
        else:
            first_input = jnp.asarray(x)
        y = first_model.input_binding().call(
            first_model, first_input, key=keys[0], iter_=iter_, kwargs={}
        )

        for model, subkey in zip(self.models[1:], keys[1:], strict=True):
            y = model.input_binding().call(
                model, jnp.asarray(y), key=subkey, iter_=iter_, kwargs={}
            )
        return jnp.asarray(y)

    def input_binding(self) -> ModelBinding:
        """Use first-stage packing and forward each stage's invocation requirements."""
        return ModelBinding.pointwise(
            self.models[0].input_binding().input_mode,
            pass_iter=any(model.input_binding().pass_iter for model in self.models),
        )

    def _value_regularity(self) -> DerivativeRegularity | None:
        return compose_regularity(*(model_regularity(model) for model in self.models))

    def model_metadata(self) -> Mapping[str, Any]:
        """Propagate the first stage's periodic-input certificate.

        Later stages see only the first stage's output, which is invariant under
        the certified input translations, so the composite inherits that
        periodicity. The composite's declared regularity bounds the certified
        derivative orders; undeclared composite regularity attaches nothing.
        Other construction certificates describe a single stage and are not
        propagated.
        """
        first = self.models[0]
        if not isinstance(first, ModelMetadataProvider):
            return {}
        certificate = first.model_metadata().get(PERIODIC_INPUT_CERTIFICATE_KEY)
        if certificate is None:
            return {}
        if not isinstance(certificate, PeriodicInputCertificate):
            raise TypeError(
                f"Metadata key {PERIODIC_INPUT_CERTIFICATE_KEY!r} must hold a "
                f"PeriodicInputCertificate, got {type(certificate).__name__}."
            )
        if certificate.input_size != _get_size(self.in_size):
            raise ValueError(
                "First-stage periodic certificate does not match Sequential input size."
            )
        regularity = compose_regularity(certificate.regularity, self._value_regularity())
        if regularity is None:
            return {}
        for model in self.models:
            randomness = model.model_execution_contract().randomness
            if (
                randomness is None
                or randomness.requires_inference_state
                or randomness.mode not in ("deterministic", "fixed-realization")
            ):
                return {}
        return {
            PERIODIC_INPUT_CERTIFICATE_KEY: PeriodicInputCertificate(
                input_size=certificate.input_size,
                periodic_inputs=certificate.periodic_inputs,
                regularity=regularity,
            )
        }


__all__ = ["Sequential"]
