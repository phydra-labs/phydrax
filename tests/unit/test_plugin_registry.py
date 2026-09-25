#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

import phydrax as phx


class _RegisteredEngine:
    pass


class _DerivedEngine(_RegisteredEngine):
    pass


class _OtherEngine:
    pass


def _activation(value):
    return value


def _same_named_activation(value):
    return value


_same_named_activation.__qualname__ = _activation.__qualname__
_same_named_activation.__name__ = _activation.__name__


def test_plugin_registration_is_exported_once_at_the_root():
    assert "OperatorArchitectureCodec" in phx.__all__
    assert "register_operator_architecture_codec" in phx.__all__
    assert "register_artifact_value" in phx.__all__
    assert not hasattr(phx.nn.operator.training, "OperatorArchitectureCodec")
    assert not hasattr(phx.nn.operator.training, "register_operator_architecture_codec")


def test_architecture_codecs_resolve_by_exact_type_and_explicit_id():
    codec = phx.register_operator_architecture_codec(
        phx.OperatorArchitectureCodec("test.registry:engine-v1", _RegisteredEngine)
    )

    assert phx.operator_architecture_codec_for(_RegisteredEngine()) is codec
    assert phx.operator_architecture_codec("test.registry:engine-v1") is codec
    assert phx.artifact_value("test.registry:engine-v1") is _RegisteredEngine
    # Registration is idempotent for the identical codec.
    assert phx.register_operator_architecture_codec(codec) is codec
    # A subclass is another architecture: no isinstance fallback.
    with pytest.raises(TypeError, match="registered architecture codec"):
        phx.operator_architecture_codec_for(_DerivedEngine())
    # No lookup by class name or entry point: unknown IDs fail closed.
    with pytest.raises(ValueError, match="Unknown operator architecture ID"):
        phx.operator_architecture_codec("_RegisteredEngine")


def test_architecture_codec_registration_rejects_ambiguous_identities():
    phx.register_operator_architecture_codec(
        phx.OperatorArchitectureCodec("test.registry:ambiguous-v1", _OtherEngine)
    )
    with pytest.raises(ValueError, match="already registered"):
        phx.register_operator_architecture_codec(
            phx.OperatorArchitectureCodec("test.registry:ambiguous-v1", _DerivedEngine)
        )
    with pytest.raises(ValueError, match="already has architecture ID"):
        phx.register_operator_architecture_codec(
            phx.OperatorArchitectureCodec("test.registry:ambiguous-v2", _OtherEngine)
        )


def test_artifact_values_resolve_by_object_identity():
    phx.register_artifact_value("test.registry:activation", _activation)

    assert phx.artifact_value_id(_activation) == "test.registry:activation"
    assert phx.artifact_value("test.registry:activation") is _activation
    # A distinct object with the same module and qualified name is unregistered.
    with pytest.raises(TypeError, match="explicit canonical registration"):
        phx.artifact_value_id(_same_named_activation)
    with pytest.raises(ValueError, match="already registered"):
        phx.register_artifact_value("test.registry:activation", _same_named_activation)
    with pytest.raises(ValueError, match="already has identity"):
        phx.register_artifact_value("test.registry:activation-alias", _activation)
    with pytest.raises(ValueError, match="Unknown artifact value ID"):
        phx.artifact_value("test.registry:missing")
