#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod

import equinox as eqx
import pytest

from phydrax._strict import StrictModule


class AbstractValue(StrictModule):
    value: eqx.AbstractVar[int]

    @abstractmethod
    def evaluate(self) -> int:
        raise NotImplementedError


class Value(AbstractValue):
    value: int

    def evaluate(self) -> int:
        return self.value


def test_abstract_var_and_method_resolve_in_one_final_class():
    with pytest.raises(TypeError, match="abstract"):
        AbstractValue()

    value = Value(3)
    assert value.evaluate() == 3
    with pytest.raises(AttributeError):
        value.value = 4
    with pytest.raises(TypeError, match=r"concrete \(final\)"):

        class InvalidChild(Value):
            pass


def test_abstract_property_can_be_implemented():
    class AbstractPropertyContract(StrictModule):
        @property
        @abstractmethod
        def label(self) -> str:
            raise NotImplementedError

    class PropertyImplementation(AbstractPropertyContract):
        @property
        def label(self) -> str:
            return "implemented"

    assert PropertyImplementation().label == "implemented"
