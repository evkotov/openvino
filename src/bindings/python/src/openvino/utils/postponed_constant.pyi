# type: ignore
from __future__ import annotations
from collections.abc import Callable
from openvino._op_base import Op
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Type
import collections.abc
import openvino._op_base
import openvino._pyopenvino
__all__ = ['Callable', 'Op', 'PartialShape', 'PostponedConstant', 'Shape', 'Tensor', 'Type', 'make_postponed_constant']
class PostponedConstant(openvino._op_base.Op):
    """
    Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose.
    """
    def __init__(self, element_type: openvino._pyopenvino.Type, shape: openvino._pyopenvino.Shape, maker: collections.abc.Callable[[openvino._pyopenvino.Tensor], None], name: typing.Optional[str] = None) -> None:
        ...
    def clone_with_new_inputs(self, new_inputs: list[openvino._pyopenvino.Tensor]) -> openvino._op_base.Op:
        ...
    def evaluate(self, outputs: list[openvino._pyopenvino.Tensor], _: list[openvino._pyopenvino.Tensor]) -> bool:
        ...
    def has_evaluate(self) -> bool:
        ...
    def validate_and_infer_types(self) -> None:
        ...
def make_postponed_constant(element_type: openvino._pyopenvino.Type, shape: openvino._pyopenvino.Shape, maker: collections.abc.Callable[[openvino._pyopenvino.Tensor], None], name: typing.Optional[str] = None) -> openvino._op_base.Op:
    ...
