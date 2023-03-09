from hippopt.base.optimization_object import OptimizationObject
from hippopt.common import Any, ClassVar, dataclasses


@dataclasses.dataclass
class Parameter(OptimizationObject):
    """"""

    ObjectType: ClassVar[str] = "parameter"
    ObjectTypeMetadata: ClassVar[dict[str, Any]] = dict(ObjectType=ObjectType)

    def zero_copy(self):
        return self.zero_copy_on_type_condition(self.ObjectType)
