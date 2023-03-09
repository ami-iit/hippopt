from hippopt.common import (
    Any,
    ClassVar,
    Type,
    TypeVar,
    Union,
    abc,
    copy,
    cs,
    dataclasses,
    np,
)

TOptimizationObject = TypeVar("TOptimizationObject", bound="OptimizationObject")
StorageType = Union[cs.MX, np.ndarray]


@dataclasses.dataclass
class OptimizationObject(abc.ABC):
    ObjectType: ClassVar[str] = "optimization_object"
    ObjectTypeMetadata: ClassVar[dict[str, Any]] = dict(ObjectType=ObjectType)

    def zero_copy_on_type_condition(
        self: TOptimizationObject, type_str: str
    ) -> TOptimizationObject:
        # Operate on a deep copy
        param = copy.deepcopy(self)
        param_dict = dataclasses.asdict(param)

        for field in dataclasses.fields(param):
            if field.metadata.get("ObjectType", "") is not type_str:
                continue

            shape = param_dict[field.name].shape

            if isinstance(param_dict[field.name], np.ndarray):
                param.__setattr__(field.name, np.zeros(shape))
            elif isinstance(param_dict[field.name], (cs.MX, cs.SX)):
                param.__setattr__(field.name, cs.MX.zeros(*shape))
            else:
                raise TypeError(type(param_dict[field.name]))

        return param


def default_storage_type(input_type: Type[OptimizationObject]):
    return dataclasses.field(
        default=None,
        metadata=input_type.ObjectTypeMetadata,
    )
