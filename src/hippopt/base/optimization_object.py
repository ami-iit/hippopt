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
    StorageType: ClassVar[str] = "generic"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)

    def get_default_initialization(
        self: TOptimizationObject, field_name: str
    ) -> np.ndarray:
        """
        Get the default initialization of a given field
        It is supposed to be called only for the fields having the StorageType metadata
        """
        return np.zeros(dataclasses.asdict(self)[field_name].shape)

    def get_default_initialized_object(self: TOptimizationObject) -> TOptimizationObject:
        """
        :return: A copy of the object with its initial values
        """

        output = copy.deepcopy(self)
        output_dict = dataclasses.asdict(output)

        for field in dataclasses.fields(output):
            if "StorageType" in field.metadata:
                output.__setattr__(
                    field.name, output.get_default_initialization(field.name)
                )
                continue

            if isinstance(output.__getattribute__(field.name), OptimizationObject):
                output.__setattr__(
                    field.name, output.__getattribute__(field.name).get_default_initialized_object()
                )

        return output


def default_storage_type(input_type: Type[OptimizationObject]):
    return dataclasses.field(
        default=None,
        metadata=input_type.StorageTypeMetadata,
    )
