import dataclasses
from typing import TypeVar

import casadi as cs

from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor

TTerrainSum = TypeVar("TTerrainSum", bound="TerrainSum")


class TerrainSum(TerrainDescriptor):
    _lhs: TerrainDescriptor = dataclasses.field(default=None)
    _rhs: TerrainDescriptor = dataclasses.field(default=None)

    def set_terms(self, lhs: TerrainDescriptor, rhs: TerrainDescriptor):
        self._lhs = lhs
        self._rhs = rhs

    def create_height_function(self) -> cs.Function:
        assert (
            self._lhs is not None and self._rhs is not None
        ), "Both lhs and rhs must be provided"

        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        # We need to subtract the point height because otherwise is counted twice
        return cs.Function(
            "terrain_sum_height",
            [point_position],
            [
                self._lhs.height_function()(point_position)
                + self._rhs.height_function()(point_position)
                - point_position[2]
            ],
            [self.get_point_position_name()],
            ["point_height"],
            self._options,
        )

    @classmethod
    def add(cls, lhs: TerrainDescriptor, rhs: TerrainDescriptor) -> TTerrainSum:
        output = cls(
            point_position_name=lhs.get_point_position_name(),
            options=lhs._options,
            name=f"{lhs.get_name()}+{rhs.get_name()}",
        )
        output.set_terms(lhs, rhs)

        return output

    def __add__(self, other: TTerrainSum) -> TTerrainSum:
        return TerrainSum.add(self, other)

    def __radd__(self, other: TTerrainSum) -> TTerrainSum:
        return TerrainSum.add(other, self)
