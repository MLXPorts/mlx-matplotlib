from typing import Literal, overload
import matplotlib._mlx_array as mlxarr
from matplotlib._mlx_typing import ArrayLike

from matplotlib.tri._triangulation import Triangulation
from matplotlib.tri._triinterpolate import TriInterpolator

class TriRefiner:
    def __init__(self, triangulation: Triangulation) -> None: ...

class UniformTriRefiner(TriRefiner):
    def __init__(self, triangulation: Triangulation) -> None: ...
    @overload
    def refine_triangulation(
        self, *, return_tri_index: Literal[True], subdiv: int = ...
    ) -> tuple[Triangulation, mlxarr.ndarray]: ...
    @overload
    def refine_triangulation(
        self, return_tri_index: Literal[False] = ..., subdiv: int = ...
    ) -> Triangulation: ...
    @overload
    def refine_triangulation(
        self, return_tri_index: bool = ..., subdiv: int = ...
    ) -> tuple[Triangulation, mlxarr.ndarray] | Triangulation: ...
    def refine_field(
        self,
        z: ArrayLike,
        triinterpolator: TriInterpolator | None = ...,
        subdiv: int = ...,
    ) -> tuple[Triangulation, mlxarr.ndarray]: ...
