from matplotlib import _tri
from matplotlib.tri._trifinder import TriFinder
import matplotlib._mlx_array as mlxarr
from matplotlib._mlx_typing import ArrayLike
from typing import Any

class Triangulation:
    x: mlxarr.ndarray
    y: mlxarr.ndarray
    mask: mlxarr.ndarray | None
    is_delaunay: bool
    triangles: mlxarr.ndarray
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        triangles: ArrayLike | None = ...,
        mask: ArrayLike | None = ...,
    ) -> None: ...
    def calculate_plane_coefficients(self, z: ArrayLike) -> mlxarr.ndarray: ...
    @property
    def edges(self) -> mlxarr.ndarray: ...
    def get_cpp_triangulation(self) -> _tri.Triangulation: ...
    def get_masked_triangles(self) -> mlxarr.ndarray: ...
    @staticmethod
    def get_from_args_and_kwargs(
        *args, **kwargs
    ) -> tuple[Triangulation, tuple[Any, ...], dict[str, Any]]: ...
    def get_trifinder(self) -> TriFinder: ...
    @property
    def neighbors(self) -> mlxarr.ndarray: ...
    def set_mask(self, mask: None | ArrayLike) -> None: ...
