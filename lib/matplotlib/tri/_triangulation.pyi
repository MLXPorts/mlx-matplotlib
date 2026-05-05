from matplotlib import _tri
from matplotlib.tri._trifinder import TriFinder
import mlx.core as mx
from matplotlib._mlx_typing import ArrayLike
from typing import Any

class Triangulation:
    x: mx.array
    y: mx.array
    mask: mx.array | None
    is_delaunay: bool
    triangles: mx.array
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        triangles: ArrayLike | None = ...,
        mask: ArrayLike | None = ...,
    ) -> None: ...
    def calculate_plane_coefficients(self, z: ArrayLike) -> mx.array: ...
    @property
    def edges(self) -> mx.array: ...
    def get_cpp_triangulation(self) -> _tri.Triangulation: ...
    def get_masked_triangles(self) -> mx.array: ...
    @staticmethod
    def get_from_args_and_kwargs(
        *args, **kwargs
    ) -> tuple[Triangulation, tuple[Any, ...], dict[str, Any]]: ...
    def get_trifinder(self) -> TriFinder: ...
    @property
    def neighbors(self) -> mx.array: ...
    def set_mask(self, mask: None | ArrayLike) -> None: ...
