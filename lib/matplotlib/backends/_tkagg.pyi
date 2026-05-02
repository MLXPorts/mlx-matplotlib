import matplotlib._mlx_array as mlxarr
from matplotlib._mlx_typing import NDArray

TK_PHOTO_COMPOSITE_OVERLAY: int
TK_PHOTO_COMPOSITE_SET: int

def blit(
    interp: int,
    photo_name: str,
    data: NDArray[mlxarr.uint8],
    comp_rule: int,
    offset: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
) -> None: ...
def enable_dpi_awareness(frame_handle: int, interp: int) -> bool | None: ...
