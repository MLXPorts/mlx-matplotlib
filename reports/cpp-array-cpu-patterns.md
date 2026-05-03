# C++ Array/Vector CPU-Only Pattern Report

Date: 2026-05-03
Branch: `codex/remove-numpy-naming`
Repo: `MLXPorts/mlx-matplotlib`

## Executive Status

The C++ tree is not using the NumPy C API directly anymore. A scan of `src/`
found no active `PyArray_*`, `arrayobject.h`, or `NPY_*` integration beyond one
comment in `src/mplutils.h`. The current C++ port has therefore cleared the old
"compile against NumPy" dependency surface.

The larger GPU/MLX problem remains. `_image`, `_path`, and `py_converters`
include MLX headers today:

- `src/_image_wrapper.cpp`
- `src/_path_wrapper.cpp`
- `src/py_converters.cpp`

Most other extension boundaries still speak Python buffer protocol, construct
`memoryview`/`bytearray` outputs, or copy through `std::vector`. That means many
MLX arrays are accepted only where the Python object happens to expose a CPU
buffer or where Python has already converted them to a buffer-like object. Those
paths do not preserve device residency and generally require CPU materialization.

## Scan Summary

Direct NumPy C/C++ API:

- No active `PyArray`, `arrayobject`, or `py::array` matches in `src/`.
- `src/mplutils.h:53` has a comment mentioning `NPY_INTP_FMT`; not a live API.

Current MLX C++ integration:

- `src/_image_wrapper.cpp:18-21` includes `mlx/array.h`, `mlx/ops.h`,
  `mlx/stream.h`, and `mlx/utils.h`.
- `src/_path_wrapper.cpp` now includes the same MLX headers for the
  `affine_transform(..., stream=...)` boundary.
- `src/py_converters.cpp:8-11` includes the same MLX headers.
- `src/_mlx_overrides.cpp` now owns the precision-sensitive Python-float to
  MLX-float64 scalar boundary for scalar mutation, scalar arithmetic, `full`,
  `full_like`, and constant `pad`. It feeds MLX an exact native-double buffer
  before reshaping to a scalar, which avoids the installed MLX scalar/update
  paths that were rounding the Python float.
- `src/meson.build:96-118` builds `_image` with `nanobind`, `mlx`, `NB_DOMAIN=mlx`,
  `NB_STATIC`, MLX include/lib dirs, and an MLX runtime rpath.
- `src/meson.build` now builds `_path` with the same MLX/nanobind include,
  compile-definition, and rpath wiring, and includes `py_converters.cpp` so the
  build process owns the stream-aware affine conversion.
- `src/meson.build` also builds `_mlx_overrides` as part of the local and CI
  extension build, so the precision fix is not a manual step.

Pattern counts in `src/`:

- `py::buffer`: present in 15 C++/header files. Largest counts:
  `_backend_agg_wrapper.cpp` 22, `tri/_tri.cpp` 11,
  `_path_wrapper.cpp` 9, `py_converters.h` 8, `tri/_tri.h` 8.
- `memoryview` / `buffer_info` / `PyByteArray` / `memcpy`: present in 8 files.
  Largest counts: `tri/_tri.cpp` 20, `ft2font_wrapper.cpp` 16,
  `_image_wrapper.cpp` 8, `_path_wrapper.cpp` 8.
- `std::vector`, `std::array`, raw `new`/`delete`, and `.data()` CPU pointer use:
  present in 19 files. Largest counts: `tri/_tri.cpp` 60, `tri/_tri.h` 26,
  `_path_wrapper.cpp` 24, `_path.h` 18, `ft2font_wrapper.cpp` 18.

## What Is Already MLX-Aware

### `_image.resample` input and affine conversion

`src/_image_wrapper.cpp` has the strongest MLX path right now.

- `get_array_info` detects `mx::array` or `__mlx_array__` objects at
  `src/_image_wrapper.cpp:158-174`.
- It stages MLX array inputs with `mx::contiguous(..., stream)` for explicit
  streams at `src/_image_wrapper.cpp:194-198`.
- It evaluates and synchronizes the specified stream before taking a data
  pointer at `src/_image_wrapper.cpp:199-207`.
- The Python API exposes `stream=` at `src/_image_wrapper.cpp:525-536`.

`src/py_converters.cpp` has a narrower MLX-aware affine conversion path:

- It detects and casts MLX arrays at `src/py_converters.cpp:60-76`.
- It parses `Stream`, `Device`, and `DeviceType` into `mx::StreamOrDevice` at
  `src/py_converters.cpp:78-112`.
- `convert_mlx_affine` makes a contiguous MLX copy on the provided stream,
  evaluates, synchronizes, and then reads the 3x3 matrix at
  `src/py_converters.cpp:114-151`.

This is useful, but it is still a staging bridge into CPU pointer algorithms.
The resampler itself remains an AGG CPU routine, not an MLX kernel.

## Highest-Severity CPU Pretenders

### 1. `py_buffer.h` is a central CPU-only abstraction

`src/py_buffer.h:16-142` defines `mpl::BufferView<T, ND>` entirely in terms of
`py::buffer.request()`, PEP 3118 formats, raw host pointers, shapes, and strides.
Nearly every C++ extension uses it.

Important details:

- Construction calls `buf.request(writable)` at `src/py_buffer.h:21-23`.
- Data access is raw host memory via `reinterpret_cast` at
  `src/py_buffer.h:58-80`.
- Element access is pointer arithmetic over the requested buffer at
  `src/py_buffer.h:83-135`.

Impact:

- This is the main reason C++ call sites keep pretending MLX arrays are CPU
  arrays.
- Any MLX object that arrives here must already be CPU-readable through Python's
  buffer protocol, or the call fails.
- There is no stream/device parameter, no `mx::array`, and no `mx::to_stream`.

Recommended replacement shape:

- Introduce a real MLX-backed view/converter for extension inputs, not another
  Python wrapper shim.
- Accept `mx::array` at the binding boundary via nanobind.
- Thread `mx::StreamOrDevice` or `mx::Stream` through each public C++ entrypoint
  that reads or writes array data.
- Only expose a host pointer after explicit, local, documented staging to CPU is
  unavoidable because a CPU library such as AGG, FreeType, Tk, or Qhull requires it.

### 2. Agg renderer boundary is still buffer-only

`src/_backend_agg_wrapper.cpp` is still the largest non-image buffer surface.

Buffer-only inputs:

- `draw_text_image`: `const py::buffer &image_obj` at
  `src/_backend_agg_wrapper.cpp:58-94`.
- `draw_image`: `const py::buffer &image_obj` at
  `src/_backend_agg_wrapper.cpp:116-130`.
- `draw_path_collection`: transforms, offsets, facecolors, edgecolors,
  linewidths, antialiaseds, and hatch colors as `py::buffer` at
  `src/_backend_agg_wrapper.cpp:133-181`.
- `draw_quad_mesh`: coordinates, offsets, facecolors, edgecolors as `py::buffer`
  at `src/_backend_agg_wrapper.cpp:184-212`.
- `draw_gouraud_triangles`: points and colors as `py::buffer` at
  `src/_backend_agg_wrapper.cpp:214-224`.

Host-buffer outputs:

- `RendererAgg` exposes its framebuffer through `py::buffer_info` at
  `src/_backend_agg_wrapper.cpp:265-276`.
- `BufferRegion` exposes saved regions through `py::buffer_info` at
  `src/_backend_agg_wrapper.cpp:279-295`.

Backing storage:

- `RendererAgg` owns raw `pixBuffer`, `hatchBuffer`, and `alphaBuffer` allocated
  with `new[]` at `src/_backend_agg.cpp:42-67`.
- `BufferRegion` owns a raw `agg::int8u*` allocated with `new[]` at
  `src/_backend_agg.h:54-68`.
- Marker rendering stores serialized scanlines in `std::vector<agg::int8u>` and
  feeds `.data()` back to AGG at `src/_backend_agg.h:538-648`.

Python bridge forces CPU buffers before C++:

- `_BufferPath` stores path vertices/codes as `memoryview` in
  `lib/matplotlib/backends/backend_agg.py:42-49`.
- `_plain_float_buffer` converts values with `mlxarr.asarray(...).tolist()`, then
  creates an `array("d")` memoryview at `lib/matplotlib/backends/backend_agg.py:61-83`.
- `draw_path_collection` and `draw_quad_mesh` call `_plain_float_buffer` for
  transforms, offsets, colors, linewidths, and mesh coordinates at
  `lib/matplotlib/backends/backend_agg.py:228-265`.
- `buffer_rgba` returns `memoryview(self._renderer)` at
  `lib/matplotlib/backends/backend_agg.py:356-357`.

Impact:

- Renderer inputs are CPU materialized before C++ sees them.
- Renderer output is a CPU buffer by design.
- This does not merely miss GPU acceleration; it also makes downstream MLX
  arrays copies of a CPU framebuffer.

Priority:

- High for API correctness and end-to-end MLX dataflow.
- Medium for actual GPU acceleration unless the renderer itself is replaced or
  partially reimplemented, because AGG is a CPU rasterizer.

### 3. Path geometry wrapper returns bytearray-backed memoryviews

`src/_path_wrapper.cpp` is still largely a CPU buffer conversion machine, but
the `affine_transform` boundary has been rewired to accept MLX arrays directly
and expose `stream=`.

Output helper:

- `make_memoryview` allocates `PyByteArray`, copies with `std::memcpy`, and casts
  to `memoryview` at `src/_path_wrapper.cpp:21-35`.

Representative CPU paths:

- `points_in_path` accepts `py::buffer`, writes a `std::vector<uint8_t>`, and
  returns a copied memoryview at `src/_path_wrapper.cpp:63-84`.
- `get_path_collection_extents` accepts transforms/offsets as `py::buffer` and
  returns `std::array` memoryviews at `src/_path_wrapper.cpp:87-111`.
- `point_in_path_collection` accepts transforms/offsets as `py::buffer`, fills
  `std::vector<int>`, converts to `std::vector<int32_t>`, and returns memoryview
  at `src/_path_wrapper.cpp:114-145`.
- `affine_transform` now accepts MLX point arrays directly, converts the
  transform matrix through `convert_trans_affine_with_stream`, and exposes
  `stream=`. It still fills `std::vector<double>` and returns a memoryview, so
  only the input/staging side is MLX-aware so far.
- `cleanup_path` fills `std::vector<double>` and `std::vector<uint8_t>` then
  returns memoryviews at `src/_path_wrapper.cpp:254-296`.
- `is_sorted_and_has_non_nan` reinterprets arbitrary input as `py::buffer` at
  `src/_path_wrapper.cpp:358-383`.

`src/py_adaptors.h` is the path input bridge:

- `PathIterator` holds `py::buffer m_vertices` and `py::buffer m_codes` at
  `src/py_adaptors.h:28-37`.
- It reinterprets Python path vertices/codes as buffers at
  `src/py_adaptors.h:89-110`.
- Iteration reads every vertex via `BufferView` at `src/py_adaptors.h:120-135`.

Python bridge still forces CPU buffers before many path C++ calls:

- `lib/matplotlib/path.py:23-52` converts MLX arrays with `.tolist()`, flattens
  in Python, builds `array("d")`, and returns a `memoryview`.
- `lib/matplotlib/path.py:55-62` converts transforms to those memoryviews.
- `lib/matplotlib/transforms.py::affine_transform` no longer uses that
  memoryview staging path for its point and matrix inputs; it passes MLX arrays
  into `_path.affine_transform`.
- `lib/matplotlib/transforms.py:68-84` does the same for transform operations.

Impact:

- This is pervasive: clipping, extents, path containment, simplification,
  affine transform, and collection geometry all go CPU.
- The actual geometry algorithms are scalar/path-walk algorithms and may remain
  CPU for a while, but inputs and outputs should stop using memoryview as the
  public C++ contract.

Priority:

- Very high. This is the biggest general-purpose C++ boundary after Agg.

### 4. Triangulation and contouring eagerly copy MLX data to `std::vector`

`src/tri/_tri_wrapper.cpp` binds everything as `py::buffer`:

- `Triangulation` constructor takes `const py::buffer&` for `x`, `y`, and
  `triangles` at `src/tri/_tri_wrapper.cpp:7-24`.
- `TriContourGenerator` takes `const py::buffer& z` at
  `src/tri/_tri_wrapper.cpp:34-45`.
- `TrapezoidMapTriFinder.find_many` takes buffers through the C++ method at
  `src/tri/_tri_wrapper.cpp:47-60`.

`src/tri/_tri.cpp` copies inputs immediately:

- `copy_1d` and `copy_2d` create `std::vector<T>` copies from `BufferView` at
  `src/tri/_tri.cpp:31-51`.
- `Triangulation` accepts `py::buffer`, builds `BufferView`, then copies `x`,
  `y`, `triangles`, optional mask, edges, and neighbors into member vectors at
  `src/tri/_tri.cpp:262-340`.
- Persistent storage is `std::vector<double> _x, _y`,
  `std::vector<int> _triangles`, `_mask`, `_edges`, and `_neighbors` at
  `src/tri/_tri.h:299-310`.
- `TriContourGenerator` stores `py::buffer _z_buf` and `BufferView<double, 1>`
  at `src/tri/_tri.h:454-458`, then reads raw `z.data()` at
  `src/tri/_tri.cpp:1152-1157`.

Outputs are also host memoryviews:

- `calculate_plane_coefficients` fills `std::vector<double>` and returns a
  bytearray memoryview at `src/tri/_tri.cpp:467-523`.
- `get_edges` and `get_neighbors` copy vectors to bytearray memoryviews at
  `src/tri/_tri.cpp:576-622`.
- `find_many` fills `std::vector<int32_t>` and returns a bytearray memoryview at
  `src/tri/_tri.cpp:1413-1439`.

Python immediately rewraps those memoryviews into MLX arrays:

- `calculate_plane_coefficients` calls `mlxarr.asarray(...)` on the C++ memoryview
  at `lib/matplotlib/tri/_triangulation.py:91-101`.
- `edges` and `neighbors` do the same at
  `lib/matplotlib/tri/_triangulation.py:103-117` and
  `lib/matplotlib/tri/_triangulation.py:208-223`.
- `get_cpp_triangulation` passes `self.x`, `self.y`, `self.triangles`, mask,
  edges, and neighbors directly into the buffer-bound C++ constructor at
  `lib/matplotlib/tri/_triangulation.py:120-134`.

Impact:

- This code is explicitly CPU data-structure code today.
- Some algorithms here are graph/topology operations that are not a trivial MLX
  elementwise rewrite, but the binding should still stop hiding host copies
  behind memoryviews.

Priority:

- High. Triangulation is a self-contained module with clear input/output arrays,
  so it is a good candidate for a clean `mx::array + stream` API even if the
  internal algorithm initially stages to CPU explicitly.

### 5. Qhull wrapper is CPU-only and says so

`src/_qhull_wrapper.cpp` still has a misleading port comment:

- The file comment says it removed the hard dependency by using Python buffers
  and returning shaped memoryviews at `src/_qhull_wrapper.cpp:1-6`.

Current behavior:

- `delaunay` accepts `const py::buffer &x` and `const py::buffer &y` at
  `src/_qhull_wrapper.cpp:234-250`.
- It reads raw `double*` data from `BufferView`.
- `delaunay_impl` copies centered points into `std::vector<coordT>` at
  `src/_qhull_wrapper.cpp:107-157`.
- It fills `std::vector<int32_t>` triangles/neighbors and returns bytearray
  memoryviews at `src/_qhull_wrapper.cpp:190-231`.

Impact:

- Qhull itself is a CPU library, so a CPU staging point is expected.
- The current API hides that staging behind Python buffers, which is the wrong
  abstraction for MLX. It should instead accept MLX arrays, require or default a
  stream/device, and make any CPU transfer explicit in C++.

Priority:

- Medium-high. The implementation will remain CPU unless Qhull is replaced, but
  the public C++ boundary should be MLX-aware.

### 6. FreeType/font path and bitmap APIs are host buffers

FreeType is CPU, but the current wrapper still exports and imports CPU memory
as if that were the array abstraction.

Current behavior:

- `FT2Font::draw_glyph_to_bitmap` takes `const py::buffer &im` at
  `src/ft2font.h:132-135` and `src/ft2font.cpp:626-653`.
- `FT2Font::get_image` returns `py::memoryview::from_buffer` over
  `std::vector<uint8_t> image` at `src/ft2font.h:147-155`, with the image stored
  at `src/ft2font.h:181-187`.
- `PyFT2Font_set_text` fills `std::vector<double> xys` and returns a bytearray
  memoryview at `src/ft2font_wrapper.cpp:711-746`.
- `PyFT2Font_get_path` fills `std::vector<double>` and
  `std::vector<unsigned char>`, copies them into bytearrays, and returns
  memoryviews at `src/ft2font_wrapper.cpp:1410-1443`.
- `FT2Image` and `FT2Font` both expose buffer protocol via `py::buffer_info` at
  `src/ft2font_wrapper.cpp:1577-1598` and
  `src/ft2font_wrapper.cpp:1789-1793`.

Impact:

- Fonts are probably not the first GPU target, but the current API keeps
  reintroducing memoryviews and host-backed arrays into rendering.
- This is also a documentation mismatch: docstrings advertise `mlxarr.ndarray`
  while C++ returns memoryviews.

Priority:

- Medium. Keep FreeType CPU, but return `mx::array` outputs and accept MLX image
  arrays with explicit stream-aware staging for bitmap drawing.

### 7. `_image` is MLX-aware at the boundary but still CPU inside

The image resample port is the current best partial pattern, but it is not a GPU
kernel.

Still CPU:

- `get_transform_mesh` builds `std::vector<double>`, copies it into a bytearray
  memoryview, calls Python transform code, then copies the result back into a
  `std::vector<double>` at `src/_image_wrapper.cpp:255-300`.
- The resample function passes raw `void*` pointers into `resample<Pixel>` at
  `src/_image_wrapper.cpp:433-464`.
- `_image_resample.h` implements the real resampling with AGG pixel formats,
  `agg::rendering_buffer`, span allocators, and scanline renderers. The core
  CPU pointer signature is at `src/_image_resample.h:690-743`.
- `calculate_rms_and_diff` now accepts MLX arrays through the same stream-aware
  `get_array_info` path as `resample`, but still computes into a
  `std::vector<unsigned char>` and returns a bytearray memoryview.

Impact:

- The `_image` module now accepts MLX arrays and respects stream staging better
  than the rest of the C++ tree.
- The device is still used to stage/evaluate arrays before raw CPU access; it is
  not using an MLX graph operation or GPU kernel for resampling.

Priority:

- High if image resampling is expected to run on GPU.
- Otherwise keep this as a compatibility bridge while replacing the memoryview
  return paths and making CPU fallback explicit.

### 8. Float64 Python scalar use is overridden at the narrow boundary

The installed MLX runtime preserves float64 arrays on CPU, but direct assignment
and arithmetic use of a bare Python float with an MLX float64 array rounded the
scalar before storage or computation. That broke exact transform identities such
as `Affine2D` rotation composition.

Current fix:

- `src/_mlx_overrides.cpp` detects float64 targets and bare Python floats.
- It copies the original C `double` bytes into a one-element `memoryview("d")`,
  constructs an MLX float64 array from that typed buffer, and reshapes it to an
  MLX scalar.
- `lib/matplotlib/_mlx_array.py` routes `mx.array.__setitem__`, scalar add,
  subtract, multiply, true divide, floor divide, modulo, power, `full`,
  `full_like`, and constant `pad` through that compiled override before calling
  the original MLX operation.
- `lib/matplotlib/tests/test_transforms.py` asserts that assigning
  `math.cos(math.radians(90))` into a float64 MLX matrix preserves the exact
  Python value, and that the same value survives scalar arithmetic and fill
  constructors.

Impact:

- This is a precision-preserving override at the boundary where matplotlib
  mutates tiny transform matrices and mixes Python scalar math with MLX float64
  arrays.
- It is intentionally narrower than vendoring the whole `mlx-precise` float64
  stack. The broader GPU float64 work still belongs in the MLX runtime source
  or in a local vendored MLX build, not in `_mlx_array.py`.

Priority:

- Keep this as the compatibility pin while deciding whether this repo should
  build against `mlx-precise` directly.

### 9. Tk blit remains CPU buffer-only

`src/_tkagg.cpp` accepts a `py::buffer data` at `src/_tkagg.cpp:97-115`, reads it
through `BufferView<unsigned char, 3>`, and then passes it into Tk.

Impact:

- Tk requires CPU pixel data, so this is a legitimate host sink.
- The input boundary should still be explicit: MLX framebuffer input plus stream,
  then a documented CPU staging copy into Tk.

Priority:

- Low for GPU acceleration, medium for honest API cleanup.

## MLX API Reference Points

The local MLX source in `/Volumes/stuff/Projects/mlxports/mlx-precise` confirms
the C++ patterns we should be using:

- `mlx::core::array` has shape, strides, dtype, itemsize, size, and `eval()` at
  `/Volumes/stuff/Projects/mlxports/mlx-precise/mlx/array.h:25-137`.
- `mx::array::data<T>()` is available after evaluation, but that is a host/device
  storage access point and should not be treated as a general array API.
- `mx::Stream` carries both stream index and device at
  `/Volumes/stuff/Projects/mlxports/mlx-precise/mlx/stream.h:9-13`.
- `mx::synchronize(Stream)` exists at
  `/Volumes/stuff/Projects/mlxports/mlx-precise/mlx/stream.h:35-39`.
- `mx::StreamOrDevice` is `std::variant<std::monostate, Stream, Device>` at
  `/Volumes/stuff/Projects/mlxports/mlx-precise/mlx/utils.h:15-17`.
- Core MLX ops accept `StreamOrDevice`, including `astype`, `as_strided`, `copy`,
  `zeros`, `reshape`, and more at
  `/Volumes/stuff/Projects/mlxports/mlx-precise/mlx/ops.h:46-125`.

## Recommended Rewire Order

### Phase 1: Pull MLX conversion out of `_image` into a shared C++ layer

Current MLX helpers are duplicated in `_image_wrapper.cpp`, `_path_wrapper.cpp`,
and `py_converters.cpp`. Before converting more modules, create a C++ helper
that is shared by extension modules and built with nanobind/MLX:

- `is_mlx_array_like`
- `as_mlx_array`
- Python-to-`mx::StreamOrDevice` conversion
- dtype/shape/stride validation
- explicit CPU staging helpers for libraries that require host pointers

This must be a C++ helper, not a Python `_mlx_array.py` wrapper-style shim.

### Phase 2: Convert path/transform boundaries first

Target files:

- `src/py_converters.h`
- `src/py_adaptors.h`
- `src/_path_wrapper.cpp`
- `lib/matplotlib/path.py`
- `lib/matplotlib/transforms.py`
- `lib/matplotlib/backends/backend_agg.py`

Reason:

- These are the most common call sites and currently force `.tolist()` and
  memoryviews before C++.
- They feed both Agg and non-Agg geometry code.

Desired direction:

- Bind path functions to `py::object` or direct `mx::array` overloads with
  `stream=` where arrays enter C++.
- Convert return values that are arrays into `mx::array` directly.
- Keep scalar/string-returning routines as-is.

### Phase 3: Convert Agg wrapper signatures

Target files:

- `src/_backend_agg_wrapper.cpp`
- `src/_backend_agg.cpp`
- `src/_backend_agg.h`
- `lib/matplotlib/backends/backend_agg.py`

Reason:

- Agg will remain CPU until the renderer is replaced, but the boundary should not
  pretend MLX arrays are buffers.

Desired direction:

- Accept MLX arrays and an optional stream/device in draw methods.
- Explicitly stage to CPU only at the AGG call boundary.
- Return renderer output as `mx::array` where Python callers expect array data,
  while keeping a documented CPU buffer escape hatch only for external APIs that
  require one.

### Phase 4: Convert triangulation and qhull I/O

Target files:

- `src/tri/_tri_wrapper.cpp`
- `src/tri/_tri.cpp`
- `src/tri/_tri.h`
- `src/_qhull_wrapper.cpp`
- `lib/matplotlib/tri/_triangulation.py`
- `lib/matplotlib/tri/_tricontour.py`

Reason:

- These modules are algorithmically CPU for now, but their inputs/outputs are
  well-defined arrays and can be made honest quickly.

Desired direction:

- Bind public constructors/methods as MLX-aware.
- For CPU-only algorithm internals, explicitly copy from `mx::array` after
  stream synchronization.
- Return `mx::array` outputs instead of bytearray memoryviews.

### Phase 5: Convert FreeType and GUI sinks

Target files:

- `src/ft2font.cpp`
- `src/ft2font.h`
- `src/ft2font_wrapper.cpp`
- `src/_tkagg.cpp`

Reason:

- These depend on CPU libraries, so the main cleanup is honest staging and return
  types, not immediate GPU execution.

Desired direction:

- Make bitmap inputs MLX-aware.
- Return glyph image/path/position arrays as MLX arrays.
- Keep external GUI/image-library sinks documented as CPU staging points.

## Bottom Line

The C++ port has one partially correct MLX/nanobind island: `_image` and affine
conversion. Everywhere else, array-like data still means Python buffer protocol,
host pointer, `std::vector`, bytearray, and memoryview.

The next real step is not another Python shim. It is to make the C++ extension
modules link MLX/nanobind where they accept array data, thread stream/device
through those APIs, and replace memoryview outputs with real `mx::array` results.
CPU-only libraries can still stage to host internally, but that staging should be
explicit and local rather than hidden in Python buffer compatibility.
