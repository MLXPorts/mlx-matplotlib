#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "_backend_agg_basic_types.h"
#include "_path.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"
#include "py_adaptors.h"
#include "py_buffer.h"
#include "py_converters.h"

namespace py = pybind11;
namespace nb = nanobind;
namespace mx = mlx::core;
using namespace pybind11::literals;

static bool has_explicit_stream(const mx::StreamOrDevice& stream)
{
    return !std::holds_alternative<std::monostate>(stream);
}

static mx::Device parse_mlx_device_repr(const std::string& repr)
{
    auto start = repr.find("Device(");
    if (start == std::string::npos) {
        throw py::type_error("stream must be an mlx.core.Stream or mlx.core.Device");
    }

    auto type_start = start + std::string("Device(").size();
    auto comma = repr.find(',', type_start);
    auto close = repr.find(')', comma);
    if (comma == std::string::npos || close == std::string::npos) {
        throw py::type_error("stream must be an mlx.core.Stream or mlx.core.Device");
    }

    auto type = repr.substr(type_start, comma - type_start);
    auto index = std::stoi(repr.substr(comma + 1, close - comma - 1));
    if (type == "cpu") {
        return mx::Device(mx::Device::cpu, index);
    }
    if (type == "gpu") {
        return mx::Device(mx::Device::gpu, index);
    }
    throw py::type_error("stream must be an mlx.core.Stream or mlx.core.Device");
}

static mx::Stream parse_mlx_stream_repr(const std::string& repr)
{
    auto device = parse_mlx_device_repr(repr);
    auto comma = repr.rfind(',');
    auto close = repr.rfind(')');
    if (comma == std::string::npos || close == std::string::npos || comma > close) {
        throw py::type_error("stream must be an mlx.core.Stream or mlx.core.Device");
    }
    auto index = std::stoi(repr.substr(comma + 1, close - comma - 1));
    return mx::Stream(index, device);
}

static mx::StreamOrDevice as_stream_or_device(const py::object& stream)
{
    if (stream.is_none()) {
        return std::monostate{};
    }

    nb::object nb_stream = nb::borrow<nb::object>(nb::handle(stream.ptr()));
    try {
        return nb::cast<mx::Stream>(nb_stream);
    } catch (const nb::cast_error&) {
    }
    try {
        return nb::cast<mx::Device>(nb_stream);
    } catch (const nb::cast_error&) {
    }
    try {
        return mx::Device(nb::cast<mx::Device::DeviceType>(nb_stream));
    } catch (const nb::cast_error&) {
    }

    auto repr = py::repr(stream).cast<std::string>();
    if (repr.rfind("Stream(", 0) == 0) {
        return parse_mlx_stream_repr(repr);
    }
    if (repr.rfind("Device(", 0) == 0) {
        return parse_mlx_device_repr(repr);
    }
    if (repr == "DeviceType.cpu") {
        return mx::Device(mx::Device::cpu);
    }
    if (repr == "DeviceType.gpu") {
        return mx::Device(mx::Device::gpu);
    }
    throw py::type_error("stream must be an mlx.core.Stream or mlx.core.Device");
}

static bool is_mlx_array_like(py::handle obj)
{
    nb::object nb_obj = nb::borrow<nb::object>(nb::handle(obj.ptr()));
    return nb::isinstance<mx::array>(nb_obj) || nb::hasattr(nb_obj, "__mlx_array__");
}

static mx::array as_mlx_array(py::handle obj)
{
    nb::object nb_obj = nb::borrow<nb::object>(nb::handle(obj.ptr()));
    if (nb::isinstance<mx::array>(nb_obj)) {
        return nb::cast<mx::array>(nb_obj);
    }
    if (nb::hasattr(nb_obj, "__mlx_array__")) {
        return nb::cast<mx::array>(nb_obj.attr("__mlx_array__")());
    }
    throw std::invalid_argument("object is not an MLX array");
}

static mx::array evaluated_mlx_array(py::handle obj,
                                    const mx::StreamOrDevice& stream)
{
    auto array = as_mlx_array(obj);
    if (has_explicit_stream(stream) || !array.flags().row_contiguous) {
        array = mx::contiguous(array, false, stream);
    }
    {
        py::gil_scoped_release release;
        array.eval();
        if (has_explicit_stream(stream)) {
            mx::synchronize(mx::to_stream(stream));
        }
    }
    return array;
}

static py::object py_from_mlx_array(const mx::array& array)
{
    nb::object nb_array = nb::cast(array);
    Py_INCREF(nb_array.ptr());
    return py::reinterpret_steal<py::object>(nb_array.ptr());
}

static bool is_supported_mlx_path_float(mx::Dtype dtype)
{
    return dtype == mx::float32 || dtype == mx::float64;
}

static mx::array mlx_affine_transform_2d(const mx::array& vertices,
                                         const mx::array& matrix,
                                         const mx::StreamOrDevice& stream)
{
    auto n = static_cast<mx::ShapeElem>(vertices.shape(0));
    auto x = mx::slice(vertices, {0, 0}, {n, 1}, {1, 1}, stream);
    auto y = mx::slice(vertices, {0, 1}, {n, 2}, {1, 1}, stream);

    auto sx = mx::slice(matrix, {0, 0}, {1, 1}, {1, 1}, stream);
    auto shx = mx::slice(matrix, {0, 1}, {1, 2}, {1, 1}, stream);
    auto tx = mx::slice(matrix, {0, 2}, {1, 3}, {1, 1}, stream);
    auto shy = mx::slice(matrix, {1, 0}, {2, 1}, {1, 1}, stream);
    auto sy = mx::slice(matrix, {1, 1}, {2, 2}, {1, 1}, stream);
    auto ty = mx::slice(matrix, {1, 2}, {2, 3}, {1, 1}, stream);

    auto out_x = mx::add(
        mx::add(mx::multiply(sx, x, stream),
                mx::multiply(shx, y, stream),
                stream),
        tx,
        stream);
    auto out_y = mx::add(
        mx::add(mx::multiply(shy, x, stream),
                mx::multiply(sy, y, stream),
                stream),
        ty,
        stream);

    std::vector<mx::array> columns;
    columns.reserve(2);
    columns.push_back(out_x);
    columns.push_back(out_y);
    return mx::concatenate(std::move(columns), 1, stream);
}

static py::object Py_mlx_affine_transform(py::handle vertices_obj,
                                          py::handle transform_obj,
                                          const mx::StreamOrDevice& stream)
{
    auto vertices = as_mlx_array(vertices_obj);
    auto matrix = as_mlx_array(transform_obj);

    if (!is_supported_mlx_path_float(vertices.dtype())) {
        throw py::value_error("vertices must be float32 or float64");
    }
    if (matrix.dtype() != vertices.dtype()) {
        throw py::value_error("affine matrix dtype must match vertices dtype");
    }
    if (matrix.ndim() != 2 || matrix.shape(0) != 3 || matrix.shape(1) != 3) {
        throw py::value_error("Invalid affine transformation matrix");
    }
    if (vertices.ndim() == 2) {
        if (vertices.shape(1) != 2) {
            throw py::value_error("vertices must have shape (N, 2)");
        }
        return py_from_mlx_array(mlx_affine_transform_2d(vertices, matrix, stream));
    }
    if (vertices.ndim() == 1) {
        if (vertices.shape(0) != 2) {
            throw std::runtime_error("Invalid vertices array.");
        }
        auto row = mx::reshape(vertices, {1, 2}, stream);
        auto transformed = mlx_affine_transform_2d(row, matrix, stream);
        return py_from_mlx_array(mx::reshape(transformed, {2}, stream));
    }

    throw py::value_error("vertices must be 1D or 2D");
}

static py::object dtype_from_buffer_format(const char *format)
{
    py::object mx_module = py::module_::import("mlx.core");
    if (std::strcmp(format, "d") == 0) {
        return mx_module.attr("float64");
    }
    if (std::strcmp(format, "i") == 0) {
        return mx_module.attr("int32");
    }
    if (std::strcmp(format, "B") == 0) {
        return mx_module.attr("uint8");
    }
    throw py::value_error("unsupported buffer format");
}

static py::object make_memoryview(const void *data,
                                  py::ssize_t nbytes,
                                  const char *format,
                                  py::tuple shape)
{
    if (nbytes == 0) {
        py::object mx_module = py::module_::import("mlx.core");
        return mx_module.attr("zeros")(shape, "dtype"_a = dtype_from_buffer_format(format));
    }

    py::bytearray ba = py::reinterpret_steal<py::bytearray>(
        PyByteArray_FromStringAndSize(nullptr, nbytes));
    if (!ba) {
        throw py::error_already_set();
    }
    if (nbytes > 0 && data != nullptr) {
        std::memcpy(PyByteArray_AsString(ba.ptr()), data, static_cast<size_t>(nbytes));
    }
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    return mv.attr("cast")(format, shape);
}

static py::list convert_polygon_vector(std::vector<Polygon> &polygons)
{
    auto result = py::list(polygons.size());

    for (size_t i = 0; i < polygons.size(); ++i) {
        const auto &poly = polygons[i];
        auto n = static_cast<py::ssize_t>(poly.size());
        result[i] = make_memoryview(reinterpret_cast<const double *>(poly.data()),
                                    n * 2 * static_cast<py::ssize_t>(sizeof(double)),
                                    "d",
                                    py::make_tuple(n, 2));
    }

    return result;
}

static bool Py_point_in_path(double x,
                             double y,
                             double r,
                             mpl::PathIterator path,
                             agg::trans_affine trans)
{
    return point_in_path(x, y, r, path, trans);
}

static py::object Py_points_in_path(const py::buffer &points_obj,
                                    double r,
                                    mpl::PathIterator path,
                                    agg::trans_affine trans)
{
    auto points = convert_points(points_obj);

    py::ssize_t n = points.shape(0);
    std::vector<uint8_t> results(static_cast<size_t>(n));
    struct OutView {
        uint8_t *ptr;
        py::ssize_t n;
        py::ssize_t ndim() const { return 1; }
        py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 0; }
        py::ssize_t size() const { return n; }
        uint8_t &operator[](py::ssize_t i) { return ptr[i]; }
        const uint8_t &operator[](py::ssize_t i) const { return ptr[i]; }
    } results_view{results.data(), n};

    points_in_path(points, r, path, trans, results_view);

    return make_memoryview(results.data(), n, "B", py::make_tuple(n));
}

static py::tuple Py_get_path_collection_extents(agg::trans_affine master_transform,
                                                mpl::PathGenerator paths,
                                                const py::buffer &transforms_obj,
                                                const py::buffer &offsets_obj,
                                                agg::trans_affine offset_trans)
{
    auto transforms = convert_transforms(transforms_obj);
    auto offsets = convert_points(offsets_obj);
    extent_limits e;

    get_path_collection_extents(master_transform, paths, transforms, offsets, offset_trans, e);

    std::array<double, 4> ext{{e.start.x, e.start.y, e.end.x, e.end.y}};
    std::array<double, 2> minpos{{e.minpos.x, e.minpos.y}};

    auto ext_mv = make_memoryview(ext.data(),
                                  4 * static_cast<py::ssize_t>(sizeof(double)),
                                  "d",
                                  py::make_tuple(2, 2));
    auto min_mv = make_memoryview(minpos.data(),
                                  2 * static_cast<py::ssize_t>(sizeof(double)),
                                  "d",
                                  py::make_tuple(2));

    return py::make_tuple(ext_mv, min_mv);
}

static py::object Py_point_in_path_collection(double x,
                                              double y,
                                              double radius,
                                              agg::trans_affine master_transform,
                                              mpl::PathGenerator paths,
                                              const py::buffer &transforms_obj,
                                              const py::buffer &offsets_obj,
                                              agg::trans_affine offset_trans,
                                              bool filled)
{
    auto transforms = convert_transforms(transforms_obj);
    auto offsets = convert_points(offsets_obj);
    std::vector<int> result;

    point_in_path_collection(x,
                             y,
                             radius,
                             master_transform,
                             paths,
                             transforms,
                             offsets,
                             offset_trans,
                             filled,
                             result);

    std::vector<std::int32_t> out(result.begin(), result.end());
    auto n = static_cast<py::ssize_t>(out.size());
    return make_memoryview(out.data(),
                           n * static_cast<py::ssize_t>(sizeof(std::int32_t)),
                           "i",
                           py::make_tuple(n));
}

static bool Py_path_in_path(mpl::PathIterator a,
                            agg::trans_affine atrans,
                            mpl::PathIterator b,
                            agg::trans_affine btrans)
{
    return path_in_path(a, atrans, b, btrans);
}

static py::list Py_clip_path_to_rect(mpl::PathIterator path, agg::rect_d rect, bool inside)
{
    auto result = clip_path_to_rect(path, rect, inside);
    return convert_polygon_vector(result);
}

static py::object Py_affine_transform(py::object vertices_obj,
                                      py::object transform_obj,
                                      py::object stream)
{
    auto stream_or_device = as_stream_or_device(stream);
    if (is_mlx_array_like(vertices_obj) && is_mlx_array_like(transform_obj)) {
        return Py_mlx_affine_transform(vertices_obj, transform_obj, stream_or_device);
    }

    agg::trans_affine trans;
    convert_trans_affine_with_stream(transform_obj, trans, stream);

    if (is_mlx_array_like(vertices_obj)) {
        auto vertices = evaluated_mlx_array(vertices_obj, stream_or_device);
        if (vertices.dtype() != mx::float64) {
            throw py::value_error("vertices must be float64");
        }
        if (vertices.ndim() == 2) {
            if (vertices.shape(1) != 2) {
                throw py::value_error("vertices must have shape (N, 2)");
            }

            auto n = static_cast<py::ssize_t>(vertices.shape(0));
            std::vector<double> result(static_cast<size_t>(n) * 2);
            struct Out2D {
                double *ptr;
                py::ssize_t n;
                py::ssize_t ndim() const { return 2; }
                py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 2; }
                py::ssize_t size() const { return n * 2; }
                double &operator()(py::ssize_t i, py::ssize_t j) { return ptr[i * 2 + j]; }
            } out{result.data(), n};

            const double *data = vertices.data<double>();
            auto stride0 = static_cast<py::ssize_t>(vertices.strides(0));
            auto stride1 = static_cast<py::ssize_t>(vertices.strides(1));
            struct In2D {
                const double *ptr;
                py::ssize_t n;
                py::ssize_t stride0;
                py::ssize_t stride1;
                py::ssize_t ndim() const { return 2; }
                py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 2; }
                py::ssize_t size() const { return n * 2; }
                const double &operator()(py::ssize_t i, py::ssize_t j) const
                {
                    return ptr[i * stride0 + j * stride1];
                }
            } in{data, n, stride0, stride1};

            affine_transform_2d(in, trans, out);
            return make_memoryview(result.data(),
                                   n * 2 * static_cast<py::ssize_t>(sizeof(double)),
                                   "d",
                                   py::make_tuple(n, 2));
        }

        if (vertices.ndim() == 1) {
            auto n = static_cast<py::ssize_t>(vertices.shape(0));
            std::vector<double> result(static_cast<size_t>(n));
            struct Out1D {
                double *ptr;
                py::ssize_t n;
                py::ssize_t ndim() const { return 1; }
                py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 0; }
                py::ssize_t size() const { return n; }
                double &operator()(py::ssize_t i) { return ptr[i]; }
            } out{result.data(), n};

            const double *data = vertices.data<double>();
            auto stride0 = static_cast<py::ssize_t>(vertices.strides(0));
            struct In1D {
                const double *ptr;
                py::ssize_t n;
                py::ssize_t stride0;
                py::ssize_t ndim() const { return 1; }
                py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 0; }
                py::ssize_t size() const { return n; }
                const double &operator()(py::ssize_t i) const { return ptr[i * stride0]; }
            } in{data, n, stride0};

            affine_transform_1d(in, trans, out);
            return make_memoryview(result.data(),
                                   n * static_cast<py::ssize_t>(sizeof(double)),
                                   "d",
                                   py::make_tuple(n));
        }

        throw py::value_error("vertices must be 1D or 2D");
    }

    py::buffer vertices_arr = py::reinterpret_borrow<py::buffer>(vertices_obj);
    auto info = vertices_arr.request(false);
    if (info.ndim == 2) {
        mpl::BufferView<double, 2> vertices(vertices_arr);
        check_trailing_shape(vertices, "vertices", 2);

        auto n = vertices.shape(0);
        std::vector<double> result(static_cast<size_t>(n) * 2);
        struct Out2D {
            double *ptr;
            py::ssize_t n;
            py::ssize_t ndim() const { return 2; }
            py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 2; }
            py::ssize_t size() const { return n * 2; }
            double &operator()(py::ssize_t i, py::ssize_t j) { return ptr[i * 2 + j]; }
        } out{result.data(), n};

        affine_transform_2d(vertices, trans, out);
        return make_memoryview(result.data(),
                               n * 2 * static_cast<py::ssize_t>(sizeof(double)),
                               "d",
                               py::make_tuple(n, 2));
    }

    if (info.ndim == 1) {
        mpl::BufferView<double, 1> vertices(vertices_arr);

        auto n = vertices.shape(0);
        std::vector<double> result(static_cast<size_t>(n));
        struct Out1D {
            double *ptr;
            py::ssize_t n;
            py::ssize_t ndim() const { return 1; }
            py::ssize_t shape(py::ssize_t i) const { return i == 0 ? n : 0; }
            py::ssize_t size() const { return n; }
            double &operator()(py::ssize_t i) { return ptr[i]; }
        } out{result.data(), n};

        affine_transform_1d(vertices, trans, out);
        return make_memoryview(result.data(),
                               n * static_cast<py::ssize_t>(sizeof(double)),
                               "d",
                               py::make_tuple(n));
    }

    throw py::value_error("vertices must be 1D or 2D");
}

static int Py_count_bboxes_overlapping_bbox(agg::rect_d bbox, const py::buffer &bboxes_obj)
{
    auto bboxes = convert_bboxes(bboxes_obj);
    return count_bboxes_overlapping_bbox(bbox, bboxes);
}

static bool Py_path_intersects_path(mpl::PathIterator p1, mpl::PathIterator p2, bool filled)
{
    agg::trans_affine t1;
    agg::trans_affine t2;

    bool result = path_intersects_path(p1, p2);
    if (filled) {
        if (!result) {
            result = path_in_path(p1, t1, p2, t2);
        }
        if (!result) {
            result = path_in_path(p2, t1, p1, t2);
        }
    }
    return result;
}

static bool Py_path_intersects_rectangle(mpl::PathIterator path,
                                        double rect_x1,
                                        double rect_y1,
                                        double rect_x2,
                                        double rect_y2,
                                        bool filled)
{
    return path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled);
}

static py::list Py_convert_path_to_polygons(mpl::PathIterator path,
                                            agg::trans_affine trans,
                                            double width,
                                            double height,
                                            bool closed_only)
{
    std::vector<Polygon> result;
    convert_path_to_polygons(path, trans, width, height, closed_only, result);
    return convert_polygon_vector(result);
}

static py::tuple Py_cleanup_path(mpl::PathIterator path,
                                 agg::trans_affine trans,
                                 bool remove_nans,
                                 agg::rect_d clip_rect,
                                 e_snap_mode snap_mode,
                                 double stroke_width,
                                 std::optional<bool> simplify,
                                 bool return_curves,
                                 SketchParams sketch)
{
    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    std::vector<double> vertices;
    std::vector<uint8_t> codes;

    cleanup_path(path,
                 trans,
                 remove_nans,
                 do_clip,
                 clip_rect,
                 snap_mode,
                 stroke_width,
                 *simplify,
                 return_curves,
                 sketch,
                 vertices,
                 codes);

    auto length = static_cast<py::ssize_t>(codes.size());
    auto v_mv = make_memoryview(vertices.data(),
                                length * 2 * static_cast<py::ssize_t>(sizeof(double)),
                                "d",
                                py::make_tuple(length, 2));
    auto c_mv = make_memoryview(codes.data(),
                                length * static_cast<py::ssize_t>(sizeof(uint8_t)),
                                "B",
                                py::make_tuple(length));

    return py::make_tuple(v_mv, c_mv);
}

const char *Py_convert_to_string__doc__ = R"""(--

Convert *path* to a bytestring.

The first five parameters (up to *sketch*) are interpreted as in `.cleanup_path`. The
following ones are detailed below.

Parameters
----------
path : Path
trans : Transform or None
clip_rect : sequence of 4 floats, or None
simplify : bool
sketch : tuple of 3 floats, or None
precision : int
    The precision used to "%.*f"-format the values.
codes : sequence of 5 bytestrings
postfix : bool
)""";

static py::object Py_convert_to_string(mpl::PathIterator path,
                                       agg::trans_affine trans,
                                       agg::rect_d cliprect,
                                       std::optional<bool> simplify,
                                       SketchParams sketch,
                                       int precision,
                                       const std::array<std::string, 5> &codes,
                                       bool postfix)
{
    std::string buffer;

    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    bool status = convert_to_string(path,
                                    trans,
                                    cliprect,
                                    *simplify,
                                    sketch,
                                    precision,
                                    codes,
                                    postfix,
                                    buffer);

    if (!status) {
        throw py::value_error("Malformed path codes");
    }

    return py::bytes(buffer);
}

const char *Py_is_sorted_and_has_non_nan__doc__ = R"""(--

Return whether the 1D *array* is monotonically increasing, ignoring NaNs, and has at
least one non-nan value.

)""";

static bool Py_is_sorted_and_has_non_nan(py::object obj)
{
    py::buffer buf = py::reinterpret_borrow<py::buffer>(obj);
    auto info = buf.request(false);
    if (info.ndim != 1) {
        throw std::invalid_argument("array must be 1D");
    }

    if (info.format == py::format_descriptor<std::int32_t>::format()
        && info.itemsize == static_cast<py::ssize_t>(sizeof(std::int32_t))) {
        return is_sorted_and_has_non_nan<std::int32_t>(mpl::BufferView<std::int32_t, 1>(buf));
    }
    if (info.format == py::format_descriptor<std::int64_t>::format()
        && info.itemsize == static_cast<py::ssize_t>(sizeof(std::int64_t))) {
        return is_sorted_and_has_non_nan<std::int64_t>(mpl::BufferView<std::int64_t, 1>(buf));
    }
    if (info.format == py::format_descriptor<float>::format()
        && info.itemsize == static_cast<py::ssize_t>(sizeof(float))) {
        return is_sorted_and_has_non_nan<float>(mpl::BufferView<float, 1>(buf));
    }
    if (info.format == py::format_descriptor<double>::format()
        && info.itemsize == static_cast<py::ssize_t>(sizeof(double))) {
        return is_sorted_and_has_non_nan<double>(mpl::BufferView<double, 1>(buf));
    }

    throw std::invalid_argument("Unsupported dtype for is_sorted_and_has_non_nan");
}

PYBIND11_MODULE(_path, m, py::mod_gil_not_used())
{
#if NB_VERSION_MAJOR > 2 || (NB_VERSION_MAJOR == 2 && NB_VERSION_MINOR >= 12)
    nb::detail::nb_module_exec(NB_DOMAIN_STR, m.ptr());
#else
    nb::detail::init(NB_DOMAIN_STR);
#endif

    m.def("point_in_path", &Py_point_in_path, "x"_a, "y"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("points_in_path", &Py_points_in_path, "points"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("get_path_collection_extents",
          &Py_get_path_collection_extents,
          "master_transform"_a,
          "paths"_a,
          "transforms"_a,
          "offsets"_a,
          "offset_transform"_a);
    m.def("point_in_path_collection",
          &Py_point_in_path_collection,
          "x"_a,
          "y"_a,
          "radius"_a,
          "master_transform"_a,
          "paths"_a,
          "transforms"_a,
          "offsets"_a,
          "offset_trans"_a,
          "filled"_a);
    m.def("path_in_path", &Py_path_in_path, "path_a"_a, "trans_a"_a, "path_b"_a, "trans_b"_a);
    m.def("clip_path_to_rect", &Py_clip_path_to_rect, "path"_a, "rect"_a, "inside"_a);
    m.def("affine_transform", &Py_affine_transform,
          "points"_a,
          "trans"_a,
          py::kw_only(),
          "stream"_a = py::none());
    m.def("count_bboxes_overlapping_bbox", &Py_count_bboxes_overlapping_bbox, "bbox"_a, "bboxes"_a);
    m.def("path_intersects_path", &Py_path_intersects_path, "path1"_a, "path2"_a, "filled"_a = false);
    m.def("path_intersects_rectangle",
          &Py_path_intersects_rectangle,
          "path"_a,
          "rect_x1"_a,
          "rect_y1"_a,
          "rect_x2"_a,
          "rect_y2"_a,
          "filled"_a = false);
    m.def("convert_path_to_polygons",
          &Py_convert_path_to_polygons,
          "path"_a,
          "trans"_a,
          "width"_a = 0.0,
          "height"_a = 0.0,
          "closed_only"_a = false);
    m.def("cleanup_path",
          &Py_cleanup_path,
          "path"_a,
          "trans"_a,
          "remove_nans"_a,
          "clip_rect"_a,
          "snap_mode"_a,
          "stroke_width"_a,
          "simplify"_a,
          "return_curves"_a,
          "sketch"_a);
    m.def("convert_to_string",
          &Py_convert_to_string,
          "path"_a,
          "trans"_a,
          "clip_rect"_a,
          "simplify"_a,
          "sketch"_a,
          "precision"_a,
          "codes"_a,
          "postfix"_a,
          Py_convert_to_string__doc__);
    m.def("is_sorted_and_has_non_nan", &Py_is_sorted_and_has_non_nan, "array"_a, Py_is_sorted_and_has_non_nan__doc__);
}
