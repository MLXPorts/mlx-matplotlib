#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "_backend_agg_basic_types.h"
#include "_path.h"
#include "py_adaptors.h"
#include "py_buffer.h"
#include "py_converters.h"

namespace py = pybind11;
using namespace pybind11::literals;

static py::memoryview make_memoryview(const void *data,
                                      py::ssize_t nbytes,
                                      const char *format,
                                      py::tuple shape)
{
    py::bytearray ba(nbytes);
    if (nbytes > 0 && data != nullptr) {
        std::memcpy(PyByteArray_AsString(ba.ptr()), data, static_cast<size_t>(nbytes));
    }
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    return mv.attr("cast")(format, shape).cast<py::memoryview>();
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

static py::memoryview Py_points_in_path(const py::buffer &points_obj,
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
        uint8_t &operator()(py::ssize_t i) { return ptr[i]; }
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

static py::memoryview Py_point_in_path_collection(double x,
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

static py::memoryview Py_affine_transform(const py::buffer &vertices_arr, agg::trans_affine trans)
{
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
    m.def("affine_transform", &Py_affine_transform, "points"_a, "trans"_a);
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
