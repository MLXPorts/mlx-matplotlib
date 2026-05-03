/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the pybind11 type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"
#include "py_buffer.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);
void convert_trans_affine_with_stream(const py::object& transform,
                                      agg::trans_affine& affine,
                                      const py::object& stream);

inline auto convert_points(const py::buffer &obj)
{
    mpl::BufferView<double, 2> view(obj);
    check_trailing_shape(view, "points", 2);
    return view;
}

inline auto convert_transforms(const py::buffer &obj)
{
    mpl::BufferView<double, 3> view(obj);
    check_trailing_shape(view, "transforms", 3, 3);
    return view;
}

inline auto convert_bboxes(const py::buffer &obj)
{
    mpl::BufferView<double, 3> view(obj);
    check_trailing_shape(view, "bbox array", 2, 2);
    return view;
}

inline auto convert_colors(const py::buffer &obj)
{
    mpl::BufferView<double, 2> view(obj);
    check_trailing_shape(view, "colors", 4);
    return view;
}

namespace PYBIND11_NAMESPACE { namespace detail {
    template <> struct type_caster<agg::rect_d> {
    public:
        PYBIND11_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.x1 = 0.0;
                value.y1 = 0.0;
                value.x2 = 0.0;
                value.y2 = 0.0;
                return true;
            }

            py::buffer rect_buf = py::reinterpret_borrow<py::buffer>(src);
            mpl::BufferView<double, 2> rect_arr;
            mpl::BufferView<double, 1> rect_vec;
            bool is_2d = false;
            bool is_1d = false;
            try {
                rect_arr = mpl::BufferView<double, 2>(rect_buf);
                is_2d = true;
            } catch (...) {
                rect_vec = mpl::BufferView<double, 1>(rect_buf);
                is_1d = true;
            }

            if (is_2d) {
                if (rect_arr.shape(0) != 2 || rect_arr.shape(1) != 2) {
                    throw py::value_error("Invalid bounding box");
                }

                value.x1 = rect_arr(0, 0);
                value.y1 = rect_arr(0, 1);
                value.x2 = rect_arr(1, 0);
                value.y2 = rect_arr(1, 1);

            } else if (is_1d) {
                if (rect_vec.shape(0) != 4) {
                    throw py::value_error("Invalid bounding box");
                }

                value.x1 = rect_vec(0);
                value.y1 = rect_vec(1);
                value.x2 = rect_vec(2);
                value.y2 = rect_vec(3);

            } else {
                throw py::value_error("Invalid bounding box");
            }

            return true;
        }
    };

    template <> struct type_caster<agg::rgba> {
    public:
        PYBIND11_TYPE_CASTER(agg::rgba, const_name("rgba"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.r = 0.0;
                value.g = 0.0;
                value.b = 0.0;
                value.a = 0.0;
            } else {
                auto rgbatuple = src.cast<py::tuple>();
                value.r = rgbatuple[0].cast<double>();
                value.g = rgbatuple[1].cast<double>();
                value.b = rgbatuple[2].cast<double>();
                switch (rgbatuple.size()) {
                case 4:
                    value.a = rgbatuple[3].cast<double>();
                    break;
                case 3:
                    value.a = 1.0;
                    break;
                default:
                    throw py::value_error("RGBA value must be 3- or 4-tuple");
                }
            }
            return true;
        }
    };

    template <> struct type_caster<agg::trans_affine> {
    public:
        PYBIND11_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                return true;
            }

            py::object transform = py::reinterpret_borrow<py::object>(src);
            if (!py::hasattr(transform, "to_values") &&
                    py::hasattr(transform, "get_affine")) {
                transform = transform.attr("get_affine")();
            }
            if (py::hasattr(transform, "to_values")) {
                py::sequence values = transform.attr("to_values")();
                if (py::len(values) != 6) {
                    throw std::invalid_argument("Invalid affine transformation values");
                }

                value.sx = py::cast<double>(values[0]);
                value.shy = py::cast<double>(values[1]);
                value.shx = py::cast<double>(values[2]);
                value.sy = py::cast<double>(values[3]);
                value.tx = py::cast<double>(values[4]);
                value.ty = py::cast<double>(values[5]);
                return true;
            }

            if (py::hasattr(transform, "__mlx_array__")) {
                transform = transform.attr("__mlx_array__")();
            }
            if (py::hasattr(transform, "tolist")) {
                py::sequence rows = transform.attr("tolist")();
                if (py::len(rows) != 3) {
                    throw std::invalid_argument("Invalid affine transformation matrix");
                }
                py::sequence row0 = rows[0].cast<py::sequence>();
                py::sequence row1 = rows[1].cast<py::sequence>();
                if (py::len(row0) != 3 || py::len(row1) != 3) {
                    throw std::invalid_argument("Invalid affine transformation matrix");
                }
                value.sx = py::cast<double>(row0[0]);
                value.shx = py::cast<double>(row0[1]);
                value.tx = py::cast<double>(row0[2]);
                value.shy = py::cast<double>(row1[0]);
                value.sy = py::cast<double>(row1[1]);
                value.ty = py::cast<double>(row1[2]);
                return true;
            }

            py::buffer buf = py::reinterpret_borrow<py::buffer>(src);
            mpl::BufferView<double, 2> array(buf);
            if (array.shape(0) != 3 || array.shape(1) != 3) {
                throw std::invalid_argument("Invalid affine transformation matrix");
            }

            value.sx = array(0, 0);
            value.shx = array(0, 1);
            value.tx = array(0, 2);
            value.shy = array(1, 0);
            value.sy = array(1, 1);
            value.ty = array(1, 2);
            return true;
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail

#endif /* MPL_PY_CONVERTERS_H */
