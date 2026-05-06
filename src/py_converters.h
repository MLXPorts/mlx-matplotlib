/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the nanobind type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"
#include "nb_compat.h"
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

inline auto convert_mlx_points(py::object obj)
{
    mpl::MlxArrayView<double, 2> view(std::move(obj), "points");
    check_trailing_shape(view, "points", 2);
    return view;
}

inline auto convert_mlx_transforms(py::object obj)
{
    mpl::MlxArrayView<double, 3> view(std::move(obj), "transforms");
    check_trailing_shape(view, "transforms", 3, 3);
    return view;
}

inline auto convert_mlx_colors(py::object obj)
{
    mpl::MlxArrayView<double, 2> view(std::move(obj), "colors");
    check_trailing_shape(view, "colors", 4);
    return view;
}

namespace nanobind { namespace detail {
    template <> struct type_caster<agg::rect_d> {
    public:
        NB_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
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
        NB_TYPE_CASTER(agg::rgba, const_name("rgba"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                value.r = 0.0;
                value.g = 0.0;
                value.b = 0.0;
                value.a = 0.0;
            } else {
                auto rgbatuple = py::cast<py::tuple>(src);
                value.r = py::cast<double>(rgbatuple[0]);
                value.g = py::cast<double>(rgbatuple[1]);
                value.b = py::cast<double>(rgbatuple[2]);
                switch (rgbatuple.size()) {
                case 4:
                    value.a = py::cast<double>(rgbatuple[3]);
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
        NB_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            // If None assume identity transform so leave affine unchanged
            if (src.is_none()) {
                return true;
            }

            convert_trans_affine(py::reinterpret_borrow<py::object>(src), value);
            return true;
        }
    };
}} // namespace nanobind::detail

#endif /* MPL_PY_CONVERTERS_H */
