#include "py_converters.h"

#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

bool is_mlx_array_like(const py::object& obj)
{
    nb::object nb_obj = nb::borrow<nb::object>(nb::handle(obj.ptr()));
    return nb::isinstance<mx::array>(nb_obj) || nb::hasattr(nb_obj, "__mlx_array__");
}

mx::array as_mlx_array(const py::object& obj)
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

bool convert_mlx_affine(const py::object& obj, agg::trans_affine& affine)
{
    if (!is_mlx_array_like(obj)) {
        return false;
    }

    auto array = as_mlx_array(obj);
    {
        py::gil_scoped_release release;
        array.eval();
    }

    if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
        throw std::invalid_argument("Invalid affine transformation matrix");
    }
    if (array.dtype() != mx::float64) {
        throw std::invalid_argument("Invalid affine transformation matrix dtype");
    }
    if (array.strides(1) != 1 || array.strides(0) != array.shape(1)) {
        throw std::invalid_argument("Invalid affine transformation matrix layout");
    }

    const double *data = array.data<double>();
    affine.sx = data[0];
    affine.shx = data[1];
    affine.tx = data[2];
    affine.shy = data[3];
    affine.sy = data[4];
    affine.ty = data[5];
    return true;
}

}  // namespace

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    // If None assume identity transform so leave affine unchanged
    if (transform.is_none()) {
        return;
    }

    py::object affine_transform = transform;
    if (!py::hasattr(affine_transform, "to_values") &&
            py::hasattr(affine_transform, "get_affine")) {
        affine_transform = affine_transform.attr("get_affine")();
    }
    if (!py::hasattr(affine_transform, "to_values")) {
        if (convert_mlx_affine(affine_transform, affine)) {
            return;
        }

        py::buffer buf = py::reinterpret_borrow<py::buffer>(affine_transform);
        mpl::BufferView<double, 2> array(buf);
        if (array.shape(0) != 3 || array.shape(1) != 3) {
            throw std::invalid_argument("Invalid affine transformation matrix");
        }

        affine.sx = array(0, 0);
        affine.shx = array(0, 1);
        affine.tx = array(0, 2);
        affine.shy = array(1, 0);
        affine.sy = array(1, 1);
        affine.ty = array(1, 2);
        return;
    }

    py::sequence values = affine_transform.attr("to_values")();
    if (py::len(values) != 6) {
        throw std::invalid_argument("Invalid affine transformation values");
    }

    affine.sx = py::cast<double>(values[0]);
    affine.shy = py::cast<double>(values[1]);
    affine.shx = py::cast<double>(values[2]);
    affine.sy = py::cast<double>(values[3]);
    affine.tx = py::cast<double>(values[4]);
    affine.ty = py::cast<double>(values[5]);
}
