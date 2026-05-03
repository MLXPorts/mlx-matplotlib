#include "py_converters.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    // If None assume identity transform so leave affine unchanged
    if (transform.is_none()) {
        return;
    }

    try {
        py::buffer buf = py::reinterpret_borrow<py::buffer>(transform);
        mpl::BufferView<double, 2> array(buf);
        if (array.shape(0) == 3 && array.shape(1) == 3) {
            affine.sx = array(0, 0);
            affine.shx = array(0, 1);
            affine.tx = array(0, 2);
            affine.shy = array(1, 0);
            affine.sy = array(1, 1);
            affine.ty = array(1, 2);
            return;
        }
    } catch (const std::exception&) {
    }

    py::object affine_transform = transform;
    if (!py::hasattr(affine_transform, "to_values") &&
            py::hasattr(affine_transform, "get_affine")) {
        affine_transform = affine_transform.attr("get_affine")();
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
