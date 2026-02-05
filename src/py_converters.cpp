#include "py_converters.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    // If None assume identity transform so leave affine unchanged
    if (transform.is_none()) {
        return;
    }

    py::buffer buf = py::reinterpret_borrow<py::buffer>(transform);
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
}
