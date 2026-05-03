#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

bool target_dtype_is_float64(const py::object& target)
{
    if (!py::hasattr(target, "dtype")) {
        return false;
    }

    py::object dtype = target.attr("dtype");
    py::object float64 = py::module_::import("mlx.core").attr("float64");
    int equal = PyObject_RichCompareBool(dtype.ptr(), float64.ptr(), Py_EQ);
    if (equal < 0) {
        throw py::error_already_set();
    }
    return equal == 1;
}

py::object float64_scalar(py::object value)
{
    if (!PyFloat_Check(value.ptr())) {
        return value;
    }

    double scalar = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }

    py::bytearray bytes = py::reinterpret_steal<py::bytearray>(
        PyByteArray_FromStringAndSize(
            reinterpret_cast<const char*>(&scalar), sizeof(scalar)));
    if (!bytes) {
        throw py::error_already_set();
    }

    py::object memoryview = py::module_::import("builtins").attr("memoryview")(bytes);
    memoryview = memoryview.attr("cast")("d");

    py::object mx = py::module_::import("mlx.core");
    py::object array = mx.attr("array")(memoryview, "dtype"_a = mx.attr("float64"));
    return mx.attr("reshape")(array, py::tuple());
}

py::object coerce_float64_value(py::object target, py::object value)
{
    if (!PyFloat_Check(value.ptr()) || !target_dtype_is_float64(target)) {
        return value;
    }
    return float64_scalar(value);
}

}  // namespace

PYBIND11_MODULE(_mlx_overrides, m)
{
    m.def("float64_scalar", &float64_scalar,
          "value"_a);
    m.def("coerce_float64_value", &coerce_float64_value,
          "target"_a,
          "value"_a);
    m.def("coerce_setitem_value", &coerce_float64_value,
          "target"_a,
          "value"_a);
}
