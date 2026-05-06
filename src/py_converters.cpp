#include "py_converters.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <variant>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

mx::Device parse_mlx_device_repr(const std::string& repr)
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

mx::Stream parse_mlx_stream_repr(const std::string& repr)
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

bool is_mlx_array_like(const py::object& obj)
{
    nb::object nb_obj = nb::borrow<nb::object>(nb::handle(obj.ptr()));
    return (nb::isinstance<mx::array>(nb_obj)
            || nb::hasattr(nb_obj, "__mlx_array__")
            || (py::hasattr(obj, "dtype") && py::hasattr(obj, "shape")));
}

mx::StreamOrDevice as_stream_or_device(const py::object& stream)
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

    auto repr = py::cast<std::string>(py::repr(stream));
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

bool convert_mlx_affine(const py::object& obj,
                        agg::trans_affine& affine,
                        const mx::StreamOrDevice& stream)
{
    if (!is_mlx_array_like(obj)) {
        return false;
    }

    auto shape = py::reinterpret_borrow<py::tuple>(obj.attr("shape"));
    if (shape.size() != 2
            || py::cast<py::ssize_t>(shape[0]) != 3
            || py::cast<py::ssize_t>(shape[1]) != 3) {
        throw std::invalid_argument("Invalid affine transformation matrix");
    }

    auto scalar = [&obj](py::ssize_t row, py::ssize_t col) {
        auto value = obj.attr("__getitem__")(py::make_tuple(row, col));
        if (py::hasattr(value, "item")) {
            value = value.attr("item")();
        }
        return py::cast<double>(value);
    };

    affine.sx = scalar(0, 0);
    affine.shx = scalar(0, 1);
    affine.tx = scalar(0, 2);
    affine.shy = scalar(1, 0);
    affine.sy = scalar(1, 1);
    affine.ty = scalar(1, 2);
    return true;
}

}  // namespace

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    convert_trans_affine_with_stream(transform, affine, py::none());
}

void convert_trans_affine_with_stream(const py::object& transform,
                                      agg::trans_affine& affine,
                                      const py::object& stream)
{
    // If None assume identity transform so leave affine unchanged
    if (transform.is_none()) {
        return;
    }

    auto stream_or_device = as_stream_or_device(stream);

    py::object affine_transform = transform;
    if (!py::hasattr(affine_transform, "to_values") &&
            py::hasattr(affine_transform, "get_affine")) {
        affine_transform = affine_transform.attr("get_affine")();
    }
    if (!py::hasattr(affine_transform, "to_values")) {
        if (convert_mlx_affine(affine_transform, affine, stream_or_device)) {
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

    py::sequence values = py::cast<py::sequence>(affine_transform.attr("to_values")());
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
