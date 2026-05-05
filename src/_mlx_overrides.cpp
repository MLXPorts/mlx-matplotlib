#include <Python.h>

#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

bool has_explicit_stream(const mx::StreamOrDevice& stream)
{
    return !std::holds_alternative<std::monostate>(stream);
}

mx::array place_on_stream(mx::array array, const mx::StreamOrDevice& stream)
{
    if (!has_explicit_stream(stream)) {
        return array;
    }
    return mx::contiguous(array, false, stream);
}

bool target_dtype_is_float64(nb::handle target)
{
    if (!nb::hasattr(target, "dtype")) {
        return false;
    }
    return nb::cast<mx::Dtype>(target.attr("dtype")) == mx::float64;
}

mx::array float64_scalar(double value, const mx::StreamOrDevice& stream)
{
    return place_on_stream(mx::array(value, mx::float64), stream);
}

mx::array as_float64_array(nb::handle value, const mx::StreamOrDevice& stream);

bool is_python_sequence(nb::handle value)
{
    if (nb::isinstance<mx::array>(value) || nb::hasattr(value, "__mlx_array__")) {
        return false;
    }
    return PySequence_Check(value.ptr())
        && !PyUnicode_Check(value.ptr())
        && !PyBytes_Check(value.ptr())
        && !PyByteArray_Check(value.ptr());
}

void collect_float64_sequence(nb::handle value,
                              std::vector<double>& data,
                              std::vector<mx::ShapeElem>& shape,
                              std::size_t depth)
{
    if (!is_python_sequence(value)) {
        if (nb::isinstance<mx::array>(value) || nb::hasattr(value, "__mlx_array__")) {
            auto array = as_float64_array(value, mx::Device(mx::Device::cpu));
            if (array.size() != 1) {
                throw std::invalid_argument(
                    "nested MLX arrays in float64_array must be scalar values");
            }
            data.push_back(array.item<double>());
            return;
        }
        double scalar = PyFloat_AsDouble(value.ptr());
        if (PyErr_Occurred()) {
            throw nb::python_error();
        }
        data.push_back(scalar);
        return;
    }

    nb::sequence sequence = nb::borrow<nb::sequence>(value);
    auto length = static_cast<mx::ShapeElem>(nb::len(sequence));
    if (shape.size() == depth) {
        shape.push_back(length);
    } else if (shape[depth] != length) {
        throw std::invalid_argument(
            "float64_array requires a rectangular Python sequence");
    }

    for (auto item : sequence) {
        collect_float64_sequence(item, data, shape, depth + 1);
    }
}

mx::array float64_array(nb::handle value, const mx::StreamOrDevice& stream)
{
    if (nb::isinstance<mx::array>(value) || nb::hasattr(value, "__mlx_array__")) {
        return as_float64_array(value, stream);
    }

    std::vector<double> data;
    std::vector<mx::ShapeElem> shape;
    collect_float64_sequence(value, data, shape, 0);
    mx::Shape mx_shape(shape.begin(), shape.end());
    auto array = mx::array(data.begin(), std::move(mx_shape), mx::float64);
    return place_on_stream(std::move(array), stream);
}

mx::Dtype requested_dtype(nb::handle dtype, mx::Dtype fallback)
{
    if (dtype.is_none()) {
        return fallback;
    }
    return nb::cast<mx::Dtype>(dtype);
}

mx::array mlx_precise_array(nb::handle value,
                            nb::object dtype,
                            const mx::StreamOrDevice& stream)
{
    auto target_dtype = requested_dtype(dtype, mx::float32);
    if (target_dtype == mx::float64) {
        return float64_array(value, stream);
    }

    if (nb::isinstance<mx::array>(value)) {
        auto array = nb::cast<mx::array>(value);
        if (!dtype.is_none() && array.dtype() != target_dtype) {
            array = mx::astype(array, target_dtype, stream);
        }
        return place_on_stream(std::move(array), stream);
    }
    if (nb::hasattr(value, "__mlx_array__")) {
        auto array = nb::cast<mx::array>(value.attr("__mlx_array__")());
        if (!dtype.is_none() && array.dtype() != target_dtype) {
            array = mx::astype(array, target_dtype, stream);
        }
        return place_on_stream(std::move(array), stream);
    }

    std::vector<double> data;
    std::vector<mx::ShapeElem> shape;
    collect_float64_sequence(value, data, shape, 0);
    mx::Shape mx_shape(shape.begin(), shape.end());
    auto array = mx::array(data.begin(), std::move(mx_shape), target_dtype);
    return place_on_stream(std::move(array), stream);
}

mx::array as_float64_array(nb::handle value, const mx::StreamOrDevice& stream)
{
    if (nb::isinstance<mx::array>(value)) {
        auto array = nb::cast<mx::array>(value);
        if (array.dtype() != mx::float64) {
            array = mx::astype(array, mx::float64, stream);
        }
        return place_on_stream(array, stream);
    }
    if (nb::hasattr(value, "__mlx_array__")) {
        auto array = nb::cast<mx::array>(value.attr("__mlx_array__")());
        if (array.dtype() != mx::float64) {
            array = mx::astype(array, mx::float64, stream);
        }
        return place_on_stream(array, stream);
    }
    if (PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr())) {
        double scalar = PyFloat_AsDouble(value.ptr());
        if (PyErr_Occurred()) {
            throw nb::python_error();
        }
        return float64_scalar(scalar, stream);
    }
    throw nb::type_error("expected an MLX array or Python scalar");
}

nb::object coerce_float64_value(nb::handle target,
                                nb::object value,
                                const mx::StreamOrDevice& stream)
{
    if (value.is_none() ||
            !PyFloat_Check(value.ptr()) ||
            !target_dtype_is_float64(target)) {
        return value;
    }

    double scalar = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred()) {
        throw nb::python_error();
    }
    return nb::cast(float64_scalar(scalar, stream));
}

mx::array log_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return mx::log(as_float64_array(value, stream), stream);
}

mx::array log2_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return mx::log2(as_float64_array(value, stream), stream);
}

mx::array log10_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return mx::log10(as_float64_array(value, stream), stream);
}

mx::array sin_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return mx::sin(as_float64_array(value, stream), stream);
}

mx::array cos_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return mx::cos(as_float64_array(value, stream), stream);
}

mx::array arange_float64(double start,
                         double stop,
                         double step,
                         const mx::StreamOrDevice& stream)
{
    if (step == 0.0) {
        throw std::invalid_argument("arange: step must not be zero");
    }
    auto length_value = std::ceil((stop - start) / step);
    if (!std::isfinite(length_value)) {
        throw std::invalid_argument("arange: range length is not finite");
    }
    if (length_value <= 0.0) {
        return mx::zeros({0}, mx::float64, stream);
    }
    if (length_value > static_cast<double>(
            std::numeric_limits<mx::ShapeElem>::max())) {
        throw std::overflow_error("arange: range length is too large");
    }

    auto length = static_cast<mx::ShapeElem>(length_value);
    auto indices = mx::arange(static_cast<double>(length), mx::float64, stream);
    return mx::add(
        float64_scalar(start, stream),
        mx::multiply(float64_scalar(step, stream), indices, stream),
        stream);
}

mx::array fft_float64(const mx::array& value,
                      int n,
                      int axis,
                      const mx::StreamOrDevice& stream)
{
    auto input = value;
    if (input.dtype() != mx::float64) {
        input = mx::astype(input, mx::float64, stream);
    }
    if (n > 0) {
        return mx::fft::fft(input, n, axis, stream);
    }
    return mx::fft::fft(input, axis, stream);
}

mx::array less_float64(nb::handle left,
                       nb::handle right,
                       const mx::StreamOrDevice& stream)
{
    return mx::less(as_float64_array(left, stream),
                    as_float64_array(right, stream),
                    stream);
}

mx::array less_equal_float64(nb::handle left,
                             nb::handle right,
                             const mx::StreamOrDevice& stream)
{
    return mx::less_equal(as_float64_array(left, stream),
                          as_float64_array(right, stream),
                          stream);
}

mx::array greater_float64(nb::handle left,
                          nb::handle right,
                          const mx::StreamOrDevice& stream)
{
    return mx::greater(as_float64_array(left, stream),
                       as_float64_array(right, stream),
                       stream);
}

mx::array greater_equal_float64(nb::handle left,
                                nb::handle right,
                                const mx::StreamOrDevice& stream)
{
    return mx::greater_equal(as_float64_array(left, stream),
                             as_float64_array(right, stream),
                             stream);
}

mx::array sliding_window_view(const mx::array& input,
                              int window_shape,
                              int axis,
                              int step,
                              const mx::StreamOrDevice& stream)
{
    if (window_shape < 0) {
        throw std::invalid_argument("window_shape must be non-negative");
    }
    if (step <= 0) {
        throw std::invalid_argument("step must be greater than zero");
    }
    if (axis < 0) {
        axis += input.ndim();
    }
    if (input.ndim() != 1 || axis != 0) {
        throw std::invalid_argument(
            "sliding_window_view currently supports one-dimensional input");
    }

    auto width = static_cast<mx::ShapeElem>(window_shape);
    auto stop = static_cast<int>(input.shape(0)) - window_shape + 1;
    if (stop <= 0) {
        return mx::zeros({0, width}, input.dtype(), stream);
    }

    std::vector<mx::array> windows;
    windows.reserve((stop + step - 1) / step);
    for (int start = 0; start < stop; start += step) {
        auto slice = mx::slice(
            input,
            {static_cast<mx::ShapeElem>(start)},
            {static_cast<mx::ShapeElem>(start + window_shape)},
            {1},
            stream);
        windows.push_back(mx::reshape(slice, {1, width}, stream));
    }
    return mx::concatenate(std::move(windows), 0, stream);
}

mx::array as_strided(const mx::array& input,
                     const std::vector<int>& shape,
                     int step,
                     const mx::StreamOrDevice& stream)
{
    if (shape.size() != 2) {
        throw std::invalid_argument("as_strided currently supports 2-D output");
    }
    if (step <= 0) {
        throw std::invalid_argument("step must be greater than zero");
    }

    auto rows = static_cast<mx::ShapeElem>(shape[0]);
    auto cols = static_cast<mx::ShapeElem>(shape[1]);
    if (rows <= 0 || cols <= 0) {
        return mx::zeros({rows, cols}, input.dtype(), stream);
    }

    std::vector<mx::array> columns;
    columns.reserve(cols);
    for (mx::ShapeElem col = 0; col < cols; ++col) {
        auto start = static_cast<mx::ShapeElem>(col * step);
        auto slice = mx::slice(
            input,
            {start},
            {static_cast<mx::ShapeElem>(start + rows)},
            {1},
            stream);
        columns.push_back(mx::reshape(slice, {rows, 1}, stream));
    }
    return mx::concatenate(std::move(columns), 1, stream);
}

mx::array percentile_linear(nb::handle value,
                            nb::handle quantiles,
                            const mx::StreamOrDevice& stream)
{
    auto data = mx::sort(mx::reshape(as_float64_array(value, stream), {-1}, stream),
                         stream);
    auto q = mx::reshape(as_float64_array(quantiles, stream), {-1}, stream);
    auto last_index = static_cast<double>(data.size() - 1);
    auto zero = float64_scalar(0.0, stream);
    auto hundred = float64_scalar(100.0, stream);
    auto last = float64_scalar(last_index, stream);

    auto idx = mx::multiply(
        mx::divide(q, hundred, stream),
        last,
        stream);
    auto lo_float = mx::clip(
        mx::floor(idx, stream),
        zero,
        last,
        stream);
    auto hi_float = mx::clip(
        mx::ceil(idx, stream),
        zero,
        last,
        stream);
    auto lo = mx::astype(lo_float, mx::int32, stream);
    auto hi = mx::astype(hi_float, mx::int32, stream);
    auto frac = mx::subtract(idx, lo_float, stream);

    auto lower = mx::take(data, lo, stream);
    auto upper = mx::take(data, hi, stream);
    return mx::add(
        lower,
        mx::multiply(frac, mx::subtract(upper, lower, stream), stream),
        stream);
}

}  // namespace

NB_MODULE(_mlx_overrides, m)
{
    m.def("float64_scalar", &float64_scalar,
          "value"_a,
          "stream"_a = nb::none());
    m.def("mlx_precise_array", &mlx_precise_array,
          "value"_a,
          "dtype"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("float64_array", &float64_array,
          "value"_a,
          "stream"_a = nb::none());
    m.def("coerce_float64_value", &coerce_float64_value,
          "target"_a,
          "value"_a.none(),
          "stream"_a = nb::none());
    m.def("coerce_setitem_value", &coerce_float64_value,
          "target"_a,
          "value"_a.none(),
          "stream"_a = nb::none());
    m.def("log_float64", &log_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("log2_float64", &log2_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("log10_float64", &log10_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("sin_float64", &sin_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("cos_float64", &cos_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("arange_float64", &arange_float64,
          "start"_a,
          "stop"_a,
          "step"_a = 1.0,
          "stream"_a = nb::none());
    m.def("fft_float64", &fft_float64,
          "value"_a,
          "n"_a = 0,
          "axis"_a = -1,
          "stream"_a = nb::none());
    m.def("less_float64", &less_float64,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("less_equal_float64", &less_equal_float64,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("greater_float64", &greater_float64,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("greater_equal_float64", &greater_equal_float64,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("sliding_window_view", &sliding_window_view,
          "input"_a,
          "window_shape"_a,
          "axis"_a = 0,
          "step"_a = 1,
          "stream"_a = nb::none());
    m.def("as_strided", &as_strided,
          "input"_a,
          "shape"_a,
          "step"_a = 1,
          "stream"_a = nb::none());
    m.def("percentile_linear", &percentile_linear,
          "value"_a,
          "quantiles"_a,
          "stream"_a = nb::none());
}
