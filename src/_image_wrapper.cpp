#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <utility>
#include <vector>

#include "_image_resample.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"
#include "py_buffer.h"
#include "py_converters.h"

namespace py = pybind11;
namespace nb = nanobind;
namespace mx = mlx::core;
using namespace pybind11::literals;

namespace {

struct ArrayInfo {
    py::object owner;
    py::buffer buffer;
    std::optional<mx::array> mlx_array;
    void *ptr = nullptr;
    py::ssize_t ndim = 0;
    py::ssize_t itemsize = 0;
    std::string format;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
};

bool has_explicit_stream(const mx::StreamOrDevice& stream)
{
    return !std::holds_alternative<std::monostate>(stream);
}

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

std::string mlx_dtype_format(const mx::Dtype &dtype)
{
    switch (dtype) {
    case mx::bool_:
        return "?";
    case mx::uint8:
        return "B";
    case mx::uint16:
        return "H";
    case mx::uint32:
        return "I";
    case mx::uint64:
        return "Q";
    case mx::int8:
        return "b";
    case mx::int16:
        return "h";
    case mx::int32:
        return "i";
    case mx::int64:
        return "q";
    case mx::float16:
        return "e";
    case mx::float32:
        return "f";
    case mx::bfloat16:
        return "B";
    case mx::float64:
        return "d";
    case mx::complex64:
        return "Zf";
    default:
        throw std::invalid_argument("unsupported MLX dtype");
    }
}

bool is_mlx_array_like(py::handle obj)
{
    nb::object nb_obj = nb::borrow<nb::object>(nb::handle(obj.ptr()));
    return nb::isinstance<mx::array>(nb_obj) || nb::hasattr(nb_obj, "__mlx_array__");
}

mx::array as_mlx_array(py::handle obj)
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

ArrayInfo get_array_info(py::handle obj,
                         bool writable,
                         const mx::StreamOrDevice& stream)
{
    if (obj.is_none()) {
        throw py::type_error("resample(): incompatible function arguments");
    }

    ArrayInfo info;
    info.owner = py::reinterpret_borrow<py::object>(obj);

    if (writable && py::hasattr(info.owner, "flags")) {
        py::object flags = info.owner.attr("flags");
        if (py::hasattr(flags, "writeable") && !py::cast<bool>(flags.attr("writeable"))) {
            throw py::value_error("Output array must be writeable");
        }
    }

    if (is_mlx_array_like(obj)) {
        info.mlx_array = as_mlx_array(obj);
        if (!writable && (has_explicit_stream(stream)
                || !info.mlx_array->flags().row_contiguous)) {
            info.mlx_array = mx::contiguous(*info.mlx_array, false, stream);
        }
        {
            py::gil_scoped_release release;
            info.mlx_array->eval();
            if (has_explicit_stream(stream)) {
                mx::synchronize(mx::to_stream(stream));
            }
        }

        info.ptr = info.mlx_array->data<void>();
        info.ndim = static_cast<py::ssize_t>(info.mlx_array->ndim());
        info.itemsize = static_cast<py::ssize_t>(info.mlx_array->itemsize());
        info.format = mlx_dtype_format(info.mlx_array->dtype());
        info.shape.assign(info.mlx_array->shape().begin(), info.mlx_array->shape().end());
        info.strides.assign(info.mlx_array->strides().begin(), info.mlx_array->strides().end());
        for (auto &stride : info.strides) {
            stride *= info.itemsize;
        }
        return info;
    }

    info.buffer = py::reinterpret_borrow<py::buffer>(obj);
    py::buffer_info buffer_info = info.buffer.request(writable);
    info.ptr = buffer_info.ptr;
    info.ndim = buffer_info.ndim;
    info.itemsize = buffer_info.itemsize;
    info.format = buffer_info.format;
    info.shape = std::move(buffer_info.shape);
    info.strides = std::move(buffer_info.strides);
    return info;
}

template <typename T>
bool buffer_is(const ArrayInfo &info)
{
    return info.itemsize == static_cast<py::ssize_t>(sizeof(T))
        && info.format == py::format_descriptor<T>::format();
}

bool is_c_contiguous(const ArrayInfo &info)
{
    if (info.ndim <= 0) {
        return true;
    }
    py::ssize_t expected = info.itemsize;
    for (py::ssize_t axis = info.ndim - 1; axis >= 0; --axis) {
        if (info.shape[axis] == 0) {
            return true;
        }
        if (info.strides[axis] != expected) {
            return false;
        }
        expected *= info.shape[axis];
    }
    return true;
}

std::vector<double> get_transform_mesh(const py::object &transform,
                                      py::ssize_t height,
                                      py::ssize_t width,
                                      const mx::StreamOrDevice& stream)
{
    auto inverse = transform.attr("inverted")();

    auto n = height * width;
    std::vector<double> input_mesh;
    input_mesh.resize(static_cast<size_t>(n) * 2);
    double *p = input_mesh.data();

    for (py::ssize_t y = 0; y < height; ++y) {
        for (py::ssize_t x = 0; x < width; ++x) {
            *p++ = static_cast<double>(x) + 0.5;
            *p++ = static_cast<double>(y) + 0.5;
        }
    }

    // Wrap as a shaped memoryview (N, 2) of doubles so Python code can consume it.
    py::bytearray ba = py::reinterpret_steal<py::bytearray>(
        PyByteArray_FromStringAndSize(nullptr,
                                      static_cast<py::ssize_t>(input_mesh.size() * sizeof(double))));
    if (!ba) {
        throw py::error_already_set();
    }
    std::memcpy(PyByteArray_AsString(ba.ptr()), input_mesh.data(), input_mesh.size() * sizeof(double));
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    mv = mv.attr("cast")("d", py::make_tuple(n, 2));

    py::object output = inverse.attr("transform")(mv);
    auto out_info = get_array_info(output, false, stream);
    if (out_info.ndim != 2 || out_info.shape[0] != n || out_info.shape[1] != 2) {
        throw std::runtime_error("Inverse transformed mesh must have shape (N, 2)");
    }
    if (!buffer_is<double>(out_info)) {
        throw std::runtime_error("Inverse transformed mesh must be float64");
    }
    if (!is_c_contiguous(out_info)) {
        throw std::runtime_error("Inverse transformed mesh must be C-contiguous");
    }

    std::vector<double> mesh;
    mesh.resize(static_cast<size_t>(n) * 2);
    std::memcpy(mesh.data(), out_info.ptr, mesh.size() * sizeof(double));
    return mesh;
}

}  // namespace

/**********************************************************************
 * Free functions
 * */

const char *image_resample__doc__ = R"""(Resample input_array, blending it in-place into output_array, using an affine transform.

Parameters
----------
input_array : 2-d or 3-d buffer
    If 2-d, the image is grayscale. If 3-d, the image must be of size 4 in the last
    dimension and represents RGBA data.

output_array : 2-d or 3-d buffer
    The dtype and number of dimensions must match `input_array`.

transform : matplotlib.transforms.Transform instance
    The transformation from the input array to the output array.

interpolation : int, default: NEAREST
    The interpolation method.

resample : bool, optional
    When `True`, use a full resampling method.

alpha : float, default: 1
    The transparency level.

norm : bool, default: False
    Whether to norm the interpolation function.

radius: float, default: 1
    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.

stream : mlx.core.Stream or mlx.core.Device, optional
    Stream or device used when staging MLX array inputs for C++ access.
)""";

static void image_resample(py::object input_array,
                           py::object output_array,
                           py::object transform,
                           int interpolation,
                           bool resample_,
                           float alpha,
                           bool norm,
                           float radius,
                           py::object stream)
{
    auto stream_or_device = as_stream_or_device(stream);
    auto in_info = get_array_info(input_array, false, stream_or_device);
    auto out_info = get_array_info(output_array, true, stream_or_device);

    if (in_info.ndim != 2 && in_info.ndim != 3) {
        throw std::invalid_argument("Input buffer must be 2D or 3D");
    }
    if (out_info.ndim != in_info.ndim) {
        throw std::invalid_argument("Input and output buffers have different dimensionalities");
    }

    if (!buffer_is<std::uint8_t>(in_info) && !buffer_is<std::int8_t>(in_info)
        && !buffer_is<std::uint16_t>(in_info) && !buffer_is<std::int16_t>(in_info)
        && !buffer_is<float>(in_info) && !buffer_is<double>(in_info)) {
        throw std::invalid_argument("arrays must be of dtype byte, short, float32 or float64");
    }
    if (in_info.itemsize != out_info.itemsize || in_info.format != out_info.format) {
        throw std::invalid_argument("Input and output buffers have mismatched types");
    }

    if (!is_c_contiguous(in_info)) {
        throw std::invalid_argument("Input buffer must be C-contiguous");
    }
    if (!is_c_contiguous(out_info)) {
        throw std::invalid_argument("Output buffer must be C-contiguous");
    }

    if (in_info.ndim == 3) {
        if (in_info.shape[2] != 4) {
            throw std::invalid_argument("3D input array must be RGBA");
        }
        if (out_info.shape[2] != 4) {
            throw std::invalid_argument("3D output array must be RGBA");
        }
    }

    resample_params_t params;
    params.interpolation = static_cast<interpolation_e>(interpolation);
    params.transform_mesh = nullptr;
    params.resample = resample_;
    params.norm = norm;
    params.radius = radius;
    params.alpha = alpha;

    std::vector<double> transform_mesh;

    const char *transform_stage = "checking transform";
    try {
        if (transform.is_none()) {
            params.is_affine = true;
        } else if (!py::hasattr(transform, "is_affine")) {
            transform_stage = "converting affine transform";
            convert_trans_affine_with_stream(transform, params.affine, stream);
            params.is_affine = true;
        } else {
            transform_stage = "reading transform.is_affine";
            bool is_affine = py::cast<bool>(transform.attr("is_affine"));
            if (is_affine) {
                transform_stage = "converting affine transform";
                convert_trans_affine_with_stream(transform, params.affine, stream);
                params.is_affine = true;
            } else {
                transform_stage = "building transform mesh";
                transform_mesh = get_transform_mesh(transform,
                                                    out_info.shape[0],
                                                    out_info.shape[1],
                                                    stream_or_device);
                params.transform_mesh = transform_mesh.data();
                params.is_affine = false;
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("image resample failed while ")
                                 + transform_stage + ": " + e.what());
    }

    auto width = static_cast<unsigned long>(in_info.shape[1]);
    auto height = static_cast<unsigned long>(in_info.shape[0]);
    auto out_width = static_cast<unsigned long>(out_info.shape[1]);
    auto out_height = static_cast<unsigned long>(out_info.shape[0]);

    void *in_ptr = in_info.ptr;
    void *out_ptr = out_info.ptr;

    // Match the historical mapping (signed treated as unsigned for the resampler).
    auto call = [&](auto tag) {
        using Pixel = decltype(tag);
        Py_BEGIN_ALLOW_THREADS
        resample<Pixel>(in_ptr, width, height, out_ptr, out_width, out_height, params);
        Py_END_ALLOW_THREADS
    };

    if (in_info.ndim == 2) {
        if (in_info.itemsize == 1) {
            call(agg::gray8{});
        } else if (in_info.itemsize == 2) {
            call(agg::gray16{});
        } else if (buffer_is<float>(in_info)) {
            call(agg::gray32{});
        } else {
            call(agg::gray64{});
        }
    } else {
        if (in_info.itemsize == 1) {
            call(agg::rgba8{});
        } else if (in_info.itemsize == 2) {
            call(agg::rgba16{});
        } else if (buffer_is<float>(in_info)) {
            call(agg::rgba32{});
        } else {
            call(agg::rgba64{});
        }
    }
}

[[noreturn]] void throw_image_comparison_failure(const std::string &message)
{
    py::object exc = py::module_::import("matplotlib.testing.exceptions")
                         .attr("ImageComparisonFailure");
    PyErr_SetString(exc.ptr(), message.c_str());
    throw py::error_already_set();
}

std::string dimensionality(py::ssize_t ndim)
{
    return std::to_string(ndim) + "-dimensional";
}

std::string shape_to_string(const ArrayInfo &info)
{
    std::string result = "(";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(info.shape[i]);
    }
    if (info.shape.size() == 1) {
        result += ",";
    }
    result += ")";
    return result;
}

mx::Dtype dtype_from_array_info(const ArrayInfo &info)
{
    if (buffer_is<std::uint8_t>(info)) {
        return mx::uint8;
    }
    if (buffer_is<std::int8_t>(info)) {
        return mx::int8;
    }
    if (buffer_is<std::uint16_t>(info)) {
        return mx::uint16;
    }
    if (buffer_is<std::int16_t>(info)) {
        return mx::int16;
    }
    if (buffer_is<float>(info)) {
        return mx::float32;
    }
    if (buffer_is<double>(info)) {
        return mx::float64;
    }
    throw std::invalid_argument("unsupported array dtype");
}

mx::Shape shape_from_array_info(const ArrayInfo &info)
{
    mx::Shape shape;
    shape.reserve(info.shape.size());
    for (auto extent : info.shape) {
        shape.push_back(static_cast<mx::ShapeElem>(extent));
    }
    return shape;
}

std::vector<std::uint8_t> copy_uint8_buffer(const ArrayInfo &info)
{
    size_t size = 1;
    for (auto extent : info.shape) {
        size *= static_cast<size_t>(extent);
    }

    std::vector<std::uint8_t> data(size);
    auto *base = static_cast<const std::uint8_t *>(info.ptr);
    for (size_t linear = 0; linear < size; ++linear) {
        auto remaining = linear;
        py::ssize_t offset = 0;
        for (py::ssize_t axis = info.ndim - 1; axis >= 0; --axis) {
            auto extent = static_cast<size_t>(info.shape[axis]);
            auto index = extent == 0 ? 0 : remaining % extent;
            remaining = extent == 0 ? 0 : remaining / extent;
            offset += static_cast<py::ssize_t>(index) * info.strides[axis];
        }
        data[linear] = *(base + offset);
    }
    return data;
}

mx::array mlx_array_from_info(const ArrayInfo &info)
{
    auto dtype = dtype_from_array_info(info);
    if (dtype != mx::uint8) {
        throw std::invalid_argument("image comparison arrays must be uint8");
    }
    auto data = copy_uint8_buffer(info);
    return mx::array(data.begin(), shape_from_array_info(info), mx::uint8);
}

static py::tuple calculate_rms_and_diff(py::object expected_image,
                                       py::object actual_image,
                                       py::object stream)
{
    auto stream_or_device = as_stream_or_device(stream);
    auto expected_info = get_array_info(expected_image, false, stream_or_device);
    auto actual_info = get_array_info(actual_image, false, stream_or_device);

    if (expected_info.ndim != 3) {
        throw_image_comparison_failure("Expected image must be 3-dimensional, but is "
                                       + dimensionality(expected_info.ndim));
    }
    if (actual_info.ndim != 3) {
        throw_image_comparison_failure("Actual image must be 3-dimensional, but is "
                                       + dimensionality(actual_info.ndim));
    }
    if (!buffer_is<std::uint8_t>(expected_info)) {
        throw_image_comparison_failure("Expected image must be uint8");
    }
    if (!buffer_is<std::uint8_t>(actual_info)) {
        throw_image_comparison_failure("Actual image must be uint8");
    }

    if (expected_info.shape[2] != 3 && expected_info.shape[2] != 4) {
        throw_image_comparison_failure("Expected image must be RGB or RGBA but has depth "
                                       + std::to_string(expected_info.shape[2]));
    }
    if (actual_info.shape[2] != expected_info.shape[2]
        || actual_info.shape[0] != expected_info.shape[0]
        || actual_info.shape[1] != expected_info.shape[1]) {
        throw_image_comparison_failure("Image sizes do not match expected size: "
                                       + shape_to_string(expected_info)
                                       + " actual size "
                                       + shape_to_string(actual_info));
    }

    auto expected = expected_info.mlx_array
        ? *expected_info.mlx_array
        : mlx_array_from_info(expected_info);
    auto actual = actual_info.mlx_array
        ? *actual_info.mlx_array
        : mlx_array_from_info(actual_info);

    auto expected_i32 = mx::astype(expected, mx::int32, stream_or_device);
    auto actual_i32 = mx::astype(actual, mx::int32, stream_or_device);
    auto diff_i32 = mx::abs(mx::subtract(expected_i32, actual_i32, stream_or_device),
                            stream_or_device);
    auto diff_float = mx::astype(diff_i32, mx::float32, stream_or_device);
    auto rms_array = mx::sqrt(mx::mean(mx::square(diff_float, stream_or_device),
                                       stream_or_device),
                              stream_or_device);
    auto diff_uint8 = mx::astype(diff_i32, mx::uint8, stream_or_device);

    double rms = 0.0;
    {
        py::gil_scoped_release release;
        rms_array.eval();
        diff_uint8.eval();
        if (has_explicit_stream(stream_or_device)) {
            mx::synchronize(mx::to_stream(stream_or_device));
        }
        rms = static_cast<double>(rms_array.item<float>());
    }

    auto height = expected_info.shape[0];
    auto width = expected_info.shape[1];
    auto depth = expected_info.shape[2];
    auto nbytes = static_cast<py::ssize_t>(height * width * depth);
    py::bytearray ba = py::reinterpret_steal<py::bytearray>(
        PyByteArray_FromStringAndSize(
            reinterpret_cast<const char *>(diff_uint8.data<std::uint8_t>()), nbytes));
    if (!ba) {
        throw py::error_already_set();
    }
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    mv = mv.attr("cast")("B", py::make_tuple(height, width, depth));

    return py::make_tuple(rms, mv);
}

PYBIND11_MODULE(_image, m)
{
#if NB_VERSION_MAJOR > 2 || (NB_VERSION_MAJOR == 2 && NB_VERSION_MINOR >= 12)
    nb::detail::nb_module_exec(NB_DOMAIN_STR, m.ptr());
#else
    nb::detail::init(NB_DOMAIN_STR);
#endif

    m.def("resample", &image_resample,
          "input_array"_a,
          "output_array"_a,
          "transform"_a = py::none(),
          "interpolation"_a = static_cast<int>(NEAREST),
          "resample"_a = false,
          "alpha"_a = 1.0f,
          "norm"_a = false,
          "radius"_a = 1.0f,
          py::kw_only(),
          "stream"_a = py::none(),
          image_resample__doc__);

    m.def("calculate_rms_and_diff", &calculate_rms_and_diff,
          "expected"_a,
          "actual"_a,
          py::kw_only(),
          "stream"_a = py::none());

    // Export interpolation enum values.
    m.attr("NEAREST") = py::int_(static_cast<int>(NEAREST));
    m.attr("BILINEAR") = py::int_(static_cast<int>(BILINEAR));
    m.attr("BICUBIC") = py::int_(static_cast<int>(BICUBIC));
    m.attr("SPLINE16") = py::int_(static_cast<int>(SPLINE16));
    m.attr("SPLINE36") = py::int_(static_cast<int>(SPLINE36));
    m.attr("HANNING") = py::int_(static_cast<int>(HANNING));
    m.attr("HAMMING") = py::int_(static_cast<int>(HAMMING));
    m.attr("HERMITE") = py::int_(static_cast<int>(HERMITE));
    m.attr("KAISER") = py::int_(static_cast<int>(KAISER));
    m.attr("QUADRIC") = py::int_(static_cast<int>(QUADRIC));
    m.attr("CATROM") = py::int_(static_cast<int>(CATROM));
    m.attr("GAUSSIAN") = py::int_(static_cast<int>(GAUSSIAN));
    m.attr("BESSEL") = py::int_(static_cast<int>(BESSEL));
    m.attr("MITCHELL") = py::int_(static_cast<int>(MITCHELL));
    m.attr("SINC") = py::int_(static_cast<int>(SINC));
    m.attr("LANCZOS") = py::int_(static_cast<int>(LANCZOS));
    m.attr("BLACKMAN") = py::int_(static_cast<int>(BLACKMAN));
}
