#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_image_resample.h"
#include "py_buffer.h"
#include "py_converters.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

template <typename T>
bool buffer_is(const py::buffer_info &info)
{
    return info.itemsize == static_cast<py::ssize_t>(sizeof(T))
        && info.format == py::format_descriptor<T>::format();
}

bool is_c_contiguous(const py::buffer_info &info)
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
                                      py::ssize_t width)
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
    py::bytearray ba(static_cast<py::ssize_t>(input_mesh.size() * sizeof(double)));
    std::memcpy(PyByteArray_AsString(ba.ptr()), input_mesh.data(), input_mesh.size() * sizeof(double));
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    mv = mv.attr("cast")("d", py::make_tuple(n, 2));

    py::object output = inverse.attr("transform")(mv);
    py::buffer out_buf = py::reinterpret_borrow<py::buffer>(output);
    auto out_info = out_buf.request(false);
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
)""";

static void image_resample(const py::buffer &input_array,
                           const py::buffer &output_array,
                           const py::object &transform,
                           interpolation_e interpolation,
                           bool resample_,
                           float alpha,
                           bool norm,
                           float radius)
{
    auto in_info = input_array.request(false);
    auto out_info = output_array.request(true);

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
        if (in_info.shape[2] != 4 || out_info.shape[2] != 4) {
            throw std::invalid_argument("3D buffers must be RGBA with trailing dimension 4");
        }
    }

    resample_params_t params;
    params.interpolation = interpolation;
    params.transform_mesh = nullptr;
    params.resample = resample_;
    params.norm = norm;
    params.radius = radius;
    params.alpha = alpha;

    std::vector<double> transform_mesh;

    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        bool is_affine = py::cast<bool>(transform.attr("is_affine"));
        if (is_affine) {
            convert_trans_affine(transform, params.affine);
            params.is_affine = true;
        } else {
            transform_mesh = get_transform_mesh(transform, out_info.shape[0], out_info.shape[1]);
            params.transform_mesh = transform_mesh.data();
            params.is_affine = false;
        }
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

static py::tuple calculate_rms_and_diff(const py::buffer &expected_image,
                                       const py::buffer &actual_image)
{
    mpl::BufferView<unsigned char, 3> expected(expected_image);
    mpl::BufferView<unsigned char, 3> actual(actual_image);

    if (expected.shape(2) != 3 && expected.shape(2) != 4) {
        throw py::value_error("Expected image must be RGB or RGBA");
    }
    if (actual.shape(2) != expected.shape(2)
        || actual.shape(0) != expected.shape(0)
        || actual.shape(1) != expected.shape(1)) {
        throw py::value_error("Images must have the same shape");
    }

    auto height = expected.shape(0);
    auto width = expected.shape(1);
    auto depth = expected.shape(2);

    std::vector<unsigned char> diff;
    diff.resize(static_cast<size_t>(height * width * depth));

    double sum_sq = 0.0;
    for (py::ssize_t y = 0; y < height; ++y) {
        for (py::ssize_t x = 0; x < width; ++x) {
            for (py::ssize_t c = 0; c < depth; ++c) {
                auto e = expected(y, x, c);
                auto a = actual(y, x, c);
                auto d = static_cast<int>(e) - static_cast<int>(a);
                sum_sq += static_cast<double>(d * d);
                diff[static_cast<size_t>((y * width + x) * depth + c)] = static_cast<unsigned char>(std::abs(d));
            }
        }
    }

    double rms = std::sqrt(sum_sq / static_cast<double>(height * width * depth));

    // Return diff as a shaped memoryview so callers can wrap it into an MLX array.
    py::bytearray ba(static_cast<py::ssize_t>(diff.size()));
    std::memcpy(PyByteArray_AsString(ba.ptr()), diff.data(), diff.size());
    py::object mv = py::module_::import("builtins").attr("memoryview")(ba);
    mv = mv.attr("cast")("B", py::make_tuple(height, width, depth));

    return py::make_tuple(rms, mv);
}

PYBIND11_MODULE(_image, m)
{
    m.def("resample", &image_resample,
          "input_array"_a,
          "output_array"_a,
          "transform"_a,
          "interpolation"_a,
          "resample"_a,
          "alpha"_a,
          "norm"_a,
          "radius"_a,
          image_resample__doc__);

    m.def("calculate_rms_and_diff", &calculate_rms_and_diff,
          "expected"_a,
          "actual"_a);

    // Export interpolation enum values.
    m.attr("NEAREST") = py::int_(NEAREST);
    m.attr("BILINEAR") = py::int_(BILINEAR);
    m.attr("BICUBIC") = py::int_(BICUBIC);
    m.attr("SPLINE16") = py::int_(SPLINE16);
    m.attr("SPLINE36") = py::int_(SPLINE36);
    m.attr("HANNING") = py::int_(HANNING);
    m.attr("HAMMING") = py::int_(HAMMING);
    m.attr("HERMITE") = py::int_(HERMITE);
    m.attr("KAISER") = py::int_(KAISER);
    m.attr("QUADRIC") = py::int_(QUADRIC);
    m.attr("CATROM") = py::int_(CATROM);
    m.attr("GAUSSIAN") = py::int_(GAUSSIAN);
    m.attr("BESSEL") = py::int_(BESSEL);
    m.attr("MITCHELL") = py::int_(MITCHELL);
    m.attr("SINC") = py::int_(SINC);
    m.attr("LANCZOS") = py::int_(LANCZOS);
    m.attr("BLACKMAN") = py::int_(BLACKMAN);
}
