#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace mlx::core {

array::array(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::move(shape),
          dtype,
          std::move(primitive),
          std::move(inputs))) {}

} // namespace mlx::core

namespace {

struct PreciseFloat64 {
    float hi;
    float lo;

    PreciseFloat64() : hi(0.0f), lo(0.0f) {}
    explicit PreciseFloat64(double value)
        : hi(static_cast<float>(value)),
          lo(std::isfinite(value)
                 ? static_cast<float>(value - static_cast<double>(hi))
                 : 0.0f) {}

    double value() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
};

static_assert(sizeof(PreciseFloat64) == sizeof(double));

enum class PreciseFloat64BinaryOp : int {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
    Power = 4,
};

enum class PreciseFloat64CompareOp : int {
    Equal = 0,
    NotEqual = 1,
    Less = 2,
    LessEqual = 3,
};

mx::Shape broadcast_binary_shape(const mx::array& left, const mx::array& right)
{
    return mx::broadcast_shapes(left.shape(), right.shape());
}

mx::Shape broadcast_ternary_shape(const mx::array& first,
                                  const mx::array& second,
                                  const mx::array& third)
{
    return mx::broadcast_shapes(
        mx::broadcast_shapes(first.shape(), second.shape()), third.shape());
}

std::size_t broadcast_offset(std::size_t out_index,
                             const mx::Shape& out_shape,
                             const mx::Shape& input_shape,
                             const mx::Strides& input_strides)
{
    std::int64_t offset = 0;
    auto out_ndim = out_shape.size();
    auto input_ndim = input_shape.size();
    for (std::size_t step = 0; step < out_ndim; ++step) {
        auto out_axis = out_ndim - 1 - step;
        auto coord = out_shape[out_axis] == 0 ? 0 : out_index % out_shape[out_axis];
        if (out_shape[out_axis] != 0) {
            out_index /= out_shape[out_axis];
        }
        if (step < input_ndim) {
            auto input_axis = input_ndim - 1 - step;
            if (input_shape[input_axis] != 1) {
                offset += static_cast<std::int64_t>(coord)
                    * input_strides[input_axis];
            }
        }
    }
    return static_cast<std::size_t>(offset);
}

std::vector<std::int64_t> broadcast_effective_strides(
    const mx::Shape& out_shape,
    const mx::Shape& input_shape,
    const mx::Strides& input_strides)
{
    std::vector<std::int64_t> strides(out_shape.size(), 0);
    auto out_ndim = out_shape.size();
    auto input_ndim = input_shape.size();
    for (std::size_t step = 0; step < out_ndim; ++step) {
        auto out_axis = out_ndim - 1 - step;
        if (step < input_ndim) {
            auto input_axis = input_ndim - 1 - step;
            auto dim = input_shape[input_axis];
            strides[out_axis] = dim == 1 ? 0 : input_strides[input_axis];
        }
    }
    return strides;
}

std::vector<std::uint64_t> shape_bytes(const mx::Shape& shape)
{
    std::vector<std::uint64_t> result;
    result.reserve(shape.size());
    for (auto dim : shape) {
        result.push_back(static_cast<std::uint64_t>(dim));
    }
    return result;
}

std::vector<std::uint64_t> contiguous_strides(const mx::Shape& shape)
{
    std::vector<std::uint64_t> strides(shape.size(), 1);
    std::uint64_t stride = 1;
    for (std::size_t step = 0; step < shape.size(); ++step) {
        auto axis = shape.size() - 1 - step;
        strides[axis] = stride;
        stride *= static_cast<std::uint64_t>(shape[axis]);
    }
    return strides;
}

class PreciseFloat64GpuTransfer : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuTransfer(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuTransfer";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        if (out.nbytes() != 0) {
            std::memcpy(out.data<void>(), inputs[0].data<void>(), out.nbytes());
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_transfer",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_copy(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
                        if (static_cast<ulong>(gid) < size) {
                            out[gid] = in[gid];
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_copy", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        encoder.set_bytes(size, 2);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64Full : public mx::Primitive {
public:
    PreciseFloat64Full(mx::Stream stream,
                       mx::Shape shape,
                       double value)
        : mx::Primitive(stream),
          shape_(std::move(shape)),
          value_(value) {}

    const char* name() const override {
        return "PreciseFloat64Full";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>&) override {
        return {shape_};
    }

    void eval_cpu(const std::vector<mx::array>&,
                  std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto ptr = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ptr[i] = value_;
        }
    }

    void eval_gpu(const std::vector<mx::array>&,
                  std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_full",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_full(
                        constant float2& value [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
                        if (static_cast<ulong>(gid) < size) {
                            out[gid] = value;
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_full", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_bytes(value_, 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        encoder.set_bytes(size, 2);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    mx::Shape shape_;
    PreciseFloat64 value_;
};

class PreciseFloat64Arange : public mx::Primitive {
public:
    PreciseFloat64Arange(mx::Stream stream,
                         mx::Shape shape,
                         double start,
                         double step)
        : mx::Primitive(stream),
          shape_(std::move(shape)),
          start_(start),
          step_(step) {}

    const char* name() const override {
        return "PreciseFloat64Arange";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>&) override {
        return {shape_};
    }

    void eval_cpu(const std::vector<mx::array>&,
                  std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto ptr = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ptr[i] = PreciseFloat64(
                start_.value() + step_.value() * static_cast<double>(i));
        }
    }

    void eval_gpu(const std::vector<mx::array>&,
                  std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_arange",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline float2 two_sum(float a, float b) {
                        float s = a + b;
                        float bb = s - a;
                        float err = (a - (s - bb)) + (b - bb);
                        return float2(s, err);
                    }

                    inline float2 quick_two_sum(float a, float b) {
                        float s = a + b;
                        return float2(s, b - (s - a));
                    }

                    inline float2 dd_add(float2 a, float2 b) {
                        float2 s = two_sum(a.x, b.x);
                        float e = s.y + a.y + b.y;
                        return quick_two_sum(s.x, e);
                    }

                    inline float2 dd_mul(float2 a, float2 b) {
                        float p = a.x * b.x;
                        float e = fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x;
                        return quick_two_sum(p, e);
                    }

                    kernel void mlx_matplotlib_precise_float64_arange(
                        constant float2& start [[buffer(0)]],
                        constant float2& step [[buffer(1)]],
                        device float2* out [[buffer(2)]],
                        constant ulong& size [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                        if (static_cast<ulong>(gid) < size) {
                            out[gid] = dd_add(
                                start,
                                dd_mul(step, float2(static_cast<float>(gid), 0.0f)));
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_arange", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_bytes(start_, 0);
        encoder.set_bytes(step_, 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        encoder.set_bytes(size, 3);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    mx::Shape shape_;
    PreciseFloat64 start_;
    PreciseFloat64 step_;
};

std::pair<bool, mx::Shape> normalize_precise_slice(
    const mx::Shape& shape,
    mx::Shape& start,
    mx::Shape stop,
    mx::Shape& strides)
{
    mx::Shape out_shape(shape.size());
    bool has_neg_strides = false;

    for (int i = 0; i < shape.size(); ++i) {
        auto n = shape[i];
        auto s = start[i];
        s = s < 0 ? s + n : s;
        auto e = stop[i];
        if (!(strides[i] < 0 && e < 0)) {
            e = e < 0 ? e + n : e;
        }

        if (strides[i] < 0) {
            has_neg_strides = true;
            auto st = std::min(s, n - 1);
            auto ed = e > -1 ? e : -1;
            start[i] = st;
            ed = ed > st ? st : ed;
            auto stride = -strides[i];
            out_shape[i] = (start[i] - ed + stride - 1) / stride;
        } else {
            auto st = std::max(static_cast<mx::ShapeElem>(0), std::min(s, n));
            auto ed = std::max(static_cast<mx::ShapeElem>(0), std::min(e, n));
            start[i] = st;
            ed = ed < st ? st : ed;
            out_shape[i] = (ed - start[i] + strides[i] - 1) / strides[i];
        }

        if (out_shape[i] == 1) {
            strides[i] = 1;
        }
    }

    return {has_neg_strides, out_shape};
}

class PreciseFloat64GpuSlice : public mx::Slice {
public:
    PreciseFloat64GpuSlice(mx::Stream stream,
                           mx::Shape start_indices,
                           mx::Shape end_indices,
                           mx::Shape strides)
        : mx::Slice(stream, start_indices, end_indices, strides),
          start_indices_(std::move(start_indices)),
          strides_(std::move(strides)) {}

    const char* name() const override {
        return "PreciseFloat64GpuSlice";
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        mx::slice(inputs[0], out, start_indices_, strides_);
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        mx::slice(inputs[0], out, start_indices_, strides_);
    }

private:
    mx::Shape start_indices_;
    mx::Shape strides_;
};

class PreciseFloat64GpuReshape : public mx::Reshape {
public:
    PreciseFloat64GpuReshape(mx::Stream stream, mx::Shape shape)
        : mx::Reshape(stream, shape), shape_(std::move(shape)) {}

    const char* name() const override {
        return "PreciseFloat64GpuReshape";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {shape_};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        eval_common(inputs[0], out);
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        eval_common(inputs[0], out);
    }

private:
    static void eval_common(const mx::array& input, mx::array& out) {
        auto [copy_necessary, out_strides] = mx::prepare_reshape(input, out);
        if (!copy_necessary) {
            mx::shared_buffer_reshape(input, out_strides, out);
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = input.data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = source[broadcast_offset(
                i, out.shape(), input.shape(), input.strides())];
        }
    }

    mx::Shape shape_;
};

class PreciseFloat64GpuTranspose : public mx::Transpose {
public:
    PreciseFloat64GpuTranspose(mx::Stream stream, std::vector<int> axes)
        : mx::Transpose(stream, axes), axes_(std::move(axes)) {}

    const char* name() const override {
        return "PreciseFloat64GpuTranspose";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        mx::Shape shape;
        shape.reserve(axes_.size());
        for (auto axis : axes_) {
            shape.push_back(inputs[0].shape(axis));
        }
        return {shape};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        eval_common(inputs[0], out, axes_);
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        eval_common(inputs[0], out, axes_);
    }

private:
    static void eval_common(const mx::array& input,
                            mx::array& out,
                            const std::vector<int>& axes) {
        mx::Strides strides;
        strides.reserve(axes.size());
        for (auto axis : axes) {
            strides.push_back(input.strides(axis));
        }
        auto [data_size, row_contiguous, col_contiguous] =
            mx::check_contiguity(out.shape(), strides);
        mx::array::Flags flags{
            data_size == input.data_size(), row_contiguous, col_contiguous};
        out.copy_shared_buffer(input, strides, flags, data_size);
    }

    std::vector<int> axes_;
};

class PreciseFloat64GpuBinary : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuBinary(PreciseFloat64BinaryOp op, mx::Stream stream)
        : mx::UnaryPrimitive(stream), op_(op) {}

    const char* name() const override {
        return "PreciseFloat64GpuBinary";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {broadcast_binary_shape(inputs[0], inputs[1])};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto left = inputs[0].data<PreciseFloat64>();
        auto right = inputs[1].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto lhs = left[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())].value();
            auto rhs = right[broadcast_offset(
                i, out.shape(), inputs[1].shape(), inputs[1].strides())].value();
            double value;
            switch (op_) {
            case PreciseFloat64BinaryOp::Add:
                value = lhs + rhs;
                break;
            case PreciseFloat64BinaryOp::Subtract:
                value = lhs - rhs;
                break;
            case PreciseFloat64BinaryOp::Multiply:
                value = lhs * rhs;
                break;
            case PreciseFloat64BinaryOp::Divide:
                value = lhs / rhs;
                break;
            case PreciseFloat64BinaryOp::Power:
                value = std::pow(lhs, rhs);
                break;
            }
            result[i] = PreciseFloat64(value);
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_binary",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline float2 two_sum(float a, float b) {
                        float s = a + b;
                        float bb = s - a;
                        float err = (a - (s - bb)) + (b - bb);
                        return float2(s, err);
                    }

                    inline float2 quick_two_sum(float a, float b) {
                        float s = a + b;
                        return float2(s, b - (s - a));
                    }

                    inline float2 dd_add(float2 a, float2 b) {
                        float2 s = two_sum(a.x, b.x);
                        float e = s.y + a.y + b.y;
                        return quick_two_sum(s.x, e);
                    }

                    inline float2 dd_sub(float2 a, float2 b) {
                        return dd_add(a, float2(-b.x, -b.y));
                    }

                    inline float2 dd_mul(float2 a, float2 b) {
                        float p = a.x * b.x;
                        float e = fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x;
                        return quick_two_sum(p, e);
                    }

                    inline float2 dd_div(float2 a, float2 b) {
                        float q1 = a.x / b.x;
                        float2 r = dd_sub(a, dd_mul(float2(q1, 0.0f), b));
                        float q2 = r.x / b.x;
                        return dd_add(float2(q1, 0.0f), float2(q2, 0.0f));
                    }

                    inline float2 dd_pow(float2 a, float2 b) {
                        float exponent_f = b.x + b.y;
                        int exponent = static_cast<int>(round(exponent_f));
                        if (abs(exponent_f - static_cast<float>(exponent)) < 0.00001f
                                && abs(exponent) <= 64) {
                            float2 result = float2(1.0f, 0.0f);
                            int count = abs(exponent);
                            for (int i = 0; i < count; ++i) {
                                result = dd_mul(result, a);
                            }
                            if (exponent < 0) {
                                result = dd_div(float2(1.0f, 0.0f), result);
                            }
                            return result;
                        }
                        float value = pow(a.x + a.y, exponent_f);
                        float hi = value;
                        return float2(hi, value - hi);
                    }

	                    kernel void mlx_matplotlib_precise_float64_binary(
	                        device const float2* left [[buffer(0)]],
	                        device const float2* right [[buffer(1)]],
	                        device float2* out [[buffer(2)]],
	                        constant ulong& size [[buffer(3)]],
	                        constant uint& ndim [[buffer(4)]],
	                        constant ulong* out_shape [[buffer(5)]],
	                        constant long* left_strides [[buffer(6)]],
	                        constant long* right_strides [[buffer(7)]],
	                        constant int& op [[buffer(8)]],
	                        uint gid [[thread_position_in_grid]]) {
	                        auto index = static_cast<ulong>(gid);
	                        if (index >= size) {
	                            return;
	                        }
	                        ulong remainder = index;
	                        long left_index = 0;
	                        long right_index = 0;
	                        for (uint step = 0; step < ndim; ++step) {
	                            uint axis = ndim - 1 - step;
	                            ulong dim = out_shape[axis];
	                            ulong coord = dim == 0 ? 0 : remainder % dim;
	                            if (dim != 0) {
	                                remainder /= dim;
	                            }
	                            left_index += static_cast<long>(coord) * left_strides[axis];
	                            right_index += static_cast<long>(coord) * right_strides[axis];
	                        }
	                        float2 lhs = left[left_index];
	                        float2 rhs = right[right_index];
	                        float2 value;
                        if (op == 0) {
                            value = dd_add(lhs, rhs);
                        } else if (op == 1) {
                            value = dd_sub(lhs, rhs);
                        } else if (op == 2) {
                            value = dd_mul(lhs, rhs);
                        } else if (op == 3) {
                            value = dd_div(lhs, rhs);
                        } else {
                            value = dd_pow(lhs, rhs);
                        }
                        out[index] = value;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_binary", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        auto left_strides = broadcast_effective_strides(
            out.shape(), inputs[0].shape(), inputs[0].strides());
        auto right_strides = broadcast_effective_strides(
            out.shape(), inputs[1].shape(), inputs[1].strides());
        auto op = static_cast<int>(op_);
        encoder.set_bytes(size, 3);
        encoder.set_bytes(ndim, 4);
        encoder.set_vector_bytes(out_shape, 5);
        encoder.set_vector_bytes(left_strides, 6);
        encoder.set_vector_bytes(right_strides, 7);
        encoder.set_bytes(op, 8);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    PreciseFloat64BinaryOp op_;
};

class PreciseFloat64GpuCompare : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuCompare(PreciseFloat64CompareOp op, mx::Stream stream)
        : mx::UnaryPrimitive(stream), op_(op) {}

    const char* name() const override {
        return "PreciseFloat64GpuCompare";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {broadcast_binary_shape(inputs[0], inputs[1])};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto left = inputs[0].data<PreciseFloat64>();
        auto right = inputs[1].data<PreciseFloat64>();
        auto result = out.data<bool>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto lhs = left[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())].value();
            auto rhs = right[broadcast_offset(
                i, out.shape(), inputs[1].shape(), inputs[1].strides())].value();
            switch (op_) {
            case PreciseFloat64CompareOp::Equal:
                result[i] = lhs == rhs;
                break;
            case PreciseFloat64CompareOp::NotEqual:
                result[i] = lhs != rhs;
                break;
            case PreciseFloat64CompareOp::Less:
                result[i] = lhs < rhs;
                break;
            case PreciseFloat64CompareOp::LessEqual:
                result[i] = lhs <= rhs;
                break;
            }
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_compare",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_compare(
                        device const float2* left [[buffer(0)]],
                        device const float2* right [[buffer(1)]],
                        device bool* out [[buffer(2)]],
                        constant ulong& size [[buffer(3)]],
                        constant uint& ndim [[buffer(4)]],
                        constant ulong* out_shape [[buffer(5)]],
                        constant long* left_strides [[buffer(6)]],
                        constant long* right_strides [[buffer(7)]],
                        constant int& op [[buffer(8)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long left_index = 0;
                        long right_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            left_index += static_cast<long>(coord) * left_strides[axis];
                            right_index += static_cast<long>(coord) * right_strides[axis];
                        }
                        float2 lhs = left[left_index];
                        float2 rhs = right[right_index];
                        float lhs_value = lhs.x + lhs.y;
                        float rhs_value = rhs.x + rhs.y;
                        if (op == 0) {
                            out[index] = lhs_value == rhs_value;
                        } else if (op == 1) {
                            out[index] = lhs_value != rhs_value;
                        } else if (op == 2) {
                            out[index] = lhs_value < rhs_value;
                        } else {
                            out[index] = lhs_value <= rhs_value;
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_compare", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        auto left_strides = broadcast_effective_strides(
            out.shape(), inputs[0].shape(), inputs[0].strides());
        auto right_strides = broadcast_effective_strides(
            out.shape(), inputs[1].shape(), inputs[1].strides());
        auto op = static_cast<int>(op_);
        encoder.set_bytes(size, 3);
        encoder.set_bytes(ndim, 4);
        encoder.set_vector_bytes(out_shape, 5);
        encoder.set_vector_bytes(left_strides, 6);
        encoder.set_vector_bytes(right_strides, 7);
        encoder.set_bytes(op, 8);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    PreciseFloat64CompareOp op_;
};

class PreciseFloat64GpuWhere : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuWhere(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuWhere";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {broadcast_ternary_shape(inputs[0], inputs[1], inputs[2])};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto condition = inputs[0].data<bool>();
        auto x = inputs[1].data<PreciseFloat64>();
        auto y = inputs[2].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto c_offset = broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides());
            auto x_offset = broadcast_offset(
                i, out.shape(), inputs[1].shape(), inputs[1].strides());
            auto y_offset = broadcast_offset(
                i, out.shape(), inputs[2].shape(), inputs[2].strides());
            result[i] = condition[c_offset] ? x[x_offset] : y[y_offset];
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_where",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_where(
                        device const bool* condition [[buffer(0)]],
                        device const float2* x [[buffer(1)]],
                        device const float2* y [[buffer(2)]],
                        device float2* out [[buffer(3)]],
                        constant ulong& size [[buffer(4)]],
                        constant uint& ndim [[buffer(5)]],
                        constant ulong* out_shape [[buffer(6)]],
                        constant long* condition_strides [[buffer(7)]],
                        constant long* x_strides [[buffer(8)]],
                        constant long* y_strides [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long condition_index = 0;
                        long x_index = 0;
                        long y_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            condition_index += static_cast<long>(coord)
                                * condition_strides[axis];
                            x_index += static_cast<long>(coord) * x_strides[axis];
                            y_index += static_cast<long>(coord) * y_strides[axis];
                        }
                        out[index] = condition[condition_index]
                            ? x[x_index]
                            : y[y_index];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_where", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[2], 2);
        encoder.set_output_array(out, 3);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        auto condition_strides = broadcast_effective_strides(
            out.shape(), inputs[0].shape(), inputs[0].strides());
        auto x_strides = broadcast_effective_strides(
            out.shape(), inputs[1].shape(), inputs[1].strides());
        auto y_strides = broadcast_effective_strides(
            out.shape(), inputs[2].shape(), inputs[2].strides());
        encoder.set_bytes(size, 4);
        encoder.set_bytes(ndim, 5);
        encoder.set_vector_bytes(out_shape, 6);
        encoder.set_vector_bytes(condition_strides, 7);
        encoder.set_vector_bytes(x_strides, 8);
        encoder.set_vector_bytes(y_strides, 9);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuCumsum1D : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuCumsum1D(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuCumsum1D";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto input = inputs[0].data<PreciseFloat64>();
        auto output = out.data<PreciseFloat64>();
        auto stride = inputs[0].strides(0);
        double running = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            running += input[i * stride].value();
            output[i] = PreciseFloat64(running);
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_cumsum_1d",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline float2 precise_add(float2 a, float2 b) {
                        float s = a.x + b.x;
                        float v = s - a.x;
                        float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
                        float hi = s + e;
                        float lo = e - (hi - s);
                        return float2(hi, lo);
                    }

                    kernel void mlx_matplotlib_precise_float64_cumsum_1d(
                        device const float2* input [[buffer(0)]],
                        device float2* output [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant long& stride [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        float2 running = float2(0.0f, 0.0f);
                        for (ulong i = 0; i <= index; ++i) {
                            running = precise_add(
                                running,
                                input[static_cast<long>(i) * stride]);
                        }
                        output[index] = running;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_cumsum_1d", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto stride = static_cast<std::int64_t>(inputs[0].strides(0));
        encoder.set_bytes(size, 2);
        encoder.set_bytes(stride, 3);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuArcTan2 : public mx::ArcTan2 {
public:
    explicit PreciseFloat64GpuArcTan2(mx::Stream stream)
        : mx::ArcTan2(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuArcTan2";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {broadcast_binary_shape(inputs[0], inputs[1])};
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_arctan2",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_arctan2(
                        device const float2* left [[buffer(0)]],
                        device const float2* right [[buffer(1)]],
                        device float2* out [[buffer(2)]],
                        constant ulong& size [[buffer(3)]],
                        constant uint& ndim [[buffer(4)]],
                        constant ulong* out_shape [[buffer(5)]],
                        constant long* left_strides [[buffer(6)]],
                        constant long* right_strides [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long left_index = 0;
                        long right_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            left_index += static_cast<long>(coord) * left_strides[axis];
                            right_index += static_cast<long>(coord) * right_strides[axis];
                        }
                        float lhs = left[left_index].x + left[left_index].y;
                        float rhs = right[right_index].x + right[right_index].y;
                        out[index] = float2(atan2(lhs, rhs), 0.0f);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_arctan2", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        auto left_strides = broadcast_effective_strides(
            out.shape(), inputs[0].shape(), inputs[0].strides());
        auto right_strides = broadcast_effective_strides(
            out.shape(), inputs[1].shape(), inputs[1].strides());
        encoder.set_bytes(size, 3);
        encoder.set_bytes(ndim, 4);
        encoder.set_vector_bytes(out_shape, 5);
        encoder.set_vector_bytes(left_strides, 6);
        encoder.set_vector_bytes(right_strides, 7);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuAbs : public mx::Abs {
public:
    explicit PreciseFloat64GpuAbs(mx::Stream stream)
        : mx::Abs(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAbs";
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_abs",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_abs(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        float2 value = in[index];
                        bool negative = value.x < 0.0f ||
                            (value.x == 0.0f && value.y < 0.0f);
                        out[index] = negative ? float2(-value.x, -value.y) : value;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_abs", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        encoder.set_bytes(size, 2);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuIsFinite : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuIsFinite(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuIsFinite";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<bool>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = std::isfinite(source[i].value());
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_isfinite",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_isfinite(
                        device const float2* in [[buffer(0)]],
                        device bool* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        float2 value = in[index];
                        out[index] = isfinite(value.x) && isfinite(value.y);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_isfinite", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        encoder.set_bytes(size, 2);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuRound : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuRound(mx::Stream stream, int decimals)
        : mx::UnaryPrimitive(stream), decimals_(decimals) {}

    const char* name() const override {
        return "PreciseFloat64GpuRound";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        double scale = std::pow(10.0, decimals_);
        for (std::size_t i = 0; i < out.size(); ++i) {
            double value = source[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())].value();
            result[i] = PreciseFloat64(std::round(value * scale) / scale);
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_round",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_round(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* out_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant float& scale [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long input_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_index += static_cast<long>(coord) * input_strides[axis];
                        }
                        float value = in[input_index].x + in[input_index].y;
                        float rounded = round(value * scale) / scale;
                        out[index] = float2(rounded, 0.0f);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_round", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        float scale = std::pow(10.0f, static_cast<float>(decimals_));
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(out_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_bytes(scale, 6);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    int decimals_;
};

class PreciseFloat64GpuMatmul : public mx::Matmul {
public:
    explicit PreciseFloat64GpuMatmul(mx::Stream stream)
        : mx::Matmul(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuMatmul";
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        if (inputs[0].ndim() != 2 || inputs[1].ndim() != 2) {
            throw std::invalid_argument(
                "precise float64 GPU matmul currently requires 2-D inputs");
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_matmul",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline float2 two_sum(float a, float b) {
                        float s = a + b;
                        float bb = s - a;
                        float err = (a - (s - bb)) + (b - bb);
                        return float2(s, err);
                    }

                    inline float2 quick_two_sum(float a, float b) {
                        float s = a + b;
                        return float2(s, b - (s - a));
                    }

                    inline float2 dd_add(float2 a, float2 b) {
                        float2 s = two_sum(a.x, b.x);
                        float e = s.y + a.y + b.y;
                        return quick_two_sum(s.x, e);
                    }

                    inline float2 dd_mul(float2 a, float2 b) {
                        float p = a.x * b.x;
                        float e = fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x;
                        return quick_two_sum(p, e);
                    }

	                    kernel void mlx_matplotlib_precise_float64_matmul(
	                        device const float2* lhs [[buffer(0)]],
	                        device const float2* rhs [[buffer(1)]],
	                        device float2* out [[buffer(2)]],
	                        constant ulong& m [[buffer(3)]],
	                        constant ulong& k [[buffer(4)]],
	                        constant ulong& n [[buffer(5)]],
	                        constant long& lhs_stride0 [[buffer(6)]],
	                        constant long& lhs_stride1 [[buffer(7)]],
	                        constant long& rhs_stride0 [[buffer(8)]],
	                        constant long& rhs_stride1 [[buffer(9)]],
	                        uint2 gid [[thread_position_in_grid]]) {
                        auto col = static_cast<ulong>(gid.x);
                        auto row = static_cast<ulong>(gid.y);
                        if (row >= m || col >= n) {
                            return;
                        }
                        float2 sum = float2(0.0f, 0.0f);
	                        for (ulong inner = 0; inner < k; ++inner) {
	                            auto lhs_index = static_cast<long>(row) * lhs_stride0
	                                + static_cast<long>(inner) * lhs_stride1;
	                            auto rhs_index = static_cast<long>(inner) * rhs_stride0
	                                + static_cast<long>(col) * rhs_stride1;
	                            sum = dd_add(
	                                sum,
	                                dd_mul(lhs[lhs_index], rhs[rhs_index]));
	                        }
                        out[row * n + col] = sum;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_matmul", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto m = static_cast<std::uint64_t>(inputs[0].shape(0));
        auto k = static_cast<std::uint64_t>(inputs[0].shape(1));
        auto n = static_cast<std::uint64_t>(inputs[1].shape(1));
	        encoder.set_bytes(m, 3);
	        encoder.set_bytes(k, 4);
	        encoder.set_bytes(n, 5);
	        auto lhs_stride0 = static_cast<std::int64_t>(inputs[0].strides(0));
	        auto lhs_stride1 = static_cast<std::int64_t>(inputs[0].strides(1));
	        auto rhs_stride0 = static_cast<std::int64_t>(inputs[1].strides(0));
	        auto rhs_stride1 = static_cast<std::int64_t>(inputs[1].strides(1));
	        encoder.set_bytes(lhs_stride0, 6);
	        encoder.set_bytes(lhs_stride1, 7);
	        encoder.set_bytes(rhs_stride0, 8);
	        encoder.set_bytes(rhs_stride1, 9);
        auto width = static_cast<NS::UInteger>(n);
        auto height = static_cast<NS::UInteger>(m);
        auto group_width = kernel->maxTotalThreadsPerThreadgroup();
        if (group_width > width) {
            group_width = width;
        }
        encoder.dispatch_threads(
            MTL::Size(width, height, 1), MTL::Size(group_width, 1, 1));
    }
};

class PreciseFloat64GpuAstypeFloat32 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAstypeFloat32(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAstypeFloat32";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<float>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = static_cast<float>(source[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())].value());
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_astype_float32",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_astype_float32(
                        device const float2* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* out_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long input_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_index += static_cast<long>(coord)
                                * input_strides[axis];
                        }
                        float2 value = in[input_index];
                        out[index] = value.x + value.y;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_astype_float32", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(out_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuAstypeInt32 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAstypeInt32(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAstypeInt32";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<std::int32_t>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = static_cast<std::int32_t>(source[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())].value());
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_astype_int32",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_astype_int32(
                        device const float2* in [[buffer(0)]],
                        device int* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* out_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long input_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_index += static_cast<long>(coord)
                                * input_strides[axis];
                        }
                        float2 value = in[input_index];
                        out[index] = static_cast<int>(value.x + value.y);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_astype_int32", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(out_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuAstypeFromFloat32 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAstypeFromFloat32(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAstypeFromFloat32";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<float>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = PreciseFloat64(source[broadcast_offset(
                i, out.shape(), inputs[0].shape(), inputs[0].strides())]);
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_astype_from_float32",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_astype_from_float32(
                        device const float* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* out_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long input_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_index += static_cast<long>(coord)
                                * input_strides[axis];
                        }
                        float value = in[input_index];
                        out[index] = float2(value, 0.0f);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_astype_from_float32", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(out_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }
};

class PreciseFloat64GpuReduceMinMax : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuReduceMinMax(mx::Stream stream, int axis, bool is_max)
        : mx::UnaryPrimitive(stream), axis_(axis), is_max_(is_max) {}

    const char* name() const override {
        return "PreciseFloat64GpuReduceMinMax";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        if (axis_ < 0) {
            return {mx::Shape{}};
        }
        auto shape = inputs[0].shape();
        shape.erase(shape.begin() + axis_);
        return {std::move(shape)};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        if (axis_ < 0) {
            double value = is_max_ ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < inputs[0].size(); ++i) {
                auto current = source[broadcast_offset(
                    i, inputs[0].shape(), inputs[0].shape(),
                    inputs[0].strides())].value();
                value = is_max_ ? std::max(value, current)
                                : std::min(value, current);
            }
            result[0] = PreciseFloat64(value);
            return;
        }

        auto input_strides = inputs[0].strides();
        auto reduce_size = static_cast<std::uint64_t>(inputs[0].shape(axis_));
        auto reduce_stride = input_strides[axis_];
        auto out_shape = out.shape();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto remainder = static_cast<std::uint64_t>(i);
            std::uint64_t base = 0;
            for (std::size_t step = 0; step < out_shape.size(); ++step) {
                auto out_axis = out_shape.size() - 1 - step;
                auto dim = static_cast<std::uint64_t>(out_shape[out_axis]);
                auto coord = dim == 0 ? 0 : remainder % dim;
                if (dim != 0) {
                    remainder /= dim;
                }
                auto input_axis = out_axis >= static_cast<std::size_t>(axis_)
                    ? out_axis + 1 : out_axis;
                base += coord * input_strides[input_axis];
            }
            double value = is_max_ ? -std::numeric_limits<double>::infinity()
                                   : std::numeric_limits<double>::infinity();
            for (std::uint64_t r = 0; r < reduce_size; ++r) {
                auto current = source[base + r * reduce_stride].value();
                value = is_max_ ? std::max(value, current)
                                : std::min(value, current);
            }
            result[i] = PreciseFloat64(value);
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_reduce_minmax",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_reduce_minmax(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& out_size [[buffer(2)]],
                        constant uint& out_ndim [[buffer(3)]],
                        constant uint& reduce_axis [[buffer(4)]],
                        constant ulong& reduce_size [[buffer(5)]],
                        constant ulong& reduce_stride [[buffer(6)]],
                        constant bool& reduce_all [[buffer(7)]],
                        constant bool& is_max [[buffer(8)]],
                        constant ulong* out_shape [[buffer(9)]],
                        constant ulong* input_strides [[buffer(10)]],
                        constant uint& input_ndim [[buffer(11)]],
                        constant ulong* input_shape [[buffer(12)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= out_size) {
                            return;
                        }
                        ulong base = 0;
                        if (reduce_all) {
                            float2 best = float2(0.0f, 0.0f);
                            float best_value = 0.0f;
                            for (ulong r = 0; r < reduce_size; ++r) {
                                ulong remainder = r;
                                ulong input_index = 0;
                                for (uint step = 0; step < input_ndim; ++step) {
                                    uint axis = input_ndim - 1 - step;
                                    ulong dim = input_shape[axis];
                                    ulong coord = dim == 0 ? 0 : remainder % dim;
                                    if (dim != 0) {
                                        remainder /= dim;
                                    }
                                    input_index += coord * input_strides[axis];
                                }
                                float2 current = in[input_index];
                                float current_value = current.x + current.y;
                                if (r == 0
                                        || (is_max && current_value > best_value)
                                        || (!is_max && current_value < best_value)) {
                                    best = current;
                                    best_value = current_value;
                                }
                            }
                            out[index] = best;
                            return;
                        } else {
                            ulong remainder = index;
                            for (uint step = 0; step < out_ndim; ++step) {
                                uint out_axis = out_ndim - 1 - step;
                                ulong dim = out_shape[out_axis];
                                ulong coord = dim == 0 ? 0 : remainder % dim;
                                if (dim != 0) {
                                    remainder /= dim;
                                }
                                uint input_axis = out_axis >= reduce_axis
                                    ? out_axis + 1 : out_axis;
                                base += coord * input_strides[input_axis];
                            }
                        }
                        float2 best = in[base];
                        float best_value = best.x + best.y;
                        for (ulong r = 1; r < reduce_size; ++r) {
                            float2 current = in[base + r * reduce_stride];
                            float current_value = current.x + current.y;
                            if ((is_max && current_value > best_value)
                                    || (!is_max && current_value < best_value)) {
                                best = current;
                                best_value = current_value;
                            }
                        }
                        out[index] = best;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_reduce_minmax", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto out_size = static_cast<std::uint64_t>(out.data_size());
        auto out_ndim = static_cast<std::uint32_t>(out.ndim());
        auto reduce_axis = static_cast<std::uint32_t>(axis_ < 0 ? 0 : axis_);
        std::vector<std::uint64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        auto reduce_all = axis_ < 0;
        auto reduce_size = reduce_all
            ? static_cast<std::uint64_t>(inputs[0].data_size())
            : static_cast<std::uint64_t>(inputs[0].shape(axis_));
        auto reduce_stride = reduce_all ? std::uint64_t{1}
                                        : input_strides[axis_];
        auto max_flag = is_max_;
        auto out_shape = shape_bytes(out.shape());
        auto input_ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto input_shape = shape_bytes(inputs[0].shape());
        encoder.set_bytes(out_size, 2);
        encoder.set_bytes(out_ndim, 3);
        encoder.set_bytes(reduce_axis, 4);
        encoder.set_bytes(reduce_size, 5);
        encoder.set_bytes(reduce_stride, 6);
        encoder.set_bytes(reduce_all, 7);
        encoder.set_bytes(max_flag, 8);
        encoder.set_vector_bytes(out_shape, 9);
        encoder.set_vector_bytes(input_strides, 10);
        encoder.set_bytes(input_ndim, 11);
        encoder.set_vector_bytes(input_shape, 12);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    int axis_;
    bool is_max_;
};

class PreciseFloat64GpuStackAxis0 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuStackAxis0(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuStackAxis0";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        if (inputs.empty()) {
            return {{0}};
        }
        mx::Shape shape;
        shape.reserve(inputs[0].ndim() + 1);
        shape.push_back(static_cast<mx::ShapeElem>(inputs.size()));
        shape.insert(shape.end(), inputs[0].shape().begin(), inputs[0].shape().end());
        return {shape};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto result = out.data<PreciseFloat64>();
        std::size_t offset = 0;
        for (const auto& input : inputs) {
            auto source = input.data<PreciseFloat64>();
            std::memcpy(result + offset, source,
                        input.size() * sizeof(PreciseFloat64));
            offset += input.size();
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_stack_axis0",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_stack_axis0(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& row_size [[buffer(2)]],
                        constant ulong& offset [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index < row_size) {
                            out[offset + index] = in[index];
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_stack_axis0", library);
        auto& encoder = device.get_command_encoder(stream().index);
        encoder.set_compute_pipeline_state(kernel);
        std::uint64_t offset = 0;
        for (const auto& input : inputs) {
            encoder.set_input_array(input, 0);
            encoder.set_output_array(out, 1);
            auto row_size = static_cast<std::uint64_t>(input.data_size());
            encoder.set_bytes(row_size, 2);
            encoder.set_bytes(offset, 3);
            auto threads = static_cast<NS::UInteger>(input.data_size());
            auto group_size = kernel->maxTotalThreadsPerThreadgroup();
            if (group_size > threads) {
                group_size = threads;
            }
            encoder.dispatch_threads(
                MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
            offset += row_size;
        }
    }
};

class PreciseFloat64GpuConcatenate : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuConcatenate(mx::Stream stream, int axis)
        : mx::UnaryPrimitive(stream), axis_(axis) {}

    const char* name() const override {
        return "PreciseFloat64GpuConcatenate";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        auto shape = inputs[0].shape();
        shape[axis_] = 0;
        for (const auto& input : inputs) {
            shape[axis_] += input.shape(axis_);
        }
        return {std::move(shape)};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto result = out.data<PreciseFloat64>();
        std::uint64_t axis_offset = 0;
        auto out_strides = contiguous_strides(out.shape());
        for (const auto& input : inputs) {
            auto source = input.data<PreciseFloat64>();
            for (std::size_t i = 0; i < input.size(); ++i) {
                auto remainder = static_cast<std::uint64_t>(i);
                std::uint64_t out_index = 0;
                for (std::size_t step = 0; step < input.ndim(); ++step) {
                    auto axis = input.ndim() - 1 - step;
                    auto dim = static_cast<std::uint64_t>(input.shape(axis));
                    auto coord = dim == 0 ? 0 : remainder % dim;
                    if (dim != 0) {
                        remainder /= dim;
                    }
                    if (static_cast<int>(axis) == axis_) {
                        coord += axis_offset;
                    }
                    out_index += coord * out_strides[axis];
                }
                result[out_index] = source[i];
            }
            axis_offset += static_cast<std::uint64_t>(input.shape(axis_));
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_concatenate",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_concatenate(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant uint& concat_axis [[buffer(4)]],
                        constant ulong& axis_offset [[buffer(5)]],
                        constant ulong* in_shape [[buffer(6)]],
                        constant ulong* out_strides [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        ulong out_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = in_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            if (axis == concat_axis) {
                                coord += axis_offset;
                            }
                            out_index += coord * out_strides[axis];
                        }
                        out[out_index] = in[index];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_concatenate", library);
        auto& encoder = device.get_command_encoder(stream().index);
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto concat_axis = static_cast<std::uint32_t>(axis_);
        auto out_strides = contiguous_strides(out.shape());
        std::uint64_t axis_offset = 0;
        for (const auto& input : inputs) {
            if (input.size() == 0) {
                continue;
            }
            encoder.set_compute_pipeline_state(kernel);
            encoder.set_input_array(input, 0);
            encoder.set_output_array(out, 1);
            auto size = static_cast<std::uint64_t>(input.data_size());
            auto in_shape = shape_bytes(input.shape());
            encoder.set_bytes(size, 2);
            encoder.set_bytes(ndim, 3);
            encoder.set_bytes(concat_axis, 4);
            encoder.set_bytes(axis_offset, 5);
            encoder.set_vector_bytes(in_shape, 6);
            encoder.set_vector_bytes(out_strides, 7);
            auto threads = static_cast<NS::UInteger>(input.data_size());
            auto group_size = kernel->maxTotalThreadsPerThreadgroup();
            if (group_size > threads) {
                group_size = threads;
            }
            encoder.dispatch_threads(
                MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
            axis_offset += static_cast<std::uint64_t>(input.shape(axis_));
        }
    }

private:
    int axis_;
};

class MlxPreciseArray : public mx::array {
public:
    using mx::array::array;

    MlxPreciseArray(mx::array value, mx::StreamOrDevice stream = {})
        : mx::array(std::move(value)), stream_(std::move(stream)) {}

    static MlxPreciseArray make(nb::handle value,
                                nb::object dtype,
                                const mx::StreamOrDevice& stream,
                                int ndmin);

    static MlxPreciseArray from_float64_data(
        const std::vector<double>& data,
        const std::vector<mx::ShapeElem>& shape,
        mx::StreamOrDevice stream = {})
    {
        auto buffer = mx::allocator::malloc(data.size() * sizeof(PreciseFloat64));
        auto out = static_cast<PreciseFloat64*>(buffer.raw_ptr());
        for (std::size_t i = 0; i < data.size(); ++i) {
            out[i] = PreciseFloat64(data[i]);
        }
        return MlxPreciseArray(
            mx::array(buffer, mx::Shape(shape.begin(), shape.end()),
                      mx::float64),
            std::move(stream));
    }

    static MlxPreciseArray from_float64_scalar(
        double value,
        mx::StreamOrDevice stream = {})
    {
        auto buffer = mx::allocator::malloc(sizeof(PreciseFloat64));
        auto out = static_cast<PreciseFloat64*>(buffer.raw_ptr());
        *out = PreciseFloat64(value);
        return MlxPreciseArray(
            mx::array(buffer, mx::Shape{}, mx::float64),
            std::move(stream));
    }

    static MlxPreciseArray transfer_float64_to_gpu(
        MlxPreciseArray input,
        const mx::StreamOrDevice& stream)
    {
        auto target = mx::to_stream(stream);
        auto primitive = std::make_shared<PreciseFloat64GpuTransfer>(target);
        return MlxPreciseArray(
            mx::array(input.shape(), mx::float64, std::move(primitive),
                      {std::move(input)}),
            stream);
    }

    static MlxPreciseArray binary_float64(
        MlxPreciseArray left,
        MlxPreciseArray right,
        PreciseFloat64BinaryOp op,
        const mx::StreamOrDevice& stream)
    {
        auto target = mx::to_stream(stream);
        auto primitive = std::make_shared<PreciseFloat64GpuBinary>(op, target);
        return MlxPreciseArray(
            mx::array(broadcast_binary_shape(left, right), mx::float64,
                      std::move(primitive),
                      {std::move(left), std::move(right)}),
            stream);
    }

    static MlxPreciseArray stack_axis0_float64(
        std::vector<MlxPreciseArray> rows,
        const mx::StreamOrDevice& stream)
    {
        if (rows.empty()) {
            return MlxPreciseArray(mx::zeros({0}, mx::float64, stream), stream);
        }
        std::vector<mx::array> inputs;
        inputs.reserve(rows.size());
        for (auto& row : rows) {
            inputs.push_back(std::move(row));
        }
        auto shape = inputs[0].shape();
        for (const auto& input : inputs) {
            if (input.shape() != shape) {
                throw std::invalid_argument(
                    "precise stack requires all rows to have the same shape");
            }
        }
        mx::Shape output_shape;
        output_shape.reserve(shape.size() + 1);
        output_shape.push_back(static_cast<mx::ShapeElem>(inputs.size()));
        output_shape.insert(output_shape.end(), shape.begin(), shape.end());
        auto target = mx::to_stream(stream);
        auto primitive = std::make_shared<PreciseFloat64GpuStackAxis0>(target);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64,
                      std::move(primitive), std::move(inputs)),
            stream);
    }

    MlxPreciseArray add(nb::handle other) const;
    MlxPreciseArray subtract(nb::handle other) const;
    MlxPreciseArray reverse_subtract(nb::handle other) const;
    MlxPreciseArray multiply(nb::handle other) const;
    MlxPreciseArray divide(nb::handle other) const;
    MlxPreciseArray reverse_divide(nb::handle other) const;
    MlxPreciseArray power(nb::handle other) const;
    MlxPreciseArray reverse_power(nb::handle other) const;
    MlxPreciseArray matmul(nb::handle other) const;
    MlxPreciseArray reverse_matmul(nb::handle other) const;
    MlxPreciseArray negative() const;
    const mx::StreamOrDevice& stream_or_device() const {
        return stream_;
    }

    static void eval(mx::array array)
    {
        if (array.status() != mx::array::Status::unscheduled) {
            array.wait();
            return;
        }
        if (!array.has_primitive()) {
            throw std::runtime_error("cannot evaluate an unevaluated precise array without a primitive");
        }

        auto stream = array.primitive().stream();
        bool precise_float64 = stream.device == mx::Device::gpu
            && array.dtype() == mx::float64;
        if (!precise_float64) {
            array.eval();
            return;
        }

        auto inputs = array.inputs();
        for (auto& input : inputs) {
            eval(input);
        }

        auto outputs = array.outputs();
        array.primitive().eval_gpu(inputs, outputs);
        array.set_status(mx::array::Status::evaluated);
        for (auto& sibling : array.siblings()) {
            sibling.set_status(mx::array::Status::evaluated);
        }
        mx::synchronize(stream);
        array.set_status(mx::array::Status::available);
        for (auto& sibling : array.siblings()) {
            sibling.set_status(mx::array::Status::available);
        }
        if (!array.is_tracer()) {
            array.detach();
        }
    }

private:
    mx::StreamOrDevice stream_;
};

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

bool targets_gpu(const mx::StreamOrDevice& stream);
mx::array packed_float64_array(const std::vector<double>& data,
                               const std::vector<mx::ShapeElem>& shape);
mx::array transfer_float64_to_gpu(mx::array input,
                                  const mx::StreamOrDevice& stream);

mx::array float64_scalar(double value, const mx::StreamOrDevice& stream)
{
    if (targets_gpu(stream)) {
        return transfer_float64_to_gpu(
            MlxPreciseArray::from_float64_scalar(value, stream),
            stream);
    }
    return place_on_stream(mx::array(value, mx::float64), stream);
}

mx::array as_float64_array(nb::handle value, const mx::StreamOrDevice& stream);
MlxPreciseArray ensure_precise_float64(MlxPreciseArray array,
                                       const mx::StreamOrDevice& stream);
MlxPreciseArray reshape_precise(nb::handle value,
                                nb::handle shape_value,
                                const mx::StreamOrDevice& stream);

bool targets_gpu(const mx::StreamOrDevice& stream)
{
    return mx::to_stream(stream).device.type == mx::Device::gpu;
}

double read_buffer_scalar(const char* ptr, const Py_buffer& view)
{
    if (view.itemsize == 8) {
        if (view.format != nullptr && view.format[0] == 'Q') {
            return static_cast<double>(*reinterpret_cast<const std::uint64_t*>(ptr));
        }
        if (view.format != nullptr && view.format[0] == 'q') {
            return static_cast<double>(*reinterpret_cast<const std::int64_t*>(ptr));
        }
        return *reinterpret_cast<const double*>(ptr);
    }
    if (view.itemsize == 4) {
        if (view.format != nullptr && view.format[0] == 'I') {
            return static_cast<double>(*reinterpret_cast<const std::uint32_t*>(ptr));
        }
        if (view.format != nullptr && view.format[0] == 'i') {
            return static_cast<double>(*reinterpret_cast<const std::int32_t*>(ptr));
        }
        return static_cast<double>(*reinterpret_cast<const float*>(ptr));
    }
    if (view.itemsize == 2) {
        if (view.format != nullptr && view.format[0] == 'H') {
            return static_cast<double>(*reinterpret_cast<const std::uint16_t*>(ptr));
        }
        return static_cast<double>(*reinterpret_cast<const std::int16_t*>(ptr));
    }
    if (view.itemsize == 1) {
        if (view.format != nullptr && view.format[0] == 'B') {
            return static_cast<double>(*reinterpret_cast<const std::uint8_t*>(ptr));
        }
        return static_cast<double>(*reinterpret_cast<const std::int8_t*>(ptr));
    }
    throw std::invalid_argument("unsupported buffer item size for MLX array ingress");
}

void collect_buffer_values(const char* base,
                           const Py_buffer& view,
                           int axis,
                           std::vector<double>& values)
{
    if (axis == view.ndim) {
        values.push_back(read_buffer_scalar(base, view));
        return;
    }
    auto stride = view.strides == nullptr ? view.itemsize : view.strides[axis];
    if (view.strides == nullptr && axis + 1 < view.ndim) {
        stride = view.itemsize;
        for (int dim = axis + 1; dim < view.ndim; ++dim) {
            stride *= view.shape[dim];
        }
    }
    for (Py_ssize_t i = 0; i < view.shape[axis]; ++i) {
        collect_buffer_values(base + i * stride, view, axis + 1, values);
    }
}

bool supports_buffer_protocol(nb::handle value)
{
    return PyObject_CheckBuffer(value.ptr()) != 0;
}

bool has_mlx_array_protocol(nb::handle value)
{
    return nb::hasattr(value, "__mlx_array__")
        || nb::hasattr(value, "__mlx__array__");
}

mx::array call_mlx_array_protocol(nb::handle value)
{
    if (nb::hasattr(value, "__mlx_array__")) {
        return nb::cast<mx::array>(value.attr("__mlx_array__")());
    }
    return nb::cast<mx::array>(value.attr("__mlx__array__")());
}

using CpuNdArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

bool try_cpu_ndarray(nb::handle value, CpuNdArray& array)
{
    return nb::try_cast(value, array);
}

int checked_shape_dim(std::int64_t dim)
{
    if (dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "Shape dimension falls outside supported int range.");
    }
    return static_cast<int>(dim);
}

mx::Shape ndarray_shape(const CpuNdArray& array)
{
    mx::Shape shape;
    shape.reserve(array.ndim());
    for (int axis = 0; axis < array.ndim(); ++axis) {
        shape.push_back(checked_shape_dim(array.shape(axis)));
    }
    return shape;
}

mx::Dtype dtype_from_ndarray(const CpuNdArray& array)
{
    auto type = array.dtype();
    if (type == nb::dtype<bool>()) {
        return mx::bool_;
    }
    if (type == nb::dtype<std::uint8_t>()) {
        return mx::uint8;
    }
    if (type == nb::dtype<std::uint16_t>()) {
        return mx::uint16;
    }
    if (type == nb::dtype<std::uint32_t>()) {
        return mx::uint32;
    }
    if (type == nb::dtype<std::uint64_t>()) {
        return mx::uint64;
    }
    if (type == nb::dtype<std::int8_t>()) {
        return mx::int8;
    }
    if (type == nb::dtype<std::int16_t>()) {
        return mx::int16;
    }
    if (type == nb::dtype<std::int32_t>()) {
        return mx::int32;
    }
    if (type == nb::dtype<std::int64_t>()) {
        return mx::int64;
    }
    if (type == nb::dtype<mx::float16_t>()) {
        return mx::float16;
    }
    if (type == nb::dtype<float>()) {
        return mx::float32;
    }
    if (type == nb::dtype<double>()) {
        return mx::float64;
    }
    if (type == nb::dtype<std::complex<float>>()
            || type == nb::dtype<std::complex<double>>()) {
        return mx::complex64;
    }
    throw std::invalid_argument("Cannot infer ndarray dtype for MlxPreciseArray.");
}

template <typename T>
MlxPreciseArray ndarray_to_mlx_contiguous(const CpuNdArray& array,
                                          mx::Dtype dtype,
                                          const mx::StreamOrDevice& stream)
{
    auto shape = ndarray_shape(array);
    return MlxPreciseArray(
        place_on_stream(
            mx::array(static_cast<const T*>(array.data()), shape, dtype),
            stream),
        stream);
}

MlxPreciseArray ndarray_to_mlx_precise(const CpuNdArray& array,
                                       mx::Dtype requested_dtype,
                                       const mx::StreamOrDevice& stream)
{
    auto shape = ndarray_shape(array);
    auto dtype = requested_dtype;
    auto type = array.dtype();
    if (type == nb::dtype<bool>()) {
        return ndarray_to_mlx_contiguous<bool>(array, dtype, stream);
    }
    if (type == nb::dtype<std::uint8_t>()) {
        return ndarray_to_mlx_contiguous<std::uint8_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::uint16_t>()) {
        return ndarray_to_mlx_contiguous<std::uint16_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::uint32_t>()) {
        return ndarray_to_mlx_contiguous<std::uint32_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::uint64_t>()) {
        return ndarray_to_mlx_contiguous<std::uint64_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::int8_t>()) {
        return ndarray_to_mlx_contiguous<std::int8_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::int16_t>()) {
        return ndarray_to_mlx_contiguous<std::int16_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::int32_t>()) {
        return ndarray_to_mlx_contiguous<std::int32_t>(array, dtype, stream);
    }
    if (type == nb::dtype<std::int64_t>()) {
        return ndarray_to_mlx_contiguous<std::int64_t>(array, dtype, stream);
    }
    if (type == nb::dtype<mx::float16_t>()) {
        return ndarray_to_mlx_contiguous<mx::float16_t>(array, dtype, stream);
    }
    if (type == nb::dtype<float>()) {
        return ndarray_to_mlx_contiguous<float>(array, dtype, stream);
    }
    if (type == nb::dtype<double>()) {
        auto data = static_cast<const double*>(array.data());
        std::size_t size = 1;
        for (auto dim : shape) {
            size *= static_cast<std::size_t>(dim);
        }
        std::vector<double> values(data, data + size);
        if (dtype == mx::float64) {
            auto precise = MlxPreciseArray::from_float64_data(
                values, std::vector<mx::ShapeElem>(shape.begin(), shape.end()),
                stream);
            if (targets_gpu(stream)) {
                return MlxPreciseArray::transfer_float64_to_gpu(
                    std::move(precise), stream);
            }
            return precise;
        }
        return MlxPreciseArray(
            place_on_stream(
                mx::array(values.begin(), std::move(shape), dtype),
                stream),
            stream);
    }
    if (type == nb::dtype<std::complex<float>>()) {
        return ndarray_to_mlx_contiguous<mx::complex64_t>(
            array, mx::complex64, stream);
    }
    throw std::invalid_argument("Cannot convert ndarray to MlxPreciseArray.");
}

MlxPreciseArray array_from_buffer(nb::handle value,
                                  mx::Dtype dtype,
                                  const mx::StreamOrDevice& stream)
{
    Py_buffer view;
    if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        throw nb::python_error();
    }
    try {
        std::vector<double> values;
        std::vector<mx::ShapeElem> shape;
        shape.reserve(view.ndim);
        std::size_t size = 1;
        for (int i = 0; i < view.ndim; ++i) {
            shape.push_back(static_cast<mx::ShapeElem>(view.shape[i]));
            size *= static_cast<std::size_t>(view.shape[i]);
        }
        values.reserve(size);
        collect_buffer_values(static_cast<const char*>(view.buf), view, 0, values);
        PyBuffer_Release(&view);
        if (dtype == mx::float64 && targets_gpu(stream)) {
            return MlxPreciseArray::transfer_float64_to_gpu(
                MlxPreciseArray::from_float64_data(values, shape, stream),
                stream);
        }
        mx::Shape mx_shape(shape.begin(), shape.end());
        return MlxPreciseArray(
            place_on_stream(mx::array(values.begin(), std::move(mx_shape), dtype),
                            stream),
            stream);
    } catch (...) {
        PyBuffer_Release(&view);
        throw;
    }
}

mx::array packed_float64_array(const std::vector<double>& data,
                               const std::vector<mx::ShapeElem>& shape)
{
    return MlxPreciseArray::from_float64_data(data, shape);
}

mx::array transfer_float64_to_gpu(mx::array input,
                                  const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::transfer_float64_to_gpu(
        MlxPreciseArray(std::move(input), stream), stream);
}

std::vector<double> read_canonical_float64_values(mx::array array)
{
    if (array.dtype() != mx::float64) {
        array = mx::astype(array, mx::float64, mx::Device(mx::Device::cpu));
    }
    array.eval();
    auto ptr = array.data<double>();
    return std::vector<double>(ptr, ptr + array.size());
}

std::vector<double> read_precise_float64_values(MlxPreciseArray array)
{
    MlxPreciseArray::eval(array);
    std::vector<double> values;
    values.reserve(array.size());
    if (targets_gpu(array.stream_or_device())) {
        auto ptr = array.data<PreciseFloat64>();
        for (std::size_t i = 0; i < array.size(); ++i) {
            values.push_back(ptr[i].value());
        }
    } else {
        auto ptr = array.data<double>();
        values.assign(ptr, ptr + array.size());
    }
    return values;
}

mx::array pack_existing_array_for_gpu(mx::array array,
                                      const mx::StreamOrDevice& stream);

MlxPreciseArray place_float64_on_explicit_stream(
    MlxPreciseArray array,
    const mx::StreamOrDevice& stream)
{
    if (!has_explicit_stream(stream) || array.dtype() != mx::float64) {
        return array;
    }
    auto source_is_gpu = targets_gpu(array.stream_or_device());
    auto target_is_gpu = targets_gpu(stream);
    if (source_is_gpu == target_is_gpu) {
        return array;
    }
    if (target_is_gpu) {
        return MlxPreciseArray(
            pack_existing_array_for_gpu(std::move(array), stream),
            stream);
    }

    auto source_shape = array.shape();
    auto values = read_precise_float64_values(std::move(array));
    mx::Shape shape(source_shape.begin(), source_shape.end());
    return MlxPreciseArray(
        place_on_stream(
            mx::array(values.begin(), std::move(shape), mx::float64),
            stream),
        stream);
}

mx::array pack_existing_array_for_gpu(mx::array array,
                                      const mx::StreamOrDevice& stream)
{
    if (array.dtype() == mx::float64 && array.has_primitive() &&
            array.primitive().stream().device.type == mx::Device::gpu) {
        return array;
    }
    auto shape = std::vector<mx::ShapeElem>(array.shape().begin(),
                                            array.shape().end());
    return transfer_float64_to_gpu(
        packed_float64_array(read_canonical_float64_values(std::move(array)),
                             shape),
        stream);
}

bool is_python_sequence(nb::handle value)
{
    if (nb::isinstance<mx::array>(value) || has_mlx_array_protocol(value)) {
        return false;
    }
    return PySequence_Check(value.ptr())
        && !PyUnicode_Check(value.ptr())
        && !PyBytes_Check(value.ptr())
        && !PyByteArray_Check(value.ptr());
}

bool sequence_contains_array(nb::handle value)
{
    CpuNdArray ndarray;
    if (nb::isinstance<mx::array>(value) || has_mlx_array_protocol(value)
            || try_cpu_ndarray(value, ndarray)) {
        return true;
    }
    if (!is_python_sequence(value)) {
        return false;
    }
    nb::sequence sequence = nb::borrow<nb::sequence>(value);
    for (auto item : sequence) {
        if (sequence_contains_array(item)) {
            return true;
        }
    }
    return false;
}

void collect_float64_sequence(nb::handle value,
                              std::vector<double>& data,
                              std::vector<mx::ShapeElem>& shape,
                              std::size_t depth)
{
    if (!is_python_sequence(value)) {
        if (nb::isinstance<mx::array>(value) || has_mlx_array_protocol(value)) {
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

bool is_float_dtype(mx::Dtype dtype)
{
    return dtype == mx::float16
        || dtype == mx::bfloat16
        || dtype == mx::float32
        || dtype == mx::float64;
}

bool is_signed_int_dtype(mx::Dtype dtype)
{
    return dtype == mx::int8
        || dtype == mx::int16
        || dtype == mx::int32
        || dtype == mx::int64;
}

bool is_unsigned_int_dtype(mx::Dtype dtype)
{
    return dtype == mx::uint8
        || dtype == mx::uint16
        || dtype == mx::uint32
        || dtype == mx::uint64;
}

mx::Dtype promote_inferred_dtype(mx::Dtype left, mx::Dtype right)
{
    if (left == right) {
        return left;
    }
    if (left == mx::float64 || right == mx::float64) {
        return mx::float64;
    }
    if (is_float_dtype(left) || is_float_dtype(right)) {
        return mx::float32;
    }
    if (left == mx::uint64 || right == mx::uint64) {
        return mx::uint64;
    }
    if (left == mx::int64 || right == mx::int64) {
        return mx::int64;
    }
    if (is_signed_int_dtype(left) || is_signed_int_dtype(right)) {
        return mx::int32;
    }
    if (is_unsigned_int_dtype(left) || is_unsigned_int_dtype(right)) {
        return mx::uint32;
    }
    return mx::bool_;
}

mx::Dtype dtype_from_buffer_format(const Py_buffer& view)
{
    const char* format = view.format;
    if (format != nullptr) {
        while (*format == '@' || *format == '=' || *format == '<'
               || *format == '>' || *format == '!') {
            ++format;
        }
        switch (*format) {
        case '?':
            return mx::bool_;
        case 'b':
            return mx::int8;
        case 'B':
            return mx::uint8;
        case 'h':
            return mx::int16;
        case 'H':
            return mx::uint16;
        case 'i':
        case 'l':
            return mx::int32;
        case 'I':
        case 'L':
            return mx::uint32;
        case 'q':
            return mx::int64;
        case 'Q':
            return mx::uint64;
        case 'e':
            return mx::float16;
        case 'f':
            return mx::float32;
        case 'd':
            return mx::float64;
        default:
            break;
        }
    }
    if (view.itemsize == 1) {
        return mx::uint8;
    }
    if (view.itemsize == 2) {
        return mx::int16;
    }
    if (view.itemsize == 4) {
        return mx::float32;
    }
    if (view.itemsize == 8) {
        return mx::float64;
    }
    return mx::float32;
}

mx::Dtype inferred_python_dtype(nb::handle value)
{
    if (nb::isinstance<mx::array>(value)) {
        return nb::cast<mx::array>(value).dtype();
    }
    if (has_mlx_array_protocol(value)) {
        return call_mlx_array_protocol(value).dtype();
    }
    CpuNdArray ndarray;
    if (try_cpu_ndarray(value, ndarray)) {
        return dtype_from_ndarray(ndarray);
    }
    if (PyBool_Check(value.ptr())) {
        return mx::bool_;
    }
    if (PyFloat_Check(value.ptr())) {
        return mx::float32;
    }
    if (PyLong_Check(value.ptr())) {
        int overflow = 0;
        PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
        if (overflow != 0 || PyErr_Occurred()) {
            PyErr_Clear();
            return mx::int64;
        }
        return mx::int32;
    }
    if (supports_buffer_protocol(value)) {
        Py_buffer view;
        if (PyObject_GetBuffer(value.ptr(), &view,
                               PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
            throw nb::python_error();
        }
        auto dtype = dtype_from_buffer_format(view);
        PyBuffer_Release(&view);
        return dtype;
    }
    if (is_python_sequence(value)) {
        nb::sequence sequence = nb::borrow<nb::sequence>(value);
        if (nb::len(sequence) == 0) {
            return mx::float32;
        }
        bool first = true;
        mx::Dtype dtype = mx::bool_;
        for (auto item : sequence) {
            auto item_dtype = inferred_python_dtype(item);
            dtype = first ? item_dtype
                          : promote_inferred_dtype(dtype, item_dtype);
            first = false;
        }
        return dtype;
    }
    return mx::float32;
}

MlxPreciseArray stack_mixed_sequence_axis0(nb::handle value,
                                           nb::object dtype,
                                           const mx::StreamOrDevice& stream)
{
    nb::sequence sequence = nb::borrow<nb::sequence>(value);
    std::vector<MlxPreciseArray> rows;
    rows.reserve(nb::len(sequence));
    bool wants_float64 = false;
    for (auto item : sequence) {
        auto row = MlxPreciseArray::make(item, dtype, stream, 0);
        wants_float64 = wants_float64 || row.dtype() == mx::float64;
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        return MlxPreciseArray::make(value, dtype, stream, 0);
    }
    if (wants_float64) {
        for (auto& row : rows) {
            row = ensure_precise_float64(std::move(row), stream);
        }
        if (targets_gpu(stream)) {
            return MlxPreciseArray::stack_axis0_float64(std::move(rows), stream);
        }
    }

    std::vector<mx::array> canonical_rows;
    canonical_rows.reserve(rows.size());
    for (auto& row : rows) {
        canonical_rows.push_back(std::move(row));
    }
    return MlxPreciseArray(mx::stack(std::move(canonical_rows), 0, stream),
                           stream);
}

mx::array float64_array(nb::handle value, const mx::StreamOrDevice& stream)
{
    if (nb::isinstance<mx::array>(value) || has_mlx_array_protocol(value)) {
        return as_float64_array(value, stream);
    }

    std::vector<double> data;
    std::vector<mx::ShapeElem> shape;
    collect_float64_sequence(value, data, shape, 0);
    if (targets_gpu(stream)) {
        return transfer_float64_to_gpu(packed_float64_array(data, shape),
                                       stream);
    }
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

mx::Shape requested_shape(nb::handle shape_value)
{
    if (PyLong_Check(shape_value.ptr())) {
        return {static_cast<mx::ShapeElem>(PyLong_AsLongLong(shape_value.ptr()))};
    }
    nb::sequence sequence = nb::borrow<nb::sequence>(shape_value);
    mx::Shape shape;
    shape.reserve(nb::len(sequence));
    for (auto item : sequence) {
        shape.push_back(static_cast<mx::ShapeElem>(nb::cast<long long>(item)));
    }
    return shape;
}

MlxPreciseArray full_float64(nb::handle shape_value,
                             double value,
                             const mx::StreamOrDevice& stream)
{
    auto shape = requested_shape(shape_value);
    if (!targets_gpu(stream)) {
        return MlxPreciseArray(
            mx::full(shape, value, mx::float64, stream), stream);
    }
    auto output_shape = shape;
    auto primitive = std::make_shared<PreciseFloat64Full>(
        mx::to_stream(stream), std::move(shape), value);
    return MlxPreciseArray(
        mx::array(std::move(output_shape), mx::float64,
                  std::move(primitive), {}),
        stream);
}

std::vector<int> requested_axes(nb::handle axes_value, int ndim)
{
    std::vector<int> axes;
    if (axes_value.is_none()) {
        axes.reserve(ndim);
        for (int axis = ndim - 1; axis >= 0; --axis) {
            axes.push_back(axis);
        }
        return axes;
    }
    nb::sequence sequence = nb::borrow<nb::sequence>(axes_value);
    axes.reserve(nb::len(sequence));
    for (auto item : sequence) {
        int axis = nb::cast<int>(item);
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            throw std::invalid_argument("[transpose] axis out of bounds");
        }
        axes.push_back(axis);
    }
    if (static_cast<int>(axes.size()) != ndim) {
        throw std::invalid_argument("[transpose] axes don't match array");
    }
    return axes;
}

MlxPreciseArray astype_mlx_precise_array(MlxPreciseArray array,
                                         mx::Dtype target_dtype,
                                         const mx::StreamOrDevice& stream)
{
    if (array.dtype() == target_dtype) {
        return array;
    }
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        array = ensure_precise_float64(std::move(array), stream);
        if (target_dtype == mx::float32) {
            auto primitive = std::make_shared<PreciseFloat64GpuAstypeFloat32>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(array.shape(), mx::float32, std::move(primitive),
                          {std::move(array)}),
                stream);
        }
        if (target_dtype == mx::int32) {
            auto primitive = std::make_shared<PreciseFloat64GpuAstypeInt32>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(array.shape(), mx::int32, std::move(primitive),
                          {std::move(array)}),
                stream);
        }
    }
    if (target_dtype == mx::float64 && targets_gpu(stream)
            && array.dtype() == mx::float32) {
        auto primitive = std::make_shared<PreciseFloat64GpuAstypeFromFloat32>(
            mx::to_stream(stream));
        return MlxPreciseArray(
            mx::array(array.shape(), mx::float64, std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    return MlxPreciseArray(mx::astype(array, target_dtype, stream), stream);
}

mx::array mlx_precise_array(nb::handle value,
                            nb::object dtype,
                            const mx::StreamOrDevice& stream)
{
    auto target_dtype = dtype.is_none()
        ? inferred_python_dtype(value)
        : requested_dtype(dtype, mx::float32);
    if (target_dtype == mx::float64) {
        return float64_array(value, stream);
    }

    if (nb::isinstance<mx::array>(value)) {
        auto array = nb::cast<mx::array>(value);
        if (!dtype.is_none() && array.dtype() != target_dtype) {
            return astype_mlx_precise_array(
                MlxPreciseArray(std::move(array), stream), target_dtype, stream);
        }
        return place_on_stream(std::move(array), stream);
    }
    if (has_mlx_array_protocol(value)) {
        auto array = call_mlx_array_protocol(value);
        if (!dtype.is_none() && array.dtype() != target_dtype) {
            return astype_mlx_precise_array(
                MlxPreciseArray(std::move(array), stream), target_dtype, stream);
        }
        return place_on_stream(std::move(array), stream);
    }
    CpuNdArray ndarray;
    if (try_cpu_ndarray(value, ndarray)) {
        return ndarray_to_mlx_precise(ndarray, target_dtype, stream);
    }

    std::vector<double> data;
    std::vector<mx::ShapeElem> shape;
    collect_float64_sequence(value, data, shape, 0);
    mx::Shape mx_shape(shape.begin(), shape.end());
    auto array = mx::array(data.begin(), std::move(mx_shape), target_dtype);
    return place_on_stream(std::move(array), stream);
}

MlxPreciseArray MlxPreciseArray::make(nb::handle value,
                                      nb::object dtype,
                                      const mx::StreamOrDevice& stream,
                                      int ndmin)
{
    nb::object actual = nb::borrow<nb::object>(value);
    if (actual.is_none()) {
        actual = nb::cast(std::vector<double>{});
    }

    auto target_dtype = dtype.is_none()
        ? inferred_python_dtype(actual)
        : requested_dtype(dtype, mx::float32);
    if (nb::isinstance<MlxPreciseArray>(actual)) {
        auto array = nb::cast<MlxPreciseArray>(actual);
        if (dtype.is_none() || array.dtype() == target_dtype) {
            if (array.dtype() == mx::float64) {
                return place_float64_on_explicit_stream(std::move(array), stream);
            }
            if (has_explicit_stream(stream)) {
                return MlxPreciseArray(
                    place_on_stream(std::move(array), stream),
                    stream);
            }
            return array;
        }
        if (target_dtype == mx::float64 && targets_gpu(stream)) {
            return MlxPreciseArray(
                pack_existing_array_for_gpu(std::move(array), stream),
                stream);
        }
        return astype_mlx_precise_array(std::move(array), target_dtype, stream);
    }
    if (nb::isinstance<mx::array>(actual)) {
        auto array = nb::cast<mx::array>(actual);
        if (dtype.is_none()) {
            if (array.dtype() == mx::float64 && targets_gpu(stream)) {
                return MlxPreciseArray(
                    pack_existing_array_for_gpu(std::move(array), stream),
                    stream);
            }
            return MlxPreciseArray(
                place_on_stream(std::move(array), stream),
                stream);
        }
        if (array.dtype() != target_dtype) {
            return astype_mlx_precise_array(
                MlxPreciseArray(std::move(array), stream), target_dtype, stream);
        }
        if (array.dtype() == mx::float64 && targets_gpu(stream)) {
            return MlxPreciseArray(
                pack_existing_array_for_gpu(std::move(array), stream),
                stream);
        }
        return MlxPreciseArray(
            place_on_stream(std::move(array), stream),
            stream);
    }
    if (has_mlx_array_protocol(actual)) {
        auto array = call_mlx_array_protocol(actual);
        if (dtype.is_none()) {
            if (array.dtype() == mx::float64 && targets_gpu(stream)) {
                return MlxPreciseArray(
                    pack_existing_array_for_gpu(std::move(array), stream),
                    stream);
            }
            return MlxPreciseArray(
                place_on_stream(std::move(array), stream),
                stream);
        }
        if (array.dtype() != target_dtype) {
            return astype_mlx_precise_array(
                MlxPreciseArray(std::move(array), stream), target_dtype, stream);
        }
        if (array.dtype() == mx::float64 && targets_gpu(stream)) {
            return MlxPreciseArray(
                pack_existing_array_for_gpu(std::move(array), stream),
                stream);
        }
        return MlxPreciseArray(
            place_on_stream(std::move(array), stream),
            stream);
    }
    CpuNdArray ndarray;
    if (try_cpu_ndarray(actual, ndarray)) {
        return ndarray_to_mlx_precise(ndarray, target_dtype, stream);
    }
    if (supports_buffer_protocol(actual)) {
        return array_from_buffer(actual, target_dtype, stream);
    }
    if (is_python_sequence(actual) && sequence_contains_array(actual)) {
        return stack_mixed_sequence_axis0(actual, dtype, stream);
    }
    if (target_dtype == mx::float64 && !is_python_sequence(actual)) {
        double scalar = PyFloat_AsDouble(actual.ptr());
        if (PyErr_Occurred()) {
            throw nb::python_error();
        }
        auto scalar_array = MlxPreciseArray(
            float64_scalar(scalar, stream), stream);
        while (scalar_array.ndim() < ndmin) {
            mx::Shape shape(scalar_array.shape().begin(),
                            scalar_array.shape().end());
            shape.insert(shape.begin(), 1);
            if (scalar_array.dtype() == mx::float64 && targets_gpu(stream)) {
                scalar_array = ensure_precise_float64(
                    std::move(scalar_array), stream);
                auto output_shape = shape;
                auto primitive = std::make_shared<PreciseFloat64GpuReshape>(
                    mx::to_stream(stream), output_shape);
                scalar_array = MlxPreciseArray(
                    mx::array(std::move(output_shape), mx::float64,
                              std::move(primitive), {std::move(scalar_array)}),
                    stream);
            } else {
                scalar_array = MlxPreciseArray(
                    mx::reshape(scalar_array, std::move(shape), stream),
                    stream);
            }
        }
        return scalar_array;
    }
    if (target_dtype == mx::float64
            && !nb::isinstance<mx::array>(actual)
            && !has_mlx_array_protocol(actual)) {
        std::vector<double> data;
        std::vector<mx::ShapeElem> shape;
        collect_float64_sequence(actual, data, shape, 0);
        while (static_cast<int>(shape.size()) < ndmin) {
            shape.insert(shape.begin(), 1);
        }
        if (targets_gpu(stream)) {
            return transfer_float64_to_gpu(
                from_float64_data(data, shape, stream), stream);
        }
        mx::Shape mx_shape(shape.begin(), shape.end());
        return MlxPreciseArray(
            place_on_stream(
                mx::array(data.begin(), std::move(mx_shape), mx::float64),
                stream),
            stream);
    }

    auto array = mlx_precise_array(actual, dtype, stream);
    while (array.ndim() < ndmin) {
        mx::Shape shape(array.shape().begin(), array.shape().end());
        shape.insert(shape.begin(), 1);
        array = mx::reshape(array, std::move(shape), stream);
    }
    return MlxPreciseArray(std::move(array), stream);
}

void eval_precise_array(const mx::array& value)
{
    MlxPreciseArray::eval(value);
}

mx::array as_float64_array(nb::handle value, const mx::StreamOrDevice& stream)
{
    if (nb::isinstance<mx::array>(value)) {
        auto array = nb::cast<mx::array>(value);
        if (targets_gpu(stream)) {
            return pack_existing_array_for_gpu(std::move(array), stream);
        }
        if (array.dtype() != mx::float64) {
            array = mx::astype(array, mx::float64, stream);
        }
        return place_on_stream(array, stream);
    }
    if (has_mlx_array_protocol(value)) {
        auto array = call_mlx_array_protocol(value);
        if (targets_gpu(stream)) {
            return pack_existing_array_for_gpu(std::move(array), stream);
        }
        if (array.dtype() != mx::float64) {
            array = mx::astype(array, mx::float64, stream);
        }
        return place_on_stream(array, stream);
    }
    CpuNdArray ndarray;
    if (try_cpu_ndarray(value, ndarray)) {
        return ndarray_to_mlx_precise(ndarray, mx::float64, stream);
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

nb::object dtype_for_operand(nb::handle value, mx::Dtype dtype)
{
    CpuNdArray ndarray;
    if (nb::isinstance<mx::array>(value) || has_mlx_array_protocol(value)
            || try_cpu_ndarray(value, ndarray)) {
        return nb::none();
    }
    if (dtype == mx::float64) {
        return nb::cast(dtype);
    }
    return nb::none();
}

MlxPreciseArray ensure_precise_float64(MlxPreciseArray array,
                                       const mx::StreamOrDevice& stream)
{
    if (targets_gpu(stream)) {
        if (array.dtype() == mx::float64
                && targets_gpu(array.stream_or_device())) {
            return array;
        }
        return MlxPreciseArray(
            pack_existing_array_for_gpu(std::move(array), stream),
            stream);
    }
    if (array.dtype() != mx::float64) {
        return MlxPreciseArray(mx::astype(array, mx::float64, stream), stream);
    }
    return array;
}

MlxPreciseArray binary_precise_arrays(MlxPreciseArray left,
                                      MlxPreciseArray right,
                                      PreciseFloat64BinaryOp op,
                                      const mx::StreamOrDevice& stream);

MlxPreciseArray binary_precise_array(const MlxPreciseArray& self,
                                     nb::handle other,
                                     PreciseFloat64BinaryOp op,
                                     bool reverse)
{
    auto stream = self.stream_or_device();
    auto self_array = MlxPreciseArray(
        static_cast<const mx::array&>(self), stream);
    auto other_array = MlxPreciseArray::make(
        other, dtype_for_operand(other, self.dtype()), stream, 0);
    auto left = reverse ? other_array : self_array;
    auto right = reverse ? self_array : other_array;
    return binary_precise_arrays(std::move(left), std::move(right), op, stream);
}

MlxPreciseArray binary_precise_arrays(MlxPreciseArray left,
                                      MlxPreciseArray right,
                                      PreciseFloat64BinaryOp op,
                                      const mx::StreamOrDevice& stream)
{
    bool precise = left.dtype() == mx::float64 || right.dtype() == mx::float64;

    if (precise) {
        left = ensure_precise_float64(std::move(left), stream);
        right = ensure_precise_float64(std::move(right), stream);
        if (targets_gpu(stream)) {
            return MlxPreciseArray::binary_float64(
                std::move(left), std::move(right), op, stream);
        }
    }

    switch (op) {
    case PreciseFloat64BinaryOp::Add:
        return MlxPreciseArray(mx::add(left, right, stream), stream);
    case PreciseFloat64BinaryOp::Subtract:
        return MlxPreciseArray(mx::subtract(left, right, stream), stream);
    case PreciseFloat64BinaryOp::Multiply:
        return MlxPreciseArray(mx::multiply(left, right, stream), stream);
    case PreciseFloat64BinaryOp::Divide:
        return MlxPreciseArray(mx::divide(left, right, stream), stream);
    case PreciseFloat64BinaryOp::Power:
        return MlxPreciseArray(mx::power(left, right, stream), stream);
    }
    throw std::invalid_argument("unknown precise float64 binary operation");
}

MlxPreciseArray MlxPreciseArray::add(nb::handle other) const
{
    return binary_precise_array(*this, other, PreciseFloat64BinaryOp::Add, false);
}

MlxPreciseArray MlxPreciseArray::subtract(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Subtract, false);
}

MlxPreciseArray MlxPreciseArray::reverse_subtract(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Subtract, true);
}

MlxPreciseArray MlxPreciseArray::multiply(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Multiply, false);
}

MlxPreciseArray MlxPreciseArray::divide(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Divide, false);
}

MlxPreciseArray MlxPreciseArray::reverse_divide(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Divide, true);
}

MlxPreciseArray MlxPreciseArray::power(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Power, false);
}

MlxPreciseArray MlxPreciseArray::reverse_power(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Power, true);
}

MlxPreciseArray matmul_precise_array(const MlxPreciseArray& self,
                                     nb::handle other,
                                     bool reverse)
{
    auto stream = self.stream_or_device();
    auto self_array = MlxPreciseArray(
        static_cast<const mx::array&>(self), stream);
    auto other_array = MlxPreciseArray::make(
        other, dtype_for_operand(other, self.dtype()), stream, 0);
    auto left = reverse ? other_array : self_array;
    auto right = reverse ? self_array : other_array;
    bool precise = left.dtype() == mx::float64 || right.dtype() == mx::float64;

    if (precise) {
        left = ensure_precise_float64(std::move(left), stream);
        right = ensure_precise_float64(std::move(right), stream);
        if (targets_gpu(stream) && left.ndim() == 2 && right.ndim() == 2) {
            if (left.shape(1) != right.shape(0)) {
                throw std::invalid_argument("[matmul] input dimensions do not match");
            }
            auto primitive = std::make_shared<PreciseFloat64GpuMatmul>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(mx::Shape{left.shape(0), right.shape(1)},
                          mx::float64,
                          std::move(primitive),
                          {std::move(left), std::move(right)}),
                stream);
        }
    }

    return MlxPreciseArray(mx::matmul(left, right, stream), stream);
}

MlxPreciseArray MlxPreciseArray::matmul(nb::handle other) const
{
    return matmul_precise_array(*this, other, false);
}

MlxPreciseArray MlxPreciseArray::reverse_matmul(nb::handle other) const
{
    return matmul_precise_array(*this, other, true);
}

MlxPreciseArray MlxPreciseArray::negative() const
{
    auto stream = stream_or_device();
    if (dtype() == mx::float64 && targets_gpu(stream)) {
        auto zero = MlxPreciseArray(float64_scalar(0.0, stream), stream);
        auto self_array = ensure_precise_float64(
            MlxPreciseArray(static_cast<const mx::array&>(*this), stream),
            stream);
        return MlxPreciseArray::binary_float64(
            std::move(zero),
            std::move(self_array),
            PreciseFloat64BinaryOp::Subtract,
            stream);
    }
    return MlxPreciseArray(mx::negative(*this, stream), stream);
}

MlxPreciseArray add_precise(nb::handle left,
                            nb::handle right,
                            const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).add(right);
}

MlxPreciseArray subtract_precise(nb::handle left,
                                 nb::handle right,
                                 const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).subtract(right);
}

MlxPreciseArray multiply_precise(nb::handle left,
                                 nb::handle right,
                                 const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).multiply(right);
}

MlxPreciseArray divide_precise(nb::handle left,
                               nb::handle right,
                               const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).divide(right);
}

MlxPreciseArray power_precise(nb::handle left,
                              nb::handle right,
                              const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).power(right);
}

MlxPreciseArray matmul_precise(nb::handle left,
                               nb::handle right,
                               const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream, 0).matmul(right);
}

MlxPreciseArray compare_precise(nb::handle left_value,
                                nb::handle right_value,
                                PreciseFloat64CompareOp op,
                                const mx::StreamOrDevice& stream)
{
    auto left = MlxPreciseArray::make(left_value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = MlxPreciseArray::make(
        right_value, dtype_for_operand(right_value, left.dtype()), actual_stream, 0);
    bool precise = left.dtype() == mx::float64 || right.dtype() == mx::float64;
    if (precise) {
        left = ensure_precise_float64(std::move(left), actual_stream);
        right = ensure_precise_float64(std::move(right), actual_stream);
        if (targets_gpu(actual_stream)) {
            auto primitive = std::make_shared<PreciseFloat64GpuCompare>(
                op, mx::to_stream(actual_stream));
            return MlxPreciseArray(
                mx::array(broadcast_binary_shape(left, right), mx::bool_,
                          std::move(primitive),
                          {std::move(left), std::move(right)}),
                actual_stream);
        }
    }
    switch (op) {
    case PreciseFloat64CompareOp::Equal:
        return MlxPreciseArray(mx::equal(left, right, actual_stream),
                               actual_stream);
    case PreciseFloat64CompareOp::NotEqual:
        return MlxPreciseArray(mx::not_equal(left, right, actual_stream),
                               actual_stream);
    case PreciseFloat64CompareOp::Less:
        return MlxPreciseArray(mx::less(left, right, actual_stream),
                               actual_stream);
    case PreciseFloat64CompareOp::LessEqual:
        return MlxPreciseArray(mx::less_equal(left, right, actual_stream),
                               actual_stream);
    }
    throw std::invalid_argument("unsupported comparison operation");
}

MlxPreciseArray equal_precise(nb::handle left,
                              nb::handle right,
                              const mx::StreamOrDevice& stream)
{
    return compare_precise(left, right, PreciseFloat64CompareOp::Equal, stream);
}

MlxPreciseArray not_equal_precise(nb::handle left,
                                  nb::handle right,
                                  const mx::StreamOrDevice& stream)
{
    return compare_precise(left, right, PreciseFloat64CompareOp::NotEqual, stream);
}

MlxPreciseArray less_precise(nb::handle left,
                             nb::handle right,
                             const mx::StreamOrDevice& stream)
{
    return compare_precise(left, right, PreciseFloat64CompareOp::Less, stream);
}

MlxPreciseArray less_equal_precise(nb::handle left,
                                   nb::handle right,
                                   const mx::StreamOrDevice& stream)
{
    return compare_precise(left, right, PreciseFloat64CompareOp::LessEqual, stream);
}

MlxPreciseArray arctan2_precise(nb::handle left_value,
                                nb::handle right_value,
                                const mx::StreamOrDevice& stream)
{
    auto left = MlxPreciseArray::make(left_value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = MlxPreciseArray::make(
        right_value, dtype_for_operand(right_value, left.dtype()), actual_stream, 0);
    bool precise = left.dtype() == mx::float64 || right.dtype() == mx::float64;
    if (precise) {
        left = ensure_precise_float64(std::move(left), actual_stream);
        right = ensure_precise_float64(std::move(right), actual_stream);
        if (targets_gpu(actual_stream)) {
            auto primitive = std::make_shared<PreciseFloat64GpuArcTan2>(
                mx::to_stream(actual_stream));
            return MlxPreciseArray(
                mx::array(broadcast_binary_shape(left, right), mx::float64,
                          std::move(primitive),
                          {std::move(left), std::move(right)}),
                actual_stream);
        }
    }
    return MlxPreciseArray(mx::arctan2(left, right, actual_stream),
                           actual_stream);
}

MlxPreciseArray where_precise(nb::handle condition_value,
                              nb::handle x_value,
                              nb::handle y_value,
                              const mx::StreamOrDevice& stream)
{
    auto condition = MlxPreciseArray::make(condition_value, nb::none(), stream, 0);
    auto x = MlxPreciseArray::make(x_value, nb::none(), stream, 0);
    auto y = MlxPreciseArray::make(
        y_value, dtype_for_operand(y_value, x.dtype()), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : x.stream_or_device();
    if (condition.dtype() != mx::bool_) {
        condition = astype_mlx_precise_array(
            std::move(condition), mx::bool_, actual_stream);
    }

    bool precise = x.dtype() == mx::float64 || y.dtype() == mx::float64;
    if (precise) {
        x = ensure_precise_float64(std::move(x), actual_stream);
        y = ensure_precise_float64(std::move(y), actual_stream);
        if (targets_gpu(actual_stream)) {
            auto primitive = std::make_shared<PreciseFloat64GpuWhere>(
                mx::to_stream(actual_stream));
            auto shape = broadcast_ternary_shape(condition, x, y);
            return MlxPreciseArray(
                mx::array(std::move(shape), mx::float64,
                          std::move(primitive),
                          {std::move(condition), std::move(x), std::move(y)}),
                actual_stream);
        }
    }
    return MlxPreciseArray(
        mx::where(condition, x, y, actual_stream), actual_stream);
}

MlxPreciseArray abs_precise(nb::handle value,
                            const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        auto primitive = std::make_shared<PreciseFloat64GpuAbs>(
            mx::to_stream(actual_stream));
        return MlxPreciseArray(
            mx::array(array.shape(), mx::float64, std::move(primitive),
                      {std::move(array)}),
            actual_stream);
    }
    return MlxPreciseArray(mx::abs(array, actual_stream), actual_stream);
}

MlxPreciseArray round_precise(nb::handle value,
                              int decimals,
                              const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        auto primitive = std::make_shared<PreciseFloat64GpuRound>(
            mx::to_stream(actual_stream), decimals);
        return MlxPreciseArray(
            mx::array(array.shape(), mx::float64, std::move(primitive),
                      {std::move(array)}),
            actual_stream);
    }
    return MlxPreciseArray(mx::round(array, decimals, actual_stream),
                           actual_stream);
}

MlxPreciseArray concatenate_precise_arrays(
    std::vector<MlxPreciseArray> precise_arrays,
    int axis,
    const mx::StreamOrDevice& stream)
{
    if (precise_arrays.empty()) {
        throw std::invalid_argument("[concatenate] No arrays provided for concatenation");
    }

    bool precise = false;
    for (const auto& array : precise_arrays) {
        precise = precise || array.dtype() == mx::float64;
    }

    if (precise) {
        for (auto& array : precise_arrays) {
            array = ensure_precise_float64(std::move(array), stream);
        }
    }

    std::vector<mx::array> inputs;
    inputs.reserve(precise_arrays.size());
    for (auto& array : precise_arrays) {
        inputs.push_back(std::move(array));
    }

    if (!precise || !targets_gpu(stream)) {
        return MlxPreciseArray(mx::concatenate(std::move(inputs), axis, stream),
                               stream);
    }

    if (inputs.size() == 1) {
        return MlxPreciseArray(std::move(inputs[0]), stream);
    }

    auto ax = mx::normalize_axis_index(
        axis, inputs[0].ndim(), "[concatenate] ");
    auto shape = inputs[0].shape();
    shape[ax] = 0;
    for (auto& array : inputs) {
        if (array.ndim() != shape.size()) {
            throw std::invalid_argument(
                "[concatenate] all input arrays must have the same ndim");
        }
        for (int dim = 0; dim < array.ndim(); ++dim) {
            if (dim != ax && array.shape(dim) != shape[dim]) {
                throw std::invalid_argument(
                    "[concatenate] input shapes differ outside the concatenation axis");
            }
        }
        shape[ax] += array.shape(ax);
    }

    auto primitive = std::make_shared<PreciseFloat64GpuConcatenate>(
        mx::to_stream(stream), ax);
    return MlxPreciseArray(
        mx::array(std::move(shape), mx::float64, std::move(primitive),
                  std::move(inputs)),
        stream);
}

MlxPreciseArray concatenate_precise(nb::handle arrays,
                                    int axis,
                                    const mx::StreamOrDevice& stream)
{
    nb::sequence sequence = nb::borrow<nb::sequence>(arrays);
    std::vector<MlxPreciseArray> precise_arrays;
    precise_arrays.reserve(nb::len(sequence));
    for (auto item : sequence) {
        precise_arrays.push_back(
            MlxPreciseArray::make(item, nb::none(), stream, 0));
    }
    return concatenate_precise_arrays(std::move(precise_arrays), axis, stream);
}

MlxPreciseArray stack_precise(nb::handle arrays,
                              int axis,
                              const mx::StreamOrDevice& stream)
{
    nb::sequence sequence = nb::borrow<nb::sequence>(arrays);
    std::vector<MlxPreciseArray> rows;
    rows.reserve(nb::len(sequence));
    bool wants_float64 = false;
    mx::StreamOrDevice actual_stream = stream;
    bool saw_first = false;
    for (auto item : sequence) {
        auto row = MlxPreciseArray::make(item, nb::none(), actual_stream, 0);
        if (!saw_first && !has_explicit_stream(stream)) {
            actual_stream = row.stream_or_device();
            saw_first = true;
        }
        wants_float64 = wants_float64 || row.dtype() == mx::float64;
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        return MlxPreciseArray(mx::stack({}, axis, actual_stream), actual_stream);
    }

    if (axis < 0) {
        axis += rows[0].ndim() + 1;
    }
    if (axis < 0 || axis > rows[0].ndim()) {
        throw std::invalid_argument("[stack] axis out of bounds");
    }

    if (wants_float64) {
        for (auto& row : rows) {
            row = ensure_precise_float64(std::move(row), actual_stream);
        }
        if (targets_gpu(actual_stream)) {
            auto stacked = MlxPreciseArray::stack_axis0_float64(
                std::move(rows), actual_stream);
            if (axis == 0) {
                return stacked;
            }
            std::vector<int> axes;
            axes.reserve(stacked.ndim());
            for (int dim = 1; dim < stacked.ndim(); ++dim) {
                axes.push_back(dim);
            }
            axes.insert(axes.begin() + axis, 0);
            mx::Shape output_shape;
            output_shape.reserve(axes.size());
            for (auto dim : axes) {
                output_shape.push_back(stacked.shape(dim));
            }
            auto primitive = std::make_shared<PreciseFloat64GpuTranspose>(
                mx::to_stream(actual_stream), axes);
            return MlxPreciseArray(
                mx::array(std::move(output_shape), mx::float64,
                          std::move(primitive), {std::move(stacked)}),
                actual_stream);
        }
    }

    std::vector<mx::array> canonical_rows;
    canonical_rows.reserve(rows.size());
    for (auto& row : rows) {
        canonical_rows.push_back(std::move(row));
    }
    return MlxPreciseArray(
        mx::stack(std::move(canonical_rows), axis, actual_stream),
        actual_stream);
}

MlxPreciseArray slice_precise_array(MlxPreciseArray array,
                                    mx::Shape start,
                                    mx::Shape stop,
                                    mx::Shape strides,
                                    const mx::StreamOrDevice& stream)
{
    if (start.size() != array.ndim() || stop.size() != array.ndim() ||
            strides.size() != array.ndim()) {
        throw std::invalid_argument("[slice] Invalid number of indices or strides");
    }
    if (array.dtype() != mx::float64 || !targets_gpu(stream)) {
        return MlxPreciseArray(
            mx::slice(array, std::move(start), std::move(stop),
                      std::move(strides), stream),
            stream);
    }

    array = ensure_precise_float64(std::move(array), stream);
    auto normalized_stop = stop;
    auto [has_neg_strides, out_shape] =
        normalize_precise_slice(array.shape(), start, normalized_stop, strides);
    if (!has_neg_strides && out_shape == array.shape()) {
        return array;
    }

    auto primitive = std::make_shared<PreciseFloat64GpuSlice>(
        mx::to_stream(stream), start, normalized_stop, strides);
    return MlxPreciseArray(
        mx::array(std::move(out_shape), mx::float64, std::move(primitive),
                  {std::move(array)}),
        stream);
}

MlxPreciseArray slice_precise(nb::handle value,
                              nb::handle start_value,
                              nb::handle stop_value,
                              nb::handle strides_value,
                              const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    return slice_precise_array(
        std::move(array), requested_shape(start_value),
        requested_shape(stop_value), requested_shape(strides_value),
        actual_stream);
}

MlxPreciseArray affine_transform_precise(nb::handle vertices_value,
                                         nb::handle matrix_value,
                                         const mx::StreamOrDevice& stream)
{
    auto vertices = MlxPreciseArray::make(vertices_value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : vertices.stream_or_device();
    auto matrix = MlxPreciseArray::make(matrix_value, nb::none(), actual_stream, 0);

    if (vertices.dtype() != mx::float32 && vertices.dtype() != mx::float64) {
        throw std::invalid_argument("vertices must be float32 or float64");
    }
    if (matrix.dtype() != vertices.dtype()) {
        throw std::invalid_argument("affine matrix dtype must match vertices dtype");
    }
    if (matrix.ndim() != 2 || matrix.shape(0) != 3 || matrix.shape(1) != 3) {
        throw std::invalid_argument("Invalid affine transformation matrix");
    }

    bool reshape_result = false;
    if (vertices.ndim() == 1) {
        if (vertices.shape(0) != 2) {
            throw std::runtime_error("Invalid vertices array.");
        }
        if (vertices.dtype() == mx::float64 && targets_gpu(actual_stream)) {
            vertices = ensure_precise_float64(std::move(vertices), actual_stream);
            auto primitive = std::make_shared<PreciseFloat64GpuReshape>(
                mx::to_stream(actual_stream), mx::Shape{1, 2});
            vertices = MlxPreciseArray(
                mx::array(mx::Shape{1, 2}, mx::float64,
                          std::move(primitive), {std::move(vertices)}),
                actual_stream);
        } else {
            vertices = MlxPreciseArray(
                mx::reshape(vertices, {1, 2}, actual_stream),
                actual_stream);
        }
        reshape_result = true;
    } else if (vertices.ndim() == 2) {
        if (vertices.shape(1) != 2) {
            throw std::invalid_argument("vertices must have shape (N, 2)");
        }
    } else {
        throw std::invalid_argument("vertices must be 1D or 2D");
    }

    auto n = static_cast<mx::ShapeElem>(vertices.shape(0));
    auto x = slice_precise_array(vertices, {0, 0}, {n, 1}, {1, 1}, actual_stream);
    auto y = slice_precise_array(vertices, {0, 1}, {n, 2}, {1, 1}, actual_stream);

    auto sx = slice_precise_array(matrix, {0, 0}, {1, 1}, {1, 1}, actual_stream);
    auto shx = slice_precise_array(matrix, {0, 1}, {1, 2}, {1, 1}, actual_stream);
    auto tx = slice_precise_array(matrix, {0, 2}, {1, 3}, {1, 1}, actual_stream);
    auto shy = slice_precise_array(matrix, {1, 0}, {2, 1}, {1, 1}, actual_stream);
    auto sy = slice_precise_array(matrix, {1, 1}, {2, 2}, {1, 1}, actual_stream);
    auto ty = slice_precise_array(matrix, {1, 2}, {2, 3}, {1, 1}, actual_stream);

    auto out_x = binary_precise_arrays(
        binary_precise_arrays(sx, x, PreciseFloat64BinaryOp::Multiply,
                              actual_stream),
        binary_precise_arrays(shx, y, PreciseFloat64BinaryOp::Multiply,
                              actual_stream),
        PreciseFloat64BinaryOp::Add,
        actual_stream);
    out_x = binary_precise_arrays(std::move(out_x), std::move(tx),
                                  PreciseFloat64BinaryOp::Add, actual_stream);
    auto out_y = binary_precise_arrays(
        binary_precise_arrays(shy, x, PreciseFloat64BinaryOp::Multiply,
                              actual_stream),
        binary_precise_arrays(sy, y, PreciseFloat64BinaryOp::Multiply,
                              actual_stream),
        PreciseFloat64BinaryOp::Add,
        actual_stream);
    out_y = binary_precise_arrays(std::move(out_y), std::move(ty),
                                  PreciseFloat64BinaryOp::Add, actual_stream);

    std::vector<MlxPreciseArray> columns;
    columns.reserve(2);
    columns.push_back(std::move(out_x));
    columns.push_back(std::move(out_y));
    auto result = concatenate_precise_arrays(std::move(columns), 1, actual_stream);
    if (reshape_result) {
        if (result.dtype() == mx::float64 && targets_gpu(actual_stream)) {
            result = ensure_precise_float64(std::move(result), actual_stream);
            auto primitive = std::make_shared<PreciseFloat64GpuReshape>(
                mx::to_stream(actual_stream), mx::Shape{2});
            return MlxPreciseArray(
                mx::array(mx::Shape{2}, mx::float64,
                          std::move(primitive), {std::move(result)}),
                actual_stream);
        }
        return MlxPreciseArray(mx::reshape(result, {2}, actual_stream),
                               actual_stream);
    }
    return result;
}

MlxPreciseArray isfinite_precise(nb::handle value,
                                 const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        array = ensure_precise_float64(std::move(array), stream);
        auto primitive = std::make_shared<PreciseFloat64GpuIsFinite>(
            mx::to_stream(stream));
        return MlxPreciseArray(
            mx::array(array.shape(), mx::bool_, std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    return MlxPreciseArray(mx::isfinite(array, stream), stream);
}

MlxPreciseArray astype_precise(nb::handle value,
                               nb::object dtype,
                               const mx::StreamOrDevice& stream)
{
    auto target_dtype = nb::cast<mx::Dtype>(dtype);
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    return astype_mlx_precise_array(std::move(array), target_dtype, stream);
}

MlxPreciseArray reshape_precise(nb::handle value,
                                nb::handle shape_value,
                                const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    auto shape = requested_shape(shape_value);
    auto output_shape = mx::Reshape::output_shape(array, shape);
    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        auto primitive = std::make_shared<PreciseFloat64GpuReshape>(
            mx::to_stream(actual_stream), output_shape);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64,
                      std::move(primitive), {std::move(array)}),
            actual_stream);
    }
    return MlxPreciseArray(
        mx::reshape(array, std::move(output_shape), actual_stream),
        actual_stream);
}

MlxPreciseArray cumsum_precise(nb::handle value,
                               nb::object axis_value,
                               const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    bool flatten = axis_value.is_none();
    int axis = 0;
    if (!flatten) {
        axis = nb::cast<int>(axis_value);
        if (axis < 0) {
            axis += array.ndim();
        }
    }
    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        if (flatten && array.ndim() != 1) {
            auto flat_size = static_cast<mx::ShapeElem>(array.size());
            auto primitive = std::make_shared<PreciseFloat64GpuReshape>(
                mx::to_stream(actual_stream), mx::Shape{flat_size});
            array = MlxPreciseArray(
                mx::array(mx::Shape{flat_size}, mx::float64,
                          std::move(primitive), {std::move(array)}),
                actual_stream);
        } else if (!flatten && axis != 0) {
            throw std::invalid_argument(
                "precise float64 GPU cumsum currently supports flattened or axis=0 1D inputs");
        }
        if (array.ndim() != 1) {
            throw std::invalid_argument(
                "precise float64 GPU cumsum currently supports 1D inputs");
        }
        auto primitive = std::make_shared<PreciseFloat64GpuCumsum1D>(
            mx::to_stream(actual_stream));
        return MlxPreciseArray(
            mx::array(array.shape(), mx::float64, std::move(primitive),
                      {std::move(array)}),
            actual_stream);
    }
    if (flatten) {
        return MlxPreciseArray(
            mx::cumsum(mx::reshape(array, {-1}, actual_stream),
                       0, false, true, actual_stream),
            actual_stream);
    }
    return MlxPreciseArray(
        mx::cumsum(array, axis, false, true, actual_stream),
        actual_stream);
}

MlxPreciseArray transpose_precise(nb::handle value,
                                  nb::handle axes_value,
                                  const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    auto axes = requested_axes(axes_value, array.ndim());
    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        mx::Shape output_shape;
        output_shape.reserve(axes.size());
        for (auto axis : axes) {
            output_shape.push_back(array.shape(axis));
        }
        auto primitive = std::make_shared<PreciseFloat64GpuTranspose>(
            mx::to_stream(actual_stream), axes);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64,
                      std::move(primitive), {std::move(array)}),
            actual_stream);
    }
    return MlxPreciseArray(
        mx::transpose(array, std::move(axes), actual_stream),
        actual_stream);
}

nb::bytes array_bytes(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    if (!array.flags().row_contiguous) {
        array = MlxPreciseArray(mx::contiguous(array, false, stream), stream);
    }
    MlxPreciseArray::eval(array);
    return nb::bytes(static_cast<const char*>(array.data<void>()),
                     array.nbytes());
}

nb::bytes float64_bytes(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::cast(mx::float64), stream, 0);
    if (!array.flags().row_contiguous) {
        array = MlxPreciseArray(mx::contiguous(array, false, stream), stream);
    }
    auto values = read_precise_float64_values(std::move(array));
    return nb::bytes(reinterpret_cast<const char*>(values.data()),
                     values.size() * sizeof(double));
}

nb::object item_precise(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    if (array.size() != 1) {
        throw nb::value_error("can only convert a size-1 array to scalar");
    }
    if (array.dtype() != mx::float64) {
        throw nb::type_error("item_precise only handles float64 arrays");
    }
    auto values = read_precise_float64_values(std::move(array));
    return nb::float_(values[0]);
}

MlxPreciseArray reduce_minmax_precise(nb::handle value,
                                      nb::object axis,
                                      bool is_max,
                                      const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream, 0);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        array = ensure_precise_float64(std::move(array), stream);
        int normalized_axis = -1;
        if (!axis.is_none()) {
            normalized_axis = mx::normalize_axis_index(
                nb::cast<int>(axis), array.ndim(), is_max ? "[max] " : "[min] ");
        }
        auto primitive = std::make_shared<PreciseFloat64GpuReduceMinMax>(
            mx::to_stream(stream), normalized_axis, is_max);
        auto shape = array.shape();
        if (normalized_axis < 0) {
            shape.clear();
        } else {
            shape.erase(shape.begin() + normalized_axis);
        }
        return MlxPreciseArray(
            mx::array(std::move(shape), mx::float64, std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    if (axis.is_none()) {
        return MlxPreciseArray(
            is_max ? mx::max(array, false, stream)
                   : mx::min(array, false, stream),
            stream);
    }
    auto ax = nb::cast<int>(axis);
    return MlxPreciseArray(
        is_max ? mx::max(array, ax, false, stream)
               : mx::min(array, ax, false, stream),
        stream);
}

mx::array allclose_precise(nb::handle left,
                           nb::handle right,
                           double rtol,
                           double atol,
                           bool equal_nan,
                           const mx::StreamOrDevice& stream)
{
    auto left_array = MlxPreciseArray::make(left, nb::none(), stream, 0);
    auto right_array = MlxPreciseArray::make(right, nb::none(), stream, 0);
    if (left_array.dtype() != mx::float64 && right_array.dtype() != mx::float64) {
        return mx::allclose(left_array, right_array, rtol, atol, equal_nan, stream);
    }
    auto left_stream = left_array.stream_or_device();
    auto right_stream = right_array.stream_or_device();
    left_array = ensure_precise_float64(std::move(left_array), left_stream);
    right_array = ensure_precise_float64(std::move(right_array), right_stream);
    if (left_array.shape() != right_array.shape()) {
        return mx::array(false);
    }
    auto left_values = read_precise_float64_values(std::move(left_array));
    auto right_values = read_precise_float64_values(std::move(right_array));
    for (std::size_t i = 0; i < left_values.size(); ++i) {
        auto a = left_values[i];
        auto b = right_values[i];
        if (std::isnan(a) || std::isnan(b)) {
            if (equal_nan && std::isnan(a) && std::isnan(b)) {
                continue;
            }
            return mx::array(false);
        }
        if (std::isinf(a) || std::isinf(b)) {
            if (a == b) {
                continue;
            }
            return mx::array(false);
        }
        if (std::abs(a - b) > atol + rtol * std::abs(b)) {
            return mx::array(false);
        }
    }
    return mx::array(true);
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
    if (targets_gpu(stream)) {
        mx::Shape shape{length};
        auto output_shape = shape;
        auto primitive = std::make_shared<PreciseFloat64Arange>(
            mx::to_stream(stream), std::move(shape), start, step);
        return mx::array(std::move(output_shape), mx::float64,
                         std::move(primitive), {});
    }
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
    nb::class_<MlxPreciseArray, mx::array>(m, "MlxPreciseArray")
        .def(
            "__init__",
            [](MlxPreciseArray* self,
               nb::object value,
               nb::object dtype,
               const mx::StreamOrDevice& stream,
               nb::object,
               nb::object,
               nb::object,
               nb::object,
               int ndmin) {
                new (self) MlxPreciseArray(
                    MlxPreciseArray::make(value, dtype, stream, ndmin));
            },
            "value"_a = nb::none(),
            "dtype"_a = nb::none(),
            "stream"_a = nb::none(),
            "copy"_a = nb::none(),
            "order"_a = nb::none(),
            "subok"_a = nb::none(),
            "like"_a = nb::none(),
            "ndmin"_a = 0)
        .def_prop_ro("T",
             [](const MlxPreciseArray& self) {
                 auto self_obj = nb::cast(self);
                 return transpose_precise(
                     self_obj, nb::none(), self.stream_or_device());
             })
        .def("astype",
             [](const MlxPreciseArray& self,
                nb::object dtype,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return astype_precise(self_obj, dtype, stream);
             },
             "dtype"_a,
             "stream"_a = nb::none())
        .def("reshape",
             [](const MlxPreciseArray& self,
                nb::handle shape,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return reshape_precise(self_obj, shape, stream);
             },
             "shape"_a,
             "stream"_a = nb::none())
        .def("transpose",
             [](const MlxPreciseArray& self,
                nb::handle axes,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return transpose_precise(self_obj, axes, stream);
             },
             "axes"_a = nb::none(),
             "stream"_a = nb::none())
        .def("cumsum",
             [](const MlxPreciseArray& self,
                nb::object axis,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return cumsum_precise(self_obj, axis, stream);
             },
             "axis"_a = nb::none(),
             "stream"_a = nb::none())
        .def("__add__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.add(other);
             },
             "other"_a)
        .def("add",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.add(other);
             },
             "other"_a)
        .def("__radd__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.add(other);
             },
             "other"_a)
        .def("__iadd__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.add(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__sub__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.subtract(other);
             },
             "other"_a)
        .def("subtract",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.subtract(other);
             },
             "other"_a)
        .def("__rsub__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.reverse_subtract(other);
             },
             "other"_a)
        .def("__isub__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.subtract(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__mul__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.multiply(other);
             },
             "other"_a)
        .def("multiply",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.multiply(other);
             },
             "other"_a)
        .def("__rmul__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.multiply(other);
             },
             "other"_a)
        .def("__imul__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.multiply(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__truediv__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.divide(other);
             },
             "other"_a)
        .def("divide",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.divide(other);
             },
             "other"_a)
        .def("__rtruediv__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.reverse_divide(other);
             },
             "other"_a)
        .def("__itruediv__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.divide(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__pow__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.power(other);
             },
             "other"_a)
        .def("__rpow__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.reverse_power(other);
             },
             "other"_a)
        .def("__ipow__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.power(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__matmul__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.matmul(other);
             },
             "other"_a)
        .def("matmul",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.matmul(other);
             },
             "other"_a)
        .def("__rmatmul__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.reverse_matmul(other);
             },
             "other"_a)
        .def("__imatmul__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.matmul(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("__neg__",
             [](const MlxPreciseArray& self) {
                 return self.negative();
             })
        .def("__abs__",
             [](const MlxPreciseArray& self) {
                 auto self_obj = nb::cast(self);
                 return abs_precise(self_obj, self.stream_or_device());
             })
        .def("__eq__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return equal_precise(self_obj, other, self.stream_or_device());
             },
             "other"_a)
        .def("__ne__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return not_equal_precise(self_obj, other, self.stream_or_device());
             },
             "other"_a)
        .def("__lt__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return less_precise(self_obj, other, self.stream_or_device());
             },
             "other"_a)
        .def("__le__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return less_equal_precise(self_obj, other, self.stream_or_device());
             },
             "other"_a)
        .def("__gt__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return less_precise(other, self_obj, self.stream_or_device());
             },
             "other"_a)
        .def("__ge__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 auto self_obj = nb::cast(self);
                 return less_equal_precise(other, self_obj, self.stream_or_device());
             },
             "other"_a)
        .def("__pos__",
             [](const MlxPreciseArray& self) {
                 return MlxPreciseArray(
                     static_cast<const mx::array&>(self),
                     self.stream_or_device());
             })
        .def("__copy__",
             [](const MlxPreciseArray& self) {
                 return MlxPreciseArray(
                     static_cast<const mx::array&>(self),
                     self.stream_or_device());
             });

    m.def("float64_scalar", &float64_scalar,
          "value"_a,
          "stream"_a = nb::none());
    m.def("mlx_precise_array", &mlx_precise_array,
          "value"_a,
          "dtype"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("full_float64", &full_float64,
          "shape"_a,
          "value"_a,
          "stream"_a = nb::none());
    m.def("eval_precise_array", &eval_precise_array,
          "value"_a);
    m.def("add_precise", &add_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("subtract_precise", &subtract_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("multiply_precise", &multiply_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("divide_precise", &divide_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("power_precise", &power_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("matmul_precise", &matmul_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("equal_precise", &equal_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("not_equal_precise", &not_equal_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("less_precise", &less_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("less_equal_precise", &less_equal_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("arctan2_precise", &arctan2_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("where_precise", &where_precise,
          "condition"_a,
          "x"_a,
          "y"_a,
          "stream"_a = nb::none());
    m.def("abs_precise", &abs_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("round_precise", &round_precise,
          "value"_a,
          "decimals"_a = 0,
          "stream"_a = nb::none());
    m.def("concatenate_precise", &concatenate_precise,
          "arrays"_a,
          "axis"_a = 0,
          "stream"_a = nb::none());
    m.def("stack_precise", &stack_precise,
          "arrays"_a,
          "axis"_a = 0,
          "stream"_a = nb::none());
    m.def("slice_precise", &slice_precise,
          "value"_a,
          "start"_a,
          "stop"_a,
          "strides"_a,
          "stream"_a = nb::none());
    m.def("affine_transform_precise", &affine_transform_precise,
          "vertices"_a,
          "matrix"_a,
          "stream"_a = nb::none());
    m.def("isfinite_precise", &isfinite_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("astype_precise", &astype_precise,
          "value"_a,
          "dtype"_a,
          "stream"_a = nb::none());
    m.def("reshape_precise", &reshape_precise,
          "value"_a,
          "shape"_a,
          "stream"_a = nb::none());
    m.def("cumsum_precise", &cumsum_precise,
          "value"_a,
          "axis"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("transpose_precise", &transpose_precise,
          "value"_a,
          "axes"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("array_bytes", &array_bytes,
          "value"_a,
          "stream"_a = nb::none());
    m.def("float64_bytes", &float64_bytes,
          "value"_a,
          "stream"_a = nb::none());
    m.def("item_precise", &item_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("reduce_minmax_precise", &reduce_minmax_precise,
          "value"_a,
          "axis"_a = nb::none(),
          "is_max"_a = false,
          "stream"_a = nb::none());
    m.def("allclose_precise", &allclose_precise,
          "left"_a,
          "right"_a,
          "rtol"_a = 1e-5,
          "atol"_a = 1e-8,
          "equal_nan"_a = false,
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
