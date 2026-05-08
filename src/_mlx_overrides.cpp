#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"
#include "mlx/version.h"

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
    Remainder = 5,
};

enum class PreciseFloat64CompareOp : int {
    Equal = 0,
    NotEqual = 1,
    Less = 2,
    LessEqual = 3,
};

enum class PreciseFloat64UnaryMathOp : int {
    Log = 0,
    Log2 = 1,
    Log10 = 2,
    Floor = 3,
    Ceil = 4,
    Sqrt = 5,
    Degrees = 6,
    Radians = 7,
    Sin = 8,
    Cos = 9,
    ArcSin = 10,
    ArcCos = 11,
    ArcTan = 12,
};

mx::Shape broadcast_binary_shape(const mx::array& left, const mx::array& right)
{
    return mx::broadcast_shapes(left.shape(), right.shape());
}

mx::Shape matmul_batch_shape(const mx::array& left, const mx::array& right)
{
    if (left.ndim() < 1 || right.ndim() < 1) {
        throw std::invalid_argument("[matmul] inputs must have at least one dimension");
    }
    mx::Shape left_batch;
    mx::Shape right_batch;
    if (left.ndim() > 2) {
        left_batch.insert(
            left_batch.end(), left.shape().begin(), left.shape().end() - 2);
    }
    if (right.ndim() > 2) {
        right_batch.insert(
            right_batch.end(), right.shape().begin(), right.shape().end() - 2);
    }
    return mx::broadcast_shapes(left_batch, right_batch);
}

mx::Shape matmul_output_shape(const mx::array& left, const mx::array& right)
{
    auto left_ndim = left.ndim();
    auto right_ndim = right.ndim();
    auto left_k = left.shape(left_ndim - 1);
    auto right_k = right_ndim == 1
        ? right.shape(0)
        : right.shape(right_ndim - 2);
    if (left_k != right_k) {
        throw std::invalid_argument("[matmul] input dimensions do not match");
    }
    auto shape = matmul_batch_shape(left, right);
    if (left_ndim != 1) {
        shape.push_back(left.shape(left_ndim - 2));
    }
    if (right_ndim != 1) {
        shape.push_back(right.shape(right_ndim - 1));
    }
    return shape;
}

mx::Shape flatten_output_shape(const mx::Shape& input_shape,
                               int start_axis,
                               int end_axis)
{
    auto ndim = static_cast<int>(input_shape.size());
    if (ndim == 0) {
        return mx::Shape{};
    }
    if (start_axis < 0) {
        start_axis += ndim;
    }
    if (end_axis < 0) {
        end_axis += ndim;
    }
    if (start_axis < 0 || end_axis < 0
            || start_axis >= ndim || end_axis >= ndim) {
        throw std::invalid_argument("[flatten] axis out of bounds");
    }
    if (start_axis > end_axis) {
        throw std::invalid_argument(
            "[flatten] start_axis must be less than or equal to end_axis");
    }
    mx::Shape shape;
    shape.reserve(input_shape.size() - (end_axis - start_axis));
    for (int axis = 0; axis < start_axis; ++axis) {
        shape.push_back(input_shape[axis]);
    }
    mx::ShapeElem flattened = 1;
    for (int axis = start_axis; axis <= end_axis; ++axis) {
        flattened *= input_shape[axis];
    }
    shape.push_back(flattened);
    for (int axis = end_axis + 1; axis < ndim; ++axis) {
        shape.push_back(input_shape[axis]);
    }
    return shape;
}

mx::metal::CommandEncoder& precise_command_encoder(mx::Stream stream)
{
#if MLX_VERSION_NUMERIC >= 31000
    return mx::metal::get_command_encoder(stream);
#else
    return mx::metal::device(stream.device).get_command_encoder(stream.index);
#endif
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

mx::Strides contiguous_row_strides(const mx::Shape& shape)
{
    mx::Strides strides(shape.size(), 1);
    std::int64_t stride = 1;
    for (std::size_t step = 0; step < shape.size(); ++step) {
        auto axis = shape.size() - 1 - step;
        strides[axis] = stride;
        stride *= static_cast<std::int64_t>(shape[axis]);
    }
    return strides;
}

std::size_t row_major_offset(std::size_t index,
                             const mx::Shape& shape,
                             const mx::Strides& strides)
{
    std::int64_t offset = 0;
    for (std::size_t step = 0; step < shape.size(); ++step) {
        auto axis = shape.size() - 1 - step;
        auto dim = shape[axis];
        auto coord = dim == 0 ? 0 : index % dim;
        if (dim != 0) {
            index /= dim;
        }
        offset += static_cast<std::int64_t>(coord) * strides[axis];
    }
    return static_cast<std::size_t>(offset);
}

std::size_t slice_offset(std::size_t index,
                         const mx::Shape& out_shape,
                         const mx::Shape& start,
                         const mx::Shape& strides,
                         const mx::Strides& input_strides)
{
    std::int64_t offset = 0;
    for (std::size_t step = 0; step < out_shape.size(); ++step) {
        auto axis = out_shape.size() - 1 - step;
        auto dim = out_shape[axis];
        auto coord = dim == 0 ? 0 : index % dim;
        if (dim != 0) {
            index /= dim;
        }
        auto input_coord = static_cast<std::int64_t>(start[axis])
            + static_cast<std::int64_t>(coord) * strides[axis];
        offset += input_coord * input_strides[axis];
    }
    return static_cast<std::size_t>(offset);
}

struct SliceUpdateOffsets {
    bool in_slice;
    std::size_t source_offset;
    std::size_t update_offset;
};

SliceUpdateOffsets slice_update_offsets(std::size_t index,
                                        const mx::Shape& shape,
                                        const mx::Strides& source_strides,
                                        const mx::Shape& start,
                                        const mx::Shape& stop,
                                        const mx::Shape& slice_strides,
                                        const mx::Strides& update_strides)
{
    bool in_slice = true;
    std::int64_t source_offset = 0;
    std::int64_t update_offset = 0;
    for (std::size_t step = 0; step < shape.size(); ++step) {
        auto axis = shape.size() - 1 - step;
        auto dim = shape[axis];
        auto coord = dim == 0 ? 0 : index % dim;
        if (dim != 0) {
            index /= dim;
        }

        auto signed_coord = static_cast<std::int64_t>(coord);
        source_offset += signed_coord * source_strides[axis];
        auto slice_step = static_cast<std::int64_t>(slice_strides[axis]);
        auto slice_start = static_cast<std::int64_t>(start[axis]);
        auto slice_stop = static_cast<std::int64_t>(stop[axis]);
        bool axis_inside = slice_step > 0
            ? (signed_coord >= slice_start && signed_coord < slice_stop)
            : (signed_coord <= slice_start && signed_coord > slice_stop);
        if (!axis_inside) {
            in_slice = false;
            continue;
        }
        auto delta = signed_coord - slice_start;
        if (slice_step == 0 || delta % slice_step != 0) {
            in_slice = false;
            continue;
        }
        update_offset += (delta / slice_step) * update_strides[axis];
    }
    return {
        in_slice,
        static_cast<std::size_t>(source_offset),
        static_cast<std::size_t>(update_offset)};
}

mx::Shape precise_reshape_output_shape(const mx::array& array,
                                       mx::Shape shape)
{
    std::int64_t inferred_axis = -1;
    std::size_t known_size = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        auto dim = shape[axis];
        if (dim == -1) {
            if (inferred_axis != -1) {
                throw std::invalid_argument(
                    "[reshape] only one dimension can be inferred");
            }
            inferred_axis = static_cast<std::int64_t>(axis);
            continue;
        }
        if (dim < 0) {
            throw std::invalid_argument("[reshape] invalid negative dimension");
        }
        known_size *= static_cast<std::size_t>(dim);
    }

    auto input_size = array.size();
    if (inferred_axis != -1) {
        if (known_size == 0 || input_size % known_size != 0) {
            throw std::invalid_argument(
                "[reshape] cannot infer dimension for requested shape");
        }
        shape[static_cast<std::size_t>(inferred_axis)] =
            static_cast<mx::ShapeElem>(input_size / known_size);
        return shape;
    }

    if (known_size != input_size) {
        throw std::invalid_argument(
            "[reshape] requested shape does not match array size");
    }
    return shape;
}

mx::Shape take_output_shape(const mx::array& input,
                            const mx::array& indices,
                            int axis,
                            bool flatten_input)
{
    if (flatten_input) {
        return indices.shape();
    }
    if (axis < 0) {
        axis += input.ndim();
    }
    if (axis < 0 || axis >= input.ndim()) {
        throw std::invalid_argument("[take] axis out of bounds");
    }

    mx::Shape shape;
    shape.reserve(input.ndim() + indices.ndim() - 1);
    for (int i = 0; i < axis; ++i) {
        shape.push_back(input.shape(i));
    }
    for (auto dim : indices.shape()) {
        shape.push_back(dim);
    }
    for (int i = axis + 1; i < input.ndim(); ++i) {
        shape.push_back(input.shape(i));
    }
    return shape;
}

std::int64_t read_int32_index(const mx::array& indices, std::size_t offset)
{
    return static_cast<std::int64_t>(indices.data<std::int32_t>()[offset]);
}

std::size_t take_source_offset(std::size_t out_index,
                               const mx::array& input,
                               const mx::array& indices,
                               const mx::Shape& out_shape,
                               int axis,
                               bool flatten_input)
{
    std::vector<mx::ShapeElem> coords(out_shape.size());
    auto remaining = out_index;
    for (std::size_t step = 0; step < out_shape.size(); ++step) {
        auto out_axis = out_shape.size() - 1 - step;
        auto dim = out_shape[out_axis];
        coords[out_axis] = dim == 0 ? 0 : remaining % dim;
        if (dim != 0) {
            remaining /= dim;
        }
    }

    std::int64_t index_offset = 0;
    if (flatten_input) {
        for (int idx_axis = 0; idx_axis < indices.ndim(); ++idx_axis) {
            index_offset += static_cast<std::int64_t>(coords[idx_axis])
                * indices.strides(idx_axis);
        }
        auto index = read_int32_index(indices, static_cast<std::size_t>(index_offset));
        if (index < 0) {
            index += static_cast<std::int64_t>(input.size());
        }
        return row_major_offset(
            static_cast<std::size_t>(index), input.shape(), input.strides());
    }

    std::int64_t input_offset = 0;
    int out_axis = 0;
    for (int input_axis = 0; input_axis < axis; ++input_axis) {
        input_offset += static_cast<std::int64_t>(coords[out_axis++])
            * input.strides(input_axis);
    }
    for (int idx_axis = 0; idx_axis < indices.ndim(); ++idx_axis) {
        index_offset += static_cast<std::int64_t>(coords[out_axis++])
            * indices.strides(idx_axis);
    }
    auto index = read_int32_index(indices, static_cast<std::size_t>(index_offset));
    if (index < 0) {
        index += input.shape(axis);
    }
    input_offset += index * input.strides(axis);
    for (int input_axis = axis + 1; input_axis < input.ndim(); ++input_axis) {
        input_offset += static_cast<std::int64_t>(coords[out_axis++])
            * input.strides(input_axis);
    }
    return static_cast<std::size_t>(input_offset);
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
        auto& encoder = precise_command_encoder(stream());
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
        auto& encoder = precise_command_encoder(stream());
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
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuSlice : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuSlice(mx::Stream stream,
                           mx::Shape start_indices,
                           mx::Shape output_shape,
                           mx::Shape strides)
        : mx::UnaryPrimitive(stream),
          start_indices_(std::move(start_indices)),
          output_shape_(std::move(output_shape)),
          strides_(std::move(strides)) {}

    const char* name() const override {
        return "PreciseFloat64GpuSlice";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>&) override {
        return {output_shape_};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = source[slice_offset(
                i, out.shape(), start_indices_, strides_, inputs[0].strides())];
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
            "mlx_matplotlib_precise_float64_slice",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline long slice_offset(
                        ulong index,
                        constant ulong* out_shape,
                        constant long* start,
                        constant long* strides,
                        constant long* input_strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = out_shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            long input_coord = start[axis]
                                + static_cast<long>(coord) * strides[axis];
                            offset += input_coord * input_strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_slice(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* out_shape [[buffer(4)]],
                        constant long* start [[buffer(5)]],
                        constant long* strides [[buffer(6)]],
                        constant long* input_strides [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        out[index] = in[slice_offset(
                            index, out_shape, start, strides, input_strides,
                            ndim)];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_slice", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto out_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> start(
            start_indices_.begin(), start_indices_.end());
        std::vector<std::int64_t> strides(strides_.begin(), strides_.end());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(out_shape, 4);
        encoder.set_vector_bytes(start, 5);
        encoder.set_vector_bytes(strides, 6);
        encoder.set_vector_bytes(input_strides, 7);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    mx::Shape start_indices_;
    mx::Shape output_shape_;
    mx::Shape strides_;
};

class PreciseFloat64GpuSliceUpdate : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuSliceUpdate(mx::Stream stream,
                                 mx::Shape start_indices,
                                 mx::Shape stop_indices,
                                 mx::Shape slice_strides)
        : mx::UnaryPrimitive(stream),
          start_indices_(std::move(start_indices)),
          stop_indices_(std::move(stop_indices)),
          slice_strides_(std::move(slice_strides)) {}

    const char* name() const override {
        return "PreciseFloat64GpuSliceUpdate";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto update = inputs[1].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto offsets = slice_update_offsets(
                i, out.shape(), inputs[0].strides(), start_indices_,
                stop_indices_, slice_strides_, inputs[1].strides());
            result[i] = offsets.in_slice
                ? update[offsets.update_offset]
                : source[offsets.source_offset];
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
            "mlx_matplotlib_precise_float64_slice_update",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_slice_update(
                        device const float2* source [[buffer(0)]],
                        device const float2* update [[buffer(1)]],
                        device float2* out [[buffer(2)]],
                        constant ulong& size [[buffer(3)]],
                        constant uint& ndim [[buffer(4)]],
                        constant ulong* shape [[buffer(5)]],
                        constant long* source_strides [[buffer(6)]],
                        constant long* update_strides [[buffer(7)]],
                        constant long* start [[buffer(8)]],
                        constant long* stop [[buffer(9)]],
                        constant long* slice_strides [[buffer(10)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }

                        auto remaining = index;
                        long source_offset = 0;
                        long update_offset = 0;
                        bool in_slice = true;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            long coord = dim == 0
                                ? 0
                                : static_cast<long>(remaining % dim);
                            if (dim != 0) {
                                remaining /= dim;
                            }

                            source_offset += coord * source_strides[axis];
                            long slice_step = slice_strides[axis];
                            long slice_start = start[axis];
                            long slice_stop = stop[axis];
                            bool axis_inside = slice_step > 0
                                ? (coord >= slice_start && coord < slice_stop)
                                : (coord <= slice_start && coord > slice_stop);
                            if (!axis_inside) {
                                in_slice = false;
                                continue;
                            }
                            long delta = coord - slice_start;
                            if (slice_step == 0 || delta % slice_step != 0) {
                                in_slice = false;
                                continue;
                            }
                            update_offset += (delta / slice_step)
                                * update_strides[axis];
                        }

                        out[index] = in_slice
                            ? update[update_offset]
                            : source[source_offset];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_slice_update", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(out.ndim());
        auto shape = shape_bytes(out.shape());
        std::vector<std::int64_t> source_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        std::vector<std::int64_t> update_strides(
            inputs[1].strides().begin(), inputs[1].strides().end());
        std::vector<std::int64_t> start(
            start_indices_.begin(), start_indices_.end());
        std::vector<std::int64_t> stop(
            stop_indices_.begin(), stop_indices_.end());
        std::vector<std::int64_t> slice_strides(
            slice_strides_.begin(), slice_strides_.end());
        encoder.set_bytes(size, 3);
        encoder.set_bytes(ndim, 4);
        encoder.set_vector_bytes(shape, 5);
        encoder.set_vector_bytes(source_strides, 6);
        encoder.set_vector_bytes(update_strides, 7);
        encoder.set_vector_bytes(start, 8);
        encoder.set_vector_bytes(stop, 9);
        encoder.set_vector_bytes(slice_strides, 10);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    mx::Shape start_indices_;
    mx::Shape stop_indices_;
    mx::Shape slice_strides_;
};

class PreciseFloat64GpuReshape : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuReshape(mx::Stream stream, mx::Shape shape)
        : mx::UnaryPrimitive(stream), shape_(std::move(shape)) {}

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
        eval_common_gpu(inputs[0], out, stream());
    }

private:
    static void eval_common(const mx::array& input, mx::array& out) {
        if (input.flags().row_contiguous) {
            auto out_strides = contiguous_row_strides(out.shape());
            auto [data_size, row_contiguous, col_contiguous] =
                mx::check_contiguity(out.shape(), out_strides);
            mx::array::Flags flags{
                data_size == input.data_size(), row_contiguous, col_contiguous};
            out.copy_shared_buffer(input, out_strides, flags, input.data_size());
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = input.data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = source[row_major_offset(
                i, input.shape(), input.strides())];
        }
    }

    static void eval_common_gpu(const mx::array& input,
                                mx::array& out,
                                mx::Stream stream) {
        if (input.flags().row_contiguous) {
            auto out_strides = contiguous_row_strides(out.shape());
            auto [data_size, row_contiguous, col_contiguous] =
                mx::check_contiguity(out.shape(), out_strides);
            mx::array::Flags flags{
                data_size == input.data_size(), row_contiguous, col_contiguous};
            out.copy_shared_buffer(input, out_strides, flags, input.data_size());
            return;
        }
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream.device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_reshape_copy",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_reshape_copy(
                        device const float2* in [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* input_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        out[index] = in[row_major_offset(
                            index, input_shape, input_strides, ndim)];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_reshape_copy", library);
        auto& encoder = precise_command_encoder(stream);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(input, 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(input.ndim());
        auto input_shape = shape_bytes(input.shape());
        std::vector<std::int64_t> input_strides(
            input.strides().begin(), input.strides().end());
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(input_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

    mx::Shape shape_;
};

class PreciseFloat64GpuTranspose : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuTranspose(mx::Stream stream, std::vector<int> axes)
        : mx::UnaryPrimitive(stream), axes_(std::move(axes)) {}

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
            case PreciseFloat64BinaryOp::Remainder:
                value = lhs - std::floor(lhs / rhs) * rhs;
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

                    inline float2 dd_remainder(float2 a, float2 b) {
                        float q = floor((a.x + a.y) / (b.x + b.y));
                        return dd_sub(a, dd_mul(float2(q, 0.0f), b));
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
	                        } else if (op == 4) {
	                            value = dd_pow(lhs, rhs);
	                        } else {
	                            value = dd_remainder(lhs, rhs);
	                        }
                        out[index] = value;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_binary", library);
        auto& encoder = precise_command_encoder(stream());
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

                    inline int dd_compare(float2 lhs, float2 rhs) {
                        if (lhs.x < rhs.x) {
                            return -1;
                        }
                        if (lhs.x > rhs.x) {
                            return 1;
                        }
                        if (lhs.y < rhs.y) {
                            return -1;
                        }
                        if (lhs.y > rhs.y) {
                            return 1;
                        }
                        return 0;
                    }

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
                        int cmp = dd_compare(lhs, rhs);
                        if (op == 0) {
                            out[index] = cmp == 0;
                        } else if (op == 1) {
                            out[index] = cmp != 0;
                        } else if (op == 2) {
                            out[index] = cmp < 0;
                        } else {
                            out[index] = cmp <= 0;
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_compare", library);
        auto& encoder = precise_command_encoder(stream());
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
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuTake : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuTake(mx::Stream stream, int axis, bool flatten_input)
        : mx::UnaryPrimitive(stream),
          axis_(axis),
          flatten_input_(flatten_input) {}

    const char* name() const override {
        return "PreciseFloat64GpuTake";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {take_output_shape(
            inputs[0], inputs[1], axis_, flatten_input_)};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = source[take_source_offset(
                i, inputs[0], inputs[1], out.shape(), axis_, flatten_input_)];
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
            "mlx_matplotlib_precise_float64_take",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline long row_major_offset(
                        long index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            long coord = dim == 0 ? 0 : index % static_cast<long>(dim);
                            if (dim != 0) {
                                index /= static_cast<long>(dim);
                            }
                            offset += coord * strides[axis];
                        }
                        return offset;
                    }

                    inline long take_source_offset(
                        ulong out_index,
                        device const int* indices,
                        constant ulong* input_shape,
                        constant long* input_strides,
                        constant long* index_strides,
                        constant ulong* out_shape,
                        constant ulong& input_size,
                        constant int& axis,
                        constant uint& input_ndim,
                        constant uint& index_ndim,
                        constant uint& out_ndim,
                        constant bool& flatten_input) {
                        long index_offset = 0;
                        long input_offset = 0;
                        auto remaining = out_index;
                        if (flatten_input) {
                            for (uint step = 0; step < out_ndim; ++step) {
                                uint out_axis = out_ndim - 1 - step;
                                ulong dim = out_shape[out_axis];
                                ulong coord = dim == 0 ? 0 : remaining % dim;
                                if (dim != 0) {
                                    remaining /= dim;
                                }
                                index_offset += static_cast<long>(coord)
                                    * index_strides[out_axis];
                            }
                            long index = static_cast<long>(indices[index_offset]);
                            if (index < 0) {
                                index += static_cast<long>(input_size);
                            }
                            return row_major_offset(
                                index, input_shape, input_strides, input_ndim);
                        }

                        for (uint step = 0; step < out_ndim; ++step) {
                            uint out_axis = out_ndim - 1 - step;
                            ulong dim = out_shape[out_axis];
                            ulong coord = dim == 0 ? 0 : remaining % dim;
                            if (dim != 0) {
                                remaining /= dim;
                            }
                            if (static_cast<int>(out_axis) < axis) {
                                input_offset += static_cast<long>(coord)
                                    * input_strides[out_axis];
                            } else if (out_axis < static_cast<uint>(axis) + index_ndim) {
                                auto index_axis = out_axis - static_cast<uint>(axis);
                                index_offset += static_cast<long>(coord)
                                    * index_strides[index_axis];
                            } else {
                                auto input_axis = out_axis - index_ndim + 1;
                                input_offset += static_cast<long>(coord)
                                    * input_strides[input_axis];
                            }
                        }
                        long index = static_cast<long>(indices[index_offset]);
                        if (index < 0) {
                            index += static_cast<long>(input_shape[axis]);
                        }
                        return input_offset + index * input_strides[axis];
                    }

                    kernel void mlx_matplotlib_precise_float64_take(
                        device const float2* in [[buffer(0)]],
                        device const int* indices [[buffer(1)]],
                        device float2* out [[buffer(2)]],
                        constant ulong& size [[buffer(3)]],
                        constant ulong* input_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant long* index_strides [[buffer(6)]],
                        constant ulong* out_shape [[buffer(7)]],
                        constant ulong& input_size [[buffer(8)]],
                        constant int& axis [[buffer(9)]],
                        constant uint& input_ndim [[buffer(10)]],
                        constant uint& index_ndim [[buffer(11)]],
                        constant uint& out_ndim [[buffer(12)]],
                        constant bool& flatten_input [[buffer(13)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        out[index] = in[take_source_offset(
                            index, indices, input_shape, input_strides,
                            index_strides, out_shape, input_size, axis,
                            input_ndim, index_ndim, out_ndim, flatten_input)];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_take", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto input_shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        std::vector<std::int64_t> index_strides(
            inputs[1].strides().begin(), inputs[1].strides().end());
        auto out_shape = shape_bytes(out.shape());
        auto input_size = static_cast<std::uint64_t>(inputs[0].size());
        auto axis = axis_;
        auto input_ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto index_ndim = static_cast<std::uint32_t>(inputs[1].ndim());
        auto out_ndim = static_cast<std::uint32_t>(out.ndim());
        auto flatten_input = flatten_input_;
        encoder.set_bytes(size, 3);
        encoder.set_vector_bytes(input_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_vector_bytes(index_strides, 6);
        encoder.set_vector_bytes(out_shape, 7);
        encoder.set_bytes(input_size, 8);
        encoder.set_bytes(axis, 9);
        encoder.set_bytes(input_ndim, 10);
        encoder.set_bytes(index_ndim, 11);
        encoder.set_bytes(out_ndim, 12);
        encoder.set_bytes(flatten_input, 13);

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
    bool flatten_input_;
};

class PreciseFloat64GpuCumsum : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuCumsum(mx::Stream stream,
                            int axis,
                            bool flatten_input,
                            bool reverse,
                            bool inclusive)
        : mx::UnaryPrimitive(stream),
          axis_(axis),
          flatten_input_(flatten_input),
          reverse_(reverse),
          inclusive_(inclusive) {}

    const char* name() const override {
        return "PreciseFloat64GpuCumsum";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        if (flatten_input_) {
            return {mx::Shape{static_cast<mx::ShapeElem>(inputs[0].size())}};
        }
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuCumsum is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_cumsum",
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

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_cumsum(
                        device const float2* input [[buffer(0)]],
                        device float2* output [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant int& axis [[buffer(6)]],
                        constant bool& flatten_input [[buffer(7)]],
                        constant bool& reverse [[buffer(8)]],
                        constant bool& inclusive [[buffer(9)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }

                        ulong axis_coord = index;
                        ulong axis_size = size;
                        long input_base = 0;
                        if (!flatten_input) {
                            ulong remainder = index;
                            for (uint step = 0; step < ndim; ++step) {
                                uint dim_axis = ndim - 1 - step;
                                ulong dim = shape[dim_axis];
                                ulong coord = dim == 0 ? 0 : remainder % dim;
                                if (dim != 0) {
                                    remainder /= dim;
                                }
                                if (static_cast<int>(dim_axis) == axis) {
                                    axis_coord = coord;
                                    axis_size = dim;
                                } else {
                                    input_base += static_cast<long>(coord)
                                        * input_strides[dim_axis];
                                }
                            }
                        }

                        ulong begin = 0;
                        ulong end = 0;
                        if (reverse) {
                            begin = inclusive ? axis_coord : axis_coord + 1;
                            end = axis_size;
                        } else {
                            begin = 0;
                            end = inclusive ? axis_coord + 1 : axis_coord;
                        }

                        float2 running = float2(0.0f, 0.0f);
                        for (ulong i = begin; i < end; ++i) {
                            long input_offset = 0;
                            if (flatten_input) {
                                input_offset = row_major_offset(
                                    i, shape, input_strides, ndim);
                            } else {
                                input_offset = input_base
                                    + static_cast<long>(i) * input_strides[axis];
                            }
                            running = precise_add(running, input[input_offset]);
                        }
                        output[index] = running;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_cumsum", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        auto axis = axis_;
        auto flatten_input = flatten_input_;
        auto reverse = reverse_;
        auto inclusive = inclusive_;
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_bytes(axis, 6);
        encoder.set_bytes(flatten_input, 7);
        encoder.set_bytes(reverse, 8);
        encoder.set_bytes(inclusive, 9);

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
    bool flatten_input_;
    bool reverse_;
    bool inclusive_;
};

class PreciseFloat64GpuArgsort : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuArgsort(mx::Stream stream, int axis, bool flatten_input)
        : mx::UnaryPrimitive(stream),
          axis_(axis),
          flatten_input_(flatten_input) {}

    const char* name() const override {
        return "PreciseFloat64GpuArgsort";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        if (flatten_input_) {
            return {mx::Shape{static_cast<mx::ShapeElem>(inputs[0].size())}};
        }
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuArgsort is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_argsort",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline bool dd_less(float2 a, float2 b) {
                        if (isnan(a.x) || isnan(b.x)) {
                            return !isnan(a.x) && isnan(b.x);
                        }
                        if (a.x < b.x) {
                            return true;
                        }
                        if (a.x > b.x) {
                            return false;
                        }
                        return a.y < b.y;
                    }

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_argsort(
                        device const float2* input [[buffer(0)]],
                        device uint* output [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant long* output_strides [[buffer(6)]],
                        constant int& axis [[buffer(7)]],
                        constant bool& flatten_input [[buffer(8)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }

                        if (flatten_input) {
                            auto current = input[
                                row_major_offset(index, shape, input_strides, ndim)];
                            ulong rank = 0;
                            for (ulong other_index = 0; other_index < size; ++other_index) {
                                auto other = input[
                                    row_major_offset(
                                        other_index, shape, input_strides, ndim)];
                                bool before = dd_less(other, current);
                                bool equal = !dd_less(current, other) && !before;
                                if (before || (equal && other_index < index)) {
                                    ++rank;
                                }
                            }
                            output[rank] = static_cast<uint>(index);
                            return;
                        }

                        ulong remainder = index;
                        long input_offset = 0;
                        long output_base = 0;
                        ulong axis_coord = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint dim_axis = ndim - 1 - step;
                            ulong dim = shape[dim_axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_offset += static_cast<long>(coord)
                                * input_strides[dim_axis];
                            if (static_cast<int>(dim_axis) == axis) {
                                axis_coord = coord;
                            } else {
                                output_base += static_cast<long>(coord)
                                    * output_strides[dim_axis];
                            }
                        }

                        long slice_base = input_offset
                            - static_cast<long>(axis_coord) * input_strides[axis];
                        auto current = input[input_offset];
                        ulong axis_size = shape[axis];
                        ulong rank = 0;
                        for (ulong other_axis_coord = 0;
                                other_axis_coord < axis_size;
                                ++other_axis_coord) {
                            auto other = input[
                                slice_base
                                + static_cast<long>(other_axis_coord)
                                    * input_strides[axis]];
                            bool before = dd_less(other, current);
                            bool equal = !dd_less(current, other) && !before;
                            if (before || (equal && other_axis_coord < axis_coord)) {
                                ++rank;
                            }
                        }
                        output[
                            output_base
                            + static_cast<long>(rank) * output_strides[axis]] =
                                static_cast<uint>(axis_coord);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_argsort", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        std::vector<std::int64_t> output_strides(
            out.strides().begin(), out.strides().end());
        auto axis = axis_;
        auto flatten_input = flatten_input_;
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_vector_bytes(output_strides, 6);
        encoder.set_bytes(axis, 7);
        encoder.set_bytes(flatten_input, 8);

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
    bool flatten_input_;
};

class PreciseFloat64GpuSort : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuSort(mx::Stream stream, int axis, bool flatten_input)
        : mx::UnaryPrimitive(stream),
          axis_(axis),
          flatten_input_(flatten_input) {}

    const char* name() const override {
        return "PreciseFloat64GpuSort";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        if (flatten_input_) {
            return {mx::Shape{static_cast<mx::ShapeElem>(inputs[0].size())}};
        }
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuSort is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_sort",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    inline bool dd_less(float2 a, float2 b) {
                        if (isnan(a.x) || isnan(b.x)) {
                            return !isnan(a.x) && isnan(b.x);
                        }
                        if (a.x < b.x) {
                            return true;
                        }
                        if (a.x > b.x) {
                            return false;
                        }
                        return a.y < b.y;
                    }

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_sort(
                        device const float2* input [[buffer(0)]],
                        device float2* output [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant long* output_strides [[buffer(6)]],
                        constant int& axis [[buffer(7)]],
                        constant bool& flatten_input [[buffer(8)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }

                        if (flatten_input) {
                            auto current = input[
                                row_major_offset(index, shape, input_strides, ndim)];
                            ulong rank = 0;
                            for (ulong other_index = 0; other_index < size; ++other_index) {
                                auto other = input[
                                    row_major_offset(
                                        other_index, shape, input_strides, ndim)];
                                bool before = dd_less(other, current);
                                bool equal = !dd_less(current, other) && !before;
                                if (before || (equal && other_index < index)) {
                                    ++rank;
                                }
                            }
                            output[rank] = current;
                            return;
                        }

                        ulong remainder = index;
                        long input_offset = 0;
                        long output_base = 0;
                        ulong axis_coord = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint dim_axis = ndim - 1 - step;
                            ulong dim = shape[dim_axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            input_offset += static_cast<long>(coord)
                                * input_strides[dim_axis];
                            if (static_cast<int>(dim_axis) == axis) {
                                axis_coord = coord;
                            } else {
                                output_base += static_cast<long>(coord)
                                    * output_strides[dim_axis];
                            }
                        }

                        long slice_base = input_offset
                            - static_cast<long>(axis_coord) * input_strides[axis];
                        auto current = input[input_offset];
                        ulong axis_size = shape[axis];
                        ulong rank = 0;
                        for (ulong other_axis_coord = 0;
                                other_axis_coord < axis_size;
                                ++other_axis_coord) {
                            auto other = input[
                                slice_base
                                + static_cast<long>(other_axis_coord)
                                    * input_strides[axis]];
                            bool before = dd_less(other, current);
                            bool equal = !dd_less(current, other) && !before;
                            if (before || (equal && other_axis_coord < axis_coord)) {
                                ++rank;
                            }
                        }
                        output[
                            output_base
                            + static_cast<long>(rank) * output_strides[axis]] =
                                current;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_sort", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        std::vector<std::int64_t> output_strides(
            out.strides().begin(), out.strides().end());
        auto axis = axis_;
        auto flatten_input = flatten_input_;
        encoder.set_bytes(size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_vector_bytes(output_strides, 6);
        encoder.set_bytes(axis, 7);
        encoder.set_bytes(flatten_input, 8);

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
    bool flatten_input_;
};

class PreciseFloat64GpuReduceSum : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuReduceSum(mx::Stream stream,
                               std::vector<int> axes,
                               bool keepdims)
        : mx::UnaryPrimitive(stream),
          axes_(std::move(axes)),
          keepdims_(keepdims) {}

    const char* name() const override {
        return "PreciseFloat64GpuReduceSum";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        mx::Shape shape;
        for (int axis = 0; axis < inputs[0].ndim(); ++axis) {
            bool reduce_axis = std::find(axes_.begin(), axes_.end(), axis)
                != axes_.end();
            if (reduce_axis) {
                if (keepdims_) {
                    shape.push_back(1);
                }
            } else {
                shape.push_back(inputs[0].shape(axis));
            }
        }
        return {std::move(shape)};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuReduceSum is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_reduce_sum",
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

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_reduce_sum(
                        device const float2* input [[buffer(0)]],
                        device float2* output [[buffer(1)]],
                        constant ulong& input_size [[buffer(2)]],
                        constant uint& ndim [[buffer(3)]],
                        constant ulong* input_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant ulong& output_size [[buffer(6)]],
                        constant uint& output_ndim [[buffer(7)]],
                        constant ulong* output_shape [[buffer(8)]],
                        constant long* output_strides [[buffer(9)]],
                        constant uchar* reduce_axes [[buffer(10)]],
                        constant ulong& reduce_size [[buffer(11)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto out_index = static_cast<ulong>(gid);
                        if (out_index >= output_size) {
                            return;
                        }
                        float2 running = float2(0.0f, 0.0f);

                        ulong remainder = out_index;
                        long input_base = 0;
                        for (uint step = 0; step < output_ndim; ++step) {
                            uint out_axis = output_ndim - 1 - step;
                            ulong dim = output_shape[out_axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            uint input_axis = out_axis;
                            if (output_ndim != ndim) {
                                uint seen_output_axes = 0;
                                for (uint candidate = 0; candidate < ndim; ++candidate) {
                                    if (reduce_axes[candidate]) {
                                        continue;
                                    }
                                    if (seen_output_axes == out_axis) {
                                        input_axis = candidate;
                                        break;
                                    }
                                    ++seen_output_axes;
                                }
                            }
                            if (!reduce_axes[input_axis]) {
                                input_base += static_cast<long>(coord)
                                    * input_strides[input_axis];
                            }
                        }

                        for (ulong reduce_index = 0;
                                reduce_index < reduce_size;
                                ++reduce_index) {
                            ulong reduce_remainder = reduce_index;
                            long reduce_offset = 0;
                            for (uint step = 0; step < ndim; ++step) {
                                uint input_axis = ndim - 1 - step;
                                if (!reduce_axes[input_axis]) {
                                    continue;
                                }
                                ulong dim = input_shape[input_axis];
                                ulong coord = dim == 0 ? 0 : reduce_remainder % dim;
                                if (dim != 0) {
                                    reduce_remainder /= dim;
                                }
                                reduce_offset += static_cast<long>(coord)
                                    * input_strides[input_axis];
                            }
                            running = dd_add(running, input[input_base + reduce_offset]);
                        }
                        output[out_index] = running;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_reduce_sum", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto input_size = static_cast<std::uint64_t>(inputs[0].size());
        auto ndim = static_cast<std::uint32_t>(inputs[0].ndim());
        auto input_shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        auto output_size = static_cast<std::uint64_t>(out.size());
        auto output_ndim = static_cast<std::uint32_t>(out.ndim());
        auto output_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> output_strides(
            out.strides().begin(), out.strides().end());
        std::vector<std::uint8_t> reduce_axes(inputs[0].ndim(), 0);
        std::uint64_t reduce_size = 1;
        for (auto axis : axes_) {
            reduce_axes[axis] = 1;
            reduce_size *= static_cast<std::uint64_t>(inputs[0].shape(axis));
        }
        encoder.set_bytes(input_size, 2);
        encoder.set_bytes(ndim, 3);
        encoder.set_vector_bytes(input_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_bytes(output_size, 6);
        encoder.set_bytes(output_ndim, 7);
        encoder.set_vector_bytes(output_shape, 8);
        encoder.set_vector_bytes(output_strides, 9);
        encoder.set_vector_bytes(reduce_axes, 10);
        encoder.set_bytes(reduce_size, 11);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    std::vector<int> axes_;
    bool keepdims_;
};
class PreciseFloat64GpuArcTan2 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuArcTan2(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuArcTan2";
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
            result[i] = PreciseFloat64(std::atan2(lhs, rhs));
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
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuAbs : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAbs(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAbs";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<PreciseFloat64>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            auto value = source[row_major_offset(
                i, inputs[0].shape(), inputs[0].strides())].value();
            result[i] = PreciseFloat64(std::fabs(value));
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
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuUnaryMath : public mx::UnaryPrimitive {
public:
    PreciseFloat64GpuUnaryMath(mx::Stream stream, PreciseFloat64UnaryMathOp op)
        : mx::UnaryPrimitive(stream), op_(op) {}

    const char* name() const override {
        return "PreciseFloat64GpuUnaryMath";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuUnaryMath is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_unary_math",
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

                    inline float2 dd_sqrt(float2 a) {
                        float root = sqrt(a.x + a.y);
                        if (!isfinite(root) || root == 0.0f) {
                            return float2(root, 0.0f);
                        }
                        float2 estimate = float2(root, 0.0f);
                        float2 residual = dd_sub(a, dd_mul(estimate, estimate));
                        return dd_add(
                            estimate,
                            float2(0.5f * residual.x / root, 0.0f));
                    }

	                    kernel void mlx_matplotlib_precise_float64_unary_math(
	                        device const float2* in [[buffer(0)]],
	                        device float2* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant uint& op [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        float value = in[index].x + in[index].y;
                        float computed;
                        if (op == 0) {
                            computed = log(value);
                        } else if (op == 1) {
                            computed = log2(value);
                        } else if (op == 2) {
                            computed = log10(value);
	                        } else if (op == 3) {
	                            computed = floor(value);
	                        } else if (op == 4) {
	                            computed = ceil(value);
                        } else if (op == 5) {
                            out[index] = dd_sqrt(in[index]);
                            return;
                        } else if (op == 6) {
                            computed = value * 57.295779513082320876798154814105f;
                        } else if (op == 7) {
                            computed = value * 0.01745329251994329576923690768489f;
                        } else if (op == 8) {
                            computed = sin(value);
                        } else if (op == 9) {
                            computed = cos(value);
                        } else if (op == 10) {
                            computed = asin(value);
                        } else if (op == 11) {
                            computed = acos(value);
                        } else {
                            computed = atan(value);
                        }
		                        out[index] = float2(computed, 0.0f);
	                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_unary_math", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto size = static_cast<std::uint64_t>(out.data_size());
        auto op = static_cast<std::uint32_t>(op_);
        encoder.set_bytes(size, 2);
        encoder.set_bytes(op, 3);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    PreciseFloat64UnaryMathOp op_;
};

class PreciseFloat64GpuConv1d : public mx::Primitive {
public:
    PreciseFloat64GpuConv1d(mx::Stream stream,
                            mx::Shape output_shape,
                            int stride,
                            int padding,
                            int dilation,
                            int groups)
        : mx::Primitive(stream),
          output_shape_(std::move(output_shape)),
          stride_(stride),
          padding_(padding),
          dilation_(dilation),
          groups_(groups) {}

    const char* name() const override {
        return "PreciseFloat64GpuConv1d";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>&) override {
        return {output_shape_};
    }

    void eval_cpu(const std::vector<mx::array>&,
                  std::vector<mx::array>&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuConv1d is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto& out = outputs[0];
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_precise_float64_conv1d",
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

                    inline long row_major_offset(
                        ulong index,
                        constant ulong* shape,
                        constant long* strides,
                        uint ndim) {
                        long offset = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = shape[axis];
                            ulong coord = dim == 0 ? 0 : index % dim;
                            if (dim != 0) {
                                index /= dim;
                            }
                            offset += static_cast<long>(coord) * strides[axis];
                        }
                        return offset;
                    }

                    kernel void mlx_matplotlib_precise_float64_conv1d(
                        device const float2* input [[buffer(0)]],
                        device const float2* weight [[buffer(1)]],
                        device float2* output [[buffer(2)]],
                        constant ulong& output_size [[buffer(3)]],
                        constant ulong* input_shape [[buffer(4)]],
                        constant long* input_strides [[buffer(5)]],
                        constant ulong* weight_shape [[buffer(6)]],
                        constant long* weight_strides [[buffer(7)]],
                        constant ulong* output_shape [[buffer(8)]],
                        constant long* output_strides [[buffer(9)]],
                        constant int& stride [[buffer(10)]],
                        constant int& padding [[buffer(11)]],
                        constant int& dilation [[buffer(12)]],
                        constant int& groups [[buffer(13)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto out_index = static_cast<ulong>(gid);
                        if (out_index >= output_size) {
                            return;
                        }

                        ulong channel = out_index % output_shape[2];
                        ulong position = (out_index / output_shape[2]) % output_shape[1];
                        ulong batch = out_index / (output_shape[2] * output_shape[1]);
                        ulong channels_per_group = output_shape[2] / static_cast<ulong>(groups);
                        ulong input_channels_per_group = weight_shape[2];
                        ulong group = channel / channels_per_group;
                        ulong input_channel_base = group * input_channels_per_group;

                        float2 running = float2(0.0f, 0.0f);
                        for (ulong kernel_index = 0; kernel_index < weight_shape[1];
                                ++kernel_index) {
                            long input_position = static_cast<long>(position)
                                    * static_cast<long>(stride)
                                + static_cast<long>(kernel_index)
                                    * static_cast<long>(dilation)
                                - static_cast<long>(padding);
                            if (input_position < 0
                                    || input_position >= static_cast<long>(input_shape[1])) {
                                continue;
                            }
                            for (ulong local_channel = 0;
                                    local_channel < input_channels_per_group;
                                    ++local_channel) {
                                ulong input_channel = input_channel_base + local_channel;
                                long input_offset =
                                    static_cast<long>(batch) * input_strides[0]
                                    + input_position * input_strides[1]
                                    + static_cast<long>(input_channel) * input_strides[2];
                                long weight_offset =
                                    static_cast<long>(channel) * weight_strides[0]
                                    + static_cast<long>(kernel_index) * weight_strides[1]
                                    + static_cast<long>(local_channel) * weight_strides[2];
                                running = dd_add(
                                    running,
                                    dd_mul(
                                        quick_two_sum(
                                            input[input_offset].x,
                                            input[input_offset].y),
                                        quick_two_sum(
                                            weight[weight_offset].x,
                                            weight[weight_offset].y)));
                            }
                        }

                        auto output_offset = row_major_offset(
                            out_index, output_shape, output_strides, 3);
                        output[output_offset] = quick_two_sum(
                            running.x, running.y);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_conv1d", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);

        auto output_size = static_cast<std::uint64_t>(out.data_size());
        auto input_shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> input_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        auto weight_shape = shape_bytes(inputs[1].shape());
        std::vector<std::int64_t> weight_strides(
            inputs[1].strides().begin(), inputs[1].strides().end());
        auto output_shape = shape_bytes(out.shape());
        std::vector<std::int64_t> output_strides(
            out.strides().begin(), out.strides().end());

        encoder.set_bytes(output_size, 3);
        encoder.set_vector_bytes(input_shape, 4);
        encoder.set_vector_bytes(input_strides, 5);
        encoder.set_vector_bytes(weight_shape, 6);
        encoder.set_vector_bytes(weight_strides, 7);
        encoder.set_vector_bytes(output_shape, 8);
        encoder.set_vector_bytes(output_strides, 9);
        encoder.set_bytes(stride_, 10);
        encoder.set_bytes(padding_, 11);
        encoder.set_bytes(dilation_, 12);
        encoder.set_bytes(groups_, 13);

        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
    }

private:
    mx::Shape output_shape_;
    int stride_;
    int padding_;
    int dilation_;
    int groups_;
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
        auto& encoder = precise_command_encoder(stream());
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
            result[i] = PreciseFloat64(std::rint(value * scale) / scale);
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

                    inline int dd_compare(float2 a, float2 b) {
                        float2 diff = dd_sub(a, b);
                        if (diff.x > 0.0f) {
                            return 1;
                        }
                        if (diff.x < 0.0f) {
                            return -1;
                        }
                        if (diff.y > 0.0f) {
                            return 1;
                        }
                        if (diff.y < 0.0f) {
                            return -1;
                        }
                        return 0;
                    }

                    inline float dd_floor_float(float2 value) {
                        float base = floor(value.x);
                        if (dd_compare(value, float2(base, 0.0f)) < 0) {
                            base -= 1.0f;
                        }
                        return base;
                    }

                    inline float2 dd_rint(float2 value) {
                        float base = dd_floor_float(value);
                        float2 fraction = dd_sub(value, float2(base, 0.0f));
                        int cmp = dd_compare(fraction, float2(0.5f, 0.0f));
                        float rounded = base;
                        if (cmp > 0) {
                            rounded = base + 1.0f;
                        } else if (cmp == 0) {
                            float parity = fmod(abs(base), 2.0f);
                            if (parity != 0.0f) {
                                rounded = base + 1.0f;
                            }
                        }
                        return float2(rounded, 0.0f);
                    }

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
                        float2 scaled = dd_mul(in[input_index], float2(scale, 0.0f));
                        out[index] = dd_div(dd_rint(scaled), float2(scale, 0.0f));
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_round", library);
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuMatmul : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuMatmul(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuMatmul";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {matmul_output_shape(inputs[0], inputs[1])};
    }

    void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
        throw std::runtime_error(
            "PreciseFloat64GpuMatmul is only valid on a GPU stream");
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        if (out.size() == 0) {
            out.set_data(mx::allocator::Buffer(nullptr));
            return;
        }
        auto left_ndim = inputs[0].ndim();
        auto right_ndim = inputs[1].ndim();

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
                        constant ulong& output_size [[buffer(3)]],
                        constant uint& output_ndim [[buffer(4)]],
                        constant ulong* output_shape [[buffer(5)]],
                        constant uint& lhs_ndim [[buffer(6)]],
                        constant ulong* lhs_shape [[buffer(7)]],
                        constant long* lhs_strides [[buffer(8)]],
                        constant uint& rhs_ndim [[buffer(9)]],
                        constant ulong* rhs_shape [[buffer(10)]],
                        constant long* rhs_strides [[buffer(11)]],
                        constant ulong& k [[buffer(12)]],
                        constant bool& lhs_vector [[buffer(13)]],
                        constant bool& rhs_vector [[buffer(14)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto out_index = static_cast<ulong>(gid);
                        if (out_index >= output_size) {
                            return;
                        }

                        uint lhs_batch_ndim = lhs_vector ? 0 : lhs_ndim - 2;
                        uint rhs_batch_ndim = rhs_vector ? 0 : rhs_ndim - 2;
                        uint matrix_ndim = (lhs_vector ? 0 : 1)
                            + (rhs_vector ? 0 : 1);
                        uint batch_ndim = output_ndim - matrix_ndim;
                        ulong row = 0;
                        ulong col = 0;
                        ulong remainder = out_index;
                        long lhs_base = 0;
                        long rhs_base = 0;

                        for (uint step = 0; step < output_ndim; ++step) {
                            uint out_axis = output_ndim - 1 - step;
                            ulong dim = output_shape[out_axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }

                            if (out_axis < batch_ndim) {
                                if (lhs_batch_ndim > 0
                                        && out_axis >= batch_ndim - lhs_batch_ndim) {
                                    uint lhs_axis = out_axis
                                        - (batch_ndim - lhs_batch_ndim);
                                    if (lhs_shape[lhs_axis] != 1) {
                                        lhs_base += static_cast<long>(coord)
                                            * lhs_strides[lhs_axis];
                                    }
                                }
                                if (rhs_batch_ndim > 0
                                        && out_axis >= batch_ndim - rhs_batch_ndim) {
                                    uint rhs_axis = out_axis
                                        - (batch_ndim - rhs_batch_ndim);
                                    if (rhs_shape[rhs_axis] != 1) {
                                        rhs_base += static_cast<long>(coord)
                                            * rhs_strides[rhs_axis];
                                    }
                                }
                                continue;
                            }

                            uint matrix_axis = out_axis - batch_ndim;
                            if (!lhs_vector && matrix_axis == 0) {
                                row = coord;
                            } else if (!rhs_vector) {
                                col = coord;
                            }
                        }

                        float2 sum = float2(0.0f, 0.0f);
                        for (ulong inner = 0; inner < k; ++inner) {
                            auto lhs_index = lhs_vector
                                ? static_cast<long>(inner) * lhs_strides[0]
                                : lhs_base
                                    + static_cast<long>(row)
                                        * lhs_strides[lhs_ndim - 2]
                                    + static_cast<long>(inner)
                                        * lhs_strides[lhs_ndim - 1];
                            auto rhs_index = rhs_vector
                                ? static_cast<long>(inner) * rhs_strides[0]
                                : rhs_base
                                    + static_cast<long>(inner)
                                        * rhs_strides[rhs_ndim - 2]
                                    + static_cast<long>(col)
                                        * rhs_strides[rhs_ndim - 1];
                            sum = dd_add(sum, dd_mul(lhs[lhs_index], rhs[rhs_index]));
                        }
                        out[out_index] = sum;
	                    }
	                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_matmul", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 2);
        auto output_size = static_cast<std::uint64_t>(out.data_size());
        auto output_ndim = static_cast<std::uint32_t>(out.ndim());
        auto output_shape = shape_bytes(out.shape());
        auto lhs_ndim = static_cast<std::uint32_t>(left_ndim);
        auto lhs_shape = shape_bytes(inputs[0].shape());
        std::vector<std::int64_t> lhs_strides(
            inputs[0].strides().begin(), inputs[0].strides().end());
        auto rhs_ndim = static_cast<std::uint32_t>(right_ndim);
        auto rhs_shape = shape_bytes(inputs[1].shape());
        std::vector<std::int64_t> rhs_strides(
            inputs[1].strides().begin(), inputs[1].strides().end());
        auto k = static_cast<std::uint64_t>(inputs[0].shape(left_ndim - 1));
        auto lhs_vector = left_ndim == 1;
        auto rhs_vector = right_ndim == 1;
        encoder.set_bytes(output_size, 3);
        encoder.set_bytes(output_ndim, 4);
        encoder.set_vector_bytes(output_shape, 5);
        encoder.set_bytes(lhs_ndim, 6);
        encoder.set_vector_bytes(lhs_shape, 7);
        encoder.set_vector_bytes(lhs_strides, 8);
        encoder.set_bytes(rhs_ndim, 9);
        encoder.set_vector_bytes(rhs_shape, 10);
        encoder.set_vector_bytes(rhs_strides, 11);
        encoder.set_bytes(k, 12);
        encoder.set_bytes(lhs_vector, 13);
        encoder.set_bytes(rhs_vector, 14);
        auto threads = static_cast<NS::UInteger>(out.data_size());
        auto group_size = kernel->maxTotalThreadsPerThreadgroup();
        if (group_size > threads) {
            group_size = threads;
        }
        encoder.dispatch_threads(
            MTL::Size(threads, 1, 1), MTL::Size(group_size, 1, 1));
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
        auto& encoder = precise_command_encoder(stream());
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
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuAstypeInt64 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAstypeInt64(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAstypeInt64";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<std::int64_t>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = static_cast<std::int64_t>(source[broadcast_offset(
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
            "mlx_matplotlib_precise_float64_astype_int64",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_astype_int64(
                        device const float2* in [[buffer(0)]],
                        device long* out [[buffer(1)]],
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
                        out[index] = static_cast<long>(value.x + value.y);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_astype_int64", library);
        auto& encoder = precise_command_encoder(stream());
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

class PreciseFloat64GpuAstypeUint8 : public mx::UnaryPrimitive {
public:
    explicit PreciseFloat64GpuAstypeUint8(mx::Stream stream)
        : mx::UnaryPrimitive(stream) {}

    const char* name() const override {
        return "PreciseFloat64GpuAstypeUint8";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>& inputs) override {
        return {inputs[0].shape()};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto source = inputs[0].data<PreciseFloat64>();
        auto result = out.data<std::uint8_t>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            result[i] = static_cast<std::uint8_t>(source[broadcast_offset(
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
            "mlx_matplotlib_precise_float64_astype_uint8",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_precise_float64_astype_uint8(
                        device const float2* in [[buffer(0)]],
                        device uchar* out [[buffer(1)]],
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
                        out[index] = static_cast<uchar>(value.x + value.y);
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_astype_uint8", library);
        auto& encoder = precise_command_encoder(stream());
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
        auto& encoder = precise_command_encoder(stream());
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

class BincountInt32 : public mx::UnaryPrimitive {
public:
    BincountInt32(mx::Stream stream, int minlength)
        : mx::UnaryPrimitive(stream), minlength_(minlength) {}

    const char* name() const override {
        return "BincountInt32";
    }

    std::vector<mx::Shape> output_shapes(
        const std::vector<mx::array>&) override {
        return {{minlength_}};
    }

    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto result = out.data<std::int32_t>();
        std::fill(result, result + out.size(), 0);
        auto source = inputs[0].data<std::int32_t>();
        for (std::size_t i = 0; i < inputs[0].size(); ++i) {
            auto value = source[row_major_offset(
                i, inputs[0].shape(), inputs[0].strides())];
            if (value >= 0 && value < minlength_) {
                ++result[value];
            }
        }
    }

    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto& device = mx::metal::device(stream().device);
        auto library = device.get_library(
            "mlx_matplotlib_bincount_int32",
            [] {
                return R"(
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mlx_matplotlib_bincount_int32_zero(
                        device atomic_int* out [[buffer(0)]],
                        constant ulong& size [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index < size) {
                            atomic_store_explicit(
                                &out[index], 0, memory_order_relaxed);
                        }
                    }

                    kernel void mlx_matplotlib_bincount_int32_count(
                        device const int* in [[buffer(0)]],
                        device atomic_int* out [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant int& minlength [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        int value = in[index];
                        if (value >= 0 && value < minlength) {
                            atomic_fetch_add_explicit(
                                &out[value], 1, memory_order_relaxed);
                        }
                    }
                )";
            });
        auto& encoder = precise_command_encoder(stream());

        auto zero_kernel = device.get_kernel(
            "mlx_matplotlib_bincount_int32_zero", library);
        encoder.set_compute_pipeline_state(zero_kernel);
        encoder.set_output_array(out, 0);
        auto out_size = static_cast<std::uint64_t>(out.size());
        encoder.set_bytes(out_size, 1);
        auto zero_threads = static_cast<NS::UInteger>(out.size());
        auto zero_group_size = zero_kernel->maxTotalThreadsPerThreadgroup();
        if (zero_group_size > zero_threads) {
            zero_group_size = zero_threads;
        }
        if (zero_threads > 0) {
            encoder.dispatch_threads(
                MTL::Size(zero_threads, 1, 1),
                MTL::Size(zero_group_size, 1, 1));
        }

        auto count_kernel = device.get_kernel(
            "mlx_matplotlib_bincount_int32_count", library);
        encoder.set_compute_pipeline_state(count_kernel);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(out, 1);
        auto input_size = static_cast<std::uint64_t>(inputs[0].size());
        encoder.set_bytes(input_size, 2);
        encoder.set_bytes(minlength_, 3);
        auto count_threads = static_cast<NS::UInteger>(inputs[0].size());
        auto count_group_size = count_kernel->maxTotalThreadsPerThreadgroup();
        if (count_group_size > count_threads) {
            count_group_size = count_threads;
        }
        if (count_threads > 0) {
            encoder.dispatch_threads(
                MTL::Size(count_threads, 1, 1),
                MTL::Size(count_group_size, 1, 1));
        }
    }

private:
    int minlength_;
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
                                bool better = is_max
                                    ? (current.x > best.x
                                       || (current.x == best.x && current.y > best.y))
                                    : (current.x < best.x
                                       || (current.x == best.x && current.y < best.y));
                                if (r == 0 || better) {
                                    best = current;
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
                        for (ulong r = 1; r < reduce_size; ++r) {
                            float2 current = in[base + r * reduce_stride];
                            bool better = is_max
                                ? (current.x > best.x
                                   || (current.x == best.x && current.y > best.y))
                                : (current.x < best.x
                                   || (current.x == best.x && current.y < best.y));
                            if (better) {
                                best = current;
                            }
                        }
                        out[index] = best;
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_reduce_minmax", library);
        auto& encoder = precise_command_encoder(stream());
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
            for (std::size_t i = 0; i < input.size(); ++i) {
                result[offset + i] = source[row_major_offset(
                    i, input.shape(), input.strides())];
            }
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
                        constant uint& ndim [[buffer(4)]],
                        constant ulong* input_shape [[buffer(5)]],
                        constant long* input_strides [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index < row_size) {
                            ulong remainder = index;
                            long input_offset = 0;
                            for (uint step = 0; step < ndim; ++step) {
                                uint axis = ndim - 1 - step;
                                ulong dim = input_shape[axis];
                                ulong coord = dim == 0 ? 0 : remainder % dim;
                                if (dim != 0) {
                                    remainder /= dim;
                                }
                                input_offset += static_cast<long>(coord)
                                    * input_strides[axis];
                            }
                            out[offset + index] = in[input_offset];
                        }
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_stack_axis0", library);
        auto& encoder = precise_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        std::uint64_t offset = 0;
        for (const auto& input : inputs) {
            encoder.set_input_array(input, 0);
            encoder.set_output_array(out, 1);
            auto row_size = static_cast<std::uint64_t>(input.size());
            auto ndim = static_cast<std::uint32_t>(input.ndim());
            auto input_shape = shape_bytes(input.shape());
            std::vector<std::int64_t> input_strides(
                input.strides().begin(), input.strides().end());
            encoder.set_bytes(row_size, 2);
            encoder.set_bytes(offset, 3);
            encoder.set_bytes(ndim, 4);
            encoder.set_vector_bytes(input_shape, 5);
            encoder.set_vector_bytes(input_strides, 6);
            auto threads = static_cast<NS::UInteger>(input.size());
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
                        constant long* in_strides [[buffer(8)]],
                        uint gid [[thread_position_in_grid]]) {
                        auto index = static_cast<ulong>(gid);
                        if (index >= size) {
                            return;
                        }
                        ulong remainder = index;
                        long in_index = 0;
                        ulong out_index = 0;
                        for (uint step = 0; step < ndim; ++step) {
                            uint axis = ndim - 1 - step;
                            ulong dim = in_shape[axis];
                            ulong coord = dim == 0 ? 0 : remainder % dim;
                            if (dim != 0) {
                                remainder /= dim;
                            }
                            in_index += static_cast<long>(coord)
                                * in_strides[axis];
                            ulong out_coord = coord;
                            if (axis == concat_axis) {
                                out_coord += axis_offset;
                            }
                            out_index += out_coord * out_strides[axis];
                        }
                        out[out_index] = in[in_index];
                    }
                )";
            });
        auto kernel = device.get_kernel(
            "mlx_matplotlib_precise_float64_concatenate", library);
        auto& encoder = precise_command_encoder(stream());
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
            auto size = static_cast<std::uint64_t>(input.size());
            auto in_shape = shape_bytes(input.shape());
            std::vector<std::int64_t> in_strides(
                input.strides().begin(), input.strides().end());
            encoder.set_bytes(size, 2);
            encoder.set_bytes(ndim, 3);
            encoder.set_bytes(concat_axis, 4);
            encoder.set_bytes(axis_offset, 5);
            encoder.set_vector_bytes(in_shape, 6);
            encoder.set_vector_bytes(out_strides, 7);
            encoder.set_vector_bytes(in_strides, 8);
            auto threads = static_cast<NS::UInteger>(input.size());
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

    MlxPreciseArray(mx::array value,
                    mx::StreamOrDevice stream,
                    std::vector<std::string> structured_names)
        : mx::array(std::move(value)),
          stream_(std::move(stream)),
          structured_names_(std::move(structured_names)) {}

    static MlxPreciseArray make(nb::handle value,
                                nb::object dtype,
                                const mx::StreamOrDevice& stream);

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
    MlxPreciseArray remainder(nb::handle other) const;
    MlxPreciseArray reverse_remainder(nb::handle other) const;
    MlxPreciseArray matmul(nb::handle other) const;
    MlxPreciseArray reverse_matmul(nb::handle other) const;
    MlxPreciseArray negative() const;
    const mx::StreamOrDevice& stream_or_device() const {
        return stream_;
    }
    bool has_structured_fields() const {
        return !structured_names_.empty();
    }
    const std::vector<std::string>& structured_names() const {
        return structured_names_;
    }
    void set_structured_names(std::vector<std::string> names) {
        structured_names_ = std::move(names);
    }

    static void eval(mx::array array)
    {
        if (array.status() == mx::array::Status::available) {
            return;
        }
        if (array.status() == mx::array::Status::evaluated) {
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
        std::vector<bool> reset_inputs;
        reset_inputs.reserve(inputs.size());
        for (auto& input : inputs) {
            reset_inputs.push_back(
                input.status() == mx::array::Status::unscheduled);
            eval(input);
        }

        auto outputs = array.outputs();
        array.primitive().eval_gpu(inputs, outputs);
        array.set_status(mx::array::Status::evaluated);
        mx::synchronize(stream);
        array.set_status(mx::array::Status::available);
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (reset_inputs[i] && inputs[i].has_primitive()) {
                inputs[i].set_status(mx::array::Status::unscheduled);
            }
        }
    }

private:
    mx::StreamOrDevice stream_;
    std::vector<std::string> structured_names_;
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
        array = mx::contiguous(array, false, mx::Device(mx::Device::cpu));
        array.eval();
        std::vector<double> values;
        values.reserve(array.size());
        if (array.dtype() == mx::bool_) {
            auto ptr = array.data<bool>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(ptr[i] ? 1.0 : 0.0);
            }
        } else if (array.dtype() == mx::int8) {
            auto ptr = array.data<std::int8_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::int16) {
            auto ptr = array.data<std::int16_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::int32) {
            auto ptr = array.data<std::int32_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::int64) {
            auto ptr = array.data<std::int64_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::uint8) {
            auto ptr = array.data<std::uint8_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::uint16) {
            auto ptr = array.data<std::uint16_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::uint32) {
            auto ptr = array.data<std::uint32_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::uint64) {
            auto ptr = array.data<std::uint64_t>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else if (array.dtype() == mx::float32) {
            auto ptr = array.data<float>();
            for (std::size_t i = 0; i < array.size(); ++i) {
                values.push_back(static_cast<double>(ptr[i]));
            }
        } else {
            array = mx::astype(array, mx::float64, mx::Device(mx::Device::cpu));
            array.eval();
            auto ptr = array.data<double>();
            values.assign(ptr, ptr + array.size());
        }
        return values;
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
            auto offset = array.flags().row_contiguous
                ? i
                : row_major_offset(i, array.shape(), array.strides());
            values.push_back(ptr[offset].value());
        }
    } else {
        auto ptr = array.data<double>();
        for (std::size_t i = 0; i < array.size(); ++i) {
            auto offset = array.flags().row_contiguous
                ? i
                : row_major_offset(i, array.shape(), array.strides());
            values.push_back(ptr[offset]);
        }
    }
    return values;
}

nb::object nested_float64_values(const std::vector<double>& values,
                                 const mx::Shape& shape,
                                 std::size_t axis,
                                 std::size_t& offset)
{
    if (axis == shape.size()) {
        return nb::float_(values[offset++]);
    }
    nb::list result;
    for (mx::ShapeElem i = 0; i < shape[axis]; ++i) {
        result.append(nested_float64_values(values, shape, axis + 1, offset));
    }
    return std::move(result);
}

nb::object tolist_precise(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.dtype() != mx::float64) {
        throw nb::type_error("tolist_precise only handles float64 arrays");
    }
    auto shape = array.shape();
    auto values = read_precise_float64_values(std::move(array));
    std::size_t offset = 0;
    return nested_float64_values(values, shape, 0, offset);
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
        return mx::float64;
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
        auto row = MlxPreciseArray::make(item, dtype, stream);
        wants_float64 = wants_float64 || row.dtype() == mx::float64;
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        return MlxPreciseArray::make(value, dtype, stream);
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
    if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) {
        return mx::bool_;
    }
    if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {
        return mx::int64;
    }
    if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {
        return mx::float64;
    }
    if (PyUnicode_Check(dtype.ptr())) {
        auto name = nb::cast<std::string>(dtype);
        if (!name.empty()
                && (name[0] == '<' || name[0] == '>' || name[0] == '='
                    || name[0] == '|' || name[0] == '!')) {
            name.erase(name.begin());
        }
        if (name == "?" || name == "bool" || name == "bool_") {
            return mx::bool_;
        }
        if (name == "b" || name == "i1" || name == "int8") {
            return mx::int8;
        }
        if (name == "h" || name == "i2" || name == "int16") {
            return mx::int16;
        }
        if (name == "i" || name == "i4" || name == "int32") {
            return mx::int32;
        }
        if (name == "l" || name == "q" || name == "i8"
                || name == "int" || name == "int64") {
            return mx::int64;
        }
        if (name == "B" || name == "u1" || name == "uint8") {
            return mx::uint8;
        }
        if (name == "H" || name == "u2" || name == "uint16") {
            return mx::uint16;
        }
        if (name == "I" || name == "u4" || name == "uint32") {
            return mx::uint32;
        }
        if (name == "L" || name == "Q" || name == "u8"
                || name == "uint64") {
            return mx::uint64;
        }
        if (name == "e" || name == "f2" || name == "float16") {
            return mx::float16;
        }
        if (name == "f" || name == "f4" || name == "float32") {
            return mx::float32;
        }
        if (name == "d" || name == "f8" || name == "float"
                || name == "float64" || name == "double"
                || name == "g" || name == "longdouble"
                || name == "float128") {
            return mx::float64;
        }
        throw std::invalid_argument("unsupported dtype string");
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
        if (target_dtype == mx::int64) {
            auto primitive = std::make_shared<PreciseFloat64GpuAstypeInt64>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(array.shape(), mx::int64, std::move(primitive),
                          {std::move(array)}),
                stream);
        }
        if (target_dtype == mx::uint8) {
            auto primitive = std::make_shared<PreciseFloat64GpuAstypeUint8>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(array.shape(), mx::uint8, std::move(primitive),
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
    if (target_dtype == mx::float64 && targets_gpu(stream)) {
        return MlxPreciseArray(
            pack_existing_array_for_gpu(std::move(array), stream),
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

bool is_structured_dtype(nb::handle dtype)
{
    if (dtype.is_none() || !PySequence_Check(dtype.ptr())
            || PyUnicode_Check(dtype.ptr()) || PyBytes_Check(dtype.ptr())) {
        return false;
    }
    auto count = nb::len(dtype);
    if (count == 0) {
        return false;
    }
    for (nb::handle field : dtype) {
        if (!PySequence_Check(field.ptr())
                || PyUnicode_Check(field.ptr())
                || nb::len(field) < 2
                || !PyUnicode_Check(field[0].ptr())) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> structured_dtype_names(nb::handle dtype)
{
    std::vector<std::string> names;
    names.reserve(nb::len(dtype));
    for (nb::handle field : dtype) {
        names.push_back(nb::cast<std::string>(field[0]));
    }
    return names;
}

MlxPreciseArray structured_sequence_to_array(
    nb::handle value,
    nb::handle dtype,
    const mx::StreamOrDevice& stream)
{
    auto names = structured_dtype_names(dtype);
    std::vector<double> data;
    std::size_t rows = 0;
    for (nb::handle row_handle : value) {
        if (!PySequence_Check(row_handle.ptr())
                || PyUnicode_Check(row_handle.ptr())
                || nb::len(row_handle) != names.size()) {
            throw nb::type_error("structured array rows must match dtype fields");
        }
        for (std::size_t i = 0; i < names.size(); ++i) {
            double scalar = PyFloat_AsDouble(row_handle[i].ptr());
            if (PyErr_Occurred()) {
                throw nb::python_error();
            }
            data.push_back(scalar);
        }
        ++rows;
    }
    std::vector<mx::ShapeElem> shape{
        static_cast<mx::ShapeElem>(rows),
        static_cast<mx::ShapeElem>(names.size())};
    MlxPreciseArray array = targets_gpu(stream)
        ? MlxPreciseArray::transfer_float64_to_gpu(
              MlxPreciseArray::from_float64_data(data, shape, stream), stream)
        : MlxPreciseArray(
              place_on_stream(
                  mx::array(data.begin(),
                            mx::Shape(shape.begin(), shape.end()),
                            mx::float64),
                  stream),
              stream);
    array.set_structured_names(std::move(names));
    return array;
}

MlxPreciseArray MlxPreciseArray::make(nb::handle value,
                                      nb::object dtype,
                                      const mx::StreamOrDevice& stream)
{
    nb::object actual = nb::borrow<nb::object>(value);
    if (actual.is_none()) {
        actual = nb::cast(std::vector<double>{});
    }
    if (is_structured_dtype(dtype)) {
        return structured_sequence_to_array(actual, dtype, stream);
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
        return scalar_array;
    }
    if (target_dtype == mx::float64
            && !nb::isinstance<mx::array>(actual)
            && !has_mlx_array_protocol(actual)) {
        std::vector<double> data;
        std::vector<mx::ShapeElem> shape;
        collect_float64_sequence(actual, data, shape, 0);
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
    return MlxPreciseArray(std::move(array), stream);
}

void eval_precise_array(nb::handle value)
{
    auto actual = nb::borrow<nb::object>(value);
    if (nb::isinstance<MlxPreciseArray>(actual)) {
        auto& target = nb::cast<MlxPreciseArray&>(actual);
        auto evaluated = MlxPreciseArray::make(
            value, nb::none(), target.stream_or_device());
        MlxPreciseArray::eval(evaluated);
        target.overwrite_descriptor(evaluated);
        return;
    }
    auto array = nb::cast<mx::array>(actual);
    MlxPreciseArray::eval(array);
}

nb::bytes array_bytes(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    if (!array.flags().row_contiguous || has_explicit_stream(actual_stream)) {
        array = MlxPreciseArray(
            mx::contiguous(array, false, actual_stream), actual_stream);
    }
    {
        nb::gil_scoped_release release;
        MlxPreciseArray::eval(array);
        if (has_explicit_stream(actual_stream)) {
            mx::synchronize(mx::to_stream(actual_stream));
        }
    }
    return nb::bytes(
        static_cast<const char*>(array.data<void>()),
        static_cast<size_t>(array.nbytes()));
}

mx::array as_float64_array(nb::handle value, const mx::StreamOrDevice& stream)
{
    if (nb::isinstance<MlxPreciseArray>(value)) {
        auto array = nb::cast<MlxPreciseArray>(value);
        if (array.dtype() != mx::float64) {
            return mx::astype(array, mx::float64, stream);
        }
        return place_float64_on_explicit_stream(std::move(array), stream);
    }
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
        other, dtype_for_operand(other, self.dtype()), stream);
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
    case PreciseFloat64BinaryOp::Remainder:
        return MlxPreciseArray(mx::remainder(left, right, stream), stream);
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

MlxPreciseArray MlxPreciseArray::remainder(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Remainder, false);
}

MlxPreciseArray MlxPreciseArray::reverse_remainder(nb::handle other) const
{
    return binary_precise_array(
        *this, other, PreciseFloat64BinaryOp::Remainder, true);
}

MlxPreciseArray matmul_precise_array(const MlxPreciseArray& self,
                                     nb::handle other,
                                     bool reverse)
{
    auto stream = self.stream_or_device();
    auto self_array = MlxPreciseArray(
        static_cast<const mx::array&>(self), stream);
    auto other_array = MlxPreciseArray::make(
        other, dtype_for_operand(other, self.dtype()), stream);
    auto left = reverse ? other_array : self_array;
    auto right = reverse ? self_array : other_array;
    bool precise = left.dtype() == mx::float64 || right.dtype() == mx::float64;

    if (precise) {
        left = ensure_precise_float64(std::move(left), stream);
        right = ensure_precise_float64(std::move(right), stream);
        if (targets_gpu(stream)) {
            auto output_shape = matmul_output_shape(left, right);
            auto primitive = std::make_shared<PreciseFloat64GpuMatmul>(
                mx::to_stream(stream));
            return MlxPreciseArray(
                mx::array(std::move(output_shape), mx::float64,
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
    return MlxPreciseArray::make(left, nb::none(), stream).add(right);
}

MlxPreciseArray subtract_precise(nb::handle left,
                                 nb::handle right,
                                 const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).subtract(right);
}

MlxPreciseArray multiply_precise(nb::handle left,
                                 nb::handle right,
                                 const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).multiply(right);
}

MlxPreciseArray divide_precise(nb::handle left,
                               nb::handle right,
                               const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).divide(right);
}

MlxPreciseArray power_precise(nb::handle left,
                              nb::handle right,
                              const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).power(right);
}

MlxPreciseArray remainder_precise(nb::handle left,
                                  nb::handle right,
                                  const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).remainder(right);
}

MlxPreciseArray matmul_precise(nb::handle left,
                               nb::handle right,
                               const mx::StreamOrDevice& stream)
{
    return MlxPreciseArray::make(left, nb::none(), stream).matmul(right);
}

MlxPreciseArray compare_precise(nb::handle left_value,
                                nb::handle right_value,
                                PreciseFloat64CompareOp op,
                                const mx::StreamOrDevice& stream)
{
    auto left = MlxPreciseArray::make(left_value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = MlxPreciseArray::make(
        right_value, dtype_for_operand(right_value, left.dtype()), actual_stream);
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

MlxPreciseArray truth_zero_for_dtype(mx::Dtype dtype,
                                     const mx::StreamOrDevice& stream)
{
    if (dtype == mx::float64 && targets_gpu(stream)) {
        return MlxPreciseArray(float64_scalar(0.0, stream), stream);
    }
    return MlxPreciseArray(mx::zeros(mx::Shape{}, dtype, stream), stream);
}

MlxPreciseArray truth_array_precise(nb::handle value,
                                    const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    if (array.dtype() == mx::bool_) {
        return array;
    }
    auto zero = truth_zero_for_dtype(array.dtype(), actual_stream);
    return compare_precise(
        nb::cast(array),
        nb::cast(zero),
        PreciseFloat64CompareOp::NotEqual,
        actual_stream);
}

MlxPreciseArray logical_not_precise(nb::handle value,
                                    const mx::StreamOrDevice& stream)
{
    auto truth = truth_array_precise(value, stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : truth.stream_or_device();
    return MlxPreciseArray(mx::logical_not(truth, actual_stream),
                           actual_stream);
}

MlxPreciseArray logical_and_precise(nb::handle left_value,
                                    nb::handle right_value,
                                    const mx::StreamOrDevice& stream)
{
    auto left = truth_array_precise(left_value, stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = truth_array_precise(right_value, actual_stream);
    return MlxPreciseArray(mx::logical_and(left, right, actual_stream),
                           actual_stream);
}

MlxPreciseArray logical_or_precise(nb::handle left_value,
                                   nb::handle right_value,
                                   const mx::StreamOrDevice& stream)
{
    auto left = truth_array_precise(left_value, stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = truth_array_precise(right_value, actual_stream);
    return MlxPreciseArray(mx::logical_or(left, right, actual_stream),
                           actual_stream);
}

bool bool_precise(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.size() != 1) {
        throw nb::value_error(
            "The truth value of an array with more than one element is ambiguous");
    }
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    auto truth = truth_array_precise(nb::cast(array), actual_stream);
    MlxPreciseArray::eval(truth);
    return truth.item<bool>();
}

MlxPreciseArray arctan2_precise(nb::handle left_value,
                                nb::handle right_value,
                                const mx::StreamOrDevice& stream)
{
    auto left = MlxPreciseArray::make(left_value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : left.stream_or_device();
    auto right = MlxPreciseArray::make(
        right_value, dtype_for_operand(right_value, left.dtype()), actual_stream);
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
    auto condition = MlxPreciseArray::make(condition_value, nb::none(), stream);
    auto x = MlxPreciseArray::make(x_value, nb::none(), stream);
    auto y = MlxPreciseArray::make(
        y_value, dtype_for_operand(y_value, x.dtype()), stream);
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
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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
            MlxPreciseArray::make(item, nb::none(), stream));
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
        auto row = MlxPreciseArray::make(item, nb::none(), actual_stream);
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
        mx::to_stream(stream), start, out_shape, strides);
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
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    return slice_precise_array(
        std::move(array), requested_shape(start_value),
        requested_shape(stop_value), requested_shape(strides_value),
        actual_stream);
}

mx::Shape slice_update_shape(const mx::array& target,
                             mx::Shape start,
                             mx::Shape stop,
                             mx::Shape strides)
{
    if (start.size() != target.ndim() || stop.size() != target.ndim() ||
            strides.size() != target.ndim()) {
        throw std::invalid_argument(
            "[slice_update] Invalid number of indices or strides");
    }
    return normalize_precise_slice(target.shape(), start, std::move(stop),
                                   strides).second;
}

MlxPreciseArray slice_update_precise_array(MlxPreciseArray target,
                                           MlxPreciseArray update,
                                           mx::Shape start,
                                           mx::Shape stop,
                                           mx::Shape strides,
                                           const mx::StreamOrDevice& stream)
{
    auto slice_shape = slice_update_shape(target, start, stop, strides);
    if (update.shape() != slice_shape) {
        throw std::invalid_argument(
            "[slice_update] update shape must match the indexed slice shape");
    }
    if (target.dtype() != mx::float64 || !targets_gpu(stream)) {
        return MlxPreciseArray(
            mx::slice_update(target, update, std::move(start),
                             std::move(stop), std::move(strides), stream),
            stream);
    }

    target = ensure_precise_float64(std::move(target), stream);
    update = ensure_precise_float64(std::move(update), stream);
    auto primitive = std::make_shared<PreciseFloat64GpuSliceUpdate>(
        mx::to_stream(stream), start, stop, strides);
    return MlxPreciseArray(
        mx::array(target.shape(), mx::float64, std::move(primitive),
                  {std::move(target), std::move(update)}),
        stream);
}

void setitem_precise(MlxPreciseArray& target,
                     nb::handle update_value,
                     nb::handle start_value,
                     nb::handle stop_value,
                     nb::handle strides_value,
                     const mx::StreamOrDevice& stream)
{
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : target.stream_or_device();
    auto update = MlxPreciseArray::make(
        update_value, nb::cast(target.dtype()), actual_stream);
    auto updated = slice_update_precise_array(
        MlxPreciseArray(static_cast<const mx::array&>(target), actual_stream),
        std::move(update),
        requested_shape(start_value),
        requested_shape(stop_value),
        requested_shape(strides_value),
        actual_stream);
    target.overwrite_descriptor(updated);
}

MlxPreciseArray affine_transform_precise(nb::handle vertices_value,
                                         nb::handle matrix_value,
                                         const mx::StreamOrDevice& stream)
{
    auto vertices = MlxPreciseArray::make(vertices_value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : vertices.stream_or_device();
    auto matrix = MlxPreciseArray::make(matrix_value, nb::none(), actual_stream);

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
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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
    auto target_dtype = requested_dtype(dtype, mx::float32);
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    return astype_mlx_precise_array(std::move(array), target_dtype, stream);
}

MlxPreciseArray bincount_int32(nb::handle value,
                               int minlength,
                               const mx::StreamOrDevice& stream)
{
    if (minlength < 0) {
        throw std::invalid_argument("[bincount] minlength must be non-negative");
    }
    auto array = MlxPreciseArray::make(value, nb::cast(mx::int32), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    array = astype_mlx_precise_array(std::move(array), mx::int32, actual_stream);
    auto primitive = std::make_shared<BincountInt32>(
        mx::to_stream(actual_stream), minlength);
    return MlxPreciseArray(
        mx::array(mx::Shape{minlength}, mx::int32, std::move(primitive),
                  {std::move(array)}),
        actual_stream);
}

MlxPreciseArray reshape_precise(nb::handle value,
                                nb::handle shape_value,
                                const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    auto shape = requested_shape(shape_value);
    auto output_shape = precise_reshape_output_shape(array, shape);
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
                               bool reverse,
                               bool inclusive,
                               const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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
        if (!flatten && (axis < 0 || axis >= array.ndim())) {
            throw std::invalid_argument("[cumsum] axis out of bounds");
        }
        auto output_shape = flatten
            ? mx::Shape{static_cast<mx::ShapeElem>(array.size())}
            : array.shape();
        auto primitive = std::make_shared<PreciseFloat64GpuCumsum>(
            mx::to_stream(actual_stream), axis, flatten, reverse, inclusive);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64, std::move(primitive),
                      {std::move(array)}),
            actual_stream);
    }
    if (flatten) {
        return MlxPreciseArray(
            mx::cumsum(mx::reshape(array, {-1}, actual_stream),
                       0, reverse, inclusive, actual_stream),
            actual_stream);
    }
    return MlxPreciseArray(
        mx::cumsum(array, axis, reverse, inclusive, actual_stream),
        actual_stream);
}

MlxPreciseArray transpose_precise(nb::handle value,
                                  nb::handle axes_value,
                                  const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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

MlxPreciseArray take_precise(nb::handle value,
                             nb::handle indices_value,
                             nb::object axis_value,
                             const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : array.stream_or_device();
    auto indices = MlxPreciseArray::make(
        indices_value, nb::cast(mx::int32), actual_stream);
    bool flatten_input = axis_value.is_none();
    int axis = 0;
    if (!flatten_input) {
        axis = nb::cast<int>(axis_value);
        if (axis < 0) {
            axis += array.ndim();
        }
    }

    if (array.dtype() == mx::float64 && targets_gpu(actual_stream)) {
        array = ensure_precise_float64(std::move(array), actual_stream);
        if (indices.dtype() != mx::int32) {
            indices = MlxPreciseArray(
                mx::astype(indices, mx::int32, actual_stream), actual_stream);
        }
        auto output_shape = take_output_shape(
            array, indices, axis, flatten_input);
        auto primitive = std::make_shared<PreciseFloat64GpuTake>(
            mx::to_stream(actual_stream), axis, flatten_input);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64,
                      std::move(primitive),
                      {std::move(array), std::move(indices)}),
            actual_stream);
    }

    if (flatten_input) {
        return MlxPreciseArray(
            mx::take(array, indices, actual_stream), actual_stream);
    }
    return MlxPreciseArray(
        mx::take(array, indices, axis, actual_stream), actual_stream);
}

mx::Shape conv1d_output_shape(const MlxPreciseArray& input,
                              const MlxPreciseArray& weight,
                              int stride,
                              int padding,
                              int dilation,
                              int groups)
{
    if (input.ndim() != 3) {
        throw std::invalid_argument("[conv1d] input must be rank 3");
    }
    if (weight.ndim() != 3) {
        throw std::invalid_argument("[conv1d] weight must be rank 3");
    }
    if (stride <= 0 || dilation <= 0) {
        throw std::invalid_argument(
            "[conv1d] stride and dilation must be positive");
    }
    if (padding < 0) {
        throw std::invalid_argument("[conv1d] padding must be non-negative");
    }
    if (groups <= 0) {
        throw std::invalid_argument("[conv1d] groups must be positive");
    }

    auto input_channels = input.shape(2);
    auto output_channels = weight.shape(0);
    auto kernel_size = weight.shape(1);
    auto weight_channels = weight.shape(2);
    if (input_channels != weight_channels * groups) {
        throw std::invalid_argument(
            "[conv1d] input channels must match weight channels times groups");
    }
    if (output_channels % groups != 0) {
        throw std::invalid_argument(
            "[conv1d] output channels must be divisible by groups");
    }

    auto numerator = input.shape(1) + 2 * padding
        - dilation * (kernel_size - 1) - 1;
    auto output_length = numerator < 0 ? 0 : numerator / stride + 1;
    return {input.shape(0), output_length, output_channels};
}

MlxPreciseArray conv1d_precise(nb::handle input_value,
                               nb::handle weight_value,
                               int stride,
                               int padding,
                               int dilation,
                               int groups,
                               const mx::StreamOrDevice& stream)
{
    auto input = MlxPreciseArray::make(input_value, nb::none(), stream);
    auto actual_stream = has_explicit_stream(stream)
        ? stream
        : input.stream_or_device();
    auto weight = MlxPreciseArray::make(weight_value, nb::none(), actual_stream);
    bool wants_float64 =
        input.dtype() == mx::float64 || weight.dtype() == mx::float64;
    auto output_shape = conv1d_output_shape(
        input, weight, stride, padding, dilation, groups);

    if (wants_float64) {
        input = ensure_precise_float64(std::move(input), actual_stream);
        weight = ensure_precise_float64(std::move(weight), actual_stream);
        if (targets_gpu(actual_stream)) {
            auto primitive = std::make_shared<PreciseFloat64GpuConv1d>(
                mx::to_stream(actual_stream), output_shape,
                stride, padding, dilation, groups);
            return MlxPreciseArray(
                mx::array(std::move(output_shape), mx::float64,
                          std::move(primitive),
                          {std::move(input), std::move(weight)}),
                actual_stream);
        }
    }

    return MlxPreciseArray(
        mx::conv1d(input, weight, stride, padding, dilation, groups,
                   actual_stream),
        actual_stream);
}

nb::object item_precise(nb::handle value, const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.size() != 1) {
        throw nb::value_error("can only convert a size-1 array to scalar");
    }
    if (array.dtype() != mx::float64) {
        throw nb::type_error("item_precise only handles float64 arrays");
    }
    auto values = read_precise_float64_values(std::move(array));
    return nb::float_(values[0]);
}

bool is_sorted_and_has_non_nan_precise(nb::handle value,
                                       const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::cast(mx::float64), stream);
    if (array.ndim() != 1) {
        throw std::invalid_argument("array must be 1D");
    }
    MlxPreciseArray::eval(array);
    bool has_value = false;
    double previous = 0.0;
    if (targets_gpu(array.stream_or_device())) {
        auto ptr = array.data<PreciseFloat64>();
        for (std::size_t i = 0; i < array.size(); ++i) {
            auto offset = array.flags().row_contiguous
                ? i
                : row_major_offset(i, array.shape(), array.strides());
            auto current = ptr[offset].value();
            if (std::isnan(current)) {
                continue;
            }
            if (has_value && current < previous) {
                return false;
            }
            previous = current;
            has_value = true;
        }
    } else {
        auto ptr = array.data<double>();
        for (std::size_t i = 0; i < array.size(); ++i) {
            auto offset = array.flags().row_contiguous
                ? i
                : row_major_offset(i, array.shape(), array.strides());
            auto current = ptr[offset];
            if (std::isnan(current)) {
                continue;
            }
            if (has_value && current < previous) {
                return false;
            }
            previous = current;
            has_value = true;
        }
    }
    return has_value;
}

MlxPreciseArray sort_float64(nb::handle value,
                             nb::object axis,
                             const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        bool flatten_input = axis.is_none();
        int actual_axis = 0;
        if (!axis.is_none()) {
            actual_axis = nb::cast<int>(axis);
            if (actual_axis < 0) {
                actual_axis += array.ndim();
            }
            if (actual_axis < 0 || actual_axis >= array.ndim()) {
                throw std::invalid_argument("[sort] axis out of bounds");
            }
        }
        array = ensure_precise_float64(std::move(array), stream);
        auto output_shape = flatten_input
            ? mx::Shape{static_cast<mx::ShapeElem>(array.size())}
            : array.shape();
        auto primitive = std::make_shared<PreciseFloat64GpuSort>(
            mx::to_stream(stream), actual_axis, flatten_input);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::float64, std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    if (axis.is_none()) {
        return MlxPreciseArray(mx::sort(array, stream), stream);
    }
    return MlxPreciseArray(
        mx::sort(array, nb::cast<int>(axis), stream), stream);
}

MlxPreciseArray argsort_float64(nb::handle value,
                                nb::object axis,
                                const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        bool flatten_input = axis.is_none();
        int actual_axis = 0;
        if (!axis.is_none()) {
            actual_axis = nb::cast<int>(axis);
            if (actual_axis < 0) {
                actual_axis += array.ndim();
            }
            if (actual_axis < 0 || actual_axis >= array.ndim()) {
                throw std::invalid_argument("[argsort] axis out of bounds");
            }
        }
        array = ensure_precise_float64(std::move(array), stream);
        auto output_shape = flatten_input
            ? mx::Shape{static_cast<mx::ShapeElem>(array.size())}
            : array.shape();
        auto primitive = std::make_shared<PreciseFloat64GpuArgsort>(
            mx::to_stream(stream), actual_axis, flatten_input);
        return MlxPreciseArray(
            mx::array(std::move(output_shape), mx::uint32, std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    if (axis.is_none()) {
        return MlxPreciseArray(mx::argsort(array, stream), stream);
    }
    return MlxPreciseArray(
        mx::argsort(array, nb::cast<int>(axis), stream), stream);
}

std::vector<int> normalize_reduce_axes(nb::object axis, int ndim)
{
    std::vector<int> axes;
    if (axis.is_none()) {
        axes.reserve(static_cast<std::size_t>(ndim));
        for (int i = 0; i < ndim; ++i) {
            axes.push_back(i);
        }
        return axes;
    }
    if (nb::isinstance<nb::tuple>(axis) || nb::isinstance<nb::list>(axis)) {
        for (nb::handle item : axis) {
            axes.push_back(nb::cast<int>(item));
        }
    } else {
        axes.push_back(nb::cast<int>(axis));
    }
    for (auto& ax : axes) {
        if (ax < 0) {
            ax += ndim;
        }
        if (ax < 0 || ax >= ndim) {
            throw std::invalid_argument("[reduction] axis out of bounds");
        }
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

std::size_t reduction_element_count(const mx::array& array,
                                    const std::vector<int>& axes)
{
    std::size_t count = 1;
    for (auto axis : axes) {
        count *= static_cast<std::size_t>(array.shape(axis));
    }
    return count;
}

MlxPreciseArray sum_float64(nb::handle value,
                            nb::object axis,
                            bool keepdims,
                            const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        auto axes = normalize_reduce_axes(axis, array.ndim());
        array = ensure_precise_float64(std::move(array), stream);
        auto primitive = std::make_shared<PreciseFloat64GpuReduceSum>(
            mx::to_stream(stream), std::move(axes), keepdims);
        return MlxPreciseArray(
            mx::array(primitive->output_shapes({array})[0], mx::float64,
                      std::move(primitive),
                      {std::move(array)}),
            stream);
    }
    if (axis.is_none()) {
        return MlxPreciseArray(mx::sum(array, keepdims, stream), stream);
    }
    auto axes = normalize_reduce_axes(axis, array.ndim());
    return MlxPreciseArray(mx::sum(array, axes, keepdims, stream), stream);
}

MlxPreciseArray mean_float64(nb::handle value,
                             nb::object axis,
                             bool keepdims,
                             const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (array.dtype() == mx::float64 && targets_gpu(stream)) {
        auto axes = normalize_reduce_axes(axis, array.ndim());
        auto count = reduction_element_count(array, axes);
        auto total = sum_float64(nb::cast(array), axis, keepdims, stream);
        return total.divide(nb::cast(
            float64_scalar(static_cast<double>(count), stream)));
    }
    if (axis.is_none()) {
        return MlxPreciseArray(mx::mean(array, keepdims, stream), stream);
    }
    auto axes = normalize_reduce_axes(axis, array.ndim());
    return MlxPreciseArray(mx::mean(array, axes, keepdims, stream), stream);
}

MlxPreciseArray reduce_minmax_precise(nb::handle value,
                                      nb::object axis,
                                      bool is_max,
                                      const mx::StreamOrDevice& stream)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
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
    auto left_array = MlxPreciseArray::make(left, nb::none(), stream);
    auto right_array = MlxPreciseArray::make(right, nb::none(), stream);
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

mx::array unary_math_float64(nb::handle value,
                             const mx::StreamOrDevice& stream,
                             PreciseFloat64UnaryMathOp op)
{
    auto array = MlxPreciseArray::make(value, nb::none(), stream);
    if (targets_gpu(stream)) {
        if (array.dtype() != mx::float64) {
            array = astype_mlx_precise_array(
                std::move(array), mx::float64, stream);
        }
        array = ensure_precise_float64(std::move(array), stream);
        auto primitive = std::make_shared<PreciseFloat64GpuUnaryMath>(
            mx::to_stream(stream), op);
        return mx::array(array.shape(), mx::float64, std::move(primitive),
                         {std::move(array)});
    }
    switch (op) {
    case PreciseFloat64UnaryMathOp::Log:
        return mx::log(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Log2:
        return mx::log2(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Log10:
        return mx::log10(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Floor:
        return mx::floor(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Ceil:
        return mx::ceil(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Sqrt:
        return mx::sqrt(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Degrees:
        return mx::degrees(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Radians:
        return mx::radians(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Sin:
        return mx::sin(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::Cos:
        return mx::cos(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::ArcSin:
        return mx::arcsin(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::ArcCos:
        return mx::arccos(as_float64_array(value, stream), stream);
    case PreciseFloat64UnaryMathOp::ArcTan:
        return mx::arctan(as_float64_array(value, stream), stream);
    }
}

mx::array log_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Log);
}

mx::array log2_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Log2);
}

mx::array log10_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Log10);
}

mx::array floor_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Floor);
}

mx::array ceil_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Ceil);
}

mx::array sqrt_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Sqrt);
}

mx::array degrees_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Degrees);
}

mx::array radians_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Radians);
}

mx::array sin_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Sin);
}

mx::array cos_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::Cos);
}

mx::array arcsin_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::ArcSin);
}

mx::array arccos_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::ArcCos);
}

mx::array arctan_float64(nb::handle value, const mx::StreamOrDevice& stream)
{
    return unary_math_float64(value, stream, PreciseFloat64UnaryMathOp::ArcTan);
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
#if MLX_VERSION_NUMERIC >= 31000
        return mx::fft::fft(input, n, axis, mx::fft::FFTNorm::Backward, stream);
#else
        return mx::fft::fft(input, n, axis, stream);
#endif
    }
#if MLX_VERSION_NUMERIC >= 31000
    return mx::fft::fft(input, axis, mx::fft::FFTNorm::Backward, stream);
#else
    return mx::fft::fft(input, axis, stream);
#endif
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
	               const mx::StreamOrDevice& stream) {
	                new (self) MlxPreciseArray(
	                    MlxPreciseArray::make(value, dtype, stream));
	            },
	            "value"_a = nb::none(),
	            "dtype"_a = nb::none(),
	            "stream"_a = nb::none())
        .def_prop_ro("T",
             [](const MlxPreciseArray& self) {
                 auto self_obj = nb::cast(self);
                 return transpose_precise(
                     self_obj, nb::none(), self.stream_or_device());
             })
        .def_prop_ro("dtype",
             [](const MlxPreciseArray& self) -> nb::object {
                 if (!self.has_structured_fields()) {
                     return nb::cast(static_cast<const mx::array&>(self).dtype());
                 }
                 nb::tuple names = nb::steal<nb::tuple>(
                     PyTuple_New(self.structured_names().size()));
                 nb::dict fields;
                 for (std::size_t i = 0; i < self.structured_names().size(); ++i) {
                     const auto& name = self.structured_names()[i];
                     PyTuple_SET_ITEM(
                         names.ptr(), i, nb::str(name.c_str()).release().ptr());
                     fields[nb::str(name.c_str())] = nb::make_tuple(mx::float64, i);
                 }
                 auto types = nb::module_::import_("types");
                 return types.attr("SimpleNamespace")(
                     "names"_a = names,
                     "fields"_a = fields);
             })
        .def_prop_ro("strides",
             [](const MlxPreciseArray& self) {
                 std::vector<std::int64_t> strides;
                 strides.reserve(self.ndim());
                 for (auto stride : self.strides()) {
                     strides.push_back(stride);
                 }
                 return strides;
             })
        .def_prop_ro("_mlx_stream",
             [](const MlxPreciseArray& self) -> nb::object {
                 if (!has_explicit_stream(self.stream_or_device())) {
                     return nb::none();
                 }
                 return nb::cast(self.stream_or_device());
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
                nb::args args,
                nb::kwargs kwargs) {
                 if (args.size() == 0) {
                     throw std::invalid_argument(
                         "reshape requires at least one shape argument");
                 }
                 mx::StreamOrDevice stream = {};
                 std::size_t known_kwargs = 0;
                 if (kwargs.ptr() != nullptr && kwargs.contains("stream")) {
                     stream = nb::cast<mx::StreamOrDevice>(kwargs["stream"]);
                     known_kwargs = 1;
                 }
                 if (kwargs.ptr() != nullptr && kwargs.size() != known_kwargs) {
                     throw std::invalid_argument(
                         "reshape only accepts the stream keyword");
                 }
                 nb::object shape = nb::none();
                 if (args.size() == 1) {
                     shape = nb::borrow<nb::object>(args[0]);
                 } else {
                     nb::list shape_values;
                     for (auto item : args) {
                         shape_values.append(nb::cast<long long>(item));
                     }
                     shape = std::move(shape_values);
                 }
                 auto self_obj = nb::cast(self);
                 return reshape_precise(self_obj, shape, stream);
             })
        .def("flatten",
             [](const MlxPreciseArray& self,
                int start_axis,
                int end_axis,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 auto output_shape = flatten_output_shape(
                     self.shape(), start_axis, end_axis);
                 nb::list shape;
                 for (auto dim : output_shape) {
                     shape.append(dim);
                 }
                 return reshape_precise(
                     self_obj,
                     shape,
                     stream);
             },
             "start_axis"_a = 0,
             "end_axis"_a = -1,
             nb::kw_only(),
             "stream"_a = nb::none())
        .def("ravel",
             [](const MlxPreciseArray& self,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return reshape_precise(
                     self_obj,
                     nb::make_tuple(static_cast<mx::ShapeElem>(self.size())),
                     stream);
             },
             nb::kw_only(),
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
                bool reverse,
                bool inclusive,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return cumsum_precise(self_obj, axis, reverse, inclusive, stream);
             },
             "axis"_a = nb::none(),
             nb::kw_only(),
             "reverse"_a = false,
             "inclusive"_a = true,
             "stream"_a = nb::none())
        .def("sum",
             [](const MlxPreciseArray& self,
                nb::object axis,
                bool keepdims,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return sum_float64(self_obj, axis, keepdims, stream);
             },
             "axis"_a = nb::none(),
             "keepdims"_a = false,
             nb::kw_only(),
             "stream"_a = nb::none())
        .def("mean",
             [](const MlxPreciseArray& self,
                nb::object axis,
                bool keepdims,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return mean_float64(self_obj, axis, keepdims, stream);
             },
             "axis"_a = nb::none(),
             "keepdims"_a = false,
             nb::kw_only(),
             "stream"_a = nb::none())
        .def("sqrt",
             [](const MlxPreciseArray& self,
                const mx::StreamOrDevice& stream) {
                 auto self_obj = nb::cast(self);
                 return sqrt_float64(self_obj, stream);
             },
             nb::kw_only(),
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
        .def("__mod__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.remainder(other);
             },
             "other"_a)
        .def("__rmod__",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.reverse_remainder(other);
             },
             "other"_a)
        .def("__imod__",
             [](MlxPreciseArray& self, nb::handle other) -> MlxPreciseArray& {
                 self.overwrite_descriptor(self.remainder(other));
                 return self;
             },
             "other"_a,
             nb::rv_policy::none)
        .def("remainder",
             [](const MlxPreciseArray& self, nb::handle other) {
                 return self.remainder(other);
             },
             "other"_a)
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
        .def("__bool__",
             [](const MlxPreciseArray& self) {
                 auto self_obj = nb::cast(self);
                 return bool_precise(self_obj, self.stream_or_device());
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
    m.def("array_bytes", &array_bytes,
          "value"_a,
          "stream"_a = nb::none());
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
    m.def("remainder_precise", &remainder_precise,
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
    m.def("logical_not_precise", &logical_not_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("logical_and_precise", &logical_and_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("logical_or_precise", &logical_or_precise,
          "left"_a,
          "right"_a,
          "stream"_a = nb::none());
    m.def("bool_precise", &bool_precise,
          "value"_a,
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
    m.def("setitem_precise", &setitem_precise,
          "target"_a,
          "update"_a,
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
    m.def("bincount_int32", &bincount_int32,
          "value"_a,
          "minlength"_a,
          "stream"_a = nb::none());
    m.def("reshape_precise", &reshape_precise,
          "value"_a,
          "shape"_a,
          "stream"_a = nb::none());
    m.def("cumsum_precise", &cumsum_precise,
          "value"_a,
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none());
    m.def("transpose_precise", &transpose_precise,
          "value"_a,
          "axes"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("take_precise", &take_precise,
          "value"_a,
          "indices"_a,
          "axis"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("conv1d_precise", &conv1d_precise,
          "input"_a,
          "weight"_a,
          "stride"_a = 1,
          "padding"_a = 0,
          "dilation"_a = 1,
          "groups"_a = 1,
          "stream"_a = nb::none());
    m.def("item_precise", &item_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("tolist_precise", &tolist_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("is_sorted_and_has_non_nan_precise",
          &is_sorted_and_has_non_nan_precise,
          "value"_a,
          "stream"_a = nb::none());
    m.def("sort_float64", &sort_float64,
          "value"_a,
          "axis"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("argsort_float64", &argsort_float64,
          "value"_a,
          "axis"_a = nb::none(),
          "stream"_a = nb::none());
    m.def("sum_float64", &sum_float64,
          "value"_a,
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "stream"_a = nb::none());
    m.def("mean_float64", &mean_float64,
          "value"_a,
          "axis"_a = nb::none(),
          "keepdims"_a = false,
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
    m.def("floor_float64", &floor_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("ceil_float64", &ceil_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("sqrt_float64", &sqrt_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("degrees_float64", &degrees_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("radians_float64", &radians_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("sin_float64", &sin_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("cos_float64", &cos_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("arcsin_float64", &arcsin_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("arccos_float64", &arccos_float64,
          "value"_a,
          "stream"_a = nb::none());
    m.def("arctan_float64", &arctan_float64,
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
