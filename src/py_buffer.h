/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_BUFFER_H
#define MPL_PY_BUFFER_H

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "nb_compat.h"

namespace mpl {

template <typename T, py::ssize_t ND>
class BufferView {
public:
    BufferView() = default;

    explicit BufferView(const py::buffer &buf, bool writable = false)
    {
        auto info = buf.request(writable);
        if (info.ndim != ND) {
            auto message = "Expected buffer with " + std::to_string(ND)
                + " dimensions, got " + std::to_string(info.ndim);
            throw py::value_error(message.c_str());
        }
        if (info.itemsize != static_cast<py::ssize_t>(sizeof(T))) {
            throw py::value_error("Unexpected itemsize in buffer");
        }
        // Match exact PEP3118 format for T.
        if (info.format != py::format_descriptor<T>::format()) {
            throw py::value_error("Unexpected format in buffer");
        }

        base_ = static_cast<unsigned char *>(info.ptr);
        for (py::ssize_t i = 0; i < ND; i++) {
            shape_[i] = info.shape[i];
            if (info.strides[i] % static_cast<py::ssize_t>(sizeof(T)) != 0) {
                throw py::value_error("Strides are not aligned to element size");
            }
            strides_[i] = info.strides[i] / static_cast<py::ssize_t>(sizeof(T));
        }

        // Compute size once for convenience.
        size_ = 1;
        for (py::ssize_t i = 0; i < ND; i++) {
            size_ *= shape_[i];
        }
    }

    [[nodiscard]] py::ssize_t ndim() const { return ND; }
    [[nodiscard]] py::ssize_t size() const { return size_; }

    [[nodiscard]] py::ssize_t shape(py::ssize_t i) const { return shape_.at(i); }
    [[nodiscard]] py::ssize_t stride(py::ssize_t i) const { return strides_.at(i); }

    [[nodiscard]] T *data() { return reinterpret_cast<T *>(base_); }
    [[nodiscard]] const T *data() const { return reinterpret_cast<const T *>(base_); }

    // Pointer access at an index (strided). This is mainly used by the C++ Agg code
    // which expects a mutable pointer to the underlying image buffer.
    [[nodiscard]] T *mutable_data(py::ssize_t i)
    {
        static_assert(ND == 1, "mutable_data(i) only available for 1D views");
        return reinterpret_cast<T *>(base_ + i * strides_[0] * sizeof(T));
    }

    [[nodiscard]] T *mutable_data(py::ssize_t i, py::ssize_t j)
    {
        static_assert(ND == 2, "mutable_data(i, j) only available for 2D views");
        auto offset = (i * strides_[0] + j * strides_[1]) * sizeof(T);
        return reinterpret_cast<T *>(base_ + offset);
    }

    [[nodiscard]] T *mutable_data(py::ssize_t i, py::ssize_t j, py::ssize_t k)
    {
        static_assert(ND == 3, "mutable_data(i, j, k) only available for 3D views");
        auto offset = (i * strides_[0] + j * strides_[1] + k * strides_[2]) * sizeof(T);
        return reinterpret_cast<T *>(base_ + offset);
    }

    // Element access (strided). Only valid for the matching dimensionality.
    T &operator()(py::ssize_t i)
    {
        static_assert(ND == 1, "operator()(i) only available for 1D views");
        return *reinterpret_cast<T *>(base_ + i * strides_[0] * sizeof(T));
    }

    const T &operator()(py::ssize_t i) const
    {
        static_assert(ND == 1, "operator()(i) only available for 1D views");
        return *reinterpret_cast<const T *>(base_ + i * strides_[0] * sizeof(T));
    }

    // Some of the legacy Matplotlib C++ code uses result[i] instead of result(i).
    T &operator[](py::ssize_t i)
    {
        static_assert(ND == 1, "operator[](i) only available for 1D views");
        return (*this)(i);
    }

    const T &operator[](py::ssize_t i) const
    {
        static_assert(ND == 1, "operator[](i) only available for 1D views");
        return (*this)(i);
    }

    T &operator()(py::ssize_t i, py::ssize_t j)
    {
        static_assert(ND == 2, "operator()(i, j) only available for 2D views");
        auto offset = (i * strides_[0] + j * strides_[1]) * sizeof(T);
        return *reinterpret_cast<T *>(base_ + offset);
    }

    const T &operator()(py::ssize_t i, py::ssize_t j) const
    {
        static_assert(ND == 2, "operator()(i, j) only available for 2D views");
        auto offset = (i * strides_[0] + j * strides_[1]) * sizeof(T);
        return *reinterpret_cast<const T *>(base_ + offset);
    }

    T &operator()(py::ssize_t i, py::ssize_t j, py::ssize_t k)
    {
        static_assert(ND == 3, "operator()(i, j, k) only available for 3D views");
        auto offset = (i * strides_[0] + j * strides_[1] + k * strides_[2]) * sizeof(T);
        return *reinterpret_cast<T *>(base_ + offset);
    }

    const T &operator()(py::ssize_t i, py::ssize_t j, py::ssize_t k) const
    {
        static_assert(ND == 3, "operator()(i, j, k) only available for 3D views");
        auto offset = (i * strides_[0] + j * strides_[1] + k * strides_[2]) * sizeof(T);
        return *reinterpret_cast<const T *>(base_ + offset);
    }

private:
    unsigned char *base_ = nullptr;
    std::array<py::ssize_t, ND> shape_{};
    std::array<py::ssize_t, ND> strides_{};
    py::ssize_t size_ = 0;
};

template <typename T, py::ssize_t ND>
class MlxArrayView {
public:
    MlxArrayView() = default;

    explicit MlxArrayView(py::object obj, const char *name = "array")
        : owner_(std::move(obj))
    {
        if (owner_.is_none()) {
            return;
        }
        if (!py::hasattr(owner_, "shape") || !py::hasattr(owner_, "__getitem__")) {
            auto message = std::string(name) + " must be an MLX array";
            throw py::type_error(message.c_str());
        }

        auto shape = py::reinterpret_borrow<py::sequence>(owner_.attr("shape"));
        if (static_cast<py::ssize_t>(py::len(shape)) != ND) {
            auto message = "Expected MLX array with " + std::to_string(ND)
                + " dimensions, got " + std::to_string(py::len(shape));
            throw py::value_error(message.c_str());
        }
        size_ = 1;
        for (py::ssize_t i = 0; i < ND; ++i) {
            shape_[i] = py::cast<py::ssize_t>(shape[i]);
            size_ *= shape_[i];
        }
    }

    [[nodiscard]] py::ssize_t ndim() const { return ND; }
    [[nodiscard]] py::ssize_t size() const { return size_; }
    [[nodiscard]] py::ssize_t shape(py::ssize_t i) const { return shape_.at(i); }

    T operator()(py::ssize_t i) const
    {
        static_assert(ND == 1, "operator()(i) only available for 1D views");
        return scalar_at(py::int_(i));
    }

    T operator[](py::ssize_t i) const
    {
        static_assert(ND == 1, "operator[](i) only available for 1D views");
        return (*this)(i);
    }

    T operator()(py::ssize_t i, py::ssize_t j) const
    {
        static_assert(ND == 2, "operator()(i, j) only available for 2D views");
        return scalar_at(py::make_tuple(i, j));
    }

    T operator()(py::ssize_t i, py::ssize_t j, py::ssize_t k) const
    {
        static_assert(ND == 3, "operator()(i, j, k) only available for 3D views");
        return scalar_at(py::make_tuple(i, j, k));
    }

private:
    T scalar_at(py::handle key) const
    {
        PyObject *result = PyObject_GetItem(owner_.ptr(), key.ptr());
        if (result == nullptr) {
            py::raise_python_error();
        }
        py::object scalar = py::steal<py::object>(result);
        py::object value = py::hasattr(scalar, "item") ? scalar.attr("item")() : scalar;
        if constexpr (std::is_same_v<T, uint8_t>) {
            return static_cast<uint8_t>(
                static_cast<unsigned long long>(py::int_(value)));
        } else if constexpr (std::is_same_v<T, bool>) {
            int truth = PyObject_IsTrue(value.ptr());
            if (truth < 0) {
                py::raise_python_error();
            }
            return truth != 0;
        } else {
            return static_cast<T>(static_cast<double>(py::float_(value)));
        }
    }

    py::object owner_;
    std::array<py::ssize_t, ND> shape_{};
    py::ssize_t size_ = 0;
};

}  // namespace mpl

#endif  // MPL_PY_BUFFER_H
