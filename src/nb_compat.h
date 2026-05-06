/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_NB_COMPAT_H
#define MPL_NB_COMPAT_H

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace py = nanobind;

namespace nanobind {

template <typename T>
struct format_descriptor;

class memoryview : public object {
    NB_OBJECT_DEFAULT(memoryview, object, "memoryview", PyMemoryView_Check)

    template <typename T>
    static memoryview from_buffer(T *data,
                                  std::initializer_list<ssize_t> shape,
                                  std::initializer_list<ssize_t> strides,
                                  bool readonly = false)
    {
        std::vector<ssize_t> shape_vec(shape);
        std::vector<ssize_t> stride_vec(strides);
        Py_buffer view {};
        view.buf = data;
        view.obj = nullptr;
        view.len = sizeof(T);
        for (auto dim : shape_vec) {
            view.len *= dim;
        }
        view.readonly = readonly;
        view.itemsize = sizeof(T);
        view.format = const_cast<char *>(format_descriptor<std::remove_cv_t<T>>::format());
        view.ndim = static_cast<int>(shape_vec.size());
        view.shape = shape_vec.data();
        view.strides = stride_vec.data();
        view.suboffsets = nullptr;
        view.internal = nullptr;
        auto *mv = PyMemoryView_FromBuffer(&view);
        if (!mv) {
            raise_python_error();
        }
        return steal<memoryview>(mv);
    }
};

template <typename T>
T reinterpret_borrow(handle h)
{
    return borrow<T>(h);
}

template <typename T>
T reinterpret_borrow(PyObject *h)
{
    return borrow<T>(handle(h));
}

template <typename T>
T reinterpret_steal(PyObject *h)
{
    return steal<T>(h);
}

struct buffer_info {
    void *ptr = nullptr;
    ssize_t itemsize = 0;
    ssize_t size = 0;
    std::string format;
    ssize_t ndim = 0;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
};

class buffer : public object {
  public:
    NB_OBJECT_DEFAULT(buffer, object, "typing.Buffer", PyObject_CheckBuffer)

    buffer_info request(bool writable = false) const
    {
        Py_buffer view {};
        int flags = PyBUF_FORMAT | PyBUF_STRIDES;
        if (writable) {
            flags |= PyBUF_WRITABLE;
        }
        if (PyObject_GetBuffer(m_ptr, &view, flags) != 0) {
            raise_python_error();
        }

        buffer_info info;
        info.ptr = view.buf;
        info.itemsize = view.itemsize;
        info.size = view.len / (view.itemsize == 0 ? 1 : view.itemsize);
        info.format = view.format ? view.format : "";
        info.ndim = view.ndim;
        info.shape.assign(view.shape, view.shape + view.ndim);
        if (view.strides) {
            info.strides.assign(view.strides, view.strides + view.ndim);
        } else {
            info.strides.resize(static_cast<std::size_t>(view.ndim));
            ssize_t stride = view.itemsize;
            for (ssize_t i = view.ndim - 1; i >= 0; --i) {
                info.strides[static_cast<std::size_t>(i)] = stride;
                stride *= view.shape[i];
            }
        }

        PyBuffer_Release(&view);
        return info;
    }
};

template <>
struct format_descriptor<bool> {
    static const char *format() { return "?"; }
};

template <>
struct format_descriptor<double> {
    static const char *format() { return "d"; }
};

template <>
struct format_descriptor<float> {
    static const char *format() { return "f"; }
};

template <>
struct format_descriptor<std::uint8_t> {
    static const char *format() { return "B"; }
};

template <>
struct format_descriptor<std::int8_t> {
    static const char *format() { return "b"; }
};

template <>
struct format_descriptor<std::uint16_t> {
    static const char *format() { return "H"; }
};

template <>
struct format_descriptor<std::int16_t> {
    static const char *format() { return "h"; }
};

template <>
struct format_descriptor<std::int32_t> {
    static const char *format() { return "i"; }
};

template <>
struct format_descriptor<std::int64_t> {
    static const char *format() { return "q"; }
};

}  // namespace nanobind

#endif  // MPL_NB_COMPAT_H
