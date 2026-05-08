/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <optional>
#include <stdexcept>
#include <string>

#include "agg_basics.h"
#include "mlx/array.h"
#include "nb_compat.h"

namespace mx = mlx::core;

namespace mpl {

inline bool python_truth(py::handle src)
{
    int truth = PyObject_IsTrue(src.ptr());
    if (truth < 0) {
        py::raise_python_error();
    }
    return truth != 0;
}

/************************************************************
 * mpl::PathIterator acts as a bridge between MLXArrayBackend and Agg.  Given a
 * pair of MLXArrayBackend arrays, vertices and codes, it iterates over
 * those vertices and codes, using the standard Agg vertex source
 * interface:
 *
 *     unsigned vertex(double* x, double* y)
 */
class PathIterator
{
    /* We hold references to the Python objects, not just the
       underlying data arrays, so that Python reference counting
       can work.
    */
    py::object m_vertices_owner;
    py::object m_codes_owner;
    std::optional<mx::array> m_vertices_array;
    std::optional<mx::array> m_codes_array;
    bool m_vertices_cpu_direct;
    bool m_codes_cpu_direct;

    enum class VertexDtype {
        Float32,
        Float64
    };
    VertexDtype m_vertices_dtype;

    unsigned m_iterator;
    unsigned m_total_vertices;

    /* This class doesn't actually do any simplification, but we
       store the value here, since it is obtained from the Python
       object.
    */
    bool m_should_simplify;
    double m_simplify_threshold;

  public:
    inline PathIterator()
        : m_vertices_cpu_direct(false),
          m_codes_cpu_direct(false),
          m_vertices_dtype(VertexDtype::Float64),
          m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
    }

    inline PathIterator(py::object vertices, py::object codes, bool should_simplify,
                        double simplify_threshold)
        : m_iterator(0)
    {
        set(vertices, codes, should_simplify, simplify_threshold);
    }

    inline PathIterator(py::object vertices, py::object codes)
        : m_iterator(0)
    {
        set(vertices, codes);
    }

    inline PathIterator(const PathIterator &other)
    {
        m_vertices_owner = other.m_vertices_owner;
        m_codes_owner = other.m_codes_owner;
        m_vertices_array = other.m_vertices_array;
        m_codes_array = other.m_codes_array;
        m_vertices_dtype = other.m_vertices_dtype;
        m_iterator = 0;
        m_total_vertices = other.m_total_vertices;
        m_vertices_cpu_direct = other.m_vertices_cpu_direct;
        m_codes_cpu_direct = other.m_codes_cpu_direct;

        m_should_simplify = other.m_should_simplify;
        m_simplify_threshold = other.m_simplify_threshold;
    }

    inline mx::array cast_mlx_array(py::object obj, const char *name)
    {
        try {
            return py::cast<mx::array>(obj);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(name) + " must be an MLX array: " + e.what());
        }
    }

    inline bool is_mlx_array_like(py::object obj)
    {
        return py::isinstance<mx::array>(obj);
    }

    inline bool has_explicit_cpu_stream(py::object obj)
    {
        if (!py::hasattr(obj, "_mlx_stream")) {
            return false;
        }
        py::object stream = obj.attr("_mlx_stream");
        if (stream.is_none()) {
            return false;
        }
        auto text = std::string(py::str(stream).c_str());
        return text.find("cpu") != std::string::npos;
    }

    inline py::object get_python_item(py::handle obj, py::handle key)
    {
        PyObject* result = PyObject_GetItem(obj.ptr(), key.ptr());
        if (result == nullptr) {
            py::raise_python_error();
        }
        return py::steal<py::object>(result);
    }

    inline py::object mlx_scalar_at(py::handle array, size_t row, int column)
    {
        return get_python_item(
            array,
            py::make_tuple(static_cast<Py_ssize_t>(row),
                           static_cast<Py_ssize_t>(column)));
    }

    inline py::object mlx_scalar_at(py::handle array, size_t idx)
    {
        py::int_ key(static_cast<Py_ssize_t>(idx));
        return get_python_item(array, key);
    }

    inline double vertex_scalar(size_t idx, int column)
    {
        if (m_vertices_cpu_direct && m_vertices_array.has_value()) {
            auto& array = *m_vertices_array;
            auto offset = static_cast<std::int64_t>(idx) * array.strides(0)
                + static_cast<std::int64_t>(column) * array.strides(1);
            if (m_vertices_dtype == VertexDtype::Float64) {
                return array.data<double>()[offset];
            }
            return static_cast<double>(array.data<float>()[offset]);
        }
        auto scalar = mlx_scalar_at(m_vertices_owner, idx, column);
        return static_cast<double>(py::float_(scalar.attr("item")()));
    }

    inline unsigned code_scalar(size_t idx)
    {
        if (m_codes_cpu_direct && m_codes_array.has_value()) {
            auto& array = *m_codes_array;
            auto offset = static_cast<std::int64_t>(idx) * array.strides(0);
            return static_cast<unsigned>(array.data<std::uint8_t>()[offset]);
        }
        auto scalar = mlx_scalar_at(m_codes_owner, idx);
        return static_cast<unsigned>(
            static_cast<unsigned long long>(py::int_(scalar.attr("item")())));
    }

    inline void set_vertices_from_mlx(py::object vertices)
    {
        auto array = cast_mlx_array(vertices, "vertices");
        if (array.ndim() != 2 || array.shape(1) != 2) {
            throw py::value_error("Invalid vertices array");
        }

        if (array.dtype() == mx::float64) {
            m_vertices_dtype = VertexDtype::Float64;
        } else if (array.dtype() == mx::float32) {
            m_vertices_dtype = VertexDtype::Float32;
        } else {
            throw py::value_error("Unsupported vertices dtype");
        }

        m_vertices_cpu_direct = has_explicit_cpu_stream(vertices);
        if (m_vertices_cpu_direct) {
            array.eval();
        }
        m_vertices_owner = vertices;
        m_vertices_array = std::move(array);
        m_total_vertices = static_cast<unsigned>(m_vertices_array->shape(0));
    }

    inline void set_codes_from_mlx(py::object codes)
    {
        auto array = cast_mlx_array(codes, "codes");
        if (array.ndim() != 1
                || static_cast<unsigned>(array.shape(0)) != m_total_vertices) {
            throw py::value_error("Invalid codes array");
        }

        if (array.dtype() != mx::uint8) {
            throw py::value_error("Invalid codes array");
        }

        m_codes_cpu_direct = has_explicit_cpu_stream(codes);
        if (m_codes_cpu_direct) {
            array.eval();
        }
        m_codes_owner = codes;
        m_codes_array = std::move(array);
    }

    inline void
    set(py::object vertices, py::object codes, bool should_simplify, double simplify_threshold)
    {
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;
        m_vertices_array.reset();
        m_codes_array.reset();
        m_vertices_cpu_direct = false;
        m_codes_cpu_direct = false;

        m_vertices_owner = vertices;
        try {
            if (!is_mlx_array_like(vertices)) {
                throw py::type_error("Path vertices must be an MLX array");
            }
            set_vertices_from_mlx(vertices);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("vertices setup: ") + e.what());
        }

        if (!codes.is_none()) {
            m_codes_owner = codes;
            try {
                if (!is_mlx_array_like(codes)) {
                    throw py::type_error("Path codes must be an MLX array");
                }
                set_codes_from_mlx(codes);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("codes setup: ") + e.what());
            }
        } else {
            m_codes_array.reset();
            m_codes_owner = py::object();
            m_codes_cpu_direct = false;
        }

        m_iterator = 0;
    }

    inline void set(py::object vertices, py::object codes)
    {
        set(vertices, codes, false, 0.0);
    }

    inline unsigned vertex(double *x, double *y)
    {
        if (m_iterator >= m_total_vertices) {
            *x = 0.0;
            *y = 0.0;
            return agg::path_cmd_stop;
        }

        const size_t idx = m_iterator++;

        *x = vertex_scalar(idx, 0);
        *y = vertex_scalar(idx, 1);

        if (m_codes_array.has_value()) {
            return code_scalar(idx);
        } else {
            return idx == 0 ? agg::path_cmd_move_to : agg::path_cmd_line_to;
        }
    }

    inline void rewind(unsigned path_id)
    {
        m_iterator = path_id;
    }

    inline unsigned total_vertices() const
    {
        return m_total_vertices;
    }

    inline bool should_simplify() const
    {
        return m_should_simplify;
    }

    inline double simplify_threshold() const
    {
        return m_simplify_threshold;
    }

    inline bool has_codes() const
    {
        return m_codes_array.has_value();
    }

    inline void *get_id()
    {
        return (void *)m_vertices_owner.ptr();
    }
};

class PathGenerator
{
    py::sequence m_paths;
    Py_ssize_t m_npaths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator() : m_npaths(0) {}

    void set(py::object obj)
    {
        m_paths = py::cast<py::sequence>(obj);
        m_npaths = static_cast<Py_ssize_t>(py::len(m_paths));
    }

    Py_ssize_t num_paths() const
    {
        return m_npaths;
    }

    Py_ssize_t size() const
    {
        return m_npaths;
    }

    path_iterator operator()(size_t i)
    {
        path_iterator path;

        auto item = m_paths[i % m_npaths];
        path = py::cast<path_iterator>(item);
        return path;
    }
};
}

namespace nanobind { namespace detail {
    template <> struct type_caster<mpl::PathIterator> {
    public:
        NB_TYPE_CASTER(mpl::PathIterator, const_name("PathIterator"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                return true;
            }

            py::object obj = py::borrow<py::object>(src);
            py::object vertices;
            py::object codes;
            bool should_simplify;
            double simplify_threshold;

            try {
                vertices = obj.attr("vertices");
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("path.vertices: ") + e.what());
            }
            try {
                codes = obj.attr("codes");
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("path.codes: ") + e.what());
            }
            try {
                should_simplify = mpl::python_truth(obj.attr("should_simplify"));
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("path.should_simplify: ") + e.what());
            }
            try {
                simplify_threshold = static_cast<double>(
                    py::float_(obj.attr("simplify_threshold")));
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("path.simplify_threshold: ") + e.what());
            }

            try {
                value.set(vertices, codes, should_simplify, simplify_threshold);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("path data: ") + e.what());
            }

            return true;
        }
    };

    template <> struct type_caster<mpl::PathGenerator> {
    public:
        NB_TYPE_CASTER(mpl::PathGenerator, const_name("PathGenerator"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            value.set(py::reinterpret_borrow<py::object>(src));
            return true;
        }
    };
}} // namespace nanobind::detail

#endif
