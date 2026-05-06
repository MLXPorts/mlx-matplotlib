/*
 * Wrapper module for libqhull, providing Delaunay triangulation.
 *
 * This MLX fork removes the hard dependency on MLXArrayBackend by using the Python buffer
 * protocol for inputs and returning shaped memoryviews for outputs.
 */
#include "nb_compat.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
extern "C" {
extern const char qh_version[];
}
#endif

#include "libqhull_r/qhull_ra.h"

#include "py_buffer.h"

#ifndef MPL_DEVNULL
#error "MPL_DEVNULL must be defined as the OS-equivalent of /dev/null"
#endif

#define STRINGIFY(x) STR(x)
#define STR(x) #x

using namespace nanobind::literals;

static const char *qhull_error_msg[6] = {
    "",                     /* 0 = qh_ERRnone */
    "input inconsistency",  /* 1 = qh_ERRinput */
    "singular input data",  /* 2 = qh_ERRsingular */
    "precision error",      /* 3 = qh_ERRprec */
    "insufficient memory",  /* 4 = qh_ERRmem */
    "internal error"};      /* 5 = qh_ERRqhull */

static void get_facet_vertices(qhT *qh, const facetT *facet, int indices[3])
{
    vertexT *vertex, **vertexp;
    FOREACHvertex_(facet->vertices) { *indices++ = qh_pointid(qh, vertex->point); }
}

static void get_facet_neighbours(const facetT *facet,
                                const std::vector<int> &tri_indices,
                                int indices[3])
{
    facetT *neighbor, **neighborp;
    FOREACHneighbor_(facet) { *indices++ = (neighbor->upperdelaunay ? -1 : tri_indices[neighbor->id]); }
}

static bool at_least_3_unique_points(py::ssize_t npoints, const double *x, const double *y)
{
    const py::ssize_t unique1 = 0;
    py::ssize_t unique2 = 0;

    if (npoints < 3) {
        return false;
    }

    for (py::ssize_t i = 1; i < npoints; ++i) {
        if (unique2 == 0) {
            if (x[i] != x[unique1] || y[i] != y[unique1]) {
                unique2 = i;
            }
        } else {
            if ((x[i] != x[unique1] || y[i] != y[unique1])
                && (x[i] != x[unique2] || y[i] != y[unique2])) {
                return true;
            }
        }
    }

    return false;
}

class QhullInfo {
public:
    QhullInfo(FILE *error_file, qhT *qh) : error_file_(error_file), qh_(qh) {}

    ~QhullInfo()
    {
        qh_freeqhull(qh_, !qh_ALL);
        int curlong, totlong;
        qh_memfreeshort(qh_, &curlong, &totlong);
        if (curlong || totlong) {
            PyErr_WarnEx(PyExc_RuntimeWarning, "Qhull could not free all allocated memory", 1);
        }

        if (error_file_ != stderr) {
            fclose(error_file_);
        }
    }

private:
    FILE *error_file_;
    qhT *qh_;
};

static py::tuple delaunay_impl(py::ssize_t npoints,
                              const double *x,
                              const double *y,
                              bool hide_qhull_errors)
{
    qhT qh_qh;
    qhT *qh = &qh_qh;
    facetT *facet;
    int ntri;
    int max_facet_id;
    int exitcode;
    const int ndim = 2;
    double x_mean = 0.0;
    double y_mean = 0.0;

    QHULL_LIB_CHECK

    std::vector<coordT> points(static_cast<size_t>(npoints) * ndim);

    for (py::ssize_t i = 0; i < npoints; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= npoints;
    y_mean /= npoints;

    for (py::ssize_t i = 0; i < npoints; ++i) {
        points[static_cast<size_t>(2 * i)] = x[i] - x_mean;
        points[static_cast<size_t>(2 * i + 1)] = y[i] - y_mean;
    }

    FILE *error_file = nullptr;
    if (hide_qhull_errors) {
        error_file = fopen(STRINGIFY(MPL_DEVNULL), "w");
        if (error_file == nullptr) {
            throw std::runtime_error("Could not open devnull");
        }
    } else {
        error_file = stderr;
    }

    QhullInfo info(error_file, qh);
    qh_zero(qh, error_file);
    exitcode = qh_new_qhull(qh,
                            ndim,
                            static_cast<int>(npoints),
                            points.data(),
                            False,
                            (char *)"qhull d Qt Qbb Qc Qz",
                            nullptr,
                            error_file);
    if (exitcode != qh_ERRnone) {
        std::string msg = py::cast<std::string>(
            py::str("Error in qhull Delaunay triangulation calculation: {} (exitcode={})")
                .format(qhull_error_msg[exitcode], exitcode));
        if (hide_qhull_errors) {
            msg += "; use python verbose option (-v) to see original qhull error.";
        }
        throw std::runtime_error(msg);
    }

    qh_triangulate(qh);

    ntri = 0;
    FORALLfacets
    {
        if (!facet->upperdelaunay) {
            ++ntri;
        }
    }

    max_facet_id = qh->facet_id - 1;
    std::vector<int> tri_indices(static_cast<size_t>(max_facet_id + 1));

    // Map facet id -> triangle index.
    int tri_index = 0;
    FORALLfacets
    {
        if (!facet->upperdelaunay) {
            tri_indices[facet->id] = tri_index++;
        }
    }

    std::vector<std::int32_t> triangles(static_cast<size_t>(ntri) * 3);
    std::vector<std::int32_t> neighbors(static_cast<size_t>(ntri) * 3);

    // Fill triangles.
    auto *tri_ptr = triangles.data();
    FORALLfacets
    {
        if (!facet->upperdelaunay) {
            int indices[3];
            get_facet_vertices(qh, facet, indices);
            *tri_ptr++ = static_cast<std::int32_t>(facet->toporient ? indices[2] : indices[0]);
            *tri_ptr++ = static_cast<std::int32_t>(facet->toporient ? indices[0] : indices[2]);
            *tri_ptr++ = static_cast<std::int32_t>(indices[1]);
        }
    }

    // Fill neighbors.
    auto *nb_ptr = neighbors.data();
    FORALLfacets
    {
        if (!facet->upperdelaunay) {
            int indices[3];
            get_facet_neighbours(facet, tri_indices, indices);
            *nb_ptr++ = static_cast<std::int32_t>(facet->toporient ? indices[2] : indices[0]);
            *nb_ptr++ = static_cast<std::int32_t>(facet->toporient ? indices[0] : indices[2]);
            *nb_ptr++ = static_cast<std::int32_t>(indices[1]);
        }
    }

    auto make_view = [&](const std::vector<std::int32_t> &v) {
        auto ba_size = static_cast<py::ssize_t>(v.size() * sizeof(std::int32_t));
        py::bytearray ba = py::reinterpret_steal<py::bytearray>(
            PyByteArray_FromStringAndSize(nullptr, ba_size));
        if (!ba) {
            py::raise_python_error();
        }
        std::memcpy(PyByteArray_AsString(ba.ptr()), v.data(), v.size() * sizeof(std::int32_t));
        py::object mv = py::module_::import_("builtins").attr("memoryview")(ba);
        return py::cast<py::memoryview>(
            mv.attr("cast")("i", py::make_tuple(static_cast<py::ssize_t>(ntri), 3)));
    };

    return py::make_tuple(make_view(triangles), make_view(neighbors));
}

static py::tuple delaunay(const py::buffer &x, const py::buffer &y, int verbose)
{
    mpl::BufferView<double, 1> x_view(x);
    mpl::BufferView<double, 1> y_view(y);

    auto npoints = x_view.shape(0);
    if (npoints != y_view.shape(0)) {
        throw std::invalid_argument("x and y must be 1D buffers of the same length");
    }
    if (npoints < 3) {
        throw std::invalid_argument("x and y buffers must have a length of at least 3");
    }
    if (!at_least_3_unique_points(npoints, x_view.data(), y_view.data())) {
        throw std::invalid_argument("x and y buffers must consist of at least 3 unique points");
    }

    return delaunay_impl(npoints, x_view.data(), y_view.data(), verbose == 0);
}

NB_MODULE(_qhull, m)
{
    m.doc() = "Computing Delaunay triangulations.\n";

    m.def("delaunay",
          &delaunay,
          "x"_a,
          "y"_a,
          "verbose"_a,
          "--\n\n"
          "Compute a Delaunay triangulation.\n\n"
          "Returns\n"
          "-------\n"
          "triangles, neighbors : shaped memoryviews, (ntri, 3)\n");

    m.def("version",
          []() { return qh_version; },
          "version()\n--\n\nReturn the qhull version string.");
}
