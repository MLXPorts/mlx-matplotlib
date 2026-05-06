#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "mplutils.h"
#include "py_converters.h"
#include "_backend_agg.h"
#include "py_buffer.h"

namespace py = nanobind;
using namespace nanobind::literals;

/**********************************************************************
 * BufferRegion
 * */

/* TODO: This doesn't seem to be used internally.  Remove? */

static void
PyBufferRegion_set_x(BufferRegion *self, int x)
{
    self->get_rect().x1 = x;
}

static void
PyBufferRegion_set_y(BufferRegion *self, int y)
{
    self->get_rect().y1 = y;
}

static py::object
PyBufferRegion_get_extents(BufferRegion *self)
{
    agg::rect_i rect = self->get_rect();

    return py::make_tuple(rect.x1, rect.y1, rect.x2, rect.y2);
}

struct AggBufferState
{
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    char format[2];
};

static int
fill_u8_rgba_buffer(PyObject *exporter,
                    Py_buffer *view,
                    void *data,
                    unsigned int width,
                    unsigned int height,
                    int flags)
{
    auto *state = static_cast<AggBufferState *>(PyMem_Malloc(sizeof(AggBufferState)));
    if (!state) {
        PyErr_NoMemory();
        return -1;
    }

    state->shape[0] = static_cast<Py_ssize_t>(height);
    state->shape[1] = static_cast<Py_ssize_t>(width);
    state->shape[2] = 4;
    state->strides[0] = static_cast<Py_ssize_t>(width * 4);
    state->strides[1] = 4;
    state->strides[2] = 1;
    state->format[0] = 'B';
    state->format[1] = '\0';

    if (PyBuffer_FillInfo(view,
                          exporter,
                          data,
                          static_cast<Py_ssize_t>(height * width * 4),
                          0,
                          flags) != 0) {
        PyMem_Free(state);
        return -1;
    }

    view->itemsize = 1;
    view->format = (flags & PyBUF_FORMAT) ? state->format : nullptr;
    view->ndim = 3;
    view->shape = (flags & PyBUF_ND) ? state->shape : nullptr;
    view->strides = (flags & PyBUF_STRIDES) ? state->strides : nullptr;
    view->suboffsets = nullptr;
    view->internal = state;
    return 0;
}

static void
release_u8_rgba_buffer(PyObject *, Py_buffer *view)
{
    if (view->internal) {
        PyMem_Free(view->internal);
        view->internal = nullptr;
    }
}

static int
RendererAgg_getbuffer(PyObject *exporter, Py_buffer *view, int flags)
{
    auto *renderer = py::inst_ptr<RendererAgg>(py::handle(exporter));
    return fill_u8_rgba_buffer(exporter,
                               view,
                               renderer->pixBuffer,
                               renderer->get_width(),
                               renderer->get_height(),
                               flags);
}

static int
BufferRegion_getbuffer(PyObject *exporter, Py_buffer *view, int flags)
{
    auto *buffer = py::inst_ptr<BufferRegion>(py::handle(exporter));
    return fill_u8_rgba_buffer(exporter,
                               view,
                               buffer->get_data(),
                               static_cast<unsigned int>(buffer->get_width()),
                               static_cast<unsigned int>(buffer->get_height()),
                               flags);
}

static PyType_Slot RendererAgg_slots[] = {
    {Py_bf_getbuffer, reinterpret_cast<void *>(RendererAgg_getbuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void *>(release_u8_rgba_buffer)},
    {0, nullptr}
};

static PyType_Slot BufferRegion_slots[] = {
    {Py_bf_getbuffer, reinterpret_cast<void *>(BufferRegion_getbuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void *>(release_u8_rgba_buffer)},
    {0, nullptr}
};

/**********************************************************************
 * RendererAgg
 * */

static void
PyRendererAgg_draw_path(RendererAgg *self,
                        py::object gc_obj,
                        py::object path_obj,
                        py::object trans_obj,
                        py::object rgbFace)
{
    GCAgg gc;
    try {
        set_gcagg_from_python(gc_obj, gc);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to convert graphics context: ") + e.what());
    }
    mpl::PathIterator path;
    try {
        path = py::cast<mpl::PathIterator>(path_obj);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to convert path: ") + e.what());
    }
    agg::trans_affine trans;
    try {
        trans = py::cast<agg::trans_affine>(trans_obj);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to convert transform: ") + e.what());
    }
    agg::rgba face;
    try {
        face = py::cast<agg::rgba>(rgbFace);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to convert face color: ") + e.what());
    }
    if (!rgbFace.is_none()) {
        if (gc.forced_alpha || py::len(py::cast<py::sequence>(rgbFace)) == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_path(gc, path, trans, face);
}

static void
PyRendererAgg_draw_text_image(RendererAgg *self,
                              const py::buffer &image_obj,
                              std::variant<double, int> vx,
                              std::variant<double, int> vy,
                              double angle,
                              py::object gc_obj)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    int x, y;

    if (auto value = std::get_if<double>(&vx)) {
        auto api = py::module_::import_("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="x", "obj_type"_a="parameter as float",
             "alternative"_a="int(x)");
        x = static_cast<int>(*value);
    } else if (auto value = std::get_if<int>(&vx)) {
        x = *value;
    } else {
        throw std::runtime_error("Should not happen");
    }

    if (auto value = std::get_if<double>(&vy)) {
        auto api = py::module_::import_("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="y", "obj_type"_a="parameter as float",
             "alternative"_a="int(y)");
        y = static_cast<int>(*value);
    } else if (auto value = std::get_if<int>(&vy)) {
        y = *value;
    } else {
        throw std::runtime_error("Should not happen");
    }

    mpl::BufferView<agg::int8u, 2> image(image_obj);

    self->draw_text_image(gc, image, x, y, angle);
}

static void
PyRendererAgg_draw_markers(RendererAgg *self,
                           py::object gc_obj,
                           mpl::PathIterator marker_path,
                           agg::trans_affine marker_path_trans,
                           mpl::PathIterator path,
                           agg::trans_affine trans,
                           py::object rgbFace)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    agg::rgba face = py::cast<agg::rgba>(rgbFace);
    if (!rgbFace.is_none()) {
        if (gc.forced_alpha || py::len(py::cast<py::sequence>(rgbFace)) == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_markers(gc, marker_path, marker_path_trans, path, trans, face);
}

static void
PyRendererAgg_draw_image(RendererAgg *self,
                         py::object gc_obj,
                         double x,
                         double y,
                         const py::buffer &image_obj)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    // TODO: This really shouldn't be mutable, but Agg's renderer buffers aren't const.
    mpl::BufferView<agg::int8u, 3> image(image_obj, true);

    x = mpl_round(x);
    y = mpl_round(y);

    gc.alpha = 1.0;
    self->draw_image(gc, x, y, image);
}

static void
PyRendererAgg_draw_path_collection(RendererAgg *self,
                                   py::object gc_obj,
                                   agg::trans_affine master_transform,
                                   mpl::PathGenerator paths,
                                   py::object transforms_obj,
                                   py::object offsets_obj,
                                   agg::trans_affine offset_trans,
                                   py::object facecolors_obj,
                                   py::object edgecolors_obj,
                                   py::object linewidths_obj,
                                   DashesVector dashes,
                                   py::object antialiaseds_obj,
                                   py::object Py_UNUSED(ignored_obj),
                                   // offset position is no longer used
                                   py::object Py_UNUSED(offset_position_obj),
                                   py::object hatchcolors_obj)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    auto transforms = convert_mlx_transforms(std::move(transforms_obj));
    auto offsets = convert_mlx_points(std::move(offsets_obj));
    auto facecolors = convert_mlx_colors(std::move(facecolors_obj));
    auto edgecolors = convert_mlx_colors(std::move(edgecolors_obj));
    auto hatchcolors = hatchcolors_obj.is_none()
        ? mpl::MlxArrayView<double, 2>()
        : convert_mlx_colors(std::move(hatchcolors_obj));
    mpl::MlxArrayView<double, 1> linewidths(std::move(linewidths_obj), "linewidths");
    mpl::MlxArrayView<uint8_t, 1> antialiaseds(std::move(antialiaseds_obj), "antialiaseds");

    self->draw_path_collection(gc,
            master_transform,
            paths,
            transforms,
            offsets,
            offset_trans,
            facecolors,
            edgecolors,
            linewidths,
            dashes,
            antialiaseds,
            hatchcolors);
}

static void
PyRendererAgg_draw_quad_mesh(RendererAgg *self,
                             py::object gc_obj,
                             agg::trans_affine master_transform,
                             unsigned int mesh_width,
                             unsigned int mesh_height,
                             py::object coordinates_obj,
                             py::object offsets_obj,
                             agg::trans_affine offset_trans,
                             py::object facecolors_obj,
                             py::object antialiased_obj,
                             py::object edgecolors_obj)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    mpl::MlxArrayView<double, 3> coordinates(std::move(coordinates_obj), "coordinates");
    auto offsets = convert_mlx_points(std::move(offsets_obj));
    auto facecolors = convert_mlx_colors(std::move(facecolors_obj));
    auto edgecolors = convert_mlx_colors(std::move(edgecolors_obj));
    int antialiased_truth = PyObject_IsTrue(antialiased_obj.ptr());
    if (antialiased_truth < 0) {
        py::raise_python_error();
    }

    self->draw_quad_mesh(gc,
            master_transform,
            mesh_width,
            mesh_height,
            coordinates,
            offsets,
            offset_trans,
            facecolors,
            antialiased_truth != 0,
            edgecolors);
}

static void
PyRendererAgg_draw_gouraud_triangles(RendererAgg *self,
                                     py::object gc_obj,
                                     py::object points_obj,
                                     py::object colors_obj,
                                     agg::trans_affine trans)
{
    GCAgg gc;
    set_gcagg_from_python(gc_obj, gc);
    mpl::MlxArrayView<double, 3> points(std::move(points_obj), "points");
    mpl::MlxArrayView<double, 3> colors(std::move(colors_obj), "colors");

    self->draw_gouraud_triangles(gc, points, colors, trans);
}

NB_MODULE(_backend_agg, m)
{
    py::class_<RendererAgg>(m, "RendererAgg", py::type_slots(RendererAgg_slots))
        .def(py::init<unsigned int, unsigned int, double>(),
             "width"_a, "height"_a, "dpi"_a)

        .def("draw_path", &PyRendererAgg_draw_path,
             "gc"_a, "path"_a, "trans"_a, "face"_a = nullptr)
        .def("draw_markers", &PyRendererAgg_draw_markers,
             "gc"_a, "marker_path"_a, "marker_path_trans"_a, "path"_a, "trans"_a,
             "face"_a = nullptr)
        .def("draw_text_image", &PyRendererAgg_draw_text_image,
             "image"_a, "x"_a, "y"_a, "angle"_a, "gc"_a)
        .def("draw_image", &PyRendererAgg_draw_image,
             "gc"_a, "x"_a, "y"_a, "image"_a)
        .def("draw_path_collection", &PyRendererAgg_draw_path_collection,
             "gc"_a, "master_transform"_a, "paths"_a, "transforms"_a, "offsets"_a,
             "offset_trans"_a, "facecolors"_a, "edgecolors"_a, "linewidths"_a,
             "dashes"_a, "antialiaseds"_a, "ignored"_a, "offset_position"_a,
             py::kw_only(), "hatchcolors"_a = py::none())
        .def("draw_quad_mesh", &PyRendererAgg_draw_quad_mesh,
             "gc"_a, "master_transform"_a, "mesh_width"_a, "mesh_height"_a,
             "coordinates"_a, "offsets"_a, "offset_trans"_a, "facecolors"_a,
             "antialiased"_a, "edgecolors"_a)
        .def("draw_gouraud_triangles", &PyRendererAgg_draw_gouraud_triangles,
             "gc"_a, "points"_a, "colors"_a, "trans"_a = nullptr)

        .def("clear", &RendererAgg::clear)

        .def("copy_from_bbox", &RendererAgg::copy_from_bbox,
             "bbox"_a)
        .def("restore_region",
             py::overload_cast<BufferRegion&>(&RendererAgg::restore_region),
             "region"_a)
        .def("restore_region",
             py::overload_cast<BufferRegion&, int, int, int, int, int, int>(&RendererAgg::restore_region),
             "region"_a, "xx1"_a, "yy1"_a, "xx2"_a, "yy2"_a, "x"_a, "y"_a)

        ;

    py::class_<BufferRegion>(m, "BufferRegion", py::type_slots(BufferRegion_slots))
        // BufferRegion is not constructible from Python, thus no py::init is added.
        .def("set_x", &PyBufferRegion_set_x)
        .def("set_y", &PyBufferRegion_set_y)
        .def("get_extents", &PyBufferRegion_get_extents);
}
