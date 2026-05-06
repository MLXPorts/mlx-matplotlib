#ifndef MPL_BACKEND_AGG_BASIC_TYPES_H
#define MPL_BACKEND_AGG_BASIC_TYPES_H

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "agg_trans_affine.h"
#include "nb_compat.h"
#include "path_converters.h"

#include "py_adaptors.h"

struct ClipPath
{
    mpl::PathIterator path;
    agg::trans_affine trans;
};

struct SketchParams
{
    double scale;
    double length;
    double randomness;
};

class Dashes
{
    typedef std::vector<std::pair<double, double> > dash_t;
    double dash_offset;
    dash_t dashes;

  public:
    double get_dash_offset() const
    {
        return dash_offset;
    }
    void set_dash_offset(double x)
    {
        dash_offset = x;
    }
    void add_dash_pair(double length, double skip)
    {
        dashes.emplace_back(length, skip);
    }
    size_t size() const
    {
        return dashes.size();
    }

    template <class T>
    void dash_to_stroke(T &stroke, double dpi, bool isaa)
    {
        double scaleddpi = dpi / 72.0;
        for (auto [val0, val1] : dashes) {
            val0 = val0 * scaleddpi;
            val1 = val1 * scaleddpi;
            if (!isaa) {
                val0 = (int)val0 + 0.5;
                val1 = (int)val1 + 0.5;
            }
            stroke.add_dash(val0, val1);
        }
        stroke.dash_start(get_dash_offset() * scaleddpi);
    }
};

typedef std::vector<Dashes> DashesVector;

class GCAgg
{
  public:
    GCAgg()
        : linewidth(1.0),
          alpha(1.0),
          cap(agg::butt_cap),
          join(agg::round_join),
          snap_mode(SNAP_FALSE)
    {
    }

    ~GCAgg()
    {
    }

    double linewidth;
    double alpha;
    bool forced_alpha;
    agg::rgba color;
    bool isaa;

    agg::line_cap_e cap;
    agg::line_join_e join;

    agg::rect_d cliprect;

    ClipPath clippath;

    Dashes dashes;

    e_snap_mode snap_mode;

    mpl::PathIterator hatchpath;
    agg::rgba hatch_color;
    double hatch_linewidth;

    SketchParams sketch;

    bool has_hatchpath()
    {
        return hatchpath.total_vertices() != 0;
    }

  private:
    // prevent copying
    GCAgg(const GCAgg &);
    GCAgg &operator=(const GCAgg &);
};

inline bool python_truth(py::handle src);

inline void set_gcagg_from_python(py::handle src, GCAgg& value)
{
    py::object obj = py::reinterpret_borrow<py::object>(src);
#define SET_GC_FIELD(name_, field_, expr_) \
    do { \
        try { \
            value.field_ = (expr_); \
        } catch (const std::exception& e) { \
            throw std::runtime_error(std::string(name_) + ": " + e.what()); \
        } \
    } while (false)
    SET_GC_FIELD("_linewidth", linewidth, py::cast<double>(obj.attr("_linewidth")));
    SET_GC_FIELD("_alpha", alpha, py::cast<double>(obj.attr("_alpha")));
    SET_GC_FIELD("_forced_alpha", forced_alpha, python_truth(obj.attr("_forced_alpha")));
    SET_GC_FIELD("_rgb", color, py::cast<agg::rgba>(obj.attr("_rgb")));
    SET_GC_FIELD("_antialiased", isaa, python_truth(obj.attr("_antialiased")));
    SET_GC_FIELD("_capstyle", cap, py::cast<agg::line_cap_e>(obj.attr("_capstyle")));
    SET_GC_FIELD("_joinstyle", join, py::cast<agg::line_join_e>(obj.attr("_joinstyle")));
    SET_GC_FIELD("get_dashes", dashes, py::cast<Dashes>(obj.attr("get_dashes")()));
    SET_GC_FIELD("get_clip_rectangle_agg", cliprect,
                 py::cast<agg::rect_d>(obj.attr("get_clip_rectangle_agg")()));
    SET_GC_FIELD("get_clip_path_agg", clippath,
                 py::cast<ClipPath>(obj.attr("get_clip_path_agg")()));
    SET_GC_FIELD("get_snap", snap_mode, py::cast<e_snap_mode>(obj.attr("get_snap")()));
    SET_GC_FIELD("get_hatch_path", hatchpath,
                 py::cast<mpl::PathIterator>(obj.attr("get_hatch_path")()));
    SET_GC_FIELD("get_hatch_color", hatch_color,
                 py::cast<agg::rgba>(obj.attr("get_hatch_color")()));
    SET_GC_FIELD("get_hatch_linewidth", hatch_linewidth,
                 py::cast<double>(obj.attr("get_hatch_linewidth")()));
    SET_GC_FIELD("get_sketch_params", sketch,
                 py::cast<SketchParams>(obj.attr("get_sketch_params")()));
#undef SET_GC_FIELD
}

inline std::string string_or_enum_value(py::handle src)
{
    if (PyUnicode_Check(src.ptr())) {
        return py::cast<std::string>(src);
    }
    if (PyObject_HasAttrString(src.ptr(), "value")) {
        return py::cast<std::string>(src.attr("value"));
    }
    return py::cast<std::string>(src);
}

inline bool python_truth(py::handle src)
{
    int truth = PyObject_IsTrue(src.ptr());
    if (truth < 0) {
        py::raise_python_error();
    }
    return truth != 0;
}

inline e_snap_mode snap_mode_from_python(py::handle src)
{
    if (src.is_none()) {
        return SNAP_AUTO;
    }
    return python_truth(src) ? SNAP_TRUE : SNAP_FALSE;
}

namespace nanobind { namespace detail {
    template <> struct type_caster<agg::line_cap_e> {
    public:
        NB_TYPE_CASTER(agg::line_cap_e, const_name("line_cap_e"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            const std::unordered_map<std::string, agg::line_cap_e> enum_values = {
                {"butt", agg::butt_cap},
                {"round", agg::round_cap},
                {"projecting", agg::square_cap},
            };
            value = enum_values.at(string_or_enum_value(src));
            return true;
        }
    };

    template <> struct type_caster<agg::line_join_e> {
    public:
        NB_TYPE_CASTER(agg::line_join_e, const_name("line_join_e"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            const std::unordered_map<std::string, agg::line_join_e> enum_values = {
                {"miter", agg::miter_join_revert},
                {"round", agg::round_join},
                {"bevel", agg::bevel_join},
            };
            value = enum_values.at(string_or_enum_value(src));
            return true;
        }
    };

    template <> struct type_caster<ClipPath> {
    public:
        NB_TYPE_CASTER(ClipPath, const_name("ClipPath"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                return true;
            }

            auto pair = py::cast<py::tuple>(src);
            if (pair.size() != 2) {
                throw py::value_error("clip path must be a path/transform pair");
            }
            if (!py::handle(pair[0]).is_none()) {
                value.path = py::cast<mpl::PathIterator>(pair[0]);
            }
            value.trans = py::cast<agg::trans_affine>(pair[1]);

            return true;
        }
    };

    template <> struct type_caster<Dashes> {
    public:
        NB_TYPE_CASTER(Dashes, const_name("Dashes"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            auto pair = py::cast<py::tuple>(src);
            if (pair.size() != 2) {
                throw py::value_error("dashes must be an offset/pattern pair");
            }
            auto dash_offset = py::cast<double>(pair[0]);

            if (py::handle(pair[1]).is_none()) {
                return true;
            }

            auto dashes_seq = py::cast<py::sequence>(pair[1]);

            auto nentries = py::len(dashes_seq);
            // If the dashpattern has odd length, iterate through it twice (in
            // accordance with the pdf/ps/svg specs).
            auto dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

            for (size_t i = 0; i < dash_pattern_length; i += 2) {
                auto length = py::cast<double>(dashes_seq[i % nentries]);
                auto skip = py::cast<double>(dashes_seq[(i + 1) % nentries]);

                value.add_dash_pair(length, skip);
            }

            value.set_dash_offset(dash_offset);

            return true;
        }
    };

    template <> struct type_caster<SketchParams> {
    public:
        NB_TYPE_CASTER(SketchParams, const_name("SketchParams"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                value.scale = 0.0;
                value.length = 0.0;
                value.randomness = 0.0;
                return true;
            }

            auto params = py::cast<py::tuple>(src);
            if (params.size() != 3) {
                throw py::value_error("sketch parameters must have three values");
            }
            value.scale = py::cast<double>(params[0]);
            value.length = py::cast<double>(params[1]);
            value.randomness = py::cast<double>(params[2]);

            return true;
        }
    };

}} // namespace nanobind::detail

#endif
