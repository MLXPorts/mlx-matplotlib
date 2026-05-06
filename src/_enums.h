#ifndef MPL_ENUMS_H
#define MPL_ENUMS_H

#include "nb_compat.h"

// Extension for nanobind: Pythonic enums.
// This allows creating classes based on ``enum.*`` types.
// This code was copied from mplcairo, with some slight tweaks.
// The API is:
//
// - P11X_DECLARE_ENUM(py_name: str, py_base_cls: str, ...: {str, enum value}):
//   py_name: The name to expose in the module.
//   py_base_cls: The name of the enum base class to use.
//   ...: The enum name/value pairs to expose.
//
//   Use this macro to declare an enum and its values.
//
// - py11x::bind_enums(m: nanobind::module):
//   m: The module to use to register the enum classes.
//
//   Place this in NB_MODULE to register the enums declared by P11X_DECLARE_ENUM.

// a1 includes the opening brace and a2 the closing brace.
// This definition is compatible with older compiler versions compared to
// #define P11X_ENUM_TYPE(...) decltype(std::map{std::pair __VA_ARGS__})::mapped_type
#define P11X_ENUM_TYPE(a1, a2, ...) decltype(std::pair a1, a2)::second_type

#define P11X_CAT2(a, b) a##b
#define P11X_CAT(a, b) P11X_CAT2(a, b)

namespace p11x {
  namespace {

    // Holder is (py_base_cls, [(name, value), ...]) before module init;
    // converted to the Python class object after init.
    auto enums = std::unordered_map<std::string, py::object>{};

    auto bind_enums(py::module_ mod) -> void
    {
      for (auto& [py_name, spec]: enums) {
        auto const& [py_base_cls, pairs] =
          py::cast<std::pair<std::string, py::object>>(spec);
        mod.attr(py::cast(py_name)) = spec =
          py::module_::import_("enum").attr(py_base_cls.c_str())(
            py_name, pairs, py::arg("module") = mod.attr("__name__"));
      }
    }
  }
}

// Immediately converting the args to a vector outside of the lambda avoids
// name collisions.
#define P11X_DECLARE_ENUM(py_name, py_base_cls, ...) \
  namespace p11x { \
    namespace { \
      [[maybe_unused]] auto const P11X_CAT(enum_placeholder_, __COUNTER__) = \
        [](auto args) { \
          py::gil_scoped_acquire gil; \
          using int_t = std::underlying_type_t<decltype(args[0].second)>; \
          auto pairs = std::vector<std::pair<std::string, int_t>>{}; \
          for (auto& [k, v]: args) { \
            pairs.emplace_back(k, int_t(v)); \
          } \
          p11x::enums[py_name] = nanobind::cast(std::pair{py_base_cls, pairs}); \
          return 0; \
        } (std::vector{std::pair __VA_ARGS__}); \
    } \
  } \
  namespace nanobind::detail { \
    template<> struct type_caster<P11X_ENUM_TYPE(__VA_ARGS__)> { \
      using type = P11X_ENUM_TYPE(__VA_ARGS__); \
      static_assert(std::is_enum_v<type>, "Not an enum"); \
      NB_TYPE_CASTER(type, const_name(py_name)); \
      bool from_python(handle src, uint8_t, cleanup_list *) { \
        auto cls = p11x::enums.at(py_name); \
        PyObject* tmp = nullptr; \
        if (nanobind::isinstance(src, cls) \
            && (tmp = PyNumber_Index(src.attr("value").ptr()))) { \
          auto ival = PyLong_AsLong(tmp); \
          value = decltype(value)(ival); \
          Py_DECREF(tmp); \
          return !(ival == -1 && !PyErr_Occurred()); \
        } else { \
          return false; \
        } \
      } \
      static handle from_cpp(decltype(value) obj, rv_policy, cleanup_list *) { \
        auto cls = p11x::enums.at(py_name); \
        return cls(std::underlying_type_t<type>(obj)).release(); \
      } \
    }; \
  }

#endif /* MPL_ENUMS_H */
