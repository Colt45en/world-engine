#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AssetResourceBridge.hpp"

namespace py = pybind11;

PYBIND11_MODULE(assets, m) {
    m.doc() = "Tier-4 Asset add-in: async asset requests with priority + memory budget";

    py::class_<AssetResourceBridge>(m, "AssetBridge")
        .def(py::init<double>(), py::arg("mem_limit_mb") = 2048.0)
        .def("register_base_path", &AssetResourceBridge::registerBasePath, py::arg("type"), py::arg("path"))
        .def("preload", &AssetResourceBridge::preload, py::arg("items"))
        .def(
            "request",
            &AssetResourceBridge::request,
            py::arg("type"),
            py::arg("id"),
            py::arg("priority") = 0,
            py::arg("on_ok") = nullptr,
            py::arg("on_err") = nullptr
        )
        .def("start", &AssetResourceBridge::start, py::arg("hz") = 30)
        .def("stop", &AssetResourceBridge::stop);
}
