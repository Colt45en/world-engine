#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quantum_thought_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qtp, m) {
    m.doc() = "Quantum Thought Pipeline (C++ core) — Python bindings";

    py::class_<ThoughtZone>(m, "ThoughtZone")
        .def_readonly("name", &ThoughtZone::name)
        .def_readonly("type", &ThoughtZone::type)
        .def_readonly("position", &ThoughtZone::position)
        .def_readonly("scale", &ThoughtZone::scale)
        .def_readonly("color", &ThoughtZone::color);

    py::class_<Agent>(m, "Agent")
        .def_readonly("id", &Agent::id)
        .def_readonly("position", &Agent::position)
        .def_readonly("maxSteps", &Agent::maxSteps)
        .def_readonly("stepSize", &Agent::stepSize)
        .def_readonly("trailStartColor", &Agent::trailStartColor)
        .def_readonly("trailEndColor", &Agent::trailEndColor);

    py::class_<QuantumThoughtPipeline>(m, "QuantumThoughtPipeline")
        .def(py::init<>())
        .def("get_zones", &QuantumThoughtPipeline::getZones, py::return_value_policy::reference_internal)
        .def("get_agent", &QuantumThoughtPipeline::getAgent, py::return_value_policy::reference_internal)
        .def("build_field", &QuantumThoughtPipeline::buildField);
}
