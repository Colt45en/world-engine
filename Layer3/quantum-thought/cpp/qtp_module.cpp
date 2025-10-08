// qtp_module.cpp
// Quantum Thought Pipeline — Python bindings (tier-4 upgrade)
//
// Features:
// - Docstrings, __repr__ for ThoughtZone/Agent
// - Safe access to C arrays (position/scale) via tuple and NumPy copy helpers
// - Module metadata (__version__, __doc__)
// - Clean return policies & comments for maintainability

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sstream>
#include <string>

#include "quantum_thought_pipeline.hpp"

namespace py = pybind11;

// Helpers to convert C arrays (float[3]) to Python-friendly objects
static py::tuple to_tuple3(const float (&a)[3]) {
    return py::make_tuple(a[0], a[1], a[2]);
}

static py::array_t<float> to_numpy3(const float (&a)[3]) {
    auto arr = py::array_t<float>(3);
    auto buf = arr.mutable_unchecked<1>();
    for (int i = 0; i < 3; ++i) {
        buf(i) = a[i];
    }
    return arr;
}

// Pretty repr helpers
static std::string zone_repr(const ThoughtZone& z) {
    std::ostringstream os;
    os << "ThoughtZone(name='" << z.name
       << "', type='" << z.type
       << "', position=(" << z.position[0] << ", " << z.position[1] << ", " << z.position[2] << ")"
       << ", scale=(" << z.scale[0] << ", " << z.scale[1] << ", " << z.scale[2] << ")"
       << ", color='" << z.color << "')";
    return os.str();
}

static std::string agent_repr(const Agent& a) {
    std::ostringstream os;
    os << "Agent(id='" << a.id
       << "', position=(" << a.position[0] << ", " << a.position[1] << ", " << a.position[2] << ")"
       << ", maxSteps=" << a.maxSteps
       << ", stepSize=" << a.stepSize
       << ", trailStartColor='" << a.trailStartColor
       << "', trailEndColor='" << a.trailEndColor << "')";
    return os.str();
}

PYBIND11_MODULE(qtp, m) {
    m.doc() = "Quantum Thought Pipeline (C++ core) — Python bindings";
    m.attr("__version__") = "0.2.0";

    // ---- ThoughtZone --------------------------------------------------------
    py::class_<ThoughtZone>(m, "ThoughtZone", R"doc(
A field zone in the Quantum Thought space.

Attributes
----------
name : str
type : str
color : str

Properties
----------
position : tuple[float, float, float]
    Read-only (x, y, z).
position_np : numpy.ndarray
    Copy of position as a 1D float array (length 3).
scale : tuple[float, float, float]
scale_np : numpy.ndarray
)doc")
        .def_property_readonly("name", [](const ThoughtZone& z) { return z.name; })
        .def_property_readonly("type", [](const ThoughtZone& z) { return z.type; })
        .def_property_readonly("color", [](const ThoughtZone& z) { return z.color; })
        .def_property_readonly(
            "position",
            [](const ThoughtZone& z) { return to_tuple3(z.position); },
            "Read-only (x, y, z) as a Python tuple."
        )
        .def_property_readonly(
            "position_np",
            [](const ThoughtZone& z) { return to_numpy3(z.position); },
            "Position as a NumPy array (copy)."
        )
        .def_property_readonly(
            "scale",
            [](const ThoughtZone& z) { return to_tuple3(z.scale); },
            "Read-only scale (sx, sy, sz) as a Python tuple."
        )
        .def_property_readonly(
            "scale_np",
            [](const ThoughtZone& z) { return to_numpy3(z.scale); },
            "Scale as a NumPy array (copy)."
        )
        .def("__repr__", &zone_repr);

    // ---- Agent --------------------------------------------------------------
    py::class_<Agent>(m, "Agent", R"doc(
An agent that traverses the Quantum Thought field.

Attributes
----------
id : str
maxSteps : int
stepSize : float
trailStartColor : str
trailEndColor : str

Properties
----------
position : tuple[float, float, float]
position_np : numpy.ndarray
)doc")
        .def_property_readonly("id", [](const Agent& a) { return a.id; })
        .def_property_readonly("maxSteps", [](const Agent& a) { return a.maxSteps; })
        .def_property_readonly("stepSize", [](const Agent& a) { return a.stepSize; })
        .def_property_readonly("trailStartColor", [](const Agent& a) { return a.trailStartColor; })
        .def_property_readonly("trailEndColor", [](const Agent& a) { return a.trailEndColor; })
        .def_property_readonly(
            "position",
            [](const Agent& a) { return to_tuple3(a.position); },
            "Read-only (x, y, z) as a Python tuple."
        )
        .def_property_readonly(
            "position_np",
            [](const Agent& a) { return to_numpy3(a.position); },
            "Position as a NumPy array (copy)."
        )
        .def("__repr__", &agent_repr);

    // ---- QuantumThoughtPipeline --------------------------------------------
    py::class_<QuantumThoughtPipeline>(m, "QuantumThoughtPipeline", R"doc(
Quantum Thought Pipeline — constructs the field and exposes zones/agent.
)doc")
        .def(py::init<>(), "Create a new pipeline and build the default field.")
        .def("build_field", &QuantumThoughtPipeline::buildField, "Rebuild the field to the default layout.")
        .def(
            "get_zones",
            &QuantumThoughtPipeline::getZones,
            py::return_value_policy::reference_internal,
            "Return a list of ThoughtZone objects (view into internal storage)."
        )
        .def(
            "get_agent",
            &QuantumThoughtPipeline::getAgent,
            py::return_value_policy::reference_internal,
            "Return the Agent (view into internal storage)."
        );
}
