// Python bindings for World Engine using Python C API
#include <Python.h>
#include "world_engine/world_engine.h"

// Module methods
static PyMethodDef WorldEngineMethods[] = {
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef worldenginemodule = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "World Engine C++ core module",
    -1,
    WorldEngineMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__core(void) {
    PyObject *m;
    
    m = PyModule_Create(&worldenginemodule);
    if (m == NULL)
        return NULL;
    
    // Add version
    PyModule_AddStringConstant(m, "__version__", WorldEngine::VERSION);
    
    return m;
}
