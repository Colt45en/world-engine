from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

try:
    import pybind11  # type: ignore[import-not-found]
except ImportError as exc:
    raise RuntimeError("pybind11 must be installed to build the quantum-thought extension") from exc

cxx_args = ["-std=c++17"]
if sys.platform == "darwin":
    cxx_args += ["-stdlib=libc++"]

ext_modules = [
    Extension(
        "qtp",
        sources=["cpp/bindings.cpp"],
        include_dirs=[pybind11.get_include(), "cpp"],
        language="c++",
        extra_compile_args=cxx_args,
    )
]

setup(
    name="quantum-thought",
    version="0.1.0",
    description="Quantum Thought Pipeline (C++ core) + Python analytics",
    package_dir={"": "python"},
    py_modules=["quantum_analytics"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
