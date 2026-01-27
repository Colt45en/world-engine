"""
World Engine Python Bindings Setup
"""
from setuptools import setup, Extension, find_packages
import os
import sys

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(project_root, 'CMakeLists.txt')):
    project_root = os.path.dirname(project_root)

# Define the extension module
world_engine_module = Extension(
    'world_engine._core',
    sources=[
        'src/python/bindings/world_engine_py.cpp',
    ],
    include_dirs=[
        os.path.join(project_root, 'include'),
    ],
    libraries=['world_engine'],
    library_dirs=[
        os.path.join(project_root, 'build', 'lib'),
    ],
    extra_compile_args=['-std=c++17'],
    language='c++',
)

setup(
    name='world_engine',
    version='1.0.0',
    description='World Engine - Multi-language game engine',
    author='World Engine Team',
    author_email='contact@worldengine.dev',
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},
    ext_modules=[world_engine_module],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
