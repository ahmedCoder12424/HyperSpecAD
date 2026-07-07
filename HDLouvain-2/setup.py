from setuptools import setup, Extension
import numpy

module = Extension(
    'louvain_module',
     sources=[
        'louvain_python_2.cpp',  # your wrapper
        'louvain.cpp',           # Louvain main implementation
        'utils.cpp'              # if graph_process or helper functions are here
    ],
    include_dirs=[numpy.get_include(), '.'],  # add your headers here
    language='c++',
     extra_compile_args=['-std=c++17', '-O3']
#   extra_compile_args=['-std=c++17', '-O0', '-g', '-fsanitize=address'],
 #    extra_link_args=['-fsanitize=address', '-ltbb']
)

setup(
    name='louvain_module',
    version='1.0',
    ext_modules=[module]
)

